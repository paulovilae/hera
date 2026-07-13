//! Handler: generate_image, vision_analysis, generate_video/animate_image.

use super::types::{HandlerOutcome, IpcPayload, IpcState};
use crate::ai::{ChatMessage, ChatRequest, ContentPart, MessageContent};

/// Returns the sd.cpp image generation base URL.
/// Override with HERA_DRAW_URL to point at a mesh node (e.g. genesis via Tailscale).
fn draw_url() -> String {
    std::env::var("HERA_DRAW_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:8999".to_string())
}

/// Display name of the active image backend, reported to callers so UIs never
/// hardcode the engine. Set HERA_DRAW_MODEL when you swap the draw model.
fn draw_model() -> String {
    std::env::var("HERA_DRAW_MODEL").unwrap_or_else(|_| "Z-Image Turbo".to_string())
}

/// Base URL of the OPTIONAL video generation engine. Video runs ONLY on
/// atlas's opportunistic GPU (same pattern as HERA_MUSIC_URL) — NEVER on
/// genesis. Genesis's GPU1 is dedicated to the LLM/image/vision/STT
/// pipeline; a prior design ran video as a local sd-cli subprocess here that
/// stopped/restarted imagineos-draw to free VRAM (GPU-swap hack, live
/// 2026-07-11..2026-07-13), which hammered imagineos-draw with hundreds of
/// cold restarts and starved GPU1, causing 39-113s image-gen times and
/// Argus VRAM-91% alerts (incident 2026-07-13, see feedback from Paulo — video
/// was never supposed to move onto genesis). Override with HERA_VIDEO_URL.
/// If atlas is offline or the video engine isn't deployed there yet, calls
/// fail cleanly (see handle_generate_video) instead of touching genesis.
fn video_url() -> String {
    std::env::var("HERA_VIDEO_URL")
        .unwrap_or_else(|_| "http://100.106.2.79:8098".to_string())
}

/// Free VRAM (MiB) on the draw GPU, or None if nvidia-smi is unavailable
/// (e.g. non-GPU node — callers should fail OPEN in that case, not block).
fn draw_gpu_free_mib() -> Option<u64> {
    let device = std::env::var("DRAW_ENGINE_CUDA_DEVICE").unwrap_or_else(|_| "1".to_string());
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", &device])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout).trim().parse::<u64>().ok()
}

/// A cold LoRA load merges extra tensors on top of the base model + VAE compute
/// buffer, needing headroom beyond the VAE's own ~3.7GB — confirmed 2026-07-13
/// (project_gpu1_vae_cpu_fallback_queued memory): auto-injecting a LoRA when
/// GPU1 was already tight crashed sd-server with a CUDA alloc OOM repeatedly,
/// even with a retry. Below this threshold, skip auto-LoRA and generate plain
/// rather than crash-loop imagineos-draw.
const MIN_FREE_MIB_FOR_AUTO_LORA: u64 = 6000;

/// Per-LoRA auto-injection weight from lora_weights.json (default 0.7 — 1.0 tends
/// to over-cook and drop quality). Set per LoRA via the Telegram /lora_weight cmd.
fn lora_weight(name: &str) -> f32 {
    std::fs::read_to_string("/home/paulo/models/image-stack/loras/lora_weights.json")
        .ok()
        .and_then(|c| serde_json::from_str::<std::collections::HashMap<String, f32>>(&c).ok())
        .and_then(|m| m.get(name).copied())
        .unwrap_or(0.7)
}

/// Handle the "generate_image" action — FLUX/sd.cpp image generation with auto-LoRA.
#[cfg_attr(not(feature = "local-llm"), allow(unused_variables))]
pub async fn handle_generate_image(request: &IpcPayload, state: &IpcState) -> HandlerOutcome {
    let prompt_val = match request.payload.get("prompt").and_then(|p| p.as_str()) {
        Some(p) => p,
        None => {
            return HandlerOutcome::Result {
                result_text: "Missing prompt".to_string(),
                origin: "unknown".to_string(),
                model: String::new(),
                tool_calls: None,
            };
        }
    };
    let prompt_raw = prompt_val.to_string();
    let mut prompt = prompt_val.to_string();

    // --- Auto-LoRA Router ---
    let triggers_path = "/home/paulo/models/image-stack/loras/triggers.json";
    if let Ok(content) = std::fs::read_to_string(triggers_path)
        && let Ok(triggers) =
            serde_json::from_str::<std::collections::HashMap<String, Vec<String>>>(&content)
    {
        let prompt_lower = prompt.to_lowercase();
        let mut claimed_keywords = std::collections::HashSet::new();

        // 1. Explicit LoRAs claim their auto-triggers first
        for (lora_name, keywords) in &triggers {
            if prompt_lower.contains(&format!("<lora:{}", lora_name.to_lowercase())) {
                for k in keywords {
                    claimed_keywords.insert(k.to_lowercase());
                }
            }
        }

        // 2. Auto-inject remaining LoRAs, avoiding claimed triggers — but only
        // if GPU1 has headroom for the cold-load spike (see MIN_FREE_MIB_FOR_AUTO_LORA).
        let free_mib = draw_gpu_free_mib();
        let has_lora_headroom = free_mib.map(|f| f >= MIN_FREE_MIB_FOR_AUTO_LORA).unwrap_or(true);
        if !has_lora_headroom {
            tracing::warn!(
                "Skipping auto-LoRA injection — GPU1 free VRAM ({:?} MiB) below {}MiB safety margin, generating plain to avoid crash-looping imagineos-draw",
                free_mib, MIN_FREE_MIB_FOR_AUTO_LORA
            );
        }
        if has_lora_headroom {
            for (lora_name, keywords) in triggers {
                if !prompt_lower.contains(&format!("<lora:{}", lora_name.to_lowercase())) {
                    for keyword in keywords {
                        let kw_lower = keyword.to_lowercase();
                        if prompt_lower.contains(&kw_lower) && !claimed_keywords.contains(&kw_lower) {
                            prompt.push_str(&format!(" <lora:{}:{:.2}>", lora_name, lora_weight(&lora_name)));
                            claimed_keywords.insert(kw_lower);
                            break;
                        }
                    }
                }
            }
        }
    }

    let width = request
        .payload
        .get("width")
        .and_then(|w| w.as_u64())
        .unwrap_or(768) as usize;
    let height = request
        .payload
        .get("height")
        .and_then(|h| h.as_u64())
        .unwrap_or(768) as usize;

    // ── Content safety gate + immutable audit (see ipc/media_safety.rs) ─────────
    // Runs BEFORE any backend call, so it covers BOTH the flux and sd.cpp paths.
    // Tier A (illegal) is blocked unconditionally — no permission bypasses it.
    // Tier B (adult NSFW) requires the bot's explicit `nsfw_allowed` permission.
    // Fail-closed: a translation/classifier outage blocks rather than passing an
    // unclassified prompt. `super::helpers::canonicalize_user_id` fills identity
    // if Imaginclaw didn't forward a sender_id.
    let media_ctx = super::media_safety::MediaRequestContext::from_payload(
        &request.payload,
        "image",
        &prompt_raw,
        &prompt,
        &draw_model(),
    );
    let (gate_decision, gate_details) =
        super::media_safety::evaluate_gate_with_engine(&state.engine, &media_ctx).await;
    if gate_decision.is_blocked() {
        super::media_safety::record_media_generation(
            &media_ctx,
            &gate_decision,
            gate_details.as_ref(),
            None,
            "png",
        );
        tracing::warn!(
            "🛡️ /draw blocked ({}) requester={} bot={} channel={}",
            gate_decision.audit_label(),
            media_ctx.requester_id,
            media_ctx.bot_name,
            media_ctx.channel
        );
        return HandlerOutcome::Result {
            result_text: gate_decision.user_message(),
            origin: "media_gate".to_string(),
            model: draw_model(),
            tool_calls: None,
        };
    }

    let result_text = {
    #[cfg(feature = "local-llm")]
    if let Some(flux) = &state.flux_engine {
        match flux.generate_image(&prompt, width, height).await {
            Ok(image_bytes) => {
                use base64::{Engine as _, engine::general_purpose};
                let b64 = general_purpose::STANDARD.encode(&image_bytes);
                super::media_safety::record_media_generation(
                    &media_ctx,
                    &gate_decision,
                    gate_details.as_ref(),
                    Some(&image_bytes),
                    "png",
                );
                return HandlerOutcome::Result {
                    result_text: format!("data:image/png;base64,{}", b64),
                    origin: "unknown".to_string(),
                    model: String::new(),
                    tool_calls: None,
                };
            }
            Err(e) => {
                tracing::error!("Flux inference error: {}", e);
            }
        }
    }
    {
        // Fallback to sd.cpp REST API (endpoint configurable via HERA_DRAW_URL)
        let draw_endpoint = format!("{}/v1/images/generations", draw_url());
        let client = reqwest::Client::new();
        // sd.cpp (OpenAI-compat) lee "size" ("WxH"), NO width/height sueltos: sin "size" usa
        // 512x512 por defecto e ignora las dimensiones pedidas (banners/wide salían cuadrados).
        let mut payload = serde_json::json!({
            "prompt": prompt,
            "width": width,
            "height": height,
            "size": format!("{width}x{height}"),
            "response_format": "b64_json"
        });
        // Optional seed for reproducibility (caller passes it; sd.cpp honors "seed").
        if let Some(seed) = request.payload.get("seed").and_then(|s| s.as_i64()) {
            payload["seed"] = serde_json::json!(seed);
        }
        // imagineos-draw crash-restarts occasionally (GPU1 VRAM is tight — see
        // project_gpu1_vae_cpu_fallback_queued memory): a crashed sd-server
        // needs ~6-10s to reload the GGUF model before it accepts connections
        // again. One retry after a short wait turns a transient crash-restart
        // window into a slower request instead of a hard user-facing error.
        let mut attempt = 0;
        loop {
            match client.post(&draw_endpoint).json(&payload).send().await {
                Ok(resp) => {
                    break if resp.status().is_success() {
                        if let Ok(json) = resp.json::<serde_json::Value>().await {
                            if let Some(b64) = json["data"][0]["b64_json"].as_str() {
                                format!("data:image/png;base64,{}", b64)
                            } else {
                                "Error: Invalid response format from sd.cpp".to_string()
                            }
                        } else {
                            "Error: Failed to parse sd.cpp JSON response".to_string()
                        }
                    } else {
                        format!("Error: sd.cpp returned status {}", resp.status())
                    };
                }
                Err(e) if attempt == 0 => {
                    attempt += 1;
                    tracing::warn!(
                        "sd.cpp connection error (attempt {attempt}), retrying in 8s in case it's mid-restart: {}",
                        e
                    );
                    tokio::time::sleep(std::time::Duration::from_secs(8)).await;
                }
                Err(e) => {
                    tracing::error!("sd.cpp connection error after retry: {}", e);
                    break format!("Error connecting to Native Image Generator: {}", e);
                }
            }
        }
    }};

    // Audit the allowed generation with the real output bytes (decoded from the
    // data URL). A failed backend call still records the attempt (no bytes).
    let output_bytes = result_text
        .strip_prefix("data:image/png;base64,")
        .and_then(|b64| {
            use base64::{Engine as _, engine::general_purpose};
            general_purpose::STANDARD.decode(b64).ok()
        });
    super::media_safety::record_media_generation(
        &media_ctx,
        &gate_decision,
        gate_details.as_ref(),
        output_bytes.as_deref(),
        "png",
    );

    HandlerOutcome::Result {
        result_text,
        origin: "sd.cpp".to_string(),
        model: draw_model(),
        tool_calls: None,
    }
}

/// Handle the "vision_analysis" action — analyze an image with vision engine.
pub async fn handle_vision_analysis(request: &IpcPayload, state: &IpcState) -> HandlerOutcome {
    let (b64, prompt) = match (
        request.payload.get("base64_image").and_then(|p| p.as_str()),
        request.payload.get("prompt").and_then(|p| p.as_str()),
    ) {
        (Some(b), Some(p)) => (b, p),
        _ => {
            return HandlerOutcome::Result {
                result_text: "Missing base64_image or prompt".to_string(),
                origin: "unknown".to_string(),
                model: String::new(),
                tool_calls: None,
            };
        }
    };

    let result_text = if let Some(vision) = &state.vision_engine {
        let chat_req = ChatRequest {
            model: "hera-vision-model".to_string(),
            vision_model: None,
            tts_model: None,
            stt_model: None,
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: MessageContent::Parts(vec![
                    ContentPart::ImageUrl {
                        image_url: crate::ai::ImageUrlContent {
                            url: format!("data:image/jpeg;base64,{}", b64),
                        },
                    },
                    ContentPart::Text {
                        text: prompt.to_string(),
                    },
                ]),
            }],
            temperature: None,
            max_tokens: Some(4096),
            top_p: None,
            top_k: None,
            presence_penalty: None,
            frequency_penalty: None,
            repeat_penalty: None,
            seed: None,
            stop: None,
            endpoint: None,
            api_key: None,
            provider: None,
            stream: None,
            nsfw: None,
            tools: None,
            tool_choice: None,
            reasoning_effort: None,
            response_format: None,
            app: None,
            priority: None,
        };

        match vision.generate_content(chat_req).await {
            Ok(resp) => {
                if let Some(choice) = resp.choices.first()
                    && let Some(content) = &choice.message.content
                {
                    content.clone()
                } else {
                    "Error: Empty vision response".to_string()
                }
            }
            Err(e) => {
                tracing::error!("Vision inference error: {}", e);
                format!("Error: {}", e)
            }
        }
    } else {
        "Hera Vision Engine (Moondream) is not loaded or unavailable.".to_string()
    };

    HandlerOutcome::Result {
        result_text,
        origin: "unknown".to_string(),
        model: String::new(),
        tool_calls: None,
    }
}

/// Handle the "generate_video" / "animate_image" action — video generation pipeline.
pub async fn handle_generate_video(request: &IpcPayload, state: &IpcState) -> HandlerOutcome {
    let prompt = match request.payload.get("prompt").and_then(|p| p.as_str()) {
        Some(p) => p,
        None => {
            return HandlerOutcome::Result {
                result_text: "Missing prompt".to_string(),
                origin: "unknown".to_string(),
                model: String::new(),
                tool_calls: None,
            };
        }
    };

    // ── Content safety gate + audit (video) ────────────────────────────────────
    // Same two-tier gate as /draw, on the user's raw prompt (before enhancement),
    // so illegal video requests are blocked before any GPU work. Fail-closed.
    let video_ctx = super::media_safety::MediaRequestContext::from_payload(
        &request.payload,
        "video",
        prompt,
        prompt,
        "video-engine",
    );
    let (video_gate, video_gate_details) =
        super::media_safety::evaluate_gate_with_engine(&state.engine, &video_ctx).await;
    if video_gate.is_blocked() {
        super::media_safety::record_media_generation(
            &video_ctx,
            &video_gate,
            video_gate_details.as_ref(),
            None,
            "mp4",
        );
        tracing::warn!(
            "🛡️ video blocked ({}) requester={} bot={}",
            video_gate.audit_label(),
            video_ctx.requester_id,
            video_ctx.bot_name
        );
        return HandlerOutcome::Result {
            result_text: video_gate.user_message(),
            origin: "media_gate".to_string(),
            model: String::new(),
            tool_calls: None,
        };
    }

    // Phase 1: Brain — Enhance the prompt
    let enhance_prompt = format!(
        "You are a video director AI. Given this brief idea, write a single detailed paragraph describing the exact visual scene for a text-to-video model. Include camera angle, lighting, motion, colors, and atmosphere. Only output the scene description, nothing else.\n\nIdea: {}",
        prompt
    );
    let chat_req = ChatRequest {
        model: "hera-local-model".to_string(),
        vision_model: None,
        tts_model: None,
        stt_model: None,
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: MessageContent::Text(enhance_prompt),
        }],
        temperature: Some(0.8),
        max_tokens: Some(300),
        top_p: None,
        top_k: None,
        presence_penalty: None,
        frequency_penalty: None,
        repeat_penalty: None,
        seed: None,
        stop: None,
        endpoint: None,
        api_key: None,
        provider: None,
        stream: None,
        nsfw: None,
        tools: None,
        tool_choice: None,
        reasoning_effort: None,
        response_format: None,
        app: None,
        priority: None,
    };

    let enhanced = match state.engine.generate_content(chat_req).await {
        Ok(resp) => resp
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .unwrap_or_else(|| prompt.to_string()),
        Err(e) => {
            tracing::error!("Brain prompt enhancement failed: {}, using raw prompt", e);
            prompt.to_string()
        }
    };
    tracing::info!(
        "🧠 Enhanced prompt: {}",
        enhanced.chars().take(120).collect::<String>()
    );

    // Phase 2: Generate FLUX anchor frame (if no user image)
    let width = request
        .payload
        .get("width")
        .and_then(|w| w.as_u64())
        .unwrap_or(480);
    let height = request
        .payload
        .get("height")
        .and_then(|h| h.as_u64())
        .unwrap_or(320);
    let num_frames = request
        .payload
        .get("num_frames")
        .and_then(|n| n.as_u64())
        .unwrap_or(81);

    let user_image_b64 = request
        .payload
        .get("base64_image")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let anchor_image_b64: Option<String> = if let Some(img) = user_image_b64 {
        tracing::info!("🖼️ Using user-provided image as anchor frame");
        Some(img)
    } else {
        tracing::info!("🎨 Generating FLUX anchor frame...");
        let flux_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .unwrap_or_default();
        let flux_draw_endpoint = format!("{}/v1/images/generations", draw_url());
        let flux_payload = serde_json::json!({
            "prompt": enhanced,
            "width": width,
            "height": height,
            "sample_steps": 4,
            "cfg_scale": 1.0,
        });
        match flux_client
            .post(&flux_draw_endpoint)
            .json(&flux_payload)
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => {
                if let Ok(json) = resp.json::<serde_json::Value>().await {
                    let b64 = json
                        .get("data")
                        .and_then(|d| d.as_array())
                        .and_then(|arr| arr.first())
                        .and_then(|item| item.get("b64_json"))
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    if b64.is_some() {
                        tracing::info!("✅ FLUX anchor frame generated!");
                    }
                    b64
                } else {
                    None
                }
            }
            _ => {
                tracing::info!("⚠️ FLUX anchor frame failed, falling back to T2V");
                None
            }
        }
    };

    // Phase 3: Generate the video — ALWAYS via atlas's opportunistic GPU,
    // NEVER on genesis (see video_url() doc comment for why). genesis's own
    // GPU/pm2 processes are never touched here.
    let output_dir = "/tmp/imagineos-canvas";
    let _ = std::fs::create_dir_all(output_dir);

    tracing::info!("🎬 Requesting video from atlas video engine...");
    let video_client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .unwrap_or_default();
    let video_payload = serde_json::json!({
        "prompt": enhanced,
        "image_base64": anchor_image_b64,
        "width": width,
        "height": height,
        "num_frames": num_frames,
    });

    let result_text = match video_client
        .post(format!("{}/generate", video_url()))
        .json(&video_payload)
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => match resp.json::<serde_json::Value>().await {
            Ok(json) => match json.get("video_base64").and_then(|v| v.as_str()) {
                Some(b64) => match base64::Engine::decode(&base64::engine::general_purpose::STANDARD, b64) {
                    Ok(bytes) => {
                        let video_id = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_nanos())
                            .unwrap_or(0);
                        let mp4_path = format!("{}/video_{}.mp4", output_dir, video_id);
                        match std::fs::write(&mp4_path, bytes) {
                            Ok(_) => {
                                tracing::info!("✅ Video generated: {}", mp4_path);
                                mp4_path
                            }
                            Err(e) => format!("Error: failed to save video: {}", e),
                        }
                    }
                    Err(e) => format!("Error: atlas video engine returned invalid video data: {}", e),
                },
                None => "Error: atlas video engine returned no video data".to_string(),
            },
            Err(e) => format!("Error: atlas video engine returned an unreadable response: {}", e),
        },
        Ok(resp) => format!(
            "Error: video generation unavailable (atlas video engine returned HTTP {})",
            resp.status()
        ),
        Err(e) => {
            tracing::warn!("Video engine (atlas) unreachable: {}", e);
            "Error: video generation unavailable — atlas's video engine is offline or not yet provisioned. Video is opportunistic capacity, not required for the rest of the platform.".to_string()
        }
    };

    // Audit the allowed video generation. Copy the mp4 into the audit store when
    // the result is a real file path under a sane size cap (video files are large;
    // above the cap we record the row without the blob).
    const VIDEO_AUDIT_MAX_BYTES: u64 = 60 * 1024 * 1024;
    let video_bytes = if (result_text.starts_with("/tmp/") || result_text.starts_with("/home/"))
        && !result_text.starts_with("Error")
    {
        std::fs::metadata(&result_text)
            .ok()
            .filter(|m| m.len() <= VIDEO_AUDIT_MAX_BYTES)
            .and_then(|_| std::fs::read(&result_text).ok())
    } else {
        None
    };
    super::media_safety::record_media_generation(
        &video_ctx,
        &video_gate,
        video_gate_details.as_ref(),
        video_bytes.as_deref(),
        "mp4",
    );

    HandlerOutcome::Result {
        result_text,
        origin: "unknown".to_string(),
        model: String::new(),
        tool_calls: None,
    }
}
