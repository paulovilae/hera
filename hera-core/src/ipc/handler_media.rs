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

/// Path to the sd.cpp CLI binary (same binary that backs imagineos-draw's
/// sd-server, just invoked as a one-shot subprocess for video — video gen is
/// slow/rare enough that a resident process isn't worth the VRAM it would
/// hold idle). Override with HERA_SD_CLI_PATH.
fn sd_cli_path() -> String {
    std::env::var("HERA_SD_CLI_PATH")
        .unwrap_or_else(|_| "/home/paulo/sd.cpp/build/bin/sd-cli".to_string())
}

/// Wan2.2 TI2V-5B GGUF weights dir (diffusion model + VAE + UMT5 text encoder).
/// Override with HERA_VIDEO_MODEL_DIR.
fn video_model_dir() -> String {
    std::env::var("HERA_VIDEO_MODEL_DIR")
        .unwrap_or_else(|_| "/home/paulo/models/video-stack/wan2.2".to_string())
}

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

        // 2. Auto-inject remaining LoRAs, avoiding claimed triggers
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

    let result_text = {
    #[cfg(feature = "local-llm")]
    if let Some(flux) = &state.flux_engine {
        match flux.generate_image(&prompt, width, height).await {
            Ok(image_bytes) => {
                use base64::{Engine as _, engine::general_purpose};
                let b64 = general_purpose::STANDARD.encode(&image_bytes);
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
        match client
            .post(&draw_endpoint)
            .json(&payload)
            .send()
            .await
        {
            Ok(resp) => {
                if resp.status().is_success() {
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
                }
            }
            Err(e) => {
                tracing::error!("sd.cpp connection error: {}", e);
                format!("Error connecting to Native Image Generator: {}", e)
            }
        }
    }};

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
        &enhanced[..enhanced.len().min(120)]
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

    // Phase 3: GPU Swap — Stop FLUX, generate video via sd-cli subprocess.
    // (sd.cpp: same binary as imagineos-draw's sd-server, run one-shot with
    // -M vid_gen instead of a resident Python/diffusers server — sidesteps
    // the accelerate cpu-offload device-mismatch bug entirely. --vae-on-cpu
    // is required: the VAE decode step needs ~11GB VRAM alongside the
    // diffusion+t5 weights, which doesn't fit GPU1's shared ~9-12GB headroom
    // next to GLiNER/whisper/vision — confirmed via standalone test 2026-07-11,
    // decode falls back to CPU in ~225s instead of OOMing.)
    tracing::info!("🔄 GPU Swap: Stopping FLUX to free VRAM for video generation...");
    let _ = tokio::process::Command::new("pm2")
        .args(["stop", "imagineos-draw"])
        .output()
        .await;
    tokio::time::sleep(std::time::Duration::from_secs(3)).await;

    let output_dir = "/tmp/imagineos-canvas";
    let _ = std::fs::create_dir_all(output_dir);
    let video_id = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
        .to_string();
    // No trailing extension here — sd-cli always appends ".avi" to -o itself
    // (MJPG AVI is hardcoded), so this base + ".avi" gives the real path.
    let avi_path = format!("{}/video_{}", output_dir, video_id);
    let mp4_path = format!("{}/video_{}.mp4", output_dir, video_id);
    let model_dir = video_model_dir();

    let mut anchor_path: Option<String> = None;
    if let Some(ref b64_img) = anchor_image_b64 {
        match base64::Engine::decode(&base64::engine::general_purpose::STANDARD, b64_img) {
            Ok(bytes) => {
                let path = format!("{}/anchor_{}.png", output_dir, video_id);
                if std::fs::write(&path, bytes).is_ok() {
                    tracing::info!("📹 Anchor image written for I2V: {}", path);
                    anchor_path = Some(path);
                }
            }
            Err(e) => tracing::warn!("Failed to decode anchor image: {}", e),
        }
    }

    let mut cmd = tokio::process::Command::new(sd_cli_path());
    cmd.env("CUDA_VISIBLE_DEVICES", "1")
        .arg("-M")
        .arg("vid_gen")
        .arg("--diffusion-model")
        .arg(format!("{}/Wan2.2-TI2V-5B-Q6_K.gguf", model_dir))
        .arg("--vae")
        .arg(format!("{}/wan2.2_vae.safetensors", model_dir))
        .arg("--t5xxl")
        .arg(format!("{}/umt5-xxl-encoder-Q8_0.gguf", model_dir))
        .arg("-p")
        .arg(&enhanced)
        .arg("--cfg-scale")
        .arg("6.0")
        .arg("--sampling-method")
        .arg("euler")
        .arg("-W")
        .arg(width.to_string())
        .arg("-H")
        .arg(height.to_string())
        .arg("--video-frames")
        .arg(num_frames.to_string())
        .arg("--flow-shift")
        .arg("3.0")
        .arg("--diffusion-fa")
        .arg("--vae-on-cpu")
        .arg("-o")
        .arg(&avi_path);

    if let Some(ref img) = anchor_path {
        tracing::info!("📹 Using I2V pipeline (anchor image)");
        cmd.arg("-i").arg(img);
    } else {
        tracing::info!("📹 Using T2V pipeline (no anchor image)");
    }

    tracing::info!("🎬 Running sd-cli video generation...");
    // sd-cli always writes MJPG AVI and appends ".avi" to -o regardless of
    // the extension given, so the real file is `avi_path` + ".avi".
    let avi_actual_path = format!("{}.avi", avi_path);
    let result_text = match cmd.kill_on_drop(true).output().await {
        Ok(out) => {
            if out.status.success() && std::path::Path::new(&avi_actual_path).exists() {
                tracing::info!("✅ Video generated: {}", avi_actual_path);
                // Transcode MJPG AVI → H.264 MP4 so browsers can play it.
                let transcode = tokio::process::Command::new("ffmpeg")
                    .args([
                        "-y",
                        "-i",
                        &avi_actual_path,
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        &mp4_path,
                    ])
                    .output()
                    .await;
                let _ = std::fs::remove_file(&avi_actual_path);
                match transcode {
                    Ok(t) if t.status.success() && std::path::Path::new(&mp4_path).exists() => {
                        mp4_path.clone()
                    }
                    Ok(t) => format!(
                        "Error: ffmpeg transcode failed: {}",
                        String::from_utf8_lossy(&t.stderr)
                    ),
                    Err(e) => format!("Error: ffmpeg unavailable: {}", e),
                }
            } else {
                format!(
                    "Error: sd-cli video generation failed: {}",
                    String::from_utf8_lossy(&out.stderr)
                )
            }
        }
        Err(e) => {
            tracing::error!("sd-cli subprocess error: {}", e);
            format!("Error: sd-cli video engine unavailable: {}", e)
        }
    };

    // GPU Swap: Restart FLUX after video generation
    tracing::info!("🔄 GPU Swap: Restarting FLUX after video generation...");
    let _ = tokio::process::Command::new("pm2")
        .args(["start", "imagineos-draw"])
        .output()
        .await;

    HandlerOutcome::Result {
        result_text,
        origin: "unknown".to_string(),
        model: String::new(),
        tool_calls: None,
    }
}
