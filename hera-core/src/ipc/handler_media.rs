//! Handler: generate_image, vision_analysis, generate_video/animate_image.

use crate::ai::{ChatMessage, ChatRequest, ContentPart, MessageContent};
use super::types::{HandlerOutcome, IpcPayload, IpcState};

/// Handle the "generate_image" action — FLUX/sd.cpp image generation with auto-LoRA.
pub async fn handle_generate_image(
    request: &IpcPayload,
    state: &IpcState,
) -> HandlerOutcome {
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
        && let Ok(triggers) = serde_json::from_str::<
            std::collections::HashMap<String, Vec<String>>,
        >(&content)
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
                    if prompt_lower.contains(&kw_lower)
                        && !claimed_keywords.contains(&kw_lower)
                    {
                        prompt.push_str(&format!(" <lora:{}:1.0>", lora_name));
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

    let result_text = if let Some(flux) = &state.flux_engine {
        match flux.generate_image(&prompt, width, height).await {
            Ok(image_bytes) => {
                use base64::{Engine as _, engine::general_purpose};
                let b64 = general_purpose::STANDARD.encode(&image_bytes);
                format!("data:image/png;base64,{}", b64)
            }
            Err(e) => {
                tracing::error!("Flux inference error: {}", e);
                format!("Error: {}", e)
            }
        }
    } else {
        // Fallback to sd.cpp REST API
        let client = reqwest::Client::new();
        let payload = serde_json::json!({
            "prompt": prompt,
            "width": width,
            "height": height,
            "response_format": "b64_json"
        });
        match client
            .post("http://127.0.0.1:8999/v1/images/generations")
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
    };

    HandlerOutcome::Result {
        result_text,
        origin: "unknown".to_string(),
        model: String::new(),
        tool_calls: None,
    }
}

/// Handle the "vision_analysis" action — analyze an image with vision engine.
pub async fn handle_vision_analysis(
    request: &IpcPayload,
    state: &IpcState,
) -> HandlerOutcome {
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
pub async fn handle_generate_video(
    request: &IpcPayload,
    state: &IpcState,
) -> HandlerOutcome {
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
        let flux_payload = serde_json::json!({
            "prompt": enhanced,
            "width": width,
            "height": height,
            "sample_steps": 4,
            "cfg_scale": 1.0,
        });
        match flux_client
            .post("http://127.0.0.1:8999/v1/images/generations")
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

    // Phase 3: GPU Swap — Stop FLUX, generate video
    tracing::info!("🔄 GPU Swap: Stopping FLUX to free VRAM for video generation...");
    let _ = tokio::process::Command::new("pm2")
        .args(["stop", "imagineos-draw"])
        .output()
        .await;
    tokio::time::sleep(std::time::Duration::from_secs(3)).await;

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()
        .unwrap_or_default();

    let mut canvas_payload = serde_json::json!({
        "prompt": enhanced,
        "width": width,
        "height": height,
        "num_frames": num_frames,
    });

    if let Some(ref b64_img) = anchor_image_b64 {
        canvas_payload["image_base64"] = serde_json::Value::String(b64_img.clone());
        tracing::info!("📹 Sending anchor image to VACE I2V pipeline");
    } else {
        tracing::info!("📹 Using T2V pipeline (no anchor image)");
    }

    let result_text = match client
        .post("http://127.0.0.1:8091/v1/video/generate")
        .json(&canvas_payload)
        .send()
        .await
    {
        Ok(resp) => {
            if resp.status().is_success() {
                if let Ok(json) = resp.json::<serde_json::Value>().await {
                    if let Some(path) = json.get("path").and_then(|p| p.as_str()) {
                        path.to_string()
                    } else {
                        "Error: Canvas returned no video path".to_string()
                    }
                } else {
                    "Error: Failed to parse Canvas response".to_string()
                }
            } else {
                format!("Error: Canvas returned status {}", resp.status())
            }
        }
        Err(e) => {
            tracing::error!("Canvas connection error: {}", e);
            format!("Error: Canvas video engine unavailable: {}", e)
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
