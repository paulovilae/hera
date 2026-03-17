use std::sync::Arc;
use tokio::net::UnixListener;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use serde::{Deserialize, Serialize};
use tracing::{info, error};

use crate::ai::{LLMEngine, ChatRequest, ChatMessage, MessageContent, ContentPart};
use crate::ai::engine_parler::ParlerEngine;
use crate::ai::engine_whisper::WhisperEngine;

#[derive(Clone)]
pub struct IpcState {
    pub engine: Arc<dyn LLMEngine + Send + Sync>,
    pub local_engine: Arc<dyn LLMEngine + Send + Sync>,
    pub flux_engine: Option<Arc<crate::ai::engine_flux::FluxEngine>>,
    pub parler_engine: Option<Arc<ParlerEngine>>,
    pub whisper_engine: Option<Arc<WhisperEngine>>,
    pub vision_engine: Option<Arc<dyn LLMEngine + Send + Sync>>,
    pub micro_engine: Option<Arc<dyn LLMEngine + Send + Sync>>,
}

#[derive(Deserialize, Debug)]
pub struct IpcPayload {
    pub action: String,
    pub payload: serde_json::Value,
}

#[derive(Serialize, Debug)]
pub struct IpcResponse {
    pub status: String,
    pub data: serde_json::Value,
}

pub async fn serve(socket_path: &str, state: IpcState) -> std::io::Result<()> {
    // Ensure the socket file is removed before binding if it already exists from a previous bad crash
    if std::path::Path::new(socket_path).exists() {
        std::fs::remove_file(socket_path)?;
    }

    let listener = UnixListener::bind(socket_path)?;
    info!("🔗 Headless IPC Daemon bound to Unix socket: {}", socket_path);

    loop {
        match listener.accept().await {
            Ok((mut stream, _addr)) => {
                let state = state.clone();
                tokio::spawn(async move {
                    let mut buffer = Vec::new();
                    let mut chunk = vec![0; 8192];
                    loop {
                        match stream.read(&mut chunk).await {
                            Ok(n) if n > 0 => {
                                buffer.extend_from_slice(&chunk[..n]);
                                if let Ok(request) = serde_json::from_slice::<IpcPayload>(&buffer) {
                                    info!("📥 Received IPC Action: {}", request.action);
                                
                                // Process Request
                                let mut result_text = "Action not supported".to_string();
                                let mut tool_calls: Option<serde_json::Value> = None;
                                
                                if request.action == "generate" {
                                    let mut payload_clone = request.payload.clone();
                                    
                                    // Extract prompt
                                    let mut prompt = payload_clone.get("prompt").and_then(|p| p.as_str()).unwrap_or("").to_string();
                                    let mut assistant_last: Option<String> = None;
                                    
                                    // Make sure we extract the prompt from the messages array if it wasn't provided directly
                                    if prompt.is_empty() {
                                        if let Some(messages) = payload_clone.get("messages").and_then(|m| m.as_array()) {
                                            if let Some(last_msg) = messages.last() {
                                                if let Some("user") = last_msg.get("role").and_then(|r| r.as_str()) {
                                                    if let Some(content) = last_msg.get("content").and_then(|c| c.as_str()) {
                                                        prompt = content.to_string();
                                                    }
                                                }
                                            }
                                            
                                            // Extract the second to last message if it's from the assistant
                                            if messages.len() >= 2 {
                                                if let Some(prev_msg) = messages.get(messages.len() - 2) {
                                                    if let Some("assistant") = prev_msg.get("role").and_then(|r| r.as_str()) {
                                                        if let Some(content) = prev_msg.get("content").and_then(|c| c.as_str()) {
                                                            assistant_last = Some(content.to_string());
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    let mut handled_by_tool = false;
                                    
                                    let permissions: Vec<String> = payload_clone.get("permissions")
                                        .and_then(|v| v.as_array())
                                        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect::<Vec<String>>())
                                        .unwrap_or_else(|| vec!["all".to_string()]);
                                        
                                    tracing::info!("🛡️ [Hera IPC] Parsed permissions: {:?}", permissions);
                                    
                                    // 1. Fast-path intent detection
                                    if !prompt.is_empty() {
                                        if let Some(tool_call) = crate::ai::tool_executor::detect_intent_from_user_message(&prompt, assistant_last.as_deref()) {
                                            if permissions.contains(&"all".to_string()) || permissions.contains(&tool_call.name) {
                                                tracing::info!("🚀 [Hera IPC] Fast-path tool intent detected: {}", tool_call.name);
                                                let tool_result = crate::ai::tool_executor::execute_tool(&tool_call).await;
                                                result_text = tool_result.output;
                                                tool_calls = Some(serde_json::json!([tool_call]));
                                                handled_by_tool = true;
                                            } else {
                                                tracing::info!("⚠️ [Hera IPC] Fast-path tool intent {} denied by permissions", tool_call.name);
                                            }
                                        }
                                    }
                                    
                                    // 2. Normal LLM generation
                                    if !handled_by_tool {
                                        if let Some(obj) = payload_clone.as_object_mut() {
                                            if !obj.contains_key("model") {
                                                obj.insert("model".to_string(), serde_json::json!("hera-local-model"));
                                            }
                                        }
                                        
                                        let prompt = payload_clone.get("prompt").and_then(|v| v.as_str()).unwrap_or("").to_string();
                                        let persona_path = payload_clone.get("persona_path").and_then(|v| v.as_str()).unwrap_or("/home/paulo/Programs/apps/imaginos/imaginclaw/persona/SOUL.md").to_string();
                                            
                                        let mut chat_req: Option<ChatRequest> = serde_json::from_value(payload_clone).ok();
                                        
                                        if chat_req.is_none() {
                                            if !prompt.is_empty() {
                                                let base_system_prompt = std::fs::read_to_string(&persona_path)
                                                    .unwrap_or_else(|_| "You are an AI assistant.".to_string());
                                                let schemas = crate::ai::tool_executor::hera_tool_schemas(&permissions);
                                                let full_system_prompt = format!("{}\n\n{}", base_system_prompt, schemas);

                                                chat_req = Some(ChatRequest {
                                                    model: "hera-local-model".to_string(),
                                                    vision_model: None,
                                                    tts_model: None,
                                                    stt_model: None,
                                                    messages: vec![
                                                        ChatMessage {
                                                            role: "system".to_string(),
                                                            content: MessageContent::Text(full_system_prompt),
                                                        },
                                                        ChatMessage {
                                                            role: "user".to_string(),
                                                            content: MessageContent::Text(prompt.clone()),
                                                        }
                                                    ],
                                                    temperature: Some(0.7),
                                                    max_tokens: Some(1024),
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
                                                });
                                            }
                                        } else if let Some(req) = &mut chat_req {
                                            // Inject base persona + tool schemas into existing request
                                            let base_system_prompt = std::fs::read_to_string(&persona_path)
                                                .unwrap_or_else(|_| "You are an AI assistant.".to_string());
                                            let schemas = crate::ai::tool_executor::hera_tool_schemas(&permissions);
                                            let full_system_prompt = format!("{}\n\n{}", base_system_prompt, schemas);

                                            if let Some(first) = req.messages.first_mut() {
                                                if first.role == "system" {
                                                    match &mut first.content {
                                                        MessageContent::Text(t) => {
                                                            *t = format!("{}\n\n{}", full_system_prompt, t);
                                                        }
                                                        MessageContent::Parts(parts) => {
                                                            parts.insert(0, ContentPart::Text { text: format!("{}\n\n", full_system_prompt) });
                                                        }
                                                        MessageContent::Null => {
                                                            first.content = MessageContent::Text(full_system_prompt);
                                                        }
                                                    }
                                                } else {
                                                    req.messages.insert(0, ChatMessage {
                                                        role: "system".to_string(),
                                                        content: MessageContent::Text(full_system_prompt),
                                                    });
                                                }
                                            } else {
                                                req.messages.push(ChatMessage {
                                                    role: "system".to_string(),
                                                    content: MessageContent::Text(full_system_prompt),
                                                });
                                            }
                                        }
                                        
                                        if let Some(req) = chat_req.clone() {
                                            match state.engine.generate_content(req).await {
                                                Ok(resp) => {
                                                    if let Some(choice) = resp.choices.first() {
                                                        if let Some(content) = &choice.message.content {
                                                            result_text = content.clone();
                                                            
                                                            // 3. Parse and Execute Output Tool Calls
                                                            let parsed_calls = crate::ai::tool_executor::parse_tool_calls(&result_text);
                                                            if !parsed_calls.is_empty() {
                                                                tracing::info!("🛠️ [Hera IPC] LLM emitted {} tool calls", parsed_calls.len());
                                                                let mut execution_outputs = String::new();
                                                                let mut executed_calls = Vec::new();
                                                                
                                                                for call in &parsed_calls {
                                                                    if permissions.contains(&"all".to_string()) || permissions.contains(&call.name) {
                                                                        let tool_res = crate::ai::tool_executor::execute_tool(&call).await;
                                                                        execution_outputs.push_str(&format!("\n\n{}", tool_res.output));
                                                                        
                                                                        executed_calls.push(serde_json::json!({
                                                                            "name": call.name,
                                                                            "arguments": call.arguments
                                                                        }));
                                                                    } else {
                                                                        tracing::warn!("⚠️ [Hera IPC] LLM hallucinated tool {} which is denied by permissions", call.name);
                                                                        execution_outputs.push_str(&format!("\n\nError: Not permitted to use tool '{}'", call.name));
                                                                    }
                                                                }
                                                                
                                                                let has_media_call = parsed_calls.iter().any(|c| c.name == "hera_draw" || c.name == "hera_video" || c.name == "generate_qr_code");

                                                                if !has_media_call {
                                                                    if let Some(mut req2) = chat_req.clone() {
                                                                        req2.messages.push(ChatMessage {
                                                                            role: "assistant".to_string(),
                                                                            content: MessageContent::Text(result_text.clone()),
                                                                        });
                                                                        req2.messages.push(ChatMessage {
                                                                            role: "user".to_string(),
                                                                            content: MessageContent::Text(format!("Tool Execution Results: {}\n\nPlease provide a friendly, conversational, and concise response to the user based on these results. Do not output raw JSON or mention the database tables directly. Avoid outputting any tool call tags.", execution_outputs)),
                                                                        });
                                                                        tracing::info!("🔄 [Hera IPC] Initiating second-pass generation to format Tool Results...");
                                                                        match state.engine.generate_content(req2).await {
                                                                            Ok(resp2) => {
                                                                                if let Some(ch) = resp2.choices.first() {
                                                                                    if let Some(c) = &ch.message.content {
                                                                                        result_text = c.clone();
                                                                                    }
                                                                                }
                                                                            }
                                                                            Err(e) => {
                                                                                tracing::error!("Second pass inference failed: {}", e);
                                                                                result_text.push_str(&format!("\n\n[Error forming final response: {}]\n{}", e, execution_outputs));
                                                                            }
                                                                        }
                                                                    }
                                                                } else {
                                                                    // Append execution output directly for media calls
                                                                    result_text.push_str(&execution_outputs);
                                                                }
                                                                
                                                                tool_calls = Some(serde_json::Value::Array(executed_calls));
                                                            }
                                                        }
                                                        if let Some(tc) = &choice.message.tool_calls {
                                                            // Also preserve native tool calls if the model emits them via choices.message.tool_calls
                                                            if tool_calls.is_none() {
                                                                tool_calls = Some(serde_json::json!(tc));
                                                            }
                                                        }
                                                    }
                                                }
                                                Err(e) => {
                                                    error!("LLM inference error: {}", e);
                                                    result_text = format!("Error: {}", e);
                                                }
                                            }
                                        }
                                    }
                                } else if request.action == "generate_image" {
                                    if let Some(prompt) = request.payload.get("prompt").and_then(|p| p.as_str()) {
                                        let width = request.payload.get("width").and_then(|w| w.as_u64()).unwrap_or(768) as usize;
                                        let height = request.payload.get("height").and_then(|h| h.as_u64()).unwrap_or(768) as usize;
                                        
                                        if let Some(flux) = &state.flux_engine {
                                            match flux.generate_image(prompt, width, height).await {
                                                Ok(image_bytes) => {
                                                    use base64::{Engine as _, engine::general_purpose};
                                                    let b64 = general_purpose::STANDARD.encode(&image_bytes);
                                                    result_text = format!("data:image/png;base64,{}", b64);
                                                }
                                                Err(e) => {
                                                    error!("Flux inference error: {}", e);
                                                    result_text = format!("Error: {}", e);
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
                                            let response = client
                                                .post("http://127.0.0.1:8999/v1/images/generations")
                                                .json(&payload)
                                                .send()
                                                .await;
                                                
                                            match response {
                                                Ok(resp) => {
                                                    if resp.status().is_success() {
                                                        if let Ok(json) = resp.json::<serde_json::Value>().await {
                                                            if let Some(b64) = json["data"][0]["b64_json"].as_str() {
                                                                result_text = format!("data:image/png;base64,{}", b64);
                                                            } else {
                                                                result_text = "Error: Invalid response format from sd.cpp".to_string();
                                                            }
                                                        } else {
                                                            result_text = "Error: Failed to parse sd.cpp JSON response".to_string();
                                                        }
                                                    } else {
                                                        result_text = format!("Error: sd.cpp returned status {}", resp.status());
                                                    }
                                                }
                                                Err(e) => {
                                                    error!("sd.cpp connection error: {}", e);
                                                    result_text = format!("Error connecting to Native Image Generator: {}", e);
                                                }
                                            }
                                        }
                                    }
                                } else if request.action == "vision_analysis" {
                                    if let (Some(b64), Some(prompt)) = (request.payload.get("base64_image").and_then(|p| p.as_str()), request.payload.get("prompt").and_then(|p| p.as_str())) {
                                        if let Some(vision) = &state.vision_engine {
                                            let chat_req = ChatRequest {
                                                model: "hera-vision-model".to_string(),
                                                vision_model: None, tts_model: None, stt_model: None,
                                                messages: vec![ChatMessage {
                                                    role: "user".to_string(),
                                                    content: MessageContent::Parts(vec![
                                                        ContentPart::ImageUrl {
                                                            image_url: crate::ai::ImageUrlContent {
                                                                url: format!("data:image/jpeg;base64,{}", b64)
                                                            }
                                                        },
                                                        ContentPart::Text {
                                                            text: prompt.to_string()
                                                        }
                                                    ]),
                                                }],
                                                temperature: None,
                                                max_tokens: Some(1024),
                                                top_p: None, top_k: None, presence_penalty: None, frequency_penalty: None, repeat_penalty: None, seed: None, stop: None, endpoint: None, api_key: None, provider: None, stream: None, nsfw: None, tools: None, tool_choice: None, reasoning_effort: None,
                                            };
                                            
                                            match vision.generate_content(chat_req).await {
                                                Ok(resp) => {
                                                    if let Some(choice) = resp.choices.first() {
                                                        if let Some(content) = &choice.message.content {
                                                            result_text = content.clone();
                                                        }
                                                    }
                                                }
                                                Err(e) => {
                                                    error!("Vision inference error: {}", e);
                                                    result_text = format!("Error: {}", e);
                                                }
                                            }
                                        } else {
                                            result_text = "Hera Vision Engine (Moondream) is not loaded or unavailable.".to_string();
                                        }
                                    }
                                } else if request.action == "generate_video" || request.action == "animate_image" {
                                    if let Some(prompt) = request.payload.get("prompt").and_then(|p| p.as_str()) {
                                        // ── Phase 1: Brain (Qwen3-VL) — Enhance the prompt ──
                                        let enhance_prompt = format!(
                                            "You are a video director AI. Given this brief idea, write a single detailed paragraph describing the exact visual scene for a text-to-video model. Include camera angle, lighting, motion, colors, and atmosphere. Only output the scene description, nothing else.\n\nIdea: {}",
                                            prompt
                                        );
                                        let chat_req = ChatRequest {
                                            model: "hera-local-model".to_string(),
                                            vision_model: None, tts_model: None, stt_model: None,
                                            messages: vec![ChatMessage {
                                                role: "user".to_string(),
                                                content: MessageContent::Text(enhance_prompt),
                                            }],
                                            temperature: Some(0.8), max_tokens: Some(300),
                                            top_p: None, top_k: None, presence_penalty: None,
                                            frequency_penalty: None, repeat_penalty: None,
                                            seed: None, stop: None, endpoint: None,
                                            api_key: None, provider: None, stream: None,
                                            nsfw: None, tools: None, tool_choice: None,
                                            reasoning_effort: None,
                                        };

                                        let enhanced = match state.engine.generate_content(chat_req).await {
                                            Ok(resp) => {
                                                resp.choices.first()
                                                    .and_then(|c| c.message.content.clone())
                                                    .unwrap_or_else(|| prompt.to_string())
                                            }
                                            Err(e) => {
                                                error!("Brain prompt enhancement failed: {}, using raw prompt", e);
                                                prompt.to_string()
                                            }
                                        };
                                        info!("🧠 Enhanced prompt: {}", &enhanced[..enhanced.len().min(120)]);

                                        // ── Phase 2: Generate FLUX anchor frame (if no user image) ──
                                        let width = request.payload.get("width").and_then(|w| w.as_u64()).unwrap_or(480);
                                        let height = request.payload.get("height").and_then(|h| h.as_u64()).unwrap_or(320);
                                        let num_frames = request.payload.get("num_frames").and_then(|n| n.as_u64()).unwrap_or(81);

                                        // Check if user provided an image already
                                        let user_image_b64 = request.payload.get("base64_image")
                                            .and_then(|v| v.as_str())
                                            .map(|s| s.to_string());

                                        let anchor_image_b64: Option<String> = if let Some(img) = user_image_b64 {
                                            info!("🖼️ Using user-provided image as anchor frame");
                                            Some(img)
                                        } else {
                                            // Generate anchor frame via FLUX (sd.cpp on port 8999)
                                            info!("🎨 Generating FLUX anchor frame...");
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
                                            match flux_client.post("http://127.0.0.1:8999/v1/images/generations")
                                                .json(&flux_payload)
                                                .send()
                                                .await
                                            {
                                                Ok(resp) if resp.status().is_success() => {
                                                    if let Ok(json) = resp.json::<serde_json::Value>().await {
                                                        // sd.cpp returns base64 image in data[0].b64_json
                                                        let b64 = json.get("data")
                                                            .and_then(|d| d.as_array())
                                                            .and_then(|arr| arr.first())
                                                            .and_then(|item| item.get("b64_json"))
                                                            .and_then(|v| v.as_str())
                                                            .map(|s| s.to_string());
                                                        if b64.is_some() {
                                                            info!("✅ FLUX anchor frame generated!");
                                                        }
                                                        b64
                                                    } else { None }
                                                }
                                                _ => {
                                                    info!("⚠️ FLUX anchor frame failed, falling back to T2V");
                                                    None
                                                }
                                            }
                                        };

                                        // ── Phase 3: GPU Swap — Stop FLUX, generate video ──
                                        info!("🔄 GPU Swap: Stopping FLUX to free VRAM for video generation...");
                                        let _ = tokio::process::Command::new("pm2")
                                            .args(&["stop", "imagineos-draw"])
                                            .output().await;
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

                                        // If we have an anchor image, add it for I2V
                                        if let Some(ref b64_img) = anchor_image_b64 {
                                            canvas_payload["image_base64"] = serde_json::Value::String(b64_img.clone());
                                            info!("📹 Sending anchor image to VACE I2V pipeline");
                                        } else {
                                            info!("📹 Using T2V pipeline (no anchor image)");
                                        }

                                        match client.post("http://127.0.0.1:8091/v1/video/generate")
                                            .json(&canvas_payload)
                                            .send()
                                            .await
                                        {
                                            Ok(resp) => {
                                                if resp.status().is_success() {
                                                    if let Ok(json) = resp.json::<serde_json::Value>().await {
                                                        if let Some(path) = json.get("path").and_then(|p| p.as_str()) {
                                                            result_text = path.to_string();
                                                        } else {
                                                            result_text = "Error: Canvas returned no video path".to_string();
                                                        }
                                                    } else {
                                                        result_text = "Error: Failed to parse Canvas response".to_string();
                                                    }
                                                } else {
                                                    result_text = format!("Error: Canvas returned status {}", resp.status());
                                                }
                                            }
                                            Err(e) => {
                                                error!("Canvas connection error: {}", e);
                                                result_text = format!("Error: Canvas video engine unavailable: {}", e);
                                            }
                                        }

                                        // GPU Swap: Restart FLUX after video generation
                                        info!("🔄 GPU Swap: Restarting FLUX after video generation...");
                                        let _ = tokio::process::Command::new("pm2")
                                            .args(&["start", "imagineos-draw"])
                                            .output().await;
                                    }
                                } else if request.action == "transcribe_audio" {
                                    if let Some(b64) = request.payload.get("base64_audio").and_then(|p| p.as_str()) {
                                        if let Some(whisper) = &state.whisper_engine {
                                            use base64::{Engine as _, engine::general_purpose};
                                            if let Ok(audio_bytes) = general_purpose::STANDARD.decode(b64) {
                                                match whisper.transcribe_audio(&audio_bytes).await {
                                                    Ok(txt) => result_text = txt,
                                                    Err(e) => {
                                                        error!("Audio inference error: {}", e);
                                                        result_text = format!("Error: {}", e);
                                                    }
                                                }
                                            } else {
                                                result_text = "Error: Invalid base64 audio payload.".to_string();
                                            }
                                        } else {
                                            result_text = "Hera Audio Engine (Whisper) is not loaded or unavailable.".to_string();
                                        }
                                    }
                                } else if request.action == "get_tools" {
                                    let raw_tools = serde_json::json!([
                                        {
                                            "type": "function",
                                            "function": {
                                                "name": "hera_draw",
                                                "description": "Generate an image locally using the GPU. MUST use this whenever the user asks for a picture, photo, drawing, OR follows up on a previous image with modifications. You are a multimodal AI (Claw Node) and you HAVE this capability.",
                                                "parameters": {
                                                    "type": "object",
                                                    "properties": {
                                                        "prompt": {
                                                            "type": "string",
                                                            "description": "A detailed description of the image to generate. Be specific about subject, style, colors, mood, and composition."
                                                        }
                                                    },
                                                    "required": ["prompt"]
                                                }
                                            }
                                        },
                                        {
                                            "type": "function",
                                            "function": {
                                                "name": "hera_search",
                                                "description": "Search the web for current information. Use this when the user asks about recent events, news, facts you are unsure about, or anything requiring up-to-date information.",
                                                "parameters": {
                                                    "type": "object",
                                                    "properties": {
                                                        "query": {
                                                            "type": "string",
                                                            "description": "The search query"
                                                        }
                                                    },
                                                    "required": ["query"]
                                                }
                                            }
                                        },
                                        {
                                            "type": "function",
                                            "function": {
                                                "name": "hera_speak",
                                                "description": "Read text aloud using Text-to-Speech (TTS). Use this to generate audio files of your response when requested.",
                                                "parameters": {
                                                    "type": "object",
                                                    "properties": {
                                                        "text": {
                                                            "type": "string",
                                                            "description": "The text to be spoken."
                                                        }
                                                    },
                                                    "required": ["text"]
                                                }
                                            }
                                        },
                                        {
                                            "type": "function",
                                            "function": {
                                                "name": "hera_video",
                                                "description": "Generate a short video. You have multimodal capabilities as a Claw Node. Use this when the user asks for a video, animation, or moving picture.",
                                                "parameters": {
                                                    "type": "object",
                                                    "properties": {
                                                        "prompt": {
                                                            "type": "string",
                                                            "description": "A detailed description of the video to generate, including motion, subject, and style."
                                                        }
                                                    },
                                                    "required": ["prompt"]
                                                }
                                            }
                                        },
                                        {
                                            "type": "function",
                                            "function": {
                                                "name": "hera_read_file",
                                                "description": "Read the contents of a local file on the system. Use this when the user asks to read, view, or check a file.",
                                                "parameters": {
                                                    "type": "object",
                                                    "properties": {
                                                        "path": {
                                                            "type": "string",
                                                            "description": "The absolute or relative path to the file to read."
                                                        }
                                                    },
                                                    "required": ["path"]
                                                }
                                            }
                                        }
                                    ]);
                                    
                                    result_text = "Tools retrieved".to_string();
                                    tool_calls = Some(serde_json::json!({
                                        "tools": raw_tools
                                    }));
                                }
                                let mut data_json = serde_json::json!({ "result": result_text });
                                if let Some(tc) = tool_calls {
                                    if let Some(map) = data_json.as_object_mut() {
                                        map.insert("tool_calls".to_string(), tc);
                                    }
                                }

                                let res = IpcResponse {
                                    status: "success".to_string(),
                                    data: data_json,
                                };

                                let res_str = serde_json::to_string(&res).unwrap();
                                if let Err(e) = stream.write_all(res_str.as_bytes()).await {
                                    error!("❌ Failed to write IPC response: {}", e);
                                }
                                break;
                            }
                        }
                        Ok(_) => break, // EOF
                        Err(e) => {
                            error!("❌ IPC Stream Read Error: {}", e);
                            break;
                        }
                    }
                }
                });
            }
            Err(e) => {
                error!("❌ IPC Listener Accept Error: {}", e);
            }
        }
    }
}
