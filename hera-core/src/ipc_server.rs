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
                                
                                if request.action == "generate" {
                                    if let Some(prompt) = request.payload.get("prompt").and_then(|p| p.as_str()) {
                                        let chat_req = ChatRequest {
                                            model: "hera-local-model".to_string(),
                                            vision_model: None,
                                            tts_model: None,
                                            stt_model: None,
                                            messages: vec![ChatMessage {
                                                role: "user".to_string(),
                                                content: MessageContent::Text(prompt.to_string()),
                                            }],
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
                                        };
                                        
                                        match state.engine.generate_content(chat_req).await {
                                            Ok(resp) => {
                                                if let Some(choice) = resp.choices.first() {
                                                    if let Some(content) = &choice.message.content {
                                                        result_text = content.clone();
                                                    }
                                                }
                                            }
                                            Err(e) => {
                                                error!("LLM inference error: {}", e);
                                                result_text = format!("Error: {}", e);
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
                                }

                                let res = IpcResponse {
                                    status: "success".to_string(),
                                    data: serde_json::json!({ "result": result_text }),
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
