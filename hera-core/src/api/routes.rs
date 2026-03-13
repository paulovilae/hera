//! External Application Programming Interfaces
//!
//! Exposes Axum web bounds bridging network traffic directly against the
//! abstracted sovereign AI endpoints (`/v1/chat/completions`).

use crate::ai::{ChatRequest, LLMEngine};
use axum::{Json, Router, extract::State, http::StatusCode, routing::{get, post}};
use serde::Serialize;
use std::sync::Arc;
use std::env;

/// Containerizes the executing backend engine bindings.
pub struct ApiState {
    pub engine: Arc<dyn LLMEngine + Send + Sync>,
    pub local_engine: Arc<dyn LLMEngine + Send + Sync>,
    pub flux_engine: Option<Arc<crate::ai::engine_flux::FluxEngine>>,
    pub parler_engine: Option<Arc<crate::ai::engine_parler::ParlerEngine>>,
    pub whisper_engine: Option<Arc<crate::ai::engine_whisper::WhisperEngine>>,
    pub vision_engine: Option<Arc<dyn LLMEngine + Send + Sync>>,
    pub micro_engine: Option<Arc<dyn LLMEngine + Send + Sync>>,
}

#[derive(Serialize)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<ModelData>,
}

#[derive(Serialize)]
pub struct ModelData {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

use crate::api::workspace;
use crate::api::agent_canvas;

/// Binds the `v1` abstraction surface into the HTTP web sockets.
pub fn create_router(state: ApiState) -> Router {
    Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(chat_completions))
        .route("/v1/messages", post(anthropic_messages))
        .route("/v1/hera/execute_tool", post(execute_tool))
        .route("/v1/hera/memory/store", post(memory_store))
        .route("/v1/hera/memory/search", post(memory_search))
        .route("/v1/hera/chat", post(hera_chat))
        .route("/v1/hera/audio", post(hera_audio))
        .route("/v1/hera/image", post(hera_image))
        .route("/v1/hera/draw", post(hera_draw))
        .route("/v1/hera/video", post(hera_video))
        .route("/v1/hera/checkpoints", get(hera_list_checkpoints))
        .route("/v1/hera/loras", get(hera_list_loras))
        .route("/v1/hera/transcribe", post(hera_transcribe))
        .route("/v1/hera/scrape", post(hera_scrape))
        .route("/v1/hera/search", post(hera_search))
        .route("/v1/hera/extract", post(crate::api::universal_extract::hera_extract_sse))
        .route("/v1/hera/generate-pdf", post(generate_pdf))
        .route("/v1/hera/system/gpu", get(system_gpu))
        .route("/v1/hera/workflow/execute", post(workflow_execute))
        .route("/v1/hera/workflow/dify", post(workflow_execute_dify))
        .route("/v1/hera/data/excel", post(crate::api::excel::parse_excel_upload))
        .route("/v1/hera/agent/canvas", post(agent_canvas::process_canvas_request))
        .route("/v1/hera/nodes", get(crate::api::nodes::list_nodes))
        .route("/v1/hera/workspaces", get(workspace::list_workspaces).post(workspace::create_workspace))
        .route("/v1/hera/workspaces/{workspace_id}", axum::routing::delete(workspace::delete_workspace))
        .route("/v1/hera/workspaces/{workspace_id}/workflows", get(workspace::list_workflows))
        .route("/v1/hera/workspaces/{workspace_id}/workflows/{workflow_id}", get(workspace::get_workflow).post(workspace::save_workflow).delete(workspace::delete_workflow))
        .route("/outputs/{filename}", get(serve_output_image))
        .with_state(Arc::new(state))
}

use axum::response::{sse::{Event, Sse}, IntoResponse};
use futures_util::stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

/// Strips `<think>...</think>` blocks from Qwen's output (internal reasoning tags)
fn strip_think_tags(text: &str) -> String {
    let mut result = text.to_string();
    while let Some(start) = result.find("<think>") {
        if let Some(end) = result.find("</think>") {
            result = format!("{}{}", &result[..start], &result[end + "</think>".len()..]);
        } else {
            // Unclosed think tag — strip from <think> to end
            result = result[..start].to_string();
            break;
        }
    }
    result.trim().to_string()
}

/// Executes an incoming multimodal request directly against the attached LLM orchestrator.
async fn chat_completions(
    State(state): State<Arc<ApiState>>,
    Json(mut payload): Json<ChatRequest>,
) -> axum::response::Response {
    // Global image detection — route image payloads to Native Vision regardless of model name
    let mut has_image = false;
    for msg in &payload.messages {
        if let crate::ai::MessageContent::Parts(parts) = &msg.content {
            for part in parts {
                if let crate::ai::ContentPart::ImageUrl { .. } = part {
                    has_image = true;
                    break;
                }
            }
        }
    }

    // If any image is found, route to Native Vision Engine first
    if has_image && payload.provider.as_deref() != Some("cloud") {
        if state.vision_engine.is_some() {
            tracing::info!("🖼️ Image payload detected → Routing locally via Native Moondream Vision Engine");
            payload.model = "moondream-q4".to_string();
            payload.provider = Some("local_native_vision".to_string());
        } else {
            tracing::warn!("🖼️ Image payload detected but Native Vision offline. Falling back to Cloud Vision");
            payload.model = "google/gemini-2.0-flash-exp:free".to_string();
            payload.provider = Some("auto".to_string());
        }
    }

    // Inherit standard interception for Hera identity routing
    let model_lower = payload.model.to_lowercase();
    let is_hera = model_lower == "hera";
    let is_ava = model_lower == "ava";

    if is_hera || is_ava {
        if is_hera {
            // 🛡️ Prompt Sanitization: OpenClaw sends massive agent prompts (30KB+) with
            // tool definitions, policies, etc. Strip everything except user/assistant messages
            // to fit within the local Qwen context window, then prepend our own Hera persona.
            let has_openclaw_tools = payload.messages.iter().any(|m| {
                if m.role == "system" {
                    if let crate::ai::MessageContent::Text(t) = &m.content {
                        return t.contains("## Tooling") || t.contains("Tool availability") || t.contains("OpenClaw");
                    }
                }
                false
            });

            if has_openclaw_tools {
                let total_bytes: usize = payload.messages.iter().map(|m| match &m.content {
                    crate::ai::MessageContent::Text(t) => t.len(),
                    crate::ai::MessageContent::Parts(p) => p.len() * 50,
                    crate::ai::MessageContent::Null => 0,
                }).sum();
                tracing::info!("🧹 [Hera] Detected OpenClaw agent prompt ({} messages, {} bytes). Soft-sanitizing...",
                    payload.messages.len(), total_bytes
                );

                // Soft sanitization: strip ONLY system messages with heavy tool definitions
                // Keep system messages that are short (personality, instructions, etc.)
                payload.messages.retain(|m| {
                    if m.role == "system" {
                        if let crate::ai::MessageContent::Text(t) = &m.content {
                            // Strip messages with tool schemas (they're huge and Hera doesn't use them)
                            if t.contains("## Tooling") || t.contains("Tool availability") || t.contains("tool_call_guidelines") {
                                return false;
                            }
                            // Keep short system messages (personality, context)
                            return t.len() < 2000;
                        }
                    }
                    true // Keep all user/assistant messages
                });

                // Keep last 50 messages to preserve conversation context within 32K window
                if payload.messages.len() > 50 {
                    // Keep first system message + last 49 conversation messages
                    let first_system = payload.messages.iter().position(|m| m.role == "system");
                    let mut kept = Vec::new();
                    if let Some(idx) = first_system {
                        kept.push(payload.messages[idx].clone());
                    }
                    let start = payload.messages.len().saturating_sub(49);
                    kept.extend(payload.messages[start..].iter()
                        .filter(|m| m.role != "system")
                        .cloned());
                    payload.messages = kept;
                }

                let sanitized_bytes: usize = payload.messages.iter().map(|m| match &m.content {
                    crate::ai::MessageContent::Text(t) => t.len(),
                    crate::ai::MessageContent::Parts(p) => p.len() * 50,
                    crate::ai::MessageContent::Null => 0,
                }).sum();
                tracing::info!("🧹 [Hera] After sanitization: {} messages, {} bytes (saved {} bytes)",
                    payload.messages.len(), sanitized_bytes, total_bytes.saturating_sub(sanitized_bytes)
                );
            }

            let persona_msg = crate::ai::ChatMessage {
                role: "system".to_string(),
                content: crate::ai::MessageContent::Text(
                    "You are Hera, the sovereign multimodal AI assistant of ImagineOS. You run on local hardware (dual RTX 3090 GPUs). You are helpful, concise, and respond in the user's language.\n\n\
                    IMPORTANT: You cannot generate images, audio, video, or search the web by yourself. \
                    You MUST use your tools (described below) to perform these actions. \
                    When asked to draw/create an image, OR when the user modifies a previously generated image (e.g. 'now with a hat', 'ahora fumando'), you MUST output a <tool_call> block for hera_draw. \
                    Do NOT pretend you already generated something. Do NOT hallucinate filenames or URLs. \
                    If no tool is needed, just respond normally.".to_string()
                ),
            };
            payload.messages.insert(0, persona_msg);
        } else {
            // Ava - the Sovereign Admin. We keep full OpenClaw tools because she needs them.
            tracing::info!("🛡️ [Ava] Detected Admin payload. Retaining full MCP tools...");
        }

        // Route directly to local engine for both
        payload.provider = Some("local_direct".to_string());
    }

    // Determine target execution boundary
    let engine_to_use = if payload.provider.as_deref() == Some("local_native_vision") {
        state.vision_engine.clone().expect("Vision engine must exist if provider is set to it")
    } else if payload.provider.as_deref() == Some("local_direct") {
        tracing::info!("⚡ Routing directly to local Sovereign LLM (bypassing cloud orchestrator)");
        state.local_engine.clone()
    } else {
        state.engine.clone()
    };

    // 🔧 Hera/Ava Tool Calling Loop — intercept tool calls, execute, re-prompt
    // Trigger for ALL local_direct requests matching the intercept boundaries
    let model_lower = payload.model.to_lowercase();
    let is_hera_direct = (model_lower == "hera" || model_lower == "ava")
        && !has_image; // Skip tool calling for vision requests (handled separately)

    if is_hera_direct {
        // Inject tool schemas into the system prompt
        let tool_schemas = crate::ai::tool_executor::hera_tool_schemas();
        // Find existing Hera persona system message and append tool schemas
        if let Some(sys_msg) = payload.messages.iter_mut().find(|m| m.role == "system") {
            if let crate::ai::MessageContent::Text(ref mut text) = sys_msg.content {
                text.push_str(&tool_schemas);
            }
        }

        // Force direct FFI engine for tool detection (bypass ContextEngine which adds its own system prompt)
        let hera_engine = state.local_engine.clone();

        // Non-streaming first pass to check for tool calls
        let mut first_pass_payload = payload.clone();
        first_pass_payload.stream = Some(false);
        first_pass_payload.max_tokens = Some(1024);
        // Add stop sequence so Qwen stops right after emitting </tool_call>
        first_pass_payload.stop = Some(vec!["</tool_call>".to_string()]);

        // Extract user's last message for fallback intent detection
        let user_last_msg = payload.messages.iter().rev()
            .find(|m| m.role == "user")
            .map(|m| match &m.content {
                crate::ai::MessageContent::Text(t) => t.clone(),
                crate::ai::MessageContent::Parts(p) => p.iter().filter_map(|part| match part {
                    crate::ai::ContentPart::Text { text } => Some(text.clone()),
                    _ => None,
                }).collect::<Vec<_>>().join(" "),
                crate::ai::MessageContent::Null => String::new(),
            })
            .unwrap_or_default();

        let assistant_last_msg = payload.messages.iter().rev()
            .find(|m| m.role == "assistant")
            .map(|m| match &m.content {
                crate::ai::MessageContent::Text(t) => t.clone(),
                crate::ai::MessageContent::Parts(p) => p.iter().filter_map(|part| match part {
                    crate::ai::ContentPart::Text { text } => Some(text.clone()),
                    _ => None,
                }).collect::<Vec<_>>().join(" "),
                crate::ai::MessageContent::Null => String::new(),
            });

        // Try intent detection FIRST (faster, model-agnostic, works with any model size)
        let intent_call = crate::ai::tool_executor::detect_intent_from_user_message(&user_last_msg, assistant_last_msg.as_deref());

        if let Some(mut tool_call) = intent_call {
            // ── STEP 1: PREPARE — Let LLM refine the raw user message into a proper tool argument ──
            tracing::info!("🎯 [Hera] Intent-based tool call: {} — running LLM pre-pass to refine arguments", tool_call.name);

            let prepare_instruction = match tool_call.name.as_str() {
                "hera_draw" => format!(
                    "The user wants you to generate an image. Their request: \"{}\"\n\n\
                    Respond with ONLY a detailed English image generation prompt for Stable Diffusion. \
                    No explanation, no commentary, just the prompt. \
                    Example: \"A majestic cat playing with a red ball in a sunlit garden, digital art, highly detailed, 8k\"\n\
                    If they ask for YOUR photo or selfie, describe a futuristic female AI agent named Hera with sleek design, glowing blue and purple accents, cyberpunk style.",
                    user_last_msg
                ),
                "hera_search" => format!(
                    "The user wants to search the web. Their request: \"{}\"\n\n\
                    Respond with ONLY a clean, well-formed search query. Fix any typos. \
                    No explanation, no commentary, just the search query.",
                    user_last_msg
                ),
                _ => format!(
                    "The user wants to use the {} tool. Their request: \"{}\"\n\n\
                    Respond with ONLY the refined input for this tool. No explanation.",
                    tool_call.name, user_last_msg
                ),
            };

            let mut prepare_payload = payload.clone();
            prepare_payload.stream = Some(false);
            prepare_payload.max_tokens = Some(150);
            prepare_payload.messages.push(crate::ai::ChatMessage {
                role: "user".to_string(),
                content: crate::ai::MessageContent::Text(prepare_instruction),
            });

            // Run LLM pre-pass to get refined argument
            let refined_arg = if payload.nsfw.unwrap_or(false) && tool_call.name == "hera_draw" {
                tracing::info!("🔞 [Hera] NSFW bypass active: Attempting Uncensored Micro-LLM refinement...");
                if let Some(micro) = &state.micro_engine {
                    match micro.generate_content(prepare_payload.clone()).await {
                        Ok(response) => {
                            let raw = response.choices.first()
                                .and_then(|c| c.message.content.as_deref())
                                .unwrap_or(&user_last_msg);
                            let cleaned = strip_think_tags(raw).trim().trim_matches('"').to_string();
                            tracing::info!("🔞 [Hera] Micro-LLM refined argument: {}", &cleaned[..cleaned.len().min(200)]);
                            cleaned
                        }
                        Err(e) => {
                            tracing::warn!("🔞 [Hera] Micro-LLM refinement failed ({}), using raw message", e);
                            user_last_msg.clone()
                        }
                    }
                } else {
                    tracing::info!("🔞 [Hera] Micro-LLM unavailable, using raw prompt bypass");
                    user_last_msg.clone()
                }
            } else {
                match hera_engine.generate_content(prepare_payload).await {
                    Ok(response) => {
                        let raw = response.choices.first()
                            .and_then(|c| c.message.content.as_deref())
                            .unwrap_or(&user_last_msg);
                        let cleaned = strip_think_tags(raw).trim().trim_matches('"').to_string();
                        tracing::info!("🎯 [Hera] LLM refined argument: {}", &cleaned[..cleaned.len().min(200)]);
                        cleaned
                    }
                    Err(e) => {
                        tracing::warn!("🎯 [Hera] LLM pre-pass failed ({}), using raw message", e);
                        user_last_msg.clone()
                    }
                }
            };

            // Update tool call arguments with the refined value
            match tool_call.name.as_str() {
                "hera_draw" => { tool_call.arguments = serde_json::json!({"prompt": refined_arg}); }
                "hera_search" => { tool_call.arguments = serde_json::json!({"query": refined_arg}); }
                "hera_speak" => { tool_call.arguments = serde_json::json!({"text": refined_arg}); }
                "hera_video" => { tool_call.arguments = serde_json::json!({"prompt": refined_arg}); }
                _ => {}
            }

            // ── STEP 2: EXECUTE — Run the tool with the refined argument ──
            let tool_result = crate::ai::tool_executor::execute_tool(&tool_call).await;
            tracing::info!("🎯 [Hera] Tool result (success={}): {}",
                tool_result.success, &tool_result.output[..tool_result.output.len().min(200)]);

            // For image generation: skip LLM narration and return MEDIA: directive directly
            // (The LLM would strip the MEDIA: line, preventing OpenClaw from sending inline images)
            if tool_call.name == "hera_draw" && tool_result.success {
                let media_line = tool_result.output.lines()
                    .find(|l| l.starts_with("MEDIA:"))
                    .unwrap_or("")
                    .to_string();
                let response_text = format!("¡Aquí tienes!\n{}", media_line);

                let direct_response = crate::ai::ChatResponse {
                    id: format!("chatcmpl-{}", std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs()),
                    object: "chat.completion".to_string(),
                    created: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs(),
                    model: "hera".to_string(),
                    choices: vec![crate::ai::ChatChoice {
                        index: 0,
                        message: crate::ai::ChatResponseMessage {
                            role: "assistant".to_string(),
                            content: Some(response_text),
                            tool_calls: None,
                        },
                        finish_reason: Some("stop".to_string()),
                    }],
                    usage: Some(crate::ai::ChatUsage {
                        prompt_tokens: 0,
                        completion_tokens: 0,
                        total_tokens: 0,
                    }),
                };

                if payload.stream.unwrap_or(false) {
                    // Emit as a single SSE chunk + [DONE]
                    let chunk = serde_json::json!({
                        "id": direct_response.id,
                        "object": "chat.completion.chunk",
                        "created": direct_response.created,
                        "model": "hera",
                        "choices": [{
                            "index": 0,
                            "delta": {"role": "assistant", "content": direct_response.choices[0].message.content},
                            "finish_reason": "stop"
                        }]
                    });
                    let chunk_str = serde_json::to_string(&chunk).unwrap_or_default();
                    let data_stream = futures_util::stream::iter(vec![
                        Ok::<_, std::convert::Infallible>(Event::default().data(chunk_str)),
                        Ok::<_, std::convert::Infallible>(Event::default().data("[DONE]")),
                    ]);
                    return Sse::new(data_stream).into_response();
                } else {
                    return axum::Json(direct_response).into_response();
                }
            }

            // For non-image tools: use LLM narration as before
            let narration_prompt = format!(
                "[System: Tool '{}' was executed. Result: {}]\n\n\
                Respond naturally to the user about this result. Be concise. Do NOT use tool_call tags.",
                tool_result.name, tool_result.output
            );

            payload.messages.push(crate::ai::ChatMessage {
                role: "user".to_string(),
                content: crate::ai::MessageContent::Text(narration_prompt),
            });

            // Stream the follow-up response
            if payload.stream.unwrap_or(false) {
                match hera_engine.generate_stream(payload).await {
                    Ok(rx) => {
                        let data_stream = ReceiverStream::new(rx).map(|res| {
                            match res {
                                Ok(chunk) => {
                                    let json = serde_json::to_string(&chunk).unwrap_or_else(|_| "{}".to_string());
                                    Ok::<_, std::convert::Infallible>(Event::default().data(json))
                                }
                                Err(e) => {
                                    let err_json = format!(r#"{{"error": "{}"}}"#, e.to_string().replace("\"", "\\\""));
                                    Ok::<_, std::convert::Infallible>(Event::default().data(err_json))
                                }
                            }
                        });
                        let done_stream = futures_util::stream::once(async {
                            Ok::<_, std::convert::Infallible>(Event::default().data("[DONE]"))
                        });
                        return Sse::new(data_stream.chain(done_stream)).into_response();
                    }
                    Err(e) => {
                        tracing::error!("Intent tool follow-up streaming failed: {:?}", e);
                        return (StatusCode::INTERNAL_SERVER_ERROR, format!("Stream error: {}", e)).into_response();
                    }
                }
            } else {
                match hera_engine.generate_content(payload).await {
                    Ok(res) => return axum::Json(res).into_response(),
                    Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("Error: {}", e)).into_response(),
                }
            }
        }

        // No intent detected — proceed with LLM first pass for tool_call tag parsing
        tracing::info!("🔧 [Hera] Running non-streaming first pass (direct FFI) to detect tool calls...");

        match hera_engine.generate_content(first_pass_payload).await {
            Ok(response) => {
                let raw_text = response.choices.first()
                    .and_then(|c| c.message.content.as_deref())
                    .unwrap_or("");

                // Strip <think> tags from Qwen's reasoning before parsing
                let response_text = strip_think_tags(raw_text);

                let tool_calls = crate::ai::tool_executor::parse_tool_calls(&response_text);

                if !tool_calls.is_empty() {
                    // Execute the first tool call
                    let tool_call = &tool_calls[0];
                    tracing::info!("🔧 [Hera] Tool call detected: {} — executing...", tool_call.name);

                    let tool_result = crate::ai::tool_executor::execute_tool(tool_call).await;
                    tracing::info!("🔧 [Hera] Tool result (success={}): {}",
                        tool_result.success, &tool_result.output[..tool_result.output.len().min(200)]);

                    // Append assistant tool call + tool result, then re-prompt for final response
                    payload.messages.push(crate::ai::ChatMessage {
                        role: "assistant".to_string(),
                        content: crate::ai::MessageContent::Text(response_text.to_string()),
                    });
                    payload.messages.push(crate::ai::ChatMessage {
                        role: "user".to_string(),
                        content: crate::ai::MessageContent::Text(
                            format!("[Tool '{}' result]: {}\n\nNow give the user a natural response based on this tool result. Do NOT use <tool_call> tags again.",
                                tool_result.name, tool_result.output)
                        ),
                    });

                    // Second pass — streaming for the final response
                    if payload.stream.unwrap_or(false) {
                        match hera_engine.generate_stream(payload).await {
                            Ok(rx) => {
                                let data_stream = ReceiverStream::new(rx).map(|res| {
                                    match res {
                                        Ok(chunk) => {
                                            let json = serde_json::to_string(&chunk).unwrap_or_else(|_| "{}".to_string());
                                            Ok::<_, std::convert::Infallible>(Event::default().data(json))
                                        }
                                        Err(e) => {
                                            let err_json = format!(r#"{{"error": "{}"}}"#, e.to_string().replace("\"", "\\\""));
                                            Ok::<_, std::convert::Infallible>(Event::default().data(err_json))
                                        }
                                    }
                                });
                                let done_stream = futures_util::stream::once(async {
                                    Ok::<_, std::convert::Infallible>(Event::default().data("[DONE]"))
                                });
                                return Sse::new(data_stream.chain(done_stream)).into_response();
                            }
                            Err(e) => {
                                tracing::error!("Tool follow-up streaming failed: {:?}", e);
                                return (StatusCode::INTERNAL_SERVER_ERROR, format!("Stream error: {}", e)).into_response();
                            }
                        }
                    } else {
                        // Non-streaming second pass
                        match hera_engine.generate_content(payload).await {
                            Ok(res) => return axum::Json(res).into_response(),
                            Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("Error: {}", e)).into_response(),
                        }
                    }
                } else {
                    // 1. Check for S.O.L operations in the raw reasoning text
                    let mut sol_results = Vec::new();
                    for line in raw_text.lines() {
                        let trimmed = line.trim();
                        if let Ok(op) = crate::sol::SolParser::parse(trimmed) {
                            let result = crate::sol::execute_sol(&op);
                            tracing::info!("🔮 [Hera] S.O.L executed: {}", result);
                            sol_results.push(result);
                        }
                    }

                    if !sol_results.is_empty() {
                        tracing::info!("🔮 [Hera] S.O.L commands detected! Injecting graph context...");
                        
                        // Append the LLM's own reasoning so it remembers its chain of thought
                        payload.messages.push(crate::ai::ChatMessage {
                            role: "assistant".to_string(),
                            content: crate::ai::MessageContent::Text(raw_text.to_string()),
                        });

                        // Append the graph context it requested
                        payload.messages.push(crate::ai::ChatMessage {
                            role: "user".to_string(),
                            content: crate::ai::MessageContent::Text(
                                format!("[S.O.L Graph Execution Results]:\n{}\n\nContinue reasoning or respond to the user based on this context.", sol_results.join("\n"))
                            ),
                        });

                        // Second pass — streaming for the final response after S.O.L execution
                        if payload.stream.unwrap_or(false) {
                            match hera_engine.generate_stream(payload).await {
                                Ok(rx) => {
                                    let data_stream = ReceiverStream::new(rx).map(|res| {
                                        match res {
                                            Ok(chunk) => {
                                                let json = serde_json::to_string(&chunk).unwrap_or_else(|_| "{}".to_string());
                                                Ok::<_, std::convert::Infallible>(Event::default().data(json))
                                            }
                                            Err(e) => {
                                                let err_json = format!(r#"{{"error": "{}"}}"#, e.to_string().replace("\"", "\\\""));
                                                Ok::<_, std::convert::Infallible>(Event::default().data(err_json))
                                            }
                                        }
                                    });
                                    let done_stream = futures_util::stream::once(async {
                                        Ok::<_, std::convert::Infallible>(Event::default().data("[DONE]"))
                                    });
                                    return Sse::new(data_stream.chain(done_stream)).into_response();
                                }
                                Err(e) => {
                                    tracing::error!("S.O.L follow-up streaming failed: {:?}", e);
                                    return (StatusCode::INTERNAL_SERVER_ERROR, format!("Stream error: {}", e)).into_response();
                                }
                            }
                        } else {
                            // Non-streaming second pass
                            match hera_engine.generate_content(payload).await {
                                Ok(res) => return axum::Json(res).into_response(),
                                Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("Error: {}", e)).into_response(),
                            }
                        }
                    } else {
                        // 2. No tool call and no S.O.L — wrap first-pass result as SSE (avoids double LLM call)
                        if payload.stream.unwrap_or(false) {
                            tracing::info!("🔧 [Hera] No tool/S.O.L detected — wrapping first-pass result as SSE");
                            let content = response_text.to_string();
                            let chunk = crate::ai::ChatStreamResponse {
                                id: response.id.clone(),
                                object: "chat.completion.chunk".to_string(),
                                created: response.created,
                                model: response.model.clone(),
                                choices: vec![crate::ai::ChatStreamChoice {
                                    index: 0,
                                    delta: crate::ai::ChatStreamDelta {
                                        role: Some("assistant".to_string()),
                                        content: Some(content),
                                        tool_calls: None,
                                    },
                                    finish_reason: Some("stop".to_string()),
                                }],
                                stats: None,
                            };
                            let chunk_json = serde_json::to_string(&chunk).unwrap_or_else(|_| "{}".to_string());
                            let data_stream = futures_util::stream::once(async move {
                                Ok::<_, std::convert::Infallible>(Event::default().data(chunk_json))
                            });
                            let done_stream = futures_util::stream::once(async {
                                Ok::<_, std::convert::Infallible>(Event::default().data("[DONE]"))
                            });
                            return Sse::new(data_stream.chain(done_stream)).into_response();
                        } else {
                            return axum::Json(response).into_response();
                        }
                    }
                }
            }
            Err(e) => {
                tracing::error!("Tool detection first-pass failed: {:?}", e);
                return (StatusCode::INTERNAL_SERVER_ERROR, format!("Inference error: {}", e)).into_response();
            }
        }
    }

    // Standard non-Hera path (unchanged)
    if payload.stream.unwrap_or(false) {
        match engine_to_use.generate_stream(payload).await {
            Ok(rx) => {
                let data_stream = tokio_stream::wrappers::ReceiverStream::new(rx).map(|res: Result<crate::ai::ChatStreamResponse, crate::ai::InferenceError>| {
                    match res {
                        Ok(chunk) => {
                            let json = serde_json::to_string(&chunk).unwrap_or_else(|_| "{}".to_string());
                            Ok::<_, std::convert::Infallible>(axum::response::sse::Event::default().data(json))
                        }
                        Err(e) => {
                            let err_json = format!(r#"{{"error": "{}"}}"#, e.to_string().replace("\"", "\\\""));
                            Ok::<_, std::convert::Infallible>(axum::response::sse::Event::default().data(err_json))
                        }
                    }
                });
                // Append [DONE] sentinel required by OpenAI-compatible clients
                let done_stream = futures_util::stream::once(async {
                    Ok::<_, std::convert::Infallible>(axum::response::sse::Event::default().data("[DONE]"))
                });
                let full_stream = data_stream.chain(done_stream);
                axum::response::Sse::new(full_stream).into_response()
            }
            Err(e) => {
                tracing::error!("Inference Engine streaming collapsed inside API bounds: {:?}", e);
                (StatusCode::INTERNAL_SERVER_ERROR, axum::Json(format!("Stream error: {}", e))).into_response()
            }
        }
    } else {
        match engine_to_use.generate_content(payload).await {
            Ok(res) => axum::Json(res).into_response(),
            Err(e) => {
                tracing::error!("Inference Engine collapsed inside API bounds: {:?}", e);
                (StatusCode::INTERNAL_SERVER_ERROR, axum::Json(format!("Inference error: {}", e))).into_response()
            }
        }
    }
}

/// Discovers hardware-mounted LLMs natively and returns a standard OpenAI payload
async fn list_models() -> Result<Json<ModelList>, (StatusCode, String)> {
    let mut data = Vec::new();
    
    // Cloud defaults and Native OS Agents
    data.push(ModelData {
        id: "hera".to_string(),
        object: "model".to_string(),
        created: 1715694400,
        owned_by: "hera-engine".to_string(),
    });
    data.push(ModelData {
        id: "gemini-2.5-flash".to_string(),
        object: "model".to_string(),
        created: 1715694400,
        owned_by: "google".to_string(),
    });
    data.push(ModelData {
        id: "gemini-3.0-pro".to_string(),
        object: "model".to_string(),
        created: 1715694400,
        owned_by: "google".to_string(),
    });

    // Native Sovereign Mount
    let model_dirs = vec![
        env::var("HERA_MODEL_DIR").unwrap_or_else(|_| "/mnt/workspace/hera-data/models/llms".to_string()),
        "/data/models/llm-stack".to_string()
    ];

    for model_dir in model_dirs {
        if let Ok(entries) = std::fs::read_dir(&model_dir) {
            for entry in entries.flatten() {
                if let Ok(file_type) = entry.file_type() {
                    if file_type.is_file() {
                        let file_name = entry.file_name().to_string_lossy().into_owned();
                        // Match valid neural binaries (focusing on Qwen/Gemma/DeepSeek for a clean UI)
                        if (file_name.ends_with(".gguf") || file_name.ends_with(".safetensors")) 
                           && (file_name.to_lowercase().contains("qwen") || 
                               file_name.to_lowercase().contains("gemma") || 
                               file_name.to_lowercase().contains("deepseek")) 
                           && !file_name.contains("mmproj") {
                            
                            // Check if it's a real model file (> 100MB) to filter out LFS pointers or error pages
                            if let Ok(metadata) = entry.metadata() {
                                if metadata.len() < 100_000_000 {
                                    continue;
                                }
                            }

                            // Check if model already added by previous directory
                            if !data.iter().any(|m| m.id == file_name) {
                                let created = entry.metadata().and_then(|m| m.created()).ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_secs()).unwrap_or(0);
                                data.push(ModelData {
                                    // Prepend the full path since Native Engine expects absolute paths for GGUFs
                                    id: format!("{}/{}", model_dir, file_name),
                                    object: "model".to_string(),
                                    created,
                                    owned_by: "sovereign-local".to_string(),
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(Json(ModelList {
        object: "list".to_string(),
        data,
    }))
}

#[derive(serde::Deserialize)]
pub struct ExecuteToolRequest {
    pub server_url: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Autonomously executes a Model Context Protocol (MCP) tool natively.
async fn execute_tool(
    Json(payload): Json<ExecuteToolRequest>,
) -> axum::response::Response {
    let client = hera_execution::mcp::client::McpHttpClient::new(&payload.server_url);
    
    // Attempt standard MCP handshake
    if let Err(e) = client.initialize().await {
        tracing::error!("Failed MCP Handshake: {:?}", e);
        return (StatusCode::BAD_GATEWAY, format!("Failed MCP Handshake: {}", e)).into_response();
    }
    
    // Execute the requested tool payload
    match client.call_tool(&payload.name, payload.arguments.clone()).await {
        Ok(res) => Json(res).into_response(),
        Err(e) => {
            tracing::error!("Failed executing MCP Tool {}: {:?}", payload.name, e);
            (StatusCode::INTERNAL_SERVER_ERROR, format!("MCP Tool Execution Error: {}", e)).into_response()
        }
    }
}

#[derive(serde::Deserialize)]
pub struct MemoryStoreRequest {
    pub id: String,
    pub collection: String,
    pub text: String,
    pub vector: Vec<f32>,
}

#[derive(serde::Deserialize)]
pub struct MemorySearchRequest {
    pub collection: String,
    pub vector: Vec<f32>,
    pub limit: Option<u64>,
}

/// Commits a new contextual vector memory to the LanceDB embedded storage layer.
async fn memory_store(Json(payload): Json<MemoryStoreRequest>) -> axum::response::Response {
    let lance_uri = std::env::var("HERA_LANCEDB_URI").unwrap_or_else(|_| "/home/paulo/Programs/apps/hera/data/lance".to_string());
    
    let store = match hera_execution::memory::lance::LanceStore::new(&lance_uri, &payload.collection).await {
        Ok(s) => s,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to init LanceDB: {:?}", e)).into_response(),
    };

    match store.store(payload.vector, serde_json::json!({
        "id": payload.id,
        "text": payload.text
    })).await {
        Ok(_) => Json(serde_json::json!({"status": "success"})).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("LanceDB Store Error: {}", e)).into_response(),
    }
}

/// Retrieves semantics and context windows from the embedded LanceDB database.
async fn memory_search(Json(payload): Json<MemorySearchRequest>) -> axum::response::Response {
    let lance_uri = std::env::var("HERA_LANCEDB_URI").unwrap_or_else(|_| "/home/paulo/Programs/apps/hera/data/lance".to_string());
    
    let store = match hera_execution::memory::lance::LanceStore::new(&lance_uri, &payload.collection).await {
        Ok(s) => s,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to init LanceDB client: {:?}", e)).into_response(),
    };

    let limit = payload.limit.unwrap_or(5);

    match store.search(payload.vector, limit).await {
        Ok(results) => Json(serde_json::json!({"status": "success", "results": results})).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("LanceDB Search Error: {}", e)).into_response(),
    }
}

#[derive(serde::Deserialize)]
pub struct HeraRequest {
    pub prompt: String,
    #[serde(default)]
    pub voice: Option<String>,
    #[serde(default)]
    pub speed: Option<f32>,
}

/// Binds full OpenAI-compatible multimodal requests into the active Hera Agent loop.
async fn hera_chat(
    State(_state): State<Arc<ApiState>>,
    Json(payload): Json<serde_json::Value>,
) -> axum::response::Response {
    // 1. Instantiate the Multimodal Agent.
    // Assuming the SmartOS/Sovereign Engine is locally bound at 3000 as per standard proxy layout
    let hera = hera_execution::agents::hera::Hera::new("http://127.0.0.1:3000");

    // 2. Route the intercepted UI message into the Agent's reasoning loop
    match hera.chat(payload).await {
        Ok(res) => {
            // Stream the proxy response directly back to the Visualizer UI
            let stream = reqwest_to_axum_stream(res);
            axum::response::Response::builder()
                .status(StatusCode::OK)
                .header("Content-Type", "text/event-stream")
                .header("Cache-Control", "no-cache")
                .header("Connection", "keep-alive")
                .body(axum::body::Body::from_stream(stream))
                .unwrap_or_else(|_| (StatusCode::INTERNAL_SERVER_ERROR, "Stream proxy failure").into_response())
        }
        Err(e) => {
            tracing::error!("Hera Agentic Execution Error: {:?}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, format!("Hera Agent Execution Error: {}", e)).into_response()
        }
    }
}

/// Helper method to pipe `reqwest::Response` streams cleanly into Axum `Body` streams.
fn reqwest_to_axum_stream(
    res: reqwest::Response,
) -> impl futures_util::Stream<Item = Result<bytes::Bytes, std::io::Error>> {
    use futures_util::StreamExt;
    res.bytes_stream().map(|result| result.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e)))
}

/// Triggers Native TTS Audio generation via Piper TTS proxy.
async fn hera_audio(
    Json(payload): Json<HeraRequest>
) -> axum::response::Response {
    let text = &payload.prompt[..payload.prompt.len().min(800)];

    // Auto-detect language from text content, or use explicit voice override
    let voice = if let Some(v) = &payload.voice {
        v.clone()
    } else {
        detect_voice(text).to_string()
    };

    let mut req_body = serde_json::json!({
        "input": text,
        "voice": voice
    });

    if let Some(spd) = payload.speed {
        if let Some(obj) = req_body.as_object_mut() {
            obj.insert("speed".to_string(), serde_json::json!(spd));
        }
    }

    let client = reqwest::Client::new();
    match client.post("http://127.0.0.1:8085/v1/audio/speech")
        .json(&req_body)
        .send()
        .await
    {
        Ok(res) if res.status().is_success() => {
            match res.bytes().await {
                Ok(bytes) => {
                    (
                        StatusCode::OK,
                        [
                            (axum::http::header::CONTENT_TYPE, "audio/wav"),
                        ],
                        bytes.to_vec(),
                    ).into_response()
                }
                Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("Piper read error: {}", e)).into_response()
            }
        }
        Ok(res) => (StatusCode::BAD_GATEWAY, format!("Piper returned {}", res.status())).into_response(),
        Err(e) => {
            tracing::error!("Piper TTS proxy error: {:?}", e);
            (StatusCode::SERVICE_UNAVAILABLE, format!("Piper TTS unavailable: {}", e)).into_response()
        }
    }
}

/// Simple language detection for TTS voice selection.
fn detect_voice(text: &str) -> &'static str {
    let lower = text.to_lowercase();
    // Spanish markers
    let has_spanish_chars = text.contains('ñ') || text.contains('¿') || text.contains('¡');
    let spanish_words = ["hola", "cómo", "qué", "está", "también", "porque", "para", "tiene",
                         "puede", "esto", "como", "pero", "todos", "cuando", "donde", "bueno"];
    let spanish_score: usize = spanish_words.iter().filter(|w| lower.contains(*w)).count()
        + if has_spanish_chars { 3 } else { 0 };

    // Portuguese markers
    let has_pt_chars = text.contains('ã') || text.contains('õ') || (text.contains('ç') && !text.contains('ñ'));
    let pt_words = ["olá", "você", "também", "não", "obrigado", "porque", "quando", "como",
                    "está", "tudo", "bom", "ainda", "muito", "então"];
    let pt_score: usize = pt_words.iter().filter(|w| lower.contains(*w)).count()
        + if has_pt_chars { 3 } else { 0 };

    if spanish_score >= 2 && spanish_score > pt_score {
        "es_MX-claude-high"
    } else if pt_score >= 2 {
        "pt_BR-faber-medium"
    } else {
        "en_US-amy-medium"
    }
}

/// Triggers Native SwarmUI Visual Synthesis via the Hera abstraction boundary.
async fn hera_image(
    State(state): State<Arc<ApiState>>,
    Json(payload): Json<HeraRequest>
) -> axum::response::Response {
    if let Some(flux) = &state.flux_engine {
        match flux.generate_image(&payload.prompt, 1360, 768).await {
            Ok(bytes) => {
                // Save out to playground so it can be seen
                let b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, bytes);
                Json(serde_json::json!({
                    "status": "success",
                    "url": format!("data:image/png;base64,{}", b64)
                })).into_response()
            }
            Err(e) => {
                tracing::error!("Native Flux error: {:?}", e);
                (StatusCode::INTERNAL_SERVER_ERROR, format!("Flux Error: {}", e)).into_response()
            }
        }
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "Native Flux Engine is not initialized on this node.".to_string()).into_response()
    }
}

#[derive(serde::Deserialize)]
pub struct HeraDrawRequest {
    pub prompt: String,
    #[serde(default)]
    pub engine: Option<String>,
    #[serde(default)]
    pub width: Option<u32>,
    #[serde(default)]
    pub height: Option<u32>,
    #[serde(default)]
    pub steps: Option<u32>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub loras: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub init_image: Option<String>,
    #[serde(default)]
    pub denoising_strength: Option<f32>,
    #[serde(default)]
    pub cfg_scale: Option<f32>,
    #[serde(default)]
    pub nsfw: Option<bool>,
}

/// Triggers Native SwarmUI Visual Synthesis via the Hera abstraction boundary.
async fn hera_draw(Json(payload): Json<HeraDrawRequest>) -> axum::response::Response {
    let mcp_url = std::env::var("HERA_MCP_URL").unwrap_or_else(|_| "http://127.0.0.1:8080".to_string());
    tracing::info!("🎨 [Hera API] Received Draw Request. Prompt: {}, MCP Default: {}", &payload.prompt, mcp_url);
    
    // We should be passing the configured SwarmUI/SD-Server port here
    let swarm_url = "http://127.0.0.1:8810".to_string(); // sd.cpp server API
    tracing::info!("🎨 [Hera API] Forwarding to: {}", swarm_url);
    
    let hera = hera_execution::agents::hera::Hera::new(&swarm_url);
    
    match hera.generate_image(
        &payload.prompt, 
        payload.engine.as_deref(),
        payload.width,
        payload.height,
        payload.steps,
        payload.model.as_deref(),
        payload.loras.as_ref(),
        payload.init_image.as_deref(),
        payload.denoising_strength,
        payload.cfg_scale,
        payload.nsfw,
    ).await {
        Ok(res) => Json(res).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("Hera Draw Execution Error: {}", e)).into_response(),
    }
}

/// Lists all available checkpoint models from the SwarmUI model directory.
async fn hera_list_checkpoints() -> axum::response::Response {
    let checkpoints = hera_execution::agents::hera::Hera::list_checkpoints();
    Json(serde_json::json!({
        "checkpoints": checkpoints
    })).into_response()
}

/// Lists all available LoRA adapters with trigger word metadata.
async fn hera_list_loras() -> axum::response::Response {
    let loras = hera_execution::agents::hera::Hera::list_loras();
    Json(serde_json::json!({
        "loras": loras
    })).into_response()
}

/// Triggers Native LTX-2 Local Video generation bindings.
async fn hera_video(Json(payload): Json<HeraRequest>) -> axum::response::Response {
    let mcp_url = std::env::var("HERA_MCP_URL").unwrap_or_else(|_| "http://127.0.0.1:8080".to_string());
    let hera = hera_execution::agents::hera::Hera::new(&mcp_url);
    
    match hera.synthesize_video(&payload.prompt).await {
        Ok(res) => Json(res).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("Hera Video Execution Error: {}", e)).into_response(),
    }
}

#[derive(serde::Deserialize)]
pub struct AudioRequest {
    pub audio: String,
}

/// Triggers Native Audio Transcription via the Hera pipeline.
async fn hera_transcribe(
    State(state): State<Arc<ApiState>>,
    Json(payload): Json<AudioRequest>
) -> axum::response::Response {
    if let Some(whisper) = &state.whisper_engine {
        match base64::Engine::decode(&base64::engine::general_purpose::STANDARD, payload.audio) {
            Ok(bytes) => {
                // Convert browser audio (webm/opus) to 16kHz mono WAV via ffmpeg
                let temp_in = format!("/tmp/stt_in_{}.webm", std::process::id());
                let temp_out = format!("/tmp/stt_out_{}.wav", std::process::id());
                
                if let Err(e) = tokio::fs::write(&temp_in, &bytes).await {
                    return Json(serde_json::json!({ "error": format!("Failed to write temp audio: {}", e) })).into_response();
                }

                let ffmpeg_result = tokio::process::Command::new("ffmpeg")
                    .args(["-y", "-i", &temp_in, "-ar", "16000", "-ac", "1", "-f", "wav", &temp_out])
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .status()
                    .await;

                let _ = tokio::fs::remove_file(&temp_in).await;

                match ffmpeg_result {
                    Ok(status) if status.success() => {
                        match tokio::fs::read(&temp_out).await {
                            Ok(wav_bytes) => {
                                let _ = tokio::fs::remove_file(&temp_out).await;
                                match whisper.transcribe_audio(&wav_bytes).await {
                                    Ok(text) => Json(serde_json::json!({
                                        "status": "success",
                                        "transcription": text
                                    })).into_response(),
                                    Err(e) => {
                                        tracing::error!("Whisper transcription error: {}", e);
                                        Json(serde_json::json!({ "error": format!("Whisper error: {}", e) })).into_response()
                                    }
                                }
                            }
                            Err(e) => Json(serde_json::json!({ "error": format!("Failed to read converted audio: {}", e) })).into_response()
                        }
                    }
                    _ => {
                        let _ = tokio::fs::remove_file(&temp_out).await;
                        Json(serde_json::json!({ "error": "ffmpeg audio conversion failed" })).into_response()
                    }
                }
            }
            Err(e) => Json(serde_json::json!({ "error": format!("Invalid base64 audio: {}", e) })).into_response(),
        }
    } else {
        Json(serde_json::json!({ "error": "Whisper engine not initialized" })).into_response()
    }
}

#[derive(Serialize)]
pub struct GpuProcess {
    pub pid: u32,
    pub name: String,
    pub used_memory: u64,
}

#[derive(Serialize)]
pub struct GpuStat {
    pub index: u32,
    pub name: String,
    pub total: u64,
    pub used: u64,
    pub free: u64,
    pub processes: Vec<GpuProcess>,
}

#[derive(Serialize)]
pub struct GpuStatsResponse {
    pub gpus: Vec<GpuStat>,
}

/// Fetches real-time GPU VRAM statistics and active compute processes.
async fn system_gpu() -> axum::response::Response {
    use std::process::Command;
    
    // 1. Fetch overall memory limits
    let output = Command::new("nvidia-smi")
        .arg("--query-gpu=index,name,memory.total,memory.used,memory.free")
        .arg("--format=csv,noheader,nounits")
        .output();

    let mut gpus = Vec::new();

    if let Ok(out) = output {
        if out.status.success() {
            let stdout_str = String::from_utf8_lossy(&out.stdout);
            for line in stdout_str.lines() {
                let parts: Vec<&str> = line.split(", ").collect();
                if parts.len() == 5 {
                    if let (Ok(idx), Ok(tot), Ok(used), Ok(free)) = (
                        parts[0].parse::<u32>(),
                        parts[2].parse::<u64>(),
                        parts[3].parse::<u64>(),
                        parts[4].parse::<u64>(),
                    ) {
                        gpus.push(GpuStat {
                            index: idx,
                            name: parts[1].to_string(),
                            total: tot,
                            used,
                            free,
                            processes: Vec::new(),
                        });
                    }
                }
            }
        }
    }

    // 2. Map Processes to GPUs directly using nvidia-smi basic command parsing
    let bash_cmd = "nvidia-smi pmon -c 1 -s m | grep -v '#' | awk '{print $1 \",\" $2 \",\" $4 \",\" $8}'";
    if let Ok(out) = Command::new("bash").arg("-c").arg(bash_cmd).output() {
         if out.status.success() {
            let stdout_str = String::from_utf8_lossy(&out.stdout);
            for line in stdout_str.lines() {
                let parts: Vec<&str> = line.split(',').collect();
                if parts.len() == 4 {
                    if let (Ok(gpu_idx), Ok(pid), Ok(mem)) = (
                        parts[0].parse::<u32>(),
                        parts[1].parse::<u32>(),
                        parts[3].parse::<u64>(),
                    ) {
                        let proc_name = parts[2].to_string();
                        if let Some(gpu) = gpus.iter_mut().find(|g| g.index == gpu_idx) {
                            gpu.processes.push(GpuProcess {
                                pid,
                                name: proc_name,
                                used_memory: mem,
                            });
                        }
                    }
                }
            }
        }
    }

    Json(GpuStatsResponse { gpus }).into_response()
}

use hera_execution::workflow::{execute_dag, WorkflowRequest};

/// Native Rust Orchestrator Execution for React Flow DAGs
async fn workflow_execute(
    Json(payload): Json<WorkflowRequest>
) -> axum::response::Response {
    let result = execute_dag(payload).await;
    Json(result).into_response()
}

use hera_execution::dify::parse_dify_json;

/// Native Rust Orchestrator Execution for Dify exported JSON DAGs
async fn workflow_execute_dify(
    body: String
) -> axum::response::Response {
    match parse_dify_json(&body) {
        Ok(req) => {
            let result = execute_dag(req).await;
            Json(result).into_response()
        },
        Err(e) => {
            (axum::http::StatusCode::BAD_REQUEST, Json(serde_json::json!({ "error": e }))).into_response()
        }
    }
}

#[derive(serde::Deserialize)]
pub struct ScrapeRequest {
    pub url: String,
}

/// Triggers Native Web Scraping via the Hera abstraction boundary.
async fn hera_scrape(Json(payload): Json<ScrapeRequest>) -> axum::response::Response {
    let mcp_url = std::env::var("HERA_MCP_URL").unwrap_or_else(|_| "http://127.0.0.1:8080".to_string());
    let hera = hera_execution::agents::hera::Hera::new(&mcp_url);
    
    match hera.native_web_scrape(&payload.url).await {
        Ok(res) => Json(serde_json::json!({"text": res})).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("Hera Scrape Error: {}", e)).into_response(),
    }
}

#[derive(serde::Deserialize)]
pub struct SearchRequest {
    pub query: String,
}

/// Triggers Native Web Search via the Hera abstraction boundary.
async fn hera_search(Json(payload): Json<SearchRequest>) -> axum::response::Response {
    let mcp_url = std::env::var("HERA_MCP_URL").unwrap_or_else(|_| "http://127.0.0.1:8080".to_string());
    let hera = hera_execution::agents::hera::Hera::new(&mcp_url);
    
    match hera.native_web_search(&payload.query).await {
        Ok(res) => Json(serde_json::json!({"results": res})).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("Hera Search Error: {}", e)).into_response(),
    }
}

/// Serve generated images from the outputs directory
async fn serve_output_image(
    axum::extract::Path(filename): axum::extract::Path<String>,
) -> impl IntoResponse {
    // Security: only allow simple filenames (no ../ or /)
    if filename.contains("..") || filename.contains('/') || filename.contains('\\') {
        return (StatusCode::BAD_REQUEST, "Invalid filename").into_response();
    }

    let output_dir = "/home/paulo/Programs/apps/hera/playground/outputs";
    let filepath = format!("{}/{}", output_dir, filename);

    match tokio::fs::read(&filepath).await {
        Ok(data) => {
            let content_type = if filename.ends_with(".png") {
                "image/png"
            } else if filename.ends_with(".jpg") || filename.ends_with(".jpeg") {
                "image/jpeg"
            } else if filename.ends_with(".webp") {
                "image/webp"
            } else if filename.ends_with(".wav") {
                "audio/wav"
            } else if filename.ends_with(".mp4") {
                "video/mp4"
            } else {
                "application/octet-stream"
            };

            (
                StatusCode::OK,
                [(axum::http::header::CONTENT_TYPE, content_type)],
                data,
            ).into_response()
        }
        Err(_) => (StatusCode::NOT_FOUND, "File not found").into_response(),
    }
}

// --- Anthropic Connector ---

#[derive(serde::Deserialize)]
pub struct AnthropicRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    pub system: Option<String>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub stream: Option<bool>,
}

#[derive(serde::Deserialize, Clone)]
pub struct AnthropicMessage {
    pub role: String, // "user" or "assistant"
    pub content: AnthropicContent,
}

#[derive(serde::Deserialize, Clone)]
#[serde(untagged)]
pub enum AnthropicContent {
    Text(String),
    Parts(Vec<AnthropicContentPart>),
}

#[derive(serde::Deserialize, Clone)]
#[serde(tag = "type")]
pub enum AnthropicContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { source: AnthropicImageSource },
}

#[derive(serde::Deserialize, Clone)]
pub struct AnthropicImageSource {
    #[serde(rename = "type")]
    pub source_type: String, // "base64"
    pub media_type: String,
    pub data: String,
}

#[derive(serde::Serialize)]
pub struct AnthropicResponse {
    pub id: String,
    pub r#type: String,
    pub role: String,
    pub model: String,
    pub content: Vec<AnthropicResponseContent>,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
}

#[derive(serde::Serialize)]
pub struct AnthropicResponseContent {
    pub r#type: String,
    pub text: String,
}

#[derive(serde::Serialize)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

/// Anthropic API compliant endpoint (`/v1/messages`)
async fn anthropic_messages(
    State(state): State<Arc<ApiState>>,
    Json(payload): Json<AnthropicRequest>,
) -> axum::response::Response {
    let mut messages = Vec::new();

    // Map system prompt
    if let Some(sys) = payload.system {
        messages.push(crate::ai::ChatMessage {
            role: "system".to_string(),
            content: crate::ai::MessageContent::Text(sys),
        });
    }

    // Map messages
    for msg in payload.messages {
        let content = match msg.content {
            AnthropicContent::Text(text) => crate::ai::MessageContent::Text(text),
            AnthropicContent::Parts(parts) => {
                let mut mapped_parts = Vec::new();
                for part in parts {
                    match part {
                        AnthropicContentPart::Text { text } => {
                            mapped_parts.push(crate::ai::ContentPart::Text { text });
                        }
                        AnthropicContentPart::Image { source } => {
                            let data_uri = format!("data:{};base64,{}", source.media_type, source.data);
                            mapped_parts.push(crate::ai::ContentPart::ImageUrl {
                                image_url: crate::ai::ImageUrlContent { url: data_uri },
                            });
                        }
                    }
                }
                crate::ai::MessageContent::Parts(mapped_parts)
            }
        };

        messages.push(crate::ai::ChatMessage {
            role: msg.role,
            content,
        });
    }

    let req = crate::ai::ChatRequest {
        model: payload.model.clone(),
        messages,
        max_tokens: payload.max_tokens,
        temperature: payload.temperature,
        stream: payload.stream,
        vision_model: None,
        tts_model: None,
        stt_model: None,
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
        nsfw: None,
        tools: None,
        tool_choice: None,
        reasoning_effort: None,
    };

    if payload.stream.unwrap_or(false) {
        return (StatusCode::BAD_REQUEST, "Streaming not yet supported via Anthropic translation.").into_response();
    }

    match state.engine.generate_content(req).await {
        Ok(res) => {
            let text = res.choices.first()
                .and_then(|c| c.message.content.clone())
                .unwrap_or_default();
            
            let id = if res.id.is_empty() { format!("msg_{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()) } else { res.id };

            let anthropic_res = AnthropicResponse {
                id,
                r#type: "message".to_string(),
                role: "assistant".to_string(),
                model: res.model,
                content: vec![AnthropicResponseContent {
                    r#type: "text".to_string(),
                    text,
                }],
                stop_reason: Some("end_turn".to_string()),
                stop_sequence: None,
                usage: AnthropicUsage {
                    input_tokens: res.usage.as_ref().map(|u| u.prompt_tokens).unwrap_or(0),
                    output_tokens: res.usage.as_ref().map(|u| u.completion_tokens).unwrap_or(0),
                },
            };

            Json(anthropic_res).into_response()
        }
        Err(e) => {
            (StatusCode::INTERNAL_SERVER_ERROR, format!("Inference error: {}", e)).into_response()
        }
    }
}

// -----------------------------------------------------------------------------
// Universal PDF Generation Endpoint
// -----------------------------------------------------------------------------

#[derive(serde::Deserialize, Clone, Debug)]
pub struct PdfGenerationRequest {
    pub schema: serde_json::Value,
    pub template_id: Option<String>,
    pub return_base64: Option<bool>,
}

pub async fn generate_pdf(
    axum::Json(payload): axum::Json<PdfGenerationRequest>,
) -> impl axum::response::IntoResponse {
    use axum::http::header::{CONTENT_DISPOSITION, CONTENT_TYPE};
    use axum::http::StatusCode;
    
    match crate::api::pdf_assembler::generate_pdf_from_schema(&payload.schema, payload.template_id) {
        Ok(pdf_bytes) => {
            if payload.return_base64.unwrap_or(false) {
                use base64::{Engine as _, engine::general_purpose};
                let b64 = general_purpose::STANDARD.encode(&pdf_bytes);
                (
                    StatusCode::OK,
                    [(CONTENT_TYPE, "application/json")],
                    serde_json::to_string(&serde_json::json!({ "pdf_base64": b64 })).unwrap(),
                ).into_response()
            } else {
                (
                    StatusCode::OK,
                    [
                        (CONTENT_TYPE, "application/pdf"),
                        (CONTENT_DISPOSITION, "attachment; filename=\"document.pdf\""),
                    ],
                    pdf_bytes,
                ).into_response()
            }
        }
        Err(e) => {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                [(CONTENT_TYPE, "application/json")],
                serde_json::to_string(&serde_json::json!({ "error": e })).unwrap(),
            ).into_response()
        }
    }
}
