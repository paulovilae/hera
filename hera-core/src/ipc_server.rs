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

fn infer_origin_from_model(model: &str) -> &'static str {
    let normalized = model.trim().to_lowercase();
    let openrouter_default = std::env::var("OPENROUTER_DEFAULT_MODEL")
        .unwrap_or_default()
        .trim()
        .to_lowercase();

    if !openrouter_default.is_empty() && normalized == openrouter_default {
        "cloud"
    } else if normalized.is_empty() {
        "unknown"
    } else {
        "local"
    }
}

async fn fetch_semantic_memory(app_name: &str) -> String {
    if app_name.is_empty() { return String::new(); }
    
    if let Ok(Ok(mut stream)) = tokio::time::timeout(std::time::Duration::from_millis(1000), tokio::net::UnixStream::connect("/tmp/memento.sock")).await {
        let msg = serde_json::json!({
            "action": "query_app",
            "payload": { "app": app_name, "query": "semantic_context" }
        });
        if stream.write_all(msg.to_string().as_bytes()).await.is_ok() {
            let mut buffer = vec![0u8; 65536];
            if let Ok(Ok(n)) = tokio::time::timeout(std::time::Duration::from_millis(1500), stream.read(&mut buffer)).await {
                if n > 0 {
                    let raw = String::from_utf8_lossy(&buffer[..n]);
                    if let Ok(resp) = serde_json::from_str::<serde_json::Value>(&raw) {
                        if let Some(ctx) = resp.get("context").and_then(|c| c.as_str()) {
                            if !ctx.is_empty() {
                                return format!("\n\n[MEMENTO SEMANTIC CORTEX INJECTION: {}]\n{}\n", app_name, ctx);
                            }
                        }
                    }
                }
            }
        }
    }
    String::new()
}

/// Fetch live DB schema from Memento for injection into tool context.
/// - SOUL (Ava) gets ALL app schemas via describe_all_apps
/// - App-specific agents get only their app's schema via describe_app
async fn fetch_db_schema_context(agent_identity: &str, app_name: &str) -> String {
    // Map agent identities to their app slugs
    let app_slug = match agent_identity.to_lowercase().as_str() {
        "soul" | "ava" | "gemini_soul" => {
            // Superuser: fetch all
            return fetch_all_apps_schema().await;
        },
        "vetra" | "vetra_soul" => "vetra",
        "movilo" | "movilo_soul" => "movilo",
        "latinos" | "latinos_soul" => "latinos",
        "garcero" | "garcero_soul" => "garcero",
        _ => {
            // Unknown agent: try using app_name from payload
            if !app_name.is_empty() {
                app_name
            } else {
                return String::new();
            }
        }
    };

    fetch_single_app_schema(app_slug).await
}

async fn fetch_single_app_schema(app_slug: &str) -> String {
    if let Ok(Ok(mut stream)) = tokio::time::timeout(
        std::time::Duration::from_millis(2000),
        tokio::net::UnixStream::connect("/tmp/memento.sock"),
    ).await {
        let msg = serde_json::json!({
            "action": "describe_app",
            "payload": { "app": app_slug }
        });
        if stream.write_all(msg.to_string().as_bytes()).await.is_ok() {
            let mut buffer = vec![0u8; 131072]; // 128KB for large schemas
            if let Ok(Ok(n)) = tokio::time::timeout(
                std::time::Duration::from_millis(3000),
                stream.read(&mut buffer),
            ).await {
                if n > 0 {
                    let raw = String::from_utf8_lossy(&buffer[..n]);
                    if let Ok(resp) = serde_json::from_str::<serde_json::Value>(&raw) {
                        if resp.get("status").and_then(|s| s.as_str()) == Some("success") {
                            if let Some(schema) = resp.get("schema").and_then(|s| s.as_object()) {
                                return format_schema_for_prompt(app_slug, schema);
                            }
                        }
                    }
                }
            }
        }
    }
    String::new()
}

async fn fetch_all_apps_schema() -> String {
    if let Ok(Ok(mut stream)) = tokio::time::timeout(
        std::time::Duration::from_millis(3000),
        tokio::net::UnixStream::connect("/tmp/memento.sock"),
    ).await {
        let msg = serde_json::json!({
            "action": "describe_all_apps",
            "payload": {}
        });
        if stream.write_all(msg.to_string().as_bytes()).await.is_ok() {
            let mut buffer = vec![0u8; 262144]; // 256KB
            if let Ok(Ok(n)) = tokio::time::timeout(
                std::time::Duration::from_millis(5000),
                stream.read(&mut buffer),
            ).await {
                if n > 0 {
                    let raw = String::from_utf8_lossy(&buffer[..n]);
                    if let Ok(resp) = serde_json::from_str::<serde_json::Value>(&raw) {
                        if let Some(apps) = resp.get("apps").and_then(|a| a.as_object()) {
                            let mut output = String::from("\n\n# DATABASE SCHEMA (Auto-Discovered)\n");
                            for (slug, tables_val) in apps {
                                if let Some(tables) = tables_val.as_object() {
                                    output.push_str(&format!("\n## App: {}\n", slug));
                                    for (table, cols) in tables {
                                        let col_names: Vec<String> = cols.as_array()
                                            .map(|arr| arr.iter().filter_map(|c| {
                                                c.get("column").and_then(|n| n.as_str()).map(|s| s.to_string())
                                            }).collect())
                                            .unwrap_or_default();
                                        output.push_str(&format!("- {} ({})\n", table, col_names.join(", ")));
                                    }
                                }
                            }
                            return output;
                        }
                    }
                }
            }
        }
    }
    String::new()
}

fn format_schema_for_prompt(app_slug: &str, schema: &serde_json::Map<String, serde_json::Value>) -> String {
    let mut output = format!("\n\n# DATABASE SCHEMA for '{}' (Auto-Discovered)\nUse these EXACT table and column names when writing SQL queries with memento_query.\n", app_slug);
    for (table, cols) in schema {
        let col_names: Vec<String> = cols.as_array()
            .map(|arr| arr.iter().filter_map(|c| {
                c.get("column").and_then(|n| n.as_str()).map(|s| s.to_string())
            }).collect())
            .unwrap_or_default();
        output.push_str(&format!("- {} ({})\n", table, col_names.join(", ")));
    }
    output
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
                                match serde_json::from_slice::<IpcPayload>(&buffer) {
                                    Ok(request) => {
                                        info!("📥 Received IPC Action: {}", request.action);
                                        
                                        // Process Request
                                        let mut result_text = "Action not supported".to_string();
                                        let mut response_origin = "unknown".to_string();
                                        let mut response_model = String::new();
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
                                                response_origin = "tool".to_string();
                                                response_model = tool_call.name.clone();
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
                                            
                                        let mut chat_req: Option<ChatRequest> = serde_json::from_value(payload_clone.clone()).ok();
                                        
                                        if chat_req.is_none() {
                                            if !prompt.is_empty() {
                                                let app_name = payload_clone.get("app").and_then(|v| v.as_str()).unwrap_or("").to_string();
                                                let memento_ctx = fetch_semantic_memory(&app_name).await;
                                                let base_system_prompt = format!("{}{}", std::fs::read_to_string(&persona_path).unwrap_or_else(|_| "You are an AI assistant.".to_string()), memento_ctx);
                                                let agent_identity = std::path::Path::new(&persona_path).file_stem().and_then(|s| s.to_str()).unwrap_or("unknown");
                                                let schemas = crate::ai::tool_executor::hera_tool_schemas(&permissions, agent_identity);
                                                let db_schema_ctx = fetch_db_schema_context(agent_identity, &app_name).await;
                                                let think_directive = "\n\nCRITICAL INSTRUCTION (INFERENCE-TIME RECALL): Before providing your final answer, you MUST systematically write out your internal reasoning step-by-step within <think> and </think> tags. Use this space to explore associations, reverse the question context, and search your internal knowledge to maximize factual recall. Do not output the final answer until after the </think> tag.";
                                                let json_directive = "\nCRITICAL TOOL RULE: If you decide to execute a tool, your ENTIRE response MUST be ONLY the raw JSON tool call. DO NOT write conversational text or explanations before or after the JSON tool block. The UI stream will crash if text and code logic bleed together.";
                                                let full_system_prompt = format!("{}\n\nCRITICAL RULE: DO NOT use tools to answer general conversational or conceptual questions like 'explain X' or 'what is Y'. If the user asks for an explanation or text-based answer, DO NOT build scripts or charts unless explicitly asked. ONLY use tools when the user explicitly requests code execution, file reading, or specific outputs.\n\n{}{}{}{}", base_system_prompt, schemas, db_schema_ctx, think_directive, json_directive);

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
                                                });
                                            }
                                        } else if let Some(req) = &mut chat_req {
                                            // Inject base persona + tool schemas into existing request
                                            let app_name = payload_clone.get("app").and_then(|v| v.as_str()).unwrap_or("").to_string();
                                            let memento_ctx = fetch_semantic_memory(&app_name).await;
                                            let base_system_prompt = format!("{}{}", std::fs::read_to_string(&persona_path).unwrap_or_else(|_| "You are an AI assistant.".to_string()), memento_ctx);
                                            let agent_identity = std::path::Path::new(&persona_path).file_stem().and_then(|s| s.to_str()).unwrap_or("unknown");
                                            let schemas = crate::ai::tool_executor::hera_tool_schemas(&permissions, agent_identity);
                                            let db_schema_ctx = fetch_db_schema_context(agent_identity, &app_name).await;
                                            let think_directive = "\n\nCRITICAL INSTRUCTION (INFERENCE-TIME RECALL): Before providing your final answer, you MUST systematically write out your internal reasoning step-by-step within <think> and </think> tags. Use this space to explore associations, reverse the question context, and search your internal knowledge to maximize factual recall. Do not output the final answer until after the </think> tag.";
                                            let json_directive = "\nCRITICAL TOOL RULE: If you decide to execute a tool, your ENTIRE response MUST be ONLY the raw JSON tool call. DO NOT write conversational text or explanations before or after the JSON tool block. The UI stream will crash if text and code logic bleed together.";
                                            let full_system_prompt = format!("{}\n\n{}{}{}{}", base_system_prompt, schemas, db_schema_ctx, think_directive, json_directive);

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
                                                    response_model = resp.model.clone();
                                                    response_origin = infer_origin_from_model(&resp.model).to_string();
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
                                                                        // Replace the system prompt to remove tool schemas — prevents recursive tool calls
                                                                        if let Some(first) = req2.messages.first_mut() {
                                                                            if first.role == "system" {
                                                                                first.content = MessageContent::Text("You are a helpful AI assistant. You have already executed tools and received the results. Your ONLY job now is to summarize the results for the user. DO NOT output any tool calls, <tool_call> tags, or function calls. DO NOT use <think> tags. Output ONLY the final answer.".to_string());
                                                                            }
                                                                        }
                                                                        req2.messages.push(ChatMessage {
                                                                            role: "assistant".to_string(),
                                                                            content: MessageContent::Text(result_text.clone()),
                                                                        });
                                                                        let json_mode = payload_clone.get("json_mode").and_then(|v| v.as_bool()).unwrap_or(false);
                                                                        let sys_msg = if json_mode {
                                                                            format!("Tool Execution Results: {}\n\nIMPORTANT: DO NOT call any more tools. DO NOT output <tool_call> tags. Provide your final response as RAW VALID JSON matching the exact schema requested in the original prompt. The JSON MUST contain a \"summary\" key with a human-readable response.", execution_outputs)
                                                                        } else {
                                                                            format!("Tool Execution Results: {}\n\nIMPORTANT: DO NOT call any more tools. DO NOT output <tool_call> tags. Provide a friendly, conversational, and concise response to the user based on these results. Do not output raw JSON or mention the database tables directly.", execution_outputs)
                                                                        };
                                                                        req2.messages.push(ChatMessage {
                                                                            role: "user".to_string(),
                                                                            content: MessageContent::Text(sys_msg),
                                                                        });
                                                                        tracing::info!("🔄 [Hera IPC] Initiating second-pass generation to format Tool Results (json_mode: {})...", json_mode);
                                                                        match state.engine.generate_content(req2).await {
                                                                            Ok(resp2) => {
                                                                                response_model = resp2.model.clone();
                                                                                response_origin = infer_origin_from_model(&resp2.model).to_string();
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
                                                    response_origin = "offline".to_string();
                                                    result_text = format!("Error: {}", e);
                                                }
                                            }
                                        }
                                    }
                                } else if request.action == "generate_stream" {
                                    let mut payload_clone = request.payload.clone();
                                    
                                    let mut prompt = payload_clone.get("prompt").and_then(|p| p.as_str()).unwrap_or("").to_string();
                                    let mut assistant_last: Option<String> = None;
                                    
                                    if prompt.is_empty() {
                                        if let Some(messages) = payload_clone.get("messages").and_then(|m| m.as_array()) {
                                            if let Some(last_msg) = messages.last() {
                                                if let Some("user") = last_msg.get("role").and_then(|r| r.as_str()) {
                                                    if let Some(content) = last_msg.get("content").and_then(|c| c.as_str()) {
                                                        prompt = content.to_string();
                                                    }
                                                }
                                            }
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
                                    
                                    let permissions: Vec<String> = payload_clone.get("permissions")
                                        .and_then(|v| v.as_array())
                                        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect::<Vec<String>>())
                                        .unwrap_or_else(|| vec!["all".to_string()]);
                                        
                                    tracing::info!("🛡️ [Hera IPC Stream] Parsed permissions: {:?}", permissions);
                                    
                                    // Fast-path intent detection
                                    if !prompt.is_empty() {
                                        if let Some(tool_call) = crate::ai::tool_executor::detect_intent_from_user_message(&prompt, assistant_last.as_deref()) {
                                            if permissions.contains(&"all".to_string()) || permissions.contains(&tool_call.name) {
                                                tracing::info!("🚀 [Hera IPC Stream] Fast-path tool intent detected: {}", tool_call.name);
                                                
                                                let status_msg = IpcResponse { status: "tool_status".to_string(), data: serde_json::json!({"name": tool_call.name.clone()}) };
                                                let mut str_msg = serde_json::to_string(&status_msg).unwrap();
                                                str_msg.push('\n');
                                                let _ = stream.write_all(str_msg.as_bytes()).await;

                                                let tool_result = crate::ai::tool_executor::execute_tool(&tool_call).await;
                                                
                                                let chunk_msg = IpcResponse { status: "chunk".to_string(), data: serde_json::json!({"text": tool_result.output}) };
                                                let mut cstr = serde_json::to_string(&chunk_msg).unwrap();
                                                cstr.push('\n');
                                                let _ = stream.write_all(cstr.as_bytes()).await;

                                                let done_msg = IpcResponse { status: "done".to_string(), data: serde_json::json!({}) };
                                                let mut dstr = serde_json::to_string(&done_msg).unwrap();
                                                dstr.push('\n');
                                                let _ = stream.write_all(dstr.as_bytes()).await;
                                                break;
                                            }
                                        }
                                    }
                                    
                                    if let Some(obj) = payload_clone.as_object_mut() {
                                        if !obj.contains_key("model") {
                                            obj.insert("model".to_string(), serde_json::json!("hera-local-model"));
                                        }
                                    }
                                    
                                    let prompt = payload_clone.get("prompt").and_then(|v| v.as_str()).unwrap_or("").to_string();
                                    let persona_path = payload_clone.get("persona_path").and_then(|v| v.as_str()).unwrap_or("/home/paulo/Programs/apps/imaginos/imaginclaw/persona/SOUL.md").to_string();
                                        
                                    let mut chat_req: Option<crate::ai::ChatRequest> = serde_json::from_value(payload_clone.clone()).ok();
                                    
                                    if chat_req.is_none() {
                                        if !prompt.is_empty() {
                                            let app_name = payload_clone.get("app").and_then(|v| v.as_str()).unwrap_or("").to_string();
                                            let memento_ctx = fetch_semantic_memory(&app_name).await;
                                            let base_system_prompt = format!("{}{}", std::fs::read_to_string(&persona_path).unwrap_or_else(|_| "You are an AI assistant.".to_string()), memento_ctx);
                                            let agent_identity = std::path::Path::new(&persona_path).file_stem().and_then(|s| s.to_str()).unwrap_or("unknown");
                                            let schemas = crate::ai::tool_executor::hera_tool_schemas(&permissions, agent_identity);
                                            let db_schema_ctx = fetch_db_schema_context(agent_identity, &app_name).await;
                                            let think_directive = "\n\nCRITICAL INSTRUCTION (INFERENCE-TIME RECALL): Before providing your final answer, you MUST systematically write out your internal reasoning step-by-step within <think> and </think> tags. Use this space to explore associations, reverse the question context, and search your internal knowledge to maximize factual recall. Do not output the final answer until after the </think> tag.";
                                            let json_directive = "\nCRITICAL TOOL RULE: If you decide to execute a tool, your ENTIRE response MUST be ONLY the raw JSON tool call. DO NOT write conversational text or explanations before or after the JSON tool block. The UI stream will crash if text and code logic bleed together.";
                                            let full_system_prompt = format!("{}\n\nCRITICAL RULE: DO NOT use tools to answer general conversational or conceptual questions like 'explain X' or 'what is Y'. If the user asks for an explanation or text-based answer, DO NOT build scripts or charts unless explicitly asked. ONLY use tools when the user explicitly requests code execution, file reading, or specific outputs.\n\n{}{}{}{}", base_system_prompt, schemas, db_schema_ctx, think_directive, json_directive);

                                            chat_req = Some(crate::ai::ChatRequest {
                                                model: "hera-local-model".to_string(),
                                                vision_model: None,
                                                tts_model: None,
                                                stt_model: None,
                                                messages: vec![
                                                    crate::ai::ChatMessage {
                                                        role: "system".to_string(),
                                                        content: crate::ai::MessageContent::Text(full_system_prompt),
                                                    },
                                                    crate::ai::ChatMessage {
                                                        role: "user".to_string(),
                                                        content: crate::ai::MessageContent::Text(prompt.clone()),
                                                    }
                                                ],
                                                temperature: Some(0.7),
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
                                            });
                                        }
                                    } else if let Some(req) = &mut chat_req {
                                        let app_name = payload_clone.get("app").and_then(|v| v.as_str()).unwrap_or("").to_string();
                                        let memento_ctx = fetch_semantic_memory(&app_name).await;
                                        let base_system_prompt = format!("{}{}", std::fs::read_to_string(&persona_path).unwrap_or_else(|_| "You are an expert AI system running within the Sovereign OS (locally on the user's hardware). You have access to powerful tools. CRITICAL RULE: DO NOT use tools to answer general conversational or conceptual questions like 'explain X' or 'what is Y'. If the user asks for an explanation or text-based answer, DO NOT build scripts or charts unless explicitly asked. ONLY use tools when the user explicitly requests code execution, file reading, or specific outputs.".to_string()), memento_ctx);
                                        let agent_identity = std::path::Path::new(&persona_path).file_stem().and_then(|s| s.to_str()).unwrap_or("unknown");
                                        let schemas = crate::ai::tool_executor::hera_tool_schemas(&permissions, agent_identity);
                                        let db_schema_ctx = fetch_db_schema_context(agent_identity, &app_name).await;
                                        let think_directive = "\n\nCRITICAL INSTRUCTION (INFERENCE-TIME RECALL): Before providing your final answer, you MUST systematically write out your internal reasoning step-by-step within <think> and </think> tags. Use this space to explore associations, reverse the question context, and search your internal knowledge to maximize factual recall. Do not output the final answer until after the </think> tag.";
                                        let json_directive = "\nCRITICAL TOOL RULE: If you decide to execute a tool, your ENTIRE response MUST be ONLY the raw JSON tool call. DO NOT write conversational text or explanations before or after the JSON tool block. The UI stream will crash if text and code logic bleed together.";
                                        let full_system_prompt = format!("{}\n\n{}{}{}{}", base_system_prompt, schemas, db_schema_ctx, think_directive, json_directive);

                                        if let Some(first) = req.messages.first_mut() {
                                            if first.role == "system" {
                                                match &mut first.content {
                                                    crate::ai::MessageContent::Text(t) => { *t = format!("{}\n\n{}", full_system_prompt, t); }
                                                    crate::ai::MessageContent::Parts(parts) => { parts.insert(0, crate::ai::ContentPart::Text { text: format!("{}\n\n", full_system_prompt) }); }
                                                    crate::ai::MessageContent::Null => { first.content = crate::ai::MessageContent::Text(full_system_prompt); }
                                                }
                                            } else {
                                                req.messages.insert(0, crate::ai::ChatMessage {
                                                    role: "system".to_string(),
                                                    content: crate::ai::MessageContent::Text(full_system_prompt),
                                                });
                                            }
                                        } else {
                                            req.messages.push(crate::ai::ChatMessage {
                                                role: "system".to_string(),
                                                content: crate::ai::MessageContent::Text(full_system_prompt),
                                            });
                                        }
                                    }
                                    
                                    if let Some(req) = chat_req.clone() {
                                        let start_msg = IpcResponse { status: "stream_start".to_string(), data: serde_json::json!({}) };
                                        let mut res_str = serde_json::to_string(&start_msg).unwrap();
                                        res_str.push('\n');
                                        let _ = stream.write_all(res_str.as_bytes()).await;

                                        let mut final_result_text = String::new();
                                        let mut buffer_flushed = false;
                                        let mut is_tool_call_mode = false;
                                        match state.engine.generate_stream(req).await {
                                            Ok(mut rx) => {
                                                while let Some(chunk_res) = rx.recv().await {
                                                    if let Ok(chunk) = chunk_res {
                                                        let chunk_text = chunk.choices.first().and_then(|c| c.delta.content.clone()).unwrap_or_default();
                                                        if chunk_text.is_empty() { continue; }
                                                        
                                                        final_result_text.push_str(&chunk_text);
                                                        
                                                        if !buffer_flushed {
                                                            let trimmed = final_result_text.trim_start();
                                                            // Check for actual tool call patterns, NOT <think> tags
                                                            let looks_like_tool = trimmed.starts_with('{') ||
                                                                trimmed.starts_with("<tool_call>") ||
                                                                trimmed.starts_with("<function-call>") ||
                                                                trimmed.starts_with("<function_call>") ||
                                                                trimmed.starts_with("<function=");
                                                            if looks_like_tool {
                                                                // Looks like a tool call, suppress streaming
                                                                is_tool_call_mode = true;
                                                            } else if final_result_text.len() > 5 {
                                                                // Definitely normal text, flush buffer and start streaming
                                                                is_tool_call_mode = false;
                                                                buffer_flushed = true;
                                                                let chunk_msg = IpcResponse { status: "chunk".to_string(), data: serde_json::json!({"text": final_result_text}) };
                                                                let mut cstr = serde_json::to_string(&chunk_msg).unwrap();
                                                                cstr.push('\n');
                                                                let _ = stream.write_all(cstr.as_bytes()).await;
                                                            }
                                                        } else if !is_tool_call_mode {
                                                            // We're in normal streaming mode, send just this chunk
                                                            let chunk_msg = IpcResponse { status: "chunk".to_string(), data: serde_json::json!({"text": chunk_text}) };
                                                            let mut cstr = serde_json::to_string(&chunk_msg).unwrap();
                                                            cstr.push('\n');
                                                            let _ = stream.write_all(cstr.as_bytes()).await;
                                                        }
                                                    }
                                                }
                                                
                                                if !buffer_flushed && !is_tool_call_mode && !final_result_text.is_empty() {
                                                    // Stream finished but was very short, flush it now
                                                    let chunk_msg = IpcResponse { status: "chunk".to_string(), data: serde_json::json!({"text": final_result_text}) };
                                                    let mut cstr = serde_json::to_string(&chunk_msg).unwrap();
                                                    cstr.push('\n');
                                                    let _ = stream.write_all(cstr.as_bytes()).await;
                                                }

                                                let parsed_calls = crate::ai::tool_executor::parse_tool_calls(&final_result_text);
                                                if !parsed_calls.is_empty() {
                                                    let mut execution_outputs = String::new();
                                                    for call in &parsed_calls {
                                                        let status_msg = IpcResponse { status: "tool_status".to_string(), data: serde_json::json!({"name": call.name.clone()}) };
                                                        let mut str_msg = serde_json::to_string(&status_msg).unwrap();
                                                        str_msg.push('\n');
                                                        let _ = stream.write_all(str_msg.as_bytes()).await;

                                                        if permissions.contains(&"all".to_string()) || permissions.contains(&call.name) {
                                                            let tool_res = crate::ai::tool_executor::execute_tool(&call).await;
                                                            execution_outputs.push_str(&format!("\n\n{}", tool_res.output));
                                                        } else {
                                                            execution_outputs.push_str(&format!("\n\nError: Not permitted to use tool '{}'", call.name));
                                                        }
                                                    }

                                                    let has_media_call = parsed_calls.iter().any(|c| c.name == "hera_draw" || c.name == "hera_video" || c.name == "generate_qr_code");
                                                    if !has_media_call {
                                                        if let Some(mut req2) = chat_req.clone() {
                                                            // Replace system prompt to remove tool schemas — prevents recursive tool calls
                                                            if let Some(first) = req2.messages.first_mut() {
                                                                if first.role == "system" {
                                                                    first.content = crate::ai::MessageContent::Text("You are a helpful AI assistant. You have already executed tools and received the results. Your ONLY job now is to summarize the results for the user. DO NOT output any tool calls, <tool_call> tags, or function calls. DO NOT use <think> tags. Output ONLY the final answer.".to_string());
                                                                }
                                                            }
                                                            req2.messages.push(crate::ai::ChatMessage {
                                                                role: "assistant".to_string(),
                                                                content: crate::ai::MessageContent::Text(final_result_text.clone()),
                                                            });
                                                            let json_mode = payload_clone.get("json_mode").and_then(|v| v.as_bool()).unwrap_or(false);
                                                            let sys_msg = if json_mode {
                                                                format!("Tool Execution Results: {}\n\nIMPORTANT: DO NOT call any more tools. DO NOT output <tool_call> tags. Provide your final response as RAW VALID JSON matching the exact schema requested in the original prompt. The JSON MUST contain a \"summary\" key with a human-readable response.", execution_outputs)
                                                            } else {
                                                                format!("Tool Execution Results: {}\n\nIMPORTANT: DO NOT call any more tools. DO NOT output <tool_call> tags. Provide a friendly, conversational, and concise response to the user based on these results. Do not output raw JSON or mention the database tables directly.", execution_outputs)
                                                            };
                                                            req2.messages.push(crate::ai::ChatMessage {
                                                                role: "user".to_string(),
                                                                content: crate::ai::MessageContent::Text(sys_msg),
                                                            });
                                                            if let Ok(mut rx2) = state.engine.generate_stream(req2).await {
                                                                while let Some(chunk_res2) = rx2.recv().await {
                                                                    if let Ok(chunk2) = chunk_res2 {
                                                                        let chunk_text = chunk2.choices.first().and_then(|c| c.delta.content.clone()).unwrap_or_default();
                                                                        let chunk_msg = IpcResponse { status: "chunk".to_string(), data: serde_json::json!({"text": chunk_text}) };
                                                                        let mut cstr = serde_json::to_string(&chunk_msg).unwrap();
                                                                        cstr.push('\n');
                                                                        let _ = stream.write_all(cstr.as_bytes()).await;
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    } else {
                                                        let chunk_msg = IpcResponse { status: "chunk".to_string(), data: serde_json::json!({"text": execution_outputs}) };
                                                        let mut cstr = serde_json::to_string(&chunk_msg).unwrap();
                                                        cstr.push('\n');
                                                        let _ = stream.write_all(cstr.as_bytes()).await;
                                                    }
                                                } else if is_tool_call_mode && !final_result_text.is_empty() {
                                                    // Suppressed the stream assuming it was a tool, but it wasn't a valid tool call.
                                                    // Let's dump the entire buffered text to the frontend so they don't get a silent blank.
                                                    let chunk_msg = IpcResponse { status: "chunk".to_string(), data: serde_json::json!({"text": final_result_text}) };
                                                    let mut cstr = serde_json::to_string(&chunk_msg).unwrap();
                                                    cstr.push('\n');
                                                    let _ = stream.write_all(cstr.as_bytes()).await;
                                                }

                                                let done_msg = IpcResponse { status: "done".to_string(), data: serde_json::json!({}) };
                                                let mut dstr = serde_json::to_string(&done_msg).unwrap();
                                                dstr.push('\n');
                                                let _ = stream.write_all(dstr.as_bytes()).await;
                                                break;
                                            }
                                            Err(e) => {
                                                let err_msg = IpcResponse { status: "error".to_string(), data: serde_json::json!({"error": e.to_string()}) };
                                                let mut estr = serde_json::to_string(&err_msg).unwrap();
                                                estr.push('\n');
                                                let _ = stream.write_all(estr.as_bytes()).await;
                                                break;
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
                                                max_tokens: Some(4096),
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
                                if let Some(map) = data_json.as_object_mut() {
                                    map.insert("origin".to_string(), serde_json::json!(response_origin));
                                    map.insert("model".to_string(), serde_json::json!(response_model));
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
                                    Err(e) => {
                                        error!("❌ IPC JSON Parse Error: {} - Buffer: {}", e, String::from_utf8_lossy(&buffer));
                                        
                                        // Send error back to client to avoid hanging
                                        let err_msg = IpcResponse { status: "error".to_string(), data: serde_json::json!({"error": format!("Invalid JSON: {}", e)}) };
                                        let mut estr = serde_json::to_string(&err_msg).unwrap();
                                        estr.push('\n');
                                        let _ = stream.write_all(estr.as_bytes()).await;
                                        break;
                                    }
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
