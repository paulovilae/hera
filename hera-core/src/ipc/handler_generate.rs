//! Handler: generate (non-streaming LLM inference with tool execution).

use super::context::{
    build_full_system_prompt, build_new_chat_request, compress_if_needed, inject_system_prompt,
    is_lightweight_conversation, parse_payload,
};
use super::helpers::infer_origin_from_model;
use super::llm_audit::{append_llm_audit_event, build_event};
use super::types::{HandlerOutcome, IpcPayload, IpcResponse, IpcState};
use crate::ai::{ChatMessage, ChatRequest, MessageContent};

use std::time::Instant;
use tokio::io::AsyncWriteExt;
use tokio::net::UnixStream;

/// Handle the "generate" IPC action — non-streaming LLM chat with tool execution.
pub async fn handle_generate(
    request: &IpcPayload,
    state: &IpcState,
    stream: &mut UnixStream,
) -> HandlerOutcome {
    let started_at = Instant::now();
    let mut payload_clone = request.payload.clone();
    let parsed = parse_payload(&payload_clone);
    let provider_requested = payload_clone
        .get("provider")
        .and_then(|value| value.as_str())
        .unwrap_or("auto")
        .to_string();

    // 1. Fast-path intent detection
    if !parsed.prompt.is_empty() {
        if let Some(tool_call) = crate::ai::tool_executor::detect_intent_from_user_message(
            &parsed.prompt,
            parsed.assistant_last.as_deref(),
        ) {
            if crate::ai::tool_executor::permissions_allow_tool(
                &parsed.permissions,
                &tool_call.name,
            ) {
                tracing::info!(
                    "🚀 [Hera IPC] Fast-path tool intent detected: {}",
                    tool_call.name
                );
                let tool_result = crate::ai::tool_executor::execute_tool(&tool_call).await;

                let res = IpcResponse {
                    status: "success".to_string(),
                    data: serde_json::json!({
                        "result": tool_result.output,
                        "origin": "tool",
                        "model": tool_call.name,
                        "tool_calls": [tool_call]
                    }),
                };
                let mut res_str = serde_json::to_string(&res).unwrap();
                res_str.push('\n');
                if let Err(e) = stream.write_all(res_str.as_bytes()).await {
                    tracing::error!("❌ Failed to write IPC response for fast-path tool: {}", e);
                }
                return HandlerOutcome::DirectResponse;
            } else {
                tracing::info!(
                    "⚠️ [Hera IPC] Fast-path tool intent {} denied by permissions",
                    tool_call.name
                );
            }
        }
    }

    // 2. Ensure model is set
    if let Some(obj) = payload_clone.as_object_mut() {
        if !obj.contains_key("model") {
            obj.insert("model".to_string(), serde_json::json!("hera-local-model"));
        }
    }

    let prompt = payload_clone
        .get("prompt")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    // 3. Build or augment ChatRequest
    let mut chat_req: Option<ChatRequest> = serde_json::from_value(payload_clone.clone()).ok();

    let lightweight_mode = is_lightweight_conversation(&parsed.prompt);
    let full_system_prompt = build_full_system_prompt(
        &parsed.persona_path,
        &parsed.app_name,
        &parsed.permissions,
        lightweight_mode,
    )
    .await;

    if chat_req.is_none() {
        if !prompt.is_empty() {
            chat_req = Some(build_new_chat_request(&prompt, full_system_prompt));
        }
    } else if let Some(req) = &mut chat_req {
        inject_system_prompt(req, full_system_prompt);
    }

    // 4. Context compression
    if let Some(req) = &mut chat_req {
        compress_if_needed(req, &state.engine).await;
    }

    // 5. Generate
    if let Some(req) = chat_req.clone() {
        let est_tokens = super::helpers::estimate_tokens(&req);
        tracing::info!(
            "📡 [Hera Generate] Starting inference for app='{}' — {} msgs, ~{} tokens (lightweight_mode={})",
            parsed.app_name,
            req.messages.len(),
            est_tokens,
            lightweight_mode
        );
        match state.engine.generate_content(req).await {
            Ok(resp) => {
                let mut response_model = resp.model.clone();
                let mut response_origin = infer_origin_from_model(&resp.model).to_string();
                let origin_emoji = match response_origin.as_str() {
                    "cloud" => "☁️",
                    "local" => "🏠",
                    _ => "❓",
                };
                tracing::info!(
                    "{} [Hera Generate] Response from {} engine — model: {}",
                    origin_emoji,
                    response_origin,
                    response_model
                );
                let mut result_text = String::new();
                let mut tool_calls: Option<serde_json::Value> = None;

                if let Some(choice) = resp.choices.first() {
                    if let Some(content) = &choice.message.content {
                        result_text = content.clone();

                        // 6. Parse and execute output tool calls
                        let parsed_calls = crate::ai::tool_executor::parse_tool_calls(&result_text);
                        if !parsed_calls.is_empty() {
                            tracing::info!(
                                "🛠️ [Hera IPC] LLM emitted {} tool calls",
                                parsed_calls.len()
                            );
                            let mut execution_outputs = String::new();
                            let mut executed_calls = Vec::new();

                            for call in &parsed_calls {
                                if crate::ai::tool_executor::permissions_allow_tool(
                                    &parsed.permissions,
                                    &call.name,
                                ) {
                                    let tool_res =
                                        crate::ai::tool_executor::execute_tool(call).await;
                                    execution_outputs.push_str(&format!("\n\n{}", tool_res.output));
                                    executed_calls.push(serde_json::json!({
                                        "name": call.name,
                                        "arguments": call.arguments
                                    }));
                                } else {
                                    tracing::warn!(
                                        "⚠️ [Hera IPC] LLM hallucinated tool {} which is denied by permissions",
                                        call.name
                                    );
                                    execution_outputs.push_str(&format!(
                                        "\n\nError: Not permitted to use tool '{}'",
                                        call.name
                                    ));
                                }
                            }

                            let has_media_call = parsed_calls.iter().any(|c| {
                                c.name == "hera_draw"
                                    || c.name == "hera_video"
                                    || c.name == "generate_qr_code"
                            });

                            if !has_media_call {
                                if let Some(mut req2) = chat_req.clone() {
                                    // Strip tool schemas to prevent recursive tool calls
                                    if let Some(first) = req2.messages.first_mut() {
                                        if first.role == "system" {
                                            first.content = MessageContent::Text(
                                                "You are a helpful AI assistant. You have already executed tools and received the results. Your ONLY job now is to summarize the results for the user. DO NOT output any tool calls, <tool_call> tags, or function calls. DO NOT use <think> tags. Output ONLY the final answer.".to_string(),
                                            );
                                        }
                                    }
                                    req2.messages.push(ChatMessage {
                                        role: "assistant".to_string(),
                                        content: MessageContent::Text(result_text.clone()),
                                    });
                                    let json_mode = payload_clone
                                        .get("json_mode")
                                        .and_then(|v| v.as_bool())
                                        .unwrap_or(false);
                                    let sys_msg = if json_mode {
                                        format!(
                                            "Tool Execution Results: {}\n\nIMPORTANT: DO NOT call any more tools. DO NOT output <tool_call> tags. Provide your final response as RAW VALID JSON matching the exact schema requested in the original prompt. The JSON MUST contain a \"summary\" key with a human-readable response.",
                                            execution_outputs
                                        )
                                    } else {
                                        format!(
                                            "Tool Execution Results: {}\n\nIMPORTANT: DO NOT call any more tools. DO NOT output <tool_call> tags. Provide a friendly, conversational, and concise response to the user based on these results. Do not output raw JSON or mention the database tables directly.",
                                            execution_outputs
                                        )
                                    };
                                    req2.messages.push(ChatMessage {
                                        role: "user".to_string(),
                                        content: MessageContent::Text(sys_msg),
                                    });
                                    tracing::info!(
                                        "🔄 [Hera IPC] Initiating second-pass generation to format Tool Results (json_mode: {})...",
                                        json_mode
                                    );
                                    match state.engine.generate_content(req2).await {
                                        Ok(resp2) => {
                                            let p2_origin = infer_origin_from_model(&resp2.model);
                                            tracing::info!(
                                                "🔄 [Hera Generate] Second-pass response from {} — model: {}",
                                                p2_origin,
                                                resp2.model
                                            );
                                            response_model = resp2.model.clone();
                                            response_origin = p2_origin.to_string();
                                            if let Some(ch) = resp2.choices.first() {
                                                if let Some(c) = &ch.message.content {
                                                    result_text = c.clone();
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            tracing::error!("Second pass inference failed: {}", e);
                                            result_text.push_str(&format!(
                                                "\n\n[Error forming final response: {}]\n{}",
                                                e, execution_outputs
                                            ));
                                        }
                                    }
                                }
                            } else {
                                result_text.push_str(&execution_outputs);
                            }

                            tool_calls = Some(serde_json::Value::Array(executed_calls));
                        }
                    }
                    if let Some(tc) = &choice.message.tool_calls {
                        if tool_calls.is_none() {
                            tool_calls = Some(serde_json::json!(tc));
                        }
                    }
                }

                append_llm_audit_event(&build_event(
                    "generate",
                    &parsed.app_name,
                    &parsed.persona_path,
                    &parsed.prompt,
                    est_tokens,
                    started_at.elapsed().as_millis() as u64,
                    None,
                    lightweight_mode,
                    &provider_requested,
                    &response_origin,
                    &response_model,
                    true,
                    tool_calls
                        .as_ref()
                        .and_then(|value| value.as_array().map(|items| items.len()))
                        .unwrap_or(0),
                    result_text.len(),
                    None,
                ));

                return HandlerOutcome::Result {
                    result_text,
                    origin: response_origin,
                    model: response_model,
                    tool_calls,
                };
            }
            Err(e) => {
                tracing::error!("LLM inference error: {}", e);
                append_llm_audit_event(&build_event(
                    "generate",
                    &parsed.app_name,
                    &parsed.persona_path,
                    &parsed.prompt,
                    est_tokens,
                    started_at.elapsed().as_millis() as u64,
                    None,
                    lightweight_mode,
                    &provider_requested,
                    "offline",
                    "",
                    false,
                    0,
                    0,
                    Some(e.to_string()),
                ));
                return HandlerOutcome::Result {
                    result_text: format!("Error: {}", e),
                    origin: "offline".to_string(),
                    model: String::new(),
                    tool_calls: None,
                };
            }
        }
    }

    HandlerOutcome::Result {
        result_text: "Action not supported".to_string(),
        origin: "unknown".to_string(),
        model: String::new(),
        tool_calls: None,
    }
}
