//! Handler: generate_stream (streaming LLM inference with tool execution).

use crate::ai::{ChatMessage, ChatRequest, MessageContent};
use super::context::{
    build_full_system_prompt, build_new_chat_request, inject_system_prompt, parse_payload,
};
use super::types::{HandlerOutcome, IpcPayload, IpcResponse, IpcState};

use tokio::io::AsyncWriteExt;
use tokio::net::UnixStream;

/// Handle the "generate_stream" IPC action — streaming LLM chat with tool execution.
pub async fn handle_generate_stream(
    request: &IpcPayload,
    state: &IpcState,
    stream: &mut UnixStream,
) -> HandlerOutcome {
    let payload_clone = request.payload.clone();
    let parsed = parse_payload(&payload_clone);

    // Fast-path intent detection
    if !parsed.prompt.is_empty() {
        if let Some(tool_call) = crate::ai::tool_executor::detect_intent_from_user_message(
            &parsed.prompt,
            parsed.assistant_last.as_deref(),
        ) {
            if parsed.permissions.contains(&"all".to_string())
                || parsed.permissions.contains(&tool_call.name)
            {
                tracing::info!(
                    "🚀 [Hera IPC Stream] Fast-path tool intent detected: {}",
                    tool_call.name
                );

                let status_msg = IpcResponse {
                    status: "tool_status".to_string(),
                    data: serde_json::json!({"name": tool_call.name.clone()}),
                };
                let mut str_msg = serde_json::to_string(&status_msg).unwrap();
                str_msg.push('\n');
                let _ = stream.write_all(str_msg.as_bytes()).await;

                let tool_result = crate::ai::tool_executor::execute_tool(&tool_call).await;

                let chunk_msg = IpcResponse {
                    status: "chunk".to_string(),
                    data: serde_json::json!({"text": tool_result.output}),
                };
                let mut cstr = serde_json::to_string(&chunk_msg).unwrap();
                cstr.push('\n');
                let _ = stream.write_all(cstr.as_bytes()).await;

                let done_msg = IpcResponse {
                    status: "done".to_string(),
                    data: serde_json::json!({}),
                };
                let mut dstr = serde_json::to_string(&done_msg).unwrap();
                dstr.push('\n');
                let _ = stream.write_all(dstr.as_bytes()).await;
                return HandlerOutcome::DirectResponse;
            }
        }
    }

    // Ensure model is set
    let mut payload_mut = payload_clone.clone();
    if let Some(obj) = payload_mut.as_object_mut() {
        if !obj.contains_key("model") {
            obj.insert(
                "model".to_string(),
                serde_json::json!("hera-local-model"),
            );
        }
    }

    let prompt = payload_mut
        .get("prompt")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    // Build or augment ChatRequest
    let mut chat_req: Option<ChatRequest> =
        serde_json::from_value(payload_mut.clone()).ok();

    let full_system_prompt = build_full_system_prompt(
        &parsed.persona_path,
        &parsed.app_name,
        &parsed.permissions,
    )
    .await;

    if chat_req.is_none() {
        if !prompt.is_empty() {
            chat_req = Some(build_new_chat_request(&prompt, full_system_prompt));
        }
    } else if let Some(req) = &mut chat_req {
        inject_system_prompt(req, full_system_prompt);
    }

    if let Some(req) = chat_req.clone() {
        let est_tokens = super::helpers::estimate_tokens(&req);
        tracing::info!(
            "🔊 [Hera Stream] Starting stream for app='{}' — {} msgs, ~{} tokens",
            parsed.app_name,
            req.messages.len(),
            est_tokens
        );

        // Send stream_start
        let start_msg = IpcResponse {
            status: "stream_start".to_string(),
            data: serde_json::json!({}),
        };
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
                        let chunk_text = chunk
                            .choices
                            .first()
                            .and_then(|c| c.delta.content.clone())
                            .unwrap_or_default();
                        if chunk_text.is_empty() {
                            continue;
                        }

                        final_result_text.push_str(&chunk_text);

                        if !buffer_flushed {
                            let trimmed = final_result_text.trim_start();
                            let looks_like_tool = trimmed.starts_with('{')
                                || trimmed.starts_with("<tool_call>")
                                || trimmed.starts_with("<function-call>")
                                || trimmed.starts_with("<function_call>")
                                || trimmed.starts_with("<function=");
                            if looks_like_tool {
                                is_tool_call_mode = true;
                            } else if final_result_text.len() > 5 {
                                is_tool_call_mode = false;
                                buffer_flushed = true;
                                let chunk_msg = IpcResponse {
                                    status: "chunk".to_string(),
                                    data: serde_json::json!({"text": final_result_text}),
                                };
                                let mut cstr = serde_json::to_string(&chunk_msg).unwrap();
                                cstr.push('\n');
                                let _ = stream.write_all(cstr.as_bytes()).await;
                            }
                        } else if !is_tool_call_mode {
                            let chunk_msg = IpcResponse {
                                status: "chunk".to_string(),
                                data: serde_json::json!({"text": chunk_text}),
                            };
                            let mut cstr = serde_json::to_string(&chunk_msg).unwrap();
                            cstr.push('\n');
                            let _ = stream.write_all(cstr.as_bytes()).await;
                        }
                    }
                }

                // Flush any remaining buffered text
                if !buffer_flushed && !is_tool_call_mode && !final_result_text.is_empty() {
                    let chunk_msg = IpcResponse {
                        status: "chunk".to_string(),
                        data: serde_json::json!({"text": final_result_text}),
                    };
                    let mut cstr = serde_json::to_string(&chunk_msg).unwrap();
                    cstr.push('\n');
                    let _ = stream.write_all(cstr.as_bytes()).await;
                }

                // Parse tool calls from accumulated text
                let parsed_calls =
                    crate::ai::tool_executor::parse_tool_calls(&final_result_text);
                if !parsed_calls.is_empty() {
                    let mut execution_outputs = String::new();
                    for call in &parsed_calls {
                        let status_msg = IpcResponse {
                            status: "tool_status".to_string(),
                            data: serde_json::json!({"name": call.name.clone()}),
                        };
                        let mut str_msg = serde_json::to_string(&status_msg).unwrap();
                        str_msg.push('\n');
                        let _ = stream.write_all(str_msg.as_bytes()).await;

                        if parsed.permissions.contains(&"all".to_string())
                            || parsed.permissions.contains(&call.name)
                        {
                            let tool_res =
                                crate::ai::tool_executor::execute_tool(call).await;
                            execution_outputs
                                .push_str(&format!("\n\n{}", tool_res.output));
                        } else {
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
                                content: MessageContent::Text(final_result_text.clone()),
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
                            if let Ok(mut rx2) =
                                state.engine.generate_stream(req2).await
                            {
                                while let Some(chunk_res2) = rx2.recv().await {
                                    if let Ok(chunk2) = chunk_res2 {
                                        let chunk_text = chunk2
                                            .choices
                                            .first()
                                            .and_then(|c| c.delta.content.clone())
                                            .unwrap_or_default();
                                        let chunk_msg = IpcResponse {
                                            status: "chunk".to_string(),
                                            data: serde_json::json!({"text": chunk_text}),
                                        };
                                        let mut cstr =
                                            serde_json::to_string(&chunk_msg).unwrap();
                                        cstr.push('\n');
                                        let _ = stream.write_all(cstr.as_bytes()).await;
                                    }
                                }
                            }
                        }
                    } else {
                        let chunk_msg = IpcResponse {
                            status: "chunk".to_string(),
                            data: serde_json::json!({"text": execution_outputs}),
                        };
                        let mut cstr = serde_json::to_string(&chunk_msg).unwrap();
                        cstr.push('\n');
                        let _ = stream.write_all(cstr.as_bytes()).await;
                    }
                } else if is_tool_call_mode && !final_result_text.is_empty() {
                    // Suppressed stream assuming tool call, but it wasn't valid — dump buffered text
                    let chunk_msg = IpcResponse {
                        status: "chunk".to_string(),
                        data: serde_json::json!({"text": final_result_text}),
                    };
                    let mut cstr = serde_json::to_string(&chunk_msg).unwrap();
                    cstr.push('\n');
                    let _ = stream.write_all(cstr.as_bytes()).await;
                }

                // Send done
                let done_msg = IpcResponse {
                    status: "done".to_string(),
                    data: serde_json::json!({}),
                };
                let mut dstr = serde_json::to_string(&done_msg).unwrap();
                dstr.push('\n');
                let _ = stream.write_all(dstr.as_bytes()).await;
            }
            Err(e) => {
                let err_msg = IpcResponse {
                    status: "error".to_string(),
                    data: serde_json::json!({"error": e.to_string()}),
                };
                let mut estr = serde_json::to_string(&err_msg).unwrap();
                estr.push('\n');
                let _ = stream.write_all(estr.as_bytes()).await;
            }
        }
    }

    HandlerOutcome::DirectResponse
}
