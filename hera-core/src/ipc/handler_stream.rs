//! Handler: generate_stream (streaming LLM inference with tool execution).

use super::context::{
    build_runtime_outcome_artifacts, parse_payload, prepare_chat_request,
    prepare_runtime_execution_context, prepare_tool_result_followup_request,
};
use super::helpers::{
    RuntimePromotionContext, infer_origin_from_model, record_observation_and_promote_runtime_hint,
    record_runtime_observation,
};
use super::llm_audit::append_llm_audit_event;
use super::runtime_tools::{
    FollowupStrategy, contextualize_tool_call, execute_parsed_tool_calls, execute_tool_followup,
    summarize_tool_output_for_user, try_plan_schema_query,
};
use super::types::{HandlerOutcome, IpcPayload, IpcResponse, IpcState};
use std::time::Instant;
use tokio::io::AsyncWriteExt;
use tokio::net::UnixStream;

/// Handle the "generate_stream" IPC action — streaming LLM chat with tool execution.
pub async fn handle_generate_stream(
    request: &IpcPayload,
    state: &IpcState,
    stream: &mut UnixStream,
) -> HandlerOutcome {
    let started_at = Instant::now();
    let payload_clone = request.payload.clone();
    let mut parsed = parse_payload(&payload_clone);
    let provider_requested = payload_clone
        .get("provider")
        .and_then(|value| value.as_str())
        .unwrap_or("auto")
        .to_string();
    let caller_overrode_budget = payload_clone.get("context_budget_mode").is_some()
        || payload_clone.get("context_budget").is_some();

    // Fast-path intent detection
    if !parsed.prompt.is_empty() {
        if let Some(tool_call) = crate::ai::tool_executor::detect_intent_from_user_message(
            &parsed.prompt,
            parsed.assistant_last.as_deref(),
        ) {
            let contextual_tool_call = contextualize_tool_call(&tool_call, &parsed);
            if crate::ai::tool_executor::permissions_allow_tool(
                &parsed.permissions,
                &contextual_tool_call.name,
            ) {
                tracing::info!(
                    "🚀 [Hera IPC Stream] Fast-path tool intent detected: {}",
                    contextual_tool_call.name
                );

                let status_msg = IpcResponse {
                    status: "tool_status".to_string(),
                    data: serde_json::json!({"name": contextual_tool_call.name.clone()}),
                };
                let mut str_msg = serde_json::to_string(&status_msg).unwrap();
                str_msg.push('\n');
                let _ = stream.write_all(str_msg.as_bytes()).await;

                let tool_result = crate::ai::tool_executor::execute_tool(&contextual_tool_call)
                    .await
                    .output;

                let chunk_msg = IpcResponse {
                    status: "chunk".to_string(),
                    data: serde_json::json!({"text": tool_result}),
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

    if let Some(planned_call) = try_plan_schema_query(&state.engine, &parsed).await {
        let contextual_tool_call = contextualize_tool_call(&planned_call, &parsed);
        if crate::ai::tool_executor::permissions_allow_tool(
            &parsed.permissions,
            &contextual_tool_call.name,
        ) {
            tracing::info!(
                "🧠 [Hera IPC Stream] Generic schema query plan generated for app '{}'",
                parsed.app_name
            );

            let status_msg = IpcResponse {
                status: "tool_status".to_string(),
                data: serde_json::json!({"name": contextual_tool_call.name.clone()}),
            };
            let mut str_msg = serde_json::to_string(&status_msg).unwrap();
            str_msg.push('\n');
            let _ = stream.write_all(str_msg.as_bytes()).await;

            let tool_result = crate::ai::tool_executor::execute_tool(&contextual_tool_call)
                .await
                .output;
            let result_text = summarize_tool_output_for_user(&state.engine, &parsed, &tool_result)
                .await
                .unwrap_or(tool_result);

            let chunk_msg = IpcResponse {
                status: "chunk".to_string(),
                data: serde_json::json!({"text": result_text}),
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

    // Ensure model is set
    let mut payload_mut = payload_clone.clone();
    if let Some(obj) = payload_mut.as_object_mut() {
        if !obj.contains_key("model") {
            obj.insert("model".to_string(), serde_json::json!("hera-local-model"));
        }
    }

    let prompt = payload_mut
        .get("prompt")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let prepared =
        prepare_runtime_execution_context(&mut parsed, caller_overrode_budget, "generate_stream")
            .await;
    let runtime_preflight = prepared.runtime_preflight;
    let prompt_assembly = prepared.prompt_assembly;
    let lightweight_mode = prepared.lightweight_mode;
    let chat_req = prepare_chat_request(
        &payload_mut,
        &prompt,
        &parsed,
        &prompt_assembly,
        &state.engine,
    )
    .await;

    if let Some(req) = chat_req.clone() {
        let est_tokens = super::helpers::estimate_tokens(&req);
        tracing::info!(
            "🔊 [Hera Stream] Starting stream for app='{}' — {} msgs, ~{} tokens (lightweight_mode={})",
            parsed.app_name,
            req.messages.len(),
            est_tokens,
            lightweight_mode
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
        let mut first_token_ms = None;
        let mut response_model = String::new();
        let mut response_origin = "unknown".to_string();
        let mut executed_tool_count = 0usize;

        match state.engine.generate_stream(req).await {
            Ok(mut rx) => {
                while let Some(chunk_res) = rx.recv().await {
                    if let Ok(chunk) = chunk_res {
                        if response_model.is_empty() && !chunk.model.is_empty() {
                            response_model = chunk.model.clone();
                            response_origin = infer_origin_from_model(&chunk.model).to_string();
                        }
                        let chunk_text = chunk
                            .choices
                            .first()
                            .and_then(|c| c.delta.content.clone())
                            .unwrap_or_default();
                        if chunk_text.is_empty() {
                            continue;
                        }
                        if first_token_ms.is_none() {
                            first_token_ms = Some(started_at.elapsed().as_millis() as u64);
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
                let parsed_calls = crate::ai::tool_executor::parse_tool_calls(&final_result_text);
                if !parsed_calls.is_empty() {
                    let tool_summary =
                        execute_parsed_tool_calls(&parsed_calls, &parsed, Some(stream)).await;
                    executed_tool_count += tool_summary.executed_tool_count;
                    let execution_outputs = tool_summary.execution_outputs;
                    if !tool_summary.has_media_call {
                        let json_mode = payload_clone
                            .get("json_mode")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false);
                        if let Some(req2) = prepare_tool_result_followup_request(
                            chat_req.clone(),
                            &final_result_text,
                            &execution_outputs,
                            json_mode,
                        ) {
                            if let Ok(followup) = execute_tool_followup(
                                &state.engine,
                                req2,
                                FollowupStrategy::Streaming(stream),
                            )
                            .await
                            {
                                if let Some(model) = followup.model {
                                    response_origin = followup.origin.unwrap_or_else(|| {
                                        infer_origin_from_model(&model).to_string()
                                    });
                                    response_model = model;
                                }
                                if !followup.text.is_empty() {
                                    final_result_text = followup.text;
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

                let duration_ms = started_at.elapsed().as_millis() as u64;
                let outcome = build_runtime_outcome_artifacts(
                    "generate_stream",
                    &parsed,
                    &prompt_assembly,
                    duration_ms,
                    first_token_ms,
                    lightweight_mode,
                    &provider_requested,
                    &response_origin,
                    &response_model,
                    true,
                    executed_tool_count,
                    final_result_text.len(),
                    est_tokens,
                    chat_req
                        .as_ref()
                        .map(|req| req.messages.len())
                        .unwrap_or_default(),
                    None,
                );
                append_llm_audit_event(&outcome.audit_event);
                record_observation_and_promote_runtime_hint(
                    outcome.observation_payload,
                    RuntimePromotionContext {
                        preflight: runtime_preflight.clone(),
                        mode: "generate_stream",
                        app_id: &parsed.app_name,
                        route_profile: &parsed.route_profile_id,
                        persona_path: &parsed.persona_path,
                        session_id: &parsed.session_id,
                        trace_id: &parsed.trace_id,
                        chat_id: &parsed.chat_id,
                        current_budget_mode: &parsed.context_budget.mode,
                        persona_drift: parsed.persona_drift,
                        success: true,
                    },
                )
                .await;
            }
            Err(e) => {
                let err_msg = IpcResponse {
                    status: "error".to_string(),
                    data: serde_json::json!({"error": e.to_string()}),
                };
                let mut estr = serde_json::to_string(&err_msg).unwrap();
                estr.push('\n');
                let _ = stream.write_all(estr.as_bytes()).await;
                let duration_ms = started_at.elapsed().as_millis() as u64;
                let error_text = e.to_string();
                let outcome = build_runtime_outcome_artifacts(
                    "generate_stream",
                    &parsed,
                    &prompt_assembly,
                    duration_ms,
                    first_token_ms,
                    lightweight_mode,
                    &provider_requested,
                    "offline",
                    "",
                    false,
                    0,
                    0,
                    est_tokens,
                    chat_req
                        .as_ref()
                        .map(|req| req.messages.len())
                        .unwrap_or_default(),
                    Some(error_text.clone()),
                );
                append_llm_audit_event(&outcome.audit_event);
                let _ = record_runtime_observation(outcome.observation_payload).await;
            }
        }
    }

    HandlerOutcome::DirectResponse
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::{
        ChatRequest, ChatResponse, ChatStreamChoice, ChatStreamDelta, ChatStreamResponse,
        InferenceError,
    };
    use std::sync::{Arc, Mutex};
    use tokio::io::AsyncReadExt;
    use tokio::sync::mpsc;

    struct MockStreamEngine {
        calls: Mutex<u32>,
    }

    impl MockStreamEngine {
        fn new() -> Self {
            Self {
                calls: Mutex::new(0),
            }
        }
    }

    #[async_trait::async_trait]
    impl crate::ai::LLMEngine for MockStreamEngine {
        async fn generate_content(
            &self,
            _req: ChatRequest,
        ) -> Result<ChatResponse, InferenceError> {
            Err(InferenceError::ExecutionFailed(
                "not used in handler_stream tests".to_string(),
            ))
        }

        async fn generate_stream(
            &self,
            _req: ChatRequest,
        ) -> Result<mpsc::Receiver<Result<ChatStreamResponse, InferenceError>>, InferenceError>
        {
            let mut guard = self.calls.lock().expect("calls lock");
            *guard += 1;
            let call_no = *guard;
            drop(guard);

            let (tx, rx) = mpsc::channel(8);
            tokio::spawn(async move {
                let text = if call_no == 1 {
                    "plain hello from stream".to_string()
                } else {
                    "final answer after tool".to_string()
                };
                let model = if call_no == 1 {
                    "mock-local-stream-model".to_string()
                } else {
                    "mock-local-followup-model".to_string()
                };
                let _ = tx
                    .send(Ok(ChatStreamResponse {
                        id: format!("stream_{call_no}"),
                        object: "chat.completion.chunk".to_string(),
                        created: 0,
                        model,
                        choices: vec![ChatStreamChoice {
                            index: 0,
                            delta: ChatStreamDelta {
                                role: None,
                                content: Some(text),
                                tool_calls: None,
                            },
                            finish_reason: Some("stop".to_string()),
                        }],
                        stats: None,
                    }))
                    .await;
            });
            Ok(rx)
        }
    }

    struct MockToolCallEngine {
        calls: Mutex<u32>,
    }

    impl MockToolCallEngine {
        fn new() -> Self {
            Self {
                calls: Mutex::new(0),
            }
        }
    }

    #[async_trait::async_trait]
    impl crate::ai::LLMEngine for MockToolCallEngine {
        async fn generate_content(
            &self,
            _req: ChatRequest,
        ) -> Result<ChatResponse, InferenceError> {
            Err(InferenceError::ExecutionFailed(
                "not used in handler_stream tests".to_string(),
            ))
        }

        async fn generate_stream(
            &self,
            _req: ChatRequest,
        ) -> Result<mpsc::Receiver<Result<ChatStreamResponse, InferenceError>>, InferenceError>
        {
            let mut guard = self.calls.lock().expect("calls lock");
            *guard += 1;
            let call_no = *guard;
            drop(guard);

            let (tx, rx) = mpsc::channel(8);
            tokio::spawn(async move {
                let (text, model) = if call_no == 1 {
                    (
                        "<tool_call>{\"name\":\"get_system_time\",\"arguments\":{}}</tool_call>"
                            .to_string(),
                        "mock-local-stream-model".to_string(),
                    )
                } else {
                    (
                        "final answer after tool".to_string(),
                        "mock-local-followup-model".to_string(),
                    )
                };
                let _ = tx
                    .send(Ok(ChatStreamResponse {
                        id: format!("stream_{call_no}"),
                        object: "chat.completion.chunk".to_string(),
                        created: 0,
                        model,
                        choices: vec![ChatStreamChoice {
                            index: 0,
                            delta: ChatStreamDelta {
                                role: None,
                                content: Some(text),
                                tool_calls: None,
                            },
                            finish_reason: Some("stop".to_string()),
                        }],
                        stats: None,
                    }))
                    .await;
            });
            Ok(rx)
        }
    }

    fn test_state(engine: Arc<dyn crate::ai::LLMEngine + Send + Sync>) -> IpcState {
        IpcState {
            engine: engine.clone(),
            local_engine: engine,
            flux_engine: None,
            parler_engine: None,
            whisper_engine: None,
            vision_engine: None,
            micro_engine: None,
        }
    }

    async fn run_stream_request(
        engine: Arc<dyn crate::ai::LLMEngine + Send + Sync>,
        payload: serde_json::Value,
    ) -> String {
        let state = test_state(engine);
        let request = IpcPayload {
            action: "generate_stream".to_string(),
            payload,
        };
        let (mut writer, mut reader) = tokio::net::UnixStream::pair().expect("unix pair");
        let outcome = handle_generate_stream(&request, &state, &mut writer).await;
        assert!(matches!(outcome, HandlerOutcome::DirectResponse));
        drop(writer);

        let mut buf = Vec::new();
        reader.read_to_end(&mut buf).await.expect("read all");
        String::from_utf8(buf).expect("utf8 output")
    }

    fn event_statuses(output: &str) -> Vec<String> {
        output
            .lines()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .filter_map(|value| {
                value
                    .get("status")
                    .and_then(|value| value.as_str())
                    .map(str::to_string)
            })
            .collect()
    }

    #[tokio::test]
    async fn handler_stream_plain_text_emits_stream_start_chunk_done() {
        let engine: Arc<dyn crate::ai::LLMEngine + Send + Sync> =
            Arc::new(MockStreamEngine::new());
        let output = run_stream_request(
            engine,
            serde_json::json!({
                "messages": [
                    {"role": "user", "content": "hola"}
                ]
            }),
        )
        .await;

        let statuses = event_statuses(&output);
        assert_eq!(statuses, vec!["stream_start", "chunk", "done"]);
        assert!(output.contains("plain hello from stream"));
    }

    #[tokio::test]
    async fn handler_stream_tool_flow_emits_tool_status_before_final_chunk_and_done() {
        let engine: Arc<dyn crate::ai::LLMEngine + Send + Sync> =
            Arc::new(MockToolCallEngine::new());
        let output = run_stream_request(
            engine,
            serde_json::json!({
                "messages": [
                    {"role": "user", "content": "hola"}
                ],
                "permissions": ["get_system_time"]
            }),
        )
        .await;

        let statuses = event_statuses(&output);
        assert_eq!(statuses, vec!["stream_start", "tool_status", "chunk", "done"]);
        let tool_idx = output.find("\"status\":\"tool_status\"").expect("tool status");
        let chunk_idx = output
            .find("final answer after tool")
            .expect("final followup chunk");
        let done_idx = output.find("\"status\":\"done\"").expect("done");
        assert!(tool_idx < chunk_idx);
        assert!(chunk_idx < done_idx);
    }
}
