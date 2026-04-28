use super::context::ParsedPayload;
use super::helpers::{fetch_single_app_schema_json, infer_origin_from_model};
use super::types::IpcResponse;
use crate::ai::{ChatMessage, ChatRequest, LLMEngine, MessageContent};
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use tokio::net::UnixStream;

pub struct ToolExecutionSummary {
    pub execution_outputs: String,
    pub executed_calls_json: Vec<serde_json::Value>,
    pub executed_tool_count: usize,
    pub has_media_call: bool,
}

pub enum FollowupStrategy<'a> {
    Buffered,
    Streaming(&'a mut UnixStream),
}

pub struct FollowupExecutionResult {
    pub text: String,
    pub model: Option<String>,
    pub origin: Option<String>,
}

pub fn contextualize_tool_call(
    tool_call: &crate::ai::tool_executor::ToolCall,
    parsed: &ParsedPayload,
) -> crate::ai::tool_executor::ToolCall {
    let mut arguments = tool_call.arguments.clone();
    let object = arguments
        .as_object_mut()
        .expect("tool call arguments should always be an object");

    if !parsed.app_name.is_empty() {
        object
            .entry("app_name".to_string())
            .or_insert_with(|| serde_json::json!(parsed.app_name));
        object
            .entry("app".to_string())
            .or_insert_with(|| serde_json::json!(parsed.app_name));
    }
    if !parsed.trace_id.is_empty() {
        object
            .entry("trace_id".to_string())
            .or_insert_with(|| serde_json::json!(parsed.trace_id));
    }
    if !parsed.session_id.is_empty() {
        object
            .entry("session_id".to_string())
            .or_insert_with(|| serde_json::json!(parsed.session_id));
    }
    if !parsed.chat_id.is_empty() {
        object
            .entry("chat_id".to_string())
            .or_insert_with(|| serde_json::json!(parsed.chat_id));
    }
    object.entry("caller".to_string()).or_insert_with(|| {
        serde_json::json!(if parsed.app_name.is_empty() {
            "hera"
        } else {
            &parsed.app_name
        })
    });

    crate::ai::tool_executor::ToolCall {
        name: tool_call.name.clone(),
        arguments,
    }
}

pub async fn execute_parsed_tool_calls(
    parsed_calls: &[crate::ai::tool_executor::ToolCall],
    parsed: &ParsedPayload,
    mut status_stream: Option<&mut UnixStream>,
) -> ToolExecutionSummary {
    let mut execution_outputs = String::new();
    let mut executed_calls_json = Vec::new();
    let mut executed_tool_count = 0usize;

    for call in parsed_calls {
        if let Some(stream) = status_stream.as_deref_mut() {
            let status_msg = IpcResponse {
                status: "tool_status".to_string(),
                data: serde_json::json!({"name": call.name.clone()}),
            };
            let mut str_msg = serde_json::to_string(&status_msg).unwrap();
            str_msg.push('\n');
            let _ = stream.write_all(str_msg.as_bytes()).await;
        }

        if crate::ai::tool_executor::permissions_allow_tool(&parsed.permissions, &call.name) {
            let contextual_call = contextualize_tool_call(call, parsed);
            let tool_res = crate::ai::tool_executor::execute_tool(&contextual_call).await;
            executed_tool_count += 1;
            execution_outputs.push_str(&format!("\n\n{}", tool_res.output));
            executed_calls_json.push(serde_json::json!({
                "name": contextual_call.name,
                "arguments": contextual_call.arguments
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

    let has_media_call = parsed_calls.iter().any(|call| {
        matches!(
            call.name.as_str(),
            "hera_draw" | "hera_video" | "generate_qr_code"
        )
    });

    ToolExecutionSummary {
        execution_outputs,
        executed_calls_json,
        executed_tool_count,
        has_media_call,
    }
}

pub async fn try_plan_schema_query(
    engine: &Arc<dyn LLMEngine + Send + Sync>,
    parsed: &ParsedPayload,
) -> Option<crate::ai::tool_executor::ToolCall> {
    if parsed.app_name.is_empty()
        || !parsed.permissions.iter().any(|perm| perm == "memento_query")
        || parsed.prompt.trim().is_empty()
        || parsed.prompt.trim_start().starts_with('/')
    {
        return None;
    }

    let schema = fetch_single_app_schema_json(&parsed.app_name).await?;
    if schema.is_empty() {
        return None;
    }

    let conversation_context = if parsed.recent_messages.is_empty() {
        String::new()
    } else {
        let start = parsed.recent_messages.len().saturating_sub(6);
        let excerpt = parsed.recent_messages[start..]
            .iter()
            .map(|(role, content)| format!("{role}: {content}"))
            .collect::<Vec<_>>()
            .join("\n");
        format!("\nRecent conversation:\n{}", excerpt)
    };
    let runtime_context = {
        let mut lines = Vec::new();
        if !parsed.sender_name.is_empty() {
            lines.push(format!("sender_name: {}", parsed.sender_name));
        }
        if !parsed.page_title.is_empty() {
            lines.push(format!("page_title: {}", parsed.page_title));
        }
        if !parsed.page_url.is_empty() {
            lines.push(format!("page_url: {}", parsed.page_url));
        }
        if !parsed.page_context.is_empty() {
            lines.push(format!("page_context: {}", parsed.page_context));
        }
        if lines.is_empty() {
            String::new()
        } else {
            format!("\nRuntime context:\n{}", lines.join("\n"))
        }
    };

    let schema_json = serde_json::to_string_pretty(&schema).ok()?;
    let planner_system = format!(
        "You are Hera's generic schema-aware query planner.\n\
Return only one raw JSON object.\n\
Never explain. Never use markdown. Never emit <tool_call> tags.\n\
Given an app schema and a user request, decide whether a SQL query is required.\n\
Allowed output schema:\n\
{{\"should_query\":true,\"query\":\"SELECT ...\",\"limit\":50,\"reason\":\"short\"}}\n\
or\n\
{{\"should_query\":false,\"reason\":\"short\"}}\n\
Rules:\n\
- SQL must be SELECT or WITH only.\n\
- Use only tables and columns present in the schema.\n\
- Prefer concise queries.\n\
- If aggregating numeric columns with SUM/AVG, CAST the aggregate to double precision so JSON transport stays typed.\n\
- If the user asks for grouping, include grouped dimensions.\n\
- Resolve ambiguous follow-up questions using the recent conversation when it is provided.\n\
- Use runtime context (current debtor/account/page context) when it identifies the subject of the request.\n\
- If runtime context already identifies the debtor or account reference, prefer that context instead of asking the user to repeat it.\n\
- If runtime context includes an exact identifier such as *_id, document_id, account_reference, reference, uuid, or pid, prefer filtering with that exact identifier over human names.\n\
- Avoid filtering by name alone when runtime context already includes a more specific identifier.\n\
- If the request can be answered without data access, set should_query=false.\n\
App: {}{}{}\nSchema:\n{}",
        parsed.app_name, conversation_context, runtime_context, schema_json
    );

    let req = ChatRequest {
        model: "hera-local-model".to_string(),
        vision_model: None,
        tts_model: None,
        stt_model: None,
        messages: vec![
            ChatMessage {
                role: "system".to_string(),
                content: MessageContent::Text(planner_system),
            },
            ChatMessage {
                role: "user".to_string(),
                content: MessageContent::Text(parsed.prompt.clone()),
            },
        ],
        temperature: Some(0.0),
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
        provider: Some("local".to_string()),
        stream: Some(false),
        nsfw: None,
        tools: None,
        tool_choice: None,
        reasoning_effort: Some("medium".to_string()),
    };

    let resp = engine.generate_content(req).await.ok()?;
    let content = resp
        .choices
        .first()
        .and_then(|choice| choice.message.content.clone())?;
    let plan_value = parse_first_json_object(&content)?;

    if !plan_value
        .get("should_query")
        .and_then(|value| value.as_bool())
        .unwrap_or(false)
    {
        return None;
    }

    let query = plan_value.get("query").and_then(|value| value.as_str())?;
    if !is_safe_select_query(query, &schema) {
        tracing::warn!(
            app = %parsed.app_name,
            "Rejected schema planner query because validation failed: {}",
            query
        );
        return None;
    }

    Some(crate::ai::tool_executor::ToolCall {
        name: "memento_query".to_string(),
        arguments: serde_json::json!({
            "app": parsed.app_name,
            "query": query,
            "limit": plan_value.get("limit").and_then(|value| value.as_u64()).unwrap_or(50)
        }),
    })
}

pub async fn summarize_tool_output_for_user(
    engine: &Arc<dyn LLMEngine + Send + Sync>,
    parsed: &ParsedPayload,
    tool_output: &str,
) -> Option<String> {
    let req = ChatRequest {
        model: "hera-local-model".to_string(),
        vision_model: None,
        tts_model: None,
        stt_model: None,
        messages: vec![
            ChatMessage {
                role: "system".to_string(),
                content: MessageContent::Text(
                    "You are Hera. Summarize tool results for the user in the same language as the original question. Be concise, clear, and directly answer the request. Do not mention SQL, tables, or internal tools.".to_string(),
                ),
            },
            ChatMessage {
                role: "user".to_string(),
                content: MessageContent::Text(format!(
                    "Original request:\n{}\n\nTool result:\n{}",
                    parsed.prompt, tool_output
                )),
            },
        ],
        temperature: Some(0.1),
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
        provider: Some("local".to_string()),
        stream: Some(false),
        nsfw: None,
        tools: None,
        tool_choice: None,
        reasoning_effort: Some("low".to_string()),
    };

    let resp = engine.generate_content(req).await.ok()?;
    resp.choices
        .first()
        .and_then(|choice| choice.message.content.clone())
        .filter(|text| !text.trim().is_empty())
}

fn parse_first_json_object(text: &str) -> Option<serde_json::Value> {
    let trimmed = if let Some(end_idx) = text.find("</think>") {
        text[end_idx + "</think>".len()..].trim()
    } else {
        text.trim()
    };

    let start = trimmed.find('{')?;
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;
    let mut end = None;

    for (idx, ch) in trimmed[start..].char_indices() {
        if in_string {
            if escaped {
                escaped = false;
                continue;
            }
            match ch {
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(start + idx + 1);
                    break;
                }
            }
            _ => {}
        }
    }

    let end = end?;
    serde_json::from_str::<serde_json::Value>(&trimmed[start..end]).ok()
}

fn is_safe_select_query(
    query: &str,
    schema: &serde_json::Map<String, serde_json::Value>,
) -> bool {
    let normalized = query.trim().to_lowercase();
    if !(normalized.starts_with("select") || normalized.starts_with("with")) {
        return false;
    }
    if normalized.contains(';') {
        return false;
    }
    for forbidden in ["insert ", "update ", "delete ", "drop ", "alter ", "truncate "] {
        if normalized.contains(forbidden) {
            return false;
        }
    }

    let known_tables: Vec<String> = schema.keys().cloned().collect();
    known_tables.iter().any(|table| normalized.contains(table))
}


pub async fn execute_tool_followup(
    engine: &Arc<dyn LLMEngine + Send + Sync>,
    request: ChatRequest,
    strategy: FollowupStrategy<'_>,
) -> Result<FollowupExecutionResult, String> {
    match strategy {
        FollowupStrategy::Buffered => {
            let response = engine
                .generate_content(request)
                .await
                .map_err(|error| error.to_string())?;
            let model = response.model.clone();
            let origin = infer_origin_from_model(&model).to_string();
            let text = response
                .choices
                .first()
                .and_then(|choice| choice.message.content.clone())
                .unwrap_or_default();

            Ok(FollowupExecutionResult {
                text,
                model: Some(model),
                origin: Some(origin),
            })
        }
        FollowupStrategy::Streaming(stream) => {
            let mut rx = engine
                .generate_stream(request)
                .await
                .map_err(|error| error.to_string())?;
            let mut text = String::new();
            let mut model = None;
            let mut origin = None;

            while let Some(chunk_res) = rx.recv().await {
                let chunk = chunk_res.map_err(|error| error.to_string())?;
                if model.is_none() && !chunk.model.is_empty() {
                    model = Some(chunk.model.clone());
                    origin = Some(infer_origin_from_model(&chunk.model).to_string());
                }

                let chunk_text = chunk
                    .choices
                    .first()
                    .and_then(|choice| choice.delta.content.clone())
                    .unwrap_or_default();
                if chunk_text.is_empty() {
                    continue;
                }

                text.push_str(&chunk_text);
                let chunk_msg = IpcResponse {
                    status: "chunk".to_string(),
                    data: serde_json::json!({ "text": chunk_text }),
                };
                let mut cstr = serde_json::to_string(&chunk_msg).unwrap();
                cstr.push('\n');
                let _ = stream.write_all(cstr.as_bytes()).await;
            }

            Ok(FollowupExecutionResult {
                text,
                model,
                origin,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::{
        ChatChoice, ChatRequest, ChatResponse, ChatResponseMessage, ChatStreamChoice,
        ChatStreamDelta, ChatStreamResponse, InferenceError, MessageContent,
    };
    use std::sync::Arc;
    use tokio::io::AsyncReadExt;
    use tokio::sync::mpsc;

    struct MockEngine;

    fn minimal_request() -> ChatRequest {
        ChatRequest {
            model: "hera-local-model".to_string(),
            vision_model: None,
            tts_model: None,
            stt_model: None,
            messages: vec![crate::ai::ChatMessage {
                role: "user".to_string(),
                content: MessageContent::Text("hi".to_string()),
            }],
            temperature: None,
            max_tokens: None,
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
        }
    }

    #[async_trait::async_trait]
    impl LLMEngine for MockEngine {
        async fn generate_content(
            &self,
            _req: ChatRequest,
        ) -> Result<ChatResponse, InferenceError> {
            Ok(ChatResponse {
                id: "resp_1".to_string(),
                object: "chat.completion".to_string(),
                created: 0,
                model: "mock-local-model".to_string(),
                choices: vec![ChatChoice {
                    index: 0,
                    message: ChatResponseMessage {
                        role: "assistant".to_string(),
                        content: Some("buffered followup".to_string()),
                        tool_calls: None,
                    },
                    finish_reason: Some("stop".to_string()),
                }],
                usage: None,
            })
        }

        async fn generate_stream(
            &self,
            _req: ChatRequest,
        ) -> Result<mpsc::Receiver<Result<ChatStreamResponse, InferenceError>>, InferenceError>
        {
            let (tx, rx) = mpsc::channel(4);
            tokio::spawn(async move {
                let _ = tx
                    .send(Ok(ChatStreamResponse {
                        id: "stream_1".to_string(),
                        object: "chat.completion.chunk".to_string(),
                        created: 0,
                        model: "mock-local-stream-model".to_string(),
                        choices: vec![ChatStreamChoice {
                            index: 0,
                            delta: ChatStreamDelta {
                                role: None,
                                content: Some("streamed followup".to_string()),
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

    #[tokio::test]
    async fn execute_tool_followup_buffered_returns_text_and_origin() {
        let engine: Arc<dyn LLMEngine + Send + Sync> = Arc::new(MockEngine);
        let result = execute_tool_followup(&engine, minimal_request(), FollowupStrategy::Buffered)
            .await
            .expect("buffered followup should succeed");

        assert_eq!(result.text, "buffered followup");
        assert_eq!(result.model.as_deref(), Some("mock-local-model"));
        assert_eq!(result.origin.as_deref(), Some("local"));
    }

    #[tokio::test]
    async fn execute_tool_followup_streaming_writes_chunk_and_returns_text() {
        let engine: Arc<dyn LLMEngine + Send + Sync> = Arc::new(MockEngine);
        let (mut writer, mut reader) = tokio::net::UnixStream::pair().expect("unix pair");

        let result = execute_tool_followup(
            &engine,
            minimal_request(),
            FollowupStrategy::Streaming(&mut writer),
        )
        .await
        .expect("streaming followup should succeed");

        let mut buf = vec![0u8; 4096];
        let n = tokio::time::timeout(
            std::time::Duration::from_millis(250),
            reader.read(&mut buf),
        )
        .await
        .expect("chunk should be written")
        .expect("read should succeed");
        let written = String::from_utf8_lossy(&buf[..n]);

        assert!(written.contains("\"status\":\"chunk\""));
        assert!(written.contains("streamed followup"));
        assert_eq!(result.text, "streamed followup");
        assert_eq!(result.model.as_deref(), Some("mock-local-stream-model"));
        assert_eq!(result.origin.as_deref(), Some("local"));
    }
}
