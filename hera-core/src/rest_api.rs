use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{get, post},
};
use futures_util::Stream;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    convert::Infallible,
    net::SocketAddr,
    time::{SystemTime, UNIX_EPOCH},
};
use tokio_stream::{StreamExt, wrappers::ReceiverStream};
use tower_http::cors::{Any, CorsLayer};
use tracing::{error, info, warn};

use crate::ai::{ChatMessage, ChatRequest, ContentPart, ImageUrlContent, MessageContent};
use crate::ipc::{IpcState, helpers::estimate_tokens};

const LOCAL_CONTEXT_SOFT_LIMIT: usize = 24_000;
const LOCAL_CHAR_SOFT_LIMIT: usize = 24_000;

#[derive(Clone)]
struct RestApiState {
    ipc: IpcState,
}

#[derive(Debug, Deserialize)]
struct AnthropicMessageRequest {
    model: String,
    max_tokens: u32,
    #[serde(default)]
    messages: Vec<AnthropicMessage>,
    #[serde(default)]
    system: Option<AnthropicContent>,
    #[serde(default)]
    metadata: Option<Value>,
    #[serde(default)]
    stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    tools: Option<Vec<Value>>,
    #[serde(default)]
    tool_choice: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct AnthropicMessage {
    role: String,
    content: AnthropicContent,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum AnthropicContent {
    Text(String),
    Blocks(Vec<AnthropicBlock>),
}

#[derive(Debug, Clone, Deserialize)]
struct AnthropicBlock {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    source: Option<AnthropicImageSource>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    input: Option<Value>,
}

#[derive(Debug, Clone, Deserialize)]
struct AnthropicImageSource {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    media_type: Option<String>,
    #[serde(default)]
    data: Option<String>,
}

#[derive(Debug, Serialize)]
struct AnthropicMessageResponse {
    id: String,
    #[serde(rename = "type")]
    object_type: &'static str,
    role: &'static str,
    content: Vec<Value>,
    model: String,
    stop_reason: String,
    stop_sequence: Option<String>,
    usage: AnthropicUsage,
}

#[derive(Debug, Serialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct CountTokensRequest {
    #[serde(default)]
    messages: Vec<AnthropicMessage>,
    #[serde(default)]
    system: Option<AnthropicContent>,
    #[serde(default)]
    tools: Option<Vec<Value>>,
}

#[derive(Debug, Serialize)]
struct CountTokensResponse {
    input_tokens: u32,
}

#[derive(Debug, Serialize)]
struct ModelListResponse {
    data: Vec<ModelInfo>,
    has_more: bool,
    first_id: Option<String>,
    last_id: Option<String>,
}

#[derive(Debug, Serialize)]
struct ModelInfo {
    #[serde(rename = "type")]
    object_type: &'static str,
    id: String,
    display_name: String,
    created_at: String,
}

pub async fn serve_rest_api(port: u16, ipc: IpcState) {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let state = RestApiState { ipc };

    let app = Router::new()
        .route("/v1/messages", post(handle_messages))
        .route("/v1/messages/count_tokens", post(handle_count_tokens))
        .route("/v1/models", get(handle_models))
        .with_state(state)
        .layer(cors);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = match tokio::net::TcpListener::bind(addr).await {
        Ok(listener) => listener,
        Err(error) => {
            warn!(
                "⚠️ Hera REST API did not start on http://{}: {}",
                addr, error
            );
            return;
        }
    };
    info!(
        "🚀 Hera REST API (Claude-compatible) bound to http://{}",
        addr
    );
    if let Err(e) = axum::serve(listener, app).await {
        error!("❌ Hera REST API Server Error: {}", e);
    }
}

async fn handle_messages(
    State(state): State<RestApiState>,
    Json(payload): Json<AnthropicMessageRequest>,
) -> Response {
    let mut chat_req = match anthropic_to_chat_request(&payload) {
        Ok(req) => req,
        Err(message) => return api_error(StatusCode::BAD_REQUEST, &message).into_response(),
    };
    info!(
        "Claude-compatible request received: model={} messages={} estimated_tokens_before={} chars_before={}",
        payload.model,
        chat_req.messages.len(),
        estimate_tokens(&chat_req),
        total_message_chars(&chat_req)
    );
    clamp_chat_request(&mut chat_req, LOCAL_CONTEXT_SOFT_LIMIT);
    info!(
        "Claude-compatible request clamped: model={} messages={} estimated_tokens_after={} chars_after={}",
        chat_req.model,
        chat_req.messages.len(),
        estimate_tokens(&chat_req),
        total_message_chars(&chat_req)
    );

    if payload.stream.unwrap_or(false) {
        return handle_messages_stream(state, payload, chat_req)
            .await
            .into_response();
    }

    match state
        .ipc
        .local_engine
        .generate_content(chat_req.clone())
        .await
    {
        Ok(resp) => {
            let response = anthropic_message_response(&payload.model, &chat_req, &resp);
            (StatusCode::OK, Json(response)).into_response()
        }
        Err(e) => api_error(
            StatusCode::BAD_GATEWAY,
            &format!("hera inference failed: {e}"),
        )
        .into_response(),
    }
}

async fn handle_messages_stream(
    state: RestApiState,
    original: AnthropicMessageRequest,
    mut chat_req: ChatRequest,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    info!(
        "Claude-compatible stream request received: model={} messages={} estimated_tokens_before={} chars_before={}",
        original.model,
        chat_req.messages.len(),
        estimate_tokens(&chat_req),
        total_message_chars(&chat_req)
    );
    clamp_chat_request(&mut chat_req, LOCAL_CONTEXT_SOFT_LIMIT);
    info!(
        "Claude-compatible stream request clamped: model={} messages={} estimated_tokens_after={} chars_after={}",
        chat_req.model,
        chat_req.messages.len(),
        estimate_tokens(&chat_req),
        total_message_chars(&chat_req)
    );
    let input_tokens = estimate_tokens(&chat_req) as u32;
    let response_model = if chat_req.model.is_empty() {
        original.model.clone()
    } else {
        chat_req.model.clone()
    };
    let message_id = new_message_id();

    let (tx, rx) = tokio::sync::mpsc::channel::<Event>(32);

    tokio::spawn(async move {
        send_sse_event(
            &tx,
            "message_start",
            json!({
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": response_model,
                    "stop_reason": null,
                    "stop_sequence": null,
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": 0
                    }
                }
            }),
        )
        .await;

        send_sse_event(
            &tx,
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "text",
                    "text": ""
                }
            }),
        )
        .await;

        let mut output_tokens = 0u32;
        let mut stop_reason = "end_turn".to_string();

        match state.ipc.local_engine.generate_stream(chat_req).await {
            Ok(mut stream) => {
                while let Some(chunk_res) = stream.recv().await {
                    match chunk_res {
                        Ok(chunk) => {
                            if let Some(choice) = chunk.choices.first() {
                                if let Some(text) = &choice.delta.content
                                    && !text.is_empty()
                                {
                                    output_tokens += (text.len() / 4).max(1) as u32;
                                    send_sse_event(
                                        &tx,
                                        "content_block_delta",
                                        json!({
                                            "type": "content_block_delta",
                                            "index": 0,
                                            "delta": {
                                                "type": "text_delta",
                                                "text": text
                                            }
                                        }),
                                    )
                                    .await;
                                }

                                if choice.delta.tool_calls.is_some() {
                                    stop_reason = "tool_use".to_string();
                                } else if let Some(finish) = &choice.finish_reason
                                    && finish == "tool_calls"
                                {
                                    stop_reason = "tool_use".to_string();
                                }
                            }
                        }
                        Err(e) => {
                            let _ = tx
                                .send(
                                    Event::default().event("error").data(
                                        serde_json::to_string(&json!({
                                            "type": "error",
                                            "error": {
                                                "type": "api_error",
                                                "message": format!("hera streaming failed: {e}")
                                            }
                                        }))
                                        .unwrap_or_else(|_| "{\"type\":\"error\"}".to_string()),
                                    ),
                                )
                                .await;
                            return;
                        }
                    }
                }
            }
            Err(e) => {
                let _ = tx
                    .send(
                        Event::default().event("error").data(
                            serde_json::to_string(&json!({
                                "type": "error",
                                "error": {
                                    "type": "api_error",
                                    "message": format!("hera streaming failed: {e}")
                                }
                            }))
                            .unwrap_or_else(|_| "{\"type\":\"error\"}".to_string()),
                        ),
                    )
                    .await;
                return;
            }
        }

        send_sse_event(
            &tx,
            "content_block_stop",
            json!({
                "type": "content_block_stop",
                "index": 0
            }),
        )
        .await;

        send_sse_event(
            &tx,
            "message_delta",
            json!({
                "type": "message_delta",
                "delta": {
                    "stop_reason": stop_reason,
                    "stop_sequence": null
                },
                "usage": {
                    "output_tokens": output_tokens
                }
            }),
        )
        .await;

        send_sse_event(&tx, "message_stop", json!({ "type": "message_stop" })).await;
    });

    let stream = ReceiverStream::new(rx).map(Ok::<Event, Infallible>);
    Sse::new(stream).keep_alive(KeepAlive::default())
}

async fn handle_count_tokens(
    Json(payload): Json<CountTokensRequest>,
) -> Result<Json<CountTokensResponse>, (StatusCode, Json<Value>)> {
    let chat_req = anthropic_count_to_chat_request(payload)
        .map_err(|message| api_error(StatusCode::BAD_REQUEST, &message))?;
    Ok(Json(CountTokensResponse {
        input_tokens: estimate_tokens(&chat_req) as u32,
    }))
}

async fn handle_models() -> Json<ModelListResponse> {
    let local_model = std::env::var("HERA_OPENAI_MODEL")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "hera-local-model".to_string());
    let cloud_model = std::env::var("OPENROUTER_DEFAULT_MODEL")
        .ok()
        .filter(|s| !s.trim().is_empty());

    let mut data = vec![ModelInfo {
        object_type: "model",
        id: local_model.clone(),
        display_name: "Hera Local".to_string(),
        created_at: "2026-01-01T00:00:00Z".to_string(),
    }];

    if let Some(cloud) = cloud_model {
        data.push(ModelInfo {
            object_type: "model",
            id: cloud,
            display_name: "Hera Cloud Fallback".to_string(),
            created_at: "2026-01-01T00:00:00Z".to_string(),
        });
    }

    Json(ModelListResponse {
        first_id: data.first().map(|m| m.id.clone()),
        last_id: data.last().map(|m| m.id.clone()),
        has_more: false,
        data,
    })
}

fn anthropic_to_chat_request(payload: &AnthropicMessageRequest) -> Result<ChatRequest, String> {
    let mut messages = Vec::new();

    if let Some(system) = &payload.system {
        let system_parts = anthropic_content_to_parts(system)?;
        messages.push(ChatMessage {
            role: "system".to_string(),
            content: system_parts,
        });
    }

    for msg in &payload.messages {
        messages.push(ChatMessage {
            role: normalize_role(&msg.role),
            content: anthropic_content_to_parts(&msg.content)?,
        });
    }

    if messages.is_empty() {
        return Err("messages must not be empty".to_string());
    }

    let _ = &payload.metadata;

    Ok(ChatRequest {
        model: payload.model.clone(),
        vision_model: None,
        tts_model: None,
        stt_model: None,
        messages,
        temperature: payload.temperature,
        max_tokens: Some(payload.max_tokens),
        top_p: payload.top_p,
        top_k: None,
        presence_penalty: None,
        frequency_penalty: None,
        repeat_penalty: None,
        seed: None,
        stop: payload.stop_sequences.clone(),
        endpoint: None,
        api_key: None,
        provider: None,
        stream: payload.stream,
        nsfw: None,
        tools: payload.tools.clone(),
        tool_choice: payload.tool_choice.clone(),
        reasoning_effort: None,
        response_format: None,
    })
}

fn anthropic_count_to_chat_request(payload: CountTokensRequest) -> Result<ChatRequest, String> {
    let mut messages = Vec::new();

    if let Some(system) = payload.system {
        messages.push(ChatMessage {
            role: "system".to_string(),
            content: anthropic_content_to_parts(&system)?,
        });
    }

    for msg in payload.messages {
        messages.push(ChatMessage {
            role: normalize_role(&msg.role),
            content: anthropic_content_to_parts(&msg.content)?,
        });
    }

    Ok(ChatRequest {
        model: "hera-local-model".to_string(),
        vision_model: None,
        tts_model: None,
        stt_model: None,
        messages,
        temperature: None,
        max_tokens: Some(1),
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
        stream: Some(false),
        nsfw: None,
        tools: payload.tools,
        tool_choice: None,
        reasoning_effort: None,
        response_format: None,
    })
}

fn anthropic_content_to_parts(content: &AnthropicContent) -> Result<MessageContent, String> {
    match content {
        AnthropicContent::Text(text) => Ok(MessageContent::Text(text.clone())),
        AnthropicContent::Blocks(blocks) => {
            let mut parts = Vec::new();
            let mut text_fallback = String::new();

            for block in blocks {
                match block.kind.as_str() {
                    "text" => {
                        if let Some(text) = &block.text {
                            parts.push(ContentPart::Text { text: text.clone() });
                            text_fallback.push_str(text);
                        }
                    }
                    "image" => {
                        let Some(source) = &block.source else {
                            return Err("image block missing source".to_string());
                        };
                        if source.kind != "base64" {
                            return Err(format!(
                                "unsupported anthropic image source type: {}",
                                source.kind
                            ));
                        }
                        let media_type = source
                            .media_type
                            .clone()
                            .unwrap_or_else(|| "image/png".to_string());
                        let data = source
                            .data
                            .clone()
                            .ok_or_else(|| "image block missing base64 data".to_string())?;
                        parts.push(ContentPart::ImageUrl {
                            image_url: ImageUrlContent {
                                url: format!("data:{media_type};base64,{data}"),
                            },
                        });
                    }
                    "tool_result" => {
                        if let Some(text) = &block.text {
                            parts.push(ContentPart::Text { text: text.clone() });
                            text_fallback.push_str(text);
                        } else if let Some(input) = &block.input {
                            let rendered =
                                serde_json::to_string(input).unwrap_or_else(|_| "{}".to_string());
                            parts.push(ContentPart::Text {
                                text: rendered.clone(),
                            });
                            text_fallback.push_str(&rendered);
                        }
                    }
                    "tool_use" => {
                        let tool_name = block.name.clone().unwrap_or_else(|| "tool".to_string());
                        let tool_input = block.input.clone().unwrap_or_else(|| json!({}));
                        let rendered = format!(
                            "<tool_call>\n{}\n</tool_call>",
                            serde_json::to_string(&json!({
                                "name": tool_name,
                                "arguments": tool_input
                            }))
                            .unwrap_or_else(|_| "{}".to_string())
                        );
                        parts.push(ContentPart::Text {
                            text: rendered.clone(),
                        });
                        text_fallback.push_str(&rendered);
                    }
                    _ => {}
                }
            }

            if !parts.is_empty() {
                Ok(MessageContent::Parts(parts))
            } else {
                Ok(MessageContent::Text(text_fallback))
            }
        }
    }
}

fn anthropic_message_response(
    requested_model: &str,
    req: &ChatRequest,
    resp: &crate::ai::ChatResponse,
) -> AnthropicMessageResponse {
    let content_text = resp
        .choices
        .first()
        .and_then(|choice| choice.message.content.clone())
        .unwrap_or_default();

    let mut content = vec![json!({
        "type": "text",
        "text": content_text,
    })];

    let tool_calls = resp
        .choices
        .first()
        .and_then(|choice| choice.message.tool_calls.clone())
        .unwrap_or_default();

    for (index, call) in tool_calls.iter().enumerate() {
        let id = call
            .get("id")
            .and_then(Value::as_str)
            .map(ToString::to_string)
            .or_else(|| {
                call.get("function")
                    .and_then(|v| v.get("name"))
                    .and_then(Value::as_str)
                    .map(|name| format!("toolu_{}_{}", name, index))
            })
            .unwrap_or_else(|| format!("toolu_{index}"));

        let (name, input) = if let Some(function) = call.get("function") {
            let name = function
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("tool")
                .to_string();
            let arguments = function
                .get("arguments")
                .and_then(Value::as_str)
                .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
                .unwrap_or_else(|| json!({}));
            (name, arguments)
        } else {
            let name = call
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("tool")
                .to_string();
            let arguments = call.get("arguments").cloned().unwrap_or_else(|| json!({}));
            (name, arguments)
        };

        content.push(json!({
            "type": "tool_use",
            "id": id,
            "name": name,
            "input": input
        }));
    }

    let usage = resp.usage.clone().unwrap_or_else(|| crate::ai::ChatUsage {
        prompt_tokens: estimate_tokens(req) as u32,
        completion_tokens: (content
            .iter()
            .filter_map(|block| block.get("text").and_then(Value::as_str))
            .map(|text| (text.len() / 4).max(1) as u32)
            .sum()),
        total_tokens: 0,
    });

    let stop_reason = if content
        .iter()
        .any(|block| block.get("type") == Some(&json!("tool_use")))
    {
        "tool_use".to_string()
    } else {
        normalize_stop_reason(
            resp.choices
                .first()
                .and_then(|choice| choice.finish_reason.as_deref()),
        )
        .to_string()
    };

    AnthropicMessageResponse {
        id: new_message_id(),
        object_type: "message",
        role: "assistant",
        content,
        model: if resp.model.is_empty() {
            requested_model.to_string()
        } else {
            resp.model.clone()
        },
        stop_reason,
        stop_sequence: None,
        usage: AnthropicUsage {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
        },
    }
}

fn normalize_role(role: &str) -> String {
    match role {
        "user" | "assistant" | "system" => role.to_string(),
        _ => "user".to_string(),
    }
}

fn clamp_chat_request(req: &mut ChatRequest, max_tokens: usize) {
    if estimate_tokens(req) <= max_tokens || req.messages.len() <= 2 {
        trim_chat_request_content(req, max_tokens);
        return;
    }

    let mut system_messages = Vec::new();
    let mut non_system_messages = Vec::new();

    for message in req.messages.drain(..) {
        if message.role == "system" {
            system_messages.push(message);
        } else {
            non_system_messages.push(message);
        }
    }

    let mut kept = Vec::new();
    for message in non_system_messages.into_iter().rev() {
        kept.push(message);

        let mut candidate = system_messages.clone();
        candidate.extend(kept.iter().rev().cloned());

        let candidate_req = ChatRequest {
            messages: candidate,
            ..req.clone()
        };

        if estimate_tokens(&candidate_req) > max_tokens {
            kept.pop();
            break;
        }
    }

    let mut final_messages = system_messages;
    final_messages.extend(kept.into_iter().rev());

    if final_messages.is_empty() && !req.messages.is_empty() {
        final_messages.push(req.messages[req.messages.len() - 1].clone());
    }

    req.messages = final_messages;
    trim_chat_request_content(req, max_tokens);
}

fn trim_chat_request_content(req: &mut ChatRequest, max_tokens: usize) {
    if estimate_tokens(req) <= max_tokens && total_message_chars(req) <= LOCAL_CHAR_SOFT_LIMIT {
        return;
    }

    let target_chars = LOCAL_CHAR_SOFT_LIMIT.min(max_tokens.saturating_mul(2));
    let current_chars = total_message_chars(req);
    if current_chars <= target_chars {
        return;
    }

    let excess = current_chars.saturating_sub(target_chars);

    for idx in 0..req.messages.len() {
        if total_message_chars(req) <= target_chars {
            break;
        }

        let is_last_user = idx + 1 == req.messages.len() && req.messages[idx].role == "user";
        if is_last_user {
            continue;
        }

        trim_message_content(&mut req.messages[idx], excess / 2 + 2048);
    }

    if total_message_chars(req) > target_chars {
        for idx in 0..req.messages.len() {
            if total_message_chars(req) <= target_chars {
                break;
            }
            trim_message_content(&mut req.messages[idx], 4096);
        }
    }
}

fn total_message_chars(req: &ChatRequest) -> usize {
    req.messages.iter().map(message_char_len).sum()
}

fn message_char_len(message: &ChatMessage) -> usize {
    match &message.content {
        MessageContent::Text(text) => text.len(),
        MessageContent::Parts(parts) => parts
            .iter()
            .map(|part| match part {
                ContentPart::Text { text } => text.len(),
                ContentPart::ImageUrl { image_url } => image_url.url.len(),
            })
            .sum(),
        MessageContent::Null => 0,
    }
}

fn trim_message_content(message: &mut ChatMessage, trim_chars: usize) {
    if trim_chars == 0 {
        return;
    }

    match &mut message.content {
        MessageContent::Text(text) => {
            *text = compact_text(text, trim_chars);
        }
        MessageContent::Parts(parts) => {
            for part in parts.iter_mut() {
                if let ContentPart::Text { text } = part {
                    *text = compact_text(text, trim_chars / 2 + 1024);
                }
            }
        }
        MessageContent::Null => {}
    }
}

fn compact_text(text: &str, trim_chars: usize) -> String {
    if text.len() <= trim_chars + 512 {
        return text.to_string();
    }

    let keep = text.len().saturating_sub(trim_chars);
    let head = keep / 2;
    let tail = keep.saturating_sub(head);
    let tail_start = text.len().saturating_sub(tail);

    format!(
        "{}\n\n[... context trimmed by Hera for local model limits ...]\n\n{}",
        &text[..head],
        &text[tail_start..]
    )
}

fn api_error(status: StatusCode, message: &str) -> (StatusCode, Json<Value>) {
    (
        status,
        Json(json!({
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": message
            }
        })),
    )
}

fn normalize_stop_reason(reason: Option<&str>) -> &'static str {
    match reason.unwrap_or("end_turn") {
        "stop" | "end_turn" => "end_turn",
        "tool_calls" | "tool_use" => "tool_use",
        "max_tokens" | "length" => "max_tokens",
        _ => "end_turn",
    }
}

async fn send_sse_event(tx: &tokio::sync::mpsc::Sender<Event>, event: &str, data: Value) {
    let _ = tx
        .send(
            Event::default()
                .event(event)
                .data(serde_json::to_string(&data).unwrap_or_else(|_| "{}".to_string())),
        )
        .await;
}

fn new_message_id() -> String {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    format!("msg_hera_{ts}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_anthropic_messages_to_chat_request() {
        let req = AnthropicMessageRequest {
            model: "test-model".to_string(),
            max_tokens: 256,
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: AnthropicContent::Blocks(vec![
                    AnthropicBlock {
                        kind: "text".to_string(),
                        text: Some("hello".to_string()),
                        source: None,
                        name: None,
                        input: None,
                    },
                    AnthropicBlock {
                        kind: "image".to_string(),
                        text: None,
                        source: Some(AnthropicImageSource {
                            kind: "base64".to_string(),
                            media_type: Some("image/png".to_string()),
                            data: Some("abcd".to_string()),
                        }),
                        name: None,
                        input: None,
                    },
                ]),
            }],
            system: Some(AnthropicContent::Text("sys".to_string())),
            metadata: None,
            stop_sequences: None,
            stream: Some(false),
            temperature: Some(0.2),
            top_p: None,
            tools: None,
            tool_choice: None,
        };

        let chat = anthropic_to_chat_request(&req).expect("conversion should succeed");
        assert_eq!(chat.model, "test-model");
        assert_eq!(chat.messages.len(), 2);
        assert_eq!(chat.messages[0].role, "system");
        assert_eq!(chat.messages[1].role, "user");

        match &chat.messages[1].content {
            MessageContent::Parts(parts) => {
                assert_eq!(parts.len(), 2);
            }
            other => panic!("expected multipart content, got {other:?}"),
        }
    }

    #[test]
    fn converts_tool_calls_to_anthropic_tool_use_blocks() {
        let req = ChatRequest {
            model: "req-model".to_string(),
            vision_model: None,
            tts_model: None,
            stt_model: None,
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: MessageContent::Text("hello".to_string()),
            }],
            temperature: None,
            max_tokens: Some(32),
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
            stream: Some(false),
            nsfw: None,
            tools: None,
            tool_choice: None,
            reasoning_effort: None,
            response_format: None,
        };

        let resp = crate::ai::ChatResponse {
            id: "chatcmpl_1".to_string(),
            object: "chat.completion".to_string(),
            created: 0,
            model: "resp-model".to_string(),
            choices: vec![crate::ai::ChatChoice {
                index: 0,
                message: crate::ai::ChatResponseMessage {
                    role: "assistant".to_string(),
                    content: Some("".to_string()),
                    tool_calls: Some(vec![json!({
                        "function": {
                            "name": "edit_file",
                            "arguments": "{\"path\":\"README.md\"}"
                        }
                    })]),
                },
                finish_reason: Some("tool_calls".to_string()),
            }],
            usage: Some(crate::ai::ChatUsage {
                prompt_tokens: 10,
                completion_tokens: 4,
                total_tokens: 14,
            }),
        };

        let anthropic = anthropic_message_response("requested", &req, &resp);
        assert_eq!(anthropic.stop_reason, "tool_use");
        assert_eq!(anthropic.content.len(), 2);
        assert_eq!(anthropic.content[1]["type"], "tool_use");
        assert_eq!(anthropic.content[1]["name"], "edit_file");
    }
}
