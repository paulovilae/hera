//! IPC context management — DRY system prompt builder, payload parsing, context compression.
//!
//! Consolidates the system prompt construction logic that was previously duplicated
//! 3 times across generate and generate_stream handlers.

use std::sync::Arc;

use crate::ai::{ChatMessage, ChatRequest, ContentPart, LLMEngine, MessageContent};
use super::helpers::{fetch_db_schema_context, fetch_semantic_memory};

/// Extracted data from the IPC payload.
pub struct ParsedPayload {
    pub prompt: String,
    pub assistant_last: Option<String>,
    pub permissions: Vec<String>,
    pub persona_path: String,
    pub app_name: String,
}

/// Default persona path when none is provided.
const DEFAULT_PERSONA: &str =
    "/home/paulo/Programs/apps/imaginos/imaginclaw/persona/SOUL.md";

/// Extract prompt, assistant_last, permissions, persona_path, and app_name from payload.
pub fn parse_payload(payload: &serde_json::Value) -> ParsedPayload {
    let mut prompt = payload
        .get("prompt")
        .and_then(|p| p.as_str())
        .unwrap_or("")
        .to_string();
    let mut assistant_last: Option<String> = None;

    // Extract prompt from messages array if not provided directly
    if prompt.is_empty() {
        if let Some(messages) = payload.get("messages").and_then(|m| m.as_array()) {
            if let Some(last_msg) = messages.last()
                && let Some("user") = last_msg.get("role").and_then(|r| r.as_str())
                && let Some(content) = last_msg.get("content").and_then(|c| c.as_str())
            {
                prompt = content.to_string();
            }

            // Extract the second-to-last message if it's from the assistant
            if messages.len() >= 2
                && let Some(prev_msg) = messages.get(messages.len() - 2)
                && let Some("assistant") = prev_msg.get("role").and_then(|r| r.as_str())
                && let Some(content) = prev_msg.get("content").and_then(|c| c.as_str())
            {
                assistant_last = Some(content.to_string());
            }
        }
    }

    let permissions: Vec<String> = payload
        .get("permissions")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect::<Vec<String>>()
        })
        .unwrap_or_else(|| vec!["all".to_string()]);

    let persona_path = payload
        .get("persona_path")
        .and_then(|v| v.as_str())
        .unwrap_or(DEFAULT_PERSONA)
        .to_string();

    let app_name = payload
        .get("app")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    ParsedPayload {
        prompt,
        assistant_last,
        permissions,
        persona_path,
        app_name,
    }
}

/// Build the full system prompt (persona + memento + tool schemas + DB schemas + directives).
///
/// This is the single DRY function replacing 3 copy-pasted blocks across generate/stream.
pub async fn build_full_system_prompt(
    persona_path: &str,
    app_name: &str,
    permissions: &[String],
) -> String {
    let memento_ctx = fetch_semantic_memory(app_name).await;
    let base_system_prompt = format!(
        "{}{}",
        std::fs::read_to_string(persona_path)
            .unwrap_or_else(|_| "You are an AI assistant.".to_string()),
        memento_ctx
    );

    let agent_identity = std::path::Path::new(persona_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    let schemas = crate::ai::tool_executor::hera_tool_schemas(permissions, agent_identity);
    let db_schema_ctx = fetch_db_schema_context(agent_identity, app_name).await;

    let think_directive = "\n\nCRITICAL INSTRUCTION (INFERENCE-TIME RECALL): Before providing your final answer, you MUST systematically write out your internal reasoning step-by-step within <think> and </think> tags. Use this space to explore associations, reverse the question context, and search your internal knowledge to maximize factual recall. Do not output the final answer until after the </think> tag.";
    let json_directive = "\nCRITICAL TOOL RULE: If you decide to execute a tool, your ENTIRE response MUST be ONLY the raw JSON tool call. DO NOT write conversational text or explanations before or after the JSON tool block. The UI stream will crash if text and code logic bleed together.";

    format!(
        "{}\n\nCRITICAL RULE: DO NOT use tools to answer general conversational or conceptual questions like 'explain X' or 'what is Y'. If the user asks for an explanation or text-based answer, DO NOT build scripts or charts unless explicitly asked. ONLY use tools when the user explicitly requests code execution, file reading, or specific outputs.\n\n{}{}{}{}",
        base_system_prompt,
        schemas,
        db_schema_ctx,
        think_directive,
        json_directive
    )
}

/// Build a new ChatRequest from scratch (when payload isn't a valid ChatRequest).
pub fn build_new_chat_request(prompt: &str, system_prompt: String) -> ChatRequest {
    ChatRequest {
        model: "hera-local-model".to_string(),
        vision_model: None,
        tts_model: None,
        stt_model: None,
        messages: vec![
            ChatMessage {
                role: "system".to_string(),
                content: MessageContent::Text(system_prompt),
            },
            ChatMessage {
                role: "user".to_string(),
                content: MessageContent::Text(prompt.to_string()),
            },
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
    }
}

/// Inject the full system prompt into an existing ChatRequest's first message.
pub fn inject_system_prompt(req: &mut ChatRequest, full_system_prompt: String) {
    if let Some(first) = req.messages.first_mut() {
        if first.role == "system" {
            match &mut first.content {
                MessageContent::Text(t) => {
                    *t = format!("{}\n\n{}", full_system_prompt, t);
                }
                MessageContent::Parts(parts) => {
                    parts.insert(
                        0,
                        ContentPart::Text {
                            text: format!("{}\n\n", full_system_prompt),
                        },
                    );
                }
                MessageContent::Null => {
                    first.content = MessageContent::Text(full_system_prompt);
                }
            }
        } else {
            req.messages.insert(
                0,
                ChatMessage {
                    role: "system".to_string(),
                    content: MessageContent::Text(full_system_prompt),
                },
            );
        }
    } else {
        req.messages.push(ChatMessage {
            role: "system".to_string(),
            content: MessageContent::Text(full_system_prompt),
        });
    }
}

/// Compress chat history if it exceeds 24k tokens (safety buffer for 32k engine limit).
pub async fn compress_if_needed(
    req: &mut ChatRequest,
    engine: &Arc<dyn LLMEngine + Send + Sync>,
) {
    let est_tokens = super::helpers::estimate_tokens(req);
    if est_tokens <= 24000 || req.messages.len() <= 6 {
        return;
    }

    tracing::warn!(
        "📦 [Hera Context] COMPRESSING — {} tokens estimated (~{} msgs). Threshold: 24k. Condensing old history...",
        est_tokens,
        req.messages.len()
    );

    // Keep the system prompt
    let sys = req.messages.remove(0);

    // Keep the last 4 messages (2 complete dialog turns)
    let keep_start = req.messages.len().saturating_sub(4);
    let recent = req.messages.drain(keep_start..).collect::<Vec<_>>();

    // The remaining messages are the "old history"
    let old_history = std::mem::take(&mut req.messages);

    // Build text to compress
    let text_to_compress = old_history
        .iter()
        .map(|m| match &m.content {
            MessageContent::Text(t) => format!("{}: {}", m.role, t),
            MessageContent::Parts(p) => {
                let mut s = String::new();
                for part in p {
                    if let ContentPart::Text { text } = part {
                        s.push_str(text);
                    }
                }
                format!("{}: {}", m.role, s)
            }
            MessageContent::Null => String::new(),
        })
        .collect::<Vec<String>>()
        .join("\n");

    // Create compression request
    let mut compression_req = req.clone();
    compression_req.temperature = Some(0.3);
    compression_req.messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: MessageContent::Text(
                "You are an expert context condenser. Summarize the following overlapping technical conversation in dense detail. Preserve code snippets, user requirements, and technical facts. Do not output anything but the summary. Do not use markdown blocks for the whole thing.".to_string(),
            ),
        },
        ChatMessage {
            role: "user".to_string(),
            content: MessageContent::Text(text_to_compress),
        },
    ];

    match engine.generate_content(compression_req).await {
        Ok(summary_resp) => {
            let compress_origin = super::helpers::infer_origin_from_model(&summary_resp.model);
            if let Some(c) = summary_resp.choices.first() {
                if let Some(summary_txt) = &c.message.content {
                    let new_est = summary_txt.len() / 4;
                    tracing::info!(
                        "✅ [Hera Context] COMPRESSION DONE — {} old msgs → {} chars (~{} tokens). Engine: {} ({})",
                        old_history.len(),
                        summary_txt.len(),
                        new_est,
                        summary_resp.model,
                        compress_origin
                    );
                    req.messages.push(sys);
                    req.messages.push(ChatMessage {
                        role: "assistant".to_string(),
                        content: MessageContent::Text(format!(
                            "[SYSTEM: Previously condensed history]\n{}",
                            summary_txt
                        )),
                    });
                    req.messages.extend(recent);
                    return;
                }
            }
            // Fallback: reassemble if summary had no content
            tracing::warn!("⚠️ [Hera Context] Compression returned EMPTY response. Reassembling {} original messages.", old_history.len());
            req.messages.push(sys);
            req.messages.extend(old_history);
            req.messages.extend(recent);
        }
        Err(e) => {
            tracing::error!(
                "❌ [Hera Context] Compression FAILED: {}. Sending {} msgs uncompressed (~{} tokens).",
                e, old_history.len(), est_tokens
            );
            req.messages.push(sys);
            req.messages.extend(old_history);
            req.messages.extend(recent);
        }
    }
}
