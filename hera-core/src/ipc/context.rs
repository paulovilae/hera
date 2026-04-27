//! IPC context management — DRY system prompt builder, payload parsing, context compression.
//!
//! Consolidates the system prompt construction logic that was previously duplicated
//! 3 times across generate and generate_stream handlers.

use std::sync::Arc;

use super::helpers::{fetch_db_schema_context, fetch_runtime_preflight, fetch_semantic_memory};
use super::llm_audit::{LlmAuditEvent, build_event};
use super::route_profiles::resolve_route_profile;
use crate::ai::{ChatMessage, ChatRequest, ContentPart, LLMEngine, MessageContent};

#[derive(Debug, Clone)]
pub struct ContextBudget {
    pub mode: String,
    pub include_memory: bool,
    pub include_tool_schemas: bool,
    pub include_db_schema: bool,
    pub allow_tools: bool,
    pub max_memory_chars: usize,
    pub max_tool_schema_chars: usize,
    pub max_db_schema_chars: usize,
    pub max_history_messages: usize,
    pub compression_trigger_tokens: usize,
}

#[derive(Debug, Clone)]
pub struct PromptAssembly {
    pub system_prompt: String,
    pub memory_chars: usize,
    pub tool_schema_chars: usize,
    pub db_schema_chars: usize,
}

pub struct RuntimeOutcomeArtifacts {
    pub audit_event: LlmAuditEvent,
    pub observation_payload: serde_json::Value,
}

pub struct PreparedRuntimeContext {
    pub runtime_preflight: Option<serde_json::Value>,
    pub prompt_assembly: PromptAssembly,
    pub lightweight_mode: bool,
}

/// Extracted data from the IPC payload.
pub struct ParsedPayload {
    pub prompt: String,
    pub assistant_last: Option<String>,
    pub recent_messages: Vec<(String, String)>,
    pub permissions: Vec<String>,
    pub persona_path: String,
    pub app_name: String,
    pub language_hint: String,
    pub trace_id: String,
    pub session_id: String,
    pub chat_id: String,
    pub route_profile_id: String,
    pub expected_persona_path: String,
    pub persona_drift: bool,
    pub context_budget: ContextBudget,
}

pub fn apply_runtime_preflight(
    parsed: &mut ParsedPayload,
    preflight: Option<&serde_json::Value>,
    caller_overrode_budget: bool,
    lightweight_mode: bool,
) {
    let Some(preflight) = preflight else {
        return;
    };

    if !caller_overrode_budget
        && let Some(mode) = preflight
            .get("recommended_budget_mode")
            .and_then(|value| value.as_str())
    {
        parsed.context_budget = context_budget_for_mode(mode, lightweight_mode);
    }

    if let Some(warnings) = preflight.get("warnings").and_then(|value| value.as_array()) {
        for warning in warnings.iter().filter_map(|value| value.as_str()) {
            tracing::warn!(
                app = %parsed.app_name,
                route_profile = %parsed.route_profile_id,
                "Memento runtime preflight warning: {}",
                warning
            );
        }
    }
}

pub async fn prepare_runtime_execution_context(
    parsed: &mut ParsedPayload,
    caller_overrode_budget: bool,
    mode: &str,
) -> PreparedRuntimeContext {
    let lightweight_mode = is_lightweight_conversation(&parsed.prompt);
    let runtime_preflight = fetch_runtime_preflight(
        &parsed.app_name,
        &parsed.route_profile_id,
        &parsed.persona_path,
        mode,
    )
    .await;
    apply_runtime_preflight(
        parsed,
        runtime_preflight.as_ref(),
        caller_overrode_budget,
        lightweight_mode,
    );
    let prompt_assembly = build_full_system_prompt(
        &parsed.persona_path,
        &parsed.app_name,
        &parsed.permissions,
        &parsed.context_budget,
        lightweight_mode,
        &parsed.language_hint,
    )
    .await;

    PreparedRuntimeContext {
        runtime_preflight,
        prompt_assembly,
        lightweight_mode,
    }
}

pub async fn prepare_chat_request(
    payload: &serde_json::Value,
    prompt: &str,
    parsed: &ParsedPayload,
    prompt_assembly: &PromptAssembly,
    engine: &Arc<dyn LLMEngine + Send + Sync>,
) -> Option<ChatRequest> {
    let mut chat_req: Option<ChatRequest> = serde_json::from_value(payload.clone()).ok();

    if chat_req.is_none() {
        if !prompt.is_empty() {
            chat_req = Some(build_new_chat_request(
                prompt,
                prompt_assembly.system_prompt.clone(),
            ));
        }
    } else if let Some(req) = &mut chat_req {
        inject_system_prompt(req, prompt_assembly.system_prompt.clone());
    }

    if let Some(req) = &mut chat_req {
        apply_history_budget(req, &parsed.context_budget);
        compress_if_needed(req, engine, &parsed.context_budget).await;
    }

    chat_req
}

pub fn prepare_tool_result_followup_request(
    base_request: Option<ChatRequest>,
    assistant_output: &str,
    execution_outputs: &str,
    json_mode: bool,
) -> Option<ChatRequest> {
    let mut req2 = base_request?;

    if let Some(first) = req2.messages.first_mut()
        && first.role == "system"
    {
        first.content = MessageContent::Text(
            "You are a helpful AI assistant. You have already executed tools and received the results. Your ONLY job now is to summarize the results for the user. DO NOT output any tool calls, <tool_call> tags, or function calls. DO NOT use <think> tags. Output ONLY the final answer.".to_string(),
        );
    }

    req2.messages.push(ChatMessage {
        role: "assistant".to_string(),
        content: MessageContent::Text(assistant_output.to_string()),
    });

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

    Some(req2)
}

pub fn build_runtime_observation_payload(
    parsed: &ParsedPayload,
    prompt_assembly: &PromptAssembly,
    duration_ms: u64,
    first_token_ms: Option<u64>,
    success: bool,
    origin: &str,
    model: &str,
    error: Option<&str>,
) -> serde_json::Value {
    let mut payload = serde_json::json!({
        "app_id": parsed.app_name,
        "route_profile": parsed.route_profile_id,
        "session_id": parsed.session_id,
        "chat_id": parsed.chat_id,
        "trace_id": parsed.trace_id,
        "persona_path": parsed.persona_path,
        "expected_persona_path": parsed.expected_persona_path,
        "persona_drift": parsed.persona_drift,
        "duration_ms": duration_ms,
        "first_token_ms": first_token_ms,
        "prompt_chars": parsed.prompt.len(),
        "tool_schema_chars": prompt_assembly.tool_schema_chars,
        "db_schema_chars": prompt_assembly.db_schema_chars,
        "memory_chars": prompt_assembly.memory_chars,
        "success": success,
        "origin": origin,
        "model": model,
        "recommended_budget_mode": parsed.context_budget.mode
    });

    if let Some(error) = error.filter(|value| !value.is_empty()) {
        payload["error"] = serde_json::Value::String(error.to_string());
    }

    payload
}

#[allow(clippy::too_many_arguments)]
pub fn build_runtime_outcome_artifacts(
    action: &str,
    parsed: &ParsedPayload,
    prompt_assembly: &PromptAssembly,
    duration_ms: u64,
    first_token_ms: Option<u64>,
    lightweight_mode: bool,
    provider_requested: &str,
    origin: &str,
    model: &str,
    success: bool,
    tool_call_count: usize,
    response_chars: usize,
    estimated_prompt_tokens: usize,
    prompt_history_messages: usize,
    error: Option<String>,
) -> RuntimeOutcomeArtifacts {
    let audit_event = build_event(
        action,
        &parsed.app_name,
        &parsed.route_profile_id,
        &parsed.trace_id,
        &parsed.session_id,
        &parsed.chat_id,
        &parsed.persona_path,
        &parsed.expected_persona_path,
        parsed.persona_drift,
        &parsed.context_budget.mode,
        prompt_history_messages,
        &parsed.prompt,
        estimated_prompt_tokens,
        prompt_assembly.memory_chars,
        prompt_assembly.tool_schema_chars,
        prompt_assembly.db_schema_chars,
        duration_ms,
        first_token_ms,
        lightweight_mode,
        provider_requested,
        origin,
        model,
        success,
        tool_call_count,
        response_chars,
        error.clone(),
    );
    let observation_payload = build_runtime_observation_payload(
        parsed,
        prompt_assembly,
        duration_ms,
        first_token_ms,
        success,
        origin,
        model,
        error.as_deref(),
    );

    RuntimeOutcomeArtifacts {
        audit_event,
        observation_payload,
    }
}

fn normalize_lightweight_prompt(prompt: &str) -> String {
    prompt
        .trim()
        .to_lowercase()
        .replace(['¡', '!', '¿', '?', '.', ',', ';', ':'], "")
}

pub fn is_lightweight_conversation(prompt: &str) -> bool {
    let normalized = normalize_lightweight_prompt(prompt);
    if normalized.is_empty() {
        return true;
    }

    let lightweight_messages = [
        "hola",
        "hola memo",
        "hola chepito",
        "buenas",
        "buenos dias",
        "buenos días",
        "buenas tardes",
        "buenas noches",
        "hey",
        "ey",
        "ola",
        "gracias",
        "muchas gracias",
        "ok gracias",
        "listo gracias",
        "quien eres",
        "quién eres",
        "que haces",
        "qué haces",
        "ayuda",
    ];

    lightweight_messages.contains(&normalized.as_str())
}

fn clamp_chars(value: String, max_chars: usize) -> String {
    if max_chars == 0 || value.len() <= max_chars {
        return value;
    }

    let mut truncated = String::new();
    for ch in value.chars().take(max_chars) {
        truncated.push(ch);
    }
    truncated.push_str("\n...[truncated by Hera context budget]");
    truncated
}

fn extract_optional_string(payload: &serde_json::Value, key: &str) -> String {
    payload
        .get(key)
        .and_then(|value| value.as_str())
        .unwrap_or("")
        .trim()
        .to_string()
}

pub(crate) fn extract_message_text(content: &serde_json::Value) -> Option<String> {
    if let Some(text) = content.as_str() {
        let trimmed = text.trim();
        return (!trimmed.is_empty()).then(|| trimmed.to_string());
    }

    if let Some(object) = content.as_object() {
        if object.get("type").and_then(|value| value.as_str()) == Some("text")
            && let Some(text) = object.get("text").and_then(|value| value.as_str())
        {
            let trimmed = text.trim();
            return (!trimmed.is_empty()).then(|| trimmed.to_string());
        }
        if let Some(text) = object.get("content").and_then(extract_message_text) {
            return Some(text);
        }
    }

    let parts = content.as_array()?;
    let joined = parts
        .iter()
        .filter_map(|part| {
            if part.get("type").and_then(|value| value.as_str()) == Some("text") {
                part.get("text").and_then(|value| value.as_str())
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    let trimmed = joined.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

pub(crate) fn latest_user_message_text(payload: &serde_json::Value) -> Option<String> {
    let messages = payload.get("messages").and_then(|value| value.as_array())?;
    messages.iter().rev().find_map(|message| {
        (message.get("role").and_then(|value| value.as_str()) == Some("user"))
            .then(|| message.get("content").and_then(extract_message_text))
            .flatten()
    })
}

pub fn context_budget_for_mode(mode: &str, lightweight_mode: bool) -> ContextBudget {
    if lightweight_mode {
        return ContextBudget {
            mode: "lightweight".to_string(),
            include_memory: false,
            include_tool_schemas: false,
            include_db_schema: false,
            allow_tools: false,
            max_memory_chars: 0,
            max_tool_schema_chars: 0,
            max_db_schema_chars: 0,
            max_history_messages: 6,
            compression_trigger_tokens: 12_000,
        };
    }

    match mode {
        "minimal" => ContextBudget {
            mode: "minimal".to_string(),
            include_memory: true,
            include_tool_schemas: false,
            include_db_schema: false,
            allow_tools: false,
            max_memory_chars: 1_200,
            max_tool_schema_chars: 0,
            max_db_schema_chars: 0,
            max_history_messages: 8,
            compression_trigger_tokens: 12_000,
        },
        "heavy" => ContextBudget {
            mode: "heavy".to_string(),
            include_memory: true,
            include_tool_schemas: true,
            include_db_schema: true,
            allow_tools: true,
            max_memory_chars: 4_000,
            max_tool_schema_chars: 24_000,
            max_db_schema_chars: 10_000,
            max_history_messages: 24,
            compression_trigger_tokens: 28_000,
        },
        _ => ContextBudget {
            mode: "standard".to_string(),
            include_memory: true,
            include_tool_schemas: true,
            include_db_schema: true,
            allow_tools: true,
            max_memory_chars: 2_400,
            max_tool_schema_chars: 14_000,
            max_db_schema_chars: 6_000,
            max_history_messages: 14,
            compression_trigger_tokens: 20_000,
        },
    }
}

fn parse_context_budget(payload: &serde_json::Value, lightweight_mode: bool) -> ContextBudget {
    let mode = payload
        .get("context_budget_mode")
        .and_then(|value| value.as_str())
        .unwrap_or(if lightweight_mode {
            "lightweight"
        } else {
            "standard"
        })
        .trim()
        .to_ascii_lowercase();
    let mut budget = context_budget_for_mode(&mode, lightweight_mode);

    if let Some(overrides) = payload
        .get("context_budget")
        .and_then(|value| value.as_object())
    {
        if let Some(value) = overrides
            .get("include_memory")
            .and_then(|value| value.as_bool())
        {
            budget.include_memory = value;
        }
        if let Some(value) = overrides
            .get("include_tool_schemas")
            .and_then(|value| value.as_bool())
        {
            budget.include_tool_schemas = value;
        }
        if let Some(value) = overrides
            .get("include_db_schema")
            .and_then(|value| value.as_bool())
        {
            budget.include_db_schema = value;
        }
        if let Some(value) = overrides
            .get("allow_tools")
            .and_then(|value| value.as_bool())
        {
            budget.allow_tools = value;
        }
        if let Some(value) = overrides
            .get("max_memory_chars")
            .and_then(|value| value.as_u64())
        {
            budget.max_memory_chars = value as usize;
        }
        if let Some(value) = overrides
            .get("max_tool_schema_chars")
            .and_then(|value| value.as_u64())
        {
            budget.max_tool_schema_chars = value as usize;
        }
        if let Some(value) = overrides
            .get("max_db_schema_chars")
            .and_then(|value| value.as_u64())
        {
            budget.max_db_schema_chars = value as usize;
        }
        if let Some(value) = overrides
            .get("max_history_messages")
            .and_then(|value| value.as_u64())
        {
            budget.max_history_messages = value as usize;
        }
        if let Some(value) = overrides
            .get("compression_trigger_tokens")
            .and_then(|value| value.as_u64())
        {
            budget.compression_trigger_tokens = value as usize;
        }
    }

    budget
}

/// Extract prompt, assistant_last, permissions, persona_path, and app_name from payload.
pub fn parse_payload(payload: &serde_json::Value) -> ParsedPayload {
    let mut prompt = payload
        .get("prompt")
        .and_then(|p| p.as_str())
        .unwrap_or("")
        .to_string();
    let mut assistant_last: Option<String> = None;
    let mut recent_messages: Vec<(String, String)> = Vec::new();

    // Extract prompt from messages array if not provided directly
    if prompt.is_empty() {
        if let Some(messages) = payload.get("messages").and_then(|m| m.as_array()) {
            recent_messages = messages
                .iter()
                .filter_map(|message| {
                    let role = message.get("role").and_then(|value| value.as_str())?;
                    let content = message.get("content").and_then(extract_message_text)?;
                    let trimmed = content.trim().to_string();
                    if trimmed.is_empty() {
                        return None;
                    }
                    Some((role.to_string(), trimmed))
                })
                .collect();
            let mut latest_user_index: Option<usize> = None;
            for (index, message) in messages.iter().enumerate().rev() {
                if let Some("user") = message.get("role").and_then(|value| value.as_str())
                    && let Some(content) = message.get("content").and_then(extract_message_text)
                {
                    prompt = content;
                    latest_user_index = Some(index);
                    break;
                }
            }

            if let Some(user_index) = latest_user_index {
                for message in messages[..user_index].iter().rev() {
                    if let Some("assistant") = message.get("role").and_then(|value| value.as_str())
                        && let Some(content) = message.get("content").and_then(extract_message_text)
                    {
                        assistant_last = Some(content);
                        break;
                    }
                }
            }
        }
    }

    let lightweight_mode = is_lightweight_conversation(&prompt);
    let permissions: Vec<String> = payload
        .get("permissions")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect::<Vec<String>>()
        })
        .unwrap_or_else(|| vec!["all".to_string()]);

    let app_name = payload
        .get("app")
        .or_else(|| payload.get("app_name"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let explicit_route_profile = payload
        .get("route_profile")
        .and_then(|value| value.as_str());
    let route_profile = resolve_route_profile(explicit_route_profile, &app_name);
    let persona_path = payload
        .get("persona_path")
        .and_then(|v| v.as_str())
        .unwrap_or(route_profile.persona_path)
        .to_string();
    let language_hint = payload
        .get("language_hint")
        .and_then(|v| v.as_str())
        .unwrap_or("auto")
        .trim()
        .to_ascii_lowercase();

    let trace_id = extract_optional_string(payload, "trace_id");
    let session_id = extract_optional_string(payload, "session_id");
    let chat_id = extract_optional_string(payload, "chat_id");
    let context_budget = if payload.get("context_budget_mode").is_some()
        || payload.get("context_budget").is_some()
    {
        parse_context_budget(payload, lightweight_mode)
    } else {
        context_budget_for_mode(route_profile.default_context_budget_mode, lightweight_mode)
    };

    let persona_drift = persona_path != route_profile.persona_path;

    ParsedPayload {
        prompt,
        assistant_last,
        recent_messages,
        permissions,
        persona_path,
        app_name,
        language_hint,
        trace_id,
        session_id,
        chat_id,
        route_profile_id: route_profile.id.to_string(),
        expected_persona_path: route_profile.persona_path.to_string(),
        persona_drift,
        context_budget,
    }
}

/// Build the full system prompt (persona + memento + tool schemas + DB schemas + directives).
///
/// This is the single DRY function replacing 3 copy-pasted blocks across generate/stream.
pub async fn build_full_system_prompt(
    persona_path: &str,
    app_name: &str,
    permissions: &[String],
    budget: &ContextBudget,
    lightweight_mode: bool,
    language_hint: &str,
) -> PromptAssembly {
    let memento_ctx = if budget.include_memory {
        clamp_chars(
            fetch_semantic_memory(app_name).await,
            budget.max_memory_chars,
        )
    } else {
        String::new()
    };
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

    let schemas = if lightweight_mode || !budget.include_tool_schemas || !budget.allow_tools {
        String::new()
    } else {
        clamp_chars(
            crate::ai::tool_executor::hera_tool_schemas(permissions, agent_identity),
            budget.max_tool_schema_chars,
        )
    };
    let db_schema_ctx = if lightweight_mode || !budget.include_db_schema {
        String::new()
    } else {
        clamp_chars(
            fetch_db_schema_context(agent_identity, app_name).await,
            budget.max_db_schema_chars,
        )
    };

    let think_directive = if lightweight_mode {
        "\n\nRespond naturally and briefly. Do not use tools. Do not use <think> tags."
    } else {
        "\n\nCRITICAL INSTRUCTION (INFERENCE-TIME RECALL): Before providing your final answer, you MUST systematically write out your internal reasoning step-by-step within <think> and </think> tags. Use this space to explore associations, reverse the question context, and search your internal knowledge to maximize factual recall. Do not output the final answer until after the </think> tag."
    };
    let json_directive = if lightweight_mode {
        ""
    } else {
        "\nCRITICAL TOOL RULE: If you decide to execute a tool, your ENTIRE response MUST be EXACTLY this format, with NO conversational text: <tool_call>{\"name\": \"function_name\", \"arguments\": {\"arg1\": \"val\"}}</tool_call>"
    };
    let language_directive = match language_hint {
        "es" => "\nLANGUAGE RULE: Respond in Spanish unless the user explicitly switches language.",
        "en" => "\nLANGUAGE RULE: Respond in English unless the user explicitly switches language.",
        "pt" => {
            "\nLANGUAGE RULE: Respond in Portuguese unless the user explicitly switches language."
        }
        _ => {
            "\nLANGUAGE RULE: Respond in the same language used by the user in their latest message. If the message is mixed, prefer the dominant language and keep the answer consistent."
        }
    };

    let system_prompt = format!(
        "{}\n\nCRITICAL RULE: DO NOT use tools to answer general conversational or conceptual questions like 'explain X' or 'what is Y'. If the user asks for an explanation or text-based answer, DO NOT build scripts or charts unless explicitly asked. ONLY use tools when the user explicitly requests code execution, file reading, or specific outputs.\n\n{}{}{}{}{}",
        base_system_prompt,
        schemas,
        db_schema_ctx,
        think_directive,
        json_directive,
        language_directive
    );

    PromptAssembly {
        system_prompt,
        memory_chars: memento_ctx.len(),
        tool_schema_chars: schemas.len(),
        db_schema_chars: db_schema_ctx.len(),
    }
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

#[cfg(test)]
mod tests {
    use super::{latest_user_message_text, parse_payload};

    #[test]
    fn parse_payload_uses_latest_user_even_if_not_last_message() {
        let payload = serde_json::json!({
            "messages": [
                {"role": "system", "content": "persona"},
                {"role": "assistant", "content": "prev"},
                {"role": "user", "content": [{"type": "text", "text": "/restart imaginclaw"}]},
                {"role": "system", "content": "app context"}
            ]
        });

        let parsed = parse_payload(&payload);

        assert_eq!(parsed.prompt, "/restart imaginclaw");
        assert_eq!(parsed.assistant_last.as_deref(), Some("prev"));
    }

    #[test]
    fn latest_user_message_text_finds_explicit_command() {
        let payload = serde_json::json!({
            "messages": [
                {"role": "assistant", "content": "old"},
                {"role": "user", "content": [{"type": "text", "text": "/status"}]},
                {"role": "assistant", "content": "later"}
            ]
        });

        assert_eq!(
            latest_user_message_text(&payload).as_deref(),
            Some("/status")
        );
    }

    #[test]
    fn latest_user_message_text_supports_object_text_content() {
        let payload = serde_json::json!({
            "messages": [
                {"role": "user", "content": {"type": "text", "text": "/restart imaginclaw"}}
            ]
        });

        assert_eq!(
            latest_user_message_text(&payload).as_deref(),
            Some("/restart imaginclaw")
        );
    }
}

pub fn apply_history_budget(req: &mut ChatRequest, budget: &ContextBudget) {
    let max_history_messages = budget.max_history_messages.max(2);
    if req.messages.len() <= max_history_messages {
        return;
    }

    let system_message = req
        .messages
        .first()
        .filter(|message| message.role == "system")
        .cloned();
    let mut non_system = req
        .messages
        .iter()
        .filter(|message| message.role != "system")
        .cloned()
        .collect::<Vec<_>>();

    if non_system.len() > max_history_messages.saturating_sub(system_message.is_some() as usize) {
        let keep = max_history_messages.saturating_sub(system_message.is_some() as usize);
        let start = non_system.len().saturating_sub(keep);
        non_system = non_system.split_off(start);
    }

    req.messages.clear();
    if let Some(system_message) = system_message {
        req.messages.push(system_message);
    }
    req.messages.extend(non_system);
}

/// Compress chat history if it exceeds 24k tokens (safety buffer for 32k engine limit).
pub async fn compress_if_needed(
    req: &mut ChatRequest,
    engine: &Arc<dyn LLMEngine + Send + Sync>,
    budget: &ContextBudget,
) {
    let est_tokens = super::helpers::estimate_tokens(req);
    if est_tokens <= budget.compression_trigger_tokens || req.messages.len() <= 6 {
        return;
    }

    tracing::warn!(
        "📦 [Hera Context] COMPRESSING — {} tokens estimated (~{} msgs). Threshold: {}. Condensing old history...",
        est_tokens,
        req.messages.len(),
        budget.compression_trigger_tokens
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
            tracing::warn!(
                "⚠️ [Hera Context] Compression returned EMPTY response. Reassembling {} original messages.",
                old_history.len()
            );
            req.messages.push(sys);
            req.messages.extend(old_history);
            req.messages.extend(recent);
        }
        Err(e) => {
            tracing::error!(
                "❌ [Hera Context] Compression FAILED: {}. Sending {} msgs uncompressed (~{} tokens).",
                e,
                old_history.len(),
                est_tokens
            );
            req.messages.push(sys);
            req.messages.extend(old_history);
            req.messages.extend(recent);
        }
    }
}
