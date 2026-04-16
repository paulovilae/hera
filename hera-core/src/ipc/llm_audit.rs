use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmAuditEvent {
    pub ts_ms: u64,
    pub action: String,
    pub app: String,
    pub route_profile: String,
    pub trace_id: String,
    pub session_id: String,
    pub chat_id: String,
    pub persona_path: String,
    pub expected_persona_path: String,
    pub persona_drift: bool,
    pub context_budget_mode: String,
    pub prompt_history_messages: usize,
    pub prompt_preview: String,
    pub prompt_chars: usize,
    pub estimated_prompt_tokens: usize,
    pub memory_chars: usize,
    pub tool_schema_chars: usize,
    pub db_schema_chars: usize,
    pub duration_ms: u64,
    pub first_token_ms: Option<u64>,
    pub lightweight_mode: bool,
    pub provider_requested: String,
    pub origin: String,
    pub model: String,
    pub success: bool,
    pub tool_call_count: usize,
    pub response_chars: usize,
    pub error: Option<String>,
}

fn audit_path() -> PathBuf {
    PathBuf::from("/tmp/hera_llm_audit.jsonl")
}

fn write_lock() -> &'static Mutex<()> {
    static WRITE_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    WRITE_LOCK.get_or_init(|| Mutex::new(()))
}

fn now_epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

pub fn prompt_preview(prompt: &str) -> String {
    let trimmed = prompt.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    let mut preview = trimmed.replace('\n', " ");
    if preview.len() > 160 {
        preview.truncate(160);
    }
    preview
}

pub fn append_llm_audit_event(event: &LlmAuditEvent) {
    let Ok(_guard) = write_lock().lock() else {
        tracing::error!("LLM audit write lock poisoned");
        return;
    };

    let path = audit_path();
    if let Some(parent) = path.parent() {
        if let Err(error) = fs::create_dir_all(parent) {
            tracing::error!(
                "Failed to create LLM audit directory {:?}: {}",
                parent,
                error
            );
            return;
        }
    }

    let Ok(mut file) = OpenOptions::new().create(true).append(true).open(&path) else {
        tracing::error!("Failed to open LLM audit log at {:?}", path);
        return;
    };

    if let Err(error) = serde_json::to_writer(&mut file, event) {
        tracing::error!("Failed to serialize LLM audit event: {}", error);
        return;
    }
    if let Err(error) = file.write_all(b"\n") {
        tracing::error!("Failed to finalize LLM audit event write: {}", error);
    }
}

pub fn build_event(
    action: &str,
    app: &str,
    route_profile: &str,
    trace_id: &str,
    session_id: &str,
    chat_id: &str,
    persona_path: &str,
    expected_persona_path: &str,
    persona_drift: bool,
    context_budget_mode: &str,
    prompt_history_messages: usize,
    prompt: &str,
    estimated_prompt_tokens: usize,
    memory_chars: usize,
    tool_schema_chars: usize,
    db_schema_chars: usize,
    duration_ms: u64,
    first_token_ms: Option<u64>,
    lightweight_mode: bool,
    provider_requested: &str,
    origin: &str,
    model: &str,
    success: bool,
    tool_call_count: usize,
    response_chars: usize,
    error: Option<String>,
) -> LlmAuditEvent {
    LlmAuditEvent {
        ts_ms: now_epoch_ms(),
        action: action.to_string(),
        app: if app.is_empty() {
            "unknown".to_string()
        } else {
            app.to_string()
        },
        route_profile: route_profile.to_string(),
        trace_id: trace_id.to_string(),
        session_id: session_id.to_string(),
        chat_id: chat_id.to_string(),
        persona_path: persona_path.to_string(),
        expected_persona_path: expected_persona_path.to_string(),
        persona_drift,
        context_budget_mode: context_budget_mode.to_string(),
        prompt_history_messages,
        prompt_preview: prompt_preview(prompt),
        prompt_chars: prompt.len(),
        estimated_prompt_tokens,
        memory_chars,
        tool_schema_chars,
        db_schema_chars,
        duration_ms,
        first_token_ms,
        lightweight_mode,
        provider_requested: provider_requested.to_string(),
        origin: origin.to_string(),
        model: model.to_string(),
        success,
        tool_call_count,
        response_chars,
        error,
    }
}
