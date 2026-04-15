use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct LlmAuditEvent {
    pub ts_ms: u64,
    pub action: String,
    pub app: String,
    pub persona_path: String,
    pub prompt_preview: String,
    pub prompt_chars: usize,
    pub estimated_prompt_tokens: usize,
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
        return;
    };

    let path = audit_path();
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    let Ok(mut file) = OpenOptions::new().create(true).append(true).open(&path) else {
        return;
    };

    if serde_json::to_writer(&mut file, event).is_ok() {
        let _ = file.write_all(b"\n");
    }
}

pub fn build_event(
    action: &str,
    app: &str,
    persona_path: &str,
    prompt: &str,
    estimated_prompt_tokens: usize,
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
        persona_path: persona_path.to_string(),
        prompt_preview: prompt_preview(prompt),
        prompt_chars: prompt.len(),
        estimated_prompt_tokens,
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
