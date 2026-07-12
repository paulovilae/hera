//! Handler: transcribe_audio (Whisper STT).

use super::types::{HandlerOutcome, IpcPayload, IpcState};
use std::path::Path;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

const WHISPER_RESTART_MARKER: &str = "/tmp/hera-whisper-restart.timestamp";
const WHISPER_RESTART_COOLDOWN_SECS: u64 = 90;

fn whisper_enabled() -> bool {
    std::env::var("HERA_ENABLE_WHISPER")
        .ok()
        .map(|value| {
            matches!(
                value.to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(true)
}

fn maybe_request_hera_restart() -> bool {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);

    if let Ok(previous) = std::fs::read_to_string(WHISPER_RESTART_MARKER)
        && let Ok(previous_ts) = previous.trim().parse::<u64>()
        && now.saturating_sub(previous_ts) < WHISPER_RESTART_COOLDOWN_SECS
    {
        return false;
    }

    if let Some(parent) = Path::new(WHISPER_RESTART_MARKER).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(WHISPER_RESTART_MARKER, now.to_string());

    match Command::new("pm2").args(["restart", "hera-core"]).spawn() {
        Ok(_) => true,
        Err(err) => {
            tracing::error!(
                "Failed to request hera-core restart for Whisper recovery: {}",
                err
            );
            false
        }
    }
}

/// Handle the "transcribe_audio" action — audio-to-text via Whisper.
pub async fn handle_transcribe_audio(request: &IpcPayload, state: &IpcState) -> HandlerOutcome {
    let b64 = match request.payload.get("base64_audio").and_then(|p| p.as_str()) {
        Some(b) => b,
        None => {
            return HandlerOutcome::Result {
                result_text: "Missing base64_audio".to_string(),
                origin: "unknown".to_string(),
                model: String::new(),
                tool_calls: None,
            };
        }
    };

    // Optional per-request language hint — e.g. "es", "en". When absent or "auto",
    // the engine falls back to HERA_WHISPER_LANGUAGE / auto-detect.
    let lang_hint: Option<String> = request
        .payload
        .get("language")
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty());

    if let Some(ref lang) = lang_hint {
        tracing::debug!(language = %lang, "STT: per-request language hint received");
    }

    let result_text = if let Some(whisper) = &state.whisper_engine {
        use base64::{Engine as _, engine::general_purpose};
        match general_purpose::STANDARD.decode(b64) {
            Ok(audio_bytes) => match whisper.transcribe_audio(&audio_bytes, lang_hint.as_deref()).await {
                Ok(txt) => {
                    if txt.trim().is_empty() {
                        "I couldn't understand the audio clearly. Please try again.".to_string()
                    } else {
                        txt
                    }
                }
                Err(e) => {
                    tracing::error!("Audio inference error: {}", e);
                    format!("Error: {}", e)
                }
            },
            Err(_) => "Error: Invalid base64 audio payload.".to_string(),
        }
    } else if !whisper_enabled() {
        "Hera Audio Engine (Whisper) is disabled by configuration (HERA_ENABLE_WHISPER=false). Enable it and restart hera-core.".to_string()
    } else if maybe_request_hera_restart() {
        "Hera Audio Engine (Whisper) is unavailable. Requested a background restart of hera-core; retry in a few seconds.".to_string()
    } else {
        "Hera Audio Engine (Whisper) is unavailable. A recovery restart was already requested recently; retry in a few seconds.".to_string()
    };

    HandlerOutcome::Result {
        result_text,
        origin: "unknown".to_string(),
        model: String::new(),
        tool_calls: None,
    }
}
