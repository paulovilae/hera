//! Cloud-agent escalation tool (Critical risk) — subprocess calls into Paulo's
//! ALREADY-authenticated cloud CLI sessions on genesis: Claude Code (Anthropic),
//! Grok (xAI), Antigravity `agy` (Gemini/Google), and Codex (OpenAI). Prompt text
//! in, response text out — this is a read-only/opinion consultation, never a
//! file-editing agent run. See `etc/SESSION_INTENT.md`
//! (`session-hera-escalate-cloud-agent`) for the design rationale and
//! `Tools/global/ai/escalate_to_cloud_agent.json` for the tool schema.
//!
//! Guardrails (do not weaken without operator sign-off):
//! - No `--dangerously-skip-permissions` / `--allow-dangerously-skip-permissions` /
//!   `--dangerously-bypass-approvals-and-sandbox` is ever passed to any of the 4
//!   CLIs.
//! - Each provider runs in its most restrictive documented read-only/plan mode
//!   (verified live against genesis, 2026-07-16):
//!     claude_code  → `--permission-mode plan --disallowedTools Bash,Edit,Write,NotebookEdit`
//!     grok         → `--permission-mode plan --tools ""`
//!     gemini (agy) → `--mode plan --sandbox`
//!     openai_codex → `--sandbox read-only --skip-git-repo-check`
//! - `current_dir` is always `ESCALATE_SCRATCH_DIR`, a scratch `/tmp` directory
//!   OUTSIDE the real repo checkout — even a CLI bug that ignored the flags above
//!   would find no git repo and no real files to write to.
//! - `stdin` is closed (`Stdio::null()`) — some of these CLIs probe stdin even in
//!   non-interactive/print mode (observed with `codex exec`).
//! - Hard per-call timeout (`TIMEOUT_S`), enforced both here and by the outer
//!   dispatcher (`Tools/global/ai/escalate_to_cloud_agent.json`'s `timeout_ms`,
//!   kept slightly above this one so this module's clearer per-provider error
//!   message wins the race).
//! - File-backed daily cap per provider (`DAILY_CAP_FILE`), default
//!   `DEFAULT_DAILY_CAP`/day/provider, overridable via `HERA_ESCALATE_DAILY_CAP`
//!   (all providers) or `HERA_ESCALATE_DAILY_CAP_<PROVIDER>` (single provider).
//!   The cap is checked and incremented BEFORE the subprocess is spawned — a
//!   denied call never touches the network.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use tracing::info;

use crate::ai::tool_executor::security::extract_tool_caller;
use crate::ai::tool_executor::{ToolCall, ToolResult};

/// Subprocess timeout — kept below the JSON tool's `timeout_ms` (180000) so a
/// slow provider surfaces THIS module's readable message instead of the generic
/// "Tool execution timed out" from `tool_executor::dispatch::execute_tool`.
const TIMEOUT_S: u64 = 170;
const DEFAULT_DAILY_CAP: u32 = 15;
const DAILY_CAP_FILE: &str = "/tmp/hera_escalate_daily_caps.json";
/// Scratch working directory for every provider subprocess. Deliberately OUTSIDE
/// the real repo checkout (`/home/paulo/Programs/apps/OS` on the laptop,
/// `/mnt/workspace/Programs/apps/OS` on genesis) so none of the 4 CLIs — even if
/// they ignored the read-only/plan flags — could discover a git repo or real
/// project files to touch.
const ESCALATE_SCRATCH_DIR: &str = "/tmp/hera_escalate_scratch";

fn arg_str<'a>(call: &'a ToolCall, key: &str) -> &'a str {
    call.arguments.get(key).and_then(|v| v.as_str()).unwrap_or("")
}

fn ok(call: &ToolCall, output: String) -> ToolResult {
    ToolResult { name: call.name.clone(), success: true, output }
}

fn err(call: &ToolCall, output: impl Into<String>) -> ToolResult {
    ToolResult { name: call.name.clone(), success: false, output: output.into() }
}

/// Pure `YYYY-MM-DD` (UTC) from `SystemTime`, no `chrono` dependency needed just
/// for a daily-bucket key. Howard Hinnant's `civil_from_days` algorithm — always
/// fed a non-negative day count here (current wall-clock dates only).
fn today_utc_date_string() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = (secs / 86_400) as i64;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}

fn daily_cap_for(provider: &str) -> u32 {
    let per_provider_key = format!("HERA_ESCALATE_DAILY_CAP_{}", provider.to_ascii_uppercase());
    std::env::var(&per_provider_key)
        .ok()
        .and_then(|value| value.trim().parse::<u32>().ok())
        .or_else(|| {
            std::env::var("HERA_ESCALATE_DAILY_CAP")
                .ok()
                .and_then(|value| value.trim().parse::<u32>().ok())
        })
        .unwrap_or(DEFAULT_DAILY_CAP)
}

static CAP_FILE_LOCK: Mutex<()> = Mutex::new(());

/// Checks and atomically increments today's counter for `provider`. Returns
/// `Err(daily_cap_reached message)` WITHOUT incrementing when the cap is already
/// hit. Callers must treat `Err` as "do not execute" — fail closed, no bypass.
fn check_and_record_daily_cap(provider: &str) -> Result<u32, String> {
    let _guard = CAP_FILE_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let today = today_utc_date_string();
    let key = format!("{provider}:{today}");

    let mut counts = std::fs::read_to_string(DAILY_CAP_FILE)
        .ok()
        .and_then(|content| serde_json::from_str::<serde_json::Value>(&content).ok())
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default();

    let current = counts.get(&key).and_then(|value| value.as_u64()).unwrap_or(0) as u32;
    let cap = daily_cap_for(provider);
    if current >= cap {
        return Err(format!(
            "daily_cap_reached: provider='{provider}' ya alcanzó el tope diario ({current}/{cap}) para {today}. \
             Ajustable con HERA_ESCALATE_DAILY_CAP_{} o HERA_ESCALATE_DAILY_CAP (default {DEFAULT_DAILY_CAP}).",
            provider.to_ascii_uppercase()
        ));
    }

    counts.insert(key, serde_json::json!(current + 1));
    // Keep only today's entries — a cheap daily reset, keeps the file from growing forever.
    let today_suffix = format!(":{today}");
    counts.retain(|bucket_key, _| bucket_key.ends_with(&today_suffix));
    if let Ok(serialized) = serde_json::to_string_pretty(&serde_json::Value::Object(counts)) {
        let _ = std::fs::write(DAILY_CAP_FILE, serialized);
    }
    Ok(current + 1)
}

fn ensure_scratch_dir() -> std::io::Result<()> {
    std::fs::create_dir_all(ESCALATE_SCRATCH_DIR)
}

fn unique_scratch_file(prefix: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    Path::new(ESCALATE_SCRATCH_DIR).join(format!("{prefix}_{nanos}_{n}.txt"))
}

fn home_dir() -> String {
    std::env::var("HOME").unwrap_or_else(|_| "/home/paulo".to_string())
}

/// Prefer the known absolute install path (matches what Paulo verified by hand);
/// fall back to a bare name resolved via PATH if the fixed path is ever missing
/// (e.g. a future reinstall to a different prefix).
fn resolve_binary(fixed_path: String, fallback_name: &str) -> String {
    if Path::new(&fixed_path).exists() {
        fixed_path
    } else {
        fallback_name.to_string()
    }
}

fn grok_binary_path() -> String {
    resolve_binary(format!("{}/.grok/bin/grok", home_dir()), "grok")
}

fn agy_binary_path() -> String {
    resolve_binary(format!("{}/.local/bin/agy", home_dir()), "agy")
}

/// One resolved provider invocation: program + args, plus an optional file to
/// read the response from instead of stdout (codex's `exec` stdout is a noisy
/// banner; `-o/--output-last-message` gives a clean text file instead).
struct ProviderCommand {
    program: String,
    args: Vec<String>,
    read_output_from: Option<PathBuf>,
}

fn build_command(provider: &str, prompt: &str) -> Result<ProviderCommand, String> {
    match provider {
        "claude_code" => Ok(ProviderCommand {
            program: "npx".to_string(),
            args: vec![
                "-y".into(),
                "@anthropic-ai/claude-code".into(),
                "-p".into(),
                prompt.to_string(),
                "--permission-mode".into(),
                "plan".into(),
                "--disallowedTools".into(),
                "Bash,Edit,Write,NotebookEdit".into(),
            ],
            read_output_from: None,
        }),
        "grok" => Ok(ProviderCommand {
            program: grok_binary_path(),
            args: vec![
                "-p".into(),
                prompt.to_string(),
                "--permission-mode".into(),
                "plan".into(),
                "--cwd".into(),
                ESCALATE_SCRATCH_DIR.into(),
                "--tools".into(),
                "".into(),
            ],
            read_output_from: None,
        }),
        "gemini" => Ok(ProviderCommand {
            program: agy_binary_path(),
            args: vec![
                "-p".into(),
                prompt.to_string(),
                "--mode".into(),
                "plan".into(),
                "--sandbox".into(),
            ],
            read_output_from: None,
        }),
        "openai_codex" => {
            let out_file = unique_scratch_file("codex_out");
            Ok(ProviderCommand {
                program: "npx".to_string(),
                args: vec![
                    "-y".into(),
                    "@openai/codex".into(),
                    "exec".into(),
                    "--sandbox".into(),
                    "read-only".into(),
                    "--skip-git-repo-check".into(),
                    "-C".into(),
                    ESCALATE_SCRATCH_DIR.into(),
                    "-o".into(),
                    out_file.to_string_lossy().into_owned(),
                    prompt.to_string(),
                ],
                read_output_from: Some(out_file),
            })
        }
        other => Err(format!(
            "provider desconocido '{other}'. Valores válidos: claude_code, grok, gemini, openai_codex."
        )),
    }
}

pub(crate) async fn execute_escalate_to_cloud_agent(call: &ToolCall) -> ToolResult {
    let provider = arg_str(call, "provider").trim().to_ascii_lowercase();
    let prompt = arg_str(call, "prompt").trim();
    let caller = extract_tool_caller(call);

    if provider.is_empty() {
        return err(call, "Falta 'provider' (claude_code | grok | gemini | openai_codex).");
    }
    if prompt.is_empty() {
        return err(call, "Falta 'prompt'.");
    }

    if let Err(e) = ensure_scratch_dir() {
        return err(
            call,
            format!("No se pudo preparar el directorio de trabajo aislado {ESCALATE_SCRATCH_DIR}: {e}"),
        );
    }

    let command_spec = match build_command(&provider, prompt) {
        Ok(spec) => spec,
        Err(e) => return err(call, e),
    };

    if let Err(cap_error) = check_and_record_daily_cap(&provider) {
        info!(
            "🚫 [Hera] escalate_to_cloud_agent DENIED (daily cap) provider={provider} caller={caller} prompt_len={}",
            prompt.len()
        );
        return err(call, cap_error);
    }

    info!(
        "☁️ [Hera] escalate_to_cloud_agent provider={provider} caller={caller} prompt_len={}",
        prompt.len()
    );

    let spawn_result = tokio::time::timeout(
        Duration::from_secs(TIMEOUT_S),
        tokio::process::Command::new(&command_spec.program)
            .args(&command_spec.args)
            .current_dir(ESCALATE_SCRATCH_DIR)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .output(),
    )
    .await;

    match spawn_result {
        Ok(Ok(out)) => {
            let stdout_text = String::from_utf8_lossy(&out.stdout).trim().to_string();
            let stderr_text = String::from_utf8_lossy(&out.stderr).trim().to_string();

            let response_text = if let Some(out_file) = &command_spec.read_output_from {
                let file_text = std::fs::read_to_string(out_file).unwrap_or_default();
                let _ = std::fs::remove_file(out_file);
                if file_text.trim().is_empty() {
                    stdout_text.clone()
                } else {
                    file_text.trim().to_string()
                }
            } else {
                stdout_text.clone()
            };

            info!(
                "☁️ [Hera] escalate_to_cloud_agent provider={provider} exit_ok={} response_len={}",
                out.status.success(),
                response_text.len()
            );

            if out.status.success() && !response_text.is_empty() {
                ok(call, response_text)
            } else {
                err(
                    call,
                    format!(
                        "provider '{provider}' terminó sin respuesta útil (exit_ok={}).\nstdout:\n{stdout_text}\nstderr:\n{stderr_text}",
                        out.status.success()
                    ),
                )
            }
        }
        Ok(Err(e)) => err(call, format!("No se pudo lanzar el proceso de '{provider}': {e}")),
        Err(_) => err(
            call,
            format!("provider '{provider}' superó el timeout de {TIMEOUT_S}s — abortado."),
        ),
    }
}
