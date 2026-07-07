//! Deployment tools for the coding/deploy agentic loop.
//! Risk: High. Requires `deploy` permission or `unsafe_all`.
//!
//! - `git_add`    — stages files: git add <files> in <path>
//! - `git_commit` — commits staged files: git commit -m <message> in <path>
//! - `pm2_restart`— restarts a PM2 service by name

use super::platform::resolve_guarded_fs_path;
use crate::ai::tool_executor::{ToolCall, ToolResult};
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;
use tracing::info;

const TIMEOUT_S: u64 = 60;

fn arg_str<'a>(call: &'a ToolCall, key: &str) -> &'a str {
    call.arguments.get(key).and_then(|v| v.as_str()).unwrap_or("")
}

fn ok(call: &ToolCall, output: String) -> ToolResult {
    ToolResult { name: call.name.clone(), success: true, output }
}

fn err(call: &ToolCall, output: impl Into<String>) -> ToolResult {
    ToolResult { name: call.name.clone(), success: false, output: output.into() }
}

async fn run_cmd(dir: &Path, program: &str, args: &[&str]) -> (bool, String) {
    let result = tokio::time::timeout(
        Duration::from_secs(TIMEOUT_S),
        tokio::process::Command::new(program)
            .args(args)
            .current_dir(dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output(),
    )
    .await;
    match result {
        Ok(Ok(out)) => {
            let combined = format!(
                "{}{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
            (out.status.success(), combined.trim().to_string())
        }
        Ok(Err(e)) => (false, format!("spawn error: {e}")),
        Err(_) => (false, format!("timed out after {TIMEOUT_S}s")),
    }
}

pub(crate) async fn execute_git_add(call: &ToolCall) -> ToolResult {
    let path = arg_str(call, "path");
    let files = arg_str(call, "files");
    if path.is_empty() {
        return err(call, "Missing 'path': absolute path to git repo root.");
    }
    let dir = match resolve_guarded_fs_path(path, true) {
        Ok(d) => d,
        Err(e) => return err(call, e),
    };
    // Accept files as a JSON array or a space-separated string; default to "."
    let files_list: Vec<String> =
        if let Some(arr) = call.arguments.get("files").and_then(|v| v.as_array()) {
            arr.iter().filter_map(|v| v.as_str()).map(String::from).collect()
        } else if !files.is_empty() {
            files.split_whitespace().map(String::from).collect()
        } else {
            vec![".".to_string()]
        };
    let mut args = vec!["add".to_string()];
    args.extend(files_list.iter().cloned());
    let args_ref: Vec<&str> = args.iter().map(String::as_str).collect();
    info!("[Hera] git_add {} files={}", dir.display(), files_list.join(" "));
    let (ok_flag, out) = run_cmd(&dir, "git", &args_ref).await;
    if ok_flag {
        ok(call, format!("git add OK\n{out}"))
    } else {
        err(call, format!("git add failed:\n{out}"))
    }
}

pub(crate) async fn execute_git_commit(call: &ToolCall) -> ToolResult {
    let path = arg_str(call, "path");
    let message = arg_str(call, "message");
    if path.is_empty() {
        return err(call, "Missing 'path'.");
    }
    if message.is_empty() {
        return err(call, "Missing 'message'.");
    }
    let dir = match resolve_guarded_fs_path(path, true) {
        Ok(d) => d,
        Err(e) => return err(call, e),
    };
    info!(
        "[Hera] git_commit {} msg={}",
        dir.display(),
        &message[..message.len().min(60)]
    );
    let (ok_flag, out) = run_cmd(&dir, "git", &["commit", "-m", message]).await;
    if ok_flag {
        ok(call, format!("Committed:\n{out}"))
    } else {
        err(call, format!("git commit failed:\n{out}"))
    }
}

pub(crate) async fn execute_pm2_restart(call: &ToolCall) -> ToolResult {
    let service = arg_str(call, "service");
    if service.is_empty() {
        return err(call, "Missing 'service': PM2 service name.");
    }
    // Prevent shell injection: only alphanumeric, dash, underscore.
    if !service.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_') {
        return err(
            call,
            format!(
                "Invalid service name '{service}': use only alphanumeric/dash/underscore."
            ),
        );
    }
    info!("[Hera] pm2_restart service={}", service);
    // On genesis pm2 lives at /usr/bin/pm2; fall back to PATH lookup.
    let pm2_bin = if Path::new("/usr/bin/pm2").exists() {
        "/usr/bin/pm2"
    } else {
        "pm2"
    };
    // Run from /tmp — neutral dir, avoids cargo workspace confusion.
    let (ok_flag, out) = run_cmd(Path::new("/tmp"), pm2_bin, &["restart", service]).await;
    if ok_flag {
        ok(call, format!("pm2 restart {service} OK:\n{out}"))
    } else {
        err(call, format!("pm2 restart {service} failed:\n{out}"))
    }
}
