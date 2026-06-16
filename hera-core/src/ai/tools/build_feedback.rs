//! Build/test feedback tools (Fase 3 — docs/AVA_CODING_AGENT_PLAN.md).
//!
//! `run_code` returns stdout/stderr as an unstructured blob, so the agent had to
//! eyeball a wall of compiler text to find what to fix. These tools run the
//! project's own toolchain and return STRUCTURED feedback the model can act on:
//! `cargo_check` parses `cargo --message-format=json` into `file:line:col [code]
//! message`; `cargo_test` / `pytest` return a pass/fail summary plus the failing
//! items. Fed back through the Fase 1 loop, this closes the real debugging cycle:
//! compile → read the error at file:line → edit → recompile.
//!
//! All run via `tokio::process` with a wall-clock timeout (so a hung build can't
//! pin a worker), confined to the same sovereign roots as the other file tools.

use super::platform::resolve_guarded_fs_path;
use crate::ai::tool_executor::{ToolCall, ToolResult};
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;
use tracing::info;

const DEFAULT_TIMEOUT_S: u64 = 180;
const MAX_TIMEOUT_S: u64 = 600;
const MAX_DIAGS: usize = 40;
const MAX_TAIL: usize = 6_000;

fn arg_str<'a>(call: &'a ToolCall, key: &str) -> &'a str {
    call.arguments.get(key).and_then(|v| v.as_str()).unwrap_or("")
}

fn ok(call: &ToolCall, success: bool, output: String) -> ToolResult {
    ToolResult { name: call.name.clone(), success, output }
}

fn err(call: &ToolCall, output: impl Into<String>) -> ToolResult {
    ToolResult { name: call.name.clone(), success: false, output: output.into() }
}

fn timeout_secs(call: &ToolCall) -> u64 {
    call.arguments
        .get("timeout_seconds")
        .and_then(|v| v.as_u64())
        .unwrap_or(DEFAULT_TIMEOUT_S)
        .clamp(5, MAX_TIMEOUT_S)
}

fn workdir(call: &ToolCall) -> Result<std::path::PathBuf, String> {
    let p = arg_str(call, "path");
    let p = if p.is_empty() { "." } else { p };
    let resolved = resolve_guarded_fs_path(p, true)?;
    if !resolved.is_dir() {
        return Err(format!("'{}' is not a directory.", resolved.display()));
    }
    Ok(resolved)
}

/// Keep the last `MAX_TAIL` bytes of a long output (errors live at the end).
fn tail(text: &str) -> String {
    let t = text.trim();
    if t.len() <= MAX_TAIL {
        t.to_string()
    } else {
        format!("...(truncated)...\n{}", &t[t.len() - MAX_TAIL..])
    }
}

/// Run a command in `dir` with a wall-clock timeout. Returns (success, stdout, stderr).
async fn run_cmd(
    dir: &Path,
    program: &str,
    args: &[&str],
    timeout_s: u64,
) -> Result<(bool, String, String), String> {
    let child = tokio::process::Command::new(program)
        .args(args)
        .current_dir(dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| format!("Failed to start '{}': {}", program, e))?;

    match tokio::time::timeout(Duration::from_secs(timeout_s), child.wait_with_output()).await {
        Ok(Ok(out)) => Ok((
            out.status.success(),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )),
        Ok(Err(e)) => Err(format!("'{}' failed: {}", program, e)),
        Err(_) => Err(format!("'{}' timed out after {}s.", program, timeout_s)),
    }
}

// ---------------------------------------------------------------------------
// cargo check (structured)
// ---------------------------------------------------------------------------

struct Diag {
    file: String,
    line: u64,
    col: u64,
    code: String,
    message: String,
}

/// Parse `cargo --message-format=json` stdout into errors (collected) and a
/// warning count.
fn parse_cargo_diagnostics(stdout: &str) -> (Vec<Diag>, usize) {
    let mut errors = Vec::new();
    let mut warnings = 0usize;

    for line in stdout.lines() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if v.get("reason").and_then(|r| r.as_str()) != Some("compiler-message") {
            continue;
        }
        let Some(msg) = v.get("message") else { continue };
        let level = msg.get("level").and_then(|l| l.as_str()).unwrap_or("");
        match level {
            "warning" => {
                warnings += 1;
                continue;
            }
            "error" => {}
            _ => continue, // note / help / failure-note
        }

        let message = msg
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("")
            .to_string();
        let code = msg
            .get("code")
            .and_then(|c| c.get("code"))
            .and_then(|c| c.as_str())
            .unwrap_or("")
            .to_string();

        let span = msg.get("spans").and_then(|s| s.as_array()).and_then(|arr| {
            arr.iter()
                .find(|s| s.get("is_primary").and_then(|p| p.as_bool()).unwrap_or(false))
                .or_else(|| arr.first())
        });
        let (file, line, col) = span
            .map(|s| {
                (
                    s.get("file_name").and_then(|f| f.as_str()).unwrap_or("").to_string(),
                    s.get("line_start").and_then(|l| l.as_u64()).unwrap_or(0),
                    s.get("column_start").and_then(|c| c.as_u64()).unwrap_or(0),
                )
            })
            .unwrap_or_default();

        errors.push(Diag { file, line, col, code, message });
    }

    (errors, warnings)
}

fn format_diag(d: &Diag) -> String {
    let loc = if d.file.is_empty() {
        String::new()
    } else if d.line > 0 {
        format!("{}:{}:{}: ", d.file, d.line, d.col)
    } else {
        format!("{}: ", d.file)
    };
    let code = if d.code.is_empty() { String::new() } else { format!("[{}] ", d.code) };
    // First line of the message only — keeps the list scannable.
    let first = d.message.lines().next().unwrap_or(&d.message);
    format!("{}{}{}", loc, code, first)
}

/// Run `cargo check` and return structured errors.
pub(crate) async fn execute_cargo_check(call: &ToolCall) -> ToolResult {
    let dir = match workdir(call) {
        Ok(d) => d,
        Err(e) => return err(call, e),
    };
    let timeout_s = timeout_secs(call);
    let include_tests = call
        .arguments
        .get("tests")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let mut args = vec!["check", "--message-format=json", "--color=never"];
    if include_tests {
        args.push("--tests");
    }

    let (success, stdout, stderr) = match run_cmd(&dir, "cargo", &args, timeout_s).await {
        Ok(r) => r,
        Err(e) => return err(call, e),
    };

    let (errors, warnings) = parse_cargo_diagnostics(&stdout);
    info!(
        "🦀 [Hera] cargo check {} → {} error(s), {} warning(s)",
        dir.display(),
        errors.len(),
        warnings
    );

    if errors.is_empty() && success {
        return ok(
            call,
            true,
            format!("cargo check OK in '{}' — 0 errors, {} warning(s).", dir.display(), warnings),
        );
    }

    // If cargo failed but produced no parseable diagnostics (e.g. a manifest or
    // resolver error), surface the stderr tail so the model still sees the cause.
    if errors.is_empty() {
        return ok(
            call,
            false,
            format!(
                "cargo check FAILED in '{}' (no compiler diagnostics parsed).\nstderr:\n{}",
                dir.display(),
                tail(&stderr)
            ),
        );
    }

    let shown = errors.len().min(MAX_DIAGS);
    let mut body = errors
        .iter()
        .take(MAX_DIAGS)
        .map(format_diag)
        .collect::<Vec<_>>()
        .join("\n");
    if errors.len() > MAX_DIAGS {
        body.push_str(&format!("\n... ({} more error(s))", errors.len() - MAX_DIAGS));
    }
    ok(
        call,
        false,
        format!(
            "cargo check FAILED in '{}' — {} error(s), {} warning(s) (showing {}):\n{}",
            dir.display(),
            errors.len(),
            warnings,
            shown,
            body
        ),
    )
}

// ---------------------------------------------------------------------------
// cargo test
// ---------------------------------------------------------------------------

/// Extract failing test names and the `test result:` summary lines from libtest output.
fn summarize_cargo_test(stdout: &str) -> (Vec<String>, Vec<String>) {
    let mut failed = Vec::new();
    let mut summaries = Vec::new();
    for line in stdout.lines() {
        let l = line.trim();
        if l.starts_with("test result:") {
            summaries.push(l.to_string());
        } else if let Some(rest) = l.strip_prefix("test ")
            && let Some(name) = rest.strip_suffix("... FAILED")
        {
            failed.push(name.trim().to_string());
        }
    }
    (failed, summaries)
}

/// Run `cargo test` and return a pass/fail summary plus failing tests. If the
/// build itself fails, surface the structured compiler errors instead.
pub(crate) async fn execute_cargo_test(call: &ToolCall) -> ToolResult {
    let dir = match workdir(call) {
        Ok(d) => d,
        Err(e) => return err(call, e),
    };
    let timeout_s = timeout_secs(call);

    // First a structured build check so build errors come back as file:line.
    let (check_ok, check_stdout, check_stderr) =
        match run_cmd(&dir, "cargo", &["check", "--tests", "--message-format=json", "--color=never"], timeout_s).await {
            Ok(r) => r,
            Err(e) => return err(call, e),
        };
    let (build_errors, _) = parse_cargo_diagnostics(&check_stdout);
    if !check_ok && !build_errors.is_empty() {
        let body = build_errors.iter().take(MAX_DIAGS).map(format_diag).collect::<Vec<_>>().join("\n");
        return ok(
            call,
            false,
            format!(
                "cargo test in '{}' — build FAILED ({} error(s)); fix these before tests run:\n{}",
                dir.display(),
                build_errors.len(),
                body
            ),
        );
    }
    if !check_ok {
        return ok(
            call,
            false,
            format!("cargo test in '{}' — build FAILED.\nstderr:\n{}", dir.display(), tail(&check_stderr)),
        );
    }

    let (success, stdout, stderr) =
        match run_cmd(&dir, "cargo", &["test", "--no-fail-fast", "--color=never"], timeout_s).await {
            Ok(r) => r,
            Err(e) => return err(call, e),
        };
    let (failed, summaries) = summarize_cargo_test(&stdout);
    info!(
        "🧪 [Hera] cargo test {} → success={} failed={}",
        dir.display(),
        success,
        failed.len()
    );

    let mut out = format!(
        "cargo test in '{}' — {}.",
        dir.display(),
        if success { "PASSED" } else { "FAILED" }
    );
    if !summaries.is_empty() {
        out.push_str(&format!("\n{}", summaries.join("\n")));
    }
    if !failed.is_empty() {
        out.push_str(&format!("\nFailing tests:\n- {}", failed.join("\n- ")));
    }
    if !success {
        let combined = format!("{stdout}\n{stderr}");
        out.push_str(&format!("\n\nOutput tail:\n{}", tail(&combined)));
    }
    ok(call, success, out)
}

// ---------------------------------------------------------------------------
// pytest
// ---------------------------------------------------------------------------

/// Extract FAILED/ERROR lines and the final summary line from pytest output.
fn summarize_pytest(stdout: &str) -> (Vec<String>, Option<String>) {
    let mut failed = Vec::new();
    let mut summary = None;
    for line in stdout.lines() {
        let l = line.trim();
        if l.starts_with("FAILED ") || l.starts_with("ERROR ") {
            failed.push(l.to_string());
        }
        // pytest's final summary line is wrapped in '=' e.g. "== 2 failed, 1 passed in 0.1s =="
        if l.starts_with("==") && (l.contains("passed") || l.contains("failed") || l.contains("error")) {
            summary = Some(l.trim_matches('=').trim().to_string());
        }
    }
    (failed, summary)
}

/// Run pytest and return a pass/fail summary plus failing items.
pub(crate) async fn execute_pytest(call: &ToolCall) -> ToolResult {
    let dir = match workdir(call) {
        Ok(d) => d,
        Err(e) => return err(call, e),
    };
    let timeout_s = timeout_secs(call);
    let target = arg_str(call, "target");

    let mut args = vec!["-m", "pytest", "-q", "--no-header", "--color=no"];
    if !target.is_empty() {
        args.push(target);
    }

    let (success, stdout, stderr) = match run_cmd(&dir, "python3", &args, timeout_s).await {
        Ok(r) => r,
        Err(e) => return err(call, e),
    };
    let (failed, summary) = summarize_pytest(&stdout);
    info!(
        "🐍 [Hera] pytest {} → success={} failed={}",
        dir.display(),
        success,
        failed.len()
    );

    let mut out = format!(
        "pytest in '{}' — {}.",
        dir.display(),
        if success { "PASSED" } else { "FAILED" }
    );
    if let Some(s) = summary {
        out.push_str(&format!("\nSummary: {}", s));
    }
    if !failed.is_empty() {
        out.push_str(&format!("\nFailures:\n- {}", failed.join("\n- ")));
    }
    if !success {
        let combined = format!("{stdout}\n{stderr}");
        out.push_str(&format!("\n\nOutput tail:\n{}", tail(&combined)));
    }
    ok(call, success, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_cargo_diagnostics_extracts_errors_and_counts_warnings() {
        let stdout = concat!(
            r#"{"reason":"compiler-message","message":{"level":"warning","message":"unused variable: `x`","code":{"code":"unused_variables"},"spans":[{"file_name":"src/a.rs","line_start":3,"column_start":9,"is_primary":true}]}}"#,
            "\n",
            r#"{"reason":"compiler-message","message":{"level":"error","message":"mismatched types","code":{"code":"E0308"},"spans":[{"file_name":"src/b.rs","line_start":12,"column_start":5,"is_primary":true}]}}"#,
            "\n",
            r#"{"reason":"compiler-artifact","target":{"name":"x"}}"#,
        );
        let (errors, warnings) = parse_cargo_diagnostics(stdout);
        assert_eq!(warnings, 1);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].file, "src/b.rs");
        assert_eq!(errors[0].line, 12);
        assert_eq!(errors[0].code, "E0308");
        let formatted = format_diag(&errors[0]);
        assert!(formatted.contains("src/b.rs:12:5"));
        assert!(formatted.contains("[E0308]"));
        assert!(formatted.contains("mismatched types"));
    }

    #[test]
    fn summarize_cargo_test_finds_failures_and_summary() {
        let stdout = concat!(
            "test tests::works ... ok\n",
            "test tests::broken ... FAILED\n",
            "test result: FAILED. 1 passed; 1 failed; 0 ignored\n",
        );
        let (failed, summaries) = summarize_cargo_test(stdout);
        assert_eq!(failed, vec!["tests::broken".to_string()]);
        assert_eq!(summaries.len(), 1);
        assert!(summaries[0].contains("1 failed"));
    }

    #[test]
    fn summarize_pytest_finds_failures_and_summary() {
        let stdout = concat!(
            "FAILED tests/test_x.py::test_add - assert 4 == 5\n",
            "==== 1 failed, 2 passed in 0.05s ====\n",
        );
        let (failed, summary) = summarize_pytest(stdout);
        assert_eq!(failed.len(), 1);
        assert!(failed[0].contains("test_add"));
        assert_eq!(summary.as_deref(), Some("1 failed, 2 passed in 0.05s"));
    }
}
