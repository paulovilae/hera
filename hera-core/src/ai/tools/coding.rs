//! Coding tools (Fase 2 — docs/AVA_CODING_AGENT_PLAN.md): the surgical
//! file-editing + code-search primitives a real coding agent needs and that
//! Hera lacked. Until now the only file mutation was `write_file` (whole-file
//! overwrite), which forces the model to regenerate an entire file to change one
//! line — the main driver of hallucination on large files.
//!
//! - `edit_file` — exact-match block replacement (old_string to new_string),
//!   the `Edit` pattern from Claude Code / claw-code `tools/`.
//! - `grep_search` — regex search across the workspace tree (bounded).
//! - `glob_search` — find files by glob pattern (bounded).
//!
//! All three reuse `platform::resolve_guarded_fs_path` so they are confined to
//! the same sovereign roots as read_file/write_file (OS root, home, /tmp).
//! Ported algorithm from claw-code `rust/crates/tools` (MIT).

use super::platform::resolve_guarded_fs_path;
use crate::ai::tool_executor::{ToolCall, ToolResult};
use std::path::{Path, PathBuf};
use tracing::info;

/// Directories never worth walking for grep/glob — build output and VCS noise.
const SKIP_DIRS: &[&str] = &[
    ".git",
    "target",
    "node_modules",
    ".syncthing",
    "dist",
    "build",
    ".cache",
];
/// Hard caps so a search can never run away on a huge tree.
const MAX_FILES_SCANNED: usize = 20_000;
const MAX_MATCHES: usize = 200;
const MAX_FILE_BYTES: u64 = 2_000_000;

fn arg_str<'a>(call: &'a ToolCall, key: &str) -> &'a str {
    call.arguments.get(key).and_then(|v| v.as_str()).unwrap_or("")
}

/// First non-empty string argument among several aliases. Different models emit
/// different argument names for the same concept (e.g. Qwen3-Coder uses
/// `file_path`/`content` for edits); accepting aliases keeps the tools usable
/// across models without retraining the harness.
fn arg_str_any<'a>(call: &'a ToolCall, keys: &[&str]) -> &'a str {
    for key in keys {
        let v = call.arguments.get(*key).and_then(|v| v.as_str()).unwrap_or("");
        if !v.is_empty() {
            return v;
        }
    }
    ""
}

fn ok(call: &ToolCall, output: String) -> ToolResult {
    ToolResult {
        name: call.name.clone(),
        success: true,
        output,
    }
}

fn err(call: &ToolCall, output: impl Into<String>) -> ToolResult {
    ToolResult {
        name: call.name.clone(),
        success: false,
        output: output.into(),
    }
}

// ---------------------------------------------------------------------------
// edit_file
// ---------------------------------------------------------------------------

/// Surgical edit: replace an exact, unique `old_string` with `new_string`.
/// Refuses ambiguous edits (0 or >1 matches) unless `replace_all` is set — this
/// is what makes it safe on large files where a whole-file rewrite would lose
/// context.
pub(crate) async fn execute_edit_file(call: &ToolCall) -> ToolResult {
    // Accept argument aliases so different models' tool-call conventions work
    // (e.g. Qwen3-Coder emits `file_path`/`content`; Claude-style emits
    // `path`/`old_string`/`new_string`).
    let path = arg_str_any(call, &["path", "file_path", "filename", "file"]);
    let old_string = arg_str_any(call, &["old_string", "old_str", "old"]);
    let new_string = arg_str_any(call, &["new_string", "new_str", "new"]);
    let whole_content = arg_str_any(call, &["content", "new_content", "file_text", "text"]);
    let replace_all = call
        .arguments
        .get("replace_all")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if path.is_empty() {
        return err(call, "Missing 'path'.");
    }

    // Whole-file mode: some models call edit_file with just a path + full new
    // content (no old_string). Honor it as a full-file write so the edit isn't
    // silently rejected.
    if old_string.is_empty() {
        if !whole_content.is_empty() {
            let resolved = match resolve_guarded_fs_path(path, true) {
                Ok(p) => p,
                Err(error) => return err(call, error),
            };
            if let Err(e) = std::fs::write(&resolved, whole_content) {
                return err(call, format!("Failed to write file '{}': {}", resolved.display(), e));
            }
            info!("✏️ [Hera] edit_file (full content) {}", resolved.display());
            return ok(
                call,
                format!(
                    "Wrote full contents to '{}' ({} lines).",
                    resolved.display(),
                    whole_content.lines().count()
                ),
            );
        }
        return err(
            call,
            "Provide either 'old_string' (with 'new_string') for a surgical edit, or 'content' for a full-file write.",
        );
    }
    if old_string == new_string {
        return err(call, "'old_string' and 'new_string' are identical; nothing to change.");
    }

    let resolved = match resolve_guarded_fs_path(path, true) {
        Ok(p) => p,
        Err(error) => return err(call, error),
    };

    let content = match std::fs::read_to_string(&resolved) {
        Ok(c) => c,
        Err(e) => return err(call, format!("Failed to read file '{}': {}", resolved.display(), e)),
    };

    let occurrences = content.matches(old_string).count();
    if occurrences == 0 {
        return err(
            call,
            format!(
                "'old_string' was not found in '{}'. Read the file first and copy the exact text (including indentation) you want to replace.",
                resolved.display()
            ),
        );
    }
    if occurrences > 1 && !replace_all {
        return err(
            call,
            format!(
                "'old_string' is not unique in '{}' ({} matches). Add surrounding context to make it unique, or set replace_all=true to replace every occurrence.",
                resolved.display(),
                occurrences
            ),
        );
    }

    let updated = if replace_all {
        content.replace(old_string, new_string)
    } else {
        content.replacen(old_string, new_string, 1)
    };

    if let Err(e) = std::fs::write(&resolved, &updated) {
        return err(call, format!("Failed to write file '{}': {}", resolved.display(), e));
    }

    let old_lines = old_string.lines().count().max(1);
    let new_lines = new_string.lines().count().max(1);
    info!(
        "✏️ [Hera] edit_file {} ({} occurrence(s), -{}/+{} lines)",
        resolved.display(),
        if replace_all { occurrences } else { 1 },
        old_lines,
        new_lines
    );
    ok(
        call,
        format!(
            "Edited '{}': replaced {} occurrence(s) (-{} / +{} lines).",
            resolved.display(),
            if replace_all { occurrences } else { 1 },
            old_lines,
            new_lines
        ),
    )
}

// ---------------------------------------------------------------------------
// grep_search
// ---------------------------------------------------------------------------

/// Regex search across a directory tree (or a single file), returning
/// `path:line:text` matches up to a hard cap.
pub(crate) async fn execute_grep_search(call: &ToolCall) -> ToolResult {
    let pattern = arg_str(call, "pattern");
    if pattern.is_empty() {
        return err(call, "Missing 'pattern' (a regular expression).");
    }
    let base = {
        let p = arg_str(call, "path");
        if p.is_empty() { "." } else { p }
    };
    let regex = match regex::Regex::new(pattern) {
        Ok(r) => r,
        Err(e) => return err(call, format!("Invalid regex '{}': {}", pattern, e)),
    };

    let resolved = match resolve_guarded_fs_path(base, true) {
        Ok(p) => p,
        Err(error) => return err(call, error),
    };

    let mut files: Vec<PathBuf> = Vec::new();
    let mut scanned = 0usize;
    collect_files(&resolved, &mut files, &mut scanned);

    let mut matches: Vec<String> = Vec::new();
    let mut files_with_hits = 0usize;
    'outer: for file in &files {
        let Ok(content) = std::fs::read_to_string(file) else {
            continue; // skip binary / unreadable
        };
        let mut hit_in_file = false;
        for (idx, line) in content.lines().enumerate() {
            if regex.is_match(line) {
                hit_in_file = true;
                let display = file.strip_prefix(&resolved).unwrap_or(file);
                let trimmed = if line.len() > 300 { &line[..300] } else { line };
                matches.push(format!("{}:{}:{}", display.display(), idx + 1, trimmed.trim_end()));
                if matches.len() >= MAX_MATCHES {
                    break 'outer;
                }
            }
        }
        if hit_in_file {
            files_with_hits += 1;
        }
    }

    info!(
        "🔎 [Hera] grep_search '{}' under {} → {} match(es) in {} file(s)",
        pattern,
        resolved.display(),
        matches.len(),
        files_with_hits
    );

    if matches.is_empty() {
        return ok(call, format!("No matches for /{}/ under '{}'.", pattern, resolved.display()));
    }
    let capped = if matches.len() >= MAX_MATCHES {
        format!("\n... (capped at {} matches)", MAX_MATCHES)
    } else {
        String::new()
    };
    ok(
        call,
        format!(
            "{} match(es) in {} file(s) for /{}/:\n{}{}",
            matches.len(),
            files_with_hits,
            pattern,
            matches.join("\n"),
            capped
        ),
    )
}

// ---------------------------------------------------------------------------
// glob_search
// ---------------------------------------------------------------------------

/// Find files whose path matches a glob pattern (`*`, `?`, `**`). Returns
/// workspace-relative paths up to a hard cap.
pub(crate) async fn execute_glob_search(call: &ToolCall) -> ToolResult {
    let pattern = arg_str(call, "pattern");
    if pattern.is_empty() {
        return err(call, "Missing 'pattern' (e.g. '**/*.rs' or 'src/*.toml').");
    }
    let base = {
        let p = arg_str(call, "path");
        if p.is_empty() { "." } else { p }
    };
    let resolved = match resolve_guarded_fs_path(base, true) {
        Ok(p) => p,
        Err(error) => return err(call, error),
    };
    let regex = match glob_to_regex(pattern) {
        Ok(r) => r,
        Err(e) => return err(call, e),
    };

    let mut files: Vec<PathBuf> = Vec::new();
    let mut scanned = 0usize;
    collect_files(&resolved, &mut files, &mut scanned);

    let mut hits: Vec<String> = Vec::new();
    for file in &files {
        let rel = file.strip_prefix(&resolved).unwrap_or(file);
        let rel_str = rel.to_string_lossy();
        if regex.is_match(&rel_str) {
            hits.push(rel_str.to_string());
            if hits.len() >= MAX_MATCHES {
                break;
            }
        }
    }
    hits.sort();

    info!(
        "🗂️ [Hera] glob_search '{}' under {} → {} file(s)",
        pattern,
        resolved.display(),
        hits.len()
    );

    if hits.is_empty() {
        return ok(call, format!("No files match '{}' under '{}'.", pattern, resolved.display()));
    }
    let capped = if hits.len() >= MAX_MATCHES {
        format!("\n... (capped at {} files)", MAX_MATCHES)
    } else {
        String::new()
    };
    ok(
        call,
        format!("{} file(s) matching '{}':\n{}{}", hits.len(), pattern, hits.join("\n"), capped),
    )
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Recursively collect files under `root`, skipping VCS/build dirs and oversized
/// files, bounded by MAX_FILES_SCANNED. If `root` is a file, returns just it.
fn collect_files(root: &Path, out: &mut Vec<PathBuf>, scanned: &mut usize) {
    if root.is_file() {
        out.push(root.to_path_buf());
        return;
    }
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        if *scanned >= MAX_FILES_SCANNED {
            return;
        }
        let path = entry.path();
        let file_type = match entry.file_type() {
            Ok(ft) => ft,
            Err(_) => continue,
        };
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if file_type.is_dir() {
            if SKIP_DIRS.contains(&name.as_ref()) || name.starts_with('.') {
                continue;
            }
            collect_files(&path, out, scanned);
        } else if file_type.is_file() {
            *scanned += 1;
            if std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0) <= MAX_FILE_BYTES {
                out.push(path);
            }
        }
    }
}

/// Convert a glob (`*`, `?`, `**`) into an anchored regex over the relative path.
fn glob_to_regex(pattern: &str) -> Result<regex::Regex, String> {
    let mut re = String::with_capacity(pattern.len() * 2 + 2);
    re.push('^');
    let bytes = pattern.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        match c {
            '*' => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'*' {
                    // ** matches across directory separators
                    re.push_str(".*");
                    i += 1;
                    // swallow an immediate trailing slash so `**/x` also matches `x`
                    if i + 1 < bytes.len() && bytes[i + 1] == b'/' {
                        i += 1;
                    }
                } else {
                    re.push_str("[^/]*");
                }
            }
            '?' => re.push_str("[^/]"),
            '.' | '+' | '(' | ')' | '|' | '^' | '$' | '{' | '}' | '[' | ']' | '\\' => {
                re.push('\\');
                re.push(c);
            }
            other => re.push(other),
        }
        i += 1;
    }
    re.push('$');
    regex::Regex::new(&re).map_err(|e| format!("Invalid glob '{}': {}", pattern, e))
}

// ---------------------------------------------------------------------------
// bash_exec helpers
// ---------------------------------------------------------------------------

async fn run_bash(command: &str, timeout_secs: u64) -> Result<(i32, String, String), String> {
    use std::process::Stdio;
    use std::time::Duration;

    let child = tokio::process::Command::new("/bin/bash")
        .arg("-c")
        .arg(command)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| format!("Failed to spawn bash: {e}"))?;

    match tokio::time::timeout(Duration::from_secs(timeout_secs), child.wait_with_output()).await {
        Ok(Ok(out)) => {
            let stdout = String::from_utf8_lossy(&out.stdout).to_string();
            let stderr = String::from_utf8_lossy(&out.stderr).to_string();
            let code = out.status.code().unwrap_or(-1);
            Ok((code, stdout, stderr))
        }
        Ok(Err(e)) => Err(format!("Command wait failed: {e}")),
        Err(_) => Err(format!("Command timed out after {timeout_secs}s.")),
    }
}

fn cap_output(s: &str, max: usize) -> String {
    if s.len() > max {
        format!("...(truncated {} chars)...\n{}", s.len(), &s[s.len() - max..])
    } else {
        s.to_string()
    }
}

// ---------------------------------------------------------------------------
// bash_exec
// ---------------------------------------------------------------------------

/// Execute a bash command with optional post-check and auto-revert.
///
/// Parameters:
/// - `command` (required): the bash command to run
/// - `timeout_seconds` (optional, default 30, max 300)
/// - `post_check` (optional): a verification command run after the main one
/// - `revert_cmd` (optional): a rollback command run if `post_check` exits non-zero
/// - `revert_on_failure` (optional bool, default false): flag to trigger revert
///
/// Risk: Critical. Requires explicit `bash_exec` grant or `unsafe_all`.
pub(crate) async fn execute_bash_exec(call: &ToolCall) -> ToolResult {
    let command = arg_str_any(call, &["command", "cmd", "bash", "script"]);
    if command.trim().is_empty() {
        return err(call, "bash_exec requires a non-empty 'command' argument.");
    }

    let timeout_secs = call.arguments
        .get("timeout_seconds")
        .and_then(|v| v.as_u64())
        .unwrap_or(30)
        .clamp(1, 300);

    let post_check = call.arguments
        .get("post_check")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let revert_cmd = call.arguments
        .get("revert_cmd")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let revert_on_failure = call.arguments
        .get("revert_on_failure")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    info!("🐚 [Hera] bash_exec ({}s): {}", timeout_secs, &command[..command.len().min(120)]);

    const MAX_OUT: usize = 8_000;
    const MAX_ERR: usize = 2_000;

    let (main_exit, main_stdout, main_stderr) = match run_bash(&command, timeout_secs).await {
        Ok(r) => r,
        Err(e) => return err(call, e),
    };

    let mut output = format!(
        "exit_code: {main_exit}\nstdout:\n{}\nstderr:\n{}",
        cap_output(&main_stdout, MAX_OUT),
        cap_output(&main_stderr, MAX_ERR),
    );

    if !post_check.trim().is_empty() {
        info!("🔍 [Hera] bash_exec post_check: {}", &post_check[..post_check.len().min(80)]);

        match run_bash(&post_check, timeout_secs.min(60)).await {
            Ok((check_exit, check_stdout, check_stderr)) => {
                output.push_str(&format!(
                    "\n\n=== POST CHECK ===\nexit_code: {check_exit}\nstdout:\n{}\nstderr:\n{}",
                    cap_output(&check_stdout, MAX_OUT / 2),
                    cap_output(&check_stderr, MAX_ERR / 2),
                ));

                if check_exit != 0 && (revert_on_failure || !revert_cmd.trim().is_empty()) {
                    if revert_cmd.trim().is_empty() {
                        output.push_str(
                            "\n\n=== REVERT ===\n[skipped — post_check failed but no revert_cmd provided]",
                        );
                    } else {
                        info!("⏪ [Hera] bash_exec reverting: {}", &revert_cmd[..revert_cmd.len().min(80)]);
                        match run_bash(&revert_cmd, timeout_secs.min(60)).await {
                            Ok((rv_exit, rv_stdout, rv_stderr)) => {
                                output.push_str(&format!(
                                    "\n\n=== REVERT ===\nexit_code: {rv_exit}\nstdout:\n{}\nstderr:\n{}",
                                    cap_output(&rv_stdout, MAX_OUT / 2),
                                    cap_output(&rv_stderr, MAX_ERR / 2),
                                ));
                            }
                            Err(e) => output.push_str(&format!("\n\n=== REVERT FAILED ===\n{e}")),
                        }
                    }
                }
            }
            Err(e) => output.push_str(&format!("\n\n=== POST CHECK ERROR ===\n{e}")),
        }
    }

    ok(call, output.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn call(name: &str, args: serde_json::Value) -> ToolCall {
        ToolCall { name: name.to_string(), arguments: args }
    }

    #[tokio::test]
    async fn edit_file_replaces_unique_block() {
        let dir = std::env::temp_dir().join("hera_edit_test_unique");
        let _ = std::fs::create_dir_all(&dir);
        let file = dir.join("a.txt");
        std::fs::write(&file, "alpha\nbeta\ngamma\n").unwrap();

        let res = execute_edit_file(&call(
            "edit_file",
            json!({"path": file.to_str().unwrap(), "old_string": "beta", "new_string": "BETA"}),
        ))
        .await;
        assert!(res.success, "{}", res.output);
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "alpha\nBETA\ngamma\n");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn edit_file_refuses_ambiguous_match() {
        let dir = std::env::temp_dir().join("hera_edit_test_ambig");
        let _ = std::fs::create_dir_all(&dir);
        let file = dir.join("b.txt");
        std::fs::write(&file, "x\nx\n").unwrap();

        let res = execute_edit_file(&call(
            "edit_file",
            json!({"path": file.to_str().unwrap(), "old_string": "x", "new_string": "y"}),
        ))
        .await;
        assert!(!res.success);
        assert!(res.output.contains("not unique"));
        // replace_all succeeds
        let res2 = execute_edit_file(&call(
            "edit_file",
            json!({"path": file.to_str().unwrap(), "old_string": "x", "new_string": "y", "replace_all": true}),
        ))
        .await;
        assert!(res2.success, "{}", res2.output);
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "y\ny\n");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn edit_file_reports_missing_old_string() {
        let dir = std::env::temp_dir().join("hera_edit_test_missing");
        let _ = std::fs::create_dir_all(&dir);
        let file = dir.join("c.txt");
        std::fs::write(&file, "hello\n").unwrap();
        let res = execute_edit_file(&call(
            "edit_file",
            json!({"path": file.to_str().unwrap(), "old_string": "nope", "new_string": "z"}),
        ))
        .await;
        assert!(!res.success);
        assert!(res.output.contains("not found"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn grep_search_finds_matches() {
        let dir = std::env::temp_dir().join("hera_grep_test");
        let _ = std::fs::create_dir_all(&dir);
        std::fs::write(dir.join("one.rs"), "fn main() {}\nlet x = 1;\n").unwrap();
        std::fs::write(dir.join("two.rs"), "fn helper() {}\n").unwrap();
        let res = execute_grep_search(&call(
            "grep_search",
            json!({"pattern": "fn \\w+\\(", "path": dir.to_str().unwrap()}),
        ))
        .await;
        assert!(res.success, "{}", res.output);
        assert!(res.output.contains("one.rs"));
        assert!(res.output.contains("two.rs"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn glob_search_matches_extension() {
        let dir = std::env::temp_dir().join("hera_glob_test");
        let sub = dir.join("src");
        let _ = std::fs::create_dir_all(&sub);
        std::fs::write(dir.join("top.rs"), "").unwrap();
        std::fs::write(sub.join("inner.rs"), "").unwrap();
        std::fs::write(dir.join("notes.txt"), "").unwrap();
        let res = execute_glob_search(&call(
            "glob_search",
            json!({"pattern": "**/*.rs", "path": dir.to_str().unwrap()}),
        ))
        .await;
        assert!(res.success, "{}", res.output);
        assert!(res.output.contains("top.rs"));
        assert!(res.output.contains("inner.rs"));
        assert!(!res.output.contains("notes.txt"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn glob_to_regex_handles_double_star() {
        let re = glob_to_regex("**/*.rs").unwrap();
        assert!(re.is_match("a.rs"));
        assert!(re.is_match("src/a.rs"));
        assert!(re.is_match("a/b/c.rs"));
        assert!(!re.is_match("a.txt"));
        let re2 = glob_to_regex("src/*.toml").unwrap();
        assert!(re2.is_match("src/Cargo.toml"));
        assert!(!re2.is_match("src/sub/Cargo.toml"));
    }
}
