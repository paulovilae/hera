//! Character-workspace file tools.
//!
//! These let a bot write ONLY into its own character workspace
//! (`Agents/workspaces/{name}/`) during the bootstrap ritual (Punto 2) and for
//! self-editing memory (Punto 3). The target directory is derived from the
//! SERVER-INJECTED `_persona_path` (stamped by `contextualize_tool_call` with
//! `.insert()`, so a bot cannot spoof it through its own tool arguments) — a bot can
//! therefore only ever touch its own workspace. Every write is additionally confined
//! with the generic Hera FS guard AND an explicit per-workspace `starts_with` check.

use crate::ai::tool_executor::{ToolCall, ToolResult};
use std::path::{Path, PathBuf};

use super::resolve_guarded_fs_path;

/// Files a bot may (re)write in its workspace. `BOOTSTRAP.md` is deliberately excluded
/// — only `finish_bootstrap` removes it — and `AGENTS.md` (operating rules) stays
/// operator-owned. Any path separator or traversal in the requested name is rejected.
const ALLOWED_WORKSPACE_FILES: &[&str] = &[
    "SOUL.md",
    "IDENTITY.md",
    "USER.md",
    "MEMORY.md",
    "TOOLS.md",
    "HEARTBEAT.md",
];

fn ok(call: &ToolCall, output: String) -> ToolResult {
    ToolResult {
        name: call.name.clone(),
        success: true,
        output,
    }
}

fn err(call: &ToolCall, msg: impl Into<String>) -> ToolResult {
    ToolResult {
        name: call.name.clone(),
        success: false,
        output: msg.into(),
    }
}

/// Derive `.../Agents/workspaces/{name}` from `.../Agents/{name}.md`.
fn workspace_dir_from_persona(persona_path: &str) -> Option<PathBuf> {
    let p = Path::new(persona_path);
    let stem = p.file_stem()?.to_str()?;
    if stem.is_empty() {
        return None;
    }
    let parent = p.parent()?;
    Some(parent.join("workspaces").join(stem))
}

/// The trusted, server-injected persona path (never a model-supplied argument).
fn persona_path_arg(call: &ToolCall) -> Option<String> {
    call.arguments
        .get("_persona_path")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
}

/// `write_workspace_file{ file, content }` — write one allowed markdown file into the
/// calling bot's own workspace. Used during bootstrap (the bot authors its SOUL /
/// IDENTITY / USER) and for self-editing memory (MEMORY.md / SOUL.md).
pub(crate) async fn execute_write_workspace_file(call: &ToolCall) -> ToolResult {
    let Some(persona_path) = persona_path_arg(call) else {
        return err(
            call,
            "workspace tools are only available inside a bot turn (no persona context).",
        );
    };
    let Some(workspace_dir) = workspace_dir_from_persona(&persona_path) else {
        return err(call, "could not derive a workspace directory for this bot.");
    };

    let file = call
        .arguments
        .get("file")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim();
    let content = call
        .arguments
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    if file.is_empty() {
        return err(
            call,
            "Missing 'file'. Allowed: SOUL.md, IDENTITY.md, USER.md, MEMORY.md, TOOLS.md, HEARTBEAT.md.",
        );
    }
    // Must be a bare filename — reject any separator or traversal before touching disk.
    if file.contains('/') || file.contains('\\') || file.contains("..") {
        return err(
            call,
            format!("Invalid file name '{file}': only a bare filename is allowed."),
        );
    }
    if !ALLOWED_WORKSPACE_FILES.contains(&file) {
        return err(
            call,
            format!(
                "File '{file}' is not writable. Allowed: {}.",
                ALLOWED_WORKSPACE_FILES.join(", ")
            ),
        );
    }

    // A bootstrapping workspace already exists; create_dir_all is a safe no-op then.
    if let Err(e) = std::fs::create_dir_all(&workspace_dir) {
        return err(call, format!("Failed to create workspace dir: {e}"));
    }
    let canonical_workspace = match std::fs::canonicalize(&workspace_dir) {
        Ok(p) => p,
        Err(e) => return err(call, format!("Failed to resolve workspace dir: {e}")),
    };
    let target = canonical_workspace.join(file);

    // Defense in depth: the generic Hera FS guard AND an explicit confinement to THIS
    // workspace. The BINDING constraint is `starts_with(canonical_workspace)` below —
    // canonical_workspace is derived from the trusted server-injected persona_path, so
    // it already pins the write to the bot's own dir. `resolve_guarded_fs_path` is the
    // outer net (allow_tmp=true keeps it from rejecting a legitimately temp-rooted
    // workspace; in production the workspace is always under the OS root anyway).
    let guarded = match resolve_guarded_fs_path(&target.to_string_lossy(), true) {
        Ok(p) => p,
        Err(e) => return err(call, e),
    };
    if !guarded.starts_with(&canonical_workspace) {
        return err(call, "refusing to write outside the bot's own workspace.");
    }

    match std::fs::write(&guarded, content) {
        Ok(_) => ok(
            call,
            format!("Wrote {file} ({} bytes) to your workspace.", content.len()),
        ),
        Err(e) => err(call, format!("Failed to write {file}: {e}")),
    }
}

/// `finish_bootstrap{}` — end the first-run ritual by deleting `BOOTSTRAP.md`. From the
/// next turn the bot runs on its assembled SOUL / IDENTITY / AGENTS.
pub(crate) async fn execute_finish_bootstrap(call: &ToolCall) -> ToolResult {
    let Some(persona_path) = persona_path_arg(call) else {
        return err(
            call,
            "workspace tools are only available inside a bot turn (no persona context).",
        );
    };
    let Some(workspace_dir) = workspace_dir_from_persona(&persona_path) else {
        return err(call, "could not derive a workspace directory for this bot.");
    };

    let bootstrap = workspace_dir.join("BOOTSTRAP.md");
    if !bootstrap.is_file() {
        return ok(
            call,
            "Bootstrap already complete — nothing to finish.".to_string(),
        );
    }

    // Confinement: the file we remove must resolve inside the derived workspace.
    let canonical_workspace = match std::fs::canonicalize(&workspace_dir) {
        Ok(p) => p,
        Err(e) => return err(call, format!("Failed to resolve workspace dir: {e}")),
    };
    let canonical_bootstrap = match std::fs::canonicalize(&bootstrap) {
        Ok(p) => p,
        Err(e) => return err(call, format!("Failed to resolve BOOTSTRAP.md: {e}")),
    };
    if !canonical_bootstrap.starts_with(&canonical_workspace) {
        return err(
            call,
            "refusing to remove a file outside the bot's own workspace.",
        );
    }

    match std::fs::remove_file(&canonical_bootstrap) {
        Ok(_) => ok(
            call,
            "Bootstrap complete. Your character is live — future turns use your SOUL/IDENTITY/AGENTS."
                .to_string(),
        ),
        Err(e) => err(call, format!("Failed to finish bootstrap: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn call(name: &str, args: serde_json::Value) -> ToolCall {
        ToolCall {
            name: name.to_string(),
            arguments: args,
        }
    }

    fn temp_workspace(tag: &str) -> (PathBuf, PathBuf) {
        // Layout: <base>/Agents/{name}.md + <base>/Agents/workspaces/{name}/
        let base = std::env::temp_dir().join(format!(
            "hera_ws_test_{}_{}",
            tag,
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&base);
        let agents = base.join("Agents");
        let ws = agents.join("workspaces").join("nova");
        std::fs::create_dir_all(&ws).unwrap();
        let persona = agents.join("nova.md");
        std::fs::write(&persona, "soul").unwrap();
        (persona, ws)
    }

    #[tokio::test]
    async fn writes_allowed_file_into_own_workspace() {
        let (persona, ws) = temp_workspace("write_ok");
        let res = execute_write_workspace_file(&call(
            "write_workspace_file",
            json!({"_persona_path": persona.to_str().unwrap(), "file": "SOUL.md", "content": "I am Nova"}),
        ))
        .await;
        assert!(res.success, "{}", res.output);
        assert_eq!(std::fs::read_to_string(ws.join("SOUL.md")).unwrap(), "I am Nova");
    }

    #[tokio::test]
    async fn rejects_path_traversal() {
        let (persona, _ws) = temp_workspace("traversal");
        let res = execute_write_workspace_file(&call(
            "write_workspace_file",
            json!({"_persona_path": persona.to_str().unwrap(), "file": "../nova.md", "content": "pwn"}),
        ))
        .await;
        assert!(!res.success);
        assert!(res.output.contains("bare filename"));
    }

    #[tokio::test]
    async fn rejects_disallowed_file() {
        let (persona, _ws) = temp_workspace("disallowed");
        let res = execute_write_workspace_file(&call(
            "write_workspace_file",
            json!({"_persona_path": persona.to_str().unwrap(), "file": "AGENTS.md", "content": "x"}),
        ))
        .await;
        assert!(!res.success);
        assert!(res.output.contains("not writable"));
    }

    #[tokio::test]
    async fn rejects_missing_persona_context() {
        let res = execute_write_workspace_file(&call(
            "write_workspace_file",
            json!({"file": "SOUL.md", "content": "x"}),
        ))
        .await;
        assert!(!res.success);
        assert!(res.output.contains("no persona context"));
    }

    #[tokio::test]
    async fn finish_bootstrap_removes_seed() {
        let (persona, ws) = temp_workspace("finish");
        std::fs::write(ws.join("BOOTSTRAP.md"), "interview me").unwrap();
        let res = execute_finish_bootstrap(&call(
            "finish_bootstrap",
            json!({"_persona_path": persona.to_str().unwrap()}),
        ))
        .await;
        assert!(res.success, "{}", res.output);
        assert!(!ws.join("BOOTSTRAP.md").exists());
    }
}
