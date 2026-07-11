//! `index_code_graph` — indexa un crate Rust del monorepo al knowledge graph
//! (capa estructural `syn` + capa semántica GLiNER/GLiREL) vía el binario
//! `index_code_graph` del crate `code-graph-kit`. Mismo patrón shell-out que
//! `build_feedback::execute_cargo_check` — corre el binario ya compilado del
//! propio monorepo, no reimplementa la extracción aquí.
//!
//! Ver `Apps/OS-v3/code-graph-kit/` y el plan
//! `/home/paulo/.claude/plans/delegated-crunching-swing.md`.

use super::platform::resolve_guarded_fs_path;
use crate::ai::tool_executor::{ToolCall, ToolResult};
use std::process::Stdio;
use std::time::Duration;
use tracing::info;

const TIMEOUT_S: u64 = 300;
// El binario vive en el crate `code-graph-kit`, dentro del árbol de OS-v3.
// Se invoca desde ahí para que `cargo run -p code-graph-kit` resuelva su
// Cargo.toml sin depender del cwd de hera-core.
const OS_V3_DIR: &str = "Apps/OS-v3";

fn arg_str<'a>(call: &'a ToolCall, key: &str) -> &'a str {
    call.arguments.get(key).and_then(|v| v.as_str()).unwrap_or("")
}

fn ok(call: &ToolCall, success: bool, output: String) -> ToolResult {
    ToolResult { name: call.name.clone(), success, output }
}

fn err(call: &ToolCall, output: impl Into<String>) -> ToolResult {
    ToolResult { name: call.name.clone(), success: false, output: output.into() }
}

pub(crate) async fn execute_index_code_graph(call: &ToolCall) -> ToolResult {
    let path_arg = arg_str(call, "path");
    if path_arg.trim().is_empty() {
        return err(call, "missing 'path': ruta ABSOLUTA del crate a indexar (donde vive su Cargo.toml).");
    }
    let slug = arg_str(call, "slug");
    if slug.trim().is_empty() {
        return err(call, "missing 'slug': identificador corto del crate, ej. \"geo-kit\".");
    }
    let crate_dir = match resolve_guarded_fs_path(path_arg, true) {
        Ok(p) => p,
        Err(e) => return err(call, e),
    };
    if !crate_dir.join("Cargo.toml").is_file() {
        return err(call, format!("'{}' no tiene Cargo.toml.", crate_dir.display()));
    }

    let os_v3_dir = match resolve_guarded_fs_path(OS_V3_DIR, true) {
        Ok(p) => p,
        Err(e) => return err(call, e),
    };

    let child = tokio::process::Command::new("cargo")
        .args([
            "run",
            "--quiet",
            "-p",
            "code-graph-kit",
            "--bin",
            "index_code_graph",
            "--",
        ])
        .arg(crate_dir.display().to_string())
        .arg(slug)
        .current_dir(&os_v3_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn();

    let child = match child {
        Ok(c) => c,
        Err(e) => return err(call, format!("no se pudo lanzar index_code_graph: {e}")),
    };

    let output = match tokio::time::timeout(Duration::from_secs(TIMEOUT_S), child.wait_with_output()).await {
        Ok(Ok(out)) => out,
        Ok(Err(e)) => return err(call, format!("index_code_graph falló: {e}")),
        Err(_) => return err(call, format!("index_code_graph timeout tras {TIMEOUT_S}s.")),
    };

    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    info!("🕸️ [Hera] index_code_graph {slug} → success={}", output.status.success());

    if !output.status.success() {
        return err(
            call,
            format!("index_code_graph FAILED para '{slug}':\nstdout:\n{stdout}\nstderr:\n{}", stderr.trim()),
        );
    }

    let summary = stdout.lines().last().unwrap_or("").trim();
    ok(call, true, format!("index_code_graph OK para '{slug}': {summary}"))
}
