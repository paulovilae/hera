//! Sovereign object-storage tool executors (MinIO / S3) via the `mc` client.
//!
//! El object store soberano (MinIO en genesis:9000) queda disponible para los
//! agentes: listar objetos, generar una URL temporal de descarga y subir un
//! archivo local. Usa el cliente `mc` (un binario) en vez de acoplar Hera a un
//! crate S3 — el alias `localminio` se configura una vez con las credenciales
//! root de MinIO. Binario configurable vía `MC_BIN` (default `/home/paulo/bin/mc`).

use crate::ai::tool_executor::{ToolCall, ToolResult};
use tokio::process::Command;

const ALIAS: &str = "localminio";

fn mc_bin() -> String {
    std::env::var("MC_BIN").unwrap_or_else(|_| "/home/paulo/bin/mc".to_string())
}

fn ok(call: &ToolCall, output: String) -> ToolResult {
    ToolResult { name: call.name.clone(), success: true, output }
}
fn err(call: &ToolCall, output: String) -> ToolResult {
    ToolResult { name: call.name.clone(), success: false, output }
}

async fn run_mc(args: &[&str]) -> Result<String, String> {
    let out = Command::new(mc_bin())
        .args(args)
        .output()
        .await
        .map_err(|e| format!("no se pudo ejecutar mc: {e}"))?;
    if out.status.success() {
        Ok(String::from_utf8_lossy(&out.stdout).to_string())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).trim().to_string())
    }
}

/// Sube un archivo local: restringido a raíces de trabajo (sin `..`, dentro de
/// /tmp, el repo OS o el home) para no exfiltrar archivos arbitrarios del sistema.
fn source_path_allowed(path: &str) -> bool {
    !path.contains("..")
        && (path.starts_with("/tmp/")
            || path.starts_with("/home/paulo/")
            || path.starts_with("/mnt/workspace/"))
}

/// `storage_list` — lista objetos de un bucket o prefijo (`bucket` o `bucket/prefijo`).
pub(crate) async fn execute_storage_list(call: &ToolCall) -> ToolResult {
    let path = call
        .arguments
        .get("path")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .trim_start_matches('/');
    let target = if path.is_empty() {
        ALIAS.to_string()
    } else {
        format!("{ALIAS}/{path}")
    };
    match run_mc(&["ls", &target]).await {
        Ok(o) if o.trim().is_empty() => ok(call, "(vacío)".to_string()),
        Ok(o) => ok(call, o),
        Err(e) => err(call, format!("storage_list falló: {e}")),
    }
}

/// `storage_get_url` — genera una URL temporal de descarga de un objeto.
pub(crate) async fn execute_storage_get_url(call: &ToolCall) -> ToolResult {
    let key = call
        .arguments
        .get("key")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .trim_start_matches('/');
    if key.is_empty() {
        return err(call, "Falta 'key' (p.ej. 'movilo-uploads/foto.jpg').".to_string());
    }
    let expire = call
        .arguments
        .get("expire")
        .and_then(|v| v.as_str())
        .unwrap_or("1h");
    match run_mc(&["share", "download", "--expire", expire, &format!("{ALIAS}/{key}")]).await {
        Ok(o) => {
            let url = o
                .lines()
                .find_map(|l| l.trim().strip_prefix("Share: "))
                .map(|s| s.to_string())
                .unwrap_or_else(|| o.trim().to_string());
            ok(call, url)
        }
        Err(e) => err(call, format!("storage_get_url falló: {e}")),
    }
}

/// `storage_put` — sube un archivo local al object store en `key` (`bucket/objeto`).
pub(crate) async fn execute_storage_put(call: &ToolCall) -> ToolResult {
    let src = call
        .arguments
        .get("source_path")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim();
    let key = call
        .arguments
        .get("key")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .trim_start_matches('/');
    if src.is_empty() || key.is_empty() {
        return err(call, "Faltan 'source_path' y/o 'key'.".to_string());
    }
    if !source_path_allowed(src) {
        return err(
            call,
            format!("Ruta '{src}' fuera de las raíces permitidas (/tmp, repo OS, home)."),
        );
    }
    match run_mc(&["cp", src, &format!("{ALIAS}/{key}")]).await {
        Ok(_) => ok(call, format!("Subido a {key}.")),
        Err(e) => err(call, format!("storage_put falló: {e}")),
    }
}
