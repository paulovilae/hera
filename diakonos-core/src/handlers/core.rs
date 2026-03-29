use anyhow::{Result, anyhow};
use serde_json::json;

use crate::modules::{module_snapshot, ModuleRegistry};

pub async fn dispatch(action: &str, payload: serde_json::Value, modules: &ModuleRegistry) -> Result<serde_json::Value> {
    match action {
        "health" => Ok(json!({
            "service": "diakonos-core",
            "socket": "/tmp/diakonos.sock",
            "status": "ok"
        })),
        "capabilities" => Ok(json!({
            "service": "diakonos-core",
            "domains": ["workflow", "docs", "market", "vector", "web", "media"],
            "execution_model": "ipc_service"
        })),
        "list_modules" => Ok(json!({
            "config_path": modules.config_path().display().to_string(),
            "modules": module_snapshot(modules).await
        })),
        "reload_modules" => Ok(json!({
            "reloaded": modules.reload().await,
            "config_path": modules.config_path().display().to_string()
        })),
        "list_tools" => {
            let tools = hera_execution::tools::get_smartos_tools();
            Ok(serde_json::to_value(tools)?)
        }
        "read_file" => {
            let path = payload
                .get("path")
                .and_then(|value| value.as_str())
                .ok_or_else(|| anyhow!("Missing `path`"))?;
            let content = std::fs::read_to_string(path)?;
            let truncated = if content.len() > 16_000 {
                format!("{}... (truncated)", &content[..16_000])
            } else {
                content
            };
            Ok(json!({
                "path": path,
                "content": truncated
            }))
        }
        _ => Err(anyhow!("Unsupported core action `{}`", action)),
    }
}
