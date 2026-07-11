//! Sovereign browser-automation tool executor (http_adapter) — docs/BROWSER_AGENT_PLAN.md.
//!
//! Llama al servicio Python `browser_service.py` (browser-use sobre Playwright + Chrome,
//! genesis, puerto por defecto `:8096`, ver `scripts/start_browser_agent.sh`). Los
//! guardrails de seguridad (allowlist de dominios nativo de browser-use + gate de
//! confirmación pre-ejecución vía `register_new_step_callback`+`agent.stop()`) viven en
//! ESE servicio, a nivel de código — este executor solo reenvía la request y el
//! `status` de la respuesta ("done" | "pending_confirmation" | "daily_cap_reached" |
//! "error"), nunca decide él mismo si algo es seguro de ejecutar.
use crate::ai::tool_executor::{ToolCall, ToolResult};
use serde_json::json;
use tracing::info;

/// Navegación web soberana (http_adapter): delega en `browser_service.py` vía HTTP local.
/// URL del servicio en `BROWSER_ACTION_URL` (default `http://127.0.0.1:8096`).
pub(crate) async fn execute_browser_action(call: &ToolCall) -> ToolResult {
    let task = call
        .arguments
        .get("task")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim();
    if task.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Falta el parámetro 'task'.".into(),
        };
    }

    let allowed_domains: Vec<String> = call
        .arguments
        .get("allowed_domains")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();
    let account_key = call.arguments.get("account_key").and_then(|v| v.as_str());
    let use_vision = call
        .arguments
        .get("use_vision")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let confirm_token = call.arguments.get("confirm_token").and_then(|v| v.as_str());
    let max_steps = call
        .arguments
        .get("max_steps")
        .and_then(|v| v.as_i64())
        .unwrap_or(15);
    let daily_cap = call
        .arguments
        .get("daily_cap")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);

    let base = std::env::var("BROWSER_ACTION_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:8096".to_string());
    let url = format!("{}/browse", base.trim_end_matches('/'));

    let payload = json!({
        "task": task,
        "allowed_domains": allowed_domains,
        "account_key": account_key,
        "use_vision": use_vision,
        "confirm_token": confirm_token,
        "max_steps": max_steps,
        "daily_cap": daily_cap,
    });

    // Timeout del cliente HTTP por debajo del timeout_ms del tool (150000 en el JSON)
    // para que este executor devuelva un error legible en vez de que el dispatcher
    // externo mate la llamada primero sin mensaje (gotcha ya visto en review_image.json).
    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(140))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            return ToolResult {
                name: call.name.clone(),
                success: false,
                output: format!("No se pudo construir el cliente HTTP: {e}"),
            }
        }
    };

    match client.post(&url).json(&payload).send().await {
        Ok(resp) => match resp.json::<serde_json::Value>().await {
            Ok(body) => {
                let status = body.get("status").and_then(|v| v.as_str()).unwrap_or("error");
                let success = status == "done";
                info!("🌐 [Hera] browser_action status={status} task={task:.60}");
                ToolResult {
                    name: call.name.clone(),
                    success,
                    output: body.to_string(),
                }
            }
            Err(e) => ToolResult {
                name: call.name.clone(),
                success: false,
                output: format!("browser-agent devolvió una respuesta no-JSON: {e}"),
            },
        },
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!(
                "No se pudo contactar al servicio browser-agent ({url}): {e}. \
                 ¿Está corriendo `pm2 status browser-agent` en genesis?"
            ),
        },
    }
}
