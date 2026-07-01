//! Mission Control tool executors (Agente Q).
//!
//! Q operates the Agile Cockpit by calling OS-v3's `/api/pm/*` endpoints,
//! authenticated with the shared internal secret (`x-os-secret`). All `mc_*`
//! tools have `execution_kind: http_adapter` and route here via dispatch.

use crate::ai::tool_executor::{ToolCall, ToolResult};
use serde_json::{json, Map, Value};
use tracing::info;

const BASE: &str = "http://127.0.0.1:5177/api/pm";

fn shared_secret() -> Option<String> {
    let path = std::env::var("OS_AUTH_SHARED_SECRET_FILE").unwrap_or_else(|_| {
        "/home/paulo/.config/imagineos/secrets/os-auth-shared-secret".to_string()
    });
    std::fs::read_to_string(path).ok().map(|s| s.trim().to_string())
}

fn fail(call: &ToolCall, msg: &str) -> ToolResult {
    ToolResult {
        name: call.name.clone(),
        success: false,
        output: msg.to_string(),
    }
}

fn arg_str(call: &ToolCall, k: &str) -> Option<String> {
    call.arguments
        .get(k)
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .filter(|s| !s.trim().is_empty())
}

fn arg_i64(call: &ToolCall, k: &str) -> Option<i64> {
    call.arguments.get(k).and_then(serde_json::Value::as_i64)
}

async fn call_pm(call: &ToolCall, method: &str, path: &str, body: Option<Value>) -> ToolResult {
    let Some(secret) = shared_secret() else {
        return fail(call, "No pude leer el secret compartido (OS_AUTH_SHARED_SECRET_FILE).");
    };
    let url = format!("{BASE}{path}");
    let client = reqwest::Client::new();
    let req = if method == "GET" {
        client.get(&url)
    } else {
        client.post(&url).json(&body.unwrap_or_else(|| json!({})))
    };
    info!("🛠️ [Q] {method} {url}");
    match req
        .header("x-os-secret", &secret)
        .header("content-type", "application/json")
        .send()
        .await
    {
        Ok(resp) => {
            let success = resp.status().is_success();
            let text = resp.text().await.unwrap_or_default();
            ToolResult {
                name: call.name.clone(),
                success,
                output: text,
            }
        }
        Err(e) => fail(call, &format!("Error de red llamando al cockpit: {e}")),
    }
}

/// Build a JSON object from (key, Option<Value>) pairs, skipping None.
fn obj(pairs: Vec<(&str, Option<Value>)>) -> Value {
    let mut m = Map::new();
    for (k, v) in pairs {
        if let Some(val) = v {
            m.insert(k.to_string(), val);
        }
    }
    Value::Object(m)
}

pub(crate) async fn execute_mc_board(call: &ToolCall) -> ToolResult {
    let Some(project) = arg_str(call, "project") else {
        return fail(call, "Falta 'project'.");
    };
    call_pm(call, "GET", &format!("/board?project={project}"), None).await
}

pub(crate) async fn execute_mc_create_story(call: &ToolCall) -> ToolResult {
    let (Some(project), Some(title)) = (arg_str(call, "project"), arg_str(call, "title")) else {
        return fail(call, "Faltan 'project' y/o 'title'.");
    };
    let body = obj(vec![
        ("project", Some(json!(project))),
        ("title", Some(json!(title))),
        ("points", arg_i64(call, "points").map(|p| json!(p))),
        ("status", arg_str(call, "status").map(|s| json!(s))),
        ("epic", arg_str(call, "epic").map(|e| json!(e))),
    ]);
    call_pm(call, "POST", "/story", Some(body)).await
}

pub(crate) async fn execute_mc_move_story(call: &ToolCall) -> ToolResult {
    let Some(id) = arg_i64(call, "id") else {
        return fail(call, "Falta 'id' (úsalo desde mc_board).");
    };
    let body = obj(vec![
        ("id", Some(json!(id))),
        ("status", arg_str(call, "status").map(|s| json!(s))),
        ("points", arg_i64(call, "points").map(|p| json!(p))),
    ]);
    call_pm(call, "POST", "/story/move", Some(body)).await
}

pub(crate) async fn execute_mc_create_sprint(call: &ToolCall) -> ToolResult {
    let (Some(project), Some(name)) = (arg_str(call, "project"), arg_str(call, "name")) else {
        return fail(call, "Faltan 'project' y/o 'name'.");
    };
    let body = obj(vec![
        ("project", Some(json!(project))),
        ("name", Some(json!(name))),
        ("goal", arg_str(call, "goal").map(|g| json!(g))),
        ("days", arg_i64(call, "days").map(|d| json!(d))),
    ]);
    call_pm(call, "POST", "/sprint", Some(body)).await
}

pub(crate) async fn execute_mc_close_sprint(call: &ToolCall) -> ToolResult {
    let Some(id) = arg_i64(call, "id") else {
        return fail(call, "Falta 'id' del sprint.");
    };
    call_pm(call, "POST", "/sprint/close", Some(json!({"id": id}))).await
}

pub(crate) async fn execute_mc_add_wishlist(call: &ToolCall) -> ToolResult {
    let (Some(project), Some(title)) = (arg_str(call, "project"), arg_str(call, "title")) else {
        return fail(call, "Faltan 'project' y/o 'title'.");
    };
    call_pm(
        call,
        "POST",
        "/wishlist",
        Some(json!({"project": project, "title": title})),
    )
    .await
}

pub(crate) async fn execute_mc_set_objective(call: &ToolCall) -> ToolResult {
    let (Some(project), Some(title)) = (arg_str(call, "project"), arg_str(call, "title")) else {
        return fail(call, "Faltan 'project' y/o 'title'.");
    };
    let body = obj(vec![
        ("project", Some(json!(project))),
        ("title", Some(json!(title))),
        ("progress", arg_i64(call, "progress").map(|p| json!(p))),
    ]);
    call_pm(call, "POST", "/objective", Some(body)).await
}
