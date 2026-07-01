//! Brand Studio tool executors (Paulo's personal-brand content studio).
//!
//! These tools operate the brand pipeline by calling PauloVila-rust's
//! `/api/brand/*` endpoints, authenticated with the shared internal secret
//! (`x-os-secret`). All 7 CRUD tools have `execution_kind: http_adapter` and
//! route here via dispatch. The 2 generation tools (generate_post,
//! format_for_platform) stay skeleton/hidden until the voice profile is seeded.

use crate::ai::tool_executor::{ToolCall, ToolResult};
use serde_json::{json, Map, Value};
use tracing::info;

const BASE: &str = "http://127.0.0.1:5176/api/brand";

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

fn arg_f64(call: &ToolCall, k: &str) -> Option<f64> {
    call.arguments.get(k).and_then(serde_json::Value::as_f64)
}

fn arg_bool(call: &ToolCall, k: &str) -> Option<bool> {
    call.arguments.get(k).and_then(serde_json::Value::as_bool)
}

/// Pass an argument through verbatim if present (arrays, etc.).
fn arg_raw(call: &ToolCall, k: &str) -> Option<Value> {
    call.arguments.get(k).cloned()
}

async fn call_brand(call: &ToolCall, path: &str, body: Value) -> ToolResult {
    let Some(secret) = shared_secret() else {
        return fail(call, "No pude leer el secret compartido (OS_AUTH_SHARED_SECRET_FILE).");
    };
    let url = format!("{BASE}{path}");
    let client = reqwest::Client::new();
    info!("🛠️ [brand] POST {url}");
    match client
        .post(&url)
        .json(&body)
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
        Err(e) => fail(call, &format!("Error de red llamando a brand studio: {e}")),
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

pub(crate) async fn execute_add_topic(call: &ToolCall) -> ToolResult {
    let Some(topic) = arg_str(call, "topic") else {
        return fail(call, "Falta 'topic'.");
    };
    let body = obj(vec![
        ("topic", Some(json!(topic))),
        ("owner", arg_str(call, "owner").map(|o| json!(o))),
        ("priority", arg_i64(call, "priority").map(|p| json!(p))),
        ("suggested_platform", arg_str(call, "suggested_platform").map(|p| json!(p))),
        ("suggested_thesis_refs", arg_raw(call, "suggested_thesis_refs")),
    ]);
    call_brand(call, "/topic", body).await
}

pub(crate) async fn execute_list_pending_drafts(call: &ToolCall) -> ToolResult {
    let body = obj(vec![
        ("owner", arg_str(call, "owner").map(|o| json!(o))),
        ("status", arg_raw(call, "status")),
        ("platform", arg_str(call, "platform").map(|p| json!(p))),
        ("limit", arg_i64(call, "limit").map(|l| json!(l))),
    ]);
    call_brand(call, "/drafts/list", body).await
}

pub(crate) async fn execute_approve_draft(call: &ToolCall) -> ToolResult {
    let (Some(draft_id), Some(action)) = (arg_i64(call, "draft_id"), arg_str(call, "action")) else {
        return fail(call, "Faltan 'draft_id' y/o 'action'.");
    };
    let body = obj(vec![
        ("draft_id", Some(json!(draft_id))),
        ("action", Some(json!(action))),
        ("scheduled_for", arg_str(call, "scheduled_for").map(|s| json!(s))),
        ("post_url", arg_str(call, "post_url").map(|u| json!(u))),
        ("feedback", arg_str(call, "feedback").map(|f| json!(f))),
    ]);
    call_brand(call, "/draft/transition", body).await
}

pub(crate) async fn execute_capture_post_metrics(call: &ToolCall) -> ToolResult {
    let (Some(draft_id), Some(capture_source)) =
        (arg_i64(call, "draft_id"), arg_str(call, "capture_source"))
    else {
        return fail(call, "Faltan 'draft_id' y/o 'capture_source'.");
    };
    let body = obj(vec![
        ("draft_id", Some(json!(draft_id))),
        ("capture_source", Some(json!(capture_source))),
        ("views", arg_i64(call, "views").map(|v| json!(v))),
        ("likes", arg_i64(call, "likes").map(|v| json!(v))),
        ("comments", arg_i64(call, "comments").map(|v| json!(v))),
        ("shares", arg_i64(call, "shares").map(|v| json!(v))),
        ("clicks", arg_i64(call, "clicks").map(|v| json!(v))),
        ("saves", arg_i64(call, "saves").map(|v| json!(v))),
        ("raw_payload", arg_str(call, "raw_payload").map(|r| json!(r))),
    ]);
    call_brand(call, "/metrics", body).await
}

pub(crate) async fn execute_voice_profile_get(call: &ToolCall) -> ToolResult {
    let body = obj(vec![
        ("owner", arg_str(call, "owner").map(|o| json!(o))),
        ("dimensions", arg_raw(call, "dimensions")),
        ("confidence_min", arg_f64(call, "confidence_min").map(|c| json!(c))),
        ("include_inactive", arg_bool(call, "include_inactive").map(|b| json!(b))),
    ]);
    call_brand(call, "/voice/get", body).await
}

pub(crate) async fn execute_voice_profile_update(call: &ToolCall) -> ToolResult {
    let Some(dimension) = arg_str(call, "dimension") else {
        return fail(call, "Falta 'dimension'.");
    };
    let body = obj(vec![
        ("dimension", Some(json!(dimension))),
        ("owner", arg_str(call, "owner").map(|o| json!(o))),
        ("value", arg_str(call, "value").map(|v| json!(v))),
        ("evidence_refs", arg_raw(call, "evidence_refs")),
        ("confidence_delta", arg_f64(call, "confidence_delta").map(|c| json!(c))),
        ("deactivate", arg_bool(call, "deactivate").map(|b| json!(b))),
    ]);
    call_brand(call, "/voice/update", body).await
}

pub(crate) async fn execute_save_thesis_doc(call: &ToolCall) -> ToolResult {
    let (Some(slug), Some(stance), Some(title), Some(body_md)) = (
        arg_str(call, "slug"),
        arg_str(call, "stance"),
        arg_str(call, "title"),
        arg_str(call, "body"),
    ) else {
        return fail(call, "Faltan 'slug', 'stance', 'title' y/o 'body'.");
    };
    let body = obj(vec![
        ("slug", Some(json!(slug))),
        ("stance", Some(json!(stance))),
        ("title", Some(json!(title))),
        ("body", Some(json!(body_md))),
        ("owner", arg_str(call, "owner").map(|o| json!(o))),
        ("supporting_data", arg_str(call, "supporting_data").map(|s| json!(s))),
    ]);
    call_brand(call, "/thesis", body).await
}
