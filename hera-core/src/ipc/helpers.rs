//! IPC helper functions — Memento integration, model origin inference, token estimation.
use tokio::io::{AsyncReadExt, AsyncWriteExt};

fn memento_request(action: &str, payload: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "action": action,
        "payload": payload,
        "client": {
            "app": "hera",
            "token": std::env::var("MEMENTO_CLIENT_TOKEN").ok()
        }
    })
}

async fn call_memento(action: &str, payload: serde_json::Value) -> Option<serde_json::Value> {
    let mut stream = tokio::time::timeout(
        std::time::Duration::from_millis(1200),
        tokio::net::UnixStream::connect("/tmp/memento.sock"),
    )
    .await
    .ok()?
    .ok()?;

    let msg = memento_request(action, payload);
    stream.write_all(msg.to_string().as_bytes()).await.ok()?;
    let _ = stream.shutdown().await;

    let mut raw_bytes = Vec::new();
    tokio::time::timeout(
        std::time::Duration::from_millis(2000),
        stream.read_to_end(&mut raw_bytes),
    )
    .await
    .ok()?
    .ok()?;

    serde_json::from_slice::<serde_json::Value>(&raw_bytes).ok()
}

/// Infer whether the response came from a local or cloud engine.
pub fn infer_origin_from_model(model: &str) -> &'static str {
    let normalized = model.trim().to_lowercase();
    let openrouter_default = std::env::var("OPENROUTER_DEFAULT_MODEL")
        .unwrap_or_default()
        .trim()
        .to_lowercase();

    if !openrouter_default.is_empty() && normalized == openrouter_default {
        "cloud"
    } else if normalized.is_empty() {
        "unknown"
    } else {
        "local"
    }
}

/// Rough token estimation: 1 token ≈ 4 characters.
pub fn estimate_tokens(req: &crate::ai::ChatRequest) -> usize {
    let mut chars = 0;
    for m in &req.messages {
        match &m.content {
            crate::ai::MessageContent::Text(t) => chars += t.len(),
            crate::ai::MessageContent::Parts(parts) => {
                for p in parts {
                    if let crate::ai::ContentPart::Text { text } = p {
                        chars += text.len();
                    }
                }
            }
            crate::ai::MessageContent::Null => {}
        }
    }
    chars / 4
}

/// Fetch semantic memory from Memento for a specific app.
pub async fn fetch_semantic_memory(app_name: &str) -> String {
    if app_name.is_empty() {
        return String::new();
    }

    if let Ok(Ok(mut stream)) = tokio::time::timeout(
        std::time::Duration::from_millis(1000),
        tokio::net::UnixStream::connect("/tmp/memento.sock"),
    )
    .await
    {
        let msg = memento_request(
            "query_app",
            serde_json::json!({ "app": app_name, "query": "semantic_context" }),
        );
        if stream.write_all(msg.to_string().as_bytes()).await.is_ok() {
            let mut buffer = vec![0u8; 65536];
            if let Ok(Ok(n)) = tokio::time::timeout(
                std::time::Duration::from_millis(1500),
                stream.read(&mut buffer),
            )
            .await
                && n > 0
            {
                let raw = String::from_utf8_lossy(&buffer[..n]);
                if let Ok(resp) = serde_json::from_str::<serde_json::Value>(&raw)
                    && let Some(ctx) = resp.get("context").and_then(|c| c.as_str())
                    && !ctx.is_empty()
                {
                    return format!(
                        "\n\n[MEMENTO SEMANTIC CORTEX INJECTION: {}]\n{}\n",
                        app_name, ctx
                    );
                }
            }
        }
    }
    String::new()
}

pub async fn fetch_runtime_preflight(
    app_id: &str,
    route_profile: &str,
    persona_path: &str,
    mode: &str,
) -> Option<serde_json::Value> {
    if app_id.is_empty() {
        return None;
    }
    call_memento(
        "get_runtime_preflight",
        serde_json::json!({
            "app_id": app_id,
            "route_profile": route_profile,
            "persona_path": persona_path,
            "mode": mode
        }),
    )
    .await
}

pub async fn record_runtime_observation(payload: serde_json::Value) -> Option<serde_json::Value> {
    call_memento("record_runtime_observation", payload).await
}

pub async fn promote_runtime_hint(payload: serde_json::Value) {
    let _ = call_memento("promote_runtime_hint", payload).await;
}

pub async fn refresh_runtime_preflight(
    app_id: &str,
    route_profile: &str,
    persona_path: &str,
    mode: &str,
    fallback: Option<serde_json::Value>,
) -> Option<serde_json::Value> {
    fetch_runtime_preflight(app_id, route_profile, persona_path, mode)
        .await
        .or(fallback)
}

#[derive(Debug, Clone)]
pub struct RuntimePromotionContext<'a> {
    pub preflight: Option<serde_json::Value>,
    pub mode: &'a str,
    pub app_id: &'a str,
    pub route_profile: &'a str,
    pub persona_path: &'a str,
    pub session_id: &'a str,
    pub trace_id: &'a str,
    pub chat_id: &'a str,
    pub current_budget_mode: &'a str,
    pub persona_drift: bool,
    pub success: bool,
}

pub async fn record_observation_and_promote_runtime_hint(
    observation_payload: serde_json::Value,
    context: RuntimePromotionContext<'_>,
) {
    let _ = record_runtime_observation(observation_payload).await;
    let postflight = refresh_runtime_preflight(
        context.app_id,
        context.route_profile,
        context.persona_path,
        context.mode,
        context.preflight,
    )
    .await;

    if let Some(hint_payload) = maybe_build_runtime_hint_promotion(
        postflight.as_ref(),
        context.app_id,
        context.route_profile,
        context.session_id,
        context.trace_id,
        context.chat_id,
        context.current_budget_mode,
        context.persona_drift,
        context.success,
    ) {
        promote_runtime_hint(hint_payload).await;
    } else if let Some(hint_payload) = maybe_build_negative_runtime_hint_promotion(
        postflight.as_ref(),
        context.app_id,
        context.route_profile,
        context.session_id,
        context.trace_id,
        context.chat_id,
        context.current_budget_mode,
        context.persona_drift,
    ) {
        promote_runtime_hint(hint_payload).await;
    }
}

pub async fn save_agent_run_summary(payload: serde_json::Value) {
    let _ = call_memento("save_agent_run_summary", payload).await;
}

fn learned_hint_in_cooldown(preflight: &serde_json::Value) -> bool {
    preflight
        .get("learned_hints")
        .and_then(|value| value.as_array())
        .map(|items| !items.is_empty())
        .unwrap_or(false)
}

pub fn maybe_build_runtime_hint_promotion(
    preflight: Option<&serde_json::Value>,
    app_id: &str,
    route_profile: &str,
    session_id: &str,
    trace_id: &str,
    chat_id: &str,
    current_budget_mode: &str,
    persona_drift: bool,
    success: bool,
) -> Option<serde_json::Value> {
    if !success || persona_drift || app_id.is_empty() || route_profile.is_empty() {
        return None;
    }

    let preflight = preflight?;
    let recommended_budget_mode = preflight
        .get("recommended_budget_mode")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())?;
    if recommended_budget_mode != current_budget_mode {
        return None;
    }

    if learned_hint_in_cooldown(preflight) {
        return None;
    }

    let matching_observation_count = preflight
        .get("matching_observation_count")
        .and_then(|value| value.as_u64())
        .unwrap_or(0);
    if matching_observation_count < 2 {
        return None;
    }

    if preflight
        .get("known_regressions")
        .and_then(|value| value.as_array())
        .map(|items| !items.is_empty())
        .unwrap_or(false)
    {
        return None;
    }

    let has_equivalent_hint = preflight
        .get("learned_hints")
        .and_then(|value| value.as_array())
        .map(|items| {
            items.iter().any(|item| {
                item.pointer("/data/recommended_budget_mode")
                    .and_then(|value| value.as_str())
                    == Some(current_budget_mode)
            })
        })
        .unwrap_or(false);
    if has_equivalent_hint {
        return None;
    }

    let source_record_id = preflight
        .pointer("/latest_observation/record_id")
        .and_then(|value| value.as_i64());
    let health_status = preflight
        .get("health_status")
        .and_then(|value| value.as_str())
        .unwrap_or("unknown");
    let title = format!("{} {} stable runtime hint", app_id, route_profile);
    let content = format!(
        "Promoted stable runtime hint for {} / {} after {} prior healthy observations. Keep context budget mode {}. health_status={}.",
        app_id, route_profile, matching_observation_count, current_budget_mode, health_status
    );

    Some(serde_json::json!({
        "app_id": app_id,
        "route_profile": route_profile,
        "session_id": session_id,
        "trace_id": trace_id,
        "chat_id": chat_id,
        "recommended_budget_mode": current_budget_mode,
        "title": title,
        "content": content,
        "hint_kind": "positive",
        "hint_ttl_hours": 24,
        "confidence": 0.91,
        "source_record_id": source_record_id
    }))
}

pub fn maybe_build_negative_runtime_hint_promotion(
    preflight: Option<&serde_json::Value>,
    app_id: &str,
    route_profile: &str,
    session_id: &str,
    trace_id: &str,
    chat_id: &str,
    current_budget_mode: &str,
    persona_drift: bool,
) -> Option<serde_json::Value> {
    if persona_drift || app_id.is_empty() || route_profile.is_empty() {
        return None;
    }

    let preflight = preflight?;
    if learned_hint_in_cooldown(preflight) {
        return None;
    }

    let recommended_budget_mode = preflight
        .get("recommended_budget_mode")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())?;
    if recommended_budget_mode == current_budget_mode {
        return None;
    }

    let regressions = preflight
        .get("known_regressions")
        .and_then(|value| value.as_array())?;
    if regressions.len() < 2 {
        return None;
    }

    let has_equivalent_hint = preflight
        .get("learned_hints")
        .and_then(|value| value.as_array())
        .map(|items| {
            items.iter().any(|item| {
                item.pointer("/data/recommended_budget_mode")
                    .and_then(|value| value.as_str())
                    == Some(recommended_budget_mode)
            })
        })
        .unwrap_or(false);
    if has_equivalent_hint {
        return None;
    }

    let source_record_id = preflight
        .pointer("/latest_observation/record_id")
        .and_then(|value| value.as_i64());
    let title = format!(
        "{} {} avoid {} budget hint",
        app_id, route_profile, current_budget_mode
    );
    let content = format!(
        "Avoid context budget mode {} for {} / {}. Repeated regressions observed; prefer {} instead.",
        current_budget_mode, app_id, route_profile, recommended_budget_mode
    );

    Some(serde_json::json!({
        "app_id": app_id,
        "route_profile": route_profile,
        "session_id": session_id,
        "trace_id": trace_id,
        "chat_id": chat_id,
        "recommended_budget_mode": recommended_budget_mode,
        "title": title,
        "content": content,
        "hint_kind": "negative",
        "hint_ttl_hours": 12,
        "confidence": 0.93,
        "source_record_id": source_record_id
    }))
}

/// Fetch live DB schema from Memento for injection into tool context.
/// - SOUL (Ava) gets ALL app schemas via describe_all_apps
/// - App-specific agents get only their app's schema via describe_app
pub async fn fetch_db_schema_context(agent_identity: &str, app_name: &str) -> String {
    let app_slug = match agent_identity.to_lowercase().as_str() {
        "soul" | "ava" | "gemini_soul" => {
            return fetch_all_apps_schema().await;
        }
        "vetra" | "vetra_soul" => "vetra",
        "movilo" | "movilo_soul" => "movilo",
        "latinos" | "latinos_soul" => "latinos",
        "garcero" | "garcero_soul" => "garcero",
        _ => {
            if !app_name.is_empty() {
                app_name
            } else {
                return String::new();
            }
        }
    };

    fetch_single_app_schema(app_slug).await
}

/// Fetch schema for a single app from Memento.
pub async fn fetch_single_app_schema(app_slug: &str) -> String {
    if let Ok(Ok(mut stream)) = tokio::time::timeout(
        std::time::Duration::from_millis(2000),
        tokio::net::UnixStream::connect("/tmp/memento.sock"),
    )
    .await
    {
        let msg = memento_request("describe_app", serde_json::json!({ "app": app_slug }));
        if stream.write_all(msg.to_string().as_bytes()).await.is_ok() {
            let _ = stream.shutdown().await;
            let mut raw_bytes = Vec::new();
            if let Ok(Ok(n)) = tokio::time::timeout(
                std::time::Duration::from_millis(3000),
                stream.read_to_end(&mut raw_bytes),
            )
            .await
                && n > 0
            {
                let raw = String::from_utf8_lossy(&raw_bytes);
                if let Ok(resp) = serde_json::from_str::<serde_json::Value>(&raw)
                    && resp.get("status").and_then(|s| s.as_str()) == Some("success")
                    && let Some(schema) = resp.get("schema").and_then(|s| s.as_object())
                {
                    return format_schema_for_prompt(app_slug, schema);
                }
            }
        }
    }
    String::new()
}

/// Fetch schemas for ALL registered apps from Memento.
pub async fn fetch_all_apps_schema() -> String {
    if let Ok(Ok(mut stream)) = tokio::time::timeout(
        std::time::Duration::from_millis(3000),
        tokio::net::UnixStream::connect("/tmp/memento.sock"),
    )
    .await
    {
        let msg = memento_request("describe_all_apps", serde_json::json!({}));
        if stream.write_all(msg.to_string().as_bytes()).await.is_ok() {
            let _ = stream.shutdown().await;
            let mut raw_bytes = Vec::new();
            if let Ok(Ok(n)) = tokio::time::timeout(
                std::time::Duration::from_millis(5000),
                stream.read_to_end(&mut raw_bytes),
            )
            .await
                && n > 0
            {
                let raw = String::from_utf8_lossy(&raw_bytes);
                if let Ok(resp) = serde_json::from_str::<serde_json::Value>(&raw)
                    && let Some(apps) = resp.get("apps").and_then(|a| a.as_object())
                {
                    let mut output = String::from("\n\n# DATABASE SCHEMA (Auto-Discovered)\n");
                    for (slug, tables_val) in apps {
                        if let Some(tables) = tables_val.as_object() {
                            output.push_str(&format!("\n## App: {}\n", slug));
                            for (table, cols) in tables {
                                let col_names: Vec<String> = cols
                                    .as_array()
                                    .map(|arr| {
                                        arr.iter()
                                            .filter_map(|c| {
                                                c.get("column")
                                                    .and_then(|n| n.as_str())
                                                    .map(|s| s.to_string())
                                            })
                                            .collect()
                                    })
                                    .unwrap_or_default();
                                output.push_str(&format!(
                                    "- {} ({})\n",
                                    table,
                                    col_names.join(", ")
                                ));
                            }
                        }
                    }
                    return output;
                }
            }
        }
    }
    String::new()
}

/// Format a single app's schema into a human-readable prompt fragment.
pub fn format_schema_for_prompt(
    app_slug: &str,
    schema: &serde_json::Map<String, serde_json::Value>,
) -> String {
    let mut output = format!(
        "\n\n# DATABASE SCHEMA for '{}' (Auto-Discovered)\nUse these EXACT table and column names when writing SQL queries with memento_query.\n",
        app_slug
    );
    for (table, cols) in schema {
        let col_names: Vec<String> = cols
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|c| {
                        c.get("column")
                            .and_then(|n| n.as_str())
                            .map(|s| s.to_string())
                    })
                    .collect()
            })
            .unwrap_or_default();
        output.push_str(&format!("- {} ({})\n", table, col_names.join(", ")));
    }
    output
}
