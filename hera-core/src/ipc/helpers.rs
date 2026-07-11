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

/// Public passthrough for tool executors that need a raw Memento IPC call
/// (e.g. `query_code_graph` wrapping `kg_graph`/`kg_neighbors`/`kg_centrality`/
/// `kg_path`/`kg_communities`). Prefer a purpose-built helper for anything
/// that runs on every turn — this is for explicit, on-demand tool calls.
pub async fn call_memento_action(action: &str, payload: serde_json::Value) -> Option<serde_json::Value> {
    call_memento(action, payload).await
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

/// Persist one pass of a `delegate_task` goal-loop as a durable `open_loop`
/// scoped_memory row. Fire-and-forget (errors only log). The registry that
/// tracks live delegate runs is in-memory only, so a `hera-core` restart would
/// silently drop the loop's progress; this row survives the restart and lets a
/// future turn recall "goal X was pursued, pass N/M, verdict …". Reuses the same
/// Memento `save_scoped_memory` action + payload shape as `save_chat_turn_event`
/// (memory_type `open_loop` is one of the durable types the self-editing
/// `save_memory` tool already emits) so no new Memento surface is introduced.
#[allow(clippy::too_many_arguments)]
pub async fn save_open_loop_memory(
    app_id: String,
    session_id: String,
    run_id: String,
    goal: String,
    pass: u32,
    max_passes: u32,
    satisfied: bool,
    verdict_reason: String,
    result_summary: String,
) {
    if goal.trim().is_empty() {
        return;
    }
    let user_id = if !session_id.trim().is_empty() {
        format!("session:{session_id}")
    } else if !app_id.trim().is_empty() {
        format!("app:{app_id}")
    } else {
        "anonymous".to_string()
    };
    // Store the structured state AS the content (compact JSON) so a restart can
    // parse it back, while still being human-readable in Memento.
    let content = serde_json::json!({
        "run_id": run_id,
        "goal": goal,
        "pass": pass,
        "max_passes": max_passes,
        "satisfied": satisfied,
        "verdict": if satisfied { "SATISFIED" } else { "CONTINUE" },
        "reason": verdict_reason,
        "result_summary": result_summary,
    })
    .to_string();
    let payload = serde_json::json!({
        "user_id": user_id,
        "tenant_id": "default",
        "app_id": app_id,
        "session_id": session_id,
        "scope": "personal",
        "source": "hera_goal_loop",
        "memory_type": "open_loop",
        "content": content,
        "tags": ["goal_loop", if satisfied { "satisfied" } else { "continue" }],
        "auto_derive": false,
    });
    if let Some(resp) = call_memento("save_scoped_memory", payload).await
        && let Some(err) = resp.get("error").and_then(|v| v.as_str())
    {
        tracing::warn!("save_open_loop_memory failed: {}", err);
    }
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
/// - Agents without memento_query access (e.g. memo) get NO schema — they use
///   high-level pre-built tools and a raw SQL surface only confuses them.
pub async fn fetch_db_schema_context(agent_identity: &str, app_name: &str) -> String {
    let app_slug = match agent_identity.to_lowercase().as_str() {
        "soul" | "ava" | "gemini_soul" => {
            return fetch_all_apps_schema().await;
        }
        "memo" => return String::new(),
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
    fetch_single_app_schema_json(app_slug)
        .await
        .as_ref()
        .map(|schema| format_schema_for_prompt(app_slug, schema))
        .unwrap_or_default()
}

pub async fn fetch_single_app_schema_json(
    app_slug: &str,
) -> Option<serde_json::Map<String, serde_json::Value>> {
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
                    return Some(schema.clone());
                }
            }
        }
    }
    None
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

// ─── Local embedding (in-process candle, local-llm builds only) ────────────

/// Embed a single text to a vector using the in-process candle model. Returns
/// None on CPU-only builds (no `local-llm`) or on any failure, so callers
/// degrade gracefully to keyword recall.
#[cfg(feature = "embeddings")]
pub fn embed_text_local(text: &str) -> Option<Vec<f32>> {
    let owned = vec![text.to_string()];
    match crate::ai::embeddings::embed_texts(&owned) {
        Ok(mut v) => v.pop(),
        Err(e) => {
            tracing::warn!("embed_text_local failed: {e}");
            None
        }
    }
}

#[cfg(not(feature = "embeddings"))]
pub fn embed_text_local(_text: &str) -> Option<Vec<f32>> {
    None
}

/// A memory entry returned by Memento's `semantic_recall`, captured for
/// post-generation attribution (recall feedback flywheel).
#[derive(Debug, Clone)]
pub struct RecalledEntry {
    pub id: i64,
    pub content: String,
}

/// Attribution payload threaded from recall-time to post-generation. Holds the
/// Memento-side `request_id` so the feedback insert lands on the right row, and
/// the recalled entries so we can match them against the model's output to
/// decide which were actually cited.
#[derive(Debug, Clone, Default)]
pub struct RecallAttribution {
    pub request_id: String,
    pub entries: Vec<RecalledEntry>,
}

/// Fetch semantically relevant memories for (user_id, app_id, session_id) by
/// embedding the query and asking Memento's semantic_recall to cosine-rerank the
/// scope-filtered rows. Returns the formatted prompt fragment plus an
/// attribution payload (None when recall was skipped or failed) that downstream
/// handlers use to report `recall_feedback` after generation.
pub async fn fetch_semantic_memories(
    user_id: &str,
    app_id: &str,
    session_id: &str,
    query: &str,
) -> (String, Option<RecallAttribution>) {
    if user_id.is_empty() || query.trim().is_empty() {
        return (String::new(), None);
    }
    let q = query.to_string();
    let embedding = match tokio::task::spawn_blocking(move || embed_text_local(&q)).await {
        Ok(Some(e)) => e,
        _ => return (String::new(), None),
    };

    let mut payload = serde_json::json!({
        "user_id": user_id,
        "query_embedding": embedding,
        "query_text": query,
        "limit": 4,
        "min_score": 0.3,
    });
    if !app_id.is_empty() {
        payload["app_id"] = serde_json::Value::String(app_id.to_string());
    }
    if !session_id.is_empty() {
        payload["session_id"] = serde_json::Value::String(session_id.to_string());
    }

    let Some(resp) = call_memento("semantic_recall", payload).await else {
        return (String::new(), None);
    };
    let request_id = resp
        .get("request_id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let Some(entries) = resp.get("entries").and_then(|v| v.as_array()) else {
        return (String::new(), None);
    };

    let mut section = String::new();
    let mut captured: Vec<RecalledEntry> = Vec::new();
    for item in entries.iter().take(4) {
        let content = match item.get("content").and_then(|v| v.as_str()) {
            Some(c) => c.trim(),
            None => continue,
        };
        if content.is_empty() {
            continue;
        }
        let snippet: String = content.chars().take(240).collect();
        section.push_str(&format!("- {}\n", snippet));
        // The id may arrive as i32 (scoped_memory.id is SERIAL); accept either width.
        let id = item
            .get("id")
            .and_then(|v| v.as_i64())
            .unwrap_or_else(|| item.get("id").and_then(|v| v.as_u64()).unwrap_or(0) as i64);
        if id > 0 {
            captured.push(RecalledEntry {
                id,
                content: content.to_string(),
            });
        }
    }
    if section.is_empty() {
        return (String::new(), None);
    }
    let prompt_fragment = format!(
        "\n\n# SEMANTICALLY RELEVANT MEMORIES (Memento cosine recall)\n{}",
        section
    );
    let attribution = if !request_id.is_empty() && !captured.is_empty() {
        Some(RecallAttribution {
            request_id,
            entries: captured,
        })
    } else {
        None
    };
    (prompt_fragment, attribution)
}

// ─── Recursive scoped-memory wiring (Memento <-> Hera) ─────────────────────

/// Derive a stable user_id for Memento scoped_memory from the identifiers Hera receives.
/// Falls back: sender_name (canonicalized) -> "chat:<chat_id>" -> "anonymous:<session_id>".
pub fn canonicalize_user_id(sender_name: &str, chat_id: &str, session_id: &str) -> String {
    let sender = sender_name.trim();
    if !sender.is_empty() {
        let canonical: String = sender
            .to_lowercase()
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '_' })
            .collect();
        let trimmed = canonical.trim_matches('_').to_string();
        if !trimmed.is_empty() {
            return trimmed;
        }
    }
    let chat = chat_id.trim();
    if !chat.is_empty() {
        return format!("chat:{}", chat);
    }
    let session = session_id.trim();
    if !session.is_empty() {
        return format!("anonymous:{}", session);
    }
    "anonymous".to_string()
}

fn format_recursive_context(ctx: &serde_json::Value) -> String {
    let mut buf = String::new();
    let mut push_list = |label: &str, list: Option<&serde_json::Value>, take: usize, max_chars: usize| {
        let Some(items) = list.and_then(|v| v.as_array()) else { return; };
        if items.is_empty() {
            return;
        }
        let mut section = String::new();
        for item in items.iter().take(take) {
            if let Some(content) = item.get("content").and_then(|v| v.as_str()) {
                let trimmed = content.trim();
                if !trimmed.is_empty() {
                    let snippet: String = trimmed.chars().take(max_chars).collect();
                    section.push_str(&format!("- {}\n", snippet));
                }
            }
        }
        if !section.is_empty() {
            buf.push_str(&format!("\n## {}\n{}", label, section));
        }
    };

    push_list("Project context", ctx.get("project_summaries"), 3, 280);
    push_list("Room context", ctx.get("room_summaries"), 3, 280);
    push_list("Session context", ctx.get("session_summaries"), 3, 280);
    push_list("Durable facts", ctx.get("durable_facts"), 3, 220);
    push_list("Recent events", ctx.get("recent_events"), 3, 220);

    if let Some(working) = ctx.get("working_context").and_then(|v| v.as_object()) {
        let mut working_buf = String::new();
        for key in ["summaries", "decisions", "preferences", "open_loops"] {
            if let Some(entries) = working.get(key).and_then(|v| v.as_array()) {
                for item in entries.iter().take(2) {
                    if let Some(content) = item.get("content").and_then(|v| v.as_str()) {
                        let trimmed = content.trim();
                        if !trimmed.is_empty() {
                            let snippet: String = trimmed.chars().take(220).collect();
                            working_buf.push_str(&format!("- [{}] {}\n", key, snippet));
                        }
                    }
                }
            }
        }
        if !working_buf.is_empty() {
            buf.push_str(&format!("\n## Working context\n{}", working_buf));
        }
    }

    if buf.is_empty() {
        String::new()
    } else {
        format!("\n\n# RECURSIVE CONTEXT (Memento scoped_memory)\n{}", buf)
    }
}

/// Fetch recursive scoped-memory context for (user_id, app_id, session_id) and format it for
/// prompt injection. Returns empty string on any error so the caller can concatenate safely.
pub async fn fetch_recursive_context(user_id: &str, app_id: &str, session_id: &str) -> String {
    if user_id.is_empty() {
        return String::new();
    }
    let mut payload = serde_json::json!({ "user_id": user_id });
    if !app_id.is_empty() {
        payload["app_id"] = serde_json::Value::String(app_id.to_string());
    }
    if !session_id.is_empty() {
        payload["session_id"] = serde_json::Value::String(session_id.to_string());
    }
    let Some(response) = call_memento("recall_recursive_context", payload).await else {
        return String::new();
    };
    let Some(ctx) = response.get("recursive_context") else {
        return String::new();
    };
    format_recursive_context(ctx)
}

/// Fetch all pinned RAG documents for a scope (app_id + expert_id) and format
/// them for injection into the stable prompt prefix. Pinned docs are the "always-on"
/// knowledge base for the agent — editable via /rag/admin without redeploying.
/// Returns empty string if no pinned docs exist or Memento is unavailable.
pub async fn fetch_rag_pinned(app_id: &str, expert_id: &str) -> String {
    if app_id.is_empty() && expert_id.is_empty() {
        return String::new();
    }
    let mut payload = serde_json::json!({});
    if !app_id.is_empty() {
        payload["app_id"] = serde_json::Value::String(app_id.to_string());
    }
    if !expert_id.is_empty() {
        payload["expert_id"] = serde_json::Value::String(expert_id.to_string());
    }
    let Some(resp) = call_memento("rag_pinned", payload).await else {
        return String::new();
    };
    let Some(docs) = resp.get("documents").and_then(|v| v.as_array()) else {
        return String::new();
    };
    if docs.is_empty() {
        return String::new();
    }
    let mut sections = Vec::new();
    let mut total_chars = 0usize;
    const MAX_PINNED_CHARS: usize = 24_000;
    for doc in docs {
        let title = doc.get("title").and_then(|v| v.as_str()).unwrap_or("Document");
        let text  = doc.get("full_text").and_then(|v| v.as_str()).unwrap_or("");
        if text.is_empty() { continue; }
        let remaining = MAX_PINNED_CHARS.saturating_sub(total_chars);
        if remaining == 0 { break; }
        let clipped: String = text.chars().take(remaining).collect();
        total_chars += clipped.chars().count();
        sections.push(format!("## {}\n{}", title, clipped));
    }
    if sections.is_empty() { return String::new(); }
    format!("\n\n# KNOWLEDGE BASE (RAG pinned documents)\n{}", sections.join("\n\n---\n\n"))
}

/// Embed the user query and return the most relevant RAG chunk snippets for the scope.
/// Injected into the dynamic_suffix (per-turn, like semantic_ctx). Returns empty
/// string if the embed model is unavailable or no chunks score above Memento's threshold.
pub async fn fetch_rag_context(app_id: &str, expert_id: &str, query: &str) -> String {
    if query.trim().is_empty() || (app_id.is_empty() && expert_id.is_empty()) {
        return String::new();
    }
    let q = query.to_string();
    let embedding = match tokio::task::spawn_blocking(move || embed_text_local(&q)).await {
        Ok(Some(e)) => e,
        _ => return String::new(),
    };
    let mut payload = serde_json::json!({ "query_embedding": embedding, "k": 5 });
    if !app_id.is_empty() {
        payload["app_id"] = serde_json::Value::String(app_id.to_string());
    }
    if !expert_id.is_empty() {
        payload["expert_id"] = serde_json::Value::String(expert_id.to_string());
    }
    let Some(resp) = call_memento("rag_search", payload).await else {
        return String::new();
    };
    let Some(results) = resp.get("results").and_then(|v| v.as_array()) else {
        return String::new();
    };
    if results.is_empty() { return String::new(); }
    let mut section = String::new();
    for hit in results.iter().take(5) {
        let title   = hit.get("title").and_then(|v| v.as_str()).unwrap_or("");
        let content = hit.get("content").and_then(|v| v.as_str()).unwrap_or("");
        if content.is_empty() { continue; }
        let snippet: String = content.chars().take(400).collect();
        section.push_str(&format!("- [{}] {}\n", title, snippet));
    }
    if section.is_empty() { return String::new(); }
    format!("\n\n# RAG CONTEXT (relevant document chunks)\n{}", section)
}

/// Silent knowledge-graph recall: matches tokens from `user_message` against
/// entity names in Memento's `app_id`-scoped KG, then expands 1 hop from the
/// matched entities. Errors/no-match -> "" (never breaks the turn). Mirrors
/// `fetch_rag_context`'s shape; injected into `dynamic_suffix`, never `stable_prefix`.
pub async fn fetch_kg_context(app_id: &str, user_message: &str) -> String {
    if app_id.is_empty() || user_message.trim().is_empty() {
        return String::new();
    }
    let graph_payload = serde_json::json!({ "app_id": app_id, "max_entities": 100 });
    let Some(graph_resp) = call_memento("kg_graph", graph_payload).await else {
        return String::new();
    };
    let Some(entities) = graph_resp.get("entities").and_then(|v| v.as_array()) else {
        return String::new();
    };
    if entities.is_empty() { return String::new(); }

    let tokens: Vec<String> = user_message
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() > 2)
        .map(|t| t.to_lowercase())
        .collect();
    if tokens.is_empty() { return String::new(); }

    let mut seeds: Vec<String> = Vec::new();
    for entity in entities {
        let (Some(id), Some(name)) = (
            entity.get("entity_id").and_then(|v| v.as_str()),
            entity.get("name").and_then(|v| v.as_str()),
        ) else { continue };
        let name_lower = name.to_lowercase();
        if tokens.iter().any(|t| name_lower.contains(t.as_str())) {
            seeds.push(id.to_string());
            if seeds.len() >= 5 { break; }
        }
    }
    if seeds.is_empty() { return String::new(); }

    let neighbors_payload = serde_json::json!({ "seeds": seeds, "hops": 1, "max_entities": 60 });
    let Some(neighbors_resp) = call_memento("kg_neighbors", neighbors_payload).await else {
        return String::new();
    };
    let Some(relations) = neighbors_resp.get("relations").and_then(|v| v.as_array()) else {
        return String::new();
    };
    if relations.is_empty() { return String::new(); }

    let names: std::collections::HashMap<String, String> = entities
        .iter()
        .filter_map(|e| {
            let id = e.get("entity_id").and_then(|v| v.as_str())?;
            let name = e.get("name").and_then(|v| v.as_str())?;
            Some((id.to_string(), name.to_string()))
        })
        .collect();

    let mut section = String::new();
    for rel in relations.iter().take(10) {
        let src = rel.get("src_id").and_then(|v| v.as_str()).unwrap_or("");
        let dst = rel.get("dst_id").and_then(|v| v.as_str()).unwrap_or("");
        let rel_type = rel.get("rel_type").and_then(|v| v.as_str()).unwrap_or("relacionado con");
        let evidence = rel.get("evidence").and_then(|v| v.as_str()).unwrap_or("");
        let src_name = names.get(src).map(|s| s.as_str()).unwrap_or(src);
        let dst_name = names.get(dst).map(|s| s.as_str()).unwrap_or(dst);
        if src_name.is_empty() || dst_name.is_empty() { continue; }
        section.push_str(&format!("- {} — {} — {} ({})\n", src_name, rel_type, dst_name, evidence));
    }
    if section.is_empty() { return String::new(); }
    format!("\n\n# CONTEXTO DE GRAFO ({})\n{}", app_id, section)
}

/// Persist a single conversation turn as a scoped_memory event. Fire-and-forget — errors only
/// log via tracing. Memento's auto_derive=true triggers session/room/project summary derivation
/// once enough events accumulate, which subsequent calls to `fetch_recursive_context` surface.
pub async fn save_chat_turn_event(
    user_id: String,
    app_id: String,
    session_id: String,
    role: String,
    content: String,
) {
    if user_id.is_empty() || content.trim().is_empty() {
        return;
    }
    let mut payload = serde_json::json!({
        "user_id": user_id,
        "tenant_id": "default",
        "app_id": app_id,
        "session_id": session_id,
        "scope": "personal",
        "source": "hera_chat",
        "memory_type": "event",
        "content": content,
        "tags": ["chat", role],
        "auto_derive": true,
    });
    // Attach a semantic embedding so this turn is retrievable by meaning, not
    // just keywords. No-op on CPU-only builds (embed_text_local returns None).
    if let Some(embedding) = embed_text_local(&content) {
        payload["embedding"] = serde_json::json!(embedding);
    }
    if let Some(resp) = call_memento("save_scoped_memory", payload).await
        && let Some(err) = resp.get("error").and_then(|v| v.as_str())
    {
        tracing::warn!("save_chat_turn_event failed: {}", err);
    }
    // Fire-and-forget KG extraction: assistant turns carry most relational signal.
    // User turns are skipped to avoid noise from questions without factual content.
    //
    // GATE on a non-empty app_id. Without it this self-amplified into a runaway:
    // the extraction's OWN `generate` call (which carries no app_id) produced a
    // JSON turn that handler_generate saved as another assistant turn, which
    // re-spawned extraction — a loop bounded only by 30B throughput (~1 every
    // 3.7 s, saturating the GPU 24/7 with unattributed inference). Internal /
    // anonymous generations (app_id == "") have no place in the per-app KG
    // anyway, so only extract for turns that belong to a real app.
    if role == "assistant" && content.len() > 80 && !app_id.trim().is_empty() {
        let (uid, aid, text) = (user_id.clone(), app_id.clone(), content.clone());
        tokio::spawn(async move {
            extract_and_save_kg_triples(&uid, &aid, &text).await;
        });
    }
}

/// Light-weight KG extraction: sends a small prompt to the local Hera model to
/// extract entity/relation triples, then upserts them into Memento's kg_store.
/// Uses a lightweight context budget and a 500-ms connect timeout so the main
/// chat path is never blocked or slowed.
async fn extract_and_save_kg_triples(user_id: &str, app_id: &str, text: &str) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let snippet = &text[..text.len().min(600)];
    let prompt = format!(
        "Extract named entities and their relations from the following text. \
         Respond ONLY with a JSON object in this exact shape: \
         {{\"triples\":[{{\"src\":{{\"name\":\"...\",\"type\":\"...\"}},\"rel\":\"...\",\"dst\":{{\"name\":\"...\",\"type\":\"...\"}},\"evidence\":\"...\"}}]}}. \
         Use types like: persona, lugar, organizacion, concepto, evento, fecha. \
         If no clear entities/relations, return {{\"triples\":[]}}.\n\n\
         Text: {snippet}"
    );

    let ipc_msg = serde_json::json!({
        "action": "generate",
        "payload": {
            "prompt": prompt,
            "max_tokens": 400,
            "temperature": 0.1,
            "context_budget_mode": "lightweight",
            "response_format": {"type": "json_object"},
            "priority": "background",
        }
    });

    let Ok(Ok(mut stream)) = tokio::time::timeout(
        std::time::Duration::from_millis(500),
        tokio::net::UnixStream::connect("/tmp/hera-core.sock"),
    ).await else { return };

    let _ = stream.write_all(ipc_msg.to_string().as_bytes()).await;
    let _ = stream.shutdown().await;

    let mut raw = Vec::new();
    let Ok(Ok(_)) = tokio::time::timeout(
        std::time::Duration::from_millis(8_000),
        stream.read_to_end(&mut raw),
    ).await else { return };

    let Ok(resp) = serde_json::from_slice::<serde_json::Value>(&raw) else { return };
    let result_str = resp.pointer("/data/result").and_then(|v| v.as_str()).unwrap_or("");
    if result_str.is_empty() { return }

    // Parse the model output — it may be a raw JSON string or already parsed
    let Ok(parsed) = serde_json::from_str::<serde_json::Value>(result_str) else { return };
    let triples = match parsed.get("triples").and_then(|v| v.as_array()) {
        Some(t) if !t.is_empty() => t.to_owned(),
        _ => return,
    };

    let kg_payload = serde_json::json!({
        "app_id": app_id,
        "source_kind": "memory",
        "doc_id": user_id,
        "triples": triples,
    });
    if let Some(resp) = call_memento("kg_upsert_triples", kg_payload).await {
        tracing::debug!(
            "[KG] upserted {} triples for user={} app={}",
            triples.len(),
            user_id,
            app_id,
        );
        if let Some(err) = resp.get("error").and_then(|v| v.as_str()) {
            tracing::warn!("[KG] kg_upsert_triples error: {}", err);
        }
    }
}

/// Persist a history-compression summary to Memento so the next session can
/// seed its context with what was discussed, without re-reading all turns.
pub async fn save_session_summary(user_id: &str, app_id: &str, session_id: &str, summary: &str) {
    if user_id.is_empty() || summary.trim().is_empty() {
        return;
    }
    let payload = serde_json::json!({
        "user_id": user_id,
        "tenant_id": "default",
        "app_id": app_id,
        "session_id": session_id,
        "scope": "personal",
        "source": "hera_compression",
        "memory_type": "session_summary",
        "content": summary,
        "tags": ["session_summary", "compressed_history"],
        "auto_derive": false,
    });
    if let Some(resp) = call_memento("save_scoped_memory", payload).await
        && let Some(err) = resp.get("error").and_then(|v| v.as_str())
    {
        tracing::warn!("save_session_summary failed: {}", err);
    }
}

/// Trigger Memento's compress_session for a given user+session so summaries are
/// derived eagerly instead of waiting for the next auto_derive threshold.
pub async fn memento_compress_session(user_id: &str, app_id: &str, session_id: &str) {
    if user_id.is_empty() || session_id.is_empty() {
        return;
    }
    let payload = serde_json::json!({
        "user_id": user_id,
        "app_id": app_id,
        "session_id": session_id,
    });
    if let Some(resp) = call_memento("compress_session", payload).await
        && let Some(err) = resp.get("error").and_then(|v| v.as_str())
    {
        tracing::warn!("memento_compress_session failed: {}", err);
    }
}

// ─── Recall feedback (Phase 2 of the embedder flywheel) ───────────────────

/// Distinctive tokens of a text: lowercased, alphanumeric-only, length >= 5.
/// We keep length >= 5 so common Spanish/English stopwords ("para", "this",
/// "and") don't pollute the overlap signal.
fn distinctive_tokens(text: &str) -> std::collections::HashSet<String> {
    text.chars()
        .map(|c| if c.is_alphanumeric() { c.to_ascii_lowercase() } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .filter(|tok| tok.chars().count() >= 5)
        .map(|tok| tok.to_string())
        .collect()
}

/// Decide which recalled ids were cited by the model's response. Heuristic:
/// for each recalled entry, count how many of its distinctive tokens (>=5
/// chars, alphanumeric) appear in the response. An entry counts as cited if
/// at least 2 distinctive tokens overlap (or 1 if the entry only has 1 such
/// token).
pub fn cited_ids_from_response(attribution: &RecallAttribution, response: &str) -> Vec<i64> {
    let response_tokens = distinctive_tokens(response);
    if response_tokens.is_empty() {
        return Vec::new();
    }
    let mut cited = Vec::new();
    for entry in &attribution.entries {
        let entry_tokens = distinctive_tokens(&entry.content);
        if entry_tokens.is_empty() {
            continue;
        }
        let overlap = entry_tokens.intersection(&response_tokens).count();
        let needed = if entry_tokens.len() <= 1 { 1 } else { 2 };
        if overlap >= needed {
            cited.push(entry.id);
        }
    }
    cited
}

/// Fire-and-forget: report which recalled ids were cited by the assistant's
/// response. Joined on `request_id` with Memento's `recall_log`, this becomes
/// (query, positives, negatives) training data for embedder reranker fine-tune.
/// Silent on failure — telemetry must never break the chat path.
pub async fn report_recall_feedback(
    attribution: Option<&RecallAttribution>,
    response_text: &str,
) {
    let Some(attribution) = attribution else {
        return;
    };
    if attribution.request_id.is_empty() || attribution.entries.is_empty() {
        return;
    }
    let cited = cited_ids_from_response(attribution, response_text);
    let feedback_kind = if cited.is_empty() { "ignored" } else { "cited" };
    let payload = serde_json::json!({
        "request_id": attribution.request_id,
        "cited_ids": cited,
        "feedback_kind": feedback_kind,
    });
    let _ = call_memento("recall_feedback", payload).await;
}

/// Which node this Hera process runs on (genesis / anchor / ...), for telemetry.
/// Node identity for telemetry: `HERA_NODE_ALIAS` overrides, else `HOSTNAME`,
/// else the kernel hostname (`/etc/hostname`), else "unknown".
///
/// HOSTNAME is a shell var and is often NOT exported into the pm2 process env,
/// so it can't be the only fallback — that left telemetry rows with an empty
/// `node`. The `/etc/hostname` read lets every node self-identify with no config.
pub fn hera_node() -> String {
    if let Ok(alias) = std::env::var("HERA_NODE_ALIAS") {
        if !alias.trim().is_empty() {
            return alias.trim().to_string();
        }
    }
    if let Ok(host) = std::env::var("HOSTNAME") {
        if !host.trim().is_empty() {
            return host.trim().to_string();
        }
    }
    if let Ok(host) = std::fs::read_to_string("/etc/hostname") {
        let host = host.trim();
        if !host.is_empty() {
            return host.to_string();
        }
    }
    "unknown".to_string()
}

/// Mint a trace id for requests that arrive without one (MCP / direct callers).
/// Every request should be correlatable across usage + tool-call telemetry and
/// the `/ops/hera/trace/<id>` timeline — an empty trace_id makes a request
/// invisible to that join. Format: `hera-<node>-<nanos>-<counter>` (no external
/// dep; the atomic counter disambiguates ids minted within the same nanosecond).
pub fn mint_trace_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("hera-{}-{:x}-{:x}", hera_node(), nanos, seq)
}

/// Char-safe truncation for telemetry previews of tool args/results. Larger cap
/// than the 160-char prompt preview (tool payloads are structured), but still
/// bounded so a giant SQL result or file blob can't bloat the row. Not a
/// redaction layer — secret-bearing args are only length-limited here.
pub fn telemetry_preview(text: &str, max_chars: usize) -> String {
    let trimmed = text.trim();
    if trimmed.chars().count() > max_chars {
        let mut s: String = trimmed.chars().take(max_chars).collect();
        s.push('…');
        s
    } else {
        trimmed.to_string()
    }
}

/// Fire-and-forget: records a usage event in Memento after each generate call.
/// Fails silently — never blocks or panics the generate path.
pub fn spawn_log_usage(
    app_id: String,
    user_id: String,
    session_id: String,
    route_profile: String,
    model: String,
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
    is_cloud: bool,
    latency_ms: u64,
    trace_id: String,
) {
    tokio::spawn(async move {
        let payload = serde_json::json!({
            "app_id": app_id,
            "user_id": user_id,
            "session_id": session_id,
            "route_profile": route_profile,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "is_cloud": is_cloud,
            "latency_ms": latency_ms,
            "trace_id": trace_id,
            "node": hera_node()
        });
        if call_memento("hera_log_usage", payload).await.is_none() {
            tracing::debug!("[Hera Usage] log_usage failed or timed out (best-effort)");
        }
    });
}

/// Fire-and-forget: records one per-tool-call telemetry row in Memento.
/// This is the durable per-tool grain that was previously discarded — the
/// data (name + args + result + success) already exists in memory at the tool
/// execution site; this persists it keyed by `trace_id`. Best-effort, never
/// blocks the tool path.
#[allow(clippy::too_many_arguments)]
pub fn spawn_log_tool_call(
    trace_id: String,
    session_id: String,
    app_id: String,
    route_profile: String,
    seq: u32,
    tool_name: String,
    args_preview: String,
    result_preview: String,
    duration_ms: u64,
    success: bool,
    error: Option<String>,
) {
    tokio::spawn(async move {
        let payload = serde_json::json!({
            "trace_id": trace_id,
            "session_id": session_id,
            "app_id": app_id,
            "route_profile": route_profile,
            "node": hera_node(),
            "seq": seq,
            "tool_name": tool_name,
            "args_preview": args_preview,
            "result_preview": result_preview,
            "duration_ms": duration_ms,
            "success": success,
            "error": error
        });
        if call_memento("hera_log_tool_call", payload).await.is_none() {
            tracing::debug!("[Hera Tool Telemetry] log_tool_call failed or timed out (best-effort)");
        }
    });
}

#[cfg(test)]
mod tests {
    use super::{canonicalize_user_id, cited_ids_from_response, RecallAttribution, RecalledEntry};

    #[test]
    fn canonicalize_prefers_sender_name() {
        assert_eq!(
            canonicalize_user_id("Paulo Vila", "chat-1", "sess-1"),
            "paulo_vila"
        );
    }

    #[test]
    fn canonicalize_falls_back_to_chat_id_when_sender_blank() {
        assert_eq!(
            canonicalize_user_id("", "telegram_42", "sess-9"),
            "chat:telegram_42"
        );
    }

    #[test]
    fn canonicalize_falls_back_to_anonymous_session() {
        assert_eq!(canonicalize_user_id("", "", "abc-123"), "anonymous:abc-123");
    }

    #[test]
    fn canonicalize_returns_anonymous_when_all_blank() {
        assert_eq!(canonicalize_user_id("", "", ""), "anonymous");
    }

    #[test]
    fn canonicalize_strips_non_alnum_from_sender() {
        // Unicode alphanumerics (í, á, é) are preserved — sender_name keeps natural diacritics.
        // Only spaces and ASCII punctuation collapse to underscores, with edge underscores trimmed.
        assert_eq!(
            canonicalize_user_id("María García-Pérez!", "", ""),
            "maría_garcía_pérez"
        );
    }

    #[test]
    fn cited_ids_matches_only_entries_with_token_overlap() {
        let attribution = RecallAttribution {
            request_id: "req-1".into(),
            entries: vec![
                RecalledEntry {
                    id: 11,
                    content: "Contracts with margin above thirty percent flagged review.".into(),
                },
                RecalledEntry {
                    id: 22,
                    content: "User prefers concise summaries in lowercase letters.".into(),
                },
                RecalledEntry {
                    id: 33,
                    content: "Birthday is March twentieth.".into(),
                },
            ],
        };
        let response = "We should flag contracts with high margin and review them.";
        let cited = cited_ids_from_response(&attribution, response);
        // Entry 11 has "contracts", "margin", "review" overlap (>=2) -> cited.
        // Entry 22 has zero distinctive overlap -> not cited.
        // Entry 33 has zero overlap -> not cited.
        assert_eq!(cited, vec![11]);
    }

    #[test]
    fn cited_ids_empty_when_response_has_no_distinctive_tokens() {
        let attribution = RecallAttribution {
            request_id: "req-x".into(),
            entries: vec![RecalledEntry {
                id: 1,
                content: "important domain fact about pricing".into(),
            }],
        };
        // Response has only short stopwords (none >= 5 chars).
        let cited = cited_ids_from_response(&attribution, "ok and yes");
        assert!(cited.is_empty());
    }
}
