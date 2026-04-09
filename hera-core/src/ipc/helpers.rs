//! IPC helper functions — Memento integration, model origin inference, token estimation.

use tokio::io::{AsyncReadExt, AsyncWriteExt};

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
        let msg = serde_json::json!({
            "action": "query_app",
            "payload": { "app": app_name, "query": "semantic_context" }
        });
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
        let msg = serde_json::json!({
            "action": "describe_app",
            "payload": { "app": app_slug }
        });
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
        let msg = serde_json::json!({
            "action": "describe_all_apps",
            "payload": {}
        });
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
