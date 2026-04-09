//! Data tool executors: Memento, Git, API requests
use crate::ai::tool_executor::{ToolCall, ToolResult};
use serde_json::Value;
use std::process::Command;
use tracing::info;

pub(crate) async fn execute_memento_query_json(call: &ToolCall) -> Result<Value, String> {
    let app = call
        .arguments
        .get("app")
        .and_then(|a| a.as_str())
        .unwrap_or("movilo");
    let query = call
        .arguments
        .get("query")
        .and_then(|q| q.as_str())
        .unwrap_or("");

    if query.is_empty() {
        return Err("Missing 'query' argument".to_string());
    }

    info!("🧠 [Memento] Querying app '{}' with: {}", app, query);

    // Connect to Memento via UDS
    match tokio::net::UnixStream::connect("/tmp/memento.sock").await {
        Ok(mut stream) => {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};

            let msg = serde_json::json!({
                "action": "query_app",
                "payload": {
                    "app": app,
                    "query": query,
                    "limit": 20
                }
            });

            if let Err(e) = stream.write_all(msg.to_string().as_bytes()).await {
                return Err(format!("Failed to send to Memento: {}", e));
            }

            // Shutdown write half to signal end-of-request so Memento flushes its response
            let _ = stream.shutdown().await;

            // Use dynamic read_to_end — stock_research rows can exceed 64KB
            let mut raw_bytes = Vec::new();
            match stream.read_to_end(&mut raw_bytes).await {
                Ok(n) if n > 0 => {
                    let response_str = String::from_utf8_lossy(&raw_bytes);
                    match serde_json::from_str::<serde_json::Value>(&response_str) {
                        Ok(res) => {
                            if res.get("status").and_then(|s| s.as_str()) == Some("success") {
                                info!(
                                    "🧠 [Memento] Got {} rows from '{}'",
                                    res.get("count").and_then(|c| c.as_i64()).unwrap_or(0),
                                    app
                                );
                                Ok(res)
                            } else {
                                let error = res
                                    .get("error")
                                    .and_then(|e| e.as_str())
                                    .unwrap_or("Unknown error");
                                Err(format!("Memento error: {}", error))
                            }
                        }
                        Err(e) => Err(format!("Failed to parse Memento response: {}", e)),
                    }
                }
                _ => Err("No response from Memento".to_string()),
            }
        }
        Err(e) => {
            tracing::error!(
                "🧠 [Memento] Failed to connect to /tmp/memento.sock: {:?}",
                e
            );
            Err(format!("Memento is not running. Error: {}", e))
        }
    }
}

pub(crate) async fn execute_memento_query(call: &ToolCall) -> ToolResult {
    match execute_memento_query_json(call).await {
        Ok(res) => {
            let app = call
                .arguments
                .get("app")
                .and_then(|a| a.as_str())
                .unwrap_or("movilo");
            let count = res.get("count").and_then(|c| c.as_i64()).unwrap_or(0);
            let rows = res.get("rows").cloned().unwrap_or(serde_json::json!([]));
            let formatted = serde_json::to_string_pretty(&rows).unwrap_or_default();
            ToolResult {
                name: call.name.clone(),
                success: true,
                output: format!(
                    "Database query returned {} results from '{}':\n{}",
                    count, app, formatted
                ),
            }
        }
        Err(error) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: error,
        },
    }
}
pub(crate) async fn execute_api_request(call: &ToolCall) -> ToolResult {
    let method = call
        .arguments
        .get("method")
        .and_then(|m| m.as_str())
        .unwrap_or("GET");
    let url = call
        .arguments
        .get("url")
        .and_then(|u| u.as_str())
        .unwrap_or("");
    let headers_str = call
        .arguments
        .get("headers")
        .and_then(|h| h.as_str())
        .unwrap_or("{}");
    let body_str = call
        .arguments
        .get("body")
        .and_then(|b| b.as_str())
        .unwrap_or("");

    if url.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Missing URL".into(),
        };
    }

    let client = reqwest::Client::new();
    let mut req = match method.to_uppercase().as_str() {
        "POST" => client.post(url),
        "PUT" => client.put(url),
        "DELETE" => client.delete(url),
        _ => client.get(url),
    };

    if let Ok(headers) = serde_json::from_str::<serde_json::Value>(headers_str)
        && let Some(obj) = headers.as_object() {
            for (k, v) in obj {
                if let Some(s) = v.as_str() {
                    req = req.header(k, s);
                }
            }
        }

    if !body_str.is_empty() {
        req = req.body(body_str.to_string());
    }

    match req.send().await {
        Ok(res) => {
            let status = res.status();
            match res.text().await {
                Ok(text) => ToolResult {
                    name: call.name.clone(),
                    success: status.is_success(),
                    output: format!("Status: {}\nBody: {}", status, text),
                },
                Err(e) => ToolResult {
                    name: call.name.clone(),
                    success: false,
                    output: format!("Failed to read response body: {}", e),
                },
            }
        }
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Request failed: {}", e),
        },
    }
}

pub(crate) async fn execute_git_manager(call: &ToolCall) -> ToolResult {
    let command = call
        .arguments
        .get("command")
        .and_then(|c| c.as_str())
        .unwrap_or("");
    let repo_path = call
        .arguments
        .get("repo_path")
        .and_then(|p| p.as_str())
        .unwrap_or("");

    if repo_path.is_empty() || command.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Missing command or repo_path".into(),
        };
    }

    let args: Vec<&str> = command.split_whitespace().collect();
    match std::process::Command::new("git")
        .current_dir(repo_path)
        .args(&args)
        .output()
    {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let success = output.status.success();
            let res = if success {
                format!("{}", stdout)
            } else {
                format!("Error: {}\n{}", stderr, stdout)
            };
            ToolResult {
                name: call.name.clone(),
                success,
                output: res,
            }
        }
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Failed to run git: {}", e),
        },
    }
}

pub(crate) async fn execute_memento_vector_search(call: &ToolCall) -> ToolResult {
    let query = call
        .arguments
        .get("query")
        .and_then(|q| q.as_str())
        .unwrap_or("");
    let limit = call
        .arguments
        .get("limit")
        .and_then(|l| l.as_u64())
        .unwrap_or(3);

    // Like memento_query, but action "vector_search"
    match tokio::net::UnixStream::connect("/tmp/memento.sock").await {
        Ok(mut stream) => {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let msg = serde_json::json!({
                "action": "vector_search",
                "payload": {
                    "query": query,
                    "limit": limit
                }
            });
            if stream.write_all(msg.to_string().as_bytes()).await.is_err() {
                return ToolResult {
                    name: call.name.clone(),
                    success: false,
                    output: "IPC Write Failed".into(),
                };
            }
            let mut buffer = vec![0u8; 65536];
            match stream.read(&mut buffer).await {
                Ok(n) if n > 0 => {
                    let response_str = String::from_utf8_lossy(&buffer[..n]);
                    ToolResult {
                        name: call.name.clone(),
                        success: true,
                        output: response_str.to_string(),
                    }
                }
                _ => ToolResult {
                    name: call.name.clone(),
                    success: false,
                    output: "No response".into(),
                },
            }
        }
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Memento socket error: {}", e),
        },
    }
}

