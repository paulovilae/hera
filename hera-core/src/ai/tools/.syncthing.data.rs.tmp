//! Data tool executors: Memento, Git, API requests
use crate::ai::tool_executor::{ToolCall, ToolResult};
use serde_json::Value;
use std::path::{Path, PathBuf};
use tracing::info;

const OS_ROOT: &str = "/home/paulo/Programs/apps/OS";

fn memento_request(action: &str, payload: Value) -> Value {
    serde_json::json!({
        "action": action,
        "payload": payload,
        "client": {
            "app": "hera",
            "token": std::env::var("MEMENTO_CLIENT_TOKEN").ok()
        }
    })
}

fn resolve_guarded_path(path: &str) -> Result<PathBuf, String> {
    let raw = Path::new(path);
    let candidate = if raw.exists() {
        std::fs::canonicalize(raw).map_err(|e| format!("Failed to resolve path: {}", e))?
    } else {
        let parent = raw
            .parent()
            .ok_or_else(|| "Path must include a parent directory".to_string())?;
        let resolved_parent = std::fs::canonicalize(parent)
            .map_err(|e| format!("Failed to resolve parent directory: {}", e))?;
        resolved_parent.join(
            raw.file_name()
                .ok_or_else(|| "Path must include a file or directory name".to_string())?,
        )
    };

    if candidate.starts_with(OS_ROOT) {
        Ok(candidate)
    } else {
        Err(format!(
            "Path '{}' is outside the allowed Hera workspace root '{}'.",
            path, OS_ROOT
        ))
    }
}

fn is_forbidden_host(host: &str) -> bool {
    let lowered = host.trim().to_lowercase();
    if lowered.is_empty() {
        return true;
    }

    if matches!(
        lowered.as_str(),
        "localhost" | "metadata.google.internal" | "metadata" | "host.docker.internal"
    ) {
        return true;
    }

    if lowered == "169.254.169.254"
        || lowered.starts_with("127.")
        || lowered.starts_with("10.")
        || lowered.starts_with("192.168.")
        || lowered.starts_with("169.254.")
        || lowered == "::1"
    {
        return true;
    }

    if let Some(rest) = lowered.strip_prefix("172.") {
        if let Some(octet) = rest.split('.').next()
            && let Ok(value) = octet.parse::<u8>()
        {
            return (16..=31).contains(&value);
        }
    }

    lowered.starts_with("fc") || lowered.starts_with("fd")
}

/// True si la IP es interna/no-ruteable (loopback, privada, link-local 169.254,
/// CGNAT tailscale 100.64/10, ULA, multicast, etc.). Se valida la IP RESUELTA,
/// no solo el hostname literal — así un dominio del atacante que resuelve a
/// 169.254.169.254 (metadata de GCP) o a una IP privada queda bloqueado (SSRF
/// por DNS rebinding).
pub(crate) fn is_blocked_ip(ip: &std::net::IpAddr) -> bool {
    match ip {
        std::net::IpAddr::V4(v4) => {
            let o = v4.octets();
            v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_broadcast()
                || v4.is_multicast()
                || v4.is_unspecified()
                || o[0] == 0
                // tailscale CGNAT 100.64.0.0/10 (malla interna)
                || (o[0] == 100 && (64..=127).contains(&o[1]))
        }
        std::net::IpAddr::V6(v6) => {
            v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_multicast()
                || (v6.segments()[0] & 0xfe00) == 0xfc00 // ULA fc00::/7
                || (v6.segments()[0] & 0xffc0) == 0xfe80 // link-local fe80::/10
                || v6
                    .to_ipv4_mapped()
                    .map(|v4| is_blocked_ip(&std::net::IpAddr::V4(v4)))
                    .unwrap_or(false)
        }
    }
}

pub(crate) async fn validate_outbound_url(url: &str) -> Result<reqwest::Url, String> {
    let parsed = reqwest::Url::parse(url).map_err(|e| format!("Invalid URL: {}", e))?;
    match parsed.scheme() {
        "http" | "https" => {}
        other => {
            return Err(format!(
                "Unsupported URL scheme '{}'. Only http and https are allowed.",
                other
            ));
        }
    }

    let host = parsed
        .host_str()
        .ok_or_else(|| "URL must include a hostname".to_string())?;
    if is_forbidden_host(host) {
        return Err(format!(
            "Outbound requests to private or loopback hosts are blocked: '{}'.",
            host
        ));
    }

    // Resolver el host y validar TODAS las IPs: cierra el bypass de DNS rebinding
    // (un dominio publico que resuelve a una IP interna/metadata).
    let port = parsed
        .port_or_known_default()
        .unwrap_or(if parsed.scheme() == "https" { 443 } else { 80 });
    let addrs = tokio::net::lookup_host((host, port))
        .await
        .map_err(|e| format!("DNS resolution failed for '{}': {}", host, e))?;
    let mut resolved_any = false;
    for addr in addrs {
        resolved_any = true;
        if is_blocked_ip(&addr.ip()) {
            return Err(format!(
                "Blocked: '{}' resolves to a private/internal address ({}).",
                host,
                addr.ip()
            ));
        }
    }
    if !resolved_any {
        return Err(format!("'{}' did not resolve to any IP address.", host));
    }

    Ok(parsed)
}

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
    let limit = call
        .arguments
        .get("limit")
        .and_then(|l| l.as_u64())
        .unwrap_or(500);

    if query.is_empty() {
        return Err("Missing 'query' argument".to_string());
    }

    info!("🧠 [Memento] Querying app '{}' with: {}", app, query);

    // Connect to Memento via UDS
    match tokio::net::UnixStream::connect("/tmp/memento.sock").await {
        Ok(mut stream) => {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};

            let msg = memento_request(
                "query_app",
                serde_json::json!({
                    "app": app,
                    "query": query,
                    "limit": limit
                }),
            );

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

    let method_upper = method.to_uppercase();
    if !matches!(
        method_upper.as_str(),
        "GET" | "POST" | "PUT" | "DELETE" | "PATCH"
    ) {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Unsupported HTTP method '{}'.", method),
        };
    }

    let parsed_url = match validate_outbound_url(url).await {
        Ok(parsed) => parsed,
        Err(error) => {
            return ToolResult {
                name: call.name.clone(),
                success: false,
                output: error,
            };
        }
    };

    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(20))
        .build()
    {
        Ok(client) => client,
        Err(error) => {
            return ToolResult {
                name: call.name.clone(),
                success: false,
                output: format!("Failed to build HTTP client: {}", error),
            };
        }
    };
    let mut req = match method_upper.as_str() {
        "POST" => client.post(parsed_url.clone()),
        "PUT" => client.put(parsed_url.clone()),
        "DELETE" => client.delete(parsed_url.clone()),
        "PATCH" => client.patch(parsed_url.clone()),
        _ => client.get(parsed_url.clone()),
    };

    if let Ok(headers) = serde_json::from_str::<serde_json::Value>(headers_str)
        && let Some(obj) = headers.as_object()
    {
        for (k, v) in obj {
            if let Some(s) = v.as_str() {
                let lower = k.to_ascii_lowercase();
                if matches!(lower.as_str(), "authorization" | "cookie" | "x-api-key") {
                    return ToolResult {
                        name: call.name.clone(),
                        success: false,
                        output: format!(
                            "Blocked outbound sensitive header '{}'. Use a dedicated adapter instead.",
                            k
                        ),
                    };
                }
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

    let resolved_repo = match resolve_guarded_path(repo_path) {
        Ok(path) => path,
        Err(error) => {
            return ToolResult {
                name: call.name.clone(),
                success: false,
                output: error,
            };
        }
    };

    let args: Vec<&str> = command.split_whitespace().collect();
    let Some(subcommand) = args.first().copied() else {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Git command cannot be empty.".into(),
        };
    };
    if !matches!(
        subcommand,
        "status" | "diff" | "log" | "show" | "branch" | "rev-parse"
    ) {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!(
                "Git subcommand '{}' is blocked. Hera only allows read-only git operations.",
                subcommand
            ),
        };
    }
    match std::process::Command::new("git")
        .current_dir(&resolved_repo)
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
            let msg = memento_request(
                "vector_search",
                serde_json::json!({
                    "query": query,
                    "limit": limit
                }),
            );
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
