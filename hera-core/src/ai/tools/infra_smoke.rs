use crate::ai::tool_executor::{
    canonical_app_search_terms, canonicalize_app_slug, load_canonical_app_registry,
    pm2_process_name_for_slug, CanonicalAppEntry, ToolCall, ToolResult,
};
use crate::ai::tools::infra_health::execute_diagnose_services;
use serde_json::Value;
use std::process::Command;
use tracing::info;

pub(crate) async fn execute_read_os_logs(call: &ToolCall) -> ToolResult {
    let service = call
        .arguments
        .get("service")
        .and_then(|s| s.as_str())
        .unwrap_or("");
    let level = call
        .arguments
        .get("level")
        .and_then(|l| l.as_str())
        .unwrap_or("");
    let search = call
        .arguments
        .get("search")
        .and_then(|s| s.as_str())
        .unwrap_or("");
    let lines = call
        .arguments
        .get("lines")
        .and_then(|l| l.as_i64())
        .unwrap_or(50)
        .clamp(1, 200) as usize;
    let service_terms = if service.is_empty() {
        Vec::new()
    } else {
        canonical_app_search_terms(service)
    };

    let log_path = "/home/paulo/Programs/apps/OS/Apps/OS-v3/storage/logs/runtime.jsonl";

    match std::fs::read_to_string(log_path) {
        Ok(content) => {
            let mut matched_logs = Vec::new();

            for line in content.lines().rev() {
                if line.trim().is_empty() {
                    continue;
                }

                let lower_line = line.to_lowercase();
                if !service_terms.is_empty() {
                    let service_match = service_terms.iter().any(|term| {
                        lower_line.contains(&format!("\"service\":\"{}\"", term))
                            || lower_line.contains(&format!("\"app\":\"{}\"", term))
                    });
                    if !service_match {
                        continue;
                    }
                }
                if !level.is_empty()
                    && !lower_line.contains(&format!("\"level\":\"{}\"", level.to_lowercase()))
                {
                    continue;
                }
                if !search.is_empty() && !lower_line.contains(&search.to_lowercase()) {
                    continue;
                }

                matched_logs.push(line.to_string());
                if matched_logs.len() >= lines {
                    break;
                }
            }

            matched_logs.reverse();
            let result_str = matched_logs.join("\n");

            ToolResult {
                name: call.name.clone(),
                success: true,
                output: format!("Found {} logs:\n{}", matched_logs.len(), result_str),
            }
        }
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Failed to read os logs from {}: {}", log_path, e),
        },
    }
}

pub(crate) async fn execute_test_apps_json(call: &ToolCall) -> Result<Value, String> {
    let apps = call
        .arguments
        .get("apps")
        .and_then(|value| value.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str())
                .collect::<Vec<_>>()
                .join(",")
        })
        .unwrap_or_default();
    let checks = call
        .arguments
        .get("checks")
        .and_then(|value| value.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str())
                .collect::<Vec<_>>()
                .join(",")
        })
        .unwrap_or_else(|| "check,test".to_string());
    let include_reference = call
        .arguments
        .get("include_reference")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    let fail_fast = call
        .arguments
        .get("fail_fast")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    let timeout_seconds = call
        .arguments
        .get("timeout_seconds")
        .and_then(|value| value.as_i64())
        .unwrap_or(600)
        .clamp(10, 3600);

    let script_path = "/home/paulo/Programs/apps/OS/Tools/global/infra/scripts/test_apps.py";
    let mut command = tokio::process::Command::new("python3");
    command.arg(script_path);
    if !apps.is_empty() {
        command.args(["--apps", &apps]);
    }
    if !checks.is_empty() {
        command.args(["--checks", &checks]);
    }
    if include_reference {
        command.arg("--include-reference");
    }
    if fail_fast {
        command.arg("--fail-fast");
    }
    command.args(["--timeout-seconds", &timeout_seconds.to_string()]);

    let output = command
        .output()
        .await
        .map_err(|error| format!("Failed to execute test_apps script: {}", error))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let payload = if stdout.trim().is_empty() {
        return Err(format!(
            "test_apps returned no JSON output. stderr: {}",
            stderr.trim()
        ));
    } else {
        stdout
    };

    let parsed: Value = serde_json::from_str(&payload).map_err(|error| {
        format!(
            "Failed to parse test_apps output: {}. stderr: {}",
            error,
            stderr.trim()
        )
    })?;

    Ok(parsed)
}

pub(crate) async fn execute_smoke_apps_json(call: &ToolCall) -> Result<Value, String> {
    let apps = call
        .arguments
        .get("apps")
        .and_then(|value| value.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str())
                .collect::<Vec<_>>()
                .join(",")
        })
        .unwrap_or_default();
    let suite = call
        .arguments
        .get("suite")
        .and_then(|value| value.as_str())
        .unwrap_or("smoke");
    let fail_fast = call
        .arguments
        .get("fail_fast")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    let timeout_seconds = call
        .arguments
        .get("timeout_seconds")
        .and_then(|value| value.as_i64())
        .unwrap_or(10)
        .clamp(1, 120);

    let script_path = "/home/paulo/Programs/apps/OS/Tools/global/infra/scripts/smoke_apps.py";
    let mut command = tokio::process::Command::new("python3");
    command.arg(script_path);
    if !apps.is_empty() {
        command.args(["--apps", &apps]);
    }
    command.args(["--suite", suite]);
    if fail_fast {
        command.arg("--fail-fast");
    }
    command.args(["--timeout-seconds", &timeout_seconds.to_string()]);

    let output = command
        .output()
        .await
        .map_err(|error| format!("Failed to execute smoke_apps script: {}", error))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if stdout.trim().is_empty() {
        return Err(format!(
            "smoke_apps returned no JSON output. stderr: {}",
            stderr.trim()
        ));
    }

    serde_json::from_str(&stdout).map_err(|error| {
        format!(
            "Failed to parse smoke_apps output: {}. stderr: {}",
            error,
            stderr.trim()
        )
    })
}

pub(crate) async fn execute_smoke_apps(call: &ToolCall) -> ToolResult {
    match execute_smoke_apps_json(call).await {
        Ok(result) => {
            let summary = result
                .get("summary")
                .cloned()
                .unwrap_or_else(|| serde_json::json!({}));
            let mut lines = vec![
                "App smoke report".to_string(),
                format!(
                    "apps: {}/{} passed",
                    summary
                        .get("apps_passed")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0),
                    summary
                        .get("apps_total")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0)
                ),
                format!(
                    "checks: {}/{} passed",
                    summary
                        .get("checks_passed")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0),
                    summary
                        .get("checks_total")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0)
                ),
            ];

            if let Some(apps) = result.get("apps").and_then(|value| value.as_array()) {
                for app in apps {
                    let slug = app
                        .get("slug")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown");
                    let ok = app.get("ok").and_then(|v| v.as_bool()).unwrap_or(false);
                    lines.push(format!("- {}: {}", slug, if ok { "ok" } else { "failed" }));
                    if let Some(results) = app.get("results").and_then(|v| v.as_array()) {
                        for item in results {
                            let path = item.get("path").and_then(|v| v.as_str()).unwrap_or("?");
                            let status =
                                if item.get("ok").and_then(|v| v.as_bool()).unwrap_or(false) {
                                    "ok"
                                } else {
                                    "failed"
                                };
                            let code = item
                                .get("status_code")
                                .and_then(|v| v.as_i64())
                                .map(|v| v.to_string())
                                .unwrap_or_else(|| "n/a".to_string());
                            lines.push(format!("  {}: {} ({})", path, status, code));
                        }
                    }
                }
            }

            ToolResult {
                name: call.name.clone(),
                success: result.get("ok").and_then(|v| v.as_bool()).unwrap_or(false),
                output: lines.join("\n"),
            }
        }
        Err(error) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: error,
        },
    }
}

pub(crate) async fn execute_review_all_apps_status_json(call: &ToolCall) -> Result<Value, String> {
    let timeout_seconds = call
        .arguments
        .get("timeout_seconds")
        .and_then(|value| value.as_i64())
        .unwrap_or(10)
        .clamp(1, 120);

    let registry = load_canonical_app_registry();
    if registry.is_empty() {
        return Err("Canonical app registry is empty.".to_string());
    }

    let mut pm2_status_by_name = std::collections::HashMap::new();
    let pm2_output = tokio::process::Command::new("pm2")
        .arg("jlist")
        .output()
        .await
        .map_err(|error| format!("Failed to read PM2 state: {}", error))?;
    if pm2_output.status.success() {
        let parsed: Vec<Value> = serde_json::from_slice(&pm2_output.stdout)
            .map_err(|error| format!("Failed to parse PM2 jlist output: {}", error))?;
        for proc in parsed {
            let name = proc
                .get("name")
                .and_then(|value| value.as_str())
                .unwrap_or("")
                .to_string();
            let status = proc
                .get("pm2_env")
                .and_then(|value| value.get("status"))
                .and_then(|value| value.as_str())
                .unwrap_or("unknown")
                .to_string();
            pm2_status_by_name.insert(name, status);
        }
    }

    let smoke = execute_smoke_apps_json(&ToolCall {
        name: "smoke_apps".to_string(),
        arguments: serde_json::json!({
            "suite": "smoke",
            "timeout_seconds": timeout_seconds,
            "fail_fast": false
        }),
    })
    .await?;

    let mut smoke_by_slug = std::collections::HashMap::new();
    if let Some(apps) = smoke.get("apps").and_then(|value| value.as_array()) {
        for app in apps {
            if let Some(slug) = app.get("slug").and_then(|value| value.as_str()) {
                smoke_by_slug.insert(slug.to_string(), app.clone());
            }
        }
    }

    let mut healthy = Vec::new();
    let mut degraded = Vec::new();
    let mut down = Vec::new();
    let mut app_rows = Vec::new();

    for entry in registry {
        let slug = entry.slug;
        let pm2_name = pm2_process_name_for_slug(&slug).to_string();
        let pm2_status = pm2_status_by_name
            .get(&pm2_name)
            .cloned()
            .unwrap_or_else(|| "missing".to_string());
        let smoke_result = smoke_by_slug.get(&slug).cloned();
        let smoke_ok = smoke_result
            .as_ref()
            .and_then(|value| value.get("ok"))
            .and_then(|value| value.as_bool())
            .unwrap_or(false);

        let summary_status = if pm2_status == "online" && smoke_ok {
            "healthy"
        } else if pm2_status == "online" {
            "degraded"
        } else {
            "down"
        };

        let note = match summary_status {
            "healthy" => "online + smoke ok".to_string(),
            "degraded" => "online but smoke failed".to_string(),
            _ => format!("pm2 status={}", pm2_status),
        };

        let row = serde_json::json!({
            "slug": slug,
            "pm2_name": pm2_name,
            "pm2_status": pm2_status,
            "smoke_ok": smoke_ok,
            "status": summary_status,
            "note": note,
            "smoke": smoke_result
        });

        match summary_status {
            "healthy" => healthy.push(row.clone()),
            "degraded" => degraded.push(row.clone()),
            _ => down.push(row.clone()),
        }
        app_rows.push(row);
    }

    Ok(serde_json::json!({
        "ok": down.is_empty() && degraded.is_empty(),
        "apps": app_rows,
        "healthy": healthy,
        "degraded": degraded,
        "down": down,
        "summary": {
            "apps_total": app_rows.len(),
            "healthy": app_rows.iter().filter(|row| row.get("status").and_then(|v| v.as_str()) == Some("healthy")).count(),
            "degraded": app_rows.iter().filter(|row| row.get("status").and_then(|v| v.as_str()) == Some("degraded")).count(),
            "down": app_rows.iter().filter(|row| row.get("status").and_then(|v| v.as_str()) == Some("down")).count()
        }
    }))
}

pub(crate) async fn execute_review_all_apps_status(call: &ToolCall) -> ToolResult {
    match execute_review_all_apps_status_json(call).await {
        Ok(result) => {
            let summary = result
                .get("summary")
                .cloned()
                .unwrap_or_else(|| serde_json::json!({}));
            let mut lines = vec![
                "Canonical app status".to_string(),
                format!(
                    "healthy: {}",
                    summary
                        .get("healthy")
                        .and_then(|value| value.as_u64())
                        .unwrap_or(0)
                ),
                format!(
                    "degraded: {}",
                    summary
                        .get("degraded")
                        .and_then(|value| value.as_u64())
                        .unwrap_or(0)
                ),
                format!(
                    "down: {}",
                    summary
                        .get("down")
                        .and_then(|value| value.as_u64())
                        .unwrap_or(0)
                ),
            ];

            for section in ["healthy", "degraded", "down"] {
                if let Some(items) = result.get(section).and_then(|value| value.as_array()) {
                    if items.is_empty() {
                        continue;
                    }
                    lines.push(format!("{}:", section));
                    for item in items {
                        let slug = item
                            .get("slug")
                            .and_then(|value| value.as_str())
                            .unwrap_or("unknown");
                        let note = item
                            .get("note")
                            .and_then(|value| value.as_str())
                            .unwrap_or("");
                        lines.push(format!("- {}: {}", slug, note));
                    }
                }
            }

            ToolResult {
                name: call.name.clone(),
                success: result
                    .get("ok")
                    .and_then(|value| value.as_bool())
                    .unwrap_or(false),
                output: lines.join("\n"),
            }
        }
        Err(error) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: error,
        },
    }
}

pub(crate) async fn execute_verify_canonical_stack_json(call: &ToolCall) -> Result<Value, String> {
    let checks = call
        .arguments
        .get("checks")
        .and_then(|value| value.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str())
                .collect::<Vec<_>>()
                .join(",")
        })
        .unwrap_or_else(|| "check".to_string());
    let timeout_seconds = call
        .arguments
        .get("timeout_seconds")
        .and_then(|value| value.as_i64())
        .unwrap_or(60)
        .clamp(10, 3600);

    let script_path =
        "/home/paulo/Programs/apps/OS/Tools/global/infra/scripts/verify_canonical_stack.py";
    let output = tokio::process::Command::new("python3")
        .arg(script_path)
        .args(["--checks", &checks])
        .args(["--timeout-seconds", &timeout_seconds.to_string()])
        .output()
        .await
        .map_err(|error| format!("Failed to execute verify_canonical_stack script: {}", error))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if stdout.trim().is_empty() {
        return Err(format!(
            "verify_canonical_stack returned no JSON output. stderr: {}",
            stderr.trim()
        ));
    }

    serde_json::from_str(&stdout).map_err(|error| {
        format!(
            "Failed to parse verify_canonical_stack output: {}. stderr: {}",
            error,
            stderr.trim()
        )
    })
}

pub(crate) async fn execute_verify_canonical_stack(call: &ToolCall) -> ToolResult {
    match execute_verify_canonical_stack_json(call).await {
        Ok(result) => {
            let summary = result
                .get("summary")
                .cloned()
                .unwrap_or_else(|| serde_json::json!({}));
            let lines = ["Canonical stack verification".to_string(),
                format!(
                    "compile: {}",
                    if summary
                        .get("compile_ok")
                        .and_then(|value| value.as_bool())
                        .unwrap_or(false)
                    {
                        "ok"
                    } else {
                        "failed"
                    }
                ),
                format!(
                    "smoke: {}",
                    if summary
                        .get("smoke_ok")
                        .and_then(|value| value.as_bool())
                        .unwrap_or(false)
                    {
                        "ok"
                    } else {
                        "failed"
                    }
                ),
                format!(
                    "regression: {}",
                    if summary
                        .get("regression_ok")
                        .and_then(|value| value.as_bool())
                        .unwrap_or(false)
                    {
                        "ok"
                    } else {
                        "failed"
                    }
                )];

            ToolResult {
                name: call.name.clone(),
                success: result
                    .get("ok")
                    .and_then(|value| value.as_bool())
                    .unwrap_or(false),
                output: lines.join("\n"),
            }
        }
        Err(error) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: error,
        },
    }
}

pub(crate) async fn execute_test_apps(call: &ToolCall) -> ToolResult {
    match execute_test_apps_json(call).await {
        Ok(result) => {
            let summary = result
                .get("summary")
                .cloned()
                .unwrap_or_else(|| serde_json::json!({}));
            let mut lines = vec![
                "App test report".to_string(),
                format!(
                    "apps: {}/{} passed",
                    summary
                        .get("apps_passed")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0),
                    summary
                        .get("apps_total")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0)
                ),
                format!(
                    "checks: {}/{} passed",
                    summary
                        .get("checks_passed")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0),
                    summary
                        .get("checks_total")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0)
                ),
            ];

            if let Some(apps) = result.get("apps").and_then(|value| value.as_array()) {
                for app in apps {
                    let slug = app
                        .get("slug")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown");
                    let ok = app.get("ok").and_then(|v| v.as_bool()).unwrap_or(false);
                    lines.push(format!("- {}: {}", slug, if ok { "ok" } else { "failed" }));
                    if let Some(results) = app.get("results").and_then(|v| v.as_array()) {
                        for item in results {
                            let check = item.get("check").and_then(|v| v.as_str()).unwrap_or("?");
                            let status =
                                if item.get("ok").and_then(|v| v.as_bool()).unwrap_or(false) {
                                    "ok"
                                } else {
                                    "failed"
                                };
                            let duration = item
                                .get("duration_seconds")
                                .and_then(|v| v.as_f64())
                                .unwrap_or(0.0);
                            lines.push(format!("  {}: {} ({:.2}s)", check, status, duration));
                        }
                    }
                }
            }

            ToolResult {
                name: call.name.clone(),
                success: result.get("ok").and_then(|v| v.as_bool()).unwrap_or(false),
                output: lines.join("\n"),
            }
        }
        Err(error) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: error,
        },
    }
}

pub(crate) async fn execute_verify_app_health_json(call: &ToolCall) -> Result<Value, String> {
    let requested_app = call
        .arguments
        .get("app")
        .and_then(|value| value.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if requested_app.is_empty() {
        return Err("Missing required 'app' parameter.".to_string());
    }
    let app = canonicalize_app_slug(&requested_app).unwrap_or(requested_app);

    let compile_checks: Vec<String> = call
        .arguments
        .get("compile_checks")
        .and_then(|value| value.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str().map(ToOwned::to_owned))
                .collect()
        })
        .filter(|checks: &Vec<String>| !checks.is_empty())
        .unwrap_or_else(|| vec!["check".to_string()]);
    let runtime_suite = call
        .arguments
        .get("runtime_suite")
        .and_then(|value| value.as_str())
        .unwrap_or("regression")
        .to_string();
    let run_runtime = call
        .arguments
        .get("run_runtime")
        .and_then(|value| value.as_bool())
        .unwrap_or(true);
    let include_logs = call
        .arguments
        .get("include_logs")
        .and_then(|value| value.as_bool())
        .unwrap_or(true);
    let timeout_seconds = call
        .arguments
        .get("timeout_seconds")
        .and_then(|value| value.as_i64())
        .unwrap_or(60)
        .clamp(5, 3600);

    let compile_call = ToolCall {
        name: "test_apps".to_string(),
        arguments: serde_json::json!({
            "apps": [app.clone()],
            "checks": compile_checks,
            "timeout_seconds": timeout_seconds,
            "fail_fast": false
        }),
    };
    let compile = execute_test_apps_json(&compile_call).await?;
    let compile_ok = compile
        .get("ok")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);

    let runtime = if run_runtime {
        Some(
            execute_smoke_apps_json(&ToolCall {
                name: "smoke_apps".to_string(),
                arguments: serde_json::json!({
                    "apps": [app.clone()],
                    "suite": runtime_suite,
                    "timeout_seconds": timeout_seconds.min(120),
                    "fail_fast": false
                }),
            })
            .await?,
        )
    } else {
        None
    };
    let runtime_ok = runtime
        .as_ref()
        .and_then(|value| value.get("ok"))
        .and_then(|value| value.as_bool())
        .unwrap_or(true);

    let overall_ok = compile_ok && runtime_ok;

    let diagnosis = if overall_ok {
        None
    } else {
        let diagnosis_call = ToolCall {
            name: "diagnose_services".to_string(),
            arguments: serde_json::json!({
                "service_filter": app,
                "include_logs": include_logs
            }),
        };
        let diagnosis_result = execute_diagnose_services(&diagnosis_call).await;
        Some(serde_json::json!({
            "ok": diagnosis_result.success,
            "output": diagnosis_result.output
        }))
    };

    let logs = if overall_ok || !include_logs {
        None
    } else {
        let logs_call = ToolCall {
            name: "read_os_logs".to_string(),
            arguments: serde_json::json!({
                "service": app,
                "lines": 40
            }),
        };
        let logs_result = execute_read_os_logs(&logs_call).await;
        Some(serde_json::json!({
            "ok": logs_result.success,
            "output": logs_result.output
        }))
    };

    Ok(serde_json::json!({
        "ok": overall_ok,
        "app": app,
        "compile": compile,
        "runtime": runtime,
        "diagnosis": diagnosis,
        "logs": logs
    }))
}

pub(crate) async fn execute_verify_app_health(call: &ToolCall) -> ToolResult {
    match execute_verify_app_health_json(call).await {
        Ok(result) => {
            let app = result
                .get("app")
                .and_then(|value| value.as_str())
                .unwrap_or("unknown");
            let compile_ok = result
                .get("compile")
                .and_then(|value| value.get("ok"))
                .and_then(|value| value.as_bool())
                .unwrap_or(false);
            let runtime_ok = result
                .get("runtime")
                .and_then(|value| value.get("ok"))
                .and_then(|value| value.as_bool())
                .unwrap_or(true);

            let mut lines = vec![
                format!("App verification: {}", app),
                format!("compile: {}", if compile_ok { "ok" } else { "failed" }),
                format!("runtime: {}", if runtime_ok { "ok" } else { "failed" }),
            ];

            if let Some(diagnosis) = result.get("diagnosis").filter(|value| !value.is_null()) {
                lines.push("diagnosis: included".to_string());
                if let Some(output) = diagnosis.get("output").and_then(|value| value.as_str()) {
                    let excerpt = output.lines().take(12).collect::<Vec<_>>().join("\n");
                    lines.push(excerpt);
                }
            }

            if let Some(logs) = result.get("logs").filter(|value| !value.is_null()) {
                lines.push("logs: included".to_string());
                if let Some(output) = logs.get("output").and_then(|value| value.as_str()) {
                    let excerpt = output.lines().take(12).collect::<Vec<_>>().join("\n");
                    lines.push(excerpt);
                }
            }

            ToolResult {
                name: call.name.clone(),
                success: result
                    .get("ok")
                    .and_then(|value| value.as_bool())
                    .unwrap_or(false),
                output: lines.join("\n"),
            }
        }
        Err(error) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: error,
        },
    }
}

