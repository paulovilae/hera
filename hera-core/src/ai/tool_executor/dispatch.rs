//! Tool dispatch and execution

use serde_json::Value;

use crate::ai::tools::{
    apps_latinos, apps_movilo, apps_vetra, brand_studio, data, geo, infra_health, infra_smoke,
    mission_control, platform, productivity, storage,
};

use super::{ToolCall, ToolResult, ToolRiskLevel};
use super::registry::{find_tool_runtime_metadata, is_registered_tool};
use super::security::{
    audit_tool_execution, caller_allowed_for_tool, extract_tool_caller, tool_allowed_callers,
    tool_risk_level, tool_timeout_ms,
};

pub(super) fn tool_result_envelope(call: &ToolCall, result: &ToolResult, duration_ms: u128) -> Value {
    serde_json::json!({
        "ok": result.success,
        "data": {
            "output": result.output,
        },
        "error": if result.success { Value::Null } else { Value::String(result.output.clone()) },
        "meta": {
            "tool": result.name,
            "caller": extract_tool_caller(call),
            "execution_kind": find_tool_runtime_metadata(&result.name).and_then(|metadata| metadata.execution_kind.clone()),
            "risk_level": match tool_risk_level(&result.name) {
                ToolRiskLevel::Low => "low",
                ToolRiskLevel::High => "high",
                ToolRiskLevel::Critical => "critical",
            },
            "duration_ms": duration_ms,
            "allowed_callers": tool_allowed_callers(&result.name),
        },
        "artifacts": []
    })
}

pub(super) fn tool_error_envelope(call: &ToolCall, error: &str, duration_ms: u128) -> Value {
    serde_json::json!({
        "ok": false,
        "data": {
            "output": Value::Null,
        },
        "error": error,
        "meta": {
            "tool": call.name,
            "caller": extract_tool_caller(call),
            "execution_kind": find_tool_runtime_metadata(&call.name).and_then(|metadata| metadata.execution_kind.clone()),
            "risk_level": match tool_risk_level(&call.name) {
                ToolRiskLevel::Low => "low",
                ToolRiskLevel::High => "high",
                ToolRiskLevel::Critical => "critical",
            },
            "duration_ms": duration_ms,
            "allowed_callers": tool_allowed_callers(&call.name),
        },
        "artifacts": []
    })
}

/// Execute a tool call using existing Hera infrastructure.
/// Returns a ToolResult with the output string.
pub async fn execute_tool(call: &ToolCall) -> ToolResult {
    tracing::info!("🔧 [Hera] Executing tool: {}", call.name);

    let start = std::time::Instant::now();
    let tool_name = call.name.clone();
    let caller = extract_tool_caller(call);
    if !caller_allowed_for_tool(&tool_name, &caller) {
        let error = format!(
            "Caller '{}' is not allowed to execute tool '{}'.",
            caller, tool_name
        );
        audit_tool_execution(
            call,
            false,
            start.elapsed().as_millis(),
            false,
            Some(&error),
        );
        return ToolResult {
            name: tool_name,
            success: false,
            output: error,
        };
    }

    let timeout = std::time::Duration::from_millis(tool_timeout_ms(&call.name));
    match tokio::time::timeout(timeout, execute_tool_inner(call)).await {
        Ok(result) => {
            audit_tool_execution(
                call,
                result.success,
                start.elapsed().as_millis(),
                false,
                None,
            );
            result
        }
        Err(_) => {
            tracing::error!(
                "⏰ [Hera] Tool '{}' TIMED OUT after {:?}. Returning error.",
                tool_name,
                timeout
            );
            let error = format!(
                "Error: Tool execution timed out after {} ms.",
                timeout.as_millis()
            );
            audit_tool_execution(call, false, start.elapsed().as_millis(), true, Some(&error));
            ToolResult {
                name: tool_name,
                success: false,
                output: error,
            }
        }
    }
}

async fn dispatch_platform_tool(call: &ToolCall) -> Option<ToolResult> {
    let result = match call.name.as_str() {
        // Mission Control — Agente Q operates the Agile Cockpit (http_adapter).
        "mc_board" => mission_control::execute_mc_board(call).await,
        "mc_create_story" => mission_control::execute_mc_create_story(call).await,
        "mc_move_story" => mission_control::execute_mc_move_story(call).await,
        "mc_create_sprint" => mission_control::execute_mc_create_sprint(call).await,
        "mc_close_sprint" => mission_control::execute_mc_close_sprint(call).await,
        "mc_add_wishlist" => mission_control::execute_mc_add_wishlist(call).await,
        "mc_set_objective" => mission_control::execute_mc_set_objective(call).await,
        "generate_image" | "hera_draw" => platform::execute_draw(call).await,
        "hera_search" => platform::execute_search(call).await,
        "geocode" => geo::execute_geocode(call).await,
        "reverse_geocode" => geo::execute_reverse_geocode(call).await,
        "storage_list" => storage::execute_storage_list(call).await,
        "storage_get_url" => storage::execute_storage_get_url(call).await,
        "storage_put" => storage::execute_storage_put(call).await,
        "hera_speak" => platform::execute_speak(call).await,
        "hera_video" => platform::execute_video(call).await,
        "hera_read_file" | "read_file" => platform::execute_read_file(call).await,
        "hera_update_soul" | "update_soul" => platform::execute_update_soul(call).await,
        "ask_user" => platform::execute_ask_user(call).await,
        "get_system_time" => platform::execute_get_system_time(call).await,
        "run_code" => platform::execute_run_code(call).await,
        "web_scraper" => platform::execute_web_scraper(call).await,
        "write_file" => platform::execute_write_file(call).await,
        "generate_access_link" => platform::execute_generate_access_link(call).await,
        "spline_interact" => platform::execute_spline_interact(call).await,
        "desktop_click" => platform::execute_desktop_click(call).await,
        "desktop_type" => platform::execute_desktop_type(call).await,
        "edit_app_theme" => platform::execute_edit_app_theme(call).await,
        "read_email" => productivity::execute_read_email(call).await,
        "list_calendar_events" => productivity::execute_list_calendar_events(call).await,
        "read_notes" => productivity::execute_read_notes(call).await,
        _ => return None,
    };
    Some(result)
}

async fn dispatch_metadata_tool(call: &ToolCall) -> Option<ToolResult> {
    let metadata = find_tool_runtime_metadata(&call.name)?;
    let execution_kind = metadata.execution_kind.as_deref()?;

    match execution_kind {
        "ipc_native" => {
            if let Some(result) = dispatch_data_tool(call).await {
                Some(result)
            } else if let Some(result) = dispatch_platform_tool(call).await {
                Some(result)
            } else if let Some(result) = dispatch_vetra_tool(call).await {
                Some(result)
            } else {
                dispatch_latinos_tool(call).await
            }
        }
        "cli" => {
            if let Some(result) = dispatch_infra_tool(call).await {
                Some(result)
            } else if let Some(result) = dispatch_data_tool(call).await {
                Some(result)
            } else if let Some(result) = dispatch_latinos_tool(call).await {
                Some(result)
            } else {
                dispatch_platform_tool(call).await
            }
        }
        "direct_rust" => {
            if let Some(result) = dispatch_platform_tool(call).await {
                Some(result)
            } else if let Some(result) = dispatch_infra_tool(call).await {
                Some(result)
            } else if let Some(result) = dispatch_vetra_tool(call).await {
                Some(result)
            } else if let Some(result) = dispatch_data_tool(call).await {
                Some(result)
            } else if let Some(result) = dispatch_movilo_tool(call).await {
                Some(result)
            } else {
                dispatch_latinos_tool(call).await
            }
        }
        "http_adapter" => {
            if let Some(result) = dispatch_brand_tool(call).await {
                Some(result)
            } else if let Some(result) = dispatch_vetra_tool(call).await {
                Some(result)
            } else if let Some(result) = dispatch_data_tool(call).await {
                Some(result)
            } else {
                dispatch_platform_tool(call).await
            }
        }
        _ => None,
    }
}

async fn dispatch_data_tool(call: &ToolCall) -> Option<ToolResult> {
    let result = match call.name.as_str() {
        "memento_query" => data::execute_memento_query(call).await,
        "api_request" => data::execute_api_request(call).await,
        "git_manager" => data::execute_git_manager(call).await,
        "memento_vector_search" => data::execute_memento_vector_search(call).await,
        "save_memory" => productivity::execute_save_memory(call).await,
        "query_memory" => productivity::execute_query_memory(call).await,
        _ => return None,
    };
    Some(result)
}

async fn dispatch_metadata_raw_json_tool(call: &ToolCall) -> Option<Result<Value, String>> {
    let metadata = find_tool_runtime_metadata(&call.name)?;
    let execution_kind = metadata.execution_kind.as_deref()?;

    match execution_kind {
        "ipc_native" if call.name == "memento_query" => {
            Some(data::execute_memento_query_json(call).await)
        }
        "ipc_native" => None,
        "cli" | "direct_rust" => dispatch_raw_json_tool(call).await,
        _ => None,
    }
}

async fn dispatch_infra_tool(call: &ToolCall) -> Option<ToolResult> {
    let result = match call.name.as_str() {
        "caddy_domain_manager" => infra_health::execute_caddy_domain_manager(call).await,
        "query_federation_state" => infra_health::execute_query_federation_state(call).await,
        "system_status" => infra_health::execute_system_status(call).await,
        "diagnose_services" => infra_health::execute_diagnose_services(call).await,
        "service_restart" => infra_health::execute_service_restart(call).await,
        "read_pm2_logs" => infra_health::execute_read_pm2_logs(call).await,
        "read_os_logs" => infra_smoke::execute_read_os_logs(call).await,
        "smoke_apps" => infra_smoke::execute_smoke_apps(call).await,
        "test_apps" => infra_smoke::execute_test_apps(call).await,
        "verify_canonical_stack" => infra_smoke::execute_verify_canonical_stack(call).await,
        "review_all_apps_status" => infra_smoke::execute_review_all_apps_status(call).await,
        "verify_app_health" => infra_smoke::execute_verify_app_health(call).await,
        _ => return None,
    };
    Some(result)
}

async fn dispatch_brand_tool(call: &ToolCall) -> Option<ToolResult> {
    let result = match call.name.as_str() {
        "add_topic" => brand_studio::execute_add_topic(call).await,
        "list_pending_drafts" => brand_studio::execute_list_pending_drafts(call).await,
        "approve_draft" => brand_studio::execute_approve_draft(call).await,
        "capture_post_metrics" => brand_studio::execute_capture_post_metrics(call).await,
        "voice_profile_get" => brand_studio::execute_voice_profile_get(call).await,
        "voice_profile_update" => brand_studio::execute_voice_profile_update(call).await,
        "save_thesis_doc" => brand_studio::execute_save_thesis_doc(call).await,
        _ => return None,
    };
    Some(result)
}

async fn dispatch_vetra_tool(call: &ToolCall) -> Option<ToolResult> {
    let result = match call.name.as_str() {
        "generate_qr_code" => apps_vetra::execute_generate_qr_code(call).await,
        "generate_contract_pdf" => apps_vetra::execute_generate_contract_pdf(call).await,
        "dispatch_email" => apps_vetra::execute_dispatch_email(call).await,
        "get_map_route" => apps_vetra::execute_get_map_route(call).await,
        "execute_workflow" => apps_vetra::execute_workflow(call).await,
        "bind_telegram_workspace" => apps_vetra::execute_bind_telegram_workspace(call).await,
        _ => return None,
    };
    Some(result)
}

async fn dispatch_movilo_tool(call: &ToolCall) -> Option<ToolResult> {
    let result = match call.name.as_str() {
        "movilo_search_providers" => apps_movilo::execute_movilo_search_providers(call).await,
        "movilo_get_plans" => apps_movilo::execute_movilo_get_plans(call).await,
        "movilo_check_affiliation" => apps_movilo::execute_movilo_check_affiliation(call).await,
        "movilo_validate_qr" => apps_movilo::execute_movilo_validate_qr(call).await,
        _ => return None,
    };
    Some(result)
}

async fn dispatch_latinos_tool(call: &ToolCall) -> Option<ToolResult> {
    let result = match call.name.as_str() {
        "list_bots" => apps_latinos::execute_list_bots(call).await,
        "list_markets" => apps_latinos::execute_list_markets(call).await,
        "get_bot_status" => apps_latinos::execute_get_bot_status(call).await,
        "market_research" | "analyze_market_research" => {
            apps_latinos::execute_market_research(call).await
        }
        "consultant_report_analyzer" => {
            apps_latinos::execute_consultant_report_analyzer(call).await
        }
        "run_backtest" => apps_latinos::execute_latinos_bridge(call, "run_backtest").await,
        "load_market_data" => apps_latinos::execute_latinos_bridge(call, "load_market_data").await,
        "scan_opportunities" => {
            apps_latinos::execute_latinos_bridge(call, "scan_opportunities").await
        }
        "generate_pdf" => apps_latinos::execute_latinos_bridge(call, "generate_pdf").await,
        _ => return None,
    };
    Some(result)
}

/// Inner dispatch — called inside the 90s timeout wrapper.
async fn execute_tool_inner(call: &ToolCall) -> ToolResult {
    if call.name.starts_with("load_skill_") {
        return platform::execute_load_skill(call).await;
    }

    if call.name == "spawn_parallel_agents" {
        return platform::execute_spawn_parallel_agents(call).await;
    }

    if call.name == "create_agent" {
        return platform::execute_create_agent(call).await;
    }

    if call.name == "create_skill" {
        return platform::execute_create_skill(call).await;
    }

    if let Some(result) = dispatch_metadata_tool(call).await {
        return result;
    }

    if is_registered_tool(&call.name) {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!(
                "Registered tool '{}' has no metadata-driven runtime dispatcher.",
                call.name
            ),
        };
    }

    ToolResult {
        name: call.name.clone(),
        success: false,
        output: format!("Unknown tool: {}", call.name),
    }
}

async fn dispatch_raw_json_tool(call: &ToolCall) -> Option<Result<Value, String>> {
    let result = match call.name.as_str() {
        "memento_query" => data::execute_memento_query_json(call).await,
        "market_research" | "analyze_market_research" => {
            apps_latinos::execute_market_research_json(call).await
        }
        "consultant_report_analyzer" => {
            apps_latinos::execute_consultant_report_analyzer_json(call).await
        }
        "smoke_apps" => infra_smoke::execute_smoke_apps_json(call).await,
        "test_apps" => infra_smoke::execute_test_apps_json(call).await,
        "verify_canonical_stack" => infra_smoke::execute_verify_canonical_stack_json(call).await,
        "review_all_apps_status" => infra_smoke::execute_review_all_apps_status_json(call).await,
        "verify_app_health" => infra_smoke::execute_verify_app_health_json(call).await,
        _ => return None,
    };
    Some(result)
}

pub async fn execute_tool_raw_json(call: &ToolCall) -> Result<Value, String> {
    let start = std::time::Instant::now();
    let tool_name = call.name.clone();
    let caller = extract_tool_caller(call);
    if !caller_allowed_for_tool(&tool_name, &caller) {
        let error = format!(
            "Caller '{}' is not allowed to execute tool '{}'.",
            caller, tool_name
        );
        audit_tool_execution(
            call,
            false,
            start.elapsed().as_millis(),
            false,
            Some(&error),
        );
        return Ok(tool_error_envelope(
            call,
            &error,
            start.elapsed().as_millis(),
        ));
    }

    let timeout = std::time::Duration::from_millis(tool_timeout_ms(&call.name));
    match tokio::time::timeout(timeout, execute_tool_raw_json_inner(call)).await {
        Ok(result) => {
            let envelope = match result {
                Ok(value) => {
                    if value.get("ok").is_some()
                        && value.get("data").is_some()
                        && value.get("meta").is_some()
                    {
                        value
                    } else {
                        serde_json::json!({
                            "ok": true,
                            "data": value,
                            "error": Value::Null,
                            "meta": {
                                "tool": call.name,
                                "caller": extract_tool_caller(call),
                                "execution_kind": find_tool_runtime_metadata(&call.name).and_then(|metadata| metadata.execution_kind.clone()),
                                "risk_level": match tool_risk_level(&call.name) {
                                    ToolRiskLevel::Low => "low",
                                    ToolRiskLevel::High => "high",
                                    ToolRiskLevel::Critical => "critical",
                                },
                                "duration_ms": start.elapsed().as_millis(),
                                "allowed_callers": tool_allowed_callers(&call.name),
                            },
                            "artifacts": []
                        })
                    }
                }
                Err(error) => tool_error_envelope(call, &error, start.elapsed().as_millis()),
            };

            audit_tool_execution(
                call,
                envelope
                    .get("ok")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false),
                start.elapsed().as_millis(),
                false,
                envelope.get("error").and_then(|v| v.as_str()),
            );
            Ok(envelope)
        }
        Err(_) => {
            tracing::error!(
                "⏰ [Hera] Tool '{}' (raw_json) TIMED OUT after {:?}.",
                tool_name,
                timeout
            );
            let error = format!(
                "Tool '{}' timed out after {} ms.",
                tool_name,
                timeout.as_millis()
            );
            audit_tool_execution(call, false, start.elapsed().as_millis(), true, Some(&error));
            Ok(tool_error_envelope(
                call,
                &error,
                start.elapsed().as_millis(),
            ))
        }
    }
}

/// Inner dispatch for raw JSON tools — called inside the 90s timeout wrapper.
async fn execute_tool_raw_json_inner(call: &ToolCall) -> Result<Value, String> {
    if let Some(result) = dispatch_metadata_raw_json_tool(call).await {
        return result;
    }

    let result = execute_tool_inner(call).await;
    Ok(tool_result_envelope(call, &result, 0))
}
