//! Permission checks, allowlist logic, and audit logging

use std::fs::OpenOptions;
use std::io::Write;

use super::{ToolCall, ToolRiskLevel, DEFAULT_TOOL_TIMEOUT_MS, HERA_TOOL_AUDIT_LOG};
use super::registry::find_tool_runtime_metadata;

pub(crate) fn tool_risk_level(tool_name: &str) -> ToolRiskLevel {
    if let Some(risk) =
        find_tool_runtime_metadata(tool_name).and_then(|metadata| metadata.risk_level)
    {
        return risk;
    }

    match tool_name {
        "run_code" | "write_file" | "update_soul" | "service_restart" | "api_request"
        | "git_manager" | "desktop_click" | "desktop_type" | "generate_access_link"
        | "bash_exec" => ToolRiskLevel::Critical,
        "read_file"
        | "web_scraper"
        | "spawn_parallel_agents"
        | "create_agent"
        | "create_skill"
        | "execute_workflow"
        | "dispatch_email"
        | "bind_telegram_workspace"
        | "edit_app_theme"
        | "git_add"
        | "git_commit"
        | "pm2_restart" => ToolRiskLevel::High,
        _ => ToolRiskLevel::Low,
    }
}

pub(super) fn parse_tool_risk_level(value: &str) -> Option<ToolRiskLevel> {
    match value.trim().to_ascii_lowercase().as_str() {
        "low" => Some(ToolRiskLevel::Low),
        "high" => Some(ToolRiskLevel::High),
        "critical" => Some(ToolRiskLevel::Critical),
        _ => None,
    }
}

pub(crate) fn tool_timeout_ms(tool_name: &str) -> u64 {
    find_tool_runtime_metadata(tool_name)
        .and_then(|metadata| metadata.timeout_ms)
        .unwrap_or(DEFAULT_TOOL_TIMEOUT_MS)
}

pub(crate) fn tool_allowed_callers(tool_name: &str) -> Vec<String> {
    find_tool_runtime_metadata(tool_name)
        .map(|metadata| metadata.allowed_callers.clone())
        .filter(|callers| !callers.is_empty())
        .unwrap_or_else(|| vec!["all".to_string()])
}

pub(crate) fn extract_tool_caller(call: &ToolCall) -> String {
    call.arguments
        .get("_hera")
        .and_then(|value| value.get("caller"))
        .or_else(|| {
            call.arguments
                .get("_hera")
                .and_then(|value| value.get("app_name"))
        })
        .or_else(|| {
            call.arguments
                .get("_hera")
                .and_then(|value| value.get("app"))
        })
        .or_else(|| call.arguments.get("app_name"))
        .or_else(|| call.arguments.get("app"))
        .or_else(|| call.arguments.get("caller"))
        .or_else(|| call.arguments.get("agent"))
        .and_then(|value| value.as_str())
        .unwrap_or("unknown")
        .to_string()
}

pub(crate) fn caller_allowed_for_tool(tool_name: &str, caller: &str) -> bool {
    let allowed_callers = tool_allowed_callers(tool_name);
    if allowed_callers.is_empty()
        || allowed_callers.iter().any(|allowed| allowed == "all")
        || caller.trim().is_empty()
    {
        return true;
    }

    let normalized_caller = caller.trim().to_ascii_lowercase();
    allowed_callers.iter().any(|allowed| {
        let normalized_allowed = allowed.trim().to_ascii_lowercase();
        normalized_allowed == normalized_caller
            || normalized_allowed == format!("app:{normalized_caller}")
            || normalized_allowed == format!("agent:{normalized_caller}")
    })
}

pub(crate) fn audit_tool_execution(
    call: &ToolCall,
    success: bool,
    duration_ms: u128,
    timed_out: bool,
    error: Option<&str>,
) {
    let argument_keys = call
        .arguments
        .as_object()
        .map(|map| map.keys().cloned().collect::<Vec<_>>())
        .unwrap_or_default();

    let record = serde_json::json!({
        "ts": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_secs())
            .unwrap_or_default(),
        "tool": call.name,
        "caller": extract_tool_caller(call),
        "success": success,
        "timed_out": timed_out,
        "duration_ms": duration_ms,
        "execution_kind": find_tool_runtime_metadata(&call.name).and_then(|metadata| metadata.execution_kind.clone()),
        "risk_level": match tool_risk_level(&call.name) {
            ToolRiskLevel::Low => "low",
            ToolRiskLevel::High => "high",
            ToolRiskLevel::Critical => "critical",
        },
        "allowed_callers": tool_allowed_callers(&call.name),
        "argument_keys": argument_keys,
        "error": error,
    });

    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(HERA_TOOL_AUDIT_LOG)
    {
        let _ = writeln!(file, "{}", record);
    }
}

pub fn permissions_allow_tool(permissions: &[String], tool_name: &str) -> bool {
    let has_explicit_tool_grant = permissions.iter().any(|permission| permission == tool_name);
    let has_all = permissions.iter().any(|permission| permission == "all");
    let has_unsafe_all = permissions
        .iter()
        .any(|permission| permission == "unsafe_all" || permission == "system_admin");

    match tool_risk_level(tool_name) {
        ToolRiskLevel::Critical => has_explicit_tool_grant || has_unsafe_all,
        ToolRiskLevel::High => has_explicit_tool_grant || has_all || has_unsafe_all,
        ToolRiskLevel::Low => has_explicit_tool_grant || has_all || has_unsafe_all,
    }
}
