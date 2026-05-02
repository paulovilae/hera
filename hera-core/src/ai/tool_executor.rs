//! Hera Tool Executor — Sovereign Tool Calling for ImagineOS
//!
//! Defines tool schemas in Qwen's native format, parses `<tool_call>` blocks
//! from Qwen output, and dispatches tool execution to existing Hera methods.

use crate::ai::tools::{
    apps_latinos, apps_movilo, apps_vetra, data, infra_health, infra_smoke, platform, productivity,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use tracing::info;

/// Tool call parsed from Qwen's output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Result of executing a tool
#[derive(Debug, Clone)]
pub struct ToolResult {
    pub name: String,
    pub success: bool,
    pub output: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolRiskLevel {
    Low,
    High,
    Critical,
}

#[derive(Debug, Clone)]
struct ToolArtifact {
    schema: Value,
    consumers: Vec<String>,
}

#[derive(Debug, Clone, Default)]
struct ToolRuntimeMetadata {
    execution_kind: Option<String>,
    risk_level: Option<ToolRiskLevel>,
    timeout_ms: Option<u64>,
    allowed_callers: Vec<String>,
}

static TOOL_RUNTIME_REGISTRY: OnceLock<HashMap<String, ToolRuntimeMetadata>> = OnceLock::new();
static REGISTERED_TOOL_NAMES: OnceLock<HashSet<String>> = OnceLock::new();
const DEFAULT_TOOL_TIMEOUT_MS: u64 = 90_000;
const HERA_TOOL_AUDIT_LOG: &str = "/tmp/hera_tool_audit.jsonl";

#[derive(Debug, Clone)]
pub(crate) struct SkillArtifact {
    pub skill_id: String,
    pub tool_name: String,
    pub description: String,
    pub content: String,
}

#[derive(Debug, Clone)]
pub(crate) struct AgentArtifact {
    pub persona: String,
}

#[derive(Debug, Clone)]
pub(crate) struct CanonicalAppEntry {
    pub slug: String,
    pub path: String,
    pub manifest: String,
}

pub(crate) fn load_canonical_app_registry() -> Vec<CanonicalAppEntry> {
    let path = "/home/paulo/Programs/apps/OS/etc/apps.toml";
    let Ok(content) = std::fs::read_to_string(path) else {
        return Vec::new();
    };

    let mut apps = Vec::new();
    let mut current_slug: Option<String> = None;
    let mut current_path: Option<String> = None;
    let mut current_manifest: Option<String> = None;

    let flush_current = |apps: &mut Vec<CanonicalAppEntry>,
                         slug: &mut Option<String>,
                         path: &mut Option<String>,
                         manifest: &mut Option<String>| {
        if let (Some(slug), Some(path), Some(manifest)) =
            (slug.take(), path.take(), manifest.take())
        {
            apps.push(CanonicalAppEntry {
                slug,
                path,
                manifest,
            });
        }
    };

    for raw_line in content.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line == "[[apps]]" {
            flush_current(
                &mut apps,
                &mut current_slug,
                &mut current_path,
                &mut current_manifest,
            );
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        let value = value.trim().trim_matches('"').to_string();
        match key {
            "slug" => current_slug = Some(value),
            "path" => current_path = Some(value),
            "manifest" => current_manifest = Some(value),
            _ => {}
        }
    }
    flush_current(
        &mut apps,
        &mut current_slug,
        &mut current_path,
        &mut current_manifest,
    );
    apps
}

fn alias_terms_for_app(entry: &CanonicalAppEntry) -> Vec<String> {
    let mut aliases = std::collections::BTreeSet::new();
    let slug = entry.slug.to_lowercase();
    aliases.insert(slug.clone());
    aliases.insert(slug.replace('-', " "));

    if let Some(last) = entry.path.split('/').next_back() {
        let lowered = last.to_lowercase();
        aliases.insert(lowered.clone());
        aliases.insert(lowered.replace('-', " "));
    }

    if let Some(last) = entry.manifest.split('/').next_back() {
        let lowered = last.trim_end_matches(".toml").to_lowercase();
        aliases.insert(lowered);
    }

    match entry.slug.as_str() {
        "latinos" => {
            aliases.insert("latinos-rust".to_string());
        }
        "vetra" => {
            aliases.insert("vetra-rust".to_string());
        }
        "movilo" => {
            aliases.insert("movilo-v3".to_string());
            aliases.insert("movilo-prod".to_string());
        }
        "os-v3" => {
            aliases.insert("os".to_string());
            aliases.insert("portal".to_string());
            aliases.insert("os-portal".to_string());
        }
        "desktop" => {
            aliases.insert("desktop-rust".to_string());
        }
        "paulo-vila-rust" => {
            aliases.insert("paulo vila".to_string());
            aliases.insert("paulovila".to_string());
            aliases.insert("paulo-vila".to_string());
        }
        "capacita" => {
            aliases.insert("capacita-rust".to_string());
        }
        _ => {}
    }

    aliases.into_iter().collect()
}

pub(crate) fn canonicalize_app_slug(input: &str) -> Option<String> {
    let needle = input.trim().to_lowercase();
    if needle.is_empty() {
        return None;
    }

    for entry in load_canonical_app_registry() {
        let aliases = alias_terms_for_app(&entry);
        if aliases.iter().any(|alias| alias == &needle) {
            return Some(entry.slug);
        }
    }
    None
}

pub(crate) fn canonical_app_search_terms(input: &str) -> Vec<String> {
    let canonical = canonicalize_app_slug(input).unwrap_or_else(|| input.trim().to_lowercase());
    let mut terms = std::collections::BTreeSet::new();

    if let Some(entry) = load_canonical_app_registry()
        .into_iter()
        .find(|entry| entry.slug == canonical)
    {
        for alias in alias_terms_for_app(&entry) {
            if !alias.is_empty() {
                terms.insert(alias);
            }
        }
        if let Some(last) = entry.path.split('/').next_back() {
            terms.insert(last.to_lowercase().replace('_', "-"));
        }
    } else if !canonical.is_empty() {
        terms.insert(canonical);
    }

    terms.into_iter().collect()
}

pub(crate) fn text_contains_app_alias(text: &str, aliases: &[String]) -> bool {
    let lower = text.to_lowercase();
    aliases.iter().any(|alias| lower.contains(alias))
}

pub(crate) fn tool_risk_level(tool_name: &str) -> ToolRiskLevel {
    if let Some(risk) =
        find_tool_runtime_metadata(tool_name).and_then(|metadata| metadata.risk_level)
    {
        return risk;
    }

    match tool_name {
        "run_code" | "write_file" | "update_soul" | "service_restart" | "api_request"
        | "git_manager" | "desktop_click" | "desktop_type" => ToolRiskLevel::Critical,
        "read_file"
        | "web_scraper"
        | "spawn_parallel_agents"
        | "create_agent"
        | "create_skill"
        | "execute_workflow"
        | "dispatch_email"
        | "bind_telegram_workspace"
        | "edit_app_theme" => ToolRiskLevel::High,
        _ => ToolRiskLevel::Low,
    }
}

fn parse_tool_risk_level(value: &str) -> Option<ToolRiskLevel> {
    match value.trim().to_ascii_lowercase().as_str() {
        "low" => Some(ToolRiskLevel::Low),
        "high" => Some(ToolRiskLevel::High),
        "critical" => Some(ToolRiskLevel::Critical),
        _ => None,
    }
}

fn tool_timeout_ms(tool_name: &str) -> u64 {
    find_tool_runtime_metadata(tool_name)
        .and_then(|metadata| metadata.timeout_ms)
        .unwrap_or(DEFAULT_TOOL_TIMEOUT_MS)
}

fn tool_allowed_callers(tool_name: &str) -> Vec<String> {
    find_tool_runtime_metadata(tool_name)
        .map(|metadata| metadata.allowed_callers.clone())
        .filter(|callers| !callers.is_empty())
        .unwrap_or_else(|| vec!["all".to_string()])
}

fn extract_tool_caller(call: &ToolCall) -> String {
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

fn caller_allowed_for_tool(tool_name: &str, caller: &str) -> bool {
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

fn tool_result_envelope(call: &ToolCall, result: &ToolResult, duration_ms: u128) -> Value {
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

fn tool_error_envelope(call: &ToolCall, error: &str, duration_ms: u128) -> Value {
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

fn audit_tool_execution(
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

pub(crate) fn permissions_allow_tool(permissions: &[String], tool_name: &str) -> bool {
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

pub(crate) fn pm2_process_name_for_slug(slug: &str) -> &str {
    match slug {
        "acciona" => "acciona-rust",
        "cartera" => "cartera-rust",
        "vetra" => "vetra-rust",
        "movilo" => "movilo",
        "latinos" => "latinos-rust",
        "os-v3" => "os-v3",
        "desktop" => "desktop-rust",
        "paulo-vila-rust" => "paulo-vila",
        "capacita" => "capacita-rust",
        "hera" => "hera-core",
        "whisper" => "hera-core",
        "audio-stt" => "hera-core",
        "audio-engine" => "hera-core",
        "argus" => "argus",
        "memento" => "memento-node",
        _ => slug,
    }
}

fn parse_tool_artifact(path: &Path) -> Option<ToolArtifact> {
    let content = std::fs::read_to_string(path).ok()?;
    let mut schema = serde_json::from_str::<Value>(&content).ok()?;
    let consumers = schema
        .get("metadata")
        .and_then(|metadata| metadata.get("consumers"))
        .and_then(|value| value.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str().map(ToString::to_string))
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec!["all".to_string()]);

    if let Some(obj) = schema.as_object_mut() {
        obj.remove("metadata");
    }

    Some(ToolArtifact { schema, consumers })
}

fn collect_tool_runtime_metadata_in_dir(
    dir: &Path,
    registry: &mut HashMap<String, ToolRuntimeMetadata>,
) {
    if !dir.exists() {
        return;
    }

    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };

    for entry in entries.flatten() {
        let entry_path = entry.path();
        if entry_path.is_dir() {
            collect_tool_runtime_metadata_in_dir(&entry_path, registry);
            continue;
        }

        if entry_path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }

        let content = match std::fs::read_to_string(&entry_path) {
            Ok(content) => content,
            Err(_) => continue,
        };
        let schema = match serde_json::from_str::<Value>(&content) {
            Ok(schema) => schema,
            Err(_) => continue,
        };
        let Some(tool_name) = schema
            .get("function")
            .and_then(|function| function.get("name"))
            .and_then(|value| value.as_str())
        else {
            continue;
        };
        let metadata = ToolRuntimeMetadata {
            execution_kind: schema
                .get("metadata")
                .and_then(|value| value.get("execution_kind"))
                .and_then(|value| value.as_str())
                .map(ToString::to_string),
            risk_level: schema
                .get("metadata")
                .and_then(|value| value.get("risk_level"))
                .and_then(|value| value.as_str())
                .and_then(parse_tool_risk_level),
            timeout_ms: schema
                .get("metadata")
                .and_then(|value| value.get("timeout_ms"))
                .and_then(|value| value.as_u64()),
            allowed_callers: schema
                .get("metadata")
                .and_then(|value| value.get("allowed_callers"))
                .and_then(|value| value.as_array())
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|item| item.as_str().map(ToString::to_string))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default(),
        };
        if metadata.execution_kind.is_some() {
            registry.insert(tool_name.to_string(), metadata);
        }
    }
}

fn collect_tool_names_in_dir(dir: &Path, names: &mut HashSet<String>) {
    if !dir.exists() {
        return;
    }

    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };

    for entry in entries.flatten() {
        let entry_path = entry.path();
        if entry_path.is_dir() {
            collect_tool_names_in_dir(&entry_path, names);
            continue;
        }
        if entry_path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }
        let content = match std::fs::read_to_string(&entry_path) {
            Ok(content) => content,
            Err(_) => continue,
        };
        let schema = match serde_json::from_str::<Value>(&content) {
            Ok(schema) => schema,
            Err(_) => continue,
        };
        if let Some(tool_name) = schema
            .get("function")
            .and_then(|function| function.get("name"))
            .and_then(|value| value.as_str())
        {
            names.insert(tool_name.to_string());
        }
    }
}

fn tool_runtime_registry() -> &'static HashMap<String, ToolRuntimeMetadata> {
    TOOL_RUNTIME_REGISTRY.get_or_init(|| {
        let mut registry = HashMap::new();
        collect_tool_runtime_metadata_in_dir(
            Path::new("/home/paulo/Programs/apps/OS/Tools"),
            &mut registry,
        );
        registry
    })
}

fn find_tool_runtime_metadata(tool_name: &str) -> Option<&'static ToolRuntimeMetadata> {
    tool_runtime_registry().get(tool_name)
}

fn registered_tool_names() -> &'static HashSet<String> {
    REGISTERED_TOOL_NAMES.get_or_init(|| {
        let mut names = HashSet::new();
        collect_tool_names_in_dir(Path::new("/home/paulo/Programs/apps/OS/Tools"), &mut names);
        names
    })
}

fn is_registered_tool(tool_name: &str) -> bool {
    registered_tool_names().contains(tool_name)
}

#[cfg(test)]
fn is_platform_tool_name(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "generate_image"
            | "hera_draw"
            | "hera_search"
            | "hera_speak"
            | "hera_video"
            | "hera_read_file"
            | "read_file"
            | "hera_update_soul"
            | "update_soul"
            | "ask_user"
            | "get_system_time"
            | "run_code"
            | "web_scraper"
            | "write_file"
            | "spline_interact"
            | "desktop_click"
            | "desktop_type"
            | "edit_app_theme"
    )
}

#[cfg(test)]
fn is_data_tool_name(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "memento_query" | "api_request" | "git_manager" | "memento_vector_search"
    )
}

#[cfg(test)]
fn is_infra_tool_name(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "caddy_domain_manager"
            | "system_status"
            | "diagnose_services"
            | "service_restart"
            | "read_pm2_logs"
            | "read_os_logs"
            | "smoke_apps"
            | "test_apps"
            | "verify_canonical_stack"
            | "review_all_apps_status"
            | "verify_app_health"
    )
}

#[cfg(test)]
fn is_vetra_tool_name(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "generate_qr_code"
            | "generate_contract_pdf"
            | "dispatch_email"
            | "get_map_route"
            | "execute_workflow"
            | "bind_telegram_workspace"
    )
}

#[cfg(test)]
fn is_movilo_tool_name(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "movilo_search_providers" | "movilo_check_affiliation" | "movilo_validate_qr"
    )
}

#[cfg(test)]
fn is_latinos_tool_name(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "list_bots"
            | "get_bot_status"
            | "market_research"
            | "analyze_market_research"
            | "consultant_report_analyzer"
            | "run_backtest"
            | "load_market_data"
            | "scan_opportunities"
    )
}

#[cfg(test)]
fn tool_has_runtime_dispatch(tool_name: &str, execution_kind: Option<&str>) -> bool {
    if matches!(
        tool_name,
        "spawn_parallel_agents" | "create_agent" | "create_skill"
    ) {
        return true;
    }

    match execution_kind {
        Some("ipc_native") => {
            is_data_tool_name(tool_name)
                || is_platform_tool_name(tool_name)
                || is_vetra_tool_name(tool_name)
                || is_latinos_tool_name(tool_name)
        }
        Some("cli") => {
            is_infra_tool_name(tool_name)
                || is_data_tool_name(tool_name)
                || is_latinos_tool_name(tool_name)
                || is_platform_tool_name(tool_name)
        }
        Some("direct_rust") => {
            is_platform_tool_name(tool_name)
                || is_infra_tool_name(tool_name)
                || is_vetra_tool_name(tool_name)
                || is_data_tool_name(tool_name)
                || is_movilo_tool_name(tool_name)
                || is_latinos_tool_name(tool_name)
        }
        Some("http_adapter") => {
            is_vetra_tool_name(tool_name)
                || is_data_tool_name(tool_name)
                || is_platform_tool_name(tool_name)
        }
        _ => false,
    }
}

#[cfg(test)]
fn tool_has_raw_json_dispatch(tool_name: &str, execution_kind: Option<&str>) -> bool {
    match execution_kind {
        Some("ipc_native") => tool_name == "memento_query",
        Some("cli") | Some("direct_rust") => matches!(
            tool_name,
            "memento_query"
                | "market_research"
                | "analyze_market_research"
                | "consultant_report_analyzer"
                | "smoke_apps"
                | "test_apps"
                | "verify_canonical_stack"
                | "review_all_apps_status"
                | "verify_app_health"
        ),
        _ => false,
    }
}

fn collect_tool_schemas_from_dir(dir: &Path, tools: &mut Vec<Value>, agent_name: &str) {
    if !dir.exists() {
        return;
    }

    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let entry_path = entry.path();
            if entry_path.is_dir() {
                collect_tool_schemas_from_dir(&entry_path, tools, agent_name);
                continue;
            }

            if entry_path.extension().and_then(|ext| ext.to_str()) != Some("json") {
                continue;
            }

            match parse_tool_artifact(&entry_path) {
                Some(artifact) => {
                    let allowed = artifact
                        .consumers
                        .iter()
                        .any(|consumer| consumer == "all" || consumer == agent_name);
                    if allowed {
                        tools.push(artifact.schema);
                    } else {
                        tracing::debug!(
                            "Skipping tool due to consumer restriction: {:?}",
                            entry_path
                        );
                    }
                }
                None => {
                    eprintln!("Warning: Failed to parse tool JSON at {:?}", entry_path);
                }
            }
        }
    }
}

fn parse_skill_artifact(skill_dir: &Path) -> Option<SkillArtifact> {
    let path = skill_dir.join("SKILL.md");
    let content = std::fs::read_to_string(&path).ok()?;
    let mut tool_name = String::new();
    let mut description = String::new();
    let mut in_frontmatter = false;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed == "---" {
            if in_frontmatter {
                break;
            }
            in_frontmatter = true;
            continue;
        }
        if !in_frontmatter {
            continue;
        }
        if let Some(value) = trimmed.strip_prefix("name:") {
            tool_name = value
                .trim()
                .trim_matches('"')
                .trim_matches('\'')
                .to_string();
        } else if let Some(value) = trimmed.strip_prefix("description:") {
            description = value
                .trim()
                .trim_matches('"')
                .trim_matches('\'')
                .to_string();
        }
    }

    if tool_name.is_empty() || description.is_empty() {
        return None;
    }

    Some(SkillArtifact {
        skill_id: skill_dir.file_name()?.to_string_lossy().to_string(),
        tool_name,
        description,
        content,
    })
}

fn collect_skill_artifacts() -> Vec<SkillArtifact> {
    let skills_dir = Path::new("/home/paulo/Programs/apps/OS/Skills");
    let mut skills = Vec::new();
    if let Ok(entries) = std::fs::read_dir(skills_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir()
                && let Some(skill) = parse_skill_artifact(&path)
            {
                skills.push(skill);
            }
        }
    }
    skills
}

pub(crate) fn find_skill_artifact(tool_name: &str) -> Option<SkillArtifact> {
    collect_skill_artifacts()
        .into_iter()
        .find(|skill| skill.tool_name == tool_name)
}

pub(crate) fn load_agent_artifact(agent_name: &str) -> AgentArtifact {
    let sanitized = agent_name
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || *ch == '_' || *ch == '-')
        .collect::<String>();
    let agent_path = PathBuf::from(format!(
        "/home/paulo/Programs/apps/OS/Agents/{}.md",
        sanitized
    ));
    let persona = std::fs::read_to_string(&agent_path)
        .unwrap_or_else(|_| format!("You are an expert {}", sanitized));
    AgentArtifact { persona }
}

/// Tool schemas in Qwen3's native Hermes-style format.
/// Uses the exact JSON schema structure that Qwen3 was trained on.
/// Reference: https://qwen3.org/docs/guides/tools
///
/// Folder structure:
///   OS/Tools/global/{db,ai,system,files,agents,workflow,misc}/*.json — always loaded
///   OS/Tools/apps/{vetra,latinos,garcero,movilo}/*.json — loaded per permissions
///
/// Permissions:
///   ["all"]       → loads everything (global + all apps)
///   ["garcero"]   → loads global + apps/garcero/
///   ["movilo"]    → loads global + apps/movilo/
///   []            → loads nothing (no tools in prompt)
pub fn hera_tool_schemas(permissions: &[String], agent_name: &str) -> String {
    let base_dir = "/home/paulo/Programs/apps/OS/Tools";
    let mut tools_vec: Vec<Value> = Vec::new();

    // Empty permissions = no tools at all (e.g., Chigüí doing pure LLM generation)
    if permissions.is_empty() {
        return "".to_string();
    }

    let has_all = permissions.contains(&"all".to_string());

    // 1. Always load global tools (recursive through topic subfolders)
    let global_dir = PathBuf::from(format!("{}/global", base_dir));
    collect_tool_schemas_from_dir(&global_dir, &mut tools_vec, agent_name);

    // 2. Load app-specific tools based on permissions
    if has_all {
        // Load ALL app tools
        let apps_dir = PathBuf::from(format!("{}/apps", base_dir));
        collect_tool_schemas_from_dir(&apps_dir, &mut tools_vec, agent_name);
    } else {
        // Load only requested app tool folders
        for perm in permissions {
            let app_dir = PathBuf::from(format!("{}/apps/{}", base_dir, perm));
            collect_tool_schemas_from_dir(&app_dir, &mut tools_vec, agent_name);
        }
    }

    // 3. Dynamic Skill Disclosure: Parse OS/Skills for SKILL.md
    for skill in collect_skill_artifacts() {
        let skill_tool = serde_json::json!({
            "type": "function",
            "function": {
                "name": skill.tool_name,
                "description": skill.description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        });
        tools_vec.push(skill_tool);
    }

    if tools_vec.is_empty() {
        return "".to_string();
    }

    let tools_json = serde_json::to_string_pretty(&tools_vec).unwrap_or_default();

    format!(
        r#"

# Tools

You may call one or more functions to assist with the user query.

You are provided with function definitions below:

{tools_json}

For each function call, return a JSON object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": "function_name", "arguments": {{"arg1": "value1"}}}}
</tool_call>"#
    )
}

/// Parse tool call blocks from LLM text output.
/// Supports multiple tag formats: <tool_call>, <function-call>, <function_call>
/// Returns empty vec if no tool calls found.
pub fn parse_tool_calls(text: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();

    // Try multiple tag formats that various LLMs hallucinate
    let tag_formats: &[(&str, &str)] = &[
        ("<tool_call>", "</tool_call>"),
        ("<tool_code>", "</tool_code>"),
        ("<function-call>", "</function-call>"),
        ("<function_call>", "</function_call>"),
    ];

    for &(open_tag, close_tag) in tag_formats {
        let mut search_from = 0;
        while let Some(start) = text[search_from..].find(open_tag) {
            let abs_start = search_from + start + open_tag.len();
            if let Some(end) = text[abs_start..].find(close_tag) {
                let abs_end = abs_start + end;
                let mut json_str = text[abs_start..abs_end].trim();
                // Strip any nested hallucinated tags
                json_str = json_str.trim_start_matches("<function-call>").trim();
                json_str = json_str.trim_end_matches("</function-call>").trim();
                json_str = json_str.trim_start_matches("<tool_call>").trim();
                json_str = json_str.trim_end_matches("</tool_call>").trim();
                json_str = json_str.trim_start_matches("<tool_code>").trim();
                json_str = json_str.trim_end_matches("</tool_code>").trim();
                json_str = json_str.trim_start_matches("```json").trim();
                json_str = json_str.trim_start_matches("```").trim();
                json_str = json_str.trim_end_matches("```").trim();

                match parse_tool_call_json(json_str) {
                    Ok(val) => {
                        append_tool_calls_from_value(&mut calls, &val, open_tag);
                    }
                    Err(e) => {
                        tracing::warn!(
                            "⚠️ [Hera] Failed to parse tool_call JSON: {} — raw: {}",
                            e,
                            json_str
                        );
                    }
                }
                search_from = abs_end + close_tag.len();
            } else {
                break;
            }
        }
        // If we found calls with this tag format, stop trying others
        if !calls.is_empty() {
            return calls;
        }
    }

    // Llama/Nemotron format: <function=NAME><parameter=KEY>VALUE</parameter>...</function>
    if calls.is_empty() {
        let mut search_from = 0;
        while let Some(start) = text[search_from..].find("<function=") {
            let abs_start = search_from + start + "<function=".len();
            // Find the function name (up to the next > character)
            if let Some(name_end) = text[abs_start..].find('>') {
                let abs_name_end = abs_start + name_end;
                let func_name = text[abs_start..abs_name_end].trim();

                // Find the closing </function> tag
                if let Some(func_end) = text[abs_name_end..].find("</function>") {
                    let abs_func_end = abs_name_end + func_end;
                    let body = &text[abs_name_end + 1..abs_func_end];

                    // Parse <parameter=KEY>VALUE</parameter> pairs
                    let mut args = serde_json::Map::new();
                    let mut param_search = 0;
                    while let Some(ps) = body[param_search..].find("<parameter=") {
                        let p_start = param_search + ps + "<parameter=".len();
                        if let Some(p_name_end) = body[p_start..].find('>') {
                            let abs_p_name_end = p_start + p_name_end;
                            let param_name = body[p_start..abs_p_name_end].trim();
                            if let Some(p_val_end) = body[abs_p_name_end + 1..].find("</parameter>")
                            {
                                let abs_p_val_end = abs_p_name_end + 1 + p_val_end;
                                let param_value = body[abs_p_name_end + 1..abs_p_val_end].trim();
                                args.insert(
                                    param_name.to_string(),
                                    serde_json::Value::String(param_value.to_string()),
                                );
                                param_search = abs_p_val_end + "</parameter>".len();
                            } else {
                                break;
                            }
                        } else {
                            break;
                        }
                    }

                    if !func_name.is_empty() {
                        calls.push(ToolCall {
                            name: func_name.to_string(),
                            arguments: serde_json::Value::Object(args),
                        });
                        info!(
                            "🔧 [Hera] Parsed Llama-style tool call: {} with args: {}",
                            func_name,
                            serde_json::to_string(&calls.last().unwrap().arguments)
                                .unwrap_or_default()
                        );
                    }
                    search_from = abs_func_end + "</function>".len();
                } else {
                    break;
                }
            } else {
                break;
            }
        }
    }

    // Fallback: If no tags were found, maybe the model just spit out raw JSON?
    // Strip <think>...</think> reasoning tags first (Qwen models produce these)
    if calls.is_empty() {
        let stripped = if let Some(end_idx) = text.find("</think>") {
            text[end_idx + 8..].trim()
        } else {
            text.trim()
        };
        if stripped.starts_with('{') {
            let mut brace_count = 0;
            let mut end_idx = 0;
            for (i, c) in stripped.char_indices() {
                if c == '{' {
                    brace_count += 1;
                } else if c == '}' {
                    brace_count -= 1;
                }

                if brace_count == 0 && i > 0 {
                    end_idx = i + 1;
                    break;
                }
            }

            let mut json_str = String::new();
            if end_idx > 0 {
                json_str = stripped[..end_idx].to_string();
            } else if brace_count > 0 {
                json_str = stripped.to_string();
                for _ in 0..brace_count {
                    json_str.push('}');
                }
            }

            if !json_str.is_empty() {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&json_str) {
                    let mut parsed_name = None;
                    let mut parsed_args = None;

                    if let (Some(name), Some(args)) = (
                        val.get("name").and_then(|n| n.as_str()),
                        val.get("arguments").or_else(|| val.get("parameters")),
                    ) {
                        parsed_name = Some(name.to_string());
                        parsed_args = Some(args.clone());
                    } else if let Some(func) = val.get("function") {
                        if let (Some(name), Some(args)) = (
                            func.get("name").and_then(|n| n.as_str()),
                            func.get("arguments").or_else(|| func.get("parameters")),
                        ) {
                            parsed_name = Some(name.to_string());
                            parsed_args = Some(args.clone());
                        }
                    }

                    if let (Some(name), Some(args)) = (parsed_name, parsed_args) {
                        calls.push(ToolCall {
                            name: name.clone(),
                            arguments: args.clone(),
                        });
                        info!(
                            "🔧 [Hera] Parsed RAW JSON tool call (after think-strip): {} with args: {}",
                            name,
                            serde_json::to_string(&args).unwrap_or_default()
                        );
                    }
                }
            }
        }
    }

    calls
}

fn parse_tool_call_json(json_str: &str) -> Result<serde_json::Value, serde_json::Error> {
    match serde_json::from_str::<serde_json::Value>(json_str) {
        Ok(value) => Ok(value),
        Err(first_error) => {
            if !matches!(
                first_error.classify(),
                serde_json::error::Category::Eof
            ) {
                return Err(first_error);
            }

            let repaired = repair_truncated_json(json_str);
            if repaired == json_str {
                return Err(first_error);
            }

            serde_json::from_str::<serde_json::Value>(&repaired).map_err(|_| first_error)
        }
    }
}

fn repair_truncated_json(input: &str) -> String {
    let mut repaired = input.trim().to_string();
    if repaired.is_empty() {
        return repaired;
    }

    let mut in_string = false;
    let mut escaped = false;
    let mut brace_balance = 0usize;
    let mut bracket_balance = 0usize;

    for ch in repaired.chars() {
        if in_string {
            if escaped {
                escaped = false;
                continue;
            }
            match ch {
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '{' => brace_balance += 1,
            '}' => brace_balance = brace_balance.saturating_sub(1),
            '[' => bracket_balance += 1,
            ']' => bracket_balance = bracket_balance.saturating_sub(1),
            _ => {}
        }
    }

    if in_string {
        repaired.push('"');
    }
    for _ in 0..bracket_balance {
        repaired.push(']');
    }
    for _ in 0..brace_balance {
        repaired.push('}');
    }

    repaired
}


fn append_tool_calls_from_value(
    calls: &mut Vec<ToolCall>,
    value: &serde_json::Value,
    source_tag: &str,
) {
    if let Some(items) = value.get("calls").and_then(|v| v.as_array()) {
        for item in items {
            append_tool_calls_from_value(calls, item, source_tag);
        }
        return;
    }

    if let Some(items) = value.as_array() {
        for item in items {
            append_tool_calls_from_value(calls, item, source_tag);
        }
        return;
    }

    let (name_val, args_val) = if let Some(func) = value.get("function") {
        (
            func.get("name"),
            func.get("arguments").or_else(|| func.get("parameters")),
        )
    } else {
        (
            value.get("name"),
            value.get("arguments").or_else(|| value.get("parameters")),
        )
    };

    if let (Some(name), Some(args)) = (name_val.and_then(|n| n.as_str()), args_val) {
        calls.push(ToolCall {
            name: name.to_string(),
            arguments: args.clone(),
        });
        info!(
            "🔧 [Hera] Parsed tool call (via {}): {} with args: {}",
            source_tag,
            name,
            serde_json::to_string(args).unwrap_or_default()
        );
    }
}

/// Fallback intent detection from the USER's original message.
/// Works with any model size since it doesn't depend on tool_call emission.
/// Returns a ToolCall if the user's intent clearly maps to a tool.
pub fn detect_intent_from_user_message(
    user_msg: &str,
    _assistant_last: Option<&str>,
) -> Option<ToolCall> {
    // Strip injected app context from Imaginclaw (e.g., "[System Context: ...]")
    // so length checks and pattern matching work on the actual user text.
    let clean_msg = if let Some(ctx_start) = user_msg.find("\n\n[System Context:") {
        &user_msg[..ctx_start]
    } else {
        user_msg
    };
    let lower = clean_msg.to_lowercase();
    let lower_trimmed = lower.trim();

    // Strip common greeting prefixes so "hola genera una imagen" matches "genera una imagen"
    let greetings = [
        "hola ",
        "hola, ",
        "oye ",
        "oye, ",
        "hey ",
        "hey, ",
        "hi ",
        "hi, ",
        "hello ",
        "hello, ",
        "buenas ",
        "buenos días ",
        "buenas tardes ",
        "buenas noches ",
        "good morning ",
        "buen día ",
        "please ",
        "por favor ",
        "ey ",
        "ey, ",
        "yo ",
        "sup ",
        "ava ",
        "ava, ",
    ];
    let lower_no_greeting = {
        let mut s = lower_trimmed;
        for g in &greetings {
            if s.starts_with(g) {
                s = s[g.len()..].trim_start();
                break; // Only strip the first greeting
            }
        }
        s
    };

    let detect_app = || -> Option<&'static str> {
        let app = load_canonical_app_registry()
            .into_iter()
            .find_map(|entry| {
                let aliases = alias_terms_for_app(&entry);
                aliases
                    .iter()
                    .any(|alias| lower.contains(alias))
                    .then_some(entry.slug)
            })?;

        match app.as_str() {
            "latinos" => Some("latinos"),
            "vetra" => Some("vetra"),
            "movilo" => Some("movilo"),
            "os-v3" => Some("os-v3"),
            "desktop" => Some("desktop"),
            "paulo-vila-rust" => Some("paulo-vila-rust"),
            "capacita" => Some("capacita"),
            _ => None,
        }
    };

    let command = lower_no_greeting.trim();
    let service_targets = [
        ("imaginclaw", "imaginclaw"),
        ("ava", "imaginclaw"),
        ("vetra", "vetra"),
        ("cartera", "cartera"),
        ("movilo", "movilo"),
        ("sentinel", "sentinel"),
        ("hera", "hera"),
        ("whisper", "whisper"),
        ("audio stt", "audio-stt"),
        ("audio engine", "audio-engine"),
        ("garcero", "garcero"),
        ("latinos", "latinos"),
        ("paulovila", "paulo-vila-rust"),
        ("paulo vila", "paulo-vila-rust"),
        ("os-v3", "os-v3"),
        ("imaginos", "imaginos"),
        ("memento", "memento"),
        ("argus", "argus"),
    ];

    if let Some(rest) = command.strip_prefix("/draw ") {
        let prompt = rest.trim();
        if !prompt.is_empty() {
            info!("🎯 [Hera] Explicit fast-path command: /draw");
            return Some(ToolCall {
                name: "hera_draw".to_string(),
                arguments: serde_json::json!({"prompt": prompt}),
            });
        }
    }

    if let Some(rest) = command.strip_prefix("/search ") {
        let query = rest.trim();
        if !query.is_empty() {
            info!("🎯 [Hera] Explicit fast-path command: /search");
            return Some(ToolCall {
                name: "hera_search".to_string(),
                arguments: serde_json::json!({"query": query}),
            });
        }
    }

    if let Some(rest) = command.strip_prefix("/speak ") {
        let text = rest.trim();
        if !text.is_empty() {
            info!("🎯 [Hera] Explicit fast-path command: /speak");
            return Some(ToolCall {
                name: "hera_speak".to_string(),
                arguments: serde_json::json!({"text": text}),
            });
        }
    }

    if let Some(rest) = command.strip_prefix("/video ") {
        let prompt = rest.trim();
        if !prompt.is_empty() {
            info!("🎯 [Hera] Explicit fast-path command: /video");
            return Some(ToolCall {
                name: "hera_video".to_string(),
                arguments: serde_json::json!({"prompt": prompt}),
            });
        }
    }

    // "Estado OS" / "Estado del sistema" — full PM2 status with restart counts
    if matches!(
        command,
        "/status"
            | "/system-status"
            | "/server-status"
            | "/health-overview"
            | "/machine-status"
            | "estado os"
            | "estado del sistema"
            | "estado del servidor"
            | "system status"
            | "server status"
            | "status of all apps"
            | "review app status"
            | "review all apps status"
    ) {
        info!("🎯 [Hera] Explicit fast-path command: system_status");
        return Some(ToolCall {
            name: "system_status".to_string(),
            arguments: serde_json::json!({}),
        });
    }

    if matches!(command, "/diagnose" | "/diagnose-services") {
        info!("🎯 [Hera] Explicit fast-path command: /diagnose");
        return Some(ToolCall {
            name: "diagnose_services".to_string(),
            arguments: serde_json::json!({}),
        });
    }

    if matches!(command, "/review-apps" | "/apps-status") {
        info!("🎯 [Hera] Explicit fast-path command: /review-apps");
        return Some(ToolCall {
            name: "review_all_apps_status".to_string(),
            arguments: serde_json::json!({
                "timeout_seconds": 10
            }),
        });
    }

    if matches!(command, "/verify-stack" | "/stack-status") {
        info!("🎯 [Hera] Explicit fast-path command: /verify-stack");
        return Some(ToolCall {
            name: "verify_canonical_stack".to_string(),
            arguments: serde_json::json!({
                "checks": ["check"],
                "timeout_seconds": 60
            }),
        });
    }

    if let Some(rest) = command.strip_prefix("/verify-app ") {
        let requested = rest.trim();
        if !requested.is_empty() {
            let matched_app = detect_app().or_else(|| {
                service_targets
                    .iter()
                    .find(|(alias, _)| requested.contains(*alias))
                    .and_then(|(_, slug)| match *slug {
                        "vetra" => Some("vetra"),
                        "cartera" => Some("cartera"),
                        "movilo" => Some("movilo"),
                        "latinos" => Some("latinos"),
                        "os-v3" => Some("os-v3"),
                        "desktop" => Some("desktop"),
                        "paulo-vila-rust" => Some("paulo-vila-rust"),
                        "capacita" => Some("capacita"),
                        _ => None,
                    })
            });

            if let Some(app) = matched_app {
                info!(
                    "🎯 [Hera] Explicit fast-path command: /verify-app for '{}'",
                    app
                );
                return Some(ToolCall {
                    name: "verify_app_health".to_string(),
                    arguments: serde_json::json!({
                        "app": app,
                        "compile_checks": ["check"],
                        "runtime_suite": "regression",
                        "run_runtime": true,
                        "include_logs": true,
                        "timeout_seconds": 60
                    }),
                });
            }
        }
    }

    if let Some(rest) = command.strip_prefix("/restart ") {
        let requested = rest.trim();
        if let Some((alias, slug)) = service_targets
            .iter()
            .find(|(alias, _)| requested.contains(*alias))
        {
            let pm2_name = pm2_process_name_for_slug(slug);
            info!(
                "🎯 [Hera] Explicit fast-path command: /restart for '{}' via alias '{}'",
                pm2_name, alias
            );
            return Some(ToolCall {
                name: "service_restart".to_string(),
                arguments: serde_json::json!({"service_name": pm2_name}),
            });
        }
    }

    None
}

/// Execute a tool call using existing Hera infrastructure.
/// Returns a ToolResult with the output string.
pub async fn execute_tool(call: &ToolCall) -> ToolResult {
    info!("🔧 [Hera] Executing tool: {}", call.name);

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
        "generate_image" | "hera_draw" => platform::execute_draw(call).await,
        "hera_search" => platform::execute_search(call).await,
        "hera_speak" => platform::execute_speak(call).await,
        "hera_video" => platform::execute_video(call).await,
        "hera_read_file" | "read_file" => platform::execute_read_file(call).await,
        "hera_update_soul" | "update_soul" => platform::execute_update_soul(call).await,
        "ask_user" => platform::execute_ask_user(call).await,
        "get_system_time" => platform::execute_get_system_time(call).await,
        "run_code" => platform::execute_run_code(call).await,
        "web_scraper" => platform::execute_web_scraper(call).await,
        "write_file" => platform::execute_write_file(call).await,
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
            if let Some(result) = dispatch_vetra_tool(call).await {
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
        "movilo_check_affiliation" => apps_movilo::execute_movilo_check_affiliation(call).await,
        "movilo_validate_qr" => apps_movilo::execute_movilo_validate_qr(call).await,
        _ => return None,
    };
    Some(result)
}

async fn dispatch_latinos_tool(call: &ToolCall) -> Option<ToolResult> {
    let result = match call.name.as_str() {
        "list_bots" => apps_latinos::execute_list_bots(call).await,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct FunctionTool {
        name: String,
        execution_kind: Option<String>,
        risk_level: Option<String>,
        path: String,
    }

    fn load_function_tools() -> Vec<FunctionTool> {
        let root = Path::new("/home/paulo/Programs/apps/OS/Tools");
        let mut tools = Vec::new();
        collect_function_tools(root, root, &mut tools);
        tools
    }

    fn collect_function_tools(root: &Path, dir: &Path, tools: &mut Vec<FunctionTool>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_function_tools(root, &path, tools);
                continue;
            }
            if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
                continue;
            }

            let Ok(content) = std::fs::read_to_string(&path) else {
                continue;
            };
            let Ok(schema) = serde_json::from_str::<Value>(&content) else {
                continue;
            };
            let Some(name) = schema
                .get("function")
                .and_then(|function| function.get("name"))
                .and_then(|value| value.as_str())
            else {
                continue;
            };

            tools.push(FunctionTool {
                name: name.to_string(),
                execution_kind: schema
                    .get("metadata")
                    .and_then(|value| value.get("execution_kind"))
                    .and_then(|value| value.as_str())
                    .map(ToString::to_string),
                risk_level: schema
                    .get("metadata")
                    .and_then(|value| value.get("risk_level"))
                    .and_then(|value| value.as_str())
                    .map(ToString::to_string),
                path: path
                    .strip_prefix(root)
                    .unwrap_or(&path)
                    .display()
                    .to_string(),
            });
        }
    }

    #[test]
    fn all_registered_function_tools_have_execution_kind() {
        let missing = load_function_tools()
            .into_iter()
            .filter(|tool| tool.execution_kind.is_none())
            .map(|tool| format!("{} ({})", tool.name, tool.path))
            .collect::<Vec<_>>();

        assert!(
            missing.is_empty(),
            "Function tools missing execution_kind: {:?}",
            missing
        );
    }

    #[test]
    fn all_registered_function_tools_have_runtime_dispatch() {
        let missing = load_function_tools()
            .into_iter()
            .filter(|tool| !tool_has_runtime_dispatch(&tool.name, tool.execution_kind.as_deref()))
            .map(|tool| {
                format!(
                    "{} [{}] ({})",
                    tool.name,
                    tool.execution_kind.unwrap_or_else(|| "missing".to_string()),
                    tool.path
                )
            })
            .collect::<Vec<_>>();

        assert!(
            missing.is_empty(),
            "Registered function tools without runtime dispatcher: {:?}",
            missing
        );
    }

    #[test]
    fn registered_tool_registry_matches_function_tools() {
        let tools = load_function_tools();
        let names_from_files = tools
            .iter()
            .map(|tool| tool.name.clone())
            .collect::<HashSet<_>>();
        let registry_names = registered_tool_names().clone();

        assert_eq!(
            registry_names, names_from_files,
            "Registered tool set drifted from Tools/*.json function tools"
        );
    }

    #[test]
    fn raw_json_dispatchers_are_declared_explicitly() {
        let missing = load_function_tools()
            .into_iter()
            .filter(|tool| {
                matches!(
                    tool.name.as_str(),
                    "memento_query"
                        | "market_research"
                        | "analyze_market_research"
                        | "consultant_report_analyzer"
                        | "smoke_apps"
                        | "test_apps"
                        | "verify_canonical_stack"
                        | "review_all_apps_status"
                        | "verify_app_health"
                ) && !tool_has_raw_json_dispatch(&tool.name, tool.execution_kind.as_deref())
            })
            .map(|tool| format!("{} ({})", tool.name, tool.path))
            .collect::<Vec<_>>();

        assert!(
            missing.is_empty(),
            "Tools expected to support raw JSON are missing dispatcher coverage: {:?}",
            missing
        );
    }

    #[test]
    fn critical_tools_declare_risk_metadata() {
        let critical = load_function_tools()
            .into_iter()
            .filter(|tool| {
                matches!(
                    tool.name.as_str(),
                    "api_request"
                        | "run_code"
                        | "write_file"
                        | "service_restart"
                        | "desktop_click"
                        | "desktop_type"
                ) && tool.risk_level.as_deref() != Some("critical")
            })
            .map(|tool| format!("{} ({})", tool.name, tool.path))
            .collect::<Vec<_>>();

        assert!(
            critical.is_empty(),
            "Critical tools missing explicit risk metadata: {:?}",
            critical
        );
    }

    #[test]
    fn explicit_restart_command_maps_to_service_restart() {
        let call = detect_intent_from_user_message("/restart imaginclaw", None)
            .expect("expected explicit /restart command to map");

        assert_eq!(call.name, "service_restart");
        assert_eq!(
            call.arguments
                .get("service_name")
                .and_then(|value| value.as_str()),
            Some("imaginclaw")
        );
    }

    #[test]
    fn explicit_status_command_maps_to_system_status() {
        let call = detect_intent_from_user_message("/status", None)
            .expect("expected explicit /status command to map");

        assert_eq!(call.name, "system_status");
    }

    #[test]
    fn parses_truncated_tool_call_json_by_closing_braces() {
        let text = r#"<tool_call>{"name":"memento_query","arguments":{"app":"cartera","query":"SELECT 1"}</tool_call>"#;

        let calls = parse_tool_calls(text);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "memento_query");
        assert_eq!(
            calls[0]
                .arguments
                .get("app")
                .and_then(|value| value.as_str()),
            Some("cartera")
        );
    }

}
