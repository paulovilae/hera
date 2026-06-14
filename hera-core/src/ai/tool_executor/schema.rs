//! Tool schema parsing and skill artifact management

use std::path::{Path, PathBuf};
use serde_json::Value;

#[derive(Debug, Clone)]
struct ToolArtifact {
    schema: Value,
    consumers: Vec<String>,
    /// `metadata.status == "skeleton_not_implemented"` — WIP tools that are
    /// never offered to the model (and are exempt from the dispatch test).
    is_skeleton: bool,
}

#[derive(Debug, Clone)]
pub struct SkillArtifact {
    pub skill_id: String,
    pub tool_name: String,
    pub description: String,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct AgentArtifact {
    pub persona: String,
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

    let is_skeleton = schema
        .get("metadata")
        .and_then(|metadata| metadata.get("status"))
        .and_then(|value| value.as_str())
        == Some("skeleton_not_implemented");

    if let Some(obj) = schema.as_object_mut() {
        obj.remove("metadata");
    }

    Some(ToolArtifact {
        schema,
        consumers,
        is_skeleton,
    })
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
                Some(artifact) if artifact.is_skeleton => {
                    tracing::debug!(
                        "Skipping skeleton_not_implemented tool: {:?}",
                        entry_path
                    );
                }
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

pub fn find_skill_artifact(tool_name: &str) -> Option<SkillArtifact> {
    collect_skill_artifacts()
        .into_iter()
        .find(|skill| skill.tool_name == tool_name)
}

pub fn load_agent_artifact(agent_name: &str) -> AgentArtifact {
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
            | "geocode"
            | "reverse_geocode"
            | "storage_list"
            | "storage_get_url"
            | "storage_put"
            | "read_email"
            | "read_notes"
            | "list_calendar_events"
            | "mc_board"
            | "mc_create_story"
            | "mc_move_story"
            | "mc_create_sprint"
            | "mc_close_sprint"
            | "mc_add_wishlist"
            | "mc_set_objective"
    )
}

#[cfg(test)]
fn is_data_tool_name(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "memento_query"
            | "api_request"
            | "git_manager"
            | "memento_vector_search"
            | "save_memory"
            | "query_memory"
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
            | "query_federation_state"
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
fn is_brand_tool_name(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "add_topic"
            | "list_pending_drafts"
            | "approve_draft"
            | "capture_post_metrics"
            | "voice_profile_get"
            | "voice_profile_update"
            | "save_thesis_doc"
    )
}

#[cfg(test)]
fn is_movilo_tool_name(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "movilo_search_providers"
            | "movilo_get_plans"
            | "movilo_check_affiliation"
            | "movilo_validate_qr"
    )
}

#[cfg(test)]
fn is_latinos_tool_name(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "list_bots"
            | "list_markets"
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
pub(crate) fn tool_has_runtime_dispatch(tool_name: &str, execution_kind: Option<&str>) -> bool {
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
            is_brand_tool_name(tool_name)
                || is_vetra_tool_name(tool_name)
                || is_data_tool_name(tool_name)
                || is_platform_tool_name(tool_name)
        }
        _ => false,
    }
}

#[cfg(test)]
pub(crate) fn tool_has_raw_json_dispatch(tool_name: &str, execution_kind: Option<&str>) -> bool {
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
