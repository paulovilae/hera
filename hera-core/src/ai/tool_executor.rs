//! Hera Tool Executor — Sovereign Tool Calling for ImagineOS
//!
//! Defines tool schemas in Qwen's native format, parses `<tool_call>` blocks
//! from Qwen output, and dispatches tool execution to existing Hera methods.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::info;
use crate::ai::tools::{
    apps_latinos, apps_movilo, apps_vetra, data, infra_health, infra_smoke, platform,
};


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

#[derive(Debug, Clone)]
struct ToolArtifact {
    schema: Value,
    consumers: Vec<String>,
}

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

pub(crate) fn pm2_process_name_for_slug(slug: &str) -> &str {
    match slug {
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
        "diakonos" => "diakonos",
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
                && let Some(skill) = parse_skill_artifact(&path) {
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
    AgentArtifact {
        persona,
    }
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

                match serde_json::from_str::<serde_json::Value>(json_str) {
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

            if end_idx > 0 {
                let json_str = &stripped[..end_idx];
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(json_str)
                    && let (Some(name), Some(args)) = (
                        val.get("name").and_then(|n| n.as_str()),
                        val.get("arguments"),
                    ) {
                        calls.push(ToolCall {
                            name: name.to_string(),
                            arguments: args.clone(),
                        });
                        info!(
                            "🔧 [Hera] Parsed RAW JSON tool call (after think-strip): {} with args: {}",
                            name,
                            serde_json::to_string(args).unwrap_or_default()
                        );
                    }
            }
        }
    }

    calls
}

fn append_tool_calls_from_value(calls: &mut Vec<ToolCall>, value: &serde_json::Value, source_tag: &str) {
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

    if let (Some(name), Some(args)) = (
        value.get("name").and_then(|n| n.as_str()),
        value.get("arguments"),
    ) {
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
    assistant_last: Option<&str>,
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

    let mentions_verification = [
        "verify",
        "verification",
        "health",
        "health check",
        "app health",
        "check app",
        "test ",
        "tests",
        "smoke",
        "regression",
        "diagnose only if",
        "if something fails",
        "if it fails",
        "only if something fails",
        "only if it fails",
    ];
    if let Some(app) = detect_app() {
        let asks_for_verification = mentions_verification.iter().any(|kw| lower.contains(kw))
            || lower_no_greeting.starts_with("test ")
            || lower_no_greeting.starts_with("verify ")
            || lower_no_greeting.starts_with("run regression")
            || lower_no_greeting.starts_with("run smoke");
        if asks_for_verification {
            let runtime_suite = if lower.contains("smoke") {
                "smoke"
            } else {
                "regression"
            };
            let run_runtime = !lower.contains("compile only") && !lower.contains("build only");
            let include_logs = !lower.contains("without logs");
            info!("🎯 [Hera] Intent detected: verify_app_health for '{}'", app);
            return Some(ToolCall {
                name: "verify_app_health".to_string(),
                arguments: serde_json::json!({
                    "app": app,
                    "compile_checks": ["check"],
                    "runtime_suite": runtime_suite,
                    "run_runtime": run_runtime,
                    "include_logs": include_logs,
                    "timeout_seconds": 60
                }),
            });
        }
    }

    let asks_for_all_app_status =
        (lower.contains("all apps") || lower.contains("all app") || lower.contains("all services"))
            && (lower.contains("status") || lower.contains("review") || lower.contains("health"));
    if asks_for_all_app_status
        || lower_no_greeting == "review teh status of all apps"
        || lower_no_greeting == "review the status of all apps"
    {
        info!("🎯 [Hera] Intent detected: review_all_apps_status from user message");
        return Some(ToolCall {
            name: "review_all_apps_status".to_string(),
            arguments: serde_json::json!({
                "timeout_seconds": 10
            }),
        });
    }

    let asks_for_stack_gate = (lower.contains("canonical stack")
        || lower.contains("release gate")
        || lower.contains("release ready"))
        || (lower.contains("verify") && lower.contains("stack"))
        || lower_no_greeting == "verify canonical stack";
    if asks_for_stack_gate {
        info!("🎯 [Hera] Intent detected: verify_canonical_stack from user message");
        return Some(ToolCall {
            name: "verify_canonical_stack".to_string(),
            arguments: serde_json::json!({
                "checks": ["check"],
                "timeout_seconds": 60
            }),
        });
    }

    // Contextual image modifier detection
    if let Some(ast) = assistant_last
        && (ast.contains("MEDIA:")
            || ast.contains("Aquí tienes")
            || ast.contains("Here is")
            || ast.contains("la imagen"))
        {
            let is_modifier = lower.starts_with("ahora ")
                || lower.starts_with("now ")
                || lower.starts_with("con ")
                || lower.starts_with("with ")
                || lower.starts_with("sin ")
                || lower.starts_with("without ")
                || lower.starts_with("mas ")
                || lower.starts_with("more ");

            if is_modifier {
                tracing::info!(
                    "🎯 [Hera] Intent detected: hera_draw from conversational context (modifier)"
                );
                return Some(ToolCall {
                    name: "hera_draw".to_string(),
                    arguments: serde_json::json!({"prompt": user_msg}),
                });
            }
        }

    // Draw/Image intent — Strict matching to prevent hijacking normal conversation
    let exact_starts = [
        "draw ",
        "dibuja ",
        "genera una imagen ",
        "create an image ",
        "make an image ",
        "generate an image ",
        "draw me ",
        "hazme un dibujo",
        "pinta ",
        "haz una imagen",
        "genera imagen",
        "crea una imagen",
        "make a picture",
        "generate a picture",
        "create a picture",
        "haz un dibujo ",
        "make me an image ",
        "draw a ",
        "hazme una foto ",
        "toma una foto ",
        "manda una foto ",
        "hazme una imagen ",
        "genera una foto ",
        "crea una foto ",
        "take a photo ",
        "send a photo ",
        "a picture of ",
        "make me a picture ",
        "send me an image ",
        "show me an image ",
        "haz una foto ",
        "a photo of ",
        "make a photo ",
        "create a photo ",
        "foto de ",
        "do a photo ",
        "do a picture ",
        "do a foto ",
        "do an image ",
    ];

    // Short exact matches
    let exact_matches = [
        "tu foto",
        "una foto",
        "mi foto",
        "dame foto",
        "dame una foto",
        "una imagen",
        "mi imagen",
        "tu imagen",
        "your photo",
        "my photo",
        "selfie",
        "retrato",
    ];

    // Broad fuzzy detection: if a short message contains an image noun + an action verb, it's a draw request
    let image_nouns = [
        "photo", "foto", "picture", "imagen", "image", "drawing", "dibujo", "selfie", "retrato",
        "pic ", "pic.",
    ];
    let action_verbs = [
        "make", "do ", "create", "take", "send", "show", "generate", "haz", "genera", "crea",
        "toma", "manda", "dame", "hazme", "draw", "paint", "pinta", "dibuja", "quiero", "want",
    ];

    let mut is_draw = false;

    if exact_starts
        .iter()
        .any(|kw| lower_trimmed.starts_with(kw) || lower_no_greeting.starts_with(kw))
    {
        is_draw = true;
    } else if clean_msg.len() < 40
        && exact_matches.iter().any(|kw| {
            lower_trimmed == *kw
                || lower_trimmed.starts_with(kw)
                || lower_no_greeting == *kw
                || lower_no_greeting.starts_with(kw)
        })
    {
        is_draw = true;
    } else if clean_msg.len() < 80 {
        // Fuzzy: short message contains both an image noun and an action verb
        let has_noun = image_nouns.iter().any(|n| lower_trimmed.contains(n));
        let has_verb = action_verbs.iter().any(|v| lower_trimmed.contains(v));
        if has_noun && has_verb {
            is_draw = true;
        }
    }

    if is_draw {
        let prompt = clean_msg.to_string();
        tracing::info!("🎯 [Hera] Strict intent detected: hera_draw from user message");
        return Some(ToolCall {
            name: "hera_draw".to_string(),
            arguments: serde_json::json!({"prompt": prompt}),
        });
    }

    // Search intent
    let search_keywords = [
        "busca ",
        "search ",
        "look up ",
        "google ",
        "find out ",
        "busca en internet",
        "search the web",
        "qué pasó con",
        "what happened with",
        "noticias de",
        "news about",
    ];
    if search_keywords.iter().any(|kw| lower.contains(kw)) {
        info!("🎯 [Hera] Intent detected: hera_search from user message");
        return Some(ToolCall {
            name: "hera_search".to_string(),
            arguments: serde_json::json!({"query": clean_msg}),
        });
    }

    // Speak intent
    let speak_keywords = [
        "say out loud",
        "di en voz alta",
        "habla ",
        "speak ",
        "read aloud",
        "lee en voz alta",
        "genera audio",
    ];
    if speak_keywords.iter().any(|kw| lower.contains(kw)) {
        info!("🎯 [Hera] Intent detected: hera_speak from user message");
        return Some(ToolCall {
            name: "hera_speak".to_string(),
            arguments: serde_json::json!({"text": clean_msg}),
        });
    }

    // Video intent
    let video_keywords = [
        "genera un video",
        "generate a video",
        "make a video",
        "create a video",
        "haz un video",
        "crea un video",
    ];
    if video_keywords.iter().any(|kw| lower.contains(kw)) {
        info!("🎯 [Hera] Intent detected: hera_video from user message");
        return Some(ToolCall {
            name: "hera_video".to_string(),
            arguments: serde_json::json!({"prompt": clean_msg}),
        });
    }

    // Service restart/fix intent — catch repair requests
    let restart_patterns = [
        "restart ",
        "reinicia ",
        "restartea ",
        "fix ",
        "arregla ",
        "repair ",
        "repara ",
        "levanta ",
        "bring up ",
        "start up ",
        "arranca ",
        "prende ",
        "reiniciar ",
        "reboot ",
    ];
    let service_targets = [
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
        ("diakonos", "diakonos"),
        ("argus", "argus"),
    ];
    for pattern in &restart_patterns {
        if lower.starts_with(pattern)
            || lower_no_greeting.starts_with(pattern)
            || lower.contains(pattern)
        {
            // Try to extract the service name from the message
            if let Some((alias, slug)) = service_targets
                .iter()
                .find(|(alias, _)| lower.contains(*alias))
            {
                let pm2_name = pm2_process_name_for_slug(slug);
                info!(
                    "🎯 [Hera] Intent detected: service_restart for '{}' via alias '{}'",
                    pm2_name, alias
                );
                return Some(ToolCall {
                    name: "service_restart".to_string(),
                    arguments: serde_json::json!({"service_name": pm2_name}),
                });
            }
        }
    }

    // Service diagnostics intent — catch questions about broken services / why something doesn't work
    let diag_keywords = [
        "diagnose",
        "diagnostica",
        "qué está mal",
        "que esta mal",
        "what's wrong",
        "whats wrong",
        "por qué no funciona",
        "porque no funciona",
        "why isn't it working",
        "why isnt it working",
        "check services",
        "revisa los servicios",
        "health check",
        "chequeo de salud",
        "servicios caídos",
        "servicios caidos",
        "services down",
        "qué pasó con",
        "que paso con",
        "500 error",
        "no responde",
        "not responding",
        "está caído",
        "esta caido",
        "is down",
        "se cayó",
        "se cayo",
        "no carga",
        "won't load",
        "diagnóstico",
        "diagnostico",
        "qué está pasando",
        "que esta pasando",
        "what's happening",
        "whats happening",
    ];
    if diag_keywords.iter().any(|kw| lower.contains(kw)) {
        info!("🎯 [Hera] Intent detected: diagnose_services from user message");
        return Some(ToolCall {
            name: "diagnose_services".to_string(),
            arguments: serde_json::json!({}),
        });
    }

    // System status intent — catch conversational questions about server/GPU/CPU/memory
    let status_keywords = [
        "system status",
        "server status",
        "gpu status",
        "cpu status",
        "how is the server",
        "como esta el server",
        "como está el server",
        "estado del servidor",
        "estado del server",
        "status del server",
        "nvidia",
        "vram",
        "gpu load",
        "gpu temp",
        "como esta el gpu",
        "como está el gpu",
        "cuanta ram",
        "cuánta ram",
        "how much ram",
        "memory usage",
        "uso de memoria",
        "system health",
        "server health",
        "how much vram",
        "cuanta vram",
        "cuánta vram",
        "que procesos",
        "qué procesos",
        "what processes",
        "esta corriendo",
        "está corriendo",
        "is running",
    ];
    if status_keywords.iter().any(|kw| lower.contains(kw)) {
        info!("🎯 [Hera] Intent detected: system_status from user message");
        return Some(ToolCall {
            name: "system_status".to_string(),
            arguments: serde_json::json!({}),
        });
    }

    None
}

/// Execute a tool call using existing Hera infrastructure.
/// Returns a ToolResult with the output string.
pub async fn execute_tool(call: &ToolCall) -> ToolResult {
    info!("🔧 [Hera] Executing tool: {}", call.name);

    // Global timeout: 90 seconds. No tool should take longer.
    let tool_name = call.name.clone();
    match tokio::time::timeout(
        std::time::Duration::from_secs(90),
        execute_tool_inner(call),
    )
    .await
    {
        Ok(result) => result,
        Err(_) => {
            tracing::error!(
                "⏰ [Hera] Tool '{}' TIMED OUT after 90s. Returning error.",
                tool_name
            );
            ToolResult {
                name: tool_name,
                success: false,
                output: "Error: Tool execution timed out after 90 seconds.".to_string(),
            }
        }
    }
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

    match call.name.as_str() {
        "hera_draw" => platform::execute_draw(call).await,
        "hera_search" => platform::execute_search(call).await,
        "hera_speak" => platform::execute_speak(call).await,
        "hera_video" => platform::execute_video(call).await,
        "hera_read_file" | "read_file" => platform::execute_read_file(call).await,
        "hera_update_soul" | "update_soul" => platform::execute_update_soul(call).await,
        "memento_query" => data::execute_memento_query(call).await,
        "api_request" => data::execute_api_request(call).await,
        "git_manager" => data::execute_git_manager(call).await,
        "memento_vector_search" => data::execute_memento_vector_search(call).await,
        "ask_user" => platform::execute_ask_user(call).await,
        "get_system_time" => platform::execute_get_system_time(call).await,
        "system_status" => infra_health::execute_system_status(call).await,
        "run_code" => platform::execute_run_code(call).await,
        "web_scraper" => platform::execute_web_scraper(call).await,
        "write_file" => platform::execute_write_file(call).await,
        "generate_qr_code" => apps_vetra::execute_generate_qr_code(call).await,
        "generate_contract_pdf" => apps_vetra::execute_generate_contract_pdf(call).await,
        "dispatch_email" => apps_vetra::execute_dispatch_email(call).await,
        "get_map_route" => apps_vetra::execute_get_map_route(call).await,
        "execute_workflow" => apps_vetra::execute_workflow(call).await,
        "movilo_search_providers" => apps_movilo::execute_movilo_search_providers(call).await,
        "movilo_check_affiliation" => apps_movilo::execute_movilo_check_affiliation(call).await,
        "movilo_validate_qr" => apps_movilo::execute_movilo_validate_qr(call).await,
        "bind_telegram_workspace" => apps_vetra::execute_bind_telegram_workspace(call).await,
        "spline_interact" => platform::execute_spline_interact(call).await,
        "desktop_click" => platform::execute_desktop_click(call).await,
        "desktop_type" => platform::execute_desktop_type(call).await,
        "read_os_logs" => infra_smoke::execute_read_os_logs(call).await,
        "diagnose_services" => infra_health::execute_diagnose_services(call).await,
        "service_restart" => infra_health::execute_service_restart(call).await,
        "read_pm2_logs" => infra_health::execute_read_pm2_logs(call).await,
        "smoke_apps" => infra_smoke::execute_smoke_apps(call).await,
        "test_apps" => infra_smoke::execute_test_apps(call).await,
        "verify_canonical_stack" => infra_smoke::execute_verify_canonical_stack(call).await,
        "review_all_apps_status" => infra_smoke::execute_review_all_apps_status(call).await,
        "verify_app_health" => infra_smoke::execute_verify_app_health(call).await,
        "market_research" => apps_latinos::execute_market_research(call).await,
        // ── Global: Market & Company Analysis (consulting-grade) ─────────
        "analyze_market_research" => apps_latinos::execute_market_research(call).await,
        "consultant_report_analyzer" => apps_latinos::execute_consultant_report_analyzer(call).await,
        // ── Latinos Quant Lab tools ──────────────────────────────────────
        "run_backtest" => apps_latinos::execute_latinos_bridge(call, "run_backtest").await,
        "load_market_data" => apps_latinos::execute_latinos_bridge(call, "load_market_data").await,
        "scan_opportunities" => apps_latinos::execute_latinos_bridge(call, "scan_opportunities").await,
        _ => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Unknown tool: {}", call.name),
        },
    }
}

pub async fn execute_tool_raw_json(call: &ToolCall) -> Result<Value, String> {
    // Global timeout: 90 seconds for raw JSON tool execution.
    let tool_name = call.name.clone();
    match tokio::time::timeout(
        std::time::Duration::from_secs(90),
        execute_tool_raw_json_inner(call),
    )
    .await
    {
        Ok(result) => result,
        Err(_) => {
            tracing::error!(
                "⏰ [Hera] Tool '{}' (raw_json) TIMED OUT after 90s.",
                tool_name
            );
            Err(format!(
                "Tool '{}' timed out after 90 seconds.",
                tool_name
            ))
        }
    }
}

/// Inner dispatch for raw JSON tools — called inside the 90s timeout wrapper.
async fn execute_tool_raw_json_inner(call: &ToolCall) -> Result<Value, String> {
    match call.name.as_str() {
        "memento_query" => data::execute_memento_query_json(call).await,
        "market_research" => apps_latinos::execute_market_research_json(call).await,
        "analyze_market_research" => apps_latinos::execute_market_research_json(call).await,
        "consultant_report_analyzer" => apps_latinos::execute_consultant_report_analyzer_json(call).await,
        "smoke_apps" => infra_smoke::execute_smoke_apps_json(call).await,
        "test_apps" => infra_smoke::execute_test_apps_json(call).await,
        "verify_canonical_stack" => infra_smoke::execute_verify_canonical_stack_json(call).await,
        "review_all_apps_status" => infra_smoke::execute_review_all_apps_status_json(call).await,
        "verify_app_health" => infra_smoke::execute_verify_app_health_json(call).await,
        _ => {
            let result = execute_tool_inner(call).await;
            if result.success {
                Ok(serde_json::json!({
                    "name": result.name,
                    "output": result.output,
                }))
            } else {
                Err(result.output)
            }
        }
    }
}
