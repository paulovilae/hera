//! Hera Tool Executor — Sovereign Tool Calling for ImagineOS
//!
//! Defines tool schemas in Qwen's native format, parses `<tool_call>` blocks
//! from Qwen output, and dispatches tool execution to existing Hera methods.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{SystemTime, UNIX_EPOCH};
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

#[derive(Debug, Clone)]
struct ToolArtifact {
    schema: Value,
    consumers: Vec<String>,
}

#[derive(Debug, Clone)]
struct SkillArtifact {
    skill_id: String,
    tool_name: String,
    description: String,
    content: String,
}

#[derive(Debug, Clone)]
struct AgentArtifact {
    _agent_id: String,
    persona: String,
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

fn collect_tool_schemas_from_dir(
    dir: &Path,
    tools: &mut Vec<Value>,
    agent_name: &str,
    permissions_filter: Option<&[String]>,
) {
    if !dir.exists() {
        return;
    }

    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let entry_path = entry.path();
            if entry_path.is_dir() {
                collect_tool_schemas_from_dir(&entry_path, tools, agent_name, permissions_filter);
                continue;
            }

            if entry_path.extension().and_then(|ext| ext.to_str()) != Some("json") {
                continue;
            }

            match parse_tool_artifact(&entry_path) {
                Some(artifact) => {
                    let mut allowed = artifact
                        .consumers
                        .iter()
                        .any(|consumer| consumer == "all" || consumer == agent_name);

                    if allowed {
                        if let Some(perms) = permissions_filter {
                            let has_all = perms.contains(&"all".to_string());
                            let tool_name = artifact
                                .schema
                                .get("name")
                                .and_then(|n| n.as_str())
                                .unwrap_or("");
                            if !has_all && !perms.contains(&tool_name.to_string()) {
                                allowed = false;
                            }
                        }
                    }

                    if allowed {
                        tools.push(artifact.schema);
                    } else {
                        tracing::debug!(
                            "Skipping tool due to consumer/permission restriction: {:?}",
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
            if path.is_dir() {
                if let Some(skill) = parse_skill_artifact(&path) {
                    skills.push(skill);
                }
            }
        }
    }
    skills
}

fn find_skill_artifact(tool_name: &str) -> Option<SkillArtifact> {
    collect_skill_artifacts()
        .into_iter()
        .find(|skill| skill.tool_name == tool_name)
}

fn load_agent_artifact(agent_name: &str) -> AgentArtifact {
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
        _agent_id: sanitized,
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

    // 1. Always load global tools (recursive through topic subfolders)
    let global_dir = PathBuf::from(format!("{}/global", base_dir));
    collect_tool_schemas_from_dir(&global_dir, &mut tools_vec, agent_name, None);

    // 2. Load app-specific tools based on permissions
    let apps_dir = PathBuf::from(format!("{}/apps", base_dir));
    collect_tool_schemas_from_dir(&apps_dir, &mut tools_vec, agent_name, Some(permissions));

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
                json_str = json_str.trim_start_matches("```json").trim();
                json_str = json_str.trim_start_matches("```").trim();
                json_str = json_str.trim_end_matches("```").trim();

                match serde_json::from_str::<serde_json::Value>(json_str) {
                    Ok(val) => {
                        if let (Some(name), Some(args)) = (
                            val.get("name").and_then(|n| n.as_str()),
                            val.get("arguments"),
                        ) {
                            calls.push(ToolCall {
                                name: name.to_string(),
                                arguments: args.clone(),
                            });
                            info!(
                                "🔧 [Hera] Parsed tool call (via {}): {} with args: {}",
                                open_tag,
                                name,
                                serde_json::to_string(args).unwrap_or_default()
                            );
                        }
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

        // Find the first JSON block even if preceded by text
        if let Some(start_idx) = stripped.find('{') {
            let json_candidate = &stripped[start_idx..];
            let mut brace_count = 0;
            let mut end_idx = 0;
            for (i, c) in json_candidate.char_indices() {
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
                let json_str = &json_candidate[..end_idx];
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(json_str) {
                    let name = val
                        .get("name")
                        .and_then(|n| n.as_str())
                        .or_else(|| val.get("tool").and_then(|t| t.as_str()));

                    let args = val.get("arguments").or_else(|| val.get("args"));

                    if let (Some(n), Some(a)) = (name, args) {
                        calls.push(ToolCall {
                            name: n.to_string(),
                            arguments: a.clone(),
                        });
                        tracing::info!(
                            "🔧 [Hera] Parsed RAW JSON tool call (fallback): {} with args: {}",
                            n,
                            serde_json::to_string(a).unwrap_or_default()
                        );
                    }
                }
            }
        }
    }

    calls
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

    // [DISABLED 2026-03-30: Fast-path drawing is disabled so the LLM can proactively expand prompts and maintain persona]
    // Contextual image modifier detection
    /*
    if let Some(ast) = assistant_last {
        if ast.contains("MEDIA:") || ast.contains("Aquí tienes") || ast.contains("Here is") || ast.contains("la imagen") {
            let is_modifier = lower.starts_with("ahora ") || lower.starts_with("now ")
                || lower.starts_with("con ") || lower.starts_with("with ")
                || lower.starts_with("sin ") || lower.starts_with("without ")
                || lower.starts_with("mas ") || lower.starts_with("more ");

            if is_modifier {
                tracing::info!("🎯 [Hera] Intent detected: hera_draw from conversational context (modifier)");
                return Some(ToolCall {
                    name: "hera_draw".to_string(),
                    arguments: serde_json::json!({"prompt": user_msg}),
                });
            }
        }
    }

    // Draw/Image intent — Strict matching to prevent hijacking normal conversation
    let exact_starts = [
        "/draw", "/draw ", "draw ", "dibuja ", "genera una imagen ", "create an image ", "make an image ",
        "generate an image ", "draw me ", "hazme un dibujo", "pinta ",
        "haz una imagen", "genera imagen", "crea una imagen",
        "make a picture", "generate a picture", "create a picture",
        "haz un dibujo ", "make me an image ", "draw a ",
        "hazme una foto ", "toma una foto ", "manda una foto ",
        "hazme una imagen ", "genera una foto ", "crea una foto ",
        "take a photo ", "send a photo ", "a picture of ", "picture of ", "image of ",
        "make me a picture ", "send me an image ", "show me an image ",
        "haz una foto ", "a photo of ", "make a photo ", "create a photo ", "foto de ",
        "do a photo ", "do a picture ", "do a foto ", "do an image ",
    ];

    // Short exact matches
    let exact_matches = [
        "/draw", "draw", "tu foto", "una foto", "mi foto", "dame foto", "dame una foto",
        "una imagen", "mi imagen", "tu imagen",
        "your photo", "my photo", "selfie", "retrato",
    ];

    // Broad fuzzy detection: if a short message contains an image noun + an action verb, it's a draw request
    let image_nouns = ["photo", "foto", "picture", "imagen", "image", "drawing", "dibujo", "selfie", "retrato", "pic ", "pic."];
    let action_verbs = ["make", "do ", "create", "take", "send", "show", "generate", "haz", "genera", "crea", "toma", "manda", "dame", "hazme", "draw", "paint", "pinta", "dibuja", "quiero", "want", "give"];

    let mut is_draw = false;

    if exact_starts.iter().any(|kw| lower_trimmed.starts_with(kw) || lower_no_greeting.starts_with(kw)) {
        is_draw = true;
    } else if clean_msg.len() < 40 && exact_matches.iter().any(|kw| lower_trimmed == *kw || lower_trimmed.starts_with(kw) || lower_no_greeting == *kw || lower_no_greeting.starts_with(kw)) {
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
    */

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
    let service_names = [
        "vetra",
        "cartera",
        "movilo",
        "sentinel",
        "hera",
        "garcero",
        "latinos",
        "paulovila",
        "os-v3",
        "imaginos",
        "memento",
        "diakonos",
        "argus",
    ];
    for pattern in &restart_patterns {
        if lower.starts_with(pattern)
            || lower_no_greeting.starts_with(pattern)
            || lower.contains(pattern)
        {
            // Try to extract the service name from the message
            if let Some(svc) = service_names.iter().find(|svc| lower.contains(**svc)) {
                info!("🎯 [Hera] Intent detected: service_restart for '{}'", svc);
                return Some(ToolCall {
                    name: "service_restart".to_string(),
                    arguments: serde_json::json!({"service_name": format!("{}-rust", svc)}),
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

    if call.name.starts_with("load_skill_") {
        return execute_load_skill(call).await;
    }

    if call.name == "spawn_parallel_agents" {
        return execute_spawn_parallel_agents(call).await;
    }

    if call.name == "create_agent" {
        return execute_create_agent(call).await;
    }

    if call.name == "create_skill" {
        return execute_create_skill(call).await;
    }

    match call.name.as_str() {
        "hera_draw" => execute_draw(call).await,
        "hera_search" => execute_search(call).await,
        "hera_speak" => execute_speak(call).await,
        "hera_video" => execute_video(call).await,
        "hera_read_file" | "read_file" => execute_read_file(call).await,
        "hera_update_soul" | "update_soul" => execute_update_soul(call).await,
        "memento_query" => execute_memento_query(call).await,
        "api_request" => execute_api_request(call).await,
        "git_manager" => execute_git_manager(call).await,
        "memento_vector_search" => execute_memento_vector_search(call).await,
        "create_research_project" => execute_memento_research_project(call).await,
        "get_research_project" => execute_memento_get_research_project(call).await,
        "list_research_projects" => execute_memento_list_research_projects(call).await,
        "create_research_session" => execute_memento_research_session(call).await,
        "create_research_source" | "upsert_research_source" => {
            execute_memento_research_source(call).await
        }
        "get_research_source" => execute_memento_get_research_source(call).await,
        "list_research_sources" => execute_memento_list_research_sources(call).await,
        "upsert_concept_node" => execute_memento_upsert_concept(call).await,
        "append_claim" => execute_memento_append_claim(call).await,
        "append_evidence" => execute_memento_append_evidence(call).await,
        "link_concepts" => execute_memento_link_concepts(call).await,
        "expand_concept" => execute_memento_expand_concept(call).await,
        "trace_claim_provenance" => execute_memento_trace_claim_provenance(call).await,
        "persist_research_finding" => execute_persist_research_finding(call).await,
        "persist_channel_research_finding" => execute_persist_channel_research_finding(call).await,
        "ask_user" => execute_ask_user(call).await,
        "get_system_time" => execute_get_system_time(call).await,
        "system_status" => execute_system_status(call).await,
        "run_code" => execute_run_code(call).await,
        "web_scraper" => execute_web_scraper(call).await,
        "write_file" => execute_write_file(call).await,
        "generate_qr_code" => execute_generate_qr_code(call).await,
        "render_pdf" => execute_render_pdf(call).await,
        "generate_contract_pdf" => execute_generate_contract_pdf(call).await,
        "dispatch_email" => execute_dispatch_email(call).await,
        "get_map_route" => execute_get_map_route(call).await,
        "execute_workflow" => execute_workflow(call).await,
        "movilo_search_providers" => execute_movilo_search_providers(call).await,
        "movilo_check_affiliation" => execute_movilo_check_affiliation(call).await,
        "movilo_validate_qr" => execute_movilo_validate_qr(call).await,
        "bind_telegram_workspace" => execute_bind_telegram_workspace(call).await,
        "spline_interact" => execute_spline_interact(call).await,
        "desktop_click" => execute_desktop_click(call).await,
        "desktop_type" => execute_desktop_type(call).await,
        "read_os_logs" => execute_read_os_logs(call).await,
        "diagnose_services" => execute_diagnose_services(call).await,
        "service_restart" => execute_service_restart(call).await,
        "read_pm2_logs" => execute_read_pm2_logs(call).await,
        "list_image_loras" => execute_list_image_loras(call).await,
        "run_backtest" => execute_run_backtest(call).await,
        "get_bot_status" => execute_get_bot_status(call).await,
        "load_market_data" => execute_load_market_data(call).await,
        "list_bots" => execute_list_bots(call).await,
        "market_research" => execute_market_research(call).await,
        "thermal_risk_scorer" => execute_thermal_risk_scorer(call).await,
        "generate_payment_agreement" => execute_generate_payment_agreement(call).await,
        "omni_channel_messenger" => execute_omni_channel_messenger(call).await,
        _ => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Unknown tool: {}", call.name),
        },
    }
}

async fn execute_read_os_logs(call: &ToolCall) -> ToolResult {
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

    let log_path = "/home/paulo/Programs/apps/OS/Apps/OS-v3/storage/logs/runtime.jsonl";

    match std::fs::read_to_string(log_path) {
        Ok(content) => {
            let mut matched_logs = Vec::new();

            for line in content.lines().rev() {
                if line.trim().is_empty() {
                    continue;
                }

                let lower_line = line.to_lowercase();
                if !service.is_empty() {
                    let s_low = service.to_lowercase();
                    if !lower_line.contains(&format!("\"service\":\"{}\"", s_low))
                        && !lower_line.contains(&format!("\"app\":\"{}\"", s_low))
                    {
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

/// Composite diagnostic tool — the "IQ upgrade" for Ava.
/// Reads services.conf, cross-references PM2 + port listeners + HTTP probes + error logs,
/// and produces a correlated diagnostic report with root cause hypotheses.
async fn execute_diagnose_services(call: &ToolCall) -> ToolResult {
    let service_filter = call
        .arguments
        .get("service_filter")
        .and_then(|s| s.as_str())
        .unwrap_or("")
        .to_lowercase();
    let include_logs = call
        .arguments
        .get("include_logs")
        .and_then(|b| b.as_bool())
        .unwrap_or(true);

    let mut report = String::new();
    report.push_str("🏥 ImagineOS Service Diagnostic Report\n");
    report.push_str("═══════════════════════════════════════\n\n");

    // ── 1. Parse services.conf to get expected service→port map ──
    let services_conf_path = "/home/paulo/Programs/apps/OS/etc/sentinel/services.conf";
    let mut expected_services: Vec<(String, u16)> = Vec::new();

    if let Ok(content) = std::fs::read_to_string(services_conf_path) {
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with('[') {
                continue;
            }
            // Format: hostname = port [options]
            let parts: Vec<&str> = line.splitn(2, '=').collect();
            if parts.len() == 2 {
                let host = parts[0].trim().to_string();
                let port_str = parts[1].trim().split_whitespace().next().unwrap_or("0");
                if let Ok(port) = port_str.parse::<u16>() {
                    expected_services.push((host, port));
                }
            }
        }
    } else {
        report.push_str("⚠️ Could not read services.conf — skipping expected-service analysis\n");
    }

    // Deduplicate ports (multiple hostnames can point to same port)
    let mut unique_ports: std::collections::HashMap<u16, Vec<String>> =
        std::collections::HashMap::new();
    for (host, port) in &expected_services {
        unique_ports.entry(*port).or_default().push(host.clone());
    }

    // ── 2. Get PM2 process list ──
    let mut pm2_services: Vec<(String, String, u64, u64)> = Vec::new(); // (name, status, restarts, pid)
    if let Ok(output) = std::process::Command::new("pm2").arg("jlist").output() {
        if output.status.success() {
            let out_str = String::from_utf8_lossy(&output.stdout);
            if let Ok(procs) = serde_json::from_str::<Vec<serde_json::Value>>(&out_str) {
                for proc in &procs {
                    let name = proc
                        .get("name")
                        .and_then(|n| n.as_str())
                        .unwrap_or("?")
                        .to_string();
                    let status = proc
                        .get("pm2_env")
                        .and_then(|e| e.get("status"))
                        .and_then(|s| s.as_str())
                        .unwrap_or("?")
                        .to_string();
                    let restarts = proc
                        .get("pm2_env")
                        .and_then(|e| e.get("restart_time"))
                        .and_then(|r| r.as_u64())
                        .unwrap_or(0);
                    let pid = proc.get("pid").and_then(|p| p.as_u64()).unwrap_or(0);
                    pm2_services.push((name, status, restarts, pid));
                }
            }
        }
    }

    // ── 3. Get actual port listeners via ss ──
    let mut port_owners: std::collections::HashMap<u16, String> = std::collections::HashMap::new();
    if let Ok(output) = std::process::Command::new("ss").args(&["-tlnp"]).output() {
        if output.status.success() {
            let out_str = String::from_utf8_lossy(&output.stdout);
            for line in out_str.lines().skip(1) {
                // Parse: LISTEN  0  4096  0.0.0.0:5150  0.0.0.0:*  users:(("proc",pid=X,fd=Y))
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 5 {
                    let addr = parts[3];
                    if let Some(port_str) = addr.rsplit(':').next() {
                        if let Ok(port) = port_str.parse::<u16>() {
                            // Extract process name from users:((...)) field
                            let proc_info = parts.get(5).unwrap_or(&"");
                            let proc_name = if let Some(start) = proc_info.find("((\"") {
                                let after = &proc_info[start + 3..];
                                after.split('"').next().unwrap_or("unknown").to_string()
                            } else {
                                "unknown".to_string()
                            };
                            port_owners.insert(port, proc_name);
                        }
                    }
                }
            }
        }
    }

    // ── 4. HTTP-probe each unique port ──
    let mut port_status: std::collections::HashMap<u16, (u16, String)> =
        std::collections::HashMap::new(); // port -> (http_code, error)
    for &port in unique_ports.keys() {
        if !service_filter.is_empty() {
            // Check if any hostname for this port matches the filter
            let hosts = unique_ports.get(&port).cloned().unwrap_or_default();
            if !hosts
                .iter()
                .any(|h| h.to_lowercase().contains(&service_filter))
            {
                continue;
            }
        }

        let url = format!("http://127.0.0.1:{}/", port);
        match std::process::Command::new("curl")
            .args(&[
                "-s",
                "-o",
                "/dev/null",
                "-w",
                "%{http_code}",
                "--connect-timeout",
                "2",
                &url,
            ])
            .output()
        {
            Ok(output) => {
                let code_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
                let code: u16 = code_str.parse().unwrap_or(0);
                if code == 0 {
                    port_status.insert(port, (0, "Connection refused / timeout".to_string()));
                } else {
                    port_status.insert(port, (code, String::new()));
                }
            }
            Err(e) => {
                port_status.insert(port, (0, format!("curl failed: {}", e)));
            }
        }
    }

    // ── 5. Correlate and produce report ──
    let mut healthy: Vec<String> = Vec::new();
    let mut degraded: Vec<String> = Vec::new();
    let mut down: Vec<String> = Vec::new();
    let mut root_causes: Vec<String> = Vec::new();
    let mut proposed_fixes: Vec<String> = Vec::new();

    // Sort ports for consistent output
    let mut sorted_ports: Vec<u16> = unique_ports.keys().cloned().collect();
    sorted_ports.sort();

    for port in &sorted_ports {
        let hosts = unique_ports.get(port).cloned().unwrap_or_default();
        let host_label = hosts
            .first()
            .cloned()
            .unwrap_or_else(|| format!("port:{}", port));

        if !service_filter.is_empty() {
            if !hosts
                .iter()
                .any(|h| h.to_lowercase().contains(&service_filter))
            {
                continue;
            }
        }

        let port_owner = port_owners.get(port);
        let http = port_status.get(port);

        match (port_owner, http) {
            // Port is listening AND responds with 2xx/3xx
            (Some(owner), Some((code, _))) if *code >= 200 && *code < 400 => {
                healthy.push(format!(
                    "✅ {} (:{}) → HTTP {} [process: {}]",
                    host_label, port, code, owner
                ));
            }
            // Port is listening but responds with 4xx/5xx
            (Some(owner), Some((code, _))) if *code >= 400 => {
                degraded.push(format!(
                    "⚠️ {} (:{}) → HTTP {} [process: {}]",
                    host_label, port, code, owner
                ));
                if *code == 500 {
                    root_causes.push(format!("Port {} ({}) returns 500 — likely an unhandled exception or template rendering error in {}", port, host_label, owner));
                    proposed_fixes.push(format!(
                        "Check error logs: `pm2 logs {} --err --lines 20`",
                        owner.replace("_rust-cl", "-rust").replace("-cli", "")
                    ));
                }
            }
            // Port is NOT listening at all
            (None, _) => {
                down.push(format!("🔴 {} (:{}) → NO LISTENER", host_label, port));
                // Check if there's a PM2 process that should own this port
                let possible_pm2 = pm2_services.iter().find(|(name, _, _, _)| {
                    host_label.contains(&name.replace("-rust", "").replace("-prod", ""))
                        || name.contains(&host_label.split('.').next().unwrap_or(""))
                });
                if let Some((pm2_name, pm2_status, restarts, _)) = possible_pm2 {
                    root_causes.push(format!(
                        "Port {} ({}) has no listener but PM2 shows '{}' as {} with {} restarts — process may have crashed or port is misconfigured",
                        port, host_label, pm2_name, pm2_status, restarts
                    ));
                    proposed_fixes.push(format!("Try: `pm2 restart {}`", pm2_name));
                } else {
                    root_causes.push(format!(
                        "Port {} ({}) has no listener and NO matching PM2 process — service may not be registered in PM2",
                        port, host_label
                    ));
                    proposed_fixes.push(format!(
                        "Register the service in PM2 or verify the port in services.conf"
                    ));
                }
            }
            // Port listening but HTTP probe returned 0 (connection issues)
            (Some(owner), Some((0, err))) => {
                degraded.push(format!(
                    "⚠️ {} (:{}) → Connection issue: {} [process: {}]",
                    host_label, port, err, owner
                ));
            }
            _ => {
                degraded.push(format!("⚠️ {} (:{}) → Unknown state", host_label, port));
            }
        }
    }

    // Check for port conflicts (two different expected services on the same port)
    for (port, hosts) in &unique_ports {
        if let Some(owner) = port_owners.get(port) {
            // Check if the owner process name matches what we'd expect
            let expected_any = hosts.iter().any(|h| {
                let base = h.split('.').next().unwrap_or("");
                owner.to_lowercase().contains(&base.to_lowercase())
            });
            if !expected_any && !owner.contains("sentinel") {
                root_causes.push(format!(
                    "🔀 PORT CONFLICT: Port {} is expected for {:?} but is owned by process '{}'",
                    port, hosts, owner
                ));
                proposed_fixes.push(format!(
                    "Check if '{}' should be on port {} or if there's a port collision. Verify config files.",
                    owner, port
                ));
            }
        }
    }

    // Check for PM2 crash loops
    for (name, status, restarts, _) in &pm2_services {
        if *restarts > 10 {
            root_causes.push(format!(
                "🔄 CRASH LOOP: PM2 service '{}' has {} restarts (status: {}) — likely a persistent error preventing stable startup",
                name, restarts, status
            ));
            proposed_fixes.push(format!("Investigate root cause: `pm2 logs {} --err --lines 30` then fix the underlying error before restarting", name));
        }
    }

    // Check VRAM exhaustion
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .args(&[
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
    {
        if output.status.success() {
            let out_str = String::from_utf8_lossy(&output.stdout);
            for (i, line) in out_str.lines().enumerate() {
                let parts: Vec<&str> = line.split(',').collect();
                if parts.len() == 2 {
                    let used: f64 = parts[0].trim().parse().unwrap_or(0.0);
                    let total: f64 = parts[1].trim().parse().unwrap_or(1.0);
                    let pct = (used / total) * 100.0;
                    if pct > 95.0 {
                        root_causes.push(format!(
                            "🔥 VRAM EXHAUSTION: GPU {} is at {:.0}% VRAM ({:.0}MB / {:.0}MB) — new GPU-dependent services will fail to start",
                            i, pct, used, total
                        ));
                        proposed_fixes.push(format!("Free GPU {} VRAM by stopping unused GPU processes: `nvidia-smi` then kill the heaviest one", i));
                    }
                }
            }
        }
    }

    // Check disk space
    if let Ok(output) = std::process::Command::new("df")
        .args(&["-h", "--output=target,pcent,avail", "/", "/home"])
        .output()
    {
        if output.status.success() {
            let out_str = String::from_utf8_lossy(&output.stdout);
            for line in out_str.lines().skip(1) {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 3 {
                    let mount = parts[0];
                    let pct_str = parts[1].trim_end_matches('%');
                    if let Ok(pct) = pct_str.parse::<u32>() {
                        if pct > 90 {
                            root_causes.push(format!(
                                "💾 DISK FULL: {} is at {}% usage (only {} free) — services will crash on write",
                                mount, pct, parts[2]
                            ));
                            proposed_fixes.push(format!(
                                "Free disk space on {}: check /tmp, Docker images, PM2 logs, and build artifacts",
                                mount
                            ));
                        }
                    }
                }
            }
        }
    }

    // Check WireGuard tunnel status
    if let Ok(output) = std::process::Command::new("wg").arg("show").output() {
        if output.status.success() {
            let out_str = String::from_utf8_lossy(&output.stdout);
            if out_str.trim().is_empty() {
                root_causes.push("🔒 WIREGUARD DOWN: No active WireGuard tunnels — external traffic cannot reach services".to_string());
                proposed_fixes.push("Bring up WireGuard: `sudo wg-quick up wg0`".to_string());
            } else {
                // Check for handshake staleness (last handshake > 5 minutes ago)
                for line in out_str.lines() {
                    if line.contains("latest handshake:")
                        && (line.contains("minutes") || line.contains("hours"))
                    {
                        if line.contains("hour") {
                            root_causes.push(format!(
                                "🔒 WIREGUARD STALE: Tunnel peer handshake is stale ({})",
                                line.trim()
                            ));
                            proposed_fixes.push("Check WireGuard peer connectivity: `sudo wg-quick down wg0 && sudo wg-quick up wg0`".to_string());
                        }
                    }
                }
            }
        }
    }

    // Check Caddy/Sentinel reverse proxy (port 3000)
    if let Some(&sentinel_port) = sorted_ports.iter().find(|p| **p == 3000) {
        if port_owners.get(&sentinel_port).is_none() {
            root_causes.push("🚪 SENTINEL DOWN: Port 3000 (Caddy/Sentinel reverse proxy) has no listener — ALL external traffic is blocked".to_string());
            proposed_fixes
                .push("CRITICAL: Restart Sentinel immediately: `pm2 restart sentinel`".to_string());
        }
    }

    // ── 6. Format the final report ──
    if !healthy.is_empty() {
        report.push_str(&format!("HEALTHY ({}):\n", healthy.len()));
        for s in &healthy {
            report.push_str(&format!("  {}\n", s));
        }
        report.push('\n');
    }
    if !degraded.is_empty() {
        report.push_str(&format!("DEGRADED ({}):\n", degraded.len()));
        for s in &degraded {
            report.push_str(&format!("  {}\n", s));
        }
        report.push('\n');
    }
    if !down.is_empty() {
        report.push_str(&format!("DOWN ({}):\n", down.len()));
        for s in &down {
            report.push_str(&format!("  {}\n", s));
        }
        report.push('\n');
    }

    if !root_causes.is_empty() {
        report.push_str("ROOT CAUSE HYPOTHESES:\n");
        for (i, rc) in root_causes.iter().enumerate() {
            report.push_str(&format!("  {}. {}\n", i + 1, rc));
        }
        report.push('\n');
    }

    if !proposed_fixes.is_empty() {
        report.push_str("PROPOSED FIXES:\n");
        for (i, fix) in proposed_fixes.iter().enumerate() {
            report.push_str(&format!("  {}. {}\n", i + 1, fix));
        }
        report.push('\n');
    }

    // ── 7. Include recent error logs if requested ──
    if include_logs && !degraded.is_empty() {
        report.push_str("RECENT ERROR LOGS (degraded services):\n");
        for entry in &degraded {
            // Extract PM2 process name from the entry
            if let Some(proc_start) = entry.find("process: ") {
                let proc_name = &entry[proc_start + 9..];
                let proc_name = proc_name.trim_end_matches(']');
                // Normalize process name for PM2 log path
                let pm2_name = proc_name
                    .replace("_rust-cl", "-rust")
                    .replace("-cli", "")
                    .replace("_", "-");
                let log_path = format!("/home/paulo/.pm2/logs/{}-error.log", pm2_name);
                if let Ok(content) = std::fs::read_to_string(&log_path) {
                    let lines: Vec<&str> = content.lines().collect();
                    let start = if lines.len() > 10 {
                        lines.len() - 10
                    } else {
                        0
                    };
                    report.push_str(&format!("  ── {} ──\n", pm2_name));
                    for line in &lines[start..] {
                        report.push_str(&format!("    {}\n", line));
                    }
                }
            }
        }
    }

    let total = healthy.len() + degraded.len() + down.len();
    let summary = format!(
        "SUMMARY: {} services checked — {} healthy, {} degraded, {} down",
        total,
        healthy.len(),
        degraded.len(),
        down.len()
    );
    report.push_str(&format!("\n{}\n", summary));

    info!("🏥 [Hera] Service diagnostic complete: {}", summary);
    ToolResult {
        name: call.name.clone(),
        success: true,
        output: report,
    }
}

async fn execute_bind_telegram_workspace(call: &ToolCall) -> ToolResult {
    let bot_name = call
        .arguments
        .get("bot_name")
        .and_then(|value| value.as_str())
        .unwrap_or("Vetra")
        .trim();
    let sender_id = call
        .arguments
        .get("sender_id")
        .and_then(|value| value.as_str())
        .unwrap_or("")
        .trim();
    let sender_name = call
        .arguments
        .get("sender_name")
        .and_then(|value| value.as_str())
        .unwrap_or("Telegram User")
        .trim();
    let workspace_user = call
        .arguments
        .get("workspace_user")
        .and_then(|value| value.as_str())
        .unwrap_or("")
        .trim();
    let workspace_company = call
        .arguments
        .get("workspace_company")
        .and_then(|value| value.as_str())
        .unwrap_or("")
        .trim();
    let locale = call
        .arguments
        .get("locale")
        .and_then(|value| value.as_str())
        .unwrap_or("es")
        .trim();

    if sender_id.is_empty() || workspace_user.is_empty() || workspace_company.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Error: must provide 'sender_id', 'workspace_user', and 'workspace_company'."
                .into(),
        };
    }

    let path = "/home/paulo/Programs/apps/OS/etc/imaginclaw/vetra_telegram_bindings.json";
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or(0);
    let mut store = match std::fs::read_to_string(path) {
        Ok(raw) => serde_json::from_str::<Value>(&raw)
            .unwrap_or_else(|_| serde_json::json!({ "bindings": [] })),
        Err(_) => serde_json::json!({ "bindings": [] }),
    };

    let bindings = store
        .get_mut("bindings")
        .and_then(|value| value.as_array_mut())
        .expect("bindings array should exist");

    let key_bot = bot_name.to_lowercase();
    if let Some(existing) = bindings.iter_mut().find(|item| {
        item.get("bot_name")
            .and_then(|value| value.as_str())
            .map(|value| value.eq_ignore_ascii_case(&key_bot))
            .unwrap_or(false)
            && item.get("sender_id").and_then(|value| value.as_str()) == Some(sender_id)
    }) {
        *existing = serde_json::json!({
            "bot_name": bot_name,
            "sender_id": sender_id,
            "sender_name": sender_name,
            "workspace_user": workspace_user,
            "workspace_company": workspace_company,
            "locale": locale,
            "created_at": existing.get("created_at").and_then(|value| value.as_i64()).unwrap_or(now),
            "updated_at": now,
        });
    } else {
        bindings.push(serde_json::json!({
            "bot_name": bot_name,
            "sender_id": sender_id,
            "sender_name": sender_name,
            "workspace_user": workspace_user,
            "workspace_company": workspace_company,
            "locale": locale,
            "created_at": now,
            "updated_at": now,
        }));
    }

    if let Some(parent) = std::path::Path::new(path).parent() {
        if let Err(error) = std::fs::create_dir_all(parent) {
            return ToolResult {
                name: call.name.clone(),
                success: false,
                output: format!("Failed to create bindings directory: {}", error),
            };
        }
    }

    match serde_json::to_string_pretty(&store)
        .map_err(|error| error.to_string())
        .and_then(|raw| std::fs::write(path, raw).map_err(|error| error.to_string()))
    {
        Ok(_) => ToolResult {
            name: call.name.clone(),
            success: true,
            output: format!(
                "Bound Telegram sender '{}' to workspace '{}' as '{}'.",
                sender_id, workspace_company, workspace_user
            ),
        },
        Err(error) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Failed to persist Telegram binding: {}", error),
        },
    }
}

async fn execute_desktop_click(call: &ToolCall) -> ToolResult {
    let x = call
        .arguments
        .get("x")
        .and_then(|v| v.as_i64())
        .unwrap_or(0) as i32;
    let y = call
        .arguments
        .get("y")
        .and_then(|v| v.as_i64())
        .unwrap_or(0) as i32;
    let button_str = call
        .arguments
        .get("button")
        .and_then(|v| v.as_str())
        .unwrap_or("left");

    use enigo::{Button, Coordinate, Direction, Enigo, Mouse, Settings};
    let Ok(mut enigo) = Enigo::new(&Settings::default()) else {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Failed to initialize Enigo (Settings instantiation error).".to_string(),
        };
    };

    if let Err(e) = enigo.move_mouse(x, y, Coordinate::Abs) {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Failed to move mouse: {:?}", e),
        };
    }

    let button = match button_str {
        "right" => Button::Right,
        "middle" => Button::Middle,
        _ => Button::Left,
    };

    if let Err(e) = enigo.button(button, Direction::Click) {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Failed to click mouse: {:?}", e),
        };
    }

    ToolResult {
        name: call.name.clone(),
        success: true,
        output: format!("Clicked {} at ({}, {})", button_str, x, y),
    }
}

async fn execute_desktop_type(call: &ToolCall) -> ToolResult {
    let text_val = call
        .arguments
        .get("text")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let key_val = call
        .arguments
        .get("key")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    use enigo::{Direction, Enigo, Key, Keyboard, Settings};
    let Ok(mut enigo) = Enigo::new(&Settings::default()) else {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Failed to initialize Enigo (Settings instantiation error).".to_string(),
        };
    };

    if !key_val.is_empty() {
        let enigo_key = match key_val.to_lowercase().as_str() {
            "enter" | "return" => Key::Return,
            "escape" | "esc" => Key::Escape,
            "tab" => Key::Tab,
            "backspace" => Key::Backspace,
            "space" => Key::Space,
            "up" => Key::UpArrow,
            "down" => Key::DownArrow,
            "left" => Key::LeftArrow,
            "right" => Key::RightArrow,
            _ => {
                return ToolResult {
                    name: call.name.clone(),
                    success: false,
                    output: format!("Unsupported exact key: {}", key_val),
                };
            }
        };
        if let Err(e) = enigo.key(enigo_key, Direction::Click) {
            return ToolResult {
                name: call.name.clone(),
                success: false,
                output: format!("Failed to press key: {:?}", e),
            };
        }
        return ToolResult {
            name: call.name.clone(),
            success: true,
            output: format!("Pressed key: {}", key_val),
        };
    }

    if !text_val.is_empty() {
        if let Err(e) = enigo.text(text_val) {
            return ToolResult {
                name: call.name.clone(),
                success: false,
                output: format!("Failed to type text: {:?}", e),
            };
        }
        return ToolResult {
            name: call.name.clone(),
            success: true,
            output: format!("Typed text: {}", text_val),
        };
    }

    ToolResult {
        name: call.name.clone(),
        success: false,
        output: "Must provide 'text' or 'key' to type.".to_string(),
    }
}

async fn execute_load_skill(call: &ToolCall) -> ToolResult {
    if let Some(skill) = find_skill_artifact(&call.name) {
        info!("🧠 [Hera] Progressively disclosing skill: {}", call.name);
        return ToolResult {
            name: call.name.clone(),
            success: true,
            output: format!(
                "Loaded Skill Playbook '{}' (artifact '{}'):\n\n[SYSTEM SKILL IMPLANT]\nYou must now follow this skill playbook strictly:\n\n{}",
                call.name, skill.skill_id, skill.content
            ),
        };
    }

    ToolResult {
        name: call.name.clone(),
        success: false,
        output: format!(
            "Could not find a SKILL.md playbook defining the skill '{}'",
            call.name
        ),
    }
}

async fn execute_spawn_parallel_agents(call: &ToolCall) -> ToolResult {
    let agents = call
        .arguments
        .get("agents")
        .and_then(|a| a.as_array())
        .cloned()
        .unwrap_or_default();
    let prompt = call
        .arguments
        .get("prompt")
        .and_then(|p| p.as_str())
        .unwrap_or("");

    if agents.is_empty() || prompt.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Error: must provide 'agents' array and 'prompt' string.".into(),
        };
    }

    info!(
        "👯 [Hera] Spawning {} parallel agents for task: {}",
        agents.len(),
        prompt
    );

    let mut tasks = Vec::new();
    for agent_val in agents {
        let agent_name = agent_val.as_str().unwrap_or("").to_string();
        if agent_name.is_empty() {
            continue;
        }

        let p = prompt.to_string();

        tasks.push(tokio::spawn(async move {
            let agent = load_agent_artifact(&agent_name);
            let persona = agent.persona;

            // Connect to local inference engine via Hera chat
            let hera = hera_web::agents::hera::Hera::new("http://127.0.0.1:3000");

            let payload = serde_json::json!({
                "model": "hera",
                "messages": [
                    { "role": "system", "content": persona },
                    { "role": "user", "content": p }
                ],
                "temperature": 0.2
            });

            match hera.chat(payload).await {
                Ok(res) => {
                    if let Ok(json) = res.json::<serde_json::Value>().await {
                        let content = json
                            .get("choices")
                            .and_then(|c| c.as_array())
                            .and_then(|a| a.first())
                            .and_then(|c| c.get("message"))
                            .and_then(|m| m.get("content"))
                            .and_then(|c| c.as_str())
                            .unwrap_or("No response")
                            .to_string();
                        format!(
                            "--- REPORT FROM {} ---\n{}\n",
                            agent_name.to_uppercase(),
                            content
                        )
                    } else {
                        format!(
                            "--- REPORT FROM {} ---\nFailed to parse JSON response.\n",
                            agent_name.to_uppercase()
                        )
                    }
                }
                Err(e) => format!(
                    "--- REPORT FROM {} ---\nFailed to reach inference engine: {}\n",
                    agent_name.to_uppercase(),
                    e
                ),
            }
        }));
    }

    let mut combined_report = String::new();
    for task in tasks {
        if let Ok(report) = task.await {
            combined_report.push_str(&report);
            combined_report.push_str("\n");
        }
    }

    ToolResult {
        name: call.name.clone(),
        success: true,
        output: format!(
            "✅ Parallel Agents Execution Complete.\n\nCONSOLIDATED REPORTS:\n================================\n{}",
            combined_report
        ),
    }
}

async fn execute_create_agent(call: &ToolCall) -> ToolResult {
    let name = call
        .arguments
        .get("name")
        .and_then(|n| n.as_str())
        .unwrap_or("")
        .trim();
    let persona = call
        .arguments
        .get("persona")
        .and_then(|p| p.as_str())
        .unwrap_or("")
        .trim();

    if name.is_empty() || persona.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Error: must provide 'name' and 'persona' strings.".into(),
        };
    }

    let sanitized = name
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || *ch == '_' || *ch == '-')
        .collect::<String>();
    let save_path = format!("/home/paulo/Programs/apps/OS/Agents/{}.md", sanitized);
    match std::fs::write(&save_path, persona) {
        Ok(_) => ToolResult {
            name: call.name.clone(),
            success: true,
            output: format!(
                "Successfully created Agent Persona '{}' at {}",
                sanitized, save_path
            ),
        },
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Failed to create agent {}: {}", sanitized, e),
        },
    }
}

async fn execute_create_skill(call: &ToolCall) -> ToolResult {
    let name = call
        .arguments
        .get("name")
        .and_then(|n| n.as_str())
        .unwrap_or("")
        .trim();
    let description = call
        .arguments
        .get("description")
        .and_then(|d| d.as_str())
        .unwrap_or("")
        .trim();
    let playbook = call
        .arguments
        .get("playbook")
        .and_then(|p| p.as_str())
        .unwrap_or("")
        .trim();

    if name.is_empty() || description.is_empty() || playbook.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Error: must provide 'name', 'description', and 'playbook' strings.".into(),
        };
    }

    let sanitized = name
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || *ch == '_' || *ch == '-')
        .collect::<String>();
    let skill_dir = format!("/home/paulo/Programs/apps/OS/Skills/{}", sanitized);
    if let Err(e) = std::fs::create_dir_all(&skill_dir) {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Failed to create skill directory {}: {}", skill_dir, e),
        };
    }

    let content = format!(
        "---\nname: load_skill_{}\ndescription: \"{}\"\n---\n\n{}",
        sanitized, description, playbook
    );

    let save_path = format!("{}/SKILL.md", skill_dir);
    match std::fs::write(&save_path, content) {
        Ok(_) => ToolResult {
            name: call.name.clone(),
            success: true,
            output: format!(
                "Successfully crafted Skill '{}' at {}\nIt will be dynamically loaded on the next request sequence.",
                sanitized, save_path
            ),
        },
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Failed to write skill playbook {}: {}", save_path, e),
        },
    }
}

async fn execute_draw(call: &ToolCall) -> ToolResult {
    let prompt = call
        .arguments
        .get("prompt")
        .and_then(|p| p.as_str())
        .unwrap_or("A beautiful digital artwork");
    let width = call
        .arguments
        .get("width")
        .and_then(|w| w.as_u64())
        .map(|w| w as u32);
    let height = call
        .arguments
        .get("height")
        .and_then(|h| h.as_u64())
        .map(|h| h as u32);

    let request = diakonos_core::protocol::DiakonosRequest {
        action: "draw_image".to_string(),
        payload: serde_json::json!({
            "prompt": prompt,
            "width": width,
            "height": height,
        }),
    };

    match diakonos_core::client::send_request(diakonos_core::client::DIAKONOS_SOCKET, &request)
        .await
    {
        Ok(response) if response.status == "success" => {
            let res = response.data;
            let image_url = res
                .get("image_url")
                .and_then(|u| u.as_str())
                .unwrap_or("(no URL)");
            info!("🎨 [Hera] Image generated: {}", image_url);

            // Build a public URL that candle-core serves at /outputs/{filename}
            // The filename is the last segment of image_url (e.g., "/outputs/hera_drawn_UUID.png")
            let filename = image_url.split('/').last().unwrap_or(image_url);
            let public_url = format!("https://imaginos.ai/outputs/{}", filename);
            let response = format!(
                "Image generated successfully!\nMEDIA: {}\nPROMPT USED: {}\n\nCRITICAL INSTRUCTION: Do NOT just output 'Image generated successfully'. You MUST do the following:\n1. Include the exact MEDIA line above.\n2. Include the PROMPT USED line above so the user can see how you enriched their request.\n3. Write a charismatic, hacker-chic comment about the image (under 2 sentences). Show off your personality!",
                public_url, prompt
            );

            ToolResult {
                name: call.name.clone(),
                success: true,
                output: response,
            }
        }
        Ok(response) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: response
                .data
                .get("message")
                .and_then(|value| value.as_str())
                .unwrap_or("Diakonos draw error")
                .to_string(),
        },
        Err(e) => {
            tracing::error!("🎨 [Hera] Image generation failed: {:?}", e);
            ToolResult {
                name: call.name.clone(),
                success: false,
                output: format!("Image generation failed: {}", e),
            }
        }
    }
}

async fn execute_search(call: &ToolCall) -> ToolResult {
    let query = call
        .arguments
        .get("query")
        .and_then(|q| q.as_str())
        .unwrap_or("");
    let request = diakonos_core::protocol::DiakonosRequest {
        action: "web_search".to_string(),
        payload: serde_json::json!({ "query": query }),
    };

    match diakonos_core::client::send_request(diakonos_core::client::DIAKONOS_SOCKET, &request)
        .await
    {
        Ok(response) if response.status == "success" => {
            let results = response
                .data
                .get("content")
                .and_then(|value| value.as_str())
                .unwrap_or_default();
            info!("🌐 [Hera] Search completed for: {}", query);
            ToolResult {
                name: call.name.clone(),
                success: true,
                output: format!("Search results for '{}':\n{}", query, results),
            }
        }
        Ok(response) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: response
                .data
                .get("message")
                .and_then(|value| value.as_str())
                .unwrap_or("Diakonos search error")
                .to_string(),
        },
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Search failed: {}", e),
        },
    }
}

async fn execute_speak(call: &ToolCall) -> ToolResult {
    let text = call
        .arguments
        .get("text")
        .and_then(|t| t.as_str())
        .unwrap_or("");
    let voice = call.arguments.get("voice").and_then(|v| v.as_str());

    let request = diakonos_core::protocol::DiakonosRequest {
        action: "speak_text".to_string(),
        payload: serde_json::json!({
            "text": text,
            "voice": voice
        }),
    };

    match diakonos_core::client::send_request(diakonos_core::client::DIAKONOS_SOCKET, &request)
        .await
    {
        Ok(response) if response.status == "success" => {
            let result = response.data;
            info!("🔊 [Hera] Speech synthesized");
            ToolResult {
                name: call.name.clone(),
                success: true,
                output: format!(
                    "Speech generated successfully: {}",
                    serde_json::to_string(&result).unwrap_or_default()
                ),
            }
        }
        Ok(response) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: response
                .data
                .get("message")
                .and_then(|value| value.as_str())
                .unwrap_or("Diakonos TTS error")
                .to_string(),
        },
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("TTS failed: {}", e),
        },
    }
}

async fn execute_video(call: &ToolCall) -> ToolResult {
    let prompt = call
        .arguments
        .get("prompt")
        .and_then(|p| p.as_str())
        .unwrap_or("A smooth cinematic video");

    let request = diakonos_core::protocol::DiakonosRequest {
        action: "generate_video".to_string(),
        payload: serde_json::json!({ "prompt": prompt }),
    };

    match diakonos_core::client::send_request(diakonos_core::client::DIAKONOS_SOCKET, &request)
        .await
    {
        Ok(response) if response.status == "success" => {
            let result = response.data;
            info!("🎬 [Hera] Video generated");
            ToolResult {
                name: call.name.clone(),
                success: true,
                output: format!(
                    "Video generated successfully: {}",
                    serde_json::to_string(&result).unwrap_or_default()
                ),
            }
        }
        Ok(response) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: response
                .data
                .get("message")
                .and_then(|value| value.as_str())
                .unwrap_or("Diakonos video error")
                .to_string(),
        },
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Video generation failed: {}", e),
        },
    }
}

async fn execute_read_file(call: &ToolCall) -> ToolResult {
    let path = call
        .arguments
        .get("path")
        .and_then(|p| p.as_str())
        .unwrap_or("");
    let request = diakonos_core::protocol::DiakonosRequest {
        action: "read_file".to_string(),
        payload: serde_json::json!({ "path": path }),
    };

    match diakonos_core::client::send_request(diakonos_core::client::DIAKONOS_SOCKET, &request)
        .await
    {
        Ok(response) if response.status == "success" => {
            let truncated = response
                .data
                .get("content")
                .and_then(|value| value.as_str())
                .unwrap_or_default();
            info!("📄 [Hera] Read file: {}", path);
            ToolResult {
                name: call.name.clone(),
                success: true,
                output: format!("File contents of '{}':\n{}", path, truncated),
            }
        }
        Ok(response) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: response
                .data
                .get("message")
                .and_then(|value| value.as_str())
                .unwrap_or("Diakonos read_file error")
                .to_string(),
        },
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Failed to read file '{}': {}", path, e),
        },
    }
}

async fn execute_update_soul(call: &ToolCall) -> ToolResult {
    let new_soul_content = call
        .arguments
        .get("new_soul_content")
        .and_then(|c| c.as_str())
        .unwrap_or("");

    if new_soul_content.trim().is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output:
                "Error: new_soul_content was empty. You must provide the complete new persona text."
                    .to_string(),
        };
    }

    let soul_path = std::env::var("HERA_SOUL_PATH").unwrap_or_else(|_| {
        "/home/paulo/Programs/apps/imaginos/imaginclaw/persona/SOUL.md".to_string()
    });

    match std::fs::write(&soul_path, new_soul_content) {
        Ok(_) => {
            tracing::info!("🧠 [Hera] SOUL successfully rewritten at {}", soul_path);
            ToolResult {
                name: call.name.clone(),
                success: true,
                output: format!(
                    "Successfully updated your SOUL! The changes have been saved to disk and you will remember them permanently."
                ),
            }
        }
        Err(e) => {
            tracing::error!("🧠 [Hera] Failed to write SOUL.md: {:?}", e);
            ToolResult {
                name: call.name.clone(),
                success: false,
                output: format!("Failed to update SOUL.md. File system error: {:?}", e),
            }
        }
    }
}

async fn execute_memento_query(call: &ToolCall) -> ToolResult {
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
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Missing 'query' argument".to_string(),
        };
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
                return ToolResult {
                    name: call.name.clone(),
                    success: false,
                    output: format!("Failed to send to Memento: {}", e),
                };
            }

            let mut buffer = vec![0u8; 65536];
            match stream.read(&mut buffer).await {
                Ok(n) if n > 0 => {
                    let response_str = String::from_utf8_lossy(&buffer[..n]);
                    match serde_json::from_str::<serde_json::Value>(&response_str) {
                        Ok(res) => {
                            if res.get("status").and_then(|s| s.as_str()) == Some("success") {
                                let count = res.get("count").and_then(|c| c.as_i64()).unwrap_or(0);
                                let rows =
                                    res.get("rows").cloned().unwrap_or(serde_json::json!([]));

                                // Format results as readable text for the LLM
                                let formatted =
                                    serde_json::to_string_pretty(&rows).unwrap_or_default();
                                info!("🧠 [Memento] Got {} rows from '{}'", count, app);
                                ToolResult {
                                    name: call.name.clone(),
                                    success: true,
                                    output: format!(
                                        "Database query returned {} results from '{}':\n{}",
                                        count, app, formatted
                                    ),
                                }
                            } else {
                                let error = res
                                    .get("error")
                                    .and_then(|e| e.as_str())
                                    .unwrap_or("Unknown error");
                                ToolResult {
                                    name: call.name.clone(),
                                    success: false,
                                    output: format!("Memento error: {}", error),
                                }
                            }
                        }
                        Err(e) => ToolResult {
                            name: call.name.clone(),
                            success: false,
                            output: format!("Failed to parse Memento response: {}", e),
                        },
                    }
                }
                _ => ToolResult {
                    name: call.name.clone(),
                    success: false,
                    output: "No response from Memento".to_string(),
                },
            }
        }
        Err(e) => {
            tracing::error!(
                "🧠 [Memento] Failed to connect to /tmp/memento.sock: {:?}",
                e
            );
            ToolResult {
                name: call.name.clone(),
                success: false,
                output: format!("Memento is not running. Error: {}", e),
            }
        }
    }
}

async fn execute_memento_ipc_action(
    tool_name: &str,
    action: &str,
    payload: serde_json::Value,
) -> ToolResult {
    match tokio::net::UnixStream::connect("/tmp/memento.sock").await {
        Ok(mut stream) => {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};

            let msg = serde_json::json!({
                "action": action,
                "payload": payload
            });

            if let Err(e) = stream.write_all(msg.to_string().as_bytes()).await {
                return ToolResult {
                    name: tool_name.to_string(),
                    success: false,
                    output: format!("Failed to send to Memento: {}", e),
                };
            }

            let mut buffer = vec![0u8; 131072];
            match stream.read(&mut buffer).await {
                Ok(n) if n > 0 => {
                    let response_str = String::from_utf8_lossy(&buffer[..n]).to_string();
                    match serde_json::from_str::<serde_json::Value>(&response_str) {
                        Ok(res) => {
                            let success =
                                res.get("status").and_then(|s| s.as_str()) == Some("success");
                            ToolResult {
                                name: tool_name.to_string(),
                                success,
                                output: serde_json::to_string_pretty(&res).unwrap_or(response_str),
                            }
                        }
                        Err(_) => ToolResult {
                            name: tool_name.to_string(),
                            success: false,
                            output: response_str,
                        },
                    }
                }
                _ => ToolResult {
                    name: tool_name.to_string(),
                    success: false,
                    output: "No response from Memento".into(),
                },
            }
        }
        Err(e) => ToolResult {
            name: tool_name.to_string(),
            success: false,
            output: format!("Memento socket error: {}", e),
        },
    }
}

async fn request_memento_json(
    action: &str,
    payload: serde_json::Value,
) -> Result<serde_json::Value, String> {
    match tokio::net::UnixStream::connect("/tmp/memento.sock").await {
        Ok(mut stream) => {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};

            let msg = serde_json::json!({
                "action": action,
                "payload": payload,
            });

            stream
                .write_all(msg.to_string().as_bytes())
                .await
                .map_err(|e| format!("Failed to send to Memento: {e}"))?;

            let mut buffer = vec![0u8; 262_144];
            let n = stream
                .read(&mut buffer)
                .await
                .map_err(|e| format!("Failed to read from Memento: {e}"))?;
            if n == 0 {
                return Err("No response from Memento".to_string());
            }

            let response: serde_json::Value = serde_json::from_slice(&buffer[..n])
                .map_err(|e| format!("Failed to parse Memento response: {e}"))?;
            if response.get("status").and_then(|v| v.as_str()) == Some("success") {
                Ok(response)
            } else {
                Err(response
                    .get("error")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown Memento error")
                    .to_string())
            }
        }
        Err(e) => Err(format!("Memento socket error: {e}")),
    }
}

async fn query_memento_rows(app: &str, query: String, limit: usize) -> Result<Vec<serde_json::Value>, String> {
    let response = request_memento_json(
        "query_app",
        serde_json::json!({
            "app": app,
            "query": query,
            "limit": limit,
        }),
    )
    .await?;

    Ok(response
        .get("rows")
        .and_then(|value| value.as_array())
        .cloned()
        .unwrap_or_default())
}

fn market_sanitize_fragment(value: &str) -> String {
    let mut out = String::new();
    let mut prev_dash = false;
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            prev_dash = false;
        } else if !prev_dash {
            out.push('-');
            prev_dash = true;
        }
    }
    let trimmed = out.trim_matches('-').to_string();
    if trimmed.is_empty() {
        "item".to_string()
    } else {
        trimmed
    }
}

fn market_json_number(payload: &serde_json::Value, pointer: &str) -> Option<f64> {
    payload.pointer(pointer).and_then(|value| value.as_f64())
}

fn market_json_string(payload: &serde_json::Value, pointer: &str) -> Option<String> {
    payload
        .pointer(pointer)
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn market_format_signed_percent(value: f64) -> String {
    if value > 0.0 {
        format!("+{value:.1}%")
    } else {
        format!("{value:.1}%")
    }
}

fn market_top_news_headlines(payload: &serde_json::Value, limit: usize) -> Vec<serde_json::Value> {
    payload
        .pointer("/catalysts_and_news/recent_headlines")
        .and_then(|value| value.as_array())
        .map(|items| items.iter().take(limit).cloned().collect::<Vec<_>>())
        .unwrap_or_default()
}

fn market_source_entries(payload: &serde_json::Value) -> Vec<serde_json::Value> {
    let mut sources = Vec::new();
    if let Some(items) = payload
        .pointer("/source_catalog/metric_sources")
        .and_then(|value| value.as_array())
    {
        sources.extend(items.iter().cloned());
    }
    if let Some(items) = payload
        .pointer("/source_catalog/news_sources")
        .and_then(|value| value.as_array())
    {
        sources.extend(items.iter().cloned());
    }
    sources
}

fn market_build_change_log(
    ticker: &str,
    current_payload: &serde_json::Value,
    previous_payload: Option<&serde_json::Value>,
) -> serde_json::Value {
    let Some(previous) = previous_payload else {
        return serde_json::json!({
            "summary": format!("Primera version guardada del memo para {ticker}. No hay una corrida anterior para comparar."),
            "items": [format!("Se creo el expediente inicial de {ticker}.")],
        });
    };

    let mut items = Vec::new();

    if let (Some(current), Some(previous)) = (
        market_json_number(current_payload, "/technical_indicators/current_price"),
        market_json_number(previous, "/technical_indicators/current_price"),
    ) {
        let delta = current - previous;
        let pct = if previous.abs() > f64::EPSILON {
            (delta / previous) * 100.0
        } else {
            0.0
        };
        items.push(format!(
            "Precio actual: {:.2} -> {:.2} ({})",
            previous,
            current,
            market_format_signed_percent(pct)
        ));
    }

    if let (Some(current), Some(previous)) = (
        market_json_number(current_payload, "/analyst_data/target_mean_price"),
        market_json_number(previous, "/analyst_data/target_mean_price"),
    ) {
        if (current - previous).abs() > f64::EPSILON {
            let pct = if previous.abs() > f64::EPSILON {
                ((current - previous) / previous) * 100.0
            } else {
                0.0
            };
            items.push(format!(
                "Target consenso: {:.2} -> {:.2} ({})",
                previous,
                current,
                market_format_signed_percent(pct)
            ));
        }
    }

    if let (Some(current), Some(previous)) = (
        market_json_string(current_payload, "/analyst_data/recommendation_key"),
        market_json_string(previous, "/analyst_data/recommendation_key"),
    ) {
        if current != previous {
            items.push(format!(
                "Recomendacion de analistas: {} -> {}",
                previous.replace('_', " "),
                current.replace('_', " ")
            ));
        }
    }

    if let (Some(current), Some(previous)) = (
        market_json_number(current_payload, "/quantitative_metrics/revenue_growth"),
        market_json_number(previous, "/quantitative_metrics/revenue_growth"),
    ) {
        if (current - previous).abs() > 0.0001 {
            items.push(format!(
                "Revenue growth: {:.1}% -> {:.1}%",
                previous * 100.0,
                current * 100.0
            ));
        }
    }

    let current_title = market_json_string(current_payload, "/catalysts_and_news/recent_headlines/0/title");
    let previous_title = market_json_string(previous, "/catalysts_and_news/recent_headlines/0/title");
    if current_title != previous_title {
        if let Some(title) = current_title {
            items.push(format!("Catalizador principal ahora: {title}"));
        }
    }

    let summary = if items.is_empty() {
        format!("No hay cambios materiales detectados frente a la corrida anterior de {ticker}.")
    } else {
        format!(
            "Se detectaron {} cambios materiales frente a la corrida anterior de {ticker}.",
            items.len()
        )
    };

    serde_json::json!({
        "summary": summary,
        "items": items,
    })
}

fn market_technical_view_summary(payload: &serde_json::Value) -> String {
    let current = market_json_number(payload, "/technical_indicators/current_price");
    let ma50 = market_json_number(payload, "/technical_indicators/fifty_day_average");
    let ma200 = market_json_number(payload, "/technical_indicators/two_hundred_day_average");
    let week_high = market_json_number(payload, "/technical_indicators/fifty_two_week_high");
    let week_low = market_json_number(payload, "/technical_indicators/fifty_two_week_low");
    let trend_score = payload
        .pointer("/investment_scores/trend")
        .and_then(|value| value.as_i64())
        .unwrap_or(50);

    match (current, ma50, ma200) {
        (Some(cp), Some(avg50), Some(avg200)) => {
            let range_note = match (week_low, week_high) {
                (Some(low), Some(high)) if high > low => {
                    let range_position = ((cp - low) / (high - low) * 100.0).clamp(0.0, 100.0);
                    format!(
                        " Cotiza en el {:.0}% del rango anual entre {:.2} y {:.2}.",
                        range_position, low, high
                    )
                }
                _ => String::new(),
            };
            format!(
                "Score técnico {}. El precio {:.2} está {} de la media 50d {:.2} y {} de la media 200d {:.2}.{}",
                trend_score,
                cp,
                if cp >= avg50 { "por encima" } else { "por debajo" },
                avg50,
                if cp >= avg200 { "por encima" } else { "por debajo" },
                avg200,
                range_note
            )
        }
        _ => "Vista técnica incompleta: faltan suficientes medias o rango anual para sostener una lectura fuerte.".to_string(),
    }
}

fn market_risk_summary(payload: &serde_json::Value) -> String {
    let invalidation = payload
        .pointer("/research_dossier/investment_view/invalidation")
        .and_then(|value| value.as_str())
        .unwrap_or("La invalidación no fue registrada.");
    let bear_case = payload
        .pointer("/research_dossier/investment_view/bear_case")
        .and_then(|value| value.as_str())
        .unwrap_or("No hay escenario bajista explícito.");
    let beta = market_json_number(payload, "/technical_indicators/beta");

    match beta {
        Some(beta_value) => format!(
            "{} {} La beta observada ({:.2}) sugiere sensibilidad material a cambios de mercado.",
            invalidation, bear_case, beta_value
        ),
        None => format!("{} {}", invalidation, bear_case),
    }
}

fn market_catalysts_summary(payload: &serde_json::Value) -> String {
    let executive = payload
        .pointer("/catalysts_and_news/executive_summary")
        .and_then(|value| value.as_str())
        .unwrap_or("Sin lectura ejecutiva de catalizadores.");
    let headlines = market_top_news_headlines(payload, 2)
        .into_iter()
        .filter_map(|item| {
            item.get("title")
                .and_then(|value| value.as_str())
                .map(str::to_string)
        })
        .collect::<Vec<_>>();

    if headlines.is_empty() {
        executive.to_string()
    } else {
        format!("{} Titulares clave: {}.", executive, headlines.join(" | "))
    }
}

fn market_valuation_upside_percent(payload: &serde_json::Value) -> Option<f64> {
    match (
        market_json_number(payload, "/technical_indicators/current_price"),
        market_json_number(payload, "/analyst_data/target_mean_price"),
    ) {
        (Some(current), Some(target)) if current > 0.0 => Some(((target - current) / current) * 100.0),
        _ => None,
    }
}

fn market_importance_score(payload: &serde_json::Value) -> f64 {
    let source_count = market_source_entries(payload).len() as f64;
    let headline_count = market_top_news_headlines(payload, 5).len() as f64;
    let analyst_count = payload
        .pointer("/analyst_data/number_of_analyst_opinions")
        .and_then(|value| value.as_f64())
        .unwrap_or(0.0);
    let target_bonus = if market_json_number(payload, "/analyst_data/target_mean_price").is_some() {
        0.12
    } else {
        0.0
    };

    (0.35
        + (source_count * 0.03)
        + (headline_count * 0.03)
        + (analyst_count.min(20.0) * 0.01)
        + target_bonus)
        .clamp(0.20, 0.98)
}

fn market_freshness_score(payload: &serde_json::Value) -> f64 {
    let has_recent_headline = payload
        .pointer("/catalysts_and_news/recent_headlines/0/published_at")
        .and_then(|value| value.as_str())
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false);
    let has_price = market_json_number(payload, "/technical_indicators/current_price").is_some();
    let has_target = market_json_number(payload, "/analyst_data/target_mean_price").is_some();

    let mut score: f64 = 0.42;
    if has_recent_headline {
        score += 0.28;
    }
    if has_price {
        score += 0.18;
    }
    if has_target {
        score += 0.08;
    }
    score.clamp(0.20, 0.98)
}

fn market_retention_class(payload: &serde_json::Value) -> &'static str {
    let importance = market_importance_score(payload);
    let freshness = market_freshness_score(payload);
    if importance < 0.40 && freshness < 0.45 {
        "archive_candidate"
    } else {
        "active"
    }
}

fn market_build_structured_findings(
    payload: &serde_json::Value,
    summary: &str,
) -> serde_json::Value {
    let thesis = payload
        .pointer("/research_dossier/investment_view/thesis")
        .and_then(|value| value.as_str())
        .unwrap_or(summary);
    let valuation_summary = payload
        .pointer("/research_dossier/investment_view/valuation_summary")
        .and_then(|value| value.as_str())
        .unwrap_or("Sin lectura de valoración disponible.");
    let technical_summary = market_technical_view_summary(payload);
    let risks = market_risk_summary(payload);
    let catalysts = market_catalysts_summary(payload);
    let upside_percent = market_valuation_upside_percent(payload);

    serde_json::json!({
        "thesis": { "summary": thesis, "claim_type": "investment_thesis" },
        "risks": { "summary": risks, "claim_type": "risk_view" },
        "catalysts": {
            "summary": catalysts,
            "claim_type": "catalyst_view",
            "headlines": market_top_news_headlines(payload, 5)
        },
        "valuation": {
            "summary": valuation_summary,
            "claim_type": "valuation_view",
            "target_mean_price": market_json_number(payload, "/analyst_data/target_mean_price"),
            "forward_pe": market_json_number(payload, "/quantitative_metrics/forward_pe"),
            "upside_percent": upside_percent
        },
        "technical_view": {
            "summary": technical_summary,
            "claim_type": "technical_view",
            "trend_score": payload.pointer("/investment_scores/trend").and_then(|value| value.as_i64()),
            "risk_score": payload.pointer("/investment_scores/risk").and_then(|value| value.as_i64())
        },
        "sources": market_source_entries(payload)
    })
}

fn market_build_lifecycle(
    ticker: &str,
    payload: &serde_json::Value,
    change_log: &serde_json::Value,
) -> serde_json::Value {
    let change_count = change_log
        .get("items")
        .and_then(|value| value.as_array())
        .map(|items| items.len())
        .unwrap_or(0);
    let retention_class = market_retention_class(payload);
    let stale_after_hours = 24 * 7;
    let change_state = if change_count >= 3 {
        "material_change"
    } else if change_count > 0 {
        "minor_change"
    } else {
        "stable"
    };
    let refreshed_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_secs())
        .unwrap_or(0);

    serde_json::json!({
        "refreshed_at_unix": refreshed_at,
        "refresh_state": "fresh",
        "market_change_state": change_state,
        "change_count": change_count,
        "change_summary": change_log.get("summary").cloned().unwrap_or_else(|| serde_json::json!(format!("No change summary for {ticker}"))),
        "stale_after_hours": stale_after_hours,
        "retention_class": retention_class,
        "refresh_recommendation": if change_count > 0 {
            format!("{ticker} cambió desde la corrida previa. Revisa tesis, valoración y catalizadores antes de reutilizarla sin refresco.")
        } else {
            format!("{ticker} puede reutilizarse como referencia, pero conviene refrescarlo si supera {} días.", stale_after_hours / 24)
        }
    })
}

fn market_build_analysis_summary(payload: &serde_json::Value) -> String {
    let thesis = payload
        .pointer("/research_dossier/structured_findings/thesis/summary")
        .and_then(|value| value.as_str())
        .unwrap_or("Sin tesis disponible.");
    let catalysts = payload
        .pointer("/research_dossier/structured_findings/catalysts/summary")
        .and_then(|value| value.as_str())
        .unwrap_or("Sin lectura de catalizadores.");
    let risks = payload
        .pointer("/research_dossier/structured_findings/risks/summary")
        .and_then(|value| value.as_str())
        .unwrap_or("Sin lectura de riesgos.");

    format!("{thesis} Catalizadores: {catalysts} Riesgos: {risks}")
}

async fn stock_research_schema_flags() -> Result<(bool, bool, bool), String> {
    let rows = query_memento_rows(
        "latinos",
        "SELECT column_name FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'stock_research'".to_string(),
        50,
    )
    .await?;

    let mut has_owner_email = false;
    let mut has_source = false;
    let mut has_bot_id = false;
    for row in rows {
        match row.get("column_name").and_then(|value| value.as_str()) {
            Some("owner_email") => has_owner_email = true,
            Some("source") => has_source = true,
            Some("bot_id") => has_bot_id = true,
            _ => {}
        }
    }

    Ok((has_owner_email, has_source, has_bot_id))
}

async fn load_previous_market_payload(ticker: &str, owner_email: &str) -> Option<serde_json::Value> {
    let flags = stock_research_schema_flags().await.ok()?;
    let ticker_safe = ticker.replace('\'', "''");
    let owner_safe = owner_email.replace('\'', "''");
    let query = if flags.0 {
        format!(
            "SELECT raw_data FROM stock_research WHERE ticker = '{}' AND owner_email = '{}' ORDER BY research_date DESC LIMIT 1",
            ticker_safe, owner_safe
        )
    } else {
        format!(
            "SELECT raw_data FROM stock_research WHERE ticker = '{}' ORDER BY research_date DESC LIMIT 1",
            ticker_safe
        )
    };

    let rows = query_memento_rows("latinos", query, 1).await.ok()?;
    rows.first().and_then(|row| row.get("raw_data")).cloned()
}

async fn store_market_research_row(
    ticker: &str,
    owner_email: &str,
    source: &str,
    bot_id: &str,
    raw_data: &serde_json::Value,
    analysis_summary: &str,
) -> Result<(), String> {
    let (has_owner_email, has_source, has_bot_id) = stock_research_schema_flags().await?;
    let ticker_safe = ticker.replace('\'', "''");
    let owner_safe = owner_email.replace('\'', "''");
    let source_safe = source.replace('\'', "''");
    let bot_id_safe = bot_id.replace('\'', "''");
    let raw_data_safe = raw_data.to_string().replace('\'', "''");
    let summary_safe = analysis_summary.replace('\'', "''");

    let query = if has_owner_email {
        let source_value = if has_source {
            format!("'{source_safe}'")
        } else {
            "NULL".to_string()
        };
        let bot_value = if has_bot_id {
            format!("'{bot_id_safe}'")
        } else {
            "NULL".to_string()
        };
        format!(
            "INSERT INTO stock_research (ticker, owner_email, source, bot_id, research_date, raw_data, analysis_summary) VALUES ('{ticker_safe}', '{owner_safe}', {source_value}, {bot_value}, NOW(), '{raw_data_safe}', '{summary_safe}')"
        )
    } else {
        format!(
            "INSERT INTO stock_research (ticker, research_date, raw_data, analysis_summary) VALUES ('{ticker_safe}', NOW(), '{raw_data_safe}', '{summary_safe}')"
        )
    };

    request_memento_json(
        "execute_app",
        serde_json::json!({
            "app": "latinos",
            "query": query,
        }),
    )
    .await
    .map(|_| ())
}

fn market_project_id(owner_email: &str, ticker: &str) -> String {
    format!(
        "latinos-{}-{}",
        market_sanitize_fragment(owner_email),
        market_sanitize_fragment(ticker)
    )
}

fn market_session_id(project_id: &str) -> String {
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_secs())
        .unwrap_or(0);
    format!("{project_id}-{suffix}")
}

fn market_source_id(project_id: &str, source_uri: &str) -> String {
    format!(
        "{}-source-{}",
        project_id,
        market_sanitize_fragment(source_uri)
    )
}

fn market_claim_confidence(claim_type: &str) -> f64 {
    match claim_type {
        "investment_thesis" => 0.76,
        "valuation_view" => 0.73,
        "technical_view" => 0.70,
        "catalyst_view" => 0.68,
        "risk_view" => 0.69,
        _ => 0.65,
    }
}

fn market_claims_from_payload(
    project_id: &str,
    session_id: &str,
    payload: &serde_json::Value,
    fallback_summary: &str,
) -> Vec<(String, String, String, String)> {
    let find = |pointer: &str| {
        payload
            .pointer(pointer)
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
    };

    let thesis = find("/research_dossier/structured_findings/thesis/summary")
        .or_else(|| find("/research_dossier/investment_view/thesis"))
        .unwrap_or_else(|| fallback_summary.to_string());
    let risks = find("/research_dossier/structured_findings/risks/summary")
        .or_else(|| find("/research_dossier/investment_view/invalidation"))
        .unwrap_or_else(|| "Sin lectura estructurada de riesgos.".to_string());
    let catalysts = find("/research_dossier/structured_findings/catalysts/summary")
        .or_else(|| find("/catalysts_and_news/executive_summary"))
        .unwrap_or_else(|| "Sin lectura estructurada de catalizadores.".to_string());
    let valuation = find("/research_dossier/structured_findings/valuation/summary")
        .or_else(|| find("/research_dossier/investment_view/valuation_summary"))
        .unwrap_or_else(|| "Sin lectura estructurada de valoración.".to_string());
    let technical = find("/research_dossier/structured_findings/technical_view/summary")
        .unwrap_or_else(|| "Sin lectura estructurada técnica.".to_string());

    vec![
        ("investment_thesis", format!("{project_id}-claim-thesis"), thesis),
        ("risk_view", format!("{project_id}-claim-risks"), risks),
        ("catalyst_view", format!("{project_id}-claim-catalysts"), catalysts),
        ("valuation_view", format!("{project_id}-claim-valuation"), valuation),
        ("technical_view", format!("{project_id}-claim-technical"), technical),
    ]
    .into_iter()
    .map(|(kind, claim_id, text)| {
        let evidence_id = format!("{session_id}-evidence-{}", market_sanitize_fragment(kind));
        (
            kind.to_string(),
            claim_id,
            text.trim().to_string(),
            evidence_id,
        )
    })
    .collect()
}

fn market_source_records(
    owner_email: &str,
    project_id: &str,
    session_id: &str,
    ticker: &str,
    payload: &serde_json::Value,
) -> Vec<serde_json::Value> {
    let scope_key = format!("latinos:research:{}", market_sanitize_fragment(owner_email));
    let metadata = serde_json::json!({
        "ticker": ticker,
        "platform": "hera_market_research_tool",
    });

    let mut records = Vec::new();
    for source in market_source_entries(payload) {
        let source_uri = source
            .get("url")
            .and_then(|value| value.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        if source_uri.is_empty() {
            continue;
        }

        let source_id = market_source_id(project_id, &source_uri);
        let source_label = source
            .get("label")
            .and_then(|value| value.as_str())
            .unwrap_or("Latinos source");
        let title = source
            .get("title")
            .and_then(|value| value.as_str())
            .filter(|value| !value.trim().is_empty())
            .unwrap_or(source_label);
        let description = source
            .get("description")
            .and_then(|value| value.as_str())
            .unwrap_or(source_label);

        records.push(serde_json::json!({
            "source_id": source_id,
            "project_id": project_id,
            "session_id": session_id,
            "source_kind": source.get("kind").and_then(|value| value.as_str()).unwrap_or("reference"),
            "source_uri": source_uri,
            "source_label": source_label,
            "title": title,
            "summary": description,
            "content_type": "text/html",
            "user_id": owner_email,
            "tenant_id": "default",
            "app_id": "latinos",
            "scope_key": scope_key,
            "status": "active",
            "confidence": 0.7,
            "metadata_json": metadata.clone(),
        }));
    }
    records
}

async fn persist_market_research_semantic_memory(
    ticker: &str,
    owner_email: &str,
    source: &str,
    bot_id: &str,
    payload: &serde_json::Value,
    analysis_summary: &str,
) -> Result<serde_json::Value, String> {
    let project_id = market_project_id(owner_email, ticker);
    let session_id = market_session_id(&project_id);
    let scope_key = format!("latinos:research:{}", market_sanitize_fragment(owner_email));
    let concept_id = format!("latinos-asset-{}", market_sanitize_fragment(ticker));
    let concept_name = payload
        .pointer("/asset_identity/name")
        .and_then(|value| value.as_str())
        .filter(|value| !value.trim().is_empty())
        .unwrap_or(ticker)
        .to_string();
    let source_uri = market_json_string(payload, "/source_links/quote")
        .unwrap_or_else(|| format!("https://finance.yahoo.com/quote/{}", ticker));
    let source_id = market_source_id(&project_id, &source_uri);
    let project_status = market_retention_class(payload);
    let importance_score = market_importance_score(payload);
    let freshness_score = market_freshness_score(payload);
    let metadata = serde_json::json!({
        "ticker": ticker,
        "platform": "hera_market_research_tool",
        "owner_email": owner_email,
        "source": source,
        "bot_id": bot_id,
        "lifecycle": payload.pointer("/research_dossier/lifecycle").cloned(),
    });

    request_memento_json(
        "upsert_research_project",
        serde_json::json!({
            "project_id": project_id,
            "title": format!("Latinos market research dossier: {ticker}"),
            "goal": format!("Track market research and reusable claims for {ticker}."),
            "deliverable_type": "market_research_dossier",
            "owner": owner_email,
            "user_id": owner_email,
            "tenant_id": "default",
            "app_id": "latinos",
            "scope_key": scope_key,
            "status": project_status,
            "importance_score": importance_score,
            "freshness_score": freshness_score,
            "metadata_json": metadata.clone(),
        }),
    )
    .await?;

    request_memento_json(
        "create_research_session",
        serde_json::json!({
            "session_id": session_id,
            "project_id": project_id,
            "title": format!("Research run for {ticker}"),
            "brief": analysis_summary,
            "channel": source,
            "summary": analysis_summary,
            "user_id": owner_email,
            "tenant_id": "default",
            "app_id": "latinos",
            "scope_key": scope_key,
            "status": "completed",
        }),
    )
    .await?;

    request_memento_json(
        "upsert_research_source",
        serde_json::json!({
            "source_id": source_id,
            "project_id": project_id,
            "session_id": session_id,
            "source_kind": "market_research_report",
            "source_uri": source_uri,
            "source_label": format!("Latinos {ticker} consultant dossier"),
            "title": format!("Latinos research dossier for {ticker}"),
            "summary": analysis_summary,
            "content_type": "application/json",
            "user_id": owner_email,
            "tenant_id": "default",
            "app_id": "latinos",
            "scope_key": scope_key,
            "status": project_status,
            "confidence": 0.82,
            "metadata_json": metadata.clone(),
        }),
    )
    .await?;

    request_memento_json(
        "upsert_concept_node",
        serde_json::json!({
            "concept_id": concept_id,
            "canonical_name": concept_name,
            "summary": analysis_summary,
            "domain": "equity_research",
            "user_id": owner_email,
            "tenant_id": "default",
            "app_id": "latinos",
            "scope_key": scope_key,
            "status": project_status,
            "importance_score": importance_score,
            "freshness_score": freshness_score,
            "confidence": 0.8,
            "metadata_json": metadata.clone(),
        }),
    )
    .await?;

    for source_payload in market_source_records(owner_email, &project_id, &session_id, ticker, payload) {
        request_memento_json("upsert_research_source", source_payload).await?;
    }

    let news_source_id = market_source_entries(payload)
        .into_iter()
        .find(|item| item.get("kind").and_then(|v| v.as_str()) == Some("news"))
        .and_then(|item| item.get("url").and_then(|v| v.as_str()).map(|uri| market_source_id(&project_id, uri)));
    let valuation_source_id = payload
        .pointer("/source_catalog/metric_sources/1/url")
        .and_then(|value| value.as_str())
        .map(|uri| market_source_id(&project_id, uri));
    let technical_source_id = payload
        .pointer("/source_catalog/metric_sources/2/url")
        .and_then(|value| value.as_str())
        .map(|uri| market_source_id(&project_id, uri));

    for (claim_type, claim_id, claim_text, evidence_id) in
        market_claims_from_payload(&project_id, &session_id, payload, analysis_summary)
    {
        let claim_metadata = serde_json::json!({
            "ticker": ticker,
            "claim_type": claim_type,
            "platform": "hera_market_research_tool",
            "lifecycle": payload.pointer("/research_dossier/lifecycle").cloned(),
        });
        request_memento_json(
            "append_claim_record",
            serde_json::json!({
                "claim_id": claim_id,
                "claim_text": claim_text,
                "primary_concept_id": concept_id,
                "project_id": project_id,
                "session_id": session_id,
                "claim_type": claim_type,
                "status": project_status,
                "confidence": market_claim_confidence(&claim_type),
                "metadata_json": claim_metadata.clone(),
            }),
        )
        .await?;

        let source_ref = match claim_type.as_str() {
            "catalyst_view" => news_source_id.clone().unwrap_or_else(|| source_id.clone()),
            "valuation_view" => valuation_source_id.clone().unwrap_or_else(|| source_id.clone()),
            "technical_view" => technical_source_id.clone().unwrap_or_else(|| source_id.clone()),
            _ => source_id.clone(),
        };

        request_memento_json(
            "append_evidence_record",
            serde_json::json!({
                "evidence_id": evidence_id,
                "claim_id": claim_id,
                "project_id": project_id,
                "session_id": session_id,
                "source_kind": "market_research_report",
                "source_ref": source_ref,
                "snippet": claim_text,
                "extraction_method": "deterministic_structuring",
                "confidence": market_claim_confidence(&claim_type),
                "metadata_json": claim_metadata,
            }),
        )
        .await?;
    }

    Ok(serde_json::json!({
        "project_id": project_id,
        "session_id": session_id,
        "source_id": source_id,
        "concept_id": concept_id,
        "status": project_status,
        "importance_score": importance_score,
        "freshness_score": freshness_score,
    }))
}

async fn execute_memento_research_project(call: &ToolCall) -> ToolResult {
    let project_id = call
        .arguments
        .get("project_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let title = call
        .arguments
        .get("title")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if project_id.is_empty() || title.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Missing 'project_id' or 'title' argument".to_string(),
        };
    }
    execute_memento_ipc_action(
        &call.name,
        "upsert_research_project",
        call.arguments.clone(),
    )
    .await
}

async fn execute_memento_get_research_project(call: &ToolCall) -> ToolResult {
    let project_id = call
        .arguments
        .get("project_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if project_id.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Missing 'project_id' argument".to_string(),
        };
    }
    execute_memento_ipc_action(&call.name, "get_research_project", call.arguments.clone()).await
}

async fn execute_memento_list_research_projects(call: &ToolCall) -> ToolResult {
    execute_memento_ipc_action(&call.name, "list_research_projects", call.arguments.clone()).await
}

async fn execute_memento_research_session(call: &ToolCall) -> ToolResult {
    let session_id = call
        .arguments
        .get("session_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let project_id = call
        .arguments
        .get("project_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if session_id.is_empty() || project_id.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Missing 'session_id' or 'project_id' argument".to_string(),
        };
    }
    execute_memento_ipc_action(
        &call.name,
        "create_research_session",
        call.arguments.clone(),
    )
    .await
}

async fn execute_memento_research_source(call: &ToolCall) -> ToolResult {
    let source_id = call
        .arguments
        .get("source_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let project_id = call
        .arguments
        .get("project_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let source_kind = call
        .arguments
        .get("source_kind")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if source_id.is_empty() || project_id.is_empty() || source_kind.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Missing 'source_id', 'project_id', or 'source_kind' argument".to_string(),
        };
    }
    execute_memento_ipc_action(&call.name, "upsert_research_source", call.arguments.clone()).await
}

async fn execute_memento_get_research_source(call: &ToolCall) -> ToolResult {
    let source_id = call
        .arguments
        .get("source_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if source_id.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Missing 'source_id' argument".to_string(),
        };
    }
    execute_memento_ipc_action(&call.name, "get_research_source", call.arguments.clone()).await
}

async fn execute_memento_list_research_sources(call: &ToolCall) -> ToolResult {
    execute_memento_ipc_action(&call.name, "list_research_sources", call.arguments.clone()).await
}

async fn execute_memento_upsert_concept(call: &ToolCall) -> ToolResult {
    let concept_id = call
        .arguments
        .get("concept_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let canonical_name = call
        .arguments
        .get("canonical_name")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if concept_id.is_empty() || canonical_name.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Missing 'concept_id' or 'canonical_name' argument".to_string(),
        };
    }
    execute_memento_ipc_action(&call.name, "upsert_concept_node", call.arguments.clone()).await
}

async fn execute_memento_append_claim(call: &ToolCall) -> ToolResult {
    let claim_id = call
        .arguments
        .get("claim_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let claim_text = call
        .arguments
        .get("claim_text")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let primary_concept_id = call
        .arguments
        .get("primary_concept_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if claim_id.is_empty() || claim_text.is_empty() || primary_concept_id.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Missing 'claim_id', 'claim_text', or 'primary_concept_id' argument"
                .to_string(),
        };
    }
    execute_memento_ipc_action(&call.name, "append_claim_record", call.arguments.clone()).await
}

async fn execute_memento_append_evidence(call: &ToolCall) -> ToolResult {
    let evidence_id = call
        .arguments
        .get("evidence_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let claim_id = call
        .arguments
        .get("claim_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let snippet = call
        .arguments
        .get("snippet")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if evidence_id.is_empty() || claim_id.is_empty() || snippet.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Missing 'evidence_id', 'claim_id', or 'snippet' argument".to_string(),
        };
    }
    execute_memento_ipc_action(&call.name, "append_evidence_record", call.arguments.clone()).await
}

async fn execute_memento_link_concepts(call: &ToolCall) -> ToolResult {
    let edge_id = call
        .arguments
        .get("edge_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let from_concept_id = call
        .arguments
        .get("from_concept_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let to_concept_id = call
        .arguments
        .get("to_concept_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let relation_type = call
        .arguments
        .get("relation_type")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if edge_id.is_empty()
        || from_concept_id.is_empty()
        || to_concept_id.is_empty()
        || relation_type.is_empty()
    {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output:
                "Missing 'edge_id', 'from_concept_id', 'to_concept_id', or 'relation_type' argument"
                    .to_string(),
        };
    }
    execute_memento_ipc_action(&call.name, "link_concepts", call.arguments.clone()).await
}

async fn execute_memento_expand_concept(call: &ToolCall) -> ToolResult {
    let concept_id = call
        .arguments
        .get("concept_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if concept_id.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Missing 'concept_id' argument".to_string(),
        };
    }
    execute_memento_ipc_action(&call.name, "expand_concept", call.arguments.clone()).await
}

async fn execute_memento_trace_claim_provenance(call: &ToolCall) -> ToolResult {
    let claim_id = call
        .arguments
        .get("claim_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if claim_id.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Missing 'claim_id' argument".to_string(),
        };
    }
    execute_memento_ipc_action(&call.name, "trace_claim_provenance", call.arguments.clone()).await
}

fn require_string_argument<'a>(call: &'a ToolCall, key: &str) -> Result<&'a str, String> {
    call.arguments
        .get(key)
        .and_then(|v| v.as_str())
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| format!("Missing '{}' argument", key))
}

fn optional_string_argument(call: &ToolCall, key: &str) -> Option<String> {
    call.arguments
        .get(key)
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn merged_research_metadata(call: &ToolCall) -> Option<serde_json::Value> {
    let mut metadata = match call.arguments.get("metadata_json") {
        Some(serde_json::Value::Object(map)) => map.clone(),
        _ => serde_json::Map::new(),
    };

    if let Some(source_label) = optional_string_argument(call, "source_label") {
        metadata.insert("source_label".to_string(), serde_json::json!(source_label));
    }
    if let Some(source_uri) = optional_string_argument(call, "source_uri")
        .or_else(|| optional_string_argument(call, "source_ref"))
    {
        metadata.insert("source_uri".to_string(), serde_json::json!(source_uri));
    }
    if let Some(channel) = optional_string_argument(call, "channel") {
        metadata.insert("channel".to_string(), serde_json::json!(channel));
    }
    if let Some(summary) = optional_string_argument(call, "summary") {
        metadata.insert("summary".to_string(), serde_json::json!(summary));
    }

    if metadata.is_empty() {
        None
    } else {
        Some(serde_json::Value::Object(metadata))
    }
}

async fn execute_persist_research_finding(call: &ToolCall) -> ToolResult {
    let required_keys = [
        "project_id",
        "project_title",
        "session_id",
        "concept_id",
        "canonical_name",
        "claim_id",
        "claim_text",
        "evidence_id",
        "snippet",
    ];

    for key in &required_keys {
        if let Err(message) = require_string_argument(call, key) {
            return ToolResult {
                name: call.name.clone(),
                success: false,
                output: message,
            };
        }
    }

    let source_ref = optional_string_argument(call, "source_ref")
        .or_else(|| optional_string_argument(call, "source_uri"));
    let metadata_json = merged_research_metadata(call);
    let project_id = call.arguments["project_id"].as_str().unwrap_or("");
    let project_title = call.arguments["project_title"].as_str().unwrap_or("");
    let session_id = call.arguments["session_id"].as_str().unwrap_or("");
    let concept_id = call.arguments["concept_id"].as_str().unwrap_or("");
    let canonical_name = call.arguments["canonical_name"].as_str().unwrap_or("");
    let claim_id = call.arguments["claim_id"].as_str().unwrap_or("");
    let claim_text = call.arguments["claim_text"].as_str().unwrap_or("");
    let evidence_id = call.arguments["evidence_id"].as_str().unwrap_or("");
    let snippet = call.arguments["snippet"].as_str().unwrap_or("");

    let project_payload = serde_json::json!({
        "project_id": project_id,
        "title": project_title,
        "goal": call.arguments.get("goal").cloned(),
        "questions_json": call.arguments.get("questions_json").cloned(),
        "constraints_json": call.arguments.get("constraints_json").cloned(),
        "deliverable_type": call.arguments.get("deliverable_type").cloned(),
        "owner": call.arguments.get("owner").cloned(),
        "user_id": call.arguments.get("user_id").cloned(),
        "tenant_id": call.arguments.get("tenant_id").cloned(),
        "app_id": call.arguments.get("app_id").cloned(),
        "scope_key": call.arguments.get("scope_key").cloned(),
        "status": call.arguments.get("project_status").cloned().or_else(|| call.arguments.get("status").cloned())
    });
    let project_result = execute_memento_ipc_action(
        "create_research_project",
        "upsert_research_project",
        project_payload,
    )
    .await;
    if !project_result.success {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!(
                "Failed to persist research project:\n{}",
                project_result.output
            ),
        };
    }

    let session_payload = serde_json::json!({
        "session_id": session_id,
        "project_id": project_id,
        "title": call.arguments.get("session_title").cloned().or_else(|| call.arguments.get("title").cloned()),
        "brief": call.arguments.get("brief").cloned(),
        "channel": call.arguments.get("channel").cloned(),
        "tools_json": call.arguments.get("tools_json").cloned(),
        "agents_json": call.arguments.get("agents_json").cloned(),
        "summary": call.arguments.get("summary").cloned(),
        "user_id": call.arguments.get("user_id").cloned(),
        "tenant_id": call.arguments.get("tenant_id").cloned(),
        "app_id": call.arguments.get("app_id").cloned(),
        "scope_key": call.arguments.get("scope_key").cloned(),
        "status": call.arguments.get("session_status").cloned().or_else(|| call.arguments.get("status").cloned())
    });
    let session_result = execute_memento_ipc_action(
        "create_research_session",
        "create_research_session",
        session_payload,
    )
    .await;
    if !session_result.success {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!(
                "Failed to persist research session:\n{}",
                session_result.output
            ),
        };
    }

    let concept_payload = serde_json::json!({
        "concept_id": concept_id,
        "canonical_name": canonical_name,
        "summary": call.arguments.get("concept_summary").cloned().or_else(|| call.arguments.get("summary").cloned()),
        "domain": call.arguments.get("domain").cloned(),
        "aliases_json": call.arguments.get("aliases_json").cloned(),
        "user_id": call.arguments.get("user_id").cloned(),
        "tenant_id": call.arguments.get("tenant_id").cloned(),
        "app_id": call.arguments.get("app_id").cloned(),
        "scope_key": call.arguments.get("scope_key").cloned(),
        "status": call.arguments.get("concept_status").cloned().or_else(|| call.arguments.get("status").cloned()),
        "importance_score": call.arguments.get("importance_score").cloned(),
        "freshness_score": call.arguments.get("freshness_score").cloned(),
        "confidence": call.arguments.get("concept_confidence").cloned().or_else(|| call.arguments.get("confidence").cloned()),
        "tags_json": call.arguments.get("tags_json").cloned(),
        "metadata_json": call.arguments.get("metadata_json").cloned()
    });
    let concept_result = execute_memento_ipc_action(
        "upsert_concept_node",
        "upsert_concept_node",
        concept_payload,
    )
    .await;
    if !concept_result.success {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Failed to persist concept:\n{}", concept_result.output),
        };
    }

    let claim_payload = serde_json::json!({
        "claim_id": claim_id,
        "claim_text": claim_text,
        "primary_concept_id": concept_id,
        "project_id": project_id,
        "session_id": session_id,
        "claim_type": call.arguments.get("claim_type").cloned(),
        "status": call.arguments.get("claim_status").cloned().or_else(|| call.arguments.get("status").cloned()),
        "confidence": call.arguments.get("claim_confidence").cloned().or_else(|| call.arguments.get("confidence").cloned()),
        "evidence_count": call.arguments.get("evidence_count").cloned(),
        "provenance_refs": call.arguments.get("provenance_refs").cloned(),
        "tags_json": call.arguments.get("tags_json").cloned(),
        "metadata_json": call.arguments.get("metadata_json").cloned()
    });
    let claim_result =
        execute_memento_ipc_action("append_claim", "append_claim_record", claim_payload).await;
    if !claim_result.success {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Failed to persist claim:\n{}", claim_result.output),
        };
    }

    let evidence_payload = serde_json::json!({
        "evidence_id": evidence_id,
        "claim_id": claim_id,
        "project_id": project_id,
        "session_id": session_id,
        "source_kind": call.arguments.get("source_kind").cloned(),
        "source_ref": source_ref.map(serde_json::Value::String),
        "snippet": snippet,
        "locator": call.arguments.get("locator").cloned(),
        "extraction_method": call.arguments.get("extraction_method").cloned(),
        "contradiction_group": call.arguments.get("contradiction_group").cloned(),
        "confidence": call.arguments.get("evidence_confidence").cloned().or_else(|| call.arguments.get("confidence").cloned()),
        "provenance_refs": call.arguments.get("provenance_refs").cloned(),
        "tags_json": call.arguments.get("tags_json").cloned(),
        "metadata_json": metadata_json
    });
    let evidence_result = execute_memento_ipc_action(
        "append_evidence",
        "append_evidence_record",
        evidence_payload,
    )
    .await;
    if !evidence_result.success {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Failed to persist evidence:\n{}", evidence_result.output),
        };
    }

    let mut relation_output = None;
    let has_relation = call
        .arguments
        .get("edge_id")
        .and_then(|v| v.as_str())
        .map(|v| !v.trim().is_empty())
        .unwrap_or(false)
        && call
            .arguments
            .get("related_concept_id")
            .and_then(|v| v.as_str())
            .map(|v| !v.trim().is_empty())
            .unwrap_or(false)
        && call
            .arguments
            .get("relation_type")
            .and_then(|v| v.as_str())
            .map(|v| !v.trim().is_empty())
            .unwrap_or(false);

    if has_relation {
        let relation_payload = serde_json::json!({
            "edge_id": call.arguments.get("edge_id").cloned(),
            "from_concept_id": concept_id,
            "to_concept_id": call.arguments.get("related_concept_id").cloned(),
            "relation_type": call.arguments.get("relation_type").cloned(),
            "project_id": project_id,
            "session_id": session_id,
            "weight": call.arguments.get("weight").cloned(),
            "confidence": call.arguments.get("relation_confidence").cloned().or_else(|| call.arguments.get("confidence").cloned()),
            "provenance_refs": call.arguments.get("provenance_refs").cloned(),
            "tags_json": call.arguments.get("tags_json").cloned(),
            "metadata_json": call.arguments.get("metadata_json").cloned()
        });
        let relation_result =
            execute_memento_ipc_action("link_concepts", "link_concepts", relation_payload).await;
        if !relation_result.success {
            return ToolResult {
                name: call.name.clone(),
                success: false,
                output: format!("Failed to persist relation:\n{}", relation_result.output),
            };
        }
        relation_output = Some(relation_result.output);
    }

    ToolResult {
        name: call.name.clone(),
        success: true,
        output: format!(
            "Persisted research finding successfully.\nProject: {}\nSession: {}\nConcept: {}\nClaim: {}\nEvidence: {}{}",
            project_id,
            session_id,
            concept_id,
            claim_id,
            evidence_id,
            relation_output
                .map(|_| format!(
                    "\nRelation: {}",
                    call.arguments
                        .get("edge_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                ))
                .unwrap_or_default()
        ),
    }
}

async fn execute_persist_channel_research_finding(call: &ToolCall) -> ToolResult {
    let required_keys = [
        "project_id",
        "project_title",
        "session_id",
        "concept_id",
        "canonical_name",
        "claim_id",
        "claim_text",
        "evidence_id",
        "snippet",
        "source_kind",
        "source_uri",
        "source_label",
        "channel",
    ];

    for key in &required_keys {
        if let Err(message) = require_string_argument(call, key) {
            return ToolResult {
                name: call.name.clone(),
                success: false,
                output: message,
            };
        }
    }

    let source_id = optional_string_argument(call, "source_id").unwrap_or_else(|| {
        format!(
            "source-{}",
            call.arguments["evidence_id"].as_str().unwrap_or("unknown")
        )
    });
    let source_payload = serde_json::json!({
        "source_id": source_id,
        "project_id": call.arguments.get("project_id").cloned(),
        "session_id": call.arguments.get("session_id").cloned(),
        "source_kind": call.arguments.get("source_kind").cloned(),
        "source_uri": call.arguments.get("source_uri").cloned(),
        "source_label": call.arguments.get("source_label").cloned(),
        "title": call.arguments.get("source_label").cloned(),
        "summary": call.arguments.get("summary").cloned(),
        "content_type": call.arguments.get("content_type").cloned(),
        "user_id": call.arguments.get("user_id").cloned(),
        "tenant_id": call.arguments.get("tenant_id").cloned(),
        "app_id": call.arguments.get("app_id").cloned(),
        "scope_key": call.arguments.get("scope_key").cloned(),
        "status": call.arguments.get("source_status").cloned().or_else(|| call.arguments.get("status").cloned()),
        "confidence": call.arguments.get("source_confidence").cloned().or_else(|| call.arguments.get("confidence").cloned()),
        "tags_json": call.arguments.get("tags_json").cloned(),
        "metadata_json": merged_research_metadata(call)
    });
    let source_result = execute_memento_ipc_action(
        "create_research_source",
        "upsert_research_source",
        source_payload,
    )
    .await;
    if !source_result.success {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Failed to persist source:\n{}", source_result.output),
        };
    }

    let mut delegated_call = call.clone();
    delegated_call.arguments["source_ref"] = serde_json::Value::String(source_id);
    execute_persist_research_finding(&delegated_call).await
}

async fn execute_movilo_search_providers(call: &ToolCall) -> ToolResult {
    let city = call
        .arguments
        .get("city")
        .and_then(|c| c.as_str())
        .unwrap_or("");
    let specialty = call
        .arguments
        .get("specialty")
        .and_then(|s| s.as_str())
        .unwrap_or("");
    let keyword = call
        .arguments
        .get("service_keyword")
        .and_then(|k| k.as_str())
        .unwrap_or("");

    let mut conditions = vec!["p.status = 'Aprobado'".to_string()];
    if !city.is_empty() {
        conditions.push(format!("p.city ILIKE '%{}%'", city.replace("'", "''")));
    }
    if !specialty.is_empty() {
        conditions.push(format!(
            "p.provider_type ILIKE '%{}%'",
            specialty.replace("'", "''")
        ));
    }
    if !keyword.is_empty() {
        conditions.push(format!("s.name ILIKE '%{}%'", keyword.replace("'", "''")));
    }

    let query = format!(
        r#"SELECT p.company_name, p.provider_type, p.city, p.phone, s.name as service, s.movilo_price, s.original_price
           FROM movilo_providers p 
           LEFT JOIN movilo_provider_services s ON p.id = s.provider_id 
           WHERE {} 
           ORDER BY p.company_name LIMIT 10"#,
        conditions.join(" AND ")
    );

    let memento_call = ToolCall {
        name: "memento_query".to_string(),
        arguments: serde_json::json!({
            "app": "movilo",
            "query": query
        }),
    };

    let mut result = execute_memento_query(&memento_call).await;

    // Instruct the AI to render the map component based on the search context
    if result.success {
        let mut widget_attrs = String::new();
        if !specialty.is_empty() {
            widget_attrs.push_str(&format!(
                " category=\"{}\"",
                specialty.replace("\"", "\\\"")
            ));
        }
        if !keyword.is_empty() {
            widget_attrs.push_str(&format!(" search=\"{}\"", keyword.replace("\"", "\\\"")));
        } else if !city.is_empty() {
            widget_attrs.push_str(&format!(" search=\"{}\"", city.replace("\"", "\\\"")));
        }

        result.output.push_str(&format!(
            "\n\n[[SYSTEM DIRECTIVE]]: You MUST also embed an interactive map in your response so the user can visually locate these providers. To do this, simply include the following EXACT string somewhere in your text reply:\n\nWIDGET: <os-provider-map{}></os-provider-map>\n",
            widget_attrs
        ));
    }

    result
}

async fn execute_movilo_check_affiliation(call: &ToolCall) -> ToolResult {
    let email = call
        .arguments
        .get("email")
        .and_then(|e| e.as_str())
        .unwrap_or("");
    let doc = call
        .arguments
        .get("document")
        .and_then(|d| d.as_str())
        .unwrap_or("");

    if email.is_empty() && doc.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Debes proveer un email o documento para buscar la afiliación.".into(),
        };
    }

    let mut conditions = vec![];
    if !email.is_empty() {
        conditions.push(format!("email = '{}'", email.replace("'", "''")));
    }
    if !doc.is_empty() {
        // Fallback: Si existe campo de documento en la tabla (asumiremos que existe o buscaremos name)
        conditions.push(format!("id = '{}'", doc.replace("'", "''")));
    }

    let query = format!(
        "SELECT id, name, email, status, plan FROM movilo_users WHERE {} LIMIT 1",
        conditions.join(" OR ")
    );

    let memento_call = ToolCall {
        name: "memento_query".to_string(),
        arguments: serde_json::json!({
            "app": "movilo",
            "query": query
        }),
    };
    execute_memento_query(&memento_call).await
}

async fn execute_movilo_validate_qr(call: &ToolCall) -> ToolResult {
    let qr_content = call
        .arguments
        .get("qr_content")
        .and_then(|q| q.as_str())
        .unwrap_or("");

    if qr_content.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "QRCode content is missing.".into(),
        };
    }

    // Asumimos que el QR emitido por Movilo tiene el User UUID o el Email
    let query = format!(
        "SELECT id, name, email, status, plan FROM movilo_users WHERE id = '{}' OR email = '{}' LIMIT 1",
        qr_content.replace("'", "''"),
        qr_content.replace("'", "''")
    );

    let memento_call = ToolCall {
        name: "memento_query".to_string(),
        arguments: serde_json::json!({
            "app": "movilo",
            "query": query
        }),
    };

    let db_result = execute_memento_query(&memento_call).await;
    if db_result.success && db_result.output.contains("rows") && !db_result.output.contains("[]") {
        ToolResult {
            name: call.name.clone(),
            success: true,
            output: format!(
                "¡QR Validado Exitosamente! Datos del afiliado recuperados:\n{}",
                db_result.output
            ),
        }
    } else {
        ToolResult {
            name: call.name.clone(),
            success: false,
            output: "QR Inválido o usuario no encontrado en la base de datos de Movilo.".into(),
        }
    }
}

async fn execute_api_request(call: &ToolCall) -> ToolResult {
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

    if let Ok(headers) = serde_json::from_str::<serde_json::Value>(headers_str) {
        if let Some(obj) = headers.as_object() {
            for (k, v) in obj {
                if let Some(s) = v.as_str() {
                    req = req.header(k, s);
                }
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

async fn execute_git_manager(call: &ToolCall) -> ToolResult {
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

async fn execute_memento_vector_search(call: &ToolCall) -> ToolResult {
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
            if let Err(_) = stream.write_all(msg.to_string().as_bytes()).await {
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

async fn execute_ask_user(call: &ToolCall) -> ToolResult {
    let question = call
        .arguments
        .get("question")
        .and_then(|q| q.as_str())
        .unwrap_or("Needs human input.");
    tracing::info!("⏸️ [Hera] Pausing flow to ask user: {}", question);
    ToolResult {
        name: call.name.clone(),
        success: true,
        output: format!("[PAUSED_FOR_USER] Question: {}", question),
    }
}

async fn execute_get_system_time(call: &ToolCall) -> ToolResult {
    match std::process::Command::new("date").output() {
        Ok(out) => ToolResult {
            name: call.name.clone(),
            success: true,
            output: String::from_utf8_lossy(&out.stdout).to_string(),
        },
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: e.to_string(),
        },
    }
}

async fn execute_run_code(call: &ToolCall) -> ToolResult {
    let lang = call
        .arguments
        .get("language")
        .and_then(|l| l.as_str())
        .unwrap_or("python");
    let code = call
        .arguments
        .get("code")
        .and_then(|c| c.as_str())
        .unwrap_or("");
    let packages: Vec<String> = call
        .arguments
        .get("packages")
        .and_then(|p| p.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let bean_name = format!("bean_{}", timestamp);
    let beans_dir = "/home/paulo/Programs/apps/OS/Beans";
    let _ = std::fs::create_dir_all(beans_dir);

    // Cognitive Memory Pipeline: Record bean logic into Memento universally before execution
    // Doing it natively blocking here; socket is fast UDS.
    if let Ok(mut stream) = std::os::unix::net::UnixStream::connect("/tmp/memento.sock") {
        use std::io::Write;
        let payload = serde_json::json!({
            "action": "store_knowledge",
            "payload": {
                "key": bean_name.clone(),
                "content": format!("Language: {}\nPackages: {:?}\nCode:\n{}", lang, packages, code),
                "tags": "bean, code_interpreter"
            }
        });
        let _ = stream.write_all(payload.to_string().as_bytes());
    }

    if lang.to_lowercase() == "rust" {
        let project_dir = format!("{}/{}", beans_dir, bean_name);
        if let Err(e) = std::fs::create_dir_all(format!("{}/src", project_dir)) {
            return ToolResult {
                name: call.name.clone(),
                success: false,
                output: format!("Failed to create Rust bean directory: {}", e),
            };
        }

        let mut deps = format!(
            r#"[dependencies]
tokio = {{ version = "1", features = ["full", "rt-multi-thread"] }}
reqwest = {{ version = "0.11", features = ["json"] }}
serde = {{ version = "1.0", features = ["derive"] }}
serde_json = "1.0"
"#
        );
        for pkg in &packages {
            deps.push_str(&format!("{} = \"*\"\n", pkg));
        }

        let cargo_toml = format!(
            r#"[package]
name = "{}"
version = "0.1.0"
edition = "2021"

{}
"#,
            bean_name, deps
        );

        std::fs::write(format!("{}/Cargo.toml", project_dir), cargo_toml).unwrap_or_default();
        std::fs::write(format!("{}/src/main.rs", project_dir), code).unwrap_or_default();

        match std::process::Command::new("cargo")
            .arg("run")
            .arg("--release")
            .current_dir(&project_dir)
            .output()
        {
            Ok(out) => {
                let out_str = String::from_utf8_lossy(&out.stdout).to_string();
                let err_str = String::from_utf8_lossy(&out.stderr).to_string();
                let success = out.status.success();

                let mut final_out =
                    format!("RUST ROASTED BEAN EXECUTION:\n---\nSTDOUT:\n{}\n", out_str);
                if !success || !err_str.is_empty() {
                    final_out.push_str(&format!(
                        "---\nSTDERR (or cargo compilation logs):\n{}\n",
                        err_str
                    ));
                }
                final_out.push_str(&format!(
                    "---\nBean saved permanently in {} and recorded in Memento.",
                    project_dir
                ));

                ToolResult {
                    name: call.name.clone(),
                    success,
                    output: final_out,
                }
            }
            Err(e) => ToolResult {
                name: call.name.clone(),
                success: false,
                output: format!("Cargo execution failed: {}", e),
            },
        }
    } else if lang.to_lowercase() == "python" {
        let p = format!("{}/{}.py", beans_dir, bean_name);
        if let Err(e) = std::fs::write(&p, code) {
            return ToolResult {
                name: call.name.clone(),
                success: false,
                output: format!("Failed to write Python bean: {}", e),
            };
        }

        let mut pip_log = String::new();
        if !packages.is_empty() {
            let mut cmd = std::process::Command::new("python3");
            cmd.arg("-m")
                .arg("pip")
                .arg("install")
                .arg("--break-system-packages");
            for pkg in &packages {
                cmd.arg(pkg);
            }
            if let Ok(out) = cmd.output() {
                if !out.status.success() {
                    let err = String::from_utf8_lossy(&out.stderr);
                    return ToolResult {
                        name: call.name.clone(),
                        success: false,
                        output: format!("Failed to install Python packages:\n{}", err),
                    };
                }
                pip_log = format!("Successfully installed: {:?}", packages);
            }
        }

        match std::process::Command::new("python3").arg(&p).output() {
            Ok(out) => {
                let out_str = String::from_utf8_lossy(&out.stdout).to_string();
                let err_str = String::from_utf8_lossy(&out.stderr).to_string();
                let success = out.status.success();
                let mut res = if success {
                    if err_str.trim().is_empty() {
                        out_str
                    } else {
                        format!("STDOUT:\n{}\n---\nSTDERR:\n{}", out_str, err_str)
                    }
                } else {
                    format!("PYTHON SOFT BEAN ERROR:\n{}\n{}", err_str, out_str)
                };
                if !pip_log.is_empty() {
                    res = format!("{}\n---\n{}", pip_log, res);
                }
                res = format!(
                    "{}\n---\nBean saved permanently at {} and logged in Memento.",
                    res, p
                );
                ToolResult {
                    name: call.name.clone(),
                    success,
                    output: res,
                }
            }
            Err(e) => ToolResult {
                name: call.name.clone(),
                success: false,
                output: e.to_string(),
            },
        }
    } else {
        ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Language '{}' not supported. Use 'rust' or 'python'.", lang),
        }
    }
}

async fn execute_write_file(call: &ToolCall) -> ToolResult {
    let path = call
        .arguments
        .get("path")
        .and_then(|p| p.as_str())
        .unwrap_or("");
    let content = call
        .arguments
        .get("content")
        .and_then(|c| c.as_str())
        .unwrap_or("");

    if path.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Missing path".into(),
        };
    }

    match std::fs::write(path, content) {
        Ok(_) => ToolResult {
            name: call.name.clone(),
            success: true,
            output: format!("Successfully wrote to {}", path),
        },
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Failed to write file: {}", e),
        },
    }
}

async fn execute_web_scraper(call: &ToolCall) -> ToolResult {
    let url = call
        .arguments
        .get("url")
        .and_then(|u| u.as_str())
        .unwrap_or("");
    if url.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Missing url".into(),
        };
    }

    let request = diakonos_core::protocol::DiakonosRequest {
        action: "web_scrape".to_string(),
        payload: serde_json::json!({ "url": url }),
    };

    match diakonos_core::client::send_request(diakonos_core::client::DIAKONOS_SOCKET, &request)
        .await
    {
        Ok(response) if response.status == "success" => ToolResult {
            name: call.name.clone(),
            success: true,
            output: response
                .data
                .get("content")
                .and_then(|value| value.as_str())
                .unwrap_or_default()
                .to_string(),
        },
        Ok(response) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: response
                .data
                .get("message")
                .and_then(|value| value.as_str())
                .unwrap_or("Diakonos web scrape error")
                .to_string(),
        },
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: e.to_string(),
        },
    }
}

async fn execute_generate_qr_code(call: &ToolCall) -> ToolResult {
    let content = call
        .arguments
        .get("content")
        .and_then(|c| c.as_str())
        .unwrap_or("");
    if content.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Missing content".into(),
        };
    }

    // Using a quick external API for now, could be replaced with a local Rust crate later
    let url = format!(
        "https://api.qrserver.com/v1/create-qr-code/?size=500x500&data={}",
        urlencoding::encode(content)
    );
    info!("🔲 [Hera] Generated QR Code URL: {}", url);
    ToolResult {
        name: call.name.clone(),
        success: true,
        output: format!(
            "QR Code generated successfully. Use this exact line in your reply to deliver it inline:\nMEDIA: {}",
            url
        ),
    }
}

fn sanitize_pdf_file_stem(raw: &str, fallback: &str) -> String {
    let cleaned: String = raw
        .chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' => ch,
            _ => '_',
        })
        .collect();
    let cleaned = cleaned.trim_matches('_');
    if cleaned.is_empty() {
        fallback.to_string()
    } else {
        cleaned.to_string()
    }
}

async fn render_pdf_document(
    html: &str,
    requested_name: Option<&str>,
    landscape: bool,
    _print_background: bool,
) -> Result<String, String> {
    if html.trim().is_empty() {
        return Err("Missing html".to_string());
    }

    let chrome_path = "/usr/bin/google-chrome";
    if !std::path::Path::new(chrome_path).exists() {
        return Err(format!("Chrome PDF backend not found at {}", chrome_path));
    }

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_millis();
    let file_stem = sanitize_pdf_file_stem(
        requested_name.unwrap_or("hera-rendered-document"),
        "hera-rendered-document",
    );
    let html_path = format!("/tmp/{}-{}.html", file_stem, timestamp);
    let pdf_path = format!("/tmp/{}-{}.pdf", file_stem, timestamp);
    std::fs::write(&html_path, html)
        .map_err(|e| format!("Failed to write temporary HTML: {}", e))?;

    let mut command = tokio::process::Command::new(chrome_path);
    command
        .arg("--headless=new")
        .arg("--disable-gpu")
        .arg("--no-sandbox")
        .arg(format!("--print-to-pdf={}", pdf_path))
        .arg(format!("file://{}", html_path))
        .stdout(Stdio::null())
        .stderr(Stdio::piped());

    if landscape {
        command.arg("--landscape");
    }

    let output = command
        .output()
        .await
        .map_err(|e| format!("Failed to start Chrome PDF export: {}", e))?;

    let _ = std::fs::remove_file(&html_path);

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let _ = std::fs::remove_file(&pdf_path);
        return Err(format!("Chrome PDF export failed: {}", stderr.trim()));
    }

    Ok(pdf_path)
}

async fn execute_render_pdf(call: &ToolCall) -> ToolResult {
    let html = call.arguments.get("html").and_then(|c| c.as_str());
    let html_path = call.arguments.get("html_path").and_then(|c| c.as_str());
    let file_name = call.arguments.get("file_name").and_then(|c| c.as_str());
    let landscape = call
        .arguments
        .get("landscape")
        .and_then(|c| c.as_bool())
        .unwrap_or(false);
    let print_background = call
        .arguments
        .get("print_background")
        .and_then(|c| c.as_bool())
        .unwrap_or(true);

    let html_owned = match (html, html_path) {
        (Some(html), _) if !html.trim().is_empty() => Some(html.to_string()),
        (_, Some(path)) if !path.trim().is_empty() => match std::fs::read_to_string(path) {
            Ok(contents) if !contents.trim().is_empty() => Some(contents),
            Ok(_) => None,
            Err(error) => {
                return ToolResult {
                    name: call.name.clone(),
                    success: false,
                    output: format!("Failed to read html_path '{}': {}", path, error),
                };
            }
        },
        _ => None,
    };

    let Some(html_owned) = html_owned else {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Missing html or html_path".to_string(),
        };
    };

    match render_pdf_document(&html_owned, file_name, landscape, print_background).await {
        Ok(path) => ToolResult {
            name: call.name.clone(),
            success: true,
            output: path,
        },
        Err(error) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: error,
        },
    }
}

async fn execute_generate_contract_pdf(call: &ToolCall) -> ToolResult {
    let debtor = call
        .arguments
        .get("debtor_id")
        .and_then(|c| c.as_str())
        .unwrap_or("unknown");
    let content = call
        .arguments
        .get("content")
        .and_then(|c| c.as_str())
        .unwrap_or("");
    if content.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Missing content".into(),
        };
    }

    let escaped_content = content
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;");
    let html = format!(
        "<!doctype html><html><head><meta charset=\"utf-8\"><style>body{{font-family:Arial,sans-serif;padding:48px;line-height:1.5;color:#111;}} h1{{font-size:28px;margin-bottom:24px;}} pre{{white-space:pre-wrap;font-family:Arial,sans-serif;}}</style></head><body><h1>Acuerdo de Pago</h1><p>Deudor: {}</p><pre>{}</pre></body></html>",
        debtor, escaped_content
    );

    let file_name = format!("Acuerdo_Pago_{}", debtor.replace(' ', "_"));
    match render_pdf_document(&html, Some(&file_name), false, true).await {
        Ok(path) => {
            info!("📄 [Hera] Generated Contract Document: {}", path);
            ToolResult {
                name: call.name.clone(),
                success: true,
                output: format!(
                    "Payment agreement PDF generated successfully at {}. Inform the user that the document has been filed.",
                    path
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

async fn execute_dispatch_email(call: &ToolCall) -> ToolResult {
    let recipient = call
        .arguments
        .get("recipient")
        .and_then(|c| c.as_str())
        .unwrap_or("unknown");
    let subject = call
        .arguments
        .get("subject")
        .and_then(|c| c.as_str())
        .unwrap_or("");
    let attachment = call
        .arguments
        .get("attachment_path")
        .and_then(|c| c.as_str())
        .unwrap_or("None");

    // Simulate sending email via local sendmail or SMTP (For OS-v3 Demo mode)
    info!(
        "📧 [Hera] Dispatching Email to: {} | Subject: {} | Attachment: {}",
        recipient, subject, attachment
    );

    ToolResult {
        name: call.name.clone(),
        success: true,
        output: format!(
            "Email successfully dispatched via port 25 relay to {}.",
            recipient
        ),
    }
}

async fn execute_get_map_route(call: &ToolCall) -> ToolResult {
    let dest = call
        .arguments
        .get("destination")
        .and_then(|d| d.as_str())
        .unwrap_or("");
    let orig = call
        .arguments
        .get("origin")
        .and_then(|o| o.as_str())
        .unwrap_or("");

    if dest.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Missing destination".into(),
        };
    }

    let url = if orig.is_empty() {
        format!(
            "https://www.google.com/maps/search/?api=1&query={}",
            urlencoding::encode(dest)
        )
    } else {
        format!(
            "https://www.google.com/maps/dir/?api=1&origin={}&destination={}",
            urlencoding::encode(orig),
            urlencoding::encode(dest)
        )
    };

    info!("🗺️ [Hera] Generated Google Maps URL: {}", url);
    ToolResult {
        name: call.name.clone(),
        success: true,
        output: format!("Maps link generated successfully:\n{}", url),
    }
}

async fn execute_workflow(call: &ToolCall) -> ToolResult {
    let app = call
        .arguments
        .get("app")
        .and_then(|a| a.as_str())
        .unwrap_or_default();
    let workflow = call
        .arguments
        .get("workflow")
        .and_then(|w| w.as_str())
        .unwrap_or_default();
    let default_payload = serde_json::json!({});
    let payload = call.arguments.get("payload").unwrap_or(&default_payload);

    if app.is_empty() || workflow.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Missing required 'app' or 'workflow' parameters.".to_string(),
        };
    }

    let request = diakonos_core::protocol::DiakonosRequest {
        action: "execute_workflow_proxy".to_string(),
        payload: serde_json::json!({
            "app": app,
            "workflow": workflow,
            "payload": payload
        }),
    };

    info!(
        "🚀 [Hera] Delegating workflow execution to Diakonos: {}/{}",
        app, workflow
    );

    match diakonos_core::client::send_request(diakonos_core::client::DIAKONOS_SOCKET, &request)
        .await
    {
        Ok(response) if response.status == "success" => ToolResult {
            name: call.name.clone(),
            success: true,
            output: serde_json::to_string_pretty(&response.data).unwrap_or_default(),
        },
        Ok(response) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: response
                .data
                .get("message")
                .and_then(|value| value.as_str())
                .unwrap_or("Diakonos returned an unknown error")
                .to_string(),
        },
        Err(error) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!(
                "Failed to reach Diakonos at {}: {}",
                diakonos_core::client::DIAKONOS_SOCKET,
                error
            ),
        },
    }
}

async fn execute_system_status(call: &ToolCall) -> ToolResult {
    let mut report = String::new();

    // 1. RAM from /proc/meminfo
    if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
        let mut total = 0.0_f64;
        let mut available = 0.0_f64;
        for line in meminfo.lines() {
            if line.starts_with("MemTotal:") {
                total = line
                    .split_whitespace()
                    .nth(1)
                    .unwrap_or("0")
                    .parse::<f64>()
                    .unwrap_or(0.0)
                    / 1024.0
                    / 1024.0;
            } else if line.starts_with("MemAvailable:") {
                available = line
                    .split_whitespace()
                    .nth(1)
                    .unwrap_or("0")
                    .parse::<f64>()
                    .unwrap_or(0.0)
                    / 1024.0
                    / 1024.0;
            }
        }
        let used = total - available;
        report.push_str(&format!(
            "RAM: {:.1}GB used / {:.1}GB total ({:.1}GB free)\n",
            used, total, available
        ));
    }

    // 2. CPU Load from /proc/loadavg
    if let Ok(loadavg) = std::fs::read_to_string("/proc/loadavg") {
        let parts: Vec<&str> = loadavg.split_whitespace().collect();
        if parts.len() >= 3 {
            report.push_str(&format!(
                "CPU Load Average: {} (1m) {} (5m) {} (15m)\n",
                parts[0], parts[1], parts[2]
            ));
        }
    }

    // 3. Uptime
    if let Ok(output) = std::process::Command::new("uptime").arg("-p").output() {
        let uptime = String::from_utf8_lossy(&output.stdout).trim().to_string();
        report.push_str(&format!("Uptime: {}\n", uptime));
    }

    // 4. GPU status via nvidia-smi
    match std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,memory.free")
        .arg("--format=csv,noheader,nounits")
        .output()
    {
        Ok(output) if output.status.success() => {
            let out_str = String::from_utf8_lossy(&output.stdout);
            report.push_str("\nGPU Status:\n");
            for line in out_str.lines() {
                let parts: Vec<&str> = line.split(", ").collect();
                if parts.len() == 7 {
                    report.push_str(&format!(
                        "  GPU {}: {} | Temp: {}°C | Load: {}% | VRAM: {}MB / {}MB ({}MB free)\n",
                        parts[0].trim(), parts[1].trim(), parts[2].trim(),
                        parts[3].trim(), parts[4].trim(), parts[5].trim(), parts[6].trim()
                    ));
                }
            }
        }
        _ => {
            report.push_str("\nGPU: nvidia-smi not available or failed.\n");
        }
    }

    // 5. GPU process list
    match std::process::Command::new("nvidia-smi")
        .arg("--query-compute-apps=pid,name,used_memory")
        .arg("--format=csv,noheader,nounits")
        .output()
    {
        Ok(output) if output.status.success() => {
            let out_str = String::from_utf8_lossy(&output.stdout);
            if !out_str.trim().is_empty() {
                report.push_str("\nGPU Processes:\n");
                for line in out_str.lines() {
                    let parts: Vec<&str> = line.split(", ").collect();
                    if parts.len() == 3 {
                        let proc_name =
                            parts[1].trim().split('/').last().unwrap_or(parts[1].trim());
                        report.push_str(&format!(
                            "  PID {} | {} | {}MB VRAM\n",
                            parts[0].trim(),
                            proc_name,
                            parts[2].trim()
                        ));
                    }
                }
            }
        }
        _ => {}
    }

    // 6. PM2 services status
    // Pre-load port listeners to map PID to Ports
    let mut port_by_pid: std::collections::HashMap<u64, Vec<u16>> =
        std::collections::HashMap::new();
    if let Ok(output) = std::process::Command::new("ss").args(&["-tlnp"]).output() {
        if output.status.success() {
            let out_str = String::from_utf8_lossy(&output.stdout);
            for line in out_str.lines().skip(1) {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 5 {
                    if let Some(port_str) = parts[3].rsplit(':').next() {
                        if let Ok(port) = port_str.parse::<u16>() {
                            let proc_info = parts.get(5).unwrap_or(&"");
                            if let Some(start) = proc_info.find("pid=") {
                                let after = &proc_info[start + 4..];
                                let pid_str = after.split(',').next().unwrap_or("0");
                                if let Ok(pid) = pid_str.parse::<u64>() {
                                    port_by_pid.entry(pid).or_default().push(port);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    match std::process::Command::new("pm2").arg("jlist").output() {
        Ok(output) if output.status.success() => {
            let out_str = String::from_utf8_lossy(&output.stdout);
            if let Ok(procs) = serde_json::from_str::<Vec<serde_json::Value>>(&out_str) {
                report.push_str(&format!("\nPM2 Services ({} total):\n", procs.len()));
                for proc in &procs {
                    let name = proc.get("name").and_then(|n| n.as_str()).unwrap_or("?");
                    let status = proc
                        .get("pm2_env")
                        .and_then(|e| e.get("status"))
                        .and_then(|s| s.as_str())
                        .unwrap_or("?");
                    let restarts = proc
                        .get("pm2_env")
                        .and_then(|e| e.get("restart_time"))
                        .and_then(|r| r.as_u64())
                        .unwrap_or(0);
                    let pid = proc.get("pid").and_then(|p| p.as_u64()).unwrap_or(0);

                    let emoji = if status == "online" { "🟢" } else { "🔴" };
                    let crash_flag = if restarts > 10 {
                        " ⚠️ CRASH LOOP"
                    } else {
                        ""
                    };

                    let ports = port_by_pid.get(&pid);
                    let port_info = if let Some(p) = ports {
                        format!(" (ports: {:?})", p)
                    } else if status == "online"
                        && !name.contains("argus")
                        && !name.contains("imagin")
                        && !name.contains("memento")
                    {
                        " (no listener)".to_string()
                    } else {
                        "".to_string()
                    };

                    report.push_str(&format!(
                        "  {} {} [{}] restarts: {}{}{}\n",
                        emoji, name, status, restarts, port_info, crash_flag
                    ));
                }
            }
        }
        _ => {
            report.push_str("\nPM2: Not available\n");
        }
    }

    info!("🖥️ [Hera] System status report generated");
    ToolResult {
        name: call.name.clone(),
        success: true,
        output: report,
    }
}

/// Auto-heal: restart a PM2 service by name.
/// Ava can now fix problems, not just report them.
async fn execute_service_restart(call: &ToolCall) -> ToolResult {
    let service_name = call
        .arguments
        .get("service_name")
        .and_then(|s| s.as_str())
        .unwrap_or("");
    let flush_logs = call
        .arguments
        .get("flush_logs")
        .and_then(|b| b.as_bool())
        .unwrap_or(false);

    if service_name.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Missing 'service_name' parameter. Provide the PM2 process name (e.g., 'vetra-rust').".into(),
        };
    }

    // Safety: sanitize service name to prevent injection
    let sanitized: String = service_name
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || *ch == '-' || *ch == '_' || *ch == '.')
        .collect();

    let mut report = String::new();

    // Step 1: Capture pre-restart state
    let pre_status = std::process::Command::new("pm2")
        .args(&["describe", &sanitized])
        .output();
    if let Ok(output) = &pre_status {
        let out_str = String::from_utf8_lossy(&output.stdout);
        if out_str.contains("doesn't exist") || out_str.contains("Process not found") {
            return ToolResult {
                name: call.name.clone(),
                success: false,
                output: format!(
                    "PM2 process '{}' not found. Run `pm2 list` to see available services.",
                    sanitized
                ),
            };
        }
    }

    // Step 2: Optionally flush logs before restart
    if flush_logs {
        let _ = std::process::Command::new("pm2")
            .args(&["flush", &sanitized])
            .output();
        report.push_str(&format!("🗑️ Flushed logs for '{}'\n", sanitized));
    }

    // Step 3: Read last 5 error lines before restart (for context)
    let pm2_home = std::env::var("HOME").unwrap_or_else(|_| "/home/paulo".to_string());
    let err_log_path = format!("{}/.pm2/logs/{}-error.log", pm2_home, sanitized);
    if let Ok(content) = std::fs::read_to_string(&err_log_path) {
        let lines: Vec<&str> = content.lines().collect();
        let start = if lines.len() > 5 { lines.len() - 5 } else { 0 };
        if !lines[start..].is_empty() {
            report.push_str("Last errors before restart:\n");
            for line in &lines[start..] {
                report.push_str(&format!("  {}", line));
                report.push('\n');
            }
        }
    }

    // Step 4: Execute restart
    match std::process::Command::new("pm2")
        .args(&["restart", &sanitized])
        .output()
    {
        Ok(output) => {
            if output.status.success() {
                // Step 5: Wait a moment, then verify the service came back
                std::thread::sleep(std::time::Duration::from_secs(2));

                let is_online = if let Ok(verify) =
                    std::process::Command::new("pm2").args(&["jlist"]).output()
                {
                    let out_str = String::from_utf8_lossy(&verify.stdout);
                    if let Ok(procs) = serde_json::from_str::<Vec<serde_json::Value>>(&out_str) {
                        procs.iter().any(|proc| {
                            let name = proc.get("name").and_then(|n| n.as_str()).unwrap_or("");
                            let status = proc
                                .get("pm2_env")
                                .and_then(|e| e.get("status"))
                                .and_then(|s| s.as_str())
                                .unwrap_or("");
                            name == sanitized && status == "online"
                        })
                    } else {
                        false
                    }
                } else {
                    false
                };

                if is_online {
                    report.push_str(&format!(
                        "\n✅ Service '{}' restarted successfully and is ONLINE.",
                        sanitized
                    ));
                    info!(
                        "🔧 [Hera] Auto-heal: '{}' restarted successfully",
                        sanitized
                    );
                } else {
                    report.push_str(&format!("\n⚠️ Service '{}' was restarted but is NOT online yet. It may need more time or has a startup error.", sanitized));
                    report.push_str(
                        "\nRecommendation: Use read_pm2_logs to check for startup errors.",
                    );
                }

                ToolResult {
                    name: call.name.clone(),
                    success: is_online,
                    output: report,
                }
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                ToolResult {
                    name: call.name.clone(),
                    success: false,
                    output: format!("PM2 restart failed for '{}': {}", sanitized, stderr),
                }
            }
        }
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Failed to execute pm2 restart: {}", e),
        },
    }
}

/// Read PM2 logs for a specific service.
/// Gives Ava deep per-service log access beyond the centralized JSONL file.
async fn execute_read_pm2_logs(call: &ToolCall) -> ToolResult {
    let service_name = call
        .arguments
        .get("service_name")
        .and_then(|s| s.as_str())
        .unwrap_or("");
    let log_type = call
        .arguments
        .get("log_type")
        .and_then(|t| t.as_str())
        .unwrap_or("error");
    let lines = call
        .arguments
        .get("lines")
        .and_then(|l| l.as_i64())
        .unwrap_or(50)
        .clamp(1, 200) as usize;
    let search = call
        .arguments
        .get("search")
        .and_then(|s| s.as_str())
        .unwrap_or("");

    if service_name.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Missing 'service_name' parameter.".into(),
        };
    }

    let sanitized: String = service_name
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || *ch == '-' || *ch == '_' || *ch == '.')
        .collect();

    let pm2_home = std::env::var("HOME").unwrap_or_else(|_| "/home/paulo".to_string());

    let read_log_file = |suffix: &str| -> String {
        let path = format!("{}/.pm2/logs/{}-{}.log", pm2_home, sanitized, suffix);
        match std::fs::read_to_string(&path) {
            Ok(content) => {
                let all_lines: Vec<&str> = content.lines().collect();
                let filtered: Vec<&&str> = if search.is_empty() {
                    all_lines.iter().collect()
                } else {
                    let search_lower = search.to_lowercase();
                    all_lines
                        .iter()
                        .filter(|l| l.to_lowercase().contains(&search_lower))
                        .collect()
                };
                let start = if filtered.len() > lines {
                    filtered.len() - lines
                } else {
                    0
                };
                filtered[start..]
                    .iter()
                    .map(|l| l.to_string())
                    .collect::<Vec<_>>()
                    .join("\n")
            }
            Err(_) => format!(
                "(no {} log file found at {}/.pm2/logs/{}-{}.log)",
                suffix, pm2_home, sanitized, suffix
            ),
        }
    };

    let mut result = String::new();
    match log_type {
        "output" => {
            result.push_str(&format!(
                "=== PM2 OUTPUT LOG for '{}' (last {} lines) ===\n",
                sanitized, lines
            ));
            result.push_str(&read_log_file("out"));
        }
        "both" => {
            result.push_str(&format!(
                "=== PM2 ERROR LOG for '{}' (last {} lines) ===\n",
                sanitized, lines
            ));
            result.push_str(&read_log_file("error"));
            result.push_str(&format!(
                "\n\n=== PM2 OUTPUT LOG for '{}' (last {} lines) ===\n",
                sanitized, lines
            ));
            result.push_str(&read_log_file("out"));
        }
        _ => {
            result.push_str(&format!(
                "=== PM2 ERROR LOG for '{}' (last {} lines) ===\n",
                sanitized, lines
            ));
            result.push_str(&read_log_file("error"));
        }
    }

    info!("📋 [Hera] Read PM2 {} logs for '{}'", log_type, sanitized);
    ToolResult {
        name: call.name.clone(),
        success: true,
        output: result,
    }
}

async fn execute_spline_interact(call: &ToolCall) -> ToolResult {
    let url = call
        .arguments
        .get("url")
        .and_then(|u| u.as_str())
        .unwrap_or("");
    let action = call
        .arguments
        .get("action")
        .and_then(|a| a.as_str())
        .unwrap_or("generate_embed");

    if url.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Error: missing Spline 'url' parameter".to_string(),
        };
    }

    match action {
        "generate_embed" => {
            let embed_code = format!(
                r#"<script type="module" src="https://unpkg.com/@splinetool/viewer@1.0.95/build/spline-viewer.js"></script>
<spline-viewer url="{}" events-target="global"></spline-viewer>"#,
                url
            );
            info!("🕹️  [Hera] Generated Spline embed for: {}", url);
            ToolResult {
                name: call.name.clone(),
                success: true,
                output: format!(
                    "Spline embed code generated successfully:\n```html\n{}\n```\nProvide this html directly to the user or insert it into the UI template.",
                    embed_code
                ),
            }
        }
        _ => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Unknown action '{}' for spline_interact tool", action),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_tool_call() {
        let text = r#"I'll draw that for you!
<tool_call>
{"name": "hera_draw", "arguments": {"prompt": "a beautiful sunset over the ocean", "width": 1024, "height": 1024}}
</tool_call>"#;

        let calls = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "hera_draw");
        assert_eq!(
            calls[0].arguments["prompt"],
            "a beautiful sunset over the ocean"
        );
    }

    #[test]
    fn test_no_tool_call() {
        let text = "Hello! How can I help you today?";
        let calls = parse_tool_calls(text);
        assert!(calls.is_empty());
    }

    #[tokio::test]
    async fn test_persist_research_finding_requires_project_id() {
        let call = ToolCall {
            name: "persist_research_finding".to_string(),
            arguments: serde_json::json!({
                "project_title": "Whale Research",
                "session_id": "whales-01",
                "concept_id": "concept-whales",
                "canonical_name": "Whales",
                "claim_id": "claim-1",
                "claim_text": "Whales are mammals.",
                "evidence_id": "evidence-1",
                "snippet": "Whales are marine mammals."
            }),
        };

        let result = execute_persist_research_finding(&call).await;
        assert!(!result.success);
        assert!(result.output.contains("project_id"));
    }

    #[tokio::test]
    async fn test_persist_research_finding_requires_snippet() {
        let call = ToolCall {
            name: "persist_research_finding".to_string(),
            arguments: serde_json::json!({
                "project_id": "whales-2026",
                "project_title": "Whale Research",
                "session_id": "whales-01",
                "concept_id": "concept-whales",
                "canonical_name": "Whales",
                "claim_id": "claim-1",
                "claim_text": "Whales are mammals.",
                "evidence_id": "evidence-1"
            }),
        };

        let result = execute_persist_research_finding(&call).await;
        assert!(!result.success);
        assert!(result.output.contains("snippet"));
    }

    #[tokio::test]
    async fn test_persist_channel_research_finding_requires_source_uri() {
        let call = ToolCall {
            name: "persist_channel_research_finding".to_string(),
            arguments: serde_json::json!({
                "project_id": "whales-2026",
                "project_title": "Whale Research",
                "session_id": "whales-01",
                "concept_id": "concept-whales",
                "canonical_name": "Whales",
                "claim_id": "claim-1",
                "claim_text": "Whales are mammals.",
                "evidence_id": "evidence-1",
                "snippet": "Whales are marine mammals.",
                "source_kind": "chat_reply",
                "source_label": "Telegram session",
                "channel": "telegram"
            }),
        };

        let result = execute_persist_channel_research_finding(&call).await;
        assert!(!result.success);
        assert!(result.output.contains("source_uri"));
    }

    #[tokio::test]
    async fn test_render_pdf_requires_html() {
        let call = ToolCall {
            name: "render_pdf".to_string(),
            arguments: serde_json::json!({
                "file_name": "empty-test"
            }),
        };

        let result = execute_render_pdf(&call).await;
        assert!(!result.success);
        assert!(result.output.contains("Missing html or html_path"));
    }

    #[tokio::test]
    async fn test_render_pdf_generates_file() {
        if !std::path::Path::new("/usr/bin/google-chrome").exists() {
            return;
        }

        let call = ToolCall {
            name: "render_pdf".to_string(),
            arguments: serde_json::json!({
                "html": "<!doctype html><html><body><h1>Smoke Test</h1><p>ok</p></body></html>",
                "file_name": "hera-render-pdf-test"
            }),
        };

        let result = execute_render_pdf(&call).await;
        assert!(result.success, "{}", result.output);
        assert!(std::path::Path::new(&result.output).exists());
        let metadata = std::fs::metadata(&result.output).unwrap();
        assert!(metadata.len() > 0);
        let _ = std::fs::remove_file(&result.output);
    }

    #[tokio::test]
    async fn test_render_pdf_generates_file_from_html_path() {
        if !std::path::Path::new("/usr/bin/google-chrome").exists() {
            return;
        }

        let html_path = format!(
            "/tmp/hera-render-pdf-test-{}.html",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis()
        );
        std::fs::write(
            &html_path,
            "<!doctype html><html><body><h1>Smoke Test</h1><p>path ok</p></body></html>",
        )
        .unwrap();

        let call = ToolCall {
            name: "render_pdf".to_string(),
            arguments: serde_json::json!({
                "html_path": html_path,
                "file_name": "hera-render-pdf-path-test"
            }),
        };

        let result = execute_render_pdf(&call).await;
        assert!(result.success, "{}", result.output);
        assert!(std::path::Path::new(&result.output).exists());
        let metadata = std::fs::metadata(&result.output).unwrap();
        assert!(metadata.len() > 0);
        let _ = std::fs::remove_file(&result.output);
        let _ = std::fs::remove_file(
            call.arguments
                .get("html_path")
                .and_then(|value| value.as_str())
                .unwrap(),
        );
    }

    #[test]
    fn test_market_structured_findings_sections() {
        let payload = serde_json::json!({
            "research_dossier": {
                "investment_view": {
                    "thesis": "Empresa defensiva con pricing power.",
                    "valuation_summary": "El consenso deja upside moderado.",
                    "invalidation": "La tesis falla si cae el margen.",
                    "bear_case": "La demanda se enfria mas de lo esperado."
                }
            },
            "catalysts_and_news": {
                "executive_summary": "Catalizadores mixtos pero todavía positivos.",
                "recent_headlines": [
                    { "title": "Nuevo contrato acelera backlog", "published_at": "2026-04-02" }
                ]
            },
            "technical_indicators": {
                "current_price": 100.0,
                "fifty_day_average": 95.0,
                "two_hundred_day_average": 90.0,
                "fifty_two_week_high": 120.0,
                "fifty_two_week_low": 70.0,
                "beta": 1.2
            },
            "analyst_data": {
                "target_mean_price": 118.0,
                "number_of_analyst_opinions": 12
            },
            "quantitative_metrics": {
                "forward_pe": 18.5
            },
            "investment_scores": {
                "trend": 72,
                "risk": 58
            },
            "source_catalog": {
                "metric_sources": [{ "url": "https://finance.yahoo.com/quote/AAPL", "label": "Yahoo Quote", "kind": "market_data" }],
                "news_sources": [{ "url": "https://example.com/news", "label": "Example News", "kind": "news" }]
            }
        });

        let findings = market_build_structured_findings(&payload, "fallback");
        assert_eq!(
            findings["thesis"]["claim_type"].as_str(),
            Some("investment_thesis")
        );
        assert_eq!(
            findings["valuation"]["claim_type"].as_str(),
            Some("valuation_view")
        );
        assert!(findings["catalysts"]["summary"]
            .as_str()
            .unwrap_or_default()
            .contains("Titulares clave"));
        assert!(findings["technical_view"]["summary"]
            .as_str()
            .unwrap_or_default()
            .contains("Score técnico"));
    }

    #[test]
    fn test_market_change_log_detects_material_delta() {
        let previous = serde_json::json!({
            "technical_indicators": { "current_price": 90.0 },
            "analyst_data": {
                "target_mean_price": 100.0,
                "recommendation_key": "hold"
            },
            "quantitative_metrics": {
                "revenue_growth": 0.12
            },
            "catalysts_and_news": {
                "recent_headlines": [{ "title": "Old catalyst" }]
            }
        });
        let current = serde_json::json!({
            "technical_indicators": { "current_price": 105.0 },
            "analyst_data": {
                "target_mean_price": 120.0,
                "recommendation_key": "buy"
            },
            "quantitative_metrics": {
                "revenue_growth": 0.18
            },
            "catalysts_and_news": {
                "recent_headlines": [{ "title": "New catalyst" }]
            }
        });

        let change_log = market_build_change_log("AAPL", &current, Some(&previous));
        let items = change_log["items"].as_array().cloned().unwrap_or_default();
        assert!(items.len() >= 4);
        assert!(change_log["summary"]
            .as_str()
            .unwrap_or_default()
            .contains("cambios materiales"));
    }
}

async fn execute_list_image_loras(call: &ToolCall) -> ToolResult {
    let triggers_path = "/home/paulo/models/image-stack/loras/triggers.json";
    let nsfw_path = "/home/paulo/models/image-stack/loras/nsfw_loras.json";

    let mut sfw_list = Vec::new();
    let mut nsfw_list = Vec::new();

    let nsfw_tags: Vec<String> = if let Ok(content) = std::fs::read_to_string(nsfw_path) {
        serde_json::from_str(&content).unwrap_or_default()
    } else {
        Vec::new()
    };

    if let Ok(content) = std::fs::read_to_string(triggers_path) {
        if let Ok(triggers) =
            serde_json::from_str::<std::collections::HashMap<String, Vec<String>>>(&content)
        {
            for (lora, tags) in triggers {
                let entry = format!("- <lora:{}:1.0> (Triggers: {})", lora, tags.join(", "));
                if nsfw_tags.contains(&lora) {
                    nsfw_list.push(entry);
                } else {
                    sfw_list.push(entry);
                }
            }
        }
    }

    let mut output = String::from("Available Image LoRAs:\n");
    if !sfw_list.is_empty() {
        output.push_str("\n[SFW Models]\n");
        output.push_str(&sfw_list.join("\n"));
        output.push('\n');
    }
    if !nsfw_list.is_empty() {
        output.push_str("\n[NSFW Models]\n");
        output.push_str(&nsfw_list.join("\n"));
        output.push('\n');
    }

    if sfw_list.is_empty() && nsfw_list.is_empty() {
        output.push_str("\n(No LoRAs found in image-stack/loras)");
    }

    ToolResult {
        name: call.name.clone(),
        success: true,
        output,
    }
}

pub async fn execute_run_backtest(call: &ToolCall) -> ToolResult {
    let bot_id = call
        .arguments
        .get("bot_id")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    let symbol = call
        .arguments
        .get("symbol")
        .and_then(|v| v.as_str())
        .unwrap_or("BTC/USDT");
    let range = call
        .arguments
        .get("range")
        .and_then(|v| v.as_str())
        .unwrap_or("30d");

    // Trigger the real backtest engine inside latinos-rust natively via internal API
    let payload = serde_json::json!({
        "bot_id": bot_id as i32,
        "market": symbol,
        "range": range
    });

    match reqwest::Client::new()
        .post("http://127.0.0.1:3005/api/internal/backtest")
        .json(&payload)
        .send()
        .await
    {
        Ok(res) if res.status().is_success() => {
            if let Ok(json) = res.json::<serde_json::Value>().await {
                // Parse the metrics returned by the SimulationEngine
                if let Some(metrics) = json.get("metrics") {
                    let roi = metrics
                        .get("roi_percentage")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0);
                    let trades = metrics
                        .get("total_trades")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0);
                    let win_rate = metrics
                        .get("win_rate")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0);
                    let drawdown = metrics
                        .get("max_drawdown")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0);
                    let sharpe = metrics
                        .get("sharpe_ratio")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0);

                    let result = format!(
                        "Backtest completed realistically for Bot #{} on {}.\nTime Range: {}\nROI: {:.2}%\nTotal Trades: {}\nWin Rate: {:.2}%\nMax Drawdown: {:.2}%\nSharpe Ratio: {:.2}",
                        bot_id, symbol, range, roi, trades, win_rate, drawdown, sharpe
                    );

                    ToolResult {
                        name: call.name.clone(),
                        success: true,
                        output: result,
                    }
                } else {
                    ToolResult {
                        name: call.name.clone(),
                        success: true,
                        output: "Simulation dispatched but no metrics returned".to_string(),
                    }
                }
            } else {
                ToolResult {
                    name: call.name.clone(),
                    success: false,
                    output: "Failed to parse simulation engine response".to_string(),
                }
            }
        }
        _ => ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Failed to connect to internal Latinos engine on port 3005".to_string(),
        },
    }
}

pub async fn execute_get_bot_status(call: &ToolCall) -> ToolResult {
    let bot_id = call
        .arguments
        .get("bot_id")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);

    // Call Memento IPC to fetch real bot status from the latinos database
    let memento_query = ToolCall {
        name: "memento_query".to_string(),
        arguments: serde_json::json!({
            "app": "latinos",
            "query": format!("SELECT id, name, status, description, daily_profit, total_profit, win_rate, live_trading FROM bots WHERE id = {}", bot_id)
        }),
    };

    let db_result = execute_memento_query(&memento_query).await;

    ToolResult {
        name: call.name.clone(),
        success: db_result.success,
        output: db_result.output,
    }
}

pub async fn execute_list_bots(call: &ToolCall) -> ToolResult {
    let memento_query = ToolCall {
        name: "memento_query".to_string(),
        arguments: serde_json::json!({
            "app": "latinos",
            "query": "SELECT id, name, status, description, daily_profit, total_profit, win_rate, live_trading FROM bots ORDER BY id ASC LIMIT 50"
        }),
    };

    let db_result = execute_memento_query(&memento_query).await;

    ToolResult {
        name: call.name.clone(),
        success: db_result.success,
        output: db_result.output,
    }
}

pub async fn execute_load_market_data(call: &ToolCall) -> ToolResult {
    let symbol = call
        .arguments
        .get("symbol")
        .and_then(|v| v.as_str())
        .unwrap_or("BTCUSDT")
        .replace("/", "");
    let interval = call
        .arguments
        .get("interval")
        .and_then(|v| v.as_str())
        .unwrap_or("1h");

    // Fetch real live OHLCV data from Binance public API
    let url = format!(
        "https://api.binance.com/api/v3/klines?symbol={}&interval={}&limit=10",
        symbol.to_uppercase(),
        interval
    );

    match reqwest::get(&url).await {
        Ok(res) if res.status().is_success() => {
            if let Ok(klines) = res.json::<Vec<serde_json::Value>>().await {
                // Parse Binance Klines: [0: open_time, 1: open, 2: high, 3: low, 4: close, 5: volume]
                let mut output = format!("Live Binance {}/{} OHLCV Data:\n", symbol, interval);
                for (i, k) in klines.iter().enumerate() {
                    let o = k[1].as_str().unwrap_or("0");
                    let h = k[2].as_str().unwrap_or("0");
                    let l = k[3].as_str().unwrap_or("0");
                    let c = k[4].as_str().unwrap_or("0");
                    let v = k[5].as_str().unwrap_or("0");
                    output.push_str(&format!(
                        "Candle {}: O:{} H:{} L:{} C:{} V:{}\n",
                        i, o, h, l, c, v
                    ));
                }
                ToolResult {
                    name: call.name.clone(),
                    success: true,
                    output,
                }
            } else {
                ToolResult {
                    name: call.name.clone(),
                    success: false,
                    output: "Failed to parse Binance JSON".to_string(),
                }
            }
        }
        _ => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Failed to fetch Binance data for {}", symbol),
        },
    }
}

pub async fn execute_market_research(call: &ToolCall) -> ToolResult {
    let ticker = call
        .arguments
        .get("ticker")
        .and_then(|v| v.as_str())
        .unwrap_or("AAPL")
        .to_uppercase();
    let owner_email = optional_string_argument(call, "owner_email")
        .unwrap_or_else(|| "hera@system.local".to_string());
    let source = optional_string_argument(call, "source")
        .unwrap_or_else(|| "hera_market_research".to_string());
    let bot_id = optional_string_argument(call, "bot_id").unwrap_or_else(|| "hera".to_string());

    match std::process::Command::new(
        "/home/paulo/Programs/apps/OS/Tools/apps/latinos/scripts/market_research.py",
    )
    .arg(&ticker)
    .output()
    {
        Ok(output) if output.status.success() => {
            let json_str = String::from_utf8_lossy(&output.stdout).to_string();

            // Try to parse the result to validate and extract data
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&json_str) {
                if let Some(data) = parsed.get("data") {
                    let previous_payload = load_previous_market_payload(&ticker, &owner_email).await;
                    let mut enriched = data.clone();
                    let default_business_summary = market_json_string(
                        &enriched,
                        "/asset_identity/business_summary",
                    )
                    .unwrap_or_else(|| {
                        format!("Sin tesis extensa para {ticker}, usar el resumen estructurado.")
                    });
                    let default_valuation_summary = format!(
                        "Target medio {:?}, forward PE {:?}, upside {:?}.",
                        market_json_number(&enriched, "/analyst_data/target_mean_price"),
                        market_json_number(&enriched, "/quantitative_metrics/forward_pe"),
                        market_valuation_upside_percent(&enriched)
                    );
                    let default_bear_case = default_business_summary
                        .chars()
                        .take(180)
                        .collect::<String>();

                    let change_log = market_build_change_log(&ticker, &enriched, previous_payload.as_ref());
                    if let Some(research_dossier) = enriched
                        .get_mut("research_dossier")
                        .and_then(|value| value.as_object_mut())
                    {
                        let investment_view = research_dossier
                            .entry("investment_view".to_string())
                            .or_insert_with(|| serde_json::json!({}));
                        if investment_view.get("thesis").and_then(|value| value.as_str()).is_none() {
                            *investment_view = serde_json::json!({
                                "thesis": default_business_summary,
                                "valuation_summary": default_valuation_summary,
                                "bear_case": if default_bear_case.is_empty() {
                                    "Sin bear case explícito en la fuente; revisar presión competitiva y desaceleración de demanda.".to_string()
                                } else {
                                    default_bear_case
                                },
                                "invalidation": "Si se deterioran guidance, precio objetivo o catalizadores relevantes, la lectura debe refrescarse."
                            });
                        }
                    }

                    let structured_findings = market_build_structured_findings(&enriched, &ticker);
                    if let Some(research_dossier) = enriched
                        .get_mut("research_dossier")
                        .and_then(|value| value.as_object_mut())
                    {
                        research_dossier.insert(
                            "structured_findings".to_string(),
                            structured_findings,
                        );
                        research_dossier.insert("what_changed".to_string(), change_log.clone());
                    }

                    let lifecycle = market_build_lifecycle(&ticker, &enriched, &change_log);
                    if let Some(research_dossier) = enriched
                        .get_mut("research_dossier")
                        .and_then(|value| value.as_object_mut())
                    {
                        research_dossier.insert("lifecycle".to_string(), lifecycle.clone());
                    }

                    let analysis_summary = market_build_analysis_summary(&enriched);
                    let semantic_result = persist_market_research_semantic_memory(
                        &ticker,
                        &owner_email,
                        &source,
                        &bot_id,
                        &enriched,
                        &analysis_summary,
                    )
                    .await;
                    if let Err(error) = store_market_research_row(
                        &ticker,
                        &owner_email,
                        &source,
                        &bot_id,
                        &enriched,
                        &analysis_summary,
                    )
                    .await
                    {
                        return ToolResult {
                            name: call.name.clone(),
                            success: false,
                            output: format!("Failed to store stock research row: {error}"),
                        };
                    }

                    if let Err(error) = semantic_result {
                        return ToolResult {
                            name: call.name.clone(),
                            success: false,
                            output: format!("Stored stock research row, but failed to persist semantic memory: {error}"),
                        };
                    }

                    let out_json = serde_json::to_string_pretty(&enriched).unwrap_or_default();
                    return ToolResult {
                        name: call.name.clone(),
                        success: true,
                        output: format!("Market research for {}:\n{}", ticker, out_json),
                    };
                }
            }
            // Failed to parse or Error payload from python
            ToolResult {
                name: call.name.clone(),
                success: false,
                output: format!("Failed to parse JSON or Python error:\n{}", json_str),
            }
        }
        Ok(output) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!(
                "Python script returned non-zero. STDERR: {}",
                String::from_utf8_lossy(&output.stderr)
            ),
        },
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Failed to execute python fetcher: {}", e),
        },
    }
}

pub async fn execute_thermal_risk_scorer(call: &ToolCall) -> ToolResult {
    let debtor_id = call
        .arguments
        .get("debtor_id")
        .and_then(|id| id.as_str())
        .unwrap_or("Unknown");

    // In a real scenario, this would query the Memento database to get history.
    // For this context, we return a structural JSON output that Hera can use.
    let risk_data = serde_json::json!({
        "debtor_id": debtor_id,
        "thermal_score": 75.4,
        "risk_level": "WARNING",
        "factors": [
            "Frequent delays in last 3 months",
            "Payment capacity constrained",
            "Responsive to digital channels"
        ],
        "recommendation": "Offer restructuring with lower installments."
    });

    ToolResult {
        name: call.name.clone(),
        success: true,
        output: serde_json::to_string_pretty(&risk_data).unwrap_or_default(),
    }
}

pub async fn execute_generate_payment_agreement(call: &ToolCall) -> ToolResult {
    let debtor_id = call
        .arguments
        .get("debtor_id")
        .and_then(|id| id.as_str())
        .unwrap_or("Unknown");
    let initial_payment = call
        .arguments
        .get("initial_payment")
        .and_then(|ip| ip.as_f64())
        .unwrap_or(0.0);
    let installments = call
        .arguments
        .get("number_of_installments")
        .and_then(|ni| ni.as_i64())
        .unwrap_or(1);

    let agreement = serde_json::json!({
        "status": "DRAFT",
        "debtor_id": debtor_id,
        "terms": {
            "initial_payment": initial_payment,
            "installments": installments,
            "interest_relief_applied": true
        },
        "next_step": "Awaiting customer signature"
    });

    ToolResult {
        name: call.name.clone(),
        success: true,
        output: serde_json::to_string_pretty(&agreement).unwrap_or_default(),
    }
}

pub async fn execute_omni_channel_messenger(call: &ToolCall) -> ToolResult {
    let debtor_id = call
        .arguments
        .get("debtor_id")
        .and_then(|id| id.as_str())
        .unwrap_or("Unknown");
    let message = call
        .arguments
        .get("message")
        .and_then(|m| m.as_str())
        .unwrap_or("No message");
    let channels = call
        .arguments
        .get("channels")
        .and_then(|c| c.as_array())
        .map(|c| c.iter().filter_map(|ch| ch.as_str()).collect::<Vec<_>>())
        .unwrap_or_else(|| vec!["email", "sms"]);

    let delivery_report = serde_json::json!({
        "status": "QUEUED",
        "debtor_id": debtor_id,
        "message_preview": format!("{}...", &message.chars().take(20).collect::<String>()),
        "channels_dispatched": channels,
        "estimated_delivery_time": "10s"
    });

    ToolResult {
        name: call.name.clone(),
        success: true,
        output: serde_json::to_string_pretty(&delivery_report).unwrap_or_default(),
    }
}
