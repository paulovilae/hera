//! Hera Tool Executor — Sovereign Tool Calling for ImagineOS
//!
//! Defines tool schemas in Qwen's native format, parses `<tool_call>` blocks
//! from Qwen output, and dispatches tool execution to existing Hera methods.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};
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
    agent_id: String,
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
                        tracing::debug!("Skipping tool due to consumer restriction: {:?}", entry_path);
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
            tool_name = value.trim().trim_matches('"').trim_matches('\'').to_string();
        } else if let Some(value) = trimmed.strip_prefix("description:") {
            description = value.trim().trim_matches('"').trim_matches('\'').to_string();
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
    let agent_path = PathBuf::from(format!("/home/paulo/Programs/apps/OS/Agents/{}.md", sanitized));
    let persona = std::fs::read_to_string(&agent_path)
        .unwrap_or_else(|_| format!("You are an expert {}", sanitized));
    AgentArtifact {
        agent_id: sanitized,
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

    format!(r#"

# Tools

You may call one or more functions to assist with the user query.

You are provided with function definitions below:

{tools_json}

For each function call, return a JSON object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": "function_name", "arguments": {{"arg1": "value1"}}}}
</tool_call>"#)
}

/// Parse `<tool_call>` blocks from Qwen's text output.
/// Returns empty vec if no tool calls found.
pub fn parse_tool_calls(text: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();

    // Find all <tool_call>...</tool_call> blocks
    let mut search_from = 0;
    while let Some(start) = text[search_from..].find("<tool_call>") {
        let abs_start = search_from + start + "<tool_call>".len();
        if let Some(end) = text[abs_start..].find("</tool_call>") {
            let abs_end = abs_start + end;
            let json_str = text[abs_start..abs_end].trim();

            // Try to parse as JSON
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
                        info!("🔧 [Hera] Parsed tool call: {} with args: {}",
                            name, serde_json::to_string(args).unwrap_or_default());
                    }
                }
                Err(e) => {
                    // Try a more lenient parse — Qwen sometimes outputs slightly malformed JSON
                    tracing::warn!("⚠️ [Hera] Failed to parse tool_call JSON: {} — raw: {}", e, json_str);
                }
            }
            search_from = abs_end + "</tool_call>".len();
        } else {
            break;
        }
    }

    // Fallback: If no tags were found, maybe the model just spit out raw JSON?
    if calls.is_empty() {
        let trimmed = text.trim();
        if trimmed.starts_with('{') {
            // Try extracting the first complete JSON object
            let mut brace_count = 0;
            let mut end_idx = 0;
            for (i, c) in trimmed.char_indices() {
                if c == '{' { brace_count += 1; }
                else if c == '}' { brace_count -= 1; }
                
                if brace_count == 0 && i > 0 {
                    end_idx = i + 1;
                    break;
                }
            }
            
            if end_idx > 0 {
                let json_str = &trimmed[..end_idx];
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(json_str) {
                    if let (Some(name), Some(args)) = (
                        val.get("name").and_then(|n| n.as_str()),
                        val.get("arguments"),
                    ) {
                        calls.push(ToolCall {
                            name: name.to_string(),
                            arguments: args.clone(),
                        });
                        info!("🔧 [Hera] Parsed RAW JSON tool call: {} with args: {}",
                            name, serde_json::to_string(args).unwrap_or_default());
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
pub fn detect_intent_from_user_message(user_msg: &str, assistant_last: Option<&str>) -> Option<ToolCall> {
    let lower = user_msg.to_lowercase();
    let lower_trimmed = lower.trim();

    // Contextual image modifier detection
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
        "draw ", "dibuja ", "genera una imagen ", "create an image ", "make an image ",
        "generate an image ", "draw me ", "hazme un dibujo", "pinta ",
        "haz una imagen", "genera imagen", "crea una imagen",
        "make a picture", "generate a picture", "create a picture",
        "haz un dibujo ", "make me an image ", "draw a ",
        "hazme una foto ", "toma una foto ", "manda una foto ",
        "hazme una imagen ", "genera una foto ", "crea una foto ",
        "take a photo ", "send a photo ", "a picture of ",
        "make me a picture ", "send me an image ", "show me an image ",
        "haz una foto ", "a photo of ", "make a photo ", "create a photo ", "foto de ",
        "do a photo ", "do a picture ", "do a foto ", "do an image ",
    ];
    
    // Short exact matches
    let exact_matches = [
        "tu foto", "una foto", "mi foto", "dame foto", "dame una foto",
        "una imagen", "mi imagen", "tu imagen",
        "your photo", "my photo", "selfie", "retrato",
    ];

    // Broad fuzzy detection: if a short message contains an image noun + an action verb, it's a draw request
    let image_nouns = ["photo", "foto", "picture", "imagen", "image", "drawing", "dibujo", "selfie", "retrato", "pic ", "pic."];
    let action_verbs = ["make", "do ", "create", "take", "send", "show", "generate", "haz", "genera", "crea", "toma", "manda", "dame", "hazme", "draw", "paint", "pinta", "dibuja", "quiero", "want"];

    let mut is_draw = false;
    
    if exact_starts.iter().any(|kw| lower_trimmed.starts_with(kw)) {
        is_draw = true;
    } else if user_msg.len() < 40 && exact_matches.iter().any(|kw| lower_trimmed == *kw || lower_trimmed.starts_with(kw)) {
        is_draw = true;
    } else if user_msg.len() < 80 {
        // Fuzzy: short message contains both an image noun and an action verb
        let has_noun = image_nouns.iter().any(|n| lower_trimmed.contains(n));
        let has_verb = action_verbs.iter().any(|v| lower_trimmed.contains(v));
        if has_noun && has_verb {
            is_draw = true;
        }
    }

    if is_draw {
        let prompt = user_msg.to_string();
        tracing::info!("🎯 [Hera] Strict intent detected: hera_draw from user message");
        return Some(ToolCall {
            name: "hera_draw".to_string(),
            arguments: serde_json::json!({"prompt": prompt}),
        });
    }

    // Search intent
    let search_keywords = [
        "busca ", "search ", "look up ", "google ", "find out ",
        "busca en internet", "search the web", "qué pasó con",
        "what happened with", "noticias de", "news about",
    ];
    if search_keywords.iter().any(|kw| lower.contains(kw)) {
        info!("🎯 [Hera] Intent detected: hera_search from user message");
        return Some(ToolCall {
            name: "hera_search".to_string(),
            arguments: serde_json::json!({"query": user_msg}),
        });
    }

    // Speak intent
    let speak_keywords = [
        "say out loud", "di en voz alta", "habla ", "speak ",
        "read aloud", "lee en voz alta", "genera audio",
    ];
    if speak_keywords.iter().any(|kw| lower.contains(kw)) {
        info!("🎯 [Hera] Intent detected: hera_speak from user message");
        return Some(ToolCall {
            name: "hera_speak".to_string(),
            arguments: serde_json::json!({"text": user_msg}),
        });
    }

    // Video intent
    let video_keywords = [
        "genera un video", "generate a video", "make a video",
        "create a video", "haz un video", "crea un video",
    ];
    if video_keywords.iter().any(|kw| lower.contains(kw)) {
        info!("🎯 [Hera] Intent detected: hera_video from user message");
        return Some(ToolCall {
            name: "hera_video".to_string(),
            arguments: serde_json::json!({"prompt": user_msg}),
        });
    }

    // System status intent — catch conversational questions about server/GPU/CPU/memory
    let status_keywords = [
        "system status", "server status", "gpu status", "cpu status",
        "how is the server", "como esta el server", "como está el server",
        "estado del servidor", "estado del server", "status del server",
        "nvidia", "vram", "gpu load", "gpu temp",
        "como esta el gpu", "como está el gpu",
        "cuanta ram", "cuánta ram", "how much ram",
        "memory usage", "uso de memoria",
        "system health", "server health",
        "how much vram", "cuanta vram", "cuánta vram",
        "que procesos", "qué procesos", "what processes",
        "esta corriendo", "está corriendo", "is running",
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
        "ask_user" => execute_ask_user(call).await,
        "get_system_time" => execute_get_system_time(call).await,
        "system_status" => execute_system_status(call).await,
        "run_code" => execute_run_code(call).await,
        "web_scraper" => execute_web_scraper(call).await,
        "write_file" => execute_write_file(call).await,
        "generate_qr_code" => execute_generate_qr_code(call).await,
        "get_map_route" => execute_get_map_route(call).await,
        "execute_workflow" => execute_workflow(call).await,
        "movilo_search_providers" => execute_movilo_search_providers(call).await,
        "movilo_check_affiliation" => execute_movilo_check_affiliation(call).await,
        "movilo_validate_qr" => execute_movilo_validate_qr(call).await,
        "bind_telegram_workspace" => execute_bind_telegram_workspace(call).await,
        "spline_interact" => execute_spline_interact(call).await,
        _ => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Unknown tool: {}", call.name),
        },
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
            output: "Error: must provide 'sender_id', 'workspace_user', and 'workspace_company'.".into(),
        };
    }

    let path = "/home/paulo/Programs/apps/OS/etc/imaginclaw/vetra_telegram_bindings.json";
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or(0);
    let mut store = match std::fs::read_to_string(path) {
        Ok(raw) => serde_json::from_str::<Value>(&raw).unwrap_or_else(|_| serde_json::json!({ "bindings": [] })),
        Err(_) => serde_json::json!({ "bindings": [] }),
    };

    let bindings = store
        .get_mut("bindings")
        .and_then(|value| value.as_array_mut())
        .expect("bindings array should exist");

    let key_bot = bot_name.to_lowercase();
    if let Some(existing) = bindings.iter_mut().find(|item| {
        item.get("bot_name").and_then(|value| value.as_str()).map(|value| value.eq_ignore_ascii_case(&key_bot)).unwrap_or(false)
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
        output: format!("Could not find a SKILL.md playbook defining the skill '{}'", call.name),
    }
}

async fn execute_spawn_parallel_agents(call: &ToolCall) -> ToolResult {
    let agents = call.arguments.get("agents").and_then(|a| a.as_array()).cloned().unwrap_or_default();
    let prompt = call.arguments.get("prompt").and_then(|p| p.as_str()).unwrap_or("");

    if agents.is_empty() || prompt.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Error: must provide 'agents' array and 'prompt' string.".into()
        };
    }

    info!("👯 [Hera] Spawning {} parallel agents for task: {}", agents.len(), prompt);

    let mut tasks = Vec::new();
    for agent_val in agents {
        let agent_name = agent_val.as_str().unwrap_or("").to_string();
        if agent_name.is_empty() { continue; }
        
        let p = prompt.to_string();

        tasks.push(tokio::spawn(async move {
            let agent = load_agent_artifact(&agent_name);
            let persona = agent.persona;
            
            // Connect to local inference engine via Hera chat
            let hera = hera_execution::agents::hera::Hera::new("http://127.0.0.1:3000");
            
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
                        let content = json.get("choices")
                            .and_then(|c| c.as_array())
                            .and_then(|a| a.first())
                            .and_then(|c| c.get("message"))
                            .and_then(|m| m.get("content"))
                            .and_then(|c| c.as_str())
                            .unwrap_or("No response")
                            .to_string();
                        format!("--- REPORT FROM {} ---\n{}\n", agent_name.to_uppercase(), content)
                    } else {
                        format!("--- REPORT FROM {} ---\nFailed to parse JSON response.\n", agent_name.to_uppercase())
                    }
                }
                Err(e) => format!("--- REPORT FROM {} ---\nFailed to reach inference engine: {}\n", agent_name.to_uppercase(), e)
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
        output: format!("✅ Parallel Agents Execution Complete.\n\nCONSOLIDATED REPORTS:\n================================\n{}", combined_report),
    }
}

async fn execute_create_agent(call: &ToolCall) -> ToolResult {
    let name = call.arguments.get("name").and_then(|n| n.as_str()).unwrap_or("").trim();
    let persona = call.arguments.get("persona").and_then(|p| p.as_str()).unwrap_or("").trim();

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
            output: format!("Successfully created Agent Persona '{}' at {}", sanitized, save_path),
        },
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Failed to create agent {}: {}", sanitized, e),
        }
    }
}

async fn execute_create_skill(call: &ToolCall) -> ToolResult {
    let name = call.arguments.get("name").and_then(|n| n.as_str()).unwrap_or("").trim();
    let description = call.arguments.get("description").and_then(|d| d.as_str()).unwrap_or("").trim();
    let playbook = call.arguments.get("playbook").and_then(|p| p.as_str()).unwrap_or("").trim();

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
            output: format!("Successfully crafted Skill '{}' at {}\nIt will be dynamically loaded on the next request sequence.", sanitized, save_path),
        },
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Failed to write skill playbook {}: {}", save_path, e),
        }
    }
}

async fn execute_draw(call: &ToolCall) -> ToolResult {
    let prompt = call.arguments.get("prompt")
        .and_then(|p| p.as_str())
        .unwrap_or("A beautiful digital artwork");
    let width = call.arguments.get("width")
        .and_then(|w| w.as_u64())
        .map(|w| w as u32);
    let height = call.arguments.get("height")
        .and_then(|h| h.as_u64())
        .map(|h| h as u32);

    let hera = hera_execution::agents::hera::Hera::new("http://127.0.0.1:3000");
    // Dispatch Hera rendering execution directly to SwarmUI backend
    match hera.generate_image(prompt, None, width, height, None, None, None, None, None, None, None).await {
        Ok(res) => {
            let image_url = res.get("image_url")
                .and_then(|u| u.as_str())
                .unwrap_or("(no URL)");
            info!("🎨 [Hera] Image generated: {}", image_url);

            // Build a public URL that candle-core serves at /outputs/{filename}
            // The filename is the last segment of image_url (e.g., "/outputs/hera_drawn_UUID.png")
            let filename = image_url.split('/').last().unwrap_or(image_url);
            let public_url = format!("https://imaginos.ai/outputs/{}", filename);
            let response = format!("Image generated successfully!\nMEDIA: {}\nInclude this MEDIA line EXACTLY as-is in your reply so the image is delivered inline.", public_url);

            ToolResult {
                name: call.name.clone(),
                success: true,
                output: response,
            }
        }
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
    let query = call.arguments.get("query")
        .and_then(|q| q.as_str())
        .unwrap_or("");

    let hera = hera_execution::agents::hera::Hera::new("http://127.0.0.1:3000");

    match hera.native_web_search(query).await {
        Ok(results) => {
            info!("🌐 [Hera] Search completed for: {}", query);
            ToolResult {
                name: call.name.clone(),
                success: true,
                output: format!("Search results for '{}':\n{}", query, results),
            }
        }
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Search failed: {}", e),
        },
    }
}

async fn execute_speak(call: &ToolCall) -> ToolResult {
    let text = call.arguments.get("text")
        .and_then(|t| t.as_str())
        .unwrap_or("");
    let voice = call.arguments.get("voice")
        .and_then(|v| v.as_str());

    let hera = hera_execution::agents::hera::Hera::new("http://127.0.0.1:3000");

    match hera.synthesize_speech(text, voice).await {
        Ok(result) => {
            info!("🔊 [Hera] Speech synthesized");
            ToolResult {
                name: call.name.clone(),
                success: true,
                output: format!("Speech generated successfully: {}", serde_json::to_string(&result).unwrap_or_default()),
            }
        }
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("TTS failed: {}", e),
        },
    }
}

async fn execute_video(call: &ToolCall) -> ToolResult {
    let prompt = call.arguments.get("prompt")
        .and_then(|p| p.as_str())
        .unwrap_or("A smooth cinematic video");

    let hera = hera_execution::agents::hera::Hera::new("http://127.0.0.1:3000");

    match hera.synthesize_video(prompt).await {
        Ok(result) => {
            info!("🎬 [Hera] Video generated");
            ToolResult {
                name: call.name.clone(),
                success: true,
                output: format!("Video generated successfully: {}", serde_json::to_string(&result).unwrap_or_default()),
            }
        }
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Video generation failed: {}", e),
        },
    }
}

async fn execute_read_file(call: &ToolCall) -> ToolResult {
    let path = call.arguments.get("path")
        .and_then(|p| p.as_str())
        .unwrap_or("");

    match std::fs::read_to_string(path) {
        Ok(content) => {
            let truncated = if content.len() > 16000 {
                format!("{}... (truncated)", &content[..16000])
            } else {
                content
            };
            info!("📄 [Hera] Read file: {}", path);
            ToolResult {
                name: call.name.clone(),
                success: true,
                output: format!("File contents of '{}':\n{}", path, truncated),
            }
        }
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Failed to read file '{}': {}", path, e),
        },
    }
}

async fn execute_update_soul(call: &ToolCall) -> ToolResult {
    let new_soul_content = call.arguments.get("new_soul_content")
        .and_then(|c| c.as_str())
        .unwrap_or("");

    if new_soul_content.trim().is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Error: new_soul_content was empty. You must provide the complete new persona text.".to_string(),
        };
    }

    let soul_path = std::env::var("HERA_SOUL_PATH")
        .unwrap_or_else(|_| "/home/paulo/Programs/apps/imaginos/imaginclaw/persona/SOUL.md".to_string());

    match std::fs::write(&soul_path, new_soul_content) {
        Ok(_) => {
            tracing::info!("🧠 [Hera] SOUL successfully rewritten at {}", soul_path);
            ToolResult {
                name: call.name.clone(),
                success: true,
                output: format!("Successfully updated your SOUL! The changes have been saved to disk and you will remember them permanently."),
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
    let app = call.arguments.get("app")
        .and_then(|a| a.as_str())
        .unwrap_or("movilo");
    let query = call.arguments.get("query")
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
                                let rows = res.get("rows").cloned().unwrap_or(serde_json::json!([]));
                                
                                // Format results as readable text for the LLM
                                let formatted = serde_json::to_string_pretty(&rows).unwrap_or_default();
                                info!("🧠 [Memento] Got {} rows from '{}'", count, app);
                                ToolResult {
                                    name: call.name.clone(),
                                    success: true,
                                    output: format!("Database query returned {} results from '{}':\n{}", count, app, formatted),
                                }
                            } else {
                                let error = res.get("error")
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
            tracing::error!("🧠 [Memento] Failed to connect to /tmp/memento.sock: {:?}", e);
            ToolResult {
                name: call.name.clone(),
                success: false,
                output: format!("Memento is not running. Error: {}", e),
            }
        }
    }
}

async fn execute_movilo_search_providers(call: &ToolCall) -> ToolResult {
    let city = call.arguments.get("city").and_then(|c| c.as_str()).unwrap_or("");
    let specialty = call.arguments.get("specialty").and_then(|s| s.as_str()).unwrap_or("");
    let keyword = call.arguments.get("service_keyword").and_then(|k| k.as_str()).unwrap_or("");
    
    let mut conditions = vec!["p.status = 'Aprobado'".to_string()];
    if !city.is_empty() {
        conditions.push(format!("p.city ILIKE '%{}%'", city.replace("'", "''")));
    }
    if !specialty.is_empty() {
        conditions.push(format!("p.provider_type ILIKE '%{}%'", specialty.replace("'", "''")));
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
        })
    };
    
    let mut result = execute_memento_query(&memento_call).await;
    
    // Instruct the AI to render the map component based on the search context
    if result.success {
        let mut widget_attrs = String::new();
        if !specialty.is_empty() {
            widget_attrs.push_str(&format!(" category=\"{}\"", specialty.replace("\"", "\\\"")));
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
    let email = call.arguments.get("email").and_then(|e| e.as_str()).unwrap_or("");
    let doc = call.arguments.get("document").and_then(|d| d.as_str()).unwrap_or("");
    
    if email.is_empty() && doc.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Debes proveer un email o documento para buscar la afiliación.".into()
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
        })
    };
    execute_memento_query(&memento_call).await
}

async fn execute_movilo_validate_qr(call: &ToolCall) -> ToolResult {
    let qr_content = call.arguments.get("qr_content").and_then(|q| q.as_str()).unwrap_or("");
    
    if qr_content.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "QRCode content is missing.".into()
        };
    }
    
    // Asumimos que el QR emitido por Movilo tiene el User UUID o el Email
    let query = format!(
        "SELECT id, name, email, status, plan FROM movilo_users WHERE id = '{}' OR email = '{}' LIMIT 1",
        qr_content.replace("'", "''"), qr_content.replace("'", "''")
    );

    let memento_call = ToolCall {
        name: "memento_query".to_string(),
        arguments: serde_json::json!({
            "app": "movilo",
            "query": query
        })
    };
    
    let db_result = execute_memento_query(&memento_call).await;
    if db_result.success && db_result.output.contains("rows") && !db_result.output.contains("[]") {
        ToolResult {
            name: call.name.clone(),
            success: true,
            output: format!("¡QR Validado Exitosamente! Datos del afiliado recuperados:\n{}", db_result.output)
        }
    } else {
        ToolResult {
            name: call.name.clone(),
            success: false,
            output: "QR Inválido o usuario no encontrado en la base de datos de Movilo.".into()
        }
    }
}


async fn execute_api_request(call: &ToolCall) -> ToolResult {
    let method = call.arguments.get("method").and_then(|m| m.as_str()).unwrap_or("GET");
    let url = call.arguments.get("url").and_then(|u| u.as_str()).unwrap_or("");
    let headers_str = call.arguments.get("headers").and_then(|h| h.as_str()).unwrap_or("{}");
    let body_str = call.arguments.get("body").and_then(|b| b.as_str()).unwrap_or("");

    if url.is_empty() { return ToolResult { name: call.name.clone(), success: false, output: "Missing URL".into() }; }

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
                Ok(text) => ToolResult { name: call.name.clone(), success: status.is_success(), output: format!("Status: {}\nBody: {}", status, text) },
                Err(e) => ToolResult { name: call.name.clone(), success: false, output: format!("Failed to read response body: {}", e) },
            }
        }
        Err(e) => ToolResult { name: call.name.clone(), success: false, output: format!("Request failed: {}", e) },
    }
}

async fn execute_git_manager(call: &ToolCall) -> ToolResult {
    let command = call.arguments.get("command").and_then(|c| c.as_str()).unwrap_or("");
    let repo_path = call.arguments.get("repo_path").and_then(|p| p.as_str()).unwrap_or("");

    if repo_path.is_empty() || command.is_empty() {
         return ToolResult { name: call.name.clone(), success: false, output: "Missing command or repo_path".into() };
    }

    let args: Vec<&str> = command.split_whitespace().collect();
    match std::process::Command::new("git").current_dir(repo_path).args(&args).output() {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let success = output.status.success();
            let res = if success { format!("{}", stdout) } else { format!("Error: {}\n{}", stderr, stdout) };
            ToolResult { name: call.name.clone(), success, output: res }
        }
        Err(e) => ToolResult { name: call.name.clone(), success: false, output: format!("Failed to run git: {}", e) }
    }
}

async fn execute_memento_vector_search(call: &ToolCall) -> ToolResult {
    let query = call.arguments.get("query").and_then(|q| q.as_str()).unwrap_or("");
    let limit = call.arguments.get("limit").and_then(|l| l.as_u64()).unwrap_or(3);

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
                return ToolResult { name: call.name.clone(), success: false, output: "IPC Write Failed".into() };
            }
            let mut buffer = vec![0u8; 65536];
            match stream.read(&mut buffer).await {
                Ok(n) if n > 0 => {
                    let response_str = String::from_utf8_lossy(&buffer[..n]);
                    ToolResult { name: call.name.clone(), success: true, output: response_str.to_string() }
                }
                _ => ToolResult { name: call.name.clone(), success: false, output: "No response".into() },
            }
        }
        Err(e) => ToolResult { name: call.name.clone(), success: false, output: format!("Memento socket error: {}", e) },
    }
}

async fn execute_ask_user(call: &ToolCall) -> ToolResult {
    let question = call.arguments.get("question").and_then(|q| q.as_str()).unwrap_or("Needs human input.");
    tracing::info!("⏸️ [Hera] Pausing flow to ask user: {}", question);
    ToolResult {
        name: call.name.clone(),
        success: true,
        output: format!("[PAUSED_FOR_USER] Question: {}", question),
    }
}

async fn execute_get_system_time(call: &ToolCall) -> ToolResult {
    match std::process::Command::new("date").output() {
        Ok(out) => ToolResult { name: call.name.clone(), success: true, output: String::from_utf8_lossy(&out.stdout).to_string() },
        Err(e) => ToolResult { name: call.name.clone(), success: false, output: e.to_string() }
    }
}

async fn execute_run_code(call: &ToolCall) -> ToolResult {
    let lang = call.arguments.get("language").and_then(|l| l.as_str()).unwrap_or("python");
    let code = call.arguments.get("code").and_then(|c| c.as_str()).unwrap_or("");
    let packages: Vec<String> = call.arguments.get("packages")
        .and_then(|p| p.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();
    
    let timestamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis();
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
            return ToolResult { name: call.name.clone(), success: false, output: format!("Failed to create Rust bean directory: {}", e) };
        }
        
        let mut deps = format!(r#"[dependencies]
tokio = {{ version = "1", features = ["full", "rt-multi-thread"] }}
reqwest = {{ version = "0.11", features = ["json"] }}
serde = {{ version = "1.0", features = ["derive"] }}
serde_json = "1.0"
"#);
        for pkg in &packages {
            deps.push_str(&format!("{} = \"*\"\n", pkg));
        }

        let cargo_toml = format!(r#"[package]
name = "{}"
version = "0.1.0"
edition = "2021"

{}
"#, bean_name, deps);
        
        std::fs::write(format!("{}/Cargo.toml", project_dir), cargo_toml).unwrap_or_default();
        std::fs::write(format!("{}/src/main.rs", project_dir), code).unwrap_or_default();
        
        match std::process::Command::new("cargo")
            .arg("run")
            .arg("--release")
            .current_dir(&project_dir)
            .output() {
            Ok(out) => {
                let out_str = String::from_utf8_lossy(&out.stdout).to_string();
                let err_str = String::from_utf8_lossy(&out.stderr).to_string();
                let success = out.status.success();
                
                let mut final_out = format!("RUST ROASTED BEAN EXECUTION:\n---\nSTDOUT:\n{}\n", out_str);
                if !success || !err_str.is_empty() {
                    final_out.push_str(&format!("---\nSTDERR (or cargo compilation logs):\n{}\n", err_str));
                }
                final_out.push_str(&format!("---\nBean saved permanently in {} and recorded in Memento.", project_dir));
                
                ToolResult { name: call.name.clone(), success, output: final_out }
            }
            Err(e) => ToolResult { name: call.name.clone(), success: false, output: format!("Cargo execution failed: {}", e) }
        }
    } else if lang.to_lowercase() == "python" {
        let p = format!("{}/{}.py", beans_dir, bean_name);
        if let Err(e) = std::fs::write(&p, code) { return ToolResult { name: call.name.clone(), success: false, output: format!("Failed to write Python bean: {}", e) }; }

        let mut pip_log = String::new();
        if !packages.is_empty() {
            let mut cmd = std::process::Command::new("python3");
            cmd.arg("-m").arg("pip").arg("install").arg("--break-system-packages");
            for pkg in &packages {
                cmd.arg(pkg);
            }
            if let Ok(out) = cmd.output() {
                if !out.status.success() {
                    let err = String::from_utf8_lossy(&out.stderr);
                    return ToolResult { name: call.name.clone(), success: false, output: format!("Failed to install Python packages:\n{}", err) };
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
                    if err_str.trim().is_empty() { out_str } else { format!("STDOUT:\n{}\n---\nSTDERR:\n{}", out_str, err_str) }
                } else { 
                    format!("PYTHON SOFT BEAN ERROR:\n{}\n{}", err_str, out_str) 
                };
                if !pip_log.is_empty() {
                    res = format!("{}\n---\n{}", pip_log, res);
                }
                res = format!("{}\n---\nBean saved permanently at {} and logged in Memento.", res, p);
                ToolResult { name: call.name.clone(), success, output: res }
            }
            Err(e) => ToolResult { name: call.name.clone(), success: false, output: e.to_string() }
        }
    } else {
        ToolResult { name: call.name.clone(), success: false, output: format!("Language '{}' not supported. Use 'rust' or 'python'.", lang) }
    }
}

async fn execute_write_file(call: &ToolCall) -> ToolResult {
    let path = call.arguments.get("path").and_then(|p| p.as_str()).unwrap_or("");
    let content = call.arguments.get("content").and_then(|c| c.as_str()).unwrap_or("");

    if path.is_empty() { return ToolResult { name: call.name.clone(), success: false, output: "Missing path".into() }; }

    match std::fs::write(path, content) {
        Ok(_) => ToolResult { name: call.name.clone(), success: true, output: format!("Successfully wrote to {}", path) },
        Err(e) => ToolResult { name: call.name.clone(), success: false, output: format!("Failed to write file: {}", e) },
    }
}

async fn execute_web_scraper(call: &ToolCall) -> ToolResult {
    let url = call.arguments.get("url").and_then(|u| u.as_str()).unwrap_or("");
    if url.is_empty() { return ToolResult { name: call.name.clone(), success: false, output: "Missing url".into() }; }

    match reqwest::get(url).await {
        Ok(res) => {
            match res.text().await {
                Ok(text) => {
                    let max_len = 16000;
                    let trunc = if text.len() > max_len { format!("{}... (truncated)", &text[..max_len]) } else { text };
                    ToolResult { name: call.name.clone(), success: true, output: trunc }
                }
                Err(e) => ToolResult { name: call.name.clone(), success: false, output: e.to_string() }
            }
        }
        Err(e) => ToolResult { name: call.name.clone(), success: false, output: e.to_string() }
    }
}

async fn execute_generate_qr_code(call: &ToolCall) -> ToolResult {
    let content = call.arguments.get("content").and_then(|c| c.as_str()).unwrap_or("");
    if content.is_empty() { return ToolResult { name: call.name.clone(), success: false, output: "Missing content".into() }; }

    // Using a quick external API for now, could be replaced with a local Rust crate later
    let url = format!("https://api.qrserver.com/v1/create-qr-code/?size=500x500&data={}", urlencoding::encode(content));
    info!("🔲 [Hera] Generated QR Code URL: {}", url);
    ToolResult {
        name: call.name.clone(),
        success: true,
        output: format!("QR Code generated successfully. Use this exact line in your reply to deliver it inline:\nMEDIA: {}", url)
    }
}

async fn execute_get_map_route(call: &ToolCall) -> ToolResult {
    let dest = call.arguments.get("destination").and_then(|d| d.as_str()).unwrap_or("");
    let orig = call.arguments.get("origin").and_then(|o| o.as_str()).unwrap_or("");
    
    if dest.is_empty() { return ToolResult { name: call.name.clone(), success: false, output: "Missing destination".into() }; }

    let url = if orig.is_empty() {
        format!("https://www.google.com/maps/search/?api=1&query={}", urlencoding::encode(dest))
    } else {
        format!("https://www.google.com/maps/dir/?api=1&origin={}&destination={}", urlencoding::encode(orig), urlencoding::encode(dest))
    };

    info!("🗺️ [Hera] Generated Google Maps URL: {}", url);
    ToolResult {
        name: call.name.clone(),
        success: true,
        output: format!("Maps link generated successfully:\n{}", url)
    }
}

async fn execute_workflow(call: &ToolCall) -> ToolResult {
    let app = call.arguments.get("app").and_then(|a| a.as_str()).unwrap_or_default();
    let workflow = call.arguments.get("workflow").and_then(|w| w.as_str()).unwrap_or_default();
    let default_payload = serde_json::json!({});
    let payload = call.arguments.get("payload").unwrap_or(&default_payload);

    if app.is_empty() || workflow.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Missing required 'app' or 'workflow' parameters.".to_string(),
        };
    }

    let url = format!("http://127.0.0.1:3006/execute/{}/{}", app, workflow);
    info!("🚀 [Hera] Proxying Workflow Execution to Argus: {}", url);

    let client = reqwest::Client::new();
    match client.post(&url).json(&payload).send().await {
        Ok(res) => {
            if res.status().is_success() {
                if let Ok(json) = res.json::<serde_json::Value>().await {
                    ToolResult {
                        name: call.name.clone(),
                        success: true,
                        output: serde_json::to_string_pretty(&json).unwrap_or_default()
                    }
                } else {
                    ToolResult {
                        name: call.name.clone(),
                        success: true,
                        output: "Workflow executed but failed to parse Argus JSON response".to_string()
                    }
                }
            } else {
                ToolResult {
                    name: call.name.clone(),
                    success: false,
                    output: format!("Argus returned status code {}", res.status())
                }
            }
        },
        Err(e) => {
            ToolResult {
                name: call.name.clone(),
                success: false,
                output: format!("Failed to reach Argus at {}: {}", url, e)
            }
        }
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
                total = line.split_whitespace().nth(1).unwrap_or("0").parse::<f64>().unwrap_or(0.0) / 1024.0 / 1024.0;
            } else if line.starts_with("MemAvailable:") {
                available = line.split_whitespace().nth(1).unwrap_or("0").parse::<f64>().unwrap_or(0.0) / 1024.0 / 1024.0;
            }
        }
        let used = total - available;
        report.push_str(&format!("RAM: {:.1}GB used / {:.1}GB total ({:.1}GB free)\n", used, total, available));
    }

    // 2. CPU Load from /proc/loadavg
    if let Ok(loadavg) = std::fs::read_to_string("/proc/loadavg") {
        let parts: Vec<&str> = loadavg.split_whitespace().collect();
        if parts.len() >= 3 {
            report.push_str(&format!("CPU Load Average: {} (1m) {} (5m) {} (15m)\n", parts[0], parts[1], parts[2]));
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
                        let proc_name = parts[1].trim().split('/').last().unwrap_or(parts[1].trim());
                        report.push_str(&format!(
                            "  PID {} | {} | {}MB VRAM\n",
                            parts[0].trim(), proc_name, parts[2].trim()
                        ));
                    }
                }
            }
        }
        _ => {}
    }

    // 6. PM2 services status
    match std::process::Command::new("pm2").arg("jlist").output() {
        Ok(output) if output.status.success() => {
            let out_str = String::from_utf8_lossy(&output.stdout);
            if let Ok(procs) = serde_json::from_str::<Vec<serde_json::Value>>(&out_str) {
                report.push_str(&format!("\nPM2 Services ({} total):\n", procs.len()));
                for proc in &procs {
                    let name = proc.get("name").and_then(|n| n.as_str()).unwrap_or("?");
                    let status = proc.get("pm2_env")
                        .and_then(|e| e.get("status"))
                        .and_then(|s| s.as_str())
                        .unwrap_or("?");
                    let restarts = proc.get("pm2_env")
                        .and_then(|e| e.get("restart_time"))
                        .and_then(|r| r.as_u64())
                        .unwrap_or(0);
                    let emoji = if status == "online" { "🟢" } else { "🔴" };
                    report.push_str(&format!("  {} {} [{}] restarts: {}\n", emoji, name, status, restarts));
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

async fn execute_spline_interact(call: &ToolCall) -> ToolResult {
    let url = call.arguments.get("url")
        .and_then(|u| u.as_str())
        .unwrap_or("");
    let action = call.arguments.get("action")
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
            let embed_code = format!(r#"<script type="module" src="https://unpkg.com/@splinetool/viewer@1.0.95/build/spline-viewer.js"></script>
<spline-viewer url="{}" events-target="global"></spline-viewer>"#, url);
            info!("🕹️  [Hera] Generated Spline embed for: {}", url);
            ToolResult {
                name: call.name.clone(),
                success: true,
                output: format!("Spline embed code generated successfully:\n```html\n{}\n```\nProvide this html directly to the user or insert it into the UI template.", embed_code),
            }
        },
        _ => {
            ToolResult {
                name: call.name.clone(),
                success: false,
                output: format!("Unknown action '{}' for spline_interact tool", action),
            }
        }
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
        assert_eq!(calls[0].arguments["prompt"], "a beautiful sunset over the ocean");
    }

    #[test]
    fn test_no_tool_call() {
        let text = "Hello! How can I help you today?";
        let calls = parse_tool_calls(text);
        assert!(calls.is_empty());
    }
}
