//! Platform tool executors: desktop, draw, search, speak, video, files, agents, skills, soul, user interaction
pub(crate) mod code;
pub(crate) mod media;
pub(crate) mod workspace;

pub(crate) use code::{execute_run_code, execute_write_file};
pub(crate) use media::{execute_draw, execute_animate_avatar, execute_speak, execute_video, execute_review_image};
pub(crate) use workspace::{execute_finish_bootstrap, execute_write_workspace_file};

use crate::ai::tool_executor::{ToolCall, ToolResult, find_skill_artifact, load_agent_artifact};
use hera_execution::agents::hera::Hera;
use serde_json::json;
use std::path::{Path, PathBuf};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
use tracing::info;

const HERA_SOCKET: &str = "/tmp/hera-core.sock";
const OS_ROOT: &str = "/home/paulo/Programs/apps/OS";
const OS_ROOT_ALT: &str = "/mnt/workspace/Programs/apps/OS";
const HOME_ROOT: &str = "/home/paulo";
const TMP_ROOT: &str = "/tmp";
const SMARTOS_ROUTER_URL: &str = "http://127.0.0.1:3000";

// Hard-blocked regardless of caller/permissions/route_profile — root confinement
// alone still lets edit_file/write_file reach secrets or corrupt a repo's .git
// (writing into .git broke a repo via Syncthing before; see
// reference_git_empty_object_corruption_syncthing). These are never legitimate
// edit/write targets for any Hera tool, so they are checked here, at the single
// choke point every fs tool resolves through, instead of trusted to prompt text.
const FORBIDDEN_PATH_PREFIXES: &[&str] = &[
    "/home/paulo/.config/imagineos/secrets",
    "/home/paulo/.ssh",
];

fn is_forbidden_path(candidate: &Path) -> bool {
    if FORBIDDEN_PATH_PREFIXES
        .iter()
        .any(|prefix| candidate.starts_with(prefix))
    {
        return true;
    }
    // Blocks any `.git` directory anywhere in the tree (OS root's own .git,
    // and every submodule's/nested repo's .git), not just the OS root's.
    candidate.components().any(|component| {
        matches!(
            component,
            std::path::Component::Normal(name) if name == ".git"
        )
    })
}

pub(crate) fn resolve_guarded_fs_path(path: &str, allow_tmp: bool) -> Result<PathBuf, String> {
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
                .ok_or_else(|| "Path must include a file name".to_string())?,
        )
    };

    if is_forbidden_path(&candidate) {
        return Err(format!(
            "Path '{}' is a hard-blocked Hera path (secrets or .git internals) and cannot be accessed by any tool.",
            path
        ));
    }

    // Allow OS root (both symlink and canonical paths), home dir, and /tmp
    let in_os_root = candidate.starts_with(OS_ROOT) || candidate.starts_with(OS_ROOT_ALT);
    let in_home = candidate.starts_with(HOME_ROOT);
    let in_tmp = allow_tmp && candidate.starts_with(TMP_ROOT);

    if in_os_root || in_home || in_tmp {
        Ok(candidate)
    } else {
        Err(format!(
            "Path '{}' is outside allowed Hera roots ('{}'{}).",
            path,
            OS_ROOT,
            if allow_tmp { ", '/tmp'" } else { "" }
        ))
    }
}

fn validate_python_package_name(package: &str) -> bool {
    !package.is_empty()
        && package.len() <= 64
        && package
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.'))
}

fn parse_ipc_result(response: &str) -> Result<String, String> {
    let mut accumulated_text = String::new();
    let mut final_result = None;

    for line in response.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let Ok(message) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };

        match message.get("status").and_then(|value| value.as_str()) {
            Some("success") => {
                if let Some(result) = message
                    .pointer("/data/result")
                    .and_then(|value| value.as_str())
                {
                    final_result = Some(result.to_string());
                }
            }
            Some("chunk") => {
                if let Some(text) = message
                    .pointer("/data/text")
                    .and_then(|value| value.as_str())
                {
                    accumulated_text.push_str(text);
                }
            }
            Some("error") => {
                let error = message
                    .pointer("/data/error")
                    .and_then(|value| value.as_str())
                    .unwrap_or("unknown Hera IPC error");
                return Err(error.to_string());
            }
            _ => {}
        }
    }

    if let Some(result) = final_result {
        Ok(result)
    } else if !accumulated_text.is_empty() {
        Ok(accumulated_text)
    } else {
        Err("No content in Hera IPC response".to_string())
    }
}

pub(super) fn hera_execution_agent() -> Hera {
    Hera::new(SMARTOS_ROUTER_URL)
}

async fn run_agent_via_hera_ipc(persona: String, prompt: String) -> Result<String, String> {
    let mut stream = UnixStream::connect(HERA_SOCKET)
        .await
        .map_err(|error| format!("failed to connect to Hera IPC: {error}"))?;

    let request = json!({
        "action": "generate",
        "payload": {
            "app": "hera",
            "messages": [
                { "role": "system", "content": persona },
                { "role": "user", "content": prompt }
            ],
            "temperature": 0.2,
            "max_tokens": 1200,
            "permissions": []
        }
    });

    let payload = format!("{}\n", request);
    stream
        .write_all(payload.as_bytes())
        .await
        .map_err(|error| format!("failed to write Hera IPC request: {error}"))?;

    stream
        .shutdown()
        .await
        .map_err(|error| format!("failed to shutdown Hera IPC write half: {error}"))?;

    let mut response = String::new();
    stream
        .read_to_string(&mut response)
        .await
        .map_err(|error| format!("failed to read Hera IPC response: {error}"))?;

    parse_ipc_result(&response)
}

fn resolve_app_main_css(app: &str) -> Option<PathBuf> {
    let needle = app.trim().to_lowercase();
    if needle.is_empty() {
        return None;
    }

    let apps_dir = PathBuf::from("/home/paulo/Programs/apps/OS/Apps");
    let entries = std::fs::read_dir(apps_dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let dir_name = entry.file_name().to_string_lossy().to_lowercase();
        let normalized = dir_name.replace('_', "-");
        if dir_name != needle && normalized != needle {
            continue;
        }
        let css_path = path.join("media/css/main.css");
        if css_path.exists() {
            return Some(css_path);
        }
    }
    None
}

pub(crate) async fn execute_edit_app_theme(call: &ToolCall) -> ToolResult {
    let app = call
        .arguments
        .get("app")
        .and_then(|value| value.as_str())
        .unwrap_or("")
        .trim();
    let theme = call
        .arguments
        .get("theme")
        .and_then(|value| value.as_str())
        .unwrap_or("both")
        .trim();
    let variables = call
        .arguments
        .get("variables")
        .and_then(|value| value.as_object());

    if app.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Missing required 'app' parameter.".to_string(),
        };
    }

    let Some(variables) = variables else {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Missing or invalid 'variables' object.".to_string(),
        };
    };

    if variables.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Theme edit requires at least one CSS variable override.".to_string(),
        };
    }

    let Some(css_path) = resolve_app_main_css(app) else {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Could not find media/css/main.css for app '{}'.", app),
        };
    };

    let selector = match theme {
        "light" => ":root, [data-theme=\"light\"]",
        "dark" => "[data-theme=\"dark\"]",
        "both" => ":root, [data-theme=\"light\"], [data-theme=\"dark\"]",
        _ => {
            return ToolResult {
                name: call.name.clone(),
                success: false,
                output: "Theme must be one of: light, dark, both.".to_string(),
            };
        }
    };

    let mut overrides = String::from("\n/* Hera theme override */\n");
    overrides.push_str(selector);
    overrides.push_str(" {\n");
    for (key, value) in variables {
        let Some(raw) = value.as_str() else {
            return ToolResult {
                name: call.name.clone(),
                success: false,
                output: format!("Theme variable '{}' must map to a string value.", key),
            };
        };
        overrides.push_str(&format!("  {}: {};\n", key.trim(), raw.trim()));
    }
    overrides.push_str("}\n");

    match std::fs::OpenOptions::new()
        .append(true)
        .open(&css_path)
        .and_then(|mut file| std::io::Write::write_all(&mut file, overrides.as_bytes()))
    {
        Ok(_) => ToolResult {
            name: call.name.clone(),
            success: true,
            output: format!(
                "App theme updated by appending CSS overrides to {}",
                css_path.display()
            ),
        },
        Err(error) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Failed to update app theme: {}", error),
        },
    }
}

pub(crate) async fn execute_desktop_click(call: &ToolCall) -> ToolResult {
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

pub(crate) async fn execute_desktop_type(call: &ToolCall) -> ToolResult {
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

pub(crate) async fn execute_load_skill(call: &ToolCall) -> ToolResult {
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

pub(crate) async fn execute_spawn_parallel_agents(call: &ToolCall) -> ToolResult {
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

            match run_agent_via_hera_ipc(persona, p).await {
                Ok(content) => format!(
                    "--- REPORT FROM {} ---\n{}\n",
                    agent_name.to_uppercase(),
                    content
                ),
                Err(error) => format!(
                    "--- REPORT FROM {} ---\nFailed to reach inference engine via Hera IPC: {}\n",
                    agent_name.to_uppercase(),
                    error
                ),
            }
        }));
    }

    let mut combined_report = String::new();
    for task in tasks {
        if let Ok(report) = task.await {
            combined_report.push_str(&report);
            combined_report.push('\n');
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

pub(crate) async fn execute_create_agent(call: &ToolCall) -> ToolResult {
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

pub(crate) async fn execute_create_skill(call: &ToolCall) -> ToolResult {
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

pub(crate) async fn execute_search(call: &ToolCall) -> ToolResult {
    let query = call
        .arguments
        .get("query")
        .and_then(|q| q.as_str())
        .unwrap_or("");
    let hera = hera_execution_agent();
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

pub(crate) async fn execute_read_file(call: &ToolCall) -> ToolResult {
    let path = call
        .arguments
        .get("path")
        .and_then(|p| p.as_str())
        .unwrap_or("");
    let resolved_path = match resolve_guarded_fs_path(path, true) {
        Ok(path) => path,
        Err(error) => {
            return ToolResult {
                name: call.name.clone(),
                success: false,
                output: error,
            };
        }
    };
    match std::fs::read_to_string(&resolved_path) {
        Ok(content) => {
            let truncated = if content.len() > 16_000 {
                format!("{}... (truncated)", &content[..16_000])
            } else {
                content
            };
            info!("📄 [Hera] Read file: {}", resolved_path.display());
            ToolResult {
                name: call.name.clone(),
                success: true,
                output: format!(
                    "File contents of '{}':\n{}",
                    resolved_path.display(),
                    truncated
                ),
            }
        }
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Failed to read file '{}': {}", resolved_path.display(), e),
        },
    }
}

pub(crate) async fn execute_update_soul(call: &ToolCall) -> ToolResult {
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
                output: "Successfully updated your SOUL! The changes have been saved to disk and you will remember them permanently.".to_string(),
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

pub(crate) async fn execute_ask_user(call: &ToolCall) -> ToolResult {
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

pub(crate) async fn execute_get_system_time(call: &ToolCall) -> ToolResult {
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

pub(crate) async fn execute_web_scraper(call: &ToolCall) -> ToolResult {
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
    let parsed_url = match reqwest::Url::parse(url) {
        Ok(url) => url,
        Err(error) => {
            return ToolResult {
                name: call.name.clone(),
                success: false,
                output: format!("Invalid URL: {}", error),
            };
        }
    };
    // Validacion SSRF UNIFICADA con data.rs: valida esquema + host prohibido +
    // resuelve el DNS y comprueba TODAS las IPs (cierra DNS rebinding a IP interna /
    // metadata de GCP). Reemplaza el blocklist debil por-string anterior.
    if let Err(error) = crate::ai::tools::data::validate_outbound_url(parsed_url.as_str()).await {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: error,
        };
    }

    let hera = hera_execution_agent();
    match hera.native_web_scrape(parsed_url.as_str()).await {
        Ok(content) => ToolResult {
            name: call.name.clone(),
            success: true,
            output: content,
        },
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: e.to_string(),
        },
    }
}

pub(crate) async fn execute_spline_interact(call: &ToolCall) -> ToolResult {
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

// ── generate_access_link ──────────────────────────────────────────────────
// Calls OS-v3 POST /api/auth/magic-link/create, returns a 30-min login URL
// that Ava can send to Paulo over Telegram.
pub(crate) async fn execute_generate_access_link(call: &ToolCall) -> ToolResult {
    let redirect = call
        .arguments
        .get("redirect")
        .and_then(|v| v.as_str())
        .unwrap_or("/editor")
        .to_string();

    // Read shared secret
    let secret_path = std::env::var("OS_AUTH_SHARED_SECRET_FILE")
        .unwrap_or_else(|_| "/home/paulo/.config/imagineos/secrets/os-auth-shared-secret".to_string());
    let secret = match std::fs::read_to_string(&secret_path) {
        Ok(s) => s.trim().to_string(),
        Err(e) => {
            return ToolResult {
                name: call.name.clone(),
                success: false,
                output: format!("Could not read shared secret: {}", e),
            }
        }
    };

    let os_v3_url = std::env::var("OS_V3_INTERNAL_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:5177".to_string());
    let endpoint = format!("{}/api/auth/magic-link/create", os_v3_url);

    let admin_email = std::env::var("OS_ADMIN_EMAIL")
        .unwrap_or_else(|_| "admin@localhost".to_string());
    let admin_name = std::env::var("OS_ADMIN_NAME")
        .unwrap_or_else(|_| "Admin".to_string());
    let body = serde_json::json!({
        "email": admin_email,
        "name": admin_name,
        "redirect": redirect,
        "admin": true
    });

    let client = reqwest::Client::new();
    match client
        .post(&endpoint)
        .header("x-os-secret", &secret)
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => {
            match resp.json::<serde_json::Value>().await {
                Ok(json) => {
                    let url = json["url"].as_str().unwrap_or("").to_string();
                    info!("🔗 [Hera] Magic link generated → {}", url);
                    ToolResult {
                        name: call.name.clone(),
                        success: true,
                        output: format!("🔗 Aquí tu acceso (válido 30 min):\n{}", url),
                    }
                }
                Err(e) => ToolResult {
                    name: call.name.clone(),
                    success: false,
                    output: format!("Invalid response from OS-v3: {}", e),
                },
            }
        }
        Ok(resp) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("OS-v3 returned HTTP {}", resp.status()),
        },
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Could not reach OS-v3: {}", e),
        },
    }
}

// ── list_image_loras ──────────────────────────────────────────────────────────
// Reads available LoRAs from the local triggers.json + lora_weights.json files,
// or from the sd.cpp /sdapi/v1/loras endpoint if the image server is reachable.
// Falls back gracefully to the file-based list; always returns something useful.
pub(crate) async fn execute_list_image_loras(call: &ToolCall) -> ToolResult {
    let loras_dir = std::env::var("HERA_LORAS_DIR")
        .unwrap_or_else(|_| "/home/paulo/models/image-stack/loras".to_string());

    let triggers_path = format!("{}/triggers.json", loras_dir);
    let weights_path = format!("{}/lora_weights.json", loras_dir);

    // Try loading triggers.json (lora_name -> [keyword, ...])
    let triggers: std::collections::HashMap<String, Vec<String>> =
        std::fs::read_to_string(&triggers_path)
            .ok()
            .and_then(|c| serde_json::from_str(&c).ok())
            .unwrap_or_default();

    let weights: std::collections::HashMap<String, f32> =
        std::fs::read_to_string(&weights_path)
            .ok()
            .and_then(|c| serde_json::from_str(&c).ok())
            .unwrap_or_default();

    if !triggers.is_empty() {
        // Build list from local files
        let mut lines: Vec<String> = triggers
            .iter()
            .map(|(name, keywords)| {
                let weight = weights.get(name).copied().unwrap_or(0.7);
                let trigger_str = keywords.join(", ");
                format!("- **{}** (weight: {:.2}) — triggers: {}", name, weight, trigger_str)
            })
            .collect();
        lines.sort();

        info!("[Hera] list_image_loras: {} LoRAs from local files", lines.len());
        return ToolResult {
            name: call.name.clone(),
            success: true,
            output: format!(
                "Available LoRAs ({} total):\n{}\n\nTo use a LoRA explicitly: `<lora:name:weight>` in your prompt.",
                lines.len(),
                lines.join("\n")
            ),
        };
    }

    // No local files — try sd.cpp REST endpoint
    let draw_base = std::env::var("HERA_DRAW_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:8999".to_string());
    let endpoint = format!("{}/sdapi/v1/loras", draw_base);
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
        .unwrap_or_default();

    match client.get(&endpoint).send().await {
        Ok(resp) if resp.status().is_success() => {
            match resp.json::<serde_json::Value>().await {
                Ok(json) => {
                    let entries = json.as_array().cloned().unwrap_or_default();
                    let lines: Vec<String> = entries
                        .iter()
                        .filter_map(|e| e.get("name").and_then(|n| n.as_str()))
                        .map(|name| format!("- **{}**", name))
                        .collect();
                    ToolResult {
                        name: call.name.clone(),
                        success: true,
                        output: format!("Available LoRAs ({} total):\n{}", lines.len(), lines.join("\n")),
                    }
                }
                Err(_) => ToolResult {
                    name: call.name.clone(),
                    success: false,
                    output: "Image server responded but returned invalid JSON for /sdapi/v1/loras".to_string(),
                },
            }
        }
        _ => {
            // Hardcoded baseline when nothing else is available
            let baseline = vec![
                "- **flux-schnell-q4** (weight: 1.00) — triggers: fast, quick draft, sketch",
                "- **realistic_vision** (weight: 0.70) — triggers: realistic, photo, portrait, photography",
            ];
            ToolResult {
                name: call.name.clone(),
                success: true,
                output: format!(
                    "LoRA list unavailable (image server offline, no local triggers.json). Known baseline LoRAs:\n{}",
                    baseline.join("\n")
                ),
            }
        }
    }
}

// ── corporate_research ────────────────────────────────────────────────────────
// Multi-phase corporate dossier tool:
//   1. Four parallel web searches (profile, market, competitors, ESG).
//   2. Assembles a raw research dossier from the results.
//   3. Synthesizes a structured McKinsey-style JSON report via Hera IPC.
pub(crate) async fn execute_corporate_research(call: &ToolCall) -> ToolResult {
    let entity = match call.arguments.get("entity").and_then(|v| v.as_str()) {
        Some(e) if !e.trim().is_empty() => e.trim().to_string(),
        _ => {
            return ToolResult {
                name: call.name.clone(),
                success: false,
                output: "Missing required argument: 'entity' (company name to research)".to_string(),
            };
        }
    };

    let depth = call
        .arguments
        .get("depth")
        .and_then(|v| v.as_str())
        .unwrap_or("standard");

    let lang = call
        .arguments
        .get("language")
        .and_then(|v| v.as_str())
        .unwrap_or("es");

    // Build search queries based on depth and language
    let queries: Vec<String> = if lang == "en" {
        vec![
            format!("{} company profile history founder products services", entity),
            format!("{} industry market size growth trends", entity),
            format!("{} main competitors market share competitive advantage", entity),
            format!("{} ESG sustainability governance shareholders ownership structure", entity),
        ]
    } else {
        vec![
            format!("{} empresa perfil corporativo historia fundador productos servicios", entity),
            format!("{} industria mercado tamaño crecimiento tendencias", entity),
            format!("{} competidores principales cuota mercado rivales ventaja competitiva", entity),
            format!("{} ESG sostenibilidad gobernanza accionistas estructura propiedad", entity),
        ]
    };

    // Limit queries for 'quick' depth
    let active_queries: Vec<&String> = match depth {
        "quick" => queries.iter().take(2).collect(),
        _ => queries.iter().collect(),
    };

    info!(
        "[Hera] corporate_research: researching '{}' ({} queries, lang={})",
        entity,
        active_queries.len(),
        lang
    );

    // Execute searches in parallel via native_web_search
    let hera = hera_execution_agent();
    let search_futures: Vec<_> = active_queries
        .iter()
        .map(|q| hera.native_web_search(q))
        .collect();

    let search_results = futures_util::future::join_all(search_futures).await;

    // Assemble raw dossier from successful searches
    let mut dossier_parts: Vec<String> = Vec::new();
    let query_labels = ["PERFIL / PROFILE", "MERCADO / MARKET", "COMPETIDORES / COMPETITORS", "ESG / GOBERNANZA"];

    for (i, result) in search_results.into_iter().enumerate() {
        let label = query_labels.get(i).copied().unwrap_or("RESEARCH");
        match result {
            Ok(content) if !content.trim().is_empty() => {
                dossier_parts.push(format!("## {}\n{}", label, content.trim()));
            }
            Ok(_) => {
                dossier_parts.push(format!("## {}\n(No results found)", label));
            }
            Err(e) => {
                dossier_parts.push(format!("## {}\n(Search error: {})", label, e));
            }
        }
    }

    if dossier_parts.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("All web searches failed for '{}'. Check hera_search availability.", entity),
        };
    }

    let raw_dossier = dossier_parts.join("\n\n");

    // LLM synthesis via Hera IPC — returns structured JSON report
    let system_prompt = if lang == "en" {
        format!(
            "You are a Senior McKinsey strategic consultant. Based ONLY on the research dossier provided, \
             generate a structured corporate profile for '{}' in JSON format with these keys: \
             company_overview, industry_context, consulting_kpis (market_share_percent, year_over_year_growth, \
             employee_count, revenue_estimate, geographic_reach), swot (strengths, weaknesses, opportunities, \
             threats as arrays), competitors (array of {{name, advantage, risk}}), \
             catalysts_and_news (recent_headlines array, executive_summary), history_background, \
             ownership_structure, esg_outlook, investment_scores (growth, operations, leadership, market as 1-10 integers). \
             DO NOT invent data. If a field is unknown, use null or an empty array.",
            entity
        )
    } else {
        format!(
            "Eres un consultor estratégico Senior de McKinsey. Basándote ÚNICAMENTE en el dossier de investigación, \
             genera un perfil corporativo estructurado en JSON para '{}' con estas claves: \
             company_overview, industry_context, consulting_kpis (market_share_percent, year_over_year_growth, \
             employee_count, revenue_estimate, geographic_reach), swot (strengths, weaknesses, opportunities, \
             threats como arrays), competitors (array de {{name, advantage, risk}}), \
             catalysts_and_news (recent_headlines array, executive_summary), history_background, \
             ownership_structure, esg_outlook, investment_scores (growth, operations, leadership, market enteros 1-10). \
             NO inventes datos. Si un campo es desconocido, usa null o array vacío.",
            entity
        )
    };

    let user_prompt = format!(
        "Empresa a investigar: {}\n\nDOSSIER DE INVESTIGACIÓN:\n{}",
        entity, raw_dossier
    );

    let mut stream = match UnixStream::connect(HERA_SOCKET).await {
        Ok(s) => s,
        Err(e) => {
            // Return raw dossier if IPC is unavailable
            return ToolResult {
                name: call.name.clone(),
                success: true,
                output: format!(
                    "Research dossier for '{entity}' (synthesis unavailable — Hera IPC error: {e}):\n\n{raw_dossier}"
                ),
            };
        }
    };

    let ipc_request = json!({
        "action": "generate",
        "payload": {
            "app": "hera",
            "messages": [
                { "role": "system", "content": system_prompt },
                { "role": "user", "content": user_prompt }
            ],
            "temperature": 0.15,
            "max_tokens": 4096,
            "permissions": [],
            "response_format": { "type": "json_object" }
        }
    });

    let payload = format!("{}\n", ipc_request);
    if let Err(e) = stream.write_all(payload.as_bytes()).await {
        return ToolResult {
            name: call.name.clone(),
            success: true,
            output: format!("Research dossier for '{entity}' (synthesis write error: {e}):\n\n{raw_dossier}"),
        };
    }
    if let Err(e) = stream.shutdown().await {
        return ToolResult {
            name: call.name.clone(),
            success: true,
            output: format!("Research dossier for '{entity}' (synthesis shutdown error: {e}):\n\n{raw_dossier}"),
        };
    }

    let mut response = String::new();
    if let Err(e) = stream.read_to_string(&mut response).await {
        return ToolResult {
            name: call.name.clone(),
            success: true,
            output: format!("Research dossier for '{entity}' (synthesis read error: {e}):\n\n{raw_dossier}"),
        };
    }

    match parse_ipc_result(&response) {
        Ok(synthesized) => {
            info!("[Hera] corporate_research: synthesis complete for '{}'", entity);
            ToolResult {
                name: call.name.clone(),
                success: true,
                output: synthesized,
            }
        }
        Err(e) => {
            // Synthesis failed — return the raw dossier so the call is still useful
            ToolResult {
                name: call.name.clone(),
                success: true,
                output: format!(
                    "Research dossier for '{entity}' (synthesis error: {e}):\n\n{raw_dossier}"
                ),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::ai::tool_executor::parse_tool_calls;
    use super::is_forbidden_path;
    use std::path::Path;

    #[test]
    fn test_forbidden_secrets_path() {
        assert!(is_forbidden_path(Path::new(
            "/home/paulo/.config/imagineos/secrets/postgres-password"
        )));
    }

    #[test]
    fn test_forbidden_ssh_path() {
        assert!(is_forbidden_path(Path::new("/home/paulo/.ssh/id_rsa")));
    }

    #[test]
    fn test_forbidden_git_at_os_root() {
        assert!(is_forbidden_path(Path::new(
            "/home/paulo/Programs/apps/OS/.git/config"
        )));
    }

    #[test]
    fn test_forbidden_git_in_nested_submodule() {
        assert!(is_forbidden_path(Path::new(
            "/home/paulo/Programs/apps/OS/Hera/.git/HEAD"
        )));
    }

    #[test]
    fn test_allowed_normal_path_not_forbidden() {
        assert!(!is_forbidden_path(Path::new(
            "/home/paulo/Programs/apps/OS/Hera/hera-core/src/ai/tools/platform.rs"
        )));
    }

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
}
