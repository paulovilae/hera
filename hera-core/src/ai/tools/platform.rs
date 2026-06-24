//! Platform tool executors: desktop, draw, search, speak, video, files, agents, skills, soul, user interaction
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

fn hera_execution_agent() -> Hera {
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

pub(crate) async fn execute_draw(call: &ToolCall) -> ToolResult {
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

    let hera = hera_execution_agent();
    match hera
        .generate_image(
            prompt, None, width, height, None, None, None, None, None, None, None,
        )
        .await
    {
        Ok(res) => {
            let image_url = res
                .get("image_url")
                .and_then(|u| u.as_str())
                .unwrap_or("(no URL)");
            info!("🎨 [Hera] Image generated: {}", image_url);

            // Build a public URL that candle-core serves at /outputs/{filename}
            // The filename is the last segment of image_url (e.g., "/outputs/hera_drawn_UUID.png")
            let filename = image_url.split('/').next_back().unwrap_or(image_url);
            let public_url = format!("https://imaginos.ai/outputs/{}", filename);
            let response = format!(
                "Image generated successfully!\nMEDIA: {}\nInclude this MEDIA line EXACTLY as-is in your reply so the image is delivered inline.",
                public_url
            );

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

pub(crate) async fn execute_animate_avatar(call: &ToolCall) -> ToolResult {
    let text = match call
        .arguments
        .get("text")
        .and_then(|t| t.as_str())
    {
        Some(t) if !t.trim().is_empty() => t,
        _ => {
            return ToolResult {
                name: call.name.clone(),
                success: false,
                output: "Missing 'text' parameter — provide the text the avatar should say."
                    .to_string(),
            };
        }
    };
    let character = call
        .arguments
        .get("character")
        .and_then(|c| c.as_str())
        .unwrap_or("edu");
    let face_url = call
        .arguments
        .get("face_url")
        .and_then(|u| u.as_str());
    let voice = call
        .arguments
        .get("voice")
        .and_then(|v| v.as_str())
        .unwrap_or("paddi");

    let hera = hera_execution_agent();
    match hera.animate_avatar(text, character, face_url, Some(voice)).await {
        Ok(res) => {
            let video_url = res
                .get("video_url")
                .and_then(|u| u.as_str())
                .unwrap_or("(no URL)");
            info!("🎬 [Hera] Avatar animation generated: {}", video_url);

            let filename = video_url.split('/').next_back().unwrap_or(video_url);
            let public_url = format!("https://imaginos.ai/outputs/{}", filename);
            let response = format!(
                "Avatar animation generated successfully!\nMEDIA: {}\nInclude this MEDIA line EXACTLY as-is in your reply so the video is delivered inline.",
                public_url
            );

            ToolResult {
                name: call.name.clone(),
                success: true,
                output: response,
            }
        }
        Err(e) => {
            tracing::error!("🎬 [Hera] Avatar animation failed: {:?}", e);
            ToolResult {
                name: call.name.clone(),
                success: false,
                output: format!("Avatar animation failed: {}", e),
            }
        }
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

pub(crate) async fn execute_speak(call: &ToolCall) -> ToolResult {
    let text = call
        .arguments
        .get("text")
        .and_then(|t| t.as_str())
        .unwrap_or("");
    let voice = call.arguments.get("voice").and_then(|v| v.as_str());

    let hera = hera_execution_agent();
    match hera.synthesize_speech(text, voice).await {
        Ok(result) => {
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
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("TTS failed: {}", e),
        },
    }
}

pub(crate) async fn execute_video(call: &ToolCall) -> ToolResult {
    let prompt = call
        .arguments
        .get("prompt")
        .and_then(|p| p.as_str())
        .unwrap_or("A smooth cinematic video");

    let hera = hera_execution_agent();
    match hera.synthesize_video(prompt).await {
        Ok(result) => {
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
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Video generation failed: {}", e),
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

pub(crate) async fn execute_run_code(call: &ToolCall) -> ToolResult {
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

    if code.trim().is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Missing code payload.".to_string(),
        };
    }
    if code.len() > 100_000 {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Code payload too large. Limit is 100000 bytes.".to_string(),
        };
    }
    if packages.len() > 16
        || packages
            .iter()
            .any(|pkg| !validate_python_package_name(pkg))
    {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Package list contains invalid names or exceeds the maximum allowed count."
                .to_string(),
        };
    }

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
            },
            "client": {
                "app": "hera",
                "token": std::env::var("MEMENTO_CLIENT_TOKEN").ok()
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

        let mut deps = r#"[dependencies]
tokio = { version = "1", features = ["full", "rt-multi-thread"] }
reqwest = { version = "0.11", features = ["json"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
"#
        .to_string();
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
        // Sanitize LLM-generated code: strip single spurious leading spaces.
        // Local models occasionally emit lines with exactly 1 leading space on
        // what should be top-level statements (e.g. " y = np.sin(x)").
        // Real Python indentation uses 2+ spaces; 1-space is almost always an LLM error.
        let sanitized: String = code
            .lines()
            .map(|line| {
                if line.starts_with(' ') && !line.starts_with("  ") {
                    line.trim_start_matches(' ')
                } else {
                    line
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        let p = format!("{}/{}.py", beans_dir, bean_name);
        if let Err(e) = std::fs::write(&p, &sanitized) {
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
                .arg("--user")
                .arg("--break-system-packages")
                .arg("--quiet");
            for pkg in &packages {
                cmd.arg(pkg);
            }
            match cmd.output() {
                Ok(out) if out.status.success() => {
                    pip_log = format!("Installed: {:?}", packages);
                }
                Ok(out) => {
                    // PEP 668 or already-installed: warn but still run the code.
                    // The package may already be available; don't abort.
                    let err = String::from_utf8_lossy(&out.stderr);
                    pip_log = format!(
                        "pip warning (code still runs): {}",
                        err.lines().next().unwrap_or("non-zero exit")
                    );
                }
                Err(e) => {
                    pip_log = format!("pip unavailable ({}), proceeding anyway", e);
                }
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

pub(crate) async fn execute_write_file(call: &ToolCall) -> ToolResult {
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

    match std::fs::write(&resolved_path, content) {
        Ok(_) => ToolResult {
            name: call.name.clone(),
            success: true,
            output: format!("Successfully wrote to {}", resolved_path.display()),
        },
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Failed to write file: {}", e),
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

    let body = serde_json::json!({
        "email": "vilapaulo@gmail.com",
        "name": "Paulo",
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

/// Sovereign image understanding — sends an image to the local VLM (Qwen2.5-VL @
/// HERA_VISION_URL, default :8083) and returns its answer. Used to describe or QA
/// images (e.g. detect headless people / deformed hands in generated covers).
/// `image` may be an http(s) URL or a local file path. `question` is optional.
pub(crate) async fn execute_review_image(call: &ToolCall) -> ToolResult {
    use base64::Engine as _;

    let fail = |msg: String| ToolResult {
        name: call.name.clone(),
        success: false,
        output: msg,
    };

    let image = call
        .arguments
        .get("image")
        .or_else(|| call.arguments.get("image_url"))
        .or_else(|| call.arguments.get("path"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if image.is_empty() {
        return fail("Missing 'image' (an http(s) URL or a local file path).".to_string());
    }
    let question = call
        .arguments
        .get("question")
        .or_else(|| call.arguments.get("prompt"))
        .and_then(|v| v.as_str())
        .filter(|s| !s.trim().is_empty())
        .unwrap_or(
            "Describe this image briefly. Then, if it has obvious generation defects \
             (a person with a missing/cut-off head or face, deformed or extra hands/limbs, \
             melted faces, garbled text), add a final line 'DEFECT: <what>'. Otherwise add 'OK'.",
        );

    // Load the image bytes (remote URL or local path) and build a data URL.
    let bytes: Vec<u8> = if image.starts_with("http://") || image.starts_with("https://") {
        match reqwest::Client::new().get(&image).send().await {
            Ok(r) => match r.bytes().await {
                Ok(b) => b.to_vec(),
                Err(e) => return fail(format!("could not read image body: {e}")),
            },
            Err(e) => return fail(format!("could not fetch image: {e}")),
        }
    } else {
        match tokio::fs::read(&image).await {
            Ok(b) => b,
            Err(e) => return fail(format!("could not read file '{image}': {e}")),
        }
    };
    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
    let data_url = format!("data:image/png;base64,{b64}");

    let vision_url = std::env::var("HERA_VISION_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:8083/v1/chat/completions".to_string());
    let payload = serde_json::json!({
        "model": "vision",
        "max_tokens": 200,
        "temperature": 0.0,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": question}
            ]
        }]
    });

    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
    {
        Ok(c) => c,
        Err(e) => return fail(format!("client build failed: {e}")),
    };
    match client.post(&vision_url).json(&payload).send().await {
        Ok(resp) => match resp.json::<serde_json::Value>().await {
            Ok(j) => {
                let text = j["choices"][0]["message"]["content"]
                    .as_str()
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if text.is_empty() {
                    return fail("vision model returned an empty response".to_string());
                }
                ToolResult {
                    name: call.name.clone(),
                    success: true,
                    output: text,
                }
            }
            Err(e) => fail(format!("could not parse vision response: {e}")),
        },
        Err(e) => fail(format!("vision request failed (is vision-review up @ {vision_url}?): {e}")),
    }
}

#[cfg(test)]
mod tests {
    use crate::ai::tool_executor::parse_tool_calls;

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
