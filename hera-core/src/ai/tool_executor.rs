//! Hera Tool Executor — Sovereign Tool Calling for ImagineOS
//!
//! Defines tool schemas in Qwen's native format, parses `<tool_call>` blocks
//! from Qwen output, and dispatches tool execution to existing Hera methods.

use serde::{Deserialize, Serialize};
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

/// Tool schemas in Qwen3's native Hermes-style format.
/// Uses the exact JSON schema structure that Qwen3 was trained on.
/// Reference: https://qwen3.org/docs/guides/tools
pub fn hera_tool_schemas() -> String {
    // Hermes-style: tools described as JSON function schemas
    let tools = serde_json::json!([
        {
            "type": "function",
            "function": {
                "name": "hera_draw",
                "description": "Generate an image locally using the GPU. MUST use this whenever the user asks for a picture, photo, drawing, OR follows up on a previous image with modifications (like 'now with a hat', 'make it red', 'ahora fumando').",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "A detailed description of the image to generate. Be specific about subject, style, colors, mood, and composition."
                        }
                    },
                    "required": ["prompt"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "hera_search",
                "description": "Search the web for current information. Use this when the user asks about recent events, news, facts you are unsure about, or anything requiring up-to-date information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "hera_speak",
                "description": "Convert text to speech audio. Use this when the user asks you to say something out loud, read text aloud, or generate audio speech.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to convert to speech"
                        },
                        "voice": {
                            "type": "string",
                            "description": "Voice ID. Options: en_US-amy-medium, es_MX-claude-high, es_ES-davefx-medium, pt_BR-faber-medium",
                            "default": "es_MX-claude-high"
                        }
                    },
                    "required": ["text"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "hera_video",
                "description": "Generate a video locally using the GPU. Use this when the user asks you to create, generate, or make a video or animation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "A detailed description of the video to generate"
                        }
                    },
                    "required": ["prompt"]
                }
            }
        }
    ]);

    let tools_json = serde_json::to_string_pretty(&tools).unwrap_or_default();

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

    calls
}

/// Fallback intent detection from the USER's original message.
/// Works with any model size since it doesn't depend on tool_call emission.
/// Returns a ToolCall if the user's intent clearly maps to a tool.
pub fn detect_intent_from_user_message(user_msg: &str, assistant_last: Option<&str>) -> Option<ToolCall> {
    let lower = user_msg.to_lowercase();

    // Contextual image modifier detection
    if let Some(ast) = assistant_last {
        if ast.contains("MEDIA:") || ast.contains("Aquí tienes") || ast.contains("Here is") || ast.contains("la imagen") {
            let is_modifier = user_msg.len() < 120 
                || lower.starts_with("ahora ") || lower.starts_with("now ") 
                || lower.starts_with("con ") || lower.starts_with("with ") 
                || lower.starts_with("sin ") || lower.starts_with("without ");
            
            if is_modifier {
                tracing::info!("🎯 [Hera] Intent detected: hera_draw from conversational context (modifier)");
                return Some(ToolCall {
                    name: "hera_draw".to_string(),
                    arguments: serde_json::json!({"prompt": user_msg}),
                });
            }
        }
    }

    // Draw/Image intent — very specific keywords to avoid false positives
    let draw_keywords = [
        "draw ", "dibuja", "genera una imagen", "create an image", "make an image",
        "generate an image", "draw me", "hazme un dibujo", "pinta",
        "haz una imagen", "genera imagen", "crea una imagen",
        "make a picture", "generate a picture", "create a picture",
        "haz un dibujo", "make me an image", "draw a ",
        // Photo/image requests
        "tu foto", "una foto", "mi foto", "dame foto", "dame una foto",
        "hazme una foto", "toma una foto", "manda una foto",
        "una imagen", "mi imagen", "tu imagen", "hazme una imagen",
        "genera una foto", "crea una foto",
        "your photo", "my photo", "take a photo", "send a photo",
        "a picture of", "make me a picture", "selfie", "retrato",
        "send me an image", "show me an image",
    ];
    if draw_keywords.iter().any(|kw| lower.contains(kw)) {
        // Extract the prompt: everything after the draw keyword
        let prompt = user_msg.to_string();
        info!("🎯 [Hera] Intent detected: hera_draw from user message");
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

    None
}

/// Execute a tool call using existing Hera infrastructure.
/// Returns a ToolResult with the output string.
pub async fn execute_tool(call: &ToolCall) -> ToolResult {
    info!("🔧 [Hera] Executing tool: {}", call.name);

    match call.name.as_str() {
        "hera_draw" => execute_draw(call).await,
        "hera_search" => execute_search(call).await,
        "hera_speak" => execute_speak(call).await,
        "hera_video" => execute_video(call).await,
        _ => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Unknown tool: {}", call.name),
        },
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
            let public_url = format!("https://hera.vilaros.ai/outputs/{}", filename);

            ToolResult {
                name: call.name.clone(),
                success: true,
                output: format!(
                    "Image generated successfully!\nMEDIA: {}\nInclude this MEDIA line EXACTLY as-is in your reply so the image is delivered inline.",
                    public_url
                ),
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
