//! Handler: execute_tool + get_tools actions.

use tokio::io::AsyncWriteExt;
use tokio::net::UnixStream;

use super::types::{HandlerOutcome, IpcPayload, IpcResponse, IpcState};

/// Handle the "execute_tool" action — direct tool invocation.
pub async fn handle_execute_tool(
    request: &IpcPayload,
    _state: &IpcState,
    stream: &mut UnixStream,
) -> HandlerOutcome {
    let tool_name = request
        .payload
        .get("tool_name")
        .and_then(|value| value.as_str())
        .unwrap_or("")
        .to_string();
    let arguments = request
        .payload
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));

    if tool_name.is_empty() {
        return HandlerOutcome::Result {
            result_text: "Missing tool_name".to_string(),
            origin: "tool".to_string(),
            model: "execute_tool".to_string(),
            tool_calls: None,
        };
    }

    let tool_call = crate::ai::tool_executor::ToolCall {
        name: tool_name.clone(),
        arguments: arguments.clone(),
    };

    match crate::ai::tool_executor::execute_tool_raw_json(&tool_call).await {
        Ok(result) => {
            let res = IpcResponse {
                status: "success".to_string(),
                data: serde_json::json!({
                    "result": result,
                    "origin": "tool",
                    "model": tool_name,
                    "tool_calls": [{
                        "name": tool_call.name,
                        "arguments": arguments
                    }]
                }),
            };
            let mut out_str = serde_json::to_string(&res).unwrap();
            out_str.push('\n');
            let _ = stream.write_all(out_str.as_bytes()).await;
            HandlerOutcome::DirectResponse
        }
        Err(error_text) => HandlerOutcome::Result {
            result_text: error_text,
            origin: "tool".to_string(),
            model: tool_name,
            tool_calls: None,
        },
    }
}

/// Handle the "get_tools" action — return available tool schemas.
pub fn handle_get_tools(_request: &IpcPayload, _state: &IpcState) -> HandlerOutcome {
    let raw_tools = serde_json::json!([
        {
            "type": "function",
            "function": {
                "name": "hera_draw",
                "description": "Generate an image locally using the GPU. MUST use this whenever the user asks for a picture, photo, drawing, OR follows up on a previous image with modifications. You are a multimodal AI (Claw Node) and you HAVE this capability.",
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
                "description": "Read text aloud using Text-to-Speech (TTS). Use this to generate audio files of your response when requested.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to be spoken."
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
                "description": "Generate a short video. You have multimodal capabilities as a Claw Node. Use this when the user asks for a video, animation, or moving picture.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "A detailed description of the video to generate, including motion, subject, and style."
                        }
                    },
                    "required": ["prompt"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "hera_read_file",
                "description": "Read the contents of a local file on the system. Use this when the user asks to read, view, or check a file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The absolute or relative path to the file to read."
                        }
                    },
                    "required": ["path"]
                }
            }
        }
    ]);

    HandlerOutcome::Result {
        result_text: "Tools retrieved".to_string(),
        origin: "unknown".to_string(),
        model: String::new(),
        tool_calls: Some(serde_json::json!({ "tools": raw_tools })),
    }
}
