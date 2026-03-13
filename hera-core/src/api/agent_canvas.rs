use crate::ai::{ChatRequest, ChatMessage, MessageContent, LLMEngine};
use axum::{Json, extract::State};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Deserialize)]
pub struct CanvasRequest {
    pub prompt: String,
    #[serde(default = "default_model")]
    pub model: String,
    pub context_shapes: Option<Vec<serde_json::Value>>,
}

fn default_model() -> String {
    "gemini-2.5-flash".to_string() // or whatever fast reasoner is configured
}

#[derive(Serialize)]
pub struct CanvasResponse {
    pub actions: Vec<serde_json::Value>,
    pub raw_thought: Option<String>,
    pub error: Option<String>,
}

const CANVAS_SYSTEM_PROMPT: &str = r#"You are the Hera Canvas Styling Agent. 
Your job is to translate user natural language requests into programmatic Tldraw actions.
You must output a JSON array of actions. Do NOT output anything outside of the JSON array, except inside <think></think> tags.

IMPORTANT RULES FOR "props":
- "color": The line/stroke color. Must be one of: "black", "blue", "red", "green", "yellow", "orange", "violet", "grey", "light-blue", "light-red", "light-green", "light-violet", "white".
- "fill": The interior fill style. Must be EXACTLY one of: "none", "semi", "solid", "pattern". NEVER put a color name in "fill". If a user wants a solid red shape, use `"color": "red"` and `"fill": "solid"`.

Available actions:
1. createShape
{
  "action": "createShape",
  "shape": {
    "type": "geo",
    "x": 100,
    "y": 100,
    "props": {
      "geo": "rectangle",
      "color": "blue",
      "fill": "solid",
      "w": 100,
      "h": 100
    }
  }
}

2. updateShape (Requires knowing the shape id from context)
{
  "action": "updateShape",
  "id": "shape:123",
  "type": "geo",
  "props": {
    "color": "red",
    "fill": "solid"
  }
}

3. deleteShape
{
  "action": "deleteShape",
  "id": "shape:123"
}

Respond ONLY with a valid JSON array of these action objects after optional <think></think> reasoning. 
Example response:
<think>The user wants a red triangle.</think>
[
  {
    "action": "createShape",
    "shape": { "type": "geo", "x": 100, "y": 100, "props": { "geo": "triangle", "color": "red", "fill": "solid" } }
  }
]
"#;

pub async fn process_canvas_request(
    State(state): State<Arc<crate::api::routes::ApiState>>,
    Json(payload): Json<CanvasRequest>,
) -> axum::response::Result<Json<CanvasResponse>, axum::http::StatusCode> {
    
    let mut context_msg = String::new();
    if let Some(shapes) = &payload.context_shapes {
        context_msg = format!("Current Canvas Context (Shapes):\n{}\n\n", serde_json::to_string(shapes).unwrap_or_default());
    }
    
    let user_msg = format!("{context_msg}User Prompt: {}", payload.prompt);

    let messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: MessageContent::Text(CANVAS_SYSTEM_PROMPT.to_string()),
        },
        ChatMessage {
            role: "user".to_string(),
            content: MessageContent::Text(user_msg),
        }
    ];

    let chat_req = ChatRequest {
        model: payload.model.clone(),
        vision_model: None,
        tts_model: None,
        stt_model: None,
        messages,
        temperature: Some(0.1),
        max_tokens: Some(2048),
        top_p: None,
        top_k: None,
        presence_penalty: None,
        frequency_penalty: None,
        repeat_penalty: None,
        seed: None,
        stop: None,
        endpoint: None,
        api_key: None,
        provider: None,
        stream: Some(false),
        tools: None,
        tool_choice: None,
        reasoning_effort: None,
        nsfw: None,
    };

    match state.engine.generate_content(chat_req).await {
        Ok(resp) => {
            let content = resp.choices.first().and_then(|c| c.message.content.clone()).unwrap_or_default();
            
            // Extract <think> if present
            let mut raw_thought = None;
            let mut json_str = content.clone();
            
            if let Some(start) = content.find("<think>") {
                if let Some(end) = content.find("</think>") {
                    let end_idx = end + "</think>".len();
                    raw_thought = Some(content[start + 7..end].trim().to_string());
                    json_str = content[end_idx..].trim().to_string();
                }
            }
            
            // Try to extract markdown JSON block
            if json_str.contains("```json") {
                if let Some(start) = json_str.find("```json") {
                    if let Some(end) = json_str[start+7..].find("```") {
                        json_str = json_str[start+7..start+7+end].trim().to_string();
                    }
                }
            } else if json_str.contains("```") {
                if let Some(start) = json_str.find("```") {
                    if let Some(end) = json_str[start+3..].find("```") {
                        json_str = json_str[start+3..start+3+end].trim().to_string();
                    }
                }
            }
            
            json_str = json_str.trim().to_string();
            
            // Parse array
            match serde_json::from_str::<Vec<serde_json::Value>>(&json_str) {
                Ok(actions) => Ok(Json(CanvasResponse { actions, raw_thought, error: None })),
                Err(_) => {
                    // Try parsing as single object
                    match serde_json::from_str::<serde_json::Value>(&json_str) {
                        Ok(single) => Ok(Json(CanvasResponse { actions: vec![single], raw_thought, error: None })),
                        Err(e) => Ok(Json(CanvasResponse { 
                            actions: vec![], 
                            raw_thought, 
                            error: Some(format!("Failed to parse JSON: {}", e)) 
                        }))
                    }
                }
            }
        },
        Err(e) => {
            Ok(Json(CanvasResponse {
                actions: vec![],
                raw_thought: None,
                error: Some(e.to_string()),
            }))
        }
    }
}
