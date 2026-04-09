//! Google Gemini High-Performance REST Engine
//!
//! Maps Universal Schema directly into Google's `generateContent` format seamlessly
//! supporting multimodal Image Base64 decoding bounds.

use crate::ai::{
    ChatChoice, ChatRequest, ChatResponse, ChatResponseMessage, ContentPart, InferenceError,
    LLMEngine, MessageContent,
};
use reqwest::Client;
use serde::{Deserialize, Serialize};

pub struct GeminiEngine {
    client: Client,
    api_key: String,
}

impl GeminiEngine {
    pub fn new(api_key: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
        }
    }
}

// --- Native Gemini Data Models ---
// (Mapping only what is strictly necessary to bounce the payload fast)

#[derive(Serialize)]
struct GeminiGenerateRequest {
    contents: Vec<GeminiContent>,
    #[serde(rename = "systemInstruction", skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GeminiSystemInstruction>,
    #[serde(rename = "generationConfig", skip_serializing_if = "Option::is_none")]
    generation_config: Option<GeminiGenerationConfig>,
}

#[derive(Serialize)]
struct GeminiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(rename = "topP", skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(rename = "topK", skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(rename = "maxOutputTokens", skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(rename = "stopSequences", skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
}

#[derive(Serialize)]
struct GeminiSystemInstruction {
    parts: Vec<GeminiPart>,
}

#[derive(Serialize, Deserialize, Clone)]
struct GeminiContent {
    role: String, // "user" or "model"
    parts: Vec<GeminiPart>,
}

#[derive(Serialize, Deserialize, Clone)]
struct GeminiPart {
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    inline_data: Option<GeminiInlineData>,
}

#[derive(Serialize, Deserialize, Clone)]
struct GeminiInlineData {
    mime_type: String, // e.g. "image/jpeg"
    data: String,      // Raw base64 bytes
}

#[derive(Deserialize)]
struct GeminiGenerateResponse {
    candidates: Option<Vec<GeminiCandidate>>,
    error: Option<GeminiError>,
    #[serde(rename = "usageMetadata")]
    usage_metadata: Option<GeminiUsage>,
}

#[derive(Deserialize)]
struct GeminiUsage {
    #[serde(rename = "promptTokenCount")]
    prompt_token_count: Option<u32>,
    #[serde(rename = "candidatesTokenCount")]
    candidates_token_count: Option<u32>,
    #[serde(rename = "totalTokenCount")]
    total_token_count: Option<u32>,
}

#[derive(Deserialize)]
struct GeminiCandidate {
    content: GeminiContent,
    #[serde(rename = "finishReason")]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct GeminiError {
    message: String,
}

#[async_trait::async_trait]
impl LLMEngine for GeminiEngine {
    async fn generate_content(&self, req: ChatRequest) -> Result<ChatResponse, InferenceError> {
        let mut gemini_contents = Vec::new();
        let mut system_instruction = None;

        for msg in req.messages {
            let role = match msg.role.as_str() {
                "system" => {
                    // Extract system instructional bounds separately for Gemini API
                    let text = match msg.content {
                        MessageContent::Text(t) => t,
                        MessageContent::Parts(parts) => parts
                            .iter()
                            .filter_map(|p| match p {
                                ContentPart::Text { text } => Some(text.clone()),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join(" "),
                        MessageContent::Null => String::new(),
                    };
                    system_instruction = Some(GeminiSystemInstruction {
                        parts: vec![GeminiPart {
                            text: Some(text),
                            inline_data: None,
                        }],
                    });
                    continue; // Skip appending to contents
                }
                "assistant" => "model",
                _ => "user",
            };

            let mut gemini_parts = Vec::new();

            match msg.content {
                MessageContent::Text(text) => {
                    gemini_parts.push(GeminiPart {
                        text: Some(text),
                        inline_data: None,
                    });
                }
                MessageContent::Null => {
                    continue;
                }
                MessageContent::Parts(parts) => {
                    for part in parts {
                        match part {
                            ContentPart::Text { text } => {
                                gemini_parts.push(GeminiPart {
                                    text: Some(text),
                                    inline_data: None,
                                });
                            }
                            ContentPart::ImageUrl { image_url } => {
                                // Extract Data URI component: "data:image/jpeg;base64,...str"
                                let url = image_url.url;
                                if let Some(stripped) = url.strip_prefix("data:")
                                    && let Some((mime, b64)) = stripped.split_once(";base64,") {
                                        gemini_parts.push(GeminiPart {
                                            text: None,
                                            inline_data: Some(GeminiInlineData {
                                                mime_type: mime.to_string(),
                                                data: b64.to_string(),
                                            }),
                                        });
                                        continue;
                                    }
                                // Fallback if format is mangled
                                return Err(InferenceError::InvalidContext(
                                    "Malformed base64 injected to Gemini model".into(),
                                ));
                            }
                        }
                    }
                }
            }

            gemini_contents.push(GeminiContent {
                role: role.to_string(),
                parts: gemini_parts,
            });
        }

        let generation_config = if req.temperature.is_some()
            || req.top_p.is_some()
            || req.max_tokens.is_some()
            || req.top_k.is_some()
            || req.stop.is_some()
        {
            Some(GeminiGenerationConfig {
                temperature: req.temperature,
                top_p: req.top_p,
                top_k: req.top_k,
                max_output_tokens: req.max_tokens,
                stop_sequences: req.stop,
            })
        } else {
            None
        };

        let gemini_req = GeminiGenerateRequest {
            contents: gemini_contents,
            system_instruction,
            generation_config,
        };

        // Standardize the model target
        let target_model = if req.model.contains("gemini") {
            req.model.clone()
        } else {
            "gemini-2.5-flash".to_string() // Fallback fast engine
        };

        let endpoint = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            target_model, self.api_key
        );

        let res = self
            .client
            .post(&endpoint)
            .json(&gemini_req)
            .send()
            .await
            .map_err(|e| {
                InferenceError::ExecutionFailed(format!("Reqwest engine collapse: {}", e))
            })?;

        let status = res.status();

        if !status.is_success() {
            let err_text = res.text().await.unwrap_or_default();
            return Err(InferenceError::ExecutionFailed(format!(
                "Gemini rejection ({}): {}",
                status, err_text
            )));
        }

        let gemini_res: GeminiGenerateResponse = res.json().await.map_err(|e| {
            InferenceError::ExecutionFailed(format!("Failed decoding neural JSON: {}", e))
        })?;

        if let Some(e) = gemini_res.error {
            return Err(InferenceError::ExecutionFailed(e.message));
        }

        let mut choices = Vec::new();

        if let Some(candidates) = gemini_res.candidates {
            for (i, cand) in candidates.into_iter().enumerate() {
                // Collapse parts down
                let text_res = cand
                    .content
                    .parts
                    .iter()
                    .filter_map(|p| p.text.clone())
                    .collect::<Vec<_>>()
                    .join("");

                choices.push(ChatChoice {
                    index: i as u32,
                    message: ChatResponseMessage {
                        role: "assistant".to_string(),
                        content: Some(text_res),
                        tool_calls: None,
                    },
                    finish_reason: cand.finish_reason,
                });
            }
        }

        let usage = gemini_res.usage_metadata.map(|meta| crate::ai::ChatUsage {
            prompt_tokens: meta.prompt_token_count.unwrap_or(0),
            completion_tokens: meta.candidates_token_count.unwrap_or(0),
            total_tokens: meta.total_token_count.unwrap_or(0),
        });

        Ok(ChatResponse {
            id: format!(
                "chatcmpl-{}",
                std::time::UNIX_EPOCH.elapsed().unwrap().as_secs()
            ),
            object: "chat.completion".to_string(),
            created: std::time::UNIX_EPOCH.elapsed().unwrap().as_secs(),
            model: target_model,
            choices,
            usage,
        })
    }
}
