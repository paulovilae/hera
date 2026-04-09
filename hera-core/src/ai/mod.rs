//! Universal Artificial Intelligence Abstraction
//!
//! Exposes a standardized LLM inference schema matching industry expectations (OpenAI format).
//! Transparently maps these requests down into optimized native execution clients
//! (e.g., Gemini HTTP endpoints, local Llama GPU endpoints).

pub mod context_engine;
pub mod engine_faster_whisper;
pub mod engine_flux;
pub mod engine_gguf;
pub mod engine_hub;
pub mod engine_moondream;
pub mod engine_parler;
pub mod engine_whisper;
pub mod gemini;
pub mod llama_ffi_engine;
pub mod native_engine;
pub mod openai_compat;
pub mod q8_t5;
pub mod quantized_qwen3_moe_local;
pub mod router;
pub mod tool_executor;
pub mod tools;

use serde::{Deserialize, Serialize};

// --- Universal API Schema ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vision_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tts_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stt_model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nsfw: Option<bool>,

    // Execution Layer bindings
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String, // "system", "user", "assistant"
    pub content: MessageContent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
    Null,
}

impl Default for MessageContent {
    fn default() -> Self {
        MessageContent::Text(String::new())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrlContent },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrlContent {
    pub url: String, // Extracted Base64 image
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Option<ChatUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatResponseMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponseMessage {
    pub role: String,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
}

// --- Engine Definition ---

/// The central trait that any internal inference engine (Gemini, Llama) must satisfy
/// to power the Sovereign OS.
#[async_trait::async_trait]
pub trait LLMEngine {
    /// Dispatches a high-level multimodal message to the underlying neural network
    async fn generate_content(
        &self,
        req: ChatRequest,
    ) -> Result<ChatResponse, crate::ai::InferenceError>;

    /// Dispatches a multimodal request that yields Server-Sent Event streaming chunks
    async fn generate_stream(
        &self,
        _req: ChatRequest,
    ) -> Result<
        tokio::sync::mpsc::Receiver<Result<ChatStreamResponse, crate::ai::InferenceError>>,
        crate::ai::InferenceError,
    > {
        Err(crate::ai::InferenceError::ExecutionFailed(
            "Streaming not natively supported by this engine layer yet".to_string(),
        ))
    }
}

#[async_trait::async_trait]
pub trait SpeechToTextEngine {
    async fn transcribe_audio(
        &self,
        wav_bytes: &[u8],
    ) -> Result<String, crate::ai::InferenceError>;
}

// --- Streaming API Schema ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatStreamResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatStreamChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stats: Option<crate::ai::native_engine::GenerationStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatStreamChoice {
    pub index: u32,
    pub delta: ChatStreamDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatStreamDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
}

#[derive(thiserror::Error, Debug)]
pub enum InferenceError {
    #[error("Engine execution collapsed: {0}")]
    ExecutionFailed(String),
    #[error("Context bounds exceeded or invalid format")]
    InvalidContext(String),
}
