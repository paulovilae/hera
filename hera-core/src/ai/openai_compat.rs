//! OpenAI Compatible Local LLM Engine
//!
//! Exposes a standardized REST client designed to interface with local sovereign LLMs
//! (like Qwen or Llama variants running on OpenClaw/vLLM) that natively support the OpenAI HTTP format.

use crate::ai::{ChatRequest, ChatResponse, InferenceError, LLMEngine};
use reqwest::Client;

pub struct OpenAICompatEngine {
    client: Client,
    endpoint_url: String, // e.g., "http://127.0.0.1:11434/v1/chat/completions" or OpenClaw address
    api_key: String,      // Optional API key for the local endpoint
}

impl OpenAICompatEngine {
    pub fn new(endpoint_url: String, api_key: String) -> Self {
        Self {
            client: Client::new(),
            endpoint_url,
            api_key,
        }
    }
}

#[async_trait::async_trait]
impl LLMEngine for OpenAICompatEngine {
    async fn generate_content(&self, req: ChatRequest) -> Result<ChatResponse, InferenceError> {
        let active_endpoint = req.endpoint.clone().unwrap_or_else(|| self.endpoint_url.clone());
        let active_key = req.api_key.clone().unwrap_or_else(|| self.api_key.clone());

        let mut request_builder = self.client.post(&active_endpoint).json(&req);

        // Inject Bearer token if an API key is provided for the local engine
        if !active_key.is_empty() {
            request_builder = request_builder.bearer_auth(&active_key);
        }

        let res = request_builder.send().await.map_err(|e| {
            InferenceError::ExecutionFailed(format!(
                "Local OpenAI compat engine unreachable: {}",
                e
            ))
        })?;

        let status = res.status();

        if !status.is_success() {
            let err_text = res.text().await.unwrap_or_default();
            return Err(InferenceError::ExecutionFailed(format!(
                "Local engine rejection ({}): {}",
                status, err_text
            )));
        }

        let oai_res: ChatResponse = res.json().await.map_err(|e| {
            InferenceError::ExecutionFailed(format!("Failed decoding local JSON response: {}", e))
        })?;

        Ok(oai_res)
    }
}
