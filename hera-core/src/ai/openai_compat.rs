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
        let active_endpoint = req
            .endpoint
            .clone()
            .unwrap_or_else(|| self.endpoint_url.clone());
        let active_key = req.api_key.clone().unwrap_or_else(|| self.api_key.clone());
        let mut normalized_req = req;

        normalized_req.stream = Some(false);

        if (normalized_req.model.is_empty() || normalized_req.model.starts_with("hera-"))
            && let Ok(explicit_model) = std::env::var("HERA_OPENAI_MODEL")
            && !explicit_model.trim().is_empty()
        {
            normalized_req.model = explicit_model.trim().to_string();
        }

        // Cloud failover: the local model name (e.g. "Qwen3.6-35B...gguf") is never
        // valid at a cloud provider, so for any cloud-routed request force the
        // configured cloud model. Works for Groq / Google / OpenRouter alike.
        if normalized_req.provider.as_deref() == Some("cloud")
            && let Ok(cloud_model) = std::env::var("HERA_CLOUD_DEFAULT_MODEL")
                .or_else(|_| std::env::var("OPENROUTER_DEFAULT_MODEL"))
            && !cloud_model.trim().is_empty()
        {
            normalized_req.model = cloud_model.trim().to_string();
        }

        if (normalized_req.model.is_empty() || normalized_req.model.starts_with("hera-"))
            && active_endpoint.contains("127.0.0.1:8080")
            && let Some(discovered_model) =
                discover_first_model_id(&self.client, &active_endpoint, &active_key).await
        {
            normalized_req.model = discovered_model;
        }

        // Force multimodal array format to avoid strict provider crashes (e.g. OpenRouter expecting objects instead of strings)
        for message in &mut normalized_req.messages {
            if let crate::ai::MessageContent::Text(text) = &message.content {
                message.content =
                    crate::ai::MessageContent::Parts(vec![crate::ai::ContentPart::Text {
                        text: text.clone(),
                    }]);
            }
        }

        let mut payload = serde_json::to_value(&normalized_req).unwrap_or_default();
        // Strip Hera-internal routing fields that are NOT part of the OpenAI API spec.
        // OpenRouter rejects `provider: "cloud"` (expects object, gets string).
        if let Some(obj) = payload.as_object_mut() {
            obj.remove("provider");
            obj.remove("endpoint");
            obj.remove("api_key");
            obj.remove("vision_model");
            obj.remove("tts_model");
            obj.remove("stt_model");
            obj.remove("nsfw");
            // Some cloud providers (e.g. Groq) return 400 on non-standard fields
            // like `reasoning_effort` (llama.cpp locally just ignores it). Strip
            // it for cloud-routed requests so the failover doesn't bounce.
            if normalized_req.provider.as_deref() == Some("cloud") {
                obj.remove("reasoning_effort");
            }
        }
        tracing::debug!("Outbound Request Payload: {}", payload);

        let mut request_builder = self.client.post(&active_endpoint).json(&payload);

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

    async fn generate_stream(
        &self,
        req: ChatRequest,
    ) -> Result<
        tokio::sync::mpsc::Receiver<Result<crate::ai::ChatStreamResponse, InferenceError>>,
        InferenceError,
    > {
        let active_endpoint = req
            .endpoint
            .clone()
            .unwrap_or_else(|| self.endpoint_url.clone());
        let active_key = req.api_key.clone().unwrap_or_else(|| self.api_key.clone());
        let mut normalized_req = req;

        normalized_req.stream = Some(true); // Ensure stream is true

        if (normalized_req.model.is_empty() || normalized_req.model.starts_with("hera-"))
            && let Ok(explicit_model) = std::env::var("HERA_OPENAI_MODEL")
            && !explicit_model.trim().is_empty()
        {
            normalized_req.model = explicit_model.trim().to_string();
        }

        // Cloud failover: the local model name (e.g. "Qwen3.6-35B...gguf") is never
        // valid at a cloud provider, so for any cloud-routed request force the
        // configured cloud model. Works for Groq / Google / OpenRouter alike.
        if normalized_req.provider.as_deref() == Some("cloud")
            && let Ok(cloud_model) = std::env::var("HERA_CLOUD_DEFAULT_MODEL")
                .or_else(|_| std::env::var("OPENROUTER_DEFAULT_MODEL"))
            && !cloud_model.trim().is_empty()
        {
            normalized_req.model = cloud_model.trim().to_string();
        }

        if (normalized_req.model.is_empty() || normalized_req.model.starts_with("hera-"))
            && active_endpoint.contains("127.0.0.1:8080")
            && let Some(discovered_model) =
                discover_first_model_id(&self.client, &active_endpoint, &active_key).await
        {
            normalized_req.model = discovered_model;
        }

        for message in &mut normalized_req.messages {
            if let crate::ai::MessageContent::Text(text) = &message.content {
                message.content =
                    crate::ai::MessageContent::Parts(vec![crate::ai::ContentPart::Text {
                        text: text.clone(),
                    }]);
            }
        }

        let mut payload = serde_json::to_value(&normalized_req).unwrap_or_default();
        if let Some(obj) = payload.as_object_mut() {
            obj.remove("provider");
            obj.remove("endpoint");
            obj.remove("api_key");
            obj.remove("vision_model");
            obj.remove("tts_model");
            obj.remove("stt_model");
            obj.remove("nsfw");
            // Some cloud providers (e.g. Groq) return 400 on non-standard fields
            // like `reasoning_effort` (llama.cpp locally just ignores it). Strip
            // it for cloud-routed requests so the failover doesn't bounce.
            if normalized_req.provider.as_deref() == Some("cloud") {
                obj.remove("reasoning_effort");
            }
        }

        let mut request_builder = self.client.post(&active_endpoint).json(&payload);
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

        let (tx, rx) = tokio::sync::mpsc::channel(100);

        tokio::spawn(async move {
            use futures_util::StreamExt;
            let mut stream = res.bytes_stream();
            let mut buffer = String::new();

            while let Some(chunk_res) = stream.next().await {
                match chunk_res {
                    Ok(bytes) => {
                        buffer.push_str(&String::from_utf8_lossy(&bytes));
                        while let Some(idx) = buffer.find('\n') {
                            let line = buffer[..idx].trim().to_string();
                            buffer = buffer[idx + 1..].to_string();

                            if let Some(json_str) = line.strip_prefix("data: ") {
                                if json_str == "[DONE]" {
                                    return; // Explicitly close the stream task when [DONE] is received to prevent Keep-Alive hangs
                                }
                                match serde_json::from_str::<crate::ai::ChatStreamResponse>(
                                    json_str,
                                ) {
                                    Ok(parsed) => {
                                        if tx.send(Ok(parsed)).await.is_err() {
                                            return;
                                        }
                                    }
                                    Err(e) => {
                                        tracing::warn!(
                                            "Failed to parse SSE chunk: {} -> {}",
                                            e,
                                            json_str
                                        );
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx
                            .send(Err(InferenceError::ExecutionFailed(format!(
                                "Stream read error: {}",
                                e
                            ))))
                            .await;
                        break;
                    }
                }
            }
        });

        Ok(rx)
    }
}

async fn discover_first_model_id(
    client: &Client,
    endpoint_url: &str,
    api_key: &str,
) -> Option<String> {
    let models_url = endpoint_url.replace("/v1/chat/completions", "/v1/models");
    let mut request_builder = client.get(models_url);
    if !api_key.is_empty() {
        request_builder = request_builder.bearer_auth(api_key);
    }

    let response = request_builder.send().await.ok()?;
    if !response.status().is_success() {
        return None;
    }

    let payload: serde_json::Value = response.json().await.ok()?;
    payload
        .get("data")
        .and_then(|v| v.as_array())
        .and_then(|items| items.first())
        .and_then(|item| item.get("id"))
        .and_then(|id| id.as_str())
        .map(str::to_string)
}
