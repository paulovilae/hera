//! Smart LLM Fallback Router
//!
//! Implements a resilient `LLMEngine` that automatically prioritizes Sovereign (Local) execution
//! via `local_engine` first. If the local container collapses or rejects the prompt, it flawlessly
//! falls backward onto a secure `cloud_engine` (e.g., Google Gemini).

use crate::ai::{ChatRequest, ChatResponse, InferenceError, LLMEngine};
use std::sync::Arc;
use tracing::{error, info, warn};

pub struct RouterEngine {
    local_engine: Arc<dyn LLMEngine + Send + Sync>,
    cloud_engine: Arc<dyn LLMEngine + Send + Sync>,
}

impl RouterEngine {
    pub fn new(
        local_engine: Arc<dyn LLMEngine + Send + Sync>,
        cloud_engine: Arc<dyn LLMEngine + Send + Sync>,
    ) -> Self {
        Self {
            local_engine,
            cloud_engine,
        }
    }
}

#[async_trait::async_trait]
impl LLMEngine for RouterEngine {
    async fn generate_content(&self, req: ChatRequest) -> Result<ChatResponse, InferenceError> {
        let provider = req.provider.as_deref().unwrap_or("auto");

        // Priority 1: Sovereign AI execution (Local GPUs)
        if provider == "auto" || provider == "local" || provider == "local_direct" {
            info!("🕯️ Routing inference execution via Local MultiModal Engine...");
            match self.local_engine.generate_content(req.clone()).await {
                Ok(response) => {
                    info!("✅ Local execution successful");
                    return Ok(response);
                }
                Err(e) => {
                    if provider == "local" || provider == "local_direct" {
                        return Err(e); // Hard fail if local is strictly requested
                    }
                    warn!(
                        "⚠️ Local inference bounds collapsed: {:?}. Attempting seamless cloud failover...",
                        e
                    );
                }
            }
        }

        // Priority 2: Cloud failover logic (e.g., Gemini Flash)
        if provider == "auto" || provider == "gemini" || provider == "cloud" {
            info!("☁️ Re-routing inference execution onto Cloud MultiModal Engine...");
            match self.cloud_engine.generate_content(req.clone()).await {
                Ok(response) => {
                    info!("✅ Cloud failover successful");
                    return Ok(response);
                }
                Err(e) => {
                    error!(
                        "❌ Sovereign architecture critical fault: Multi-tier inference exhausted. {:?}",
                        e
                    );
                    return Err(e);
                }
            }
        }

        Err(InferenceError::ExecutionFailed(
            "All available multimodal bounds collapsed (Local & Cloud). Cannot proceed.".to_string(),
        ))
    }

    async fn generate_stream(
        &self,
        req: ChatRequest,
    ) -> Result<tokio::sync::mpsc::Receiver<Result<crate::ai::ChatStreamResponse, InferenceError>>, InferenceError> {
        let provider = req.provider.as_deref().unwrap_or("auto");

        if provider == "auto" || provider == "local" || provider == "local_direct" {
            info!("🕯️ Routing STREAMING inference via Local MultiModal Engine...");
            match self.local_engine.generate_stream(req.clone()).await {
                Ok(stream) => return Ok(stream),
                Err(e) => {
                    if provider == "local" || provider == "local_direct" { return Err(e); }
                    warn!("⚠️ Local streaming failed: {:?}. Attempting seamless cloud failover...", e);
                }
            }
        }

        if provider == "auto" || provider == "gemini" || provider == "cloud" {
            info!("☁️ Re-routing STREAMING inference onto Cloud MultiModal Engine...");
            match self.cloud_engine.generate_stream(req.clone()).await {
                Ok(stream) => { return Ok(stream); }
                Err(e) => { error!("☁️ Cloud Streaming fallback also crashed: {:?}", e); }
            }
        }

        Err(InferenceError::ExecutionFailed(
            "All streaming bounds collapsed (Local & Cloud). Cannot proceed.".to_string(),
        ))
    }
}
