//! Lightweight HTTP client for Argus's `GET /api/recommended-variant`.
//!
//! Argus owns hardware probing + sovereign-mesh detection + cluster placement
//! policy. When Hera needs to know "what LLM variant should I run on this node?"
//! the answer lives there, not in Hera. This client is a 1-call wrapper so we
//! can ask without duplicating the probing logic.
//!
//! Argus listens on `127.0.0.1:3006` (plain HTTP, internal-only). Override the
//! base URL with `HERA_ARGUS_URL` for tests or multi-node debugging.
//!
//! Failure mode: if Argus isn't running, we return `None` (and the caller falls
//! back to whatever the local env tells it). This module never panics.

use serde::{Deserialize, Serialize};
use std::time::Duration;

const DEFAULT_ARGUS_URL: &str = "http://127.0.0.1:3006";
const FETCH_TIMEOUT: Duration = Duration::from_millis(1_500);

/// Mirror of `argus::api::RecommendedVariantResponse`. We re-declare it so Hera
/// doesn't have to depend on the Argus crate (they live in separate workspaces).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendedVariant {
    pub strategy_kind: String,
    pub model_name: Option<String>,
    pub local_light_model: Option<String>,
    pub remote_mesh_ip: Option<String>,
    pub is_heavy_loaded: Option<bool>,
    pub mesh_connected: bool,
    pub hardware: HardwareSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareSummary {
    pub is_server: bool,
    pub has_nvidia_gpu: bool,
    pub total_memory_gb: u64,
    pub cpu_cores: usize,
    pub gpu_count: usize,
    pub max_total_vram_mb: u64,
}

impl RecommendedVariant {
    /// Best-effort label for the model that should be running locally on this node.
    /// Useful for logs + telemetry. None means we couldn't reach Argus.
    pub fn effective_local_model(&self) -> Option<&str> {
        match self.strategy_kind.as_str() {
            "offline_first_local" => self.model_name.as_deref(),
            "hybrid_mesh_delegate" => self.local_light_model.as_deref(),
            _ => None,
        }
    }

    /// True if Argus thinks heavy work should be delegated over the sovereign mesh.
    pub fn should_delegate_heavy(&self) -> bool {
        self.strategy_kind == "hybrid_mesh_delegate" && self.remote_mesh_ip.is_some()
    }
}

fn argus_base_url() -> String {
    std::env::var("HERA_ARGUS_URL").unwrap_or_else(|_| DEFAULT_ARGUS_URL.to_string())
}

/// Fetch the recommended variant from Argus. Returns `None` if Argus is down,
/// the HTTP body is unparseable, or the request times out.
pub async fn fetch_recommended_variant() -> Option<RecommendedVariant> {
    let url = format!("{}/api/recommended-variant", argus_base_url());
    let client = reqwest::Client::builder()
        .timeout(FETCH_TIMEOUT)
        .build()
        .ok()?;

    let response = match client.get(&url).send().await {
        Ok(r) => r,
        Err(error) => {
            tracing::debug!(target = "argus_client", error = %error, "argus unreachable");
            return None;
        }
    };

    if !response.status().is_success() {
        tracing::debug!(
            target = "argus_client",
            status = %response.status(),
            "argus returned non-2xx"
        );
        return None;
    }

    match response.json::<RecommendedVariant>().await {
        Ok(variant) => Some(variant),
        Err(error) => {
            tracing::debug!(target = "argus_client", error = %error, "argus response decode failed");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offline_first_local_resolves_model() {
        let variant = RecommendedVariant {
            strategy_kind: "offline_first_local".to_string(),
            model_name: Some("llama3-70b-instruct.gguf".to_string()),
            local_light_model: None,
            remote_mesh_ip: None,
            is_heavy_loaded: Some(true),
            mesh_connected: false,
            hardware: HardwareSummary {
                is_server: true,
                has_nvidia_gpu: true,
                total_memory_gb: 64,
                cpu_cores: 16,
                gpu_count: 1,
                max_total_vram_mb: 24_576,
            },
        };
        assert_eq!(
            variant.effective_local_model(),
            Some("llama3-70b-instruct.gguf")
        );
        assert!(!variant.should_delegate_heavy());
    }

    #[test]
    fn hybrid_mesh_resolves_light_model_and_delegates() {
        let variant = RecommendedVariant {
            strategy_kind: "hybrid_mesh_delegate".to_string(),
            model_name: None,
            local_light_model: Some("llama3-8b-instruct.gguf".to_string()),
            remote_mesh_ip: Some("10.100.0.2:3002".to_string()),
            is_heavy_loaded: None,
            mesh_connected: true,
            hardware: HardwareSummary {
                is_server: false,
                has_nvidia_gpu: false,
                total_memory_gb: 8,
                cpu_cores: 2,
                gpu_count: 0,
                max_total_vram_mb: 0,
            },
        };
        assert_eq!(
            variant.effective_local_model(),
            Some("llama3-8b-instruct.gguf")
        );
        assert!(variant.should_delegate_heavy());
    }
}
