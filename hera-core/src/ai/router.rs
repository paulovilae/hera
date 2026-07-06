//! Smart LLM Fallback Router
//!
//! Implements a resilient `LLMEngine` that automatically prioritizes Sovereign (Local) execution
//! via `local_engine` first. If the local container collapses or rejects the prompt, it flawlessly
//! falls backward onto a secure `cloud_engine` (e.g., Google Gemini).

use crate::ai::{ChatRequest, ChatResponse, InferenceError, LLMEngine};
use serde::Deserialize;
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use std::{
    fs,
    path::{Path, PathBuf},
};
use tokio::sync::Semaphore;
use tracing::{error, info, warn};

pub struct RouterEngine {
    primary_engine: Arc<dyn LLMEngine + Send + Sync>,
    secondary_engine: Option<Arc<dyn LLMEngine + Send + Sync>>,
    tertiary_engine: Option<Arc<dyn LLMEngine + Send + Sync>>,
    cloud_engine: Arc<dyn LLMEngine + Send + Sync>,
}

// NOTE (2026-05-30): the router used to append a Memo mascot image to EVERY local
// response (`tag_response_engine`). That hardcoded Movilo's mascot onto every app's
// output — Consulting briefs, dossiers and any artifact generated through Hera ended
// with a blue-cat picture. The marker was also redundant: each bot's persona (memo.md,
// chigui.md, ...) already instructs the model to emit its OWN signature/mascot, and
// Imaginclaw does not parse the marker. So per-app identity lives in the persona, not
// the router. Removed: Memo is no longer forced everywhere; non-chat artifacts stay clean.

impl RouterEngine {
    pub fn new(
        local_engine: Arc<dyn LLMEngine + Send + Sync>,
        cloud_engine: Arc<dyn LLMEngine + Send + Sync>,
    ) -> Self {
        Self::with_fallbacks(local_engine, None, None, cloud_engine)
    }

    pub fn with_fallbacks(
        primary_engine: Arc<dyn LLMEngine + Send + Sync>,
        secondary_engine: Option<Arc<dyn LLMEngine + Send + Sync>>,
        tertiary_engine: Option<Arc<dyn LLMEngine + Send + Sync>>,
        cloud_engine: Arc<dyn LLMEngine + Send + Sync>,
    ) -> Self {
        Self {
            primary_engine,
            secondary_engine,
            tertiary_engine,
            cloud_engine,
        }
    }
}

/// Concurrency admission control (Hera concurrency/backpressure fix, 2026-07-05).
///
/// `hera-core` used to forward every concurrent `generate` request straight onto
/// the single GPU `llama-server` (:8080) with NO admission control — under load
/// (real bots + background derivation + agentic loops + `spawn_parallel_agents`
/// fan-out, which self-connects back through this same IPC `generate` path)
/// requests piled up, tail latency hit 49s, and clients timed out mid-flight
/// ("Failed to write IPC response: Broken pipe (os error 32)"). See
/// docs/HERA_CONCURRENCY_BACKPRESSURE.md for the full diagnosis.
///
/// Verified live on genesis (2026-07-05): the omni server's own `/props` reports
/// `total_slots: 4` (llama-server auto-picked N=4 for `--parallel -1`).
/// `Hera/scripts/start_native_omni.sh` now pins `--parallel` explicitly via
/// `IMAGINEOS_OMNI_PARALLEL_SLOTS` (default 4) so the real backend slot count
/// stays deterministic across restarts — keep `HERA_PRIMARY_ENGINE_SLOTS` (below)
/// equal to that value; they are two separate processes/env domains so nothing
/// enforces the equality automatically.
///
/// This semaphore bounds how many requests may be in-flight against the PRIMARY
/// engine at once. Excess requests wait up to
/// `HERA_PRIMARY_ENGINE_ACQUIRE_TIMEOUT_SECS` (default 8s) for a free slot; if
/// none frees up in time, the primary attempt is treated exactly like a
/// primary-engine error and the existing secondary/tertiary/cloud failover chain
/// takes over — a fast, honest failover beats a silent 49s stall that
/// broken-pipes the caller.
///
/// Known scope limit (documented, not silently dropped): this bounds admission
/// uniformly for every caller — it does not yet distinguish interactive chat
/// from background derivation (KG extraction, session/room/project summary
/// compression). `ChatRequest` carries no priority/route_profile field today,
/// so a true two-lane scheme needs that plumbed through every call site first.
/// Uniform bounded-wait-then-failover still stops any single caller (including
/// a runaway background task) from queuing indefinitely, which was the acute
/// failure mode — full priority lanes are a follow-up, not this fix.
fn primary_engine_semaphore() -> Arc<Semaphore> {
    static SEM: OnceLock<Arc<Semaphore>> = OnceLock::new();
    SEM.get_or_init(|| {
        let slots = std::env::var("HERA_PRIMARY_ENGINE_SLOTS")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(4);
        Arc::new(Semaphore::new(slots))
    })
    .clone()
}

fn primary_engine_acquire_timeout() -> Duration {
    let secs = std::env::var("HERA_PRIMARY_ENGINE_ACQUIRE_TIMEOUT_SECS")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(8);
    Duration::from_secs(secs)
}

#[async_trait::async_trait]
impl LLMEngine for RouterEngine {
    async fn generate_content(&self, req: ChatRequest) -> Result<ChatResponse, InferenceError> {
        let provider = req.provider.as_deref().unwrap_or("auto");
        let local_req = prepare_local_request(&req);
        let cloud_req = prepare_cloud_request(&req);
        let cloud_allowed = cloud_fallback_allowed(&req);

        // Priority 1-3: Sovereign AI execution across owned compute
        if provider == "auto" || provider == "local" || provider == "local_direct" {
            let primary_req = with_endpoint_override(local_req.clone(), "HERA_PRIMARY_OMNI_URL");
            info!(
                "🕯️ Routing inference execution via Primary Sovereign Engine (model='{}')...",
                primary_req.model
            );
            let permit = tokio::time::timeout(
                primary_engine_acquire_timeout(),
                primary_engine_semaphore().acquire_owned(),
            )
            .await;
            match permit {
                Ok(Ok(_permit)) => {
                    match self.primary_engine.generate_content(primary_req).await {
                        Ok(response) => {
                            info!("✅ Primary sovereign execution successful");
                            return Ok(response);
                        }
                        Err(e) => {
                            if provider == "local_direct" {
                                return Err(e); // Hard fail if local is strictly requested
                            }
                            warn!(
                                "⚠️ Primary sovereign execution failed: {:?}. Attempting standby failover...",
                                e
                            );
                        }
                    }
                }
                Ok(Err(_closed)) => {
                    warn!(
                        "⚠️ Primary engine semaphore closed unexpectedly; treating as saturated."
                    );
                    if provider == "local_direct" {
                        return Err(InferenceError::ExecutionFailed(
                            "Primary sovereign engine unavailable (semaphore closed).".to_string(),
                        ));
                    }
                }
                Err(_elapsed) => {
                    warn!(
                        "⚠️ Primary sovereign engine saturated (all slots busy for {:?}) — failing over instead of queuing further...",
                        primary_engine_acquire_timeout()
                    );
                    if provider == "local_direct" {
                        return Err(InferenceError::ExecutionFailed(
                            "Primary sovereign engine busy (backend saturated); local_direct disallows failover.".to_string(),
                        ));
                    }
                }
            }

            if let Some(secondary_engine) = &self.secondary_engine {
                let secondary_req = with_explicit_endpoint(
                    local_req.clone(),
                    std::env::var("HERA_SECONDARY_OMNI_URL").ok(),
                );
                info!("🕯️ Routing inference execution via Secondary Sovereign Engine...");
                match secondary_engine.generate_content(secondary_req).await {
                    Ok(response) => {
                        info!("✅ Secondary sovereign execution successful");
                        return Ok(response);
                    }
                    Err(e) => {
                        if provider == "local" {
                            return Err(e);
                        }
                        warn!(
                            "⚠️ Secondary sovereign execution failed: {:?}. Attempting tertiary failover...",
                            e
                        );
                    }
                }
            }

            if let Some(tertiary_engine) = &self.tertiary_engine {
                let tertiary_req = with_explicit_endpoint(
                    local_req.clone(),
                    std::env::var("HERA_TERTIARY_OMNI_URL").ok(),
                );
                info!("🕯️ Routing inference execution via Tertiary Sovereign Engine...");
                match tertiary_engine.generate_content(tertiary_req).await {
                    Ok(response) => {
                        info!("✅ Tertiary sovereign execution successful");
                        return Ok(response);
                    }
                    Err(e) => {
                        if provider == "local" {
                            return Err(e);
                        }
                        warn!(
                            "⚠️ Tertiary sovereign execution failed: {:?}. Attempting cloud failover...",
                            e
                        );
                    }
                }
            }
        }

        // Priority 4: Commercial cloud failover
        if (provider == "auto" || provider == "gemini" || provider == "cloud") && cloud_allowed {
            if let Err(reason) = crate::ai::cloud_budget::check_and_record_cloud_call(
                crate::ai::cloud_budget::estimate_tokens(&cloud_req),
            ) {
                warn!("🛑 Cloud call denied by cost-safety gate: {}", reason);
                return Err(InferenceError::ExecutionFailed(reason));
            }
            info!("☁️ Re-routing inference execution onto Cloud MultiModal Engine...");
            match self.cloud_engine.generate_content(cloud_req).await {
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

        if !cloud_allowed && (provider == "auto" || provider == "gemini" || provider == "cloud") {
            return Err(InferenceError::ExecutionFailed(
                "External cloud fallback is disallowed by current sovereign policy.".to_string(),
            ));
        }

        Err(InferenceError::ExecutionFailed(
            "All available multimodal bounds collapsed (Local & Cloud). Cannot proceed."
                .to_string(),
        ))
    }

    async fn generate_stream(
        &self,
        req: ChatRequest,
    ) -> Result<
        tokio::sync::mpsc::Receiver<Result<crate::ai::ChatStreamResponse, InferenceError>>,
        InferenceError,
    > {
        let provider = req.provider.as_deref().unwrap_or("auto");
        let local_req = prepare_local_request(&req);
        let cloud_req = prepare_cloud_request(&req);
        let cloud_allowed = cloud_fallback_allowed(&req);

        if provider == "auto" || provider == "local" || provider == "local_direct" {
            let primary_req = with_endpoint_override(local_req.clone(), "HERA_PRIMARY_OMNI_URL");
            info!(
                "🕯️ Routing STREAMING inference via Primary Sovereign Engine (model='{}')...",
                primary_req.model
            );
            let permit = tokio::time::timeout(
                primary_engine_acquire_timeout(),
                primary_engine_semaphore().acquire_owned(),
            )
            .await;
            match permit {
                Ok(Ok(owned_permit)) => {
                    match self.primary_engine.generate_stream(primary_req).await {
                        Ok(mut inner_rx) => {
                            // Hold the slot permit for the WHOLE stream lifetime, not just
                            // admission — a streaming generation still occupies a real GPU
                            // slot until its last token, so release-on-admission would
                            // undercount true concurrency.
                            let (tx, rx) = tokio::sync::mpsc::channel(32);
                            tokio::spawn(async move {
                                let _permit = owned_permit; // dropped (released) when this task ends
                                while let Some(item) = inner_rx.recv().await {
                                    if tx.send(item).await.is_err() {
                                        break;
                                    }
                                }
                            });
                            return Ok(rx);
                        }
                        Err(e) => {
                            if provider == "local_direct" {
                                return Err(e);
                            }
                            warn!(
                                "⚠️ Primary sovereign streaming failed: {:?}. Attempting standby failover...",
                                e
                            );
                        }
                    }
                }
                Ok(Err(_closed)) => {
                    warn!(
                        "⚠️ Primary engine semaphore closed unexpectedly; treating as saturated."
                    );
                    if provider == "local_direct" {
                        return Err(InferenceError::ExecutionFailed(
                            "Primary sovereign engine unavailable (semaphore closed).".to_string(),
                        ));
                    }
                }
                Err(_elapsed) => {
                    warn!(
                        "⚠️ Primary sovereign engine saturated (all slots busy for {:?}) — failing over instead of queuing further...",
                        primary_engine_acquire_timeout()
                    );
                    if provider == "local_direct" {
                        return Err(InferenceError::ExecutionFailed(
                            "Primary sovereign engine busy (backend saturated); local_direct disallows failover.".to_string(),
                        ));
                    }
                }
            }

            if let Some(secondary_engine) = &self.secondary_engine {
                let secondary_req = with_explicit_endpoint(
                    local_req.clone(),
                    std::env::var("HERA_SECONDARY_OMNI_URL").ok(),
                );
                match secondary_engine.generate_stream(secondary_req).await {
                    Ok(stream) => return Ok(stream),
                    Err(e) => {
                        if provider == "local" {
                            return Err(e);
                        }
                        warn!(
                            "⚠️ Secondary sovereign streaming failed: {:?}. Attempting tertiary failover...",
                            e
                        );
                    }
                }
            }

            if let Some(tertiary_engine) = &self.tertiary_engine {
                let tertiary_req = with_explicit_endpoint(
                    local_req.clone(),
                    std::env::var("HERA_TERTIARY_OMNI_URL").ok(),
                );
                match tertiary_engine.generate_stream(tertiary_req).await {
                    Ok(stream) => return Ok(stream),
                    Err(e) => {
                        if provider == "local" {
                            return Err(e);
                        }
                        warn!(
                            "⚠️ Tertiary sovereign streaming failed: {:?}. Attempting cloud failover...",
                            e
                        );
                    }
                }
            }
        }

        if (provider == "auto" || provider == "gemini" || provider == "cloud") && cloud_allowed {
            if let Err(reason) = crate::ai::cloud_budget::check_and_record_cloud_call(
                crate::ai::cloud_budget::estimate_tokens(&cloud_req),
            ) {
                warn!(
                    "🛑 Cloud streaming call denied by cost-safety gate: {}",
                    reason
                );
                return Err(InferenceError::ExecutionFailed(reason));
            }
            info!(
                "☁️ Re-routing STREAMING inference onto Cloud MultiModal Engine (model='{}')...",
                cloud_req.model
            );
            match self.cloud_engine.generate_stream(cloud_req).await {
                Ok(stream) => {
                    return Ok(stream);
                }
                Err(e) => {
                    error!("☁️ Cloud Streaming fallback also crashed: {:?}", e);
                }
            }
        }

        if !cloud_allowed && (provider == "auto" || provider == "gemini" || provider == "cloud") {
            return Err(InferenceError::ExecutionFailed(
                "External cloud fallback is disallowed by current sovereign policy.".to_string(),
            ));
        }

        Err(InferenceError::ExecutionFailed(
            "All streaming bounds collapsed (Local & Cloud). Cannot proceed.".to_string(),
        ))
    }
}

#[derive(Debug, Default, Deserialize)]
struct WorkloadPolicyFile {
    #[serde(default)]
    workload_policies: Vec<WorkloadPolicyEntry>,
}

#[derive(Debug, Deserialize)]
struct WorkloadPolicyEntry {
    policy: WorkloadPolicy,
}

#[derive(Debug, Deserialize)]
struct WorkloadPolicy {
    class: String,
    cloud_fallback: String,
}

#[derive(Debug, Default, Deserialize)]
struct AppCloudPolicyFile {
    #[serde(default)]
    app_cloud_policies: Vec<AppCloudPolicyEntry>,
}

#[derive(Debug, Deserialize)]
struct AppCloudPolicyEntry {
    policy: AppCloudPolicy,
}

#[derive(Debug, Deserialize)]
struct AppCloudPolicy {
    app: String,
    cloud_fallback: String,
}

#[derive(Debug, Default, Deserialize)]
struct NodeRegistryFile {
    #[serde(default)]
    nodes: Vec<NodeEntry>,
}

#[derive(Debug, Deserialize)]
struct NodeEntry {
    id: String,
    profile: NodeProfile,
}

#[derive(Debug, Deserialize)]
struct NodeProfile {
    alias: String,
    hostname: String,
    #[serde(default)]
    network_identity: NodeNetworkIdentity,
}

#[derive(Debug, Default, Deserialize)]
struct NodeNetworkIdentity {
    #[serde(default)]
    network_profile: String,
}

#[derive(Debug, Default, Deserialize)]
struct NetworkProfileFile {
    #[serde(default)]
    network_profiles: Vec<NetworkProfileEntry>,
}

#[derive(Debug, Deserialize)]
struct NetworkProfileEntry {
    id: String,
    profile: NetworkProfile,
}

#[derive(Debug, Deserialize)]
struct NetworkProfile {
    mode: String,
}

fn prepare_local_request(req: &ChatRequest) -> ChatRequest {
    let mut normalized = req.clone();
    if normalized.model.is_empty() || normalized.model.starts_with("hera-") {
        normalized.model = std::env::var("HERA_OPENAI_MODEL")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_default();
    }
    normalized.provider = Some("local".to_string());
    normalized
}

fn prepare_cloud_request(req: &ChatRequest) -> ChatRequest {
    let mut normalized = req.clone();
    if normalized.model.is_empty() || normalized.model.starts_with("hera-") {
        normalized.model = std::env::var("OPENROUTER_DEFAULT_MODEL")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| "nvidia/nemotron-3-nano-30b-a3b:free".to_string());
    }
    normalized.provider = Some("cloud".to_string());
    normalized
}

fn with_endpoint_override(req: ChatRequest, env_key: &str) -> ChatRequest {
    with_explicit_endpoint(req, std::env::var(env_key).ok())
}

fn with_explicit_endpoint(mut req: ChatRequest, endpoint: Option<String>) -> ChatRequest {
    if let Some(endpoint) = endpoint.filter(|value| !value.trim().is_empty()) {
        req.endpoint = Some(endpoint);
    }
    req
}

/// Sovereign-first master switch for paid cloud inference.
///
/// Cloud is DENIED by default. It is only enabled when `HERA_ALLOW_CLOUD_FALLBACK`
/// is explicitly set to a truthy value (`1` / `true` / `yes`). This is the opposite
/// of an opt-out: an absent or empty var means "no cloud, no charges".
///
/// Background: a deploy without this var set silently billed ~$100 to OpenRouter
/// because the previous logic defaulted to ALLOW (2026-06-09 incident).
pub fn cloud_globally_enabled() -> bool {
    std::env::var("HERA_ALLOW_CLOUD_FALLBACK")
        .ok()
        .is_some_and(|value| matches!(value.trim(), "1" | "true" | "TRUE" | "True" | "yes" | "YES"))
}

fn cloud_fallback_allowed(req: &ChatRequest) -> bool {
    // Sovereign-first: cloud stays off unless explicitly enabled.
    if !cloud_globally_enabled() {
        return false;
    }

    if current_network_mode().as_deref() == Some("enterprise_private") {
        return false;
    }

    // Cost-safety gate (Capa 3), app→never_external: takes precedence over
    // the workload-class inference below, which only looks at request SHAPE
    // (image present? stt/tts model? max_tokens/model name?) and — as of
    // this writing — never actually emits "stateful" for a normal chat
    // request. Compliance-sensitive apps (legal/judicial data, Memento's
    // cross-app memory store) must never leave sovereign compute regardless
    // of what the classifier concludes about THIS particular message.
    // Registry-driven (`Apps/OS-v3/registry/app_cloud_policy.yaml`) so new
    // apps can be added without a rebuild.
    if let Some(app_name) = req.app.as_deref() {
        if app_cloud_fallback_policy(app_name).as_deref() == Some("never_external") {
            return false;
        }
    }

    match inferred_workload_class(req).as_deref() {
        Some("stateful") => false,
        Some(class_name) => workload_cloud_fallback_policy(class_name)
            .map(|policy| policy != "never_external")
            .unwrap_or(true),
        None => true,
    }
}

fn inferred_workload_class(req: &ChatRequest) -> Option<String> {
    if req.stt_model.is_some() || req.tts_model.is_some() {
        return Some("speech".to_string());
    }
    let has_image = req.messages.iter().any(|message| match &message.content {
        crate::ai::MessageContent::Parts(parts) => parts
            .iter()
            .any(|part| matches!(part, crate::ai::ContentPart::ImageUrl { .. })),
        _ => false,
    });
    if has_image || req.vision_model.is_some() {
        return Some("vision_light".to_string());
    }
    if req.max_tokens.unwrap_or_default() >= 4096 || req.model.contains("70b") {
        return Some("llm_heavy".to_string());
    }
    Some("llm_small".to_string())
}

fn workload_cloud_fallback_policy(class_name: &str) -> Option<String> {
    let registry_dir = locate_os_v3_registry_dir()?;
    let file =
        load_yaml::<WorkloadPolicyFile>(&registry_dir.join("workload_policies.yaml")).ok()?;
    file.workload_policies
        .into_iter()
        .find(|entry| entry.policy.class == class_name)
        .map(|entry| entry.policy.cloud_fallback)
}

fn app_cloud_fallback_policy(app_name: &str) -> Option<String> {
    let registry_dir = locate_os_v3_registry_dir()?;
    let file = load_yaml::<AppCloudPolicyFile>(&registry_dir.join("app_cloud_policy.yaml")).ok()?;
    file.app_cloud_policies
        .into_iter()
        .find(|entry| entry.policy.app.eq_ignore_ascii_case(app_name))
        .map(|entry| entry.policy.cloud_fallback)
}

fn current_network_mode() -> Option<String> {
    let registry_dir = locate_os_v3_registry_dir()?;
    let nodes = load_yaml::<NodeRegistryFile>(&registry_dir.join("nodes.yaml"))
        .ok()?
        .nodes;
    let profiles = load_yaml::<NetworkProfileFile>(&registry_dir.join("network_profiles.yaml"))
        .ok()?
        .network_profiles;
    let current_alias = current_node_alias(&nodes)?;
    let node = nodes
        .iter()
        .find(|node| node.id == current_alias || node.profile.alias == current_alias)?;
    let profile_id = &node.profile.network_identity.network_profile;
    profiles
        .into_iter()
        .find(|profile| profile.id == *profile_id)
        .map(|profile| profile.profile.mode)
}

fn locate_os_v3_registry_dir() -> Option<PathBuf> {
    if let Ok(explicit) = std::env::var("HERA_OSV3_REGISTRY_DIR") {
        let path = PathBuf::from(explicit);
        if path.exists() {
            return Some(path);
        }
    }

    let candidates = [
        "../../Apps/OS-v3/registry",
        "../Apps/OS-v3/registry",
        "../../../Apps/OS-v3/registry",
        "./Apps/OS-v3/registry",
    ];

    candidates
        .iter()
        .map(PathBuf::from)
        .find(|path| path.exists())
}

fn load_yaml<T>(path: &Path) -> Result<T, std::io::Error>
where
    T: for<'de> Deserialize<'de> + Default,
{
    if !path.exists() {
        return Ok(T::default());
    }
    let raw = fs::read_to_string(path)?;
    serde_yaml::from_str(&raw).map_err(|error| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Failed to parse {}: {}", path.display(), error),
        )
    })
}

fn current_node_alias(nodes: &[NodeEntry]) -> Option<String> {
    if let Ok(alias) = std::env::var("HERA_NODE_ALIAS") {
        if !alias.trim().is_empty() {
            return Some(alias);
        }
    }

    let hostname = std::env::var("HOSTNAME").ok()?;
    nodes
        .iter()
        .find(|node| {
            node.profile.hostname == hostname
                || node.profile.alias == hostname
                || node.id == hostname
        })
        .map(|node| node.id.clone())
}
