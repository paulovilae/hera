use std::sync::Arc;
use tracing::{Level, info};

use hera_core::capabilities::{CapabilityId, CapabilityRegistry};
use hera_core::hardware::discover_docker_services;
use hera_core::ai::SpeechToTextEngine;
use hera_core::ipc_server::{IpcState, serve};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_max_level(Level::DEBUG)
        .init();

    info!("🕯️ Candle Core Hardware Orchestrator - Initializing");

    // Load environment before any engine initialization. Hera often runs from
    // `Hera/hera-core`, but the actual env files may live in parent workspaces.
    let env_candidates = ["./.env.local", "../.env.local", "../../.env.local"];
    load_env_from_candidates(&env_candidates);

    // Last-resort recovery for cloud fallback in dev worktrees where the key was
    // stored in sibling app env files instead of Hera's own env file.
    if std::env::var("OPENROUTER_API_KEY")
        .unwrap_or_default()
        .is_empty()
    {
        let fallback_candidates = [
            "../../Apps/Vetra-v2-legacy/.env.local",
            "../../../vetra2/.env",
            "../../../builder/.env.local",
        ];
        if let Some(key) = find_env_value(&fallback_candidates, "OPENROUTER_API_KEY") {
            unsafe {
                std::env::set_var("OPENROUTER_API_KEY", key);
            }
            info!("🔐 Recovered OPENROUTER_API_KEY from sibling app environment");
        }
    }

    let services = discover_docker_services();
    info!("Active Local Containers: {}", services.len());

    let capabilities = CapabilityRegistry::detect();
    capabilities.log_summary();

    // Initialize Flux Native engine lazily or synchronously
    // By Sovereign Directive, Candle FLUX is deprecated due to VRAM inefficiency.
    // Image Generation is now delegated to the native sd.cpp REST node in ipc_server.rs.
    let flux_engine: Option<Arc<hera_core::ai::engine_flux::FluxEngine>> = None;

    // Initialize Parler-TTS Native engine
    let parler_engine = if capabilities.runtime_enabled(CapabilityId::AudioTts) {
        info!("🎤 Initializing Native Parler-TTS Audio Engine...");
        match hera_core::ai::engine_parler::ParlerEngine::new() {
            Ok(engine) => {
                info!("🎤 Native Parler-TTS Engine mounted to VRAM.");
                Some(Arc::new(engine))
            }
            Err(e) => {
                tracing::error!("Failed to mount Native Parler-TTS: {:?}", e);
                None
            }
        }
    } else {
        None
    };

    // Initialize Whisper Native engine
    let whisper_engine: Option<Arc<dyn SpeechToTextEngine + Send + Sync>> = if capabilities.runtime_enabled(CapabilityId::AudioStt) {
        info!("👂 Initializing Native Whisper STT Engine...");
        match std::env::var("HERA_STT_BACKEND")
            .unwrap_or_else(|_| "whisper".to_string())
            .to_ascii_lowercase()
            .as_str()
        {
            "faster-whisper" | "faster_whisper" => match hera_core::ai::engine_faster_whisper::FasterWhisperEngine::new() {
                Ok(engine) => {
                    info!("👂 faster-whisper STT backend mounted.");
                    Some(Arc::new(engine))
                }
                Err(e) => {
                    tracing::error!("Failed to mount faster-whisper STT backend: {:?}", e);
                    None
                }
            },
            _ => match hera_core::ai::engine_whisper::WhisperEngine::new() {
                Ok(engine) => {
                    info!("👂 Native Whisper Engine mounted to VRAM.");
                    Some(Arc::new(engine))
                }
                Err(e) => {
                    tracing::error!("Failed to mount Native Whisper: {:?}", e);
                    None
                }
            },
        }
    } else {
        None
    };

    // Initialize LlamaBackend globally
    let _llama_backend = Arc::new(
        llama_cpp_2::llama_backend::LlamaBackend::init()
            .expect("Failed to initialize global LlamaBackend"),
    );

    // Mount Sovereign Local LLM Engine
    let local_engine: Arc<dyn hera_core::ai::LLMEngine + Send + Sync> = if capabilities
        .runtime_enabled(CapabilityId::LocalLlm)
    {
        info!("🧠 Initializing Sovereign Local LLM Engine (via Local Omni Node)...");
        let local_omni = hera_core::ai::openai_compat::OpenAICompatEngine::new(
            "http://127.0.0.1:8080/v1/chat/completions".to_string(),
            "".to_string(),
        );
        info!("🧠 Sovereign Native Omni Engine mounted!");
        Arc::new(local_omni)
    } else {
        info!("🧠 Sovereign Local LLM disabled via environment flag (HERA_ENABLE_LLM).");
        struct DisabledEngine;
        #[async_trait::async_trait]
        impl hera_core::ai::LLMEngine for DisabledEngine {
            async fn generate_content(
                &self,
                _req: hera_core::ai::ChatRequest,
            ) -> Result<hera_core::ai::ChatResponse, hera_core::ai::InferenceError> {
                Err(hera_core::ai::InferenceError::ExecutionFailed(
                    "Local LLM engine is explicitly disabled.".to_string(),
                ))
            }
        }
        Arc::new(DisabledEngine) as Arc<dyn hera_core::ai::LLMEngine + Send + Sync>
    };

    // Mount OpenRouter Cloud Fallback Engine
    let openrouter_key = std::env::var("OPENROUTER_API_KEY").unwrap_or_default();
    let openrouter_model = std::env::var("OPENROUTER_DEFAULT_MODEL")
        .unwrap_or_else(|_| "nvidia/nemotron-3-nano-30b-a3b:free".to_string());
    let cloud_engine: Arc<dyn hera_core::ai::LLMEngine + Send + Sync> = if !openrouter_key
        .is_empty()
    {
        info!(
            "☁️ OpenRouter cloud fallback configured (default model: {})",
            openrouter_model
        );
        Arc::new(hera_core::ai::openai_compat::OpenAICompatEngine::new(
            "https://openrouter.ai/api/v1/chat/completions".to_string(),
            openrouter_key.clone(),
        ))
    } else {
        tracing::warn!("⚠️ No OPENROUTER_API_KEY set. Cloud fallback disabled.");
        struct NoCloudEngine;
        #[async_trait::async_trait]
        impl hera_core::ai::LLMEngine for NoCloudEngine {
            async fn generate_content(
                &self,
                _req: hera_core::ai::ChatRequest,
            ) -> Result<hera_core::ai::ChatResponse, hera_core::ai::InferenceError> {
                Err(hera_core::ai::InferenceError::ExecutionFailed(
                    "No cloud fallback configured.".to_string(),
                ))
            }
        }
        Arc::new(NoCloudEngine) as Arc<dyn hera_core::ai::LLMEngine + Send + Sync>
    };

    // Compose: Local-first → Cloud fallback
    let router_engine: Arc<dyn hera_core::ai::LLMEngine + Send + Sync> = Arc::new(
        hera_core::ai::router::RouterEngine::new(local_engine, cloud_engine),
    );

    // Orchestrator Backend — uses same local-first RouterEngine for sovereign operation
    info!("🧠 Context Orchestrator mounted via local Sovereign Router (Qwen FFI → cloud fallback)");
    let orchestrator_engine: Arc<dyn hera_core::ai::LLMEngine + Send + Sync> =
        Arc::clone(&router_engine);

    // Mount Native Vision Engine (Unified Omni)
    let vision_engine: Option<Arc<dyn hera_core::ai::LLMEngine + Send + Sync>> = {
        info!("👁️ Native Vision Engine (Unified Local Omni) mounted via local network.");
        Some(Arc::new(
            hera_core::ai::openai_compat::OpenAICompatEngine::new(
                "http://127.0.0.1:8080/v1/chat/completions".to_string(),
                "".to_string(),
            ),
        ))
    };

    let context_engine = hera_core::ai::context_engine::ContextEngine::new(
        orchestrator_engine,
        Arc::clone(&router_engine),
        vision_engine.clone(),
    );

    // Mount Micro-LLM Uncensored Fallback Engine (Unified Omni)
    let micro_engine: Option<Arc<dyn hera_core::ai::LLMEngine + Send + Sync>> = {
        info!("🧠 Initializing Micro-LLM Engine (Unified Local Omni)...");
        Some(Arc::new(
            hera_core::ai::openai_compat::OpenAICompatEngine::new(
                "http://127.0.0.1:8080/v1/chat/completions".to_string(),
                "".to_string(),
            ),
        ))
    };

    // Mount the Headless IPC Socket Layer
    let state = IpcState {
        engine: Arc::new(context_engine),
        local_engine: Arc::clone(&router_engine),
        flux_engine,
        parler_engine,
        whisper_engine,
        vision_engine: vision_engine.clone(),
        micro_engine,
    };

    // Default socket path for Vilaros OS & Imaginclaw
    let socket_path = "/tmp/hera-core.sock";

    info!("🚀 Core AI Layer booting in PURE SPEED mode.");

    // Spawn REST API server in the background
    tokio::spawn(async move {
        hera_core::rest_api::serve_rest_api(3002).await;
    });

    // Spawn autonomous emergency watchdog
    hera_core::watchdog::spawn_watchdog();

    // Block on IPC Listener natively
    if let Err(e) = serve(socket_path, state).await {
        tracing::error!("❌ Fatal IPC Server Error: {}", e);
    }
}

fn load_env_from_candidates(paths: &[&str]) {
    for path in paths {
        let env_path = std::path::Path::new(path);
        if !env_path.exists() {
            continue;
        }
        if let Ok(contents) = std::fs::read_to_string(env_path) {
            for line in contents.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                if let Some((key, val)) = line.split_once('=') {
                    unsafe {
                        std::env::set_var(key.trim(), normalize_env_value(val));
                    }
                }
            }
            info!("📄 Loaded environment from {}", env_path.display());
        }
    }
}

fn find_env_value(paths: &[&str], key: &str) -> Option<String> {
    for path in paths {
        let env_path = std::path::Path::new(path);
        let contents = std::fs::read_to_string(env_path).ok()?;
        for line in contents.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (candidate_key, candidate_value) = line.split_once('=')?;
            if candidate_key.trim() == key {
                let value = normalize_env_value(candidate_value);
                if !value.is_empty() {
                    return Some(value);
                }
            }
        }
    }
    None
}

fn normalize_env_value(value: &str) -> String {
    value
        .trim()
        .trim_matches('"')
        .trim_matches('\'')
        .to_string()
}
