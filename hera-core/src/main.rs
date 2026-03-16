use std::sync::Arc;
use tokio::net::TcpListener;
use tracing::{Level, info};

use hera_core::hardware::discover_docker_services;
use hera_core::ipc_server::{serve, IpcState};
use hera_core::ai::native_engine::get_or_init_engine;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_max_level(Level::DEBUG)
        .init();

    info!("🕯️ Candle Core Hardware Orchestrator - Initializing");

    // Load environment from .env.local FIRST — before any engine initialization
    let env_path = std::path::Path::new("./.env.local");
    if env_path.exists() {
        if let Ok(contents) = std::fs::read_to_string(env_path) {
            for line in contents.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') { continue; }
                if let Some((key, val)) = line.split_once('=') {
                    unsafe { std::env::set_var(key.trim(), val.trim()); }
                }
            }
            info!("📄 Loaded environment from .env.local");
        }
    }

    let services = discover_docker_services();
    info!("Active Local Containers: {}", services.len());
    
    // Initialize Flux Native engine lazily or synchronously
    // By Sovereign Directive, Candle FLUX is deprecated due to VRAM inefficiency.
    // Image Generation is now delegated to the native sd.cpp REST node in ipc_server.rs.
    let flux_engine: Option<Arc<hera_core::ai::engine_flux::FluxEngine>> = None;

    // Initialize Parler-TTS Native engine
    let parler_engine = if std::env::var("HERA_ENABLE_PARLER").unwrap_or_else(|_| "true".to_string()) == "true" {
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
    let whisper_engine = if std::env::var("HERA_ENABLE_WHISPER").unwrap_or_else(|_| "true".to_string()) == "true" {
        info!("👂 Initializing Native Whisper STT Engine...");
        match hera_core::ai::engine_whisper::WhisperEngine::new() {
            Ok(engine) => {
                info!("👂 Native Whisper Engine mounted to VRAM.");
                Some(Arc::new(engine))
            }
            Err(e) => {
                tracing::error!("Failed to mount Native Whisper: {:?}", e);
                None
            }
        }
    } else {
        None
    };

    // Initialize LlamaBackend globally
    let llama_backend = Arc::new(llama_cpp_2::llama_backend::LlamaBackend::init().expect("Failed to initialize global LlamaBackend"));

    // Mount Sovereign Local LLM Engine
    let enable_llm = std::env::var("HERA_ENABLE_LLM").unwrap_or_else(|_| "true".to_string()) == "true";
    let local_engine: Arc<dyn hera_core::ai::LLMEngine + Send + Sync> = if enable_llm {
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
            async fn generate_content(&self, _req: hera_core::ai::ChatRequest) -> Result<hera_core::ai::ChatResponse, hera_core::ai::InferenceError> {
                Err(hera_core::ai::InferenceError::ExecutionFailed("Local LLM engine is explicitly disabled.".to_string()))
            }
        }
        Arc::new(DisabledEngine) as Arc<dyn hera_core::ai::LLMEngine + Send + Sync>
    };

    // Mount OpenRouter Cloud Fallback Engine
    let openrouter_key = std::env::var("OPENROUTER_API_KEY").unwrap_or_default();
    let openrouter_model = std::env::var("OPENROUTER_DEFAULT_MODEL")
        .unwrap_or_else(|_| "nvidia/nemotron-3-nano-30b-a3b:free".to_string());
    let cloud_engine: Arc<dyn hera_core::ai::LLMEngine + Send + Sync> = if !openrouter_key.is_empty() {
        info!("☁️ OpenRouter cloud fallback configured (default model: {})", openrouter_model);
        Arc::new(hera_core::ai::openai_compat::OpenAICompatEngine::new(
            "https://openrouter.ai/api/v1/chat/completions".to_string(),
            openrouter_key.clone(),
        ))
    } else {
        tracing::warn!("⚠️ No OPENROUTER_API_KEY set. Cloud fallback disabled.");
        struct NoCloudEngine;
        #[async_trait::async_trait]
        impl hera_core::ai::LLMEngine for NoCloudEngine {
            async fn generate_content(&self, _req: hera_core::ai::ChatRequest) -> Result<hera_core::ai::ChatResponse, hera_core::ai::InferenceError> {
                Err(hera_core::ai::InferenceError::ExecutionFailed("No cloud fallback configured.".to_string()))
            }
        }
        Arc::new(NoCloudEngine) as Arc<dyn hera_core::ai::LLMEngine + Send + Sync>
    };

    // Compose: Local-first → Cloud fallback
    let router_engine: Arc<dyn hera_core::ai::LLMEngine + Send + Sync> = Arc::new(hera_core::ai::router::RouterEngine::new(local_engine, cloud_engine));

    // Orchestrator Backend — uses same local-first RouterEngine for sovereign operation
    info!("🧠 Context Orchestrator mounted via local Sovereign Router (Qwen FFI → cloud fallback)");
    let orchestrator_engine: Arc<dyn hera_core::ai::LLMEngine + Send + Sync> = Arc::clone(&router_engine);

    // Mount Native Vision Engine (Unified Omni)
    let vision_engine: Option<Arc<dyn hera_core::ai::LLMEngine + Send + Sync>> = {
        info!("👁️ Native Vision Engine (Unified Local Omni) mounted via local network.");
        Some(Arc::new(hera_core::ai::openai_compat::OpenAICompatEngine::new(
            "http://127.0.0.1:8080/v1/chat/completions".to_string(),
            "".to_string(),
        )))
    };

    let context_engine = hera_core::ai::context_engine::ContextEngine::new(
        orchestrator_engine,
        Arc::clone(&router_engine),
        vision_engine.clone(),
    );

    // Mount Micro-LLM Uncensored Fallback Engine (Unified Omni)
    let micro_engine: Option<Arc<dyn hera_core::ai::LLMEngine + Send + Sync>> = {
        info!("🧠 Initializing Micro-LLM Engine (Unified Local Omni)...");
        Some(Arc::new(hera_core::ai::openai_compat::OpenAICompatEngine::new(
            "http://127.0.0.1:8080/v1/chat/completions".to_string(),
            "".to_string(),
        )))
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

    // Block on IPC Listener natively
    if let Err(e) = serve(socket_path, state).await {
        tracing::error!("❌ Fatal IPC Server Error: {}", e);
    }
}
