use std::sync::Arc;
use tokio::net::TcpListener;
use tracing::{Level, info};

use axum::{
    routing::get,
    Router,
};
use serde_json::json;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use hera_core::hardware::discover_docker_services;
use hera_core::api::routes::{create_router, ApiState};
use hera_core::ai::native_engine::get_or_init_engine;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_max_level(Level::DEBUG)
        .init();

    info!("🕯️ Candle Core Hardware Orchestrator - Initializing");

    // Load environment from .env.local FIRST — before any engine initialization
    let env_path = std::path::Path::new("/home/paulo/Programs/apps/hera/.env.local");
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
    let flux_engine = if std::env::var("HERA_ENABLE_FLUX").unwrap_or_else(|_| "true".to_string()) == "true" {
        info!("🎨 Initializing Native FLUX.1 Vision Engine...");
        match hera_core::ai::engine_flux::FluxEngine::new() {
            Ok(engine) => {
                info!("🎨 Native FLUX.1 Vision Engine mounted to VRAM.");
                Some(Arc::new(engine))
            }
            Err(e) => {
                tracing::error!("Failed to mount Native FLUX.1: {:?}", e);
                None
            }
        }
    } else {
        None
    };

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
        info!("🧠 Initializing Sovereign Local LLM Engine (Qwen GGUF)...");
        match hera_core::ai::llama_ffi_engine::LlamaFfiEngine::new(llama_backend.clone(), "/data/models/llm-stack/Qwen3.5-4B-Abliterated-Claude-4.6-Opus-Reasoning-Distilled.Q4_K_M.gguf") {
            Ok(engine) => {
                info!("🧠 Sovereign FFI LLM Engine mounted!");
                Arc::new(engine)
            }
            Err(e) => {
                tracing::warn!("⚠️ Local FFI LLM Engine failed to mount: {}. Using cloud-only mode.", e);
                struct FallbackEngine;
                #[async_trait::async_trait]
                impl hera_core::ai::LLMEngine for FallbackEngine {
                    async fn generate_content(&self, _req: hera_core::ai::ChatRequest) -> Result<hera_core::ai::ChatResponse, hera_core::ai::InferenceError> {
                        Err(hera_core::ai::InferenceError::ExecutionFailed("No local LLM model loaded. Set HERA_CANDLE_MODEL_ID to a valid GGUF path.".to_string()))
                    }
                }
                Arc::new(FallbackEngine) as Arc<dyn hera_core::ai::LLMEngine + Send + Sync>
            }
        }
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

    // Mount Native Vision Engine (Moondream)
    let vision_engine: Option<Arc<dyn hera_core::ai::LLMEngine + Send + Sync>> = match hera_core::ai::engine_moondream::MoondreamFfiEngine::new().await {
        Ok(engine) => {
            info!("👁️ Native Vision Engine (Moondream) mounted to VRAM.");
            Some(Arc::new(engine))
        }
        Err(e) => {
            tracing::error!("Failed to mount Native Vision Engine: {:?}", e);
            None
        }
    };

    let context_engine = hera_core::ai::context_engine::ContextEngine::new(
        orchestrator_engine,
        Arc::clone(&router_engine),
        vision_engine.clone(),
    );

    // Mount Micro-LLM Uncensored Fallback Engine
    let micro_engine_path = "/data/models/llm-stack/Qwen3.5-4B-Abliterated-Claude-4.6-Opus-Reasoning-Distilled.Q4_K_M.gguf";
    let micro_engine: Option<Arc<dyn hera_core::ai::LLMEngine + Send + Sync>> = if std::path::Path::new(micro_engine_path).exists() {
        info!("🧠 Initializing Micro-LLM Uncensored Engine (Qwen 1.5B Abliterated)...");
        match hera_core::ai::llama_ffi_engine::LlamaFfiEngine::new(llama_backend.clone(), micro_engine_path) {
            Ok(engine) => {
                info!("🧠 Micro-LLM Engine mounted!");
                Some(Arc::new(engine))
            }
            Err(e) => {
                tracing::warn!("⚠️ Micro-LLM failed to mount: {}", e);
                None
            }
        }
    } else {
        tracing::warn!("⚠️ Micro-LLM model NOT found. NSFW bypass will use raw prompt directly.");
        None
    };

    // Mount the API Layer
    let state = ApiState {
        engine: Arc::new(context_engine),
        local_engine: Arc::clone(&router_engine),
        flux_engine,
        parler_engine,
        whisper_engine,
        vision_engine: vision_engine.clone(),
        micro_engine,
    };
    
    let app = create_router(state);
    let port = "3305";
    let listener = TcpListener::bind(format!("0.0.0.0:{}", port)).await.unwrap();

    info!("🚀 Core API Layer bound and listening on HTTP port {}", port);
    
    // Block on Axum server natively
    axum::serve(listener, app).await.unwrap();
}
