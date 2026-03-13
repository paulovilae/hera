use std::path::PathBuf;
use std::sync::Mutex;
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

use crate::ai::native_engine::EngineBackend;
use crate::ai::quantized_qwen3_moe_local::GGUFQWenMoE;

pub fn gguf_dtype_for_device(device: &Device) -> DType {
    if matches!(device, Device::Cpu) {
        DType::F32
    } else {
        DType::F16
    }
}

pub fn load_tokenizer_for_gguf() -> Result<Tokenizer, Box<dyn std::error::Error + Send + Sync>> {
    if let Ok(path) = std::env::var("HERA_CANDLE_TOKENIZER_PATH") {
        let tok = Tokenizer::from_file(path)?;
        return Ok(tok);
    }

    let tokenizer_repo = std::env::var("HERA_CANDLE_TOKENIZER_REPO")
        .unwrap_or_else(|_| "Qwen/Qwen3-8B".to_string());
    let tokenizer_revision = std::env::var("HERA_CANDLE_TOKENIZER_REVISION")
        .unwrap_or_else(|_| "main".to_string());
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        tokenizer_repo,
        RepoType::Model,
        tokenizer_revision,
    ));
    let tokenizer_filename = repo.get("tokenizer.json")?;
    let tok = Tokenizer::from_file(tokenizer_filename)?;
    Ok(tok)
}

pub(crate) fn load_gguf_backend(
    model_id: &str,
    device: &Device,
) -> Result<(EngineBackend, Tokenizer, String), Box<dyn std::error::Error + Send + Sync>> {
    let model_path = PathBuf::from(model_id);
    if !model_path.exists() {
        return Err(format!("GGUF model path not found: {}", model_path.display()).into());
    }

    let dtype = gguf_dtype_for_device(device);
    let tokenizer = load_tokenizer_for_gguf()?;

    let mut file = std::fs::File::open(&model_path)?;
    let content = gguf_file::Content::read(&mut file)?;

    let arch = match content.metadata.get("general.architecture") {
        Some(gguf_file::Value::String(s)) => s.clone(),
        _ => "unknown".to_string(),
    };

    if arch != "qwen2" && arch != "qwen3" && !arch.ends_with("moe") {
        return Err(format!("Unsupported Neural Architecture: '{}'. Hera Native Engine currently limits execution to sovereign 'qwen2/qwen3' topologies (Dense & MoE).", arch).into());
    }
    let is_moe = arch.ends_with("moe") || content.metadata.contains_key("qwen2.expert_feed_forward_length") || content.metadata.contains_key("qwen3.expert_feed_forward_length");

    let backend = if is_moe {
        println!(
            "[LLM_ENGINE]: Loading Qwen3 MoE GGUF backend from {} ...",
            model_path.display()
        );
        let model = GGUFQWenMoE::from_gguf(content, &mut file, device, dtype)?;
        EngineBackend::Qwen3MoeGguf(Mutex::new(model))
    } else if arch == "qwen3" {
        println!(
            "[LLM_ENGINE]: Loading standard Qwen3 Dense GGUF backend from {} ...",
            model_path.display()
        );
        let model = candle_transformers::models::quantized_qwen3::ModelWeights::from_gguf(content, &mut file, device)?;
        EngineBackend::Qwen3Gguf(Mutex::new(model))
    } else {
        println!(
            "[LLM_ENGINE]: Loading standard Qwen2 Dense GGUF backend from {} ...",
            model_path.display()
        );
        let model = candle_transformers::models::quantized_qwen2::ModelWeights::from_gguf(content, &mut file, device)?;
        EngineBackend::Qwen2Gguf(Mutex::new(model))
    };

    Ok((
        backend,
        tokenizer,
        model_id.to_string(),
    ))
}
