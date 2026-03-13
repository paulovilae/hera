use std::sync::Mutex;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2::{Config as Qwen2Config, ModelForCausalLM as Qwen2};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

use crate::ai::native_engine::EngineBackend;

pub(crate) fn load_hub_backend(
    model_id: &str,
    revision: &str,
    device: &Device,
) -> Result<(EngineBackend, Tokenizer, String), Box<dyn std::error::Error + Send + Sync>> {
    println!("[LLM_ENGINE]: Downloading/Loading {} weights...", model_id);
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));

    let tokenizer_filename = repo.get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename)?;

    let config_filename = repo.get("config.json")?;
    let qwen_config: Qwen2Config = serde_json::from_slice(&std::fs::read(config_filename)?)?;

    let weights_filename = repo.get("model.safetensors")?;
    let mut active_device = device.clone();
    let mut active_dtype = if matches!(active_device, Device::Cpu) {
        DType::F32
    } else {
        DType::F16
    };
    let mut vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_filename.clone()], active_dtype, &active_device)?
    };
    let mut model = Qwen2::new(&qwen_config, vb)?;

    if !matches!(active_device, Device::Cpu) {
        let warmup = Tensor::new(&[1u32], &active_device)
            .and_then(|t| t.unsqueeze(0))
            .and_then(|input| model.forward(&input, 0));
        if let Err(err) = warmup {
            let err_str = err.to_string();
            if err_str.contains("no cuda implementation for rms-norm") {
                println!("[LLM_ENGINE]: CUDA backend missing rms-norm kernel. Falling back to CPU.");
                active_device = Device::Cpu;
                active_dtype = DType::F32;
                vb = unsafe {
                    VarBuilder::from_mmaped_safetensors(
                        &[weights_filename.clone()],
                        active_dtype,
                        &active_device,
                    )?
                };
                model = Qwen2::new(&qwen_config, vb)?;
            } else {
                return Err(format!("CUDA warmup failed: {err_str}").into());
            }
        }
    }

    Ok((
        EngineBackend::Qwen2(Mutex::new(model)),
        tokenizer,
        model_id.to_string(),
    ))
}
