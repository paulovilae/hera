use anyhow::{Context, Result};
use candle_core::{IndexOp, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::{clip, flux};
use image::ImageFormat;
use std::io::Cursor;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;

#[derive(Clone)]
pub struct FluxEngine {
    device: candle_core::Device,
    t5_model: Arc<tokio::sync::Mutex<crate::ai::q8_t5::T5EncoderModel>>,
    t5_tokenizer: Arc<Tokenizer>,
    clip_model: Arc<Mutex<clip::text_model::ClipTextTransformer>>,
    clip_tokenizer: Arc<Tokenizer>,
    flux_model: Arc<Mutex<flux::quantized_model::Flux>>,
    ae_model: Arc<Mutex<flux::autoencoder::AutoEncoder>>,
}

impl FluxEngine {
    pub fn new() -> Result<Self> {
        let device_id = std::env::var("HERA_FLUX_DEVICE_ID")
            .unwrap_or_else(|_| "1".to_string())
            .parse::<usize>()
            .unwrap_or(1);
        let device = candle_core::Device::new_cuda(device_id).unwrap_or(candle_core::Device::Cpu);
        // By default, Candle works best in bf16/f32 for FLUX
        let dtype = device.bf16_default_to_f32();

        let api = match std::env::var("HF_TOKEN") {
            Ok(token) => hf_hub::api::sync::ApiBuilder::new()
                .with_token(Some(token))
                .build()?,
            Err(_) => hf_hub::api::sync::Api::new()?,
        };
        let bf_repo = api.repo(hf_hub::Repo::model(
            "receptektas/black-forest-labs-ae_safetensors".to_string(),
        ));

        // 1. T5 Loading (Using Q8_0 Quantized GGUF Encoder-only model to save ~32GB of VRAM/Disk overhead)
        let t5_config_repo = api.repo(hf_hub::Repo::model(
            "city96/t5-v1_1-xxl-encoder-bf16".to_string(),
        ));
        let t5_config_str = std::fs::read_to_string(t5_config_repo.get("config.json")?)?;
        let t5_config: crate::ai::q8_t5::Config = serde_json::from_str(&t5_config_str)?;

        let t5_repo = api.repo(hf_hub::Repo::model(
            "city96/t5-v1_1-xxl-encoder-gguf".to_string(),
        ));
        let t5_file = t5_repo.get("t5-v1_1-xxl-encoder-Q8_0.gguf")?;
        let t5_vb = crate::ai::q8_t5::VarBuilder::from_gguf(t5_file, &device)?;
        let t5_model = crate::ai::q8_t5::T5EncoderModel::load(t5_vb, &t5_config)?;

        let t5_tokenizer_file = api
            .model("lmz/mt5-tokenizers".to_string())
            .get("t5-v1_1-xxl.tokenizer.json")?;
        let t5_tokenizer = Tokenizer::from_file(t5_tokenizer_file)
            .map_err(|e| anyhow::anyhow!("Failed to load T5 tokenizer: {}", e))?;

        // 2. CLIP Loading
        let clip_repo = api.repo(hf_hub::Repo::model(
            "openai/clip-vit-large-patch14".to_string(),
        ));
        let clip_file = clip_repo.get("model.safetensors")?;
        let clip_vb = unsafe { VarBuilder::from_mmaped_safetensors(&[clip_file], dtype, &device)? };
        let clip_config = clip::text_model::ClipTextConfig {
            vocab_size: 49408,
            projection_dim: 768,
            activation: clip::text_model::Activation::QuickGelu,
            intermediate_size: 3072,
            embed_dim: 768,
            max_position_embeddings: 77,
            pad_with: None,
            num_hidden_layers: 12,
            num_attention_heads: 12,
        };
        let clip_model =
            clip::text_model::ClipTextTransformer::new(clip_vb.pp("text_model"), &clip_config)?;
        let clip_tokenizer_file = clip_repo.get("tokenizer.json")?;
        let clip_tokenizer = Tokenizer::from_file(clip_tokenizer_file)
            .map_err(|e| anyhow::anyhow!("Failed to load CLIP tokenizer: {}", e))?;

        // 3. FLUX Loading (Quantized quickly)
        let flux_cfg = flux::model::Config::schnell();
        let flux_file = api
            .repo(hf_hub::Repo::model("lmz/candle-flux".to_string()))
            .get("flux1-schnell.gguf")?;
        let flux_vb =
            candle_transformers::quantized_var_builder::VarBuilder::from_gguf(flux_file, &device)?;
        let flux_model = flux::quantized_model::Flux::new(&flux_cfg, flux_vb)?;

        // 4. AutoEncoder Loading
        let ae_cfg = flux::autoencoder::Config::schnell();
        let ae_file = bf_repo.get("ae.safetensors")?;
        let ae_vb = unsafe { VarBuilder::from_mmaped_safetensors(&[ae_file], dtype, &device)? };
        let ae_model = flux::autoencoder::AutoEncoder::new(&ae_cfg, ae_vb)?;

        Ok(Self {
            device,
            t5_model: Arc::new(tokio::sync::Mutex::new(t5_model)),
            t5_tokenizer: Arc::new(t5_tokenizer),
            clip_model: Arc::new(Mutex::new(clip_model)),
            clip_tokenizer: Arc::new(clip_tokenizer),
            flux_model: Arc::new(Mutex::new(flux_model)),
            ae_model: Arc::new(Mutex::new(ae_model)),
        })
    }

    pub async fn generate_image(
        &self,
        prompt: &str,
        width: usize,
        height: usize,
    ) -> Result<Vec<u8>> {
        let dtype = self.device.bf16_default_to_f32();

        let mut t5_tokens = self
            .t5_tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("{}", e))?
            .get_ids()
            .to_vec();
        t5_tokens.resize(256, 0);

        let t5_model_clone = self.t5_model.clone();
        let device_clone = self.device.clone();
        let t5_emb = tokio::task::spawn_blocking(move || {
            let mut model = t5_model_clone.blocking_lock();
            let tokens = Tensor::new(&t5_tokens[..], &device_clone)?.unsqueeze(0)?;
            model.forward(&tokens)
        })
        .await
        .map_err(|e| anyhow::anyhow!("T5 Panic: {}", e))??;

        let clip_tokens = self
            .clip_tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("{}", e))?
            .get_ids()
            .to_vec();

        let clip_model_clone = self.clip_model.clone();
        let device_clone = self.device.clone();
        let clip_emb = tokio::task::spawn_blocking(move || {
            let model = clip_model_clone.blocking_lock();
            let tokens = Tensor::new(&clip_tokens[..], &device_clone)?.unsqueeze(0)?;
            model.forward(&tokens)
        })
        .await
        .map_err(|e| anyhow::anyhow!("CLIP Panic: {}", e))??;

        let img_noise =
            flux::sampling::get_noise(1, height, width, &self.device)?.to_dtype(dtype)?;

        let state = flux::sampling::State::new(
            &t5_emb.to_dtype(candle_core::DType::F32)?,
            &clip_emb.to_dtype(candle_core::DType::F32)?,
            &img_noise.to_dtype(candle_core::DType::F32)?,
        )?;

        let timesteps = flux::sampling::get_schedule(4, None);

        let flux_model = self.flux_model.lock().await;
        // FLUX sampling loop internally executed by denoise array wrapper
        let img_latents = flux::sampling::denoise(
            &*flux_model,
            &state.img,
            &state.img_ids,
            &state.txt,
            &state.txt_ids,
            &state.vec,
            &timesteps,
            4.,
        )?
        .to_dtype(dtype)?;
        // drop lock here early
        drop(flux_model);

        let img_latents = flux::sampling::unpack(&img_latents, height, width)?;
        let ae = self.ae_model.lock().await;
        let decoded = ae.decode(&img_latents)?;
        drop(ae);

        let img =
            ((decoded.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(candle_core::DType::U8)?;

        let img_data = img.i(0)?.flatten_all()?.to_vec1::<u8>()?;
        // 3 channels (RGB) x H x W
        let h = img.dim(2)?;
        let w = img.dim(3)?;

        let rgb_image = image::RgbImage::from_raw(w as u32, h as u32, img_data)
            .context("Failed to reconstruct image from tensor raw bytes")?;

        let mut cursor = Cursor::new(Vec::new());
        rgb_image.write_to(&mut cursor, ImageFormat::Png)?;

        Ok(cursor.into_inner())
    }
}
