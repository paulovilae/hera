use anyhow::{Context, Result};
use candle_core::{DType, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::parler_tts::{Config, Model};
use hound::{SampleFormat, WavSpec, WavWriter};
use std::io::Cursor;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;

#[derive(Clone)]
pub struct ParlerEngine {
    device: candle_core::Device,
    model: Arc<Mutex<Model>>,
    tokenizer: Arc<Tokenizer>,
    config: Config,
}

impl ParlerEngine {
    pub fn new() -> Result<Self> {
        let device = candle_core::Device::new_cuda(1).unwrap_or(candle_core::Device::Cpu);
        let api = match std::env::var("HF_TOKEN") {
            Ok(token) => hf_hub::api::sync::ApiBuilder::new()
                .with_token(Some(token))
                .build()?,
            Err(_) => hf_hub::api::sync::Api::new()?,
        };

        // We use parler-tts-mini-v1
        let repo = api.repo(hf_hub::Repo::model(
            "parler-tts/parler-tts-mini-v1".to_string(),
        ));

        let model_file = repo.get("model.safetensors")?;
        let config_file = repo.get("config.json")?;
        let tokenizer_file = repo.get("tokenizer.json")?;

        let tokenizer = Tokenizer::from_file(tokenizer_file)
            .map_err(|e| anyhow::anyhow!("Failed to load Parler Tokenizer: {}", e))?;

        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)? };

        let config_str = std::fs::read_to_string(config_file)?;
        let config: Config = serde_json::from_str(&config_str)?;

        let model = Model::new(&config, vb)?;

        Ok(Self {
            device,
            model: Arc::new(Mutex::new(model)),
            tokenizer: Arc::new(tokenizer),
            config,
        })
    }

    pub async fn synthesize_speech(
        &self,
        prompt: &str,
        voice_description: Option<&str>,
    ) -> Result<Vec<u8>> {
        let default_desc = "A clear and professional voice with a moderate speed and pitch, recorded in very high quality.";
        let description = voice_description.unwrap_or(default_desc);

        let desc_tokens = self
            .tokenizer
            .encode(description, true)
            .map_err(|e| anyhow::anyhow!("Failed encoding description: {}", e))?
            .get_ids()
            .to_vec();
        let desc_tensor = Tensor::new(desc_tokens, &self.device)?.unsqueeze(0)?;

        let prompt_tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("Failed encoding prompt: {}", e))?
            .get_ids()
            .to_vec();
        let prompt_tensor = Tensor::new(prompt_tokens, &self.device)?.unsqueeze(0)?;

        // Default sampler parameters
        let lp = candle_transformers::generation::LogitsProcessor::new(
            0,    // seed
            None, // temperature
            None, // top_p
        );

        let mut model_lock = self.model.lock().await;

        let max_steps = 1500; // Allow sufficient audio length
        let codes = model_lock.generate(&prompt_tensor, &desc_tensor, lp, max_steps)?;

        let codes = codes.unsqueeze(0)?;
        let pcm_tensor = model_lock
            .audio_encoder
            .decode_codes(&codes.to_device(&self.device)?)?;

        drop(model_lock);

        // pcm_tensor shape is typically (batch, channels, length). We want the first items.
        let pcm = pcm_tensor.i((0, 0))?;

        // Simple manual amplitude normalization (avoiding example crate dependency)
        // Find max absolute value
        let pcm_vec = pcm.to_vec1::<f32>()?;
        let mut max_abs = 0.0_f32;
        for &sample in &pcm_vec {
            let abs = sample.abs();
            if abs > max_abs {
                max_abs = abs;
            }
        }

        let mut normalized_pcm = Vec::with_capacity(pcm_vec.len());
        let scale = if max_abs > 0.0 { 0.95 / max_abs } else { 1.0 };
        for sample in pcm_vec {
            normalized_pcm.push(sample * scale);
        }

        // Encode to WAV
        let spec = WavSpec {
            channels: 1,
            sample_rate: self.config.audio_encoder.sampling_rate,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };

        let mut cursor = Cursor::new(Vec::new());
        {
            let mut writer =
                WavWriter::new(&mut cursor, spec).context("Failed to initialize WAV writer")?;
            for sample in normalized_pcm {
                writer
                    .write_sample(sample)
                    .context("Failed to write audiom sample")?;
            }
            writer.finalize().context("Failed to finalize WAV")?;
        }

        Ok(cursor.into_inner())
    }
}
