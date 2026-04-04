use anyhow::Result;
use candle_core::{Device, IndexOp, Tensor};
use candle_transformers::models::whisper::{self as m, Config, audio};
use hound::WavReader;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;

pub struct WhisperEngine {
    device: Device,
    model: Arc<Mutex<m::quantized_model::Whisper>>,
    tokenizer: Arc<Tokenizer>,
    config: Config,
    mel_filters: Vec<f32>,
}

impl WhisperEngine {
    pub fn new() -> Result<Self> {
        let device = Device::new_cuda(1).unwrap_or(Device::Cpu);
        let api = match std::env::var("HF_TOKEN") {
            Ok(token) => hf_hub::api::sync::ApiBuilder::new()
                .with_token(Some(token))
                .build()?,
            Err(_) => hf_hub::api::sync::Api::new()?,
        };
        // Use an underscore to suppress unused variable warning
        let _dataset = api.dataset("Narsil/candle-examples".to_string());
        let repo = api.repo(hf_hub::Repo::model("lmz/candle-whisper".to_string()));

        let config_file = repo.get("config-tiny-en.json")?;
        let tokenizer_file = repo.get("tokenizer-tiny-en.json")?;
        let weights_file = repo.get("model-tiny-en-q80.gguf")?;

        let config_str = std::fs::read_to_string(config_file)?;
        let config: Config = serde_json::from_str(&config_str)?;
        let tokenizer = Tokenizer::from_file(tokenizer_file)
            .map_err(|e| anyhow::anyhow!("Failed to load Whisper Tokenizer: {}", e))?;

        let mel_bytes = match config.num_mel_bins {
            80 => include_bytes!("melfilters.bytes").as_slice(),
            128 => include_bytes!("melfilters128.bytes").as_slice(),
            nmel => anyhow::bail!("unexpected num_mel_bins {nmel}"),
        };
        let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
        <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(
            mel_bytes,
            &mut mel_filters,
        );

        let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
            &weights_file,
            &device,
        )?;
        let model = m::quantized_model::Whisper::load(&vb, config.clone())?;

        Ok(Self {
            device,
            model: Arc::new(Mutex::new(model)),
            tokenizer: Arc::new(tokenizer),
            config,
            mel_filters,
        })
    }

    pub async fn transcribe_audio(&self, wav_bytes: &[u8]) -> Result<String> {
        let mut reader = WavReader::new(std::io::Cursor::new(wav_bytes))?;
        let spec = reader.spec();
        let sample_rate = spec.sample_rate;

        if sample_rate != m::SAMPLE_RATE as u32 {
            anyhow::bail!(
                "Input file must have a {} sampling rate, but got {}",
                m::SAMPLE_RATE,
                sample_rate
            );
        }

        // Simplify decoder logic internally handling stereo/mono internally vs directly to f32.
        let mut pcm_data: Vec<f32> = if spec.sample_format == hound::SampleFormat::Int {
            reader
                .samples::<i16>()
                .map(|s| s.unwrap_or(0) as f32 / 32768.0)
                .collect()
        } else {
            reader.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect()
        };

        // Convert stereo to mono if needed
        if spec.channels == 2 {
            let mut mono = Vec::with_capacity(pcm_data.len() / 2);
            for chunk in pcm_data.chunks(2) {
                mono.push((chunk[0] + chunk[1]) / 2.0);
            }
            pcm_data = mono;
        }

        let mel = audio::pcm_to_mel(&self.config, &pcm_data, &self.mel_filters);
        let mel_len = mel.len();
        let mel = Tensor::from_vec(
            mel,
            (
                1,
                self.config.num_mel_bins,
                mel_len / self.config.num_mel_bins,
            ),
            &self.device,
        )?;

        let mut model_lock = self.model.lock().await;

        // Standard Decoder Execution
        let mut text_output = String::new();
        let (_, _, content_frames) = mel.dims3()?;
        let mut seek = 0;

        let sot_token = self.tokenizer.token_to_id(m::SOT_TOKEN).unwrap();
        let transcribe_token = self.tokenizer.token_to_id(m::TRANSCRIBE_TOKEN).unwrap();
        let eot_token = self.tokenizer.token_to_id(m::EOT_TOKEN).unwrap();
        let no_timestamps_token = self.tokenizer.token_to_id(m::NO_TIMESTAMPS_TOKEN).unwrap();

        while seek < content_frames {
            let segment_size = usize::min(content_frames - seek, m::N_FRAMES);
            let mel_segment = mel.narrow(2, seek, segment_size)?;

            let audio_features = model_lock.encoder.forward(&mel_segment, true)?;
            let sample_len = self.config.max_target_positions / 2;
            let mut tokens = vec![sot_token];

            tokens.push(transcribe_token);
            tokens.push(no_timestamps_token);

            for i in 0..sample_len {
                let tokens_t = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
                let ys = model_lock
                    .decoder
                    .forward(&tokens_t, &audio_features, i == 0)?;

                let (_, seq_len, _) = ys.dims3()?;
                let logits = model_lock
                    .decoder
                    .final_linear(&ys.i((..1, seq_len - 1..))?)?
                    .i(0)?
                    .i(0)?;

                // Pure greedy decode approach
                let logits_v: Vec<f32> = logits.to_vec1()?;
                let next_token = logits_v
                    .iter()
                    .enumerate()
                    .max_by(|(_, u), (_, v)| u.total_cmp(v))
                    .map(|(id, _)| id as u32)
                    .unwrap();

                tokens.push(next_token);
                if next_token == eot_token {
                    break;
                }
            }

            let segment_text = self
                .tokenizer
                .decode(&tokens, true)
                .map_err(anyhow::Error::msg)?;
            text_output.push_str(&segment_text);

            seek += segment_size;
        }

        Ok(text_output)
    }
}
