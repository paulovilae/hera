use crate::ai::{InferenceError, SpeechToTextEngine};
use anyhow::Result;
use candle_core::{Device, IndexOp, Tensor};
use candle_transformers::models::whisper::{self as m, Config, audio};
use hound::WavReader;
use std::path::PathBuf;
use std::process::Command;
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
        let model_config = WhisperModelConfig::from_env();
        let repo = api.repo(hf_hub::Repo::model(model_config.repo.clone()));

        let config_file = repo.get(&model_config.config_file)?;
        let tokenizer_file = repo.get(&model_config.tokenizer_file)?;
        let weights_file = repo.get(&model_config.weights_file)?;

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

    pub async fn transcribe_audio(&self, wav_bytes: &[u8], lang: Option<&str>) -> Result<String> {
        let normalized_wav = normalize_audio_to_wav(wav_bytes)?;
        let mut reader = WavReader::new(std::io::Cursor::new(normalized_wav))?;
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

        let language_token = whisper_language_token_for(&self.tokenizer, lang);

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

            if let Some(language_token) = language_token {
                tokens.push(language_token);
            }
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

#[async_trait::async_trait]
impl SpeechToTextEngine for WhisperEngine {
    async fn transcribe_audio(&self, wav_bytes: &[u8], lang: Option<&str>) -> Result<String, InferenceError> {
        WhisperEngine::transcribe_audio(self, wav_bytes, lang)
            .await
            .map_err(|err| InferenceError::ExecutionFailed(err.to_string()))
    }
}

struct WhisperModelConfig {
    repo: String,
    config_file: String,
    tokenizer_file: String,
    weights_file: String,
}

impl WhisperModelConfig {
    fn from_env() -> Self {
        if let Some(explicit) = Self::from_explicit_env() {
            return explicit;
        }

        match std::env::var("HERA_WHISPER_MODEL")
            .unwrap_or_else(|_| "tiny".to_string())
            .to_ascii_lowercase()
            .as_str()
        {
            "tiny-en" | "tiny_en" | "en" | "english" => Self {
                repo: "lmz/candle-whisper".to_string(),
                config_file: "config-tiny-en.json".to_string(),
                tokenizer_file: "tokenizer-tiny-en.json".to_string(),
                weights_file: "model-tiny-en-q80.gguf".to_string(),
            },
            "small" => Self {
                repo: "oxide-lab/whisper-small-GGUF".to_string(),
                config_file: "config.json".to_string(),
                tokenizer_file: "tokenizer.json".to_string(),
                weights_file: "whisper-small-q4_k.gguf".to_string(),
            },
            _ => Self {
                repo: "lmz/candle-whisper".to_string(),
                config_file: "config-tiny.json".to_string(),
                tokenizer_file: "tokenizer-tiny.json".to_string(),
                weights_file: "model-tiny-q80.gguf".to_string(),
            },
        }
    }

    fn from_explicit_env() -> Option<Self> {
        let repo = std::env::var("HERA_WHISPER_REPO").ok()?;
        let config_file = std::env::var("HERA_WHISPER_CONFIG_FILE").ok()?;
        let tokenizer_file = std::env::var("HERA_WHISPER_TOKENIZER_FILE").ok()?;
        let weights_file = std::env::var("HERA_WHISPER_WEIGHTS_FILE").ok()?;

        Some(Self {
            repo,
            config_file,
            tokenizer_file,
            weights_file,
        })
    }
}

/// Resolve the Whisper language token for a given transcription request.
///
/// Priority: per-request `lang` hint → `HERA_WHISPER_LANGUAGE` env var → auto-detect (None).
/// `lang` values of `""`, `"auto"`, or `"automatic"` are treated as "use the env default".
fn whisper_language_token_for(tokenizer: &Tokenizer, lang: Option<&str>) -> Option<u32> {
    // Per-request hint takes priority when it is a non-empty, non-auto value.
    if let Some(req_lang) = lang {
        let req_lang = req_lang.trim().to_ascii_lowercase();
        if !req_lang.is_empty() && req_lang != "auto" && req_lang != "automatic" {
            tracing::debug!(language = %req_lang, "STT: using per-request language override");
            return tokenizer.token_to_id(&format!("<|{}|>", req_lang));
        }
    }

    // Fall back to environment default.
    let language = std::env::var("HERA_WHISPER_LANGUAGE")
        .unwrap_or_else(|_| "auto".to_string())
        .trim()
        .to_ascii_lowercase();

    if language.is_empty() || language == "auto" {
        return None;
    }

    tokenizer.token_to_id(&format!("<|{}|>", language))
}

fn normalize_audio_to_wav(audio_bytes: &[u8]) -> Result<Vec<u8>> {
    if let Ok(reader) = WavReader::new(std::io::Cursor::new(audio_bytes)) {
        let spec = reader.spec();
        if spec.sample_rate == m::SAMPLE_RATE as u32 && spec.channels == 1 {
            return Ok(audio_bytes.to_vec());
        }
    }

    let temp_dir = std::env::temp_dir();
    let nonce = format!(
        "{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|value| value.as_millis())
            .unwrap_or(0)
    );
    let input_path = temp_file_path(&temp_dir, &nonce, "input_audio.bin");
    let output_path = temp_file_path(&temp_dir, &nonce, "normalized_audio.wav");

    std::fs::write(&input_path, audio_bytes)?;

    let ffmpeg_output = Command::new("ffmpeg")
        .args([
            "-y",
            "-i",
            input_path.to_string_lossy().as_ref(),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            output_path.to_string_lossy().as_ref(),
        ])
        .output()?;

    let _ = std::fs::remove_file(&input_path);

    if !ffmpeg_output.status.success() {
        let stderr = String::from_utf8_lossy(&ffmpeg_output.stderr);
        let _ = std::fs::remove_file(&output_path);
        anyhow::bail!("Failed to normalize audio with ffmpeg: {}", stderr.trim());
    }

    let normalized = std::fs::read(&output_path)?;
    let _ = std::fs::remove_file(&output_path);
    Ok(normalized)
}

fn temp_file_path(temp_dir: &PathBuf, nonce: &str, filename: &str) -> PathBuf {
    temp_dir.join(format!("hera-whisper-{}-{}", nonce, filename))
}
