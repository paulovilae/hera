use crate::ai::{InferenceError, SpeechToTextEngine};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

pub struct FasterWhisperEngine {
    script_path: Arc<PathBuf>,
    python_bin: Arc<String>,
    model: Arc<String>,
    device: Arc<String>,
    compute_type: Arc<String>,
}

impl FasterWhisperEngine {
    pub fn new() -> anyhow::Result<Self> {
        let script_path = PathBuf::from(env_or(
            "HERA_FASTER_WHISPER_SCRIPT",
            "../hera-core/scripts/faster_whisper_stt.py",
        ));
        if !script_path.exists() {
            anyhow::bail!(
                "faster-whisper helper script not found at {}",
                script_path.display()
            );
        }

        Ok(Self {
            script_path: Arc::new(script_path),
            python_bin: Arc::new(env_or("HERA_FASTER_WHISPER_PYTHON", "python3")),
            model: Arc::new(env_or("HERA_FASTER_WHISPER_MODEL", "small")),
            device: Arc::new(env_or("HERA_FASTER_WHISPER_DEVICE", "auto")),
            compute_type: Arc::new(env_or("HERA_FASTER_WHISPER_COMPUTE_TYPE", "int8")),
        })
    }
}

#[async_trait::async_trait]
impl SpeechToTextEngine for FasterWhisperEngine {
    async fn transcribe_audio(&self, wav_bytes: &[u8]) -> Result<String, InferenceError> {
        let normalized_wav = normalize_audio_to_wav(wav_bytes).map_err(|err| {
            InferenceError::ExecutionFailed(format!("Audio normalization failed: {}", err))
        })?;

        let temp_dir = std::env::temp_dir();
        let nonce = format!(
            "{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|value| value.as_millis())
                .unwrap_or(0)
        );
        let input_path = temp_file_path(&temp_dir, &nonce, "faster-whisper-input.wav");
        std::fs::write(&input_path, normalized_wav).map_err(|err| {
            InferenceError::ExecutionFailed(format!("Failed to write temp audio file: {}", err))
        })?;

        let output = Command::new(self.python_bin.as_ref())
            .args([
                self.script_path.to_string_lossy().as_ref(),
                "--audio",
                input_path.to_string_lossy().as_ref(),
                "--model",
                self.model.as_ref(),
                "--device",
                self.device.as_ref(),
                "--compute-type",
                self.compute_type.as_ref(),
            ])
            .output()
            .map_err(|err| {
                InferenceError::ExecutionFailed(format!(
                    "Failed to launch faster-whisper helper: {}",
                    err
                ))
            })?;

        let _ = std::fs::remove_file(&input_path);

        if !output.status.success() {
            return Err(InferenceError::ExecutionFailed(format!(
                "faster-whisper helper failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            )));
        }

        let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
        Ok(text)
    }
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn normalize_audio_to_wav(audio_bytes: &[u8]) -> anyhow::Result<Vec<u8>> {
    if let Ok(reader) = hound::WavReader::new(std::io::Cursor::new(audio_bytes)) {
        let spec = reader.spec();
        if spec.sample_rate == 16000 && spec.channels == 1 {
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

fn temp_file_path(temp_dir: &Path, nonce: &str, filename: &str) -> PathBuf {
    temp_dir.join(format!("hera-{}-{}", nonce, filename))
}
