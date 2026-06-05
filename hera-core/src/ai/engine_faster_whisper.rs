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

/// Resolve the faster-whisper helper script robustly, independent of the process
/// cwd. pm2 launches hera-core from `Hera/` (not `Hera/hera-core/`), so the old
/// default `../hera-core/scripts/...` resolved to a nonexistent `OS/hera-core/scripts`
/// and the STT backend silently failed to mount (the mic "didn't hear" anything).
fn resolve_script_path() -> anyhow::Result<PathBuf> {
    // 1. Explicit override always wins (use this on nodes where the repo lives at a
    //    different path than the build node, e.g. a scp'd binary on anchor).
    if let Ok(p) = std::env::var("HERA_FASTER_WHISPER_SCRIPT") {
        let pb = PathBuf::from(p);
        if pb.exists() {
            return Ok(pb);
        }
        anyhow::bail!(
            "HERA_FASTER_WHISPER_SCRIPT set but not found at {}",
            pb.display()
        );
    }
    // 2. Compile-time crate dir — absolute, correct regardless of cwd on the build node.
    let manifest =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts/faster_whisper_stt.py");
    if manifest.exists() {
        return Ok(manifest);
    }
    // 3. Alongside the running executable (covers scp'd binaries shipping scripts next to them).
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        let cand = dir.join("scripts/faster_whisper_stt.py");
        if cand.exists() {
            return Ok(cand);
        }
    }
    // 4. Legacy cwd-relative fallback (works only when cwd is Hera/hera-core).
    let legacy = PathBuf::from("../hera-core/scripts/faster_whisper_stt.py");
    if legacy.exists() {
        return Ok(legacy);
    }
    anyhow::bail!(
        "faster-whisper helper script not found (tried {} and cwd-relative); \
         set HERA_FASTER_WHISPER_SCRIPT to an absolute path",
        manifest.display()
    );
}

impl FasterWhisperEngine {
    pub fn new() -> anyhow::Result<Self> {
        let script_path = resolve_script_path()?;

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
