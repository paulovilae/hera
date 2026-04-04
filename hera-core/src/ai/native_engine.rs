use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use crate::ai::quantized_qwen3_moe_local::GGUFQWenMoE;
use candle_core::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::qwen2::ModelForCausalLM as Qwen2;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

pub struct LlmConfig {
    pub model_id: String,
    pub revision: String,
    pub temperature: f64,
}

impl Default for LlmConfig {
    fn default() -> Self {
        let default_local_moe = "/data/models/llm-stack/Qwen3-30B-A3B-Q4_K_M.gguf";
        let model_id = std::env::var("HERA_CANDLE_MODEL_ID").unwrap_or_else(|_| {
            if Path::new(default_local_moe).exists() {
                default_local_moe.to_string()
            } else {
                "Qwen/Qwen3-8B".to_string()
            }
        });
        let revision = std::env::var("HERA_CANDLE_REVISION").unwrap_or_else(|_| "main".to_string());
        let speed_mode = std::env::var("HERA_SPEED_MODE")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(true);
        let temperature = std::env::var("HERA_CANDLE_TEMPERATURE")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(if speed_mode { 0.0 } else { 0.7 });
        Self {
            model_id,
            revision,
            temperature,
        }
    }
}

pub(crate) enum EngineBackend {
    Qwen2(Mutex<Qwen2>),
    Qwen2Gguf(Mutex<candle_transformers::models::quantized_qwen2::ModelWeights>),
    Qwen3Gguf(Mutex<candle_transformers::models::quantized_qwen3::ModelWeights>),
    Qwen3MoeGguf(Mutex<GGUFQWenMoE>),
}

pub(crate) trait ModelForward {
    fn forward_step(&mut self, input: &Tensor, index_pos: usize) -> candle_core::Result<Tensor>;
    fn reset_cache(&mut self);
}

impl ModelForward for Qwen2 {
    fn forward_step(&mut self, input: &Tensor, index_pos: usize) -> candle_core::Result<Tensor> {
        self.forward(input, index_pos)
    }
    fn reset_cache(&mut self) {
        // Qwen2 safetensors: no kv_cache clearing exposed in candle-transformers 0.9.2
    }
}
impl ModelForward for candle_transformers::models::quantized_qwen2::ModelWeights {
    fn forward_step(&mut self, input: &Tensor, index_pos: usize) -> candle_core::Result<Tensor> {
        self.forward(input, index_pos)
    }
    fn reset_cache(&mut self) {
        // quantized_qwen2: no clear_kv_cache exposed in candle-transformers 0.9.2
    }
}
impl ModelForward for candle_transformers::models::quantized_qwen3::ModelWeights {
    fn forward_step(&mut self, input: &Tensor, index_pos: usize) -> candle_core::Result<Tensor> {
        self.forward(input, index_pos)
    }
    fn reset_cache(&mut self) {
        self.clear_kv_cache();
    }
}
impl ModelForward for GGUFQWenMoE {
    fn forward_step(&mut self, input: &Tensor, index_pos: usize) -> candle_core::Result<Tensor> {
        self.forward(input, index_pos)
    }
    fn reset_cache(&mut self) {
        self.clear_kv_cache();
    }
}

pub struct NativeLlmEngine {
    backend: EngineBackend,
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub temperature: f64,
    pub model_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationStats {
    pub model: String,
    pub max_context_tokens: usize,
    pub max_new_tokens: usize,
    pub effective_context_tokens: usize,
    pub truncated_prompt_tokens: usize,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    pub reading_ms: u128,
    pub generation_ms: u128,
    pub total_ms: u128,
    pub tokens_per_second: f64,
    pub timed_out: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResult {
    pub text: String,
    pub stats: GenerationStats,
}

pub static LLM_ENGINE: OnceLock<Arc<NativeLlmEngine>> = OnceLock::new();

pub fn get_or_init_engine() -> Result<Arc<NativeLlmEngine>, String> {
    if let Some(engine) = LLM_ENGINE.get() {
        return Ok(engine.clone());
    }

    match init_llm_engine(LlmConfig::default()) {
        Ok(_) => Ok(LLM_ENGINE.get().unwrap().clone()),
        Err(e) => Err(format!("Failed to initialize LLM Engine: {e}")),
    }
}

fn resolve_device() -> Result<Device, Box<dyn std::error::Error + Send + Sync>> {
    let device_pref = std::env::var("HERA_DEVICE").unwrap_or_else(|_| "auto".to_string());
    let cuda_device_env = std::env::var("HERA_CUDA_DEVICE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());

    let device = match device_pref.as_str() {
        "cpu" => Device::Cpu,
        "cuda" => {
            if let Some(idx) = cuda_device_env {
                Device::new_cuda(idx)
                    .map_err(|e| format!("CUDA requested on device {idx}, but unavailable: {e}"))?
            } else {
                Device::new_cuda(0)
                    .map_err(|e| format!("CUDA requested on device 0, but unavailable: {e}"))?
            }
        }
        _ => {
            if let Some(idx) = cuda_device_env {
                match Device::new_cuda(idx) {
                    Ok(cuda) => cuda,
                    Err(err) => {
                        println!(
                            "[LLM_ENGINE]: CUDA device {idx} unavailable ({err}), falling back to CPU."
                        );
                        Device::Cpu
                    }
                }
            } else {
                let mut selected = None;

                // Smart VRAM Profiler: Probe the NVIDIA driver for free memory
                if let Ok(output) = std::process::Command::new("nvidia-smi")
                    .args([
                        "--query-gpu=index,memory.free",
                        "--format=csv,noheader,nounits",
                    ])
                    .output()
                {
                    if output.status.success() {
                        if let Ok(stdout) = String::from_utf8(output.stdout) {
                            let mut best_idx = 0;
                            let mut max_free = 0;

                            for line in stdout.lines() {
                                let parts: Vec<&str> = line.split(',').collect();
                                if parts.len() == 2 {
                                    if let (Ok(idx), Ok(free)) = (
                                        parts[0].trim().parse::<usize>(),
                                        parts[1].trim().parse::<u64>(),
                                    ) {
                                        if free > max_free {
                                            max_free = free;
                                            best_idx = idx;
                                        }
                                    }
                                }
                            }

                            if max_free > 0 {
                                println!(
                                    "[LLM_ENGINE]: 🧠 Smart VRAM Profiler selected GPU {} ({} MiB free VRAM)",
                                    best_idx, max_free
                                );
                                if let Ok(cuda) = Device::new_cuda(best_idx) {
                                    selected = Some(cuda);
                                }
                            }
                        }
                    }
                }

                // Fallback to naive iteration if nvidia-smi fails
                if selected.is_none() {
                    for idx in 0..8 {
                        if let Ok(cuda) = Device::new_cuda(idx) {
                            selected = Some(cuda);
                            break;
                        }
                    }
                }
                match selected {
                    Some(cuda) => cuda,
                    None => {
                        println!(
                            "[LLM_ENGINE]: CUDA not available on devices 0..7, falling back to CPU."
                        );
                        Device::Cpu
                    }
                }
            }
        }
    };
    Ok(device)
}

pub fn init_llm_engine(config: LlmConfig) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let legacy_block = std::env::var("HERA_ALLOW_LEGACY_CANDLE_15")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if config.model_id.to_lowercase().contains("qwen1.5") && !legacy_block {
        return Err("legacy Candle model blocked: set HERA_CANDLE_MODEL_ID to a modern model (e.g. Qwen/Qwen3-8B or GGUF Qwen3.5 MoE) or explicitly set HERA_ALLOW_LEGACY_CANDLE_15=true".into());
    }
    if config.model_id.to_lowercase().contains("qwen3.5")
        && (config.model_id.ends_with(".gguf") || Path::new(&config.model_id).exists())
    {
        return Err("this Candle build (0.9.2) does not support qwen35moe GGUF yet. Use a qwen3_moe GGUF (e.g. /data/models/llm-stack/Qwen3-30B-A3B-Q4_K_M.gguf) for Candle-only mode.".into());
    }

    let device = resolve_device()?;
    println!("[LLM_ENGINE]: Using device: {:?}", device);

    let is_gguf_path = config.model_id.ends_with(".gguf") || Path::new(&config.model_id).exists();
    println!(
        "[LLM_ENGINE]: Resolved model_id: '{}', is_gguf_path: {}",
        config.model_id, is_gguf_path
    );

    let (backend, tokenizer, loaded_model_id) = if is_gguf_path {
        crate::ai::engine_gguf::load_gguf_backend(&config.model_id, &device)?
    } else {
        crate::ai::engine_hub::load_hub_backend(&config.model_id, &config.revision, &device)?
    };

    let engine = Arc::new(NativeLlmEngine {
        backend,
        tokenizer,
        device,
        temperature: config.temperature,
        model_id: loaded_model_id.clone(),
    });

    let _ = LLM_ENGINE.set(engine);
    println!("[LLM_ENGINE]: {loaded_model_id} Loaded.");
    Ok(())
}

impl NativeLlmEngine {
    pub fn generate_response_with_stats(&self, full_prompt: &str) -> GenerationResult {
        let total_start = Instant::now();

        let tokenization_start = Instant::now();
        let tokens_res = self.tokenizer.encode(full_prompt, true);
        if tokens_res.is_err() {
            return GenerationResult {
                text: "Error: Tokenization failed".to_string(),
                stats: GenerationStats {
                    model: self.model_id.clone(),
                    max_context_tokens: 0,
                    max_new_tokens: 0,
                    effective_context_tokens: 0,
                    truncated_prompt_tokens: 0,
                    prompt_tokens: 0,
                    completion_tokens: 0,
                    total_tokens: 0,
                    reading_ms: tokenization_start.elapsed().as_millis(),
                    generation_ms: 0,
                    total_ms: total_start.elapsed().as_millis(),
                    tokens_per_second: 0.0,
                    timed_out: false,
                },
            };
        }

        let mut tokens = tokens_res.unwrap().get_ids().to_vec();
        let prompt_tokens_original = tokens.len();
        let tokenization_ms = tokenization_start.elapsed().as_millis();

        let eos_token_id = self
            .tokenizer
            .token_to_id("<|im_end|>")
            .or_else(|| self.tokenizer.token_to_id("<|endoftext|>"));

        let max_new_tokens = std::env::var("HERA_MAX_NEW_TOKENS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(96);
        let max_context_tokens = std::env::var("HERA_MAX_CONTEXT_TOKENS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(65_536);
        let reserved_for_generation = max_new_tokens.saturating_add(1);
        let allowed_prompt_tokens = max_context_tokens
            .saturating_sub(reserved_for_generation)
            .max(1);
        let truncated_prompt_tokens = prompt_tokens_original.saturating_sub(allowed_prompt_tokens);
        if tokens.len() > allowed_prompt_tokens {
            let keep_from = tokens.len() - allowed_prompt_tokens;
            tokens = tokens.split_off(keep_from);
        }
        let prompt_tokens = tokens.len();

        let max_gen_ms = std::env::var("HERA_MAX_GEN_MS")
            .ok()
            .and_then(|v| v.parse::<u128>().ok())
            .unwrap_or(60_000);
        let min_completion_tokens = std::env::var("HERA_MIN_COMPLETION_TOKENS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(12);

        let mut generated_token_ids: Vec<u32> = Vec::new();
        let mut generated_text = String::new();
        let mut index_pos = 0usize;
        let mut timed_out = false;
        let mut completion_tokens = 0usize;
        let mut reading_ms = tokenization_ms;
        let mut generation_ms = 0u128;
        let mut failure_reason: Option<String> = None;
        let sampling_seed = std::env::var("HERA_SAMPLING_SEED")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(299_792_458);
        let top_k = std::env::var("HERA_TOP_K")
            .ok()
            .and_then(|v| v.parse::<usize>().ok());
        let top_p = std::env::var("HERA_TOP_P")
            .ok()
            .and_then(|v| v.parse::<f64>().ok());
        let repeat_penalty = std::env::var("HERA_REPEAT_PENALTY")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(1.0);
        let repeat_last_n = std::env::var("HERA_REPEAT_LAST_N")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(64);
        let sampling = if self.temperature <= 0.0 {
            Sampling::ArgMax
        } else {
            match (top_k, top_p) {
                (None, None) => Sampling::All {
                    temperature: self.temperature,
                },
                (Some(k), None) => Sampling::TopK {
                    k,
                    temperature: self.temperature,
                },
                (None, Some(p)) => Sampling::TopP {
                    p,
                    temperature: self.temperature,
                },
                (Some(k), Some(p)) => Sampling::TopKThenTopP {
                    k,
                    p,
                    temperature: self.temperature,
                },
            }
        };
        let mut logits_processor = LogitsProcessor::from_sampling(sampling_seed, sampling);

        {
            let mut q2_lock;
            let mut q2g_lock;
            let mut q3g_lock;
            let mut q3m_lock;

            let model: &mut dyn ModelForward = match &self.backend {
                EngineBackend::Qwen2(mutex) => {
                    q2_lock = match mutex.lock() {
                        Ok(l) => l,
                        Err(_) => {
                            return GenerationResult {
                                text: "Error: Failed to lock model".to_string(),
                                stats: GenerationStats {
                                    model: self.model_id.clone(),
                                    max_context_tokens,
                                    max_new_tokens,
                                    effective_context_tokens: prompt_tokens,
                                    truncated_prompt_tokens,
                                    prompt_tokens,
                                    completion_tokens: 0,
                                    total_tokens: prompt_tokens,
                                    reading_ms,
                                    generation_ms: 0,
                                    total_ms: total_start.elapsed().as_millis(),
                                    tokens_per_second: 0.0,
                                    timed_out: false,
                                },
                            };
                        }
                    };
                    &mut *q2_lock
                }
                EngineBackend::Qwen2Gguf(mutex) => {
                    q2g_lock = match mutex.lock() {
                        Ok(l) => l,
                        Err(_) => {
                            return GenerationResult {
                                text: "Error: Failed to lock model".to_string(),
                                stats: GenerationStats {
                                    model: self.model_id.clone(),
                                    max_context_tokens,
                                    max_new_tokens,
                                    effective_context_tokens: prompt_tokens,
                                    truncated_prompt_tokens,
                                    prompt_tokens,
                                    completion_tokens: 0,
                                    total_tokens: prompt_tokens,
                                    reading_ms,
                                    generation_ms: 0,
                                    total_ms: total_start.elapsed().as_millis(),
                                    tokens_per_second: 0.0,
                                    timed_out: false,
                                },
                            };
                        }
                    };
                    &mut *q2g_lock
                }
                EngineBackend::Qwen3Gguf(mutex) => {
                    q3g_lock = match mutex.lock() {
                        Ok(l) => l,
                        Err(_) => {
                            return GenerationResult {
                                text: "Error: Failed to lock model".to_string(),
                                stats: GenerationStats {
                                    model: self.model_id.clone(),
                                    max_context_tokens,
                                    max_new_tokens,
                                    effective_context_tokens: prompt_tokens,
                                    truncated_prompt_tokens,
                                    prompt_tokens,
                                    completion_tokens: 0,
                                    total_tokens: prompt_tokens,
                                    reading_ms,
                                    generation_ms: 0,
                                    total_ms: total_start.elapsed().as_millis(),
                                    tokens_per_second: 0.0,
                                    timed_out: false,
                                },
                            };
                        }
                    };
                    &mut *q3g_lock
                }
                EngineBackend::Qwen3MoeGguf(mutex) => {
                    q3m_lock = match mutex.lock() {
                        Ok(l) => l,
                        Err(_) => {
                            return GenerationResult {
                                text: "Error: Failed to lock model".to_string(),
                                stats: GenerationStats {
                                    model: self.model_id.clone(),
                                    max_context_tokens,
                                    max_new_tokens,
                                    effective_context_tokens: prompt_tokens,
                                    truncated_prompt_tokens,
                                    prompt_tokens,
                                    completion_tokens: 0,
                                    total_tokens: prompt_tokens,
                                    reading_ms,
                                    generation_ms: 0,
                                    total_ms: total_start.elapsed().as_millis(),
                                    tokens_per_second: 0.0,
                                    timed_out: false,
                                },
                            };
                        }
                    };
                    &mut *q3m_lock
                }
            };

            // Clear KV cache before each inference to prevent stale attention state
            model.reset_cache();

            for index in 0..max_new_tokens {
                let step_start = Instant::now();
                if generation_ms > max_gen_ms {
                    timed_out = true;
                    break;
                }

                let (context_size, start_pos) = if index == 0 {
                    (tokens.len(), 0)
                } else {
                    (1, tokens.len() - 1)
                };
                let context = &tokens[start_pos..];
                let input_tensor =
                    match Tensor::new(context, &self.device).and_then(|t| t.unsqueeze(0)) {
                        Ok(t) => t,
                        Err(err) => {
                            failure_reason = Some(format!("tensor_build_failed: {err}"));
                            break;
                        }
                    };
                let logits = match model.forward_step(&input_tensor, index_pos) {
                    Ok(l) => l,
                    Err(err) => {
                        failure_reason = Some(format!("model_forward_failed: {err}"));
                        break;
                    }
                };
                let mut logits = logits;
                if logits.rank() == 3 {
                    logits = match logits.squeeze(0) {
                        Ok(l) => l,
                        Err(err) => {
                            failure_reason = Some(format!("logits_squeeze_r3: {err}"));
                            break;
                        }
                    };
                }
                if logits.rank() == 2 {
                    let seq_len = logits.dim(0).unwrap_or(1);
                    if seq_len > 1 {
                        logits = match logits.get(seq_len - 1) {
                            Ok(l) => l,
                            Err(err) => {
                                failure_reason = Some(format!("logits_get_r2: {err}"));
                                break;
                            }
                        };
                    } else {
                        logits = match logits.squeeze(0) {
                            Ok(l) => l,
                            Err(err) => {
                                failure_reason = Some(format!("logits_squeeze_r2: {err}"));
                                break;
                            }
                        };
                    }
                }
                let logits = if repeat_penalty == 1.0 {
                    logits
                } else {
                    let start_at = generated_token_ids.len().saturating_sub(repeat_last_n);
                    match candle_transformers::utils::apply_repeat_penalty(
                        &logits,
                        repeat_penalty,
                        &generated_token_ids[start_at..],
                    ) {
                        Ok(v) => v,
                        Err(err) => {
                            failure_reason = Some(format!("repeat_penalty_failed: {err}"));
                            break;
                        }
                    }
                };
                let next_token = match logits_processor.sample(&logits) {
                    Ok(t) => t,
                    Err(err) => {
                        failure_reason = Some(format!("token_sample_failed: {err}"));
                        break;
                    }
                };

                tokens.push(next_token);

                if Some(next_token) == eos_token_id && completion_tokens >= min_completion_tokens {
                    break;
                }
                if Some(next_token) == eos_token_id {
                    continue;
                }

                completion_tokens += 1;
                generated_token_ids.push(next_token);
                index_pos += context_size;

                let step_ms = step_start.elapsed().as_millis();
                if index == 0 {
                    reading_ms += step_ms;
                } else {
                    generation_ms += step_ms;
                }
            }
        }

        let total_ms = total_start.elapsed().as_millis();
        let measured_gen_ms = if generation_ms == 0 {
            total_ms
        } else {
            generation_ms
        };
        let tokens_per_second = if measured_gen_ms > 0 {
            (completion_tokens as f64) / (measured_gen_ms as f64 / 1000.0)
        } else {
            0.0
        };

        let stats = GenerationStats {
            model: self.model_id.clone(),
            max_context_tokens,
            max_new_tokens,
            effective_context_tokens: prompt_tokens,
            truncated_prompt_tokens,
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            reading_ms,
            generation_ms,
            total_ms,
            tokens_per_second,
            timed_out,
        };

        if let Ok(decoded) = self.tokenizer.decode(&generated_token_ids, true) {
            generated_text = decoded;
        }

        if generated_text.trim().is_empty() {
            if let Some(reason) = &failure_reason {
                eprintln!("[LLM_ENGINE]: Generation failed before first token: {reason}");
                return GenerationResult {
                    text: format!("Error native generation: {reason}"),
                    stats,
                };
            }
        }

        GenerationResult {
            text: generated_text.trim().to_string(),
            stats,
        }
    }

    pub fn generate_response_stream(
        &self,
        prompt: &str,
        tx: tokio::sync::mpsc::Sender<Result<crate::ai::ChatStreamResponse, InferenceError>>,
        chat_id: String,
        created: u64,
    ) {
        let _ = tx.blocking_send(Ok(crate::ai::ChatStreamResponse {
            id: chat_id.clone(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: self.model_id.clone(),
            choices: vec![crate::ai::ChatStreamChoice {
                index: 0,
                delta: crate::ai::ChatStreamDelta {
                    role: Some("assistant".to_string()),
                    content: Some("".to_string()),
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            stats: None,
        }));

        let total_start = Instant::now();
        let tokenization_start = Instant::now();
        let tokens_res = self.tokenizer.encode(prompt, true);
        if tokens_res.is_err() {
            let _ = tx.blocking_send(Err(InferenceError::ExecutionFailed(
                "Tokenization of prompt failed".to_string(),
            )));
            return;
        }

        let mut tokens = tokens_res.unwrap().get_ids().to_vec();
        let eos_token_id = self
            .tokenizer
            .token_to_id("<|im_end|>")
            .or_else(|| self.tokenizer.token_to_id("<|endoftext|>"));
        let max_new_tokens = std::env::var("HERA_MAX_NEW_TOKENS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(2048);

        let max_context_tokens = std::env::var("HERA_MAX_CONTEXT_TOKENS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(65_536);
        let reserved_for_generation = max_new_tokens.saturating_add(1);
        let allowed_prompt_tokens = max_context_tokens
            .saturating_sub(reserved_for_generation)
            .max(1);
        if tokens.len() > allowed_prompt_tokens {
            let keep_from = tokens.len() - allowed_prompt_tokens;
            tokens = tokens.split_off(keep_from);
        }

        let max_gen_ms = std::env::var("HERA_MAX_GEN_MS")
            .ok()
            .and_then(|v| v.parse::<u128>().ok())
            .unwrap_or(60_000);
        let min_completion_tokens = std::env::var("HERA_MIN_COMPLETION_TOKENS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(12);

        let mut generated_token_ids: Vec<u32> = Vec::new();
        let mut index_pos = 0usize;
        let mut completion_tokens = 0usize;
        let mut generation_ms = 0u128;
        let mut prev_text_len = 0;

        let sampling_seed = std::env::var("HERA_SAMPLING_SEED")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(299_792_458);
        let top_k = std::env::var("HERA_TOP_K")
            .ok()
            .and_then(|v| v.parse::<usize>().ok());
        let top_p = std::env::var("HERA_TOP_P")
            .ok()
            .and_then(|v| v.parse::<f64>().ok());
        let repeat_penalty = std::env::var("HERA_REPEAT_PENALTY")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(1.0);
        let repeat_last_n = std::env::var("HERA_REPEAT_LAST_N")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(64);

        let sampling = if self.temperature <= 0.0 {
            Sampling::ArgMax
        } else {
            match (top_k, top_p) {
                (None, None) => Sampling::All {
                    temperature: self.temperature,
                },
                (Some(k), None) => Sampling::TopK {
                    k,
                    temperature: self.temperature,
                },
                (None, Some(p)) => Sampling::TopP {
                    p,
                    temperature: self.temperature,
                },
                (Some(k), Some(p)) => Sampling::TopKThenTopP {
                    k,
                    p,
                    temperature: self.temperature,
                },
            }
        };
        let mut logits_processor = candle_transformers::generation::LogitsProcessor::from_sampling(
            sampling_seed,
            sampling,
        );

        {
            let mut q2_lock;
            let mut q2g_lock;
            let mut q3g_lock;
            let mut q3m_lock;

            let model: &mut dyn ModelForward = match &self.backend {
                EngineBackend::Qwen2(mutex) => {
                    q2_lock = match mutex.lock() {
                        Ok(l) => l,
                        Err(_) => {
                            let _ = tx.blocking_send(Err(InferenceError::ExecutionFailed(
                                "Mutex lock failed".into(),
                            )));
                            return;
                        }
                    };
                    &mut *q2_lock
                }
                EngineBackend::Qwen2Gguf(mutex) => {
                    q2g_lock = match mutex.lock() {
                        Ok(l) => l,
                        Err(_) => {
                            let _ = tx.blocking_send(Err(InferenceError::ExecutionFailed(
                                "Mutex lock failed".into(),
                            )));
                            return;
                        }
                    };
                    &mut *q2g_lock
                }
                EngineBackend::Qwen3Gguf(mutex) => {
                    q3g_lock = match mutex.lock() {
                        Ok(l) => l,
                        Err(_) => {
                            let _ = tx.blocking_send(Err(InferenceError::ExecutionFailed(
                                "Mutex lock failed".into(),
                            )));
                            return;
                        }
                    };
                    &mut *q3g_lock
                }
                EngineBackend::Qwen3MoeGguf(mutex) => {
                    q3m_lock = match mutex.lock() {
                        Ok(l) => l,
                        Err(_) => {
                            let _ = tx.blocking_send(Err(InferenceError::ExecutionFailed(
                                "Mutex lock failed".into(),
                            )));
                            return;
                        }
                    };
                    &mut *q3m_lock
                }
            };

            // Clear KV cache before each streaming inference to prevent stale attention state
            model.reset_cache();

            for index in 0..max_new_tokens {
                let step_start = Instant::now();
                if generation_ms > max_gen_ms {
                    break;
                }

                let (context_size, start_pos) = if index == 0 {
                    (tokens.len(), 0)
                } else {
                    (1, tokens.len() - 1)
                };
                let context = &tokens[start_pos..];

                let input_tensor =
                    match Tensor::new(context, &self.device).and_then(|t| t.unsqueeze(0)) {
                        Ok(t) => t,
                        Err(e) => {
                            let _ = tx.blocking_send(Err(InferenceError::ExecutionFailed(
                                format!("Tensor: {}", e),
                            )));
                            break;
                        }
                    };

                let logits = match model.forward_step(&input_tensor, index_pos) {
                    Ok(l) => l,
                    Err(e) => {
                        let _ = tx.blocking_send(Err(InferenceError::ExecutionFailed(format!(
                            "Forward: {}",
                            e
                        ))));
                        break;
                    }
                };

                let mut logits = logits;
                if logits.rank() == 3 {
                    logits = match logits.squeeze(0) {
                        Ok(l) => l,
                        Err(e) => {
                            let _ = tx.blocking_send(Err(InferenceError::ExecutionFailed(
                                format!("Logits squeeze r3: {}", e),
                            )));
                            break;
                        }
                    };
                }
                if logits.rank() == 2 {
                    let seq_len = logits.dim(0).unwrap_or(1);
                    if seq_len > 1 {
                        logits = match logits.get(seq_len - 1) {
                            Ok(l) => l,
                            Err(e) => {
                                let _ = tx.blocking_send(Err(InferenceError::ExecutionFailed(
                                    format!("Logits get r2: {}", e),
                                )));
                                break;
                            }
                        };
                    } else {
                        logits = match logits.squeeze(0) {
                            Ok(l) => l,
                            Err(e) => {
                                let _ = tx.blocking_send(Err(InferenceError::ExecutionFailed(
                                    format!("Logits squeeze r2: {}", e),
                                )));
                                break;
                            }
                        };
                    }
                }

                let logits = if repeat_penalty == 1.0 {
                    logits
                } else {
                    let start_at = generated_token_ids.len().saturating_sub(repeat_last_n);
                    match candle_transformers::utils::apply_repeat_penalty(
                        &logits,
                        repeat_penalty,
                        &generated_token_ids[start_at..],
                    ) {
                        Ok(v) => v,
                        Err(e) => {
                            let _ = tx.blocking_send(Err(InferenceError::ExecutionFailed(
                                format!("Repeat penalty: {}", e),
                            )));
                            break;
                        }
                    }
                };

                let next_token = match logits_processor.sample(&logits) {
                    Ok(t) => t,
                    Err(e) => {
                        let _ = tx.blocking_send(Err(InferenceError::ExecutionFailed(format!(
                            "Sampling: {}",
                            e
                        ))));
                        break;
                    }
                };

                tokens.push(next_token);

                if Some(next_token) == eos_token_id && completion_tokens >= min_completion_tokens {
                    break;
                }
                if Some(next_token) == eos_token_id {
                    continue;
                }

                completion_tokens += 1;
                generated_token_ids.push(next_token);
                index_pos += context_size;

                // Progressive decoding
                if let Ok(decoded) = self.tokenizer.decode(&generated_token_ids, true) {
                    if decoded.len() > prev_text_len {
                        let new_text = decoded[prev_text_len..].to_string();
                        prev_text_len = decoded.len();

                        let _ = tx.blocking_send(Ok(crate::ai::ChatStreamResponse {
                            id: chat_id.clone(),
                            object: "chat.completion.chunk".to_string(),
                            created,
                            model: self.model_id.clone(),
                            choices: vec![crate::ai::ChatStreamChoice {
                                index: 0,
                                delta: crate::ai::ChatStreamDelta {
                                    role: None,
                                    content: Some(new_text),
                                    tool_calls: None,
                                },
                                finish_reason: None,
                            }],
                            stats: None,
                        }));
                    }
                }

                if index > 0 {
                    generation_ms += step_start.elapsed().as_millis();
                }
            }
        }

        let total_ms = total_start.elapsed().as_millis();
        let _tokenization_ms = tokenization_start.elapsed().as_millis();
        let reading_ms = total_ms.saturating_sub(generation_ms);
        let tokens_per_second = if generation_ms > 0 {
            (completion_tokens as f64) / (generation_ms as f64 / 1000.0)
        } else {
            0.0
        };

        let stats = GenerationStats {
            model: self.model_id.clone(),
            max_context_tokens,
            max_new_tokens,
            effective_context_tokens: tokens.len(),
            truncated_prompt_tokens: 0,
            prompt_tokens: tokens.len(),
            completion_tokens,
            total_tokens: tokens.len() + completion_tokens,
            reading_ms,
            generation_ms,
            total_ms,
            tokens_per_second,
            timed_out: generation_ms > max_gen_ms,
        };

        let _ = tx.blocking_send(Ok(crate::ai::ChatStreamResponse {
            id: chat_id.clone(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: self.model_id.clone(),
            choices: vec![crate::ai::ChatStreamChoice {
                index: 0,
                delta: crate::ai::ChatStreamDelta {
                    role: None,
                    content: None,
                    tool_calls: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            stats: Some(stats),
        }));
    }
}

use crate::ai::{
    ChatChoice, ChatRequest, ChatResponse, ChatResponseMessage, ChatUsage, ContentPart,
    InferenceError, LLMEngine, MessageContent,
};
use std::time::{SystemTime, UNIX_EPOCH};

#[async_trait::async_trait]
impl LLMEngine for NativeLlmEngine {
    async fn generate_content(&self, req: ChatRequest) -> Result<ChatResponse, InferenceError> {
        // Build the prompt string using ChatML format
        let mut full_prompt = String::new();

        if let Some(tools) = &req.tools {
            if !tools.is_empty() {
                full_prompt.push_str("<|im_start|>system\nYou are an AI assistant with access to the following tools. You may trigger a tool by generating a valid JSON payload wrapped in <tool_call> tags. Your available tools:\n");
                if let Ok(tools_json) = serde_json::to_string_pretty(tools) {
                    full_prompt.push_str(&tools_json);
                }
                full_prompt.push_str("\n<|im_end|>\n");
            }
        }

        for msg in &req.messages {
            let role = match msg.role.as_str() {
                "system" => "system",
                "user" => "user",
                "assistant" => "assistant",
                _ => "user",
            };
            let text = match &msg.content {
                MessageContent::Text(t) => t.clone(),
                MessageContent::Null => String::new(),
                MessageContent::Parts(parts) => {
                    let mut combined = String::new();
                    for part in parts {
                        match part {
                            ContentPart::Text { text } => {
                                combined.push_str(text);
                                combined.push('\n');
                            }
                            ContentPart::ImageUrl { .. } => {
                                return Err(InferenceError::ExecutionFailed(
                                    "Local text model does not support image_url inputs. Router will fallback.".to_string()
                                ));
                            }
                        }
                    }
                    combined
                }
            };
            full_prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", role, text));
        }
        full_prompt.push_str("<|im_start|>assistant\n");

        unsafe {
            if let Some(t) = req.temperature {
                std::env::set_var("HERA_CANDLE_TEMPERATURE", t.to_string());
            }
            if let Some(mt) = req.max_tokens {
                std::env::set_var("HERA_MAX_NEW_TOKENS", mt.to_string());
            }
            if let Some(p) = req.top_p {
                std::env::set_var("HERA_TOP_P", p.to_string());
            }
            if let Some(k) = req.top_k {
                std::env::set_var("HERA_TOP_K", k.to_string());
            }
            if let Some(rp) = req.repeat_penalty {
                std::env::set_var("HERA_REPEAT_PENALTY", rp.to_string());
            }
            if let Some(s) = req.seed {
                std::env::set_var("HERA_SAMPLING_SEED", s.to_string());
            }
        }

        // Execute natively through Candle tensors
        let result = self.generate_response_with_stats(&full_prompt);

        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Using a simple generated ID since uuid crate is absent
        let chat_id = format!("chatcmpl-{}", created);

        // If the native engine produced an error, propagate it as Err so the
        // RouterEngine can fall back to the OpenRouter cloud engine.
        if result.text.starts_with("Error") {
            return Err(InferenceError::ExecutionFailed(result.text));
        }

        Ok(ChatResponse {
            id: chat_id,
            object: "chat.completion".to_string(),
            created,
            model: req.model.clone(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatResponseMessage {
                    role: "assistant".to_string(),
                    content: Some(result.text),
                    tool_calls: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(ChatUsage {
                prompt_tokens: result.stats.prompt_tokens as u32,
                completion_tokens: result.stats.completion_tokens as u32,
                total_tokens: result.stats.total_tokens as u32,
            }),
        })
    }

    async fn generate_stream(
        &self,
        req: ChatRequest,
    ) -> Result<
        tokio::sync::mpsc::Receiver<Result<crate::ai::ChatStreamResponse, InferenceError>>,
        InferenceError,
    > {
        let (tx, rx) = tokio::sync::mpsc::channel(100);

        // Build the prompt string using ChatML format
        let mut full_prompt = String::new();

        if let Some(tools) = &req.tools {
            if !tools.is_empty() {
                full_prompt.push_str("<|im_start|>system\nYou are an AI assistant with access to the following tools. You may trigger a tool by generating a valid JSON payload wrapped in <tool_call> tags. Your available tools:\n");
                if let Ok(tools_json) = serde_json::to_string_pretty(tools) {
                    full_prompt.push_str(&tools_json);
                }
                full_prompt.push_str("\n<|im_end|>\n");
            }
        }

        for msg in &req.messages {
            let role = match msg.role.as_str() {
                "system" => "system",
                "user" => "user",
                "assistant" => "assistant",
                _ => "user",
            };
            let text = match &msg.content {
                MessageContent::Text(t) => t.clone(),
                MessageContent::Null => String::new(),
                MessageContent::Parts(parts) => {
                    let mut combined = String::new();
                    for part in parts {
                        match part {
                            ContentPart::Text { text } => {
                                combined.push_str(text);
                                combined.push('\n');
                            }
                            ContentPart::ImageUrl { .. } => {
                                return Err(InferenceError::ExecutionFailed(
                                    "Local text model does not support image_url inputs. Router will fallback.".to_string()
                                ));
                            }
                        }
                    }
                    combined
                }
            };
            full_prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", role, text));
        }
        full_prompt.push_str("<|im_start|>assistant\n");

        unsafe {
            if let Some(t) = req.temperature {
                std::env::set_var("HERA_CANDLE_TEMPERATURE", t.to_string());
            }
            if let Some(mt) = req.max_tokens {
                std::env::set_var("HERA_MAX_NEW_TOKENS", mt.to_string());
            }
            if let Some(p) = req.top_p {
                std::env::set_var("HERA_TOP_P", p.to_string());
            }
            if let Some(k) = req.top_k {
                std::env::set_var("HERA_TOP_K", k.to_string());
            }
            if let Some(rp) = req.repeat_penalty {
                std::env::set_var("HERA_REPEAT_PENALTY", rp.to_string());
            }
            if let Some(s) = req.seed {
                std::env::set_var("HERA_SAMPLING_SEED", s.to_string());
            }
        }

        tokio::task::spawn_blocking(move || {
            let created = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            let chat_id = format!("chatcmpl-{}", created);

            if let Ok(arc_engine) = get_or_init_engine() {
                arc_engine.generate_response_stream(&full_prompt, tx, chat_id, created);
            }
        });

        Ok(rx)
    }
}

pub struct LazyNativeEngine;

#[async_trait::async_trait]
impl LLMEngine for LazyNativeEngine {
    async fn generate_content(&self, req: ChatRequest) -> Result<ChatResponse, InferenceError> {
        let req_clone = req.clone();
        let engine_result = tokio::task::spawn_blocking(move || {
            unsafe {
                std::env::set_var("HERA_CANDLE_MODEL_ID", &req_clone.model);
            }
            get_or_init_engine()
        })
        .await
        .map_err(|e| InferenceError::ExecutionFailed(format!("Task spawn panic: {}", e)))?;

        let engine = engine_result
            .map_err(|e| InferenceError::ExecutionFailed(format!("Engine init failed: {}", e)))?;
        engine.generate_content(req).await
    }

    async fn generate_stream(
        &self,
        req: ChatRequest,
    ) -> Result<
        tokio::sync::mpsc::Receiver<Result<crate::ai::ChatStreamResponse, InferenceError>>,
        InferenceError,
    > {
        let req_clone = req.clone();
        let engine_result = tokio::task::spawn_blocking(move || {
            unsafe {
                std::env::set_var("HERA_CANDLE_MODEL_ID", &req_clone.model);
            }
            get_or_init_engine()
        })
        .await
        .map_err(|e| InferenceError::ExecutionFailed(format!("Task spawn panic: {}", e)))?;

        let engine = engine_result
            .map_err(|e| InferenceError::ExecutionFailed(format!("Engine init failed: {}", e)))?;
        engine.generate_stream(req).await
    }
}
