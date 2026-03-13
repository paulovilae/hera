use crate::ai::{ChatRequest, ChatResponse, ChatChoice, ChatResponseMessage, ChatUsage, InferenceError, LLMEngine, MessageContent, ContentPart, ChatStreamResponse, ChatStreamChoice, ChatStreamDelta};
use async_trait::async_trait;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tracing::{info, error};

pub struct LlamaFfiEngine {
    backend: Arc<LlamaBackend>,
    model: Arc<LlamaModel>,
}

impl LlamaFfiEngine {
    pub fn new(backend: Arc<LlamaBackend>, model_path: &str) -> std::result::Result<Self, String> {
        info!("🧠 [FFI] Loading model from {}...", model_path);
        let model_params = LlamaModelParams::default().with_n_gpu_layers(99);
        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
            .map_err(|e| format!("Failed to load model: {}", e))?;
            
        info!("✅ [FFI] Model loaded natively via C-Bindings into VRAM.");

        Ok(Self {
            backend,
            model: Arc::new(model),
        })
    }
}

#[async_trait]
impl LLMEngine for LlamaFfiEngine {
    async fn generate_content(&self, req: ChatRequest) -> Result<ChatResponse, InferenceError> {
        let mut stream_req = req.clone();
        stream_req.stream = Some(true); // Ensure stream mode

        let mut rx = self.generate_stream(stream_req).await?;
        
        let mut full_content = String::new();
        let mut prompt_tokens = 0;
        let mut new_tokens = 0;
        let mut total_tokens = 0;

        while let Some(res) = rx.recv().await {
            match res {
                Ok(chunk) => {
                    if let Some(choice) = chunk.choices.first() {
                        if let Some(content) = &choice.delta.content {
                            full_content.push_str(content);
                        }
                    }
                    if let Some(stats) = chunk.stats {
                        prompt_tokens = stats.prompt_tokens as u32;
                        new_tokens = stats.completion_tokens as u32;
                        total_tokens = stats.total_tokens as u32;
                    }
                }
                Err(e) => return Err(e),
            }
        }

        Ok(ChatResponse {
            id: format!("chatcmpl-ffi"),
            object: "chat.completion".to_string(),
            created: 0,
            model: "qwen-3.5-35b-ffi".to_string(),
            choices: vec![
                ChatChoice {
                    index: 0,
                    message: ChatResponseMessage {
                        role: "assistant".to_string(),
                        content: Some(full_content),
                        tool_calls: None,
                    },
                    finish_reason: Some("stop".to_string()),
                }
            ],
            usage: Some(ChatUsage {
                prompt_tokens,
                completion_tokens: new_tokens,
                total_tokens,
            }),
        })
    }

    async fn generate_stream(
        &self,
        req: ChatRequest,
    ) -> Result<mpsc::Receiver<Result<ChatStreamResponse, InferenceError>>, InferenceError> {
        let (tx, rx) = mpsc::channel::<Result<ChatStreamResponse, InferenceError>>(32);
        
        let prompt = req.messages.iter()
            .map(|m| {
                let role = &m.role;
                let text = match &m.content {
                    MessageContent::Text(t) => t.clone(),
                    MessageContent::Parts(p) => p.iter().filter_map(|part| match part {
                        ContentPart::Text { text } => Some(text.clone()),
                        _ => None,
                    }).collect::<Vec<_>>().join(" "),
                    MessageContent::Null => String::new(),
                };
                format!("<|im_start|>{role}\n{text}<|im_end|>\n")
            })
            .collect::<Vec<_>>()
            .join("");
            
        let max_tokens = req.max_tokens.unwrap_or(1024) as i32;
        // ⚠️ Removing the <think> tag injection. The Qwen llama.cpp tokenization natively segfaults on specific 
        // byte alignments when trailing `<think>\n` is appended without trailing whitespace.
        let mut final_prompt = format!("{prompt}<|im_start|>assistant\n");

        let model = self.model.clone();
        let backend = self.backend.clone();

        info!("🧠 [FFI] Tokenizing prompt for STREAMING generation... ({} bytes)", final_prompt.len());
        let snippet = final_prompt.chars().take(500).collect::<String>();
        info!("🧠 [FFI] Prompt snippet: {:?}", snippet);

        tokio::task::spawn_blocking(move || {
            let start_time = std::time::Instant::now();
            
            eprintln!(">> Calling model.str_to_token with AddBos::Never ON EXACT BYTE LENGTH: {}", final_prompt.len());
            let mut tokens = match model.str_to_token(&final_prompt, llama_cpp_2::model::AddBos::Never) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!(">> Tokenization Error: {}", e);
                    let _ = tx.blocking_send(Err(InferenceError::ExecutionFailed(format!("Tokenize error: {}", e))));
                    return;
                }
            };

            // 🛡️ Batch Overflow Protection: truncate prompt to leave room for generation
            let max_ctx: usize = 32768;
            let generation_headroom = max_tokens as usize + 64; // leave space for output
            let max_prompt_tokens = max_ctx.saturating_sub(generation_headroom);
            if tokens.len() > max_prompt_tokens {
                eprintln!(">> ⚠️ [FFI] Prompt has {} tokens, truncating to {} (max_ctx={}, headroom={})", 
                    tokens.len(), max_prompt_tokens, max_ctx, generation_headroom);
                tokens.truncate(max_prompt_tokens);
            }

            eprintln!(">> Successfully tokenized {} tokens.", tokens.len());
            let prompt_tokens_len = tokens.len();

            let ctx_params = llama_cpp_2::context::params::LlamaContextParams::default()
                .with_n_ctx(Some(std::num::NonZeroU32::new(32768).unwrap()))
                .with_n_batch(8192);
                
            let mut ctx = match model.new_context(&backend, ctx_params) {
                Ok(c) => c,
                Err(e) => {
                    let _ = tx.blocking_send(Err(InferenceError::ExecutionFailed(format!("Context alloc error: {}", e))));
                    return;
                }
            };

            // 🛡️ Chunked prompt evaluation: process in safe batches of 512 to prevent
            // GGML_ASSERT(n_tokens_all <= cparams.n_batch) crash.
            let mut n_cur = 0;
            let safe_batch_size: usize = 512;
            if tokens.len() > 1 {
                let prompt_tokens_no_last = &tokens[..tokens.len() - 1];
                for chunk in prompt_tokens_no_last.chunks(safe_batch_size) {
                    let mut prompt_batch = llama_cpp_2::llama_batch::LlamaBatch::new(chunk.len(), 1);
                    for &token in chunk {
                        prompt_batch.add(token, n_cur, &[0], false).unwrap();
                        n_cur += 1;
                    }
                    if let Err(e) = ctx.decode(&mut prompt_batch) {
                        let _ = tx.blocking_send(Err(InferenceError::ExecutionFailed(format!("Prompt decode error at pos {}: {}", n_cur, e))));
                        return;
                    }
                }
            }

            // Now evaluate the final token in a batch of size 1 with logits=true
            // so its batch index is 0, matching its output index of 0.
            let mut batch = llama_cpp_2::llama_batch::LlamaBatch::new(1, 1);
            if let Some(&last_token) = tokens.last() {
                batch.add(last_token, n_cur, &[0], true).unwrap();
                n_cur += 1;
                if let Err(e) = ctx.decode(&mut batch) {
                    let _ = tx.blocking_send(Err(InferenceError::ExecutionFailed(format!("Final prompt decode error: {}", e))));
                    return;
                }
            }

            let reading_ms = start_time.elapsed().as_millis();
            let gen_start = std::time::Instant::now();

            let mut generated_tokens = 0;
            let mut decoder = encoding_rs::UTF_8.new_decoder();

            for _ in 0..max_tokens {
                // `batch` is now guaranteed to have exactly 1 token (the last one evaluated),
                // so `batch.n_tokens() - 1` is exactly 0.
                let candidates_iter = ctx.candidates_ith(batch.n_tokens() - 1);
                eprintln!(">> Building LlamaTokenDataArray");
                let mut candidates = llama_cpp_2::token::data_array::LlamaTokenDataArray::from_iter(candidates_iter, false);
                eprintln!(">> Sampling token");
                let token_id = candidates.sample_token_greedy();
                eprintln!(">> Sampled token ID: {}", token_id.0);
                
                if token_id == model.token_eos() || model.is_eog_token(token_id) {
                    eprintln!(">> Hit EOS or EOG");
                    break;
                }

                let mut piece_str = String::new();
                eprintln!(">> Converting token to piece");
                if let Ok(piece) = model.token_to_piece(token_id, &mut decoder, false, None) {
                    piece_str = piece;
                }
                eprintln!(">> Converted piece: '{}'", piece_str.replace('\n', "\\n"));

                batch.clear();
                if let Err(e) = batch.add(token_id, n_cur, &[0], true) {
                    let _ = tx.blocking_send(Err(InferenceError::ExecutionFailed(format!("Batch add generation error: {}", e))));
                    break;
                }
                    
                if let Err(e) = ctx.decode(&mut batch) {
                    let _ = tx.blocking_send(Err(InferenceError::ExecutionFailed(format!("Decode step error: {}", e))));
                    break;
                }

                let resp = ChatStreamResponse {
                    id: "chatcmpl-ffi".to_string(),
                    object: "chat.completion.chunk".to_string(),
                    created: 0,
                    model: "qwen-3.5-35b-ffi".to_string(),
                    choices: vec![ChatStreamChoice {
                        index: 0,
                        delta: ChatStreamDelta {
                            role: Some("assistant".to_string()),
                            content: Some(piece_str),
                            tool_calls: None,
                        },
                        finish_reason: None,
                    }],
                    stats: None,
                };
                
                if tx.blocking_send(Ok(resp)).is_err() {
                    break; // Client disconnected
                }
                    
                n_cur += 1;
                generated_tokens += 1;
            }

            let generation_ms = gen_start.elapsed().as_millis();
            let total_ms = start_time.elapsed().as_millis();
            let tps = if generation_ms > 0 {
                (generated_tokens as f64) / (generation_ms as f64 / 1000.0)
            } else {
                0.0
            };

            let stats = crate::ai::native_engine::GenerationStats {
                model: "qwen-3.5-35b-ffi".to_string(),
                max_context_tokens: 8192,
                max_new_tokens: max_tokens as usize,
                effective_context_tokens: prompt_tokens_len,
                truncated_prompt_tokens: 0,
                prompt_tokens: prompt_tokens_len,
                completion_tokens: generated_tokens,
                total_tokens: prompt_tokens_len + generated_tokens,
                reading_ms,
                generation_ms,
                total_ms,
                tokens_per_second: tps,
                timed_out: false,
            };

            // Send final completion marker
            let final_resp = ChatStreamResponse {
                id: "chatcmpl-ffi".to_string(),
                object: "chat.completion.chunk".to_string(),
                created: 0,
                model: "qwen-3.5-35b-ffi".to_string(),
                choices: vec![ChatStreamChoice {
                    index: 0,
                    delta: ChatStreamDelta {
                        role: None,
                        content: None,
                        tool_calls: None,
                    },
                    finish_reason: Some("stop".to_string()),
                }],
                stats: Some(stats),
            };
            let _ = tx.blocking_send(Ok(final_resp));
        });

        Ok(rx)
    }
}
