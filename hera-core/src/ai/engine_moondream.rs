use crate::ai::{ChatRequest, ChatResponse, ChatChoice, ChatResponseMessage, ChatUsage, InferenceError, LLMEngine, MessageContent, ContentPart, ChatStreamResponse};
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tracing::{info, error};

use candle_transformers::{
    generation::LogitsProcessor,
    models::quantized_moondream,
};
use tokenizers::Tokenizer;
use candle_core as candle;
use candle_core::{DType, Device, Tensor};

pub struct MoondreamFfiEngine {
    model: Arc<Mutex<quantized_moondream::Model>>,
    tokenizer: Arc<Tokenizer>,
    device: Device,
}

impl MoondreamFfiEngine {
    pub async fn new() -> Result<Self, String> {
        info!("👁️ [Vision] Initializing Native Moondream Engine...");
        
        // Define local model paths rather than using HuggingFace Hub to prevent DNS and Tokio issues
        let model_file = std::path::PathBuf::from("/data/models/moondream/model-q4_0.gguf");
        let tokenizer_file = std::path::PathBuf::from("/data/models/moondream/tokenizer.json");
        
        info!("👁️ [Vision] Loading local tokenizer...");
        let tokenizer = Tokenizer::from_file(&tokenizer_file)
            .map_err(|e| format!("Tokenizer load error: {}", e))?;

        // Try GPU 1 first (less contended — GPU 0 hosts LLM + sd-server),
        // then GPU 0, then CPU as ultimate fallback
        let device = if candle::utils::cuda_is_available() {
            Device::new_cuda(1)
                .or_else(|_| Device::new_cuda(0))
                .unwrap_or(Device::Cpu)
        } else {
            Device::Cpu
        };

        info!("👁️ [Vision] Loading Moondream quantized model into {:?}...", device);
        let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
            &model_file,
            &device,
        ).map_err(|e| format!("GGUF var builder error: {}", e))?;
        
        let config = candle_transformers::models::moondream::Config::v2();
        let model = quantized_moondream::Model::new(&config, vb)
            .map_err(|e| format!("Model instantiation error: {}", e))?;

        info!("✅ [Vision] Native Moondream loaded successfully.");

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            tokenizer: Arc::new(tokenizer),
            device,
        })
    }

    /// Loads base64 dataURI to a Tensor
    fn load_image_from_base64(&self, b64: &str) -> Result<Tensor, String> {
        let clean_b64 = if b64.starts_with("data:image") {
            b64.split(',').nth(1).unwrap_or(b64)
        } else {
            b64
        };
        let image_data = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, clean_b64)
            .map_err(|e: base64::DecodeError| format!("Base64 decode error: {}", e))?;

        let img = image::load_from_memory(&image_data)
            .map_err(|e: image::ImageError| format!("Image parse error: {}", e))?
            .resize_to_fill(378, 378, image::imageops::FilterType::Triangle)
            .to_rgb8();

        let data = img.into_raw();
        let data_tensor = Tensor::from_vec(data, (378, 378, 3), &Device::Cpu)
            .map_err(|e: candle_core::Error| e.to_string())?
            .permute((2, 0, 1))
            .map_err(|e: candle_core::Error| e.to_string())?;

        let mean = Tensor::new(&[0.5f32, 0.5, 0.5], &Device::Cpu)
            .unwrap()
            .reshape((3, 1, 1))
            .unwrap();
            
        let std = Tensor::new(&[0.5f32, 0.5, 0.5], &Device::Cpu)
            .unwrap()
            .reshape((3, 1, 1))
            .unwrap();

        let normalized = (data_tensor.to_dtype(candle_core::DType::F32).unwrap() / 255.).unwrap()
            .broadcast_sub(&mean).unwrap()
            .broadcast_div(&std).unwrap();

        normalized.to_device(&self.device).map_err(|e: candle_core::Error| e.to_string())
    }
}

#[async_trait]
impl LLMEngine for MoondreamFfiEngine {
    async fn generate_content(&self, req: ChatRequest) -> Result<ChatResponse, InferenceError> {
        // Extract prompt & image
        let mut prompt_text = String::new();
        let mut base64_image = String::new();

        for msg in &req.messages {
            if let MessageContent::Parts(parts) = &msg.content {
                for part in parts {
                    match part {
                        ContentPart::Text { text } => {
                            prompt_text.push_str(text);
                            prompt_text.push(' ');
                        }
                        ContentPart::ImageUrl { image_url } => {
                            base64_image = image_url.url.clone();
                        }
                    }
                }
            }
        }

        if base64_image.is_empty() {
            prompt_text = req.messages.last().map(|m| match &m.content {
                MessageContent::Text(t) => t.clone(),
                    MessageContent::Null => String::new(),
                MessageContent::Parts(p) => p.iter().filter_map(|part| match part {
                    ContentPart::Text { text } => Some(text.clone()),
                    _ => None,
                }).collect::<Vec<_>>().join(" "),
            }).unwrap_or_default();
        }

        if prompt_text.is_empty() {
            prompt_text = "Describe this image in detail.".to_string();
        }

        let formatted_prompt = format!("\n\nQuestion: {0}\n\nAnswer:", prompt_text.trim());
        let tokens = self.tokenizer.encode(formatted_prompt, true).unwrap();
        let mut token_ids = tokens.get_ids().to_vec();

        let special_token = self.tokenizer.get_vocab(true).get("<|endoftext|>").copied().unwrap_or(50256);

        let mut model = self.model.lock().await;
        model.text_model.clear_kv_cache();

        let image_tensor = if !base64_image.is_empty() {
            info!("👁️ [Vision] Parsing image...");
            let image = self.load_image_from_base64(&base64_image)
                .map_err(|e| InferenceError::ExecutionFailed(format!("Image prep error: {}", e)))?;
            let image_embeds = image.unsqueeze(0).unwrap();
            let encoded = image_embeds.apply(model.vision_encoder())
                .map_err(|e| InferenceError::ExecutionFailed(format!("Vision encoder error: {}", e)))?;
            Some(encoded)
        } else {
            None
        };

        info!("👁️ [Vision] Generating text...");
        let mut generated_text = String::new();
        let mut logits_processor = candle_transformers::generation::LogitsProcessor::new(299792458, req.temperature.map(|t| t as f64).filter(|&t| t > 0.0), None);
        
        let max_tokens = req.max_tokens.unwrap_or(250) as usize;

        // Perform initial forward pass with the full context (prompt + optional image)
        let initial_input = Tensor::new(token_ids.as_slice(), &self.device).unwrap().unsqueeze(0).unwrap();
        
        let mut logits = if let Some(embeds) = &image_tensor {
            let bos_tensor = Tensor::new(&[special_token], &self.device).unwrap().unsqueeze(0).unwrap();
            model.text_model.forward_with_img(&bos_tensor, &initial_input, embeds).unwrap()
        } else {
            model.text_model.forward(&initial_input).unwrap()
        };

        let logits_sq = logits.squeeze(0).unwrap();
        let logits_f32 = if logits_sq.rank() == 1 {
            logits_sq
        } else {
            let seq_len = logits_sq.dim(0).unwrap();
            logits_sq.get(seq_len - 1).unwrap()
        }.to_dtype(DType::F32).unwrap();

        // Process the first generated token
        let mut next_token = logits_processor.sample(&logits_f32).unwrap();
        token_ids.push(next_token);

        if let Ok(piece) = self.tokenizer.decode(&[next_token], true) {
            generated_text.push_str(&piece);
        }

        // Auto-regressive generation loop
        for _ in 1..max_tokens {
            if next_token == special_token || token_ids.ends_with(&[27, 10619, 29]) {
                break;
            }

            // Only feed the newly generated token; the KV cache handles the rest
            let input = Tensor::new(&[next_token], &self.device).unwrap().unsqueeze(0).unwrap();
            
            logits = model.text_model.forward(&input).unwrap();
            let logits_sq = logits.squeeze(0).unwrap();
            let logits_f32 = if logits_sq.rank() == 1 {
                logits_sq
            } else {
                let seq_len = logits_sq.dim(0).unwrap();
                logits_sq.get(seq_len - 1).unwrap()
            }.to_dtype(DType::F32).unwrap();
            
            next_token = logits_processor.sample(&logits_f32).unwrap();
            token_ids.push(next_token);

            if let Ok(piece) = self.tokenizer.decode(&[next_token], true) {
                generated_text.push_str(&piece);
            }
        }


        Ok(ChatResponse {
            id: "chatcmpl-vision".to_string(),
            object: "chat.completion".to_string(),
            created: 0,
            model: "moondream-q4".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatResponseMessage {
                    role: "assistant".to_string(),
                    content: Some(generated_text.trim().to_string()),
                    tool_calls: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(ChatUsage { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }),
        })
    }

    async fn generate_stream(&self, req: ChatRequest) -> Result<mpsc::Receiver<Result<ChatStreamResponse, InferenceError>>, InferenceError> {
        let (tx, rx) = mpsc::channel(1);
        let resp = self.generate_content(req).await;
        tokio::spawn(async move {
            match resp {
                Ok(r) => {
                    let _ = tx.send(Ok(ChatStreamResponse {
                        id: r.id,
                        object: "chat.completion.chunk".to_string(),
                        created: r.created,
                        model: r.model,
                        choices: vec![crate::ai::ChatStreamChoice {
                            index: 0,
                            delta: crate::ai::ChatStreamDelta {
                                role: Some("assistant".to_string()),
                                content: r.choices[0].message.content.clone(),
                                tool_calls: None,
                            },
                            finish_reason: Some("stop".to_string()),
                        }],
                        stats: None,
                    })).await;
                }
                Err(e) => {
                    let _ = tx.send(Err(e)).await;
                }
            }
        });
        Ok(rx)
    }
}
