//! Local text embeddings via candle + a sentence-transformers BERT model
//! (all-MiniLM-L6-v2, 384-dim). Runs on CPU intentionally: the model is small,
//! a short text embeds in a few ms, and CPU tensors are Send+Sync (CUDA tensors
//! are not), which keeps the lazily-loaded model usable across tokio tasks.
//!
//! Exposed over IPC as the `embed` action so any bundle component (e.g. Memento
//! recall via Hera) can turn text into a vector without loading an ML stack.

use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use tokenizers::Tokenizer;

/// Directory holding config.json, tokenizer.json and model.safetensors.
/// Overridable so non-genesis nodes can point at their own copy.
const MODEL_DIR_ENV: &str = "HERA_EMBED_MODEL_DIR";
// Stable symlink → the multilingual sentence-transformers snapshot
// (paraphrase-multilingual-MiniLM-L12-v2, 384-dim, strong in Spanish). The
// symlink decouples the code from HF's content-hashed snapshot directory.
const DEFAULT_MODEL_DIR: &str = "/home/paulo/.cache/imagineos-embed-model";

struct EmbedModel {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

static MODEL: OnceLock<Mutex<EmbedModel>> = OnceLock::new();

fn model_dir() -> String {
    std::env::var(MODEL_DIR_ENV).unwrap_or_else(|_| DEFAULT_MODEL_DIR.to_string())
}

fn load_model() -> Result<EmbedModel> {
    let dir = model_dir();
    let device = Device::Cpu;

    let config_str = std::fs::read_to_string(format!("{dir}/config.json"))
        .with_context(|| format!("embed: reading config.json in {dir}"))?;
    let config: Config =
        serde_json::from_str(&config_str).context("embed: parsing BERT config.json")?;

    let mut tokenizer = Tokenizer::from_file(format!("{dir}/tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("embed: loading tokenizer.json: {e}"))?;
    // Deterministic single-sequence encoding; we do our own mean pooling.
    if let Some(pp) = tokenizer.get_padding_mut() {
        pp.strategy = tokenizers::PaddingStrategy::BatchLongest;
    }

    let weights_path = format!("{dir}/model.safetensors");
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)? };
    let model = BertModel::load(vb, &config).context("embed: loading BertModel weights")?;

    Ok(EmbedModel {
        model,
        tokenizer,
        device,
    })
}

fn model() -> Result<&'static Mutex<EmbedModel>> {
    if let Some(m) = MODEL.get() {
        return Ok(m);
    }
    let loaded = load_model()?;
    // Race-tolerant: if another thread set it first, ours is dropped.
    let _ = MODEL.set(Mutex::new(loaded));
    MODEL
        .get()
        .ok_or_else(|| anyhow::anyhow!("embed: model cell unexpectedly empty"))
}

/// L2-normalized mean-pooled embedding for one text. 384 floats for MiniLM-L6.
fn embed_one(m: &EmbedModel, text: &str) -> Result<Vec<f32>> {
    let encoding = m
        .tokenizer
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!("embed: tokenize failed: {e}"))?;
    let ids = encoding.get_ids().to_vec();
    let mask: Vec<u32> = encoding.get_attention_mask().to_vec();
    let seq_len = ids.len();

    let input_ids = Tensor::new(ids.as_slice(), &m.device)?.unsqueeze(0)?;
    let token_type_ids = input_ids.zeros_like()?;
    let attention_mask = Tensor::new(mask.as_slice(), &m.device)?.unsqueeze(0)?;

    // [1, seq, hidden]
    let hidden = m
        .model
        .forward(&input_ids, &token_type_ids, Some(&attention_mask))?;

    // Mean pooling weighted by the attention mask.
    let mask_f = attention_mask.to_dtype(DType::F32)?.unsqueeze(2)?; // [1, seq, 1]
    let masked = hidden.broadcast_mul(&mask_f)?; // [1, seq, hidden]
    let summed = masked.sum(1)?; // [1, hidden]
    let counts = mask_f.sum(1)?; // [1, 1]
    let mean = summed.broadcast_div(&(counts + 1e-9)?)?; // [1, hidden]

    // L2 normalize.
    let norm = mean.sqr()?.sum_keepdim(1)?.sqrt()?; // [1, 1]
    let normed = mean.broadcast_div(&(norm + 1e-12)?)?; // [1, hidden]

    let vec = normed.squeeze(0)?.to_vec1::<f32>()?;
    let _ = seq_len;
    Ok(vec)
}

/// Embed a batch of texts (sequentially; texts are short and few per request).
pub fn embed_texts(texts: &[String]) -> Result<Vec<Vec<f32>>> {
    let guard = model()?
        .lock()
        .map_err(|_| anyhow::anyhow!("embed: model mutex poisoned"))?;
    let mut out = Vec::with_capacity(texts.len());
    for t in texts {
        out.push(embed_one(&guard, t)?);
    }
    Ok(out)
}

/// Embedding dimensionality (384 for all-MiniLM-L6-v2). Loads the model on first call.
pub fn embed_dim() -> Result<usize> {
    let v = embed_texts(&["dim_probe".to_string()])?;
    Ok(v.first().map(|e| e.len()).unwrap_or(0))
}
