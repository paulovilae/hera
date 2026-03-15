//! T5 model implementation with quantization support.
//! Customized for GGUF parsing.

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::Activation;
use serde::Deserialize;
use std::sync::Arc;
use candle_core::quantized::QTensor;
use candle_transformers::models::t5::{deserialize_feed_forward_proj_activation, ActivationWithOptionalGating};
pub use candle_transformers::quantized_var_builder::VarBuilder;

#[derive(Debug, Clone)]
pub struct QMatMul {
    inner: candle_core::quantized::QMatMul,
    span: tracing::Span,
}

impl QMatMul {
    pub fn new_exact(name: &str, vb: &VarBuilder) -> Result<Self> {
        let weight = vb.get_no_shape(name)?;
        Ok(Self {
            inner: candle_core::quantized::QMatMul::from_arc(weight)?,
            span: tracing::span!(tracing::Level::TRACE, "qmatmul"),
        })
    }
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

#[derive(Debug, Clone)]
pub struct Embedding {
    inner: candle_nn::Embedding,
    span: tracing::Span,
}

impl Embedding {
    pub fn new_exact(name: &str, hidden_size: usize, vb: &VarBuilder) -> Result<Self> {
        let embeddings = vb.get_no_shape(name)?.dequantize(vb.device())?;
        let inner = candle_nn::Embedding::new(embeddings, hidden_size);
        Ok(Self { inner, span: tracing::span!(tracing::Level::TRACE, "embedding") })
    }
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

fn default_relative_attention_max_distance() -> usize { 128 }
fn default_is_decoder() -> bool { false }
fn default_use_cache() -> bool { true }

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub d_model: usize,
    pub d_kv: usize,
    pub d_ff: usize,
    pub num_layers: usize,
    pub num_decoder_layers: Option<usize>,
    pub num_heads: usize,
    pub relative_attention_num_buckets: usize,
    #[serde(default = "default_relative_attention_max_distance")]
    pub relative_attention_max_distance: usize,
    pub dropout_rate: f64,
    pub layer_norm_epsilon: f64,
    pub initializer_factor: f64,
    #[serde(default, deserialize_with = "deserialize_feed_forward_proj_activation")]
    pub feed_forward_proj: ActivationWithOptionalGating,
    #[serde(default = "default_is_decoder")]
    pub is_decoder: bool,
    #[serde(default)]
    pub is_encoder_decoder: bool,
    #[serde(default = "default_use_cache")]
    pub use_cache: bool,
    #[serde(default)]
    pub pad_token_id: usize,
    #[serde(default)]
    pub eos_token_id: usize,
}

#[derive(Debug, Clone)]
struct T5LayerNorm {
    weight: Tensor,
    variance_epsilon: f64,
    span: tracing::Span,
}

impl T5LayerNorm {
    fn load(eps: f64, name: &str, vb: &VarBuilder) -> Result<Self> {
        let weight = vb.get_no_shape(name)?.dequantize(vb.device())?;
        Ok(Self {
            weight,
            variance_epsilon: eps,
            span: tracing::span!(tracing::Level::TRACE, "layer-norm"),
        })
    }
}

impl Module for T5LayerNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let dtype = xs.dtype();
        let xs_f32 = xs.to_dtype(DType::F32)?;
        let variance = xs_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let xs = xs.broadcast_div(&(variance + self.variance_epsilon)?.sqrt()?)?;
        let xs = xs.to_dtype(dtype)?;
        let xs = xs.broadcast_mul(&self.weight)?;
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
struct T5DenseActDense {
    wi: QMatMul,
    wo: QMatMul,
    act: Activation,
    span: tracing::Span,
}

impl T5DenseActDense {
    fn load(prefix: &str, vb: &VarBuilder, _cfg: &Config) -> Result<Self> {
        let wi = QMatMul::new_exact(&format!("{}.ffn_up.weight", prefix), vb)?;
        let wo = QMatMul::new_exact(&format!("{}.ffn_down.weight", prefix), vb)?;
        Ok(Self {
            wi,
            wo,
            act: Activation::Relu,
            span: tracing::span!(tracing::Level::TRACE, "dense-act-dense"),
        })
    }
}

impl Module for T5DenseActDense {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let xs = self.wi.forward(xs)?;
        let xs = self.act.forward(&xs)?;
        let xs = self.wo.forward(&xs)?;
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
struct T5DenseGatedActDense {
    wi_0: QMatMul,
    wi_1: QMatMul,
    wo: QMatMul,
    act: Activation,
    span: tracing::Span,
}

impl T5DenseGatedActDense {
    fn load(prefix: &str, vb: &VarBuilder, cfg: &Config) -> Result<Self> {
        let wi_0 = QMatMul::new_exact(&format!("{}.ffn_up.weight", prefix), vb)?;
        let wi_1 = QMatMul::new_exact(&format!("{}.ffn_gate.weight", prefix), vb)?;
        let wo = QMatMul::new_exact(&format!("{}.ffn_down.weight", prefix), vb)?;
        Ok(Self {
            wi_0,
            wi_1,
            wo,
            act: cfg.feed_forward_proj.activation,
            span: tracing::span!(tracing::Level::TRACE, "dense-gated-act-dense"),
        })
    }
}

impl Module for T5DenseGatedActDense {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let hidden_gelu = self.act.forward(&self.wi_0.forward(xs)?)?;
        let hidden_linear = self.wi_1.forward(xs)?;
        let xs = hidden_gelu.broadcast_mul(&hidden_linear)?;
        let xs = self.wo.forward(&xs)?;
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
struct T5LayerFF {
    dense_act: Option<T5DenseActDense>,
    gated_dense_act: Option<T5DenseGatedActDense>,
    layer_norm: T5LayerNorm,
    span: tracing::Span,
}

impl T5LayerFF {
    fn load(prefix: &str, vb: &VarBuilder, cfg: &Config) -> Result<Self> {
        let layer_norm =
            T5LayerNorm::load(cfg.layer_norm_epsilon, &format!("{}.ffn_norm.weight", prefix), vb)?;
        let (dense_act, gated_dense_act) = if cfg.feed_forward_proj.gated {
            (
                None,
                Some(T5DenseGatedActDense::load(prefix, vb, cfg)?),
            )
        } else {
            (
                Some(T5DenseActDense::load(prefix, vb, cfg)?),
                None,
            )
        };
        Ok(Self {
            dense_act,
            gated_dense_act,
            layer_norm,
            span: tracing::span!(tracing::Level::TRACE, "layer-ff"),
        })
    }
}

impl Module for T5LayerFF {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let ys = self.layer_norm.forward(xs)?;
        let ys = match &self.dense_act {
            Some(dense_act) => dense_act.forward(&ys)?,
            None => self.gated_dense_act.as_ref().unwrap().forward(&ys)?,
        };
        let xs = (xs + ys)?;
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
struct T5Attention {
    q: QMatMul,
    k: QMatMul,
    v: QMatMul,
    o: QMatMul,
    n_heads: usize,
    d_kv: usize,
    relative_attention_bias: Option<Embedding>,
    relative_attention_num_buckets: usize,
    relative_attention_max_distance: usize,
    inner_dim: usize,
    span: tracing::Span,
    span_mm: tracing::Span,
    span_sm: tracing::Span,
}

impl T5Attention {
    fn load(
        prefix: &str,
        has_relative_attention_bias: bool,
        vb: &VarBuilder,
        cfg: &Config,
    ) -> Result<Self> {
        let inner_dim = cfg.num_heads * cfg.d_kv;
        let q = QMatMul::new_exact(&format!("{}.attn_q.weight", prefix), vb)?;
        let k = QMatMul::new_exact(&format!("{}.attn_k.weight", prefix), vb)?;
        let v = QMatMul::new_exact(&format!("{}.attn_v.weight", prefix), vb)?;
        let o = QMatMul::new_exact(&format!("{}.attn_o.weight", prefix), vb)?;
        let relative_attention_bias = if has_relative_attention_bias {
            let emb = Embedding::new_exact(
                &format!("{}.attn_rel_b.weight", prefix),
                cfg.num_heads,
                vb,
            )?;
            Some(emb)
        } else {
            None
        };
        Ok(Self {
            q,
            k,
            v,
            o,
            n_heads: cfg.num_heads,
            d_kv: cfg.d_kv,
            relative_attention_bias,
            relative_attention_num_buckets: cfg.relative_attention_num_buckets,
            relative_attention_max_distance: cfg.relative_attention_max_distance,
            inner_dim,
            span: tracing::span!(tracing::Level::TRACE, "attention"),
            span_mm: tracing::span!(tracing::Level::TRACE, "attention-mm"),
            span_sm: tracing::span!(tracing::Level::TRACE, "attention-sm"),
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        position_bias: Option<&Tensor>,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let _enter = self.span.enter();
        let (b_sz, q_len) = (xs.dim(0)?, xs.dim(1)?);
        let kv_len = q_len;
        let q = self.q.forward(xs)?;
        let k = self.k.forward(xs)?;
        let v = self.v.forward(xs)?;
        let q = q
            .reshape((b_sz, q_len, self.n_heads, self.d_kv))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, kv_len, self.n_heads, self.d_kv))?
            .transpose(1, 2)?.contiguous()?;
        let v = v
            .reshape((b_sz, kv_len, self.n_heads, self.d_kv))?
            .transpose(1, 2)?.contiguous()?;

        let scores = {
            let _enter = self.span_mm.enter();
            q.matmul(&k.t()?)?
        };
        let scores = match mask {
            None => scores,
            Some(mask) => masked_fill(
                &scores,
                &mask
                    .unsqueeze(0)?
                    .unsqueeze(0)?
                    .repeat((b_sz, self.n_heads))?,
                f32::NEG_INFINITY,
            )?,
        };

        let (scores, position_bias) = match position_bias {
            Some(position_bias) => (
                scores.broadcast_add(position_bias)?,
                Some(position_bias.clone()),
            ),
            None => match &self.relative_attention_bias {
                None => (scores, None),
                Some(relative_attention_bias) => {
                    let (q_start, q_end) = (0_u32, kv_len as u32);
                    let num_buckets = self.relative_attention_num_buckets as u32 / 2;
                    let max_exact = num_buckets / 2;
                    let relative_position = (q_start..q_end)
                        .map(|i| {
                            (0..kv_len as u32)
                                .map(|j| {
                                    if i < j {
                                        if j - i < max_exact { j - i + num_buckets } else {
                                            let b = f32::log(
                                                (j - i) as f32 / max_exact as f32,
                                                self.relative_attention_max_distance as f32 / max_exact as f32,
                                            ) * (num_buckets - max_exact) as f32;
                                            u32::min(max_exact + num_buckets + b as u32, self.relative_attention_num_buckets as u32 - 1)
                                        }
                                    } else if i - j < max_exact { i - j } else {
                                        let b = f32::log(
                                            (i - j) as f32 / max_exact as f32,
                                            self.relative_attention_max_distance as f32 / max_exact as f32,
                                        ) * (num_buckets - max_exact) as f32;
                                        max_exact + b as u32
                                    }
                                })
                                .collect::<Vec<u32>>()
                        })
                        .collect::<Vec<Vec<_>>>();
                    let relative_buckets = Tensor::new(relative_position, q.device())?;
                    let position_bias = relative_attention_bias
                        .forward(&relative_buckets)?
                        .permute((2, 0, 1))?
                        .unsqueeze(0)?;
                    (scores.broadcast_add(&position_bias)?, Some(position_bias))
                }
            },
        };

        let attn_weights = {
            let _enter = self.span_sm.enter();
            candle_nn::ops::softmax_last_dim(&scores)?
        };
        let attn_output = attn_weights.matmul(&v)?;
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.inner_dim))?;
        let attn_output = self.o.forward(&attn_output)?;
        Ok((attn_output, position_bias))
    }
}

#[derive(Debug, Clone)]
struct T5Block {
    self_attn: T5Attention,
    layer_norm: T5LayerNorm,
    ff: T5LayerFF,
    span: tracing::Span,
}

impl T5Block {
    fn load(
        prefix: &str,
        has_relative_attention_bias: bool,
        vb: &VarBuilder,
        cfg: &Config,
    ) -> Result<Self> {
        let self_attn = T5Attention::load(prefix, has_relative_attention_bias, vb, cfg)?;
        let layer_norm = T5LayerNorm::load(cfg.layer_norm_epsilon, &format!("{}.attn_norm.weight", prefix), vb)?;
        let ff = T5LayerFF::load(prefix, vb, cfg)?;
        Ok(Self {
            self_attn,
            layer_norm,
            ff,
            span: tracing::span!(tracing::Level::TRACE, "block"),
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        position_bias: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let _enter = self.span.enter();
        let normed_xs = self.layer_norm.forward(xs)?;
        let (ys, position_bias) = self.self_attn.forward(&normed_xs, position_bias, None)?;
        let xs = (xs + ys)?;
        let xs = self.ff.forward(&xs)?;
        Ok((xs, position_bias))
    }
}

#[derive(Debug, Clone)]
struct T5Stack {
    block: Vec<T5Block>,
    shared: Arc<Embedding>,
    final_layer_norm: T5LayerNorm,
    span: tracing::Span,
}

impl T5Stack {
    fn load(vb: &VarBuilder, shared: &Arc<Embedding>, cfg: &Config) -> Result<Self> {
        let block = (0..cfg.num_layers)
            .map(|i| T5Block::load(&format!("enc.blk.{}", i), i == 0, vb, cfg))
            .collect::<Result<Vec<_>>>()?;
        let final_layer_norm = T5LayerNorm::load(
            cfg.layer_norm_epsilon,
            "enc.output_norm.weight",
            vb,
        )?;
        Ok(Self {
            block,
            shared: shared.clone(),
            final_layer_norm,
            span: tracing::span!(tracing::Level::TRACE, "stack"),
        })
    }

    fn forward(
        &mut self,
        input_ids: &Tensor,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let input_embeds = self.shared.as_ref().forward(input_ids)?;
        let mut hidden_states = input_embeds;
        let mut position_bias = None;
        for block in self.block.iter_mut() {
            (hidden_states, position_bias) = block.forward(
                &hidden_states,
                position_bias.as_ref(),
            )?
        }
        self.final_layer_norm.forward(&hidden_states)
    }
}

#[derive(Debug, Clone)]
pub struct T5EncoderModel {
    encoder: T5Stack,
    device: Device,
    span: tracing::Span,
}

impl T5EncoderModel {
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let shared = Embedding::new_exact("token_embd.weight", cfg.d_model, &vb)?;
        let shared = Arc::new(shared);
        let encoder = T5Stack::load(&vb, &shared, cfg)?;
        Ok(Self {
            encoder,
            device: vb.device().clone(),
            span: tracing::span!(tracing::Level::TRACE, "encoder"),
        })
    }

    pub fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.encoder.forward(input_ids)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}
