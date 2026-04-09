//! IPC type definitions — structs shared across all handler modules.

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use tokio::net::UnixStream;

use crate::ai::engine_parler::ParlerEngine;
use crate::ai::{LLMEngine, SpeechToTextEngine};

/// Shared engine state passed to every handler.
#[derive(Clone)]
pub struct IpcState {
    pub engine: Arc<dyn LLMEngine + Send + Sync>,
    pub local_engine: Arc<dyn LLMEngine + Send + Sync>,
    pub flux_engine: Option<Arc<crate::ai::engine_flux::FluxEngine>>,
    pub parler_engine: Option<Arc<ParlerEngine>>,
    pub whisper_engine: Option<Arc<dyn SpeechToTextEngine + Send + Sync>>,
    pub vision_engine: Option<Arc<dyn LLMEngine + Send + Sync>>,
    pub micro_engine: Option<Arc<dyn LLMEngine + Send + Sync>>,
}

/// Incoming IPC request.
#[derive(Deserialize, Debug)]
pub struct IpcPayload {
    pub action: String,
    pub payload: serde_json::Value,
}

/// Outgoing IPC response.
#[derive(Serialize, Debug)]
pub struct IpcResponse {
    pub status: String,
    pub data: serde_json::Value,
}

/// Outcome of a handler invocation.
pub enum HandlerOutcome {
    /// Handler wrote its own response to the stream (e.g. streaming, fast-path).
    DirectResponse,
    /// Handler produced result data for the dispatcher to format and send.
    Result {
        result_text: String,
        origin: String,
        model: String,
        tool_calls: Option<serde_json::Value>,
    },
}

/// Write a serialized `IpcResponse` to the Unix socket stream.
pub async fn send_ipc_response(stream: &mut UnixStream, response: &IpcResponse) {
    let mut res_str = serde_json::to_string(response).unwrap();
    res_str.push('\n');
    if let Err(e) = stream.write_all(res_str.as_bytes()).await {
        tracing::error!("❌ Failed to write IPC response: {}", e);
    }
}
