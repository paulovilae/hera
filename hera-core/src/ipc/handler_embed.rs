//! Handler: embed — turn text into vectors using the local sentence-transformers
//! model. Lets other bundle components (e.g. Memento semantic recall) get
//! embeddings over IPC without loading an ML stack themselves.
//!
//! Request:  { "action": "embed", "payload": { "texts": ["..."] } }
//!           (also accepts a single { "text": "..." })
//! Response: { "status": "success", "data": { "embeddings": [[..]], "dim": 384 } }

use super::types::{HandlerOutcome, IpcPayload, IpcResponse, IpcState, send_ipc_response};

fn extract_texts(payload: &serde_json::Value) -> Vec<String> {
    if let Some(arr) = payload.get("texts").and_then(|v| v.as_array()) {
        return arr
            .iter()
            .filter_map(|v| v.as_str())
            .map(|s| s.to_string())
            .filter(|s| !s.trim().is_empty())
            .collect();
    }
    if let Some(text) = payload.get("text").and_then(|v| v.as_str()) {
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            return vec![trimmed.to_string()];
        }
    }
    Vec::new()
}

pub async fn handle_embed(
    request: &IpcPayload,
    _state: &IpcState,
    stream: &mut tokio::net::UnixStream,
) -> HandlerOutcome {
    let texts = extract_texts(&request.payload);
    if texts.is_empty() {
        send_ipc_response(
            stream,
            &IpcResponse {
                status: "error".to_string(),
                data: serde_json::json!({ "error": "embed: missing non-empty 'texts' or 'text'" }),
            },
        )
        .await;
        return HandlerOutcome::DirectResponse;
    }

    let response = compute_embeddings(texts).await;
    send_ipc_response(stream, &response).await;
    HandlerOutcome::DirectResponse
}

#[cfg(feature = "embeddings")]
async fn compute_embeddings(texts: Vec<String>) -> IpcResponse {
    // CPU-bound work: run off the async runtime.
    let result =
        tokio::task::spawn_blocking(move || crate::ai::embeddings::embed_texts(&texts)).await;

    match result {
        Ok(Ok(embeddings)) => {
            let dim = embeddings.first().map(|e| e.len()).unwrap_or(0);
            IpcResponse {
                status: "success".to_string(),
                data: serde_json::json!({ "embeddings": embeddings, "dim": dim }),
            }
        }
        Ok(Err(e)) => IpcResponse {
            status: "error".to_string(),
            data: serde_json::json!({ "error": format!("embed failed: {e}") }),
        },
        Err(e) => IpcResponse {
            status: "error".to_string(),
            data: serde_json::json!({ "error": format!("embed task join error: {e}") }),
        },
    }
}

#[cfg(not(feature = "embeddings"))]
async fn compute_embeddings(_texts: Vec<String>) -> IpcResponse {
    IpcResponse {
        status: "error".to_string(),
        data: serde_json::json!({
            "error": "embeddings unavailable: hera-core built without the `embeddings` feature on this node"
        }),
    }
}
