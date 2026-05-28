//! Handler: `recommended_variant` — proxy Argus's model selection over IPC.
//!
//! Lets any bundle component (apps, scripts, sub-agents) ask Hera "what local
//! model should this node be running?" via the same UDS they already use for
//! everything else. Pure read-through to Argus's HTTP endpoint; no caching here
//! because Argus probes hardware on every call and updates much faster than
//! Hera knows about.
//!
//! Request:  `{ "action": "recommended_variant", "payload": {} }`
//! Response: `{ "status": "success", "data": <RecommendedVariant> }`
//!           or `{ "status": "error", "data": { "error": "argus unreachable" } }`

use super::argus_client;
use super::types::{HandlerOutcome, IpcPayload, IpcResponse, IpcState, send_ipc_response};

pub async fn handle_recommended_variant(
    _request: &IpcPayload,
    _state: &IpcState,
    stream: &mut tokio::net::UnixStream,
) -> HandlerOutcome {
    let response = match argus_client::fetch_recommended_variant().await {
        Some(variant) => IpcResponse {
            status: "success".to_string(),
            data: serde_json::to_value(&variant).unwrap_or(serde_json::json!({})),
        },
        None => IpcResponse {
            status: "error".to_string(),
            data: serde_json::json!({
                "error": "argus unreachable: GET /api/recommended-variant failed (check pm2 argus status)"
            }),
        },
    };
    send_ipc_response(stream, &response).await;
    HandlerOutcome::DirectResponse
}
