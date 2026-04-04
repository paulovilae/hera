use axum::{Json, Router, extract::Path, routing::post};
use serde::Serialize;
use std::net::SocketAddr;
use tower_http::cors::{Any, CorsLayer};
use tracing::{error, info};

use crate::ai::tool_executor::{ToolCall, execute_tool};

#[derive(Serialize)]
pub struct ApiResponse {
    pub success: bool,
    pub output: String,
}

pub async fn serve_rest_api(port: u16) {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/tools/:tool_name", post(handle_tool_execution))
        .layer(cors);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    info!(
        "🚀 Hera REST API (Direct Execution) bound to http://{}",
        addr
    );

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    if let Err(e) = axum::serve(listener, app).await {
        error!("❌ Hera REST API Server Error: {}", e);
    }
}

async fn handle_tool_execution(
    Path(tool_name): Path<String>,
    Json(payload): Json<serde_json::Value>,
) -> Json<ApiResponse> {
    info!(
        "⚡ [Hera REST] Direct execution request for tool: {}",
        tool_name
    );

    // Verify it's a valid object payload for arguments, or just package it up
    let arguments = if payload.is_object() {
        payload
    } else {
        serde_json::json!({})
    };

    let call = ToolCall {
        name: tool_name.clone(),
        arguments,
    };

    // We execute the requested tool synchronously in this tokio task
    let result = execute_tool(&call).await;

    Json(ApiResponse {
        success: result.success,
        output: result.output,
    })
}
