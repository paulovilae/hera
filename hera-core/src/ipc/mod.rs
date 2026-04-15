//! Modular IPC subsystem — thin dispatcher routing to domain-specific handlers.
//!
//! Architecture:
//!   ipc_server.rs (legacy re-export) → ipc/mod.rs (dispatcher) → handler_*.rs
//!
//! Each handler returns a `HandlerOutcome` that either:
//!   - `DirectResponse`: the handler already wrote its own response to the socket
//!   - `Result { ... }`: data for the dispatcher to format and send

pub mod context;
pub mod handler_audio;
pub mod handler_dag;
pub mod handler_generate;
pub mod handler_lora;
pub mod handler_media;
pub mod handler_stream;
pub mod handler_tools;
pub mod helpers;
pub mod types;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixListener;
use tracing::{error, info};

pub use types::{HandlerOutcome, IpcPayload, IpcResponse, IpcState};

/// Main IPC server loop — binds to Unix socket and dispatches to handlers.
pub async fn serve(socket_path: &str, state: IpcState) -> std::io::Result<()> {
    // Clean up stale socket
    if std::path::Path::new(socket_path).exists() {
        std::fs::remove_file(socket_path)?;
    }

    let listener = UnixListener::bind(socket_path)?;
    info!(
        "🔗 Headless IPC Daemon bound to Unix socket: {}",
        socket_path
    );

    loop {
        match listener.accept().await {
            Ok((mut stream, _addr)) => {
                let state = state.clone();
                tokio::spawn(async move {
                    let mut buffer = Vec::new();
                    let mut chunk = vec![0; 65536];
                    loop {
                        match stream.read(&mut chunk).await {
                            Ok(n) if n > 0 => {
                                buffer.extend_from_slice(&chunk[..n]);
                                match serde_json::from_slice::<IpcPayload>(&buffer) {
                                    Ok(request) => {
                                        info!("📥 Received IPC Action: {}", request.action);
                                        let outcome = dispatch(&request, &state, &mut stream).await;
                                        match outcome {
                                            HandlerOutcome::DirectResponse => {
                                                // Handler already wrote to stream
                                            }
                                            HandlerOutcome::Result {
                                                result_text,
                                                origin,
                                                model,
                                                tool_calls,
                                            } => {
                                                send_result(
                                                    &mut stream,
                                                    &result_text,
                                                    &origin,
                                                    &model,
                                                    tool_calls,
                                                )
                                                .await;
                                            }
                                        }
                                        break;
                                    }
                                    Err(e) => {
                                        if e.is_eof() {
                                            continue;
                                        }
                                        error!(
                                            "❌ IPC JSON Parse Error: {} - buffered_bytes={}",
                                            e,
                                            buffer.len()
                                        );
                                        let err_msg = IpcResponse {
                                            status: "error".to_string(),
                                            data: serde_json::json!({
                                                "error": format!("Invalid JSON: {}", e)
                                            }),
                                        };
                                        let mut estr = serde_json::to_string(&err_msg).unwrap();
                                        estr.push('\n');
                                        let _ = stream.write_all(estr.as_bytes()).await;
                                        break;
                                    }
                                }
                            }
                            Ok(_) => break, // EOF
                            Err(e) => {
                                error!("❌ IPC Stream Read Error: {}", e);
                                break;
                            }
                        }
                    }
                });
            }
            Err(e) => {
                error!("❌ IPC Listener Accept Error: {}", e);
            }
        }
    }
}

/// Route an IPC request to the correct handler based on `action`.
async fn dispatch(
    request: &IpcPayload,
    state: &IpcState,
    stream: &mut tokio::net::UnixStream,
) -> HandlerOutcome {
    match request.action.as_str() {
        "execute_tool" => handler_tools::handle_execute_tool(request, state, stream).await,
        "generate" => handler_generate::handle_generate(request, state, stream).await,
        "generate_stream" => handler_stream::handle_generate_stream(request, state, stream).await,
        "generate_image" => handler_media::handle_generate_image(request, state).await,
        "vision_analysis" => handler_media::handle_vision_analysis(request, state).await,
        "execute_dag" => handler_dag::handle_execute_dag(request, state).await,
        "generate_video" | "animate_image" => {
            handler_media::handle_generate_video(request, state).await
        }
        "transcribe_audio" => handler_audio::handle_transcribe_audio(request, state).await,
        "get_tools" => handler_tools::handle_get_tools(request, state),
        "download_lora" => handler_lora::handle_download_lora(request).await,
        _ => HandlerOutcome::Result {
            result_text: format!("Unknown action: {}", request.action),
            origin: "unknown".to_string(),
            model: String::new(),
            tool_calls: None,
        },
    }
}

/// Format and send a `HandlerOutcome::Result` as an IPC response.
async fn send_result(
    stream: &mut tokio::net::UnixStream,
    result_text: &str,
    origin: &str,
    model: &str,
    tool_calls: Option<serde_json::Value>,
) {
    let mut data_json = serde_json::json!({ "result": result_text });
    if let Some(tc) = tool_calls
        && let Some(map) = data_json.as_object_mut()
    {
        map.insert("tool_calls".to_string(), tc);
    }
    if let Some(map) = data_json.as_object_mut() {
        map.insert("origin".to_string(), serde_json::json!(origin));
        map.insert("model".to_string(), serde_json::json!(model));
    }

    let res = IpcResponse {
        status: "success".to_string(),
        data: data_json,
    };
    let mut res_str = serde_json::to_string(&res).unwrap();
    res_str.push('\n');
    if let Err(e) = stream.write_all(res_str.as_bytes()).await {
        error!("❌ Failed to write IPC response: {}", e);
    }
}
