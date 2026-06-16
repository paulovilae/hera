//! Modular IPC subsystem — thin dispatcher routing to domain-specific handlers.
//!
//! Architecture:
//!   ipc_server.rs (legacy re-export) → ipc/mod.rs (dispatcher) → handler_*.rs
//!
//! Each handler returns a `HandlerOutcome` that either:
//!   - `DirectResponse`: the handler already wrote its own response to the socket
//!   - `Result { ... }`: data for the dispatcher to format and send

pub mod agentic_loop;
pub mod argus_client;
pub mod context;
pub mod difficulty;
pub mod handler_argus;
pub mod handler_audio;
pub mod handler_dag;
pub mod handler_delegation;
pub mod handler_embed;
pub mod handler_generate;
pub mod handler_health;
pub mod handler_lora;
pub mod handler_media;
pub mod handler_stream;
pub mod handler_tools;
pub mod helpers;
pub mod llm_audit;
pub mod route_profiles;
pub mod runtime_tools;
pub mod types;

use std::os::unix::fs::PermissionsExt;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixListener;
use tokio::sync::Semaphore;
use tracing::{error, info, warn};

pub use types::{HandlerOutcome, IpcPayload, IpcResponse, IpcState};

/// Maximum concurrent LLM requests — prevents thread-pile-up under burst load.
const MAX_CONCURRENT_REQUESTS: usize = 12;

/// Tope duro del buffer de lectura por conexión. Sin esto, un cliente que envía
/// bytes sin cerrar un JSON válido hace crecer el buffer indefinidamente (OOM).
const MAX_REQUEST_BYTES: usize = 8 * 1024 * 1024;

/// Main IPC server loop — binds to Unix socket and dispatches to handlers.
pub async fn serve(socket_path: &str, state: IpcState) -> std::io::Result<()> {
    // Clean up stale socket
    if std::path::Path::new(socket_path).exists() {
        std::fs::remove_file(socket_path)?;
    }

    let listener = UnixListener::bind(socket_path)?;
    // Restringir el socket al usuario propietario (0600). /tmp es world-traversable;
    // sin esto cualquier proceso/usuario local puede conectar y ejecutar acciones
    // (execute_tool, generate con tools, run_code) = RCE local sin autenticar. El
    // kernel aplica el permiso del inodo en connect(), igual que hace Memento.
    std::fs::set_permissions(socket_path, std::fs::Permissions::from_mode(0o600))?;
    info!(
        "🔗 Headless IPC Daemon bound to Unix socket: {} (mode 0600)",
        socket_path
    );

    let semaphore = Arc::new(Semaphore::new(MAX_CONCURRENT_REQUESTS));

    loop {
        match listener.accept().await {
            Ok((mut stream, _addr)) => {
                let state = state.clone();
                let sem = semaphore.clone();
                tokio::spawn(async move {
                    let permit = match sem.clone().try_acquire_owned() {
                        Ok(p) => p,
                        Err(_) => {
                            warn!("⚠️ IPC concurrency limit reached ({MAX_CONCURRENT_REQUESTS}), waiting for slot");
                            match sem.acquire_owned().await {
                                Ok(p) => p,
                                Err(e) => {
                                    error!("❌ Semaphore closed: {e}");
                                    return;
                                }
                            }
                        }
                    };
                    let _permit = permit; // held until task completes
                    let mut buffer = Vec::new();
                    let mut chunk = vec![0; 65536];
                    loop {
                        match stream.read(&mut chunk).await {
                            Ok(n) if n > 0 => {
                                buffer.extend_from_slice(&chunk[..n]);
                                if buffer.len() > MAX_REQUEST_BYTES {
                                    error!(
                                        "❌ IPC request excede el tope ({} bytes), abortando conexión",
                                        MAX_REQUEST_BYTES
                                    );
                                    let err_msg = IpcResponse {
                                        status: "error".to_string(),
                                        data: serde_json::json!({
                                            "error": "request too large"
                                        }),
                                    };
                                    if let Ok(mut estr) = serde_json::to_string(&err_msg) {
                                        estr.push('\n');
                                        let _ = stream.write_all(estr.as_bytes()).await;
                                    }
                                    break;
                                }
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
        "embed" => handler_embed::handle_embed(request, state, stream).await,
        "recommended_variant" => {
            handler_argus::handle_recommended_variant(request, state, stream).await
        }
        "generate" => handler_generate::handle_generate(request, state, stream).await,
        "generate_stream" => handler_stream::handle_generate_stream(request, state, stream).await,
        "delegate_task" => handler_delegation::handle_delegate_task(request, state, stream).await,
        "list_agent_runs" => {
            handler_delegation::handle_list_agent_runs(request, state, stream).await
        }
        "get_agent_run" => handler_delegation::handle_get_agent_run(request, state, stream).await,
        "await_agent_run" => {
            handler_delegation::handle_await_agent_run(request, state, stream).await
        }
        "cancel_agent_run" => {
            handler_delegation::handle_cancel_agent_run(request, state, stream).await
        }
        "resume_agent_run" => {
            handler_delegation::handle_resume_agent_run(request, state, stream).await
        }
        "summarize_agent_run" => {
            handler_delegation::handle_summarize_agent_run(request, state, stream).await
        }
        "route_health" => handler_health::handle_route_health(request, state, stream).await,
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
