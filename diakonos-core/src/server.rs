use anyhow::Result;
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixListener;
use tracing::{error, info};

use crate::handlers;
use crate::modules::{action_module, ModuleCategory, ModuleRegistry};
use crate::protocol::{DiakonosRequest, DiakonosResponse};

pub async fn serve(socket_path: &str, modules: ModuleRegistry) -> std::io::Result<()> {
    if std::path::Path::new(socket_path).exists() {
        std::fs::remove_file(socket_path)?;
    }

    let listener = UnixListener::bind(socket_path)?;
    info!("Diakonos IPC bound to {}", socket_path);

    loop {
        match listener.accept().await {
            Ok((mut stream, _)) => {
                let modules = modules.clone();
                tokio::spawn(async move {
                    let mut buffer = Vec::new();
                    let mut chunk = vec![0; 8192];

                    loop {
                        match stream.read(&mut chunk).await {
                            Ok(n) if n > 0 => {
                                buffer.extend_from_slice(&chunk[..n]);
                                match serde_json::from_slice::<DiakonosRequest>(&buffer) {
                                    Ok(request) => {
                                        let response = handle_request(request, modules.clone()).await;
                                        match serde_json::to_vec(&response) {
                                            Ok(serialized) => {
                                                if let Err(err) = stream.write_all(&serialized).await {
                                                    error!("Diakonos write error: {}", err);
                                                }
                                            }
                                            Err(err) => {
                                                error!("Diakonos serialization error: {}", err);
                                            }
                                        }
                                        break;
                                    }
                                    Err(parse_err) if parse_err.is_eof() => {
                                        continue;
                                    }
                                    Err(err) => {
                                        let response = DiakonosResponse {
                                            status: "error".to_string(),
                                            data: json!({ "message": format!("Invalid request: {}", err) }),
                                        };
                                        match serde_json::to_vec(&response) {
                                            Ok(serialized) => {
                                                let _ = stream.write_all(&serialized).await;
                                            }
                                            Err(ser_err) => {
                                                error!("Diakonos serialization error: {}", ser_err);
                                            }
                                        }
                                        break;
                                    }
                                }
                            }
                            Ok(_) => {
                                break;
                            }
                            Err(err) => {
                                error!("Diakonos read error: {}", err);
                                break;
                            }
                        }
                    }
                });
            }
            Err(err) => {
                error!("Diakonos accept error: {}", err);
            }
        }
    }
}

async fn handle_request(request: DiakonosRequest, modules: ModuleRegistry) -> DiakonosResponse {
    match dispatch(request, modules).await {
        Ok(data) => DiakonosResponse {
            status: "success".to_string(),
            data,
        },
        Err(err) => DiakonosResponse {
            status: "error".to_string(),
            data: json!({ "message": err.to_string() }),
        },
    }
}

async fn dispatch(request: DiakonosRequest, modules: ModuleRegistry) -> Result<serde_json::Value> {
    if let Some(category) = action_module(&request.action) {
        if !modules.is_enabled(category).await {
            return Err(anyhow::anyhow!(
                "Module `{}` is disabled. Update {} and wait for auto-reload or call `reload_modules`.",
                category.as_str(),
                modules.config_path().display()
            ));
        }
    }

    match action_module(&request.action) {
        Some(ModuleCategory::Core) | Some(ModuleCategory::Docs) => {
            handlers::core::dispatch(&request.action, request.payload, &modules).await
        }
        Some(ModuleCategory::Workflow) => {
            handlers::workflow::dispatch(&request.action, request.payload).await
        }
        Some(ModuleCategory::Web) => {
            handlers::web::dispatch(&request.action, request.payload).await
        }
        Some(ModuleCategory::Media) => {
            handlers::media::dispatch(&request.action, request.payload).await
        }
        Some(ModuleCategory::Market) | Some(ModuleCategory::Vector) => {
            Err(anyhow::anyhow!("Action `{}` is not mapped yet", request.action))
        }
        None => Err(anyhow::anyhow!("Unsupported action `{}`", request.action)),
    }
}
