use anyhow::{Context, Result};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;

use crate::protocol::{DiakonosRequest, DiakonosResponse};

pub const DIAKONOS_SOCKET: &str = "/tmp/diakonos.sock";

pub async fn send_request(
    socket_path: &str,
    request: &DiakonosRequest,
) -> Result<DiakonosResponse> {
    let mut stream = UnixStream::connect(socket_path)
        .await
        .with_context(|| format!("Failed to connect to {}", socket_path))?;

    let serialized = serde_json::to_vec(request)?;
    stream.write_all(&serialized).await?;
    stream.shutdown().await?;

    let mut buffer = Vec::new();
    stream.read_to_end(&mut buffer).await?;
    let response: DiakonosResponse = serde_json::from_slice(&buffer)?;
    Ok(response)
}
