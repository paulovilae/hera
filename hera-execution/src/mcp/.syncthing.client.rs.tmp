use std::sync::Arc;
use reqwest::Client;
use serde_json::json;
use thiserror::Error;

use super::types::*;

#[derive(Error, Debug)]
pub enum McpError {
    #[error("Network Error: {0}")]
    Reqwest(#[from] reqwest::Error),
    #[error("Parse Error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("RPC Error [{0}]: {1}")]
    RpcError(i64, String),
    #[error("Missing Result")]
    MissingResult,
}

pub struct McpHttpClient {
    endpoint: String,
    client: Arc<Client>,
}

impl McpHttpClient {
    pub fn new(endpoint: &str) -> Self {
        Self {
            endpoint: endpoint.to_string(),
            client: Arc::new(Client::new()),
        }
    }

    async fn send_request(&self, method: &str, params: serde_json::Value) -> Result<serde_json::Value, McpError> {
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(uuid::Uuid::new_v4().to_string()),
            method: method.to_string(),
            params: Some(params),
        };

        let response = self.client.post(&self.endpoint)
            .json(&req)
            .send()
            .await?;

        let rpc_res: JsonRpcResponse = response.json().await?;

        if let Some(err) = rpc_res.error {
            return Err(McpError::RpcError(err.code, err.message));
        }

        rpc_res.result.ok_or(McpError::MissingResult)
    }

    pub async fn initialize(&self) -> Result<InitializeResult, McpError> {
        let params = json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "roots": { "listChanged": false }
            },
            "clientInfo": {
                "name": "Hera Engine",
                "version": "1.0.0"
            }
        });

        let result = self.send_request("initialize", params).await?;
        
        // Notify initialization
        let _ = self.client.post(&self.endpoint)
            .json(&JsonRpcRequest {
                jsonrpc: "2.0".to_string(),
                id: None,
                method: "notifications/initialized".to_string(),
                params: None,
            })
            .send()
            .await;

        let init_res: InitializeResult = serde_json::from_value(result)?;
        Ok(init_res)
    }

    pub async fn list_tools(&self) -> Result<ListToolsResult, McpError> {
        let result = self.send_request("tools/list", json!({})).await?;
        let list_res: ListToolsResult = serde_json::from_value(result)?;
        Ok(list_res)
    }

    pub async fn call_tool(&self, name: &str, arguments: serde_json::Value) -> Result<CallToolResult, McpError> {
        let params = json!({
            "name": name,
            "arguments": arguments
        });

        let result = self.send_request("tools/call", params).await?;
        let call_res: CallToolResult = serde_json::from_value(result)?;
        Ok(call_res)
    }
}
