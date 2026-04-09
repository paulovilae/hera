use hera_core::ai::tool_executor::{ToolCall, execute_tool, hera_tool_schemas};
use rmcp::{
    ServerHandler, ServiceExt,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{ServerCapabilities, ServerInfo},
    schemars, tool, tool_handler, tool_router,
    transport::stdio,
};
use serde::Deserialize;
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;

const HERA_SOCKET: &str = "/tmp/hera-core.sock";

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct GenerateTextParams {
    #[schemars(description = "The user prompt to send to Hera")]
    prompt: String,
    #[schemars(description = "Optional path to an OS/Agents persona file")]
    #[serde(default)]
    persona_path: String,
    #[schemars(description = "Maximum output tokens")]
    #[serde(default)]
    max_tokens: Option<u32>,
    #[schemars(description = "Sampling temperature")]
    #[serde(default)]
    temperature: Option<f32>,
    #[schemars(description = "Allowed Hera tool permissions, e.g. ['all'] or ['memento_query']")]
    #[serde(default)]
    permissions: Vec<String>,
    #[schemars(description = "Optional app identifier for memory/schema context")]
    #[serde(default)]
    app: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct ExecuteToolParams {
    #[schemars(description = "Exact Hera tool name")]
    name: String,
    #[schemars(description = "JSON arguments object to pass to the tool")]
    #[serde(default)]
    arguments: serde_json::Value,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct MementoQueryParams {
    #[schemars(description = "App slug registered in Memento")]
    app: String,
    #[schemars(description = "SQL SELECT query")]
    query: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct SpawnParallelAgentsParams {
    #[schemars(description = "Agent persona filenames without .md")]
    agents: Vec<String>,
    #[schemars(description = "Prompt sent to each agent")]
    prompt: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct ListToolSchemasParams {
    #[schemars(description = "Agent identity used for app-specific schema resolution")]
    #[serde(default = "default_agent_name")]
    agent_name: String,
    #[schemars(description = "Allowed Hera permissions")]
    #[serde(default = "default_permissions")]
    permissions: Vec<String>,
}

fn default_agent_name() -> String {
    "external_agent".to_string()
}

fn default_permissions() -> Vec<String> {
    vec!["all".to_string()]
}

async fn send_ipc_generate(params: &GenerateTextParams) -> String {
    let payload = json!({
        "action": "generate",
        "payload": {
            "prompt": params.prompt,
            "persona_path": if params.persona_path.is_empty() { serde_json::Value::Null } else { json!(params.persona_path) },
            "max_tokens": params.max_tokens.unwrap_or(800),
            "temperature": params.temperature.unwrap_or(0.3),
            "permissions": if params.permissions.is_empty() { json!(["all"]) } else { json!(params.permissions) },
            "app": params.app
        }
    });

    match UnixStream::connect(HERA_SOCKET).await {
        Ok(mut stream) => {
            let msg_bytes = payload.to_string();
            if let Err(error) = stream.write_all(msg_bytes.as_bytes()).await {
                return format!("Error writing to Hera socket: {}", error);
            }
            if let Err(error) = stream.shutdown().await {
                return format!("Error shutting down Hera write side: {}", error);
            }

            let mut response = String::new();
            if let Err(error) = stream.read_to_string(&mut response).await {
                return format!("Error reading from Hera socket: {}", error);
            }

            match serde_json::from_str::<serde_json::Value>(&response) {
                Ok(parsed) => parsed
                    .pointer("/data/result")
                    .and_then(|value| value.as_str())
                    .map(|value| value.to_string())
                    .unwrap_or(response),
                Err(_) => response,
            }
        }
        Err(error) => format!(
            "Cannot connect to Hera daemon at {}. Is it running? Error: {}",
            HERA_SOCKET, error
        ),
    }
}

#[derive(Debug, Clone)]
struct HeraMcp {
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl HeraMcp {
    pub fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        description = "Send a text generation request to Hera via its Unix socket. This is the sovereign local-first generation entrypoint for external MCP clients."
    )]
    async fn generate_text(&self, Parameters(params): Parameters<GenerateTextParams>) -> String {
        send_ipc_generate(&params).await
    }

    #[tool(
        description = "Execute any existing Hera tool by exact name with a JSON arguments object. This reuses Hera's canonical tool executor rather than reimplementing tool logic."
    )]
    async fn execute_tool(&self, Parameters(params): Parameters<ExecuteToolParams>) -> String {
        let call = ToolCall {
            name: params.name,
            arguments: params.arguments,
        };
        let result = execute_tool(&call).await;
        result.output
    }

    #[tool(description = "Query a registered app database through Hera's memento_query tool.")]
    async fn memento_query(&self, Parameters(params): Parameters<MementoQueryParams>) -> String {
        let call = ToolCall {
            name: "memento_query".to_string(),
            arguments: json!({
                "app": params.app,
                "query": params.query,
            }),
        };
        let result = execute_tool(&call).await;
        result.output
    }

    #[tool(
        description = "Spawn multiple OS/Agents personas in parallel through Hera's canonical spawn_parallel_agents tool."
    )]
    async fn spawn_parallel_agents(
        &self,
        Parameters(params): Parameters<SpawnParallelAgentsParams>,
    ) -> String {
        let call = ToolCall {
            name: "spawn_parallel_agents".to_string(),
            arguments: json!({
                "agents": params.agents,
                "prompt": params.prompt,
            }),
        };
        let result = execute_tool(&call).await;
        result.output
    }

    #[tool(
        description = "Return the currently available Hera tool schemas for a given external agent identity and permission scope."
    )]
    async fn list_tool_schemas(
        &self,
        Parameters(params): Parameters<ListToolSchemasParams>,
    ) -> String {
        hera_tool_schemas(&params.permissions, &params.agent_name)
    }
}

#[tool_handler]
impl ServerHandler for HeraMcp {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions(
                "Hera MCP is the external adapter over the sovereign Hera runtime. \
                 Use generate_text() for local-first reasoning through /tmp/hera-core.sock. \
                 Use execute_tool() or the dedicated wrappers for canonical Hera tools such as memento_query and spawn_parallel_agents."
                    .to_string(),
            )
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .init();

    tracing::info!("Starting Hera MCP Bridge");

    let service = HeraMcp::new().serve(stdio()).await.inspect_err(|error| {
        tracing::error!("Hera MCP serving error: {:?}", error);
    })?;

    service.waiting().await?;
    Ok(())
}
