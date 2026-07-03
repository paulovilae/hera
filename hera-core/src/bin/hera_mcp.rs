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

/// Override with `HERA_SOCKET_PATH` to point this MCP bridge at a forwarded socket
/// (e.g. an SSH-tunneled path to a remote node's real hera-core) instead of a local
/// daemon — useful when the local machine only runs a lightweight fallback instance.
fn hera_socket_path() -> String {
    std::env::var("HERA_SOCKET_PATH").unwrap_or_else(|_| "/tmp/hera-core.sock".to_string())
}

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
    #[schemars(
        description = "Optional Hera route profile id (e.g. \"coding\" to engage Ava Coder's agentic loop + 6 coding tools, or \"ops\"). Default None keeps generic routing."
    )]
    #[serde(default)]
    route_profile: Option<String>,
    #[schemars(
        description = "Optional caller identity used for tool allowed_callers gating (e.g. \"ava_coder\" / \"coding\"). Default None falls back to app/hera."
    )]
    #[serde(default)]
    caller: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct ExecuteToolParams {
    #[schemars(description = "Exact Hera tool name")]
    name: String,
    // HashMap forces schemars to emit type:object so MCP clients serialize correctly.
    #[schemars(description = "Arguments to pass to the tool (key-value pairs)")]
    #[serde(default)]
    arguments: std::collections::HashMap<String, serde_json::Value>,
    #[schemars(
        description = "Optional caller identity used for tool allowed_callers gating (e.g. \"coding\" / \"ava_coder\"). Default None applies caller=\"external_mcp\"."
    )]
    #[serde(default)]
    caller: Option<String>,
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

/// Patterns in the user prompt that signal an intent to get JSON output via text directive.
/// These trigger tool-call hijacking in the model (it interprets the JSON-shaped instruction
/// as a tool use signal). Fix: engage `response_format: json_object` instead and warn.
fn has_json_directive(prompt: &str) -> bool {
    let p = prompt.to_lowercase();
    p.contains("solo con json")
        || p.contains("only json")
        || p.contains("formato json")
        || p.contains("in json format")
        || (p.contains("responde") && p.contains("json") && p.contains("solo"))
        || (p.contains("respond") && p.contains("json") && p.contains("only"))
}

async fn send_ipc_generate(params: &GenerateTextParams) -> String {
    let permissions = if params.permissions.is_empty() {
        json!(["all"])
    } else {
        json!(params.permissions)
    };
    let mut inner = json!({
        "prompt": params.prompt,
        "persona_path": if params.persona_path.is_empty() { serde_json::Value::Null } else { json!(params.persona_path) },
        "max_tokens": params.max_tokens.unwrap_or(800),
        "temperature": params.temperature.unwrap_or(0.3),
        "permissions": permissions,
        "app": params.app,
        // MCP callers explicitly request tools via permissions; prevent lightweight-mode
        // detection from silently disabling them for short prompts (e.g. "hola").
        "context_budget_mode": "standard",
    });

    // JSON trap guard: text directives like "Responde SOLO con JSON" cause the model
    // to interpret the response as a tool call (hijack). Engage native JSON mode via
    // response_format instead — the engine honors it without hijacking.
    // Prefer structured text output: "Lista numerada, cada ítem: CAMPO: X | CAMPO: Y"
    if has_json_directive(&params.prompt) {
        tracing::warn!(
            "JSON directive detected in prompt — auto-enabling response_format:json_object \
             to prevent tool hijacking. Prefer structured text output next time: \
             'Lista numerada, cada ítem en una línea: CAMPO1: X | CAMPO2: Y'"
        );
        inner["response_format"] = json!({"type": "json_object"});
    }

    // Thread the route profile through so an MCP client can request the dedicated
    // coding surface ("coding"): that profile maps to the heavy context budget AND
    // is what `is_coding_context` keys off, which engages the agentic loop + low
    // deterministic temperature + the 6 coding tools. Default None = no key sent =
    // current generic behaviour (no regression).
    if let Some(route_profile) = params
        .route_profile
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        inner["route_profile"] = json!(route_profile);
    }

    // Optional explicit caller for tool `allowed_callers` gating. The execution-time
    // caller otherwise derives from `app` (or "hera" when app is empty); set this to
    // "ava_coder"/"coding" if the app field can't be used. Hera's `contextualize_tool_call`
    // only fills `caller` when absent, so an explicit value here wins.
    if let Some(caller) = params
        .caller
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        inner["caller"] = json!(caller);
    }

    let payload = json!({
        "action": "generate",
        "payload": inner,
    });

    let socket_path = hera_socket_path();
    match UnixStream::connect(&socket_path).await {
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
            socket_path, error
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
        let raw = serde_json::Value::Object(params.arguments.into_iter().collect());
        // Unwrap double-serialized args: some MCP clients (or intermediate layers) encode the
        // arguments object as a JSON string. When the resulting object has exactly one entry
        // whose VALUE is a JSON string that itself parses as an object, peel that outer layer.
        let arguments = if let serde_json::Value::Object(ref map) = raw {
            if map.len() == 1 {
                if let Some((_, serde_json::Value::String(s))) = map.iter().next() {
                    serde_json::from_str::<serde_json::Value>(s)
                        .ok()
                        .filter(|v| v.is_object())
                        .unwrap_or(raw.clone())
                } else {
                    raw
                }
            } else {
                raw
            }
        } else {
            raw
        };
        // Inject caller into arguments._hera.caller so tool_executor::security
        // can gate allowed_callers. Default "external_mcp" for MCP clients.
        let caller = params
            .caller
            .filter(|c| !c.trim().is_empty())
            .unwrap_or_else(|| "external_mcp".to_string());
        let arguments = match arguments {
            serde_json::Value::Object(mut map) => {
                let hera_entry = map
                    .entry("_hera".to_string())
                    .or_insert_with(|| json!({}));
                if let serde_json::Value::Object(hera_map) = hera_entry {
                    hera_map
                        .entry("caller".to_string())
                        .or_insert_with(|| json!(caller));
                }
                serde_json::Value::Object(map)
            }
            other => other,
        };
        let call = ToolCall {
            name: params.name,
            arguments,
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
