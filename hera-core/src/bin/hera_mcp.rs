use hera_core::ai::tool_executor::{ToolCall, execute_tool, hera_tool_schemas, tool_is_critical};
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

/// Gate for the 6 coding agentic-loop tools (edit_file, write_file, grep_search, glob_search,
/// cargo_check, cargo_test) exposed over MCP. OFF by default — an external MCP client gets
/// filesystem write + arbitrary cargo execution on this box the moment it can reach this
/// bridge, so these tools (and any Critical-risk tool reachable via the generic
/// `execute_tool` passthrough below) require an explicit opt-in per host.
fn coding_tools_enabled() -> bool {
    std::env::var("HERA_MCP_CODING_TOOLS").ok().as_deref() == Some("1")
}

const CODING_TOOLS_DISABLED_MSG: &str =
    "Refused: coding tools are disabled on this Hera MCP bridge. Set HERA_MCP_CODING_TOOLS=1 \
     in the environment running hera_mcp to enable edit_file/write_file/grep_search/\
     glob_search/cargo_check/cargo_test (directly or via execute_tool for any Critical-risk \
     tool).";

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
struct EditFileParams {
    #[schemars(description = "Absolute or relative path to the existing file to edit")]
    path: String,
    #[schemars(
        description = "Exact text to replace, copied verbatim from the file (include enough surrounding context to be unique)"
    )]
    old_string: String,
    #[schemars(description = "Replacement text")]
    new_string: String,
    #[schemars(
        description = "Replace every occurrence of old_string instead of requiring a single unique match. Defaults to false."
    )]
    #[serde(default)]
    replace_all: Option<bool>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct WriteFileParams {
    #[schemars(description = "Absolute or relative path to the file to create or overwrite")]
    path: String,
    #[schemars(description = "Full text content to write into the file")]
    content: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct GrepSearchParams {
    #[schemars(description = "A Rust regular expression to match against each line")]
    pattern: String,
    #[schemars(description = "Optional directory or file to search under. Defaults to the current directory.")]
    #[serde(default)]
    path: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct GlobSearchParams {
    #[schemars(description = "Glob pattern, e.g. '**/*.rs', 'src/*.toml', '**/handler_*.rs'")]
    pattern: String,
    #[schemars(description = "Optional base directory to search under. Defaults to the current directory.")]
    #[serde(default)]
    path: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct CargoCheckParams {
    #[schemars(description = "REQUIRED. Absolute path to the Rust project directory (where Cargo.toml lives)")]
    path: String,
    #[schemars(description = "Max seconds to wait for the build. Defaults to 180, max 600.")]
    #[serde(default)]
    timeout_seconds: Option<u64>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct CargoTestParams {
    #[schemars(description = "REQUIRED. Absolute path to the Rust project directory (where Cargo.toml lives)")]
    path: String,
    #[schemars(description = "Max seconds to wait. Defaults to 180, max 600.")]
    #[serde(default)]
    timeout_seconds: Option<u64>,
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

/// True when this request targets Ava's dedicated coding surface — i.e. the caller
/// asked for `route_profile="coding"` (or set `caller`/`app` to "coding"/"ava_coder").
/// The 6 surgical coding tools live behind that surface and 4 of them
/// (edit_file/write_file/cargo_check/cargo_test) are Critical risk.
fn is_coding_surface(params: &GenerateTextParams) -> bool {
    let is_coding = |value: &str| matches!(value.trim(), "coding" | "ava_coder");
    params
        .route_profile
        .as_deref()
        .map(is_coding)
        .unwrap_or(false)
        || params.caller.as_deref().map(is_coding).unwrap_or(false)
        || is_coding(&params.app)
}

async fn send_ipc_generate(params: &GenerateTextParams) -> String {
    // Base permission set requested by the MCP client.
    let mut perms: Vec<String> = if params.permissions.is_empty() {
        vec!["all".to_string()]
    } else {
        params.permissions.clone()
    };

    // Coding surface: mirror `bin/claude.rs --coding`. The `all` wildcard grants
    // Low/High-risk tools but NOT Critical ones, and the 6 coding tools include 4
    // Critical (edit_file/write_file/cargo_check/cargo_test). Without `unsafe_all`
    // the agentic loop offers the tools but execution is denied → the model reports
    // "permission restrictions" and never completes (the observed Bug 2). Grant the
    // explicit coding tools + `unsafe_all` so `route_profile="coding"` works out of
    // the box from any MCP client, not just the SSH `claude --coding` path. Only
    // fires when the caller explicitly opted into the coding surface (no regression
    // for generic requests).
    if is_coding_surface(params) {
        for tool in [
            "read_file",
            "write_file",
            "edit_file",
            "grep_search",
            "glob_search",
            "cargo_check",
            "cargo_test",
            "unsafe_all",
        ] {
            if !perms.iter().any(|existing| existing == tool) {
                perms.push(tool.to_string());
            }
        }
    }
    let permissions = json!(perms);

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
    } else if is_coding_surface(params) {
        // Coding surface without an explicit caller: default it so the coding tools'
        // `allowed_callers` gating and `is_coding_context` resolve to the coding
        // persona (mirrors `bin/claude.rs` setting `app="coding"`).
        inner["caller"] = json!("coding");
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
        // Retroactive gate: this generic passthrough can already reach the 6 coding tools
        // (and any other Critical-risk tool) today — every Critical tool JSON lists
        // "external_mcp" in allowed_callers, and this bridge defaults the caller to exactly
        // that, so `caller_allowed_for_tool` trivially passes. Without this check, adding
        // typed wrappers below with their own gate would be security theater: the same
        // capability stays reachable unguarded through this method. Non-Critical tools
        // (e.g. memento_query, grep_search) are unaffected.
        if tool_is_critical(&params.name) && !coding_tools_enabled() {
            return CODING_TOOLS_DISABLED_MSG.to_string();
        }
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
        description = "Surgically edit an existing file by replacing an exact block of text (Hera's edit_file tool). DISABLED unless HERA_MCP_CODING_TOOLS=1 is set on this bridge."
    )]
    async fn edit_file(&self, Parameters(params): Parameters<EditFileParams>) -> String {
        if !coding_tools_enabled() {
            return CODING_TOOLS_DISABLED_MSG.to_string();
        }
        let call = ToolCall {
            name: "edit_file".to_string(),
            arguments: json!({
                "path": params.path,
                "old_string": params.old_string,
                "new_string": params.new_string,
                "replace_all": params.replace_all.unwrap_or(false),
                "_hera": {"caller": "external_mcp"},
            }),
        };
        execute_tool(&call).await.output
    }

    #[tool(
        description = "Create a new file or completely overwrite an existing file (Hera's write_file tool). DISABLED unless HERA_MCP_CODING_TOOLS=1 is set on this bridge."
    )]
    async fn write_file(&self, Parameters(params): Parameters<WriteFileParams>) -> String {
        if !coding_tools_enabled() {
            return CODING_TOOLS_DISABLED_MSG.to_string();
        }
        let call = ToolCall {
            name: "write_file".to_string(),
            arguments: json!({
                "path": params.path,
                "content": params.content,
                "_hera": {"caller": "external_mcp"},
            }),
        };
        execute_tool(&call).await.output
    }

    #[tool(
        description = "Search the file tree for lines matching a regular expression (Hera's grep_search tool). DISABLED unless HERA_MCP_CODING_TOOLS=1 is set on this bridge."
    )]
    async fn grep_search(&self, Parameters(params): Parameters<GrepSearchParams>) -> String {
        if !coding_tools_enabled() {
            return CODING_TOOLS_DISABLED_MSG.to_string();
        }
        let call = ToolCall {
            name: "grep_search".to_string(),
            arguments: json!({
                "pattern": params.pattern,
                "path": params.path.unwrap_or_default(),
                "_hera": {"caller": "external_mcp"},
            }),
        };
        execute_tool(&call).await.output
    }

    #[tool(
        description = "Find files whose path matches a glob pattern (Hera's glob_search tool). DISABLED unless HERA_MCP_CODING_TOOLS=1 is set on this bridge."
    )]
    async fn glob_search(&self, Parameters(params): Parameters<GlobSearchParams>) -> String {
        if !coding_tools_enabled() {
            return CODING_TOOLS_DISABLED_MSG.to_string();
        }
        let call = ToolCall {
            name: "glob_search".to_string(),
            arguments: json!({
                "pattern": params.pattern,
                "path": params.path.unwrap_or_default(),
                "_hera": {"caller": "external_mcp"},
            }),
        };
        execute_tool(&call).await.output
    }

    #[tool(
        description = "Run 'cargo check' on a Rust project and return structured compiler errors (Hera's cargo_check tool). DISABLED unless HERA_MCP_CODING_TOOLS=1 is set on this bridge."
    )]
    async fn cargo_check(&self, Parameters(params): Parameters<CargoCheckParams>) -> String {
        if !coding_tools_enabled() {
            return CODING_TOOLS_DISABLED_MSG.to_string();
        }
        let mut arguments = json!({
            "path": params.path,
            "_hera": {"caller": "external_mcp"},
        });
        if let Some(timeout_seconds) = params.timeout_seconds {
            arguments["timeout_seconds"] = json!(timeout_seconds);
        }
        let call = ToolCall {
            name: "cargo_check".to_string(),
            arguments,
        };
        execute_tool(&call).await.output
    }

    #[tool(
        description = "Run 'cargo test' on a Rust project and return a pass/fail summary (Hera's cargo_test tool). DISABLED unless HERA_MCP_CODING_TOOLS=1 is set on this bridge."
    )]
    async fn cargo_test(&self, Parameters(params): Parameters<CargoTestParams>) -> String {
        if !coding_tools_enabled() {
            return CODING_TOOLS_DISABLED_MSG.to_string();
        }
        let mut arguments = json!({
            "path": params.path,
            "_hera": {"caller": "external_mcp"},
        });
        if let Some(timeout_seconds) = params.timeout_seconds {
            arguments["timeout_seconds"] = json!(timeout_seconds);
        }
        let call = ToolCall {
            name: "cargo_test".to_string(),
            arguments,
        };
        execute_tool(&call).await.output
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
