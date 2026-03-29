use std::collections::HashMap;
use std::process::Command;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug, Clone, Default)]
pub struct NodeData {
    pub label: Option<String>,
    #[serde(rename = "type")]
    pub node_type: Option<String>,
    pub value: Option<String>,
    pub language: Option<String>,
    pub code: Option<String>,
    #[serde(rename = "systemPrompt")]
    pub system_prompt: Option<String>,
    pub provider: Option<String>,
    #[serde(rename = "targetModel")]
    pub target_model: Option<String>,
}

#[derive(Deserialize, Serialize, Debug, Clone, Default)]
pub struct XYPosition {
    pub x: f64,
    pub y: f64,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct GraphNode {
    pub id: String,
    #[serde(rename = "type", default = "default_node_type")]
    pub node_type: String,
    #[serde(default)]
    pub position: XYPosition,
    #[serde(default)]
    pub data: NodeData,
}

fn default_node_type() -> String { "universalBlock".to_string() }

#[derive(Deserialize, Serialize, Debug, Clone, Default)]
pub struct EdgeStyle {
    pub stroke: Option<String>,
    #[serde(rename = "strokeWidth")]
    pub stroke_width: Option<f64>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct GraphEdge {
    pub id: String,
    pub source: String,
    pub target: String,
    #[serde(default)]
    pub animated: bool,
    #[serde(default)]
    pub style: Option<EdgeStyle>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct WorkflowRequest {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct WorkflowResponse {
    pub results: HashMap<String, String>,
    pub errors: HashMap<String, String>,
}

pub async fn execute_dag(req: WorkflowRequest) -> WorkflowResponse {
    let mut results = HashMap::new();
    let mut errors = HashMap::new();
    let mut memory_vault: HashMap<String, String> = HashMap::new();
    
    // Simple topological sort approach or just linear execution of input -> next
    // For now, since it's a simple chain often, let's find the input node.
    let input_node = req.nodes.iter().find(|n| n.data.node_type.as_deref() == Some("input"));
    
    let mut current_id = match input_node {
        Some(node) => node.id.clone(),
        None => {
            if req.nodes.len() == 1 {
                req.nodes[0].id.clone() // Isolated Execution Mode (Playground)
            } else {
                errors.insert("global".to_string(), "No Workflow Input block found.".to_string());
                return WorkflowResponse { results, errors };
            }
        }
    };
    
    let mut current_payload = match input_node {
        Some(node) => node.data.value.clone().unwrap_or_default(),
        None => "{}".to_string()
    };
    
    let mut adjacency = HashMap::new();
    for edge in &req.edges {
        adjacency.insert(edge.source.clone(), edge.target.clone());
    }
    
    let node_map: HashMap<String, &GraphNode> = req.nodes.iter().map(|n| (n.id.clone(), n)).collect();

    loop {
        // Execute current node if needed
        let node = match node_map.get(&current_id) {
            Some(n) => n,
            None => break,
        };
        
        // Skip execution for input node as it's just passing its value as initial payload
        if node.data.node_type.as_deref() != Some("input") {
            // Check if current payload is a reference and resolve it
            let resolved_payload = if current_payload.starts_with("[Ref: ") && current_payload.ends_with("]") {
                // Check local execution memory vault
                if let Some(content) = memory_vault.get(&current_payload) {
                    content.clone()
                } else {
                    // Fallback simulated resolution if not found in current run
                    format!("(Dereferenced Content from {})", current_payload)
                }
            } else {
                current_payload.clone()
            };

            // Check type and execute
            match node.data.node_type.as_deref() {
                Some("code") => {
                    let lang = node.data.language.as_deref().unwrap_or("python");
                    let code = node.data.code.as_deref().unwrap_or("");
                    let uuid_str = uuid::Uuid::new_v4().to_string();
                    println!("executing code block for {} with lang: {} and code len: {}", uuid_str, lang, code.len());
                    println!("CODE CONTENT:\n{}", code);
                    
                    let output_res = if lang == "rust" {
                        // Dynamically compile and run Rust using the pre-warmed Cargo sandbox
                        let sandbox_target = format!("/tmp/sandbox_{}", uuid_str);
                        
                        // Fast clone the cached sandbox (preserves compiled dependencies)
                        let _ = Command::new("cp")
                            .arg("-a")
                            .arg("/tmp/hera_sandbox")
                            .arg(&sandbox_target)
                            .output();
                            
                        let main_rs = format!("{}/src/main.rs", sandbox_target);
                        std::fs::write(&main_rs, code).unwrap_or_default();
                        
                        let run_status = Command::new("cargo")
                            .arg("run")
                            .arg("--quiet")
                            .arg("--manifest-path")
                            .arg(format!("{}/Cargo.toml", sandbox_target))
                            .env("WORKFLOW_INPUT", &resolved_payload)
                            .output();
                            
                        // Cleanup
                        let _ = std::fs::remove_dir_all(&sandbox_target);
                        
                        match run_status {
                            Ok(run_out) if run_out.status.success() => {
                                String::from_utf8_lossy(&run_out.stdout).to_string()
                            },
                            Ok(run_out) => {
                                format!("Cargo Error:\n{}", String::from_utf8_lossy(&run_out.stderr))
                            },
                            Err(e) => format!("Failed to invoke cargo: {}", e)
                        }
                    } else {
                        // Python
                        let py_path = format!("/tmp/dynamic_node_{}.py", uuid_str);
                        std::fs::write(&py_path, code).unwrap_or_default();
                        
                        let run_status = Command::new("python3")
                            .arg(&py_path)
                            .env("WORKFLOW_INPUT", &resolved_payload)
                            .output();
                        match run_status {
                            Ok(run_out) if run_out.status.success() => String::from_utf8_lossy(&run_out.stdout).to_string(),
                            Ok(run_out) => format!("Python error: {}", String::from_utf8_lossy(&run_out.stderr)),
                            Err(e) => format!("Failed to invoke python3: {}", e)
                        }
                    };
                    
                    current_payload = output_res.clone();
                    results.insert(node.id.clone(), output_res);
                },
                Some("llm") => {
                    #[cfg(feature = "web")]
                    let out = {
                        let mut payload_map = serde_json::Map::new();

                        let model_opt = node
                            .data
                            .value
                            .clone()
                            .unwrap_or_else(|| "nvidia/nemotron-3-nano-30b-a3b:free".to_string());

                        payload_map.insert("model".to_string(), serde_json::Value::String(model_opt));
                        payload_map.insert(
                            "provider".to_string(),
                            serde_json::Value::String("local".to_string()),
                        );

                        let mut messages = Vec::new();
                        if let Some(sys) = &node.data.system_prompt {
                            if !sys.is_empty() {
                                let mut sys_msg = serde_json::Map::new();
                                sys_msg.insert(
                                    "role".to_string(),
                                    serde_json::Value::String("system".to_string()),
                                );
                                sys_msg.insert(
                                    "content".to_string(),
                                    serde_json::Value::String(sys.clone()),
                                );
                                messages.push(serde_json::Value::Object(sys_msg));
                            }
                        }

                        let mut usr_msg = serde_json::Map::new();
                        usr_msg.insert(
                            "role".to_string(),
                            serde_json::Value::String("user".to_string()),
                        );
                        usr_msg.insert(
                            "content".to_string(),
                            serde_json::Value::String(resolved_payload.clone()),
                        );
                        messages.push(serde_json::Value::Object(usr_msg));

                        payload_map.insert("messages".to_string(), serde_json::Value::Array(messages));
                        payload_map.insert("temperature".to_string(), serde_json::json!(0.7));
                        payload_map.insert("max_tokens".to_string(), serde_json::json!(1024));

                        let client = reqwest::Client::new();
                        match client
                            .post("http://127.0.0.1:3005/v1/chat/completions")
                            .json(&payload_map)
                            .send()
                            .await
                        {
                            Ok(res) if res.status().is_success() => match res.json::<serde_json::Value>().await {
                                Ok(json) => json["choices"][0]["message"]["content"]
                                    .as_str()
                                    .unwrap_or("Empty response")
                                    .to_string(),
                                Err(e) => format!("Failed to parse response: {}", e),
                            },
                            Ok(res) => format!("LLM API Error: {}", res.status()),
                            Err(e) => format!("Network Error: {}", e),
                        }
                    };

                    #[cfg(not(feature = "web"))]
                    let out = "The `web` feature is disabled, so workflow LLM HTTP execution is unavailable.".to_string();

                    current_payload = out.clone();
                    results.insert(node.id.clone(), out);
                },
                Some("memory") => {
                    // The memory node takes the large payload, saves it, and outputs a coordinate
                    let mem_uuid = uuid::Uuid::new_v4().to_string();
                    let out = format!("[Ref: memory_vault_{}]", mem_uuid);
                    // Store the actual heavy payload in the vault mapped to the reference
                    memory_vault.insert(out.clone(), current_payload.clone());
                    current_payload = out.clone();
                    results.insert(node.id.clone(), out);
                },
                Some("market") => {
                    #[cfg(feature = "market")]
                    let out = {
                        let symbol = node.data.value.clone().unwrap_or_else(|| "AAPL".to_string());
                        hera_market::quote_range_json(&symbol, "1d", "1mo").await
                    };

                    #[cfg(not(feature = "market"))]
                    let out =
                        "The `market` feature is disabled, so market data execution is unavailable."
                            .to_string();

                    current_payload = out.clone();
                    results.insert(node.id.clone(), out);
                },
                Some("broker") => {
                    let provider = node.data.provider.clone().unwrap_or_else(|| "paper".to_string());
                    let symbol = node.data.value.clone().unwrap_or_else(|| "AAPL".to_string());
                    let action = node.data.system_prompt.clone().unwrap_or_else(|| "BUY".to_string());
                    let qty = node.data.target_model.clone().unwrap_or_else(|| "10".to_string());
                    
                    let out = serde_json::json!({
                        "status": "simulated_success",
                        "broker": provider,
                        "symbol": symbol,
                        "action": action,
                        "quantity": qty,
                        "details": "Order executed via simulated MCP execution layer bridge."
                    }).to_string();
                    
                    current_payload = out.clone();
                    results.insert(node.id.clone(), out);
                },
                Some("mcp") => {
                    let out = format!("Simulated MCP execution. Payload: {}", resolved_payload);
                    current_payload = out.clone();
                    results.insert(node.id.clone(), out);
                },
                Some("scrape") => {
                    #[cfg(all(feature = "web", feature = "agents"))]
                    let out = {
                        let mcp_url = std::env::var("HERA_MCP_URL")
                            .unwrap_or_else(|_| "http://127.0.0.1:8080".to_string());
                        let hera = crate::agents::hera::Hera::new(&mcp_url);
                        let url_to_scrape = resolved_payload.trim();
                        match hera.native_web_scrape(url_to_scrape).await {
                            Ok(text) => text,
                            Err(e) => format!("Scrape Error: {}", e),
                        }
                    };

                    #[cfg(not(all(feature = "web", feature = "agents")))]
                    let out =
                        "The `web` and `agents` features are required for scrape nodes."
                            .to_string();

                    current_payload = out.clone();
                    results.insert(node.id.clone(), out);
                },
                Some("search") => {
                    #[cfg(all(feature = "web", feature = "agents"))]
                    let out = {
                        let mcp_url = std::env::var("HERA_MCP_URL")
                            .unwrap_or_else(|_| "http://127.0.0.1:8080".to_string());
                        let hera = crate::agents::hera::Hera::new(&mcp_url);
                        let query = resolved_payload.trim();
                        match hera.native_web_search(query).await {
                            Ok(text) => text,
                            Err(e) => format!("Search Error: {}", e),
                        }
                    };

                    #[cfg(not(all(feature = "web", feature = "agents")))]
                    let out =
                        "The `web` and `agents` features are required for search nodes."
                            .to_string();

                    current_payload = out.clone();
                    results.insert(node.id.clone(), out);
                },
                _ => {}
            }
        }
        
        // Move to next node
        if let Some(next_id) = adjacency.get(&current_id) {
            current_id = next_id.clone();
        } else {
            break;
        }
    }
    
    WorkflowResponse { results, errors }
}
