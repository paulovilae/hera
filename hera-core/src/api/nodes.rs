use axum::Json;
use serde::Serialize;
use std::fs;
use std::path::PathBuf;

#[derive(Serialize)]
pub struct RegistryNode {
    pub id: String,
    pub label: String,
    pub description: String,
    pub icon: String,
    pub color_theme: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workflow_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workspace_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r#type: Option<String>, // 'workflow' or None
}

#[derive(Serialize)]
pub struct NodeCategory {
    pub name: String,
    pub nodes: Vec<RegistryNode>,
}

fn get_base_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/paulo".to_string());
    let mut path = PathBuf::from(home);
    path.push(".hera");
    path.push("workspaces");
    path
}

pub async fn list_nodes() -> Json<Vec<NodeCategory>> {
    let mut categories = vec![
        NodeCategory {
            name: "Intelligence".to_string(),
            nodes: vec![
                RegistryNode {
                    id: "agent".to_string(),
                    label: "Sovereign Agent".to_string(),
                    description: "Vision, Audio & MCP".to_string(),
                    icon: "🤖".to_string(),
                    color_theme: "cyan".to_string(),
                    workflow_id: None,
                    workspace_id: None,
                    r#type: None,
                },
                RegistryNode {
                    id: "llm".to_string(),
                    label: "Neural Core".to_string(),
                    description: "LLM Inference Node".to_string(),
                    icon: "🧠".to_string(),
                    color_theme: "blue".to_string(),
                    workflow_id: None,
                    workspace_id: None,
                    r#type: None,
                },
                RegistryNode {
                    id: "memory".to_string(),
                    label: "Semantic Layer".to_string(),
                    description: "Fast Vector Storage".to_string(),
                    icon: "📚".to_string(),
                    color_theme: "purple".to_string(),
                    workflow_id: None,
                    workspace_id: None,
                    r#type: None,
                },
            ],
        },
        NodeCategory {
            name: "Execution".to_string(),
            nodes: vec![
                RegistryNode {
                    id: "mcp".to_string(),
                    label: "Execution Block".to_string(),
                    description: "MCP Bound Router".to_string(),
                    icon: "🛠️".to_string(),
                    color_theme: "emerald".to_string(),
                    workflow_id: None,
                    workspace_id: None,
                    r#type: None,
                },
                RegistryNode {
                    id: "code".to_string(),
                    label: "Code Block".to_string(),
                    description: "Native RS / PY".to_string(),
                    icon: "⚡".to_string(),
                    color_theme: "red".to_string(),
                    workflow_id: None,
                    workspace_id: None,
                    r#type: None,
                },
            ],
        },
        NodeCategory {
            name: "I/O & Triggers".to_string(),
            nodes: vec![
                RegistryNode {
                    id: "input".to_string(),
                    label: "Text Input".to_string(),
                    description: "User / API Input".to_string(),
                    icon: "⌨️".to_string(),
                    color_theme: "slate".to_string(),
                    workflow_id: None,
                    workspace_id: None,
                    r#type: None,
                },
                RegistryNode {
                    id: "output".to_string(),
                    label: "Text Output".to_string(),
                    description: "Final Response".to_string(),
                    icon: "📄".to_string(),
                    color_theme: "slate".to_string(),
                    workflow_id: None,
                    workspace_id: None,
                    r#type: None,
                },
            ],
        },
    ];

    // Read workspaces dynamically
    let base_dir = get_base_dir();
    if let Ok(entries) = fs::read_dir(&base_dir) {
        for entry in entries.flatten() {
            if entry.path().is_dir() {
                if let Some(ws_name) = entry.file_name().to_str() {
                    let mut ws_path = base_dir.clone();
                    ws_path.push(ws_name);

                    if let Ok(wf_entries) = fs::read_dir(&ws_path) {
                        let mut sub_nodes = Vec::new();
                        for wf_entry in wf_entries.flatten() {
                            let file_path = wf_entry.path();
                            if file_path.is_file() && file_path.extension().and_then(|s| s.to_str()) == Some("json") {
                                if let Some(wf_name) = file_path.file_stem().and_then(|s| s.to_str()) {
                                    sub_nodes.push(RegistryNode {
                                        id: format!("wf_{}_{}", ws_name, wf_name),
                                        label: wf_name.to_string(),
                                        description: "Sub-Workflow".to_string(),
                                        icon: "🔗".to_string(),
                                        color_theme: "slate".to_string(),
                                        workflow_id: Some(wf_name.to_string()),
                                        workspace_id: Some(ws_name.to_string()),
                                        r#type: Some("workflow".to_string()),
                                    });
                                }
                            }
                        }

                        if !sub_nodes.is_empty() {
                            categories.push(NodeCategory {
                                name: format!("Workspace: {}", ws_name),
                                nodes: sub_nodes,
                            });
                        }
                    }
                }
            }
        }
    }

    Json(categories)
}
