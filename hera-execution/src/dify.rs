use serde::Deserialize;
use std::collections::HashMap;
use crate::workflow::{WorkflowRequest, GraphNode, GraphEdge, NodeData};

#[derive(Deserialize, Debug)]
pub struct DifyWorkflow {
    pub graph: DifyGraph,
}

#[derive(Deserialize, Debug)]
pub struct DifyGraph {
    #[serde(default)]
    pub nodes: Vec<DifyNode>,
    #[serde(default)]
    pub edges: Vec<DifyEdge>,
}

#[derive(Deserialize, Debug)]
pub struct DifyNode {
    pub id: String,
    pub data: DifyNodeData,
}

#[derive(Deserialize, Debug)]
pub struct DifyNodeData {
    pub title: Option<String>,
    #[serde(rename = "type")]
    pub node_type: String,
    pub desc: Option<String>,
    pub code: Option<String>,
    pub model: Option<String>,
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub variables: Vec<HashMap<String, String>>,
}

#[derive(Deserialize, Debug)]
pub struct DifyEdge {
    pub id: String,
    pub source: String,
    pub target: String,
}

/// Converts a standard Dify DSL exported JSON representation into the native
/// Hera execution DAG schema.
pub fn parse_dify_json(json_str: &str) -> Result<WorkflowRequest, String> {
    let dify: DifyWorkflow = serde_json::from_str(json_str)
        .map_err(|e| format!("Failed to parse Dify JSON: {}", e))?;

    let mut req_nodes = Vec::new();
    let mut req_edges = Vec::new();

    for edge in dify.graph.edges {
        req_edges.push(GraphEdge {
            id: edge.id,
            source: edge.source,
            target: edge.target,
            animated: true,
            style: None,
        });
    }

    for node in dify.graph.nodes {
        // Translation logic depending on type
        let ntype = match node.data.node_type.as_str() {
            "start" => "input",
            "end" => "output",
            "llm" => "llm",
            "code" => "code",
            "tool" => "mcp",
            _ => node.data.node_type.as_str()
        };
        
        // Translate Dify's internal representation to Hera execution blocks
        let req_node = GraphNode {
            id: node.id,
            node_type: "universalBlock".to_string(),
            position: Default::default(),
            data: NodeData {
                label: node.data.title,
                node_type: Some(ntype.to_string()),
                value: node.data.model,
                language: Some("python".to_string()), // Default Dify code block assumption for Python3
                code: node.data.code,
                system_prompt: node.data.system_prompt,
                ..Default::default()
            }
        };

        req_nodes.push(req_node);
    }

    Ok(WorkflowRequest {
        nodes: req_nodes,
        edges: req_edges,
    })
}
