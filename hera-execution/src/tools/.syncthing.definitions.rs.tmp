use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolArgument {
    #[serde(rename = "type")]
    pub arg_type: String,
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enum_values: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputSchema {
    #[serde(rename = "type")]
    pub schema_type: String, // usually "object"
    pub properties: std::collections::HashMap<String, ToolArgument>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub required: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    #[serde(rename = "inputSchema")]
    pub input_schema: InputSchema,
}

/// Helper to build a ToolDefinition easily
pub fn build_tool(
    name: &str,
    description: &str,
    properties: std::collections::HashMap<String, ToolArgument>,
    required: Vec<String>,
) -> ToolDefinition {
    ToolDefinition {
        name: name.to_string(),
        description: description.to_string(),
        input_schema: InputSchema {
            schema_type: "object".to_string(),
            properties,
            required,
        },
    }
}
