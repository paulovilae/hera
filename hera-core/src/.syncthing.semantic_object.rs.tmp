use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// The universal type of a semantic object in the graph.
/// This enum will expand dynamically as new features are integrated.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ObjectType {
    Contract,
    Clause,
    Paragraph,
    Image,
    VectorShape,
    Slide,
    Template,
    #[serde(other)]
    Unknown,
}

/// The core anatomy of EVERY entity in the ImagineOS system.
/// All LLM interactions use this graph representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticObject {
    pub id: String,
    pub parent_id: Option<String>,
    pub title: String,
    pub object_type: ObjectType,
    pub content: serde_json::Value,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl SemanticObject {
    pub fn new(id: impl Into<String>, title: impl Into<String>, object_type: ObjectType) -> Self {
        Self {
            id: id.into(),
            parent_id: None,
            title: title.into(),
            object_type,
            content: serde_json::Value::Null,
            metadata: HashMap::new(),
        }
    }

    /// Sets the parent context of this object
    pub fn with_parent(mut self, parent_id: impl Into<String>) -> Self {
        self.parent_id = Some(parent_id.into());
        self
    }
}
