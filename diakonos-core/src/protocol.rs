use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct DiakonosRequest {
    pub action: String,
    #[serde(default)]
    pub payload: serde_json::Value,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct DiakonosResponse {
    pub status: String,
    pub data: serde_json::Value,
}
