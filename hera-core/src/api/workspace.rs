use axum::{extract::Path, http::StatusCode, response::IntoResponse, Json};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use hera_execution::workflow::WorkflowRequest; // React Flow exported DAG representation

#[derive(Serialize)]
pub struct WorkspaceList {
    pub workspaces: Vec<WorkspaceItem>,
}

#[derive(Serialize, Deserialize)]
pub struct WorkspaceItem {
    pub id: String,
    pub name: String,
}

#[derive(Serialize)]
pub struct WorkflowList {
    pub workflows: Vec<WorkflowItem>,
}

#[derive(Serialize, Deserialize)]
pub struct WorkflowItem {
    pub id: String,
    pub name: String,
}

#[derive(Deserialize)]
pub struct CreateWorkspaceRequest {
    pub name: String,
}

#[derive(Deserialize)]
pub struct CreateWorkflowRequest {
    pub name: String,
    pub data: WorkflowRequest,
}

fn get_base_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/paulo".to_string());
    let mut path = PathBuf::from(home);
    path.push(".hera");
    path.push("workspaces");
    if !path.exists() {
        fs::create_dir_all(&path).unwrap_or_default();
    }
    path
}

pub async fn list_workspaces() -> impl IntoResponse {
    let base_dir = get_base_dir();
    let mut workspaces = Vec::new();

    if let Ok(entries) = fs::read_dir(base_dir) {
        for entry in entries.flatten() {
            if entry.path().is_dir() {
                if let Some(name) = entry.file_name().to_str() {
                    workspaces.push(WorkspaceItem {
                        id: name.to_string(), // Dir name serves as ID here since we create safe names
                        name: name.to_string(),
                    });
                }
            }
        }
    }

    Json(WorkspaceList { workspaces })
}

pub async fn create_workspace(Json(req): Json<CreateWorkspaceRequest>) -> impl IntoResponse {
    let base_dir = get_base_dir();
    
    // Sanitize dir name
    let safe_name: String = req.name.chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect();
    
    let mut path = base_dir.clone();
    path.push(&safe_name);

    if path.exists() {
        return (StatusCode::CONFLICT, Json(serde_json::json!({ "error": "Workspace already exists" }))).into_response();
    }

    match fs::create_dir(&path) {
        Ok(_) => Json(WorkspaceItem { id: safe_name.clone(), name: safe_name }).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e.to_string() }))).into_response(),
    }
}

pub async fn delete_workspace(Path(workspace_id): Path<String>) -> impl IntoResponse {
    let mut path = get_base_dir();
    path.push(&workspace_id);

    if !path.exists() {
        return (StatusCode::NOT_FOUND, Json(serde_json::json!({ "error": "Workspace not found" }))).into_response();
    }

    match fs::remove_dir_all(&path) {
        Ok(_) => (StatusCode::OK, Json(serde_json::json!({ "success": true }))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e.to_string() }))).into_response(),
    }
}

pub async fn list_workflows(Path(workspace_id): Path<String>) -> impl IntoResponse {
    let mut path = get_base_dir();
    path.push(&workspace_id);

    if !path.exists() {
        return (StatusCode::NOT_FOUND, Json(serde_json::json!({ "error": "Workspace not found" }))).into_response();
    }

    let mut workflows = Vec::new();
    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.flatten() {
            let file_path = entry.path();
            if file_path.is_file() && file_path.extension().and_then(|s| s.to_str()) == Some("json") {
                if let Some(name) = file_path.file_stem().and_then(|s| s.to_str()) {
                    workflows.push(WorkflowItem {
                        id: name.to_string(),
                        name: name.to_string(), // In a more complex setup, name would be read from JSON
                    });
                }
            }
        }
    }

    Json(WorkflowList { workflows }).into_response()
}

pub async fn get_workflow(Path((workspace_id, workflow_id)): Path<(String, String)>) -> impl IntoResponse {
    let mut path = get_base_dir();
    path.push(&workspace_id);
    path.push(format!("{}.json", workflow_id));

    if !path.exists() {
        return (StatusCode::NOT_FOUND, Json(serde_json::json!({ "error": "Workflow not found" }))).into_response();
    }

    match fs::read_to_string(&path) {
        Ok(content) => {
            match serde_json::from_str::<WorkflowRequest>(&content) {
                Ok(data) => (StatusCode::OK, Json(data)).into_response(),
                Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": "Failed to parse json", "details": e.to_string() }))).into_response()
            }
        },
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e.to_string() }))).into_response(),
    }
}

pub async fn save_workflow(Path((workspace_id, workflow_id)): Path<(String, String)>, Json(req): Json<WorkflowRequest>) -> impl IntoResponse {
    let mut path = get_base_dir();
    path.push(&workspace_id);
    
    if !path.exists() {
        return (StatusCode::NOT_FOUND, Json(serde_json::json!({ "error": "Workspace not found" }))).into_response();
    }

    path.push(format!("{}.json", workflow_id));

    match serde_json::to_string_pretty(&req) {
        Ok(json_content) => {
            match fs::write(&path, json_content) {
                Ok(_) => (StatusCode::OK, Json(serde_json::json!({ "success": true, "id": workflow_id }))).into_response(),
                Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e.to_string() }))).into_response(),
            }
        },
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": "Serialization failed", "details": e.to_string() }))).into_response(),
    }
}

pub async fn delete_workflow(Path((workspace_id, workflow_id)): Path<(String, String)>) -> impl IntoResponse {
    let mut path = get_base_dir();
    path.push(&workspace_id);
    path.push(format!("{}.json", workflow_id));

    if !path.exists() {
        return (StatusCode::NOT_FOUND, Json(serde_json::json!({ "error": "Workflow not found" }))).into_response();
    }

    match fs::remove_file(&path) {
        Ok(_) => (StatusCode::OK, Json(serde_json::json!({ "success": true }))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e.to_string() }))).into_response(),
    }
}
