use crate::ipc::types::{HandlerOutcome, IpcPayload, IpcState};
use serde_json::json;
use tracing::{error, info};

#[cfg(feature = "execution-tools")]
use hera_execution::workflow::{WorkflowRequest, execute_dag};

pub async fn handle_execute_dag(request: &IpcPayload, _state: &IpcState) -> HandlerOutcome {
    info!("🚀 Executing DAG via Hera Engine");

    #[cfg(not(feature = "execution-tools"))]
    {
        return HandlerOutcome::Result {
            result_text: "execution-tools feature not enabled in Hera".to_string(),
            origin: "system".to_string(),
            model: "".to_string(),
            tool_calls: None,
        };
    }

    #[cfg(feature = "execution-tools")]
    {
        // Parse the WorkflowRequest from the incoming payload
        let req_res: Result<WorkflowRequest, _> = serde_json::from_value(request.payload.clone());

        match req_res {
            Ok(workflow_req) => {
                info!(
                    "📦 DAG parsed successfully with {} nodes and {} edges",
                    workflow_req.nodes.len(),
                    workflow_req.edges.len()
                );
                let response = execute_dag(workflow_req).await;

                // Pack the workflow response into a JSON structure under 'result'
                // This maps Node IDs to their execution outputs.
                let result_json = json!({
                    "results": response.results,
                    "errors": response.errors,
                });

                HandlerOutcome::Result {
                    result_text: result_json.to_string(),
                    origin: "system".to_string(),
                    model: "".to_string(),
                    tool_calls: None,
                }
            }
            Err(e) => {
                error!("❌ Failed to parse DAG payload: {}", e);
                HandlerOutcome::Result {
                    result_text: format!("Failed to parse workflow request: {}", e),
                    origin: "system".to_string(),
                    model: "".to_string(),
                    tool_calls: None,
                }
            }
        }
    }
}
