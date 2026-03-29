use anyhow::{Result, anyhow};
use serde_json::json;

pub async fn dispatch(action: &str, payload: serde_json::Value) -> Result<serde_json::Value> {
    match action {
        "execute_dag" => {
            let req: hera_execution::workflow::WorkflowRequest = serde_json::from_value(payload)?;
            let result = hera_execution::workflow::execute_dag(req).await;
            Ok(serde_json::to_value(result)?)
        }
        "parse_dify" => {
            let raw = payload
                .get("dsl")
                .and_then(|value| value.as_str())
                .ok_or_else(|| anyhow!("Missing `dsl` string"))?;
            let result = hera_execution::dify::parse_dify_json(raw)
                .map_err(anyhow::Error::msg)?;
            Ok(serde_json::to_value(result)?)
        }
        "execute_workflow_proxy" => {
            let app = payload
                .get("app")
                .and_then(|value| value.as_str())
                .unwrap_or_default();
            let workflow = payload
                .get("workflow")
                .and_then(|value| value.as_str())
                .unwrap_or_default();
            let req_payload = payload
                .get("payload")
                .cloned()
                .unwrap_or_else(|| json!({}));

            if app.is_empty() || workflow.is_empty() {
                return Err(anyhow!("Missing required `app` or `workflow` parameters"));
            }

            let url = format!("http://127.0.0.1:3006/execute/{}/{}", app, workflow);
            let client = reqwest::Client::new();
            let response = client.post(&url).json(&req_payload).send().await?;
            let status = response.status();
            let body = response.text().await.unwrap_or_default();

            if status.is_success() {
                let parsed = serde_json::from_str::<serde_json::Value>(&body)
                    .unwrap_or_else(|_| json!({ "raw": body }));
                Ok(json!({
                    "status": status.as_u16(),
                    "url": url,
                    "body": parsed
                }))
            } else {
                Err(anyhow!("Argus returned status {}: {}", status, body))
            }
        }
        _ => Err(anyhow!("Unsupported workflow action `{}`", action)),
    }
}
