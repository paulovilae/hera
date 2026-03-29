use anyhow::{Result, anyhow};
use serde_json::json;

pub async fn dispatch(action: &str, payload: serde_json::Value) -> Result<serde_json::Value> {
    let hera = hera_execution::agents::hera::Hera::new("http://127.0.0.1:3000");
    match action {
        "web_scrape" => {
            let url = payload
                .get("url")
                .and_then(|value| value.as_str())
                .ok_or_else(|| anyhow!("Missing `url`"))?;
            let text = hera.native_web_scrape(url).await?;
            Ok(json!({
                "url": url,
                "content": text
            }))
        }
        "web_search" => {
            let query = payload
                .get("query")
                .and_then(|value| value.as_str())
                .ok_or_else(|| anyhow!("Missing `query`"))?;
            let text = hera.native_web_search(query).await?;
            Ok(json!({
                "query": query,
                "content": text
            }))
        }
        _ => Err(anyhow!("Unsupported web action `{}`", action)),
    }
}
