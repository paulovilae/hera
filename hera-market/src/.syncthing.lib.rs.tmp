pub async fn quote_range_json(symbol: &str, interval: &str, range: &str) -> String {
    let provider = match yahoo_finance_api::YahooConnector::new() {
        Ok(provider) => provider,
        Err(e) => return serde_json::json!({"error": format!("Provider Error: {}", e)}).to_string(),
    };

    match provider.get_quote_range(symbol, interval, range).await {
        Ok(response) => match response.quotes() {
            Ok(quotes) => serde_json::to_string(&quotes).unwrap_or_else(|_| "[]".to_string()),
            Err(e) => serde_json::json!({"error": format!("Parsing Error: {}", e)}).to_string(),
        },
        Err(e) => serde_json::json!({"error": format!("API Error: {}", e)}).to_string(),
    }
}
