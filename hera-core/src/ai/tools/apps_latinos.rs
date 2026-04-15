//! Latinos Trading Quant Lab tool executors
use crate::ai::tool_executor::{ToolCall, ToolResult};
use crate::ai::tools::data::execute_memento_query;
use serde_json::Value;
use std::process::Command;

pub(crate) async fn execute_market_research_json(call: &ToolCall) -> Result<Value, String> {
    let ticker = call
        .arguments
        .get("ticker")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| "Missing 'ticker' argument".to_string())?;

    let script_path = "/home/paulo/Programs/apps/OS/Tools/apps/latinos/scripts/market_research.py";
    let output = Command::new("python3")
        .arg(script_path)
        .arg(ticker)
        .output()
        .map_err(|error| format!("Failed to execute market_research script: {}", error))?;

    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
    }

    let stdout = String::from_utf8(output.stdout)
        .map_err(|error| format!("Failed to decode tool output: {}", error))?;
    let parsed = serde_json::from_str::<Value>(&stdout)
        .map_err(|error| format!("Failed to parse market_research output: {}", error))?;

    if parsed.get("error").is_some() {
        Err(parsed
            .get("error")
            .and_then(|value| value.as_str())
            .unwrap_or("market_research returned an error")
            .to_string())
    } else {
        Ok(parsed)
    }
}

pub(crate) async fn execute_market_research(call: &ToolCall) -> ToolResult {
    match execute_market_research_json(call).await {
        Ok(result) => ToolResult {
            name: call.name.clone(),
            success: true,
            output: serde_json::to_string_pretty(&result).unwrap_or_else(|_| result.to_string()),
        },
        Err(error) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: error,
        },
    }
}

// ── FMP/SEC Consultant Report Analyzer ────────────────────────────────────────
pub(crate) async fn execute_consultant_report_analyzer_json(
    call: &ToolCall,
) -> Result<Value, String> {
    let ticker = call
        .arguments
        .get("ticker")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| "Missing 'ticker' argument".to_string())?;

    let _focus = call
        .arguments
        .get("focus")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .unwrap_or("General comprehensive fundamentals");

    // Phase 4 FMP integration (mocked fallback if no keys configured)
    let fmp_key = std::env::var("FMP_API_KEY").unwrap_or_default();

    if fmp_key.is_empty() {
        return Ok(serde_json::json!({
            "ticker": ticker,
            "executive_summary": format!("Consultant analysis for {ticker}: FMP API key missing. Live data unavailable."),
            "investment_thesis": "Research pending. Connect a data provider to generate live analysis.",
            "is_mock": true,
            "status": "pending_data",
            "technical_analysis": { "overall_assessment": "Data Unavailable" },
            "price_info": { "change_percent": 0.0, "current_price": 0.0 },
            "analyst_recommendations": { "consensus": "N/A", "upside_percent": 0.0 },
            "fundamentals": { "pe_ratio": null, "revenue_growth": 0.0, "market_cap": "N/A", "beta": 1.0 },
            "catalysts_and_news": { "recent_headlines": [], "upcoming_events": [] },
            "risks": ["Live data feed not configured."]
        }));
    }

    // [REAL FMP LOGIC WOULD GO HERE IN FUTURE PHASES]
    Ok(
        serde_json::json!({ "ticker": ticker, "status": "error", "message": "FMP logic not yet implemented" }),
    )
}

pub(crate) async fn execute_consultant_report_analyzer(call: &ToolCall) -> ToolResult {
    match execute_consultant_report_analyzer_json(call).await {
        Ok(result) => ToolResult {
            name: call.name.clone(),
            success: true,
            output: serde_json::to_string_pretty(&result).unwrap_or_else(|_| result.to_string()),
        },
        Err(error) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: error,
        },
    }
}

pub(crate) async fn execute_list_bots(call: &ToolCall) -> ToolResult {
    let memento_call = ToolCall {
        name: "memento_query".to_string(),
        arguments: serde_json::json!({
            "app": "latinos",
            "query": "SELECT id, name, status, live_trading, created_at, updated_at FROM latinos_bots ORDER BY updated_at DESC NULLS LAST, created_at DESC LIMIT 25"
        }),
    };

    let mut result = execute_memento_query(&memento_call).await;
    result.name = call.name.clone();
    result
}

pub(crate) async fn execute_get_bot_status(call: &ToolCall) -> ToolResult {
    let Some(bot_id) = call
        .arguments
        .get("bot_id")
        .and_then(|value| value.as_i64())
    else {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Missing required 'bot_id' parameter.".to_string(),
        };
    };
    let include_trades = call
        .arguments
        .get("include_trades")
        .and_then(|value| value.as_bool())
        .unwrap_or(true);
    let trade_limit = call
        .arguments
        .get("trade_limit")
        .and_then(|value| value.as_i64())
        .unwrap_or(10)
        .clamp(1, 50);

    let bot_call = ToolCall {
        name: "memento_query".to_string(),
        arguments: serde_json::json!({
            "app": "latinos",
            "query": format!("SELECT id, name, status, live_trading, live_metrics, updated_at FROM latinos_bots WHERE id = {} LIMIT 1", bot_id)
        }),
    };
    let bot_result = execute_memento_query(&bot_call).await;
    if !bot_result.success {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: bot_result.output,
        };
    }

    if !include_trades {
        return ToolResult {
            name: call.name.clone(),
            success: true,
            output: format!("Bot status:\n{}", bot_result.output),
        };
    }

    let trades_call = ToolCall {
        name: "memento_query".to_string(),
        arguments: serde_json::json!({
            "app": "latinos",
            "query": format!("SELECT symbol, side, price, amount, status, pnl, timestamp FROM latinos_trades WHERE bot_id = {} ORDER BY timestamp DESC LIMIT {}", bot_id, trade_limit)
        }),
    };
    let trades_result = execute_memento_query(&trades_call).await;

    ToolResult {
        name: call.name.clone(),
        success: true,
        output: if trades_result.success {
            format!(
                "Bot status:\n{}\n\nRecent trades:\n{}",
                bot_result.output, trades_result.output
            )
        } else {
            format!(
                "Bot status:\n{}\n\nRecent trades could not be loaded:\n{}",
                bot_result.output, trades_result.output
            )
        },
    }
}

// ── Latinos Quant Lab: Generic Bridge ───────────────────────────────────────
// Single dispatcher for run_backtest, load_market_data, scan_opportunities.
// Calls Tools/apps/latinos/scripts/latinos_bridge.py with the tool name + args.
pub(crate) async fn execute_latinos_bridge(call: &ToolCall, tool: &str) -> ToolResult {
    let script = "/home/paulo/Programs/apps/OS/Tools/apps/latinos/scripts/latinos_bridge.py";

    // Build CLI args from the tool call arguments
    let mut cmd = tokio::process::Command::new("python3");
    cmd.arg(script).arg(tool);

    // Forward known arguments as flags
    if let Some(v) = call.arguments.get("bot_id").and_then(|v| v.as_i64()) {
        cmd.args(["--bot-id", &v.to_string()]);
    }
    if let Some(v) = call.arguments.get("symbol").and_then(|v| v.as_str()) {
        cmd.args(["--symbol", v]);
    }
    if let Some(v) = call.arguments.get("interval").and_then(|v| v.as_str()) {
        cmd.args(["--interval", v]);
    }
    if let Some(v) = call.arguments.get("range").and_then(|v| v.as_str()) {
        cmd.args(["--range", v]);
    }
    if let Some(v) = call
        .arguments
        .get("initial_capital")
        .and_then(|v| v.as_f64())
    {
        cmd.args(["--initial-capital", &v.to_string()]);
    }
    if let Some(v) = call.arguments.get("source").and_then(|v| v.as_str()) {
        cmd.args(["--source", v]);
    }
    if let Some(v) = call.arguments.get("min_score").and_then(|v| v.as_f64()) {
        cmd.args(["--min-score", &v.to_string()]);
    }
    if let Some(v) = call.arguments.get("limit").and_then(|v| v.as_i64()) {
        cmd.args(["--limit", &v.to_string()]);
    }

    match cmd.output().await {
        Ok(output) if output.status.success() => {
            let out = String::from_utf8_lossy(&output.stdout).trim().to_string();
            ToolResult {
                name: call.name.clone(),
                success: true,
                output: out,
            }
        }
        Ok(output) => {
            let err = String::from_utf8_lossy(&output.stderr).trim().to_string();
            ToolResult {
                name: call.name.clone(),
                success: false,
                output: format!("Bridge error: {}", err),
            }
        }
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Failed to spawn latinos_bridge.py: {}", e),
        },
    }
}
