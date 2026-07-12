//! Hera Tool Executor — Sovereign Tool Calling for ImagineOS
//!
//! Defines tool schemas in Qwen's native format, parses `<tool_call>` blocks
//! from Qwen output, and dispatches tool execution to existing Hera methods.

use serde::{Deserialize, Serialize};
use tracing::info;

pub mod dispatch;
pub mod intent;
pub mod registry;
pub mod schema;
pub mod security;

pub use self::dispatch::execute_tool_raw_json;
pub use self::intent::detect_intent_from_user_message;
pub use self::schema::hera_tool_schemas;
pub use self::schema::collect_hera_tool_schemas;
pub use self::schema::{AgentArtifact, SkillArtifact};
pub use self::registry::{
    canonicalize_app_slug, canonical_app_search_terms, load_canonical_app_registry,
    pm2_process_name_for_slug, text_contains_app_alias, CanonicalAppEntry,
};
pub use self::schema::{find_skill_artifact, load_agent_artifact};
pub use self::security::permissions_allow_tool;
pub use self::security::tool_is_critical;
pub use self::dispatch::execute_tool;

const DEFAULT_TOOL_TIMEOUT_MS: u64 = 90_000;
const HERA_TOOL_AUDIT_LOG: &str = "/tmp/hera_tool_audit.jsonl";

/// Tool call parsed from Qwen's output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Result of executing a tool
#[derive(Debug, Clone)]
pub struct ToolResult {
    pub name: String,
    pub success: bool,
    pub output: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolRiskLevel {
    Low,
    High,
    Critical,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ToolRuntimeMetadata {
    execution_kind: Option<String>,
    risk_level: Option<ToolRiskLevel>,
    timeout_ms: Option<u64>,
    allowed_callers: Vec<String>,
}

/// Parse tool call blocks from LLM text output.
/// Supports multiple tag formats: <tool_call>, <function-call>, <function_call>
/// Returns empty vec if no tool calls found.
pub fn parse_tool_calls(text: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();

    // Try multiple tag formats that various LLMs hallucinate
    let tag_formats: &[(&str, &str)] = &[
        ("<tool_call>", "</tool_call>"),
        ("<tool_code>", "</tool_code>"),
        ("<function-call>", "</function-call>"),
        ("<function_call>", "</function_call>"),
    ];

    for &(open_tag, close_tag) in tag_formats {
        let mut search_from = 0;
        while let Some(start) = text[search_from..].find(open_tag) {
            let abs_start = search_from + start + open_tag.len();
            if let Some(end) = text[abs_start..].find(close_tag) {
                let abs_end = abs_start + end;
                let mut json_str = text[abs_start..abs_end].trim();
                // Strip any nested hallucinated tags
                json_str = json_str.trim_start_matches("<function-call>").trim();
                json_str = json_str.trim_end_matches("</function-call>").trim();
                json_str = json_str.trim_start_matches("<tool_call>").trim();
                json_str = json_str.trim_end_matches("</tool_call>").trim();
                json_str = json_str.trim_start_matches("<tool_code>").trim();
                json_str = json_str.trim_end_matches("</tool_code>").trim();
                json_str = json_str.trim_start_matches("```json").trim();
                json_str = json_str.trim_start_matches("```").trim();
                json_str = json_str.trim_end_matches("```").trim();

                match parse_tool_call_json(json_str) {
                    Ok(val) => {
                        append_tool_calls_from_value(&mut calls, &val, open_tag);
                    }
                    Err(e) => {
                        tracing::warn!(
                            "⚠️ [Hera] Failed to parse tool_call JSON: {} — raw: {}",
                            e,
                            json_str
                        );
                    }
                }
                search_from = abs_end + close_tag.len();
            } else {
                // Closing tag missing. Observed live 2026-07-11 (3/6 runs of
                // `scripts/hera_diagnose_incident.sh` against the real Qwen3
                // model on genesis): asked to "copy this exact <tool_call>{...}
                // </tool_call> format", the model faithfully reproduces the
                // opening tag and a complete, valid JSON object, then stops
                // WITHOUT emitting `</tool_call>` — e.g.
                // `<tool_call>{"name":"read_pm2_logs","arguments":{...}}}` with
                // no closing tag at all (sometimes with a stray trailing brace
                // too). Neither the tag-pair path above nor the bare-JSON
                // fallback below catches this: the former requires the close
                // tag, the latter requires the text to literally start with
                // `{` (it starts with `<tool_call>`). Recover by brace-balance
                // extracting the JSON object right after the open tag — same
                // technique already used for the untagged fallback — so a
                // well-formed-but-unterminated call still executes instead of
                // silently leaking into the model's "final answer" text.
                if let Some(json_str) = extract_balanced_json_object(&text[abs_start..]) {
                    match parse_tool_call_json(&json_str) {
                        Ok(val) => {
                            append_tool_calls_from_value(&mut calls, &val, open_tag);
                        }
                        Err(e) => {
                            tracing::warn!(
                                "⚠️ [Hera] Failed to parse unterminated tool_call JSON: {} — raw: {}",
                                e,
                                json_str
                            );
                        }
                    }
                }
                break;
            }
        }
        // If we found calls with this tag format, stop trying others
        if !calls.is_empty() {
            return calls;
        }
    }

    // Llama/Nemotron format: <function=NAME><parameter=KEY>VALUE</parameter>...</function>
    if calls.is_empty() {
        let mut search_from = 0;
        while let Some(start) = text[search_from..].find("<function=") {
            let abs_start = search_from + start + "<function=".len();
            // Find the function name (up to the next > character)
            if let Some(name_end) = text[abs_start..].find('>') {
                let abs_name_end = abs_start + name_end;
                let func_name = text[abs_start..abs_name_end].trim();

                // Find the closing </function> tag
                if let Some(func_end) = text[abs_name_end..].find("</function>") {
                    let abs_func_end = abs_name_end + func_end;
                    let body = &text[abs_name_end + 1..abs_func_end];

                    // Parse <parameter=KEY>VALUE</parameter> pairs
                    let mut args = serde_json::Map::new();
                    let mut param_search = 0;
                    while let Some(ps) = body[param_search..].find("<parameter=") {
                        let p_start = param_search + ps + "<parameter=".len();
                        if let Some(p_name_end) = body[p_start..].find('>') {
                            let abs_p_name_end = p_start + p_name_end;
                            let param_name = body[p_start..abs_p_name_end].trim();
                            if let Some(p_val_end) = body[abs_p_name_end + 1..].find("</parameter>")
                            {
                                let abs_p_val_end = abs_p_name_end + 1 + p_val_end;
                                let param_value = body[abs_p_name_end + 1..abs_p_val_end].trim();
                                args.insert(
                                    param_name.to_string(),
                                    serde_json::Value::String(param_value.to_string()),
                                );
                                param_search = abs_p_val_end + "</parameter>".len();
                            } else {
                                break;
                            }
                        } else {
                            break;
                        }
                    }

                    if !func_name.is_empty() {
                        calls.push(ToolCall {
                            name: func_name.to_string(),
                            arguments: serde_json::Value::Object(args),
                        });
                        info!(
                            "🔧 [Hera] Parsed Llama-style tool call: {} with args: {}",
                            func_name,
                            serde_json::to_string(&calls.last().unwrap().arguments)
                                .unwrap_or_default()
                        );
                    }
                    search_from = abs_func_end + "</function>".len();
                } else {
                    break;
                }
            } else {
                break;
            }
        }
    }

    // Fallback: If no tags were found, maybe the model just spit out raw JSON?
    // Strip <think>...</think> reasoning tags first (Qwen models produce these)
    if calls.is_empty() {
        let stripped = if let Some(end_idx) = text.find("</think>") {
            text[end_idx + 8..].trim()
        } else {
            text.trim()
        };
        // A weak local model sometimes drops the <tool_call> wrapper entirely and
        // just fences the JSON like a normal answer (```json\n{...}\n```). Without
        // this strip, `starts_with('{')` below is false (the text starts with the
        // fence marker) and the whole call is silently dropped, surfacing to the
        // caller as a "final answer" that is actually an unexecuted tool call.
        // Observed 2026-07-07: hera_compile.sh on Consulting-rust got exactly this
        // shape back verbatim instead of a real cargo_check run.
        let stripped = stripped
            .strip_prefix("```json")
            .or_else(|| stripped.strip_prefix("```"))
            .map(|rest| rest.trim_start())
            .unwrap_or(stripped);
        let stripped = stripped
            .strip_suffix("```")
            .map(|rest| rest.trim_end())
            .unwrap_or(stripped);
        if stripped.starts_with('{') {
            let mut brace_count = 0;
            let mut end_idx = 0;
            for (i, c) in stripped.char_indices() {
                if c == '{' {
                    brace_count += 1;
                } else if c == '}' {
                    brace_count -= 1;
                }

                if brace_count == 0 && i > 0 {
                    end_idx = i + 1;
                    break;
                }
            }

            let mut json_str = String::new();
            if end_idx > 0 {
                json_str = stripped[..end_idx].to_string();
            } else if brace_count > 0 {
                json_str = stripped.to_string();
                for _ in 0..brace_count {
                    json_str.push('}');
                }
            }

            if !json_str.is_empty() {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&json_str) {
                    let mut parsed_name = None;
                    let mut parsed_args = None;

                    if let (Some(name), Some(args)) = (
                        val.get("name").and_then(|n| n.as_str()),
                        val.get("arguments").or_else(|| val.get("parameters")),
                    ) {
                        parsed_name = Some(name.to_string());
                        parsed_args = Some(args.clone());
                    } else if let Some(func) = val.get("function") {
                        if let (Some(name), Some(args)) = (
                            func.get("name").and_then(|n| n.as_str()),
                            func.get("arguments").or_else(|| func.get("parameters")),
                        ) {
                            parsed_name = Some(name.to_string());
                            parsed_args = Some(args.clone());
                        }
                    }

                    if let (Some(name), Some(args)) = (parsed_name, parsed_args) {
                        calls.push(ToolCall {
                            name: name.clone(),
                            arguments: args.clone(),
                        });
                        info!(
                            "🔧 [Hera] Parsed RAW JSON tool call (after think-strip): {} with args: {}",
                            name,
                            serde_json::to_string(&args).unwrap_or_default()
                        );
                    }
                }
            }
        }
    }

    calls
}

fn parse_tool_call_json(json_str: &str) -> Result<serde_json::Value, serde_json::Error> {
    match serde_json::from_str::<serde_json::Value>(json_str) {
        Ok(value) => Ok(value),
        Err(first_error) => {
            if !matches!(
                first_error.classify(),
                serde_json::error::Category::Eof
            ) {
                return Err(first_error);
            }

            let repaired = repair_truncated_json(json_str);
            if repaired == json_str {
                return Err(first_error);
            }

            serde_json::from_str::<serde_json::Value>(&repaired).map_err(|_| first_error)
        }
    }
}

fn repair_truncated_json(input: &str) -> String {
    let mut repaired = input.trim().to_string();
    if repaired.is_empty() {
        return repaired;
    }

    let mut in_string = false;
    let mut escaped = false;
    let mut brace_balance = 0usize;
    let mut bracket_balance = 0usize;

    for ch in repaired.chars() {
        if in_string {
            if escaped {
                escaped = false;
                continue;
            }
            match ch {
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '{' => brace_balance += 1,
            '}' => brace_balance = brace_balance.saturating_sub(1),
            '[' => bracket_balance += 1,
            ']' => bracket_balance = bracket_balance.saturating_sub(1),
            _ => {}
        }
    }

    if in_string {
        repaired.push('"');
    }
    for _ in 0..bracket_balance {
        repaired.push(']');
    }
    for _ in 0..brace_balance {
        repaired.push('}');
    }

    repaired
}

/// Extract the first balanced `{...}` JSON object from the start of `text`
/// (after trimming leading whitespace), tolerating trailing garbage after the
/// matching close brace (e.g. a stray extra `}` or missing closing tag — see
/// the caller in `parse_tool_calls`). String contents are scanned with the
/// same escape-aware tracking as `repair_truncated_json` so a `}` inside a
/// quoted argument value never miscounts. Returns `None` if the text doesn't
/// start with `{` or the braces never balance.
fn extract_balanced_json_object(text: &str) -> Option<String> {
    let trimmed = text.trim_start();
    if !trimmed.starts_with('{') {
        return None;
    }

    let mut in_string = false;
    let mut escaped = false;
    let mut brace_balance = 0i32;

    for (i, ch) in trimmed.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else {
                match ch {
                    '\\' => escaped = true,
                    '"' => in_string = false,
                    _ => {}
                }
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '{' => brace_balance += 1,
            '}' => {
                brace_balance -= 1;
                if brace_balance == 0 {
                    return Some(trimmed[..=i].to_string());
                }
            }
            _ => {}
        }
    }

    None
}

fn append_tool_calls_from_value(
    calls: &mut Vec<ToolCall>,
    value: &serde_json::Value,
    source_tag: &str,
) {
    if let Some(items) = value.get("calls").and_then(|v| v.as_array()) {
        for item in items {
            append_tool_calls_from_value(calls, item, source_tag);
        }
        return;
    }

    if let Some(items) = value.as_array() {
        for item in items {
            append_tool_calls_from_value(calls, item, source_tag);
        }
        return;
    }

    let (name_val, args_val) = if let Some(func) = value.get("function") {
        (
            func.get("name"),
            func.get("arguments").or_else(|| func.get("parameters")),
        )
    } else {
        (
            value.get("name"),
            value.get("arguments").or_else(|| value.get("parameters")),
        )
    };

    if let (Some(name), Some(args)) = (name_val.and_then(|n| n.as_str()), args_val) {
        calls.push(ToolCall {
            name: name.to_string(),
            arguments: args.clone(),
        });
        info!(
            "🔧 [Hera] Parsed tool call (via {}): {} with args: {}",
            source_tag,
            name,
            serde_json::to_string(args).unwrap_or_default()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::collections::HashSet;
    use serde_json::Value;
    use self::registry::registered_tool_names;
    use self::schema::{tool_has_runtime_dispatch, tool_has_raw_json_dispatch};

    #[derive(Debug)]
    struct FunctionTool {
        name: String,
        execution_kind: Option<String>,
        risk_level: Option<String>,
        status: Option<String>,
        path: String,
    }

    fn load_function_tools() -> Vec<FunctionTool> {
        let root = Path::new("/home/paulo/Programs/apps/OS/Tools");
        let mut tools = Vec::new();
        collect_function_tools(root, root, &mut tools);
        tools
    }

    fn collect_function_tools(root: &Path, dir: &Path, tools: &mut Vec<FunctionTool>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_function_tools(root, &path, tools);
                continue;
            }
            if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
                continue;
            }

            let Ok(content) = std::fs::read_to_string(&path) else {
                continue;
            };
            let Ok(schema) = serde_json::from_str::<Value>(&content) else {
                continue;
            };
            let Some(name) = schema
                .get("function")
                .and_then(|function| function.get("name"))
                .and_then(|value| value.as_str())
            else {
                continue;
            };

            tools.push(FunctionTool {
                name: name.to_string(),
                execution_kind: schema
                    .get("metadata")
                    .and_then(|value| value.get("execution_kind"))
                    .and_then(|value| value.as_str())
                    .map(ToString::to_string),
                risk_level: schema
                    .get("metadata")
                    .and_then(|value| value.get("risk_level"))
                    .and_then(|value| value.as_str())
                    .map(ToString::to_string),
                status: schema
                    .get("metadata")
                    .and_then(|value| value.get("status"))
                    .and_then(|value| value.as_str())
                    .map(ToString::to_string),
                path: path
                    .strip_prefix(root)
                    .unwrap_or(&path)
                    .display()
                    .to_string(),
            });
        }
    }

    #[test]
    fn all_registered_function_tools_have_execution_kind() {
        let missing = load_function_tools()
            .into_iter()
            .filter(|tool| tool.execution_kind.is_none())
            .map(|tool| format!("{} ({})", tool.name, tool.path))
            .collect::<Vec<_>>();

        assert!(
            missing.is_empty(),
            "Function tools missing execution_kind: {:?}",
            missing
        );
    }

    #[test]
    fn all_registered_function_tools_have_runtime_dispatch() {
        // Skeleton tools (status == "skeleton_not_implemented") are intentionally
        // un-dispatched WIP: they are hidden from the model (see parse_tool_artifact)
        // and therefore exempt from the dispatch-coverage invariant.
        let missing = load_function_tools()
            .into_iter()
            .filter(|tool| tool.status.as_deref() != Some("skeleton_not_implemented"))
            .filter(|tool| !tool_has_runtime_dispatch(&tool.name, tool.execution_kind.as_deref()))
            .map(|tool| {
                format!(
                    "{} [{}] ({})",
                    tool.name,
                    tool.execution_kind.unwrap_or_else(|| "missing".to_string()),
                    tool.path
                )
            })
            .collect::<Vec<_>>();

        assert!(
            missing.is_empty(),
            "Registered function tools without runtime dispatcher: {:?}",
            missing
        );
    }

    #[test]
    fn registered_tool_registry_matches_function_tools() {
        let tools = load_function_tools();
        let names_from_files = tools
            .iter()
            .map(|tool| tool.name.clone())
            .collect::<HashSet<_>>();
        let registry_names = registered_tool_names().clone();

        assert_eq!(
            registry_names, names_from_files,
            "Registered tool set drifted from Tools/*.json function tools"
        );
    }

    #[test]
    fn raw_json_dispatchers_are_declared_explicitly() {
        let missing = load_function_tools()
            .into_iter()
            .filter(|tool| {
                matches!(
                    tool.name.as_str(),
                    "memento_query"
                        | "market_research"
                        | "analyze_market_research"
                        | "consultant_report_analyzer"
                        | "smoke_apps"
                        | "test_apps"
                        | "verify_canonical_stack"
                        | "review_all_apps_status"
                        | "verify_app_health"
                ) && !tool_has_raw_json_dispatch(&tool.name, tool.execution_kind.as_deref())
            })
            .map(|tool| format!("{} ({})", tool.name, tool.path))
            .collect::<Vec<_>>();

        assert!(
            missing.is_empty(),
            "Tools expected to support raw JSON are missing dispatcher coverage: {:?}",
            missing
        );
    }

    #[test]
    fn critical_tools_declare_risk_metadata() {
        let critical = load_function_tools()
            .into_iter()
            .filter(|tool| {
                matches!(
                    tool.name.as_str(),
                    "api_request"
                        | "run_code"
                        | "write_file"
                        | "service_restart"
                        | "desktop_click"
                        | "desktop_type"
                ) && tool.risk_level.as_deref() != Some("critical")
            })
            .map(|tool| format!("{} ({})", tool.name, tool.path))
            .collect::<Vec<_>>();

        assert!(
            critical.is_empty(),
            "Critical tools missing explicit risk metadata: {:?}",
            critical
        );
    }

    #[test]
    fn explicit_restart_command_maps_to_service_restart() {
        let call = detect_intent_from_user_message("/restart imaginclaw", None)
            .expect("expected explicit /restart command to map");

        assert_eq!(call.name, "service_restart");
        assert_eq!(
            call.arguments
                .get("service_name")
                .and_then(|value| value.as_str()),
            Some("imaginclaw")
        );
    }

    #[test]
    fn explicit_status_command_maps_to_system_status() {
        let call = detect_intent_from_user_message("/status", None)
            .expect("expected explicit /status command to map");

        assert_eq!(call.name, "system_status");
    }

    #[test]
    fn parses_truncated_tool_call_json_by_closing_braces() {
        let text = r#"<tool_call>{"name":"memento_query","arguments":{"app":"cartera","query":"SELECT 1"}</tool_call>"#;

        let calls = parse_tool_calls(text);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "memento_query");
        assert_eq!(
            calls[0]
                .arguments
                .get("app")
                .and_then(|value| value.as_str()),
            Some("cartera")
        );
    }

    #[test]
    fn parses_bare_json_fenced_tool_call_without_wrapper_tags() {
        // Regression for 2026-07-07: a weak local model dropped the <tool_call>
        // wrapper and just fenced the JSON like a normal markdown answer. Before
        // the fix, `starts_with('{')` failed (text started with the fence marker)
        // and the call was silently dropped — the caller saw the raw JSON as a
        // "final answer" instead of an executed tool.
        let text = "```json\n{\"name\":\"cargo_check\",\"arguments\":{\"path\":\"/mnt/workspace/Programs/apps/OS/Apps/Consulting-rust\"}}\n```";

        let calls = parse_tool_calls(text);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "cargo_check");
        assert_eq!(
            calls[0]
                .arguments
                .get("path")
                .and_then(|value| value.as_str()),
            Some("/mnt/workspace/Programs/apps/OS/Apps/Consulting-rust")
        );
    }

    #[test]
    fn parses_bare_json_fence_without_language_tag() {
        let text = "```\n{\"name\":\"read_file\",\"arguments\":{\"path\":\"/x\"}}\n```";
        let calls = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read_file");
    }

    #[test]
    fn parses_tool_call_missing_closing_tag() {
        // Regression for 2026-07-11: live runs of
        // scripts/hera_diagnose_incident.sh against genesis's real model
        // showed the model, when told to "copy this exact <tool_call>{...}
        // </tool_call> format", reproducing the open tag + a complete valid
        // JSON object but never emitting `</tool_call>` at all (3/6 runs,
        // reproducible). Before this fix the whole line leaked verbatim into
        // the "final answer" instead of executing the call.
        let text = r#"<tool_call>{"name":"read_pm2_logs","arguments":{"service_name":"hera-core","lines":100,"log_type":"error"}}}"#;

        let calls = parse_tool_calls(text);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read_pm2_logs");
        assert_eq!(
            calls[0]
                .arguments
                .get("service_name")
                .and_then(|value| value.as_str()),
            Some("hera-core")
        );
    }

    #[test]
    fn missing_closing_tag_without_valid_json_yields_no_call() {
        // Sanity check for the recovery path: if what follows the open tag
        // isn't valid/balanced JSON at all, extract_balanced_json_object must
        // return None rather than panicking or fabricating a call.
        let text = "<tool_call>not even close to json, just rambling";
        let calls = parse_tool_calls(text);
        assert!(calls.is_empty());
    }
}
