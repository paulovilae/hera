//! Agentic multi-turn tool loop (Fase 1 — docs/AVA_CODING_AGENT_PLAN.md).
//!
//! Today `handler_generate` runs a single-shot path: it executes ONE batch of
//! tool calls and then does ONE formatting pass — it never re-feeds a tool
//! result back to the model, so Ava cannot do `act → observe error → fix →
//! repeat`. This module adds the missing loop: generate → execute tools →
//! re-inject the results as a new turn → generate again, until the model stops
//! emitting tool calls or a hard budget is hit.
//!
//! It is gated behind `HERA_AGENTIC_LOOP` and only changes behaviour for
//! tool-enabled requests; the bots' current path is untouched when the flag is
//! off. The loop REUSES the existing tool executors (`execute_parsed_tool_calls`)
//! — it does not add or change any tool.
//!
//! Adapted from the bounded agent loop in claw-code `rust/crates/runtime` and
//! opencrust `opencrust-agents` (both MIT).

use super::context::ParsedPayload;
use super::helpers::infer_origin_from_model;
use super::runtime_tools::execute_parsed_tool_calls;
use crate::ai::tool_executor::ToolCall;
use crate::ai::{ChatMessage, ChatRequest, LLMEngine, MessageContent};
use std::sync::Arc;

/// Default hard cap on tool→observe→tool rounds. Override with
/// `HERA_AGENTIC_MAX_ITERS`. Kept generous: a real coding task (read → edit →
/// build → read error → fix → build) easily needs a dozen rounds.
const DEFAULT_MAX_ITERS: usize = 25;

/// How many times a repeated (no-progress) tool call earns a corrective nudge
/// before the loop gives up. Closing immediately on the first repeat killed
/// legitimate debugging (the model retries a near-miss edit); a nudge to change
/// approach lets weak local models recover.
const MAX_NOPROGRESS_NUDGES: usize = 2;

/// Tools that mutate source. After one of these succeeds, the agent should
/// verify before declaring the task done.
const EDIT_TOOLS: &[&str] = &["edit_file", "write_file"];
/// Tools whose green result clears the "needs verification" flag (compiles OK).
const VERIFY_TOOLS: &[&str] = &["cargo_check", "cargo_test", "pytest"];
/// Tools whose green result is a STRONG signal the task is actually done (tests
/// pass, not merely compiles). Only these trigger the efficiency early-close, so
/// "verified" never fires on `cargo_check` alone (compiles ≠ correct).
const VERIFY_CLOSE_TOOLS: &[&str] = &["cargo_test", "pytest"];

/// Outcome of a full agentic loop run.
pub struct AgenticLoopOutcome {
    pub result_text: String,
    pub model: String,
    pub origin: String,
    /// Every tool call executed across all rounds (for the IPC `tool_calls` field).
    pub executed_calls_json: Vec<serde_json::Value>,
    /// How many model turns were taken (1 = answered without any tool).
    pub iterations: usize,
    /// "done" | "verified" | "max_iters" | "no_progress" | "empty" | "error"
    pub stop_reason: &'static str,
}

/// Whether the multi-turn loop is enabled. Off by default (sovereign-safe rollout).
pub fn agentic_loop_enabled() -> bool {
    matches!(
        std::env::var("HERA_AGENTIC_LOOP")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

/// Whether the verify-before-done gate is active. Opt-in (`HERA_VERIFY_GATE`):
/// an eval run on 2026-06-15 did not show it improving pass@1 (the signal was
/// dominated by model run-to-run variance), so it stays off by default until a
/// multi-run eval can prove it helps. See docs/AVA_CODING_AGENT_PLAN.md.
fn verify_gate_enabled() -> bool {
    matches!(
        std::env::var("HERA_VERIFY_GATE")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

/// Sampling temperature for the agentic (coding) path. Coding wants
/// determinism, not creativity: the default generate path uses 0.7, which made
/// eval results swing wildly. Low temperature both improves reliability on code
/// and makes measurement reproducible. Override with `HERA_CODING_TEMP`.
fn coding_temp() -> f32 {
    std::env::var("HERA_CODING_TEMP")
        .ok()
        .and_then(|v| v.trim().parse::<f32>().ok())
        .filter(|t| (0.0..=2.0).contains(t))
        .unwrap_or(0.2)
}

/// Whether this request is a coding / agentic-build context (the `coding` route
/// profile, an `ava_coder`-style agent, or a `coding` app) as opposed to a
/// normal tool-using conversational bot. Only a coding context gets the low
/// deterministic temperature; conversational bots keep their persona tone so
/// enabling the loop platform-wide does not make the tutors/Memo/Vetra robotic.
fn is_coding_context(parsed: &ParsedPayload) -> bool {
    let route = parsed.route_profile_id.to_ascii_lowercase();
    let app = parsed.app_name.to_ascii_lowercase();
    route == "coding"
        || route == "ava_coder"
        || app == "coding"
        || app.contains("coder")
}

fn max_iters() -> usize {
    std::env::var("HERA_AGENTIC_MAX_ITERS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value >= 1)
        .unwrap_or(DEFAULT_MAX_ITERS)
}

/// Extract tool calls from one model turn: both the text-embedded `<tool_call>`
/// form and the structured `choice.message.tool_calls` array. Mirrors the
/// extraction in `handler_generate` so the loop and the legacy path agree.
fn extract_tool_calls(content: &str, structured: &Option<Vec<serde_json::Value>>) -> Vec<ToolCall> {
    let mut calls = crate::ai::tool_executor::parse_tool_calls(content);

    if let Some(tc_array) = structured {
        for tc in tc_array {
            let mut extracted_name = None;
            let mut extracted_args = None;

            if let (Some(name), Some(args)) = (
                tc.get("name").and_then(|n| n.as_str()),
                tc.get("arguments").or_else(|| tc.get("parameters")),
            ) {
                extracted_name = Some(name);
                extracted_args = Some(args);
            } else if let Some(func) = tc.get("function")
                && let (Some(name), Some(args)) = (
                    func.get("name").and_then(|n| n.as_str()),
                    func.get("arguments").or_else(|| func.get("parameters")),
                )
            {
                extracted_name = Some(name);
                extracted_args = Some(args);
            }

            if let (Some(name), Some(args)) = (extracted_name, extracted_args) {
                let mut args_val = args.clone();
                if let Some(s) = args.as_str()
                    && let Ok(parsed_args) = serde_json::from_str(s)
                {
                    args_val = parsed_args;
                }
                calls.push(ToolCall {
                    name: name.to_string(),
                    arguments: args_val,
                });
            }
        }
    }

    calls
}

/// Update the "edited but not yet verified green" flag from a round's executed
/// (tool_name, success) results, in order. A successful edit sets it; a green
/// verification clears it; a failed/denied edit or a red verification leaves it.
fn update_edited_pending(mut pending: bool, results: &[(String, bool)]) -> bool {
    for (name, success) in results {
        if !*success {
            continue;
        }
        if EDIT_TOOLS.contains(&name.as_str()) {
            pending = true;
        } else if VERIFY_TOOLS.contains(&name.as_str()) {
            pending = false;
        }
    }
    pending
}

/// Stable signature of a round's tool calls, used to detect a model that is
/// stuck repeating the same call without making progress.
fn calls_signature(calls: &[ToolCall]) -> String {
    calls
        .iter()
        .map(|call| format!("{}:{}", call.name, call.arguments))
        .collect::<Vec<_>>()
        .join("|")
}

/// Append one completed tool round to the running conversation: the model's raw
/// output (which contained the tool call) as the assistant turn, then the tool
/// results as a user turn. Unlike `prepare_tool_result_followup_request`, the
/// non-closing instruction KEEPS the door open for further tool calls — that is
/// the whole point of the loop. The closing instruction forbids more tools so
/// the model produces a final answer.
fn append_tool_round(
    req: &mut ChatRequest,
    assistant_output: &str,
    execution_outputs: &str,
    closing: bool,
) {
    req.messages.push(ChatMessage {
        role: "assistant".to_string(),
        content: MessageContent::Text(assistant_output.to_string()),
    });

    let instruction = if closing {
        format!(
            "Tool execution results:{execution_outputs}\n\nIMPORTANT: Do NOT call any more tools or emit <tool_call> tags. Give your final answer to the user now, based on these results."
        )
    } else {
        format!(
            "Tool execution results:{execution_outputs}\n\nReview the results. If the task is complete, give your final answer to the user with no tool call. If you still need to act (fix an error, edit another file, run the build again), emit the next <tool_call> now."
        )
    };

    req.messages.push(ChatMessage {
        role: "user".to_string(),
        content: MessageContent::Text(instruction),
    });
}

/// Run the bounded agentic loop. `base_request` already carries the full system
/// prompt (persona + tool schemas) assembled by `prepare_chat_request`.
pub async fn run_agentic_loop(
    engine: &Arc<dyn LLMEngine + Send + Sync>,
    base_request: ChatRequest,
    parsed: &ParsedPayload,
) -> AgenticLoopOutcome {
    let max = max_iters();
    let mut req = base_request;
    // Temperature policy: the loop is now enabled platform-wide so that ANY
    // tool-using bot gains iterative tool use + self-correction of failed
    // queries. But the low deterministic temperature only helps coding — it
    // dries out conversational personas (tutors, Memo, Vetra). So pin the low
    // temp ONLY for a coding context; everyone else keeps their persona's
    // temperature untouched. See docs/AVA_CODING_AGENT_PLAN.md.
    if is_coding_context(parsed) {
        req.temperature = Some(coding_temp());
    }
    let mut executed_calls_json: Vec<serde_json::Value> = Vec::new();
    let mut last_model = String::new();
    let mut last_origin = "local".to_string();
    let mut last_text = String::new();
    let mut last_signature: Option<String> = None;
    // Verify-before-done gate: track whether files were edited without a
    // subsequent green verification, and whether we've already nudged once.
    let mut edited_pending = false;
    let mut nudged = false;
    let mut no_progress_nudges = 0usize;
    // Whether any edit has succeeded in this run. Used to close the loop once the
    // work has been verified green, without firing on a model that merely runs a
    // verification BEFORE editing (e.g. a project that already compiles).
    let mut ever_edited = false;

    for iter in 0..max {
        let resp = match engine.generate_content(req.clone()).await {
            Ok(resp) => resp,
            Err(error) => {
                tracing::error!("🔁 [Hera Loop] inference error at iter {}: {}", iter, error);
                let result_text = if last_text.is_empty() {
                    format!("Error: {error}")
                } else {
                    last_text
                };
                return AgenticLoopOutcome {
                    result_text,
                    model: last_model,
                    origin: "offline".to_string(),
                    executed_calls_json,
                    iterations: iter + 1,
                    stop_reason: "error",
                };
            }
        };

        last_model = resp.model.clone();
        last_origin = infer_origin_from_model(&resp.model).to_string();

        let Some(choice) = resp.choices.into_iter().next() else {
            return AgenticLoopOutcome {
                result_text: last_text,
                model: last_model,
                origin: last_origin,
                executed_calls_json,
                iterations: iter + 1,
                stop_reason: "empty",
            };
        };
        let content = choice.message.content.unwrap_or_default();
        last_text = content.clone();

        let calls = extract_tool_calls(&content, &choice.message.tool_calls);
        if calls.is_empty() {
            // Verify-before-done gate: the model is trying to finish, but it
            // edited files and never confirmed they build/pass. Nudge it once to
            // run a verification before accepting the answer. This targets the
            // observed failure where the model declares "done" on code that
            // doesn't compile. Only fires once to avoid looping forever.
            if verify_gate_enabled() && edited_pending && !nudged {
                nudged = true;
                tracing::info!(
                    "🔁 [Hera Loop] verify-before-done gate fired at iter {} — nudging to verify edits",
                    iter
                );
                req.messages.push(ChatMessage {
                    role: "assistant".to_string(),
                    content: MessageContent::Text(content),
                });
                req.messages.push(ChatMessage {
                    role: "user".to_string(),
                    content: MessageContent::Text(
                        "Before you finish: you edited files but have not confirmed the result builds and passes. Run the appropriate verification now (cargo_check or cargo_test for Rust; run the tests for other languages). If it reports errors, fix them and verify again. Only give your final answer once verification is green."
                            .to_string(),
                    ),
                });
                continue;
            }
            // The model produced a plain answer — the task is done.
            return AgenticLoopOutcome {
                result_text: content,
                model: last_model,
                origin: last_origin,
                executed_calls_json,
                iterations: iter + 1,
                stop_reason: "done",
            };
        }

        let signature = calls_signature(&calls);
        let no_progress = last_signature.as_deref() == Some(signature.as_str());
        last_signature = Some(signature);

        tracing::info!(
            "🔁 [Hera Loop] iter {} executing {} tool call(s){}",
            iter,
            calls.len(),
            if no_progress { " (repeat — closing)" } else { "" }
        );

        let summary = execute_parsed_tool_calls(&calls, parsed, None).await;
        // A verification (cargo_check/cargo_test/pytest) that succeeded this round
        // means GREEN — the verify tools set success from the process exit code, so
        // a failing test reports success=false (see ai/tools/build_feedback.rs).
        let round_edited = summary
            .executed_results
            .iter()
            .any(|(name, ok)| *ok && EDIT_TOOLS.contains(&name.as_str()));
        let round_verified_green = summary
            .executed_results
            .iter()
            .any(|(name, ok)| *ok && VERIFY_CLOSE_TOOLS.contains(&name.as_str()));
        if round_edited {
            ever_edited = true;
        }
        edited_pending = update_edited_pending(edited_pending, &summary.executed_results);
        executed_calls_json.extend(summary.executed_calls_json);
        let outputs = summary.execution_outputs;

        // Hard cap reached → final answer.
        if iter + 1 >= max {
            append_tool_round(&mut req, &content, &outputs, true);
            return close_with_final_answer(
                engine, req, executed_calls_json, last_model, last_origin, outputs, iter + 1,
                "max_iters",
            )
            .await;
        }

        // Efficiency close: once we have edited something and a verification just
        // ran green with no edit left unverified, the task is in a confirmed-good
        // state — stop instead of letting the model re-run the same verification.
        // Observed without this: 7 redundant cargo_test calls after the fix was
        // already green (11 iters / 190s for a one-line fix). Guarded by
        // `ever_edited` so a pre-edit verification (a project that already builds)
        // does not close the loop before the real work happens.
        if ever_edited && round_verified_green && !edited_pending {
            tracing::info!(
                "🔁 [Hera Loop] verified-green close at iter {} (edited + green + nothing pending)",
                iter
            );
            append_tool_round(&mut req, &content, &outputs, true);
            return close_with_final_answer(
                engine, req, executed_calls_json, last_model, last_origin, outputs, iter + 1,
                "verified",
            )
            .await;
        }

        if no_progress {
            // The model repeated the same call with the same result. Instead of
            // giving up, nudge it to change approach — weak local models often
            // recover once told explicitly to stop repeating. Give up only after
            // a few such nudges.
            if no_progress_nudges < MAX_NOPROGRESS_NUDGES {
                no_progress_nudges += 1;
                tracing::info!(
                    "🔁 [Hera Loop] no-progress recovery nudge {}/{} at iter {}",
                    no_progress_nudges,
                    MAX_NOPROGRESS_NUDGES,
                    iter
                );
                req.messages.push(ChatMessage {
                    role: "assistant".to_string(),
                    content: MessageContent::Text(content),
                });
                req.messages.push(ChatMessage {
                    role: "user".to_string(),
                    content: MessageContent::Text(format!(
                        "Tool execution results:{outputs}\n\nYou repeated the same tool call and got the same result — that is not progress. Do NOT repeat it. Re-read the file and the exact error, then try a DIFFERENT fix with different arguments. If an edit_file failed with 'old_string not found', read the file again with read_file and copy the exact text, including indentation, before editing."
                    )),
                });
                continue;
            }
            append_tool_round(&mut req, &content, &outputs, true);
            return close_with_final_answer(
                engine, req, executed_calls_json, last_model, last_origin, outputs, iter + 1,
                "no_progress",
            )
            .await;
        }

        append_tool_round(&mut req, &content, &outputs, false);
    }

    // Unreachable in practice (the loop returns on the last iteration), but keep
    // a defined outcome rather than panicking.
    AgenticLoopOutcome {
        result_text: last_text,
        model: last_model,
        origin: last_origin,
        executed_calls_json,
        iterations: max,
        stop_reason: "max_iters",
    }
}

/// Run the final no-tools pass and package the outcome. Falls back to the raw
/// tool output if the model fails to produce a closing answer.
#[allow(clippy::too_many_arguments)]
async fn close_with_final_answer(
    engine: &Arc<dyn LLMEngine + Send + Sync>,
    req: ChatRequest,
    executed_calls_json: Vec<serde_json::Value>,
    fallback_model: String,
    fallback_origin: String,
    fallback_outputs: String,
    iterations: usize,
    stop_reason: &'static str,
) -> AgenticLoopOutcome {
    match engine.generate_content(req).await {
        Ok(resp) => {
            let model = resp.model.clone();
            let origin = infer_origin_from_model(&model).to_string();
            let text = resp
                .choices
                .into_iter()
                .next()
                .and_then(|choice| choice.message.content)
                .filter(|text| !text.trim().is_empty())
                .unwrap_or_else(|| fallback_outputs.trim().to_string());
            AgenticLoopOutcome {
                result_text: text,
                model,
                origin,
                executed_calls_json,
                iterations: iterations + 1,
                stop_reason,
            }
        }
        Err(error) => {
            tracing::error!("🔁 [Hera Loop] closing pass failed: {}", error);
            AgenticLoopOutcome {
                result_text: fallback_outputs.trim().to_string(),
                model: fallback_model,
                origin: fallback_origin,
                executed_calls_json,
                iterations: iterations + 1,
                stop_reason,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::{
        ChatChoice, ChatResponse, ChatResponseMessage, ChatStreamResponse, InferenceError,
    };
    use crate::ipc::context::{ParsedPayload, context_budget_for_mode};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::mpsc;

    /// Mock engine that returns a scripted sequence of assistant contents, one
    /// per `generate_content` call.
    struct ScriptedEngine {
        scripts: Vec<String>,
        calls: AtomicUsize,
    }

    impl ScriptedEngine {
        fn new(scripts: Vec<&str>) -> Self {
            Self {
                scripts: scripts.into_iter().map(String::from).collect(),
                calls: AtomicUsize::new(0),
            }
        }
    }

    #[async_trait::async_trait]
    impl LLMEngine for ScriptedEngine {
        async fn generate_content(
            &self,
            _req: ChatRequest,
        ) -> Result<ChatResponse, InferenceError> {
            let idx = self.calls.fetch_add(1, Ordering::SeqCst);
            let content = self
                .scripts
                .get(idx)
                .cloned()
                .unwrap_or_else(|| "Final answer.".to_string());
            Ok(ChatResponse {
                id: "resp".to_string(),
                object: "chat.completion".to_string(),
                created: 0,
                model: "mock-local-model".to_string(),
                choices: vec![ChatChoice {
                    index: 0,
                    message: ChatResponseMessage {
                        role: "assistant".to_string(),
                        content: Some(content),
                        tool_calls: None,
                    },
                    finish_reason: Some("stop".to_string()),
                }],
                usage: None,
            })
        }

        async fn generate_stream(
            &self,
            _req: ChatRequest,
        ) -> Result<mpsc::Receiver<Result<ChatStreamResponse, InferenceError>>, InferenceError>
        {
            let (_tx, rx) = mpsc::channel(1);
            Ok(rx)
        }
    }

    fn test_parsed() -> ParsedPayload {
        // Empty permissions → any tool call is denied by execute_parsed_tool_calls,
        // which returns a graceful "Not permitted" string WITHOUT touching the real
        // tool backends. That keeps this unit test hermetic while still exercising
        // the full loop control flow (parse → execute → re-feed → re-generate).
        ParsedPayload {
            prompt: "do the thing".to_string(),
            assistant_last: None,
            recent_messages: Vec::new(),
            permissions: Vec::new(),
            persona_path: String::new(),
            app_name: String::new(),
            language_hint: String::new(),
            trace_id: String::new(),
            session_id: String::new(),
            chat_id: String::new(),
            app_id: String::new(),
            sender_name: String::new(),
            page_url: String::new(),
            page_title: String::new(),
            page_context: String::new(),
            route_profile_id: String::new(),
            expected_persona_path: String::new(),
            persona_drift: false,
            context_budget: context_budget_for_mode("standard", false),
            reasoning_effort: "medium".to_string(),
        }
    }

    fn base_request() -> ChatRequest {
        ChatRequest {
            model: "hera-local-model".to_string(),
            vision_model: None,
            tts_model: None,
            stt_model: None,
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: MessageContent::Text("do the thing".to_string()),
            }],
            temperature: None,
            max_tokens: None,
            top_p: None,
            top_k: None,
            presence_penalty: None,
            frequency_penalty: None,
            repeat_penalty: None,
            seed: None,
            stop: None,
            endpoint: None,
            api_key: None,
            provider: Some("local".to_string()),
            stream: Some(false),
            nsfw: None,
            tools: None,
            tool_choice: None,
            reasoning_effort: None,
            response_format: None,
        }
    }

    #[test]
    fn extract_tool_calls_reads_text_and_structured() {
        let text = "<tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":\"/x\"}}</tool_call>";
        let calls = extract_tool_calls(text, &None);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read_file");

        let structured = Some(vec![serde_json::json!({
            "name": "grep_search",
            "arguments": {"pattern": "fn main"}
        })]);
        let calls = extract_tool_calls("", &structured);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "grep_search");
    }

    #[test]
    fn edited_pending_tracks_edit_then_verify() {
        let edit = vec![("edit_file".to_string(), true)];
        let green = vec![("cargo_check".to_string(), true)];
        let red = vec![("cargo_check".to_string(), false)];
        // a successful edit raises the flag
        assert!(update_edited_pending(false, &edit));
        // a green verify clears it
        assert!(!update_edited_pending(true, &green));
        // a red verify leaves it raised
        assert!(update_edited_pending(true, &red));
        // a denied/failed edit does not raise it
        assert!(!update_edited_pending(false, &[("edit_file".to_string(), false)]));
        // edit then green-verify in the same round ends clear
        let both = vec![("edit_file".to_string(), true), ("cargo_check".to_string(), true)];
        assert!(!update_edited_pending(false, &both));
    }

    #[test]
    fn calls_signature_is_stable_and_distinct() {
        let a = vec![ToolCall {
            name: "read_file".to_string(),
            arguments: serde_json::json!({"path": "/x"}),
        }];
        let b = vec![ToolCall {
            name: "read_file".to_string(),
            arguments: serde_json::json!({"path": "/y"}),
        }];
        assert_eq!(calls_signature(&a), calls_signature(&a.clone()));
        assert_ne!(calls_signature(&a), calls_signature(&b));
    }

    #[tokio::test]
    async fn loop_returns_immediately_when_no_tool_call() {
        let engine: Arc<dyn LLMEngine + Send + Sync> =
            Arc::new(ScriptedEngine::new(vec!["Just an answer, no tools."]));
        let outcome = run_agentic_loop(&engine, base_request(), &test_parsed()).await;
        assert_eq!(outcome.stop_reason, "done");
        assert_eq!(outcome.iterations, 1);
        assert!(outcome.executed_calls_json.is_empty());
        assert!(outcome.result_text.contains("no tools"));
    }

    #[tokio::test]
    async fn loop_executes_tool_then_finishes_on_plain_answer() {
        // Round 1: emit a tool call. Round 2: after seeing the (denied) result,
        // give a plain final answer → loop ends with "done".
        let engine: Arc<dyn LLMEngine + Send + Sync> = Arc::new(ScriptedEngine::new(vec![
            "<tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":\"/x\"}}</tool_call>",
            "All done: the file path was rejected, here is my conclusion.",
        ]));
        let outcome = run_agentic_loop(&engine, base_request(), &test_parsed()).await;
        assert_eq!(outcome.stop_reason, "done");
        assert_eq!(outcome.iterations, 2);
        // Reaching the round-2 scripted answer ("conclusion") proves the loop
        // re-fed the tool result and re-generated — the core of Fase 1. The call
        // is denied (empty permissions) so it is not counted in executed_calls_json.
        assert!(outcome.executed_calls_json.is_empty());
        assert!(outcome.result_text.contains("conclusion"));
    }

    #[tokio::test]
    async fn loop_recovers_then_closes_on_persistent_no_progress() {
        // The model emits the SAME tool call every round. The guard nudges it to
        // change approach MAX_NOPROGRESS_NUDGES times before finally giving up —
        // so it takes several identical rounds, not two, to reach the close.
        let same = "<tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":\"/x\"}}</tool_call>";
        let engine: Arc<dyn LLMEngine + Send + Sync> =
            Arc::new(ScriptedEngine::new(vec![same, same, same, same, same]));
        let outcome = run_agentic_loop(&engine, base_request(), &test_parsed()).await;
        assert_eq!(outcome.stop_reason, "no_progress");
        // Recovery nudges delayed the close past the first repeat.
        assert!(outcome.iterations >= 2 + MAX_NOPROGRESS_NUDGES);
    }
}
