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
use crate::ai::{ChatMessage, ChatRequest, ChatUsage, LLMEngine, MessageContent};
use std::sync::Arc;
use tokio::net::UnixStream;

/// Default hard cap on tool→observe→tool rounds. Override with
/// `HERA_AGENTIC_MAX_ITERS`. Kept generous: a real coding task (read → edit →
/// build → read error → fix → build) easily needs a dozen rounds.
const DEFAULT_MAX_ITERS: usize = 25;

/// How many times a repeated (no-progress) tool call earns a corrective nudge
/// before the loop gives up. Closing immediately on the first repeat killed
/// legitimate debugging (the model retries a near-miss edit); a nudge to change
/// approach lets weak local models recover.
const MAX_NOPROGRESS_NUDGES: usize = 2;

/// Consecutive rounds of the SAME tool name (ignoring arguments) — restricted to
/// passive/read-only rounds, see `round_is_passive` — before the loop treats it
/// as no-progress, same as an exact-signature repeat.
///
/// Diagnosed 2026-07-12 against a real production hang (genesis, hera-core,
/// `stop_reason=error` at iterations=18): a weak local model dodged the
/// exact-signature repeat guard below by trivially varying one argument each
/// round — `read_pm2_logs` called 10 rounds in a row against the SAME service
/// with only `lines`/`log_type` changing (100→50→20→10→10→20→20→20→20). Each
/// round looked "new" to `calls_signature` (exact string match), so the guard
/// never fired past the first pair; the loop ground on, appending each round's
/// raw log dump to the context, until per-round latency grew from ~1s to 70s+
/// and the local inference engine itself failed to answer (connection error to
/// the llama.cpp server) — the loop's actual failure mode was an unbounded,
/// never-pruned context, not the tool logic. This catches the "same tool,
/// cosmetically different args, no real state change" class without touching
/// the legitimate edit→verify→edit debug loop the exact guard already protects
/// (a round containing an edit/verify tool always resets this streak to 0).
const SAME_TOOL_STREAK_LIMIT: usize = 4;

/// Tools that mutate source. After one of these succeeds, the agent should
/// verify before declaring the task done.
const EDIT_TOOLS: &[&str] = &["edit_file", "write_file"];
/// Tools whose green result clears the "needs verification" flag (compiles OK).
const VERIFY_TOOLS: &[&str] = &["cargo_check", "cargo_test", "pytest"];
/// Tools whose green result is a STRONG signal the task is actually done (tests
/// pass, not merely compiles). Only these trigger the efficiency early-close, so
/// "verified" never fires on `cargo_check` alone (compiles ≠ correct).
pub const VERIFY_CLOSE_TOOLS: &[&str] = &["cargo_test", "pytest"];

/// Token usage summed across every turn of an agentic loop run (the engine
/// reports usage per-turn; a single loop run can take many turns, so this is
/// a running total, not a single response's `ChatUsage`).
#[derive(Debug, Clone, Copy, Default)]
pub struct LoopUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl LoopUsage {
    fn add(&mut self, usage: &Option<ChatUsage>) {
        if let Some(usage) = usage {
            self.prompt_tokens += usage.prompt_tokens;
            self.completion_tokens += usage.completion_tokens;
            self.total_tokens += usage.total_tokens;
        }
    }

    fn plus(&self, usage: &Option<ChatUsage>) -> LoopUsage {
        let mut merged = *self;
        merged.add(usage);
        merged
    }
}

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
    /// Summed token usage across all turns. Previously always zero — the loop
    /// discarded per-turn `ChatResponse.usage` entirely (see hera_usage_events
    /// diagnostic 2026-07-13: every claude_code row had real latency but
    /// total_tokens=0).
    pub usage: LoopUsage,
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

/// Whether this is an interactive operator CLI surface (coding or ops) that
/// should run the agentic loop on the STREAMING path with live tool_status.
/// Streaming is a production hot path (widgets stream for fast token UX); gating
/// the loop to these surfaces keeps the loop OUT of widget streaming so their
/// behaviour is unchanged. Non-streaming `generate` still runs the loop globally.
pub fn is_agentic_cli_surface(parsed: &ParsedPayload) -> bool {
    let route = parsed.route_profile_id.to_ascii_lowercase();
    let app = parsed.app_name.to_ascii_lowercase();
    is_coding_context(parsed) || route == "ops" || app == "ops"
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

/// Whether a round's results include a successful source edit.
fn round_did_edit(results: &[(String, bool)]) -> bool {
    results
        .iter()
        .any(|(name, ok)| *ok && EDIT_TOOLS.contains(&name.as_str()))
}

/// Whether a round's results include a green TEST run (cargo_test/pytest) — the
/// strong signal used for the efficiency early-close. Deliberately excludes
/// `cargo_check`: compiling is not the same as tests passing.
fn round_passed_tests(results: &[(String, bool)]) -> bool {
    results
        .iter()
        .any(|(name, ok)| *ok && VERIFY_CLOSE_TOOLS.contains(&name.as_str()))
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

/// Whether every call in a round is passive — NOT an edit or verification tool.
/// A round that edits or verifies is real progress and must never count toward
/// the same-tool-name "fishing" streak (`SAME_TOOL_STREAK_LIMIT`); only rounds
/// of pure re-reading/re-diagnosing without acting on it should.
fn round_is_passive(calls: &[ToolCall]) -> bool {
    calls.iter().all(|call| {
        !EDIT_TOOLS.contains(&call.name.as_str()) && !VERIFY_TOOLS.contains(&call.name.as_str())
    })
}

/// Tool-NAME-only signature of a round (arguments ignored) — coarser than
/// `calls_signature`, used to detect "same tool, different-looking args, no
/// real progress" streaks the exact-signature guard cannot see.
fn tool_names_signature(calls: &[ToolCall]) -> String {
    calls
        .iter()
        .map(|call| call.name.as_str())
        .collect::<Vec<_>>()
        .join(",")
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
    // When streaming (CLI coding/ops), a `tool_status` event is emitted per round
    // so the operator sees the tools execute live. Non-streaming callers pass None.
    mut status_stream: Option<&mut UnixStream>,
) -> AgenticLoopOutcome {
    let max = max_iters();
    // Wave 3 observability (docs/HERA_OBSERVABILITY_WAVE3_INFLIGHT.md): a long
    // loop previously emitted nothing until it finished, indistinguishable from
    // a hang from outside. `loop_started` anchors the elapsed time in the
    // per-iteration log lines below; `trace_id` keys the in-flight registry so
    // `hera_inflight` can report iteration/tool progress while this runs.
    let loop_started = std::time::Instant::now();
    let trace_id = parsed.trace_id.clone();
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
    let mut usage = LoopUsage::default();
    let mut last_model = String::new();
    let mut last_origin = "local".to_string();
    let mut last_text = String::new();
    let mut last_signature: Option<String> = None;
    // See SAME_TOOL_STREAK_LIMIT: tracks consecutive passive rounds using the
    // same tool name (regardless of arguments), to catch a model that dodges
    // the exact-signature repeat guard by trivially varying one argument.
    let mut last_tool_names: Option<String> = None;
    let mut same_tool_streak: usize = 0;
    // Verify-before-done gate: track whether files were edited without a
    // subsequent green verification, and whether we've already nudged once.
    let mut edited_pending = false;
    let mut nudged = false;
    let mut no_progress_nudges = 0usize;
    // Whether any edit has succeeded in this run. Used to close the loop once the
    // work has been verified green, without firing on a model that merely runs a
    // verification BEFORE editing (e.g. a project that already compiles).
    let mut ever_edited = false;
    // Required-tools hard gate: caller declared (via `required_tools` on the
    // payload) that specific tools MUST succeed before the loop is allowed to
    // close with a plain answer. Diagnosed gap: the local model sometimes
    // explores a task extensively (20+ read/grep/glob calls) and then declares
    // itself "done" with a text-only answer, never having called the write it
    // was explicitly asked to perform — no amount of prompt wording reliably
    // prevented this. Unlike the verify gate above (opt-in, unproven), this one
    // has an explicit caller-declared contract, so it defaults ON whenever
    // `required_tools` is non-empty.
    let mut required_tools_done: std::collections::HashSet<String> =
        std::collections::HashSet::new();
    let mut required_tool_nudged = false;

    for iter in 0..max {
        // Per-iteration START log + registry update — this is what kills the
        // "looks hung" problem: `pm2 logs hera-core` now shows the loop
        // advancing instead of going silent until the terminal state.
        tracing::info!(
            "🔁 [Hera] iter {}/{} tool=- elapsed={}ms",
            iter + 1,
            max,
            loop_started.elapsed().as_millis()
        );
        super::inflight::set_iteration(&trace_id, (iter + 1) as u32, max as u32);

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
                    usage,
                };
            }
        };

        usage.add(&resp.usage);
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
                usage,
            };
        };
        let content = choice.message.content.unwrap_or_default();
        last_text = content.clone();

        let calls = extract_tool_calls(&content, &choice.message.tool_calls);
        if calls.is_empty() {
            // Required-tools hard gate: the model wants to give a plain answer,
            // but the caller declared a tool (e.g. write_file) that MUST succeed
            // first and it never did. Nudge once, explicitly naming the missing
            // tool(s) — then give up with a distinct stop_reason so the caller
            // (which should verify the actual artifact, e.g. checking the file
            // exists — never trust stop_reason alone) can tell "declared done
            // without doing it" apart from a normal "done".
            let missing_required: Vec<&String> = parsed
                .required_tools
                .iter()
                .filter(|required| !required_tools_done.contains(*required))
                .collect();
            if !missing_required.is_empty() && !required_tool_nudged {
                required_tool_nudged = true;
                let missing_list = missing_required
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                tracing::info!(
                    "🔁 [Hera Loop] required-tools gate fired at iter {} — missing: {}",
                    iter,
                    missing_list
                );
                req.messages.push(ChatMessage {
                    role: "assistant".to_string(),
                    content: MessageContent::Text(content),
                });
                req.messages.push(ChatMessage {
                    role: "user".to_string(),
                    content: MessageContent::Text(format!(
                        "Your task is NOT complete: you must call the following tool(s) before finishing: {missing_list}. Do not explain or summarize — emit the <tool_call> for it now, with the actual content/arguments needed."
                    )),
                });
                continue;
            }
            if !missing_required.is_empty() {
                tracing::warn!(
                    "🔁 [Hera Loop] required-tools gate: giving up after nudge, still missing: {}",
                    missing_required
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                return AgenticLoopOutcome {
                    result_text: content,
                    model: last_model,
                    origin: last_origin,
                    executed_calls_json,
                    iterations: iter + 1,
                    stop_reason: "required_tool_missing",
                    usage,
                };
            }
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
                usage,
            };
        }

        // Coarser, name-only streak (see SAME_TOOL_STREAK_LIMIT doc comment):
        // computed BEFORE the exact-signature check so it also counts the
        // rounds the exact check already catches (those are still "the same
        // tool name", just also identical args).
        let passive_round = round_is_passive(&calls);
        let names_only = tool_names_signature(&calls);
        if passive_round && last_tool_names.as_deref() == Some(names_only.as_str()) {
            same_tool_streak += 1;
        } else {
            same_tool_streak = usize::from(passive_round);
        }
        last_tool_names = Some(names_only.clone());
        let stale_same_tool = passive_round && same_tool_streak >= SAME_TOOL_STREAK_LIMIT;

        let signature = calls_signature(&calls);
        let exact_repeat = last_signature.as_deref() == Some(signature.as_str());
        last_signature = Some(signature);
        let no_progress = exact_repeat || stale_same_tool;

        tracing::info!(
            "🔁 [Hera Loop] iter {} executing {} tool call(s){}",
            iter,
            calls.len(),
            if exact_repeat {
                " (exact repeat)"
            } else if stale_same_tool {
                " (same-tool streak, varying args)"
            } else {
                ""
            }
        );

        let tool_names = calls
            .iter()
            .map(|call| call.name.as_str())
            .collect::<Vec<_>>()
            .join(",");
        tracing::info!(
            "🔁 [Hera] iter {}/{} tool={} elapsed={}ms",
            iter + 1,
            max,
            tool_names,
            loop_started.elapsed().as_millis()
        );
        super::inflight::set_tool(&trace_id, Some(tool_names.as_str()));

        let summary = execute_parsed_tool_calls(&calls, parsed, status_stream.as_deref_mut()).await;
        // A verification (cargo_check/cargo_test/pytest) that succeeded this round
        // means GREEN — the verify tools set success from the process exit code, so
        // a failing test reports success=false (see ai/tools/build_feedback.rs).
        let round_verified_green = round_passed_tests(&summary.executed_results);
        if round_did_edit(&summary.executed_results) {
            ever_edited = true;
        }
        for (name, success) in &summary.executed_results {
            if *success && parsed.required_tools.iter().any(|required| required == name) {
                required_tools_done.insert(name.clone());
            }
        }
        edited_pending = update_edited_pending(edited_pending, &summary.executed_results);
        executed_calls_json.extend(summary.executed_calls_json);
        let outputs = summary.execution_outputs;

        // Hard cap reached → final answer.
        if iter + 1 >= max {
            append_tool_round(&mut req, &content, &outputs, true);
            return close_with_final_answer(
                engine, req, executed_calls_json, last_model, last_origin, outputs, iter + 1,
                "max_iters", usage,
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
                "verified", usage,
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
                    "🔁 [Hera Loop] no-progress recovery nudge {}/{} at iter {} ({})",
                    no_progress_nudges,
                    MAX_NOPROGRESS_NUDGES,
                    iter,
                    if exact_repeat { "exact repeat" } else { "same-tool streak" }
                );
                req.messages.push(ChatMessage {
                    role: "assistant".to_string(),
                    content: MessageContent::Text(content),
                });
                let nudge_text = if exact_repeat {
                    format!(
                        "Tool execution results:{outputs}\n\nYou repeated the same tool call and got the same result — that is not progress. Do NOT repeat it. Re-read the file and the exact error, then try a DIFFERENT fix with different arguments. If an edit_file failed with 'old_string not found', read the file again with read_file and copy the exact text, including indentation, before editing."
                    )
                } else {
                    format!(
                        "Tool execution results:{outputs}\n\nYou have called {names_only} {same_tool_streak} times in a row with only cosmetic argument changes (e.g. a different line count or filter) and never edited or verified anything — that is re-reading the same information, not progress. Stop calling it. Either act on what you already know (edit_file/write_file, then verify) or give your final answer now."
                    )
                };
                req.messages.push(ChatMessage {
                    role: "user".to_string(),
                    content: MessageContent::Text(nudge_text),
                });
                continue;
            }
            append_tool_round(&mut req, &content, &outputs, true);
            return close_with_final_answer(
                engine, req, executed_calls_json, last_model, last_origin, outputs, iter + 1,
                "no_progress", usage,
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
        usage,
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
    usage: LoopUsage,
) -> AgenticLoopOutcome {
    match engine.generate_content(req).await {
        Ok(resp) => {
            let usage = usage.plus(&resp.usage);
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
                usage,
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
                usage,
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
            required_tools: Vec::new(),
        }
    }

    fn test_parsed_with_required_tools(required_tools: Vec<&str>) -> ParsedPayload {
        ParsedPayload {
            required_tools: required_tools.into_iter().map(String::from).collect(),
            ..test_parsed()
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
            app: None,
            priority: None,
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
        let outcome = run_agentic_loop(&engine, base_request(), &test_parsed(), None).await;
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
        let outcome = run_agentic_loop(&engine, base_request(), &test_parsed(), None).await;
        assert_eq!(outcome.stop_reason, "done");
        assert_eq!(outcome.iterations, 2);
        // Reaching the round-2 scripted answer ("conclusion") proves the loop
        // re-fed the tool result and re-generated — the core of Fase 1. The call
        // is denied (empty permissions) so it is not counted in executed_calls_json.
        assert!(outcome.executed_calls_json.is_empty());
        assert!(outcome.result_text.contains("conclusion"));
    }

    #[tokio::test]
    async fn required_tools_gate_nudges_then_gives_up_if_never_called() {
        // The model never emits a tool call, twice in a row (a text-only
        // "exploration then done" pattern). required_tools=["write_file"] means
        // the loop must NOT accept the first plain answer — it nudges once, and
        // only gives up (with a distinct stop_reason) if the model still never
        // calls the required tool.
        let engine: Arc<dyn LLMEngine + Send + Sync> = Arc::new(ScriptedEngine::new(vec![
            "Here is my summary of the app, no file written.",
            "I have finished analyzing the app.",
        ]));
        let parsed = test_parsed_with_required_tools(vec!["write_file"]);
        let outcome = run_agentic_loop(&engine, base_request(), &parsed, None).await;
        assert_eq!(outcome.stop_reason, "required_tool_missing");
        // 2 generations happened (the nudge bought it a second try) — proves the
        // gate did not accept the first plain answer as "done".
        assert_eq!(outcome.iterations, 2);
    }

    #[tokio::test]
    async fn required_tools_gate_is_inert_when_not_declared() {
        // Same script as above, but no required_tools declared: normal "done"
        // behaviour on the very first plain answer, unaffected by the new gate.
        let engine: Arc<dyn LLMEngine + Send + Sync> =
            Arc::new(ScriptedEngine::new(vec!["Just an answer, no tools."]));
        let outcome = run_agentic_loop(&engine, base_request(), &test_parsed(), None).await;
        assert_eq!(outcome.stop_reason, "done");
        assert_eq!(outcome.iterations, 1);
    }

    #[tokio::test]
    async fn loop_recovers_then_closes_on_persistent_no_progress() {
        // The model emits the SAME tool call every round. The guard nudges it to
        // change approach MAX_NOPROGRESS_NUDGES times before finally giving up —
        // so it takes several identical rounds, not two, to reach the close.
        let same = "<tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":\"/x\"}}</tool_call>";
        let engine: Arc<dyn LLMEngine + Send + Sync> =
            Arc::new(ScriptedEngine::new(vec![same, same, same, same, same]));
        let outcome = run_agentic_loop(&engine, base_request(), &test_parsed(), None).await;
        assert_eq!(outcome.stop_reason, "no_progress");
        // Recovery nudges delayed the close past the first repeat.
        assert!(outcome.iterations >= 2 + MAX_NOPROGRESS_NUDGES);
    }

    #[tokio::test]
    async fn loop_closes_on_same_tool_streak_with_varying_args() {
        // Reproduces the real production hang diagnosed 2026-07-12 (genesis,
        // hera-core, iterations=18 stop_reason=error): the model calls the SAME
        // tool every round but trivially varies one argument each time, so
        // `calls_signature` never matches two consecutive rounds and the exact
        // no-progress guard alone never fires. Without SAME_TOOL_STREAK_LIMIT
        // this script would run all 9 scripted rounds; with it, the loop must
        // close well before exhausting the script (and long before max_iters).
        let calls = [100_u32, 100, 50, 20, 10, 10, 20, 20, 20]
            .iter()
            .map(|lines| {
                format!(
                    "<tool_call>{{\"name\":\"read_pm2_logs\",\"arguments\":{{\"service_name\":\"hera-core\",\"lines\":{lines},\"log_type\":\"error\"}}}}</tool_call>"
                )
            })
            .collect::<Vec<_>>();
        let scripts: Vec<&str> = calls.iter().map(String::as_str).collect();
        let engine: Arc<dyn LLMEngine + Send + Sync> = Arc::new(ScriptedEngine::new(scripts));
        let outcome = run_agentic_loop(&engine, base_request(), &test_parsed(), None).await;
        assert_eq!(outcome.stop_reason, "no_progress");
        // The whole point: it must NOT grind through all 9 scripted rounds (the
        // pre-fix behaviour) — it should close several rounds earlier once the
        // same-tool streak crosses SAME_TOOL_STREAK_LIMIT.
        assert!(
            outcome.iterations < calls.len(),
            "expected an early close, got {} iterations (script had {})",
            outcome.iterations,
            calls.len()
        );
    }

    #[tokio::test]
    async fn same_tool_streak_is_inert_when_edit_or_verify_interleaved() {
        // A legitimate edit→verify→edit debug loop also calls a small set of
        // tools repeatedly, but it must NEVER trip the same-tool-streak guard —
        // only pure re-reading without acting on it should. Script: edit, check
        // (fails), edit, check (fails), edit, check (fails) — 3x each tool,
        // which would exceed SAME_TOOL_STREAK_LIMIT if edit/verify rounds counted.
        let edit = "<tool_call>{\"name\":\"edit_file\",\"arguments\":{\"path\":\"/x\",\"old\":\"a\",\"new\":\"b\"}}</tool_call>";
        let check = "<tool_call>{\"name\":\"cargo_check\",\"arguments\":{}}</tool_call>";
        let engine: Arc<dyn LLMEngine + Send + Sync> = Arc::new(ScriptedEngine::new(vec![
            edit, check, edit, check, edit, check, edit, check,
        ]));
        let outcome = run_agentic_loop(&engine, base_request(), &test_parsed(), None).await;
        // Every tool call is denied (empty permissions in test_parsed), so this
        // never reaches "verified" — the point is only that it is NOT closed as
        // "no_progress" by the same-tool-streak guard despite 4x edit_file and
        // 4x cargo_check across the run (each individually below the streak
        // limit, and never consecutive with itself since they alternate).
        assert_ne!(outcome.stop_reason, "no_progress");
    }

    #[test]
    fn round_is_passive_excludes_edit_and_verify_tools() {
        assert!(round_is_passive(&[ToolCall {
            name: "read_pm2_logs".to_string(),
            arguments: serde_json::json!({}),
        }]));
        assert!(!round_is_passive(&[ToolCall {
            name: "edit_file".to_string(),
            arguments: serde_json::json!({}),
        }]));
        assert!(!round_is_passive(&[ToolCall {
            name: "cargo_check".to_string(),
            arguments: serde_json::json!({}),
        }]));
        // A round mixing a passive read with an edit is NOT passive — any
        // real action in the round disqualifies it.
        assert!(!round_is_passive(&[
            ToolCall { name: "read_file".to_string(), arguments: serde_json::json!({}) },
            ToolCall { name: "write_file".to_string(), arguments: serde_json::json!({}) },
        ]));
    }

    #[test]
    fn tool_names_signature_ignores_arguments() {
        let a = vec![ToolCall {
            name: "read_pm2_logs".to_string(),
            arguments: serde_json::json!({"lines": 100}),
        }];
        let b = vec![ToolCall {
            name: "read_pm2_logs".to_string(),
            arguments: serde_json::json!({"lines": 10, "log_type": "both"}),
        }];
        // Same tool name, very different arguments — the whole point of this
        // signature is that it does NOT distinguish them, unlike calls_signature.
        assert_eq!(tool_names_signature(&a), tool_names_signature(&b));
        assert_ne!(calls_signature(&a), calls_signature(&b));
    }

    // --- Efficiency early-close decision logic (stop_reason = "verified") ---
    // The full loop can't produce a GREEN tool result hermetically (the unit test
    // runs with empty permissions, so every tool is denied — see test_parsed).
    // These tests pin the pure decision predicates the close is built from, which
    // is where the regression risk lives. End-to-end behaviour is covered by the
    // live eval in docs/AVA_CODING_AGENT_PLAN.md.

    #[test]
    fn round_did_edit_detects_only_successful_edits() {
        assert!(round_did_edit(&[("edit_file".to_string(), true)]));
        assert!(round_did_edit(&[("write_file".to_string(), true)]));
        // A failed/denied edit is not an edit.
        assert!(!round_did_edit(&[("edit_file".to_string(), false)]));
        // A verification is not an edit.
        assert!(!round_did_edit(&[("cargo_test".to_string(), true)]));
        assert!(!round_did_edit(&[]));
    }

    #[test]
    fn round_passed_tests_requires_green_test_not_check() {
        // A green test run is the strong signal that closes the loop.
        assert!(round_passed_tests(&[("cargo_test".to_string(), true)]));
        assert!(round_passed_tests(&[("pytest".to_string(), true)]));
        // cargo_check green is NOT sufficient: compiling is not tests passing.
        // This is the guard for the correctness refinement — "verified" must mean
        // tests pass, never merely that the code compiles.
        assert!(!round_passed_tests(&[("cargo_check".to_string(), true)]));
        // A red test does not close.
        assert!(!round_passed_tests(&[("cargo_test".to_string(), false)]));
        assert!(!round_passed_tests(&[]));
    }

    #[test]
    fn edited_pending_tracks_edit_then_green_verification() {
        // Reading a file does not mark work pending.
        assert!(!update_edited_pending(false, &[("read_file".to_string(), true)]));
        // A successful edit marks work pending verification.
        assert!(update_edited_pending(false, &[("edit_file".to_string(), true)]));
        // A green cargo_check clears the pending flag (it compiles).
        assert!(!update_edited_pending(true, &[("cargo_check".to_string(), true)]));
        // A green cargo_test also clears it.
        assert!(!update_edited_pending(true, &[("cargo_test".to_string(), true)]));
        // A failed verification leaves the pending flag set.
        assert!(update_edited_pending(true, &[("cargo_test".to_string(), false)]));
        // Edit then green check in the same round, in order: set then cleared.
        assert!(!update_edited_pending(
            false,
            &[("edit_file".to_string(), true), ("cargo_check".to_string(), true)]
        ));
    }
}
