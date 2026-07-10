use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
use tokio::time::{Duration, sleep};

use super::agentic_loop::VERIFY_CLOSE_TOOLS;
use super::helpers::{save_agent_run_summary, save_open_loop_memory};
use super::types::{HandlerOutcome, IpcPayload, IpcResponse, IpcState, send_ipc_response};
use crate::ai::tool_executor::load_agent_artifact;

const HERA_SOCKET: &str = "/tmp/hera-core.sock";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DelegationAgentSpec {
    agent: String,
    prompt: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DelegateTaskRequest {
    app: Option<String>,
    trace_id: Option<String>,
    session_id: Option<String>,
    chat_id: Option<String>,
    goal: String,
    strategy: Option<String>,
    wait_for_completion: Option<bool>,
    agents: Vec<DelegationAgentSpec>,
    /// Opt-in: evaluate the `goal` after each delegation pass and re-run until it
    /// is satisfied or the pass budget is exhausted (the "open loop" behaviour).
    /// Absent/false → today's single-pass delegation, byte-for-byte unchanged.
    #[serde(default)]
    goal_loop: Option<bool>,
    /// Override for `HERA_GOAL_LOOP_MAX_PASSES` (default 5). Only used when
    /// `goal_loop` is engaged.
    #[serde(default)]
    max_passes: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AgentRunItem {
    agent: String,
    status: String,
    output: Option<String>,
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AgentRunRecord {
    run_id: String,
    app: String,
    trace_id: String,
    session_id: String,
    chat_id: String,
    goal: String,
    strategy: String,
    status: String,
    created_at_ms: u64,
    updated_at_ms: u64,
    route_profile: String,
    agent_specs: Vec<DelegationAgentSpec>,
    agents: Vec<AgentRunItem>,
    aggregate_result: Option<String>,
    recommendation: Option<String>,
    /// Goal-loop metadata (None for classic single-pass runs). `goal_passes` is
    /// how many delegation passes ran before the loop stopped; `goal_judge_reason`
    /// is the last verdict's reason (from verify-close or the LLM judge).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    goal_passes: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    goal_judge_reason: Option<String>,
}

#[derive(Debug)]
struct AgentRunHandle {
    abort_handles: Vec<tokio::task::AbortHandle>,
}

fn run_registry() -> &'static Mutex<HashMap<String, AgentRunRecord>> {
    static REGISTRY: OnceLock<Mutex<HashMap<String, AgentRunRecord>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn run_handles() -> &'static Mutex<HashMap<String, AgentRunHandle>> {
    static HANDLES: OnceLock<Mutex<HashMap<String, AgentRunHandle>>> = OnceLock::new();
    HANDLES.get_or_init(|| Mutex::new(HashMap::new()))
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_millis() as u64)
        .unwrap_or(0)
}

fn new_run_id() -> String {
    format!("agent_run_{}", now_ms())
}

async fn run_agent_via_hera_ipc(persona: String, prompt: String) -> Result<String, String> {
    let mut stream = UnixStream::connect(HERA_SOCKET)
        .await
        .map_err(|error| format!("failed to connect to Hera IPC: {error}"))?;

    let request = serde_json::json!({
        "action": "generate",
        "payload": {
            "app": "hera",
            "messages": [
                { "role": "system", "content": persona },
                { "role": "user", "content": prompt }
            ],
            "temperature": 0.2,
            "max_tokens": 1200,
            "permissions": []
        }
    });

    let payload = format!("{}\n", request);
    stream
        .write_all(payload.as_bytes())
        .await
        .map_err(|error| format!("failed to write Hera IPC request: {error}"))?;

    stream
        .shutdown()
        .await
        .map_err(|error| format!("failed to shutdown Hera IPC write half: {error}"))?;

    let mut response = String::new();
    stream
        .read_to_string(&mut response)
        .await
        .map_err(|error| format!("failed to read Hera IPC response: {error}"))?;

    for line in response.lines() {
        let Ok(message) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        match message.get("status").and_then(|value| value.as_str()) {
            Some("success") => {
                if let Some(result) = message
                    .pointer("/data/result")
                    .and_then(|value| value.as_str())
                {
                    return Ok(result.to_string());
                }
            }
            Some("error") => {
                return Err(message
                    .pointer("/data/error")
                    .and_then(|value| value.as_str())
                    .unwrap_or("unknown Hera IPC error")
                    .to_string());
            }
            _ => {}
        }
    }

    Err("No content in Hera IPC response".to_string())
}

fn update_run(run_id: &str, mutate: impl FnOnce(&mut AgentRunRecord)) {
    let Ok(mut registry) = run_registry().lock() else {
        tracing::error!("agent run registry lock poisoned");
        return;
    };
    if let Some(record) = registry.get_mut(run_id) {
        mutate(record);
        record.updated_at_ms = now_ms();
    }
}

fn summarize_run(record: &AgentRunRecord) -> String {
    let mut sections = Vec::new();
    for item in &record.agents {
        let header = format!("--- {} ({}) ---", item.agent.to_uppercase(), item.status);
        let body = item
            .output
            .clone()
            .or(item.error.clone())
            .unwrap_or_else(|| "No output".to_string());
        sections.push(format!("{header}\n{body}"));
    }
    sections.join("\n\n")
}

fn derive_recommendation(record: &AgentRunRecord) -> Option<String> {
    let failed = record
        .agents
        .iter()
        .filter(|item| item.status != "completed")
        .count();
    if failed > 0 {
        Some(format!(
            "{} agent(s) failed; inspect outputs before automatic promotion.",
            failed
        ))
    } else {
        Some(
            "Delegation completed cleanly; candidate for Memento run summary persistence."
                .to_string(),
        )
    }
}

async fn persist_run_summary(record: &AgentRunRecord) {
    save_agent_run_summary(serde_json::json!({
        "app_id": record.app,
        "run_id": record.run_id,
        "route_profile": record.route_profile,
        "trace_id": record.trace_id,
        "session_id": record.session_id,
        "chat_id": record.chat_id,
        "goal": record.goal,
        "status": record.status,
        "recommendation": record.recommendation,
        "aggregate_result": record.aggregate_result,
        "summary": summarize_run(record),
        "agents": record.agents,
        "agent_specs": record.agent_specs,
    }))
    .await;
}

fn is_terminal_status(status: &str) -> bool {
    matches!(status, "completed" | "failed" | "cancelled" | "goal_unmet")
}

fn get_run(run_id: &str) -> Option<AgentRunRecord> {
    run_registry()
        .lock()
        .ok()
        .and_then(|registry| registry.get(run_id).cloned())
}

/// Run one delegation pass (all agents in parallel) to completion.
///
/// `intermediate` is set by the goal loop: when true, the pass finalizes to the
/// NON-terminal `evaluating` status instead of `completed`, so a client polling
/// `await_agent_run` keeps waiting while the goal is judged and (possibly) more
/// passes run — otherwise it would catch the transient per-pass `completed` and
/// return a false "done" mid-loop. Classic single-pass callers pass false and
/// get the unchanged terminal `completed`.
async fn spawn_delegate_run(
    request: DelegateTaskRequest,
    existing_run_id: Option<String>,
    intermediate: bool,
) -> AgentRunRecord {
    let run_id = existing_run_id.unwrap_or_else(new_run_id);
    let app = request.app.clone().unwrap_or_else(|| "unknown".to_string());
    let strategy = request
        .strategy
        .clone()
        .unwrap_or_else(|| "parallel".to_string());
    let mut record = AgentRunRecord {
        run_id: run_id.clone(),
        app: app.clone(),
        trace_id: request.trace_id.unwrap_or_default(),
        session_id: request.session_id.unwrap_or_default(),
        chat_id: request.chat_id.unwrap_or_default(),
        goal: request.goal.clone(),
        strategy,
        status: "running".to_string(),
        created_at_ms: now_ms(),
        updated_at_ms: now_ms(),
        route_profile: format!("{}_delegation", app),
        agent_specs: request.agents.clone(),
        agents: request
            .agents
            .iter()
            .map(|spec| AgentRunItem {
                agent: spec.agent.clone(),
                status: "queued".to_string(),
                output: None,
                error: None,
            })
            .collect(),
        aggregate_result: None,
        recommendation: None,
        goal_passes: None,
        goal_judge_reason: None,
    };

    {
        let Ok(mut registry) = run_registry().lock() else {
            tracing::error!("agent run registry lock poisoned");
            return record;
        };
        registry.insert(run_id.clone(), record.clone());
    }

    let mut join_handles = Vec::new();
    let mut abort_handles = Vec::new();
    for spec in request.agents {
        let run_id_cloned = run_id.clone();
        let agent_name = spec.agent.clone();
        let sub_prompt = spec.prompt.unwrap_or_else(|| request.goal.clone());
        let join = tokio::spawn(async move {
            update_run(&run_id_cloned, |item| {
                if let Some(agent) = item
                    .agents
                    .iter_mut()
                    .find(|agent| agent.agent == agent_name)
                {
                    agent.status = "running".to_string();
                }
            });

            let artifact = load_agent_artifact(&agent_name);
            match run_agent_via_hera_ipc(artifact.persona, sub_prompt).await {
                Ok(output) => {
                    update_run(&run_id_cloned, |item| {
                        if let Some(agent) = item
                            .agents
                            .iter_mut()
                            .find(|agent| agent.agent == agent_name)
                        {
                            agent.status = "completed".to_string();
                            agent.output = Some(output);
                            agent.error = None;
                        }
                    });
                }
                Err(error) => {
                    update_run(&run_id_cloned, |item| {
                        if let Some(agent) = item
                            .agents
                            .iter_mut()
                            .find(|agent| agent.agent == agent_name)
                        {
                            agent.status = "failed".to_string();
                            agent.error = Some(error);
                            agent.output = None;
                        }
                    });
                }
            }
        });
        abort_handles.push(join.abort_handle());
        join_handles.push(join);
    }

    {
        let Ok(mut handles) = run_handles().lock() else {
            tracing::error!("agent run handles lock poisoned");
            return record;
        };
        handles.insert(run_id.clone(), AgentRunHandle { abort_handles });
    }

    for handle in join_handles {
        let _ = handle.await;
    }

    update_run(&run_id, |item| {
        item.status = if intermediate {
            "evaluating".to_string()
        } else {
            "completed".to_string()
        };
        item.aggregate_result = Some(summarize_run(item));
        item.recommendation = derive_recommendation(item);
    });
    let _ = run_handles()
        .lock()
        .map(|mut handles| handles.remove(&run_id));

    record = get_run(&run_id).unwrap_or(record);
    persist_run_summary(&record).await;
    record
}

// ---------------------------------------------------------------------------
// Goal loop (trigger + goal + check + stop)
//
// Extends `delegate_task` so the `goal` field stops being decorative text and is
// actually evaluated: after each delegation pass the runner decides whether the
// goal is satisfied (verify-close reuse, else an LLM-as-judge call). If not, and
// pass budget remains, it re-runs the agents with accumulated context; on budget
// exhaustion the run ends in `goal_unmet`, never a false `completed`. This is
// ADDITIVE — it only engages when the caller sets `goal_loop: true` and the goal
// is substantial; every other request keeps the classic single-pass behaviour.
// ---------------------------------------------------------------------------

/// A goal is "substantial" enough to loop on when it reads like a real objective,
/// not a decorative one-liner. Cheap guard so `goal_loop: true` on a throwaway
/// goal ("test", "run agents") still short-circuits to a single pass.
fn goal_is_substantial(goal: &str) -> bool {
    let trimmed = goal.trim();
    trimmed.len() >= 40 && trimmed.split_whitespace().count() >= 6
}

/// Resolve the hard cap on delegation passes: explicit request override →
/// `HERA_GOAL_LOOP_MAX_PASSES` → default 5, clamped to a sane ceiling so a bad
/// env value can never spin the runner forever.
fn goal_loop_max_passes(override_val: Option<u32>) -> u32 {
    override_val
        .or_else(|| {
            std::env::var("HERA_GOAL_LOOP_MAX_PASSES")
                .ok()
                .and_then(|value| value.trim().parse::<u32>().ok())
        })
        .filter(|value| *value >= 1)
        .unwrap_or(5)
        .min(50)
}

fn truncate_for_context(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let head: String = text.chars().take(max_chars).collect();
    format!("{head}… [truncated]")
}

/// Best-effort reuse of the agentic loop's `VERIFY_CLOSE_TOOLS` pattern at the
/// delegation layer: if a delegated agent's output carries an explicit green
/// verification signal (a verify-close tool name AND an unambiguous pass marker)
/// and nothing failed, the goal is considered met without spending a judge call.
/// Deliberately strict — false SATISFIED is the dangerous direction — so it stays
/// inert unless the signal is unmistakable (today's delegate agents run with no
/// tools, so this is mostly a forward-looking hook for tool-enabled delegation).
fn aggregate_shows_verify_close(aggregate: &str, any_failed: bool) -> bool {
    if any_failed {
        return false;
    }
    let lower = aggregate.to_ascii_lowercase();
    let mentions_verify_tool = VERIFY_CLOSE_TOOLS
        .iter()
        .any(|tool| lower.contains(&tool.to_ascii_lowercase()));
    if !mentions_verify_tool {
        return false;
    }
    const GREEN_MARKERS: &[&str] = &[
        "test result: ok",
        "tests passed",
        "all tests pass",
        "0 failed",
        "passed; 0 failed",
    ];
    GREEN_MARKERS.iter().any(|marker| lower.contains(marker))
}

/// Parse an LLM judge reply robustly (Hera's local JSON generation is unreliable,
/// so the judge answers in plain text, not JSON). Returns `(satisfied, reason)`.
/// Conservative by construction: ambiguity or an unparseable reply → NOT
/// satisfied, so the loop never marks a run `completed` on a fuzzy verdict.
fn parse_judge_verdict(text: &str) -> (bool, String) {
    let upper = text.to_ascii_uppercase();
    let reason: String = text.trim().chars().take(300).collect();
    let has_sat = upper.contains("SATISFIED")
        && !upper.contains("NOT SATISFIED")
        && !upper.contains("UNSATISFIED")
        && !upper.contains("NOT YET SATISFIED");
    let has_cont = upper.contains("CONTINUE");
    match (has_sat, has_cont) {
        (true, false) => (true, reason),
        (true, true) => {
            // Both tokens present — trust whichever the model stated first.
            match (upper.find("SATISFIED"), upper.find("CONTINUE")) {
                (Some(sat), Some(cont)) if sat < cont => (true, reason),
                _ => (false, reason),
            }
        }
        _ => (false, reason),
    }
}

/// LLM-as-judge: ask the local model whether `aggregate` satisfies `goal`.
async fn judge_goal(goal: &str, aggregate: &str) -> (bool, String) {
    let persona = "You are a strict goal-completion judge. You are given a GOAL and \
the RESULT that autonomous agents produced. Decide whether the RESULT actually \
satisfies the GOAL. Reply with a SINGLE line that starts with exactly SATISFIED \
or CONTINUE, followed by a one-sentence reason. Be conservative: if the result is \
incomplete, vague, contradictory, or merely promises to do the work, answer \
CONTINUE."
        .to_string();
    let prompt = format!(
        "GOAL:\n{goal}\n\nRESULT:\n{}\n\nIs the goal satisfied? Answer SATISFIED or CONTINUE with a brief reason.",
        truncate_for_context(aggregate, 4000)
    );
    match run_agent_via_hera_ipc(persona, prompt).await {
        Ok(text) => parse_judge_verdict(&text),
        Err(error) => (
            false,
            format!("judge call failed, treating goal as not satisfied: {error}"),
        ),
    }
}

/// Decide whether the goal is met after a pass: verify-close short-circuit first
/// (cheap, no model call), otherwise the LLM judge.
async fn evaluate_goal(goal: &str, aggregate: &str, any_failed: bool) -> (bool, String) {
    if aggregate_shows_verify_close(aggregate, any_failed) {
        return (
            true,
            "verify-close: a verification tool reported green in the delegated output".to_string(),
        );
    }
    judge_goal(goal, aggregate).await
}

/// Build the delegation request for pass N. Pass 1 is the caller's request as-is;
/// later passes append the accumulated context (what was tried, why it fell
/// short) to each agent's prompt so the next attempt addresses the gap.
fn build_pass_request(
    base: &DelegateTaskRequest,
    goal: &str,
    accumulated_context: &str,
    pass: u32,
) -> DelegateTaskRequest {
    if pass <= 1 || accumulated_context.trim().is_empty() {
        return base.clone();
    }
    let agents = base
        .agents
        .iter()
        .map(|spec| {
            let original = spec.prompt.clone().unwrap_or_else(|| goal.to_string());
            DelegationAgentSpec {
                agent: spec.agent.clone(),
                prompt: Some(format!(
                    "{original}\n\n---\nCONTEXT FROM PRIOR ATTEMPTS (the goal is NOT yet satisfied — address the gaps, do not repeat what already failed):\n{accumulated_context}"
                )),
            }
        })
        .collect();
    DelegateTaskRequest {
        agents,
        wait_for_completion: Some(true),
        ..base.clone()
    }
}

/// Run the goal loop: repeat delegation passes, evaluating the goal after each,
/// until satisfied or the pass budget is exhausted. Every pass is persisted to
/// Memento as an `open_loop` row so progress survives a `hera-core` restart. The
/// run's registry record is updated in place (reusing `run_id`), ending in
/// `completed` (goal met) or `goal_unmet` (budget exhausted) — never a false
/// `completed`.
async fn run_goal_loop(request: DelegateTaskRequest, run_id: String) -> AgentRunRecord {
    let max_passes = goal_loop_max_passes(request.max_passes);
    let goal = request.goal.clone();
    let app_id = request.app.clone().unwrap_or_else(|| "hera".to_string());
    let session_id = request
        .session_id
        .clone()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| run_id.clone());

    let mut accumulated_context = String::new();
    let mut last_reason = String::new();

    for pass in 1..=max_passes {
        let pass_request = build_pass_request(&request, &goal, &accumulated_context, pass);
        let record = spawn_delegate_run(pass_request, Some(run_id.clone()), true).await;        let aggregate = record.aggregate_result.clone().unwrap_or_default();
        let any_failed = record
            .agents
            .iter()
            .any(|item| item.status != "completed");

        let (satisfied, reason) = evaluate_goal(&goal, &aggregate, any_failed).await;
        last_reason = reason.clone();

        // Durable per-pass state (survives a hera-core restart; registry is RAM-only).
        save_open_loop_memory(
            app_id.clone(),
            session_id.clone(),
            run_id.clone(),
            goal.clone(),
            pass,
            max_passes,
            satisfied,
            reason.clone(),
            truncate_for_context(&aggregate, 1500),
        )
        .await;

        if satisfied {
            tracing::info!(
                "🎯 [Hera goal-loop] run {} satisfied on pass {}/{}",
                run_id,
                pass,
                max_passes
            );
            update_run(&run_id, |item| {
                item.status = "completed".to_string();
                item.goal_passes = Some(pass);
                item.goal_judge_reason = Some(reason.clone());
            });
            if let Some(final_record) = get_run(&run_id) {
                persist_run_summary(&final_record).await;
            }
            break;
        }

        accumulated_context = format!(
            "Pass {pass}/{max_passes} result summary:\n{}\n\nWhy the goal was NOT satisfied: {reason}",
            truncate_for_context(&aggregate, 2000)
        );

        if pass >= max_passes {
            tracing::warn!(
                "🎯 [Hera goal-loop] run {} exhausted {} passes without satisfying the goal",
                run_id,
                max_passes
            );
            update_run(&run_id, |item| {
                item.status = "goal_unmet".to_string();
                item.goal_passes = Some(pass);
                item.goal_judge_reason = Some(reason.clone());
                item.recommendation = Some(format!(
                    "Goal not satisfied after {max_passes} passes: {reason}"
                ));
            });
            if let Some(final_record) = get_run(&run_id) {
                persist_run_summary(&final_record).await;
            }
        }
    }

    get_run(&run_id).unwrap_or_else(|| AgentRunRecord {
        run_id: run_id.clone(),
        app: app_id,
        trace_id: String::new(),
        session_id,
        chat_id: String::new(),
        goal,
        strategy: "goal_loop".to_string(),
        status: "goal_unmet".to_string(),
        created_at_ms: now_ms(),
        updated_at_ms: now_ms(),
        route_profile: "goal_loop".to_string(),
        agent_specs: Vec::new(),
        agents: Vec::new(),
        aggregate_result: None,
        recommendation: None,
        goal_passes: Some(max_passes),
        goal_judge_reason: Some(last_reason),
    })
}

/// Build the initial `queued` record for a background run (goal-loop or classic),
/// so `get`/`await` find it immediately after the IPC response returns.
fn build_kickoff_record(run_id: &str, payload: &DelegateTaskRequest) -> AgentRunRecord {
    let app = payload.app.clone().unwrap_or_else(|| "unknown".to_string());
    AgentRunRecord {
        run_id: run_id.to_string(),
        app: app.clone(),
        trace_id: payload.trace_id.clone().unwrap_or_default(),
        session_id: payload.session_id.clone().unwrap_or_default(),
        chat_id: payload.chat_id.clone().unwrap_or_default(),
        goal: payload.goal.clone(),
        strategy: payload
            .strategy
            .clone()
            .unwrap_or_else(|| "parallel".to_string()),
        status: "queued".to_string(),
        created_at_ms: now_ms(),
        updated_at_ms: now_ms(),
        route_profile: format!("{}_delegation", app),
        agent_specs: payload.agents.clone(),
        agents: payload
            .agents
            .iter()
            .map(|spec| AgentRunItem {
                agent: spec.agent.clone(),
                status: "queued".to_string(),
                output: None,
                error: None,
            })
            .collect(),
        aggregate_result: None,
        recommendation: None,
        goal_passes: None,
        goal_judge_reason: None,
    }
}

pub async fn handle_delegate_task(
    request: &IpcPayload,
    _state: &IpcState,
    stream: &mut tokio::net::UnixStream,
) -> HandlerOutcome {
    let Ok(payload) = serde_json::from_value::<DelegateTaskRequest>(request.payload.clone()) else {
        send_ipc_response(
            stream,
            &IpcResponse {
                status: "error".to_string(),
                data: serde_json::json!({"error": "invalid delegate_task payload"}),
            },
        )
        .await;
        return HandlerOutcome::DirectResponse;
    };

    if payload.goal.trim().is_empty() || payload.agents.is_empty() {
        send_ipc_response(
            stream,
            &IpcResponse {
                status: "error".to_string(),
                data: serde_json::json!({"error": "delegate_task requires goal and at least one agent"}),
            },
        )
        .await;
        return HandlerOutcome::DirectResponse;
    }

    // Engage the evaluated goal loop only when explicitly opted in AND the goal
    // reads like a real objective. Every other request falls through to the
    // classic single-pass delegation below, byte-for-byte unchanged.
    let use_goal_loop =
        payload.goal_loop.unwrap_or(false) && goal_is_substantial(&payload.goal);
    let wait = payload.wait_for_completion.unwrap_or(true);

    if use_goal_loop {
        if wait {
            let record = run_goal_loop(payload, new_run_id()).await;
            send_ipc_response(
                stream,
                &IpcResponse {
                    status: "success".to_string(),
                    data: serde_json::to_value(record).unwrap_or_else(|_| serde_json::json!({})),
                },
            )
            .await;
            return HandlerOutcome::DirectResponse;
        }

        let run_id = new_run_id();
        let kickoff = build_kickoff_record(&run_id, &payload);
        if let Ok(mut registry) = run_registry().lock() {
            registry.insert(run_id.clone(), kickoff.clone());
        }
        let (response_run_id, goal, agent_count) =
            (run_id.clone(), kickoff.goal.clone(), kickoff.agents.len());
        let background_run_id = run_id.clone();
        tokio::spawn(async move {
            let _ = run_goal_loop(payload, background_run_id).await;
        });
        send_ipc_response(
            stream,
            &IpcResponse {
                status: "success".to_string(),
                data: serde_json::json!({
                    "run_id": response_run_id,
                    "status": "queued",
                    "goal": goal,
                    "agent_count": agent_count,
                    "goal_loop": true,
                }),
            },
        )
        .await;
        return HandlerOutcome::DirectResponse;
    }

    if wait {
        let record = spawn_delegate_run(payload, None, false).await;
        send_ipc_response(
            stream,
            &IpcResponse {
                status: "success".to_string(),
                data: serde_json::to_value(record).unwrap_or_else(|_| serde_json::json!({})),
            },
        )
        .await;
        return HandlerOutcome::DirectResponse;
    }

    let run_id = new_run_id();
    let kickoff = build_kickoff_record(&run_id, &payload);
    if let Ok(mut registry) = run_registry().lock() {
        registry.insert(run_id.clone(), kickoff.clone());
    }
    let (response_run_id, goal, agent_count) =
        (run_id.clone(), kickoff.goal.clone(), kickoff.agents.len());
    let mut background_payload = payload;
    background_payload.wait_for_completion = Some(true);
    let background_run_id = run_id.clone();
    tokio::spawn(async move {
        let _ = spawn_delegate_run(background_payload, Some(background_run_id), false).await;    });

    send_ipc_response(
        stream,
        &IpcResponse {
            status: "success".to_string(),
            data: serde_json::json!({
                "run_id": response_run_id,
                "status": "queued",
                "goal": goal,
                "agent_count": agent_count,
            }),
        },
    )
    .await;
    HandlerOutcome::DirectResponse
}

pub async fn handle_await_agent_run(
    request: &IpcPayload,
    _state: &IpcState,
    stream: &mut tokio::net::UnixStream,
) -> HandlerOutcome {
    let run_id = request
        .payload
        .get("run_id")
        .and_then(|value| value.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if run_id.is_empty() {
        send_ipc_response(
            stream,
            &IpcResponse {
                status: "error".to_string(),
                data: serde_json::json!({"error": "run_id is required"}),
            },
        )
        .await;
        return HandlerOutcome::DirectResponse;
    }

    let timeout_ms = request
        .payload
        .get("timeout_ms")
        .and_then(|value| value.as_u64())
        .unwrap_or(30_000)
        .clamp(100, 600_000);
    let poll_ms = request
        .payload
        .get("poll_ms")
        .and_then(|value| value.as_u64())
        .unwrap_or(100)
        .clamp(25, 2_000);
    let started = now_ms();

    loop {
        if let Some(record) = get_run(&run_id) {
            if is_terminal_status(&record.status) {
                send_ipc_response(
                    stream,
                    &IpcResponse {
                        status: "success".to_string(),
                        data: serde_json::to_value(record)
                            .unwrap_or_else(|_| serde_json::json!({})),
                    },
                )
                .await;
                return HandlerOutcome::DirectResponse;
            }
        } else {
            send_ipc_response(
                stream,
                &IpcResponse {
                    status: "error".to_string(),
                    data: serde_json::json!({"error": "agent run not found"}),
                },
            )
            .await;
            return HandlerOutcome::DirectResponse;
        }

        if now_ms().saturating_sub(started) >= timeout_ms {
            send_ipc_response(
                stream,
                &IpcResponse {
                    status: "success".to_string(),
                    data: serde_json::json!({
                        "run_id": run_id,
                        "status": "timeout",
                        "timed_out": true
                    }),
                },
            )
            .await;
            return HandlerOutcome::DirectResponse;
        }

        sleep(Duration::from_millis(poll_ms)).await;
    }
}

pub async fn handle_list_agent_runs(
    _request: &IpcPayload,
    _state: &IpcState,
    stream: &mut tokio::net::UnixStream,
) -> HandlerOutcome {
    let runs = run_registry()
        .lock()
        .map(|registry| registry.values().cloned().collect::<Vec<_>>())
        .unwrap_or_default();
    send_ipc_response(
        stream,
        &IpcResponse {
            status: "success".to_string(),
            data: serde_json::json!({ "runs": runs }),
        },
    )
    .await;
    HandlerOutcome::DirectResponse
}

pub async fn handle_get_agent_run(
    request: &IpcPayload,
    _state: &IpcState,
    stream: &mut tokio::net::UnixStream,
) -> HandlerOutcome {
    let run_id = request
        .payload
        .get("run_id")
        .and_then(|value| value.as_str())
        .unwrap_or("")
        .trim();
    if run_id.is_empty() {
        send_ipc_response(
            stream,
            &IpcResponse {
                status: "error".to_string(),
                data: serde_json::json!({"error": "run_id is required"}),
            },
        )
        .await;
        return HandlerOutcome::DirectResponse;
    }
    let run = get_run(run_id);
    match run {
        Some(record) => {
            send_ipc_response(
                stream,
                &IpcResponse {
                    status: "success".to_string(),
                    data: serde_json::to_value(record).unwrap_or_else(|_| serde_json::json!({})),
                },
            )
            .await;
        }
        None => {
            send_ipc_response(
                stream,
                &IpcResponse {
                    status: "error".to_string(),
                    data: serde_json::json!({"error": "agent run not found"}),
                },
            )
            .await;
        }
    }
    HandlerOutcome::DirectResponse
}

pub async fn handle_cancel_agent_run(
    request: &IpcPayload,
    _state: &IpcState,
    stream: &mut tokio::net::UnixStream,
) -> HandlerOutcome {
    let run_id = request
        .payload
        .get("run_id")
        .and_then(|value| value.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if run_id.is_empty() {
        send_ipc_response(
            stream,
            &IpcResponse {
                status: "error".to_string(),
                data: serde_json::json!({"error": "run_id is required"}),
            },
        )
        .await;
        return HandlerOutcome::DirectResponse;
    }

    let mut cancelled = false;
    if let Ok(mut handles) = run_handles().lock()
        && let Some(handle) = handles.remove(&run_id)
    {
        for abort in handle.abort_handles {
            abort.abort();
        }
        cancelled = true;
    }
    if cancelled {
        update_run(&run_id, |record| {
            record.status = "cancelled".to_string();
            for agent in &mut record.agents {
                if agent.status == "queued" || agent.status == "running" {
                    agent.status = "cancelled".to_string();
                    agent.error = Some("Cancelled by caller".to_string());
                }
            }
        });
        if let Some(record) = get_run(&run_id) {
            persist_run_summary(&record).await;
        }
    }

    send_ipc_response(
        stream,
        &IpcResponse {
            status: if cancelled { "success" } else { "error" }.to_string(),
            data: serde_json::json!({
                "run_id": run_id,
                "cancelled": cancelled,
                "error": if cancelled { serde_json::Value::Null } else { serde_json::json!("agent run not found or already finished") }
            }),
        },
    )
    .await;
    HandlerOutcome::DirectResponse
}

pub async fn handle_resume_agent_run(
    request: &IpcPayload,
    _state: &IpcState,
    stream: &mut tokio::net::UnixStream,
) -> HandlerOutcome {
    let run_id = request
        .payload
        .get("run_id")
        .and_then(|value| value.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if run_id.is_empty() {
        send_ipc_response(
            stream,
            &IpcResponse {
                status: "error".to_string(),
                data: serde_json::json!({"error": "run_id is required"}),
            },
        )
        .await;
        return HandlerOutcome::DirectResponse;
    }

    let Some(record) = get_run(&run_id) else {
        send_ipc_response(
            stream,
            &IpcResponse {
                status: "error".to_string(),
                data: serde_json::json!({"error": "agent run not found"}),
            },
        )
        .await;
        return HandlerOutcome::DirectResponse;
    };

    if record.status != "cancelled" && record.status != "failed" {
        send_ipc_response(
            stream,
            &IpcResponse {
                status: "error".to_string(),
                data: serde_json::json!({"error": "only cancelled or failed runs can be resumed"}),
            },
        )
        .await;
        return HandlerOutcome::DirectResponse;
    }

    let payload = DelegateTaskRequest {
        app: Some(record.app.clone()),
        trace_id: Some(record.trace_id.clone()),
        session_id: Some(record.session_id.clone()),
        chat_id: Some(record.chat_id.clone()),
        goal: record.goal.clone(),
        strategy: Some(record.strategy.clone()),
        wait_for_completion: Some(
            request
                .payload
                .get("wait_for_completion")
                .and_then(|value| value.as_bool())
                .unwrap_or(false),
        ),
        agents: record.agent_specs.clone(),
        // Resume replays the classic single-pass delegation; goal-loop re-runs are
        // driven by a fresh delegate_task, not by resume.
        goal_loop: None,
        max_passes: None,
    };

    if payload.wait_for_completion == Some(true) {
        let resumed = spawn_delegate_run(payload, Some(run_id), false).await;
        send_ipc_response(
            stream,
            &IpcResponse {
                status: "success".to_string(),
                data: serde_json::to_value(resumed).unwrap_or_else(|_| serde_json::json!({})),
            },
        )
        .await;
        return HandlerOutcome::DirectResponse;
    }

    update_run(&run_id, |item| {
        item.status = "queued".to_string();
        item.aggregate_result = None;
        item.recommendation = None;
        for agent in &mut item.agents {
            agent.status = "queued".to_string();
            agent.output = None;
            agent.error = None;
        }
    });
    let background_run_id = run_id.clone();
    tokio::spawn(async move {
        let _ = spawn_delegate_run(payload, Some(background_run_id), false).await;
    });
    let resumed = get_run(&run_id).unwrap_or(record);
    send_ipc_response(
        stream,
        &IpcResponse {
            status: "success".to_string(),
            data: serde_json::to_value(resumed).unwrap_or_else(|_| serde_json::json!({})),
        },
    )
    .await;
    HandlerOutcome::DirectResponse
}

pub async fn handle_summarize_agent_run(
    request: &IpcPayload,
    _state: &IpcState,
    stream: &mut tokio::net::UnixStream,
) -> HandlerOutcome {
    let run_id = request
        .payload
        .get("run_id")
        .and_then(|value| value.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if run_id.is_empty() {
        send_ipc_response(
            stream,
            &IpcResponse {
                status: "error".to_string(),
                data: serde_json::json!({"error": "run_id is required"}),
            },
        )
        .await;
        return HandlerOutcome::DirectResponse;
    }

    let Some(record) = get_run(&run_id) else {
        send_ipc_response(
            stream,
            &IpcResponse {
                status: "error".to_string(),
                data: serde_json::json!({"error": "agent run not found"}),
            },
        )
        .await;
        return HandlerOutcome::DirectResponse;
    };

    let summary = summarize_run(&record);
    if request
        .payload
        .get("persist")
        .and_then(|value| value.as_bool())
        .unwrap_or(true)
    {
        persist_run_summary(&record).await;
    }

    send_ipc_response(
        stream,
        &IpcResponse {
            status: "success".to_string(),
            data: serde_json::json!({
                "run_id": run_id,
                "status": record.status,
                "summary": summary,
                "recommendation": record.recommendation,
                "aggregate_result": record.aggregate_result,
            }),
        },
    )
    .await;
    HandlerOutcome::DirectResponse
}

#[cfg(test)]
mod goal_loop_tests {
    use super::*;

    #[test]
    fn substantial_goal_gates_decorative_one_liners() {
        assert!(!goal_is_substantial("test"));
        assert!(!goal_is_substantial("run the agents"));
        // Long enough in chars but too few words is still rejected.
        assert!(!goal_is_substantial(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        ));
        assert!(goal_is_substantial(
            "Refactor the auth module so every handler uses the shared session guard and it compiles"
        ));
    }

    #[test]
    fn max_passes_resolution_and_clamp() {
        // Explicit override wins and is honoured.
        assert_eq!(goal_loop_max_passes(Some(3)), 3);
        // 0 is invalid → falls through to default.
        assert_eq!(goal_loop_max_passes(Some(0)), 5);
        // No override, no env → default 5.
        assert_eq!(goal_loop_max_passes(None), 5);
        // Absurd override is clamped to the ceiling.
        assert_eq!(goal_loop_max_passes(Some(9999)), 50);
    }

    #[test]
    fn judge_verdict_parses_conservatively() {
        // Clear SATISFIED.
        assert!(parse_judge_verdict("SATISFIED the goal is fully met.").0);
        // Clear CONTINUE.
        assert!(!parse_judge_verdict("CONTINUE the result is incomplete.").0);
        // Negations never count as satisfied.
        assert!(!parse_judge_verdict("NOT SATISFIED, keep going.").0);
        assert!(!parse_judge_verdict("The goal is UNSATISFIED so far.").0);
        assert!(!parse_judge_verdict("NOT YET SATISFIED.").0);
        // Both tokens: the one stated FIRST wins.
        assert!(parse_judge_verdict("SATISFIED. No need to CONTINUE.").0);
        assert!(!parse_judge_verdict("CONTINUE — do not mark it SATISFIED yet.").0);
        // Garbage / empty → not satisfied (never a false completion).
        assert!(!parse_judge_verdict("").0);
        assert!(!parse_judge_verdict("hmm, maybe?").0);
    }

    #[test]
    fn verify_close_requires_tool_name_and_green_marker_and_no_failure() {
        // Tool name + green marker + nothing failed → close.
        assert!(aggregate_shows_verify_close(
            "ran cargo_test: test result: ok. 12 passed; 0 failed",
            false
        ));
        // Same signal but a delegated agent failed → do NOT close.
        assert!(!aggregate_shows_verify_close(
            "ran cargo_test: test result: ok. 12 passed; 0 failed",
            true
        ));
        // Merely suggesting to run tests (no green marker) → do NOT close.
        assert!(!aggregate_shows_verify_close(
            "you should run cargo_test to be sure",
            false
        ));
        // Green words but no verify tool named → do NOT close.
        assert!(!aggregate_shows_verify_close("all tests pass, trust me", false));
    }

    #[test]
    fn later_pass_request_appends_context_first_pass_untouched() {
        let base = DelegateTaskRequest {
            app: Some("hera".to_string()),
            trace_id: None,
            session_id: None,
            chat_id: None,
            goal: "do the thing well".to_string(),
            strategy: None,
            wait_for_completion: None,
            agents: vec![DelegationAgentSpec {
                agent: "worker".to_string(),
                prompt: Some("original prompt".to_string()),
            }],
            goal_loop: Some(true),
            max_passes: None,
        };
        // Pass 1 is the request as-is.
        let p1 = build_pass_request(&base, &base.goal, "some ctx", 1);
        assert_eq!(p1.agents[0].prompt.as_deref(), Some("original prompt"));
        // Pass 2 appends the accumulated context to the agent prompt.
        let p2 = build_pass_request(&base, &base.goal, "why it failed", 2);
        let prompt = p2.agents[0].prompt.clone().unwrap();
        assert!(prompt.starts_with("original prompt"));
        assert!(prompt.contains("why it failed"));
        assert_eq!(p2.wait_for_completion, Some(true));
    }
}
