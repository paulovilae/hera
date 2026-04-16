use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
use tokio::time::{Duration, sleep};

use super::helpers::save_agent_run_summary;
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
    matches!(status, "completed" | "failed" | "cancelled")
}

fn get_run(run_id: &str) -> Option<AgentRunRecord> {
    run_registry()
        .lock()
        .ok()
        .and_then(|registry| registry.get(run_id).cloned())
}

async fn spawn_delegate_run(
    request: DelegateTaskRequest,
    existing_run_id: Option<String>,
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
        item.status = "completed".to_string();
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

    let wait = payload.wait_for_completion.unwrap_or(true);
    if wait {
        let record = spawn_delegate_run(payload, None).await;
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
    let kickoff = AgentRunRecord {
        run_id: run_id.clone(),
        app: payload.app.clone().unwrap_or_else(|| "unknown".to_string()),
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
        route_profile: format!(
            "{}_delegation",
            payload.app.clone().unwrap_or_else(|| "unknown".to_string())
        ),
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
    };
    if let Ok(mut registry) = run_registry().lock() {
        registry.insert(run_id.clone(), kickoff.clone());
    }
    let payload_clone = DelegateTaskRequest {
        app: Some(kickoff.app.clone()),
        trace_id: Some(kickoff.trace_id.clone()),
        session_id: Some(kickoff.session_id.clone()),
        chat_id: Some(kickoff.chat_id.clone()),
        goal: kickoff.goal.clone(),
        strategy: Some(kickoff.strategy.clone()),
        wait_for_completion: Some(true),
        agents: payload.agents.clone(),
    };
    let background_run_id = run_id.clone();
    tokio::spawn(async move {
        let _ = spawn_delegate_run(payload_clone, Some(background_run_id)).await;
    });

    send_ipc_response(
        stream,
        &IpcResponse {
            status: "success".to_string(),
            data: serde_json::json!({
                "run_id": run_id,
                "status": "queued",
                "goal": kickoff.goal,
                "agent_count": kickoff.agents.len(),
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
    };

    if payload.wait_for_completion == Some(true) {
        let resumed = spawn_delegate_run(payload, Some(run_id)).await;
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
        let _ = spawn_delegate_run(payload, Some(background_run_id)).await;
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
