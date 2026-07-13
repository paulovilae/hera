//! Handler: generate (non-streaming LLM inference with tool execution).

use super::context::{
    build_runtime_outcome_artifacts, latest_user_message_text, parse_payload, prepare_chat_request,
    prepare_runtime_execution_context, prepare_tool_result_followup_request,
};
use super::helpers::{
    RuntimePromotionContext, canonicalize_user_id, hera_node, infer_origin_from_model,
    record_observation_and_promote_runtime_hint, record_runtime_observation, report_recall_feedback,
    save_chat_turn_event, spawn_log_usage,
};
use super::inflight;
use super::llm_audit::append_llm_audit_event;
use super::runtime_tools::{
    FollowupStrategy, contextualize_tool_call, execute_parsed_tool_calls, execute_tool_followup,
    summarize_tool_output_for_user, try_plan_schema_query,
};
use super::types::{HandlerOutcome, IpcPayload, IpcResponse, IpcState};
use std::time::Instant;
use tokio::io::AsyncWriteExt;
use tokio::net::UnixStream;

/// Handle the "generate" IPC action — non-streaming LLM chat with tool execution.
pub async fn handle_generate(
    request: &IpcPayload,
    state: &IpcState,
    stream: &mut UnixStream,
) -> HandlerOutcome {
    let started_at = Instant::now();
    let mut payload_clone = request.payload.clone();
    let mut parsed = parse_payload(&payload_clone);
    let provider_requested = payload_clone
        .get("provider")
        .and_then(|value| value.as_str())
        .unwrap_or("auto")
        .to_string();
    let caller_overrode_budget = payload_clone.get("context_budget_mode").is_some()
        || payload_clone.get("context_budget").is_some();
    let explicit_user_command = latest_user_message_text(&payload_clone)
        .filter(|message| message.trim_start().starts_with('/'));

    // 1. Fast-path intent detection
    let fast_path_prompt = explicit_user_command
        .as_deref()
        .unwrap_or(parsed.prompt.as_str());
    tracing::info!(
        "🧪 [Hera IPC] Fast-path probe explicit_user_command={:?} parsed_prompt={:?} fast_path_prompt={:?}",
        explicit_user_command,
        parsed.prompt,
        fast_path_prompt
    );
    // Tool paths are gated on the budget's allow_tools. A caller that asks for
    // pure generation (allow_tools=false, e.g. dossier section/metadata/chart
    // synthesis) must NOT have its request hijacked into a tool call — the
    // fast-path intent detector and the schema-query planner below both fire
    // proactively off the prompt content and would otherwise turn a "write me
    // charts JSON" call into a query_memory/memento_vector_search execution.
    if parsed.context_budget.allow_tools && !fast_path_prompt.is_empty() {
        if fast_path_prompt.trim_start().starts_with('/') {
            tracing::info!(
                "🧭 [Hera IPC] Explicit command candidate received: {}",
                fast_path_prompt.trim()
            );
        }
        if let Some(tool_call) = crate::ai::tool_executor::detect_intent_from_user_message(
            fast_path_prompt,
            parsed.assistant_last.as_deref(),
        ) {
            let contextual_tool_call = contextualize_tool_call(&tool_call, &parsed);
            if crate::ai::tool_executor::permissions_allow_tool(
                &parsed.permissions,
                &contextual_tool_call.name,
            ) {
                tracing::info!(
                    "🚀 [Hera IPC] Fast-path tool intent detected: {}",
                    contextual_tool_call.name
                );
                let fast_path_result = crate::ai::tool_executor::execute_tool(&contextual_tool_call)
                    .await
                    .output;

                let res = IpcResponse {
                    status: "success".to_string(),
                    data: serde_json::json!({
                        "result": fast_path_result,
                        "origin": "tool",
                        "model": contextual_tool_call.name,
                        "tool_calls": [contextual_tool_call]
                    }),
                };
                let mut res_str = serde_json::to_string(&res).unwrap();
                res_str.push('\n');
                if let Err(e) = stream.write_all(res_str.as_bytes()).await {
                    tracing::error!("❌ Failed to write IPC response for fast-path tool: {}", e);
                }
                return HandlerOutcome::DirectResponse;
            } else {
                tracing::info!(
                    "⚠️ [Hera IPC] Fast-path tool intent {} denied by permissions",
                    contextual_tool_call.name
                );
            }
        } else if fast_path_prompt.trim_start().starts_with('/') {
            tracing::warn!(
                "⚠️ [Hera IPC] Explicit command was not recognized by fast-path: {}",
                fast_path_prompt.trim()
            );
        }
    }

    let schema_plan_started_at = Instant::now();
    if parsed.context_budget.allow_tools
        && let Some(planned_call) = try_plan_schema_query(&state.engine, &parsed).await
    {
        let planner_latency_ms = schema_plan_started_at.elapsed().as_millis();
        let contextual_tool_call = contextualize_tool_call(&planned_call, &parsed);
        if crate::ai::tool_executor::permissions_allow_tool(
            &parsed.permissions,
            &contextual_tool_call.name,
        ) {
            let initial_query = contextual_tool_call
                .arguments
                .get("query")
                .and_then(|value| value.as_str())
                .unwrap_or("")
                .to_string();
            tracing::info!(
                "🧠 [Hera IPC] Generic schema query plan generated for app '{}' in {}ms",
                parsed.app_name,
                planner_latency_ms
            );
            let tool_started_at = Instant::now();
            let mut tool_result = crate::ai::tool_executor::execute_tool(&contextual_tool_call)
                .await
                .output;
            let mut replanned = false;
            if super::runtime_tools::should_retry_schema_query(&tool_result) {
                let retry_started_at = Instant::now();
                if let Some(replanned_call) = super::runtime_tools::retry_plan_schema_query(
                    &state.engine,
                    &parsed,
                    &initial_query,
                    &tool_result,
                )
                .await
                {
                    let contextual_retry_call = contextualize_tool_call(&replanned_call, &parsed);
                    tracing::info!(
                        "🧠 [Hera IPC] Retrying schema query for app '{}' after {}ms planner retry",
                        parsed.app_name,
                        retry_started_at.elapsed().as_millis()
                    );
                    tool_result = crate::ai::tool_executor::execute_tool(&contextual_retry_call)
                        .await
                        .output;
                    replanned = true;
                }
            }
            let tool_latency_ms = tool_started_at.elapsed().as_millis();
            let summarize_started_at = Instant::now();
            let result_text = summarize_tool_output_for_user(&state.engine, &parsed, &tool_result)
                .await
                .unwrap_or(tool_result);
            let summarize_latency_ms = summarize_started_at.elapsed().as_millis();
            tracing::info!(
                "⏱️ [Hera IPC] Schema planner path app='{}' planner_ms={} tool_ms={} summarize_ms={} retried={}",
                parsed.app_name,
                planner_latency_ms,
                tool_latency_ms,
                summarize_latency_ms,
                replanned
            );

            let res = IpcResponse {
                status: "success".to_string(),
                data: serde_json::json!({
                    "result": result_text,
                    "origin": "tool",
                    "model": contextual_tool_call.name,
                    "tool_calls": [contextual_tool_call]
                }),
            };
            let mut res_str = serde_json::to_string(&res).unwrap();
            res_str.push('\n');
            if let Err(e) = stream.write_all(res_str.as_bytes()).await {
                tracing::error!("❌ Failed to write IPC response for schema planner tool: {}", e);
            }
            return HandlerOutcome::DirectResponse;
        }
    }

    // 2. Ensure model is set
    if let Some(obj) = payload_clone.as_object_mut() {
        if !obj.contains_key("model") {
            obj.insert("model".to_string(), serde_json::json!("hera-local-model"));
        }
    }

    let prompt = payload_clone
        .get("prompt")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let prepared =
        prepare_runtime_execution_context(&mut parsed, caller_overrode_budget, "generate").await;
    let runtime_preflight = prepared.runtime_preflight;
    let prompt_assembly = prepared.prompt_assembly;
    let lightweight_mode = prepared.lightweight_mode;
    let chat_req = prepare_chat_request(
        &payload_clone,
        &prompt,
        &parsed,
        &prompt_assembly,
        &state.engine,
    )
    .await;

    // 5. Generate
    if let Some(req) = chat_req.clone() {
        // Wave 3 in-flight registry: insert right before the real generation
        // attempt (fast-path/schema-planner short-circuits above never reach
        // here — they're cheap and don't need liveness tracking). Every return
        // from this point on must `inflight::remove` on its way out.
        inflight::insert(
            &parsed.trace_id,
            &parsed.app_name,
            &parsed.route_profile_id,
            &hera_node(),
        );
        let est_tokens = super::helpers::estimate_tokens(&req);
        tracing::info!(
            "📡 [Hera Generate] Starting inference for app='{}' — {} msgs, ~{} tokens (lightweight_mode={})",
            parsed.app_name,
            req.messages.len(),
            est_tokens,
            lightweight_mode
        );

        // Fase 1 (docs/AVA_CODING_AGENT_PLAN.md): real multi-turn agentic loop.
        // Behind HERA_AGENTIC_LOOP, and only for tool-enabled requests. Replaces
        // the single-shot "execute 1 batch → format → done" path below with
        // generate → execute tools → re-feed results → repeat. Reuses the exact
        // same tool executors; the bots' current behaviour is unchanged when the
        // flag is off. Fast-path + schema-planner above still short-circuit first.
        if parsed.context_budget.allow_tools && super::agentic_loop::agentic_loop_enabled() {
            let loop_outcome =
                super::agentic_loop::run_agentic_loop(&state.engine, req, &parsed, None).await;
            tracing::info!(
                "🔁 [Hera IPC] Agentic loop finished: iterations={} stop_reason={} tools={}",
                loop_outcome.iterations,
                loop_outcome.stop_reason,
                loop_outcome.executed_calls_json.len()
            );
            let loop_usage = loop_outcome.usage;
            let result_text = loop_outcome.result_text;
            let response_origin = loop_outcome.origin;
            let response_model = loop_outcome.model;
            let tool_calls = if loop_outcome.executed_calls_json.is_empty() {
                None
            } else {
                Some(serde_json::Value::Array(loop_outcome.executed_calls_json))
            };

            let duration_ms = started_at.elapsed().as_millis() as u64;
            let outcome = build_runtime_outcome_artifacts(
                "generate",
                &parsed,
                &prompt_assembly,
                duration_ms,
                None,
                lightweight_mode,
                &provider_requested,
                &response_origin,
                &response_model,
                true,
                tool_calls
                    .as_ref()
                    .and_then(|value| value.as_array().map(|items| items.len()))
                    .unwrap_or(0),
                result_text.len(),
                est_tokens,
                chat_req
                    .as_ref()
                    .map(|req| req.messages.len())
                    .unwrap_or_default(),
                None,
            );
            append_llm_audit_event(&outcome.audit_event);
            record_observation_and_promote_runtime_hint(
                outcome.observation_payload,
                RuntimePromotionContext {
                    preflight: runtime_preflight.clone(),
                    mode: "generate",
                    app_id: &parsed.app_name,
                    route_profile: &parsed.route_profile_id,
                    persona_path: &parsed.persona_path,
                    session_id: &parsed.session_id,
                    trace_id: &parsed.trace_id,
                    chat_id: &parsed.chat_id,
                    current_budget_mode: &parsed.context_budget.mode,
                    persona_drift: parsed.persona_drift,
                    success: true,
                },
            )
            .await;

            if !lightweight_mode && parsed.context_budget.include_memory {
                let user_id = canonicalize_user_id(
                    &parsed.sender_name,
                    &parsed.chat_id,
                    &parsed.session_id,
                );
                let app_id = parsed.app_name.clone();
                let session_id = parsed.session_id.clone();
                let user_content = parsed.prompt.clone();
                let assistant_content = result_text.clone();
                let attribution = prompt_assembly.recall_attribution.clone();
                tokio::spawn(async move {
                    save_chat_turn_event(
                        user_id.clone(),
                        app_id.clone(),
                        session_id.clone(),
                        "user".to_string(),
                        user_content,
                    )
                    .await;
                    save_chat_turn_event(
                        user_id,
                        app_id,
                        session_id,
                        "assistant".to_string(),
                        assistant_content.clone(),
                    )
                    .await;
                    report_recall_feedback(attribution.as_ref(), &assistant_content).await;
                });
            }

            // Path A — usage logging (best-effort, fire-and-forget). Summed
            // across every turn of the agentic loop (see LoopUsage in
            // agentic_loop.rs) — previously hardcoded to 0,0,0 because the
            // loop discarded per-turn ChatResponse.usage entirely.
            spawn_log_usage(
                parsed.app_name.clone(),
                canonicalize_user_id(&parsed.sender_name, &parsed.chat_id, &parsed.session_id),
                parsed.session_id.clone(),
                parsed.route_profile_id.clone(),
                response_model.clone(),
                loop_usage.prompt_tokens,
                loop_usage.completion_tokens,
                loop_usage.total_tokens,
                response_origin.contains("cloud"),
                duration_ms,
                parsed.trace_id.clone(),
            );

            inflight::remove(&parsed.trace_id);
            return HandlerOutcome::Result {
                result_text,
                origin: response_origin,
                model: response_model,
                tool_calls,
            };
        }

        match state.engine.generate_content(req).await {
            Ok(resp) => {
                // Capture usage tokens before resp is consumed by choices access.
                let resp_usage = resp.usage.clone();
                let mut response_model = resp.model.clone();
                let mut response_origin = infer_origin_from_model(&resp.model).to_string();
                let origin_emoji = match response_origin.as_str() {
                    "cloud" => "☁️",
                    "local" => "🏠",
                    _ => "❓",
                };
                tracing::info!(
                    "{} [Hera Generate] Response from {} engine — model: {}",
                    origin_emoji,
                    response_origin,
                    response_model
                );
                let mut result_text = String::new();
                let mut tool_calls: Option<serde_json::Value> = None;

                if let Some(choice) = resp.choices.first() {
                    if let Some(content) = &choice.message.content {
                        result_text = content.clone();
                    }

                    // 6. Parse and execute output tool calls — only when tools
                    // are allowed. A pure-generation caller (allow_tools=false)
                    // must get its text back verbatim, never routed through a
                    // tool execution + second-pass that discards the answer.
                    let mut parsed_calls = if parsed.context_budget.allow_tools {
                        crate::ai::tool_executor::parse_tool_calls(&result_text)
                    } else {
                        Vec::new()
                    };

                    if let (true, Some(tc_array)) =
                        (parsed.context_budget.allow_tools, &choice.message.tool_calls)
                    {
                        for tc in tc_array {
                            let mut extracted_name = None;
                            let mut extracted_args = None;
                            
                            if let (Some(name), Some(args)) = (
                                tc.get("name").and_then(|n| n.as_str()),
                                tc.get("arguments").or_else(|| tc.get("parameters")),
                            ) {
                                extracted_name = Some(name);
                                extracted_args = Some(args);
                            } else if let Some(func) = tc.get("function") {
                                if let (Some(name), Some(args)) = (
                                    func.get("name").and_then(|n| n.as_str()),
                                    func.get("arguments").or_else(|| func.get("parameters")),
                                ) {
                                    extracted_name = Some(name);
                                    extracted_args = Some(args);
                                }
                            }
                            
                            if let (Some(name), Some(args)) = (extracted_name, extracted_args) {
                                let mut args_val = args.clone();
                                if let Some(s) = args.as_str() {
                                    if let Ok(parsed_args) = serde_json::from_str(s) {
                                        args_val = parsed_args;
                                    }
                                }
                                parsed_calls.push(crate::ai::tool_executor::ToolCall {
                                    name: name.to_string(),
                                    arguments: args_val,
                                });
                            }
                        }
                    }

                    if !parsed_calls.is_empty() {
                        tracing::info!(
                            "🛠️ [Hera IPC] LLM emitted {} tool calls",
                            parsed_calls.len()
                        );
                        let tool_summary =
                            execute_parsed_tool_calls(&parsed_calls, &parsed, None).await;
                        let execution_outputs = tool_summary.execution_outputs;

                        if !tool_summary.has_media_call {
                            let json_mode = payload_clone
                                .get("json_mode")
                                .and_then(|v| v.as_bool())
                                .unwrap_or(false);
                            if let Some(req2) = prepare_tool_result_followup_request(
                                chat_req.clone(),
                                &result_text,
                                &execution_outputs,
                                json_mode,
                            ) {
                                tracing::info!(
                                    "🔄 [Hera IPC] Initiating second-pass generation to format Tool Results (json_mode: {})...",
                                    json_mode
                                );
                                match execute_tool_followup(
                                    &state.engine,
                                    req2,
                                    FollowupStrategy::Buffered,
                                )
                                .await
                                {
                                    Ok(followup) => {
                                        let p2_origin =
                                            followup.origin.as_deref().unwrap_or("unknown");
                                        let p2_model = followup.model.as_deref().unwrap_or("");
                                        tracing::info!(
                                            "🔄 [Hera Generate] Second-pass response from {} — model: {}",
                                            p2_origin,
                                            p2_model
                                        );
                                        response_model =
                                            followup.model.unwrap_or_else(String::new);
                                        response_origin = p2_origin.to_string();
                                        if !followup.text.is_empty() {
                                            result_text = followup.text;
                                        }
                                    }
                                    Err(e) => {
                                        tracing::error!("Second pass inference failed: {}", e);
                                        result_text.push_str(&format!(
                                            "\n\n[Error forming final response: {}]\n{}",
                                            e, execution_outputs
                                        ));
                                    }
                                }
                            }
                        } else {
                            result_text.push_str(&execution_outputs);
                        }

                        tool_calls =
                            Some(serde_json::Value::Array(tool_summary.executed_calls_json));
                    }
                }

                // Frame B3: quality cascade. If a non-tool local answer is poor
                // (empty / too short / disclaims) and didn't already come from
                // cloud, escalate once to the cloud failover. Sovereign-first:
                // this only fires on demonstrated local-quality failure, and only
                // replaces the answer if the escalation is actually better.
                //
                // Gated on the master cloud switch: with HERA_ALLOW_CLOUD_FALLBACK
                // unset (the default) this never touches a paid provider — it was
                // the missing gate behind the 2026-06-09 OpenRouter billing incident.
                if tool_calls.is_none()
                    && response_origin != "cloud"
                    && crate::ai::router::cloud_globally_enabled()
                {
                    let difficulty = super::difficulty::Difficulty::from_reasoning_effort(
                        &parsed.reasoning_effort,
                    );
                    if super::difficulty::is_low_quality_answer(&result_text, difficulty)
                        && let Some(mut cloud_req) = chat_req.clone()
                    {
                        cloud_req.provider = Some("cloud".to_string());
                        tracing::info!(
                            "⬆️ [Hera B3] Local answer low-quality (difficulty={}); escalating to cloud failover",
                            difficulty.as_str()
                        );
                        match state.engine.generate_content(cloud_req).await {
                            Ok(resp2) => {
                                if let Some(choice) = resp2.choices.first()
                                    && let Some(content) = &choice.message.content
                                    && !super::difficulty::is_low_quality_answer(content, difficulty)
                                {
                                    result_text = content.clone();
                                    response_origin = "cloud".to_string();
                                    response_model = resp2.model.clone();
                                }
                            }
                            Err(e) => {
                                tracing::warn!("B3 cloud escalation failed: {}", e);
                            }
                        }
                    }
                }

                let duration_ms = started_at.elapsed().as_millis() as u64;
                let outcome = build_runtime_outcome_artifacts(
                    "generate",
                    &parsed,
                    &prompt_assembly,
                    duration_ms,
                    None,
                    lightweight_mode,
                    &provider_requested,
                    &response_origin,
                    &response_model,
                    true,
                    tool_calls
                        .as_ref()
                        .and_then(|value| value.as_array().map(|items| items.len()))
                        .unwrap_or(0),
                    result_text.len(),
                    est_tokens,
                    chat_req
                        .as_ref()
                        .map(|req| req.messages.len())
                        .unwrap_or_default(),
                    None,
                );
                append_llm_audit_event(&outcome.audit_event);
                record_observation_and_promote_runtime_hint(
                    outcome.observation_payload,
                    RuntimePromotionContext {
                        preflight: runtime_preflight.clone(),
                        mode: "generate",
                        app_id: &parsed.app_name,
                        route_profile: &parsed.route_profile_id,
                        persona_path: &parsed.persona_path,
                        session_id: &parsed.session_id,
                        trace_id: &parsed.trace_id,
                        chat_id: &parsed.chat_id,
                        current_budget_mode: &parsed.context_budget.mode,
                        persona_drift: parsed.persona_drift,
                        success: true,
                    },
                )
                .await;

                if !lightweight_mode && parsed.context_budget.include_memory {
                    let user_id = canonicalize_user_id(
                        &parsed.sender_name,
                        &parsed.chat_id,
                        &parsed.session_id,
                    );
                    let app_id = parsed.app_name.clone();
                    let session_id = parsed.session_id.clone();
                    let user_content = parsed.prompt.clone();
                    let assistant_content = result_text.clone();
                    let attribution = prompt_assembly.recall_attribution.clone();
                    tokio::spawn(async move {
                        save_chat_turn_event(
                            user_id.clone(),
                            app_id.clone(),
                            session_id.clone(),
                            "user".to_string(),
                            user_content,
                        )
                        .await;
                        save_chat_turn_event(
                            user_id,
                            app_id,
                            session_id,
                            "assistant".to_string(),
                            assistant_content.clone(),
                        )
                        .await;
                        // Phase 2 flywheel: report which recalled ids the model cited
                        // so Memento can build (positives, negatives) training data.
                        report_recall_feedback(attribution.as_ref(), &assistant_content).await;
                    });
                }

                // Path B — usage logging (best-effort, fire-and-forget).
                //
                // resp.usage is normally populated correctly here (llama-server's
                // OpenAI-compatible response includes a real `usage` object, and
                // openai_compat.rs/gemini.rs both map it into ChatUsage — verified
                // live 2026-07-05 via a direct curl to :8080). It can still be
                // `None` for response shapes that don't carry it (e.g. a B3 cloud
                // escalation reuses the ORIGINAL local resp_usage, not resp2's —
                // a separate known gap, not fixed here). Rather than log a hard
                // 0/0/0 (which silently undercounts real spend/local-compute use
                // in hera_usage_events), fall back to the same cheap char/4
                // estimate already computed for the audit event (`est_tokens`)
                // for the prompt side, and estimate completion tokens the same
                // way from the final result_text. Approximate, but far closer
                // to reality than zero, and free (no extra engine round-trip).
                {
                    let (prompt_tokens, completion_tokens, total_tokens) = match &resp_usage {
                        Some(usage) => (
                            usage.prompt_tokens,
                            usage.completion_tokens,
                            usage.total_tokens,
                        ),
                        None => {
                            let est_prompt = est_tokens as u32;
                            let est_completion = (result_text.len() / 4) as u32;
                            (est_prompt, est_completion, est_prompt + est_completion)
                        }
                    };
                    spawn_log_usage(
                        parsed.app_name.clone(),
                        canonicalize_user_id(
                            &parsed.sender_name,
                            &parsed.chat_id,
                            &parsed.session_id,
                        ),
                        parsed.session_id.clone(),
                        parsed.route_profile_id.clone(),
                        response_model.clone(),
                        prompt_tokens,
                        completion_tokens,
                        total_tokens,
                        response_origin.contains("cloud"),
                        duration_ms,
                        parsed.trace_id.clone(),
                    );
                }

                inflight::remove(&parsed.trace_id);
                return HandlerOutcome::Result {
                    result_text,
                    origin: response_origin,
                    model: response_model,
                    tool_calls,
                };
            }
            Err(e) => {
                tracing::error!("LLM inference error: {}", e);
                let duration_ms = started_at.elapsed().as_millis() as u64;
                let error_text = e.to_string();
                let outcome = build_runtime_outcome_artifacts(
                    "generate",
                    &parsed,
                    &prompt_assembly,
                    duration_ms,
                    None,
                    lightweight_mode,
                    &provider_requested,
                    "offline",
                    "",
                    false,
                    0,
                    0,
                    est_tokens,
                    chat_req
                        .as_ref()
                        .map(|req| req.messages.len())
                        .unwrap_or_default(),
                    Some(error_text.clone()),
                );
                append_llm_audit_event(&outcome.audit_event);
                let _ = record_runtime_observation(outcome.observation_payload).await;
                inflight::remove(&parsed.trace_id);
                return HandlerOutcome::Result {
                    result_text: format!("Error: {}", error_text),
                    origin: "offline".to_string(),
                    model: String::new(),
                    tool_calls: None,
                };
            }
        }
    }

    HandlerOutcome::Result {
        result_text: "Action not supported".to_string(),
        origin: "unknown".to_string(),
        model: String::new(),
        tool_calls: None,
    }
}
