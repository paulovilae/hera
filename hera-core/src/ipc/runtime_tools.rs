use super::context::ParsedPayload;
use super::helpers::{
    fetch_single_app_schema_json, infer_origin_from_model, spawn_log_tool_call, telemetry_preview,
};
use super::types::IpcResponse;
use crate::ai::{ChatMessage, ChatRequest, LLMEngine, MessageContent};
use std::collections::HashSet;
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use tokio::net::UnixStream;

/// Max chars kept for a tool arg/result preview in durable telemetry.
const TOOL_PREVIEW_CHARS: usize = 2000;

pub struct ToolExecutionSummary {
    pub execution_outputs: String,
    pub executed_calls_json: Vec<serde_json::Value>,
    pub executed_tool_count: usize,
    pub has_media_call: bool,
    /// (tool_name, success) per call this round — lets the agentic loop reason
    /// about whether an edit was followed by a green verification.
    pub executed_results: Vec<(String, bool)>,
}

pub enum FollowupStrategy<'a> {
    Buffered,
    Streaming(&'a mut UnixStream),
}

pub struct FollowupExecutionResult {
    pub text: String,
    pub model: Option<String>,
    pub origin: Option<String>,
}

pub fn contextualize_tool_call(
    tool_call: &crate::ai::tool_executor::ToolCall,
    parsed: &ParsedPayload,
) -> crate::ai::tool_executor::ToolCall {
    let mut arguments = tool_call.arguments.clone();
    let object = arguments
        .as_object_mut()
        .expect("tool call arguments should always be an object");

    if !parsed.app_name.is_empty() {
        object
            .entry("app_name".to_string())
            .or_insert_with(|| serde_json::json!(parsed.app_name));
        object
            .entry("app".to_string())
            .or_insert_with(|| serde_json::json!(parsed.app_name));
    }
    if !parsed.trace_id.is_empty() {
        object
            .entry("trace_id".to_string())
            .or_insert_with(|| serde_json::json!(parsed.trace_id));
    }
    if !parsed.route_profile_id.is_empty() {
        object
            .entry("route_profile".to_string())
            .or_insert_with(|| serde_json::json!(parsed.route_profile_id));
    }
    if !parsed.session_id.is_empty() {
        object
            .entry("session_id".to_string())
            .or_insert_with(|| serde_json::json!(parsed.session_id));
    }
    if !parsed.chat_id.is_empty() {
        object
            .entry("chat_id".to_string())
            .or_insert_with(|| serde_json::json!(parsed.chat_id));
    }
    object.entry("caller".to_string()).or_insert_with(|| {
        serde_json::json!(if parsed.app_name.is_empty() {
            "hera"
        } else {
            &parsed.app_name
        })
    });
    if !parsed.persona_path.is_empty() {
        // Trusted server-side value. INSERT (overwrite) rather than entry().or_insert so a
        // bot cannot spoof its workspace target via a model-supplied `_persona_path`.
        // Consumed by the character-workspace tools to confine writes to the bot's own
        // Agents/workspaces/{name}/ directory.
        object.insert(
            "_persona_path".to_string(),
            serde_json::json!(parsed.persona_path),
        );
    }

    crate::ai::tool_executor::ToolCall {
        name: tool_call.name.clone(),
        arguments,
    }
}

pub async fn execute_parsed_tool_calls(
    parsed_calls: &[crate::ai::tool_executor::ToolCall],
    parsed: &ParsedPayload,
    mut status_stream: Option<&mut UnixStream>,
) -> ToolExecutionSummary {
    let mut execution_outputs = String::new();
    let mut executed_calls_json = Vec::new();
    let mut executed_tool_count = 0usize;
    let mut executed_results: Vec<(String, bool)> = Vec::new();

    for (seq, call) in parsed_calls.iter().enumerate() {
        let args_preview = telemetry_preview(&call.arguments.to_string(), TOOL_PREVIEW_CHARS);

        if let Some(stream) = status_stream.as_deref_mut() {
            // Enriched live event: the chat frontend ignores the extra fields
            // (backward-compatible), while an ops subscriber can now show
            // trace_id + what the tool received, not just its name.
            let status_msg = IpcResponse {
                status: "tool_status".to_string(),
                data: serde_json::json!({
                    "name": call.name.clone(),
                    "trace_id": parsed.trace_id.clone(),
                    "args_preview": args_preview.clone(),
                }),
            };
            let mut str_msg = serde_json::to_string(&status_msg).unwrap();
            str_msg.push('\n');
            let _ = stream.write_all(str_msg.as_bytes()).await;
        }

        if crate::ai::tool_executor::permissions_allow_tool(&parsed.permissions, &call.name) {
            let contextual_call = contextualize_tool_call(call, parsed);
            // Durable per-tool-call telemetry is emitted inside `execute_tool`
            // itself (the universal chokepoint) so the fast-path, streaming and
            // retry paths are covered too — not just this parsed-loop.
            let tool_res = crate::ai::tool_executor::execute_tool(&contextual_call).await;
            executed_tool_count += 1;
            execution_outputs.push_str(&format!("\n\n{}", tool_res.output));
            executed_results.push((contextual_call.name.clone(), tool_res.success));
            executed_calls_json.push(serde_json::json!({
                "name": contextual_call.name,
                "arguments": contextual_call.arguments
            }));
        } else {
            tracing::warn!(
                "⚠️ [Hera IPC] LLM hallucinated tool {} which is denied by permissions",
                call.name
            );
            execution_outputs.push_str(&format!(
                "\n\nError: Not permitted to use tool '{}'",
                call.name
            ));
            executed_results.push((call.name.clone(), false));

            // Record the denied attempt too — a spike of these is a signal.
            spawn_log_tool_call(
                parsed.trace_id.clone(),
                parsed.session_id.clone(),
                parsed.app_name.clone(),
                parsed.route_profile_id.clone(),
                seq as u32,
                call.name.clone(),
                args_preview,
                String::new(),
                0,
                false,
                Some("denied by permissions".to_string()),
            );
        }
    }

    let has_media_call = parsed_calls.iter().any(|call| {
        matches!(
            call.name.as_str(),
            "hera_draw" | "hera_video" | "generate_qr_code"
        )
    }) || execution_outputs.contains("MEDIA: ");

    ToolExecutionSummary {
        execution_outputs,
        executed_calls_json,
        executed_tool_count,
        has_media_call,
        executed_results,
    }
}

pub async fn try_plan_schema_query(
    engine: &Arc<dyn LLMEngine + Send + Sync>,
    parsed: &ParsedPayload,
) -> Option<crate::ai::tool_executor::ToolCall> {
    plan_schema_query_internal(engine, parsed, None).await
}

pub async fn retry_plan_schema_query(
    engine: &Arc<dyn LLMEngine + Send + Sync>,
    parsed: &ParsedPayload,
    previous_query: &str,
    previous_result: &str,
) -> Option<crate::ai::tool_executor::ToolCall> {
    let feedback = format!(
        "Previous query attempt:\n{}\n\nPrevious result:\n{}\n\n\
Choose a different table/column strategy if the previous query failed, referenced a missing column, \
or returned zero rows despite the runtime context identifying the subject.",
        previous_query, previous_result
    );
    let replanned = plan_schema_query_internal(engine, parsed, Some(&feedback)).await?;
    let new_query = replanned
        .arguments
        .get("query")
        .and_then(|value| value.as_str())
        .unwrap_or("")
        .trim();
    if new_query.eq_ignore_ascii_case(previous_query.trim()) {
        return None;
    }
    Some(replanned)
}

fn normalized_slug(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>()
        .split('-')
        .filter(|segment| !segment.is_empty())
        .collect::<Vec<_>>()
        .join("-")
}

fn structured_runtime_context(parsed: &ParsedPayload) -> String {
    let mut lines = Vec::new();
    let mut workspace_user = None;
    let mut workspace_company_slug = None;
    if !parsed.sender_name.is_empty() {
        lines.push(format!("sender_name: {}", parsed.sender_name));
    }
    if !parsed.page_title.is_empty() {
        lines.push(format!("page_title: {}", parsed.page_title));
    }
    if !parsed.page_url.is_empty() {
        lines.push(format!("page_url: {}", parsed.page_url));
    }
    if !parsed.page_context.is_empty() {
        lines.push(format!("page_context_raw: {}", parsed.page_context));
        if let Ok(page_context) =
            serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(&parsed.page_context)
        {
            for (key, value) in &page_context {
                if let Some(text) = value.as_str() {
                    lines.push(format!("{key}: {text}"));
                    if key == "workspace_user" && !text.trim().is_empty() {
                        workspace_user = Some(text.to_string());
                    }
                }
            }
            if let Some(company) = page_context
                .get("workspace_company")
                .and_then(|value| value.as_str())
            {
                let slug = page_context
                    .get("workspace_company_slug")
                    .and_then(|value| value.as_str())
                    .filter(|value| !value.trim().is_empty())
                    .map(str::to_string)
                    .unwrap_or_else(|| normalized_slug(company));
                if !slug.is_empty() {
                    lines.push(format!("workspace_company_slug: {}", slug));
                    workspace_company_slug = Some(slug);
                }
            }
        }
    }
    if let Some(user) = workspace_user {
        lines.push(format!(
            "runtime_filter_hint_user: use '{}' for person-scoped columns such as user_id, applicant_user_id, owner_email, assignee_user_id, uploaded_by, created_by, or *_email",
            user
        ));
    }
    if let Some(company_slug) = workspace_company_slug {
        lines.push(format!(
            "runtime_filter_hint_org: use '{}' for organization-scoped columns such as company_id, client_id, workspace_id, account_id, or owner_id",
            company_slug
        ));
    }
    if lines.is_empty() {
        String::new()
    } else {
        format!("\nRuntime context:\n{}", lines.join("\n"))
    }
}

fn normalized_token(value: &str) -> String {
    value.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
}

fn tokenize(value: &str) -> Vec<String> {
    normalized_token(value)
        .split_whitespace()
        .filter(|token| token.len() >= 2)
        .map(|token| token.to_string())
        .collect()
}

fn is_first_person_request(prompt: &str) -> bool {
    let normalized = normalized_token(prompt);
    [
        " my ",
        " mine ",
        " our ",
        " mis ",
        " mi ",
        " mio ",
        " mia ",
        " mias ",
        " mios ",
        " nuestro ",
        " nuestra ",
        " nuestros ",
        " nuestras ",
    ]
    .iter()
    .any(|needle| normalized.contains(needle))
}

fn schema_column_names(cols: &serde_json::Value) -> Vec<String> {
    cols.as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|c| c.get("column").and_then(|n| n.as_str()))
                .map(|name| name.to_string())
                .collect()
        })
        .unwrap_or_default()
}

fn runtime_identifier_tokens(parsed: &ParsedPayload) -> HashSet<String> {
    let mut tokens = HashSet::new();
    if !parsed.sender_name.is_empty() {
        tokens.extend(tokenize(&parsed.sender_name));
    }
    if let Ok(page_context) =
        serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(&parsed.page_context)
    {
        for value in page_context.values() {
            if let Some(text) = value.as_str() {
                tokens.extend(tokenize(text));
            }
        }
    }
    tokens
}

fn has_runtime_ownership_columns(columns: &[String]) -> bool {
    columns.iter().any(|column| {
        column.ends_with("_id")
            || column.ends_with("_email")
            || column.contains("owner")
            || column.contains("applicant")
            || column.contains("client")
            || column.contains("company")
            || column.contains("party_")
            || column == "user_id"
    })
}

fn ownership_column_strength(columns: &[String]) -> (usize, usize) {
    let mut direct = 0usize;
    let mut indirect = 0usize;
    for column in columns {
        if matches!(
            column.as_str(),
            "user_id" | "company_id" | "client_id" | "owner_id" | "owner_email"
                | "applicant_user_id" | "assignee_user_id" | "created_by" | "uploaded_by"
        ) || column.starts_with("owner_")
            || column.starts_with("applicant_")
            || column.starts_with("assignee_")
        {
            direct += 1;
        } else if column.ends_with("_email") || column.contains("party_") {
            indirect += 1;
        }
    }
    (direct, indirect)
}

fn reduce_schema_for_planner(
    schema: &serde_json::Map<String, serde_json::Value>,
    parsed: &ParsedPayload,
) -> serde_json::Map<String, serde_json::Value> {
    if schema.len() <= 12 {
        return schema.clone();
    }

    let prompt_tokens: HashSet<String> = tokenize(&parsed.prompt).into_iter().collect();
    let runtime_tokens = runtime_identifier_tokens(parsed);
    let prefer_runtime_ownership = !runtime_tokens.is_empty() && is_first_person_request(&parsed.prompt);

    let mut scored_tables = schema
        .iter()
        .map(|(table, cols)| {
            let columns = schema_column_names(cols);
            let mut score = 0usize;
            let (direct_ownership_columns, indirect_ownership_columns) =
                ownership_column_strength(&columns);

            let table_tokens: HashSet<String> = tokenize(table).into_iter().collect();
            let matched_prompt_terms = prompt_tokens
                .iter()
                .filter(|token| {
                    table_tokens.contains(*token)
                        || columns.iter().any(|column| tokenize(column).iter().any(|part| part == *token))
                })
                .count();
            score += matched_prompt_terms * 6;

            let matched_runtime_terms = runtime_tokens
                .iter()
                .filter(|token| {
                    table_tokens.contains(*token)
                        || columns.iter().any(|column| tokenize(column).iter().any(|part| part == *token))
                })
                .count();
            score += matched_runtime_terms * 4;

            if prefer_runtime_ownership && has_runtime_ownership_columns(&columns) {
                score += 12;
                score += direct_ownership_columns * 7;
                score += indirect_ownership_columns * 2;
            }

            if columns.iter().any(|column| column == "created_at" || column == "updated_at") {
                score += 1;
            }

            (
                table.clone(),
                cols.clone(),
                score,
                matched_prompt_terms,
                direct_ownership_columns,
            )
        })
        .collect::<Vec<_>>();

    scored_tables.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| a.0.cmp(&b.0)));

    let direct_owner_candidates_exist = prefer_runtime_ownership
        && scored_tables
            .iter()
            .any(|(_, _, score, _matched_prompt_terms, direct_ownership_columns)| {
                *score > 0 && *direct_ownership_columns > 0
            });

    let kept = scored_tables
        .iter()
        .filter(|(_, _, _, _matched_prompt_terms, direct_ownership_columns)| {
            !direct_owner_candidates_exist || *direct_ownership_columns > 0
        })
        .take_while(|(_, _, score, _, _)| *score > 0)
        .take(12)
        .map(|(table, cols, _, _, _)| (table.clone(), cols.clone()))
        .collect::<serde_json::Map<_, _>>();

    if kept.is_empty() {
        schema.clone()
    } else {
        kept
    }
}

pub(crate) fn should_retry_schema_query(result_text: &str) -> bool {
    result_text.contains("returned 0 results")
        || result_text.contains("Query error:")
        || result_text.contains("does not exist")
        || result_text.contains("Memento error:")
}

async fn plan_schema_query_internal(
    engine: &Arc<dyn LLMEngine + Send + Sync>,
    parsed: &ParsedPayload,
    feedback: Option<&str>,
) -> Option<crate::ai::tool_executor::ToolCall> {
    if parsed.app_name.is_empty()
        || !parsed.permissions.iter().any(|perm| perm == "memento_query")
        || parsed.prompt.trim().is_empty()
        || parsed.prompt.trim_start().starts_with('/')
    {
        return None;
    }

    let schema = fetch_single_app_schema_json(&parsed.app_name).await?;
    if schema.is_empty() {
        return None;
    }

    let conversation_context = if parsed.recent_messages.is_empty() {
        String::new()
    } else {
        let start = parsed.recent_messages.len().saturating_sub(6);
        let excerpt = parsed.recent_messages[start..]
            .iter()
            .map(|(role, content)| format!("{role}: {content}"))
            .collect::<Vec<_>>()
            .join("\n");
        format!("\nRecent conversation:\n{}", excerpt)
    };
    let runtime_context = structured_runtime_context(parsed);
    let retry_feedback = feedback
        .map(|value| format!("\nPlanner feedback:\n{}", value))
        .unwrap_or_default();

    let reduced_schema = reduce_schema_for_planner(&schema, parsed);
    tracing::info!(
        "🧠 [Hera IPC] Schema planner app='{}' reduced tables {} -> {}",
        parsed.app_name,
        schema.len(),
        reduced_schema.len()
    );
    let schema_json = serde_json::to_string_pretty(&reduced_schema).ok()?;
    let planner_system = format!(
        "You are Hera's generic schema-aware query planner.\n\
Return only one raw JSON object.\n\
Never explain. Never use markdown. Never emit <tool_call> tags.\n\
Given an app schema and a user request, decide whether a SQL query is required.\n\
Allowed output schema:\n\
{{\"should_query\":true,\"query\":\"SELECT ...\",\"limit\":50,\"reason\":\"short\"}}\n\
or\n\
{{\"should_query\":false,\"reason\":\"short\"}}\n\
Rules:\n\
- SQL must be SELECT or WITH only.\n\
- Use only tables and columns present in the schema.\n\
- Prefer concise queries.\n\
- If aggregating numeric columns with SUM/AVG, CAST the aggregate to double precision so JSON transport stays typed.\n\
- If the user asks for grouping, include grouped dimensions.\n\
- Do not invent categorical filters or enum values unless the user explicitly asked for that category.\n\
- Do not guess status values, workflow states, or labels such as pending/active/completed unless they were stated by the user or are unambiguously required by runtime context.\n\
- For global totals like the total value of a portfolio, prefer summing an explicit total column or the whole table without filters unless the user requested a subset.\n\
- Resolve ambiguous follow-up questions using the recent conversation when it is provided.\n\
- Use runtime context (current debtor/account/page context) when it identifies the subject of the request.\n\
- If runtime context already identifies the debtor or account reference, prefer that context instead of asking the user to repeat it.\n\
- If runtime context includes an exact identifier such as *_id, document_id, account_reference, reference, uuid, or pid, prefer filtering with that exact identifier over human names.\n\
- If runtime context includes user/company identifiers, treat first-person requests like 'my', 'mis', 'mine', 'our', or 'nuestro' as user-scoped and add ownership filters.\n\
- Prefer tables that can be filtered by the runtime identifiers already present in context, such as company_id, client_id, owner_id, owner_email, applicant_user_id, user_id, or party_*_email.\n\
- When multiple tables match the domain noun (for example a summary table versus a dossier table), prefer the table that contains both the requested business fields and the ownership filter columns required by runtime context.\n\
- Distinguish person-scoped identifiers from organization-scoped identifiers. Use user/email runtime values for person-scoped columns, and company/account slug runtime values for organization-scoped columns.\n\
- Prefer direct ownership columns such as company_id, client_id, workspace_id, account_id, owner_id, owner_email, applicant_user_id, or user_id over weaker contact columns such as party_*, name, or generic email fields when both could match.\n\
- If the previous query failed because a column was missing or returned zero rows, do not reuse the same table/column combination on retry.\n\
- Avoid filtering by name alone when runtime context already includes a more specific identifier.\n\
- If the request can be answered without data access, set should_query=false.\n\
App: {}{}{}{}\nSchema:\n{}",
        parsed.app_name, conversation_context, runtime_context, retry_feedback, schema_json
    );

    let req = ChatRequest {
        model: "hera-local-model".to_string(),
        vision_model: None,
        tts_model: None,
        stt_model: None,
        messages: vec![
            ChatMessage {
                role: "system".to_string(),
                content: MessageContent::Text(planner_system),
            },
            ChatMessage {
                role: "user".to_string(),
                content: MessageContent::Text(parsed.prompt.clone()),
            },
        ],
        temperature: Some(0.0),
        max_tokens: Some(300),
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
        reasoning_effort: Some("medium".to_string()),
        response_format: None,
        app: None,
        priority: None,
    };

    let resp = engine.generate_content(req).await.ok()?;
    let content = resp
        .choices
        .first()
        .and_then(|choice| choice.message.content.clone())?;
    let plan_value = parse_first_json_object(&content)?;

    if !plan_value
        .get("should_query")
        .and_then(|value| value.as_bool())
        .unwrap_or(false)
    {
        return None;
    }

    let query = plan_value.get("query").and_then(|value| value.as_str())?;
    if !is_safe_select_query(query, &reduced_schema) {
        tracing::warn!(
            app = %parsed.app_name,
            "Rejected schema planner query because validation failed: {}",
            query
        );
        return None;
    }

    Some(crate::ai::tool_executor::ToolCall {
        name: "memento_query".to_string(),
        arguments: serde_json::json!({
            "app": parsed.app_name,
            "query": query,
            "limit": plan_value.get("limit").and_then(|value| value.as_u64()).unwrap_or(50)
        }),
    })
}

pub async fn summarize_tool_output_for_user(
    engine: &Arc<dyn LLMEngine + Send + Sync>,
    parsed: &ParsedPayload,
    tool_output: &str,
) -> Option<String> {
    let req = ChatRequest {
        model: "hera-local-model".to_string(),
        vision_model: None,
        tts_model: None,
        stt_model: None,
        messages: vec![
            ChatMessage {
                role: "system".to_string(),
                content: MessageContent::Text(
                    "You are Hera. Summarize tool results for the user in the same language as the original question. Be concise, clear, and directly answer the request. Do not mention SQL, tables, or internal tools.".to_string(),
                ),
            },
            ChatMessage {
                role: "user".to_string(),
                content: MessageContent::Text(format!(
                    "Original request:\n{}\n\nTool result:\n{}",
                    parsed.prompt, tool_output
                )),
            },
        ],
        temperature: Some(0.1),
        max_tokens: Some(300),
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
        reasoning_effort: Some("low".to_string()),
        response_format: None,
        app: None,
        priority: None,
    };

    let resp = engine.generate_content(req).await.ok()?;
    resp.choices
        .first()
        .and_then(|choice| choice.message.content.clone())
        .filter(|text| !text.trim().is_empty())
}

fn parse_first_json_object(text: &str) -> Option<serde_json::Value> {
    let trimmed = if let Some(end_idx) = text.find("</think>") {
        text[end_idx + "</think>".len()..].trim()
    } else {
        text.trim()
    };

    let start = trimmed.find('{')?;
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;
    let mut end = None;

    for (idx, ch) in trimmed[start..].char_indices() {
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
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(start + idx + 1);
                    break;
                }
            }
            _ => {}
        }
    }

    let end = end?;
    serde_json::from_str::<serde_json::Value>(&trimmed[start..end]).ok()
}

fn is_safe_select_query(
    query: &str,
    schema: &serde_json::Map<String, serde_json::Value>,
) -> bool {
    let normalized = query.trim().to_lowercase();
    if !(normalized.starts_with("select") || normalized.starts_with("with")) {
        return false;
    }
    if normalized.contains(';') {
        return false;
    }
    for forbidden in ["insert ", "update ", "delete ", "drop ", "alter ", "truncate "] {
        if normalized.contains(forbidden) {
            return false;
        }
    }

    let known_tables: Vec<String> = schema.keys().cloned().collect();
    known_tables.iter().any(|table| normalized.contains(table))
}


pub async fn execute_tool_followup(
    engine: &Arc<dyn LLMEngine + Send + Sync>,
    request: ChatRequest,
    strategy: FollowupStrategy<'_>,
) -> Result<FollowupExecutionResult, String> {
    match strategy {
        FollowupStrategy::Buffered => {
            let response = engine
                .generate_content(request)
                .await
                .map_err(|error| error.to_string())?;
            let model = response.model.clone();
            let origin = infer_origin_from_model(&model).to_string();
            let text = response
                .choices
                .first()
                .and_then(|choice| choice.message.content.clone())
                .unwrap_or_default();

            Ok(FollowupExecutionResult {
                text,
                model: Some(model),
                origin: Some(origin),
            })
        }
        FollowupStrategy::Streaming(stream) => {
            let mut rx = engine
                .generate_stream(request)
                .await
                .map_err(|error| error.to_string())?;
            let mut text = String::new();
            let mut model = None;
            let mut origin = None;

            while let Some(chunk_res) = rx.recv().await {
                let chunk = chunk_res.map_err(|error| error.to_string())?;
                if model.is_none() && !chunk.model.is_empty() {
                    model = Some(chunk.model.clone());
                    origin = Some(infer_origin_from_model(&chunk.model).to_string());
                }

                let chunk_text = chunk
                    .choices
                    .first()
                    .and_then(|choice| choice.delta.content.clone())
                    .unwrap_or_default();
                if chunk_text.is_empty() {
                    continue;
                }

                text.push_str(&chunk_text);
                let chunk_msg = IpcResponse {
                    status: "chunk".to_string(),
                    data: serde_json::json!({ "text": chunk_text }),
                };
                let mut cstr = serde_json::to_string(&chunk_msg).unwrap();
                cstr.push('\n');
                let _ = stream.write_all(cstr.as_bytes()).await;
            }

            Ok(FollowupExecutionResult {
                text,
                model,
                origin,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::{
        ChatChoice, ChatRequest, ChatResponse, ChatResponseMessage, ChatStreamChoice,
        ChatStreamDelta, ChatStreamResponse, InferenceError, MessageContent,
    };
    use std::sync::Arc;
    use tokio::io::AsyncReadExt;
    use tokio::sync::mpsc;

    struct MockEngine;

    fn minimal_request() -> ChatRequest {
        ChatRequest {
            model: "hera-local-model".to_string(),
            vision_model: None,
            tts_model: None,
            stt_model: None,
            messages: vec![crate::ai::ChatMessage {
                role: "user".to_string(),
                content: MessageContent::Text("hi".to_string()),
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
            provider: None,
            stream: None,
            nsfw: None,
            tools: None,
            tool_choice: None,
            reasoning_effort: None,
            response_format: None,
            app: None,
            priority: None,
        }
    }

    #[async_trait::async_trait]
    impl LLMEngine for MockEngine {
        async fn generate_content(
            &self,
            _req: ChatRequest,
        ) -> Result<ChatResponse, InferenceError> {
            Ok(ChatResponse {
                id: "resp_1".to_string(),
                object: "chat.completion".to_string(),
                created: 0,
                model: "mock-local-model".to_string(),
                choices: vec![ChatChoice {
                    index: 0,
                    message: ChatResponseMessage {
                        role: "assistant".to_string(),
                        content: Some("buffered followup".to_string()),
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
            let (tx, rx) = mpsc::channel(4);
            tokio::spawn(async move {
                let _ = tx
                    .send(Ok(ChatStreamResponse {
                        id: "stream_1".to_string(),
                        object: "chat.completion.chunk".to_string(),
                        created: 0,
                        model: "mock-local-stream-model".to_string(),
                        choices: vec![ChatStreamChoice {
                            index: 0,
                            delta: ChatStreamDelta {
                                role: None,
                                content: Some("streamed followup".to_string()),
                                tool_calls: None,
                            },
                            finish_reason: Some("stop".to_string()),
                        }],
                        stats: None,
                    }))
                    .await;
            });
            Ok(rx)
        }
    }

    #[tokio::test]
    async fn execute_tool_followup_buffered_returns_text_and_origin() {
        let engine: Arc<dyn LLMEngine + Send + Sync> = Arc::new(MockEngine);
        let result = execute_tool_followup(&engine, minimal_request(), FollowupStrategy::Buffered)
            .await
            .expect("buffered followup should succeed");

        assert_eq!(result.text, "buffered followup");
        assert_eq!(result.model.as_deref(), Some("mock-local-model"));
        assert_eq!(result.origin.as_deref(), Some("local"));
    }

    #[tokio::test]
    async fn execute_tool_followup_streaming_writes_chunk_and_returns_text() {
        let engine: Arc<dyn LLMEngine + Send + Sync> = Arc::new(MockEngine);
        let (mut writer, mut reader) = tokio::net::UnixStream::pair().expect("unix pair");

        let result = execute_tool_followup(
            &engine,
            minimal_request(),
            FollowupStrategy::Streaming(&mut writer),
        )
        .await
        .expect("streaming followup should succeed");

        let mut buf = vec![0u8; 4096];
        let n = tokio::time::timeout(
            std::time::Duration::from_millis(250),
            reader.read(&mut buf),
        )
        .await
        .expect("chunk should be written")
        .expect("read should succeed");
        let written = String::from_utf8_lossy(&buf[..n]);

        assert!(written.contains("\"status\":\"chunk\""));
        assert!(written.contains("streamed followup"));
        assert_eq!(result.text, "streamed followup");
        assert_eq!(result.model.as_deref(), Some("mock-local-stream-model"));
        assert_eq!(result.origin.as_deref(), Some("local"));
    }
}
