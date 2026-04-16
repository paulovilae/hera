use std::collections::HashMap;
use std::fs;
use std::path::Path;

use serde::Serialize;

use super::helpers::fetch_runtime_preflight;
use super::llm_audit::LlmAuditEvent;
use super::route_profiles::all_route_profiles;
use super::types::{HandlerOutcome, IpcPayload, IpcResponse, IpcState, send_ipc_response};

#[derive(Debug, Clone, Serialize)]
struct RouteHealthSummary {
    route_profile: String,
    app: String,
    total_events: usize,
    error_events: usize,
    persona_drift_events: usize,
    p95_ms: Option<u64>,
    first_token_p95_ms: Option<u64>,
    target_p95_ms: u64,
    target_first_token_ms: Option<u64>,
    learned_hint_count: usize,
    active_negative_hint: bool,
    recommended_budget_mode: Option<String>,
    status: String,
    recommendations: Vec<String>,
}

fn percentile_u64(mut values: Vec<u64>, fraction: f64) -> Option<u64> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable();
    let idx = ((values.len().saturating_sub(1)) as f64 * fraction).round() as usize;
    values.get(idx).copied()
}

fn load_audit_events() -> Vec<LlmAuditEvent> {
    let path = Path::new("/tmp/hera_llm_audit.jsonl");
    let Ok(content) = fs::read_to_string(path) else {
        return Vec::new();
    };
    content
        .lines()
        .filter_map(|line| serde_json::from_str::<LlmAuditEvent>(line).ok())
        .collect()
}

fn baseline_metrics_for(profile_id: &str) -> Option<serde_json::Value> {
    let filename = match profile_id {
        "vetra_widget" => "vetra_http_stream.json",
        "movilo_widget" => "movilo_http_stream.json",
        "consulting_widget" => "consulting_http_stream.json",
        "latinos_widget" => "latinos_http_stream.json",
        "cartera_widget" => "cartera_http_stream.json",
        _ => return None,
    };
    let path = Path::new("/home/paulo/Programs/apps/OS/benchmarks/baselines").join(filename);
    let Ok(content) = fs::read_to_string(path) else {
        return None;
    };
    serde_json::from_str::<serde_json::Value>(&content).ok()
}

pub async fn handle_route_health(
    request: &IpcPayload,
    _state: &IpcState,
    stream: &mut tokio::net::UnixStream,
) -> HandlerOutcome {
    let app_filter = request
        .payload
        .get("app")
        .and_then(|value| value.as_str())
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| !value.is_empty());

    let mut grouped: HashMap<String, Vec<LlmAuditEvent>> = HashMap::new();
    for event in load_audit_events() {
        if let Some(app) = &app_filter
            && event.app.to_ascii_lowercase() != *app
        {
            continue;
        }
        grouped
            .entry(event.route_profile.clone())
            .or_default()
            .push(event);
    }

    let mut summaries = Vec::new();
    let profiles = all_route_profiles();
    for profile in profiles {
        if let Some(app) = &app_filter
            && !profile.app.is_empty()
            && profile.app != app
        {
            continue;
        }
        let route_profile = profile.id.to_string();
        let events = grouped.remove(profile.id).unwrap_or_default();
        if events.is_empty() {
            let preflight = fetch_runtime_preflight(
                profile.app,
                profile.id,
                profile.persona_path,
                if profile.prefer_stream {
                    "generate_stream"
                } else {
                    "generate"
                },
            )
            .await;
            let learned_hint_count = preflight
                .as_ref()
                .and_then(|value| value.get("learned_hints"))
                .and_then(|value| value.as_array())
                .map(|items| items.len())
                .unwrap_or(0);
            let active_negative_hint = preflight
                .as_ref()
                .and_then(|value| value.get("learned_hints"))
                .and_then(|value| value.as_array())
                .map(|items| {
                    items.iter().any(|item| {
                        item.pointer("/data/hint_kind")
                            .and_then(|value| value.as_str())
                            == Some("negative")
                    })
                })
                .unwrap_or(false);
            let recommended_budget_mode = preflight
                .as_ref()
                .and_then(|value| value.get("recommended_budget_mode"))
                .and_then(|value| value.as_str())
                .map(str::to_string);
            let mut recommendations =
                vec!["No recent Hera audit events for this route profile.".to_string()];
            if let Some(preflight) = preflight.as_ref()
                && let Some(warnings) = preflight.get("warnings").and_then(|value| value.as_array())
            {
                recommendations.extend(
                    warnings
                        .iter()
                        .filter_map(|value| value.as_str().map(str::to_string)),
                );
            }
            summaries.push(RouteHealthSummary {
                route_profile,
                app: profile.app.to_string(),
                total_events: 0,
                error_events: 0,
                persona_drift_events: 0,
                p95_ms: None,
                first_token_p95_ms: None,
                target_p95_ms: profile.target_p95_ms,
                target_first_token_ms: profile.target_first_token_ms,
                learned_hint_count,
                active_negative_hint,
                recommended_budget_mode,
                status: "unknown".to_string(),
                recommendations,
            });
            continue;
        }
        let durations = events
            .iter()
            .map(|item| item.duration_ms)
            .collect::<Vec<_>>();
        let first_token = events
            .iter()
            .filter_map(|item| item.first_token_ms)
            .collect::<Vec<_>>();
        let error_events = events.iter().filter(|item| !item.success).count();
        let persona_drift_events = events.iter().filter(|item| item.persona_drift).count();
        let p95_ms = percentile_u64(durations, 0.95);
        let first_token_p95_ms = percentile_u64(first_token, 0.95);
        let mut recommendations = Vec::new();
        let preflight = fetch_runtime_preflight(
            profile.app,
            profile.id,
            profile.persona_path,
            if profile.prefer_stream {
                "generate_stream"
            } else {
                "generate"
            },
        )
        .await;
        let learned_hint_count = preflight
            .as_ref()
            .and_then(|value| value.get("learned_hints"))
            .and_then(|value| value.as_array())
            .map(|items| items.len())
            .unwrap_or(0);
        let active_negative_hint = preflight
            .as_ref()
            .and_then(|value| value.get("learned_hints"))
            .and_then(|value| value.as_array())
            .map(|items| {
                items.iter().any(|item| {
                    item.pointer("/data/hint_kind")
                        .and_then(|value| value.as_str())
                        == Some("negative")
                })
            })
            .unwrap_or(false);
        let recommended_budget_mode = preflight
            .as_ref()
            .and_then(|value| value.get("recommended_budget_mode"))
            .and_then(|value| value.as_str())
            .map(str::to_string);

        if persona_drift_events > 0 {
            recommendations.push(
                "Persona drift detected; align app route with route_profile persona_path."
                    .to_string(),
            );
        }
        if let Some(p95) = p95_ms
            && p95 > profile.target_p95_ms
        {
            recommendations.push(format!(
                "P95 latency {} ms exceeds route target {} ms; inspect prompt budget and tool/schema inflation.",
                p95, profile.target_p95_ms
            ));
        }
        if let (Some(actual), Some(target)) = (first_token_p95_ms, profile.target_first_token_ms)
            && actual > target
        {
            recommendations.push(format!(
                "First-token P95 {} ms exceeds target {} ms; prefer streaming path and lighter context budgets.",
                actual, target
            ));
        }
        if let Some(baseline) = baseline_metrics_for(&route_profile) {
            if let Some(metrics) = baseline.get("metrics") {
                let baseline_p95 = metrics.get("p95_ms").and_then(|value| value.as_f64());
                if let (Some(actual), Some(previous)) = (p95_ms, baseline_p95)
                    && previous > 0.0
                {
                    let regression = ((actual as f64 - previous) / previous) * 100.0;
                    if regression > 20.0 {
                        recommendations.push(format!(
                            "P95 regression {:.2}% versus current baseline; mark route degraded and re-run bench suite.",
                            regression
                        ));
                    }
                }
            }
        }
        if let Some(preflight) = preflight.as_ref() {
            if let Some(warnings) = preflight.get("warnings").and_then(|value| value.as_array()) {
                recommendations.extend(
                    warnings
                        .iter()
                        .filter_map(|value| value.as_str().map(str::to_string)),
                );
            }
            if learned_hint_count > 0 {
                recommendations.push(format!(
                    "Memento has {} learned runtime hint(s) for this route; keep execution aligned with promoted heuristics.",
                    learned_hint_count
                ));
            }
            if active_negative_hint {
                recommendations.push(
                    "Active negative runtime hint present; treat this route as degraded until the hint expires or is superseded."
                        .to_string(),
                );
            }
        }

        let status = if error_events > 0 || persona_drift_events > 0 || active_negative_hint {
            "degraded"
        } else if recommendations.is_empty() {
            "healthy"
        } else {
            "watch"
        };

        summaries.push(RouteHealthSummary {
            route_profile,
            app: profile.app.to_string(),
            total_events: events.len(),
            error_events,
            persona_drift_events,
            p95_ms,
            first_token_p95_ms,
            target_p95_ms: profile.target_p95_ms,
            target_first_token_ms: profile.target_first_token_ms,
            learned_hint_count,
            active_negative_hint,
            recommended_budget_mode,
            status: status.to_string(),
            recommendations,
        });
    }

    summaries.sort_by(|left, right| left.route_profile.cmp(&right.route_profile));
    send_ipc_response(
        stream,
        &IpcResponse {
            status: "success".to_string(),
            data: serde_json::json!({ "routes": summaries }),
        },
    )
    .await;
    HandlerOutcome::DirectResponse
}
