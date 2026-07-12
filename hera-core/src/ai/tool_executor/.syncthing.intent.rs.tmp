//! Intent detection from user messages

use tracing::info;

use super::{ToolCall, registry::{load_canonical_app_registry, alias_terms_for_app, pm2_process_name_for_slug}};

/// Fallback intent detection from the USER's original message.
/// Works with any model size since it doesn't depend on tool_call emission.
/// Returns a ToolCall if the user's intent clearly maps to a tool.
pub fn detect_intent_from_user_message(
    user_msg: &str,
    _assistant_last: Option<&str>,
) -> Option<ToolCall> {
    // Strip injected app context from Imaginclaw (e.g., "[System Context: ...]")
    // so length checks and pattern matching work on the actual user text.
    let clean_msg = if let Some(ctx_start) = user_msg.find("\n\n[System Context:") {
        &user_msg[..ctx_start]
    } else {
        user_msg
    };
    let lower = clean_msg.to_lowercase();
    let lower_trimmed = lower.trim();

    // Strip common greeting prefixes so "hola genera una imagen" matches "genera una imagen"
    let greetings = [
        "hola ",
        "hola, ",
        "oye ",
        "oye, ",
        "hey ",
        "hey, ",
        "hi ",
        "hi, ",
        "hello ",
        "hello, ",
        "buenas ",
        "buenos días ",
        "buenas tardes ",
        "buenas noches ",
        "good morning ",
        "buen día ",
        "please ",
        "por favor ",
        "ey ",
        "ey, ",
        "yo ",
        "sup ",
        "ava ",
        "ava, ",
    ];
    let lower_no_greeting = {
        let mut s = lower_trimmed;
        for g in &greetings {
            if s.starts_with(g) {
                s = s[g.len()..].trim_start();
                break; // Only strip the first greeting
            }
        }
        s
    };

    let detect_app = || -> Option<&'static str> {
        let app = load_canonical_app_registry()
            .into_iter()
            .find_map(|entry| {
                let aliases = alias_terms_for_app(&entry);
                aliases
                    .iter()
                    .any(|alias| lower.contains(alias))
                    .then_some(entry.slug)
            })?;

        match app.as_str() {
            "latinos" => Some("latinos"),
            "vetra" => Some("vetra"),
            "movilo" => Some("movilo"),
            "os-v3" => Some("os-v3"),
            "desktop" => Some("desktop"),
            "paulo-vila-rust" => Some("paulo-vila-rust"),
            "capacita" => Some("capacita"),
            _ => None,
        }
    };

    let command = lower_no_greeting.trim();
    let service_targets = [
        ("imaginclaw", "imaginclaw"),
        ("ava", "imaginclaw"),
        ("vetra", "vetra"),
        ("cartera", "cartera"),
        ("movilo", "movilo"),
        ("sentinel", "sentinel"),
        ("hera", "hera"),
        ("whisper", "whisper"),
        ("audio stt", "audio-stt"),
        ("audio engine", "audio-engine"),
        ("garcero", "garcero"),
        ("latinos", "latinos"),
        ("paulovila", "paulo-vila-rust"),
        ("paulo vila", "paulo-vila-rust"),
        ("os-v3", "os-v3"),
        ("imaginos", "imaginos"),
        ("memento", "memento"),
        ("argus", "argus"),
    ];

    if let Some(rest) = command.strip_prefix("/draw ") {
        let prompt = rest.trim();
        if !prompt.is_empty() {
            info!("🎯 [Hera] Explicit fast-path command: /draw");
            return Some(ToolCall {
                name: "hera_draw".to_string(),
                arguments: serde_json::json!({"prompt": prompt}),
            });
        }
    }

    if let Some(rest) = command.strip_prefix("/search ") {
        let query = rest.trim();
        if !query.is_empty() {
            info!("🎯 [Hera] Explicit fast-path command: /search");
            return Some(ToolCall {
                name: "hera_search".to_string(),
                arguments: serde_json::json!({"query": query}),
            });
        }
    }

    if let Some(rest) = command.strip_prefix("/speak ") {
        let text = rest.trim();
        if !text.is_empty() {
            info!("🎯 [Hera] Explicit fast-path command: /speak");
            return Some(ToolCall {
                name: "hera_speak".to_string(),
                arguments: serde_json::json!({"text": text}),
            });
        }
    }

    if let Some(rest) = command.strip_prefix("/video ") {
        let prompt = rest.trim();
        if !prompt.is_empty() {
            info!("🎯 [Hera] Explicit fast-path command: /video");
            return Some(ToolCall {
                name: "hera_video".to_string(),
                arguments: serde_json::json!({"prompt": prompt}),
            });
        }
    }

    // "Estado OS" / "Estado del sistema" — full PM2 status with restart counts
    if matches!(
        command,
        "estado de bots"
            | "estado bots"
            | "bots"
            | "bot status"
            | "status de bots"
            | "estado robots"
            | "estado de robots"
    ) {
        info!("🎯 [Hera] Explicit fast-path command: list_bots");
        return Some(ToolCall {
            name: "list_bots".to_string(),
            arguments: serde_json::json!({}),
        });
    }

    if matches!(
        command,
        "ver mercados"
            | "mercados"
            | "listar mercados"
            | "lista de mercados"
            | "available markets"
            | "view markets"
            | "markets"
    ) {
        info!("🎯 [Hera] Explicit fast-path command: list_markets");
        return Some(ToolCall {
            name: "list_markets".to_string(),
            arguments: serde_json::json!({"limit": 25}),
        });
    }

    if matches!(
        command,
        "/status"
            | "/system-status"
            | "/server-status"
            | "/health-overview"
            | "/machine-status"
            | "estado os"
            | "estado del sistema"
            | "estado del servidor"
            | "system status"
            | "server status"
            | "status of all apps"
            | "review app status"
            | "review all apps status"
    ) {
        info!("🎯 [Hera] Explicit fast-path command: system_status");
        return Some(ToolCall {
            name: "system_status".to_string(),
            arguments: serde_json::json!({}),
        });
    }

    if matches!(command, "/diagnose" | "/diagnose-services") {
        info!("🎯 [Hera] Explicit fast-path command: /diagnose");
        return Some(ToolCall {
            name: "diagnose_services".to_string(),
            arguments: serde_json::json!({}),
        });
    }

    if matches!(command, "/review-apps" | "/apps-status") {
        info!("🎯 [Hera] Explicit fast-path command: /review-apps");
        return Some(ToolCall {
            name: "review_all_apps_status".to_string(),
            arguments: serde_json::json!({
                "timeout_seconds": 10
            }),
        });
    }

    if matches!(command, "/verify-stack" | "/stack-status") {
        info!("🎯 [Hera] Explicit fast-path command: /verify-stack");
        return Some(ToolCall {
            name: "verify_canonical_stack".to_string(),
            arguments: serde_json::json!({
                "checks": ["check"],
                "timeout_seconds": 60
            }),
        });
    }

    if let Some(rest) = command.strip_prefix("/verify-app ") {
        let requested = rest.trim();
        if !requested.is_empty() {
            let matched_app = detect_app().or_else(|| {
                service_targets
                    .iter()
                    .find(|(alias, _)| requested.contains(*alias))
                    .and_then(|(_, slug)| match *slug {
                        "vetra" => Some("vetra"),
                        "cartera" => Some("cartera"),
                        "movilo" => Some("movilo"),
                        "latinos" => Some("latinos"),
                        "os-v3" => Some("os-v3"),
                        "desktop" => Some("desktop"),
                        "paulo-vila-rust" => Some("paulo-vila-rust"),
                        "capacita" => Some("capacita"),
                        _ => None,
                    })
            });

            if let Some(app) = matched_app {
                info!(
                    "🎯 [Hera] Explicit fast-path command: /verify-app for '{}'",
                    app
                );
                return Some(ToolCall {
                    name: "verify_app_health".to_string(),
                    arguments: serde_json::json!({
                        "app": app,
                        "compile_checks": ["check"],
                        "runtime_suite": "regression",
                        "run_runtime": true,
                        "include_logs": true,
                        "timeout_seconds": 60
                    }),
                });
            }
        }
    }

    if let Some(rest) = command.strip_prefix("/restart ") {
        let requested = rest.trim();
        if let Some((alias, slug)) = service_targets
            .iter()
            .find(|(alias, _)| requested.contains(*alias))
        {
            let pm2_name = pm2_process_name_for_slug(slug);
            info!(
                "🎯 [Hera] Explicit fast-path command: /restart for '{}' via alias '{}'",
                pm2_name, alias
            );
            return Some(ToolCall {
                name: "service_restart".to_string(),
                arguments: serde_json::json!({"service_name": pm2_name}),
            });
        }
    }

    None
}
