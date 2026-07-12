//! Cloud cost-safety gate — Capa 3 of the 2026-06-09 OpenRouter billing
//! incident fix (see memory `project_openrouter_billing_incident`).
//!
//! Capa 1/2 (commits 01f1159/49e3455) made cloud fallback default-DENY and
//! gated the B3 quality cascade behind that same switch. What was still
//! missing — the actual "freno" — is enforced here, in two independent,
//! default-safe gates that sit at the router's single cloud-call chokepoint
//! (both `generate_content` and `generate_stream`, which is also where the
//! B3 escalation in `handler_generate.rs` routes through, since it reuses
//! the same `RouterEngine`):
//!
//! 1. **Rate limit** — caps cloud calls per rolling window. This is what
//!    would have capped a runaway B3-escalation loop or a `spawn_parallel_agents`
//!    fan-out that all decide the local answer looks "low quality" at once.
//! 2. **Daily token budget** — a rough proxy for $ spend (Hera has no
//!    real-time per-model pricing table), capped per rolling 24h.
//!
//! A third gate, `enforce_free_tier_model`, is defined below but is invoked
//! from `openai_compat.rs` right where the outbound model is FINALIZED
//! (after `HERA_CLOUD_DEFAULT_MODEL`/`OPENROUTER_DEFAULT_MODEL` overrides
//! are applied) — that's the only place guaranteed to see the literal
//! string about to hit the wire, regardless of how routing got there.
//!
//! All limits are env-configurable with sane defaults; absence of env vars
//! means the DEFAULT (conservative) cap applies — never "no cap".

use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::ai::ChatRequest;

struct RateWindow {
    window_start: Instant,
    calls: u32,
}

fn rate_window() -> &'static Mutex<RateWindow> {
    static WINDOW: OnceLock<Mutex<RateWindow>> = OnceLock::new();
    WINDOW.get_or_init(|| {
        Mutex::new(RateWindow {
            window_start: Instant::now(),
            calls: 0,
        })
    })
}

struct TokenBudget {
    day_start: Instant,
    tokens: u64,
}

fn token_budget() -> &'static Mutex<TokenBudget> {
    static BUDGET: OnceLock<Mutex<TokenBudget>> = OnceLock::new();
    BUDGET.get_or_init(|| {
        Mutex::new(TokenBudget {
            day_start: Instant::now(),
            tokens: 0,
        })
    })
}

fn rate_window_secs() -> Duration {
    Duration::from_secs(
        std::env::var("HERA_CLOUD_RATE_WINDOW_SECS")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(3600),
    )
}

fn rate_max_calls() -> u32 {
    std::env::var("HERA_CLOUD_MAX_CALLS_PER_WINDOW")
        .ok()
        .and_then(|value| value.trim().parse::<u32>().ok())
        .unwrap_or(20)
}

fn daily_token_budget() -> u64 {
    std::env::var("HERA_CLOUD_MAX_TOKENS_PER_DAY")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(200_000)
}

/// Cheap chars/4 estimate — same heuristic already used in
/// `ipc/handler_generate.rs` for usage-log fallback when a real `usage`
/// object isn't available yet (pre-flight, so it never is here).
pub fn estimate_tokens(req: &ChatRequest) -> u64 {
    let chars: usize = req
        .messages
        .iter()
        .map(|message| match &message.content {
            crate::ai::MessageContent::Text(text) => text.len(),
            crate::ai::MessageContent::Parts(parts) => parts
                .iter()
                .map(|part| match part {
                    crate::ai::ContentPart::Text { text } => text.len(),
                    _ => 0,
                })
                .sum(),
            crate::ai::MessageContent::Null => 0,
        })
        .sum();
    (chars / 4) as u64 + req.max_tokens.unwrap_or(0) as u64
}

/// Call once, right before actually dispatching a cloud request. Returns
/// `Err` with a human-readable reason if the call would exceed the rate
/// limit or the daily token budget. The caller must treat that exactly like
/// a cloud-disallowed / cloud-failed path — fail closed, never charge.
pub fn check_and_record_cloud_call(estimated_tokens: u64) -> Result<(), String> {
    {
        let mut window = rate_window()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if window.window_start.elapsed() >= rate_window_secs() {
            window.window_start = Instant::now();
            window.calls = 0;
        }
        if window.calls >= rate_max_calls() {
            return Err(format!(
                "Cloud rate limit hit ({} calls per {:?}); denying further cloud calls until the window resets (cost-safety gate, HERA_CLOUD_MAX_CALLS_PER_WINDOW).",
                rate_max_calls(),
                rate_window_secs()
            ));
        }
        window.calls += 1;
    }

    {
        let mut budget = token_budget()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if budget.day_start.elapsed() >= Duration::from_secs(86_400) {
            budget.day_start = Instant::now();
            budget.tokens = 0;
        }
        if budget.tokens >= daily_token_budget() {
            return Err(format!(
                "Cloud daily token budget exhausted ({} tokens); denying further cloud calls until the day resets (cost-safety gate, HERA_CLOUD_MAX_TOKENS_PER_DAY).",
                daily_token_budget()
            ));
        }
        budget.tokens += estimated_tokens;
    }

    Ok(())
}

/// Rejects non-free-tier OpenRouter models unless explicitly overridden.
///
/// This is "enforcement", not "pin by env": Capa 1/2 already default the
/// *code's own fallback* to a `:free` model, but nothing stopped an operator
/// (or a copy-pasted `.env`) from setting `HERA_CLOUD_DEFAULT_MODEL` /
/// `OPENROUTER_DEFAULT_MODEL` to a paid model — which is exactly the
/// 2026-06-09 incident (`google/gemini-2.5-pro` via a linked OpenRouter
/// card). This check runs regardless of where the model string came from.
///
/// Scoped to OpenRouter specifically: Groq / Google AI Studio direct /
/// Cerebras are free-tier by ACCOUNT (no card on file), so their model ids
/// don't use the `:free` naming convention — gating them the same way would
/// break the legitimate no-card backup chain documented in
/// `project_openrouter_billing_incident`.
pub fn enforce_free_tier_model(endpoint: &str, model: &str) -> Result<(), String> {
    let allow_paid = std::env::var("HERA_ALLOW_PAID_CLOUD_MODELS")
        .ok()
        .is_some_and(|value| {
            matches!(value.trim(), "1" | "true" | "TRUE" | "True" | "yes" | "YES")
        });
    if allow_paid {
        return Ok(());
    }
    if endpoint.contains("openrouter.ai") && !model.trim().ends_with(":free") {
        return Err(format!(
            "Cloud model '{model}' rejected: OpenRouter calls must target a ':free' model unless HERA_ALLOW_PAID_CLOUD_MODELS is explicitly set (cost-safety gate, see 2026-06-09 OpenRouter billing incident)."
        ));
    }
    Ok(())
}
