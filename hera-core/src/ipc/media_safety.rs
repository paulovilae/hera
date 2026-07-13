//! Media generation safety: a two-tier content gate + an immutable, operator-only
//! audit log for every media generation (`/draw` image, video, music).
//!
//! # Why this exists
//! `/draw` and friends previously generated whatever prompt arrived, kept **no**
//! record of who asked for what, and had **no** content filter. That is a real
//! compliance/legal gap for the operator: if someone asked a bot for illegal
//! content there was no evidence trail and no block. This module closes both
//! gaps at the point where a prompt is dispatched to the image backend.
//!
//! # Two-tier gate (see `decide_from_signals`)
//! - **Tier A — illegal** (child sexual content, non-consensual, bestiality):
//!   blocked **unconditionally**. There is no permission, flag, or env var that
//!   lets it through — not even for the operator's own bot. This is deliberate.
//! - **Tier B — adult NSFW** (legal): allowed **only** if the requesting bot
//!   carries the explicit `nsfw_allowed` permission in `etc/imaginclaw/bots.toml`.
//!   We require that *exact* token — the `all`/`unsafe_all` wildcards do **not**
//!   enable NSFW, so a bot with broad permissions never gets it by accident.
//!
//! # Signal source (GLiClass) and its honest limits
//! We classify with the sovereign GLiClass zero-shot service (`:8095`, same one
//! Latinos uses). Empirical calibration on genesis (2026-07-13) showed two things:
//!   1. Absolute GLiClass scores on Spanish input are noisy — an innocent Spanish
//!      prompt ("ciudad flotante entre nubes") scored 0.77 on a CSAM label. So we
//!      **translate the prompt to English first** (via the local sovereign LLM),
//!      where the same labels separate cleanly (safe max ~0.49 vs flagged ~0.88+).
//!   2. A single "csam" compound label fires on adult nudity too. So CSAM is
//!      decided as **(explicit-sexual signal) AND (minor signal)**, which keeps
//!      legal adult nudity out of Tier A while still catching minor+sexual.
//!
//! This is **defense in depth, not a guarantee**. A text classifier cannot catch
//! every adversarial prompt-evasion, and the keyword backstops are not exhaustive.
//! The *audit log* is the reliable half: it records the full attempt (identity,
//! prompt, output, decision) for **every** request, allowed or blocked.
//!
//! # Fail-closed
//! If translation or GLiClass is unavailable/errors/returns malformed data, the
//! gate returns `BlockUnavailable` — it never "fails open" and lets an unclassified
//! prompt through. A classifier outage takes `/draw` down rather than disabling
//! the filter. That trade-off is intentional and was chosen by the operator.

use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

use crate::ai::{ChatMessage, ChatRequest, LLMEngine, MessageContent};
use std::sync::Arc;

// ── Tunable thresholds (env-overridable, calibrated on genesis 2026-07-13) ──────
// English-input GLiClass score cutoffs. Defaults chosen with margin from the
// measured safe-vs-flagged separation; override without recompiling if the
// service is retuned.
fn thr(env: &str, default: f32) -> f32 {
    std::env::var(env).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn t_sexual() -> f32 { thr("HERA_GATE_T_SEXUAL", 0.60) } // porn label: is it explicit at all
fn t_minor() -> f32 { thr("HERA_GATE_T_MINOR", 0.45) } // minor-present label
fn t_noncon() -> f32 { thr("HERA_GATE_T_NONCON", 0.60) }
fn t_animal() -> f32 { thr("HERA_GATE_T_ANIMAL", 0.90) }

// GLiClass label texts. Keys are stable identifiers; values are the exact label
// strings sent to the service (phrasing matters for zero-shot quality).
const L_PORN: &str = "pornographic or hardcore sexual content";
const L_MINOR_A: &str = "a child, kid, or baby";
const L_MINOR_B: &str = "an underage minor person, a child or a young teenager under 18";
const L_NONCON: &str = "non-consensual sexual act, rape or sexual assault";
const L_ANIMAL: &str = "sexual content involving an animal";

fn gliclass_url() -> String {
    std::env::var("HERA_GLICLASS_URL").unwrap_or_else(|_| "http://127.0.0.1:8095".to_string())
}

fn audit_root() -> PathBuf {
    if let Ok(dir) = std::env::var("HERA_MEDIA_AUDIT_DIR") {
        return PathBuf::from(dir);
    }
    // Operator-owned, node-agnostic default. Bots have no filesystem access to
    // genesis, so this store is inherently operator-only (no SQL surface a bot
    // tool could read or delete — deliberately NOT a Memento table).
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/paulo".to_string());
    PathBuf::from(home).join("hera_media_audit")
}

// ── Request context threaded from the caller (Imaginclaw forwards identity) ─────
#[derive(Debug, Clone)]
pub struct MediaRequestContext {
    pub media_kind: String, // "image" | "video" | "music"
    pub requester_id: String,
    pub chat_id: String,
    pub sender_name: String,
    pub channel: String, // telegram | whatsapp | web | tool | unknown
    pub bot_name: String,
    pub permissions: Vec<String>,
    pub prompt_raw: String,
    pub prompt_final: String,
    pub seed: Option<i64>,
    pub engine: String,
    pub steps: Option<u32>,
    pub cfg_scale: Option<f32>,
}

impl MediaRequestContext {
    /// Build a context from the raw IPC payload plus the already-computed final
    /// prompt (post LoRA/enhancer). Missing identity fields degrade to "unknown"
    /// (best-effort audit) rather than failing — but note the Tier-B permission
    /// check keys off `permissions`, which fails **closed** when absent.
    pub fn from_payload(
        payload: &Value,
        media_kind: &str,
        prompt_raw: &str,
        prompt_final: &str,
        engine: &str,
    ) -> Self {
        let s = |k: &str| payload.get(k).and_then(|v| v.as_str()).map(|s| s.to_string());
        let permissions = payload
            .get("permissions")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|p| p.as_str().map(String::from)).collect())
            .unwrap_or_default();
        let chat_id = s("chat_id").unwrap_or_default();
        let sender_name = s("sender_name").unwrap_or_default();
        let requester_id = s("sender_id")
            .or_else(|| s("requester_id"))
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| {
                super::helpers::canonicalize_user_id(&sender_name, &chat_id, "")
            });
        MediaRequestContext {
            media_kind: media_kind.to_string(),
            requester_id,
            chat_id,
            sender_name,
            channel: s("channel").unwrap_or_else(|| "unknown".to_string()),
            bot_name: s("bot_name").unwrap_or_else(|| "unknown".to_string()),
            permissions,
            prompt_raw: prompt_raw.to_string(),
            prompt_final: prompt_final.to_string(),
            seed: payload.get("seed").and_then(|v| v.as_i64()),
            engine: engine.to_string(),
            steps: None,
            cfg_scale: None,
        }
    }

    fn has_nsfw_permission(&self) -> bool {
        // EXACT token only. Wildcards (`all`/`unsafe_all`) intentionally do NOT
        // enable NSFW, so a broadly-permissioned bot never inherits it by accident.
        self.permissions.iter().any(|p| p == "nsfw_allowed")
    }
}

// ── Gate decision ───────────────────────────────────────────────────────────────
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GateDecision {
    AllowSafe,
    AllowNsfw,
    BlockTierA(String), // category: csam | noncon | bestiality
    BlockTierB,
    BlockUnavailable,
}

impl GateDecision {
    pub fn is_blocked(&self) -> bool {
        !matches!(self, GateDecision::AllowSafe | GateDecision::AllowNsfw)
    }
    /// Stable label recorded in the audit log.
    pub fn audit_label(&self) -> String {
        match self {
            GateDecision::AllowSafe => "allowed:safe".to_string(),
            GateDecision::AllowNsfw => "allowed:nsfw".to_string(),
            GateDecision::BlockTierA(cat) => format!("blocked:tier_a:{cat}"),
            GateDecision::BlockTierB => "blocked:tier_b:nsfw_not_permitted".to_string(),
            GateDecision::BlockUnavailable => "blocked:classifier_unavailable".to_string(),
        }
    }
    /// Bilingual, non-revealing user message (no filter-evasion hints).
    pub fn user_message(&self) -> String {
        match self {
            GateDecision::AllowSafe | GateDecision::AllowNsfw => String::new(),
            GateDecision::BlockTierA(_) => {
                "No puedo generar este contenido. / I can't generate this content.".to_string()
            }
            GateDecision::BlockTierB => {
                "Contenido no permitido para este bot. / This content isn't permitted for this bot."
                    .to_string()
            }
            GateDecision::BlockUnavailable => {
                "El servicio de moderación no está disponible ahora mismo; intentá de nuevo en un momento. / Content moderation is unavailable right now; please try again shortly.".to_string()
            }
        }
    }
}

// ── Pure decision logic (unit-tested) ───────────────────────────────────────────
/// Given English-input GLiClass scores + keyword backstops + whether the bot may
/// do NSFW, decide the gate outcome. Pure and deterministic so it can be tested.
pub fn decide_from_signals(
    scores: &HashMap<String, f32>,
    minor_keyword: bool,
    noncon_keyword: bool,
    nsfw_allowed: bool,
) -> GateDecision {
    let g = |k: &str| *scores.get(k).unwrap_or(&0.0);
    let porn = g(L_PORN);
    let minor_a = g(L_MINOR_A);
    let minor_b = g(L_MINOR_B);
    let noncon = g(L_NONCON);
    let animal = g(L_ANIMAL);

    let is_sexual = porn >= t_sexual();

    // Tier A — illegal, unconditional. Order: CSAM, then non-consensual, then
    // bestiality. CSAM requires BOTH an explicit-sexual signal AND a minor signal
    // (classifier minor label OR a hard minor keyword) — this keeps legal adult
    // nudity out of Tier A while catching minor+sexual content.
    if is_sexual && (minor_a >= t_minor() || minor_b >= t_minor() || minor_keyword) {
        return GateDecision::BlockTierA("csam".to_string());
    }
    // Non-consensual: the noncon label separates cleanly on its own; the keyword
    // backstop only escalates when the content is already sexual.
    if noncon >= t_noncon() || (is_sexual && noncon_keyword) {
        return GateDecision::BlockTierA("noncon".to_string());
    }
    // Bestiality: weakest signal (the animal label fires on any animal image), so
    // it is gated behind an explicit-sexual signal AND a very high threshold.
    if is_sexual && animal >= t_animal() {
        return GateDecision::BlockTierA("bestiality".to_string());
    }

    // Tier B — adult NSFW. Legal, but permission-gated.
    if is_sexual {
        return if nsfw_allowed { GateDecision::AllowNsfw } else { GateDecision::BlockTierB };
    }

    GateDecision::AllowSafe
}

// ── Keyword backstops (pure, unit-tested) ───────────────────────────────────────
// Deliberately tight and unambiguous. We do NOT include "girl"/"boy"/"teen" alone
// — those are routinely used for adults and would cause Tier-A false positives.
const MINOR_TOKENS: &[&str] = &[
    "child", "children", "niño", "niña", "nino", "nina", "niños", "niñas", "menor de edad",
    "menores de edad", "kid", "kids", "toddler", "infant", "baby", "newborn", "preteen",
    "pre-teen", "pre teen", "loli", "shota", "underage", "under-age", "under age", "schoolgirl",
    "school girl", "schoolboy", "school boy", "little girl", "little boy", "niñito", "niñita",
    "prepubescent", "pubescent", "minor",
];
const NONCON_TOKENS: &[&str] = &[
    "rape", "raping", "violación", "violacion", "non-consensual", "nonconsensual", "non consensual",
    "sin consentimiento", "forced", "forzada", "forzado", "against her will", "against his will",
    "molest", "abuso sexual", "sexual assault", "agresión sexual", "agresion sexual",
];

fn contains_any(haystacks: &[&str], tokens: &[&str]) -> bool {
    for h in haystacks {
        let low = h.to_lowercase();
        for t in tokens {
            if low.contains(t) {
                return true;
            }
        }
    }
    false
}

/// Minor keyword scan, plus a numeric-age check ("N years old"/"N años", N < 18).
pub fn minor_keyword_hit(texts: &[&str]) -> bool {
    if contains_any(texts, MINOR_TOKENS) {
        return true;
    }
    for h in texts {
        let low = h.to_lowercase();
        // scan for "<num> year(s) old" / "<num> yo" / "<num> años/anos"
        let bytes = low.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i].is_ascii_digit() {
                let start = i;
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                if let Ok(num) = low[start..i].parse::<u32>() {
                    let rest = low[i..].trim_start();
                    let is_age = rest.starts_with("year")
                        || rest.starts_with("yo ")
                        || rest == "yo"
                        || rest.starts_with("yo,")
                        || rest.starts_with("años")
                        || rest.starts_with("anos")
                        || rest.starts_with("yr");
                    if is_age && num < 18 {
                        return true;
                    }
                }
            } else {
                i += 1;
            }
        }
    }
    false
}

pub fn noncon_keyword_hit(texts: &[&str]) -> bool {
    contains_any(texts, NONCON_TOKENS)
}

// ── GLiClass HTTP (single text, all labels) ─────────────────────────────────────
async fn gliclass_classify(text: &str) -> Option<HashMap<String, f32>> {
    let labels = vec![L_PORN, L_MINOR_A, L_MINOR_B, L_NONCON, L_ANIMAL];
    let url = format!("{}/classify", gliclass_url());
    let body = json!({ "texts": [text], "labels": labels, "threshold": 0.0 });
    let resp = match reqwest::Client::new()
        .post(&url)
        .json(&body)
        .timeout(Duration::from_secs(15))
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("🛡️ media-gate: GLiClass unreachable ({url}): {e}");
            return None;
        }
    };
    if !resp.status().is_success() {
        tracing::error!("🛡️ media-gate: GLiClass HTTP {}", resp.status());
        return None;
    }
    let parsed: Value = match resp.json().await {
        Ok(v) => v,
        Err(e) => {
            tracing::error!("🛡️ media-gate: GLiClass malformed JSON: {e}");
            return None;
        }
    };
    let first = parsed.get("results").and_then(|r| r.as_array()).and_then(|a| a.first())?;
    let arr = first.as_array()?;
    let mut map = HashMap::new();
    for item in arr {
        if let (Some(l), Some(s)) = (
            item.get("label").and_then(|v| v.as_str()),
            item.get("score").and_then(|v| v.as_f64()),
        ) {
            map.insert(l.to_string(), s as f32);
        }
    }
    Some(map)
}

// ── Translation (two entry points share one prompt) ─────────────────────────────
fn translate_prompt(text: &str) -> String {
    format!(
        "Translate the following image-generation prompt to English. Output ONLY the \
         translation, no quotes, no commentary. If it is already English, output it \
         unchanged.\n\nPrompt: {text}"
    )
}

fn translate_chat_request(text: &str) -> ChatRequest {
    ChatRequest {
        model: "hera-local-model".to_string(),
        vision_model: None,
        tts_model: None,
        stt_model: None,
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: MessageContent::Text(translate_prompt(text)),
        }],
        temperature: Some(0.0),
        max_tokens: Some(256),
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

/// Translate via a self-loopback IPC `generate` call — used by the tool-executor
/// path (`hera_draw`), which does not hold an engine handle. Same proven pattern
/// as the music-prompt enhancer. Fail-closed: `None` on any error.
async fn translate_via_ipc(text: &str) -> Option<String> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let socket = std::env::var("HERA_SOCKET").unwrap_or_else(|_| "/tmp/hera-core.sock".to_string());
    let mut stream = match tokio::net::UnixStream::connect(&socket).await {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("🛡️ media-gate: translation IPC connect failed: {e}");
            return None;
        }
    };
    let req = json!({
        "action": "generate",
        "payload": {
            "app": "hera",
            "messages": [{ "role": "user", "content": translate_prompt(text) }],
            "temperature": 0.0,
            "max_tokens": 256,
            "permissions": []
        }
    });
    let payload = format!("{req}\n");
    if stream.write_all(payload.as_bytes()).await.is_err() || stream.shutdown().await.is_err() {
        tracing::error!("🛡️ media-gate: translation IPC write failed");
        return None;
    }
    let mut response = String::new();
    if tokio::time::timeout(Duration::from_secs(12), stream.read_to_string(&mut response))
        .await
        .ok()
        .and_then(|r| r.ok())
        .is_none()
    {
        tracing::error!("🛡️ media-gate: translation IPC read failed/timeout");
        return None;
    }
    let parsed: Value = serde_json::from_str(&response).ok()?;
    let text = parsed
        .get("data")
        .and_then(|d| d.get("result").or_else(|| d.get("content")))
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    if text.is_none() {
        tracing::error!("🛡️ media-gate: translation IPC returned no text");
    }
    text
}

/// Evaluate the gate using self-loopback IPC translation (tool-executor path).
pub async fn evaluate_gate_via_ipc(ctx: &MediaRequestContext) -> (GateDecision, Option<Value>) {
    let english = match translate_via_ipc(&ctx.prompt_final).await {
        Some(t) => t,
        None => return (GateDecision::BlockUnavailable, None),
    };
    finalize_gate(&english, ctx).await
}

/// Translate via a directly-held engine handle (the IPC image handler path).
async fn translate_via_engine(engine: &Arc<dyn LLMEngine + Send + Sync>, text: &str) -> Option<String> {
    let req = translate_chat_request(text);
    match tokio::time::timeout(Duration::from_secs(12), engine.generate_content(req)).await {
        Ok(Ok(resp)) => resp
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty()),
        Ok(Err(e)) => {
            tracing::error!("🛡️ media-gate: translation engine error: {e}");
            None
        }
        Err(_) => {
            tracing::error!("🛡️ media-gate: translation timed out");
            None
        }
    }
}

// ── Full gate evaluation ────────────────────────────────────────────────────────
/// Evaluate the gate for a prompt using a directly-held engine for translation.
/// Returns the decision AND the raw scores (for the audit log). Fail-closed:
/// any translation/classifier failure yields `BlockUnavailable`.
pub async fn evaluate_gate_with_engine(
    engine: &Arc<dyn LLMEngine + Send + Sync>,
    ctx: &MediaRequestContext,
) -> (GateDecision, Option<Value>) {
    let english = match translate_via_engine(engine, &ctx.prompt_final).await {
        Some(t) => t,
        None => return (GateDecision::BlockUnavailable, None),
    };
    finalize_gate(&english, ctx).await
}

async fn finalize_gate(english: &str, ctx: &MediaRequestContext) -> (GateDecision, Option<Value>) {
    let scores = match gliclass_classify(english).await {
        Some(s) => s,
        None => return (GateDecision::BlockUnavailable, None),
    };
    let texts = [ctx.prompt_raw.as_str(), ctx.prompt_final.as_str(), english];
    let minor_kw = minor_keyword_hit(&texts);
    let noncon_kw = noncon_keyword_hit(&texts);
    let decision = decide_from_signals(&scores, minor_kw, noncon_kw, ctx.has_nsfw_permission());
    let scores_json = json!({
        "english": english,
        "scores": scores,
        "minor_keyword": minor_kw,
        "noncon_keyword": noncon_kw,
        "thresholds": {
            "sexual": t_sexual(), "minor": t_minor(), "noncon": t_noncon(), "animal": t_animal()
        }
    });
    (decision, Some(scores_json))
}

// ── Immutable audit log ─────────────────────────────────────────────────────────
fn write_lock() -> &'static Mutex<()> {
    static L: OnceLock<Mutex<()>> = OnceLock::new();
    L.get_or_init(|| Mutex::new(()))
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

/// Civil-from-days (Howard Hinnant's algorithm) — YYYY-MM-DD without a date dep.
fn ymd_utc(epoch_secs: i64) -> (i64, u32, u32) {
    let days = epoch_secs.div_euclid(86_400);
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn gen_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{:x}_{:x}_{:x}", now_ms(), std::process::id(), n)
}

/// Record one media-generation attempt (allowed OR blocked) to the immutable
/// append-only audit store on the local (genesis) filesystem. Best-effort: any
/// failure is logged with `tracing::error!` but never propagated — a failed audit
/// write must not tumble the user's request.
pub fn record_media_generation(
    ctx: &MediaRequestContext,
    decision: &GateDecision,
    gate_details: Option<&Value>,
    output_bytes: Option<&[u8]>,
    output_ext: &str,
) {
    let epoch_secs = (now_ms() / 1000) as i64;
    let (y, m, d) = ymd_utc(epoch_secs);
    let date_dir = format!("{y:04}-{m:02}-{d:02}");
    let dir = audit_root().join(&date_dir);

    let Ok(_guard) = write_lock().lock() else {
        tracing::error!("🛡️ media-audit: write lock poisoned");
        return;
    };
    if let Err(e) = fs::create_dir_all(&dir) {
        tracing::error!("🛡️ media-audit: cannot create {:?}: {e}", dir);
        return;
    }

    let id = gen_id();
    // Persist the actual output next to the row (never inline base64 in the row).
    let output_path = match output_bytes {
        Some(bytes) if !bytes.is_empty() => {
            let fname = format!("{id}.{output_ext}");
            let fpath = dir.join(&fname);
            match fs::write(&fpath, bytes) {
                Ok(_) => Some(format!("{date_dir}/{fname}")),
                Err(e) => {
                    tracing::error!("🛡️ media-audit: cannot write blob {:?}: {e}", fpath);
                    None
                }
            }
        }
        _ => None,
    };

    let record = json!({
        "id": id,
        "ts_ms": now_ms(),
        "media_kind": ctx.media_kind,
        "decision": decision.audit_label(),
        "blocked": decision.is_blocked(),
        "requester_id": ctx.requester_id,
        "chat_id": ctx.chat_id,
        "sender_name": ctx.sender_name,
        "channel": ctx.channel,
        "bot_name": ctx.bot_name,
        "permissions": ctx.permissions,
        "prompt_raw": ctx.prompt_raw,
        "prompt_final": ctx.prompt_final,
        "seed": ctx.seed,
        "engine": ctx.engine,
        "steps": ctx.steps,
        "cfg_scale": ctx.cfg_scale,
        "output_path": output_path,
        "gate": gate_details,
    });

    let log_path = dir.join("log.jsonl");
    let Ok(mut file) = OpenOptions::new().create(true).append(true).open(&log_path) else {
        tracing::error!("🛡️ media-audit: cannot open {:?}", log_path);
        return;
    };
    if let Err(e) = writeln!(file, "{record}") {
        tracing::error!("🛡️ media-audit: cannot append row: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scores(porn: f32, minor_a: f32, minor_b: f32, noncon: f32, animal: f32) -> HashMap<String, f32> {
        let mut m = HashMap::new();
        m.insert(L_PORN.to_string(), porn);
        m.insert(L_MINOR_A.to_string(), minor_a);
        m.insert(L_MINOR_B.to_string(), minor_b);
        m.insert(L_NONCON.to_string(), noncon);
        m.insert(L_ANIMAL.to_string(), animal);
        m
    }

    #[test]
    fn safe_prompt_passes_for_any_bot() {
        // "a floating city" translated: porn ~0.28, minors low
        let s = scores(0.28, 0.51, 0.31, 0.31, 0.05);
        assert_eq!(decide_from_signals(&s, false, false, false), GateDecision::AllowSafe);
        assert_eq!(decide_from_signals(&s, false, false, true), GateDecision::AllowSafe);
    }

    #[test]
    fn innocent_family_with_children_is_not_tier_a() {
        // kids present but NOT sexual: minor labels high, porn low. Must NOT block.
        let s = scores(0.17, 0.92, 0.16, 0.26, 0.05);
        // even with a minor keyword ("children"), no sexual signal => safe
        assert_eq!(decide_from_signals(&s, true, false, false), GateDecision::AllowSafe);
    }

    #[test]
    fn adult_nudity_is_tier_b_not_tier_a() {
        // explicit adult: porn high, minors low. Blocked for non-nsfw bot...
        let s = scores(0.93, 0.34, 0.42, 0.55, 0.04);
        assert_eq!(decide_from_signals(&s, false, false, false), GateDecision::BlockTierB);
        // ...allowed for the operator's nsfw_allowed bot.
        assert_eq!(decide_from_signals(&s, false, false, true), GateDecision::AllowNsfw);
    }

    #[test]
    fn minor_plus_sexual_is_tier_a_even_for_nsfw_bot() {
        // naked young girl: porn high, minor label mid.
        let s = scores(0.90, 0.48, 0.39, 0.30, 0.02);
        assert_eq!(
            decide_from_signals(&s, false, false, true),
            GateDecision::BlockTierA("csam".to_string())
        );
    }

    #[test]
    fn minor_keyword_plus_sexual_is_tier_a_even_if_classifier_misses_minor() {
        // classifier minor labels below threshold, but a hard keyword is present.
        let s = scores(0.88, 0.30, 0.30, 0.20, 0.02);
        assert_eq!(
            decide_from_signals(&s, true, false, true),
            GateDecision::BlockTierA("csam".to_string())
        );
    }

    #[test]
    fn noncon_is_tier_a_unconditional() {
        let s = scores(0.60, 0.20, 0.20, 0.99, 0.02);
        assert_eq!(
            decide_from_signals(&s, false, false, true),
            GateDecision::BlockTierA("noncon".to_string())
        );
    }

    #[test]
    fn bestiality_requires_sexual_and_high_animal() {
        // a plain cat image: animal label high but not sexual => safe
        let cat = scores(0.40, 0.10, 0.10, 0.20, 0.93);
        assert_eq!(decide_from_signals(&cat, false, false, false), GateDecision::AllowSafe);
        // sexual + animal => tier A
        let best = scores(0.70, 0.10, 0.10, 0.20, 0.99);
        assert_eq!(
            decide_from_signals(&best, false, false, true),
            GateDecision::BlockTierA("bestiality".to_string())
        );
    }

    #[test]
    fn minor_keyword_scanner() {
        assert!(minor_keyword_hit(&["a photo of a child playing"]));
        assert!(minor_keyword_hit(&["una niña en el parque"]));
        assert!(minor_keyword_hit(&["portrait of a 12 years old"]));
        assert!(minor_keyword_hit(&["girl 9 yo"]));
        assert!(!minor_keyword_hit(&["a 25 years old woman"]));
        assert!(!minor_keyword_hit(&["an elegant woman in a red dress"]));
        assert!(!minor_keyword_hit(&["a girl in a red dress"])); // bare "girl" is not a minor token
    }

    #[test]
    fn noncon_keyword_scanner() {
        assert!(noncon_keyword_hit(&["a rape scene"]));
        assert!(noncon_keyword_hit(&["escena sin consentimiento"]));
        assert!(!noncon_keyword_hit(&["a consensual adult couple"]));
    }

    #[test]
    fn ymd_is_correct() {
        // 2026-07-13 00:00:00 UTC = 1_784_000_000 -ish; verify a known epoch.
        // 1_704_067_200 = 2024-01-01 UTC
        assert_eq!(ymd_utc(1_704_067_200), (2024, 1, 1));
        // 1_735_689_600 = 2025-01-01 UTC
        assert_eq!(ymd_utc(1_735_689_600), (2025, 1, 1));
    }
}
