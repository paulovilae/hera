//! Query difficulty classification for sovereign effort routing (frame B1+B3).
//!
//! Maps an incoming user prompt to Trivial / Normal / Hard so the local engine
//! can scale its effort (context budget, reasoning_effort, think directive, and
//! preferred local model tier) WITHOUT defaulting to cloud — cloud stays a
//! failover only, per the platform's sovereign-first rule.
//!
//! Hybrid classifier: a cheap deterministic heuristic decides the clear cases;
//! gray-zone prompts are broken by cosine similarity to a small set of labeled
//! exemplars using Hera's local embedder (frame A). On CPU-only builds the
//! embedder is unavailable, so the gray zone falls back to the heuristic score.

use std::sync::OnceLock;

use super::context::is_lightweight_conversation;
use super::helpers::embed_text_local;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Difficulty {
    Trivial,
    Normal,
    Hard,
}

impl Difficulty {
    pub fn as_str(self) -> &'static str {
        match self {
            Difficulty::Trivial => "trivial",
            Difficulty::Normal => "normal",
            Difficulty::Hard => "hard",
        }
    }

    /// Context-budget mode this difficulty maps to. Trivial is handled upstream
    /// by the lightweight path; Normal keeps the route-profile default; Hard
    /// pushes to the heavy budget.
    pub fn budget_mode(self) -> Option<&'static str> {
        match self {
            Difficulty::Hard => Some("heavy"),
            _ => None,
        }
    }

    /// Recover the difficulty from a reasoning_effort string (so handlers can
    /// avoid reclassifying a prompt they already classified upstream).
    pub fn from_reasoning_effort(effort: &str) -> Self {
        match effort {
            "high" => Difficulty::Hard,
            "low" => Difficulty::Trivial,
            _ => Difficulty::Normal,
        }
    }

    /// reasoning_effort hint for the model.
    pub fn reasoning_effort(self) -> &'static str {
        match self {
            Difficulty::Trivial => "low",
            Difficulty::Normal => "medium",
            Difficulty::Hard => "high",
        }
    }
}

// Hard-signal keyword groups (Spanish + English). Presence of any keyword in a
// group contributes one point for that group.
const CODE_KW: &[&str] = &[
    "```", "código", "code", "function", "función", "def ", "fn ", "class ", "sql",
    "query", "script", "implementa", "implement", "debug", "stack trace", "compile",
    "refactor", "regex", "endpoint", "api ",
];
const MATH_KW: &[&str] = &[
    "calcula", "calculate", "ecuación", "equation", "integral", "derivada", "derivative",
    "demuestra", "prove", "teorema", "theorem", "probabilidad", "optimiza", "optimize",
    "algoritmo", "algorithm", "complejidad", "big-o", "matriz", "estadística",
];
const REASON_KW: &[&str] = &[
    "analiza", "analyze", "compara", "compare", "evalúa", "evaluate", "por qué", "why",
    "paso a paso", "step by step", "diseña", "design", "estrategia", "strategy", "razona",
    "justifica", "justify", "pros y contras", "ventajas y desventajas", "trade-off",
    "tradeoff", "explica por qué", "explain why", "plan ", "arquitectura", "architecture",
];

/// Heuristic score: higher = harder.
fn heuristic_score(prompt: &str) -> i32 {
    let lower = prompt.to_lowercase();
    let mut score = 0;

    let len = prompt.chars().count();
    if len > 280 {
        score += 1;
    }
    if len > 600 {
        score += 1;
    }

    // One point per signal group present, plus a density bonus when several
    // hard-signal keywords co-occur (a request loaded with reasoning/code verbs
    // is harder than one with a single incidental keyword).
    let mut total_hits = 0usize;
    for group in [CODE_KW, MATH_KW, REASON_KW] {
        let group_count = group.iter().filter(|kw| lower.contains(**kw)).count();
        if group_count > 0 {
            score += 1;
        }
        total_hits += group_count;
    }
    if total_hits >= 3 {
        score += 1;
    }

    if lower.contains("```") {
        score += 2;
    }

    let question_marks = prompt.matches('?').count() + prompt.matches('¿').count();
    if question_marks >= 3 {
        score += 1;
    }

    score
}

const HARD_THRESHOLD: i32 = 2;
const EASY_CEILING: i32 = 0; // score 0 => Normal; score 1 => gray zone (embedding)

/// Labeled exemplars for the embedding tiebreaker (gray zone only).
const HARD_EXEMPLARS: &[&str] = &[
    "Diseña la arquitectura de un sistema distribuido tolerante a fallos y explica los trade-offs",
    "Demuestra paso a paso por qué este algoritmo tiene complejidad O(n log n)",
    "Analiza las ventajas y desventajas de migrar de microservicios a un monolito modular",
    "Implementa una función que resuelva este problema y explica tu razonamiento",
    "Compara tres estrategias de inversión y justifica cuál conviene para este perfil de riesgo",
];
const EASY_EXEMPLARS: &[&str] = &[
    "¿Qué hora es?",
    "Gracias por la ayuda",
    "Resume esto en una frase",
    "¿Cuál es la capital de Francia?",
    "Dame el saldo de mi cuenta",
];

struct Centroids {
    hard: Vec<f32>,
    easy: Vec<f32>,
}

static CENTROIDS: OnceLock<Option<Centroids>> = OnceLock::new();

fn mean_vec(vectors: &[Vec<f32>]) -> Option<Vec<f32>> {
    let first = vectors.first()?;
    let dim = first.len();
    if dim == 0 {
        return None;
    }
    let mut acc = vec![0.0f32; dim];
    for v in vectors {
        if v.len() != dim {
            return None;
        }
        for (i, x) in v.iter().enumerate() {
            acc[i] += x;
        }
    }
    let n = vectors.len() as f32;
    for x in acc.iter_mut() {
        *x /= n;
    }
    Some(acc)
}

fn embed_all(texts: &[&str]) -> Option<Vec<Vec<f32>>> {
    let mut out = Vec::with_capacity(texts.len());
    for t in texts {
        out.push(embed_text_local(t)?);
    }
    Some(out)
}

fn centroids() -> Option<&'static Centroids> {
    CENTROIDS
        .get_or_init(|| {
            let hard = embed_all(HARD_EXEMPLARS).and_then(|v| mean_vec(&v))?;
            let easy = embed_all(EASY_EXEMPLARS).and_then(|v| mean_vec(&v))?;
            Some(Centroids { hard, easy })
        })
        .as_ref()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Break a gray-zone case by embedding similarity. Returns Hard if the prompt is
/// closer to the hard centroid, else Normal. Falls back to Normal when the
/// embedder is unavailable (CPU-only build) so behavior stays defined.
fn embedding_tiebreak(prompt: &str) -> Difficulty {
    let Some(c) = centroids() else {
        return Difficulty::Normal;
    };
    let Some(v) = embed_text_local(prompt) else {
        return Difficulty::Normal;
    };
    if cosine(&v, &c.hard) > cosine(&v, &c.easy) {
        Difficulty::Hard
    } else {
        Difficulty::Normal
    }
}

/// Classify a prompt's difficulty. Trivial is decided by the existing lightweight
/// whitelist; otherwise the heuristic decides clear cases and the embedder breaks
/// the gray zone.
pub fn classify(prompt: &str) -> Difficulty {
    if is_lightweight_conversation(prompt) {
        return Difficulty::Trivial;
    }
    let score = heuristic_score(prompt);
    if score >= HARD_THRESHOLD {
        return Difficulty::Hard;
    }
    if score <= EASY_CEILING {
        return Difficulty::Normal;
    }
    // Gray zone (score == 2): let the embedder decide.
    embedding_tiebreak(prompt)
}

// Phrases that signal the local model failed to actually answer.
const INCAPACITY_MARKERS: &[&str] = &[
    "no sé", "no se ", "no puedo", "no tengo información", "no tengo informacion",
    "no tengo acceso", "no tengo suficiente", "lo siento, no", "no estoy seguro",
    "no dispongo", "i don't know", "i do not know", "i cannot", "i can't",
    "i'm unable", "i am unable", "as an ai", "no answer",
];

/// Frame B3: decide whether a local answer is poor enough to escalate to the
/// cloud failover. Conservative on purpose — only escalates for non-trivial
/// queries when the answer is empty, far too short, or explicitly disclaims.
pub fn is_low_quality_answer(answer: &str, difficulty: Difficulty) -> bool {
    if difficulty == Difficulty::Trivial {
        return false;
    }
    let trimmed = answer.trim();
    let len = trimmed.chars().count();
    if len == 0 {
        return true;
    }
    // A Hard query answered in a single short sentence is suspect; a Normal query
    // is allowed to be brief.
    let min_len = if difficulty == Difficulty::Hard { 40 } else { 12 };
    if len < min_len {
        return true;
    }
    let lower = trimmed.to_lowercase();
    INCAPACITY_MARKERS.iter().any(|m| lower.contains(m))
}

#[cfg(test)]
mod tests {
    use super::{classify, heuristic_score, is_low_quality_answer, Difficulty};

    #[test]
    fn low_quality_detects_disclaimer() {
        assert!(is_low_quality_answer(
            "Lo siento, no tengo información sobre eso.",
            Difficulty::Normal
        ));
    }

    #[test]
    fn low_quality_allows_good_answer() {
        assert!(!is_low_quality_answer(
            "El saldo de la cuenta 4082 es $1,234.50 al cierre de hoy.",
            Difficulty::Normal
        ));
    }

    #[test]
    fn trivial_never_escalates() {
        assert!(!is_low_quality_answer("ok", Difficulty::Trivial));
    }

    #[test]
    fn greetings_are_trivial() {
        assert_eq!(classify("hola"), Difficulty::Trivial);
        assert_eq!(classify("gracias"), Difficulty::Trivial);
    }

    #[test]
    fn code_request_is_hard() {
        let p = "Implementa una función en Rust que parsee este JSON y explica tu razonamiento paso a paso";
        assert_eq!(classify(p), Difficulty::Hard);
    }

    #[test]
    fn reasoning_request_is_hard() {
        let p = "Analiza las ventajas y desventajas de esta arquitectura y justifica tu recomendación con un plan";
        assert!(heuristic_score(p) >= 2);
        assert_eq!(classify(p), Difficulty::Hard);
    }

    #[test]
    fn short_factual_is_normal() {
        assert_eq!(classify("cual es el saldo de la cuenta 4082"), Difficulty::Normal);
    }
}
