//! Universal AI Feedback Module
//!
//! Provides a cross-app feedback mechanism where every AI interaction
//! can receive a like/dislike signal. This implicit feedback is logged
//! to Memento and used as Bayesian training signal.
//!
//! # Architecture
//! Any ImagineOS app (Movilo, Vetra, Imaginclaw, etc.) can:
//! 1. Record an AI response with `record_response()`
//! 2. Attach user feedback with `record_feedback()`
//! 3. The feedback is forwarded to Memento via UDS IPC as a
//!    `log_interaction` with the choice encoding the like/dislike.
//!
//! This creates a universal preference signal across the entire OS.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════
// Feedback Types
// ═══════════════════════════════════════════════════════════════════

/// User feedback signal on an AI interaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FeedbackSignal {
    /// User liked the response (thumbs up)
    Like,
    /// User disliked the response (thumbs down)
    Dislike,
    /// No feedback given (implicit — user moved on without reacting)
    NoFeedback,
}

impl FeedbackSignal {
    /// Convert to a numeric value for Bayesian processing.
    /// Like = 1.0, Dislike = 0.0, NoFeedback = 0.5 (neutral)
    pub fn to_score(&self) -> f64 {
        match self {
            FeedbackSignal::Like => 1.0,
            FeedbackSignal::Dislike => 0.0,
            FeedbackSignal::NoFeedback => 0.5,
        }
    }

    /// Convert to a choice index for Bayesian update.
    /// In a 2-option model: [dislike, like], Like = 1, Dislike = 0
    pub fn to_choice_index(&self) -> usize {
        match self {
            FeedbackSignal::Like => 1,
            FeedbackSignal::Dislike => 0,
            FeedbackSignal::NoFeedback => 1, // Treat no-feedback as mild positive
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// AI Interaction Record
// ═══════════════════════════════════════════════════════════════════

/// Captures a single AI interaction with optional feedback.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiInteraction {
    /// Unique ID for this interaction
    pub interaction_id: String,
    /// Source app (e.g., "movilo", "vetra", "imaginclaw", "hera")
    pub source_app: String,
    /// User identifier
    pub user_id: String,
    /// The domain/context this interaction belongs to
    pub domain: String,
    /// The AI model that generated the response
    pub model: String,
    /// The user's prompt/query (truncated for storage)
    pub prompt_summary: String,
    /// The AI's response (truncated for storage)
    pub response_summary: String,
    /// Features of this response (for Bayesian matching)
    /// e.g., {"confidence": 0.85, "length": 0.6, "specificity": 0.7}
    pub response_features: HashMap<String, f64>,
    /// User's feedback signal
    pub feedback: FeedbackSignal,
    /// ISO8601 timestamp
    pub timestamp: String,
}

impl AiInteraction {
    /// Create a new interaction record (feedback defaults to NoFeedback).
    pub fn new(
        source_app: &str,
        user_id: &str,
        domain: &str,
        model: &str,
        prompt: &str,
        response: &str,
    ) -> Self {
        let id = format!(
            "ai-{}-{}",
            source_app,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
        );

        Self {
            interaction_id: id,
            source_app: source_app.to_string(),
            user_id: user_id.to_string(),
            domain: domain.to_string(),
            model: model.to_string(),
            prompt_summary: truncate(prompt, 200),
            response_summary: truncate(response, 500),
            response_features: HashMap::new(),
            feedback: FeedbackSignal::NoFeedback,
            timestamp: epoch_timestamp(),
        }
    }

    /// Add response features for Bayesian matching.
    pub fn with_features(mut self, features: HashMap<String, f64>) -> Self {
        self.response_features = features;
        self
    }

    /// Attach user feedback.
    pub fn with_feedback(mut self, feedback: FeedbackSignal) -> Self {
        self.feedback = feedback;
        self
    }

    /// Build the Memento IPC payload for `log_interaction`.
    pub fn to_memento_payload(&self) -> serde_json::Value {
        let options = serde_json::json!([
            {"label": "dislike", "score": 0.0},
            {"label": "like", "score": 1.0}
        ]);
        serde_json::json!({
            "action": "log_interaction",
            "payload": {
                "session_id": self.interaction_id,
                "user_id": self.user_id,
                "domain": format!("{}:{}", self.source_app, self.domain),
                "round": 0,
                "options_json": options.to_string(),
                "choice_index": self.feedback.to_choice_index(),
                "prior_json": serde_json::to_string(&self.response_features).unwrap_or_default(),
            }
        })
    }
}

// ═══════════════════════════════════════════════════════════════════
// Feedback Aggregator
// ═══════════════════════════════════════════════════════════════════

/// Aggregates feedback statistics across interactions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackStats {
    pub total_interactions: usize,
    pub likes: usize,
    pub dislikes: usize,
    pub no_feedback: usize,
    pub by_app: HashMap<String, AppFeedbackStats>,
    pub by_model: HashMap<String, ModelFeedbackStats>,
}

/// Per-app feedback breakdown.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AppFeedbackStats {
    pub likes: usize,
    pub dislikes: usize,
    pub total: usize,
}

impl AppFeedbackStats {
    pub fn satisfaction_rate(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.likes as f64 / self.total as f64
    }
}

/// Per-model feedback breakdown.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelFeedbackStats {
    pub likes: usize,
    pub dislikes: usize,
    pub total: usize,
}

impl ModelFeedbackStats {
    pub fn satisfaction_rate(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.likes as f64 / self.total as f64
    }
}

impl FeedbackStats {
    /// Compute aggregate stats from a list of interactions.
    pub fn from_interactions(interactions: &[AiInteraction]) -> Self {
        let mut stats = Self {
            total_interactions: interactions.len(),
            likes: 0,
            dislikes: 0,
            no_feedback: 0,
            by_app: HashMap::new(),
            by_model: HashMap::new(),
        };

        for interaction in interactions {
            match interaction.feedback {
                FeedbackSignal::Like => stats.likes += 1,
                FeedbackSignal::Dislike => stats.dislikes += 1,
                FeedbackSignal::NoFeedback => stats.no_feedback += 1,
            }

            // Per-app stats
            let app_stats = stats
                .by_app
                .entry(interaction.source_app.clone())
                .or_default();
            app_stats.total += 1;
            match interaction.feedback {
                FeedbackSignal::Like => app_stats.likes += 1,
                FeedbackSignal::Dislike => app_stats.dislikes += 1,
                _ => {}
            }

            // Per-model stats
            let model_stats = stats.by_model.entry(interaction.model.clone()).or_default();
            model_stats.total += 1;
            match interaction.feedback {
                FeedbackSignal::Like => model_stats.likes += 1,
                FeedbackSignal::Dislike => model_stats.dislikes += 1,
                _ => {}
            }
        }

        stats
    }

    /// Overall satisfaction rate (likes / total with feedback).
    pub fn satisfaction_rate(&self) -> f64 {
        let with_feedback = self.likes + self.dislikes;
        if with_feedback == 0 {
            return 0.0;
        }
        self.likes as f64 / with_feedback as f64
    }
}

// ═══════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}…", &s[..max_len])
    }
}

fn epoch_timestamp() -> String {
    format!(
        "{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    )
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_interactions() -> Vec<AiInteraction> {
        vec![
            AiInteraction::new(
                "movilo",
                "user-1",
                "search",
                "gemini-2.5",
                "Find a dentist",
                "Dr. Garcia",
            )
            .with_feedback(FeedbackSignal::Like),
            AiInteraction::new(
                "movilo",
                "user-1",
                "search",
                "gemini-2.5",
                "Find a cardiologist",
                "Dr. Lopez",
            )
            .with_feedback(FeedbackSignal::Dislike),
            AiInteraction::new(
                "vetra",
                "user-2",
                "analysis",
                "qwen-2.5",
                "Analyze AAPL",
                "Buy signal",
            )
            .with_feedback(FeedbackSignal::Like),
            AiInteraction::new(
                "imaginclaw",
                "user-1",
                "chat",
                "gemini-2.5",
                "Hello",
                "Hi there!",
            )
            .with_feedback(FeedbackSignal::NoFeedback),
            AiInteraction::new(
                "vetra",
                "user-2",
                "analysis",
                "qwen-2.5",
                "Analyze MSFT",
                "Hold",
            )
            .with_feedback(FeedbackSignal::Like),
        ]
    }

    #[test]
    fn test_feedback_signal_score() {
        assert_eq!(FeedbackSignal::Like.to_score(), 1.0);
        assert_eq!(FeedbackSignal::Dislike.to_score(), 0.0);
        assert_eq!(FeedbackSignal::NoFeedback.to_score(), 0.5);
    }

    #[test]
    fn test_feedback_signal_choice_index() {
        assert_eq!(FeedbackSignal::Like.to_choice_index(), 1);
        assert_eq!(FeedbackSignal::Dislike.to_choice_index(), 0);
    }

    #[test]
    fn test_interaction_creation() {
        let interaction = AiInteraction::new(
            "movilo",
            "user-42",
            "search",
            "gemini-2.5",
            "Find me a doctor",
            "Dr. Garcia is available",
        );
        assert_eq!(interaction.source_app, "movilo");
        assert_eq!(interaction.user_id, "user-42");
        assert_eq!(interaction.feedback, FeedbackSignal::NoFeedback);
        assert!(interaction.interaction_id.starts_with("ai-movilo-"));
    }

    #[test]
    fn test_interaction_with_features() {
        let mut features = HashMap::new();
        features.insert("confidence".to_string(), 0.85);
        features.insert("length".to_string(), 0.6);

        let interaction = AiInteraction::new(
            "hera",
            "user-1",
            "general",
            "local-llm",
            "What is Rust?",
            "Rust is a systems programming language",
        )
        .with_features(features);

        assert_eq!(interaction.response_features.len(), 2);
        assert_eq!(interaction.response_features["confidence"], 0.85);
    }

    #[test]
    fn test_memento_payload_format() {
        let interaction = AiInteraction::new(
            "movilo",
            "user-1",
            "search",
            "gemini-2.5",
            "Find a dentist",
            "Dr. Garcia",
        )
        .with_feedback(FeedbackSignal::Like);

        let payload = interaction.to_memento_payload();
        assert_eq!(payload["action"], "log_interaction");
        assert_eq!(payload["payload"]["user_id"], "user-1");
        assert_eq!(payload["payload"]["domain"], "movilo:search");
        assert_eq!(payload["payload"]["choice_index"], 1); // Like = 1
    }

    #[test]
    fn test_feedback_stats_aggregation() {
        let interactions = sample_interactions();
        let stats = FeedbackStats::from_interactions(&interactions);

        assert_eq!(stats.total_interactions, 5);
        assert_eq!(stats.likes, 3);
        assert_eq!(stats.dislikes, 1);
        assert_eq!(stats.no_feedback, 1);
        assert_eq!(stats.satisfaction_rate(), 0.75); // 3 likes / 4 with feedback
    }

    #[test]
    fn test_per_app_stats() {
        let interactions = sample_interactions();
        let stats = FeedbackStats::from_interactions(&interactions);

        let movilo = stats.by_app.get("movilo").unwrap();
        assert_eq!(movilo.total, 2);
        assert_eq!(movilo.likes, 1);
        assert_eq!(movilo.dislikes, 1);
        assert_eq!(movilo.satisfaction_rate(), 0.5);

        let vetra = stats.by_app.get("vetra").unwrap();
        assert_eq!(vetra.total, 2);
        assert_eq!(vetra.likes, 2);
        assert_eq!(vetra.satisfaction_rate(), 1.0);
    }

    #[test]
    fn test_per_model_stats() {
        let interactions = sample_interactions();
        let stats = FeedbackStats::from_interactions(&interactions);

        let gemini = stats.by_model.get("gemini-2.5").unwrap();
        assert_eq!(gemini.total, 3);
        assert_eq!(gemini.likes, 1);
        assert_eq!(gemini.dislikes, 1);

        let qwen = stats.by_model.get("qwen-2.5").unwrap();
        assert_eq!(qwen.total, 2);
        assert_eq!(qwen.likes, 2);
    }

    #[test]
    fn test_truncation() {
        let long_prompt = "A".repeat(500);
        let interaction = AiInteraction::new(
            "hera",
            "user-1",
            "chat",
            "llm",
            &long_prompt,
            "Short response",
        );
        assert!(interaction.prompt_summary.len() <= 204); // 200 + "…" (3 bytes UTF-8)
    }

    #[test]
    fn test_stats_serialization() {
        let interactions = sample_interactions();
        let stats = FeedbackStats::from_interactions(&interactions);
        let json = serde_json::to_string(&stats).unwrap();
        let restored: FeedbackStats = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.total_interactions, stats.total_interactions);
        assert_eq!(restored.likes, stats.likes);
    }

    #[test]
    fn test_empty_stats() {
        let stats = FeedbackStats::from_interactions(&[]);
        assert_eq!(stats.total_interactions, 0);
        assert_eq!(stats.satisfaction_rate(), 0.0);
    }
}
