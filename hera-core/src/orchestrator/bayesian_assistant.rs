//! Bayesian Assistant — Multi-round adaptive recommendation agent
//!
//! Implements the Bayesian Assistant from Google Research's paper:
//! "Bayesian teaching enables probabilistic reasoning in large language models"
//!
//! The assistant maintains a belief distribution over user preferences,
//! updates it via Bayes' rule after each interaction, and recommends
//! items that maximize expected utility under current beliefs.

use serde::{Deserialize, Serialize};
use super::preference_model::{
    self, DomainSchema, Item, PreferenceDistribution,
};

// ═══════════════════════════════════════════════════════════════════
// Recommendation Result
// ═══════════════════════════════════════════════════════════════════

/// The assistant's recommendation plus metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    /// Index of the recommended item in the options array
    pub chosen_index: usize,
    /// Current confidence level (entropy-based, lower = more confident)
    pub entropy: f64,
    /// Current interaction round
    pub round: usize,
}

/// Record of a single interaction round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionRecord {
    /// The options presented
    pub options: Vec<Item>,
    /// What the assistant recommended
    pub recommended_index: usize,
    /// What the user actually chose
    pub user_choice_index: usize,
    /// Whether the assistant's recommendation matched
    pub correct: bool,
    /// Round number
    pub round: usize,
}

// ═══════════════════════════════════════════════════════════════════
// Bayesian Assistant
// ═══════════════════════════════════════════════════════════════════

/// A stateful multi-round recommendation assistant that uses Bayesian
/// inference to learn user preferences from their choices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BayesianAssistant {
    /// Current belief distribution over user preferences
    distribution: PreferenceDistribution,
    /// Softmax temperature for the likelihood model
    temperature: f64,
    /// Current interaction round (starts at 0)
    round: usize,
    /// History of all interactions
    history: Vec<InteractionRecord>,
}

impl BayesianAssistant {
    /// Create a new assistant with uniform prior beliefs for the given domain.
    ///
    /// # Arguments
    /// * `domain` — describes the features of items in this domain
    /// * `temperature` — softmax sharpness (default: 1.0, higher = sharper)
    pub fn new(domain: DomainSchema, temperature: f64) -> Self {
        Self {
            distribution: preference_model::uniform_prior(domain),
            temperature,
            round: 0,
            history: Vec::new(),
        }
    }

    /// Create with default temperature (1.0).
    pub fn with_domain(domain: DomainSchema) -> Self {
        Self::new(domain, 1.0)
    }

    /// Recommend the best item from the given options based on current beliefs.
    pub fn recommend(&self, options: &[Item]) -> Recommendation {
        let chosen_index = preference_model::predict_best(&self.distribution, options);
        let entropy = preference_model::entropy(&self.distribution);

        Recommendation {
            chosen_index,
            entropy,
            round: self.round,
        }
    }

    /// Observe the user's actual choice and update beliefs accordingly.
    ///
    /// Returns the interaction record for this round.
    pub fn observe(
        &mut self,
        options: &[Item],
        user_choice_index: usize,
    ) -> InteractionRecord {
        // Get our recommendation before updating
        let recommendation = self.recommend(options);

        // Update beliefs via Bayes' rule
        self.distribution = preference_model::bayesian_update(
            &self.distribution,
            user_choice_index,
            options,
            self.temperature,
        );

        let record = InteractionRecord {
            options: options.to_vec(),
            recommended_index: recommendation.chosen_index,
            user_choice_index,
            correct: recommendation.chosen_index == user_choice_index,
            round: self.round,
        };

        self.history.push(record.clone());
        self.round += 1;

        record
    }

    /// Get the current entropy (uncertainty) of the belief distribution.
    /// Lower entropy = more confident in the user's preferences.
    pub fn confidence(&self) -> f64 {
        preference_model::entropy(&self.distribution)
    }

    /// Get the current round number.
    pub fn current_round(&self) -> usize {
        self.round
    }

    /// Get the full interaction history.
    pub fn history(&self) -> &[InteractionRecord] {
        &self.history
    }

    /// Get accuracy so far (fraction of correct recommendations).
    pub fn accuracy(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        let correct = self.history.iter().filter(|r| r.correct).count();
        correct as f64 / self.history.len() as f64
    }

    /// Reset the assistant back to uniform prior (clears history).
    pub fn reset(&mut self) {
        self.distribution =
            preference_model::uniform_prior(self.distribution.domain.clone());
        self.round = 0;
        self.history.clear();
    }

    /// Get the current belief distribution (for serialization/persistence).
    pub fn distribution(&self) -> &PreferenceDistribution {
        &self.distribution
    }

    /// Restore from a previously saved distribution.
    pub fn restore_distribution(&mut self, dist: PreferenceDistribution) {
        self.distribution = dist;
    }

    /// Get marginal preference probabilities for each feature.
    pub fn marginals(
        &self,
    ) -> std::collections::HashMap<
        String,
        std::collections::HashMap<preference_model::FeaturePreference, f64>,
    > {
        preference_model::marginal_preferences(&self.distribution)
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn flight_domain() -> DomainSchema {
        DomainSchema {
            name: "flights".to_string(),
            features: vec![
                "cost".to_string(),
                "duration".to_string(),
                "stops".to_string(),
            ],
        }
    }

    fn sample_options() -> Vec<Item> {
        vec![
            Item { features: vec![0.8, 0.3, 0.0] }, // expensive, short, direct
            Item { features: vec![0.3, 0.7, 0.5] }, // cheap, long, 1 stop
            Item { features: vec![0.5, 0.5, 1.0] }, // medium, medium, many stops
        ]
    }

    #[test]
    fn test_new_assistant_has_uniform_prior() {
        let assistant = BayesianAssistant::with_domain(flight_domain());
        assert_eq!(assistant.current_round(), 0);
        assert!(assistant.history().is_empty());

        // Entropy should be maximal for uniform distribution
        let ent = assistant.confidence();
        assert!(ent > 0.0, "Initial entropy should be positive");
    }

    #[test]
    fn test_multi_round_accuracy_improves() {
        let mut assistant = BayesianAssistant::new(flight_domain(), 2.0);

        // Simulate a user who always prefers low cost (picks cheapest)
        let mut correct_count = [0usize; 5];
        let total_simulations = 50;

        for _ in 0..total_simulations {
            assistant.reset();

            for round in 0..5 {
                let options = vec![
                    Item { features: vec![0.9, 0.5, 0.3] }, // expensive
                    Item { features: vec![0.1, 0.5, 0.6] }, // cheap (user picks this)
                    Item { features: vec![0.5, 0.5, 0.1] }, // medium
                ];

                let rec = assistant.recommend(&options);
                if rec.chosen_index == 1 {
                    correct_count[round] += 1;
                }
                assistant.observe(&options, 1); // user always picks cheap
            }
        }

        let accuracy_round1 = correct_count[0] as f64 / total_simulations as f64;
        let accuracy_round5 = correct_count[4] as f64 / total_simulations as f64;

        // The core claim: accuracy should improve over rounds
        assert!(
            accuracy_round5 >= accuracy_round1,
            "Round 5 accuracy ({}) should be >= round 1 accuracy ({})",
            accuracy_round5,
            accuracy_round1
        );

        // By round 5 with consistent evidence, should be very accurate
        assert!(
            accuracy_round5 > 0.7,
            "Round 5 accuracy ({}) should be > 0.7 with consistent evidence",
            accuracy_round5
        );
    }

    #[test]
    fn test_confidence_increases_with_evidence() {
        let mut assistant = BayesianAssistant::new(flight_domain(), 1.0);
        let initial_entropy = assistant.confidence();

        let options = sample_options();
        assistant.observe(&options, 1); // user picks option 1
        let entropy_after_1 = assistant.confidence();

        assistant.observe(&options, 1); // user picks option 1 again
        let entropy_after_2 = assistant.confidence();

        assert!(
            entropy_after_1 < initial_entropy,
            "Entropy should decrease after 1 observation"
        );
        assert!(
            entropy_after_2 < entropy_after_1,
            "Entropy should decrease further after 2 observations"
        );
    }

    #[test]
    fn test_observe_records_history() {
        let mut assistant = BayesianAssistant::with_domain(flight_domain());
        let options = sample_options();

        assistant.observe(&options, 0);
        assistant.observe(&options, 1);
        assistant.observe(&options, 2);

        assert_eq!(assistant.history().len(), 3);
        assert_eq!(assistant.current_round(), 3);
        assert_eq!(assistant.history()[0].user_choice_index, 0);
        assert_eq!(assistant.history()[1].user_choice_index, 1);
        assert_eq!(assistant.history()[2].user_choice_index, 2);
    }

    #[test]
    fn test_reset_clears_state() {
        let mut assistant = BayesianAssistant::with_domain(flight_domain());
        let options = sample_options();

        assistant.observe(&options, 0);
        assistant.observe(&options, 0);
        let post_entropy = assistant.confidence();

        assistant.reset();
        let reset_entropy = assistant.confidence();

        assert!(
            reset_entropy > post_entropy,
            "Reset should restore to higher (uniform) entropy"
        );
        assert_eq!(assistant.current_round(), 0);
        assert!(assistant.history().is_empty());
    }

    #[test]
    fn test_accuracy_tracking() {
        let mut assistant = BayesianAssistant::new(flight_domain(), 2.0);

        // First, train the assistant with 5 rounds of consistent data
        for _ in 0..5 {
            let options = vec![
                Item { features: vec![0.9, 0.5, 0.5] },
                Item { features: vec![0.1, 0.5, 0.5] }, // cheap
            ];
            assistant.observe(&options, 1);
        }

        // Accuracy should be between 0 and 1
        let acc = assistant.accuracy();
        assert!(acc >= 0.0 && acc <= 1.0);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut assistant = BayesianAssistant::new(flight_domain(), 1.5);
        let options = sample_options();
        assistant.observe(&options, 0);

        let json = serde_json::to_string(&assistant).unwrap();
        let restored: BayesianAssistant = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.current_round(), assistant.current_round());
        assert_eq!(restored.history().len(), assistant.history().len());
        assert!(
            (restored.confidence() - assistant.confidence()).abs() < 1e-10,
            "Restored entropy should match original"
        );
    }

    #[test]
    fn test_marginal_reflects_evidence() {
        let mut assistant = BayesianAssistant::new(flight_domain(), 2.0);

        // User consistently picks low cost
        for _ in 0..5 {
            let options = vec![
                Item { features: vec![0.9, 0.5, 0.5] }, // expensive
                Item { features: vec![0.1, 0.5, 0.5] }, // cheap
            ];
            assistant.observe(&options, 1);
        }

        let marginals = assistant.marginals();
        let cost_marginal = marginals.get("cost").unwrap();

        // After consistent low-cost choices, StrongLow/WeakLow should dominate
        let low_prob = cost_marginal
            .get(&preference_model::FeaturePreference::StrongLow)
            .unwrap_or(&0.0)
            + cost_marginal
                .get(&preference_model::FeaturePreference::WeakLow)
                .unwrap_or(&0.0);

        let high_prob = cost_marginal
            .get(&preference_model::FeaturePreference::StrongHigh)
            .unwrap_or(&0.0)
            + cost_marginal
                .get(&preference_model::FeaturePreference::WeakHigh)
                .unwrap_or(&0.0);

        assert!(
            low_prob > high_prob,
            "Low-cost preference ({}) should dominate high-cost ({}) after consistent evidence",
            low_prob,
            high_prob
        );
    }
}
