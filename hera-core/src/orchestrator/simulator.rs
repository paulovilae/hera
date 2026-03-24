//! Simulator — Synthetic User Generation for Bayesian Teaching
//!
//! Generates synthetic user profiles with known preferences, runs
//! multi-round interactions against the Bayesian Assistant, and
//! captures full interaction traces for training data generation.
//!
//! This implements the "Bayesian Teaching" pipeline from Google Research:
//! 1. Define domain schemas with realistic feature sets
//! 2. Generate diverse synthetic users with controlled preferences
//! 3. Simulate multi-round recommendation sessions
//! 4. Record (prompt, completion) pairs showing optimal Bayesian reasoning

use serde::{Deserialize, Serialize};
use super::preference_model::{
    DomainSchema, FeaturePreference, Item, PreferenceDistribution,
};
use super::bayesian_assistant::BayesianAssistant;
use super::benchmark::SimulatedUser;

// ═══════════════════════════════════════════════════════════════════
// Predefined Domain Schemas
// ═══════════════════════════════════════════════════════════════════

/// Healthcare provider domain (Movilo).
pub fn movilo_provider_domain() -> DomainSchema {
    DomainSchema {
        name: "movilo_providers".to_string(),
        features: vec![
            "cost".to_string(),         // consultation cost (normalized)
            "rating".to_string(),       // patient rating [0,1]
            "distance".to_string(),     // distance from user
            "wait_time".to_string(),    // average wait time
            "experience".to_string(),   // years of experience (normalized)
        ],
    }
}

/// Trade/financial instrument domain (Vetra).
pub fn vetra_trade_domain() -> DomainSchema {
    DomainSchema {
        name: "vetra_trades".to_string(),
        features: vec![
            "volatility".to_string(),   // price volatility
            "yield_rate".to_string(),    // expected return
            "risk_score".to_string(),    // calculated risk
            "liquidity".to_string(),     // trade volume / liquidity
        ],
    }
}

/// Flight booking domain (from the paper's examples).
pub fn flight_domain() -> DomainSchema {
    DomainSchema {
        name: "flights".to_string(),
        features: vec![
            "cost".to_string(),
            "duration".to_string(),
            "stops".to_string(),
        ],
    }
}

// ═══════════════════════════════════════════════════════════════════
// Interaction Trace — captures a full session for training
// ═══════════════════════════════════════════════════════════════════

/// A single round in an interaction trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceRound {
    pub round: usize,
    /// The options presented to the user
    pub options: Vec<Item>,
    /// What the Bayesian assistant recommended
    pub assistant_pick: usize,
    /// What the user actually chose
    pub user_choice: usize,
    /// Entropy before this round's observation
    pub entropy_before: f64,
    /// Entropy after this round's observation
    pub entropy_after: f64,
    /// Whether the assistant was correct
    pub correct: bool,
    /// Marginal probabilities per feature after update
    pub marginals_after: std::collections::HashMap<
        String,
        std::collections::HashMap<String, f64>,
    >,
}

/// A complete interaction session trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionTrace {
    pub domain: String,
    pub user_id: String,
    /// The user's true (ground truth) preferences
    pub true_preferences: Vec<String>,
    pub temperature: f64,
    pub rounds: Vec<TraceRound>,
    pub final_accuracy: f64,
}

// ═══════════════════════════════════════════════════════════════════
// Simulator Engine
// ═══════════════════════════════════════════════════════════════════

/// Configuration for the simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    pub domain: DomainSchema,
    pub n_users: usize,
    pub n_rounds: usize,
    pub n_options: usize,
    pub temperature: f64,
    /// Base seed for deterministic reproducibility
    pub base_seed: u64,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            domain: flight_domain(),
            n_users: 100,
            n_rounds: 5,
            n_options: 3,
            temperature: 2.0,
            base_seed: 42,
        }
    }
}

/// Generate deterministic random options for a round.
fn generate_options(n_features: usize, n_options: usize, seed: u64) -> Vec<Item> {
    let mut items = Vec::with_capacity(n_options);
    let mut rng = seed;
    for _ in 0..n_options {
        let mut features = Vec::with_capacity(n_features);
        for _ in 0..n_features {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            features.push(((rng >> 33) as f64) / (u32::MAX as f64));
        }
        items.push(Item { features });
    }
    items
}

/// Convert FeaturePreference to a human-readable string.
fn pref_to_string(pref: &FeaturePreference) -> String {
    match pref {
        FeaturePreference::StrongHigh => "strong_high".to_string(),
        FeaturePreference::WeakHigh => "weak_high".to_string(),
        FeaturePreference::None => "none".to_string(),
        FeaturePreference::WeakLow => "weak_low".to_string(),
        FeaturePreference::StrongLow => "strong_low".to_string(),
    }
}

/// Convert marginals (using FeaturePreference keys) to string-keyed map for serialization.
fn stringify_marginals(
    marginals: &std::collections::HashMap<String, std::collections::HashMap<FeaturePreference, f64>>,
) -> std::collections::HashMap<String, std::collections::HashMap<String, f64>> {
    marginals
        .iter()
        .map(|(feat, prefs)| {
            let string_map: std::collections::HashMap<String, f64> = prefs
                .iter()
                .map(|(pref, &prob)| (pref_to_string(pref), prob))
                .collect();
            (feat.clone(), string_map)
        })
        .collect()
}

/// Run a full simulation and return all interaction traces.
pub fn run_simulation(config: &SimulationConfig) -> Vec<InteractionTrace> {
    let n_features = config.domain.features.len();
    let mut traces = Vec::with_capacity(config.n_users);

    for user_idx in 0..config.n_users {
        let user_seed = config.base_seed.wrapping_mul(user_idx as u64 + 1).wrapping_add(7919);
        let user = SimulatedUser::random(n_features, user_seed);
        let mut assistant = BayesianAssistant::new(config.domain.clone(), config.temperature);

        let true_prefs: Vec<String> = user.preferences.iter().map(pref_to_string).collect();
        let mut rounds = Vec::with_capacity(config.n_rounds);

        for round in 0..config.n_rounds {
            let option_seed = config
                .base_seed
                .wrapping_mul((user_idx * 1000 + round) as u64 + 1)
                .wrapping_add(104729);
            let options = generate_options(n_features, config.n_options, option_seed);

            let entropy_before = assistant.confidence();
            let rec = assistant.recommend(&options);
            let user_choice = user.choose(&options);

            // Observe and update beliefs
            assistant.observe(&options, user_choice);

            let entropy_after = assistant.confidence();
            let marginals = stringify_marginals(&assistant.marginals());

            rounds.push(TraceRound {
                round,
                options: options.clone(),
                assistant_pick: rec.chosen_index,
                user_choice,
                entropy_before,
                entropy_after,
                correct: rec.chosen_index == user_choice,
                marginals_after: marginals,
            });
        }

        let final_accuracy = assistant.accuracy();

        traces.push(InteractionTrace {
            domain: config.domain.name.clone(),
            user_id: format!("synth-user-{:04}", user_idx),
            true_preferences: true_prefs,
            temperature: config.temperature,
            rounds,
            final_accuracy,
        });
    }

    traces
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_movilo_domain_has_5_features() {
        let domain = movilo_provider_domain();
        assert_eq!(domain.features.len(), 5);
        assert!(domain.features.contains(&"cost".to_string()));
        assert!(domain.features.contains(&"rating".to_string()));
    }

    #[test]
    fn test_vetra_domain_has_4_features() {
        let domain = vetra_trade_domain();
        assert_eq!(domain.features.len(), 4);
        assert!(domain.features.contains(&"volatility".to_string()));
        assert!(domain.features.contains(&"yield_rate".to_string()));
    }

    #[test]
    fn test_simulation_produces_correct_trace_count() {
        let config = SimulationConfig {
            domain: flight_domain(),
            n_users: 5,
            n_rounds: 3,
            n_options: 3,
            temperature: 1.0,
            base_seed: 42,
        };

        let traces = run_simulation(&config);
        assert_eq!(traces.len(), 5);
        for trace in &traces {
            assert_eq!(trace.rounds.len(), 3);
            assert_eq!(trace.domain, "flights");
        }
    }

    #[test]
    fn test_simulation_is_deterministic() {
        let config = SimulationConfig {
            domain: flight_domain(),
            n_users: 3,
            n_rounds: 3,
            n_options: 3,
            temperature: 1.0,
            base_seed: 42,
        };

        let traces1 = run_simulation(&config);
        let traces2 = run_simulation(&config);

        for (t1, t2) in traces1.iter().zip(traces2.iter()) {
            assert_eq!(t1.true_preferences, t2.true_preferences);
            assert_eq!(t1.final_accuracy, t2.final_accuracy);
            for (r1, r2) in t1.rounds.iter().zip(t2.rounds.iter()) {
                assert_eq!(r1.user_choice, r2.user_choice);
                assert_eq!(r1.assistant_pick, r2.assistant_pick);
            }
        }
    }

    #[test]
    fn test_entropy_decreases_in_traces() {
        let config = SimulationConfig {
            domain: flight_domain(),
            n_users: 10,
            n_rounds: 5,
            n_options: 3,
            temperature: 2.0,
            base_seed: 42,
        };

        let traces = run_simulation(&config);

        // On average, entropy should decrease
        let avg_first_entropy: f64 = traces
            .iter()
            .map(|t| t.rounds[0].entropy_before)
            .sum::<f64>()
            / traces.len() as f64;

        let avg_last_entropy: f64 = traces
            .iter()
            .map(|t| t.rounds.last().unwrap().entropy_after)
            .sum::<f64>()
            / traces.len() as f64;

        assert!(
            avg_last_entropy < avg_first_entropy,
            "Average entropy should decrease: first={}, last={}",
            avg_first_entropy,
            avg_last_entropy
        );
    }

    #[test]
    fn test_trace_serialization() {
        let config = SimulationConfig {
            domain: flight_domain(),
            n_users: 2,
            n_rounds: 2,
            n_options: 3,
            temperature: 1.0,
            base_seed: 42,
        };

        let traces = run_simulation(&config);
        let json = serde_json::to_string(&traces).unwrap();
        let restored: Vec<InteractionTrace> = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.len(), traces.len());
    }

    #[test]
    fn test_movilo_simulation() {
        let config = SimulationConfig {
            domain: movilo_provider_domain(),
            n_users: 20,
            n_rounds: 5,
            n_options: 4,
            temperature: 2.0,
            base_seed: 99,
        };

        let traces = run_simulation(&config);
        assert_eq!(traces.len(), 20);
        assert_eq!(traces[0].domain, "movilo_providers");
        // Each trace should have correct preference count
        for trace in &traces {
            assert_eq!(trace.true_preferences.len(), 5);
        }
    }
}
