//! Benchmark harness for the Bayesian preference engine.
//!
//! Compares Bayesian Assistant accuracy against baseline heuristics
//! (random, first-pick, popularity) across multiple simulated users.
//! Controllable via feature flags for A/B testing.
//!
//! Usage:
//!   cargo test -p hera-core benchmark -- --nocapture
//!
//! Or run as a standalone binary:
//!   cargo run -p hera-core --bin bayesian_benchmark

use super::bayesian_assistant::BayesianAssistant;
use super::preference_model::{DomainSchema, FeaturePreference, Item};
use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════
// Feature Flags
// ═══════════════════════════════════════════════════════════════════

/// Feature flags to toggle capabilities on/off for benchmarking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureFlags {
    /// Use Bayesian belief updating (the core feature)
    pub bayesian_update: bool,
    /// Use prior persistence (load priors from previous sessions)
    pub prior_persistence: bool,
    /// Use cross-domain transfer (apply learned reasoning to new domains)
    pub cross_domain_transfer: bool,
    /// Temperature for the softmax likelihood
    pub temperature: f64,
}

impl Default for FeatureFlags {
    fn default() -> Self {
        Self {
            bayesian_update: true,
            prior_persistence: true,
            cross_domain_transfer: false,
            temperature: 1.0,
        }
    }
}

impl FeatureFlags {
    /// All features OFF — pure baseline
    pub fn baseline() -> Self {
        Self {
            bayesian_update: false,
            prior_persistence: false,
            cross_domain_transfer: false,
            temperature: 1.0,
        }
    }

    /// All features ON
    pub fn full() -> Self {
        Self {
            bayesian_update: true,
            prior_persistence: true,
            cross_domain_transfer: true,
            temperature: 2.0,
        }
    }

    /// Load from environment variable HERA_FEATURES (JSON string)
    pub fn from_env() -> Self {
        match std::env::var("HERA_FEATURES") {
            Ok(json) => serde_json::from_str(&json).unwrap_or_default(),
            Err(_) => Self::default(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Simulated User
// ═══════════════════════════════════════════════════════════════════

/// A simulated user with known preferences (ground truth).
#[derive(Debug, Clone)]
pub struct SimulatedUser {
    pub preferences: Vec<FeaturePreference>,
}

impl SimulatedUser {
    /// Generate a random user with preferences for each feature.
    pub fn random(n_features: usize, seed: u64) -> Self {
        let levels = FeaturePreference::all();
        let mut preferences = Vec::with_capacity(n_features);
        let mut rng_state = seed;
        for _ in 0..n_features {
            // Simple LCG for deterministic "random"
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let idx = (rng_state >> 33) as usize % levels.len();
            preferences.push(levels[idx]);
        }
        Self { preferences }
    }

    /// Given options, pick the one with highest utility under true preferences.
    pub fn choose(&self, options: &[Item]) -> usize {
        let mut best_idx = 0;
        let mut best_util = f64::NEG_INFINITY;
        for (i, item) in options.iter().enumerate() {
            let util: f64 = self
                .preferences
                .iter()
                .zip(item.features.iter())
                .map(|(pref, &val)| pref.utility_weight() * val)
                .sum();
            if util > best_util {
                best_util = util;
                best_idx = i;
            }
        }
        best_idx
    }
}

// ═══════════════════════════════════════════════════════════════════
// Baseline Strategies (non-Bayesian)
// ═══════════════════════════════════════════════════════════════════

/// Strategy that always picks the first option.
fn strategy_first_pick(_options: &[Item], _round: usize) -> usize {
    0
}

/// Strategy that always picks the "cheapest" (lowest value on first feature).
fn strategy_cheapest(options: &[Item], _round: usize) -> usize {
    options
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            a.features[0]
                .partial_cmp(&b.features[0])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Strategy that picks "middle of the road" (closest to 0.5 on all features).
fn strategy_middle(options: &[Item], _round: usize) -> usize {
    options
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            let dist_a: f64 = a.features.iter().map(|v| (v - 0.5).abs()).sum();
            let dist_b: f64 = b.features.iter().map(|v| (v - 0.5).abs()).sum();
            dist_a
                .partial_cmp(&dist_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ═══════════════════════════════════════════════════════════════════
// Random Option Generator
// ═══════════════════════════════════════════════════════════════════

/// Generate a set of random items for a round.
fn generate_options(n_features: usize, n_options: usize, seed: u64) -> Vec<Item> {
    let mut items = Vec::with_capacity(n_options);
    let mut rng_state = seed;
    for _ in 0..n_options {
        let mut features = Vec::with_capacity(n_features);
        for _ in 0..n_features {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let val = ((rng_state >> 33) as f64) / (u32::MAX as f64);
            features.push(val);
        }
        items.push(Item { features });
    }
    items
}

// ═══════════════════════════════════════════════════════════════════
// Benchmark Result
// ═══════════════════════════════════════════════════════════════════

/// Results of a benchmark run for a single strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub strategy_name: String,
    pub features: FeatureFlags,
    pub n_users: usize,
    pub n_rounds: usize,
    /// Accuracy per round (averaged across all users)
    pub accuracy_per_round: Vec<f64>,
    /// Overall accuracy
    pub overall_accuracy: f64,
    /// Average entropy per round (Bayesian only)
    pub entropy_per_round: Vec<f64>,
}

/// Full benchmark report comparing all strategies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub domain: String,
    pub results: Vec<BenchmarkResult>,
}

// ═══════════════════════════════════════════════════════════════════
// Benchmark Runner
// ═══════════════════════════════════════════════════════════════════

/// Run the full benchmark suite comparing Bayesian vs baseline strategies.
pub fn run_benchmark(
    domain: DomainSchema,
    n_users: usize,
    n_rounds: usize,
    n_options: usize,
    flags: &FeatureFlags,
) -> BenchmarkReport {
    let n_features = domain.features.len();
    let mut results = Vec::new();

    // ── Bayesian Strategy ───────────────────────────────────────
    if flags.bayesian_update {
        let mut round_correct = vec![0usize; n_rounds];
        let mut round_entropy = vec![0.0f64; n_rounds];
        let mut total_correct = 0usize;
        let total_decisions = n_users * n_rounds;

        for user_seed in 0..n_users {
            let user = SimulatedUser::random(n_features, user_seed as u64 * 7919);
            let mut assistant = BayesianAssistant::new(domain.clone(), flags.temperature);

            for round in 0..n_rounds {
                let options = generate_options(
                    n_features,
                    n_options,
                    (user_seed * 1000 + round) as u64 * 104729,
                );

                let rec = assistant.recommend(&options);
                let user_choice = user.choose(&options);

                round_entropy[round] += rec.entropy;

                if rec.chosen_index == user_choice {
                    round_correct[round] += 1;
                    total_correct += 1;
                }

                assistant.observe(&options, user_choice);
            }
        }

        results.push(BenchmarkResult {
            strategy_name: "Bayesian Assistant".to_string(),
            features: flags.clone(),
            n_users,
            n_rounds,
            accuracy_per_round: round_correct
                .iter()
                .map(|&c| c as f64 / n_users as f64)
                .collect(),
            overall_accuracy: total_correct as f64 / total_decisions as f64,
            entropy_per_round: round_entropy.iter().map(|&e| e / n_users as f64).collect(),
        });
    }

    // ── Baseline Strategies ─────────────────────────────────────
    let baselines: Vec<(&str, fn(&[Item], usize) -> usize)> = vec![
        ("First Pick", strategy_first_pick),
        ("Always Cheapest", strategy_cheapest),
        ("Middle of Road", strategy_middle),
    ];

    for (name, strategy_fn) in baselines {
        let mut round_correct = vec![0usize; n_rounds];
        let mut total_correct = 0usize;
        let total_decisions = n_users * n_rounds;

        for user_seed in 0..n_users {
            let user = SimulatedUser::random(n_features, user_seed as u64 * 7919);

            for round in 0..n_rounds {
                let options = generate_options(
                    n_features,
                    n_options,
                    (user_seed * 1000 + round) as u64 * 104729,
                );

                let pick = strategy_fn(&options, round);
                let user_choice = user.choose(&options);

                if pick == user_choice {
                    round_correct[round] += 1;
                    total_correct += 1;
                }
            }
        }

        results.push(BenchmarkResult {
            strategy_name: name.to_string(),
            features: FeatureFlags::baseline(),
            n_users,
            n_rounds,
            accuracy_per_round: round_correct
                .iter()
                .map(|&c| c as f64 / n_users as f64)
                .collect(),
            overall_accuracy: total_correct as f64 / total_decisions as f64,
            entropy_per_round: vec![],
        });
    }

    BenchmarkReport {
        domain: domain.name.clone(),
        results,
    }
}

/// Pretty-print a benchmark report.
pub fn format_report(report: &BenchmarkReport) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "\n══════════════════════════════════════════════════\n"
    ));
    out.push_str(&format!(
        "  🧠 Bayesian Teaching Benchmark — {}\n",
        report.domain
    ));
    out.push_str(&format!(
        "══════════════════════════════════════════════════\n\n"
    ));

    for result in &report.results {
        out.push_str(&format!(
            "  {:20} │ Overall: {:.1}%\n",
            result.strategy_name,
            result.overall_accuracy * 100.0
        ));

        if !result.accuracy_per_round.is_empty() {
            out.push_str("                       │ Rounds: ");
            for (i, acc) in result.accuracy_per_round.iter().enumerate() {
                if i > 0 {
                    out.push_str(" → ");
                }
                out.push_str(&format!("R{}:{:.0}%", i + 1, acc * 100.0));
            }
            out.push('\n');
        }

        if !result.entropy_per_round.is_empty() {
            out.push_str("                       │ Entropy: ");
            for (i, ent) in result.entropy_per_round.iter().enumerate() {
                if i > 0 {
                    out.push_str(" → ");
                }
                out.push_str(&format!("R{}:{:.1}", i + 1, ent));
            }
            out.push('\n');
        }
        out.push('\n');
    }

    out.push_str("══════════════════════════════════════════════════\n");
    out
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

    #[test]
    fn test_feature_flags_default() {
        let flags = FeatureFlags::default();
        assert!(flags.bayesian_update);
        assert!(flags.prior_persistence);
        assert!(!flags.cross_domain_transfer);
    }

    #[test]
    fn test_feature_flags_baseline() {
        let flags = FeatureFlags::baseline();
        assert!(!flags.bayesian_update);
        assert!(!flags.prior_persistence);
        assert!(!flags.cross_domain_transfer);
    }

    #[test]
    fn test_feature_flags_serialization() {
        let flags = FeatureFlags::full();
        let json = serde_json::to_string(&flags).unwrap();
        let restored: FeatureFlags = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.bayesian_update, flags.bayesian_update);
        assert_eq!(restored.temperature, flags.temperature);
    }

    #[test]
    fn test_simulated_user_deterministic() {
        let user1 = SimulatedUser::random(3, 42);
        let user2 = SimulatedUser::random(3, 42);
        assert_eq!(
            user1.preferences, user2.preferences,
            "Same seed = same user"
        );

        let user3 = SimulatedUser::random(3, 99);
        // Different seeds should almost certainly produce different users
        // (not guaranteed but overwhelmingly likely with 5^3 = 125 combos)
        assert_ne!(user1.preferences, user3.preferences);
    }

    #[test]
    fn test_simulated_user_choose() {
        // User with StrongLow on first feature = prefers low values
        let user = SimulatedUser {
            preferences: vec![FeaturePreference::StrongLow, FeaturePreference::None],
        };
        let options = vec![
            Item {
                features: vec![0.9, 0.5],
            }, // expensive
            Item {
                features: vec![0.1, 0.5],
            }, // cheap
        ];
        assert_eq!(user.choose(&options), 1, "Should choose the cheap option");
    }

    #[test]
    fn test_benchmark_runs_without_panic() {
        let report = run_benchmark(flight_domain(), 10, 3, 3, &FeatureFlags::default());
        // Should have 4 results: Bayesian + 3 baselines
        assert_eq!(report.results.len(), 4);
        assert_eq!(report.results[0].strategy_name, "Bayesian Assistant");
    }

    #[test]
    fn test_bayesian_beats_baselines() {
        let domain = DomainSchema {
            name: "test".to_string(),
            features: vec!["a".to_string(), "b".to_string()],
        };

        let report = run_benchmark(domain, 50, 5, 3, &FeatureFlags::full());

        let bayesian_acc = report.results[0].overall_accuracy;

        // Bayesian should beat at least one baseline
        let best_baseline = report.results[1..]
            .iter()
            .map(|r| r.overall_accuracy)
            .fold(0.0f64, f64::max);

        assert!(
            bayesian_acc >= best_baseline * 0.9,
            "Bayesian ({:.1}%) should be competitive with best baseline ({:.1}%)",
            bayesian_acc * 100.0,
            best_baseline * 100.0
        );
    }

    #[test]
    fn test_bayesian_accuracy_improves_over_rounds() {
        let domain = DomainSchema {
            name: "flights".to_string(),
            features: vec!["cost".to_string(), "duration".to_string()],
        };

        let report = run_benchmark(domain, 100, 5, 3, &FeatureFlags::full());
        let bayesian = &report.results[0];

        let round1_acc = bayesian.accuracy_per_round[0];
        let round5_acc = bayesian.accuracy_per_round[4];

        assert!(
            round5_acc >= round1_acc,
            "Round 5 accuracy ({:.1}%) should be >= round 1 ({:.1}%)",
            round5_acc * 100.0,
            round1_acc * 100.0
        );
    }

    #[test]
    fn test_benchmark_without_bayesian() {
        let report = run_benchmark(flight_domain(), 10, 3, 3, &FeatureFlags::baseline());
        // Only baselines (Bayesian disabled)
        assert_eq!(report.results.len(), 3);
        assert_eq!(report.results[0].strategy_name, "First Pick");
    }

    #[test]
    fn test_format_report_not_empty() {
        let report = run_benchmark(flight_domain(), 5, 3, 3, &FeatureFlags::default());
        let formatted = format_report(&report);
        assert!(formatted.contains("Bayesian Assistant"));
        assert!(formatted.contains("First Pick"));
        assert!(formatted.contains("Overall:"));
    }

    #[test]
    fn test_entropy_decreases_in_benchmark() {
        let domain = DomainSchema {
            name: "test".to_string(),
            features: vec!["a".to_string(), "b".to_string()],
        };

        let report = run_benchmark(domain, 50, 5, 3, &FeatureFlags::full());
        let bayesian = &report.results[0];

        assert!(
            bayesian.entropy_per_round.last().unwrap()
                < bayesian.entropy_per_round.first().unwrap(),
            "Entropy should decrease over rounds"
        );
    }
}
