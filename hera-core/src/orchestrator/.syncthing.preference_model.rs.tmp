//! Bayesian Preference Model
//!
//! Implements the symbolic Bayesian inference engine described in Google Research's
//! "Bayesian teaching enables probabilistic reasoning in large language models"
//! (March 2026). This module provides:
//!
//! - A domain-agnostic preference representation
//! - Uniform prior initialization
//! - Bayes' rule posterior update
//! - Prediction (argmax expected utility)
//! - Entropy-based confidence measurement
//!
//! Operates in log-probability space for numerical stability.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════
// Domain Schema — defines what features an item has
// ═══════════════════════════════════════════════════════════════════

/// Describes a domain (flights, hotels, providers, etc.) by its feature names.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainSchema {
    /// Human-readable domain name (e.g. "flights", "providers")
    pub name: String,
    /// Feature names in this domain (e.g. ["departure_time", "duration", "stops", "cost"])
    pub features: Vec<String>,
}

// ═══════════════════════════════════════════════════════════════════
// Feature Preferences
// ═══════════════════════════════════════════════════════════════════

/// Possible user preference for a single feature.
/// Matches the paper's 5-level preference scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FeaturePreference {
    /// Strong preference for high values
    StrongHigh,
    /// Weak preference for high values
    WeakHigh,
    /// No preference (indifferent)
    None,
    /// Weak preference for low values
    WeakLow,
    /// Strong preference for low values
    StrongLow,
}

impl FeaturePreference {
    /// All possible preference levels
    pub fn all() -> &'static [FeaturePreference] {
        &[
            FeaturePreference::StrongHigh,
            FeaturePreference::WeakHigh,
            FeaturePreference::None,
            FeaturePreference::WeakLow,
            FeaturePreference::StrongLow,
        ]
    }

    /// Returns the utility weight for this preference given a normalized feature value [0, 1].
    /// Positive weights favor higher values, negative favor lower.
    pub fn utility_weight(&self) -> f64 {
        match self {
            FeaturePreference::StrongHigh => 2.0,
            FeaturePreference::WeakHigh => 1.0,
            FeaturePreference::None => 0.0,
            FeaturePreference::WeakLow => -1.0,
            FeaturePreference::StrongLow => -2.0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Preference Profile — one preference per feature
// ═══════════════════════════════════════════════════════════════════

/// A complete user preference profile: one FeaturePreference per feature.
pub type PreferenceProfile = Vec<FeaturePreference>;

// ═══════════════════════════════════════════════════════════════════
// Item — an option presented to the user
// ═══════════════════════════════════════════════════════════════════

/// An item (e.g., a flight, hotel, provider) with normalized feature values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Item {
    /// Feature values normalized to [0, 1] range.
    /// Must have the same length as the domain's features.
    pub features: Vec<f64>,
}

// ═══════════════════════════════════════════════════════════════════
// Preference Distribution — prior/posterior over all profiles
// ═══════════════════════════════════════════════════════════════════

/// Probability distribution over all possible preference profiles.
/// Operates in log-probability space for numerical stability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreferenceDistribution {
    /// The domain this distribution applies to
    pub domain: DomainSchema,
    /// All possible preference profiles (enumerated)
    pub profiles: Vec<PreferenceProfile>,
    /// Log-probabilities for each profile (same order as `profiles`)
    pub log_probs: Vec<f64>,
}

impl PreferenceDistribution {
    /// Enumerate all possible preference profiles for the domain.
    /// For N features with 5 levels each, this produces 5^N profiles.
    fn enumerate_profiles(n_features: usize) -> Vec<PreferenceProfile> {
        let levels = FeaturePreference::all();
        let n_profiles = levels.len().pow(n_features as u32);
        let mut profiles = Vec::with_capacity(n_profiles);

        for i in 0..n_profiles {
            let mut profile = Vec::with_capacity(n_features);
            let mut idx = i;
            for _ in 0..n_features {
                profile.push(levels[idx % levels.len()]);
                idx /= levels.len();
            }
            profiles.push(profile);
        }

        profiles
    }

    /// Compute utility of an item under a given preference profile.
    pub fn utility(profile: &PreferenceProfile, item: &Item) -> f64 {
        profile
            .iter()
            .zip(item.features.iter())
            .map(|(pref, &val)| pref.utility_weight() * val)
            .sum()
    }

    /// Log-sum-exp for numerical stability (avoids exp overflow/underflow).
    fn log_sum_exp(log_vals: &[f64]) -> f64 {
        if log_vals.is_empty() {
            return f64::NEG_INFINITY;
        }
        let max_val = log_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if max_val == f64::NEG_INFINITY {
            return f64::NEG_INFINITY;
        }
        max_val
            + log_vals
                .iter()
                .map(|&v| (v - max_val).exp())
                .sum::<f64>()
                .ln()
    }
}

// ═══════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════

/// Create a uniform (flat) prior over all preference profiles.
pub fn uniform_prior(domain: DomainSchema) -> PreferenceDistribution {
    let profiles = PreferenceDistribution::enumerate_profiles(domain.features.len());
    let n = profiles.len() as f64;
    let log_prob = -(n.ln());

    PreferenceDistribution {
        domain,
        log_probs: vec![log_prob; profiles.len()],
        profiles,
    }
}

/// Bayesian update: compute posterior given a user choice.
///
/// Uses a softmax likelihood model: the probability that a user with profile `θ`
/// chooses item `c` from options `[o_1, ..., o_k]` is proportional to
/// `exp(β * utility(θ, c))` where β is a temperature parameter.
///
/// # Arguments
/// * `prior` — current belief distribution
/// * `chosen_index` — which item the user picked (0-indexed)
/// * `options` — all items that were presented
/// * `temperature` — softmax sharpness (higher = more deterministic choices)
pub fn bayesian_update(
    prior: &PreferenceDistribution,
    chosen_index: usize,
    options: &[Item],
    temperature: f64,
) -> PreferenceDistribution {
    assert!(chosen_index < options.len(), "chosen_index out of bounds");

    let mut new_log_probs = Vec::with_capacity(prior.profiles.len());

    for (profile, &log_prior) in prior.profiles.iter().zip(prior.log_probs.iter()) {
        // Compute log-likelihood: softmax over utilities
        let utilities: Vec<f64> = options
            .iter()
            .map(|item| temperature * PreferenceDistribution::utility(profile, item))
            .collect();

        let log_normalizer = PreferenceDistribution::log_sum_exp(&utilities);
        let log_likelihood = utilities[chosen_index] - log_normalizer;

        // Posterior ∝ prior × likelihood (in log space: addition)
        new_log_probs.push(log_prior + log_likelihood);
    }

    // Normalize the posterior
    let log_evidence = PreferenceDistribution::log_sum_exp(&new_log_probs);
    for lp in &mut new_log_probs {
        *lp -= log_evidence;
    }

    PreferenceDistribution {
        domain: prior.domain.clone(),
        profiles: prior.profiles.clone(),
        log_probs: new_log_probs,
    }
}

/// Predict the best item from a set of options given current beliefs.
/// Returns the index of the item with highest expected utility.
pub fn predict_best(dist: &PreferenceDistribution, options: &[Item]) -> usize {
    assert!(!options.is_empty(), "options must not be empty");

    let mut best_idx = 0;
    let mut best_expected = f64::NEG_INFINITY;

    for (item_idx, item) in options.iter().enumerate() {
        // E[utility(item)] = Σ_θ P(θ) × utility(θ, item)
        let expected: f64 = dist
            .profiles
            .iter()
            .zip(dist.log_probs.iter())
            .map(|(profile, &log_p)| {
                let p = log_p.exp();
                p * PreferenceDistribution::utility(profile, item)
            })
            .sum();

        if expected > best_expected {
            best_expected = expected;
            best_idx = item_idx;
        }
    }

    best_idx
}

/// Compute the entropy (uncertainty) of the distribution.
/// Returns bits of uncertainty. Lower = more confident.
pub fn entropy(dist: &PreferenceDistribution) -> f64 {
    -dist
        .log_probs
        .iter()
        .map(|&lp| {
            let p = lp.exp();
            if p > 0.0 { p * lp } else { 0.0 }
        })
        .sum::<f64>()
}

/// Compute the marginal probability for each preference level for each feature.
/// Useful for visualization and debugging.
pub fn marginal_preferences(
    dist: &PreferenceDistribution,
) -> HashMap<String, HashMap<FeaturePreference, f64>> {
    let mut marginals: HashMap<String, HashMap<FeaturePreference, f64>> = HashMap::new();

    for (feat_idx, feat_name) in dist.domain.features.iter().enumerate() {
        let mut feat_marginal: HashMap<FeaturePreference, f64> = HashMap::new();
        for pref in FeaturePreference::all() {
            feat_marginal.insert(*pref, 0.0);
        }

        for (profile, &log_p) in dist.profiles.iter().zip(dist.log_probs.iter()) {
            let p = log_p.exp();
            *feat_marginal.entry(profile[feat_idx]).or_insert(0.0) += p;
        }

        marginals.insert(feat_name.clone(), feat_marginal);
    }

    marginals
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn test_domain() -> DomainSchema {
        DomainSchema {
            name: "flights".to_string(),
            features: vec!["cost".to_string(), "duration".to_string()],
        }
    }

    #[test]
    fn test_uniform_prior_sums_to_one() {
        let prior = uniform_prior(test_domain());
        let total: f64 = prior.log_probs.iter().map(|lp| lp.exp()).sum();
        assert!(
            (total - 1.0).abs() < 1e-10,
            "Prior should sum to 1.0, got {}",
            total
        );
    }

    #[test]
    fn test_uniform_prior_profile_count() {
        let prior = uniform_prior(test_domain());
        // 2 features × 5 levels = 5^2 = 25 profiles
        assert_eq!(prior.profiles.len(), 25);
        assert_eq!(prior.log_probs.len(), 25);
    }

    #[test]
    fn test_bayesian_update_shifts_posterior() {
        let prior = uniform_prior(test_domain());
        let options = vec![
            Item {
                features: vec![0.9, 0.5],
            }, // expensive, medium
            Item {
                features: vec![0.1, 0.5],
            }, // cheap, medium
        ];

        // User picks the cheap option
        let posterior = bayesian_update(&prior, 1, &options, 1.0);
        let total: f64 = posterior.log_probs.iter().map(|lp| lp.exp()).sum();
        assert!(
            (total - 1.0).abs() < 1e-10,
            "Posterior should sum to 1.0, got {}",
            total
        );

        // Entropy should decrease (more certain)
        let prior_entropy = entropy(&prior);
        let post_entropy = entropy(&posterior);
        assert!(
            post_entropy < prior_entropy,
            "Posterior entropy ({}) should be less than prior entropy ({})",
            post_entropy,
            prior_entropy
        );
    }

    #[test]
    fn test_predict_best_matches_strong_preference() {
        let prior = uniform_prior(test_domain());

        // Simulate 3 rounds where user always picks low cost
        let mut dist = prior;
        for _ in 0..3 {
            let options = vec![
                Item {
                    features: vec![0.8, 0.5],
                }, // expensive
                Item {
                    features: vec![0.2, 0.5],
                }, // cheap
                Item {
                    features: vec![0.5, 0.5],
                }, // medium
            ];
            dist = bayesian_update(&dist, 1, &options, 2.0); // user picks cheap
        }

        // Now predict: should pick the cheap option
        let test_options = vec![
            Item {
                features: vec![0.9, 0.6],
            }, // expensive
            Item {
                features: vec![0.1, 0.4],
            }, // cheap
        ];

        let prediction = predict_best(&dist, &test_options);
        assert_eq!(
            prediction, 1,
            "Should predict the cheap option after consistent evidence"
        );
    }

    #[test]
    fn test_entropy_decreases_with_evidence() {
        let prior = uniform_prior(test_domain());
        let initial_entropy = entropy(&prior);

        let options = vec![
            Item {
                features: vec![0.9, 0.3],
            },
            Item {
                features: vec![0.1, 0.7],
            },
        ];

        let round1 = bayesian_update(&prior, 0, &options, 1.0);
        let entropy1 = entropy(&round1);

        let round2 = bayesian_update(&round1, 0, &options, 1.0);
        let entropy2 = entropy(&round2);

        assert!(
            entropy1 < initial_entropy,
            "Entropy after 1 round should decrease"
        );
        assert!(
            entropy2 < entropy1,
            "Entropy after 2 rounds should decrease further"
        );
    }

    #[test]
    fn test_numerical_stability_extreme_values() {
        let prior = uniform_prior(test_domain());
        let options = vec![
            Item {
                features: vec![0.0, 0.0],
            },
            Item {
                features: vec![1.0, 1.0],
            },
        ];

        // High temperature — should not produce NaN
        let posterior = bayesian_update(&prior, 0, &options, 100.0);
        for &lp in &posterior.log_probs {
            assert!(!lp.is_nan(), "Log-prob should not be NaN");
            assert!(
                !lp.is_infinite() || lp == f64::NEG_INFINITY,
                "Only -inf is acceptable"
            );
        }

        let total: f64 = posterior.log_probs.iter().map(|lp| lp.exp()).sum();
        assert!(
            (total - 1.0).abs() < 1e-6,
            "Even with extreme temperature, posterior should normalize"
        );
    }

    #[test]
    fn test_marginal_preferences() {
        let prior = uniform_prior(test_domain());
        let marginals = marginal_preferences(&prior);

        // With uniform prior, each level should have probability 1/5 = 0.2
        for feat in &prior.domain.features {
            let feat_marg = marginals.get(feat).unwrap();
            for pref in FeaturePreference::all() {
                let p = feat_marg[pref];
                assert!(
                    (p - 0.2).abs() < 1e-10,
                    "Uniform marginal for {:?} should be 0.2, got {}",
                    pref,
                    p
                );
            }
        }
    }

    #[test]
    fn test_serialization_roundtrip() {
        let prior = uniform_prior(test_domain());
        let json = serde_json::to_string(&prior).unwrap();
        let deserialized: PreferenceDistribution = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.profiles.len(), prior.profiles.len());
        assert_eq!(deserialized.log_probs.len(), prior.log_probs.len());
    }
}
