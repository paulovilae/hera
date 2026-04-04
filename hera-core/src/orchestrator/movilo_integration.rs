//! Movilo Integration — Bayesian Provider Recommendation
//!
//! This module connects Movilo's provider search with the Bayesian preference
//! engine. Given a user's prior distribution and a list of available providers,
//! it calculates a personalized score for each provider and ranks them.
//!
//! # Implicit Feedback (Audio Suggestion)
//! Instead of relying on explicit "Like" / "Dislike" buttons in Movilo, we
//! deduce feedback implicitly:
//! - If a user clicks a provider and stays on their profile / initiates contact -> Strong Positive
//! - If a user clicks but returns to search in < 5 seconds -> Weak Negative
//! - If a user ignores a recommended provider and scrolls past -> Weak Negative
//! - If a user immediately refines their search filters -> Negative

use super::preference_model::{Item, PreferenceDistribution};
use serde::{Deserialize, Serialize};

#[cfg(test)]
use super::simulator::movilo_provider_domain;

/// Represents a Movilo Provider as received from the DB/API.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MoviloProvider {
    pub id: String,
    pub company_name: String,
    pub provider_type: String,
    pub city: String,
    // Typical features
    pub rating: f64,           // 1.0 to 5.0
    pub min_price: f64,        // In COP (e.g. 50000.0)
    pub distance_km: f64,      // e.g. 2.5
    pub wait_days: f64,        // e.g. 1.0
    pub years_experience: f64, // e.g. 10.0
}

/// A ranked recommendation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankedProvider {
    pub provider: MoviloProvider,
    pub bayesian_score: f64,
}

impl MoviloProvider {
    /// Convert this real-world provider into a Bayesian `Item` using the
    /// shared Movilo domain schema.
    ///
    /// The Movilo domain features are (in order):
    /// 0: cost (normalized low=good)
    /// 1: rating (normalized high=good)
    /// 2: distance (normalized low=good)
    /// 3: wait_time (normalized low=good)
    /// 4: experience (normalized high=good)
    pub fn to_bayesian_item(&self) -> Item {
        // Normalization bounds (heuristics based on typical Movilo data)
        let max_price = 300_000.0;
        let max_dist = 20.0;
        let max_wait = 30.0;
        let max_exp = 30.0;

        // Normalize features to [0.0, 1.0] where 1.0 is "highest manifestation of the trait"
        // Note: For cost/distance/wait, a *lower* actual value might be preferred,
        // but the feature value itself just represents magnitude. The PreferenceDistribution
        // will learn `StrongLow` if the user prefers cheap/close providers.
        let f_cost = (self.min_price / max_price).clamp(0.0, 1.0);
        let f_rating = (self.rating / 5.0).clamp(0.0, 1.0);
        let f_dist = (self.distance_km / max_dist).clamp(0.0, 1.0);
        let f_wait = (self.wait_days / max_wait).clamp(0.0, 1.0);
        let f_exp = (self.years_experience / max_exp).clamp(0.0, 1.0);

        Item {
            features: vec![f_cost, f_rating, f_dist, f_wait, f_exp],
        }
    }
}

/// Rank a list of Movilo providers for a specific user based on their Bayesian prior.
pub fn rank_providers(
    providers: &[MoviloProvider],
    prior: &PreferenceDistribution,
) -> Vec<RankedProvider> {
    if providers.is_empty() {
        return vec![];
    }

    let items: Vec<Item> = providers.iter().map(|p| p.to_bayesian_item()).collect();

    // Calculate expected utility for each item
    let mut ranked: Vec<RankedProvider> = providers
        .iter()
        .zip(items.iter())
        .map(|(p, item)| {
            let expected_utility: f64 = prior
                .profiles
                .iter()
                .zip(prior.log_probs.iter())
                .map(|(profile, &log_p)| {
                    let p_prob = log_p.exp();
                    p_prob * PreferenceDistribution::utility(profile, item)
                })
                .sum();

            RankedProvider {
                provider: p.clone(),
                bayesian_score: expected_utility,
            }
        })
        .collect();

    // Sort descending by expected utility
    ranked.sort_by(|a, b| {
        b.bayesian_score
            .partial_cmp(&a.bayesian_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    ranked
}

// ═══════════════════════════════════════════════════════════════════
// Implicit Feedback Logic
// ═══════════════════════════════════════════════════════════════════

/// Implicit feedback events tracked by the Movilo frontend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MoviloImplicitAction {
    /// User clicked provider and stayed for > 15 seconds, or booked
    DwellLongOrBook,
    /// User clicked provider but returned to search in < 3 seconds
    DwellShortBounce,
    /// User immediately changed filters/search terms after seeing results
    ImmediateSearchRefine,
    /// User viewed the list but took no action
    NoAction,
}

impl MoviloImplicitAction {
    /// Convert implicit action to Bayesian Feedback score (0.0 to 1.0)
    /// Like = 1.0 (Strong Positive)
    /// Dislike = 0.0 (Strong Negative)
    pub fn to_feedback_score(&self) -> f64 {
        match self {
            Self::DwellLongOrBook => 1.0,       // Strong Like
            Self::DwellShortBounce => 0.1,      // Strong Dislike (misleading info)
            Self::ImmediateSearchRefine => 0.2, // Dislike (results irrelevant)
            Self::NoAction => 0.4,              // Mild Dislike (didn't catch eye)
        }
    }

    /// Convert to a discrete choice index [0 = Dislike, 1 = Like]
    pub fn to_choice_index(&self) -> usize {
        match self {
            Self::DwellLongOrBook => 1,
            _ => 0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orchestrator::preference_model::{FeaturePreference, uniform_prior};

    fn sample_providers() -> Vec<MoviloProvider> {
        vec![
            MoviloProvider {
                id: "prov-1".to_string(), // Expensive, very experienced
                company_name: "Premium Dental".to_string(),
                provider_type: "Odontólogo".to_string(),
                city: "Bogotá".to_string(),
                rating: 4.9,
                min_price: 250_000.0,
                distance_km: 15.0,
                wait_days: 5.0,
                years_experience: 25.0,
            },
            MoviloProvider {
                id: "prov-2".to_string(), // Cheap, close, less experienced
                company_name: "Barrio Dental".to_string(),
                provider_type: "Odontólogo".to_string(),
                city: "Bogotá".to_string(),
                rating: 4.1,
                min_price: 45_000.0,
                distance_km: 1.5,
                wait_days: 1.0,
                years_experience: 3.0,
            },
            MoviloProvider {
                id: "prov-3".to_string(), // Middle ground
                company_name: "Centro Sonrisas".to_string(),
                provider_type: "Odontólogo".to_string(),
                city: "Bogotá".to_string(),
                rating: 4.5,
                min_price: 120_000.0,
                distance_km: 5.0,
                wait_days: 2.0,
                years_experience: 10.0,
            },
        ]
    }

    #[test]
    fn test_conversion_to_bayesian_item() {
        let provider = &sample_providers()[0];
        let item = provider.to_bayesian_item();

        assert_eq!(item.features.len(), 5);
        // Cost: 250k / 300k = ~0.833
        assert!((item.features[0] - 0.833).abs() < 0.01);
        // Exper: 25 / 30 = ~0.833
        assert!((item.features[4] - 0.833).abs() < 0.01);
    }

    #[test]
    fn test_rank_providers_with_uniform_prior() {
        let providers = sample_providers();
        let domain = movilo_provider_domain();
        let prior = uniform_prior(domain);

        let ranked = rank_providers(&providers, &prior);
        assert_eq!(ranked.len(), 3);
        // With uniform prior, scores should be relatively close to each other
        assert!((ranked[0].bayesian_score - ranked[2].bayesian_score).abs() < 2.0);
    }

    #[test]
    fn test_rank_providers_prefers_cheap() {
        let providers = sample_providers();
        let domain = movilo_provider_domain();
        let mut prior = uniform_prior(domain);

        // Force user to strongly prefer LOW cost
        // Cost is index 0 in movilo domain
        for (i, profile) in prior.profiles.iter().enumerate() {
            if profile[0] == FeaturePreference::StrongLow {
                prior.log_probs[i] += 10.0; // Boost probability in log space
            } else {
                prior.log_probs[i] -= 10.0;
            }
        }

        // Normalize log_probs
        let max_lp = prior
            .log_probs
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let log_sum_exp = max_lp
            + prior
                .log_probs
                .iter()
                .map(|&v| (v - max_lp).exp())
                .sum::<f64>()
                .ln();
        for lp in &mut prior.log_probs {
            *lp -= log_sum_exp;
        }

        let ranked = rank_providers(&providers, &prior);

        // The cheapest provider (prov-2) should be ranked first
        assert_eq!(ranked[0].provider.id, "prov-2");
        assert_eq!(ranked[0].provider.company_name, "Barrio Dental");

        // The most expensive (prov-1) should be last
        assert_eq!(ranked[2].provider.id, "prov-1");
    }

    #[test]
    fn test_implicit_feedback_to_scores() {
        assert_eq!(
            MoviloImplicitAction::DwellLongOrBook.to_feedback_score(),
            1.0
        );
        assert_eq!(
            MoviloImplicitAction::DwellShortBounce.to_feedback_score(),
            0.1
        );

        assert_eq!(MoviloImplicitAction::DwellLongOrBook.to_choice_index(), 1);
        assert_eq!(
            MoviloImplicitAction::ImmediateSearchRefine.to_choice_index(),
            0
        );
    }
}
