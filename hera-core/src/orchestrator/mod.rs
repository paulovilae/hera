//! Orchestrator — Intelligent reasoning and decision coordination layer.
//!
//! Contains the Bayesian preference engine for learning user preferences
//! through multi-round interactions, as described in Google Research's
//! "Bayesian teaching enables probabilistic reasoning in LLMs" (March 2026).

pub mod preference_model;
pub mod bayesian_assistant;
pub mod benchmark;
pub mod simulator;
pub mod training_data;
pub mod fine_tune;
pub mod feedback;
// phase 3: cross-app integration
pub mod movilo_integration;
