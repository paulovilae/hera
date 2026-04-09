//! Training Data Generator — JSONL fine-tuning dataset builder
//!
//! Converts interaction traces from the simulator into (prompt, completion)
//! pairs suitable for LoRA fine-tuning of LLMs. The format teaches the LLM
//! to reason like a Bayesian assistant.
//!
//! Output format: JSONL where each line is a training example with:
//! - `prompt`: the context (user history + current options)
//! - `completion`: the Bayesian assistant's reasoning + recommendation

use super::preference_model::Item;
use super::simulator::InteractionTrace;
use serde::{Deserialize, Serialize};
use std::io::Write;

// ═══════════════════════════════════════════════════════════════════
// Training Example
// ═══════════════════════════════════════════════════════════════════

/// A single training example for fine-tuning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    /// System prompt describing the assistant's role
    pub system: String,
    /// User prompt with context and current options
    pub prompt: String,
    /// Target completion showing Bayesian reasoning
    pub completion: String,
    /// Metadata for filtering/analysis
    pub metadata: ExampleMetadata,
}

/// Metadata about a training example.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExampleMetadata {
    pub domain: String,
    pub user_id: String,
    pub round: usize,
    pub entropy_before: f64,
    pub entropy_after: f64,
    pub correct: bool,
}

// ═══════════════════════════════════════════════════════════════════
// Dataset Generation
// ═══════════════════════════════════════════════════════════════════

/// Format an item's features as a readable string.
fn format_item(item: &Item, feature_names: &[String], idx: usize) -> String {
    let attrs: Vec<String> = feature_names
        .iter()
        .zip(item.features.iter())
        .map(|(name, val)| format!("{}={:.2}", name, val))
        .collect();
    format!("  Option {}: {}", idx + 1, attrs.join(", "))
}

/// Build the system prompt describing the Bayesian assistant.
fn system_prompt(domain: &str) -> String {
    format!(
        "You are a Bayesian recommendation assistant for the {} domain. \
         You maintain a probability distribution over user preferences and update \
         it using Bayes' rule after each user choice. Analyze the user's history, \
         reason about their likely preferences, and recommend the best option. \
         Show your reasoning about preference probabilities.",
        domain
    )
}

/// Build a user prompt for a specific round.
fn build_prompt(trace: &InteractionTrace, round_idx: usize, feature_names: &[String]) -> String {
    let mut prompt = String::new();

    // Include history of previous rounds
    if round_idx > 0 {
        prompt.push_str("Previous interactions:\n");
        for r in &trace.rounds[..round_idx] {
            let opts: Vec<String> = r
                .options
                .iter()
                .enumerate()
                .map(|(i, item)| format_item(item, feature_names, i))
                .collect();
            prompt.push_str(&format!(
                "Round {}: Presented:\n{}\n  User chose: Option {}\n\n",
                r.round + 1,
                opts.join("\n"),
                r.user_choice + 1,
            ));
        }
    }

    // Current round options
    let current = &trace.rounds[round_idx];
    let current_opts: Vec<String> = current
        .options
        .iter()
        .enumerate()
        .map(|(i, item)| format_item(item, feature_names, i))
        .collect();

    prompt.push_str(&format!(
        "Current options (Round {}):\n{}\n\nWhich option should I recommend?",
        round_idx + 1,
        current_opts.join("\n"),
    ));

    prompt
}

/// Build the completion showing Bayesian reasoning.
fn build_completion(
    trace: &InteractionTrace,
    round_idx: usize,
    feature_names: &[String],
) -> String {
    let round = &trace.rounds[round_idx];
    let mut completion = String::new();

    // Show reasoning about preferences
    if round_idx == 0 {
        completion.push_str("Since this is the first interaction, I start with a uniform prior ");
        completion.push_str("over all possible preference profiles.\n\n");
    } else {
        completion.push_str("Based on the user's previous choices, I update my beliefs:\n");

        // Show top marginal preferences
        for (feat, prefs) in &round.marginals_after {
            let mut sorted_prefs: Vec<(&String, &f64)> = prefs.iter().collect();
            sorted_prefs.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
            if let Some(&(ref top_pref, top_prob)) = sorted_prefs.first()
                && *top_prob > 0.3 {
                    completion.push_str(&format!(
                        "- {}: likely {} ({:.0}%)\n",
                        feat,
                        top_pref,
                        top_prob * 100.0
                    ));
                }
        }
        completion.push('\n');
    }

    // Show confidence
    completion.push_str(&format!(
        "Current uncertainty (entropy): {:.2} bits\n\n",
        round.entropy_before
    ));

    // Recommendation with justification
    let rec_item = &round.options[round.assistant_pick];
    let attrs: Vec<String> = feature_names
        .iter()
        .zip(rec_item.features.iter())
        .map(|(name, val)| format!("{}={:.2}", name, val))
        .collect();

    completion.push_str(&format!(
        "I recommend **Option {}** ({}) as it best matches the estimated preference profile.",
        round.assistant_pick + 1,
        attrs.join(", ")
    ));

    completion
}

/// Convert a list of interaction traces into training examples.
pub fn traces_to_training_data(
    traces: &[InteractionTrace],
    feature_names: &[String],
) -> Vec<TrainingExample> {
    let mut examples = Vec::new();

    for trace in traces {
        let sys = system_prompt(&trace.domain);

        for round_idx in 0..trace.rounds.len() {
            let round = &trace.rounds[round_idx];
            let prompt = build_prompt(trace, round_idx, feature_names);
            let completion = build_completion(trace, round_idx, feature_names);

            examples.push(TrainingExample {
                system: sys.clone(),
                prompt,
                completion,
                metadata: ExampleMetadata {
                    domain: trace.domain.clone(),
                    user_id: trace.user_id.clone(),
                    round: round_idx,
                    entropy_before: round.entropy_before,
                    entropy_after: round.entropy_after,
                    correct: round.correct,
                },
            });
        }
    }

    examples
}

/// Write training examples as JSONL to a file.
pub fn write_jsonl(examples: &[TrainingExample], path: &str) -> std::io::Result<usize> {
    let mut file = std::fs::File::create(path)?;
    let mut count = 0;

    for example in examples {
        let line = serde_json::to_string(example)
            .map_err(|e| std::io::Error::other(e))?;
        writeln!(file, "{}", line)?;
        count += 1;
    }

    Ok(count)
}

/// Write training examples in ChatML/conversation format for llama.cpp fine-tuning.
pub fn write_chatml(examples: &[TrainingExample], path: &str) -> std::io::Result<usize> {
    let mut file = std::fs::File::create(path)?;
    let mut count = 0;

    for example in examples {
        let conversation = serde_json::json!({
            "conversations": [
                {"role": "system", "content": example.system},
                {"role": "user", "content": example.prompt},
                {"role": "assistant", "content": example.completion},
            ]
        });
        let line = serde_json::to_string(&conversation)
            .map_err(|e| std::io::Error::other(e))?;
        writeln!(file, "{}", line)?;
        count += 1;
    }

    Ok(count)
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::super::simulator::{self, SimulationConfig, flight_domain};
    use super::*;

    fn sample_traces() -> Vec<InteractionTrace> {
        let config = SimulationConfig {
            domain: flight_domain(),
            n_users: 3,
            n_rounds: 3,
            n_options: 3,
            temperature: 1.0,
            base_seed: 42,
        };
        simulator::run_simulation(&config)
    }

    #[test]
    fn test_training_data_count() {
        let traces = sample_traces();
        let features = vec![
            "cost".to_string(),
            "duration".to_string(),
            "stops".to_string(),
        ];
        let examples = traces_to_training_data(&traces, &features);

        // 3 users × 3 rounds = 9 examples
        assert_eq!(examples.len(), 9);
    }

    #[test]
    fn test_training_example_structure() {
        let traces = sample_traces();
        let features = vec![
            "cost".to_string(),
            "duration".to_string(),
            "stops".to_string(),
        ];
        let examples = traces_to_training_data(&traces, &features);

        for ex in &examples {
            assert!(!ex.system.is_empty(), "System prompt should not be empty");
            assert!(!ex.prompt.is_empty(), "User prompt should not be empty");
            assert!(!ex.completion.is_empty(), "Completion should not be empty");
            assert!(
                ex.completion.contains("recommend"),
                "Completion should contain recommendation"
            );
        }
    }

    #[test]
    fn test_later_rounds_include_history() {
        let traces = sample_traces();
        let features = vec![
            "cost".to_string(),
            "duration".to_string(),
            "stops".to_string(),
        ];
        let examples = traces_to_training_data(&traces, &features);

        // Round 0 should NOT have "Previous interactions"
        let round0 = &examples[0];
        assert!(!round0.prompt.contains("Previous interactions"));

        // Round 1+ SHOULD have "Previous interactions"
        let round1 = &examples[1];
        assert!(round1.prompt.contains("Previous interactions"));
    }

    #[test]
    fn test_write_jsonl_to_file() {
        let traces = sample_traces();
        let features = vec![
            "cost".to_string(),
            "duration".to_string(),
            "stops".to_string(),
        ];
        let examples = traces_to_training_data(&traces, &features);

        let path = "/tmp/test_bayesian_training.jsonl";
        let count = write_jsonl(&examples, path).unwrap();
        assert_eq!(count, 9);

        // Verify file content
        let content = std::fs::read_to_string(path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 9);

        // Each line should be valid JSON
        for line in &lines {
            let parsed: serde_json::Value = serde_json::from_str(line).unwrap();
            assert!(parsed["system"].is_string());
            assert!(parsed["prompt"].is_string());
            assert!(parsed["completion"].is_string());
        }

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_write_chatml_format() {
        let traces = sample_traces();
        let features = vec![
            "cost".to_string(),
            "duration".to_string(),
            "stops".to_string(),
        ];
        let examples = traces_to_training_data(&traces, &features);

        let path = "/tmp/test_bayesian_chatml.jsonl";
        let count = write_chatml(&examples, path).unwrap();
        assert_eq!(count, 9);

        let content = std::fs::read_to_string(path).unwrap();
        for line in content.lines() {
            let parsed: serde_json::Value = serde_json::from_str(line).unwrap();
            let convos = parsed["conversations"].as_array().unwrap();
            assert_eq!(convos.len(), 3);
            assert_eq!(convos[0]["role"], "system");
            assert_eq!(convos[1]["role"], "user");
            assert_eq!(convos[2]["role"], "assistant");
        }

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_metadata_is_populated() {
        let traces = sample_traces();
        let features = vec![
            "cost".to_string(),
            "duration".to_string(),
            "stops".to_string(),
        ];
        let examples = traces_to_training_data(&traces, &features);

        for ex in &examples {
            assert!(!ex.metadata.domain.is_empty());
            assert!(!ex.metadata.user_id.is_empty());
            assert!(ex.metadata.entropy_before > 0.0);
        }
    }
}
