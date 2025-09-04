use inferlet::sampler::Sampler;
use pico_args::Arguments;
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rand::rngs::ThreadRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::ffi::OsString;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::time::Instant;
use inferlet::traits::Tokenize;

/// Injects a watermark by partitioning the vocabulary into a "green list" and a "red list"
/// based on the hash of the previous token. It then boosts the probabilities of tokens
/// in the green list.
pub struct WatermarkSampler {
    /// The proportion of the vocabulary to be included in the green list (e.g., 0.5 for 50%).
    gamma: f32,
    /// The bias added to the logits of the green-listed tokens to increase their probability.
    delta: f32,
    /// The previously generated token ID, used to seed the green/red list generation.
    /// `None` indicates the start of a sequence.
    previous_token: Option<u32>,
    /// Random number generator for sampling.
    rng: ThreadRng,
}

impl WatermarkSampler {
    /// Creates a new `WatermarkSampler`.
    ///
    /// # Arguments
    /// * `gamma` - The proportion of the vocabulary to be in the green list (0.0 to 1.0).
    /// * `delta` - The bias to add to the logits of green-listed tokens.
    pub fn new(gamma: f32, delta: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&gamma),
            "gamma must be between 0.0 and 1.0."
        );
        Self {
            gamma,
            delta,
            previous_token: None, // No previous token at the beginning
            rng: rand::rng(),
        }
    }

    /// Hashes the previous token to create a seed for the RNG.
    /// This ensures the green/red list is deterministic based on the context.
    fn get_seed(&self) -> u64 {
        match self.previous_token {
            Some(token) => {
                let mut hasher = DefaultHasher::new();
                token.hash(&mut hasher);
                hasher.finish()
            }
            // Use a default seed for the very first token in a sequence.
            None => 0,
        }
    }
}

impl Sampler for WatermarkSampler {
    fn sample(&mut self, ids: &[u32], probs: &[f32]) -> u32 {
        if ids.is_empty() {
            // Handle the edge case of an empty input.
            // This could be an error or a special token like EOS.
            // Here, we'll arbitrarily return 0.
            self.previous_token = Some(0);
            return 0;
        }

        let seed = self.get_seed();
        let mut seeded_rng = rand::rngs::StdRng::seed_from_u64(seed);

        // Create a new set of probabilities by applying the watermark bias.
        let mut watermarked_probs = probs.to_vec();

        // The number of tokens to be considered "green" from the given `ids`.
        let green_list_size = (ids.len() as f32 * self.gamma).round() as usize;

        // Generate green list indices based on the provided `ids`.
        // We do this by creating a shuffled list of indices from 0 to ids.len() - 1
        // and taking the first `green_list_size` elements.
        let mut indices: Vec<usize> = (0..ids.len()).collect();
        indices.shuffle(&mut seeded_rng);

        let green_indices: std::collections::HashSet<usize> =
            indices.into_iter().take(green_list_size).collect();

        // Apply the bias `delta` to the log-probabilities of the green-listed tokens.
        // Since we have probabilities, we convert to logits, add delta, then convert back.
        // logit = log(p)
        // new_logit = log(p) + delta
        // new_p = exp(log(p) + delta) = p * exp(delta)
        let exp_delta = self.delta.exp();
        for i in 0..watermarked_probs.len() {
            if green_indices.contains(&i) {
                watermarked_probs[i] *= exp_delta;
            }
        }

        // Normalize the probabilities so they sum to 1 again.
        let prob_sum: f32 = watermarked_probs.iter().sum();
        if prob_sum > 0.0 {
            for p in &mut watermarked_probs {
                *p /= prob_sum;
            }
        }

        // Sample from the new, watermarked distribution.
        let dist = WeightedIndex::new(&watermarked_probs)
            .expect("Failed to create watermarked distribution.");
        let chosen_idx = dist.sample(&mut self.rng);

        let chosen_id = ids[chosen_idx];

        // Update the previous token for the next sampling step.
        self.previous_token = Some(chosen_id);

        chosen_id
    }
}
#[inferlet::main]
async fn main() -> Result<(), String> {
    // 1. Get arguments from the inferlet environment and prepare the parser.
    let mut args = Arguments::from_vec(
        inferlet::get_arguments()
            .into_iter()
            .map(OsString::from)
            .collect(),
    );

    // 3. Parse arguments, falling back to defaults if they are not provided.
    let prompt = args
        .opt_value_from_str(["-p", "--prompt"])
        .map_err(|e| e.to_string())?
        .unwrap_or_else(|| "Explain the LLM decoding process ELI5.".to_string());

    let max_num_outputs: u32 = args
        .opt_value_from_str(["-n", "--max-tokens"])
        .map_err(|e| e.to_string())?
        .unwrap_or(256);

    // Ensure no unknown arguments were passed.
    let remaining = args.finish();
    if !remaining.is_empty() {
        return Err(format!(
            "Unknown arguments found: {:?}. Use --help for usage.",
            remaining
        ));
    }

    // --- Main logic starts here ---
    let start = Instant::now();

    let model = inferlet::get_auto_model();
    let tokenizer = model.get_tokenizer();
    let mut sampler = WatermarkSampler::new(0.5, 0.0);
    let mut stop_condition = inferlet::stop_condition::any(
        inferlet::stop_condition::Until::new(tokenizer.tokenize("<|eot_id|>")),
        inferlet::stop_condition::Length::new(max_num_outputs as usize),
    );

    let mut ctx = model.create_context();

    // 4. Use the parsed prompt and max_num_outputs.
    ctx.fill("<|begin_of_text|>");
    ctx.fill("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|>");
    ctx.fill(&format!(
        "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>",
        prompt
    ));
    ctx.fill("<|start_header_id|>assistant<|end_header_id|>\n\n");

    let text = ctx.generate(&mut sampler, &mut stop_condition).await;
    let token_ids = tokenizer.tokenize(&text);
    println!("Output: {:?} (total elapsed: {:?})", text, start.elapsed());

    // Compute per-token latency, avoiding division by zero.
    if !token_ids.is_empty() {
        println!(
            "Per token latency: {:?}",
            start.elapsed() / (token_ids.len() as u32)
        );
    }

    Ok(())
}
