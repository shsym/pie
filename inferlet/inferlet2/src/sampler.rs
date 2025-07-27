use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rand::rngs::ThreadRng;

pub trait Sampler {
    /// Samples a token ID from a given sparse distribution of token IDs and their probabilities.
    ///
    /// # Arguments
    /// * `ids` - A slice of token IDs.
    /// * `probs` - A slice of corresponding probabilities for each token ID.
    fn sample(&mut self, ids: &[u32], probs: &[f32]) -> u32;
}

// --- Provided Greedy Sampler ---

pub struct GreedySampler {}

impl GreedySampler {
    pub fn new() -> Self {
        Self {}
    }
}

impl Sampler for GreedySampler {
    fn sample(&mut self, ids: &[u32], probs: &[f32]) -> u32 {
        let mut max_prob = f32::NEG_INFINITY;
        let mut max_prob_idx = 0;
        for (i, &prob) in probs.iter().enumerate() {
            if prob > max_prob {
                max_prob = prob;
                max_prob_idx = i;
            }
        }
        ids[max_prob_idx]
    }
}

/// Adjusts the randomness of the distribution. A higher temperature (e.g., > 1.0) makes the output
/// more random, while a lower temperature (e.g., < 1.0) makes it more deterministic. A temperature
/// of 0 is equivalent to greedy sampling.
pub struct TemperatureSampler {
    temperature: f32,
    rng: ThreadRng,
}

impl TemperatureSampler {
    pub fn new(temperature: f32) -> Self {
        assert!(temperature >= 0.0, "Temperature must be non-negative.");
        Self {
            temperature,
            rng: rand::thread_rng(),
        }
    }
}

impl Sampler for TemperatureSampler {
    fn sample(&mut self, ids: &[u32], probs: &[f32]) -> u32 {
        if self.temperature == 0.0 {
            return GreedySampler::new().sample(ids, probs);
        }

        // Scale probabilities with temperature: p_new = p^(1/T)
        // This is done by taking log, dividing by T, and taking exp.
        // Or more directly, p.powf(1.0 / T).
        let temp_inv = 1.0 / self.temperature;
        let scaled_probs: Vec<f32> = probs.iter().map(|p| p.powf(temp_inv)).collect();

        // Sample from the new distribution
        let dist =
            WeightedIndex::new(&scaled_probs).expect("Failed to create weighted distribution.");
        let chosen_idx = dist.sample(&mut self.rng);

        ids[chosen_idx]
    }
}

/// Filters the vocabulary to the `k` most likely next tokens and then samples from this reduced set.
/// This avoids picking very unlikely tokens.
pub struct TopKSampler {
    k: usize,
    rng: ThreadRng,
}

impl TopKSampler {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            rng: rand::thread_rng(),
        }
    }
}

impl Sampler for TopKSampler {
    fn sample(&mut self, ids: &[u32], probs: &[f32]) -> u32 {
        let mut candidates: Vec<(u32, f32)> =
            ids.iter().copied().zip(probs.iter().copied()).collect();

        // Sort candidates by probability in descending order
        candidates
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Truncate to the top k candidates, ensuring we don't go out of bounds
        let k = self.k.min(candidates.len()).max(1);
        let top_k_candidates = &candidates[..k];

        // Separate IDs and probabilities for weighted sampling
        let top_k_ids: Vec<u32> = top_k_candidates.iter().map(|(id, _)| *id).collect();
        let top_k_probs: Vec<f32> = top_k_candidates.iter().map(|(_, prob)| *prob).collect();

        // Perform weighted sampling
        let dist =
            WeightedIndex::new(&top_k_probs).expect("Failed to create weighted distribution.");
        let chosen_idx = dist.sample(&mut self.rng);

        top_k_ids[chosen_idx]
    }
}

/// Filters the vocabulary to the smallest set of tokens whose cumulative probability exceeds `p`.
/// This method is adaptive; the size of the set depends on the probability distribution.
pub struct TopPSampler {
    p: f32,
    rng: ThreadRng,
}

impl TopPSampler {
    pub fn new(p: f32) -> Self {
        assert!((0.0..=1.0).contains(&p), "p must be between 0.0 and 1.0.");
        Self {
            p,
            rng: rand::thread_rng(),
        }
    }
}

impl Sampler for TopPSampler {
    fn sample(&mut self, ids: &[u32], probs: &[f32]) -> u32 {
        let mut candidates: Vec<(u32, f32)> =
            ids.iter().copied().zip(probs.iter().copied()).collect();

        // Sort candidates by probability in descending order
        candidates
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Find the nucleus set by accumulating probabilities
        let mut cumulative_prob = 0.0;
        let mut nucleus_end_idx = candidates.len();
        for (i, &(_, prob)) in candidates.iter().enumerate() {
            cumulative_prob += prob;
            if cumulative_prob >= self.p {
                nucleus_end_idx = i + 1; // Include the current element
                break;
            }
        }

        // Ensure at least one token is in the nucleus
        let nucleus = &candidates[..nucleus_end_idx.max(1)];

        // Separate IDs and probabilities for weighted sampling
        let nucleus_ids: Vec<u32> = nucleus.iter().map(|(id, _)| *id).collect();
        let nucleus_probs: Vec<f32> = nucleus.iter().map(|(_, prob)| *prob).collect();

        // Perform weighted sampling
        let dist =
            WeightedIndex::new(&nucleus_probs).expect("Failed to create weighted distribution.");
        let chosen_idx = dist.sample(&mut self.rng);

        nucleus_ids[chosen_idx]
    }
}

/// Selects tokens from a set where the token's self-information is close to the entropy of the
/// entire distribution. It aims to filter out tokens that are "unlikely" yet have high probability
/// in long-tail distributions.
pub struct TypicalSampler {
    /// The cumulative probability mass to select from the 'typical' set, e.g., 0.95
    mass: f32,
    rng: ThreadRng,
}

impl TypicalSampler {
    pub fn new(mass: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&mass),
            "mass must be between 0.0 and 1.0."
        );
        Self {
            mass,
            rng: rand::thread_rng(),
        }
    }
}

impl Sampler for TypicalSampler {
    fn sample(&mut self, ids: &[u32], probs: &[f32]) -> u32 {
        if probs.is_empty() {
            return 0;
        } // Edge case

        // 1. Calculate entropy H(P) = -Σ p_i * log2(p_i)
        let entropy = -probs
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| p * p.log2())
            .sum::<f32>();

        // 2. Calculate the absolute difference between each token's self-information and the entropy
        let mut candidates: Vec<(u32, f32, f32)> = ids
            .iter()
            .copied()
            .zip(probs.iter().copied())
            .map(|(id, prob)| {
                let neg_log_prob = if prob > 0.0 {
                    -prob.log2()
                } else {
                    f32::INFINITY
                };
                let diff = (neg_log_prob - entropy).abs();
                (id, prob, diff) // (id, original_prob, entropy_diff)
            })
            .collect();

        // 3. Sort candidates by their difference to the entropy (ascending)
        candidates.sort_unstable_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

        // 4. Select the smallest set whose cumulative probability exceeds `mass`
        let mut cumulative_prob = 0.0;
        let mut typical_set_end_idx = candidates.len();
        for (i, &(_, prob, _)) in candidates.iter().enumerate() {
            cumulative_prob += prob;
            if cumulative_prob >= self.mass {
                typical_set_end_idx = i + 1;
                break;
            }
        }

        let typical_set = &candidates[..typical_set_end_idx.max(1)];

        // 5. Perform weighted sampling on the resulting typical set
        let typical_ids: Vec<u32> = typical_set.iter().map(|(id, _, _)| *id).collect();
        let typical_probs: Vec<f32> = typical_set.iter().map(|(_, prob, _)| *prob).collect();

        if typical_ids.is_empty() {
            return GreedySampler::new().sample(ids, probs);
        }

        let dist = WeightedIndex::new(&typical_probs).unwrap();
        let chosen_idx = dist.sample(&mut self.rng);

        typical_ids[chosen_idx]
    }
}
