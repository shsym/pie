pub trait Sampler {
    fn sample(&mut self, token_ids: &[u32], logits: &[f32]) -> u32;
}

pub struct GreedySampler {}

impl GreedySampler {
    pub fn new() -> Self {
        Self {}
    }
}

impl Sampler for GreedySampler {
    fn sample(&mut self, token_ids: &[u32], logits: &[f32]) -> u32 {
        // find the token with the highest logit
        let mut max_logit = f32::NEG_INFINITY;
        let mut max_logit_idx = 0;
        for (i, &logit) in logits.iter().enumerate() {
            if logit > max_logit {
                max_logit = logit;
                max_logit_idx = i;
            }
        }
        token_ids[max_logit_idx]
    }
}
