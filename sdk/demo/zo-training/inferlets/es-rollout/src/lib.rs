//! Perform ES rollouts — generate text with a perturbed adapter.
//!
//! Accepts a list of rollout tasks (each with a uid, problem, and ZO seed),
//! generates text for each using the adapter with the given seed perturbation,
//! and returns the generated texts as a JSON array.

use inferlet::{
    Context,
    adapter::Adapter,
    inference::Sampler,
    model::Model,
    runtime,
    Result,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    /// Adapter name.
    name: String,
    /// System prompt for generation.
    system_prompt: String,
    /// Maximum tokens to generate per rollout.
    max_num_outputs: usize,
    /// Rollout tasks.
    rollouts: Vec<Rollout>,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct Rollout {
    uid: String,
    task: String,
    seed: i64,
}

#[inferlet::main]
async fn main(input: Input) -> Result<Vec<String>> {
    let model_name = runtime::models().into_iter().next()
        .ok_or("No models available")?;
    let model = Model::load(&model_name)?;

    let adapter = Adapter::open(&model, &input.name)
        .ok_or_else(|| format!("Adapter '{}' not found", input.name))?;

    let sampler = Sampler::TopP((0.6, 0.95));
    let mut results: Vec<String> = Vec::with_capacity(input.rollouts.len());

    for rollout in &input.rollouts {
        let mut ctx = Context::new(&model)?;
        ctx.system(&input.system_prompt);
        ctx.user(&rollout.task);
        ctx.cue();

        let text = ctx.generate(sampler.clone())
            .with_adapter(&adapter)
            .with_zo_seed(rollout.seed)
            .with_max_tokens(input.max_num_outputs)
            .collect_text()
            .await?;

        results.push(text);
    }

    Ok(results)
}
