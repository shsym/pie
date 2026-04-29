//! Demonstrates CacheBack speculative decoding — using a cached draft model.
//!
//! This inferlet implements speculative decoding with a manual generation loop.
//! A separate "drafter" context generates candidate tokens, which the main
//! context verifies in a single forward pass. Accepted tokens are committed;
//! rejected tokens are rolled back.
//!
//! Draft tokens are sent as regular `input_tokens` (not speculative_tokens)
//! so the runtime's working_page_tokens tracking remains consistent.

use inferlet::{
    Context, model::Model, runtime,
    ForwardPassExt, Result,
    inference::{ForwardPass, Output, Sampler},
};
use serde::Deserialize;
use std::time::Instant;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_draft_length")]
    draft_length: usize,
}

fn default_prompt() -> String { "Explain quantum computing.".to_string() }
fn default_max_tokens() -> usize { 256 }
fn default_draft_length() -> usize { 4 }

/// Simple greedy drafter using an independent context.
struct GreedyDrafter {
    model: Model,
    draft_ctx: Context,
    page_size: u32,
}

impl GreedyDrafter {
    fn new(model: &Model) -> Result<Self> {
        let draft_ctx = Context::new(model)?;
        let page_size = draft_ctx.page_size();
        Ok(Self {
            model: Model::load(&runtime::models()[0])?,
            draft_ctx,
            page_size,
        })
    }

    /// Generate `draft_length` tokens greedily using the draft context.
    fn draft(&self, seed_token: u32, draft_length: usize) -> Vec<u32> {
        let mut tokens = Vec::new();
        let mut current = seed_token;

        for _ in 0..draft_length {
            let wpt = self.draft_ctx.inner().working_page_token_count();
            let seq_len = self.draft_ctx.inner().committed_page_count() * self.page_size + wpt;

            // Reserve pages
            let current_working = self.draft_ctx.inner().working_page_count();
            let total_after = wpt + 1;
            let pages_needed = (total_after + self.page_size - 1) / self.page_size;
            let additional = pages_needed.saturating_sub(current_working);
            if additional > 0 {
                if self.draft_ctx.inner().reserve_working_pages(additional).is_err() {
                    break;
                }
            }

            let pass = ForwardPass::new(&self.model);
            pass.context(self.draft_ctx.inner());
            pass.input_tokens(&[current], &[seq_len]);
            pass.sampler(&[0], &Sampler::ARGMAX);

            let output = pass.execute();
            let Ok(future_output) = output else { break };
            let Some(result) = future_output.get() else { break };

            match result.first_token() {
                Some(t) => {
                    current = t;
                    tokens.push(current);
                }
                None => break,
            }
        }

        tokens
    }

    /// Roll back the last `n` tokens from the draft context by truncating
    /// working page tokens and releasing working pages if needed.
    fn rollback(&self, n: u32) {
        if n > 0 {
            self.draft_ctx.inner().truncate_working_page_tokens(n);
        }
    }
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let prompt = input.prompt;
    let max_tokens = input.max_tokens;
    let draft_length = input.draft_length;

    let start = Instant::now();
    let models = runtime::models();
    let model = Model::load(models.first().ok_or("No models available")?)?;
    let tokenizer = model.tokenizer();
    let stop_tokens = inferlet::instruct::chat::stop_tokens(&model);
    let page_size = {
        let tmp = Context::new(&model)?;
        tmp.page_size()
    };

    let mut ctx = Context::new(&model)?;

    // Fill prompt and flush to populate the KV cache.
    ctx.system("You are a helpful assistant.")
        .user(&prompt)
        .cue();
    ctx.flush().await?;

    // Bootstrap: sample the first token.
    let first_token = {
        let wpt = ctx.inner().working_page_token_count();
        let seq_len = ctx.inner().committed_page_count() * page_size + wpt;

        let cue_tokens = inferlet::instruct::chat::cue(&model);
        let trigger = *cue_tokens.last().unwrap_or(&0);

        let current_working = ctx.inner().working_page_count();
        let total_after = wpt + 1;
        let pages_needed = (total_after + page_size - 1) / page_size;
        let additional = pages_needed.saturating_sub(current_working);
        if additional > 0 {
            ctx.inner().reserve_working_pages(additional)
                .map_err(|e| format!("Failed to reserve pages: {}", e))?;
        }

        let pass = ForwardPass::new(&model);
        pass.context(ctx.inner());
        pass.input_tokens(&[trigger], &[seq_len]);
        pass.sampler(&[0], &Sampler::ARGMAX);
        let output = pass.execute_async().await
            .map_err(|e| format!("Bootstrap failed: {}", e))?;
        output.first_token().ok_or("Bootstrap produced no token".to_string())?
    };

    // Create the drafter.
    let drafter = GreedyDrafter::new(&model)?;

    let mut all_generated: Vec<u32> = vec![first_token];
    let mut anchor = first_token;
    let mut total_accepted = 1usize;
    let mut total_steps = 0usize;

    // Main speculative decoding loop.
    while total_accepted < max_tokens {
        // Step 1: Draft tokens using the separate drafter context.
        let draft_tokens = drafter.draft(anchor, draft_length);
        if draft_tokens.is_empty() {
            // Drafter failed; fall back to normal decode with anchor only.
            break;
        }

        // Step 2: Build verification pass on the main context.
        // Send [anchor] + [draft_tokens] as regular input_tokens.
        let mut verify_input = vec![anchor];
        verify_input.extend_from_slice(&draft_tokens);
        let input_count = verify_input.len();

        let wpt = ctx.inner().working_page_token_count();
        let seq_len = ctx.inner().committed_page_count() * page_size + wpt;

        // Reserve pages for all input tokens.
        let current_working = ctx.inner().working_page_count();
        let total_after = wpt + input_count as u32;
        let pages_needed = (total_after + page_size - 1) / page_size;
        let additional = pages_needed.saturating_sub(current_working);
        if additional > 0 {
            ctx.inner().reserve_working_pages(additional)
                .map_err(|e| format!("Failed to reserve pages: {}", e))?;
        }

        let pass = ForwardPass::new(&model);
        pass.context(ctx.inner());

        let positions: Vec<u32> = (seq_len..seq_len + input_count as u32).collect();
        pass.input_tokens(&verify_input, &positions);

        // Sample at all positions to verify drafts.
        let sample_indices: Vec<u32> = (0..input_count as u32).collect();
        pass.sampler(&sample_indices, &Sampler::ARGMAX);

        let output = pass.execute_async().await
            .map_err(|e| format!("Verification forward pass failed: {}", e))?;

        let verified: Vec<u32> = output.tokens().collect();

        if verified.is_empty() {
            break;
        }

        // Step 3: Determine how many draft tokens match the verifier.
        // verified[0] = model prediction after anchor position
        // verified[i] = model prediction after draft_tokens[i-1] position
        // Draft tokens[i] matches if verified[i] == draft_tokens[i]
        let mut accepted_count = 1; // The first verified token is always accepted
        for i in 1..verified.len().min(draft_tokens.len() + 1) {
            let draft_idx = i - 1;
            if draft_idx < draft_tokens.len() && verified[i - 1] == draft_tokens[draft_idx] {
                accepted_count += 1;
            } else {
                break;
            }
        }

        let newly_accepted: Vec<u32> = verified[..accepted_count.min(verified.len())].to_vec();

        // Step 4: Commit accepted tokens, truncate rejected ones.
        let n_rejected = (input_count as u32) - (accepted_count as u32);
        if n_rejected > 0 {
            ctx.inner().truncate_working_page_tokens(n_rejected);
        }

        // Commit full pages from the accepted tokens.
        let new_wpt = ctx.inner().working_page_token_count();
        let pages_to_commit = new_wpt / page_size;
        if pages_to_commit > 0 {
            ctx.inner().commit_working_pages(pages_to_commit)
                .map_err(|e| format!("Failed to commit pages: {}", e))?;
        }

        // Roll back rejected draft tokens from the drafter context too.
        let drafter_rejected = draft_length as u32 - (accepted_count.saturating_sub(1) as u32);
        if drafter_rejected > 0 {
            drafter.rollback(drafter_rejected);
        }

        // Check for stop tokens.
        let mut hit_stop = false;
        for &t in &newly_accepted {
            if stop_tokens.contains(&t) {
                hit_stop = true;
                break;
            }
            all_generated.push(t);
            total_accepted += 1;
        }

        if hit_stop || total_accepted >= max_tokens {
            break;
        }

        // The new anchor is the last accepted token.
        anchor = *newly_accepted.last().unwrap_or(&anchor);
        total_steps += 1;
    }

    // Decode output.
    let text = tokenizer.decode(&all_generated)
        .map_err(|e| format!("Decode failed: {e}"))?;

    println!(
        "--- CacheBack Decoding (draft_length={}, steps={}) ---",
        draft_length, total_steps
    );
    println!("Generated in {:?}", start.elapsed());
    println!("Output:\n{}", text);

    Ok(String::new())
}
