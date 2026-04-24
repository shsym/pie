//! Demonstrates Jacobi decoding — parallel speculation via fixed-point iteration.
//!
//! Instead of autoregressive decoding, Jacobi decoding initializes N positions with
//! guessed tokens, then iteratively runs forward passes until the predictions converge
//! (reach a fixed point). This can verify multiple tokens per forward pass.
//!
//! All tokens (verified anchor + speculative guesses) are sent as regular `input_tokens`
//! and buffered beforehand so the runtime's fill() check passes. After each forward pass,
//! only the accepted prefix is committed to the KV cache.

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
    #[serde(default = "default_window_size")]
    window_size: usize,
}

fn default_prompt() -> String { "Write a poem about the ocean.".to_string() }
fn default_max_tokens() -> usize { 256 }
fn default_window_size() -> usize { 5 }

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let prompt = input.prompt;
    let max_tokens = input.max_tokens;
    let window_size = input.window_size;

    let start = Instant::now();
    let models = runtime::models();
    let model = Model::load(models.first().ok_or("No models available")?)?;
    let tokenizer = model.tokenizer();
    let stop_tokens = inferlet::instruct::chat::stop_tokens(&model);

    let mut ctx = Context::new(&model)?;
    let page_size = ctx.page_size();

    // Fill the prompt using the Context API and flush to run prefill.
    ctx.system("You are a helpful assistant.")
        .user(&prompt)
        .cue();
    ctx.flush().await?;

    println!(
        "--- Jacobi Decoding (window_size={}, page_size={}) ---",
        window_size, page_size
    );

    let mut all_generated: Vec<u32> = Vec::new();
    let sampler = Sampler::TopP((0.0, 1.0));

    // Bootstrap: sample the first token from the prefilled context.
    // After flush(), the context has pages allocated with prompt KV.
    // We need to run a 1-token decode step to get the first output token.
    {
        let wpt = ctx.inner().working_page_token_count();
        let seq_len = ctx.inner().committed_page_count() * page_size + wpt;

        // Use the last cue token as decode trigger
        let cue_tokens = inferlet::instruct::chat::cue(&model);
        let trigger = *cue_tokens.last().unwrap_or(&0);

        let current_working_pages = ctx.inner().working_page_count();
        let total_tokens_after = wpt + 1;
        let total_pages_needed = (total_tokens_after + page_size - 1) / page_size;
        let additional_pages = total_pages_needed.saturating_sub(current_working_pages);
        if additional_pages > 0 {
            ctx.inner().reserve_working_pages(additional_pages)
                .map_err(|e| format!("Failed to reserve pages: {}", e))?;
        }

        let pass = ForwardPass::new(&model);
        pass.context(ctx.inner());
        pass.input_tokens(&[trigger], &[seq_len]);
        pass.sampler(&[0], sampler.clone());
        let output = pass.execute_async().await
            .map_err(|e| format!("Bootstrap forward pass failed: {}", e))?;
        match output {
            Output::Tokens(t) if !t.is_empty() => {
                all_generated.push(t[0]);
            }
            _ => return Err("Bootstrap sampling produced no token".to_string()),
        }
    }
    let mut anchor = all_generated[0];

    // Speculative guesses initialized to copies of the anchor
    let mut window: Vec<u32> = vec![anchor; window_size];
    let mut total_accepted = 1; // anchor already counted

    while total_accepted < max_tokens {
        let wpt = ctx.inner().working_page_token_count();
        let seq_len = ctx.inner().committed_page_count() * page_size + wpt;

        // Build the full token list: [anchor] + [window guesses]
        let mut input_all = vec![anchor];
        input_all.extend_from_slice(&window);

        let input_count = input_all.len();

        // Reserve additional pages
        let current_working_pages = ctx.inner().working_page_count();
        let total_tokens_after = wpt + input_count as u32;
        let total_pages_needed = (total_tokens_after + page_size - 1) / page_size;
        let additional_pages = total_pages_needed.saturating_sub(current_working_pages);
        if additional_pages > 0 {
            ctx.inner().reserve_working_pages(additional_pages)
                .map_err(|e| format!("Failed to reserve pages: {}", e))?;
        }

        let pass = ForwardPass::new(&model);
        pass.context(ctx.inner());

        // All tokens go as regular input (so they're all embedded and processed)
        let positions: Vec<u32> = (seq_len..seq_len + input_count as u32).collect();
        pass.input_tokens(&input_all, &positions);

        // Request sampling at all positions
        let sample_indices: Vec<u32> = (0..input_count as u32).collect();
        pass.sampler(&sample_indices, sampler.clone());

        let output = pass.execute_async().await
            .map_err(|e| format!("Forward pass failed: {}", e))?;

        let predicted_tokens = match output {
            Output::Tokens(tokens) => tokens,
            _ => break,
        };

        if predicted_tokens.is_empty() {
            break;
        }

        // Jacobi verification: find the longest converged prefix.
        let mut accepted_count = 1; // Always accept the first prediction
        for i in 1..predicted_tokens.len().min(window.len() + 1) {
            let i_window = i - 1;
            if i_window < window.len() && predicted_tokens[i - 1] == window[i_window] {
                accepted_count += 1;
            } else {
                break;
            }
        }

        let newly_accepted: Vec<u32> = predicted_tokens[..accepted_count.min(predicted_tokens.len())]
            .to_vec();

        // Check for stop tokens
        let mut stop_at = newly_accepted.len();
        for (i, &t) in newly_accepted.iter().enumerate() {
            if stop_tokens.contains(&t) {
                stop_at = i;
                break;
            }
        }
        let final_accepted = &newly_accepted[..stop_at];

        all_generated.extend_from_slice(final_accepted);
        total_accepted += final_accepted.len();

        // Commit pages for accepted tokens only: anchor + accepted
        let commit_count = 1 + final_accepted.len();
        let new_wpt = wpt + commit_count as u32;
        let pages_to_commit = new_wpt / page_size;
        if pages_to_commit > 0 {
            ctx.inner().commit_working_pages(pages_to_commit)
                .map_err(|e| format!("Failed to commit pages: {}", e))?;
        }

        // Pop un-accepted speculative tokens from working pages
        let speculative_count = input_count as u32 - commit_count as u32;
        if speculative_count > 0 {
            ctx.inner().truncate_working_page_tokens(speculative_count);
        }

        if stop_at < newly_accepted.len() || total_accepted >= max_tokens {
            break;
        }

        // Prepare for next iteration
        let last_accepted = *final_accepted.last().unwrap();
        anchor = last_accepted;

        // Build next window of guesses
        window = if accepted_count < predicted_tokens.len() {
            let mut w: Vec<u32> = predicted_tokens[accepted_count..].to_vec();
            w.truncate(window_size);
            while w.len() < window_size {
                w.push(last_accepted);
            }
            w
        } else {
            vec![last_accepted; window_size]
        };
    }

    let text = tokenizer.decode(&all_generated)?;
    println!(
        "Generated {} tokens in {:?} ({:.1} tokens/s)",
        all_generated.len(),
        start.elapsed(),
        all_generated.len() as f64 / start.elapsed().as_secs_f64()
    );
    println!("Output:\n{}", text);

    Ok(String::new())
}
