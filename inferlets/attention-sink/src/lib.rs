//! Demonstrates attention sink — bounded KV cache with preserved initial tokens.
//!
//! This method maintains an "attention sink" of initial tokens plus a sliding window
//! of the most recent tokens. Tokens between the sink and the window are masked via
//! a per-step attention mask, preventing the model from attending to them.
//!
//! NOTE: Full KV cache eviction is not yet supported by the runtime. The runtime's
//! `release_working_pages` API only frees *uncommitted* pages; committed pages persist.
//! This implementation uses `ForwardPass::attention_mask` to apply per-step masking,
//! so the model ignores masked tokens, but the KV memory is not freed.

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
    #[serde(default = "default_sink_size")]
    sink_size: u32,
    #[serde(default = "default_window_size")]
    window_size: u32,
}

fn default_prompt() -> String { "Tell me a long story about a cat.".to_string() }
fn default_max_tokens() -> usize { 512 }
fn default_sink_size() -> u32 { 4 }
fn default_window_size() -> u32 { 64 }

/// Build a BRLE attention mask for attention sink with sliding window.
///
/// The mask has shape [seq_len] where 1 = attend, 0 = masked.
/// Layout: [sink_tokens][masked_middle][window_tokens]
///
/// BRLE (binary run-length encoding):
///   [count_of_0s, count_of_1s, count_of_0s, count_of_1s, ...]
fn build_sink_mask(seq_len: u32, sink_size: u32, window_size: u32) -> Vec<u32> {
    let total_kept = sink_size + window_size;
    if seq_len <= total_kept {
        // Entire sequence fits — attend to everything: BRLE = [0, seq_len]
        vec![0, seq_len]
    } else {
        // Layout: [sink_size attend] [middle masked] [window_size attend]
        let middle_masked = seq_len - total_kept;
        // BRLE: [0, sink_size, middle_masked, window_size]
        // Meaning: skip 0 zeros, attend sink_size, skip middle_masked, attend window_size
        vec![0, sink_size, middle_masked, window_size]
    }
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let prompt = input.prompt;
    let max_tokens = input.max_tokens;
    let sink_size = input.sink_size;
    let window_size = input.window_size;

    let start = Instant::now();
    let models = runtime::models();
    let model = Model::load(models.first().ok_or("No models available")?)?;

    let mut ctx = Context::new(&model)?;
    let page_size = ctx.page_size();
    let stop_tokens = inferlet::instruct::chat::stop_tokens(&model);

    // Build prompt tokens into pending_tokens (manual tokenization for the
    // manual forward-pass loop below — we need explicit token control).
    let mut pending_tokens: Vec<u32> = Vec::new();
    pending_tokens.extend(inferlet::instruct::chat::system(&model, "You are a helpful assistant."));
    pending_tokens.extend(inferlet::instruct::chat::user(&model, &prompt));
    pending_tokens.extend(inferlet::instruct::chat::cue(&model));

    println!(
        "--- Attention Sink (sink={}, window={}, page_size={}) ---",
        sink_size, window_size, page_size
    );

    // Manual generation loop with per-step attention masking.
    // The first iteration processes the full prompt. Subsequent iterations
    // process one generated token at a time, applying the sink+window mask.
    let mut generated_tokens: Vec<u32> = Vec::new();
    let sampler = Sampler::TopP((0.0, 1.0));

    for _step in 0..max_tokens {
        if pending_tokens.is_empty() {
            break;
        }

        let wpt = ctx.inner().working_page_token_count();
        let seq_len = ctx.inner().committed_page_count() * page_size + wpt;

        // Reserve additional pages for the new token(s)
        let current_working_pages = ctx.inner().working_page_count();
        let total_tokens_after = wpt + pending_tokens.len() as u32;
        let total_pages_needed = (total_tokens_after + page_size - 1) / page_size;
        let additional_pages = total_pages_needed.saturating_sub(current_working_pages);
        if additional_pages > 0 {
            ctx.inner().reserve_working_pages(additional_pages)
                .map_err(|e| format!("Failed to reserve pages: {}", e))?;
        }

        let pass = ForwardPass::new(&model);
        pass.context(ctx.inner());

        let positions: Vec<u32> = (seq_len..seq_len + pending_tokens.len() as u32).collect();
        pass.input_tokens(&pending_tokens, &positions);

        // Apply attention sink mask
        let total_seq_len = seq_len + pending_tokens.len() as u32;
        let total_kept = sink_size + window_size;
        if total_seq_len > total_kept {
            let mask = build_sink_mask(total_seq_len, sink_size, window_size);
            let masks: Vec<Vec<u32>> = (0..pending_tokens.len())
                .map(|_| mask.clone())
                .collect();
            pass.attention_mask(&masks);
        }

        let last_token_idx = (pending_tokens.len() - 1) as u32;
        pass.sampler(&[last_token_idx], sampler.clone());

        let output = pass.execute_async().await?;

        let new_tokens = match output {
            Output::Tokens(tokens) => tokens,
            _ => break,
        };

        if new_tokens.is_empty() {
            break;
        }

        // Page management: commit full pages
        let new_wpt = wpt + pending_tokens.len() as u32;
        let pages_to_commit = new_wpt / page_size;

        if pages_to_commit > 0 {
            ctx.inner().commit_working_pages(pages_to_commit)
                .map_err(|e| format!("Failed to commit pages: {}", e))?;
        }

        // Check for stop tokens
        let token = new_tokens[0];
        if stop_tokens.contains(&token) {
            break;
        }

        generated_tokens.push(token);

        // Buffer the accepted token for the next step
        pending_tokens = vec![token];
    }

    let tokenizer = model.tokenizer();
    let text = tokenizer.decode(&generated_tokens)?;
    println!("Generated {} tokens in {:?}", generated_tokens.len(), start.elapsed());
    println!("Output:\n{}", text);

    Ok(String::new())
}
