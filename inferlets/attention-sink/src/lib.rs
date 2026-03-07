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
    context::Context, model::Model, runtime,
    ContextExt, ForwardPassExt, InstructExt, Result,
    inference::{ForwardPass, Output, Sampler},
};
use std::time::Instant;

const HELP: &str = "\
Usage: attention-sink [OPTIONS]

A program to demonstrate attention sink with bounded KV cache.

Options:
  -p, --prompt <PROMPT>        The prompt text [default: \"Tell me a long story about a cat.\"]
  -n, --max-tokens <TOKENS>    Maximum number of tokens to generate [default: 512]
  -s, --sink-size <SIZE>       Number of initial sink tokens to preserve [default: 4]
  -w, --window-size <SIZE>     Sliding window size in tokens [default: 64]
  -h, --help                   Prints this help message";

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
async fn main(args: Vec<String>) -> Result<String> {
    let mut args = inferlet::parse_args(args);

    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(String::new());
    }

    let prompt: String = args
        .value_from_str(["-p", "--prompt"])
        .unwrap_or_else(|_| "Tell me a long story about a cat.".to_string());
    let max_tokens: usize = args.value_from_str(["-n", "--max-tokens"]).unwrap_or(512);
    let sink_size: u32 = args.value_from_str(["-s", "--sink-size"]).unwrap_or(4);
    let window_size: u32 = args.value_from_str(["-w", "--window-size"]).unwrap_or(64);

    let start = Instant::now();
    let models = runtime::models();
    let model = Model::load(models.first().ok_or("No models available")?)?;

    let ctx = Context::create(&model)?;
    let page_size = ctx.tokens_per_page();
    let stop_tokens = Context::stop_tokens(&model);

    let mut pending_tokens: Vec<u32> = Vec::new();
    pending_tokens.extend(ctx.system("You are a helpful assistant."));
    pending_tokens.extend(ctx.user(&prompt));
    pending_tokens.extend(ctx.cue());
    ctx.flush(&pending_tokens).await?;
    pending_tokens.clear();

    println!(
        "--- Attention Sink (sink={}, window={}, page_size={}) ---",
        sink_size, window_size, page_size
    );

    // Manual generation loop with per-step attention masking.
    // This follows the same page management pattern as the SDK's
    // TokenStream::step(), but adds a sink-window attention_mask on
    // each ForwardPass.
    let mut generated_tokens: Vec<u32> = Vec::new();
    let sampler = Sampler::TopP((0.0, 1.0));

    // Bootstrap: sample the first token from the flushed KV cache
    let bootstrap_pass = ForwardPass::new(&model);
    bootstrap_pass.context(&ctx);
    bootstrap_pass.sampler(&[0], sampler.clone());
    let bootstrap_output = bootstrap_pass.execute_async().await?;
    match bootstrap_output {
        Output::Tokens(t) if !t.is_empty() => {
            if stop_tokens.contains(&t[0]) {
                // First token is stop — nothing to generate
            } else {
                generated_tokens.push(t[0]);
                pending_tokens = vec![t[0]];
            }
        }
        _ => {}
    }

    for _step in 1..max_tokens {
        if pending_tokens.is_empty() {
            break;
        }

        let wpt = ctx.working_page_token_count();
        let seq_len = ctx.committed_page_count() * page_size + wpt;

        // Reserve additional pages for the new token(s)
        let current_working_pages = ctx.working_page_count();
        let total_tokens_after = wpt + pending_tokens.len() as u32;
        let total_pages_needed = (total_tokens_after + page_size - 1) / page_size;
        let additional_pages = total_pages_needed.saturating_sub(current_working_pages);
        if additional_pages > 0 {
            ctx.reserve_working_pages(additional_pages)
                .map_err(|e| format!("Failed to reserve pages: {}", e))?;
        }

        let pass = ForwardPass::new(&model);
        pass.context(&ctx);

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

        // Page management: commit pages
        let new_wpt = wpt + pending_tokens.len() as u32;
        let pages_to_commit = new_wpt / page_size;

        if pages_to_commit > 0 {
            ctx.commit_working_pages(pages_to_commit)
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
