use inferlet::Context;
use inferlet::sampler::{self, Sampler};
use inferlet::stop_condition::{self, StopCondition};
use inferlet::traits::Tokenize;
use pico_args::Arguments;
use std::ffi::OsString;
use std::time::Instant;

/// Defines the command-line interface and help message.
const HELP: &str = r#"
Usage: program [OPTIONS]

A simple inferlet to run a chat model with a configurable attention sink.

Options:
  -p, --prompt <STRING>        The prompt to send to the model.
                               (default: "Explain LLM decoding process in ELI5.")
  -n, --max-tokens <INT>       The maximum number of new tokens to generate.
                               (default: 512)
  -s, --sink-size <INT>        The initial size of the attention sink.
                               (default: 64)
  -w, --sink-window <INT>      The sliding window size for the attention sink.
                               (default: 32)
  -h, --help                   Print help information.
"#;

/// Generates text autoregressively using an attention sink to manage a rolling KV cache.
///
/// This method is an optimized version of `generate` designed for handling very long
/// sequences. It implements a sliding window attention mechanism, often called an
/// "attention sink," to keep the KV cache size bounded.
///
pub async fn generate_with_attention_sink<S: Sampler, C: StopCondition>(
    ctx: &mut Context,
    sampler: &mut S,
    stop_condition: &mut C,
    attention_sink_initial_size: usize,
    attention_sink_window_size: usize,
) -> String {
    let mut generated_token_ids = Vec::new();
    let max_cache_size = attention_sink_initial_size + attention_sink_window_size;

    // The autoregressive generation loop
    loop {
        // 1. Decode the next token
        let dist = ctx.decode_step().await;
        let next_token_id = sampler.sample(&dist.ids, &dist.probs);
        ctx.fill_token(next_token_id);
        generated_token_ids.push(next_token_id);

        // 2. Check if the stop condition is met
        if stop_condition.should_stop(&generated_token_ids) {
            break;
        }

        // 3. Apply attention sink logic
        // This logic operates on the committed tokens that are present in the KV cache.
        let committed_len = ctx.token_ids.len();

        if committed_len > max_cache_size {
            // Determine the range of tokens to "evict" from the attention window.
            // These are the tokens that are no longer in the initial sink or the
            // recent sliding window.
            let num_to_evict = committed_len - max_cache_size;
            let evict_start = attention_sink_initial_size;
            let evict_end = attention_sink_initial_size + num_to_evict;

            // Mask this range so the model ignores it in future forward passes.
            ctx.mask_token_range(evict_start, evict_end, true);

            // Attempt to drop any full KV cache pages that are now fully masked.
            // This is the step that actually frees memory. It will only act when
            // a page is completely filled with masked-out tokens.
            ctx.drop_masked_kv_pages();
        }
    }

    ctx.tokenizer.detokenize(&generated_token_ids)
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

    // 2. Handle the --help flag.
    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(());
    }

    // 3. Parse arguments, falling back to defaults if they are not provided.
    let max_num_outputs: usize = args
        .opt_value_from_str(["-n", "--max-tokens"])
        .map_err(|e| e.to_string())?
        .unwrap_or(512);

    let attention_sink_initial_size: usize = args
        .opt_value_from_str(["-s", "--sink-size"])
        .map_err(|e| e.to_string())?
        .unwrap_or(64);

    let attention_sink_window_size: usize = args
        .opt_value_from_str(["-w", "--sink-window"])
        .map_err(|e| e.to_string())?
        .unwrap_or(32);

    let prompt = args
        .opt_value_from_str(["-p", "--prompt"])
        .map_err(|e| e.to_string())?
        .unwrap_or_else(|| "Explain LLM decoding process in ELI5.".to_string());

    // 4. Ensure no unknown arguments were passed.
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

    let mut ctx = Context::new(&model);
    ctx.fill("<|begin_of_text|>");
    ctx.fill("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|>");
    ctx.fill(&format!(
        "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>",
        prompt
    ));
    ctx.fill("<|start_header_id|>assistant<|end_header_id|>\n\n");

    let mut sampler = sampler::GreedySampler::new();

    let mut stop_condition = stop_condition::any(
        stop_condition::Until::new(tokenizer.tokenize("<|eot_id|>")),
        stop_condition::Length::new(max_num_outputs),
    );

    // Call the standalone function with the parsed arguments
    let output = generate_with_attention_sink(
        &mut ctx,
        &mut sampler,
        &mut stop_condition,
        attention_sink_initial_size,
        attention_sink_window_size,
    )
    .await;

    let elapsed = start.elapsed();
    let output_token_ids = tokenizer.tokenize(&output);

    println!("\n--- Output ---\n{}\n--------------", output);

    println!(
        "Total elapsed: {:?}, Tokens generated: {}",
        elapsed,
        output_token_ids.len()
    );

    // compute per token latency
    if !output_token_ids.is_empty() {
        println!(
            "Per-token latency: {:?}",
            elapsed / output_token_ids.len() as u32
        );
    }

    Ok(())
}
