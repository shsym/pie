use inferlet::sampler::{self, Sampler};
use inferlet::stop_condition::{self, StopCondition};
use inferlet::traits::{Forward, Tokenize};

use inferlet::Context;
use inferlet::brle::Brle;
use std::time::Instant;

// NOTE: This is not a full implementation of Parallel Jacobi Decoding (PJD).
// It demonstrates a single step in the PJD process without the refinement step,
pub async fn generate_with_pjd<S: Sampler, C: StopCondition>(
    ctx: &mut Context,
    sampler: &mut S,
    stop_condition: &mut C,
    gamma: usize,
) -> String {
    let mut all_generated_tokens = Vec::new();
    // Fallback to 0 if the tokenizer doesn't have a specific UNK token ID.
    let unk_token_id = 0;

    loop {
        // 1. Check stop condition before generating more tokens.
        if stop_condition.should_stop(&all_generated_tokens) {
            break;
        }

        // 3. Draft a batch using the seed and `gamma` unknown tokens.

        let pending_tokens = std::mem::take(&mut ctx.token_ids_pending);

        let draft_tokens = vec![unk_token_id; gamma];
        let batch_tokens = [pending_tokens.as_slice(), draft_tokens.as_slice()].concat();
        let batch_len = batch_tokens.len();

        let batch_positions = {
            let pos_offset = ctx.position_ids.last().map(|&p| p + 1).unwrap_or(0);
            (pos_offset..pos_offset + batch_len as u32).collect::<Vec<u32>>()
        };

        // 4. Run verification pass.
        ctx.grow_kv_pages(batch_len);

        let total_ctx_len = ctx.token_ids.len() + batch_len;
        // Prepare the attention masks for the batch.
        let masks_for_batch = causal_mask(total_ctx_len as u32, batch_len as u32);

        let sample_indices: Vec<u32> =
            ((pending_tokens.len() as u32 - 1)..(pending_tokens.len() + gamma) as u32).collect();

        // Run a single forward pass to get distributions for all `gamma + 1` positions.
        let p = ctx.queue.create_forward_pass();
        p.kv_cache(&ctx.kv_pages, ctx.kv_page_last_len);
        p.input_tokens(&batch_tokens, &batch_positions);
        p.attention_mask(&masks_for_batch);
        p.output_distributions(&sample_indices, 1.0, None);

        let output_distributions = p.execute().await.distributions.unwrap();

        // 5. Sample from each distribution to get the accepted tokens.
        let accepted_tokens: Vec<u32> = output_distributions
            .iter()
            .map(|dist| sampler.sample(&dist.ids, &dist.probs))
            .collect();

        // print inputs
        // println!("inputs: {:?}", batch_tokens);
        // println!("positions: {:?}", batch_positions);
        // println!("masks: {:?}", masks_for_batch);
        // println!("sample indices: {:?}", sample_indices);
        // println!("accepted tokens: {:?}", accepted_tokens);

        // 6. Update state for the next iteration.
        // The KV cache was updated based on the `unk` draft. We commit these draft
        // tokens to the history to keep the context consistent.
        ctx.token_ids.extend_from_slice(&pending_tokens);
        ctx.token_ids.extend_from_slice(&accepted_tokens[..gamma]);
        ctx.position_ids.extend_from_slice(&batch_positions);

        // The generated output string is built from the newly sampled tokens.
        all_generated_tokens.extend_from_slice(&accepted_tokens);

        // The accepted tokens become the input for the next round (the flush step).
        ctx.fill_token(accepted_tokens[gamma]);
    }

    // Clean up any remaining pending tokens after the loop finishes.
    ctx.tokenizer.detokenize(&all_generated_tokens)
}

pub fn causal_mask(num_total_tokens: u32, num_input_tokens: u32) -> Vec<Vec<u32>> {
    let mut mask = Vec::new();
    let offset = num_total_tokens - num_input_tokens;
    for i in 0..num_input_tokens {
        mask.push(Brle::new((offset + i + 1) as usize).buffer);
    }
    mask
}

#[inferlet::main]
async fn main() -> Result<(), String> {
    let start = Instant::now();

    let max_num_outputs = 256;
    let speculation_length = 3; // Number of tokens to speculate in parallel

    let model = inferlet::get_auto_model();
    let tokenizer = model.get_tokenizer();

    let mut ctx = Context::new(&model);
    ctx.fill("<|begin_of_text|>");
    ctx.fill("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|>");
    ctx.fill("<|start_header_id|>user<|end_header_id|>\n\nExplain LLM decoding process in ELI5.<|eot_id|>");
    ctx.fill("<|start_header_id|>assistant<|end_header_id|>\n\n");

    let mut sampler = sampler::GreedySampler::new();

    let mut stop_condition = stop_condition::any(
        stop_condition::Until::new(tokenizer.tokenize("<|eot_id|>")),
        stop_condition::Length::new(max_num_outputs),
    );

    //let mut stop_condition =         stop_condition::Length::new(max_num_outputs);

    println!(
        "Starting generation with Parallel Jacobi Decoding (speculation_length = {})...",
        speculation_length
    );

    // Call the standalone function
    let output = generate_with_pjd(
        &mut ctx,
        &mut sampler,
        &mut stop_condition,
        speculation_length,
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
