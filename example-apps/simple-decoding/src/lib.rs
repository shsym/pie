use std::time::Instant;
use symphony::RunSync;

struct SimpleDecoding;

fn llama3_format(prompt: &str, hint: Option<&str>, system: Option<&str>) -> String {
    let system_msg = system.unwrap_or("You are a helpful, respectful and honest assistant.");
    format!(
        "<|begin_of_text|>\
<|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>\
<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>\
<|start_header_id|>assistant<|end_header_id|>\n\n{}",
        system_msg,
        prompt,
        hint.unwrap_or("")
    )
}

// create a default stream constant
const MAIN: u32 = 0;
const MAX_NUM_OUTPUTS: usize = 32;

impl RunSync for SimpleDecoding {
    fn run() -> Result<(), String> {
        let start = Instant::now();

        let prompt = llama3_format("Explain the LLM decoding process ELI5.", None, None);

        let prompt_tokens = symphony::inference::tokenize(&prompt);
        let eos_token = symphony::inference::tokenize("<|eot_id|>")[0];

        let block_size = symphony::inference::get_block_size() as usize;

        let range = (0..(prompt_tokens.len() + MAX_NUM_OUTPUTS) as u32).collect::<Vec<u32>>();

        // Step 1. Do "prefilling" of the context blocks
        let mut context_blocks = {
            let num_context_blocks = prompt_tokens.len() / block_size;
            let prompt_token_embeds = symphony::inference::allocate_embeds(
                MAIN,
                (num_context_blocks * block_size) as u32,
            );

            // embed texts
            symphony::inference::embed_text(
                MAIN,
                &prompt_token_embeds,
                &prompt_tokens[0..(num_context_blocks * block_size)],
                &range[0..(num_context_blocks * block_size)],
            );

            // allocate blocks
            let prefilled_blocks =
                symphony::inference::allocate_blocks(MAIN, num_context_blocks as u32);

            // fill blocks (=prefilling in the classic LLM inference settings)
            // While this fill block operation is submitted sequentially in a loop, it is batched internally by the controller.
            // So we can expect a good performance here.
            for i in 0..num_context_blocks {
                let offset = i * block_size;

                // we don't need to store the output embeds
                symphony::inference::fill_block(
                    MAIN,
                    prefilled_blocks[i],
                    &prefilled_blocks[..(i + 1)],
                    &prompt_token_embeds[offset..offset + block_size],
                    &[],
                );
            }

            // Free the prompt token embeds
            symphony::inference::deallocate_embeds(MAIN, &prompt_token_embeds);

            prefilled_blocks
        };

        //// Step 2. Do the actual decoding

        let input_block_embeds = symphony::inference::allocate_embeds(MAIN, block_size as u32);
        let output_block_embeds = symphony::inference::allocate_embeds(MAIN, block_size as u32);

        let idx_offset = context_blocks.len() * block_size;

        // put the remaining tokens into the last block
        let remaining_tokens = prompt_tokens[idx_offset..].to_vec();
        let next_dist = symphony::inference::allocate_dists(MAIN, 1);

        // initialize input_embeds with the leftover blocks.
        symphony::inference::embed_text(
            MAIN,
            &input_block_embeds[..remaining_tokens.len()],
            &prompt_tokens[idx_offset..idx_offset + remaining_tokens.len()],
            &range[idx_offset..idx_offset + remaining_tokens.len()],
        );

        let mut valid_len = remaining_tokens.len();
        let mut working_block_idx = context_blocks.len();

        let mut output_tokens = Vec::new();

        // allocate a new block
        context_blocks.push(symphony::inference::allocate_blocks(MAIN, 1)[0]);

        for i in 0..MAX_NUM_OUTPUTS {
            let offset = (i + valid_len - 1) % block_size;

            symphony::inference::fill_block(
                MAIN,
                context_blocks[working_block_idx],
                &context_blocks[..working_block_idx + 1], // the context should be inclusive of the current block
                &input_block_embeds[..offset + 1],
                &output_block_embeds[..offset + 1],
            );

            // let's sample the next token
            symphony::inference::decode_token_dist(
                MAIN,
                &output_block_embeds[offset..offset + 1],
                &next_dist,
            );

            // Right now, this is a blocking operation. We will soon provide an async version.
            let sampled = symphony::inference::sample_top_k(MAIN, &next_dist, 1);

            let next_token = sampled[0][0];

            // Check the EOS token (TODO)
            if next_token == eos_token {
                break;
            }

            output_tokens.push(next_token);

            symphony::inference::embed_text(
                MAIN,
                &input_block_embeds[offset..offset + 1],
                &[next_token],
                &[(working_block_idx * block_size + valid_len) as u32],
            );

            if offset == block_size - 1 {
                // move to the next block
                working_block_idx += 1;
                context_blocks.push(symphony::inference::allocate_blocks(MAIN, 1)[0]);
            }
        }

        let duration = start.elapsed();

        let output_text = symphony::inference::detokenize(&output_tokens);
        println!("Output text: {:?}", output_text);

        // Print elapsed time in milliseconds
        println!("Time elapsed: {:?} ms", duration);

        Ok(())
    }
}

symphony::main_sync!(SimpleDecoding);
