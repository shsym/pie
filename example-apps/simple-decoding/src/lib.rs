use std::time::Instant;
use symphony::RunSync;

struct SimpleDecoding;

// create a default stream constant
const MAIN: u32 = 0;

impl RunSync for SimpleDecoding {
    fn run() -> Result<(), String> {
        // Test the system ping latency

        for _ in 0..10 {
            let start = Instant::now();
            let resp = symphony::ping::ping("hello");
            let duration = start.elapsed();
            println!("Ping response: {:?}, Time elapsed: {:?}", resp, duration);
        }

        let start = Instant::now();
        println!("Running the simple decoding example...");

        let mut ctx = symphony::Context::new(MAIN);

        ctx.fill("<|begin_of_text|>");
        ctx.fill("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|>");
        ctx.fill("<|start_header_id|>user<|end_header_id|>\n\nExplain the LLM decoding process ELI5.<|eot_id|>");
        ctx.fill("<|start_header_id|>assistant<|end_header_id|>\n\n");

        let output_text = ctx.generate_until("<|eot_id|>", 100);
        let duration = start.elapsed();

        println!("Output text: {:?}", output_text);

        // Print elapsed time in milliseconds
        println!("Time elapsed: {:?}", duration);

        Ok(())
    }
}

symphony::main_sync!(SimpleDecoding);

//
// fn run() -> Result<(), String> {
//     // Test the system ping latency
//
//     for _ in 0..10 {
//         let start = Instant::now();
//         let resp = symphony::ping::ping("hello");
//         let duration = start.elapsed();
//         println!("Ping response: {:?}, Time elapsed: {:?}", resp, duration);
//     }
//
//     let start = Instant::now();
//     println!("Running the simple decoding example...");
//     let prompt = llama3_format("Explain the LLM decoding process ELI5.", None, None);
//
//     let prompt_tokens = symphony::l4m::tokenize(&prompt);
//     let eos_token = symphony::l4m::tokenize("<|eot_id|>")[0];
//
//     let block_size = symphony::l4m::get_block_size() as usize;
//
//     let range = (0..(prompt_tokens.len() + MAX_NUM_OUTPUTS) as u32).collect::<Vec<u32>>();
//
//     // Step 1. Do "prefilling" of the context blocks
//     let mut context_blocks = {
//         let num_context_blocks = prompt_tokens.len() / block_size;
//         let prompt_token_embeds =
//             symphony::l4m::allocate_embeds(MAIN, (num_context_blocks * block_size) as u32);
//
//         // embed texts
//         symphony::l4m::embed_text(
//             MAIN,
//             &prompt_token_embeds,
//             &prompt_tokens[0..(num_context_blocks * block_size)],
//             &range[0..(num_context_blocks * block_size)],
//         );
//
//         // allocate blocks
//         let prefilled_blocks = symphony::l4m::allocate_blocks(MAIN, num_context_blocks as u32);
//
//         // fill blocks (=prefilling in the classic LLM inference settings)
//         // While this fill block operation is submitted sequentially in a loop, it is batched internally by the controller.
//         // So we can expect a good performance here.
//         for i in 0..num_context_blocks {
//             let offset = i * block_size;
//
//             // we don't need to store the output embeds
//             symphony::l4m::fill_block(
//                 MAIN,
//                 prefilled_blocks[i],
//                 &prefilled_blocks[..(i + 1)],
//                 &prompt_token_embeds[offset..offset + block_size],
//                 &[],
//             );
//         }
//
//         // Free the prompt token embeds
//         symphony::l4m::deallocate_embeds(MAIN, &prompt_token_embeds);
//
//         prefilled_blocks
//     };
//
//     //// Step 2. Do the actual decoding
//
//     let input_block_embeds = symphony::l4m::allocate_embeds(MAIN, block_size as u32);
//     let output_block_embeds = symphony::l4m::allocate_embeds(MAIN, block_size as u32);
//
//     let idx_offset = context_blocks.len() * block_size;
//
//     // put the remaining tokens into the last block
//     let remaining_tokens = prompt_tokens[idx_offset..].to_vec();
//     let next_dist = symphony::l4m::allocate_dists(MAIN, 1);
//
//     // initialize input_embeds with the leftover blocks.
//     symphony::l4m::embed_text(
//         MAIN,
//         &input_block_embeds[..remaining_tokens.len()],
//         &prompt_tokens[idx_offset..idx_offset + remaining_tokens.len()],
//         &range[idx_offset..idx_offset + remaining_tokens.len()],
//     );
//
//     let valid_len = remaining_tokens.len();
//     let mut working_block_idx = context_blocks.len();
//
//     let mut output_tokens = Vec::new();
//
//     // allocate a new block
//     context_blocks.push(symphony::l4m::allocate_blocks(MAIN, 1)[0]);
//
//     for i in 0..MAX_NUM_OUTPUTS {
//         let start_time = Instant::now();
//
//         let offset = (i + valid_len - 1) % block_size;
//
//         symphony::l4m::fill_block(
//             MAIN,
//             context_blocks[working_block_idx],
//             &context_blocks[..working_block_idx + 1], // the context should be inclusive of the current block
//             &input_block_embeds[..offset + 1],
//             &output_block_embeds[..offset + 1],
//         );
//
//         // let's sample the next token
//         symphony::l4m::decode_token_dist(
//             MAIN,
//             &output_block_embeds[offset..offset + 1],
//             &next_dist,
//         );
//
//         // Right now, this is a blocking operation. We will soon provide an async version.
//         let sampled = symphony::l4m::sample_top_k(MAIN, &next_dist, 1);
//
//         let (next_tokens, probs) = &sampled[0];
//
//         let next_token = next_tokens[0];
//
//         // Check the EOS token (TODO)
//         if next_token == eos_token {
//             break;
//         }
//
//         output_tokens.push(next_token);
//
//         let next_offset = (offset + 1) % block_size;
//
//         symphony::l4m::embed_text(
//             MAIN,
//             &input_block_embeds[next_offset..next_offset + 1],
//             &[next_token],
//             &[(working_block_idx * block_size + valid_len + i) as u32],
//         );
//
//         if next_offset == 0 {
//             // move to the next block
//             working_block_idx += 1;
//             context_blocks.push(symphony::l4m::allocate_blocks(MAIN, 1)[0]);
//             println!("Allocated a new block at index {}", working_block_idx);
//         }
//
//         let duration = start_time.elapsed();
//         println!("Time elapsed for iteration {}: {:?}", i, duration);
//     }
//
//     let duration = start.elapsed();
//
//     let output_text = symphony::l4m::detokenize(&output_tokens);
//     println!("Output text: {:?}", output_text);
//
//     // Print elapsed time in milliseconds
//     println!("Time elapsed: {:?}", duration);
//
//     Ok(())
// }
