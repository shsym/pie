use futures::stream::StreamExt;
use std::time::Instant;
use symphony::Run;

struct SimpleDecoding;

impl Run for SimpleDecoding {
    async fn run() -> Result<(), String> {
        let start = Instant::now();
        //let mut prev = start; // track the time of the previous token

        let max_num_outputs = 256;

        let mut ctx = symphony::Context::new();
        ctx.fill("<|begin_of_text|>").await;
        ctx.fill("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|>").await;
        ctx.fill("<|start_header_id|>user<|end_header_id|>\n\nExplain the LLM decoding process ELI5.<|eot_id|>").await;
        ctx.fill("<|start_header_id|>assistant<|end_header_id|>\n\n")
            .await;

        let mut output_stream = ctx
            .generate_stream_until("<|eot_id|>", max_num_outputs)
            .await;

        let mut token_ids = Vec::new();
        while let Some(token_id) = output_stream.next().await {
            //let now = Instant::now();
            //let token_latency = now.duration_since(prev);
            //println!("Token: {} (latency: {:?})", token_id, token_latency);
            //prev = now;
            token_ids.push(token_id);
        }
        let text = symphony::l4m::detokenize(&token_ids);
        println!("Output: {:?} (total elapsed: {:?})", text, start.elapsed());
        
        // compute per token latency
        println!("Per token latency: {:?}", start.elapsed() / token_ids.len() as u32);

        Ok(())
    }
}

symphony::main!(SimpleDecoding);
