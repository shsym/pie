use std::time::Instant;
use symphony::Run;

struct SimpleDecoding;

// create a default stream constant

impl Run for SimpleDecoding {
    async fn run() -> Result<(), String> {
        let start = Instant::now();

        let max_num_outputs = 128;

        let mut ctx = symphony::Context::new();
        ctx.fill("<|begin_of_text|>").await;
        ctx.fill("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|>").await;
        ctx.fill("<|start_header_id|>user<|end_header_id|>\n\nExplain the LLM decoding process ELI5.<|eot_id|>").await;
        ctx.fill("<|start_header_id|>assistant<|end_header_id|>\n\n")
            .await;

        let output_text = ctx.generate_until("<|eot_id|>", max_num_outputs).await;

        println!("Output: {:?} (elapsed: {:?})", output_text, start.elapsed());

        Ok(())
    }
}

symphony::main!(SimpleDecoding);
