use std::time::Instant;
use symphony::Run;

struct ParallelGeneration;

// create a default stream constant

impl Run for ParallelGeneration {
    async fn run() -> Result<(), String> {
        let start = Instant::now();

        let max_num_outputs = 128;

        // tokio spawn
        let handle1 = symphony::tokio::spawn(async move {
            let mut ctx = symphony::Context::new();
            ctx.fill("<|begin_of_text|>");
            ctx.fill("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|>");
            ctx.fill("<|start_header_id|>user<|end_header_id|>\n\nExplain the LLM decoding process ELI5.<|eot_id|>");
            ctx.fill("<|start_header_id|>assistant<|end_header_id|>\n\n");

            let output_text1 = ctx.generate_until("<|eot_id|>", max_num_outputs);

            println!(
                "Output: {:?} (elapsed: {:?})",
                output_text1,
                start.elapsed()
            );
        });

        let handle2 = symphony::tokio::spawn(async move {
            let mut ctx = symphony::Context::new();
            ctx.fill("<|begin_of_text|>");
            ctx.fill("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|>");
            ctx.fill("<|start_header_id|>user<|end_header_id|>\n\nExplain the Espresso making process ELI5.<|eot_id|>");
            ctx.fill("<|start_header_id|>assistant<|end_header_id|>\n\n");

            let output_text2 = ctx.generate_until("<|eot_id|>", max_num_outputs);

            println!(
                "Output: {:?} (elapsed: {:?})",
                output_text2,
                start.elapsed()
            );
        });

        // wait for both tasks to complete
        handle1.await.unwrap();
        handle2.await.unwrap();

        Ok(())
    }
}

symphony::main!(ParallelGeneration);
