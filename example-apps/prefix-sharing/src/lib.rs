use std::time::Instant;
use symphony::Run;

struct PrefixSharing;

// create a default stream constant

impl Run for PrefixSharing {
    async fn run() -> Result<(), String> {
        let start = Instant::now();

        let max_num_outputs = 128;

        let mut ctx = symphony::Context::create();
        ctx.fill("<|begin_of_text|>").await;
        ctx.fill("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|>").await;

        let mut ctx_sub1 = ctx.fork().await;
        let mut ctx_sub2 = ctx.fork().await;

        ctx_sub1.fill("<|start_header_id|>user<|end_header_id|>\n\nExplain the LLM decoding process ELI5.<|eot_id|>").await;
        ctx_sub1
            .fill("<|start_header_id|>assistant<|end_header_id|>\n\n")
            .await;

        ctx_sub2.fill("<|start_header_id|>user<|end_header_id|>\n\nExplain the Espresso making process ELI5.<|eot_id|>").await;
        ctx_sub2
            .fill("<|start_header_id|>assistant<|end_header_id|>\n\n")
            .await;

        let output_text1 = ctx_sub1.generate_until("<|eot_id|>", max_num_outputs).await;

        println!(
            "Output: {:?} (elapsed: {:?})",
            output_text1,
            start.elapsed()
        );

        let output_text2 = ctx_sub2.generate_until("<|eot_id|>", max_num_outputs).await;

        println!(
            "Output: {:?} (elapsed: {:?})",
            output_text2,
            start.elapsed()
        );

        Ok(())
    }
}

symphony::main!(PrefixSharing);
