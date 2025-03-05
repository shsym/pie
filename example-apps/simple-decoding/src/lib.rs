use std::time::Instant;
use symphony::RunSync;

struct SimpleDecoding;

// create a default stream constant

impl RunSync for SimpleDecoding {
    fn run() -> Result<(), String> {
        let start = Instant::now();

        let max_num_outputs = 128;

        let mut ctx = symphony::Context::new();
        ctx.fill("<|begin_of_text|>");
        ctx.fill("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|>");
        ctx.fill("<|start_header_id|>user<|end_header_id|>\n\nExplain the LLM decoding process ELI5.<|eot_id|>");
        ctx.fill("<|start_header_id|>assistant<|end_header_id|>\n\n");

        let output_text = ctx.generate_until("<|eot_id|>", max_num_outputs);

        println!("Output: {:?} (elapsed: {:?})", output_text, start.elapsed());

        Ok(())
    }
}

symphony::main_sync!(SimpleDecoding);
