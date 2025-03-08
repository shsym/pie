use std::time::Instant;
use symphony::Run;

struct ParallelGeneration;

// create a default stream constant

impl Run for ParallelGeneration {
    async fn run() -> Result<(), String> {
        let start = Instant::now();

        let max_num_outputs = 32;

        let mut common = symphony::Context::new();
        common.fill("<|begin_of_text|>").await;
        common.fill("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|>").await;

        // tokio spawn
        let mut ctx1 = common.fork().await;
        let handle1 = async move {
            ctx1.fill(
                "<|start_header_id|>user<|end_header_id|>\n\nExplain Pulmonary Embolism<|eot_id|>",
            )
            .await;
            ctx1.fill("<|start_header_id|>assistant<|end_header_id|>\n\n")
                .await;

            let output = ctx1.generate_until("<|eot_id|>", max_num_outputs).await;

            println!("Output: {:?} (elapsed: {:?})", output, start.elapsed());
        };

        let mut ctx2 = common.fork().await;
        let handle2 = async move {
            ctx2.fill("<|start_header_id|>user<|end_header_id|>\n\nExplain the Espresso making process ELI5.<|eot_id|>").await;
            ctx2.fill("<|start_header_id|>assistant<|end_header_id|>\n\n")
                .await;

            let output = ctx2.generate_until("<|eot_id|>", max_num_outputs).await;

            println!("Output: {:?} (elapsed: {:?})", output, start.elapsed());
        };

        // wait for both tasks to complete
        //(handle1, handle2).join().await;
        futures::future::join(handle1, handle2).await;

        Ok(())
    }
}

symphony::main!(ParallelGeneration);
