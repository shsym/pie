use std::time::Instant;

#[inferlet::main]
async fn main() -> Result<(), String> {
    let start = Instant::now();

    let max_num_outputs = 32;

    let model = inferlet::get_auto_model();
    let mut common = model.create_context();
    common.fill("<|begin_of_text|>");
    common.fill("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|>");

    // tokio spawn
    let mut ctx1 = common.fork();
    let handle1 = async move {
        ctx1.fill(
            "<|start_header_id|>user<|end_header_id|>\n\nExplain Pulmonary Embolism<|eot_id|>",
        );
        ctx1.fill("<|start_header_id|>assistant<|end_header_id|>\n\n");

        let output = ctx1.generate_until(max_num_outputs).await;

        println!("Output: {:?} (elapsed: {:?})", output, start.elapsed());
    };

    let mut ctx2 = common.fork();
    let handle2 = async move {
        ctx2.fill("<|start_header_id|>user<|end_header_id|>\n\nExplain the Espresso making process ELI5.<|eot_id|>");
        ctx2.fill("<|start_header_id|>assistant<|end_header_id|>\n\n");

        let output = ctx2.generate_until(max_num_outputs).await;

        println!("Output: {:?} (elapsed: {:?})", output, start.elapsed());
    };

    // wait for both tasks to complete
    //(handle1, handle2).join().await;
    futures::future::join(handle1, handle2).await;

    Ok(())
}
