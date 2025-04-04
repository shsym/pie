use std::time::{Duration, Instant};

#[symphony::main]
async fn main() -> Result<(), String> {
    const PING_COUNT: usize = 5;
    let mut total_duration = Duration::ZERO;

    // println!("Starting ping test with {} iterations...", PING_COUNT);

    // for i in 1..=PING_COUNT {
    //     let start = Instant::now();
    //     let resp = symphony::ping::ping("hello");
    //     let elapsed = start.elapsed();
    //     total_duration += elapsed;

    //     //println!("Ping {}: Response: {:?}, Latency: {:?}", i, resp, elapsed);
    // }

    // let avg_latency = total_duration / PING_COUNT as u32;
    // //println!("\nPing test completed.");
    // //println!("Total Pings: {}", PING_COUNT);
    // println!("Average Latency: {:?}", avg_latency);

    //let mut prev = start; // track the time of the previous token
    let start = Instant::now();

    let max_num_outputs = 32;

    let available_models = symphony::available_models();

    let model = symphony::Model::new(available_models.first().unwrap()).unwrap();
    let tokenizer = model.get_tokenizer();

    let mut ctx = model.create_context();

    ctx.fill("<|begin_of_text|>");
    ctx.fill("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|>");
    ctx.fill("<|start_header_id|>user<|end_header_id|>\n\nExplain the LLM decoding process ELI5.<|eot_id|>");
    ctx.fill("<|start_header_id|>assistant<|end_header_id|>\n\n");

    let text = ctx.generate_until("<|eot_id|>", max_num_outputs).await;
    let token_ids = tokenizer.encode(&text);
    println!("Output: {:?} (total elapsed: {:?})", text, start.elapsed());

    // compute per token latency
    println!(
        "Per token latency: {:?}",
        start.elapsed() / token_ids.len() as u32
    );

    Ok(())
}
