use std::time::{Duration, Instant};

#[pie::main]
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

    let available_models = pie::available_models();

    let model = pie::Model::new(available_models.first().unwrap()).unwrap();
    let tokenizer = model.get_tokenizer();

    let mut ctx = model.create_context();

    ctx.fill("<｜begin▁of▁sentence｜>\n");
    ctx.fill("You are a helpful, respectful and honest assistant.\n");
    ctx.fill("<｜User｜> Explain the LLM decoding process ELI5.\n");
    ctx.fill("<｜Assistant｜> ");
    // ctx.fill("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWrite a Python function to calculate factorial.<|im_end|>\n<|im_start|>assistant\n");

    let text = ctx.generate_until("<｜end▁of▁sentence｜>", max_num_outputs).await;
    let token_ids = tokenizer.encode(&text);
    println!("Output: {:?} (total elapsed: {:?})", text, start.elapsed());

    // compute per token latency
    println!(
        "Per token latency: {:?}",
        start.elapsed() / token_ids.len() as u32
    );

    Ok(())
}
