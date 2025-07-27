use inferlet2::traits::Tokenize;
use std::time::{Duration, Instant};

#[inferlet2::main]
async fn main() -> Result<(), String> {
    let start = Instant::now();

    let max_num_outputs = 128;

    let available_models = inferlet2::get_all_models();

    let model = inferlet2::get_model(available_models.first().unwrap()).unwrap();
    let queue = model.create_queue();
    let tokenizer = queue.get_tokenizer();

    let mut ctx = inferlet2::Context::new(&model);

    ctx.fill("<|begin_of_text|>");
    ctx.fill("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|>");
    ctx.fill("<|start_header_id|>user<|end_header_id|>\n\nExplain the LLM decoding process ELI5.<|eot_id|>");
    ctx.fill("<|start_header_id|>assistant<|end_header_id|>\n\n");

    let text = ctx.generate_until("<|eot_id|>", max_num_outputs).await;
    let token_ids = tokenizer.tokenize(&text);
    println!("Output: {:?} (total elapsed: {:?})", text, start.elapsed());

    // compute per token latency
    println!(
        "Per token latency: {:?}",
        start.elapsed() / token_ids.len() as u32
    );

    Ok(())
}
