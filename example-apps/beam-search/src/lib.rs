use inferlet2::traits::Tokenize;
use std::time::{Duration, Instant};

#[inferlet2::main]
async fn main() -> Result<(), String> {
    let start = Instant::now();

    let max_num_outputs = 128;
    let beam_size = 5;

    let model = inferlet2::get_auto_model();
    let tokenizer = model.get_tokenizer();

    let mut ctx = model.create_context();

    let mut stop_condition = inferlet2::stop_condition::any(
        inferlet2::stop_condition::Until::new(tokenizer.tokenize("<|eot_id|>")),
        inferlet2::stop_condition::Length::new(max_num_outputs),
    );

    ctx.fill("<|begin_of_text|>");
    ctx.fill("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|>");
    ctx.fill("<|start_header_id|>user<|end_header_id|>\n\nExplain the LLM decoding process ELI5.<|eot_id|>");
    ctx.fill("<|start_header_id|>assistant<|end_header_id|>\n\n");

    let text = ctx.generate_with_beam(&mut stop_condition, beam_size).await;
    let token_ids = tokenizer.tokenize(&text);
    println!("Output: {:?} (total elapsed: {:?})", text, start.elapsed());

    // compute per token latency
    println!(
        "Per token latency: {:?}",
        start.elapsed() / token_ids.len() as u32
    );

    Ok(())
}
