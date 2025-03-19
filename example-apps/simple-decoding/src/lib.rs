use std::time::Instant;

#[symphony::main]
async fn main() -> Result<(), String> {
    let start = Instant::now();
    //let mut prev = start; // track the time of the previous token

    let max_num_outputs = 256;

    let available_models = symphony::available_models();

    println!("Available models: {:?}", available_models);

    let model = symphony::Model::new(symphony::available_models().first().unwrap()).unwrap();
    let tokenizer = model.get_tokenizer();

    let mut ctx = model.create_context();

    ctx.fill("<|begin_of_text|>").await;
    ctx.fill("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|>").await;
    ctx.fill("<|start_header_id|>user<|end_header_id|>\n\nExplain the LLM decoding process ELI5.<|eot_id|>").await;
    ctx.fill("<|start_header_id|>assistant<|end_header_id|>\n\n")
        .await;

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
