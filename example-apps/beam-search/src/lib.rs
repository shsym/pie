use symphony::stop_condition;

#[symphony::main]
async fn main() -> Result<(), String> {
    let max_num_outputs = 32;

    let available_models = symphony::available_models();

    let model = symphony::Model::new(available_models.first().unwrap()).unwrap();
    let tokenizer = model.get_tokenizer();

    let mut ctx = model.create_context();

    ctx.fill("<|begin_of_text|>");
    ctx.fill("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|>");
    ctx.fill("<|start_header_id|>user<|end_header_id|>\n\nExplain the LLM decoding process ELI5.<|eot_id|>");
    ctx.fill("<|start_header_id|>assistant<|end_header_id|>\n\n");

    let stop_str_token_ids = tokenizer.encode("<|eot_id|>");

    let mut cond = stop_condition::any(
        stop_condition::Until::new(stop_str_token_ids),
        stop_condition::Length::new(max_num_outputs),
    );

    let text = ctx.generate_with_beam(&mut cond, 5).await;

    Ok(())
}
