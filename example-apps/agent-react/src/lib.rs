use symphony::wstd::time::Duration;

#[symphony::main]
async fn main() -> Result<(), String> {
    let max_num_outputs = 32;

    let available_models = symphony::available_models();

    // Simulate agentic behavior

    let model = symphony::Model::new(available_models.first().unwrap()).unwrap();

    let mut ctx = model.create_context();

    ctx.fill("<|begin_of_text|>");
    ctx.fill("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|>");
    ctx.fill("<|start_header_id|>user<|end_header_id|>\n\nExplain the LLM decoding process ELI5.<|eot_id|>");
    ctx.fill("<|start_header_id|>assistant<|end_header_id|>\n\n");

    let text = ctx.generate_until("<|eot_id|>", max_num_outputs).await;

    // simulate function calling
    symphony::wstd::task::sleep(Duration::from_millis(100)).await;

    ctx.fill("result from the function call");

    let text = ctx.generate_until("<|eot_id|>", max_num_outputs).await;

    Ok(())
}
