
#[symphony::main]
async fn main() -> Result<(), String> {
    let prompt = symphony::messaging_async::receive().await;
    let max_num_outputs: usize = symphony::messaging_async::receive().await.parse().unwrap_or(32);
    
    let available_models = symphony::available_models();

    let model = symphony::Model::new(available_models.first().unwrap()).unwrap();

    let mut ctx = model.create_context();

    ctx.fill("<|begin_of_text|>").await;
    ctx.fill("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|>").await;
    ctx.fill("<|start_header_id|>user<|end_header_id|>\n\n").await;
    ctx.fill(&prompt).await;
    ctx.fill("<|eot_id|>").await;
    ctx.fill("<|start_header_id|>assistant<|end_header_id|>\n\n")
        .await;

    let text = ctx.generate_until("<|eot_id|>", max_num_outputs).await;
    symphony::messaging::send(&text);

    Ok(())
}
