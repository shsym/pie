#[symphony::main]
async fn main() -> Result<(), String> {
    let prompt = symphony::messaging_async::receive().await;
    let max_num_outputs: usize = symphony::messaging_async::receive()
        .await
        .parse()
        .unwrap_or(32);

    let num_prompts = symphony::messaging_async::receive()
        .await
        .parse()
        .unwrap_or(1);

    let available_models = symphony::available_models();

    let model = symphony::Model::new(available_models.first().unwrap()).unwrap();

    let mut futures = Vec::new();
    for _ in 0..num_prompts {

        let mut ctx = model.create_context();
        let prompt = prompt.clone();
        let future= async move {

            ctx.fill("<|begin_of_text|>");
            ctx.fill("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|>");
            ctx.fill("<|start_header_id|>user<|end_header_id|>\n\n");
            ctx.fill(&prompt);
            ctx.fill("<|eot_id|>");
            ctx.fill("<|start_header_id|>assistant<|end_header_id|>\n\n");

            ctx.generate_until("<|eot_id|>", max_num_outputs).await
        };
        futures.push(future);
    }

    let results = futures::future::join_all(futures).await;
    let text = results.join("\n\n");
    symphony::messaging::send(&text);

    Ok(())
}
