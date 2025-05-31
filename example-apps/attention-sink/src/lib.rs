#[pie::main]
async fn main() -> Result<(), String> {
    let prompt = pie::messaging_async::receive().await;
    let max_num_outputs: usize = pie::messaging_async::receive()
        .await
        .parse()
        .unwrap_or(32);

    let num_prompts = pie::messaging_async::receive()
        .await
        .parse()
        .unwrap_or(1);

    let available_models = pie::available_models();

    let model = pie::Model::new(available_models.first().unwrap()).unwrap();
    let tokenizer = model.get_tokenizer();
    let eot_id = tokenizer.encode("<|eot_id|>")[0];
    let mut futures = Vec::new();
    for _ in 0..num_prompts {
        let mut ctx = model.create_context();
        let prompt = prompt.clone();
        let tokenizer = tokenizer.clone();
        let future = async move {
            ctx.fill("<|begin_of_text|>");
            ctx.fill("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|>");
            ctx.fill("<|start_header_id|>user<|end_header_id|>\n\n");
            ctx.fill(&prompt);
            ctx.fill("<|eot_id|>");
            ctx.fill("<|start_header_id|>assistant<|end_header_id|>\n\n");

            //ctx.generate_until("<|eot_id|>", max_num_outputs).await
            let mut output = Vec::new();

            for _ in 0..max_num_outputs {
                let (token_ids, _) = ctx.next().await;
                let token_id = token_ids[0];
                output.push(token_id);
                if token_id == eot_id {
                    break;
                }
                ctx.apply_sink(1, 8).await;
                ctx.fill_tokens(vec![token_id]);
            }

            tokenizer.decode(&output)
        };
        futures.push(future);
    }

    let results = futures::future::join_all(futures).await;
    let text = results.join("\n\n");
    pie::messaging::send(&text);

    Ok(())
}
