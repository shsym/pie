use pie::wstd::time::Duration;

#[pie::main]
async fn main() -> Result<(), String> {

    let input_prompt = pie::messaging_async::receive()
        .await
        .parse()
        .unwrap_or("".to_string());

    let max_num_outputs = pie::messaging_async::receive()
        .await
        .parse()
        .unwrap_or(32);
    let num_fc = pie::messaging_async::receive()
        .await
        .parse()
        .unwrap_or(1);

    let num_insts = pie::messaging_async::receive()
        .await
        .parse()
        .unwrap_or(1);


    let use_cache = pie::messaging_async::receive()
        .await
        .parse()
        .unwrap_or(false);

    let use_asyncfc = pie::messaging_async::receive()
        .await
        .parse()
        .unwrap_or(false);

    let use_ctx_mask = pie::messaging_async::receive()
        .await
        .parse()
        .unwrap_or(false);

    let available_models = pie::available_models();

    // Simulate agentic behavior

    let model = pie::Model::new(available_models.first().unwrap()).unwrap();


    let mut cache_ctx = if use_cache {
        let mut cache_ctx = model.create_context();
        cache_ctx.fill("<|begin_of_text|>");
        cache_ctx.fill("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|>");
        cache_ctx.fill("<|start_header_id|>user<|end_header_id|>\n\n");
        cache_ctx.fill(&input_prompt);
        cache_ctx.fill("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");
        cache_ctx.flush();
        Some(cache_ctx)
    } else {
        None
    };


    let mut futures = Vec::new();
    for _ in 0..num_insts {
        let mut ctx = if let Some(cache_ctx) = &cache_ctx {
            cache_ctx.fork_unsafe()
        } else {
            let mut ctx = model.create_context();
            ctx.fill("<|begin_of_text|>");
            ctx.fill("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|>");
            ctx.fill("<|start_header_id|>user<|end_header_id|>\n\n");
            ctx.fill(&input_prompt);
            ctx.fill("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");
            ctx
        };
        let future = async move {


            if use_asyncfc {
                for _ in 0..(max_num_outputs * num_fc) {
                    let (token_ids, _) = ctx.next().await;
                    let next_token = token_ids.first().unwrap();
                    ctx.fill_token(*next_token);
        
                    if use_ctx_mask {
                        ctx.apply_sink(1, 1).await;
                    }
                }
                return ctx.get_text();
            }
            else {

                for _ in 0..num_fc {
                    for _ in 0..max_num_outputs {
                        let (token_ids, _) = ctx.next().await;
                        let next_token = token_ids.first().unwrap();
                        ctx.fill_token(*next_token);
            
                        if use_ctx_mask {
                            ctx.apply_sink(1, 1).await;
                        }
                    }

                    // simulate function calling
                    pie::wstd::task::sleep(Duration::from_millis(100)).await;
                }
                return ctx.get_text();
            }
        
        };
        futures.push(future);
    }

    let results = futures::future::join_all(futures).await;


    Ok(())
}
