use inferlet::stop_condition::{self, StopCondition};

#[inferlet::main]
async fn main() -> Result<(), String> {
    let prompt = inferlet::messaging_async::receive().await;
    let max_num_outputs = inferlet::messaging_async::receive()
        .await
        .parse()
        .unwrap_or(32);
    let num_prompts = inferlet::messaging_async::receive()
        .await
        .parse()
        .unwrap_or(1);

    let available_models = inferlet::available_models();

    let model = inferlet::Model::new(available_models.first().unwrap()).unwrap();
    let tokenizer = model.get_tokenizer();
    let stop_str_token_ids = tokenizer.encode("<|eot_id|>");

    let mut futures = Vec::new();
    for _ in 0..num_prompts {
        let mut ctx = model.create_context();
        let prompt = prompt.clone();
        let stop_str_token_ids = stop_str_token_ids.clone();
        let tokenizer = tokenizer.clone();
        let future = async move {
            ctx.fill("<|begin_of_text|>");
            ctx.fill("<|start_header_id|>system<|end_header_id|>\n\n.<|eot_id|>");
            ctx.fill("<|start_header_id|>user<|end_header_id|>\n\n");
            ctx.fill(&prompt);
            ctx.fill("<|eot_id|>");
            ctx.fill("<|start_header_id|>assistant<|end_header_id|>\n\n");

            let mut cond = stop_condition::any(
                stop_condition::Until::new(stop_str_token_ids),
                stop_condition::Length::new(max_num_outputs),
            );
            let beam_size = 3;

            let mut beams = Vec::new();
            beams.push((ctx.fork(), vec![], 0.0));

            let mut final_generated = String::new();
            let mut break_flag = false;

            //let mut i = 0;
            loop {
                //i += 1;
                //println!("i: {}", i);

                // check if we meets the stop condition
                for (_, generated, _) in beams.iter() {
                    if cond.should_stop(&generated) {
                        final_generated = tokenizer.decode(generated);
                        break_flag = true;
                        break;
                    }
                }
                if break_flag {
                    break;
                }

                let mut next_beams = Vec::new();
                for (mut beam, generated, score) in beams.into_iter() {
                    // println!("--------------------------------");
                    // println!("pending before next: {:?}", beam.pending_token_ids);

                    let (next_token_ids, next_score) = beam.next().await;
                    //println!("pending before fork: {:?}", beam.pending_token_ids);

                    for i in 0..beam_size {
                        let mut next_beam = beam.fork_unsafe();
                        //println!("pending after fork: {:?}", next_beam.pending_token_ids);

                        next_beam.pending_token_ids.push(next_token_ids[i]);
                        let next_generated = [generated.as_slice(), &[next_token_ids[i]]].concat();
                        let next_score = score + next_score[i];
                        next_beams.push((next_beam, next_generated, next_score));
                        //println!("pending after fork2: {:?}", next_beam.pending_token_ids);

                    }

                    // last next_beam pending_token_ids is empty
                    //println!("pending after next: {:?}", next_beams.last().unwrap().0.pending_token_ids);
                }

                // get the top k beams by score in next_beam_cands
                // sort the next_beam_cands by score
                // and only keep the first beam_size beams
                next_beams.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
                next_beams.truncate(beam_size);
                //println!("next_beams: {:?}", next_beams);

                beams = next_beams;
            }

            return final_generated;
        };
        futures.push(future);
    }

    let results = futures::future::join_all(futures).await;
    let text = results.join("\n\n");
    inferlet::messaging::send(&text);

    Ok(())
}
