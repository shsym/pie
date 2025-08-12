use pico_args::Arguments;
use std::ffi::OsString;
use std::fmt;
use std::str::FromStr;
use inferlet::traits::Optimize;

#[inferlet::main]
async fn main() -> Result<(), String> {

    let mut model = inferlet::get_auto_model();
    let queue = model.create_queue();

    let tokenizer = model.get_tokenizer();

    //queue.update_adapter("evo", scores, seeds)
    model.set_adapter("evo",13513513);

    // if command == "initialize" {
    //     // create an mutable adapter with the specified name.
    //     should be only used once.
     // CLI input: "name"

    //     queue.create_adapter("evo");
    //
    //     // also create a prefix cache
    //
    // }
    //
    // else if command == "rollout" {
    //
    //     // do rollout with a random perturbation, or without perturbation (for evaluation)
    //     13513513 is a received seed
    //     questio
    //     model.set_adapter("evo", 13513513);
    //
    //     // do a fork on a cached context.
    //     for each zip(question, seed):
    //         future.push(ctx.generate_until("<|eot_id|>", max_num_outputs).await;
    ///    // await the future
    ///    inferlet::send the output in json
    // }
    //
    // else if command == "update" {
    //     // update the population with the results of the rollouts
    //     input = seeds:list[i64], scores:list[f32]
    //     queue.update_adapter("evo", &scores, &seeds);
    // }
    //

    
    
    
    

    Ok(())
}
