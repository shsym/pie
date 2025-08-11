use pico_args::Arguments;
use std::ffi::OsString;
use std::fmt;
use std::str::FromStr;

#[inferlet::main]
async fn main() -> Result<(), String> {


    // overall flow - 1 inferlet per each node

    // 1. create an adapter (set the population size, etc).
    // 2. prepare the dataloader
    // in the training loop:
    //      1. receive n random questions per population (random seed)
    //      2. do rollout with random seeds
    //      3. evaluate the score of the random seed
    //      4. broadcast the score with others -> just send the score and seed back to the user
    //      5. receive the score from others
    //      6. update the population

    
    
    
    

    Ok(())
}
