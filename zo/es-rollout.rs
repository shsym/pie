use futures::future::join_all;
use inferlet::{
    self,
    context::Context,
};
use inferlet::traits::Optimize;
use pico_args::Arguments;
use serde::Serialize;
use std::ffi::OsString;
use std::future::Future;
use std::pin::Pin;

/// Defines the command-line interface and help message.
const HELP: &str = r#"
Usage: es-rollout [OPTIONS]

Performs a parallel rollout for an Evolution Strategies adapter. It applies a
unique seed to the model for each batch of tasks and generates outputs.

Options:
      --name <STRING>            The name of the adapter to use.
      --seeds <I64,...>          A comma-separated list of i64 seeds for the adapter.
      --tasks-json <JSON_STRING> A JSON string representing an array of task prompts.
      --max-num-outputs <INT>    The maximum number of new tokens to generate per task.
  -h, --help                     Print this help information.
"#;

#[inferlet::main]
async fn main() -> Result<(), String> {
    // --- 1. Argument Parsing ---
    let mut args = Arguments::from_vec(
        inferlet::get_arguments()
            .into_iter()
            .map(OsString::from)
            .collect(),
    );

    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(());
    }

    // Parse required arguments.
    let name: String = args
        .value_from_str("--name")
        .map_err(|e| e.to_string())?;

    let max_num_outputs: usize = args
        .value_from_str("--max-num-outputs")
        .map_err(|e| e.to_string())?;

    // Parse comma-separated list for seeds.
    let seeds: Vec<i64> = args
        .value_from_fn("--seeds", |s| {
            s.split(',')
                .map(|v| v.parse::<i64>().map_err(|_| "Invalid seed value"))
                .collect::<Result<Vec<_>, _>>()
        })
        .map_err(|e| e.to_string())?;

    // Parse the tasks from a JSON string.
    let tasks_json: String = args
        .value_from_str("--tasks-json")
        .map_err(|e| e.to_string())?;
    let tasks: Vec<String> = serde_json::from_str(&tasks_json)
        .map_err(|e| format!("Failed to parse tasks JSON: {}", e))?;


    if tasks.len() != seeds.len() {
        return Err(format!(
            "The number of tasks ({}) must the same as the number of seeds ({}).",
            tasks.len(),
            seeds.len()
        ));
    }

    // The futures vector for a single-threaded (Wasm) environment does not need the `Send` bound.
    let mut futures: Vec<Pin<Box<dyn Future<Output = String>>>> = vec![];

    // import the main adapter
    let model = inferlet::get_auto_model();
    let queue = model.create_queue();

    let es_adapter = queue.import_adapter(&name);

    println!("🚀 Starting parallel rollout...");
    for i in 0..seeds.len() {

        let task = tasks[i].clone();
        let seed = seeds[i].clone();

        //println!("task: {}", &task);

        // Get a new model instance for each seed.
        let mut model_with_adapter = model.clone();
        model_with_adapter.set_adapter(es_adapter, seed);

        let mut ctx = model_with_adapter.create_context();

        // Create an async block that owns the forked context and the task string.
        let generation_future = async move {
            ctx.fill(&task);
            ctx.generate_until("<|eot_id|>", max_num_outputs).await
        };

        // Box the future to store it in the vector with other futures.
        futures.push(Box::pin(generation_future));

    }

    // --- 4. Collect Results and Send ---
    println!("⏳ Waiting for {} tasks to complete...", futures.len());
    let results: Vec<String> = join_all(futures).await;

    // Serialize the collected text outputs into a JSON string array.
    let response_json = serde_json::to_string(&results)
        .map_err(|e| format!("Failed to serialize final results: {}", e))?;

    // Send the final JSON response back to the caller.
    inferlet::send(&response_json);
    println!("✅ Rollout complete. Sent {} results.", results.len());

    Ok(())
}
