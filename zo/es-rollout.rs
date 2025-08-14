use futures::future::join_all;
use inferlet::{
    self,
    context::Context,
};
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
      --prefix <STRING>          The text prefix to use for all tasks.
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
    let prefix: String = args
        .value_from_str("--prefix")
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


    // --- 2. Input Validation ---
    if seeds.is_empty() {
        return Err("At least one seed must be provided.".to_string());
    }
    // This check ensures that the tasks can be evenly distributed among the seeds.
    if tasks.len() % seeds.len() != 0 {
        return Err(format!(
            "The number of tasks ({}) must be a multiple of the number of seeds ({}).",
            tasks.len(),
            seeds.len()
        ));
    }

    // --- 3. Parallel Rollout ---
    let num_tasks_per_seed = tasks.len() / seeds.len();
    // The futures vector for a single-threaded (Wasm) environment does not need the `Send` bound.
    let mut futures: Vec<Pin<Box<dyn Future<Output = String>>>> = vec![];

    println!("🚀 Starting parallel rollout...");
    for (i, &seed) in seeds.iter().enumerate() {
        // Get a new model instance for each seed.
        let mut model = inferlet::get_auto_model();
        // Apply the adapter with the current seed.
        model.set_adapter(&name, seed);

        // Create the base context from scratch using the provided prefix.
        let mut base_ctx = model.create_context();
        base_ctx.fill(&prefix);
        base_ctx.flush(); // Ensure the prefix KV cache is computed before forking.

        // Get the slice of tasks assigned to this seed.
        let start_index = i * num_tasks_per_seed;
        let end_index = (i + 1) * num_tasks_per_seed;
        let assigned_tasks = &tasks[start_index..end_index];

        // For each task, create a future that owns its context.
        for task in assigned_tasks {
            let mut forked_ctx = base_ctx.fork();
            let task_owned = task.clone(); // Clone the task to move it into the async block.

            // Create an async block that owns the forked context and the task string.
            // This solves the lifetime problem.
            let generation_future = async move {
                forked_ctx.fill(&task_owned);
                forked_ctx.generate_until("<|eot_id|>", max_num_outputs).await
            };

            // Box the future to store it in the vector with other futures.
            futures.push(Box::pin(generation_future));
        }
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
