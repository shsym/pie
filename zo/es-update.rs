use inferlet::{self};
use inferlet::traits::Optimize;
use pico_args::Arguments;
use std::ffi::OsString;
use inferlet::wstd::time::Duration;

/// Defines the command-line interface and help message.
const HELP: &str = r#"
Usage: es-update [OPTIONS]

Updates an Evolution Strategies (ES) adapter with new scores for a given set of seeds.

Options:
      --name <STRING>            The name of the adapter to update.
      --seeds <I64,...>          A comma-separated list of i64 seeds corresponding to the rollouts.
      --scores <F32,...>         A comma-separated list of f32 scores for each rollout.
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

    // Parse comma-separated lists for seeds and scores.
    let seeds: Vec<i64> = args
        .value_from_fn("--seeds", |s| {
            s.split(',')
                .map(|v| v.parse::<i64>().map_err(|_| "Invalid seed value"))
                .collect::<Result<Vec<_>, _>>()
        })
        .map_err(|e| e.to_string())?;

    let scores: Vec<f32> = args
        .value_from_fn("--scores", |s| {
            s.split(',')
                .map(|v| v.parse::<f32>().map_err(|_| "Invalid score value"))
                .collect::<Result<Vec<_>, _>>()
        })
        .map_err(|e| e.to_string())?;

    // --- 2. Input Validation ---
    if seeds.is_empty() {
        return Err("At least one seed and score must be provided.".to_string());
    }
    if seeds.len() != scores.len() {
        return Err(format!(
            "The number of seeds ({}) must match the number of scores ({}).",
            seeds.len(),
            scores.len()
        ));
    }

    // --- 3. Adapter Update ---
    println!("🔧 Initializing model and queue to update adapter '{}'...", name);
    let model = inferlet::get_auto_model();
    let queue = model.create_queue();

    println!(
        "Updating adapter '{}' with {} scores...",
        name,
        scores.len()
    );

    // Perform the update operation.
    queue.update_adapter(&name, &scores, &seeds);

    // sleep for 100ms
    inferlet::wstd::task::sleep(Duration::from_millis(100)).await;

    println!("✅ Adapter '{}' updated successfully.", name);

    Ok(())
}
