use inferlet::traits::Adapter;
use inferlet::wstd::time::Duration;
use inferlet::{self, get_auto_model, store_exists, store_set, traits::Evolve};
use pico_args::Arguments;
use std::ffi::OsString;

/// Defines the command-line interface and help message.
const HELP: &str = r#"
Usage: es-init [OPTIONS]

Initializes an Evolution Strategies (ES) adapter and caches a text prefix
to accelerate future inference runs. All arguments are required.

Options:
      --name <STRING>            The unique name for the adapter.
      --rank <INT>               The rank of the LoRA-like adapter matrices.
      --alpha <FLOAT>            The alpha scaling factor for the adapter.
      --population-size <INT>    The number of individuals in the ES population.
      --mu-fraction <FLOAT>      The fraction of top performers to use for updates (0.0 to 1.0).
      --initial-sigma <FLOAT>    The initial standard deviation for the ES noise.
  -h, --help                     Print this help information.
"#;

#[inferlet::main]
async fn main() -> Result<(), String> {
    // --- 1. Argument Parsing ---
    // Fetch command-line arguments provided by the inferlet runtime.
    let mut args = Arguments::from_vec(
        inferlet::get_arguments()
            .into_iter()
            .map(OsString::from)
            .collect(),
    );

    // Handle the --help flag. If present, print the help message and exit.
    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(());
    }

    // Parse all required arguments for creating the adapter and caching the prefix.
    // The .map_err() converts parsing errors into a user-friendly String format.
    let name: String = args.value_from_str("--name").map_err(|e| e.to_string())?;
    let rank: u32 = args.value_from_str("--rank").map_err(|e| e.to_string())?;
    let alpha: f32 = args.value_from_str("--alpha").map_err(|e| e.to_string())?;
    let population_size: u32 = args
        .value_from_str("--population-size")
        .map_err(|e| e.to_string())?;
    let mu_fraction: f32 = args
        .value_from_str("--mu-fraction")
        .map_err(|e| e.to_string())?;
    let initial_sigma: f32 = args
        .value_from_str("--initial-sigma")
        .map_err(|e| e.to_string())?;

    // Ensure no unknown arguments were passed.
    let remaining = args.finish();
    if !remaining.is_empty() {
        return Err(format!(
            "Unknown arguments found: {:?}. Use --help for usage.",
            remaining
        ));
    }

    // --- 2. Initialization and Adapter Creation ---
    println!("🔧 Initializing model and queue...");
    let model = get_auto_model();
    let queue = model.create_queue();

    if !store_exists(&name) {
        let adapter_id = queue.allocate_adapter();

        // Create the Evolution Strategies adapter with the specified hyperparameters.
        queue.initialize_adapter(
            adapter_id,
            rank,
            alpha,
            population_size,
            mu_fraction,
            initial_sigma,
        );

        queue.export_adapter(adapter_id, &name);
        //store_set(&name, "");
    }

    // queue.upload_adapter(adapter_id, "")

    inferlet::wstd::task::sleep(Duration::from_millis(100)).await;
    println!("✅ Adapter '{}' created successfully.", name);

    Ok(())
}
