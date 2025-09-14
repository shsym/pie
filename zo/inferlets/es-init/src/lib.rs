use inferlet::interface::Adapter;
use inferlet::wstd::time::Duration;
use inferlet::{
    self, Args, Blob, Result, get_auto_model, interface::Evolve, store_exists, store_set,
};

#[inferlet::main]
async fn main(mut args: Args) -> Result<()> {
    let name: String = args.value_from_str("--name")?;
    let rank: u32 = args.value_from_str("--rank")?;
    let alpha: f32 = args.value_from_str("--alpha")?;
    let population_size: u32 = args.value_from_str("--population-size")?;
    let mu_fraction: f32 = args.value_from_str("--mu-fraction")?;
    let initial_sigma: f32 = args.value_from_str("--initial-sigma")?;
    let upload: Option<String> = args.opt_value_from_str("--upload")?;

    // --- 2. Initialization and Adapter Creation ---
    let model = get_auto_model();
    let queue = model.create_queue();

    let adapter_id = if !store_exists(&name) {
        let adapter_id = queue.allocate_adapter();
        println!("🔧 Initializing adapter...");

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
        store_set(&name, "true");

        adapter_id
    } else {
        println!("🔧 Existing adapter found. Importing adapter...");

        queue.import_adapter(&name)
    };

    // If the --upload argument was provided with a non-empty string, upload the blob.
    if let Some(upload_file_name) = upload {
        if !upload_file_name.is_empty() {
            println!(
                "🚀 Uploading to adapter '{}' with filename '{}'...",
                name, upload_file_name
            );
            queue.upload_adapter(adapter_id, &upload_file_name, Blob::new(vec![]));
        }
    }

    inferlet::wstd::task::sleep(Duration::from_millis(100)).await;
    println!("✅ Adapter '{}' created or imported successfully.", name);

    Ok(())
}
