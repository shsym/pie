use inferlet::wstd::time::Duration;
use inferlet::{self, Adapter, Args, Evolve, Result, bail};

#[inferlet::main]
async fn main(mut args: Args) -> Result<()> {
    // Parse required arguments.
    let name: String = args.value_from_str("--name")?;
    let seeds: Vec<i64> = args.value_from_fn("--seeds", |s| {
        s.split(',')
            .map(|v| v.parse::<i64>())
            .collect::<Result<Vec<_>, _>>()
    })?;
    let scores: Vec<f32> = args.value_from_fn("--scores", |s| {
        s.split(',')
            .map(|v| v.parse::<f32>())
            .collect::<Result<Vec<_>, _>>()
    })?;
    let max_sigma: f32 = args.value_from_str("--max-sigma")?;
    let download: Option<String> = args.opt_value_from_str("--download")?;

    // --- 2. Input Validation ---
    if seeds.is_empty() {
        bail!("At least one seed and score must be provided.");
    }
    if seeds.len() != scores.len() {
        bail!(
            "The number of seeds ({}) must match the number of scores ({}).",
            seeds.len(),
            scores.len()
        );
    }

    // --- 3. Adapter Update ---
    println!(
        "🔧 Initializing model and queue to update adapter '{}'...",
        &name
    );
    let model = inferlet::get_auto_model();
    let queue = model.create_queue();

    println!(
        "Updating adapter '{}' with {} scores (max_sigma = {})...",
        &name,
        scores.len(),
        max_sigma
    );
    let es_adapter = queue.import_adapter(&name);

    // Perform the update operation with max_sigma.
    queue.update_adapter(es_adapter, scores, seeds, max_sigma);

    // If a download path was provided, download the adapter.
    if let Some(download_path) = &download {
        if !download_path.is_empty() {
            println!(
                "📥 Downloading adapter '{}' to '{}'...",
                name, download_path
            );
            let _ = queue.download_adapter(es_adapter, download_path).await;
        }
    }

    // sleep for 100ms
    inferlet::wstd::task::sleep(Duration::from_millis(100)).await;

    println!("✅ Adapter '{}' updated successfully.", name);

    Ok(())
}
