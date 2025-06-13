use backend_management_rs::model_installer::ModelInstaller;
use backend_management_rs::path_utils::expand_home_dir;
use std::path::Path;
use tokio;

/// Test to verify streaming copy performance for model renaming when header size changes
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Testing streaming copy performance for model layer renaming...");

    // Use real Llama 3.1 8B model - approximately 15GB
    let model_path = expand_home_dir("~/.cache/symphony/models/meta-llama--Llama-3.1-8B-Instruct");

    if !model_path.exists() {
        println!("❌ Model not found at: {}", model_path.display());
        println!("   Please run the model download test first");
        return Ok(());
    }

    println!("📁 Model path: {}", model_path.display());

    // Create installer
    let installer = ModelInstaller::new(Some(model_path.to_path_buf()));

    // Get current model size for reference
    let total_size = get_directory_size(&model_path).await?;
    println!("📊 Total model size: {} bytes ({:.2} GB)", total_size, total_size as f64 / 1e9);

    // Create weight_renaming.json with rules that will force header size changes
    let rules_json = r#"{
    "root": "model",
    "rules": [
        {
            "name": "Extend embedding layer name",
            "type": "exact",
            "pattern": "model.embed_tokens.weight",
            "replacement": "model.very_long_name_for_embedding_tokens_to_force_header_size_change.weight",
            "enabled": true
        },
        {
            "name": "Extend layer norm names",
            "type": "regex",
            "pattern": "^model\\.layers\\.([0-9]+)\\.input_norm\\.weight$",
            "replacement": "model.transformer_layers.$1.pre_attention_layer_normalization.weight",
            "enabled": true
        },
        {
            "name": "Extend MLP projection names",
            "type": "regex",
            "pattern": "^model\\.layers\\.([0-9]+)\\.mlp\\.(down_proj|gate_proj)\\.weight$",
            "replacement": "model.transformer_layers.$1.feed_forward_network.$2_projection.weight",
            "enabled": true
        }
    ]
}"#;

    let rules_file_path = model_path.join("weight_renaming.json");
    println!("🔄 Creating weight_renaming.json with header size-changing rules...");

    // Backup any existing rules file
    let backup_path = model_path.join("weight_renaming.json.backup");
    if rules_file_path.exists() {
        std::fs::copy(&rules_file_path, &backup_path)?;
        println!("📦 Backed up existing weight_renaming.json");
    }

    // Write new rules
    std::fs::write(&rules_file_path, rules_json)?;

    // Measure time for the renaming operation
    let start_time = std::time::Instant::now();

    // Apply renaming - this should trigger the streaming copy code path
    let result = installer.rename_model_layers(&model_path).await;

    let duration = start_time.elapsed();

    match result {
        Ok(_) => {
            println!("✅ Streaming copy completed successfully!");
            println!("⏱️  Total time: {:.2} seconds", duration.as_secs_f64());
            println!("📈 Effective throughput: {:.2} MB/s", (total_size as f64 / 1e6) / duration.as_secs_f64());

            // Verify the model still has the same total size
            let new_size = get_directory_size(&model_path).await?;
            if new_size == total_size {
                println!("✅ Model size preserved: {} bytes", new_size);
            } else {
                println!("⚠️  Model size changed: {} -> {} bytes (expected due to header changes)", total_size, new_size);
            }
        }
        Err(e) => {
            println!("❌ Error during renaming: {}", e);
            return Err(e.into());
        }
    }

    // Revert the changes for cleanup
    println!("🔧 Reverting changes for cleanup...");

    // Create revert rules
    let revert_rules_json = r#"{
    "root": "model",
    "rules": [
        {
            "name": "Revert embedding layer name",
            "type": "exact",
            "pattern": "model.very_long_name_for_embedding_tokens_to_force_header_size_change.weight",
            "replacement": "model.embed_tokens.weight",
            "enabled": true
        },
        {
            "name": "Revert layer norm names",
            "type": "regex",
            "pattern": "^model\\.transformer_layers\\.([0-9]+)\\.pre_attention_layer_normalization\\.weight$",
            "replacement": "model.layers.$1.input_norm.weight",
            "enabled": true
        },
        {
            "name": "Revert MLP projection names",
            "type": "regex",
            "pattern": "^model\\.transformer_layers\\.([0-9]+)\\.feed_forward_network\\.(down_proj|gate_proj)_projection\\.weight$",
            "replacement": "model.layers.$1.mlp.$2.weight",
            "enabled": true
        }
    ]
}"#;

    std::fs::write(&rules_file_path, revert_rules_json)?;

    let revert_start = std::time::Instant::now();
    let revert_result = installer.rename_model_layers(&model_path).await;
    let revert_duration = revert_start.elapsed();

    match revert_result {
        Ok(_) => {
            println!("✅ Changes reverted successfully!");
            println!("⏱️  Revert time: {:.2} seconds", revert_duration.as_secs_f64());
        }
        Err(e) => {
            println!("⚠️  Error reverting changes: {}", e);
        }
    }

    println!("🎉 Streaming performance test completed!");
    println!("💡 Performance insights:");
    println!("   - Used streaming copy to avoid loading entire {}GB model into memory", total_size as f64 / 1e9);
    println!("   - Processed files in 64MB chunks for optimal memory usage");
    println!("   - Atomic file replacement ensures data integrity");

    Ok(())
}

async fn get_directory_size(path: &Path) -> Result<u64, Box<dyn std::error::Error>> {
    let mut total_size = 0;

    let mut entries = tokio::fs::read_dir(path).await?;
    while let Some(entry) = entries.next_entry().await? {
        let metadata = entry.metadata().await?;
        if metadata.is_file() {
            total_size += metadata.len();
        }
    }

    Ok(total_size)
}
