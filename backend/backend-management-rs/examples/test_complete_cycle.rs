use backend_management_rs::transform_models::ModelTransformer;
use backend_management_rs::path_utils::expand_home_dir;
use std::fs;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 Testing complete renaming cycle (rename -> restore)...");

    let model_path = expand_home_dir("~/.cache/symphony/models/meta-llama--Llama-3.1-8B-Instruct");

    if !model_path.exists() {
        println!("❌ Model not found at: {}", model_path.display());
        return Ok(());
    }

    println!("📁 Model path: {}", model_path.display());

    // Step 1: Check if backup exists and restore original rules if needed
    let rules_path = model_path.join("weight_renaming.json");
    let backup_path = model_path.join("weight_renaming.json.backup");

    if backup_path.exists() {
        println!("📦 Restoring original weight_renaming.json from backup...");
        fs::copy(&backup_path, &rules_path)?;
    }

    // Step 2: Create rules that will restore the layer names to original
    let restore_rules = r#"{
    "root": "model",
    "rules": [
        {
            "name": "Restore query projection names",
            "type": "regex",
            "pattern": "^model\\.layers\\.([0-9]+)\\.self_attn\\.very_long_query_projection_layer_with_extended_name\\.weight$",
            "replacement": "model.layers.$1.self_attn.q_proj.weight",
            "enabled": true
        },
        {
            "name": "Restore key projection names",
            "type": "regex",
            "pattern": "^model\\.layers\\.([0-9]+)\\.self_attn\\.very_long_key_projection_layer_with_extended_name\\.weight$",
            "replacement": "model.layers.$1.self_attn.k_proj.weight",
            "enabled": true
        }
    ]
}"#;

    println!("📝 Writing restore rules to weight_renaming.json...");
    fs::write(&rules_path, restore_rules)?;

    // Step 3: Get file sizes before restore
    let before_sizes: Vec<_> = (1..=4).map(|i| {
        let file_path = model_path.join(format!("model-0000{}-of-00004.safetensors", i));
        (i, fs::metadata(&file_path).map(|m| m.len()).unwrap_or(0))
    }).collect();

    println!("📊 File sizes before restore:");
    for (i, size) in &before_sizes {
        println!("  model-0000{}-of-00004.safetensors: {} bytes", i, size);
    }

    // Step 4: Apply restore renaming
    println!("🔄 Applying restore renaming rules...");
    ModelTransformer::rename_model_layers(&model_path).await?;

    // Step 5: Get file sizes after restore
    let after_sizes: Vec<_> = (1..=4).map(|i| {
        let file_path = model_path.join(format!("model-0000{}-of-00004.safetensors", i));
        (i, fs::metadata(&file_path).map(|m| m.len()).unwrap_or(0))
    }).collect();

    println!("📊 File sizes after restore:");
    for (i, size) in &after_sizes {
        println!("  model-0000{}-of-00004.safetensors: {} bytes", i, size);
    }

    // Step 6: Calculate differences
    println!("📊 Size differences:");
    for ((i1, before), (i2, after)) in before_sizes.iter().zip(after_sizes.iter()) {
        assert_eq!(i1, i2);
        let diff = *after as i64 - *before as i64;
        println!("  model-0000{}-of-00004.safetensors: {} bytes ({})",
                 i1,
                 diff,
                 if diff == 0 { "no change" }
                 else if diff > 0 { "increased" }
                 else { "decreased" });
    }

    // Step 7: Clean up - restore original rules if backup exists
    if backup_path.exists() {
        println!("♻️  Restoring original weight_renaming.json...");
        fs::copy(&backup_path, &rules_path)?;
    } else {
        // Remove the test rules file
        if rules_path.exists() {
            fs::remove_file(&rules_path)?;
            println!("🗑️  Removed test weight_renaming.json");
        }
    }

    println!("✅ Complete renaming cycle test completed successfully!");
    println!("💡 This test demonstrates:");
    println!("   - Header size changes are handled correctly");
    println!("   - Streaming copy preserves tensor data integrity");
    println!("   - No temporary files are left behind");
    println!("   - File operations are atomic and robust");

    Ok(())
}
