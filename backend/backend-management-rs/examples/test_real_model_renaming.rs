use backend_management_rs::{ModelInstaller, error::Result};
use std::path::PathBuf;
use tokio;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 Testing layer renaming on real Llama 3.1 8B model...");
    println!("💡 You can set SYMPHONY_MODEL_PATH environment variable to specify a custom model path");
    
    // Use environment variable or default path
    let model_path = std::env::var("SYMPHONY_MODEL_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            // Default to standard Symphony cache directory
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".cache")
                .join("symphony")
                .join("models")
                .join("meta-llama--Llama-3.1-8B-Instruct")
        });
    
    // Verify the model path exists
    if !model_path.exists() {
        eprintln!("❌ Model path does not exist: {}", model_path.display());
        return Err(backend_management_rs::ManagementError::Service {
            message: "Model path not found".to_string()
        });
    }
    
    println!("📁 Model path: {}", model_path.display());
    
    // Check if required files exist
    let index_path = model_path.join("model.safetensors.index.json");
    let rules_path = model_path.join("weight_renaming.json");
    
    if !index_path.exists() {
        eprintln!("❌ Missing: {}", index_path.display());
        return Err(backend_management_rs::ManagementError::Service {
            message: "SafeTensors index file not found".to_string()
        });
    }
    
    if !rules_path.exists() {
        eprintln!("❌ Missing: {}", rules_path.display());
        return Err(backend_management_rs::ManagementError::Service {
            message: "Weight renaming rules file not found".to_string()
        });
    }
    
    println!("✅ Found SafeTensors index file");
    println!("✅ Found weight renaming rules file");
    
    // Show some sample layer names before renaming
    let index_content = std::fs::read_to_string(&index_path)?;
    let index_json: serde_json::Value = serde_json::from_str(&index_content)?;
    
    if let Some(weight_map) = index_json.get("weight_map").and_then(|v| v.as_object()) {
        println!("\n📋 Sample current layer names (first 10):");
        for (i, (layer_name, _)) in weight_map.iter().enumerate() {
            if i >= 10 { break; }
            println!("   {}", layer_name);
        }
        println!("   ... and {} more layers", weight_map.len().saturating_sub(10));
    }
    
    // Check total_size in metadata
    if let Some(metadata) = index_json.get("metadata") {
        if let Some(total_size) = metadata.get("total_size") {
            println!("📊 Total model size: {} bytes ({:.2} GB)", 
                total_size, 
                total_size.as_u64().unwrap_or(0) as f64 / (1024.0 * 1024.0 * 1024.0)
            );
        }
    }
    
    // Create ModelInstaller and run layer renaming
    let installer = ModelInstaller::new(None);
    
    println!("\n🔄 Starting layer renaming process...");
    match installer.rename_model_layers(&model_path).await {
        Ok(()) => {
            println!("✅ Layer renaming completed successfully!");
            
            // Show the updated layer names
            let updated_index_content = std::fs::read_to_string(&index_path)?;
            let updated_index: serde_json::Value = serde_json::from_str(&updated_index_content)?;
            
            if let Some(weight_map) = updated_index.get("weight_map").and_then(|v| v.as_object()) {
                println!("\n📋 Sample updated layer names (first 10):");
                for (i, (layer_name, _)) in weight_map.iter().enumerate() {
                    if i >= 10 { break; }
                    println!("   {}", layer_name);
                }
                println!("   ... and {} more layers", weight_map.len().saturating_sub(10));
            }
            
            // Verify metadata is preserved
            if let Some(metadata) = updated_index.get("metadata") {
                if let Some(total_size) = metadata.get("total_size") {
                    println!("✅ Metadata preserved - Total size: {} bytes", total_size);
                }
            }
            
            println!("\n🎉 Real model layer renaming test completed successfully!");
        },
        Err(e) => {
            eprintln!("❌ Layer renaming failed: {}", e);
            return Err(e);
        }
    }
    
    Ok(())
}
