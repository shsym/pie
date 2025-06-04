use backend_management_rs::path_utils::expand_home_dir;
use std::path::PathBuf;
use tokio::fs;
use tokio::io::AsyncReadExt;

async fn extract_layer_names_from_file(file_path: &PathBuf) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut file = fs::File::open(file_path).await?;

    // Read the first 8 bytes to get header size
    let mut size_bytes = [0u8; 8];
    file.read_exact(&mut size_bytes).await?;
    let header_size = u64::from_le_bytes(size_bytes);

    // Read the header
    let mut header_bytes = vec![0u8; header_size as usize];
    file.read_exact(&mut header_bytes).await?;

    // Parse header as JSON to extract tensor names
    let header: serde_json::Value = serde_json::from_slice(&header_bytes)?;

    let mut layer_names = Vec::new();
    if let Some(obj) = header.as_object() {
        for key in obj.keys() {
            if key != "__metadata__" {
                layer_names.push(key.clone());
            }
        }
    }

    Ok(layer_names)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Inspecting layer names in Llama 3.1 8B model...");

    let model_path = expand_home_dir("~/.cache/symphony/models/meta-llama--Llama-3.1-8B-Instruct");

    if !model_path.exists() {
        println!("❌ Model not found at: {}", model_path.display());
        return Ok(());
    }

    println!("📁 Model path: {}", model_path.display());

    // Find the first SafeTensors file to inspect
    let mut entries = fs::read_dir(&model_path).await?;
    let mut safetensors_file = None;

    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if let Some(extension) = path.extension() {
            if extension == "safetensors" {
                safetensors_file = Some(path);
                break;
            }
        }
    }

    if let Some(file_path) = safetensors_file {
        println!("📄 Inspecting file: {}", file_path.display());

        let layer_names = extract_layer_names_from_file(&file_path).await?;

        println!("📊 Found {} tensor entries", layer_names.len());
        println!("📝 First 20 tensor names:");

        for (i, name) in layer_names.iter().take(20).enumerate() {
            println!("  {}: {}", i + 1, name);
        }

        if layer_names.len() > 20 {
            println!("  ... and {} more", layer_names.len() - 20);
        }

        // Look for attention layers specifically
        println!("\n🔍 Attention layer names (q_proj, k_proj, v_proj):");
        let attention_layers: Vec<_> = layer_names.iter()
            .filter(|name| name.contains("attn") && (name.contains("q_proj") || name.contains("k_proj") || name.contains("v_proj")))
            .take(10)
            .collect();

        for (i, name) in attention_layers.iter().enumerate() {
            println!("  {}: {}", i + 1, name);
        }

        // Look for any self_attn layers
        println!("\n🔍 All self_attn layer names:");
        let self_attn_layers: Vec<_> = layer_names.iter()
            .filter(|name| name.contains("self_attn"))
            .take(15)
            .collect();

        for (i, name) in self_attn_layers.iter().enumerate() {
            println!("  {}: {}", i + 1, name);
        }
    } else {
        println!("❌ No SafeTensors files found in model directory");
    }

    println!("\n✅ Layer inspection complete!");

    Ok(())
}
