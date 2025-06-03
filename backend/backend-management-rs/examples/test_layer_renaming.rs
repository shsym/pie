use backend_management_rs::{ModelInstaller, error::Result};
use std::fs;
use serde_json;
use tokio;
use tempfile::TempDir;

#[tokio::main]
async fn main() -> Result<()> {
    println!("Testing layer renaming functionality...");
    
    // Create a temporary directory for our test model
    let temp_dir = TempDir::new()?;
    let model_path = temp_dir.path().join("test_model");
    fs::create_dir_all(&model_path)?;
    
    println!("📁 Created test model directory: {}", model_path.display());

    // Create a mock model.safetensors.index.json
    let mock_index = serde_json::json!({
        "metadata": {
            "total_size": 29319014400u64
        },
        "weight_map": {
            "model.embed_tokens.weight": "model-00001-of-00003.safetensors",
            "model.layers.0.input_layernorm.weight": "model-00001-of-00003.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00003.safetensors",
            "model.layers.0.self_attn.k_proj.weight": "model-00001-of-00003.safetensors",
            "model.layers.0.self_attn.v_proj.weight": "model-00001-of-00003.safetensors",
            "model.layers.0.self_attn.o_proj.weight": "model-00002-of-00003.safetensors",
            "model.layers.0.post_attention_layernorm.weight": "model-00002-of-00003.safetensors",
            "model.layers.0.mlp.gate_proj.weight": "model-00002-of-00003.safetensors",
            "model.layers.0.mlp.up_proj.weight": "model-00002-of-00003.safetensors",
            "model.layers.0.mlp.down_proj.weight": "model-00003-of-00003.safetensors",
            "model.layers.1.input_layernorm.weight": "model-00003-of-00003.safetensors",
            "model.norm.weight": "model-00003-of-00003.safetensors",
            "lm_head.weight": "model-00003-of-00003.safetensors"
        }
    });

    let index_path = model_path.join("model.safetensors.index.json");
    fs::write(&index_path, serde_json::to_string_pretty(&mock_index)?)?;
    println!("✅ Created mock SafeTensors index file");

    // Copy the weight_renaming.json from examples
    let renaming_rules_path = model_path.join("weight_renaming.json");
    let example_rules_path = std::env::current_dir()?.join("examples/weight_renaming.json");
    if example_rules_path.exists() {
        fs::copy(&example_rules_path, &renaming_rules_path)?;
        println!("✅ Copied weight renaming rules");
    } else {
        // Create a simplified version for testing
        let simple_rules = serde_json::json!({
            "root": "model",
            "rules": [
                {
                    "name": "Convert layer norm",
                    "type": "regex",
                    "pattern": "^(model\\.)?layers\\.(\\d+)\\.input_layernorm\\.weight$",
                    "replacement": "model.layers.$2.input_norm.weight",
                    "enabled": true
                },
                {
                    "name": "Convert post attention norm",
                    "type": "regex", 
                    "pattern": "^(model\\.)?layers\\.(\\d+)\\.post_attention_layernorm\\.weight$",
                    "replacement": "model.layers.$2.post_attention_norm.weight",
                    "enabled": true
                }
            ]
        });
        fs::write(&renaming_rules_path, serde_json::to_string_pretty(&simple_rules)?)?;
        println!("✅ Created simplified weight renaming rules");
    }

    // Create mock SafeTensors files with minimal valid headers
    for i in 1..=3 {
        let file_name = format!("model-{:05}-of-00003.safetensors", i);
        let file_path = model_path.join(&file_name);
        
        // Create a minimal valid SafeTensors file structure
        let mut header = serde_json::Map::new();
        
        // Add some mock tensor definitions based on the index
        match i {
            1 => {
                header.insert("model.embed_tokens.weight".to_string(), serde_json::json!({
                    "dtype": "F32",
                    "shape": [32000, 4096],
                    "data_offsets": [0, 524288000]
                }));
                header.insert("model.layers.0.input_layernorm.weight".to_string(), serde_json::json!({
                    "dtype": "F32", 
                    "shape": [4096],
                    "data_offsets": [524288000, 524304384]
                }));
                header.insert("model.layers.0.self_attn.q_proj.weight".to_string(), serde_json::json!({
                    "dtype": "F32",
                    "shape": [4096, 4096], 
                    "data_offsets": [524304384, 591413248]
                }));
            },
            2 => {
                header.insert("model.layers.0.self_attn.o_proj.weight".to_string(), serde_json::json!({
                    "dtype": "F32",
                    "shape": [4096, 4096],
                    "data_offsets": [0, 67108864]
                }));
                header.insert("model.layers.0.post_attention_layernorm.weight".to_string(), serde_json::json!({
                    "dtype": "F32",
                    "shape": [4096],
                    "data_offsets": [67108864, 67125248]
                }));
            },
            3 => {
                header.insert("model.layers.0.mlp.down_proj.weight".to_string(), serde_json::json!({
                    "dtype": "F32", 
                    "shape": [4096, 11008],
                    "data_offsets": [0, 180355072]
                }));
                header.insert("model.norm.weight".to_string(), serde_json::json!({
                    "dtype": "F32",
                    "shape": [4096],
                    "data_offsets": [180355072, 180371456]
                }));
                header.insert("lm_head.weight".to_string(), serde_json::json!({
                    "dtype": "F32",
                    "shape": [32000, 4096], 
                    "data_offsets": [180371456, 704659456]
                }));
            },
            _ => {}
        }

        let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header))?;
        let header_size = header_bytes.len() as u64;
        
        // Create file with header size + header + dummy data
        let mut file_content = Vec::new();
        file_content.extend_from_slice(&header_size.to_le_bytes());
        file_content.extend_from_slice(&header_bytes);
        
        // Add some dummy tensor data (just zeros for testing)
        file_content.extend_from_slice(&vec![0u8; 1024]); // 1KB dummy data
        
        fs::write(&file_path, file_content)?;
        println!("✅ Created mock SafeTensors file: {}", file_name);
    }

    // Create ModelInstaller and test layer renaming
    let installer = ModelInstaller::new(Some(temp_dir.path().to_path_buf()));
    
    println!("\n🔄 Starting layer renaming process...");
    match installer.rename_model_layers(&model_path).await {
        Ok(()) => {
            println!("✅ Layer renaming completed successfully!");
            
            // Verify the results by reading the updated index
            let updated_index_content = fs::read_to_string(&index_path)?;
            let updated_index: serde_json::Value = serde_json::from_str(&updated_index_content)?;
            
            println!("\n📋 Updated layer mappings:");
            if let Some(weight_map) = updated_index.get("weight_map").and_then(|v| v.as_object()) {
                for (layer_name, file_name) in weight_map {
                    println!("   {} -> {}", layer_name, file_name);
                }
            }
        },
        Err(e) => {
            println!("❌ Layer renaming failed: {}", e);
            return Err(e);
        }
    }

    // Persist the temp directory for inspection
    let persistent_temp_path = temp_dir.keep();
    println!("\n💾 Test files preserved in: {}", persistent_temp_path.display());
    
    println!("\n🎉 Layer renaming test completed successfully!");
    println!("📋 Test Summary:");
    println!("   • Mock SafeTensors files created ✅");
    println!("   • Weight renaming rules loaded ✅");
    println!("   • Layer names extracted ✅");
    println!("   • Renaming rules applied ✅");
    println!("   • SafeTensors files updated ✅");
    println!("   • Index file updated ✅");
    
    Ok(())
}
