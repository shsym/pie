use backend_management_rs::{ModelInstaller, error::Result};
use backend_management_rs::config::{ModelInfo, ModelArchInfo};
use std::fs;
use tokio;
use tempfile::TempDir;

#[tokio::main]
async fn main() -> Result<()> {
    println!("Testing ModelInfo struct with new nested architecture...");
    
    // Create a test ModelInfo with the new nested structure
    let arch_info = ModelArchInfo {
        architectures: vec!["LlamaForCausalLM".to_string()],
        vocab_size: Some(128256),
        hidden_size: Some(4096),
        num_attention_heads: Some(32),
        num_hidden_layers: Some(32),
        intermediate_size: Some(14336),
        hidden_act: Some("silu".to_string()),
        hidden_dropout_prob: Some(0.0),
        attention_probs_dropout_prob: Some(0.0),
        max_position_embeddings: Some(131072),
        type_vocab_size: Some(1),
        layer_norm_eps: Some(1e-5),
        tie_word_embeddings: Some(false),
        bos_token_id: Some(128000),
        eos_token_id: Some(vec![128001]),
        pad_token_id: Some(128001),
        torch_dtype: Some("bfloat16".to_string()),
    };

    let model_info = ModelInfo {
        name: "Llama-3.1-8B-Instruct".to_string(),
        fullname: "meta-llama/Llama-3.1-8B-Instruct".to_string(),
        model_type: "llama3".to_string(),
        arch_info,
    };

    // Test that our refactored structure works correctly
    let json = serde_json::to_string_pretty(&model_info)?;
    println!("✅ ModelInfo serialization successful!");
    
    // Test that we can deserialize it back
    let _: ModelInfo = serde_json::from_str(&json)?;
    println!("✅ ModelInfo deserialization successful!");
    
    // Create a temporary directory for test artifacts
    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path().to_path_buf();
    println!("📁 Using temporary directory: {}", temp_path.display());
    
    // Test ModelInstaller creation with temporary directory
    let _installer = ModelInstaller::new(Some(temp_path.clone()));
    println!("✅ ModelInstaller creation successful!");
    
    // Write the test ModelInfo to a file in the temporary directory
    let test_file_path = temp_path.join("test_model_info.json");
    fs::write(&test_file_path, &json)?;
    println!("📝 Test ModelInfo written to: {}", test_file_path.display());
    
    // Persist the temp directory by keeping it (prevents automatic cleanup)
    let persistent_temp_path = temp_dir.keep();
    println!("💾 Test files preserved in: {}", persistent_temp_path.display());
    
    println!("\n🎉 All tests passed! Our refactoring is working correctly.");
    println!("The new ModelInfo struct properly encapsulates architecture info in ModelArchInfo.");
    println!("\n📋 Test Summary:");
    println!("   • ModelInfo serialization ✅");
    println!("   • ModelInfo deserialization ✅");
    println!("   • ModelInstaller creation ✅");
    println!("   • Test file creation ✅");
    println!("   • Test artifacts preserved for inspection ✅");
    
    Ok(())
}
