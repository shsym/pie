use backend_management_rs::transform_tokenizer::{convert_hf_tokenizer_to_symphony, extract_tokenizer_metadata, HfTokenizerJson};
use backend_management_rs::path_utils::expand_home_dir_str;
use std::path::Path;
use serde_json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = expand_home_dir_str("~/.cache/symphony/models/meta-llama--Llama-3.1-8B-Instruct");
    let tokenizer_json_path = format!("{}/tokenizer.json", model_dir);
    let info_file_path = format!("{}/symphony_model_info.json", model_dir);
    
    println!("Extracting tokenizer metadata and updating model info...");
    
    // Read and parse the tokenizer.json file
    let tokenizer_content = std::fs::read_to_string(&tokenizer_json_path)?;
    let hf_tokenizer: HfTokenizerJson = serde_json::from_str(&tokenizer_content)?;
    
    // Extract metadata first
    let metadata = extract_tokenizer_metadata(&hf_tokenizer)?;
    println!("Extracted metadata: {:#?}", metadata);
    
    // Convert tokenizer with metadata
    convert_hf_tokenizer_to_symphony(Path::new(&model_dir), Path::new(&info_file_path)).await?;
    
    println!("Tokenizer metadata updated successfully!");
    
    Ok(())
}
