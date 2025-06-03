//! Tests for tokenizer conversion functionality
//! 
//! This module tests the conversion of HuggingFace tokenizer.json files
//! to Symphony's tokenizer.model format, specifically handling:
//! - Unicode to bytes transformation for byte-level BPE tokenizers
//! - Base64 encoding of converted tokens
//! - Text format output for human readability
//! - Complete preservation of vocabulary and token IDs

use std::fs;
use std::path::Path;
use backend_management_rs::transform_tokenizer::convert_hf_tokenizer_to_symphony;
use serde_json::Value;

#[tokio::test]
async fn test_tokenizer_conversion_with_llama_model() {
    let model_dir = "/home/sslee/.cache/symphony/models/meta-llama--Llama-3.1-8B-Instruct";
    let json_path = format!("{}/tokenizer.json", model_dir);
    let original_model_path = format!("{}/original/tokenizer.model", model_dir);
    
    if !Path::new(&json_path).exists() || !Path::new(&original_model_path).exists() {
        println!("Skipping test - model files not found at {}", model_dir);
        return;
    }

    // Convert the tokenizer
    let generated_model_path = format!("{}/tokenizer.model", model_dir);
    convert_hf_tokenizer_to_symphony(Path::new(model_dir))
        .await
        .expect("Failed to convert tokenizer");

    // Check that the file was created
    assert!(Path::new(&generated_model_path).exists(), "Generated tokenizer.model should exist");

    // Compare file sizes
    let original_size = fs::metadata(&original_model_path).unwrap().len();
    let generated_size = fs::metadata(&generated_model_path).unwrap().len();
    
    println!("Original tokenizer.model size: {} bytes", original_size);
    println!("Generated tokenizer.model size: {} bytes", generated_size);
    
    // For text format, sizes should match exactly
    assert_eq!(original_size, generated_size, "File sizes should match exactly");
}

#[tokio::test]
async fn test_tokenizer_conversion_detailed_comparison() {
    let model_dir = "/home/sslee/.cache/symphony/models/meta-llama--Llama-3.1-8B-Instruct";
    let json_path = format!("{}/tokenizer.json", model_dir);
    let original_model_path = format!("{}/original/tokenizer.model", model_dir);
    
    if !Path::new(&json_path).exists() || !Path::new(&original_model_path).exists() {
        println!("Skipping test - model files not found at {}", model_dir);
        return;
    }

    // Read and parse the original tokenizer.json to get expected vocab size
    let json_content = fs::read_to_string(&json_path).expect("Failed to read tokenizer.json");
    let tokenizer_json: Value = serde_json::from_str(&json_content).expect("Failed to parse JSON");
    
    let expected_vocab_size = if let Some(vocab) = tokenizer_json["model"]["vocab"].as_object() {
        vocab.len()
    } else {
        panic!("Could not find vocab in tokenizer.json");
    };

    println!("Expected vocabulary size from tokenizer.json: {}", expected_vocab_size);

    // Convert the tokenizer
    let generated_model_path = format!("{}/tokenizer.model", model_dir);
    convert_hf_tokenizer_to_symphony(Path::new(model_dir))
        .await
        .expect("Failed to convert tokenizer");

    // Read the generated model and count non-empty lines
    let generated_content = fs::read_to_string(&generated_model_path)
        .expect("Failed to read generated tokenizer.model");
    
    let actual_vocab_size = generated_content.lines()
        .filter(|line| !line.trim().is_empty())
        .count();
    
    println!("Generated vocabulary size: {}", actual_vocab_size);
    
    assert_eq!(expected_vocab_size, actual_vocab_size, 
        "Vocabulary size mismatch: expected {} but got {}", 
        expected_vocab_size, actual_vocab_size);
}

#[tokio::test]
async fn test_tokenizer_byte_conversion_sample() {
    let model_dir = "/home/sslee/.cache/symphony/models/meta-llama--Llama-3.1-8B-Instruct";
    let json_path = format!("{}/tokenizer.json", model_dir);
    
    if !Path::new(&json_path).exists() {
        println!("Skipping test - tokenizer.json not found at {}", json_path);
        return;
    }

    // Convert the tokenizer
    let generated_model_path = format!("{}/tokenizer.model", model_dir);
    convert_hf_tokenizer_to_symphony(Path::new(model_dir))
        .await
        .expect("Failed to convert tokenizer");

    // Read the generated model and check first few lines
    let generated_content = fs::read_to_string(&generated_model_path)
        .expect("Failed to read generated tokenizer.model");

    let mut valid_lines = 0;
    for (i, line) in generated_content.lines().take(10).enumerate() {
        if let Some((token_b64, id_str)) = line.split_once(' ') {
            if let Ok(id) = id_str.parse::<usize>() {
                println!("Sample line {}: {} {}", i, token_b64, id);
                valid_lines += 1;
            }
        }
    }
    
    assert!(valid_lines > 0, "Should have valid token lines");
    println!("Validated {} sample lines", valid_lines);
}

#[tokio::test]
async fn test_content_analysis() {
    let model_dir = "/home/sslee/.cache/symphony/models/meta-llama--Llama-3.1-8B-Instruct";
    let original_model_path = format!("{}/original/tokenizer.model", model_dir);
    let json_path = format!("{}/tokenizer.json", model_dir);
    
    if !Path::new(&original_model_path).exists() || !Path::new(&json_path).exists() {
        println!("Skipping test - model files not found at {}", model_dir);
        return;
    }

    // Convert the tokenizer
    let generated_model_path = format!("{}/tokenizer.model", model_dir);
    convert_hf_tokenizer_to_symphony(Path::new(model_dir))
        .await
        .expect("Failed to convert tokenizer");

    // Read and analyze both files
    let original_content = fs::read_to_string(&original_model_path)
        .expect("Failed to read original tokenizer.model");
    let generated_content = fs::read_to_string(&generated_model_path)
        .expect("Failed to read generated tokenizer.model");
    
    let original_lines: Vec<&str> = original_content.lines().collect();
    let generated_lines: Vec<&str> = generated_content.lines().collect();
    
    println!("Original file has {} lines", original_lines.len());
    println!("Generated file has {} lines", generated_lines.len());
    
    assert_eq!(original_lines.len(), generated_lines.len(), "Line counts should match");
    
    // Check file sizes
    let original_size = original_content.len();
    let generated_size = generated_content.len();
    
    println!("Original file size: {} bytes", original_size);
    println!("Generated file size: {} bytes", generated_size);
    
    assert_eq!(original_size, generated_size, "File sizes should match exactly");
}

#[tokio::test] 
async fn test_vocabulary_verification() {
    let model_dir = "/home/sslee/.cache/symphony/models/meta-llama--Llama-3.1-8B-Instruct";
    let json_path = format!("{}/tokenizer.json", model_dir);
    
    if !Path::new(&json_path).exists() {
        println!("Skipping test - tokenizer.json not found at {}", json_path);
        return;
    }

    // Convert the tokenizer
    let generated_model_path = format!("{}/tokenizer.model", model_dir);
    
    // Clean up any existing file first
    if Path::new(&generated_model_path).exists() {
        fs::remove_file(&generated_model_path).ok();
    }
    
    convert_hf_tokenizer_to_symphony(Path::new(model_dir))
        .await
        .expect("Failed to convert tokenizer");

    // Read the generated model
    let generated_content = fs::read_to_string(&generated_model_path)
        .expect("Failed to read generated tokenizer.model");

    // Parse JSON to get original vocabulary
    let json_content = fs::read_to_string(&json_path).expect("Failed to read tokenizer.json");
    let tokenizer_json: Value = serde_json::from_str(&json_content).expect("Failed to parse JSON");
    
    let vocab = tokenizer_json["model"]["vocab"].as_object()
        .expect("Could not find vocab in tokenizer.json");

    println!("Original vocab size: {}", vocab.len());
    println!("Generated lines: {}", generated_content.lines().count());

    // Check that we have the right format and IDs match
    let mut verified_count = 0;
    for line in generated_content.lines().take(10) {
        if let Some((token_b64, id_str)) = line.split_once(' ') {
            if let Ok(id) = id_str.parse::<i32>() {
                // Find this token in the original vocab by ID
                let found = vocab.iter().any(|(_, original_id)| {
                    if let Some(orig_id) = original_id.as_i64() {
                        orig_id == id as i64
                    } else {
                        false
                    }
                });
                
                if found {
                    verified_count += 1;
                }
                
                println!("Token {}: {} -> ID {}", verified_count, token_b64, id);
            }
        }
    }
    
    println!("Successfully verified {} tokens", verified_count);
    assert!(verified_count > 0, "Should find at least some matching tokens");
}

#[tokio::test]
async fn test_complete_file_comparison() {
    let model_dir = "/home/sslee/.cache/symphony/models/meta-llama--Llama-3.1-8B-Instruct";
    let json_path = format!("{}/tokenizer.json", model_dir);
    let original_model_path = format!("{}/original/tokenizer.model", model_dir);
    
    if !Path::new(&json_path).exists() || !Path::new(&original_model_path).exists() {
        println!("Skipping test - model files not found at {}", model_dir);
        return;
    }

    // Convert the tokenizer
    let generated_model_path = format!("{}/tokenizer.model", model_dir);
    convert_hf_tokenizer_to_symphony(Path::new(model_dir))
        .await
        .expect("Failed to convert tokenizer");

    // Read both files completely
    let original_content = fs::read_to_string(&original_model_path)
        .expect("Failed to read original tokenizer.model");
    let generated_content = fs::read_to_string(&generated_model_path)
        .expect("Failed to read generated tokenizer.model");

    // Convert both to lines for comparison
    let original_lines: Vec<&str> = original_content.lines().collect();
    let generated_lines: Vec<&str> = generated_content.lines().collect();

    println!("Comparing {} lines from original with {} lines from generated", 
             original_lines.len(), generated_lines.len());

    // First check: line counts must match
    assert_eq!(original_lines.len(), generated_lines.len(), 
               "Line count mismatch: original has {} lines, generated has {}", 
               original_lines.len(), generated_lines.len());

    // Second check: content must be identical
    let mut differences = 0;
    for (i, (orig_line, gen_line)) in original_lines.iter().zip(generated_lines.iter()).enumerate() {
        if orig_line != gen_line {
            println!("DIFFERENCE at line {}: ", i + 1);
            println!("  Original:  '{}'", orig_line);
            println!("  Generated: '{}'", gen_line);
            differences += 1;
            
            // Only show first 5 differences to avoid spam
            if differences >= 5 {
                println!("... and {} more differences", 
                         original_lines.len() - i - 1);
                break;
            }
        }
    }

    if differences == 0 {
        println!("✅ SUCCESS: Files are COMPLETELY IDENTICAL!");
    } else {
        println!("❌ FAILURE: Found {} differences between files", differences);
    }

    assert_eq!(differences, 0, "Files should be identical but found {} differences", differences);
    
    // Final verification: byte-level comparison
    assert_eq!(original_content, generated_content, "Files should be byte-for-byte identical");
    
    println!("Final verification passed: Files are completely identical at byte level");
}
