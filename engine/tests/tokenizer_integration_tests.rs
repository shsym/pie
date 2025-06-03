use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use serde_json;

// Import the tokenizer module from the engine
use pie_rt::tokenizer::{
    configurable_tokenizer, load_symphony_tokenizer, llama3_tokenizer, BytePairEncoder,
};

// Helper function to get the test model path
fn get_test_model_path() -> PathBuf {
    let home = std::env::var("HOME").expect("HOME environment variable not set");
    PathBuf::from(home).join(".cache/symphony/models/meta-llama--Llama-3.1-8B-Instruct")
}

// Helper function to check if test model exists
fn test_model_exists() -> bool {
    let model_path = get_test_model_path();
    model_path.exists() && model_path.join("tokenizer.model").exists()
}

#[test]
fn test_tokenizer_metadata_loading() {
    if !test_model_exists() {
        println!("Skipping test: Llama-3.1-8B-Instruct model not found");
        return;
    }

    let model_path = get_test_model_path();
    let model_path_str = model_path.to_str().unwrap();
    
    // Test loading tokenizer with symphony metadata
    let tokenizer_result = load_symphony_tokenizer(model_path_str);
    assert!(tokenizer_result.is_ok(), "Failed to load symphony tokenizer: {:?}", tokenizer_result.err());
    
    let tokenizer = tokenizer_result.unwrap();
    
    // Test basic tokenization
    let test_text = "Hello, world!";
    let tokens = tokenizer.encode_with_special_tokens(test_text);
    assert!(!tokens.is_empty(), "Tokenization should produce non-empty token list");
    
    // Test detokenization
    let decoded_result = tokenizer.decode(&tokens);
    assert!(decoded_result.is_ok(), "Detokenization failed: {:?}", decoded_result.err());
    
    let decoded_text = decoded_result.unwrap();
    assert_eq!(decoded_text, test_text, "Round-trip tokenization failed");
    
    println!("✓ Basic tokenizer metadata loading and functionality test passed");
}

#[test]
fn test_special_tokens_from_metadata() {
    if !test_model_exists() {
        println!("Skipping test: Llama-3.1-8B-Instruct model not found");
        return;
    }

    let model_path = get_test_model_path();
    let model_path_str = model_path.to_str().unwrap();
    
    let tokenizer = load_symphony_tokenizer(model_path_str)
        .expect("Failed to load symphony tokenizer");
    
    // Get all special tokens
    let special_tokens = tokenizer.special_tokens();
    
    // Test that expected Llama 3.1 special tokens are present
    let expected_tokens = vec![
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
        "<|finetune_right_pad_id|>",
    ];
    
    for expected_token in expected_tokens {
        assert!(special_tokens.contains(expected_token), 
               "Special token '{}' not found in tokenizer", expected_token);
    }
    
    // Test tokenizing text with special tokens
    let text_with_special = "<|begin_of_text|>Hello<|end_of_text|>";
    let tokens = tokenizer.encode_with_special_tokens(text_with_special);
    assert!(!tokens.is_empty(), "Should be able to tokenize text with special tokens");
    
    // Test that we can decode back
    let decoded = tokenizer.decode(&tokens).expect("Should be able to decode");
    assert_eq!(decoded, text_with_special, "Round-trip with special tokens failed");
    
    println!("✓ Special tokens from metadata test passed");
    println!("  Found {} special tokens", special_tokens.len());
}

#[test]
fn test_vocabulary_access() {
    if !test_model_exists() {
        println!("Skipping test: Llama-3.1-8B-Instruct model not found");
        return;
    }

    let model_path = get_test_model_path();
    let model_path_str = model_path.to_str().unwrap();
    
    let tokenizer = load_symphony_tokenizer(model_path_str)
        .expect("Failed to load symphony tokenizer");
    
    // Test vocabulary access
    let vocabs = tokenizer.get_vocabs();
    assert!(!vocabs.is_empty(), "Vocabulary should not be empty");
    
    // Test that vocabulary size matches expected size from metadata (128,000)
    // Note: This might be slightly different due to special tokens
    assert!(vocabs.len() >= 128000, 
           "Vocabulary size should be at least 128,000, got {}", vocabs.len());
    
    // Test that we can access individual tokens
    for i in 0..std::cmp::min(100, vocabs.len()) {
        let token_bytes = &vocabs[i];
        assert!(!token_bytes.is_empty(), "Token {} should not be empty", i);
    }
    
    println!("✓ Vocabulary access test passed");
    println!("  Total vocabulary size: {}", vocabs.len());
}

#[test]
fn test_configurable_vs_legacy_tokenizer() {
    if !test_model_exists() {
        println!("Skipping test: Llama-3.1-8B-Instruct model not found");
        return;
    }

    let model_path = get_test_model_path();
    let tokenizer_model_path = model_path.join("tokenizer.model");
    let tokenizer_model_str = tokenizer_model_path.to_str().unwrap();
    let model_path_str = model_path.to_str().unwrap();
    
    // Load both tokenizers
    let legacy_tokenizer = llama3_tokenizer(tokenizer_model_str)
        .expect("Failed to load legacy tokenizer");
    
    let configurable_tokenizer = load_symphony_tokenizer(model_path_str)
        .expect("Failed to load configurable tokenizer");
    
    // Test that both produce the same results for basic text
    let test_texts = vec![
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "This is a test with numbers: 123 and symbols: !@#$%",
    ];
    
    for test_text in test_texts {
        let legacy_tokens = legacy_tokenizer.encode_with_special_tokens(test_text);
        let config_tokens = configurable_tokenizer.encode_with_special_tokens(test_text);
        
        // The tokens might be different due to different special token handling,
        // but both should be able to decode back to original text
        let legacy_decoded = legacy_tokenizer.decode(&legacy_tokens)
            .expect("Legacy tokenizer should decode");
        let config_decoded = configurable_tokenizer.decode(&config_tokens)
            .expect("Configurable tokenizer should decode");
        
        assert_eq!(legacy_decoded, test_text, "Legacy tokenizer round-trip failed");
        assert_eq!(config_decoded, test_text, "Configurable tokenizer round-trip failed");
    }
    
    println!("✓ Configurable vs legacy tokenizer comparison test passed");
}

#[test]
fn test_metadata_file_parsing() {
    if !test_model_exists() {
        println!("Skipping test: Llama-3.1-8B-Instruct model not found");
        return;
    }

    let model_path = get_test_model_path();
    let metadata_file = model_path.join("symphony_model_info.json");
    
    // Check that metadata file exists
    assert!(metadata_file.exists(), "symphony_model_info.json should exist at {:?}", metadata_file);
    
    // Read and parse metadata file
    let metadata_content = fs::read_to_string(&metadata_file)
        .expect("Should be able to read metadata file");
    
    let metadata: serde_json::Value = serde_json::from_str(&metadata_content)
        .expect("Metadata file should be valid JSON");
    
    // Check that required fields exist
    assert!(metadata.get("tokenizer_metadata").is_some(), 
           "tokenizer_metadata field should exist");
    
    let tokenizer_metadata = metadata["tokenizer_metadata"].as_object()
        .expect("tokenizer_metadata should be an object");
    
    // Check required metadata fields
    assert!(tokenizer_metadata.contains_key("model_type"), 
           "model_type should be present");
    assert!(tokenizer_metadata.contains_key("vocab_size"), 
           "vocab_size should be present");
    assert!(tokenizer_metadata.contains_key("special_tokens"), 
           "special_tokens should be present");
    assert!(tokenizer_metadata.contains_key("added_tokens"), 
           "added_tokens should be present");
    
    // Validate specific values
    let model_type = tokenizer_metadata["model_type"].as_str()
        .expect("model_type should be a string");
    assert_eq!(model_type, "BPE", "Model type should be BPE");
    
    let vocab_size = tokenizer_metadata["vocab_size"].as_u64()
        .expect("vocab_size should be a number");
    assert_eq!(vocab_size, 128000, "Vocab size should be 128,000");
    
    // Check that special tokens exist
    let special_tokens = tokenizer_metadata["special_tokens"].as_object()
        .expect("special_tokens should be an object");
    assert!(!special_tokens.is_empty(), "Should have special tokens");
    
    // Check that added tokens exist
    let added_tokens = tokenizer_metadata["added_tokens"].as_array()
        .expect("added_tokens should be an array");
    assert!(!added_tokens.is_empty(), "Should have added tokens");
    
    println!("✓ Metadata file parsing test passed");
    println!("  Model type: {}", model_type);
    println!("  Vocab size: {}", vocab_size);
    println!("  Special tokens: {}", special_tokens.len());
    println!("  Added tokens: {}", added_tokens.len());
}

#[test]
fn test_performance_comparison() {
    if !test_model_exists() {
        println!("Skipping test: Llama-3.1-8B-Instruct model not found");
        return;
    }

    let model_path = get_test_model_path();
    let tokenizer_model_path = model_path.join("tokenizer.model");
    let tokenizer_model_str = tokenizer_model_path.to_str().unwrap();
    let model_path_str = model_path.to_str().unwrap();
    
    // Load both tokenizers
    let legacy_tokenizer = llama3_tokenizer(tokenizer_model_str)
        .expect("Failed to load legacy tokenizer");
    
    let configurable_tokenizer = load_symphony_tokenizer(model_path_str)
        .expect("Failed to load configurable tokenizer");
    
    // Create test text
    let test_text = "The quick brown fox jumps over the lazy dog. ".repeat(100);
    
    // Time legacy tokenizer
    let start = std::time::Instant::now();
    for _ in 0..10 {
        let tokens = legacy_tokenizer.encode_with_special_tokens(&test_text);
        let _decoded = legacy_tokenizer.decode(&tokens).unwrap();
    }
    let legacy_duration = start.elapsed();
    
    // Time configurable tokenizer
    let start = std::time::Instant::now();
    for _ in 0..10 {
        let tokens = configurable_tokenizer.encode_with_special_tokens(&test_text);
        let _decoded = configurable_tokenizer.decode(&tokens).unwrap();
    }
    let configurable_duration = start.elapsed();
    
    println!("✓ Performance comparison test completed");
    println!("  Legacy tokenizer: {:?}", legacy_duration);
    println!("  Configurable tokenizer: {:?}", configurable_duration);
    
    // Performance should be reasonably similar (within 5x)
    let ratio = configurable_duration.as_millis() as f64 / legacy_duration.as_millis() as f64;
    assert!(ratio < 5.0, "Configurable tokenizer should not be more than 5x slower than legacy");
}
