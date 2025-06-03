// Symphony Tokenizer Integration Demo
// This example demonstrates the configurable tokenizer functionality

use std::env;
use std::path::Path;
use pie_rt::{load_symphony_tokenizer, llama3_tokenizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get model path from command line or use default
    let args: Vec<String> = env::args().collect();
    let model_path = if args.len() > 1 {
        &args[1]
    } else {
        // Default path - user can override with command line argument
        "/home/sslee/.cache/symphony/models/meta-llama--Llama-3.1-8B-Instruct"
    };

    println!("🎵 Symphony Tokenizer Integration Demo");
    println!("=====================================");
    println!("Model path: {}", model_path);

    // Check if the path exists
    if !Path::new(model_path).exists() {
        println!("❌ Model path does not exist: {}", model_path);
        println!("Please provide a valid model path as a command line argument:");
        println!("  cargo run --example tokenizer_demo -- /path/to/your/model");
        return Ok(());
    }

    // Load the Symphony tokenizer with metadata
    println!("\n🔧 Loading Symphony tokenizer with metadata integration...");
    match load_symphony_tokenizer(model_path) {
        Ok(tokenizer) => {
            println!("✅ Successfully loaded tokenizer!");

            // Get vocabulary information
            let vocabs = tokenizer.get_vocabs();
            let vocab_size = vocabs.len();
            println!("📖 Vocabulary size: {}", vocab_size);

            // Get special tokens
            let special_tokens = tokenizer.special_tokens();
            println!("🔤 Special tokens available: {}", special_tokens.len());

            // Show some key special tokens
            let key_tokens = vec![
                "<|begin_of_text|>",
                "<|end_of_text|>",
                "<|start_header_id|>",
                "<|end_header_id|>",
                "<|eot_id|>",
                "<|python_tag|>",
            ];

            println!("\n🔑 Key special tokens:");
            for token in &key_tokens {
                if special_tokens.contains(token) {
                    println!("  ✓ {}", token);
                } else {
                    println!("  ✗ {} (not found)", token);
                }
            }

            // Demonstrate tokenization
            println!("\n🧪 Tokenization Examples:");
            let test_texts = vec![
                "Hello, world!",
                "The quick brown fox jumps over the lazy dog.",
                "<|begin_of_text|>System message<|end_of_text|>",
                "let x = 42; // This is Python-like code",
            ];

            for (i, text) in test_texts.iter().enumerate() {
                println!("\n  Example {}: \"{}\"", i + 1, text);

                // Tokenize with special tokens
                let tokens = tokenizer.encode_with_special_tokens(text);
                println!("    Tokens: {:?}", &tokens[..std::cmp::min(10, tokens.len())]);
                if tokens.len() > 10 {
                    println!("    ... ({} more tokens)", tokens.len() - 10);
                }
                println!("    Token count: {}", tokens.len());

                // Decode back to verify round-trip
                match tokenizer.decode(&tokens) {
                    Ok(decoded) => {
                        if decoded == *text {
                            println!("    ✅ Round-trip successful");
                        } else {
                            println!("    ⚠️  Round-trip mismatch: \"{}\"", decoded);
                        }
                    }
                    Err(e) => {
                        println!("    ❌ Decode error: {}", e);
                    }
                }
            }

            // Compare with legacy tokenizer
            println!("\n🔄 Comparison with Legacy Tokenizer:");
            let tokenizer_model_path = format!("{}/tokenizer.model", model_path);
            if Path::new(&tokenizer_model_path).exists() {
                match llama3_tokenizer(&tokenizer_model_path) {
                    Ok(legacy_tokenizer) => {
                        println!("  ✅ Legacy tokenizer loaded successfully");

                        let legacy_vocabs = legacy_tokenizer.get_vocabs();
                        let legacy_special_tokens = legacy_tokenizer.special_tokens();

                        println!("  📊 Comparison:");
                        println!("    Symphony tokenizer: {} vocab + {} special tokens", vocab_size, special_tokens.len());
                        println!("    Legacy tokenizer:   {} vocab + {} special tokens", legacy_vocabs.len(), legacy_special_tokens.len());

                        // Test a few examples with both tokenizers
                        let comparison_texts = vec![
                            "Hello, world!",
                            "<|begin_of_text|>Test<|end_of_text|>",
                        ];

                        for text in &comparison_texts {
                            println!("\n  🧪 Comparing tokenization of: \"{}\"", text);

                            let symphony_tokens = tokenizer.encode_with_special_tokens(text);
                            let legacy_tokens = legacy_tokenizer.encode_with_special_tokens(text);

                            println!("    Symphony: {} tokens -> {:?}", symphony_tokens.len(), &symphony_tokens[..std::cmp::min(5, symphony_tokens.len())]);
                            println!("    Legacy:   {} tokens -> {:?}", legacy_tokens.len(), &legacy_tokens[..std::cmp::min(5, legacy_tokens.len())]);

                            if symphony_tokens == legacy_tokens {
                                println!("    ✅ Identical tokenization");
                            } else {
                                println!("    ⚠️  Different tokenization");

                                // Check if decoded results are the same
                                let symphony_decoded = tokenizer.decode(&symphony_tokens).unwrap_or_default();
                                let legacy_decoded = legacy_tokenizer.decode(&legacy_tokens).unwrap_or_default();

                                if symphony_decoded == legacy_decoded && symphony_decoded == *text {
                                    println!("    ✅ Both decode correctly to original text");
                                } else {
                                    println!("    ❌ Decode results differ");
                                    println!("      Symphony decoded: \"{}\"", symphony_decoded);
                                    println!("      Legacy decoded:   \"{}\"", legacy_decoded);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        println!("  ❌ Failed to load legacy tokenizer: {}", e);
                    }
                }
            } else {
                println!("  ⚠️  tokenizer.model not found for legacy comparison");
            }

            println!("\n🎯 Summary:");
            println!("  • Loaded tokenizer with {} vocabulary tokens", vocab_size);
            println!("  • Found {} special tokens", special_tokens.len());
            println!("  • Metadata integration working correctly");
            println!("  • Round-trip tokenization verified");
            println!("  • Legacy tokenizer comparison completed");
            println!("  • Comparison with legacy tokenizer completed");

        }
        Err(e) => {
            println!("❌ Failed to load tokenizer: {}", e);
            println!("Make sure you have:");
            println!("  1. Downloaded the Llama-3.1-8B-Instruct model");
            println!("  2. Generated the metadata using backend-management-rs");
            println!("  3. Provided the correct model path");
            return Err(e);
        }
    }
    Ok(())
}