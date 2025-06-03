use serde_json;

// Import the structs
use backend_management_rs::config::{ModelInfo, ModelArchInfo};

fn main() {
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

    // Serialize to JSON to test the structure
    let json = serde_json::to_string_pretty(&model_info).expect("Failed to serialize");
    println!("Model Info JSON structure:");
    println!("{}", json);

    // Test deserialization
    let deserialized: ModelInfo = serde_json::from_str(&json).expect("Failed to deserialize");
    println!("\nDeserialization successful!");
    println!("Model name: {}", deserialized.name);
    println!("Model type: {}", deserialized.model_type);
    println!("Architecture: {:?}", deserialized.arch_info.architectures);
    println!("Vocab size: {:?}", deserialized.arch_info.vocab_size);
    println!("Hidden size: {:?}", deserialized.arch_info.hidden_size);
}
