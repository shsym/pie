//! Tokenizer transformation utilities
//!
//! This module provides functionality to convert Hugging Face tokenizer configurations
//! into Symphony's tokenizer.model format for BPE tokenizers.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tokio::fs;
use tracing::{info, warn};
use crate::error::{ManagementError, Result};

/// Hugging Face tokenizer JSON structure (partial)
#[derive(Debug, Deserialize)]
pub struct HfTokenizerJson {
    model: HfTokenizerModel,
    #[serde(default)]
    added_tokens: Vec<HfAddedToken>,
    #[serde(default)]
    pre_tokenizer: Option<HfPreTokenizer>,
    #[serde(default)]
    post_processor: Option<HfPostProcessor>,
    #[serde(default)]
    decoder: Option<HfDecoder>,
}

#[derive(Debug, Deserialize)]
struct HfTokenizerModel {
    #[serde(rename = "type")]
    model_type: String,
    vocab: HashMap<String, u32>,
    #[serde(default)]
    merges: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct HfAddedToken {
    id: u32,
    content: String,
    special: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct HfPreTokenizer {
    #[serde(rename = "type")]
    pre_tokenizer_type: Option<String>,
    pretokenizers: Option<Vec<HfPreTokenizerComponent>>,
}

#[derive(Debug, Deserialize)]
struct HfPreTokenizerComponent {
    #[serde(rename = "type")]
    component_type: String,
    pattern: Option<HfPattern>,
}

#[derive(Debug, Deserialize)]
struct HfPattern {
    #[serde(rename = "Regex")]
    regex: Option<String>,
}

#[derive(Debug, Deserialize)]
struct HfPostProcessor {
    #[serde(rename = "type")]
    post_processor_type: Option<String>,
    processors: Option<Vec<serde_json::Value>>,
    special_tokens: Option<HashMap<String, HfSpecialTokenInfo>>,
}

#[derive(Debug, Deserialize)]
struct HfSpecialTokenInfo {
    id: String,
    ids: Vec<u32>,
    tokens: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct HfDecoder {
    #[serde(rename = "type")]
    decoder_type: Option<String>,
}

/// Tokenizer metadata extracted from HuggingFace tokenizer.json
#[derive(Debug, Serialize, Deserialize)]
pub struct TokenizerMetadata {
    /// Model type (e.g., "BPE", "WordPiece", "SentencePiece")
    pub model_type: String,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Pre-tokenization regex pattern for text splitting
    pub pre_tokenizer_pattern: Option<String>,
    /// Special tokens with their IDs
    pub special_tokens: HashMap<String, u32>,
    /// Added tokens (including special tokens from added_tokens field)
    pub added_tokens: Vec<AddedTokenInfo>,
    /// Post-processor configuration
    pub post_processor_type: Option<String>,
    /// Decoder configuration
    pub decoder_type: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AddedTokenInfo {
    pub id: u32,
    pub content: String,
    pub special: bool,
}

/// Special tokens configuration for Symphony
#[derive(Debug, Serialize, Deserialize)]
pub struct SpecialTokensConfig {
    pub tokens: HashMap<String, u32>,
}

/// Convert HuggingFace tokenizer to Symphony tokenizer.model format
pub async fn convert_hf_tokenizer_to_symphony(model_path: &Path, info_file_path: &Path) -> Result<()> {
    info!("Converting HF tokenizer to Symphony format in: {:?}", model_path);

    let tokenizer_json_path = model_path.join("tokenizer.json");
    if !tokenizer_json_path.exists() {
        return Err(ManagementError::Service {
            message: "tokenizer.json not found in model directory".to_string(),
        });
    }

    // Parse HF tokenizer JSON
    let tokenizer_content = fs::read_to_string(&tokenizer_json_path).await
        .map_err(|e| ManagementError::Service {
            message: format!("Failed to read tokenizer.json: {}", e),
        })?;

    let hf_tokenizer: HfTokenizerJson = serde_json::from_str(&tokenizer_content)
        .map_err(|e| ManagementError::Service {
            message: format!("Failed to parse tokenizer.json: {}", e),
        })?;

    // Extract tokenizer metadata
    let tokenizer_metadata = extract_tokenizer_metadata(&hf_tokenizer)?;

    // Convert vocab from json into byte sequence and write tokenizer.model
    let tokenizer_model_path = model_path.join("tokenizer.model");
    // Check if tokenizer.model already exists
    if tokenizer_model_path.exists() {
        info!("tokenizer.model already exists, skipping conversion");
    } else {
        convert_vocab_to_bytes(&hf_tokenizer.model.vocab, &tokenizer_model_path).await?;
        info!("Created tokenizer.model from vocabulary");
    }

    // Update or create the model info file with tokenizer metadata
    update_symphony_model_info(info_file_path, &tokenizer_metadata).await?;

    info!("Successfully converted HF tokenizer to Symphony format");
    Ok(())
}

/// Extract tokenizer metadata from HuggingFace tokenizer configuration
pub fn extract_tokenizer_metadata(hf_tokenizer: &HfTokenizerJson) -> Result<TokenizerMetadata> {
    // Extract pre-tokenizer pattern
    let pre_tokenizer_pattern = hf_tokenizer.pre_tokenizer
        .as_ref()
        .and_then(|pt| pt.pretokenizers.as_ref())
        .and_then(|pts| pts.iter().find(|pt| pt.component_type == "Split"))
        .and_then(|pt| pt.pattern.as_ref())
        .and_then(|pattern| pattern.regex.clone());

    // Extract special tokens from post_processor
    let mut special_tokens = HashMap::new();
    if let Some(post_processor) = &hf_tokenizer.post_processor {
        if let Some(special_token_map) = &post_processor.special_tokens {
            for (token_str, token_info) in special_token_map {
                if let Some(first_id) = token_info.ids.first() {
                    special_tokens.insert(token_str.clone(), *first_id);
                }
            }
        }
    }

    // Extract added tokens
    let added_tokens: Vec<AddedTokenInfo> = hf_tokenizer.added_tokens
        .iter()
        .map(|token| AddedTokenInfo {
            id: token.id,
            content: token.content.clone(),
            special: token.special.unwrap_or(false),
        })
        .collect();

    // Also add special tokens to the special_tokens map
    for token in &added_tokens {
        if token.special {
            special_tokens.insert(token.content.clone(), token.id);
        }
    }

    Ok(TokenizerMetadata {
        model_type: hf_tokenizer.model.model_type.clone(),
        vocab_size: hf_tokenizer.model.vocab.len(),
        pre_tokenizer_pattern,
        special_tokens,
        added_tokens,
        post_processor_type: hf_tokenizer.post_processor
            .as_ref()
            .and_then(|pp| pp.post_processor_type.clone()),
        decoder_type: hf_tokenizer.decoder
            .as_ref()
            .and_then(|d| d.decoder_type.clone()),
    })
}

/// Update model info file with tokenizer metadata
async fn update_symphony_model_info(info_file_path: &Path, tokenizer_metadata: &TokenizerMetadata) -> Result<()> {
    // Read existing info or create new
    let mut model_info: serde_json::Value = if info_file_path.exists() {
        let existing_content = fs::read_to_string(info_file_path).await
            .map_err(|e| ManagementError::Service {
                message: format!("Failed to read existing model info file: {}", e),
            })?;
        
        serde_json::from_str(&existing_content)
            .map_err(|e| ManagementError::Service {
                message: format!("Failed to parse existing model info file: {}", e),
            })?
    } else {
        serde_json::json!({})
    };

    // Add tokenizer metadata
    // Set tokenizer_path to the parent directory of the info file (the model directory)
    let model_path = info_file_path.parent()
        .ok_or_else(|| ManagementError::Service {
            message: "Invalid info file path".to_string(),
        })?;
    model_info["tokenizer_path"] = serde_json::Value::String(model_path.to_string_lossy().to_string());
    model_info["tokenizer_metadata"] = serde_json::to_value(tokenizer_metadata)
        .map_err(|e| ManagementError::Service {
            message: format!("Failed to serialize tokenizer metadata: {}", e),
        })?;

    // Write updated info
    let info_content = serde_json::to_string_pretty(&model_info)
        .map_err(|e| ManagementError::Service {
            message: format!("Failed to serialize updated model info: {}", e),
        })?;

    fs::write(info_file_path, info_content).await
        .map_err(|e| ManagementError::Service {
            message: format!("Failed to write updated model info file: {}", e),
        })?;

    info!("Updated model info file with tokenizer metadata: {:?}", info_file_path);
    Ok(())
}

/// Create inverse mapping from unicode back to original bytes
/// This reverses the OpenAI bytes_to_unicode() transformation
fn unicode_to_bytes() -> HashMap<char, u8> {
    let mut byte_to_unicode = HashMap::new();
    
    // The 188 integers that render fine in their original form and need no shifting
    let mut bs = Vec::new();
    bs.extend(b'!'..=b'~');  // 33..126
    bs.extend(0xa1..=0xac);  // 161..172 (¡..¬)
    bs.extend(0xae..=0xff);  // 174..255 (®..ÿ)
    
    let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
    
    // Add the other 68 integers that need shifting
    let mut n = 0u32;
    for b in 0..=255u8 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    
    // Create mapping from shifted unicode back to original bytes
    for (byte_val, unicode_val) in bs.iter().zip(cs.iter()) {
        let unicode_char = char::from_u32(*unicode_val).unwrap();
        byte_to_unicode.insert(unicode_char, *byte_val);
    }
    
    byte_to_unicode
}

/// Convert vocabulary from UTF-8 strings to bytes and serialize to tokenizer.model
async fn convert_vocab_to_bytes(vocab: &HashMap<String, u32>, output_path: &Path) -> Result<()> {
    info!("Converting vocabulary from UTF-8 to bytes format");
    
    let unicode_to_byte_map = unicode_to_bytes();
    let mut converted_vocab: HashMap<Vec<u8>, u32> = HashMap::new();
    
    for (token_str, token_id) in vocab {
        // Convert the UTF-8 token string back to its original byte representation
        let mut byte_sequence = Vec::new();
        
        for ch in token_str.chars() {
            if let Some(&byte_val) = unicode_to_byte_map.get(&ch) {
                byte_sequence.push(byte_val);
            } else {
                warn!("Character '{}' not found in unicode_to_bytes mapping, using UTF-8 encoding", ch);
                // Fallback: encode as UTF-8 bytes
                byte_sequence.extend(ch.to_string().as_bytes());
            }
        }
        
        converted_vocab.insert(byte_sequence, *token_id);
    }
    
    info!("Converted {} vocabulary entries from UTF-8 to bytes", converted_vocab.len());
    
    // Serialize the converted vocabulary to tokenizer.model
    let serialized_data = serialize_vocab_to_model(&converted_vocab)?;
    
    fs::write(output_path, serialized_data).await
        .map_err(|e| ManagementError::Service {
            message: format!("Failed to write tokenizer.model: {}", e),
        })?;
    
    info!("Successfully wrote tokenizer.model to: {:?}", output_path);
    Ok(())
}

/// Serialize the byte-based vocabulary to Symphony's tokenizer.model format
/// Format: base64_encoded_token_bytes token_id (one per line, sorted by token_id)
fn serialize_vocab_to_model(vocab: &HashMap<Vec<u8>, u32>) -> Result<Vec<u8>> {
    use base64::{Engine as _, engine::general_purpose};
    
    // Sort entries by token_id for consistent output
    let mut sorted_entries: Vec<_> = vocab.iter().collect();
    sorted_entries.sort_by_key(|(_, &id)| id);
    
    let mut lines = Vec::new();
    
    // Write each vocabulary entry as: base64_token token_id
    for (token_bytes, &token_id) in sorted_entries {
        let base64_token = general_purpose::STANDARD.encode(token_bytes);
        let line = format!("{} {}\n", base64_token, token_id);
        lines.push(line);
    }
    
    let output = lines.join("");
    Ok(output.into_bytes())
}
