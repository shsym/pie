use base64::{Engine as _, engine::general_purpose};
use fancy_regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
pub type Rank = u32;

// Structures for deserializing tokenizer metadata from symphony_model_info.json
#[derive(Debug, Deserialize, Serialize)]
struct AddedTokenInfo {
    content: String,
    #[serde(default)]
    id: Option<u32>,
    #[serde(default)]
    special: bool,
}

#[derive(Debug, Deserialize, Serialize)]
struct TokenizerMetadata {
    model_type: String,
    vocab_size: usize,
    pre_tokenizer_pattern: Option<String>,
    special_tokens: HashMap<String, u32>,
    added_tokens: Vec<AddedTokenInfo>,
    post_processor_type: Option<String>,
    decoder_type: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SymphonyModelInfo {
    tokenizer_path: Option<String>,
    tokenizer_metadata: Option<TokenizerMetadata>,
    // ... other fields can be added as needed
}

// The code below is copied from the tiktoken.
// https://github.com/openai/tiktoken/blob/main/src/lib.rs

fn _byte_pair_merge(ranks: &HashMap<Vec<u8>, Rank>, piece: &[u8]) -> Vec<(usize, Rank)> {
    let mut parts = Vec::with_capacity(piece.len() + 1);

    let mut min_rank: (Rank, usize) = (Rank::MAX, usize::MAX);
    for i in 0..piece.len() - 1 {
        let rank = *ranks.get(&piece[i..i + 2]).unwrap_or(&Rank::MAX);
        if rank < min_rank.0 {
            min_rank = (rank, i);
        }
        parts.push((i, rank));
    }
    parts.push((piece.len() - 1, Rank::MAX));
    parts.push((piece.len(), Rank::MAX));

    let get_rank = {
        |parts: &Vec<(usize, Rank)>, i: usize| {
            if (i + 3) < parts.len() {
                *ranks
                    .get(&piece[parts[i].0..parts[i + 3].0])
                    .unwrap_or(&Rank::MAX)
            } else {
                Rank::MAX
            }
        }
    };

    while min_rank.0 != Rank::MAX {
        let i = min_rank.1;

        if i > 0 {
            parts[i - 1].1 = get_rank(&parts, i - 1);
        }
        parts[i].1 = get_rank(&parts, i);
        parts.remove(i + 1);

        min_rank = (Rank::MAX, usize::MAX);
        for (i, &(_, rank)) in parts[..parts.len() - 1].iter().enumerate() {
            if rank < min_rank.0 {
                min_rank = (rank, i);
            }
        }
    }
    parts
}

pub fn byte_pair_encode(piece: &[u8], ranks: &HashMap<Vec<u8>, Rank>) -> Vec<Rank> {
    if piece.len() == 1 {
        return vec![ranks[piece]];
    }
    _byte_pair_merge(ranks, piece)
        .windows(2)
        .map(|part| ranks[&piece[part[0].0..part[1].0]])
        .collect()
}

pub fn byte_pair_split<'a>(piece: &'a [u8], ranks: &HashMap<Vec<u8>, Rank>) -> Vec<&'a [u8]> {
    assert!(piece.len() > 1);
    _byte_pair_merge(ranks, piece)
        .windows(2)
        .map(|part| &piece[part[0].0..part[1].0])
        .collect()
}

#[derive(Debug, Clone)]
pub struct DecodeKeyError {
    pub token: Rank,
}

impl std::fmt::Display for DecodeKeyError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Invalid token for decoding: {}", self.token)
    }
}

impl std::error::Error for DecodeKeyError {}

#[derive(Debug, Clone)]
pub struct DecodeError {
    pub message: String,
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Could not decode tokens: {}", self.message)
    }
}

impl std::error::Error for DecodeError {}

#[derive(Clone, Debug)]
pub struct BytePairEncoder {
    encoder: HashMap<Vec<u8>, Rank>,
    special_tokens_encoder: HashMap<String, Rank>,
    decoder: HashMap<Rank, Vec<u8>>,
    special_tokens_decoder: HashMap<Rank, Vec<u8>>,
    regex: Regex,
    special_regex: Regex,
    sorted_token_bytes: Vec<Vec<u8>>,
}

impl BytePairEncoder {
    pub fn decode(&self, tokens: &[Rank]) -> Result<String, DecodeError> {
        // First, decode raw bytes from the tokens.
        let decoded_bytes = self.decode_bytes(tokens).map_err(|err| DecodeError {
            message: err.to_string(),
        })?;

        // Then, convert the bytes to a UTF-8 string.
        // Using `from_utf8_lossy` would silently replace invalid sequences with
        // the Unicode replacement character;
        let decoded_string = String::from_utf8_lossy(&*decoded_bytes).to_string();

        Ok(decoded_string)
    }
    fn decode_bytes(&self, tokens: &[Rank]) -> Result<Vec<u8>, DecodeKeyError> {
        let mut ret = Vec::with_capacity(tokens.len() * 2);
        for &token in tokens {
            let token_bytes = match self.decoder.get(&token) {
                Some(bytes) => bytes,
                None => self
                    .special_tokens_decoder
                    .get(&token)
                    .ok_or(DecodeKeyError { token })?,
            };
            ret.extend(token_bytes);
        }
        Ok(ret)
    }

    pub fn encode(&self, text: &str, allowed_special: &HashSet<&str>) -> Vec<Rank> {
        let mut ret = vec![];

        let mut start = 0;
        let mut last_piece_token_len = 0;
        loop {
            let mut next_special;
            let mut start_find = start;
            loop {
                next_special = self.special_regex.find_from_pos(text, start_find).unwrap();
                match next_special {
                    Some(m) => {
                        if allowed_special.contains(&text[m.start()..m.end()]) {
                            break;
                        }
                        start_find = m.start() + 1;
                    }
                    None => break,
                }
            }
            let end = next_special.map_or(text.len(), |m| m.start());

            for mat in self.regex.find_iter(&text[start..end]) {
                let piece = mat.unwrap().as_str().as_bytes();
                if let Some(token) = self.encoder.get(piece) {
                    last_piece_token_len = 1;
                    ret.push(*token);
                    continue;
                }
                let tokens = byte_pair_encode(piece, &self.encoder);
                last_piece_token_len = tokens.len();
                ret.extend(&tokens);
            }

            match next_special {
                // And here we push the special token
                Some(m) => {
                    let piece = m.as_str();
                    let token = self.special_tokens_encoder[piece];
                    ret.push(token);
                    start = m.end();
                    last_piece_token_len = 0;
                }
                None => break,
            }
        }

        ret
    }

    fn new(
        encoder: HashMap<Vec<u8>, Rank>,
        special_tokens_encoder: HashMap<String, Rank>,
        pattern: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let regex = Regex::new(pattern)
            .map_err(|e| format!("Failed to compile regex pattern '{}': {}", pattern, e))?;

        let special_regex = {
            let parts = special_tokens_encoder
                .keys()
                .map(|s| fancy_regex::escape(s))
                .collect::<Vec<_>>();
            Regex::new(&parts.join("|"))
                .map_err(|e| format!("Failed to compile special tokens regex: {}", e))?
        };

        let decoder: HashMap<Rank, Vec<u8>> =
            encoder.iter().map(|(k, v)| (*v, k.clone())).collect();

        assert_eq!(
            encoder.len(),
            decoder.len(),
            "Encoder and decoder must be of equal length; maybe you had duplicate token indices in your encoder?"
        );

        let special_tokens_decoder: HashMap<Rank, Vec<u8>> = special_tokens_encoder
            .iter()
            .map(|(k, v)| (*v, k.as_bytes().to_vec()))
            .collect();

        // Clone because I don't know how to tell Rust I'm not going to change the map
        let mut sorted_token_bytes: Vec<Vec<u8>> = encoder.keys().cloned().collect();
        sorted_token_bytes.sort();

        Ok(Self {
            encoder,
            special_tokens_encoder,
            decoder,
            special_tokens_decoder,
            regex,
            special_regex,
            sorted_token_bytes,
        })
    }

    pub fn special_tokens(&self) -> HashSet<&str> {
        self.special_tokens_encoder
            .keys()
            .map(|s| s.as_str())
            .collect()
    }

    pub fn encode_with_special_tokens(&self, text: &str) -> Vec<Rank> {
        let allowed_special = self.special_tokens();
        self.encode(text, &allowed_special)
    }

    pub fn get_vocabs(&self) -> Vec<Vec<u8>> {
        self.sorted_token_bytes.clone()
    }
}

fn load_merge_rules(path: &str) -> Result<HashMap<Vec<u8>, Rank>, Box<dyn std::error::Error>> {
    // Read the entire file as a UTF-8 string
    let contents = fs::read_to_string(path)?;

    let mut ret = HashMap::new();

    for (line_number, line) in contents.lines().enumerate() {
        let line = line.trim();
        // Skip empty lines
        if line.is_empty() {
            continue;
        }

        // Expect two parts: base64-encoded token and rank
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() != 2 {
            return Err(format!(
                "Error parsing line {}: expected two parts, got {} (line: {:?})",
                line_number,
                parts.len(),
                line
            )
            .into());
        }

        let b64_token = parts[0];
        let rank_str = parts[1];

        // Decode base64 token
        let decoded_token = general_purpose::STANDARD
            .decode(b64_token)
            .map_err(|e| format!("Error decoding base64 at line {}: {}", line_number, e))?;

        // Parse rank into i32
        let rank = rank_str
            .parse::<Rank>()
            .map_err(|e| format!("Error parsing rank at line {}: {}", line_number, e))?;

        // Insert into the HashMap
        ret.insert(decoded_token, rank);
    }

    Ok(ret)
}

// https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py
/// Create a configurable tokenizer using metadata from symphony_model_info.json
pub fn configurable_tokenizer(
    tokenizer_model_path: &str,
    metadata: &TokenizerMetadata,
) -> Result<BytePairEncoder, Box<dyn std::error::Error>> {
    let mergeable_ranks = load_merge_rules(tokenizer_model_path)?;

    // Require a valid pre-tokenizer pattern from metadata
    let pattern = metadata.pre_tokenizer_pattern
        .as_deref()
        .ok_or("No pre_tokenizer_pattern found in model metadata file")?;

    // Validate the regex pattern from metadata
    fancy_regex::Regex::new(pattern)
        .map_err(|e| format!("Invalid regex pattern in model metadata file: '{}' - {}", pattern, e))?;


    println!("Using tokenizer pattern: {}", pattern);

    // Create special tokens encoder from metadata
    let mut special_tokens_encoder = HashMap::new();

    // Add special tokens from the special_tokens map
    for (token, id) in &metadata.special_tokens {
        special_tokens_encoder.insert(token.clone(), *id);
    }

    // Add special tokens from added_tokens list (if they have IDs)
    for added_token in &metadata.added_tokens {
        if added_token.special {
            if let Some(id) = added_token.id {
                special_tokens_encoder.insert(added_token.content.clone(), id);
            } else {
                // For added tokens without explicit IDs, assign them after the base vocabulary
                let next_id = mergeable_ranks.len() as Rank + special_tokens_encoder.len() as Rank;
                special_tokens_encoder.insert(added_token.content.clone(), next_id);
            }
        }
    }

    // If no special tokens were found in metadata, fall back to hardcoded Llama3 tokens
    if special_tokens_encoder.is_empty() {
        println!("No special tokens found in metadata, falling back to default Llama3 tokens");
        let default_special_tokens = vec![
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",
        ];

        let num_base_tokens = mergeable_ranks.len() as Rank;
        special_tokens_encoder = default_special_tokens
            .into_iter()
            .enumerate()
            .map(|(i, s)| (s.to_string(), num_base_tokens + i as Rank))
            .collect();
    }

    println!("Created tokenizer with {} base tokens and {} special tokens",
             mergeable_ranks.len(), special_tokens_encoder.len());

    let encoder = BytePairEncoder::new(mergeable_ranks, special_tokens_encoder, pattern)?;
    Ok(encoder)
}

/// Legacy tokenizer function for backward compatibility
pub fn llama3_tokenizer(path: &str) -> Result<BytePairEncoder, Box<dyn std::error::Error>> {
    // Create default metadata for legacy calls
    let default_metadata = TokenizerMetadata {
        model_type: "BPE".to_string(),
        vocab_size: 0, // Will be determined from merge rules
        pre_tokenizer_pattern: Some(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+".to_string()),
        special_tokens: HashMap::new(),
        added_tokens: vec![
            AddedTokenInfo { content: "<|begin_of_text|>".to_string(), id: None, special: true },
            AddedTokenInfo { content: "<|end_of_text|>".to_string(), id: None, special: true },
            AddedTokenInfo { content: "<|reserved_special_token_0|>".to_string(), id: None, special: true },
            AddedTokenInfo { content: "<|reserved_special_token_1|>".to_string(), id: None, special: true },
            AddedTokenInfo { content: "<|reserved_special_token_2|>".to_string(), id: None, special: true },
            AddedTokenInfo { content: "<|reserved_special_token_3|>".to_string(), id: None, special: true },
            AddedTokenInfo { content: "<|start_header_id|>".to_string(), id: None, special: true },
            AddedTokenInfo { content: "<|end_header_id|>".to_string(), id: None, special: true },
            AddedTokenInfo { content: "<|reserved_special_token_4|>".to_string(), id: None, special: true },
            AddedTokenInfo { content: "<|eot_id|>".to_string(), id: None, special: true },
        ],
        post_processor_type: None,
        decoder_type: None,
    };

    configurable_tokenizer(path, &default_metadata)
}

/// Recursively search for tokenizer.model in a directory and its subdirectories
fn find_tokenizer_model_recursive(dir_path: &str) -> Result<String, Box<dyn std::error::Error>> {
    use std::fs;

    fn search_dir(dir: &Path) -> Result<String, Box<dyn std::error::Error>> {
        // Check if tokenizer.model exists in current directory
        let tokenizer_model = dir.join("tokenizer.model");
        if tokenizer_model.exists() {
            return Ok(tokenizer_model.to_string_lossy().to_string());
        }

        // Recursively search subdirectories
        if dir.is_dir() {
            for entry in fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_dir() {
                    if let Ok(found) = search_dir(&path) {
                        return Ok(found);
                    }
                }
            }
        }

        Err("tokenizer.model not found".into())
    }

    search_dir(Path::new(dir_path))
}

/// Load tokenizer from either Symphony model metadata or HuggingFace model directory
/// This function tries to detect the tokenizer type and load it appropriately
pub fn load_symphony_tokenizer(model_path: &str) -> Result<BytePairEncoder, Box<dyn std::error::Error>> {
    let model_dir = Path::new(model_path);

    // First, try to read Symphony model metadata for precise tokenizer path and configuration
    let json_info = model_dir.join("symphony_model_info.json");
    if json_info.exists() {
        let content = std::fs::read_to_string(&json_info)?;
        let info: SymphonyModelInfo = serde_json::from_str(&content)?;

        // If we have tokenizer metadata, use the configurable tokenizer
        if let Some(metadata) = &info.tokenizer_metadata {
            println!("Found tokenizer metadata in Symphony model info");

            // Try to find the tokenizer.model file
            let mut tokenizer_model_path = None;

            // First check if explicit tokenizer_path is provided
            if let Some(tokenizer_path) = &info.tokenizer_path {
                let tokenizer_dir = Path::new(tokenizer_path);
                let tokenizer_model = tokenizer_dir.join("tokenizer.model");
                if tokenizer_model.exists() {
                    tokenizer_model_path = Some(tokenizer_model.to_string_lossy().to_string());
                } else if let Ok(found_tokenizer) = find_tokenizer_model_recursive(tokenizer_path) {
                    tokenizer_model_path = Some(found_tokenizer);
                }
            }

            // If not found via explicit path, search in model directory
            if tokenizer_model_path.is_none() {
                let tokenizer_model = model_dir.join("tokenizer.model");
                if tokenizer_model.exists() {
                    tokenizer_model_path = Some(tokenizer_model.to_string_lossy().to_string());
                } else if let Ok(found_tokenizer) = find_tokenizer_model_recursive(model_path) {
                    tokenizer_model_path = Some(found_tokenizer);
                }
            }

            if let Some(path) = tokenizer_model_path {
                println!("Loading configurable tokenizer from: {}", path);
                println!("Model type: {}, Vocab size: {}", metadata.model_type, metadata.vocab_size);
                return configurable_tokenizer(&path, metadata);
            } else {
                println!("Warning: tokenizer metadata found but tokenizer.model file not found");
            }
        }

        // Fallback to legacy loading if tokenizer_path exists but no metadata
        if let Some(tokenizer_path) = &info.tokenizer_path {
            println!("Loading tokenizer from Symphony metadata (legacy mode): {}", tokenizer_path);
            let tokenizer_dir = Path::new(tokenizer_path);

            // Check for tokenizer.model in the specified path
            let tokenizer_model = tokenizer_dir.join("tokenizer.model");
            if tokenizer_model.exists() {
                println!("Found tokenizer.model at specified path, attempting to load as SentencePiece/Llama tokenizer");
                return llama3_tokenizer(tokenizer_model.to_str().unwrap());
            }

            // If not found directly, search recursively in the tokenizer path
            if let Ok(found_tokenizer) = find_tokenizer_model_recursive(tokenizer_path) {
                println!("Found tokenizer.model in subdirectory: {}", found_tokenizer);
                return llama3_tokenizer(&found_tokenizer);
            }
        }
    }

    // Fallback to standard HuggingFace model directory detection
    // Check for different tokenizer files that might exist in the model directory
    let tokenizer_model = model_dir.join("tokenizer.model");
    let tokenizer_json = model_dir.join("tokenizer.json");
    let vocab_file = model_dir.join("vocab.txt");

    // Try SentencePiece tokenizer first (common for Llama, T5, etc.)
    if tokenizer_model.exists() {
        println!("Found tokenizer.model, attempting to load as SentencePiece/Llama tokenizer");
        return llama3_tokenizer(tokenizer_model.to_str().unwrap());
    }

    // If not found in the root directory, search recursively in subdirectories
    if let Ok(found_tokenizer) = find_tokenizer_model_recursive(model_path) {
        println!("Found tokenizer.model in subdirectory: {}", found_tokenizer);
        return llama3_tokenizer(&found_tokenizer);
    }

    // For now, fall back to the hardcoded test tokenizer
    // TODO: Implement support for other tokenizer formats (tokenizer.json, vocab.txt)
    if tokenizer_json.exists() {
        println!("Found tokenizer.json, but HuggingFace tokenizer.json support is not yet implemented");
        println!("Falling back to default tokenizer");
    }

    if vocab_file.exists() {
        println!("Found vocab.txt, but vocab.txt support is not yet implemented");
        println!("Falling back to default tokenizer");
    }

    // Fallback to test tokenizer
    let fallback_paths = [
        "../test-tokenizer/tokenizer.model",
        "test-tokenizer/tokenizer.model",
    ];

    for path in &fallback_paths {
        if Path::new(path).exists() {
            println!("Using fallback tokenizer from {}", path);
            return llama3_tokenizer(path);
        }
    }

    Err("No suitable tokenizer found".into())
}


