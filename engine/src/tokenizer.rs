use base64::{Engine as _, engine::general_purpose};
use fancy_regex::Regex;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf}; // Added PathBuf
use tokenizers::tokenizer::{Result as TokenizerResult, Tokenizer}; // Added
use tokenizers::models::bpe::BPE; // Added for BPE model access, might need others

pub type Rank = u32;

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
    // For tokenizers loaded from tokenizer.json
    hf_tokenizer: Option<Tokenizer>,

    // Existing fields for manually constructed tokenizers (like llama3_tokenizer)
    encoder: HashMap<Vec<u8>, Rank>,
    special_tokens_encoder: HashMap<String, Rank>,
    decoder: HashMap<Rank, Vec<u8>>,
    special_tokens_decoder: HashMap<Rank, Vec<u8>>,
    regex: Option<Regex>, // Made Option
    special_regex: Option<Regex>, // Made Option
    sorted_token_bytes: Vec<Vec<u8>>,
}

impl BytePairEncoder {
    pub fn decode(&self, tokens: &[Rank]) -> Result<String, DecodeError> {
        if let Some(ref hf_tok) = self.hf_tokenizer {
            // Assuming tokens are u32 IDs. The decode method in tokenizers crate takes &[u32].
            let allowed_special = self.special_tokens();
            hf_tok.decode(tokens, !allowed_special.is_empty()).map_err(|e| DecodeError { message: format!("HuggingFace decode error: {}", e) })
        } else {
            // Fallback to existing manual decoding logic
            let decoded_bytes = self.decode_bytes(tokens).map_err(|err| DecodeError {
                message: err.to_string(),
            })?;
            String::from_utf8(decoded_bytes).map_err(|e| DecodeError { message: format!("UTF-8 conversion error: {}", e) })
        }
    }

    fn decode_bytes(&self, tokens: &[Rank]) -> Result<Vec<u8>, DecodeKeyError> {
        if self.hf_tokenizer.is_some() {
            eprintln!("Warning: decode_bytes called when hf_tokenizer is present. This indicates a logic error.");
            return Err(DecodeKeyError{ token: Rank::MAX }); // Use Rank::MAX to indicate this specific error state
        }
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
        if let Some(ref hf_tok) = self.hf_tokenizer {
            let allowed_special = self.special_tokens();
            match hf_tok.encode(text, !allowed_special.is_empty()) {
                Ok(encoding) => encoding.get_ids().to_vec(),
                Err(e) => {
                    eprintln!("HuggingFace encode error: {}", e);
                    vec![]
                }
            }
        } else {
            // Fallback to existing manual encoding logic
            let mut ret = vec![];
            let mut current_pos = 0; // Tracks the current position in the input `text`

            loop { // Outer loop to process text in segments (normal + special)
                let mut next_special_token_match: Option<fancy_regex::Match<'_>> = None;

                // Phase 1: Find the next allowed special token
                if let Some(ref spec_regex) = self.special_regex {
                    let mut search_offset = current_pos;
                    loop { // Inner loop to find an *allowed* special token
                        if search_offset >= text.len() { // Optimization: stop if past end of text
                            break;
                        }
                        match spec_regex.find_from_pos(text, search_offset) {
                            Ok(Some(m)) => {
                                if allowed_special.contains(&text[m.start()..m.end()]) {
                                    next_special_token_match = Some(m);
                                    break; // Found allowed special token
                                }
                                // Special token found, but not in allowed_special. Continue search.
                                search_offset = m.start() + 1;
                            }
                            Ok(None) => break, // No more special tokens from this position
                            Err(_e) => {
                                // eprintln!("Special regex execution error: {}", _e);
                                break; // Error in regex, stop searching for special tokens here
                            }
                        }
                    }
                }

                // Phase 2: Process the text segment before the found special token (or to the end)
                let segment_end = next_special_token_match.as_ref().map_or(text.len(), |m| m.start());

                if current_pos < segment_end {
                    if let Some(ref main_r) = self.regex {
                        for mat_result in main_r.find_iter(&text[current_pos..segment_end]) {
                            match mat_result {
                                Ok(mat) => {
                                    let piece = mat.as_str().as_bytes();
                                    if piece.is_empty() { continue; }

                                    if let Some(token) = self.encoder.get(piece) {
                                        ret.push(*token);
                                    } else {
                                        let tokens = byte_pair_encode(piece, &self.encoder);
                                        ret.extend(&tokens);
                                    }
                                }
                                Err(_e) => { /* eprintln!("Main regex find_iter error: {}", _e); */ }
                            }
                        }
                    } else {
                        // eprintln!("Warning: Main regex (self.regex) is None during manual encoding of segment: '{}'", &text[current_pos..segment_end]);
                        // If main_r is None, this segment cannot be tokenized by the BPE logic.
                        // Depending on desired behavior, one might add raw bytes or specific unknown tokens.
                        // For now, it's skipped as per original implicit behavior if regex was None.
                    }
                }

                // Phase 3: Process the special token, if found
                if let Some(m) = next_special_token_match {
                    let special_piece = m.as_str();
                    if let Some(token) = self.special_tokens_encoder.get(special_piece) {
                        ret.push(*token);
                    } else {
                        // eprintln!("Warning: Special token '{}' matched by regex but not in encoder.", special_piece);
                    }
                    current_pos = m.end(); 
                    if current_pos >= text.len() {
                        break; // Reached or passed end of text
                    }
                } else {
                    // No more special tokens found (or no special_regex)
                    break; // Exit the outer loop
                }
            }
            ret
        }
    }

    // Constructor for manually built BPE (like Llama3 specific one)
    fn new_manual(
        encoder: HashMap<Vec<u8>, Rank>,
        special_tokens_encoder: HashMap<String, Rank>,
        pattern: &str,
    ) -> Self {
        let regex = Regex::new(pattern).ok();
        let special_regex = {
            if !special_tokens_encoder.is_empty() {
                let parts = special_tokens_encoder
                    .keys()
                    .map(|s| fancy_regex::escape(s))
                    .collect::<Vec<_>>();
                Regex::new(&parts.join("|")).ok()
            } else {
                None
            }
        };

        let decoder: HashMap<Rank, Vec<u8>> =
            encoder.iter().map(|(k, v)| (*v, k.clone())).collect();

        let special_tokens_decoder: HashMap<Rank, Vec<u8>> = special_tokens_encoder
            .iter()
            .map(|(k, v)| (*v, k.as_bytes().to_vec()))
            .collect();

        let mut sorted_token_bytes: Vec<Vec<u8>> = encoder.keys().cloned().collect();
        sorted_token_bytes.sort();

        Self {
            hf_tokenizer: None,
            encoder,
            special_tokens_encoder,
            decoder,
            special_tokens_decoder,
            regex,
            special_regex,
            sorted_token_bytes,
        }
    }

    // Constructor for HF Tokenizer
    fn new_hf(tokenizer: Tokenizer) -> Self {
        // For HF tokenizers, most of the detailed fields are managed internally by `tokenizer`.
        // We might need to extract some info if other parts of your code rely on them.
        // For now, create a minimal BytePairEncoder.
        Self {
            hf_tokenizer: Some(tokenizer),
            encoder: HashMap::new(), // Not directly used
            special_tokens_encoder: HashMap::new(), // Not directly used
            decoder: HashMap::new(), // Not directly used
            special_tokens_decoder: HashMap::new(), // Not directly used
            regex: None, // Not directly used
            special_regex: None, // Not directly used
            sorted_token_bytes: Vec::new(), // Not directly used
        }
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
pub fn llama3_tokenizer(path: &str) -> Result<BytePairEncoder, Box<dyn std::error::Error>> {
    // Example usage
    let mergeable_ranks = load_merge_rules(path)?;
    let special_tokens = vec![
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
    let pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    let num_base_tokens = mergeable_ranks.len() as Rank;

    let special_tokens_encoder: HashMap<String, Rank> = special_tokens
        .into_iter()
        .enumerate()
        .map(|(i, s)| (s.to_string(), num_base_tokens + i as Rank))
        .collect();

    let encoder = BytePairEncoder::new_manual(mergeable_ranks, special_tokens_encoder, pattern);
    // [9906, 11, 856, 5679, 374, 19369]
    // encode text
    //let text = "Hello, my dog is cute";
    //let tokens = encoder.encode_with_special_tokens(text);
    //println!("Encoded tokens: {:?}", tokens);

    Ok(encoder)
}

/// Attempt to load a tokenizer from a HuggingFace model directory
/// This function tries to detect the tokenizer type and load it appropriately
pub fn load_hf_tokenizer(model_path_or_file: &str) -> Result<BytePairEncoder, Box<dyn std::error::Error>> {
    let path = Path::new(model_path_or_file);

    // Determine if model_path_or_file is a directory or a specific file
    let tokenizer_json_path: PathBuf;
    let tokenizer_model_path: PathBuf;

    if path.is_dir() {
        tokenizer_json_path = path.join("tokenizer.json");
        tokenizer_model_path = path.join("tokenizer.model");
        println!("Searching for tokenizer in directory: {:?}", path);
    } else if path.is_file() {
        if path.file_name().unwrap_or_default() == "tokenizer.json" {
            tokenizer_json_path = path.to_path_buf();
            tokenizer_model_path = path.with_file_name("tokenizer.model"); // Sibling .model file
             println!("Provided path is tokenizer.json: {:?}", tokenizer_json_path);
        } else if path.file_name().unwrap_or_default() == "tokenizer.model" {
            tokenizer_model_path = path.to_path_buf();
            tokenizer_json_path = path.with_file_name("tokenizer.json"); // Sibling .json file
            println!("Provided path is tokenizer.model: {:?}", tokenizer_model_path);
        } else {
            // Assuming it's a model file for llama3_tokenizer if it's not tokenizer.json
            println!("Provided path is a file, attempting to load as Llama3/SentencePiece: {:?}", path);
            let path_str = path.to_str().ok_or_else(|| format!("Path contains invalid UTF-8: {:?}", path))?;
            return llama3_tokenizer(path_str);
        }
    } else {
        return Err(format!("Tokenizer path is not a valid file or directory: {}", model_path_or_file).into());
    }

    // Attempt 1: Try loading tokenizer.json using Hugging Face tokenizers crate
    if tokenizer_json_path.exists() {
        println!("Found tokenizer.json, attempting to load with Hugging Face tokenizers: {:?}", tokenizer_json_path);
        match Tokenizer::from_file(&tokenizer_json_path) {
            Ok(hf_tok) => {
                println!("Successfully loaded tokenizer.json from {:?}", tokenizer_json_path);
                return Ok(BytePairEncoder::new_hf(hf_tok));
            }
            Err(e) => {
                println!("Failed to load tokenizer.json with Hugging Face tokenizers from {:?}: {}. Falling back.", tokenizer_json_path, e);
            }
        }
    }

    // Attempt 2: Try SentencePiece tokenizer.model (Llama3 style)
    if tokenizer_model_path.exists() {
        println!("Found tokenizer.model, attempting to load as SentencePiece/Llama tokenizer: {:?}", tokenizer_model_path);
        let model_path_str = tokenizer_model_path.to_str().ok_or_else(|| format!("Path contains invalid UTF-8: {:?}", tokenizer_model_path))?;
        return llama3_tokenizer(model_path_str);
    }
    
    // Fallback to searching in parent directory if original path was a specific file
    if path.is_file() && path.parent().is_some() {
        let parent_dir = path.parent().unwrap(); // Safe due to is_some() check
        let tokenizer_json_in_parent = parent_dir.join("tokenizer.json");
        if tokenizer_json_in_parent.exists() && tokenizer_json_in_parent != tokenizer_json_path { 
             println!("Attempting to load tokenizer.json from parent directory: {:?}", tokenizer_json_in_parent);
             match Tokenizer::from_file(&tokenizer_json_in_parent) {
                Ok(hf_tok) => {
                    println!("Successfully loaded tokenizer.json from {:?}", tokenizer_json_in_parent);
                    return Ok(BytePairEncoder::new_hf(hf_tok));
                }
                Err(e) => {
                     println!("Failed to load tokenizer.json with Hugging Face tokenizers from {:?}: {}", tokenizer_json_in_parent, e);
                }
            }
        }
        let tokenizer_model_in_parent = parent_dir.join("tokenizer.model");
        if tokenizer_model_in_parent.exists() && tokenizer_model_in_parent != tokenizer_model_path {
            println!("Attempting to load tokenizer.model from parent directory: {:?}", tokenizer_model_in_parent);
            let model_parent_path_str = tokenizer_model_in_parent.to_str().ok_or_else(|| format!("Path contains invalid UTF-8: {:?}", tokenizer_model_in_parent))?;
            return llama3_tokenizer(model_parent_path_str);
        }
    }

    // Fallback to hardcoded test tokenizer (if any, or remove this section)
    let fallback_paths = [
        "../test-tokenizer/tokenizer.model", // Adjust as needed
        "test-tokenizer/tokenizer.model",
    ];
    for fallback_path_str in &fallback_paths {
        let fallback_path = Path::new(fallback_path_str);
        if fallback_path.exists() {
            println!("Using fallback Llama3/SentencePiece tokenizer from {}", fallback_path_str);
            return llama3_tokenizer(fallback_path_str);
        }
    }

    Err(format!("No suitable tokenizer found for path: {}", model_path_or_file).into())
}

/// Load tokenizer from Symphony model metadata
pub fn load_symphony_tokenizer(model_info_path_or_tokenizer_file: &str) -> Result<BytePairEncoder, Box<dyn std::error::Error>> {
    println!("load_symphony_tokenizer called with: {}", model_info_path_or_tokenizer_file);
    let path = Path::new(model_info_path_or_tokenizer_file);

    // Check if the provided path is directly a tokenizer file (e.g. tokenizer.json or tokenizer.model)
    if path.is_file() && (path.file_name().unwrap_or_default() == "tokenizer.json" || path.file_name().unwrap_or_default() == "tokenizer.model") {
        println!("Path appears to be a direct tokenizer file, attempting to load with load_hf_tokenizer: {:?}", path);
        return load_hf_tokenizer(model_info_path_or_tokenizer_file);
    }

    let symphony_metadata_file = path.join("symphony_model_info.json");
    println!("Looking for symphony_model_info.json at: {:?}", symphony_metadata_file);

    if symphony_metadata_file.exists() {
        let content = fs::read_to_string(&symphony_metadata_file)?;
        let info: serde_json::Value = serde_json::from_str(&content)?;

        if let Some(tokenizer_path_str) = info.get("tokenizer_path").and_then(|p| p.as_str()) {
            println!("Found tokenizer_path in symphony_model_info.json: {}", tokenizer_path_str);
            
            let mut absolute_tokenizer_path = PathBuf::from(tokenizer_path_str);
            if !absolute_tokenizer_path.is_absolute() {
                if let Some(base_dir) = symphony_metadata_file.parent() {
                    absolute_tokenizer_path = base_dir.join(tokenizer_path_str);
                     println!("Resolved relative tokenizer_path to: {:?}", absolute_tokenizer_path);
                } else {
                    return Err(format!("Could not get parent directory of symphony_model_info.json: {:?}", symphony_metadata_file).into());
                }
            }
            println!("Attempting to load tokenizer specified in metadata: {:?}", absolute_tokenizer_path);
            let abs_path_str = absolute_tokenizer_path.to_str().ok_or_else(|| format!("Path contains invalid UTF-8: {:?}", absolute_tokenizer_path))?;
            return load_hf_tokenizer(abs_path_str);
        } else {
            println!("'tokenizer_path' not found in {:?}, will try to load tokenizer from the directory itself.", symphony_metadata_file);
        }
    } else {
        println!("{:?} not found.", symphony_metadata_file);
    }
    
    println!("Falling back to load_hf_tokenizer with the original path: {}", model_info_path_or_tokenizer_file);
    load_hf_tokenizer(model_info_path_or_tokenizer_file)
}
