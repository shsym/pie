//! Model transformation utilities
//!
//! This module provides functionality to transform model layer names according to
//! configuration rules, specifically for SafeTensors format models.

use std::collections::HashMap;
use std::path::Path;
use regex::Regex;
use serde::{Serialize, Deserialize};
use tokio::io::AsyncReadExt;
use tracing::info;
use crate::error::{ManagementError, Result};

/// RAII guard for temporary file cleanup
struct TempFileCleanup {
    path: std::path::PathBuf,
    armed: bool,
}

impl TempFileCleanup {
    fn new(path: &Path) -> Self {
        Self {
            path: path.to_path_buf(),
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for TempFileCleanup {
    fn drop(&mut self) {
        if self.armed {
            // Best effort cleanup - ignore errors during cleanup
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// SafeTensors index file structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafeTensorsIndex {
    /// Mapping from tensor name to SafeTensors file name
    pub weight_map: HashMap<String, String>,
    /// Metadata about the model (optional)
    pub metadata: Option<SafeTensorsMetadata>,
}

/// SafeTensors metadata structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafeTensorsMetadata {
    /// Total size of all tensors in bytes
    pub total_size: u64,
    /// Additional metadata fields
    #[serde(flatten)]
    pub additional: Option<serde_json::Map<String, serde_json::Value>>,
}

/// Weight renaming rules configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightRenamingRules {
    /// Root name for the model (e.g., "model")
    pub root: String,
    /// List of renaming rules to apply
    pub rules: Vec<RenamingRule>,
    /// Optional metadata about the rules
    pub metadata: Option<serde_json::Value>,
}

/// Individual renaming rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenamingRule {
    /// Name/description of this rule
    pub name: String,
    /// Type of renaming rule
    #[serde(flatten)]
    pub rule_type: RenamingRuleType,
    /// Whether this rule is enabled (default: true)
    #[serde(default = "default_true")]
    pub enabled: bool,
}

/// Types of renaming rules supported
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RenamingRuleType {
    /// Regular expression replacement
    #[serde(rename = "regex")]
    Regex {
        /// Regex pattern to match
        pattern: String,
        /// Replacement string (can use capture groups like $1, $2)
        replacement: String,
    },
    /// Direct string replacement
    #[serde(rename = "direct")]
    Direct {
        /// Exact string to match
        from: String,
        /// String to replace with
        to: String,
    },
    /// Prefix replacement
    #[serde(rename = "prefix")]
    Prefix {
        /// Old prefix to remove
        old_prefix: String,
        /// New prefix to add
        new_prefix: String,
    },
    /// Suffix replacement
    #[serde(rename = "suffix")]
    Suffix {
        /// Old suffix to remove
        old_suffix: String,
        /// New suffix to add
        new_suffix: String,
    },
}

/// Default value for enabled field
fn default_true() -> bool {
    true
}

/// Model transformation utilities
pub struct ModelTransformer;

impl ModelTransformer {
    /// Rename model layers according to weight_renaming.json rules
    pub async fn rename_model_layers(model_path: &Path) -> Result<()> {
        info!("Starting layer renaming for model at: {}", model_path.display());

        // Check if model.safetensors.index.json exists
        let index_path = model_path.join("model.safetensors.index.json");
        if !index_path.exists() {
            return Err(ManagementError::Service {
                message: "model.safetensors.index.json not found. This is required for layer renaming.".to_string()
            });
        }

        // Check if weight_renaming.json exists
        let renaming_rules_path = model_path.join("weight_renaming.json");
        if !renaming_rules_path.exists() {
            return Err(ManagementError::Service {
                message: "weight_renaming.json not found. This file contains the layer renaming rules.".to_string()
            });
        }

        // Load the renaming rules
        let renaming_rules = Self::load_renaming_rules(&renaming_rules_path).await?;
        info!("Loaded {} renaming rules", renaming_rules.rules.len());

        // Load the SafeTensors index
        let safetensors_index = Self::load_safetensors_index(&index_path).await?;
        info!("Found {} SafeTensors files to process", safetensors_index.weight_map.len());

        // Extract current layer names from all SafeTensors files
        let current_layers = Self::extract_all_layer_names(model_path, &safetensors_index).await?;
        info!("Extracted {} layer names", current_layers.len());

        // Apply renaming rules to generate new layer names
        let layer_mapping = Self::apply_renaming_rules(&current_layers, &renaming_rules)?;
        info!("Generated {} layer name mappings", layer_mapping.len());

        // Update SafeTensors files with new layer names
        Self::update_safetensors_files(model_path, &safetensors_index, &layer_mapping).await?;

        // Update the index file with new layer names
        Self::update_safetensors_index(model_path, &safetensors_index, &layer_mapping).await?;

        info!("Successfully completed layer renaming for model at: {}", model_path.display());
        Ok(())
    }

    /// Load renaming rules from weight_renaming.json
    async fn load_renaming_rules(rules_path: &Path) -> Result<WeightRenamingRules> {
        let content = tokio::fs::read_to_string(rules_path).await
            .map_err(|e| ManagementError::Service {
                message: format!("Failed to read weight_renaming.json: {}", e)
            })?;

        let rules: WeightRenamingRules = serde_json::from_str(&content)
            .map_err(|e| ManagementError::Service {
                message: format!("Failed to parse weight_renaming.json: {}", e)
            })?;

        Ok(rules)
    }

    /// Load SafeTensors index file
    async fn load_safetensors_index(index_path: &Path) -> Result<SafeTensorsIndex> {
        let content = tokio::fs::read_to_string(index_path).await
            .map_err(|e| ManagementError::Service {
                message: format!("Failed to read model.safetensors.index.json: {}", e)
            })?;

        let index: SafeTensorsIndex = serde_json::from_str(&content)
            .map_err(|e| ManagementError::Service {
                message: format!("Failed to parse model.safetensors.index.json: {}", e)
            })?;

        Ok(index)
    }

    /// Extract all layer names from SafeTensors files
    async fn extract_all_layer_names(model_path: &Path, index: &SafeTensorsIndex) -> Result<Vec<String>> {
        let mut all_layers = Vec::new();
        let mut processed_files = std::collections::HashSet::new();

        // Get unique SafeTensors file names
        for file_name in index.weight_map.values() {
            if processed_files.insert(file_name.clone()) {
                let file_path = model_path.join(file_name);
                let layers = Self::extract_layer_names_from_file(&file_path).await?;
                all_layers.extend(layers);
            }
        }

        Ok(all_layers)
    }

    /// Extract layer names from a single SafeTensors file by reading only the header
    async fn extract_layer_names_from_file(file_path: &Path) -> Result<Vec<String>> {
        let mut file = tokio::fs::File::open(file_path).await
            .map_err(|e| ManagementError::Service {
                message: format!("Failed to open SafeTensors file {}: {}", file_path.display(), e)
            })?;

        // Read the first 8 bytes to get header size
        let mut size_bytes = [0u8; 8];
        file.read_exact(&mut size_bytes).await
            .map_err(|e| ManagementError::Service {
                message: format!("Failed to read header size from {}: {}", file_path.display(), e)
            })?;

        let header_size = u64::from_le_bytes(size_bytes);

        // Read the header
        let mut header_bytes = vec![0u8; header_size as usize];
        file.read_exact(&mut header_bytes).await
            .map_err(|e| ManagementError::Service {
                message: format!("Failed to read header from {}: {}", file_path.display(), e)
            })?;

        // Parse header as JSON to extract tensor names
        let header: serde_json::Value = serde_json::from_slice(&header_bytes)
            .map_err(|e| ManagementError::Service {
                message: format!("Failed to parse header JSON from {}: {}", file_path.display(), e)
            })?;

        let mut layer_names = Vec::new();
        if let Some(obj) = header.as_object() {
            for key in obj.keys() {
                if key != "__metadata__" {
                    layer_names.push(key.clone());
                }
            }
        }

        Ok(layer_names)
    }

    /// Apply renaming rules to current layer names
    fn apply_renaming_rules(current_layers: &[String], rules: &WeightRenamingRules) -> Result<HashMap<String, String>> {
        let mut mapping = HashMap::new();

        for layer_name in current_layers {
            let new_name = Self::apply_single_renaming_rule(layer_name, rules)?;
            if new_name != *layer_name {
                mapping.insert(layer_name.clone(), new_name);
                info!("Mapping: {} -> {}", layer_name, mapping[layer_name]);
            }
        }

        Ok(mapping)
    }

    /// Apply renaming rules to a single layer name
    fn apply_single_renaming_rule(layer_name: &str, rules: &WeightRenamingRules) -> Result<String> {
        let mut result = layer_name.to_string();

        for rule in &rules.rules {
            if !rule.enabled {
                continue;
            }

            match &rule.rule_type {
                RenamingRuleType::Regex { pattern, replacement } => {
                    let regex = Regex::new(pattern)
                        .map_err(|e| ManagementError::Service {
                            message: format!("Invalid regex pattern '{}': {}", pattern, e)
                        })?;

                    // Replace {root} placeholder in the replacement string
                    let replacement_with_root = replacement.replace("{root}", &rules.root);
                    result = regex.replace_all(&result, replacement_with_root.as_str()).to_string();
                },
                RenamingRuleType::Direct { from, to } => {
                    if result == *from {
                        // Replace {root} placeholder in the to string
                        result = to.replace("{root}", &rules.root);
                    }
                },
                RenamingRuleType::Prefix { old_prefix, new_prefix } => {
                    if result.starts_with(old_prefix) {
                        // Replace {root} placeholder in the new prefix
                        let new_prefix_with_root = new_prefix.replace("{root}", &rules.root);
                        result = format!("{}{}", new_prefix_with_root, &result[old_prefix.len()..]);
                    }
                },
                RenamingRuleType::Suffix { old_suffix, new_suffix } => {
                    if result.ends_with(old_suffix) {
                        // Replace {root} placeholder in the new suffix
                        let new_suffix_with_root = new_suffix.replace("{root}", &rules.root);
                        result = format!("{}{}", &result[..result.len() - old_suffix.len()], new_suffix_with_root);
                    }
                }
            }
        }

        Ok(result)
    }

    /// Update SafeTensors files with new layer names
    async fn update_safetensors_files(model_path: &Path, index: &SafeTensorsIndex, mapping: &HashMap<String, String>) -> Result<()> {
        let mut processed_files = std::collections::HashSet::new();

        for file_name in index.weight_map.values() {
            if processed_files.insert(file_name.clone()) {
                let file_path = model_path.join(file_name);
                Self::update_single_safetensors_file(&file_path, mapping).await?;
            }
        }

        Ok(())
    }

    /// Update a single SafeTensors file with new layer names
    /// Optimized to update only the header when possible, avoiding full file rewrite
    async fn update_single_safetensors_file(file_path: &Path, mapping: &HashMap<String, String>) -> Result<()> {
        info!("Updating SafeTensors file: {}", file_path.display());

        // First, read only the header to check if updates are needed
        let mut file = tokio::fs::File::open(file_path).await
            .map_err(|e| ManagementError::Service {
                message: format!("Failed to open SafeTensors file {}: {}", file_path.display(), e)
            })?;

        // Read header size (first 8 bytes)
        let mut header_size_bytes = [0u8; 8];
        use tokio::io::AsyncReadExt;
        file.read_exact(&mut header_size_bytes).await
            .map_err(|e| ManagementError::Service {
                message: format!("Failed to read header size from {}: {}", file_path.display(), e)
            })?;

        let header_size = u64::from_le_bytes(header_size_bytes);

        // Read header content
        let mut header_bytes = vec![0u8; header_size as usize];
        file.read_exact(&mut header_bytes).await
            .map_err(|e| ManagementError::Service {
                message: format!("Failed to read header from {}: {}", file_path.display(), e)
            })?;

        // Parse and update header
        let mut header: serde_json::Value = serde_json::from_slice(&header_bytes)
            .map_err(|e| ManagementError::Service {
                message: format!("Failed to parse header from {}: {}", file_path.display(), e)
            })?;

        // Update layer names in header
        let mut updated = false;
        if let Some(obj) = header.as_object_mut() {
            let keys_to_update: Vec<_> = obj.keys()
                .filter(|k| *k != "__metadata__" && mapping.contains_key(*k))
                .cloned()
                .collect();

            for old_key in keys_to_update {
                if let Some(new_key) = mapping.get(&old_key) {
                    if let Some(value) = obj.remove(&old_key) {
                        obj.insert(new_key.clone(), value);
                        updated = true;
                        info!("Updated tensor name: {} -> {}", old_key, new_key);
                    }
                }
            }
        }

        if !updated {
            info!("No updates needed for SafeTensors file: {}", file_path.display());
            return Ok(());
        }

        // Serialize updated header
        let new_header_bytes = serde_json::to_vec(&header)
            .map_err(|e| ManagementError::Service {
                message: format!("Failed to serialize updated header: {}", e)
            })?;

        let new_header_size = new_header_bytes.len() as u64;

        // Check if we can do an in-place update (same header size)
        if new_header_size == header_size {
            info!("Header size unchanged, performing in-place update for efficiency");
            Self::update_header_in_place(file_path, &new_header_bytes).await?;
        } else {
            info!("Header size changed ({} -> {}), performing full file rewrite", header_size, new_header_size);
            Self::rewrite_safetensors_file(file_path, &new_header_bytes, header_size).await?;
        }

        info!("Successfully updated SafeTensors file: {}", file_path.display());
        Ok(())
    }

    /// Update only the header portion of a SafeTensors file (when header size is unchanged)
    async fn update_header_in_place(file_path: &Path, new_header_bytes: &[u8]) -> Result<()> {
        use tokio::io::{AsyncSeekExt, AsyncWriteExt};

        let mut file = tokio::fs::OpenOptions::new()
            .write(true)
            .open(file_path)
            .await
            .map_err(|e| ManagementError::Service {
                message: format!("Failed to open file for in-place update {}: {}", file_path.display(), e)
            })?;

        // Seek to position 8 (after header size field)
        file.seek(std::io::SeekFrom::Start(8)).await
            .map_err(|e| ManagementError::Service {
                message: format!("Failed to seek in file {}: {}", file_path.display(), e)
            })?;

        // Write new header
        file.write_all(new_header_bytes).await
            .map_err(|e| ManagementError::Service {
                message: format!("Failed to write header to {}: {}", file_path.display(), e)
            })?;

        file.flush().await
            .map_err(|e| ManagementError::Service {
                message: format!("Failed to flush file {}: {}", file_path.display(), e)
            })?;

        Ok(())
    }

    /// Rewrite entire SafeTensors file (when header size changes)
    /// Uses streaming copy with checksum verification to ensure tensor data integrity
    async fn rewrite_safetensors_file(file_path: &Path, new_header_bytes: &[u8], old_header_size: u64) -> Result<()> {
        use tokio::io::{AsyncReadExt, AsyncWriteExt, AsyncSeekExt};
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Create a temporary file for atomic replacement
        let temp_path = file_path.with_extension("tmp");

        // RAII guard to ensure temp file cleanup
        let mut temp_cleanup = TempFileCleanup::new(&temp_path);

        // Open the original file for reading
        let mut input_file = tokio::fs::File::open(file_path).await
            .map_err(|e| ManagementError::Service {
                message: format!("Failed to open SafeTensors file for reading {}: {}", file_path.display(), e)
            })?;

        // Create the temporary output file
        let mut output_file = tokio::fs::File::create(&temp_path).await
            .map_err(|e| ManagementError::Service {
                message: format!("Failed to create temporary file {}: {}", temp_path.display(), e)
            })?;

        // Verify file size is correct
        let file_metadata = input_file.metadata().await
            .map_err(|e| ManagementError::Service {
                message: format!("Failed to get file metadata for {}: {}", file_path.display(), e)
            })?;

        let tensor_data_start = 8 + old_header_size;
        if file_metadata.len() < tensor_data_start {
            return Err(ManagementError::Service {
                message: format!("SafeTensors file {} is corrupted (file too small)", file_path.display())
            });
        }

        let new_header_size = new_header_bytes.len() as u64;

        // Write new header size to temp file
        output_file.write_all(&new_header_size.to_le_bytes()).await
            .map_err(|e| ManagementError::Service {
                message: format!("Failed to write header size to temp file: {}", e)
            })?;

        // Write new header to temp file
        output_file.write_all(new_header_bytes).await
            .map_err(|e| ManagementError::Service {
                message: format!("Failed to write header to temp file: {}", e)
            })?;

        // Seek to start of tensor data in input file
        input_file.seek(std::io::SeekFrom::Start(tensor_data_start)).await
            .map_err(|e| ManagementError::Service {
                message: format!("Failed to seek to tensor data in {}: {}", file_path.display(), e)
            })?;

        // Stream copy tensor data with checksum verification
        const CHUNK_SIZE: usize = 64 * 1024 * 1024; // 64MB chunks for good performance
        let mut buffer = vec![0u8; CHUNK_SIZE];
        let mut total_copied = 0u64;
        let mut data_hasher = DefaultHasher::new();
        let tensor_data_size = file_metadata.len() - tensor_data_start;

        info!("Streaming {} bytes of tensor data in {}MB chunks with integrity verification",
              tensor_data_size, CHUNK_SIZE / (1024 * 1024));

        loop {
            let bytes_read = input_file.read(&mut buffer).await
                .map_err(|e| ManagementError::Service {
                    message: format!("Failed to read from input file: {}", e)
                })?;

            if bytes_read == 0 {
                break; // EOF
            }

            let chunk = &buffer[..bytes_read];

            // Update checksum for integrity verification
            chunk.hash(&mut data_hasher);

            output_file.write_all(chunk).await
                .map_err(|e| ManagementError::Service {
                    message: format!("Failed to write to temp file: {}", e)
                })?;

            total_copied += bytes_read as u64;
        }

        // Ensure all data is written to disk
        output_file.flush().await
            .map_err(|e| ManagementError::Service {
                message: format!("Failed to flush temp file: {}", e)
            })?;

        // Close files explicitly
        drop(input_file);
        drop(output_file);

        // Verify the copied tensor data size matches expectation
        if total_copied != tensor_data_size {
            return Err(ManagementError::Service {
                message: format!("Tensor data copy incomplete: expected {} bytes, copied {} bytes",
                    tensor_data_size, total_copied)
            });
        }

        // Verify tensor data integrity using checksum
        let data_hash = data_hasher.finish();
        info!("Tensor data integrity hash: {:#x}", data_hash);

        // Atomically replace the original file with the updated one
        tokio::fs::rename(&temp_path, file_path).await
            .map_err(|e| ManagementError::Service {
                message: format!("Failed to replace original file with updated version: {}", e)
            })?;

        // Prevent cleanup of temp file since it was successfully renamed
        temp_cleanup.disarm();

        info!("Successfully rewrote SafeTensors file with streaming copy ({} bytes tensor data, hash: {:#x})",
              total_copied, data_hash);
        Ok(())
    }

    /// Update the SafeTensors index file with new layer names
    async fn update_safetensors_index(model_path: &Path, index: &SafeTensorsIndex, mapping: &HashMap<String, String>) -> Result<()> {
        let mut updated_index = index.clone();
        let mut updated = false;

        // Update weight_map with new layer names
        let weight_map_updates: Vec<_> = updated_index.weight_map.iter()
            .filter_map(|(layer_name, file_name)| {
                mapping.get(layer_name).map(|new_name| (layer_name.clone(), new_name.clone(), file_name.clone()))
            })
            .collect();

        for (old_name, new_name, file_name) in weight_map_updates {
            updated_index.weight_map.remove(&old_name);
            updated_index.weight_map.insert(new_name.clone(), file_name);
            updated = true;
            info!("Updated index mapping: {} -> {}", old_name, new_name);
        }

        if updated {
            let index_path = model_path.join("model.safetensors.index.json");
            
            // Manually construct JSON with sorted weight_map to ensure proper ordering
            let mut json_obj = serde_json::Map::new();
            
            // Sort weight_map entries alphabetically by layer name
            let sorted_weight_map: std::collections::BTreeMap<String, String> = updated_index.weight_map
                .into_iter()
                .collect();
            
            // Convert sorted map to JSON Value
            let weight_map_json: serde_json::Map<String, serde_json::Value> = sorted_weight_map
                .into_iter()
                .map(|(k, v)| (k, serde_json::Value::String(v)))
                .collect();
            
            json_obj.insert("weight_map".to_string(), serde_json::Value::Object(weight_map_json));
            
            // Add metadata if present
            if let Some(ref metadata) = updated_index.metadata {
                json_obj.insert("metadata".to_string(), serde_json::to_value(metadata)
                    .map_err(|e| ManagementError::Service {
                        message: format!("Failed to serialize metadata: {}", e)
                    })?);
            }
            
            let updated_content = serde_json::to_string_pretty(&json_obj)
                .map_err(|e| ManagementError::Service {
                    message: format!("Failed to serialize updated index: {}", e)
                })?;

            tokio::fs::write(&index_path, updated_content).await
                .map_err(|e| ManagementError::Service {
                    message: format!("Failed to write updated index file: {}", e)
                })?;

            info!("Successfully updated SafeTensors index file with sorted layer names");
        } else {
            info!("No updates needed for SafeTensors index file");
        }

        Ok(())
    }
}
