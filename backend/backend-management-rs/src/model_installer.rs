//! Model installation utilities using HuggingFace CLI
//!
//! This module provides functionality to download and install models and tokenizers
//! from HuggingFace Hub using the huggingface-cli download command.
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::collections::HashMap;
use tokio::process::Command;
use tracing::{info, warn, error};
use serde::{Serialize, Deserialize};
use regex::Regex;
use crate::error::{ManagementError, Result};
use crate::config::{ModelInfo, ModelArchInfo};

/// Information about an installed model with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstalledModelInfo {
    /// Original model name from HuggingFace (e.g., "meta-llama/Llama-3.1-8B-Instruct")
    pub model_name: String,
    /// Local name for the model (e.g., "Llama-3.1-8B-Instruct")
    pub local_name: String,
    /// Model type for backend selection (e.g., "llama3", "deepseek")
    pub model_type: String,
    /// Local path where model is installed
    pub path: PathBuf,
    /// Path to tokenizer files (usually same as model path)
    pub tokenizer_path: Option<PathBuf>,
    /// Model architectures from config.json
    pub architectures: Vec<String>,
    /// When the model was installed (ISO timestamp or UNIX timestamp)
    pub installed_at: Option<String>,
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

impl InstalledModelInfo {
    /// Parse model info from an old-style text info file
    pub fn from_info_file(content: &str) -> Self {
        let mut model_name = String::new();
        let mut local_name = String::new();
        let mut model_type = String::new();
        let mut path = PathBuf::new();

        for line in content.lines() {
            let line = line.trim();
            if let Some((key, value)) = line.split_once(':') {
                let key = key.trim();
                let value = value.trim();
                match key {
                    "Model Name" => model_name = value.to_string(),
                    "Local Name" => local_name = value.to_string(),
                    "Model Type" => model_type = value.to_string(),
                    "Path" => path = PathBuf::from(value),
                    _ => {}
                }
            }
        }

        InstalledModelInfo {
            model_name,
            local_name,
            model_type,
            path: path.clone(),
            tokenizer_path: Some(path),
            architectures: vec![],
            installed_at: None,
        }
    }
}

/// Model installation manager
#[derive(Debug, Clone)]
pub struct ModelInstaller {
    /// Base directory where models are stored
    models_dir: PathBuf,
    /// HuggingFace CLI executable path
    hf_cli_path: String,
    /// Cache directory for HuggingFace CLI (optional)
    cache_dir: Option<PathBuf>,
}

impl ModelInstaller {
    /// Create a new model installer
    pub fn new(models_dir: Option<PathBuf>) -> Self {
        let models_dir = models_dir.unwrap_or_else(|| {
            // Default to ~/.cache/symphony/models
            let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
            PathBuf::from(home).join(".cache").join("symphony").join("models")
        });

        Self {
            models_dir,
            hf_cli_path: "huggingface-cli".to_string(),
            cache_dir: None,
        }
    }

    /// Set custom HuggingFace CLI executable path
    pub fn with_hf_cli_path(mut self, hf_cli_path: String) -> Self {
        self.hf_cli_path = hf_cli_path;
        self
    }

    /// Set custom cache directory for HuggingFace CLI
    pub fn with_cache_dir(mut self, cache_dir: PathBuf) -> Self {
        self.cache_dir = Some(cache_dir);
        self
    }

    /// Install a model and tokenizer from HuggingFace Hub using huggingface-cli
    pub async fn install_model(&self, model_name: &str) -> Result<PathBuf> {
        // Validate model name format first
        if let Err(e) = Self::validate_model_name_format(model_name) {
            // Try to provide helpful suggestions
            let suggestions = Self::suggest_model_name(model_name);
            if !suggestions.is_empty() {
                return Err(ManagementError::InvalidInput {
                    message: format!(
                        "{}. Did you mean one of these?\n{}",
                        e,
                        suggestions.iter()
                            .take(3)
                            .map(|s| format!("  - {}", s))
                            .collect::<Vec<_>>()
                            .join("\n")
                    )
                });
            }
            return Err(e);
        }

        self.install_model_with_options(model_name, None, None).await
    }

    /// Install a model with specific options
    pub async fn install_model_with_options(
        &self,
        model_name: &str,
        revision: Option<&str>,
        files_to_download: Option<Vec<&str>>,
    ) -> Result<PathBuf> {
        // Validate model name format first
        Self::validate_model_name_format(model_name)?;
        
        info!("Installing model: {}", model_name);

        // Create models directory if it doesn't exist
        tokio::fs::create_dir_all(&self.models_dir).await
            .map_err(|e| ManagementError::Service {
                message: format!("Failed to create models directory: {}", e)
            })?;

        // Check if model is already installed
        let model_path = self.models_dir.join(model_name.replace("/", "--"));
        if model_path.exists() {
            // Validate that essential files are present
            if self.validate_model_installation(&model_path).await? {
                info!("Model {} already exists and is valid at {:?}", model_name, model_path);
                return Ok(model_path);
            } else {
                warn!("Model {} exists but is incomplete, re-downloading", model_name);
                // Remove incomplete installation and proceed with download
                if let Err(e) = tokio::fs::remove_dir_all(&model_path).await {
                    warn!("Failed to remove incomplete model directory: {}", e);
                }
            }
        }

        info!("Downloading model {} using huggingface-cli", model_name);
        
        // Execute huggingface-cli download
        self.download_model_with_hf_cli(model_name, revision, files_to_download, &model_path).await?;

        // Create model info file for Symphony compatibility first
        self.create_model_info_file(model_name, &model_path).await?;

        // Convert HuggingFace tokenizer to Symphony format if present
        // Pass the info file path so it can update it with tokenizer metadata
        let info_file_path = model_path.join("symphony_model_info.json");
        if let Err(e) = crate::transform_tokenizer::convert_hf_tokenizer_to_symphony(&model_path, &info_file_path).await {
            warn!("Failed to convert tokenizer for model {}: {}", model_name, e);
            // Continue with installation even if tokenizer conversion fails
        }

        info!("Successfully installed model: {}", model_name);
        Ok(model_path)
    }

    /// Download model using huggingface-cli command
    async fn download_model_with_hf_cli(
        &self,
        repo_id: &str,
        revision: Option<&str>,
        files_to_download: Option<Vec<&str>>,
        local_dir: &Path,
    ) -> Result<()> {
        // Ensure the local directory exists (like mkdir -p)
        tokio::fs::create_dir_all(local_dir).await
            .map_err(|e| ManagementError::Service {
                message: format!("Failed to create local directory: {}", e)
            })?;

        let mut cmd = Command::new(&self.hf_cli_path);
        cmd.arg("download");
        cmd.arg(repo_id);

        // Add revision if specified
        if let Some(rev) = revision {
            cmd.arg("--revision").arg(rev);
        }

        // Add specific files if provided
        if let Some(files) = files_to_download {
            for file in files {
                cmd.arg(file);
            }
        }
        // If no specific files provided, download everything (simplified approach)

        // Set local directory
        cmd.arg("--local-dir").arg(local_dir);

        // Set custom cache directory if specified
        if let Some(cache) = &self.cache_dir {
            cmd.arg("--cache-dir").arg(cache);
        }

        // Set up stdio for streaming output
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        info!("Executing: {} {:?}", self.hf_cli_path, cmd.as_std().get_args().collect::<Vec<_>>());

        let mut child = cmd.spawn()
            .map_err(|e| ManagementError::Service {
                message: format!("Failed to start huggingface-cli: {}. Make sure huggingface-cli is installed and in PATH", e)
            })?;

        // Stream output in real-time
        use tokio::io::{AsyncBufReadExt, BufReader};
        
        let stdout = child.stdout.take().unwrap();
        let stderr = child.stderr.take().unwrap();
        let stdout_reader = BufReader::new(stdout);
        let stderr_reader = BufReader::new(stderr);
        
        // Stream stdout
        let stdout_task = tokio::spawn(async move {
            let mut lines = stdout_reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                info!("HF_CLI: {}", line);
            }
        });
        
        // Stream stderr
        let stderr_task = tokio::spawn(async move {
            let mut lines = stderr_reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                if line.contains("warning") || line.contains("Warning") || line.contains("FutureWarning") {
                    warn!("HF_CLI_WARN: {}", line);
                } else if line.contains("Downloading") || line.contains("Download complete") || 
                         line.contains("Fetching") || line.contains("Moving file") ||
                         line.contains("Xet Storage") || line.starts_with("Fetching ") {
                    // These are normal progress messages that huggingface-cli sends to stderr
                    info!("HF_CLI: {}", line);
                } else if line.trim().is_empty() {
                    // Skip empty lines
                    continue;
                } else {
                    // Only log actual errors as errors
                    error!("HF_CLI_ERR: {}", line);
                }
            }
        });

        // Wait for completion
        let output = child.wait().await
            .map_err(|e| ManagementError::Service {
                message: format!("Failed to wait for huggingface-cli process: {}", e)
            })?;

        // Wait for output tasks to complete
        let _ = tokio::join!(stdout_task, stderr_task);

        if output.success() {
            info!("Model download completed successfully");
            Ok(())
        } else {
            error!("huggingface-cli failed with exit code: {:?}", output.code());
            Err(ManagementError::Service {
                message: format!("huggingface-cli failed with exit code: {:?}", output.code())
            })
        }
    }

    /// Validate that a model installation is complete and has essential files
    async fn validate_model_installation(&self, model_path: &Path) -> Result<bool> {
        // Check if the directory exists
        if !model_path.exists() {
            return Ok(false);
        }

        // Essential files that should be present
        let essential_files = vec![
            "config.json",
            "symphony_model_info.json"
        ];

        // Check for essential files
        for file in &essential_files {
            let file_path = model_path.join(file);
            if !file_path.exists() {
                warn!("Essential file missing: {:?}", file_path);
                return Ok(false);
            }
        }

        // Check for at least one tokenizer file
        let tokenizer_files = vec![
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json"
        ];
        
        let has_tokenizer = tokenizer_files.iter().any(|file| {
            model_path.join(file).exists()
        });

        if !has_tokenizer {
            warn!("No tokenizer files found in model directory: {:?}", model_path);
            return Ok(false);
        }

        // Check for at least one model weight file
        let model_files = vec![
            "model.safetensors",
            "pytorch_model.bin",
            "model.bin"
        ];

        let has_weights = model_files.iter().any(|file| {
            model_path.join(file).exists()
        }) || {
            // Check for sharded models (model-00001-of-*.safetensors or pytorch_model-00001-of-*.bin)
            match std::fs::read_dir(model_path) {
                Ok(entries) => {
                    entries.into_iter().any(|entry| {
                        if let Ok(entry) = entry {
                            let name = entry.file_name().to_string_lossy().to_lowercase();
                            name.contains("model-") && (name.ends_with(".safetensors") || name.ends_with(".bin"))
                        } else {
                            false
                        }
                    })
                }
                Err(_) => false
            }
        };

        if !has_weights {
            warn!("No model weight files found in model directory: {:?}", model_path);
            return Ok(false);
        }

        info!("Model validation successful for: {:?}", model_path);
        Ok(true)
    }

    /// Create model info file for Symphony compatibility
    async fn create_model_info_file(&self, model_name: &str, model_path: &Path) -> Result<()> {
        // Try to read config.json to get model details
        let config_path = model_path.join("config.json");
        // Read and encapsulate model architecture info
        let (model_type, arch_info) = if config_path.exists() {
            match tokio::fs::read_to_string(&config_path).await {
                Ok(content) => match serde_json::from_str::<serde_json::Value>(&content) {
                    Ok(cfg) => {
                        let model_type = cfg.get("model_type").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();
                        let arch_info = ModelArchInfo {
                            architectures: cfg.get("architectures").and_then(|v| v.as_array())
                                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect()).unwrap_or_default(),
                            vocab_size: cfg.get("vocab_size").and_then(|v| v.as_u64()),
                            hidden_size: cfg.get("hidden_size").and_then(|v| v.as_u64()),
                            num_attention_heads: cfg.get("num_attention_heads").and_then(|v| v.as_u64()),
                            num_hidden_layers: cfg.get("num_hidden_layers").and_then(|v| v.as_u64()),
                            intermediate_size: cfg.get("intermediate_size").and_then(|v| v.as_u64()),
                            hidden_act: cfg.get("hidden_act").and_then(|v| v.as_str().map(String::from)),
                            hidden_dropout_prob: cfg.get("hidden_dropout_prob").and_then(|v| v.as_f64().map(|f| f as f32)),
                            attention_probs_dropout_prob: cfg.get("attention_probs_dropout_prob").and_then(|v| v.as_f64().map(|f| f as f32)),
                            max_position_embeddings: cfg.get("max_position_embeddings").and_then(|v| v.as_u64()),
                            type_vocab_size: cfg.get("type_vocab_size").and_then(|v| v.as_u64()),
                            layer_norm_eps: cfg.get("layer_norm_eps").and_then(|v| v.as_f64().map(|f| f as f32)),
                            tie_word_embeddings: cfg.get("tie_word_embeddings").and_then(|v| v.as_bool()),
                            bos_token_id: cfg.get("bos_token_id").and_then(|v| v.as_u64()),
                            eos_token_id: cfg.get("eos_token_id").and_then(|v| if let Some(arr) = v.as_array() {
                                    Some(arr.iter().filter_map(|e| e.as_u64()).collect())
                                } else if let Some(id) = v.as_u64() {
                                    Some(vec![id])
                                } else { None }),
                            pad_token_id: cfg.get("pad_token_id").and_then(|v| v.as_u64()),
                            torch_dtype: cfg.get("torch_dtype").and_then(|v| v.as_str().map(String::from)),
                        };
                        (model_type, arch_info)
                    }
                    Err(e) => {
                        warn!("Failed to parse config.json: {}", e);
                        ("unknown".to_string(), ModelArchInfo::default())
                    }
                },
                Err(e) => {
                    warn!("Failed to read config.json: {}", e);
                    ("unknown".to_string(), ModelArchInfo::default())
                }
            }
        } else {
            ("unknown".to_string(), ModelArchInfo::default())
        };

        let model_info = ModelInfo {
             name: model_name.split('/').last().unwrap_or(model_name).to_string(),
             fullname: model_name.to_string(),
             model_type,
             arch_info,
         };

        let info_file = model_path.join("symphony_model_info.json");
        let info_content = serde_json::to_string_pretty(&model_info)
            .map_err(|e| ManagementError::Service {
                message: format!("Failed to serialize model info: {}", e)
            })?;

        tokio::fs::write(&info_file, info_content).await
            .map_err(|e| ManagementError::Service {
                message: format!("Failed to write model info file: {}", e)
            })?;

        info!("Created model info file: {:?}", info_file);
        Ok(())
    }

    /// Remove a model and its associated files
    pub async fn uninstall_model(&self, model_name: &str) -> Result<PathBuf> {
        let model_dir = self.get_model_path(model_name);
        
        if !model_dir.exists() {
            return Err(crate::ManagementError::Model {
                message: format!("Model '{}' is not installed", model_name),
            });
        }

        info!("Uninstalling model '{}' from {:?}", model_name, model_dir);
        
        // Remove the entire model directory
        tokio::fs::remove_dir_all(&model_dir).await
            .map_err(|e| crate::ManagementError::Service {
                message: format!("Failed to remove model directory: {}", e),
            })?;

        info!("Successfully uninstalled model '{}'", model_name);
        Ok(model_dir)
    }

    /// List all installed models
    pub async fn list_installed_models(&self) -> Result<Vec<InstalledModelInfo>> {
        let mut models = Vec::new();
        
        if !self.models_dir.exists() {
            return Ok(models);
        }

        let mut entries = tokio::fs::read_dir(&self.models_dir).await
            .map_err(|e| crate::ManagementError::Service {
                message: format!("Failed to read models directory: {}", e),
            })?;

        while let Some(entry) = entries.next_entry().await
            .map_err(|e| crate::ManagementError::Service {
                message: format!("Failed to read directory entry: {}", e),
            })? {
            
            if entry.file_type().await.map(|ft| ft.is_dir()).unwrap_or(false) {
                let model_name = entry.file_name().to_string_lossy().to_string();
                
                // Skip directories that start with '.' (like .locks, .git, etc.)
                if model_name.starts_with('.') {
                    continue;
                }
                
                if let Ok(model_info) = self.get_model_info(&model_name).await {
                    models.push(model_info);
                }
            }
        }

        Ok(models)
    }

    /// Remove an installed model
    pub async fn remove_model(&self, model_name: &str) -> Result<()> {
        let model_path = self.models_dir.join(model_name.replace("/", "--"));
        
        if !model_path.exists() {
            return Err(ManagementError::Service {
                message: format!("Model {} is not installed", model_name)
            });
        }

        info!("Removing model: {} from {:?}", model_name, model_path);
        tokio::fs::remove_dir_all(&model_path).await
            .map_err(|e| ManagementError::Service {
                message: format!("Failed to remove model directory: {}", e)
            })?;

        info!("Successfully removed model: {}", model_name);
        Ok(())
    }

    /// Get the path where a model would be installed
    pub fn get_model_path(&self, model_name: &str) -> PathBuf {
        self.models_dir.join(model_name.replace("/", "--"))
    }

    /// Check if a model is installed
    pub async fn is_model_installed(&self, model_name: &str) -> bool {
        self.get_model_path(model_name).exists()
    }

    /// Get model info from installed model
    pub async fn get_model_info(&self, model_name: &str) -> Result<InstalledModelInfo> {
        let model_path = self.get_model_path(model_name);
        
        if !model_path.exists() {
            return Err(ManagementError::Service {
                message: format!("Model {} is not installed", model_name)
            });
        }

        // Try to read the JSON info file first
        let json_info_file = model_path.join("symphony_model_info.json");
        if json_info_file.exists() {
            let content = tokio::fs::read_to_string(&json_info_file).await
                .map_err(|e| ManagementError::Service {
                    message: format!("Failed to read model info: {}", e)
                })?;
            
            let info: serde_json::Value = serde_json::from_str(&content)
                .map_err(|e| ManagementError::Service {
                    message: format!("Failed to parse model info JSON: {}", e)
                })?;
            
            return Ok(InstalledModelInfo {
                model_name: info.get("fullname").and_then(|v| v.as_str()).unwrap_or(model_name).to_string(),
                local_name: info.get("name").and_then(|v| v.as_str()).unwrap_or(model_name).to_string(),
                model_type: info.get("type").and_then(|v| v.as_str()).unwrap_or("unknown").to_string(),
                path: model_path.clone(),
                tokenizer_path: info.get("tokenizer_path").and_then(|v| v.as_str()).map(PathBuf::from),
                architectures: info.get("architectures").and_then(|v| v.as_array()).map(|arr| 
                    arr.iter().filter_map(|v| v.as_str().map(String::from)).collect()
                ).unwrap_or_default(),
                installed_at: info.get("installed_at").and_then(|v| {
                    // Handle both string (ISO datetime) and number (timestamp) formats
                    if let Some(s) = v.as_str() {
                        Some(s.to_string())
                    } else if let Some(ts) = v.as_u64() {
                        // Convert timestamp to ISO string
                        use std::time::SystemTime;
                        SystemTime::UNIX_EPOCH
                            .checked_add(std::time::Duration::from_secs(ts))
                            .and_then(|time| {
                                let datetime: chrono::DateTime<chrono::Utc> = time.into();
                                Some(datetime.to_rfc3339())
                            })
                    } else {
                        None
                    }
                }),
            });
        }

        // Try to read the old text info file and parse manually
        let text_info_file = model_path.join("symphony_model_info.txt");
        if text_info_file.exists() {
            let content = tokio::fs::read_to_string(&text_info_file).await
                .map_err(|e| ManagementError::Service {
                    message: format!("Failed to read model info: {}", e)
                })?;
            
            return Ok(InstalledModelInfo::from_info_file(&content));
        }

        // Fallback: basic info from path
        Ok(InstalledModelInfo {
            model_name: model_name.to_string(),
            local_name: model_name.split('/').last().unwrap_or(model_name).to_string(),
            model_type: "unknown".to_string(),
            path: model_path.clone(),
            tokenizer_path: Some(model_path),
            architectures: Vec::new(),
            installed_at: None,
        })
    }

    /// Resolve a local model name to the original HuggingFace model name
    /// This allows users to uninstall models using either the original name or local name
    pub async fn resolve_model_name(&self, name_or_local_name: &str) -> Result<String> {
        // First, check if it's already an original model name (contains "/")
        if name_or_local_name.contains("/") && self.is_model_installed(name_or_local_name).await {
            return Ok(name_or_local_name.to_string());
        }
        
        // If not, search through all installed models to find a match by local_name
        let installed_models = self.list_installed_models().await?;
        
        for model_info in installed_models {
            // Check if the input matches the local name
            if model_info.local_name == name_or_local_name {
                return Ok(model_info.model_name);
            }
            // Also check if it matches the last part of the original model name
            // (for cases where local_name wasn't explicitly set)
            if let Some(last_part) = model_info.model_name.split('/').last() {
                if last_part == name_or_local_name {
                    return Ok(model_info.model_name);
                }
            }
        }
        
        // If no match found, return the original name (let the caller handle the error)
        Ok(name_or_local_name.to_string())
    }

    /// Validate that the model name follows HuggingFace repository format (organization/model-name)
    fn validate_model_name_format(model_name: &str) -> Result<()> {
        // Check if the model name contains a slash (indicating organization/model format)
        if !model_name.contains('/') {
            return Err(ManagementError::InvalidInput {
                message: format!(
                    "Invalid model name format: '{}'. \
                    Model names must include the organization/username (e.g., 'meta-llama/Llama-3.1-8B-Instruct'). \
                    Common formats:\n\
                    - meta-llama/Llama-3.1-8B-Instruct\n\
                    - microsoft/DialoGPT-medium\n\
                    - huggingface/CodeBERTa-small-v1\n\
                    Please visit https://huggingface.co/models to find the correct repository path.",
                    model_name
                )
            });
        }

        // Additional validation: check for reasonable format
        let parts: Vec<&str> = model_name.split('/').collect();
        if parts.len() != 2 {
            return Err(ManagementError::InvalidInput {
                message: format!(
                    "Invalid model name format: '{}'. \
                    Expected format: 'organization/model-name' (exactly one slash). \
                    Examples: 'meta-llama/Llama-3.1-8B-Instruct', 'microsoft/DialoGPT-medium'",
                    model_name
                )
            });
        }

        let (org, model) = (parts[0], parts[1]);
        
        // Check for empty parts
        if org.trim().is_empty() || model.trim().is_empty() {
            return Err(ManagementError::InvalidInput {
                message: format!(
                    "Invalid model name format: '{}'. \
                    Both organization and model name must be non-empty. \
                    Expected format: 'organization/model-name'",
                    model_name
                )
            });
        }

        // Check for invalid characters that would cause issues with file paths
        let invalid_chars = ['\\', ':', '*', '?', '"', '<', '>', '|'];
        for &ch in &invalid_chars {
            if model_name.contains(ch) {
                return Err(ManagementError::InvalidInput {
                    message: format!(
                        "Invalid model name format: '{}'. \
                        Model names cannot contain invalid characters: {}",
                        model_name,
                        invalid_chars.iter().collect::<String>()
                    )
                });
            }
        }

        Ok(())
    }

    /// Suggest common model names if user provides incomplete format
    fn suggest_model_name(partial_name: &str) -> Vec<String> {
        let suggestions = vec![
            ("llama", vec![
                "meta-llama/Llama-3.1-8B-Instruct",
                "meta-llama/Llama-3.1-7B-Instruct", 
                "meta-llama/Llama-2-7b-chat-hf",
                "meta-llama/Llama-2-13b-chat-hf"
            ])
        ];

        let partial_lower = partial_name.to_lowercase();
        for (keyword, models) in suggestions {
            if partial_lower.contains(keyword) {
                return models.into_iter().map(String::from).collect();
            }
        }

        vec![]
    }

    /// Rename model layers according to weight_renaming.json rules
    pub async fn rename_model_layers(&self, model_path: &Path) -> Result<()> {
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
        let renaming_rules = self.load_renaming_rules(&renaming_rules_path).await?;
        info!("Loaded {} renaming rules", renaming_rules.rules.len());

        // Load the SafeTensors index
        let safetensors_index = self.load_safetensors_index(&index_path).await?;
        info!("Found {} SafeTensors files to process", safetensors_index.weight_map.len());

        // Extract current layer names from all SafeTensors files
        let current_layers = self.extract_all_layer_names(model_path, &safetensors_index).await?;
        info!("Extracted {} layer names", current_layers.len());

        // Apply renaming rules to generate new layer names
        let layer_mapping = self.apply_renaming_rules(&current_layers, &renaming_rules)?;
        info!("Generated {} layer name mappings", layer_mapping.len());

        // Update SafeTensors files with new layer names
        self.update_safetensors_files(model_path, &safetensors_index, &layer_mapping).await?;

        // Update the index file with new layer names
        self.update_safetensors_index(model_path, &safetensors_index, &layer_mapping).await?;

        info!("Successfully completed layer renaming for model at: {}", model_path.display());
        Ok(())
    }

    /// Load renaming rules from weight_renaming.json
    async fn load_renaming_rules(&self, rules_path: &Path) -> Result<WeightRenamingRules> {
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
    async fn load_safetensors_index(&self, index_path: &Path) -> Result<SafeTensorsIndex> {
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
    async fn extract_all_layer_names(&self, model_path: &Path, index: &SafeTensorsIndex) -> Result<Vec<String>> {
        let mut all_layers = Vec::new();
        let mut processed_files = std::collections::HashSet::new();

        // Get unique SafeTensors file names
        for file_name in index.weight_map.values() {
            if processed_files.insert(file_name.clone()) {
                let file_path = model_path.join(file_name);
                let layers = self.extract_layer_names_from_file(&file_path).await?;
                all_layers.extend(layers);
            }
        }

        Ok(all_layers)
    }

    /// Extract layer names from a single SafeTensors file by reading only the header
    async fn extract_layer_names_from_file(&self, file_path: &Path) -> Result<Vec<String>> {
        let mut file = tokio::fs::File::open(file_path).await
            .map_err(|e| ManagementError::Service {
                message: format!("Failed to open SafeTensors file {}: {}", file_path.display(), e)
            })?;

        // Read the first 8 bytes to get header size
        let mut size_bytes = [0u8; 8];
        use tokio::io::AsyncReadExt;
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
    fn apply_renaming_rules(&self, current_layers: &[String], rules: &WeightRenamingRules) -> Result<HashMap<String, String>> {
        let mut mapping = HashMap::new();

        for layer_name in current_layers {
            let new_name = self.apply_single_renaming_rule(layer_name, rules)?;
            if new_name != *layer_name {
                mapping.insert(layer_name.clone(), new_name);
                info!("Mapping: {} -> {}", layer_name, mapping[layer_name]);
            }
        }

        Ok(mapping)
    }

    /// Apply renaming rules to a single layer name
    fn apply_single_renaming_rule(&self, layer_name: &str, rules: &WeightRenamingRules) -> Result<String> {
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
    async fn update_safetensors_files(&self, model_path: &Path, index: &SafeTensorsIndex, mapping: &HashMap<String, String>) -> Result<()> {
        let mut processed_files = std::collections::HashSet::new();

        for file_name in index.weight_map.values() {
            if processed_files.insert(file_name.clone()) {
                let file_path = model_path.join(file_name);
                self.update_single_safetensors_file(&file_path, mapping).await?;
            }
        }

        Ok(())
    }

    /// Update a single SafeTensors file with new layer names
    async fn update_single_safetensors_file(&self, file_path: &Path, mapping: &HashMap<String, String>) -> Result<()> {
        info!("Updating SafeTensors file: {}", file_path.display());

        // Read the file
        let file_content = tokio::fs::read(file_path).await
            .map_err(|e| ManagementError::Service {
                message: format!("Failed to read SafeTensors file {}: {}", file_path.display(), e)
            })?;

        // Parse header size
        if file_content.len() < 8 {
            return Err(ManagementError::Service {
                message: format!("SafeTensors file {} is too small", file_path.display())
            });
        }

        let header_size = u64::from_le_bytes([
            file_content[0], file_content[1], file_content[2], file_content[3],
            file_content[4], file_content[5], file_content[6], file_content[7]
        ]);

        if file_content.len() < (8 + header_size as usize) {
            return Err(ManagementError::Service {
                message: format!("SafeTensors file {} has invalid header size", file_path.display())
            });
        }

        // Extract and parse header
        let header_bytes = &file_content[8..8 + header_size as usize];
        let mut header: serde_json::Value = serde_json::from_slice(header_bytes)
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

        if updated {
            // Serialize updated header
            let new_header_bytes = serde_json::to_vec(&header)
                .map_err(|e| ManagementError::Service {
                    message: format!("Failed to serialize updated header: {}", e)
                })?;

            // Create new file content
            let new_header_size = new_header_bytes.len() as u64;
            let mut new_content = Vec::new();
            
            // Write new header size
            new_content.extend_from_slice(&new_header_size.to_le_bytes());
            
            // Write new header
            new_content.extend_from_slice(&new_header_bytes);
            
            // Write original tensor data
            new_content.extend_from_slice(&file_content[8 + header_size as usize..]);

            // Write updated file
            tokio::fs::write(file_path, new_content).await
                .map_err(|e| ManagementError::Service {
                    message: format!("Failed to write updated SafeTensors file {}: {}", file_path.display(), e)
                })?;

            info!("Successfully updated SafeTensors file: {}", file_path.display());
        } else {
            info!("No updates needed for SafeTensors file: {}", file_path.display());
        }

        Ok(())
    }

    /// Update the SafeTensors index file with new layer names
    async fn update_safetensors_index(&self, model_path: &Path, index: &SafeTensorsIndex, mapping: &HashMap<String, String>) -> Result<()> {
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
            let updated_content = serde_json::to_string_pretty(&updated_index)
                .map_err(|e| ManagementError::Service {
                    message: format!("Failed to serialize updated index: {}", e)
                })?;

            tokio::fs::write(&index_path, updated_content).await
                .map_err(|e| ManagementError::Service {
                    message: format!("Failed to write updated index file: {}", e)
                })?;

            info!("Successfully updated SafeTensors index file");
        } else {
            info!("No updates needed for SafeTensors index file");
        }

        Ok(())
    }
}
