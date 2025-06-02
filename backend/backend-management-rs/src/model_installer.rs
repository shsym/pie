//! Model installation utilities using HuggingFace CLI
//!
//! This module provides functionality to download and install models and tokenizers
//! from HuggingFace Hub using the huggingface-cli download command.

use crate::error::{ManagementError, Result};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use tokio::process::Command;
use tracing::{info, error, warn};

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
        self.install_model_with_options(model_name, None, None).await
    }

    /// Install a model with specific options
    pub async fn install_model_with_options(
        &self,
        model_name: &str,
        revision: Option<&str>,
        files_to_download: Option<Vec<&str>>,
    ) -> Result<PathBuf> {
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

        // Create model info file for Symphony compatibility
        self.create_model_info_file(model_name, &model_path).await?;

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

        // Set local directory and avoid symlinks for reliability
        cmd.arg("--local-dir").arg(local_dir);
        cmd.arg("--local-dir-use-symlinks").arg("False");

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
        use serde_json::json;
        use std::time::SystemTime;

        // Try to read config.json to get model details
        let config_path = model_path.join("config.json");
        let (model_type, architectures, vocab_size) = if config_path.exists() {
            match tokio::fs::read_to_string(&config_path).await {
                Ok(config_content) => {
                    match serde_json::from_str::<serde_json::Value>(&config_content) {
                        Ok(config) => {
                            let model_type = config.get("model_type")
                                .and_then(|v| v.as_str())
                                .unwrap_or("unknown")
                                .to_string();
                            let architectures = config.get("architectures")
                                .and_then(|v| v.as_array())
                                .map(|arr| arr.iter()
                                    .filter_map(|v| v.as_str().map(String::from))
                                    .collect::<Vec<_>>())
                                .unwrap_or_default();
                            let vocab_size = config.get("vocab_size")
                                .and_then(|v| v.as_u64());
                            (model_type, architectures, vocab_size)
                        }
                        Err(e) => {
                            warn!("Failed to parse config.json: {}", e);
                            ("unknown".to_string(), Vec::new(), None)
                        }
                    }
                }
                Err(e) => {
                    warn!("Failed to read config.json: {}", e);
                    ("unknown".to_string(), Vec::new(), None)
                }
            }
        } else {
            ("unknown".to_string(), Vec::new(), None)
        };

        // Determine tokenizer class by checking for tokenizer files
        let tokenizer_class = if model_path.join("tokenizer.json").exists() {
            "PreTrainedTokenizer"
        } else if model_path.join("tokenizer.model").exists() {
            "SentencePieceTokenizer"
        } else {
            "Unknown"
        };

        let model_info = json!({
            "model_name": model_name,
            "local_name": model_name.split('/').last().unwrap_or(model_name),
            "model_type": model_type,
            "architectures": architectures,
            "vocab_size": vocab_size,
            "tokenizer_class": tokenizer_class,
            "path": model_path.display().to_string(),
            "tokenizer_path": model_path.display().to_string(),
            "installed_at": SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            "installation_method": "huggingface-cli",
            "installation_successful": true
        });

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
    pub async fn list_installed_models(&self) -> Result<Vec<ModelInfo>> {
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
    pub async fn get_model_info(&self, model_name: &str) -> Result<ModelInfo> {
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
            
            return Ok(ModelInfo {
                model_name: info.get("model_name").and_then(|v| v.as_str()).unwrap_or(model_name).to_string(),
                local_name: info.get("local_name").and_then(|v| v.as_str()).unwrap_or(model_name).to_string(),
                model_type: info.get("model_type").and_then(|v| v.as_str()).unwrap_or("unknown").to_string(),
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

        // Try to read the old text info file
        let text_info_file = model_path.join("symphony_model_info.txt");
        if text_info_file.exists() {
            let content = tokio::fs::read_to_string(&text_info_file).await
                .map_err(|e| ManagementError::Service {
                    message: format!("Failed to read model info: {}", e)
                })?;
            
            return Ok(ModelInfo::from_info_file(&content));
        }

        // Fallback: basic info from path
        Ok(ModelInfo {
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

}

/// Information about an installed model
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub model_name: String,
    pub local_name: String,
    pub model_type: String,
    pub path: PathBuf,
    pub tokenizer_path: Option<PathBuf>,
    pub architectures: Vec<String>,
    pub installed_at: Option<String>,
}

impl ModelInfo {
    fn from_info_file(content: &str) -> Self {
        let mut model_name = String::new();
        let mut local_name = String::new();
        let mut model_type = "unknown".to_string();
        let mut path = PathBuf::new();
        let mut tokenizer_path = None;
        let mut architectures = Vec::new();
        let mut installed_at = None;

        for line in content.lines() {
            if let Some((key, value)) = line.split_once(':') {
                match key.trim() {
                    "Model" => model_name = value.trim().to_string(),
                    "LocalName" => local_name = value.trim().to_string(),
                    "Type" => model_type = value.trim().to_string(),
                    "Path" => path = PathBuf::from(value.trim()),
                    "TokenizerPath" => tokenizer_path = Some(PathBuf::from(value.trim())),
                    "InstalledAt" => installed_at = Some(value.trim().to_string()),
                    "Architecture" => {
                        // Parse architecture list (could be like "['Qwen3ForCausalLM']")
                        let arch_str = value.trim();
                        if arch_str.starts_with('[') && arch_str.ends_with(']') {
                            // Simple parsing for now
                            architectures.push(arch_str.trim_matches(['[', ']', '\'', '"']).to_string());
                        } else {
                            architectures.push(arch_str.to_string());
                        }
                    },
                    _ => {}
                }
            }
        }

        // If local_name is empty, derive it from model_name
        if local_name.is_empty() && !model_name.is_empty() {
            local_name = model_name.split('/').last().unwrap_or(&model_name).to_string();
        }

        Self {
            model_name,
            local_name,
            model_type,
            path,
            tokenizer_path,
            architectures,
            installed_at,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_model_installer_creation() {
        let temp_dir = TempDir::new().unwrap();
        let installer = ModelInstaller::new(Some(temp_dir.path().to_path_buf()));
        
        assert_eq!(installer.models_dir, temp_dir.path());
        assert_eq!(installer.hf_cli_path, "huggingface-cli");
        assert!(installer.cache_dir.is_none());
    }

    #[test]
    fn test_model_installer_with_custom_hf_cli() {
        let temp_dir = TempDir::new().unwrap();
        let installer = ModelInstaller::new(Some(temp_dir.path().to_path_buf()))
            .with_hf_cli_path("custom-hf-cli".to_string())
            .with_cache_dir(temp_dir.path().join("cache"));
        
        assert_eq!(installer.hf_cli_path, "custom-hf-cli");
        assert_eq!(installer.cache_dir, Some(temp_dir.path().join("cache")));
    }

    #[test]
    fn test_model_info_parsing() {
        let content = r#"Model: test/model
Type: qwen3
Installed: 1234567890
Architecture: ['Qwen3ForCausalLM']"#;
        
        let info = ModelInfo::from_info_file(content);
        assert_eq!(info.model_name, "test/model");
        assert_eq!(info.model_type, "qwen3");
        assert_eq!(info.architectures, vec!["Qwen3ForCausalLM"]);
    }

    #[tokio::test]
    async fn test_create_model_info_file() {
        let temp_dir = TempDir::new().unwrap();
        let installer = ModelInstaller::new(Some(temp_dir.path().to_path_buf()));
        let model_path = temp_dir.path().join("test-model");
        
        tokio::fs::create_dir_all(&model_path).await.unwrap();
        
        // Create a fake config.json
        let config = serde_json::json!({
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "vocab_size": 32000
        });
        tokio::fs::write(
            model_path.join("config.json"),
            serde_json::to_string_pretty(&config).unwrap()
        ).await.unwrap();
        
        // Create tokenizer.json
        tokio::fs::write(
            model_path.join("tokenizer.json"),
            "{}"
        ).await.unwrap();
        
        installer.create_model_info_file("test/model", &model_path).await.unwrap();
        
        let info_file = model_path.join("symphony_model_info.json");
        assert!(info_file.exists());
        
        let content = tokio::fs::read_to_string(info_file).await.unwrap();
        let info: serde_json::Value = serde_json::from_str(&content).unwrap();
        
        assert_eq!(info["model_name"], "test/model");
        assert_eq!(info["model_type"], "llama");
        assert_eq!(info["tokenizer_class"], "PreTrainedTokenizer");
        assert_eq!(info["installation_method"], "huggingface-cli");
    }

    #[test]
    fn test_model_path_generation() {
        let temp_dir = TempDir::new().unwrap();
        let installer = ModelInstaller::new(Some(temp_dir.path().to_path_buf()));
        
        let path = installer.get_model_path("microsoft/DialoGPT-medium");
        assert_eq!(path, temp_dir.path().join("microsoft--DialoGPT-medium"));
    }
}
