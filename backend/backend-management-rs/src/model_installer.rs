//! Model installation utilities using HuggingFace transformers
//!
//! This module provides functionality to download and install models and tokenizers
//! from HuggingFace Hub using Python's transformers library.

use crate::error::{ManagementError, Result};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use tokio::process::Command;
use tracing::{info, error};

/// Model installation manager
#[derive(Debug, Clone)]
pub struct ModelInstaller {
    /// Base directory where models are stored
    models_dir: PathBuf,
    /// Python executable path
    python_path: String,
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
            python_path: "python3".to_string(),
        }
    }

    /// Set custom Python executable path
    pub fn with_python_path(mut self, python_path: String) -> Self {
        self.python_path = python_path;
        self
    }

    /// Install a model and tokenizer from HuggingFace Hub
    pub async fn install_model(&self, model_name: &str) -> Result<PathBuf> {
        info!("Installing model: {}", model_name);

        // Create models directory if it doesn't exist
        tokio::fs::create_dir_all(&self.models_dir).await
            .map_err(|e| ManagementError::Service {
                message: format!("Failed to create models directory: {}", e)
            })?;

        // Check if model is already installed
        let model_path = self.models_dir.join(model_name.replace("/", "--"));
        if model_path.exists() {
            info!("Model {} already exists at {:?}", model_name, model_path);
            return Ok(model_path);
        }

        // Generate the Python installation script
        let install_script = self.generate_install_script(model_name, &model_path)?;
        
        // Write script to temporary file
        let script_path = self.models_dir.join("install_model.py");
        tokio::fs::write(&script_path, install_script).await
            .map_err(|e| ManagementError::Service {
                message: format!("Failed to write install script: {}", e)
            })?;

        info!("Running model installation script for {}", model_name);
        
        // Execute the Python script with streaming output
        let mut child = Command::new(&self.python_path)
            .arg(&script_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| ManagementError::Service {
                message: format!("Failed to start Python installation process: {}", e)
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
                info!("MODEL_INSTALL: {}", line);
            }
        });
        
        // Stream stderr
        let stderr_task = tokio::spawn(async move {
            let mut lines = stderr_reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                error!("MODEL_INSTALL_ERR: {}", line);
            }
        });

        // Wait for completion
        let output = child.wait().await
            .map_err(|e| ManagementError::Service {
                message: format!("Failed to wait for installation process: {}", e)
            })?;

        // Wait for output tasks to complete
        let _ = tokio::join!(stdout_task, stderr_task);

        // Clean up script file
        let _ = tokio::fs::remove_file(&script_path).await;

        if output.success() {
            info!("Successfully installed model: {}", model_name);
            Ok(model_path)
        } else {
            error!("Model installation failed with exit code: {:?}", output.code());
            Err(ManagementError::Service {
                message: format!("Model installation failed with exit code: {:?}", output.code())
            })
        }
    }

    /// Generate Python script for model installation
    fn generate_install_script(&self, model_name: &str, model_path: &Path) -> Result<String> {
        let script = format!(r#"#!/usr/bin/env python3
"""
HuggingFace model and tokenizer installation script for Symphony.
This script downloads both the model and tokenizer from HuggingFace Hub.
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

try:
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    from huggingface_hub import snapshot_download, HfApi
    import tqdm
except ImportError as e:
    print(f"Error: Required packages not installed: {{e}}", file=sys.stderr)
    print("Please install: pip install transformers huggingface_hub tqdm", file=sys.stderr)
    sys.exit(1)

def install_model(model_name: str, target_path: str):
    """Download model and tokenizer from HuggingFace Hub"""
    try:
        print(f"🔍 Starting installation of model: {{model_name}}")
        print(f"📂 Target directory: {{target_path}}")
        
        # Create target directory
        os.makedirs(target_path, exist_ok=True)
        
        # Check if model exists on HuggingFace Hub
        print("🌐 Checking model availability on HuggingFace Hub...")
        try:
            api = HfApi()
            model_info = api.model_info(model_name)
            print(f"✅ Model found: {{model_info.modelId}}")
            print(f"📊 Downloads: {{model_info.downloads or 'Unknown'}}")
            if hasattr(model_info, 'library_name') and model_info.library_name:
                print(f"📚 Library: {{model_info.library_name}}")
        except Exception as e:
            print(f"⚠️  Warning: Could not fetch model info: {{e}}")
        
        # Download the complete model repository
        print("📥 Starting model download...")
        print("This may take several minutes depending on model size and network speed...")
        
        try:
            snapshot_download(
                repo_id=model_name,
                local_dir=target_path,
                local_dir_use_symlinks=False,
                resume_download=True,
                tqdm_class=tqdm.tqdm
            )
            print("✅ Model download completed!")
        except Exception as e:
            print(f"❌ Download failed: {{e}}", file=sys.stderr)
            return False
        
        # Verify the installation by loading config
        print("🔍 Verifying installation...")
        try:
            config = AutoConfig.from_pretrained(target_path)
            model_type = config.model_type if hasattr(config, 'model_type') else 'unknown'
            architectures = config.architectures if hasattr(config, 'architectures') else []
            vocab_size = config.vocab_size if hasattr(config, 'vocab_size') else None
            
            print(f"✅ Model configuration loaded successfully")
            print(f"🏗️  Model type: {{model_type}}")
            print(f"🏛️  Architecture: {{architectures}}")
            if vocab_size:
                print(f"📖 Vocabulary size: {{vocab_size:,}}")
        except Exception as e:
            print(f"❌ Could not load model config: {{e}}", file=sys.stderr)
            return False
        
        # Test tokenizer loading
        print("🔍 Testing tokenizer...")
        tokenizer_path = target_path
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            tokenizer_class = type(tokenizer).__name__
            print(f"✅ Tokenizer loaded: {{tokenizer_class}}")
            print(f"📖 Tokenizer vocab size: {{tokenizer.vocab_size:,}}")
            
            # Test encoding/decoding
            test_text = "Hello, world!"
            tokens = tokenizer.encode(test_text)
            decoded = tokenizer.decode(tokens)
            print(f"🧪 Tokenizer test: '{{test_text}}' -> {{len(tokens)}} tokens -> '{{decoded}}'")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not load tokenizer: {{e}}", file=sys.stderr)
            tokenizer_class = "Unknown"
        
        # Write comprehensive model info
        print("📝 Writing model metadata...")
        model_info = {{
            "model_name": model_name,
            "local_name": model_name.split('/')[-1] if '/' in model_name else model_name,
            "model_type": model_type,
            "architectures": architectures,
            "vocab_size": vocab_size,
            "tokenizer_class": tokenizer_class,
            "path": target_path,
            "tokenizer_path": tokenizer_path,
            "installed_at": datetime.now().isoformat(),
            "installation_successful": True
        }}
        
        info_file = Path(target_path) / "symphony_model_info.json"
        with open(info_file, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"✅ Successfully installed {{model_name}} to {{target_path}}")
        print(f"📄 Model info saved to {{info_file}}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error installing model {{model_name}}: {{e}}", file=sys.stderr)
        return False

if __name__ == "__main__":
    model_name = "{model_name}"
    target_path = "{model_path}"
    
    success = install_model(model_name, target_path)
    sys.exit(0 if success else 1)
"#, 
            model_name = model_name,
            model_path = model_path.display()
        );
        
        Ok(script)
    }

    /// Remove a model and its associated files
    pub async fn uninstall_model(&self, model_name: &str) -> Result<PathBuf> {
        let model_dir = self.get_model_dir(model_name);
        
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
                let dir_name = entry.file_name().to_string_lossy().to_string();
                // Convert directory name back to original model name (reverse the "/" -> "--" replacement)
                let model_name = dir_name.replace("--", "/");
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
        self.get_model_dir(model_name)
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
                installed_at: info.get("installed_at").and_then(|v| v.as_str()).map(String::from),
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

    /// Get the directory path for a specific model
    fn get_model_dir(&self, model_name: &str) -> PathBuf {
        self.models_dir.join(model_name.replace("/", "--"))
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
        assert_eq!(installer.python_path, "python3");
    }

    #[test]
    fn test_model_info_parsing() {
        let content = r#"Model: test/model
Type: qwen3
Installed: 1234567890
Architecture: ['Qwen3ForCausalLM']"#;
        
        let info = ModelInfo::from_info_file(content);
        assert_eq!(info.name, "test/model");
        assert_eq!(info.model_type, "qwen3");
        assert_eq!(info.architectures, vec!["Qwen3ForCausalLM"]);
    }
}
