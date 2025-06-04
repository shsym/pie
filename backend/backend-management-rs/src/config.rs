use crate::error::{ConfigError, ConfigResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::{info, debug};

/// Main configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Mapping of model types to backend script names
    pub model_backends: HashMap<String, String>,
    /// Network endpoint configurations
    pub endpoints: EndpointConfig,
    /// Logging configuration
    pub logging: LoggingConfig,
    /// List of supported models with their metadata
    pub supported_models: Vec<ModelInfo>,
}

impl Config {
    /// Load configuration from a JSON file
    pub fn load<P: AsRef<Path>>(config_path: P) -> ConfigResult<Config> {
        let config_path = config_path.as_ref();
        debug!("Loading config from: {}", config_path.display());
        
        if !config_path.exists() {
            return Err(ConfigError::FileNotFound { 
                path: config_path.to_path_buf() 
            });
        }
        
        let content = std::fs::read_to_string(config_path)
            .map_err(|e| ConfigError::ParseError(format!("Failed to read config file: {}", e)))?;
        
        let config: Config = serde_json::from_str(&content)
            .map_err(|e| ConfigError::InvalidJson { source: e })?;
        
        // Validate the loaded configuration
        config.validate()?;
        
        info!("Loaded configuration with {} supported models", config.supported_models.len());
        Ok(config)
    }
    
    /// Load configuration with default path fallback
    pub fn load_default() -> ConfigResult<Config> {
        // Try standard config locations
        let config_paths = [
            PathBuf::from("config.json"),
            // Root of the cargo project
            PathBuf::from("../../config.json"),
            // Try relative to source (for development)
            PathBuf::from("../backend-management/config.json"),
        ];
        
        for path in &config_paths {
            if path.exists() {
                debug!("Found config at: {}", path.display());
                return Self::load(path);
            }
        }
        
        Err(ConfigError::FileNotFound { 
            path: PathBuf::from("config.json (searched multiple locations)") 
        })
    }
    
    /// Get backend script name for a model type
    pub fn get_backend_script(&self, model_type: &str) -> Option<&String> {
        self.model_backends.get(model_type)
    }
    
    /// Find model type by model name
    pub fn get_model_type(&self, model_name: &str) -> Option<&str> {
        self.supported_models
            .iter()
            .find(|model| model.name == model_name || model.fullname == model_name)
            .map(|model| model.model_type.as_str())
    }
    
    /// Get full model name by short name or full name
    pub fn get_full_model_name(&self, model_name: &str) -> Option<&str> {
        self.supported_models
            .iter()
            .find(|model| model.name == model_name || model.fullname == model_name)
            .map(|model| model.fullname.as_str())
    }
    
    /// Get all supported model names
    pub fn get_supported_models(&self) -> Vec<&str> {
        self.supported_models
            .iter()
            .map(|model| model.name.as_str())
            .collect()
    }

    /// Validate the configuration
    fn validate(&self) -> ConfigResult<()> {
        // Validate endpoints
        if self.endpoints.client_handshake.is_empty() {
            return Err(ConfigError::MissingField {
                field: "endpoints.client_handshake".to_string(),
            });
        }

        if self.endpoints.cli_management.is_empty() {
            return Err(ConfigError::MissingField {
                field: "endpoints.cli_management".to_string(),
            });
        }

        // Validate model backends
        if self.model_backends.is_empty() {
            return Err(ConfigError::MissingField {
                field: "model_backends".to_string(),
            });
        }

        // Validate supported models
        for (i, model) in self.supported_models.iter().enumerate() {
            if model.name.is_empty() {
                return Err(ConfigError::InvalidValue {
                    field: format!("supported_models[{}].name", i),
                    value: "empty string".to_string(),
                });
            }

            if !self.model_backends.contains_key(&model.model_type) {
                return Err(ConfigError::InvalidValue {
                    field: format!("supported_models[{}].type", i),
                    value: format!("Unknown model type: {}", model.model_type),
                });
            }
        }

        // Validate logging level
        match self.logging.level.to_uppercase().as_str() {
            "TRACE" | "DEBUG" | "INFO" | "WARN" | "ERROR" => {}
            _ => {
                return Err(ConfigError::InvalidValue {
                    field: "logging.level".to_string(),
                    value: self.logging.level.clone(),
                });
            }
        }

        Ok(())
    }

    /// Build model type mapping for quick lookups
    pub fn build_model_type_mapping(&self) -> HashMap<String, String> {
        self.supported_models
            .iter()
            .map(|model| (model.name.clone(), model.model_type.clone()))
            .collect()
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model_backends: HashMap::from([
                ("llama3".to_string(), "l4m_backend.py".to_string()),
                ("deepseek".to_string(), "deepseek_backend.py".to_string()),
            ]),
            endpoints: EndpointConfig {
                client_handshake: "ipc:///tmp/symphony-ipc".to_string(),
                cli_management: "ipc:///tmp/symphony-cli".to_string(),
            },
            logging: LoggingConfig {
                level: "INFO".to_string(),
                format: "%(asctime)s [%(levelname)8s] %(name)s: %(message)s".to_string(),
                date_format: "%Y-%m-%d %H:%M:%S".to_string(),
            },
            supported_models: vec![
                ModelInfo {
                    name: "Llama-3.1-8B-Instruct".to_string(),
                    fullname: "meta-llama/Llama-3.1-8B-Instruct".to_string(),
                    model_type: "llama3".to_string(),
                    arch_info: ModelArchInfo::default(),
                },
                ModelInfo {
                    name: "Llama-3.2-1B-Instruct".to_string(),
                    fullname: "meta-llama/Llama-3.2-1B-Instruct".to_string(),
                    model_type: "llama3".to_string(),
                    arch_info: ModelArchInfo::default(),
                },
                ModelInfo {
                    name: "DeepSeek-V3-0324".to_string(),
                    fullname: "deepseek-ai/DeepSeek-V3-0324".to_string(),
                    model_type: "deepseek".to_string(),
                    arch_info: ModelArchInfo::default(),
                },
            ],
        }
    }
}

/// Network endpoint configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointConfig {
    /// Endpoint for client handshake requests
    pub client_handshake: String,
    /// Endpoint for CLI management commands
    pub cli_management: String,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level (DEBUG, INFO, WARNING, ERROR)
    pub level: String,
    /// Log message format string
    pub format: String,
    /// Date format string
    pub date_format: String,
}

/// Model architecture and hyperparameter info required for loading
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelArchInfo {
    /// Model architectures from config.json
    #[serde(default)]
    pub architectures: Vec<String>,
    #[serde(default)] pub vocab_size: Option<u64>,
    #[serde(default)] pub hidden_size: Option<u64>,
    #[serde(default)] pub num_attention_heads: Option<u64>,
    #[serde(default)] pub num_hidden_layers: Option<u64>,
    #[serde(default)] pub intermediate_size: Option<u64>,
    #[serde(default)] pub hidden_act: Option<String>,
    #[serde(default)] pub hidden_dropout_prob: Option<f32>,
    #[serde(default)] pub attention_probs_dropout_prob: Option<f32>,
    #[serde(default)] pub max_position_embeddings: Option<u64>,
    #[serde(default)] pub type_vocab_size: Option<u64>,
    #[serde(default)] pub layer_norm_eps: Option<f32>,
    #[serde(default)] pub tie_word_embeddings: Option<bool>,
    #[serde(default)] pub bos_token_id: Option<u64>,
    #[serde(default)] pub eos_token_id: Option<Vec<u64>>,
    #[serde(default)] pub pad_token_id: Option<u64>,
    #[serde(default)] pub torch_dtype: Option<String>,
}
/// Information about a supported model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Short name for the model (e.g., "Llama-3.1-8B-Instruct")
    pub name: String,
    /// Full name/path for the model (e.g., "meta-llama/Llama-3.1-8B-Instruct")
    pub fullname: String,
    /// Model type for backend selection (e.g., "llama3", "deepseek")
    #[serde(rename = "type")]
    pub model_type: String,
    /// Nested architecture and hyperparameter info
    #[serde(default)]
    pub arch_info: ModelArchInfo,
}
