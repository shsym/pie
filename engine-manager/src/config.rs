use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use anyhow::{Result, Context};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub system: SystemConfig,
    pub services: ServicesConfig,
    pub endpoints: EndpointsConfig,
    pub logging: LoggingConfig,
    pub models: ModelsConfig,
    pub backends: BackendsConfig,
    pub paths: PathsConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    pub name: String,
    pub version: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServicesConfig {
    pub engine_manager: EngineManagerConfig,
    pub engine: EngineConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineManagerConfig {
    pub host: String,
    pub port: u16,
    pub binary_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    pub binary_name: String,
    pub default_port: u16,
    pub base_args: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointsConfig {
    pub client_handshake: String,
    pub cli_management: String,
    pub management_service: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
    pub format: String,
    pub date_format: String,
    pub directory: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsConfig {
    pub available: Vec<String>,
    pub default: String,
    pub supported_models: Vec<SupportedModel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupportedModel {
    pub name: String,
    pub fullname: String,
    #[serde(rename = "type")]
    pub model_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendsConfig {
    pub model_backends: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathsConfig {
    pub engine_binary_search: Vec<String>,
    pub engine_manager_binary_search: Vec<String>,
}

impl Config {
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let config_content = fs::read_to_string(&path)
            .with_context(|| format!("Failed to read config file: {}", path.as_ref().display()))?;

        let config: Config = serde_json::from_str(&config_content)
            .with_context(|| format!("Failed to parse config file: {}", path.as_ref().display()))?;

        Ok(config)
    }

    pub fn load_default() -> Result<Self> {
        let config_path = Path::new("config.json");
        if config_path.exists() {
            Self::load_from_file(config_path)
        } else {
            anyhow::bail!("Default config file 'config.json' not found in current directory")
        }
    }
}
