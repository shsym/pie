use backend_management_rs::config::Config;
use backend_management_rs::types::*;
use std::collections::HashMap;
use std::path::PathBuf;
use tempfile::{tempdir, TempDir};
use tokio::process::{Child, Command};

/// Test utilities for creating mock objects and test scenarios
pub struct TestUtils;

impl TestUtils {
    /// Create a mock process for testing
    pub fn create_mock_process() -> Child {
        // Create a long-running process that we can control
        // Using 'sleep' as a harmless long-running process
        Command::new("sleep")
            .arg("3600") // Sleep for 1 hour
            .kill_on_drop(true)
            .spawn()
            .expect("Failed to start mock process")
    }

    /// Create a test config with custom values
    pub fn create_test_config() -> Config {
        Config {
            model_backends: HashMap::from([
                ("llama3".to_string(), "l4m_backend.py".to_string()),
                ("deepseek".to_string(), "deepseek_backend.py".to_string()),
            ]),
            endpoints: backend_management_rs::config::EndpointConfig {
                client_handshake: "ipc:///tmp/symphony-test-client".to_string(),
                cli_management: "ipc:///tmp/symphony-test-cli".to_string(),
            },
            logging: backend_management_rs::config::LoggingConfig {
                level: "DEBUG".to_string(),
                format: "%(asctime)s [%(levelname)8s] %(name)s: %(message)s".to_string(),
                date_format: "%Y-%m-%d %H:%M:%S".to_string(),
            },
            supported_models: vec![
                backend_management_rs::config::ModelInfo {
                    name: "Llama-3.1-8B-Instruct".to_string(),
                    fullname: "meta-llama/Llama-3.1-8B-Instruct".to_string(),
                    model_type: "llama3".to_string(),
                    arch_info: backend_management_rs::config::ModelArchInfo::default(),
                },
                backend_management_rs::config::ModelInfo {
                    name: "DeepSeek-V3-0324".to_string(),
                    fullname: "deepseek-ai/DeepSeek-V3-0324".to_string(),
                    model_type: "deepseek".to_string(),
                    arch_info: backend_management_rs::config::ModelArchInfo::default(),
                },
            ],
        }
    }

    /// Create a temporary config file
    pub fn create_temp_config_file(config: &Config) -> (TempDir, PathBuf) {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let config_path = temp_dir.path().join("config.json");
        
        let config_json = serde_json::to_string_pretty(config)
            .expect("Failed to serialize config");
        
        std::fs::write(&config_path, config_json)
            .expect("Failed to write config file");
        
        (temp_dir, config_path)
    }

    /// Load the real Python config file if it exists
    pub fn try_load_python_config() -> Option<Config> {
        let python_config_path = PathBuf::from("../backend-management/config.json");
        Config::load(&python_config_path).ok()
    }

    /// Create a test management command
    pub fn create_test_command(command: &str) -> ManagementCommand {
        ManagementCommand::new(command.to_string(), HashMap::new())
    }

    /// Create a test management command with parameters
    pub fn create_test_command_with_params(
        command: &str,
        params: HashMap<String, serde_json::Value>,
    ) -> ManagementCommand {
        ManagementCommand::new(command.to_string(), params)
    }

    /// Convert HashMap to serde_json::Value for test responses
    pub fn hashmap_to_value(map: HashMap<String, serde_json::Value>) -> Option<serde_json::Value> {
        Some(serde_json::Value::Object(
            map.into_iter()
                .map(|(k, v)| (k, v))
                .collect()
        ))
    }
}

/// Macro for creating test parameters easily
#[macro_export]
macro_rules! test_params {
    ($($key:expr => $value:expr),*) => {{
        let mut params = std::collections::HashMap::new();
        $(
            params.insert($key.to_string(), serde_json::json!($value));
        )*
        params
    }};
}
