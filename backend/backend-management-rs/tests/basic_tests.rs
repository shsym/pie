use backend_management_rs::config::Config;
use backend_management_rs::types::*;
use backend_management_rs::error::{ManagementError, ConfigError, ProcessError};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, SystemTime};
use tempfile::{tempdir, TempDir};

mod common;
use common::TestUtils;

/// Basic test for model instance creation
#[cfg(test)]
mod model_instance_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_model_instance_creation() {
        let process = TestUtils::create_mock_process();
        let instance = ModelInstance::new(
            "Llama-3.1-8B-Instruct".to_string(),
            "llama3".to_string(),
            "ipc:///tmp/test".to_string(),
            process,
            None,
        );
        
        assert_eq!(instance.model_name, "Llama-3.1-8B-Instruct");
        assert_eq!(instance.model_type, "llama3");
        assert_eq!(instance.endpoint, "ipc:///tmp/test");
        assert!(instance.started_at <= SystemTime::now());
    }
    
    #[tokio::test]
    async fn test_model_instance_is_alive() {
        let process = TestUtils::create_mock_process();
        let mut instance = ModelInstance::new(
            "test-model".to_string(),
            "test-type".to_string(),
            "ipc:///tmp/test".to_string(),
            process,
            None,
        );
        
        // Process should be alive initially
        assert!(instance.is_alive());
        
        // Terminate the process
        let _ = instance.process.kill().await;
        let _ = instance.process.wait().await;
        
        // Process should now be dead
        assert!(!instance.is_alive());
    }
    
    #[tokio::test]
    async fn test_model_instance_terminate() {
        let process = TestUtils::create_mock_process();
        let mut instance = ModelInstance::new(
            "test-model".to_string(),
            "test-type".to_string(),
            "ipc:///tmp/test".to_string(),
            process,
            None,
        );
        
        // Process should be alive
        assert!(instance.is_alive());
        
        // Terminate the instance
        let result = instance.terminate().await;
        assert!(result.is_ok());
        
        // Process should now be dead
        assert!(!instance.is_alive());
    }
    
    #[tokio::test]
    async fn test_model_instance_process_id() {
        let process = TestUtils::create_mock_process();
        let pid = process.id();
        let instance = ModelInstance::new(
            "test-model".to_string(),
            "test-type".to_string(),
            "ipc:///tmp/test".to_string(),
            process,
            None,
        );
        
        assert_eq!(instance.get_process_id(), pid);
    }
}

/// Test ManagementCommand functionality
#[cfg(test)]
mod management_command_tests {
    use super::*;
    
    #[test]
    fn test_command_creation() {
        let mut params = HashMap::new();
        params.insert("key".to_string(), serde_json::json!("value"));
        let cmd = ManagementCommand::new("status".to_string(), params.clone());
        
        assert_eq!(cmd.command, "status");
        assert_eq!(cmd.params, params);
        assert!(!cmd.correlation_id.is_empty());
    }
    
    #[test]
    fn test_command_correlation_id_uniqueness() {
        let cmd1 = ManagementCommand::new("test".to_string(), HashMap::new());
        let cmd2 = ManagementCommand::new("test".to_string(), HashMap::new());
        
        assert_ne!(cmd1.correlation_id, cmd2.correlation_id);
    }
    
    #[test]
    fn test_command_serialization() {
        let mut params = HashMap::new();
        params.insert("model_name".to_string(), serde_json::json!("test-model"));
        params.insert("param_num".to_string(), serde_json::json!(42));
        let cmd = ManagementCommand::new("load_model".to_string(), params);
        
        let serialized = serde_json::to_string(&cmd).expect("Failed to serialize");
        let deserialized: ManagementCommand = serde_json::from_str(&serialized)
            .expect("Failed to deserialize");
        
        assert_eq!(cmd.command, deserialized.command);
        assert_eq!(cmd.params, deserialized.params);
        assert_eq!(cmd.correlation_id, deserialized.correlation_id);
    }
}

/// Test ManagementResponse functionality
#[cfg(test)]
mod management_response_tests {
    use super::*;
    
    #[test]
    fn test_success_response() {
        let data = serde_json::json!({
            "message": "Operation successful"
        });
        let response = ManagementResponse::success("test-id".to_string(), Some(data.clone()));
        
        assert!(response.success);
        assert_eq!(response.correlation_id, "test-id");
        assert_eq!(response.data, Some(data));
        assert!(response.error.is_none());
    }
    
    #[test]
    fn test_error_response() {
        let response = ManagementResponse::error("test-id".to_string(), "Test error".to_string());
        
        assert!(!response.success);
        assert_eq!(response.correlation_id, "test-id");
        assert_eq!(response.error, Some("Test error".to_string()));
        assert!(response.data.is_none());
    }
    
    #[test]
    fn test_response_serialization() {
        let data = serde_json::json!({
            "status": "running",
            "count": 3
        });
        let response = ManagementResponse::success("test-123".to_string(), Some(data));
        
        let serialized = serde_json::to_string(&response).expect("Failed to serialize");
        let deserialized: ManagementResponse = serde_json::from_str(&serialized)
            .expect("Failed to deserialize");
        
        assert_eq!(response.success, deserialized.success);
        assert_eq!(response.correlation_id, deserialized.correlation_id);
        assert_eq!(response.data, deserialized.data);
        assert_eq!(response.error, deserialized.error);
    }
}

/// Test config loading functionality
#[cfg(test)]
mod config_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_config_loading() {
        let config = TestUtils::create_test_config();
        let (_temp_dir, config_path) = TestUtils::create_temp_config_file(&config);
        
        let loaded_config = Config::load(&config_path).expect("Failed to load config");
        assert_eq!(loaded_config.endpoints.cli_management, config.endpoints.cli_management);
        assert_eq!(loaded_config.model_backends, config.model_backends);
    }
    
    #[tokio::test]
    async fn test_config_file_not_found() {
        let invalid_path = PathBuf::from("/nonexistent/config.json");
        
        let result = Config::load(&invalid_path);
        assert!(result.is_err());
        
        match result.unwrap_err() {
            ConfigError::FileNotFound { .. } => {},
            _ => panic!("Expected FileNotFound error"),
        }
    }
    
    #[tokio::test]
    async fn test_config_invalid_json() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let config_path = temp_dir.path().join("invalid_config.json");
        
        std::fs::write(&config_path, "invalid json {").expect("Failed to write invalid config");
        
        let result = Config::load(&config_path);
        assert!(result.is_err());
        
        match result.unwrap_err() {
            ConfigError::InvalidJson { .. } => {},
            _ => panic!("Expected InvalidJson error"),
        }
    }
    
    #[tokio::test]
    async fn test_load_real_python_config() {
        if let Some(config) = TestUtils::try_load_python_config() {
            // Verify specific config values are loaded correctly
            assert!(config.model_backends.contains_key("llama3"));
            assert!(!config.endpoints.client_handshake.is_empty());
            assert!(!config.endpoints.cli_management.is_empty());
        }
    }
}

/// Test error handling
#[cfg(test)]
mod error_handling_tests {
    use super::*;
    
    #[test]
    fn test_management_error_from_config_error() {
        let config_error = ConfigError::FileNotFound { path: PathBuf::from("/test/path") };
        let management_error = ManagementError::from(config_error);
        
        match management_error {
            ManagementError::Config(ConfigError::FileNotFound { path }) => {
                assert_eq!(path, PathBuf::from("/test/path"));
            },
            _ => panic!("Expected Config error"),
        }
    }
    
    #[test]
    fn test_management_error_from_process_error() {
        let process_error = ProcessError::SpawnFailed("test error".to_string());
        let management_error = ManagementError::from(process_error);
        
        match management_error {
            ManagementError::Process(ProcessError::SpawnFailed(msg)) => {
                assert_eq!(msg, "test error");
            },
            _ => panic!("Expected Process error"),
        }
    }
    
    #[test]
    fn test_unknown_model_error() {
        let error = ManagementError::UnknownModel("test-model".to_string());
        let error_str = error.to_string();
        assert!(error_str.contains("test-model"));
    }
}
