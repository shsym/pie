use backend_management_rs::config::Config;
use backend_management_rs::types::*;
use backend_management_rs::service::{ManagementServiceTrait, ManagementServiceFactory};
use backend_management_rs::{ManagementServiceImpl};
use backend_management_rs::error::{ManagementError, ConfigError, ProcessError};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, SystemTime};
use tempfile::{tempdir, TempDir};
use tokio::time::timeout;

mod common;
use common::TestUtils;

/// Test ModelInstance functionality
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
        let mut process = TestUtils::create_mock_process();
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
    
    #[tokio::test]
    async fn test_model_instance_with_config_path() {
        let process = TestUtils::create_mock_process();
        let config_path = PathBuf::from("/tmp/test-config.json");
        let instance = ModelInstance::new(
            "test-model".to_string(),
            "test-type".to_string(),
            "ipc:///tmp/test".to_string(),
            process,
            Some(config_path.clone()),
        );
        
        assert_eq!(instance.config_path, Some(config_path));
    }
}

/// Test ManagementCommand functionality
#[cfg(test)]
mod management_command_tests {
    use super::*;
    
    #[test]
    fn test_command_creation() {
        let params = test_params! {
            "key" => "value"
        };
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
    fn test_command_with_empty_params() {
        let cmd = ManagementCommand::new("status".to_string(), HashMap::new());
        
        assert_eq!(cmd.command, "status");
        assert!(cmd.params.is_empty());
        assert!(!cmd.correlation_id.is_empty());
    }
    
    #[test]
    fn test_command_serialization() {
        let params = test_params! {
            "model_name" => "test-model",
            "param_num" => 42
        };
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
        let data = test_params! {
            "message" => "Operation successful"
        };
        let response = ManagementResponse::success("test-id".to_string(), TestUtils::hashmap_to_value(data.clone()));
        
        assert!(response.success);
        assert_eq!(response.correlation_id, "test-id");
        assert_eq!(response.data, TestUtils::hashmap_to_value(data));
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
        let data = test_params! {
            "status" => "running",
            "count" => 3
        };
        let response = ManagementResponse::success("test-123".to_string(), TestUtils::hashmap_to_value(data));
        
        let serialized = serde_json::to_string(&response).expect("Failed to serialize");
        let deserialized: ManagementResponse = serde_json::from_str(&serialized)
            .expect("Failed to deserialize");
        
        assert_eq!(response.success, deserialized.success);
        assert_eq!(response.correlation_id, deserialized.correlation_id);
        assert_eq!(response.data, deserialized.data);
        assert_eq!(response.error, deserialized.error);
    }
}

/// Test service initialization and configuration loading
#[cfg(test)]
mod service_initialization_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_service_creation_with_valid_config() {
        let config = TestUtils::create_test_config();
        let (_temp_dir, config_path) = TestUtils::create_temp_config_file(&config);
        
        let service: Result<ManagementServiceImpl, _> = ManagementServiceImpl::create_service(&config_path).await;
        assert!(service.is_ok());
    }
    
    #[tokio::test]
    async fn test_service_creation_with_invalid_config_path() {
        let invalid_path = PathBuf::from("/nonexistent/config.json");
        
        let service = ManagementServiceImpl::create_service(&invalid_path).await;
        assert!(service.is_err());
        
        match service.unwrap_err() {
            ManagementError::Config(ConfigError::FileNotFound { path: _ }) => {},
            _ => panic!("Expected ConfigError::FileNotFound"),
        }
    }
    
    #[tokio::test]
    async fn test_service_creation_with_invalid_json() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let config_path = temp_dir.path().join("invalid_config.json");
        
        std::fs::write(&config_path, "invalid json {").expect("Failed to write invalid config");
        
        let service = ManagementServiceImpl::create_service(&config_path).await;
        assert!(service.is_err());
        
        match service.unwrap_err() {
            ManagementError::Config(ConfigError::InvalidJson { .. }) => {},
            _ => panic!("Expected ConfigError::InvalidJson"),
        }
    }
    
    #[tokio::test]
    async fn test_service_load_real_python_config() {
        if let Some(config) = TestUtils::try_load_python_config() {
            let (_temp_dir, config_path) = TestUtils::create_temp_config_file(&config);
            
            let service = ManagementServiceImpl::create_service(&config_path).await;
            assert!(service.is_ok());
            
            // Verify specific config values are loaded correctly
            assert!(config.model_backends.contains_key("llama3"));
            assert!(!config.endpoints.client_handshake.is_empty());
            assert!(!config.endpoints.cli_management.is_empty());
        }
    }
}

/// Test model type determination and mapping
#[cfg(test)]
mod model_type_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_model_type_exact_matches() {
        let config = TestUtils::create_test_config();
        let (_temp_dir, config_path) = TestUtils::create_temp_config_file(&config);
        
        let service = ManagementServiceImpl::create_service(&config_path).await
            .expect("Failed to create service");
        
        // Test exact matches from config
        assert!(matches!(service.get_model_type("Llama-3.1-8B-Instruct"), Ok(ref model_type) if model_type == "llama3"));
        assert!(matches!(service.get_model_type("DeepSeek-V3-0324"), Ok(ref model_type) if model_type == "deepseek"));
    }
    
    #[tokio::test]
    async fn test_model_type_unknown_model() {
        let config = TestUtils::create_test_config();
        let (_temp_dir, config_path) = TestUtils::create_temp_config_file(&config);
        
        let service = ManagementServiceImpl::create_service(&config_path).await
            .expect("Failed to create service");
        
        let result = service.get_model_type("unknown-model");
        assert!(result.is_err());
        
        match result.unwrap_err() {
            ManagementError::UnknownModel(_) => {},
            _ => panic!("Expected UnknownModel error"),
        }
    }
    
    #[tokio::test]
    async fn test_model_type_case_sensitivity() {
        let config = TestUtils::create_test_config();
        let (_temp_dir, config_path) = TestUtils::create_temp_config_file(&config);
        
        let service = ManagementServiceImpl::create_service(&config_path).await
            .expect("Failed to create service");
        
        // Test case-sensitive matching - this should fail and suggest correct case
        let result = service.get_model_type("llama-3.1-8b-instruct");
        assert!(result.is_err());
        
        // Error should contain suggestion for correct case
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Llama-3.1-8B-Instruct"));
    }
}

/// Test endpoint generation
#[cfg(test)]
mod endpoint_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_unique_endpoint_generation() {
        let config = TestUtils::create_test_config();
        let (_temp_dir, config_path) = TestUtils::create_temp_config_file(&config);
        
        let service = ManagementServiceImpl::create_service(&config_path).await
            .expect("Failed to create service");
        
        let endpoint1 = service.generate_unique_endpoint();
        let endpoint2 = service.generate_unique_endpoint();
        
        assert_ne!(endpoint1, endpoint2);
        assert!(endpoint1.starts_with("ipc:///tmp/symphony-model-"));
        assert!(endpoint2.starts_with("ipc:///tmp/symphony-model-"));
    }
    
    #[tokio::test]
    async fn test_endpoint_format() {
        let config = TestUtils::create_test_config();
        let (_temp_dir, config_path) = TestUtils::create_temp_config_file(&config);
        
        let service = ManagementServiceImpl::create_service(&config_path).await
            .expect("Failed to create service");
        
        let endpoint = service.generate_unique_endpoint();
        
        // Should follow the pattern: ipc:///tmp/symphony-model-{uuid}
        assert!(endpoint.starts_with("ipc:///tmp/symphony-model-"));
        
        // Extract the UUID part and verify it's a valid UUID format
        let uuid_part = endpoint.strip_prefix("ipc:///tmp/symphony-model-").unwrap();
        assert_eq!(uuid_part.len(), 8); // Short UUID (first 8 characters)
        assert!(uuid_part.chars().all(|c| c.is_ascii_alphanumeric() || c == '-')); // Valid UUID characters
    }
}

/// Test error handling patterns
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
    fn test_error_display() {
        let error = ManagementError::UnknownModel("test-model".to_string());
        let error_str = error.to_string();
        assert!(error_str.contains("test-model"));
    }
}
