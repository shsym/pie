use backend_management_rs::types::*;
use backend_management_rs::{ManagementServiceImpl};
use backend_management_rs::service::ManagementServiceFactory;
use backend_management_rs::error::ManagementError;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;
use tokio::time::sleep;
use serde_json::json;

mod common;
use common::TestUtils;

/// Integration tests for the complete service workflow
#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_service_initialization_integration() {
        let config = TestUtils::create_test_config();
        let (_temp_dir, config_path) = TestUtils::create_temp_config_file(&config);
        
        // Test service creation and initialization
        let service: Result<ManagementServiceImpl, _> = ManagementServiceImpl::create_service(&config_path).await;
        assert!(service.is_ok());
        
        let _service = service.unwrap();
        
        // TODO: Test socket initialization once implemented
        // assert!(service.initialize_sockets().await.is_ok());
    }
    
    #[tokio::test]
    async fn test_service_config_compatibility() {
        // Test that Rust service can use Python config
        if let Some(python_config) = TestUtils::try_load_python_config() {
            let (_temp_dir, config_path) = TestUtils::create_temp_config_file(&python_config);
            
            let service: Result<ManagementServiceImpl, _> = ManagementServiceImpl::create_service(&config_path).await;
            assert!(service.is_ok());
            
            let service = service.unwrap();
            
            // Verify config fields are properly loaded
            let test_models = vec!["Llama-3.1-8B-Instruct", "DeepSeek-V3-0324"];
            for model in test_models {
                let model_type = service.get_model_type(model);
                assert!(model_type.is_ok(), "Failed to get model type for {}", model);
            }
        }
    }
}

/// Test process management integration
#[cfg(test)]
mod process_management_integration_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_model_instance_lifecycle() {
        let process = TestUtils::create_mock_process();
        let pid = process.id();
        
        let mut instance = ModelInstance::new(
            "test-model".to_string(),
            "test-type".to_string(),
            "ipc:///tmp/test-lifecycle".to_string(),
            process,
            None,
            vec!["l4m".to_string()],
        );
        
        // Process should be alive initially
        assert!(instance.is_alive());
        assert_eq!(instance.get_process_id(), pid);
        
        // Terminate the instance
        let result = instance.terminate().await;
        assert!(result.is_ok());
        
        // Process should be dead
        assert!(!instance.is_alive());
    }
    
    #[tokio::test]
    async fn test_multiple_model_instances() {
        let mut instances = Vec::new();
        
        // Create multiple model instances
        for i in 0..3 {
            let process = TestUtils::create_mock_process();
            let instance = ModelInstance::new(
                format!("test-model-{}", i),
                "test-type".to_string(),
                format!("ipc:///tmp/test-{}", i),
                process,
                None,
                vec!["l4m".to_string()],
            );
            instances.push(instance);
        }
        
        // All should be alive
        for instance in &mut instances {
            assert!(instance.is_alive());
        }
        
        // Terminate all instances
        for instance in &mut instances {
            let result = instance.terminate().await;
            assert!(result.is_ok());
        }
        
        // All should be dead
        for instance in &mut instances {
            assert!(!instance.is_alive());
        }
    }
    
    #[tokio::test]
    async fn test_process_cleanup_on_drop() {
        let process = TestUtils::create_mock_process();
        let pid = process.id();
        
        {
            let mut instance = ModelInstance::new(
                "test-cleanup".to_string(),
                "test-type".to_string(),
                "ipc:///tmp/test-cleanup".to_string(),
                process,
                None,
                vec!["l4m".to_string()],
            );
            
            assert!(instance.is_alive());
            assert_eq!(instance.get_process_id(), pid);
        } // instance goes out of scope here
        
        // Give some time for cleanup
        sleep(Duration::from_millis(100)).await;
        
        // TODO: Add verification that process was properly cleaned up
        // This would require checking if the PID is still alive in the system
    }
}

/// Test command-response flow integration
#[cfg(test)]
mod command_response_integration_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_status_command_flow() {
        let cmd = TestUtils::create_test_command("status");
        let correlation_id = cmd.correlation_id.clone();
        
        // Simulate service processing the command
        let data = test_params! {
            "status" => "running",
            "uptime_seconds" => 123,
            "model_count" => 0
        };
        let response = ManagementResponse::success(correlation_id.clone(), TestUtils::hashmap_to_value(data.clone()));
        
        // Verify correlation
        assert_eq!(cmd.correlation_id, response.correlation_id);
        assert!(response.success);
        assert_eq!(response.data, TestUtils::hashmap_to_value(data));
    }
    
    #[tokio::test]
    async fn test_load_model_command_flow() {
        let params = test_params! {
            "model_name" => "Llama-3.1-8B-Instruct"
        };
        let cmd = TestUtils::create_test_command_with_params("load_model", params);
        let correlation_id = cmd.correlation_id.clone();
        
        // Simulate successful model loading
        let response_data = test_params! {
            "model_name" => "Llama-3.1-8B-Instruct",
            "endpoint" => "ipc:///tmp/symphony-model-abc123",
            "pid" => 12345,
            "model_type" => "llama3"
        };
        let response = ManagementResponse::success(correlation_id.clone(), TestUtils::hashmap_to_value(response_data));
        
        assert_eq!(cmd.correlation_id, response.correlation_id);
        assert!(response.success);
        
        let data = response.data.unwrap();
        assert_eq!(data["model_name"], serde_json::json!("Llama-3.1-8B-Instruct"));
        assert!(data["endpoint"].as_str().unwrap().starts_with("ipc:///tmp/symphony-model-"));
    }
    
    #[tokio::test]
    async fn test_load_model_error_flow() {
        let params = test_params! {
            "model_name" => "unknown-model"
        };
        let cmd = TestUtils::create_test_command_with_params("load_model", params);
        let correlation_id = cmd.correlation_id.clone();
        
        // Simulate model loading error
        let response = ManagementResponse::error(
            correlation_id.clone(),
            "Unknown model 'unknown-model' - not found in configuration".to_string(),
        );
        
        assert_eq!(cmd.correlation_id, response.correlation_id);
        assert!(!response.success);
        assert!(response.error.as_ref().unwrap().contains("unknown-model"));
    }
    
    #[tokio::test]
    async fn test_unload_model_command_flow() {
        let params = test_params! {
            "model_name" => "Llama-3.1-8B-Instruct"
        };
        let cmd = TestUtils::create_test_command_with_params("unload_model", params);
        let correlation_id = cmd.correlation_id.clone();
        
        // Simulate successful model unloading
        let response_data = test_params! {
            "model_name" => "Llama-3.1-8B-Instruct",
            "message" => "Model successfully unloaded"
        };
        let response = ManagementResponse::success(correlation_id.clone(), TestUtils::hashmap_to_value(response_data));
        
        assert_eq!(cmd.correlation_id, response.correlation_id);
        assert!(response.success);
    }
    
    #[tokio::test]
    async fn test_list_models_command_flow() {
        let cmd = TestUtils::create_test_command("list_models");
        let correlation_id = cmd.correlation_id.clone();
        
        // Simulate listing models response
        let models_data = serde_json::json!([
            {
                "model_name": "Llama-3.1-8B-Instruct",
                "model_type": "llama3",
                "endpoint": "ipc:///tmp/symphony-model-abc123",
                "pid": 12345,
                "status": "running"
            },
            {
                "model_name": "DeepSeek-V3-0324",
                "model_type": "deepseek", 
                "endpoint": "ipc:///tmp/symphony-model-def456",
                "pid": 12346,
                "status": "running"
            }
        ]);
        
        let mut response_data = HashMap::new();
        response_data.insert("models".to_string(), models_data);
        
        let response = ManagementResponse::success(correlation_id.clone(), TestUtils::hashmap_to_value(response_data));
        
        assert_eq!(cmd.correlation_id, response.correlation_id);
        assert!(response.success);
        
        let models = &response.data.unwrap()["models"];
        assert!(models.is_array());
        assert_eq!(models.as_array().unwrap().len(), 2);
    }
}

/// Test concurrent operations integration
#[cfg(test)]
mod concurrent_integration_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_concurrent_command_processing() {
        let mut commands = Vec::new();
        let mut correlation_ids = Vec::new();
        
        // Create multiple commands
        for i in 0..5 {
            let params = test_params! {
                "test_param" => format!("value_{}", i)
            };
            let cmd = TestUtils::create_test_command_with_params("test_command", params);
            correlation_ids.push(cmd.correlation_id.clone());
            commands.push(cmd);
        }
        
        // Simulate concurrent processing
        let mut responses = Vec::new();
        for (i, cmd) in commands.iter().enumerate() {
            let data = test_params! {
                "result" => format!("processed_{}", i)
            };
            let response = ManagementResponse::success(cmd.correlation_id.clone(), TestUtils::hashmap_to_value(data));
            responses.push(response);
        }
        
        // Verify all correlations are correct
        for (i, response) in responses.iter().enumerate() {
            assert_eq!(response.correlation_id, correlation_ids[i]);
            assert!(response.success);
        }
    }
    
    #[tokio::test]
    async fn test_multiple_model_loading_simulation() {
        let model_names = vec![
            "Llama-3.1-8B-Instruct",
            "DeepSeek-V3-0324",
        ];
        
        let mut commands = Vec::new();
        let mut responses = Vec::new();
        
        // Create load commands for each model
        for model_name in &model_names {
            let params = test_params! {
                "model_name" => model_name
            };
            let cmd = TestUtils::create_test_command_with_params("load_model", params);
            
            // Simulate successful loading response
            let response_data = test_params! {
                "model_name" => model_name,
                "endpoint" => format!("ipc:///tmp/symphony-model-{}", uuid::Uuid::new_v4()),
                "pid" => 12345 + commands.len(),
                "model_type" => if model_name.contains("Llama") { "llama3" } else { "deepseek" }
            };
            let response = ManagementResponse::success(cmd.correlation_id.clone(), TestUtils::hashmap_to_value(response_data));
            
            commands.push(cmd);
            responses.push(response);
        }
        
        // Verify all models were "loaded"
        assert_eq!(commands.len(), model_names.len());
        assert_eq!(responses.len(), model_names.len());
        
        for (cmd, response) in commands.iter().zip(responses.iter()) {
            assert_eq!(cmd.correlation_id, response.correlation_id);
            assert!(response.success);
        }
    }
}

/// Test error handling integration
#[cfg(test)]
mod error_integration_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_service_error_propagation() {
        // Test configuration error propagation
        let invalid_path = PathBuf::from("/nonexistent/config.json");
        let service_result: Result<ManagementServiceImpl, _> = ManagementServiceImpl::create_service(&invalid_path).await;
        
        assert!(service_result.is_err());
        let error = service_result.unwrap_err();
        
        match error {
            ManagementError::Config(_) => {}, // Expected
            _ => panic!("Expected Config error, got: {:?}", error),
        }
    }
    
    #[tokio::test]
    async fn test_model_not_found_error_handling() {
        let config = TestUtils::create_test_config();
        let (_temp_dir, config_path) = TestUtils::create_temp_config_file(&config);
        
        let service = ManagementServiceImpl::create_service(&config_path).await
            .expect("Failed to create service");
        
        let result = service.get_model_type("nonexistent-model");
        assert!(result.is_err());
        
        match result.unwrap_err() {
            ManagementError::UnknownModel(model_name) => {
                assert!(model_name.starts_with("nonexistent-model"));
                assert!(model_name.contains("Available models:"));
            },
            _ => panic!("Expected UnknownModel error"),
        }
    }
    
    #[tokio::test]
    async fn test_error_response_correlation() {
        let params = test_params! {
            "model_name" => "invalid-model"
        };
        let cmd = TestUtils::create_test_command_with_params("load_model", params);
        let correlation_id = cmd.correlation_id.clone();
        
        // Simulate error response
        let response = ManagementResponse::error(
            correlation_id.clone(),
            "Model loading failed: backend script not found".to_string(),
        );
        
        assert_eq!(cmd.correlation_id, response.correlation_id);
        assert!(!response.success);
        assert!(response.error.is_some());
        assert!(response.data.is_none());
    }
}

/// Test service lifecycle integration
#[cfg(test)]
mod lifecycle_integration_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_service_startup_sequence() {
        let config = TestUtils::create_test_config();
        let (_temp_dir, config_path) = TestUtils::create_temp_config_file(&config);
        
        // Test service creation
        let service: Result<ManagementServiceImpl, _> = ManagementServiceImpl::create_service(&config_path).await;
        assert!(service.is_ok());
        
        // TODO: Test socket initialization, message handling setup
        // Once these are implemented, add tests here
    }
    
    #[tokio::test]
    async fn test_graceful_shutdown_simulation() {
        let mut instances = Vec::new();
        
        // Create some model instances
        for i in 0..3 {
            let process = TestUtils::create_mock_process();
            let instance = ModelInstance::new(
                format!("test-model-{}", i),
                "test-type".to_string(),
                format!("ipc:///tmp/test-shutdown-{}", i),
                process,
                None,
                vec!["l4m".to_string()],
            );
            instances.push(instance);
        }
        
        // Verify all are alive
        for instance in &mut instances {
            assert!(instance.is_alive());
        }
        
        // Simulate graceful shutdown
        for instance in &mut instances {
            let result = instance.terminate().await;
            assert!(result.is_ok());
        }
        
        // Verify all are terminated
        for instance in &mut instances {
            assert!(!instance.is_alive());
        }
    }
    
    #[tokio::test]
    async fn test_service_status_reporting() {
        // Test status command with no models loaded
        let cmd = TestUtils::create_test_command("status");
        let correlation_id = cmd.correlation_id.clone();
        
        // Get expected endpoint values from a test config instance
        let test_config = TestUtils::create_test_config();

        let data = test_params! {
            "status" => "running",
            "client_endpoint" => test_config.endpoints.client_handshake.clone(),
            "cli_endpoint" => test_config.endpoints.cli_management.clone(),
            "models_loaded_count" => 0,
            "active_model_details" => json!([])
        };
        let response = ManagementResponse::success(correlation_id, TestUtils::hashmap_to_value(data.clone()));
        
        assert!(response.success);
        let response_data = response.data.as_ref().unwrap();
        assert_eq!(response_data["status"], serde_json::json!("running"));
        assert_eq!(response_data["client_endpoint"], serde_json::json!(test_config.endpoints.client_handshake));
        assert_eq!(response_data["cli_endpoint"], serde_json::json!(test_config.endpoints.cli_management));
        assert_eq!(response_data["models_loaded_count"], serde_json::json!(0));
        assert!(response_data["active_model_details"].is_array());
        assert_eq!(response_data["active_model_details"].as_array().unwrap().len(), 0);
    }
}
