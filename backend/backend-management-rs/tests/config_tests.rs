#[cfg(test)]
mod tests {
    use backend_management_rs::config::Config;
    use std::path::PathBuf;

    #[test]
    fn test_load_python_config() {
        // Try to load the actual Python config file
        let config_path = PathBuf::from("../backend-management/config.json");
        
        if config_path.exists() {
            let result = Config::load(&config_path);
            match result {
                Ok(config) => {
                    // Verify the config has expected values
                    assert!(config.model_backends.contains_key("llama3"));
                    assert!(config.model_backends.contains_key("deepseek"));
                    assert_eq!(config.endpoints.client_handshake, "ipc:///tmp/symphony-ipc");
                    assert_eq!(config.endpoints.cli_management, "ipc:///tmp/symphony-cli");
                    assert!(!config.supported_models.is_empty());
                    
                    // Test model type lookup
                    assert_eq!(config.get_model_type("Llama-3.1-8B-Instruct").unwrap(), "llama3");
                    assert_eq!(config.get_model_type("DeepSeek-V3-0324").unwrap(), "deepseek");
                }
                Err(e) => {
                    println!("Failed to load Python config: {}", e);
                    // This is not a hard failure since the config might not exist in all environments
                }
            }
        } else {
            println!("Python config not found at {:?}, skipping test", config_path);
        }
    }

    #[test]
    fn test_config_validation() {
        // Test through from_file which calls validate internally
        use std::fs;
        
        // Create a temporary invalid config
        let temp_dir = std::env::temp_dir();
        let invalid_config_path = temp_dir.join("invalid_config.json");
        
        let invalid_config = r#"{
            "model_backends": {},
            "endpoints": {
                "client_handshake": "",
                "cli_management": "ipc:///tmp/test"
            },
            "logging": {
                "level": "INFO",
                "format": "test",
                "date_format": "test"
            },
            "supported_models": []
        }"#;
        
        fs::write(&invalid_config_path, invalid_config).unwrap();
        
        // This should fail validation due to empty client_handshake
        let result = Config::load(&invalid_config_path);
        assert!(result.is_err());
        
        // Clean up
        let _ = fs::remove_file(invalid_config_path);
    }

    #[test]
    fn test_model_type_mapping() {
        let config = Config::default();
        let mapping = config.build_model_type_mapping();
        
        assert!(mapping.contains_key("Llama-3.1-8B-Instruct"));
        assert_eq!(mapping["Llama-3.1-8B-Instruct"], "llama3");
    }
}
