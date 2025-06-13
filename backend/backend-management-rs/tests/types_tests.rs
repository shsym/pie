#[cfg(test)]
mod tests {
    use backend_management_rs::types::*;
    use std::collections::HashMap;

    #[test]
    fn test_management_command_creation() {
        let params = HashMap::new();
        let cmd = ManagementCommand::new("test_command".to_string(), params.clone());
        
        assert_eq!(cmd.command, "test_command");
        assert_eq!(cmd.params, params);
        assert!(!cmd.correlation_id.is_empty());
    }

    #[test]
    fn test_management_command_with_correlation_id() {
        let params = HashMap::new();
        let correlation_id = "test-id-123".to_string();
        let cmd = ManagementCommand::with_correlation_id(
            "test".to_string(),
            params.clone(),
            correlation_id.clone()
        );
        
        assert_eq!(cmd.correlation_id, correlation_id);
    }

    #[test]
    fn test_unique_correlation_ids() {
        let params = HashMap::new();
        let cmd1 = ManagementCommand::new("test".to_string(), params.clone());
        let cmd2 = ManagementCommand::new("test".to_string(), params);
        
        assert_ne!(cmd1.correlation_id, cmd2.correlation_id);
    }

    #[test]
    fn test_management_response_success() {
        let response = ManagementResponse::success(
            "test-id".to_string(),
            Some(serde_json::json!({"status": "ok"}))
        );
        
        assert!(response.success);
        assert!(response.error.is_none());
        assert!(response.data.is_some());
    }

    #[test]
    fn test_management_response_error() {
        let response = ManagementResponse::error(
            "test-id".to_string(),
            "Something went wrong".to_string()
        );
        
        assert!(!response.success);
        assert!(response.data.is_none());
        assert_eq!(response.error.unwrap(), "Something went wrong");
    }
}
