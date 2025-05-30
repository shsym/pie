use crate::config::Config;
use crate::error::Result;
use crate::types::{ManagementCommand, ManagementResponse, ModelInstanceStatus};
use async_trait::async_trait;
use std::path::PathBuf;

/// Trait defining the core management service operations
#[async_trait]
pub trait ManagementServiceTrait {
    /// Start the management service
    async fn start(&mut self) -> Result<()>;

    /// Stop the management service gracefully
    async fn stop(&mut self) -> Result<()>;

    /// Check if the service is running
    fn is_running(&self) -> bool;

    /// Get the current status of the service and all model instances
    async fn get_status(&mut self) -> Result<ServiceStatus>;

    /// Load/start a model instance
    async fn load_model(
        &mut self,
        model_name: &str,
        config_path: Option<PathBuf>,
    ) -> Result<ModelInstanceStatus>;

    /// Unload/stop a model instance
    async fn unload_model(&mut self, model_name: &str) -> Result<()>;

    /// Get status of a specific model instance
    async fn get_model_status(&mut self, model_name: &str) -> Result<ModelInstanceStatus>;

    /// List all running model instances
    async fn list_models(&mut self) -> Result<Vec<ModelInstanceStatus>>;

    /// Handle a management command (from CLI)
    async fn handle_command(&mut self, command: ManagementCommand) -> Result<ManagementResponse>;

    /// Handle a client handshake request
    async fn handle_client_handshake(&mut self, request: &[u8]) -> Result<Vec<u8>>;

    /// Perform health check on all running instances
    async fn health_check(&mut self) -> Result<Vec<ModelInstanceStatus>>;
}

/// Overall service status
#[derive(Debug, Clone)]
pub struct ServiceStatus {
    /// Whether the service is running
    pub is_running: bool,
    /// Number of active model instances
    pub active_models: usize,
    /// Service uptime in seconds
    pub uptime_seconds: u64,
    /// Service configuration
    pub config: Config,
    /// Status of all model instances
    pub model_instances: Vec<ModelInstanceStatus>,
    /// Service endpoints
    pub endpoints: ServiceEndpoints,
}

/// Service endpoint information
#[derive(Debug, Clone)]
pub struct ServiceEndpoints {
    /// Client handshake endpoint
    pub client_handshake: String,
    /// CLI management endpoint
    pub cli_management: String,
}

/// Factory trait for creating management service instances
pub trait ManagementServiceFactory {
    type Service: ManagementServiceTrait;

    /// Create a new management service instance
    fn create(
        config: Option<PathBuf>,
        backend_base_path: Option<PathBuf>,
    ) -> Result<Self::Service>;

    /// Create a management service with default configuration
    fn create_default() -> Result<Self::Service> {
        Self::create(None, None)
    }

    /// Create a service from a config file path (helper for tests)
    async fn create_service(config_path: &PathBuf) -> Result<Self::Service> {
        Self::create(Some(config_path.clone()), None)
    }
}

/// Utility functions for the management service
pub mod utils {
    use uuid::Uuid;

    /// Generate a unique IPC endpoint for a model instance
    pub fn generate_unique_endpoint() -> String {
        let instance_id = Uuid::new_v4()
            .to_string()
            .chars()
            .take(8)
            .collect::<String>();
        format!("ipc:///tmp/symphony-model-{}", instance_id)
    }

    /// Check if an endpoint is valid IPC format
    pub fn is_valid_ipc_endpoint(endpoint: &str) -> bool {
        endpoint.starts_with("ipc://")
    }

    /// Extract the socket path from an IPC endpoint
    pub fn extract_socket_path(endpoint: &str) -> Option<String> {
        if endpoint.starts_with("ipc://") {
            Some(endpoint[6..].to_string())
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::utils::*;

    #[test]
    fn test_generate_unique_endpoint() {
        let endpoint1 = generate_unique_endpoint();
        let endpoint2 = generate_unique_endpoint();
        
        assert_ne!(endpoint1, endpoint2);
        assert!(endpoint1.starts_with("ipc:///tmp/symphony-model-"));
        assert!(endpoint2.starts_with("ipc:///tmp/symphony-model-"));
    }

    #[test]
    fn test_is_valid_ipc_endpoint() {
        assert!(is_valid_ipc_endpoint("ipc:///tmp/test"));
        assert!(is_valid_ipc_endpoint("ipc://./test.sock"));
        assert!(!is_valid_ipc_endpoint("tcp://localhost:5555"));
        assert!(!is_valid_ipc_endpoint("invalid"));
    }

    #[test]
    fn test_extract_socket_path() {
        assert_eq!(
            extract_socket_path("ipc:///tmp/test"),
            Some("/tmp/test".to_string())
        );
        assert_eq!(
            extract_socket_path("ipc://./test.sock"),
            Some("./test.sock".to_string())
        );
        assert_eq!(extract_socket_path("tcp://localhost:5555"), None);
    }
}
