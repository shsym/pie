use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, Duration};
use uuid::Uuid;
use tokio::process::Child as TokioChild;
use tracing::{info, warn, error};

/// Represents a running model backend instance
#[derive(Debug)]
pub struct ModelInstance {
    /// Name of the model (e.g., "Llama-3.1-8B-Instruct")
    pub model_name: String,
    /// Type of the model (e.g., "llama3", "deepseek")
    pub model_type: String,
    /// IPC endpoint for this model instance
    pub endpoint: String,
    /// Running process handle (async version)
    pub process: TokioChild,
    /// Optional path to model configuration file
    pub config_path: Option<std::path::PathBuf>,
    /// Timestamp when the instance was started
    pub started_at: SystemTime,
}

impl ModelInstance {
    /// Create a new model instance
    pub fn new(
        model_name: String,
        model_type: String,
        endpoint: String,
        process: TokioChild,
        config_path: Option<std::path::PathBuf>,
    ) -> Self {
        Self {
            model_name,
            model_type,
            endpoint,
            process,
            config_path,
            started_at: SystemTime::now(),
        }
    }

    /// Check if the backend process is still running
    pub fn is_alive(&mut self) -> bool {
        match self.process.try_wait() {
            Ok(None) => true,  // Process is still running
            Ok(Some(_)) => false, // Process has exited
            Err(_) => false,   // Error checking process status
        }
    }

    /// Get the process ID
    pub fn pid(&self) -> Option<u32> {
        self.process.id()
    }

    /// Gracefully terminate the backend process
    pub async fn terminate(&mut self) -> Result<(), std::io::Error> {
        info!("Terminating model instance: {}", self.model_name);
        
        // Send SIGTERM first
        if let Err(e) = self.process.kill().await {
            warn!("Failed to send SIGTERM to process: {}", e);
            return Err(e);
        }

        // Wait up to 10 seconds for graceful shutdown
        let timeout = Duration::from_secs(10);
        match tokio::time::timeout(timeout, self.process.wait()).await {
            Ok(Ok(status)) => {
                info!("Process {} exited with status: {}", self.model_name, status);
                Ok(())
            }
            Ok(Err(e)) => {
                error!("Error waiting for process {}: {}", self.model_name, e);
                Err(e)
            }
            Err(_) => {
                warn!("Process {} did not exit gracefully, sending SIGKILL", self.model_name);
                // Force kill if timeout
                self.process.kill().await
            }
        }
    }

    /// Get uptime duration
    pub fn uptime(&self) -> Duration {
        self.started_at.elapsed().unwrap_or(Duration::ZERO)
    }

    /// Get the process ID if available (alias for pid)
    pub fn get_process_id(&self) -> Option<u32> {
        self.pid()
    }
}

/// Represents a command from the CLI tool to the management service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagementCommand {
    /// The command to execute (e.g., "status", "load_model", "unload_model")
    pub command: String,
    /// Parameters for the command
    pub params: HashMap<String, serde_json::Value>,
    /// Unique correlation ID for request/response matching
    pub correlation_id: String,
}

impl ManagementCommand {
    /// Create a new management command with auto-generated correlation ID
    pub fn new(command: String, params: HashMap<String, serde_json::Value>) -> Self {
        Self {
            command,
            params,
            correlation_id: Uuid::new_v4().to_string(),
        }
    }

    /// Create a new management command with specific correlation ID
    pub fn with_correlation_id(
        command: String,
        params: HashMap<String, serde_json::Value>,
        correlation_id: String,
    ) -> Self {
        Self {
            command,
            params,
            correlation_id,
        }
    }
}

/// Response to a management command
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagementResponse {
    /// Correlation ID matching the request
    pub correlation_id: String,
    /// Whether the command was successful
    pub success: bool,
    /// Response data (command-specific)
    pub data: Option<serde_json::Value>,
    /// Error message if success is false
    pub error: Option<String>,
}

impl ManagementResponse {
    /// Create a successful response
    pub fn success(correlation_id: String, data: Option<serde_json::Value>) -> Self {
        Self {
            correlation_id,
            success: true,
            data,
            error: None,
        }
    }

    /// Create an error response
    pub fn error(correlation_id: String, error: String) -> Self {
        Self {
            correlation_id,
            success: false,
            data: None,
            error: Some(error),
        }
    }
}

/// Status of a model instance for reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInstanceStatus {
    pub model_name: String,
    pub model_type: String,
    pub endpoint: String,
    pub is_alive: bool,
    pub pid: Option<u32>,
    pub started_at: SystemTime,
    pub uptime: Duration,
    pub config_path: Option<std::path::PathBuf>,
}

impl From<&mut ModelInstance> for ModelInstanceStatus {
    fn from(instance: &mut ModelInstance) -> Self {
        Self {
            model_name: instance.model_name.clone(),
            model_type: instance.model_type.clone(),
            endpoint: instance.endpoint.clone(),
            is_alive: instance.is_alive(),
            pid: instance.pid(),
            started_at: instance.started_at,
            uptime: instance.uptime(),
            config_path: instance.config_path.clone(),
        }
    }
}
