use crate::types::ModelInstance;
use crate::error::{ProcessError, ProcessResult};
use crate::config::Config;
use crate::proto::handshake;
use prost::Message;
use std::path::{Path, PathBuf};
use tokio::process::Command;
use tokio::time::Duration;
use tracing::{info, warn, debug};
use uuid::Uuid;
use zeromq::{DealerSocket, Socket, SocketRecv, SocketSend, ZmqMessage};

/// Process manager for backend model instances
#[derive(Clone, Debug)]
pub struct ProcessManager {
    /// Base path to backend scripts
    backend_base_path: PathBuf,
    /// Configuration
    config: Config,
}

impl ProcessManager {
    /// Create a new process manager
    pub fn new(backend_base_path: PathBuf, config: Config) -> Self {
        Self {
            backend_base_path,
            config,
        }
    }

    /// Generate a unique IPC endpoint
    pub fn generate_unique_endpoint(&self) -> String {
        let instance_id = Uuid::new_v4().to_string();
        let short_id = &instance_id[..8];
        format!("ipc:///tmp/symphony-model-{}", short_id)
    }

    /// Get the default backend base path using multiple strategies
    pub fn get_default_backend_path() -> PathBuf {
        // Strategy 1: Look relative to current executable
        if let Ok(exe_path) = std::env::current_exe() {
            if let Some(parent) = exe_path.parent() {
                let backend_path = parent.join("../../backend/backend-python");
                if backend_path.exists() {
                    return backend_path;
                }
            }
        }

        // Strategy 2: Look relative to current working directory
        let cwd_backend = std::env::current_dir()
            .unwrap_or_default()
            .join("backend/backend-python");
        
        if cwd_backend.exists() {
            return cwd_backend;
        }

        // Strategy 3: Walk up directory tree looking for symphony project root
        let mut current_path = std::env::current_dir().unwrap_or_default();
        for _ in 0..5 {
            // Check if this looks like symphony project root
            if current_path.join("backend").exists() 
                && current_path.join("engine").exists() 
                && current_path.join("example-apps").exists() {
                
                let potential_backend = current_path.join("backend/backend-python");
                if potential_backend.exists() {
                    return potential_backend;
                }
            }
            
            // Move up one directory
            match current_path.parent() {
                Some(parent) => current_path = parent.to_path_buf(),
                None => break,
            }
        }

        // Fallback: return the expected path even if it doesn't exist
        warn!("Could not find backend-python directory, using fallback");
        PathBuf::from("backend/backend-python")
    }

    /// Spawn a new backend process for a model
    pub async fn spawn_model_instance(
        &self,
        model_name: &str,
        config_path: Option<&Path>,
    ) -> ProcessResult<ModelInstance> {
        // Get model type from configuration
        let model_type = self.config.get_model_type(model_name)
            .ok_or_else(|| ProcessError::UnknownModel(model_name.to_string()))?;

        // Get full model name (e.g., "meta-llama/Llama-3.1-8B-Instruct" instead of "Llama-3.1-8B-Instruct")
        let full_model_name = self.config.get_full_model_name(model_name)
            .ok_or_else(|| ProcessError::UnknownModel(model_name.to_string()))?;

        // Get backend script for this model type
        let backend_script = self.config.get_backend_script(model_type)
            .ok_or_else(|| ProcessError::UnknownBackend(model_type.to_string()))?;

        // Generate unique endpoint
        let endpoint = self.generate_unique_endpoint();

        // Build command to spawn backend process
        let script_path = self.backend_base_path.join(backend_script);
        
        if !script_path.exists() {
            return Err(ProcessError::ScriptNotFound(script_path));
        }

        let mut command = Command::new("python3");
        command
            .arg(&script_path)
            .arg("--ipc-endpoint")
            .arg(&endpoint)
            .arg("--model-name")
            .arg(full_model_name)
            .current_dir(&self.backend_base_path);

        // Dynamic model loading is now supported via --model-name argument

        // Set environment variables
        command
            .env("PYTHONPATH", &self.backend_base_path)
            .env("PIE_MODEL_NAME", full_model_name)
            .env("PIE_ENDPOINT", &endpoint);

        info!("Spawning backend process for model '{}' (full name: '{}') with endpoint '{}'", model_name, full_model_name, endpoint);
        debug!("Command: {:?}", command);

        // Spawn the process
        let process = command.spawn()
            .map_err(|e| ProcessError::SpawnFailed(format!("Failed to spawn {}: {}", script_path.display(), e)))?;

        info!("Successfully spawned model instance: {} (PID: {:?})", model_name, process.id());

        // Give the backend a moment to start up
        tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;

        // Discover protocols from the backend
        let supported_protocols = match self.discover_protocols(&endpoint).await {
            Ok(protocols) => {
                info!("Discovered protocols for {}: {:?}", model_name, protocols);
                protocols
            }
            Err(e) => {
                warn!("Failed to discover protocols for {}: {}. Using fallback protocols.", model_name, e);
                // Fallback to basic protocols if discovery fails
                vec!["l4m".to_string(), "ping".to_string()]
            }
        };

        let instance = ModelInstance::new(
            model_name.to_string(),
            model_type.to_string(),
            endpoint,
            process,
            config_path.map(|p| p.to_path_buf()),
            supported_protocols,
        );

        Ok(instance)
    }

    /// Health check for a model instance
    pub async fn health_check(&self, instance: &ModelInstance) -> bool {
        // For now, just check if process is alive
        // In the future, this could ping the endpoint to verify it's responding
        instance.process.id().is_some()
    }

    /// Get backend script path for a model type
    pub fn get_backend_script_path(&self, model_type: &str) -> Option<PathBuf> {
        self.config.get_backend_script(model_type)
            .map(|script| self.backend_base_path.join(script))
    }

    /// Discover protocols supported by a backend by performing a handshake
    pub async fn discover_protocols(&self, endpoint: &str) -> Result<Vec<String>, ProcessError> {
        debug!("Starting protocol discovery for endpoint: {}", endpoint);
        
        // Retry discovery with exponential backoff
        let mut retry_count = 0;
        let max_retries = 5;
        let mut delay_ms = 500; // Start with 500ms delay
        
        loop {
            debug!("Protocol discovery attempt {}/{} for endpoint: {}", retry_count + 1, max_retries + 1, endpoint);
            
            let result = self.try_discover_protocols(endpoint).await;
            
            match result {
                Ok(protocols) => {
                    debug!("Successfully discovered protocols from {}: {:?}", endpoint, protocols);
                    return Ok(protocols);
                }
                Err(ProcessError::ZmqError(ref err_str)) if err_str.contains("Connection refused") || err_str.contains("No such file") => {
                    if retry_count < max_retries {
                        warn!("Backend not ready yet, retrying in {}ms (attempt {}/{})", delay_ms, retry_count + 1, max_retries + 1);
                        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                        retry_count += 1;
                        delay_ms = std::cmp::min(delay_ms * 2, 5000); // Exponential backoff, max 5s
                    } else {
                        warn!("Failed to discover protocols for endpoint {} after {} attempts: {}. Using fallback protocols.", endpoint, max_retries + 1, err_str);
                        return Err(ProcessError::ZmqError(err_str.clone()));
                    }
                }
                Err(e) => {
                    warn!("Protocol discovery failed for {}: {}. Using fallback protocols.", endpoint, e);
                    return Err(e);
                }
            }
        }
    }
    
    /// Single attempt to discover protocols using DEALER socket
    async fn try_discover_protocols(&self, endpoint: &str) -> Result<Vec<String>, ProcessError> {
        let mut socket = DealerSocket::new();
        
        // Connect to the backend
        socket.connect(endpoint).await
            .map_err(|e| ProcessError::ZmqError(format!("Failed to connect to {}: {}", endpoint, e)))?;
        
        // Create handshake request
        let request = handshake::Request {};
        let request_bytes = request.encode_to_vec();
        
        debug!("Sending handshake request ({} bytes) to {}", request_bytes.len(), endpoint);
        
        // Send handshake request using ZmqMessage
        let zmq_request = ZmqMessage::from(request_bytes);
        socket.send(zmq_request).await
            .map_err(|e| ProcessError::ZmqError(format!("Failed to send handshake to {}: {}", endpoint, e)))?;
        
        debug!("Waiting for handshake response from {}", endpoint);
        
        // Receive response with timeout
        let timeout_duration = Duration::from_secs(3);
        let zmq_response = tokio::time::timeout(timeout_duration, socket.recv()).await
            .map_err(|_| ProcessError::ZmqError(format!("Timeout waiting for response from {}", endpoint)))?
            .map_err(|e| ProcessError::ZmqError(format!("Failed to receive response from {}: {}", endpoint, e)))?;
        
        // Extract the response bytes
        let response_frame = zmq_response.get(0)
            .ok_or_else(|| ProcessError::ProtocolError("Empty response from backend".to_string()))?;
        
        debug!("Received handshake response ({} bytes) from {}", response_frame.len(), endpoint);
        
        // Parse the protobuf response
        let response = handshake::Response::decode(&response_frame[..])
            .map_err(|e| ProcessError::ProtocolError(format!("Failed to decode handshake response: {}", e)))?;
        
        debug!("Successfully parsed protocols from backend {}: {:?}", endpoint, response.protocols);
        Ok(response.protocols)
    }
}
