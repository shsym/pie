use crate::types::ModelInstance;
use crate::error::{ProcessError, ProcessResult};
use crate::config::Config;
use std::path::{Path, PathBuf};
use tokio::process::Command;
use tracing::{info, warn, error, debug};
use uuid::Uuid;

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
                let backend_path = parent.join("../../backend/backend-flashinfer");
                if backend_path.exists() {
                    return backend_path;
                }
            }
        }

        // Strategy 2: Look relative to current working directory
        let cwd_backend = std::env::current_dir()
            .unwrap_or_default()
            .join("backend/backend-flashinfer");
        
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
                
                let potential_backend = current_path.join("backend/backend-flashinfer");
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
        warn!("Could not find backend-flashinfer directory, using fallback");
        PathBuf::from("backend/backend-flashinfer")
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
            .arg("--model")
            .arg(model_name)
            .arg("--endpoint")
            .arg(&endpoint);

        // Add config path if provided
        if let Some(config) = config_path {
            command.arg("--config").arg(config);
        }

        // Set environment variables
        command
            .env("PYTHONPATH", &self.backend_base_path)
            .env("SYMPHONY_MODEL_NAME", model_name)
            .env("SYMPHONY_ENDPOINT", &endpoint);

        info!("Spawning backend process for model '{}' with endpoint '{}'", model_name, endpoint);
        debug!("Command: {:?}", command);

        // Spawn the process
        let process = command.spawn()
            .map_err(|e| ProcessError::SpawnFailed(format!("Failed to spawn {}: {}", script_path.display(), e)))?;

        let instance = ModelInstance::new(
            model_name.to_string(),
            model_type.to_string(),
            endpoint,
            process,
            config_path.map(|p| p.to_path_buf()),
        );

        info!("Successfully spawned model instance: {} (PID: {:?})", model_name, instance.pid());
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
}
