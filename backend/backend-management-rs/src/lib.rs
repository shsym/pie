//! Symphony Management Service - Rust Implementation
//!
//! This crate provides a Rust implementation of the Symphony Management Service,
//! which manages backend model instances and handles client handshakes.

pub mod config;
pub mod error;
pub mod service;
pub mod types;
pub mod process_manager;
pub mod zmq_handler;
pub mod proto;
pub mod cli;
pub mod utils;
pub mod model_installer;

// Re-export commonly used types
pub use config::Config;
pub use error::{ManagementError, Result};
pub use service::{ManagementServiceTrait, ServiceStatus};
pub use types::{ManagementCommand, ManagementResponse, ModelInstance, ModelInstanceStatus};
pub use utils::{cleanup_ipc_socket, cleanup_all_symphony_sockets};
pub use model_installer::{ModelInstaller, ModelInfo};

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};
use tokio::sync::mpsc;
use tracing::{info, warn, error, debug};

/// Main management service implementation
#[derive(Debug)]
pub struct ManagementServiceImpl {
    /// Configuration
    config: Config,
    /// Process manager for spawning backends
    process_manager: process_manager::ProcessManager,
    /// Registry of running model instances
    model_instances: Arc<RwLock<HashMap<String, ModelInstance>>>,
    /// ZMQ handler for client and CLI communication
    zmq_handler: Option<zmq_handler::ZmqHandler>,
    /// Service running state
    is_running: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Service start time
    started_at: SystemTime,
    /// Command channel for CLI communication
    command_tx: Option<mpsc::Sender<(ManagementCommand, mpsc::Sender<ManagementResponse>)>>,
    /// Shutdown signal sender
    shutdown_tx: Option<mpsc::Sender<()>>,
    /// Model installer for downloading models from HuggingFace
    model_installer: model_installer::ModelInstaller,
}

impl service::ManagementServiceFactory for ManagementServiceImpl {
    type Service = Self;

    fn create(
        config_path: Option<PathBuf>,
        backend_base_path: Option<PathBuf>,
    ) -> Result<Self::Service> {
        // Load configuration
        let config = if let Some(path) = config_path {
            Config::load(path)?
        } else {
            Config::load_default()?
        };

        // Determine backend base path
        let backend_path = backend_base_path
            .unwrap_or_else(|| process_manager::ProcessManager::get_default_backend_path());

        info!("Using backend path: {}", backend_path.display());

        // Create model instances registry
        let model_instances = Arc::new(RwLock::new(HashMap::new()));

        // Create process manager
        let process_manager = process_manager::ProcessManager::new(backend_path, config.clone());

        // Create ZMQ handler
        let zmq_handler = zmq_handler::ZmqHandler::new(
            config.endpoints.client_handshake.clone(),
            config.endpoints.cli_management.clone(),
            model_instances.clone(),
        );

        Ok(Self {
            config,
            process_manager,
            model_instances,
            zmq_handler: Some(zmq_handler),
            is_running: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            command_tx: None,
            shutdown_tx: None,
            started_at: SystemTime::now(),
            model_installer: model_installer::ModelInstaller::new(None),
        })
    }
}

#[async_trait::async_trait]
impl service::ManagementServiceTrait for ManagementServiceImpl {
    async fn start(&mut self) -> Result<()> {
        info!("Starting Symphony Management Service");

        // Mark as running
        self.is_running.store(true, std::sync::atomic::Ordering::SeqCst);

        // Initialize ZMQ handler
        if let Some(ref mut zmq_handler) = self.zmq_handler {
            zmq_handler.init().await?;
        }

        // Create command channel for CLI communication
        let (command_tx, mut command_rx) = mpsc::channel::<(ManagementCommand, mpsc::Sender<ManagementResponse>)>(100);
        self.command_tx = Some(command_tx.clone());

        // Create shutdown channel
        let (shutdown_tx, _shutdown_rx) = mpsc::channel::<()>(1);
        let shutdown_tx_clone = shutdown_tx.clone();
        let shutdown_tx_for_commands = shutdown_tx.clone();
        self.shutdown_tx = Some(shutdown_tx);

        // Start ZMQ handler in background
        // Use a dedicated task for ZMQ since sockets are not Send
        if let Some(mut zmq_handler) = self.zmq_handler.take() {
            let command_tx_clone = command_tx.clone();
            let _is_running_clone = self.is_running.clone();
            
            // Spawn ZMQ handler in a blocking task since ZMQ is not async-friendly
            std::thread::spawn(move || {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(async move {
                    // Create a shutdown receiver for this thread
                    let (_tx, rx) = mpsc::channel::<()>(1);
                    
                    // Run ZMQ handler with proper error handling
                    if let Err(e) = zmq_handler.run(command_tx_clone, rx).await {
                        error!("ZMQ handler error: {}", e);
                    }
                });
            });
        }

        // Set up signal handling for graceful shutdown
        let is_running_signal = self.is_running.clone();
        
        tokio::spawn(async move {
            Self::setup_signal_handlers(is_running_signal, shutdown_tx_clone).await;
        });

        // Start command processing loop
        let process_manager = self.process_manager.clone();
        let model_instances = Arc::clone(&self.model_instances);
        let model_installer = self.model_installer.clone();
        let is_running = self.is_running.clone();
        
        tokio::spawn(async move {
            while is_running.load(std::sync::atomic::Ordering::SeqCst) {
                if let Some((command, response_tx)) = command_rx.recv().await {
                    // Handle stop-service command specially
                    if command.command == "stop-service" {
                        info!("Received stop-service command, initiating graceful shutdown");
                        let response = ManagementResponse::success(
                            command.correlation_id.clone(),
                            Some(serde_json::json!({
                                "message": "Service shutdown initiated"
                            }))
                        );
                        let _ = response_tx.send(response).await;
                        
                        // Trigger shutdown
                        is_running.store(false, std::sync::atomic::Ordering::SeqCst);
                        let _ = shutdown_tx_for_commands.send(()).await;
                    } else {
                        let response = Self::process_command(command, &process_manager, &model_instances, &model_installer).await;
                        let _ = response_tx.send(response).await;
                    }
                } else {
                    break;
                }
            }
        });

        info!("Symphony Management Service started successfully");
        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        info!("Stopping Symphony Management Service");

        // Mark as not running
        self.is_running.store(false, std::sync::atomic::Ordering::SeqCst);

        // Send shutdown signal
        if let Some(shutdown_tx) = self.shutdown_tx.take() {
            let _ = shutdown_tx.send(()).await;
        }

        // Extract all running model instances
        let instances: Vec<(String, ModelInstance)> = {
            let mut instances = self.model_instances.write().unwrap();
            instances.drain().collect()
        };

        // Terminate all model instances
        for (model_name, mut instance) in instances {
            info!("Terminating model instance: {}", model_name);
            if let Err(e) = instance.terminate().await {
                warn!("Failed to terminate {}: {}", model_name, e);
            }
        }

        // Clean up CLI management IPC sockets
        cleanup_ipc_socket(&self.config.endpoints.client_handshake);
        cleanup_ipc_socket(&self.config.endpoints.cli_management);
        
        // Clean up any remaining Symphony IPC sockets
        cleanup_all_symphony_sockets();

        info!("Symphony Management Service stopped");
        Ok(())
    }

    fn is_running(&self) -> bool {
        self.is_running.load(std::sync::atomic::Ordering::SeqCst)
    }

    async fn get_status(&mut self) -> Result<service::ServiceStatus> {
        let instances = self.model_instances.read().unwrap();
        let model_count = instances.len();
        
        let models: Vec<_> = instances.iter().map(|(name, instance)| {
            ModelInstanceStatus {
                model_name: name.clone(),
                model_type: instance.model_type.clone(),
                endpoint: instance.endpoint.clone(),
                pid: instance.pid(),
                started_at: instance.started_at,
                uptime: instance.uptime(),
                is_alive: instance.process.id().is_some(),
                config_path: instance.config_path.clone(),
                supported_protocols: instance.supported_protocols.clone(),
            }
        }).collect();

        Ok(ServiceStatus {
            is_running: self.is_running(),
            active_models: model_count,
            uptime_seconds: self.started_at.elapsed().unwrap_or(Duration::ZERO).as_secs(),
            config: self.config.clone(),
            model_instances: models,
            endpoints: service::ServiceEndpoints {
                client_handshake: self.config.endpoints.client_handshake.clone(),
                cli_management: self.config.endpoints.cli_management.clone(),
            },
        })
    }

    async fn load_model(
        &mut self,
        model_name: &str,
        config_path: Option<PathBuf>,
    ) -> Result<ModelInstanceStatus> {
        info!("Loading model: {}", model_name);

        // Check if model is already loaded
        {
            let instances = self.model_instances.read().unwrap();
            if instances.contains_key(model_name) {
                return Err(ManagementError::Model {
                    message: format!("Model '{}' is already loaded", model_name),
                });
            }
        }

        // Spawn new instance (protocols are discovered inside spawn_model_instance)
        let instance = self.process_manager
            .spawn_model_instance(model_name, config_path.as_deref())
            .await?;

        let status = ModelInstanceStatus {
            model_name: instance.model_name.clone(),
            model_type: instance.model_type.clone(),
            endpoint: instance.endpoint.clone(),
            pid: instance.pid(),
            started_at: instance.started_at,
            uptime: instance.uptime(),
            is_alive: instance.process.id().is_some(),
            config_path: instance.config_path.clone(),
            supported_protocols: instance.supported_protocols.clone(),
        };

        // Add to registry
        {
            let mut instances = self.model_instances.write().unwrap();
            instances.insert(model_name.to_string(), instance);
        }

        info!("Successfully loaded model: {}", model_name);
        Ok(status)
    }

    async fn unload_model(&mut self, model_name: &str) -> Result<()> {
        info!("Unloading model: {}", model_name);

        let instance_opt = {
            let mut instances = self.model_instances.write().unwrap();
            instances.remove(model_name)
        };
        
        if let Some(mut instance) = instance_opt {
            instance.terminate().await?;
            info!("Successfully unloaded model: {}", model_name);
            Ok(())
        } else {
            Err(ManagementError::Model {
                message: format!("Model '{}' is not loaded", model_name),
            })
        }
    }

    async fn get_model_status(&mut self, model_name: &str) -> Result<ModelInstanceStatus> {
        let instances = self.model_instances.read().unwrap();
        
        if let Some(instance) = instances.get(model_name) {
            Ok(ModelInstanceStatus {
                model_name: instance.model_name.clone(),
                model_type: instance.model_type.clone(),
                endpoint: instance.endpoint.clone(),
                pid: instance.pid(),
                started_at: instance.started_at,
                uptime: instance.uptime(),
                is_alive: instance.process.id().is_some(),
                config_path: instance.config_path.clone(),
                supported_protocols: instance.supported_protocols.clone(),
            })
        } else {
            Err(ManagementError::Model {
                message: format!("Model '{}' is not loaded", model_name),
            })
        }
    }

    async fn list_models(&mut self) -> Result<Vec<ModelInstanceStatus>> {
        let instances = self.model_instances.read().unwrap();
        
        let models = instances.iter().map(|(_, instance)| {
            ModelInstanceStatus {
                model_name: instance.model_name.clone(),
                model_type: instance.model_type.clone(),
                endpoint: instance.endpoint.clone(),
                pid: instance.pid(),
                started_at: instance.started_at,
                uptime: instance.uptime(),
                is_alive: instance.process.id().is_some(),
                config_path: instance.config_path.clone(),
                supported_protocols: instance.supported_protocols.clone(),
            }
        }).collect();

        Ok(models)
    }

    async fn handle_command(&mut self, command: ManagementCommand) -> Result<ManagementResponse> {
        // Process the command using the existing process_command method
        let process_manager = &self.process_manager;
        let model_instances = Arc::clone(&self.model_instances);
        
        Ok(Self::process_command(command, process_manager, &model_instances, &self.model_installer).await)
    }

    async fn handle_client_handshake(&mut self, request: &[u8]) -> Result<Vec<u8>> {
        use crate::proto::handshake;
        use prost::Message;

        // Parse the handshake request
        let _handshake_request = handshake::Request::decode(request)
            .map_err(|e| ManagementError::Protocol {
                message: format!("Failed to decode handshake request: {}", e),
            })?;

        info!("Received handshake from client");

        // Create response
        let response = handshake::Response {
            protocols: vec!["symphony-management-v1".to_string()],
        };

        // Encode response
        let mut response_buf = Vec::new();
        response.encode(&mut response_buf)
            .map_err(|e| ManagementError::Protocol {
                message: format!("Failed to encode handshake response: {}", e),
            })?;

        Ok(response_buf)
    }

    async fn health_check(&mut self) -> Result<Vec<ModelInstanceStatus>> {
        info!("Performing health check on all model instances");
        
        let mut instances = self.model_instances.write().unwrap();
        let mut healthy_instances = Vec::new();
        let mut instances_to_remove = Vec::new();

        for (model_name, instance) in instances.iter_mut() {
            // Check if process is still alive
            let is_alive = match instance.process.try_wait() {
                Ok(Some(_)) => {
                    // Process has exited
                    warn!("Model instance '{}' has unexpectedly exited", model_name);
                    false
                }
                Ok(None) => {
                    // Process is still running
                    true
                }
                Err(e) => {
                    // Error checking status
                    warn!("Error checking status of '{}': {}", model_name, e);
                    false
                }
            };

            if is_alive {
                healthy_instances.push(ModelInstanceStatus {
                    model_name: instance.model_name.clone(),
                    model_type: instance.model_type.clone(),
                    endpoint: instance.endpoint.clone(),
                    pid: instance.pid(),
                    started_at: instance.started_at,
                    uptime: instance.uptime(),
                    is_alive: true,
                    config_path: instance.config_path.clone(),
                    supported_protocols: instance.supported_protocols.clone(),
                });
            } else {
                // Mark for removal
                instances_to_remove.push(model_name.clone());
            }
        }

        // Remove dead instances from registry
        for model_name in instances_to_remove {
            instances.remove(&model_name);
            warn!("Removed dead model instance '{}' from registry", model_name);
        }

        info!("Health check complete: {}/{} instances healthy", 
              healthy_instances.len(), instances.len());
        
        Ok(healthy_instances)
    }
}

impl ManagementServiceImpl {
    /// Set up signal handlers for graceful shutdown
    async fn setup_signal_handlers(
        is_running: Arc<std::sync::atomic::AtomicBool>, 
        shutdown_tx: mpsc::Sender<()>
    ) {
        use std::sync::atomic::Ordering;
        
        #[cfg(unix)]
        {
            use tokio::signal::unix::{signal, SignalKind};
            
            let mut sigterm = signal(SignalKind::terminate()).expect("Failed to set up SIGTERM handler");
            let mut sigint = signal(SignalKind::interrupt()).expect("Failed to set up SIGINT handler");
            
            tokio::select! {
                _ = sigterm.recv() => {
                    info!("Received SIGTERM, initiating graceful shutdown");
                    is_running.store(false, Ordering::SeqCst);
                    let _ = shutdown_tx.send(()).await;
                }
                _ = sigint.recv() => {
                    info!("Received SIGINT (Ctrl+C), initiating graceful shutdown");
                    is_running.store(false, Ordering::SeqCst);
                    let _ = shutdown_tx.send(()).await;
                }
            }
        }
        
        #[cfg(windows)]
        {
            use tokio::signal::windows::{ctrl_c, ctrl_break};
            
            let mut ctrl_c = ctrl_c().expect("Failed to set up Ctrl+C handler");
            let mut ctrl_break = ctrl_break().expect("Failed to set up Ctrl+Break handler");
            
            tokio::select! {
                _ = ctrl_c.recv() => {
                    info!("Received Ctrl+C, initiating graceful shutdown");
                    is_running.store(false, Ordering::SeqCst);
                    let _ = shutdown_tx.send(()).await;
                }
                _ = ctrl_break.recv() => {
                    info!("Received Ctrl+Break, initiating graceful shutdown");
                    is_running.store(false, Ordering::SeqCst);
                    let _ = shutdown_tx.send(()).await;
                }
            }
        }
    }

    /// Process a management command
    async fn process_command(
        command: ManagementCommand,
        process_manager: &process_manager::ProcessManager,
        model_instances: &Arc<RwLock<HashMap<String, ModelInstance>>>,
        model_installer: &model_installer::ModelInstaller,
    ) -> ManagementResponse {
        debug!("Processing command: {}", command.command);

        match command.command.as_str() {
            "status" => {
                let instances = model_instances.read().unwrap();
                let model_count = instances.len();
                let mut models = Vec::new();

                for (name, instance) in instances.iter() {
                    models.push(serde_json::json!({
                        "model_name": name,
                        "model_type": instance.model_type,
                        "endpoint": instance.endpoint,
                        "pid": instance.pid(),
                        "uptime": instance.uptime().as_secs(),
                        "is_alive": instance.process.id().is_some(),
                    }));
                }

                ManagementResponse::success(
                    command.correlation_id.clone(),
                    Some(serde_json::json!({
                        "is_running": true,
                        "model_count": model_count,
                        "models": models,
                    }))
                )
            }
            "load-model" => {
                let model_name = command.params.get("model_name")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();

                if model_name.is_empty() {
                    return ManagementResponse::error(
                        command.correlation_id.clone(),
                        "Missing model_name parameter".to_string()
                    );
                }

                // Check if already loaded
                {
                    let instances = model_instances.read().unwrap();
                    if instances.contains_key(model_name) {
                        return ManagementResponse::error(
                            command.correlation_id.clone(),
                            format!("Model '{}' is already loaded", model_name)
                        );
                    }
                }

                // Get config path if provided
                let config_path = command.params.get("config_path")
                    .and_then(|v| v.as_str())
                    .map(PathBuf::from);

                // Spawn instance
                match process_manager.spawn_model_instance(model_name, config_path.as_deref()).await {
                    Ok(instance) => {
                        let response_data = serde_json::json!({
                            "model_name": instance.model_name,
                            "model_type": instance.model_type,
                            "endpoint": instance.endpoint,
                            "pid": instance.pid(),
                        });

                        // Add to registry
                        {
                            let mut instances = model_instances.write().unwrap();
                            instances.insert(model_name.to_string(), instance);
                        }

                        ManagementResponse::success(
                            command.correlation_id.clone(),
                            Some(response_data)
                        )
                    }
                    Err(e) => ManagementResponse::error(
                        command.correlation_id.clone(),
                        format!("Failed to load model: {}", e)
                    ),
                }
            }
            "unload-model" => {
                let model_name = command.params.get("model_name")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();

                if model_name.is_empty() {
                    return ManagementResponse::error(
                        command.correlation_id.clone(),
                        "Missing model_name parameter".to_string()
                    );
                }

                let mut instances = model_instances.write().unwrap();
                if let Some(mut instance) = instances.remove(&model_name) {
                    let model_name_clone = model_name.clone();
                    // Terminate in background to avoid blocking
                    tokio::spawn(async move {
                        if let Err(e) = instance.terminate().await {
                            error!("Failed to terminate {}: {}", model_name_clone, e);
                        }
                    });
                    ManagementResponse::success(
                        command.correlation_id.clone(),
                        Some(serde_json::json!({
                            "message": format!("Model '{}' unloaded successfully", model_name)
                        }))
                    )
                } else {
                    ManagementResponse::error(
                        command.correlation_id.clone(),
                        format!("Model '{}' is not loaded", model_name)
                    )
                }
            }
            "list-models" => {
                // Get loaded models
                let loaded_models: Vec<_> = {
                    let instances = model_instances.read().unwrap();
                    instances.iter().map(|(name, instance)| {
                        serde_json::json!({
                            "model_name": name,
                            "model_type": instance.model_type,
                            "endpoint": instance.endpoint,
                            "pid": instance.pid(),
                            "uptime": instance.uptime().as_secs(),
                            "is_alive": instance.process.id().is_some(),
                        })
                    }).collect()
                }; // Drop the guard here

                // Get installed models
                let installed_models = match model_installer.list_installed_models().await {
                    Ok(models) => models.into_iter().map(|model_info| {
                        serde_json::json!({
                            "model_name": model_info.model_name,
                            "local_name": model_info.local_name,
                            "model_type": model_info.model_type,
                            "path": model_info.path,
                            "tokenizer_path": model_info.tokenizer_path,
                            "installed_at": model_info.installed_at,
                            "architectures": model_info.architectures
                        })
                    }).collect::<Vec<_>>(),
                    Err(e) => {
                        warn!("Failed to list installed models: {}", e);
                        Vec::new()
                    }
                };

                ManagementResponse::success(
                    command.correlation_id.clone(),
                    Some(serde_json::json!({
                        "loaded_models": loaded_models,
                        "installed_models": installed_models
                    }))
                )
            }
            "install-model" => {
                let model_name = command.params.get("model_name")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();

                if model_name.is_empty() {
                    return ManagementResponse::error(
                        command.correlation_id.clone(),
                        "Missing model_name parameter".to_string()
                    );
                }

                let local_name = command.params.get("local_name")
                    .and_then(|v| v.as_str())
                    .unwrap_or_else(|| {
                        // Extract model name from HF path (e.g., "meta-llama/Llama-3.1-8B" -> "Llama-3.1-8B")
                        model_name.split('/').last().unwrap_or(model_name)
                    });

                let force = command.params.get("force")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);

                // Install the model using HuggingFace transformers
                match Self::install_model_from_hf(model_installer, model_name, local_name, force).await {
                    Ok(installation_info) => {
                        ManagementResponse::success(
                            command.correlation_id.clone(),
                            Some(installation_info)
                        )
                    }
                    Err(e) => {
                        ManagementResponse::error(
                            command.correlation_id.clone(),
                            format!("Failed to install model: {}", e)
                        )
                    }
                }
            }
            "uninstall-model" => {
                let model_name = match command.params.get("model_name").and_then(|v| v.as_str()) {
                    Some(name) => name,
                    None => return ManagementResponse::error(
                        command.correlation_id.clone(),
                        "Missing required parameter: model_name".to_string()
                    ),
                };

                let force = command.params.get("force")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);

                // Uninstall the model
                match Self::uninstall_model_from_storage(model_installer, model_name, force).await {
                    Ok(uninstall_info) => {
                        ManagementResponse::success(
                            command.correlation_id.clone(),
                            Some(uninstall_info)
                        )
                    }
                    Err(e) => {
                        ManagementResponse::error(
                            command.correlation_id.clone(),
                            format!("Failed to uninstall model: {}", e)
                        )
                    }
                }
            }
            _ => {
                warn!("Unknown command: {}", command.command);
                ManagementResponse::error(
                    command.correlation_id.clone(),
                    format!("Unknown command: {}", command.command)
                )
            }
        }
    }

    /// Get model type for a model name (from service implementation)
    pub fn get_model_type(&self, model_name: &str) -> Result<String> {
        self.config.get_model_type(model_name)
            .map(|s| s.to_string())
            .ok_or_else(|| {
                let available_models = self.config.get_supported_models();
                let suggestion = if available_models.is_empty() {
                    "No models are configured".to_string()
                } else {
                    format!("Available models: {}", available_models.join(", "))
                };
                ManagementError::UnknownModel(format!("{} ({})", model_name, suggestion))
            })
    }

    /// Generate a unique endpoint
    pub fn generate_unique_endpoint(&self) -> String {
        self.process_manager.generate_unique_endpoint()
    }

    /// Install a model from HuggingFace Hub using transformers
    async fn install_model_from_hf(
        model_installer: &model_installer::ModelInstaller,
        model_name: &str, 
        local_name: &str, 
        force: bool
    ) -> Result<serde_json::Value> {
        info!("Installing model '{}' from HuggingFace Hub as '{}'", model_name, local_name);
        
        // Check if model is already installed (unless force is true)
        if !force && model_installer.is_model_installed(model_name).await {
            let model_info = model_installer.get_model_info(model_name).await?;
            return Ok(serde_json::json!({
                "status": "already_installed",
                "model_name": model_name,
                "local_name": local_name,
                "path": model_info.path,
                "model_type": model_info.model_type
            }));
        }
        
        // Install the model
        let model_path = model_installer.install_model(model_name).await?;
        let model_info = model_installer.get_model_info(model_name).await?;
        
        info!("Successfully installed model '{}' to {:?}", model_name, model_path);
        
        Ok(serde_json::json!({
            "status": "installed",
            "model_name": model_name,
            "local_name": local_name,
            "path": model_path,
            "model_type": model_info.model_type,
            "architectures": model_info.architectures
        }))
    }

    /// Uninstall a model from local storage
    async fn uninstall_model_from_storage(
        model_installer: &model_installer::ModelInstaller,
        model_name: &str,
        _force: bool
    ) -> Result<serde_json::Value> {
        info!("Uninstalling model '{}' from local storage", model_name);
        
        // Resolve the model name (in case user provided local name instead of original name)
        let resolved_name = model_installer.resolve_model_name(model_name).await?;
        
        // Check if model is installed
        if !model_installer.is_model_installed(&resolved_name).await {
            return Ok(serde_json::json!({
                "status": "not_found",
                "model_name": model_name,
                "message": format!("Model '{}' is not installed", model_name)
            }));
        }
        
        // TODO: Check if model is currently loaded and handle force flag
        // For now, we'll proceed with uninstallation
        
        // Uninstall the model using the resolved name
        let removed_path = model_installer.uninstall_model(&resolved_name).await?;
        
        info!("Successfully uninstalled model '{}' (resolved as '{}') from {:?}", model_name, resolved_name, removed_path);
        
        Ok(serde_json::json!({
            "status": "uninstalled",
            "model_name": model_name,
            "resolved_name": resolved_name,
            "path": removed_path,
            "message": format!("Model '{}' uninstalled successfully", model_name)
        }))
    }
}

impl Drop for ManagementServiceImpl {
    fn drop(&mut self) {
        // Perform emergency cleanup on unexpected shutdown
        info!("ManagementServiceImpl dropping, performing emergency cleanup");
        
        // Clean up CLI management IPC sockets
        cleanup_ipc_socket(&self.config.endpoints.client_handshake);
        cleanup_ipc_socket(&self.config.endpoints.cli_management);
        
        // Clean up any remaining Symphony IPC sockets
        cleanup_all_symphony_sockets();
    }
}
