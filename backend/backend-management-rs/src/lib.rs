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

// Re-export commonly used types
pub use config::Config;
pub use error::{ManagementError, Result};
pub use service::{ManagementServiceTrait, ServiceStatus};
pub use types::{ManagementCommand, ManagementResponse, ModelInstance, ModelInstanceStatus};

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::{mpsc, RwLock};
use tracing::{info, warn, error, debug};
use uuid::Uuid;

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

        // Create process manager
        let process_manager = process_manager::ProcessManager::new(backend_path, config.clone());

        // Create ZMQ handler
        let zmq_handler = zmq_handler::ZmqHandler::new(
            config.endpoints.client_handshake.clone(),
            config.endpoints.cli_management.clone(),
        );

        Ok(Self {
            config,
            process_manager,
            model_instances: Arc::new(RwLock::new(HashMap::new())),
            zmq_handler: Some(zmq_handler),
            is_running: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            command_tx: None,
            shutdown_tx: None,
            started_at: SystemTime::now(),
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
            zmq_handler.init()?;
        }

        // Create command channel for CLI communication
        let (command_tx, mut command_rx) = mpsc::channel::<(ManagementCommand, mpsc::Sender<ManagementResponse>)>(100);
        self.command_tx = Some(command_tx.clone());

        // Create shutdown channel
        let (shutdown_tx, shutdown_rx) = mpsc::channel::<()>(1);
        self.shutdown_tx = Some(shutdown_tx);

        // Start ZMQ handler in background
        // TODO: Fix ZMQ Send issue - ZMQ sockets are not Send/Sync
        /*
        if let Some(mut zmq_handler) = self.zmq_handler.take() {
            let is_running = self.is_running.clone();
            tokio::spawn(async move {
                if let Err(e) = zmq_handler.run(command_tx, shutdown_rx).await {
                    error!("ZMQ handler error: {}", e);
                }
                is_running.store(false, std::sync::atomic::Ordering::SeqCst);
            });
        }
        */

        // Start command processing loop
        let process_manager = self.process_manager.clone();
        let model_instances = Arc::clone(&self.model_instances);
        let is_running = self.is_running.clone();
        
        tokio::spawn(async move {
            while is_running.load(std::sync::atomic::Ordering::SeqCst) {
                if let Some((command, response_tx)) = command_rx.recv().await {
                    let response = Self::process_command(command, &process_manager, &model_instances).await;
                    let _ = response_tx.send(response).await;
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

        // Terminate all running model instances
        let mut instances = self.model_instances.write().await;
        for (model_name, mut instance) in instances.drain() {
            info!("Terminating model instance: {}", model_name);
            if let Err(e) = instance.terminate().await {
                warn!("Failed to terminate {}: {}", model_name, e);
            }
        }

        info!("Symphony Management Service stopped");
        Ok(())
    }

    fn is_running(&self) -> bool {
        self.is_running.load(std::sync::atomic::Ordering::SeqCst)
    }

    async fn get_status(&mut self) -> Result<service::ServiceStatus> {
        let instances = self.model_instances.read().await;
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
            let instances = self.model_instances.read().await;
            if instances.contains_key(model_name) {
                return Err(ManagementError::Model {
                    message: format!("Model '{}' is already loaded", model_name),
                });
            }
        }

        // Spawn new instance
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
        };

        // Add to registry
        {
            let mut instances = self.model_instances.write().await;
            instances.insert(model_name.to_string(), instance);
        }

        info!("Successfully loaded model: {}", model_name);
        Ok(status)
    }

    async fn unload_model(&mut self, model_name: &str) -> Result<()> {
        info!("Unloading model: {}", model_name);

        let mut instances = self.model_instances.write().await;
        
        if let Some(mut instance) = instances.remove(model_name) {
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
        let instances = self.model_instances.read().await;
        
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
            })
        } else {
            Err(ManagementError::Model {
                message: format!("Model '{}' is not loaded", model_name),
            })
        }
    }

    async fn list_models(&mut self) -> Result<Vec<ModelInstanceStatus>> {
        let instances = self.model_instances.read().await;
        
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
            }
        }).collect();

        Ok(models)
    }

    async fn handle_command(&mut self, command: ManagementCommand) -> Result<ManagementResponse> {
        // Process the command using the existing process_command method
        let process_manager = &self.process_manager;
        let model_instances = Arc::clone(&self.model_instances);
        
        Ok(Self::process_command(command, process_manager, &model_instances).await)
    }

    async fn handle_client_handshake(&mut self, request: &[u8]) -> Result<Vec<u8>> {
        use crate::proto::handshake;
        use prost::Message;

        // Parse the handshake request
        let handshake_request = handshake::Request::decode(request)
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
        // Reuse the list_models implementation for health check
        self.list_models().await
    }
}

impl ManagementServiceImpl {
    /// Process a management command
    async fn process_command(
        command: ManagementCommand,
        process_manager: &process_manager::ProcessManager,
        model_instances: &Arc<RwLock<HashMap<String, ModelInstance>>>,
    ) -> ManagementResponse {
        debug!("Processing command: {}", command.command);

        match command.command.as_str() {
            "status" => {
                let instances = model_instances.read().await;
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
                    let instances = model_instances.read().await;
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
                            let mut instances = model_instances.write().await;
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

                let mut instances = model_instances.write().await;
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
                let instances = model_instances.read().await;
                let models: Vec<_> = instances.iter().map(|(name, instance)| {
                    serde_json::json!({
                        "model_name": name,
                        "model_type": instance.model_type,
                        "endpoint": instance.endpoint,
                        "pid": instance.pid(),
                        "uptime": instance.uptime().as_secs(),
                        "is_alive": instance.process.id().is_some(),
                    })
                }).collect();

                ManagementResponse::success(
                    command.correlation_id.clone(),
                    Some(serde_json::json!({
                        "models": models
                    }))
                )
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
}
