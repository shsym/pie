//! Command processing implementation for ManagementServiceImpl
//!
//! This module contains the command processing logic separated from the main lib.rs file
//! for better code organization and maintainability.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use tracing::{info, warn, error, debug};

use crate::{ManagementServiceImpl, Result};
use crate::types::{ManagementCommand, ManagementResponse, ModelInstance};
use crate::{process_manager, model_installer};

impl ManagementServiceImpl {
    /// Process management commands
    pub(crate) async fn process_command(
        command: ManagementCommand,
        process_manager: &process_manager::ProcessManager,
        model_instances: &Arc<RwLock<HashMap<String, ModelInstance>>>,
        model_installer: &model_installer::ModelInstaller,
    ) -> ManagementResponse {
        debug!("Processing command: {}", command.command);

        match command.command.as_str() {
            "status" => Self::handle_status_command(command, model_instances).await,
            "load-model" => Self::handle_load_model_command(command, process_manager, model_instances).await,
            "unload-model" => Self::handle_unload_model_command(command, model_instances).await,
            "list-models" => Self::handle_list_models_command(command, model_instances, model_installer).await,
            "install-model" => Self::handle_install_model_command(command, model_installer).await,
            "uninstall-model" => Self::handle_uninstall_model_command(command, model_installer).await,
            _ => {
                warn!("Unknown command: {}", command.command);
                ManagementResponse::error(
                    command.correlation_id.clone(),
                    format!("Unknown command: {}", command.command)
                )
            }
        }
    }

    /// Handle status command
    async fn handle_status_command(
        command: ManagementCommand,
        model_instances: &Arc<RwLock<HashMap<String, ModelInstance>>>,
    ) -> ManagementResponse {
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

    /// Handle load-model command
    async fn handle_load_model_command(
        command: ManagementCommand,
        process_manager: &process_manager::ProcessManager,
        model_instances: &Arc<RwLock<HashMap<String, ModelInstance>>>,
    ) -> ManagementResponse {
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

    /// Handle unload-model command
    async fn handle_unload_model_command(
        command: ManagementCommand,
        model_instances: &Arc<RwLock<HashMap<String, ModelInstance>>>,
    ) -> ManagementResponse {
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

    /// Handle list-models command
    async fn handle_list_models_command(
        command: ManagementCommand,
        model_instances: &Arc<RwLock<HashMap<String, ModelInstance>>>,
        model_installer: &model_installer::ModelInstaller,
    ) -> ManagementResponse {
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

    /// Handle install-model command
    async fn handle_install_model_command(
        command: ManagementCommand,
        model_installer: &model_installer::ModelInstaller,
    ) -> ManagementResponse {
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

    /// Handle uninstall-model command
    async fn handle_uninstall_model_command(
        command: ManagementCommand,
        model_installer: &model_installer::ModelInstaller,
    ) -> ManagementResponse {
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
