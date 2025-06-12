use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Backend information returned from engine-manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendInfo {
    pub backend_id: String,
    pub status: String,
    pub capabilities: Vec<String>,
    pub management_api_address: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub registered_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_heartbeat: Option<String>,
}

/// Backend information summary from engine-manager (matches actual API)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendInfoSummary {
    pub backend_id: String, // Note: engine-manager uses Uuid but we'll convert to String
    pub status: String,
    pub management_api_address: String,
    pub capabilities: Vec<String>,
}

/// Response from engine-manager's /backends endpoint (matches actual API)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListBackendsResponse {
    pub backends: Vec<BackendInfoSummary>,
}

/// HTTP client for communicating with engine-manager
pub struct EngineManagerClient {
    client: reqwest::Client,
    base_url: String,
}

impl EngineManagerClient {
    pub fn new(engine_manager_endpoint: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: engine_manager_endpoint.to_string(),
        }
    }

    /// Query engine-manager for all registered backends
    pub async fn list_backends(&self) -> Result<HashMap<String, BackendInfo>> {
        tracing::info!("Listing backends from engine-manager at: {}", self.base_url);
        let url = format!("{}/backends", self.base_url);
        tracing::debug!("Making GET request to: {}", url);

        let response = self.client
            .get(&url)
            .timeout(std::time::Duration::from_secs(10))
            .send()
            .await
            .context("Failed to send request to engine-manager")?;

        tracing::debug!("Response status: {}", response.status());

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            tracing::error!("Engine-manager returned error status: {} - {}", status, error_text);
            return Err(anyhow!(
                "Engine-manager returned error status: {} - {}",
                status,
                error_text
            ));
        }

        let response_text = response.text().await.context("Failed to read response body")?;
        tracing::debug!("Raw response body: {}", response_text);

        let list_response: ListBackendsResponse = serde_json::from_str(&response_text)
            .context("Failed to parse backends list response")?;

        tracing::info!("Successfully parsed {} backends from engine-manager", list_response.backends.len());

        // Convert Vec<BackendInfoSummary> to HashMap<String, BackendInfo>
        let mut backends_map = HashMap::new();
        for summary in list_response.backends {
            let backend_info = BackendInfo {
                backend_id: summary.backend_id.clone(),
                status: summary.status,
                capabilities: summary.capabilities,
                management_api_address: summary.management_api_address,
                registered_at: None,
                last_heartbeat: None,
            };

            tracing::info!("Backend '{}': status={}, management_api_address={}, capabilities={:?}",
                backend_info.backend_id, backend_info.status, backend_info.management_api_address, backend_info.capabilities);

            backends_map.insert(summary.backend_id, backend_info);
        }

        Ok(backends_map)
    }

    /// Find a compatible backend for the requested model
    pub async fn find_compatible_backend(&self, model_name: &str) -> Result<Option<BackendInfo>> {
        let backends = self.list_backends().await?;

        for (backend_id, backend_info) in backends {
            // Check if this backend is active (not just registered)
            if backend_info.status != "Running" {
                tracing::debug!("Skipping backend '{}' with status '{}' (not Running)", backend_id, backend_info.status);
                continue;
            }

            // Check if backend supports the requested model
            if self.backend_supports_model(&backend_info, model_name) {
                tracing::info!(
                    "Found compatible backend '{}' for model '{}': {}",
                    backend_id,
                    model_name,
                    backend_info.management_api_address
                );
                return Ok(Some(backend_info));
            } else {
                tracing::debug!("Backend '{}' does not support model '{}'", backend_id, model_name);
            }
        }

        tracing::warn!("No compatible backend found for model '{}'", model_name);
        Ok(None)
    }

    /// Check if a backend supports a specific model
    fn backend_supports_model(&self, backend_info: &BackendInfo, model_name: &str) -> bool {
        tracing::debug!("Checking if backend supports model '{}'. Backend capabilities: {:?}", model_name, backend_info.capabilities);

        // Check capabilities for model support
        for capability in &backend_info.capabilities {
            if capability.starts_with("model:") {
                let supported_model = &capability[6..]; // Remove "model:" prefix
                tracing::debug!("Found model capability: '{}' -> '{}'", capability, supported_model);
                if supported_model == model_name {
                    tracing::info!("Exact model match found: '{}' == '{}'", supported_model, model_name);
                    return true;
                }
            }
        }

        // TODO: Add more sophisticated model compatibility checking
        // For now, we'll also check if the capability contains the model name
        let fallback_match = backend_info.capabilities.iter().any(|cap| cap.contains(model_name));
        if fallback_match {
            tracing::info!("Fallback model match found for '{}'", model_name);
        } else {
            tracing::debug!("No model match found for '{}'", model_name);
        }
        fallback_match
    }
}

/// Get the model endpoint from a compatible backend's management API
pub async fn get_model_endpoint_from_backend(
    backend_info: &BackendInfo,
    model_name: &str,
) -> Result<String> {
    let client = reqwest::Client::new();

    // First check if the backend is healthy
    let health_url = format!("{}/manage/health", backend_info.management_api_address);
    tracing::debug!("Checking backend health at: {}", health_url);

    let health_response = client
        .get(&health_url)
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .context("Failed to get health status from backend")?;

    if !health_response.status().is_success() {
        return Err(anyhow!(
            "Backend health check failed: {} - {}",
            health_response.status(),
            health_response.text().await.unwrap_or_else(|_| "Unknown error".to_string())
        ));
    }

    let health_data: serde_json::Value = health_response
        .json()
        .await
        .context("Failed to parse backend health response")?;

    tracing::info!("Backend health check successful: {:?}", health_data);

    // Check if the backend has the model loaded, or if it supports the model
    let models_url = format!("{}/manage/models", backend_info.management_api_address);
    tracing::debug!("Checking backend models at: {}", models_url);

    let models_response = client
        .get(&models_url)
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .context("Failed to get models from backend")?;

    if !models_response.status().is_success() {
        return Err(anyhow!(
            "Backend models request failed: {} - {}",
            models_response.status(),
            models_response.text().await.unwrap_or_else(|_| "Unknown error".to_string())
        ));
    }

    let models_data: serde_json::Value = models_response
        .json()
        .await
        .context("Failed to parse backend models response")?;

    tracing::info!("Backend models data: {:?}", models_data);

    // Check if the model is already loaded
    if let Some(loaded_models) = models_data.get("loaded_models") {
        if let Some(loaded_models_obj) = loaded_models.as_object() {
            if loaded_models_obj.contains_key(model_name) {
                // Model is already loaded, return the IPC endpoint from capabilities
                if let Some(ipc_endpoint) = backend_info.capabilities.iter()
                    .find(|cap| cap.starts_with("ipc_endpoint:"))
                    .map(|cap| &cap[13..]) // Remove "ipc_endpoint:" prefix
                {
                    tracing::info!("Found loaded model '{}' with IPC endpoint: {}", model_name, ipc_endpoint);
                    return Ok(ipc_endpoint.to_string());
                }
            }
        }
    }

    // Check if the model is supported (can be loaded)
    if let Some(supported_models) = models_data.get("supported_models") {
        if let Some(supported_array) = supported_models.as_array() {
            for supported_model in supported_array {
                if let Some(supported_name) = supported_model.as_str() {
                    if supported_name == model_name {
                        // Model is supported but not loaded, we could load it here
                        // For now, return the IPC endpoint from capabilities
                        if let Some(ipc_endpoint) = backend_info.capabilities.iter()
                            .find(|cap| cap.starts_with("ipc_endpoint:"))
                            .map(|cap| &cap[13..]) // Remove "ipc_endpoint:" prefix
                        {
                            tracing::info!("Found supported model '{}' with IPC endpoint: {}", model_name, ipc_endpoint);
                            return Ok(ipc_endpoint.to_string());
                        }
                    }
                }
            }
        }
    }

    Err(anyhow!("Model '{}' not found in backend status", model_name))
}

/// Discover and connect to a backend for the specified model
pub async fn discover_backend_for_model(
    engine_manager_endpoint: &str,
    model_name: &str,
) -> Result<String> {
    let client = EngineManagerClient::new(engine_manager_endpoint);

    // Find a compatible backend
    let backend_info = client
        .find_compatible_backend(model_name)
        .await?
        .ok_or_else(|| {
            anyhow!(
                "No compatible backend found for model '{}'. Please start a backend that supports this model.",
                model_name
            )
        })?;

    // Get the model endpoint from the backend
    get_model_endpoint_from_backend(&backend_info, model_name).await
}
