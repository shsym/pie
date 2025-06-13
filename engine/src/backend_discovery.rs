use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::RwLock;

/// Result of backend discovery for a model
#[derive(Debug, Clone)]
pub struct BackendDiscoveryResult {
    pub backend_endpoint: String,
    pub backend_id: String,
}

/// Model-to-backend cache: maps model names to backend endpoints and service IDs
#[derive(Debug, Clone)]
pub struct ModelBackendInfo {
    pub backend_endpoint: String,
    pub service_id: Option<usize>, // Set when service is created
    pub backend_id: String,
    pub last_verified: std::time::Instant,
}

static MODEL_BACKEND_CACHE: std::sync::LazyLock<RwLock<HashMap<String, ModelBackendInfo>>> =
    std::sync::LazyLock::new(|| {
        RwLock::new(HashMap::new())
    });

/// Engine manager endpoint for dynamic model discovery
static ENGINE_MANAGER_ENDPOINT: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    std::env::var("ENGINE_MANAGER_ENDPOINT")
        .unwrap_or_else(|_| "http://127.0.0.1:8080".to_string())
});

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

        tracing::debug!("Successfully parsed {} backends from engine-manager", list_response.backends.len());

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

            tracing::debug!("Backend '{}': status={}, management_api_address={}, capabilities={:?}",
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
) -> Result<BackendDiscoveryResult> {
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
    let backend_endpoint = get_model_endpoint_from_backend(&backend_info, model_name).await?;

    Ok(BackendDiscoveryResult {
        backend_endpoint,
        backend_id: backend_info.backend_id,
    })
}

/// Get a model backend info from the cache ONLY (no discovery)
pub fn get_cached_model_backend_info(model_name: &str) -> Option<ModelBackendInfo> {
    if let Ok(cache) = MODEL_BACKEND_CACHE.read() {
        if let Some(info) = cache.get(model_name) {
            tracing::info!("Found cached backend info for model '{}': {}", model_name, info.backend_endpoint);
            return Some(info.clone());
        }
    }
    tracing::debug!("No cached backend info found for model '{}'", model_name);
    None
}

/// Get a model backend info from the cache, or discover it if not cached
/// This function is for internal engine use only, not for bindings
pub async fn get_model_backend_info(model_name: &str) -> anyhow::Result<Option<ModelBackendInfo>> {
    // First check cache
    if let Ok(cache) = MODEL_BACKEND_CACHE.read() {
        if let Some(info) = cache.get(model_name) {
            // Check if cache entry is still fresh (within 5 minutes)
            if info.last_verified.elapsed() < std::time::Duration::from_secs(300) {
                tracing::info!("Found cached backend info for model '{}': {}", model_name, info.backend_endpoint);
                return Ok(Some(info.clone()));
            } else {
                tracing::info!("Cache entry for model '{}' is stale, will refresh", model_name);
            }
        }
    }

    // Not in cache or stale, discover backend for this model
    tracing::info!("Discovering backend for model '{}'", model_name);
    match discover_backend_for_model(&ENGINE_MANAGER_ENDPOINT, model_name).await {
        Ok(discovery_result) => {
            let backend_info = ModelBackendInfo {
                backend_endpoint: discovery_result.backend_endpoint.clone(),
                service_id: None, // Will be set when service is created
                backend_id: discovery_result.backend_id,
                last_verified: std::time::Instant::now(),
            };

            // Update cache
            if let Ok(mut cache) = MODEL_BACKEND_CACHE.write() {
                cache.insert(model_name.to_string(), backend_info.clone());
                tracing::info!("Cached backend info for model '{}': endpoint={}, backend_id={}",
                    model_name, discovery_result.backend_endpoint, backend_info.backend_id);
            }

            Ok(Some(backend_info))
        }
        Err(e) => {
            tracing::warn!("Failed to discover backend for model '{}': {}", model_name, e);
            Ok(None)
        }
    }
}

/// Update the service ID for a model in the cache
pub fn set_model_service_id(model_name: &str, service_id: usize) {
    if let Ok(mut cache) = MODEL_BACKEND_CACHE.write() {
        if let Some(info) = cache.get_mut(model_name) {
            info.service_id = Some(service_id);
            tracing::info!("Updated service ID for model '{}' to {}", model_name, service_id);
        }
    }
}

/// Get the service ID for a model from the cache
pub fn get_model_service_id(model_name: &str) -> Option<usize> {
    if let Ok(cache) = MODEL_BACKEND_CACHE.read() {
        if let Some(info) = cache.get(model_name) {
            return info.service_id;
        }
    }
    None
}

/// Update the model-backend cache when backends are discovered
pub fn update_model_backend_cache_from_discovery(backends: &HashMap<String, BackendInfo>) {
    let mut cache_updates = HashMap::new();

    for (backend_id, backend_info) in backends {
        if backend_info.status != "Running" {
            continue;
        }

        // Extract IPC endpoint from capabilities
        let ipc_endpoint = backend_info.capabilities.iter()
            .find(|cap| cap.starts_with("ipc_endpoint:"))
            .map(|cap| &cap[13..]) // Remove "ipc_endpoint:" prefix
            .unwrap_or("unknown");

        // Extract models from capabilities
        for capability in &backend_info.capabilities {
            if capability.starts_with("model:") {
                let model_name = &capability[6..]; // Remove "model:" prefix

                let backend_info = ModelBackendInfo {
                    backend_endpoint: ipc_endpoint.to_string(),
                    service_id: None, // Services created on demand
                    backend_id: backend_id.clone(),
                    last_verified: std::time::Instant::now(),
                };

                cache_updates.insert(model_name.to_string(), backend_info);
            }
        }
    }

    // Update the cache with discovered models
    if let Ok(mut cache) = MODEL_BACKEND_CACHE.write() {
        for (model_name, backend_info) in cache_updates {
            // Only update if not already cached or if we have a service ID to preserve
            if let Some(existing) = cache.get(&model_name) {
                let mut updated_info = backend_info;
                updated_info.service_id = existing.service_id; // Preserve existing service ID
                cache.insert(model_name.clone(), updated_info);
            } else {
                tracing::info!("Added new model '{}' to cache with backend '{}'", model_name, backend_info.backend_endpoint);
                cache.insert(model_name.clone(), backend_info);
            }
        }
    }
}

/// Start a background task to periodically update the model-backend cache
pub fn start_periodic_cache_updates() {
    let update_interval = std::time::Duration::from_secs(1); // Update every second for low latency

    tokio::spawn(async move {
        let mut interval = tokio::time::interval(update_interval);

        loop {
            interval.tick().await;

            // Update the cache from engine-manager
            match discover_available_models().await {
                Ok(discovered_models) => {
                    if !discovered_models.is_empty() {
                        crate::l4m::set_available_models(&discovered_models);

                        // Create L4M services for newly discovered models
                        // Note: This will only create services for models that don't already have one
                        create_l4m_services_for_discovered_models(&discovered_models).await;

                        tracing::debug!("Periodic cache update: found {} models", discovered_models.len());
                    }
                }
                Err(e) => {
                    tracing::debug!("Periodic cache update failed: {}", e);
                }
            }
        }
    });

    tracing::info!("Started periodic model-backend cache updates (interval: {:?})", update_interval);
}

/// Async function to discover models from all registered backends
/// Note) this function is called periodically to refresh the model list
pub async fn discover_available_models() -> anyhow::Result<Vec<String>> {
    let client = EngineManagerClient::new(&ENGINE_MANAGER_ENDPOINT);
    tracing::debug!("[BackendDiscovery] Created engine manager client for endpoint: {}", *ENGINE_MANAGER_ENDPOINT);

    let backends = match client.list_backends().await {
        Ok(backends) => {
            tracing::debug!("[BackendDiscovery] Raw backends data: {:#?}", backends);
            backends
        }
        Err(e) => {
            tracing::error!("[BackendDiscovery] Failed to list backends from engine manager: {}", e);
            return Err(e);
        }
    };

    // Update the model-backend cache from discovered backends
    update_model_backend_cache_from_discovery(&backends);

    // Start with empty list of models; discover fresh set from registered backends
    // We cannot simply start from the cached models because the backend capabilities may have changed
    let mut models = Vec::new();

    for (backend_id, backend_info) in &backends {
        tracing::debug!("[BackendDiscovery] Backend '{}' capabilities: {:?}", backend_id, backend_info.capabilities);

        if backend_info.status != "Running" {
            tracing::warn!("[BackendDiscovery] Skipping backend '{}' because status is not 'Running': {}", backend_id, backend_info.status);
            continue;
        }

        // Extract models from backend capabilities
        for capability in &backend_info.capabilities {
            tracing::debug!("[BackendDiscovery] Processing capability: '{}'", capability);
            // Check and extract model capability prefix
            if let Some(model_name) = capability.strip_prefix("model:") {
                tracing::debug!("[BackendDiscovery] Found model: '{}'", model_name);
                let model_str = model_name.to_string();
                if !models.contains(&model_str) {
                    models.push(model_str.clone());
                    tracing::debug!("Discovered a new model '{}' from backend '{}'", model_str, backend_id);
                } else {
                    tracing::debug!("Model '{}' already discovered", model_str);
                }
            } else {
                tracing::debug!("[BackendDiscovery] Ignoring non-model capability: '{}'", capability);
            }
        }
    }

    if models.is_empty() {
        tracing::warn!("No models discovered from registered backends (total backends: {})", backends.len());
        if backends.len() > 0 {
            tracing::debug!("Backends were found but no models extracted. Check backend capabilities format.");
        } else {
            tracing::debug!("No backends found at all. Check if backend is registered with engine-manager.");
        }
    } else {
        tracing::debug!("Discovered {} models from backends: {:?}", models.len(), models);
    }

    Ok(models)
}

/// Create L4M services for newly discovered models that don't already have services
async fn create_l4m_services_for_discovered_models(discovered_models: &[String]) {
    use crate::{backend, l4m::L4m, service};

    for model_name in discovered_models {
        // Check if this model already has a service
        if service::get_service_id(model_name).is_some() {
            tracing::debug!("Model '{}' already has a service, skipping creation", model_name);
            continue;
        }

        // Check if we have backend info for this model
        if let Some(backend_info) = get_cached_model_backend_info(model_name) {
            tracing::info!("Creating L4M service for model '{}' with backend at '{}'", model_name, backend_info.backend_endpoint);

            // Create a ZMQ backend connection to the endpoint
            match backend::ZmqBackend::bind(&backend_info.backend_endpoint).await {
                Ok(backend) => {
                    // Create a new L4M service with this backend
                    let l4m_service = L4m::new(backend).await;

                    // Add the service dynamically to the controller
                    match service::add_service_runtime(model_name, l4m_service) {
                        Ok(_) => {
                            tracing::info!("Successfully added L4M service for model '{}'", model_name);

                            // Get the service ID for the newly added service and update cache
                            if let Some(service_id) = service::get_service_id(model_name) {
                                set_model_service_id(model_name, service_id);
                                tracing::info!("Updated cache with service ID {} for model '{}'", service_id, model_name);
                            }
                        }
                        Err(e) => {
                            tracing::error!("Failed to add L4M service to controller for model '{}': {:?}", model_name, e);
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("Failed to connect to backend at '{}' for model '{}': {}", backend_info.backend_endpoint, model_name, e);
                }
            }
        } else {
            tracing::warn!("No backend info found in cache for model '{}' - this should not happen", model_name);
        }
    }
}
