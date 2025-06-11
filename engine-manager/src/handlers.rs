use crate::models::{
    BackendRegistrationRequest, BackendRegistrationResponse, HeartbeatResponse, ListBackendsResponse,
};
use crate::state::SharedState;
use axum::{extract::{Path, State}, http::StatusCode, Json};
use uuid::Uuid;
use serde_json::{json, Value};
use std::process::{Command, Child, Stdio};
use std::sync::{Arc, Mutex};

// Global state for controller processes
static CONTROLLER_PROCESSES: std::sync::OnceLock<Arc<Mutex<ControllerProcesses>>> = std::sync::OnceLock::new();

#[derive(Default)]
struct ControllerProcesses {
    engine_process: Option<Child>,
}

fn get_controller_processes() -> &'static Arc<Mutex<ControllerProcesses>> {
    CONTROLLER_PROCESSES.get_or_init(|| Arc::new(Mutex::new(ControllerProcesses::default())))
}

// GET /health
pub async fn health_handler() -> Json<Value> {
    Json(json!({
        "status": "healthy",
        "service": "pie-engine-manager",
        "timestamp": chrono::Utc::now().to_rfc3339()
    }))
}

// POST /backends/register
pub async fn register_backend_handler(
    State(state): State<SharedState>,
    Json(payload): Json<BackendRegistrationRequest>,
) -> Result<Json<BackendRegistrationResponse>, StatusCode> {
    // Basic validation
    if payload.management_api_address.is_empty() {
        tracing::error!("Registration attempt with empty management_api_address");
        return Err(StatusCode::BAD_REQUEST);
    }

    if payload.capabilities.is_empty() {
        tracing::error!("Registration attempt with empty capabilities");
        return Err(StatusCode::BAD_REQUEST);
    }

    // Basic URL format validation
    if !payload.management_api_address.starts_with("http://") && !payload.management_api_address.starts_with("https://") {
        tracing::error!("Registration attempt with invalid management_api_address format: {}", payload.management_api_address);
        return Err(StatusCode::BAD_REQUEST);
    }

    tracing::info!("Registering backend with address: {}", payload.management_api_address);

    // Explicitly scope the lock to ensure it's released
    let backend_id = {
        let mut state_guard = state.write().unwrap();
        state_guard.register_backend(payload.capabilities, payload.management_api_address)
    };

    Ok(Json(BackendRegistrationResponse { backend_id }))
}

// POST /backends/{backend_id}/heartbeat
pub async fn heartbeat_handler(
    State(state): State<SharedState>,
    Path(backend_id): Path<Uuid>,
) -> Result<Json<HeartbeatResponse>, StatusCode> {
    tracing::info!("Received heartbeat for backend_id: {}", backend_id);

    // Explicitly scope the lock to ensure it's released
    let result = {
        let mut state_guard = state.write().unwrap();
        state_guard.record_heartbeat(&backend_id)
    };

    match result {
        Ok(status_change) => Ok(Json(HeartbeatResponse {
            message: "Heartbeat received".to_string(),
            status_updated_to: status_change,
        })),
        Err(_) => {
            tracing::warn!("Heartbeat received for unknown backend_id: {}", backend_id);
            Err(StatusCode::NOT_FOUND)
        }
    }
}

// GET /backends
pub async fn list_backends_handler(
    State(state): State<SharedState>,
) -> Result<Json<ListBackendsResponse>, StatusCode> {
    tracing::info!("Listing all backends");

    // Explicitly scope the lock to ensure it's released
    let summaries = {
        let state_guard = state.read().unwrap();
        state_guard.get_all_backends_summary()
    };

    Ok(Json(ListBackendsResponse {
        backends: summaries,
    }))
}

// POST /backends/{backend_id}/terminate
pub async fn terminate_backend_handler(
    State(state): State<SharedState>,
    Path(backend_id): Path<Uuid>,
) -> Result<StatusCode, StatusCode> {
    tracing::info!("Terminating backend_id: {}", backend_id);

    // Explicitly scope the lock to ensure it's released
    let result = {
        let mut state_guard = state.write().unwrap();
        state_guard.terminate_backend(&backend_id)
    };

    match result {
        Ok(_) => {
            tracing::info!("Backend {} marked for termination", backend_id);
            Ok(StatusCode::OK)
        }
        Err(_) => {
            tracing::warn!("Terminate request for unknown backend_id: {}", backend_id);
            Err(StatusCode::NOT_FOUND)
        }
    }
}

// GET /controller/status
pub async fn controller_status_handler() -> Json<Value> {
    let processes = get_controller_processes();
    let processes_guard = processes.lock().unwrap();

    Json(json!({
        "status": "healthy",
        "service": "pie-engine-manager",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "controller": {
            "engine_process_running": processes_guard.engine_process.is_some()
        }
    }))
}

// POST /controller/start
pub async fn controller_start_handler() -> Result<Json<Value>, StatusCode> {
    let processes = get_controller_processes();
    let mut processes_guard = processes.lock().unwrap();

    if processes_guard.engine_process.is_some() {
        return Ok(Json(json!({
            "status": "already_running",
            "message": "Controller engine process is already running"
        })));
    }

    // For now, we'll just simulate starting an engine process
    // In Phase 3, this would actually start the engine
    tracing::info!("Starting controller engine process (simulated)");

    Ok(Json(json!({
        "status": "started",
        "message": "Controller engine process started successfully"
    })))
}

// POST /controller/stop
pub async fn controller_stop_handler() -> Result<Json<Value>, StatusCode> {
    let processes = get_controller_processes();
    let mut processes_guard = processes.lock().unwrap();

    if let Some(mut engine_process) = processes_guard.engine_process.take() {
        tracing::info!("Stopping controller engine process");
        if let Err(e) = engine_process.kill() {
            tracing::error!("Failed to kill engine process: {}", e);
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
        if let Err(e) = engine_process.wait() {
            tracing::error!("Failed to wait for engine process: {}", e);
        }
    }

    Ok(Json(json!({
        "status": "stopped",
        "message": "Controller stopped successfully"
    })))
}

// POST /shutdown - Gracefully shutdown the engine-manager service
pub async fn shutdown_handler(
    State(state): State<SharedState>,
) -> Result<Json<Value>, StatusCode> {
    tracing::info!("=== SHUTDOWN HANDLER CALLED ===");
    tracing::info!("Received shutdown request");

    // First, terminate all registered backends
    let backends_to_terminate = {
        let state_guard = state.read().unwrap();
        tracing::info!("Found {} backends registered for termination", state_guard.backends.len());
        for (id, info) in &state_guard.backends {
            tracing::info!("Backend {} at {}", id, info.management_api_address);
        }
        state_guard.backends.clone()
    };

    if !backends_to_terminate.is_empty() {
        tracing::info!("Terminating {} registered backends", backends_to_terminate.len());

        for (backend_id, backend_info) in backends_to_terminate {
            let mgmt_api_address = &backend_info.management_api_address;
            tracing::info!("Sending terminate signal to backend {} at {}", backend_id, mgmt_api_address);

            // Send terminate request to backend's management API
            let terminate_url = format!("{}/manage/terminate", mgmt_api_address);
            tracing::info!("Constructed terminate URL: {}", terminate_url);
            let client = reqwest::Client::new();

            match client.post(&terminate_url).send().await {
                Ok(response) if response.status().is_success() => {
                    tracing::info!("Successfully sent terminate signal to backend {}", backend_id);
                }
                Ok(response) => {
                    tracing::warn!("Backend {} responded with status {} to terminate request",
                                 backend_id, response.status());
                }
                Err(e) => {
                    tracing::error!("Failed to send terminate signal to backend {}: {}", backend_id, e);
                }
            }
        }

        // Wait a bit for backends to terminate gracefully
        tracing::info!("Waiting for backends to terminate gracefully...");
        tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
    } else {
        tracing::info!("No backends registered, proceeding with shutdown");
    }

    // Then stop any running engine processes
    let processes = get_controller_processes();
    let mut processes_guard = processes.lock().unwrap();

    if let Some(mut engine_process) = processes_guard.engine_process.take() {
        tracing::info!("Stopping engine process before shutdown");
        if let Err(e) = engine_process.kill() {
            tracing::error!("Failed to kill engine process: {}", e);
        }
        if let Err(e) = engine_process.wait() {
            tracing::error!("Failed to wait for engine process: {}", e);
        }
    }

    // Spawn a task to shutdown the server after a short delay
    // This allows the response to be sent before the server shuts down
    tokio::spawn(async {
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        tracing::info!("Shutting down engine-manager service");
        std::process::exit(0);
    });

    Ok(Json(json!({
        "status": "shutting_down",
        "message": "Engine-manager service is shutting down after terminating all backends"
    })))
}
