use crate::models::{
    BackendRegistrationRequest, BackendRegistrationResponse, HeartbeatResponse, ListBackendsResponse,
};
use crate::state::SharedState;
use crate::config::Config;
use axum::{extract::{Path, State}, http::StatusCode, Json};
use uuid::Uuid;
use serde_json::{json, Value};
use std::process::{Command, Child, Stdio};
use std::sync::{Arc, Mutex};
use std::path::{Path as StdPath, PathBuf};
use std::fs::{OpenOptions, create_dir_all};
use anyhow::{Result, Context};

/// Resolve a path to an absolute path
fn resolve_absolute_path<P: AsRef<StdPath>>(path: P) -> Result<PathBuf> {
    let path = path.as_ref();
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        let current_dir = std::env::current_dir()
            .context("Failed to get current directory")?;
        Ok(current_dir.join(path))
    }
}

// Global state for controller processes
static CONTROLLER_PROCESSES: std::sync::OnceLock<Arc<Mutex<ControllerProcesses>>> = std::sync::OnceLock::new();

#[derive(Default)]
struct ControllerProcesses {
    engine_process: Option<Child>,
    engine_port: Option<u16>,
}

fn get_controller_processes() -> &'static Arc<Mutex<ControllerProcesses>> {
    CONTROLLER_PROCESSES.get_or_init(|| Arc::new(Mutex::new(ControllerProcesses::default())))
}

fn find_engine_binary(config: &Config) -> Result<PathBuf> {
    let binary_name = &config.services.engine.binary_name;

    // Try to find it in PATH first
    if let Ok(output) = Command::new("which").arg(binary_name).output() {
        if output.status.success() {
            let path_output = String::from_utf8_lossy(&output.stdout);
            let path_str = path_output.trim();
            if !path_str.is_empty() {
                tracing::info!("Found {} in PATH: {}", binary_name, path_str);
                return Ok(PathBuf::from(path_str));
            }
        }
    }

    // Try the configured search paths
    for path_str in &config.paths.engine_binary_search {
        let path = StdPath::new(path_str);
        if path.exists() {
            tracing::info!("Found {} at configured path: {}", binary_name, path_str);
            return Ok(path.to_path_buf());
        }
    }

    Err(anyhow::anyhow!("Could not find {} binary. Please ensure it's compiled or in PATH.", binary_name))
}

fn start_engine_process(config: &Config, config_path: &str, port: u16, manager_port: u16) -> Result<Child> {
    let binary_path = find_engine_binary(config)?;

    // Create logs directory if it doesn't exist
    create_dir_all(&config.logging.directory).map_err(|e| anyhow::anyhow!("Failed to create logs directory: {}", e))?;

    // Create log file for engine
    let timestamp = chrono::Utc::now().format("%Y-%m-%d").to_string();
    let log_file_name = format!("{}/engine-stdout-{}.log", config.logging.directory, timestamp);
    let log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_file_name)
        .map_err(|e| anyhow::anyhow!("Failed to create log file: {}", e))?;

    // Resolve absolute config path
    let absolute_config_path = resolve_absolute_path(config_path)?;

    let mut cmd = Command::new(&binary_path);

    // Start with base args from config
    let mut args = config.services.engine.base_args.clone();

    // Add port
    args.push("--port".to_string());
    args.push(port.to_string());

    // Add engine-manager endpoint (engine now gets config via engine-manager endpoint)
    args.push("--engine-manager".to_string());
    args.push(format!("http://127.0.0.1:{}", manager_port));

    cmd.args(&args);

    // Redirect output to log file
    cmd.stdout(Stdio::from(log_file.try_clone().map_err(|e| anyhow::anyhow!("Failed to clone log file: {}", e))?))
       .stderr(Stdio::from(log_file))
       .stdin(Stdio::null());

    tracing::info!("Starting engine from: {}", binary_path.display());
    tracing::info!("Engine logs will be written to: {}", log_file_name);

    let child = cmd.spawn().map_err(|e| anyhow::anyhow!("Failed to start engine process: {}", e))?;

    tracing::info!("Engine process started with PID: {}", child.id());
    Ok(child)
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
    tracing::debug!("Listing all backends");

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

    let engine_status = if let Some(ref _engine_process) = processes_guard.engine_process {
        json!({
            "running": true,
            "port": processes_guard.engine_port,
            "url": processes_guard.engine_port.map(|p| format!("ws://127.0.0.1:{}", p))
        })
    } else {
        json!({
            "running": false,
            "port": null,
            "url": null
        })
    };

    Json(json!({
        "status": "healthy",
        "service": "pie-engine-manager",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "controller": {
            "engine_process": engine_status
        }
    }))
}

// POST /controller/start
pub async fn controller_start_handler(State(state): State<SharedState>) -> Result<Json<Value>, StatusCode> {
    // Get the config path and manager port from shared state
    let (config_path, manager_port) = {
        let state_guard = state.read().unwrap();
        (
            state_guard.config_path.clone().unwrap_or_else(|| "config.json".to_string()),
            state_guard.manager_port.unwrap_or(8080)
        )
    };

    // Load the unified configuration from the absolute path
    let config = match Config::load_from_file(&config_path) {
        Ok(config) => config,
        Err(e) => {
            tracing::error!("Failed to load configuration from {}: {}", config_path, e);
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };

    let processes = get_controller_processes();
    let mut processes_guard = processes.lock().unwrap();

    if processes_guard.engine_process.is_some() {
        return Ok(Json(json!({
            "status": "already_running",
            "message": "Controller engine process is already running",
            "engine_port": processes_guard.engine_port
        })));
    }

    // Use the configured default engine port
    let engine_port = config.services.engine.default_port;

    // Start the actual engine process
    match start_engine_process(&config, &config_path, engine_port, manager_port) {
        Ok(engine_process) => {
            tracing::info!("Engine process started successfully on port {}", engine_port);
            processes_guard.engine_process = Some(engine_process);
            processes_guard.engine_port = Some(engine_port);

            Ok(Json(json!({
                "status": "started",
                "message": "Controller engine process started successfully",
                "engine_port": engine_port,
                "engine_url": format!("ws://127.0.0.1:{}", engine_port)
            })))
        }
        Err(e) => {
            tracing::error!("Failed to start engine process: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
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
        processes_guard.engine_port = None;
        tracing::info!("Engine process stopped successfully");
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
