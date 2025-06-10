use crate::models::{
    BackendRegistrationRequest, BackendRegistrationResponse, HeartbeatResponse, ListBackendsResponse,
};
use crate::state::SharedState;
use axum::{extract::{Path, State}, http::StatusCode, Json};
use uuid::Uuid;

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
