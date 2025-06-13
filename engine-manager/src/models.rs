use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BackendRegistrationRequest {
    pub capabilities: Vec<String>,
    pub management_api_address: String, // e.g., "http://localhost:8081/manage"
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BackendRegistrationResponse {
    pub backend_id: Uuid,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum BackendStatus {
    Initializing,
    Running,
    Unresponsive,
    Terminated,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BackendInfo {
    pub backend_id: Uuid,
    pub capabilities: Vec<String>,
    pub management_api_address: String,
    pub status: BackendStatus,
    pub last_heartbeat: Option<std::time::SystemTime>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HeartbeatResponse {
    pub message: String,
    pub status_updated_to: Option<BackendStatus>,
}

// For listing backends
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ListBackendsResponse {
    pub backends: Vec<BackendInfoSummary>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BackendInfoSummary {
    pub backend_id: Uuid,
    pub status: BackendStatus,
    pub management_api_address: String,
    pub capabilities: Vec<String>,
}
