use crate::models::{BackendInfo, BackendStatus};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use uuid::Uuid;

pub type SharedState = Arc<RwLock<AppState>>;

#[derive(Default)]
pub struct AppState {
    pub backends: HashMap<Uuid, BackendInfo>,
}

impl AppState {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn register_backend(&mut self, capabilities: Vec<String>, management_api_address: String) -> Uuid {
        let backend_id = Uuid::new_v4();
        let backend_info = BackendInfo {
            backend_id,
            capabilities,
            management_api_address,
            status: BackendStatus::Initializing,
            last_heartbeat: None,
        };
        self.backends.insert(backend_id, backend_info);
        backend_id
    }

    pub fn record_heartbeat(&mut self, backend_id: &Uuid) -> Result<BackendStatus, &'static str> {
        if let Some(backend) = self.backends.get_mut(backend_id) {
            backend.last_heartbeat = Some(std::time::SystemTime::now());
            let old_status = backend.status.clone();
            if old_status == BackendStatus::Initializing {
                backend.status = BackendStatus::Running;
            } else if old_status == BackendStatus::Unresponsive {
                 backend.status = BackendStatus::Running; // Recovered
            }
            // If already Running, just update heartbeat time.
            // If Terminated, a heartbeat should ideally not occur or be ignored.
            Ok(backend.status.clone())
        } else {
            Err("Backend not found")
        }
    }

    pub fn get_all_backends_summary(&self) -> Vec<crate::models::BackendInfoSummary> {
        self.backends
            .values()
            .map(|info| crate::models::BackendInfoSummary {
                backend_id: info.backend_id,
                status: info.status.clone(),
                management_api_address: info.management_api_address.clone(),
                capabilities: info.capabilities.clone(),
            })
            .collect()
    }
}
