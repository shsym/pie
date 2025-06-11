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

    pub fn record_heartbeat(&mut self, backend_id: &Uuid) -> Result<Option<BackendStatus>, &'static str> {
        if let Some(backend) = self.backends.get_mut(backend_id) {
            backend.last_heartbeat = Some(std::time::SystemTime::now());
            let old_status = backend.status.clone();
            let status_changed = if old_status == BackendStatus::Initializing {
                backend.status = BackendStatus::Running;
                true
            } else if old_status == BackendStatus::Unresponsive {
                 backend.status = BackendStatus::Running; // Recovered
                 true
            } else {
                // If already Running, just update heartbeat time.
                // If Terminated, a heartbeat should ideally not occur or be ignored.
                false
            };

            if status_changed {
                Ok(Some(backend.status.clone()))
            } else {
                Ok(None)
            }
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

    pub fn terminate_backend(&mut self, backend_id: &Uuid) -> Result<(), &'static str> {
        if let Some(backend) = self.backends.get_mut(backend_id) {
            backend.status = BackendStatus::Terminated;
            Ok(())
        } else {
            Err("Backend not found")
        }
    }

    /// Check for backends that haven't sent heartbeats within the timeout period
    /// and mark them as unresponsive. Returns the number of backends marked as unresponsive.
    pub fn check_for_timeouts(&mut self, timeout_secs: u64) -> Vec<(Uuid, String)> {
        let now = std::time::SystemTime::now();
        let mut marked_unresponsive = Vec::new();

        for (backend_id, backend) in &mut self.backends {
            // Only check running backends for timeouts
            if backend.status != BackendStatus::Running {
                continue;
            }

            if let Some(last_heartbeat) = backend.last_heartbeat {
                if let Ok(elapsed) = now.duration_since(last_heartbeat) {
                    if elapsed.as_secs() > timeout_secs {
                        let backend_name = backend.capabilities.iter()
                            .find(|cap| cap.starts_with("name:"))
                            .map(|cap| cap[5..].to_string())
                            .unwrap_or_else(|| format!("Backend-{}", backend_id));

                        backend.status = BackendStatus::Unresponsive;
                        marked_unresponsive.push((*backend_id, backend_name));
                    }
                }
            }
        }

        marked_unresponsive
    }
}
