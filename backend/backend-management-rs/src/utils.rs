//! Utility functions for the Symphony Management Service

use std::path::Path;
use tracing::{info, warn};

/// Extract IPC socket path from endpoint URL
pub fn extract_ipc_path(endpoint: &str) -> Option<&str> {
    if endpoint.starts_with("ipc://") {
        Some(&endpoint[6..]) // Remove "ipc://" prefix
    } else {
        None
    }
}

/// Clean up IPC socket file if it exists
pub fn cleanup_ipc_socket(endpoint: &str) {
    if let Some(socket_path) = extract_ipc_path(endpoint) {
        if Path::new(socket_path).exists() {
            match std::fs::remove_file(socket_path) {
                Ok(()) => info!("Cleaned up IPC socket: {}", socket_path),
                Err(e) => warn!("Failed to remove IPC socket {}: {}", socket_path, e),
            }
        }
    }
}

/// Clean up all Symphony-related IPC sockets
pub fn cleanup_all_symphony_sockets() {
    info!("Cleaning up all Symphony IPC sockets");
    
    // Common patterns for Symphony IPC sockets
    let socket_patterns = [
        "/tmp/symphony-ipc",
        "/tmp/symphony-cli", 
        "/tmp/symphony-test-client",
        "/tmp/symphony-test-cli",
    ];
    
    // Clean up known socket files
    for pattern in &socket_patterns {
        if Path::new(pattern).exists() {
            match std::fs::remove_file(pattern) {
                Ok(()) => info!("Cleaned up IPC socket: {}", pattern),
                Err(e) => warn!("Failed to remove IPC socket {}: {}", pattern, e),
            }
        }
    }
    
    // Clean up model instance sockets (symphony-model-*)
    if let Ok(entries) = std::fs::read_dir("/tmp") {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                if name.starts_with("symphony-model-") {
                    let path = entry.path();
                    match std::fs::remove_file(&path) {
                        Ok(()) => info!("Cleaned up model IPC socket: {:?}", path),
                        Err(e) => warn!("Failed to remove model IPC socket {:?}: {}", path, e),
                    }
                }
            }
        }
    }
}
