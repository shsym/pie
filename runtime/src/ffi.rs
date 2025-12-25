//! FFI module containing all PyO3 bindings.
//!
//! This module serves as the boundary between Rust and Python.
//! All Python-exposed types and functions are defined here.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::path::PathBuf;
use tokio::sync::oneshot;

use crate::auth::AuthorizedUsers;
use crate::engine::{self, Config as EngineConfig};

/// Configuration for the PIE server, exposed to Python.
#[pyclass]
#[derive(Clone)]
pub struct ServerConfig {
    #[pyo3(get, set)]
    pub host: String,
    #[pyo3(get, set)]
    pub port: u16,
    #[pyo3(get, set)]
    pub enable_auth: bool,
    #[pyo3(get, set)]
    pub cache_dir: String,
    #[pyo3(get, set)]
    pub verbose: bool,
    #[pyo3(get, set)]
    pub log_path: Option<String>,
}

#[pymethods]
impl ServerConfig {
    #[new]
    #[pyo3(signature = (host = String::from("127.0.0.1"), port = 8080, enable_auth = true, cache_dir = None, verbose = false, log_path = None))]
    fn new(
        host: String,
        port: u16,
        enable_auth: bool,
        cache_dir: Option<String>,
        verbose: bool,
        log_path: Option<String>,
    ) -> Self {
        let cache_dir = cache_dir.unwrap_or_else(|| {
            dirs::home_dir()
                .map(|h| h.join(".pie").join("programs"))
                .unwrap_or_else(|| PathBuf::from(".pie/programs"))
                .to_string_lossy()
                .to_string()
        });
        ServerConfig {
            host,
            port,
            enable_auth,
            cache_dir,
            verbose,
            log_path,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ServerConfig(host='{}', port={}, enable_auth={}, cache_dir='{}', verbose={}, log_path={:?})",
            self.host, self.port, self.enable_auth, self.cache_dir, self.verbose, self.log_path
        )
    }
}

impl From<ServerConfig> for EngineConfig {
    fn from(cfg: ServerConfig) -> Self {
        EngineConfig {
            host: cfg.host,
            port: cfg.port,
            enable_auth: cfg.enable_auth,
            cache_dir: PathBuf::from(cfg.cache_dir),
            verbose: cfg.verbose,
            log: cfg.log_path.map(PathBuf::from),
        }
    }
}

/// Starts the PIE server and returns the internal auth token.
///
/// This function blocks until the server is ready, then returns the token
/// that can be used for internal authentication.
///
/// Args:
///     config: Server configuration
///     authorized_users_path: Optional path to authorized_users.toml
///
/// Returns:
///     The internal authentication token as a string.
#[pyfunction]
#[pyo3(signature = (config, authorized_users_path = None))]
fn start_server(
    py: Python<'_>,
    config: ServerConfig,
    authorized_users_path: Option<String>,
) -> PyResult<String> {
    // Allow other Python threads to run while we block
    py.allow_threads(|| {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;

        rt.block_on(async {
            let authorized_users = match authorized_users_path {
                Some(path) => AuthorizedUsers::load(&PathBuf::from(path)).map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to load authorized users: {}", e))
                })?,
                None => AuthorizedUsers::default(),
            };

            let (ready_tx, ready_rx) = oneshot::channel();
            let (shutdown_tx, shutdown_rx) = oneshot::channel();

            let engine_config: EngineConfig = config.into();

            // Spawn the server in a background task
            let server_handle = tokio::spawn(async move {
                engine::run_server(engine_config, authorized_users, ready_tx, shutdown_rx).await
            });

            // Wait for server to be ready and get the internal auth token
            let internal_token = ready_rx.await.map_err(|e| {
                PyRuntimeError::new_err(format!("Server failed to start: {}", e))
            })?;

            // Store handles for later cleanup (in a real implementation, you'd want
            // to return these or store them in a manager object)
            // For now, we'll let the server run and return the token
            std::mem::forget(shutdown_tx);
            std::mem::forget(server_handle);

            Ok(internal_token)
        })
    })
}

/// Python module definition for pie_server_rs
#[pymodule(name = "pie_server_rs")]
pub fn pie_server_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ServerConfig>()?;
    m.add_function(wrap_pyfunction!(start_server, m)?)?;
    Ok(())
}
