//! FFI module containing all PyO3 bindings.
//!
//! This module serves as the boundary between Rust and Python.
//! All Python-exposed types and functions are defined here.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
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
    pub log_dir: Option<String>,
    #[pyo3(get, set)]
    pub registry: String,
}

#[pymethods]
impl ServerConfig {
    #[new]
    fn new(
        host: String,
        port: u16,
        enable_auth: bool,
        cache_dir: String,
        verbose: bool,
        log_dir: String,
        registry: String,
    ) -> Self {
        ServerConfig {
            host,
            port,
            enable_auth,
            cache_dir,
            verbose,
            log_dir: Some(log_dir),
            registry,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ServerConfig(host='{}', port={}, enable_auth={}, cache_dir='{}', verbose={}, log_dir={:?}, registry='{}')",
            self.host, self.port, self.enable_auth, self.cache_dir, self.verbose, self.log_dir, self.registry
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
            log_dir: cfg.log_dir.map(PathBuf::from),
            registry: cfg.registry,
        }
    }
}

/// Handle to a running server, allowing graceful shutdown.
#[pyclass]
pub struct ServerHandle {
    internal_token: String,
    shutdown_tx: Arc<Mutex<Option<oneshot::Sender<()>>>>,
    runtime: Arc<tokio::runtime::Runtime>,
}

#[pymethods]
impl ServerHandle {
    /// Get the internal authentication token.
    #[getter]
    fn internal_token(&self) -> String {
        self.internal_token.clone()
    }

    /// Gracefully shut down the server.
    fn shutdown(&self) -> PyResult<()> {
        let mut guard = self.shutdown_tx.lock().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to acquire lock: {}", e))
        })?;

        if let Some(tx) = guard.take() {
            tx.send(()).map_err(|_| {
                PyRuntimeError::new_err("Failed to send shutdown signal (server may already be stopped)")
            })?;
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("Server already shut down"))
        }
    }

    /// Check if the server is still running.
    fn is_running(&self) -> bool {
        self.shutdown_tx.lock().map(|g| g.is_some()).unwrap_or(false)
    }

    /// Get list of registered model names (backends that have connected).
    fn registered_models(&self) -> Vec<String> {
        crate::model::registered_models()
    }

    fn __repr__(&self) -> String {
        let running = if self.is_running() { "running" } else { "stopped" };
        format!("ServerHandle(status={}, token={}...)", running, &self.internal_token[..8])
    }
}

/// Initialize the Python backend in-process and return a dispatcher.
///
/// This creates a Runtime and dispatcher for direct FFI calls, enabling\n/// low-latency communication with Python.
///
/// Args:
///     model_config: Dictionary containing model configuration
///
/// Returns:
///     A Runtime instance for use with the FFI queue worker
#[pyfunction]
fn initialize_backend(py: Python<'_>, model_config: &Bound<'_, pyo3::types::PyDict>) -> PyResult<PyObject> {
    // Import pie_worker modules
    let config_mod = py.import("pie_worker.config")?;
    let runtime_mod = py.import("pie_worker.runtime")?;
    
    // Create RuntimeConfig using from_args with **kwargs
    let config_cls = config_mod.getattr("RuntimeConfig")?;
    let from_args = config_cls.getattr("from_args")?;
    let config = from_args.call((), Some(model_config))?;
    
    // Create and return Runtime instance directly
    let runtime_cls = runtime_mod.getattr("Runtime")?;
    let runtime = runtime_cls.call1((config,))?;
    
    Ok(runtime.unbind())
}


/// Starts the PIE server and returns a handle for management.
///
/// This function blocks until the server is ready, then returns a ServerHandle
/// that can be used for graceful shutdown and status checks.
///
/// Args:
///     config: Server configuration
///     authorized_users_path: Optional path to authorized_users.toml
///
/// Returns:
///     A ServerHandle object with the internal token and shutdown capability.
#[pyfunction]
#[pyo3(signature = (config, authorized_users_path = None))]
fn start_server(
    py: Python<'_>,
    config: ServerConfig,
    authorized_users_path: Option<String>,
) -> PyResult<ServerHandle> {
    // Allow other Python threads to run while we block
    py.allow_threads(|| {
        let rt = Arc::new(tokio::runtime::Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?);

        let internal_token = rt.block_on(async {
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
            tokio::spawn(async move {
                if let Err(e) = engine::run_server(engine_config, authorized_users, ready_tx, shutdown_rx).await {
                    eprintln!("[PIE] Server error: {}", e);
                }
            });

            // Wait for server to be ready and get the internal auth token
            let internal_token = ready_rx.await.map_err(|e| {
                PyRuntimeError::new_err(format!("Server failed to start: {}", e))
            })?;

            Ok::<(String, oneshot::Sender<()>), PyErr>((internal_token, shutdown_tx))
        })?;

        let (token, shutdown_tx) = internal_token;

        // Keep the runtime alive by leaking it (it will be cleaned up when the handle is dropped)
        // We use Arc to share the runtime
        let runtime_clone = Arc::clone(&rt);
        std::mem::forget(rt);

        Ok(ServerHandle {
            internal_token: token,
            shutdown_tx: Arc::new(Mutex::new(Some(shutdown_tx))),
            runtime: runtime_clone,
        })
    })
}


/// Starts the PIE server with an FFI backend using a queue-based approach.
///
/// Python must create the FfiQueue and start a worker thread BEFORE calling this,
/// so the worker can respond to the handshake.
///
/// Args:
///     config: Server configuration
///     authorized_users_path: Optional path to authorized_users.toml
///     queue: FfiQueue instance (Python creates this and starts worker first)
///
/// Returns:
///     ServerHandle with shutdown capability.
#[pyfunction]
#[pyo3(signature = (config, authorized_users_path, queue))]
fn start_server_with_ffi(
    py: Python<'_>,
    config: ServerConfig,
    authorized_users_path: Option<String>,
    queue: crate::model::ffi_queue::FfiQueue,
) -> PyResult<ServerHandle> {
    use crate::model::{self, Model, SchedulerConfig};
    
    // Allow other Python threads to run while we block
    // IMPORTANT: Worker thread must already be polling the queue!
    py.allow_threads(|| {
        let rt = Arc::new(tokio::runtime::Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?);

        let result = rt.block_on(async {
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
            tokio::spawn(async move {
                if let Err(e) = engine::run_server(engine_config, authorized_users, ready_tx, shutdown_rx).await {
                    eprintln!("[PIE] Server error: {}", e);
                }
            });

            // Wait for server to be ready and get the internal auth token
            let internal_token = ready_rx.await.map_err(|e| {
                PyRuntimeError::new_err(format!("Server failed to start: {}", e))
            })?;

            // Create Model with FFI backend using the queue from Python
            let scheduler_config = SchedulerConfig::default();
            let model = Model::new(queue, scheduler_config).await
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create FFI model: {}", e)))?;
            
            // Get model name before registering
            let model_name = model.name().to_string();
            
            // Register the model with the engine
            let model_id = model::install_model(model_name.clone(), model);
            if model_id.is_none() {
                return Err(PyRuntimeError::new_err(format!(
                    "Failed to register model '{}': already exists", model_name
                )));
            }

            Ok::<(String, oneshot::Sender<()>), PyErr>((internal_token, shutdown_tx))
        })?;

        let (token, shutdown_tx) = result;

        // Keep the runtime alive
        let runtime_clone = Arc::clone(&rt);
        std::mem::forget(rt);

        Ok(ServerHandle {
            internal_token: token,
            shutdown_tx: Arc::new(Mutex::new(Some(shutdown_tx))),
            runtime: runtime_clone,
        })
    })
}

/// Python module definition for _pie
#[pymodule]
pub fn _pie(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ServerConfig>()?;
    m.add_class::<ServerHandle>()?;
    m.add_class::<crate::model::ffi_queue::FfiQueue>()?;
    m.add_function(wrap_pyfunction!(start_server, m)?)?;
    m.add_function(wrap_pyfunction!(start_server_with_ffi, m)?)?;
    m.add_function(wrap_pyfunction!(initialize_backend, m)?)?;
    Ok(())
}

