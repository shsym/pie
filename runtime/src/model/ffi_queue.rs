//! Lock-free FFI queue for efficient Rust ↔ Python communication.
//!
//! This module provides a high-performance queue system that allows:
//! - Rust async tasks to push requests without GIL overhead
//! - Python worker to poll for requests without blocking Rust
//! - Responses to flow back efficiently

use crossbeam::channel::{self, Receiver, Sender, TryRecvError};
use dashmap::DashMap;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::oneshot;

/// A request in the queue.
struct FfiRequest {
    id: u64,
    method: String,
    payload: Vec<u8>,
}

/// Response channel for a pending request.
type ResponseTx = oneshot::Sender<Vec<u8>>;

/// Inner state shared between Rust and Python.
struct FfiQueueInner {
    /// Channel for sending requests to Python
    request_tx: Sender<FfiRequest>,
    request_rx: Receiver<FfiRequest>,
    
    /// Pending response channels, keyed by request ID
    pending: DashMap<u64, ResponseTx>,
    
    /// Counter for generating unique request IDs
    next_id: AtomicU64,
}

/// The queue system for FFI communication.
/// 
/// This is shared between Rust and Python via PyO3.
/// Uses internal Arc for efficient cloning.
#[pyclass]
#[derive(Clone)]
pub struct FfiQueue {
    inner: Arc<FfiQueueInner>,
}

#[pymethods]
impl FfiQueue {
    /// Create a new FfiQueue.
    #[new]
    fn new_py() -> Self {
        Self::new()
    }
    
    /// Poll for the next request from Rust.
    /// 
    /// Returns None if no request is available.
    /// Returns (request_id, method_name, payload_bytes) if a request is ready.
    fn poll(&self, py: Python<'_>) -> Option<(u64, String, Py<PyBytes>)> {
        match self.inner.request_rx.try_recv() {
            Ok(req) => {
                let payload = PyBytes::new(py, &req.payload).into();
                Some((req.id, req.method, payload))
            }
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => None,
        }
    }
    
    /// Block until a request is available (with timeout in milliseconds).
    /// 
    /// Returns None on timeout or disconnect.
    /// Releases GIL while waiting for efficiency.
    fn poll_blocking(&self, py: Python<'_>, timeout_ms: u64) -> Option<(u64, String, Py<PyBytes>)> {
        let timeout = std::time::Duration::from_millis(timeout_ms);
        let inner = Arc::clone(&self.inner);
        
        // Release GIL while waiting
        let maybe_req = py.allow_threads(|| {
            match inner.request_rx.recv_timeout(timeout) {
                Ok(req) => Some(req),
                Err(_) => None,
            }
        });
        
        maybe_req.map(|req| {
            let payload = PyBytes::new(py, &req.payload).into();
            (req.id, req.method, payload)
        })
    }
    
    /// Send a response back to Rust for the given request ID.
    fn respond(&self, request_id: u64, response: &[u8]) -> bool {
        if let Some((_, tx)) = self.inner.pending.remove(&request_id) {
            tx.send(response.to_vec()).is_ok()
        } else {
            false
        }
    }
    
    /// Get the number of pending requests waiting for responses.
    fn pending_count(&self) -> usize {
        self.inner.pending.len()
    }
}

impl FfiQueue {
    /// Create a new FFI queue.
    pub fn new() -> Self {
        let (request_tx, request_rx) = channel::unbounded();
        
        Self {
            inner: Arc::new(FfiQueueInner {
                request_tx,
                request_rx,
                pending: DashMap::new(),
                next_id: AtomicU64::new(1),
            }),
        }
    }
    
    /// Send a request to Python and get a receiver for the response.
    /// 
    /// This does NOT block or acquire GIL - it just pushes to the queue.
    pub fn send_request(&self, method: String, payload: Vec<u8>) -> (u64, oneshot::Receiver<Vec<u8>>) {
        let id = self.inner.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        
        // Store the response channel
        self.inner.pending.insert(id, tx);
        
        // Send request to queue (non-blocking)
        let _ = self.inner.request_tx.send(FfiRequest { id, method, payload });
        
        (id, rx)
    }
}

impl Default for FfiQueue {
    fn default() -> Self {
        Self::new()
    }
}
