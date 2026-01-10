//! Worker loop implementation for PyCrust.
//!
//! IMPORTANT: The worker loop runs on the main Python thread to ensure
//! compatibility with torch.compile/dynamo. The GIL is released during
//! idle periods using py.allow_threads().

use crate::convert::{msgpack_to_pyobject, pyobject_to_msgpack};
use iceoryx2::prelude::*;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Maximum message size (4MB).
const MAX_MESSAGE_SIZE: usize = 4_194_304;

mod status {
    pub const OK: u8 = 0;
    pub const METHOD_NOT_FOUND: u8 = 1;
    pub const INVALID_PARAMS: u8 = 2;
    pub const INTERNAL_ERROR: u8 = 3;
}

#[pyfunction]
#[pyo3(signature = (service_name, dispatch_callback))]
pub fn run_worker(py: Python<'_>, service_name: &str, dispatch_callback: Py<PyAny>) -> PyResult<()> {
    // Suppress iceoryx2 config warnings
    iceoryx2::prelude::set_log_level(iceoryx2::prelude::LogLevel::Error);
    
    // Initialize iceoryx2 (GIL held - this is fast)
    let node = NodeBuilder::new().create::<ipc::Service>()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create node: {}", e)))?;
    
    let service = node.service_builder(&service_name.try_into().unwrap())
        .request_response::<[u8], [u8]>()
        .max_active_requests_per_client(256)
        .max_loaned_requests(256)
        .open_or_create()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create service: {}", e)))?;

    let server = service.server_builder()
        .initial_max_slice_len(4096)
        .allocation_strategy(AllocationStrategy::PowerOfTwo)
        .create()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create server: {}", e)))?;

    eprintln!("[pycrust] Listening on service: {}", service_name);

    let mut err_buf = Vec::new();
    let mut serial_buf = Vec::with_capacity(4096);
    let mut idle_count: u32 = 0;

    loop {
        // Check for Ctrl+C periodically
        if idle_count % 100 == 0 {
            py.check_signals()?;
        }

        // Poll for requests (GIL held - iceoryx2 types are !Send)
        // The actual poll is very fast (~1us)
        match server.receive() {
            Ok(Some(active_request)) => {
                idle_count = 0;
                let payload_shm = active_request.payload();

                match rmp_serde::from_slice::<Request>(payload_shm) {
                    Ok(req) => {
                        // Call Python dispatch (GIL held - required for torch.compile)
                        let (status_code, response_bytes_res) = {
                            match msgpack_to_pyobject(py, req.payload) {
                                Ok(py_args) => {
                                    match dispatch_callback.call1(py, (req.method, py_args)) {
                                        Ok(py_res) => {
                                            match pyobject_to_msgpack(py_res.bind(py)) {
                                                Ok(b) => (status::OK, Ok(b)),
                                                Err(e) => (status::INTERNAL_ERROR, Err(e.to_string())),
                                            }
                                        }
                                        Err(e) => {
                                            let s = e.to_string();
                                            let code = if s.contains("Method not found") { status::METHOD_NOT_FOUND } else { status::INTERNAL_ERROR };
                                            (code, Err(s))
                                        }
                                    }
                                }
                                Err(e) => (status::INVALID_PARAMS, Err(e.to_string())),
                            }
                        };

                        if req.id != 0 {
                            let payload_slice = match &response_bytes_res {
                                Ok(bytes) => bytes.as_slice(),
                                Err(msg) => {
                                    err_buf.clear();
                                    let _ = rmp_serde::encode::write_named(&mut err_buf, msg);
                                    err_buf.as_slice()
                                }
                            };

                            let resp = Response {
                                id: req.id,
                                status: status_code,
                                payload: payload_slice,
                            };

                            serial_buf.clear();
                            if let Ok(_) = rmp_serde::encode::write_named(&mut serial_buf, &resp) {
                                if serial_buf.len() <= MAX_MESSAGE_SIZE {
                                    if let Ok(sample) = active_request.loan_slice_uninit(serial_buf.len()) {
                                        let sample = sample.write_from_slice(&serial_buf);
                                        let _ = sample.send();
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => eprintln!("[pycrust] Bad header: {}", e),
                }
            }
            Ok(None) => {
                // No request - use tiered idle with GIL RELEASED during sleep
                idle_count = idle_count.saturating_add(1);
                
                if idle_count < 1000 {
                    // PHASE 1: Hot Spin - Ultra-low latency (GIL held, but very short)
                    std::hint::spin_loop();
                } else if idle_count < 10_000 {
                    // PHASE 2: Micro Sleep - Release GIL
                    py.allow_threads(|| {
                        std::thread::sleep(Duration::from_micros(5));
                    });
                } else if idle_count < 100_000 {
                    // PHASE 3: Short Sleep - Release GIL
                    py.allow_threads(|| {
                        std::thread::sleep(Duration::from_micros(50));
                    });
                } else {
                    // PHASE 4: Deep Sleep - Release GIL
                    py.allow_threads(|| {
                        std::thread::sleep(Duration::from_millis(1));
                    });
                }
            }
            Err(e) => {
                eprintln!("[pycrust] Receive error: {}", e);
                py.allow_threads(|| {
                    std::thread::sleep(Duration::from_millis(50));
                });
            }
        }
    }
}

#[derive(Deserialize)]
struct Request<'a> {
    id: u64,
    method: &'a str,
    #[serde(with = "serde_bytes")]
    payload: &'a [u8],
}

#[derive(Serialize)]
struct Response<'a> {
    id: u64,
    status: u8,
    #[serde(with = "serde_bytes")]
    payload: &'a [u8],
}
