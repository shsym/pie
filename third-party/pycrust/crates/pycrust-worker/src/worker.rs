//! Worker loop implementation for PyCrust.

use crate::convert::{msgpack_to_pyobject, pyobject_to_msgpack};
use iceoryx2::prelude::*;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// Maximum message size in bytes (4MB).
const MAX_MESSAGE_SIZE: usize = 4194304;

/// Number of spin iterations per idle loop.
const SPIN_ITERATIONS: u32 = 1000;

/// Number of idle loops before sleeping (spin budget).
const IDLE_THRESHOLD: u32 = 100;

/// Check Python signals every N idle loops.
const SIGNAL_CHECK_INTERVAL: u32 = 50;

/// Status codes for RPC responses.
mod status {
    pub const OK: u8 = 0;
    pub const METHOD_NOT_FOUND: u8 = 1;
    #[allow(dead_code)]
    pub const INVALID_PARAMS: u8 = 2;
    pub const INTERNAL_ERROR: u8 = 3;
}

/// Run the worker loop, dispatching RPC calls to the Python callback.
#[pyfunction]
#[pyo3(signature = (service_name, dispatch_callback))]
pub fn run_worker(py: Python<'_>, service_name: &str, dispatch_callback: PyObject) -> PyResult<()> {
    // Create iceoryx2 node
    let node = NodeBuilder::new()
        .create::<ipc::Service>()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create node: {}", e)))?;

    // Subscribe to request topic
    let req_service_name = format!("{}_req", service_name);
    let req_service = node
        .service_builder(&req_service_name.as_str().try_into().unwrap())
        .publish_subscribe::<[u8]>()
        .open_or_create()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create request service: {}", e)))?;

    let req_subscriber = req_service.subscriber_builder().create()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create request subscriber: {}", e)))?;

    // Publish to response topic
    let res_service_name = format!("{}_res", service_name);
    let res_service = node
        .service_builder(&res_service_name.as_str().try_into().unwrap())
        .publish_subscribe::<[u8]>()
        .open_or_create()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create response service: {}", e)))?;

    let res_publisher = res_service
        .publisher_builder()
        .initial_max_slice_len(MAX_MESSAGE_SIZE)
        .create()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create response publisher: {}", e)))?;

    eprintln!("[pycrust-worker] Listening on service: {}", service_name);

    let mut idle_count: u32 = 0;

    // Main worker loop
    loop {
        // ... (signal checks) ...
        if idle_count % SIGNAL_CHECK_INTERVAL == 0 {
            py.check_signals()?;
        }

// ... (imports remain)

                // Poll for incoming requests
        match req_subscriber.receive() {
            Ok(Some(sample)) => {
                // Parse request without GIL
                // Sample is not Send, so we must copy the payload bytes.
                let payload_vec = sample.payload().to_vec();
                let parse_result = py.allow_threads(|| {
                    parse_request(&payload_vec)
                });

                let (id, response_status, response_payload) = match parse_result {
                    Ok(parsed) => {
                        let id = parsed.id;
                        let method = parsed.method;

                        // Create a slice from the original vector using the indices
                        let payload_slice = &payload_vec[parsed.payload_start..parsed.payload_start + parsed.payload_len];
                        
                        // Convert payload to Python object (Requires GIL)
                        let py_args_result = msgpack_to_pyobject(py, payload_slice);

                        match py_args_result {
                            Ok(py_args) => {
                                // Call the dispatch callback
                                let result = dispatch_callback.call1(py, (method, py_args));

                                match result {
                                    Ok(py_result) => {
                                        match pyobject_to_msgpack(py_result.bind(py)) {
                                            Ok(response_bytes) => (id, status::OK, response_bytes),
                                            Err(e) => {
                                                let error_msg = format!("Failed to serialize result: {}", e);
                                                let error_bytes = rmp_serde::to_vec_named(&error_msg).unwrap_or_default();
                                                (id, status::INTERNAL_ERROR, error_bytes)
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        // "Method not found" check
                                        let error_str = e.to_string();
                                        let (status_code, error_msg) = if error_str.contains("Method not found") {
                                            (status::METHOD_NOT_FOUND, error_str)
                                        } else {
                                            (status::INTERNAL_ERROR, error_str)
                                        };
                                        let error_bytes = rmp_serde::to_vec_named(&error_msg).unwrap_or_default();
                                        (id, status_code, error_bytes)
                                    }
                                }
                            }
                            Err(e) => {
                                let error_msg = format!("Failed to convert payload: {}", e);
                                let error_bytes = rmp_serde::to_vec_named(&error_msg).unwrap_or_default();
                                (id, status::INTERNAL_ERROR, error_bytes)
                            }
                        }
                    }
                    Err(error_info) => {
                        (error_info.id, status::INTERNAL_ERROR, error_info.error_payload)
                    }
                };

                // Prepare response bytes without GIL
                let response_bytes_result = py.allow_threads(move || {
                    serialize_response(id, response_status, &response_payload)
                });

                // Send response
                if let Ok(encoded) = response_bytes_result {
                    if let Ok(sample) = res_publisher.loan_slice_uninit(encoded.len()) {
                        let sample = sample.write_from_slice(&encoded);
                        let _ = sample.send();
                    }
                }

                idle_count = 0;
            }
            // ... (rest of loop)
            Ok(None) => {
                idle_count = idle_count.saturating_add(1);
                if idle_count < IDLE_THRESHOLD {
                    py.allow_threads(|| {
                        for _ in 0..SPIN_ITERATIONS {
                            std::hint::spin_loop();
                        }
                    });
                } else {
                    py.allow_threads(|| {
                        std::thread::sleep(std::time::Duration::from_micros(10));
                    });
                }
            }
            Err(e) => {
                eprintln!("[pycrust-worker] Receive error: {}", e);
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

struct ParsedRequestHeader {
    id: u64,
    method: String,
    payload_start: usize,
    payload_len: usize,
}

struct ParseError {
    id: u64,
    error_payload: Vec<u8>,
}

/// Parse the raw request bytes into structured data.
fn parse_request(sample: &[u8]) -> Result<ParsedRequestHeader, ParseError> {
    // Zero-copy deserialization using serde
    match rmp_serde::from_slice::<Request>(sample) {
        Ok(request) => {
            // Calculate offsets for payload to avoid copying it
            let start = request.payload.as_ptr() as usize - sample.as_ptr() as usize;
            let len = request.payload.len();
            
            Ok(ParsedRequestHeader {
                id: request.id,
                method: request.method.to_string(), // Copy string to own it
                payload_start: start,
                payload_len: len,
            })
        },
        Err(e) => {
            // Try to extract ID if possible, otherwise 0
            let error_msg = format!("Failed to deserialize request: {}", e);
            let error_bytes = rmp_serde::to_vec_named(&error_msg).unwrap_or_default();
            Err(ParseError { id: 0, error_payload: error_bytes })
        }
    }
}

#[derive(Serialize)]
struct Response<'a> {
    id: u64,
    status: u8,
    #[serde(with = "serde_bytes")]
    payload: &'a [u8],
}

/// Serialize the response components into MessagePack bytes.
fn serialize_response(
    id: u64,
    status: u8,
    payload: &[u8],
) -> Result<Vec<u8>, rmp_serde::encode::Error> {
    let response = Response {
        id,
        status,
        payload,
    };
    rmp_serde::to_vec_named(&response)
}
