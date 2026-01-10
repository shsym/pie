//! FFI bridge for calling Python from Rust.
//!
//! This module provides direct function calls to Python without any IPC,
//! eliminating polling overhead completely.

use crate::error::{FfiError, Result};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::{de::DeserializeOwned, Serialize};

/// Status codes returned by Python dispatch.
mod status {
    pub const OK: u8 = 0;
    pub const METHOD_NOT_FOUND: u8 = 1;
    pub const INVALID_PARAMS: u8 = 2;
    pub const INTERNAL_ERROR: u8 = 3;
}

/// FFI client for direct Python calls.
///
/// This client calls Python functions directly via PyO3, with no IPC overhead.
/// The Python dispatcher must be registered before use.
///
/// # Example
///
/// ```ignore
/// use pycrust_ffi::FfiClient;
///
/// Python::with_gil(|py| {
///     let dispatcher = /* get Python dispatcher */;
///     let client = FfiClient::new(py, dispatcher)?;
///     
///     let result: i32 = client.call(py, "add", &(10, 20))?;
/// });
/// ```
pub struct FfiClient {
    /// Python dispatch callback: fn(method: str, payload: bytes) -> (status: int, result: bytes)
    dispatcher: Py<PyAny>,
}

impl FfiClient {
    /// Create a new FFI client with a Python dispatcher.
    ///
    /// The dispatcher should be a callable that takes (method: str, payload: bytes)
    /// and returns (status: int, result: bytes).
    pub fn new(dispatcher: Py<PyAny>) -> Self {
        Self { dispatcher }
    }

    /// Call a Python method with typed arguments and return value.
    ///
    /// This is a blocking call that holds the GIL for the duration.
    ///
    /// # Type Parameters
    ///
    /// * `T` - Argument type (must implement Serialize)
    /// * `R` - Return type (must implement DeserializeOwned)
    pub fn call<T, R>(&self, py: Python<'_>, method: &str, args: &T) -> Result<R>
    where
        T: Serialize,
        R: DeserializeOwned,
    {
        // Serialize arguments to MessagePack
        let payload = rmp_serde::to_vec_named(args)?;
        
        // Call Python dispatcher
        let payload_bytes = PyBytes::new(py, &payload);
        let result = self.dispatcher.call1(py, (method, payload_bytes))?;
        
        // Parse response tuple (status, result_bytes)
        let result = result.bind(py);
        let status: u8 = result.get_item(0)?.extract()?;
        let response_bytes: Vec<u8> = result.get_item(1)?.extract()?;
        
        match status {
            status::OK => {
                let value: R = rmp_serde::from_slice(&response_bytes)?;
                Ok(value)
            }
            status::METHOD_NOT_FOUND => {
                let msg: String = rmp_serde::from_slice(&response_bytes).unwrap_or_default();
                Err(FfiError::MethodNotFound(msg))
            }
            status::INVALID_PARAMS => {
                let msg: String = rmp_serde::from_slice(&response_bytes).unwrap_or_default();
                Err(FfiError::InvalidParams(msg))
            }
            _ => {
                let msg: String = rmp_serde::from_slice(&response_bytes).unwrap_or_default();
                Err(FfiError::InternalError(msg))
            }
        }
    }

    /// Low-level call with raw byte payloads.
    ///
    /// This is useful when serialization/deserialization is handled separately.
    pub fn call_raw(&self, py: Python<'_>, method: &str, payload: &[u8]) -> Result<Vec<u8>> {
        // Call Python dispatcher
        let payload_bytes = PyBytes::new(py, payload);
        let result = self.dispatcher.call1(py, (method, payload_bytes))?;
        
        // Parse response tuple (status, result_bytes)
        let result = result.bind(py);
        let status: u8 = result.get_item(0)?.extract()?;
        let response_bytes: Vec<u8> = result.get_item(1)?.extract()?;
        
        match status {
            status::OK => Ok(response_bytes),
            status::METHOD_NOT_FOUND => {
                let msg: String = rmp_serde::from_slice(&response_bytes).unwrap_or_default();
                Err(FfiError::MethodNotFound(msg))
            }
            status::INVALID_PARAMS => {
                let msg: String = rmp_serde::from_slice(&response_bytes).unwrap_or_default();
                Err(FfiError::InvalidParams(msg))
            }
            _ => {
                let msg: String = rmp_serde::from_slice(&response_bytes).unwrap_or_default();
                Err(FfiError::InternalError(msg))
            }
        }
    }

    /// Fire-and-forget notification.
    ///
    /// Calls the Python method but doesn't wait for or parse the response.
    /// Useful for logging, metrics, or one-way commands.
    pub fn notify<T>(&self, py: Python<'_>, method: &str, args: &T) -> Result<()>
    where
        T: Serialize,
    {
        let payload = rmp_serde::to_vec_named(args)?;
        let payload_bytes = PyBytes::new(py, &payload);
        
        // Call but ignore result
        let _ = self.dispatcher.call1(py, (method, payload_bytes))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_status_codes() {
        assert_eq!(status::OK, 0);
        assert_eq!(status::METHOD_NOT_FOUND, 1);
        assert_eq!(status::INVALID_PARAMS, 2);
        assert_eq!(status::INTERNAL_ERROR, 3);
    }
}
