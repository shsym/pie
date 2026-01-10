//! Error types for PyCrust FFI.

use std::fmt;

/// Result type for PyCrust FFI operations.
pub type Result<T> = std::result::Result<T, FfiError>;

/// Errors that can occur during FFI RPC calls.
#[derive(Debug)]
pub enum FfiError {
    /// Method not found on the Python worker.
    MethodNotFound(String),
    
    /// Invalid parameters passed to the method.
    InvalidParams(String),
    
    /// Internal error in the Python worker.
    InternalError(String),
    
    /// Serialization error (MessagePack).
    SerializationError(String),
    
    /// Python exception occurred.
    PythonError(String),
}

impl fmt::Display for FfiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FfiError::MethodNotFound(m) => write!(f, "Method not found: {}", m),
            FfiError::InvalidParams(m) => write!(f, "Invalid params: {}", m),
            FfiError::InternalError(m) => write!(f, "Internal error: {}", m),
            FfiError::SerializationError(m) => write!(f, "Serialization error: {}", m),
            FfiError::PythonError(m) => write!(f, "Python error: {}", m),
        }
    }
}

impl std::error::Error for FfiError {}

impl From<pyo3::PyErr> for FfiError {
    fn from(err: pyo3::PyErr) -> Self {
        FfiError::PythonError(err.to_string())
    }
}

impl From<rmp_serde::encode::Error> for FfiError {
    fn from(err: rmp_serde::encode::Error) -> Self {
        FfiError::SerializationError(err.to_string())
    }
}

impl From<rmp_serde::decode::Error> for FfiError {
    fn from(err: rmp_serde::decode::Error) -> Self {
        FfiError::SerializationError(err.to_string())
    }
}
