//! Protocol types for PyCrust RPC.

use serde::{Deserialize, Serialize};

/// Request message sent from Rust client to Python worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcRequest {
    /// Unique request identifier for correlation.
    pub id: u64,
    /// Method name to invoke on the worker.
    pub method: String,
    /// MessagePack-encoded arguments.
    #[serde(with = "serde_bytes")]
    pub payload: Vec<u8>,
}

/// Response message sent from Python worker to Rust client.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcResponse {
    /// Request ID this response corresponds to.
    pub id: u64,
    /// Status code (0 = success, non-zero = error).
    pub status: u8,
    /// MessagePack-encoded result or error message.
    #[serde(with = "serde_bytes")]
    pub payload: Vec<u8>,
}

/// Status codes for RPC responses.
pub mod status {
    /// Request completed successfully.
    pub const OK: u8 = 0;
    /// The requested method was not found.
    pub const METHOD_NOT_FOUND: u8 = 1;
    /// Invalid parameters provided to the method.
    pub const INVALID_PARAMS: u8 = 2;
    /// Internal error during method execution.
    pub const INTERNAL_ERROR: u8 = 3;
    /// Validation error (e.g., Pydantic validation failed).
    pub const VALIDATION_ERROR: u8 = 4;
}
