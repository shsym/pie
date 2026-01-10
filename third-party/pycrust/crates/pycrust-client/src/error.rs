//! Error types for PyCrust client.

use thiserror::Error;

/// Error type for RPC operations.
#[derive(Error, Debug)]
pub enum RpcError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] rmp_serde::encode::Error),

    #[error("Deserialization error: {0}")]
    DeserializationError(#[from] rmp_serde::decode::Error),

    #[error("Transport error: {0}")]
    TransportError(String),

    #[error("Remote error (status {status}): {message}")]
    RemoteError { status: u8, message: String },

    #[error("Timeout waiting for response")]
    Timeout,

    #[error("Channel closed")]
    ChannelClosed,

    #[error("Service not available: {0}")]
    ServiceNotAvailable(String),

    #[error("Protocol error: {0}")]
    ProtocolError(String),
}

/// Result type alias for RPC operations.
pub type Result<T> = std::result::Result<T, RpcError>;
