use std::io;
use std::path::PathBuf;
use thiserror::Error;

/// Main error type for the management service
#[derive(Error, Debug)]
pub enum ManagementError {
    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),

    #[error("Process management error: {0}")]
    Process(#[from] ProcessError),

    #[error("ZMQ communication error: {0}")]
    Zmq(String),

    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("UUID error: {0}")]
    Uuid(#[from] uuid::Error),

    #[error("Protocol encoding error: {message}")]
    Protocol { message: String },

    #[error("Service error: {message}")]
    Service { message: String },

    #[error("Model error: {message}")]
    Model { message: String },

    #[error("Unknown model: {0}")]
    UnknownModel(String),

    #[error("Invalid input: {message}")]
    InvalidInput { message: String },
}

#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("Configuration file not found: {path}")]
    FileNotFound { path: PathBuf },

    #[error("Invalid JSON in configuration file: {source}")]
    InvalidJson {
        #[from]
        source: serde_json::Error,
    },

    #[error("Missing required configuration field: {field}")]
    MissingField { field: String },

    #[error("Invalid configuration value for {field}: {value}")]
    InvalidValue { field: String, value: String },

    #[error("Parse error: {0}")]
    ParseError(String),
}

#[derive(Error, Debug)]
pub enum ProcessError {
    #[error("Failed to start process: {command}")]
    StartFailed { command: String },

    #[error("Process not found: {id}")]
    NotFound { id: String },

    #[error("Process already running: {id}")]
    AlreadyRunning { id: String },

    #[error("Process termination timeout: {id}")]
    TerminationTimeout { id: String },

    #[error("Process crashed: {id}, exit_code: {exit_code:?}")]
    Crashed {
        id: String,
        exit_code: Option<i32>,
    },

    #[error("Failed to spawn process: {0}")]
    SpawnFailed(String),

    #[error("Unknown model: {0}")]
    UnknownModel(String),

    #[error("Unknown backend type: {0}")]
    UnknownBackend(String),

    #[error("Backend script not found: {0}")]
    ScriptNotFound(PathBuf),

    #[error("ZMQ error: {0}")]
    ZmqError(String),

    #[error("Protocol error: {0}")]
    ProtocolError(String),
}

impl From<prost::EncodeError> for ManagementError {
    fn from(err: prost::EncodeError) -> Self {
        ManagementError::Protocol {
            message: format!("Encoding error: {}", err)
        }
    }
}

impl From<prost::DecodeError> for ManagementError {
    fn from(err: prost::DecodeError) -> Self {
        ManagementError::Protocol {
            message: format!("Decoding error: {}", err)
        }
    }
}

/// Result type alias for convenience
pub type Result<T> = std::result::Result<T, ManagementError>;
pub type ConfigResult<T> = std::result::Result<T, ConfigError>;
pub type ProcessResult<T> = std::result::Result<T, ProcessError>;
