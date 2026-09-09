use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("contract: {0}")]
    Contract(String),

    #[error("shard: {0}")]
    Shard(String),

    #[error("checkpoint: {0}")]
    Checkpoint(String),

    #[error("unsupported: {0}")]
    Unsupported(String),

    #[error("overflow: {0}")]
    Overflow(String),

    #[error("internal: {0}")]
    Internal(String),
}

impl From<ztensor::Error> for Error {
    fn from(err: ztensor::Error) -> Self {
        match err {
            ztensor::Error::Unsupported(_) => Self::Unsupported(err.to_string()),
            ztensor::Error::Reject {
                rule: ztensor::Rule::Version,
                ..
            } => Self::Unsupported(err.to_string()),
            _ => Self::Checkpoint(err.to_string()),
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

pub trait OrOverflow<T> {
    fn or_overflow(self, message: impl Into<String>) -> Result<T>;
}

impl<T> OrOverflow<T> for Option<T> {
    fn or_overflow(self, message: impl Into<String>) -> Result<T> {
        self.ok_or_else(|| Error::Overflow(message.into()))
    }
}

impl<T> OrOverflow<T> for std::result::Result<T, std::num::TryFromIntError> {
    fn or_overflow(self, message: impl Into<String>) -> Result<T> {
        self.map_err(|_| Error::Overflow(message.into()))
    }
}
