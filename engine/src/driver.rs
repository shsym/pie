use crate::object::ObjectError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DriverError {
    #[error("Mutex lock failed")]
    LockError,

    #[error("Channel send error")]
    SendError,

    #[error("Object error: {0}")]
    ObjectError(#[from] ObjectError),

    #[error("Event dispatcher error: {0}")]
    EventDispatcherError(#[from] anyhow::Error),
}
