pub mod l4m;
pub mod l4m_vision;
pub mod messaging;
pub mod ping;
pub mod runtime;

use crate::instance::Id as InstanceId;
use crate::object;
use crate::object::ObjectError;
use std::any::{Any, TypeId};
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::hash::Hash;
use std::mem;
use std::time::{Duration, Instant};
use thiserror::Error;
// Common driver routines

#[derive(Error, Debug)]
pub enum DriverError {
    #[error("Mutex lock failed")]
    LockError,

    #[error("Object error: {0}")]
    ObjectError(#[from] ObjectError),

    #[error("Event dispatcher error: {0}")]
    EventDispatcherError(#[from] anyhow::Error),

    #[error("Failed to acquire a vspace id")]
    VspacePoolAcquireFailed,

    #[error("Failed to release the vspace id")]
    VspacePoolReleaseFailed,

    #[error("Vspace not found for instance {0}")]
    VspaceNotFound(InstanceId),

    #[error("Instance already exists: {0}")]
    InstanceAlreadyExists(InstanceId),

    #[error("Instance not found: {0}")]
    InstanceNotFound(InstanceId),

    #[error("Namespace not found: {0}")]
    NamespaceNotFound(usize),

    #[error("Send error: {0}")]
    SendError(String),

    #[error("Some error: {0}")]
    Other(String),
}

pub type DynCommand = Box<dyn Any + Send>;

pub trait Driver {
    fn accepts(&self) -> &[TypeId];

    fn create_inst(&mut self, inst: InstanceId) -> Result<(), DriverError>;
    fn destroy_inst(&mut self, inst: InstanceId) -> Result<(), DriverError>;

    fn submit(&mut self, inst: InstanceId, cmd: DynCommand) -> Result<(), DriverError>;

    async fn flush(&mut self) -> Result<(), DriverError>;
}
