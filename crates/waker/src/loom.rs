#[cfg(loom)]
pub(crate) use ::loom::sync::{
    Mutex, RwLock,
    atomic::{AtomicU64, Ordering},
};
#[cfg(not(loom))]
pub(crate) use std::sync::{
    Mutex, OnceLock, RwLock,
    atomic::{AtomicU64, Ordering},
};
