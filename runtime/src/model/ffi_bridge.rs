//! FFI bridge for direct Python calls from the Rust runtime.
//!
//! This module provides an async wrapper that uses a lock-free queue
//! to communicate with Python without spawn_blocking overhead.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │  Rust Async Runtime                                             │
//! │                                                                 │
//! │   client.call("fire_batch", args).await                         │
//! │     → serialize args                                            │
//! │     → queue.send_request(payload)  [lock-free push]             │
//! │     → await response channel                                    │
//! └───────────────────────────────────────────────────────────────┬─┘
//!                                                                 │
//! ┌───────────────────────────────────────────────────────────────▼─┐
//! │  Python Worker (owns CUDA)                                     │
//! │                                                                 │
//! │   request = queue.poll_blocking(100)                            │
//! │   result = fire_batch(...)                                      │
//! │   queue.respond(id, result)                                     │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use crate::model::ffi_queue::FfiQueue;
use anyhow::Result;
use serde::{de::DeserializeOwned, Serialize};

/// Async FFI client that uses a queue-based approach.
///
/// This avoids spawn_blocking overhead by using a lock-free queue
/// that Python polls directly.
#[derive(Clone)]
pub struct AsyncFfiClient {
    queue: FfiQueue,
}

impl AsyncFfiClient {
    /// Create a new AsyncFfiClient with a shared queue.
    ///
    /// The queue should be passed to Python for polling.
    pub fn new_with_queue(queue: FfiQueue) -> Self {
        Self { queue }
    }

    /// Get a reference to the underlying queue.
    pub fn queue(&self) -> &FfiQueue {
        &self.queue
    }

    /// Call a Python method asynchronously.
    ///
    /// This serializes the args, pushes to the queue (lock-free),
    /// and awaits the response from Python.
    pub async fn call<T, R>(&self, method: &str, args: &T) -> Result<R>
    where
        T: Serialize,
        R: DeserializeOwned,
    {
        // Serialize arguments
        let payload = rmp_serde::to_vec_named(args)
            .map_err(|e| anyhow::anyhow!("Failed to serialize args: {}", e))?;

        // Send to queue and get response channel
        let (_id, rx) = self.queue.send_request(method.to_string(), payload);

        // Await response (this doesn't block the executor)
        let response = rx
            .await
            .map_err(|_| anyhow::anyhow!("Response channel closed"))?;

        // Deserialize response
        rmp_serde::from_slice(&response)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize response: {}", e))
    }

    /// Fire-and-forget notification.
    pub async fn notify<T>(&self, method: &str, args: &T) -> Result<()>
    where
        T: Serialize,
    {
        let _: () = self.call(method, args).await?;
        Ok(())
    }

    /// Call with timeout.
    pub async fn call_with_timeout<T, R>(
        &self,
        method: &str,
        args: &T,
        timeout: std::time::Duration,
    ) -> Result<R>
    where
        T: Serialize,
        R: DeserializeOwned,
    {
        tokio::time::timeout(timeout, self.call(method, args))
            .await
            .map_err(|_| anyhow::anyhow!("FFI call timed out"))?
    }
}

/// Timing breakdown for profiling (kept for compatibility).
#[derive(Debug, Default)]
pub struct FfiTiming {
    pub serialize_us: u64,
    pub queue_push_us: u64,
    pub response_wait_us: u64,
    pub deserialize_us: u64,
    pub total_us: u64,
}
