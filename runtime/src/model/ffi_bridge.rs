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
    #[tracing::instrument(
        name = "rust.ffi_call",
        skip(self, args),
        fields(rpc_method = %method)
    )]
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

/// Unified backend client trait for both FFI (in-process) and IPC (cross-process) communication.
#[async_trait::async_trait]
pub trait BackendClient: Send + Sync {
    /// Call a method with serialized payload and receive serialized response.
    async fn call_raw(&self, method: &str, payload: Vec<u8>) -> Result<Vec<u8>>;
    
    /// Call a method with timeout.
    async fn call_raw_with_timeout(
        &self,
        method: &str,
        payload: Vec<u8>,
        timeout: std::time::Duration,
    ) -> Result<Vec<u8>> {
        tokio::time::timeout(timeout, self.call_raw(method, payload))
            .await
            .map_err(|_| anyhow::anyhow!("Backend call timed out"))?
    }
}

#[async_trait::async_trait]
impl BackendClient for AsyncFfiClient {
    async fn call_raw(&self, method: &str, payload: Vec<u8>) -> Result<Vec<u8>> {
        let (_id, rx) = self.queue.send_request(method.to_string(), payload);
        rx.await.map_err(|_| anyhow::anyhow!("Response channel closed"))
    }
}

/// Async IPC client for cross-process communication.
///
/// Uses ipc-channel to communicate with Python processes in other PIDs.
#[derive(Clone)]
pub struct AsyncIpcClient {
    backend: std::sync::Arc<crate::model::ffi_ipc::FfiIpcBackend>,
}

impl AsyncIpcClient {
    /// Create a new IPC client from an FfiIpcBackend.
    pub fn new(backend: std::sync::Arc<crate::model::ffi_ipc::FfiIpcBackend>) -> Self {
        Self { backend }
    }
    
    /// Call a Python method asynchronously via IPC.
    pub async fn call<T, R>(&self, method: &str, args: &T) -> Result<R>
    where
        T: Serialize,
        R: DeserializeOwned,
    {
        // Serialize arguments
        let payload = rmp_serde::to_vec_named(args)
            .map_err(|e| anyhow::anyhow!("Failed to serialize args: {}", e))?;
        
        // Send via IPC
        let response = self.backend.call(method, payload).await?;
        
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
            .map_err(|_| anyhow::anyhow!("IPC call timed out"))?
    }
}

#[async_trait::async_trait]
impl BackendClient for AsyncIpcClient {
    async fn call_raw(&self, method: &str, payload: Vec<u8>) -> Result<Vec<u8>> {
        self.backend.call(method, payload).await
    }
}
