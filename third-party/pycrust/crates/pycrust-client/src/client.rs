//! RPC client implementation for PyCrust.

use crate::error::{Result, RpcError};
use crate::protocol::{status, RpcRequest};
use crate::transport::TransportHandle;
use serde::{de::DeserializeOwned, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::oneshot;

/// Maximum payload size (must match transport MAX_MESSAGE_SIZE).
const MAX_PAYLOAD_SIZE: usize = 4194304;

/// RPC client for communicating with Python workers.
///
/// The client manages the IPC connection and maps asynchronous requests
/// to responses using correlation IDs. Response dispatching happens in
/// a dedicated OS thread for minimum latency.
pub struct RpcClient {
    transport: TransportHandle,
    next_id: AtomicU64,
}

impl RpcClient {
    /// Connect to a PyCrust service.
    ///
    /// This creates the transport connection with a dedicated OS thread
    /// that handles IPC polling and response dispatching.
    pub async fn connect(service_name: &str) -> Result<Self> {
        let transport = TransportHandle::connect(service_name)?;

        Ok(Self {
            transport,
            next_id: AtomicU64::new(1),
        })
    }

    /// Call a remote method with typed arguments and return value.
    ///
    /// # Type Parameters
    ///
    /// * `T` - Argument type (must implement Serialize)
    /// * `R` - Return type (must implement DeserializeOwned)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use serde::{Deserialize, Serialize};
    ///
    /// #[derive(Serialize)]
    /// struct AddArgs { a: i32, b: i32 }
    ///
    /// let result: i32 = client.call("add", &AddArgs { a: 10, b: 20 }).await?;
    /// ```
    pub async fn call<T, R>(&self, method: &str, args: &T) -> Result<R>
    where
        T: Serialize,
        R: DeserializeOwned,
    {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let payload = rmp_serde::to_vec_named(args)?;

        // Validate payload size before sending
        if payload.len() > MAX_PAYLOAD_SIZE {
            return Err(RpcError::TransportError(format!(
                "Payload size {} exceeds maximum {} bytes",
                payload.len(),
                MAX_PAYLOAD_SIZE
            )));
        }

        let request = RpcRequest {
            id,
            method: method.to_string(),
            payload,
        };

        // Create oneshot channel for response
        let (tx, rx) = oneshot::channel();

        // Register pending request - the transport thread will dispatch directly
        self.transport.shared().pending.insert(id, tx);

        // Send request - cleanup pending on failure
        if let Err(e) = self.transport.send_request(request) {
            self.transport.shared().pending.remove(&id);
            return Err(e);
        }

        // Wait for response - woken directly by the transport thread
        let response = match rx.await {
            Ok(result) => result?,
            Err(_) => {
                // Channel was dropped (transport shutdown)
                self.transport.shared().pending.remove(&id);
                return Err(RpcError::ChannelClosed);
            }
        };

        // Check status
        if response.status != status::OK {
            let message: String = rmp_serde::from_slice(&response.payload)
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(RpcError::RemoteError {
                status: response.status,
                message,
            });
        }

        // Deserialize result
        let result: R = rmp_serde::from_slice(&response.payload)?;
        Ok(result)
    }

    /// Call a remote method with timeout.
    ///
    /// Similar to `call`, but returns an error if the response is not
    /// received within the specified duration. Properly cleans up pending
    /// requests on timeout.
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
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let payload = rmp_serde::to_vec_named(args)?;

        // Validate payload size before sending
        if payload.len() > MAX_PAYLOAD_SIZE {
            return Err(RpcError::TransportError(format!(
                "Payload size {} exceeds maximum {} bytes",
                payload.len(),
                MAX_PAYLOAD_SIZE
            )));
        }

        let request = RpcRequest {
            id,
            method: method.to_string(),
            payload,
        };

        // Create oneshot channel for response
        let (tx, rx) = oneshot::channel();

        // Register pending request
        self.transport.shared().pending.insert(id, tx);

        // Send request - cleanup pending on failure
        if let Err(e) = self.transport.send_request(request) {
            self.transport.shared().pending.remove(&id);
            return Err(e);
        }

        // Wait for response with timeout
        let response = match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(result)) => result?,
            Ok(Err(_)) => {
                // Channel was dropped
                self.transport.shared().pending.remove(&id);
                return Err(RpcError::ChannelClosed);
            }
            Err(_) => {
                // Timeout - CRITICAL: clean up pending request to prevent memory leak
                self.transport.shared().pending.remove(&id);
                return Err(RpcError::Timeout);
            }
        };

        // Check status
        if response.status != status::OK {
            let message: String = rmp_serde::from_slice(&response.payload)
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(RpcError::RemoteError {
                status: response.status,
                message,
            });
        }

        // Deserialize result
        let result: R = rmp_serde::from_slice(&response.payload)?;
        Ok(result)
    }

    /// Close the client and cleanup resources.
    pub async fn close(mut self) {
        self.transport.shutdown();
    }
}
