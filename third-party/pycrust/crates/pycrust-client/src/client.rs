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
    /// Public API: Call without timeout
    pub async fn call<T, R>(&self, method: &str, args: &T) -> Result<R>
    where
        T: Serialize,
        R: DeserializeOwned,
    {
        // Use a very long duration effectively acting as "forever"
        self.invoke_rpc(method, args, None).await
    }

    /// Public API: Call with timeout
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
        self.invoke_rpc(method, args, Some(timeout)).await
    }

    /// Public API: Fire-and-forget notification.
    ///
    /// Sends a request without waiting for a response.
    /// Useful for logging, metrics, or one-way commands.
    /// Uses ID = 0 to signal to the worker that no response is needed.
    pub fn notify<T>(&self, method: &str, args: &T) -> Result<()>
    where
        T: Serialize,
    {
        // ID 0 is reserved for notifications (no response)
        let id = 0;
        let payload = rmp_serde::to_vec_named(args)?;

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

        // Just send it, don't register pending
        self.transport.send_request(request)
    }

    /// Private unified implementation
    async fn invoke_rpc<T, R>(
        &self,
        method: &str,
        args: &T,
        timeout: Option<std::time::Duration>,
    ) -> Result<R>
    where
        T: Serialize,
        R: DeserializeOwned,
    {
        // Fetch ID, ensure we never use 0 which is reserved for notifications
        let mut id = self.next_id.fetch_add(1, Ordering::SeqCst);
        if id == 0 {
             id = self.next_id.fetch_add(1, Ordering::SeqCst);
        }

        let payload = rmp_serde::to_vec_named(args)?;

        if payload.len() > MAX_PAYLOAD_SIZE {
            return Err(RpcError::TransportError(format!(
                "Payload size {} exceeds maximum {} bytes",
                payload.len(),
                MAX_PAYLOAD_SIZE
            )));
        }

        // Create the response channel
        let (tx, rx) = oneshot::channel();

        // 1. REGISTER: Insert into pending map
        self.transport.shared().pending.insert(id, tx);

        let request = RpcRequest {
            id,
            method: method.to_string(),
            payload,
        };

        // 2. SEND: Dispatch to transport thread
        if let Err(e) = self.transport.send_request(request) {
            // Cleanup on send failure
            self.transport.shared().pending.remove(&id);
            return Err(e);
        }

        // 3. AWAIT: Wait for response or timeout
        let response_result = match timeout {
            Some(duration) => {
                match tokio::time::timeout(duration, rx).await {
                    Ok(res) => res, // Inner result from oneshot
                    Err(_) => {
                        // TIMEOUT OCCURRED
                        // Critical: Remove from map so we don't leak memory
                        self.transport.shared().pending.remove(&id);
                        return Err(RpcError::Timeout);
                    }
                }
            }
            None => rx.await,
        };

        // Handle Channel Errors (Transport closed)
        let response = match response_result {
            Ok(res) => res?, // Unwrap Result<RpcResponse>
            Err(_) => {
                // Sender dropped without sending (Transport died)
                self.transport.shared().pending.remove(&id);
                return Err(RpcError::ChannelClosed);
            }
        };

        // 4. PROCESS: Check Status
        if response.status != status::OK {
            // Attempt to decode error message
            let msg: String = rmp_serde::from_slice(&response.payload)
                .unwrap_or_else(|_| "Unknown remote error".to_string());
            return Err(RpcError::RemoteError {
                status: response.status,
                message: msg,
            });
        }

        // 5. DESERIALIZE: Happy path
        // Fix: Use correct error variant (DeserializationError) which accepts decode::Error from from_slice
        rmp_serde::from_slice(&response.payload).map_err(RpcError::DeserializationError)
    }

    /// Close the client and cleanup resources.
    pub async fn close(mut self) {
        self.transport.shutdown();
    }
}
