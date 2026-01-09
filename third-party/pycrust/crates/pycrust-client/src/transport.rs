//! iceoryx2 transport layer for PyCrust.
//!
//! This module handles the actual IPC communication using iceoryx2.
//! The transport runs in a dedicated OS thread that does busy-spinning
//! for minimum latency, directly dispatching responses to waiting futures.

use crate::error::{Result, RpcError};
use crate::protocol::{RpcRequest, RpcResponse};
use dashmap::DashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use tokio::sync::oneshot;

/// Maximum message size in bytes (4MB).
const MAX_MESSAGE_SIZE: usize = 4194304;

/// Maximum buffer size for subscribers (handles concurrent requests).
const SUBSCRIBER_BUFFER_SIZE: usize = 256;

/// Number of spin iterations per poll cycle.
/// With ~1ns per spin_loop hint, 100 spins ≈ 100ns of spinning.
const SPIN_ITERATIONS: u32 = 300;

/// Number of idle loops before sleeping (spin budget).
/// After this many consecutive empty polls, we sleep briefly.
const IDLE_THRESHOLD: u32 = 10000;

/// Message to send to the transport thread.
pub enum TransportCommand {
    /// Send a request to the worker.
    SendRequest(RpcRequest),
    /// Shutdown the transport thread.
    Shutdown,
}

/// Shared state between client and transport thread.
pub struct TransportShared {
    /// Pending requests waiting for responses.
    pub pending: DashMap<u64, oneshot::Sender<Result<RpcResponse>>>,
    /// Flag to signal shutdown.
    pub running: AtomicBool,
}

/// Handle to the transport running in a dedicated thread.
pub struct TransportHandle {
    /// Channel to send commands to the transport thread.
    cmd_tx: mpsc::Sender<TransportCommand>,
    /// Shared state with the transport thread.
    shared: Arc<TransportShared>,
    /// Handle to the transport thread.
    thread_handle: Option<thread::JoinHandle<()>>,
}

impl TransportHandle {
    /// Connect to a PyCrust service.
    ///
    /// Spawns a dedicated thread for the transport that does busy-spinning
    /// and directly dispatches responses to waiting futures.
    pub fn connect(service_name: &str) -> Result<Self> {
        let (cmd_tx, cmd_rx) = mpsc::channel::<TransportCommand>();

        let shared = Arc::new(TransportShared {
            pending: DashMap::new(),
            running: AtomicBool::new(true),
        });

        let service_name = service_name.to_string();
        let thread_shared = Arc::clone(&shared);
        let thread_handle = thread::spawn(move || {
            if let Err(e) = run_transport_loop(&service_name, cmd_rx, thread_shared) {
                eprintln!("[pycrust] Transport error: {}", e);
            }
        });

        Ok(Self {
            cmd_tx,
            shared,
            thread_handle: Some(thread_handle),
        })
    }

    /// Get shared state for registering pending requests.
    pub fn shared(&self) -> &Arc<TransportShared> {
        &self.shared
    }

    /// Send a request to the worker.
    pub fn send_request(&self, request: RpcRequest) -> Result<()> {
        self.cmd_tx
            .send(TransportCommand::SendRequest(request))
            .map_err(|_| RpcError::ChannelClosed)
    }

    /// Shutdown the transport.
    pub fn shutdown(&mut self) {
        self.shared.running.store(false, Ordering::Relaxed);
        let _ = self.cmd_tx.send(TransportCommand::Shutdown);
        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for TransportHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Run the transport loop in a dedicated thread.
///
/// This thread does busy-spinning on iceoryx2 and directly dispatches
/// responses to the waiting oneshot channels, completely bypassing Tokio.
fn run_transport_loop(
    service_name: &str,
    cmd_rx: mpsc::Receiver<TransportCommand>,
    shared: Arc<TransportShared>,
) -> Result<()> {
    use iceoryx2::port::publisher::Publisher;
    use iceoryx2::port::subscriber::Subscriber;
    use iceoryx2::prelude::*;

    let node = NodeBuilder::new()
        .create::<ipc::Service>()
        .map_err(|e| RpcError::ConnectionFailed(format!("Failed to create node: {}", e)))?;

    // Create request topic: {service_name}_req
    let req_service_name = format!("{}_req", service_name);
    let req_service = node
        .service_builder(&req_service_name.as_str().try_into().unwrap())
        .publish_subscribe::<[u8]>()
        .open_or_create()
        .map_err(|e| {
            RpcError::ConnectionFailed(format!("Failed to create request service: {}", e))
        })?;

    let req_publisher: Publisher<ipc::Service, [u8], ()> = req_service
        .publisher_builder()
        .initial_max_slice_len(MAX_MESSAGE_SIZE)
        .create()
        .map_err(|e| {
            RpcError::ConnectionFailed(format!("Failed to create request publisher: {}", e))
        })?;

    // Create response topic: {service_name}_res
    let res_service_name = format!("{}_res", service_name);
    let res_service = node
        .service_builder(&res_service_name.as_str().try_into().unwrap())
        .publish_subscribe::<[u8]>()
        .subscriber_max_buffer_size(SUBSCRIBER_BUFFER_SIZE)
        .open_or_create()
        .map_err(|e| {
            RpcError::ConnectionFailed(format!("Failed to create response service: {}", e))
        })?;

    let res_subscriber: Subscriber<ipc::Service, [u8], ()> = res_service
        .subscriber_builder()
        .buffer_size(SUBSCRIBER_BUFFER_SIZE)
        .create()
        .map_err(|e| {
            RpcError::ConnectionFailed(format!("Failed to create response subscriber: {}", e))
        })?;

    // Track consecutive idle iterations for adaptive sleeping
    let mut idle_count: u32 = 0;

    while shared.running.load(Ordering::Relaxed) {
        let mut had_activity = false;

        // Check for commands (non-blocking)
        match cmd_rx.try_recv() {
            Ok(TransportCommand::SendRequest(request)) => {
                if let Err(e) = send_request(&req_publisher, &request) {
                    eprintln!("[pycrust] Failed to send request: {}", e);
                }
                had_activity = true;
            }
            Ok(TransportCommand::Shutdown) => {
                break;
            }
            Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => {
                break;
            }
        }

        // Check for responses (non-blocking) and dispatch directly
        match res_subscriber.receive() {
            Ok(Some(sample)) => {
                let payload = sample.payload();
                match rmp_serde::from_slice::<RpcResponse>(payload) {
                    Ok(response) => {
                        // Directly dispatch to the waiting future - no Tokio involved!
                        if let Some((_, sender)) = shared.pending.remove(&response.id) {
                            let _ = sender.send(Ok(response));
                        }
                    }
                    Err(e) => {
                        // Log deserialization error - this is a protocol error
                        eprintln!(
                            "[pycrust] Failed to deserialize response ({} bytes): {}",
                            payload.len(),
                            e
                        );
                    }
                }
                had_activity = true;
            }
            Ok(None) => {}
            Err(e) => {
                eprintln!("[pycrust] Receive error: {}", e);
            }
        }

        // Tight spinning for minimum latency
        if had_activity {
            idle_count = 0;
        } else {
            idle_count = idle_count.saturating_add(1);

            if idle_count < IDLE_THRESHOLD {
                // Spin-wait: tight loop for minimum latency
                for _ in 0..SPIN_ITERATIONS {
                    std::hint::spin_loop();
                }
            } else {
                // After many idle iterations, sleep briefly to save CPU
                std::thread::sleep(std::time::Duration::from_micros(1));
            }
        }
    }

    Ok(())
}

/// Send a request using the publisher.
fn send_request(
    publisher: &iceoryx2::port::publisher::Publisher<iceoryx2::service::ipc::Service, [u8], ()>,
    request: &RpcRequest,
) -> Result<()> {
    let encoded = rmp_serde::to_vec_named(request)?;

    let sample = publisher
        .loan_slice_uninit(encoded.len())
        .map_err(|e| RpcError::TransportError(format!("Failed to loan sample: {}", e)))?;

    let sample = sample.write_from_slice(&encoded);
    sample
        .send()
        .map_err(|e| RpcError::TransportError(format!("Failed to send request: {}", e)))?;

    Ok(())
}
