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
use std::time::{Duration, Instant};
use tokio::sync::oneshot;

/// Maximum message size in bytes (4MB).
const MAX_MESSAGE_SIZE: usize = 4194304;

/// Maximum commands to process per cycle (prevents starvation).
const CMD_BATCH_LIMIT: usize = 32;

/// Number of spin iterations before yielding.
const SPIN_ITERATIONS: u32 = 100;

/// Request timeout for zombie cleanup.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

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
        // Signal shutdown but don't block on join
        // This prevents deadlocking the async runtime
        self.shared.running.store(false, Ordering::Relaxed);
        let _ = self.cmd_tx.send(TransportCommand::Shutdown);
        // Thread will exit on its own when it sees running=false or Shutdown
        // We take the handle to prevent double-join if shutdown() was already called
        let _ = self.thread_handle.take();
    }
}

/// Active request being polled for response.
struct ActiveReq {
    id: u64,
    pending: iceoryx2::pending_response::PendingResponse<
        iceoryx2::service::ipc::Service,
        [u8],
        (),
        [u8],
        (),
    >,
    sent_at: Instant,
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
    use iceoryx2::prelude::*;

    // Suppress iceoryx2 config warnings
    set_log_level(LogLevel::Error);

    let node = NodeBuilder::new()
        .create::<ipc::Service>()
        .map_err(|e| RpcError::ConnectionFailed(format!("Failed to create node: {}", e)))?;

    let service = node
        .service_builder(&service_name.try_into().unwrap())
        .request_response::<[u8], [u8]>()
        .max_active_requests_per_client(256)
        .max_loaned_requests(256)
        .open_or_create()
        .map_err(|e| {
            RpcError::ConnectionFailed(format!("Failed to create service: {}", e))
        })?;

    let client: iceoryx2::port::client::Client<
        iceoryx2::service::ipc::Service,
        [u8],
        (),
        [u8],
        (),
    > = service.client_builder()
        .initial_max_slice_len(1024)
        .allocation_strategy(iceoryx2::prelude::AllocationStrategy::PowerOfTwo)
        .create().map_err(|e| {
        RpcError::ConnectionFailed(format!("Failed to create client: {}", e))
    })?;

    let mut active_requests: Vec<ActiveReq> = Vec::with_capacity(256);
    let mut serial_buf = Vec::with_capacity(4096);
    let mut idle_count: u32 = 0;

    while shared.running.load(Ordering::Relaxed) {
        let mut had_activity = false;

        // 1. Process Commands (BOUNDED to prevent starvation)
        let mut cmds_processed = 0;
        loop {
            if cmds_processed >= CMD_BATCH_LIMIT { break; }
            
            match cmd_rx.try_recv() {
                Ok(TransportCommand::SendRequest(request)) => {
                    cmds_processed += 1;
                    serial_buf.clear();
                    
                    if let Err(e) = rmp_serde::encode::write_named(&mut serial_buf, &request) {
                        eprintln!("[pycrust] Failed to serialize request: {}", e);
                        if let Some((_, sender)) = shared.pending.remove(&request.id) {
                            let _ = sender.send(Err(RpcError::SerializationError(e)));
                        }
                    } else if serial_buf.len() > MAX_MESSAGE_SIZE {
                        eprintln!("[pycrust] Request too large: {}", serial_buf.len());
                        if let Some((_, sender)) = shared.pending.remove(&request.id) {
                            let _ = sender.send(Err(RpcError::TransportError(format!("Request too large: {}", serial_buf.len()))));
                        }
                    } else {
                        match client.loan_slice_uninit(serial_buf.len()) {
                            Ok(sample) => {
                                let sample = sample.write_from_slice(&serial_buf);
                                match sample.send() {
                                    Ok(pending) => {
                                        if request.id != 0 {
                                            active_requests.push(ActiveReq {
                                                id: request.id,
                                                pending,
                                                sent_at: Instant::now(),
                                            });
                                        }
                                        had_activity = true;
                                    }
                                    Err(e) => {
                                        eprintln!("[pycrust] Failed to send request: {}", e);
                                        if let Some((_, sender)) = shared.pending.remove(&request.id) {
                                            let _ = sender.send(Err(RpcError::TransportError(format!("Failed to send: {}", e))));
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                eprintln!("[pycrust] Failed to loan sample: {}", e);
                                if let Some((_, sender)) = shared.pending.remove(&request.id) {
                                    let _ = sender.send(Err(RpcError::TransportError(format!("Failed to loan: {}", e))));
                                }
                            }
                        }
                    }
                }
                Ok(TransportCommand::Shutdown) => return Ok(()),
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => return Ok(()),
            }
        }

        // 2. Poll Active Requests (with timeout cleanup)
        let mut i = 0;
        while i < active_requests.len() {
            let req = &mut active_requests[i];

            // Timeout check (zombie cleanup)
            if req.sent_at.elapsed() > REQUEST_TIMEOUT {
                if let Some((_, sender)) = shared.pending.remove(&req.id) {
                    let _ = sender.send(Err(RpcError::Timeout));
                }
                active_requests.swap_remove(i);
                continue;
            }

            match req.pending.receive() {
                Ok(Some(sample)) => {
                    had_activity = true;
                    let payload: &[u8] = sample.payload();
                    
                    match rmp_serde::from_slice::<RpcResponse>(payload) {
                        Ok(response) => {
                            if let Some((_, sender)) = shared.pending.remove(&response.id) {
                                let _ = sender.send(Ok(response));
                            }
                        }
                        Err(e) => {
                            eprintln!("[pycrust] Failed to deserialize response: {}", e);
                            if let Some((_, sender)) = shared.pending.remove(&req.id) {
                                let _ = sender.send(Err(RpcError::ProtocolError(format!("Deserialization failed: {}", e))));
                            }
                        }
                    }
                    active_requests.swap_remove(i);
                    // Don't increment i - swap_remove moved a new item here
                }
                Ok(None) => {
                    i += 1; // No response yet, check next
                }
                Err(e) => {
                    eprintln!("[pycrust] Receive error for req {}: {}", req.id, e);
                    if let Some((_, sender)) = shared.pending.remove(&req.id) {
                        let _ = sender.send(Err(RpcError::TransportError(format!("Receive error: {}", e))));
                    }
                    active_requests.swap_remove(i);
                }
            }
        }

        // 3. Adaptive Spin/Sleep (Tiered)
        if had_activity {
            idle_count = 0;
        } else {
            idle_count = idle_count.saturating_add(1);

            if idle_count < 1000 {
                // PHASE 1: Hot Spin - Ultra-low latency
                std::hint::spin_loop();
            } else if idle_count < 10_000 {
                // PHASE 2: Micro Sleep - Still responsive
                thread::sleep(Duration::from_micros(5));
            } else if idle_count < 100_000 {
                // PHASE 3: Short Sleep - Save CPU but stay ready
                thread::sleep(Duration::from_micros(50));
            } else {
                // PHASE 4: Deep Sleep - Minimal CPU usage
                thread::sleep(Duration::from_millis(1));
            }
        }
    }

    Ok(())
}
