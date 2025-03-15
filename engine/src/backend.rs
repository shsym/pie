use futures::SinkExt;
use prost::bytes::{Buf, Bytes};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use tokio::task::JoinHandle;
use zeromq::{DealerSocket, Socket, SocketRecv, SocketSend, ZmqError, ZmqMessage};

use prost::{DecodeError, Message};
use thiserror::Error;
use tokio::sync::mpsc;

mod pb_bindings {
    include!(concat!(env!("OUT_DIR"), "/handshake.rs"));
}

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("Handshake failed")]
    HandshakeFailed,

    #[error("ZeroMQ error: {0}")]
    Zmq(#[from] ZmqError),

    #[error("Failed to decode message: {0}")]
    Decode(#[from] DecodeError),

    #[error("Backend channel closed unexpectedly")]
    ChannelClosed,

    #[error("Correlation id not found in event dispatcher: {0}")]
    CorrelationIdNotFound(u32),

    #[error("Unsupported protocol: {0}")]
    UnsupportedProtocol(String),
}

/// The backend trait used for both real and simulated backends.
pub trait Backend: Clone + Send + Sync + 'static {
    fn protocols(&self) -> &[String];

    fn get_protocol_idx(&self, protocol: &str) -> Result<u8, BackendError> {
        self.protocols()
            .iter()
            .position(|p| p == protocol)
            .map(|idx| idx as u8)
            .ok_or_else(|| BackendError::UnsupportedProtocol(protocol.to_string()))
    }
    
    async fn send(&self, protocol_idx: u8, payload: Vec<u8>) -> Result<(), BackendError>;

    /// Registers a listener (an event dispatcher) for a given protocol index.
    fn listen(&self, protocol_idx: u8, tx: mpsc::Sender<Vec<u8>>);
}

/// Implementation using ZeroMQ as transport.
#[derive(Debug, Clone)]
pub struct ZmqBackend {
    protocols: Vec<String>,
    command_tx: mpsc::Sender<(u8, Vec<u8>)>,
    event_dispatchers: Arc<Mutex<Vec<Option<mpsc::Sender<Vec<u8>>>>>>,
    event_loop_handle: Arc<JoinHandle<()>>,
}

impl ZmqBackend {
    pub async fn bind(endpoint: &str) -> Result<Self, BackendError> {
        let mut socket = DealerSocket::new();
        socket.connect(endpoint).await?;
        println!("Connected to server at {}", endpoint);

        // Perform handshake
        let pb_request = pb_bindings::Request {};
        let zmq_request = ZmqMessage::from(pb_request.encode_to_vec());

        socket.send(zmq_request).await.map_err(BackendError::Zmq)?;

        let zmq_response = socket.recv().await.map_err(BackendError::Zmq)?;
        let response_frame = zmq_response.get(0).ok_or(BackendError::HandshakeFailed)?;
        let pb_response = pb_bindings::Response::decode(response_frame.as_ref());
        let protocols = match pb_response {
            Ok(resp) => {
                println!("Handshake successful");
                resp.protocols
            }
            Err(_) => {
                println!("Handshake failed");
                return Err(BackendError::HandshakeFailed);
            }
        };

        let (command_tx, rx) = mpsc::channel(1000);
        let event_dispatchers = Arc::new(Mutex::new(vec![None; protocols.len()]));

        // Spawn the event loop task.
        let event_loop_handle =
            tokio::spawn(Self::event_loop(socket, rx, event_dispatchers.clone()));

        Ok(ZmqBackend {
            protocols,
            command_tx,
            event_dispatchers,
            event_loop_handle: Arc::new(event_loop_handle),
        })
    }

    async fn event_loop(
        mut socket: DealerSocket,
        mut rx: mpsc::Receiver<(u8, Vec<u8>)>,
        event_dispatchers: Arc<Mutex<Vec<Option<mpsc::Sender<Vec<u8>>>>>>,
    ) {
        loop {
            tokio::select! {
                // Handle outgoing commands.
                maybe_command = rx.recv() => {
                    if let Some((protocol, command)) = maybe_command {
                        let mut zmq_frames = VecDeque::new();
                        // Create a frame for the protocol identifier.
                        zmq_frames.push_back(Bytes::copy_from_slice(&[protocol]));
                        // Create a frame for the command payload.
                        zmq_frames.push_back(Bytes::from(command));

                        let zmq_message = match ZmqMessage::try_from(zmq_frames) {
                            Ok(msg) => msg,
                            Err(e) => {
                                eprintln!("Failed to construct ZMQ message: {:?}", e);
                                continue;
                            }
                        };

                        if let Err(e) = socket.send(zmq_message).await {
                            eprintln!("Socket send failed: {:?}", e);
                        }
                    } else {
                        println!("Command channel closed, shutting down event loop.");
                        break;
                    }
                },

                // Handle incoming messages from the server.
                result = socket.recv() => {
                    match result {
                        Ok(msg) => {
                            if msg.len() != 2 {
                                eprintln!("Invalid message received from server: {:?}", msg);
                                continue;
                            }
                            // Safely extract the protocol identifier.
                            let protocol_byte = msg.get(0)
                                .and_then(|frame| frame.first())
                                .copied().unwrap_or(0);
                            let protocol_idx = protocol_byte as usize;
                            let payload = msg.get(1).unwrap().to_vec();

                            let dispatchers = event_dispatchers.lock().unwrap();
                            if protocol_idx >= dispatchers.len() || dispatchers[protocol_idx].is_none() {
                                eprintln!("No event dispatcher found for protocol index: {}", protocol_idx);
                                continue;
                            }
                            if let Err(e) = dispatchers[protocol_idx].as_ref().unwrap().send(payload).await {
                                eprintln!("Failed to dispatch event for protocol {}: {:?}", protocol_idx, e);
                            }
                        },
                        Err(e) => {
                            eprintln!("Socket receive error: {:?}", e);
                            break;
                        }
                    }
                }
            }
        }
    }
}

impl Backend for ZmqBackend {
    fn protocols(&self) -> &[String] {
        &self.protocols
    }

    async fn send(&self, protocol_idx: u8, payload: Vec<u8>) -> Result<(), BackendError> {
        self.command_tx
            .send((protocol_idx, payload))
            .await
            .map_err(|_| BackendError::ChannelClosed)
    }

    fn listen(&self, protocol_idx: u8, tx: mpsc::Sender<Vec<u8>>) {
        let mut dispatchers = self.event_dispatchers.lock().unwrap();
        if (protocol_idx as usize) < dispatchers.len() {
            dispatchers[protocol_idx as usize] = Some(tx);
        } else {
            eprintln!("Protocol index {} out of range", protocol_idx);
        }
    }
}

/// The simulation trait – note it works on raw bytes.
pub trait Simulate: Clone + Send + Sync + 'static {
    fn protocols(&self) -> &[String];
    fn simulate(&mut self, command: Vec<u8>) -> Option<Vec<u8>>;
}

/// A simulated backend implementation.
#[derive(Debug, Clone)]
pub struct SimulatedBackend<F> {
    protocols: Vec<String>,
    command_tx: mpsc::Sender<(u8, Vec<u8>)>,
    event_dispatchers: Arc<Mutex<Vec<Option<mpsc::Sender<Vec<u8>>>>>>,
    event_loop_handle: Arc<JoinHandle<()>>,
    simulator: F,
}

impl<F> SimulatedBackend<F>
where
    F: Simulate + 'static,
{
    pub async fn new(simulator: F) -> Self {
        let protocols = simulator.protocols().to_vec();
        let (command_tx, rx) = mpsc::channel(1000);
        let event_dispatchers = Arc::new(Mutex::new(vec![None; protocols.len()]));

        let simulator_clone = simulator.clone();

        let event_loop_handle = tokio::spawn(Self::event_loop(
            rx,
            event_dispatchers.clone(),
            simulator_clone,
        ));

        Self {
            protocols,
            command_tx,
            event_dispatchers,
            event_loop_handle: Arc::new(event_loop_handle),
            simulator,
        }
    }

    async fn event_loop(
        mut rx: mpsc::Receiver<(u8, Vec<u8>)>,
        event_dispatchers: Arc<Mutex<Vec<Option<mpsc::Sender<Vec<u8>>>>>>,
        mut simulator: F,
    ) {
        while let Some((protocol, command)) = rx.recv().await {
            if let Some(response) = simulator.simulate(command) {
                let maybe_dispatcher = {
                    let dispatchers = event_dispatchers.lock().unwrap();
                    dispatchers
                        .get(protocol as usize)
                        .and_then(|opt| opt.clone())
                };

                if let Some(tx) = maybe_dispatcher {
                    if let Err(e) = tx.send(response).await {
                        eprintln!(
                            "Failed to send simulated response for protocol {}: {:?}",
                            protocol, e
                        );
                    }
                } else {
                    eprintln!("No event dispatcher found for protocol index: {}", protocol);
                }
            }
        }
    }
}

impl<F> Backend for SimulatedBackend<F>
where
    F: Simulate + 'static,
{
    fn protocols(&self) -> &[String] {
        &self.protocols
    }

    async fn send(&self, protocol_idx: u8, payload: Vec<u8>) -> Result<(), BackendError> {
        self.command_tx
            .send((protocol_idx, payload))
            .await
            .map_err(|_| BackendError::ChannelClosed)
    }

    fn listen(&self, protocol_idx: u8, tx: mpsc::Sender<Vec<u8>>) {
        let mut dispatchers = self.event_dispatchers.lock().unwrap();
        if (protocol_idx as usize) < dispatchers.len() {
            dispatchers[protocol_idx as usize] = Some(tx);
        } else {
            eprintln!("Protocol index {} out of range", protocol_idx);
        }
    }
}
