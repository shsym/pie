use crate::controller::{ControllerError, EventHandle, Namespace};
use crate::utils::IdPool;
use prost::Message;
use std::collections::HashMap;
use std::fmt::Debug;
use std::mem;
use std::sync::{Arc, Mutex};
use tokio::task::JoinHandle;
use zeromq::{DealerSocket, Socket, SocketRecv, SocketSend, ZmqError, ZmqMessage};

pub mod sdi {
    include!(concat!(env!("OUT_DIR"), "/sdi.rs"));
}

use prost::DecodeError;
use thiserror::Error;
use tokio::sync::mpsc;
use tokio::sync::mpsc::Sender;

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("ZeroMQ error: {0}")]
    Zmq(#[from] ZmqError),

    #[error("Failed to decode message: {0}")]
    Decode(#[from] DecodeError),

    #[error("Backend channel closed unexpectedly")]
    ChannelClosed,

    #[error("Correlation id not found in event dispatcher: {0}")]
    CorrelationIdNotFound(u32),
}

pub trait ExecuteCommand: Debug + Send + Sync + 'static {
    async fn exec(&self, cmd: sdi::Request) -> Result<(), BackendError>;

    fn report_to(&self, tx: mpsc::Sender<sdi::Response>);
}

#[derive(Debug, Clone)]
pub struct Backend {
    cmd_tx: mpsc::Sender<sdi::Request>,
    evt_tx: Arc<Mutex<Option<mpsc::Sender<sdi::Response>>>>, // no tokio mutex. Because it will be only mutated once.
    handle: Arc<JoinHandle<()>>,
}

impl Backend {
    pub async fn bind(endpoint: &str) -> Result<Self, ZmqError> {
        // bind the zmq socket
        let mut socket = DealerSocket::new();
        socket.connect(endpoint).await?;
        println!("Connected to server at {endpoint}");

        // create event dispatcher

        let (tx, rx) = mpsc::channel::<sdi::Request>(1000);
        let event_tx = Arc::new(Mutex::new(None));

        // 3) Spawn the single I/O driver task that handles all read/write from the socket
        let handle = tokio::spawn(Self::backend_routine(socket, rx, event_tx.clone()));

        let backend = Backend {
            cmd_tx: tx,
            evt_tx: event_tx,
            handle: Arc::new(handle),
        };

        Ok(backend)
    }

    async fn backend_routine(
        mut socket: DealerSocket,
        mut rx: mpsc::Receiver<sdi::Request>,
        evt_tx: Arc<Mutex<Option<mpsc::Sender<sdi::Response>>>>,
    ) {
        loop {
            tokio::select! {
                // A) Incoming requests from the channel => send to server
                maybe_req = rx.recv() => {
                    match maybe_req {
                        Some(cmd) => {
                            let bytes = cmd.encode_to_vec();
                            if let Err(e) = socket.send(ZmqMessage::from(bytes)).await {
                                eprintln!("Socket send failed: {:?}", e);
                            }

                        },
                        None => {
                            // channel closed => no more requests
                            println!("Request channel closed, shutting down driver.");
                            break;
                        }
                    }
                },

                // B) Incoming responses from the server
                result = socket.recv() => {
                    match result {
                        Ok(msg) => {
                            let payload = msg.get(0).unwrap();
                            match sdi::Response::decode(payload.as_ref()) {
                                Ok(resp) => {
                                    let a = evt_tx.lock().unwrap();
                                    if let Some(tx) = &*a {
                                        let _ = tx.send(resp);
                                    } else {
                                        eprintln!("No event dispatcher found for response: {:?}", resp);
                                    }
                                }
                                Err(err) => {
                                    eprintln!("Failed to parse Response from server: {:?}", err);
                                }
                            }
                        },
                        Err(e) => {
                            eprintln!("Socket receive error: {:?}", e);
                            // Possibly break or keep going...
                            break;
                        }
                    }
                }
            }
        }
    }
}

impl ExecuteCommand for Backend {
    async fn exec(&self, cmd: sdi::Request) -> Result<(), BackendError> {
        self.cmd_tx
            .send(cmd)
            .await
            .map_err(|_| BackendError::ChannelClosed)?;

        Ok(())
    }

    fn report_to(&self, tx: Sender<sdi::Response>) {
        self.evt_tx.lock().unwrap().replace(tx);
    }
}
