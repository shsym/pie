use crate::driver_l4m::{DriverError, EventHandle, Namespace};
use crate::utils::IdPool;
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::mem;
use std::sync::{Arc};
use tokio::task::JoinHandle;
use zeromq::{DealerSocket, Socket, SocketRecv, SocketSend, ZmqError, ZmqMessage};

use prost::DecodeError;
use thiserror::Error;
use tokio::sync::{mpsc, Mutex};
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

pub trait ExecuteCommand<A, B>: Send + Sync + 'static {
    async fn exec(&self, cmd: A) -> Result<(), BackendError>;

    async fn report_to(&self, tx: mpsc::Sender<B>);
}

#[derive(Debug, Clone)]
pub struct ZmqBackend<A, B> {
    cmd_tx: Sender<A>,
    evt_tx: Arc<Mutex<Option<Sender<B>>>>, // no tokio mutex. Because it will be only mutated once.
    handle: Arc<JoinHandle<()>>,
}

impl<A, B> ZmqBackend<A, B>
where
    A: prost::Message + 'static,
    B: prost::Message + Default + 'static,
{
    pub async fn bind(endpoint: &str) -> Result<Self, ZmqError> {
        // bind the zmq socket
        let mut socket = DealerSocket::new();
        socket.connect(endpoint).await?;
        println!("Connected to server at {endpoint}");

        // create event dispatcher

        let (tx, rx) = mpsc::channel::<A>(1000);
        let event_tx = Arc::new(Mutex::new(None));

        // 3) Spawn the single I/O driver task that handles all read/write from the socket
        let handle = tokio::spawn(Self::backend_routine(socket, rx, event_tx.clone()));

        let backend = ZmqBackend {
            cmd_tx: tx,
            evt_tx: event_tx,
            handle: Arc::new(handle),
        };

        Ok(backend)
    }

    async fn backend_routine(
        mut socket: DealerSocket,
        mut rx: mpsc::Receiver<A>,
        evt_tx: Arc<Mutex<Option<Sender<B>>>>,
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
                            match B::decode(payload.as_ref()) {
                                Ok(resp) => {
                                    let a = evt_tx.lock().await;
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

impl<A, B> ExecuteCommand<A, B> for ZmqBackend<A, B>
where
    A: prost::Message + 'static,
    B: prost::Message + Default + 'static,
{
    async fn exec(&self, cmd: A) -> Result<(), BackendError> {
        self.cmd_tx
            .send(cmd)
            .await
            .map_err(|_| BackendError::ChannelClosed)?;

        Ok(())
    }

    async fn report_to(&self, tx: Sender<B>) {
        self.evt_tx.lock().await.replace(tx);
    }
}
