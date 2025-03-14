use crate::utils::IdPool;
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::mem;
use std::sync::Arc;
use tokio::task::JoinHandle;
use zeromq::{DealerSocket, Socket, SocketRecv, SocketSend, ZmqError, ZmqMessage};

use prost::DecodeError;
use thiserror::Error;
use tokio::sync::mpsc::Sender;
use tokio::sync::{Mutex, mpsc};

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

pub trait Protocol<A, B>: Clone + Send + Sync + 'static {
    async fn exec(&self, cmd: A) -> Result<(), BackendError>;

    async fn report_to(&self, tx: mpsc::Sender<B>);
}

#[derive(Debug)]
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
    pub async fn bind(endpoint: &str, protocol: &str) -> Result<Self, ZmqError> {
        // bind the zmq socket
        let mut socket = DealerSocket::new();
        socket.connect(endpoint).await?;
        println!("Connected to server at {endpoint}");

        // do a handshake with the server
        socket.send(ZmqMessage::from(protocol)).await?;
        
        // if let Some(response) = socket.recv().await?.get(0) {
        //     match response.as_ref() {
        //         b"\x01" => println!("Handshake successful (True)"),
        //         b"\x00" => println!("Handshake failed (False)"),
        //         _ => println!("Unexpected response: {:?}", response),
        //     }
        // } else {
        //     println!("No response received");
        // }

        let resp = socket.recv().await?;

        //println!("Handshake response: {:?}", resp);

        match resp.get(0).unwrap().as_ref() {
            b"\x01" => println!("Handshake successful"),
            _ => {
                println!("Handshake failed (False)");
                return Err(ZmqError::Other("Handshake failed"));
            }
        }

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
                            // if msg.len() != 2 {
                            //     eprintln!("Invalid message received from server: {:?}", msg);
                            //     continue;
                            // }
                            let payload = msg.get(0).unwrap();
                            match B::decode(payload.as_ref()) {
                                Ok(resp) => {
                                    let a = evt_tx.lock().await;
                                    if let Some(tx) = &*a {
                                        let _ = tx.send(resp).await;
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

impl<A, B> Clone for ZmqBackend<A, B>
where
    A: 'static + prost::Message,
    B: 'static + Default + prost::Message,
{
    fn clone(&self) -> Self {
        Self {
            cmd_tx: self.cmd_tx.clone(),
            evt_tx: self.evt_tx.clone(),
            handle: self.handle.clone(),
        }
    }
}

impl<A, B> Protocol<A, B> for ZmqBackend<A, B>
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

pub trait Simulate<A, B>: Clone + Send + Sync + 'static {
    fn simulate(&mut self, cmd: A) -> Option<B>;
}

pub struct SimulatedBackend<A, B, F> {
    cmd_tx: mpsc::Sender<A>,
    evt_tx: Arc<Mutex<Option<mpsc::Sender<B>>>>,
    handle: Arc<JoinHandle<()>>,
    simulator: F,
}

impl<A, B, F> SimulatedBackend<A, B, F>
where
    A: prost::Message + 'static,
    B: Default + prost::Message + 'static,
    F: Simulate<A, B>,
{
    pub async fn new(simulator: F) -> Self {
        let (cmd_tx, rx) = mpsc::channel::<A>(1000);
        let evt_tx = Arc::new(Mutex::new(None));

        // Clone simulate so one copy is passed to the backend routine.
        let handle = tokio::spawn(Self::backend_routine(
            rx,
            Arc::clone(&evt_tx),
            simulator.clone(),
        ));

        Self {
            cmd_tx,
            evt_tx,
            handle: Arc::new(handle),
            simulator,
        }
    }

    async fn backend_routine(
        mut rx: mpsc::Receiver<A>,
        evt_tx: Arc<Mutex<Option<mpsc::Sender<B>>>>,
        mut simulator: F,
    ) {
        while let Some(cmd) = rx.recv().await {
            let resp = simulator.simulate(cmd);

            if let Some(resp) = resp {
                let guard = evt_tx.lock().await;
                if let Some(tx) = &*guard {
                    // Send the response; ignore errors if the receiver has been dropped.
                    let _ = tx.send(resp).await;
                } else {
                    eprintln!("No event dispatcher found for response: {:?}", resp);
                }
            }
        }
    }
}

impl<A, B, F> Clone for SimulatedBackend<A, B, F>
where
    A: 'static + prost::Message,
    B: 'static + Default + prost::Message,
    F: Simulate<A, B>,
{
    fn clone(&self) -> Self {
        Self {
            cmd_tx: self.cmd_tx.clone(),
            evt_tx: Arc::clone(&self.evt_tx),
            handle: Arc::clone(&self.handle),
            simulator: self.simulator.clone(),
        }
    }
}

impl<A, B, F> Protocol<A, B> for SimulatedBackend<A, B, F>
where
    A: prost::Message + 'static,
    B: prost::Message + Default + 'static,
    F: Simulate<A, B>,
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
