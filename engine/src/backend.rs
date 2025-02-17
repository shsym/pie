use crate::backend;
use crate::controller::{ControllerError, EventHandle, IrEvent, Namespace};
use crate::utils::IdPool;
use prost::Message;
use std::collections::HashMap;
use std::mem;
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use zeromq::{DealerSocket, Socket, SocketRecv, SocketSend, ZmqError, ZmqMessage};

pub mod sdi {
    include!(concat!(env!("OUT_DIR"), "/sdi.rs"));
}

use prost::DecodeError;
use thiserror::Error;

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

pub trait ExecuteCommand {
    async fn exec(
        &self,
        cmd: sdi::request::Command,
        resp: Option<oneshot::Sender<sdi::response::Payload>>,
    ) -> Result<(), BackendError>;
}

struct BackendDriver {
    event_dispatcher: EventDispatcher,
    cmd_id_pool: IdPool<u32>,
    //submitted: Vec<sdi::request::Command>,
}

impl BackendDriver {
    fn new() -> Self {
        Self {
            event_dispatcher: EventDispatcher::new(),
            cmd_id_pool: IdPool::new(u32::MAX),
        }
    }

    fn outbound(
        &mut self,
        cmd: sdi::request::Command,
        evt_handles: Vec<EventHandle>,
    ) -> sdi::Request {
        let correlation_id = self.cmd_id_pool.acquire().unwrap();

        let request = sdi::Request {
            correlation_id,
            command: Some(cmd),
        };

        // if at least one sender is present, add it to the event dispatcher.
        let has_event = evt_handles.iter().any(|s| s.is_some());
        if has_event {
            self.event_dispatcher
                .table
                .insert(correlation_id, evt_handles);
        }

        request
    }

    fn inbound(&mut self, resp: sdi::Response) {
        // send this evt somewhere elese.

        let correlation_id = resp.correlation_id;

        let ir_events = match resp.payload.unwrap() {
            sdi::response::Payload::SampleTopK(batch) => batch
                .items
                .into_iter()
                .map(|item| IrEvent::SampleTopK(item))
                .collect(),
            sdi::response::Payload::GetTokenDistribution(batch) => batch
                .items
                .into_iter()
                .map(|item| IrEvent::GetTokenDistribution(item))
                .collect(),
        };

        self.event_dispatcher.dispatch(correlation_id, ir_events);
    }
}

#[derive(Debug, Clone)]
pub struct Backend {
    cmd_tx: mpsc::Sender<Vec<(sdi::request::Command, Vec<EventHandle>)>>,
    handle: Arc<JoinHandle<()>>,
}

impl Backend {
    pub async fn bind(endpoint: &str) -> Result<Self, ZmqError> {
        // bind the zmq socket
        let mut socket = DealerSocket::new();
        socket.connect(endpoint).await?;
        println!("Connected to server at {endpoint}");

        // create event dispatcher

        let (tx, rx) = mpsc::channel::<Vec<(sdi::request::Command, Vec<EventHandle>)>>(1000);

        // 3) Spawn the single I/O driver task that handles all read/write from the socket
        let handle = tokio::spawn(Self::backend_routine(socket, rx));

        let backend = Backend {
            cmd_tx: tx,
            handle: Arc::new(handle),
        };

        Ok(backend)
    }

    async fn backend_routine(
        mut socket: DealerSocket,
        mut rx: mpsc::Receiver<Vec<(sdi::request::Command, Vec<EventHandle>)>>,
    ) {
        let mut driver = BackendDriver::new();

        loop {
            tokio::select! {
                // A) Incoming requests from the channel => send to server
                maybe_req = rx.recv() => {
                    match maybe_req {
                        Some(cmds) => {
                            for (cmd, evt) in cmds {
                                let req = driver.outbound(cmd, evt);
                                let bytes = req.encode_to_vec();
                                if let Err(e) = socket.send(ZmqMessage::from(bytes)).await {
                                    eprintln!("Socket send failed: {:?}", e);
                                }
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
                                    driver.inbound(resp);
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
    async fn exec(
        &self,
        cmd: sdi::request::Command,
        resp: Option<oneshot::Sender<sdi::response::Payload>>,
    ) -> Result<(), BackendError> {
        let evt = match resp {
            Some(s) => vec![EventHandle::GetTokenDistribution(s)],
            None => vec![EventHandle::None],
        };

        self.cmd_tx
            .send(vec![(cmd, evt)])
            .await
            .map_err(|_| BackendError::ChannelClosed)?;

        Ok(())
    }
}

#[derive(Debug)]
struct EventDispatcher {
    // maps correlation_id to a list of senders.
    table: HashMap<u32, Vec<EventHandle>>,
}

impl EventDispatcher {
    fn new() -> Self {
        Self {
            table: HashMap::new(),
        }
    }

    fn register(&mut self, correlation_id: u32, sender: Vec<EventHandle>) {
        self.table.insert(correlation_id, sender);
    }

    fn dispatch(&mut self, correlation_id: u32, event: Vec<IrEvent>) {
        // zip senders and evnt
        let senders = self.table.get_mut(&correlation_id).unwrap();
        assert_eq!(senders.len(), event.len());

        for (sender, evt) in senders.drain(..).zip(event.into_iter()) {
            match sender {
                EventHandle::None => {}
                EventHandle::SampleTopK(s) => {
                    if let IrEvent::SampleTopK(mut resp) = evt {
                        let _ = s.send(mem::take(&mut resp.token_ids));
                    } else {
                        eprintln!("Unexpected event type");
                    }
                }
                EventHandle::GetTokenDistribution(s) => {
                    if let IrEvent::GetTokenDistribution(mut resp) = evt {
                        let _ = s.send(mem::take(&mut resp.distribution));
                    } else {
                        eprintln!("Unexpected event type");
                    }
                }
            }
        }
    }
}
