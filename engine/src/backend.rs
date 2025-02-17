use std::collections::HashMap;
use std::mem;
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use zeromq::{DealerSocket, Socket, ZmqError, ZmqMessage};
use crate::controller::{ControllerError, EventHandle, Namespace};
use prost::Message;
use crate::backend;

pub mod sdi {
    include!(concat!(env!("OUT_DIR"), "/sdi.rs"));
}


pub struct Backend {
    
    cmd_tx: mpsc::Sender<Vec<(sdi::request::Command, Vec<EventHandle>)>>,
    
    staged: Arc<Mutex<Vec<sdi::Request>>>,
    submitted: Arc<Mutex<Vec<sdi::Request>>>,
    // queue, cmd_buffer, scheduled, submitted
    handle: Arc<Mutex<Option<JoinHandle<()>>>>,
    // zmq handles

    // event dispatcher
    event_dispatcher: Arc<Mutex<EventDispatcher>>,



}


impl Backend {
    
    
    pub fn new() {
        
        
        
    }

    pub fn submit(&self, cmd: sdi::Request) -> Result<(), ControllerError> {
        // Lock and take the staged commands.
        let mut staged = self.staged.lock().map_err(|_| ControllerError::LockError)?;

        // Push the command into the staged commands.
        staged.push(cmd);

        // add the commands to the staged queue
        let mut staged = Vec::new();

        // Add the batched commands to the staged queue.
        for (payload, evt_handles) in batched_payloads {
            let correlation_id = self.acquire_id(Namespace::Cmd)?;

            staged.push(backend::sdi::Command {
                correlation_id,
                payload: Some(payload),
            });

            // if at least one sender is present, add it to the event dispatcher.
            let has_event = evt_handles.iter().any(|s| s.is_some());
            if has_event {
                let mut dispatcher = self
                    .event_dispatcher
                    .lock()
                    .map_err(|_| ControllerError::LockError)?;
                dispatcher.table.insert(correlation_id, evt_handles);
            }
        }
        
        
        Ok(())
        
        
    }
    
    
    pub async fn commit(&self) -> Result<(), ControllerError> {
        // Lock and take the staged commands.
        let mut staged = self.staged.lock().map_err(|_| ControllerError::LockError)?;

        // Lock the sender.
        let tx_guard = self
            .socket_tx
            .lock()
            .map_err(|_| ControllerError::LockError)?;
        let tx = tx_guard.as_ref().ok_or(ControllerError::LockError)?;

        // Send the staged commands, replacing them with an empty Vec.
        tx.send(mem::take(&mut *staged))
            .await
            .map_err(|_| ControllerError::SendError)?;

        Ok(())
    }

    pub async fn bind(&mut self, endpoint: &str) -> Result<(), ZmqError> {
        // bind the zmq socket
        let mut socket = DealerSocket::new();
        socket.connect(endpoint).await?;
        println!("Connected to server at {endpoint}");

        let (tx, rx) = mpsc::channel::<Vec<(sdi::request::Command, Vec<EventHandle>)>>(1000);

        self.socket_tx = Arc::new(Mutex::new(Some(tx)));

        // 3) Spawn the single I/O driver task that handles all read/write from the socket
        let handle = tokio::spawn(Self::socket_driver(
            socket,
            rx,
            self.event_dispatcher.clone(),
        ));

        self.handle = Arc::new(Mutex::new(Some(handle)));

        Ok(())
    }

    async fn socket_driver(
        mut socket: DealerSocket,
        mut rx: mpsc::Receiver<Vec<sdi::Command>>,
        evt_dispatch: Arc<Mutex<EventDispatcher>>,
    ) {
        loop {
            tokio::select! {
                // A) Incoming requests from the channel => send to server
                maybe_req = rx.recv() => {
                    match maybe_req {
                        Some(cmds) => {
                            for cmd in cmds {
                                let bytes = cmd.encode_to_vec();
                                if let Err(e) = socket.send(ZmqMessage::from(bytes)).await {
                                    eprintln!("Socket send failed: {:?}", e);
                                    // You might choose to break or keep trying
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
                            // Dealer/Router typically has 2 frames: [identity, payload]
                            let payload = msg.get(0).unwrap();
                            match sdi::Event::decode(payload.as_ref()) {
                                Ok(evt) => {

                                    // send this evt somewhere elese.

                                    let correlation_id = evt.correlation_id;

                                    let ir_events = match evt.payload.unwrap() {
                                        sdi::event::Payload::SampleTopK(batch) => {
                                            batch.items.into_iter().map(|item| IrEvent::SampleTopK(item)
                                            ).collect()
                                        },
                                        sdi::event::Payload::GetTokenDistribution(batch) => {
                                             batch.items.into_iter().map(|item| IrEvent::GetTokenDistribution(item)
                                            ).collect()
                                        }
                                    };

                                    let mut dispatcher = evt_dispatch.lock().unwrap();
                                    dispatcher.dispatch(correlation_id, ir_events);

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

    fn dispatch(&mut self, correlation_id: u32, event: Vec<crate::controller::IrEvent>) {
        // zip senders and evnt
        let senders = self.table.get_mut(&correlation_id).unwrap();
        assert_eq!(senders.len(), event.len());

        for (sender, evt) in senders.drain(..).zip(event.into_iter()) {
            match sender {
                EventHandle::None => {}
                EventHandle::SampleTopK(s) => {
                    if let crate::controller::IrEvent::SampleTopK(mut resp) = evt {
                        let _ = s.send(mem::take(&mut resp.token_ids));
                    } else {
                        eprintln!("Unexpected event type");
                    }
                }
                EventHandle::GetTokenDistribution(s) => {
                    if let crate::controller::IrEvent::GetTokenDistribution(mut resp) = evt {
                        let _ = s.send(mem::take(&mut resp.distribution));
                    } else {
                        eprintln!("Unexpected event type");
                    }
                }
            }
        }
    }
}