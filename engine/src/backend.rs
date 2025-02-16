use std::collections::HashMap;
use std::mem;
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use zeromq::{DealerSocket, ZmqError};
use crate::controller::ControllerError;
use prost::Message;


pub mod sdi {
    include!(concat!(env!("OUT_DIR"), "/sdi.rs"));
}


pub struct Backend {
    staged: Arc<Mutex<Vec<sdi::Command>>>,
    submitted: Arc<Mutex<Vec<sdi::Command>>>,
    // queue, cmd_buffer, scheduled, submitted
    socket_tx: Arc<Mutex<Option<mpsc::Sender<Vec<sdi::Command>>>>>,
    handle: Arc<Mutex<Option<JoinHandle<()>>>>,
    // zmq handles

    // event dispatcher
    event_dispatcher: Arc<Mutex<EventDispatcher>>,



}


impl Backend {
    
    
    pub fn new() {
        
        
        
        ///
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

        let (tx, rx) = mpsc::channel::<Vec<crate::controller::sdi::Command>>(100);

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
        mut rx: mpsc::Receiver<Vec<crate::controller::sdi::Command>>,
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