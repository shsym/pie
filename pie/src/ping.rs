// Minimalistic driver implementation

use crate::backend;
use crate::backend::Backend;
use crate::service::Service;
use crate::utils::IdPool;
use dashmap::DashMap;
use prost::Message;

use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, oneshot};

pub const PROTOCOL: &str = "ping";

mod pb_bindings {
    include!(concat!(env!("OUT_DIR"), "/ping.rs"));
}

#[derive(Debug)]
pub enum Command {
    Ping {
        message: String,
        handler: oneshot::Sender<String>,
    },
}

#[derive(Debug)]
pub enum Event {
    Pong(oneshot::Sender<String>),
}
#[derive(Debug)]
pub struct Ping<B> {
    backend: B,
    protocol_id: u8,
    cmd_id_pool: IdPool<u32>,
    cmd_queue: Arc<Mutex<Vec<Command>>>,
    event_table: Arc<DashMap<u32, Event>>,
    event_loop_handle: tokio::task::JoinHandle<()>,
}

impl<B> Ping<B>
where
    B: Backend,
{
    pub async fn new(backend: B) -> Self {
        let protocol_id = backend
            .protocol_index(PROTOCOL)
            .expect("Failed to get protocol index");

        let (tx, rx) = mpsc::channel(1000);
        backend.register_listener(protocol_id, tx).await;

        let event_table = Arc::new(DashMap::new());
        let event_loop_handle = tokio::spawn(Self::event_loop(rx, event_table.clone()));

        Self {
            backend,
            protocol_id,

            cmd_id_pool: IdPool::new(u32::MAX),
            cmd_queue: Arc::new(Mutex::new(Vec::new())),
            event_table,
            event_loop_handle,
        }
    }

    async fn event_loop(
        mut rx: mpsc::Receiver<Vec<u8>>,
        event_table: Arc<DashMap<u32, Event>>,
    ) {
        while let Some(pong) = rx.recv().await {
            let pong = pb_bindings::Pong::decode(pong.as_slice()).unwrap();

            let correlation_id = pong.correlation_id;
            if let Some((_, event)) = event_table.remove(&correlation_id) {
                match event {
                    Event::Pong(sender) => {
                        // should be ok, not unwrap.
                        // the inst may have dropped even before the pong is received
                        sender.send(pong.message).ok();
                    }
                }
            }
        }
    }
}

//#[async_trait]
impl<B> Service for Ping<B>
where
    B: Backend,
{
    type Command = Command;

    async fn handle(&mut self, cmd: Self::Command) {
        match cmd {
            Command::Ping { message, handler } => {
                let correlation_id = self.cmd_id_pool.acquire().unwrap();

                let ping = pb_bindings::Ping {
                    correlation_id,
                    message,
                };
                self.event_table
                    .insert(correlation_id, Event::Pong(handler));
                self.backend
                    .send(self.protocol_id, ping.encode_to_vec())
                    .unwrap();
            }
        }
    }
}

#[derive(Clone)]
pub struct Simulator {
    pub protocols: Vec<String>,
}

impl Simulator {
    pub fn new() -> Self {
        Self {
            protocols: vec![PROTOCOL.to_string()],
        }
    }
}

impl backend::Simulate for Simulator {
    fn protocols(&self) -> &[String] {
        self.protocols.as_slice()
    }

    fn simulate(&mut self, command: Vec<u8>) -> Option<Vec<u8>> {
        let cmd = pb_bindings::Ping::decode(command.as_slice()).unwrap();
        Some(
            pb_bindings::Pong {
                correlation_id: cmd.correlation_id,
                message: format!("Pong: {}", cmd.message),
            }
            .encode_to_vec(),
        )
    }
}
