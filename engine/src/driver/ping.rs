// Minimalistic driver implementation

use crate::backend;
use crate::driver::DriverError;
use crate::utils::IdPool;
use dashmap::DashMap;
use futures::SinkExt;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};

pub const PROTOCOL: &str = "ping";

mod pb_bindings {
    include!(concat!(env!("OUT_DIR"), "/ping.rs"));
}

pub trait ExecuteCommand: backend::ExecuteCommand<pb_bindings::Ping, pb_bindings::Pong> {}
impl<T> ExecuteCommand for T where T: backend::ExecuteCommand<pb_bindings::Ping, pb_bindings::Pong> {}

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
pub struct Driver<B> {
    backend: B,
    cmd_id_pool: IdPool<u32>,
    event_table: Arc<DashMap<u32, Event>>,
    event_loop_handle: tokio::task::JoinHandle<()>,
}

impl<B> Driver<B>
where
    B: ExecuteCommand,
{
    pub async fn new(backend: B) -> Self {
        let (tx, rx) = mpsc::channel(1000);
        backend.report_to(tx).await;

        let event_table = Arc::new(DashMap::new());
        let event_loop_handle = tokio::spawn(Self::event_loop(rx, event_table.clone()));

        Self {
            backend,
            cmd_id_pool: IdPool::new(u32::MAX),
            event_table,
            event_loop_handle,
        }
    }

    pub async fn submit(&mut self, cmd: Command) -> Result<(), DriverError> {
        match cmd {
            Command::Ping { message, handler } => {
                let correlation_id = self
                    .cmd_id_pool
                    .acquire()
                    .map_err(|e| DriverError::LockError)?;
                let ping = pb_bindings::Ping {
                    correlation_id,
                    message,
                };
                self.event_table
                    .insert(correlation_id, Event::Pong(handler));
                self.backend
                    .exec(ping)
                    .await
                    .map_err(|e| DriverError::Other(e.to_string()))?;
            }
        }
        Ok(())
    }

    pub async fn event_loop(
        mut rx: mpsc::Receiver<pb_bindings::Pong>,
        mut event_table: Arc<DashMap<u32, Event>>,
    ) {
        while let Some(pong) = rx.recv().await {
            let correlation_id = pong.correlation_id;
            if let Some((_, event)) = event_table.remove(&correlation_id) {
                match event {
                    Event::Pong(sender) => {
                        sender.send(pong.message).unwrap();
                    }
                }
            }
        }
    }
}

#[derive(Clone)]
pub struct Simulator {}

impl backend::Simulate<pb_bindings::Ping, pb_bindings::Pong> for Simulator {
    fn simulate(&mut self, cmd: pb_bindings::Ping) -> Option<pb_bindings::Pong> {
        Some(pb_bindings::Pong {
            correlation_id: cmd.correlation_id,
            message: format!("Pong: {}", cmd.message),
        })
    }
}
