// Minimalistic driver implementation

use crate::backend_old;
use crate::driver::DriverError;
use crate::instance::Id as InstanceId;
use crate::utils::IdPool;
use dashmap::DashMap;
use futures::SinkExt;
use std::collections::VecDeque;
use std::mem;
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, oneshot};

pub const PROTOCOL: &str = "ping";

mod pb_bindings {
    include!(concat!(env!("OUT_DIR"), "/ping.rs"));
}

pub trait CompatibleBackend: backend_old::Protocol<pb_bindings::Ping, pb_bindings::Pong> {}
impl<T> CompatibleBackend for T where T: backend_old::Protocol<pb_bindings::Ping, pb_bindings::Pong> {}

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
    cmd_id_pool: IdPool<u32>,
    cmd_queue: Arc<Mutex<Vec<Command>>>,
    event_table: Arc<DashMap<u32, Event>>,
    event_loop_handle: tokio::task::JoinHandle<()>,
}

impl<B> Ping<B>
where
    B: CompatibleBackend,
{
    pub async fn new(backend: B) -> Self {
        let (tx, rx) = mpsc::channel(1000);
        backend.report_to(tx).await;

        let event_table = Arc::new(DashMap::new());
        let event_loop_handle = tokio::spawn(Self::event_loop(rx, event_table.clone()));

        Self {
            backend,
            cmd_id_pool: IdPool::new(u32::MAX),
            cmd_queue: Arc::new(Mutex::new(Vec::new())),
            event_table,
            event_loop_handle,
        }
    }

    pub fn submit(&mut self, _: InstanceId, cmd: Command) -> Result<(), DriverError> {
        let mut cmd_queue = self.cmd_queue.lock().map_err(|e| DriverError::LockError)?;
        cmd_queue.push(cmd);
        Ok(())
    }

    pub async fn flush(&mut self) -> Result<(), DriverError> {
        let cmd_queue = {
            let mut cmd_queue = self.cmd_queue.lock().map_err(|e| DriverError::LockError)?;
            cmd_queue.drain(..).collect::<Vec<_>>()
        };

        for cmd in cmd_queue {
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
        }

        Ok(())
    }

    async fn event_loop(
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

impl backend_old::Simulate<pb_bindings::Ping, pb_bindings::Pong> for Simulator {
    fn simulate(&mut self, cmd: pb_bindings::Ping) -> Option<pb_bindings::Pong> {
        Some(pb_bindings::Pong {
            correlation_id: cmd.correlation_id,
            message: format!("Pong: {}", cmd.message),
        })
    }
}
