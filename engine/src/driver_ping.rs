use crate::backend::BackendError;
use crate::driver::DriverError;
use crate::driver_l4m::EventHandle;
use crate::object::VspaceId;
use crate::utils::Stream;
use crate::{backend, utils};
use dashmap::DashMap;
use rand::Rng;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, mpsc, oneshot};
use tokio::task::JoinHandle;

pub const PROTOCOL: &str = "ping";

mod ping {
    include!(concat!(env!("OUT_DIR"), "/ping.rs"));
}

pub trait ExecuteCommand: backend::ExecuteCommand<ping::Ping, ping::Pong> {}
impl<T> ExecuteCommand for T where T: backend::ExecuteCommand<ping::Ping, ping::Pong> {}

#[derive(Debug)]
pub struct Driver<B> {
    backend: B,
    event_table: Arc<DashMap<u32, oneshot::Sender<String>>>,
    cmd_id_pool: utils::IdPool<u32>,
}

impl<B> Driver<B>
where
    B: ExecuteCommand,
{
    pub async fn new(b: B) -> Self {
        let (tx, rx) = mpsc::channel(1000);

        let event_table = Arc::new(DashMap::new());

        let resp_handler = tokio::spawn(Self::handle_pong(rx, event_table.clone()));
        b.report_to(tx).await;

        Self {
            backend: b,
            event_table,
            cmd_id_pool: utils::IdPool::new(100000),
        }
    }

    pub async fn ping(
        &mut self,
        message: String,
        handler: oneshot::Sender<String>,
    ) -> Result<(), DriverError> {
        let correlation_id = self
            .cmd_id_pool
            .acquire()
            .map_err(|e| DriverError::LockError)?;

        let msg = ping::Ping {
            correlation_id,
            message,
        };

        self.event_table.insert(correlation_id, handler);

        self.backend
            .exec(msg)
            .await
            .map_err(|_| DriverError::SendError)?;

        Ok(())
    }

    async fn handle_pong(
        mut rx: mpsc::Receiver<ping::Pong>,
        mut event_table: Arc<DashMap<u32, oneshot::Sender<String>>>,
    ) {
        while let Some(pong) = rx.recv().await {
            let correlation_id = pong.correlation_id;
            if let Some((_, e)) = event_table.remove(&correlation_id) {
                e.send(pong.message).unwrap();
            }
        }
    }
}

#[derive(Clone)]
pub struct Simulator {}

impl backend::Simulate<ping::Ping, ping::Pong> for Simulator {
    fn simulate(&mut self, cmd: ping::Ping) -> Option<ping::Pong> {
        Some(ping::Pong {
            correlation_id: cmd.correlation_id,
            message: format!("Pong: {}", cmd.message),
        })
    }
}
