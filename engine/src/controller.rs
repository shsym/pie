use crate::driver::{AnyCommand, Driver, DriverError, NameSelector, Operation, Router};
use crate::instance::Id as InstanceId;
use crate::object::{IdMapper, VspaceId};
use crate::runtime::{Reporter, Runtime};
use crate::server::ServerMessage;
use crate::utils::Stream;
use crate::{driver, object, utils};
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;
use std::time::Instant;
use thiserror::Error;
use tokio::sync::mpsc::{Sender, UnboundedSender, channel, unbounded_channel};
use tokio::sync::{RwLock, mpsc};
use tokio::task;

#[derive(Debug, Error)]
pub enum ControllerError {
    #[error("Driver '{0}' not found")]
    DriverNotFound(String),
    
    #[error("Invalid driver index: {0}")]
    InvalidDriverIndex(usize),
}

type DynCommand = Box<dyn Any + Send + Sync>;

struct ControllerBuilder {
    maps: HashMap<String, usize>,
    channels: Vec<UnboundedSender<Operation<DynCommand>>>,
}

impl ControllerBuilder {
    pub fn new() -> Self {
        let mut builder = Self {
            maps: HashMap::new(),
            channels: Vec::new(),
        };
        builder.install_basics();
        builder
    }

    fn install_basics(&mut self) {
        self.install("runtime", driver::RuntimeHelper::new("0.1.0"));
        self.install("messaging", driver::Messaging::new());
    }

    pub fn install<T>(&mut self, name: &str, mut driver: T) -> &mut Self
    where
        T: Driver + 'static,
    {
        let (tx, mut rx) = unbounded_channel();

        self.channels.push(tx);
        self.maps.insert(name.to_string(), self.channels.len() - 1);

        task::spawn(async move {
            while let Some(op) = rx.recv().await {
                match op {
                    Operation::Create(inst) => driver.create(inst),
                    Operation::Destroy(inst) => driver.destroy(inst),
                    Operation::Dispatch(inst, cmd) => {
                        driver.dispatch(inst, cmd.downcast().unwrap()).await
                    }
                }
            }
        });

        self
    }

    pub fn build(self) -> Controller {
        let dispatcher = CommandDispatcher {
            maps: Arc::new(self.maps),
            channels: Arc::new(self.channels),
        };

        Controller { dispatcher }
    }
}

pub struct Controller {
    dispatcher: CommandDispatcher,
}

impl Controller {
    fn create(&self, inst: InstanceId) {
        for (chan) in self.dispatcher.channels.iter() {
            chan.send(Operation::Create(inst)).unwrap();
        }
    }

    fn destroy(&self, inst: InstanceId) {
        for (chan) in self.dispatcher.channels.iter() {
            chan.send(Operation::Destroy(inst)).unwrap();
        }
    }

    fn dispatcher(&self) -> &CommandDispatcher {
        &self.dispatcher
    }
}

pub struct CommandDispatcher {
    maps: Arc<HashMap<String, usize>>,
    channels: Arc<Vec<UnboundedSender<Operation<DynCommand>>>>,
}

impl CommandDispatcher {
    pub fn get_driver_idx(&self, driver_name: &str) -> Option<usize> {
        self.maps.get(driver_name).map(|idx| *idx)
    }

    pub fn dispatch<C>(
        &self,
        driver_idx: usize,
        inst: InstanceId,
        cmd: C,
    ) -> Result<(), ControllerError>
    where
        C: Any + Send + Sync + Debug + 'static,
    {
        let cmd = Box::new(cmd);

        self.channels
            .get(driver_idx)
            .ok_or(ControllerError::InvalidDriverIndex(driver_idx))?
            .send(Operation::Dispatch(inst, cmd))
            .unwrap();

        Ok(())
    }

    pub fn dispatch_with<C>(
        &self,
        driver_name: &str,
        inst: InstanceId,
        cmd: C,
    ) -> Result<(), ControllerError>
    where
        C: Any + Send + Sync + Debug + 'static,
    {
        let driver_idx = self
            .maps
            .get(driver_name)
            .ok_or(ControllerError::DriverNotFound(driver_name.to_string()))?;

        self.dispatch(*driver_idx, inst, cmd)?;

        Ok(())
    }
}
