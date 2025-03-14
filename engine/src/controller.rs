use crate::driver::{Driver, DriverError, DynCommand};
use crate::instance::Id as InstanceId;
use crate::object::{IdMapper, VspaceId};
use crate::runtime::Runtime;
use crate::server::ServerMessage;
use crate::utils::Stream;
use crate::{driver, object, utils};
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use thiserror::Error;
use tokio::sync::mpsc;
use tokio::sync::mpsc::{UnboundedSender, unbounded_channel};
use tokio::task;

#[derive(Debug, Error)]
pub enum ControllerError {
    #[error("Failed to acquire a vspace id")]
    VspacePoolAcquireFailed,

    #[error("Failed to release the vspace id")]
    VspacePoolReleaseFailed,

    #[error("Vspace not found for instance {0}")]
    VspaceNotFound(InstanceId),

    #[error("Instance not found: {0}")]
    InstanceNotFound(InstanceId),

    #[error("Client not found for instance {0}")]
    ClientNotFound(InstanceId),

    #[error("Driver error: {0}")]
    DriverError(String),

    #[error("Exported block resource not found: {0}")]
    ExportedBlockNotFound(String),

    #[error("Send error: {0}")]
    SendError(String),
}

enum Cmd {
    CreateInst(InstanceId),
    DestroyInst(InstanceId),
    Submit(InstanceId, DynCommand),
}

pub struct Controller {
    runtime: Arc<Runtime>,
    drivers: HashMap<TypeId, UnboundedSender<Cmd>>,

    accepted_types: Vec<TypeId>,
}

impl Controller {
    pub fn new(runtime: Arc<Runtime>) -> Self {
        Controller {
            runtime,
            drivers: HashMap::new(),
            accepted_types: Vec::new(),
        }
    }

    pub fn install<T>(&mut self, mut driver: T)
    where
        T: Driver + 'static,
    {
        let (tx, mut rx) = unbounded_channel();

        for type_id in driver.accepts() {
            self.drivers.insert(*type_id, tx.clone());

            if !self.accepted_types.contains(type_id) {
                self.accepted_types.push(*type_id);
            } else {
                eprintln!("Warning: Driver already exists for type {:?}", type_id);
            }
        }

        let mut runtime = self.runtime.clone();

        task::spawn(async move {
            while let Some(cmd) = rx.recv().await {
                match cmd {
                    Cmd::CreateInst(inst) => {
                        driver.create_inst(inst).unwrap();
                    }
                    Cmd::DestroyInst(inst) => {
                        driver.destroy_inst(inst).unwrap();
                    }
                    Cmd::Submit(inst, cmd) => {
                        if let Err(reason) = driver.submit(inst, cmd).await {
                            runtime.terminate_program(inst, reason.to_string()).await;
                        } else {
                            driver.flush().await.unwrap();
                        }
                    }
                }
            }
        });
    }
}

impl Driver for Controller {
    fn accepts(&self) -> &[TypeId] {
        self.accepted_types.as_slice()
    }

    fn create_inst(&mut self, inst: InstanceId) -> Result<(), DriverError> {
        for driver in self.drivers.values() {
            driver.send(Cmd::CreateInst(inst)).unwrap();
        }
        Ok(())
    }

    fn destroy_inst(&mut self, inst: InstanceId) -> Result<(), DriverError> {
        for driver in self.drivers.values() {
            driver.send(Cmd::DestroyInst(inst)).unwrap();
        }
        Ok(())
    }

    fn submit(&mut self, inst: InstanceId, cmd: DynCommand) -> Result<(), DriverError> {
        let type_id = cmd.as_any().type_id();

        let driver = self
            .drivers
            .get(&type_id)
            .ok_or(ControllerError::DriverError(format!(
                "Driver not found for command {:?}",
                type_id
            )))?;
        driver.send(Cmd::Submit(inst, cmd)).unwrap();
        Ok(())
    }

    async fn flush(&mut self) -> Result<(), DriverError> {
        // Flushing is done in the driver tasks
        Ok(())
    }
}
