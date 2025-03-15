use crate::driver::{AnyCommand, Driver, DriverError, DynCommand, NameSelector, Router};
use crate::instance::Id as InstanceId;
use crate::object::{IdMapper, VspaceId};
use crate::runtime::{Reporter, Runtime};
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

pub struct Controller {
    router: Router,
    reporter: Reporter, //drivers: HashMap<TypeId, UnboundedSender<Cmd>>,
}

impl Controller {
    pub fn new(reporter: Reporter) -> Self {
        let mut router = Router::new();

        router.install(driver::runtime::RuntimeHelper::new("0.1.0"));
        router.install(driver::messaging::Messaging::new());
        router.install(driver::ping::Ping::new());

        router.install(NameSelector::with(vec![
            ("llama3-1b".to_string(), driver::l4m::L4m::new()),
            ("llama3-7b".to_string(), driver::l4m::L4m::new()),
        ]));

        Controller { router, reporter }
    }
}

impl Driver for Controller {
    type Command = AnyCommand;

    fn create(&mut self, inst: InstanceId) {
        self.router.create(inst)
    }

    fn destroy(&mut self, inst: InstanceId) {
        self.router.destroy(inst)
    }

    async fn dispatch(&mut self, inst: InstanceId, cmd: Self::Command) {
        self.router.dispatch(inst, cmd).await
    }

    fn reporter(&self) -> Option<&Reporter> {
        Some(&self.reporter)
    }
}
