use crate::driver::{Driver, DriverError};
use crate::instance::Id as InstanceId;
use crate::runtime::Reporter;
use tokio::sync::oneshot;

#[derive(Debug)]
pub enum Command {
    GetRuntimeVersion { handle: oneshot::Sender<String> },

    GetInstanceId { handle: oneshot::Sender<String> },
}

pub struct RuntimeHelper {
    runtime_version: String,
}

impl RuntimeHelper {
    pub fn new(runtime_version: &str) -> Self {
        RuntimeHelper {
            runtime_version: runtime_version.to_string(),
        }
    }
}

impl Driver for RuntimeHelper {
    type Command = Command;

    async fn dispatch(&mut self, inst: InstanceId, cmd: Self::Command) {
        match cmd {
            Command::GetRuntimeVersion { handle } => {
                handle.send(self.runtime_version.clone()).unwrap();
            }
            Command::GetInstanceId { handle } => {
                handle.send(inst.to_string()).unwrap();
            }
        }
    }
}
