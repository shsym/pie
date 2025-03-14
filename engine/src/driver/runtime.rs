use crate::driver::DriverError;
use crate::instance::Id as InstanceId;
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
    pub fn new(runtime_version: String) -> Self {
        RuntimeHelper { runtime_version }
    }

    pub fn submit(&mut self, inst: InstanceId, cmd: Command) -> Result<(), DriverError> {
        match cmd {
            Command::GetRuntimeVersion { handle } => {
                handle.send(self.runtime_version.clone()).unwrap();
            }
            Command::GetInstanceId { handle } => {
                handle.send(inst.to_string()).unwrap();
            }
        }

        Ok(())
    }
}
