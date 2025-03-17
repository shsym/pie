use crate::instance::{InstanceState};
use crate::{bindings, service};
use tokio::sync::oneshot;
use crate::controller::Command;

impl bindings::wit::symphony::app::system::Host for InstanceState {
    async fn get_runtime_version(&mut self) -> anyhow::Result<String, wasmtime::Error> {
        let (tx, rx) = oneshot::channel();

        self.send_cmd(Command::System(
            service::runtime::Command::GetRuntimeVersion { handle: tx },
        ));

        let result = rx.await?;
        Ok(result)
    }

    async fn get_instance_id(&mut self) -> anyhow::Result<String, wasmtime::Error> {
        let (tx, rx) = oneshot::channel();

        self.send_cmd(Command::System(service::runtime::Command::GetInstanceId {
            handle: tx,
        }));

        let result = rx.await?;
        Ok(result)
    }
}
