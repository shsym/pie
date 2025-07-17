use crate::bindings;
use crate::instance::InstanceState;
use crate::runtime::Command;
use tokio::sync::oneshot;

impl bindings::wit::pie::nbi::runtime::Host for InstanceState {
    async fn get_version(&mut self) -> anyhow::Result<String, wasmtime::Error> {
        let (tx, rx) = oneshot::channel();

        Command::GetVersion { event: tx }.dispatch()?;

        let result = rx.await?;
        Ok(result)
    }

    async fn get_instance_id(&mut self) -> anyhow::Result<String, wasmtime::Error> {
        Ok(self.id().to_string())
    }
}
