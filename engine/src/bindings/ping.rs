use crate::instance::{InstanceState};
use crate::{bindings, driver};
use tokio::sync::oneshot;
use crate::controller::Command;

impl bindings::wit::symphony::app::ping::Host for InstanceState {
    async fn ping(&mut self, message: String) -> anyhow::Result<String, wasmtime::Error> {
        let (tx, rx) = oneshot::channel();

        self.send_cmd(Command::Ping(driver::ping::Command::Ping {
            message,
            handler: tx,
        }));

        let result = rx.await.or(Err(wasmtime::Error::msg("Ping failed")))?;
        Ok(result)
    }
}
