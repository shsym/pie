use crate::instance::InstanceState;
use crate::ping::Command;
use crate::{bindings, service};
use tokio::sync::oneshot;

impl bindings::wit::symphony::app::ping::Host for InstanceState {
    async fn ping(&mut self, message: String) -> anyhow::Result<String, wasmtime::Error> {
        let (tx, rx) = oneshot::channel();

        let service_id = service::get_service_id("ping").unwrap();

        service::dispatch(
            service_id,
            Command::Ping {
                message,
                handler: tx,
            },
        )?;

        let result = rx.await.or(Err(wasmtime::Error::msg("Ping failed")))?;
        Ok(result)
    }
}
