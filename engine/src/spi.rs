use std::future::Future;
use wasmtime::component::Resource;
use wasmtime::component::{bindgen, ResourceTable};
use wasmtime::Result;
use wasmtime_wasi::{WasiCtx, WasiCtxBuilder, WasiView};

use crate::ClientId;
use tokio::sync::mpsc::{Receiver, Sender};
use uuid::Uuid;

bindgen!({
    path: "../spi/app/wit",
    world: "app",
    async: true,
    with: {
        "spi:lm/inference/language-model": LanguageModel,
        "spi:app/system/channel": Channel
    },
    // Interactions with `ResourceTable` can possibly trap so enable the ability
    // to return traps from generated functions.
    trappable_imports: true,
});

pub struct InstanceState {
    pub instance_id: Uuid,

    pub wasi_ctx: WasiCtx,
    pub resource_table: ResourceTable,

    // For communication between the instance and the host
    pub sender: Sender<InstanceMessage>,
    pub receiver: Receiver<InstanceMessage>,
}

pub struct InstanceMessage {
    pub instance_id: Uuid,
    pub channel_id: u32,
    pub message: String,
}

impl WasiView for InstanceState {
    fn table(&mut self) -> &mut ResourceTable {
        &mut self.resource_table
    }
    fn ctx(&mut self) -> &mut WasiCtx {
        &mut self.wasi_ctx
    }
}

impl InstanceState {
    pub fn new(
        instance_id: Uuid,
        sender: Sender<InstanceMessage>,
        receiver: Receiver<InstanceMessage>,
    ) -> Self {
        let mut builder = WasiCtx::builder();
        builder.inherit_stderr().inherit_network().inherit_stdout();

        InstanceState {
            instance_id,
            wasi_ctx: builder.build(),
            resource_table: ResourceTable::new(),
            sender,
            receiver,
        }
    }
}

pub struct LanguageModel {
    model_id: String,
}

pub struct Channel {
    channel_id: u32,
}

impl spi::lm::inference::Host for InstanceState {}
impl spi::lm::inference::HostLanguageModel for InstanceState {
    async fn new(&mut self, model_id: String) -> Result<Resource<LanguageModel>, wasmtime::Error> {
        let handle = LanguageModel { model_id };
        Ok(self.resource_table.push(handle)?)
    }

    async fn tokenize(
        &mut self,
        resource: Resource<LanguageModel>,
        text: String,
    ) -> Result<Vec<u32>, wasmtime::Error> {
        Ok(vec![0])
    }

    async fn detokenize(
        &mut self,
        resource: Resource<LanguageModel>,
        tokens: Vec<u32>,
    ) -> Result<String, wasmtime::Error> {
        Ok("Hello".to_string())
    }

    async fn predict(
        &mut self,
        resource: Resource<LanguageModel>,
        tokens: Vec<u32>,
    ) -> Result<u32, wasmtime::Error> {
        Ok(7)
    }

    async fn drop(&mut self, resource: Resource<LanguageModel>) -> Result<()> {
        let _ = self.resource_table.delete(resource)?;

        Ok(())
    }
}
//
impl spi::app::system::Host for InstanceState {
    async fn get_version(&mut self) -> Result<String, wasmtime::Error> {
        Ok("0.1.0".to_string())
    }
}

impl spi::app::system::HostChannel for InstanceState {
    async fn new(&mut self, channel_id: u32) -> Result<Resource<Channel>, wasmtime::Error> {
        let handle = Channel { channel_id };
        Ok(self.resource_table.push(handle)?)
    }

    async fn send(&mut self, resource: Resource<Channel>, message: String) -> Result<()> {
        let channel_id = self.resource_table.get(&resource)?.channel_id;

        let message = InstanceMessage {
            instance_id: self.instance_id,
            channel_id,
            message,
        };

        self.sender.send(message).await?;

        Ok(())
    }

    async fn receive(&mut self, resource: Resource<Channel>) -> Result<String, wasmtime::Error> {
        let channel_id = self.resource_table.get(&resource)?.channel_id;

        let message = self.receiver.recv().await;

        if let Some(message) = message {
            if message.channel_id == channel_id && message.instance_id == self.instance_id {
                return Ok(message.message);
            }
        }

        Ok("".to_string())
    }

    async fn drop(&mut self, resource: Resource<Channel>) -> Result<()> {
        let _ = self.resource_table.delete(resource)?;
        Ok(())
    }
}

impl spi::lm::kvcache::Host for InstanceState {}
