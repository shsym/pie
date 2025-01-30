use wasmtime::component::Resource;
use wasmtime::component::{bindgen, ResourceTable};
use wasmtime::Result;
use wasmtime_wasi::{WasiCtx, WasiCtxBuilder, WasiView};

use tokio::sync::mpsc::{Receiver, Sender};
use uuid::Uuid;

bindgen!({
    path: "../spi/app/wit",
    world: "app",
    async: true,
    with: {
        "spi:lm/inference/language-model": LanguageModel,
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
    pub inst2server: Sender<InstanceMessage>,
    pub server2inst: Receiver<InstanceMessage>,
}

pub struct InstanceMessage {
    pub instance_id: Uuid,
    pub dest_id: u32,
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
        inst2server: Sender<InstanceMessage>,
        server2inst: Receiver<InstanceMessage>,
    ) -> Self {
        let mut builder = WasiCtx::builder();
        builder.inherit_stderr().inherit_network().inherit_stdout();

        InstanceState {
            instance_id,
            wasi_ctx: builder.build(),
            resource_table: ResourceTable::new(),
            inst2server,
            server2inst,
        }
    }
}

pub struct LanguageModel {
    model_id: String,
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

    async fn send(&mut self, dest_id: u32, message: String) -> Result<()> {
        self.inst2server
            .send(InstanceMessage {
                instance_id: self.instance_id,
                dest_id,
                message,
            })
            .await?;

        Ok(())
    }

    async fn receive(&mut self) -> Result<String, wasmtime::Error> {
        if let Some(message) = self.server2inst.recv().await {
            return Ok(message.message);
        }

        Ok("".to_string())
    }
}

impl spi::lm::kvcache::Host for InstanceState {}
