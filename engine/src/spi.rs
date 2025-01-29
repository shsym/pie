use std::future::Future;
use wasmtime::component::Resource;
use wasmtime::component::{bindgen, ResourceTable};
use wasmtime::Result;
use wasmtime_wasi::{WasiCtx, WasiCtxBuilder, WasiView};

bindgen!({
    path: "../spi/app/wit",
    world: "app",
    async: true,
    with: {
        "spi:lm/inference/language-model": LanguageModel
    },
    // Interactions with `ResourceTable` can possibly trap so enable the ability
    // to return traps from generated functions.
    trappable_imports: true,
});

pub struct ComponentRunStates {
    // These two are required basically as a standard way to enable the impl of WasiView
    pub wasi_ctx: WasiCtx,
    pub resource_table: ResourceTable,
}

impl WasiView for ComponentRunStates {
    fn table(&mut self) -> &mut ResourceTable {
        &mut self.resource_table
    }
    fn ctx(&mut self) -> &mut WasiCtx {
        &mut self.wasi_ctx
    }
}

impl ComponentRunStates {
    pub fn new() -> Self {
        let mut builder = WasiCtx::builder();
        builder.inherit_stderr().inherit_network().inherit_stdout();

        ComponentRunStates {
            wasi_ctx: builder.build(),
            resource_table: ResourceTable::new(),
        }
    }
}

pub struct LanguageModel {
    model_id: String,
}

impl spi::lm::inference::Host for ComponentRunStates {}
impl spi::lm::inference::HostLanguageModel for ComponentRunStates {
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

    async fn drop(&mut self, rep: Resource<LanguageModel>) -> Result<()> {
        Ok(())
    }
}
//
impl spi::app::system::Host for ComponentRunStates {
    async fn ask(&mut self, question: String) -> Result<String, wasmtime::Error> {
        // print the question, and randomly return an answer
        println!("Asked: {}", question);
        Ok("My answer is yolo!".to_string())
    }

    async fn tell(&mut self, message: String) -> Result<()> {
        // print the message
        println!("Told: {}", message);
        Ok(())
    }
}

impl spi::lm::kvcache::Host for ComponentRunStates {}
