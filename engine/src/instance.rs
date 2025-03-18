use dashmap::DashMap;

use uuid::Uuid;
use wasmtime::component::ResourceTable;
use wasmtime_wasi::{IoView, WasiCtx, WasiView};
use wasmtime_wasi_http::{WasiHttpCtx, WasiHttpView};

pub type Id = Uuid;

pub struct InstanceState {
    id: Id,
    wasi_ctx: WasiCtx,
    resource_table: ResourceTable,
    http_ctx: WasiHttpCtx,
}

type ResourceId = u32;
pub struct ReadyResources {
    sample_top_k: DashMap<ResourceId, Vec<(Vec<u32>, Vec<f32>)>>,
}

impl IoView for InstanceState {
    fn table(&mut self) -> &mut ResourceTable {
        &mut self.resource_table
    }
}

impl WasiView for InstanceState {
    fn ctx(&mut self) -> &mut WasiCtx {
        &mut self.wasi_ctx
    }
}

impl WasiHttpView for InstanceState {
    fn ctx(&mut self) -> &mut WasiHttpCtx {
        &mut self.http_ctx
    }
}

impl InstanceState {
    pub async fn new(id: Uuid) -> Self {
        let mut builder = WasiCtx::builder();
        builder.inherit_stderr().inherit_network().inherit_stdout();

        // send construct cmd
        //cmd_buffer.send((id, Command::CreateInstance));

        InstanceState {
            id,
            wasi_ctx: builder.build(),
            resource_table: ResourceTable::new(),
            http_ctx: WasiHttpCtx::new(),
        }
    }
    
    pub fn id(&self) -> Id {
        self.id
    }
}
//
// impl Drop for InstanceState {
//     fn drop(&mut self) {
//         self.cmd_buffer.send((self.id, Command::DestroyInstance));
//     }
// }
