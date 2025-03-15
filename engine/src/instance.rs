use crate::driver::{AnyCommand, DynCommand};
use crate::tokenizer::BytePairEncoder;
use crate::utils::IdPool;
use crate::{driver, object};
use dashmap::DashMap;
use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;
use std::mem;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;
use wasmtime::Result;
use wasmtime::component::{Resource, ResourceTable};
use wasmtime_wasi::{
    DynPollable, IoView, Pollable, WasiCtx, WasiCtxBuilder, WasiView, async_trait, subscribe,
};
use wasmtime_wasi_http::{WasiHttpCtx, WasiHttpView};

pub type Id = Uuid;

pub struct InstanceState {
    id: Id,

    wasi_ctx: WasiCtx,
    resource_table: ResourceTable,
    http_ctx: WasiHttpCtx,

    cmd_buffer: UnboundedSender<(Id, AnyCommand)>,
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
    pub async fn new(id: Uuid, cmd_buffer: UnboundedSender<(Id, AnyCommand)>) -> Self {
        let mut builder = WasiCtx::builder();
        builder.inherit_stderr().inherit_network().inherit_stdout();

        // send construct cmd
        //cmd_buffer.send((id, Command::CreateInstance));

        InstanceState {
            id,
            wasi_ctx: builder.build(),
            resource_table: ResourceTable::new(),
            http_ctx: WasiHttpCtx::new(),
            cmd_buffer,
        }
    }

    pub fn send_cmd<T>(&self, cmd: T) -> Result<(), wasmtime::Error>
    where
        T: Send,
    {
        let dyn_cmd = AnyCommand::new(cmd);
        self.cmd_buffer
            .send((self.id, dyn_cmd))
            .map_err(|e| wasmtime::Error::msg(format!("Send error: {}", e)))
    }
}
//
// impl Drop for InstanceState {
//     fn drop(&mut self) {
//         self.cmd_buffer.send((self.id, Command::DestroyInstance));
//     }
// }
