mod l4m;
mod l4m_vision;
mod ping;

use crate::driver::StreamId;
use crate::instance_old::spi::app::l4m::HostSampleTopKResult;
use crate::instance_old::spi::app::l4m_vision::BaseModel;
use crate::tokenizer::BytePairEncoder;
use crate::utils::IdPool;
use crate::{driver, object};
use dashmap::DashMap;
use std::fmt::Debug;
use std::mem;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc::{Receiver, Sender, UnboundedSender};
use tokio::sync::oneshot;
use uuid::Uuid;
use wasmtime::Result;
use wasmtime::component::{Resource, ResourceTable};
use wasmtime_wasi::{
    DynPollable, IoView, Pollable, WasiCtx, WasiCtxBuilder, WasiView, async_trait, bindings,
    subscribe,
};
use wasmtime_wasi_http::{WasiHttpCtx, WasiHttpView};

mod wit_bindings {
    use super::*;

    wasmtime::component::bindgen!({
        path: "../api/wit",
        world: "app",
        async: true,
        with: {
            "wasi:io/poll": wasmtime_wasi::bindings::io::poll,
            "spi:app/l4m/sample-top-k-result": l4m::SampleTopKResult,
            "spi:app/l4m/model": l4m::Model,
            "spi:app/l4m/tokenizer": l4m::Tokenizer,
            "spi:app/l4m-vision/model": super::VisionModel,

        },
        trappable_imports: true,
    });
}
pub type Id = Uuid;

pub struct InstanceState {
    id: Id,

    wasi_ctx: WasiCtx,
    resource_table: ResourceTable,
    http_ctx: WasiHttpCtx,

    cmd_buffer: UnboundedSender<(Id, Command)>,

    evt_from_system: Receiver<String>,
    evt_from_origin: Receiver<String>,
    evt_from_peers: Receiver<(String, String)>,


    l4m_cache:l4m::Cache,

    resource_ids: IdPool<ResourceId>,
    ready_resources: Arc<ReadyResources>,
}

type ResourceId = u32;
pub struct ReadyResources {
    sample_top_k: DashMap<ResourceId, Vec<(Vec<u32>, Vec<f32>)>>,
}

// implements send
#[derive(Debug)]
pub enum Command {
    // Init -------------------------------------
    CreateInstance {
        handle: oneshot::Sender<Arc<driver_l4m::Utils>>,
    },

    DestroyInstance,

    // Communication -------------------------------------
    SendToOrigin {
        message: String,
    },

    BroadcastToPeers {
        topic: String,
        message: String,
    },

    Subscribe {
        topic: String,
    },

    Unsubscribe {
        topic: String,
    },

    L4m {
        stream: StreamId,
        cmd: driver::l4m::Command,
    },
    L4mVision(driver::l4m_vision::Command),
    Ping(driver::ping::Command),
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
    pub async fn new(
        id: Uuid,
        cmd_buffer: UnboundedSender<(Id, Command)>,
        evt_from_system: Receiver<String>,
        evt_from_origin: Receiver<String>,
        evt_from_peers: Receiver<(String, String)>,
        //l4m_driver_utils: Arc<driver_l4m::Utils>,
    ) -> Self {
        let mut builder = WasiCtx::builder();
        builder.inherit_stderr().inherit_network().inherit_stdout();

        // send construct cmd
        let (tx, rx) = oneshot::channel();
        cmd_buffer.send((id, Command::CreateInstance { handle: tx }));
        let l4m_driver_utils = rx.await.unwrap();

        InstanceState {
            id,
            wasi_ctx: builder.build(),
            resource_table: ResourceTable::new(),
            http_ctx: WasiHttpCtx::new(),
            cmd_buffer,
            evt_from_system,
            evt_from_origin,
            evt_from_peers,
            allocator: driver_l4m::IdPool::new(1000, 1000, 1000),
            l4m_driver_utils,
        }
    }

    fn send_l4m_cmd(
        &self,
        stream: StreamId,
        cmd: driver::l4m::Command,
    ) -> Result<(), wasmtime::Error> {
        self.cmd_buffer
            .send((self.id, Command::L4m { stream, cmd }))
            .map_err(|e| wasmtime::Error::msg(format!("Send error: {}", e)))
    }
}

impl Drop for InstanceState {
    fn drop(&mut self) {
        self.cmd_buffer.send((self.id, Command::DestroyInstance));
    }
}

pub struct VisionModel {
    name: String,
}

//
impl wit_bindings::spi::app::system::Host for InstanceState {
    async fn get_version(&mut self) -> Result<String, wasmtime::Error> {
        Ok("0.1.0".to_string())
    }

    async fn get_instance_id(&mut self) -> Result<String, wasmtime::Error> {
        Ok(self.id.to_string())
    }

    async fn send_to_origin(&mut self, message: String) -> Result<(), wasmtime::Error> {
        self.cmd_buffer
            .send((self.id, Command::SendToOrigin { message }));
        Ok(())
    }

    async fn receive_from_origin(&mut self) -> Result<String, wasmtime::Error> {
        self.evt_from_origin
            .recv()
            .await
            .ok_or(wasmtime::Error::msg("No more events"))
    }

    async fn broadcast_to_peers(
        &mut self,
        topic: String,
        message: String,
    ) -> Result<(), wasmtime::Error> {
        self.cmd_buffer
            .send((self.id, Command::BroadcastToPeers { topic, message }));
        Ok(())
    }

    async fn receive_from_peers(&mut self) -> Result<(String, String), wasmtime::Error> {
        self.evt_from_peers
            .recv()
            .await
            .ok_or(wasmtime::Error::msg("No more events"))
    }

    async fn subscribe(&mut self, topic: String) -> Result<(), wasmtime::Error> {
        self.cmd_buffer
            .send((self.id, Command::Subscribe { topic }));
        Ok(())
    }

    async fn unsubscribe(&mut self, topic: String) -> Result<(), wasmtime::Error> {
        self.cmd_buffer
            .send((self.id, Command::Unsubscribe { topic }));
        Ok(())
    }
}

impl wit_bindings::spi::app::ping::Host for InstanceState {
    async fn ping(&mut self, message: String) -> Result<String, wasmtime::Error> {
        let (tx, rx) = oneshot::channel();

        self.cmd_buffer.send((
            self.id,
            Command::Ping {
                message,
                handle: tx,
            },
        ));

        let result = rx.await.or(Err(wasmtime::Error::msg("Ping failed")))?;
        Ok(result)
    }
}

impl wit_bindings::spi::app::l4m_vision::Host for InstanceState {
    async fn extend_model(
        &mut self,
        model: Resource<l4m::Model>,
    ) -> Result<Option<Resource<VisionModel>>, wasmtime::Error> {
        let vision_model = VisionModel {
            name: "".to_string(),
        };

        let res = self.table().push(vision_model)?;

        Ok(Some(res))
    }
    //
    // async fn embed_image(
    //     &mut self,
    //     model: Resource<Model>,
    //     stream: u32,
    //     embs: Vec<object::IdRepr>,
    //     image_blob: Vec<u8>,
    // ) -> Result<(), wasmtime::Error> {
    //     let cmd = Command::EmbedImage {
    //         stream,
    //         embs: object::Id::map_from_repr(embs),
    //         image: "url".to_string(),
    //     };
    //
    //     self.cmd_buffer.send((self.id, cmd));
    //
    //     Ok(())
    // }
}

impl wit_bindings::spi::app::l4m_vision::HostModel for InstanceState {
    async fn get_base_model(
        &mut self,
        self_: Resource<VisionModel>,
    ) -> Result<Resource<l4m::Model>> {
        todo!()
    }

    async fn embed_image(
        &mut self,
        self_: Resource<VisionModel>,
        stream_id: u32,
        embs: Vec<u32>,
        image_blob: Vec<u8>,
    ) -> Result<()> {
        todo!()
    }

    async fn drop(&mut self, rep: Resource<VisionModel>) -> Result<()> {
        todo!()
    }
}
