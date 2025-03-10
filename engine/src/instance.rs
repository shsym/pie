use crate::instance::spi::app::l4m::HostSampleTopKResult;
use crate::tokenizer::BytePairEncoder;
use crate::utils::IdPool;
use crate::{driver_l4m, lm, object};
use dashmap::DashMap;
use std::fmt::Debug;
use std::mem;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc::{Receiver, Sender, UnboundedSender};
use tokio::sync::oneshot;
use uuid::Uuid;
use wasmtime::Result;
use wasmtime::component::{Resource, ResourceTable, bindgen};
use wasmtime_wasi::{
    DynPollable, IoView, Pollable, WasiCtx, WasiCtxBuilder, WasiView, async_trait, bindings,
    subscribe,
};
use wasmtime_wasi_http::{WasiHttpCtx, WasiHttpView};
//use wasmtime_wasi_io::poll::{};

bindgen!({
    path: "../api/wit",
    world: "app",
    async: true,
    with: {
        "wasi:io/poll": wasmtime_wasi::bindings::io::poll,
        "spi:app/l4m/sample-top-k-result": SampleTopKResult,
        "spi:app/l4m/model": Model,
        //"spi:app/l4m/echo-result": EchoResult,
        // "spi:lm/inference/token-distribution": TokenDistribution,
        // "spi:lm/kvcache/token": CachedToken,
        // "spi:lm/kvcache/token-list": CachedTokenList,
    },
    // Interactions with `ResourceTable` can possibly trap so enable the ability
    // to return traps from generated functions.
    trappable_imports: true,
});

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

    allocator: driver_l4m::IdPool,

    l4m_driver_utils: Arc<driver_l4m::Utils>,

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

    Ping {
        message: String,
        handle: oneshot::Sender<String>,
    },

    // Block commands -------------------------------------
    AllocateBlocks {
        stream: u32,
        blocks: Vec<object::Id<lm::KvBlock>>,
    },

    DeallocateBlocks {
        stream: u32,
        blocks: Vec<object::Id<lm::KvBlock>>,
    },

    FillBlock {
        stream: u32,
        block: object::Id<lm::KvBlock>,
        context: Vec<object::Id<lm::KvBlock>>,
        inputs: Vec<object::Id<lm::TokenEmb>>,
        outputs: Vec<object::Id<lm::TokenEmb>>,
    },

    ExportBlocks {
        blocks: Vec<object::Id<lm::KvBlock>>,
        resource_name: String,
    },

    ImportBlocks {
        blocks: Vec<object::Id<lm::KvBlock>>,
        resource_name: String,
    },

    GetAllExportedBlocks {
        handle: oneshot::Sender<Vec<(String, u32)>>,
    },

    CopyBlock {
        stream: u32,
        src_block: object::Id<lm::KvBlock>,
        dst_block: object::Id<lm::KvBlock>,
        src_token_offset: u32,
        dst_token_offset: u32,
        size: u32,
    },

    MaskBlock {
        stream: u32,
        block: object::Id<lm::KvBlock>,
        mask: Vec<bool>,
    },

    // Embed ctrl
    AllocateEmb {
        stream: u32,
        embs: Vec<object::Id<lm::TokenEmb>>,
    },

    DeallocateEmb {
        stream: u32,
        embs: Vec<object::Id<lm::TokenEmb>>,
    },

    EmbedText {
        stream: u32,
        embs: Vec<object::Id<lm::TokenEmb>>,
        text: Vec<u32>,
        positions: Vec<u32>,
    },

    EmbedImage {
        stream: u32,
        embs: Vec<object::Id<lm::TokenEmb>>,
        image: String,
    },

    // Output emb ctrl
    AllocateDist {
        stream: u32,
        dists: Vec<object::Id<lm::TokenDist>>,
    },

    DeallocateDist {
        stream: u32,
        dists: Vec<object::Id<lm::TokenDist>>,
    },

    DecodeTokenDist {
        stream: u32,
        embs: Vec<object::Id<lm::TokenEmb>>,
        dists: Vec<object::Id<lm::TokenDist>>,
    },

    SampleTopK {
        stream: u32,
        dist: object::Id<lm::TokenDist>,
        k: u32,
        handle: oneshot::Sender<(Vec<u32>, Vec<f32>)>,
    },
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
            resource_ids: IdPool::new(u32::MAX),
            ready_resources: Arc::new(ReadyResources {
                sample_top_k: DashMap::new(),
            }),
        }
    }
}

impl Drop for InstanceState {
    fn drop(&mut self) {
        self.cmd_buffer.send((self.id, Command::DestroyInstance));
    }
}

pub struct Model {
    name: String,
}

//
impl spi::app::system::Host for InstanceState {
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

impl spi::app::ping::Host for InstanceState {
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

impl HostSampleTopKResult for InstanceState {
    async fn subscribe(
        &mut self,
        this: Resource<SampleTopKResult>,
    ) -> Result<Resource<DynPollable>> {
        subscribe(self.table(), this)
    }

    async fn get(
        &mut self,
        this: Resource<SampleTopKResult>,
    ) -> Result<Option<Vec<(Vec<u32>, Vec<f32>)>>> {
        let r = self.table().get_mut(&this)?;

        if r.done {
            Ok(Some(r.results.clone()))
        } else {
            Ok(None)
        }
    }

    async fn drop(&mut self, this: Resource<SampleTopKResult>) -> Result<()> {
        //println!("dropped");
        let _ = self.table().delete(this)?;
        Ok(())
    }
}

impl spi::app::l4m::HostModel for InstanceState {
    async fn get_block_size(&mut self, model: Resource<Model>) -> Result<u32, wasmtime::Error> {
        Ok(self.l4m_driver_utils.block_size)
    }

    async fn get_all_adapters(
        &mut self,
        model: Resource<Model>,
    ) -> Result<Vec<String>, wasmtime::Error> {
        todo!()
    }

    async fn allocate_blocks(
        &mut self,
        model: Resource<Model>,
        stream: u32,
        count: u32,
    ) -> Result<Vec<object::IdRepr>, wasmtime::Error> {
        let blocks = self.allocator.acquire_many(count as usize)?;

        // also let the server know
        let cmd = Command::AllocateBlocks {
            stream,
            blocks: blocks.clone(),
        };

        self.cmd_buffer.send((self.id, cmd));
        Ok(object::Id::map_to_repr(blocks))
    }

    async fn allocate_embeds(
        &mut self,
        model: Resource<Model>,

        stream: u32,
        count: u32,
    ) -> Result<Vec<object::IdRepr>, wasmtime::Error> {
        let embs = self
            .allocator
            .acquire_many::<lm::TokenEmb>(count as usize)?;

        let cmd = Command::AllocateEmb {
            stream,
            embs: embs.clone(),
        };

        self.cmd_buffer.send((self.id, cmd));
        Ok(object::Id::map_to_repr(embs))
    }

    async fn allocate_dists(
        &mut self,
        model: Resource<Model>,

        stream: u32,
        count: u32,
    ) -> Result<Vec<object::IdRepr>, wasmtime::Error> {
        let dists = self
            .allocator
            .acquire_many::<lm::TokenDist>(count as usize)?;

        let cmd = Command::AllocateDist {
            stream,
            dists: dists.clone(),
        };
        self.cmd_buffer.send((self.id, cmd));

        Ok(object::Id::map_to_repr(dists))
    }

    async fn deallocate_blocks(
        &mut self,
        model: Resource<Model>,

        stream: u32,
        ids: Vec<object::IdRepr>,
    ) -> Result<(), wasmtime::Error> {
        let blocks = object::Id::map_from_repr(ids);
        self.allocator.release_many(&blocks)?;
        let cmd = Command::DeallocateBlocks { stream, blocks };

        self.cmd_buffer.send((self.id, cmd));
        Ok(())
    }
    async fn deallocate_embeds(
        &mut self,
        model: Resource<Model>,

        stream: u32,
        ids: Vec<object::IdRepr>,
    ) -> Result<(), wasmtime::Error> {
        let embs = object::Id::<lm::TokenEmb>::map_from_repr(ids);
        self.allocator.release_many(&embs)?;

        let cmd = Command::DeallocateEmb { stream, embs };
        self.cmd_buffer.send((self.id, cmd));
        Ok(())
    }

    async fn deallocate_dists(
        &mut self,
        model: Resource<Model>,

        stream: u32,
        ids: Vec<object::IdRepr>,
    ) -> Result<(), wasmtime::Error> {
        let dists = object::Id::<lm::TokenDist>::map_from_repr(ids);
        self.allocator.release_many(&dists)?;

        let cmd = Command::DeallocateDist { stream, dists };

        self.cmd_buffer.send((self.id, cmd));

        Ok(())
    }

    async fn fill_block(
        &mut self,
        model: Resource<Model>,

        stream: u32,
        block: object::IdRepr,
        context: Vec<object::IdRepr>,
        inputs: Vec<object::IdRepr>,
        outputs: Vec<object::IdRepr>,
    ) -> Result<(), wasmtime::Error> {
        let cmd = Command::FillBlock {
            stream,
            block: block.into(),
            context: object::Id::map_from_repr(context),
            inputs: object::Id::map_from_repr(inputs),
            outputs: object::Id::map_from_repr(outputs),
        };

        self.cmd_buffer.send((self.id, cmd));

        Ok(())
    }

    async fn fill_block_with_adapter(
        &mut self,
        model: Resource<Model>,
        stream: u32,
        adapter: String,
        block: object::IdRepr,
        context: Vec<object::IdRepr>,
        inputs: Vec<object::IdRepr>,
        outputs: Vec<object::IdRepr>,
    ) -> Result<(), wasmtime::Error> {
        let cmd = Command::FillBlock {
            stream,
            block: block.into(),
            context: object::Id::map_from_repr(context),
            inputs: object::Id::map_from_repr(inputs),
            outputs: object::Id::map_from_repr(outputs),
        };

        self.cmd_buffer.send((self.id, cmd));

        Ok(())
    }

    async fn copy_block(
        &mut self,
        model: Resource<Model>,

        stream: u32,
        src: object::IdRepr,
        dst: object::IdRepr,
        src_offset: u32,
        dst_offset: u32,
        size: u32,
    ) -> Result<(), wasmtime::Error> {
        let cmd = Command::CopyBlock {
            stream,
            src_block: object::Id::new(src),
            dst_block: object::Id::new(dst),
            src_token_offset: src_offset,
            dst_token_offset: dst_offset,
            size,
        };

        self.cmd_buffer.send((self.id, cmd));

        Ok(())
    }

    async fn mask_block(
        &mut self,
        model: Resource<Model>,

        stream: u32,
        block: object::IdRepr,
        mask: Vec<u32>,
    ) -> Result<(), wasmtime::Error> {
        let cmd = Command::MaskBlock {
            stream,
            block: object::Id::new(block),
            mask: mask.into_iter().map(|x| x != 0).collect(),
        };

        self.cmd_buffer.send((self.id, cmd));

        Ok(())
    }

    async fn export_blocks(
        &mut self,
        model: Resource<Model>,

        src: Vec<object::IdRepr>,
        name: String,
    ) -> Result<(), wasmtime::Error> {
        let cmd = Command::ExportBlocks {
            blocks: object::Id::map_from_repr(src),
            resource_name: name,
        };

        self.cmd_buffer.send((self.id, cmd));

        Ok(())
    }

    async fn import_blocks(
        &mut self,
        model: Resource<Model>,

        dst: Vec<object::IdRepr>,
        name: String,
    ) -> Result<(), wasmtime::Error> {
        let cmd = Command::ImportBlocks {
            blocks: object::Id::map_from_repr(dst),
            resource_name: name,
        };

        self.cmd_buffer.send((self.id, cmd));

        Ok(())
    }

    async fn get_all_exported_blocks(
        &mut self,
        model: Resource<Model>,
    ) -> Result<Vec<(String, u32)>, wasmtime::Error> {
        let (tx, rx) = oneshot::channel();

        let cmd = Command::GetAllExportedBlocks { handle: tx };
        self.cmd_buffer.send((self.id, cmd));

        let result = rx
            .await
            .or(Err(wasmtime::Error::msg("GetAllExportedBlocks failed")))?;

        Ok(result)
    }

    async fn embed_text(
        &mut self,
        model: Resource<Model>,

        stream: u32,
        embs: Vec<object::IdRepr>,
        tokens: Vec<u32>,
        positions: Vec<u32>,
    ) -> Result<(), wasmtime::Error> {
        let cmd = Command::EmbedText {
            stream,
            embs: object::Id::map_from_repr(embs),
            text: tokens,
            positions,
        };

        self.cmd_buffer.send((self.id, cmd));

        Ok(())
    }

    async fn decode_token_dist(
        &mut self,
        model: Resource<Model>,

        stream: u32,
        embs: Vec<object::IdRepr>,
        dists: Vec<object::IdRepr>,
    ) -> Result<(), wasmtime::Error> {
        let cmd = Command::DecodeTokenDist {
            stream,
            embs: object::Id::map_from_repr(embs),
            dists: object::Id::map_from_repr(dists),
        };

        self.cmd_buffer.send((self.id, cmd));

        Ok(())
    }

    async fn sample_top_k(
        &mut self,
        model: Resource<Model>,

        stream: u32,
        embs: Vec<object::IdRepr>,
        k: u32,
    ) -> Result<Resource<SampleTopKResult>, wasmtime::Error> {
        // create a vector of oneshot channels
        //let start = std::time::Instant::now();

        let mut receivers = Vec::with_capacity(embs.len());
        for i in 0..embs.len() {
            let (tx, rx) = oneshot::channel();

            receivers.push(rx);
            let cmd = Command::SampleTopK {
                stream,
                dist: object::Id::new(embs[i]),
                k,
                handle: tx,
            };

            self.cmd_buffer.send((self.id, cmd));
        }

        let top_k_result = SampleTopKResult {
            receivers,
            results: Vec::new(),
            done: false,
        };

        let res = self.table().push(top_k_result)?;
        Ok(res)
    }

    // async fn sample_top_k_value(
    //     &mut self,
    //     handle: u32,
    // ) -> Result<Option<Vec<(Vec<u32>, Vec<f32>)>>, wasmtime::Error> {
    //     if let Some((handle, res)) = self.ready_resources.sample_top_k.remove(&handle) {
    //         self.resource_ids.release(handle);
    //         Ok(Some(res))
    //     } else {
    //         Ok(None)
    //     }
    // }

    async fn tokenize(
        &mut self,
        model: Resource<Model>,
        text: String,
    ) -> Result<Vec<u32>, wasmtime::Error> {
        let tokens = self
            .l4m_driver_utils
            .tokenizer
            .encode_with_special_tokens(text.as_str());
        Ok(tokens)
    }

    async fn detokenize(
        &mut self,
        model: Resource<Model>,
        tokens: Vec<u32>,
    ) -> Result<String, wasmtime::Error> {
        let text = self.l4m_driver_utils.tokenizer.decode(tokens.as_slice())?;
        Ok(text)
    }

    async fn get_vocabs(
        &mut self,
        model: Resource<Model>,
    ) -> Result<Vec<Vec<u8>>, wasmtime::Error> {
        let vocabs = self.l4m_driver_utils.tokenizer.get_vocabs();
        Ok(vocabs)
    }
    async fn drop(&mut self, model: Resource<Model>) -> Result<(), wasmtime::Error> {
        Ok(())
    }
}

impl spi::app::l4m::Host for InstanceState {
    async fn get_model(&mut self, value: String) -> Result<Option<Resource<Model>>> {
        todo!()
    }

    async fn get_all_models(&mut self) -> Result<Vec<String>> {
        todo!()
    }
}

impl spi::app::l4m_vision::Host for InstanceState {
    async fn embed_image(
        &mut self,
        stream: u32,
        embs: Vec<object::IdRepr>,
        url: String,
    ) -> Result<(), wasmtime::Error> {
        let cmd = Command::EmbedImage {
            stream,
            embs: object::Id::map_from_repr(embs),
            image: url,
        };

        self.cmd_buffer.send((self.id, cmd));

        Ok(())
    }

    async fn embed_video(
        &mut self,
        stream: u32,
        embs: Vec<object::IdRepr>,
        url: String,
    ) -> Result<(), wasmtime::Error> {
        todo!()
    }
}

pub struct SampleTopKResult {
    receivers: Vec<oneshot::Receiver<(Vec<u32>, Vec<f32>)>>,
    results: Vec<(Vec<u32>, Vec<f32>)>,
    done: bool,
}

#[async_trait]
impl Pollable for SampleTopKResult {
    async fn ready(&mut self) {
        // if results are already computed, return
        if self.done {
            return;
        }

        //println!("SampleTopKResult Polling");
        for rx in &mut self.receivers {
            let result = rx.await.unwrap();
            self.results.push(result);
        }
        self.done = true;
    }
}

struct Deadline {
    done: bool,
}

#[async_trait::async_trait]
impl Pollable for Deadline {
    async fn ready(&mut self) {
        if self.done {
            return;
        }

        let start = std::time::Instant::now();

        // sleep for 1 sec
        tokio::time::sleep(Duration::from_secs(1)).await;

        // match self {
        //     Deadline::Past => {}
        //     Deadline::Instant(instant) => tokio::time::sleep_until(*instant).await,
        //     Deadline::Never => std::future::pending().await,
        // }

        let duration = start.elapsed();
        self.done = true;
        println!("Deadline took: {:?}", duration);
    }
}

//
// impl HostPollable for InstanceState {
//     async fn ready(&mut self, self_: Resource<DynPollable>) -> Result<bool> {
//         todo!()
//     }
//
//     async fn block(&mut self, self_: Resource<DynPollable>) -> Result<()> {
//         todo!()
//     }
//
//     fn drop(&mut self, rep: Resource<DynPollable>) -> Result<()> {
//         todo!()
//     }
// }
//
// impl poll::Host for InstanceState {
//     async fn poll(&mut self, in_: Vec<Resource<DynPollable>>) -> Result<Vec<u32>> {
//         todo!()
//     }
// }

/////
//
// struct Listener<T, F> {
//     receiver: oneshot::Receiver<T>,
//     on_data: F,
// }
// impl<T, F> Listener<T, F>
// where
//     T: Debug + Clone + Send + Sync + 'static,
//     F: Fn(T) + Send + Sync + 'static,
// {
//     fn new(receiver: oneshot::Receiver<T>, on_data: F) -> Self {
//         Listener { receiver, on_data }
//     }
// }
//
// #[async_trait]
// impl<T, F> Pollable for Listener<T, F>
// where
//     T: Debug + Clone + Send + Sync + 'static,
//     F: Fn(T) + Send + Sync + 'static,
// {
//     async fn ready(&mut self) {
//         let result = (&mut self.receiver).await.unwrap();
//
//         (self.on_data)(result);
//     }
// }
//
// struct MultiListener<T, F> {
//     receivers: Vec<oneshot::Receiver<T>>,
//     on_data: F,
//     ss: u32,
// }
// impl<T, F> MultiListener<T, F>
// where
//     T: Debug + Clone + Send + Sync + 'static,
//     F: Fn(Vec<T>) + Send + Sync + 'static,
// {
//     fn new(receivers: Vec<oneshot::Receiver<T>>, on_data: F) -> Self {
//         MultiListener {
//             receivers,
//             on_data,
//             ss: 0,
//         }
//     }
// }
//
// #[async_trait]
// impl<T, F> Pollable for MultiListener<T, F>
// where
//     T: Debug + Clone + Send + Sync + 'static,
//     F: Fn(Vec<T>) + Send + Sync + 'static,
// {
//     async fn ready(&mut self) {
//         self.ss += 1;
//         println!("Pollin!!!!!!!! {:?}", self.ss);
//
//         let results = futures::future::join_all(&mut self.receivers)
//             .await
//             .into_iter()
//             .map(|result| result.unwrap())
//             .collect::<Vec<_>>();
//
//         println!("Done {:?}", results);
//         (self.on_data)(results);
//     }
// }
