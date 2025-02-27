use std::sync::Arc;
use wasmtime::Result;
use wasmtime::component::{ResourceTable, bindgen};
use wasmtime_wasi::{IoView, WasiCtx, WasiCtxBuilder, WasiView};

use crate::tokenizer::BytePairEncoder;
use crate::{driver_l4m, lm, object};
use tokio::sync::mpsc::{Receiver, Sender, UnboundedSender};
use tokio::sync::oneshot;
use uuid::Uuid;

bindgen!({
    path: "../api/app/wit",
    world: "app",
    async: true,
    // with: {
    //     "spi:lm/inference/language-model": LanguageModel,
    //     "spi:lm/inference/token-distribution": TokenDistribution,
    //     "spi:lm/kvcache/token": CachedToken,
    //     "spi:lm/kvcache/token-list": CachedTokenList,
    // },
    // Interactions with `ResourceTable` can possibly trap so enable the ability
    // to return traps from generated functions.
    trappable_imports: true,
});

pub type Id = Uuid;

pub struct InstanceState {
    id: Id,

    wasi_ctx: WasiCtx,
    resource_table: ResourceTable,

    cmd_buffer: UnboundedSender<(Id, Command)>,

    evt_from_origin: Receiver<String>,
    evt_from_peers: Receiver<(String, String)>,

    allocator: driver_l4m::IdPool,

    l4m_driver_utils: Arc<driver_l4m::Utils>,
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
        emb: object::Id<lm::TokenDist>,
        k: u32,
        handle: oneshot::Sender<Vec<u32>>,
    },

    GetTokenDist {
        stream: u32,
        dist: object::Id<lm::TokenDist>,
        handle: oneshot::Sender<Vec<f32>>,
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

impl InstanceState {
    pub async fn new(
        id: Uuid,
        cmd_buffer: UnboundedSender<(Id, Command)>,
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
            cmd_buffer,
            evt_from_origin,
            evt_from_peers,
            allocator: driver_l4m::IdPool::new(1000, 1000, 1000),
            l4m_driver_utils,
        }
    }
}

impl Drop for InstanceState {
    fn drop(&mut self) {
        self.cmd_buffer.send((self.id, Command::DestroyInstance));
    }
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

impl spi::lm::inference::Host for InstanceState {
    async fn get_block_size(&mut self) -> Result<u32, wasmtime::Error> {
        Ok(self.l4m_driver_utils.block_size)
    }

    async fn allocate_blocks(
        &mut self,
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

    async fn deallocate_blocks(
        &mut self,
        stream: u32,
        ids: Vec<object::IdRepr>,
    ) -> Result<(), wasmtime::Error> {
        let blocks = object::Id::map_from_repr(ids);
        self.allocator.release_many(&blocks)?;
        let cmd = Command::DeallocateBlocks { stream, blocks };

        self.cmd_buffer.send((self.id, cmd));
        Ok(())
    }

    async fn allocate_embeds(
        &mut self,
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

    async fn deallocate_embeds(
        &mut self,
        stream: u32,
        ids: Vec<object::IdRepr>,
    ) -> Result<(), wasmtime::Error> {
        let embs = object::Id::<lm::TokenEmb>::map_from_repr(ids);
        self.allocator.release_many(&embs)?;

        let cmd = Command::DeallocateEmb { stream, embs };
        self.cmd_buffer.send((self.id, cmd));
        Ok(())
    }

    async fn allocate_dists(
        &mut self,
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

    async fn deallocate_dists(
        &mut self,
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

    async fn copy_block(
        &mut self,
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

    async fn get_all_exported_blocks(&mut self) -> Result<Vec<(String, u32)>, wasmtime::Error> {
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

    async fn decode_token_dist(
        &mut self,
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
        stream: u32,
        embs: Vec<object::IdRepr>,
        k: u32,
    ) -> Result<Vec<Vec<u32>>, wasmtime::Error> {
        // create a vector of oneshot channels
        //let start = std::time::Instant::now();

        let mut receivers = Vec::with_capacity(embs.len());
        for i in 0..embs.len() {
            let (tx, rx) = oneshot::channel();

            receivers.push(rx);

            let cmd = Command::SampleTopK {
                stream,
                emb: object::Id::new(embs[i]),
                k,
                handle: tx,
            };

            self.cmd_buffer.send((self.id, cmd));
        }

        let mut results = Vec::with_capacity(embs.len());
        for rx in receivers {
            let result = rx
                .await
                .or(Err(wasmtime::Error::msg("SampleTopK failed")))?;
            results.push(result);
        }

        // let duration = start.elapsed();
        // println!("SampleTopK took: {:?}", duration);

        Ok(results)
    }

    async fn get_token_dist(
        &mut self,
        stream: u32,
        dist: object::IdRepr,
    ) -> Result<Vec<f32>, wasmtime::Error> {
        let (tx, rx) = oneshot::channel();

        let cmd = Command::GetTokenDist {
            stream,
            dist: object::Id::new(dist),
            handle: tx,
        };
        self.cmd_buffer.send((self.id, cmd));

        let result = rx
            .await
            .or(Err(wasmtime::Error::msg("GetTokenDist failed")))?;

        Ok(result)
    }

    async fn tokenize(&mut self, text: String) -> Result<Vec<u32>, wasmtime::Error> {
        let tokens = self
            .l4m_driver_utils
            .tokenizer
            .encode_with_special_tokens(text.as_str());
        Ok(tokens)
    }

    async fn detokenize(&mut self, tokens: Vec<u32>) -> Result<String, wasmtime::Error> {
        let text = self.l4m_driver_utils.tokenizer.decode(tokens.as_slice())?;
        Ok(text)
    }
}
