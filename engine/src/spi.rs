use std::sync::Arc;
use wasmtime::component::{bindgen, ResourceTable};
use wasmtime::Result;
use wasmtime_wasi::{WasiCtx, WasiCtxBuilder, WasiView};

use crate::object;
use crate::backend::InstanceId;
use crate::tokenizer::BytePairEncoder;
use tokio::sync::mpsc::{Receiver, Sender};
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

pub struct InstanceState {
    id: InstanceId,

    wasi_ctx: WasiCtx,
    resource_table: ResourceTable,

    cmd_buffer: Sender<(InstanceId, Command)>,

    evt_from_origin: Receiver<String>,
    evt_from_peers: Receiver<(String, String)>,

    allocator: object::IdPool,

    utils: InstanceUtils,
}

pub struct InstanceUtils {
    pub tokenizer: Arc<BytePairEncoder>,
}

// implements send
pub enum Command {
    // Communication
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

    // Block commands
    AllocateBlocks {
        stream: u32,
        blocks: Vec<object::Id>,
    },

    DeallocateBlocks {
        stream: u32,
        blocks: Vec<object::Id>,
    },

    FillBlocks {
        stream: u32,
        blocks: Vec<object::Id>,
        context: Vec<object::Id>,
        inputs: Vec<object::Id>,
        outputs: Vec<Option<object::Id>>,
    },

    ExportBlocks {
        blocks: Vec<object::Id>,
        resource_name: String,
    },

    ImportBlocks {
        blocks: Vec<object::Id>,
        resource_name: String,
    },

    CopyBlock {
        stream: u32,
        src_block: object::Id,
        dst_block: object::Id,
        src_token_offset: u32,
        dst_token_offset: u32,
        size: u32,
    },

    MaskBlock {
        stream: u32,
        block: object::Id,
        mask: Vec<bool>,
    },

    // Embed ctrl
    AllocateEmb {
        stream: u32,
        embs: Vec<object::Id>,
    },

    DeallocateEmb {
        stream: u32,
        embs: Vec<object::Id>,
    },

    EmbedText {
        stream: u32,
        embs: Vec<object::Id>,
        text: Vec<u32>,
    },

    EmbedImage {
        stream: u32,
        embs: Vec<object::Id>,
        image: String,
    },

    // Output emb ctrl
    AllocateDist {
        stream: u32,
        dists: Vec<object::Id>,
    },

    DeallocateDist {
        stream: u32,
        dists: Vec<object::Id>,
    },

    DecodeTokenDist {
        stream: u32,
        embs: Vec<object::Id>,
        dists: Vec<object::Id>,
    },

    SampleTopK {
        stream: u32,
        emb: object::Id,
        k: u32,
        handle: oneshot::Sender<Vec<u32>>,
    },

    GetTokenDist {
        stream: u32,
        dist: object::Id,
        handle: oneshot::Sender<Vec<f32>>,
    },
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
        id: Uuid,
        cmd_buffer: Sender<(InstanceId, Command)>,
        evt_from_origin: Receiver<String>,
        evt_from_peers: Receiver<(String, String)>,
        utils: InstanceUtils,
    ) -> Self {
        let mut builder = WasiCtx::builder();
        builder.inherit_stderr().inherit_network().inherit_stdout();

        InstanceState {
            id,
            wasi_ctx: builder.build(),
            resource_table: ResourceTable::new(),
            cmd_buffer,
            evt_from_origin,
            evt_from_peers,
            allocator: object::IdPool::new(1000, 1000),
            utils,
        }
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
            .send((self.id, Command::SendToOrigin { message }))
            .await?;
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
            .send((self.id, Command::BroadcastToPeers { topic, message }))
            .await?;
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
            .send((self.id, Command::Subscribe { topic }))
            .await?;
        Ok(())
    }
}

impl spi::lm::inference::Host for InstanceState {
    async fn allocate(
        &mut self,
        stream: u32,
        obj: spi::lm::inference::Object,
        count: u32,
    ) -> Result<Vec<u32>, wasmtime::Error> {
        let mut ids = Vec::with_capacity(count as usize);

        let ns = obj.to_namespace();

        for _ in 0..count {
            let id = self
                .allocator
                .acquire(ns)
                .or(Err(wasmtime::Error::msg("No more free blocks")))?;

            ids.push(id);
        }

        // also let the server know
        let cmd = Command::AllocateBlocks {
            stream,
            blocks: ids.clone(),
        };

        self.cmd_buffer.send((self.id, cmd)).await?;

        Ok(ids)
    }

    async fn deallocate(
        &mut self,
        stream: u32,
        obj: spi::lm::inference::Object,
        ids: Vec<u32>,
    ) -> Result<(), wasmtime::Error> {
        let ns = obj.to_namespace();

        for id in ids.iter().copied() {
            self.allocator
                .release(ns, id)
                .or(Err(wasmtime::Error::msg("Invalid ID")))?;
        }

        let cmd = Command::DeallocateBlocks {
            stream,
            blocks: ids,
        };

        self.cmd_buffer.send((self.id, cmd)).await?;

        // self.cmd_buffer
        //     .send(Msg {
        //         instance_id: self.instance_id,
        //         cmd,
        //     })
        //     .await?;

        Ok(())
    }

    async fn fill_blocks(
        &mut self,
        stream: u32,
        blocks: Vec<u32>,
        context: Vec<u32>,
        inputs: Vec<u32>,
        outputs: Vec<u32>,
    ) -> Result<(), wasmtime::Error> {
        let cmd = Command::FillBlocks {
            stream,
            blocks,
            context,
            inputs,
            outputs: outputs.into_iter().map(|x| Some(x)).collect(),
        };

        self.cmd_buffer.send((self.id, cmd)).await?;

        Ok(())
    }

    async fn copy_block(
        &mut self,
        stream: u32,
        src: u32,
        dst: u32,
        src_offset: u32,
        dst_offset: u32,
        size: u32,
    ) -> Result<(), wasmtime::Error> {
        let cmd = Command::CopyBlock {
            stream,
            src_block: src,
            dst_block: dst,
            src_token_offset: src_offset,
            dst_token_offset: dst_offset,
            size,
        };

        self.cmd_buffer.send((self.id, cmd)).await?;

        Ok(())
    }

    async fn mask_block(
        &mut self,
        stream: u32,
        block: u32,
        mask: Vec<u32>,
    ) -> Result<(), wasmtime::Error> {
        let cmd = Command::MaskBlock {
            stream,
            block,
            mask: mask.into_iter().map(|x| x != 0).collect(),
        };

        self.cmd_buffer.send((self.id, cmd)).await?;

        Ok(())
    }

    async fn export_blocks(&mut self, src: Vec<u32>, name: String) -> Result<(), wasmtime::Error> {
        let cmd = Command::ExportBlocks {
            blocks: src,
            resource_name: name,
        };

        self.cmd_buffer.send((self.id, cmd)).await?;

        Ok(())
    }

    async fn import_blocks(&mut self, dst: Vec<u32>, name: String) -> Result<(), wasmtime::Error> {
        let cmd = Command::ImportBlocks {
            blocks: dst,
            resource_name: name,
        };

        self.cmd_buffer.send((self.id, cmd)).await?;

        Ok(())
    }

    async fn embed_text(
        &mut self,
        stream: u32,
        embs: Vec<u32>,
        tokens: Vec<u32>,
    ) -> Result<(), wasmtime::Error> {
        let cmd = Command::EmbedText {
            stream,
            embs,
            text: tokens,
        };

        self.cmd_buffer.send((self.id, cmd)).await?;

        Ok(())
    }

    async fn embed_image(
        &mut self,
        stream: u32,
        embs: Vec<u32>,
        url: String,
    ) -> Result<(), wasmtime::Error> {
        let cmd = Command::EmbedImage {
            stream,
            embs,
            image: url,
        };

        self.cmd_buffer.send((self.id, cmd)).await?;

        Ok(())
    }

    async fn embed_video(
        &mut self,
        stream: u32,
        embs: Vec<u32>,
        url: String,
    ) -> Result<(), wasmtime::Error> {
        todo!()
    }

    async fn decode_token_dist(
        &mut self,
        stream: u32,
        embs: Vec<u32>,
        dists: Vec<u32>,
    ) -> Result<(), wasmtime::Error> {
        let cmd = Command::DecodeTokenDist {
            stream,
            embs,
            dists,
        };

        self.cmd_buffer.send((self.id, cmd)).await?;

        Ok(())
    }

    async fn sample_top_k(
        &mut self,
        stream: u32,
        embs: Vec<u32>,
        k: u32,
    ) -> Result<Vec<u32>, wasmtime::Error> {
        let cmd = Command::SampleTopK {
            stream,
            emb: embs[0],
            k,
            handle: oneshot::channel().0,
        };

        let (tx, rx) = oneshot::channel();

        self.cmd_buffer.send((self.id, cmd)).await?;

        let result = rx
            .await
            .or(Err(wasmtime::Error::msg("SampleTopK failed")))?;

        Ok(result)
    }

    async fn get_token_dist(
        &mut self,
        stream: u32,
        dist: u32,
    ) -> Result<Vec<f32>, wasmtime::Error> {
        let cmd = Command::GetTokenDist {
            stream,
            dist,
            handle: oneshot::channel().0,
        };

        let (tx, rx) = oneshot::channel();

        self.cmd_buffer.send((self.id, cmd)).await?;

        let result = rx
            .await
            .or(Err(wasmtime::Error::msg("GetTokenDist failed")))?;

        Ok(result)
    }

    async fn tokenize(&mut self, text: String) -> Result<Vec<u32>, wasmtime::Error> {
        let tokens = self
            .utils
            .tokenizer
            .encode_with_special_tokens(text.as_str());

        Ok(tokens)
    }

    async fn detokenize(&mut self, tokens: Vec<u32>) -> Result<String, wasmtime::Error> {
        let text = self.utils.tokenizer.decode(tokens.as_slice())?;

        Ok(text)
    }
}

impl spi::lm::inference::Object {
    fn to_namespace(&self) -> object::Namespace {
        match self {
            spi::lm::inference::Object::KvBlock => object::Namespace::KvBlock,
            spi::lm::inference::Object::Emb => object::Namespace::Emb,
            spi::lm::inference::Object::Dist => object::Namespace::Dist,
        }
    }
}

// impl spi::lm::inference::HostLanguageModel for InstanceState {
//     async fn new(&mut self, model_id: String) -> Result<Resource<LanguageModel>, wasmtime::Error> {
//         let handle = LanguageModel { model_id };
//         Ok(self.resource_table.push(handle)?)
//     }
//
//     async fn tokenize(
//         &mut self,
//         resource: Resource<LanguageModel>,
//         text: String,
//     ) -> Result<Vec<u32>, wasmtime::Error> {
//         Ok(vec![0])
//     }
//
//     async fn detokenize(
//         &mut self,
//         resource: Resource<LanguageModel>,
//         tokens: Vec<u32>,
//     ) -> Result<String, wasmtime::Error> {
//         Ok("Hello".to_string())
//     }
//
//     async fn predict(
//         &mut self,
//         resource: Resource<LanguageModel>,
//         cache: Resource<CachedTokenList>,
//         tokens: Vec<u32>,
//     ) -> Result<Vec<Resource<TokenDistribution>>, wasmtime::Error> {
//         Ok(vec![])
//     }
//
//     async fn drop(&mut self, resource: Resource<LanguageModel>) -> Result<()> {
//         let _ = self.resource_table.delete(resource)?;
//
//         Ok(())
//     }
// }
//
// impl spi::lm::inference::HostTokenDistribution for InstanceState {
//     async fn sample_p(
//         &mut self,
//         resource: Resource<TokenDistribution>,
//     ) -> Result<u32, wasmtime::Error> {
//         Ok(0)
//     }
//
//     async fn top_k(
//         &mut self,
//         resource: Resource<TokenDistribution>,
//         k: u32,
//     ) -> Result<Vec<u32>, wasmtime::Error> {
//         Ok(vec![1, 2])
//     }
//
//     async fn drop(&mut self, resource: Resource<TokenDistribution>) -> Result<()> {
//         let _ = self.resource_table.delete(resource)?;
//
//         Ok(())
//     }
// }
//
// impl spi::lm::kvcache::Host for InstanceState {}
//
// impl spi::lm::kvcache::HostToken for InstanceState {
//     async fn position(&mut self, resource: Resource<CachedToken>) -> Result<u32, wasmtime::Error> {
//         Ok(0)
//     }
//
//     async fn token_id(&mut self, resource: Resource<CachedToken>) -> Result<u32, wasmtime::Error> {
//         Ok(1)
//     }
//
//     async fn drop(&mut self, resource: Resource<CachedToken>) -> Result<()> {
//         let _ = self.resource_table.delete(resource)?;
//         Ok(())
//     }
// }
//
// impl spi::lm::kvcache::HostTokenList for InstanceState {
//     //
//     //
//     /*
//         constructor(tokens: list<token>);
//
//     // mutating methods
//     push: func(token: token);
//     pop: func() -> token;
//     extend: func(tokens: token-list);
//     splice: func(start: u32, delete-count: u32, tokens: token-list);
//
//     // non-mutating methods
//     length: func() -> u32;
//     slice: func(start: u32, end: u32) -> token-list;
//     concat: func(cache: token-list) -> token-list;
//     index: func(position: u32) -> token;
//      */
//
//     //
//     // 1) Constructor
//     //
//     //    WIT signature (approx):
//     //    constructor(tokens: list<token>) -> token-list
//     //
//     //    * `tokens` here is a Vec<Resource<CachedToken>> from the generated code.
//     //
//     async fn new(
//         &mut self,
//         tokens: Vec<Resource<CachedToken>>,
//     ) -> Result<Resource<CachedTokenList>> {
//         // Collect actual CachedToken data from each resource
//         let mut list_data = Vec::with_capacity(tokens.len());
//         for token_resource in tokens {
//             // Get a reference to the cached token from the table:
//             let token_ref = self.resource_table.get(&token_resource)?;
//             // Clone it if you plan to store a copy
//             list_data.push(token_ref.clone());
//         }
//
//         // Create a new CachedTokenList resource
//         let token_list = CachedTokenList { tokens: list_data };
//         let resource_handle = self.resource_table.push(token_list)?;
//         Ok(resource_handle)
//     }
//
//     //
//     // 2) push: func(token: token);
//     //
//     //    * Mutates the list by pushing a new token onto it.
//     //    * The `token` is a Resource<CachedToken>.
//     //
//     async fn push(
//         &mut self,
//         list_resource: Resource<CachedTokenList>,
//         token_resource: Resource<CachedToken>,
//     ) -> Result<()> {
//         let token_ref = self.resource_table.get(&token_resource)?.clone();
//         let token_list = self.resource_table.get_mut(&list_resource)?;
//         token_list.tokens.push(token_ref.clone());
//         Ok(())
//     }
//
//     //
//     // 3) pop: func() -> token;
//     //
//     //    * Mutates the list by popping the last token
//     //      and returns it as a fresh Resource<CachedToken>.
//     //
//     async fn pop(
//         &mut self,
//         list_resource: Resource<CachedTokenList>,
//     ) -> Result<Resource<CachedToken>> {
//         let token_list = self.resource_table.get_mut(&list_resource)?;
//
//         let popped = token_list
//             .tokens
//             .pop()
//             .ok_or_else(|| anyhow::anyhow!("Cannot pop from an empty list"))?;
//
//         // Push the popped token into the resource table so the caller can use it
//         let popped_resource = self.resource_table.push(popped)?;
//         Ok(popped_resource)
//     }
//
//     //
//     // 4) extend: func(tokens: token-list);
//     //
//     //    * Mutates the `list_resource` by extending with all the tokens
//     //      from `other_list_resource`.
//     //    * We do NOT remove them from the `other_list`; we just copy them.
//     //
//     async fn extend(
//         &mut self,
//         list_resource: Resource<CachedTokenList>,
//         other_list_resource: Resource<CachedTokenList>,
//     ) -> Result<()> {
//         Ok(())
//     }
//
//     //
//     // 5) splice: func(start: u32, delete_count: u32, tokens: token-list);
//     //
//     //    * Removes `delete_count` items from `list_resource` starting at `start`
//     //      and inserts tokens from `other_list_resource` in their place.
//     //
//     async fn splice(
//         &mut self,
//         list_resource: Resource<CachedTokenList>,
//         start: u32,
//         delete_count: u32,
//         other_list_resource: Resource<CachedTokenList>,
//     ) -> Result<()> {
//         Ok(())
//     }
//
//     //
//     // 6) length: func() -> u32;
//     //
//     async fn length(&mut self, list_resource: Resource<CachedTokenList>) -> Result<u32> {
//         let token_list = self.resource_table.get(&list_resource)?;
//         Ok(token_list.tokens.len() as u32)
//     }
//
//     //
//     // 7) slice: func(start: u32, end: u32) -> token-list;
//     //
//     //    * Returns a new token-list resource with tokens from [start..end).
//     //
//     async fn slice(
//         &mut self,
//         list_resource: Resource<CachedTokenList>,
//         start: u32,
//         end: u32,
//     ) -> Result<Resource<CachedTokenList>> {
//         Ok(list_resource)
//     }
//
//     //
//     // 8) concat: func(cache: token-list) -> token-list;
//     //
//     //    * Returns a new token-list that is `list_resource` + `other_list_resource`.
//     //
//     async fn concat(
//         &mut self,
//         list_resource: Resource<CachedTokenList>,
//         other_list_resource: Resource<CachedTokenList>,
//     ) -> Result<Resource<CachedTokenList>> {
//         //
//         Ok(other_list_resource)
//     }
//
//     //
//     // 9) index: func(position: u32) -> token;
//     //
//     //    * Returns the token at `position` in `list_resource`.
//     //      We wrap it in a new Resource<CachedToken> so that the caller
//     //      can manipulate it.
//     //
//     async fn index(
//         &mut self,
//         list_resource: Resource<CachedTokenList>,
//         position: u32,
//     ) -> Result<Resource<CachedToken>> {
//         let token_list = self.resource_table.get(&list_resource)?;
//         let idx = position as usize;
//         if idx >= token_list.tokens.len() {
//             return Err(anyhow::anyhow!("Index out of range"));
//         }
//
//         let token = token_list.tokens[idx].clone();
//         // Put this token into the resource table and return that resource
//         let token_resource = self.resource_table.push(token)?;
//         Ok(token_resource)
//     }
//
//     async fn drop(&mut self, resource: Resource<CachedTokenList>) -> Result<()> {
//         let _ = self.resource_table.delete(resource)?;
//         Ok(())
//     }
// }
