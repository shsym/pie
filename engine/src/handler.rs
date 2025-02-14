use crate::state::{
    get_stream_id, Addr, BlockError, CausalLanguageModel, CausalTransformer, ImageEmbedder,
    InstanceId, KvBlock, KvBlockManager, ObjectAllocator, ObjectId, ObjectManager, TokenEmb,
    TokenEmbManager, VideoEmbedder,
};
use std::collections::{HashMap, HashSet};
use tokio::sync::oneshot;

#[derive(Debug)]
pub struct Resource {
    owner_id: InstanceId,
    addrs: Vec<Addr>,
}

impl Resource {
    pub fn new(owner_id: InstanceId, addrs: Vec<Addr>) -> Self {
        Self { owner_id, addrs }
    }
}

#[derive(Debug)]
pub struct Instance {
    owned_resources: Vec<String>,
    //usage_stats: HashMap<String, usize>,
}

impl Instance {
    pub fn new() -> Self {
        Self {
            owned_resources: vec![],
            //usage_stats: HashMap::new(),
        }
    }
}

pub struct ServerState<B> {
    backend: B,
    // resource name -> Resource handle
    resources: HashMap<String, Resource>,
    // instance_id -> Instance
    instances: HashMap<InstanceId, Instance>,

    // managers
    kv_blocks: KvBlockManager<B>,
    token_embs: TokenEmbManager<B>,
    // command batchers
    //fill_cmd_batcher: FillBlockCmdBatcher,
    //img_cmd_batcher: CreateImageTokensCmdBatcher,
}

// Remote Tensor Interface. (RTI)
// RemoteTensorCollection
// RemoteTensorStorage...
//     -- alloc(type, addr)
//     -- dealloc(addr)
//     -- compute( input_addrs, output_addrs, op, etc. )

impl<B> ServerState<B>
where
    B: CausalTransformer + Clone,
{
    pub fn new(backend: B) -> Self {
        Self {
            backend: backend.clone(),
            resources: HashMap::new(),
            instances: HashMap::new(),
            kv_blocks: KvBlockManager::new(backend.clone()),
            token_embs: TokenEmbManager::new(backend),
            //fill_cmd_batcher: FillBlockCmdBatcher::new(),
        }
    }

    pub fn init_instance(&mut self, inst_id: InstanceId) -> Result<(), BlockError> {
        self.instances.insert(inst_id, Instance::new());
        self.kv_blocks.init_instance(inst_id)?;
        self.token_embs.init_instance(inst_id)?;
        Ok(())
    }

    pub fn destroy_instance(&mut self, inst_id: &InstanceId) -> Result<(), BlockError> {
        if let Some(inst) = self.instances.get(inst_id) {
            for r in &inst.owned_resources {
                self.resources.remove(r);
            }
        }
        self.instances.remove(inst_id);
        self.kv_blocks.destroy_instance(inst_id)?;
        self.token_embs.destroy_instance(inst_id)?;

        Ok(())
    }

    pub fn allocate_kv_block(
        &mut self,
        inst_id: &InstanceId,
        local_stream_id: Option<u32>,
    ) -> Result<Addr, BlockError> {
        self.kv_blocks
            .alloc(inst_id, local_stream_id, KvBlock::new())
    }

    pub fn deallocate_kv_block(
        &mut self,
        inst_id: &InstanceId,
        local_stream_id: Option<u32>,
        addr: Addr,
    ) -> Result<(), BlockError> {
        self.kv_blocks.dealloc(inst_id, local_stream_id, addr)
    }

    pub fn export_kv_blocks(
        &mut self,
        inst_id: &InstanceId,
        resource_name: String,
        addrs: Vec<Addr>,
    ) -> Result<(), BlockError> {
        self.resources
            .insert(resource_name.clone(), Resource::new(*inst_id, addrs));
        self.instances
            .get_mut(inst_id)
            .ok_or(BlockError::InstanceNotFound)?
            .owned_resources
            .push(resource_name);
        Ok(())
    }

    pub fn import_kv_blocks(
        &mut self,
        inst_id: &InstanceId,
        resource_name: String,
    ) -> Result<Vec<Addr>, BlockError> {
        let res = self
            .resources
            .get(&resource_name)
            .ok_or(BlockError::ResourceNotFound)?;

        let mut addrs = Vec::with_capacity(res.addrs.len());
        for src_addr in &res.addrs {
            addrs.push(
                self.kv_blocks
                    .create_ref(inst_id, &res.owner_id, *src_addr)?,
            );
        }

        Ok(addrs)
    }

    pub fn fill_kv_block(
        &mut self,
        inst_id: &InstanceId,
        local_stream_id: Option<u32>,
        addr: Addr,
        ctx_addrs: Vec<Addr>,
        input_embs: Vec<Addr>,
        output_embs: Option<Vec<Addr>>,
    ) -> Result<(), BlockError> {
        // create "resolved" cmd.

        let block_ptr = self.kv_blocks.resolve(inst_id, addr)?;
        let ctx_block_ptrs = self.kv_blocks.resolve_many(inst_id, &ctx_addrs)?;

        let input_emb_ptrs = self.token_embs.resolve_many(inst_id, &input_embs)?;

        // let output_emb_ptrs = if let Some(output_embs) = output_embs {
        //     Some(self.token_embs.resolve_many(inst_id, &output_embs)?)
        // } else {
        //     None
        // };

        let output_emb_ptrs = output_embs
            .map(|emb| self.token_embs.resolve_many(inst_id, &emb))
            .transpose()?;

        self.kv_blocks.get_mut(inst_id, addr)?.filled = false;

        self.backend.fill(
            get_stream_id(inst_id, local_stream_id),
            block_ptr,
            ctx_block_ptrs,
            input_emb_ptrs,
            output_emb_ptrs,
        )?;

        Ok(())
    }

    pub fn copy_kv_block(
        &mut self,
        inst_id: &InstanceId,
        local_stream_id: Option<u32>,
        src_addr: Addr,
        dst_addr: Addr,
        src_token_offset: u32,
        dst_token_offset: u32,
        token_count: u32,
    ) -> Result<(), BlockError> {
        self.kv_blocks.copy_tokens(
            inst_id,
            local_stream_id,
            src_addr,
            dst_addr,
            src_token_offset,
            dst_token_offset,
            token_count,
        )
    }

    pub fn mask_kv_block(
        &mut self,
        inst_id: &InstanceId,
        local_stream_id: Option<u32>,
        addr: Addr,
        mask: Vec<bool>,
    ) -> Result<(), BlockError> {
        self.kv_blocks
            .mask_tokens(inst_id, local_stream_id, addr, &mask)
    }

    pub fn allocate_token_emb(
        &mut self,
        inst_id: &InstanceId,
        local_stream_id: Option<u32>,
    ) -> Result<Addr, BlockError> {
        self.token_embs
            .alloc(inst_id, local_stream_id, TokenEmb::new())
    }

    pub fn deallocate_token_emb(
        &mut self,
        inst_id: &InstanceId,
        local_stream_id: Option<u32>,
        addr: Addr,
    ) -> Result<(), BlockError> {
        self.token_embs.dealloc(inst_id, local_stream_id, addr)
    }
}

// For causal LMs

impl<B> ServerState<B>
where
    B: CausalLanguageModel,
{
    pub fn next_token_dist(
        &self,
        inst_id: &InstanceId,
        local_stream_id: Option<u32>,

        emb_ptr: Addr,
        dist_ptr: Addr,
    ) -> Result<(), BlockError> {
        let emb_ptr = self.token_embs.resolve(inst_id, emb_ptr)?;
        let dist_ptr = self.token_embs.resolve(inst_id, dist_ptr)?;

        self.backend
            .next_token_dist(get_stream_id(inst_id, local_stream_id), emb_ptr, dist_ptr)?;

        Ok(())
    }

    pub fn sample_top_k(
        &self,
        inst_id: &InstanceId,
        local_stream_id: Option<u32>,
        dist_ptr: Addr,
        k: u32,
    ) -> Result<oneshot::Receiver<Vec<u32>>, BlockError> {
        let dist_ptr = self.token_embs.resolve(inst_id, dist_ptr)?;
        self.backend
            .sample_top_k(get_stream_id(inst_id, local_stream_id), dist_ptr, k)
    }
}

// For multimodal LLMs
impl<B> ServerState<B>
where
    B: ImageEmbedder + VideoEmbedder + ObjectAllocator<TokenEmb>,
{
    pub fn embed_image(
        &mut self,
        inst_id: &InstanceId,
        local_stream_id: Option<u32>,

        token_addrs: Vec<Addr>,
        image_url: String,
    ) -> Result<(), BlockError> {
        let ptrs = self.token_embs.resolve_many(inst_id, &token_addrs)?;

        self.backend
            .embed_img(get_stream_id(inst_id, local_stream_id), ptrs, image_url)?;

        Ok(())
    }

    pub fn embed_video(
        &mut self,
        inst_id: &InstanceId,
        local_stream_id: Option<u32>,

        token_addrs: Vec<Addr>,
        video_url: String,
    ) -> Result<(), BlockError> {
        let ptrs = self.token_embs.resolve_many(inst_id, &token_addrs)?;

        self.backend
            .embed_vid(get_stream_id(inst_id, local_stream_id), ptrs, video_url)?;

        Ok(())
    }
}

//
// pub struct FillBlockCmdBatcher {
//     queue: Vec<_FillBlockCmd>,
//     redundancy_check: HashSet<RemoteObjId>,
// }
//
// impl FillBlockCmdBatcher {
//     pub fn new() -> Self {
//         Self {
//             queue: vec![],
//             redundancy_check: HashSet::new(),
//         }
//     }
//
//     pub fn add(&mut self, cmd: _FillBlockCmd) {
//         let ptr = cmd.block_ptr;
//         // If there's already a command for this block, replace it
//         if self.redundancy_check.remove(&ptr) {
//             // remove from the queue
//             let idx = self.queue.iter().position(|c| c.block_ptr == ptr);
//             if let Some(i) = idx {
//                 self.queue.remove(i);
//             }
//         }
//         self.redundancy_check.insert(ptr);
//         self.queue.push(cmd);
//     }
//
//     pub fn batch(&self) {
//
//
//
//         ///
//     }
//
//     pub fn clear(&mut self) {
//         self.queue.clear();
//         self.redundancy_check.clear();
//     }
// }

/// Internal struct that won't be exposed outside the module
#[derive(Debug, Clone)]
struct _FillBlockCmd {
    block_ptr: ObjectId,
    ctx_block_ptrs: Vec<ObjectId>,
    mask: Vec<bool>,
    input_emb_ptrs: Vec<ObjectId>,
    output_emb_ptrs: Vec<ObjectId>,
}

#[derive(Debug, Clone)]
pub enum Command {
    AllocateKvBlocks(AllocateKvBlocksCmd),
    DeallocateKvBlocks(DeallocateKvBlocksCmd),
    AvailableKvBlocks(AvailableKvBlocksCmd),
    ExportKvBlocks(ExportKvBlocksCmd),
    ImportKvBlocks(ImportKvBlocksCmd),
    FillKvBlock(FillKvBlockCmd),
    CopyKvBlock(CopyKvBlockCmd),
    MaskKvBlock(MaskKvBlockCmd),
    AllocateTokenEmbeds(AllocateTokenEmbedsCmd),
    DeallocateTokenEmbeds(DeallocateTokenEmbedsCmd),
    AvailableTokenEmbeds(AvailableTokenEmbedsCmd),
    EmbedImage(EmbedImageCmd),
    EmbedVideo(EmbedVideoCmd),
    Decode(DecodeCmd),
    GetNextTokenDist(GetNextTokenDistCmd),
    GetFeatureVector(GetFeatureVectorCmd),
}

// =============================================================================
// Individual command payloads
// =============================================================================

#[derive(Debug, Clone)]
pub struct AllocateKvBlocksCmd {
    pub num_blocks: usize,
}

#[derive(Debug, Clone)]
pub struct DeallocateKvBlocksCmd {
    pub addr_offset: Addr,
    pub count: usize,
}

#[derive(Debug, Clone)]
pub struct AvailableKvBlocksCmd;

#[derive(Debug, Clone)]
pub struct ExportKvBlocksCmd {
    pub resource_name: String,
    pub addr_offset: Addr,
    pub count: usize,
}

#[derive(Debug, Clone)]
pub struct ImportKvBlocksCmd {
    pub resource_name: String,
}

#[derive(Debug, Clone)]
pub struct FillKvBlockCmd {
    pub addr: Addr,
    pub ctx_addrs: Vec<Addr>,
    pub mask: Vec<bool>,
    /// For simplicity, unify “input_embeds” or “tokens” into one name
    pub input_embs: Vec<Addr>,
    pub output_embs: Vec<Addr>,
}

#[derive(Debug, Clone)]
pub struct CopyKvBlockCmd {
    pub src_addr: Addr,
    pub dst_addr: Addr,
    pub src_token_offset: usize,
    pub dst_token_offset: usize,
    pub token_count: usize,
}

#[derive(Debug, Clone)]
pub struct MaskKvBlockCmd {
    pub addr: Addr,
    pub token_offset: usize,
    pub token_count: usize,
}

#[derive(Debug, Clone)]
pub struct AllocateTokenEmbedsCmd {
    pub num_tokens: usize,
}

#[derive(Debug, Clone)]
pub struct DeallocateTokenEmbedsCmd {
    pub addr_offset: Addr,
    pub count: usize,
}

#[derive(Debug, Clone)]
pub struct AvailableTokenEmbedsCmd;

#[derive(Debug, Clone)]
pub struct EmbedImageCmd {
    pub image_url: String,
}

#[derive(Debug, Clone)]
pub struct EmbedVideoCmd {
    pub video_url: String,
}

#[derive(Debug, Clone)]
pub struct DecodeCmd;

#[derive(Debug, Clone)]
pub struct GetNextTokenDistCmd {
    pub addr: Addr,
    pub offset: usize,
    pub size: usize,
    pub drop_output_embed: bool,
}

#[derive(Debug, Clone)]
pub struct GetFeatureVectorCmd {
    pub addr: Addr,
    pub offset: usize,
    pub size: usize,
    pub drop_output_embed: bool,
}
