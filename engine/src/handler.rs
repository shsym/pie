use crate::remote_obj::{
    Addr, BlockError, InstanceId, KvBlockManager, KvBlockStorage, RemoteObj, RemoteObjId,
    RemoteObjManager, TokenEmbManager, TokenEmbStorage,
};
use std::collections::HashMap;

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

pub struct ServerState {
    // resource name -> Resource handle
    resources: HashMap<String, Resource>,
    // instance_id -> Instance
    instances: HashMap<InstanceId, Instance>,

    // managers
    kv_blocks: KvBlockManager,
    token_embs: TokenEmbManager,
    // command batchers
    fill_cmd_batcher: FillBlockCmdBatcher,
    //img_cmd_batcher: CreateImageTokensCmdBatcher,
}

impl ServerState {
    pub fn new(kv_block_storage: KvBlockStorage, token_emb_storage: TokenEmbStorage) -> Self {
        Self {
            resources: HashMap::new(),
            instances: HashMap::new(),
            kv_blocks: KvBlockManager::new(kv_block_storage),
            token_embs: TokenEmbManager::new(token_emb_storage),
            fill_cmd_batcher: FillBlockCmdBatcher::new(),
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

    pub fn allocate_kv_block(&mut self, inst_id: &InstanceId) -> Result<Addr, BlockError> {
        self.kv_blocks.alloc(inst_id)
    }

    pub fn deallocate_kv_block(
        &mut self,
        inst_id: &InstanceId,
        addr: Addr,
    ) -> Result<(), BlockError> {
        self.kv_blocks.dealloc(inst_id, addr)
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
        addr: Addr,
        ctx_addrs: Vec<Addr>,
        mask: Vec<bool>,
        input_embs: Vec<Addr>,
        output_embs: Vec<Addr>,
    ) -> Result<(), BlockError> {
        // create "resolved" cmd.

        let block_ptr = self.kv_blocks.get(inst_id, addr)?.id();
        let ctx_block_ptrs = self
            .kv_blocks
            .get_many(inst_id, &ctx_addrs)?
            .iter()
            .map(|a| a.id())
            .collect();

        let input_emb_ptrs = self
            .token_embs
            .get_many(inst_id, &input_embs)?
            .iter()
            .map(|a| a.id())
            .collect();

        let output_emb_ptrs = self
            .token_embs
            .get_many(inst_id, &output_embs)?
            .iter()
            .map(|a| a.id())
            .collect();

        let cmd = _FillBlockCmd {
            block_ptr,
            ctx_block_ptrs,
            mask,
            input_emb_ptrs,
            output_emb_ptrs,
        };

        self.kv_blocks.get_mut(inst_id, addr)?.filled = false;
        self.fill_cmd_batcher.add(cmd);

        Ok(())
    }
}

pub struct FillBlockCmdBatcher {
    queue: Vec<_FillBlockCmd>,
    redundancy_check: HashMap<Addr, _FillBlockCmd>,
}

impl FillBlockCmdBatcher {
    pub fn new() -> Self {
        Self {
            queue: vec![],
            redundancy_check: HashMap::new(),
        }
    }

    pub fn add(&mut self, cmd: _FillBlockCmd) {
        let ptr = cmd.block.pointer;
        // If there's already a command for this block, replace it
        if let Some(prev) = self.redundancy_check.remove(&ptr) {
            // remove from the queue
            let idx = self.queue.iter().position(|c| c.block.pointer == ptr);
            if let Some(i) = idx {
                self.queue.remove(i);
            }
        }
        self.redundancy_check.insert(ptr, cmd.clone());
        self.queue.push(cmd);
    }

    /// Very simplified version of the "batch" concept (the Python code uses numpy/torch).
    pub fn batch(&self) {
        // We won't replicate the entire numeric logic here.
        // We'll just show how you might iterate over the queued commands.
        for cmd in &self.queue {
            println!(
                "Batch item for block pointer={} with {} context blocks",
                cmd.block.pointer,
                cmd.ctx_blocks.len()
            );
        }
    }

    pub fn clear(&mut self) {
        self.queue.clear();
        self.redundancy_check.clear();
    }
}

/// Internal struct that won't be exposed outside the module
#[derive(Debug, Clone)]
struct _FillBlockCmd {
    block_ptr: RemoteObjId,
    ctx_block_ptrs: Vec<RemoteObjId>,
    mask: Vec<bool>,
    input_emb_ptrs: Vec<RemoteObjId>,
    output_emb_ptrs: Vec<RemoteObjId>,
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
