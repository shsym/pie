use std::collections::HashMap;
use std::hash::Hash;

use crate::object;

use crate::object::{KvBlock, TokenDist, TokenEmb};
use tokio::sync::oneshot;
use uuid::Uuid;
use crate::utils::Stream;

pub type InstanceId = Uuid;
//pub type StreamId = (u128, u32);

pub type Addr = object::Id;

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

pub struct ControllerManager<B> {
    backend: B,
    // resource name -> Resource handle
    resources: HashMap<String, Resource>,
    // instance_id -> Instance
    instances: HashMap<InstanceId, Instance>,

    // managers
    kv_blocks: KvBlockManager<B>,
    token_embs: TokenEmbManager<B>,
}
//
//
//
// impl object::Allocator<KvBlock> for Controller<B> {
//     fn alloc(&mut self, _local_stream_id: Option<u32>) -> Result<Addr, BlockError> {
//         unimplemented!()
//     }
//
//     fn dealloc(&mut self, _local_stream_id: Option<u32>, _addr: Addr) -> Result<(), BlockError> {
//         unimplemented!()
//     }
// }

impl<B> ControllerManager<B>
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

impl<B> ControllerManager<B>
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
impl<B> ControllerManager<B>
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

// Backend trait for filling key-value blocks. (GPT-like models)
pub trait CausalTransformer: object::Allocator<KvBlock> + object::Allocator<TokenEmb> {
    fn fill(
        &self,
        stream_id: Stream,
        addr: object::Id,
        ctx_addrs: Vec<object::Id>,
        input_embs: Vec<object::Id>,
        output_embs: Option<Vec<object::Id>>,
    ) -> Result<(), BlockError>;

    fn copy_tokens(
        &self,
        stream_id: StreamId,
        src_ptr: object::Id,
        dst_ptr: object::Id,
        src_offset: u32,
        dst_offset: u32,
        size: u32,
    ) -> Result<(), BlockError>;

    fn mask_tokens(
        &self,
        stream_id: StreamId,
        ptr: object::Id,
        mask: &[bool],
    ) -> Result<(), BlockError>;
}

// probably unused in the first version. For BERT-like models.
pub trait FullTransformer: object::Allocator<TokenEmb> {
    fn fill(
        &self,
        stream_id: StreamId,
        mask: Vec<bool>,
        input_embs: Vec<object::Id>,
        output_embs: Vec<object::Id>,
    ) -> Result<(), BlockError>;
}

// could be used for other LLM architectures like SSMs
pub trait Rnn: object::Allocator<TokenEmb> {
    fn fill(
        &self,
        stream_id: StreamId,
        state: object::Id,
        output_embs: Vec<object::Id>,
    ) -> Result<(), BlockError>;
}

// ------------------------------------------------------------

pub struct KvBlockManager<B> {
    kv_blocks: HashMap<object::Id, KvBlock>,
    token_embs: HashMap<object::Id, TokenEmb>,
    virtual_addr_maps: HashMap<InstanceId, AddrMap<Addr, object::Id>>,

    // primary storage
    backend: B,
}

impl<B> KvBlockManager<B> {
    pub fn new(backend: B) -> Self {
        Self {
            kv_blocks: HashMap::new(),
            token_embs: HashMap::new(),
            virtual_addr_maps: HashMap::new(),
            backend,
        }
    }
}

impl<B> ObjectManager<KvBlock, B> for KvBlockManager<B>
where
    B: object::Allocator<KvBlock>,
{
    fn objects(&self) -> &HashMap<Addr, KvBlock> {
        &self.kv_blocks
    }

    fn objects_mut(&mut self) -> &mut HashMap<Addr, KvBlock> {
        &mut self.kv_blocks
    }

    fn addr_maps(&self) -> &HashMap<InstanceId, AddrMap<Addr, object::Id>> {
        &self.virtual_addr_maps
    }

    fn addr_maps_mut(&mut self) -> &mut HashMap<InstanceId, AddrMap<Addr, object::Id>> {
        &mut self.virtual_addr_maps
    }

    fn backend(&self) -> &B {
        &self.backend
    }
}

impl<B> KvBlockManager<B>
where
    B: CausalTransformer,
{
    pub fn copy_tokens(
        &mut self,
        inst_id: &InstanceId,
        local_stream_id: Option<u32>,
        src_addr: Addr,
        dst_addr: Addr,
        src_token_offset: u32,
        dst_token_offset: u32,
        size: u32,
    ) -> Result<(), BlockError> {
        let stream_id = get_stream_id(inst_id, local_stream_id);

        let src_block_ptr = self.resolve(inst_id, src_addr)?;
        let dst_block_ptr = self.resolve(inst_id, dst_addr)?;

        self.backend().copy_tokens(
            stream_id,
            src_block_ptr,
            dst_block_ptr,
            src_token_offset,
            dst_token_offset,
            size,
        )?;

        // First, get a temporary copy of the data from the source block.
        // Here we use an immutable borrow so that we can later borrow the destination mutably.
        let (src_position_ids, src_occupied) = {
            let src_block = self.get(inst_id, src_addr)?;
            (
                src_block.position_ids
                    [src_token_offset as usize..(src_token_offset + size) as usize]
                    .to_vec(),
                src_block.occupied[src_token_offset as usize..(src_token_offset + size) as usize]
                    .to_vec(),
            )
        };

        // Now get a mutable borrow of the destination block and update its data.
        let dst_block = self.get_mut(inst_id, dst_addr)?;
        for i in 0..size as usize {
            dst_block.position_ids[dst_token_offset as usize + i] = src_position_ids[i];
            dst_block.occupied[dst_token_offset as usize + i] = src_occupied[i];
        }
        Ok(())
    }

    pub fn mask_tokens(
        &mut self,
        inst_id: &InstanceId,
        local_stream_id: Option<u32>,
        virtual_addr: Addr,
        mask: &[bool],
    ) -> Result<(), BlockError> {
        let stream_id = get_stream_id(inst_id, local_stream_id);

        let block_ptr = self.resolve(inst_id, virtual_addr)?;
        self.backend.mask_tokens(stream_id, block_ptr, mask)?;

        let block = self.get_mut(inst_id, virtual_addr)?;
        for i in 0..mask.len() {
            block.occupied[i] = mask[i];
        }

        Ok(())
    }
}

// ------------------------------------------------------------

pub struct TokenEmbManager<B> {
    token_embs: HashMap<object::Id, TokenEmb>,
    virtual_addr_maps: HashMap<InstanceId, AddrMap<Addr, object::Id>>,

    // primary storage
    storage: B,
}

impl<B> TokenEmbManager<B> {
    pub fn new(storage: B) -> Self {
        Self {
            token_embs: HashMap::new(),
            virtual_addr_maps: HashMap::new(),
            storage,
        }
    }
}

impl<B> ObjectManager<TokenEmb, B> for TokenEmbManager<B>
where
    B: object::Allocator<TokenEmb>,
{
    fn objects(&self) -> &HashMap<Addr, TokenEmb> {
        &self.token_embs
    }

    fn objects_mut(&mut self) -> &mut HashMap<Addr, TokenEmb> {
        &mut self.token_embs
    }

    fn addr_maps(&self) -> &HashMap<InstanceId, AddrMap<Addr, object::Id>> {
        &self.virtual_addr_maps
    }

    fn addr_maps_mut(&mut self) -> &mut HashMap<InstanceId, AddrMap<Addr, object::Id>> {
        &mut self.virtual_addr_maps
    }

    fn backend(&self) -> &B {
        &self.storage
    }
}

// ------------------------------------------------------------

pub struct TokenDistManager<B> {
    token_dists: HashMap<object::Id, TokenDist>,
    virtual_addr_maps: HashMap<InstanceId, AddrMap<Addr, object::Id>>,

    // primary storage
    storage: B,
}

impl<B> TokenDistManager<B> {
    pub fn new(storage: B) -> Self {
        Self {
            token_dists: HashMap::new(),
            virtual_addr_maps: HashMap::new(),
            storage,
        }
    }
}

impl<B> ObjectManager<TokenDist, B> for TokenDistManager<B>
where
    B: object::Allocator<TokenDist>,
{
    fn objects(&self) -> &HashMap<Addr, TokenDist> {
        &self.token_dists
    }

    fn objects_mut(&mut self) -> &mut HashMap<Addr, TokenDist> {
        &mut self.token_dists
    }

    fn addr_maps(&self) -> &HashMap<InstanceId, AddrMap<Addr, object::Id>> {
        &self.virtual_addr_maps
    }

    fn addr_maps_mut(&mut self) -> &mut HashMap<InstanceId, AddrMap<Addr, object::Id>> {
        &mut self.virtual_addr_maps
    }

    fn backend(&self) -> &B {
        &self.storage
    }
}

// ------------------------------------------------------------

pub trait CausalLanguageModel<K>:
    object::MappedAllocator<TokenEmb, K> + object::MappedAllocator<TokenDist, K>
where
    K: Hash + Copy,
{
    fn next_token_dist(
        &self,
        stream: Stream,
        vspace_id: &K,
        emb_ptr: object::Id,
        dist_ptr: object::Id,
    ) -> Result<(), BlockError>;

    fn sample_top_k(
        &self,
        stream_id: StreamId,
        dist_ptr: object::Id,
        k: u32,
    ) -> Result<oneshot::Receiver<Vec<u32>>, BlockError>;

    // todo: design a better struct to represent distributions
    fn get_raw_dist(
        &self,
        stream_id: StreamId,
        dist_ptr: object::Id,
    ) -> Result<oneshot::Receiver<Vec<f32>>, BlockError>;
}

pub trait MaskedLanguageModel: object::Allocator<TokenEmb> + object::Allocator<TokenDist> {
    fn token_dist(
        &self,
        stream_id: StreamId,
        emb_ptr: object::Id,
        dist_ptr: object::Id,
    ) -> Result<(), BlockError>;
}

// ------------------------------------------------------------

// Trait for backends that can embed images.
pub trait ImageEmbedder: object::Allocator<TokenEmb> {
    fn embed_img(
        &self,
        stream_id: StreamId,
        addrs: Vec<object::Id>,
        url: String,
    ) -> Result<(), BlockError>;
}

// Trait for backends that can embed videos.
pub trait VideoEmbedder: object::Allocator<TokenEmb> {
    fn embed_vid(
        &self,
        stream_id: StreamId,
        addrs: Vec<object::Id>,
        url: String,
    ) -> Result<(), BlockError>;
}

// ------------------------------------------------------------

#[derive(Debug)]
pub enum BlockError {
    NoFreeBlocks,
    NotEnoughFreeBlocks { requested: usize, available: usize },
    VirtualAddressTranslationFailed,
    BlockNotFound,
    InstanceNotFound,
    InstanceAlreadyExists,
    ResourceNotFound,
    ResourcePermissionDenied,
    LockError,
    SendError,
}
