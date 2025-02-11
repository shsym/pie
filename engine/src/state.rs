use std::collections::HashMap;
use std::hash::Hash;

use crate::utils::{Counter, IdPool};
use num_traits::PrimInt;
use tokio::sync::oneshot;
use uuid::Uuid;

pub type InstanceId = Uuid;
pub type Addr = usize;

// ------------------------------------------------------------

pub type RemoteObjId = usize;

// this helps the backend to make optimization decisions.

pub trait ReferenceCounted {
    /// Increments the reference count.
    fn add_ref(&self);

    /// Decrements the reference count.
    ///
    /// Returns `true` if the count has reached zero, indicating that the object
    /// can be safely cleaned up.
    fn release(&self) -> bool;

    /// Returns the current reference count.
    fn ref_count(&self) -> usize;
}

pub trait ObjectAllocator<T: ReferenceCounted> {
    type RawRepr;

    fn alloc(&self, stream_id: &InstanceId) -> Result<RemoteObjId, BlockError>;

    fn dealloc(&self, stream_id: &InstanceId, obj_id: RemoteObjId) -> Result<(), BlockError>;

    // retrieve the underlying data from the backend.
    // this is unlikely to be used in practice, except for debugging, implementing some new sampling algos, and so on.
    fn raw_repr(
        &self,
        stream_id: &InstanceId,
        obj_id: RemoteObjId,
        sender: oneshot::Sender<Self::RawRepr>,
    ) -> Result<(), BlockError>;

    fn available(&self) -> usize;
}

// reference-counted object manager
pub trait ObjectManager<T: ReferenceCounted, B: ObjectAllocator<T>> {
    fn objects(&self) -> &HashMap<RemoteObjId, T>;

    fn objects_mut(&mut self) -> &mut HashMap<RemoteObjId, T>;

    fn addr_maps(&self) -> &HashMap<InstanceId, AddrMap<Addr, RemoteObjId>>;

    fn addr_maps_mut(&mut self) -> &mut HashMap<InstanceId, AddrMap<Addr, RemoteObjId>>;

    fn backend(&self) -> &B;

    fn init_instance(&mut self, inst_id: InstanceId) -> Result<(), BlockError> {
        if self.addr_maps().contains_key(&inst_id) {
            return Err(BlockError::InstanceAlreadyExists);
        }

        self.addr_maps_mut().insert(inst_id, AddrMap::new());
        Ok(())
    }

    fn destroy_instance(&mut self, inst_id: &InstanceId) -> Result<(), BlockError> {
        // Collect the keys into a Vec, which drops the borrow on `self` after collection.
        let addrs: Vec<_> = self
            .addr_maps()
            .get(&inst_id)
            .ok_or(BlockError::InstanceNotFound)?
            .mapping
            .keys()
            .copied() // Copy the keys to get owned values (assuming keys are Copy)
            .collect();

        // Now iterate over the owned keys, and it's safe to call mutable methods on `self`
        for addr in addrs {
            self.dealloc(inst_id, addr)?;
        }

        self.addr_maps_mut().remove(&inst_id);

        Ok(())
    }

    fn alloc(&mut self, inst_id: &InstanceId, obj: T) -> Result<Addr, BlockError> {
        let new_g_addr = self.backend().alloc(inst_id)?;

        let new_addr = self
            .addr_maps_mut()
            .get_mut(&inst_id)
            .ok_or(BlockError::InstanceNotFound)?
            .register(new_g_addr);

        obj.add_ref();

        self.objects_mut().insert(new_g_addr, obj);
        Ok(new_addr)
    }

    fn create_ref(
        &mut self,
        inst_id: &InstanceId,
        src_inst_id: &InstanceId,
        src_addr: Addr,
    ) -> Result<Addr, BlockError> {
        let src_g_addr = self
            .addr_maps()
            .get(&src_inst_id)
            .ok_or(BlockError::InstanceNotFound)?
            .resolve(src_addr)?;

        // insert a new virtual address
        let new_v_addr = self
            .addr_maps_mut()
            .get_mut(&inst_id)
            .ok_or(BlockError::InstanceNotFound)?
            .register(src_g_addr);

        // increase ref count
        self.objects()
            .get(&src_g_addr)
            .ok_or(BlockError::BlockNotFound)?
            .add_ref();

        Ok(new_v_addr)
    }

    fn dealloc(&mut self, inst_id: &InstanceId, addr: Addr) -> Result<(), BlockError> {
        // remove and get the global address
        let g_addr = self
            .addr_maps_mut()
            .get_mut(&inst_id)
            .ok_or(BlockError::InstanceNotFound)?
            .unregister(addr)?;

        let remove_entirely = self
            .objects()
            .get(&g_addr)
            .ok_or(BlockError::BlockNotFound)?
            .release();

        // remove the block if the ref count is 0
        if remove_entirely {
            self.backend().dealloc(inst_id, g_addr)?;
            self.objects_mut().remove(&g_addr);
        }

        Ok(())
    }

    fn resolve(&self, inst_id: &InstanceId, addr: Addr) -> Result<RemoteObjId, BlockError> {
        self.addr_maps()
            .get(&inst_id)
            .ok_or(BlockError::InstanceNotFound)?
            .resolve(addr)
    }

    fn resolve_many(
        &self,
        inst_id: &InstanceId,
        addrs: &[Addr],
    ) -> Result<Vec<RemoteObjId>, BlockError> {
        self.addr_maps()
            .get(&inst_id)
            .ok_or(BlockError::InstanceNotFound)?
            .resolve_many(addrs)
    }

    fn get(&self, inst_id: &InstanceId, addr: Addr) -> Result<&T, BlockError> {
        let g_addr = self.resolve(inst_id, addr)?;
        self.objects().get(&g_addr).ok_or(BlockError::BlockNotFound)
    }

    fn get_mut(&mut self, inst_id: &InstanceId, addr: Addr) -> Result<&mut T, BlockError> {
        let g_addr = self.resolve(inst_id, addr)?;
        self.objects_mut()
            .get_mut(&g_addr)
            .ok_or(BlockError::BlockNotFound)
    }

    fn get_many(&self, inst_id: &InstanceId, addrs: &[Addr]) -> Result<Vec<&T>, BlockError> {
        self.resolve_many(inst_id, addrs)?
            .iter()
            .map(|&g_addr| self.objects().get(&g_addr).ok_or(BlockError::BlockNotFound))
            .collect::<Result<Vec<_>, _>>()
    }

    fn available_objs(&self) -> usize {
        self.backend().available()
    }
}

// ------------------------------------------------------------

pub struct KvBlock {
    counter: Counter,

    // pos ids and vacancy maps
    pub position_ids: Vec<u32>,
    pub occupied: Vec<bool>,
    pub filled: bool,
}

// Backend trait for filling key-value blocks. (GPT-like models)
pub trait CausalTransformer: ObjectAllocator<KvBlock> + ObjectAllocator<TokenEmb> {
    fn fill(
        &self,
        stream_id: &InstanceId,
        addr: RemoteObjId,
        ctx_addrs: Vec<RemoteObjId>,
        mask: Vec<bool>,
        input_embs: Vec<RemoteObjId>,
        output_embs: Vec<RemoteObjId>,
    ) -> Result<(), BlockError>;

    fn copy_tokens(
        &self,
        stream_id: &InstanceId,
        src_ptr: RemoteObjId,
        dst_ptr: RemoteObjId,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
    ) -> Result<(), BlockError>;

    fn mask_tokens(
        &self,
        stream_id: &InstanceId,
        ptr: RemoteObjId,
        mask: &[bool],
    ) -> Result<(), BlockError>;
}

// probably unused in the first version. For BERT-like models.
pub trait FullTransformer: ObjectAllocator<TokenEmb> {
    fn fill(
        &self,
        inst_id: &InstanceId,
        mask: Vec<bool>,
        input_embs: Vec<RemoteObjId>,
        output_embs: Vec<RemoteObjId>,
    ) -> Result<(), BlockError>;
}

// could be used for other LLM architectures like SSMs
pub trait Rnn: ObjectAllocator<TokenEmb> {
    fn fill(
        &self,
        inst_id: &InstanceId,
        state: RemoteObjId,
        output_embs: Vec<RemoteObjId>,
    ) -> Result<(), BlockError>;
}

// ------------------------------------------------------------

impl KvBlock {
    pub fn new() -> Self {
        KvBlock {
            counter: Counter::new(0),
            position_ids: Vec::new(),
            occupied: Vec::new(),
            filled: false,
        }
    }
}

impl ReferenceCounted for KvBlock {
    fn add_ref(&self) {
        self.counter.inc();
    }

    fn release(&self) -> bool {
        self.counter.dec() <= 0
    }

    fn ref_count(&self) -> usize {
        self.counter.get() as usize
    }
}

pub struct KvBlockManager<B> {
    kv_blocks: HashMap<RemoteObjId, KvBlock>,
    token_embs: HashMap<RemoteObjId, TokenEmb>,
    virtual_addr_maps: HashMap<InstanceId, AddrMap<Addr, RemoteObjId>>,

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
    B: ObjectAllocator<KvBlock>,
{
    fn objects(&self) -> &HashMap<Addr, KvBlock> {
        &self.kv_blocks
    }

    fn objects_mut(&mut self) -> &mut HashMap<Addr, KvBlock> {
        &mut self.kv_blocks
    }

    fn addr_maps(&self) -> &HashMap<InstanceId, AddrMap<Addr, RemoteObjId>> {
        &self.virtual_addr_maps
    }

    fn addr_maps_mut(&mut self) -> &mut HashMap<InstanceId, AddrMap<Addr, RemoteObjId>> {
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
        src_addr: Addr,
        dst_addr: Addr,
        src_token_offset: usize,
        dst_token_offset: usize,
        size: usize,
    ) -> Result<(), BlockError> {
        let src_block_ptr = self.resolve(inst_id, src_addr)?;
        let dst_block_ptr = self.resolve(inst_id, dst_addr)?;

        self.backend().copy_tokens(
            inst_id,
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
                src_block.position_ids[src_token_offset..src_token_offset + size].to_vec(),
                src_block.occupied[src_token_offset..src_token_offset + size].to_vec(),
            )
        };

        // Now get a mutable borrow of the destination block and update its data.
        let dst_block = self.get_mut(inst_id, dst_addr)?;
        for i in 0..size {
            dst_block.position_ids[dst_token_offset + i] = src_position_ids[i];
            dst_block.occupied[dst_token_offset + i] = src_occupied[i];
        }
        Ok(())
    }

    pub fn mask_tokens(
        &mut self,
        inst_id: &InstanceId,
        virtual_addr: Addr,
        mask: &[bool],
    ) -> Result<(), BlockError> {
        let block_ptr = self.resolve(inst_id, virtual_addr)?;
        self.backend.mask_tokens(inst_id, block_ptr, mask)?;

        let block = self.get_mut(inst_id, virtual_addr)?;
        for i in 0..mask.len() {
            block.occupied[i] = mask[i];
        }

        Ok(())
    }
}

// ------------------------------------------------------------

pub struct TokenEmb {
    counter: Counter,
}

impl TokenEmb {
    pub fn new() -> Self {
        TokenEmb {
            counter: Counter::new(0),
        }
    }
}

impl ReferenceCounted for TokenEmb {
    fn add_ref(&self) {
        self.counter.inc();
    }

    fn release(&self) -> bool {
        self.counter.dec() <= 0
    }

    fn ref_count(&self) -> usize {
        self.counter.get() as usize
    }
}

pub struct TokenEmbManager<B> {
    token_embs: HashMap<RemoteObjId, TokenEmb>,
    virtual_addr_maps: HashMap<InstanceId, AddrMap<Addr, RemoteObjId>>,

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
    B: ObjectAllocator<TokenEmb>,
{
    fn objects(&self) -> &HashMap<Addr, TokenEmb> {
        &self.token_embs
    }

    fn objects_mut(&mut self) -> &mut HashMap<Addr, TokenEmb> {
        &mut self.token_embs
    }

    fn addr_maps(&self) -> &HashMap<InstanceId, AddrMap<Addr, RemoteObjId>> {
        &self.virtual_addr_maps
    }

    fn addr_maps_mut(&mut self) -> &mut HashMap<InstanceId, AddrMap<Addr, RemoteObjId>> {
        &mut self.virtual_addr_maps
    }

    fn backend(&self) -> &B {
        &self.storage
    }
}

// ------------------------------------------------------------

// distribution

pub struct TokenDist {
    counter: Counter,
}

impl TokenDist {
    pub fn new() -> Self {
        TokenDist {
            counter: Counter::new(0),
        }
    }
}

impl ReferenceCounted for TokenDist {
    fn add_ref(&self) {
        self.counter.inc();
    }

    fn release(&self) -> bool {
        self.counter.dec() <= 0
    }

    fn ref_count(&self) -> usize {
        self.counter.get() as usize
    }
}

pub struct TokenDistManager<B> {
    token_dists: HashMap<RemoteObjId, TokenDist>,
    virtual_addr_maps: HashMap<InstanceId, AddrMap<Addr, RemoteObjId>>,

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
    B: ObjectAllocator<TokenDist>,
{
    fn objects(&self) -> &HashMap<Addr, TokenDist> {
        &self.token_dists
    }

    fn objects_mut(&mut self) -> &mut HashMap<Addr, TokenDist> {
        &mut self.token_dists
    }

    fn addr_maps(&self) -> &HashMap<InstanceId, AddrMap<Addr, RemoteObjId>> {
        &self.virtual_addr_maps
    }

    fn addr_maps_mut(&mut self) -> &mut HashMap<InstanceId, AddrMap<Addr, RemoteObjId>> {
        &mut self.virtual_addr_maps
    }

    fn backend(&self) -> &B {
        &self.storage
    }
}

// ------------------------------------------------------------

pub trait CausalLanguageModel: ObjectAllocator<TokenEmb> + ObjectAllocator<TokenDist> {
    fn next_token_dist(
        &self,
        inst_id: &InstanceId,
        emb_ptr: RemoteObjId,
        dist_ptr: RemoteObjId,
    ) -> Result<(), BlockError>;

    fn sample_top_k(
        &self,
        inst_id: &InstanceId,
        dist_ptr: RemoteObjId,
        k: usize,
        sender: oneshot::Sender<Vec<usize>>,
    ) -> Result<(), BlockError>;

    // todo: design a better struct to represent distributions
    fn get_raw_dist(
        &self,
        inst_id: &InstanceId,
        dist_ptr: RemoteObjId,
        sender: oneshot::Sender<Vec<f32>>,
    ) -> Result<(), BlockError>;
}

pub trait MaskedLanguageModel: ObjectAllocator<TokenEmb> + ObjectAllocator<TokenDist> {
    fn token_dist(
        &self,
        inst_id: &InstanceId,
        emb_ptr: RemoteObjId,
        dist_ptr: RemoteObjId,
    ) -> Result<(), BlockError>;
}

// ------------------------------------------------------------

// Trait for backends that can embed images.
pub trait ImageEmbedder: ObjectAllocator<TokenEmb> {
    fn embed_img(
        &self,
        inst_id: &InstanceId,
        addrs: Vec<RemoteObjId>,
        url: String,
    ) -> Result<(), BlockError>;
}

// Trait for backends that can embed videos.
pub trait VideoEmbedder: ObjectAllocator<TokenEmb> {
    fn embed_vid(
        &self,
        inst_id: &InstanceId,
        addrs: Vec<RemoteObjId>,
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
}


// ------------------------------------------------------------
// AddressMap
// ------------------------------------------------------------
#[derive(Debug)]
pub struct AddrMap<K, V> {
    mapping: HashMap<K, V>,
    addr_pool: IdPool<K>,
}

impl<K, V> AddrMap<K, V>
where
    K: PrimInt + Hash,
    V: Copy,
{
    pub fn new() -> Self {
        Self {
            mapping: HashMap::new(),
            addr_pool: IdPool::new(K::max_value()),
        }
    }

    pub fn register(&mut self, g_addr: V) -> K {
        let new_v_addr = self.addr_pool.acquire().unwrap();
        self.mapping.insert(new_v_addr, g_addr);
        new_v_addr
    }

    pub fn unregister(&mut self, v_addr: K) -> Result<V, BlockError> {
        let removed_g_addr = self
            .mapping
            .remove(&v_addr)
            .ok_or(BlockError::BlockNotFound)?;
        self.addr_pool.release(v_addr)?;

        Ok(removed_g_addr)
    }

    pub fn resolve(&self, v_addr: K) -> Result<V, BlockError> {
        self.mapping
            .get(&v_addr)
            .copied()
            .ok_or(BlockError::VirtualAddressTranslationFailed)
    }

    pub fn resolve_many(&self, v_addrs: &[K]) -> Result<Vec<V>, BlockError> {
        let mut g_addrs = Vec::with_capacity(v_addrs.len());
        for v_addr in v_addrs {
            let global_addr = self.resolve(v_addr.clone())?;
            g_addrs.push(global_addr);
        }
        Ok(g_addrs)
    }
}
