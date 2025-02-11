use crate::state::{
    Addr, BlockError, CausalTransformer, ImageEmbedder, InstanceId, KvBlock, KvBlockManipulator,
    ObjectAllocator, RemoteObjId, TokenDist, TokenEmb,
};
use crate::utils::IdPool;
use std::cell::RefCell;
use std::rc::Rc;
use tokio::sync::oneshot::Sender;
use uuid::Uuid;

type StreamId = Uuid;

/// Intermediate representation of a command to be executed by the backend.
/// This must not be exposed to other modules.
#[derive(Debug)]
enum Command {
    // Embs
    AllocateEmb(usize),
    DeallocateEmb(usize),

    // KvBlocks
    AllocateKvBlock(usize),
    DeallocateKvBlock(usize),
    CopyKvBlock(usize, usize, usize, usize, usize),
    MaskKvBlock(usize, Vec<bool>),
    FillKvBlock(usize, Vec<usize>, Vec<bool>, Vec<usize>, Vec<usize>),
}

#[derive(Debug, Clone)]
pub struct Backend {
    block_size: usize,
    inner: Rc<RefCell<BackendInner>>,
}

// more sophisticated forms include: MultiNodeBackend, etc.

#[derive(Debug)]
struct BackendInner {
    kv_block_id_pool: IdPool<RemoteObjId>,
    emb_id_pool: IdPool<RemoteObjId>,
    cmd_queue: Vec<(StreamId, Command)>,
}

impl Backend {
    pub fn new(block_size: usize, max_kv_blocks: usize, max_embs: usize) -> Self {
        Self {
            block_size,
            inner: Rc::new(RefCell::new(BackendInner {
                kv_block_id_pool: IdPool::new(max_kv_blocks),
                emb_id_pool: IdPool::new(max_embs),
                cmd_queue: Vec::new(),
            })),
        }
    }

    pub fn enqueue_cmd(&self, stream_id: &StreamId, cmd: Command) -> Result<(), BlockError> {
        self.inner.borrow_mut().cmd_queue.push((*stream_id, cmd));
        Ok(())
    }
}

impl ObjectAllocator<TokenEmb> for Backend {
    type RawRepr = Vec<f32>;

    fn alloc(&self, stream_id: &StreamId) -> Result<RemoteObjId, BlockError> {
        let new_obj_id = self
            .inner
            .borrow_mut()
            .emb_id_pool
            .acquire()
            .ok_or(BlockError::NoFreeBlocks)?;

        let cmd = Command::AllocateEmb(new_obj_id);
        self.enqueue_cmd(stream_id, cmd)?;

        Ok(new_obj_id)
    }

    fn dealloc(&self, stream_id: &StreamId, obj_id: RemoteObjId) -> Result<(), BlockError> {
        // Release the object id back to the pool.
        self.inner.borrow_mut().emb_id_pool.release(obj_id)?;

        let cmd = Command::DeallocateEmb(obj_id);
        self.enqueue_cmd(stream_id, cmd)?;

        Ok(())
    }

    fn raw_repr(
        &self,
        stream_id: &InstanceId,
        obj_id: RemoteObjId,
        sender: Sender<Self::RawRepr>,
    ) -> Result<(), BlockError> {
        todo!()
    }

    fn available(&self) -> usize {
        self.inner.borrow().emb_id_pool.available()
    }
}

impl ObjectAllocator<KvBlock> for Backend {
    // The raw representation of a kv block is not really useful in any way. So we just use usize.
    type RawRepr = usize;

    fn alloc(&self, stream_id: &StreamId) -> Result<RemoteObjId, BlockError> {
        let new_obj_id = self
            .inner
            .borrow_mut()
            .kv_block_id_pool
            .acquire()
            .ok_or(BlockError::NoFreeBlocks)?;

        let cmd = Command::AllocateKvBlock(new_obj_id);
        self.enqueue_cmd(stream_id, cmd)?;

        Ok(new_obj_id)
    }

    fn dealloc(&self, stream_id: &StreamId, obj_id: RemoteObjId) -> Result<(), BlockError> {
        // Release the object id back to the pool.
        self.inner.borrow_mut().kv_block_id_pool.release(obj_id)?;

        let cmd = Command::DeallocateKvBlock(obj_id);
        self.enqueue_cmd(stream_id, cmd)?;

        Ok(())
    }

    fn raw_repr(
        &self,
        stream_id: &InstanceId,
        obj_id: RemoteObjId,
        sender: Sender<Self::RawRepr>,
    ) -> Result<(), BlockError> {
        sender.send(obj_id).unwrap();
        Ok(())
    }

    fn available(&self) -> usize {
        self.inner.borrow().kv_block_id_pool.available()
    }
}

impl ObjectAllocator<TokenDist> for Backend {
    type RawRepr = Vec<f32>;

    fn alloc(&self, stream_id: &InstanceId) -> Result<RemoteObjId, BlockError> {
        todo!()
    }

    fn dealloc(&self, stream_id: &InstanceId, obj_id: RemoteObjId) -> Result<(), BlockError> {
        todo!()
    }

    fn raw_repr(
        &self,
        stream_id: &InstanceId,
        obj_id: RemoteObjId,
        sender: Sender<Self::RawRepr>,
    ) -> Result<(), BlockError> {
        todo!()
    }

    fn available(&self) -> usize {
        todo!()
    }
}

impl CausalTransformer for Backend {
    fn fill(
        &self,
        stream_id: &StreamId,
        ptr: RemoteObjId,
        ctx_ptrs: Vec<RemoteObjId>,
        mask: Vec<bool>,
        input_embs: Vec<RemoteObjId>,
        output_embs: Vec<RemoteObjId>,
    ) -> Result<(), BlockError> {
        // create "resolved" cmd.

        let cmd = Command::FillKvBlock(ptr, ctx_ptrs, mask, input_embs, output_embs);

        self.enqueue_cmd(stream_id, cmd)?;
        Ok(())
    }

    fn copy_tokens(
        &self,
        stream_id: &StreamId,
        src_ptr: RemoteObjId,
        dst_ptr: RemoteObjId,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
    ) -> Result<(), BlockError> {
        let cmd = Command::CopyKvBlock(src_ptr, dst_ptr, src_offset, dst_offset, size);

        self.enqueue_cmd(stream_id, cmd)?;
        Ok(())
    }

    fn mask_tokens(
        &self,
        stream_id: &StreamId,
        ptr: RemoteObjId,
        mask: &[bool],
    ) -> Result<(), BlockError> {
        let cmd = Command::MaskKvBlock(ptr, mask.to_vec());
        self.enqueue_cmd(stream_id, cmd)?;
        Ok(())
    }
}

/// for multimodal LLMs

impl ImageEmbedder for Backend {
    fn embed_img(
        &self,
        stream_id: &StreamId,
        addrs: Vec<RemoteObjId>,
        url: String,
    ) -> Result<(), BlockError> {
        todo!()
    }
}
