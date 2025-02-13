use crate::state::{
    BlockError, CausalLanguageModel, CausalTransformer, ImageEmbedder, KvBlock, ObjectAllocator,
    RemoteObjId, StreamId, TokenDist, TokenEmb,
};
use crate::utils::IdPool;
use prost::Message;
use std::collections::HashMap;
use std::mem;
use std::sync::{Arc, Mutex};
use tokio::sync::oneshot::Sender;

mod sdi {
    include!(concat!(env!("OUT_DIR"), "/sdi.rs"));
}

/// Intermediate representation of a command to be executed by the backend.
/// This must not be exposed to other modules.
#[derive(Debug)]
enum IrCommand {
    // Embs
    AllocateEmb(u32),
    DeallocateEmb(u32),

    // KvBlocks
    AllocateKvBlock(u32),
    DeallocateKvBlock(u32),
    CopyKvBlock(u32, u32, u32, u32, u32),
    MaskKvBlock(u32, Vec<bool>),
    FillKvBlock(u32, Vec<u32>, Vec<u32>, Option<Vec<u32>>),
}

// Define actual cmd, interpretable by the backend.

// Allocate(Type, entities:[id])
// Deallocate(Type, entities:[id])
// Fill (List[id,
// LongFill using bitsets.. implement this later.

#[derive(Debug, Clone)]
pub struct Backend {
    block_size: u32,
    namespace: Arc<Mutex<ObjNamespace>>,

    cmd_buffer: Arc<Mutex<Vec<(StreamId, IrCommand)>>>,

    pending: Arc<Mutex<Pending>>,
    staged: Arc<Mutex<Vec<sdi::Command>>>,
    submitted: Arc<Mutex<Vec<sdi::Command>>>,
    // queue, cmd_buffer, scheduled, submitted
}

// more sophisticated forms include: MultiNodeBackend, etc.
#[derive(Debug)]
struct ObjNamespace {
    kv_block_id_pool: IdPool<RemoteObjId>,
    emb_id_pool: IdPool<RemoteObjId>,
}
#[derive(Debug)]
struct Pending {
    // these cmds are only be fired when it contains "enough" commands to be batched.
    fill_block: BatchStrategy<sdi::FillBlock>,
    copy_block: BatchStrategy<sdi::CopyBlock>,
    embed_image: BatchStrategy<sdi::EmbedImage>,
    decode: BatchStrategy<sdi::DecodeRequest>,
}

/// "K-or-T" Strategy
// 	For instance: If queue size reaches K, launch immediately; otherwise launch after T ms if K isn’t reached.
// 	This ensures that the GPU does not stay idle for too long (bounded by T) and that short bursts of arrivals form a large enough batch to get good utilization (bounded by K).
#[derive(Debug)]
struct BatchStrategy<T> {
    cmd: T,

    max_wait_time: f64,
    max_size: u32,

    current_wait_time: f64,
    current_size: u32,
}

impl<T> BatchStrategy<T> {
    fn new(cmd: T, max_wait_time: f64, max_size: u32) -> Self {
        Self {
            cmd,
            max_wait_time,
            max_size,
            current_wait_time: 0.0,
            current_size: 0,
        }
    }

    fn clear(&mut self) {
        self.current_wait_time = 0.0;
        self.current_size = 0;
    }

    fn inc(&mut self, time_elapsed: f64) {
        self.current_size += 1;
        self.current_wait_time += time_elapsed;
    }

    fn is_ready(&self) -> bool {
        self.current_size >= self.max_size
            || (self.current_wait_time >= self.max_wait_time && self.current_size > 0)
    }
}

impl Pending {
    fn new(max_wait_time: f64, max_size: u32) -> Self {
        Self {
            fill_block: BatchStrategy::new(sdi::FillBlock::default(), max_wait_time, max_size),
            copy_block: BatchStrategy::new(sdi::CopyBlock::default(), max_wait_time, max_size),
            embed_image: BatchStrategy::new(sdi::EmbedImage::default(), max_wait_time, max_size),
            decode: BatchStrategy::new(sdi::DecodeRequest::default(), max_wait_time, max_size),
        }
    }
}

impl ObjNamespace {
    fn new(max_kv_blocks: u32, max_embs: u32) -> Self {
        Self {
            kv_block_id_pool: IdPool::new(max_kv_blocks),
            emb_id_pool: IdPool::new(max_embs),
        }
    }

    fn acquire_id(&mut self, namespace: usize) -> Result<RemoteObjId, BlockError> {
        match namespace {
            0 => self.kv_block_id_pool.acquire(),
            1 => self.emb_id_pool.acquire(),
            _ => return Err(BlockError::VirtualAddressTranslationFailed),
        }
        .ok_or(BlockError::NoFreeBlocks)
    }

    fn release_id(&mut self, namespace: usize, id: RemoteObjId) -> Result<(), BlockError> {
        match namespace {
            0 => self.kv_block_id_pool.release(id),
            1 => self.emb_id_pool.release(id),
            _ => Err(BlockError::VirtualAddressTranslationFailed),
        }
    }

    fn remaining_ids(&self, namespace: usize) -> usize {
        match namespace {
            0 => self.kv_block_id_pool.available(),
            1 => self.emb_id_pool.available(),
            _ => 0,
        }
    }
}

impl Backend {
    pub fn new(block_size: u32, max_kv_blocks: u32, max_embs: u32) -> Self {
        Self {
            block_size,
            namespace: Arc::new(Mutex::new(ObjNamespace::new(max_kv_blocks, max_embs))),
            cmd_buffer: Arc::new(Mutex::new(Vec::new())),
            pending: Arc::new(Mutex::new(Pending::new(10.0, 0))),
            staged: Arc::new(Mutex::new(Vec::new())),
            submitted: Arc::new(Mutex::new(vec![])),
        }
    }

    pub fn enqueue_cmd(&self, stream_id: StreamId, cmd: IrCommand) -> Result<(), BlockError> {
        let mut inner = self.cmd_buffer.lock().map_err(|_| BlockError::LockError)?;
        inner.push((stream_id, cmd));
        Ok(())
    }

    fn acquire_id(&self, namespace: usize) -> Result<RemoteObjId, BlockError> {
        let mut inner = self.namespace.lock().map_err(|_| BlockError::LockError)?;
        inner.acquire_id(namespace)
    }

    fn release_id(&self, namespace: usize, id: RemoteObjId) -> Result<(), BlockError> {
        let mut inner = self.namespace.lock().map_err(|_| BlockError::LockError)?;
        inner.release_id(namespace, id)
    }

    fn remaining_ids(&self, namespace: usize) -> usize {
        let inner = self.namespace.lock().unwrap();
        inner.remaining_ids(namespace)
    }

    pub fn schedule(&self, time_elapsed: f64) -> Result<(), BlockError> {
        let mut pending = self.pending.lock().map_err(|_| BlockError::LockError)?;

        // first move all the commands from cmd_buffer to pending (buffer items are removed)
        let mut commands_by_stream = {
            let mut cmd_buffer = self.cmd_buffer.lock().map_err(|_| BlockError::LockError)?;

            let mut stream_commands = HashMap::new();

            for (stream_id, command) in cmd_buffer.drain(..) {
                stream_commands
                    .entry(stream_id)
                    .or_insert_with(Vec::new)
                    .push(command);
            }

            stream_commands
            // drop the lock on cmd_buffer
        };

        // do batching!!
        //
        // cmd_set = HashMap<CmdType, Vec<Cmd>>
        //
        let mut alloc_cmd = sdi::Allocate {
            kind: Vec::new(),
            object_ids: Vec::new(),
        };

        let mut dealloc_cmd = sdi::Deallocate {
            kind: Vec::new(),
            object_ids: Vec::new(),
        };

        let mut mask_cmd = sdi::MaskBlock { items: Vec::new() };

        for (_stream_id, cmd_list) in commands_by_stream.iter_mut() {
            let mut processed_count = 0;

            for cmd in cmd_list.iter() {
                let mut should_stop = true;
                match cmd {
                    IrCommand::AllocateEmb(id) => {
                        alloc_cmd.kind.push(sdi::ObjectKind::Emb.into());
                        alloc_cmd.object_ids.push(*id);
                        // Processed, so keep_on remains false (we can drop this command)
                    }
                    IrCommand::DeallocateEmb(id) => {
                        dealloc_cmd.kind.push(sdi::ObjectKind::Emb.into());
                        dealloc_cmd.object_ids.push(*id);
                    }
                    IrCommand::AllocateKvBlock(id) => {
                        alloc_cmd.kind.push(sdi::ObjectKind::KvBlock.into());
                        alloc_cmd.object_ids.push(*id);
                    }
                    IrCommand::DeallocateKvBlock(id) => {
                        dealloc_cmd.kind.push(sdi::ObjectKind::KvBlock.into());
                        dealloc_cmd.object_ids.push(*id);
                    }
                    IrCommand::CopyKvBlock(src, dst, src_offset, dst_offset, size) => {
                        pending.copy_block.cmd.items.push(sdi::CopyBlockItem {
                            source_block_id: *src,
                            destination_block_id: *dst,
                            source_start: *src_offset,
                            destination_start: *dst_offset,
                            length: *size,
                        });
                    }
                    IrCommand::MaskKvBlock(id, mask) => {
                        mask_cmd.items.push(sdi::MaskBlockItem {
                            block_id: *id,
                            mask: mask.to_vec(),
                        });
                    }
                    IrCommand::FillKvBlock(ptr, ctx_ptrs, input_embs, output_embs) => {
                        pending.fill_block.cmd.items.push(sdi::FillBlockItem {
                            block_id: *ptr,
                            context_block_ids: ctx_ptrs.to_vec(),
                            input_embedding_ids: input_embs.to_vec(),
                            output_embedding_ids: output_embs.clone().unwrap_or_default(),
                        });

                        should_stop = false;
                    }
                }
                processed_count += 1;
                if should_stop {
                    break;
                }
            }

            // Only remove the commands that were processed.
            cmd_list.drain(0..processed_count);
        }

        // add the commands to the staged queue
        let mut staged = self.staged.lock().map_err(|_| BlockError::LockError)?;

        if alloc_cmd.object_ids.len() > 0 {
            staged.push(sdi::Command {
                correlation_id: 0,
                payload: Some(sdi::command::Payload::Allocate(alloc_cmd)),
            });
        }

        if dealloc_cmd.object_ids.len() > 0 {
            staged.push(sdi::Command {
                correlation_id: 0,
                payload: Some(sdi::command::Payload::Deallocate(dealloc_cmd)),
            });
        }

        if mask_cmd.items.len() > 0 {
            staged.push(sdi::Command {
                correlation_id: 0,
                payload: Some(sdi::command::Payload::MaskBlock(mask_cmd)),
            });
        }

        // fill decision.
        if pending.fill_block.is_ready() {
            staged.push(sdi::Command {
                correlation_id: 0,
                payload: Some(sdi::command::Payload::FillBlock(mem::take(
                    &mut pending.fill_block.cmd,
                ))),
            });
            pending.fill_block.clear();
        }

        // drop the lock on pending
        Ok(())
    }
}

impl ObjectAllocator<TokenEmb> for Backend {
    type RawRepr = Vec<f32>;

    fn alloc(&self, stream_id: StreamId) -> Result<RemoteObjId, BlockError> {
        let new_obj_id = self.acquire_id(1)?;

        let cmd = IrCommand::AllocateEmb(new_obj_id);
        self.enqueue_cmd(stream_id, cmd)?;

        Ok(new_obj_id)
    }

    fn dealloc(&self, stream_id: StreamId, obj_id: RemoteObjId) -> Result<(), BlockError> {
        // Release the object id back to the pool.
        self.release_id(1, obj_id)?;

        let cmd = IrCommand::DeallocateEmb(obj_id);
        self.enqueue_cmd(stream_id, cmd)?;

        Ok(())
    }

    fn raw_repr(
        &self,
        stream_id: StreamId,
        obj_id: RemoteObjId,
        sender: Sender<Self::RawRepr>,
    ) -> Result<(), BlockError> {
        todo!()
    }

    fn available(&self) -> usize {
        self.remaining_ids(1)
    }
}

impl ObjectAllocator<KvBlock> for Backend {
    // The raw representation of a kv block is not really useful in any way. So we just use usize.
    type RawRepr = RemoteObjId;

    fn alloc(&self, stream_id: StreamId) -> Result<RemoteObjId, BlockError> {
        let new_obj_id = self.acquire_id(0)?;

        let cmd = IrCommand::AllocateKvBlock(new_obj_id);
        self.enqueue_cmd(stream_id, cmd)?;

        Ok(new_obj_id)
    }

    fn dealloc(&self, stream_id: StreamId, obj_id: RemoteObjId) -> Result<(), BlockError> {
        // Release the object id back to the pool.
        self.release_id(0, obj_id)?;

        let cmd = IrCommand::DeallocateKvBlock(obj_id);
        self.enqueue_cmd(stream_id, cmd)?;

        Ok(())
    }

    fn raw_repr(
        &self,
        stream_id: StreamId,
        obj_id: RemoteObjId,
        sender: Sender<Self::RawRepr>,
    ) -> Result<(), BlockError> {
        sender.send(obj_id).unwrap();
        Ok(())
    }

    fn available(&self) -> usize {
        self.remaining_ids(0)
    }
}

impl ObjectAllocator<TokenDist> for Backend {
    type RawRepr = Vec<f32>;

    fn alloc(&self, stream_id: StreamId) -> Result<RemoteObjId, BlockError> {
        todo!()
    }

    fn dealloc(&self, stream_id: StreamId, obj_id: RemoteObjId) -> Result<(), BlockError> {
        todo!()
    }

    fn raw_repr(
        &self,
        stream_id: StreamId,
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
        stream_id: StreamId,
        ptr: RemoteObjId,
        ctx_ptrs: Vec<RemoteObjId>,
        input_embs: Vec<RemoteObjId>,
        output_embs: Option<Vec<RemoteObjId>>,
    ) -> Result<(), BlockError> {
        // create "resolved" cmd.

        let cmd = IrCommand::FillKvBlock(ptr, ctx_ptrs, input_embs, output_embs);

        self.enqueue_cmd(stream_id, cmd)?;
        Ok(())
    }

    fn copy_tokens(
        &self,
        stream_id: StreamId,
        src_ptr: RemoteObjId,
        dst_ptr: RemoteObjId,
        src_offset: u32,
        dst_offset: u32,
        size: u32,
    ) -> Result<(), BlockError> {
        let cmd = IrCommand::CopyKvBlock(src_ptr, dst_ptr, src_offset, dst_offset, size);

        self.enqueue_cmd(stream_id, cmd)?;
        Ok(())
    }

    fn mask_tokens(
        &self,
        stream_id: StreamId,
        ptr: RemoteObjId,
        mask: &[bool],
    ) -> Result<(), BlockError> {
        let cmd = IrCommand::MaskKvBlock(ptr, mask.to_vec());
        self.enqueue_cmd(stream_id, cmd)?;
        Ok(())
    }
}

impl CausalLanguageModel for Backend {
    fn next_token_dist(
        &self,
        stream_id: StreamId,
        emb_ptr: RemoteObjId,
        dist_ptr: RemoteObjId,
    ) -> Result<(), BlockError> {
        todo!()
    }

    fn sample_top_k(
        &self,
        stream_id: StreamId,
        dist_ptr: RemoteObjId,
        k: usize,
        sender: Sender<Vec<usize>>,
    ) -> Result<(), BlockError> {
        todo!()
    }

    fn get_raw_dist(
        &self,
        stream_id: StreamId,
        dist_ptr: RemoteObjId,
        sender: Sender<Vec<f32>>,
    ) -> Result<(), BlockError> {
        todo!()
    }
}

/// for multimodal LLMs

impl ImageEmbedder for Backend {
    fn embed_img(
        &self,
        stream_id: StreamId,
        addrs: Vec<RemoteObjId>,
        url: String,
    ) -> Result<(), BlockError> {
        todo!()
    }
}
