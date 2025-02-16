use crate::lm::{
    CausalLanguageModel, CausalTransformer, ImageEmbedder, InstanceId, KvBlock, TokenDist, TokenEmb,
};
use crate::utils::{Counter, Stream};
use crate::{backend, object, utils};
use std::collections::HashMap;
use std::fmt::Debug;
use std::mem;
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use zeromq::{DealerSocket, Socket, SocketRecv, SocketSend, ZmqError, ZmqMessage};

use crate::object::{Id, ObjectError, VspaceId};
use thiserror::Error;
use tokio::task::JoinError;

#[derive(Error, Debug)]
pub enum ControllerError {
    #[error("Mutex lock failed")]
    LockError,

    #[error("Socket transmitter not available")]
    MissingSocket,

    #[error("Channel send error")]
    SendError,

    #[error("ZeroMQ error: {0}")]
    ZmqError(#[from] ZmqError),

    #[error("Task join error: {0}")]
    JoinError(#[from] JoinError),

    #[error("Decode error: {0}")]
    DecodeError(#[from] prost::DecodeError),

    #[error("Object error: {0}")]
    ObjectError(#[from] ObjectError),
}

/// Intermediate representation of a command to be executed by the backend.
/// This must not be exposed to other modules.
#[derive(Debug)]
enum IrCommand {
    // Embs
    Allocate(backend::sdi::Allocate),
    Deallocate(backend::sdi::Allocate),
    CopyBlock(backend::sdi::CopyBlock),
    MaskBlock(backend::sdi::MaskBlock),
    FillBlock(backend::sdi::FillBlock),
    EmbedImage(backend::sdi::EmbedImage),
    EmbedText(backend::sdi::EmbedText),
    DecodeTokenDistribution(backend::sdi::DecodeTokenDistribution),
    SampleTopKRequest(backend::sdi::SampleTopKRequest),
    GetTokenDistributionRequest(backend::sdi::GetTokenDistributionRequest),
}

// Hidden
#[derive(Debug)]
enum IrEvent {
    SampleTopK(backend::sdi::SampleTopKResponse),
    GetTokenDistribution(backend::sdi::GetTokenDistributionResponse),
}

#[derive(Debug)]
pub struct Resource {
    owner_id: InstanceId,
    addrs: Vec<object::Id<KvBlock>>,
}

impl Resource {
    pub fn new(owner_id: InstanceId, addrs: Vec<object::Id<KvBlock>>) -> Self {
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

#[derive(Debug)]
pub enum EventHandle {
    None,
    SampleTopK(oneshot::Sender<Vec<u32>>),
    GetTokenDistribution(oneshot::Sender<Vec<f32>>),
}

impl EventHandle {
    fn is_some(&self) -> bool {
        match self {
            EventHandle::None => false,
            _ => true,
        }
    }
}

#[derive(Debug)]
pub struct Controller<B> {
    block_size: u32,

    cmd_buffer: Vec<(Stream, IrCommand, EventHandle)>,
    cmd_batcher: CommandBatcher,
    backend: B,

    // object allocations
    id_pool: IdPool,
    ref_counter: RefCounter,
    vspaces: HashMap<object::VspaceId, IdMap>,
}

impl<B> Controller<B> {
    pub fn new(backend: B) -> Self {
        Self {
            block_size,
            cmd_buffer: Vec::new(),
            cmd_batcher: CommandBatcher::new(0.0, 1, 1),
            backend,
            id_pool: IdPool::new(max_kv_blocks, max_embs),
            ref_counter: RefCounter::new(),
            vspaces: HashMap::new(),
        }
    }

    pub fn enqueue_cmd(&mut self, stream: Stream, cmd: IrCommand) -> Result<(), ControllerError> {
        self.cmd_buffer.push((stream, cmd, EventHandle::None));
        Ok(())
    }

    pub fn enqueue_cmd_with_event(
        &mut self,
        stream: Stream,
        cmd: IrCommand,
        evt: EventHandle,
    ) -> Result<(), ControllerError> {
        self.cmd_buffer.push((stream, cmd, evt));
        Ok(())
    }

    pub fn schedule(&self, curr_timestamp: f64) -> Result<(), ControllerError> {
        let mut pending = self
            .cmd_batcher
            .lock()
            .map_err(|_| ControllerError::LockError)?;

        // first move all the commands from cmd_buffer to pending (buffer items are removed)
        let mut commands_by_stream = {
            let mut cmd_buffer = self
                .cmd_batcher
                .lock()
                .map_err(|_| ControllerError::LockError)?;

            let mut stream_commands = HashMap::new();

            for (stream_id, command, sender) in cmd_buffer.drain(..) {
                stream_commands
                    .entry(stream_id)
                    .or_insert_with(Vec::new)
                    .push((command, sender));
            }

            stream_commands
            // drop the lock on cmd_buffer
        };

        // Horizontal batching: group commands by stream and type.
        for (_stream_id, cmd_list) in commands_by_stream.iter_mut() {
            let mut prev_cmd = None;

            loop {
                if cmd_list.is_empty() {
                    break;
                }
                let (cmd, sender) = cmd_list.pop().unwrap();
                let curr_cmd = mem::discriminant(&cmd);

                // Vertical batching: Same kind of consecutive commands are batched together.
                // if the current command is different from the previous one, stop batching.
                if let Some(prev_cmd) = prev_cmd {
                    if prev_cmd != curr_cmd {
                        break;
                    }
                }

                pending.push(cmd, curr_timestamp, sender);
                prev_cmd = Some(curr_cmd);
            }
        }

        let batched_payloads = pending.batch_all(curr_timestamp);

        // add the commands to the staged queue
        let mut staged = self.staged.lock().map_err(|_| ControllerError::LockError)?;

        // Add the batched commands to the staged queue.
        for (payload, evt_handles) in batched_payloads {
            let correlation_id = self.acquire_id(object::Namespace::Cmd)?;

            staged.push(sdi::Command {
                correlation_id,
                payload: Some(payload),
            });

            // if at least one sender is present, add it to the event dispatcher.
            let has_event = evt_handles.iter().any(|s| s.is_some());
            if has_event {
                let mut dispatcher = self
                    .event_dispatcher
                    .lock()
                    .map_err(|_| ControllerError::LockError)?;
                dispatcher.table.insert(correlation_id, evt_handles);
            }
        }

        // drop the lock on pending
        Ok(())
    }
}

// more sophisticated forms include: MultiNodeBackend, etc.

#[derive(Debug)]
struct CommandBatcher {
    allocate: BatchQueue<sdi::Allocate>,
    deallocate: BatchQueue<sdi::Allocate>,
    copy_block: BatchQueue<sdi::CopyBlock>,
    mask_block: BatchQueue<sdi::MaskBlock>,
    embed_text: BatchQueue<sdi::EmbedText>,
    embed_image: BatchQueue<sdi::EmbedImage>,

    // these cmds are only be fired when it contains "enough" commands to be batched.
    fill_block: BatchQueue<sdi::FillBlock>,
    decode_token_distribution: BatchQueue<sdi::DecodeTokenDistribution>,
    sample_top_k: BatchQueue<sdi::SampleTopKRequest>,
}

/// "K-or-T" Strategy
// 	For instance: If queue size reaches K, launch immediately; otherwise launch after T ms if K isn’t reached.
// 	This ensures that the GPU does not stay idle for too long (bounded by T) and that short bursts of arrivals form a large enough batch to get good utilization (bounded by K).
#[derive(Debug)]
struct BatchQueue<T> {
    // cmd, timestamp, response_sender
    items: Vec<(T, f64, EventHandle)>,

    max_wait_time: f64,
    min_size: usize,
    max_size: usize,
}

impl<T> BatchQueue<T> {
    fn eager() -> Self {
        Self {
            items: Vec::new(),
            max_wait_time: 0.0,
            min_size: 1,
            max_size: usize::MAX,
        }
    }

    fn k_only(min_size: usize, max_size: Option<usize>) -> Self {
        Self {
            items: Vec::new(),
            max_wait_time: f64::MAX,
            min_size,
            max_size: max_size.unwrap_or(min_size),
        }
    }

    fn t_only(max_wait_time: f64) -> Self {
        Self {
            items: Vec::new(),
            max_wait_time,
            min_size: 1,
            max_size: usize::MAX,
        }
    }

    fn k_or_t(max_wait_time: f64, min_size: usize, max_size: Option<usize>) -> Self {
        Self {
            items: Vec::new(),
            max_wait_time,
            min_size,
            max_size: max_size.unwrap_or(min_size),
        }
    }

    fn take(&mut self) -> (Vec<T>, Vec<EventHandle>) {
        let drain_count = self.items.len().min(self.max_size);
        self.items
            .drain(..drain_count)
            .map(|(item, _, sender)| (item, sender))
            .unzip()
    }

    fn push(&mut self, item: T, curr_timestamp: f64, evt: EventHandle) {
        self.items.push((item, curr_timestamp, evt));
    }

    fn is_ready(&self, curr_timestamp: f64) -> bool {
        let num_items = self.items.len();

        if num_items > 0 {
            let longest_wait_time = curr_timestamp - self.items[0].1;
            if num_items >= self.min_size || longest_wait_time >= self.max_wait_time {
                return true;
            }
        }
        false
    }

    fn batch(&mut self, curr_timestamp: f64) -> Option<(Vec<T>, Vec<EventHandle>)> {
        if self.is_ready(curr_timestamp) {
            Some(self.take())
        } else {
            None
        }
    }
}

impl CommandBatcher {
    fn new(max_wait_time: f64, min_size: usize, max_size: usize) -> Self {
        Self {
            allocate: BatchQueue::eager(),
            deallocate: BatchQueue::eager(),
            copy_block: BatchQueue::k_or_t(max_wait_time, min_size, Some(max_size)),
            mask_block: BatchQueue::eager(),
            embed_text: BatchQueue::eager(),
            embed_image: BatchQueue::k_or_t(max_wait_time, min_size, Some(max_size)),
            fill_block: BatchQueue::k_or_t(max_wait_time, min_size, Some(max_size)),
            sample_top_k: BatchQueue::k_or_t(max_wait_time, min_size, Some(max_size)),
            decode_token_distribution: BatchQueue::eager(),
        }
    }

    fn push(&mut self, cmd: IrCommand, curr_timestamp: f64, evt: EventHandle) {
        match cmd {
            IrCommand::Allocate(item) => {
                self.allocate.push(item, curr_timestamp, evt);
            }
            IrCommand::Deallocate(item) => {
                self.deallocate.push(item, curr_timestamp, evt);
            }
            IrCommand::CopyBlock(item) => {
                self.copy_block.push(item, curr_timestamp, evt);
            }
            IrCommand::MaskBlock(item) => {
                self.mask_block.push(item, curr_timestamp, evt);
            }
            IrCommand::FillBlock(item) => {
                self.fill_block.push(item, curr_timestamp, evt);
            }
            IrCommand::EmbedImage(item) => {
                self.embed_image.push(item, curr_timestamp, evt);
            }
            IrCommand::EmbedText(item) => {
                self.embed_text.push(item, curr_timestamp, evt);
            }
            IrCommand::SampleTopKRequest(item) => {
                self.sample_top_k.push(item, curr_timestamp, evt);
            }
            IrCommand::DecodeTokenDistribution(item) => {
                self.decode_token_distribution
                    .push(item, curr_timestamp, evt);
            }
            IrCommand::GetTokenDistributionRequest(_) => todo!(),
        }
    }

    fn batch_all(&mut self, curr_timestamp: f64) -> Vec<(sdi::command::Payload, Vec<EventHandle>)> {
        let mut cmds = Vec::new();

        if let Some((items, senders)) = self.allocate.batch(curr_timestamp) {
            cmds.push((
                sdi::command::Payload::Allocate(sdi::BatchAllocate { items }),
                senders,
            ));
        }

        if let Some((items, senders)) = self.deallocate.batch(curr_timestamp) {
            cmds.push((
                sdi::command::Payload::Deallocate(sdi::BatchDeallocate { items }),
                senders,
            ));
        }

        if let Some((items, senders)) = self.copy_block.batch(curr_timestamp) {
            cmds.push((
                sdi::command::Payload::CopyBlock(sdi::BatchCopyBlock { items }),
                senders,
            ));
        }

        if let Some((items, senders)) = self.mask_block.batch(curr_timestamp) {
            cmds.push((
                sdi::command::Payload::MaskBlock(sdi::BatchMaskBlock { items }),
                senders,
            ));
        }

        if let Some((items, senders)) = self.embed_text.batch(curr_timestamp) {
            cmds.push((
                sdi::command::Payload::EmbedText(sdi::BatchEmbedText { items }),
                senders,
            ));
        }

        if let Some((items, senders)) = self.embed_image.batch(curr_timestamp) {
            cmds.push((
                sdi::command::Payload::EmbedImage(sdi::BatchEmbedImage { items }),
                senders,
            ));
        }

        if let Some((items, senders)) = self.fill_block.batch(curr_timestamp) {
            cmds.push((
                sdi::command::Payload::FillBlock(sdi::BatchFillBlock { items }),
                senders,
            ));
        }

        if let Some((items, senders)) = self.sample_top_k.batch(curr_timestamp) {
            cmds.push((
                sdi::command::Payload::SampleTopKRequest(sdi::BatchSampleTopKRequest { items }),
                senders,
            ));
        }

        cmds
    }
}

pub trait ControllerManaged: Debug + Sized + Send + Sync {
    const NAMESPACE: Namespace;
    //fn get_namespace() -> Namespace;

    pub fn sdi_object_kind() -> i32 {
        match Self::NAMESPACE {
            Namespace::KvBlock => backend::sdi::ObjectKind::KvBlock.into(),
            Namespace::Emb => backend::sdi::ObjectKind::Emb.into(),
            Namespace::Dist => backend::sdi::ObjectKind::Dist.into(),
        }
    }
}

impl ControllerManaged for KvBlock {
    const NAMESPACE: Namespace = Namespace::KvBlock;
}

impl ControllerManaged for TokenEmb {
    const NAMESPACE: Namespace = Namespace::Emb;
}

impl ControllerManaged for TokenDist {
    const NAMESPACE: Namespace = Namespace::Dist;
}

#[derive(Debug, Clone, Copy)]
pub enum Namespace {
    KvBlock = 0,
    Emb = 1,
    Dist = 2,
}

impl<B, T> object::Allocator<T> for Controller<B>
where
    T: ControllerManaged,
{
    fn alloc_many(
        &mut self,
        stream: Stream,
        count: usize,
    ) -> Result<Vec<object::Id<T>>, ObjectError> {
        let ids = self.id_pool.acquire_many::<T>(count)?;

        // init the ref counter
        for id in &ids {
            self.ref_counter.init(*id);
        }

        // Request the backend to allocate the objects.
        let kind = T::sdi_object_kind();
        let cons_id = object::Id::group_consecutive_ids(&ids);

        for (id_offset, size) in cons_id {
            let cmd = IrCommand::Allocate(backend::sdi::Allocate {
                kind,
                object_id_offset: id_offset.into(),
                count: size,
            });
            self.cmd_buffer.push((stream, cmd, EventHandle::None));
        }
        Ok(ids)
    }

    fn dealloc_many(&mut self, stream: Stream, ids: &[object::Id<T>]) -> Result<(), ObjectError> {
        let cons_id = object::Id::group_consecutive_ids(ids);

        // destroy the ref counter
        for id in ids {
            self.ref_counter.destroy(id);
        }

        let kind = T::sdi_object_kind();

        for (id_offset, size) in cons_id {
            let cmd = IrCommand::Deallocate(backend::sdi::Allocate {
                kind,
                object_id_offset: id_offset.into(),
                count: size,
            });
            self.cmd_buffer.push((stream, cmd, EventHandle::None));
        }

        Ok(())
    }

    fn available(&self) -> usize {
        self.id_pool.available::<T>()
    }
}



impl<B, T> object::MappedAllocator<T> for Controller<B>
where
    T: ControllerManaged,
{
    fn vspaces(&self) -> &HashMap<VspaceId, HashMap<Id<T>, Id<T>>> {
        &self.vspaces
    }

    fn vspaces_mut(&mut self) -> &mut HashMap<VspaceId, HashMap<Id<T>, Id<T>>> {
        &mut self.vspaces
    }

    fn ref_inc(&mut self, id: &Id<T>) -> Result<(), ObjectError> {
        todo!()
    }

    fn ref_dec(&mut self, id: &Id<T>) -> Result<bool, ObjectError> {
        todo!()
    }

    fn ref_count(&self, id: &Id<T>) -> Result<usize, ObjectError> {
        todo!()
    }
}

impl CausalTransformer for Controller {
    fn fill(
        &self,
        stream: Stream,
        ptr: object::Id<KvBlock>,
        ctx_ptrs: Vec<object::Id<KvBlock>>,
        input_embs: Vec<object::Id<TokenEmb>>,
        output_embs: Vec<Option<object::Id<TokenEmb>>>,
    ) -> Result<(), ControllerError> {
        let cmd = IrCommand::FillBlock(sdi::FillBlock {
            block_id: ptr.into(),
            context_block_ids: ctx_ptrs.into_iter().map(|id| id.into()).collect(),
            input_embedding_ids: input_embs.into_iter().map(|id| id.into()).collect(),
            output_embedding_ids: output_embs
                .into_iter()
                .map(|id| id.map(|id| id.into()))
                .collect(),
        });

        self.enqueue_cmd(stream, cmd)
    }

    fn copy_tokens(
        &self,
        stream_id: Stream,
        src_ptr: object::Id<KvBlock>,
        dst_ptr: object::Id<KvBlock>,
        src_offset: u32,
        dst_offset: u32,
        size: u32,
    ) -> Result<(), ControllerError> {
        let cmd = IrCommand::CopyBlock(sdi::CopyBlock {
            source_block_id: src_ptr.into(),
            destination_block_id: dst_ptr.into(),
            source_start: src_offset,
            destination_start: dst_offset,
            length: size,
        });

        self.enqueue_cmd(stream_id, cmd)
    }

    fn mask_tokens(
        &self,
        stream_id: Stream,
        ptr: object::Id<KvBlock>,
        mask: &[bool],
    ) -> Result<(), ControllerError> {
        let cmd = IrCommand::MaskBlock(sdi::MaskBlock {
            block_id: ptr.into(),
            mask: mask.to_vec(),
        });
        self.enqueue_cmd(stream_id, cmd)
    }
}

impl CausalLanguageModel for Controller {
    fn next_token_dist(
        &self,
        stream_id: Stream,
        emb_ptr: object::Id<TokenEmb>,
        dist_ptr: object::Id<TokenDist>,
    ) -> Result<(), ControllerError> {
        let cmd = IrCommand::DecodeTokenDistribution(sdi::DecodeTokenDistribution {
            embedding_id: emb_ptr,
            distribution_id: dist_ptr,
        });

        self.enqueue_cmd(stream_id, cmd)
    }

    fn sample_top_k(
        &self,
        stream_id: Stream,
        dist_ptr: object::Id,
        k: u32,
    ) -> Result<oneshot::Receiver<Vec<u32>>, ControllerError> {
        // create a new event handle

        let cmd = IrCommand::SampleTopKRequest(sdi::SampleTopKRequest {
            distribution_id: dist_ptr,
            k,
        });

        let (tx, rx) = oneshot::channel::<Vec<u32>>();
        let handle = EventHandle::SampleTopK(tx);

        self.enqueue_cmd_with_event(stream_id, cmd, handle)?;
        Ok(rx)
    }

    fn get_raw_dist(
        &self,
        stream_id: Stream,
        dist_ptr: object::Id,
    ) -> Result<oneshot::Receiver<Vec<f32>>, ControllerError> {
        let cmd = IrCommand::GetTokenDistributionRequest(sdi::GetTokenDistributionRequest {
            distribution_id: dist_ptr,
        });

        let (tx, rx) = oneshot::channel::<Vec<f32>>();
        let handle = EventHandle::GetTokenDistribution(tx);

        self.enqueue_cmd_with_event(stream_id, cmd, handle)?;
        Ok(rx)
    }
}

/// for multimodal LLMs

impl ImageEmbedder for Controller {
    fn embed_img(
        &self,
        stream_id: Stream,
        addrs: Vec<object::Id>,
        url: String,
    ) -> Result<(), ControllerError> {
        let cmd = IrCommand::EmbedImage(sdi::EmbedImage {
            embedding_ids: addrs,
            url,
        });

        self.enqueue_cmd(stream_id, cmd)
    }
}

#[derive(Debug)]
pub struct IdPool {
    kv_block_id_pool: utils::IdPool<object::IdRepr>,
    emb_id_pool: utils::IdPool<object::IdRepr>,
    dist_id_pool: utils::IdPool<object::IdRepr>,
}
impl IdPool {
    pub fn new(max_kv_blocks: u32, max_embs: u32) -> Self {
        Self {
            kv_block_id_pool: utils::IdPool::new(max_kv_blocks),
            emb_id_pool: utils::IdPool::new(max_embs),
            dist_id_pool: utils::IdPool::new(max_embs),
        }
    }

    // Helper that returns a mutable reference to the appropriate pool.
    fn pool_mut<T: ControllerManaged>(&mut self) -> &mut utils::IdPool<object::IdRepr> {
        match T::NAMESPACE {
            Namespace::KvBlock => &mut self.kv_block_id_pool,
            Namespace::Emb => &mut self.emb_id_pool,
            Namespace::Dist => &mut self.dist_id_pool,
        }
    }

    // Helper that returns an immutable reference.
    fn pool<T: ControllerManaged>(&self) -> &utils::IdPool<object::IdRepr> {
        match T::NAMESPACE {
            Namespace::KvBlock => &self.kv_block_id_pool,
            Namespace::Emb => &self.emb_id_pool,
            Namespace::Dist => &self.dist_id_pool,
        }
    }

    pub fn acquire<T: ControllerManaged>(&mut self) -> Result<object::Id<T>, ObjectError> {
        let id = self
            .pool_mut::<T>()
            .acquire()
            .map_err(|_| ObjectError::NoAvailableSpace)?;
        Ok(object::Id::new(id))
    }

    pub fn acquire_many<T: ControllerManaged>(
        &mut self,
        count: usize,
    ) -> Result<Vec<object::Id<T>>, ObjectError> {
        let ids = self
            .pool_mut::<T>()
            .acquire_many(count)
            .map_err(|_| ObjectError::NoAvailableSpace)?;
        Ok(object::Id::map_from_repr(ids))
    }

    pub fn release<T: ControllerManaged>(&mut self, id: &object::Id<T>) -> Result<(), ObjectError> {
        self.pool_mut::<T>()
            .release(id.into())
            .map_err(|e| ObjectError::AddressPoolError(e.to_string()))
    }

    pub fn release_many<T: ControllerManaged>(
        &mut self,
        ids: &[object::Id<T>],
    ) -> Result<(), ObjectError> {
        let raw_ids = object::Id::ref_as_repr(ids);
        self.pool_mut::<T>()
            .release_many(raw_ids)
            .map_err(|e| ObjectError::AddressPoolError(e.to_string()))
    }

    pub fn available<T: ControllerManaged>(&self) -> usize {
        self.pool::<T>().available()
    }
}

#[derive(Debug)]
pub struct IdMap {
    kv_block_id_map: HashMap<object::VspaceId, HashMap<object::IdRepr, object::IdRepr>>,
    emb_id_map: HashMap<object::VspaceId, HashMap<object::IdRepr, object::IdRepr>>,
    dist_id_map: HashMap<object::VspaceId, HashMap<object::IdRepr, object::IdRepr>>,
}
impl IdMap {
    pub fn new() -> Self {
        Self {
            kv_block_id_map: HashMap::new(),
            emb_id_map: HashMap::new(),
            dist_id_map: HashMap::new(),
        }
    }

    // Helper method to get a mutable reference to the appropriate map.
    fn map_mut<T: ControllerManaged>(&mut self) -> &mut HashMap<object::IdRepr, object::IdRepr> {
        match T::NAMESPACE {
            Namespace::KvBlock => &mut self.kv_block_id_map,
            Namespace::Emb => &mut self.emb_id_map,
            Namespace::Dist => &mut self.dist_id_map,
        }
    }

    // Helper method to get an immutable reference to the appropriate map.
    fn map<T: ControllerManaged>(&self) -> &HashMap<object::IdRepr, object::IdRepr> {
        match T::NAMESPACE {
            Namespace::KvBlock => &self.kv_block_id_map,
            Namespace::Emb => &self.emb_id_map,
            Namespace::Dist => &self.dist_id_map,
        }
    }

    pub fn insert<T: ControllerManaged>(&mut self, vid: object::Id<T>, id: object::Id<T>) {
        let (src, dst) = (vid.into(), id.into());
        self.map_mut::<T>().insert(src, dst);
    }

    pub fn remove<T: ControllerManaged>(
        &mut self,
        vid: object::Id<T>,
    ) -> Result<object::Id<T>, ObjectError> {
        let vid = vid.into();
        let id = self
            .map_mut::<T>()
            .remove(&vid)
            .ok_or(ObjectError::ObjectNotFound)?;
        Ok(object::Id::<T>::new(id))
    }

    pub fn get<T: ControllerManaged>(
        &self,
        vid: object::Id<T>,
    ) -> Result<object::Id<T>, ObjectError> {
        let vid = vid.into();
        let id = self
            .map::<T>()
            .get(&vid)
            .ok_or(ObjectError::ObjectNotFound)?;
        Ok(object::Id::<T>::new(*id))
    }
}

#[derive(Debug)]
struct RefCounter {
    kv_block_counter: HashMap<object::Id<KvBlock>, Counter>,
    emb_counter: HashMap<object::Id<TokenEmb>, Counter>,
    dist_counter: HashMap<object::Id<TokenDist>, Counter>,
}

impl RefCounter {
    pub fn new() -> Self {
        Self {
            kv_block_counter: HashMap::new(),
            emb_counter: HashMap::new(),
            dist_counter: HashMap::new(),
        }
    }

    fn counter<T: ControllerManaged>(&self) -> &HashMap<object::Id<T>, Counter> {
        match T::NAMESPACE {
            Namespace::KvBlock => &self.kv_block_counter,
            Namespace::Emb => &self.emb_counter,
            Namespace::Dist => &self.dist_counter,
        }
    }

    fn counter_mut<T: ControllerManaged>(&mut self) -> &mut HashMap<object::Id<T>, Counter> {
        match T::NAMESPACE {
            Namespace::KvBlock => &mut self.kv_block_counter,
            Namespace::Emb => &mut self.emb_counter,
            Namespace::Dist => &mut self.dist_counter,
        }
    }

    pub fn init<T: ControllerManaged>(&mut self, id: object::Id<T>) {
        self.counter_mut::<T>().insert(id, Counter::new(0));
    }

    pub fn destroy<T: ControllerManaged>(&mut self, id: &object::Id<T>) {
        self.counter_mut::<T>().remove(&id);
    }

    pub fn inc<T: ControllerManaged>(&mut self, id: &object::Id<T>) {
        self.counter_mut::<T>().get_mut(&id).unwrap().inc();
    }

    pub fn dec<T: ControllerManaged>(&mut self, id: &object::Id<T>) -> bool {
        self.counter_mut::<T>().get_mut(&id).unwrap().dec() <= 0
    }

    pub fn get<T: ControllerManaged>(&self, id: &object::Id<T>) -> usize {
        self.counter::<T>().get(&id).unwrap().get() as usize
    }
}
