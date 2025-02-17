use crate::lm::{
    CausalLanguageModel, CausalTransformer, ImageEmbedder, KvBlock, TokenDist, TokenEmb,
};
use crate::utils::{Counter, Stream};
use crate::{backend, instance, object, utils};
use anyhow::{anyhow, bail, ensure, Context};
use std::collections::HashMap;
use std::fmt::Debug;
use std::mem;
use std::sync::Arc;
use tokio::sync::{oneshot, Mutex};

use crate::backend::ExecuteCommand;
use crate::instance::Id as InstanceId;
use crate::object::{Allocator, Fetcher, Id as ObjectId, IdMapper, IdRepr, ObjectError, VspaceId};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ControllerError {
    #[error("Mutex lock failed")]
    LockError,

    #[error("Socket transmitter not available")]
    MissingSocket,

    #[error("Channel send error")]
    SendError,

    #[error("Object error: {0}")]
    ObjectError(#[from] ObjectError),

    #[error("Event dispatcher error: {0}")]
    EventDispatcherError(#[from] anyhow::Error),
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
pub enum EventHandle {
    None,
    SampleTopK(oneshot::Sender<Vec<u32>>),
    GetTokenDistribution(oneshot::Sender<Vec<f32>>),
}

impl EventHandle {
    pub fn is_some(&self) -> bool {
        match self {
            EventHandle::None => false,
            _ => true,
        }
    }
}

#[derive(Debug)]
pub struct Controller<B> {
    cmd_buffer: Vec<(Stream, IrCommand, EventHandle)>,
    cmd_batcher: CommandBatcher,
    cmd_id_pool: utils::IdPool<u32>,
    backend: B,

    // object allocations
    obj_id_pool: IdPool,
    obj_ref_counter: RefCounter,
    obj_id_spaces: HashMap<VspaceId, IdMap>,

    event_dispatcher: Arc<Mutex<EventDispatcher>>,
    resp_handler: tokio::task::JoinHandle<()>,
}

impl<B> Controller<B>
where
    B: ExecuteCommand,
{
    pub async fn new(backend: B) -> Self {
        let (tx, rx) = tokio::sync::mpsc::channel(1000);

        let dispatcher = Arc::new(Mutex::new(EventDispatcher::new()));

        let resp_handler = tokio::spawn(Self::handle_responses(rx, dispatcher.clone()));
        backend.report_to(tx);

        Self {
            cmd_buffer: Vec::new(),
            cmd_batcher: CommandBatcher::new(0.0, 1, 1),
            cmd_id_pool: utils::IdPool::new(u32::MAX),
            backend,
            obj_id_pool: IdPool::new(10000, 10000),
            obj_ref_counter: RefCounter::new(),
            obj_id_spaces: HashMap::new(),
            event_dispatcher: dispatcher,
            resp_handler,
        }
    }

    pub fn init_space(&mut self, space: VspaceId) -> Result<(), ObjectError> {
        if self.obj_id_spaces.contains_key(&space) {
            return Err(ObjectError::VSpaceAlreadyExists(space));
        }

        self.obj_id_spaces.insert(space, IdMap::new());
        Ok(())
    }

    pub fn destroy_space(&mut self, space: &VspaceId) -> Result<(), ObjectError> {
        // first, un-assign all the objects in the space
        let kv_blocks: Vec<ObjectId<KvBlock>> = self.list(space)?;
        let embs: Vec<ObjectId<TokenEmb>> = self.list(space)?;
        let dists: Vec<ObjectId<TokenDist>> = self.list(space)?;

        self.unassign_all(space, &kv_blocks)?;
        self.unassign_all(space, &embs)?;
        self.unassign_all(space, &dists)?;

        self.obj_id_spaces.remove(space).unwrap();
        Ok(())
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

    pub async fn submit(&mut self, curr_timestamp: f64) -> Result<(), ControllerError> {
        // first move all the commands from cmd_buffer to pending (buffer items are removed)
        let mut commands_by_stream = HashMap::new();

        for (stream_id, command, sender) in self.cmd_buffer.drain(..) {
            commands_by_stream
                .entry(stream_id)
                .or_insert_with(Vec::new)
                .push((command, sender));
        }

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

                self.cmd_batcher.push(cmd, curr_timestamp, sender);
                prev_cmd = Some(curr_cmd);
            }
        }

        let batched_payloads = self.cmd_batcher.batch_all(curr_timestamp);

        for (cmd, senders) in batched_payloads {
            // TODO: fix error type
            let correlation_id = self
                .cmd_id_pool
                .acquire()
                .map_err(|e| ControllerError::LockError)?;

            // register events if there are any
            if senders.iter().any(|s| s.is_some()) {
                self.event_dispatcher
                    .lock()
                    .await
                    .register(correlation_id, senders);
            }

            let req = backend::sdi::Request {
                correlation_id,
                command: Some(cmd),
            };

            self.backend
                .exec(req)
                .await
                .map_err(|_| ControllerError::SendError)?;
        }

        // drop the lock on pending
        Ok(())
    }

    async fn handle_responses(
        mut rx: tokio::sync::mpsc::Receiver<backend::sdi::Response>,
        dispatcher: Arc<Mutex<EventDispatcher>>,
    ) {
        loop {
            let resp = rx.recv().await.unwrap();

            let correlation_id = resp.correlation_id;
            let payload = resp.payload.unwrap();

            let mut dispatcher = dispatcher.lock().await;

            let ir_events = match payload {
                backend::sdi::response::Payload::SampleTopK(batch) => batch
                    .items
                    .into_iter()
                    .map(|item| IrEvent::SampleTopK(item))
                    .collect(),
                backend::sdi::response::Payload::GetTokenDistribution(batch) => batch
                    .items
                    .into_iter()
                    .map(|item| IrEvent::GetTokenDistribution(item))
                    .collect(),
            };

            dispatcher.dispatch(correlation_id, ir_events).unwrap();
        }
    }
}

#[derive(Debug)]
struct EventDispatcher {
    // maps correlation_id to a list of senders.
    table: HashMap<u32, Vec<EventHandle>>,
}

impl EventDispatcher {
    fn new() -> Self {
        Self {
            table: HashMap::new(),
        }
    }

    fn register(&mut self, correlation_id: u32, sender: Vec<EventHandle>) {
        self.table.insert(correlation_id, sender);
    }

    fn dispatch(&mut self, correlation_id: u32, events: Vec<IrEvent>) -> anyhow::Result<()> {
        // Remove the handlers associated with the given correlation ID.
        let senders = self.table.remove(&correlation_id).ok_or_else(|| {
            anyhow!(
                "No event handlers found for correlation_id: {}",
                correlation_id
            )
        })?;

        // Ensure the number of senders matches the number of events.
        ensure!(
            senders.len() == events.len(),
            "Length mismatch: {} senders vs {} events",
            senders.len(),
            events.len()
        );

        // Iterate over each (sender, event) pair.
        for (sender, event) in senders.into_iter().zip(events.into_iter()) {
            match sender {
                EventHandle::None => { /* No action needed */ }
                EventHandle::SampleTopK(s) => {
                    if let IrEvent::SampleTopK(mut resp) = event {
                        s.send(mem::take(&mut resp.token_ids))
                            .map_err(|e| anyhow!("Failed to send SampleTopK event: {:?}", e))?;
                    } else {
                        bail!(
                            "Mismatched event type: expected SampleTopK for correlation_id: {}",
                            correlation_id
                        );
                    }
                }
                EventHandle::GetTokenDistribution(s) => {
                    if let IrEvent::GetTokenDistribution(mut resp) = event {
                        s.send(mem::take(&mut resp.distribution)).map_err(|e| {
                            anyhow!("Failed to send GetTokenDistribution event: {:?}", e)
                        })?;
                    } else {
                        bail!(
                        "Mismatched event type: expected GetTokenDistribution for correlation_id: {}",
                        correlation_id
                    );
                    }
                }
            }
        }

        Ok(())
    }
}

// more sophisticated forms include: MultiNodeBackend, etc.

#[derive(Debug)]
struct CommandBatcher {
    allocate: BatchQueue<backend::sdi::Allocate>,
    deallocate: BatchQueue<backend::sdi::Allocate>,
    copy_block: BatchQueue<backend::sdi::CopyBlock>,
    mask_block: BatchQueue<backend::sdi::MaskBlock>,
    embed_text: BatchQueue<backend::sdi::EmbedText>,
    embed_image: BatchQueue<backend::sdi::EmbedImage>,

    // these cmds are only be fired when it contains "enough" commands to be batched.
    fill_block: BatchQueue<backend::sdi::FillBlock>,
    decode_token_distribution: BatchQueue<backend::sdi::DecodeTokenDistribution>,
    sample_top_k: BatchQueue<backend::sdi::SampleTopKRequest>,
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

    fn batch_all(
        &mut self,
        curr_timestamp: f64,
    ) -> Vec<(backend::sdi::request::Command, Vec<EventHandle>)> {
        let mut cmds = Vec::new();

        if let Some((items, senders)) = self.allocate.batch(curr_timestamp) {
            cmds.push((
                backend::sdi::request::Command::Allocate(backend::sdi::BatchAllocate { items }),
                senders,
            ));
        }

        if let Some((items, senders)) = self.deallocate.batch(curr_timestamp) {
            cmds.push((
                backend::sdi::request::Command::Deallocate(backend::sdi::BatchDeallocate { items }),
                senders,
            ));
        }

        if let Some((items, senders)) = self.copy_block.batch(curr_timestamp) {
            cmds.push((
                backend::sdi::request::Command::CopyBlock(backend::sdi::BatchCopyBlock { items }),
                senders,
            ));
        }

        if let Some((items, senders)) = self.mask_block.batch(curr_timestamp) {
            cmds.push((
                backend::sdi::request::Command::MaskBlock(backend::sdi::BatchMaskBlock { items }),
                senders,
            ));
        }

        if let Some((items, senders)) = self.embed_text.batch(curr_timestamp) {
            cmds.push((
                backend::sdi::request::Command::EmbedText(backend::sdi::BatchEmbedText { items }),
                senders,
            ));
        }

        if let Some((items, senders)) = self.embed_image.batch(curr_timestamp) {
            cmds.push((
                backend::sdi::request::Command::EmbedImage(backend::sdi::BatchEmbedImage { items }),
                senders,
            ));
        }

        if let Some((items, senders)) = self.fill_block.batch(curr_timestamp) {
            cmds.push((
                backend::sdi::request::Command::FillBlock(backend::sdi::BatchFillBlock { items }),
                senders,
            ));
        }

        if let Some((items, senders)) = self.sample_top_k.batch(curr_timestamp) {
            cmds.push((
                backend::sdi::request::Command::SampleTopKRequest(
                    backend::sdi::BatchSampleTopKRequest { items },
                ),
                senders,
            ));
        }

        cmds
    }
}

pub trait ControllerManaged: Debug + Sized + Send + Sync {
    const NAMESPACE: Namespace;
    //fn get_namespace() -> Namespace;

    fn sdi_object_kind() -> i32 {
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

impl<B, T> Allocator<T> for Controller<B>
where
    T: ControllerManaged,
    B: ExecuteCommand,
{
    fn alloc_all(&mut self, stream: Stream, count: usize) -> Result<Vec<ObjectId<T>>, ObjectError> {
        let ids = self.obj_id_pool.acquire_many::<T>(count)?;

        // init the ref counter
        for id in &ids {
            self.obj_ref_counter.init(*id); // set the ref count to 0
        }

        // Request the backend to allocate the objects.
        let kind = T::sdi_object_kind();
        let cons_id = ObjectId::group_consecutive_ids(&ids);

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

    fn dealloc_all(&mut self, stream: Stream, ids: &[ObjectId<T>]) -> Result<(), ObjectError> {
        let mut rm_ids = Vec::new();
        for id in ids {
            // safety check
            let free = self.obj_ref_counter.get(id) <= 0;
            if free {
                rm_ids.push(*id);
                self.obj_id_pool.release(id)?;
                self.obj_ref_counter.destroy(id);
            }
        }

        let cons_id = ObjectId::group_consecutive_ids(&rm_ids);

        // destroy the ref counter
        // for id in rm_ids.iter() {
        //     self.ref_counter.destroy(id);
        // }

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
        self.obj_id_pool.available::<T>()
    }

    fn increment_ref_count(&mut self, id: &ObjectId<T>) -> Result<(), ObjectError> {
        Ok(self.obj_ref_counter.inc(id))
    }

    fn decrement_ref_count(&mut self, id: &ObjectId<T>) -> Result<bool, ObjectError> {
        Ok(self.obj_ref_counter.dec(id))
    }

    fn ref_count(&self, id: &ObjectId<T>) -> Result<usize, ObjectError> {
        Ok(self.obj_ref_counter.get(id))
    }
}

impl<B, T> object::IdMapper<T> for Controller<B>
where
    T: ControllerManaged,
    B: ExecuteCommand,
{
    fn exists(&self, space: &VspaceId, src: &ObjectId<T>) -> bool {
        self.obj_id_spaces
            .get(space)
            .map_or(false, |vspace| vspace.exists(src))
    }

    fn list(&self, space: &VspaceId) -> Result<Vec<ObjectId<T>>, ObjectError> {
        self.obj_id_spaces
            .get(space)
            .ok_or(ObjectError::VSpaceNotFound)?
            .list()
    }

    fn lookup_all(
        &self,
        space: &VspaceId,
        srcs: &[ObjectId<T>],
    ) -> Result<Vec<ObjectId<T>>, ObjectError> {
        self.obj_id_spaces
            .get(space)
            .ok_or(ObjectError::VSpaceNotFound)?
            .lookup_all(srcs)
    }

    fn assign_all(
        &mut self,
        space: &VspaceId,
        srcs: &[ObjectId<T>],
        tgts: &[ObjectId<T>],
    ) -> Result<(), ObjectError> {
        // increase the ref count for the target objects
        for tgt in tgts.iter() {
            self.increment_ref_count(tgt)?;
        }

        let vspace = self
            .obj_id_spaces
            .get_mut(space)
            .ok_or(ObjectError::VSpaceNotFound)?;

        for (src, tgt) in srcs.iter().zip(tgts.into_iter()) {
            vspace.assign(*src, *tgt);
        }

        Ok(())
    }

    fn unassign_all(&mut self, space: &VspaceId, srcs: &[ObjectId<T>]) -> Result<(), ObjectError> {
        let vspace = self
            .obj_id_spaces
            .get_mut(space)
            .ok_or(ObjectError::VSpaceNotFound)?;

        let mut tgts = Vec::with_capacity(srcs.len());
        for src in srcs.iter() {
            let tgt = vspace.unassign(src)?;
            tgts.push(tgt);
        }

        for tgt in tgts.iter() {
            let free = self.decrement_ref_count(tgt)?;
            if free {
                self.dealloc(Stream::default(), tgt)?;
            }
        }

        Ok(())
    }
}

impl<B> CausalTransformer for Controller<B>
where
    B: ExecuteCommand,
{
    fn fill(
        &mut self,
        stream: Stream,
        space: &VspaceId,
        ptr: ObjectId<KvBlock>,
        ctx_ptrs: Vec<ObjectId<KvBlock>>,
        input_embs: Vec<ObjectId<TokenEmb>>,
        output_embs: Vec<ObjectId<TokenEmb>>,
    ) -> Result<(), ControllerError> {
        // lookup the objects from the space
        let ptr = self.lookup(space, &ptr)?;
        let ctx_ptrs = self.lookup_all(space, &ctx_ptrs)?;
        let input_embs = self.lookup_all(space, &input_embs)?;
        let output_embs = self.lookup_all(space, &output_embs)?;

        let cmd = IrCommand::FillBlock(backend::sdi::FillBlock {
            block_id: ptr.into(),
            context_block_ids: ObjectId::map_to_repr(ctx_ptrs),
            input_embedding_ids: ObjectId::map_to_repr(input_embs),
            output_embedding_ids: ObjectId::map_to_repr(output_embs),
        });

        self.enqueue_cmd(stream, cmd)
    }

    fn copy_tokens(
        &mut self,
        stream: Stream,
        space: &VspaceId,
        src_ptr: ObjectId<KvBlock>,
        dst_ptr: ObjectId<KvBlock>,
        src_offset: u32,
        dst_offset: u32,
        size: u32,
    ) -> Result<(), ControllerError> {
        let src_ptr = self.lookup(space, &src_ptr)?;
        let dst_ptr = self.lookup(space, &dst_ptr)?;

        let cmd = IrCommand::CopyBlock(backend::sdi::CopyBlock {
            source_block_id: src_ptr.into(),
            destination_block_id: dst_ptr.into(),
            source_start: src_offset,
            destination_start: dst_offset,
            length: size,
        });

        self.enqueue_cmd(stream, cmd)
    }

    fn mask_tokens(
        &mut self,
        stream: Stream,
        space: &VspaceId,
        ptr: ObjectId<KvBlock>,
        mask: &[bool],
    ) -> Result<(), ControllerError> {
        let ptr = self.lookup(space, &ptr)?;

        let cmd = IrCommand::MaskBlock(backend::sdi::MaskBlock {
            block_id: ptr.into(),
            mask: mask.to_vec(),
        });
        self.enqueue_cmd(stream, cmd)
    }
}

impl<B> CausalLanguageModel for Controller<B>
where
    B: ExecuteCommand,
{
    fn next_token_dist(
        &mut self,
        stream: Stream,
        space: &VspaceId,
        emb_ptr: ObjectId<TokenEmb>,
        dist_ptr: ObjectId<TokenDist>,
    ) -> Result<(), ControllerError> {
        let emb_ptr = self.lookup(space, &emb_ptr)?;
        let dist_ptr = self.lookup(space, &dist_ptr)?;

        let cmd = IrCommand::DecodeTokenDistribution(backend::sdi::DecodeTokenDistribution {
            embedding_id: emb_ptr.into(),
            distribution_id: dist_ptr.into(),
        });

        self.enqueue_cmd(stream, cmd)
    }

    fn sample_top_k(
        &mut self,
        stream: Stream,
        space: &VspaceId,
        dist_ptr: ObjectId<TokenDist>,
        k: u32,
    ) -> Result<oneshot::Receiver<Vec<u32>>, ControllerError> {
        let dist_ptr = self.lookup(space, &dist_ptr)?;

        let cmd = IrCommand::SampleTopKRequest(backend::sdi::SampleTopKRequest {
            distribution_id: dist_ptr.into(),
            k,
        });
        // create a new event handle

        let (tx, rx) = oneshot::channel::<Vec<u32>>();
        let handle = EventHandle::SampleTopK(tx);

        self.enqueue_cmd_with_event(stream, cmd, handle)?;
        Ok(rx)
    }

    // fn get_raw_dist(
    //     &self,
    //     stream_id: Stream,
    //     vspace_id: &VspaceId,
    //     dist_ptr: ObjectId<TokenDist>,
    // ) -> Result<oneshot::Receiver<Vec<f32>>, ControllerError> {
    //     let cmd =
    //         IrCommand::GetTokenDistributionRequest(backend::sdi::GetTokenDistributionRequest {
    //             distribution_id: dist_ptr,
    //         });
    //
    //     let (tx, rx) = oneshot::channel::<Vec<f32>>();
    //     let handle = EventHandle::GetTokenDistribution(tx);
    //
    //     self.enqueue_cmd_with_event(stream_id, cmd, handle)?;
    //     Ok(rx)
    // }
}

impl<B> Fetcher<TokenDist> for Controller<B>
where
    B: ExecuteCommand,
{
    type RawRepr = Vec<f32>;

    fn fetch(
        &mut self,
        stream: Stream,
        space: &VspaceId,
        ptr: &ObjectId<TokenDist>,
        sender: oneshot::Sender<Self::RawRepr>,
    ) -> Result<(), ObjectError> {
        let ptr = self.lookup(space, &ptr)?;

        let cmd =
            IrCommand::GetTokenDistributionRequest(backend::sdi::GetTokenDistributionRequest {
                distribution_id: ptr.into(),
            });

        let handle = EventHandle::GetTokenDistribution(sender);

        self.enqueue_cmd_with_event(stream, cmd, handle)
            .map_err(|e| {
                ObjectError::BackendError("Failed to fetch token distribution".to_string())
            })
    }
}

/// for multimodal LLMs

impl<B> ImageEmbedder for Controller<B>
where
    B: ExecuteCommand,
{
    fn embed_img(
        &mut self,
        stream: Stream,
        space: &VspaceId,
        addrs: Vec<ObjectId<TokenEmb>>,
        url: String,
    ) -> Result<(), ControllerError> {
        let addrs = self.lookup_all(space, &addrs)?;

        let cmd = IrCommand::EmbedImage(backend::sdi::EmbedImage {
            embedding_ids: ObjectId::map_to_repr(addrs),
            url,
        });

        self.enqueue_cmd(stream, cmd)
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

    pub fn acquire<T: ControllerManaged>(&mut self) -> Result<ObjectId<T>, ObjectError> {
        let id = self
            .pool_mut::<T>()
            .acquire()
            .map_err(|_| ObjectError::NoAvailableSpace)?;
        Ok(ObjectId::new(id))
    }

    pub fn acquire_many<T: ControllerManaged>(
        &mut self,
        count: usize,
    ) -> Result<Vec<ObjectId<T>>, ObjectError> {
        let ids = self
            .pool_mut::<T>()
            .acquire_many(count)
            .map_err(|_| ObjectError::NoAvailableSpace)?;
        Ok(ObjectId::map_from_repr(ids))
    }

    pub fn release<T: ControllerManaged>(&mut self, id: &ObjectId<T>) -> Result<(), ObjectError> {
        self.pool_mut::<T>()
            .release(id.into())
            .map_err(|e| ObjectError::AddressPoolError(e.to_string()))
    }

    pub fn release_many<T: ControllerManaged>(
        &mut self,
        ids: &[ObjectId<T>],
    ) -> Result<(), ObjectError> {
        let raw_ids = ObjectId::ref_as_repr(ids);
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
    kv_block_id_map: HashMap<object::IdRepr, object::IdRepr>,
    emb_id_map: HashMap<object::IdRepr, object::IdRepr>,
    dist_id_map: HashMap<object::IdRepr, object::IdRepr>,
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
    fn mapping_mut<T: ControllerManaged>(
        &mut self,
    ) -> &mut HashMap<object::IdRepr, object::IdRepr> {
        match T::NAMESPACE {
            Namespace::KvBlock => &mut self.kv_block_id_map,
            Namespace::Emb => &mut self.emb_id_map,
            Namespace::Dist => &mut self.dist_id_map,
        }
    }

    // Helper method to get an immutable reference to the appropriate map.
    fn mapping<T: ControllerManaged>(&self) -> &HashMap<object::IdRepr, object::IdRepr> {
        match T::NAMESPACE {
            Namespace::KvBlock => &self.kv_block_id_map,
            Namespace::Emb => &self.emb_id_map,
            Namespace::Dist => &self.dist_id_map,
        }
    }

    fn exists<T: ControllerManaged>(&self, vid: &ObjectId<T>) -> bool {
        self.mapping::<T>().contains_key(&vid.into())
    }

    fn lookup<T: ControllerManaged>(&self, vid: &ObjectId<T>) -> Result<ObjectId<T>, ObjectError> {
        self.mapping::<T>()
            .get(&vid.into())
            .map(|id| ObjectId::<T>::new(*id))
            .ok_or(ObjectError::ObjectNotFound)
    }

    fn lookup_all<T: ControllerManaged>(
        &self,
        vids: &[ObjectId<T>],
    ) -> Result<Vec<ObjectId<T>>, ObjectError> {
        let map = self.mapping::<T>();
        let mut ids = Vec::with_capacity(vids.len());

        for vid in vids {
            let id = map.get(&vid.into()).ok_or(ObjectError::ObjectNotFound)?;
            ids.push(ObjectId::new(*id));
        }
        Ok(ids)
    }

    fn assign<T: ControllerManaged>(&mut self, vid: ObjectId<T>, id: ObjectId<T>) {
        let (src, dst) = (vid.into(), id.into());
        self.mapping_mut::<T>().insert(src, dst);
    }

    fn unassign<T: ControllerManaged>(
        &mut self,
        vid: &ObjectId<T>,
    ) -> Result<ObjectId<T>, ObjectError> {
        let vid = vid.into();
        let id = self
            .mapping_mut::<T>()
            .remove(&vid)
            .ok_or(ObjectError::ObjectNotFound)?;
        Ok(ObjectId::<T>::new(id))
    }

    fn list<T: ControllerManaged>(&self) -> Result<Vec<ObjectId<T>>, ObjectError> {
        Ok(self
            .mapping::<T>()
            .keys()
            .map(|id| ObjectId::<T>::new(*id))
            .collect())
    }
}

#[derive(Debug)]
struct RefCounter {
    kv_block_counter: HashMap<IdRepr, Counter>,
    emb_counter: HashMap<IdRepr, Counter>,
    dist_counter: HashMap<IdRepr, Counter>,
}

impl RefCounter {
    pub fn new() -> Self {
        Self {
            kv_block_counter: HashMap::new(),
            emb_counter: HashMap::new(),
            dist_counter: HashMap::new(),
        }
    }

    fn counter<T: ControllerManaged>(&self) -> &HashMap<IdRepr, Counter> {
        match T::NAMESPACE {
            Namespace::KvBlock => &self.kv_block_counter,
            Namespace::Emb => &self.emb_counter,
            Namespace::Dist => &self.dist_counter,
        }
    }

    fn counter_mut<T: ControllerManaged>(&mut self) -> &mut HashMap<IdRepr, Counter> {
        match T::NAMESPACE {
            Namespace::KvBlock => &mut self.kv_block_counter,
            Namespace::Emb => &mut self.emb_counter,
            Namespace::Dist => &mut self.dist_counter,
        }
    }

    pub fn init<T: ControllerManaged>(&mut self, id: ObjectId<T>) {
        self.counter_mut::<T>().insert(id.into(), Counter::new(0));
    }

    pub fn destroy<T: ControllerManaged>(&mut self, id: &ObjectId<T>) {
        self.counter_mut::<T>().remove(&id.into());
    }

    pub fn inc<T: ControllerManaged>(&mut self, id: &ObjectId<T>) {
        self.counter_mut::<T>().get_mut(&id.into()).unwrap().inc();
    }

    pub fn dec<T: ControllerManaged>(&mut self, id: &ObjectId<T>) -> bool {
        self.counter_mut::<T>().get_mut(&id.into()).unwrap().dec() <= 0
    }

    pub fn get<T: ControllerManaged>(&self, id: &ObjectId<T>) -> usize {
        self.counter::<T>().get(&id.into()).unwrap().get() as usize
    }
}
