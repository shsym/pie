use crate::lm::{
    CausalLanguageModel, CausalTransformer, ImageEmbedder, KvBlock, TokenDist, TokenEmb,
};
use crate::utils::{Counter, Stream};
use crate::{backend, object, tokenizer, utils};
use anyhow::{anyhow, bail, ensure};
use rand::Rng;
use std::collections::HashMap;
use std::fmt::Debug;
use std::mem;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, mpsc, oneshot};

use crate::backend::BackendError;
use crate::driver::DriverError;
use crate::object::{Allocator, Fetcher, Id as ObjectId, IdMapper, IdRepr, ObjectError, VspaceId};
use crate::tokenizer::BytePairEncoder;
use thiserror::Error;
use tokio::sync::mpsc::Sender;
use tokio::task::JoinHandle;

const TOKENIZER_MODEL: &str = "../test-tokenizer/tokenizer.model";

mod l4m {
    include!(concat!(env!("OUT_DIR"), "/l4m.rs"));
}

// blanket implementation for all compatible backends
pub trait ExecuteCommand: backend::ExecuteCommand<l4m::Request, l4m::Response> {}
impl<T> ExecuteCommand for T where T: backend::ExecuteCommand<l4m::Request, l4m::Response> {}

/// Intermediate representation of a command to be executed by the backend.
/// This must not be exposed to other modules.
#[derive(Debug)]
enum Command {
    // Embs
    Allocate(l4m::Allocate),
    Deallocate(l4m::Allocate),
    CopyBlock(l4m::CopyBlock),
    MaskBlock(l4m::MaskBlock),
    FillBlock(l4m::FillBlock),
    //EmbedImage(l4m::EmbedImage),
    EmbedText(l4m::EmbedText),
    DecodeTokenDistribution(l4m::DecodeTokenDistribution),
    SampleTopKRequest(l4m::SampleTopKRequest),
    GetTokenDistributionRequest(l4m::GetTokenDistributionRequest),
    //Ping(l4m::PingRequest),
}

// Hidden
#[derive(Debug)]
enum Event {
    SampleTopK(l4m::SampleTopKResponse),
    GetTokenDistribution(l4m::GetTokenDistributionResponse),
    //Ping(l4m::PingResponse),
    GetInfo(l4m::GetInfoResponse),
}

#[derive(Debug)]
pub enum EventHandle {
    None,
    SampleTopK(oneshot::Sender<Vec<u32>>),
    GetTokenDistribution(oneshot::Sender<Vec<f32>>),
    //Ping(oneshot::Sender<String>),
    GetInfo(oneshot::Sender<Info>),
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
pub struct Info {
    pub version: String,
    pub model_name: String,
    pub block_size: u32,
    pub num_blocks: u32,
    pub num_embeddings: u32,
    pub num_distributions: u32,
}

// User-facing API
#[derive(Debug)]
pub struct Utils {
    pub tokenizer: BytePairEncoder,
    pub block_size: u32,
}

#[derive(Debug)]
pub struct Driver<B> {
    cmd_buffer: Vec<(Stream, Command, EventHandle)>,
    cmd_batcher: CommandBatcher,
    cmd_id_pool: utils::IdPool<u32>,
    backend: B,

    // object allocations
    obj_id_pool: IdPool,
    obj_ref_counter: RefCounter,
    obj_id_spaces: HashMap<VspaceId, IdMap>,

    event_dispatcher: Arc<Mutex<EventDispatcher>>,
    resp_handler: tokio::task::JoinHandle<()>,

    pub info: Info,
    pub utils: Arc<Utils>,
}

impl<B> Driver<B>
where
    B: ExecuteCommand,
{
    pub async fn new(b: B) -> Self {
        let (tx, rx) = tokio::sync::mpsc::channel(1000);

        let dispatcher = Arc::new(Mutex::new(EventDispatcher::new()));

        let resp_handler = tokio::spawn(Self::handle_responses(rx, dispatcher.clone()));
        b.report_to(tx).await;

        // retrieve the backend info
        let info = {
            let (info_tx, info_rx) = oneshot::channel();

            let req = l4m::Request {
                correlation_id: 0,
                command: Some(l4m::request::Command::GetInfo(l4m::GetInfoRequest {})),
            };

            dispatcher
                .lock()
                .await
                .register(req.correlation_id, vec![EventHandle::GetInfo(info_tx)]);
            b.exec(req).await.unwrap();

            info_rx.await.unwrap()
        };

        let utils = Utils {
            // TODO: load the tokenizer model based on the info.model_name
            tokenizer: tokenizer::llama3_tokenizer(TOKENIZER_MODEL).expect("Tokenizer load failed"),
            block_size: info.block_size,
        };

        println!(
            "The backend info: version={}, model_name={}, block_size={}, num_blocks={}, num_embeddings={}, num_distributions={}",
            info.version,
            info.model_name,
            info.block_size,
            info.num_blocks,
            info.num_embeddings,
            info.num_distributions
        );

        let driver = Self {
            cmd_buffer: Vec::new(),
            cmd_batcher: CommandBatcher::new(Duration::from_secs_f32(0.0), 1, 1),
            cmd_id_pool: utils::IdPool::new(u32::MAX),
            backend: b,
            obj_id_pool: IdPool::new(info.block_size, info.num_embeddings, info.num_distributions),
            obj_ref_counter: RefCounter::new(),
            obj_id_spaces: HashMap::new(),
            event_dispatcher: dispatcher,
            resp_handler,
            info,
            utils: Arc::new(utils),
        };

        driver
    }

    pub fn init_space(&mut self, space: VspaceId) -> Result<(), ObjectError> {
        if self.obj_id_spaces.contains_key(&space) {
            return Err(ObjectError::VSpaceAlreadyExists(space));
        }

        self.obj_id_spaces.insert(space, IdMap::new());
        Ok(())
    }

    pub fn destroy_space(&mut self, stream: Stream, space: &VspaceId) -> Result<(), ObjectError> {
        // first, un-assign all the objects in the space
        let kv_blocks: Vec<ObjectId<KvBlock>> = self.list(space)?;
        let embs: Vec<ObjectId<TokenEmb>> = self.list(space)?;
        let dists: Vec<ObjectId<TokenDist>> = self.list(space)?;

        self.unassign_all(stream, space, &kv_blocks)?;
        self.unassign_all(stream, space, &embs)?;
        self.unassign_all(stream, space, &dists)?;

        self.obj_id_spaces.remove(space).unwrap();
        Ok(())
    }

    fn enqueue_cmd(&mut self, stream: Stream, cmd: Command) -> Result<(), DriverError> {
        self.cmd_buffer.push((stream, cmd, EventHandle::None));
        Ok(())
    }

    fn enqueue_cmd_with_event(
        &mut self,
        stream: Stream,
        cmd: Command,
        evt: EventHandle,
    ) -> Result<(), DriverError> {
        self.cmd_buffer.push((stream, cmd, evt));
        Ok(())
    }

    pub async fn submit(&mut self, curr_timestamp: Instant) -> Result<(), DriverError> {
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
            self.exec_in_backend(cmd, senders).await?;
        }

        // drop the lock on pending
        Ok(())
    }

    async fn exec_in_backend(
        &mut self,
        cmd: l4m::request::Command,
        senders: Vec<EventHandle>,
    ) -> Result<(), DriverError> {
        // TODO: fix error type
        let correlation_id = self
            .cmd_id_pool
            .acquire()
            .map_err(|e| DriverError::LockError)?;

        // register events if there are any
        if senders.iter().any(|s| s.is_some()) {
            self.event_dispatcher
                .lock()
                .await
                .register(correlation_id, senders);
        }

        let req = l4m::Request {
            correlation_id,
            command: Some(cmd),
        };

        self.backend
            .exec(req)
            .await
            .map_err(|_| DriverError::SendError)?;

        Ok(())
    }

    async fn handle_responses(
        mut rx: tokio::sync::mpsc::Receiver<l4m::Response>,
        dispatcher: Arc<Mutex<EventDispatcher>>,
    ) {
        while let Some(resp) = rx.recv().await {
            let correlation_id = resp.correlation_id;
            let payload = resp.command.unwrap();

            let mut dispatcher = dispatcher.lock().await;

            let ir_events = match payload {
                l4m::response::Command::SampleTopK(batch) => batch
                    .items
                    .into_iter()
                    .map(|item| Event::SampleTopK(item))
                    .collect(),
                l4m::response::Command::GetTokenDistribution(batch) => batch
                    .items
                    .into_iter()
                    .map(|item| Event::GetTokenDistribution(item))
                    .collect(),
                // l4m::response::Command::Ping(item) => {
                //     vec![Event::Ping(item)]
                // }
                l4m::response::Command::GetInfo(item) => {
                    vec![Event::GetInfo(item)]
                }
            };

            dispatcher.dispatch(correlation_id, ir_events).unwrap();
        }
    }

    // pub async fn ping(
    //     &mut self,
    //     message: String,
    //     handler: oneshot::Sender<String>,
    // ) -> Result<(), DriverError> {
    //     let cmd = l4m::request::Command::Ping(l4m::PingRequest { message });
    //
    //     self.exec_in_backend(cmd, vec![EventHandle::Ping(handler)])
    //         .await?;
    //
    //     Ok(())
    // }

    pub async fn get_info(&mut self) -> Result<Info, DriverError> {
        let cmd = l4m::request::Command::GetInfo(l4m::GetInfoRequest {});

        let (tx, rx) = oneshot::channel();

        self.exec_in_backend(cmd, vec![EventHandle::GetInfo(tx)])
            .await?;

        let resp = rx.await.map_err(|_| DriverError::SendError)?;

        Ok(resp)
    }
}

impl<B> Drop for Driver<B> {
    fn drop(&mut self) {
        // drop the event dispatcher
        let _ = self.resp_handler.abort();
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

    fn dispatch(&mut self, correlation_id: u32, events: Vec<Event>) -> anyhow::Result<()> {
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
                    if let Event::SampleTopK(mut resp) = event {
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
                    if let Event::GetTokenDistribution(mut resp) = event {
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
                // EventHandle::Ping(s) => {
                //     if let Event::Ping(mut resp) = event {
                //         s.send(mem::take(&mut resp.message))
                //             .map_err(|e| anyhow!("Failed to send Ping event: {:?}", e))?;
                //     } else {
                //         bail!(
                //             "Mismatched event type: expected Ping for correlation_id: {}",
                //             correlation_id
                //         );
                //     }
                // }
                EventHandle::GetInfo(s) => {
                    if let Event::GetInfo(mut resp) = event {
                        let info = Info {
                            version: resp.version,
                            model_name: resp.model_name,
                            block_size: resp.block_size,
                            num_blocks: resp.num_available_blocks,
                            num_embeddings: resp.num_available_embeddings,
                            num_distributions: resp.num_available_distributions,
                        };

                        s.send(info)
                            .map_err(|e| anyhow!("Failed to send GetInfo event: {:?}", e))?;
                    } else {
                        bail!(
                            "Mismatched event type: expected GetInfo for correlation_id: {}",
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
    allocate: BatchQueue<l4m::Allocate>,
    deallocate: BatchQueue<l4m::Allocate>,
    copy_block: BatchQueue<l4m::CopyBlock>,
    mask_block: BatchQueue<l4m::MaskBlock>,
    embed_text: BatchQueue<l4m::EmbedText>,
    //embed_image: BatchQueue<l4m::EmbedImage>,

    // these cmds are only be fired when it contains "enough" commands to be batched.
    fill_block: BatchQueue<l4m::FillBlock>,
    decode_token_distribution: BatchQueue<l4m::DecodeTokenDistribution>,
    sample_top_k: BatchQueue<l4m::SampleTopKRequest>,
}

/// "K-or-T" Strategy
// 	For instance: If queue size reaches K, launch immediately; otherwise launch after T ms if K isn’t reached.
// 	This ensures that the GPU does not stay idle for too long (bounded by T) and that short bursts of arrivals form a large enough batch to get good utilization (bounded by K).
#[derive(Debug)]
struct BatchQueue<T> {
    // cmd, timestamp, response_sender
    items: Vec<(T, Instant, EventHandle)>,

    max_wait_time: Duration,
    min_size: usize,
    max_size: usize,
}

impl<T> BatchQueue<T> {
    fn eager() -> Self {
        Self {
            items: Vec::new(),
            max_wait_time: Duration::from_secs_f32(0.0),
            min_size: 1,
            max_size: usize::MAX,
        }
    }

    fn k_only(min_size: usize, max_size: Option<usize>) -> Self {
        Self {
            items: Vec::new(),
            max_wait_time: Duration::MAX,
            min_size,
            max_size: max_size.unwrap_or(min_size),
        }
    }

    fn t_only(max_wait_time: Duration) -> Self {
        Self {
            items: Vec::new(),
            max_wait_time,
            min_size: 1,
            max_size: usize::MAX,
        }
    }

    fn k_or_t(max_wait_time: Duration, min_size: usize, max_size: Option<usize>) -> Self {
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

    fn push(&mut self, item: T, curr_timestamp: Instant, evt: EventHandle) {
        self.items.push((item, curr_timestamp, evt));
    }

    fn is_ready(&self, curr_timestamp: Instant) -> bool {
        let num_items = self.items.len();

        if num_items > 0 {
            let longest_wait_time = curr_timestamp - self.items[0].1;
            if num_items >= self.min_size || longest_wait_time >= self.max_wait_time {
                return true;
            }
        }
        false
    }

    fn batch(&mut self, curr_timestamp: Instant) -> Option<(Vec<T>, Vec<EventHandle>)> {
        if self.is_ready(curr_timestamp) {
            Some(self.take())
        } else {
            None
        }
    }
}

impl CommandBatcher {
    fn new(max_wait_time: Duration, min_size: usize, max_size: usize) -> Self {
        Self {
            allocate: BatchQueue::eager(),
            deallocate: BatchQueue::eager(),
            copy_block: BatchQueue::k_or_t(max_wait_time, min_size, Some(max_size)),
            mask_block: BatchQueue::eager(),
            embed_text: BatchQueue::eager(),
            //embed_image: BatchQueue::k_or_t(max_wait_time, min_size, Some(max_size)),
            fill_block: BatchQueue::eager(),
            sample_top_k: BatchQueue::eager(),
            decode_token_distribution: BatchQueue::eager(),
        }
    }

    fn push(&mut self, cmd: Command, curr_timestamp: Instant, evt: EventHandle) {
        match cmd {
            Command::Allocate(item) => {
                self.allocate.push(item, curr_timestamp, evt);
            }
            Command::Deallocate(item) => {
                self.deallocate.push(item, curr_timestamp, evt);
            }
            Command::CopyBlock(item) => {
                self.copy_block.push(item, curr_timestamp, evt);
            }
            Command::MaskBlock(item) => {
                self.mask_block.push(item, curr_timestamp, evt);
            }
            Command::FillBlock(item) => {
                self.fill_block.push(item, curr_timestamp, evt);
            }
            // Command::EmbedImage(item) => {
            //     self.embed_image.push(item, curr_timestamp, evt);
            // }
            Command::EmbedText(item) => {
                self.embed_text.push(item, curr_timestamp, evt);
            }
            Command::SampleTopKRequest(item) => {
                self.sample_top_k.push(item, curr_timestamp, evt);
            }
            Command::DecodeTokenDistribution(item) => {
                self.decode_token_distribution
                    .push(item, curr_timestamp, evt);
            }
            Command::GetTokenDistributionRequest(_) => todo!(),
            _ => {
                eprintln!("Unsupported command type: {:?}", cmd);
            }
        }
    }

    fn batch_all(
        &mut self,
        curr_timestamp: Instant,
    ) -> Vec<(l4m::request::Command, Vec<EventHandle>)> {
        let mut cmds = Vec::new();

        if let Some((items, senders)) = self.allocate.batch(curr_timestamp) {
            cmds.push((
                l4m::request::Command::Allocate(l4m::BatchAllocate { items }),
                senders,
            ));
        }

        if let Some((items, senders)) = self.deallocate.batch(curr_timestamp) {
            cmds.push((
                l4m::request::Command::Deallocate(l4m::BatchDeallocate { items }),
                senders,
            ));
        }

        if let Some((items, senders)) = self.copy_block.batch(curr_timestamp) {
            cmds.push((
                l4m::request::Command::CopyBlock(l4m::BatchCopyBlock { items }),
                senders,
            ));
        }

        if let Some((items, senders)) = self.mask_block.batch(curr_timestamp) {
            cmds.push((
                l4m::request::Command::MaskBlock(l4m::BatchMaskBlock { items }),
                senders,
            ));
        }

        if let Some((items, senders)) = self.embed_text.batch(curr_timestamp) {
            cmds.push((
                l4m::request::Command::EmbedText(l4m::BatchEmbedText { items }),
                senders,
            ));
        }

        // if let Some((items, senders)) = self.embed_image.batch(curr_timestamp) {
        //     cmds.push((
        //         l4m::request::Command::EmbedImage(l4m::BatchEmbedImage { items }),
        //         senders,
        //     ));
        // }

        if let Some((items, senders)) = self.fill_block.batch(curr_timestamp) {
            cmds.push((
                l4m::request::Command::FillBlock(l4m::BatchFillBlock { items }),
                senders,
            ));
        }

        if let Some((items, senders)) = self.decode_token_distribution.batch(curr_timestamp) {
            cmds.push((
                l4m::request::Command::DecodeTokenDistribution(l4m::BatchDecodeTokenDistribution {
                    items,
                }),
                senders,
            ));
        }

        if let Some((items, senders)) = self.sample_top_k.batch(curr_timestamp) {
            cmds.push((
                l4m::request::Command::SampleTopKRequest(l4m::BatchSampleTopKRequest { items }),
                senders,
            ));
        }

        cmds
    }
}

pub trait Managed: Debug + Sized + Send + Sync {
    const NAMESPACE: Namespace;
    //fn get_namespace() -> Namespace;

    fn sdi_object_kind() -> i32 {
        match Self::NAMESPACE {
            Namespace::KvBlock => l4m::ObjectKind::KvBlock.into(),
            Namespace::Emb => l4m::ObjectKind::Emb.into(),
            Namespace::Dist => l4m::ObjectKind::Dist.into(),
        }
    }
}

impl Managed for KvBlock {
    const NAMESPACE: Namespace = Namespace::KvBlock;
}

impl Managed for TokenEmb {
    const NAMESPACE: Namespace = Namespace::Emb;
}

impl Managed for TokenDist {
    const NAMESPACE: Namespace = Namespace::Dist;
}

#[derive(Debug, Clone, Copy)]
pub enum Namespace {
    KvBlock = 0,
    Emb = 1,
    Dist = 2,
}

impl<B, T> Allocator<T> for Driver<B>
where
    T: Managed,
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
            let cmd = Command::Allocate(l4m::Allocate {
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
            let cmd = Command::Deallocate(l4m::Allocate {
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

impl<B, T> object::IdMapper<T> for Driver<B>
where
    T: Managed,
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

    fn unassign_all(
        &mut self,
        stream: Stream,
        space: &VspaceId,
        srcs: &[ObjectId<T>],
    ) -> Result<(), ObjectError> {
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
                self.dealloc(stream, tgt)?;
            }
        }

        Ok(())
    }
}

impl<B> CausalTransformer for Driver<B>
where
    B: backend::ExecuteCommand<l4m::Request, l4m::Response>,
{
    fn fill(
        &mut self,
        stream: Stream,
        space: &VspaceId,
        ptr: ObjectId<KvBlock>,
        ctx_ptrs: Vec<ObjectId<KvBlock>>,
        input_embs: Vec<ObjectId<TokenEmb>>,
        output_embs: Vec<ObjectId<TokenEmb>>,
    ) -> Result<(), DriverError> {
        // lookup the objects from the space
        let ptr = self.lookup(space, &ptr)?;
        let ctx_ptrs = self.lookup_all(space, &ctx_ptrs)?;
        let input_embs = self.lookup_all(space, &input_embs)?;
        let output_embs = self.lookup_all(space, &output_embs)?;

        let cmd = Command::FillBlock(l4m::FillBlock {
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
    ) -> Result<(), DriverError> {
        let src_ptr = self.lookup(space, &src_ptr)?;
        let dst_ptr = self.lookup(space, &dst_ptr)?;

        let cmd = Command::CopyBlock(l4m::CopyBlock {
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
    ) -> Result<(), DriverError> {
        let ptr = self.lookup(space, &ptr)?;

        let cmd = Command::MaskBlock(l4m::MaskBlock {
            block_id: ptr.into(),
            mask: mask.to_vec(),
        });
        self.enqueue_cmd(stream, cmd)
    }
}

impl<B> CausalLanguageModel for Driver<B>
where
    B: ExecuteCommand,
{
    fn embed_text(
        &mut self,
        stream: Stream,
        space: &VspaceId,
        addrs: Vec<ObjectId<TokenEmb>>,
        text_tokens: Vec<u32>,
        positions: Vec<u32>,
    ) -> Result<(), DriverError> {
        let addrs = self.lookup_all(space, &addrs)?;

        for (addr, (text_token, pos)) in addrs.iter().zip(text_tokens.iter().zip(positions.iter()))
        {
            let cmd = Command::EmbedText(l4m::EmbedText {
                embedding_id: addr.into(),
                token_id: *text_token,
                position_id: *pos,
            });
            self.enqueue_cmd(stream, cmd)?;
        }

        Ok(())
    }

    fn next_token_dist(
        &mut self,
        stream: Stream,
        space: &VspaceId,
        emb_ptr: Vec<ObjectId<TokenEmb>>,
        dist_ptr: Vec<ObjectId<TokenDist>>,
    ) -> Result<(), DriverError> {
        let emb_ptr = self.lookup_all(space, &emb_ptr)?;
        let dist_ptr = self.lookup_all(space, &dist_ptr)?;

        for (emb_ptr, dist_ptr) in emb_ptr.iter().zip(dist_ptr.iter()) {
            let cmd = Command::DecodeTokenDistribution(l4m::DecodeTokenDistribution {
                embedding_id: emb_ptr.into(),
                distribution_id: dist_ptr.into(),
            });
            self.enqueue_cmd(stream, cmd)?;
        }

        Ok(())
    }

    fn sample_top_k(
        &mut self,
        stream: Stream,
        space: &VspaceId,
        dist_ptr: &ObjectId<TokenDist>,
        k: u32,
        handle: oneshot::Sender<Vec<u32>>,
    ) -> Result<(), DriverError> {
        let dist_ptr = self.lookup(space, &dist_ptr)?;

        let cmd = Command::SampleTopKRequest(l4m::SampleTopKRequest {
            distribution_id: dist_ptr.into(),
            k,
        });
        // create a new event handle
        let handle = EventHandle::SampleTopK(handle);

        self.enqueue_cmd_with_event(stream, cmd, handle)?;
        Ok(())
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

impl<B> Fetcher<TokenDist> for Driver<B>
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

        let cmd = Command::GetTokenDistributionRequest(l4m::GetTokenDistributionRequest {
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

#[derive(Debug)]
pub struct IdPool {
    kv_block_id_pool: utils::IdPool<object::IdRepr>,
    emb_id_pool: utils::IdPool<object::IdRepr>,
    dist_id_pool: utils::IdPool<object::IdRepr>,
}
impl IdPool {
    pub fn new(max_kv_blocks: u32, max_embs: u32, max_dists: u32) -> Self {
        Self {
            kv_block_id_pool: utils::IdPool::new(max_kv_blocks),
            emb_id_pool: utils::IdPool::new(max_embs),
            dist_id_pool: utils::IdPool::new(max_dists),
        }
    }

    // Helper that returns a mutable reference to the appropriate pool.
    fn pool_mut<T: Managed>(&mut self) -> &mut utils::IdPool<object::IdRepr> {
        match T::NAMESPACE {
            Namespace::KvBlock => &mut self.kv_block_id_pool,
            Namespace::Emb => &mut self.emb_id_pool,
            Namespace::Dist => &mut self.dist_id_pool,
        }
    }

    // Helper that returns an immutable reference.
    fn pool<T: Managed>(&self) -> &utils::IdPool<object::IdRepr> {
        match T::NAMESPACE {
            Namespace::KvBlock => &self.kv_block_id_pool,
            Namespace::Emb => &self.emb_id_pool,
            Namespace::Dist => &self.dist_id_pool,
        }
    }

    pub fn acquire<T: Managed>(&mut self) -> Result<ObjectId<T>, ObjectError> {
        let id = self
            .pool_mut::<T>()
            .acquire()
            .map_err(|_| ObjectError::NoAvailableSpace)?;
        Ok(ObjectId::new(id))
    }

    pub fn acquire_many<T: Managed>(
        &mut self,
        count: usize,
    ) -> Result<Vec<ObjectId<T>>, ObjectError> {
        let ids = self
            .pool_mut::<T>()
            .acquire_many(count)
            .map_err(|_| ObjectError::NoAvailableSpace)?;
        Ok(ObjectId::map_from_repr(ids))
    }

    pub fn release<T: Managed>(&mut self, id: &ObjectId<T>) -> Result<(), ObjectError> {
        self.pool_mut::<T>()
            .release(id.into())
            .map_err(|e| ObjectError::AddressPoolError(e.to_string()))
    }

    pub fn release_many<T: Managed>(&mut self, ids: &[ObjectId<T>]) -> Result<(), ObjectError> {
        let raw_ids = ObjectId::ref_as_repr(ids);
        self.pool_mut::<T>()
            .release_many(raw_ids)
            .map_err(|e| ObjectError::AddressPoolError(e.to_string()))
    }

    pub fn available<T: Managed>(&self) -> usize {
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
    fn mapping_mut<T: Managed>(&mut self) -> &mut HashMap<object::IdRepr, object::IdRepr> {
        match T::NAMESPACE {
            Namespace::KvBlock => &mut self.kv_block_id_map,
            Namespace::Emb => &mut self.emb_id_map,
            Namespace::Dist => &mut self.dist_id_map,
        }
    }

    // Helper method to get an immutable reference to the appropriate map.
    fn mapping<T: Managed>(&self) -> &HashMap<object::IdRepr, object::IdRepr> {
        match T::NAMESPACE {
            Namespace::KvBlock => &self.kv_block_id_map,
            Namespace::Emb => &self.emb_id_map,
            Namespace::Dist => &self.dist_id_map,
        }
    }

    fn exists<T: Managed>(&self, vid: &ObjectId<T>) -> bool {
        self.mapping::<T>().contains_key(&vid.into())
    }

    fn lookup<T: Managed>(&self, vid: &ObjectId<T>) -> Result<ObjectId<T>, ObjectError> {
        self.mapping::<T>()
            .get(&vid.into())
            .map(|id| ObjectId::<T>::new(*id))
            .ok_or(ObjectError::ObjectNotFound)
    }

    fn lookup_all<T: Managed>(
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

    fn assign<T: Managed>(&mut self, vid: ObjectId<T>, id: ObjectId<T>) {
        let (src, dst) = (vid.into(), id.into());
        self.mapping_mut::<T>().insert(src, dst);
    }

    fn unassign<T: Managed>(&mut self, vid: &ObjectId<T>) -> Result<ObjectId<T>, ObjectError> {
        let vid = vid.into();
        let id = self
            .mapping_mut::<T>()
            .remove(&vid)
            .ok_or(ObjectError::ObjectNotFound)?;
        Ok(ObjectId::<T>::new(id))
    }

    fn list<T: Managed>(&self) -> Result<Vec<ObjectId<T>>, ObjectError> {
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

    fn counter<T: Managed>(&self) -> &HashMap<IdRepr, Counter> {
        match T::NAMESPACE {
            Namespace::KvBlock => &self.kv_block_counter,
            Namespace::Emb => &self.emb_counter,
            Namespace::Dist => &self.dist_counter,
        }
    }

    fn counter_mut<T: Managed>(&mut self) -> &mut HashMap<IdRepr, Counter> {
        match T::NAMESPACE {
            Namespace::KvBlock => &mut self.kv_block_counter,
            Namespace::Emb => &mut self.emb_counter,
            Namespace::Dist => &mut self.dist_counter,
        }
    }

    pub fn init<T: Managed>(&mut self, id: ObjectId<T>) {
        self.counter_mut::<T>().insert(id.into(), Counter::new(0));
    }

    pub fn destroy<T: Managed>(&mut self, id: &ObjectId<T>) {
        self.counter_mut::<T>().remove(&id.into());
    }

    pub fn inc<T: Managed>(&mut self, id: &ObjectId<T>) {
        self.counter_mut::<T>().get_mut(&id.into()).unwrap().inc();
    }

    pub fn dec<T: Managed>(&mut self, id: &ObjectId<T>) -> bool {
        self.counter_mut::<T>().get_mut(&id.into()).unwrap().dec() <= 0
    }

    pub fn get<T: Managed>(&self, id: &ObjectId<T>) -> usize {
        self.counter::<T>().get(&id.into()).unwrap().get() as usize
    }
}

#[derive(Clone)]
pub struct Simulator {}

impl backend::Simulate<l4m::Request, l4m::Response> for Simulator {
    fn simulate(&mut self, req: l4m::Request) -> Option<l4m::Response> {
        let resp_payload = match req.command.unwrap() {
            l4m::request::Command::SampleTopKRequest(batch) => {
                let items = batch.items.into_iter().map(|item| {
                    let mut rng = rand::rng();
                    let token_ids: Vec<_> =
                        (0..item.k).map(|_| rng.random_range(0..1000)).collect();

                    l4m::SampleTopKResponse { token_ids }
                });

                Some(l4m::response::Command::SampleTopK(
                    l4m::BatchSampleTopKResponse {
                        items: items.collect(),
                    },
                ))
            }
            l4m::request::Command::GetTokenDistribution(batch) => {
                let items = batch.items.into_iter().map(|item| {
                    let mut rng = rand::rng();
                    let distribution: Vec<_> =
                        (0..1000).map(|_| rng.random_range(0.0..1.0)).collect();
                    l4m::GetTokenDistributionResponse { distribution }
                });
                Some(l4m::response::Command::GetTokenDistribution(
                    l4m::BatchGetTokenDistributionResponse {
                        items: items.collect(),
                    },
                ))
            }

            l4m::request::Command::GetInfo(_) => {
                Some(l4m::response::Command::GetInfo(l4m::GetInfoResponse {
                    version: "0.1.0".to_string(),
                    model_name: "DummyModel".to_string(),
                    block_size: 128,
                    num_available_blocks: 1000000,
                    num_available_embeddings: 1000000,
                    num_available_distributions: 100000,
                }))
            }
            _ => None,
        };

        if let Some(payload) = resp_payload {
            Some(l4m::Response {
                correlation_id: req.correlation_id,
                command: Some(payload),
            })
        } else {
            None
        }
    }
}
