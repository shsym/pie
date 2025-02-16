use crate::backend::{Addr, CausalLanguageModel, CausalTransformer, ImageEmbedder, InstanceId};
use crate::object;
use crate::utils::Stream;
use prost::Message;
use std::collections::HashMap;
use std::mem;
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use zeromq::{DealerSocket, Socket, SocketRecv, SocketSend, ZmqError, ZmqMessage};

mod sdi {
    include!(concat!(env!("OUT_DIR"), "/sdi.rs"));
}

use crate::object::{KvBlock, Object, ObjectError, TokenDist, TokenEmb};
use thiserror::Error;
use tokio::sync::oneshot::Sender;
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
    Allocate(sdi::Allocate),
    Deallocate(sdi::Allocate),
    CopyBlock(sdi::CopyBlock),
    MaskBlock(sdi::MaskBlock),
    FillBlock(sdi::FillBlock),
    EmbedImage(sdi::EmbedImage),
    EmbedText(sdi::EmbedText),
    DecodeTokenDistribution(sdi::DecodeTokenDistribution),
    SampleTopKRequest(sdi::SampleTopKRequest),
    GetTokenDistributionRequest(sdi::GetTokenDistributionRequest),
}

// Hidden
#[derive(Debug)]
enum IrEvent {
    SampleTopK(sdi::SampleTopKResponse),
    GetTokenDistribution(sdi::GetTokenDistributionResponse),
}

#[derive(Debug)]
struct EventDispatcher {
    // maps correlation_id to a list of senders.
    table: HashMap<u32, Vec<EventHandle>>,
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

impl EventDispatcher {
    fn new() -> Self {
        Self {
            table: HashMap::new(),
        }
    }

    fn register(&mut self, correlation_id: u32, sender: Vec<EventHandle>) {
        self.table.insert(correlation_id, sender);
    }

    fn dispatch(&mut self, correlation_id: u32, event: Vec<IrEvent>) {
        // zip senders and evnt
        let senders = self.table.get_mut(&correlation_id).unwrap();
        assert_eq!(senders.len(), event.len());

        for (sender, evt) in senders.drain(..).zip(event.into_iter()) {
            match sender {
                EventHandle::None => {}
                EventHandle::SampleTopK(s) => {
                    if let IrEvent::SampleTopK(mut resp) = evt {
                        let _ = s.send(mem::take(&mut resp.token_ids));
                    } else {
                        eprintln!("Unexpected event type");
                    }
                }
                EventHandle::GetTokenDistribution(s) => {
                    if let IrEvent::GetTokenDistribution(mut resp) = evt {
                        let _ = s.send(mem::take(&mut resp.distribution));
                    } else {
                        eprintln!("Unexpected event type");
                    }
                }
            }
        }
    }
}
// Define actual cmd, interpretable by the backend.

// Allocate(Type, entities:[id])
// Deallocate(Type, entities:[id])
// Fill (List[id,
// LongFill using bitsets.. implement this later.

#[derive(Debug)]
pub struct Controller {
    block_size: u32,

    cmd_buffer: Arc<Mutex<Vec<(Stream, IrCommand, EventHandle)>>>,

    pending: Arc<Mutex<Pending>>,
    staged: Arc<Mutex<Vec<sdi::Command>>>,
    submitted: Arc<Mutex<Vec<sdi::Command>>>,
    // queue, cmd_buffer, scheduled, submitted
    socket_tx: Arc<Mutex<Option<mpsc::Sender<Vec<sdi::Command>>>>>,
    handle: Arc<Mutex<Option<JoinHandle<()>>>>,
    // zmq handles

    // event dispatcher
    event_dispatcher: Arc<Mutex<EventDispatcher>>,

    // object allocations
    id_pool: object::IdPool,

    kv_blocks: HashMap<object::Id<KvBlock>, KvBlock>,
    token_embs: HashMap<object::Id<TokenEmb>, TokenEmb>,
    virtual_addr_maps: HashMap<InstanceId, object::IdMap>,
}

impl Controller {
    pub fn new(block_size: u32, max_kv_blocks: u32, max_embs: u32) -> Self {
        Self {
            block_size,
            id_pool: object::IdPool::new(max_kv_blocks, max_embs),
            cmd_buffer: Arc::new(Mutex::new(Vec::new())),
            pending: Arc::new(Mutex::new(Pending::new(10.0, 1, 1))),
            staged: Arc::new(Mutex::new(Vec::new())),
            submitted: Arc::new(Mutex::new(vec![])),
            socket_tx: Arc::new(Mutex::new(None)),
            handle: Arc::new(Mutex::new(None)),
            event_dispatcher: Arc::new(Mutex::new(EventDispatcher {
                table: HashMap::new(),
            })),
        }
    }

    pub fn enqueue_cmd(&self, stream: Stream, cmd: IrCommand) -> Result<(), ControllerError> {
        let mut inner = self
            .cmd_buffer
            .lock()
            .map_err(|_| ControllerError::LockError)?;
        inner.push((stream, cmd, EventHandle::None));
        Ok(())
    }

    pub fn enqueue_cmd_with_event(
        &self,
        stream: Stream,
        cmd: IrCommand,
        evt: EventHandle,
    ) -> Result<(), ControllerError> {
        let mut inner = self
            .cmd_buffer
            .lock()
            .map_err(|_| ControllerError::LockError)?;

        inner.push((stream, cmd, evt));
        Ok(())
    }

    pub fn schedule(&self, curr_timestamp: f64) -> Result<(), ControllerError> {
        let mut pending = self
            .pending
            .lock()
            .map_err(|_| ControllerError::LockError)?;

        // first move all the commands from cmd_buffer to pending (buffer items are removed)
        let mut commands_by_stream = {
            let mut cmd_buffer = self
                .cmd_buffer
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

    pub async fn commit(&self) -> Result<(), ControllerError> {
        // Lock and take the staged commands.
        let mut staged = self.staged.lock().map_err(|_| ControllerError::LockError)?;

        // Lock the sender.
        let tx_guard = self
            .socket_tx
            .lock()
            .map_err(|_| ControllerError::LockError)?;
        let tx = tx_guard.as_ref().ok_or(ControllerError::LockError)?;

        // Send the staged commands, replacing them with an empty Vec.
        tx.send(mem::take(&mut *staged))
            .await
            .map_err(|_| ControllerError::SendError)?;

        Ok(())
    }

    pub async fn bind(&mut self, endpoint: &str) -> Result<(), ZmqError> {
        // bind the zmq socket
        let mut socket = DealerSocket::new();
        socket.connect(endpoint).await?;
        println!("Connected to server at {endpoint}");

        let (tx, rx) = mpsc::channel::<Vec<sdi::Command>>(100);

        self.socket_tx = Arc::new(Mutex::new(Some(tx)));

        // 3) Spawn the single I/O driver task that handles all read/write from the socket
        let handle = tokio::spawn(Self::socket_driver(
            socket,
            rx,
            self.event_dispatcher.clone(),
        ));

        self.handle = Arc::new(Mutex::new(Some(handle)));

        Ok(())
    }

    async fn socket_driver(
        mut socket: DealerSocket,
        mut rx: mpsc::Receiver<Vec<sdi::Command>>,
        evt_dispatch: Arc<Mutex<EventDispatcher>>,
    ) {
        loop {
            tokio::select! {
                // A) Incoming requests from the channel => send to server
                maybe_req = rx.recv() => {
                    match maybe_req {
                        Some(cmds) => {
                            for cmd in cmds {
                                let bytes = cmd.encode_to_vec();
                                if let Err(e) = socket.send(ZmqMessage::from(bytes)).await {
                                    eprintln!("Socket send failed: {:?}", e);
                                    // You might choose to break or keep trying
                                }
                            }
                        },
                        None => {
                            // channel closed => no more requests
                            println!("Request channel closed, shutting down driver.");
                            break;
                        }
                    }
                },

                // B) Incoming responses from the server
                result = socket.recv() => {
                    match result {
                        Ok(msg) => {
                            // Dealer/Router typically has 2 frames: [identity, payload]
                            let payload = msg.get(0).unwrap();
                            match sdi::Event::decode(payload.as_ref()) {
                                Ok(evt) => {

                                    // send this evt somewhere elese.

                                    let correlation_id = evt.correlation_id;

                                    let ir_events = match evt.payload.unwrap() {
                                        sdi::event::Payload::SampleTopK(batch) => {
                                            batch.items.into_iter().map(|item| IrEvent::SampleTopK(item)
                                            ).collect()
                                        },
                                        sdi::event::Payload::GetTokenDistribution(batch) => {
                                             batch.items.into_iter().map(|item| IrEvent::GetTokenDistribution(item)
                                            ).collect()
                                        }
                                    };

                                    let mut dispatcher = evt_dispatch.lock().unwrap();
                                    dispatcher.dispatch(correlation_id, ir_events);

                                }
                                Err(err) => {
                                    eprintln!("Failed to parse Response from server: {:?}", err);
                                }
                            }
                        },
                        Err(e) => {
                            eprintln!("Socket receive error: {:?}", e);
                            // Possibly break or keep going...
                            break;
                        }
                    }
                }
            }
        }
    }
}

// more sophisticated forms include: MultiNodeBackend, etc.

#[derive(Debug)]
struct Pending {
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

impl Pending {
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

impl object::Allocator<TokenEmb> for Controller
{
    type RawRepr = Vec<f32>;

    fn objects(&self) -> &HashMap<object::Id<TokenEmb>, TokenEmb> {
        &self.token_embs
    }

    fn objects_mut(&mut self) -> &mut HashMap<object::Id<TokenEmb>, TokenEmb> {
        &mut self.token_embs
    }

    fn alloc(&mut self, stream: Stream, object: TokenEmb) -> Result<object::Id<TokenEmb>, ObjectError> {
        let id = self.id_pool.acquire()?;

        self.token_embs.insert(id, object);

        let cmd = IrCommand::Allocate(sdi::Allocate {
            kind: sdi::ObjectKind::Emb.into(),
            object_id_offset: id.into(),
            count: 1,
        });
        self.enqueue_cmd(stream, cmd)
            .map_err(|_| ObjectError::BackendError("Failed to enqueue command".into()))?;

        Ok(id)
    }

    fn dealloc(&mut self, stream: Stream, id: object::Id<TokenEmb>) -> Result<(), ObjectError> {
        // Release the object id back to the pool.
        self.id_pool.release(id)?;

        let cmd = IrCommand::Deallocate(sdi::Allocate {
            kind: sdi::ObjectKind::Emb.into(),
            object_id_offset: id.into(),
            count: 1,
        });
        self.enqueue_cmd(stream, cmd)
            .map_err(|_| ObjectError::BackendError("Failed to enqueue command".into()))?;

        Ok(())
    }

    fn raw_repr(
        &self,
        stream: Stream,
        id: object::Id<TokenEmb>,
        sender: Sender<Self::RawRepr>,
    ) -> Result<(), ObjectError> {
        todo!()
    }

    fn available(&self) -> usize {
        self.id_pool.available::<TokenEmb>()
    }
}

impl object::Allocator<KvBlock> for Controller {
    // The raw representation of a kv block is not really useful in any way. So we just use usize.
    type RawRepr = object::Id<KvBlock>;

    fn objects(&self) -> &HashMap<object::Id<KvBlock>, KvBlock> {
        &self.kv_blocks
    }

    fn objects_mut(&mut self) -> &mut HashMap<object::Id<KvBlock>, KvBlock> {
        &mut self.kv_blocks
    }

    fn alloc(&mut self, stream: Stream, object: KvBlock) -> Result<object::Id<KvBlock>, ObjectError> {
        let id = self.id_pool.acquire()?;

        self.kv_blocks.insert(id, object);

        let cmd = IrCommand::Allocate(sdi::Allocate {
            kind: sdi::ObjectKind::KvBlock.into(),
            object_id_offset: id.into(),
            count: 1,
        });
        self.enqueue_cmd(stream, cmd)
            .map_err(|_| ObjectError::BackendError("Failed to enqueue command".into()))?;

        Ok(id)
    }

    fn dealloc(&mut self, stream: Stream, id: object::Id<KvBlock>) -> Result<(), ObjectError> {
        // Release the object id back to the pool.
        self.id_pool.release(id)?;

        let cmd = IrCommand::Deallocate(sdi::Allocate {
            kind: sdi::ObjectKind::KvBlock.into(),
            object_id_offset: id.into(),
            count: 1,
        });
        self.enqueue_cmd(stream, cmd)
            .map_err(|_| ObjectError::BackendError("Failed to enqueue command".into()))?;

        Ok(())
    }

    fn raw_repr(
        &self,
        stream: Stream,
        id: object::Id<KvBlock>,
        sender: Sender<Self::RawRepr>,
    ) -> Result<(), ObjectError> {
        todo!()
    }

    fn available(&self) -> usize {
        self.id_pool.available::<KvBlock>()
    }
}

impl object::Allocator<TokenDist> for Controller {
    type RawRepr = Vec<f32>;

    fn objects(&self) -> &HashMap<object::Id<TokenDist>, TokenDist> {
        todo!()
    }

    fn objects_mut(&mut self) -> &mut HashMap<object::Id<TokenDist>, TokenDist> {
        todo!()
    }

    fn alloc(&mut self, stream: Stream, object: TokenDist) -> Result<object::Id<TokenDist>, ObjectError> {
        todo!()
    }

    fn dealloc(&mut self, stream: Stream, id: object::Id<TokenDist>) -> Result<(), ObjectError> {
        todo!()
    }

    fn raw_repr(
        &self,
        stream: Stream,
        id: object::Id<TokenDist>,
        sender: Sender<Self::RawRepr>,
    ) -> Result<(), ObjectError> {
        todo!()
    }

    fn available(&self) -> usize {
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
            output_embedding_ids: output_embs.into_iter().map(|id| id.map(|id| id.into())).collect(),
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
