use crate::state::{
    BlockError, CausalLanguageModel, CausalTransformer, ImageEmbedder, KvBlock, ObjectAllocator,
    RemoteObjId, StreamId, TokenDist, TokenEmb,
};
use crate::utils::IdPool;
use prost::Message;
use std::collections::HashMap;
use std::mem;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use tokio::sync::oneshot::Sender;
use tokio::task::JoinHandle;
use zeromq::{DealerSocket, Socket, SocketRecv, SocketSend, ZmqError, ZmqMessage};

mod sdi {
    include!(concat!(env!("OUT_DIR"), "/sdi.rs"));
}

/// Intermediate representation of a command to be executed by the backend.
/// This must not be exposed to other modules.
#[derive(Debug)]
enum IrCommand {
    // Embs
    Allocate(sdi::AllocateItem),
    Deallocate(sdi::AllocateItem),
    CopyBlock(sdi::CopyBlockItem),
    MaskBlock(sdi::MaskBlockItem),
    FillBlock(sdi::FillBlockItem),
    EmbedImage(sdi::EmbedImageItem),
    EmbedText(sdi::EmbedTextItem),
    DecodeRequest(sdi::DecodeRequestItem),
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
    socket_tx: Arc<Mutex<Option<mpsc::Sender<Vec<sdi::Command>>>>>,
    handle: Arc<Mutex<Option<JoinHandle<()>>>>,
    // zmq handles
    //handle: Arc<JoinHandle<()>>,
}

impl Backend {
    pub fn new(block_size: u32, max_kv_blocks: u32, max_embs: u32) -> Self {
        Self {
            block_size,
            namespace: Arc::new(Mutex::new(ObjNamespace::new(max_kv_blocks, max_embs))),
            cmd_buffer: Arc::new(Mutex::new(Vec::new())),
            pending: Arc::new(Mutex::new(Pending::new(10.0, 1, 1))),
            staged: Arc::new(Mutex::new(Vec::new())),
            submitted: Arc::new(Mutex::new(vec![])),
            socket_tx: Arc::new(Mutex::new(None)),
            handle: Arc::new(Mutex::new(None)),
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

    pub fn schedule(&self, curr_timestamp: f64) -> Result<(), BlockError> {
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

        // Horizontal batching: group commands by stream and type.
        for (_stream_id, cmd_list) in commands_by_stream.iter_mut() {
            let mut prev_cmd = None;

            loop {
                if cmd_list.is_empty() {
                    break;
                }
                let cmd = cmd_list.pop().unwrap();
                let curr_cmd = mem::discriminant(&cmd);

                // Vertical batching: Same kind of consecutive commands are batched together.
                // if the current command is different from the previous one, stop batching.
                if let Some(prev_cmd) = prev_cmd {
                    if prev_cmd != curr_cmd {
                        break;
                    }
                }

                pending.push(cmd, curr_timestamp);
                prev_cmd = Some(curr_cmd);
            }
        }

        let batched_payloads = pending.batch_all(curr_timestamp);

        // add the commands to the staged queue
        let mut staged = self.staged.lock().map_err(|_| BlockError::LockError)?;

        // Add the batched commands to the staged queue.
        for payload in batched_payloads {
            staged.push(sdi::Command {
                correlation_id: 0,
                payload: Some(payload),
            });
        }

        // drop the lock on pending
        Ok(())
    }

    pub async fn commit(&self) -> Result<(), BlockError> {
        // Lock and take the staged commands.
        let mut staged = self.staged.lock().map_err(|_| BlockError::LockError)?;

        // Lock the sender.
        let tx_guard = self.socket_tx.lock().map_err(|_| BlockError::LockError)?;
        let tx = tx_guard.as_ref().ok_or(BlockError::LockError)?;

        // Send the staged commands, replacing them with an empty Vec.
        tx.send(mem::take(&mut *staged))
            .await
            .map_err(|_| BlockError::SendError)?;

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
        let handle = tokio::spawn(Self::socket_driver(socket, rx));

        self.handle = Arc::new(Mutex::new(Some(handle)));

        Ok(())
    }

    async fn socket_driver(
        mut socket: DealerSocket,
        mut rx: mpsc::Receiver<Vec<sdi::Command>>,
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
                            match sdi::Command::decode(payload.as_ref()) {
                                Ok(resp) => {
                                    println!("---> Received response: {:?}", resp);
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
struct ObjNamespace {
    kv_block_id_pool: IdPool<RemoteObjId>,
    emb_id_pool: IdPool<RemoteObjId>,
}
#[derive(Debug)]
struct Pending {
    allocate: BatchQueue<sdi::AllocateItem>,
    deallocate: BatchQueue<sdi::AllocateItem>,
    copy_block: BatchQueue<sdi::CopyBlockItem>,
    mask_block: BatchQueue<sdi::MaskBlockItem>,
    embed_text: BatchQueue<sdi::EmbedTextItem>,
    embed_image: BatchQueue<sdi::EmbedImageItem>,

    // these cmds are only be fired when it contains "enough" commands to be batched.
    fill_block: BatchQueue<sdi::FillBlockItem>,
    decode_req: BatchQueue<sdi::DecodeRequestItem>,
}

/// "K-or-T" Strategy
// 	For instance: If queue size reaches K, launch immediately; otherwise launch after T ms if K isn’t reached.
// 	This ensures that the GPU does not stay idle for too long (bounded by T) and that short bursts of arrivals form a large enough batch to get good utilization (bounded by K).
#[derive(Debug)]
struct BatchQueue<T> {
    items: Vec<(T, f64)>,

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

    fn take(&mut self) -> Vec<T> {
        let drain_count = self.items.len().min(self.max_size);
        self.items
            .drain(..drain_count)
            .map(|(item, _)| item)
            .collect()
    }

    fn push(&mut self, item: T, curr_timestamp: f64) {
        self.items.push((item, curr_timestamp));
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

    fn batch(&mut self, curr_timestamp: f64) -> Option<Vec<T>> {
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
            decode_req: BatchQueue::k_or_t(max_wait_time, min_size, Some(max_size)),
        }
    }

    fn push(&mut self, cmd: IrCommand, curr_timestamp: f64) {
        match cmd {
            IrCommand::Allocate(item) => {
                self.allocate.push(item, curr_timestamp);
            }
            IrCommand::Deallocate(item) => {
                self.deallocate.push(item, curr_timestamp);
            }
            IrCommand::CopyBlock(item) => {
                self.copy_block.push(item, curr_timestamp);
            }
            IrCommand::MaskBlock(item) => {
                self.mask_block.push(item, curr_timestamp);
            }
            IrCommand::FillBlock(item) => {
                self.fill_block.push(item, curr_timestamp);
            }
            IrCommand::EmbedImage(item) => {
                self.embed_image.push(item, curr_timestamp);
            }
            IrCommand::EmbedText(item) => {
                self.embed_text.push(item, curr_timestamp);
            }
            IrCommand::DecodeRequest(item) => {
                self.decode_req.push(item, curr_timestamp);
            }
        }
    }

    fn batch_all(&mut self, curr_timestamp: f64) -> Vec<sdi::command::Payload> {
        let mut cmds = Vec::new();

        if let Some(items) = self.allocate.batch(curr_timestamp) {
            cmds.push(sdi::command::Payload::Allocate(sdi::Allocate { items }));
        }

        if let Some(items) = self.deallocate.batch(curr_timestamp) {
            cmds.push(sdi::command::Payload::Deallocate(sdi::Deallocate { items }));
        }

        if let Some(items) = self.copy_block.batch(curr_timestamp) {
            cmds.push(sdi::command::Payload::CopyBlock(sdi::CopyBlock { items }));
        }

        if let Some(items) = self.mask_block.batch(curr_timestamp) {
            cmds.push(sdi::command::Payload::MaskBlock(sdi::MaskBlock { items }));
        }

        if let Some(items) = self.embed_text.batch(curr_timestamp) {
            cmds.push(sdi::command::Payload::EmbedText(sdi::EmbedText { items }));
        }

        if let Some(items) = self.embed_image.batch(curr_timestamp) {
            cmds.push(sdi::command::Payload::EmbedImage(sdi::EmbedImage { items }));
        }

        if let Some(items) = self.fill_block.batch(curr_timestamp) {
            cmds.push(sdi::command::Payload::FillBlock(sdi::FillBlock { items }));
        }

        if let Some(items) = self.decode_req.batch(curr_timestamp) {
            cmds.push(sdi::command::Payload::DecodeRequest(sdi::DecodeRequest {
                items,
            }));
        }

        cmds
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

impl ObjectAllocator<TokenEmb> for Backend {
    type RawRepr = Vec<f32>;

    fn alloc(&self, stream_id: StreamId) -> Result<RemoteObjId, BlockError> {
        let new_obj_id = self.acquire_id(1)?;

        let cmd = IrCommand::Allocate(sdi::AllocateItem {
            kind: sdi::ObjectKind::Emb.into(),
            object_id_offset: new_obj_id,
            count: 1,
        });
        self.enqueue_cmd(stream_id, cmd)?;

        Ok(new_obj_id)
    }

    fn dealloc(&self, stream_id: StreamId, obj_id: RemoteObjId) -> Result<(), BlockError> {
        // Release the object id back to the pool.
        self.release_id(1, obj_id)?;

        let cmd = IrCommand::Deallocate(sdi::AllocateItem {
            kind: sdi::ObjectKind::Emb.into(),
            object_id_offset: obj_id,
            count: 1,
        });
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

        let cmd = IrCommand::Allocate(sdi::AllocateItem {
            kind: sdi::ObjectKind::KvBlock.into(),
            object_id_offset: new_obj_id,
            count: 1,
        });
        self.enqueue_cmd(stream_id, cmd)?;

        Ok(new_obj_id)
    }

    fn dealloc(&self, stream_id: StreamId, obj_id: RemoteObjId) -> Result<(), BlockError> {
        // Release the object id back to the pool.
        self.release_id(0, obj_id)?;

        let cmd = IrCommand::Deallocate(sdi::AllocateItem {
            kind: sdi::ObjectKind::KvBlock.into(),
            object_id_offset: obj_id,
            count: 1,
        });
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

        let cmd = IrCommand::FillBlock(sdi::FillBlockItem {
            block_id: ptr,
            context_block_ids: ctx_ptrs,
            input_embedding_ids: input_embs,
            output_embedding_ids: output_embs.unwrap_or_default(),
        });

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
        let cmd = IrCommand::CopyBlock(sdi::CopyBlockItem {
            source_block_id: src_ptr,
            destination_block_id: dst_ptr,
            source_start: src_offset,
            destination_start: dst_offset,
            length: size,
        });

        self.enqueue_cmd(stream_id, cmd)?;
        Ok(())
    }

    fn mask_tokens(
        &self,
        stream_id: StreamId,
        ptr: RemoteObjId,
        mask: &[bool],
    ) -> Result<(), BlockError> {
        let cmd = IrCommand::MaskBlock(sdi::MaskBlockItem {
            block_id: ptr,
            mask: mask.to_vec(),
        });
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
