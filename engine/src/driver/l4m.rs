use crate::backend::Backend;
use crate::batching::{Batchable, Batcher, BatchingStrategy, KorTStrategy};
use crate::driver::{Driver, DriverError, DynCommand};
use crate::instance::Id as InstanceId;
use crate::object::{
    IdRepr, ObjectError, ObjectManager, ObjectType, VspaceId, group_consecutive_ids,
};
use crate::runtime::Reporter;
use crate::tokenizer::BytePairEncoder;
use crate::utils::{Counter, IdPool};
use crate::{backend, lm, object, tokenizer, utils};
use anyhow::anyhow;
use dashmap::DashMap;
use prost::Message;
use rand::Rng;
use std::any::TypeId;
use std::cmp::{Ordering, PartialEq};
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::mem;
use std::mem::Discriminant;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc::Receiver;
use tokio::sync::{mpsc, oneshot};
use tokio::task;

pub const PROTOCOLS: [&str; 2] = ["l4m", "l4m-vision"]; // for future backward compatibility
const PROTOCOL_BASE: usize = 0;
const PROTOCOL_VISION: usize = 1;

mod pb_bindings {
    include!(concat!(env!("OUT_DIR"), "/l4m.rs"));
}

mod pb_bindings_vision {
    include!(concat!(env!("OUT_DIR"), "/l4m.vision.rs"));
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

pub type LocalStreamId = u32;

#[derive(Debug, Clone, Copy)]
pub struct Stream(InstanceId, LocalStreamId, StreamPriority);

impl Stream {
    pub fn new(inst: InstanceId, stream_id: LocalStreamId) -> Self {
        Self(inst, stream_id, StreamPriority::Normal)
    }

    pub fn set_priority(&mut self, priority: StreamPriority) {
        self.2 = priority;
    }
}

// Equality and hashing ignore the priority field.
impl PartialEq for Stream {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.1 == other.1
    }
}

impl Eq for Stream {}

impl Hash for Stream {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
        self.1.hash(state);
    }
}

// Ordering only compares the StreamPriority.
impl PartialOrd for Stream {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.2.cmp(&other.2))
    }
}

impl Ord for Stream {
    fn cmp(&self, other: &Self) -> Ordering {
        self.2.cmp(&other.2)
    }
}

// Depending on your needs you may want High to be considered either greater or less than Low.
// Here we simply derive the order in the declared order (High < Normal < Low).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum StreamPriority {
    High,
    Normal,
    Low,
}

#[derive(Debug)]
pub enum Command {
    GetBlockSize {
        handle: oneshot::Sender<u32>,
    },

    GetTokenizer {
        handle: oneshot::Sender<Arc<BytePairEncoder>>,
    },

    GetAllExportedBlocks {
        handle: oneshot::Sender<Vec<(String, IdRepr)>>,
    },

    Allocate {
        stream_id: LocalStreamId,
        ty: ManagedTypes,
        ids: Vec<IdRepr>,
    },

    Deallocate {
        stream_id: LocalStreamId,
        ty: ManagedTypes,
        ids: Vec<IdRepr>,
    },

    FillBlock {
        stream_id: LocalStreamId,
        block: IdRepr,
        context: Vec<IdRepr>,
        inputs: Vec<IdRepr>,
        outputs: Vec<IdRepr>,
    },

    ExportBlocks {
        blocks: Vec<IdRepr>,
        resource_name: String,
    },

    ImportBlocks {
        blocks: Vec<IdRepr>,
        resource_name: String,
    },

    CopyBlock {
        stream_id: LocalStreamId,
        src_block: IdRepr,
        dst_block: IdRepr,
        src_token_offset: u32,
        dst_token_offset: u32,
        size: u32,
    },

    MaskBlock {
        stream_id: LocalStreamId,
        block: IdRepr,
        mask: Vec<bool>,
    },

    EmbedText {
        stream_id: LocalStreamId,
        embs: Vec<IdRepr>,
        text: Vec<u32>,
        positions: Vec<u32>,
    },

    DecodeTokenDist {
        stream_id: LocalStreamId,
        embs: Vec<IdRepr>,
        dists: Vec<IdRepr>,
    },

    SampleTopK {
        stream_id: LocalStreamId,
        dist: IdRepr,
        k: u32,
        handle: oneshot::Sender<(Vec<u32>, Vec<f32>)>,
    },

    Synchronize {
        stream_id: LocalStreamId,
        handle: oneshot::Sender<()>,
    },

    SetStreamPriority {
        stream_id: LocalStreamId,
        priority: StreamPriority,
    },

    //// ------ Vision specific commands ------ ////
    EmbedImage {
        stream_id: LocalStreamId,
        embs: Vec<IdRepr>,
        image_blob: Vec<u8>,
    },
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
enum BatchGroup {
    Allocate,
    Deallocate,
    FillBlock,
    CopyBlock,
    MaskBlock,
    EmbedText,
    DecodeTokenDist,
    SampleTopK,
    Synchronize,
    EmbedImage,
}

impl Batchable<BatchGroup> for Command {
    fn strategy(&self) -> Box<dyn BatchingStrategy> {
        match self {
            Command::Allocate { .. } => KorTStrategy::eager().into_box(),
            Command::Deallocate { .. } => KorTStrategy::eager().into_box(),
            Command::FillBlock { .. } => KorTStrategy::eager().into_box(),
            Command::CopyBlock { .. } => KorTStrategy::eager().into_box(),
            Command::MaskBlock { .. } => KorTStrategy::eager().into_box(),
            Command::EmbedText { .. } => KorTStrategy::eager().into_box(),
            Command::DecodeTokenDist { .. } => KorTStrategy::eager().into_box(),
            Command::SampleTopK { .. } => KorTStrategy::eager().into_box(),
            Command::Synchronize { .. } => KorTStrategy::immediate().into_box(),
            Command::EmbedImage { .. } => KorTStrategy::eager().into_box(),
            _ => unreachable!(),
        }
    }

    fn group(&self) -> BatchGroup {
        match self {
            Command::Allocate { .. } => BatchGroup::Allocate,
            Command::Deallocate { .. } => BatchGroup::Deallocate,
            Command::FillBlock { .. } => BatchGroup::FillBlock,
            Command::CopyBlock { .. } => BatchGroup::CopyBlock,
            Command::MaskBlock { .. } => BatchGroup::MaskBlock,
            Command::EmbedText { .. } => BatchGroup::EmbedText,
            Command::DecodeTokenDist { .. } => BatchGroup::DecodeTokenDist,
            Command::SampleTopK { .. } => BatchGroup::SampleTopK,
            Command::Synchronize { .. } => BatchGroup::Synchronize,
            Command::EmbedImage { .. } => BatchGroup::EmbedImage,
            _ => unreachable!(),
        }
    }
}

#[derive(Debug)]
pub enum Event {
    SampleTopK(oneshot::Sender<(Vec<u32>, Vec<f32>)>),

    GetInfo(oneshot::Sender<Info>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ManagedTypes {
    KvBlock,
    TokenEmb,
    TokenDist,
}

impl ObjectType for ManagedTypes {
    fn is_sharable(&self) -> bool {
        match self {
            ManagedTypes::KvBlock => true,
            ManagedTypes::TokenEmb => false,
            ManagedTypes::TokenDist => false,
        }
    }

    fn allow_remapping(&self) -> bool {
        match self {
            ManagedTypes::KvBlock => false,
            ManagedTypes::TokenEmb => true,
            ManagedTypes::TokenDist => true,
        }
    }
}

#[derive(Debug)]
pub struct L4m<B> {
    backend: B,
    reporter: Reporter,
    protocol_ids: Vec<u8>,

    cmd_id_pool: IdPool<u32>,
    cmd_batcher: Batcher<Command, Stream, BatchGroup>,

    event_table: Arc<DashMap<u32, Vec<Event>>>,
    event_loop_handle: task::JoinHandle<()>,

    subscriptions: HashMap<String, Vec<InstanceId>>,
    exported_blocks: HashMap<String, ExportedBlocks>,

    // Rc for shared ownership across multiple "extension drivers"
    objects: ObjectManager<InstanceId, ManagedTypes>,

    stream_priorities: HashMap<Stream, StreamPriority>,

    info: Info,
    tokenizer: Arc<BytePairEncoder>,
}

impl<B> Driver for L4m<B>
where
    B: Backend,
{
    type Command = Command;

    fn create(&mut self, inst: InstanceId) {}

    fn destroy(&mut self, inst: InstanceId) {
        let mut cmds = Vec::new();

        for ty in [
            ManagedTypes::KvBlock,
            ManagedTypes::TokenEmb,
            ManagedTypes::TokenDist,
        ] {
            cmds.push(Command::Deallocate {
                stream_id: 0,
                ty,
                ids: self.objects.all_names(ty, inst).unwrap(),
            })
        }

        for cmd in cmds {
            self.dispatch(inst, cmd);
        }

        // Remove all exported blocks
        self.exported_blocks.retain(|_, v| v.owner != inst);
    }

    async fn dispatch(&mut self, inst: InstanceId, cmd: Self::Command) {
        match self.translate_cmd(inst, cmd) {
            Ok(cmd) => match cmd {
                // Should be sent to backend
                Some((cmd, mut stream)) => {
                    // adjust stream priority
                    if let Some(priority) = self.stream_priorities.get(&stream) {
                        stream.set_priority(*priority)
                    }

                    self.cmd_batcher.push(stream, cmd, Instant::now());

                    for (_, cmd_batch) in self.cmd_batcher.batch(Instant::now()) {
                        self.commit_backend(cmd_batch).await;
                    }
                }

                // No need to send to backend
                None => {}
            },
            Err(e) => self.reporter.error(inst, e.to_string()),
        }
    }

    fn reporter(&self) -> Option<&Reporter> {
        Some(&self.reporter)
    }
}

impl<B> L4m<B>
where
    B: Backend,
{
    pub async fn new(backend: B, reporter: Reporter) -> Result<Self, DriverError> {
        let protocol_ids = PROTOCOLS
            .iter()
            .map(|protoc| {
                backend.get_protocol_idx(protoc).map_err(|e| {
                    DriverError::Other(format!("Failed to get protocol index: {}", e.to_string()))
                })
            })
            .collect::<Result<Vec<u8>, DriverError>>()?;

        let (tx, rx) = mpsc::channel(1000);
        backend.listen(0, tx);

        let event_table = Arc::new(DashMap::new());
        let event_loop_handle = tokio::spawn(Self::event_loop(rx, event_table.clone()));

        let info = {
            let (info_tx, info_rx) = oneshot::channel();

            event_table.insert(0, vec![Event::GetInfo(info_tx)]);

            backend
                .send(
                    protocol_ids[PROTOCOL_BASE],
                    pb_bindings::Request {
                        correlation_id: 0,
                        command: Some(pb_bindings::request::Command::GetInfo(
                            pb_bindings::GetInfoRequest {},
                        )),
                    }
                    .encode_to_vec(),
                )
                .await;
            info_rx.await.unwrap()
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

        // TODO: load the tokenizer model based on the info.model_name
        let tokenizer = tokenizer::llama3_tokenizer("../test-tokenizer/tokenizer.model")
            .expect("Tokenizer load failed");

        let mut objects = ObjectManager::new();
        objects.set_capacity(ManagedTypes::KvBlock, info.num_blocks as usize)?;
        objects.set_capacity(ManagedTypes::TokenEmb, info.num_embeddings as usize)?;
        objects.set_capacity(ManagedTypes::TokenDist, info.num_distributions as usize)?;

        let driver = Self {
            backend,
            reporter,
            protocol_ids,
            cmd_id_pool: IdPool::new(u32::MAX),
            cmd_batcher: Batcher::new(),
            event_table,
            event_loop_handle,
            subscriptions: HashMap::new(),
            exported_blocks: HashMap::new(),
            objects,
            stream_priorities: HashMap::new(),
            info,
            tokenizer: Arc::new(tokenizer),
        };

        Ok(driver)
    }

    fn translate_cmd(
        &mut self,
        inst: InstanceId,
        cmd: Command,
    ) -> Result<Option<(Command, Stream)>, DriverError> {
        //let mut objects = &self.objects;

        let resolved_cmd = match cmd {
            Command::GetBlockSize { handle } => {
                handle
                    .send(self.info.block_size)
                    .map_err(|_| DriverError::SendError("GetBlockSize failed.".to_string()))?;
                None
            }

            Command::GetTokenizer { handle } => {
                handle
                    .send(self.tokenizer.clone())
                    .map_err(|_| DriverError::SendError("GetTokenizer failed.".to_string()))?;
                None
            }

            Command::GetAllExportedBlocks { handle } => {
                let catalogue = self
                    .exported_blocks
                    .iter()
                    .map(|(k, v)| (k.clone(), v.addrs.len() as u32))
                    .collect();
                handle.send(catalogue).map_err(|_| {
                    DriverError::SendError("GetAllExportedBlocks failed.".to_string())
                })?;
                None
            }

            Command::Allocate { stream_id, ty, ids } => {
                let ids = self.objects.create_many(ty, inst, ids)?;

                Some((
                    Command::Allocate { stream_id, ty, ids },
                    Stream::new(inst, stream_id),
                ))
            }
            Command::Deallocate { stream_id, ty, ids } => {
                let ids = self.objects.destroy_many(ty, inst, &ids)?;

                if ids.is_empty() {
                    return Ok(None);
                }

                Some((
                    Command::Deallocate { stream_id, ty, ids },
                    Stream::new(inst, stream_id),
                ))
            }
            Command::FillBlock {
                stream_id,
                mut block,
                mut context,
                mut inputs,
                mut outputs,
            } => {
                self.objects
                    .translate(ManagedTypes::KvBlock, inst, &mut block)?;
                self.objects
                    .translate_many(ManagedTypes::KvBlock, inst, &mut context)?;
                self.objects
                    .translate_many(ManagedTypes::TokenEmb, inst, &mut inputs)?;
                self.objects
                    .translate_many(ManagedTypes::TokenEmb, inst, &mut outputs)?;

                Some((
                    Command::FillBlock {
                        stream_id,
                        block,
                        context,
                        inputs,
                        outputs,
                    },
                    Stream::new(inst, stream_id),
                ))
            }
            Command::ExportBlocks {
                mut blocks,
                resource_name,
            } => {
                self.objects
                    .translate_many(ManagedTypes::KvBlock, inst, &mut blocks)?;

                self.exported_blocks
                    .insert(resource_name, ExportedBlocks::new(inst, blocks));

                None
            }
            Command::ImportBlocks {
                blocks,
                resource_name,
            } => {
                let exported =
                    self.exported_blocks
                        .get(&resource_name)
                        .ok_or(DriverError::Other(format!(
                            "Resource not found {}",
                            resource_name
                        )))?;

                self.objects.create_ref_many(
                    ManagedTypes::KvBlock,
                    inst,
                    blocks,
                    &exported.addrs,
                )?;

                None
            }

            Command::CopyBlock {
                stream_id,
                mut src_block,
                mut dst_block,
                src_token_offset,
                dst_token_offset,
                size,
            } => {
                self.objects
                    .translate(ManagedTypes::KvBlock, inst, &mut src_block)?;
                self.objects
                    .translate(ManagedTypes::KvBlock, inst, &mut dst_block)?;

                Some((
                    Command::CopyBlock {
                        stream_id,
                        src_block,
                        dst_block,
                        src_token_offset,
                        dst_token_offset,
                        size,
                    },
                    Stream::new(inst, stream_id),
                ))
            }
            Command::MaskBlock {
                stream_id,
                mut block,
                mask,
            } => {
                self.objects
                    .translate(ManagedTypes::KvBlock, inst, &mut block)?;

                Some((
                    Command::MaskBlock {
                        stream_id,
                        block,
                        mask,
                    },
                    Stream::new(inst, stream_id),
                ))
            }
            Command::EmbedText {
                stream_id,
                mut embs,
                text,
                positions,
            } => {
                self.objects
                    .translate_many(ManagedTypes::TokenEmb, inst, &mut embs)?;

                Some((
                    Command::EmbedText {
                        stream_id,
                        embs,
                        text,
                        positions,
                    },
                    Stream::new(inst, stream_id),
                ))
            }
            Command::DecodeTokenDist {
                stream_id,
                mut embs,
                mut dists,
            } => {
                self.objects
                    .translate_many(ManagedTypes::TokenEmb, inst, &mut embs)?;
                self.objects
                    .translate_many(ManagedTypes::TokenDist, inst, &mut dists)?;

                Some((
                    Command::DecodeTokenDist {
                        stream_id,
                        embs,
                        dists,
                    },
                    Stream::new(inst, stream_id),
                ))
            }
            Command::SampleTopK {
                stream_id,
                mut dist,
                k,
                handle,
            } => {
                self.objects
                    .translate(ManagedTypes::TokenDist, inst, &mut dist)?;

                Some((
                    Command::SampleTopK {
                        stream_id,
                        dist,
                        k,
                        handle,
                    },
                    Stream::new(inst, stream_id),
                ))
            }
            Command::Synchronize { stream_id, handle } => Some((
                Command::Synchronize { stream_id, handle },
                Stream::new(inst, stream_id),
            )),
            Command::SetStreamPriority {
                stream_id,
                priority,
            } => {
                self.stream_priorities
                    .insert(Stream::new(inst, stream_id), priority);
                None
            }

            Command::EmbedImage {
                stream_id,
                mut embs,
                image_blob,
            } => {
                self.objects
                    .translate_many(ManagedTypes::TokenEmb, inst, &mut embs)?;

                Some((
                    Command::EmbedImage {
                        stream_id,
                        embs,
                        image_blob,
                    },
                    Stream::new(inst, stream_id),
                ))
            }
        };

        Ok(resolved_cmd)
    }

    async fn commit_backend(&mut self, batch: Vec<Command>) {
        let batch_type = batch.first().unwrap().group();
        let correlation_id = self.cmd_id_pool.acquire().unwrap();

        let ((protocol, payload), event) = match batch_type.clone() {
            BatchGroup::Allocate => encode_pb_batch_allocate(correlation_id, batch),
            BatchGroup::Deallocate => encode_pb_batch_deallocate(correlation_id, batch),
            BatchGroup::FillBlock => encode_pb_batch_fill_block(correlation_id, batch),
            BatchGroup::CopyBlock => encode_pb_batch_copy_block(correlation_id, batch),
            BatchGroup::MaskBlock => encode_pb_batch_mask_block(correlation_id, batch),
            BatchGroup::EmbedText => encode_pb_batch_embed_text(correlation_id, batch),
            BatchGroup::DecodeTokenDist => encode_pb_batch_decode_token_dist(correlation_id, batch),
            BatchGroup::SampleTopK => encode_pb_batch_sample_topk(correlation_id, batch),
            BatchGroup::Synchronize => {
                let cmd = batch.into_iter().next().unwrap();
                match cmd {
                    Command::Synchronize { stream_id, handle } => {
                        handle.send(()).unwrap();
                    }
                    _ => unreachable!(),
                }
                return;
            }
            BatchGroup::EmbedImage => encode_pb_batch_embed_image(correlation_id, batch),
            _ => unreachable!(),
        };

        if let Some(events) = event {
            self.event_table.insert(correlation_id, events);
        }

        self.backend
            .send(self.protocol_ids[protocol], payload)
            .await
            .unwrap();
    }

    async fn event_loop(mut rx: Receiver<Vec<u8>>, event_table: Arc<DashMap<u32, Vec<Event>>>) {
        while let Some(resp) = rx.recv().await {
            let resp = pb_bindings::Response::decode(resp.as_ref()).unwrap();

            let correlation_id = resp.correlation_id;
            let payload = resp.command.unwrap();

            if let Some((_, senders)) = event_table.remove(&correlation_id) {
                match payload {
                    pb_bindings::response::Command::SampleTopK(batch) => {
                        for (item, event) in batch.items.into_iter().zip(senders) {
                            match event {
                                Event::SampleTopK(handle) => {
                                    handle.send((item.token_ids, item.probabilities));
                                }
                                _ => unreachable!(),
                            }
                        }
                    }

                    pb_bindings::response::Command::GetInfo(info) => {
                        let sender = senders.into_iter().next().unwrap();
                        match sender {
                            Event::GetInfo(handle) => {
                                handle.send(Info {
                                    version: info.version,
                                    model_name: info.model_name,
                                    block_size: info.block_size,
                                    num_blocks: info.num_available_blocks,
                                    num_embeddings: info.num_available_embeddings,
                                    num_distributions: info.num_available_distributions,
                                });
                            }
                            _ => unreachable!(),
                        }
                    }
                }
            }
        }
    }
}

#[derive(Debug)]
struct ExportedBlocks {
    owner: InstanceId,
    addrs: Vec<IdRepr>,
}

impl ExportedBlocks {
    pub fn new(owner: InstanceId, addrs: Vec<IdRepr>) -> Self {
        Self { owner, addrs }
    }
}

#[derive(Clone)]
pub struct Simulator {
    protocols: Vec<String>,
}

impl Simulator {
    pub fn new() -> Self {
        Self {
            protocols: PROTOCOLS.iter().map(|e| e.to_string()).collect(),
        }
    }
}

impl backend::Simulate for Simulator {
    fn protocols(&self) -> &[String] {
        self.protocols.as_slice()
    }

    fn simulate(&mut self, command: Vec<u8>) -> Option<Vec<u8>> {
        let req = pb_bindings::Request::decode(command.as_ref()).unwrap();

        let resp_payload = match req.command.unwrap() {
            pb_bindings::request::Command::SampleTopKRequest(batch) => {
                let items = batch.items.into_iter().map(|item| {
                    let mut rng = rand::rng();
                    let token_ids: Vec<_> =
                        (0..item.k).map(|_| rng.random_range(0..1000)).collect();

                    let probs: Vec<_> = (0..item.k).map(|_| rng.random()).collect();

                    pb_bindings::SampleTopKResponse {
                        token_ids,
                        probabilities: probs,
                    }
                });

                Some(pb_bindings::response::Command::SampleTopK(
                    pb_bindings::BatchSampleTopKResponse {
                        items: items.collect(),
                    },
                ))
            }

            pb_bindings::request::Command::GetInfo(_) => Some(
                pb_bindings::response::Command::GetInfo(pb_bindings::GetInfoResponse {
                    version: "0.1.0".to_string(),
                    model_name: "DummyModel".to_string(),
                    block_size: 128,
                    num_available_blocks: 1000000,
                    num_available_embeddings: 1000000,
                    num_available_distributions: 100000,
                }),
            ),
            _ => None,
        };

        if let Some(payload) = resp_payload {
            Some(
                pb_bindings::Response {
                    correlation_id: req.correlation_id,
                    command: Some(payload),
                }
                .encode_to_vec(),
            )
        } else {
            None
        }
    }
}

// ----

fn encode_pb_batch_allocate_inner(batch: Vec<Command>) -> Vec<pb_bindings::Allocate> {
    let mut items = Vec::new();
    for cmd in batch {
        match cmd {
            Command::Allocate { stream_id, ty, ids }
            | Command::Deallocate { stream_id, ty, ids } => {
                let kind = match ty {
                    ManagedTypes::KvBlock => pb_bindings::ObjectKind::KvBlock,
                    ManagedTypes::TokenEmb => pb_bindings::ObjectKind::Emb,
                    ManagedTypes::TokenDist => pb_bindings::ObjectKind::Dist,
                    _ => unreachable!(),
                }
                .into();

                for (offset, size) in group_consecutive_ids(&ids) {
                    let pb = pb_bindings::Allocate {
                        kind,
                        object_id_offset: offset,
                        count: size as u32,
                    };
                    items.push(pb);
                }
            }
            _ => unreachable!(),
        }
    }
    items
}

fn encode_pb_batch_allocate(
    correlation_id: u32,
    batch: Vec<Command>,
) -> ((usize, Vec<u8>), Option<Vec<Event>>) {
    let payload = pb_bindings::Request {
        correlation_id,
        command: Some(pb_bindings::request::Command::Allocate(
            pb_bindings::BatchAllocate {
                items: encode_pb_batch_allocate_inner(batch),
            },
        )),
    }
    .encode_to_vec();
    ((PROTOCOL_BASE, payload), None)
}

fn encode_pb_batch_deallocate(
    correlation_id: u32,
    batch: Vec<Command>,
) -> ((usize, Vec<u8>), Option<Vec<Event>>) {
    let payload = pb_bindings::Request {
        correlation_id,
        command: Some(pb_bindings::request::Command::Deallocate(
            pb_bindings::BatchDeallocate {
                items: encode_pb_batch_allocate_inner(batch),
            },
        )),
    }
    .encode_to_vec();
    ((PROTOCOL_BASE, payload), None)
}
fn encode_pb_batch_fill_block(
    correlation_id: u32,
    batch: Vec<Command>,
) -> ((usize, Vec<u8>), Option<Vec<Event>>) {
    let mut items = Vec::new();
    for cmd in batch {
        match cmd {
            Command::FillBlock {
                stream_id,
                block,
                context,
                inputs,
                outputs,
            } => {
                let pb = pb_bindings::FillBlock {
                    block_id: block,
                    context_block_ids: context,
                    input_embedding_ids: inputs,
                    output_embedding_ids: outputs,
                };
                items.push(pb);
            }
            _ => unreachable!(),
        }
    }
    let cmd = pb_bindings::request::Command::FillBlock(pb_bindings::BatchFillBlock { items });
    let payload = pb_bindings::Request {
        correlation_id,
        command: Some(cmd),
    }
    .encode_to_vec();
    ((PROTOCOL_BASE, payload), None)
}

fn encode_pb_batch_copy_block(
    correlation_id: u32,
    batch: Vec<Command>,
) -> ((usize, Vec<u8>), Option<Vec<Event>>) {
    let mut items = Vec::new();
    for cmd in batch {
        match cmd {
            Command::CopyBlock {
                stream_id,
                src_block,
                dst_block,
                src_token_offset,
                dst_token_offset,
                size,
            } => {
                let pb = pb_bindings::CopyBlock {
                    source_block_id: src_block,
                    destination_block_id: dst_block,
                    source_start: src_token_offset,
                    destination_start: dst_token_offset,
                    length: size,
                };
                items.push(pb);
            }
            _ => unreachable!(),
        }
    }
    let cmd = pb_bindings::request::Command::CopyBlock(pb_bindings::BatchCopyBlock { items });
    let payload = pb_bindings::Request {
        correlation_id,
        command: Some(cmd),
    }
    .encode_to_vec();
    ((PROTOCOL_BASE, payload), None)
}

fn encode_pb_batch_mask_block(
    correlation_id: u32,
    batch: Vec<Command>,
) -> ((usize, Vec<u8>), Option<Vec<Event>>) {
    let mut items = Vec::new();
    for cmd in batch {
        match cmd {
            Command::MaskBlock {
                stream_id,
                block,
                mask,
            } => {
                let pb = pb_bindings::MaskBlock {
                    block_id: block,
                    mask,
                };
                items.push(pb);
            }
            _ => unreachable!(),
        }
    }
    let cmd = pb_bindings::request::Command::MaskBlock(pb_bindings::BatchMaskBlock { items });
    let payload = pb_bindings::Request {
        correlation_id,
        command: Some(cmd),
    }
    .encode_to_vec();
    ((PROTOCOL_BASE, payload), None)
}

fn encode_pb_batch_embed_text(
    correlation_id: u32,
    batch: Vec<Command>,
) -> ((usize, Vec<u8>), Option<Vec<Event>>) {
    let mut items = Vec::new();
    for cmd in batch {
        match cmd {
            Command::EmbedText {
                stream_id,
                embs,
                text,
                positions,
            } => {
                for i in 0..embs.len() {
                    let pb = pb_bindings::EmbedText {
                        embedding_id: embs[i],
                        token_id: text[i],
                        position_id: positions[i],
                    };
                    items.push(pb);
                }
            }
            _ => unreachable!(),
        }
    }
    let cmd = pb_bindings::request::Command::EmbedText(pb_bindings::BatchEmbedText { items });
    let payload = pb_bindings::Request {
        correlation_id,
        command: Some(cmd),
    }
    .encode_to_vec();
    ((PROTOCOL_BASE, payload), None)
}

fn encode_pb_batch_decode_token_dist(
    correlation_id: u32,
    batch: Vec<Command>,
) -> ((usize, Vec<u8>), Option<Vec<Event>>) {
    let mut items = Vec::new();
    for cmd in batch {
        match cmd {
            Command::DecodeTokenDist {
                stream_id,
                embs,
                dists,
            } => {
                for i in 0..embs.len() {
                    let pb = pb_bindings::DecodeTokenDistribution {
                        embedding_id: embs[i],
                        distribution_id: dists[i],
                    };
                    items.push(pb);
                }
            }
            _ => unreachable!(),
        }
    }
    let cmd = pb_bindings::request::Command::DecodeTokenDistribution(
        pb_bindings::BatchDecodeTokenDistribution { items },
    );
    let payload = pb_bindings::Request {
        correlation_id,
        command: Some(cmd),
    }
    .encode_to_vec();
    ((PROTOCOL_BASE, payload), None)
}

fn encode_pb_batch_sample_topk(
    correlation_id: u32,
    batch: Vec<Command>,
) -> ((usize, Vec<u8>), Option<Vec<Event>>) {
    let mut items = Vec::new();
    let mut events = Vec::new();
    for cmd in batch {
        match cmd {
            Command::SampleTopK {
                stream_id,
                dist,
                k,
                handle,
            } => {
                let pb = pb_bindings::SampleTopKRequest {
                    distribution_id: dist,
                    k,
                };
                items.push(pb);
                events.push(Event::SampleTopK(handle));
            }
            _ => unreachable!(),
        }
    }
    let cmd =
        pb_bindings::request::Command::SampleTopKRequest(pb_bindings::BatchSampleTopKRequest {
            items,
        });
    let payload = pb_bindings::Request {
        correlation_id,
        command: Some(cmd),
    }
    .encode_to_vec();
    ((PROTOCOL_BASE, payload), Some(events))
}

fn encode_pb_batch_embed_image(
    correlation_id: u32,
    batch: Vec<Command>,
) -> ((usize, Vec<u8>), Option<Vec<Event>>) {
    let mut items = Vec::new();
    for cmd in batch {
        match cmd {
            Command::EmbedImage {
                stream_id,
                embs,
                image_blob,
            } => {
                let pb = pb_bindings_vision::EmbedImage {
                    embedding_ids: embs,
                    image_blob,
                };
                items.push(pb);
            }
            _ => unreachable!(),
        }
    }
    let cmd =
        pb_bindings_vision::request::Command::EmbedImage(pb_bindings_vision::BatchEmbedImage {
            items,
        });
    let payload = pb_bindings_vision::Request {
        correlation_id,
        command: Some(cmd),
    }
    .encode_to_vec();
    ((PROTOCOL_VISION, payload), None)
}
