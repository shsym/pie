use crate::batching::{Batchable, Batcher, BatchingStrategy, KorTStrategy};
use crate::controller_old::ControllerError;
use crate::driver::{Driver, DriverError, DynCommand};
use crate::instance::Id as InstanceId;
use crate::object::{
    IdRepr, ObjectError, ObjectManager, ObjectType, VspaceId, group_consecutive_ids,
};
use crate::tokenizer::BytePairEncoder;
use crate::utils::{Counter, IdPool};
use crate::{backend_old, lm, object, tokenizer, utils};
use anyhow::anyhow;
use dashmap::DashMap;
use rand::Rng;
use std::any::TypeId;
use std::cell::RefCell;
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

pub const PROTOCOL: &str = "l4m"; // for future backward compatibility

mod pb_bindings {
    include!(concat!(env!("OUT_DIR"), "/l4m.rs"));
}

pub trait CompatibleBackend:
    backend_old::Protocol<pb_bindings::Request, pb_bindings::Response>
{
}
impl<T> CompatibleBackend for T where
    T: backend_old::Protocol<pb_bindings::Request, pb_bindings::Response>
{
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

    Cleanup,
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
        todo!()
    }

    fn allow_remapping(&self) -> bool {
        todo!()
    }

    fn max_capacity(&self) -> IdRepr {
        todo!()
    }
}

#[derive(Debug)]
pub struct L4m<B> {
    backend: B,
    cmd_id_pool: IdPool<u32>,
    cmd_batcher: Batcher<Command, Stream, BatchGroup>,

    event_table: Arc<DashMap<u32, Vec<Event>>>,
    event_loop_handle: task::JoinHandle<()>,

    subscriptions: HashMap<String, Vec<InstanceId>>,
    exported_blocks: HashMap<String, ExportedBlocks>,

    // Rc for shared ownership across multiple "extension drivers"
    objects: Rc<RefCell<ObjectManager<InstanceId, ManagedTypes>>>,

    stream_priorities: HashMap<Stream, StreamPriority>,

    info: Info,
    tokenizer: Arc<BytePairEncoder>,
}

// Read-only view of the object registry
#[derive(Debug)]
pub struct ObjectView {
    objects: Rc<RefCell<ObjectManager<InstanceId, ManagedTypes>>>,
}

impl ObjectView {
    fn new(objects: Rc<RefCell<ObjectManager<InstanceId, ManagedTypes>>>) -> Self {
        Self { objects }
    }

    pub fn translate(
        &self,
        ty: ManagedTypes,
        inst: InstanceId,
        id: &mut IdRepr,
    ) -> Result<(), ObjectError> {
        let object = self.objects.borrow();
        object.translate(ty, inst, id)
    }

    pub fn translate_many(
        &self,
        ty: ManagedTypes,
        inst: InstanceId,
        ids: &mut [IdRepr],
    ) -> Result<(), ObjectError> {
        let object = self.objects.borrow();
        object.translate_many(ty, inst, ids)
    }
}

impl<B> Driver for L4m<B>
where
    B: CompatibleBackend,
{
    fn accepts(&self) -> &[TypeId] {
        vec![TypeId::of::<Command>()]
    }

    fn create_inst(&mut self, inst: InstanceId) -> Result<(), DriverError> {
        todo!()
    }

    fn destroy_inst(&mut self, inst: InstanceId) -> Result<(), DriverError> {
        todo!()
    }

    fn submit(&mut self, inst: InstanceId, cmd: DynCommand) -> Result<(), DriverError> {
        if let Some(cmd) = cmd.as_any().downcast_ref::<Command>() {
            if let Some((cmd, mut stream)) = self.translate_cmd(inst, cmd)? {
                // adjust stream priority
                if let Some(priority) = self.stream_priorities.get(&stream) {
                    stream.set_priority(*priority)
                }

                self.cmd_batcher.push(stream, cmd, Instant::now());
            }
            Ok(())
        } else {
            return Err(DriverError::Other("Command type mismatch.".to_string()));
        }

        todo!()
    }

    async fn flush(&mut self) -> Result<(), DriverError> {
        for (_, cmd_batch) in self.cmd_batcher.batch(Instant::now()) {
            self.commit_backend(cmd_batch).await?;
        }

        Ok(())
    }
}

impl<B> L4m<B>
where
    B: CompatibleBackend,
{
    pub async fn new(backend: B) {
        let (tx, rx) = mpsc::channel(1000);
        backend.report_to(tx).await;

        let event_table = Arc::new(DashMap::new());
        let event_loop_handle = tokio::spawn(Self::event_loop(rx, event_table.clone()));

        let info = {
            let (info_tx, info_rx) = oneshot::channel();

            event_table.insert(0, vec![Event::GetInfo(info_tx)]);

            backend
                .exec(pb_bindings::Request {
                    correlation_id: 0,
                    command: Some(pb_bindings::request::Command::GetInfo(
                        pb_bindings::GetInfoRequest {},
                    )),
                })
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

        let driver = Self {
            backend,
            cmd_id_pool: IdPool::new(u32::MAX),
            cmd_batcher: Batcher::new(),
            event_table,
            event_loop_handle,
            subscriptions: HashMap::new(),
            exported_blocks: HashMap::new(),
            objects: Rc::new(RefCell::new(ObjectManager::new())),
            stream_priorities: HashMap::new(),
            info,
            tokenizer: Arc::new(tokenizer),
        };
    }

    pub fn get_object_registry_view(&self) -> ObjectView {
        ObjectView::new(self.objects.clone())
    }

    fn cleanup(&mut self, inst: InstanceId) -> Result<(), DriverError> {
        let mut cmds = Vec::new();

        for ty in [
            ManagedTypes::KvBlock,
            ManagedTypes::TokenEmb,
            ManagedTypes::TokenDist,
        ] {
            let mut obj_mgr = RefCell::borrow_mut(&self.objects);

            cmds.push(Command::Deallocate {
                stream_id: 0,
                ty,
                ids: obj_mgr.all_names(ty, inst)?,
            })
        }

        for cmd in cmds {
            self.submit(inst, DynCommand::new(cmd))?;
        }

        // Remove all exported blocks
        self.exported_blocks.retain(|_, v| v.owner != inst);

        Ok(())
    }

    fn translate_cmd(
        &mut self,
        inst: InstanceId,
        cmd: Command,
    ) -> Result<Option<(Command, Stream)>, DriverError> {
        let mut objects = RefCell::borrow_mut(&self.objects);

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
                let ids = objects.create_many(ty, inst, ids)?;

                Some((
                    Command::Allocate { stream_id, ty, ids },
                    Stream::new(inst, stream_id),
                ))
            }
            Command::Deallocate { stream_id, ty, ids } => {
                let ids = objects.destroy_many(ty, inst, &ids)?;

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
                objects.translate(ManagedTypes::KvBlock, inst, &mut block)?;
                objects.translate_many(ManagedTypes::KvBlock, inst, &mut context)?;
                objects.translate_many(ManagedTypes::TokenEmb, inst, &mut inputs)?;
                objects.translate_many(ManagedTypes::TokenEmb, inst, &mut outputs)?;

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
                objects.translate_many(ManagedTypes::KvBlock, inst, &mut blocks)?;

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

                objects.create_ref_many(ManagedTypes::KvBlock, inst, blocks, &exported.addrs)?;

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
                objects.translate(ManagedTypes::KvBlock, inst, &mut src_block)?;
                objects.translate(ManagedTypes::KvBlock, inst, &mut dst_block)?;

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
                objects.translate(ManagedTypes::KvBlock, inst, &mut block)?;

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
                objects.translate_many(ManagedTypes::TokenEmb, inst, &mut embs)?;

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
                objects.translate_many(ManagedTypes::TokenEmb, inst, &mut embs)?;
                objects.translate_many(ManagedTypes::TokenDist, inst, &mut dists)?;

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
                objects.translate(ManagedTypes::TokenDist, inst, &mut dist)?;

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

            Command::Cleanup => {
                self.cleanup(inst)?;
                None
            }
        };

        Ok(resolved_cmd)
    }

    async fn commit_backend(&mut self, cmd_batch: Vec<Command>) -> Result<(), DriverError> {
        let cmd_type = cmd_batch.first().unwrap().group();

        let (cmd, event) = match cmd_type.clone() {
            BatchGroup::Allocate | BatchGroup::Deallocate => {
                let mut items = Vec::new();
                for cmd in cmd_batch {
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

                let pb = if cmd_type == BatchGroup::Allocate {
                    pb_bindings::request::Command::Allocate(pb_bindings::BatchAllocate { items })
                } else {
                    pb_bindings::request::Command::Deallocate(pb_bindings::BatchDeallocate {
                        items,
                    })
                };

                (pb, None)
            }

            BatchGroup::FillBlock => {
                let mut items = Vec::new();
                for cmd in cmd_batch {
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
                (
                    pb_bindings::request::Command::FillBlock(pb_bindings::BatchFillBlock { items }),
                    None,
                )
            }

            BatchGroup::CopyBlock => {
                let mut items = Vec::new();
                for cmd in cmd_batch {
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

                (
                    pb_bindings::request::Command::CopyBlock(pb_bindings::BatchCopyBlock { items }),
                    None,
                )
            }
            BatchGroup::MaskBlock => {
                let mut items = Vec::new();

                for cmd in cmd_batch {
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

                (
                    pb_bindings::request::Command::MaskBlock(pb_bindings::BatchMaskBlock { items }),
                    None,
                )
            }
            BatchGroup::EmbedText => {
                let mut items = Vec::new();

                for cmd in cmd_batch {
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

                (
                    pb_bindings::request::Command::EmbedText(pb_bindings::BatchEmbedText { items }),
                    None,
                )
            }
            BatchGroup::DecodeTokenDist => {
                let mut items = Vec::new();

                for cmd in cmd_batch {
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

                (
                    pb_bindings::request::Command::DecodeTokenDistribution(
                        pb_bindings::BatchDecodeTokenDistribution { items },
                    ),
                    None,
                )
            }
            BatchGroup::SampleTopK => {
                let mut items = Vec::new();
                let mut events = Vec::new();

                for cmd in cmd_batch {
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

                            // register event handle
                            events.push(Event::SampleTopK(handle));
                        }
                        _ => unreachable!(),
                    }
                }
                (
                    pb_bindings::request::Command::SampleTopKRequest(
                        pb_bindings::BatchSampleTopKRequest { items },
                    ),
                    Some(events),
                )
            }

            BatchGroup::Synchronize => {
                let cmd = cmd_batch.into_iter().next().unwrap();

                match cmd {
                    Command::Synchronize { stream_id, handle } => {
                        handle.send(()).unwrap();
                    }
                    _ => unreachable!(),
                }

                return Ok(());
            }
            _ => unreachable!(),
        };

        let correlation_id = self
            .cmd_id_pool
            .acquire()
            .map_err(|e| DriverError::LockError)?;

        let req = pb_bindings::Request {
            correlation_id,
            command: Some(cmd),
        };

        if let Some(events) = event {
            self.event_table.insert(correlation_id, events);
        }

        self.backend
            .exec(req)
            .await
            .map_err(|e| DriverError::Other(e.to_string()))?;

        Ok(())
    }

    async fn event_loop(
        mut rx: Receiver<pb_bindings::Response>,
        event_table: Arc<DashMap<u32, Vec<Event>>>,
    ) {
        while let Some(resp) = rx.recv().await {
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
pub struct Simulator {}

impl backend_old::Simulate<pb_bindings::Request, pb_bindings::Response> for Simulator {
    fn simulate(&mut self, req: pb_bindings::Request) -> Option<pb_bindings::Response> {
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
            Some(pb_bindings::Response {
                correlation_id: req.correlation_id,
                command: Some(payload),
            })
        } else {
            None
        }
    }
}
