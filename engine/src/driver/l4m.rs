// Command -> Enum that defines the commands that can be sent to the backend

// Driver

// Executor -> the Trait that defines compatible backends

// Simulator

use crate::backend::ExecuteCommand;
use crate::controller::ControllerError;
use crate::driver::{BatchQueue, BatchingStrategy, DriverError, KorTStrategy, StreamId};
use crate::instance::Id as InstanceId;
use crate::lm::KvBlock;
use crate::object::{IdRepr, ObjectError, VspaceId, group_consecutive_ids};
use crate::tokenizer::BytePairEncoder;
use crate::utils::{Counter, IdPool, RefCounter, TranslationTable};
use crate::{backend, instance, lm, object, tokenizer, utils};
use anyhow::anyhow;
use dashmap::DashMap;
use rand::Rng;
use std::cmp::PartialEq;
use std::collections::{HashMap, VecDeque};
use std::mem;
use std::mem::Discriminant;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc::Receiver;
use tokio::sync::{mpsc, oneshot};

mod protobuf {
    include!(concat!(env!("OUT_DIR"), "/l4m.rs"));
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

#[derive(Debug)]
pub struct Utils {
    pub tokenizer: BytePairEncoder,
    pub block_size: u32,
}

#[derive(Debug)]
pub enum Command {
    Allocate {
        namespace: Namespace,
        ids: Vec<IdRepr>,
    },

    Deallocate {
        namespace: Namespace,
        ids: Vec<IdRepr>,
    },

    FillBlock {
        block: IdRepr,
        context: Vec<IdRepr>,
        inputs: Vec<IdRepr>,
        outputs: Vec<IdRepr>,
    },

    ExportBlocks {
        blocks: Vec<IdRepr>,
        resource_name: String,
        sticky: bool,
    },

    ImportBlocks {
        blocks: Vec<IdRepr>,
        resource_name: String,
    },

    GetAllExportedBlocks {
        handle: oneshot::Sender<Vec<(String, IdRepr)>>,
    },

    CopyBlock {
        src_block: IdRepr,
        dst_block: IdRepr,
        src_token_offset: u32,
        dst_token_offset: u32,
        size: u32,
    },

    MaskBlock {
        block: IdRepr,
        mask: Vec<bool>,
    },

    EmbedText {
        embs: Vec<IdRepr>,
        text: Vec<u32>,
        positions: Vec<u32>,
    },

    DecodeTokenDist {
        embs: Vec<IdRepr>,
        dists: Vec<IdRepr>,
    },

    SampleTopK {
        dist: IdRepr,
        k: u32,
        handle: oneshot::Sender<(Vec<u32>, Vec<f32>)>,
    },
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
enum CommandType {
    Allocate,
    Deallocate,
    FillBlock,
    ExportBlocks,
    ImportBlocks,
    GetAllExportedBlocks,
    CopyBlock,
    MaskBlock,
    EmbedText,
    DecodeTokenDist,
    SampleTopK,
}

impl Command {
    fn to_type(&self) -> CommandType {
        match self {
            Command::Allocate { .. } => CommandType::Allocate,
            Command::Deallocate { .. } => CommandType::Deallocate,
            Command::FillBlock { .. } => CommandType::FillBlock,
            Command::ExportBlocks { .. } => CommandType::ExportBlocks,
            Command::ImportBlocks { .. } => CommandType::ImportBlocks,
            Command::GetAllExportedBlocks { .. } => CommandType::GetAllExportedBlocks,
            Command::CopyBlock { .. } => CommandType::CopyBlock,
            Command::MaskBlock { .. } => CommandType::MaskBlock,
            Command::EmbedText { .. } => CommandType::EmbedText,
            Command::DecodeTokenDist { .. } => CommandType::DecodeTokenDist,
            Command::SampleTopK { .. } => CommandType::SampleTopK,
        }
    }
}

pub enum Event {
    SampleTopK(oneshot::Sender<(Vec<u32>, Vec<f32>)>),

    GetInfo(oneshot::Sender<Info>),
}

pub struct Driver<B> {
    backend: B,
    cmd_id_pool: IdPool<u32>,
    cmd_queue_by_stream: HashMap<(InstanceId, StreamId), VecDeque<Command>>,
    cmd_batcher: CommandBatcher,

    event_table: Arc<DashMap<u32, Vec<Event>>>,
    event_loop_handle: tokio::task::JoinHandle<()>,

    subscriptions: HashMap<String, Vec<InstanceId>>,
    exported_blocks: HashMap<String, ExportedBlocks>,

    obj_id_pools: HashMap<Namespace, IdPool<IdRepr>>,
    obj_ref_counters: HashMap<Namespace, RefCounter<IdRepr>>,
    obj_id_spaces: HashMap<(InstanceId, Namespace), TranslationTable<IdRepr>>,

    info: Info,
    utils: Arc<Utils>,
}

impl<B> Driver<B>
where
    B: ExecuteCommand<protobuf::Request, protobuf::Response>,
{
    pub async fn new(backend: B) {
        let (tx, rx) = mpsc::channel(1000);
        backend.report_to(tx).await;

        let event_table = Arc::new(DashMap::new());
        let event_loop_handle = tokio::spawn(Self::handle_response(rx, event_table.clone()));

        let info = {
            let (info_tx, info_rx) = oneshot::channel();

            event_table.insert(0, vec![Event::GetInfo(info_tx)]);

            backend
                .exec(protobuf::Request {
                    correlation_id: 0,
                    command: Some(protobuf::request::Command::GetInfo(
                        protobuf::GetInfoRequest {},
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

        let utils = Utils {
            // TODO: load the tokenizer model based on the info.model_name
            tokenizer: tokenizer::llama3_tokenizer("../test-tokenizer/tokenizer.model")
                .expect("Tokenizer load failed"),
            block_size: info.block_size,
        };

        let batch_policy = vec![
            (CommandType::Allocate, KorTStrategy::eager().into_box()),
            (CommandType::Deallocate, KorTStrategy::eager().into_box()),
            (CommandType::FillBlock, KorTStrategy::eager().into_box()),
            (CommandType::CopyBlock, KorTStrategy::eager().into_box()),
            (CommandType::MaskBlock, KorTStrategy::eager().into_box()),
            (CommandType::EmbedText, KorTStrategy::eager().into_box()),
            (
                CommandType::DecodeTokenDist,
                KorTStrategy::eager().into_box(),
            ),
            (CommandType::SampleTopK, KorTStrategy::eager().into_box()),
        ];

        let driver = Self {
            backend,
            cmd_id_pool: IdPool::new(u32::MAX),
            cmd_queue_by_stream: HashMap::new(),
            cmd_batcher: CommandBatcher::new(batch_policy),
            event_table,
            event_loop_handle,
            subscriptions: HashMap::new(),
            exported_blocks: HashMap::new(),
            obj_id_pools: HashMap::new(),
            obj_ref_counters: HashMap::new(),
            obj_id_spaces: HashMap::new(),
            info,
            utils: Arc::new(utils),
        };
    }

    pub fn init_instance(&mut self, inst: InstanceId) -> Result<(), DriverError> {
        for namespace in Namespace::iter_all() {
            if self.obj_id_spaces.contains_key(&(inst, namespace)) {
                return Err(DriverError::InstanceAlreadyExists(inst));
            }

            self.obj_id_pools
                .insert(namespace, IdPool::new(IdRepr::MAX));

            self.obj_ref_counters.insert(namespace, RefCounter::new());
            self.obj_id_spaces
                .insert((inst, namespace), TranslationTable::new());
        }

        Ok(())
    }

    pub fn destroy_instance(&mut self, inst: InstanceId) -> Result<(), DriverError> {
        for namespace in Namespace::iter_all() {
            let ids = self
                .obj_id_spaces
                .get(&(inst, namespace))
                .ok_or(DriverError::InstanceNotFound(inst))?
                .to_list();

            self.submit(inst, 0, Command::Deallocate { namespace, ids })?;
            self.obj_id_spaces.remove(&(inst, namespace));
        }

        // Remove all exported blocks
        self.exported_blocks.retain(|_, v| v.owner != inst);

        Ok(())
    }

    pub fn submit(
        &mut self,
        inst: InstanceId,
        stream: StreamId,
        cmd: Command,
    ) -> Result<(), DriverError> {
        if let Some(cmd) = self.process_command(inst, cmd)? {
            self.cmd_queue_by_stream
                .entry((inst, stream))
                .or_insert_with(VecDeque::new)
                .push_back(cmd);
        }
        Ok(())
    }

    fn process_command(
        &mut self,
        inst: InstanceId,
        cmd: Command,
    ) -> Result<Option<Command>, DriverError> {
        // Convert virtual id -> real id

        let resolved_cmd = match cmd {
            Command::Allocate { namespace, ids } => {
                // first check the validity of virtual ID.
                // To ensure that the user is not mapping the VID to the same PID multiple times.
                // If the ID is already mapped, return an error.
                if !namespace.allow_remapping() {
                    let space = self
                        .obj_id_spaces
                        .get(&(inst, namespace))
                        .ok_or(DriverError::InstanceNotFound(inst))?;

                    if let Some(id) = ids.iter().find(|id| space.exists(id)) {
                        return Err(DriverError::Other(format!("ID already exists: {}", id)));
                    }
                }

                let phys_ids = self
                    .obj_id_pools
                    .get_mut(&namespace)
                    .unwrap()
                    .acquire_many(ids.len())
                    .map_err(|e| DriverError::ObjectError(ObjectError::NoAvailableSpace))?;

                self.obj_id_spaces
                    .get_mut(&(inst, namespace))
                    .ok_or(DriverError::InstanceNotFound(inst))?
                    .assign_all(&ids, &phys_ids);

                // If the object is sharable, manage ref counters
                if namespace.is_sharable() {
                    self.obj_ref_counters
                        .get_mut(&namespace)
                        .unwrap()
                        .init_all(&phys_ids);
                }

                Some(Command::Allocate {
                    namespace,
                    ids: phys_ids,
                })
            }
            Command::Deallocate { namespace, mut ids } => {
                self.obj_id_spaces
                    .get(&(inst, namespace))
                    .ok_or(DriverError::InstanceNotFound(inst))?
                    .translate_all(&mut ids)
                    .map_err(|e| DriverError::ObjectError(e));

                let phys_ids = ids;

                let phys_ids_freed = if namespace.is_sharable() {
                    let mut phys_ids_freed = Vec::new();

                    // Decrement ref count
                    for phys_id in phys_ids {
                        let free = self
                            .obj_ref_counters
                            .get_mut(&namespace)
                            .unwrap()
                            .dec(phys_id);

                        if free {
                            self.obj_id_pools
                                .get_mut(&namespace)
                                .unwrap()
                                .release(phys_id);

                            phys_ids_freed.push(phys_id);
                        }
                    }
                    phys_ids_freed
                } else {
                    self.obj_id_pools
                        .get_mut(&namespace)
                        .unwrap()
                        .release_many(&phys_ids);
                    phys_ids
                };

                Some(Command::Deallocate {
                    namespace,
                    ids: phys_ids_freed,
                })
            }
            Command::FillBlock {
                mut block,
                mut context,
                mut inputs,
                mut outputs,
            } => {
                let (phys_block, phys_context) = {
                    let space = self
                        .obj_id_spaces
                        .get_mut(&(inst, Namespace::KV_BLOCK))
                        .ok_or(DriverError::InstanceNotFound(inst))?;

                    space
                        .translate(&mut block)
                        .map_err(|e| DriverError::ObjectError(e))?;

                    space
                        .translate_all(&mut context)
                        .map_err(|e| DriverError::ObjectError(e))?;

                    (block, context)
                };

                let (phys_inputs, phys_outputs) = {
                    let space = self
                        .obj_id_spaces
                        .get_mut(&(inst, Namespace::TOKEN_EMB))
                        .unwrap();

                    space
                        .translate_all(&mut inputs)
                        .map_err(|e| DriverError::ObjectError(e))?;

                    space
                        .translate_all(&mut outputs)
                        .map_err(|e| DriverError::ObjectError(e))?;

                    (inputs, outputs)
                };

                Some(Command::FillBlock {
                    block: phys_block,
                    context: phys_context,
                    inputs: phys_inputs,
                    outputs: phys_outputs,
                })
            }
            Command::ExportBlocks {
                mut blocks,
                resource_name,
                sticky,
            } => {
                self.obj_id_spaces
                    .get(&(inst, Namespace::KV_BLOCK))
                    .unwrap()
                    .translate_all(&mut blocks)
                    .map_err(|e| DriverError::ObjectError(e))?;

                let phys_block_ids = blocks;

                // If sticky, increment ref count so that the blocks are not deallocated even if the owner program terminates.
                if sticky {
                    self.obj_ref_counters
                        .get_mut(&Namespace::KV_BLOCK)
                        .unwrap()
                        .inc_all(&phys_block_ids);
                }

                let exported_blocks = ExportedBlocks::new(inst, phys_block_ids);
                self.exported_blocks.insert(resource_name, exported_blocks);

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

                let space = self
                    .obj_id_spaces
                    .get_mut(&(inst, Namespace::KV_BLOCK))
                    .unwrap();

                if let Some(id) = blocks.iter().find(|id| space.exists(id)) {
                    return Err(DriverError::Other(format!("ID already exists: {}", id)));
                } else {
                    space.assign_all(&blocks, &exported.addrs);
                    self.obj_ref_counters
                        .get_mut(&Namespace::KV_BLOCK)
                        .unwrap()
                        .inc_all(&exported.addrs);
                }
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
            Command::CopyBlock {
                src_block,
                dst_block,
                src_token_offset,
                dst_token_offset,
                size,
            } => {
                let space = self
                    .obj_id_spaces
                    .get_mut(&(inst, Namespace::KV_BLOCK))
                    .ok_or(DriverError::InstanceNotFound(inst))?;

                Some(Command::CopyBlock {
                    src_block: space.lookup(&src_block)?,
                    dst_block: space.lookup(&dst_block)?,
                    src_token_offset,
                    dst_token_offset,
                    size,
                })
            }
            Command::MaskBlock { mut block, mask } => {
                self.obj_id_spaces
                    .get_mut(&(inst, Namespace::KV_BLOCK))
                    .ok_or(DriverError::InstanceNotFound(inst))?
                    .translate(&mut block);

                Some(Command::MaskBlock { block, mask })
            }
            Command::EmbedText {
                mut embs,
                text,
                positions,
            } => {
                self.obj_id_spaces
                    .get_mut(&(inst, Namespace::TOKEN_EMB))
                    .ok_or(DriverError::InstanceNotFound(inst))?
                    .translate_all(&mut embs)
                    .map_err(|e| DriverError::ObjectError(e))?;

                Some(Command::EmbedText {
                    embs,
                    text,
                    positions,
                })
            }
            Command::DecodeTokenDist {
                mut embs,
                mut dists,
            } => {
                self.obj_id_spaces
                    .get(&(inst, Namespace::TOKEN_EMB))
                    .ok_or(DriverError::InstanceNotFound(inst))?
                    .translate_all(&mut embs)
                    .map_err(|e| DriverError::ObjectError(e))?;

                self.obj_id_spaces
                    .get(&(inst, Namespace::TOKEN_DIST))
                    .ok_or(DriverError::InstanceNotFound(inst))?
                    .translate_all(&mut dists)
                    .map_err(|e| DriverError::ObjectError(e))?;

                Some(Command::DecodeTokenDist { embs, dists })
            }
            Command::SampleTopK {
                mut dist,
                k,
                handle,
            } => {
                self.obj_id_spaces
                    .get_mut(&(inst, Namespace::TOKEN_DIST))
                    .ok_or(DriverError::InstanceNotFound(inst))?
                    .translate(&mut dist)
                    .map_err(|e| DriverError::ObjectError(e))?;

                Some(Command::SampleTopK { dist, k, handle })
            }
        };

        Ok(resolved_cmd)
    }

    pub async fn flush(&mut self, now: Instant) -> Result<(), DriverError> {
        let mut keys_to_remove = Vec::new();

        // Horizontal batching: group commands by stream and type.
        for (key, cmd_queue) in self.cmd_queue_by_stream.iter_mut() {
            // non-flushed commands sharing the same stream in the cmd_batcher
            // None -> no commands in the batch queue with the same stream
            let mut prev_cmd_type = self.cmd_batcher.cmd_type(key);

            while !cmd_queue.is_empty() {
                let curr_cmd_type = cmd_queue.front().unwrap().to_type();

                // Vertical batching: Same kind of consecutive commands are batched together.
                // if the current command is different from the previous one, stop batching.
                if let Some(prev_cmd_type) = prev_cmd_type {
                    if curr_cmd_type != prev_cmd_type {
                        break;
                    }
                }
                prev_cmd_type = Some(curr_cmd_type);

                let cmd = cmd_queue.pop_front().unwrap();
                self.cmd_batcher.push(*key, cmd, now);
            }

            // remove the vecdeque if it is empty
            if cmd_queue.is_empty() {
                keys_to_remove.push(*key);
            }
        }

        // Remove empty queues outside the loop.
        for key in keys_to_remove {
            self.cmd_queue_by_stream.remove(&key);
        }

        for cmd_batch in self.cmd_batcher.batch_all(now) {
            self.commit_backend(cmd_batch).await?;
        }

        Ok(())
    }

    async fn commit_backend(&mut self, cmd_batch: Vec<Command>) -> Result<(), DriverError> {
        let cmd_type = cmd_batch.first().unwrap().to_type();

        let (cmd, event) = match cmd_type {
            CommandType::Allocate | CommandType::Deallocate => {
                let mut items = Vec::new();
                for cmd in cmd_batch {
                    match cmd {
                        Command::Allocate { namespace, ids }
                        | Command::Deallocate { namespace, ids } => {
                            let kind = match namespace {
                                Namespace::KV_BLOCK => protobuf::ObjectKind::KvBlock,
                                Namespace::TOKEN_EMB => protobuf::ObjectKind::Emb,
                                Namespace::TOKEN_DIST => protobuf::ObjectKind::Dist,
                                _ => unreachable!(),
                            }
                            .into();

                            for (offset, size) in group_consecutive_ids(&ids) {
                                let pb = protobuf::Allocate {
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
                (
                    protobuf::request::Command::Allocate(protobuf::BatchAllocate { items }),
                    None,
                )
            }

            CommandType::FillBlock => {
                let mut items = Vec::new();
                for cmd in cmd_batch {
                    match cmd {
                        Command::FillBlock {
                            block,
                            context,
                            inputs,
                            outputs,
                        } => {
                            let pb = protobuf::FillBlock {
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
                    protobuf::request::Command::FillBlock(protobuf::BatchFillBlock { items }),
                    None,
                )
            }

            CommandType::CopyBlock => {
                let mut items = Vec::new();
                for cmd in cmd_batch {
                    match cmd {
                        Command::CopyBlock {
                            src_block,
                            dst_block,
                            src_token_offset,
                            dst_token_offset,
                            size,
                        } => {
                            let pb = protobuf::CopyBlock {
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
                    protobuf::request::Command::CopyBlock(protobuf::BatchCopyBlock { items }),
                    None,
                )
            }
            CommandType::MaskBlock => {
                let mut items = Vec::new();

                for cmd in cmd_batch {
                    match cmd {
                        Command::MaskBlock { block, mask } => {
                            let pb = protobuf::MaskBlock {
                                block_id: block,
                                mask,
                            };
                            items.push(pb);
                        }
                        _ => unreachable!(),
                    }
                }

                (
                    protobuf::request::Command::MaskBlock(protobuf::BatchMaskBlock { items }),
                    None,
                )
            }
            CommandType::EmbedText => {
                let mut items = Vec::new();

                for cmd in cmd_batch {
                    match cmd {
                        Command::EmbedText {
                            embs,
                            text,
                            positions,
                        } => {
                            for i in 0..embs.len() {
                                let pb = protobuf::EmbedText {
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
                    protobuf::request::Command::EmbedText(protobuf::BatchEmbedText { items }),
                    None,
                )
            }
            CommandType::DecodeTokenDist => {
                let mut items = Vec::new();

                for cmd in cmd_batch {
                    match cmd {
                        Command::DecodeTokenDist { embs, dists } => {
                            for i in 0..embs.len() {
                                let pb = protobuf::DecodeTokenDistribution {
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
                    protobuf::request::Command::DecodeTokenDistribution(
                        protobuf::BatchDecodeTokenDistribution { items },
                    ),
                    None,
                )
            }
            CommandType::SampleTopK => {
                let mut items = Vec::new();
                let mut events = Vec::new();

                for cmd in cmd_batch {
                    match cmd {
                        Command::SampleTopK { dist, k, handle } => {
                            let pb = protobuf::SampleTopKRequest {
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
                    protobuf::request::Command::SampleTopKRequest(
                        protobuf::BatchSampleTopKRequest { items },
                    ),
                    Some(events),
                )
            }
            _ => unreachable!(),
        };

        let correlation_id = self
            .cmd_id_pool
            .acquire()
            .map_err(|e| DriverError::LockError)?;

        let req = protobuf::Request {
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

    async fn handle_response(
        mut rx: Receiver<protobuf::Response>,
        event_table: Arc<DashMap<u32, Vec<Event>>>,
    ) {
        while let Some(resp) = rx.recv().await {
            let correlation_id = resp.correlation_id;
            let payload = resp.command.unwrap();

            if let Some((_, senders)) = event_table.remove(&correlation_id) {
                match payload {
                    protobuf::response::Command::SampleTopK(batch) => {
                        for (item, event) in batch.items.into_iter().zip(senders) {
                            match event {
                                Event::SampleTopK(handle) => {
                                    handle.send((item.token_ids, item.probabilities));
                                }
                                _ => unreachable!(),
                            }
                        }
                    }

                    protobuf::response::Command::GetInfo(info) => {
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
struct CommandBatcher {
    type_tracker: HashMap<(InstanceId, StreamId), CommandType>,
    type_tracker_inv: HashMap<CommandType, Vec<(InstanceId, StreamId)>>,
    queue: HashMap<CommandType, BatchQueue<Command>>,
}

impl CommandBatcher {
    fn new(policies: Vec<(CommandType, Box<dyn BatchingStrategy>)>) -> Self {
        let mut queue = HashMap::new();
        let mut type_tracker_inv = HashMap::new();
        for (cmd_type, strategy) in policies {
            queue.insert(cmd_type, BatchQueue::new(strategy));
            type_tracker_inv.insert(cmd_type, Vec::new());
        }

        Self {
            type_tracker: HashMap::new(),
            type_tracker_inv,
            queue,
        }
    }

    fn cmd_type(&self, key: &(InstanceId, StreamId)) -> Option<CommandType> {
        self.type_tracker.get(key).cloned()
    }

    fn push(&mut self, key: (InstanceId, StreamId), cmd: Command, now: Instant) {
        let cmd_type = cmd.to_type();

        // Ensure the key's command type is consistent.
        if let Some(&existing_type) = self.type_tracker.get(&key) {
            assert_eq!(
                existing_type, cmd_type,
                "Mismatched command type for key {:?}",
                key
            );
        } else {
            self.type_tracker.insert(key, cmd_type);
            self.type_tracker_inv.get_mut(&cmd_type).unwrap().push(key);
        }

        self.queue.get_mut(&cmd_type).unwrap().push(cmd, now);
    }

    fn batch_all(&mut self, now: Instant) -> Vec<Vec<Command>> {
        //
        let mut all_batched_cmds = Vec::new();

        for (cmd_type, queue) in self.queue.iter_mut() {
            if let Some(cmds) = queue.batch(now) {
                for key in self
                    .type_tracker_inv
                    .get_mut(cmd_type)
                    .unwrap()
                    .drain(..cmds.len())
                {
                    self.type_tracker.remove(&key);
                }

                all_batched_cmds.push(cmds);
            }
        }

        all_batched_cmds
    }
}

#[derive(Debug)]
struct ExportedBlocks {
    owner: instance::Id,
    addrs: Vec<IdRepr>,
}

impl ExportedBlocks {
    pub fn new(owner: instance::Id, addrs: Vec<IdRepr>) -> Self {
        Self { owner, addrs }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct Namespace(usize);

impl Namespace {
    pub const KV_BLOCK: Namespace = Namespace(0);
    pub const TOKEN_EMB: Namespace = Namespace(1);
    pub const TOKEN_DIST: Namespace = Namespace(2);
    pub fn size() -> usize {
        3
    }

    pub fn is_sharable(&self) -> bool {
        match self.0 {
            0 => true,
            1 => false,
            2 => false,
            _ => unreachable!(),
        }
    }

    pub fn allow_remapping(&self) -> bool {
        match self.0 {
            0 => false,
            1 => true,
            2 => true,
            _ => unreachable!(),
        }
    }

    pub fn iter_all() -> impl Iterator<Item = Namespace> {
        (0..Self::size()).map(Namespace)
    }
}

#[derive(Clone)]
pub struct Simulator {}

impl backend::Simulate<protobuf::Request, protobuf::Response> for Simulator {
    fn simulate(&mut self, req: protobuf::Request) -> Option<protobuf::Response> {
        let resp_payload = match req.command.unwrap() {
            protobuf::request::Command::SampleTopKRequest(batch) => {
                let items = batch.items.into_iter().map(|item| {
                    let mut rng = rand::rng();
                    let token_ids: Vec<_> =
                        (0..item.k).map(|_| rng.random_range(0..1000)).collect();

                    let probs: Vec<_> = (0..item.k).map(|_| rng.random()).collect();

                    protobuf::SampleTopKResponse {
                        token_ids,
                        probabilities: probs,
                    }
                });

                Some(protobuf::response::Command::SampleTopK(
                    protobuf::BatchSampleTopKResponse {
                        items: items.collect(),
                    },
                ))
            }

            protobuf::request::Command::GetInfo(_) => Some(protobuf::response::Command::GetInfo(
                protobuf::GetInfoResponse {
                    version: "0.1.0".to_string(),
                    model_name: "DummyModel".to_string(),
                    block_size: 128,
                    num_available_blocks: 1000000,
                    num_available_embeddings: 1000000,
                    num_available_distributions: 100000,
                },
            )),
            _ => None,
        };

        if let Some(payload) = resp_payload {
            Some(protobuf::Response {
                correlation_id: req.correlation_id,
                command: Some(payload),
            })
        } else {
            None
        }
    }
}
