use crate::backend::ExecuteCommand;
use crate::controller::ControllerError;
use crate::driver::{
    BatchQueue, Batchable, Batcher, BatchingStrategy, DriverError, KorTStrategy, StreamId,
};
use crate::instance::Id as InstanceId;
use crate::object::{
    IdRepr, ObjectError, ObjectManager, ObjectType, VspaceId, group_consecutive_ids,
};
use crate::tokenizer::BytePairEncoder;
use crate::utils::{Counter, IdPool};
use crate::{backend, instance, lm, object, tokenizer, utils};
use anyhow::anyhow;
use dashmap::DashMap;
use rand::Rng;
use std::cell::RefCell;
use std::cmp::PartialEq;
use std::collections::{HashMap, VecDeque};
use std::mem;
use std::mem::Discriminant;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc::Receiver;
use tokio::sync::{mpsc, oneshot};

pub const PROTOCOL: &str = "l4m"; // for future backward compatibility

mod pb_bindings {
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
        ty: ManagedTypes,
        ids: Vec<IdRepr>,
    },

    Deallocate {
        ty: ManagedTypes,
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
enum CommandGroup {
    Allocate,
    Deallocate,
    FillBlock,
    CopyBlock,
    MaskBlock,
    EmbedText,
    DecodeTokenDist,
    SampleTopK,
}

impl Batchable<CommandGroup> for Command {
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
            _ => unreachable!(),
        }
    }

    fn group(&self) -> CommandGroup {
        match self {
            Command::Allocate { .. } => CommandGroup::Allocate,
            Command::Deallocate { .. } => CommandGroup::Deallocate,
            Command::FillBlock { .. } => CommandGroup::FillBlock,
            Command::CopyBlock { .. } => CommandGroup::CopyBlock,
            Command::MaskBlock { .. } => CommandGroup::MaskBlock,
            Command::EmbedText { .. } => CommandGroup::EmbedText,
            Command::DecodeTokenDist { .. } => CommandGroup::DecodeTokenDist,
            Command::SampleTopK { .. } => CommandGroup::SampleTopK,
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
pub struct Driver<B> {
    backend: B,
    cmd_id_pool: IdPool<u32>,
    cmd_batcher: Batcher<Command, (InstanceId, StreamId), CommandGroup>,

    event_table: Arc<DashMap<u32, Vec<Event>>>,
    event_loop_handle: tokio::task::JoinHandle<()>,

    subscriptions: HashMap<String, Vec<InstanceId>>,
    exported_blocks: HashMap<String, ExportedBlocks>,

    // Rc for shared ownership across multiple "extension drivers"
    objects: Rc<RefCell<ObjectManager<InstanceId, ManagedTypes>>>,

    info: Info,
    utils: Arc<Utils>,
}

// Read-only view of the object registry
#[derive(Debug)]
pub struct ObjectRegistryView {
    objects: Rc<RefCell<ObjectManager<InstanceId, ManagedTypes>>>,
}

impl ObjectRegistryView {
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

impl<B> Driver<B>
where
    B: ExecuteCommand<pb_bindings::Request, pb_bindings::Response>,
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

        let utils = Utils {
            // TODO: load the tokenizer model based on the info.model_name
            tokenizer: tokenizer::llama3_tokenizer("../test-tokenizer/tokenizer.model")
                .expect("Tokenizer load failed"),
            block_size: info.block_size,
        };

        let driver = Self {
            backend,
            cmd_id_pool: IdPool::new(u32::MAX),
            cmd_batcher: Batcher::new(),
            event_table,
            event_loop_handle,
            subscriptions: HashMap::new(),
            exported_blocks: HashMap::new(),
            objects: Rc::new(RefCell::new(ObjectManager::new())),
            info,
            utils: Arc::new(utils),
        };
    }

    pub fn get_object_registry_view(&self) -> ObjectRegistryView {
        ObjectRegistryView::new(self.objects.clone())
    }

    pub fn init_instance(&mut self, inst: InstanceId) -> Result<(), DriverError> {
        // let mut objects = RefCell::borrow_mut(&self.objects);
        //
        // for namespace in Namespace::iter_all() {
        //     if objects.vid.contains_key(&(inst, namespace)) {
        //         return Err(DriverError::InstanceAlreadyExists(inst));
        //     }
        //
        //     objects
        //         .vid
        //         .insert((inst, namespace), TranslationTable::new());
        // }

        Ok(())
    }

    pub fn destroy_instance(&mut self, inst: InstanceId) -> Result<(), DriverError> {
        let mut cmds = Vec::new();

        for ty in [
            ManagedTypes::KvBlock,
            ManagedTypes::TokenEmb,
            ManagedTypes::TokenDist,
        ] {
            let mut obj_mgr = RefCell::borrow_mut(&self.objects);

            cmds.push(Command::Deallocate {
                ty,
                ids: obj_mgr.all_names(ty, inst)?,
            })
        }

        for cmd in cmds {
            self.submit(inst, 0, cmd, Instant::now())?;
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
        now: Instant,
    ) -> Result<(), DriverError> {
        if let Some(cmd) = self.translate_cmd(inst, cmd)? {
            self.cmd_batcher.push((inst, stream), cmd, now);
        }
        Ok(())
    }

    fn translate_cmd(
        &mut self,
        inst: InstanceId,
        cmd: Command,
    ) -> Result<Option<Command>, DriverError> {
        // Convert virtual id -> real id

        //let mut objects = RefCell::borrow_mut(&self.objects);
        let mut objects = RefCell::borrow_mut(&self.objects);

        let resolved_cmd = match cmd {
            Command::Allocate { ty, ids } => {
                let ids = objects.create_many(ty, inst, ids)?;

                Some(Command::Allocate { ty, ids })
            }
            Command::Deallocate { ty, ids } => {
                let ids = objects.destroy_many(ty, inst, &ids)?;

                if ids.is_empty() {
                    return Ok(None);
                }

                Some(Command::Deallocate { ty, ids })
            }
            Command::FillBlock {
                mut block,
                mut context,
                mut inputs,
                mut outputs,
            } => {
                objects.translate(ManagedTypes::KvBlock, inst, &mut block)?;
                objects.translate_many(ManagedTypes::KvBlock, inst, &mut context)?;
                objects.translate_many(ManagedTypes::TokenEmb, inst, &mut inputs)?;
                objects.translate_many(ManagedTypes::TokenEmb, inst, &mut outputs)?;

                Some(Command::FillBlock {
                    block,
                    context,
                    inputs,
                    outputs,
                })
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
                mut src_block,
                mut dst_block,
                src_token_offset,
                dst_token_offset,
                size,
            } => {
                objects.translate(ManagedTypes::KvBlock, inst, &mut src_block)?;
                objects.translate(ManagedTypes::KvBlock, inst, &mut dst_block)?;

                Some(Command::CopyBlock {
                    src_block,
                    dst_block,
                    src_token_offset,
                    dst_token_offset,
                    size,
                })
            }
            Command::MaskBlock { mut block, mask } => {
                objects.translate(ManagedTypes::KvBlock, inst, &mut block)?;

                Some(Command::MaskBlock { block, mask })
            }
            Command::EmbedText {
                mut embs,
                text,
                positions,
            } => {
                objects.translate_many(ManagedTypes::TokenEmb, inst, &mut embs)?;

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
                objects.translate_many(ManagedTypes::TokenEmb, inst, &mut embs)?;
                objects.translate_many(ManagedTypes::TokenDist, inst, &mut dists)?;

                Some(Command::DecodeTokenDist { embs, dists })
            }
            Command::SampleTopK {
                mut dist,
                k,
                handle,
            } => {
                objects.translate(ManagedTypes::TokenDist, inst, &mut dist)?;

                Some(Command::SampleTopK { dist, k, handle })
            }
        };

        Ok(resolved_cmd)
    }

    pub async fn flush(&mut self, now: Instant) -> Result<(), DriverError> {
        for (_, cmd_batch) in self.cmd_batcher.batch(now) {
            self.commit_backend(cmd_batch).await?;
        }

        Ok(())
    }

    async fn commit_backend(&mut self, cmd_batch: Vec<Command>) -> Result<(), DriverError> {
        let cmd_type = cmd_batch.first().unwrap().group();

        let (cmd, event) = match cmd_type.clone() {
            CommandGroup::Allocate | CommandGroup::Deallocate => {
                let mut items = Vec::new();
                for cmd in cmd_batch {
                    match cmd {
                        Command::Allocate { ty, ids } | Command::Deallocate { ty, ids } => {
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

                let pb = if cmd_type == CommandGroup::Allocate {
                    pb_bindings::request::Command::Allocate(pb_bindings::BatchAllocate { items })
                } else {
                    pb_bindings::request::Command::Deallocate(pb_bindings::BatchDeallocate {
                        items,
                    })
                };

                (pb, None)
            }

            CommandGroup::FillBlock => {
                let mut items = Vec::new();
                for cmd in cmd_batch {
                    match cmd {
                        Command::FillBlock {
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

            CommandGroup::CopyBlock => {
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
            CommandGroup::MaskBlock => {
                let mut items = Vec::new();

                for cmd in cmd_batch {
                    match cmd {
                        Command::MaskBlock { block, mask } => {
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
            CommandGroup::EmbedText => {
                let mut items = Vec::new();

                for cmd in cmd_batch {
                    match cmd {
                        Command::EmbedText {
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
            CommandGroup::DecodeTokenDist => {
                let mut items = Vec::new();

                for cmd in cmd_batch {
                    match cmd {
                        Command::DecodeTokenDist { embs, dists } => {
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
            CommandGroup::SampleTopK => {
                let mut items = Vec::new();
                let mut events = Vec::new();

                for cmd in cmd_batch {
                    match cmd {
                        Command::SampleTopK { dist, k, handle } => {
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
    owner: instance::Id,
    addrs: Vec<IdRepr>,
}

impl ExportedBlocks {
    pub fn new(owner: instance::Id, addrs: Vec<IdRepr>) -> Self {
        Self { owner, addrs }
    }
}

#[derive(Clone)]
pub struct Simulator {}

impl backend::Simulate<pb_bindings::Request, pb_bindings::Response> for Simulator {
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
