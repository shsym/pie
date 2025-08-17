use crate::backend::Backend;
use crate::batching::{Batchable, Batcher, BatchingStrategy, ManualStrategy};
use crate::instance::Id as InstanceId;
use crate::object::{IdRepr, ObjectManager, ObjectType, group_consecutive_ids};
use crate::service::{Service, ServiceError, install_service};
use crate::tokenizer::{BytePairEncoder, load_merge_rules};
use crate::utils::IdPool;
use crate::{backend, batching, runtime, service};
use dashmap::DashMap;
use prost::Message;
use rand::Rng;
use std::cmp::{Ordering, PartialEq};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::AtomicBool;

use serde::Serialize;
use tokio::sync::RwLock;

use crate::runtime::trap;
use std::sync::{Arc, LazyLock, OnceLock};
use std::time::{Duration, Instant};
use tokio::sync::mpsc::{Receiver, Sender};
use tokio::sync::{mpsc, oneshot};
use tokio::time::timeout;

mod pb_bindings {
    include!(concat!(env!("OUT_DIR"), "/l4m.rs"));
}

mod pb_bindings_vision {
    include!(concat!(env!("OUT_DIR"), "/l4m.vision.rs"));
}

macro_rules! try_trap {
    ($result:expr, $inst_id:expr, $msg:expr) => {
        match $result {
            Ok(val) => val,
            Err(e) => {
                runtime::trap($inst_id, format!("{}: {}", $msg, e));
                return None;
            }
        }
    };
}
// Protocol definitions - backends dynamically report their supported protocols
const PROTOCOL_BASE: usize = 0;
const PROTOCOL_VISION: usize = 1;
const GLOBAL_OWNER_ID: InstanceId = InstanceId::from_u128(0);

static AVAILABLE_MODELS: std::sync::LazyLock<boxcar::Vec<(String, usize)>> =
    std::sync::LazyLock::new(boxcar::Vec::new);

/// Defines the configurable batching strategy options.
#[derive(Debug, Clone, Copy)]
pub enum BatchingStrategyConfiguration {
    /// Batch based only on a timeout.
    TOnly { t: Duration },
    /// Batch based only on the number of items.
    KOnly { k: usize },
    /// Batch based on whichever condition is met first: size `k` or timeout `t`.
    KOrT { k: usize, t: Duration },
    /// Use the adaptive (manual trigger) strategy.
    Adaptive,
}

static FORWARD_STRATEGY: OnceLock<BatchingStrategyConfiguration> = OnceLock::new();

pub fn set_batching_strategy(
    strategy: BatchingStrategyConfiguration,
) -> Result<(), BatchingStrategyConfiguration> {
    FORWARD_STRATEGY.set(strategy)
}

/// Holds shared triggers for manual batching strategies.
struct ManualTriggers {
    fill_block_trigger: Arc<AtomicBool>,
    forward_text_trigger: Arc<AtomicBool>,
}

// A static, lazily-initialized context to hold the triggers.
static TRIGGERS: std::sync::LazyLock<ManualTriggers> =
    std::sync::LazyLock::new(|| ManualTriggers {
        fill_block_trigger: Arc::new(AtomicBool::new(true)),
        forward_text_trigger: Arc::new(AtomicBool::new(true)),
    });

pub async fn attach_new_remote_backend(name: &str, endpoint: String) -> Option<()> {
    let backend = match backend::ZmqBackend::bind(&endpoint).await {
        Ok(b) => b,
        Err(_) => return None,
    };

    let l4m = L4m::new(backend).await;
    let model_name = l4m.info.model_name.clone();

    if let Some(service_id) = install_service(name, l4m) {
        AVAILABLE_MODELS.push((model_name, service_id));
        Some(())
    } else {
        None
    }
}

pub async fn attach_new_backend<B>(name: &str, backend: B) -> Option<()>
where
    B: Backend + 'static,
{
    let l4m = L4m::new(backend).await;
    let model_name = l4m.info.model_name.clone();

    if let Some(service_id) = install_service(name, l4m) {
        AVAILABLE_MODELS.push((model_name, service_id));
        Some(())
    } else {
        None
    }
}

pub async fn gather_stats() -> String {
    let mut stats = Vec::<Info>::new();
    for (_, (_, service_id)) in AVAILABLE_MODELS.iter() {
        let (tx, rx) = oneshot::channel();

        Command::GetInfo { handle: tx }.dispatch(*service_id).ok();

        if let Ok(info) = rx.await {
            stats.push(info);
        }
    }
    serde_json::to_string(&stats).unwrap_or_else(|_| "Serialization error".to_string())
}

pub fn available_models() -> Vec<String> {
    AVAILABLE_MODELS
        .iter()
        .map(|(_, (model_name, _))| model_name.clone())
        .collect()
}

pub fn model_service_id(model_name: &str) -> Option<usize> {
    AVAILABLE_MODELS
        .iter()
        .find(|(_, (name, _))| name == model_name)
        .map(|(_, (_, service_id))| *service_id)
}

pub fn cleanup_instance(inst_id: InstanceId) {
    AVAILABLE_MODELS.iter().for_each(|(_, (_, service_id))| {
        Command::Destroy { inst_id }.dispatch(*service_id).ok();
    })
}

#[derive(Debug, Serialize)]
pub struct Info {
    pub version: String,
    pub model_name: String,
    pub kv_page_size: u32,
    pub num_kv_pages: u32,
    pub num_embeddings: u32,
    pub num_adapters: u32,

    #[serde(skip)]
    pub tokenizer: Arc<BytePairEncoder>,
}

pub type LocalStreamId = u32;

#[derive(Debug, Clone, Copy, Default)]
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

impl Default for StreamPriority {
    fn default() -> Self {
        StreamPriority::Normal
    }
}

#[derive(Debug)]
pub enum Command {
    Destroy {
        inst_id: InstanceId,
    },

    GetInfo {
        handle: oneshot::Sender<Info>,
    },

    GetBlockSize {
        handle: oneshot::Sender<u32>,
    },

    GetTokenizer {
        handle: oneshot::Sender<Arc<BytePairEncoder>>,
    },

    Allocate {
        inst_id: InstanceId,
        stream_id: LocalStreamId,
        ty: ManagedTypes,
        ids: Vec<IdRepr>,
    },

    Deallocate {
        inst_id: InstanceId,
        stream_id: LocalStreamId,
        ty: ManagedTypes,
        ids: Vec<IdRepr>,
    },

    // --------------------
    Export {
        inst_id: InstanceId,
        ty: ManagedTypes,
        ids: Vec<IdRepr>,
        resource_name: String,
    },

    GetExportedList {
        inst_id: InstanceId,
        ty: ManagedTypes,
        handle: oneshot::Sender<Vec<(String, u32)>>,
    },

    Unexport {
        inst_id: InstanceId,
        ty: ManagedTypes,
        resource_name: String,
    },

    Import {
        inst_id: InstanceId,
        ty: ManagedTypes,
        ids: Vec<IdRepr>,
        resource_name: String,
    },

    // --------------------
    Forward {
        inst_id: InstanceId,
        stream_id: LocalStreamId,
        kv_page_last_len: u32,
        kv_pages: Vec<IdRepr>,
        input_embeds: Vec<IdRepr>,
        output_embeds: Vec<IdRepr>,
    },

    CopyKvPage {
        inst_id: InstanceId,
        stream_id: LocalStreamId,
        src_kv_page: IdRepr,
        dst_kv_page: IdRepr,
        src_token_offset: u32,
        dst_token_offset: u32,
        size: u32,
    },

    MaskKvPage {
        inst_id: InstanceId,
        stream_id: LocalStreamId,
        kv_page: IdRepr,
        mask: Vec<bool>,
    },

    EmbedText {
        inst_id: InstanceId,
        stream_id: LocalStreamId,
        embs: Vec<IdRepr>,
        text: Vec<u32>,
        positions: Vec<u32>,
    },

    ForwardText {
        inst_id: InstanceId,
        stream_id: LocalStreamId,
        kv_page_last_len: u32,
        kv_pages: Vec<IdRepr>,
        text: Vec<u32>,
        positions: Vec<u32>,
        mask: Vec<Vec<u32>>,
        output_indices: Vec<u32>,
        handle: Option<oneshot::Sender<Vec<(Vec<u32>, Vec<f32>)>>>,
    },

    SampleTopK {
        inst_id: InstanceId,
        stream_id: LocalStreamId,
        emb_id: IdRepr,
        k: u32,
        handle: oneshot::Sender<(Vec<u32>, Vec<f32>)>,
    },

    Synchronize {
        inst_id: InstanceId,
        stream_id: LocalStreamId,
        handle: oneshot::Sender<()>,
    },

    SetStreamPriority {
        inst_id: InstanceId,
        stream_id: LocalStreamId,
        priority: StreamPriority,
    },

    //// ------ Vision specific commands ------ ////
    EmbedImage {
        inst_id: InstanceId,
        stream_id: LocalStreamId,
        embs: Vec<IdRepr>,
        image_blob: Vec<u8>,
    },

    DebugQuery {
        inst_id: InstanceId,
        stream_id: LocalStreamId,
        query: String,
        handle: oneshot::Sender<String>,
    },

    /// ---- Optimizer -----
    InitializeAdapter {
        inst_id: InstanceId,
        stream_id: LocalStreamId,
        adapter: IdRepr,
        rank: u32,
        alpha: f32,
        population_size: u32,
        mu_fraction: f32,
        initial_sigma: f32,
    },

    MutateAdapters {
        inst_id: InstanceId,
        stream_id: LocalStreamId,
        adapters: Vec<IdRepr>,
        parent: IdRepr,
        seeds: Vec<i64>,
    },

    UpdateAdapter {
        inst_id: InstanceId,
        stream_id: LocalStreamId,
        adapter: IdRepr,
        scores: Vec<f32>,
        seeds: Vec<i64>,
        max_sigma: f32,
    },

    ForwardWithAdapter {
        inst_id: InstanceId,
        stream_id: LocalStreamId,
        adapter: IdRepr,
        kv_page_last_len: u32,
        kv_pages: Vec<IdRepr>,
        text: Vec<u32>,
        positions: Vec<u32>,
        mask: Vec<Vec<u32>>,
        output_indices: Vec<u32>,
        handle: Option<oneshot::Sender<Vec<(Vec<u32>, Vec<f32>)>>>,
    },
}

impl Command {
    pub fn dispatch(self, service_id: usize) -> Result<(), ServiceError> {
        service::dispatch(service_id, self)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
enum BatchGroup {
    GetInfo,
    Allocate,
    Deallocate,
    FillBlock,
    CopyBlock,
    MaskBlock,
    EmbedText,
    ForwardText,
    SampleTopK,
    Synchronize,
    EmbedImage,
    DebugQuery,
    //
    InitializeAdapter,
    MutateAdapters,
    UpdateAdapter,
    ForwardWithAdapter,
}

impl Batchable<BatchGroup> for Command {
    fn strategy(&self) -> Box<dyn BatchingStrategy> {
        match self {
            Command::GetInfo { .. } => batching::immediate(),
            Command::Allocate { .. } => {
                //batching::t_only(Duration::from_micros(100))
                batching::eager()
            }
            Command::Deallocate { .. } => {
                //batching::t_only(Duration::from_micros(100))
                batching::eager()
            }
            Command::Forward { .. } => {
                //
                //batching::k_or_t(Duration::from_millis(10), 30, None)
                // 7ms, 14ms
                //batching::t_only(Duration::from_millis(14))
                //batching::eager()

                // Box::new(batching::ManualStrategy::new(
                //     TRIGGERS.fill_block_trigger.clone(),
                // ));

                let config = FORWARD_STRATEGY
                    .get()
                    .unwrap_or(&BatchingStrategyConfiguration::Adaptive);
                match *config {
                    BatchingStrategyConfiguration::TOnly { t } => batching::t_only(t),
                    BatchingStrategyConfiguration::KOnly { k } => batching::k_only(k, None),
                    BatchingStrategyConfiguration::KOrT { k, t } => batching::k_or_t(t, k, None),
                    BatchingStrategyConfiguration::Adaptive => Box::new(
                        batching::ManualStrategy::new(TRIGGERS.fill_block_trigger.clone()),
                    ),
                }
            }
            Command::CopyKvPage { .. } => batching::eager(),
            Command::MaskKvPage { .. } => batching::eager(),
            Command::EmbedText { .. } => {
                //batching::t_only(Duration::from_micros(100))
                batching::eager()
            }

            Command::ForwardText { .. } => {
                // Box::new(batching::ManualStrategy::new(
                //     TRIGGERS.forward_text_trigger.clone(),
                // ))

                let config = FORWARD_STRATEGY
                    .get()
                    .unwrap_or(&BatchingStrategyConfiguration::Adaptive);
                match *config {
                    BatchingStrategyConfiguration::TOnly { t } => batching::t_only(t),
                    BatchingStrategyConfiguration::KOnly { k } => batching::k_only(k, None),
                    BatchingStrategyConfiguration::KOrT { k, t } => batching::k_or_t(t, k, None),
                    BatchingStrategyConfiguration::Adaptive => Box::new(
                        batching::ManualStrategy::new(TRIGGERS.forward_text_trigger.clone()),
                    ),
                }

                //batching::eager()
            }

            Command::SampleTopK { .. } => {
                //batching::t_only(Duration::from_micros(100))
                batching::eager()
            }
            Command::Synchronize { .. } => batching::eager(),
            Command::EmbedImage { .. } => batching::eager(),
            Command::DebugQuery { .. } => batching::eager(),
            Command::InitializeAdapter { .. } => batching::eager(),
            Command::MutateAdapters { .. } => batching::eager(),
            Command::UpdateAdapter { .. } => batching::eager(),
            Command::ForwardWithAdapter { .. } => Box::new(batching::ManualStrategy::new(
                TRIGGERS.forward_text_trigger.clone(),
            )),
            _ => unreachable!(),
        }
    }

    fn group(&self) -> BatchGroup {
        match self {
            Command::GetInfo { .. } => BatchGroup::GetInfo,
            Command::Allocate { .. } => BatchGroup::Allocate,
            Command::Deallocate { .. } => BatchGroup::Deallocate,
            Command::Forward { .. } => BatchGroup::FillBlock,
            Command::CopyKvPage { .. } => BatchGroup::CopyBlock,
            Command::MaskKvPage { .. } => BatchGroup::MaskBlock,
            Command::EmbedText { .. } => BatchGroup::EmbedText,
            Command::ForwardText { .. } => BatchGroup::ForwardText,
            Command::SampleTopK { .. } => BatchGroup::SampleTopK,
            Command::Synchronize { .. } => BatchGroup::Synchronize,
            Command::EmbedImage { .. } => BatchGroup::EmbedImage,
            Command::DebugQuery { .. } => BatchGroup::DebugQuery,
            Command::InitializeAdapter { .. } => BatchGroup::InitializeAdapter,
            Command::MutateAdapters { .. } => BatchGroup::MutateAdapters,
            Command::UpdateAdapter { .. } => BatchGroup::UpdateAdapter,
            Command::ForwardWithAdapter { .. } => BatchGroup::ForwardWithAdapter,

            _ => unreachable!(),
        }
    }
}

#[derive(Debug)]
pub enum Event {
    GetInfo(oneshot::Sender<Info>),
    SampleTopK(oneshot::Sender<(Vec<u32>, Vec<f32>)>),
    ForwardText(Option<oneshot::Sender<Vec<(Vec<u32>, Vec<f32>)>>>),
    DebugQuery(oneshot::Sender<String>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ManagedTypes {
    KvPage,
    Embed,
    Adapter,
}

impl ObjectType for ManagedTypes {
    fn is_sharable(&self) -> bool {
        match self {
            ManagedTypes::KvPage => true,
            ManagedTypes::Embed => true,
            ManagedTypes::Adapter => true,
        }
    }

    fn allow_remapping(&self) -> bool {
        match self {
            ManagedTypes::KvPage => false,
            ManagedTypes::Embed => false,
            ManagedTypes::Adapter => false,
        }
    }
}
#[derive(Debug)]
pub struct L4mStat {
    total_calls: u32,
}

#[derive(Debug)]
pub struct L4m {
    scheduler: Sender<(Stream, Command)>,
    scheduler_loop_handle: tokio::task::JoinHandle<()>,
    event_loop_handle: tokio::task::JoinHandle<()>,
    exported: HashMap<ManagedTypes, HashMap<String, Vec<u32>>>,
    global_kv_page_id_pool: IdPool<u32>,
    objects: ObjectManager<InstanceId, ManagedTypes>,
    instance_launch_order: Vec<InstanceId>,
    stream_priorities: HashMap<Stream, StreamPriority>,
    info: Info,
    stats: L4mStat,
}

//#[async_trait]
impl Service for L4m {
    type Command = Command;

    async fn handle(&mut self, cmd: Self::Command) {
        self.stats.total_calls += 1;
        if let Command::Destroy { inst_id } = cmd {
            // Remove the instance from the launch order tracking.
            self.instance_launch_order.retain(|&id| id != inst_id);

            for cmd in self.get_cleanup_cmds(inst_id) {
                self.handle_cmd(cmd).await;
            }
        } else {
            self.handle_cmd(cmd).await;
        }
    }
}

impl L4m {
    pub async fn new<B>(backend: B) -> Self
    where
        B: Backend + 'static,
    {
        let (event_tx, event_rx) = mpsc::channel(1024 * 8);
        let (scheduler_tx, scheduler_rx) = mpsc::channel(1024 * 8);

        backend.register_listener(0, event_tx).await;
        let event_table = Arc::new(DashMap::new());
        let event_loop_handle = tokio::spawn(Self::event_loop(event_rx, event_table.clone()));

        let scheduler_loop_handle =
            tokio::spawn(Self::scheduler_loop(backend, event_table, scheduler_rx));

        let (info_tx, info_rx) = oneshot::channel();

        scheduler_tx
            .send((Stream::default(), Command::GetInfo { handle: info_tx }))
            .await
            .unwrap();

        let info = info_rx.await.unwrap();

        tracing::info!(
            "Backend service started: version={}, model_name={}, kv_page_size={}, num_kv_pages={}, num_embeddings={}, num_adapters={}",
            info.version,
            info.model_name,
            info.kv_page_size,
            info.num_kv_pages,
            info.num_embeddings,
            info.num_adapters
        );

        let mut objects = ObjectManager::new();
        objects
            .set_capacity(ManagedTypes::KvPage, info.num_kv_pages as IdRepr)
            .unwrap();
        objects
            .set_capacity(ManagedTypes::Embed, info.num_embeddings as IdRepr)
            .unwrap();
        objects
            .set_capacity(ManagedTypes::Adapter, info.num_adapters as IdRepr)
            .unwrap();

        let driver = Self {
            scheduler: scheduler_tx,
            scheduler_loop_handle,
            event_loop_handle,
            exported: HashMap::new(),
            global_kv_page_id_pool: IdPool::new(u32::MAX),
            objects,
            instance_launch_order: Vec::new(),
            stream_priorities: HashMap::new(),
            info,
            stats: L4mStat { total_calls: 0 },
        };

        driver
    }

    pub fn print_stats(&self) {
        let mut stats = Vec::new();
        for &managed_type in &[ManagedTypes::KvPage, ManagedTypes::Embed] {
            let current = self.objects.available(managed_type).unwrap();
            let capacity: usize = self.objects.capacity(managed_type).unwrap() as usize;
            let used = capacity - current;
            let percentage = (used as f32 / capacity as f32) * 100.0;

            let type_name = match managed_type {
                ManagedTypes::KvPage => "kvpage",
                ManagedTypes::Embed => "embed",
                ManagedTypes::Adapter => "adapter",
                // _ => "unknown",
            };

            stats.push(format!(
                "{}: {} / {} ({:.2}% used)",
                type_name, used, capacity, percentage
            ));
        }

        stats.push(format!("Total calls: {}", self.stats.total_calls));

        tracing::info!("{}", stats.join(" | "));
    }

    fn get_cleanup_cmds(&mut self, inst_id: InstanceId) -> Vec<Command> {
        let mut cmds = Vec::new();

        for ty in [ManagedTypes::KvPage, ManagedTypes::Embed] {
            if let Ok(ids) = self.objects.all_names(ty, inst_id) {
                cmds.push(Command::Deallocate {
                    inst_id,
                    stream_id: 0,
                    ty,
                    ids,
                });
            }
        }

        cmds
    }

    async fn handle_cmd(&mut self, cmd: Command) {
        match self.resolve_cmd(cmd) {
            // Should be sent to backend
            Some((cmd, mut stream)) => {
                // adjust stream priority
                if let Some(priority) = self.stream_priorities.get(&stream) {
                    stream.set_priority(*priority)
                }

                self.scheduler.send((stream, cmd)).await.unwrap();
            }

            // No need to send to backend
            None => {}
        }
    }

    fn resolve_cmd(&mut self, cmd: Command) -> Option<(Command, Stream)> {
        match cmd {
            Command::Destroy { .. } => {
                unreachable!()
            }

            Command::GetInfo { handle } => Some((Command::GetInfo { handle }, Stream::default())),

            Command::GetBlockSize { handle } => {
                handle.send(self.info.kv_page_size).ok();
                None
            }

            Command::GetTokenizer { handle } => {
                handle.send(self.info.tokenizer.clone()).ok();
                None
            }

            Command::Allocate {
                inst_id,
                stream_id,
                ty,
                ids,
            } => {
                // check available space
                if !self.instance_launch_order.contains(&inst_id) {
                    self.instance_launch_order.push(inst_id);
                }

                // Loop to free resources if the initial check fails.
                // TODO: implement better resource management like CPU page swapping.
                while self.objects.available(ty).unwrap() < ids.len() {
                    let requester_index = self
                        .instance_launch_order
                        .iter()
                        .position(|&id| id == inst_id)
                        .unwrap(); // Requester is guaranteed to be in the list.

                    let victim_to_terminate = self
                        .instance_launch_order
                        .iter()
                        .enumerate()
                        .rev() // Start search from the newest instance.
                        .find(|(index, _id)| *index > requester_index);

                    if let Some((victim_index, &victim_id)) = victim_to_terminate {
                        tracing::warn!(
                            "Resource contention: Instance {:?} is terminating newer instance {:?} to free resources.",
                            inst_id,
                            victim_id
                        );

                        // Deallocate all resources for the victim instance.
                        for resource_type in [ManagedTypes::KvPage, ManagedTypes::Embed] {
                            if let Ok(victim_ids) = self.objects.all_names(resource_type, victim_id)
                            {
                                if !victim_ids.is_empty() {
                                    if let Ok(physical_ids) = self.objects.destroy_many(
                                        resource_type,
                                        victim_id,
                                        &victim_ids,
                                    ) {
                                        let dealloc_cmd = Command::Deallocate {
                                            inst_id: victim_id,
                                            stream_id: 0,
                                            ty: resource_type,
                                            ids: physical_ids,
                                        };
                                        let stream = Stream::new(victim_id, 0);
                                        self.scheduler.try_send((stream, dealloc_cmd)).ok();
                                    }
                                }
                            }
                        }

                        self.instance_launch_order.remove(victim_index);

                        runtime::trap(
                            victim_id,
                            "terminated by the system, due to resource contention",
                        );
                    } else {
                        runtime::trap(
                            inst_id,
                            "l4m::allocation failed. Not enough available space, and no newer instances to terminate.",
                        );
                        return None;
                    }
                }

                let ids = try_trap!(
                    self.objects.create_many(ty, inst_id, ids),
                    inst_id,
                    "l4m::allocation failed"
                );

                Some((
                    Command::Allocate {
                        inst_id,
                        stream_id,
                        ty,
                        ids,
                    },
                    Stream::new(inst_id, stream_id),
                ))
            }

            Command::Deallocate {
                inst_id,
                stream_id,
                ty,
                ids,
            } => {
                // if ty == ManagedTypes::TokenEmb {
                //     println!("deallocating tokenemb, ids: {:?}", ids);
                //     println!("available tokenemb: {:?}", self.objects.available(ty));
                // }

                let ids = try_trap!(
                    self.objects.destroy_many(ty, inst_id, &ids),
                    inst_id,
                    "l4m::deallocation failed"
                );

                // if ty == ManagedTypes::TokenEmb {
                //     println!("available tokenemb after deallocation: {:?}", self.objects.available(ty));
                // }

                if ids.is_empty() {
                    return None;
                }

                Some((
                    Command::Deallocate {
                        inst_id,
                        stream_id,
                        ty,
                        ids,
                    },
                    Stream::new(inst_id, stream_id),
                ))
            }

            Command::Export {
                inst_id,
                ty,
                mut ids,
                resource_name,
            } => {
                // Translate logical page names to physical block addresses.
                try_trap!(
                    self.objects.translate_many(ty, inst_id, &mut ids),
                    inst_id,
                    "export failed. some blocks are invalid"
                );

                // check if it has already expored.
                if self
                    .exported
                    .entry(ty.clone())
                    .or_default()
                    .contains_key(&resource_name)
                {
                    trap(inst_id, "Already exported");
                    return None;
                }

                // prevent from deallocation
                self.objects.inc_ref_count_many(ty, &ids);

                self.exported
                    .get_mut(&ty)
                    .unwrap()
                    .insert(resource_name, ids);

                None
            }

            Command::GetExportedList {
                inst_id: _, // Not used by this command.
                ty,
                handle,
            } => {
                // Collect the list of exported resources of a given type.
                // The list contains the resource name and the number of items it holds.
                let list = self
                    .exported
                    .get(&ty)
                    .map(|exports| {
                        exports
                            .iter()
                            .map(|(name, ids)| (name.clone(), ids.len() as u32))
                            .collect()
                    })
                    .unwrap_or_default();
                handle.send(list).ok();
                None
            }
            Command::Unexport {
                inst_id,
                ty,
                resource_name,
            } => {
                // Find and remove the resource from the exported list.
                if let Some(physical_ids) = self
                    .exported
                    .get_mut(&ty)
                    .and_then(|exports| exports.remove(&resource_name))
                {
                    // Decrement the reference count that was incremented during `Export`.
                    // This allows the underlying objects to be deallocated if no other
                    // instances are referencing them.
                    self.objects.dec_ref_count_many(ty, &physical_ids);
                } else {
                    runtime::trap(
                        inst_id,
                        &format!("Unexport failed: resource '{}' not found.", resource_name),
                    );
                }
                None
            }
            Command::Import {
                inst_id,
                ty,
                ids,
                resource_name,
            } => {
                // Find the physical IDs associated with the exported resource name.
                match self.exported.get(&ty).and_then(|m| m.get(&resource_name)) {
                    Some(physical_ids) => {
                        // Ensure the user provides a matching number of logical IDs.
                        if ids.len() != physical_ids.len() {
                            runtime::trap(
                                inst_id,
                                &format!(
                                    "Import failed: ID count mismatch (provided {}, required {}).",
                                    ids.len(),
                                    physical_ids.len()
                                ),
                            );
                            return None;
                        }
                        // Create a new reference for the current instance, mapping its logical IDs
                        // to the existing physical objects.
                        try_trap!(
                            self.objects.create_ref_many(ty, inst_id, ids, physical_ids),
                            inst_id,
                            "l4m::import failed"
                        );
                    }
                    None => {
                        runtime::trap(
                            inst_id,
                            &format!("Import failed: resource '{}' not found.", resource_name),
                        );
                    }
                }
                None
            }

            Command::Forward {
                inst_id,
                stream_id,
                kv_page_last_len,
                mut kv_pages,
                mut input_embeds,
                mut output_embeds,
            } => {
                if kv_page_last_len == 0 || kv_page_last_len > self.info.kv_page_size {
                    // error
                    runtime::trap(
                        inst_id,
                        format!(
                            "forward failed. kv_page_last_len ({}) is 0 or greater than the page size ({})",
                            kv_page_last_len, self.info.kv_page_size
                        ),
                    );
                    return None;
                }

                let max_tokens =
                    self.info.kv_page_size * (kv_pages.len() as u32 - 1) + kv_page_last_len;

                if input_embeds.len() > max_tokens as usize {
                    // error
                    runtime::trap(
                        inst_id,
                        format!(
                            "l4m::fill_block failed. inputs length is greater than the max tokens: {} > {}",
                            input_embeds.len(),
                            max_tokens
                        ),
                    );
                    return None;
                }

                try_trap!(
                    self.objects
                        .translate_many(ManagedTypes::KvPage, inst_id, &mut kv_pages),
                    inst_id,
                    "l4m::fill_block failed. some context blocks are invalid"
                );
                try_trap!(
                    self.objects
                        .translate_many(ManagedTypes::Embed, inst_id, &mut input_embeds),
                    inst_id,
                    "l4m::fill_block failed. some input embeddings are invalid"
                );
                try_trap!(
                    self.objects
                        .translate_many(ManagedTypes::Embed, inst_id, &mut output_embeds),
                    inst_id,
                    "l4m::fill_block failed. some output embeddings are invalid"
                );

                Some((
                    Command::Forward {
                        inst_id,
                        stream_id,
                        kv_page_last_len,
                        kv_pages,
                        input_embeds,
                        output_embeds,
                    },
                    Stream::new(inst_id, stream_id),
                ))
            }

            Command::ForwardText {
                inst_id,
                stream_id,
                kv_page_last_len,
                mut kv_pages,
                text,
                positions,
                mask,
                output_indices,
                handle,
            } => {
                try_trap!(
                    self.objects
                        .translate_many(ManagedTypes::KvPage, inst_id, &mut kv_pages),
                    inst_id,
                    "l4m::fill_block failed. some context blocks are invalid"
                );

                Some((
                    Command::ForwardText {
                        inst_id,
                        stream_id,
                        kv_page_last_len,
                        kv_pages,
                        text,
                        positions,
                        mask,
                        output_indices,
                        handle,
                    },
                    Stream::new(inst_id, stream_id),
                ))
            }

            Command::CopyKvPage {
                inst_id,
                stream_id,
                src_kv_page: mut src_block,
                dst_kv_page: mut dst_block,
                src_token_offset,
                dst_token_offset,
                size,
            } => {
                try_trap!(
                    self.objects
                        .translate(ManagedTypes::KvPage, inst_id, &mut src_block),
                    inst_id,
                    "l4m::copy_block failed. invalid source block"
                );
                try_trap!(
                    self.objects
                        .translate(ManagedTypes::KvPage, inst_id, &mut dst_block),
                    inst_id,
                    "l4m::copy_block failed. invalid destination block"
                );

                Some((
                    Command::CopyKvPage {
                        inst_id,
                        stream_id,
                        src_kv_page: src_block,
                        dst_kv_page: dst_block,
                        src_token_offset,
                        dst_token_offset,
                        size,
                    },
                    Stream::new(inst_id, stream_id),
                ))
            }

            Command::MaskKvPage {
                inst_id,
                stream_id,
                kv_page: mut block,
                mask,
            } => {
                try_trap!(
                    self.objects
                        .translate(ManagedTypes::KvPage, inst_id, &mut block),
                    inst_id,
                    "l4m::mask_block failed. invalid block"
                );

                Some((
                    Command::MaskKvPage {
                        inst_id,
                        stream_id,
                        kv_page: block,
                        mask,
                    },
                    Stream::new(inst_id, stream_id),
                ))
            }

            Command::EmbedText {
                inst_id,
                stream_id,
                mut embs,
                text,
                positions,
            } => {
                try_trap!(
                    self.objects
                        .translate_many(ManagedTypes::Embed, inst_id, &mut embs),
                    inst_id,
                    "l4m::embed_text failed. invalid embeddings"
                );

                Some((
                    Command::EmbedText {
                        inst_id,
                        stream_id,
                        embs,
                        text,
                        positions,
                    },
                    Stream::new(inst_id, stream_id),
                ))
            }

            Command::SampleTopK {
                inst_id,
                stream_id,
                mut emb_id,
                k,
                handle,
            } => {
                try_trap!(
                    self.objects
                        .translate(ManagedTypes::Embed, inst_id, &mut emb_id),
                    inst_id,
                    "l4m::sample_topk failed. invalid distribution"
                );

                Some((
                    Command::SampleTopK {
                        inst_id,
                        stream_id,
                        emb_id,
                        k,
                        handle,
                    },
                    Stream::new(inst_id, stream_id),
                ))
            }

            Command::Synchronize {
                inst_id,
                stream_id,
                handle,
            } => Some((
                Command::Synchronize {
                    inst_id,
                    stream_id,
                    handle,
                },
                Stream::new(inst_id, stream_id),
            )),

            Command::SetStreamPriority {
                inst_id,
                stream_id,
                priority,
            } => {
                self.stream_priorities
                    .insert(Stream::new(inst_id, stream_id), priority);
                None
            }

            Command::EmbedImage {
                inst_id,
                stream_id,
                mut embs,
                image_blob,
            } => {
                try_trap!(
                    self.objects
                        .translate_many(ManagedTypes::Embed, inst_id, &mut embs),
                    inst_id,
                    "l4m::embed_image failed. invalid embeddings"
                );

                Some((
                    Command::EmbedImage {
                        inst_id,
                        stream_id,
                        embs,
                        image_blob,
                    },
                    Stream::new(inst_id, stream_id),
                ))
            }

            Command::DebugQuery {
                inst_id,
                stream_id,
                query,
                handle,
            } => {
                // For debug queries, we simply send the query to the backend.
                // The backend will handle it and return a response.
                Some((
                    Command::DebugQuery {
                        inst_id,
                        stream_id,
                        query,
                        handle,
                    },
                    Stream::new(inst_id, stream_id),
                ))
            }

            Command::InitializeAdapter {
                inst_id,
                stream_id,
                mut adapter,
                rank,
                alpha,
                population_size,
                mu_fraction,
                initial_sigma,
            } => {
                try_trap!(
                    self.objects
                        .translate(ManagedTypes::Adapter, inst_id, &mut adapter),
                    inst_id,
                    "InitializeAdapter failed. some context blocks are invalid"
                );
                Some((
                    Command::InitializeAdapter {
                        inst_id,
                        stream_id,
                        adapter,
                        rank,
                        alpha,
                        population_size,
                        mu_fraction,
                        initial_sigma,
                    },
                    Stream::new(inst_id, stream_id),
                ))
            }

            Command::MutateAdapters {
                inst_id,
                stream_id,
                mut adapters,
                mut parent,
                seeds,
            } => {
                try_trap!(
                    self.objects
                        .translate_many(ManagedTypes::Adapter, inst_id, &mut adapters),
                    inst_id,
                    "MutateAdapters failed. some context blocks are invalid"
                );

                try_trap!(
                    self.objects
                        .translate(ManagedTypes::Adapter, inst_id, &mut parent),
                    inst_id,
                    "InitializeAdapter failed. some context blocks are invalid"
                );

                Some((
                    Command::MutateAdapters {
                        inst_id,
                        stream_id,
                        adapters,
                        parent,
                        seeds,
                    },
                    Stream::new(inst_id, stream_id),
                ))
            }

            Command::UpdateAdapter {
                inst_id,
                stream_id,
                mut adapter,
                scores,
                seeds,
                max_sigma,
            } => {
                try_trap!(
                    self.objects
                        .translate(ManagedTypes::Adapter, inst_id, &mut adapter),
                    inst_id,
                    "UpdateAdapter failed. some context blocks are invalid"
                );
                Some((
                    Command::UpdateAdapter {
                        inst_id,
                        stream_id,
                        adapter,
                        scores,
                        seeds,
                        max_sigma,
                    },
                    Stream::new(inst_id, stream_id),
                ))
            }

            Command::ForwardWithAdapter {
                inst_id,
                stream_id,
                mut adapter,
                kv_page_last_len,
                mut kv_pages,
                text,
                positions,
                mask,
                output_indices,
                handle,
            } => {
                try_trap!(
                    self.objects
                        .translate(ManagedTypes::Adapter, inst_id, &mut adapter),
                    inst_id,
                    "ForwardWithAdapter failed. some context blocks are invalid"
                );
                try_trap!(
                    self.objects
                        .translate_many(ManagedTypes::KvPage, inst_id, &mut kv_pages),
                    inst_id,
                    "l4m::fill_block failed. some context blocks are invalid"
                );

                Some((
                    Command::ForwardWithAdapter {
                        inst_id,
                        stream_id,
                        adapter,
                        kv_page_last_len,
                        kv_pages,
                        text,
                        positions,
                        mask,
                        output_indices,
                        handle,
                    },
                    Stream::new(inst_id, stream_id),
                ))
            }
        }
    }

    async fn scheduler_loop<B>(
        backend: B,
        event_table: Arc<DashMap<u32, Vec<Event>>>,
        mut rx: Receiver<(Stream, Command)>,
    ) where
        B: Backend,
    {
        let mut sch = CommandScheduler::new(backend, event_table);

        loop {
            let res: Result<Option<(Stream, Command)>, tokio::time::error::Elapsed> =
                if sch.has_pending_command() {
                    //println!("scheduler has pending command");
                    // With pending tasks, wait up to 100µs for a new command.
                    timeout(Duration::from_micros(100), rx.recv()).await
                } else {
                    // Without pending tasks, wait indefinitely.
                    Ok(rx.recv().await)
                };

            match res {
                Ok(Some((stream, cmd))) => {
                    sch.submit(stream, cmd, Instant::now());
                    sch.update(Instant::now()).await;
                }
                Ok(None) => break, // The channel closed.
                Err(_) => {
                    // Timeout. Still need to flush pending commands.
                    sch.update(Instant::now()).await;
                }
            }
        }
    }

    async fn event_loop(mut rx: Receiver<Vec<u8>>, event_table: Arc<DashMap<u32, Vec<Event>>>) {
        while let Some(resp) = rx.recv().await {
            let resp = pb_bindings::Response::decode(resp.as_ref()).unwrap();

            let correlation_id = resp.correlation_id;
            let payload = resp.command.unwrap();

            // Handle BatchSync separately, as it's a general signal.
            if let pb_bindings::response::Command::BatchSync(..) = &payload {
                // Set the trigger to true. The CommandScheduler will pick this up
                // on its next update and fire a batch of FillBlock commands.
                TRIGGERS
                    .fill_block_trigger
                    .store(true, std::sync::atomic::Ordering::SeqCst);
                //println!("BatchSync triggered");
            }

            if let Some((_, senders)) = event_table.remove(&correlation_id) {
                match payload {
                    pb_bindings::response::Command::SampleTopK(batch) => {
                        for (item, event) in batch.items.into_iter().zip(senders) {
                            match event {
                                Event::SampleTopK(handle) => {
                                    handle.send((item.token_ids, item.probabilities)).ok();
                                }
                                _ => unreachable!(),
                            }
                        }
                    }

                    pb_bindings::response::Command::GetInfo(info) => {
                        let sender = senders.into_iter().next().unwrap();
                        match sender {
                            Event::GetInfo(handle) => {
                                let tokenizer = info.tokenizer.unwrap();
                                let merge_table = tokenizer.merge_table;
                                let special_tokens = tokenizer.special_tokens;
                                let pattern = tokenizer.split_regex;
                                let escape_non_printable = tokenizer.escape_non_printable;

                                let tokenizer = Arc::new(BytePairEncoder::new(
                                    merge_table,
                                    special_tokens,
                                    &pattern,
                                    escape_non_printable,
                                ));

                                handle
                                    .send(Info {
                                        version: info.version,
                                        model_name: info.model_name,
                                        kv_page_size: info.kv_page_size,
                                        num_kv_pages: info.num_available_kv_pages,
                                        num_embeddings: info.num_available_embeddings,
                                        num_adapters: info.num_available_adapters,
                                        tokenizer,
                                    })
                                    .ok();
                            }
                            _ => unreachable!(),
                        }
                    }

                    pb_bindings::response::Command::ForwardText(batch) => {
                        TRIGGERS
                            .forward_text_trigger
                            .store(true, std::sync::atomic::Ordering::SeqCst);

                        for (item, event) in batch.items.into_iter().zip(senders) {
                            let mut distribs = Vec::new();

                            for d in item.distributions {
                                distribs.push((d.ids, d.probs));
                            }

                            match event {
                                Event::ForwardText(handle) => {
                                    if let Some(h) = handle {
                                        h.send(distribs).ok();
                                    }
                                }
                                _ => unreachable!(),
                            }
                        }
                    }

                    pb_bindings::response::Command::BatchSync(..) => {
                        // fire the next batch, or set the ready flag to true.
                    }

                    pb_bindings::response::Command::DebugQuery(batch) => {
                        for (item, event) in batch.items.into_iter().zip(senders) {
                            match event {
                                Event::DebugQuery(handle) => {
                                    handle.send(item.response).ok();
                                }
                                _ => unreachable!(),
                            }
                        }
                    }
                }
            }
        }
    }
}

struct CommandScheduler<B> {
    backend: B,

    protocol_ids: Vec<u8>,

    cmd_id_pool: IdPool<u32>,
    cmd_batcher: Batcher<Command, Stream, BatchGroup>,

    event_table: Arc<DashMap<u32, Vec<Event>>>,
}

impl<B> CommandScheduler<B>
where
    B: Backend + 'static,
{
    fn new(backend: B, event_table: Arc<DashMap<u32, Vec<Event>>>) -> Self {
        let protocol_ids = backend
            .supported_protocols()
            .iter()
            .map(|protoc| {
                backend.protocol_index(protoc).expect(&format!(
                    "Failed to get protocol index: UnsupportedProtocol(\"{}\")",
                    protoc
                ))
            })
            .collect::<Vec<u8>>();

        Self {
            backend,
            protocol_ids,
            cmd_id_pool: IdPool::new(u32::MAX),
            cmd_batcher: Batcher::new(),
            event_table,
        }
    }

    fn submit(&mut self, stream: Stream, cmd: Command, now: Instant) {
        self.cmd_batcher.push(stream, cmd, now);
    }

    fn has_pending_command(&self) -> bool {
        self.cmd_batcher.has_pending_items()
    }

    async fn update(&mut self, now: Instant) {
        //println!("time: {:?}", now);
        for (_, cmd_batch) in self.cmd_batcher.batch(now) {
            self.flush(cmd_batch).await;
        }
    }

    async fn flush(&mut self, batch: Vec<Command>) {
        let batch_type = batch.first().unwrap().group();
        let correlation_id = self.cmd_id_pool.acquire().unwrap();

        let ((protocol, payload), event) = match batch_type.clone() {
            BatchGroup::GetInfo => encode_pb_get_info(correlation_id, batch),
            BatchGroup::Allocate => encode_pb_batch_allocate(correlation_id, batch),
            BatchGroup::Deallocate => encode_pb_batch_deallocate(correlation_id, batch),
            BatchGroup::FillBlock => encode_pb_batch_fill_block(correlation_id, batch),
            BatchGroup::ForwardText => encode_pb_batch_forward_text(correlation_id, batch),
            BatchGroup::CopyBlock => encode_pb_batch_copy_block(correlation_id, batch),
            BatchGroup::MaskBlock => encode_pb_batch_mask_block(correlation_id, batch),
            BatchGroup::EmbedText => encode_pb_batch_embed_text(correlation_id, batch),
            BatchGroup::SampleTopK => encode_pb_batch_sample_topk(correlation_id, batch),
            BatchGroup::Synchronize => {
                let cmd = batch.into_iter().next().unwrap();
                match cmd {
                    Command::Synchronize {
                        inst_id: _,
                        stream_id: _,
                        handle,
                    } => {
                        handle.send(()).unwrap();
                    }
                    _ => unreachable!(),
                }
                return;
            }
            BatchGroup::EmbedImage => encode_pb_batch_embed_image(correlation_id, batch),
            BatchGroup::DebugQuery => encode_pb_batch_debug_query(correlation_id, batch),
            BatchGroup::InitializeAdapter => {
                encode_pb_batch_initialize_adapter(correlation_id, batch)
            }
            BatchGroup::MutateAdapters => encode_pb_batch_mutate_adapters(correlation_id, batch),
            BatchGroup::UpdateAdapter => encode_pb_batch_update_adapter(correlation_id, batch),
            BatchGroup::ForwardWithAdapter => {
                encode_pb_batch_forward_with_adapter(correlation_id, batch)
            } // _ => unreachable!(),
        };

        if let Some(events) = event {
            self.event_table.insert(correlation_id, events);
        }

        self.backend
            .send(self.protocol_ids[protocol], payload)
            .unwrap();
    }
}

#[derive(Clone)]
pub struct Simulator {
    protocols: Vec<String>,
    tokenizer_merge_table: HashMap<u32, Vec<u8>>,
}

impl Simulator {
    pub fn new() -> Self {
        let tokenizer_merge_table =
            load_merge_rules("asset/model-test.vocab").expect("Failed to load tokenizer vocab");

        Self {
            protocols: vec!["l4m".to_string()],
            tokenizer_merge_table,
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
                    model_name: "test-model".to_string(),
                    kv_page_size: 128,
                    num_available_kv_pages: 1000000,
                    num_available_embeddings: 1000000,
                    num_available_adapters: 100000,
                    tokenizer: Some(pb_bindings::Tokenizer {
                        merge_table: self.tokenizer_merge_table.clone(),
                        special_tokens: HashMap::from([
                            ("<|begin_of_text|>".to_string(), 128000),
                            ("<|end_of_text|>".to_string(), 128001),
                            ("<|start_header_id|>".to_string(), 128006),
                            ("<|end_header_id|>".to_string(), 128007),
                            ("<|eot_id|>".to_string(), 128009)
                        ]),
                        split_regex: r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+".to_string(),
                        escape_non_printable: false,
                    }),
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
            Command::Allocate {
                inst_id: _,
                stream_id: _,
                ty,
                ids,
            }
            | Command::Deallocate {
                inst_id: _,
                stream_id: _,
                ty,
                ids,
            } => {
                let kind = match ty {
                    ManagedTypes::KvPage => pb_bindings::ObjectKind::KvBlock,
                    ManagedTypes::Embed => pb_bindings::ObjectKind::Emb,
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

fn encode_pb_batch_deallocate_inner(batch: Vec<Command>) -> Vec<pb_bindings::Deallocate> {
    let mut items = Vec::new();
    for cmd in batch {
        match cmd {
            Command::Allocate {
                inst_id: _,
                stream_id: _,
                ty,
                ids,
            }
            | Command::Deallocate {
                inst_id: _,
                stream_id: _,
                ty,
                ids,
            } => {
                let kind = match ty {
                    ManagedTypes::KvPage => pb_bindings::ObjectKind::KvBlock,
                    ManagedTypes::Embed => pb_bindings::ObjectKind::Emb,
                    ManagedTypes::Adapter => pb_bindings::ObjectKind::Adapter,
                    _ => unreachable!(),
                }
                .into();

                for (offset, size) in group_consecutive_ids(&ids) {
                    let pb = pb_bindings::Deallocate {
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
                items: encode_pb_batch_deallocate_inner(batch),
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
            Command::Forward {
                inst_id: _,
                stream_id: _,
                kv_page_last_len: last_block_len,
                kv_pages: context,
                input_embeds: inputs,
                output_embeds: outputs,
            } => {
                let pb = pb_bindings::FillBlock {
                    last_block_len: last_block_len,
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

fn encode_pb_batch_forward_text(
    correlation_id: u32,
    batch: Vec<Command>,
) -> ((usize, Vec<u8>), Option<Vec<Event>>) {
    let mut items = Vec::new();
    let mut events = Vec::new();
    for cmd in batch {
        match cmd {
            Command::ForwardText {
                inst_id: _,
                stream_id: _,
                kv_page_last_len,
                kv_pages: kv_page_ids,
                text,
                positions,
                mask,
                output_indices,
                handle,
            } => {
                let mask = mask
                    .into_iter()
                    .map(|b| pb_bindings::BrleBuffer { buffer: b })
                    .collect();

                let pb = pb_bindings::ForwardText {
                    kv_page_ids,
                    kv_page_last_len,
                    token_ids: text,
                    position_ids: positions,
                    mask,
                    output_indices,
                };
                items.push(pb);
                events.push(Event::ForwardText(handle));
            }
            _ => unreachable!(),
        }
    }
    let cmd = pb_bindings::request::Command::ForwardText(pb_bindings::BatchForwardText { items });
    let payload = pb_bindings::Request {
        correlation_id,
        command: Some(cmd),
    }
    .encode_to_vec();
    ((PROTOCOL_BASE, payload), Some(events))
}

fn encode_pb_batch_copy_block(
    correlation_id: u32,
    batch: Vec<Command>,
) -> ((usize, Vec<u8>), Option<Vec<Event>>) {
    let mut items = Vec::new();
    for cmd in batch {
        match cmd {
            Command::CopyKvPage {
                inst_id: _,
                stream_id: _,
                src_kv_page: src_block,
                dst_kv_page: dst_block,
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
            Command::MaskKvPage {
                inst_id: _,
                stream_id: _,
                kv_page: block,
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
                inst_id: _,
                stream_id: _,
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

fn encode_pb_batch_sample_topk(
    correlation_id: u32,
    batch: Vec<Command>,
) -> ((usize, Vec<u8>), Option<Vec<Event>>) {
    let mut items = Vec::new();
    let mut events = Vec::new();
    for cmd in batch {
        match cmd {
            Command::SampleTopK {
                inst_id: _,
                stream_id: _,
                emb_id: dist,
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
                inst_id: _,
                stream_id: _,
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

fn encode_pb_batch_debug_query(
    correlation_id: u32,
    batch: Vec<Command>,
) -> ((usize, Vec<u8>), Option<Vec<Event>>) {
    let mut items = Vec::new();
    let mut events = Vec::new();

    for cmd in batch {
        match cmd {
            Command::DebugQuery {
                inst_id: _,
                stream_id: _,
                query,
                handle,
            } => {
                let pb = pb_bindings::DebugQueryRequest { query };
                items.push(pb);
                events.push(Event::DebugQuery(handle));
            }
            _ => unreachable!(),
        }
    }
    let cmd =
        pb_bindings::request::Command::DebugQueryRequest(pb_bindings::BatchDebugQueryRequest {
            items,
        });
    let payload = pb_bindings::Request {
        correlation_id,
        command: Some(cmd),
    }
    .encode_to_vec();
    ((PROTOCOL_BASE, payload), Some(events))
}

fn encode_pb_batch_initialize_adapter(
    correlation_id: u32,
    batch: Vec<Command>,
) -> ((usize, Vec<u8>), Option<Vec<Event>>) {
    let cmd = match batch.into_iter().next().unwrap() {
        Command::InitializeAdapter {
            inst_id,
            stream_id,
            adapter,
            rank,
            alpha,
            population_size,
            mu_fraction,
            initial_sigma,
        } => pb_bindings::request::Command::InitializeAdapter(pb_bindings::InitializeAdapter {
            adapter,
            rank,
            alpha,
            population_size,
            mu_fraction,
            initial_sigma,
        }),
        _ => unreachable!(),
    };

    let payload = pb_bindings::Request {
        correlation_id,
        command: Some(cmd),
    }
    .encode_to_vec();
    ((PROTOCOL_BASE, payload), None)
}

fn encode_pb_batch_mutate_adapters(
    correlation_id: u32,
    batch: Vec<Command>,
) -> ((usize, Vec<u8>), Option<Vec<Event>>) {
    let cmd = match batch.into_iter().next().unwrap() {
        Command::MutateAdapters {
            inst_id,
            stream_id,
            adapters,
            parent,
            seeds,
        } => pb_bindings::request::Command::MutateAdapters(pb_bindings::MutateAdapters {
            adapters,
            parent,
            seeds,
        }),
        _ => unreachable!(),
    };

    let payload = pb_bindings::Request {
        correlation_id,
        command: Some(cmd),
    }
    .encode_to_vec();
    ((PROTOCOL_BASE, payload), None)
}

fn encode_pb_batch_update_adapter(
    correlation_id: u32,
    batch: Vec<Command>,
) -> ((usize, Vec<u8>), Option<Vec<Event>>) {
    let cmd = match batch.into_iter().next().unwrap() {
        Command::UpdateAdapter {
            inst_id,
            stream_id,
            adapter,
            scores,
            seeds,
            max_sigma,
        } => pb_bindings::request::Command::UpdateAdapter(pb_bindings::UpdateAdapter {
            adapter,
            scores,
            seeds,
            max_sigma,
        }),
        _ => unreachable!(),
    };

    let payload = pb_bindings::Request {
        correlation_id,
        command: Some(cmd),
    }
    .encode_to_vec();
    ((PROTOCOL_BASE, payload), None)
}

fn encode_pb_batch_forward_with_adapter(
    correlation_id: u32,
    batch: Vec<Command>,
) -> ((usize, Vec<u8>), Option<Vec<Event>>) {
    let mut items = Vec::new();
    let mut events = Vec::new();
    for cmd in batch {
        match cmd {
            Command::ForwardWithAdapter {
                inst_id: _,
                stream_id: _,
                adapter,
                kv_page_last_len,
                kv_pages: kv_page_ids,
                text,
                positions,
                mask,
                output_indices,
                handle,
            } => {
                let mask = mask
                    .into_iter()
                    .map(|b| pb_bindings::BrleBuffer { buffer: b })
                    .collect();

                let pb = pb_bindings::ForwardWithAdapter {
                    adapter,
                    kv_page_ids,
                    kv_page_last_len,
                    token_ids: text,
                    position_ids: positions,
                    mask,
                    output_indices,
                };
                items.push(pb);
                events.push(Event::ForwardText(handle));
            }
            _ => unreachable!(),
        }
    }
    let cmd =
        pb_bindings::request::Command::ForwardWithAdapter(pb_bindings::BatchForwardWithAdapter {
            items,
        });
    let payload = pb_bindings::Request {
        correlation_id,
        command: Some(cmd),
    }
    .encode_to_vec();
    ((PROTOCOL_BASE, payload), Some(events))
}

fn encode_pb_get_info(
    correlation_id: u32,
    cmd: Vec<Command>,
) -> ((usize, Vec<u8>), Option<Vec<Event>>) {
    let cmd = cmd.into_iter().next().unwrap();

    match cmd {
        Command::GetInfo { handle } => {
            let cmd = pb_bindings::Request {
                correlation_id,
                command: Some(pb_bindings::request::Command::GetInfo(
                    pb_bindings::GetInfoRequest {},
                )),
            }
            .encode_to_vec();
            ((PROTOCOL_BASE, cmd), Some(vec![Event::GetInfo(handle)]))
        }
        _ => unreachable!(),
    }
}
