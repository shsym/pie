use crate::backend::Backend;
use crate::batching::{Batchable, Batcher, BatchingStrategy};
use crate::instance::Id as InstanceId;
use crate::object::{IdRepr, ObjectManager, ObjectType, group_consecutive_ids};
use crate::service::{Service, ServiceError};
use crate::tokenizer::BytePairEncoder;
use crate::utils::IdPool;
use crate::{backend, batching, runtime, service, tokenizer};
use dashmap::DashMap;
use prost::Message;
use rand::Rng;
use std::cmp::{Ordering, PartialEq};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};
use tokio::sync::mpsc::{Receiver, Sender};
use tokio::sync::{mpsc, oneshot};
use tokio::task;
use tokio::time::timeout;
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

static AVAILABLE_MODELS: OnceLock<Vec<String>> = OnceLock::new();

pub fn set_available_models<T>(models: T)
where
    T: IntoIterator,
    T::Item: ToString,
{
    let models = models
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<String>>();
    AVAILABLE_MODELS.get_or_init(|| models);
}

pub fn available_models() -> &'static [String] {
    AVAILABLE_MODELS.get().unwrap()
}

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
    PrintStats,

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

    GetAllExportedBlocks {
        handle: oneshot::Sender<Vec<(String, IdRepr)>>,
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

    FillBlock {
        inst_id: InstanceId,
        stream_id: LocalStreamId,
        last_block_len: u32,
        context: Vec<IdRepr>,
        inputs: Vec<IdRepr>,
        outputs: Vec<IdRepr>,
    },

    ExportBlocks {
        inst_id: InstanceId,
        blocks: Vec<IdRepr>,
        resource_name: String,
    },

    ImportBlocks {
        inst_id: InstanceId,
        blocks: Vec<IdRepr>,
        resource_name: String,
    },

    CopyBlock {
        inst_id: InstanceId,
        stream_id: LocalStreamId,
        src_block: IdRepr,
        dst_block: IdRepr,
        src_token_offset: u32,
        dst_token_offset: u32,
        size: u32,
    },

    MaskBlock {
        inst_id: InstanceId,
        stream_id: LocalStreamId,
        block: IdRepr,
        mask: Vec<bool>,
    },

    EmbedText {
        inst_id: InstanceId,
        stream_id: LocalStreamId,
        embs: Vec<IdRepr>,
        text: Vec<u32>,
        positions: Vec<u32>,
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
    SampleTopK,
    Synchronize,
    EmbedImage,
}

impl Batchable<BatchGroup> for Command {
    fn strategy(&self) -> Box<dyn BatchingStrategy> {
        match self {
            Command::GetInfo { .. } => batching::immediate(),
            Command::Allocate { .. } => {
                //batching::t_only(Duration::from_micros(100))
                batching::eager()
            },
            Command::Deallocate { .. } => {
                //batching::t_only(Duration::from_micros(100))
                batching::eager()
            },
            Command::FillBlock { .. } => {
                //
                //batching::k_or_t(Duration::from_millis(10), 30, None)
                // 7ms, 14ms
                batching::t_only(Duration::from_millis(14))
                //batching::eager()
            }
            Command::CopyBlock { .. } => batching::eager(),
            Command::MaskBlock { .. } => batching::eager(),
            Command::EmbedText { .. } => {
                //batching::t_only(Duration::from_micros(100))
                batching::eager()
            },

            Command::SampleTopK { .. } => {
                //batching::t_only(Duration::from_micros(100))
                batching::eager()
            },
            Command::Synchronize { .. } => batching::eager(),
            Command::EmbedImage { .. } => batching::eager(),
            _ => unreachable!(),
        }
    }

    fn group(&self) -> BatchGroup {
        match self {
            Command::GetInfo { .. } => BatchGroup::GetInfo,
            Command::Allocate { .. } => BatchGroup::Allocate,
            Command::Deallocate { .. } => BatchGroup::Deallocate,
            Command::FillBlock { .. } => BatchGroup::FillBlock,
            Command::CopyBlock { .. } => BatchGroup::CopyBlock,
            Command::MaskBlock { .. } => BatchGroup::MaskBlock,
            Command::EmbedText { .. } => BatchGroup::EmbedText,
            Command::SampleTopK { .. } => BatchGroup::SampleTopK,
            Command::Synchronize { .. } => BatchGroup::Synchronize,
            Command::EmbedImage { .. } => BatchGroup::EmbedImage,
            _ => unreachable!(),
        }
    }
}

#[derive(Debug)]
pub enum Event {
    GetInfo(oneshot::Sender<Info>),
    SampleTopK(oneshot::Sender<(Vec<u32>, Vec<f32>)>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ManagedTypes {
    KvBlock,
    TokenEmb,
}

impl ObjectType for ManagedTypes {
    fn is_sharable(&self) -> bool {
        match self {
            ManagedTypes::KvBlock => true,
            ManagedTypes::TokenEmb => false,
        }
    }

    fn allow_remapping(&self) -> bool {
        match self {
            ManagedTypes::KvBlock => false,
            ManagedTypes::TokenEmb => true,
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
    scheduler_loop_handle: task::JoinHandle<()>,
    event_loop_handle: task::JoinHandle<()>,
    exported_blocks: HashMap<String, ExportedBlocks>,
    objects: ObjectManager<InstanceId, ManagedTypes>,
    stream_priorities: HashMap<Stream, StreamPriority>,
    info: Info,
    tokenizer: Arc<BytePairEncoder>,
    stats: L4mStat,
}

//#[async_trait]
impl Service for L4m {
    type Command = Command;

    async fn handle(&mut self, cmd: Self::Command) {


        self.stats.total_calls += 1;
        if let Command::Destroy { inst_id } = cmd {
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

        println!(
            "The backend info: version={}, model_name={}, block_size={}, num_blocks={}, num_embeddings={}, num_distributions={}",
            info.version,
            info.model_name,
            info.block_size,
            info.num_blocks,
            info.num_embeddings,
            info.num_distributions
        );

        // Add the model name to the available models
        set_available_models([info.model_name.clone()]);

        // Attempt to load tokenizer
        use std::fs;
        use std::path::{Path, PathBuf};
        use serde_json::Value;

        let home = std::env::var("HOME").unwrap_or_default();
        let cache_base = PathBuf::from(&home)
            .join(".cache").join("symphony").join("models");
        let mut tokenizer = None;

        let model_path = PathBuf::from(&info.model_name);

        if model_path.is_absolute() && model_path.is_dir() {
            println!("Model name is an absolute path: {:?}", model_path);
            // Attempt 1: Load from symphony_model_info.json in the model_path
            let model_info_json_path = model_path.join("symphony_model_info.json");
            if model_info_json_path.exists() {
                println!("Attempting to load tokenizer from symphony_model_info.json: {:?}", model_info_json_path);
                if let Ok(content) = fs::read_to_string(&model_info_json_path) {
                    if let Ok(json) = serde_json::from_str::<Value>(&content) {
                        if let Some(tok_str) = json.get("tokenizer_path").and_then(Value::as_str) {
                            // tok_str could be relative to the symphony_model_info.json or absolute
                            let mut absolute_tokenizer_path = PathBuf::from(tok_str);
                            if !absolute_tokenizer_path.is_absolute() {
                                absolute_tokenizer_path = model_path.join(tok_str);
                            }
                            println!("Loading tokenizer from metadata path: {:?}", absolute_tokenizer_path);
                            if let Ok(tok) = tokenizer::load_symphony_tokenizer(absolute_tokenizer_path.to_str().unwrap()) {
                                tokenizer = Some(tok);
                                println!("Loaded tokenizer from {:?}", absolute_tokenizer_path);
                            }
                        }
                    }
                }
            }

            // Attempt 2: If not found, try tokenizer.json directly in model_path
            if tokenizer.is_none() {
                let direct_tokenizer_json_path = model_path.join("tokenizer.json");
                if direct_tokenizer_json_path.exists() {
                    println!("Attempting to load tokenizer.json directly from {:?}", direct_tokenizer_json_path);
                    if let Ok(tok) = tokenizer::load_hf_tokenizer(direct_tokenizer_json_path.to_str().unwrap()) {
                        tokenizer = Some(tok);
                        println!("Loaded tokenizer from {:?}", direct_tokenizer_json_path);
                    }
                }
            }
        } else {
            println!("Model name is not an absolute path, searching in cache: {} for {}", cache_base.display(), info.model_name);
            // Existing logic: Iterate through cache_base or construct path
            // Attempt 1: Find via symphony_model_info.json in subdirectories of cache_base
            if let Ok(dir_iter) = fs::read_dir(&cache_base) {
                for entry in dir_iter.flatten() {
                    let model_dir_path = entry.path();
                    if model_dir_path.is_dir() {
                        let symphony_info_path = model_dir_path.join("symphony_model_info.json");
                        if symphony_info_path.exists() {
                            if let Ok(content) = fs::read_to_string(&symphony_info_path) {
                                if let Ok(json) = serde_json::from_str::<Value>(&content) {
                                    if json.get("local_name").and_then(Value::as_str) == Some(info.model_name.as_str()) ||
                                       json.get("model_name").and_then(Value::as_str) == Some(info.model_name.as_str()) {
                                        if let Some(tok_str) = json.get("tokenizer_path").and_then(Value::as_str) {
                                            let mut absolute_tokenizer_path = PathBuf::from(tok_str);
                                            if !absolute_tokenizer_path.is_absolute() {
                                                absolute_tokenizer_path = model_dir_path.join(tok_str);
                                            }
                                            println!("Loading tokenizer from metadata path: {:?}", absolute_tokenizer_path);
                                            if let Ok(tok) = tokenizer::load_symphony_tokenizer(absolute_tokenizer_path.to_str().unwrap()) {
                                                tokenizer = Some(tok);
                                                println!("Loaded tokenizer from {:?}", absolute_tokenizer_path);
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Attempt 2: If not found, try tokenizer.json in a conventionally named cache directory
            if tokenizer.is_none() {
                let model_specific_cache_dir = cache_base.join(info.model_name.replace("/", "--"));
                let tokenizer_json_path = model_specific_cache_dir.join("tokenizer.json");
                if tokenizer_json_path.exists() {
                    println!("Attempting to load tokenizer.json from conventional cache path: {:?}", tokenizer_json_path);
                    if let Ok(tok) = tokenizer::load_hf_tokenizer(tokenizer_json_path.to_str().unwrap()) {
                        tokenizer = Some(tok);
                        println!("Loaded tokenizer from {:?}", tokenizer_json_path);
                    }
                }
            }
        }

        let tokenizer = tokenizer.expect(&format!(
            "Failed to load tokenizer for model \'{}\'.\\nInstall with: symphony-cli install-model {}",
            info.model_name, info.model_name
        ));

        let mut objects = ObjectManager::new();
        objects
            .set_capacity(ManagedTypes::KvBlock, info.num_blocks as IdRepr)
            .unwrap();
        objects
            .set_capacity(ManagedTypes::TokenEmb, info.num_embeddings as IdRepr)
            .unwrap();

        let driver = Self {
            scheduler: scheduler_tx,
            scheduler_loop_handle,
            event_loop_handle,
            exported_blocks: HashMap::new(),
            objects,
            stream_priorities: HashMap::new(),
            info,
            tokenizer: Arc::new(tokenizer),
            stats: L4mStat { total_calls: 0 },
        };

        driver
    }

    pub fn print_stats(&self) {
        let mut stats = Vec::new();
        for &managed_type in &[ManagedTypes::KvBlock, ManagedTypes::TokenEmb] {
            let current = self.objects.available(managed_type).unwrap();
            let capacity: usize = self.objects.capacity(managed_type).unwrap() as usize;
            let used = capacity - current;
            let percentage = (used as f32 / capacity as f32) * 100.0;

            let type_name = match managed_type {
                ManagedTypes::KvBlock => "kvpage",
                ManagedTypes::TokenEmb => "emb",
                _ => "unknown",
            };

            stats.push(format!(
                "{}: {} / {} ({:.2}% used)",
                type_name, used, capacity, percentage
            ));
        }

        stats.push(format!("Total calls: {}", self.stats.total_calls));

        println!("{}", stats.join(" | "));
    }

    fn get_cleanup_cmds(&mut self, inst_id: InstanceId) -> Vec<Command> {
        let mut cmds = Vec::new();

        for ty in [ManagedTypes::KvBlock, ManagedTypes::TokenEmb] {
            if let Ok(ids) = self.objects.all_names(ty, inst_id) {
                cmds.push(Command::Deallocate {
                    inst_id,
                    stream_id: 0,
                    ty,
                    ids,
                });
            }
        }

        //println!("deallocating all objects for instance: {:?}", cmds);

        // Remove all exported blocks
        self.exported_blocks.retain(|_, v| v.owner != inst_id);

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
            Command::PrintStats => {
                self.stats.total_calls -=1; // compensate for the print stats call

                self.print_stats();
                None
            }

            Command::Destroy { .. } => {
                unreachable!()
            }

            Command::GetInfo { handle } => Some((Command::GetInfo { handle }, Stream::default())),

            Command::GetBlockSize { handle } => {
                handle.send(self.info.block_size).ok();
                None
            }

            Command::GetTokenizer { handle } => {
                handle.send(self.tokenizer.clone()).ok();
                None
            }

            Command::GetAllExportedBlocks { handle } => {
                let catalogue = self
                    .exported_blocks
                    .iter()
                    .map(|(k, v)| (k.clone(), v.addrs.len() as u32))
                    .collect();
                handle.send(catalogue).ok();
                None
            }

            Command::Allocate {
                inst_id,
                stream_id,
                ty,
                ids,
            } => {

                // check available space
                if self.objects.available(ty).unwrap() < ids.len() {
                    runtime::trap(
                        inst_id,
                        "l4m::allocation failed. not enough available space",
                    );
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

            Command::FillBlock {
                inst_id,
                stream_id,
                last_block_len,
                mut context,
                mut inputs,
                mut outputs,
            } => {
                if last_block_len == 0 || last_block_len > self.info.block_size {
                    // error
                    runtime::trap(
                        inst_id,
                        format!(
                            "l4m::fill_block failed. last_block_len ({}) is 0 or greater than the block size ({})",
                            last_block_len, self.info.block_size
                        ),
                    );
                    return None;
                }

                let max_tokens = self.info.block_size * (context.len() as u32 - 1) + last_block_len;

                if inputs.len() > max_tokens as usize {
                    // error
                    runtime::trap(
                        inst_id,
                        format!(
                            "l4m::fill_block failed. inputs length is greater than the max tokens: {} > {}",
                            inputs.len(),
                            max_tokens
                        ),
                    );
                    return None;
                }

                try_trap!(
                    self.objects
                        .translate_many(ManagedTypes::KvBlock, inst_id, &mut context),
                    inst_id,
                    "l4m::fill_block failed. some context blocks are invalid"
                );
                try_trap!(
                    self.objects
                        .translate_many(ManagedTypes::TokenEmb, inst_id, &mut inputs),
                    inst_id,
                    "l4m::fill_block failed. some input embeddings are invalid"
                );
                try_trap!(
                    self.objects
                        .translate_many(ManagedTypes::TokenEmb, inst_id, &mut outputs),
                    inst_id,
                    "l4m::fill_block failed. some output embeddings are invalid"
                );

                Some((
                    Command::FillBlock {
                        inst_id,
                        stream_id,
                        last_block_len,
                        context,
                        inputs,
                        outputs,
                    },
                    Stream::new(inst_id, stream_id),
                ))
            }

            Command::ExportBlocks {
                inst_id,
                mut blocks,
                resource_name,
            } => {
                try_trap!(
                    self.objects
                        .translate_many(ManagedTypes::KvBlock, inst_id, &mut blocks),
                    inst_id,
                    "l4m::export_blocks failed. some blocks are invalid"
                );

                self.exported_blocks
                    .insert(resource_name, ExportedBlocks::new(inst_id, blocks));
                None
            }

            Command::ImportBlocks {
                inst_id,
                blocks,
                resource_name,
            } => {
                let exported = match self.exported_blocks.get(&resource_name) {
                    Some(exp) => exp,
                    None => {
                        runtime::trap(
                            inst_id,
                            format!(
                                "l4m::import_blocks failed. resource {} not found",
                                resource_name
                            ),
                        );
                        return None;
                    }
                };

                try_trap!(
                    self.objects.create_ref_many(
                        ManagedTypes::KvBlock,
                        inst_id,
                        blocks,
                        &exported.addrs
                    ),
                    inst_id,
                    "l4m::import_blocks failed"
                );
                None
            }

            Command::CopyBlock {
                inst_id,
                stream_id,
                mut src_block,
                mut dst_block,
                src_token_offset,
                dst_token_offset,
                size,
            } => {
                try_trap!(
                    self.objects
                        .translate(ManagedTypes::KvBlock, inst_id, &mut src_block),
                    inst_id,
                    "l4m::copy_block failed. invalid source block"
                );
                try_trap!(
                    self.objects
                        .translate(ManagedTypes::KvBlock, inst_id, &mut dst_block),
                    inst_id,
                    "l4m::copy_block failed. invalid destination block"
                );

                Some((
                    Command::CopyBlock {
                        inst_id,
                        stream_id,
                        src_block,
                        dst_block,
                        src_token_offset,
                        dst_token_offset,
                        size,
                    },
                    Stream::new(inst_id, stream_id),
                ))
            }

            Command::MaskBlock {
                inst_id,
                stream_id,
                mut block,
                mask,
            } => {
                try_trap!(
                    self.objects
                        .translate(ManagedTypes::KvBlock, inst_id, &mut block),
                    inst_id,
                    "l4m::mask_block failed. invalid block"
                );

                Some((
                    Command::MaskBlock {
                        inst_id,
                        stream_id,
                        block,
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
                        .translate_many(ManagedTypes::TokenEmb, inst_id, &mut embs),
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
                        .translate(ManagedTypes::TokenEmb, inst_id, &mut emb_id),
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
                        .translate_many(ManagedTypes::TokenEmb, inst_id, &mut embs),
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
                                handle
                                    .send(Info {
                                        version: info.version,
                                        model_name: info.model_name,
                                        block_size: info.block_size,
                                        num_blocks: info.num_available_blocks,
                                        num_embeddings: info.num_available_embeddings,
                                        num_distributions: info.num_available_distributions,
                                    })
                                    .ok();
                            }
                            _ => unreachable!(),
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
                backend
                    .protocol_index(protoc)
                    .expect(&format!("Failed to get protocol index: UnsupportedProtocol(\"{}\")", protoc))
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
            BatchGroup::CopyBlock => encode_pb_batch_copy_block(correlation_id, batch),
            BatchGroup::MaskBlock => encode_pb_batch_mask_block(correlation_id, batch),
            BatchGroup::EmbedText => encode_pb_batch_embed_text(correlation_id, batch),
            BatchGroup::SampleTopK => encode_pb_batch_sample_topk(correlation_id, batch),
            BatchGroup::Synchronize => {
                let cmd = batch.into_iter().next().unwrap();
                match cmd {
                    Command::Synchronize {
                        inst_id,
                        stream_id,
                        handle,
                    } => {
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
            .unwrap();
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
            protocols: vec!["l4m".to_string()],
        }
    }

    pub fn new_with_protocols(protocols: Vec<String>) -> Self {
        Self { protocols }
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
            Command::Allocate {
                inst_id,
                stream_id,
                ty,
                ids,
            }
            | Command::Deallocate {
                inst_id,
                stream_id,
                ty,
                ids,
            } => {
                let kind = match ty {
                    ManagedTypes::KvBlock => pb_bindings::ObjectKind::KvBlock,
                    ManagedTypes::TokenEmb => pb_bindings::ObjectKind::Emb,
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
                inst_id,
                stream_id,
                ty,
                ids,
            }
            | Command::Deallocate {
                inst_id,
                stream_id,
                ty,
                ids,
            } => {
                let kind = match ty {
                    ManagedTypes::KvBlock => pb_bindings::ObjectKind::KvBlock,
                    ManagedTypes::TokenEmb => pb_bindings::ObjectKind::Emb,
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
            Command::FillBlock {
                inst_id,
                stream_id,
                last_block_len,
                context,
                inputs,
                outputs,
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

fn encode_pb_batch_copy_block(
    correlation_id: u32,
    batch: Vec<Command>,
) -> ((usize, Vec<u8>), Option<Vec<Event>>) {
    let mut items = Vec::new();
    for cmd in batch {
        match cmd {
            Command::CopyBlock {
                inst_id,
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
                inst_id,
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
                inst_id,
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

fn encode_pb_batch_sample_topk(
    correlation_id: u32,
    batch: Vec<Command>,
) -> ((usize, Vec<u8>), Option<Vec<Event>>) {
    let mut items = Vec::new();
    let mut events = Vec::new();
    for cmd in batch {
        match cmd {
            Command::SampleTopK {
                inst_id,
                stream_id,
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
                inst_id,
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
