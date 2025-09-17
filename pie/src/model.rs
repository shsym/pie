use crate::batching::{BatchingConfig, MultiStreamBatcher};
use crate::handler::{Handler, get_batching_config};
use crate::instance::InstanceId;
use crate::resource::{ResourceId, ResourceManager, ResourceTypeId};
use crate::runtime::trap_exception;
use crate::service::{self, Service, ServiceError};
use crate::tokenizer::BytePairEncoder;
use bytes::Bytes;
use futures::future;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, mpsc, oneshot};
use tokio::task::JoinHandle;
use zeromq::{DealerSocket, Socket, SocketRecv, SocketSend, ZmqMessage};

pub type HandlerId = u32;

pub type CmdQueueId = u32;

static REGISTERED_MODELS: std::sync::LazyLock<boxcar::Vec<(String, usize)>> =
    std::sync::LazyLock::new(boxcar::Vec::new);

pub fn registered_models() -> Vec<String> {
    REGISTERED_MODELS
        .iter()
        .map(|(_, (model_name, _))| model_name.clone())
        .collect()
}

pub fn register_model(model_name: String, service_id: usize) {
    REGISTERED_MODELS.push((model_name, service_id));
}

pub fn model_service_id(model_name: &str) -> Option<usize> {
    REGISTERED_MODELS
        .iter()
        .find(|(_, (name, _))| name == model_name)
        .map(|(_, (_, service_id))| *service_id)
}

pub fn cleanup_instance(inst_id: InstanceId) {
    REGISTERED_MODELS.iter().for_each(|(_, (_, service_id))| {
        Command::Cleanup { inst_id }.dispatch(*service_id).ok();
    })
}

/// Asynchronously collects runtime statistics from all registered models.
pub async fn runtime_stats() -> HashMap<String, String> {
    let mut aggregated_stats = HashMap::new();
    let mut futures = Vec::new();

    // Dispatch requests to all models concurrently.
    for (_, (model_name, service_id)) in REGISTERED_MODELS.iter() {
        let (tx, rx) = oneshot::channel();
        let cmd = Command::GetRuntimeStats { response: tx };

        if cmd.dispatch(*service_id).is_ok() {
            // Store the model name and the future for its response.
            futures.push((model_name.clone(), rx));
        } else {
            // Handle cases where the service is unavailable immediately.
            aggregated_stats.insert(
                format!("{}.error", model_name),
                "failed to dispatch command to service".to_string(),
            );
        }
    }

    // Await all responses in parallel.
    let results = future::join_all(
        futures
            .into_iter()
            .map(async move |(name, rx)| (name, rx.await)),
    )
    .await;

    // Process the results.
    for (model_name, result) in results {
        match result {
            Ok(model_stats) => {
                for (key, value) in model_stats {
                    aggregated_stats.insert(format!("{}.{}", model_name, key), value);
                }
            }
            Err(e) => {
                // The service might have crashed or failed to respond.
                aggregated_stats.insert(
                    format!("{}.error", model_name),
                    format!("failed to receive stats from service: {}", e),
                );
            }
        }
    }

    aggregated_stats
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HandshakeRequest {
    pub version: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HandshakeResponse {
    pub version: String,
    pub model_name: String,
    pub model_traits: Vec<String>,
    pub model_description: String,
    pub prompt_template: String,
    pub prompt_template_type: String,
    pub prompt_stop_tokens: Vec<String>,
    pub kv_page_size: u32,
    pub resources: HashMap<u32, u32>,
    pub tokenizer_merge_table: HashMap<u32, Vec<u8>>,
    pub tokenizer_special_tokens: HashMap<String, u32>,
    pub tokenizer_split_regex: String,
    pub tokenizer_escape_non_printable: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QueryRequest {
    pub query: String,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct QueryResponse {
    pub value: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ForwardPassRequest {
    pub input_tokens: Vec<u32>,
    pub input_token_positions: Vec<u32>,
    pub input_embed_ptrs: Vec<u32>,
    pub input_embed_positions: Vec<u32>,
    pub adapter: Option<u32>,
    pub adapter_seed: Option<i64>,
    pub mask: Vec<Vec<u32>>,
    pub kv_page_ptrs: Vec<u32>,
    pub kv_page_last_len: u32,
    pub output_token_indices: Vec<u32>,
    pub output_token_samplers: Vec<HashMap<String, rmpv::Value>>,
    pub output_embed_ptrs: Vec<u32>,
    pub output_embed_indices: Vec<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ForwardPassResponse {
    pub tokens: Vec<u32>,
    pub dists: Vec<(Vec<u32>, Vec<f32>)>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbedImageRequest {
    pub embed_ptrs: Vec<u32>,
    pub image_blob: Vec<u8>,
    pub position_offset: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InitializeAdapterRequest {
    pub adapter_ptr: u32,
    pub rank: u32,
    pub alpha: f32,
    pub population_size: u32,
    pub mu_fraction: f32,
    pub initial_sigma: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UpdateAdapterRequest {
    pub adapter_ptr: u32,
    pub scores: Vec<f32>,
    pub seeds: Vec<i64>,
    pub max_sigma: f32,
}

/// Defines the set of operations available for the key-value store.
#[derive(Debug)]
pub enum Command {
    Submit {
        inst_id: InstanceId,
        cmd_queue_id: CmdQueueId,
        handler: Handler,
        data: Bytes,
        response: Option<oneshot::Sender<Bytes>>,
    },
    GetInfo {
        response: oneshot::Sender<ModelInfo>,
    },
    GetRuntimeStats {
        response: oneshot::Sender<HashMap<String, String>>,
    },
    Allocate {
        inst_id: InstanceId,
        type_id: ResourceTypeId,
        count: usize,
        response: oneshot::Sender<Vec<ResourceId>>,
    },
    Deallocate {
        inst_id: InstanceId,
        type_id: ResourceTypeId,
        ptrs: Vec<ResourceId>,
    },
    Cleanup {
        inst_id: InstanceId,
    },
    GetAllExported {
        type_id: ResourceTypeId,
        response: oneshot::Sender<Vec<(String, Vec<ResourceId>)>>,
    },
    Export {
        inst_id: InstanceId,
        type_id: ResourceTypeId,
        ptrs: Vec<ResourceId>,
        name: String,
    },
    Import {
        inst_id: InstanceId,
        type_id: ResourceTypeId,
        name: String,
        response: oneshot::Sender<Vec<ResourceId>>,
    },
    ReleaseExported {
        inst_id: InstanceId,
        type_id: ResourceTypeId,
        name: String,
    },
}

impl Command {
    pub fn dispatch(self, service_id: usize) -> Result<(), ServiceError> {
        service::dispatch(service_id, self)
    }
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub traits: Vec<String>,
    pub description: String,
    pub prompt_template: String,
    pub prompt_template_type: String,
    pub prompt_stop_tokens: Vec<String>,
    pub tokenizer: Arc<BytePairEncoder>,
    pub kv_page_size: u32,
}

/// An in-memory key-value store service.
#[derive(Debug)]
pub struct Model {
    info: ModelInfo,
    resource_manager: ResourceManager,

    shutdown_tx: broadcast::Sender<()>,
    scheduler_tx:
        mpsc::UnboundedSender<(Handler, CmdQueueId, Option<oneshot::Sender<Bytes>>, Bytes)>,
    scheduling_worker_handle: Option<JoinHandle<()>>,
    backend_worker_handle: Option<JoinHandle<()>>,
}

impl Model {
    pub async fn new(endpoint: &str) -> anyhow::Result<Self> {
        let mut socket = DealerSocket::new();
        socket.connect(endpoint).await?;

        let mut batching_config = get_batching_config();
        let mut batch_triggers = HashMap::new();
        for (handler, cfg) in batching_config.iter_mut() {
            if let BatchingConfig::Triggered { trigger, .. } = cfg {
                let t = Arc::new(AtomicBool::new(true));
                batch_triggers.insert(handler.get_handler_id(), t.clone());
                *trigger = Some(t.clone());
            }
        }

        let (backend_tx, backend_rx) = mpsc::unbounded_channel();
        let (scheduler_tx, scheduler_rx) = mpsc::unbounded_channel();

        // NEW: Shutdown channel for workers.
        let (shutdown_tx, shutdown_rx) = broadcast::channel(1);

        let backend_worker_handle = tokio::spawn(Self::backend_worker(
            socket,
            backend_rx,
            batch_triggers,
            shutdown_rx,
        ));
        let scheduling_worker_handle = tokio::spawn(Self::scheduling_worker(
            batching_config,
            scheduler_rx,
            backend_tx,
            shutdown_tx.subscribe(),
        ));

        let (handshake_tx, handshake_rx) = oneshot::channel();
        scheduler_tx.send((
            Handler::Handshake,
            0,
            Some(handshake_tx),
            Bytes::from(rmp_serde::to_vec_named(&HandshakeRequest {
                version: "0.1.0".to_string(),
            })?),
        ))?;

        let handshake_info = rmp_serde::from_slice::<HandshakeResponse>(&handshake_rx.await?)?;

        let tokenizer = Arc::new(BytePairEncoder::new(
            handshake_info.tokenizer_merge_table.into_iter().collect(),
            handshake_info.tokenizer_special_tokens,
            &handshake_info.tokenizer_split_regex,
            handshake_info.tokenizer_escape_non_printable,
        ));

        let info = ModelInfo {
            name: handshake_info.model_name,
            traits: handshake_info.model_traits,
            description: handshake_info.model_description,
            prompt_template: handshake_info.prompt_template,
            prompt_template_type: handshake_info.prompt_template_type,
            prompt_stop_tokens: handshake_info.prompt_stop_tokens,
            tokenizer,
            kv_page_size: handshake_info.kv_page_size,
        };

        let resource_manager = ResourceManager::new(handshake_info.resources);

        Ok(Model {
            info,
            resource_manager,
            scheduler_tx,
            shutdown_tx,
            scheduling_worker_handle: Some(scheduling_worker_handle),
            backend_worker_handle: Some(backend_worker_handle),
        })
    }

    // NEW: Graceful shutdown method.
    pub async fn shutdown(mut self) -> anyhow::Result<()> {
        println!("[Info] Shutting down model service...");
        self.shutdown_tx.send(())?;

        if let Some(handle) = self.scheduling_worker_handle.take() {
            handle.await?;
        }
        if let Some(handle) = self.backend_worker_handle.take() {
            handle.await?;
        }
        println!("[Info] Model service shut down gracefully.");
        Ok(())
    }

    async fn scheduling_worker(
        config: HashMap<Handler, BatchingConfig>,
        mut rx: mpsc::UnboundedReceiver<(
            Handler,
            CmdQueueId,
            Option<oneshot::Sender<Bytes>>,
            Bytes,
        )>,
        backend_tx: mpsc::UnboundedSender<(Handler, Vec<(Option<oneshot::Sender<Bytes>>, Bytes)>)>,
        mut shutdown_rx: broadcast::Receiver<()>,
    ) {
        let mut batcher = MultiStreamBatcher::new(config.clone());
        let mut next_poll_duration: Option<Duration> = None;
        const IDLE_TIMEOUT: Duration = Duration::from_secs(1);

        loop {
            let sleep_duration = next_poll_duration.unwrap_or(IDLE_TIMEOUT);

            tokio::select! {
                _ = shutdown_rx.recv() => {
                    println!("[Info] Shutdown signal received, exiting scheduling worker.");
                    break;
                },
                maybe_msg = rx.recv() => {
                    if let Some((handler, cmd_queue_id, sender, msg)) = maybe_msg {
                        if !config.contains_key(&handler) {
                            if backend_tx.send((handler, vec![(sender, msg)])).is_err() {
                               eprintln!("[Warn] Backend channel closed, could not send non-batched message.");
                            }
                            continue;
                        }
                        batcher.push(cmd_queue_id, handler, (sender, msg), Instant::now());
                    } else {
                        println!("[Info] Command channel closed, shutting down scheduler handler.");
                        break;
                    }
                },
                _ = tokio::time::sleep(sleep_duration) => {}
            }

            let batch = batcher.poll(Instant::now());
            for (batch_handler, batch_payload) in batch {
                if batch_handler == Handler::Synchronize {
                    if let Some((sender, _)) = batch_payload.into_iter().next() {
                        if let Some(sender) = sender {
                            // FIX: Log error if send fails.
                            if sender.send(Bytes::default()).is_err() {
                                println!(
                                    "[Warn] Synchronize response channel closed before sending."
                                );
                            }
                        }
                    }
                    continue;
                }

                if !batch_payload.is_empty() {
                    if backend_tx.send((batch_handler, batch_payload)).is_err() {
                        eprintln!("[Warn] Backend channel closed, could not send batch.");
                    }
                }
            }
            next_poll_duration = batcher.next_poll_in(Instant::now());
        }
    }

    async fn backend_worker(
        mut socket: DealerSocket,
        mut rx: mpsc::UnboundedReceiver<(Handler, Vec<(Option<oneshot::Sender<Bytes>>, Bytes)>)>,
        batch_triggers: HashMap<HandlerId, Arc<AtomicBool>>,
        mut shutdown_rx: broadcast::Receiver<()>,
    ) {
        let mut corr_id: u32 = 0;
        // Store Instant for request timeout handling.
        let mut event_table: HashMap<(u32, usize), (oneshot::Sender<Bytes>, Instant)> =
            HashMap::new();
        const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

        loop {
            // NEW: Dynamic sleep for timeout checks.
            let sleep_duration = event_table
                .values()
                .map(|(_, instant)| {
                    instant.saturating_duration_since(Instant::now()) + REQUEST_TIMEOUT
                })
                .min()
                .unwrap_or(REQUEST_TIMEOUT);

            tokio::select! {
                _ = shutdown_rx.recv() => {
                    println!("[Info] Shutdown signal received, exiting backend worker.");
                    break;
                },
                maybe_command = rx.recv() => {
                    if let Some((handler, payload)) = maybe_command {
                        let current_corr_id = corr_id;
                        let (senders, batch): (Vec<_>, Vec<_>) = payload.into_iter().unzip();
                        let mut frames: VecDeque<Bytes> = VecDeque::with_capacity(2 + batch.len());
                        frames.push_back(Bytes::copy_from_slice(&current_corr_id.to_be_bytes()));
                        frames.push_back(Bytes::copy_from_slice(&handler.get_handler_id().to_be_bytes()));
                        frames.extend(batch);

                        if let Err(e) = socket.send(ZmqMessage::try_from(frames).unwrap()).await {
                            eprintln!("[Error] Socket send failed for corr_id {}: {:?}", current_corr_id, e);
                            continue;
                        }

                        for (idx, sender) in senders.into_iter().enumerate() {
                            if let Some(sender) = sender {
                                // NEW: Store the Instant along with the sender.
                                event_table.insert((current_corr_id, idx), (sender, Instant::now()));
                            }
                        }
                        corr_id = corr_id.wrapping_add(1);
                    } else {
                        println!("[Info] Command channel closed, shutting down backend handler.");
                        break;
                    }
                },
                result = socket.recv() => {
                    match result {
                        Ok(zmq_msg) => {
                            let mut frames = zmq_msg.into_vecdeque();
                            // Safely parse incoming frames to prevent panics.
                            let Some(corr_id_bytes) = frames.pop_front() else { continue; };
                            let Some(handler_id_bytes) = frames.pop_front() else { continue; };
                            let Ok(corr_id_slice) = corr_id_bytes.as_ref().try_into() else { continue; };
                            let Ok(handler_id_slice) = handler_id_bytes.as_ref().try_into() else { continue; };

                            let received_corr_id = u32::from_be_bytes(corr_id_slice);
                            let received_handler_id = u32::from_be_bytes(handler_id_slice);

                            for (idx, payload) in frames.into_iter().enumerate() {
                                let key = (received_corr_id, idx);
                                if let Some((sender, _)) = event_table.remove(&key) {
                                    // println!("data: {:?}", payload);
                                    let _ = sender.send(payload);
                                }
                            }

                            if let Some(trigger) = batch_triggers.get(&received_handler_id) {
                                trigger.store(true, std::sync::atomic::Ordering::SeqCst);
                            }
                        },
                        Err(e) => {
                            eprintln!("[Error] Socket receive error: {}. Shutting down.", e);
                            break;
                        }
                    }
                },
                _ = tokio::time::sleep(sleep_duration) => {
                    let now = Instant::now();
                    event_table.retain(|_key, (sender, instant)| {
                        if now.duration_since(*instant) > REQUEST_TIMEOUT {
                            // The receiver will get a RecvError, indicating the request failed.
                            eprintln!("[Warn] Request timed out. Dropping sender.");
                            false
                        } else {
                            true
                        }
                    });
                }
            }
        }
    }

    pub fn submit(
        &self,
        handler: Handler,
        cmd_queue_id: CmdQueueId,
        sender: Option<oneshot::Sender<Bytes>>,
        msg: Bytes,
    ) {
        // FIX: Log error if the scheduler channel is closed.
        if self
            .scheduler_tx
            .send((handler, cmd_queue_id, sender, msg))
            .is_err()
        {
            eprintln!("[Error] Failed to submit command: Scheduler channel is closed.");
        }
    }

    /// Gathers detailed runtime statistics for this specific model instance.
    pub fn runtime_stats(&self) -> HashMap<String, String> {
        let mut stats = HashMap::new();
        stats.insert("model.name".to_string(), self.info.name.clone());

        // Collect stats from the ResourceManager.
        self.resource_manager.append_stats_to(&mut stats);

        stats
    }
}

impl Service for Model {
    type Command = Command;

    async fn handle(&mut self, cmd: Self::Command) {
        match cmd {
            Command::Submit {
                inst_id: _,
                handler,
                cmd_queue_id,
                data,
                response,
            } => {
                self.submit(handler, cmd_queue_id, response, data);
            }

            Command::GetInfo { response } => {
                if response.send(self.info.clone()).is_err() {
                    println!("[Warn] GetInfo response channel closed before sending.");
                }
            }

            Command::GetRuntimeStats { response } => {
                if response.send(self.runtime_stats()).is_err() {
                    println!("[Warn] GetRuntimeStats response channel closed before sending.");
                }
            }

            Command::Allocate {
                inst_id,
                type_id,
                count,
                response,
            } => match self
                .resource_manager
                .allocate_with_oom(inst_id, type_id, count)
            {
                Ok(allocated_ids) => {
                    if response.send(allocated_ids).is_err() {
                        println!("[Warn] Allocate response channel closed before sending.");
                    }
                }
                Err(e) => trap_exception(inst_id, e),
            },
            Command::Deallocate {
                inst_id,
                type_id,
                ptrs,
            } => {
                if let Err(e) = self.resource_manager.deallocate(inst_id, type_id, ptrs) {
                    trap_exception(inst_id, e);
                }
            }
            Command::Cleanup { inst_id } => {
                if let Err(e) = self.resource_manager.cleanup(inst_id) {
                    trap_exception(inst_id, e);
                }
            }
            Command::GetAllExported { type_id, response } => {
                let list = self.resource_manager.get_all_exported(type_id);
                if response.send(list).is_err() {
                    println!("[Warn] GetAllExported response channel closed before sending.");
                }
            }
            Command::Export {
                inst_id,
                type_id,
                ptrs,
                name,
            } => {
                if let Err(e) = self.resource_manager.export(inst_id, type_id, ptrs, name) {
                    trap_exception(inst_id, e);
                }
            }
            Command::Import {
                inst_id,
                type_id,
                name,
                response,
            } => match self.resource_manager.import(type_id, name) {
                Ok(ptrs) => {
                    if response.send(ptrs).is_err() {
                        println!("[Warn] Import response channel closed before sending.");
                    }
                }
                Err(e) => trap_exception(inst_id, e),
            },
            Command::ReleaseExported {
                inst_id,
                type_id,
                name,
            } => {
                if let Err(e) = self.resource_manager.release_exported(type_id, name) {
                    trap_exception(inst_id, e);
                }
            }
        }
    }
}
