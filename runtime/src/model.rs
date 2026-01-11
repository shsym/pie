pub mod actor;
pub mod batching;
pub mod ffi_bridge;
pub mod ffi_queue;
pub mod request;
pub mod resource;
pub mod tokenizer;

use super::model::batching::{AdaptiveScheduler, SharedScheduler};
use super::model::ffi_bridge::AsyncFfiClient;
use super::model::request::{
    BatchedForwardPassRequest, BatchedForwardPassResponse, ForwardPassRequest, ForwardPassResponse,
    HandshakeRequest, HandshakeResponse, QueryRequest, QueryResponse, Request,
};
use super::model::resource::{ResourceId, ResourceManager, ResourceTypeId};
use super::model::tokenizer::BytePairEncoder;
use super::runtime::{self, TerminationCause};
use super::service::ServiceCommand;
use crate::instance::InstanceId;
use anyhow::Result;
use bytes::Bytes;
use futures::future;
use serde::{de::DeserializeOwned, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::sync::{broadcast, mpsc, oneshot};
use tokio::task::{self, JoinHandle};

// Re-export SchedulerConfig for public API
pub use batching::SchedulerConfig;

/// Backend for RPC calls to Python via FFI.
#[derive(Clone)]
pub struct RpcBackend(AsyncFfiClient);

impl RpcBackend {
    /// Create a new RpcBackend from an FFI client.
    pub fn new(client: AsyncFfiClient) -> Self {
        RpcBackend(client)
    }

    /// Call a remote method with typed arguments and return value.
    pub async fn call<T, R>(&self, method: &str, args: &T) -> Result<R>
    where
        T: Serialize + Send + Sync + Clone + 'static,
        R: DeserializeOwned + Send + 'static,
    {
        self.0.call(method, args).await
    }

    /// Call with timeout.
    pub async fn call_with_timeout<T, R>(
        &self,
        method: &str,
        args: &T,
        timeout: Duration,
    ) -> Result<R>
    where
        T: Serialize + Send + Sync + Clone + 'static,
        R: DeserializeOwned + Send + 'static,
    {
        self.0.call_with_timeout(method, args, timeout).await
    }

    /// Fire-and-forget notification.
    pub async fn notify<T>(&self, method: &str, args: &T) -> Result<()>
    where
        T: Serialize + Send + Sync + Clone + 'static,
    {
        self.0.notify(method, args).await
    }
}

pub type HandlerId = u32;
pub type CmdQueueId = u32;

static MODEL_DISPATCHER: LazyLock<ModelDispatcher> = LazyLock::new(|| ModelDispatcher {
    models: boxcar::Vec::new(),
});

#[derive(Debug, Error)]
pub enum ModelDispatchError {
    #[error("Invalid model index: {0}")]
    InvalidModelIndex(usize),
}

#[derive(Debug)]
struct ModelDispatcher {
    models: boxcar::Vec<(String, mpsc::UnboundedSender<Command>)>,
}

pub fn install_model(model_name: String, mut model: Model) -> Option<usize> {
    for (_, (existing_name, _)) in MODEL_DISPATCHER.models.iter() {
        if existing_name == &model_name {
            return None;
        }
    }

    let (tx, mut rx) = mpsc::unbounded_channel();
    MODEL_DISPATCHER.models.push((model_name, tx));
    let model_id = MODEL_DISPATCHER.models.count() - 1;

    task::spawn(async move {
        while let Some(cmd) = rx.recv().await {
            model.handle(cmd).await;
        }
    });

    Some(model_id)
}

pub fn registered_models() -> Vec<String> {
    MODEL_DISPATCHER
        .models
        .iter()
        .map(|(_, (name, _))| name.clone())
        .collect()
}

pub fn model_service_id(model_name: &str) -> Option<usize> {
    MODEL_DISPATCHER
        .models
        .iter()
        .find(|(_, (name, _))| name == model_name)
        .map(|(idx, _)| idx)
}

pub fn cleanup_instance(inst_id: InstanceId) {
    for (model_id, _) in MODEL_DISPATCHER.models.iter() {
        Command::Cleanup { inst_id }.dispatch(model_id).ok();
    }
}

pub async fn runtime_stats() -> HashMap<String, String> {
    let mut aggregated_stats = HashMap::new();
    let mut futures = Vec::new();

    for (model_id, (model_name, _)) in MODEL_DISPATCHER.models.iter() {
        let (tx, rx) = oneshot::channel();
        let cmd = Command::GetRuntimeStats { response: tx };

        if cmd.dispatch(model_id).is_ok() {
            futures.push((model_name.clone(), rx));
        } else {
            aggregated_stats.insert(
                format!("{}.error", model_name),
                "failed to dispatch command to service".to_string(),
            );
        }
    }

    let results = future::join_all(
        futures
            .into_iter()
            .map(async move |(name, rx)| (name, rx.await)),
    )
    .await;

    for (model_name, result) in results {
        match result {
            Ok(model_stats) => {
                for (key, value) in model_stats {
                    aggregated_stats.insert(format!("{}.{}", model_name, key), value);
                }
            }
            Err(e) => {
                aggregated_stats.insert(
                    format!("{}.error", model_name),
                    format!("failed to receive stats from service: {}", e),
                );
            }
        }
    }

    aggregated_stats
}

pub fn submit_request(
    service_id: usize,
    cmd_queue_id: CmdQueueId,
    priority: u32,
    req: Request,
) -> Result<()> {
    Command::Submit {
        cmd_queue_id,
        priority,
        req,
    }
    .dispatch(service_id)?;
    Ok(())
}

fn terminate_instance_with_exception<T>(inst_id: InstanceId, exception: T)
where
    T: ToString,
{
    runtime::Command::TerminateInstance {
        inst_id,
        notification_to_client: Some(TerminationCause::Exception(exception.to_string())),
    }
    .dispatch();
}

#[derive(Debug)]
pub enum Command {
    Submit {
        cmd_queue_id: CmdQueueId,
        priority: u32,
        req: Request,
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
    // Actor Commands
    ActorGlobalContextRef { username: String, uid: String },
    ActorGlobalContextDestroy { username: String, uid: String },
    ActorGlobalContextExtend { username: String, uid: String, page_ids: Vec<u32>, last_page_len: u32 },
    ActorGlobalContextTrim { username: String, uid: String, len: u32 },
    ActorGlobalContextRead { username: String, uid: String, num_tokens: u32, offset: u32, response: oneshot::Sender<Vec<u32>> },
    ActorAdapterRef { username: String, uid: String },
    ActorAdapterDestroy { username: String, uid: String },
    ActorAdapterBlank { username: String, uid: String, rank: u32, alpha: f32 },
    ActorAdapterLoad { username: String, uid: String, path: String },
    ActorOptimizerRef { username: String, uid: String },
    ActorOptimizerDestroy { username: String, uid: String },
    ActorOptimizerLoad { username: String, uid: String, path: String },
    ActorOptimizerSave { username: String, uid: String, path: String },
    ActorOptimizerInitialize { username: String, uid: String, adapter_uid: String, params: Vec<u8> },
    ActorOptimizerUpdate { username: String, uid: String, params: Vec<u8> },
}

impl Command {
    pub fn dispatch(self, model_id: usize) -> Result<(), ModelDispatchError> {
        let (_, tx) = MODEL_DISPATCHER
            .models
            .get(model_id)
            .ok_or(ModelDispatchError::InvalidModelIndex(model_id))?;
        tx.send(self).unwrap();
        Ok(())
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
    pub max_batch_tokens: usize,
}

/// Model service for communication with Python backend via FFI.
pub struct Model {
    info: ModelInfo,
    resource_manager: ResourceManager,
    shutdown_tx: broadcast::Sender<()>,
    backend: RpcBackend,
    /// Batch limits from handshake
    max_batch_tokens: usize,
    max_batch_size: usize,
    /// Channel for forward pass requests
    forward_pass_tx: mpsc::UnboundedSender<(ForwardPassRequest, Option<oneshot::Sender<ForwardPassResponse>>)>,
    worker_handle: Option<JoinHandle<()>>,
    /// Scheduler configuration
    scheduler_config: SchedulerConfig,
}

impl Model {
    /// Get the model name.
    pub fn name(&self) -> &str {
        &self.info.name
    }

    /// Create a new Model with an FFI backend using a queue.
    ///
    /// The queue should be shared with Python which polls it for requests.
    /// FfiQueue uses internal Arc so cloning shares state.
    pub async fn new(queue: ffi_queue::FfiQueue, scheduler_config: SchedulerConfig) -> Result<Self> {
        let ffi_client = AsyncFfiClient::new_with_queue(queue);
        let backend = RpcBackend::new(ffi_client);
        Self::new_with_backend(backend, scheduler_config).await
    }

    /// Internal constructor that works with any backend.
    async fn new_with_backend(backend: RpcBackend, scheduler_config: SchedulerConfig) -> Result<Self> {
        let handshake_info = Self::handshake(&backend).await?;

        let (forward_pass_tx, forward_pass_rx) = mpsc::unbounded_channel();
        let (shutdown_tx, shutdown_rx) = broadcast::channel(1);

        let max_batch_tokens = handshake_info.max_batch_tokens;
        let max_batch_size = handshake_info.max_batch_size;

        // Create shared scheduler for adaptive batching
        let scheduler = Arc::new(Mutex::new(AdaptiveScheduler::new(
            scheduler_config.clone(),
            max_batch_size,
        )));

        let worker_handle = tokio::spawn(Self::inference_worker(
            backend.clone(),
            forward_pass_rx,
            shutdown_rx,
            max_batch_tokens,
            max_batch_size,
            Arc::clone(&scheduler),
            scheduler_config.max_in_flight_batches,
        ));

        let tokenizer = Arc::new(BytePairEncoder::new(
            handshake_info.tokenizer_num_vocab,
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
            max_batch_tokens: handshake_info.max_batch_tokens,
        };

        let resource_manager = ResourceManager::new(handshake_info.resources);

        Ok(Model {
            info,
            resource_manager,
            backend,
            max_batch_tokens,
            max_batch_size,
            forward_pass_tx,
            shutdown_tx,
            worker_handle: Some(worker_handle),
            scheduler_config,
        })
    }

    async fn handshake(backend: &RpcBackend) -> Result<HandshakeResponse> {
        const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(30);
        let req = HandshakeRequest { version: "0.1.0".to_string() };
        let response: HandshakeResponse = backend
            .call_with_timeout("handshake", &req, HANDSHAKE_TIMEOUT)
            .await?;
        Ok(response)
    }

    pub async fn shutdown(mut self) -> Result<()> {
        self.shutdown_tx.send(())?;
        if let Some(handle) = self.worker_handle.take() {
            handle.await?;
        }
        Ok(())
    }

    /// Adaptive inference worker with concurrent batch execution.
    ///
    /// Design:
    /// - Uses adaptive scheduler to decide when to fire batches (optimizing for throughput)
    /// - Multiple batches execute concurrently via spawned tasks (pipelining)
    /// - Scheduler tracks arrival rate and latency to make optimal decisions
    /// - Overlaps batch accumulation with GPU inference for maximum utilization
    async fn inference_worker(
        backend: RpcBackend,
        mut req_rx: mpsc::UnboundedReceiver<(ForwardPassRequest, Option<oneshot::Sender<ForwardPassResponse>>)>,
        mut shutdown_rx: broadcast::Receiver<()>,
        max_batch_tokens: usize,
        max_batch_size: usize,
        scheduler: SharedScheduler,
        max_in_flight_batches: usize,
    ) {
        const REQUEST_TIMEOUT: Duration = Duration::from_secs(300);
        const SCHEDULER_POLL_INTERVAL: Duration = Duration::from_millis(1);

        let mut batch: Vec<(ForwardPassRequest, Option<oneshot::Sender<ForwardPassResponse>>)> = Vec::new();
        let mut total_tokens = 0usize;

        // Track in-flight batch tasks for concurrent execution
        let in_flight_counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (completion_tx, mut completion_rx) = mpsc::unbounded_channel::<(usize, usize, Duration)>();

        loop {
            // Process any completed batches (non-blocking)
            while let Ok((batch_size, tokens_in_batch, latency)) = completion_rx.try_recv() {
                in_flight_counter.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                let mut sched = scheduler.lock().unwrap();
                sched.on_batch_complete(batch_size, tokens_in_batch, latency);
            }

            // If batch is empty, wait for at least one request
            if batch.is_empty() {
                let first_request = tokio::select! {
                    _ = shutdown_rx.recv() => break,
                    maybe_req = req_rx.recv() => {
                        match maybe_req {
                            Some(req) => req,
                            None => break,
                        }
                    }
                };

                // Record arrival and add to batch
                {
                    let mut sched = scheduler.lock().unwrap();
                    let arrival_time = first_request.0.arrival_time.unwrap_or_else(Instant::now);
                    sched.on_request_arrival(arrival_time);
                }
                total_tokens = first_request.0.input_tokens.len();
                batch.push(first_request);
            }

            // Try to accumulate more requests (non-blocking)
            loop {
                match req_rx.try_recv() {
                    Ok(req) => {
                        {
                            let mut sched = scheduler.lock().unwrap();
                            let arrival_time = req.0.arrival_time.unwrap_or_else(Instant::now);
                            sched.on_request_arrival(arrival_time);
                        }
                        total_tokens += req.0.input_tokens.len();
                        batch.push(req);

                        // Check capacity limits
                        if batch.len() >= max_batch_size || total_tokens >= max_batch_tokens {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }

            let in_flight = in_flight_counter.load(std::sync::atomic::Ordering::SeqCst);

            // Check if scheduler recommends firing
            let should_fire = {
                let sched = scheduler.lock().unwrap();
                sched.should_fire(batch.len(), total_tokens, max_batch_size, max_batch_tokens, in_flight)
            };

            // Also check if we're at in-flight limit - if so, must wait for completion
            if should_fire && in_flight < max_in_flight_batches {
                // Fire the batch concurrently (non-blocking)
                let batch_to_fire = std::mem::take(&mut batch);
                let batch_size = batch_to_fire.len();
                let tokens_in_batch = total_tokens;
                total_tokens = 0;

                // Mark batch as fired and increment in-flight counter
                {
                    let mut sched = scheduler.lock().unwrap();
                    sched.on_batch_fired();
                }
                in_flight_counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

                // Spawn batch execution as a background task (concurrent pipelining)
                let backend_clone = backend.clone();
                let completion_tx_clone = completion_tx.clone();
                tokio::spawn(async move {
                    let start_time = Instant::now();
                    Self::execute_forward_pass_batch(&backend_clone, batch_to_fire, REQUEST_TIMEOUT).await;
                    let latency = start_time.elapsed();
                    // Report completion for scheduler feedback
                    completion_tx_clone.send((batch_size, tokens_in_batch, latency)).ok();
                });
            } else if in_flight >= max_in_flight_batches {
                // At in-flight limit - wait for a batch to complete before continuing
                tokio::select! {
                    _ = shutdown_rx.recv() => break,
                    maybe_completion = completion_rx.recv() => {
                        if let Some((batch_size, tokens_in_batch, latency)) = maybe_completion {
                            in_flight_counter.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                            let mut sched = scheduler.lock().unwrap();
                            sched.on_batch_complete(batch_size, tokens_in_batch, latency);
                        }
                    }
                }
            } else {
                // Wait briefly for more requests before checking again
                tokio::select! {
                    _ = shutdown_rx.recv() => break,
                    _ = tokio::time::sleep(SCHEDULER_POLL_INTERVAL) => {}
                    maybe_req = req_rx.recv() => {
                        match maybe_req {
                            Some(req) => {
                                {
                                    let mut sched = scheduler.lock().unwrap();
                                    let arrival_time = req.0.arrival_time.unwrap_or_else(Instant::now);
                                    sched.on_request_arrival(arrival_time);
                                }
                                total_tokens += req.0.input_tokens.len();
                                batch.push(req);
                            }
                            None => break,
                        }
                    }
                }
            }
        }

        // On shutdown, fire any remaining batch and wait for in-flight batches
        if !batch.is_empty() {
            Self::execute_forward_pass_batch(&backend, batch, REQUEST_TIMEOUT).await;
        }

        // Wait for all in-flight batches to complete
        while in_flight_counter.load(std::sync::atomic::Ordering::SeqCst) > 0 {
            if completion_rx.recv().await.is_some() {
                in_flight_counter.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            } else {
                break;
            }
        }
    }

    /// Execute a batch of forward pass requests via fire_batch RPC
    #[tracing::instrument(
        name = "rust.fire_batch",
        skip(backend, requests),
        fields(batch_size = requests.len())
    )]
    async fn execute_forward_pass_batch(
        backend: &RpcBackend,
        requests: Vec<(ForwardPassRequest, Option<oneshot::Sender<ForwardPassResponse>>)>,
        timeout: Duration,
    ) {
        let mut batch_req = BatchedForwardPassRequest::new();
        for (fp_req, _) in &requests {
            batch_req.add_request(fp_req);
        }

        // Inject trace context for cross-language propagation
        // Uses tracing-opentelemetry to get current span context
        use tracing_opentelemetry::OpenTelemetrySpanExt;
        let current_span = tracing::Span::current();
        let context = current_span.context();
        use opentelemetry::trace::TraceContextExt;
        let otel_span = context.span();
        let span_context = otel_span.span_context();
        if span_context.is_valid() {
            // Format as W3C traceparent: version-trace_id-span_id-flags
            let traceparent = format!(
                "00-{}-{}-{:02x}",
                span_context.trace_id(),
                span_context.span_id(),
                span_context.trace_flags().to_u8()
            );
            batch_req.set_trace_context(traceparent);
        }

        let result: Result<BatchedForwardPassResponse, _> = backend
            .call_with_timeout("fire_batch", &batch_req, timeout)
            .await;

        match result {
            Ok(batch_resp) => {
                let mut resp_iter = batch_resp.results.into_iter();
                for (_, resp_tx) in requests {
                    if let Some(tx) = resp_tx {
                        if let Some(resp) = resp_iter.next() {
                            tx.send(resp).ok();
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("[Error] fire_batch failed: {:?}", e);
            }
        }
    }

    /// Execute eager RPC calls (non-batched)
    async fn execute_query(backend: &RpcBackend, req: QueryRequest) -> Option<QueryResponse> {
        const TIMEOUT: Duration = Duration::from_secs(30);
        backend.call_with_timeout("query", &req, TIMEOUT).await.ok()
    }

    pub fn submit(&self, _cmd_queue_id: CmdQueueId, _priority: u32, req: Request) {
        match req {
            Request::ForwardPass(mut fp_req, resp_tx) => {
                // Capture arrival time before queuing to avoid measurement distortion
                // when requests pile up behind the in-flight limit.
                fp_req.arrival_time = Some(Instant::now());
                if self.forward_pass_tx.send((fp_req, resp_tx)).is_err() {
                    eprintln!("[Error] Forward pass channel closed");
                }
            }
            Request::Query(query_req, resp_tx) => {
                let backend_clone = self.backend.clone();
                tokio::spawn(async move {
                    if let Some(resp) = Self::execute_query(&backend_clone, query_req).await {
                        resp_tx.send(resp).ok();
                    }
                });
            }
            Request::EmbedImage(req) => {
                let backend = self.backend.clone();
                tokio::spawn(async move {
                    if let Err(e) = backend.notify("embed_image", &req).await {
                        eprintln!("[Error] embed_image failed: {:?}", e);
                    }
                });
            }
            Request::InitializeAdapter(req) => {
                let backend = self.backend.clone();
                tokio::spawn(async move {
                    if let Err(e) = backend.notify("initialize_adapter", &req).await {
                        eprintln!("[Error] initialize_adapter failed: {:?}", e);
                    }
                });
            }
            Request::UpdateAdapter(req) => {
                let backend = self.backend.clone();
                tokio::spawn(async move {
                    if let Err(e) = backend.notify("update_adapter", &req).await {
                        eprintln!("[Error] update_adapter failed: {:?}", e);
                    }
                });
            }
            Request::UploadAdapter(req) => {
                let backend = self.backend.clone();
                tokio::spawn(async move {
                    if let Err(e) = backend.notify("upload_adapter", &req).await {
                        eprintln!("[Error] upload_adapter failed: {:?}", e);
                    }
                });
            }
            Request::DownloadAdapter(req, resp_tx) => {
                let backend = self.backend.clone();
                tokio::spawn(async move {
                    let result: Result<Vec<u8>, _> = backend
                        .call_with_timeout("download_adapter", &req, Duration::from_secs(60))
                        .await;
                    if let Ok(data) = result {
                        resp_tx.send(Bytes::from(data)).ok();
                    }
                });
            }
            Request::Handshake(_, _) => {
                eprintln!("[Warn] Unexpected handshake request in submit");
            }
            Request::Synchronize(tx) => {
                tx.send(()).ok();
            }
        }
    }

    pub fn runtime_stats(&self) -> HashMap<String, String> {
        let mut stats = HashMap::new();
        stats.insert("model.name".to_string(), self.info.name.clone());
        self.resource_manager.append_stats_to(&mut stats);
        stats
    }

    async fn handle(&mut self, cmd: Command) {
        match cmd {
            Command::Submit { cmd_queue_id, priority, req } => {
                self.submit(cmd_queue_id, priority, req);
            }
            Command::GetInfo { response } => {
                response.send(self.info.clone()).ok();
            }
            Command::GetRuntimeStats { response } => {
                response.send(self.runtime_stats()).ok();
            }
            Command::Allocate { inst_id, type_id, count, response } => {
                match self.resource_manager.allocate_with_oom(inst_id, type_id, count) {
                    Ok(allocated_ids) => { response.send(allocated_ids).ok(); }
                    Err(e) => terminate_instance_with_exception(inst_id, e),
                }
            }
            Command::Deallocate { inst_id, type_id, ptrs } => {
                if let Err(e) = self.resource_manager.deallocate(inst_id, type_id, ptrs) {
                    terminate_instance_with_exception(inst_id, e);
                }
            }
            Command::Cleanup { inst_id } => {
                if let Err(e) = self.resource_manager.cleanup(inst_id) {
                    terminate_instance_with_exception(inst_id, e);
                }
            }
            Command::GetAllExported { type_id, response } => {
                response.send(self.resource_manager.get_all_exported(type_id)).ok();
            }
            Command::Export { inst_id, type_id, ptrs, name } => {
                if let Err(e) = self.resource_manager.export(inst_id, type_id, ptrs, name) {
                    terminate_instance_with_exception(inst_id, e);
                }
            }
            Command::Import { inst_id, type_id, name, response } => {
                match self.resource_manager.import(type_id, name) {
                    Ok(ptrs) => { response.send(ptrs).ok(); }
                    Err(e) => terminate_instance_with_exception(inst_id, e),
                }
            }
            Command::ReleaseExported { inst_id, type_id, name } => {
                if let Err(e) = self.resource_manager.release_exported(type_id, name) {
                    terminate_instance_with_exception(inst_id, e);
                }
            }
            // Actor Commands (stubs)
            Command::ActorGlobalContextRef { .. } |
            Command::ActorGlobalContextDestroy { .. } |
            Command::ActorGlobalContextExtend { .. } |
            Command::ActorGlobalContextTrim { .. } |
            Command::ActorAdapterRef { .. } |
            Command::ActorAdapterDestroy { .. } |
            Command::ActorAdapterBlank { .. } |
            Command::ActorAdapterLoad { .. } |
            Command::ActorOptimizerRef { .. } |
            Command::ActorOptimizerDestroy { .. } |
            Command::ActorOptimizerLoad { .. } |
            Command::ActorOptimizerSave { .. } |
            Command::ActorOptimizerInitialize { .. } |
            Command::ActorOptimizerUpdate { .. } => {}
            Command::ActorGlobalContextRead { response, .. } => {
                response.send(vec![]).ok();
            }
        }
    }
}
