pub mod actor;
pub mod batching;
pub mod request;
pub mod resource;
pub mod tokenizer;

use super::model::batching::{BatchPolicySelector, BatchScheduler, ThresholdPolicy};
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
use pycrust_client::RpcClient;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::sync::{broadcast, mpsc, oneshot, Notify};
use tokio::task::{self, JoinHandle};

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

/// Model service using pycrust RPC for communication with Python backend.
pub struct Model {
    info: ModelInfo,
    resource_manager: ResourceManager,
    shutdown_tx: broadcast::Sender<()>,
    rpc_client: Arc<RpcClient>,
    /// Channel for forward pass requests (only forward pass is batched)
    forward_pass_tx: mpsc::UnboundedSender<(ForwardPassRequest, Option<oneshot::Sender<ForwardPassResponse>>)>,
    scheduling_worker_handle: Option<JoinHandle<()>>,
    backend_worker_handle: Option<JoinHandle<()>>,
}

impl Model {
    pub async fn new(service_name: &str) -> Result<Self> {
        let rpc_client = Arc::new(
            RpcClient::connect(service_name)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to connect to pycrust service: {}", e))?,
        );

        let handshake_info = Self::handshake(&rpc_client).await?;

        let forward_pass_policy = ThresholdPolicy::t_only(Duration::from_micros(50));
        let batch_policy = BatchPolicySelector::new(forward_pass_policy);

        let (forward_pass_tx, forward_pass_rx) = mpsc::unbounded_channel();
        let (batch_tx, batch_rx) = mpsc::unbounded_channel();
        let (shutdown_tx, shutdown_rx) = broadcast::channel(1);

        let scheduler_notify = Arc::new(Notify::new());

        let backend_worker_handle = tokio::spawn(Self::backend_worker(
            Arc::clone(&rpc_client),
            batch_rx,
            scheduler_notify.clone(),
            shutdown_rx,
        ));

        let scheduling_worker_handle = tokio::spawn(Self::scheduling_worker(
            batch_policy,
            forward_pass_rx,
            batch_tx,
            scheduler_notify,
            shutdown_tx.subscribe(),
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
            rpc_client,
            forward_pass_tx,
            shutdown_tx,
            scheduling_worker_handle: Some(scheduling_worker_handle),
            backend_worker_handle: Some(backend_worker_handle),
        })
    }

    async fn handshake(rpc_client: &RpcClient) -> Result<HandshakeResponse> {
        const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(30);
        let req = HandshakeRequest { version: "0.1.0".to_string() };
        let response: HandshakeResponse = rpc_client
            .call_with_timeout("handshake", &req, HANDSHAKE_TIMEOUT)
            .await
            .map_err(|e| anyhow::anyhow!("Handshake failed: {}", e))?;
        Ok(response)
    }

    pub async fn shutdown(mut self) -> Result<()> {
        self.shutdown_tx.send(())?;
        if let Some(handle) = self.scheduling_worker_handle.take() {
            handle.await?;
        }
        if let Some(handle) = self.backend_worker_handle.take() {
            handle.await?;
        }
        Ok(())
    }

    /// Scheduling worker - batches only ForwardPass requests
    async fn scheduling_worker(
        batch_policy: BatchPolicySelector,
        mut req_rx: mpsc::UnboundedReceiver<(ForwardPassRequest, Option<oneshot::Sender<ForwardPassResponse>>)>,
        batch_tx: mpsc::UnboundedSender<Vec<(ForwardPassRequest, Option<oneshot::Sender<ForwardPassResponse>>)>>,
        scheduler_notify: Arc<Notify>,
        mut shutdown_rx: broadcast::Receiver<()>,
    ) {
        let mut sched: BatchScheduler<(ForwardPassRequest, Option<oneshot::Sender<ForwardPassResponse>>)> =
            BatchScheduler::new(batch_policy);
        let mut next_poll_duration: Option<Duration> = None;
        const IDLE_TIMEOUT: Duration = Duration::from_secs(1);

        loop {
            let sleep_duration = next_poll_duration.unwrap_or(IDLE_TIMEOUT);

            tokio::select! {
                _ = shutdown_rx.recv() => break,
                maybe_msg = req_rx.recv() => {
                    if let Some((req, resp_tx)) = maybe_msg {
                        sched.push(0, 0, (req, resp_tx), Instant::now());
                    } else {
                        break;
                    }
                },
                _ = scheduler_notify.notified() => {},
                _ = tokio::time::sleep(sleep_duration) => {}
            }

            let batches = sched.schedule(Instant::now());
            for batch in batches {
                batch_tx.send(batch).ok();
            }
            next_poll_duration = sched.next_poll_in(Instant::now());
        }
    }

    /// Backend worker - handles only batched ForwardPass requests
    async fn backend_worker(
        rpc_client: Arc<RpcClient>,
        mut batch_rx: mpsc::UnboundedReceiver<Vec<(ForwardPassRequest, Option<oneshot::Sender<ForwardPassResponse>>)>>,
        scheduler_notify: Arc<Notify>,
        mut shutdown_rx: broadcast::Receiver<()>,
    ) {
        const REQUEST_TIMEOUT: Duration = Duration::from_secs(300);

        loop {
            tokio::select! {
                _ = shutdown_rx.recv() => break,
                maybe_batch = batch_rx.recv() => {
                    if let Some(requests) = maybe_batch {
                        Self::execute_forward_pass_batch(&rpc_client, requests, REQUEST_TIMEOUT).await;
                        scheduler_notify.notify_one();
                    } else {
                        break;
                    }
                }
            }
        }
    }

    /// Execute a batch of forward pass requests via fire_batch RPC
    async fn execute_forward_pass_batch(
        rpc_client: &RpcClient,
        requests: Vec<(ForwardPassRequest, Option<oneshot::Sender<ForwardPassResponse>>)>,
        timeout: Duration,
    ) {
        let mut batch_req = BatchedForwardPassRequest::new();
        for (fp_req, _) in &requests {
            batch_req.add_request(fp_req);
        }

        let result: Result<BatchedForwardPassResponse, _> = rpc_client
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
    async fn execute_query(rpc_client: &RpcClient, req: QueryRequest) -> Option<QueryResponse> {
        const TIMEOUT: Duration = Duration::from_secs(30);
        rpc_client.call_with_timeout("query", &req, TIMEOUT).await.ok()
    }

    pub fn submit(&self, _cmd_queue_id: CmdQueueId, _priority: u32, req: Request) {
        match req {
            Request::ForwardPass(fp_req, resp_tx) => {
                if self.forward_pass_tx.send((fp_req, resp_tx)).is_err() {
                    eprintln!("[Error] Forward pass channel closed");
                }
            }
            Request::Query(query_req, resp_tx) => {
                let rpc_client = Arc::clone(&self.rpc_client);
                tokio::spawn(async move {
                    if let Some(resp) = Self::execute_query(&rpc_client, query_req).await {
                        resp_tx.send(resp).ok();
                    }
                });
            }
            Request::EmbedImage(req) => {
                let rpc_client = Arc::clone(&self.rpc_client);
                tokio::spawn(async move {
                    let _: Result<(), _> = rpc_client
                        .call_with_timeout("embed_image", &req, Duration::from_secs(60))
                        .await;
                });
            }
            Request::InitializeAdapter(req) => {
                let rpc_client = Arc::clone(&self.rpc_client);
                tokio::spawn(async move {
                    let _: Result<(), _> = rpc_client
                        .call_with_timeout("initialize_adapter", &req, Duration::from_secs(60))
                        .await;
                });
            }
            Request::UpdateAdapter(req) => {
                let rpc_client = Arc::clone(&self.rpc_client);
                tokio::spawn(async move {
                    let _: Result<(), _> = rpc_client
                        .call_with_timeout("update_adapter", &req, Duration::from_secs(60))
                        .await;
                });
            }
            Request::UploadAdapter(req) => {
                let rpc_client = Arc::clone(&self.rpc_client);
                tokio::spawn(async move {
                    let _: Result<(), _> = rpc_client
                        .call_with_timeout("upload_adapter", &req, Duration::from_secs(60))
                        .await;
                });
            }
            Request::DownloadAdapter(req, resp_tx) => {
                let rpc_client = Arc::clone(&self.rpc_client);
                tokio::spawn(async move {
                    let result: Result<Vec<u8>, _> = rpc_client
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
