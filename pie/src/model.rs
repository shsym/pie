use crate::backend::{Backend, BackendError};
use crate::batching::{BatchingConfig, MultiStreamBatcher};
use crate::batching_old::Batcher;
use crate::handler::{
    BatchSchedulingPolicy, Handler, HandshakeRequest, HandshakeResponse, get_batching_config,
};
use crate::instance::Id as InstanceId;
use crate::model_old::{Event, Stream};
use crate::runtime::{TerminationCause, trap, trap_exception};
use crate::service::{self, Service, ServiceError};
use crate::utils::IdPool;
use bytes::{Buf, Bytes};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::cmp::PartialEq;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet, VecDeque};
use std::future;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};
use tokio::sync::mpsc::{Receiver, UnboundedReceiver, UnboundedSender, unbounded_channel};
use tokio::sync::{Mutex, mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio::time::timeout;
use zeromq::{DealerSocket, Socket, SocketRecv, SocketSend, ZmqMessage};

static SERVICE_ID_KVS: OnceLock<usize> = OnceLock::new();

/// Dispatches a command to the key-value store service.
pub fn dispatch(command: Command) -> Result<(), ServiceError> {
    let service_id = *SERVICE_ID_KVS.get_or_init(|| service::get_service_id("resource").unwrap());
    service::dispatch(service_id, command)
}

pub type ResourceId = u32;
pub type ResourceTypeId = u32;

pub type HandlerId = u32;

pub type CmdQueueId = u32;

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

    Allocate {
        inst_id: InstanceId,
        type_id: ResourceTypeId,
        count: usize,
        response: oneshot::Sender<Vec<ResourceId>>, // physical ids
    },

    Deallocate {
        inst_id: InstanceId,
        type_id: ResourceTypeId,
        ptrs: Vec<ResourceId>, // physical ids
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

/// An in-memory key-value store service.
#[derive(Debug)]
pub struct Model {
    res_pool: HashMap<ResourceTypeId, IdPool<ResourceId>>,
    res_exported: HashMap<ResourceTypeId, HashMap<String, Vec<ResourceId>>>,
    res_allocated: HashMap<(ResourceTypeId, InstanceId), HashSet<ResourceId>>,

    // heuristic used by oom killer
    inst_start_time: HashMap<InstanceId, Instant>,

    // event loops
    scheduler_tx: UnboundedSender<(Handler, CmdQueueId, Option<oneshot::Sender<Bytes>>, Bytes)>,
    scheduling_worker_handle: JoinHandle<()>,
    backend_worker_handle: JoinHandle<()>,
}

impl Model {
    pub async fn new(endpoint: &str) -> anyhow::Result<Self> {
        // Use the connection helper with spinner from zmq_handler
        let mut socket = DealerSocket::new();
        socket.connect(endpoint).await?;

        // iterate through all handler enums
        let mut batching_config = get_batching_config();
        let mut batch_triggers = HashMap::new();
        for (handler, cfg) in batching_config.iter_mut() {
            match cfg {
                BatchingConfig::Triggered {
                    mut trigger,
                    min_wait_time,
                } => {
                    let t = Arc::new(AtomicBool::new(true));
                    batch_triggers.insert(handler.get_handler_id(), t.clone());
                    trigger = Some(t.clone());
                }
                _ => {}
            }
        }

        let (backend_tx, backend_rx) = unbounded_channel();
        let (scheduler_tx, scheduler_rx) = unbounded_channel();

        // Spawn the event loop task.
        let backend_worker_handle =
            tokio::spawn(Self::backend_worker(socket, backend_rx, batch_triggers));

        let scheduling_worker_handle = tokio::spawn(Self::scheduling_worker(
            batching_config,
            scheduler_rx,
            backend_tx,
        ));

        // do the handshake
        let (handshake_tx, handshake_rx) = oneshot::channel();

        scheduler_tx.send((
            Handler::Handshake,
            0,
            Some(handshake_tx),
            Bytes::from(
                rmp_serde::to_vec_named(&HandshakeRequest {
                    version: "0.1.0".to_string(),
                })
                .unwrap(),
            ),
        ))?;

        let model_info = rmp_serde::from_slice::<HandshakeResponse>(&handshake_rx.await?.to_vec())?;

        let mut res_pool = HashMap::new();

        for (res_id, capacity) in model_info.resources.iter() {
            res_pool
                .entry(*res_id)
                .or_insert_with(|| IdPool::new(*capacity));
        }

        Ok(Model {
            res_pool,
            res_exported: HashMap::new(),
            res_allocated: HashMap::new(),
            inst_start_time: HashMap::new(),
            scheduler_tx,
            scheduling_worker_handle,
            backend_worker_handle,
        })
    }

    async fn scheduling_worker(
        config: HashMap<Handler, BatchingConfig>,
        mut rx: UnboundedReceiver<(Handler, CmdQueueId, Option<oneshot::Sender<Bytes>>, Bytes)>,
        backend_tx: UnboundedSender<(Handler, Vec<(Option<oneshot::Sender<Bytes>>, Bytes)>)>,
    ) {
        let mut batcher = MultiStreamBatcher::new(config.clone());
        let mut next_poll_duration: Option<Duration> = None;

        // A default timeout to use when the batcher has no pending items.
        // This prevents a busy-loop while still allowing the scheduler to be responsive.
        const IDLE_TIMEOUT: Duration = Duration::from_secs(1);

        loop {
            // Use the specific duration from the batcher, or fall back to the idle timeout.
            let sleep_duration = next_poll_duration.unwrap_or(IDLE_TIMEOUT);

            tokio::select! {
                // Branch 1: A new message is received.
                maybe_msg = rx.recv() => {
                    if let Some((handler, cmd_queue_id, sender, msg)) = maybe_msg {
                        // Eagerly send if no batching config exists for this handler.
                        if !config.contains_key(&handler) {
                            let _ = backend_tx.send((handler, vec![(sender, msg)]));
                            continue;
                        }

                        // Otherwise, add the command to the batcher.
                        batcher.push(cmd_queue_id, handler, (sender, msg), Instant::now());
                    } else {
                        // Channel closed, shut down the scheduler.
                        println!("[Info] Command channel closed, shutting down scheduler handler.");
                        break;
                    }
                },
                // Branch 2: The timeout expires.
                _ = tokio::time::sleep(sleep_duration) => {
                    // Timeout fired. Proceed to the polling logic below.
                }
            }

            // Poll the batcher for any ready batches after either event.
            let batch = batcher.poll(Instant::now());

            for (batch_handler, batch_payload) in batch {
                if !batch_payload.is_empty() {
                    let _ = backend_tx.send((batch_handler, batch_payload));
                }
            }

            // Recalculate the next required timeout for the next loop iteration.
            next_poll_duration = batcher.next_poll_in(Instant::now());
        }
    }

    /// Manages communication with a backend service over a ZMQ DEALER socket.
    ///
    /// This function runs an event loop that performs two main tasks concurrently:
    /// 1.  Receives commands from an MPSC channel, bundles them into a ZMQ message,
    ///     and sends them to the backend. It registers reply handlers (`oneshot::Sender`)
    ///     to await the responses.
    /// 2.  Receives multipart ZMQ messages from the backend, parses them, and uses a
    ///     correlation ID to route the reply payloads to the correct waiting handlers.
    ///
    /// # Message Formats
    ///
    /// ## Outgoing (Client -> Server)
    /// * Frame 0: `corr_id` (4 bytes, u32 big-endian)
    /// * Frame 1: `handler_id` (4 bytes, u32 big-endian)
    /// * Frame 2...N: `payload_bytes`
    ///
    /// ## Incoming (Server -> Client)
    /// * Frame 0: `corr_id` (4 bytes, u32 big-endian)
    /// * Frame 1...N: `reply_payload_bytes`
    ///
    async fn backend_worker(
        mut socket: DealerSocket,
        mut rx: UnboundedReceiver<(Handler, Vec<(Option<oneshot::Sender<Bytes>>, Bytes)>)>,
        batch_triggers: HashMap<HandlerId, Arc<AtomicBool>>,
    ) {
        // A correlation ID to match outgoing requests with incoming replies.
        // wrapping_add ensures it safely wraps around on overflow instead of panicking.
        let mut corr_id: u32 = 0;

        // Stores the `oneshot` senders to send replies back to the original requesters.
        // The key is a tuple of (correlation_id, batch_index).
        let mut event_table: HashMap<(u32, usize), oneshot::Sender<Bytes>> = HashMap::new();

        loop {
            tokio::select! {
                // Branch 1: Handle outgoing commands received from other parts of the application.
                maybe_command = rx.recv() => {
                    if let Some((handler, payload)) = maybe_command {
                        let current_corr_id = corr_id;

                        // Unzip payload into two parts
                        let (senders, batch): (Vec<_>, Vec<_>) = payload.into_iter().unzip();

                        // Construct the multipart ZMQ message.
                        let mut frames: VecDeque<Bytes> = VecDeque::with_capacity(2 + batch.len());
                        frames.push_back(Bytes::copy_from_slice(&current_corr_id.to_be_bytes()));
                        frames.push_back(Bytes::copy_from_slice(&handler.get_handler_id().to_be_bytes()));
                        frames.extend(batch);

                        let zmq_msg = match ZmqMessage::try_from(frames) {
                            Ok(msg) => msg,
                            Err(e) => {
                                eprintln!("Failed to construct ZMQ message: {:?}", e);
                                continue;
                            }
                        };

                        // Send the message. If it fails, the `senders` are dropped,
                        // which notifies the receivers that the operation failed.
                        if let Err(e) = socket.send(zmq_msg).await {
                            eprintln!("[Error] Socket send failed for corr_id {}: {:?}", current_corr_id, e);
                            continue; // Do not proceed to register handlers.
                        }

                        // If sending was successful, register the reply handlers.
                        for (idx, sender) in senders.into_iter().enumerate() {
                            if let Some(sender) = sender {
                                event_table.insert((current_corr_id, idx), sender);
                            }
                        }

                        // Increment the correlation ID for the next message.
                        corr_id = corr_id.wrapping_add(1);

                    } else {
                        // The channel is closed, indicating the application is shutting down.
                        println!("[Info] Command channel closed, shutting down backend handler.");
                        break;
                    }
                },

                // Branch 2: Handle incoming replies from the backend server.
                result = socket.recv() => {
                    match result {
                        Ok(zmq_msg) => {
                            // We expect at least a correlation ID frame.
                            if zmq_msg.is_empty() {
                                eprintln!("[Warn] Received an empty ZMQ message. Ignoring.");
                                continue;
                            }
                            let mut frames = zmq_msg.into_vecdeque();

                            // The first frame is the correlation ID. ZmqMessage acts like a VecDeque.
                            let corr_id_bytes = frames.pop_front().unwrap();
                            let handler_id_bytes = frames.pop_front().unwrap();

                            // Parse the correlation ID from the first frame.
                            let received_corr_id = u32::from_be_bytes(corr_id_bytes.as_ref().try_into().unwrap());
                            let received_handler_id = u32::from_be_bytes(handler_id_bytes.as_ref().try_into().unwrap());

                            // The remaining frames are payloads. We iterate through them and
                            // find the corresponding `oneshot::Sender` to send the reply back.
                            for (idx, payload) in frames.into_iter().enumerate() {
                                let key = (received_corr_id, idx);
                                if let Some(sender) = event_table.remove(&key) {
                                    // Send the payload. We don't care if the receiver was dropped.
                                    let _ = sender.send(payload);
                                }
                            }

                            // Update the trigger if it exists
                            if let Some(trigger) = batch_triggers.get(&received_handler_id) {
                                trigger.store(true, std::sync::atomic::Ordering::SeqCst);
                            }

                        },
                        Err(e) => {
                            eprintln!("[Error] Socket receive error: {}. Shutting down.", e);
                            break;
                        }
                    }
                }
            }
        }
    }

    pub fn submit(&self) {}

    pub fn available(&self, type_id: ResourceTypeId) -> anyhow::Result<usize> {
        let pool = self.res_pool.get(&type_id).ok_or(anyhow::anyhow!(
            "Resource pool for type {:?} does not exist",
            type_id
        ))?;
        Ok(pool.available())
    }

    pub fn allocate(
        &mut self,
        inst_id: InstanceId,
        type_id: ResourceTypeId,
        count: usize,
    ) -> anyhow::Result<Vec<ResourceId>> {
        let pool = self
            .res_pool
            .entry(type_id)
            .or_insert_with(|| IdPool::new(ResourceId::MAX));

        if pool.available() < count {
            return Err(anyhow::anyhow!("Out of memory"));
        }

        let allocated = pool.acquire_many(count)?;

        // Record the start time if this is the first allocation for this instance.
        self.inst_start_time
            .entry(inst_id)
            .or_insert_with(Instant::now);

        // update the allocated map
        self.res_allocated
            .entry((type_id, inst_id))
            .or_insert_with(HashSet::new)
            .extend(&allocated);

        Ok(allocated)
    }

    pub fn deallocate(
        &mut self,
        inst_id: InstanceId,
        type_id: ResourceTypeId,
        ptrs: Vec<ResourceId>,
    ) -> anyhow::Result<()> {
        let allocated = self
            .res_allocated
            .get_mut(&(type_id, inst_id))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Instance {:?} has no allocated resources of type {:?}",
                    inst_id,
                    type_id
                )
            })?;

        let pool = self.res_pool.get_mut(&type_id).ok_or_else(|| {
            anyhow::anyhow!("Resource pool for type {:?} does not exist", type_id)
        })?;

        for ptr in ptrs {
            if allocated.remove(&ptr) {
                pool.release(ptr)?;
            } else {
                return Err(anyhow::anyhow!(
                    "Pointer {:?} is not allocated to instance {:?}",
                    ptr,
                    inst_id
                ));
            }
        }

        Ok(())
    }

    /// Finds and terminates victim instances until the requested `size` of resources of `ty` is available.
    pub fn oom_kill(
        &mut self,
        type_id: ResourceTypeId,
        size: usize,
        inst_id_to_exclude: InstanceId,
    ) -> anyhow::Result<()> {
        // 1. Get the start time of the instance requesting memory.
        // If it has no start time (i.e., no resources), no other instance can be "newer".
        let requester_start_time = match self.inst_start_time.get(&inst_id_to_exclude) {
            Some(time) => *time,
            None => {
                return Err(anyhow::anyhow!(
                    "OOM unrecoverable: Requesting instance has no recorded start time, so no newer instances can be killed."
                ));
            }
        };

        // 2. Loop until enough resources are freed.
        loop {
            // Check available memory, handling potential errors if the resource pool doesn't exist.
            let available_count = self.available(type_id)?;
            if available_count >= size {
                break; // Success, enough memory is now free.
            }

            // 3. Heuristic: Find the newest instance that is *strictly newer* than the requester.
            let victim_id = self
                .inst_start_time
                .iter()
                // The key change: only consider instances started AFTER the requester.
                .filter(|(_, time)| **time > requester_start_time)
                // Find the newest among the filtered candidates.
                .max_by_key(|(_, time)| **time)
                .map(|(id, _)| *id);

            if let Some(victim_id) = victim_id {
                // A victim was found, clean up its resources.
                self.cleanup(victim_id)?;

                // Terminate the victim instance.
                trap(
                    victim_id,
                    TerminationCause::OutOfResources(
                        "Terminated by OOM killer to free resources for an older instance"
                            .to_string(),
                    ),
                );
            } else {
                // No more instances newer than the requester could be found.
                return Err(anyhow::anyhow!(
                    "OOM unrecoverable: Not enough memory could be freed after terminating all newer instances."
                ));
            }
        }

        Ok(())
    }

    /// Deallocates all resources associated with a given instance.
    pub fn cleanup(&mut self, inst_id: InstanceId) -> anyhow::Result<()> {
        let mut to_release_by_type: HashMap<ResourceTypeId, Vec<ResourceId>> = HashMap::new();

        // `retain` iterates and removes entries for which the closure returns false.
        self.res_allocated.retain(|(ty, id), ptrs| {
            if *id == inst_id {
                // Collect the pointers to be released for the matching instance.
                to_release_by_type
                    .entry(*ty)
                    .or_default()
                    .extend(ptrs.iter().copied());
                // Return `false` to remove this entry from `self.allocated`.
                false
            } else {
                // Keep entries for other instances.
                true
            }
        });

        // Release the collected pointers back to their respective pools.
        for (ty, ptrs) in to_release_by_type {
            let pool = self.res_pool.get_mut(&ty).ok_or_else(|| {
                anyhow::anyhow!("Cleanup failed: Resource pool for type {:?} not found", ty)
            })?;

            for ptr in ptrs {
                pool.release(ptr)?;
            }
        }

        // Remove the instance's start time entry.
        self.inst_start_time.remove(&inst_id);

        Ok(())
    }

    pub fn export(
        &mut self,
        inst_id: InstanceId,
        type_id: ResourceTypeId,
        ptrs: Vec<ResourceId>,
        name: String,
    ) -> anyhow::Result<()> {
        // We need a mutable reference to the allocated set to remove pointers.
        let allocated = self
            .res_allocated
            .get_mut(&(type_id, inst_id))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Instance {:?} has no allocated resources of type {:?}",
                    inst_id,
                    type_id
                )
            })?;

        // 1. Membership check: Ensure the instance owns all pointers before transferring them.
        for ptr in &ptrs {
            if !allocated.contains(ptr) {
                return Err(anyhow::anyhow!(
                    "Invalid pointer {:?} for instance {:?}",
                    ptr,
                    inst_id
                ));
            }
        }

        // 2. Add the resource to the public export map, but only if the name isn't taken.
        // We get or create the nested map for the given resource type first.
        let type_exports = self.res_exported.entry(type_id).or_default();

        // The key change is here: using the Entry API.
        match type_exports.entry(name) {
            // This arm runs if the name is already in use.
            Entry::Occupied(entry) => {
                // Return an error to prevent overwriting and leaking the old resource.
                Err(anyhow::anyhow!(
                    "Exported resource with name '{}' already exists for type {:?}",
                    entry.key(),
                    type_id
                ))
            }
            // This arm runs if the name is available.
            Entry::Vacant(entry) => {
                // 3. After validation, remove the pointers from the instance's ownership.
                // This is done only after we've confirmed the export name is available.
                for ptr in &ptrs {
                    allocated.remove(ptr);
                }

                // 4. Safely insert the new exported resource.
                entry.insert(ptrs);
                Ok(())
            }
        }
    }

    pub fn import(
        &mut self,
        type_id: ResourceTypeId,
        name: String,
    ) -> anyhow::Result<Vec<ResourceId>> {
        let ptrs = self
            .res_exported
            .get(&type_id)
            .and_then(|exports| exports.get(&name))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Exported resource '{}' not found for type {:?}",
                    name,
                    type_id
                )
            })?
            .clone();

        Ok(ptrs)
    }

    pub fn release_exported(
        &mut self,
        type_id: ResourceTypeId,
        name: String,
    ) -> anyhow::Result<()> {
        let type_exports = self.res_exported.get_mut(&type_id).ok_or_else(|| {
            anyhow::anyhow!("No resources of type {:?} have been exported", type_id)
        })?;

        // Try to remove the named resource.
        if let Some(ptrs_to_release) = type_exports.remove(&name) {
            // If found, get the corresponding resource pool.
            let pool = self.res_pool.get_mut(&type_id).ok_or_else(|| {
                anyhow::anyhow!("Resource pool for type {:?} does not exist", type_id)
            })?;

            // Release each pointer back into the pool.
            for ptr in ptrs_to_release {
                pool.release(ptr)?;
            }
            Ok(())
        } else {
            Err(anyhow::anyhow!(
                "Exported resource with name '{}' not found",
                name
            ))
        }
    }
}

impl Service for Model {
    type Command = Command;

    async fn handle(&mut self, cmd: Self::Command) {
        match cmd {
            Command::Allocate {
                inst_id,
                type_id: ty,
                count,
                response,
            } => {
                // Check if there is enough memory before attempting to allocate.
                if self.available(ty).unwrap_or(0) < count {
                    // Not enough memory, trigger the OOM killer.
                    if let Err(kill_err) = self.oom_kill(ty, count, inst_id) {
                        // OOM killing failed because it couldn't free enough memory. Trap the requester.
                        trap_exception(inst_id, kill_err);
                        return;
                    }
                }

                // Proceed with the allocation. A successful oom_kill guarantees enough space.
                match self.allocate(inst_id, ty, count) {
                    Ok(v) => {
                        let _ = response.send(v);
                    }
                    Err(e) => {
                        // This path is defensive; it shouldn't be reached if oom_kill succeeded.
                        trap_exception(inst_id, e);
                    }
                }
            }

            Command::Deallocate {
                inst_id,
                type_id: ty,
                ptrs,
            } => {
                if let Err(e) = self.deallocate(inst_id, ty, ptrs) {
                    trap_exception(inst_id, e);
                }
            }

            Command::Cleanup { inst_id } => {
                if let Err(e) = self.cleanup(inst_id) {
                    trap_exception(inst_id, e);
                }
            }

            Command::GetAllExported {
                type_id: ty,
                response,
            } => {
                let list = self
                    .res_exported
                    .get(&ty)
                    .map(|exports| {
                        // Clone each name and its associated Vec<ResourceId> just once.
                        exports
                            .iter()
                            .map(|(name, ptrs)| (name.clone(), ptrs.clone()))
                            .collect()
                    })
                    .unwrap_or_default();
                let _ = response.send(list);
            }

            Command::Export {
                inst_id,
                type_id: ty,
                ptrs,
                name,
            } => {
                if let Err(e) = self.export(inst_id, ty, ptrs, name) {
                    trap_exception(inst_id, e);
                }
            }

            Command::Import {
                inst_id,
                type_id: ty,
                name,
                response,
            } => match self.import(ty, name) {
                Ok(ptr) => {
                    let _ = response.send(ptr);
                }
                Err(e) => trap_exception(inst_id, e),
            },

            Command::ReleaseExported {
                inst_id,
                type_id: ty,
                name,
            } => {
                if let Err(e) = self.release_exported(ty, name) {
                    trap_exception(inst_id, e);
                }
            }
        }
    }
}
