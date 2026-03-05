//! # Context Module
//!
//! Manages named execution contexts with KV cache state for model inference.
//!
//! Each model gets a dedicated ContextManager actor that handles:
//! - Context creation, destruction, and forking
//! - Page management (commit, reserve, release)
//! - Contention resolution via dual-queue protocol
//!
//! ## Storage Split
//!
//! - `ContextBuffer` (global DashMap) — token-level data accessed by WIT host
//!   functions without going through the actor: tokens_filled, tokens_buffered,
//!   committed_len, max_committed_position.
//!
//! - `Context` (local HashMap on ContextManager) — all KV/page state accessed
//!   exclusively within the actor: working_pages, committed_tip, lineage, state,
//!   device, etc.
pub mod pagestore;
pub use pagestore::compute_last_page_len;
pub(crate) mod arbiter;
mod contention;
mod restore;

use dashmap::DashMap;
use std::collections::{HashMap, VecDeque, BinaryHeap};
use std::sync::LazyLock;
use std::time::Instant;
use tokio::sync::oneshot;
use anyhow::{Result, Context as _};

use crate::service::{ServiceArray, ServiceHandler};
use crate::adapter::AdapterId;
use crate::inference::brle::Brle;
use crate::inference::request::{ForwardPassRequest, BatchedForwardPassRequest};
use crate::process::ProcessId;
use crate::device::{self, DeviceId};

use pagestore::{PhysicalPageId, PageHash, PageStore};
use arbiter::Arbiter;
use contention::{AllocWaiter, RestoreWaiter};

// =============================================================================
// Public Types
// =============================================================================

pub type ContextId = u64;

// =============================================================================
// Globals
// =============================================================================

static SERVICES: LazyLock<ServiceArray<Message>> = LazyLock::new(ServiceArray::new);
static BUFFERS: LazyLock<DashMap<(usize, ContextId), ContextBuffer>> = LazyLock::new(DashMap::new);
static PAGE_SIZES: LazyLock<boxcar::Vec<usize>> = LazyLock::new(boxcar::Vec::new);

// =============================================================================
// Public API
// =============================================================================

/// Spawns a new context manager for a model.
pub fn spawn(page_size: usize, num_gpu_pages: Vec<usize>, num_cpu_pages: Vec<usize>) -> usize {
    PAGE_SIZES.push(page_size);
    SERVICES.spawn(move || ContextManager::new(
        SERVICES.len().saturating_sub(1), page_size, &num_gpu_pages, &num_cpu_pages,
    )).expect("Failed to spawn context manager")
}

// ---------- Actor-routed ----------

pub async fn open(model_idx: usize, username: String, name: String) -> Result<ContextId> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Open { username, name, response: tx })?;
    rx.await.context("context::open: actor dropped response")?
}

pub async fn create(model_idx: usize) -> Result<ContextId> {
    create_owned(model_idx, None).await
}

pub async fn create_owned(model_idx: usize, owner: Option<ProcessId>) -> Result<ContextId> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Create { owner, response: tx })?;
    rx.await.context("context::create_owned: actor dropped response")?
}

pub async fn save(model_idx: usize, id: ContextId, username: String, name: String) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Save { id, username, name, response: tx })?;
    rx.await.context("context::save: actor dropped response")?
}

pub async fn snapshot(model_idx: usize, id: ContextId, username: String) -> Result<String> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Snapshot { id, username, response: tx })?;
    rx.await.context("context::snapshot: actor dropped response")?
}

pub async fn delete(model_idx: usize, username: String, name: String) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Delete { username, name, response: tx })?;
    rx.await.context("context::delete: actor dropped response")?
}

pub async fn destroy(model_idx: usize, id: ContextId, force: bool) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Destroy { id, force, response: tx })?;
    rx.await.context("context::destroy: actor dropped response")?
}

pub async fn fork(model_idx: usize, id: ContextId) -> Result<ContextId> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Fork { id, response: tx })?;
    rx.await.context("context::fork: actor dropped response")?
}

pub async fn commit_pages(model_idx: usize, id: ContextId, page_indices: Vec<u32>) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::CommitPages { id, page_indices, response: tx })?;
    rx.await.context("context::commit_pages: actor dropped response")?
}

pub async fn reserve_pages(model_idx: usize, id: ContextId, num_pages: u32) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::ReservePages { id, num_pages, response: tx })?;
    rx.await.context("context::reserve_pages: actor dropped response")?
}

pub fn release_pages(model_idx: usize, id: ContextId, num_pages: u32) -> Result<()> {
    SERVICES.send(model_idx, Message::ReleasePages { id, num_pages })
}

pub async fn get_physical_page_ids(model_idx: usize, id: ContextId) -> Result<HashMap<DeviceId, Vec<PhysicalPageId>>> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::GetPhysicalPageIds { id, response: tx })?;
    rx.await.context("context::get_physical_page_ids: actor dropped response")?
}


pub async fn commit_replay_chunk(
    model_idx: usize, id: ContextId, num_pages: u32,
    tokens: Vec<u32>, positions: Vec<u32>, masks: Vec<Brle>, adapter: Option<AdapterId>,
) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::CommitReplayFill {
        id, num_pages, tokens, positions, masks, adapter, response: tx,
    })?;
    rx.await.context("context::commit_replay_chunk: actor dropped response")?
}

pub fn finish_restore(model_idx: usize, id: ContextId) -> Result<()> {
    SERVICES.send(model_idx, Message::FinishRestore { id })
}

/// Unpin context: Pinned → Active. Fire-and-forget actor message.
/// Also executes deferred suspension if `pending_suspend` was set.
pub fn clear_pinned(model_idx: usize, id: ContextId) {
    let _ = SERVICES.send(model_idx, Message::ClearPinned { id });
}

pub async fn get_stats(model_idx: usize) -> Vec<(usize, usize)> {
    let (tx, rx) = oneshot::channel();
    let _ = SERVICES.send(model_idx, Message::GetStats { response: tx });
    rx.await.unwrap_or_default()
}

pub async fn is_active(model_idx: usize, id: ContextId) -> bool {
    let (tx, rx) = oneshot::channel();
    let _ = SERVICES.send(model_idx, Message::IsActive { id, response: tx });
    rx.await.unwrap_or(false)
}

pub async fn debug_context_state(model_idx: usize, id: ContextId) -> String {
    let (tx, rx) = oneshot::channel();
    let _ = SERVICES.send(model_idx, Message::DebugState { id, response: tx });
    rx.await.unwrap_or_else(|_| "MISSING".to_string())
}

// ---------- Arbiter policy (broadcast to all models) ----------

pub fn set_dag_weights(weight: f64, pid_values: HashMap<ProcessId, f64>) {
    for model_idx in 0..SERVICES.len() {
        let _ = SERVICES.send(model_idx, Message::SetDagWeights {
            weight, pid_values: pid_values.clone(),
        });
    }
}

// ---------- Direct (no actor, uses global BUFFERS DashMap) ----------

pub fn tokens_per_page(model_idx: usize, _id: ContextId) -> u32 {
    PAGE_SIZES.get(model_idx).map(|v| *v as u32).unwrap_or(0)
}

pub fn committed_page_count(model_idx: usize, id: ContextId) -> u32 {
    BUFFERS.get(&(model_idx, id)).map(|b| b.committed_len as u32).unwrap_or(0)
}

pub fn kv_len(model_idx: usize, id: ContextId) -> u32 {
    let page_size = PAGE_SIZES.get(model_idx).copied().unwrap_or(0);
    BUFFERS.get(&(model_idx, id))
        .map(|b| (b.committed_len * page_size + b.tokens_filled.len()) as u32)
        .unwrap_or(0)
}

pub fn get_cursor(model_idx: usize, id: ContextId) -> u32 {
    BUFFERS.get(&(model_idx, id)).map(|b| b.tokens_filled.len() as u32).unwrap_or(0)
}

pub fn set_cursor(model_idx: usize, id: ContextId, cursor: u32) -> Result<()> {
    let mut buf = BUFFERS.get_mut(&(model_idx, id))
        .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
    let max = buf.tokens_filled.len();
    if cursor as usize > max { anyhow::bail!("cursor {} out of range 0..={}", cursor, max); }
    buf.tokens_filled.truncate(cursor as usize);
    Ok(())
}

pub fn last_position(model_idx: usize, id: ContextId) -> Option<u32> {
    BUFFERS.get(&(model_idx, id)).and_then(|b| {
        let max_filled = b.tokens_filled.iter().map(|t| t.position).max();
        match (b.max_committed_position, max_filled) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        }
    })
}

pub fn get_buffered_tokens(model_idx: usize, id: ContextId) -> Vec<u32> {
    BUFFERS.get(&(model_idx, id)).map(|b| b.tokens_buffered.clone()).unwrap_or_default()
}

pub fn set_buffered_tokens(model_idx: usize, id: ContextId, tokens: Vec<u32>) -> Result<()> {
    let mut buf = BUFFERS.get_mut(&(model_idx, id))
        .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
    buf.tokens_buffered = tokens;
    Ok(())
}

pub fn append_buffered_tokens(model_idx: usize, id: ContextId, tokens: Vec<u32>) -> Result<()> {
    let mut buf = BUFFERS.get_mut(&(model_idx, id))
        .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
    buf.tokens_buffered.extend(tokens);
    Ok(())
}

pub fn fill(
    model_idx: usize, id: ContextId, n: usize,
    positions: Vec<u32>, masks: Vec<Brle>, adapter: Option<AdapterId>,
) -> Result<()> {
    let mut buf = BUFFERS.get_mut(&(model_idx, id))
        .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
    if n > buf.tokens_buffered.len() {
        anyhow::bail!("fill: n ({}) > tokens_buffered ({})", n, buf.tokens_buffered.len());
    }
    if positions.len() != n { anyhow::bail!("positions length {} != n {}", positions.len(), n); }
    if !masks.is_empty() && masks.len() != n { anyhow::bail!("masks length {} != n {}", masks.len(), n); }

    let tokens: Vec<u32> = buf.tokens_buffered.drain(..n).collect();
    for (i, token) in tokens.into_iter().enumerate() {
        buf.tokens_filled.push(TokenInfo {
            token, position: positions[i],
            mask: if masks.is_empty() { Brle::new(0) } else { masks[i].clone() },
            adapter,
        });
    }
    Ok(())
}

// =============================================================================
// ContextBuffer — lives in global DashMap (WIT direct access)
// =============================================================================

#[derive(Debug, Clone)]
pub(crate) struct ContextBuffer {
    pub tokens_filled: Vec<TokenInfo>,
    pub tokens_buffered: Vec<u32>,
    pub committed_len: usize,
    pub max_committed_position: Option<u32>,
}

impl ContextBuffer {
    fn new() -> Self {
        ContextBuffer {
            tokens_filled: Vec::new(),
            tokens_buffered: Vec::new(),
            committed_len: 0,
            max_committed_position: None,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct TokenInfo {
    pub token: u32,
    pub position: u32,
    pub mask: Brle,
    pub adapter: Option<AdapterId>,
}

// =============================================================================
// Internal Types
// =============================================================================

#[derive(Debug, Clone)]
pub(crate) enum Record {
    Fill {
        tokens: Vec<u32>,
        positions: Vec<u32>,
        mask: Vec<Brle>,
        adapter: Option<AdapterId>,
    },
}

#[derive(Debug, Clone)]
pub struct ReplayFill {
    pub context_id: ContextId,
    pub tokens: Vec<u32>,
    pub positions: Vec<u32>,
    pub masks: Vec<Brle>,
    pub adapter: Option<AdapterId>,
    pub physical_page_ids: Vec<PhysicalPageId>,
    pub device_id: DeviceId,
    pub kv_len: u32,
    pub last_page_len: u32,
    pub num_pages: u32,
}

// =============================================================================
// Process State
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProcessState {
    Running,
    Pending,
}

#[derive(Debug)]
pub(crate) struct ProcessInfo {
    pub state: ProcessState,
    pub context_ids: Vec<ContextId>,
}

// =============================================================================
// Context — lives in local HashMap on ContextManager (actor-only)
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ContextState {
    /// Active on GPU, ready for inference. Evictable.
    Active,
    /// Active on GPU, forward pass in progress — NOT immediately evictable.
    /// Eviction is deferred via `pending_suspend` flag.
    Pinned,
    /// Committed chain refcounts released, working pages on CPU.
    Suspended,
}

#[derive(Debug, Clone)]
pub(crate) struct Context {
    /// Process that owns this context (None for named snapshots).
    pub owner: Option<ProcessId>,
    /// Device this context is on (None if fully evicted).
    pub device: Option<DeviceId>,
    /// Physical page IDs for uncommitted (working) pages on GPU.
    pub working_pages: Vec<PhysicalPageId>,
    /// CPU slots for working pages when suspended.
    pub working_pages_cpu: Vec<PhysicalPageId>,
    /// Tip of the committed hash chain (None if no commits yet).
    pub committed_tip: Option<PageHash>,
    /// Full token lineage for replay after eviction.
    pub lineage: Vec<Record>,

    // Scheduling
    pub state: ContextState,
    /// Deferred suspension flag: set when context is Pinned and selected as victim.
    /// Actual suspension happens on clear_pinned.
    pub pending_suspend: bool,
    pub last_access: Instant,
}

impl Context {
    fn new(owner: Option<ProcessId>) -> Self {
        Context {
            owner,
            device: None,
            working_pages: Vec::new(),
            working_pages_cpu: Vec::new(),
            committed_tip: None,
            lineage: Vec::new(),
            state: ContextState::Active,
            pending_suspend: false,
            last_access: Instant::now(),
        }
    }
}

// =============================================================================
// ContextManager
// =============================================================================

#[derive(Debug)]
pub(crate) struct ContextManager {
    pub(crate) devices: Vec<PageStore>,
    pub(crate) page_size: usize,
    pub(crate) model_idx: usize,
    pub(crate) name_to_id: HashMap<(String, String), ContextId>,
    next_id: u64,
    pub(crate) arbiter: Arbiter,
    /// Actor-local context state (page management, device, lineage, etc.)
    pub(crate) contexts: HashMap<ContextId, Context>,
    /// FIFO queue: deferred allocation requests from Running processes.
    pub(crate) try_alloc: VecDeque<AllocWaiter>,
    /// Priority heap: Pending processes waiting for restoration.
    pub(crate) try_restore: BinaryHeap<RestoreWaiter>,
    /// Per-process state tracking.
    pub(crate) processes: HashMap<ProcessId, ProcessInfo>,
    /// Side-map: number of Pinned contexts still awaiting clear_pinned per process.
    /// Restore is blocked until this reaches 0. Avoids BinaryHeap interior mutation.
    pub(crate) pending_pinned_counts: HashMap<ProcessId, usize>,
    /// Side-map: pending alloc requests per Pending process.
    /// Replayed after restoration completes.
    pub(crate) pending_allocs_map: HashMap<ProcessId, Vec<AllocWaiter>>,
    pub(crate) msg_counter: u64,
}

impl ContextManager {
    pub(crate) fn new(model_idx: usize, page_size: usize, num_gpu_pages: &[usize], num_cpu_pages: &[usize]) -> Self {
        let devices: Vec<_> = num_gpu_pages.iter().zip(num_cpu_pages.iter())
            .map(|(&gpu, &cpu)| PageStore::new(page_size, gpu, cpu))
            .collect();
        ContextManager {
            devices, page_size, model_idx,
            name_to_id: HashMap::new(), next_id: 1,
            arbiter: Arbiter::new(),
            contexts: HashMap::new(),
            try_alloc: VecDeque::new(),
            try_restore: BinaryHeap::new(),
            processes: HashMap::new(),
            pending_pinned_counts: HashMap::new(),
            pending_allocs_map: HashMap::new(),
            msg_counter: 0,
        }
    }

    fn next_id(&mut self) -> ContextId {
        let id = self.next_id; self.next_id += 1; id
    }

    fn select_device(&self) -> usize {
        self.devices.iter().enumerate()
            .min_by(|(_, a), (_, b)| {
                let a_free = a.free_gpu_pages();
                let b_free = b.free_gpu_pages();
                b_free.cmp(&a_free) // most free pages first
            })
            .map(|(i, _)| i).unwrap_or(0)
    }

    /// Get or register a process, returning mutable reference to its info.
    pub(crate) fn ensure_process(&mut self, pid: ProcessId) -> &mut ProcessInfo {
        self.processes.entry(pid).or_insert_with(|| ProcessInfo {
            state: ProcessState::Running,
            context_ids: Vec::new(),
        })
    }

    /// Read a ContextBuffer field from the DashMap.
    fn buf(&self, id: ContextId) -> Option<dashmap::mapref::one::Ref<'_, (usize, ContextId), ContextBuffer>> {
        BUFFERS.get(&(self.model_idx, id))
    }

    fn buf_mut(&self, id: ContextId) -> Option<dashmap::mapref::one::RefMut<'_, (usize, ContextId), ContextBuffer>> {
        BUFFERS.get_mut(&(self.model_idx, id))
    }

    // ==================== Core Operations ====================

    pub(crate) fn create(&mut self, owner: Option<ProcessId>) -> Result<ContextId> {
        let id = self.next_id();
        let mut ctx = Context::new(owner);
        let dev = ctx.device.map(|d| d as usize).unwrap_or_else(|| self.select_device());
        ctx.device = Some(dev as DeviceId);

        self.contexts.insert(id, ctx);
        BUFFERS.insert((self.model_idx, id), ContextBuffer::new());

        if let Some(pid) = owner {
            let proc = self.ensure_process(pid);
            proc.context_ids.push(id);
        }

        Ok(id)
    }

    pub(crate) fn save(&mut self, id: ContextId, username: String, name: String) -> Result<()> {
        if self.name_to_id.contains_key(&(username.clone(), name.clone())) {
            anyhow::bail!("Snapshot name already exists: {}", name);
        }

        let ctx = self.contexts.get(&id).ok_or_else(|| anyhow::anyhow!("Context not found"))?;
        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let tip = ctx.committed_tip;
        let lineage = ctx.lineage.clone();

        let buf = self.buf(id).ok_or_else(|| anyhow::anyhow!("Buffer not found"))?;
        let committed_len = buf.committed_len;
        let max_pos = buf.max_committed_position;
        let mut snapshot_buffered: Vec<u32> = buf.tokens_filled.iter().map(|t| t.token).collect();
        snapshot_buffered.extend_from_slice(&buf.tokens_buffered);
        drop(buf);

        if let Some(tip_hash) = tip {
            self.devices[dev_idx].acquire_chain(tip_hash);
        }

        let snapshot_id = self.next_id();
        // Snapshot context: no working pages, no owner
        self.contexts.insert(snapshot_id, Context {
            owner: None,
            device: Some(dev_idx as DeviceId),
            working_pages: Vec::new(),
            working_pages_cpu: Vec::new(),
            committed_tip: tip,
            lineage,
            state: ContextState::Active,
            pending_suspend: false,
            last_access: Instant::now(),
        });
        BUFFERS.insert((self.model_idx, snapshot_id), ContextBuffer {
            tokens_filled: Vec::new(),
            tokens_buffered: snapshot_buffered,
            committed_len,
            max_committed_position: max_pos,
        });
        self.name_to_id.insert((username, name), snapshot_id);
        Ok(())
    }

    pub(crate) fn snapshot(&mut self, id: ContextId, username: String) -> Result<String> {
        let name = format!("__snapshot_{}", self.next_id());
        self.save(id, username, name.clone())?;
        Ok(name)
    }

    pub(crate) fn delete(&mut self, username: String, name: String) -> Result<()> {
        let snapshot_id = self.name_to_id.remove(&(username, name))
            .ok_or_else(|| anyhow::anyhow!("Snapshot not found"))?;
        self.destroy_context(snapshot_id)
    }

    pub(crate) fn destroy_context(&mut self, id: ContextId) -> Result<()> {
        let ctx = self.contexts.remove(&id)
            .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
        let buf = BUFFERS.remove(&(self.model_idx, id)).map(|(_, b)| b);
        let committed_len = buf.as_ref().map(|b| b.committed_len).unwrap_or(0);

        let dev_idx = ctx.device.unwrap_or(0) as usize;

        if let Some(pid) = ctx.owner {
            if let Some(proc) = self.processes.get_mut(&pid) {
                proc.context_ids.retain(|&c| c != id);
            }
            self.arbiter.uncommit(pid, dev_idx, committed_len);
            self.arbiter.remove_working(pid, dev_idx, ctx.working_pages.len());
        }

        // Release committed chain (skip if already released during suspension)
        if let Some(tip_hash) = ctx.committed_tip {
            let dev = &mut self.devices[dev_idx];
            if ctx.state != ContextState::Suspended {
                dev.release_chain(tip_hash);
            }
            dev.remove_index_cache(tip_hash);
        }

        // Free GPU working pages
        self.devices[dev_idx].free_working(&ctx.working_pages);
        // Free CPU working pages
        self.devices[dev_idx].free_cpu_slots(&ctx.working_pages_cpu);

        self.name_to_id.retain(|_, v| *v != id);
        Ok(())
    }

    pub(crate) fn fork(&mut self, id: ContextId) -> Result<ContextId> {
        let ctx = self.contexts.get(&id).ok_or_else(|| anyhow::anyhow!("Context not found"))?;
        let owner = ctx.owner;
        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let tip = ctx.committed_tip;
        let lineage = ctx.lineage.clone();

        let buf = self.buf(id).ok_or_else(|| anyhow::anyhow!("Buffer not found"))?;
        let committed_len = buf.committed_len;
        let max_pos = buf.max_committed_position;
        let mut new_buffered: Vec<u32> = buf.tokens_filled.iter().map(|t| t.token).collect();
        new_buffered.extend_from_slice(&buf.tokens_buffered);
        drop(buf);

        if let Some(tip_hash) = tip {
            self.devices[dev_idx].acquire_chain(tip_hash);
        }

        let new_id = self.next_id();
        self.contexts.insert(new_id, Context {
            owner,
            device: Some(dev_idx as DeviceId),
            working_pages: Vec::new(),
            working_pages_cpu: Vec::new(),
            committed_tip: tip,
            lineage,
            state: ContextState::Active,
            pending_suspend: false,
            last_access: Instant::now(),
        });
        BUFFERS.insert((self.model_idx, new_id), ContextBuffer {
            tokens_filled: Vec::new(),
            tokens_buffered: new_buffered,
            committed_len,
            max_committed_position: max_pos,
        });

        if let Some(pid) = owner {
            let proc = self.ensure_process(pid);
            proc.context_ids.push(new_id);
        }

        Ok(new_id)
    }

    // ==================== Page Management ====================

    pub(crate) fn free_pages(&mut self, id: ContextId, num_pages: u32) -> Result<()> {
        let ctx = self.contexts.get_mut(&id).ok_or_else(|| anyhow::anyhow!("Context not found"))?;

        let n = (num_pages as usize).min(ctx.working_pages.len());
        if n == 0 { return Ok(()); }

        let start = ctx.working_pages.len() - n;
        let to_free: Vec<PhysicalPageId> = ctx.working_pages.drain(start..).collect();
        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let owner = ctx.owner;

        // Truncate filled tokens in the buffer
        let tokens_to_remove = n * self.page_size;
        if tokens_to_remove > 0 {
            if let Some(mut buf) = self.buf_mut(id) {
                let len = buf.tokens_filled.len();
                buf.tokens_filled.truncate(len.saturating_sub(tokens_to_remove));
            }
        }

        self.devices[dev_idx].free_working(&to_free);

        if let Some(pid) = owner {
            self.arbiter.remove_working(pid, dev_idx, n);
        }
        Ok(())
    }

    pub(crate) fn commit_pages_impl(&mut self, id: ContextId, indices: Vec<u32>) -> Result<()> {
        let page_size = self.page_size;
        let ctx = self.contexts.get(&id).ok_or_else(|| anyhow::anyhow!("Context not found"))?;

        // Suspended context → logical commit
        if ctx.working_pages.is_empty() && !indices.is_empty() {
            return self.commit_pages_logical(id, indices);
        }

        for &idx in &indices {
            if idx as usize >= ctx.working_pages.len() {
                anyhow::bail!("Invalid page index: {}", idx);
            }
        }
        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let old_tip = ctx.committed_tip;
        let prev_hash = ctx.committed_tip.unwrap_or(0);
        let working_phys: Vec<PhysicalPageId> = indices.iter()
            .map(|&idx| ctx.working_pages[idx as usize])
            .collect();
        let owner = ctx.owner;

        // Read token data from the ContextBuffer
        let buf = self.buf(id).ok_or_else(|| anyhow::anyhow!("Buffer not found"))?;
        let mut tokens = Vec::new();
        let mut positions = Vec::new();
        let mut masks = Vec::new();
        let mut all_positions = Vec::new();

        for &idx in &indices {
            let start = idx as usize * page_size;
            let end = start + page_size;
            if end > buf.tokens_filled.len() {
                anyhow::bail!("Page {} not fully filled: need {} tokens but only have {}",
                    idx, end, buf.tokens_filled.len());
            }
            for i in start..end {
                let info = &buf.tokens_filled[i];
                tokens.push(info.token);
                positions.push(info.position);
                masks.push(info.mask.clone());
                all_positions.push(info.position);
            }
        }

        if let Some(max_committed) = buf.max_committed_position {
            for &pos in &all_positions {
                if pos <= max_committed {
                    anyhow::bail!("Position {} must be > max committed position {}", pos, max_committed);
                }
            }
        }
        let lineage_adapter = buf.tokens_filled.first().and_then(|t| t.adapter);
        drop(buf);

        // Compute hashes and commit pages
        let dev = &mut self.devices[dev_idx];
        let hashes = dev.compute_page_hashes(&tokens, &positions, &masks, prev_hash);

        let mut new_phys = Vec::new();
        let mut running_prev = prev_hash;
        for (i, &hash) in hashes.iter().enumerate() {
            let (phys, _freed) = dev.commit_working(hash, running_prev, working_phys[i]);
            new_phys.push(phys);
            running_prev = hash;
        }

        let new_tip = *hashes.last()
            .ok_or_else(|| anyhow::anyhow!("No page hashes computed during commit"))?;
        dev.update_index_cache(new_tip, old_tip, &new_phys);

        // Update Context (local)
        let ctx = self.contexts.get_mut(&id)
            .ok_or_else(|| anyhow::anyhow!("Context lost during commit"))?;
        let mut sorted_indices: Vec<usize> = indices.iter().map(|&i| i as usize).collect();
        sorted_indices.sort_unstable();
        for &idx in sorted_indices.iter().rev() {
            ctx.working_pages.remove(idx);
        }
        ctx.committed_tip = Some(new_tip);

        // Append to lineage
        if let Some(Record::Fill { tokens: t, positions: p, mask: m, adapter: a }) = ctx.lineage.last_mut() {
            if *a == lineage_adapter {
                t.extend_from_slice(&tokens);
                p.extend_from_slice(&positions);
                m.extend_from_slice(&masks);
            } else {
                ctx.lineage.push(Record::Fill { tokens, positions, mask: masks, adapter: lineage_adapter });
            }
        } else {
            ctx.lineage.push(Record::Fill { tokens, positions, mask: masks, adapter: lineage_adapter });
        }

        // Update ContextBuffer (DashMap)
        if let Some(mut buf) = self.buf_mut(id) {
            for &idx in sorted_indices.iter().rev() {
                let start = idx * page_size;
                let end = ((idx + 1) * page_size).min(buf.tokens_filled.len());
                buf.tokens_filled.drain(start..end);
            }
            buf.committed_len += indices.len();
            buf.max_committed_position = all_positions.iter().copied().max()
                .or(buf.max_committed_position);
        }

        if let Some(pid) = owner {
            self.arbiter.commit(pid, dev_idx, indices.len());
        }

        Ok(())
    }

    fn commit_pages_logical(&mut self, id: ContextId, indices: Vec<u32>) -> Result<()> {
        let page_size = self.page_size;
        let ctx = self.contexts.get(&id).ok_or_else(|| anyhow::anyhow!("Context not found"))?;
        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let prev_hash = ctx.committed_tip.unwrap_or(0);
        let owner = ctx.owner;

        let buf = self.buf(id).ok_or_else(|| anyhow::anyhow!("Buffer not found"))?;
        let mut tokens = Vec::new();
        let mut positions = Vec::new();
        let mut masks = Vec::new();

        for &idx in &indices {
            let start = idx as usize * page_size;
            let end = start + page_size;
            if end > buf.tokens_filled.len() {
                anyhow::bail!("Page {} not fully filled: need {} tokens but only have {}", idx, end, buf.tokens_filled.len());
            }
            for i in start..end {
                let info = &buf.tokens_filled[i];
                tokens.push(info.token);
                positions.push(info.position);
                masks.push(info.mask.clone());
            }
        }

        if let Some(max_committed) = buf.max_committed_position {
            for &pos in &positions {
                if pos <= max_committed {
                    anyhow::bail!("Position {} must be > max committed position {}", pos, max_committed);
                }
            }
        }
        let lineage_adapter = buf.tokens_filled.first().and_then(|t| t.adapter);
        drop(buf);

        let max_position = positions.iter().copied().max();
        let hashes = self.devices[dev_idx].compute_page_hashes(&tokens, &positions, &masks, prev_hash);
        let new_tip = *hashes.last()
            .ok_or_else(|| anyhow::anyhow!("No page hashes computed during logical commit"))?;

        {
            let dev = &mut self.devices[dev_idx];
            let mut running_prev = prev_hash;
            for &hash in &hashes {
                dev.insert_chain_link(hash, running_prev);
                running_prev = hash;
            }
        }

        // Update Context (local)
        let ctx = self.contexts.get_mut(&id)
            .ok_or_else(|| anyhow::anyhow!("Context lost during logical commit"))?;
        ctx.committed_tip = Some(new_tip);

        if let Some(Record::Fill { tokens: t, positions: p, mask: m, adapter: a }) = ctx.lineage.last_mut() {
            if *a == lineage_adapter {
                t.extend_from_slice(&tokens);
                p.extend_from_slice(&positions);
                m.extend_from_slice(&masks);
            } else {
                ctx.lineage.push(Record::Fill { tokens, positions, mask: masks, adapter: lineage_adapter });
            }
        } else {
            ctx.lineage.push(Record::Fill { tokens, positions, mask: masks, adapter: lineage_adapter });
        }

        // Update ContextBuffer (DashMap)
        if let Some(mut buf) = self.buf_mut(id) {
            let mut sorted_indices: Vec<usize> = indices.iter().map(|&i| i as usize).collect();
            sorted_indices.sort_unstable();
            for &idx in sorted_indices.iter().rev() {
                let start = idx * page_size;
                let end = ((idx + 1) * page_size).min(buf.tokens_filled.len());
                buf.tokens_filled.drain(start..end);
            }
            buf.committed_len += indices.len();
            buf.max_committed_position = max_position
                .or(buf.max_committed_position);
        }

        if let Some(pid) = owner {
            self.arbiter.commit(pid, dev_idx, indices.len());
        }

        Ok(())
    }

    pub(crate) fn get_physical_page_ids_impl(&mut self, id: ContextId) -> Result<HashMap<DeviceId, Vec<PhysicalPageId>>> {
        let ctx = self.contexts.get_mut(&id).ok_or_else(|| anyhow::anyhow!("Context not found"))?;
        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let tip = ctx.committed_tip;
        let working = ctx.working_pages.clone();

        // Transition Active → Pinned: prevents eviction while process holds page IDs.
        // Eviction of Pinned contexts is deferred via `pending_suspend` flag.
        if ctx.state == ContextState::Active {
            ctx.state = ContextState::Pinned;
        }

        let mut pages = Vec::new();

        // Committed pages
        if let Some(tip_hash) = tip {
            let dev = &mut self.devices[dev_idx];
            pages.extend(dev.resolve_physical(tip_hash));
        }

        // Working pages (appended after committed)
        pages.extend(&working);

        let mut result = HashMap::new();
        result.insert(dev_idx as DeviceId, pages);
        Ok(result)
    }

    fn build_debug_state(&self, id: ContextId) -> String {
        let ctx_info = self.contexts.get(&id).map(|ctx| {
            format!(
                "state={:?} working_pages={} working_pages_cpu={} tip={:?}",
                ctx.state, ctx.working_pages.len(), ctx.working_pages_cpu.len(),
                ctx.committed_tip,
            )
        }).unwrap_or_else(|| "CTX_MISSING".to_string());

        let buf_info = BUFFERS.get(&(self.model_idx, id)).map(|b| {
            format!(
                "committed_len={} tokens_filled={} tokens_buffered={}",
                b.committed_len, b.tokens_filled.len(), b.tokens_buffered.len(),
            )
        }).unwrap_or_else(|| "BUF_MISSING".to_string());

        format!("{} {}", ctx_info, buf_info)
    }
}

// =============================================================================
// Message & ServiceHandler
// =============================================================================

#[derive(Debug)]
pub(crate) enum Message {
    Open { username: String, name: String, response: oneshot::Sender<Result<ContextId>> },
    Create { owner: Option<ProcessId>, response: oneshot::Sender<Result<ContextId>> },
    Save { id: ContextId, username: String, name: String, response: oneshot::Sender<Result<()>> },
    Snapshot { id: ContextId, username: String, response: oneshot::Sender<Result<String>> },
    Delete { username: String, name: String, response: oneshot::Sender<Result<()>> },
    Destroy { id: ContextId, force: bool, response: oneshot::Sender<Result<()>> },
    Fork { id: ContextId, response: oneshot::Sender<Result<ContextId>> },
    CommitPages { id: ContextId, page_indices: Vec<u32>, response: oneshot::Sender<Result<()>> },
    ReservePages { id: ContextId, num_pages: u32, response: oneshot::Sender<Result<()>> },
    ReleasePages { id: ContextId, num_pages: u32 },
    GetPhysicalPageIds { id: ContextId, response: oneshot::Sender<Result<HashMap<DeviceId, Vec<PhysicalPageId>>>> },
    CommitReplayFill {
        id: ContextId, num_pages: u32,
        tokens: Vec<u32>, positions: Vec<u32>, masks: Vec<Brle>, adapter: Option<AdapterId>,
        response: oneshot::Sender<Result<()>>,
    },
    FinishRestore { id: ContextId },
    ClearPinned { id: ContextId },
    GetStats { response: oneshot::Sender<Vec<(usize, usize)>> },
    IsActive { id: ContextId, response: oneshot::Sender<bool> },
    DebugState { id: ContextId, response: oneshot::Sender<String> },
    SetDagWeights { weight: f64, pid_values: HashMap<ProcessId, f64> },
}

impl ContextManager {
    /// Dispatch replay forward passes for KV cache recomputation.
    ///
    /// Each ReplayFill contains tokens that need to be run through the model
    /// to regenerate KV data for committed pages lost during eviction.
    /// After the forward pass, the working pages are committed.
    async fn dispatch_replay_chunks(&mut self, chunks: Vec<ReplayFill>) {
        if chunks.is_empty() { return; }

        let mut ctx_ids_needing_finish: Vec<ContextId> = Vec::new();

        for chunk in chunks {
            let ctx_id = chunk.context_id;

            // Build a minimal forward pass request (no sampling — just KV fill)
            let fwd_req = ForwardPassRequest {
                context_id: 0, // unused — page IDs provided directly
                tokens: chunk.tokens.clone(),
                positions: chunk.positions.clone(),
                speculative_tokens: Vec::new(),
                speculative_positions: Vec::new(),
                output_speculative_tokens: false,
                masks: chunk.masks.clone(),
                logit_mask: None,
                sampling_indices: Vec::new(),
                samplers: Vec::new(),
                adapter_id: chunk.adapter,
                adapter_seed: None,
                arrival_time: None,
            };

            let phys_ids: Vec<u32> = chunk.physical_page_ids.iter().map(|&p| p as u32).collect();
            let mut batch = BatchedForwardPassRequest::new(chunk.device_id);
            batch.add_request(&fwd_req, &phys_ids, chunk.last_page_len);

            let result = device::call_with_timeout::<_, crate::inference::request::BatchedForwardPassResponse>(
                chunk.device_id, "fire_batch", &batch,
                std::time::Duration::from_secs(30),
            ).await;

            if let Err(e) = result {
                eprintln!("REPLAY_FWD_FAIL ctx={ctx_id} device={} err={e:#}", chunk.device_id);
                // RPC failed — still finish restore to avoid permanently Pinned state.
                // The context will have a truncated committed chain but can still serve.
                if !ctx_ids_needing_finish.contains(&ctx_id) {
                    ctx_ids_needing_finish.push(ctx_id);
                }
                continue;
            }

            let _ = self.commit_replay_chunk_impl(
                ctx_id, chunk.num_pages,
                chunk.tokens, chunk.positions, chunk.masks, chunk.adapter,
            );
            if !ctx_ids_needing_finish.contains(&ctx_id) {
                ctx_ids_needing_finish.push(ctx_id);
            }
        }

        for id in ctx_ids_needing_finish {
            self.finish_restore_impl(id);
        }
    }
}

impl ServiceHandler for ContextManager {
    type Message = Message;

    async fn handle(&mut self, msg: Message) {
        self.msg_counter += 1;

        match msg {
            Message::Open { username, name, response } => {
                let result = match self.name_to_id.get(&(username, name)) {
                    Some(&snapshot_id) => self.fork(snapshot_id),
                    None => Err(anyhow::anyhow!("Snapshot not found")),
                };
                let _ = response.send(result);
            }
            Message::Create { owner, response } => { let _ = response.send(self.create(owner)); }
            Message::Save { id, username, name, response } => {
                let _ = response.send(self.save(id, username, name));
            }
            Message::Snapshot { id, username, response } => {
                let _ = response.send(self.snapshot(id, username));
            }
            Message::Delete { username, name, response } => {
                let _ = response.send(self.delete(username, name));
            }
            Message::Destroy { id, force: _, response } => {
                let _ = response.send(self.destroy_context(id));
                let chunks = self.drain_queues();
                self.dispatch_replay_chunks(chunks).await;
            }
            Message::Fork { id, response } => { let _ = response.send(self.fork(id)); }
            Message::CommitPages { id, page_indices, response } => {
                let _ = response.send(self.commit_pages_impl(id, page_indices));
                let chunks = self.drain_queues();
                self.dispatch_replay_chunks(chunks).await;
            }
            Message::ReservePages { id, num_pages, response } => {
                self.handle_reserve_pages(id, num_pages, response);
            }
            Message::ReleasePages { id, num_pages } => {
                let _ = self.free_pages(id, num_pages);
                let chunks = self.drain_queues();
                self.dispatch_replay_chunks(chunks).await;
            }
            Message::GetPhysicalPageIds { id, response } => {
                let _ = response.send(self.get_physical_page_ids_impl(id));
            }
            Message::CommitReplayFill { id, num_pages, tokens, positions, masks, adapter, response } => {
                let _ = response.send(self.commit_replay_chunk_impl(id, num_pages, tokens, positions, masks, adapter));
            }
            Message::FinishRestore { id } => {
                self.finish_restore_impl(id);
                let chunks = self.drain_queues();
                self.dispatch_replay_chunks(chunks).await;
            }
            Message::ClearPinned { id } => {
                let chunks = self.handle_clear_pinned(id);
                self.dispatch_replay_chunks(chunks).await;
            }
            Message::GetStats { response } => {
                let stats: Vec<_> = self.devices.iter().map(|d| d.stats()).collect();
                let _ = response.send(stats);
            }
            Message::IsActive { id, response } => {
                let active = self.contexts.get(&id)
                    .map(|ctx| matches!(ctx.state, ContextState::Active | ContextState::Pinned))
                    .unwrap_or(false);
                let _ = response.send(active);
            }
            Message::DebugState { id, response } => {
                let _ = response.send(self.build_debug_state(id));
            }
            Message::SetDagWeights { weight, pid_values } => {
                for (pid, value) in pid_values {
                    self.arbiter.set_weight(pid, weight * value);
                }
                let chunks = self.drain_queues();
                self.dispatch_replay_chunks(chunks).await;
            }
        }
    }
}
