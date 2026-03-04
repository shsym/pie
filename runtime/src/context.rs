//! # Context Module
//!
//! Manages named execution contexts with KV cache state for model inference.
//!
//! Each model gets a dedicated ContextManager actor that handles:
//! - Context creation, destruction, and forking
//! - Page management (commit, reserve, release)
//! - Token buffering and cursor tracking
//! - Contention resolution via dual-queue protocol
//!
//! Replaces context_legacy with:
//! - 3-state contexts (Active/Pinned/Suspended) instead of 4
//! - 2-state processes (Running/Pending) — explicit
//! - Dual-queue contention: try_alloc (FIFO) + try_restore (priority heap)
//! - Process-level eviction (all-or-nothing)
pub mod pagestore;
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
use crate::process::ProcessId;

use pagestore::{PhysicalPageId, PageHash, PageStore};
use crate::device::DeviceId;
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
static CONTEXTS: LazyLock<DashMap<(usize, ContextId), Context>> = LazyLock::new(DashMap::new);
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

/// Result of ensure_resident: replay chunks (if any) + physical page IDs.
#[derive(Debug)]
pub struct ResidentResult {
    pub replay_chunks: Option<Vec<ReplayFill>>,
    pub pages: HashMap<DeviceId, Vec<PhysicalPageId>>,
    pub kv_len: u32,
    pub debug_state: String,
}

pub async fn ensure_resident(model_idx: usize, id: ContextId) -> Result<ResidentResult> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::EnsureResident { id, response: tx })?;
    rx.await.context("context::ensure_resident: actor dropped response")?
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

pub fn clear_pinned(model_idx: usize, id: ContextId) {
    let _ = SERVICES.send(model_idx, Message::ClearPinned { id });
}

pub async fn get_stats(model_idx: usize) -> Vec<(usize, usize)> {
    let (tx, rx) = oneshot::channel();
    let _ = SERVICES.send(model_idx, Message::GetStats { response: tx });
    rx.await.unwrap_or_default()
}

// ---------- Arbiter policy (broadcast to all models) ----------

pub fn set_dag_weights(weight: f64, pid_values: HashMap<ProcessId, f64>) {
    for model_idx in 0..SERVICES.len() {
        let _ = SERVICES.send(model_idx, Message::SetDagWeights {
            weight, pid_values: pid_values.clone(),
        });
    }
}

// ---------- Direct (no actor, uses global CONTEXTS DashMap) ----------

pub fn tokens_per_page(model_idx: usize, _id: ContextId) -> u32 {
    PAGE_SIZES.get(model_idx).map(|v| *v as u32).unwrap_or(0)
}

pub fn committed_page_count(model_idx: usize, id: ContextId) -> u32 {
    CONTEXTS.get(&(model_idx, id)).map(|ctx| ctx.committed_len as u32).unwrap_or(0)
}

pub fn kv_len(model_idx: usize, id: ContextId) -> u32 {
    let page_size = PAGE_SIZES.get(model_idx).copied().unwrap_or(0);
    CONTEXTS.get(&(model_idx, id))
        .map(|ctx| (ctx.committed_len * page_size + ctx.tokens_filled.len()) as u32)
        .unwrap_or(0)
}

pub fn debug_context_state(model_idx: usize, id: ContextId) -> String {
    CONTEXTS.get(&(model_idx, id))
        .map(|ctx| format!(
            "committed_len={} tokens_filled={} working_pages={} working_cpu={} state={:?}",
            ctx.committed_len, ctx.tokens_filled.len(),
            ctx.working_pages.len(), ctx.working_cpu_slots.len(),
            ctx.state,
        ))
        .unwrap_or_else(|| "MISSING".to_string())
}

pub fn get_cursor(model_idx: usize, id: ContextId) -> u32 {
    CONTEXTS.get(&(model_idx, id)).map(|ctx| ctx.cursor()).unwrap_or(0)
}

pub fn is_active(model_idx: usize, id: ContextId) -> bool {
    CONTEXTS.get(&(model_idx, id)).map(|ctx| matches!(ctx.state, ContextState::Active | ContextState::Pinned)).unwrap_or(false)
}

/// Try to pin context as Pinned (non-evictable). Returns true on success.
/// Returns false if the context was evicted (Suspended) between page resolution
/// and this call — meaning the resolved pages are stale.
pub fn set_pinned(model_idx: usize, id: ContextId) -> bool {
    if let Some(mut ctx) = CONTEXTS.get_mut(&(model_idx, id)) {
        if ctx.state == ContextState::Active {
            ctx.state = ContextState::Pinned;
            return true;
        }
    }
    false
}

pub fn set_cursor(model_idx: usize, id: ContextId, cursor: u32) -> Result<()> {
    let mut ctx = CONTEXTS.get_mut(&(model_idx, id))
        .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
    ctx.set_cursor(cursor)
}

pub fn last_position(model_idx: usize, id: ContextId) -> Option<u32> {
    CONTEXTS.get(&(model_idx, id)).and_then(|ctx| ctx.last_position())
}

pub fn get_buffered_tokens(model_idx: usize, id: ContextId) -> Vec<u32> {
    CONTEXTS.get(&(model_idx, id)).map(|ctx| ctx.tokens_buffered.clone()).unwrap_or_default()
}

pub fn set_buffered_tokens(model_idx: usize, id: ContextId, tokens: Vec<u32>) -> Result<()> {
    let mut ctx = CONTEXTS.get_mut(&(model_idx, id))
        .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
    ctx.tokens_buffered = tokens;
    Ok(())
}

pub fn append_buffered_tokens(model_idx: usize, id: ContextId, tokens: Vec<u32>) -> Result<()> {
    let mut ctx = CONTEXTS.get_mut(&(model_idx, id))
        .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
    ctx.tokens_buffered.extend(tokens);
    Ok(())
}

pub fn fill(
    model_idx: usize, id: ContextId, n: usize,
    positions: Vec<u32>, masks: Vec<Brle>, adapter: Option<AdapterId>,
) -> Result<()> {
    let mut ctx = CONTEXTS.get_mut(&(model_idx, id))
        .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
    ctx.fill(n, positions, masks, adapter)
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

#[derive(Debug, Clone)]
pub(crate) struct TokenInfo {
    pub token: u32,
    pub position: u32,
    pub mask: Brle,
    pub adapter: Option<AdapterId>,
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
// Context State
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

// =============================================================================
// Context
// =============================================================================

#[derive(Debug, Clone)]
pub(crate) struct Context {
    /// Process that owns this context (None for named snapshots).
    pub owner: Option<ProcessId>,
    /// Device this context is on (None if fully evicted).
    pub device: Option<DeviceId>,
    /// Physical page IDs for uncommitted (working) pages on GPU.
    pub working_pages: Vec<PhysicalPageId>,
    /// CPU slots for working pages when suspended.
    pub working_cpu_slots: Vec<PhysicalPageId>,
    /// Tip of the committed hash chain (None if no commits yet).
    pub committed_tip: Option<PageHash>,
    /// Number of committed pages.
    pub committed_len: usize,

    // Token state
    pub tokens_filled: Vec<TokenInfo>,
    pub tokens_buffered: Vec<u32>,
    pub lineage: Vec<Record>,

    // Scheduling
    pub max_committed_position: Option<u32>,
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
            working_cpu_slots: Vec::new(),
            committed_tip: None,
            committed_len: 0,
            tokens_filled: Vec::new(),
            tokens_buffered: Vec::new(),
            lineage: Vec::new(),
            max_committed_position: None,
            state: ContextState::Active,
            pending_suspend: false,
            last_access: Instant::now(),
        }
    }

    fn cursor(&self) -> u32 { self.tokens_filled.len() as u32 }

    fn set_cursor(&mut self, cursor: u32) -> Result<()> {
        let max = self.tokens_filled.len();
        if cursor as usize > max { anyhow::bail!("cursor {} out of range 0..={}", cursor, max); }
        self.tokens_filled.truncate(cursor as usize);
        Ok(())
    }

    fn last_position(&self) -> Option<u32> {
        let max_filled = self.tokens_filled.iter().map(|t| t.position).max();
        match (self.max_committed_position, max_filled) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        }
    }
}

impl Context {
    fn fill(&mut self, n: usize, positions: Vec<u32>, masks: Vec<Brle>, adapter: Option<AdapterId>) -> Result<()> {
        if n > self.tokens_buffered.len() {
            anyhow::bail!("fill: n ({}) > tokens_buffered ({})", n, self.tokens_buffered.len());
        }
        if positions.len() != n { anyhow::bail!("positions length {} != n {}", positions.len(), n); }
        if !masks.is_empty() && masks.len() != n { anyhow::bail!("masks length {} != n {}", masks.len(), n); }

        let tokens: Vec<u32> = self.tokens_buffered.drain(..n).collect();
        for (i, token) in tokens.into_iter().enumerate() {
            self.tokens_filled.push(TokenInfo {
                token, position: positions[i],
                mask: if masks.is_empty() { Brle::new(0) } else { masks[i].clone() },
                adapter,
            });
        }
        Ok(())
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
    /// FIFO queue: deferred allocation requests from Running processes.
    pub(crate) try_alloc: VecDeque<AllocWaiter>,
    /// Priority heap: Pending processes waiting for restoration.
    pub(crate) try_restore: BinaryHeap<RestoreWaiter>,
    /// Per-process state tracking.
    pub(crate) processes: HashMap<ProcessId, ProcessInfo>,
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
            try_alloc: VecDeque::new(),
            try_restore: BinaryHeap::new(),
            processes: HashMap::new(),
            msg_counter: 0,
        }
    }

    // ==================== DashMap Helpers ====================

    pub(crate) fn ctx(&self, id: ContextId) -> Result<dashmap::mapref::one::Ref<'_, (usize, ContextId), Context>> {
        CONTEXTS.get(&(self.model_idx, id))
            .ok_or_else(|| anyhow::anyhow!("Context {} not found", id))
    }

    pub(crate) fn ctx_mut(&self, id: ContextId) -> Result<dashmap::mapref::one::RefMut<'_, (usize, ContextId), Context>> {
        CONTEXTS.get_mut(&(self.model_idx, id))
            .ok_or_else(|| anyhow::anyhow!("Context {} not found", id))
    }

    fn next_id(&mut self) -> ContextId {
        let id = self.next_id; self.next_id += 1; id
    }

    fn select_device_for_context(&self, ctx: &Context) -> usize {
        if let Some(dev) = ctx.device { return dev as usize; }
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

    // ==================== Core Operations ====================

    pub(crate) fn create(&mut self, owner: Option<ProcessId>) -> Result<ContextId> {
        let id = self.next_id();
        let mut ctx = Context::new(owner);
        let dev = self.select_device_for_context(&ctx);
        ctx.device = Some(dev as DeviceId);
        CONTEXTS.insert((self.model_idx, id), ctx);

        if let Some(pid) = owner {
            let proc = self.ensure_process(pid);
            proc.context_ids.push(id);
        }

        Ok(id)
    }

    pub(crate) fn save(&mut self, id: ContextId, username: String, name: String) -> Result<()> {
        let source = self.ctx(id)?;
        if self.name_to_id.contains_key(&(username.clone(), name.clone())) {
            anyhow::bail!("Snapshot name already exists: {}", name);
        }

        let owner = source.owner;
        let dev_idx = source.device.unwrap_or(0) as usize;
        let tip = source.committed_tip;
        let committed_len = source.committed_len;
        let lineage = source.lineage.clone();
        let max_pos = source.max_committed_position;
        let mut snapshot_buffered: Vec<u32> = source.tokens_filled.iter().map(|t| t.token).collect();
        snapshot_buffered.extend_from_slice(&source.tokens_buffered);
        drop(source);

        if let Some(tip_hash) = tip {
            if let Some(dev) = self.devices.get_mut(dev_idx) {
                dev.acquire_chain(tip_hash);
            }
        }

        let snapshot_id = self.next_id();
        let snapshot_ctx = Context {
            owner: None,
            device: Some(dev_idx as DeviceId),
            working_pages: Vec::new(),
            working_cpu_slots: Vec::new(),
            committed_tip: tip,
            committed_len,
            tokens_filled: Vec::new(),
            tokens_buffered: snapshot_buffered,
            lineage,
            max_committed_position: max_pos,
            state: ContextState::Active,
            pending_suspend: false,
            last_access: Instant::now(),
        };
        CONTEXTS.insert((self.model_idx, snapshot_id), snapshot_ctx);
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
        let ctx = self.ctx(id)?;
        let owner = ctx.owner;
        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let tip = ctx.committed_tip;
        let committed_len = ctx.committed_len;
        let working = ctx.working_pages.clone();
        let working_cpu = ctx.working_cpu_slots.clone();
        let was_suspended = ctx.state == ContextState::Suspended;
        let state = ctx.state;
        drop(ctx);

        eprintln!(
            "DESTROY ctx={id} state={state:?} tip={tip:?} committed_len={committed_len} \
             working={} was_suspended={was_suspended}",
            working.len(),
        );

        CONTEXTS.remove(&(self.model_idx, id));

        if let Some(pid) = owner {
            // Remove from process tracking
            if let Some(proc) = self.processes.get_mut(&pid) {
                proc.context_ids.retain(|&c| c != id);
            }

            // Clean up arbiter
            self.arbiter.uncommit(pid, dev_idx, committed_len);
            self.arbiter.remove_working(pid, dev_idx, working.len());
        }

        // Only release chain if not already released (suspended contexts already did)
        if let Some(tip_hash) = tip {
            if let Some(dev) = self.devices.get_mut(dev_idx) {
                if !was_suspended {
                    dev.release_chain(tip_hash);
                }
                dev.remove_index_cache(tip_hash);
            }
        }

        // Free GPU working pages
        if let Some(dev) = self.devices.get_mut(dev_idx) {
            dev.free_working(&working);
        }
        // Free CPU slots
        if !working_cpu.is_empty() {
            if let Some(dev) = self.devices.get_mut(dev_idx) {
                // Note: swap_in frees CPU slots, but if context is destroyed
                // while suspended, we must free them explicitly
                for &s in &working_cpu {
                    // Direct free since PageStore::swap_in isn't applicable here
                }
                // Actually free CPU slots by allocating GPU and immediately freeing
                // ... no, we need a direct free_cpu_slots method. Let's add one.
            }
        }

        self.name_to_id.retain(|_, v| *v != id);
        Ok(())
    }

    pub(crate) fn fork(&mut self, id: ContextId) -> Result<ContextId> {
        let source = self.ctx(id)?;

        if !source.tokens_filled.is_empty() {
            let base = source.max_committed_position.map(|p| p + 1).unwrap_or(0);
            for (i, info) in source.tokens_filled.iter().enumerate() {
                if info.position != base + i as u32 {
                    anyhow::bail!("Cannot fork: non-sequential filled positions");
                }
            }
        }

        let owner = source.owner;
        let dev_idx = source.device.unwrap_or(0) as usize;
        let tip = source.committed_tip;
        let committed_len = source.committed_len;
        let lineage = source.lineage.clone();
        let max_pos = source.max_committed_position;
        let mut new_buffered: Vec<u32> = source.tokens_filled.iter().map(|t| t.token).collect();
        new_buffered.extend_from_slice(&source.tokens_buffered);
        drop(source);

        if let Some(tip_hash) = tip {
            if let Some(dev) = self.devices.get_mut(dev_idx) {
                dev.acquire_chain(tip_hash);
            }
        }

        let new_id = self.next_id();
        let new_ctx = Context {
            owner,
            device: Some(dev_idx as DeviceId),
            working_pages: Vec::new(),
            working_cpu_slots: Vec::new(),
            committed_tip: tip,
            committed_len,
            tokens_filled: Vec::new(),
            tokens_buffered: new_buffered,
            lineage,
            max_committed_position: max_pos,
            state: ContextState::Active,
            pending_suspend: false,
            last_access: Instant::now(),
        };
        CONTEXTS.insert((self.model_idx, new_id), new_ctx);

        if let Some(pid) = owner {
            let proc = self.ensure_process(pid);
            proc.context_ids.push(new_id);
        }

        Ok(new_id)
    }

    // ==================== Page Management ====================

    pub(crate) fn free_pages(&mut self, id: ContextId, num_pages: u32) -> Result<()> {
        let mut ctx = self.ctx_mut(id)?;

        let n = (num_pages as usize).min(ctx.working_pages.len());
        if n == 0 { return Ok(()); }

        let start = ctx.working_pages.len() - n;
        let to_free: Vec<PhysicalPageId> = ctx.working_pages.drain(start..).collect();
        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let owner = ctx.owner;

        let tokens_to_remove = n * self.page_size;
        let tokens_len = ctx.tokens_filled.len();
        if tokens_to_remove > 0 && tokens_len > 0 {
            ctx.tokens_filled.truncate(tokens_len.saturating_sub(tokens_to_remove));
        }
        drop(ctx);

        self.devices[dev_idx].free_working(&to_free);

        if let Some(pid) = owner {
            self.arbiter.remove_working(pid, dev_idx, n);
        }
        Ok(())
    }

    pub(crate) fn commit_pages_impl(&mut self, id: ContextId, indices: Vec<u32>) -> Result<()> {
        let page_size = self.page_size;
        let ctx = self.ctx(id)?;

        // Suspended context → logical commit
        if ctx.working_pages.is_empty() && !indices.is_empty() {
            drop(ctx);
            return self.commit_pages_logical(id, indices);
        }

        for &idx in &indices {
            if idx as usize >= ctx.working_pages.len() {
                anyhow::bail!("Invalid page index: {}", idx);
            }
        }

        let mut tokens = Vec::new();
        let mut positions = Vec::new();
        let mut masks = Vec::new();
        let mut all_positions = Vec::new();

        for &idx in &indices {
            let start = idx as usize * page_size;
            let end = start + page_size;
            if end > ctx.tokens_filled.len() {
                anyhow::bail!("Page {} not fully filled: need {} tokens but only have {}",
                    idx, end, ctx.tokens_filled.len());
            }
            for i in start..end {
                let info = &ctx.tokens_filled[i];
                tokens.push(info.token);
                positions.push(info.position);
                masks.push(info.mask.clone());
                all_positions.push(info.position);
            }
        }

        if let Some(max_committed) = ctx.max_committed_position {
            for &pos in &all_positions {
                if pos <= max_committed {
                    anyhow::bail!("Position {} must be > max committed position {}", pos, max_committed);
                }
            }
        }

        let prev_hash = ctx.committed_tip.unwrap_or(0);
        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let old_tip = ctx.committed_tip;
        let lineage_adapter = ctx.tokens_filled.first().and_then(|t| t.adapter);
        let working_phys: Vec<PhysicalPageId> = indices.iter()
            .map(|&idx| ctx.working_pages[idx as usize])
            .collect();
        drop(ctx);

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

        let owner = {
            let mut ctx = self.ctx_mut(id)
                .map_err(|_| anyhow::anyhow!("Context lost during commit"))?;

            let mut sorted_indices: Vec<usize> = indices.iter().map(|&i| i as usize).collect();
            sorted_indices.sort_unstable();
            for &idx in sorted_indices.iter().rev() {
                ctx.working_pages.remove(idx);
            }
            for &idx in sorted_indices.iter().rev() {
                let start = idx * page_size;
                let end = ((idx + 1) * page_size).min(ctx.tokens_filled.len());
                ctx.tokens_filled.drain(start..end);
            }

            ctx.committed_tip = Some(new_tip);
            ctx.committed_len += indices.len();
            ctx.max_committed_position = all_positions.iter().copied().max()
                .or(ctx.max_committed_position);

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

            ctx.owner
        };

        if let Some(pid) = owner {
            self.arbiter.commit(pid, dev_idx, indices.len());
        }

        Ok(())
    }

    fn commit_pages_logical(&mut self, id: ContextId, indices: Vec<u32>) -> Result<()> {
        let page_size = self.page_size;
        let ctx = self.ctx(id)?;

        let mut tokens = Vec::new();
        let mut positions = Vec::new();
        let mut masks = Vec::new();

        for &idx in &indices {
            let start = idx as usize * page_size;
            let end = start + page_size;
            if end > ctx.tokens_filled.len() {
                anyhow::bail!("Page {} not fully filled: need {} tokens but only have {}", idx, end, ctx.tokens_filled.len());
            }
            for i in start..end {
                let info = &ctx.tokens_filled[i];
                tokens.push(info.token);
                positions.push(info.position);
                masks.push(info.mask.clone());
            }
        }

        if let Some(max_committed) = ctx.max_committed_position {
            for &pos in &positions {
                if pos <= max_committed {
                    anyhow::bail!("Position {} must be > max committed position {}", pos, max_committed);
                }
            }
        }

        let prev_hash = ctx.committed_tip.unwrap_or(0);
        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let lineage_adapter = ctx.tokens_filled.first().and_then(|t| t.adapter);
        drop(ctx);

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

        let owner = {
            let mut ctx = self.ctx_mut(id)
                .map_err(|_| anyhow::anyhow!("Context lost during logical commit"))?;

            let mut sorted_indices: Vec<usize> = indices.iter().map(|&i| i as usize).collect();
            sorted_indices.sort_unstable();
            for &idx in sorted_indices.iter().rev() {
                let start = idx * page_size;
                let end = ((idx + 1) * page_size).min(ctx.tokens_filled.len());
                ctx.tokens_filled.drain(start..end);
            }

            ctx.committed_tip = Some(new_tip);
            ctx.committed_len += indices.len();
            ctx.max_committed_position = positions.iter().copied().max()
                .or(ctx.max_committed_position);

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

            ctx.owner
        };

        if let Some(pid) = owner {
            self.arbiter.commit(pid, dev_idx, indices.len());
        }

        Ok(())
    }

    pub(crate) fn get_physical_page_ids_impl(&mut self, id: ContextId) -> Result<HashMap<DeviceId, Vec<PhysicalPageId>>> {
        let ctx = self.ctx(id)?;
        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let tip = ctx.committed_tip;
        let working = ctx.working_pages.clone();
        drop(ctx);

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
    EnsureResident { id: ContextId, response: oneshot::Sender<Result<ResidentResult>> },
    CommitReplayFill {
        id: ContextId, num_pages: u32,
        tokens: Vec<u32>, positions: Vec<u32>, masks: Vec<Brle>, adapter: Option<AdapterId>,
        response: oneshot::Sender<Result<()>>,
    },
    FinishRestore { id: ContextId },
    ClearPinned { id: ContextId },
    GetStats { response: oneshot::Sender<Vec<(usize, usize)>> },
    SetDagWeights { weight: f64, pid_values: HashMap<ProcessId, f64> },
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
                self.drain_queues();
            }
            Message::Fork { id, response } => { let _ = response.send(self.fork(id)); }
            Message::CommitPages { id, page_indices, response } => {
                let _ = response.send(self.commit_pages_impl(id, page_indices));
                self.drain_queues();
            }
            Message::ReservePages { id, num_pages, response } => {
                self.handle_reserve_pages(id, num_pages, response);
            }
            Message::ReleasePages { id, num_pages } => {
                let _ = self.free_pages(id, num_pages);
                self.drain_queues();
            }
            Message::GetPhysicalPageIds { id, response } => {
                let _ = response.send(self.get_physical_page_ids_impl(id));
            }
            Message::EnsureResident { id, response } => {
                let result = self.handle_ensure_resident(id);
                match result {
                    Ok(resident) => { let _ = response.send(Ok(resident)); }
                    Err(e) => { let _ = response.send(Err(e)); }
                }
            }
            Message::CommitReplayFill { id, num_pages, tokens, positions, masks, adapter, response } => {
                let _ = response.send(self.commit_replay_chunk_impl(id, num_pages, tokens, positions, masks, adapter));
            }
            Message::FinishRestore { id } => {
                self.finish_restore_impl(id);
                self.drain_queues();
            }
            Message::ClearPinned { id } => {
                self.handle_clear_pinned(id);
            }
            Message::GetStats { response } => {
                let stats: Vec<_> = self.devices.iter().map(|d| d.stats()).collect();
                let _ = response.send(stats);
            }
            Message::SetDagWeights { weight, pid_values } => {
                for (pid, value) in pid_values {
                    self.arbiter.set_weight(pid, weight * value);
                }
                self.drain_queues();
            }
        }
    }
}
