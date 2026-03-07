//! # Context Module
//!
//! Manages named execution contexts with KV cache state for model inference.
//! All context state (token data, page management, scheduling) lives in the
//! actor-local `Context` struct, accessed exclusively through the `ContextManager`
//! actor via typed `Message` variants.
//!
//! Each model gets a dedicated ContextManager actor that handles:
//! - Context creation, destruction, and forking
//! - Page management (commit, reserve, release)
//! - Contention resolution via dual-queue protocol
pub mod pagestore;
pub(crate) mod sched;
mod suspend;
mod snapshot;
mod replay;

use std::collections::{HashMap, VecDeque, BinaryHeap};
use std::sync::LazyLock;
use std::time::Instant;
use tokio::sync::oneshot;
use anyhow::{Result, Context as _};


use crate::service::{ServiceArray, ServiceHandler};
use crate::adapter::AdapterId;
use crate::inference::brle::Brle;
use crate::process::ProcessId;
use crate::device::{self, DeviceId};

use pagestore::{PhysicalPageId, PageHash, PageStore};
use sched::ProcessEntry;

use suspend::{AllocWaiter, DeferredOp, RestoreWaiter};

// =============================================================================
// Public Types
// =============================================================================

pub type ContextId = u64;

// =============================================================================
// Globals
// =============================================================================

pub(crate) static SERVICES: LazyLock<ServiceArray<Message>> = LazyLock::new(ServiceArray::new);
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

pub async fn take(model_idx: usize, username: String, name: String) -> Result<ContextId> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Take { username, name, response: tx })?;
    rx.await.context("context::take: actor dropped response")?
}

pub async fn create(model_idx: usize, owner: Option<ProcessId>) -> Result<ContextId> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Create { owner, response: tx })?;
    rx.await.context("context::create: actor dropped response")?
}

/// Save a context under a name. If `name` is None, auto-generates a snapshot name.
/// Returns the name used (only meaningful when auto-generated).
pub async fn save(model_idx: usize, id: ContextId, username: String, name: Option<String>) -> Result<Option<String>> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Save { id, username, name, response: tx })?;
    rx.await.context("context::save: actor dropped response")?
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

pub async fn commit_working_pages(model_idx: usize, id: ContextId, num_pages: usize) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::CommitWorkingPages { id, num_pages, response: tx })?;
    rx.await.context("context::commit_working_pages: actor dropped response")?
}

pub async fn reserve_working_pages(model_idx: usize, id: ContextId, num_pages: usize) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::ReserveWorkingPages { id, num_pages, response: tx })?;
    rx.await.context("context::reserve_working_pages: actor dropped response")?
}

pub fn release_working_pages(model_idx: usize, id: ContextId, num_pages: usize) -> Result<()> {
    SERVICES.send(model_idx, Message::ReleaseWorkingPages { id, num_pages })
}

/// Pin context for a forward pass: Active → Pinned.
/// Returns a PinnedContext with physical page IDs, kv_len, and last_page_len.
/// The context is non-evictable until `unpin`.
pub async fn pin(model_idx: usize, id: ContextId, num_input_tokens: u32) -> Result<PinnedContext> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Pin { id, num_input_tokens, response: tx })?;
    rx.await.context("context::pin: actor dropped response")?
}

/// Unpin context: Pinned → Active. Fire-and-forget actor message.
/// Also executes deferred suspension if `pending_suspend` was set.
pub fn unpin(model_idx: usize, id: ContextId) {
    let _ = SERVICES.send(model_idx, Message::Unpin { id });
}

pub async fn get_stats(model_idx: usize) -> Vec<(usize, usize)> {
    let (tx, rx) = oneshot::channel();
    let _ = SERVICES.send(model_idx, Message::GetStats { response: tx });
    rx.await.unwrap_or_default()
}

pub async fn debug_context_state(model_idx: usize, id: ContextId) -> String {
    let (tx, rx) = oneshot::channel();
    let _ = SERVICES.send(model_idx, Message::DebugState { id, response: tx });
    rx.await.unwrap_or_else(|_| "MISSING".to_string())
}

// ---------- Scheduling policy (broadcast to all models) ----------

pub fn set_priority(weight: f64, pid_values: HashMap<ProcessId, f64>) {
    for model_idx in 0..SERVICES.len() {
        let _ = SERVICES.send(model_idx, Message::SetPriority {
            weight, pid_values: pid_values.clone(),
        });
    }
}

// ---------- Direct (no actor) ----------

pub fn tokens_per_page(model_idx: usize) -> u32 {
    PAGE_SIZES.get(model_idx).map(|v| *v as u32).unwrap_or(0)
}

// ---------- Actor-routed read/write APIs ----------

pub async fn committed_page_count(model_idx: usize, id: ContextId) -> Result<u32> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::CommittedPageCount { id, response: tx })?;
    rx.await.context("context::committed_page_count: actor dropped response")
}

pub async fn working_page_count(model_idx: usize, id: ContextId) -> Result<u32> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::WorkingPageCount { id, response: tx })?;
    rx.await.context("context::working_page_count: actor dropped response")
}

pub async fn working_page_token_count(model_idx: usize, id: ContextId) -> Result<u32> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::WorkingPageTokenCount { id, response: tx })?;
    rx.await.context("context::working_page_token_count: actor dropped response")
}

pub async fn truncate_working_page_tokens(model_idx: usize, id: ContextId, count: u32) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::TruncateWorkingPageTokens { id, count, response: tx })?;
    rx.await.context("context::truncate_working_page_tokens: actor dropped response")?
}

pub async fn append_working_page_tokens(
    model_idx: usize, id: ContextId, tokens: Vec<u32>,
    positions: Vec<u32>, masks: Vec<Brle>, adapter: Option<AdapterId>,
) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::AppendWorkingPageTokens {
        id, tokens, positions, masks, adapter, response: tx,
    })?;
    rx.await.context("context::append_working_page_tokens: actor dropped response")?
}

// =============================================================================
// PinnedContext — returned by pin()
// =============================================================================

#[derive(Debug)]
pub struct PinnedContext {
    pub device: DeviceId,
    pub pages: Vec<PhysicalPageId>,
    pub kv_len: u32,
    pub last_page_len: u32,
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
    /// Physical page IDs for uncommitted (working) pages.
    /// On GPU when Active/Pinned, on CPU when Suspended.
    pub working_pages: Vec<PhysicalPageId>,
    /// Tip of the committed hash chain (None if no commits yet).
    pub committed_tip: Option<PageHash>,
    /// Number of committed pages.
    pub committed_len: usize,
    /// Full token lineage for replay after eviction.
    pub lineage: Vec<Record>,

    // Token-level data (previously in ContextTokens / BUFFERS DashMap)
    /// Tokens that have been forwarded but not yet committed to a page.
    pub working_page_tokens: Vec<TokenInfo>,

    /// Maximum position value across all committed tokens. Need to check the validity of committed tokens
    pub max_committed_position: Option<u32>,

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
            committed_tip: None,
            lineage: Vec::new(),
            working_page_tokens: Vec::new(),
            committed_len: 0,
            max_committed_position: None,
            state: ContextState::Active,
            pending_suspend: false,
            last_access: Instant::now(),
        }
    }

    pub fn is_active(&self) -> bool { self.state == ContextState::Active }
    pub fn is_suspended(&self) -> bool { self.state == ContextState::Suspended }
    pub fn is_pinned(&self) -> bool { self.state == ContextState::Pinned }
}

// =============================================================================
// ContextManager
// =============================================================================

#[derive(Debug)]
pub(crate) struct ContextManager {
    pub(crate) devices: Vec<PageStore>,
    pub(crate) page_size: usize,
    pub(crate) model_idx: usize,
    pub(crate) snapshots: HashMap<(String, String), ContextId>,
    next_id: u64,
    /// Per-process scheduling, ownership, and transient suspension state.
    pub(crate) processes: HashMap<ProcessId, ProcessEntry>,
    /// Actor-local context state (page management, device, lineage, etc.)
    pub(crate) contexts: HashMap<ContextId, Context>,
    /// FIFO queue: deferred allocation requests from Running processes.
    pub(crate) try_alloc: VecDeque<AllocWaiter>,
    /// Priority heap: Pending processes waiting for restoration.
    pub(crate) try_restore: BinaryHeap<RestoreWaiter>,
}

impl ContextManager {
    pub(crate) fn new(model_idx: usize, page_size: usize, num_gpu_pages: &[usize], num_cpu_pages: &[usize]) -> Self {
        let devices: Vec<_> = num_gpu_pages.iter().zip(num_cpu_pages.iter())
            .map(|(&gpu, &cpu)| PageStore::new(page_size, gpu, cpu))
            .collect();
        ContextManager {
            devices, page_size, model_idx,
            snapshots: HashMap::new(), next_id: 1,
            processes: HashMap::new(),
            contexts: HashMap::new(),
            try_alloc: VecDeque::new(),
            try_restore: BinaryHeap::new(),
        }
    }

    fn next_id(&mut self) -> ContextId {
        let id = self.next_id; self.next_id += 1; id
    }

    fn least_loaded_device(&self) -> usize {
        self.devices.iter().enumerate()
            .min_by(|(_, a), (_, b)| {
                let a_free = a.available_gpu_pages();
                let b_free = b.available_gpu_pages();
                b_free.cmp(&a_free) // most free pages first
            })
            .map(|(i, _)| i).unwrap_or(0)
    }

    // ==================== Core Operations ====================

    pub(crate) fn create(&mut self, owner: Option<ProcessId>) -> Result<ContextId> {
        let id = self.next_id();
        let mut ctx = Context::new(owner);
        ctx.device = Some(self.least_loaded_device());

        self.contexts.insert(id, ctx);

        if let Some(pid) = owner {
            let proc = self.process_entry(pid);
            proc.context_ids.push(id);
        }

        Ok(id)
    }

    pub(crate) fn destroy(&mut self, id: ContextId) -> Result<()> {
        let ctx = self.contexts.remove(&id)
            .ok_or_else(|| anyhow::anyhow!("Context {id} not found"))?;

        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let committed_len = ctx.committed_len;

        if let Some(pid) = ctx.owner {
            let proc = self.process_entry(pid);
            proc.context_ids.retain(|&c| c != id);
            let d = proc.device_mut(dev_idx);
            d.committed -= committed_len;
            d.working -= ctx.working_pages.len();

            // Cancel deferred_op if it references the destroyed context.
            let cancel_deferred = match &proc.deferred_op {
                Some(DeferredOp::Alloc(a)) => a.context_id == id,
                Some(DeferredOp::Pin { context_id, .. }) => *context_id == id,
                None => false,
            };
            if cancel_deferred {
                match proc.deferred_op.take() {
                    Some(DeferredOp::Alloc(a)) => { let _ = a.response.send(Err(anyhow::anyhow!("Context destroyed"))); }
                    Some(DeferredOp::Pin { response, .. }) => { let _ = response.send(Err(anyhow::anyhow!("Context destroyed"))); }
                    None => {}
                }
            }

            // Clean up empty process entries
            if proc.context_ids.is_empty() {
                self.processes.remove(&pid);
            }
        }

        // Clean stale AllocWaiter entries from try_alloc for this context
        let mut i = 0;
        while i < self.try_alloc.len() {
            if self.try_alloc[i].context_id == id {
                let waiter = self.try_alloc.remove(i).unwrap();
                let _ = waiter.response.send(Err(anyhow::anyhow!("Context destroyed")));
            } else {
                i += 1;
            }
        }

        // Release committed chain (skip if already released during suspension)
        if let Some(tip_hash) = ctx.committed_tip {
            let dev = &mut self.devices[dev_idx];
            if !ctx.is_suspended() {
                dev.release_chain(tip_hash);
            }
            dev.remove_index_cache(tip_hash);
        }

        // Free working pages (GPU or CPU depending on state)
        if ctx.is_suspended() {
            self.devices[dev_idx].free_cpu_pages(&ctx.working_pages);
        } else {
            self.devices[dev_idx].free_gpu_pages(&ctx.working_pages);
        }

        self.snapshots.retain(|_, v| *v != id);

        self.drain_queues();
        Ok(())
    }

    pub(crate) fn fork(&mut self, id: ContextId) -> Result<ContextId> {
        let ctx = self.contexts.get(&id).ok_or_else(|| anyhow::anyhow!("Context not found"))?;
        let owner = ctx.owner;
        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let device = dev_idx as DeviceId;
        let src_on_gpu = !ctx.is_suspended();

        // Snapshot source state.
        let tip = ctx.committed_tip;
        let committed_len = ctx.committed_len;
        let max_pos = ctx.max_committed_position;
        let lineage = ctx.lineage.clone();
        let forked_tokens = ctx.working_page_tokens.clone();

        //// REVIEW: I feel like clone here is not needed
        let src = ctx.working_pages.clone();

        if let Some(h) = tip {
            self.devices[dev_idx].acquire_chain(h);
        }

        // Allocate destination pages: prefer GPU, fall back to CPU.
        let dev = &mut self.devices[dev_idx];
        let (dst, dst_on_gpu) = if src.is_empty() {
            (Vec::new(), true)
        } else if let Some(p) = dev.alloc_gpu_pages(src.len()) {
            (p, true)
        } else if let Some(p) = dev.alloc_cpu_pages(src.len()) {
            (p, false)
        } else {
            anyhow::bail!("Out of memory")
        };

        // Copy source → destination.
        if !src.is_empty() {
            let _ = match (src_on_gpu, dst_on_gpu) {
                (true,  true)  => device::copy_d2d(device, &src, &dst),
                (true,  false) => device::copy_d2h(device, &src, &dst),
                (false, true)  => device::copy_h2d(device, &dst, &src),
                (false, false) => device::copy_h2h(device, &src, &dst),
            };
        }

        let suspended = !dst_on_gpu;
        if suspended {
            if let Some(h) = tip { self.devices[dev_idx].release_chain(h); }
        }

        let new_id = self.next_id();
        self.contexts.insert(new_id, Context {
            owner,
            device: Some(device),
            working_pages: dst,
            committed_tip: tip,
            committed_len,
            max_committed_position: max_pos,
            lineage,
            working_page_tokens: forked_tokens,
            state: if suspended { ContextState::Suspended } else { ContextState::Active },
            pending_suspend: false,
            last_access: Instant::now(),
        });

        if let Some(pid) = owner {
            let proc = self.process_entry(pid);
            proc.context_ids.push(new_id);
            if !suspended {
                let d = proc.device_mut(dev_idx);
                d.working += src.len();
                d.committed += committed_len;
            }
        }

        Ok(new_id)
    }

    // ==================== Page Management ====================

    /// Handle a ReserveWorkingPages message per DESIGN.md §4 (steps 0–6).
    pub(crate) fn request_reserve_working_pages(
        &mut self,
        id: ContextId,
        num_pages: usize,
        response: oneshot::Sender<anyhow::Result<()>>,
    ) {
        if num_pages == 0 {
            let _ = response.send(Ok(()));
            return;
        }

        let ctx = match self.contexts.get(&id) {
            Some(c) => c,
            None => { let _ = response.send(Err(anyhow::anyhow!("Context not found"))); return; }
        };
        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let owner = ctx.owner;

        // num_pages is the additional number of pages to allocate.
        let additional = num_pages;

        // Alloc is only possible through an owning process.
        let pid = match owner {
            Some(pid) => pid,
            None => { let _ = response.send(Err(anyhow::anyhow!("Context has no owning process"))); return; }
        };

        // Step 0: SUSPENSION CHECK
        // If the owning process is Pending, store as deferred_op.
        let proc = self.process_entry(pid);
        if proc.state == sched::ProcessState::Pending {
            proc.deferred_op = Some(DeferredOp::Alloc(AllocWaiter {
                context_id: id, device: dev_idx,
                num_pages: additional, response,
            }));
            return;
        }

        // Step 1: FIFO GATE — if try_alloc has pending requests, enqueue behind them.
        if !self.try_alloc.is_empty() {
            self.try_alloc.push_back(suspend::AllocWaiter {
                context_id: id, device: dev_idx,
                num_pages: additional, response,
            });
            return;
        }

        // Step 2: PRIORITY GATE — compare requester floor vs try_restore head.
        let requester_floor = {
            let proc = self.process_entry(pid);
            let pages = proc.pages_on(dev_idx);
            proc.weight * (pages + additional) as f64
        };

        if let Some(top) = self.try_restore.peek() {
            let top_pid = top.process_id;
            let top_ready = self.processes.get(&top_pid).map(|p| p.pending_pinned == 0).unwrap_or(true);
            if top_ready && requester_floor < top.effective_priority() {
                // Requester loses priority gate → suspend and enqueue in try_restore.
                let (pinned, _) = self.suspend_process(pid);
                self.enqueue_restore(pid, requester_floor, pinned);
                self.process_entry(pid).deferred_op = Some(DeferredOp::Alloc(AllocWaiter {
                    context_id: id, device: dev_idx,
                    num_pages: additional, response,
                }));
                return;
            }
        }

        // Step 3: TRY ALLOCATE from free pool.
        if self.reserve_working_pages(id, dev_idx, additional, Some(pid)).is_ok() {
            let _ = response.send(Ok(()));
            return;
        }


        // Step 4: EVICTION LOOP — with deferred page tracking.
        // Instead of breaking on first Pinned encounter, we continue evicting
        // and track how many pages will eventually be freed when Pinned contexts
        // clear. We stop once free + deferred >= needed.
        let mut deferred_pages: usize = 0;
        let mut has_deferred = false;

        loop {
            match self.find_eviction_victim(dev_idx, requester_floor, Some(pid)) {
                Some(victim_pid) => {
                    let victim_floor = self.processes.get(&victim_pid)
                        .map(|e| e.priority_on(dev_idx)).unwrap_or(0.0);
                    let estimate = self.estimate_suspend(victim_pid, dev_idx);
                    let (pinned, _) = self.suspend_process(victim_pid);
                    self.enqueue_restore(victim_pid, victim_floor, pinned);

                    // Accumulate deferred page count from Pinned contexts.
                    if estimate.pinned_count > 0 {
                        deferred_pages += estimate.total_deferred();
                        has_deferred = true;
                    }

                    // Retry alloc after victim suspension freed pages.
                    if self.reserve_working_pages(id, dev_idx, additional, Some(pid)).is_ok() {
                        let _ = response.send(Ok(()));
                        self.drain_queues();
                        return;
                    }

                    // Check: will deferred pages close the gap?
                    if has_deferred {
                        let free_now = self.devices[dev_idx].available_gpu_pages();
                        if free_now + deferred_pages >= additional {
                            break; // Deferred pages will cover it
                        }
                    }
                    // Not enough even with deferred — keep looking for victims.
                }
                None => break,
            }
        }

        if has_deferred {
            // Deferred pages from Pinned contexts will cover the gap.
            // Enqueue in try_alloc (requester stays Running).
            self.try_alloc.push_back(suspend::AllocWaiter {
                context_id: id, device: dev_idx,
                num_pages: additional, response,
            });
        } else {
            // Step 5: NO VICTIM — requester self-suspends.
            let (pinned, _) = self.suspend_process(pid);
            self.enqueue_restore(pid, requester_floor, pinned);
            self.process_entry(pid).deferred_op = Some(DeferredOp::Alloc(AllocWaiter {
                context_id: id, device: dev_idx,
                num_pages: additional, response,
            }));
        }
    }

    /// Try to allocate GPU working pages for a context.
    pub(crate) fn reserve_working_pages(
        &mut self,
        id: ContextId,
        dev_idx: usize,
        num_pages: usize,
        owner: Option<ProcessId>,
    ) -> Result<()> {
        let pages = self.devices[dev_idx].alloc_gpu_pages(num_pages)
            .ok_or_else(|| anyhow::anyhow!("Insufficient GPU pages (need {num_pages})"))?;
        let n = pages.len();
        if let Some(ctx) = self.contexts.get_mut(&id) {
            ctx.working_pages.extend(pages);
            ctx.device = Some(dev_idx);
        }
        if let Some(pid) = owner {
            self.process_entry(pid).device_mut(dev_idx).working += n;
        }
        Ok(())
    }

    

    pub(crate) fn release_working_pages(&mut self, id: ContextId, num_pages: usize) -> Result<()> {
        let ctx = self.contexts.get_mut(&id).ok_or_else(|| anyhow::anyhow!("Context not found"))?;
        if num_pages == 0 { return Ok(()); }
        if num_pages > ctx.working_pages.len() {
            anyhow::bail!("release: requested {num_pages}, have {}", ctx.working_pages.len());
        }

        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let owner = ctx.owner;
        let suspended = ctx.is_suspended();

        let start = ctx.working_pages.len() - num_pages;
        let to_free: Vec<_> = ctx.working_pages.drain(start..).collect();

        // Truncate working page tokens
        let tokens_to_remove = num_pages * self.page_size;
        let len = ctx.working_page_tokens.len();
        ctx.working_page_tokens.truncate(len.saturating_sub(tokens_to_remove));

        if suspended {
            self.devices[dev_idx].free_cpu_pages(&to_free);
        } else {
            self.devices[dev_idx].free_gpu_pages(&to_free);
            if let Some(pid) = owner {
                self.process_entry(pid).device_mut(dev_idx).working -= num_pages;
            }
        }
        self.drain_queues();
        Ok(())
    }

    pub(crate) fn commit_working_pages(&mut self, id: ContextId, num_pages: usize) -> Result<()> {
        let page_size = self.page_size;
        let ctx = self.contexts.get(&id).ok_or_else(|| anyhow::anyhow!("Context not found"))?;
        if num_pages == 0 { return Ok(()); }
        if num_pages > ctx.working_pages.len() {
            anyhow::bail!("commit: requested {num_pages}, have {}", ctx.working_pages.len());
        }

        let total_tokens = num_pages * page_size;
        if total_tokens > ctx.working_page_tokens.len() {
            anyhow::bail!("commit: need {total_tokens} tokens, have {}", ctx.working_page_tokens.len());
        }

        let suspended = ctx.is_suspended();
        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let old_tip = ctx.committed_tip;
        let prev_hash = old_tip.unwrap_or(0);
        let owner = ctx.owner;
        let pages = ctx.working_pages[..num_pages].to_vec();

        // Extract token data for the pages being committed.
        let mut tokens = Vec::with_capacity(total_tokens);
        let mut positions = Vec::with_capacity(total_tokens);
        let mut masks = Vec::with_capacity(total_tokens);
        for info in &ctx.working_page_tokens[..total_tokens] {
            tokens.push(info.token);
            positions.push(info.position);
            masks.push(info.mask.clone());
        }

        // Validate positions are strictly after any previously committed position.
        if let Some(max_committed) = ctx.max_committed_position {
            for &pos in &positions {
                if pos <= max_committed {
                    anyhow::bail!("Position {} must be > max committed position {}", pos, max_committed);
                }
            }
        }
        let lineage_adapter = ctx.working_page_tokens.first().and_then(|t| t.adapter);

        // Compute content-based hashes (no physical page IDs needed).
        let hashes = pagestore::compute_page_hashes(page_size, &tokens, &positions, &masks, prev_hash);
        let dev = &mut self.devices[dev_idx];
        let new_tip = *hashes.last()
            .ok_or_else(|| anyhow::anyhow!("No page hashes computed"))?;

        // Commit: physical (GPU promotion + dedup) or logical (chain metadata only).
        if suspended {
            let mut prev = prev_hash;
            for &hash in &hashes {
                dev.insert_chain_link(hash, prev);
                prev = hash;
            }
            dev.free_cpu_pages(&pages);
        } else {
            let mut new_phys = Vec::with_capacity(num_pages);
            let mut prev = prev_hash;
            for (i, &hash) in hashes.iter().enumerate() {

                // Dedup: page already exists on GPU
                let phys = if let Some((existing, rc)) = dev.pages.get_mut(&hash) {
                    *rc += 1;
                    let p = *existing;
                    dev.free_gpu_pages(&[pages[i]]);
                    p
                } else {
                    dev.insert_chain_link(hash, prev);
                    pages[i]
                };
                new_phys.push(phys);
                prev = hash;
            }
            dev.update_index_cache(new_tip, old_tip, &new_phys);
        }

        // Update context state.
        let ctx = self.contexts.get_mut(&id)
            .ok_or_else(|| anyhow::anyhow!("Context lost during commit"))?;
        ctx.working_pages.drain(..num_pages);
        ctx.working_page_tokens.drain(..total_tokens);
        ctx.committed_tip = Some(new_tip);
        ctx.committed_len += num_pages;
        ctx.max_committed_position = positions.iter().copied().max()
            .or(ctx.max_committed_position);

        // Append to lineage (merge with last record if same adapter).
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

        // Scheduler accounting (skip for Suspended — already zeroed).
        if !suspended {
            if let Some(pid) = owner {
                let d = self.process_entry(pid).device_mut(dev_idx);
                d.committed += num_pages;
                d.working -= num_pages;
            }
        }

        self.drain_queues();
        Ok(())
    }

    /// Try to pin a context for a forward pass.
    ///
    /// If the context is Active, transitions to Pinned and sends the result
    /// immediately. If the owning process is Pending (suspended), the pin is
    /// deferred as a `DeferredOp::Pin` — the caller awaits the oneshot and
    /// resumes automatically after restoration.
    pub(crate) fn request_pin(
        &mut self,
        id: ContextId,
        num_input_tokens: u32,
        response: oneshot::Sender<Result<PinnedContext>>,
    ) {
        let ctx = match self.contexts.get(&id) {
            Some(c) => c,
            None => { let _ = response.send(Err(anyhow::anyhow!("Context not found"))); return; }
        };

        // Extract what we need before dropping the immutable borrow.
        let is_active = ctx.is_active();
        let owner = ctx.owner;
        let state = ctx.state;

        // If context is not Active, check if the process is Pending → defer.
        if !is_active {
            if let Some(pid) = owner {
                let proc = self.process_entry(pid);
                if proc.state == sched::ProcessState::Pending {
                    proc.deferred_op = Some(DeferredOp::Pin {
                        context_id: id,
                        num_input_tokens,
                        response,
                    });
                    return;
                }
            }
            let _ = response.send(Err(anyhow::anyhow!("request_pin: context not active (state={:?})", state)));
            return;
        }

        let result = self.pin(id, num_input_tokens);
        let _ = response.send(result);
    }

    /// Core pin logic: resolve physical pages, transition to Pinned.
    /// Accepts Active or Pinned contexts (Pinned for deferred pins replayed
    /// while restore replay is still in-flight).
    /// Called by `request_pin()` directly and by `restore_process()` for deferred pins.
    pub(crate) fn pin(&mut self, id: ContextId, num_input_tokens: u32) -> Result<PinnedContext> {
        let ctx = self.contexts.get_mut(&id).ok_or_else(|| anyhow::anyhow!("Context not found"))?;
        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let tip = ctx.committed_tip;
        let working = ctx.working_pages.clone();
        let kv_len = (ctx.committed_len * self.page_size + ctx.working_page_tokens.len()) as u32;

        if ctx.is_suspended() {
            anyhow::bail!("pin: context is suspended (cannot pin)");
        }
        ctx.state = ContextState::Pinned;

        let mut page_ids = Vec::new();

        // Committed pages
        if let Some(tip_hash) = tip {
            let dev = &mut self.devices[dev_idx];
            page_ids.extend(dev.resolve_physical(tip_hash));
        }

        // Working pages (appended after committed)
        page_ids.extend(&working);

        let num_pages = page_ids.len() as u32;
        let page_size = self.page_size as u32;
        let total_kv = kv_len + num_input_tokens;
        let last_page_len = pagestore::compute_last_page_len(total_kv, num_pages, page_size);

        Ok(PinnedContext { device: dev_idx as DeviceId, pages: page_ids, kv_len, last_page_len })
    }

    /// Handle ClearPinned message: Pinned → Active, then check deferred suspension.
    /// If `pending_suspend` was set, executes the deferred suspension and
    /// decrements `ProcessEntry.pending_pinned`.
    pub(crate) fn unpin(&mut self, id: ContextId) {
        let (is_pinned, pending) = match self.contexts.get(&id) {
            Some(ctx) if ctx.is_pinned() => (true, ctx.pending_suspend),
            _ => return,
        };

        if !is_pinned { return; }

        if pending {
            // Deferred suspension: context stays Pinned until suspend_context
            // transitions it to Suspended.
            let owner = self.contexts.get(&id).and_then(|c| c.owner);
            self.suspend_context(id);

            // Decrement pending_pinned_count for the owning process.
            if let Some(pid) = owner {
                if let Some(proc) = self.processes.get_mut(&pid) {
                    proc.pending_pinned = proc.pending_pinned.saturating_sub(1);
                }
            }

            // Deferred suspension may have freed pages — drain queues.
            self.drain_queues();
            return;
        }

        // No deferred suspension — normal Pinned → Active transition.
        if let Some(ctx) = self.contexts.get_mut(&id) {
            ctx.state = ContextState::Active;
        }
    }

    /// Central queue drain: called after any event that frees GPU pages.
    ///
    /// Phase 1: try_alloc (FIFO) — serve front-of-queue allocations.
    /// Phase 2: try_restore (priority heap) — restore highest-priority Pending
    ///          process, then replay its deferred_op.
    pub(crate) fn drain_queues(&mut self) {
        // Phase 1: try_alloc FIFO (head-of-line blocking)
        while let Some(front) = self.try_alloc.front() {
            let dev_idx = front.device as usize;
            let n = front.num_pages;
            let ctx_id = front.context_id;
            // End borrow of `front` before accessing self.contexts / &mut self.
            let _ = front;

            let owner = self.contexts.get(&ctx_id).and_then(|c| c.owner);
            if self.reserve_working_pages(ctx_id, dev_idx, n, owner).is_ok() {
                let waiter = self.try_alloc.pop_front().unwrap();
                let _ = waiter.response.send(Ok(()));
            } else {
                break; // Not enough pages for front of queue
            }
        }

        // Phase 2: try_restore (priority heap)
        // Only proceed if try_alloc is empty (allocs have strict priority).
        if !self.try_alloc.is_empty() { return; }

        while let Some(top) = self.try_restore.peek() {
            let pid = top.process_id;

            // Block if still has Pinned contexts clearing.
            if self.processes.get(&pid).map(|p| p.pending_pinned).unwrap_or(0) > 0 {
                break;
            }

            // Admission check: enough pages on all devices?
            if !self.can_restore_process(pid) {
                break;
            }

            let waiter = self.try_restore.pop().unwrap();
            if let Err(e) = self.restore_process(waiter.process_id) {
                eprintln!("RESTORE_PROCESS_FAIL pid={} err={e:#}", waiter.process_id);
            }
        }
    }

    // ==================== Extracted handle() helpers ====================


    pub(crate) fn stats(&self) -> Vec<(usize, usize)> {
        self.devices.iter().map(|d| d.stats()).collect()
    }

    pub(crate) fn set_priority_internal(&mut self, weight: f64, pid_values: HashMap<ProcessId, f64>) {
        for (pid, value) in pid_values {
            self.process_entry(pid).weight = weight * value;
        }
        self.drain_queues();
    }

    pub(crate) fn working_page_count(&self, id: ContextId) -> u32 {
        self.contexts.get(&id)
            .map(|c| c.working_pages.len() as u32)
            .unwrap_or(0)
    }

    pub(crate) fn committed_page_count(&self, id: ContextId) -> u32 {
        self.contexts.get(&id)
            .map(|c| c.committed_len as u32)
            .unwrap_or(0)
    }

    pub(crate) fn working_page_token_count(&self, id: ContextId) -> u32 {
        self.contexts.get(&id)
            .map(|c| c.working_page_tokens.len() as u32)
            .unwrap_or(0)
    }

    pub(crate) fn truncate_working_page_tokens(&mut self, id: ContextId, count: u32) -> Result<()> {
        let ctx = self.contexts.get_mut(&id)
            .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
        let max = ctx.working_page_tokens.len();
        if count as usize > max {
            anyhow::bail!("truncate count {} out of range 0..={}", count, max);
        }
        ctx.working_page_tokens.truncate(count as usize);
        Ok(())
    }

    pub(crate) fn append_working_page_tokens(
        &mut self, id: ContextId, tokens: Vec<u32>, positions: Vec<u32>,
        masks: Vec<Brle>, adapter: Option<AdapterId>,
    ) -> Result<()> {
        let n = tokens.len();
        if positions.len() != n {
            anyhow::bail!("positions length {} != n {}", positions.len(), n);
        }
        if !masks.is_empty() && masks.len() != n {
            anyhow::bail!("masks length {} != n {}", masks.len(), n);
        }
        let ctx = self.contexts.get_mut(&id)
            .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
        for (i, token) in tokens.into_iter().enumerate() {
            ctx.working_page_tokens.push(TokenInfo {
                token, position: positions[i],
                mask: if masks.is_empty() { Brle::new(0) } else { masks[i].clone() },
                adapter,
            });
        }
        Ok(())
    }

    pub(crate) fn debug_state(&self, id: ContextId) -> String {
        match self.contexts.get(&id) {
            Some(ctx) => format!("{ctx:?}"),
            None => "NOT_FOUND".to_string(),
        }
    }
}

// =============================================================================
// Message & ServiceHandler
// =============================================================================

#[derive(Debug)]
pub(crate) enum Message {
    Open { username: String, name: String, response: oneshot::Sender<Result<ContextId>> },
    Create { owner: Option<ProcessId>, response: oneshot::Sender<Result<ContextId>> },
    Save { id: ContextId, username: String, name: Option<String>, response: oneshot::Sender<Result<Option<String>>> },
    Delete { username: String, name: String, response: oneshot::Sender<Result<()>> },
    Destroy { id: ContextId, force: bool, response: oneshot::Sender<Result<()>> },
    Fork { id: ContextId, response: oneshot::Sender<Result<ContextId>> },
    Take { username: String, name: String, response: oneshot::Sender<Result<ContextId>> },
    CommitWorkingPages { id: ContextId, num_pages: usize, response: oneshot::Sender<Result<()>> },
    ReserveWorkingPages { id: ContextId, num_pages: usize, response: oneshot::Sender<Result<()>> },
    ReleaseWorkingPages { id: ContextId, num_pages: usize },
    Pin { id: ContextId, num_input_tokens: u32, response: oneshot::Sender<Result<PinnedContext>> },
    Unpin { id: ContextId },
    FinishRestore { id: ContextId },
    GetStats { response: oneshot::Sender<Vec<(usize, usize)>> },

    // Actor-routed read/write APIs (previously DashMap-based)
    WorkingPageCount { id: ContextId, response: oneshot::Sender<u32> },
    CommittedPageCount { id: ContextId, response: oneshot::Sender<u32> },
    WorkingPageTokenCount { id: ContextId, response: oneshot::Sender<u32> },
    TruncateWorkingPageTokens { id: ContextId, count: u32, response: oneshot::Sender<Result<()>> },
    AppendWorkingPageTokens {
        id: ContextId, tokens: Vec<u32>, positions: Vec<u32>,
        masks: Vec<Brle>, adapter: Option<AdapterId>,
        response: oneshot::Sender<Result<()>>,
    },

    DebugState { id: ContextId, response: oneshot::Sender<String> },
    SetPriority { weight: f64, pid_values: HashMap<ProcessId, f64> },
}

impl ServiceHandler for ContextManager {
    type Message = Message;

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::Open { username, name, response } => {
                let _ = response.send(self.open(username, name));
            }
            Message::Take { username, name, response } => {
                let _ = response.send(self.take(username, name));
            }
            Message::Create { owner, response } => {
                let _ = response.send(self.create(owner));
            }
            Message::Save { id, username, name, response } => {
                let _ = response.send(self.save(id, username, name));
            }
            Message::Delete { username, name, response } => {
                let _ = response.send(self.delete(username, name));
            }
            Message::Destroy { id, force: _, response } => {
                let _ = response.send(self.destroy(id));
            }
            Message::Fork { id, response } => {
                let _ = response.send(self.fork(id));
            }
            Message::CommitWorkingPages { id, num_pages, response } => {
                let _ = response.send(self.commit_working_pages(id, num_pages));
            }
            Message::ReserveWorkingPages { id, num_pages, response } => {
                self.request_reserve_working_pages(id, num_pages, response);
            }
            Message::ReleaseWorkingPages { id, num_pages } => {
                let _ = self.release_working_pages(id, num_pages);
            }
            Message::Pin { id, num_input_tokens, response } => {
                self.request_pin(id, num_input_tokens, response);
            }
            Message::Unpin { id } => {
                self.unpin(id);
            }
            Message::FinishRestore { id } => {
                self.finish_restore(id);
            }
            Message::GetStats { response } => {
                let _ = response.send(self.stats());
            }
            Message::SetPriority { weight, pid_values } => {
                self.set_priority_internal(weight, pid_values);
            }
            Message::WorkingPageCount { id, response } => {
                let _ = response.send(self.working_page_count(id));
            }
            Message::CommittedPageCount { id, response } => {
                let _ = response.send(self.committed_page_count(id));
            }
            Message::WorkingPageTokenCount { id, response } => {
                let _ = response.send(self.working_page_token_count(id));
            }
            Message::TruncateWorkingPageTokens { id, count, response } => {
                let _ = response.send(self.truncate_working_page_tokens(id, count));
            }
            Message::AppendWorkingPageTokens { id, tokens, positions, masks, adapter, response } => {
                let _ = response.send(self.append_working_page_tokens(id, tokens, positions, masks, adapter));
            }
            Message::DebugState { id, response } => {
                let _ = response.send(self.debug_state(id));
            }
        }
    }
}
