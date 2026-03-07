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
mod restore;

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
use restore::ReplayFill;
use suspend::{AllocWaiter, RestoreWaiter};

// =============================================================================
// Public Types
// =============================================================================

pub type ContextId = u64;

// =============================================================================
// Globals
// =============================================================================

static SERVICES: LazyLock<ServiceArray<Message>> = LazyLock::new(ServiceArray::new);
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
    /// Physical page IDs for uncommitted (working) pages on GPU.
    pub working_pages: Vec<PhysicalPageId>,
    /// CPU pages for working pages when suspended.
    pub working_pages_cpu: Vec<PhysicalPageId>,
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
            working_pages_cpu: Vec::new(),
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
    /// Accumulated replay chunks from the current message handler.
    /// Flushed via `dispatch_replays` at the end of each `handle()` call.
    pub(crate) pending_replays: Vec<ReplayFill>,
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
            pending_replays: Vec::new(),
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

            // Clean up empty process entries
            if self.processes.get(&pid).map(|p| p.context_ids.is_empty()).unwrap_or(false) {
                if let Some(entry) = self.processes.remove(&pid) {
                    for alloc in entry.pending_allocs {
                        let _ = alloc.response.send(Err(anyhow::anyhow!("Context destroyed")));
                    }
                }
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
            if ctx.state != ContextState::Suspended {
                dev.release_chain(tip_hash);
            }
            dev.remove_index_cache(tip_hash);
        }

        // Free GPU working pages
        self.devices[dev_idx].free_gpu_pages(&ctx.working_pages);
        // Free CPU working pages
        self.devices[dev_idx].free_cpu_pages(&ctx.working_pages_cpu);

        self.snapshots.retain(|_, v| *v != id);

        self.drain_queues();
        Ok(())
    }

    pub(crate) fn fork(&mut self, id: ContextId) -> Result<ContextId> {
        let ctx = self.contexts.get(&id).ok_or_else(|| anyhow::anyhow!("Context not found"))?;
        let owner = ctx.owner;
        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let device = dev_idx as DeviceId;

        // Snapshot source state (borrows released before &mut self calls).
        let tip = ctx.committed_tip;
        let committed_len = ctx.committed_len;
        let max_pos = ctx.max_committed_position;
        let lineage = ctx.lineage.clone();
        let forked_tokens = ctx.working_page_tokens.clone();

        // Determine source working pages: GPU-resident if active, CPU if suspended.
        let (src, on_gpu) = if !ctx.working_pages.is_empty() {
            (ctx.working_pages.clone(), true)
        } else if !ctx.working_pages_cpu.is_empty() {
            (ctx.working_pages_cpu.clone(), false)
        } else {
            (Vec::new(), true) // no working pages at all
        };

        // Share the committed chain with the fork.
        if let Some(h) = tip {
            self.devices[dev_idx].acquire_chain(h);
        }

        // Clone working pages: try GPU first, fall back to CPU (→ suspended).
        let dev = &mut self.devices[dev_idx];
        let (fork_gpu, fork_cpu, suspended) = if src.is_empty() {
            (Vec::new(), Vec::new(), false)
        } else if let Some(dst) = dev.alloc_gpu_pages(src.len()) {
            let _ = if on_gpu {
                device::copy_d2d(device, &src, &dst)
            } else {
                device::copy_h2d(device, &dst, &src)
            };
            (dst, Vec::new(), false)
        } else if let Some(cpu) = dev.alloc_cpu_pages(src.len()) {
            let _ = if on_gpu {
                device::copy_d2h(device, &src, &cpu)
            } else {
                device::copy_h2h(device, &src, &cpu)
            };
            (Vec::new(), cpu, true)
        } else {
            anyhow::bail!("Out of memory")
        };

        // Suspended fork can't use the committed chain — release it.
        if suspended {
            if let Some(h) = tip { self.devices[dev_idx].release_chain(h); }
        }

        let n_working = fork_gpu.len();
        let new_id = self.next_id();
        self.contexts.insert(new_id, Context {
            owner,
            device: Some(device),
            working_pages: fork_gpu,
            working_pages_cpu: fork_cpu,
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
                d.working += n_working;
                d.committed += committed_len;
            }
        }

        Ok(new_id)
    }

    // ==================== Page Management ====================

    pub(crate) fn release_working_pages(&mut self, id: ContextId, num_pages: usize) -> Result<()> {
        let ctx = self.contexts.get_mut(&id).ok_or_else(|| anyhow::anyhow!("Context not found"))?;
        if num_pages == 0 { return Ok(()); }

        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let owner = ctx.owner;
        let suspended = ctx.state == ContextState::Suspended;

        // Determine source: GPU working pages (Active/Pinned) or CPU pages (Suspended)
        let (to_free, on_cpu) = if !ctx.working_pages.is_empty() {
            if num_pages > ctx.working_pages.len() {
                anyhow::bail!("release: requested {num_pages}, have {}", ctx.working_pages.len());
            }
            let start = ctx.working_pages.len() - num_pages;
            (ctx.working_pages.drain(start..).collect::<Vec<_>>(), false)
        } else if !ctx.working_pages_cpu.is_empty() {
            if num_pages > ctx.working_pages_cpu.len() {
                anyhow::bail!("release: requested {num_pages}, have {}", ctx.working_pages_cpu.len());
            }
            let start = ctx.working_pages_cpu.len() - num_pages;
            (ctx.working_pages_cpu.drain(start..).collect::<Vec<_>>(), true)
        } else {
            anyhow::bail!("release: no working pages");
        };

        // Truncate working page tokens
        let tokens_to_remove = num_pages * self.page_size;
        if tokens_to_remove > 0 {
            let len = ctx.working_page_tokens.len();
            ctx.working_page_tokens.truncate(len.saturating_sub(tokens_to_remove));
        }

        if on_cpu {
            self.devices[dev_idx].free_cpu_pages(&to_free);
        } else {
            self.devices[dev_idx].free_gpu_pages(&to_free);
        }

        // Scheduler accounting: skip for Suspended (already zeroed during suspension)
        if !suspended {
            if let Some(pid) = owner {
                let d = self.process_entry(pid).device_mut(dev_idx);
                d.working -= num_pages;
            }
        }
        self.drain_queues();
        Ok(())
    }

    pub(crate) fn commit_working_pages(&mut self, id: ContextId, num_pages: usize) -> Result<()> {
        let page_size = self.page_size;
        let ctx = self.contexts.get(&id).ok_or_else(|| anyhow::anyhow!("Context not found"))?;

        if num_pages == 0 { return Ok(()); }

        // Determine source: GPU working pages (Active/Pinned) or CPU pages (Suspended)
        let suspended = ctx.state == ContextState::Suspended;
        let source_len = if !ctx.working_pages.is_empty() {
            ctx.working_pages.len()
        } else if !ctx.working_pages_cpu.is_empty() {
            ctx.working_pages_cpu.len()
        } else {
            anyhow::bail!("Cannot commit: context has no working pages");
        };

        if num_pages > source_len {
            anyhow::bail!("Cannot commit {} pages, only {} working pages available", num_pages, source_len);
        }
        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let old_tip = ctx.committed_tip;
        let prev_hash = ctx.committed_tip.unwrap_or(0);
        let working_phys: Vec<PhysicalPageId> = ctx.working_pages.get(..num_pages)
            .map(|s| s.to_vec()).unwrap_or_default();
        let working_cpu: Vec<PhysicalPageId> = if suspended {
            ctx.working_pages_cpu.get(..num_pages).map(|s| s.to_vec()).unwrap_or_default()
        } else {
            Vec::new()
        };
        let owner = ctx.owner;

        // Read token data from the Context
        let mut tokens = Vec::new();
        let mut positions = Vec::new();
        let mut masks = Vec::new();
        let mut all_positions = Vec::new();

        let total_needed = num_pages * page_size;
        if total_needed > ctx.working_page_tokens.len() {
            anyhow::bail!("Cannot commit {} pages: need {} tokens but only have {}",
                num_pages, total_needed, ctx.working_page_tokens.len());
        }
        for i in 0..total_needed {
            let info = &ctx.working_page_tokens[i];
            tokens.push(info.token);
            positions.push(info.position);
            masks.push(info.mask.clone());
            all_positions.push(info.position);
        }

        if let Some(max_committed) = ctx.max_committed_position {
            for &pos in &all_positions {
                if pos <= max_committed {
                    anyhow::bail!("Position {} must be > max committed position {}", pos, max_committed);
                }
            }
        }
        let lineage_adapter = ctx.working_page_tokens.first().and_then(|t| t.adapter);

        // Compute hashes (content-only, no physical page IDs needed)
        let dev = &mut self.devices[dev_idx];
        let hashes = dev.compute_page_hashes(&tokens, &positions, &masks, prev_hash);

        let new_tip = *hashes.last()
            .ok_or_else(|| anyhow::anyhow!("No page hashes computed during commit"))?;

        if suspended {
            // Logical commit: register chain links only (no GPU page entries).
            // The restore path will detect these as non-resident and replay them.
            let mut running_prev = prev_hash;
            for &hash in &hashes {
                dev.insert_chain_link(hash, running_prev);
                running_prev = hash;
            }
            // Free CPU pages that held the working data
            dev.free_cpu_pages(&working_cpu);
            // No index_cache update — no GPU physical pages to cache.
        } else {
            // Physical commit: promote GPU working pages via commit_working (dedup).
            let mut new_phys = Vec::new();
            let mut running_prev = prev_hash;
            for (i, &hash) in hashes.iter().enumerate() {
                let (phys, _freed) = dev.commit_working(hash, running_prev, working_phys[i]);
                new_phys.push(phys);
                running_prev = hash;
            }
            dev.update_index_cache(new_tip, old_tip, &new_phys);
        }

        // Update Context (local)
        let ctx = self.contexts.get_mut(&id)
            .ok_or_else(|| anyhow::anyhow!("Context lost during commit"))?;
        if suspended {
            ctx.working_pages_cpu.drain(..num_pages);
        } else {
            ctx.working_pages.drain(..num_pages);
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

        // Update token data on Context
        ctx.working_page_tokens.drain(..total_needed);
        ctx.committed_len += num_pages;
        ctx.max_committed_position = all_positions.iter().copied().max()
            .or(ctx.max_committed_position);

        // Scheduler accounting: skip for Suspended (already zeroed during suspension)
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

    pub(crate) fn pin(&mut self, id: ContextId, num_input_tokens: u32) -> Result<PinnedContext> {
        let ctx = self.contexts.get_mut(&id).ok_or_else(|| anyhow::anyhow!("Context not found"))?;
        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let tip = ctx.committed_tip;
        let working = ctx.working_pages.clone();
        let kv_len = (ctx.committed_len * self.page_size + ctx.working_page_tokens.len()) as u32;

        // Transition Active → Pinned: prevents eviction while process holds page IDs.
        // Eviction of Pinned contexts is deferred via `pending_suspend` flag.
        if ctx.state == ContextState::Active {
            ctx.state = ContextState::Pinned;
        }

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

    // ==================== Extracted handle() helpers ====================

    pub(crate) fn open(&mut self, username: String, name: String) -> Result<ContextId> {
        match self.snapshots.get(&(username, name)) {
            Some(&snapshot_id) => self.fork(snapshot_id),
            None => Err(anyhow::anyhow!("Snapshot not found")),
        }
    }

    pub(crate) fn unpin(&mut self, id: ContextId) {
        self.handle_clear_pinned(id);
    }

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
                self.handle_reserve_working_pages(id, num_pages, response);
            }
            Message::ReleaseWorkingPages { id, num_pages } => {
                let _ = self.release_working_pages(id, num_pages);
            }
            Message::Pin { id, num_input_tokens, response } => {
                let _ = response.send(self.pin(id, num_input_tokens));
            }
            Message::Unpin { id } => {
                self.unpin(id);
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
        // Single flush: dispatch all replay chunks accumulated during this message.
        let replays = std::mem::take(&mut self.pending_replays);
        self.dispatch_replays(replays).await;
    }
}
