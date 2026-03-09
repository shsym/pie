//! # Context Module — KV Cache Management
//!
//! Manages execution contexts with KV cache state for model inference.
//! Each model gets a dedicated `ContextManager` actor. All state lives
//! actor-locally — no interior mutability, no cross-actor locks.
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────┐
//! │                 Context Actor (per model)                     │
//! │                                                              │
//! │  ┌──────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐ │
//! │  │PageStore  │  │ ProcessEntry│  │alloc_queue│  │restore_queue│
//! │  │(per device)│  │(scheduling)│  │  (FIFO)   │  │(pri heap) │ │
//! │  └──────────┘  └───────────┘  └───────────┘  └───────────┘ │
//! └──────────────────────────────────────────────────────────────┘
//!         ▲                                      │
//!         │  reserve / commit / destroy          │ suspend / restore
//!         │  pin / unpin                         ▼
//! ┌──────────────────────────────────────────────────────────────┐
//! │  Process (inferlet)                                          │
//! │  Single-threaded WASM — blocked on response channel when     │
//! │  enqueued. Cannot make other WIT calls until response.       │
//! └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## State Model
//!
//! **Process** (2 states):
//! - **Running** — all contexts Active, inferlet executing.
//! - **Pending** — contexts Suspended, blocked on restore_queue.
//!
//! **Context** (3 states):
//!
//! | State         | Working Pages | Committed Chain      | Evictable?           |
//! |---------------|---------------|----------------------|----------------------|
//! | **Active**    | On GPU        | Refcounted in trie   | Yes                  |
//! | **Pinned**    | On GPU        | Refcounted in trie   | Deferred (`pending_suspend`) |
//! | **Suspended** | On CPU (swap) | Released (metadata)  | Nothing to evict     |
//!
//! ## Page Types
//!
//! **Working pages** (`working_pages: Vec<PhysicalPageId>`):
//! - GPU-exclusive, mutable, owned by one context.
//! - On suspend: D2H copy to CPU swap pool. On restore: H2D back.
//! - If CPU swap pool full: OOM. Working pages are NOT replayable.
//!
//! **Committed pages** (content-addressed via chained `PageHash`):
//! - Shared across contexts via refcount in PageStore (Radix Trie).
//! - `committed_hashes: Vec<PageHash>` — ordered root-to-tip chain.
//! - On suspend: refcounts decremented; hashes kept as metadata for restore.
//! - On restore: longest prefix match → replay missing suffix.
//!
//! ## Module Structure
//!
//! - `context.rs` — Public API, `Message` enum, `ServiceHandler`, core ops.
//! - `pagestore.rs` — `PageStore`: Radix Trie CAS cache + physical page pools.
//! - `sched.rs` — `ProcessEntry`, invested-importance scheduling (π = w·p).
//! - `contention.rs` — Eviction, suspension, `alloc_queue`, `with_gpu_pages`.
//! - `restore.rs` — Restoration, replay planning, `restore_queue`.
//! - `snapshot.rs` — Named snapshot save/load/fork/take.
//!
//! ## WIT Host Function Pattern
//!
//! Every WIT function that touches pages goes through the actor:
//! ```text
//! async fn wit_reserve_pages(pid, ctx_id, n) -> Result<()> {
//!     context::reserve_pages(model_idx, ctx_id, n).await
//! }
//! ```
//! No `wait_if_pending` needed. The process is single-threaded WASM —
//! when `reserve_pages` gets enqueued, the process blocks on `.await`.
//! When the actor serves the request (from alloc_queue or after restore),
//! it sends Ok and the process resumes, already Running.
// Radix Trie with path-inclusive refcounting — handles incremental
// commits and dedup correctly, all operations O(depth).
pub mod pagestore;
pub(crate) mod sched;
mod contention;
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

use contention::{PendingAlloc, PendingRestore};

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

pub async fn lookup_snapshot(model_idx: usize, username: String, name: String) -> Result<ContextId> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::LookupSnapshot { username, name, response: tx })?;
    rx.await.context("context::lookup_snapshot: actor dropped response")?
}

pub async fn take(model_idx: usize, username: String, name: String, owner: ProcessId) -> Result<ContextId> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Take { username, name, owner, response: tx })?;
    rx.await.context("context::take: actor dropped response")?
}

pub async fn create(model_idx: usize, owner: ProcessId) -> Result<ContextId> {
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

pub async fn destroy(model_idx: usize, id: ContextId) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Destroy { id, response: tx })?;
    rx.await.context("context::destroy: actor dropped response")?
}

/// Destroy all contexts owned by a process across all models.
/// Called on WASM instance drop for automatic cleanup.
pub fn destroy_all(pid: ProcessId) {
    for model_idx in 0..SERVICES.len() {
        let _ = SERVICES.send(model_idx, Message::DestroyAll { pid });
    }
}

pub async fn fork(model_idx: usize, id: ContextId, owner: ProcessId) -> Result<ContextId> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Fork { id, owner, response: tx })?;
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
    adapter_seed: Option<i64>,
) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::AppendWorkingPageTokens {
        id, tokens, positions, masks, adapter, adapter_seed, response: tx,
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
    pub adapter_seed: Option<i64>,
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
        adapter_seed: Option<i64>,
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
    /// Ordered committed page hashes (root-to-tip). Replaces committed_tip + committed_len.
    pub committed_hashes: Vec<PageHash>,
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
            committed_hashes: Vec::new(),
            lineage: Vec::new(),
            working_page_tokens: Vec::new(),
            max_committed_position: None,
            state: ContextState::Active,
            pending_suspend: false,
            last_access: Instant::now(),
        }
    }

    pub fn is_active(&self) -> bool { self.state == ContextState::Active }
    pub fn is_suspended(&self) -> bool { self.state == ContextState::Suspended }
    pub fn is_pinned(&self) -> bool { self.state == ContextState::Pinned }

    /// Tip of the committed hash chain (last element), or None if empty.
    pub fn committed_tip(&self) -> Option<PageHash> { self.committed_hashes.last().copied() }
    /// Number of committed pages.
    pub fn committed_len(&self) -> usize { self.committed_hashes.len() }
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
    /// FIFO queue: deferred GPU page requests from Running processes.
    pub(crate) alloc_queue: VecDeque<PendingAlloc>,
    /// Priority heap: Pending processes waiting for restoration.
    pub(crate) restore_queue: BinaryHeap<PendingRestore>,
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
            alloc_queue: VecDeque::new(),
            restore_queue: BinaryHeap::new(),
        }
    }

    fn next_id(&mut self) -> ContextId {
        let id = self.next_id; self.next_id += 1; id
    }

    fn least_loaded_device(&self) -> usize {
        self.devices.iter().enumerate()
            .max_by_key(|(_, d)| d.available_gpu_pages())
            .map(|(i, _)| i).unwrap_or(0)
    }

    // ==================== Core Operations ====================

    pub(crate) fn create(&mut self, owner: ProcessId) -> Result<ContextId> {
        let id = self.next_id();
        let mut ctx = Context::new(Some(owner));
        ctx.device = Some(self.least_loaded_device());

        self.contexts.insert(id, ctx);

        let proc = self.process_entry(owner);
        proc.context_ids.push(id);

        Ok(id)
    }

    pub(crate) fn destroy(&mut self, id: ContextId) -> Result<()> {
        let ctx = self.contexts.remove(&id)
            .ok_or_else(|| anyhow::anyhow!("Context {id} not found"))?;

        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let committed_len = ctx.committed_len();

        if let Some(pid) = ctx.owner {
            let proc = self.process_entry(pid);
            proc.context_ids.retain(|&c| c != id);

            // Pinned context with deferred suspension: decrement pending_pinned
            // so drain_queues doesn't block on a context that no longer exists.
            if ctx.is_pinned() && ctx.pending_suspend {
                proc.pending_pinned = proc.pending_pinned.saturating_sub(1);
            }

            // Only decrement accounting if context was not Suspended
            // (suspend_process already zeroed device pages).
            if !ctx.is_suspended() {
                let d = proc.device_mut(dev_idx);
                d.committed -= committed_len;
                d.working -= ctx.working_pages.len();
            }

            // Drop deferred ops that reference the destroyed context.
            // Dropping the closure drops the captured Sender, closing the channel.
            proc.deferred_ops.retain(|op| op.context_id != Some(id));

            // Clean up empty process entries
            if proc.context_ids.is_empty() {
                self.processes.remove(&pid);
            }
        }

        // Drop stale PendingAlloc entries from alloc_queue for this context.
        self.alloc_queue.retain(|pa| pa.context_id != Some(id));

        // Release committed chain (skip if already released during suspension)
        if !ctx.committed_hashes.is_empty() && !ctx.is_suspended() {
            self.devices[dev_idx].release(&ctx.committed_hashes);
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

    /// Destroy all contexts owned by a process.
    /// Drops all deferred ops (closing channels), frees all resources, removes the process entry.
    pub(crate) fn destroy_all(&mut self, pid: ProcessId) {
        let proc = match self.processes.remove(&pid) {
            Some(p) => p,
            None => return,
        };

        // Drop stale PendingAlloc entries from alloc_queue for this process's contexts.
        let ctx_ids: std::collections::HashSet<ContextId> = proc.context_ids.iter().copied().collect();
        self.alloc_queue.retain(|pa| {
            !pa.context_id.map_or(false, |cid| ctx_ids.contains(&cid))
        });

        // Remove from restore_queue
        self.restore_queue.retain(|w| w.process_id != pid);

        // Destroy all owned contexts
        for ctx_id in proc.context_ids {
            if let Some(ctx) = self.contexts.remove(&ctx_id) {
                let dev_idx = ctx.device.unwrap_or(0) as usize;
                if !ctx.committed_hashes.is_empty() && !ctx.is_suspended() {
                    self.devices[dev_idx].release(&ctx.committed_hashes);
                }
                if ctx.is_suspended() {
                    self.devices[dev_idx].free_cpu_pages(&ctx.working_pages);
                } else {
                    self.devices[dev_idx].free_gpu_pages(&ctx.working_pages);
                }
                self.snapshots.retain(|_, v| *v != ctx_id);
            }
        }

        self.drain_queues();
    }
    

    // ==================== Page Management ====================

    /// Handle a ReserveWorkingPages message per DESIGN.md §4.
    /// Delegates to the universal `with_gpu_pages` contention primitive.
    pub(crate) fn reserve_working_pages(
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

        // Alloc is only possible through an owning process.
        let pid = match owner {
            Some(pid) => pid,
            None => { let _ = response.send(Err(anyhow::anyhow!("Context has no owning process"))); return; }
        };

        self.with_gpu_pages(pid, dev_idx, num_pages, Some(id), move |mgr, pages| {
            let n = pages.len();
            if let Some(ctx) = mgr.contexts.get_mut(&id) {
                ctx.working_pages.extend_from_slice(&pages);
                ctx.device = Some(dev_idx);
            }
            mgr.process_entry(pid).device_mut(dev_idx).working += n;
            let _ = response.send(Ok(()));
        });
    }

    pub(crate) fn release_working_pages(&mut self, id: ContextId, num_pages: usize) -> Result<()> {
        let ctx = self.contexts.get_mut(&id).ok_or_else(|| anyhow::anyhow!("Context not found"))?;
        if num_pages == 0 { return Ok(()); }
        if num_pages > ctx.working_pages.len() {
            anyhow::bail!("release: requested {num_pages}, have {}", ctx.working_pages.len());
        }

        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let start = ctx.working_pages.len() - num_pages;
        let to_free: Vec<_> = ctx.working_pages.drain(start..).collect();

        // Truncate working page tokens
        let tokens_to_remove = num_pages * self.page_size;
        let len = ctx.working_page_tokens.len();
        ctx.working_page_tokens.truncate(len.saturating_sub(tokens_to_remove));

        if ctx.is_suspended() {
            self.devices[dev_idx].free_cpu_pages(&to_free);
        } else {
            self.devices[dev_idx].free_gpu_pages(&to_free);
            if let Some(pid) = ctx.owner {
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

        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let prev_hash = ctx.committed_tip().unwrap_or(0);
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
        let lineage_adapter_seed = ctx.working_page_tokens.first().and_then(|t| t.adapter_seed);

        // Compute content-based hashes (includes adapter_seed so ZO-perturbed
        // pages are not shared with unperturbed or differently-seeded pages).
        let hashes = pagestore::compute_page_hashes(page_size, &tokens, &positions, &masks, prev_hash, lineage_adapter_seed);
        let existing_prefix = ctx.committed_hashes.clone();
        let dev = &mut self.devices[dev_idx];

        // Commit: physical (GPU promotion + dedup) or logical (metadata only).
        if ctx.is_suspended() {
            // Suspended: no GPU pages, just free the CPU working pages.
            dev.free_cpu_pages(&pages);
        } else {
            // Use extend to navigate the trie through the existing
            // committed chain before inserting new pages as children.
            dev.extend(&existing_prefix, &hashes, &pages);
        }

        // Update context state.
        let ctx = self.contexts.get_mut(&id)
            .ok_or_else(|| anyhow::anyhow!("Context lost during commit"))?;
        ctx.working_pages.drain(..num_pages);
        ctx.working_page_tokens.drain(..total_tokens);
        ctx.committed_hashes.extend_from_slice(&hashes);
        ctx.max_committed_position = positions.iter().copied().max()
            .or(ctx.max_committed_position);

        // Append to lineage (merge with last record if same adapter AND seed).
        if let Some(Record::Fill { tokens: t, positions: p, mask: m, adapter: a, adapter_seed: s }) = ctx.lineage.last_mut() {
            if *a == lineage_adapter && *s == lineage_adapter_seed {
                t.extend_from_slice(&tokens);
                p.extend_from_slice(&positions);
                m.extend_from_slice(&masks);
            } else {
                ctx.lineage.push(Record::Fill { tokens, positions, mask: masks, adapter: lineage_adapter, adapter_seed: lineage_adapter_seed });
            }
        } else {
            ctx.lineage.push(Record::Fill { tokens, positions, mask: masks, adapter: lineage_adapter, adapter_seed: lineage_adapter_seed });
        }

        // Scheduler accounting (skip for Suspended — already zeroed).
        if !ctx.is_suspended() {
            if let Some(pid) = ctx.owner {
                let d = self.process_entry(pid).device_mut(dev_idx);
                d.committed += num_pages;
                d.working -= num_pages;
            }
        }

        self.drain_queues();
        Ok(())
    }

    /// Handle a Pin request: transition an Active context to Pinned.
    ///
    /// Resolves physical page IDs for committed + working pages, computes
    /// `last_page_len`, and returns `PinnedContext` for the inference forward pass.
    /// If the owning process is Pending, defers the operation until restoration.
    pub(crate) fn pin(
        &mut self,
        id: ContextId,
        num_input_tokens: u32,
        response: oneshot::Sender<Result<PinnedContext>>,
    ) {
        let ctx = match self.contexts.get(&id) {
            Some(c) => c,
            None => { let _ = response.send(Err(anyhow::anyhow!("Context not found"))); return; }
        };

        let pid = match ctx.owner {
            Some(pid) => pid,
            None => { let _ = response.send(Err(anyhow::anyhow!("Context has no owning process"))); return; }
        };

        self.with_gpu(pid, Some(id), move |mgr| {
            let result = (|| -> Result<PinnedContext> {
                let ctx = mgr.contexts.get_mut(&id)
                    .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
                if ctx.is_suspended() {
                    anyhow::bail!("pin: context is suspended (cannot pin)");
                }
                let dev_idx = ctx.device.unwrap_or(0) as usize;
                let committed_hashes = ctx.committed_hashes.clone();
                let working = ctx.working_pages.clone();
                let kv_len = (ctx.committed_len() * mgr.page_size
                    + ctx.working_page_tokens.len()) as u32;
                ctx.state = ContextState::Pinned;

                let mut page_ids = Vec::new();
                if !committed_hashes.is_empty() {
                    page_ids.extend(mgr.devices[dev_idx].physical_ids(&committed_hashes));
                }
                page_ids.extend(&working);

                let num_pages = page_ids.len() as u32;
                let page_size = mgr.page_size as u32;
                let total_kv = kv_len + num_input_tokens;
                let last_page_len = pagestore::compute_last_page_len(
                    total_kv, num_pages, page_size,
                );
                Ok(PinnedContext {
                    device: dev_idx as DeviceId, pages: page_ids,
                    kv_len, last_page_len,
                })
            })();
            let _ = response.send(result);
        });
    }

    /// Handle ClearPinned message: Pinned → Active, then check deferred suspension.
    /// If `pending_suspend` was set, executes the deferred suspension and
    /// decrements `ProcessEntry.pending_pinned`.
    pub(crate) fn unpin(&mut self, id: ContextId) {
        let pending = match self.contexts.get(&id) {
            Some(ctx) if ctx.is_pinned() => ctx.pending_suspend,
            _ => return,
        };

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
    /// Phase 1: alloc_queue (FIFO) — invoke deferred GPU operation callbacks.
    /// Phase 2: restore_queue (priority heap) — restore highest-priority Pending
    ///          process, then replay its deferred_op.
    pub(crate) fn drain_queues(&mut self) {
        // Phase 1: alloc_queue FIFO (head-of-line blocking)
        while let Some(front) = self.alloc_queue.front() {
            let (dev_idx, n) = (front.device, front.num_pages);
            if self.devices[dev_idx].available_gpu_pages() < n {
                break;
            }
            let waiter = self.alloc_queue.pop_front().unwrap();
            let pages = self.devices[dev_idx].alloc_gpu_pages(n).unwrap();
            (waiter.on_alloc)(self, pages);
        }

        // Phase 2: restore_queue (priority heap)
        // Only proceed if alloc_queue is empty (allocs have strict priority).
        if !self.alloc_queue.is_empty() { return; }

        while let Some(top) = self.restore_queue.peek() {
            let pid = top.process_id;

            // Block if still has Pinned contexts clearing.
            if self.processes.get(&pid).map(|p| p.pending_pinned).unwrap_or(0) > 0 {
                break;
            }

            // Admission check: enough pages on all devices?
            if !self.can_restore_all(pid) {
                break;
            }

            let waiter = self.restore_queue.pop().unwrap();
            if let Err(e) = self.restore_all(waiter.process_id) {
                eprintln!("RESTORE_ALL_FAIL pid={} err={e:#}", waiter.process_id);
            }
        }
    }

    // ==================== Extracted handle() helpers ====================


    pub(crate) fn stats(&self) -> Vec<(usize, usize)> {
        self.devices.iter().map(|d| d.stats()).collect()
    }

    pub(crate) fn set_priority(&mut self, weight: f64, pid_values: HashMap<ProcessId, f64>) {
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
            .map(|c| c.committed_len() as u32)
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
        masks: Vec<Brle>, adapter: Option<AdapterId>, adapter_seed: Option<i64>,
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
                adapter_seed,
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
    LookupSnapshot { username: String, name: String, response: oneshot::Sender<Result<ContextId>> },
    Create { owner: ProcessId, response: oneshot::Sender<Result<ContextId>> },
    Save { id: ContextId, username: String, name: Option<String>, response: oneshot::Sender<Result<Option<String>>> },
    Delete { username: String, name: String, response: oneshot::Sender<Result<()>> },
    Destroy { id: ContextId, response: oneshot::Sender<Result<()>> },
    Fork { id: ContextId, owner: ProcessId, response: oneshot::Sender<Result<ContextId>> },
    Take { username: String, name: String, owner: ProcessId, response: oneshot::Sender<Result<ContextId>> },
    CommitWorkingPages { id: ContextId, num_pages: usize, response: oneshot::Sender<Result<()>> },
    ReserveWorkingPages { id: ContextId, num_pages: usize, response: oneshot::Sender<Result<()>> },
    ReleaseWorkingPages { id: ContextId, num_pages: usize },
    Pin { id: ContextId, num_input_tokens: u32, response: oneshot::Sender<Result<PinnedContext>> },
    Unpin { id: ContextId },
    ReplayComplete { id: ContextId },
    GetStats { response: oneshot::Sender<Vec<(usize, usize)>> },

    // Actor-routed read/write APIs (previously DashMap-based)
    WorkingPageCount { id: ContextId, response: oneshot::Sender<u32> },
    CommittedPageCount { id: ContextId, response: oneshot::Sender<u32> },
    WorkingPageTokenCount { id: ContextId, response: oneshot::Sender<u32> },
    TruncateWorkingPageTokens { id: ContextId, count: u32, response: oneshot::Sender<Result<()>> },
    AppendWorkingPageTokens {
        id: ContextId, tokens: Vec<u32>, positions: Vec<u32>,
        masks: Vec<Brle>, adapter: Option<AdapterId>, adapter_seed: Option<i64>,
        response: oneshot::Sender<Result<()>>,
    },

    DebugState { id: ContextId, response: oneshot::Sender<String> },
    SetPriority { weight: f64, pid_values: HashMap<ProcessId, f64> },
    DestroyAll { pid: ProcessId },
}

impl ServiceHandler for ContextManager {
    type Message = Message;

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::LookupSnapshot { username, name, response } => {
                let result = self.snapshots.get(&(username, name))
                    .copied()
                    .ok_or_else(|| anyhow::anyhow!("Snapshot not found"));
                let _ = response.send(result);
            }
            Message::Take { username, name, owner, response } => {
                self.take(username, name, owner, response);
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
            Message::Destroy { id, response } => {
                let _ = response.send(self.destroy(id));
            }
            Message::Fork { id, owner, response } => {
                self.fork(id, owner, response);
            }
            Message::CommitWorkingPages { id, num_pages, response } => {
                let _ = response.send(self.commit_working_pages(id, num_pages));
            }
            Message::ReserveWorkingPages { id, num_pages, response } => {
                self.reserve_working_pages(id, num_pages, response);
            }
            Message::ReleaseWorkingPages { id, num_pages } => {
                let _ = self.release_working_pages(id, num_pages);
            }
            Message::Pin { id, num_input_tokens, response } => {
                self.pin(id, num_input_tokens, response);
            }
            Message::Unpin { id } => {
                self.unpin(id);
            }
            Message::ReplayComplete { id } => {
                self.replay_complete(id);
            }
            Message::GetStats { response } => {
                let _ = response.send(self.stats());
            }
            Message::SetPriority { weight, pid_values } => {
                self.set_priority(weight, pid_values);
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
            Message::AppendWorkingPageTokens { id, tokens, positions, masks, adapter, adapter_seed, response } => {
                let _ = response.send(self.append_working_page_tokens(id, tokens, positions, masks, adapter, adapter_seed));
            }
            Message::DebugState { id, response } => {
                let _ = response.send(self.debug_state(id));
            }
            Message::DestroyAll { pid } => {
                self.destroy_all(pid);
            }
        }
    }
}
