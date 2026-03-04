//! Contention — Dual-Queue Contention Resolution Protocol.
//!
//! Implements the two-queue model from DESIGN.md:
//! - `try_alloc` (FIFO): deferred allocation requests from Running processes
//! - `try_restore` (Priority Heap): Pending processes awaiting full restoration
//!
//! ## Core Flows
//!
//! `reserve_pages` → attempt alloc → eviction loop → self-suspend to try_alloc
//! `drain_queues`  → Phase 1 (try_alloc FIFO) → Phase 2 (try_restore heap)
//! `clear_pinned`  → check pending_suspend flag → execute deferred suspension

use std::cmp::Ordering;
use std::time::Instant;
use tokio::sync::oneshot;

use crate::device::DeviceId;
use crate::process::ProcessId;

use super::{
    ContextId, ContextManager, ContextState, ProcessState,
    CONTEXTS, PAGE_SIZES, ResidentResult,
};
use super::pagestore::PhysicalPageId;

// =============================================================================
// Queue Items
// =============================================================================

/// A deferred page allocation request (try_alloc queue).
#[derive(Debug)]
pub(crate) struct AllocWaiter {
    pub context_id: ContextId,
    pub device: DeviceId,
    pub num_pages: usize,
    pub response: oneshot::Sender<anyhow::Result<()>>,
}

/// A deferred process restoration request (try_restore queue).
#[derive(Debug)]
pub(crate) struct RestoreWaiter {
    pub process_id: ProcessId,
    pub priority_floor: f64,
    pub enqueued_at: Instant,
    pub response: oneshot::Sender<anyhow::Result<ResidentResult>>,
    pub context_id: ContextId,
}

const AGING_RATE: f64 = 0.01;

impl RestoreWaiter {
    fn effective_priority(&self) -> f64 {
        let wait_secs = self.enqueued_at.elapsed().as_secs_f64();
        self.priority_floor + AGING_RATE * wait_secs
    }
}

impl PartialEq for RestoreWaiter {
    fn eq(&self, other: &Self) -> bool {
        self.process_id == other.process_id
    }
}
impl Eq for RestoreWaiter {}

impl PartialOrd for RestoreWaiter {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RestoreWaiter {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap — highest effective priority pops first.
        self.effective_priority()
            .partial_cmp(&other.effective_priority())
            .unwrap_or(Ordering::Equal)
    }
}

// =============================================================================
// Contention methods on ContextManager
// =============================================================================

impl ContextManager {

    // ==================== reserve_pages flow ====================

    /// Handle a ReservePages message per DESIGN.md §4:
    /// 1. FIFO gate: if try_alloc non-empty, enqueue
    /// 2. Try direct alloc from free pool
    /// 3. Eviction loop: find cheapest victim, suspend, retry
    /// 4. If no victim, self-suspend to try_alloc queue
    pub(crate) fn handle_reserve_pages(
        &mut self,
        id: ContextId,
        num_pages: u32,
        response: oneshot::Sender<anyhow::Result<()>>,
    ) {
        if num_pages == 0 {
            let _ = response.send(Ok(()));
            return;
        }

        let ctx = match self.ctx(id) {
            Ok(c) => c,
            Err(e) => { let _ = response.send(Err(e)); return; }
        };
        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let owner = ctx.owner;
        let current_working = ctx.working_pages.len();
        drop(ctx);

        let additional = (num_pages as usize).saturating_sub(current_working);
        if additional == 0 {
            let _ = response.send(Ok(()));
            return;
        }

        // Step 1: FIFO gate — if try_alloc has pending requests, queue behind them
        if !self.try_alloc.is_empty() {
            self.try_alloc.push_back(AllocWaiter {
                context_id: id, device: dev_idx as DeviceId,
                num_pages: additional, response,
            });
            return;
        }

        // Step 2: Try direct allocation from free pool
        if let Some(pages) = self.devices[dev_idx].alloc_working(additional) {
            self.apply_alloc(id, dev_idx, pages, owner);
            let _ = response.send(Ok(()));
            return;
        }

        // Step 3: Eviction loop
        let requester_floor = owner
            .map(|pid| self.arbiter.priority_at(&pid, self.arbiter.pages_on(&pid, dev_idx) + additional))
            .unwrap_or(0.0);

        loop {
            match self.find_eviction_victim(dev_idx, requester_floor, owner) {
                Some(victim_pid) => {
                    self.suspend_process(victim_pid, dev_idx);

                    if let Some(pages) = self.devices[dev_idx].alloc_working(additional) {
                        self.apply_alloc(id, dev_idx, pages, owner);
                        let _ = response.send(Ok(()));
                        self.drain_queues();
                        return;
                    }
                }
                None => break,
            }
        }

        // Step 4: No victim found — enqueue in try_alloc
        self.try_alloc.push_back(AllocWaiter {
            context_id: id, device: dev_idx as DeviceId,
            num_pages: additional, response,
        });
    }

    /// Apply an allocation to a context (after successful alloc_working).
    fn apply_alloc(
        &mut self,
        id: ContextId,
        dev_idx: usize,
        pages: Vec<PhysicalPageId>,
        owner: Option<ProcessId>,
    ) {
        let n = pages.len();
        if let Ok(mut ctx) = self.ctx_mut(id) {
            ctx.working_pages.extend(pages);
            ctx.device = Some(dev_idx as DeviceId);
        }
        if let Some(pid) = owner {
            self.arbiter.add_working(pid, dev_idx, n);
        }
    }

    // ==================== Eviction ====================

    /// Find the cheapest eviction victim on a device.
    /// Returns the ProcessId of the victim, or None if no evictable process
    /// has lower priority than `requester_floor`.
    fn find_eviction_victim(
        &self,
        dev_idx: usize,
        requester_floor: f64,
        requester: Option<ProcessId>,
    ) -> Option<ProcessId> {
        let mut cheapest: Option<(ProcessId, f64, Instant)> = None;

        for (&pid, proc) in &self.processes {
            // Don't evict the requester
            if requester == Some(pid) { continue; }
            // Only evict Running processes (Pending already suspended)
            if proc.state != ProcessState::Running { continue; }
            // Must have pages on this device
            let pages = self.arbiter.pages_on(&pid, dev_idx);
            if pages == 0 { continue; }

            let priority = self.arbiter.priority(&pid, dev_idx);

            // Anti-thrashing: only evict if victim's invested importance
            // is strictly less than requester's post-allocation floor.
            if priority >= requester_floor { continue; }

            let created = self.arbiter.created_at(&pid).unwrap_or_else(Instant::now);
            match cheapest {
                None => cheapest = Some((pid, priority, created)),
                Some((_, best_pri, best_time)) => {
                    // Lower priority first; FCFS tiebreak (older = evicted first)
                    if priority < best_pri || (priority == best_pri && created < best_time) {
                        cheapest = Some((pid, priority, created));
                    }
                }
            }
        }

        cheapest.map(|(pid, _, _)| pid)
    }

    /// Suspend all contexts of a process on a device (cooperative suspension).
    /// - Active contexts: immediately suspend
    /// - Pinned contexts: set `pending_suspend` flag (deferred)
    pub(crate) fn suspend_process(&mut self, pid: ProcessId, dev_idx: usize) {
        let ctx_ids: Vec<ContextId> = self.processes.get(&pid)
            .map(|p| p.context_ids.clone())
            .unwrap_or_default();

        let mut all_suspended = true;

        for ctx_id in &ctx_ids {
            if let Some(ctx) = CONTEXTS.get(&(self.model_idx, *ctx_id)) {
                let dev = ctx.device.unwrap_or(0) as usize;
                if dev != dev_idx { continue; }

                match ctx.state {
                    ContextState::Active => {
                        drop(ctx);
                        self.suspend_context(*ctx_id);
                    }
                    ContextState::Pinned => {
                        drop(ctx);
                        // Deferred suspension
                        if let Some(mut ctx) = CONTEXTS.get_mut(&(self.model_idx, *ctx_id)) {
                            ctx.pending_suspend = true;
                        }
                        all_suspended = false;
                    }
                    ContextState::Suspended => {
                        // Already suspended, nothing to do
                    }
                }
            }
        }

        // Transition process to Pending if all contexts are suspended
        if all_suspended {
            if let Some(proc) = self.processes.get_mut(&pid) {
                proc.state = ProcessState::Pending;
            }
        }

        // Zero out arbiter accounting for this device
        self.arbiter.zero_device(pid, dev_idx);
    }

    /// Suspend a single Active context: swap working pages GPU→CPU, release chain.
    pub(crate) fn suspend_context(&mut self, ctx_id: ContextId) {
        let ctx = match self.ctx(ctx_id) {
            Ok(c) => c,
            Err(_) => return,
        };
        if ctx.state != ContextState::Active { return; }

        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let working = ctx.working_pages.clone();
        let tip = ctx.committed_tip;
        drop(ctx);

        // Phase 1: Swap working pages to CPU
        if !working.is_empty() {
            let dev = &mut self.devices[dev_idx];
            match dev.swap_out(&working) {
                Ok(swap_ops) => {
                    let cpu_slots: Vec<PhysicalPageId> = swap_ops.iter().map(|op| op.cpu_slot).collect();
                    if let Ok(mut ctx) = self.ctx_mut(ctx_id) {
                        ctx.working_pages.clear();
                        ctx.working_cpu_slots = cpu_slots;
                    }
                }
                Err(e) => {
                    eprintln!("SUSPEND_SWAP_FAIL ctx={ctx_id} err={e}");
                    // Continue with suspension anyway — lose working pages
                    // Extract working pages before borrowing devices
                    let pages_to_free: Vec<PhysicalPageId> = if let Ok(mut ctx) = self.ctx_mut(ctx_id) {
                        let pages = ctx.working_pages.clone();
                        ctx.working_pages.clear();
                        pages
                    } else {
                        Vec::new()
                    };
                    self.devices[dev_idx].free_working(&pages_to_free);
                }
            }
        }

        // Phase 2: Release committed chain refcounts
        if let Some(tip_hash) = tip {
            if let Some(dev) = self.devices.get_mut(dev_idx) {
                dev.release_chain(tip_hash);
                dev.remove_index_cache(tip_hash);
            }
        }

        // Phase 3: Mark suspended
        if let Ok(mut ctx) = self.ctx_mut(ctx_id) {
            ctx.state = ContextState::Suspended;
            ctx.pending_suspend = false;
        }

        // Evict unreferenced committed pages freed by the chain release
        self.devices[dev_idx].evict_unreferenced();
    }

    // ==================== clear_pinned ====================

    /// Handle ClearPinned message: Pinned → Active, then check deferred suspension.
    pub(crate) fn handle_clear_pinned(&mut self, id: ContextId) {
        let pending = if let Some(mut ctx) = CONTEXTS.get_mut(&(self.model_idx, id)) {
            if ctx.state == ContextState::Pinned {
                let pending = ctx.pending_suspend;
                if pending {
                    // Deferred suspension: Pinned → Active → Suspended
                    ctx.state = ContextState::Active;
                    ctx.pending_suspend = false;
                } else {
                    ctx.state = ContextState::Active;
                }
                pending
            } else {
                false
            }
        } else {
            return;
        };

        if pending {
            self.suspend_context(id);

            // Check if the owning process should transition to Pending
            let owner = CONTEXTS.get(&(self.model_idx, id)).and_then(|c| c.owner);
            if let Some(pid) = owner {
                let all_suspended = self.processes.get(&pid)
                    .map(|p| p.context_ids.iter().all(|&cid| {
                        CONTEXTS.get(&(self.model_idx, cid))
                            .map(|c| c.state == ContextState::Suspended)
                            .unwrap_or(true)
                    }))
                    .unwrap_or(true);

                if all_suspended {
                    if let Some(proc) = self.processes.get_mut(&pid) {
                        proc.state = ProcessState::Pending;
                    }
                }
            }
        }

        self.drain_queues();
    }

    // ==================== drain_queues ====================

    /// Central queue drain: called after any event that frees GPU pages.
    ///
    /// Phase 1: try_alloc (FIFO) — serve front-of-queue allocations.
    /// Phase 2: try_restore (priority heap) — restore highest-priority Pending process.
    pub(crate) fn drain_queues(&mut self) {
        // Phase 1: try_alloc FIFO
        while let Some(front) = self.try_alloc.front() {
            let dev_idx = front.device as usize;
            let n = front.num_pages;

            if let Some(pages) = self.devices[dev_idx].alloc_working(n) {
                let waiter = self.try_alloc.pop_front().unwrap();
                self.apply_alloc(waiter.context_id, dev_idx, pages, None);
                let _ = waiter.response.send(Ok(()));
            } else {
                break; // Not enough pages for front of queue
            }
        }

        // Phase 2: try_restore (priority heap)
        // Only proceed if try_alloc is empty (allocs have strict priority)
        if !self.try_alloc.is_empty() { return; }

        if let Some(top) = self.try_restore.peek() {
            let pid = top.process_id;
            let ctx_id = top.context_id;

            // Try to restore the process
            match self.try_restore_process(pid, ctx_id) {
                Ok(result) => {
                    let waiter = self.try_restore.pop().unwrap();
                    let _ = waiter.response.send(Ok(result));
                }
                Err(_) => {
                    // Not enough pages to restore — leave in queue
                }
            }
        }
    }

    // ==================== handle_ensure_resident ====================

    /// Handle EnsureResident: check if context is resident, restore if needed.
    pub(crate) fn handle_ensure_resident(&mut self, id: ContextId) -> anyhow::Result<ResidentResult> {
        let ctx = self.ctx(id)?;
        let state = ctx.state;
        let owner = ctx.owner;
        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let has_cpu_working = !ctx.working_cpu_slots.is_empty();
        drop(ctx);

        match state {
            ContextState::Active | ContextState::Pinned => {
                // Already resident — resolve pages and pin
                let pages = self.get_physical_page_ids_impl(id)?;
                let phys_len: usize = pages.values().map(|v| v.len()).sum();

                let (kv_len, debug_state) = {
                    let page_size = PAGE_SIZES.get(self.model_idx).copied().unwrap_or(0);
                    CONTEXTS.get(&(self.model_idx, id))
                        .map(|ctx| {
                            let kv = (ctx.committed_len * page_size + ctx.tokens_filled.len()) as u32;
                            let state = format!(
                                "committed_len={} tokens_filled={} working_pages={} working_cpu={} state={:?} phys_len={}",
                                ctx.committed_len, ctx.tokens_filled.len(),
                                ctx.working_pages.len(), ctx.working_cpu_slots.len(),
                                ctx.state, phys_len,
                            );
                            (kv, state)
                        })
                        .unwrap_or((0, "MISSING".to_string()))
                };

                // Pin as non-evictable
                if let Some(mut ctx) = CONTEXTS.get_mut(&(self.model_idx, id)) {
                    ctx.state = ContextState::Pinned;
                }

                Ok(ResidentResult {
                    replay_chunks: None,
                    pages,
                    kv_len,
                    debug_state,
                })
            }
            ContextState::Suspended => {
                // Need restoration — enqueue in try_restore or attempt immediately
                if let Some(pid) = owner {
                    let floor = self.arbiter.priority_at(&pid, self.arbiter.pages_on(&pid, dev_idx) + 1);

                    match self.try_restore_process(pid, id) {
                        Ok(result) => Ok(result),
                        Err(_) => {
                            // Not enough pages — this will be communicated back to caller
                            anyhow::bail!("Insufficient pages for restoration")
                        }
                    }
                } else {
                    // Ownerless (snapshot) — try direct restore
                    self.restore_context(id)?;
                    let pages = self.get_physical_page_ids_impl(id)?;
                    Ok(ResidentResult {
                        replay_chunks: None,
                        pages,
                        kv_len: 0,
                        debug_state: "restored_snapshot".to_string(),
                    })
                }
            }
        }
    }
}
