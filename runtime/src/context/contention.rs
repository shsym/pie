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


use crate::device;
use crate::process::ProcessId;

use super::{
    ContextId, ContextManager, ContextState, ProcessState,
    ReplayFill,
};
use super::pagestore::PhysicalPageId;

// =============================================================================
// Queue Items
// =============================================================================

/// A deferred page allocation request (try_alloc queue).
#[derive(Debug)]
pub(crate) struct AllocWaiter {
    pub context_id: ContextId,
    pub device: usize,
    pub num_pages: usize,
    pub response: oneshot::Sender<anyhow::Result<()>>,
}

/// A deferred process restoration request (try_restore queue).
/// Mutable per-process state (pending_allocs, pending_pinned_count) is stored
/// in side-maps on ContextManager to avoid BinaryHeap interior mutation.
#[derive(Debug)]
pub(crate) struct RestoreWaiter {
    pub process_id: ProcessId,
    pub priority_floor: f64,
    pub enqueued_at: Instant,
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

    /// Handle a ReservePages message per DESIGN.md §4 (steps 0–6).
    /// Returns replay chunks from drain_queues (if eviction triggers restoration).
    pub(crate) fn handle_reserve_pages(
        &mut self,
        id: ContextId,
        num_pages: u32,
        response: oneshot::Sender<anyhow::Result<()>>,
    ) -> Vec<ReplayFill> {
        if num_pages == 0 {
            let _ = response.send(Ok(()));
            return Vec::new();
        }

        let ctx = match self.contexts.get(&id) {
            Some(c) => c,
            None => { let _ = response.send(Err(anyhow::anyhow!("Context not found"))); return Vec::new(); }
        };
        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let owner = ctx.owner;
        let current_working = ctx.working_pages.len();

        let additional = (num_pages as usize).saturating_sub(current_working);
        if additional == 0 {
            let _ = response.send(Ok(()));
            return Vec::new();
        }

        // Step 0: SUSPENSION CHECK
        // If the owning process is Pending, attach alloc to its pending_allocs.
        if let Some(pid) = owner {
            if self.processes.get(&pid).map(|p| p.state == ProcessState::Pending).unwrap_or(false) {
                self.pending_allocs_map.entry(pid).or_default().push(AllocWaiter {
                    context_id: id, device: dev_idx,
                    num_pages: additional, response,
                });
                return Vec::new();
            }
        }

        // Step 1: FIFO GATE — if try_alloc has pending requests, enqueue behind them.
        if !self.try_alloc.is_empty() {
            self.try_alloc.push_back(AllocWaiter {
                context_id: id, device: dev_idx,
                num_pages: additional, response,
            });
            return Vec::new();
        }

        // Step 2: PRIORITY GATE — compare requester floor vs try_restore head.
        let requester_floor = owner
            .map(|pid| self.arbiter.priority_at(&pid, self.arbiter.pages_on(&pid, dev_idx) + additional))
            .unwrap_or(0.0);

        if let Some(pid) = owner {
            if let Some(top) = self.try_restore.peek() {
                let top_pid = top.process_id;
                let top_ready = self.pending_pinned_counts.get(&top_pid).copied().unwrap_or(0) == 0;
                if top_ready && requester_floor < top.effective_priority() {
                    // Requester loses priority gate → suspend and enqueue in try_restore.
                    let (pinned_count, _) = self.suspend_process(pid);
                    self.enqueue_restore(pid, requester_floor, pinned_count);
                    self.pending_allocs_map.entry(pid).or_default().push(AllocWaiter {
                        context_id: id, device: dev_idx,
                        num_pages: additional, response,
                    });
                    return Vec::new();
                }
            }
        }

        // Step 3: TRY ALLOCATE from free pool.
        if let Some(pages) = self.devices[dev_idx].alloc_working(additional) {
            self.apply_alloc(id, dev_idx, pages, owner);
            let _ = response.send(Ok(()));
            return Vec::new();
        }

        // Step 4: EVICT unreferenced committed pages (rc=0) and retry.
        self.devices[dev_idx].evict_unreferenced();
        if let Some(pages) = self.devices[dev_idx].alloc_working(additional) {
            self.apply_alloc(id, dev_idx, pages, owner);
            let _ = response.send(Ok(()));
            return Vec::new();
        }

        // Step 5: EVICTION LOOP
        let mut all_pinned_break = false;
        loop {
            match self.find_eviction_victim(dev_idx, requester_floor, owner) {
                Some(victim_pid) => {
                    let victim_floor = self.arbiter.priority(&victim_pid, dev_idx);
                    let (pinned_count, active_count) = self.suspend_process(victim_pid);
                    self.enqueue_restore(victim_pid, victim_floor, pinned_count);

                    // Step 5c: If ALL of victim's contexts were Pinned (none
                    // Active), no pages freed immediately. Break → try_alloc.
                    if active_count == 0 && pinned_count > 0 {
                        all_pinned_break = true;
                        break;
                    }

                    // Step 5d: Retry alloc after eviction freed pages.
                    if let Some(pages) = self.devices[dev_idx].alloc_working(additional) {
                        self.apply_alloc(id, dev_idx, pages, owner);
                        let _ = response.send(Ok(()));
                        return self.drain_queues();
                    }

                    // If any Pinned contexts, stop looping — deferred pages
                    // won't free until clear_pinned. Fall through to step 6.
                    if pinned_count > 0 {
                        break;
                    }
                    // No Pinned: loop to find another victim.
                }
                None => break,
            }
        }

        if all_pinned_break {
            // Step 5c target: enqueue in try_alloc (requester stays Running).
            self.try_alloc.push_back(AllocWaiter {
                context_id: id, device: dev_idx,
                num_pages: additional, response,
            });
        } else if let Some(pid) = owner {
            // Step 6: NO VICTIM / PINNED VICTIM — requester self-suspends.
            let (pinned_count, _) = self.suspend_process(pid);
            self.enqueue_restore(pid, requester_floor, pinned_count);
            self.pending_allocs_map.entry(pid).or_default().push(AllocWaiter {
                context_id: id, device: dev_idx,
                num_pages: additional, response,
            });
        } else {
            // Owner-less context: enqueue in try_alloc FIFO.
            self.try_alloc.push_back(AllocWaiter {
                context_id: id, device: dev_idx,
                num_pages: additional, response,
            });
        }
        Vec::new()
    }

    /// Apply an allocation to a context (after successful alloc_working).
    pub(crate) fn apply_alloc(
        &mut self,
        id: ContextId,
        dev_idx: usize,
        pages: Vec<PhysicalPageId>,
        owner: Option<ProcessId>,
    ) {
        let n = pages.len();
        if let Some(ctx) = self.contexts.get_mut(&id) {
            ctx.working_pages.extend(pages);
            ctx.device = Some(dev_idx);
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

    /// Helper: enqueue a RestoreWaiter for a suspended process.
    fn enqueue_restore(&mut self, pid: ProcessId, priority_floor: f64, pinned_count: usize) {
        if pinned_count > 0 {
            *self.pending_pinned_counts.entry(pid).or_insert(0) += pinned_count;
        }
        self.try_restore.push(RestoreWaiter {
            process_id: pid,
            priority_floor,
            enqueued_at: Instant::now(),
        });
    }

    /// Suspend all contexts of a process across ALL devices (cooperative suspension).
    /// - Active contexts: immediately suspend (working→CPU, chain released)
    /// - Pinned contexts: set `pending_suspend` flag (deferred)
    ///
    /// Always marks the process as Pending. Returns `(pinned_count, active_count)`:
    /// - `pinned_count`: Pinned contexts deferred (for `pending_pinned_counts`)
    /// - `active_count`: Active contexts immediately suspended (pages freed now)
    pub(crate) fn suspend_process(&mut self, pid: ProcessId) -> (usize, usize) {
        let ctx_ids: Vec<ContextId> = self.processes.get(&pid)
            .map(|p| p.context_ids.clone())
            .unwrap_or_default();

        let mut pinned_count: usize = 0;
        let mut active_count: usize = 0;
        let mut devices_touched: Vec<usize> = Vec::new();

        for &ctx_id in &ctx_ids {
            let (state, dev) = match self.contexts.get(&ctx_id) {
                Some(ctx) => (ctx.state, ctx.device.unwrap_or(0) as usize),
                None => continue,
            };

            if !devices_touched.contains(&dev) {
                devices_touched.push(dev);
            }

            match state {
                ContextState::Active => {
                    self.suspend_context(ctx_id);
                    active_count += 1;
                }
                ContextState::Pinned => {
                    // Deferred suspension
                    if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
                        ctx.pending_suspend = true;
                    }
                    pinned_count += 1;
                }
                ContextState::Suspended => {
                    // Already suspended, nothing to do
                }
            }
        }

        // Always transition to Pending — the process is conceptually suspended
        // even if some contexts have deferred suspension.
        if let Some(proc) = self.processes.get_mut(&pid) {
            proc.state = ProcessState::Pending;
        }

        // Zero out arbiter accounting for ALL devices the process has contexts on
        for &dev in &devices_touched {
            self.arbiter.zero_device(pid, dev);
        }

        (pinned_count, active_count)
    }

    /// Suspend a single Active context: swap working pages GPU→CPU, release chain.
    pub(crate) fn suspend_context(&mut self, ctx_id: ContextId) {
        let (dev_idx, working, tip) = match self.contexts.get(&ctx_id) {
            Some(ctx) if ctx.state == ContextState::Active || ctx.state == ContextState::Pinned => {
                (ctx.device.unwrap_or(0) as usize, ctx.working_pages.clone(), ctx.committed_tip)
            }
            _ => return,
        };

        // Phase 1: Swap working pages to CPU
        if !working.is_empty() {
            let dev = &mut self.devices[dev_idx];
            match dev.alloc_cpu_pages(working.len()) {
                Some(cpu_pages) => {
                    // Copy GPU → CPU, then free GPU pages
                    let _ = device::copy_d2h(dev_idx, &working, &cpu_pages);
                    self.devices[dev_idx].free_working(&working);

                    if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
                        ctx.working_pages.clear();
                        ctx.working_pages_cpu = cpu_pages;
                    }
                }
                None => {
                    eprintln!("SUSPEND_SWAP_FAIL ctx={ctx_id} err=No free CPU pages");
                    // Continue with suspension anyway — lose working pages
                    let pages_to_free = if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
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
        if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
            ctx.state = ContextState::Suspended;
            ctx.pending_suspend = false;
        }

        // NOTE: evict_unreferenced is NOT called here per DESIGN.md §4:
        // "it does not clean up unreferenced (rc=0) pages. It's the role of reserve_pages."
        // This ensures separation: suspend_context only releases refcounts,
        // reserve_pages (step 4) handles the actual eviction.
    }

    // ==================== clear_pinned ====================

    /// Handle ClearPinned message: Pinned → Active, then check deferred suspension.
    /// If `pending_suspend` was set, executes the deferred suspension and
    /// decrements the `pending_pinned_counts` side-map.
    /// Returns replay chunks from drain_queues (if deferred suspension freed pages).
    pub(crate) fn handle_clear_pinned(&mut self, id: ContextId) -> Vec<ReplayFill> {
        let (is_pinned, pending) = match self.contexts.get(&id) {
            Some(ctx) if ctx.state == ContextState::Pinned => (true, ctx.pending_suspend),
            _ => return Vec::new(),
        };

        if !is_pinned { return Vec::new(); }

        if pending {
            // Deferred suspension: context stays Pinned until suspend_context
            // transitions it to Suspended.
            let owner = self.contexts.get(&id).and_then(|c| c.owner);
            self.suspend_context(id);

            // Decrement pending_pinned_count for the owning process.
            if let Some(pid) = owner {
                if let Some(count) = self.pending_pinned_counts.get_mut(&pid) {
                    *count = count.saturating_sub(1);
                    if *count == 0 {
                        self.pending_pinned_counts.remove(&pid);
                    }
                }
            }

            // Deferred suspension may have freed pages — drain queues.
            return self.drain_queues();
        }

        // No deferred suspension — normal Pinned → Active transition.
        if let Some(ctx) = self.contexts.get_mut(&id) {
            ctx.state = ContextState::Active;
        }

        Vec::new()
    }

    // ==================== drain_queues ====================

    /// Central queue drain: called after any event that frees GPU pages.
    ///
    /// Phase 1: try_alloc (FIFO) — serve front-of-queue allocations.
    /// Phase 2: try_restore (priority heap) — restore highest-priority Pending
    ///          process, then replay its pending_allocs.
    pub(crate) fn drain_queues(&mut self) -> Vec<ReplayFill> {
        // Phase 1: try_alloc FIFO (head-of-line blocking)
        while let Some(front) = self.try_alloc.front() {
            let dev_idx = front.device as usize;
            let n = front.num_pages;

            // Evict unreferenced committed pages (rc=0) before attempting alloc
            self.devices[dev_idx].evict_unreferenced();
            if let Some(pages) = self.devices[dev_idx].alloc_working(n) {
                let waiter = self.try_alloc.pop_front().unwrap();
                let owner = self.contexts.get(&waiter.context_id).and_then(|c| c.owner);
                self.apply_alloc(waiter.context_id, dev_idx, pages, owner);
                let _ = waiter.response.send(Ok(()));
            } else {
                break; // Not enough pages for front of queue
            }
        }

        let mut all_replay_chunks = Vec::new();

        // Phase 2: try_restore (priority heap)
        // Only proceed if try_alloc is empty (allocs have strict priority).
        if !self.try_alloc.is_empty() { return all_replay_chunks; }

        while let Some(top) = self.try_restore.peek() {
            let pid = top.process_id;

            // Block if still has Pinned contexts clearing.
            if self.pending_pinned_counts.get(&pid).copied().unwrap_or(0) > 0 {
                break;
            }

            // Admission check: enough pages on all devices?
            if !self.can_restore_process(pid) {
                break;
            }

            let waiter = self.try_restore.pop().unwrap();
            let chunks = self.restore_and_replay(waiter.process_id);
            all_replay_chunks.extend(chunks);
        }

        all_replay_chunks
    }
}

