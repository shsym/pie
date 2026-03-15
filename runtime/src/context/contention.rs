//! Contention — Dual-Queue GPU Page Resolution Protocol.
//!
//! When GPU pages are scarce, this module decides who waits, who yields,
//! and in what order pages are reclaimed.
//!
//! ## Two Queues
//!
//! - **`alloc_queue`** (FIFO): deferred GPU page requests from Running
//!   processes that won the priority gate but found no free pages. Head-of-line
//!   blocking — served in arrival order as pages free up.
//!
//! - **`restore_queue`** (Priority Heap): Pending processes awaiting full
//!   restoration. Ordered by `effective_priority = floor + AGING_RATE × wait_s`.
//!   AGING_RATE = 0.01 provides starvation freedom.
//!
//! ## `with_gpu_pages` — Full Contention Flow
//!
//! ```text
//! 0. SUSPENSION CHECK: process Pending → stash as deferred_op, return
//! 1. FIFO GATE: alloc_queue non-empty → enqueue at tail, return
//! 2. PRIORITY GATE: requester_floor < restore_queue.peek() → self-suspend
//! 3. TRY ALLOCATE from free pool → success: invoke callback
//! 4. EVICTION LOOP:
//!    a. find_eviction_victim (lowest π, requester_floor check)
//!    b. suspend_process(victim) → Active: immediate, Pinned: deferred
//!    c. track deferred_pages from Pinned contexts
//!    d. retry alloc → success: done
//!    e. free + deferred ≥ needed → enqueue in alloc_queue
//! 5. NO VICTIM → requester self-suspends
//! ```
//!
//! ## `drain_queues` — Called on Every Page-Free Event
//!
//! Phase 1 (alloc_queue FIFO): serve head if pages available, else stop.
//! Phase 2 (restore_queue heap): if alloc_queue empty, pop highest-priority
//! Pending process if all devices have enough pages, then `restore_all`.
//!
//! Triggers: `unpin` with `pending_suspend`, context destruction, commit dedup.
//!
//! ## `suspend_context` — Three Phases
//!
//! 1. **Swap working pages** GPU → CPU (D2H copy, free GPU slots).
//! 2. **Release committed chain** refcounts (rc→0 pages become evictable).
//! 3. **Mark Suspended** (`pending_suspend = false`).
//!
//! ## Anti-Thrashing Guarantee
//!
//! After process A evicts process B:
//! - `π_A = w_A · (p_A + n)` (A holds more pages now)
//! - B's restore floor = `w_B · n_B ≤ w_A · (p_A + n)` (A won)
//! - B cannot evict A back. **Eviction is monotonically forward-progressing.**

use std::cmp::Ordering;
use std::collections::VecDeque;
use std::fmt;
use std::time::Instant;



use crate::device;
use crate::process::ProcessId;

use super::{
    ContextId, ContextManager, ContextState,
};
use super::pagestore::PhysicalPageId;
use super::sched::ProcessState;


// =============================================================================
// Queue Items
// =============================================================================

/// A deferred GPU page operation.
///
/// Used in both `alloc_queue` (FIFO for contention) and `deferred_ops`
/// (process-level deferral during suspension). On success, `on_alloc` is
/// called with pre-allocated pages. On deferral, the entire struct is stashed.
/// On cancellation (destroy), the struct is dropped — dropping the closure
/// drops the captured `oneshot::Sender`, closing the channel.
pub(crate) struct PendingAlloc {
    pub device: usize,
    pub num_pages: usize,
    /// Context this operation belongs to (for destroy filtering). None for Take.
    pub context_id: Option<ContextId>,
    /// Requester's priority floor at the time of enqueue (weight × post-alloc pages).
    pub priority_floor: f64,
    /// Callback invoked with allocated pages on success.
    pub on_alloc: Box<dyn FnOnce(&mut ContextManager, Vec<PhysicalPageId>) + Send>,
}

impl fmt::Debug for PendingAlloc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PendingAlloc")
            .field("device", &self.device)
            .field("num_pages", &self.num_pages)
            .field("context_id", &self.context_id)
            .field("priority_floor", &self.priority_floor)
            .field("on_alloc", &"<closure>")
            .finish()
    }
}


/// A deferred process restoration request (restore_queue queue).
#[derive(Debug)]
pub(crate) struct PendingRestore {
    pub process_id: ProcessId,
    pub priority_floor: f64,
    pub enqueued_at: Instant,
}

const AGING_RATE: f64 = 0.01;

impl PendingRestore {
    pub(crate) fn effective_priority(&self) -> f64 {
        let wait_secs = self.enqueued_at.elapsed().as_secs_f64();
        self.priority_floor + AGING_RATE * wait_secs
    }
}

impl PartialEq for PendingRestore {
    fn eq(&self, other: &Self) -> bool {
        self.process_id == other.process_id
    }
}
impl Eq for PendingRestore {}

impl PartialOrd for PendingRestore {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PendingRestore {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap — highest effective priority pops first.
        self.effective_priority()
            .partial_cmp(&other.effective_priority())
            .unwrap_or(Ordering::Equal)
    }
}

/// Result of suspending a process, with page-level accounting for a target device.
pub(crate) struct SuspendResult {
    pub pinned_count: usize,
    pub active_count: usize,
    /// GPU working pages freed immediately (from Active contexts on target device).
    pub working_freed: usize,
    /// GPU working pages deferred (from Pinned contexts on target device).
    pub working_deferred: usize,
    /// Committed pages that became unreferenced on target device (rc→0, from Active contexts).
    pub committed_freed: usize,
    /// Committed pages estimated to become unreferenced on target device
    /// once Pinned contexts clear (from Pinned contexts).
    pub committed_deferred: usize,
}

impl SuspendResult {
    /// Total GPU pages that will eventually be reclaimable on the target device.
    pub fn total_deferred(&self) -> usize {
        self.working_deferred + self.committed_deferred
    }
}

// =============================================================================
// Suspension methods on ContextManager
// =============================================================================

impl ContextManager {

    // ==================== GPU Page Contention ====================

    /// Pending-aware operation helper (no pages needed).
    ///
    /// If the process is Pending, defers `on_ready` as a zero-page `PendingAlloc`.
    /// Otherwise calls `on_ready` immediately. Used for operations like `pin`
    /// that don't need page allocation but must respect process suspension.
    pub(crate) fn with_gpu(
        &mut self,
        pid: ProcessId,
        context_id: Option<ContextId>,
        on_ready: impl FnOnce(&mut ContextManager) + Send + 'static,
    ) {
        self.with_gpu_pages(pid, 0, 0, context_id, move |mgr, _pages| on_ready(mgr));
    }

    /// Universal GPU page contention primitive.
    ///
    /// Attempts to allocate `num_pages` GPU pages on `dev_idx` for process `pid`.
    /// Goes through: Pending check → num_pages==0 fast-path →
    /// priority gate → free pool → eviction loop → self-suspend.
    ///
    /// On success, invokes `on_alloc` with the allocated pages.
    /// On deferral, the operation is stashed (in `alloc_queue` or `deferred_ops`)
    /// and will be replayed when pages become available.
    pub(crate) fn with_gpu_pages(
        &mut self,
        pid: ProcessId,
        dev_idx: usize,
        num_pages: usize,
        context_id: Option<ContextId>,
        on_alloc: impl FnOnce(&mut ContextManager, Vec<PhysicalPageId>) + Send + 'static,
    ) {
        // SUSPENSION CHECK: If the owning process is Pending, store as deferred op.
        {
            let proc = self.process_entry(pid);
            if proc.state == ProcessState::Pending {
                let pending = PendingAlloc {
                    device: dev_idx, num_pages, context_id, priority_floor: 0.0,
                    on_alloc: Box::new(on_alloc),
                };
                proc.deferred_ops.push(pending);
                return;
            }
        }

        // Short-circuit: no pages needed.
        if num_pages == 0 {
            (on_alloc)(self, Vec::new());
            return;
        }

        // Compute requester's priority floor (for priority gate and eviction).
        let requester_floor = {
            let proc = self.process_entry(pid);
            let pages = proc.pages_on(dev_idx);
            proc.weight * (pages + num_pages) as f64
        };

        // Step 1: PRIORITY GATE — compare requester floor vs restore_queue head.
        if let Some(top) = self.restore_queue.peek() {
            let top_pid = top.process_id;
            let top_ready = self.processes.get(&top_pid)
                .map(|p| p.pending_pinned == 0).unwrap_or(true);
            if top_ready && requester_floor < top.effective_priority() {
                // Requester loses priority gate → suspend and enqueue in restore_queue.
                // tracing::warn!("[CONTENTION] PRIORITY_GATE: pid={} floor={:.2} < top={:.2} (top_pid={}) → self-suspend | free={} rq={} aq={}",
                //     &pid.to_string()[..8], requester_floor, top.effective_priority(), &top_pid.to_string()[..8],
                //     self.devices[dev_idx].available_gpu_pages(),
                //     self.restore_queue.len(), self.alloc_queue.len());
                let pending = PendingAlloc {
                    device: dev_idx, num_pages, context_id, priority_floor: requester_floor,
                    on_alloc: Box::new(on_alloc),
                };
                let (pinned, _) = self.suspend_process(pid);
                self.enqueue_restore(pid, requester_floor, pinned);
                self.process_entry(pid).deferred_ops.push(pending);
                return;
            }
        }

        // Step 2: TRY ALLOCATE from free pool.
        if let Some(pages) = self.devices[dev_idx].alloc_gpu_pages(num_pages) {
            (on_alloc)(self, pages);
            return;
        }

        // Step 3: EVICTION LOOP — with deferred page tracking.
        // Structured to break with `alloc_result` to avoid conditional moves of `on_alloc`.
        let mut deferred_pages: usize = 0;
        let mut has_deferred = false;
        let mut alloc_result: Option<Vec<PhysicalPageId>> = None;

        loop {
            match self.find_eviction_victim(dev_idx, requester_floor, Some(pid)) {
                Some(victim_pid) => {
                    let victim_floor = self.processes.get(&victim_pid)
                        .map(|e| e.priority_on(dev_idx)).unwrap_or(0.0);
                    let estimate = self.estimate_suspend(victim_pid, dev_idx);
                    let (pinned, _) = self.suspend_process(victim_pid);
                    self.enqueue_restore(victim_pid, victim_floor, pinned);

                    if estimate.pinned_count > 0 {
                        deferred_pages += estimate.total_deferred();
                        has_deferred = true;
                    }

                    // Retry alloc after victim suspension freed pages.
                    if let Some(pages) = self.devices[dev_idx].alloc_gpu_pages(num_pages) {
                        // tracing::warn!("[CONTENTION] EVICT_OK: pid={} got {} pages after {} rounds | free_after={}",
                        //     &pid.to_string()[..8], num_pages, eviction_rounds, free_after);
                        alloc_result = Some(pages);
                        break;
                    }

                    if has_deferred {
                        let free_now = self.devices[dev_idx].available_gpu_pages();
                        if free_now + deferred_pages >= num_pages {
                            // tracing::warn!("[CONTENTION] EVICT_DEFER: pid={} wait for deferred ({} free + {} deferred >= {} needed) after {} rounds",
                            //     &pid.to_string()[..8], free_now, deferred_pages, num_pages, eviction_rounds);
                            break;
                        }
                    }
                }
                None => {
                    break;
                },
            }
        }

        // Post-loop: handle eviction results.
        if let Some(pages) = alloc_result {
            self.drain_queues();
            (on_alloc)(self, pages);
            return;
        }

        let pending = PendingAlloc {
            device: dev_idx, num_pages, context_id, priority_floor: requester_floor,
            on_alloc: Box::new(on_alloc),
        };

        if has_deferred {
            // Step 4: Deferred pages from Pinned contexts will cover the gap.
            // tracing::warn!("[CONTENTION] ALLOC_Q: pid={} enqueued in alloc_queue (need={}) | aq_len={}",
            //     &pid.to_string()[..8], num_pages, self.alloc_queue.len() + 1);
            self.alloc_queue.push_back(pending);
            return;
        }

        // Step 5: NO VICTIM — requester self-suspends.
        // tracing::warn!("[CONTENTION] SELF_SUSPEND: pid={} (floor={:.2}) no victim, self-suspending | free={} rq={} procs={}",
        //     &pid.to_string()[..8], requester_floor,
        //     self.devices[dev_idx].available_gpu_pages(),
        //     self.restore_queue.len(), self.processes.len());
        let (pinned, _) = self.suspend_process(pid);
        self.enqueue_restore(pid, requester_floor, pinned);
        self.process_entry(pid).deferred_ops.push(pending);
    }

    // ==================== Eviction ====================


    /// Find the cheapest eviction victim on a device.
    /// Returns the ProcessId of the victim, or None if no evictable process
    /// has lower priority than `requester_floor`.
    pub(crate) fn find_eviction_victim(
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
            let pages = proc.pages_on(dev_idx);
            if pages == 0 { continue; }

            let priority = proc.priority_on(dev_idx);

            // Anti-thrashing: only evict if victim's invested importance
            // is strictly less than requester's post-allocation floor.
            if priority >= requester_floor { continue; }

            let created = proc.created_at;
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

    /// Helper: enqueue a PendingRestore for a suspended process.
    pub(crate) fn enqueue_restore(&mut self, pid: ProcessId, priority_floor: f64, pinned_count: usize) {
        if pinned_count > 0 {
            if let Some(proc) = self.processes.get_mut(&pid) {
                proc.pending_pinned += pinned_count;
            }
        }
        self.restore_queue.push(PendingRestore {
            process_id: pid,
            priority_floor,
            enqueued_at: Instant::now(),
        });
    }

    /// Suspend all contexts of a process across ALL devices (cooperative suspension).
    /// - Active contexts: immediately suspend (working→CPU, chain released)
    /// - Pinned contexts: set `pending_suspend` flag (deferred)
    ///
    /// Always marks the process as Pending. Returns `(pinned_count, active_count)`.
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
                    if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
                        ctx.pending_suspend = true;
                    }
                    pinned_count += 1;
                }
                ContextState::Suspended => {}
            }
        }

        if let Some(proc) = self.processes.get_mut(&pid) {
            proc.state = ProcessState::Pending;
            proc.pending_replay_count = 0;
            for &dev in &devices_touched {
                proc.zero_device(dev);
            }
        }

        // Drain alloc_queue entries belonging to this process's contexts
        // and stash them as deferred_ops so they fire correctly on restore.
        let ctx_set: std::collections::HashSet<ContextId> = ctx_ids.iter().copied().collect();
        let mut kept = VecDeque::new();
        while let Some(entry) = self.alloc_queue.pop_front() {
            if entry.context_id.map_or(false, |cid| ctx_set.contains(&cid)) {
                self.processes.get_mut(&pid).unwrap().deferred_ops.push(entry);
            } else {
                kept.push_back(entry);
            }
        }
        self.alloc_queue = kept;

        // tracing::warn!("[CONTENTION] SUSPEND: pid={} active={} pinned={} ctxs={} | deferred_ops={}",
        //     &pid.to_string()[..8], active_count, pinned_count, ctx_ids.len(),
        //     self.processes.get(&pid).map(|p| p.deferred_ops.len()).unwrap_or(0));

        (pinned_count, active_count)
    }

    /// Estimate per-device page reclamation if `suspend_process(pid)` were called.
    /// Read-only: does not modify any state. Call BEFORE `suspend_process`.
    pub(crate) fn estimate_suspend(&self, pid: ProcessId, dev_idx: usize) -> SuspendResult {
        let ctx_ids = match self.processes.get(&pid) {
            Some(p) => &p.context_ids,
            None => return SuspendResult {
                pinned_count: 0, active_count: 0,
                working_freed: 0, working_deferred: 0,
                committed_freed: 0, committed_deferred: 0,
            },
        };

        let mut pinned_count: usize = 0;
        let mut active_count: usize = 0;
        let mut working_freed: usize = 0;
        let mut working_deferred: usize = 0;
        let mut committed_freed: usize = 0;
        let mut committed_deferred: usize = 0;

        for &ctx_id in ctx_ids {
            let ctx = match self.contexts.get(&ctx_id) {
                Some(c) => c,
                None => continue,
            };
            let dev = ctx.device.unwrap_or(0) as usize;

            match ctx.state {
                ContextState::Active => {
                    active_count += 1;
                    if dev == dev_idx {
                        working_freed += ctx.working_pages.len();
                        committed_freed += self.devices[dev].count_reclaimable(&ctx.committed_hashes);
                    }
                }
                ContextState::Pinned => {
                    pinned_count += 1;
                    if dev == dev_idx {
                        working_deferred += ctx.working_pages.len();
                        committed_deferred += self.devices[dev].count_reclaimable(&ctx.committed_hashes);
                    }
                }
                ContextState::Suspended => {}
            }
        }

        SuspendResult {
            pinned_count, active_count,
            working_freed, working_deferred,
            committed_freed, committed_deferred,
        }
    }

    /// Suspend a single Active context: swap working pages GPU→CPU, release chain.
    pub(crate) fn suspend_context(&mut self, ctx_id: ContextId) {
        let (dev_idx, working, committed_hashes) = match self.contexts.get(&ctx_id) {
            Some(ctx) if ctx.is_active() || ctx.is_pinned() => {
                (ctx.device.unwrap_or(0) as usize, ctx.working_pages.clone(), ctx.committed_hashes.clone())
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
                    self.devices[dev_idx].free_gpu_pages(&working);

                    if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
                        ctx.working_pages = cpu_pages;
                    }
                }
                None => {
                    tracing::error!(ctx = ctx_id, "suspend swap failed: no free CPU pages");
                    // Continue with suspension anyway — lose working pages
                    let pages_to_free = if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
                        let pages = ctx.working_pages.clone();
                        ctx.working_pages.clear();
                        pages
                    } else {
                        Vec::new()
                    };
                    self.devices[dev_idx].free_gpu_pages(&pages_to_free);
                }
            }
        }

        // Phase 2: Release committed chain refcounts
        if !committed_hashes.is_empty() {
            if let Some(dev) = self.devices.get_mut(dev_idx) {
                dev.release(&committed_hashes);
            }
        }

        // Phase 3: Mark suspended
        if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
            ctx.state = ContextState::Suspended;
            ctx.pending_suspend = false;
        }
    }
}
