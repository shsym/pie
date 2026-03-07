//! Suspension — Dual-Queue Contention Resolution Protocol.
//!
//! Implements the two-queue model from DESIGN.md:
//! - `try_alloc` (FIFO): deferred allocation requests from Running processes
//! - `try_restore` (Priority Heap): Pending processes awaiting full restoration
//!
//! ## Core Flows
//!
//! `reserve_working_pages` → attempt alloc → eviction loop → self-suspend to try_alloc
//! `drain_queues`  → Phase 1 (try_alloc FIFO) → Phase 2 (try_restore heap)
//! `clear_pinned`  → check pending_suspend flag → execute deferred suspension

use std::cmp::Ordering;
use std::time::Instant;
use tokio::sync::oneshot;


use crate::device;
use crate::process::ProcessId;

use super::{
    ContextId, ContextManager, ContextState,
};
use super::sched::ProcessState;


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

/// A deferred operation held while a process is Pending (suspended).
/// At most one can be active per process — the WASM guest is single-threaded
/// and blocked on the response channel.
#[derive(Debug)]
pub(crate) enum DeferredOp {
    /// Deferred page allocation (from `reserve_working_pages`).
    Alloc(AllocWaiter),
    /// Deferred pin (from `pin` on a non-active context).
    Pin {
        context_id: ContextId,
        num_input_tokens: u32,
        response: oneshot::Sender<anyhow::Result<super::PinnedContext>>,
    },
}

/// A deferred process restoration request (try_restore queue).
#[derive(Debug)]
pub(crate) struct RestoreWaiter {
    pub process_id: ProcessId,
    pub priority_floor: f64,
    pub enqueued_at: Instant,
}

const AGING_RATE: f64 = 0.01;

impl RestoreWaiter {
    pub(crate) fn effective_priority(&self) -> f64 {
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

    /// Helper: enqueue a RestoreWaiter for a suspended process.
    pub(crate) fn enqueue_restore(&mut self, pid: ProcessId, priority_floor: f64, pinned_count: usize) {
        if pinned_count > 0 {
            if let Some(proc) = self.processes.get_mut(&pid) {
                proc.pending_pinned += pinned_count;
            }
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
            for &dev in &devices_touched {
                proc.zero_device(dev);
            }
        }

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
                        committed_freed += ctx.committed_tip
                            .map(|t| self.devices[dev].estimate_chain_release(t))
                            .unwrap_or(0);
                    }
                }
                ContextState::Pinned => {
                    pinned_count += 1;
                    if dev == dev_idx {
                        working_deferred += ctx.working_pages.len();
                        committed_deferred += ctx.committed_tip
                            .map(|t| self.devices[dev].estimate_chain_release(t))
                            .unwrap_or(0);
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
        let (dev_idx, working, tip) = match self.contexts.get(&ctx_id) {
            Some(ctx) if ctx.is_active() || ctx.is_pinned() => {
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
                    self.devices[dev_idx].free_gpu_pages(&working);

                    if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
                        ctx.working_pages = cpu_pages;
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
                    self.devices[dev_idx].free_gpu_pages(&pages_to_free);
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
    }
}
