//! Scheduling — Market Tick, Contention, and Eviction.
//!
//! This module handles all scheduling economics and GPU page contention:
//!
//! - **Market tick**: Per-device price discovery (clearing price = min bid),
//!   Shapley cost-sharing, rent payments, default flagging, and dividends.
//! - **Contention**: `when_allocated` — the universal GPU page contention
//!   primitive. Handles free-pool allocation, eviction loops, priority gates,
//!   and deferred-op stashing for suspended contexts.
//! - **Eviction**: `find_eviction_victim` — selects the best eviction target
//!   based on `(defaulted DESC, bid ASC, spawn_time DESC)`.
//! - **Suspension**: `suspend` — stashes pages to CPU, releases GPU refcounts.
//! - **Queue draining**: `drain_queues` — serves `alloc_queue` (FIFO) and
//!   `restore_queue` (highest bid) as pages become available.
//!
//! ## Shapley Cost-Sharing
//!
//! Contexts sharing a KV-cache prefix split its physical cost equally via
//! Shapley values. `effective_pages(i)` = unique pages + Σ shared_segment_pages / refcount.
//! The auction uses effective pages for packing, payment, and sorting.
//!
//! ## Conservation Invariant
//!
//! At all times: `Σ balance_i = Σ endowment_i − M(t)`, where M(t) is cumulative
//! make cost. Rent revenue is exactly redistributed as dividends.

use std::cmp::Ordering;
use std::fmt;
use std::time::Instant;

use crate::device::{self, DeviceId};
use crate::process::ProcessId;

use super::pagestore::PhysicalPageId;
use super::{Context, ContextId, ContextManager, State, MARKET};

// =============================================================================
// ProcessEntry — Wallet + Ownership
// =============================================================================

/// Per-process wallet and ownership record.
///
/// Credits are shared across all contexts owned by the process. Payments are
/// charged per-context to this balance. Scheduling (eviction, restoration)
/// operates on individual contexts using their own bids.
#[derive(Debug)]
pub(crate) struct ProcessEntry {
    /// Credit balance (global, usable on any device).
    /// Set at admission; debited by payments, credited by dividends.
    pub balance: f64,
    /// Context IDs owned by this process.
    pub context_ids: Vec<ContextId>,
    /// Birth timestamp — used as FCFS tiebreaker at equal bid.
    pub created_at: Instant,
    /// Per-process token budget requested at launch time (None = use default).
    pub token_budget: Option<usize>,
    /// Initial credit endowment (fixed at creation, used for dividend weighting).
    /// Cannot be gamed by splitting or bidding — Sybil-resistant.
    pub endowment: f64,
}

impl ProcessEntry {
    pub(crate) fn new() -> Self {
        ProcessEntry {
            balance: 0.0,
            context_ids: Vec::new(),
            created_at: Instant::now(),
            token_budget: None,
            endowment: 0.0,
        }
    }
}

// =============================================================================
// Price Curves
// =============================================================================



// =============================================================================
// AuctionResult
// =============================================================================

/// Per-device auction outcome, computed each tick.
#[derive(Debug, Clone, Default)]
pub(crate) struct AuctionResult {
    /// GPU clearing price: the marginal context's bid (highest excluded bid).
    /// Zero when device is not fully packed.
    pub clearing_price: f64,
    /// CPU clearing price: for the CPU-tier cache auction.
    pub cpu_clearing_price: f64,
    /// Total revenue collected from critical-value payments this step (GPU + CPU).
    pub total_revenue: f64,
    /// Dividend rate for this device: device_revenue / total_endowment.
    /// Multiplied by a process's endowment to get its per-device dividend.
    pub dividend_per_endowment: f64,
}

// =============================================================================
// PendingAlloc — Deferred GPU Page Operation
// =============================================================================

/// A deferred GPU page operation stored on `ctx.deferred_ops`.
///
/// On success, `on_alloc` is called with pre-allocated pages.
/// On cancellation (destroy), the struct is dropped — dropping the closure
/// drops the captured `oneshot::Sender`, closing the channel.
pub(crate) struct PendingAlloc {
    pub device: usize,
    pub num_pages: usize,
    /// Callback invoked with allocated pages on success.
    pub on_alloc: Box<dyn FnOnce(&mut ContextManager, Vec<PhysicalPageId>) + Send>,
}

impl fmt::Debug for PendingAlloc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PendingAlloc")
            .field("device", &self.device)
            .field("num_pages", &self.num_pages)
            .field("on_alloc", &"<closure>")
            .finish()
    }
}

// =============================================================================
// Process Lifecycle
// =============================================================================

impl ContextManager {

    /// Explicitly register a process with its token budget.
    /// Creates a `ProcessEntry` with the correct endowment.
    /// Called once per process lifetime (from `InstanceState::new`).
    ///
    /// Panics on double-registration (indicates a bug in the caller).
    pub(crate) fn register_process(&mut self, pid: ProcessId, token_budget: Option<usize>) {
        assert!(
            !self.processes.contains_key(&pid),
            "register_process: process {pid} already registered"
        );
        let mut entry = ProcessEntry::new();
        entry.token_budget = token_budget;
        let credit = match token_budget {
            Some(budget) => {
                // credit = ⌈token_budget / page_size⌉
                let credit = budget.div_ceil(self.page_size.max(1));
                credit as f64
            }
            None => self.default_endowment,
        };
        entry.balance = credit;
        entry.endowment = credit;
        self.processes.insert(pid, entry);
        if let Some(market) = MARKET.get(self.model_idx) {
            market.balances.insert(pid, credit);
            market.endowments.insert(pid, credit);
        }
    }

    /// Unregister a process: destroy all owned contexts and remove the process entry.
    /// Called on WASM instance drop for automatic cleanup.
    pub(crate) fn unregister_process(&mut self, pid: ProcessId) {
        let proc = match self.processes.remove(&pid) {
            Some(p) => p,
            None => return,
        };

        // Drop this process's contexts from alloc_queue.
        let ctx_ids: std::collections::HashSet<ContextId> = proc.context_ids.iter().copied().collect();
        self.alloc_queue.retain(|ctx_id| !ctx_ids.contains(ctx_id));

        // Remove from restore_queue
        self.restore_queue.retain(|ctx_id| !ctx_ids.contains(ctx_id));

        // Destroy all owned contexts
        for ctx_id in proc.context_ids {
            if let Some(ctx) = self.contexts.remove(&ctx_id) {
                let dev_idx = ctx.device.unwrap_or(0) as usize;
                if !ctx.committed_hashes.is_empty() && !ctx.is_off_gpu() {
                    self.gpu_stores[dev_idx].release(&ctx.committed_hashes);
                }
                if !ctx.working_pages.is_empty() {
                    self.gpu_stores[dev_idx].free(&ctx.working_pages);
                }
                if !ctx.cpu_working_pages.is_empty() {
                    self.cpu_stores[dev_idx].free(&ctx.cpu_working_pages);
                }
                if ctx.is_off_gpu() && !ctx.committed_hashes.is_empty() {
                    self.cpu_stores[dev_idx].release(&ctx.committed_hashes);
                }
                self.snapshots.retain(|_, v| *v != ctx_id);
                self.remove_context_caches(ctx_id);
            }
        }

        if let Some(market) = MARKET.get(self.model_idx) {
            market.balances.remove(&pid);
            market.endowments.remove(&pid);
        }
        self.drain_queues();
    }

    /// Get a mutable reference to a registered process's entry.
    /// Panics if the process is not registered — callers must ensure
    /// `register_process()` was called first.
    pub(crate) fn process_entry(&mut self, pid: ProcessId) -> &mut ProcessEntry {
        self.processes.get_mut(&pid)
            .expect("process_entry: process not registered (missing register_process call)")
    }

    // =========================================================================
    // Best Device — Per-Context Shapley Placement (§4.3)
    // =========================================================================

    /// Evaluate the cheapest device for a single context using Shapley costs.
    ///
    /// Computes `effective_pages(ctx, d) × clearing_price(d)` on every device,
    /// accounting for migration cost. Returns the device index with lowest cost.
    ///
    /// Called from `drain_queues` before each restore.
    pub(crate) fn best_device_for(&self, ctx: &Context) -> usize {
        let num_devices = self.gpu_stores.len();
        let current_dev = ctx.device.unwrap_or(0) as usize;
        if num_devices <= 1 { return current_dev; }

        let hashes = &ctx.committed_hashes;
        if hashes.is_empty() { return current_dev; }

        let working_count = if ctx.is_off_gpu() {
            ctx.suspended_working_count as f64
        } else {
            ctx.working_pages.len() as f64
        };

        let mut best_dev = current_dev;
        let mut best_cost = f64::MAX;

        for d in 0..num_devices {
            let shared = self.gpu_stores[d].prefix_len(hashes);
            let shared_eff = if shared > 0 {
                self.gpu_stores[d].effective_pages(&hashes[..shared])
            } else {
                0.0
            };
            let unique_eff = (hashes.len() - shared) as f64 + working_count;
            let eff = shared_eff + unique_eff;
            let price = self.auction_results.get(d)
                .map(|a| a.clearing_price).unwrap_or(0.0);

            let migration_cost = if d != current_dev {
                hashes.len() as f64
            } else {
                0.0
            };

            let total_cost = eff * price + migration_cost;
            if total_cost < best_cost {
                best_cost = total_cost;
                best_dev = d;
            }
        }

        best_dev
    }

    // =========================================================================
    // Market Tick — Price Discovery + Payments + Dividends
    // =========================================================================

    /// Run one market tick for a single device: price discovery + payments + dividends.
    ///
    /// Called once per batch completion on the device that just finished.
    /// Only runs the auction for `dev_idx`, charges payments to contexts on
    /// that device, and distributes dividends proportional to the single-device
    /// revenue.
    ///
    /// No eviction or page movement — those are handled by `when_allocated`
    /// and `drain_queues`.
    pub(crate) fn tick(&mut self, dev_idx: usize, latency_secs: f64) {
        // Ensure auction_results vec is large enough.
        if self.auction_results.len() <= dev_idx {
            self.auction_results.resize(dev_idx + 1, AuctionResult::default());
        }

        // All contexts on this device are admitted by construction (the
        // contention/eviction system enforces physical capacity).
        // Clearing price = smallest bid on the device.
        let mut clearing_price = f64::MAX;

        // Pass 1: find clearing price (min bid).
        for ctx in self.contexts.values() {
            if ctx.owner.is_none() { continue; }
            if ctx.device.unwrap_or(0) as usize != dev_idx { continue; }
            if ctx.is_off_gpu() { continue; }
            clearing_price = clearing_price.min(ctx.bid);
        }
        if clearing_price == f64::MAX { clearing_price = 0.0; }

        // Pass 2: compute Shapley effective pages, accumulate per-context rent.
        let mut ctx_payments: Vec<(ContextId, ProcessId, f64)> = Vec::new();
        let mut device_revenue = 0.0f64;

        for (&ctx_id, ctx) in &self.contexts {
            if ctx.owner.is_none() { continue; }
            if ctx.device.unwrap_or(0) as usize != dev_idx { continue; }
            if ctx.is_off_gpu() { continue; }

            let raw_pages = ctx.committed_hashes.len() + ctx.working_pages.len();
            if raw_pages == 0 { continue; }

            let eff = self.gpu_stores[dev_idx].effective_pages(&ctx.committed_hashes)
                + ctx.working_pages.len() as f64;

            let payment = clearing_price * eff;
            device_revenue += payment;
            if let Some(pid) = ctx.owner {
                ctx_payments.push((ctx_id, pid, payment));
            }
        }

        // Endowment-proportional dividends from this device's revenue.
        let total_endowment: f64 = self.processes.values().map(|p| p.endowment).sum();
        let dividend_rate = if total_endowment > 0.0 { device_revenue / total_endowment } else { 0.0 };

        self.auction_results[dev_idx] = AuctionResult {
            clearing_price,
            cpu_clearing_price: 0.0,
            total_revenue: device_revenue,
            dividend_per_endowment: dividend_rate,
        };

        // Pass 3: charge rent, flag defaulted contexts.
        // A context is defaulted if its process can't afford the full payment.
        for (ctx_id, pid, payment) in &ctx_payments {
            if let Some(proc) = self.processes.get_mut(pid) {
                let defaulted = proc.balance < *payment;
                let actual = payment.min(proc.balance);
                proc.balance -= actual;
                if let Some(ctx) = self.contexts.get_mut(ctx_id) {
                    ctx.defaulted = defaulted;
                }
            }
        }

        // Dividends: credit all processes.
        for proc in self.processes.values_mut() {
            proc.balance += dividend_rate * proc.endowment;
        }

        // Publish market data + balances to lock-free cache.
        if let Some(market) = MARKET.get(self.model_idx) {
            market.clearing_prices.insert(dev_idx, clearing_price);
            // EWA-smooth the tick latency (α = 0.1).
            let alpha = 0.1;
            let prev = market.tick_latency_ewa.get(&dev_idx).map(|v| *v).unwrap_or(latency_secs);
            market.tick_latency_ewa.insert(dev_idx, alpha * latency_secs + (1.0 - alpha) * prev);
            let rate_sum: f64 = self.auction_results.iter()
                .map(|a| a.dividend_per_endowment).sum();
            market.set_dividend_rate(rate_sum);
            for (&pid, proc) in &self.processes {
                market.balances.insert(pid, proc.balance);
            }
        }
    }

    /// Set a context's bid (willingness to pay per page per step).
    pub(crate) fn bid(&mut self, id: ContextId, bid: f64) -> anyhow::Result<()> {
        match self.contexts.get_mut(&id) {
            Some(ctx) => {
                ctx.bid = bid.max(0.0);
                Ok(())
            }
            None => Err(anyhow::anyhow!("Context {id} not found")),
        }
    }

    // =========================================================================
    // Make Cost
    // =========================================================================

    /// Charge make cost for newly allocated pages to the owning process.
    /// Called at point of allocation (not retroactively in tick).
    pub(crate) fn charge_make_cost(&mut self, ctx_id: ContextId, num_pages: usize) {
        if num_pages == 0 { return; }
        let pid = match self.contexts.get(&ctx_id).and_then(|c| c.owner) {
            Some(pid) => pid,
            None => return,
        };
        if let Some(proc) = self.processes.get_mut(&pid) {
            proc.balance -= num_pages as f64;
        }
    }

    /// Check if the owning process can afford the make cost for `num_pages`.
    pub(crate) fn can_afford(&self, ctx_id: ContextId, num_pages: usize) -> bool {
        if num_pages == 0 { return true; }
        let pid = match self.contexts.get(&ctx_id).and_then(|c| c.owner) {
            Some(pid) => pid,
            None => return true, // snapshots (no owner) are always allowed
        };
        match self.processes.get(&pid) {
            Some(proc) => proc.balance >= num_pages as f64,
            None => false,
        }
    }

    // =========================================================================
    // GPU Page Contention
    // =========================================================================

    /// Pending-aware operation helper (no pages needed).
    ///
    /// If the owning context is Suspended, defers `on_ready` as a zero-page
    /// `PendingAlloc`. Otherwise calls `on_ready` immediately. Used for
    /// operations like `pin` that don't need page allocation but must respect
    /// context suspension.
    pub(crate) fn when_active(
        &mut self,
        ctx_id: ContextId,
        on_ready: impl FnOnce(&mut ContextManager) + Send + 'static,
    ) {
        self.when_allocated(ctx_id, 0, 0, move |mgr, _pages| on_ready(mgr));
    }

    /// Universal GPU page contention primitive.
    ///
    /// Attempts to allocate `num_pages` GPU pages on `dev_idx` for context `ctx_id`.
    /// Goes through: Suspended check → num_pages==0 fast-path →
    /// priority gate → free pool → eviction loop → self-suspend.
    ///
    /// On success, invokes `on_alloc` with the allocated pages.
    /// On deferral, the operation is stashed on `ctx.deferred_ops`
    /// and will be replayed when pages become available.
    pub(crate) fn when_allocated(
        &mut self,
        ctx_id: ContextId,
        dev_idx: usize,
        num_pages: usize,
        on_alloc: impl FnOnce(&mut ContextManager, Vec<PhysicalPageId>) + Send + 'static,
    ) {
        // SUSPENSION CHECK: If the context is Suspended, store as deferred op.
        if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
            if ctx.is_off_gpu() {
                let pending = PendingAlloc {
                    device: dev_idx, num_pages,
                    on_alloc: Box::new(on_alloc),
                };
                ctx.deferred_ops.push(pending);
                return;
            }
        } else {
            tracing::error!("when_allocated: context not found: {}", ctx_id);
            return;
        }

        // Short-circuit: no pages needed.
        if num_pages == 0 {
            (on_alloc)(self, Vec::new());
            return;
        }

        // CREDIT CHECK: if process can't afford make cost, self-suspend.
        if !self.can_afford(ctx_id, num_pages) {
            let pending = PendingAlloc {
                device: dev_idx, num_pages,
                on_alloc: Box::new(on_alloc),
            };
            self.contexts.get_mut(&ctx_id).unwrap().deferred_ops.push(pending);
            self.suspend(ctx_id);
            self.enqueue_restore(ctx_id);
            self.drain_queues();
            return;
        }

        // Compute requester's bid from the context.
        let requester_bid = self.contexts[&ctx_id].bid;

        // Step 2: PRIORITY GATE — compare requester bid vs restore_queue head.
        // If a higher-bidding Suspended context is waiting, the requester yields.
        if let Some(top_id) = self.highest_bid_in_restore_queue() {
            let top_bid = self.contexts[&top_id].bid;
            if requester_bid < top_bid {
                let pending = PendingAlloc {
                    device: dev_idx, num_pages,
                    on_alloc: Box::new(on_alloc),
                };
                self.suspend(ctx_id);
                self.enqueue_restore(ctx_id);
                self.contexts.get_mut(&ctx_id).unwrap().deferred_ops.push(pending);
                self.drain_queues();
                return;
            }
        }

        // Step 3: TRY ALLOCATE from free pool.
        if let Some(pages) = self.gpu_stores[dev_idx].alloc(num_pages) {
            self.charge_make_cost(ctx_id, pages.len());
            (on_alloc)(self, pages);
            return;
        }

        // Step 4: EVICTION LOOP — with deferred page tracking.
        let mut deferred_pages: usize = 0;
        let mut has_deferred = false;
        let mut alloc_result: Option<Vec<PhysicalPageId>> = None;

        loop {
            match self.find_eviction_victim(dev_idx, requester_bid, Some(ctx_id)) {
                Some(victim_ctx_id) => {
                    let victim = self.contexts.get(&victim_ctx_id).unwrap();

                    if victim.is_pinned() {
                        // Pinned victim: set pending_suspend, pages freed later on unpin.
                        let reclaimable = victim.working_pages.len()
                            + self.gpu_stores[dev_idx].count_reclaimable(&victim.committed_hashes);
                        self.contexts.get_mut(&victim_ctx_id).unwrap().pending_suspend = true;
                        deferred_pages += reclaimable;
                        has_deferred = true;
                    } else {
                        // Active victim: suspend immediately.
                        self.suspend(victim_ctx_id);
                        self.enqueue_restore(victim_ctx_id);
                    }

                    // Retry alloc after victim suspension freed pages.
                    if let Some(pages) = self.gpu_stores[dev_idx].alloc(num_pages) {
                        alloc_result = Some(pages);
                        break;
                    }

                    if has_deferred {
                        let free_now = self.gpu_stores[dev_idx].available();
                        if free_now + deferred_pages >= num_pages {
                            break;
                        }
                    }
                }
                None => break,
            }
        }

        // Post-loop: handle eviction results.
        if let Some(pages) = alloc_result {
            self.charge_make_cost(ctx_id, pages.len());
            (on_alloc)(self, pages);
            self.drain_queues();
        } else {
            // No pages available — defer the operation.
            let pending = PendingAlloc {
                device: dev_idx, num_pages,
                on_alloc: Box::new(on_alloc),
            };
            self.contexts.get_mut(&ctx_id).unwrap().deferred_ops.push(pending);

            if has_deferred {
                // Step 5: Deferred pages from Pinned contexts will cover the gap.
                self.alloc_queue.push_back(ctx_id);
            } else {
                // Step 6: NO VICTIM — requester self-suspends.
                self.suspend(ctx_id);
                self.enqueue_restore(ctx_id);
                self.drain_queues();
            }
        }
    }

    // =========================================================================
    // Eviction
    // =========================================================================

    /// Find the best eviction victim context on a device.
    ///
    /// Priority: defaulted contexts first (regardless of bid), then by
    /// lowest bid, then by most-recently-spawned process (FCFS tiebreaker).
    ///
    /// Non-defaulted victims must have bid ≤ requester_bid.
    /// Defaulted victims are always eligible (they can't pay rent).
    pub(crate) fn find_eviction_victim(
        &self,
        dev_idx: usize,
        requester_bid: f64,
        requester: Option<ContextId>,
    ) -> Option<ContextId> {
        // (defaulted, bid, spawn_time, ctx_id) — best victim has highest
        // defaulted, lowest bid, latest spawn_time.
        let mut best: Option<(bool, f64, Instant, ContextId)> = None;

        for (&ctx_id, ctx) in &self.contexts {
            if requester == Some(ctx_id) { continue; }
            if ctx.is_off_gpu() { continue; }
            let ctx_dev = ctx.device.unwrap_or(0) as usize;
            if ctx_dev != dev_idx { continue; }
            let pages = ctx.committed_hashes.len() + ctx.working_pages.len();
            if pages == 0 { continue; }
            if ctx.pending_suspend { continue; }

            // Non-defaulted contexts must have bid ≤ requester's bid.
            // Defaulted contexts are always evictable (they owe rent).
            if !ctx.defaulted && ctx.bid > requester_bid { continue; }

            let spawn_time = ctx.owner
                .and_then(|pid| self.processes.get(&pid))
                .map(|p| p.created_at)
                .unwrap_or_else(Instant::now);

            let dominated = if let Some((best_def, best_bid, best_time, _)) = best {
                // Prefer defaulted over non-defaulted
                (ctx.defaulted && !best_def)
                // Same default status: prefer lower bid
                || (ctx.defaulted == best_def && ctx.bid < best_bid)
                // Same default + bid: prefer later spawn (FCFS)
                || (ctx.defaulted == best_def && ctx.bid == best_bid && spawn_time > best_time)
            } else {
                true
            };
            if dominated {
                best = Some((ctx.defaulted, ctx.bid, spawn_time, ctx_id));
            }
        }

        best.map(|(_, _, _, ctx_id)| ctx_id)
    }

    /// Helper: enqueue a context for restoration.
    pub(crate) fn enqueue_restore(&mut self, ctx_id: ContextId) {
        self.restore_queue.push(ctx_id);
    }

    /// Find the best context to restore from the queue.
    /// Non-defaulted contexts are preferred (by highest bid).
    /// Defaulted contexts are deprioritized — only restored when no
    /// non-defaulted candidates remain.
    pub(crate) fn highest_bid_in_restore_queue(&self) -> Option<ContextId> {
        self.restore_queue.iter()
            .filter_map(|&id| self.contexts.get(&id).map(|c| (id, c.defaulted, c.bid)))
            .max_by(|a, b| {
                // Non-defaulted (false) before defaulted (true), then highest bid.
                b.1.cmp(&a.1)
                    .then(a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal))
            })
            .map(|(id, _, _)| id)
    }

    // =========================================================================
    // CPU Eviction — tier-boundary contention (STORAGE.md §4.1b)
    // =========================================================================

    /// Find the best CPU eviction victim on a device.
    ///
    /// Iterates all **Stashed** contexts (CPU-resident pages) on the device.
    /// Returns the context with the lowest bid, using FCFS (latest spawn
    /// time first) as tiebreaker.
    ///
    /// The requester is excluded. Only victims with bid ≤ `requester_bid`
    /// are eligible (a higher-bid context should not be evicted to make
    /// room for a lower-bid one).
    fn find_cpu_eviction_victim(
        &self,
        dev_idx: usize,
        requester_bid: f64,
        requester: Option<ContextId>,
    ) -> Option<ContextId> {
        let mut best: Option<(f64, Instant, ContextId)> = None;

        for (&ctx_id, ctx) in &self.contexts {
            if requester == Some(ctx_id) { continue; }
            if !ctx.is_stashed() { continue; }
            let ctx_dev = ctx.device.unwrap_or(0) as usize;
            if ctx_dev != dev_idx { continue; }

            // Must have CPU-resident pages (working stash or committed stash).
            let has_cpu_working = !ctx.cpu_working_pages.is_empty();
            let has_cpu_committed = !ctx.committed_hashes.is_empty()
                && self.cpu_stores[dev_idx].prefix_len(&ctx.committed_hashes) > 0;
            if !has_cpu_working && !has_cpu_committed { continue; }

            // Only evict contexts with bid ≤ requester's bid.
            if ctx.bid > requester_bid { continue; }

            let spawn_time = ctx.owner
                .and_then(|pid| self.processes.get(&pid))
                .map(|p| p.created_at)
                .unwrap_or_else(Instant::now);

            let dominated = if let Some((best_bid, best_time, _)) = best {
                ctx.bid < best_bid
                || (ctx.bid == best_bid && spawn_time > best_time)
            } else {
                true
            };
            if dominated {
                best = Some((ctx.bid, spawn_time, ctx_id));
            }
        }

        best.map(|(_, _, ctx_id)| ctx_id)
    }

    /// Evict a Stashed context's pages from CPU to recompute.
    ///
    /// Releases committed pages from the CPU FlatPageStore (rc--, free at
    /// rc=0) and frees working page stash from the CPU pool. Transitions
    /// the context from Stashed to Suspended (no CPU cache, full recompute
    /// on restore).
    fn evict_from_cpu(&mut self, ctx_id: ContextId) {
        let (dev_idx, committed_hashes, cpu_working) = match self.contexts.get(&ctx_id) {
            Some(ctx) if ctx.is_stashed() => (
                ctx.device.unwrap_or(0) as usize,
                ctx.committed_hashes.clone(),
                ctx.cpu_working_pages.clone(),
            ),
            _ => return,
        };

        // Release committed pages from CPU store.
        if !committed_hashes.is_empty() {
            self.cpu_stores[dev_idx].release(&committed_hashes);
        }

        // Free working page stash from CPU pool.
        if !cpu_working.is_empty() {
            self.cpu_stores[dev_idx].free(&cpu_working);
            if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
                ctx.cpu_working_pages.clear();
            }
        }

        // Transition Stashed → Suspended (no longer has CPU pages).
        if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
            ctx.state = State::Suspended;
        }
    }

    // =========================================================================
    // Suspension
    // =========================================================================

    /// Suspend a single Active context: stash pages to CPU, then release GPU.
    ///
    /// Uses `would_free()` to identify which committed pages will reach rc=0
    /// on release, and stashes only those to CPU via FlatPageStore. Shared
    /// prefix pages (rc > 1) stay on GPU.
    ///
    /// When the CPU pool is full, runs an eviction loop to free CPU pages
    /// from the lowest-bid suspended context before falling through to
    /// recompute (STORAGE.md §4.1b).
    pub(crate) fn suspend(&mut self, ctx_id: ContextId) {
        let (dev_idx, working, committed_hashes) = match self.contexts.get(&ctx_id) {
            Some(ctx) if ctx.is_active() || ctx.is_pinned() => {
                (ctx.device.unwrap_or(0) as usize, ctx.working_pages.clone(), ctx.committed_hashes.clone())
            }
            _ => return,
        };

        // Compute total CPU pages needed upfront for a single eviction pass.
        let requester_bid = self.contexts.get(&ctx_id).map(|c| c.bid).unwrap_or(0.0);
        let working_cpu_needed = working.len();
        let evictable_hashes = if !committed_hashes.is_empty() {
            self.gpu_stores[dev_idx].would_free(&committed_hashes)
        } else {
            Vec::new()
        };
        let committed_cpu_needed = evictable_hashes.len();
        let total_cpu_needed = working_cpu_needed + committed_cpu_needed;

        // Single eviction pass: free enough CPU pages for both phases.
        if total_cpu_needed > 0 && self.cpu_stores[dev_idx].available() < total_cpu_needed {
            while self.cpu_stores[dev_idx].available() < total_cpu_needed {
                match self.find_cpu_eviction_victim(dev_idx, requester_bid, Some(ctx_id)) {
                    Some(victim_id) => {
                        self.evict_from_cpu(victim_id);
                    }
                    None => break,
                }
            }
        }

        // All-or-nothing: only stash to CPU if enough space for the full
        // request. Partial stashing (working on CPU, committed dropped) would
        // waste CPU capacity on a context that still needs full recompute.
        let cpu_offload = total_cpu_needed > 0
            && self.cpu_stores[dev_idx].available() >= total_cpu_needed;

        // Phase 1: Stash working pages to CPU.
        // Working pages are exclusive — just D2H copy to CPU pool.
        if !working.is_empty() {
            let ctx = self.contexts.get_mut(&ctx_id).unwrap();
            ctx.suspended_working_count = ctx.working_pages.len();
            ctx.working_pages.clear();

            if cpu_offload {
                if let Some(cpu_pages) = self.cpu_stores[dev_idx].alloc(working.len()) {
                    let _ = device::copy_d2h(dev_idx as DeviceId, &working, &cpu_pages);
                    let ctx = self.contexts.get_mut(&ctx_id).unwrap();
                    ctx.cpu_working_pages = cpu_pages;
                }
            }

            self.gpu_stores[dev_idx].free(&working);
        }

        // Phase 2: Stash evictable committed pages to CPU.
        // Only pages with rc=1 (will reach rc=0 on release) need stashing.
        // Shared prefix pages (rc > 1) stay on GPU for other contexts.
        if cpu_offload && !evictable_hashes.is_empty() {
            let gpu_phys = self.gpu_stores[dev_idx].physical_ids(&evictable_hashes);
            if let Some(cpu_pages) = self.cpu_stores[dev_idx].alloc(gpu_phys.len()) {
                let _ = device::copy_d2h(dev_idx as DeviceId, &gpu_phys, &cpu_pages);
                self.cpu_stores[dev_idx].insert(&evictable_hashes, &cpu_pages);
            }
        }

        // Phase 3: Release committed chain refcounts from GPU trie.
        if !committed_hashes.is_empty() {
            if let Some(dev) = self.gpu_stores.get_mut(dev_idx) {
                dev.release(&committed_hashes);
            }
        }

        // Phase 4: Mark stashed or suspended.
        let ctx = self.contexts.get_mut(&ctx_id).unwrap();
        ctx.state = if cpu_offload { State::Stashed } else { State::Suspended };
        ctx.pending_suspend = false;

        // Remove this context from alloc_queue (can't serve while suspended;
        // deferred_ops are already on the context and will replay on restore).
        self.alloc_queue.retain(|&id| id != ctx_id);
    }

    /// Voluntarily suspend a context (program-initiated).
    pub(crate) fn voluntary_suspend(&mut self, id: ContextId) -> anyhow::Result<()> {
        match self.contexts.get(&id) {
            Some(ctx) if ctx.is_active() => {
                self.suspend(id);
                self.enqueue_restore(id);
                self.drain_queues();
                Ok(())
            }
            Some(ctx) if ctx.is_off_gpu() => Ok(()), // already off GPU
            Some(_) => Err(anyhow::anyhow!("Context {id} is pinned, cannot voluntarily suspend")),
            None => Err(anyhow::anyhow!("Context {id} not found")),
        }
    }
}
