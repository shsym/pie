//! Batch scheduling policies.
//!
//! Two policies live here:
//!   - `AdaptiveBatchPolicy` (default) — production scheduler.
//!   - `GreedyPolicy` — zero-state baseline. Useful for comparison and
//!     correctness debugging.
//!
//! Selection at runtime via `PIE_POLICY` env var in `scheduler.rs`
//! (default = adaptive). Most callers should leave it default; greedy is
//! retained because its simplicity makes it a useful sanity reference
//! ("does the *scheduler* itself work, ignoring policy?") and a
//! Pareto-optimal choice when the workload is genuinely RPS=1.
//!
//! ## Design history (why this is what it is)
//!
//! Pie performs **iteration-level batching**: each in-flight context
//! re-submits a forward-pass request after every token. The scheduler
//! accumulates these into a batch and the policy decides when to fire.
//!
//! An earlier version of this file used `AdaptiveThroughputPolicy`,
//! which fired when `idle > ref_l/B` (where `ref_l` was a leaky-max of
//! observed batch latencies, decayed by `0.9999` per batch). The
//! intuition was: bigger batches need shorter waits because they're
//! already "full enough". That policy had three failure modes that
//! showed up clearly under benchmark:
//!
//!   1. **Low-RPS over-wait.** With a single in-flight context, B=1
//!      and `ref_l/B = ref_l`. The 50 ms `max_wait_time` cap fires the
//!      batch every iteration, paying ~50 ms of pure waiting per token.
//!      Per-request latency at RPS=1 was ~3× a no-wait baseline.
//!
//!   2. **Idle-phase pathology under churn.** `ref_l` is a max-ratchet,
//!      so once a high-RPS phase elevates it, decay is glacial (half-life
//!      ~7000 batches). A subsequent idle phase still has `ref_l` at the
//!      old peak, so every B=1 batch waits its full 50 ms cap. We
//!      measured idle-phase p50 of 995 ms vs 541 ms for the new policy
//!      under a 0.5↔64 RPS churn pattern.
//!
//!   3. **Premature mid-burst fire.** As B ramps up during peer
//!      accumulation, `ref_l/B` shrinks aggressively. A 0.5 ms gap
//!      between two peer arrivals at B=4 trips `idle > ref_l/B` and the
//!      batch fires fragmented (we saw `B=4 fired with pinned=13` in
//!      instrumented runs). Stragglers go in the next batch and pay an
//!      extra full iteration of latency.
//!
//! Several intermediate fixes were tried — adding a "cohort cap"
//! (`B >= pinned_count → fire`), faster `DECAY`, idle-based grace
//! periods, λ-tracking — and the path through them led to the present
//! design, which is **simpler than any intermediate**:
//!
//!   - The cohort cap alone fixes (1) and (2).
//!   - Removing the `idle > ref_l/B` rule entirely fixes (3).
//!   - That removal also drops the need for `DECAY` smoothing, since
//!     the latency value is no longer used as a sensitivity-amplified
//!     budget.
//!
//! What remains is two firing conditions plus a safety bound, with no
//! magic constants. See `AdaptiveBatchPolicy` below.

use std::time::{Duration, Instant};

use super::scheduler::{Decision, SchedulingPolicy};

// =============================================================================
// AdaptiveBatchPolicy — the default. Cohort cap + last-batch-time bound.
// =============================================================================
//
// **Firing rule** (in evaluation order):
//
//   1. `B >= max_batch_size`          — structural device limit
//   2. `B >= pinned_count(device)`    — every active context is in the
//                                       batch; no peer can join this
//                                       iteration. Fire (the load-bearing
//                                       condition; see below).
//   3. `last_latency == 0.0`          — cold start, no measurement yet
//   4. `start.elapsed() >= last_latency`
//                                     — absolute wait bound. Waiting
//                                       longer than one batch's compute
//                                       time costs more in queue delay
//                                       than any plausible saving from a
//                                       larger batch.
//   otherwise: `Wait(last_latency - start.elapsed())`
//
// **Why the cohort cap is load-bearing.** In iteration-level batching,
// `pinned_count(device)` is the count of currently-active contexts on
// that device. After a batch fires, every one of those contexts will
// eventually submit its next forward-pass request. Once we have all of
// them in the accumulator (B == pinned_count), waiting longer can't
// produce a bigger batch from existing peers — it can only catch
// brand-new contexts that pin during the wait. At realistic Poisson
// arrival rates that's rarely worth one extra batch's worth of delay
// for the existing peers. So: fire.
//
// Pinned-count semantics are explicitly NOT "currently submitting" —
// it includes contexts mid-CPU-work between iterations. That sometimes
// over-counts: if a peer's inferlet is slow (e.g., calling out to a
// tool), B can stay below pinned_count indefinitely. That's what
// condition (4) is for.
//
// **Why `last_latency` (not `ref_l` with smoothing).** The original
// policy used a leaky-max of recent batch latencies. The smoothing
// served two purposes: (a) damping spikes, (b) preventing a death
// spiral where a single small batch shrinks the budget so far that all
// subsequent batches are also small. With the cohort cap as the primary
// fire mechanism, neither problem materializes:
//
//   - The bound only fires when B < pinned_count — i.e., when peers are
//     stuck. In that regime, firing fast IS correct; there's nothing to
//     wait for.
//   - When the bound *does* fire, the resulting batch's actual decode
//     time becomes the next `last_latency`. The value re-calibrates
//     within 1–2 batches.
//
// Empirically validated: tested under steady RPS sweeps (1, 4, 16, 64),
// symmetric churn (32↔1), severe idle↔burst churn (0.5↔64), and rapid
// 2-second bursts. No spiral observed in any pattern.
//
// **Why `start.elapsed()`, not `idle`.** An earlier variant used
// `idle = time since last_arrival` for the bound. That mechanism resets
// on every peer arrival, so at high RPS new peer arrivals during the
// wait kept extending it indefinitely (runaway: 10 s p50 at RPS=64,
// achieved-rate halved). Bounding by `start.elapsed()` (time since the
// *first* arrival of this batch) makes the wait monotonic and bounded
// by `last_latency` regardless of mid-wait arrivals.

pub(super) struct AdaptiveBatchPolicy {
    max_batch_size: usize,
    /// Index of the device this policy is scheduling for. Used to read
    /// `pinned_count` (lock-free atomic, updated on context pin/unpin).
    device_idx: usize,
    /// `Some(t)` while a batch is accumulating; `None` after fire and
    /// before the next first arrival.
    batch_start_time: Option<Instant>,
    /// Most recent batch's compute time, in seconds. Updated on
    /// `on_complete`. Used as the absolute upper bound on how long to
    /// wait when `B < pinned_count`. Zero until the second batch
    /// completes (the first is skipped due to CUDA/warmup overhead).
    last_latency: f64,
    /// Total batches the policy has seen complete. Used to skip the
    /// first batch's `on_complete` (warmup latency is unrepresentative).
    batches_completed: usize,
}

impl AdaptiveBatchPolicy {
    pub fn new(max_batch_size: usize, device_idx: usize) -> Self {
        Self {
            max_batch_size,
            device_idx,
            batch_start_time: None,
            last_latency: 0.0,
            batches_completed: 0,
        }
    }
}

impl SchedulingPolicy for AdaptiveBatchPolicy {
    fn on_arrival(&mut self) {
        if self.batch_start_time.is_none() {
            self.batch_start_time = Some(Instant::now());
        }
    }

    fn on_complete(&mut self, latency: Duration) {
        self.batches_completed += 1;
        // Skip the first batch: CUDA graph capture, kernel-cache warmup,
        // and one-shot allocations make its latency unrepresentative of
        // steady-state. After that, just take the latest value — no
        // smoothing, no decay. See header comment for why.
        if self.batches_completed > 1 {
            self.last_latency = latency.as_secs_f64();
        }
    }

    fn on_fired(&mut self) {
        self.batch_start_time = None;
    }

    fn decide(&self, current_batch_size: usize) -> Decision {
        // (1) Structural cap.
        if current_batch_size >= self.max_batch_size {
            return Decision::Fire;
        }
        // (2) Cohort cap: the load-bearing fire condition. Every active
        //     context on this device is in the batch already.
        let active = crate::context::pinned_count(self.device_idx);
        if active > 0 && current_batch_size >= active {
            return Decision::Fire;
        }
        // (3) Cold start: no characteristic time observed yet — make
        //     forward progress rather than waiting on an undefined bound.
        if self.last_latency == 0.0 {
            return Decision::Fire;
        }
        // (4) Absolute wait bound: never wait longer than one previous
        //     batch's compute time. Beyond that, the in-batch peers are
        //     paying more in queue latency than any straggler can save.
        let Some(start) = self.batch_start_time else {
            // Defensive: batch_start_time should always be Some when
            // decide is called (set on first on_arrival). If somehow
            // not, fire rather than getting stuck.
            return Decision::Fire;
        };
        let elapsed = start.elapsed().as_secs_f64();
        if elapsed >= self.last_latency {
            return Decision::Fire;
        }
        // Otherwise, keep waiting. The scheduler's tokio::select wakes
        // on the next arrival or the timeout, whichever comes first.
        Decision::Wait(Duration::from_secs_f64(self.last_latency - elapsed))
    }
}

// =============================================================================
// GreedyPolicy — fire immediately. Zero state.
// =============================================================================
//
// Retained as a reference baseline. Useful when:
//   - Debugging the scheduler itself: any non-trivial policy bug looks
//     like a scheduler bug. Swap in greedy and see if the symptom
//     persists; if not, the policy is the cause.
//   - Workloads where you genuinely want zero coalescing (RPS=1, latency
//     above all else, or driving a single interactive session).
//
// At any RPS > 1, greedy produces small B=1–2 batches and pays per-batch
// fixed overhead on every token — measured p50 stays at ~1100–1250 ms
// regardless of load (vs 350–800 ms for adaptive across the same range).
// Throughput at saturation suffers correspondingly (~6% below adaptive
// at RPS=64 in our matrix).

pub(super) struct GreedyPolicy;

impl GreedyPolicy {
    pub fn new() -> Self {
        Self
    }
}

impl SchedulingPolicy for GreedyPolicy {
    fn on_arrival(&mut self) {}
    fn on_complete(&mut self, _latency: Duration) {}
    fn on_fired(&mut self) {}

    fn decide(&self, _current_batch_size: usize) -> Decision {
        // The scheduler only calls `decide` when the batch is non-empty,
        // and the BatchAccumulator already enforces `max_batch_size`
        // upstream of the policy. So: just fire, every time.
        Decision::Fire
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- AdaptiveBatchPolicy ------------------------------------------------

    #[test]
    fn cold_start_fires() {
        let policy = AdaptiveBatchPolicy::new(512, 0);
        // No batches completed yet — last_latency = 0, no characteristic
        // time. Fire to make progress.
        assert!(matches!(policy.decide(1), Decision::Fire));
    }

    #[test]
    fn fires_at_max_batch_size() {
        let policy = AdaptiveBatchPolicy::new(512, 0);
        assert!(matches!(policy.decide(512), Decision::Fire));
    }

    #[test]
    fn skips_first_batch_in_latency_update() {
        let mut policy = AdaptiveBatchPolicy::new(512, 0);
        policy.on_complete(Duration::from_millis(500));
        assert_eq!(
            policy.last_latency, 0.0,
            "first batch is skipped (CUDA/warmup overhead inflates it)"
        );
    }

    #[test]
    fn last_latency_tracks_most_recent_completed_batch() {
        let mut policy = AdaptiveBatchPolicy::new(512, 0);
        policy.on_complete(Duration::from_millis(500)); // skipped (first)
        policy.on_complete(Duration::from_millis(30));
        assert!((policy.last_latency - 0.030).abs() < 1e-6);
        // No smoothing: a smaller subsequent batch overwrites.
        policy.on_complete(Duration::from_millis(5));
        assert!((policy.last_latency - 0.005).abs() < 1e-6);
    }

    // ---- GreedyPolicy -------------------------------------------------------

    #[test]
    fn greedy_always_fires() {
        let policy = GreedyPolicy::new();
        assert!(matches!(policy.decide(1), Decision::Fire));
        assert!(matches!(policy.decide(100), Decision::Fire));
        assert!(matches!(policy.decide(512), Decision::Fire));
    }
}
