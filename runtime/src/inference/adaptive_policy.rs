//! Adaptive throughput scheduling policy.
//!
//! **Principle**: fire when `idle > L_ref / B`, where L_ref is a
//! leaky max of observed batch latencies.
//!
//! `ref_l = max(ref_l × decay, latency)` — a max-ratchet with a
//! slow leak. L_ref naturally tracks the heaviest workload (instantly
//! ramps up on large batches) while slowly decaying stale outliers.
//!
//! This avoids the **death spiral** (where small batches → low L →
//! tight budgets → even smaller batches) because small batches can't
//! actively decrease L_ref — it only decays passively by the decay
//! factor. A single large batch fully restores it.
//!
//! The `/B` denominator naturally scales the budget:
//! - Small B → generous budget → waits to accumulate more.
//! - Large B → tiny budget → fires quickly.

use std::time::{Duration, Instant};

use super::scheduler::{BatchStats, Decision, SchedulingPolicy};

/// Leaky-max batch scheduling policy.
///
/// Tracks a slowly-decaying maximum of observed batch latencies.
/// Fire when idle time exceeds `ref_l / batch_size`.
pub(super) struct AdaptiveThroughputPolicy {
    max_batch_size: usize,
    max_wait_time: Duration,
    batch_start_time: Option<Instant>,
    last_arrival_time: Option<Instant>,
    /// Leaky max of observed batch latency (seconds).
    ref_l: f64,
    /// Batch counter (to skip first batch's warmup overhead).
    batches_completed: usize,
}

/// Per-batch decay factor for ref_l. Half-life ≈ 7000 batches.
const DECAY: f64 = 0.9999;

impl AdaptiveThroughputPolicy {
    pub fn new(
        max_batch_size: usize,
        max_wait_time: Duration,
        _min_batch_for_optimization: usize,
    ) -> Self {
        Self {
            max_batch_size,
            max_wait_time,
            batch_start_time: None,
            last_arrival_time: None,
            ref_l: 0.0,
            batches_completed: 0,
        }
    }
}

impl SchedulingPolicy for AdaptiveThroughputPolicy {
    fn on_arrival(&mut self, _arrival_time: Instant) {
        self.last_arrival_time = Some(Instant::now());
        if self.batch_start_time.is_none() {
            self.batch_start_time = Some(Instant::now());
        }
    }

    fn on_complete(&mut self, stats: &BatchStats) {
        self.batches_completed += 1;
        let latency = stats.latency.as_secs_f64();

        // Skip first batch (CUDA/warmup overhead inflates latency).
        // Leaky max: decays slowly, refreshed by any new peak.
        if self.batches_completed > 1 {
            self.ref_l = (self.ref_l * DECAY).max(latency);
        }
    }

    fn on_fired(&mut self) {
        self.batch_start_time = None;
        self.last_arrival_time = None;
    }

    fn decide(
        &self,
        current_batch_size: usize,
        _current_total_tokens: usize,
    ) -> Decision {
        if current_batch_size >= self.max_batch_size {
            return Decision::Fire;
        }

        // Cold start: no latency data yet.
        if self.ref_l == 0.0 {
            return Decision::Fire;
        }

        let Some(start) = self.batch_start_time else {
            return Decision::Fire;
        };

        // Fire when idle > ref_l / B.
        let budget = self.ref_l / current_batch_size.max(1) as f64;

        let idle = self.last_arrival_time
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or(f64::MAX);

        if idle > budget {
            return Decision::Fire;
        }

        if start.elapsed() >= self.max_wait_time {
            return Decision::Fire;
        }

        Decision::Wait(Duration::from_secs_f64(budget - idle))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fires_during_cold_start() {
        let policy = AdaptiveThroughputPolicy::new(512, Duration::from_millis(50), 1);
        assert!(matches!(policy.decide(10, 10), Decision::Fire));
    }

    #[test]
    fn fires_at_max_batch_size() {
        let policy = AdaptiveThroughputPolicy::new(512, Duration::from_millis(50), 1);
        assert!(matches!(policy.decide(512, 512), Decision::Fire));
    }

    #[test]
    fn skips_warmup_batch() {
        let mut policy = AdaptiveThroughputPolicy::new(512, Duration::from_millis(50), 1);
        policy.on_complete(&BatchStats {
            batch_size: 1,
            total_tokens: 1,
            latency: Duration::from_millis(500),
        });
        assert_eq!(policy.ref_l, 0.0, "first batch should be skipped");
    }

    #[test]
    fn ref_l_tracks_peak_and_decays() {
        let mut policy = AdaptiveThroughputPolicy::new(512, Duration::from_millis(50), 1);
        // Skip warmup.
        policy.on_complete(&BatchStats { batch_size: 1, total_tokens: 1, latency: Duration::from_millis(500) });

        // Large batch sets ref_l.
        policy.on_complete(&BatchStats { batch_size: 300, total_tokens: 300, latency: Duration::from_millis(80) });
        assert!((policy.ref_l - 0.080).abs() < 0.001);

        // Small batch: ref_l decays but doesn't drop to small batch's latency.
        policy.on_complete(&BatchStats { batch_size: 5, total_tokens: 5, latency: Duration::from_millis(3) });
        assert!(policy.ref_l > 0.079, "should only decay by DECAY factor, got {:.4}", policy.ref_l);

        // New peak refreshes ref_l fully.
        policy.on_complete(&BatchStats { batch_size: 400, total_tokens: 400, latency: Duration::from_millis(100) });
        assert!((policy.ref_l - 0.100).abs() < 0.001);
    }
}
