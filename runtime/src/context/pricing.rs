//! Pricing — Scarcity-Based Market Prices.
//!
//! Computes per-device prices for holding KV cache pages (`keep`) and
//! producing tokens (`make`). Prices are zero when a resource has spare
//! capacity and rise convexly as utilization approaches 1.
//!
//! ## Price Curves
//!
//! - **`keep_price(λ)`**: Cost per page per step to hold cache.
//!   `ρ(λ) = λ² / (1 − λ)`. Zero at λ=0, divergent near λ=1.
//!
//! - **`make_price(μ, c_min)`**: Cost to produce one token of KV cache.
//!   `c(μ) = c_min + μ² / (1 − μ)`. Floor at `c_min` when idle.
//!
//! ## Rent and Interest
//!
//! Each tick:
//! 1. Compute per-device prices from utilization.
//! 2. Deduct rent from each process: `Σ_i keep_i × adjusted_pages_i`.
//! 3. Pool total rent; redistribute as interest proportional to idle credits.
//!
//! Credits are conserved: `Σ_j balance_j + reserve = E` for all t.

use std::collections::HashMap;

use crate::process::ProcessId;

use super::pagestore::PageStore;
use super::sched::ProcessEntry;

// =============================================================================
// Price Curves
// =============================================================================

/// Convex scarcity curve for cache holding cost.
/// Zero at zero load, diverges near full utilization.
pub fn keep_price(lambda: f64) -> f64 {
    if lambda <= 0.0 {
        return 0.0;
    }
    lambda * lambda / (1.0 - lambda).max(1e-6)
}

/// Convex scarcity curve for token production cost.
/// Floor at `c_min` when idle, diverges near full utilization.
pub fn make_price(mu: f64, c_min: f64) -> f64 {
    if mu <= 0.0 {
        return c_min;
    }
    c_min + mu * mu / (1.0 - mu).max(1e-6)
}

// =============================================================================
// Market State
// =============================================================================

/// Per-device market prices, computed each tick.
#[derive(Debug, Clone)]
pub(crate) struct DevicePrice {
    /// Cost per page per step to hold cache on GPU tier.
    pub keep_gpu: f64,
    /// Cost per page per step to hold cache on CPU tier.
    pub keep_cpu: f64,
    /// Cost to produce one token of KV cache on this device.
    pub make: f64,
}

impl Default for DevicePrice {
    fn default() -> Self {
        DevicePrice {
            keep_gpu: 0.0,
            keep_cpu: 0.0,
            make: 0.0,
        }
    }
}

/// Global market state, held actor-locally on ContextManager.
#[derive(Debug, Clone)]
pub(crate) struct MarketState {
    /// Per-device prices (indexed by device ordinal).
    pub device_prices: Vec<DevicePrice>,
    /// Interest rate: total_rent_pool / total_idle_credits.
    pub interest_rate: f64,
    /// Rent collected this tick (for diagnostics).
    pub total_rent_pool: f64,
}

impl MarketState {
    pub fn new(num_devices: usize) -> Self {
        MarketState {
            device_prices: vec![DevicePrice::default(); num_devices],
            interest_rate: 0.0,
            total_rent_pool: 0.0,
        }
    }
}

// =============================================================================
// Tick — Rent Collection and Interest Distribution
// =============================================================================

/// Minimum compute cost floor (even at zero utilization, producing
/// tokens has a nonzero cost so programs value existing cache).
const C_MIN: f64 = 0.01;

/// CPU tier discount factor relative to GPU tier.
/// CPU pages are cheaper to hold because they don't consume GPU memory.
const CPU_TIER_DISCOUNT: f64 = 0.1;

/// Execute one market tick: compute prices, collect rent, distribute interest.
///
/// Called once per batch step. Mutates process balances in-place.
pub(crate) fn tick(
    devices: &[PageStore],
    processes: &mut HashMap<ProcessId, ProcessEntry>,
    market: &mut MarketState,
) {
    let num_devices = devices.len();

    // Phase 1: Compute per-device prices from utilization.
    for (i, dev) in devices.iter().enumerate() {
        let (used, total) = dev.stats();
        let lambda = if total > 0 { used as f64 / total as f64 } else { 0.0 };
        // For now, compute utilization is not tracked per-device;
        // use a placeholder (0.0 = idle pricing). Will be wired to
        // actual batch utilization later.
        let mu = 0.0;

        let price = if i < market.device_prices.len() {
            &mut market.device_prices[i]
        } else {
            market.device_prices.resize(num_devices, DevicePrice::default());
            &mut market.device_prices[i]
        };
        price.keep_gpu = keep_price(lambda);
        price.keep_cpu = keep_price(lambda) * CPU_TIER_DISCOUNT;
        price.make = make_price(mu, C_MIN);
    }

    // Phase 2: Collect rent from each process.
    let mut total_rent = 0.0f64;
    let mut total_idle_credits = 0.0f64;

    for proc in processes.values_mut() {
        let mut proc_rent = 0.0f64;
        for (&dev_id, dev_pages) in &proc.devices {
            if dev_id < market.device_prices.len() {
                let price = &market.device_prices[dev_id];
                // All committed + working pages on GPU pay GPU keep rate.
                proc_rent += price.keep_gpu * dev_pages.total() as f64;
            }
        }

        // Deduct rent (balance cannot go below zero — see no-bankruptcy proof).
        let actual_rent = proc_rent.min(proc.balance);
        proc.balance -= actual_rent;
        total_rent += actual_rent;
    }

    // Phase 3: Distribute interest proportional to idle credits.
    // Idle credits = balance not currently "invested" in pages.
    for proc in processes.values() {
        let invested = proc.devices.values()
            .map(|d| d.total() as f64)
            .sum::<f64>();
        let idle = (proc.balance - invested).max(0.0);
        total_idle_credits += idle;
    }

    if total_idle_credits > 0.0 && total_rent > 0.0 {
        let rate = total_rent / total_idle_credits;
        for proc in processes.values_mut() {
            let invested = proc.devices.values()
                .map(|d| d.total() as f64)
                .sum::<f64>();
            let idle = (proc.balance - invested).max(0.0);
            proc.balance += rate * idle;
        }
        market.interest_rate = rate;
    } else {
        market.interest_rate = 0.0;
    }

    market.total_rent_pool = total_rent;
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keep_price_zero_at_zero_load() {
        assert_eq!(keep_price(0.0), 0.0);
    }

    #[test]
    fn keep_price_positive_at_half_load() {
        assert!(keep_price(0.5) > 0.0);
    }

    #[test]
    fn keep_price_monotonically_increasing() {
        let mut prev = keep_price(0.0);
        for i in 1..100 {
            let lambda = i as f64 / 100.0;
            let current = keep_price(lambda);
            assert!(current >= prev, "keep_price not monotonic at λ={lambda}: {prev} -> {current}");
            prev = current;
        }
    }

    #[test]
    fn keep_price_diverges_near_one() {
        let at_half = keep_price(0.5);
        let at_99 = keep_price(0.99);
        assert!(at_99 > 100.0 * at_half,
            "Expected divergence: keep(0.99)={at_99} should be >100× keep(0.5)={at_half}");
    }

    #[test]
    fn make_price_floor_at_c_min() {
        let c_min = 0.05;
        assert_eq!(make_price(0.0, c_min), c_min);
    }

    #[test]
    fn make_price_increases_with_utilization() {
        let c_min = 0.01;
        let at_zero = make_price(0.0, c_min);
        let at_half = make_price(0.5, c_min);
        let at_90 = make_price(0.9, c_min);
        assert!(at_half > at_zero);
        assert!(at_90 > at_half);
    }

    #[test]
    fn keep_price_negative_input() {
        assert_eq!(keep_price(-0.5), 0.0);
    }

    #[test]
    fn make_price_negative_input() {
        assert_eq!(make_price(-0.5, 0.01), 0.01);
    }
}
