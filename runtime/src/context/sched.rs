//! Scheduling — DAG-Aware Invested-Importance Scheduling.
//!
//! Decides which process should yield pages when GPU memory is under pressure.
//! Uses an invested importance model:
//!
//!   π_d = w_i · p_d
//!
//! where `w_i` is the SRPT-derived process weight (`total_steps / remaining_steps`)
//! and `p_d` is the page count for that process on device `d`.
//! Higher invested importance = harder to evict.
//!
//! ## Eviction Condition
//!
//! Evict victim V for requester R iff:
//!   π_V < π_R(p_R + n) = w_R · (p_R + n)
//!
//! This is the post-allocation floor: only evict if the requester's
//! invested importance AFTER allocation exceeds the victim's current.

use std::collections::HashMap;
use std::time::Instant;

use crate::device::DeviceId;
use crate::process::ProcessId;

use super::{ContextId, ContextManager};
use super::suspend::AllocWaiter;

// =============================================================================
// DevicePages (per-device accounting)
// =============================================================================

/// Per-device page accounting for a process.
#[derive(Debug, Clone)]
pub(crate) struct DevicePages {
    pub committed: usize,
    pub working: usize,
}

impl DevicePages {
    fn new() -> Self {
        DevicePages { committed: 0, working: 0 }
    }

    pub fn total(&self) -> usize {
        self.committed + self.working
    }
}

// =============================================================================
// ProcessEntry
// =============================================================================

/// Per-process scheduling and ownership state.
#[derive(Debug)]
pub(crate) struct ProcessEntry {
    /// SRPT weight: `total_steps / remaining_steps` for the owning workflow.
    /// Higher = workflow closer to completion = harder to evict. Default 1.0.
    pub weight: f64,
    /// Per-device page accounting.
    pub devices: HashMap<DeviceId, DevicePages>,
    /// Birth timestamp — used as FCFS tiebreaker at equal invested importance.
    pub created_at: Instant,
    /// Process scheduling state (Running vs Pending).
    pub state: ProcessState,
    /// Context IDs owned by this process.
    pub context_ids: Vec<ContextId>,
    /// Number of Pinned contexts still awaiting clear_pinned.
    /// Restore is blocked until this reaches 0.
    pub pending_pinned: usize,
    /// Pending alloc requests accumulated while Pending.
    /// Replayed after restoration completes.
    pub pending_allocs: Vec<AllocWaiter>,
}

impl ProcessEntry {
    fn new() -> Self {
        ProcessEntry {
            weight: 1.0,
            devices: HashMap::new(),
            created_at: Instant::now(),
            state: ProcessState::Running,
            context_ids: Vec::new(),
            pending_pinned: 0,
            pending_allocs: Vec::new(),
        }
    }

    pub fn device_mut(&mut self, dev: DeviceId) -> &mut DevicePages {
        self.devices.entry(dev).or_insert_with(DevicePages::new)
    }

    pub fn pages_on(&self, dev: DeviceId) -> usize {
        self.devices.get(&dev).map(|d| d.total()).unwrap_or(0)
    }

    /// π_d = w_i · p_d
    pub fn priority_on(&self, dev: DeviceId) -> f64 {
        self.weight * self.pages_on(dev) as f64
    }

    /// Invested importance at an arbitrary page count on a device.
    pub fn priority_at(&self, p: usize) -> f64 {
        self.weight * p as f64
    }

    pub fn total_pages(&self) -> usize {
        self.devices.values().map(|d| d.total()).sum()
    }

    /// Zero out all pages for this process on a specific device (bulk suspension).
    pub fn zero_device(&mut self, dev: DeviceId) {
        if let Some(d) = self.devices.get_mut(&dev) {
            d.committed = 0;
            d.working = 0;
        }
    }

    /// Set exact page counts for a device (used by restore).
    pub fn set_device(&mut self, dev: DeviceId, committed: usize, working: usize) {
        let d = self.device_mut(dev);
        d.committed = committed;
        d.working = working;
    }
}

// =============================================================================
// Process State
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProcessState {
    Running,
    Pending,
}

// =============================================================================
// Scheduling methods on ContextManager
// =============================================================================

impl ContextManager {

    /// Get or register a process, returning mutable reference to its entry.
    pub(crate) fn process_entry(&mut self, pid: ProcessId) -> &mut ProcessEntry {
        self.processes.entry(pid).or_insert_with(ProcessEntry::new)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::pagestore::PageStore;

    use uuid::Uuid;

    fn pid(n: u64) -> ProcessId { Uuid::from_u128(n as u128) }
    const DEV0: DeviceId = 0;
    const DEV1: DeviceId = 1;

    /// Create a minimal ContextManager for scheduling tests.
    fn test_mgr() -> ContextManager {
        ContextManager::new(0, 16, &[64], &[64])
    }

    #[test]
    fn entry_created_on_add_working() {
        let mut mgr = test_mgr();
        mgr.process_entry(pid(1)).device_mut(DEV0).working += 5;
        assert_eq!(mgr.processes.len(), 1);
    }

    #[test]
    fn per_device_isolation() {
        let mut mgr = test_mgr();
        mgr.process_entry(pid(1)).device_mut(DEV0).working += 50;
        mgr.process_entry(pid(1)).device_mut(DEV1).working += 10;

        let proc = mgr.processes.get(&pid(1)).unwrap();
        let p0 = proc.priority_on(DEV0);
        let p1 = proc.priority_on(DEV1);
        assert!(p0 > p1, "device 0 (50p) should have higher priority than device 1 (10p): {p0} > {p1}");

        assert_eq!(proc.pages_on(DEV0), 50);
        assert_eq!(proc.pages_on(DEV1), 10);
    }

    #[test]
    fn priority_increases_with_more_pages() {
        let mut mgr = test_mgr();

        mgr.process_entry(pid(1)).device_mut(DEV0).working += 5;
        let p5 = mgr.processes.get(&pid(1)).unwrap().priority_on(DEV0);

        mgr.process_entry(pid(1)).device_mut(DEV0).working += 15;
        let p20 = mgr.processes.get(&pid(1)).unwrap().priority_on(DEV0);

        assert!(p20 > p5, "more pages = higher invested importance: p20={p20} > p5={p5}");
    }

    #[test]
    fn higher_weight_means_higher_priority() {
        let mut mgr = test_mgr();
        mgr.process_entry(pid(1)).device_mut(DEV0).working += 10;
        mgr.process_entry(pid(2)).device_mut(DEV0).working += 10;

        mgr.process_entry(pid(1)).weight = 5.0;
        mgr.process_entry(pid(2)).weight = 1.0;

        let p1 = mgr.processes.get(&pid(1)).unwrap().priority_on(DEV0);
        let p2 = mgr.processes.get(&pid(2)).unwrap().priority_on(DEV0);
        assert!(p1 > p2, "higher weight should mean higher priority: {p1} > {p2}");
    }

    #[test]
    fn commit_moves_working_to_committed() {
        let mut mgr = test_mgr();
        mgr.process_entry(pid(1)).device_mut(DEV0).working += 10;
        assert_eq!(mgr.processes.get(&pid(1)).unwrap().pages_on(DEV0), 10);

        {
            let d = mgr.process_entry(pid(1)).device_mut(DEV0);
            d.committed += 5;
            d.working -= 5;
        }
        assert_eq!(mgr.processes.get(&pid(1)).unwrap().pages_on(DEV0), 10); // total unchanged
    }

    #[test]
    fn zero_device_clears_pages() {
        let mut mgr = test_mgr();
        {
            let d = mgr.process_entry(pid(1)).device_mut(DEV0);
            d.working += 5;
            d.committed += 5;
            d.working -= 5;
            d.working += 3;
        }
        mgr.process_entry(pid(1)).device_mut(DEV1).working += 7;

        let proc = mgr.processes.get(&pid(1)).unwrap();
        assert_eq!(proc.pages_on(DEV0), 8);
        assert_eq!(proc.pages_on(DEV1), 7);

        mgr.process_entry(pid(1)).zero_device(DEV0);
        let proc = mgr.processes.get(&pid(1)).unwrap();
        assert_eq!(proc.pages_on(DEV0), 0);
        assert_eq!(proc.pages_on(DEV1), 7); // unaffected
    }

    #[test]
    fn bigger_node_has_higher_priority() {
        let mut mgr = test_mgr();
        mgr.process_entry(pid(1)).device_mut(DEV0).working += 100;
        mgr.process_entry(pid(2)).device_mut(DEV0).working += 10;

        let p1 = mgr.processes.get(&pid(1)).unwrap().priority_on(DEV0);
        let p2 = mgr.processes.get(&pid(2)).unwrap().priority_on(DEV0);
        assert!(p1 > p2, "bigger node should have higher priority: {p1} > {p2}");
    }

    #[test]
    fn cross_device_pages_dont_affect_priority() {
        let mut mgr = test_mgr();
        mgr.process_entry(pid(1)).device_mut(DEV0).working += 30;
        mgr.process_entry(pid(1)).device_mut(DEV1).working += 40;
        mgr.process_entry(pid(2)).device_mut(DEV0).working += 30;

        let p1 = mgr.processes.get(&pid(1)).unwrap().priority_on(DEV0);
        let p2 = mgr.processes.get(&pid(2)).unwrap().priority_on(DEV0);
        assert!((p1 - p2).abs() < 1e-10,
            "same device pages should yield same priority: {p1} vs {p2}");
    }

    /// Proves the anti-thrashing property of invested importance.
    #[test]
    fn post_allocation_floor_prevents_thrashing() {
        let mut mgr = test_mgr();
        mgr.process_entry(pid(1)).device_mut(DEV0).working += 30; // A holds 30 (won eviction)

        let b_current = mgr.processes.get(&pid(2)).map(|e| e.pages_on(DEV0)).unwrap_or(0); // = 0
        let b_weight = mgr.processes.get(&pid(2)).map(|e| e.weight).unwrap_or(1.0);
        let b_floor = b_weight * (b_current + 30) as f64;
        let a_priority = mgr.processes.get(&pid(1)).unwrap().priority_on(DEV0);

        assert!(a_priority >= b_floor,
            "A ({a_priority}) should NOT be evictable by B (floor={b_floor})");
        assert!((a_priority - b_floor).abs() < 1e-10);
    }

    #[test]
    fn set_device_round_trip() {
        let mut mgr = test_mgr();
        {
            let d = mgr.process_entry(pid(1)).device_mut(DEV0);
            d.working += 5;
            d.committed += 5;
            d.working -= 5;
            d.working += 3;
        }

        let pre_pages = mgr.processes.get(&pid(1)).unwrap().pages_on(DEV0);
        assert_eq!(pre_pages, 8);

        mgr.process_entry(pid(1)).zero_device(DEV0);
        assert_eq!(mgr.processes.get(&pid(1)).unwrap().pages_on(DEV0), 0);

        mgr.process_entry(pid(1)).set_device(DEV0, 5, 3);
        assert_eq!(mgr.processes.get(&pid(1)).unwrap().pages_on(DEV0), pre_pages);
    }

    #[test]
    fn zero_then_partial_restore_then_commit() {
        let mut mgr = test_mgr();
        {
            let d = mgr.process_entry(pid(1)).device_mut(DEV0);
            d.working += 10;
            d.committed += 10;
            d.working -= 10;
            d.working += 3;
        }

        let pre_pages = mgr.processes.get(&pid(1)).unwrap().pages_on(DEV0);
        assert_eq!(pre_pages, 13);

        mgr.process_entry(pid(1)).zero_device(DEV0);
        assert_eq!(mgr.processes.get(&pid(1)).unwrap().pages_on(DEV0), 0);

        // Restore working pages (swap-in)
        mgr.process_entry(pid(1)).device_mut(DEV0).working += 3;

        // Restore only prefix-matched committed (6 of 10 matched)
        mgr.process_entry(pid(1)).set_device(DEV0, 6, 3);

        // Replay remaining 4 pages: add as working, then commit
        {
            let d = mgr.process_entry(pid(1)).device_mut(DEV0);
            d.working += 4;
            d.committed += 4;
            d.working -= 4;
        }

        assert_eq!(mgr.processes.get(&pid(1)).unwrap().pages_on(DEV0), pre_pages,
            "partial restore + replay commit must not double-count");
    }
}
