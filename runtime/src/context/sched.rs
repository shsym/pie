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

/// Per-process scheduling state. Page accounting is per-device.
#[derive(Debug, Clone)]
pub(crate) struct ProcessEntry {
    /// SRPT weight: `total_steps / remaining_steps` for the owning workflow.
    /// Higher = workflow closer to completion = harder to evict. Default 1.0.
    pub weight: f64,
    /// Per-device page accounting.
    pub devices: HashMap<DeviceId, DevicePages>,
    /// Birth timestamp — used as FCFS tiebreaker at equal invested importance.
    pub created_at: Instant,
}

impl ProcessEntry {
    fn new() -> Self {
        ProcessEntry {
            weight: 1.0,
            devices: HashMap::new(),
            created_at: Instant::now(),
        }
    }

    fn device_mut(&mut self, dev: DeviceId) -> &mut DevicePages {
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
}

// =============================================================================
// Process State (scheduling metadata)
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProcessState {
    Running,
    Pending,
}

#[derive(Debug)]
pub(crate) struct ProcessInfo {
    pub state: ProcessState,
    pub context_ids: Vec<ContextId>,
}

// =============================================================================
// Scheduling methods on ContextManager
// =============================================================================

impl ContextManager {

    // ==================== Entry Lifecycle ====================

    pub(crate) fn sched_remove_entry(&mut self, pid: &ProcessId) {
        self.sched_entries.remove(pid);
    }

    pub(crate) fn sched_entry_is_empty(&self, pid: &ProcessId) -> bool {
        self.sched_entries.get(pid).map_or(true, |e| e.total_pages() == 0)
    }

    // ==================== DAG Weights ====================

    pub(crate) fn sched_set_priority(&mut self, pid: ProcessId, weight: f64) {
        let e = self.sched_entry_mut(pid);
        e.weight = weight;
    }

    // ==================== Accounting ====================

    pub(crate) fn sched_add_working(&mut self, pid: ProcessId, device: DeviceId, pages: usize) {
        let e = self.sched_entry_mut(pid);
        let d = e.device_mut(device);
        d.working += pages;
    }

    pub(crate) fn sched_remove_working(&mut self, pid: ProcessId, device: DeviceId, pages: usize) {
        let e = self.sched_entry_mut(pid);
        let d = e.device_mut(device);
        d.working = d.working.saturating_sub(pages);
    }

    pub(crate) fn sched_commit(&mut self, pid: ProcessId, device: DeviceId, pages: usize) {
        let e = self.sched_entry_mut(pid);
        let d = e.device_mut(device);
        d.committed += pages;
        d.working = d.working.saturating_sub(pages);
    }

    pub(crate) fn sched_uncommit(&mut self, pid: ProcessId, device: DeviceId, pages: usize) {
        let e = self.sched_entry_mut(pid);
        let d = e.device_mut(device);
        d.committed = d.committed.saturating_sub(pages);
    }

    /// Zero out all pages for a process on a specific device (bulk suspension).
    pub(crate) fn sched_zero_device(&mut self, pid: ProcessId, device: DeviceId) {
        if let Some(e) = self.sched_entries.get_mut(&pid) {
            if let Some(d) = e.devices.get_mut(&device) {
                d.committed = 0;
                d.working = 0;
            }
        }
    }

    /// Set exact page counts for a process/device (used by restore).
    pub(crate) fn sched_set_device(&mut self, pid: ProcessId, device: DeviceId, committed: usize, working: usize) {
        let e = self.sched_entry_mut(pid);
        let d = e.device_mut(device);
        d.committed = committed;
        d.working = working;
    }

    // ==================== Policy Queries ====================

    /// Invested importance of a process on a specific device: w_i · p_d.
    pub(crate) fn sched_priority(&self, pid: &ProcessId, device: DeviceId) -> f64 {
        self.sched_entries.get(pid).map(|e| e.priority_on(device)).unwrap_or(0.0)
    }

    /// Pages held by a process on a specific device.
    pub(crate) fn sched_pages_on(&self, pid: &ProcessId, device: DeviceId) -> usize {
        self.sched_entries.get(pid).map(|e| e.pages_on(device)).unwrap_or(0)
    }

    /// Invested importance at an arbitrary page count on a device.
    /// For unknown processes, uses default weight (1.0).
    pub(crate) fn sched_priority_at(&self, pid: &ProcessId, pages: usize) -> f64 {
        let weight = self.sched_entries.get(pid).map(|e| e.weight).unwrap_or(1.0);
        weight * pages as f64
    }

    /// Birth timestamp of a process (for FCFS tiebreaking).
    pub(crate) fn sched_created_at(&self, pid: &ProcessId) -> Option<Instant> {
        self.sched_entries.get(pid).map(|e| e.created_at)
    }

    pub(crate) fn sched_entry_count(&self) -> usize {
        self.sched_entries.len()
    }

    /// Get a reference to a process entry.
    pub(crate) fn sched_get(&self, pid: &ProcessId) -> Option<&ProcessEntry> {
        self.sched_entries.get(pid)
    }

    // ==================== Process Management ====================

    /// Get or register a process, returning mutable reference to its info.
    pub(crate) fn ensure_process(&mut self, pid: ProcessId) -> &mut ProcessInfo {
        self.processes.entry(pid).or_insert_with(|| ProcessInfo {
            state: ProcessState::Running,
            context_ids: Vec::new(),
        })
    }

    // ==================== Internal Helpers ====================

    fn sched_entry_mut(&mut self, pid: ProcessId) -> &mut ProcessEntry {
        self.sched_entries.entry(pid).or_insert_with(ProcessEntry::new)
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
        mgr.sched_add_working(pid(1), DEV0, 5);
        assert_eq!(mgr.sched_entry_count(), 1);
    }

    #[test]
    fn per_device_isolation() {
        let mut mgr = test_mgr();
        mgr.sched_add_working(pid(1), DEV0, 50);
        mgr.sched_add_working(pid(1), DEV1, 10);

        let p0 = mgr.sched_priority(&pid(1), DEV0);
        let p1 = mgr.sched_priority(&pid(1), DEV1);
        assert!(p0 > p1, "device 0 (50p) should have higher priority than device 1 (10p): {p0} > {p1}");

        assert_eq!(mgr.sched_pages_on(&pid(1), DEV0), 50);
        assert_eq!(mgr.sched_pages_on(&pid(1), DEV1), 10);
    }

    #[test]
    fn priority_increases_with_more_pages() {
        let mut mgr = test_mgr();

        mgr.sched_add_working(pid(1), DEV0, 5);
        let p5 = mgr.sched_priority(&pid(1), DEV0);

        mgr.sched_add_working(pid(1), DEV0, 15);
        let p20 = mgr.sched_priority(&pid(1), DEV0);

        assert!(p20 > p5, "more pages = higher invested importance: p20={p20} > p5={p5}");
    }

    #[test]
    fn higher_weight_means_higher_priority() {
        let mut mgr = test_mgr();
        mgr.sched_add_working(pid(1), DEV0, 10);
        mgr.sched_add_working(pid(2), DEV0, 10);

        mgr.sched_set_priority(pid(1), 5.0);
        mgr.sched_set_priority(pid(2), 1.0);

        let p1 = mgr.sched_priority(&pid(1), DEV0);
        let p2 = mgr.sched_priority(&pid(2), DEV0);
        assert!(p1 > p2, "higher weight should mean higher priority: {p1} > {p2}");
    }

    #[test]
    fn commit_moves_working_to_committed() {
        let mut mgr = test_mgr();
        mgr.sched_add_working(pid(1), DEV0, 10);
        assert_eq!(mgr.sched_pages_on(&pid(1), DEV0), 10);

        mgr.sched_commit(pid(1), DEV0, 5);
        assert_eq!(mgr.sched_pages_on(&pid(1), DEV0), 10); // total unchanged
    }

    #[test]
    fn zero_device_clears_pages() {
        let mut mgr = test_mgr();
        mgr.sched_add_working(pid(1), DEV0, 5);
        mgr.sched_commit(pid(1), DEV0, 5);
        mgr.sched_add_working(pid(1), DEV0, 3);
        mgr.sched_add_working(pid(1), DEV1, 7);

        assert_eq!(mgr.sched_pages_on(&pid(1), DEV0), 8);
        assert_eq!(mgr.sched_pages_on(&pid(1), DEV1), 7);

        mgr.sched_zero_device(pid(1), DEV0);
        assert_eq!(mgr.sched_pages_on(&pid(1), DEV0), 0);
        assert_eq!(mgr.sched_pages_on(&pid(1), DEV1), 7); // unaffected
    }

    #[test]
    fn bigger_node_has_higher_priority() {
        let mut mgr = test_mgr();
        mgr.sched_add_working(pid(1), DEV0, 100);
        mgr.sched_add_working(pid(2), DEV0, 10);

        let p1 = mgr.sched_priority(&pid(1), DEV0);
        let p2 = mgr.sched_priority(&pid(2), DEV0);
        assert!(p1 > p2, "bigger node should have higher priority: {p1} > {p2}");
    }

    #[test]
    fn cross_device_pages_dont_affect_priority() {
        let mut mgr = test_mgr();
        mgr.sched_add_working(pid(1), DEV0, 30);
        mgr.sched_add_working(pid(1), DEV1, 40);
        mgr.sched_add_working(pid(2), DEV0, 30);

        let p1 = mgr.sched_priority(&pid(1), DEV0);
        let p2 = mgr.sched_priority(&pid(2), DEV0);
        assert!((p1 - p2).abs() < 1e-10,
            "same device pages should yield same priority: {p1} vs {p2}");
    }

    /// Proves the anti-thrashing property of invested importance.
    #[test]
    fn post_allocation_floor_prevents_thrashing() {
        let mut mgr = test_mgr();
        mgr.sched_add_working(pid(1), DEV0, 30); // A holds 30 (won eviction)

        let b_current = mgr.sched_pages_on(&pid(2), DEV0); // = 0
        let b_floor = mgr.sched_priority_at(&pid(2), b_current + 30);
        let a_priority = mgr.sched_priority(&pid(1), DEV0);

        assert!(a_priority >= b_floor,
            "A ({a_priority}) should NOT be evictable by B (floor={b_floor})");
        assert!((a_priority - b_floor).abs() < 1e-10);
    }

    #[test]
    fn set_device_round_trip() {
        let mut mgr = test_mgr();
        mgr.sched_add_working(pid(1), DEV0, 5);
        mgr.sched_commit(pid(1), DEV0, 5);
        mgr.sched_add_working(pid(1), DEV0, 3);

        let pre_pages = mgr.sched_pages_on(&pid(1), DEV0);
        assert_eq!(pre_pages, 8);

        mgr.sched_zero_device(pid(1), DEV0);
        assert_eq!(mgr.sched_pages_on(&pid(1), DEV0), 0);

        mgr.sched_set_device(pid(1), DEV0, 5, 3);
        assert_eq!(mgr.sched_pages_on(&pid(1), DEV0), pre_pages);
    }

    #[test]
    fn zero_then_partial_restore_then_commit() {
        let mut mgr = test_mgr();
        mgr.sched_add_working(pid(1), DEV0, 10);
        mgr.sched_commit(pid(1), DEV0, 10);
        mgr.sched_add_working(pid(1), DEV0, 3);

        let pre_pages = mgr.sched_pages_on(&pid(1), DEV0);
        assert_eq!(pre_pages, 13);

        mgr.sched_zero_device(pid(1), DEV0);
        assert_eq!(mgr.sched_pages_on(&pid(1), DEV0), 0);

        // Restore working pages (swap-in)
        mgr.sched_add_working(pid(1), DEV0, 3);

        // Restore only prefix-matched committed (6 of 10 matched)
        mgr.sched_set_device(pid(1), DEV0, 6, 3);

        // Replay remaining 4 pages: add as working, then commit
        mgr.sched_add_working(pid(1), DEV0, 4);
        mgr.sched_commit(pid(1), DEV0, 4);

        assert_eq!(mgr.sched_pages_on(&pid(1), DEV0), pre_pages,
            "partial restore + replay commit must not double-count");
    }
}
