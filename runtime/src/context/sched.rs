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
    /// Number of contexts still being replayed (awaiting FinishRestore).
    /// Deferred op fires when this reaches 0.
    pub pending_replay_count: usize,
    /// Deferred operations held while Pending.
    /// Multiple may be active if the WASM guest has concurrent tasks.
    /// All are replayed after restoration completes.
    pub deferred_ops: Vec<super::suspend::PendingAlloc>,
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
            pending_replay_count: 0,
            deferred_ops: Vec::new(),
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
