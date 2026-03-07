use std::time::Instant;

use anyhow::Result;

use super::{
    Context, ContextId, ContextManager, ContextState,
};
use crate::device::{self, DeviceId};

// =============================================================================
// Persistence methods on ContextManager
// =============================================================================

impl ContextManager {

    pub(crate) fn open(&mut self, username: String, name: String) -> Result<ContextId> {
        match self.snapshots.get(&(username, name)) {
            Some(&snapshot_id) => self.fork(snapshot_id),
            None => Err(anyhow::anyhow!("Snapshot not found")),
        }
    }

    /// Save/snapshot a context. If `name` is None, auto-generates a snapshot name.
    /// Returns the name used (Some only when auto-generated).
    pub(crate) fn save(&mut self, id: ContextId, username: String, name: Option<String>) -> Result<Option<String>> {
        let (name, auto_generated) = match name {
            Some(n) => (n, false),
            None => (format!("__snapshot_{}", self.next_id()), true),
        };

        if self.snapshots.contains_key(&(username.clone(), name.clone())) {
            anyhow::bail!("Snapshot name already exists: {}", name);
        }

        let ctx = self.contexts.get(&id).ok_or_else(|| anyhow::anyhow!("Context not found"))?;
        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let tip = ctx.committed_tip;
        let lineage = ctx.lineage.clone();
        let src_working = ctx.working_pages.clone();

        let committed_len = ctx.committed_len;
        let max_pos = ctx.max_committed_position;
        let snapshot_filled = ctx.working_page_tokens.clone();

        if let Some(tip_hash) = tip {
            self.devices[dev_idx].acquire_chain(tip_hash);
        }

        // Snapshot working pages: try GPU-first, fall back to CPU swap pool.
        let (snapshot_working, snapshot_state) = if !src_working.is_empty() {
            let n = src_working.len();
            if let Some(dst_pages) = self.devices[dev_idx].alloc_gpu_pages(n) {
                // GPU → GPU copy
                let _ = device::copy_d2d(dev_idx as DeviceId, &src_working, &dst_pages);
                (dst_pages, ContextState::Active)
            } else if let Some(cpu_pages) = self.devices[dev_idx].alloc_cpu_pages(n) {
                // Fallback: GPU → CPU copy (source GPU pages stay intact)
                let _ = device::copy_d2h(dev_idx as DeviceId, &src_working, &cpu_pages);
                (cpu_pages, ContextState::Suspended)
            } else {
                eprintln!("SNAPSHOT_PAGE_COPY_FAIL ctx={id}: no GPU or CPU pages available");
                (Vec::new(), ContextState::Active)
            }
        } else {
            (Vec::new(), ContextState::Active)
        };

        let snapshot_id = self.next_id();
        self.contexts.insert(snapshot_id, Context {
            owner: None,
            device: Some(dev_idx as DeviceId),
            working_pages: snapshot_working,
            committed_tip: tip,
            lineage,
            working_page_tokens: snapshot_filled,
            committed_len,
            max_committed_position: max_pos,
            state: snapshot_state,
            pending_suspend: false,
            last_access: Instant::now(),
        });
        self.snapshots.insert((username, name.clone()), snapshot_id);
        Ok(if auto_generated { Some(name) } else { None })
    }

    pub(crate) fn delete(&mut self, username: String, name: String) -> Result<()> {
        let snapshot_id = self.snapshots.remove(&(username, name))
            .ok_or_else(|| anyhow::anyhow!("Snapshot not found"))?;

        if let Some(ctx) = self.contexts.remove(&snapshot_id) {
            let dev_idx = ctx.device.unwrap_or(0) as usize;
            if let Some(tip_hash) = ctx.committed_tip {
                self.devices[dev_idx].release_chain(tip_hash);
                self.devices[dev_idx].remove_index_cache(tip_hash);
            }
            // Free snapshot working pages (GPU or CPU depending on state)
            if ctx.is_suspended() {
                self.devices[dev_idx].free_cpu_pages(&ctx.working_pages);
            } else {
                self.devices[dev_idx].free_gpu_pages(&ctx.working_pages);
            }
        }

        Ok(())
    }

    /// Take ownership of a saved snapshot. The snapshot is deleted and its
    /// resources are transferred to the new context. GPU working pages are
    /// moved directly (no D2D copy needed). CPU working pages are restored
    /// via H2D copy. Committed pages are ref-bumped (shared via CAS).
    pub(crate) fn take(&mut self, username: String, name: String) -> Result<ContextId> {
        let key = (username, name);
        let snapshot_id = *self.snapshots.get(&key)
            .ok_or_else(|| anyhow::anyhow!("Snapshot not found"))?;

        let snap = self.contexts.remove(&snapshot_id)
            .ok_or_else(|| anyhow::anyhow!("Snapshot context missing"))?;

        self.snapshots.remove(&key);

        let dev_idx = snap.device.unwrap_or(0) as usize;

        // Committed chain: the snapshot held one acquire_chain reference.
        // By consuming the snapshot (removing it from contexts), we transfer
        // that reference to the new context — no extra acquire/release needed.

        // Working pages: GPU pages transfer directly, CPU pages need H2D copy
        let (new_working, new_state) = if snap.working_pages.is_empty() {
            (Vec::new(), ContextState::Active)
        } else if snap.is_suspended() {
            // CPU → GPU: allocate GPU pages and copy
            let n = snap.working_pages.len();
            if let Some(gpu_pages) = self.devices[dev_idx].alloc_gpu_pages(n) {
                let _ = device::copy_h2d(dev_idx as DeviceId, &gpu_pages, &snap.working_pages);
                // Free CPU pages after H2D copy is issued
                self.devices[dev_idx].free_cpu_pages(&snap.working_pages);
                (gpu_pages, ContextState::Active)
            } else {
                // GPU OOM — can't restore, free the orphaned CPU pages
                self.devices[dev_idx].free_cpu_pages(&snap.working_pages);
                (Vec::new(), ContextState::Active)
            }
        } else {
            // GPU → new context: direct ownership transfer (zero-copy)
            (snap.working_pages.clone(), ContextState::Active)
        };

        let new_id = self.next_id();
        self.contexts.insert(new_id, Context {
            owner: None,
            device: Some(dev_idx as DeviceId),
            working_pages: new_working,
            committed_tip: snap.committed_tip,
            lineage: snap.lineage,
            working_page_tokens: snap.working_page_tokens,
            committed_len: snap.committed_len,
            max_committed_position: snap.max_committed_position,
            state: new_state,
            pending_suspend: false,
            last_access: Instant::now(),
        });

        Ok(new_id)
    }
}
