use std::time::Instant;

use anyhow::Result;

use super::{
    Context, ContextId, ContextManager, ContextState,
};
use crate::device::{self, DeviceId};
use crate::process::ProcessId;

// =============================================================================
// Persistence methods on ContextManager
// =============================================================================

impl ContextManager {

    pub(crate) fn open(&mut self, username: String, name: String, owner: ProcessId) -> Result<ContextId> {
        match self.snapshots.get(&(username, name)) {
            Some(&snapshot_id) => self.fork(snapshot_id, owner),
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
        let committed_hashes = ctx.committed_hashes.clone();
        let lineage = ctx.lineage.clone();
        let src_working = ctx.working_pages.clone();

        let max_pos = ctx.max_committed_position;
        let snapshot_filled = ctx.working_page_tokens.clone();

        if !committed_hashes.is_empty() {
            self.devices[dev_idx].retain(&committed_hashes);
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
            committed_hashes: committed_hashes.clone(),
            lineage,
            working_page_tokens: snapshot_filled,
            max_committed_position: max_pos,
            state: snapshot_state,
            pending_suspend: false,
            last_access: Instant::now(),
        });
        self.snapshots.insert((username, name.clone()), snapshot_id);

        // If snapshot ended up Suspended (CPU fallback for working pages),
        // release the refcounts we acquired. Suspended invariant: no held refcounts.
        // On open/fork, retain will be called to re-acquire.
        if snapshot_state == ContextState::Suspended && !committed_hashes.is_empty() {
            self.devices[dev_idx].release(&committed_hashes);
        }

        Ok(if auto_generated { Some(name) } else { None })
    }

    pub(crate) fn delete(&mut self, username: String, name: String) -> Result<()> {
        let snapshot_id = self.snapshots.remove(&(username, name))
            .ok_or_else(|| anyhow::anyhow!("Snapshot not found"))?;

        if let Some(ctx) = self.contexts.remove(&snapshot_id) {
            let dev_idx = ctx.device.unwrap_or(0) as usize;
            if !ctx.committed_hashes.is_empty() && !ctx.is_suspended() {
                self.devices[dev_idx].release(&ctx.committed_hashes);
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
    pub(crate) fn take(&mut self, username: String, name: String, owner: ProcessId) -> Result<ContextId> {
        let key = (username, name);
        let snapshot_id = *self.snapshots.get(&key)
            .ok_or_else(|| anyhow::anyhow!("Snapshot not found"))?;

        let snap = self.contexts.remove(&snapshot_id)
            .ok_or_else(|| anyhow::anyhow!("Snapshot context missing"))?;

        self.snapshots.remove(&key);

        let dev_idx = snap.device.unwrap_or(0) as usize;
        let was_suspended = snap.is_suspended();

        // Working pages: GPU pages transfer directly, CPU pages need H2D copy
        let new_working = if snap.working_pages.is_empty() {
            Vec::new()
        } else if was_suspended {
            // CPU → GPU: allocate GPU pages and copy
            let n = snap.working_pages.len();
            if let Some(gpu_pages) = self.devices[dev_idx].alloc_gpu_pages(n) {
                let _ = device::copy_h2d(dev_idx as DeviceId, &gpu_pages, &snap.working_pages);
                // Free CPU pages after H2D copy is issued
                self.devices[dev_idx].free_cpu_pages(&snap.working_pages);
                gpu_pages
            } else {
                // GPU OOM — can't restore working pages. Bail to avoid page/token desync.
                self.devices[dev_idx].free_cpu_pages(&snap.working_pages);
                anyhow::bail!("take: no GPU pages for working page swap-in");
            }
        } else {
            // GPU → new context: direct ownership transfer (zero-copy, snap is already consumed)
            snap.working_pages
        };

        // Committed chain: if snapshot was Suspended, refcounts were released during save.
        // We need to re-acquire them for the new Active context.
        // If snapshot was Active, refcounts transfer directly (snapshot consumed).
        if was_suspended && !snap.committed_hashes.is_empty() {
            self.devices[dev_idx].retain(&snap.committed_hashes);
        }

        let committed_len = snap.committed_hashes.len();
        let working_len = new_working.len();

        let new_id = self.next_id();
        self.contexts.insert(new_id, Context {
            owner: Some(owner),
            device: Some(dev_idx as DeviceId),
            working_pages: new_working,
            committed_hashes: snap.committed_hashes,
            lineage: snap.lineage,
            working_page_tokens: snap.working_page_tokens,
            max_committed_position: snap.max_committed_position,
            state: ContextState::Active,
            pending_suspend: false,
            last_access: Instant::now(),
        });

        // Register with owning process
        {
            let proc = self.process_entry(owner);
            proc.context_ids.push(new_id);
            let d = proc.device_mut(dev_idx);
            d.committed += committed_len;
            d.working += working_len;
        }

        Ok(new_id)
    }
}
