//! Restoration — Process Recovery and Replay Planning.
//!
//! When a Pending process wins the `restore_queue`, this module brings it
//! back to Running state.
//!
//! ## `restore_all(pid)` — Full Process Recovery
//!
//! 1. Call `restore(ctx)` for each Suspended context.
//! 2. Re-set scheduler accounting from actual context state.
//! 3. Transition process to Running.
//! 4. Fire deferred ops (or wait for replay completion).
//!
//! ## `restore(ctx)` — Single Context Recovery
//!
//! Three phases:
//! 1. **Swap-in**: alloc GPU pages, H2D copy from CPU, free CPU slots.
//! 2. **Prefix match**: `prefix_len(committed_hashes)` → acquire refcounts
//!    for GPU-resident prefix via `retain`.
//! 3. **Replay suffix**: alloc fresh GPU pages for missing suffix, register
//!    in PageStore via `commit_batch`, spawn replay forward passes. Context
//!    stays Pinned until all replay passes complete (`replay_complete`).
//!
//! **Invariant**: restoration never evicts. The admission check in
//! `can_restore_all` guarantees sufficient pages on all devices before
//! `restore_all` is called.
//!
//! ## `replay_complete(ctx_id)` — Post-Replay Transition
//!
//! Pinned → Active. When all replay passes for a process finish
//! (`pending_replay_count` → 0), deferred ops fire. If the process was
//! re-suspended mid-replay (`pending_suspend`), the context is re-suspended
//! instead.

use std::collections::HashMap;

use super::{
    ContextId, ContextManager, ContextState, SERVICES,
    Record,
};
use super::sched::ProcessState;
use super::pagestore::{PhysicalPageId, compute_last_page_len};
use crate::inference;
use crate::inference::request::ForwardPassRequest;
use crate::process::ProcessId;

// =============================================================================
// Restore methods on ContextManager
// =============================================================================


impl ContextManager {

    /// Admission check: can this process be fully restored?
    /// Checks that all devices have enough free GPU pages for the process's
    /// working pages (on CPU) plus replay pages plus deferred alloc requirements.
    pub(crate) fn can_restore_all(&mut self, pid: ProcessId) -> bool {
        let ctx_ids = match self.processes.get(&pid) {
            Some(p) => &p.context_ids,
            None => return false,
        };

        // Compute required pages per device
        let mut required: HashMap<usize, usize> = HashMap::new();

        for &ctx_id in ctx_ids {
            if let Some(ctx) = self.contexts.get(&ctx_id) {
                if !ctx.is_suspended() { continue; }
                let dev_idx = ctx.device.unwrap_or(0) as usize;
                // Working pages on CPU that need GPU allocation for swap-in
                *required.entry(dev_idx).or_insert(0) += ctx.working_pages.len();
                // Pages needing replay: check prefix match, count missing suffix
                if !ctx.committed_hashes.is_empty() {
                    let dev = &self.devices[dev_idx];
                    let prefix_len = dev.prefix_len(&ctx.committed_hashes);
                    let replay_pages = ctx.committed_hashes.len().saturating_sub(prefix_len);
                    *required.entry(dev_idx).or_insert(0) += replay_pages;
                }
            }
        }

        // Add deferred op requirements
        if let Some(proc) = self.processes.get(&pid) {
            for op in &proc.deferred_ops {
                if op.num_pages > 0 {
                    *required.entry(op.device).or_insert(0) += op.num_pages;
                }
            }
        }

        // Check availability on all devices
        for (&dev_idx, &needed) in &required {
            if self.devices[dev_idx].available_gpu_pages() < needed {
                return false;
            }
        }

        true
    }

    /// Restore a process and prepare replay forward passes.
    /// Called by drain_queues after admission check passes.
    ///
    /// Deferred ops fire only after all replay forward passes complete
    /// (in `replay_complete`). If no replays are needed, deferred ops
    /// fire immediately here.
    pub(crate) fn restore_all(&mut self, pid: ProcessId) -> anyhow::Result<()> {
        let ctx_ids: Vec<ContextId> = self.processes.get(&pid)
            .map(|p| p.context_ids.clone())
            .ok_or_else(|| anyhow::anyhow!("Process not found"))?;

        let mut replay_count: usize = 0;

        // Restore all suspended contexts
        for &ctx_id in &ctx_ids {
            let has_replay = self.restore(ctx_id)?;
            if has_replay { replay_count += 1; }
        }

        // Set scheduler accounting from actual context state.
        // suspend_process zeroed all devices; now set exact counts from restored contexts.
        for &ctx_id in &ctx_ids {
            let ctx = self.contexts.get(&ctx_id).unwrap();
            let dev_idx = ctx.device.unwrap_or(0) as usize;
            let committed_len = ctx.committed_len();
            let working_len = ctx.working_pages.len();
            let d = self.process_entry(pid).device_mut(dev_idx);
            d.committed += committed_len;
            d.working += working_len;
        }

        // Transition process to Running
        if let Some(proc) = self.processes.get_mut(&pid) {
            proc.state = ProcessState::Running;
            proc.pending_replay_count = replay_count;
        }

        // If no replays needed, fire deferred ops immediately (no ReplayComplete coming)
        if replay_count == 0 {
            self.fire_deferred_ops(pid);
        }

        Ok(())
    }


    /// Restore a single suspended context.
    ///
    /// Phase 1: Swap-in working pages from CPU → GPU
    /// Phase 2: Acquire refcounts for GPU-resident committed prefix
    /// Phase 3: Eagerly promote suffix pages and spawn replay forward passes
    ///
    /// After this function returns, the context is fully restored from the
    /// metadata perspective. Forward passes fill KV data in the background
    /// while the context is Pinned.
    ///
    /// Returns `true` if replay forward passes were spawned, `false` if
    /// full prefix match (no replay needed).
    pub(crate) fn restore(&mut self, ctx_id: ContextId) -> anyhow::Result<bool> {
        let ctx = self.contexts.get(&ctx_id)
            .ok_or_else(|| anyhow::anyhow!("Context not found"))?;

        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let cpu_pages = ctx.working_pages.clone();
        let committed_hashes = ctx.committed_hashes.clone();

        // Phase 1: Swap-in working pages from CPU → GPU
        if !cpu_pages.is_empty() {
            let dev = &mut self.devices[dev_idx];
            let gpu_pages = dev.alloc_gpu_pages(cpu_pages.len())
                .ok_or_else(|| anyhow::anyhow!("No free GPU pages for working swap-in"))?;
            let _ = crate::device::copy_h2d(dev_idx, &gpu_pages, &cpu_pages);
            self.devices[dev_idx].free_cpu_pages(&cpu_pages);
            if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
                ctx.working_pages = gpu_pages;
            }
        }

        // Phase 2: Acquire refcounts for GPU-resident committed prefix.
        let prefix_len = if !committed_hashes.is_empty() {
            let dev = &mut self.devices[dev_idx];
            let prefix_len = dev.prefix_len(&committed_hashes);
            if prefix_len > 0 {
                dev.retain(&committed_hashes[..prefix_len]);
            }
            prefix_len
        } else {
            0
        };

        // Phase 3: Allocate suffix GPU pages and spawn replay forward passes.
        let suffix_count = committed_hashes.len().saturating_sub(prefix_len);
        let suffix_pages = if suffix_count > 0 {
            self.devices[dev_idx].alloc_gpu_pages(suffix_count)
                .ok_or_else(|| anyhow::anyhow!("No GPU pages for replay but admission check passed"))?
        } else {
            Vec::new()
        };
        let has_replay = self.spawn_replay_passes(ctx_id, dev_idx, prefix_len, suffix_pages)?;

        // Pinned while forward passes fill KV data; Active if full prefix match.
        if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
            ctx.state = if has_replay { ContextState::Pinned } else { ContextState::Active };
        }

        Ok(has_replay)
    }

    /// Promote pre-allocated suffix pages and spawn replay forward passes.
    ///
    /// Registers the provided `suffix_pages` in the page store for the suffix
    /// (pages not GPU-resident), then spawns a tokio task to fill KV data.
    /// Returns `true` if any passes were spawned.
    ///
    /// All GPU page allocation must be done by the caller.
    pub(crate) fn spawn_replay_passes(
        &mut self,
        ctx_id: ContextId,
        dev_idx: usize,
        prefix_len: usize,
        suffix_pages: Vec<PhysicalPageId>,
    ) -> anyhow::Result<bool> {
        let ctx = self.contexts.get(&ctx_id)
            .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
        let lineage = ctx.lineage.clone();
        let committed_hashes = ctx.committed_hashes.clone();
        let committed_len = committed_hashes.len();
        let page_size = self.page_size;

        if prefix_len >= committed_len {
            return Ok(false);
        }

        // Register pre-allocated suffix pages in the page store.
        // The forward pass will fill KV data; until then the context is Pinned.
        let suffix_count = committed_len - prefix_len;
        let suffix_hashes = &committed_hashes[prefix_len..];
        anyhow::ensure!(suffix_pages.len() == suffix_count,
            "suffix page count mismatch: got {}, need {suffix_count}", suffix_pages.len());
        let suffix_phys = suffix_pages;

        // Register suffix pages in PageStore — navigate through the retained
        // prefix so the trie correctly chains the suffix as children.
        self.devices[dev_idx].extend(
            &committed_hashes[..prefix_len],
            suffix_hashes,
            &suffix_phys,
        );

        // Build the full physical page table (prefix + suffix).
        let full_phys = self.devices[dev_idx].physical_ids(&committed_hashes);

        // Build forward pass requests from the lineage (for tokens/positions/masks).
        let prefix_tokens = prefix_len * page_size;
        let committed_tokens = committed_len * page_size;
        let mut kv_so_far = prefix_tokens as u32;

        let mut requests: Vec<(ForwardPassRequest, Vec<PhysicalPageId>, u32)> = Vec::new();
        let mut token_offset = 0usize;
        let mut pages_emitted = prefix_len;

        for record in &lineage {
            match record {
                Record::Fill { tokens, positions, mask, adapter } => {
                    let record_end = token_offset + tokens.len();

                    if record_end <= prefix_tokens {
                        token_offset = record_end;
                        continue;
                    }
                    if token_offset >= committed_tokens {
                        break;
                    }

                    let start_in_record = prefix_tokens.saturating_sub(token_offset);
                    let end_in_record = (committed_tokens - token_offset).min(tokens.len());
                    let suffix_tokens = &tokens[start_in_record..end_in_record];
                    let suffix_positions = &positions[start_in_record..end_in_record];
                    let suffix_masks = &mask[start_in_record..end_in_record];

                    let num_pages = suffix_tokens.len() / page_size;
                    if num_pages == 0 {
                        token_offset = record_end;
                        continue;
                    }

                    let aligned_len = num_pages * page_size;
                    pages_emitted += num_pages;

                    // Page table for this chunk: all pages up to current position
                    let phys_ids = full_phys[..pages_emitted].to_vec();

                    let num_input = aligned_len as u32;
                    let total_kv = kv_so_far + num_input;
                    let total_pages_for_fwd = phys_ids.len() as u32;
                    let last_page_len = compute_last_page_len(total_kv, total_pages_for_fwd, page_size as u32);

                    let fwd_req = ForwardPassRequest {
                        context_id: 0,
                        tokens: suffix_tokens[..aligned_len].to_vec(),
                        positions: suffix_positions[..aligned_len].to_vec(),
                        speculative_tokens: Vec::new(),
                        speculative_positions: Vec::new(),
                        output_speculative_tokens: false,
                        masks: suffix_masks[..aligned_len].to_vec(),
                        logit_mask: None,
                        sampling_indices: Vec::new(),
                        samplers: Vec::new(),
                        adapter_id: *adapter,
                        adapter_seed: None,
                        arrival_time: None,
                    };

                    requests.push((fwd_req, phys_ids, last_page_len));
                    kv_so_far += num_input;
                    token_offset = record_end;
                }
            }
        }

        if requests.is_empty() {
            return Ok(false);
        }

        // Spawn a task that submits forward passes sequentially, then
        // sends ReplayComplete to unpin the context.
        let model_idx = self.model_idx;
        let device_id = dev_idx;

        tokio::spawn(async move {
            for (fwd_req, phys_ids, last_page_len) in requests {
                let result = inference::submit(
                    model_idx, fwd_req,
                    device_id, phys_ids,
                    last_page_len,
                ).await;

                if let Err(e) = result {
                    eprintln!("REPLAY_FWD_FAIL ctx={ctx_id} device={device_id} err={e:#}");
                    break; // Later chunks depend on this one's KV data
                }
            }

            // Unpin after all chunks complete (or first failure)
            let _ = SERVICES.send(model_idx, super::Message::ReplayComplete { id: ctx_id });
        });

        Ok(true)
    }

    /// Fire all of a process's deferred operations.
    /// Called when all replay forward passes have completed.
    fn fire_deferred_ops(&mut self, pid: ProcessId) {
        let ops = self.processes.get_mut(&pid)
            .map(|p| std::mem::take(&mut p.deferred_ops))
            .unwrap_or_default();
        for op in ops {
            if op.num_pages == 0 {
                (op.on_alloc)(self, Vec::new());
            } else if let Some(pages) = self.devices[op.device].alloc_gpu_pages(op.num_pages) {
                (op.on_alloc)(self, pages);
            } else {
                // Pages no longer available — re-enter contention via alloc_queue.
                self.alloc_queue.push_back(op);
            }
        }
    }

    /// Handle a completed replay forward pass: transition context out of
    /// Pinned, then fire deferred ops when all replay forward passes for
    /// the owning process have completed.
    ///
    /// Called by the actor when a ReplayComplete message arrives.
    ///
    /// NOTE: We intentionally do NOT reuse `unpin()` here because `unpin`
    /// calls `drain_queues`, which can re-restore this same process via
    /// `restore_all` — overwriting `pending_replay_count` before
    /// we get to decrement it (reentrancy corruption).
    pub(crate) fn replay_complete(&mut self, id: ContextId) {
        let (owner, pending) = match self.contexts.get(&id) {
            Some(ctx) if ctx.is_pinned() => (ctx.owner, ctx.pending_suspend),
            _ => return,
        };

        if pending {
            // Re-suspension was requested while replay was in-flight.
            self.suspend_context(id);
            if let Some(pid) = owner {
                if let Some(proc) = self.processes.get_mut(&pid) {
                    proc.pending_pinned = proc.pending_pinned.saturating_sub(1);
                    proc.pending_replay_count = proc.pending_replay_count.saturating_sub(1);
                }
            }
            // Don't fire deferred ops — process was re-suspended.
            // Don't drain_queues here — the suspended pages will be
            // reclaimed when the last pending_pinned clears via unpin.
            return;
        }

        // Normal path: Pinned → Active
        if let Some(ctx) = self.contexts.get_mut(&id) {
            ctx.state = ContextState::Active;
        }

        if let Some(pid) = owner {
            let should_fire = if let Some(proc) = self.processes.get_mut(&pid) {
                proc.pending_replay_count = proc.pending_replay_count.saturating_sub(1);
                proc.pending_replay_count == 0 && proc.state == ProcessState::Running
            } else {
                false
            };
            if should_fire {
                self.fire_deferred_ops(pid);
            }
        }
    }
}
