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
    /// working pages (recomputed) plus replay pages plus deferred alloc requirements.
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
                // Working pages to re-allocate (recomputed via replay)
                *required.entry(dev_idx).or_insert(0) += ctx.suspended_working_count;
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

        let _deferred_count = self.processes.get(&pid)
            .map(|p| p.deferred_ops.len()).unwrap_or(0);

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

        // tracing::debug!("[LIFECYCLE] RESTORE_ALL: pid={} ctxs={} replays={} deferred={} | free={:?}",
        //     &pid.to_string()[..8], ctx_ids.len(), replay_count, deferred_count,
        //     self.devices.iter().map(|d| d.available_gpu_pages()).collect::<Vec<_>>());

        // If no replays needed, fire deferred ops immediately (no ReplayComplete coming)
        if replay_count == 0 {
            self.fire_deferred_ops(pid);
        }

        Ok(())
    }


    /// Restore a single suspended context.
    ///
    /// Phase 1: Allocate fresh GPU pages for working data (recomputed via replay)
    /// Phase 2: Acquire refcounts for GPU-resident committed prefix
    /// Phase 3: Eagerly promote suffix pages and spawn replay forward passes
    ///          (includes working token replay after committed suffix)
    ///
    /// After this function returns, the context is fully restored from the
    /// metadata perspective. Forward passes fill KV data in the background
    /// while the context is Pinned.
    ///
    /// Returns `true` if replay forward passes were spawned, `false` if
    /// full prefix match (no replay needed) and no working tokens to recompute.
    pub(crate) fn restore(&mut self, ctx_id: ContextId) -> anyhow::Result<bool> {
        let ctx = self.contexts.get(&ctx_id)
            .ok_or_else(|| anyhow::anyhow!("Context not found"))?;

        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let working_count = ctx.suspended_working_count;
        let committed_hashes = ctx.committed_hashes.clone();

        // Phase 1: Allocate fresh GPU pages for working data.
        // KV will be recomputed via replay using working_page_tokens.
        if working_count > 0 {
            let dev = &mut self.devices[dev_idx];
            let gpu_pages = dev.alloc_gpu_pages(working_count)
                .ok_or_else(|| anyhow::anyhow!("No free GPU pages for working re-alloc"))?;
            if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
                ctx.working_pages = gpu_pages;
                ctx.suspended_working_count = 0;
            }
        }

        // Phase 2: Acquire refcounts for GPU-resident committed prefix.
        let prefix_len = if !committed_hashes.is_empty() {
            let dev = &mut self.devices[dev_idx];
            let prefix_len = dev.prefix_len(&committed_hashes);
            if prefix_len > 0 {
                dev.fork(&committed_hashes[..prefix_len]);
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
    /// Also includes a final replay chunk for working page tokens if present.
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
        let working_page_tokens = ctx.working_page_tokens.clone();
        let working_pages = ctx.working_pages.clone();
        let page_size = self.page_size;

        if prefix_len >= committed_len && working_page_tokens.is_empty() {
            return Ok(false);
        }

        // Register pre-allocated suffix pages in the page store.
        // The forward pass will fill KV data; until then the context is Pinned.
        let suffix_count = committed_len - prefix_len;
        if suffix_count > 0 {
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
        }

        // Build the full physical page table (prefix + suffix).
        let full_committed_phys = self.devices[dev_idx].physical_ids(&committed_hashes);

        // Build forward pass requests from the lineage (for tokens/positions/masks).
        let prefix_tokens = prefix_len * page_size;
        let committed_tokens = committed_len * page_size;
        let mut kv_so_far = prefix_tokens as u32;

        let mut requests: Vec<(ForwardPassRequest, Vec<PhysicalPageId>, u32)> = Vec::new();
        let mut token_offset = 0usize;
        let mut pages_emitted = prefix_len;

        for record in &lineage {
            match record {
                Record::Fill { tokens, positions, mask, adapter, adapter_seed } => {
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
                    let phys_ids = full_committed_phys[..pages_emitted].to_vec();

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
                        adapter_seed: *adapter_seed,
                        arrival_time: None,
                    };

                    requests.push((fwd_req, phys_ids, last_page_len));
                    kv_so_far += num_input;
                    token_offset = record_end;
                }
            }
        }

        // Append working page tokens to the replay.
        // Try to merge into the last request if adapter matches; otherwise
        // create a separate forward pass request.
        if !working_page_tokens.is_empty() && !working_pages.is_empty() {
            let recomputable = (working_page_tokens.len() + page_size - 1) / page_size;
            let num_replay_pages = recomputable.min(working_pages.len());
            let num_replay_tokens = working_page_tokens.len().min(num_replay_pages * page_size);

            if num_replay_tokens > 0 {
                let mut tokens = Vec::with_capacity(num_replay_tokens);
                let mut positions = Vec::with_capacity(num_replay_tokens);
                let mut masks = Vec::with_capacity(num_replay_tokens);
                let adapter = working_page_tokens[0].adapter;
                let adapter_seed = working_page_tokens[0].adapter_seed;

                for info in &working_page_tokens[..num_replay_tokens] {
                    tokens.push(info.token);
                    positions.push(info.position);
                    masks.push(info.mask.clone());
                }

                // Try to merge into the last committed suffix request.
                let merged = if let Some((last_req, last_phys, _last_page_len)) = requests.last_mut() {
                    if last_req.adapter_id == adapter && last_req.adapter_seed == adapter_seed {
                        // Extend the last request's tokens/positions/masks.
                        last_req.tokens.extend_from_slice(&tokens);
                        last_req.positions.extend_from_slice(&positions);
                        last_req.masks.extend_from_slice(&masks);

                        // Extend the page table with working pages.
                        last_phys.extend_from_slice(&working_pages[..num_replay_pages]);

                        // Recompute last_page_len for the merged request.
                        let num_input = num_replay_tokens as u32;
                        kv_so_far += num_input;
                        let total_pages_for_fwd = last_phys.len() as u32;
                        *_last_page_len = compute_last_page_len(kv_so_far, total_pages_for_fwd, page_size as u32);

                        true
                    } else {
                        false
                    }
                } else {
                    false
                };

                if !merged {
                    // Different adapter or no prior requests — separate forward pass.
                    let mut phys_ids = full_committed_phys.clone();
                    phys_ids.extend_from_slice(&working_pages[..num_replay_pages]);

                    let num_input = num_replay_tokens as u32;
                    let total_kv = kv_so_far + num_input;
                    let total_pages_for_fwd = phys_ids.len() as u32;
                    let last_page_len = compute_last_page_len(total_kv, total_pages_for_fwd, page_size as u32);

                    let fwd_req = ForwardPassRequest {
                        context_id: 0,
                        tokens,
                        positions,
                        speculative_tokens: Vec::new(),
                        speculative_positions: Vec::new(),
                        output_speculative_tokens: false,
                        masks,
                        logit_mask: None,
                        sampling_indices: Vec::new(),
                        samplers: Vec::new(),
                        adapter_id: adapter,
                        adapter_seed,
                        arrival_time: None,
                    };

                    requests.push((fwd_req, phys_ids, last_page_len));
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
                    tracing::error!(ctx = ctx_id, device = device_id, "replay forward pass failed: {e:#}");
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
    ///
    /// Ordering is critical: if any operation fails to allocate and re-enters
    /// the alloc_queue, all subsequent operations must also be deferred to
    /// preserve causal ordering (e.g., `pin` must not fire before
    /// `reserve_working_pages` allocates its pages).
    fn fire_deferred_ops(&mut self, pid: ProcessId) {
        let ops = self.processes.get_mut(&pid)
            .map(|p| std::mem::take(&mut p.deferred_ops))
            .unwrap_or_default();
        let mut stalled = false;
        for op in ops {
            if stalled {
                // A prior op didn't get its pages — queue everything after it.
                self.alloc_queue.push_back(op);
            } else if op.num_pages == 0 {
                (op.on_alloc)(self, Vec::new());
            } else if let Some(pages) = self.devices[op.device].alloc_gpu_pages(op.num_pages) {
                (op.on_alloc)(self, pages);
            } else {
                // Pages no longer available — this and all remaining ops re-enter contention.
                self.alloc_queue.push_back(op);
                stalled = true;
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
