use std::collections::HashMap;

use super::{
    ContextId, ContextManager, ContextState, ProcessState,
    Record, ReplayFill,
    BUFFERS, PAGE_SIZES,
};
use super::pagestore::{PhysicalPageId, compute_last_page_len};
use crate::device::DeviceId;
use crate::process::ProcessId;

// =============================================================================
// Restore methods on ContextManager
// =============================================================================

impl ContextManager {
    /// Admission check: can this process be fully restored?
    /// Checks that all devices have enough free GPU pages for the process's
    /// working pages (on CPU) plus pending_alloc requirements.
    pub(crate) fn can_restore_process(&self, pid: ProcessId) -> bool {
        let ctx_ids = match self.processes.get(&pid) {
            Some(p) => &p.context_ids,
            None => return false,
        };

        // Compute required pages per device
        let mut required: HashMap<usize, usize> = HashMap::new();

        for &ctx_id in ctx_ids {
            if let Some(ctx) = self.contexts.get(&ctx_id) {
                if ctx.state != ContextState::Suspended { continue; }
                let dev_idx = ctx.device.unwrap_or(0) as usize;
                // Working pages on CPU that need GPU allocation for swap-in
                *required.entry(dev_idx).or_insert(0) += ctx.working_pages_cpu.len();
                // TODO: Add pages_needing_replay estimate here.
                // Currently replay is not executed during restore_context, so
                // this is 0. When replay is wired up, we'll need to walk the
                // lineage + prefix match to compute the replay page count.
            }
        }

        // Add pending_alloc requirements
        if let Some(allocs) = self.pending_allocs_map.get(&pid) {
            for alloc in allocs {
                let dev_idx = alloc.device as usize;
                *required.entry(dev_idx).or_insert(0) += alloc.num_pages;
            }
        }

        // Check availability on all devices
        for (&dev_idx, &needed) in &required {
            if self.devices[dev_idx].free_gpu_pages() < needed {
                return false;
            }
        }

        true
    }

    /// Restore a process and replay its pending_allocs.
    /// Called by drain_queues after admission check passes.
    pub(crate) fn restore_and_replay(&mut self, pid: ProcessId) {
        let ctx_ids: Vec<ContextId> = self.processes.get(&pid)
            .map(|p| p.context_ids.clone())
            .unwrap_or_default();

        // Restore all suspended contexts
        for &ctx_id in &ctx_ids {
            let is_suspended = self.contexts.get(&ctx_id)
                .map(|c| c.state == ContextState::Suspended)
                .unwrap_or(false);
            if is_suspended {
                let _ = self.restore_context(ctx_id);
            }
        }

        // Transition process to Running
        if let Some(proc) = self.processes.get_mut(&pid) {
            proc.state = ProcessState::Running;
        }

        // Replay pending_allocs: allocate pages and fire response channels
        if let Some(allocs) = self.pending_allocs_map.remove(&pid) {
            for alloc in allocs {
                let dev_idx = alloc.device as usize;
                if let Some(pages) = self.devices[dev_idx].alloc_working(alloc.num_pages) {
                    self.apply_alloc(alloc.context_id, dev_idx, pages, Some(pid));
                    let _ = alloc.response.send(Ok(()));
                } else {
                    // Shouldn't happen — admission check verified availability.
                    let _ = alloc.response.send(Err(anyhow::anyhow!(
                        "Insufficient pages during restore replay (should not happen)"
                    )));
                }
            }
        }
    }


    /// Restore a single suspended context.
    ///
    /// Phase 1: Swap-in working pages from CPU → GPU
    /// Phase 2: Rebuild committed chain via prefix match + acquire
    pub(crate) fn restore_context(&mut self, ctx_id: ContextId) -> anyhow::Result<()> {
        let ctx = self.contexts.get(&ctx_id)
            .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
        if ctx.state != ContextState::Suspended {
            return Ok(()); // Already active
        }

        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let cpu_slots = ctx.working_pages_cpu.clone();
        let tip = ctx.committed_tip;
        let owner = ctx.owner;

        // Phase 1: Swap-in working pages from CPU → GPU
        if !cpu_slots.is_empty() {
            let dev = &mut self.devices[dev_idx];
            let swap_ops = dev.swap_in(&cpu_slots)?;

            let new_gpu_pages: Vec<PhysicalPageId> = swap_ops.iter().map(|op| op.gpu_phys).collect();

            if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
                ctx.working_pages = new_gpu_pages.clone();
                ctx.working_pages_cpu.clear();
            }

            // Track swap-in as working pages in arbiter
            if let Some(pid) = owner {
                self.arbiter.add_working(pid, dev_idx, new_gpu_pages.len());
            }
        }

        // Phase 2: Rebuild committed chain via longest prefix match
        if let Some(tip_hash) = tip {
            let dev = &mut self.devices[dev_idx];
            let chain = dev.walk_chain(tip_hash);
            let prefix_len = dev.longest_prefix_length(&chain);

            // Acquire refcounts for prefix-matched pages
            if prefix_len > 0 {
                let prefix_tip = chain[prefix_len - 1];
                dev.acquire_chain(prefix_tip);

                // Truncate committed_tip to the prefix tip.
                // Pages beyond the prefix are not GPU-resident — stale references.
                if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
                    ctx.committed_tip = Some(prefix_tip);
                }
            } else {
                // No prefix match at all — clear committed_tip.
                if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
                    ctx.committed_tip = None;
                }
            }

            // Update arbiter: move prefix_len pages from working to committed.
            // (add_working was already called in Phase 1; commit() subtracts
            // from working and adds to committed, which is correct even for
            // multi-context processes since both ops are additive.)
            if let Some(pid) = owner {
                if prefix_len > 0 {
                    // We don't actually have prefix_len working pages to "commit" —
                    // these are directly acquired committed pages. Add them as
                    // working first then commit to get the correct accounting.
                    self.arbiter.add_working(pid, dev_idx, prefix_len);
                    self.arbiter.commit(pid, dev_idx, prefix_len);
                }
            }
        }

        // Phase 3: Mark context as Active
        if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
            ctx.state = ContextState::Active;
            ctx.pending_suspend = false;
        }

        Ok(())
    }

    /// Build replay chunks for pages that need to be re-computed after restore.
    /// Returns empty vec if no replay is needed (full prefix match).
    fn build_replay_chunks(&mut self, ctx_id: ContextId) -> anyhow::Result<Vec<ReplayFill>> {
        let ctx = self.contexts.get(&ctx_id)
            .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let tip = ctx.committed_tip;
        let lineage = ctx.lineage.clone();

        let tip_hash = match tip {
            Some(h) => h,
            None => return Ok(Vec::new()),
        };

        let dev = &self.devices[dev_idx];
        let chain = dev.walk_chain(tip_hash);
        let prefix_len = dev.longest_prefix_length(&chain);

        if prefix_len >= chain.len() {
            return Ok(Vec::new()); // All pages are GPU-resident
        }

        // Compute how many tokens need replay
        let page_size = self.page_size;
        let skip_tokens = prefix_len * page_size;

        // Walk lineage to collect tokens for the pages that need replay
        let mut replay_chunks = Vec::new();
        let mut lineage_offset = 0usize;

        for record in &lineage {
            match record {
                Record::Fill { tokens, positions, mask, adapter } => {
                    let record_len = tokens.len();
                    let record_end = lineage_offset + record_len;

                    if record_end <= skip_tokens {
                        // This entire record is in the prefix — skip
                        lineage_offset = record_end;
                        continue;
                    }

                    let start_in_record = skip_tokens.saturating_sub(lineage_offset);
                    let chunk_tokens = &tokens[start_in_record..];

                    // Split into page-aligned chunks
                    for (i, page_tokens) in chunk_tokens.chunks(page_size).enumerate() {
                        let page_start = start_in_record + i * page_size;
                        let page_end = page_start + page_tokens.len();
                        if page_tokens.len() < page_size { continue; } // Partial page, skip

                        let page_pos = &positions[page_start..page_end];
                        let page_masks = &mask[page_start..page_end];

                        // Allocate working pages for this replay chunk
                        let working_pages = match self.devices[dev_idx].alloc_working(1) {
                            Some(p) => p,
                            None => anyhow::bail!("No GPU pages for replay"),
                        };

                        let kv_len = ((prefix_len + replay_chunks.len() + 1) * page_size) as u32;
                        let num_pages = (prefix_len + replay_chunks.len() + 1) as u32;

                        replay_chunks.push(ReplayFill {
                            tokens: page_tokens.to_vec(),
                            positions: page_pos.to_vec(),
                            masks: page_masks.to_vec(),
                            adapter: *adapter,
                            physical_page_ids: working_pages,
                            device_id: dev_idx as DeviceId,
                            kv_len,
                            last_page_len: compute_last_page_len(kv_len, num_pages, page_size as u32),
                            num_pages,
                        });
                    }

                    lineage_offset = record_end;
                }
            }
        }

        Ok(replay_chunks)
    }

    /// Commit a replay chunk (called during restore, after forward pass).
    pub(crate) fn commit_replay_chunk_impl(
        &mut self,
        id: ContextId,
        _num_pages: u32,
        tokens: Vec<u32>,
        positions: Vec<u32>,
        masks: Vec<crate::inference::brle::Brle>,
        _adapter: Option<crate::adapter::AdapterId>,
    ) -> anyhow::Result<()> {
        let ctx = self.contexts.get(&id)
            .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let prev_hash = ctx.committed_tip.unwrap_or(0);
        let old_tip = ctx.committed_tip;
        let working_phys = ctx.working_pages.clone();
        let owner = ctx.owner;

        let dev = &mut self.devices[dev_idx];
        let hashes = dev.compute_page_hashes(&tokens, &positions, &masks, prev_hash);

        let mut new_phys = Vec::new();
        let mut running_prev = prev_hash;
        for (i, &hash) in hashes.iter().enumerate() {
            if i < working_phys.len() {
                let (phys, _) = dev.commit_working(hash, running_prev, working_phys[i]);
                new_phys.push(phys);
            }
            running_prev = hash;
        }

        let new_tip = *hashes.last()
            .ok_or_else(|| anyhow::anyhow!("No hashes computed during replay commit"))?;
        dev.update_index_cache(new_tip, old_tip, &new_phys);

        // Update Context (local)
        if let Some(ctx) = self.contexts.get_mut(&id) {
            ctx.committed_tip = Some(new_tip);
            // Drain the working pages that were committed during replay
            let to_remove = new_phys.len().min(ctx.working_pages.len());
            ctx.working_pages.drain(..to_remove);
        }

        // Update ContextBuffer (DashMap) — increment committed_len
        if let Some(mut buf) = self.buf_mut(id) {
            buf.committed_len += new_phys.len();
        }

        if let Some(pid) = owner {
            self.arbiter.commit(pid, dev_idx, new_phys.len());
        }

        Ok(())
    }

    /// Finish restoration: transition context from replay to fully active.
    pub(crate) fn finish_restore_impl(&mut self, id: ContextId) {
        if let Some(ctx) = self.contexts.get_mut(&id) {
            ctx.state = ContextState::Pinned;
        }
    }
}
