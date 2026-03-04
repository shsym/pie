//! Restore — Process and Context Restoration Logic.
//!
//! Handles the CPU→GPU migration path for suspended contexts:
//! 1. Swap-in working pages from CPU
//! 2. Rebuild committed chain via longest prefix match + replay
//! 3. Transition context Active, process Running

use std::collections::HashMap;

use super::{
    ContextId, ContextManager, ContextState, ProcessState,
    Record, ReplayFill, ResidentResult,
    CONTEXTS, PAGE_SIZES,
};
use super::pagestore::{PhysicalPageId, compute_last_page_len};
use crate::device::DeviceId;
use crate::process::ProcessId;

// =============================================================================
// Restore methods on ContextManager
// =============================================================================

impl ContextManager {
    /// Attempt to restore a complete process. Returns Ok(ResidentResult) if
    /// successful, Err if insufficient GPU pages.
    pub(crate) fn try_restore_process(
        &mut self,
        pid: ProcessId,
        requesting_ctx_id: ContextId,
    ) -> anyhow::Result<ResidentResult> {
        let ctx_ids: Vec<ContextId> = self.processes.get(&pid)
            .map(|p| p.context_ids.clone())
            .unwrap_or_default();

        // Restore all contexts for this process
        for &ctx_id in &ctx_ids {
            if let Some(ctx) = CONTEXTS.get(&(self.model_idx, ctx_id)) {
                if ctx.state == ContextState::Suspended {
                    drop(ctx);
                    self.restore_context(ctx_id)?;
                }
            }
        }

        // Transition process to Running
        if let Some(proc) = self.processes.get_mut(&pid) {
            proc.state = ProcessState::Running;
        }

        // Build result for the requesting context
        let pages = self.get_physical_page_ids_impl(requesting_ctx_id)?;
        let phys_len: usize = pages.values().map(|v| v.len()).sum();

        let (kv_len, debug_state) = {
            let page_size = PAGE_SIZES.get(self.model_idx).copied().unwrap_or(0);
            CONTEXTS.get(&(self.model_idx, requesting_ctx_id))
                .map(|ctx| {
                    let kv = (ctx.committed_len * page_size + ctx.tokens_filled.len()) as u32;
                    let state = format!(
                        "committed_len={} tokens_filled={} working_pages={} working_cpu={} state={:?} phys_len={}",
                        ctx.committed_len, ctx.tokens_filled.len(),
                        ctx.working_pages.len(), ctx.working_cpu_slots.len(),
                        ctx.state, phys_len,
                    );
                    (kv, state)
                })
                .unwrap_or((0, "MISSING".to_string()))
        };

        // Pin context as non-evictable
        if let Some(mut ctx) = CONTEXTS.get_mut(&(self.model_idx, requesting_ctx_id)) {
            ctx.state = ContextState::Pinned;
        }

        // Check if we need replay chunks
        let replay_chunks = self.build_replay_chunks(requesting_ctx_id)?;
        if replay_chunks.is_empty() {
            Ok(ResidentResult {
                replay_chunks: None,
                pages,
                kv_len,
                debug_state,
            })
        } else {
            Ok(ResidentResult {
                replay_chunks: Some(replay_chunks),
                pages: HashMap::new(),
                kv_len: 0,
                debug_state: "replay".to_string(),
            })
        }
    }

    /// Restore a single suspended context.
    ///
    /// Phase 1: Swap-in working pages from CPU → GPU
    /// Phase 2: Rebuild committed chain via prefix match + acquire
    pub(crate) fn restore_context(&mut self, ctx_id: ContextId) -> anyhow::Result<()> {
        let ctx = self.ctx(ctx_id)?;
        if ctx.state != ContextState::Suspended {
            return Ok(()); // Already active
        }

        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let cpu_slots = ctx.working_cpu_slots.clone();
        let tip = ctx.committed_tip;
        let owner = ctx.owner;
        drop(ctx);

        // Phase 1: Swap-in working pages from CPU → GPU
        if !cpu_slots.is_empty() {
            let dev = &mut self.devices[dev_idx];
            let swap_ops = dev.swap_in(&cpu_slots)?;

            let new_gpu_pages: Vec<PhysicalPageId> = swap_ops.iter().map(|op| op.gpu_phys).collect();

            if let Ok(mut ctx) = self.ctx_mut(ctx_id) {
                ctx.working_pages = new_gpu_pages.clone();
                ctx.working_cpu_slots.clear();
            }

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
                // Walk chain to the prefix tip and acquire
                let prefix_tip = chain[prefix_len - 1];
                dev.acquire_chain(prefix_tip);
            }

            if let Some(pid) = owner {
                self.arbiter.set_device(pid, dev_idx,
                    prefix_len + self.arbiter.pages_on(&pid, dev_idx).saturating_sub(cpu_slots.len()),
                    cpu_slots.len(),
                );
            }
        }

        // Phase 3: Mark context as Active
        if let Ok(mut ctx) = self.ctx_mut(ctx_id) {
            ctx.state = ContextState::Active;
            ctx.pending_suspend = false;
        }

        Ok(())
    }

    /// Build replay chunks for pages that need to be re-computed after restore.
    /// Returns empty vec if no replay is needed (full prefix match).
    fn build_replay_chunks(&mut self, ctx_id: ContextId) -> anyhow::Result<Vec<ReplayFill>> {
        let ctx = self.ctx(ctx_id)?;
        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let tip = ctx.committed_tip;
        let lineage = ctx.lineage.clone();
        let committed_len = ctx.committed_len;
        drop(ctx);

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
        let pages_to_replay = chain.len() - prefix_len;
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
                    let chunk_positions = &positions[start_in_record..];
                    let chunk_masks = &mask[start_in_record..];

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
        num_pages: u32,
        tokens: Vec<u32>,
        positions: Vec<u32>,
        masks: Vec<crate::inference::brle::Brle>,
        adapter: Option<crate::adapter::AdapterId>,
    ) -> anyhow::Result<()> {
        let ctx = self.ctx(id)?;
        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let prev_hash = ctx.committed_tip.unwrap_or(0);
        let old_tip = ctx.committed_tip;
        let working_phys = ctx.working_pages.clone();
        let owner = ctx.owner;
        drop(ctx);

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

        {
            let mut ctx = self.ctx_mut(id)?;
            ctx.committed_tip = Some(new_tip);
            // Drain the working pages that were committed during replay
            let to_remove = new_phys.len().min(ctx.working_pages.len());
            ctx.working_pages.drain(..to_remove);
        }

        if let Some(pid) = owner {
            self.arbiter.commit(pid, dev_idx, new_phys.len());
        }

        Ok(())
    }

    /// Finish restoration: transition context from replay to fully active.
    pub(crate) fn finish_restore_impl(&mut self, id: ContextId) {
        if let Some(mut ctx) = CONTEXTS.get_mut(&(self.model_idx, id)) {
            ctx.state = ContextState::Pinned;
        }
    }
}
