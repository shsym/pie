use std::collections::HashMap;



use super::{
    ContextId, ContextManager, ContextState,
    Record,
};
use super::sched::ProcessState;
use super::suspend::DeferredOp;
use super::pagestore::{PhysicalPageId, compute_last_page_len, compute_page_hashes};
use crate::adapter::AdapterId;
use crate::device::{self, DeviceId};
use crate::inference;
use crate::inference::brle::Brle;
use crate::inference::request::ForwardPassRequest;
use crate::process::ProcessId;

// =============================================================================
// ReplayFill
// =============================================================================

#[derive(Debug, Clone)]
pub struct ReplayFill {
    pub context_id: ContextId,
    pub tokens: Vec<u32>,
    pub positions: Vec<u32>,
    pub masks: Vec<Brle>,
    pub adapter: Option<AdapterId>,
    pub physical_page_ids: Vec<PhysicalPageId>,
    pub device_id: DeviceId,
    pub kv_len: u32,
    pub last_page_len: u32,
    pub num_pages: u32,
}

// =============================================================================
// Restore methods on ContextManager
// =============================================================================


impl ContextManager {

    /// Dispatch replay forward passes for KV cache recomputation.
    ///
    /// Each ReplayFill contains tokens that need to be run through the model
    /// to regenerate KV data for committed pages lost during eviction.
    /// After the forward pass, the working pages are committed.
    pub(crate) async fn dispatch_replays(&mut self, chunks: Vec<ReplayFill>) {
        if chunks.is_empty() { return; }

        let mut ctx_ids_needing_finish: Vec<ContextId> = Vec::new();

        for chunk in chunks {
            let ctx_id = chunk.context_id;

            // Build a minimal forward pass request (no sampling — just KV fill)
            let fwd_req = ForwardPassRequest {
                context_id: 0, // unused — page IDs provided directly
                tokens: chunk.tokens.clone(),
                positions: chunk.positions.clone(),
                speculative_tokens: Vec::new(),
                speculative_positions: Vec::new(),
                output_speculative_tokens: false,
                masks: chunk.masks.clone(),
                logit_mask: None,
                sampling_indices: Vec::new(),
                samplers: Vec::new(),
                adapter_id: chunk.adapter,
                adapter_seed: None,
                arrival_time: None,
            };

            let result = inference::submit(
                self.model_idx, fwd_req,
                chunk.device_id as usize, chunk.physical_page_ids.clone(),
                chunk.last_page_len,
            ).await;

            if let Err(e) = result {
                eprintln!("REPLAY_FWD_FAIL ctx={ctx_id} device={} err={e:#}", chunk.device_id);
                // RPC failed — still finish restore to avoid permanently Pinned state.
                // The context will have a truncated committed chain but can still serve.
                if !ctx_ids_needing_finish.contains(&ctx_id) {
                    ctx_ids_needing_finish.push(ctx_id);
                }
                continue;
            }

            let _ = self.commit_replay_chunk(
                ctx_id, chunk.num_pages,
                chunk.tokens, chunk.positions, chunk.masks, chunk.adapter,
            );
            if !ctx_ids_needing_finish.contains(&ctx_id) {
                ctx_ids_needing_finish.push(ctx_id);
            }
        }

        for id in ctx_ids_needing_finish {
            self.finish_restore(id);
        }
    }

    /// Admission check: can this process be fully restored?
    /// Checks that all devices have enough free GPU pages for the process's
    /// working pages (on CPU) plus replay pages plus deferred alloc requirements.
    pub(crate) fn can_restore_process(&mut self, pid: ProcessId) -> bool {
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
                // Pages needing replay: walk chain, compute prefix match, count missing suffix
                if let Some(tip_hash) = ctx.committed_tip {
                    let dev = &self.devices[dev_idx];
                    let chain = dev.walk_chain(tip_hash);
                    let prefix_len = dev.longest_prefix_length(&chain);
                    let replay_pages = chain.len().saturating_sub(prefix_len);
                    *required.entry(dev_idx).or_insert(0) += replay_pages;
                }
            }
        }

        // Add deferred alloc requirements (Pin doesn't need extra pages)
        if let Some(proc) = self.processes.get(&pid) {
            if let Some(DeferredOp::Alloc(ref alloc)) = proc.deferred_op {
                let dev_idx = alloc.device as usize;
                *required.entry(dev_idx).or_insert(0) += alloc.num_pages;
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

    /// Restore a process and replay its deferred_op.
    /// Called by drain_queues after admission check passes.
    /// Returns replay chunks that need forward passes dispatched by the caller.
    pub(crate) fn restore_and_replay(&mut self, pid: ProcessId) -> Vec<ReplayFill> {
        let ctx_ids: Vec<ContextId> = self.processes.get(&pid)
            .map(|p| p.context_ids.clone())
            .unwrap_or_default();

        let mut all_replay_chunks = Vec::new();

        // Restore all suspended contexts
        for &ctx_id in &ctx_ids {
            let is_suspended = self.contexts.get(&ctx_id)
                .map(|c| c.is_suspended())
                .unwrap_or(false);
            if is_suspended {
                match self.restore_context(ctx_id) {
                    Ok(chunks) => all_replay_chunks.extend(chunks),
                    Err(e) => {
                        eprintln!("RESTORE_CONTEXT_FAIL ctx={ctx_id} err={e:#}");
                    }
                }
            }
        }

        // Transition process to Running
        if let Some(proc) = self.processes.get_mut(&pid) {
            proc.state = ProcessState::Running;
        }

        // Replay deferred operation: allocate pages / pin and fire response channel
        let deferred = self.processes.get_mut(&pid)
            .and_then(|p| p.deferred_op.take());
        if let Some(op) = deferred {
            match op {
                DeferredOp::Alloc(alloc) => {
                    let dev_idx = alloc.device as usize;
                    if self.reserve_working_pages(alloc.context_id, dev_idx, alloc.num_pages, Some(pid)).is_ok() {
                        let _ = alloc.response.send(Ok(()));
                    } else {
                        // Shouldn't happen — admission check verified availability.
                        let _ = alloc.response.send(Err(anyhow::anyhow!(
                            "Insufficient pages during restore replay (should not happen)"
                        )));
                    }
                }
                DeferredOp::Pin { context_id, num_input_tokens, response } => {
                    let result = self.pin(context_id, num_input_tokens);
                    let _ = response.send(result);
                }
            }
        }

        all_replay_chunks
    }


    /// Restore a single suspended context.
    ///
    /// Phase 1: Swap-in working pages from CPU → GPU
    /// Phase 2: Rebuild committed chain via prefix match + acquire
    /// Phase 3: Build replay chunks for suffix pages (not GPU-resident)
    ///
    /// Returns ReplayFill chunks that need forward passes to recompute KV data.
    /// Empty vec means full prefix match (no replay needed).
    pub(crate) fn restore_context(&mut self, ctx_id: ContextId) -> anyhow::Result<Vec<ReplayFill>> {
        let ctx = self.contexts.get(&ctx_id)
            .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
        if !ctx.is_suspended() {
            return Ok(Vec::new()); // Already active
        }

        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let cpu_pages = ctx.working_pages.clone();
        let tip = ctx.committed_tip;
        let owner = ctx.owner;

        // Phase 1: Swap-in working pages from CPU → GPU
        if !cpu_pages.is_empty() {
            let dev = &mut self.devices[dev_idx];
            let gpu_pages = dev.alloc_gpu_pages(cpu_pages.len())
                .ok_or_else(|| anyhow::anyhow!("No free GPU pages for working swap-in"))?;

            // Copy CPU → GPU, then free CPU pages
            let _ = device::copy_h2d(dev_idx, &gpu_pages, &cpu_pages);
            self.devices[dev_idx].free_cpu_pages(&cpu_pages);

            if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
                ctx.working_pages = gpu_pages.clone();
            }

            // Track swap-in as working pages in scheduler
            if let Some(pid) = owner {
                self.process_entry(pid).device_mut(dev_idx).working += gpu_pages.len();
            }
        }

        // Phase 2: Rebuild committed chain via longest prefix match
        let prefix_len = if let Some(tip_hash) = tip {
            let dev = &mut self.devices[dev_idx];
            let chain = dev.walk_chain(tip_hash);
            let prefix_len = dev.longest_prefix_length(&chain);

            // Acquire refcounts for prefix-matched pages
            if prefix_len > 0 {
                let prefix_tip = chain[prefix_len - 1];
                dev.acquire_chain(prefix_tip);

                // Rebuild index_cache for the prefix tip
                let _ = dev.resolve_physical(prefix_tip);

                // Truncate committed_tip to the prefix tip.
                if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
                    ctx.committed_tip = Some(prefix_tip);
                }
            } else {
                // No prefix match at all — clear committed_tip.
                if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
                    ctx.committed_tip = None;
                }
            }

            // Update committed_len and max_committed_position on Context
            // to match the truncated prefix. Recompute max_committed_position
            // from the lineage to avoid stale values rejecting valid commits.
            if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
                ctx.committed_len = prefix_len;
                if prefix_len == 0 {
                    ctx.max_committed_position = None;
                } else {
                    // Recompute from lineage: max position in the first
                    // prefix_len * page_size tokens.
                    let prefix_tokens = prefix_len * self.page_size;
                    let mut count = 0usize;
                    let mut max_p: Option<u32> = None;
                    for record in &ctx.lineage {
                        match record {
                            Record::Fill { positions, .. } => {
                                for &pos in positions {
                                    if count >= prefix_tokens { break; }
                                    max_p = Some(max_p.map_or(pos, |m: u32| m.max(pos)));
                                    count += 1;
                                }
                            }
                        }
                        if count >= prefix_tokens { break; }
                    }
                    ctx.max_committed_position = max_p;
                }
            }

            // Update scheduler accounting for prefix-matched committed pages
            if let Some(pid) = owner {
                if prefix_len > 0 {
                    self.process_entry(pid).device_mut(dev_idx).committed += prefix_len;
                }
            }

            prefix_len
        } else {
            0
        };

        // Phase 3: Build replay chunks for pages beyond the prefix
        let replay_chunks = self.build_replay_chunks(ctx_id, dev_idx, prefix_len)?;

        if replay_chunks.is_empty() {
            // No replay needed — full prefix match. Mark Active immediately.
            if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
                ctx.state = ContextState::Active;
                ctx.pending_suspend = false;
            }
        } else {
            // Replay needed — mark as Pinned to prevent eviction while
            // KV recovery forward passes are in-flight. Transitions to
            // Active after finish_restore is called.
            if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
                ctx.state = ContextState::Pinned;
                ctx.pending_suspend = false;
            }
        }

        Ok(replay_chunks)
    }

    /// Build replay chunks for committed pages beyond the prefix match.
    ///
    /// Flattens the lineage, computes page hashes, allocates working pages
    /// for the suffix (pages not GPU-resident), and builds ReplayFill structs
    /// with full page tables for forward pass dispatch.
    fn build_replay_chunks(
        &mut self,
        ctx_id: ContextId,
        dev_idx: usize,
        prefix_len: usize,
    ) -> anyhow::Result<Vec<ReplayFill>> {
        let ctx = self.contexts.get(&ctx_id)
            .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
        let lineage = ctx.lineage.clone();
        let owner = ctx.owner;
        let page_size = self.page_size;

        // Flatten lineage into contiguous arrays
        let mut all_tokens = Vec::new();
        let mut all_positions = Vec::new();
        let mut all_masks = Vec::new();
        let mut adapters: Vec<(usize, Option<crate::adapter::AdapterId>)> = Vec::new();

        for record in &lineage {
            match record {
                Record::Fill { tokens, positions, mask, adapter } => {
                    adapters.push((all_tokens.len(), *adapter));
                    all_tokens.extend_from_slice(tokens);
                    all_positions.extend_from_slice(positions);
                    all_masks.extend_from_slice(mask);
                }
            }
        }

        // Only full pages are committable
        let page_aligned = (all_tokens.len() / page_size) * page_size;
        let total_committed_pages = page_aligned / page_size;

        // All committed pages are prefix-matched — no replay needed
        if prefix_len >= total_committed_pages {
            return Ok(Vec::new());
        }

        let matched_tokens = prefix_len * page_size;
        let mut kv_so_far = matched_tokens as u32;

        // Phase 1 working pages are stale (they reference old KV data from
        // before suspension). We keep track of them to drain after replay.
        let initial_working_len = self.contexts.get(&ctx_id)
            .map(|c| c.working_pages.len()).unwrap_or(0);

        let mut chunks = Vec::new();
        let mut offset = matched_tokens;

        while offset < page_aligned {
            // Find the adapter for this position
            let adapter = adapters.iter().rev()
                .find(|(start, _)| *start <= offset)
                .and_then(|(_, a)| *a);

            // Find the next adapter boundary
            let next_adapter_start = adapters.iter()
                .find(|(start, _)| *start > offset)
                .map(|(start, _)| *start)
                .unwrap_or(page_aligned);

            let chunk_end = next_adapter_start.min(page_aligned);
            let chunk_tokens = &all_tokens[offset..chunk_end];
            let chunk_positions = &all_positions[offset..chunk_end];
            let chunk_masks = &all_masks[offset..chunk_end];

            // Number of full pages in this chunk
            let full_pages = chunk_tokens.len() / page_size;
            if full_pages == 0 {
                offset = chunk_end;
                continue;
            }

            // Only take page-aligned tokens for the replay chunk
            let aligned_len = full_pages * page_size;

            // Allocate working pages for replay
            let new_pages = match self.devices[dev_idx].alloc_gpu_pages(full_pages) {
                Some(p) => p,
                None => anyhow::bail!("No GPU pages for replay (should not happen: admission check passed)"),
            };

            // Track in context and scheduler
            if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
                ctx.working_pages.extend(&new_pages);
            }
            if let Some(pid) = owner {
                self.process_entry(pid).device_mut(dev_idx).working += full_pages;
            }

            // Build physical page IDs: committed prefix + all working pages after initial
            let phys_ids = {
                let mut all_phys = if let Some(tip) = self.contexts.get(&ctx_id)
                    .and_then(|c| c.committed_tip)
                {
                    self.devices[dev_idx].resolve_physical(tip)
                } else {
                    Vec::new()
                };
                // Extend with working pages allocated for replay (skip stale Phase 1 pages)
                if let Some(ctx) = self.contexts.get(&ctx_id) {
                    all_phys.extend_from_slice(&ctx.working_pages[initial_working_len..]);
                }
                all_phys
            };

            let num_input = aligned_len as u32;
            let total_kv = kv_so_far + num_input;
            let total_pages_for_fwd = phys_ids.len() as u32;
            let last_page_len = compute_last_page_len(total_kv, total_pages_for_fwd, page_size as u32);

            chunks.push(ReplayFill {
                context_id: ctx_id,
                tokens: chunk_tokens[..aligned_len].to_vec(),
                positions: chunk_positions[..aligned_len].to_vec(),
                masks: chunk_masks[..aligned_len].to_vec(),
                adapter,
                physical_page_ids: phys_ids,
                device_id: dev_idx as DeviceId,
                kv_len: kv_so_far,
                last_page_len,
                num_pages: full_pages as u32,
            });
            kv_so_far += num_input;
            offset += aligned_len;
        }

        // Drain stale Phase 1 working pages (they hold old KV data)
        if initial_working_len > 0 && !chunks.is_empty() {
            // Clear working_page_tokens first (separate borrow scope)
            if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
                ctx.working_page_tokens.clear();
            }
            let stale_pages: Vec<PhysicalPageId> = {
                if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
                    ctx.working_pages.drain(..initial_working_len).collect()
                } else {
                    Vec::new()
                }
            };
            self.devices[dev_idx].free_gpu_pages(&stale_pages);
            if let Some(pid) = owner {
                let d = self.process_entry(pid).device_mut(dev_idx);
                d.working -= initial_working_len;
            }
        }

        Ok(chunks)
    }

    /// Commit a replay chunk (called during restore, after forward pass).
    pub(crate) fn commit_replay_chunk(
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

        let hashes = compute_page_hashes(self.page_size, &tokens, &positions, &masks, prev_hash);
        let dev = &mut self.devices[dev_idx];

        let mut new_phys = Vec::new();
        let mut running_prev = prev_hash;
        for (i, &hash) in hashes.iter().enumerate() {
            if i < working_phys.len() {
                let (phys, _) = dev.promote_page(hash, running_prev, working_phys[i]);
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

        // Update committed_len on Context
        if let Some(ctx) = self.contexts.get_mut(&id) {
            ctx.committed_len += new_phys.len();
        }

        if let Some(pid) = owner {
            let d = self.process_entry(pid).device_mut(dev_idx);
            d.committed += new_phys.len();
            d.working -= new_phys.len();
        }

        Ok(())
    }

    /// Finish restoration: transition context from replay to fully active.
    pub(crate) fn finish_restore(&mut self, id: ContextId) {
        if let Some(ctx) = self.contexts.get_mut(&id) {
            ctx.state = ContextState::Active;
        }
    }
}
