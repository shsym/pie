//! Suspension, eviction, and residency restoration for contexts.
//!
//! This module handles the lifecycle of GPU residency: suspending contexts
//! to CPU, restoring them from CPU, and replaying committed chains.

use std::collections::HashMap;
use std::time::Instant;
use anyhow::Result;
use serde::Serialize;

use crate::adapter::AdapterId;
use crate::process::ProcessId;
use crate::device::{self, DeviceId};
use crate::inference::brle::Brle;

use super::{CONTEXTS, Context, ContextId, ContextState, Record, ReplayFill, TokenInfo};
use super::kvcache::{self, PhysicalPageId, PageHash};
use super::manager::ContextManager;
use super::waitqueue::WaitNeeded;

// =============================================================================
// ReplayPlan — intermediate data for ensure_resident decomposition
// =============================================================================

/// Computed by `build_replay_plan`, consumed by `execute_replay`.
struct ReplayPlan {
    all_tokens: Vec<u32>,
    all_positions: Vec<u32>,
    all_masks: Vec<Brle>,
    adapters: Vec<(usize, Option<AdapterId>)>,
    matched_tokens: usize,
    kv_so_far: u32,
}

// =============================================================================
// Eviction & Suspension
// =============================================================================

impl ContextManager {
    /// Find the cheapest active context on a given device whose group
    /// utility is below `floor_utility`, or equal with a later creation time
    /// (FCFS tiebreaker: older processes are harder to evict).
    ///
    /// Tie-breaking among candidates:
    ///   1. Lowest priority first
    ///   2. At equal priority: newer process first (FCFS — older wins)
    ///   3. At equal priority & age: most GPU pages first (spread eviction)
    ///   4. Within same node: oldest context access (LRU)
    ///
    /// Unowned contexts (snapshots, orphans) have 0 priority via the
    /// fallback heuristic (no arbiter node), so they are naturally
    /// the cheapest victims — no separate tier needed.
    pub(crate) fn find_cheapest_victim(
        &self, dev_idx: usize, floor_utility: f64, requester: Option<ProcessId>,
    ) -> Option<ContextId> {
        let requester_birth = requester
            .and_then(|pid| self.arbiter.node_created_at(&pid));

        // (id, priority, birth, node_pages, last_access)
        let mut best: Option<(ContextId, f64, Option<Instant>, usize, Instant)> = None;

        for entry in CONTEXTS.iter() {
            let &(model_idx, ctx_id) = entry.key();
            if model_idx != self.model_idx { continue; }
            let ctx = entry.value();
            if ctx.state != ContextState::Active {
            // Also consider Suspended contexts that still hold working pages.
            // These occur when a failed restore couldn't swap back to CPU.
            if !(ctx.state == ContextState::Suspended && !ctx.working_pages.is_empty()) {
                continue;
            }
        }
            if !ctx.has_gpu_pages() { continue; }
            if ctx.device != Some(dev_idx as DeviceId) { continue; }
            // Skip contexts with in-flight working pages when CPU swap
            // cannot offload them — eviction would lose uncommitted KV data.
            if !ctx.working_pages.is_empty()
                && !self.devices[dev_idx].can_swap_working(ctx.working_pages.len())
            {
                continue;
            }

            let (priority, node_pages) = ctx.owner
                .map(|pid| {
                    (self.arbiter.priority(&pid, dev_idx), self.arbiter.node_pages(&pid, dev_idx))
                })
                .unwrap_or((0.0, 0));

            // Skip victims with strictly higher priority.
            // Equal priority victims ARE evictable — the anti-thrashing
            // invariant still holds: after R evicts V, R has p_R+n pages
            // while V has 0. V's post-allocation floor = w*(0+n) ≤ w*(p_R+n)
            // = R's new priority, so V cannot evict R back.
            if priority > floor_utility + 1e-9 { continue; }

            let victim_birth = ctx.owner
                .and_then(|pid| self.arbiter.node_created_at(&pid));

            let dominated = match &best {
                None => true,
                Some((_, best_u, best_birth, best_gp, best_t)) => {
                    if priority < *best_u {
                        true
                    } else if (priority - *best_u).abs() < 1e-9 {
                        // Among equal-priority candidates, prefer newest (most recently created)
                        match (victim_birth, best_birth) {
                            (Some(v), Some(b)) if v > *b => true,
                            (Some(v), Some(b)) if v < *b => false,
                            _ => {
                                // Same birth or unknown: fall back to page count / LRU
                                if node_pages > *best_gp {
                                    true
                                } else if node_pages == *best_gp {
                                    ctx.last_access < *best_t
                                } else {
                                    false
                                }
                            }
                        }
                    } else {
                        false
                    }
                }
            };

            if dominated {
                best = Some((ctx_id, priority, victim_birth, node_pages, ctx.last_access));
            }
        }

        best.map(|(id, _, _, _, _)| id)
    }

    pub(crate) async fn suspend_context(&mut self, id: ContextId) {
        let (working, tip, dev_idx) = {
            let ctx = match CONTEXTS.get(&(self.model_idx, id)) {
                Some(ctx) => ctx, None => return,
            };
            // Skip if context transitioned to InFlight between victim selection and now,
            // or is mid-replay (Restoring).  Restoring contexts have in-flight replay
            // forward passes that reference their GPU pages — eviction would cause
            // use-after-free on the GPU.
            if ctx.state == ContextState::InFlight || ctx.state == ContextState::Restoring { return; }
            if !ctx.has_gpu_pages() { return; }
            (ctx.working_pages.clone(), ctx.committed_tip, ctx.device.unwrap_or(0) as usize)
        };

        // Phase 1: Prepare swap — allocate CPU slots (does NOT free GPU pages yet)
        let swap_ops = if !working.is_empty() {
            match self.devices[dev_idx].swap_out_working(&working) {
                Ok(ops) => ops,
                Err(e) => {
                    tracing::warn!("Cannot swap working pages for {id}, discarding (will replay): {e}");
                    self.devices[dev_idx].free_working(&working);
                    Vec::new()
                }
            }
        } else {
            Vec::new()
        };

        // Phase 2: Execute D2H copy, then free GPU pages.
        // GPU pages must remain allocated during the copy to prevent
        // use-after-free corruption from page reallocation.
        if !swap_ops.is_empty() {
            #[derive(Serialize)]
            struct SwapOutRequest { phys_ids: Vec<u32>, slots: Vec<PhysicalPageId> }
            let request = SwapOutRequest {
                phys_ids: swap_ops.iter().map(|op| op.gpu_phys).collect(),
                slots: swap_ops.iter().map(|op| op.cpu_slot).collect(),
            };
            let _: Result<(), _> = device::call(dev_idx as DeviceId, "copy_d2h", &request).await;
            // Now safe to free GPU pages — D2H copy is complete.
            let gpu_pages: Vec<PhysicalPageId> = swap_ops.iter().map(|op| op.gpu_phys).collect();
            self.devices[dev_idx].free_working(&gpu_pages);
        }

        // Phase 3: Release committed chain refcounts
        if let Some(tip_hash) = tip {
            let dev = &mut self.devices[dev_idx];
            dev.release_chain(tip_hash);
            dev.remove_index_cache(tip_hash);
            dev.evict_unreferenced();
        }

        // Update context state
        if let Some(mut ctx) = CONTEXTS.get_mut(&(self.model_idx, id)) {
            ctx.working_cpu_slots = swap_ops.iter().map(|op| op.cpu_slot).collect();
            ctx.working_pages.clear();
            ctx.state = ContextState::Suspended;
            ctx.last_access = Instant::now();

            let owner = ctx.owner;
            let committed_len = ctx.committed_len;
            drop(ctx);
            if let Some(pid) = owner {
                self.arbiter.suspend(pid, dev_idx, committed_len, swap_ops.len());
            }
        }

        tracing::info!("Suspended context {} on model {}", id, self.model_idx);
    }

    /// Resolve physical page IDs and atomically pin the context as InFlight.
    ///
    /// This runs inside the actor, serialized with eviction operations.
    /// By setting InFlight here, we guarantee no eviction can happen between
    /// page resolution and the forward pass completing (cleared by clear_in_flight).
    pub(crate) fn get_physical_page_ids(&mut self, id: ContextId) -> Result<HashMap<DeviceId, Vec<PhysicalPageId>>> {
        let ctx = self.ctx(id)?;
        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let committed_len = ctx.committed_len;
        let working_len = ctx.working_pages.len();
        let tokens_filled_len = ctx.tokens_filled.len();
        let tip = ctx.committed_tip;
        let mut phys = if let Some(t) = tip {
            drop(ctx);
            self.devices[dev_idx].resolve_physical(t)
        } else {
            drop(ctx);
            Vec::new()
        };

        let committed_resolved = phys.len();

        // Sanity check: resolved committed pages must match committed_len
        if committed_resolved != committed_len {
            let chain_info = self.devices[dev_idx].debug_chain_info(tip.unwrap_or(0));
            eprintln!(
                "PAGE_COUNT_MISMATCH ctx={id} resolved={committed_resolved} committed_len={committed_len} \
                 working={working_len} tokens_filled={tokens_filled_len} tip={tip:?} chain=[{chain_info}]",
            );
        }

        let committed_phys_len = phys.len();
        if let Some(ctx) = CONTEXTS.get(&(self.model_idx, id)) {
            let wp = ctx.working_pages.len();
            phys.extend_from_slice(&ctx.working_pages);
            if wp != working_len {
                eprintln!(
                    "WORKING_PAGES_CHANGED ctx={id} initial={working_len} now={wp} \
                     committed_phys={committed_phys_len} final_phys={}", phys.len(),
                );
            }
        }

        let final_len = phys.len();
        // Log whenever working pages claim to exist but phys count doesn't match
        if working_len > 0 && final_len != committed_len + working_len {
            eprintln!(
                "PHYS_WORKING_MISMATCH ctx={id} committed_len={committed_len} \
                 committed_resolved={committed_resolved} working_len={working_len} \
                 final_phys={final_len} tip={tip:?}",
            );
        }

        let mut result = HashMap::new();
        if !phys.is_empty() {
            result.insert(dev_idx as DeviceId, phys);
        }
        Ok(result)
    }

    pub(crate) async fn ensure_resident(&mut self, id: ContextId) -> Result<Option<Vec<ReplayFill>>, WaitNeeded> {
        let ctx = self.ctx(id).map_err(WaitNeeded::Fatal)?;
        let tip = ctx.committed_tip;
        let working_cpu = ctx.working_cpu_slots.clone();
        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let was_suspended = ctx.state == ContextState::Suspended;
        let owner = ctx.owner;
        drop(ctx);

        // Fast path: not suspended, no work to do
        if !was_suspended && working_cpu.is_empty() {
            if let Some(mut ctx) = CONTEXTS.get_mut(&(self.model_idx, id)) {
                ctx.last_access = Instant::now();
            }
            return Ok(None);
        }

        eprintln!(
            "ENSURE_RESIDENT_SLOW ctx={id} was_suspended={was_suspended} \
             working_cpu={} tip={tip:?}",
            working_cpu.len(),
        );

        // Phase 1: Swap working pages from CPU back to GPU
        //
        // Track the working page count *before* Phase 1 so we can roll back
        // if Phase 2 fails.  restore_working_pages sets ctx.working_pages to
        // the newly-swapped pages, so anything beyond pre_phase1_len was
        // added by Phase 1.
        let pre_phase1_len = CONTEXTS.get(&(self.model_idx, id))
            .map(|c| c.working_pages.len()).unwrap_or(0);

        // Phase 1: restore working pages from CPU → GPU
        if !working_cpu.is_empty() {
            self.restore_working_pages(id, dev_idx, &working_cpu, owner).await?;
        }

        // Phase 2: Ensure committed chain is resident
        if let Some(tip_hash) = tip {
            let (_, discarded) = self.devices[dev_idx].classify_chain(tip_hash);

            if discarded.is_empty() {
                if was_suspended {
                    self.devices[dev_idx].acquire_chain(tip_hash);
                    // Rebuild index_cache entry that was removed by
                    // suspend_context → remove_index_cache. Without this,
                    // the next commit_pages → update_index_cache falls back
                    // to rebuild_page_table which filter_maps evicted pages,
                    // permanently undersizing the cache entry.
                    let _ = self.devices[dev_idx].resolve_physical(tip_hash);
                    if let Some(pid) = owner {
                        let committed_len = CONTEXTS.get(&(self.model_idx, id))
                            .map(|c| c.committed_len).unwrap_or(0);
                        self.arbiter.restore(pid, dev_idx, committed_len, 0);
                    }
                    // Safety clamp: if working pages are gone (swap failure
                    // during suspension freed them instead of swapping to CPU),
                    // tokens_filled references KV data that no longer exists.
                    // Clear to prevent kv_len inflation / FlashInfer OOB reads.
                    if let Some(mut ctx) = CONTEXTS.get_mut(&(self.model_idx, id)) {
                        if ctx.working_pages.is_empty() && !ctx.tokens_filled.is_empty() {
                            eprintln!(
                                "CLAMP_TOKENS_FILLED ctx={id} \
                                 committed_len={} tokens_filled={} \
                                 (swap failure: working pages gone)",
                                ctx.committed_len, ctx.tokens_filled.len(),
                            );
                            ctx.tokens_filled.clear();
                        }
                    }
                    // Debug: log context state at no-replay restore
                    if let Some(ctx) = CONTEXTS.get(&(self.model_idx, id)) {
                        eprintln!(
                            "NO_REPLAY_RESTORE ctx={id} committed_len={} \
                             tokens_filled={} working_pages={} working_cpu={}",
                            ctx.committed_len, ctx.tokens_filled.len(),
                            ctx.working_pages.len(), ctx.working_cpu_slots.len(),
                        );
                    }
                }
                if let Some(mut ctx) = CONTEXTS.get_mut(&(self.model_idx, id)) {
                    ctx.state = ContextState::Active;
                    ctx.last_access = Instant::now();
                }
                return Ok(None);
            }

            // Committed chain discarded — replay needed.
            if let Some(mut ctx) = CONTEXTS.get_mut(&(self.model_idx, id)) {
                ctx.state = ContextState::Restoring;
            }

            // Save committed metadata BEFORE build_replay_plan destructively
            // modifies it. If execute_replay fails (OOM), we must restore these
            // so the context doesn't permanently lose its committed state.
            let saved_committed_tip = CONTEXTS.get(&(self.model_idx, id))
                .and_then(|c| c.committed_tip);
            let saved_committed_len = CONTEXTS.get(&(self.model_idx, id))
                .map(|c| c.committed_len).unwrap_or(0);
            let saved_max_committed_position = CONTEXTS.get(&(self.model_idx, id))
                .and_then(|c| c.max_committed_position);
            let saved_tokens_filled: Vec<TokenInfo> = CONTEXTS.get(&(self.model_idx, id))
                .map(|c| c.tokens_filled.clone()).unwrap_or_default();

            if let Some((plan, prefix_hashes)) = self.build_replay_plan(id, dev_idx) {
                // Acquire refcounts for the prefix-matched pages BEFORE replay.
                if !prefix_hashes.is_empty() {
                    self.devices[dev_idx].longest_prefix_match(&prefix_hashes);
                }

                match self.execute_replay(id, dev_idx, plan, owner).await {
                    Ok(chunks) => {
                        // Replay succeeded — restore arbiter accounting.
                        if let Some(pid) = owner {
                            let matched_committed = CONTEXTS.get(&(self.model_idx, id))
                                .map(|c| c.committed_len).unwrap_or(0);
                            self.arbiter.restore(pid, dev_idx, matched_committed, 0);
                        }

                        // Restore partial page data (tokens_filled) that was
                        // cleared by build_replay_plan. The lineage only contains
                        // committed tokens, so replay doesn't cover the partial
                        // page. The working page KV data is still valid (restored
                        // by Phase 1), so these tokens still map to correct KV.
                        if let Some(mut ctx) = CONTEXTS.get_mut(&(self.model_idx, id)) {
                            ctx.tokens_filled = saved_tokens_filled;
                        }

                        if chunks.is_empty() {
                            if let Some(mut ctx) = CONTEXTS.get_mut(&(self.model_idx, id)) {
                                ctx.state = ContextState::Active;
                            }
                            return Ok(None);
                        }
                        return Ok(Some(chunks));
                    }
                    Err(e) => {
                        // Phase 2 failed — rollback prefix refcounts we just acquired.
                        if !prefix_hashes.is_empty() {
                            self.devices[dev_idx].release_chain(
                                *prefix_hashes.last().unwrap()
                            );
                        }
                        // Restore committed metadata that build_replay_plan
                        // destructively modified. Without this, the context
                        // permanently loses its committed state on OOM.
                        if let Some(mut ctx) = CONTEXTS.get_mut(&(self.model_idx, id)) {
                            ctx.committed_tip = saved_committed_tip;
                            ctx.committed_len = saved_committed_len;
                            ctx.max_committed_position = saved_max_committed_position;
                            ctx.tokens_filled = saved_tokens_filled;
                        }
                        // rollback_replay already freed replay-allocated pages.
                        // Swap Phase 1's working pages back to CPU.
                        if let Err(oom) = self.rollback_phase1_to_cpu(id, dev_idx, owner, pre_phase1_len).await {
                            tracing::error!("rollback_phase1_to_cpu OOM for ctx {id}: {oom:#}");
                        }
                        return Err(e);
                    }
                }
            }

            // All pages prefix-matched, no replay needed.
            // CRITICAL: acquire refcounts on the committed chain.
            // build_replay_plan → longest_prefix_length is read-only — it
            // does NOT acquire refcounts.  Without this, the dedup-shared
            // pages can be evicted by another context while this context's
            // forward pass reads them → use-after-free KV corruption.
            self.devices[dev_idx].acquire_chain(tip_hash);
            if let Some(pid) = owner {
                let committed_len = CONTEXTS.get(&(self.model_idx, id))
                    .map(|c| c.committed_len).unwrap_or(0);
                self.arbiter.restore(pid, dev_idx, committed_len, 0);
            }
            // Safety clamp: same as NRR path — if working pages are gone
            // (swap failure), clear stale tokens_filled.
            if let Some(mut ctx) = CONTEXTS.get_mut(&(self.model_idx, id)) {
                if ctx.working_pages.is_empty() && !ctx.tokens_filled.is_empty() {
                    eprintln!(
                        "CLAMP_TOKENS_FILLED ctx={id} \
                         committed_len={} tokens_filled={} \
                         (replay all-matched, swap failure)",
                        ctx.committed_len, ctx.tokens_filled.len(),
                    );
                    ctx.tokens_filled.clear();
                }
                ctx.state = ContextState::InFlight;
            }
            return Ok(None);
        }

        // No committed chain
        if let Some(mut ctx) = CONTEXTS.get_mut(&(self.model_idx, id)) {
            ctx.state = ContextState::Active;
            ctx.last_access = Instant::now();
        }
        Ok(None)
    }

    /// Phase 1: swap working pages from CPU back to GPU.
    async fn restore_working_pages(
        &mut self, id: ContextId, dev_idx: usize,
        working_cpu: &[PhysicalPageId], owner: Option<ProcessId>,
    ) -> Result<(), WaitNeeded> {
        let swap_ops = match self.devices[dev_idx].swap_in_working(working_cpu) {
            Ok(ops) => ops,
            Err(_) => {
                let needed = working_cpu.len();
                let gpu_pages = self.allocate_working_with_suspension(dev_idx, needed, owner).await?;
                self.devices[dev_idx].free_working(&gpu_pages);
                self.devices[dev_idx].swap_in_working(working_cpu)?
            }
        };

        if !swap_ops.is_empty() {
            #[derive(Serialize)]
            struct SwapInRequest { phys_ids: Vec<u32>, slots: Vec<PhysicalPageId> }
            let request = SwapInRequest {
                phys_ids: swap_ops.iter().map(|op| op.gpu_phys).collect(),
                slots: swap_ops.iter().map(|op| op.cpu_slot).collect(),
            };
            let _: () = device::call(dev_idx as DeviceId, "copy_h2d", &request).await
                .map_err(|e| anyhow::anyhow!("copy_h2d RPC failed: {e}"))?;

            // Free CPU slots AFTER the H2D copy completes. Before this point,
            // the CPU buffer data must remain intact — freeing earlier allows
            // another context's suspension to overwrite the slot via D2H copy.
            let cpu_slots: Vec<_> = swap_ops.iter().map(|op| op.cpu_slot).collect();
            self.devices[dev_idx].free_cpu_slots(&cpu_slots);

            let new_working: Vec<_> = swap_ops.iter().map(|op| op.gpu_phys).collect();
            let swapped_count = new_working.len();
            if let Some(mut ctx) = CONTEXTS.get_mut(&(self.model_idx, id)) {
                ctx.working_pages = new_working;
                ctx.working_cpu_slots.clear();
            }
            if let Some(pid) = owner {
                self.arbiter.add_working(pid, dev_idx, swapped_count);
            }
        }
        Ok(())
    }

    /// Phase 2a: flatten lineage, compute hashes, find prefix.
    ///
    /// Returns `(ReplayPlan, prefix_hashes)`. The prefix hashes are returned
    /// so the caller can acquire refcounts AFTER successful replay (transactional).
    /// `build_replay_plan` itself uses `longest_prefix_length` (read-only) to
    /// avoid refcount side-effects that would need rollback on failure.
    fn build_replay_plan(&mut self, id: ContextId, dev_idx: usize) -> Option<(ReplayPlan, Vec<PageHash>)> {
        let lineage = CONTEXTS.get(&(self.model_idx, id))
            .map(|ctx| ctx.lineage.clone()).unwrap_or_default();

        let mut all_tokens = Vec::new();
        let mut all_positions = Vec::new();
        let mut all_masks = Vec::new();
        let mut adapters = Vec::new();

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

        let page_aligned = (all_tokens.len() / self.page_size) * self.page_size;
        let (matched_pages, all_hashes) = if page_aligned > 0 {
            let hashes = self.devices[dev_idx].compute_page_hashes(
                &all_tokens[..page_aligned],
                &all_positions[..page_aligned],
                &all_masks[..page_aligned],
                0,
            );
            // Read-only prefix check — no refcount side-effects.
            let matched = self.devices[dev_idx].longest_prefix_length(&hashes);
            (matched, hashes)
        } else {
            (0, Vec::new())
        };

        let matched_tokens = matched_pages * self.page_size;
        let kv_so_far = matched_tokens as u32;

        // Prefix hashes for deferred refcount acquisition (caller acquires
        // after successful replay to keep the restore transactional).
        let prefix_hashes: Vec<PageHash> = all_hashes[..matched_pages].to_vec();

        if matched_pages > 0 {
            let prefix_hash_slice = &all_hashes[..matched_pages];
            let Some(&new_tip) = prefix_hash_slice.last() else { return None; };
            // Populate index_cache for the prefix tip (side-effect of resolve_physical).
            let _phys = self.devices[dev_idx].resolve_physical(new_tip);
            if let Some(mut ctx) = CONTEXTS.get_mut(&(self.model_idx, id)) {
                ctx.committed_tip = Some(new_tip);
                ctx.committed_len = matched_pages;
                // Reset max_committed_position to match the prefix we recovered.
                // Without this, replay's commit_pages rejects positions that are
                // <= the pre-eviction max_committed_position.
                ctx.max_committed_position = all_positions[..matched_tokens].iter().copied().max();
                // NOTE: tokens_filled is NOT cleared here. If all committed
                // tokens matched (no replay needed), tokens_filled still
                // references valid KV data in working pages restored by Phase 1.
                // It is cleared below only when replay IS needed.
            }
        } else {
            if let Some(mut ctx) = CONTEXTS.get_mut(&(self.model_idx, id)) {
                ctx.committed_tip = None;
                ctx.committed_len = 0;
                ctx.max_committed_position = None;
                ctx.tokens_filled.clear();
            }
        }

        if matched_tokens >= all_tokens.len() {
            return None;
        }

        // Replay IS needed: clear stale tokens_filled. These reference KV
        // from working pages that were freed during eviction. The replay
        // forward pass will recompute the KV data for these tokens.
        if let Some(mut ctx) = CONTEXTS.get_mut(&(self.model_idx, id)) {
            ctx.tokens_filled.clear();
        }

        Some((ReplayPlan { all_tokens, all_positions, all_masks, adapters, matched_tokens, kv_so_far }, prefix_hashes))
    }

    /// Phase 2b: build replay chunks for tokens after the matched prefix.
    ///
    /// Self-cleaning on failure: if allocation fails midway, frees any
    /// working pages allocated during this call and resets state to
    /// `Suspended` for a clean retry.
    async fn execute_replay(
        &mut self, id: ContextId, dev_idx: usize,
        plan: ReplayPlan, owner: Option<ProcessId>,
    ) -> Result<Vec<ReplayFill>, WaitNeeded> {
        let ReplayPlan { all_tokens, all_positions, all_masks, adapters, matched_tokens, mut kv_so_far } = plan;

        // Track how many working pages existed before this call so we
        // can rollback exactly the pages WE allocated on failure.
        let initial_working_len = CONTEXTS.get(&(self.model_idx, id))
            .map(|c| c.working_pages.len()).unwrap_or(0);

        // When Phase 1 restored working pages from CPU but committed pages
        // are discarded, those Phase 1 pages hold stale KV data.  Rather
        // than freeing them NOW (which causes CUDA errors from pool churn
        // at extreme contention), we keep them in ctx.working_pages but
        // exclude them from phys_ids.  After building all replay chunks we
        // drain them.

        let mut chunks = Vec::new();
        let mut offset = matched_tokens;
        let mut pages_added_to_arbiter = 0;

        while offset < all_tokens.len() {
            let adapter = adapters.iter().rev()
                .find(|(start, _)| *start <= offset)
                .and_then(|(_, a)| *a);

            let next_adapter_start = adapters.iter()
                .find(|(start, _)| *start > offset)
                .map(|(start, _)| *start)
                .unwrap_or(all_tokens.len());

            let chunk_end = next_adapter_start;
            let chunk_tokens = &all_tokens[offset..chunk_end];
            let chunk_positions = &all_positions[offset..chunk_end];
            let chunk_masks = &all_masks[offset..chunk_end];

            // Allocate pages for ALL tokens (including partial page) — the
            // model needs to write KV data for all of them.
            let total_pages = (chunk_tokens.len() + self.page_size - 1) / self.page_size;
            // But only FULL pages get committed. The partial page stays as a
            // working page with its KV data preserved in tokens_filled.
            let full_pages = chunk_tokens.len() / self.page_size;

            let new_pages = match self.allocate_working_with_suspension(dev_idx, total_pages, owner).await {
                Ok(pages) => pages,
                Err(e) => {
                    // Rollback: free working pages we allocated in THIS call.
                    self.rollback_replay(id, dev_idx, owner, initial_working_len, pages_added_to_arbiter);
                    return Err(e);
                }
            };

            {
                let mut ctx = CONTEXTS.get_mut(&(self.model_idx, id))
                    .ok_or_else(|| anyhow::anyhow!("Context lost during replay"))?;
                ctx.working_pages.extend(&new_pages);
                // NOTE: Do NOT touch tokens_buffered or tokens_filled here.
                // The inferlet may have pending data in these buffers from an
                // in-flight operation. Replay data goes directly into ReplayFill.
            }
            if let Some(pid) = owner {
                self.arbiter.add_working(pid, dev_idx, total_pages);
            }
            pages_added_to_arbiter += total_pages;

            // Build phys_ids EXCLUDING stale Phase 1 pages — only use
            // replay-allocated pages (those after initial_working_len).
            let phys_ids = {
                let mut all = if let Some(tip) = CONTEXTS.get(&(self.model_idx, id))
                    .and_then(|c| c.committed_tip) {
                    self.devices[dev_idx].resolve_physical(tip)
                } else { Vec::new() };
                if let Some(ctx) = CONTEXTS.get(&(self.model_idx, id)) {
                    all.extend_from_slice(&ctx.working_pages[initial_working_len..]);
                }
                all
            };

            let num_input = chunk_tokens.len() as u32;
            let total_kv = kv_so_far + num_input;
            let total_pages_for_fwd = phys_ids.len() as u32;
            let last_page_len = kvcache::compute_last_page_len(total_kv, total_pages_for_fwd, self.page_size as u32);

            chunks.push(ReplayFill {
                tokens: chunk_tokens.to_vec(),
                positions: chunk_positions.to_vec(),
                masks: chunk_masks.to_vec(),
                adapter,
                physical_page_ids: phys_ids,
                device_id: dev_idx as DeviceId,
                kv_len: kv_so_far,
                last_page_len,
                num_pages: full_pages as u32,
            });
            kv_so_far += num_input;
            offset = chunk_end;
        }

        // All chunks built. Now drain stale Phase 1 pages from the front
        // of ctx.working_pages so that commit_replay_chunk (which indexes
        // from position 0) sees only the replay-allocated pages.
        //
        // CRITICAL: also clear tokens_filled.  Those tokens reference KV data
        // from the freed Phase 1 working pages.  If left in tokens_filled,
        // kv_len (= committed_len * page_size + tokens_filled.len()) is
        // inflated by the stale count → wrong last_page_len and position IDs
        // → model reads past actual KV data → corruption.
        if initial_working_len > 0 {
            let stale_pages: Vec<PhysicalPageId> = {
                let mut ctx = CONTEXTS.get_mut(&(self.model_idx, id))
                    .ok_or_else(|| anyhow::anyhow!("Context lost during replay drain"))?;
                // Clear tokens_filled: these reference KV data from
                // the stale Phase 1 working pages being freed below.
                ctx.tokens_filled.clear();
                ctx.working_pages.drain(..initial_working_len).collect()
            };
            self.devices[dev_idx].free_working(&stale_pages);
            if let Some(pid) = owner {
                self.arbiter.remove_working(pid, dev_idx, initial_working_len);
            }
        }

        // Repopulate tokens_filled with the partial page remainder.
        //
        // The replay forward pass wrote KV data for ALL lineage tokens
        // (including the partial last page). commit_replay_chunk only
        // commits FULL pages. The remainder tokens beyond the last full
        // page boundary exist as KV data in a working page but are not
        // tracked by committed_len. We must record them in tokens_filled
        // so that kv_len = committed_len * page_size + tokens_filled.len()
        // accurately reflects the total KV entries.
        let total_committed_tokens: usize = matched_tokens
            + chunks.iter().map(|c| c.num_pages as usize * self.page_size).sum::<usize>();
        if total_committed_tokens < all_tokens.len() {
            let remainder = all_tokens.len() - total_committed_tokens;
            eprintln!(
                "REPOPULATE ctx={id} matched={matched_tokens} replay_committed={} \
                 total_committed={total_committed_tokens} all_tokens={} remainder={remainder}",
                chunks.iter().map(|c| c.num_pages as usize * self.page_size).sum::<usize>(),
                all_tokens.len(),
            );
            if let Some(mut ctx) = CONTEXTS.get_mut(&(self.model_idx, id)) {
                // tokens_filled was already cleared above (or by build_replay_plan).
                // Repopulate with the partial page remainder.
                for i in total_committed_tokens..all_tokens.len() {
                    ctx.tokens_filled.push(TokenInfo {
                        token: all_tokens[i],
                        position: all_positions[i],
                        mask: all_masks[i].clone(),
                        adapter: adapters.iter().rev()
                            .find(|(start, _)| *start <= i)
                            .and_then(|(_, a)| *a),
                    });
                }
                eprintln!(
                    "REPOPULATE_DONE ctx={id} tokens_filled_len={}",
                    ctx.tokens_filled.len(),
                );
            }
        }

        Ok(chunks)
    }

    /// Rollback working pages allocated during a failed `execute_replay`.
    /// Frees the pages, undoes arbiter accounting, and resets state to Suspended.
    fn rollback_replay(
        &mut self, id: ContextId, dev_idx: usize,
        owner: Option<ProcessId>, initial_working_len: usize, arbiter_pages: usize,
    ) {
        let to_free = {
            let mut ctx = match CONTEXTS.get_mut(&(self.model_idx, id)) {
                Some(ctx) => ctx,
                None => return,
            };
            let current_len = ctx.working_pages.len();
            if current_len <= initial_working_len {
                ctx.state = ContextState::Suspended;
                return;
            }
            let pages: Vec<PhysicalPageId> = ctx.working_pages.drain(initial_working_len..).collect();
            ctx.state = ContextState::Suspended;
            pages
        };

        self.devices[dev_idx].free_working(&to_free);
        if let Some(pid) = owner {
            self.arbiter.remove_working(pid, dev_idx, arbiter_pages);
        }
    }

    /// Rollback Phase 1 (restore_working_pages) when Phase 2 fails.
    ///
    /// After `rollback_replay` has cleaned up Phase 2's allocations, this
    /// swaps Phase 1's working pages back from GPU to CPU, preserving the
    /// data for a future restore attempt.  If the CPU swap fails (no CPU
    /// slots), returns an error — this is a true OOM and the owning process
    /// should be failed.
    async fn rollback_phase1_to_cpu(
        &mut self, id: ContextId, dev_idx: usize,
        owner: Option<ProcessId>, pre_phase1_len: usize,
    ) -> Result<(), anyhow::Error> {
        let phase1_pages = {
            let mut ctx = match CONTEXTS.get_mut(&(self.model_idx, id)) {
                Some(ctx) => ctx,
                None => return Ok(()),
            };
            let current_len = ctx.working_pages.len();
            if current_len <= pre_phase1_len {
                return Ok(());
            }
            ctx.working_pages.drain(pre_phase1_len..).collect::<Vec<_>>()
        };

        let n = phase1_pages.len();

        // Try to swap the pages back to CPU (allocates CPU slots, does NOT free GPU).
        let swap_ops = match self.devices[dev_idx].swap_out_working(&phase1_pages) {
            Ok(ops) => ops,
            Err(_) => {
                // No CPU slots — true OOM. Free the GPU pages and propagate error.
                self.devices[dev_idx].free_working(&phase1_pages);
                if let Some(pid) = owner {
                    self.arbiter.remove_working(pid, dev_idx, n);
                }
                anyhow::bail!("CPU page pool exhausted during rollback (OOM)");
            }
        };

        // Await D2H copy, then free GPU pages (same correctness pattern as suspend_context).
        if !swap_ops.is_empty() {
            #[derive(Serialize)]
            struct SwapOutRequest { phys_ids: Vec<u32>, slots: Vec<PhysicalPageId> }
            let request = SwapOutRequest {
                phys_ids: swap_ops.iter().map(|op| op.gpu_phys).collect(),
                slots: swap_ops.iter().map(|op| op.cpu_slot).collect(),
            };
            let _: Result<(), _> = device::call(dev_idx as DeviceId, "copy_d2h", &request).await;
            // Now safe to free GPU pages.
            let gpu_pages: Vec<PhysicalPageId> = swap_ops.iter().map(|op| op.gpu_phys).collect();
            self.devices[dev_idx].free_working(&gpu_pages);
        }

        // Store new CPU slots so the context can be restored again later.
        if let Some(mut ctx) = CONTEXTS.get_mut(&(self.model_idx, id)) {
            ctx.working_cpu_slots = swap_ops.iter().map(|op| op.cpu_slot).collect();
        }
        if let Some(pid) = owner {
            self.arbiter.remove_working(pid, dev_idx, n);
        }
        Ok(())
    }

    /// Commit a replay chunk using provided data directly, without touching tokens_filled.
    /// This avoids the race where replay overwrites the inferlet's in-flight buffer data.
    pub(crate) fn commit_replay_chunk(
        &mut self, id: ContextId, num_pages: u32,
        tokens: Vec<u32>, positions: Vec<u32>, masks: Vec<Brle>, adapter: Option<AdapterId>,
    ) -> Result<()> {
        if num_pages == 0 { return Ok(()); }

        let page_size = self.page_size;
        let ctx = self.ctx(id)?;

        let prev_hash = ctx.committed_tip.unwrap_or(0);
        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let old_tip = ctx.committed_tip;

        let working_phys: Vec<PhysicalPageId> = (0..num_pages as usize)
            .map(|i| ctx.working_pages[i])
            .collect();

        drop(ctx);

        let hashes = self.devices[dev_idx].compute_page_hashes(&tokens, &positions, &masks, prev_hash);

        let mut new_phys = Vec::new();
        let mut running_prev = prev_hash;
        for (i, &hash) in hashes.iter().enumerate() {
            let (phys, _freed) = self.devices[dev_idx].commit_working_page(hash, running_prev, working_phys[i]);
            new_phys.push(phys);
            running_prev = hash;
        }

        let new_tip = *hashes.last()
            .ok_or_else(|| anyhow::anyhow!("No page hashes computed during replay commit"))?;
        self.devices[dev_idx].update_index_cache(new_tip, old_tip, &new_phys);

        let owner = {
            let mut ctx = self.ctx_mut(id)
                .map_err(|_| anyhow::anyhow!("Context lost during replay commit"))?;

            // Remove working pages that were committed
            let mut sorted_indices: Vec<usize> = (0..num_pages as usize).collect();
            sorted_indices.sort_unstable();
            for &idx in sorted_indices.iter().rev() {
                ctx.working_pages.remove(idx);
            }

            ctx.committed_tip = Some(new_tip);
            ctx.committed_len += num_pages as usize;
            ctx.max_committed_position = positions.iter().copied().max()
                .or(ctx.max_committed_position);

            // NOTE: Do NOT append to lineage here. The replay data already
            // exists in the lineage — that's where build_replay_plan read it
            // from. Appending again would duplicate it on every evict/restore cycle.

            ctx.owner
        };

        if let Some(pid) = owner {
            self.arbiter.commit(pid, dev_idx, num_pages as usize);
        }

        Ok(())
    }

    pub(crate) fn finish_restore(&mut self, id: ContextId) {
        if let Some(mut ctx) = CONTEXTS.get_mut(&(self.model_idx, id)) {
            // Restoring → InFlight: skip the Active window entirely.
            // The process will call clear_in_flight after inference::submit.
            ctx.state = ContextState::InFlight;
        }
    }
}
