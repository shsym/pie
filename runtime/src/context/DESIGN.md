# Context Module — KV Cache Management Design

Implementation blueprint for the `context` module. This replaces `context_legacy`.

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                      Context Actor (per model)                 │
│                                                                │
│  ┌────────────┐  ┌────────────┐  ┌───────────┐  ┌───────────┐ │
│  │ PageStore   │  │  Arbiter   │  │ try_alloc │  │try_restore│ │
│  │ (per device)│  │(DAG-aware) │  │  (FIFO)   │  │(pri heap) │ │
│  └────────────┘  └────────────┘  └───────────┘  └───────────┘ │
└────────────────────────────────────────────────────────────────┘
        ▲                                      │
        │  reserve / commit / destroy          │ set Pending
        │  get_physical_page_ids               │ via process state
        │  clear_pinned                        ▼
┌────────────────────────────────────────────────────────────────┐
│  Process (inferlet)                                            │
│  Single-threaded WASM — blocked on response channel when       │
│  enqueued. Cannot make other WIT calls until response arrives. │
└────────────────────────────────────────────────────────────────┘
```

---

## 1. State Model

### Process: 2 States
```
Pending ──(restored)──► Running ──(suspended/no victim)──► Pending
```
| State       | Description |
|-------------|-------------|
| **Running** | All contexts Active. Inferlet executing. |
| **Pending** | Contexts Suspended (or never allocated). Blocked on response channel. |

### Context: 3 States
```
Active ──(get_physical_page_ids)──► Pinned ──(clear_pinned)──► Active
  │                                    │
  │ (suspend: immediate)               │ (suspend: deferred via pending_suspend)
  ▼                                    ▼
Suspended                           Suspended (on clear_pinned)
```

| State         | Working Pages | Committed Chain | Evictable |
|---------------|--------------|-----------------|-----------|
| **Active**    | On GPU       | Refcounted      | Yes       |
| **Pinned**    | On GPU       | Refcounted      | Deferred only (flag) |
| **Suspended** | On CPU       | Released         | Nothing to evict |

**`pending_suspend: bool`** — Set on Pinned contexts when selected as victim.
Actual suspension deferred until `clear_pinned`.

---

## 2. Page Types

### Working Pages
- `working_pages: Vec<PhysicalPageId>` — GPU pages, exclusive to one context.
- On suspend: D2H copy to CPU. Stored in `working_pages_cpu: Vec<PhysicalPageId>`.
- On restore: H2D copy back. CPU slots freed.
- **If CPU swap pool full → OOM the process.** Working pages are NOT replayable.
- Developer contract: commit pages eagerly once full.

### Committed Pages
- Content-addressed via chained `PageHash` (BLAKE3).
- Shared across contexts via refcount in PageStore.
- `committed_tip: Option<PageHash>` — tip of the hash chain.
- On suspend: refcounts decremented. Pages with `rc=0` become evictable.
- On restore: longest prefix match → replay missing suffix.

---

## 3. Module Structure

```
context.rs              # Public API, Message enum, ServiceHandler
context/
├── DESIGN.md           # This file
├── pagestore.rs        # PageStore: CAS cache + physical page pools (per device)
├── arbiter.rs          # Invested-importance scheduling (DAG-aware)
├── contention.rs       # Eviction, suspension, try_alloc queue
└── restore.rs          # Restoration, replay planning, try_restore queue
```

### pagestore.rs — Per-Device Page Cache

```rust
pub type PageHash = u64;
pub type PhysicalPageId = u32;

pub struct PageStore {
    page_size: usize,

    // Content-addressed storage: hash → (gpu_phys, refcount)
    pages: FxHashMap<PageHash, (PhysicalPageId, usize)>,
    // Backward chain links: hash → prev_hash (0 = root)
    chain: FxHashMap<PageHash, PageHash>,

    // Physical pools
    gpu: PagePool,    // GPU free list
    cpu: PagePool,    // CPU free list (working page swap only)
}
```

Key operations:
| Method | Description |
|--------|-------------|
| `alloc_gpu_pages(n)` | Alloc n GPU pages from free pool |
| `free_gpu_pages(pages)` | Return GPU pages to free pool |
| `commit_working(hash, prev, phys)` | Promote working → committed (dedup-aware) |
| `acquire_chain(tip)` | Increment refcounts root→tip |
| `release_chain(tip)` | Decrement refcounts, return rc=0 hashes |
| `evict_unreferenced()` | Free GPU pages for rc=0 committed pages |
| `swap_out(gpu_pages)` | Alloc CPU slots, build D2H SwapOps |
| `swap_in(cpu_slots)` | Alloc GPU pages, build H2D SwapOps |
| `longest_prefix_length(hashes)` | Read-only prefix match (for restore planning) |
| `compute_page_hashes(tokens, pos, masks, prev)` | Chained content hashing |

### arbiter.rs — DAG-Aware Scheduling

```rust
pub struct Arbiter {
    entries: FxHashMap<ProcessId, ProcessEntry>,
}

struct ProcessEntry {
    weight: f64,              // w_i from workflow.rs (SRPT: total/remaining)
    devices: HashMap<DeviceId, DevicePages>,
    created_at: Instant,
}

struct DevicePages {
    working: usize,
    committed: usize,
}
```

Priority formula: `π(process, device) = w_i × p_d`
where `p_d = working + committed` pages on device d.

Victim selection: `π_victim < π_requester_post_allocation`

Tie-breaking (cheapest first):
1. Lowest π
2. Newer process first (FCFS: older gets priority).

---

## 4. Contention Protocol

### reserve_pages(ctx, n) — Full Flow

```
0. SUSPENSION CHECK: If ctx.state == Suspended
   → find RestoreWaiter for this process in try_restore
   → add this alloc request to RestoreWaiter.pending_allocs (hold response channel)
   → return

1. FIFO GATE: If try_alloc is non-empty
   → enqueue this request at tail of try_alloc
   → return (process stays Running, awaits response channel)

2. PRIORITY GATE: Compute requester_floor = w_R × (p_R + n)
   If requester_floor < try_restore.peek().effective_priority
   → suspend ALL of requester's contexts (Active immediately, Pinned deferred)
   → set requester state = Pending
   → Create RestoreWaiter and enqueue in try_restore
     (with pending_allocs = this request, pending_pinned_count = number of Pinned contexts)
   → return

3. TRY ALLOCATE from free pool
   → success: return pages

4. EVICT unreferenced committed pages (rc=0)
   → retry alloc, if success: return pages

5. EVICTION LOOP
   loop:
     a. SELECT VICTIM via Arbiter floor rule
        → no victim found: break → goto step 6
     b. SUSPEND VICTIM
        For each of victim's contexts:
          Active → suspend immediately (swap working→CPU, release chain + evict_unreferenced)
          Pinned → set pending_suspend = true (deferred)
        Set victim process state = Pending
        Enqueue victim in try_restore (empty pending_allocs, pending_pinned_count = number of Pinned contexts)
     c. If ALL of victim's contexts were Pinned:
          // No pages actually freed immediately. Stop looping to avoid
          // cascading suspensions that free nothing.
          break → enqueue in try_alloc (FIFO), return
     d. Retry alloc → if success: return pages
     // else: loop again only if there were no Pinned contexts in this eviction, try another victim

6. NO VICTIM / ALL PINNED, REQUESTER LOSES
   Suspend ALL of requester's contexts (Active immediately, Pinned deferred)
   Set requester state = Pending
   Enqueue in try_restore (with pending_allocs, pending_pinned_count = number of Pinned contexts)
   Return
```

### suspend_context(ctx_id) — Three Phases
```
Phase 1: Swap working pages GPU → CPU
   swap_out(working_pages) → CPU slots + SwapOps
   fire D2H copy RPC (fire-and-forget)
   free GPU pages
   ctx.working_pages_cpu = new CPU slots
   ctx.working_pages.clear()
   If CPU pool full → OOM the owning process

Phase 2: Release committed chain
   release_chain(committed_tip) → decrement refcounts
   * it does not clean up unreferenced (rc=0) pages. It's the role of reserve_pages.
   Remove committed_tip

Phase 3: Update state
   ctx.state = Suspended
   Update Arbiter: zero this process's DevicePages on this device
```

### clear_pinned(ctx_id) — Deferred Suspension
```
if ctx.pending_suspend:
    suspend_context(ctx_id)
    ctx.pending_suspend = false
    decrement owning process's RestoreWaiter.pending_pinned_count
    drain_queues()
else:
    ctx.state = Active
```

---

## 5. Queue Drain Protocol

### drain_queues() — Called on Every Page-Free Event

Trigger points:
- `clear_pinned` with `pending_suspend` (deferred suspension completes)
- Process termination -> context destruction
- Working page commit (dedup hit frees working page)

```
drain_queues():
    // Phase 1: try_alloc (FIFO, head-of-line blocking)
    while let Some(head) = try_alloc.front():
        if free_pages[head.device] >= head.num_pages:
            pop head → allocate pages → send response
        else:
            return   // exit drain_queues

    // Phase 2: try_restore (priority heap, checks ALL devices)
    while let Some(top) = try_restore.peek():
        if top.pending_pinned_count > 0:
            break   // still has Pinned contexts clearing, can't restore yet
        can_restore = for ALL devices d that process has contexts on:
            required[d] = working_cpu_count[d]
                        + pages_needing_replay[d]
                        + pending_alloc_pages[d]  // from RestoreWaiter.pending_allocs
            available[d] >= required[d]
        if can_restore:
            pop top → restore_process(top)
        else:
            break
```

### try_alloc Queue (FIFO)
```rust
struct AllocWaiter {
    context_id: ContextId,
    device: DeviceId,
    num_pages: usize,
    response: oneshot::Sender<Result<()>>,
}
// VecDeque<AllocWaiter> — single global FIFO, head-of-line blocking.
```

Only processes that passed the priority gate AND won eviction comparison can enter.
Pre-approved — no re-prioritization needed.

### try_restore Queue (Priority Heap + Aging)
```rust
struct RestoreWaiter {
    process_id: ProcessId,
    priority_floor: f64,
    enqueued_at: Instant,
    /// Number of Pinned contexts still awaiting clear_pinned.
    /// Restore is blocked until this reaches 0.
    pending_pinned_count: usize,
    /// Pending alloc requests to replay after restoration.
    /// Response channels are held here — the process is blocked awaiting them.
    pending_allocs: Vec<AllocWaiter>,
}
// BinaryHeap<RestoreWaiter> — global (checks all devices on drain)
```

Ordering: `effective_priority = priority_floor + AGING_RATE × wait_seconds`
AGING_RATE = 0.01 — starvation freedom.

After `restore_process` completes, pending allocs are re-attempted. If allocation
succeeds, the response channel fires Ok. The process unblocks and resumes.

---

## 6. Restoration

### restore_process(pid)
```
for each context in process:
    restore_context(ctx)
set process state = Running
replay pending_allocs from RestoreWaiter (allocate + send response)
```

### restore_context(ctx)
```
Phase 1: Swap in working pages
    alloc GPU pages (exact count = working_pages_cpu.len())
    H2D copy from CPU (fire and forget)
    free CPU slots
    ctx.working_pages = new GPU pages
    ctx.working_pages_cpu.clear()

Phase 2: Rebuild committed chain
    walk lineage → compute page hashes
    longest_prefix_length(hashes) → prefix_len
    acquire refcounts for prefix hashes
    if prefix_len == total → done (no replay needed)
    build replay plan for hashes[prefix_len..]
    execute replay: alloc working → forward pass → commit

set ctx.state = Active
Update Arbiter: set DevicePages for this process/device to new working + committed counts
```

**Invariant**: Restoration never evicts. Admission check guarantees pages available.

---

## 7. WIT Host Function Pattern

Every WIT function that requires a live process:

```rust
async fn wit_reserve_pages(pid: ProcessId, ctx_id: ContextId, n: u32) -> Result<()> {
    // Actor holds the response channel if enqueued (try_alloc or try_restore).
    // Resolves when: direct alloc, drain serves try_alloc, or process restored.
    context::reserve_pages(model_idx, ctx_id, n).await
}
```

**No `wait_if_pending` needed.** The process is single-threaded (WASM). When
`reserve_pages` gets enqueued, the process is blocked on the response channel
`.await`. It cannot make any other WIT calls. When the actor serves the request
(from try_alloc or after restoration from try_restore), it sends Ok on the
channel and the process resumes — already Running.

---

## 8. Anti-Thrashing Guarantee

After Process A evicts Process B:
- `π_A = w_A × (p_A + n)` (A holds more pages now)
- `π_B = w_B × 0` (B holds nothing)
- B's post-restoration floor = `w_B × n_B`
- Since `w_B × n_B ≤ w_A × (p_A + n)` (A won), B cannot evict A back.
- **Eviction is monotonically forward-progressing.**

---

## 9. Key Differences from context_legacy

| Aspect | context_legacy | context (new) |
|--------|---------------|---------------|
| Eviction unit | Context | Process (all contexts) |
| InFlight handling | Full state with TOCTOU prevention | Deferred via `pending_suspend` flag on Pinned contexts |
| Wait queues | Single queue (Allocate + Restore) | Two queues: try_alloc (FIFO) + try_restore (heap) |
| Context states | 4 (Active/InFlight/Suspended/Restoring) | 3 (Active/Pinned/Suspended) |
| Process states | Implicit | Explicit 2-state (Running/Pending) |
| Blocking model | Response channels only | Response channels (process blocked on oneshot) |
| CPU swap failure | Leaked working pages fallback | OOM (working pages not replayable) |
| Restore trigger | `try_serve_waiters` at 6+ points | `drain_queues` on page-free events |
| Queue admission | Per-context committed bypass | Priority gate (requester vs try_restore head) + FIFO gate (try_alloc non-empty) |
