# KV Cache Contention Resolution

End-to-end documentation of how the Pie runtime resolves GPU memory
contention for KV cache pages across concurrent processes.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│ Process (inferlet)                                      │
│   ReservePages ──► Context Actor ──► DevicePageCache    │
│   CommitPages      (message loop)    (GPU/CPU pools)    │
│   EnsureResident       │                                │
│                   ┌────┴────┐                           │
│                   │ Arbiter │  (invested-importance)     │
│                   └────┬────┘                           │
│                        │                                │
│              ┌─────────┴──────────┐                     │
│              │    Wait Queue      │  (priority heap     │
│              │  (per-device)      │   with aging)       │
│              └────────────────────┘                     │
└─────────────────────────────────────────────────────────┘
```

All contention logic runs **inside the context actor** — a single-threaded
async message loop (one per model). This serializes all page accounting,
eviction, and state transitions, eliminating data races.

---

## 1. Context State Machine

Every context tracks its GPU residency via a four-state machine:

```
           ┌──────────────────────────────────────┐
           │                                      │
    ┌──────▼──────┐   get_physical_page_ids  ┌────┴─────┐
    │   Active    │ ─────────────────────────►│ InFlight │
    │  (on GPU)   │                           │(pinned)  │
    └──────┬──────┘                           └────┬─────┘
           │                                       │
   suspend_context                          clear_in_flight
           │                                       │
    ┌──────▼──────┐    ensure_resident       ┌─────▼─────┐
    │  Suspended  │ ─────────────────────────►│ Restoring │
    │ (on CPU)    │◄─────────────────────────┤           │
    └─────────────┘    rollback (on failure)  └───────────┘
```

| State | GPU pages | Evictable | Description |
|-------|-----------|-----------|-------------|
| **Active** | Working + committed chain | Yes | Ready for scheduling |
| **InFlight** | Working + committed chain | No | Forward pass executing — pinned |
| **Suspended** | None (or leaked working) | Only if working_pages present | Evicted to CPU / waiting for restore |
| **Restoring** | Partial (Phase 1 done) | No | Swap-in / replay in progress |

**Key invariant:** `InFlight` contexts are never evicted. `get_physical_page_ids`
atomically resolves page addresses AND pins the context as InFlight, preventing
eviction between page resolution and the actual GPU kernel execution.

---

## 2. Page Types

A context's GPU footprint consists of two kinds of pages:

### Working Pages
Uncommitted token data being filled by the inferlet. Stored in `ctx.working_pages`
(a `Vec<PhysicalPageId>` of GPU page IDs). These can be **swapped to CPU** via
D2H copy when the context is suspended, then restored via H2D copy. If CPU swap
is unavailable, it should fail the process with OOM error.

### Committed Pages
Immutable, content-addressed KV cache pages managed by `DevicePageCache`. Identified
by chained hashes (`PageHash`). Shared across contexts via refcounting — two contexts
with the same prefix share the same physical pages. When a context is suspended, its
committed chain **refcounts are released** (not copied to CPU). On restore, the chain
is re-acquired via prefix matching; any evicted pages are replayed from lineage.

---

## 3. Priority Model: Invested Importance

The **arbiter** (`arbiter.rs`) implements an invested-importance scheduling policy.

### Formula

```
π(process, device) = w_i × p_d
```

| Symbol | Meaning |
|--------|---------|
| `w_i` | Process weight — set by the DAG scheduler as `total_steps / remaining_steps` (SRPT). Processes closer to completion have higher weight. |
| `p_d` | Total pages held by the process on device `d` (committed + working). |
| `π` | Invested importance — the process's priority for keeping its pages. |

### Eviction Condition

Requester R can evict victim V **iff**:

```
π_V < π_R(p_R + n)  =  w_R × (p_R + n)
```

where `n` is the number of pages R is requesting. This is the **post-allocation
floor**: R's priority after the allocation succeeds.

### Anti-Thrashing Guarantee

After R evicts V:
- R holds `p_R + n` pages → `π_R = w_R × (p_R + n)`
- V holds 0 pages → V's post-allocation floor = `w_V × (0 + n) = w_V × n`
- Since `w_V × n ≤ w_R × (p_R + n)` (R won the comparison), V cannot immediately
  evict R back. **Eviction is monotonically forward-progressing.**

---

## 4. Eviction Cascade

When a context needs GPU pages and none are free, it triggers eviction:

### `allocate_working_with_suspension(dev, num_pages, requester)`

```
1. Try direct allocation from free pool
   └─ Success → return pages
   └─ Fail → enter eviction loop

2. Eviction loop:
   a. Compute requester_floor = w_R × (p_R + num_pages)
   b. find_cheapest_victim(dev, floor, requester)
      └─ No victim found → return Err(WaitNeeded::NeedPages)
      └─ Found victim → suspend_context(victim)
   c. Try allocation again
      └─ Success → return pages
      └─ Fail → repeat from (a)
```

### `find_cheapest_victim(dev, floor, requester)` — Tie-Breaking

Scans all **Active** contexts (and Suspended contexts with leaked working pages) on the device:

1. **Priority filter:** Skip if `π_victim > floor` (higher-priority than requester)
2. **Lowest priority first:** Prefer victim with smallest `π`
3. **FCFS tie-break:** Among equal-priority victims, prefer the newest process (older processes get priority — they're closer to completion)
4. **Page spread:** Prefer the process with the most GPU pages (spread eviction cost)
5. **LRU within node:** Prefer the least-recently-accessed context

Unowned contexts (snapshots, orphans) have `w = 0`, so they are naturally the cheapest victims.

### `suspend_context(id)` — Three Phases

```
Phase 1: Swap working pages GPU → CPU
   swap_out_working(working_pages)
    → allocates CPU slots, frees GPU pages
    → fires async D2H copy RPC (fire-and-forget)
    → on failure: discard working pages (will replay)

Phase 2: Release committed chain
   release_chain(committed_tip)
    → decrements refcounts on committed pages
   evict_unreferenced()
    → pages with refcount=0 become evictable (LRU pool)

Phase 3: Update context state
   ctx.working_cpu_slots = [new CPU slots]
   ctx.working_pages = []
   ctx.state = Suspended
```

---

## 5. Restoration

When a Suspended context needs to run again, `ensure_resident` restores it:

### Phase 1: Swap Working Pages CPU → GPU

```
restore_working_pages(id, dev, cpu_slots, owner):
   swap_in_working(cpu_slots)
    → allocates GPU pages, frees CPU slots
    → H2D copy RPC (awaited)
    → ctx.working_pages = [new GPU pages]
    → ctx.working_cpu_slots.clear()
```

If the GPU free pool is empty, the fallback path calls
`allocate_working_with_suspension` to evict other contexts first.

### Phase 2: Ensure Committed Chain Resident

```
classify_chain(committed_tip):
   → (resident, discarded) partition of the committed page chain

If discarded is empty:
   → All committed pages still on GPU → acquire refcounts → done

If discarded is non-empty:
   build_replay_plan(id, dev):
      → flatten lineage, compute hashes, find longest prefix match
      → prefix_hashes = pages already on GPU (reusable)
      → returns ReplayPlan with tokens/positions/masks to replay

   execute_replay(id, dev, plan, owner):
      → allocates working pages for each replay chunk
      → builds ReplayFill structs (data to re-execute on GPU)
      → on failure: rollback_replay + rollback_phase1_to_cpu
```

### Failure Rollback

If Phase 2 fails (no pages for replay):

```
1. rollback_replay(initial_working_len)
   → drains working_pages added DURING replay
   → frees those GPU pages
   → sets state = Suspended

2. rollback_phase1_to_cpu(pre_phase1_len)
   → drains Phase 1's working pages from ctx
   → swap_out_working(phase1_pages)  [allocates new CPU slots]
   → fires D2H copy RPC (fire-and-forget)
   → ctx.working_cpu_slots = [new CPU slots]
   → data preserved on CPU for future restore attempt
   → on CPU pool exhaustion: discard pages (full replay needed)
```

This sequence ensures **no data loss** and **no page leak** on restore failure.

---

## 6. Wait Queue

When allocation/restoration fails and no victim can be evicted, the request
enters a **per-device priority queue** (`BinaryHeap<PageWaiter>`).

### Waiter Types

| Type | Trigger | Action When Served |
|------|---------|-------------------|
| `Allocate` | `ReservePages` fails | Allocate working pages, extend `ctx.working_pages` |
| `Restore` | `EnsureResident` fails | Call `ensure_resident` again |

### Priority Ordering

Waiters are ordered by **effective floor** (highest first):

```
effective_floor = priority_floor + AGING_RATE × wait_seconds
```

- `priority_floor = w_i × (p_current + n)` — the requester's post-allocation importance
- `AGING_RATE = 1.0` — every second of waiting adds 1.0 to effective priority

**Starvation freedom:** even the lowest-priority waiter eventually reaches the
top of the queue as its age boost grows without bound.

### Queue Admission

Before trying allocation directly, the handler checks if the current request
should **yield to higher-priority waiters**:

```
should_queue = !has_committed
            && floor < top_waiter.effective_floor()
```

- Contexts with committed pages **skip the queue** — they have already invested GPU
  resources and must complete to free them.
- A request with lower floor than the queue head is enqueued immediately without
  attempting allocation (prevents queue starvation).

---

## 7. Drain: `try_serve_waiters`

Called after **any page-freeing event** (CommitPages, ReleasePages, ClearInFlight,
Destroy, FinishRestore):

```
loop:
   if no free GPU pages → break
   pop highest-priority waiter from queue
   match waiter:
     Allocate →
       try allocate_working(num_pages)
         success → extend ctx.working_pages, send Ok
         fail   → re-push waiter, break
     Restore →
       call ensure_resident(ctx_id)
         success → resolve pages, pin InFlight, send Ok
         NeedPages → re-push waiter, break
         Fatal → send error, continue
```

**Critical design constraint:** `try_serve_waiters` **never evicts** active
contexts. It only uses pages that are already free (from the free pool or
unreferenced committed pages). This prevents cascading evictions where a
waiter evicts a context that is about to complete its forward pass and
naturally free pages.

---

## 8. End-to-End Flow

Here's the complete lifecycle of a page allocation under contention:

```
Inferlet calls reserve_pages(ctx, 1)
    │
    ▼
ReservePages message arrives at context actor
    │
    ├─ Queue admission check: should I yield to waiters?
    │   └─ YES → enqueue immediately
    │   └─ NO  → try allocate_working_with_suspension
    │
    ▼
allocate_working_with_suspension(dev, 1, pid)
    │
    ├─ Free pool has pages → allocate, return Ok
    │
    ├─ No free pages → eviction loop:
    │   │   floor = w_R × (p_R + 1)
    │   │   victim = find_cheapest_victim(floor)
    │   │
    │   ├─ victim found → suspend_context(victim)
    │   │   │  swap working → CPU
    │   │   │  release committed chain
    │   │   │  try allocate again
    │   │   └─ success → return Ok
    │   │
    │   └─ no victim (all higher priority or InFlight)
    │       └─ return Err(WaitNeeded::NeedPages)
    │
    ▼
WaitNeeded::NeedPages
    │
    ▼
Enqueue PageWaiter::Allocate {
    floor = w_R × (p_R + 1),
    enqueued_at = now()
}
    │
    ... time passes, other contexts complete ...
    │
    ▼
ClearInFlight for a completed context
    │
    ▼
try_serve_waiters(device)
    │
    ├─ pages available?
    │   └─ pop highest-priority waiter (considering age boost)
    │   └─ allocate pages → send Ok to process
    │
    └─ process receives Ok, continues execution
```

---

## 9. Resolved Edge Cases

### Terminated Context While Queued
A waiter whose context was destroyed is skipped in `try_serve_waiters`:
```rust
if !CONTEXTS.contains_key(&(model_idx, context_id)) {
    response.send(Err("context destroyed while waiting"));
    continue;
}
```
If a context vanishes after allocation, the pages are immediately freed.

### Phase 1/Phase 2 Restore Failure
When `ensure_resident` Phase 2 fails, `rollback_phase1_to_cpu` swaps Phase 1's
working pages back to CPU (preserving data). The context returns to `Suspended`
with working data on CPU, ready for a future restore attempt.

### Suspended Contexts With Leaked Working Pages
If `rollback_phase1_to_cpu` cannot allocate CPU slots (CPU pool exhausted), it
discards the working pages. The context enters `Suspended` with non-empty
`working_pages`. Safety nets:
- `has_gpu_pages()` returns `true` for any context with non-empty `working_pages`
- `find_cheapest_victim` considers `Suspended` contexts with working pages as
  eviction candidates
- `suspend_context` can evict such contexts, freeing their stranded working pages

### InFlight Pinning (TOCTOU Prevention)
Between resolving physical page IDs and executing the GPU kernel, eviction must
be prevented. `get_physical_page_ids` atomically sets `state = InFlight`, and
`InFlight` contexts are excluded from `find_cheapest_victim` and `suspend_context`.

### CPU Swap Unavailability
If `can_swap_working(n)` returns false (no CPU budget or CPU pool full), that
context is skipped during victim selection. Its working pages would be lost on
eviction, so it's only evictable when CPU swap is available.

### Committed-Context Priority Bypass
Contexts with committed pages (`committed_len > 0`) skip the wait queue entirely.
They have already invested GPU resources (committed pages hold refcounts); delaying
them would waste those investments. They always attempt direct allocation with
eviction.
