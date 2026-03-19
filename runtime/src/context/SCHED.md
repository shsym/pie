# Credit-Market Scheduling via Knapsack Auction

A mechanism-design approach to GPU memory scheduling for inference serving.

---

## 1. Problem Statement

We allocate a perishable good — GPU KV-cache pages, renewed each batch step — among heterogeneous buyers (inference contexts) with private, time-varying valuations and a hard capacity constraint.

**Desiderata:**

- **Allocative efficiency.** Highest-value contexts are served each step.
- **Truthfulness.** No context gains by misreporting its value.
- **Conservation.** Total credits are preserved (modulo real compute cost). No bankruptcy spiral.
- **Stability.** No evict→restore→re-evict thrashing.
- **Simplicity.** Minimal moving parts. Analytically tractable for formal results.

**Core claim.** A single mechanism — a greedy knapsack auction with critical-value payments and endowment-proportional dividends, composed with Shapley cost-sharing for prefix colocation — achieves all five properties with three equations and six API functions.

---

## 2. Model

### 2.1 Entities

**Process.** A wallet. Holds a credit balance and owns one or more contexts. No scheduling state of its own — a process is alive while it has at least one context.

**Context.** The schedulable unit. Has a page count `n_i`, a private value `v_i` (per page, per step), and a state: ADMITTED (pages on GPU, running) or EXCLUDED (pages off GPU, idle). Each context belongs to exactly one process and draws from its balance.

**Device.** A GPU with `C_d` page slots. Each device runs an independent auction. Multiple tiers (GPU, CPU cache) each run their own auction.

### 2.2 Credits

Each credit represents one page of compute. A process admitted with token budget `T` receives an endowment:

```
E = ⌈T / page_size⌉   credits
```

Credits are destroyed only by the **make cost**: producing a new KV page via forward pass costs 1 credit. All other credit flows (rent, dividends) are conservative — they redistribute but do not create or destroy.

### 2.3 Invariant

At all times:

```
Σ_i balance_i  =  Σ_i endowment_i  −  M(t)
```

where `M(t)` is the cumulative make cost (total new pages ever produced across all contexts). This is an accounting identity, not an approximation.

---

## 3. Mechanism: Per-Step Knapsack Auction

Each batch step, each device runs the following procedure independently.

### 3.1 Bid Collection

Each context `i` on device `d` submits a bid `b_i` representing its value per page per step. The SDK provides a default (Section 5) but programs may override it.

### 3.2 Greedy Allocation

Sort contexts by `b_i` descending. Pack greedily until capacity is full:

```
ADMITTED ← ∅
used ← 0

for ctx in sorted(contexts, key=bid, descending):
    if used + ctx.pages ≤ C_d:
        ADMITTED ← ADMITTED ∪ {ctx}
        used ← used + ctx.pages
    else:
        marginal ← ctx
        break

EXCLUDED ← all contexts ∉ ADMITTED
```

At most one context is fractionally admissible (the marginal context). We round down: the marginal context is excluded. This wastes at most `max_i(n_i)` pages of capacity.

### 3.3 Critical-Value Payments

Each admitted context pays its **critical value** — the minimum bid at which it would still be admitted, given all other bids. This is the standard truthful payment rule for single-parameter domains (Lehmann, O'Callaghan, Shoham 2002).

For the greedy-by-density knapsack:

```
payment_i = critical_bid_i × n_i
```

where `critical_bid_i` is the highest bid at which context `i` would be displaced from the packing. For contexts well above the margin, this equals the marginal context's bid. For contexts near the margin, it may differ.

**Property:** Truthful bidding is a dominant strategy. Each context's payment depends only on others' bids, so no context can lower its payment by misreporting.

### 3.4 Endowment-Proportional Dividend

Total revenue `R = Σ_{i ∈ ADMITTED} payment_i` is redistributed to **all** contexts (admitted and excluded) proportional to their initial endowment:

```
dividend_i = (E_i / Σ_j E_j) × R
```

**Why endowment, not balance or per-context uniform?**

- **Per-context uniform** (`R/N`) is vulnerable to Sybil attack: a process splits into many tiny contexts to harvest more dividends.
- **Balance-proportional** creates intertemporal coupling: your bid affects the clearing price, which affects revenue, which affects dividends proportional to your balance. This breaks DSIC in the repeated game.
- **Endowment-proportional** is fixed at creation, independent of current behavior. Cannot be gamed by splitting (total endowment is conserved). Your dividend is independent of your bid. DSIC is preserved.

### 3.5 Net Balance Update

Each step, for each context:

```
balance_i ← balance_i − payment_i + dividend_i − make_cost_i

where:
    payment_i  = critical_bid_i × n_i    (if ADMITTED, else 0)
    dividend_i   = (E_i / Σ_j E_j) × R    (always)
    make_cost_i = max(0, total_pages_i − pages_paid_for_i)    (new pages only)
```

`pages_paid_for_i` is a high-water mark. Restoration after exclusion is free — the context already paid when it first produced those pages.

### 3.6 Anti-Thrashing

A context near the clearing price can oscillate between ADMITTED and EXCLUDED across consecutive steps. Each transition has a physical cost (GPU↔CPU page transfer). Three mechanisms prevent thrashing without distorting bids:

1. **Make cost.** Each restore cycle consumes credits (1 per replayed page). A thrashing context burns through its budget faster than a stable one, naturally reducing its bid and excluding it.

2. **FCFS tiebreaker.** At equal bids, the eviction victim is the context whose owning process was spawned most recently. Incumbents have deterministic priority — no bid inflation needed.

3. **Admission check.** A suspended context is only restored when the device has enough free pages for working + replay + deferred ops, the process can afford the deferred ops' make cost, and the process can afford at least one step of rent at the current clearing price.

4. **Default eviction.** Each tick, contexts whose process cannot afford full rent are flagged `defaulted`. Defaulted contexts are evicted first, regardless of bid. While suspended, the process accumulates dividends; the context is restored only when the balance recovers enough to pay rent.

**Property:** Anti-thrashing is achieved without modifying the bid signal. The auction uses raw bids; stability comes from economic cost (make cost), deterministic tiebreaking (FCFS), and automatic default enforcement.

---

## 4. Multi-Device Placement with Shapley Cost-Sharing

### 4.1 The Separation Principle

Two concerns, two mechanisms:

| Concern | Mechanism | Timescale |
|---------|-----------|-----------|
| **Placement**: which device holds a context? | Shapley cost-sharing (planner) | Periodic (every K steps) |
| **Admission**: does a context run this step? | Knapsack auction (market) | Every step |

Placement is a public-goods problem (shared prefixes are non-rival within a device). Markets fail on public goods. Admission is a private-goods problem (GPU pages are rival). Markets succeed here.

### 4.2 Effective Pages via Shapley Values

For tree-structured KV caches, each shared prefix segment `s` has a set of contexts sharing it. The Shapley value — the unique cost-sharing rule satisfying efficiency, symmetry, and additivity (Shapley 1953) — simplifies to equal-split per segment:

```
effective_pages(i, d) = unique_pages_i + Σ_s [pages_s / refcount_s(d)]
```

where `s` ranges over shared prefix segments on device `d`, and `refcount_s(d)` is the number of contexts sharing segment `s` on device `d`.

### 4.3 Placement Algorithm

For each unplaced or migration-candidate context:

```
assign context i to:

    argmin_d [ effective_pages(i, d) × price_prev(d)  +  migration_cost(i, d) ]

where:
    migration_cost(i, d) = transfer_pages(i) × MAKE_COST   if d ≠ current_device(i)
                         = 0                                 if d = current_device(i)
```

`price_prev(d)` is the clearing price from the most recent auction on device `d`. Migration only occurs when rent savings exceed transfer cost.

### 4.4 Prefix Colocation Emerges

No explicit colocation rule is needed. Colocation reduces `effective_pages`, which reduces rent, which improves competitive position. Example:

```
Context A: 200 unique pages, 500-page shared prefix with B
Context B: 150 unique pages, 500-page shared prefix with A

Colocated on one device:
    effective_A = 200 + 500/2 = 450
    effective_B = 150 + 500/2 = 400
    total = 850

Split across devices:
    effective_A = 200 + 500/1 = 700
    effective_B = 150 + 500/1 = 650
    total = 1350
```

The placement algorithm discovers colocation because it minimizes cost.

### 4.5 Auction Uses Effective Pages

The per-device auction (Section 3) uses `effective_pages_i` in place of raw `n_i` for sorting, payment computation, and dividend impact. This is the interface between the two levels.

### 4.6 Revenue Pooling

Revenue is pooled globally across all devices. A context's dividend is `(E_i / Σ_j E_j) × R_global`. This creates a mild cross-subsidy (busy devices fund idle-device dividends) but decouples the program's income from its placement, which the program cannot control. The alternative — per-device pools — would make the dividend placement-dependent, coupling the two levels and complicating the theoretical story.

---

## 5. Optimal Bid

### 5.1 Derivation

A context's choice each step is binary: be admitted (compute, pay rent, forgo dividend net of payment) or be excluded (idle, earn dividend). The indifference bid — the value at which a context is indifferent — is:

```
bid_i = (balance_i / horizon_i − dividend_i) / effective_pages_i
```

where `horizon_i` is the expected remaining steps.

**Interpretation:** `balance_i / horizon_i` is the per-step value of computing (budget spread over remaining lifetime). `dividend_i` is the opportunity cost of computing (forgoing the dividend difference between being excluded and admitted). The difference, normalized by pages, is the net value per page.

### 5.2 Properties

- When `dividend ≈ 0` (low load), this reduces to `balance / (horizon × pages)` — matching the original system's bid formula.
- When `dividend > balance/horizon` (heavy load), the bid goes negative. The context prefers exclusion. **The compute-or-wait decision is embedded in the bid**, not computed separately.
- Truthful reporting of this value is a dominant strategy under critical-value payments.

### 5.3 Horizon Estimation

The bid quality depends on `horizon_i`. The SDK provides defaults:

| Source | Horizon |
|--------|---------|
| Explicit `with_horizon(n)` | `n` steps |
| `max_tokens` set | `max_tokens − tokens_generated` |
| Neither | 4096 (conservative default) |

Programs with better information can override. The mechanism is truthful given the declared value — horizon error is an information problem, not a mechanism failure.

---

## 6. Tiered Storage

When a context is excluded from GPU, its pages descend through a storage hierarchy:

| Tier | Medium | Restoration Cost | Capacity |
|------|--------|-----------------|----------|
| GPU | HBM | 0 (already there) | `C_d` pages |
| CPU | Host DRAM | ~0.16ms/page (PCIe) | `C_cpu` pages |
| Evicted | None | ~10ms/page (recompute) | Unlimited |

**Each tier runs a separate auction.** A context excluded from GPU may still win CPU cache if its bid exceeds the CPU clearing price. The CPU clearing price is naturally lower because holding a CPU-cached page is less valuable (it only saves restoration cost, not compute):

```
CPU bid_i = restore_cost_saved_i / effective_pages_i
          = (recompute_cost − transfer_cost) / effective_pages_i
```

This replaces the original design's `CPU_TIER_DISCOUNT` with a market-derived price.

---

## 7. Tool-Call Yielding

When a context enters an external wait (tool call, API call), its per-step compute value drops to zero. The decision is binary under critical-value payments:

- **Hold** (keep generation bid): pays `rent × pages` per step but resumes instantly.
- **Fold** (bid zero): pays nothing but risks restore-queue delay.

Under Vickrey payments, the process pays the clearing price regardless of its bid level. Bidding above the clearing price does not increase the payment — it only reduces eviction risk. Therefore, holding and folding have the same per-step cost for any bid above the margin.

### 7.1 Affordability Threshold

The process should hold iff its competitiveness surplus covers the idle rent. From the bid formula (§5.1):

```
generation_bid = (balance / remaining − dividend) / pages
```

Requiring that the process remain competitive after paying idle rent for `W` steps:

```
balance − rent × pages × W  >  (rent × pages + dividend) × remaining
```

Substituting `balance − (rent × pages + dividend) × remaining = remaining × pages × (bid − rent)`:

```
W  <  remaining × (bid − rent) / rent
```

Or equivalently:

```
W / remaining  <  (bid − rent) / rent
```

The right-hand side is the process's **competitiveness surplus** — how far above the clearing price its bid sits, as a fraction of the clearing price.

### 7.2 Properties

- **Endogenous threshold.** No tuning parameter. The hold/fold boundary is determined entirely by the process's market position.
- **Well above margin** (`bid = 5 × rent`): can idle for 4× its remaining horizon — effectively always holds.
- **Near margin** (`bid ≈ rent`): surplus is near zero — folds immediately.
- **Below margin** (`bid < rent`): already being evicted — fold is the only option.
- **No contention** (`rent = 0`): residency is free — always hold.

### 7.3 Role of Latency

The system exposes `latency(ctx)`, the EWA-smoothed per-tick wall-clock time of the device. This converts the expected wait duration (seconds) to market steps:

```
W = expected_wait_seconds / latency
```

Without this signal, the process would need a hardcoded step-duration assumption. With it, the threshold adapts to actual device speed.

### 7.4 Information Asymmetry

The process does not observe the restore-queue depth or expected restoration delay. This asymmetry is intentional: the hold/fold decision is framed as **budget affordability** rather than restoration cost estimation. The process asks "can I afford the insurance premium?" (observable) instead of "how expensive would eviction be?" (unobservable).

The bid is restored automatically when the wait completes.

---

## 8. Lifecycle

### 8.1 Admission

A new process arrives with token budget `T`. The system creates a wallet with endowment `E = ⌈T / page_size⌉` and a first context. The context enters the auction immediately. No admission gate — if the system is overloaded, the clearing price is high and a low-value newcomer is simply not admitted. If idle, the price is near zero and the newcomer runs for free.

### 8.2 Steady State

```
┌─────────────────────────────────────────────────────┐
│  BATCH STEP                                         │
│                                                     │
│  1. Contexts submit bids                            │
│     SDK default: (balance/horizon − dividend) / pages │
│     or: program overrides with custom bid            │
│                                                     │
│  2. Placement check (every K steps)                 │
│     Shapley cost → assign to optimal device          │
│     Migrate if rent savings > transfer cost          │
│                                                     │
│  3. Auction clears (per device, per tier)            │
│     Greedy pack by bid descending                    │
│     Critical-value payments                          │
│     FCFS tiebreaker for incumbents                   │
│                                                     │
│  4. Payments and dividends                             │
│     Admitted: balance −= payment                     │
│     Everyone: balance += endowment-proportional share│
│                                                     │
│  5. Make cost for new pages                          │
│     balance −= new pages above high-water mark       │
│                                                     │
│  6. Page movement                                    │
│     Newly excluded: GPU → CPU cache or evict         │
│     Newly admitted: restore from CPU or recompute    │
│                                                     │
│  7. Forward pass                                     │
│     Admitted contexts run one decode step             │
│     Excluded contexts idle                           │
│                                                     │
│  8. Publish signals                                  │
│     price(device, tier), dividend(), balance()         │
└─────────────────────────────────────────────────────┘
```

### 8.3 Budget Exhaustion

A context's balance drains through make costs (permanent) and net rent (payment minus dividend). Lifetime:

```
lifetime ≈ E / (make_rate + net_rent_rate)
```

As balance approaches zero, the bid approaches zero, and the context is naturally excluded. While excluded, it earns dividend and never reaches exactly zero. This is the **no-bankruptcy guarantee**.

A context terminates when computation completes (EOS token, max_tokens reached) or when balance is too low to ever be competitively re-admitted.

### 8.4 Multi-Context Processes

A process with multiple contexts (multi-turn agent, parallel generations) has each context bid independently from the shared wallet. High-value contexts win pages; low-value ones yield and earn dividend. The market performs optimal internal allocation without any process-level coordination.

---

## 9. Guest API

```wit
price:        func() -> f64              // effective price for this context
rent:         func(ctx) -> f64           // clearing price on this context's device
dividend:     func() -> f64              // this context's dividend last step
balance:      func() -> f64              // current credit balance
latency:      func(ctx) -> f64           // EWA-smoothed tick latency (seconds)

bid:          func(value: f64)           // override SDK default
suspend:      func()                     // voluntary exclusion
```

The guest reads market signals (`price`, `rent`, `dividend`, `latency`), computes or overrides a bid, and the system handles allocation, placement, migration, and prefix optimization.

---

## 10. Theoretical Properties

### Theorem 1 (Efficiency)

The per-step allocation maximizes total declared value subject to the capacity constraint, up to an integrality gap of one marginal context.

*Proof sketch:* Greedy packing by value-per-page is optimal for the LP relaxation of the 0-1 knapsack. The LP relaxation is tight: at most one fractional variable. Rounding down loses at most one context.

### Theorem 2 (Truthfulness)

Under critical-value payments, truthful bidding is a dominant strategy in the stage game.

*Proof:* Standard result for greedy mechanisms in single-parameter domains. Each context's payment is determined by others' bids. A context cannot influence its payment by changing its own bid; it can only affect whether it is admitted. Since admission at the critical value is exactly the break-even point, bidding above or below true value cannot increase utility. (Lehmann, O'Callaghan, Shoham 2002.)

### Theorem 3 (Approximate Truthfulness in the Repeated Game)

In the T-step repeated game with N contexts, the mechanism is ε-DSIC with:

```
ε ≤ max_i(payment_i) / R  =  O(n_max / Σ_j n_j)
```

where `n_max` is the largest context's page count.

*Proof sketch:* A context's bid at step t affects revenue R_t, which affects dividends. But its share of the effect is bounded by its payment share, which is O(1/N) for contexts of similar size. As N grows, the incentive to deviate vanishes.

### Theorem 4 (Conservation)

Total credits satisfy `Σ_i balance_i(t) = Σ_i E_i − M(t)` at all times. Rent revenue is exactly redistributed. No context reaches zero balance in finite time while excluded.

*Proof:* Rent payments sum to R. Dividends sum to R (they are a partition of the revenue). Net flow from rent is zero. Make costs are the only credit destruction. An excluded context pays no rent and receives positive dividend, so its balance strictly increases while excluded.

### Theorem 5 (Efficient Placement)

Under Shapley cost-sharing, the cost-minimizing placement is the one that maximizes prefix sharing. Contexts sharing a common prefix are colocated in any cost-minimizing assignment.

*Proof sketch:* Shapley costs are additive over segments. Colocation reduces the per-context cost of each shared segment by a factor of 1/refcount. The placement minimizes Σ effective_pages × price, which is minimized when shared segments have maximum refcount — i.e., colocation.

### Theorem 6 (Composition)

The two-level mechanism (Shapley placement + knapsack auction) preserves all properties of the single-device auction. Placement affects effective_pages but not the auction's structure. Migration occurs only when rent savings exceed transfer cost (individual rationality of migration).

### Theorem 7 (Stability)

The clearing price is computed exactly each step (no dynamics, no convergence). Make cost penalizes thrashing economically; FCFS tiebreaking provides deterministic incumbent priority at equal bids. The system has no oscillatory modes.

*Proof:* The auction is a static optimization, not a dynamic process. There is no feedback loop to oscillate. Each restore→re-evict cycle consumes make cost credits, strictly reducing the thrashing context's future bid. The admission check prevents restoring contexts that would immediately re-suspend.

---

## 11. Comparison with Original Design

| Aspect | Original Design | New Design |
|--------|----------------|------------|
| **Congestion signal** | M/M/1 tax curve `λ/(1−λ)` | Clearing price (dual variable of knapsack) |
| **Make cost** | Separate mechanism (1 credit/page) | Same (1 credit/page, consumed) |
| **Interest** | Tax pool redistributed ∝ balance | Revenue redistributed ∝ endowment |
| **Bid formula** | Prescribed: `balance/(horizon × pages)` | Default provided; override allowed. Truthfulness from payment rule, not formula. |
| **Compute-or-wait** | Explicit MCC vs MBW check | Embedded in bid (negative bid = prefer waiting) |
| **Anti-thrashing** | Utilization cap + budget check | Make cost + FCFS tiebreaker + admission check |
| **Prefix sharing** | Shared pages split tax implicitly | Shapley cost-sharing (unique, axiomatic) |
| **Device placement** | Not specified | Shapley cost minimization |
| **Truthfulness** | DSIC via convex tax (requires correct formula) | DSIC via critical-value payments (model-free) |
| **Conservation** | Tax in = interest out | Payment in = dividend out |
| **Moving parts** | 5 mechanisms + 2 heuristics | 1 auction + 1 cost-sharing rule |

---

