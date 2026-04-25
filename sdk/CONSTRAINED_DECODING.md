# Constrained Decoding & Generation Behavior

This document covers behavior shared across the Rust, Python, and JavaScript
SDKs, plus the places where they diverge intentionally.

---

## Auto-flush + auto-cue (Python and JS only)

In Python and JavaScript, calling `generate()` (or any `generate_*` /
`generate*` convenience) implicitly calls `cue()` and then `flush()` before
sampling begins. Pending fills from `system()` / `user()` / etc. land in the
KV cache, the model is cued to begin generating, and the stream returns.

**Python**: pass `auto_flush=False` to `generate(...)` to disable.

**JavaScript**: there is no escape hatch — call `flush()` yourself before
`generate()` if you need to inspect the buffer; the auto-flush will then be
a no-op.

**Rust** does not auto-flush. Call `ctx.cue()` before `ctx.generate(...)` —
the first `step()` flushes the buffer:

```rust
ctx.user("hello").cue();
let text = ctx.generate(Sampler::ARGMAX).collect_text().await?;
```

This asymmetry tracks language conventions: Python and JS lean on convenient
defaults; Rust prefers explicit control flow.

---

## Pick a sampler

| Use case | Sampler |
| --- | --- |
| Grammar / regex / JSON-schema constrained | `Sampler::ARGMAX` (Rust) / `Sampler.argmax()` (Python, JS) |
| Free-form text | `Sampler.top_p(0.6, 0.95)` (Python) / `Sampler::top_p(0.6, 0.95)` (Rust) / `Sampler.topP(0.6, 0.95)` (JS) |
| Deterministic free-form | `Sampler.greedy()` (alias for argmax) |
| Distribution-shape experimentation | `Sampler.top_k(t, k)`, `Sampler.min_p(t, p)`, `Sampler.top_k_top_p(t, k, p)` |

For grammar-constrained generation, argmax is almost always the right
choice — most masked positions have only a handful of valid tokens and
stochastic sampling rarely helps.

---

## `Schema` vs `Constraint` vs custom impl

| You want… | Use |
| --- | --- |
| JSON conforming to a schema string | `Schema.json_schema(s)` / `Schema::JsonSchema(s)` |
| Any valid JSON | `Schema.json()` / `Schema::Json` |
| Strings matching a regex | `Schema.regex(p)` / `Schema::Regex(p)` |
| Custom EBNF grammar | `Schema.ebnf(g)` / `Schema::Ebnf(g)` |
| Reuse a precompiled grammar across contexts | `Schema.grammar(g)` / `Schema::Grammar(&g)` |
| Banned tokens, learned constraints, anything stateful that isn't a grammar | Implement `Constraint` / `Constrain` directly |

Schemas are declarative; the SDK compiles them into a stateful matcher and
drives it per generated token. Custom `Constraint` impls do the driving
themselves.

### Typed JSON

| SDK | API | Validator |
| --- | --- | --- |
| Rust | `collect_json::<T>()` | `schemars::JsonSchema` derive on `T` |
| Python (untyped) | `generate_json(sampler, schema=...)` returns `dict`/`list`/primitive | — |
| Python (typed, optional) | `generate_pydantic(MyModel, sampler=...)` | pydantic v2 (only available if installed) |
| JS / TS (untyped) | `generateJson({sampler, schema})` returns `unknown` | — |
| JS / TS (typed) | `generateJson({sampler, schema, parse: MyZod.parse})` | bring your own (Zod, arktype, …) |

---

## Composition asymmetry

**Rust** composes constraints by repeating `with_constraint(...)` — masks
are AND-ed across all attached constraints per step:

```rust
ctx.generate(Sampler::ARGMAX)
    .with_schema(Schema::Ebnf(grammar))?
    .with_constraint(BannedTokens::new(...))
    .collect_text().await?;
```

**Python and JS** take a single constraint at a time. To compose, write a
custom `Constraint` whose `step()` ANDs masks from two underlying
constraints internally. The BRLE-AND helper isn't exposed in those SDKs
yet — most users don't need composition, and the workaround is a ~20-line
wrapper class.

If you find yourself wanting composition in Python or JS, file an issue.

---

## Structured returns

`session.send` and the inferlet's `main` return value both accept structured
values across all three SDKs — the framework JSON-serializes for you.

| SDK | `session.send` | `main` return |
| --- | --- | --- |
| Rust | always `&str` (byte-level WIT API) | `T: Serialize` via `#[inferlet::main]` |
| Python | `str \| dict \| list \| BaseModel \| ...` | same — bakery wrapper auto-stringifies |
| JS / TS | `string \| object \| number \| boolean \| ...` | same |

So for the typed-JSON path, you don't have to `JSON.stringify` (or
`json.dumps`) manually:

```typescript
const person = await ctx.generateJson<Person>({ ..., schema: PERSON_SCHEMA });
session.send(`Hello ${person.name}!`);  // string interpolation
return person;                          // auto-stringified
```

```python
person = await ctx.generate_pydantic(Person, sampler=Sampler.argmax())
session.send(person)   # pydantic.BaseModel → model_dump_json()
return person          # same
```

## Mutual exclusion

`constrain` and `logit_mask` (Python) / `logitMask` (JS) are mutually
exclusive. The `logit_mask` parameter is a static BRLE applied identically
every step — useful for stateless constraints (banned tokens) but wrong
for stateful matchers, which need to advance per generated token.

Pass `logit_mask` for stateless masks, `constrain` for everything else.

In Rust, similarly: `with_schema(Schema::*)` is incompatible with
`collect_json::<T>()` because both define the JSON schema, and disagreement
is silent. The Rust SDK errors at runtime if both are set.
