"""The kernel-vocabulary audit, done by NAME rather than by prefix.

The old sweep matched `launch_*` / `dispatch_*` in the model C++ and compared
that against the table. Two things were wrong with it: it missed launchers
whose names start with neither (every `ops::` one that does not), and it
counted C++ identifiers and comment text that happen to match the pattern.

This one goes the other way round. It reads the HEADERS to learn what a
launcher IS -- a declaration whose parameter list takes a `cudaStream_t` --
and then looks for those exact names at call sites. A name that is not a
launcher cannot be a false positive, and a launcher with an unusual prefix
cannot be missed.
"""
import re, pathlib, collections

ROOT = pathlib.Path(".")
HDRS = list((ROOT/"crates/kernels-cuda/csrc/src").rglob("*.hpp")) + \
       list((ROOT/"crates/kernels-cuda/csrc/third_party").rglob("*.hpp"))

def params(text, open_at):
    depth, i = 1, open_at
    while i < len(text) and depth:
        i += 1
        if i < len(text):
            if text[i] == '(': depth += 1
            elif text[i] == ')': depth -= 1
    return text[open_at:i]

launchers = {}
for h in HDRS:
    src = h.read_text(errors="ignore")
    for m in re.finditer(
        r'^\s*(?:inline\s+|static\s+)*(?:void|bool|int)\s+(\w+)\s*\(', src, re.M
    ):
        p = params(src, m.end() - 1)
        if "cudaStream_t" in p or "cublasHandle_t" in p:
            launchers[m.group(1)] = h.name

if len(launchers) < 100:
    raise SystemExit(
        f"only {len(launchers)} launchers found under crates/kernels-cuda, "
        f"and the repair is NOT to fix the glob -- read the paragraph above "
        f"`HDRS`. The host C++ tree this scans was DELETED, not moved. A "
        f"launcher is defined here as a host declaration taking a "
        f"`cudaStream_t` or `cublasHandle_t`, and the live JIT crate has no "
        f"host C++ at all: its device text is `.cuh` compiled by NVRTC at "
        f"run time and launched by name from Rust. There is no declaration "
        f"to find, so repointing the glob would swap this honest failure for "
        f"a silent pass.\n\n"
        f"What answering it needs, measured 2026-08-21 rather than guessed: "
        f"`crates/kernels-cuda/kernels/**/*.cuh` holds 347 distinct "
        f"`__global__` entry points. The table under `src/` yields 68 symbol "
        f"strings at this file's `glob`, 271 at `rglob`. The overlap is 23 "
        f"and 34. So `__global__` is NOT simply the new spelling of "
        f"'declared' -- the table names logical kernels and the device text "
        f"is mostly templated internals, and 313 of 347 are named nowhere. "
        f"Deciding which `__global__`s are the crate's VOCABULARY, as "
        f"opposed to its private helpers, is the design question. It is a "
        f"day's work on the naming, not an edit to a path."
    )

tbl_src = "".join(
    p.read_text() for p in sorted((ROOT/"crates/kernels-cuda/src").glob("*.rs"))
)
declared = {s.split("::")[-1] for s in re.findall(r'"([a-z0-9_:]+)"', tbl_src)}
if not declared:
    raise SystemExit(
        "no symbols found in crates/kernels-cuda/src/*.rs -- the table moved "
        "again, and this audit reports nonsense rather than nothing when that "
        "happens. Fix the glob above."
    )

emitted = set()
for f in list((ROOT/"crates/model/src").rglob("emit.rs")) + [ROOT/"crates/model-compiler/src/lower.rs"]:
    if f.exists():
        emitted |= {s.split("::")[-1] for s in re.findall(r'"([a-z0-9_:]+)"', f.read_text())}

MODELS = ROOT/"crates/driver-cuda/csrc/src/model"
per_family = collections.defaultdict(set)
for cpp in MODELS.rglob("*"):
    if cpp.suffix not in (".cpp", ".hpp", ".cu", ".cuh", ".inc"): continue
    fam = cpp.relative_to(MODELS).parts[0]
    if fam.endswith((".cpp", ".hpp")): fam = "(root)"
    src = cpp.read_text(errors="ignore")
    src = re.sub(r'//[^\n]*', '', src)
    src = re.sub(r'/\*.*?\*/', '', src, flags=re.S)
    for name in launchers:
        if re.search(r'\b' + re.escape(name) + r'\s*\(', src):
            per_family[fam].add(name)

print(f"launchers declared in headers: {len(launchers)}")
print(f"symbols in the kernel table:   {len(declared)}")
print()
total_gap = set()
for fam in sorted(per_family):
    gap = sorted(n for n in per_family[fam] if n not in declared and n not in emitted)
    if gap:
        print(f"{fam}: {len(gap)}")
        for n in gap: print(f"     {n:52s} {launchers[n]}")
        total_gap |= set(gap)
print()
print(f"DISTINCT UNDECLARED LAUNCHERS ACROSS ALL FAMILIES: {len(total_gap)}")
print()

EXPECTED_RESIDUE = [
    ("plan_attention_", "prefix",
     "host PREPARES. A prepare is what `needs` names, not something a trace\n"
     "records; declaring one would make the table claim a statement exists\n"
     "that does not."),
    ("prepare_attention_", "prefix", None),
    ("all_reduce_", "prefix",
     "a COLLECTIVE, and the shell's to schedule. The kernel moved to\n"
     "kernels-cuda/comm/ because it computes (a reduction fused with a\n"
     "residual add and an RMSNorm), but no trace names it: Rust emits nothing\n"
     "for it, the string dispatcher has no entry, and the model C++ reaches it\n"
     "as a METHOD on the TP plane (`tp->all_reduce_bf16(...)`), chosen by the\n"
     "custom-vs-NCCL fallback policy that stays driver-side. A table row would\n"
     "assert a statement that cannot be written. (These match here only\n"
     "because the scan is by name, and a method call spells the same name as\n"
     "a free one.)"),
    ("can_fuse_residual_rmsnorm", "exact",
     "the predicate of that same custom-vs-NCCL policy."),
    ("set_stream", "exact", "a cuBLAS handle setter."),
    ("maybe_bench_", "prefix", "a benchmark harness."),
    ("lm_head_argmax_chunked", "exact",
     "a host-side chunking helper over the real launcher."),
    ("causal_conv1d_prefill_noact_bf16", "exact",
     "same case as `argmax_bf16` below: fired only from a hand-written\n"
     "forward (`gemma4_audio_forward.cu`'s lconv1d), so no trace records it\n"
     "and a row would have no DSL statement behind it. The activated form\n"
     "`causal_conv1d_prefill_bf16` IS declared -- Qwen3.5 traces that one."),
    ("argmax_bf16", "exact",
     "fired only from a HAND-WRITTEN forward (`csm_backbone_forward.cu`), not\n"
     "a traced one. A row for it was written and `model`'s\n"
     "`the_table_is_exactly_the_dsl_surface` rejected it: that test holds the\n"
     "table and `dsl::cuda` to the same set, and a DSL statement is something\n"
     "a trace RECORDS. Nothing traces this argmax, so the statement would\n"
     "have no caller and the row would claim a surface that is not there.\n"
     "If CSM ever gets a declared forward, this entry is what should go."),
]

def excused(name: str) -> bool:
    return any(name.startswith(k) if kind == "prefix" else name == k
               for k, kind, _ in EXPECTED_RESIDUE)

print("EXPECTED RESIDUE -- these SHOULD be absent from the table:\n")
for key, kind, why in EXPECTED_RESIDUE:
    label = f"{key}*" if kind == "prefix" else key
    if why is None:
        print(f"  {label}")
        continue
    head, *rest = why.split("\n")
    print(f"  {label:26s}{head}")
    for line in rest:
        print(f"  {'':26s}{line}")

stale = [k for k, kind, _ in EXPECTED_RESIDUE
         if not any(n.startswith(k) if kind == "prefix" else n == k
                    for n in total_gap)]
if stale:
    print("\nNOTE: nothing matched these entries any more, so they now excuse")
    print("nothing and can go: " + ", ".join(stale))

unexpected = sorted(n for n in total_gap if not excused(n))
if unexpected:
    raise SystemExit(
        "\n" + "=" * 72 + "\n"
        f"{len(unexpected)} launcher(s) a model fires with no declaration to\n"
        "state them, and no entry above excusing that:\n\n"
        + "".join(f"    {n:52s} {launchers[n]}\n" for n in unexpected)
        + "\nEither give each a row in crates/kernels-cuda/src/<family>.rs, or\n"
        "-- if it genuinely cannot be written as a traced statement -- add it\n"
        "to EXPECTED_RESIDUE with the reason why.\n" + "=" * 72
    )

print(f"\nAll {len(total_gap)} undeclared launchers are accounted for.")
