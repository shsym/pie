"""The gate-coverage audit: a hand-maintained crate list rots, quietly.

The `workspace-check` job lints and formats a LIST of crates rather than
the workspace, and a list is a claim about the workspace that nothing
checks. This tree has already paid for that claim twice.

The rename of 2026-08-08 collapsed six crate names onto three. It left
`-p eta-compiler` in one clippy step FOUR TIMES, which is harmless in
itself -- but the reason it took months to notice is the same reason a
crate DROPPING OUT would have taken months to notice: nobody reads a
fifteen-entry `-p` list. When the duplication was finally found, the
first question was whether a crate had fallen out rather than merely
doubled, and answering it meant diffing the list against
`cargo metadata` by hand.

The second time was worse and silent. Sixteen of the twenty ungated
crates had drifted out of rustfmt -- `runtime` by 146 hunks, and by 144 the
ahead-of-time CUDA archive crate that has since been deleted -- because
they had never been in the fmt list at
all. There was no signal. A crate that is not in the list is
indistinguishable, from CI's output, from a crate that passes.

So this is the check that makes the list say what it means: **every
workspace member is either gated or excluded on purpose.**

## Why exclusions are named here and not merely absent

An absent crate and an excluded crate look identical in `ci.yml` -- both
are "not in the `-p` list". The difference is that one is a decision and
the other is an oversight, and only one of them should survive a reader.
So an exclusion has to be written down WITH ITS REASON, and a crate that
is neither gated nor listed below is an error.

That also gives the exclusions an expiry: `EXCLUSIONS` says what would
have to change for the crate to join, so the entry is falsifiable rather
than permanent. The ones left are waiting on a toolkit, a GPU, a Mac, or
a rewrite that is actively in flight.

## Gates on other jobs count as gates

`STEPS` reads steps by name from the whole workflow, not from one job, so
a crate gated on the macOS runner is GATED and not merely excused.

`engine-metal` was the case that shaped this and it has now been through
the whole cycle. It began as an exclusion reading "gated on the macOS job
instead, in BOTH feature halves" -- a true sentence that nothing checked,
so deleting those two steps would have left this audit green while the
crate went unlinted. Naming them in `STEPS` turned the sentence into a
check. Then R3 put the crate out of the workspace and deleted the macOS
job, and the check fired from the other direction: the step names were
gone and the audit said so by name. Both failure modes it was built for
have now happened to the same crate, and it reported each.

Duplicates are therefore counted WITHIN a step rather than across them. A
crate named twice in one command is the bug that happened here; a crate
named in two steps is a crate linted in two feature configurations, which
is why two steps are allowed to name one -- a lint that fires in only one
feature half is a lint nobody sees.

## What it does not check

That the gate PASSES -- CI runs the gate itself for that. This only
checks that the gate is asked about every crate. A gate that is asked
and fails is a red build; a gate that is never asked is a green one, and
the second is what this is for.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / ".github" / "workflows" / "ci.yml"

FORWARDS = [("runtime", "worker"), ("worker", "pie")]

FORWARD_EXCLUSIONS = {
    ("worker", "pie"): {
        "nixl": (
            "`transport`'s NIXL engine is a stub -- `crates/transport/src/"
            "lib.rs` says so and `Error` has a variant for being asked for "
            "it in a build that lacks it. Forwarding it would put a "
            "selectable flag on the CLI that turns on nothing a user can "
            "reach. Forward it when the engine exists."
        ),
    },
}

EXCLUSIONS = {
    "fmt": {
        "engine-cuda": (
            "being rewritten wholesale (108 commits in three days); "
            "reformatting it would collide with in-flight work for no "
            "benefit to files that are being replaced. Add it when the "
            "rewrite lands."
        ),
        "kernels-cuda": (
            "1,320 drifted hunks against 47 commits in three days, three "
            "of them in the last day: the crate the CUDA rewrite moved "
            "INTO, so it is the churn itself rather than a crate that "
            "drifted once. Re-measure with `cargo fmt --check -p "
            "kernels-cuda | grep -c '^Diff'` and `git log "
            "--since='24 hours ago' -- crates/kernels-cuda`."
        ),
    },
    "clippy": {
        "engine-cuda": "needs nvcc, and is being rewritten",
        "pie-server-py": "a pyo3 extension; built by maturin, not by this job",
    },
}

STEPS = {
    "fmt": ["cargo fmt (compiler crates)"],
    "clippy": [
        "cargo clippy (deny warnings)",
        "cargo clippy (model, deny warnings)",
    ],
}

def members():
    """Every crate Cargo considers part of this workspace."""
    return {name for name in packages()}

def packages():
    """`name -> declared features` for every workspace member."""
    out = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return {p["name"]: p["features"] for p in json.loads(out.stdout)["packages"]}

def forward_problems(pkgs):
    """Features that stop at a library instead of reaching the binary.

    A feature is only real if the package that produces a BINARY can select
    it. `runtime` and `worker` are libraries, so every flavor and probe they
    offer has to be re-declared twice on the way out -- by hand, in three
    manifests, with nothing checking that the copies agree.

    They have not agreed. The root package lost its `engine-metal` forward
    when the C++ Metal driver was retired and never got it back, leaving
    five dead `#[cfg]` blocks in `pie init`'s config template (see
    `scripts/cfg-feature-audit.py`, which catches the OTHER half of that
    bug: the `cfg` naming a feature nobody declares). `profile-hot-path`
    and `profile-all` reached `worker` and stopped. `nixl` still does, and
    is named below so that it stops being an accident.

    Read out of `cargo metadata`, so this sees what Cargo resolved rather
    than what a manifest appears to say.
    """
    problems = []
    for inner, outer in FORWARDS:
        forwarded = {
            edge.split("/", 1)[1]
            for values in pkgs[outer].values()
            for edge in values
            if edge.startswith(f"{inner}/")
        }
        allowed = FORWARD_EXCLUSIONS.get((inner, outer), {})
        for feature in sorted(pkgs[inner]):
            if feature == "default" or feature in forwarded:
                continue
            if feature in allowed:
                continue
            problems.append(
                f"forward: `{inner}/{feature}` is not forwarded by "
                f"`{outer}`, so no build of `{outer}` can select it. Add "
                f"the forward, or add it to FORWARD_EXCLUSIONS[({inner!r}, "
                f"{outer!r})] in this file WITH THE REASON."
            )
        for feature in sorted(allowed):
            if feature not in pkgs[inner]:
                problems.append(
                    f"forward: `{inner}/{feature}` is excused from being "
                    f"forwarded but `{inner}` no longer declares it. Drop "
                    f"the entry."
                )
            elif feature in forwarded:
                problems.append(
                    f"forward: `{inner}/{feature}` is excused from being "
                    f"forwarded and IS forwarded by `{outer}`. The "
                    f"exclusion is stale -- drop it."
                )
    return problems

def steps_by_name():
    """`ci.yml`'s `run:` bodies, keyed by step name.

    Parsed by indentation rather than with `yaml`, so that the audit needs
    nothing installed to answer -- refusing to answer for want of a
    dependency is the failure mode it exists to prevent, and an audit that
    takes a different code path depending on an optional import is two
    audits with one name.

    A step ends at the next non-blank line indented no further than its own
    `- name:`, which is where the next step or the next job key begins. The
    comment blocks in this workflow sit at exactly that indentation, so a
    step's body stops before the prose introducing the next one -- which
    matters, because that prose quotes `-p` lists.
    """
    steps = {}
    lines = CI.read_text().split("\n")
    index = 0
    while index < len(lines):
        head = re.match(r"^(\s*)- name:\s*(.+?)\s*$", lines[index])
        if not head:
            index += 1
            continue
        indent = len(head.group(1))
        body = []
        index += 1
        while index < len(lines):
            line = lines[index]
            if line.strip() and len(line) - len(line.lstrip()) <= indent:
                break
            body.append(line)
            index += 1
        block = "\n".join(body)
        run = re.search(r"^[ \t]*run:[ \t]*\|?[ \t]*", block, re.M)
        steps[head.group(2)] = block[run.end() :] if run else ""
    return steps

def listed(body):
    """The crates a `-p`-style command names, and how often."""
    counts = {}
    for crate in re.findall(r"-p\s+([A-Za-z0-9_-]+)", body):
        counts[crate] = counts.get(crate, 0) + 1
    return counts

def main():
    pkgs = packages()
    all_members = set(pkgs)
    bodies = steps_by_name()
    problems = forward_problems(pkgs)

    for gate, step_names in STEPS.items():
        gated = {}
        for step in step_names:
            if step not in bodies:
                problems.append(
                    f"{gate}: ci.yml has no step named {step!r}. This audit "
                    f"reads the step by name, so a rename silently stops it "
                    f"checking anything -- update STEPS in this file."
                )
                continue
            for crate, count in listed(bodies[step]).items():
                if count > 1:
                    problems.append(
                        f"{gate}: `-p {crate}` appears {count} times in "
                        f"{step!r}. Harmless to run and a sign nobody is "
                        f"reading the list -- which is how a crate goes "
                        f"missing from it."
                    )
                gated[crate] = gated.get(crate, 0) + count

        for crate in sorted(gated):
            if crate not in all_members:
                problems.append(
                    f"{gate}: `-p {crate}` is not a workspace member. It was "
                    f"renamed or removed, and the gate has been silently "
                    f"narrower ever since."
                )

        excluded = EXCLUSIONS.get(gate, {})
        for crate in sorted(excluded):
            if crate not in all_members:
                problems.append(
                    f"{gate}: {crate} is excluded but is not a workspace "
                    f"member. Drop the entry."
                )
            if crate in gated:
                problems.append(
                    f"{gate}: {crate} is both gated and excluded. The "
                    f"exclusion is stale -- drop it."
                )

        for crate in sorted(all_members - set(gated) - set(excluded)):
            problems.append(
                f"{gate}: {crate} is neither gated nor excluded. Add it to "
                f"the {gate} step in ci.yml, or to EXCLUSIONS['{gate}'] in "
                f"this file WITH THE REASON. A crate that is merely absent "
                f"looks exactly like a crate that passes."
            )

    if problems:
        print("ci-gate-audit: the gate lists do not describe the workspace\n")
        for problem in problems:
            print(f"  * {problem}")
        print(f"\n{len(problems)} problem(s).")
        return 1

    for gate in STEPS:
        covered = len(all_members) - len(EXCLUSIONS.get(gate, {}))
        print(
            f"ci-gate-audit: {gate} covers {covered}/{len(all_members)} "
            f"crates, {len(EXCLUSIONS.get(gate, {}))} excluded on purpose."
        )
    return 0

if __name__ == "__main__":
    sys.exit(main())
