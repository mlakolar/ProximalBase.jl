# Whole-owner BlockIterate replacement (#861)

This campaign reuses #852's locked environment, measurement worker, sample
exports, warmup, calibration, source digests, descriptor generator and original
raw `AtomIterate` adapter. It adds whole-owner replacement fixtures without
changing the frozen audit or ordinary block-update methods.

Use Linux, Julia 1.13.3, one Julia/BLAS thread and an exclusively reserved CPU.
From the repository root, archive master **eb74a914** (the pre-change baseline)
and original **bd0740d0ebbebcc76fe208e4318fe6fc5451d34b**:

```sh
mkdir -p /tmp/copy861-master /tmp/copy861-original /tmp/copy861-timed
git archive eb74a914 src | tar -x -C /tmp/copy861-master
git archive bd0740d0ebbebcc76fe208e4318fe6fc5451d34b src | tar -x -C /tmp/copy861-original
git archive 9b12babc69d9939b21d38f564db132c23429dbe9 src | tar -x -C /tmp/copy861-timed
export JULIA_DEPOT_PATH=/tmp/copy861-depot
JULIA=/absolute/path/to/julia-1.13.3/bin/julia
"$JULIA" --startup-file=no --project=benchmark/audit852 -e 'using Pkg; Pkg.instantiate()'
python3 benchmark/copy861/run.py --julia "$JULIA" --baseline /tmp/copy861-master --original /tmp/copy861-original --candidate /tmp/copy861-timed --output /tmp/copy861-check --check --rounds 1
python3 benchmark/copy861/run.py --julia "$JULIA" --baseline /tmp/copy861-master --original /tmp/copy861-original --candidate /tmp/copy861-timed --output /tmp/copy861-broad --rounds 7
python3 benchmark/copy861/run.py --julia "$JULIA" --baseline /tmp/copy861-master --original /tmp/copy861-original --candidate /tmp/copy861-timed --output /tmp/copy861-confirm --rounds 21 --start-round 8
python3 benchmark/copy861/analyze.py /tmp/copy861-broad --output /tmp/copy861-summary7
python3 benchmark/copy861/analyze.py /tmp/copy861-confirm --output /tmp/copy861-summary21
python3 benchmark/copy861/analyze.py /tmp/copy861-broad /tmp/copy861-confirm --output /tmp/copy861-summary28
PB_SOURCE=/tmp/copy861-master PB_DIAGNOSTICS=/tmp/copy861-diagnostics taskset -c 0 "$JULIA" --startup-file=no -O2 --check-bounds=no --project=benchmark/audit852 benchmark/copy861/diagnose.jl > /tmp/copy861-components.csv
python3 benchmark/copy861/run.py --julia "$JULIA" --baseline /tmp/copy861-master --original /tmp/copy861-original --candidate "$PWD" --output /tmp/copy861-exact-check --exact852 --check --rounds 1
python3 benchmark/copy861/run.py --julia "$JULIA" --baseline /tmp/copy861-master --original /tmp/copy861-original --candidate "$PWD" --output /tmp/copy861-exact7 --exact852 --rounds 7
python3 benchmark/copy861/run.py --julia "$JULIA" --baseline /tmp/copy861-master --original /tmp/copy861-original --candidate "$PWD" --output /tmp/copy861-exact21 --exact852 --rounds 21 --start-round 8
python3 benchmark/copy861/analyze.py /tmp/copy861-exact7 /tmp/copy861-exact21 --output /tmp/copy861-exact-summary
```

Choose the allowed CPU with `--cpu`; default is the lowest allowed CPU.
`--candidate` can select an immutable source snapshot. Each round starts four
fresh sequential workers: original, master, proposed and independently loaded
master A/A. Latin rotation balances process positions over the combined 28
rounds; fixture order reverses each round. Seeds vary by round and match across
roles. Timing uses one evaluation per sample with restoration outside timing;
the worker retains every sample, allocation bytes/counts, pointers, calibration,
source/harness hashes and runtime provenance. The campaign records hashes of
the reused #852 files as well as its own files. Bounds-enabled checks run each
case twice before any measurement. The original raw constructor supports both
precisions and all descriptor forms; it has no active-group metadata, so only
values are comparable there. Its historical allocating copy rebuilds views.

The 300 cases distinguish shared layout, equal independent layout, explicit
`samegroups` cache hit, empty/active destination, active zeros, different-layout
fallback, self-copy, `copy!`, mixed Float32/64, allocating copy, and four complete
restart cycles. Each restart copies a baseline, changes one entry in every
active group, checkpoints, restores and resumes. This measures replacement in
context; it does not establish full solver convergence or total solve time.
Payloads are nonuniform and reference values/activity/order/maps are checked
independently. Ordinary update APIs are unchanged.

Active broad destinations contain the source-active groups plus the same number
of destination-only groups. Source active sets are randomized; alternate active
groups deliberately hold zeros in the `shared_zeros` cases. The `--exact852`
mode directly reuses the unchanged historical `block_cases` copy/copyto fixtures
for Float32 strided and Float64 contiguous singletons at 81/819 active groups.
It keeps the original `fill!(dest,0)` preparation, which retains activity in the
current owner. The final-source focal campaign runs on an entirely quiet host.

The broad timing source precedes the per-group `zero(T)` hoist needed for
`BigFloat` object reuse. Final-source Float32/64 native code is identical (only
declared global addresses differ in public wrappers); both trees/listings are
preserved. Analyze broad and final-source focal campaigns separately, since
their source digests intentionally differ. The frozen broad measurement harness
and updated exact harness are both archived; the current analyzer includes
subsequent verification-strengthening changes without altering timed operations.

The analysis verifies sample medians/minima, complete round/role pairing and
duplicate absence. It reports medians of paired ratios and seeded 10,000-resample
95% bootstrap intervals over independent rounds. Seven-round and 21-round
results must be assessed separately, with A/A and absolute durations. Small
timer-limited cases and allocator variance need controls; do not attribute
every ratio above 1.05 to source changes.

Component preflight and code-equivalence reproductions are under
`docs/audit/block-copy-861/`: run `preflight.jl SOURCE` with the locked benchmark
environment; run `codegen.jl SOURCE OUTPUT` once per source in fresh processes,
then `verify_codegen.py BEFORE AFTER comparison.json`. These are diagnostics,
separate from headline paired trials. See the audit report for archive replay,
raw samples/provenance and the remaining historical performance gaps.

To reproduce the separate allocating-copy context control, put these two case
IDs in `copy-control.txt`:

```text
block/Float32/n8192/s81/b1/strided/copy
block/Float32/n8192/s819/b1/strided/copy
```

```sh
python3 benchmark/copy861/run.py --julia "$JULIA" --baseline /tmp/copy861-master --original /tmp/copy861-original --candidate "$PWD" --output /tmp/copy861-copy-control24 --exact852 --rounds 24 --roles baseline,candidate,baselineAA,candidateAA --cases-file copy-control.txt
python3 benchmark/copy861/analyze.py /tmp/copy861-copy-control24 --output /tmp/copy861-copy-summary
```

This follows an isolated >5% allocating-copy flag in the focal confirmation,
whose interval included parity. Both sources receive independent A/A controls;
24 rounds balance all four process positions. The control is reported alongside
the original focal observations, with its small residual timing difference.
