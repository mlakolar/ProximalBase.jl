# Whole-iterate BlockIterate replacement strategies (#872)

This campaign compares strategies for `_copy_active!`, the step of
`copyto!(dest::BlockIterate, src::BlockIterate)` that runs after the
self-copy, length, layout and conversion checks. It reuses #861's `owner`,
restore, validation and restart helpers (`../copy861/cases.jl`) and #852's
locked environment and measurement worker (`../audit852`). Ordinary block
writes and `copyto!(blockview(x, k), buffer)` (#860) are outside its scope.

Use Linux, Julia 1.13.3, one Julia/BLAS thread and an exclusively reserved CPU.
From the repository root:

```sh
W=$(mktemp -d)                                   # every output goes here
BASE=a5010d20446b0fcd518e5ec6f0194d55bd227f21   # master before the change
python3 benchmark/copy872/make_sources.py "$BASE" $W/sources
export JULIA_DEPOT_PATH=$W/depot
JULIA=/absolute/path/to/julia-1.13.3/bin/julia
"$JULIA" --startup-file=no --project=benchmark/audit852 -e 'using Pkg; Pkg.instantiate()'
"$JULIA" --startup-file=no --project=benchmark/audit852 benchmark/copy872/check_all.jl $W/sources
S=$W/sources; cp -r $S/C $S/CAA; cp -r $S/Cn $S/CnAA
run() { python3 benchmark/copy872/run.py --julia "$JULIA" "$@"; }
# strategies: 7-round exploration, 21-round confirmation of C
run --source original=$S/O --source baseline=$S/A --source baselineAA=$S/AA --source B=$S/B \
  --source C=$S/C --source C2=$S/C2 --source D=$S/D --source CD=$S/CD --output $W/explore --rounds 7
run --source original=$S/O --source baseline=$S/A --source baselineAA=$S/AA --source C=$S/C \
  --source CAA=$S/CAA --source CD=$S/CD --output $W/confirm --rounds 21 --start-round 8
# refinements of C on the cells it regressed, then the full grid for Cn
python3 benchmark/copy872/targeted_cases.py > $W/cases-targeted.txt
run --source baseline=$S/A --source baselineAA=$S/AA --source C=$S/C --source Cn=$S/Cn --source Cs=$S/Cs \
  --cases-file $W/cases-targeted.txt --output $W/target --rounds 21 --start-round 8
run --source baseline=$S/A --source baselineAA=$S/AA --source Cn=$S/Cn --source CnAA=$S/CnAA \
  --output $W/cn7 --rounds 7
run --source baseline=$S/A --source baselineAA=$S/AA --source Cn=$S/Cn --source CnAA=$S/CnAA \
  --output $W/cn21 --rounds 21 --start-round 8
# the merged source, which adds eligibility checks to Cn's out-of-line copies
run --source baseline=$S/A --source baselineAA=$S/AA --source final=$PWD --source finalAA=$PWD \
  --cases-file $W/cases-targeted.txt --output $W/final --rounds 21 --start-round 8
for c in explore confirm target cn7 cn21 final; do
  python3 -I benchmark/copy872/analyze.py $W/$c --output $W/summary-$c
done
for t in $S/A $S/C $S/Cn $S/Cs $PWD; do
  taskset -c 0 "$JULIA" --startup-file=no -O2 --check-bounds=no --project=benchmark/audit852 \
    benchmark/copy872/codegen.jl $t $W/codegen
done
taskset -c 0 "$JULIA" --startup-file=no -O2 --check-bounds=no --project=benchmark/audit852 \
  benchmark/copy872/diag.jl $W/sources $W/diag
```

`make_sources.py` builds every tree in its output directory, which it requires to
lie outside the repository. `A` and `AA` are
the base commit, `O` is the original implementation used by #852, and every
other tree is the base commit with `_copy_active!` replaced by
`candidates/<name>.jl`:

| Tree | `_copy_active!` |
|---|---|
| B | when every source group is active: dense value copy, then the per-group metadata loop |
| C | when every source group is active: dense value copy and bulk copies of `active` and `position` |
| C2 | C, plus a bulk `position` copy at any activity |
| D | for singleton layouts (one entry per group), loops that index each group's entry directly; master's loops otherwise |
| CD | C's all-active path, then D's singleton path, then master's loops |
| Cn | C, with the bulk copies in a separate `@noinline` function; the merged code adds the eligibility checks below to that function |
| Cs | a dispatch between the bulk copies and the unchanged master loops |
| X | an unconditional dense copy; a negative control that must fail `check.jl` |

`check.jl` compares one tree with a model written from the documented contract:
values with signed zeros, NaN and Inf, active IDs and their order, reverse
positions and counts, inactive and active zeros, storage ownership, layout-cache
identity, conversion preflight, the different-layout fallback, self-copy,
Float32/Float64 conversion and `BigFloat`. `check_all.jl` runs it in fresh
bounds-checked processes for `A` and every candidate tree (not `AA` or `O`), and
passes only when every tree but `X` passes and `X` completes its checks and fails.

Each round of `run.py` starts one fresh worker per role in rotated order and
reverses the fixture order every second round. Seeds vary by round and match
across roles. The grid covers Float32 and Float64, n = 8192 and 65536, group
sizes 1, 8 and 64, contiguous, strided and arbitrary layouts, and 1, 10, 50
and 100% source activity. Destinations share or equal the source layout, are
empty or active, hold destination-only groups, use the `samegroups` cache, mix
precisions, or run a checkpoint/restart sequence. Run an A/A control for master
and for the candidate (a second, identical tree) whenever ratios are reported.

`analyze.py` checks every sample's median and minimum, that every configured case
ran in every round for every role, the rotation, one Julia binary, and that each
A/A control (a role named `<role>AA`) loaded a tree identical to its role. The
configured cases are the keys `run.py` writes to `campaign.json`; for campaigns
written before that, they are regenerated from the recorded grid, and for a campaign
run from a case file only their number is checked. It
reports each role's median paired ratio to `baseline` with seeded 10,000-resample
95% bootstrap intervals over rounds, and counts a case as an allocation increase
when, in some round, the role allocates more bytes or more times than the baseline
process of that round. Assess the 7- and
21-round campaigns separately. Small, timer-limited cases need the A/A
controls; not every ratio above 1.05 comes from a source change.

`diag.jl` loads trees A, B, C, C2, D and CD as separate modules, times the
public `copyto!` on identical fixtures, splits master's replacement into its
component loops, and times layout equality for independently built layouts.
`codegen.jl` writes the native code of the replacement functions for one tree
and prints, per function, its instruction, push and call counts, the number of
instructions up to and including its first jump, and whether it reads an fs- or gs-based
(thread-local) address. The candidate headers and D/CD's trailing blank lines
were edited after the campaigns, so regenerated trees differ from the measured
ones in those comment and blank lines only.

## Results (exclusive Intel Xeon E5-2640 v4, Julia 1.13.3)

The full record, with tables, is on #872. Raw samples are kept outside the
repository. In summary:

- When every source group is active, copying `storage`, `active` and `position`
  as whole arrays gives the state of the per-group loops, provided `active` and
  `position` of both iterates have one entry per group and no copy writes a byte
  that a later copy or the source reads. The merged code checks sufficient
  conditions and otherwise runs the loops: the byte range of each of the
  destination's three arrays must be disjoint from those of its other two and of
  the source's three, except a value array held by both, and storage whose element
  type is an isbits union (type tags outside the element bytes) is not copied
  whole. With an inactive source
  group, the copies can differ: the loops keep the destination's values there,
  such as a `-0.0`, unless the destination has the group active, when they clear
  it to `+0.0`.
  C, Cn and CD run full-activity replacement at a median 0.11–0.17× of master for
  each group size; B gains less, C2 is up to 2.6× slower at 1% activity, and D
  gains at most 1–7%, at partial activity with singleton groups.
- C was 6–10% slower than master on the smallest 1% restart cells in the
  exploration, the 21-round confirmation and the targeted campaign; in the
  21-round confirmation, the only campaign with a C A/A role, so was that
  identical tree. Its copies gave
  `_copy_active!` thread-local access and 9–13 calls; moving them to another
  function (Cn, Cs) removed the slowdown.
- Cn on the full grid: 21-round medians at partial activity are 0.999–1.000
  (master A/A 0.999–1.000), no case has a median above 1.05 with an interval
  above 1.0, and no case allocates more than master.
- Layout equality between independently built layouts costs 11–40 µs at
  K = 8192 singleton groups; sharing a layout through `similar`/`copy` or one
  `samegroups` call avoids it, so the layout cache is unchanged.
