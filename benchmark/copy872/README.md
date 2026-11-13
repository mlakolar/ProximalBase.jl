# Whole-array BlockIterate replacement (#872)

This benchmark compares master's `_copy_active!` with the merged one. `_copy_active!`
is the step of `copyto!(dest::BlockIterate, src::BlockIterate)` that runs after the
self-copy, length, layout and conversion checks. When every source group is active,
the merged version copies `storage`, `active` and `position` as whole arrays in an
out-of-line function. The benchmark reuses #861's `owner`, restore, validation and
restart helpers (`../copy861/cases.jl`) and #852's locked environment and measurement
worker (`../audit852`). Ordinary block writes and `copyto!(blockview(x, k), buffer)`
(#860) are outside its scope.

Use Linux, Julia 1.13.3, one Julia/BLAS thread and an exclusively reserved CPU.
From the repository root, with the merged source checked out:

```sh
W=$(mktemp -d)                                   # every output goes here
BASE=a5010d20446b0fcd518e5ec6f0194d55bd227f21   # master measured as the baseline
python3 benchmark/copy872/make_sources.py "$BASE" $W/sources
export JULIA_DEPOT_PATH=$W/depot
JULIA=/absolute/path/to/julia-1.13.3/bin/julia
"$JULIA" --startup-file=no --project=benchmark/audit852 -e 'using Pkg; Pkg.instantiate()'
# correctness: master passes and the negative control fails, then the merged source
"$JULIA" --startup-file=no --project=benchmark/audit852 benchmark/copy872/check_all.jl $W/sources
"$JULIA" --startup-file=no --project=benchmark/audit852 --check-bounds=yes \
  benchmark/copy872/check.jl $PWD 4000 872
# timing: targeted 21-round confirmation against master, with A/A controls
S=$W/sources
python3 benchmark/copy872/targeted_cases.py > $W/cases-targeted.txt
python3 benchmark/copy872/run.py --julia "$JULIA" --cpu 2 \
  --source baseline=$S/A --source baselineAA=$S/AA --source final=$PWD --source finalAA=$PWD \
  --cases-file $W/cases-targeted.txt --output $W/final --rounds 21 --start-round 8
python3 -I benchmark/copy872/analyze.py $W/final --output $W/summary-final
# native code of the replacement path
for t in $S/A $PWD; do
  taskset -c 2 "$JULIA" --startup-file=no -O2 --check-bounds=no --project=benchmark/audit852 \
    benchmark/copy872/codegen.jl $t $W/codegen
done
```

`--cpu` and `taskset -c` name the reserved CPU. Leaving out `--cases-file` runs the
full grid; run it as a 7-round exploration and a separate 21-round confirmation
(`--start-round 8`), and assess the two separately.

The baseline is `a5010d20`. Its `copyto!(::BlockIterate, ::BlockIterate)` and
`_copy_active!` differ from those of `1f3749f3`, the base of the merged change, in one
comment only; other files differ between the two commits.

`make_sources.py` builds its trees in an output directory that it requires to lie
outside the repository. `A` and `AA` are the base commit. Each file in `candidates/`
gives a tree with `_copy_active!` replaced by that file; the only one is `X`, an
unconditional dense copy of values and metadata, a negative control that must fail
`check.jl`.

`check.jl` compares one tree with a model written from the documented contract:
values with signed zeros, NaN and Inf, active IDs and their order, reverse
positions and counts, inactive and active zeros, storage ownership, layout-cache
identity, conversion preflight, the different-layout fallback, self-copy,
Float32/Float64 conversion and `BigFloat`. `check_all.jl` runs it in fresh
bounds-checked processes for `A` and every tree built from `candidates/` (not `AA`),
and passes only when every such tree but `X` passes and `X` completes its checks
and fails.

Each round of `run.py` starts one fresh worker per role in rotated order and
reverses the fixture order every second round. Seeds vary by round and match
across roles. The full grid covers Float32 and Float64, n = 8192 and 65536, group
sizes 1, 8 and 64, contiguous, strided and arbitrary layouts, and 1, 10, 50
and 100% source activity. Destinations share or equal the source layout, are
empty or active, hold destination-only groups, use the `samegroups` cache, mix
precisions, or run a checkpoint/restart sequence. `targeted_cases.py` selects every
1% and every 100% case and the 10% `restart_shared` cases (648 cases). Run an A/A
control for master and for the merged source (a second role loading an identical
tree) whenever ratios are reported.

`analyze.py` checks every sample's median and minimum, that every configured case
ran in every round for every role, the rotation, one Julia binary, and that each
A/A control (a role named `<role>AA`) loaded a tree identical to its role. The
configured cases are the keys `run.py` writes to `campaign.json`; for campaigns
written before that, they are regenerated from the recorded grid, and for a campaign
run from a case file only their number is checked. It
reports each role's median paired ratio to `baseline` with seeded 10,000-resample
95% bootstrap intervals over rounds, and counts a case as an allocation increase
when, in some round, the role allocates more bytes or more times than the baseline
process of that round. Small, timer-limited cases need the A/A controls; not every
ratio above 1.05 comes from a source change.

`codegen.jl` writes the native code of the replacement functions for one tree
and prints, per function, its instruction, push and call counts, the number of
instructions up to and including its first jump, and whether it reads an fs- or gs-based
(thread-local) address.

The strategy exploration that led to the merged design compared further
replacements (B, C, C2, D, CD, Cn, Cs) and timed their components with `diag.jl`.
Those candidates, `diag.jl` and the commands that reproduce those campaigns are in
this directory at commit `ca19cbde714fd71c4daa1581e5ff4a6756897d78`; the results are
on #872.

## Results (exclusive Intel Xeon E5-2640 v4, Julia 1.13.3)

The full record, with tables, is on #872 and #914. Raw samples are kept outside the
repository. In summary:

- When every source group is active, copying `storage`, `active` and `position`
  as whole arrays gives the state of the per-group loops, provided `active` and
  `position` of both iterates have one entry per group and the `storage`,
  `active` and `position` arrays of the two iterates do not overlap in memory.
  Iterates built by the constructors, `copy` and `similar` own independent
  arrays of these three kinds, while `copy` and `similar` share the
  layout (`groups`, `group_of`, layout cache); overlapping `storage`, `active` or
  `position` arrays passed to the seven-argument constructor are unsupported (see
  the `BlockIterate` docstring).
  The merged code checks the lengths and otherwise runs the loops. With an
  inactive source group, the copies can differ: the loops keep the destination's
  values there, such as a `-0.0`, unless the destination has the group active,
  when they clear it to `+0.0`.
- Copies inlined into `_copy_active!` gave it thread-local access and 9–13 calls,
  and were 6–10% slower than master on the smallest 1% restart cells; the merged
  code keeps them in a separate `@noinline` function.
- The merged source, targeted 21-round confirmation: median paired ratios to master
  are 1.001 at 1% activity, 1.003 at 10% and 0.142 at 100% (0.053–0.952 across
  cases). No case has a median above 1.05 with an interval above 1.0, and no case
  allocates more than master.
- Layout equality between independently built layouts costs 11–40 µs at
  K = 8192 singleton groups; sharing a layout through `similar`/`copy` or one
  `samegroups` call avoids it, so the layout cache is unchanged.
