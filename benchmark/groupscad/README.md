# GroupSCAD benchmark

`run.jl` times one source tree per process: GroupSCAD of the tree under test (role X), a second compilation of it
(XC, the A/A control) and GroupNormL2 of a reference tree on the same groups (R, the cost reference, a different
function). The header of `run.jl` lists the cases, the operations, the verification before timing and the output
files. `summarize.py` reports one result directory, or compares a candidate directory with a legacy directory on the
cases whose saved outputs agree; the outputs are saved for the first input of each case, while the timed samples rotate
through the input pool (with fewer sample setups than `PB_INPUTS`, some inputs are not timed).

The verification of X against the BigFloat radial map uses `test/scad_reference.jl` of the tree that holds `run.jl`,
whichever tree `PB_SOURCE` names.

From the package root, with the base of the pull request as the reference tree and the frozen `legacy/full` tree:

```sh
mkdir -p /tmp/pb1070-base /tmp/pb1070-legacy
git archive <base sha> src | tar -x -C /tmp/pb1070-base
git archive eff21b6d7d33c622313122f46c6ee71693c101ee src | tar -x -C /tmp/pb1070-legacy
julia scripts/instantiate.jl benchmark
# a check of the outputs, without timing
PB_SOURCE="$PWD" PB_REFERENCE=/tmp/pb1070-base PB_OUTPUT=/tmp/pb1070-check PB_CHECK_ONLY=1 \
  julia --startup-file=no --project=benchmark benchmark/groupscad/run.jl
# the candidate and the legacy tree, each in its own process, in both module load orders
for order in 0 1; do
  PB_SOURCE="$PWD" PB_REFERENCE=/tmp/pb1070-base PB_OUTPUT=/tmp/pb1070-cand-$order PB_REVERSE=$order PB_ROUNDS=12 \
    julia --startup-file=no --threads=1 --check-bounds=no --project=benchmark benchmark/groupscad/run.jl
  PB_SOURCE=/tmp/pb1070-legacy PB_REFERENCE=/tmp/pb1070-base PB_OUTPUT=/tmp/pb1070-legacy-$order PB_REVERSE=$order PB_ROUNDS=12 PB_STRICT=0 \
    julia --startup-file=no --threads=1 --check-bounds=no --project=benchmark benchmark/groupscad/run.jl
done
python3 benchmark/groupscad/summarize.py /tmp/pb1070-cand-0               # X/R, A/A and allocations of one process
python3 benchmark/groupscad/summarize.py /tmp/pb1070-cand-0 /tmp/pb1070-legacy-0
```

Run each Julia version (1.10 and 1.13) on an exclusive CPU node with one Julia thread, without other CPU-heavy work.
`PB_FILTER` selects cases by a regular expression on their ids (for example `Float64/m32/perm/above`), `PB_INPUTS` sets the
number of cycled inputs per case (default 8), `PB_SAMPLES` caps the samples of each trial (default 40), and
`PB_SECONDS` is the elapsed-time threshold that BenchmarkTools checks between samples (default 0.025). A full run has 1440 cases.
The steps are `below`, `at` and `above` for `γ` below, at and above `a - 1`, and the regimes are `zero`, `soft`, `middle`
and `kept`. With `PB_STRICT=0` a result of X that does not match the reference is recorded as `fail` in `verification.csv`
instead of stopping the run; `summarize.py` compares the two trees only on cases whose outputs agree and lists the others
with `agree = no`.
