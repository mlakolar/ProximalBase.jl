# GroupSCAD benchmark

`run.jl` times one source tree per process: GroupSCAD of the tree under test (role X), a second compilation of it
(XC, the A/A control) and GroupNormL2 of a reference tree on the same groups (R, the cost reference, a different
function). The header of `run.jl` lists the cases, the operations, the verification before timing and the output
files. `summarize.py` reports one result directory, or compares a candidate directory with a legacy directory on the
cases whose saved outputs agree for every input (`outputs_all/`); every input of each case is verified against the BigFloat
reference before timing, while the timed samples rotate through the input pool (with fewer sample setups than `PB_INPUTS`,
some inputs are not timed). A BenchmarkTools run that throws the `InexactError` of `sizehint!` (or ends with fewer than three samples) is repeated
(`PB_RETRIES` attempts in all, default 3, at least 1; `PB_SAMPLES` at least 3); a repeated run and every cell-round whose attempts were
exhausted are listed in `failures.csv`, and an exhausted cell-round is left out of `raw.csv` and `ratios.csv` for all roles.
`summarize.py` compares every input for directories in the current format (declared inputs and `outputs_all/`), after checking
that the first-input file and the `outputs_all/` file of each case are present with exactly the expected byte length; it
reports a missing, truncated or inconsistent record as `incompatible` and falls back to the first input only for a
recognizable older directory (neither declared inputs nor `outputs_all/`), which it labels as such.

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

## Diagnosis of the in-place, value and compact-block costs

`diag_inplace.jl` runs, in one process, a baseline tree (`PB_SOURCE`) with a second and a third compilation of it (A/A controls),
the archived tree, GroupNormL2, and optionally a tree for the value probes (`PB_VARIANT`) and one for the in-place probes and the
compact block sweeps (`PB_VARIANT_INPLACE`, with a duplicate compilation), on the inputs of `run.jl`: the branch taken per group,
the in-place proximal map and its components, the value and fused value for small groups, and the compact block sweeps with
`out === u` and into separate outputs. The header of `diag_inplace.jl` lists the probes, the cases and the output files;
`make_variants.py` makes the two experimental variants from a baseline whose source has the pattern that it edits.
`diag.sh` runs it from a package root (a clean `git archive`), with the trees given by environment variables. The head of
the pull request is compared with the baseline `9d2fefc71` by passing it as both variant trees:

```sh
git archive <head sha> | (mkdir -p /tmp/pb1070-pkg && tar -x -C /tmp/pb1070-pkg)
for t in head:<head sha> base9d:9d2fefc71 legacy:eff21b6d7d33c622313122f46c6ee71693c101ee ref:a6b16d63d9c5102b1ac82d9995014145681d4343; do
  mkdir -p /tmp/pb1070-${t%%:*} && git archive ${t#*:} src | tar -x -C /tmp/pb1070-${t%%:*}
done
cd /tmp/pb1070-pkg
PB_SOURCE=/tmp/pb1070-base9d PB_VARIANT=/tmp/pb1070-head PB_VARIANT_INPLACE=/tmp/pb1070-head \
  PB_LEGACY=/tmp/pb1070-legacy PB_REFERENCE=/tmp/pb1070-ref PB_OUTPUT=/tmp/pb1070-diag JULIA=julia PB_REPS=8 \
  bash benchmark/groupscad/diag.sh
```

(The `git archive` commands run in a clone of the repository.) A filter such as `PB_FILTER='/blocks$'` restricts the run to the compact
block sweeps. The tree digests and the module load order that the run used are written to `provenance.txt`.

`test_summarize.py` is a self-test of the comparison rules of `summarize.py` (`python3 benchmark/groupscad/test_summarize.py`).
