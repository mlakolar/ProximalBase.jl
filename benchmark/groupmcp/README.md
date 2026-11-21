# GroupMCP benchmark

`run.jl` times one source tree per process: GroupMCP of the tree under test (role X), a second compilation of it
(XC, the A/A control) and GroupNormL2 of a reference tree on the same groups (R, the cost reference). The header of
`run.jl` lists the cases, the operations, the verification before timing and the output files. `summarize.py`
reports one result directory, or compares a candidate directory with a legacy directory on the cases whose outputs
agree.

From the package root, with the base of the pull request as the reference tree and the frozen `legacy/full` tree:

```sh
mkdir -p /tmp/pb1069-base /tmp/pb1069-legacy
git archive <base sha> src | tar -x -C /tmp/pb1069-base
git archive eff21b6d7d33c622313122f46c6ee71693c101ee src | tar -x -C /tmp/pb1069-legacy
julia scripts/instantiate.jl benchmark
# a check of the outputs, without timing
PB_SOURCE="$PWD" PB_REFERENCE=/tmp/pb1069-base PB_OUTPUT=/tmp/pb1069-check PB_CHECK_ONLY=1 \
  julia --startup-file=no --project=benchmark benchmark/groupmcp/run.jl
# the candidate and the legacy tree, each in its own process, in both compile orders
for order in 0 1; do
  PB_SOURCE="$PWD" PB_REFERENCE=/tmp/pb1069-base PB_OUTPUT=/tmp/pb1069-cand-$order PB_REVERSE=$order PB_ROUNDS=12 \
    julia --startup-file=no --threads=1 --check-bounds=no --project=benchmark benchmark/groupmcp/run.jl
  PB_SOURCE=/tmp/pb1069-legacy PB_REFERENCE=/tmp/pb1069-base PB_OUTPUT=/tmp/pb1069-legacy-$order PB_REVERSE=$order PB_ROUNDS=12 PB_STRICT=0 \
    julia --startup-file=no --threads=1 --check-bounds=no --project=benchmark benchmark/groupmcp/run.jl
done
python3 benchmark/groupmcp/summarize.py /tmp/pb1069-cand-0               # X/R, A/A and allocations of one process
python3 benchmark/groupmcp/summarize.py /tmp/pb1069-cand-0 /tmp/pb1069-legacy-0
```

Run each Julia version (1.10 and 1.13) on an exclusive CPU node with one Julia thread, without other CPU-heavy work.
`PB_FILTER` selects cases by a regular expression on their ids (for example `Float64/m32/perm`), `PB_INPUTS` sets the
number of cycled inputs per case (default 8), and `PB_SAMPLES` and `PB_SECONDS` cap each trial (defaults 40 and
0.025). A full run has 1080 cases. With `PB_STRICT=0` a result of X that does not match the closed form of the radial map
is recorded as `fail` in `verification.csv` instead of stopping the run; `summarize.py` compares the two trees only on
cases whose outputs agree and lists the others with `agree = no`.
