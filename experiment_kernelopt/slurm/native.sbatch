#!/bin/bash
# ProximalBase's own paired harnesses (benchmark/*.jl of the baseline tree) on a candidate tree, in both role
# orders, for one Julia version: the gate that the maintainers apply.
#   sbatch --partition=<p> --constraint=<kind> --exclusive --mem=0 \
#     --export=ALL,KO_ROOT=<root>,KO_TAG=<tag>,KO_KIND=<kind>,KO_CAND=<tree name>,KO_JULIA=<version>,KO_NATIVE="norml1_infinite_weight l2_norms" slurm/native.sbatch
#SBATCH --job-name=ko-native
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=ko-native-%j.out
set -uo pipefail
: "${KO_ROOT:?}" "${KO_TAG:?}" "${KO_KIND:?}" "${KO_CAND:?}" "${KO_JULIA:?}" "${KO_NATIVE:?}"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
V=$KO_JULIA
J=$HOME/julia/julia-$V/bin/julia
BENCH=$KO_ROOT/trees/base/benchmark
ENVDIR=$KO_ROOT/benchenv-v${V%.*}
mkdir -p "$ENVDIR"; [ -f "$ENVDIR/Project.toml" ] || cp "$BENCH/Project.toml" "$ENVDIR/"
JULIA_NUM_PRECOMPILE_TASKS=1 "$J" --startup-file=no -t1 --project="$ENVDIR" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile(); using BenchmarkTools, PrecompileTools' || { echo "environment failed"; exit 1; }
CORES=$(taskset -cp $$ | sed 's/.*: //')
CORE=$(python3 -c "
import sys
cs=[]
for part in sys.argv[1].split(','):
    a,_,b=part.partition('-'); cs+=list(range(int(a),int(b or a)+1))
print(cs[2] if len(cs)>2 else cs[0])" "$CORES")
echo "job $SLURM_JOB_ID on $(hostname): pinned to $CORE; candidate $KO_CAND; julia $V"
status=0
for H in $KO_NATIVE; do
  for ORDER in fwd rev; do
    if [ $ORDER = fwd ]; then FIRST=base; SECOND=$KO_CAND; else FIRST=$KO_CAND; SECOND=base; fi
    OUT=$KO_ROOT/results/$KO_TAG/$KO_KIND-julia$V-$KO_CAND-$H-$ORDER
    if [ -f "$OUT/DONE" ]; then echo "skip $OUT"; continue; fi
    mkdir -p "$OUT"; lscpu > "$OUT/lscpu.txt"; uptime > "$OUT/uptime_start.txt"; echo "$SLURM_JOB_ID" > "$OUT/JOBID"
    printf 'first = %s @ %s\nsecond = %s @ %s\n' "$FIRST" "$(cat $KO_ROOT/trees/$FIRST/TREE_SHA)" "$SECOND" "$(cat $KO_ROOT/trees/$SECOND/TREE_SHA)" > "$OUT/roles.txt"
    EXTRA=()
    [ "$H" = l2_cold_paths ] && EXTRA+=(PB_EXPECT_DIFF=same)
    [ -n "${KO_NATIVE_FILTER:-}" ] && [ "$H" = "${KO_NATIVE_FILTER_FOR:-}" ] && EXTRA+=("PB_FILTER=$KO_NATIVE_FILTER")
    if [ "$H" = ridge910 ]; then
      # the harness of #910: a base, named candidates and its own A/A control; PB_REVERSE swaps the timing order
      CANDS="$KO_CAND=$KO_ROOT/trees/$KO_CAND"
      for extra in ${KO_NATIVE_MORE:-}; do CANDS="$CANDS,$extra=$KO_ROOT/trees/$extra"; done
      env PB_BASE=$KO_ROOT/trees/base PB_CANDIDATES="$CANDS" PB_REVERSE=$([ $ORDER = rev ] && echo 1 || echo 0) PB_OUTPUT=$OUT PB_ROUNDS=${KO_ROUNDS:-21} \
        taskset -c "$CORE" "$J" --startup-file=no --threads=1 --check-bounds=no --project="$ENVDIR" "$BENCH/$H.jl" > "$OUT/run.log" 2>&1
    else
    env "${EXTRA[@]}" PB_FIRST=$KO_ROOT/trees/$FIRST PB_SECOND=$KO_ROOT/trees/$SECOND PB_OUTPUT=$OUT PB_ROUNDS=${KO_ROUNDS:-21} \
      taskset -c "$CORE" "$J" --startup-file=no --threads=1 --check-bounds=no --project="$ENVDIR" "$BENCH/$H.jl" > "$OUT/run.log" 2>&1
    fi
    rc=$?
    uptime > "$OUT/uptime_end.txt"
    echo "$H $ORDER ($FIRST first): rc=$rc $(tail -1 "$OUT/run.log" | cut -c1-120)"
    if [ $rc -eq 0 ]; then touch "$OUT/DONE"; else status=1; fi
  done
done
exit $status
