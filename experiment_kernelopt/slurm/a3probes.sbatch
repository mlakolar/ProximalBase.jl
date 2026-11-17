#!/bin/bash
# Probes for the comparison of the two soft-threshold formulas on one node, Julia 1.13.4: the bitwise comparison of the
# trees (bench/equiv_trees.jl, with and without bounds checks), then KO_PROCS fresh processes of the offset sweep
# (bench/offset_probe.jl, library loops only, KO_OFFSET_SIZES) and of the ordinary-allocation probe (bench/alloc_probe.jl).
#   sbatch --exclusive --mem=0 --export=ALL,KO_ROOT=<root>,KO_TAG=<tag>,KO_KIND=<kind>,KO_REF=<tree>,KO_CAND=<tree> slurm/a3probes.sbatch
#SBATCH --job-name=ko-a3probes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
set -uo pipefail
: "${KO_ROOT:?}" "${KO_TAG:?}" "${KO_KIND:?}" "${KO_REF:?}" "${KO_CAND:?}"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export KO_OFFSET_SIZES=${KO_OFFSET_SIZES:-1024,2048,4096,8192,16384,32768,65536,262144,1048576}
export KO_OFFSET_IMPLS=${KO_OFFSET_IMPLS:-'^(l1|enet)/(ref|cand|refC)$|^copy$'}
CORES=$(taskset -cp $$ | sed 's/.*: //')
CORE=$(python3 -c "
import sys
cs=[]
for part in sys.argv[1].split(','):
    a,_,b=part.partition('-'); cs+=list(range(int(a),int(b or a)+1))
print(cs[2] if len(cs)>2 else cs[0])" "$CORES")
NODE=$(hostname -s)
echo "job $SLURM_JOB_ID on $NODE: pinned to $CORE; $(date)"
cat /sys/kernel/mm/transparent_hugepage/enabled /sys/kernel/mm/transparent_hugepage/defrag
grep -E "MemFree|AnonHugePages" /proc/meminfo; lscpu | grep -E "Model name|L1d|L2|L3|NUMA node\(s\)"
V=${KO_JULIA:-1.13.4}
J=$HOME/julia/julia-$V/bin/julia
B=$KO_ROOT/experiment_kernelopt/bench; E=$KO_ROOT/experiment_kernelopt/env-v${V%.*}
OUT=$KO_ROOT/results/$KO_TAG; mkdir -p "$OUT"
STEM=$OUT/$KO_KIND-$NODE-julia$V-$KO_CAND
cat "$KO_ROOT/trees/$KO_REF/TREE_SHA" "$KO_ROOT/trees/$KO_CAND/TREE_SHA" "$KO_ROOT/experiment_kernelopt/HARNESS_SHA" > "$STEM.shas"; echo "$SLURM_JOB_ID" > "$STEM.jobid"
status=0
for CB in no yes; do
  "$J" --startup-file=no --threads=1 --check-bounds=$CB --project="$E" "$B/equiv_trees.jl" "$KO_ROOT/trees/$KO_REF" "$KO_CAND=$KO_ROOT/trees/$KO_CAND" ${KO_EQUIV_EXTRA:-} > "$STEM-equiv-bounds$CB.log" 2>&1
  rc=$?; echo "equivalence, check-bounds=$CB: rc=$rc $(tail -1 "$STEM-equiv-bounds$CB.log")"; [ $rc -eq 0 ] || status=1
done
for P in $(seq 1 "${KO_PROCS:-3}"); do
  for S in offset alloc; do
    F=$STEM-$S-p$P.csv
    if [ $S = offset ]; then ARGS="${KO_ROUNDS:-11}"; else ARGS="${KO_ALLOC_ROUNDS:-7} ${KO_ALLOC_TRIALS:-16}"; fi
    taskset -c "$CORE" "$J" --startup-file=no --threads=1 --check-bounds=no --project="$E" "$B/${S}_probe.jl" "$KO_ROOT/trees/$KO_REF" "$KO_ROOT/trees/$KO_CAND" "$F" $ARGS > "$F.log" 2>&1
    rc=$?; echo "$S probe, process $P: rc=$rc $(tail -1 "$F.log")"; [ $rc -eq 0 ] || status=1
  done
done
echo "finished $(date)"
exit $status
