#!/bin/bash
# Memory-layout probe (bench/layout_probe.jl, or KO_SCRIPT=offset_probe.jl) of two trees on one node: KO_PROCS fresh processes per Julia version.
#   sbatch --exclusive --mem=0 --export=ALL,KO_ROOT=<root>,KO_TAG=<tag>,KO_KIND=<kind>,KO_REF=base,KO_CAND=A3 slurm/probe.sbatch
#SBATCH --job-name=ko-probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
set -uo pipefail
: "${KO_ROOT:?}" "${KO_TAG:?}" "${KO_KIND:?}" "${KO_REF:?}" "${KO_CAND:?}"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CORES=$(taskset -cp $$ | sed 's/.*: //')
CORE=$(python3 -c "
import sys
cs=[]
for part in sys.argv[1].split(','):
    a,_,b=part.partition('-'); cs+=list(range(int(a),int(b or a)+1))
print(cs[2] if len(cs)>2 else cs[0])" "$CORES")
NODE=$(hostname -s)
echo "job $SLURM_JOB_ID on $NODE: pinned to $CORE; $(date)"
cat /sys/kernel/mm/transparent_hugepage/enabled /sys/kernel/mm/transparent_hugepage/defrag; cat /proc/sys/kernel/numa_balancing
numactl --hardware 2>/dev/null | head -6; grep -E "MemFree|AnonHugePages" /proc/meminfo; lscpu | grep -E "Model name|L2|L3|NUMA"
OUT=$KO_ROOT/results/$KO_TAG; mkdir -p "$OUT"
status=0
for V in ${KO_JULIAS:-1.13.4 1.10.13}; do
  J=$HOME/julia/julia-$V/bin/julia
  for P in $(seq 1 "${KO_PROCS:-3}"); do
    F=$OUT/$KO_KIND-$NODE-julia$V-$KO_CAND-p$P.csv
    taskset -c "$CORE" "$J" --startup-file=no --threads=1 --check-bounds=no --project="$KO_ROOT/experiment_kernelopt/env-v${V%.*}" \
      "$KO_ROOT/experiment_kernelopt/bench/${KO_SCRIPT:-layout_probe.jl}" "$KO_ROOT/trees/$KO_REF" "$KO_ROOT/trees/$KO_CAND" "$F" "${KO_ROUNDS:-15}" > "$F.log" 2>&1
    rc=$?; echo "julia $V process $P: rc=$rc $(tail -1 "$F.log")"; [ $rc -eq 0 ] || status=1
    echo "$SLURM_JOB_ID" > "$F.jobid"
  done
done
exit $status
