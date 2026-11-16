#!/bin/bash
# Formal paired comparison of source trees on one node: for each Julia version, KO_PROCS fresh processes. Processes
# 2k-1 and 2k compile the trees in a rotation of KO_TREE_LIST and in its reversal, with the A/A control (the second
# compilation of the reference) last in the first and first in the second. Each process has
# KO_ROUNDS alternating rounds.
#   sbatch --partition=<p> --constraint=<kind> --exclusive --mem=0 \
#     --export=ALL,KO_ROOT=<root>,KO_TAG=<tag>,KO_KIND=<kind>,KO_REF=base,KO_TREE_LIST="base A B",KO_SUITES=l1,l2,group,solver slurm/formal.sbatch
#SBATCH --job-name=ko-formal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=10:00:00
#SBATCH --output=ko-formal-%j.out
set -uo pipefail
: "${KO_ROOT:?}" "${KO_TAG:?}" "${KO_KIND:?}" "${KO_TREE_LIST:?}" "${KO_REF:?}"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export KO_ROUNDS=${KO_ROUNDS:-21} KO_BATCH_MS=${KO_BATCH_MS:-2} KO_SAMPLES=${KO_SAMPLES:-3}
CORES=$(taskset -cp $$ | sed 's/.*: //')
CORE=$(python3 -c "
import sys
cs=[]
for part in sys.argv[1].split(','):
    a,_,b=part.partition('-'); cs+=list(range(int(a),int(b or a)+1))
print(cs[2] if len(cs)>2 else cs[0])" "$CORES")
echo "job $SLURM_JOB_ID on $(hostname): allowed cores $CORES, pinned to $CORE"
read -r -a NAMES <<< "$KO_TREE_LIST"
N=${#NAMES[@]}
NPROC=${KO_PROCS:-4}
# processes 2k-1 and 2k use rotation (k-1)·STEP of the list and its reversal
STEP=$(( N / ((NPROC + 1) / 2) > 0 ? N / ((NPROC + 1) / 2) : 1 ))
status=0
for V in ${KO_JULIAS:-1.13.4 1.10.13}; do
  J=$HOME/julia/julia-$V/bin/julia
  ENVDIR=$KO_ROOT/experiment_kernelopt/env-v${V%.*}
  for P in $(seq 1 "$NPROC"); do
    ORDER=()
    ROT=$(( ((P - 1) / 2) * STEP ))
    for i in $(seq 0 $((N-1))); do ORDER+=("${NAMES[$(( (i + ROT) % N ))]}"); done
    CF=0
    if [ $((P % 2)) -eq 0 ]; then
      REV=(); for ((i=N-1; i>=0; i--)); do REV+=("${ORDER[$i]}"); done; ORDER=("${REV[@]}"); CF=1
    fi
    TREES=""
    for nm in "${ORDER[@]}"; do TREES="$TREES$nm=$KO_ROOT/trees/$nm;"; done
    OUT=$KO_ROOT/results/$KO_TAG/$KO_KIND-julia$V-p$P
    if [ -f "$OUT/DONE" ]; then echo "skip $OUT"; continue; fi
    mkdir -p "$OUT"
    lscpu > "$OUT/lscpu.txt"; uptime > "$OUT/uptime_start.txt"
    scontrol show job "$SLURM_JOB_ID" > "$OUT/scontrol_job.txt" 2>&1
    cp "$0" "$OUT/job_script.sh"; echo "$SLURM_JOB_ID" > "$OUT/JOBID"
    KO_TREES="$TREES" KO_CONTROL_FIRST=$CF KO_OUTPUT=$OUT taskset -c "$CORE" "$J" --startup-file=no --threads=1 --check-bounds=no --project="$ENVDIR" \
      "$KO_ROOT/experiment_kernelopt/bench/run.jl" > "$OUT/run.log" 2>&1
    rc=$?
    uptime > "$OUT/uptime_end.txt"
    echo "julia $V process $P order ${ORDER[*]} control_first=$CF: $(tail -1 "$OUT/run.log")"
    if [ $rc -eq 0 ]; then touch "$OUT/DONE"; else echo "FAILED rc=$rc"; status=1; fi
  done
done
exit $status
