#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=oneweek
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --open-mode=append
# Calibration of concurrency (orchestrator-owned): the same PB_TREES and filter run by 1, then 5, then 10
# (then NMAX) concurrent pinned processes on one exclusive node, sequentially. Env: RUN_DIR, ORDERS_FILE (one line),
# PB_FILTER, PB_ROUNDS, PHASES (e.g. "1 5 10").
set -uo pipefail
echo "job=$SLURM_JOB_ID node=$(hostname) start=$(date -Is)"
P=/scratch2/mkolar/pb-901x; JULIA=$HOME/julia/julia-1.13.3/bin/julia
export JULIA_DEPOT_PATH=$P/benchdepot:$P/shared-depot: JULIA_PKG_OFFLINE=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$RUN_DIR"; { lscpu; echo; squeue -h -w "$(hostname)" -o "%u %j %C"; } > node_start.txt
$JULIA --project=$P/benchenv -e 'using PrecompileTools, Statistics, Printf, Random' || exit 3
TREES=$(head -1 "$ORDERS_FILE")
S0=($(lscpu -p=CPU,SOCKET | grep -v '^#' | awk -F, '$2==0{print $1}')); S1=($(lscpu -p=CPU,SOCKET | grep -v '^#' | awk -F, '$2==1{print $1}'))
for n in $PHASES; do
  step=$(( (${#S0[@]} * 2) / n )); [ $step -lt 1 ] && step=1
  CORES=(); i=0
  while [ ${#CORES[@]} -lt $n ]; do idx=$(( (i / 2) * step )); if [ $((i % 2)) -eq 0 ]; then c=${S0[$idx]:-}; else c=${S1[$idx]:-}; fi; [ -z "$c" ] && break; CORES+=($c); i=$((i+1)); done
  echo "phase $n cores ${CORES[*]} $(date -Is)"
  for j in "${!CORES[@]}"; do
    PB_TREES="$TREES" PB_OUTPUT="$RUN_DIR/p${n}_c${CORES[$j]}" PB_CORE=${CORES[$j]} \
      taskset -c "${CORES[$j]}" $JULIA --threads=1 --check-bounds=no --startup-file=no --project=$P/benchenv "$RUN_DIR/screen901.jl" > "$RUN_DIR/p${n}_c${CORES[$j]}.log" 2>&1 &
  done
  wait
  echo "phase $n done $(date -Is)"
done
squeue -h -w "$(hostname)" -o "%u %j %C" > node_end.txt; echo "end=$(date -Is)"
