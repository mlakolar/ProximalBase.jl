#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=oneweek
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --open-mode=append
# Comprehensive campaign job (orchestrator). TASKS_FILE lines: "<subdir> <script> <first root> <second root> <shard i/n> [extra env k=v ...]"
# run by NCONC workers pinned to spaced cores (both sockets). Each task writes RUN_DIR/<subdir>/ and a DONE marker.
set -uo pipefail
echo "restart=${SLURM_RESTART_COUNT:-0} job=$SLURM_JOB_ID node=$(hostname) start=$(date -Is)"
P=/scratch2/mkolar/pb-901x; JULIA=$HOME/julia/julia-1.13.3/bin/julia
export JULIA_DEPOT_PATH=$P/benchdepot:$P/shared-depot: JULIA_PKG_OFFLINE=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$RUN_DIR"; { lscpu; echo; squeue -h -w "$(hostname -s)" -o "%u %j %C"; } > node_start_$SLURM_JOB_ID.txt
$JULIA --project=$P/benchenv -e 'using PrecompileTools, Statistics, Printf, Random, BenchmarkTools' || exit 3
S0=($(lscpu -p=CPU,SOCKET | grep -v '^#' | awk -F, '$2==0{print $1}')); S1=($(lscpu -p=CPU,SOCKET | grep -v '^#' | awk -F, '$2==1{print $1}'))
step=$(( (${#S0[@]} * 2) / NCONC )); [ $step -lt 1 ] && step=1
CORES=(); i=0
while [ ${#CORES[@]} -lt $NCONC ]; do idx=$(( (i / 2) * step )); if [ $((i % 2)) -eq 0 ]; then c=${S0[$idx]:-}; else c=${S1[$idx]:-}; fi; [ -z "$c" ] && break; CORES+=($c); i=$((i+1)); done
echo "cores ${CORES[*]}"
mapfile -t TASKS < "$TASKS_FILE"
worker() {
  local slot=$1 core=${CORES[$1]}
  for t in $(seq "$slot" "${#CORES[@]}" $((${#TASKS[@]} - 1))); do
    read -r sub script first second shard extra <<< "${TASKS[$t]}"
    [ -f "$RUN_DIR/$sub/DONE" ] && continue
    rm -rf "$RUN_DIR/$sub"; mkdir -p "$RUN_DIR/$sub"
    echo "job $SLURM_JOB_ID node $(hostname -s) core $core" > "$RUN_DIR/$sub/job_meta.txt"
    env $extra PB_FIRST="$first" PB_SECOND="$second" PB_SHARD="$shard" PB_OUTPUT="$RUN_DIR/$sub" PB_TOLERANT=1 \
      taskset -c "$core" $JULIA --threads=1 --check-bounds=no --startup-file=no --project=$P/benchenv "$RUN_DIR/$script" \
      > "$RUN_DIR/$sub.log" 2>&1 && touch "$RUN_DIR/$sub/DONE"
    echo "task $sub core $core exit $? $(date -Is)"
  done
}
for slot in $(seq 0 $((${#CORES[@]} - 1))); do worker "$slot" & done
wait
squeue -h -w "$(hostname -s)" -o "%u %j %C" > node_end_$SLURM_JOB_ID.txt; echo "end=$(date -Is)"
