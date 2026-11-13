#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=oneweek
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --open-mode=append
# Common screening job of the #901 experiment (orchestrator-owned). Submit with submit_screen.sh.
# Env (exported by the submit script):
#   RUN_DIR    results directory (unique per job)
#   ORDERS     ';'-separated list of PB_TREES strings, separated by '|' (one per role order), e.g.
#              "base=/t/a;C=/t/c|C=/t/c;base=/t/a"
#   SHARDS     shards per order (each (order, shard) is one process)
#   NCONC      concurrent processes (pinned to spaced cores, one per core)
#   PB_FILTER, PB_ROUNDS, PB_SAMPLES, PB_TARGET_NS   passed to screen901.jl
#   SCRIPT     path of screen901.jl (default: the copy in the run dir)
set -uo pipefail
echo "restart_count=${SLURM_RESTART_COUNT:-0} job=$SLURM_JOB_ID node=$(hostname) start=$(date -Is)"
P=/scratch2/mkolar/pb-901x
JULIA=$HOME/julia/julia-1.13.3/bin/julia
export JULIA_DEPOT_PATH=$P/benchdepot:$P/shared-depot: JULIA_PKG_OFFLINE=true
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
SCRIPT=${SCRIPT:-$RUN_DIR/screen901.jl}
mkdir -p "$RUN_DIR"; cd "$RUN_DIR"
{ lscpu; echo; numactl -H 2>/dev/null; echo; squeue -h -w "$(hostname)" -o "%u %j %C"; } > node_start.txt
$JULIA --project=$P/benchenv -e 'using PrecompileTools, Statistics, Printf, Random' || exit 3
# spaced cores, alternating sockets: physical core ids from lscpu (one thread per core on Discovery)
mapfile -t ALL < <(lscpu -p=CPU,SOCKET | grep -v '^#' | sort -t, -k2,2n -k1,1n | awk -F, '{print $2" "$1}')
NCPU=${#ALL[@]}
S0=($(lscpu -p=CPU,SOCKET | grep -v '^#' | awk -F, '$2==0{print $1}')); S1=($(lscpu -p=CPU,SOCKET | grep -v '^#' | awk -F, '$2==1{print $1}'))
CORES=()
step=$(( (${#S0[@]} * 2) / NCONC )); [ $step -lt 1 ] && step=1
i=0; while [ ${#CORES[@]} -lt $NCONC ]; do
  idx=$(( (i / 2) * step )); if [ $((i % 2)) -eq 0 ]; then c=${S0[$idx]:-}; else c=${S1[$idx]:-}; fi
  [ -z "$c" ] && break; CORES+=($c); i=$((i+1)); done
echo "cores: ${CORES[*]} (ncpu=$NCPU)"
mapfile -t ORD < "$ORDERS_FILE"
TASKS=()
for o in "${!ORD[@]}"; do for s in $(seq 1 "$SHARDS"); do TASKS+=("$o $s"); done; done
worker() {
  local slot=$1 core=${CORES[$1]}
  for t in $(seq "$slot" "${#CORES[@]}" $((${#TASKS[@]} - 1))); do
    read -r o s <<< "${TASKS[$t]}"
    local out="$RUN_DIR/o${o}_s${s}"
    [ -f "$out/DONE" ] && continue
    rm -rf "$out"
    PB_TREES="${ORD[$o]}" PB_OUTPUT="$out" PB_SHARD="$s/$SHARDS" PB_CORE=$core \
      taskset -c "$core" $JULIA --threads=1 --check-bounds=no --startup-file=no --project=$P/benchenv "$SCRIPT" \
      > "$RUN_DIR/o${o}_s${s}.log" 2>&1 && touch "$out/DONE"
    echo "task o${o}_s${s} core $core exit $? $(date -Is)"
  done
}
for slot in $(seq 0 $((${#CORES[@]} - 1))); do worker "$slot" & done
wait
squeue -h -w "$(hostname)" -o "%u %j %C" > node_end.txt
echo "end=$(date -Is)"
