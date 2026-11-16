#!/bin/bash
# One node, both Julia versions in sequence on the same pinned core, so that the version comparison shares the
# node. Submit with:
#   sbatch --partition=<p> --constraint=<cpu kind> [--exclusive | --cpus-per-task=4 --mem=16G] \
#     --export=ALL,KO_ROOT=<root>,KO_TAG=<tag>,KO_KIND=<cpu kind>,KO_TREES="base=<root>/trees/base"[,KO_SUITES=..,KO_ROUNDS=..] slurm/screen.sbatch
#SBATCH --job-name=ko-screen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH --output=ko-screen-%j.out
set -uo pipefail
: "${KO_ROOT:?}" "${KO_TAG:?}" "${KO_KIND:?}" "${KO_TREES:?}"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# the third core of the allocation (the first of a small one), away from core 0
CORES=$(taskset -cp $$ | sed 's/.*: //')
CORE=$(python3 -c "
import sys
cs=[]
for part in sys.argv[1].split(','):
    a,_,b=part.partition('-'); cs+=list(range(int(a),int(b or a)+1))
print(cs[2] if len(cs)>2 else cs[0])" "$CORES")
echo "job $SLURM_JOB_ID on $(hostname): allowed cores $CORES, pinned to $CORE"
status=0
for V in ${KO_JULIAS:-1.13.4 1.10.13}; do
  J=$HOME/julia/julia-$V/bin/julia
  OUT=$KO_ROOT/results/$KO_TAG/$KO_KIND-julia$V-job$SLURM_JOB_ID
  if [ -f "$OUT/DONE" ]; then echo "skip $OUT"; continue; fi
  mkdir -p "$OUT"
  lscpu > "$OUT/lscpu.txt"; uptime > "$OUT/uptime_start.txt"
  scontrol show job "$SLURM_JOB_ID" > "$OUT/scontrol_job.txt" 2>&1
  cp "$0" "$OUT/job_script.sh"
  ENVDIR=$KO_ROOT/experiment_kernelopt/env-v${V%.*}
  KO_OUTPUT=$OUT taskset -c "$CORE" "$J" --startup-file=no --threads=1 --check-bounds=no --project="$ENVDIR" \
    "$KO_ROOT/experiment_kernelopt/bench/run.jl" > "$OUT/run.log" 2>&1
  rc=$?
  uptime > "$OUT/uptime_end.txt"
  tail -3 "$OUT/run.log"
  if [ $rc -eq 0 ]; then touch "$OUT/DONE"; else echo "FAILED rc=$rc julia $V"; status=1; fi
done
exit $status
