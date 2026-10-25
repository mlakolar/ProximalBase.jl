#!/bin/bash -l
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:55:00
#SBATCH --job-name=pb-baseline-20261025-full-v2
#SBATCH --output=production-v2-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load parallel/20240522
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export JULIA_PKG_PRECOMPILE_AUTO=0 JULIA_NUM_PRECOMPILE_TASKS=1 JULIA_PKG_OFFLINE=true
export JULIA_DEPOT_PATH="$PWD/depot:/home1/mkolar/juliaPkg/runs/20261025-pb-readme-audit/depot:/home1/mkolar/juliaPkg/runs/20261025-pb-followups/depot:/home1/mkolar/juliaPkg/runs/20261024T1838Z-pb-review/depot-recovery:/home1/mkolar/juliaPkg/runs/20261024T1838Z-pb-review/depot-norm:/home1/mkolar/juliaPkg/runs/20261024T1838Z-pb-review/depot:/home1/mkolar/juliaPkg/depot:"
export PB_SOURCE="$PWD/source-v5"
export PB_SOURCE_SHA=2eab0955b4e33121582f2abbb73a7f87803b1631
export PB_HARNESS_SHA=c47dd77c599baf57642ed81dac0d79aa3a508e6f
export PB_OUTPUT="$PWD/full-v2" PB_MODE=full PB_SAMPLES=40 PB_SECONDS=0.1 PB_PROFILE_SECONDS=0.2
unset PB_CASE_FILTER
[[ ! -e "$PB_OUTPUT" ]]
mkdir "$PB_OUTPUT"
export PB_NODE_FILE="$PWD/nodes-full-v2-$SLURM_JOB_ID.txt"
scontrol show hostnames "$SLURM_JOB_NODELIST" > "$PB_NODE_FILE"
[[ $(wc -l < "$PB_NODE_FILE") -eq 4 ]]
srun --exclusive --exact -N4 -n4 --ntasks-per-node=1 -c1 --mem=4G --cpu-bind=cores \
    bash -c 'lscpu > "$SLURM_SUBMIT_DIR/hardware-full-v2-$SLURMD_NODENAME.txt"'
# Unchanged analysis and allocation-profile controls passed in13422184; reuse.
set +e
parallel --jobs 4 --colsep '\t' --joblog "parallel-full-v2-$SLURM_JOB_ID.tsv" \
    bash "$PWD/worker-v2.sh" '{%}' '{1}' '{2}' :::: tasks-v1.tsv
workers_status=$?
python3 "$PB_SOURCE/benchmark/campaign/summarize.py" "$PB_OUTPUT" "$PWD/full-v2-summary" --expected-runs=10
summary_status=$?
set -e
printf 'workers=%s summary=%s\n' "$workers_status" "$summary_status" > "outcomes-full-v2-$SLURM_JOB_ID.txt"
tar -czf "full-v2-results-$SLURM_JOB_ID.tar.gz" full-v2 full-v2-summary
[[ "$workers_status" -eq 0 && "$summary_status" -eq 0 ]]
