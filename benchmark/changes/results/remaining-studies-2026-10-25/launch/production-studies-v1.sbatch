#!/bin/bash -l
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --job-name=pb-opt-20261025-studies-full-v1
#SBATCH --output=production-studies-v1-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source common.sh
module load parallel/20240522
export PB_OUTPUT="$PWD/studies-full-v1" PB_MODE=full PB_SAMPLES=40 PB_SECONDS=0.05 PB_PROFILE_SECONDS=0.1
export PB_SOURCE_SHA=0111d1bb4ca51b40c28d8ea5dab0f63be4d76e6a PB_HARNESS_SHA=42d6119b0c3f4ad3dc63b4574b3f6466516458b5
export PB_EXPECTED_PACKAGE_ROOT="$PWD/before" PB_MEASUREMENT_SCRIPT="$PWD/studies-v3/benchmark/changes/remaining.jl"
[[ ! -e "$PB_OUTPUT" ]]
mkdir "$PB_OUTPUT"
export PB_NODE_FILE="$PWD/nodes-studies-v1-$SLURM_JOB_ID.txt"
scontrol show hostnames "$SLURM_JOB_NODELIST" > "$PB_NODE_FILE"
set +e
parallel --jobs 4 --colsep '\t' --joblog "parallel-studies-v1-$SLURM_JOB_ID.tsv" bash "$PWD/worker-studies-v1.sh" '{%}' '{1}' '{2}' :::: tasks.tsv
workers=$?
analysis=0
python3 studies-v3/benchmark/campaign/summarize.py studies-full-v1 studies-summary-v1 || analysis=1
python3 studies-v3/benchmark/campaign/validate.py studies-full-v1 studies-summary-v1/validation.json || analysis=1
python3 study_report.py studies-full-v1 studies-summary-v1 || analysis=1
printf 'workers=%s analysis=%s\n' "$workers" "$analysis" > "outcomes-studies-$SLURM_JOB_ID.txt"
tar -czf "studies-results-$SLURM_JOB_ID.tar.gz" studies-full-v1 studies-summary-v1
[[ "$workers" -eq 0 && "$analysis" -eq 0 ]]
