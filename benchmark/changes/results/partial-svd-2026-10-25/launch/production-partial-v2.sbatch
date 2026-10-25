#!/bin/bash -l
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --job-name=pb-opt-20261025-partial-full-v2
#SBATCH --output=production-partial-v2-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source common.sh
module load parallel/20240522
[[ ! -e partial2-before && ! -e partial2-after ]]
mkdir partial2-before partial2-after
export PB_NODE_FILE="$PWD/nodes-partial-v2-$SLURM_JOB_ID.txt"
scontrol show hostnames "$SLURM_JOB_NODELIST" > "$PB_NODE_FILE"
srun --exclusive --exact -N4 -n4 --ntasks-per-node=1 -c1 --mem=4G --cpu-bind=cores bash -c 'lscpu > "$SLURM_SUBMIT_DIR/hardware-partial2-$SLURMD_NODENAME.txt"'
set +e
parallel --jobs 4 --colsep '\t' --joblog "parallel-partial-v2-$SLURM_JOB_ID.tsv" bash "$PWD/worker-partial-v2.sh" '{%}' '{1}' '{2}' :::: tasks.tsv
workers=$?
analysis=0
for variant in before after; do
  python3 partial-v5/benchmark/campaign/summarize.py "partial2-$variant" "partial2-$variant-summary" || analysis=1
  python3 partial-v5/benchmark/campaign/validate.py "partial2-$variant" "partial2-$variant-summary/validation.json" || analysis=1
done
python3 partial-v5/benchmark/campaign/compare.py partial2-before partial2-after partial2-comparison || analysis=1
python3 changes_report.py partial2-before-summary partial2-after-summary partial2-comparison || analysis=1
printf 'workers=%s analysis=%s\n' "$workers" "$analysis" > "outcomes-partial2-$SLURM_JOB_ID.txt"
tar -czf "partial2-results-$SLURM_JOB_ID.tar.gz" partial2-before partial2-after partial2-before-summary partial2-after-summary partial2-comparison
[[ "$workers" -eq 0 && "$analysis" -eq 0 ]]
