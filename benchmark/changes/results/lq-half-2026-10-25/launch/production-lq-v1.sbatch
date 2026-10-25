#!/bin/bash -l
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --job-name=pb-opt-20261025-lq-full-v1
#SBATCH --output=production-lq-v1-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source common.sh
module load parallel/20240522
[[ ! -e lq-before && ! -e lq-after ]]
mkdir lq-before lq-after
export PB_NODE_FILE="$PWD/nodes-lq-v1-$SLURM_JOB_ID.txt"
scontrol show hostnames "$SLURM_JOB_NODELIST" > "$PB_NODE_FILE"
srun --exclusive --exact -N4 -n4 --ntasks-per-node=1 -c1 --mem=4G --cpu-bind=cores bash -c 'lscpu > "$SLURM_SUBMIT_DIR/hardware-lq-$SLURMD_NODENAME.txt"'
set +e
parallel --jobs 4 --colsep '\t' --joblog "parallel-lq-v1-$SLURM_JOB_ID.tsv" bash "$PWD/worker-lq-v1.sh" '{%}' '{1}' '{2}' :::: tasks.tsv
workers=$?
analysis=0
for variant in before after; do
  python3 lq-v2/benchmark/campaign/summarize.py "lq-$variant" "lq-$variant-summary" || analysis=1
  python3 lq-v2/benchmark/campaign/validate.py "lq-$variant" "lq-$variant-summary/validation.json" || analysis=1
done
python3 lq-v2/benchmark/campaign/compare.py lq-before lq-after lq-comparison || analysis=1
python3 changes_report.py lq-before-summary lq-after-summary lq-comparison || analysis=1
printf 'workers=%s analysis=%s\n' "$workers" "$analysis" > "outcomes-lq-$SLURM_JOB_ID.txt"
tar -czf "lq-results-$SLURM_JOB_ID.tar.gz" lq-before lq-after lq-before-summary lq-after-summary lq-comparison
[[ "$workers" -eq 0 && "$analysis" -eq 0 ]]
