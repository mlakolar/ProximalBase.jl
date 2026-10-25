#!/bin/bash -l
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:12:00
#SBATCH --job-name=pb-opt-20261025-entropy-full-v2
#SBATCH --output=production-entropy-v2-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source common.sh
module load parallel/20240522
[[ ! -e entropy2-before && ! -e entropy2-after ]]
mkdir entropy2-before entropy2-after
# Preserve valid LTS pairs; replace only the invalid Julia1.13 comparison.
cp entropy-before/julia-1.10.12-* entropy2-before/
cp entropy-after/julia-1.10.12-* entropy2-after/
printf "%s\n" "Reused LTS ten pairs from13422635; runtime source verified in13422706" > entropy2-reused-LTS.txt
export PB_NODE_FILE="$PWD/nodes-entropy-v2-$SLURM_JOB_ID.txt"
scontrol show hostnames "$SLURM_JOB_NODELIST" > "$PB_NODE_FILE"
srun --exclusive --exact -N4 -n4 --ntasks-per-node=1 -c1 --mem=4G --cpu-bind=cores bash -c 'lscpu > "$SLURM_SUBMIT_DIR/hardware-entropy2-$SLURMD_NODENAME.txt"'
set +e
parallel --jobs 4 --colsep '\t' --joblog "parallel-entropy-v2-$SLURM_JOB_ID.tsv" bash "$PWD/worker-entropy-v2.sh" '{%}' '{1}' '{2}' :::: tasks-entropy-fixed.tsv
workers=$?
analysis=0
for variant in before after; do
  python3 entropy-v1/benchmark/campaign/summarize.py "entropy2-$variant" "entropy2-$variant-summary" || analysis=1
  python3 entropy-v1/benchmark/campaign/validate.py "entropy2-$variant" "entropy2-$variant-summary/validation.json" || analysis=1
done
python3 entropy-v1/benchmark/campaign/compare.py entropy2-before entropy2-after entropy2-comparison || analysis=1
python3 changes_report.py entropy2-before-summary entropy2-after-summary entropy2-comparison || analysis=1
printf 'workers=%s analysis=%s\n' "$workers" "$analysis" > "outcomes-entropy2-$SLURM_JOB_ID.txt"
tar -czf "entropy2-results-$SLURM_JOB_ID.tar.gz" entropy2-before entropy2-after entropy2-before-summary entropy2-after-summary entropy2-comparison
[[ "$workers" -eq 0 && "$analysis" -eq 0 ]]
