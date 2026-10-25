#!/bin/bash -l
#SBATCH --account=mkolar_1314
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --dependency=afterany:13422265
#SBATCH --job-name=pb-baseline-20261025-analysis-v1
#SBATCH --output=analysis-v1-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONDONTWRITEBYTECODE=1
code="$PWD/analysis-v1/benchmark/campaign"
python3 "$code/check_analysis.py" > "analysis-controls-v1-$SLURM_JOB_ID.log" 2>&1
python3 "$code/summarize.py" "$PWD/full-v2" "$PWD/analysis-v1-summary" --expected-runs=10
python3 "$code/validate.py" "$PWD/full-v2" "$PWD/analysis-v1-summary/validation.json"
python3 insights.py "$PWD/full-v2" "$PWD/analysis-v1-summary"
cp "analysis-controls-v1-$SLURM_JOB_ID.log" analysis-v1-summary/
tar -czf "analysis-v1-results-$SLURM_JOB_ID.tar.gz" analysis-v1-summary
