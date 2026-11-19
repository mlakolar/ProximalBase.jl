#!/usr/bin/env bash
set -euo pipefail
root=$1
srun --exact -n1 -c1 --mem=8G --cpu-bind=cores bash "$root/driver/run-paired.sh" "$root" "$root/results/$SLURM_JOB_ID"
