#!/usr/bin/env bash
set -euo pipefail
srun --exact -n1 -c1 --mem=8G --cpu-bind=cores bash "$1/driver/paired-run.sh" "$@"
