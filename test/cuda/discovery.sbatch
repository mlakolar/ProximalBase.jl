#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --job-name=pb-gpu103
#SBATCH --output=slurm-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export JULIA_NUM_THREADS=4 JULIA_NUM_PRECOMPILE_TASKS=4
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export JULIA_DEPOT_PATH="${SLURM_SUBMIT_DIR}/depot:/home1/mkolar/juliaPkg/depot"
JULIA=/home1/mkolar/juliaPkg/julia-1.10.12/bin/julia
nvidia-smi
"$JULIA" --project=source/test/cuda source/test/cuda/instantiate.jl
"$JULIA" --project=source/test/cuda source/test/cuda/runtests.jl
# Normal CI's reference backend must still pass the shared cases.
"$JULIA" --project=source/test/cuda -e 'include("source/test/gpu.jl")'
