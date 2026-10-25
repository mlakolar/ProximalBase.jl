#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=pb-gpu110
#SBATCH --output=slurm-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export JULIA_NUM_THREADS=1 JULIA_NUM_PRECOMPILE_TASKS=4
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PB_BLAS_THREADS=4
export JULIA_DEPOT_PATH="$SLURM_SUBMIT_DIR/depot:/home1/mkolar/juliaPkg/runs/20261026-pb-gpu110/depot:/home1/mkolar/juliaPkg/runs/20261025-pb-gpu103/depot:/home1/mkolar/juliaPkg/depot"
JULIA=/home1/mkolar/juliaPkg/julia-1.10.12/bin/julia
nvidia-smi > hardware.txt
"$JULIA" --project=source/test/cuda source/test/cuda/instantiate.jl
# CUDA 13 drops compilation for V100 (sm70). Pin the toolkit independently
# of the driver, and restart Julia before validation. Keep this preference receipt.
export PB_CUDA_RUNTIME="${PB_CUDA_RUNTIME:-12.8}"
"$JULIA" --project=source/test/cuda -e 'using CUDA; CUDA.set_runtime_version!(VersionNumber(ENV["PB_CUDA_RUNTIME"]))'
"$JULIA" --project=source/test/cuda source/test/cuda/runtests.jl > validation.log 2>&1
"$JULIA" --project=source/test/cuda source/benchmark/gpu_solver/check.jl > solver-check.log 2>&1
PB_MODE=small PB_OUTPUT=small "$JULIA" --project=source/test/cuda source/benchmark/gpu_solver/run.jl > small.log 2>&1
PB_MODE=full PB_OUTPUT=full "$JULIA" --project=source/test/cuda source/benchmark/gpu_solver/run.jl > full.log 2>&1
python3 source/benchmark/gpu_solver/summarize.py small
python3 source/benchmark/gpu_solver/summarize.py full
# Profile the small and large Float64 cases separately; confirm their observed
# CPU/GPU ordering in full/summary.md before interpreting them as win/loss cases.
for shape in '400 40' '10000 1000'; do
    read -r PB_N PB_P <<< "$shape"
    export PB_N PB_P
    nvidia-smi --query-gpu=timestamp,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv -l 1 > "telemetry-${PB_N}.csv" &
    monitor=$!
    trap 'kill "$monitor" 2>/dev/null || true' EXIT
    "$JULIA" --project=source/test/cuda source/benchmark/gpu_solver/profile.jl > "profile-${PB_N}.log" 2>&1
    kill "$monitor" 2>/dev/null || true
    wait "$monitor" 2>/dev/null || true
    trap - EXIT
done
