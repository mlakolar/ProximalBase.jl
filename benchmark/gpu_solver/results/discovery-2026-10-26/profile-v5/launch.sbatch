#!/bin/bash
#SBATCH --account=mkolar_1314
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=pb-gpu110-profile-v5
#SBATCH --output=slurm-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
export JULIA_DEPOT_PATH=/home1/mkolar/juliaPkg/runs/20261026-pb-gpu110/attempt-v4/depot:/home1/mkolar/juliaPkg/runs/20261026-pb-gpu110/depot:/home1/mkolar/juliaPkg/runs/20261025-pb-gpu103/depot:/home1/mkolar/juliaPkg/depot
JULIA=/home1/mkolar/juliaPkg/julia-1.10.12/bin/julia
PROJECT=/home1/mkolar/juliaPkg/runs/20261026-pb-gpu110/attempt-v4/source/test/cuda
nvidia-smi > hardware.txt
printf '%s\n' 'Profile source:' > source.txt
git -C source rev-parse HEAD >> source.txt
printf '%s\n' 'Package source:' >> source.txt
git -C ../attempt-v4/source rev-parse HEAD >> source.txt
for shape in '400 40' '10000 1000'; do
    read -r PB_N PB_P <<< "$shape"
    export PB_N PB_P
    nvidia-smi --query-gpu=timestamp,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv -l 1 > "telemetry-${PB_N}.csv" &
    monitor=$!
    trap 'kill "$monitor" 2>/dev/null || true' EXIT
    "$JULIA" --project="$PROJECT" source/benchmark/gpu_solver/profile.jl > "profile-${PB_N}.log" 2>&1
    # Fail rather than silently accept another profile with no printed trace.
    grep -q 'Host-side activity' "profile-${PB_N}.log"
    grep -q 'Device-side activity' "profile-${PB_N}.log"
    kill "$monitor" 2>/dev/null || true
    wait "$monitor" 2>/dev/null || true
    trap - EXIT
done
