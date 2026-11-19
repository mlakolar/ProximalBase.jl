#!/usr/bin/env bash
set -euo pipefail
root=$1
version=$2
label=$3
study="$root/pr-long-norm-study"
case "$label" in
    PR) checkout="$root/live-981/PR" ;;
    pairwise) checkout="$root/pr-long-norm" ;;
    *) exit 2 ;;
esac
julia="$HOME/.julia/juliaup/julia-$version+0.x64.linux.gnu/bin/julia"
harness="$root/harness/experiment981"
export JULIA_DEPOT_PATH="$root/depot:$HOME/.julia:" JULIA_PKG_OFFLINE=true
export JULIA_LOAD_PATH="@:$root/test-env/$version:@stdlib"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 JULIA_NUM_THREADS=1
for mode in default O0; do
    opts=()
    if [[ "$mode" == O0 ]]; then opts=(-O0); fi
    prefix="$study/accuracy/$label-$version-$mode"
    run() { "$julia" --startup-file=no "${opts[@]}" --project="$checkout" "$@"; }
    if run --check-bounds=yes -e '
        using Test, LinearAlgebra
        BLAS.set_num_threads(1)
        include(joinpath(ARGS[1], "test", "testutils.jl"))
        @testset "NormLinf KKT and prox" begin
            include(joinpath(ARGS[1], "test", "kkt_linf.jl"))
            include(joinpath(ARGS[1], "test", "normlinf.jl"))
        end
    ' "$checkout" > "$prefix-package.log" 2>&1; then
        printf '0\n' > "$prefix-package.exit"
    else
        printf '%s\n' "$?" > "$prefix-package.exit"
    fi
    if run "$study/driver/normlinf-correctness-regressions.jl" > "$prefix-stress.tsv" 2>"$prefix-stress.log"; then
        printf '0\n' > "$prefix-stress.exit"
    else
        printf '%s\n' "$?" > "$prefix-stress.exit"
    fi
    "$julia" --startup-file=no "${opts[@]}" --project="$root/base" "$harness/accuracy.jl" \
        "$label" "$checkout" "$prefix-oracle.tsv" > "$prefix-oracle.log" 2>&1
    run "$study/driver/norm-oracle.jl" > "$prefix-norm-oracle.tsv" 2>"$prefix-norm-oracle.log"
    run "$study/driver/longsum.jl" > "$prefix-longsum.log" 2>&1
    printf '%s %s %s complete\n' "$label" "$version" "$mode"
done
