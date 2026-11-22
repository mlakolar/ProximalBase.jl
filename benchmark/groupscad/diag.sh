#!/bin/bash
# Run benchmark/groupscad/diag_inplace.jl from a package root (a clean `git archive`, no .git needed).
#
#   PB_SOURCE=<candidate tree> PB_LEGACY=<archived tree> PB_REFERENCE=<tree with GroupNormL2> PB_OUTPUT=<dir> \
#     [JULIA=julia] [PB_VARIANT=<value variant tree> PB_VARIANT_INPLACE=<in-place variant tree>] \
#     [PB_NO_INSTANTIATE=1] [PB_FILTER=<regex>] [PB_REPS=8] [PB_SAMPLES=200] [PB_SECONDS=0.05] [PB_REVERSE=1] \
#     bash benchmark/groupscad/diag.sh
#
# The trees are folders with a src/ (for example `git archive <sha> src | tar -x -C <tree>`): the candidate commit, the archived
# implementation eff21b6d7d33c622313122f46c6ee71693c101ee, and the base of the pull request a6b16d63d9c5102b1ac82d9995014145681d4343.
# When PB_VARIANT or PB_VARIANT_INPLACE is not set, the two variant trees are made from PB_SOURCE in PB_OUTPUT/variants
# (make_variants.py, which needs python3). Run it on an exclusive node, one Julia thread, once per Julia version and CPU.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
julia="${JULIA:-julia}"
: "${PB_SOURCE:?}" "${PB_LEGACY:?}" "${PB_REFERENCE:?}" "${PB_OUTPUT:?}"
mkdir -p "$PB_OUTPUT"
if [[ -z "${PB_VARIANT:-}" || -z "${PB_VARIANT_INPLACE:-}" ]]; then
  python3 "$root/benchmark/groupscad/make_variants.py" "$PB_SOURCE" "$PB_OUTPUT/variants"
  export PB_VARIANT="$PB_OUTPUT/variants/vstored" PB_VARIANT_INPLACE="$PB_OUTPUT/variants/vinplace"
fi
if [[ "${PB_NO_INSTANTIATE:-0}" != "1" ]]; then
  "$julia" --startup-file=no "$root/scripts/instantiate.jl" benchmark
fi
export PB_SOURCE PB_LEGACY PB_REFERENCE PB_OUTPUT
cd "$root"
"$julia" --startup-file=no --threads=1 --check-bounds=no --project=benchmark benchmark/groupscad/diag_inplace.jl 2>&1 | tee "$PB_OUTPUT/diag.log"
