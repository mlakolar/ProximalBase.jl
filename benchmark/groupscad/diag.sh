#!/bin/bash
# Run benchmark/groupscad/diag_inplace.jl from a package root (a clean `git archive`, no .git needed).
#
#   PB_SOURCE=<baseline tree> PB_LEGACY=<archived tree> PB_REFERENCE=<tree with GroupNormL2> PB_OUTPUT=<dir> \
#     [JULIA=julia] [PB_VARIANT=<tree for the V probes>] [PB_VARIANT_INPLACE=<tree for the W probes>] \
#     [PB_NO_INSTANTIATE=1] [PB_FILTER=<regex>] [PB_REPS=8] [PB_SAMPLES=200] [PB_SECONDS=0.05] [PB_REVERSE=1] \
#     bash benchmark/groupscad/diag.sh
#
# The trees are folders with a src/ (for example `mkdir -p <tree> && git archive <sha> src | tar -x -C <tree>`): the archived
# implementation eff21b6d7d33c622313122f46c6ee71693c101ee, the base of the pull request a6b16d63d9c5102b1ac82d9995014145681d4343,
# and the commits to compare. The two variant flags are independent. To compare the head with a baseline give PB_SOURCE = the
# baseline and the head as both PB_VARIANT and PB_VARIANT_INPLACE (see the header of diag_inplace.jl). A flag that is not set is
# filled by make_variants.py from PB_SOURCE into PB_OUTPUT/variants (python3): PB_VARIANT by the value variant and
# PB_VARIANT_INPLACE by the in-place variant, which exists only for a baseline whose source has the pattern that it edits (the
# tree of 9d2fefc71); where it is not made, the W probes and the compact sweeps are not run. Run it on an exclusive node, one
# Julia thread, once per Julia version and CPU.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
julia="${JULIA:-julia}"
: "${PB_SOURCE:?}" "${PB_LEGACY:?}" "${PB_REFERENCE:?}" "${PB_OUTPUT:?}"
mkdir -p "$PB_OUTPUT"
need=()
[[ -n "${PB_VARIANT:-}" ]] || need+=(vstored)
[[ -n "${PB_VARIANT_INPLACE:-}" ]] || need+=(vinplace)
if (( ${#need[@]} )); then
  for name in "${need[@]}"; do rm -rf "$PB_OUTPUT/variants/$name"; done
  python3 "$root/benchmark/groupscad/make_variants.py" "$PB_SOURCE" "$PB_OUTPUT/variants" "${need[@]}"
  [[ -n "${PB_VARIANT:-}" || ! -d "$PB_OUTPUT/variants/vstored" ]] || export PB_VARIANT="$PB_OUTPUT/variants/vstored"
  [[ -n "${PB_VARIANT_INPLACE:-}" || ! -d "$PB_OUTPUT/variants/vinplace" ]] || export PB_VARIANT_INPLACE="$PB_OUTPUT/variants/vinplace"
fi
if [[ "${PB_NO_INSTANTIATE:-0}" != "1" ]]; then
  "$julia" --startup-file=no "$root/scripts/instantiate.jl" benchmark
fi
export PB_SOURCE PB_LEGACY PB_REFERENCE PB_OUTPUT
cd "$root"
"$julia" --startup-file=no --threads=1 --check-bounds=no --project=benchmark benchmark/groupscad/diag_inplace.jl 2>&1 | tee "$PB_OUTPUT/diag.log"
