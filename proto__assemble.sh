#!/bin/bash
# Rebuild the PB #911 run tree from the two pinned checkouts and this (flat) evidence bundle.
#   assemble.sh <ProximalSolvers checkout at 3408e1f> <ProximalBase checkout at 1f3749f3> <bundle dir>
# Creates <PS>/.codex-orchestrator/runs/run-pb911/{proto,pkgs,out}: the scripts, the 10 PS-copy variants
# (K, K2, K3, B1, B2, B3, B3u, H, H1, C), PB master (pkgs/pb-master), the disposable PB refactor (pkgs/pb-H)
# and the package pairs for compile.jl (pkgs/bnd). Then, from <PS>:
#   julia --startup-file=no .codex-orchestrator/runs/run-pb911/proto/numerics.jl OFFLINE=false   (first run: fetch deps)
#   julia --startup-file=no .codex-orchestrator/runs/run-pb911/proto/<script>.jl [NAME=value ...]
# Scripts needing threads: julia -t 4 ... probes.jl. Exclusive-node jobs ran harness.jl with
#   MODE=count,diag,alloc,types,kern,time CASES=lasso,B,Bwarm,C,D,big,rawB,mixB,B0,Blow,Bhigh SOLVERS=cd,cds,pg,apg ROUNDS=60
# and compile.jl REPS_PRE=3 REPS_TTFX=5, each pinned to one core (taskset -c 2; compile.jl on cores 2-3).
set -euo pipefail
PS=$(cd "$1" && pwd); PB=$(cd "$2" && pwd); B=$(cd "$3" && pwd)
[ "$(git -C "$PS" rev-parse HEAD)" = 3408e1fc24840596e4f67deb797699e7338aa213 ] || { echo "PS must be at 3408e1f"; exit 1; }
[ "$(git -C "$PB" rev-parse HEAD)" = 1f3749f3332d1551bd65805d25786f3f53731db2 ] || { echo "PB must be at 1f3749f3"; exit 1; }
RUN=$PS/.codex-orchestrator/runs/run-pb911
rm -rf "$RUN"; mkdir -p "$RUN"/{proto/variants,pkgs,out,memo}
for f in "$B"/proto__*; do cp "$f" "$RUN/proto/${f##*/proto__}"; done
for v in K K2 K3 B1 B2 B3 H H1 C; do
  mkdir -p "$RUN/proto/variants/$v"; git -C "$PS" archive HEAD src | tar -x -C "$RUN/proto/variants/$v"
done
for v in B1 B2 B3 H H1 C; do
  (cd "$RUN/proto/variants/$v" && patch -s -p1 < "$B/variant_${v}__ps_side.diff")
  cp "$B/variant_${v}__pb_oracle.jl" "$RUN/proto/variants/$v/src/pb_oracle.jl"
done
cp -r "$RUN/proto/variants/B3" "$RUN/proto/variants/B3u"
sed -i 's/^const UNCHECKED = false.*/const UNCHECKED = true/' "$RUN/proto/variants/B3u/src/pb_oracle.jl"
mkdir -p "$RUN/pkgs/pb-master"; git -C "$PB" archive HEAD Project.toml src test | tar -x -C "$RUN/pkgs/pb-master"
echo "$(git -C "$PB" rev-parse HEAD) $(git -C "$PB" rev-parse 'HEAD^{tree}')" > "$RUN/pkgs/pb-master/TREE"
cp -r "$RUN/pkgs/pb-master" "$RUN/pkgs/pb-H"
(cd "$RUN/pkgs/pb-H" && patch -s -p1 < "$B/pbH_refactor_vs_pb-master_1f3749f3.diff")
echo "pb-master 1f3749f3 + PB #911 H refactor (disposable)" > "$RUN/pkgs/pb-H/TREE"
python3 "$RUN/proto/make_boundary.py" "$RUN"
echo "assembled $RUN"
