#!/bin/bash
# Final comparison campaign for a pair of trees (orchestrator):
#   submit_final.sh <session> <rep tag> <Xlabel>=<worktree>@<rev> <Ylabel>=<worktree>@<rev> [nconc=10]
# Two exclusive xeon-2640v4 jobs, forward (X first) and reversed (Y first). Each runs, in fresh pinned processes:
#   l2_cold_paths_x.jl (5,046 cases, 40 shards), l2_norms_x.jl (368 cases, 8 shards), and the shared-kernel consumer
#   harnesses norml1_infinite_weight.jl, norml1_value_underflow.jl, linear_sparse_value.jl and run.jl (core probes),
#   each with its own independently compiled A/A control.
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd); SSHPY=~/.claude/skills/slurm-workflow/scripts/ssh_session.py
BENCH=/home/mkolar/projects/juliaPkg/pb901x/head/benchmark
SS=$1; REP=$2; XS=$3; YS=$4; NCONC=${5:-10}
tree() { local rest=${1#*=}; "$H/push_tree.sh" "$SS" "${rest%@*}" "${rest##*@}"; }
XL=${XS%%=*}; YL=${YS%%=*}; XP=$(tree "$XS"); YP=$(tree "$YS")
for dir in fwd rev; do
  if [ $dir = fwd ]; then F=$XP; S=$YP; else F=$YP; S=$XP; fi
  RUN=/scratch2/mkolar/pb-901x/runs/final-$YL-vs-$XL-$REP-$dir
  TF=$(mktemp)
  for i in $(seq 1 40); do echo "cold_s$i l2_cold_paths_x.jl $F $S $i/40 PB_ROUNDS=11"; done > "$TF"
  for i in $(seq 1 8); do echo "norms_s$i l2_norms_x.jl $F $S $i/8 PB_ROUNDS=11"; done >> "$TF"
  echo "l1inf norml1_infinite_weight.jl $F $S 1/1 PB_ROUNDS=15" >> "$TF"
  echo "l1under norml1_value_underflow.jl $F $S 1/1 PB_ROUNDS=15" >> "$TF"
  echo "linsparse linear_sparse_value.jl $F $S 1/1 PB_ROUNDS=15" >> "$TF"
  echo "core run.jl $F $S 1/1 PB_ROUNDS=11 PB_BASELINE=$F PB_CANDIDATE=$S" >> "$TF"
  python3 "$SSHPY" run "$SS" "mkdir -p $RUN" >&2
  for f in "$H/../l2_cold_paths_x.jl" "$H/../l2_norms_x.jl" "$H/campaign_job.sbatch" "$BENCH/norml1_infinite_weight.jl" \
           "$BENCH/norml1_value_underflow.jl" "$BENCH/linear_sparse_value.jl" "$BENCH/run.jl" "$BENCH/probes.jl" \
           "$BENCH/differential.jl" "$BENCH/Project.toml"; do
    python3 "$SSHPY" upload "$SS" "$f" "$RUN/$(basename "$f")" >&2; done
  # run.jl records SHA-256 digests of Project.toml, Manifest.toml, differential.jl next to itself
  python3 "$SSHPY" run "$SS" "cp /scratch2/mkolar/pb-901x/benchenv/Manifest.toml $RUN/Manifest.toml" >&2
  python3 "$SSHPY" upload "$SS" "$TF" "$RUN/tasks.txt" >&2; rm -f "$TF"
  printf 'X %s %s\nY %s %s\n' "$XL" "$XP" "$YL" "$YP" > /tmp/.pair.$$; python3 "$SSHPY" upload "$SS" /tmp/.pair.$$ "$RUN/pair.txt" >&2; rm -f /tmp/.pair.$$
  J=$(python3 "$SSHPY" run "$SS" "cd $RUN && sbatch --parsable --job-name=pb901x-final-$YL-$XL-$REP-$dir --comment=pb901x-final-$YL-$XL-$REP-$dir --constraint=xeon-2640v4 --time=08:00:00 --output=$RUN/job.out --export=ALL,RUN_DIR=$RUN,TASKS_FILE=$RUN/tasks.txt,NCONC=$NCONC campaign_job.sbatch")
  echo "$YL vs $XL $REP $dir job $J run $RUN"
done
