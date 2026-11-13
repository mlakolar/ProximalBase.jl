#!/bin/bash
# Comprehensive campaign for one candidate (orchestrator):
#   submit_campaign.sh <session> <label> <worktree> <rev> <rep tag> [cold shards=40] [norm shards=8] [nconc=10]
# Two exclusive xeon-2640v4 jobs: forward (X = base, Y = candidate) and reversed (X = candidate, Y = base), each
# running every shard of l2_cold_paths_x.jl (5,046 cases) and l2_norms_x.jl (368 cases) in fresh pinned processes.
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd); SSHPY=~/.claude/skills/slurm-workflow/scripts/ssh_session.py
SS=$1; L=$2; WT=$3; REV=$4; REP=$5; NC=${6:-40}; NN=${7:-8}; NCONC=${8:-10}
B=$("$H/push_tree.sh" "$SS" /home/mkolar/projects/juliaPkg/pb901x/base HEAD)
C=$("$H/push_tree.sh" "$SS" "$WT" "$REV")
SHA=$(basename "$C")
for dir in fwd rev; do
  RUN=/scratch2/mkolar/pb-901x/runs/campaign-$L-${SHA:0:8}-$REP-$dir
  if [ $dir = fwd ]; then F=$B; S=$C; else F=$C; S=$B; fi
  TF=$(mktemp)
  for i in $(seq 1 $NC); do echo "cold_s$i l2_cold_paths_x.jl $F $S $i/$NC PB_ROUNDS=11"; done > "$TF"
  for i in $(seq 1 $NN); do echo "norms_s$i l2_norms_x.jl $F $S $i/$NN PB_ROUNDS=11"; done >> "$TF"
  python3 "$SSHPY" run "$SS" "mkdir -p $RUN" >&2
  for f in "$H/../l2_cold_paths_x.jl" "$H/../l2_norms_x.jl" "$H/campaign_job.sbatch"; do python3 "$SSHPY" upload "$SS" "$f" "$RUN/$(basename "$f")" >&2; done
  python3 "$SSHPY" upload "$SS" "$TF" "$RUN/tasks.txt" >&2; rm -f "$TF"
  J=$(python3 "$SSHPY" run "$SS" "cd $RUN && sbatch --parsable --job-name=pb901x-camp-$L-$REP-$dir --comment=pb901x-camp-$L-$REP-$dir --constraint=xeon-2640v4 --time=08:00:00 --output=$RUN/job.out --export=ALL,RUN_DIR=$RUN,TASKS_FILE=$RUN/tasks.txt,NCONC=$NCONC campaign_job.sbatch")
  echo "$L $REP $dir job $J run $RUN sha $SHA"
done
