#!/bin/bash
# Focused cold-path / l2_norms run (orchestrator): selected cases of l2_cold_paths_x.jl (and optionally l2_norms_x.jl)
# for several trees against base, NREP fresh processes per role order, one exclusive xeon-2640v4 node.
#   submit_focus.sh <session> <name> <cold filter regex> <norms filter regex or -> <nrep> <label>=<worktree>@<rev> [...]
# Regexes must not contain spaces, '*' or '?' (use '.+' or '.{0,}'). Tasks per tree: cold fwd/rev x nrep (+ norms).
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd); SSHPY=~/.claude/skills/slurm-workflow/scripts/ssh_session.py
SS=$1; NAME=$2; CF=$3; NF=$4; NREP=$5; shift 5
B=$("$H/push_tree.sh" "$SS" /home/mkolar/projects/juliaPkg/pb901x/base HEAD)
RUN=/scratch2/mkolar/pb-901x/runs/focus-$NAME-$(date +%Y%m%dT%H%M%S)
TF=$(mktemp)
for spec in "$@"; do
  L=${spec%%=*}; rest=${spec#*=}; WT=${rest%@*}; REV=${rest##*@}
  C=$("$H/push_tree.sh" "$SS" "$WT" "$REV")
  for r in $(seq 1 $NREP); do
    echo "${L}_cold_fwd_r$r l2_cold_paths_x.jl $B $C 1/1 PB_ROUNDS=15 PB_FILTER=$CF" >> "$TF"
    echo "${L}_cold_rev_r$r l2_cold_paths_x.jl $C $B 1/1 PB_ROUNDS=15 PB_FILTER=$CF" >> "$TF"
    if [ "$NF" != "-" ]; then
      echo "${L}_norms_fwd_r$r l2_norms_x.jl $B $C 1/1 PB_ROUNDS=15 PB_FILTER=$NF" >> "$TF"
      echo "${L}_norms_rev_r$r l2_norms_x.jl $C $B 1/1 PB_ROUNDS=15 PB_FILTER=$NF" >> "$TF"
    fi
  done
done
NT=$(wc -l < "$TF"); NCONC=$(( NT < 10 ? NT : 10 ))
python3 "$SSHPY" run "$SS" "mkdir -p $RUN" >&2
for f in "$H/../l2_cold_paths_x.jl" "$H/../l2_norms_x.jl" "$H/campaign_job.sbatch"; do python3 "$SSHPY" upload "$SS" "$f" "$RUN/$(basename "$f")" >&2; done
python3 "$SSHPY" upload "$SS" "$TF" "$RUN/tasks.txt" >&2; rm -f "$TF"
J=$(python3 "$SSHPY" run "$SS" "cd $RUN && sbatch --parsable --job-name=pb901x-focus-$NAME --comment=pb901x-focus-$NAME --constraint=xeon-2640v4 --time=01:30:00 --output=$RUN/job.out --export=ALL,RUN_DIR=$RUN,TASKS_FILE=$RUN/tasks.txt,NCONC=$NCONC campaign_job.sbatch")
echo "job $J run $RUN tasks $NT"
