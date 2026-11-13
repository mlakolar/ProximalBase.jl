#!/bin/bash
# Submit a common screening job (orchestrator-owned).
#   submit_screen.sh -s <ssh-session> -n <run-name> -c <label>=<worktree>[@<rev>] [-c ...] \
#       [-f <filter-regex>] [-r rounds=11] [-k shards=3] [-p nconc=5] [-o fwd,rev] [-C xeon-2640v4] [-t 04:00:00]
# Trees: base and ref are always included (base first = X in the forward order). Each -c adds a tree.
# Orders: fwd = base, ref, cands...; rev = <first -c>, base, ref, other cands (the first candidate becomes X).
# Prints: job id and remote run dir /scratch2/mkolar/pb-901x/runs/<run-name>-<timestamp>.
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd)
SSHPY=~/.claude/skills/slurm-workflow/scripts/ssh_session.py
R=/home/mkolar/projects/juliaPkg/pb901x
SESSION=""; NAME=""; CANDS=(); FILTER="."; ROUNDS=11; SHARDS=3; NCONC=5; ORDERS_SPEC="fwd,rev"; CONSTRAINT=xeon-2640v4; TIME=01:00:00
SAMPLES=5; TARGET=200000
while getopts "s:n:c:f:r:k:p:o:C:t:S:T:" opt; do case $opt in
  s) SESSION=$OPTARG;; n) NAME=$OPTARG;; c) CANDS+=("$OPTARG");; f) FILTER=$OPTARG;; r) ROUNDS=$OPTARG;;
  k) SHARDS=$OPTARG;; p) NCONC=$OPTARG;; o) ORDERS_SPEC=$OPTARG;; C) CONSTRAINT=$OPTARG;; t) TIME=$OPTARG;;
  S) SAMPLES=$OPTARG;; T) TARGET=$OPTARG;; *) exit 2;; esac; done
[ -n "$SESSION" ] && [ -n "$NAME" ] || { echo "need -s and -n" >&2; exit 2; }
declare -a LABELS PATHS
add() { LABELS+=("$1"); PATHS+=("$("$H/push_tree.sh" "$SESSION" "$2" "$3")"); }
add base "$R/base" HEAD
add ref "$R/ref" HEAD
for c in "${CANDS[@]}"; do
  lab=${c%%=*}; rest=${c#*=}; wt=${rest%@*}; rev=HEAD; [[ $rest == *@* ]] && rev=${rest##*@}
  add "$lab" "$wt" "$rev"
done
tr() { echo -n "$1=$2"; }
FWD=""; for i in "${!LABELS[@]}"; do FWD+="${LABELS[$i]}=${PATHS[$i]};"; done
REV=""
if [ ${#LABELS[@]} -ge 3 ]; then
  REV="${LABELS[2]}=${PATHS[2]};${LABELS[0]}=${PATHS[0]};${LABELS[1]}=${PATHS[1]};"
  for i in $(seq 3 $((${#LABELS[@]} - 1))); do REV+="${LABELS[$i]}=${PATHS[$i]};"; done
else
  REV="${LABELS[1]}=${PATHS[1]};${LABELS[0]}=${PATHS[0]};"
fi
ORDERS=""
IFS=',' read -r -a OS <<< "$ORDERS_SPEC"
for o in "${OS[@]}"; do
  case $o in fwd) ORDERS+="${ORDERS:+|}$FWD";; rev) ORDERS+="${ORDERS:+|}$REV";;
             fwd*) ORDERS+="${ORDERS:+|}$FWD";; rev*) ORDERS+="${ORDERS:+|}$REV";; *) echo "order $o" >&2; exit 2;; esac
done
STAMP=$(date +%Y%m%dT%H%M%S)
RUN=/scratch2/mkolar/pb-901x/runs/$NAME-$STAMP
python3 "$SSHPY" run "$SESSION" "mkdir -p $RUN" >&2
python3 "$SSHPY" upload "$SESSION" "$H/../screen901.jl" "$RUN/screen901.jl" >&2
python3 "$SSHPY" upload "$SESSION" "$H/screen_job.sbatch" "$RUN/screen_job.sbatch" >&2
printf '%s\n' "$ORDERS" | sed 's/|/\n/g' > /tmp/.orders.$$ ; python3 "$SSHPY" upload "$SESSION" /tmp/.orders.$$ "$RUN/orders.txt" >&2; rm -f /tmp/.orders.$$
JOB=$(python3 "$SSHPY" run "$SESSION" "cd $RUN && sbatch --parsable --job-name=pb901x-$NAME --comment=pb901x-$NAME \
  --constraint=$CONSTRAINT --time=$TIME --output=$RUN/job.out \
  --export=ALL,RUN_DIR=$RUN,SHARDS=$SHARDS,NCONC=$NCONC,PB_FILTER='$FILTER',PB_ROUNDS=$ROUNDS,PB_SAMPLES=$SAMPLES,PB_TARGET_NS=$TARGET,ORDERS_FILE=$RUN/orders.txt \
  screen_job.sbatch")
echo "job $JOB run $RUN"
