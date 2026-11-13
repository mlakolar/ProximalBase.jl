#!/bin/bash
# Download a screening run (results only) and merge nothing: fetch_run.sh <ssh-session> <remote run dir> <local parent dir>
set -euo pipefail
SSHPY=~/.claude/skills/slurm-workflow/scripts/ssh_session.py
mkdir -p "$3"; python3 "$SSHPY" run "$1" "cd $(dirname $2) && tar -czf - --exclude='*.jl' --exclude='*.sbatch' $(basename $2)" | tar -xzf - -C "$3"
echo "$3/$(basename $2)"; ls -d "$3/$(basename $2)"/o*_s* 2>/dev/null | grep -v '\.log$' | tr '\n' ' '; echo
