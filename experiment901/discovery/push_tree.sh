#!/bin/bash
# Copy the committed src/ and Project.toml of a worktree's commit to Discovery, keyed by full SHA.
#   push_tree.sh <ssh-session> <worktree> [<commit-ish>]   -> prints the remote tree path
# Uncommitted edits are never copied (git archive of the commit). Idempotent: an existing tree is kept.
set -euo pipefail
SESSION=$1; WT=$2; REV=${3:-HEAD}
SSHPY=~/.claude/skills/slurm-workflow/scripts/ssh_session.py
SHA=$(git -C "$WT" rev-parse "$REV^{commit}")
DEST=/scratch2/mkolar/pb-901x/trees/$SHA
if python3 "$SSHPY" run "$SESSION" "test -f $DEST/SHA" 2>/dev/null; then echo "$DEST"; exit 0; fi
git -C "$WT" archive --format=tar "$SHA" src Project.toml | \
  python3 "$SSHPY" run "$SESSION" "set -e; mkdir -p $DEST.tmp && tar -x -C $DEST.tmp && echo $SHA > $DEST.tmp/SHA && rm -rf $DEST && mv $DEST.tmp $DEST" >&2
echo "$DEST"
