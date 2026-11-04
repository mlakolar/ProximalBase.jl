#!/bin/sh
# Re-create the #556 audit inventory cache. Usage: sh fetch.sh <empty cache dir>
# Needs an authenticated `gh` and python3. Writes the raw API pages, then the
# GraphQL extras (reviews, review threads, closing and cross references), then
# one chronological bundle per closed item in <dir>/items/NNNN.md.
set -eu
REPO=mlakolar/ProximalBase.jl
D=$1; HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$D" && cd "$D"
date -u +%FT%TZ > fetched-at.txt
gh api --paginate --slurp "repos/$REPO/pulls?state=closed&per_page=100"  > closed-pr-pages.json
gh api --paginate --slurp "repos/$REPO/issues?state=closed&per_page=100" > closed-issue-pages.json   # includes PRs; filtered later
gh api --paginate --slurp "repos/$REPO/issues/comments?per_page=100"     > issue-comments-pages.json  # conversation comments, issues and PRs
gh api --paginate --slurp "repos/$REPO/pulls/comments?per_page=100"      > review-comments-pages.json # inline review comments
gh api --paginate --slurp "repos/$REPO/issues?state=open&per_page=100"   > open-issue-pages.json
python3 -I "$HERE/gql_fetch.py" "$D"   # pr-graphql.json, issue-graphql.json (fails loudly on API errors; reports truncation)
python3 -I "$HERE/bundle.py" "$D"      # items/NNNN.md, items-meta.json
