#!/usr/bin/env python3
"""Classify the files changed by a CI event as documentation-only or not.

The classifier compares the checked-out HEAD with the base the event is
validated against: the first parent of the pull-request merge commit, or the
`before` commit of a push to master or release/0.6. After a successful classification, when
`GITHUB_OUTPUT` names a file, it appends `docs_only=true|false` and
`base=<sha>` (empty when no base is usable). If `before` is still
unavailable after a fetch attempt, the result is `docs_only=false`.

A change set is documentation-only when it is non-empty and every path that
`git diff --raw --no-renames --ignore-submodules=none` reports (a rename
appears as a deletion plus an addition) is on the path allowlist below and is
an added, modified or deleted regular file without the execute bit. Any other
result, including an unusable base, is `docs_only=false`. The allowlist is a
path rule; it does not say which files Documenter builds.

An unexpected event, a malformed `before` value or a failed `git diff` exits
nonzero without writing `GITHUB_OUTPUT` records; diagnostics may still be
printed.

No third-party Python dependencies.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import PurePosixPath
from typing import NamedTuple

# Allowed (status, old mode, new mode) triples from `git diff --raw`. Mode
# 000000 marks the absent side of an add (A) or delete (D). The execute bit,
# symlinks (120000), submodules (160000) and type changes are not documentation.
DOCS_TRANSITIONS = frozenset({
    ("M", "100644", "100644"),
    ("A", "000000", "100644"),
    ("D", "100644", "000000"),
})
# Suffixes allowed anywhere under docs/ (matched case-sensitively).
DOCS_SUFFIXES = frozenset({".md", ".svg", ".png", ".mmd"})
# Suffixes allowed in addition under this directory, which the docs job's checks read.
AUDIT_DATA_DIR = ("docs", "audit", "docs-tracks-1019")
AUDIT_DATA_SUFFIXES = frozenset({".tsv", ".json", ".csv", ".txt"})
SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
ZERO_SHA = "0" * 40


class ClassificationError(Exception):
    """The event or the repository state cannot be classified."""


class Change(NamedTuple):
    path: str
    old_mode: str
    new_mode: str
    status: str


def git(*args: str, check: bool = True) -> bytes:
    return subprocess.run(
        ["git", *args], check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ).stdout


def changed_paths(base: str, head: str) -> list[Change]:
    """Return a Change for each path that differs between two commits.

    Without rename detection, a rename is a deletion of the old path plus an
    addition of the new path, so both paths are returned.
    """
    out = git("diff", "--raw", "--no-renames", "--ignore-submodules=none", "-z", "--no-abbrev", base, head)
    tokens = out.split(b"\0")
    if tokens and tokens[-1] == b"":
        tokens.pop()
    if len(tokens) % 2:
        raise ClassificationError("unexpected git diff --raw output")
    result = []
    for meta, raw_path in zip(tokens[0::2], tokens[1::2]):
        fields = meta.decode("ascii").split()
        if not fields[0].startswith(":") or len(fields) != 5:
            raise ClassificationError(f"unexpected git diff --raw entry: {meta!r}")
        result.append(Change(os.fsdecode(raw_path), fields[0][1:], fields[1], fields[4]))
    return result


def is_docs_path(change: Change) -> bool:
    if (change.status, change.old_mode, change.new_mode) not in DOCS_TRANSITIONS:
        return False
    pure = PurePosixPath(change.path)
    parts, suffix = pure.parts, pure.suffix
    if len(parts) == 1:
        return suffix == ".md"
    if parts[0] != "docs":
        return False
    if suffix in DOCS_SUFFIXES:
        return True
    return parts[: len(AUDIT_DATA_DIR)] == AUDIT_DATA_DIR and len(parts) > len(AUDIT_DATA_DIR) \
        and suffix in AUDIT_DATA_SUFFIXES


def classify(changes: list[Change]) -> bool:
    return bool(changes) and all(is_docs_path(change) for change in changes)


def _commit_available(sha: str) -> bool:
    return subprocess.run(
        ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def base_for_event(event_name: str, env) -> str | None:
    """Return the commit to compare HEAD with, or None for "not docs-only"."""
    if event_name == "pull_request":
        parents = git("rev-list", "--parents", "-n", "1", "HEAD").decode().split()[1:]
        if len(parents) != 2:
            raise ClassificationError(
                f"expected the pull-request merge commit as HEAD, found {len(parents)} parent(s)"
            )
        return parents[0]
    if event_name == "push":
        before = env.get("PUSH_BEFORE_SHA", "")
        if before == "":
            print("push has no previous commit (missing before)")
            return None
        if not SHA_PATTERN.fullmatch(before):
            raise ClassificationError(f"PUSH_BEFORE_SHA is not a 40-digit lowercase hex SHA: {before!r}")
        if before == ZERO_SHA:
            print("push has no previous commit (all-zero before)")
            return None
        if not _commit_available(before):
            # A failed fetch is tolerated: the base is then reported unusable.
            subprocess.run(
                ["git", "fetch", "--no-tags", "--depth=1", "origin", before],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        if not _commit_available(before):
            print(f"previous commit {before} is not available locally")
            return None
        return before
    if event_name == "workflow_dispatch":
        print("workflow_dispatch is not classified")
        return None
    raise ClassificationError(f"unexpected event: {event_name!r}")


def main() -> int:
    try:
        event = os.environ.get("GITHUB_EVENT_NAME", "")
        print(f"event: {event}")
        head = git("rev-parse", "HEAD").decode().strip()
        base = base_for_event(event, os.environ)
        print(f"base: {base}")
        print(f"head: {head}")
        if base is None:
            docs_only, reason = False, "no usable base commit"
        else:
            changes = changed_paths(base, head)
            for change in changes:
                verdict = "docs" if is_docs_path(change) else "code"
                # The name is escaped and follows fixed text, so the printed
                # line does not begin with the file name.
                print(f"  path {verdict} {change.status} {change.old_mode}>{change.new_mode} {change.path!r}")
            docs_only = classify(changes)
            if not changes:
                reason = "empty change set"
            elif docs_only:
                reason = f"all {len(changes)} changed path(s) are documentation"
            else:
                reason = "at least one changed path is not documentation"
    except (ClassificationError, subprocess.CalledProcessError) as error:
        detail = (getattr(error, "stderr", b"") or b"").decode(errors="replace").strip()
        print(f"error: {error} {detail!r}", file=sys.stderr)
        return 1
    value = "true" if docs_only else "false"
    print(f"docs_only={value} ({reason})")
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"docs_only={value}\nbase={base or ''}\n")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(f"Change classification: `docs_only={value}` ({reason})\n")
    return 0


if __name__ == "__main__":
    # Print unusual file names as escapes instead of failing on encoding.
    sys.stdout.reconfigure(errors="backslashreplace")
    sys.exit(main())
