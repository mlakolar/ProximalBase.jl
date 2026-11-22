#!/usr/bin/env python3
"""Classify the files changed by a CI event as documentation-only or not.

The classifier compares the checked-out HEAD with the base the event is
validated against: the first parent of the pull-request merge commit, or the
`before` commit of a master push. It writes `docs_only=true|false` to
`GITHUB_OUTPUT`. A change counts as documentation-only only when the change set
is non-empty and every changed path, including both sides of a rename and every
deleted path, is on the allowlist below. Any other situation, including an
unusable base, is reported as `docs_only=false`. An unexpected event, a failed
git command or any other error exits nonzero and writes no output.

No third-party Python dependencies.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import PurePosixPath

# 100644/100755 are regular files; 000000 is the new mode of a deletion.
# Symlinks (120000) and submodules (160000) are not documentation.
DOCS_MODES = frozenset({"100644", "100755", "000000"})
# Allowed anywhere under docs/ (rendered pages and their images).
DOCS_SUFFIXES = frozenset({".md", ".svg", ".png", ".mmd"})
# Allowed under docs/audit/ in addition: data files that the docs job checks read.
AUDIT_DATA_SUFFIXES = frozenset({".tsv", ".json", ".csv", ".txt"})
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
ZERO_SHA = "0" * 40


class ClassificationError(Exception):
    """The event or the repository state cannot be classified."""


def git(*args: str, check: bool = True) -> bytes:
    return subprocess.run(
        ["git", *args], check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ).stdout


def changed_paths(base: str, head: str) -> list[tuple[str, str]]:
    """Return (path, new_mode) for each path changed between two commits.

    Without rename detection, a rename is a deletion of the old path plus an
    addition of the new path, so both paths are returned.
    """
    out = git("diff", "--raw", "--no-renames", "-z", "--no-abbrev", base, head)
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
        result.append((os.fsdecode(raw_path), fields[1]))
    return result


def is_docs_path(path: str, new_mode: str) -> bool:
    if new_mode not in DOCS_MODES:
        return False
    parts = PurePosixPath(path).parts
    suffix = PurePosixPath(path).suffix
    if len(parts) == 1:
        return suffix == ".md"
    if parts[0] != "docs":
        return False
    if suffix in DOCS_SUFFIXES:
        return True
    return len(parts) >= 3 and parts[1] == "audit" and suffix in AUDIT_DATA_SUFFIXES


def classify(paths: list[tuple[str, str]]) -> bool:
    return bool(paths) and all(is_docs_path(p, m) for p, m in paths)


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
        if not SHA_PATTERN.match(before):
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
            paths = changed_paths(base, head)
            for path, mode in paths:
                print(f"  {'docs' if is_docs_path(path, mode) else 'code'}  {mode}  {path}")
            docs_only = classify(paths)
            if not paths:
                reason = "empty change set"
            elif docs_only:
                reason = f"all {len(paths)} changed path(s) are documentation"
            else:
                reason = "at least one changed path is not documentation"
    except (ClassificationError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", b"") or b""
        print(f"error: {error} {detail.decode(errors='replace').strip()}", file=sys.stderr)
        return 1
    value = "true" if docs_only else "false"
    print(f"docs_only={value} ({reason})")
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"docs_only={value}\n")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(f"Change classification: `docs_only={value}` ({reason})\n")
    return 0


if __name__ == "__main__":
    # Print unusual file names as escapes instead of failing on encoding.
    sys.stdout.reconfigure(errors="backslashreplace")
    sys.exit(main())
