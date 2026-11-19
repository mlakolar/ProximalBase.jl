#!/usr/bin/env python3
"""Report removed docs/src lines not matched by exact newly added occurrences.

Usage: python3 docs/check_migration_lines.py BASE [--blocking]
This independent report uses no migration baseline or ledger. Reflow and link
repairs are intentionally reported; unchanged duplicate copies provide no credit.
"""
import argparse
from collections import Counter
from difflib import SequenceMatcher
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent


def removed_lines(base, root=ROOT):
    """Compare a Git tree-ish with tracked and nonignored untracked working files."""
    root = Path(root)

    def git(*args):
        result = subprocess.run(["git", "-C", str(root), "--literal-pathspecs", *args],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode:
            raise ValueError(result.stderr.decode("utf-8", errors="replace").strip())
        return result.stdout

    tree = git("rev-parse", "--verify", "--end-of-options", f"{base}^{{tree}}").decode().strip()
    old_files = {}
    for entry in git("ls-tree", "-rz", tree, "--", "docs/src").split(b"\0"):
        if entry:
            metadata, path = entry.split(b"\t", 1)
            mode, kind, oid = metadata.decode("ascii").split()
            old_files[path.decode("utf-8")] = (mode, kind, oid)
    # Git excludes clean sparse omissions; disabling renames retains both paths.
    changed_paths = git("diff", "--name-only", "-z", "--no-renames", tree, "--", "docs/src")
    untracked_paths = git("ls-files", "--others", "--exclude-standard", "-z", "--", "docs/src")
    paths = {path.decode("utf-8") for path in (changed_paths + untracked_paths).split(b"\0") if path}
    removed, added, changed = [], Counter(), []

    def lines(data, path):
        if b"\0" in data:
            raise ValueError(f"Changed binary file is unsupported: {path}")
        try:
            result = data.decode("utf-8").split("\n")
        except UnicodeDecodeError as error:
            raise ValueError(f"Changed non-UTF-8 file is unsupported: {path}") from error
        if result[-1] == "":
            result.pop()  # Only the synthetic empty line after a final LF.
        return result

    for path in sorted(paths):
        entry = old_files.get(path)
        if entry and entry[1] != "blob":
            raise ValueError(f"Unsupported Git object in docs/src: {path} ({entry[1]})")
        before = git("cat-file", "blob", entry[2]) if entry else b""
        current = root / path
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            mode = 0
        if stat.S_ISREG(mode):
            after = current.read_bytes()
        elif stat.S_ISLNK(mode):
            after = os.fsencode(os.readlink(current))
        elif not mode or (stat.S_ISDIR(mode) and entry):
            after = b""  # A former file may have become a directory.
        else:
            raise ValueError(f"Unsupported working-tree file: {path}")
        if before == after and (not entry or (entry[0] == "120000") == stat.S_ISLNK(mode)):
            continue
        if stat.S_ISLNK(mode) or (entry and entry[0] not in ("100644", "100755")):
            raise ValueError(f"Changed symlink or unsupported file: {path}")
        changed.append(path)
        old, new = lines(before, path), lines(after, path)
        for tag, start, end, new_start, new_end in SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
            if tag in ("delete", "replace"):
                removed.extend({"path": path, "line": index + 1, "text": old[index]}
                               for index in range(start, end) if old[index].strip())
            if tag in ("insert", "replace"):
                added.update(line for line in new[new_start:new_end] if line.strip())
    added_count, unmatched = sum(added.values()), []
    for item in removed:
        if added[item["text"]]:
            added[item["text"]] -= 1
        else:
            unmatched.append(item)
    return {"base": base, "base_tree": tree, "changed_paths": changed,
            "removed": len(removed), "added": added_count,
            "matched": len(removed) - len(unmatched), "unmatched": unmatched}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", help="Git tree-ish to compare with the current working tree")
    parser.add_argument("--blocking", action="store_true", help="exit 1 for unmatched lines (default: report-only)")
    args = parser.parse_args()
    report = removed_lines(args.base)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if args.blocking and report["unmatched"] else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError) as error:
        print(f"Removed-line report could not run: {error}", file=sys.stderr)
        sys.exit(2)
