#!/usr/bin/env python3
"""Validate the inputs of the Full validation workflow and emit its plan.

The inputs arrive in the environment (`PB_MODE`, `PB_PLATFORMS`, `PB_JULIA`, `PB_FILES`,
`PB_EXPECTED_SHA`, plus `GITHUB_SHA`); an empty value stands for the default.  When
`GITHUB_OUTPUT` names a file, the script appends `matrix`, `mode` and `files` to it; it writes
`plan.json` for the verdict job.  `full` and `release` need all platforms and both Julia versions;
`focused` accepts a platform or Julia restriction and needs a list of test names separated by
spaces or tabs, which the workflow passes to the runner with `--exact`.

It imports only the standard library and `full_validation_verdict`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import full_validation_verdict as verdict

PLATFORMS = {
    "linux": "ubuntu-latest",
    "windows": "windows-latest",
    "macos": "macos-latest",
}
PLATFORM_CHOICES = ("all", *PLATFORMS)
JULIA_CHOICES = ("both", *verdict.JULIAS)
DEFAULTS = {"mode": "full", "platforms": "all", "julia": "both", "files": "", "expected_sha": ""}
FILE_NAME_RE = re.compile(r"[a-z0-9_]+")
SEPARATOR_RE = re.compile(r"[ \t]+")
SHA_RE = re.compile(r"[0-9a-f]{40}")
SHOWN_LENGTH = 60


class PlanInputError(Exception):
    """One or more inputs are invalid; `problems` lists each."""

    def __init__(self, problems):
        super().__init__("; ".join(problems))
        self.problems = list(problems)


def split_files(text):
    """Split `files` on spaces and tabs; return (names, problems).

    Other whitespace, such as a newline, is a problem: the names reach the test
    runner as separate arguments, and a list with another separator is more likely
    a paste error than a request.
    """
    other = sorted({ch for ch in text if ch.isspace() and ch not in " \t"})
    if other:
        return [], [f"files may be separated by spaces and tabs only; found {shown(''.join(other))}"]
    return [token for token in SEPARATOR_RE.split(text) if token], []


def shown(text):
    """A bounded repr, so that untrusted text stays on one line of bounded length."""
    text = str(text)
    return repr(text if len(text) <= SHOWN_LENGTH else text[:SHOWN_LENGTH] + "...")


def check_files(tokens, selection):
    """Return the problems of the requested file names.

    A name is exact: the workflow selects it with the runner's `--exact` option, so a
    name that is a prefix of a longer name selects only itself.
    """
    problems = []
    if len(tokens) > verdict.MAX_FOCUSED_FILES:
        problems.append(f"files lists {len(tokens)} names, at most {verdict.MAX_FOCUSED_FILES} are allowed")
    known = set(selection)
    seen = set()
    for token in tokens:
        if not FILE_NAME_RE.fullmatch(token):
            problems.append(
                f"file {shown(token)} is not a plain test name: use lowercase letters, digits and "
                "underscores only (no paths or wildcards)"
            )
        elif token not in known:
            problems.append(f"file {shown(token)} is not a test name in the full selection of test/tiers.toml")
        if token in seen:
            problems.append(f"file {shown(token)} is listed more than once")
        seen.add(token)
    return problems


def make_plan(inputs, github_sha, selection):
    """Return the plan dictionary for `inputs`, or raise PlanInputError with the problems found."""
    values = {key: (inputs.get(key) or "") for key in DEFAULTS}
    mode = values["mode"].strip() or DEFAULTS["mode"]
    platforms = values["platforms"].strip() or DEFAULTS["platforms"]
    julia = values["julia"].strip() or DEFAULTS["julia"]
    expected_sha = values["expected_sha"]
    tokens, problems = split_files(values["files"])

    if mode not in verdict.MODES:
        problems.append(f"mode {shown(mode)} is not one of {', '.join(verdict.MODES)}")
    if platforms not in PLATFORM_CHOICES:
        problems.append(f"platforms {shown(platforms)} is not one of {', '.join(PLATFORM_CHOICES)}")
    if julia not in JULIA_CHOICES:
        problems.append(f"julia {shown(julia)} is not one of {', '.join(JULIA_CHOICES)}")

    if mode == "focused":
        if not tokens:
            problems.append("focused mode needs files: a space-separated list of test names from test/tiers.toml")
    elif values["files"]:
        problems.append(f"files must be empty in {shown(mode)} mode; it applies to focused mode only")
    if tokens and mode == "focused":
        problems.extend(check_files(tokens, selection))

    if expected_sha:
        if not SHA_RE.fullmatch(expected_sha):
            problems.append(f"expected_sha {shown(expected_sha)} is not 40 lowercase hexadecimal characters")
        elif expected_sha != github_sha:
            problems.append(
                f"expected_sha {expected_sha} does not match the commit of this run, {github_sha}"
            )
    elif mode == "release":
        problems.append("release mode needs expected_sha, the commit to validate")

    if mode in ("full", "release"):
        if platforms != "all":
            problems.append(
                f"{mode} mode needs platforms=all, not {shown(platforms)}; use focused mode for a restricted run"
            )
        if julia != "both":
            problems.append(
                f"{mode} mode needs julia=both, not {shown(julia)}; use focused mode for a restricted run"
            )

    if problems:
        raise PlanInputError(problems)

    oses = [os_name for os_name in verdict.OSES if platforms == "all" or PLATFORMS[platforms] == os_name]
    julias = list(verdict.JULIAS) if julia == "both" else [julia]
    include = []
    for os_name in oses:
        for julia_version in julias:
            if mode == "focused":
                cells = [(os_name, julia_version, 1)]
            else:
                cells = [(os_name, julia_version, shard) for shard in verdict.SHARDS]
            for cell in cells:
                include.append(
                    {
                        "os": cell[0],
                        "julia": cell[1],
                        "shard": cell[2],
                        "shards": 1 if mode == "focused" else verdict.SHARD_COUNT,
                        "label": (
                            verdict.focused_job_name(cell[0], cell[1])
                            if mode == "focused"
                            else verdict.job_name(*cell)
                        ),
                    }
                )
    if mode != "focused":
        cells = sorted((c["os"], c["julia"], c["shard"]) for c in include)
        if cells != sorted(verdict.expected_cells()):
            raise PlanInputError(["the full matrix differs from the verdict script's expected cells"])
    return {
        "mode": mode,
        "platforms": platforms,
        "julia": julia,
        "files": tokens,
        "expected_sha": expected_sha,
        "commit": github_sha,
        "matrix": {"include": include},
    }


def outputs_text(plan):
    """The `name=value` lines for GITHUB_OUTPUT; the values hold no newline."""
    return (
        f"matrix={json.dumps(plan['matrix'], separators=(',', ':'))}\n"
        f"mode={plan['mode']}\n"
        f"files={' '.join(plan['files'])}\n"
    )


def main(argv=None, environ=None):
    environ = os.environ if environ is None else environ
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tiers", default="test/tiers.toml", help="path of test/tiers.toml")
    parser.add_argument("--plan-out", default="plan.json", help="file to write the plan to")
    parser.add_argument("--github-output", default=environ.get("GITHUB_OUTPUT"), help="file to append the job outputs to")
    args = parser.parse_args(argv)

    inputs = {
        "mode": environ.get("PB_MODE"),
        "platforms": environ.get("PB_PLATFORMS"),
        "julia": environ.get("PB_JULIA"),
        "files": environ.get("PB_FILES"),
        "expected_sha": environ.get("PB_EXPECTED_SHA"),
    }
    try:
        selection = verdict.load_selection(args.tiers)
        plan = make_plan(inputs, environ.get("GITHUB_SHA", ""), selection)
    except PlanInputError as error:
        for problem in error.problems:
            print(f"full-validation plan: {problem}", file=sys.stderr)
        return 1
    except (verdict.TiersError, OSError, ValueError) as error:
        print(f"full-validation plan: cannot read {args.tiers}: {error}", file=sys.stderr)
        return 1
    Path(args.plan_out).write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    if args.github_output:
        with open(args.github_output, "a", encoding="utf-8") as handle:
            handle.write(outputs_text(plan))
    print(
        f"full-validation plan: mode={plan['mode']} platforms={plan['platforms']} julia={plan['julia']} "
        f"files={' '.join(plan['files']) or 'none'} jobs={len(plan['matrix']['include'])}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
