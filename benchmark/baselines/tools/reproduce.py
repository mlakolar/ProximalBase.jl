#!/usr/bin/env python3
"""Regenerate benchmark/baselines/iterate-family-0.5.{md,csv} from the archived inputs and compare.

usage: python3 benchmark/baselines/tools/reproduce.py [--archive PATH] [--keep DIR]

The script finds the repository from its own location, so it can be run from any directory when it is
given by its path. It

  1. checks the SHA-256 of data/iterate-family-0.5-evidence.tar.gz against the pinned value
     (skipped when --archive is given, which is meant for tests);
  2. extracts the 45 expected members, and nothing else, into a temporary directory;
  3. runs the committed make_baseline.py there, with `python -I -B`, relative input and output
     names and the repository's benchmark/ directory for `validate.timer_limited`;
  4. compares the generated md and csv byte for byte with the committed files.

Nothing is written inside the repository. --keep DIR keeps the extraction and the generated files
in DIR, which must not exist or must be empty.

Exit codes: 0 both files are identical; 1 a generated file differs from the committed one; 2 any other
failure (archive, usage, generator, file system or unexpected error).
Python 3.11 or later (the generator needs tomllib); standard library only.
"""
import argparse, hashlib, os, shutil, subprocess, sys, tarfile, tempfile, traceback, zlib

sys.dont_write_bytecode = True

ARCHIVE_SHA256 = "ae0235adad9f932ceb95ea3ad1709621d73fa0a506dd231b9c442fd005b1bf87"
TOP = "iterate-family-0.5-evidence"
HARNESS_COMMIT = "5d43d165722bfb5a401abf735ba2a673d9ee0946"
SOURCE_COMMIT = "def81ec9d021c39ed1727cfeb9598357291dd60d"

RUNS = [f"{s}-{k}/julia-1.13.3-rep0{r}-{f}"
        for s in ("base", "aa") for k in ("groups", "layouts") for r in (1, 2, 3)
        for f in ("metrics.tsv", "metadata.toml", "errors.tsv")]
EXPECTED = sorted(RUNS + [f"cmp-{k}/{f}" for k in ("groups", "layouts")
                          for f in ("comparison.csv", "coverage.json", "REPORT.md")]
                  + ["crosscheck-pinned-start.tsv", "crosscheck-pinned-end.tsv", "restore-probe.txt"])

BASELINES = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(BASELINES))
DEFAULT_ARCHIVE = os.path.join(BASELINES, "data", f"{TOP}.tar.gz")
GENERATOR = os.path.join(BASELINES, "tools", "make_baseline.py")
OUTPUTS = ("iterate-family-0.5.md", "iterate-family-0.5.csv")


class ToolError(Exception):
    """An archive, usage or generator problem (exit code 2)."""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract(archive, dest):
    """Write the members of the archive below dest, after checking that they are exactly EXPECTED."""
    try:
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            names = [m.name for m in members]
            if len(set(names)) != len(names):
                raise ToolError("the archive has duplicate member names")
            expected = [f"{TOP}/{n}" for n in EXPECTED]
            if sorted(names) != expected:
                missing = sorted(set(expected) - set(names))
                extra = sorted(set(names) - set(expected))
                raise ToolError(f"the archive is not the expected set of {len(expected)} files; "
                                f"missing: {missing[:3]}; unexpected: {extra[:3]}")
            for m in members:
                if not m.isfile():
                    raise ToolError(f"member is not a regular file: {m.name}")
            for m in sorted(members, key=lambda m: m.name):
                target = os.path.join(dest, *m.name.split("/"))
                os.makedirs(os.path.dirname(target), exist_ok=True)
                data = tar.extractfile(m).read()
                if len(data) != m.size:
                    raise ToolError(f"truncated member: {m.name}")
                with open(target, "xb") as fh:
                    fh.write(data)
    except (tarfile.TarError, EOFError, zlib.error, OSError) as exc:
        if isinstance(exc, ToolError):
            raise
        raise ToolError(f"cannot read the archive {archive}: {type(exc).__name__}: {exc}")


def generate(work):
    """Run the committed generator in work; return the generated file names relative to work."""
    out_dir = "generated"
    os.makedirs(os.path.join(work, out_dir), exist_ok=True)
    ev = TOP
    cmd = [sys.executable, "-I", "-B", GENERATOR,
           "--reps", "3", "--groups", f"{ev}/base-groups", "--layouts", f"{ev}/base-layouts",
           "--benchmark-dir", os.path.join(REPO, "benchmark"),
           "--harness-commit", HARNESS_COMMIT, "--source-commit", SOURCE_COMMIT,
           "--crosscheck", f"{ev}/crosscheck-pinned-start.tsv",
           "--crosscheck-end", f"{ev}/crosscheck-pinned-end.tsv",
           "--restore-probe", f"{ev}/restore-probe.txt",
           "--aa-compare", f"{ev}/cmp-groups/comparison.csv", f"{ev}/cmp-layouts/comparison.csv",
           "--out-md", f"{out_dir}/{OUTPUTS[0]}", "--out-csv", f"{out_dir}/{OUTPUTS[1]}"]
    proc = subprocess.run(cmd, cwd=work, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise ToolError(f"make_baseline.py exited with {proc.returncode}\n{proc.stderr.strip()}")
    return [f"{out_dir}/{n}" for n in OUTPUTS]


def run(args):
    archive = os.path.abspath(args.archive) if args.archive else DEFAULT_ARCHIVE
    if not os.path.isfile(archive):
        raise ToolError(f"no such archive: {archive}")
    if args.archive is None:
        digest = sha256_file(archive)
        if digest != ARCHIVE_SHA256:
            raise ToolError(f"the archive has SHA-256 {digest}, expected {ARCHIVE_SHA256}")
        print(f"archive SHA-256 {digest} (pinned value)")
    if args.keep:
        keep = os.path.abspath(args.keep)
        if os.path.exists(keep) and (not os.path.isdir(keep) or os.listdir(keep)):
            raise ToolError(f"--keep needs a missing or empty directory: {args.keep}")
        os.makedirs(keep, exist_ok=True)
        work, cleanup = keep, False
    else:
        work, cleanup = tempfile.mkdtemp(prefix="iterate-family-0.5-"), True
    try:
        extract(archive, work)
        print(f"extracted {len(EXPECTED)} members")
        generated = generate(work)
        status = 0
        for gen, name in zip(generated, OUTPUTS):
            committed = os.path.join(BASELINES, name)
            if not os.path.isfile(committed):
                raise ToolError(f"missing committed file {name}")
            a, b = sha256_file(os.path.join(work, gen)), sha256_file(committed)
            if a == b:
                print(f"{name}: identical, SHA-256 {a}")
            else:
                print(f"{name}: DIFFERENT, generated {a}, committed {b}")
                status = 1
        print("result:", "both files identical" if status == 0 else "a generated file differs from the committed one")
        return status
    finally:
        if cleanup:
            shutil.rmtree(work, ignore_errors=True)


def main(argv):
    if sys.version_info < (3, 11):
        print("reproduce.py: Python 3.11 or later is required", file=sys.stderr)
        return 2
    ap = argparse.ArgumentParser(description="Regenerate the iterate-family 0.5 baseline from its archived inputs.")
    ap.add_argument("--archive", help="use this archive instead of the committed one (no hash check)")
    ap.add_argument("--keep", metavar="DIR", help="keep the extraction and the generated files in DIR")
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:
        return 2 if exc.code else 0
    try:
        return run(args)
    except ToolError as exc:
        print(f"reproduce.py: {exc}", file=sys.stderr)
        return 2
    except (OSError, EOFError, zlib.error, tarfile.TarError) as exc:
        print(f"reproduce.py: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"reproduce.py: unexpected error: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
