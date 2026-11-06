#!/usr/bin/env python3
"""Regenerate benchmark/baselines/iterate-family-0.5.{md,csv} from the archived inputs, compare, and check.

usage: python3 benchmark/baselines/tools/reproduce.py [--archive PATH] [--keep DIR] [--baseline-dir DIR]

The script finds the repository from its own location, so it can be run from any directory when it is
given by its path. It

  1. checks the SHA-256 of data/iterate-family-0.5-evidence.tar.gz against the pinned value
     (skipped when --archive is given, which is meant for tests);
  2. extracts the 45 expected members, and nothing else, into a temporary directory;
  3. runs the structure and identity checks of check_evidence.py on the extraction;
  4. runs the committed make_baseline.py there, with `python -I -B`, relative input and output
     names and historical/ as --benchmark-dir, so that the timer-limited rule is the frozen one, and
     compares the generated md and csv byte for byte with the committed files;
  5. runs the original verify_csv3.py, which must report no mismatch and the pinned counts;
  6. runs the A/A and counts checks of check_evidence.py (skipped if step 3 found a structure error).

A failed step does not stop the later ones, so that every problem is printed. Nothing is written inside
the repository. --keep DIR keeps the extraction and the generated files in DIR, which must not exist
or must be empty. --baseline-dir DIR reads the committed md and csv from DIR (meant for tests).

Exit codes: 0 everything passed; 1 a generated file differs from the committed one or a check failed
(a check or output failure takes precedence over a crash of a later step); 2 any other failure (archive, usage,
generator, file system or unexpected error).
Python 3.11 or later (the generator needs tomllib); standard library only.
"""
import argparse, hashlib, os, re, shutil, subprocess, sys, tarfile, tempfile, traceback, zlib

sys.dont_write_bytecode = True
TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)          # under `python -I` the script's directory is not on sys.path
import check_evidence

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

BASELINES = os.path.dirname(TOOLS)
DEFAULT_ARCHIVE = os.path.join(BASELINES, "data", f"{TOP}.tar.gz")
GENERATOR = os.path.join(TOOLS, "make_baseline.py")
HISTORICAL = os.path.join(TOOLS, "historical")      # holds the frozen validate.timer_limited
VERIFY = os.path.join(TOOLS, "verify_csv3.py")
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
        raise ToolError(f"cannot read or extract the archive {archive}: {type(exc).__name__}: {exc}")


def generate(work):
    """Run the committed generator in work; return the generated file names relative to work."""
    out_dir = "generated"
    os.makedirs(os.path.join(work, out_dir), exist_ok=True)
    ev = TOP
    cmd = [sys.executable, "-I", "-B", GENERATOR,
           "--reps", "3", "--groups", f"{ev}/base-groups", "--layouts", f"{ev}/base-layouts",
           "--benchmark-dir", HISTORICAL,
           "--harness-commit", HARNESS_COMMIT, "--source-commit", SOURCE_COMMIT,
           "--crosscheck", f"{ev}/crosscheck-pinned-start.tsv",
           "--crosscheck-end", f"{ev}/crosscheck-pinned-end.tsv",
           "--restore-probe", f"{ev}/restore-probe.txt",
           "--aa-compare", f"{ev}/cmp-groups/comparison.csv", f"{ev}/cmp-layouts/comparison.csv",
           "--out-md", f"{out_dir}/{OUTPUTS[0]}", "--out-csv", f"{out_dir}/{OUTPUTS[1]}"]
    proc = subprocess.run(cmd, cwd=work, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        last = (proc.stderr.strip().splitlines() or [""])[-1]      # its traceback ends with the cause
        raise ToolError(f"make_baseline.py exited with {proc.returncode}: {last}")
    return [f"{out_dir}/{n}" for n in OUTPUTS]


def verify_csv3(work, baseline):
    """Run the original verify_csv3.py on the extraction; return the problems (empty if it passes)."""
    proc = subprocess.run([sys.executable, "-I", "-B", VERIFY, os.path.join(work, TOP), os.path.join(baseline, OUTPUTS[1])],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    lines, problems = proc.stdout.splitlines() + ["", "", ""], []
    if proc.returncode != 0:      # it crashed on data the checks of the evidence should have named
        return [f"verify_csv3.py exited with {proc.returncode}: {(proc.stderr.strip().splitlines() or [''])[-1]}"]
    c = check_evidence.COUNTS
    if lines[0] != f"rows {c['cases']} | mismatches: none":
        problems.append(f"verify_csv3.py: first line is {lines[0]!r}, expected 'rows {c['cases']} | mismatches: none'")
    m = re.fullmatch(r"timer-limited \(either side\) (\d+); not timer-limited (\d+); can gate (\d+) \(csv (\d+)\); "
                     r"A/A conclusive among not timer-limited (\d+) \(.*\)", lines[2])
    want = (c["timer_limited"], c["not_timer_limited"], c["gate5"], c["gate5"], c["aa_conclusive"])
    if not m or tuple(map(int, m.groups())) != want:
        problems.append(f"verify_csv3.py: counts line is {lines[2]!r}, expected timer-limited, not timer-limited, "
                        f"can gate (csv), conclusive {want}")
    return problems


def report(step, problems):
    for p in problems:
        print(f"{step}: FAILED: {p}")
    return bool(problems)


def run(args):
    archive = os.path.abspath(args.archive) if args.archive else DEFAULT_ARCHIVE
    baseline = os.path.abspath(args.baseline_dir) if args.baseline_dir else BASELINES
    if not os.path.isfile(archive):
        raise ToolError(f"no such archive: {archive}")
    print("python", " ".join(sys.version.split()))
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
        failed, crash = False, None
        errs = check_evidence.Errors()
        ev = check_evidence.check_structure(os.path.join(work, TOP), baseline, errs)
        if ev is not None:
            check_evidence.check_identity(ev, errs)
        crashed = errs.crashed
        failed |= report("structure and identity", errs.list())
        if not errs.list():
            print("structure and identity: passed")
        try:
            generated = generate(work)
            for gen, name in zip(generated, OUTPUTS):
                committed = os.path.join(baseline, name)
                if not os.path.isfile(committed):
                    raise ToolError(f"missing committed file {committed}")
                a, b = sha256_file(os.path.join(work, gen)), sha256_file(committed)
                if a == b:
                    print(f"{name}: identical, SHA-256 {a}")
                else:
                    print(f"{name}: DIFFERENT, generated {a}, committed {b}")
                    failed = True
        except Exception as exc:         # a crash must not hide the checks that follow, nor an evidence failure above
            crash = crash or exc
            print(f"generator: FAILED: {type(exc).__name__}: {exc}")
        try:
            problems = verify_csv3(work, baseline) if os.path.isfile(os.path.join(baseline, OUTPUTS[1])) else ["no committed csv"]
            failed |= report("verify_csv3", problems)
            if not problems:
                print("verify_csv3: no mismatch; rows, timer-limited, gate and conclusive counts as pinned")
        except Exception as exc:
            crash = crash or exc
            print(f"verify_csv3: FAILED: {type(exc).__name__}: {exc}")
        if ev is None or crashed:
            print("A/A and counts: skipped, an earlier evidence check failed to complete" if crashed else
                  "A/A and counts: skipped, the structure check failed")
        else:
            errs, info = check_evidence.Errors(), []
            check_evidence.check_aa_counts(ev, errs, info)
            print("\n".join(info))
            failed |= report("A/A and counts", errs.list())
            if not errs.list():
                print("A/A and counts: passed")
        if failed and crash is not None:
            print(f"{type(crash).__name__} in a later step is reported above; the evidence or output failure gives the exit code")
        if crash is not None and not failed:
            raise crash
        print("result:", "a generated file differs or a check failed" if failed else "both files identical, all checks passed")
        return 1 if failed else 0
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
    ap.add_argument("--baseline-dir", metavar="DIR", help="read the committed md and csv from DIR (for tests)")
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
