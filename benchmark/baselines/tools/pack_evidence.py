#!/usr/bin/env python3
"""Pack the archived inputs of the iterate-family 0.5 baseline into a deterministic tar.gz.

usage: pack_evidence.py SRC_DIR OUT.tar.gz

SRC_DIR is a directory that holds exactly the 45 files of the evidence set (see EXPECTED below),
for example the directory `iterate-family-0.5-evidence` of an extracted archive. Any other file,
symlink or directory is refused. OUT.tar.gz must not exist.

The archive holds the files under the top-level directory `iterate-family-0.5-evidence/`, sorted
by name, as regular files with mode 0644, mtime 0, uid and gid 0 and empty owner names, in GNU tar
format, compressed with gzip level 9 with mtime 0 and no stored file name. The same input gives the
same bytes with the same zlib; zlib-ng produces other deflate bytes, so the SHA-256 of the
uncompressed tar is printed as well. Python 3.11 or later; standard library only.

Exit codes: 0 packed, 2 usage error or input that is not the expected set.
"""
import gzip, hashlib, io, os, stat, sys, tarfile

TOP = "iterate-family-0.5-evidence"
RUNS = [f"{s}-{k}/julia-1.13.3-rep0{r}-{f}"
        for s in ("base", "aa") for k in ("groups", "layouts") for r in (1, 2, 3)
        for f in ("metrics.tsv", "metadata.toml", "errors.tsv")]
EXPECTED = sorted(RUNS + [f"cmp-{k}/{f}" for k in ("groups", "layouts")
                          for f in ("comparison.csv", "coverage.json", "REPORT.md")]
                  + ["crosscheck-pinned-start.tsv", "crosscheck-pinned-end.tsv", "restore-probe.txt"])
assert len(EXPECTED) == 45 and len(set(EXPECTED)) == 45


def fail(msg):
    print(f"pack_evidence.py: {msg}", file=sys.stderr)
    sys.exit(2)


def collect(src):
    """Return the contents of the 45 files, or exit if the directory is not exactly the set."""
    found, dirs = {}, set()
    for root, subdirs, files in os.walk(src, followlinks=False):
        rel_root = os.path.relpath(root, src).replace(os.sep, "/")
        for d in subdirs:
            path = os.path.join(root, d)
            if stat.S_ISLNK(os.lstat(path).st_mode):
                fail(f"symbolic link {path}")
            dirs.add(d if rel_root == "." else f"{rel_root}/{d}")
        for f in files:
            path = os.path.join(root, f)
            rel = f if rel_root == "." else f"{rel_root}/{f}"
            if not stat.S_ISREG(os.lstat(path).st_mode):
                fail(f"not a regular file: {rel}")
            found[rel] = path
    wanted_dirs = {n.rsplit("/", 1)[0] for n in EXPECTED if "/" in n}
    if sorted(found) != EXPECTED or dirs != wanted_dirs:
        missing = sorted(set(EXPECTED) - set(found))
        extra = sorted(set(found) - set(EXPECTED)) + sorted(dirs - wanted_dirs)
        fail(f"not the expected set of files; missing: {missing[:3]}{'...' if len(missing) > 3 else ''}; "
             f"unexpected: {extra[:3]}{'...' if len(extra) > 3 else ''}")
    out = []
    for name in EXPECTED:
        with open(found[name], "rb") as fh:
            out.append((name, fh.read()))
    return out


def main(argv):
    if len(argv) != 2:
        fail("usage: pack_evidence.py SRC_DIR OUT.tar.gz")
    src, out = argv
    if not os.path.isdir(src):
        fail(f"not a directory: {src}")
    if os.path.lexists(out):
        fail(f"refusing to overwrite {out}")
    members = collect(src)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.GNU_FORMAT) as tar:
        for name, data in members:
            info = tarfile.TarInfo(f"{TOP}/{name}")
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(data))
    tar_bytes = buf.getvalue()
    with open(out, "xb") as fh:
        with gzip.GzipFile(fileobj=fh, mode="wb", mtime=0, compresslevel=9, filename="") as gz:
            gz.write(tar_bytes)
    with open(out, "rb") as fh:
        gz_bytes = fh.read()
    print(f"members {len(members)}")
    print(f"tar.gz {len(gz_bytes)} bytes sha256 {hashlib.sha256(gz_bytes).hexdigest()}")
    print(f"tar    {len(tar_bytes)} bytes sha256 {hashlib.sha256(tar_bytes).hexdigest()}")


if __name__ == "__main__":
    main(sys.argv[1:])
