# Retained raw data and provenance (#852)

These Git-tracked archives are immutable at the audit commit and independently
pinned by [SHA256SUMS](SHA256SUMS). [inventory.json](inventory.json) gives exact
byte/member counts, and [campaign-archive-map.json](campaign-archive-map.json)
maps measurement roots and generated harnesses to archives. No Julia binaries,
user credentials, or external mutable artifact links are required.

| Archive | Compressed bytes | Members | Contents |
|---|---:|---:|---|
| [excluded-development.tar.gz](excluded-development.tar.gz) | 499,034 | 3,804 | Failed/stopped development checks and boxed-callback pilot; excluded from every final result. |
| [raw-controls.tar.gz](raw-controls.tar.gz) | 27,551,778 | 153,515 | All remaining campaign roots and their exact generated analysis/timing harnesses. |
| [raw-primary.tar.gz](raw-primary.tar.gz) | 10,830,977 | 60,496 | Seven-round broad suite, native21 follow-up, allocation isolation including harness. |
| [sources-validation.tar.gz](sources-validation.tar.gz) | 7,085,399 | 5,243 | Four pristine source archives, strict correctness runs, code/assembly dumps, compilation diagnostics, logs, primary harness snapshot. |

All observations and individual time/GC samples for the 17 final campaigns are
in the first two archives. Each fresh process retains its provenance, stdout,
stderr, fixture metadata, seed, pointer/placement information and source/binary/
harness digests. The primary seven timing files match the committed
[runtime manifest](../research/runtime-hashes.json). Support files accumulated
during investigation, so the primary full-directory hash changes over time;
those hashes remain in provenance and are not misrepresented as one snapshot.
Each generated control harness is retained exactly and its full code digest
verifies against every worker that used it.

The sources-validation harness snapshot includes the then-current diagnostics
and supporting scripts; the final committed harness is the canonical entry
point. These support additions do not change the seven frozen primary timing
files. Historical baseline files remain byte-for-byte original. The excluded
pilot's exact unversioned harness was not fully captured; its raw files and
available development code/logs are retained without claiming exact pilot replay.

From the **repository root**, verify and extract to a new directory:

```sh
(cd docs/audit/iterate-performance-852/artifacts && sha256sum -c SHA256SUMS)
mkdir /tmp/pb852-published-replay
tar -xzf docs/audit/iterate-performance-852/artifacts/raw-primary.tar.gz -C /tmp/pb852-published-replay
tar -xzf docs/audit/iterate-performance-852/artifacts/raw-controls.tar.gz -C /tmp/pb852-published-replay
tar -xzf docs/audit/iterate-performance-852/artifacts/sources-validation.tar.gz -C /tmp/pb852-published-replay
python3 benchmark/audit852/replay_results.py --results docs/audit/iterate-performance-852/results --raw-parent /tmp/pb852-published-replay
```

Python 3.11+ suffices; no Julia execution or plotting dependency is needed to
recompute all exported summary metric cells from retained raw samples. The
replay uses each generated analysis harness recorded by campaigns.csv and the
committed primary analyze.py for MAIN. It checks incompatibilities separately,
ignoring only the explicit export attribution/scope annotations. Expect minutes
and many small files, rather than a single large CSV.

To independently recheck sample p90, source identity, all provenance endpoints,
compiler/thread/runtime constraints and generated code hashes as well, provide
the same Julia binary distribution (hashes in ../research/julia-distribution.json):

```sh
python3 benchmark/audit852/verify_integrity.py --results docs/audit/iterate-performance-852/results --raw-parent /tmp/pb852-published-replay --sources /tmp/pb852-published-replay/pb852-frozen --julia /absolute/path/to/julia-1.13.3/bin/julia
```

This reads/checks files rather than launching timing. A different platform build
can have a different executable hash despite the same version string. To make
new performance measurements on another machine, follow the full
[benchmark recipe](../../../../benchmark/audit852/README.md) and compare the
revisions there; do not divide new absolute times by this machine's timings.
The independent extracted-archive replay is recorded in
[archive-validation.json](../research/archive-validation.json).
