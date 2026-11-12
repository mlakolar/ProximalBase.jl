# Retained evidence

Verify these immutable Git artifacts before extraction:

```sh
cd docs/audit/block-updates-860/artifacts
sha256sum -c SHA256SUMS
mkdir -p /tmp/pb860-replay
for archive in raw-measurements.tar.gz sources-and-codegen.tar.gz validation-logs.tar.gz; do
  tar -xzf "$archive" -C /tmp/pb860-replay
 done
```

`inventory.json` gives every member's size and SHA256, in addition to archive
hashes. No Julia distribution is redistributed. Per-process provenance records
Julia 1.13.3, its executable hash, compiler/thread flags and source hashes; the
harness Project/Manifest pins dependencies. The original #852 distribution
manifest supplies additional compiler/library provenance for this same install.

- `raw-measurements.tar.gz`: full observations, individual time/GC samples,
  machine/fixture/source provenance and executed harness snapshots for the
  seven-round historical diagnosis, initial bulk discovery, placement-controlled
  broad comparison, final 21-round confirmation and bounds-enabled references.
  `diagnosis-harness/` retains the older diagnosis runtime and shared #852 helpers.
- `sources-and-codegen.tar.gz`: exact frozen `sources/base`, `sources/bulk-inline`,
  `sources/bulk-outlined` and `sources/singleton` trees, complete typed/LLVM/native
  output in `generated-code/`, and the generator harness/environment. These are
  rejected research candidates; package source is unchanged.
- `validation-logs.tar.gz`: complete local bounds-enabled package tests before
  and after rebasing onto merged #861, plus latest outlined/singleton reference
  tests. Final Linux/current-Julia CI is linked from PR #867.

Replay statistics from the repository root without Julia or new measurements:

```sh
python3 benchmark/audit860/analyze.py /tmp/pb860-replay/confirmation21 \
  --numerator candidate --output /tmp/pb860-singleton.csv
python3 benchmark/audit860/analyze.py /tmp/pb860-replay/confirmation21 \
  --numerator outlined --output /tmp/pb860-outlined.csv
python3 benchmark/audit860/analyze.py /tmp/pb860-replay/confirmation21 \
  --numerator bulk --output /tmp/pb860-bulk.csv
python3 benchmark/audit860/analyze.py /tmp/pb860-replay/confirmation21 \
  --numerator baseAA --output /tmp/pb860-AA.csv
```

These four outputs match `results/confirmation-*.csv` exactly. The analyzer
requires all expected source/round/case pairs, checks compiler/seed/source
provenance and recomputes medians from raw samples before bootstrapping rounds.
It reads recorded source paths as identities; it does not need those original
absolute directories to exist. To rerun Julia measurements, use the frozen
source directories above or reconstruct them with the rejected patches, and
follow `benchmark/audit860/README.md`.

The stopped/incorrect preliminary development campaigns are excluded from
published results and these archives. No observations from a candidate that
contained the concurrent #861 patch enter any reported comparison.
