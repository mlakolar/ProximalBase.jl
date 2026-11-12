# Benchmark evidence

The [sanitized audit release](https://github.com/mlakolar/ProximalBase.jl/releases/tag/audit860-2026-11-12)
contains one attachment, published with explicit approval of its exact bytes:

[audit860-sanitized.tar.gz](https://github.com/mlakolar/ProximalBase.jl/releases/download/audit860-2026-11-12/audit860-sanitized.tar.gz)
— 25,944,785 bytes.

SHA256: `c1d1a84d1cd3a7d69525d36b0b3134dc59101c0327961f37cc0ca9b251312c61`.

The local checksum matched the independent contents/packaging review receipt
before upload. An anonymous public download subsequently matched the same
checksum and byte count. No original archives were attached to this release.
Raw results, logs, archives and generated inventories are excluded from Git's
current tree.

## Privacy and contents

Host/user identifiers and original path roots are removed or anonymized.
Recorded fixture pointers and literal runtime addresses are removed; diagnostic
code listings use `ADDRESS_REDACTED` for address operands. Original allocator
placement cannot be reconstructed from the sanitized data. Seeds and held
padding sizes remain, as do CPU specifications, Julia/compiler information,
binary/source hashes, timings, GC samples and allocation observations.
Archive owner/group names and timestamps are normalized.

The bundle includes `README-sanitized.md`, `INDEPENDENT-REVIEW.md`,
`PRIVACY-SCAN.json`, `EQUIVALENCE.json`, `SANITIZATION.json`, `inventory.json`
and `SHA256SUMS`. Independent review checked all 1,420 archive members, including
all 93 byte-identical source/CSV files and 672 generated-code listings with
599 address substitutions. All 185 raw-result files preserve 99,168 observations
and 7,929,762 individual timing samples across the separate campaigns.
Original execution hashes identify executed snapshots; inventory hashes identify
the sanitized files. Frozen runners can collect identifying metadata on new
runs, so sanitize newly generated outputs separately before sharing them.

Original unsanitized archives were previously committed in `cc447af9` and
removed from the current tree in `81e973c6`. This attachment does not erase those
historical Git objects. Their cleanup is a separate maintainer decision.

## Summaries

The bundled `results/regression-ledger.csv` accounts for all discovery/confirmation
median flags above 1.05: 64 contrasts, 41 repeated median flags, 37 confirmation
intervals entirely above 1.05, and no timer-limited rows. Discovery and controlled
phases are not pooled. The final comparison has 576 cases, five roles and
21 fresh rounds.

## Reproduction

The maintained [recipe](../../../benchmark/audit860/README.md) reconstructs the
frozen experiments using append-only research patches. For statistical replay,
verify the attachment's SHA256 above, extract it, and run from its root:

```sh
tar -xzf audit860-sanitized.tar.gz
cd audit860-sanitized
sha256sum --quiet --check SHA256SUMS
python3 generator/audit860/analyze.py confirmation21 --numerator candidate --output singleton.csv
python3 generator/audit860/analyze.py confirmation21 --numerator outlined --output outlined.csv
python3 generator/audit860/analyze.py confirmation21 --numerator bulk --output bulk.csv
python3 generator/audit860/analyze.py confirmation21 --numerator baseAA --output AA.csv
cmp singleton.csv results/confirmation-singleton.csv
cmp outlined.csv results/confirmation-outlined.csv
cmp bulk.csv results/confirmation-bulk.csv
cmp AA.csv results/confirmation-AA.csv
python3 generator/audit860/ledger.py results regression-ledger.csv
cmp regression-ledger.csv results/regression-ledger.csv
```

Independent review performed all four confirmation replays and the ledger replay,
matching the bundled CSVs byte for byte. None of the rejected patches changes
the shipped package.
