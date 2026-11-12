# Approved sanitized evidence

[Audit release](https://github.com/mlakolar/ProximalBase.jl/releases/tag/audit860-2026-11-12),
sole attachment [audit860-sanitized.tar.gz](https://github.com/mlakolar/ProximalBase.jl/releases/download/audit860-2026-11-12/audit860-sanitized.tar.gz)
(25,944,785 bytes).

SHA256: `c1d1a84d1cd3a7d69525d36b0b3134dc59101c0327961f37cc0ca9b251312c61`.

The owner approved these exact independently reviewed bytes. Anonymous download
verification matched the checksum. Do not replace or republish this archive.
It contains raw timing/GC samples, allocation counts, seeds/padding, compiler/Julia
and hardware information, frozen source/harness snapshots, generated code,
comparison CSVs, regression ledger, inventories and privacy/review receipts.
Identifying environment metadata and runtime addresses are removed/anonymized;
original allocator placement cannot be reconstructed. Original execution hashes
and sanitized-file inventory hashes intentionally identify different snapshots.

Verify the archive checksum above, extract it and replay from the extracted root:

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
python3 generator/audit860/ledger.py results ledger.csv
cmp ledger.csv results/regression-ledger.csv
```

Python 3.11+ supplies `tomllib`; statistical replay needs no Julia or repository
checkout. Independent review reproduced these five outputs byte for byte.
The archived runner documents full fixture reproduction with Julia 1.13.3 and
the locked environment. Its synthetic paths are labels, so configure source/output
paths for a new run. Frozen runners can collect identifying metadata; sanitize
new outputs before sharing. No new benchmark campaign is part of this finalization.
The clean replacement does not erase [remaining historical exposure](remediation-876.md).
