# Benchmark evidence

The prepared evidence is independently verified and retained locally. Public
upload to the dedicated GitHub audit attachment is awaiting explicit approval.
No raw results, logs, binary archives or generated inventories remain in Git.

Planned destination: `https://github.com/mlakolar/ProximalBase.jl/releases/tag/audit860-2026-11-12`.

| Prepared file | Bytes | SHA256 |
|:--|--:|:--|

| `raw-measurements.tar.gz` | 23,277,764 | `52f9e5cd0a4d1b9ae2b7992884a675ff51c8809fbe24a80e34849a101a67db9b` |

| `sources-and-codegen.tar.gz` | 3,002,996 | `1fb1572839ca5d553947df4efa6c74eecc39f7b4954cfc2e711fd01a33c05f45` |

| `validation-logs.tar.gz` | 5,094 | `a99a57809d650ede971be6ee900992d663d8d19eac99464125f3c8d15295c2f0` |

| `summaries.tar.gz` | 356,444 | `6cc28e85cfaa0af298ab513de2e11feee3dd9ce4bf71d996dcdf01e42bad7a7e` |


The local publication directory is `/tmp/pb860/publication`. `SHA256SUMS` and
`inventory.json` accompany the archives; the latter records every member hash.

## Contents

- `raw-measurements.tar.gz`: individual timings/GC samples, process/source/compiler
  provenance, fixture addresses and executed harness snapshots.
- `sources-and-codegen.tar.gz`: exact baseline and rejected source trees, full
  typed/LLVM/native listings, generator and locked environment.
- `validation-logs.tar.gz`: complete local bounds-enabled tests and references,
  including the final 20-file run after #861.
- `summaries.tar.gz`: all twelve comparison CSVs and the 64-row regression ledger.

## Summaries

The ledger accounts for every discovery/confirmation median above 1.05:
64 contrasts, 41 repeated median flags, 37 confirmation intervals entirely
above 1.05, and no timer-limited rows. Discovery and controlled phases are
not pooled. The final comparison has 576 cases, five roles and 21 fresh rounds.

## Reproduction

The maintained [recipe](../../../benchmark/audit860/README.md) reconstructs
the frozen experimental candidates with append-only research patches. It
does not copy a moving checkout, which could include unrelated #861 changes.
After obtaining the prepared archives, verify `SHA256SUMS`, extract them into
one new directory, and replay statistics without Julia:

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

Independent review has already performed all four archive replays and matched
the prepared CSVs byte for byte. Sources and artifacts are research evidence;
none of the rejected patches changes the shipped package.
