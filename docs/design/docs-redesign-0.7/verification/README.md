# Verification record for the capability claims

The evidence behind the [capability-table prototype](../prototypes/capabilities.md),
the four function entries and the drift findings of the [audit](../audit.md). All runs
used Julia 1.13.4 at the audited revision, locally. Run the scripts from the
repository root with `julia --project=docs <script>`.

## Files

| File | What it does | Evidence |
|:--|:--|:--|
| `pilot_norml1.jl`, `pilot_leastsquaresloss.jl`, `pilot_groupnorml2.jl`, `pilot_indballl1.jl` | constructors and their validation, value, gradient, prox, `value_and_prox!`, `prox_block`, `kkt_violation`, storage kinds, `Scaled`, display | each operation compared with an independent formula; printed `check … OK/FAIL` lines |
| `pilot_*.txt` | the output of each script | rerun on the audited revision: identical |
| `smoke_matrix.jl` | every function type × 15 kinds of input × 14 operations | each call compared with the dense `Float64` result for the same point: a consistency check, not an independent formula |
| `operations.tsv` | one row per function × operation (33 functions including three `Scaled` wrappers, 17 operations) | support, evidence level, restriction note, probe detail, test references |
| `representations.tsv` | one row per function × kind of input (15 kinds) | `ok`, a mismatch or the exception for each operation group; test references |

The single `FAIL` line, in `pilot_groupnorml2.txt`, is the `value_and_prox!` rounding
discrepancy recorded in the audit's
[separately recorded findings](../audit.md#defects-and-gaps-recorded-separately).

## Evidence levels in `operations.tsv`

| Level | Meaning | Rows |
|:--|:--|--:|
| `probed+tested` | probed, and a test file exercises the call | 87 |
| `tested` | a test file exercises the call; not probed independently | 123 |
| `probed` | probed (consistency or formula), no test found | 38 |
| `throws` | the call throws (`MethodError` or an explicit `ArgumentError`) | 280 |
| `method-only` | a method exists, but neither a probe nor a test was found | 3 |
| `untested-fallback` | the generic no-op `reset_cache!` | 30 |

Test references were found by searching the test files for the operation near a
constructor of the function. No test was run, and the search can miss or overcount.

## What was not verified

- Julia 1.10 or any version other than 1.13.4.
- Offset arrays, `Float16`, element types other than `Float64` and `Float32` beyond the
  smoke cases, mixed precision, non-finite inputs, concurrency and partially overlapping
  arrays.
- `Scaled` around inner functions other than `NormL1`, `LeastSquaresLoss` and
  `GroupNormL2`.
- The conditions in the capability notes that come from the documentation rather
  than from probes (for example the restriction rule of `IndBox` and the step rule of
  `LogDetLoss`) were read from the docstrings and guarantee pages, not re-probed.

G4 replaces this record with a column of the API coverage matrix that names the test
behind each cell.
