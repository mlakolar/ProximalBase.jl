# Verification record for the four pilot entries

Reproducible probes behind the four [function-entry prototypes](../prototypes/functions/).
They ran on Julia 1.13.4 at the audited revision, locally. Run them from the repository
root with `julia --project=docs <script>`; the outputs below are the unedited output of
that command.

| Script | Output |
|:--|:--|
| `pilot_norml1.jl` | `pilot_norml1.txt` |
| `pilot_leastsquaresloss.jl` | `pilot_leastsquaresloss.txt` |
| `pilot_groupnorml2.jl` | `pilot_groupnorml2.txt` |
| `pilot_indballl1.jl` | `pilot_indballl1.txt` |

## What the lines mean

Lines are classified by what they compare, not by how they are printed:

- **Reference comparisons** compare a result with a value the script computes
  independently or states as expected: a formula, the normal equations, or the
  sorting-based ℓ₁ projection (its comparison on random inputs and on a matrix in
  `pilot_indballl1.jl`). Most are `check` lines; the matrix-response gradient and
  proximal point in `pilot_leastsquaresloss.jl` print their comparison without the
  `check` prefix.
- **Consistency checks** compare one ProximalBase call with another: `value_and_prox!`
  against `f(out)` (`pilot_norml1.jl`, `pilot_leastsquaresloss.jl`, `pilot_groupnorml2.jl`)
  and the projection at different steps (`pilot_indballl1.jl`).
- **Observed returns** are the other lines: what a call returned or which exception it
  threw, with no reference value. Section headings start with `==`.

The single `FAIL` line, in `pilot_groupnorml2.txt`, is the consistency check of
`value_and_prox!`: the rounding discrepancy recorded in the audit's
[separately recorded findings](../audit.md#defects-and-gaps-recorded-separately).

## Capability cells the scripts exercise

| Function | Operations exercised | Inputs exercised | Not exercised |
|:--|:--|:--|:--|
| `NormL1` | value, prox (scalar and array steps), `value_and_prox!`, coordinate `prox_block`, the three `kkt_violation` forms on a vector; gradient, group `prox_block` and the whole-point score on a `SymmetricSparseIterate` throw | vector, strided view, matrix, `Symmetric`, `SparseIterate` (vector and matrix), `BlockIterate`, `SymmetricSparseIterate` | `restrict`; the other score forms on iterates |
| `LeastSquaresLoss` | value, gradient, `value_and_gradient`, `lipschitz`, prox, `value_and_prox!`, whole-point score, `restrict`, `reset_cache!`, held-predictor calls; coordinate `prox_block` and coordinate scores throw | vector, strided view, `SparseIterate` (vector), `BlockIterate`, matrix with a matrix response; a matrix with a vector response, `Symmetric` and `SymmetricSparseIterate` throw | matrix `SparseIterate` |
| `GroupNormL2` | value, prox, `value_and_prox!`, `prox_block`, `prox_block!`, per-group and whole-point scores; the coordinate screen, `restrict` and `prox_block` under `Scaled` throw | vector, strided view, matrix, `Symmetric`, `SparseIterate` (vector), `BlockIterate` (block write and a per-group score with matching groups); `SymmetricSparseIterate` throws | prox and the whole-point score on a `BlockIterate`; scores with mismatched layouts |
| `IndBallL1` | value, projection at several steps, `value_and_prox!`; gradient, `kkt_violation`, `prox_block` and `lipschitz` throw | vector, strided view, matrix, `Symmetric`, `SparseIterate` (vector), `BlockIterate`; `SymmetricSparseIterate` throws | `restrict` |

The scripts also include some non-finite inputs and steps, an integer level and integer
data, `Float32` and `BigFloat` data with a `Float64` function, and exact in-place calls.
They do not cover Julia versions other than 1.13.4, offset arrays, `Float16`,
concurrency, partially overlapping arrays or most endpoint rules, which the guarantee
pages own.

## The rest of the capability tables

Only the cells listed above rest on these scripts. All other cells of the
[capability-table prototype](../prototypes/capabilities.md), including the cells of the
pilot functions marked "not exercised", are provisional: they come from an exploratory
probe run during the audit, whose scripts are not part of this record, and from the
docstrings. G4 establishes each cell by the rule in the
[table design](../compatibility-tables.md#what-counts-as-evidence).
