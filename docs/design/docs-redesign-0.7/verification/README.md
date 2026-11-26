# Verification record for the four pilot entries

Reproducible probes behind the four [function-entry prototypes](../prototypes/functions/).
They ran on Julia 1.13.4 at the audited revision, locally. Run them from the
repository root with `julia --project=docs <script>`; the outputs below were produced
that way.

| Script | Output |
|:--|:--|
| `pilot_norml1.jl` | `pilot_norml1.txt` |
| `pilot_leastsquaresloss.jl` | `pilot_leastsquaresloss.txt` |
| `pilot_groupnorml2.jl` | `pilot_groupnorml2.txt` |
| `pilot_indballl1.jl` | `pilot_indballl1.txt` |

Each script covers the constructors and their validation, the value, gradient or
proximal operator, `value_and_prox!`, `prox_block`, `kkt_violation`, a set of input
kinds, `Scaled` and the printed form. Two kinds of line appear in the outputs:

- a line `check … OK` or `check … FAIL` compares the result with a value computed
  independently in the script (a formula, normal equations, or a sorting-based
  projection);
- every other line records what a call returned or which exception it threw, without
  an independent reference.

The single `FAIL` line, in `pilot_groupnorml2.txt`, is the `value_and_prox!` rounding
discrepancy recorded in the audit's
[separately recorded findings](../audit.md#defects-and-gaps-recorded-separately).

## Scope

The scripts include some non-finite inputs and steps, an integer level and integer
data, `Float32` and `BigFloat` data with a `Float64` function, and exact in-place calls.
They do not cover Julia versions other than 1.13.4, offset arrays, `Float16`,
concurrency, partially overlapping arrays or most endpoint rules, which the guarantee
pages own.

## The capability tables

The cells of the [capability-table prototype](../prototypes/capabilities.md) for these
four functions rest on these scripts. The cells for the other functions come from an
exploratory probe run during the audit, whose scripts are not part of this record, and
from the docstrings and guarantee pages; they are to be confirmed in G4, which records
the test behind each cell in the API coverage matrix.
