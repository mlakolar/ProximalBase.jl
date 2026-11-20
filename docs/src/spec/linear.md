# [LinearFunction and ZeroFunction numerical contract](@id spec-linear)

This page gathers the existing numerical rules for `LinearFunction`, `ZeroFunction`.
The text is preserved from the earlier manual. Clause conversion and status
assignments are a separate review step; unstated status remains unclassified.
Read [How to read the numerical contracts](index.md) for authority and ownership.

## [Identity](@id spec-linear-identity)

**Links only until clause conversion.** Constructors and API help: [`LinearFunction`](@ref) and [`ZeroFunction`](@ref).

Sources include [`src/functions/linear.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/functions/linear.jl)
and [`src/functions/zero.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/functions/zero.jl).
The [API coverage matrix](../api/coverage.md#API-coverage-matrix) lists the supported interface.

## [Definition and domain](@id spec-linear-definition)

**Links only until clause conversion.** Definitions and ordinary examples remain in [Linear function](../smooth.md#Linear-function) and [Zero function](../smooth.md#Zero-function).

## [Traits](@id spec-linear-traits)

**Links only until clause conversion.** Trait declarations are described by [`LinearFunction`](@ref) and [`ZeroFunction`](@ref) and [the function guide](../smooth.md#Linear-function).

## [Supported arguments](@id spec-linear-arguments)

**Links only until clause conversion.** Operation signatures and representations are described by [`LinearFunction`](@ref) and [`ZeroFunction`](@ref) and [the function guide](../smooth.md#Linear-function).

## [Preconditions](@id spec-linear-preconditions)

**Links only until clause conversion.** Use the [shape and mutation rules](../numerical_usage.md#Shapes-and-destination-first-mutation) and [supported precision](../numerics.md#Supported-precision-and-oracle-agreement), together with the family conditions linked in [Supported arguments](@ref spec-linear-arguments).

## [Results](@id spec-linear-results)

**Links only until clause conversion.** Return values are described by [`LinearFunction`](@ref) and [`ZeroFunction`](@ref). Shared output and aliasing rules are in [Ownership and aliases](../numerical_usage.md#Ownership-and-aliases).

## [Ordinary-path clauses](@id spec-linear-ordinary)

**Links only until clause conversion.** The ordinary mathematical formulas and examples remain in [the function guide](../smooth.md#Linear-function).

## [Range clauses](@id spec-linear-range)

For the `LinearFunction` value at a matrix `SparseIterate`, the finite
inputs are `d`, the nonzero stored entries and the coefficients at them (for a `Diagonal`
coefficient array, the nonzero stored diagonal entries and their coefficients: an off-diagonal
`NaN` or `Inf` of the iterate is ignored): the
recomputation sums their products as exponent-separated terms (as the `QuadraticFunction`
recovery does), collapses the sum and adds `d` to it in its own units, and a nonfinite coefficient at a zero of the
iterate does not prevent it. It reads the nonzero stored entries, `O(nactive(x))`; for a
`Diagonal` coefficient array it scans the `p` diagonal positions, `O(p)`, whatever
`nactive(x)` is. At a dense array, a vector `SparseIterate` and a `SymmetricSparseIterate`, the
recomputation sums the same exponent-separated terms when `d`, all coefficients and all entries of
the argument are finite.

At a matrix `SparseIterate` the value `f(x)` is computed from the stored entries of
`x` whose value is nonzero, in `O(nactive(x))` and without forming a zero for an unstored
entry; the order of the summation is not specified. A `Diagonal` coefficient array is the
exception to the cost: its value is summed over the nonzero stored diagonal entries in
`O(p)` (it scans the `p` diagonal positions), and the other entries of `x` are ignored, as
in the diagonal-only `dot` of `Diagonal`. An entry that is not stored, or is stored as
`+0` or `-0`, adds nothing, even when the coefficient at that position is `NaN`,
`Inf` or `-Inf`, and `c` is not read there; a nonzero entry follows ordinary
arithmetic, also when it or its coefficient is not finite. Historical traversal measurements
are recorded in the [implementation notes](https://github.com/mlakolar/ProximalBase.jl/blob/master/docs/design/function-docs-history-921.md#linearfunction-sparse-value-measurements). The constant `d` is added
after the sum. When the ordinary value is not finite but `d`, the nonzero entries
and the coefficients at them are, the value is recomputed from those entries (a
coefficient at a zero does not prevent it) as exponent-separated terms, each product
formed from its two factors without scaling one before the other acts. The sum is then
collapsed to one value with its own exponent, and `d` is added to that value in its own
units, so that when the compensated sum of the products cancels to zero the value is `d`,
and a small `d` is not absorbed by the larger terms. The accuracy is the absolute bound
`B = 4 n eps(T) (sum |cᵢxᵢ| + |d|)`, not relative to a dot much smaller than its terms, and the
returned value is finite when the exact value plus `B` is within the range of the type. When the
products lie far beyond the range, so that `B` exceeds it, there is no guarantee of a finite
value: extreme cancellation can give an infinity for a finite exact value (see
[Overflow and underflow of values](../numerics.md#Overflow-and-underflow-of-values)).

For a dense `c` that is not a `Diagonal`, `f(Matrix(x))` and `dot(c, x) + d` take the
ordinary `dot`, which visits all positions, so a `NaN`
coefficient at a zero gives `NaN` there: the value differs from them on purpose in that
case. When the participating entries of `x`, the coefficients at them, the coefficients
at the positions where `x` is zero, and `d` are all finite and the ordinary products and partial sums remain finite, the
value agrees with them to the accuracy of the sum. When a
participating entry is not finite, the order of summation can change the IEEE class: with
`c = ones(2, 2)`, `d = 0` and `h = 2^exponent(floatmax(T))`, storing `x[2] = h`, `x[3] = h`
and then `x[1] = -Inf`, a sum that adds `h` and `h` before the `-Inf` gives `NaN`, while
`f(Matrix(x))` and `dot(c, x) + d` give `-Inf`; `f(x)` may give either. This rule is for the matrix
`SparseIterate`: a vector `SparseIterate` keeps its own rules, and a `SymmetricSparseIterate`
forms the terms at its stored zeros, so a `NaN` coefficient there gives `NaN`.

```@example library_linear_sparse
using ProximalBase, LinearAlgebra
x = SparseIterate(2, 2)
x[1, 1] = 2.0
c = [1.0 NaN; 1.0 1.0]            # NaN at a position that x does not store
f = LinearFunction(c, 0.25)
@assert f(x) == 2.25
@assert isnan(f(Matrix(x)))
f(x)
```

## [Endpoint clauses](@id spec-linear-endpoints)

| Case | Retained behavior |
|:--|:--|
| `ZeroFunction` value | Zero even at NaN/Inf input entries |

**`LinearFunction`**

Non-finite entries follow arithmetic,
except at a zero of a matrix `SparseIterate` (below); finite-data overflowed values
have a scaled recovery path.

See [Sparse product semantics](@ref Sparse-product-semantics) for the shared rule.

The `LinearFunction` value at a matrix `SparseIterate` follows the same rule, as
its own computation and not through `dot(c, x)`: a position that is not stored, or is
stored as `+0` or `-0`, contributes nothing and `c` is not read there, so a `NaN` or
`±Inf` coefficient at it does not reach the value, while a nonzero entry keeps
ordinary arithmetic. For a dense `c` that is not a `Diagonal`, `f(Matrix(x))` takes the
dense IEEE dot and gives `NaN` for such a coefficient.

```@example numerical_linear_sparse
using ProximalBase, LinearAlgebra
x = SparseIterate(2, 2)
x[1, 1] = 2
activate!(x, 4)                    # the entry (2, 2) is stored as a zero
c = [1.0 NaN; Inf 1.0]
f = LinearFunction(c, 0.25)
@assert f(x) == 2.25
@assert isnan(f(Matrix(x)))
f(x)
```

## [Failure behavior](@id spec-linear-failures)

**Links only until clause conversion.** Read [Supported arguments](@ref spec-linear-arguments), [Preconditions](@ref spec-linear-preconditions) and [Endpoint clauses](@ref spec-linear-endpoints), together with the failure behavior in [`LinearFunction`](@ref) and [`ZeroFunction`](@ref).

## [Limitations and non-guarantees](@id spec-linear-limitations)

**Links only until clause conversion.** Shared limits on range recovery and arithmetic-order agreement remain in [Range guarantees](@ref numerical-value-range).

## [Clauses owned elsewhere](@id spec-linear-elsewhere)

**Links only until clause conversion.** The following rules retain their shared owners.

| Subject | Owning section |
|:--|:--|
| Supported precision and oracle agreement | [Supported precision and oracle agreement](../numerics.md#Supported-precision-and-oracle-agreement) |
| Shared range bounds and recovery limits | [Range guarantees](@ref numerical-value-range) |
| Shapes and destination writes | [Shapes and destination-first mutation](../numerical_usage.md#Shapes-and-destination-first-mutation) |
| Ownership and aliasing | [Ownership and aliases](../numerical_usage.md#Ownership-and-aliases) |
| Sparse products and stored zeros | [Sparse product semantics](@ref Sparse-product-semantics) |

## [Evidence](@id spec-linear-evidence)

These existing tests contain the numerical checks for this family. Their assertions define the cases covered; listing a test does not establish a broader guarantee.

- [`test/linear_shapes.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/linear_shapes.jl)
- [`test/linear_sparse_value.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/linear_sparse_value.jl)
- [`test/linear_value_recovery.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/linear_value_recovery.jl)
- [`test/sparse_value_cost.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/sparse_value_cost.jl)
- [`test/kkt_point.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/kkt_point.jl)

## [History](@id spec-linear-history)

The [documentation migration](https://github.com/mlakolar/ProximalBase.jl/issues/1019) records the preserved source text and its review. Numerical decisions remain with their linked source, tests and issue records.
