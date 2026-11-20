# [SqrtLeastSquaresLoss numerical contract](@id spec-sqrt_leastsquares)

This page gathers the existing numerical rules for `SqrtLeastSquaresLoss`.
The text is preserved from the earlier manual. Clause conversion and status
assignments are a separate review step; unstated status remains unclassified.
Read [How to read the numerical contracts](index.md) for authority and ownership.

## [Identity](@id spec-sqrt_leastsquares-identity)

**Links only until clause conversion.** Constructors and API help: [`SqrtLeastSquaresLoss`](@ref).

Source: [`src/functions/sqrt_leastsquares.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/functions/sqrt_leastsquares.jl).
The [API coverage matrix](../api/coverage.md#API-coverage-matrix) lists the supported interface.

## [Definition and domain](@id spec-sqrt_leastsquares-definition)

**Links only until clause conversion.** The mathematical definition and ordinary examples are in [the function guide](../smooth.md#Square-root-least-squares-is-not-smooth).

## [Traits](@id spec-sqrt_leastsquares-traits)

**Links only until clause conversion.** Trait declarations are described by [`SqrtLeastSquaresLoss`](@ref) and [the function guide](../smooth.md#Square-root-least-squares-is-not-smooth).

## [Supported arguments](@id spec-sqrt_leastsquares-arguments)

**Links only until clause conversion.** Operation signatures and representations are described by [`SqrtLeastSquaresLoss`](@ref) and [the function guide](../smooth.md#Square-root-least-squares-is-not-smooth).

## [Preconditions](@id spec-sqrt_leastsquares-preconditions)

**Links only until clause conversion.** Use the [shape and mutation rules](../numerical_usage.md#Shapes-and-destination-first-mutation) and [supported precision](../numerics.md#Supported-precision-and-oracle-agreement), together with the family conditions linked in [Supported arguments](@ref spec-sqrt_leastsquares-arguments).

## [Results](@id spec-sqrt_leastsquares-results)

**Links only until clause conversion.** Return values are described by [`SqrtLeastSquaresLoss`](@ref). Shared output and aliasing rules are in [Ownership and aliases](../numerical_usage.md#Ownership-and-aliases).

## [Ordinary-path clauses](@id spec-sqrt_leastsquares-ordinary)

**Links only until clause conversion.** The ordinary mathematical formulas and examples remain in [the function guide](../smooth.md#Square-root-least-squares-is-not-smooth).

## [Range clauses](@id spec-sqrt_leastsquares-range)

The definitions of `u` and the boundary regions referred to below are in the [least-squares predictor range discussion](../smooth.md#Least-squares-at-a-predictor-or-residual-that-the-caller-holds).

**Accuracy and range.** The value is never formed as ``\sqrt{2\,v_{\text{LS}}}`` from the least-squares value
``v_{\text{LS}}``, whose squares overflow where the norm does not. For strided `Float32` or `Float64` vectors of one
element type, which covers both oracles, the sum of squares ``s`` is formed in `T` in one vectorizing loop up to
``2^{17}`` entries and by pairwise summation above. The other arguments of `f(β)`, such as mixed element types or
non-strided arrays, take generic sums in one loop at every length. The product ``\alpha\sqrt s`` is kept when ``s`` is not below ``n\,\texttt{floatmin}(T)/\varepsilon_T`` and the
product is below ``\texttt{floatmax}(T)/2`` (a `NaN` or infinite ``s`` is not kept). Otherwise, and for every sum that is not finite, the norm is recomputed
from the squares in units of a power of two, with ``\alpha`` folded in and the squares summed compensated, in
at least `Float64`. For finite entries that `T` represents exactly, ``n\le 2^{40}`` of them, with ``(n+3)\varepsilon_T\le 1/16`` or
strided `Float32` or `Float64` vectors of one element type, and ``v`` the exact value, the result is within ``(n/2+4)\,\varepsilon_T|v|`` of ``v``, plus two units of
the smallest subnormal where ``|v|<\texttt{floatmin}(T)``; with ``u`` and ``w=(n/2+4)\varepsilon_T`` as in the
range statement above, the result is finite where ``|v|(1+w)\le\texttt{floatmax}(T)``, is `Inf` above
``(\texttt{floatmax}(T)+u/2)(1+w)``, and between the two is `Inf` or finite within the bound (`floatmax(T)` is possible
within ``w`` of the threshold, below it as well as above it). The proof is in the comments of `src/functions/sqrt_leastsquares.jl`. `f(β)` returns its value converted to `float(eltype(β))`, so the bound holds for `eltype(β) == T`, and a narrower `β` adds one rounding to its type. An entry that `T` does not represent exactly (an `Int` `Y` above `maxintfloat(T)`, a `Rational` `Y`) is rounded to `T` in the sum, outside the bound. An exact fit gives
`+0`; a `NaN` entry gives `NaN`; an entry that is `±Inf` gives `Inf`, unless it meets an `Inf` of the same sign in `z`
and `Y`, where the difference is `NaN`.

## [Endpoint clauses](@id spec-sqrt_leastsquares-endpoints)

An entry of
``X\beta`` that overflows is not formed again, unlike `LeastSquaresLoss`: the value is then the norm of the computed
``X\beta-Y``, typically `Inf` or `NaN`.

## [Failure behavior](@id spec-sqrt_leastsquares-failures)

**Links only until clause conversion.** Read [Supported arguments](@ref spec-sqrt_leastsquares-arguments), [Preconditions](@ref spec-sqrt_leastsquares-preconditions) and [Endpoint clauses](@ref spec-sqrt_leastsquares-endpoints), together with the failure behavior in [`SqrtLeastSquaresLoss`](@ref).

## [Limitations and non-guarantees](@id spec-sqrt_leastsquares-limitations)

**The wrapper.** `Scaled(f, α)(β)` is the generic wrapper. It receives `f(β)` already rounded to `float(eltype(β))`,
forms the product with ``\alpha`` from that rounded value and returns it converted to `float(eltype(β))`. A `β` narrower
than `T` therefore rounds the inner value to its own type first: with `f = SqrtLeastSquaresLoss([0.0], fill(0.25, 1, 1))`,
`β = Float32[nextfloat(0f0)]` and `α = 4.0`, `f(β)` and `Scaled(f, α)(β)` are `0.0f0`, while the oracle at `X * β` is about
`1.4e-45`. For `eltype(β) == T`, with a `Float64` factor on `Float32` data the result can differ from the oracle in the last bit, and where
the oracle recovers a value that the wrapper's product loses it can differ on the exceptional range (issue #609, as for
the least-squares loss). `Scaled(f, Inf)(β)` is `Inf * f(β)`, which is `NaN` at an exact fit; the oracles refuse the
factor `Inf`.

## [Clauses owned elsewhere](@id spec-sqrt_leastsquares-elsewhere)

**Links only until clause conversion.** The following rules retain their shared owners.

| Subject | Owning section |
|:--|:--|
| Supported precision and oracle agreement | [Supported precision and oracle agreement](../numerics.md#Supported-precision-and-oracle-agreement) |
| Shared range bounds and recovery limits | [Range guarantees](@ref numerical-value-range) |
| Shapes and destination writes | [Shapes and destination-first mutation](../numerical_usage.md#Shapes-and-destination-first-mutation) |
| Ownership and aliasing | [Ownership and aliases](../numerical_usage.md#Ownership-and-aliases) |
| Sparse products and stored zeros | [Sparse product semantics](@ref Sparse-product-semantics) |

## [Evidence](@id spec-sqrt_leastsquares-evidence)

These existing tests contain the numerical checks for this family. Their assertions define the cases covered; listing a test does not establish a broader guarantee.

- [`test/sqrt_leastsquares.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/sqrt_leastsquares.jl)

## [History](@id spec-sqrt_leastsquares-history)

The [documentation migration](https://github.com/mlakolar/ProximalBase.jl/issues/1019) records the preserved source text and its review. Numerical decisions remain with their linked source, tests and issue records.
