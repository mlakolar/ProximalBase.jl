# [KronQuadratic numerical contract](@id spec-kronquadratic)

## [Identity](@id spec-kronquadratic-identity)

[`KronQuadratic`](@ref KronQuadratic) is the Kronecker-structured quadratic
of a matrix argument. Its constructor and oracles are implemented in
[`src/functions/kronquadratic.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/functions/kronquadratic.jl).

## [Definition and domain](@id spec-kronquadratic-definition)

`KronQuadratic(A, B, C = nothing)` represents
``q(X)=\frac12\langle X,AXB\rangle_F+\langle C,X\rangle_F`` for a
``p\times q`` matrix ``X``, with ``C=0`` when it is omitted. This equals
``\frac12\operatorname{vec}(X)^\top(B\otimes A)\operatorname{vec}(X)+\operatorname{vec}(C)^\top\operatorname{vec}(X)``.

`A` is ``p\times p`` and `B` is ``q\times q`` with ``p,q\geq1``. Both are
finite and exactly symmetric: mirrored entries compare equal, or the argument
is a `Symmetric`, `Hermitian` or `Diagonal` wrapper. `C` is a finite
``p\times q`` matrix. `A`, `B` and `C` have one element type `T`, `Float32`
or `Float64`. The constructor stores dense copies; later changes to the
caller's arrays do not change the function.

The gradient over ``p\times q`` matrices is ``AXB+C``. Over symmetric
``p\times p`` matrices with the Frobenius inner product it is
``\frac12(AXB+BXA)+C`` for a symmetric `C`.

## [Traits](@id spec-kronquadratic-traits)

The function is smooth. It is not proximable or separable, and
`isconvex` is false. The function is convex when `A` and `B` are both
positive semidefinite or both negative semidefinite; the constructor does not
check semidefiniteness. `lipschitz` returns the computed product
``\|A\|_2\|B\|_2`` of the spectral norms, which is not rounded up to a
certified bound. `size(f)` is ``(p, q)``.

## [Supported arguments](@id spec-kronquadratic-arguments)

A matrix argument is a one-based `AbstractMatrix{T}` of size ``(p, q)``:
dense matrices, strided views, `SparseIterate`, `Symmetric` wrappers and
`matrixview` of a symmetric iterate. A matrix argument whose columns are not
contiguous is copied into the function's scratch before the products.

A [`SymmetricSparseIterate`](@ref) argument has element type `T` and
dimension ``p``, and requires ``p=q``.

`gradient!` and `value_and_gradient!` write into a one-based writable
`StridedMatrix{T}` of size ``(p, q)``, row-strided views included. The
allocating `gradient` and `value_and_gradient` return a `Matrix{T}`. Vector
arguments, `BlockIterate`, symmetric-iterate destinations and proximal
operators are unsupported.

## [Preconditions](@id spec-kronquadratic-preconditions)

The matrices that [`ProximalBase.kronquadratic_data`](@ref) returns are the
stored copies; treat them as read-only. The oracles write scratch matrices
stored in the function: calls on one object must not run concurrently. Use
one object per task.

## [Results](@id spec-kronquadratic-results)

The value has type `T`. For a matrix argument the gradient is ``AXB+C``.
For a symmetric iterate it is ``\frac12(W+W^\top)+C`` with ``W=AXB``,
computed once for each off-diagonal pair and written to both entries, so the
destination is exactly symmetric. Its off-diagonal entry ``(r,c)`` is half
the derivative of ``q`` along the coordinate ``(r,c)``, which moves both
mirrored entries. The value is the same formula for both arguments.

## [Ordinary-path clauses](@id spec-kronquadratic-ordinary)

For a matrix argument, the oracles form ``AX`` and ``(AX)B`` with BLAS and
the value as ``\langle X,AXB\rangle/2+\langle C,X\rangle``. At a symmetric
iterate with ``n`` nonzero active values, ``AX`` is accumulated over the
active coordinates when ``2n\leq p^2/8``, and formed by a dense product of a
copy otherwise. A zero coefficient, stored or absent, contributes nothing to
that accumulation. A value-only call on the accumulating route forms ``AX``
and ``BX`` over the active coordinates and sums
``\sum_{ij}(AX)_{ij}(BX)_{ji}``. Results are rounded in working precision,
and the two routes can differ by rounding.

## [Range clauses](@id spec-kronquadratic-range)

Values and gradients use ordinary working-precision arithmetic, without the
range recovery that the [shared range guarantees](@ref numerical-value-range)
describe for other quadratic values. A product that overflows gives `Inf` or
`NaN`, and a value below the subnormal range rounds to zero.

## [Endpoint clauses](@id spec-kronquadratic-endpoints)

`NaN` and infinite entries of a matrix argument propagate by IEEE
arithmetic. At a symmetric iterate on the accumulating route, zero
coefficients are skipped; the subsequent products follow IEEE arithmetic.
An empty symmetric iterate has value zero and gradient `C` (zero when `C` is
omitted).

## [Failure behavior](@id spec-kronquadratic-failures)

The constructor throws `ArgumentError` for element types other than one
common `Float32` or `Float64`, with a suggested conversion such as
`Float64.(C)`; for empty `A` or `B`; for non-finite entries; for offset axes;
and for an `A` or `B` that is not exactly symmetric, with the message of
the shared symmetry check. A nonsquare `A` or `B`, or a `C` of the wrong
size, throws `DimensionMismatch`.

The oracles check, before any write, the element types of the argument and
the destination (`ArgumentError`), one-based axes (`ArgumentError`), sizes
(`DimensionMismatch`), a writable strided destination (`ArgumentError`), a
destination that shares memory with the argument, `A`, `B` or `C`
(`ArgumentError`), and at a symmetric iterate ``p=q`` (`DimensionMismatch`)
and an exactly symmetric `C` (`ArgumentError`). `C` is not averaged.

## [Limitations and non-guarantees](@id spec-kronquadratic-limitations)

There is no proximal operator, Hessian-vector product, coordinate curvature
or `restrict`. Convexity is not certified. Overflow and underflow are not
recovered. `Float32` and `Float64` results depend on the BLAS kernel's
summation order.

## [Clauses owned elsewhere](@id spec-kronquadratic-elsewhere)

This section links to the shared owners.

- [Array and precision conventions](@ref numerical-usage)
- [Shared range guarantees](@ref numerical-value-range)
- [Symmetric parameters](@ref symmetric-parameter-guide)
- [Kronecker quadratic guide](@ref Kronecker-quadratic)
- [Scaling a function](@ref scaling-guide)

## [Evidence](@id spec-kronquadratic-evidence)

[`test/kronquadratic.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/kronquadratic.jl)
checks values and gradients against analytic cases and Kronecker-product
references in `Float32` and `Float64`, central differences along matrix and
symmetric coordinate directions, the D-trace stationarity residual, a
penalized proximal-gradient loop on symmetric iterates with independent KKT
scores, the coordinate-descent direction model, `Scaled`, validation before
writes and warmed allocations. The
[`KronQuadratic` API example](@ref KronQuadratic) is executed by the strict
documentation build.

## [History](@id spec-kronquadratic-history)

The [restoration issue #1059](https://github.com/mlakolar/ProximalBase.jl/issues/1059)
records the numerical decisions and test plan.
