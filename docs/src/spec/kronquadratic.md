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
check semidefiniteness. The exact product ``\|A\|_2\|B\|_2`` of the spectral
norms is a Lipschitz constant of the gradient; `lipschitz` returns its
working-precision approximation, which can round below it. `size(f)` is
``(p, q)``.

## [Supported arguments](@id spec-kronquadratic-arguments)

A matrix argument is a one-based `AbstractMatrix{T}` of size ``(p, q)``:
dense matrices, strided views, `SparseIterate`, `Symmetric` wrappers and
`matrixview` of a symmetric iterate. An accepted argument (see the failure
behavior below) is read directly by the products when it is a `StridedMatrix`
with first stride 1 and a column stride of at least ``\max(1, p)``; any other
accepted matrix argument, reversed columns included, is copied into the
function's scratch first.

A [`SymmetricSparseIterate`](@ref) argument has element type `T` and
dimension ``p``, and requires ``p=q``.

`gradient!` and `value_and_gradient!` write into a one-based
`StridedMatrix{T}` of size ``(p, q)``, row-strided views included; a
reinterpretation passed as the destination must be one Base can read and
write. With a matrix argument, an accepted destination that is not a
`StridedMatrix` with first stride 1 and column stride of at least
``\max(1, p)`` receives a copy of the result from scratch; with a symmetric
iterate, the destination is written entry by entry. `gradient` returns a
`Matrix{T}`, and `value_and_gradient` returns `(v, G)` with `v::T` and
`G::Matrix{T}`; both check the argument before they allocate. Vector
arguments, `BlockIterate`, symmetric-iterate destinations and proximal
operators are unsupported.

## [Preconditions](@id spec-kronquadratic-preconditions)

The matrices that [`ProximalBase.kronquadratic_data`](@ref) returns are the
stored copies, ordinary mutable matrices; treat them as read-only. The oracles
write scratch matrices stored in the function: calls on one object must not
run concurrently. Use one object per task.

## [Results](@id spec-kronquadratic-results)

The value has type `T`. For a matrix argument the gradient is ``AXB+C``.
For a symmetric iterate it is ``\frac12(W+W^\top)+C`` with ``W=AXB``. Each
off-diagonal pair is computed once and the same value, `NaN` included, is
written to both entries, so `issymmetric` holds for results without `NaN`.
Its off-diagonal entry ``(r,c)`` is half the derivative of ``q`` along the
coordinate ``(r,c)``, which moves both mirrored entries. The value is the
same formula for both arguments.

## [Ordinary-path clauses](@id spec-kronquadratic-ordinary)

For a matrix argument, the oracles form ``AX`` and ``(AX)B`` with `mul!`,
which uses BLAS for eligible layouts and sizes, and the value as
``\langle X,AXB\rangle/2+\langle C,X\rangle``. At a symmetric iterate with
``n`` nonzero active values, ``AX`` is accumulated over the active coordinates
when ``2n\leq p^2/8`` (the accumulating route), and formed by `mul!` on a
dense copy otherwise. A zero coefficient, stored or absent, contributes
nothing to that accumulation. The gradient then forms ``(AX)B`` with `mul!`;
on the accumulating route its value is a pass over the stored entries. A
value-only call on the accumulating route forms ``AX`` and ``BX`` over the
active coordinates and sums ``\sum_{ij}(AX)_{ij}(BX)_{ji}`` in tiles of up to
``32\times32`` entries, without forming ``(AX)B``. Results are rounded in
working precision, and the routes can differ by rounding.

## [Range clauses](@id spec-kronquadratic-range)

Values and gradients use ordinary working-precision arithmetic, without the
range recovery that the [shared range guarantees](@ref numerical-value-range)
describe for other quadratic values. The selected kernel rounds its
operations in `T`; the accumulating kernels use `muladd`, and BLAS kernels
can also fuse a multiplication with an addition and round once. A result can
be subnormal, zero or nonfinite where the exact value is not, and the outcome
depends on the route and on the order of the operations. For example, the
accumulating symmetric routes add the two mirrored entries of `C` before
multiplying by the coefficient, which can overflow where the dense route does
not.

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
(`DimensionMismatch`), a strided destination (`ArgumentError`), a
reinterpretation passed as the destination that Base cannot read and write,
or passed as the argument that Base cannot read, directly or through views and
reshapes (`ArgumentError`), an argument or destination that shares memory
with the scratch of the function (`ArgumentError`), a destination that
shares memory with the argument as `Base.mightalias` reports it, also through
`Symmetric`, `Hermitian`, triangular, `Transpose` and `Adjoint` wrappers and
the backing vectors of `Diagonal`, `Bidiagonal`, `SymTridiagonal` and
`Tridiagonal` arguments, or with `A`, `B` or `C` (`ArgumentError`), and at a
symmetric iterate ``p=q`` (`DimensionMismatch`) and, when `C` is supplied,
its exact symmetry (`ArgumentError`). `C` is not averaged.

These checks cover the arrays the caller passes: the destination, the
argument, `A`, `B` and `C`. Storage inside other wrappers is not checked for
readability. If a read of it fails during the call, for example a padded
reinterpretation inside `Symmetric` or `transpose`, the call throws that
error with the destination and the argument unchanged; the scratch of the
function may be partly written, and the next call overwrites it.

## [Limitations and non-guarantees](@id spec-kronquadratic-limitations)

There is no proximal operator, Hessian-vector product, coordinate curvature
or `restrict`. Convexity is not certified. Overflow and underflow are not
recovered. Readability of storage inside wrappers is not checked before the
call (see the failure behavior). The summation order depends on the route:
BLAS kernels where `mul!` uses them; the column accumulation of the
accumulating route; the tiled sum of its value-only call; and Julia's `dot`
or a pass over the stored entries for the other values.

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
