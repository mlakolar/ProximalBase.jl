# Sparse iterates

ProximalBase provides three iterate types for coordinate descent and active-set
procedures. We separate the choice of storage from the update formulas,
which are derived in [Coordinate descent](@ref).

| Iterate | Array represented | One coordinate |
|:--------|:------------------|:---------------|
| [`SparseIterate`](@ref) | A sparse vector or matrix | One entry |
| [`SymmetricSparseIterate`](@ref) | A sparse symmetric matrix | One lower-triangular entry and its symmetric counterpart |
| [`AtomIterate`](@ref) | A vector partitioned into groups | One group of entries |

The entries of an iterate are real. A complex element type, or a complex
operand of an iterate product, raises an `ArgumentError`; see
[Precision](@ref).

## Entries and coordinates

[`prox_entry`](@ref)`(g, v, i, γ)` computes the scalar proximal operator of
entry ``i`` without writing the iterate. In contrast,
[`prox_coordinate!`](@ref)`(x, g, k, γ)` reads coordinate ``k`` of the iterate,
applies its proximal operator and writes the result. The calling procedure
must supply the gradient step before applying this proximal operation.

The two calls index different spaces:
- `prox_entry` takes an entry ``i`` of the array, the index of the weights and
  bounds.
- `prox_coordinate!` takes a coordinate ``k`` of the iterate.

The two coincide for a `SparseIterate`. A coordinate is an entry of the lower
triangle for a `SymmetricSparseIterate`, and a group for an `AtomIterate`.

## SparseIterate

A [`SparseIterate`](@ref) is a vector or matrix that stores its nonzero
entries in a list. Three arrays hold them: the values, their indices, and for
every index the position of its value in the list (zero when it is not stored).
Reading or writing any entry is therefore ``O(1)``, and a product ``Xx`` costs
``O(n\cdot\mathrm{nnz})``. The stored entries are the active coordinates
([`activecoordinates`](@ref), [`nactive`](@ref)). Setting an entry to zero keeps
it stored, so that an active set does not change during a sweep. `dropzeros!`
removes the zeros between sweeps. `nnz(x)` counts stored zeros too.
`nonzeros(x)` gives a view of the stored values, paired with
`activecoordinates(x)` in storage order; the view is invalidated when the
stored set changes. `findnz(x)` returns copies of the stored indices and
values in sorted order, including stored zeros.

### Matrix coefficients

`mul!(out, A, x, α, β)` with a matrix `x` walks the stored entries of `x` when
at most 1/32 of its entries are stored, and its columns in order otherwise. The
first reads one column of `A` per stored entry, in storage order; the second
reads `A` sequentially. Which is faster depends mostly on the length ``m`` of a
column of `A`. With entries stored in random order (as an active-set solver
creates them), the density at which the two walks take equal time (median over
shapes of `x`) falls with ``m``. For `Float64` it is 7–16% for 64 to 512 rows,
3–5% for 1024, about 1.5–2% for 2048 and 1–5% for 4096. The values are larger
for `Float32`. They were measured on Apple M3 Pro, AMD EPYC (Zen 3) and Intel
Xeon (Skylake-SP), and differ by a factor of two or more between these CPUs.

The fixed 1/32 is a compromise. Averaged over densities from 0.5% to 50% (log
scale, equal weight) it costs 2.5–5% of the time of the faster walk, 11–32% for
64-row `Float32` products. The best constant, 1/16, costs 1–3%. A cutoff
`min(1/8, 256/(m*sizeof(T)))` for ``m`` rows costs 0.6–1.5%. That gain over
1/16 was under the 2 points required, so no rule that depends on the size of
`A` was added (the margin was fixed after the first data, not before).

## SymmetricSparseIterate

A [`SymmetricSparseIterate`](@ref) is a sparse symmetric ``p \times p`` matrix
that stores its lower triangle. For indexing, length, iteration and
broadcasting it is an ordinary ``p \times p`` matrix, except that writing
``x_{rc}`` also writes ``x_{cr}``.

Read and write entries through the lower triangle, ``r \ge c``, for example
with `for c in 1:p, r in c:p`. The position `x[c, r]` is the same entry as
`x[r, c]`, so writing either position writes both. A loop over every position
writes each off-diagonal entry twice, and the second write wins.

Whole-value writes are checked. These include the constructor, `copyto!`,
`copy!`, broadcasting into ``x``, `map!`, `axpy!`, `axpby!`, `transpose!`,
`adjoint!`, `permutedims!` and `mul!`. Indexed array assignment of every
position in order (`x[:, :] = A`, `x[:] = v`), and broadcasting and `fill!`
into full, in-order views and reshapes (`x[:, :] .= A`, `vec(x) .= v`), use
the same checks. They cover a view whose parent is ``x``, a reshape whose
parent is ``x``, and a view of such a reshape. Use `:` or integer vectors
matching each axis in order for a full two-index assignment or a full matrix
view. A full single-index assignment accepts `:`, the integers `1:length(x)`
in order, or `CartesianIndices(x)` with its shape. `copyto!(x, Q)` from an orthogonal factor is checked too.

Matrix values must have the axes of ``x``, with equal entries ``(r, c)`` and
``(c, r)`` and `NaN` equal to `NaN`. These comparisons precede conversion to
the element type of ``x``. Linear full writes reshape their values in
column-major order for the check; `x[:] = v` needs ``p^2`` values. If the
axes differ or symmetry fails, the checked writers throw a
`DimensionMismatch` or an `ArgumentError` before writing anything.
`Symmetric(A, :L)` stores the lower triangle of a matrix ``A`` that is not
symmetric, and `Symmetric(A, :U)` its upper triangle. Each entry of the lower
triangle is written once. Sources that are symmetric by their type, such as
another `SymmetricSparseIterate`, a `Symmetric` or a `Diagonal`, skip the
symmetry check. A broadcast whose arguments do not show that its value is
symmetric is first evaluated at both positions of each off-diagonal entry.
A function with side effects can therefore run more often and give a value
that is not symmetric.

With two array operands, `mul!` into ``x`` checks the unscaled product for
exact symmetry in a buffer promoted from the destination and operand types,
applies scalars and conversion at the final write, and leaves the product
out when α = 0, as LinearAlgebra does. For `mul!` by a number, the array
operand must have the axes of ``x``. The seven-argument
`mul!(x, tA, tB, A, B, α, β)`, when available, uses the same checks with
flags `'N'`, `'T'` and `'C'` for unchanged, transposed and adjoint operands;
other flags throw before writing. `fillband!(x, v, l, u)` checks that `v` is
real and writes a symmetric band (`l == -u`) through the lower triangle
once. An empty band (`l > u`) returns ``x`` unchanged.

Indexed array assignment, broadcasting and `fill!` into part of ``x``, such
as a row, a column, a block or a partial mask, throw an `ArgumentError`
before writing. Logical-mask indexed assignment is rejected even for an
all-true mask. `copyto!` with offsets or index ranges (including transpose
flags), `copy_transpose!` and `copy_adjoint!` with index ranges, `copyto!` from a
tuple, a generator or a number, and the indexed `axpy!` also throw before
writing. Out-of-bounds indices still throw a `BoundsError`.

`reverse!`, `sort!`, `accumulate!`, `kron!` and `copytrito!` (when available)
throw an `ArgumentError` before writing. This also rejects `reverse(x)`
with or without `dims`, `cumsum!`, `cumprod!`, and allocating `cumsum`,
`cumprod` and `accumulate` with `dims`. `circshift!` and `circshift` of a
nonempty ``x`` by a nonempty shift throw too.
`triu!(x, k)` returns ``x`` unchanged when ``k \le 1-p``, and `tril!(x, k)`
when ``k \ge p-1``. Both return ``x`` for every `k` when ``p=0``. Otherwise
they throw an `ArgumentError` before writing.

`lmul!`, `rmul!`, `ldiv!` and `rdiv!` by a matrix, a factorization, an
orthogonal factor or a rotation throw an `ArgumentError` before writing.
So does `ldiv!(x, A, B)` for such an operator ``A``. Reductions into ``x``
(`sum!`, `prod!`, `maximum!`, `minimum!`, `count!`, `all!`, `any!`,
`extrema!`, `findmax!` and `findmin!` with ``x`` as either destination),
in-place factorizations (`lu!`, `qr!`, `cholesky!` and `bunchkaufman!`),
`get!` and `sortperm!` also throw before writing. Nonempty bands of `fillband!` that are not symmetric,
`read!`, `readbytes!` and `hex2bytes!` also throw before writing; these
reading and decoding calls do not consume their input. Apply these
operations to `Matrix(x)`, or compute into a dense matrix and `copyto!` a
symmetric result.

`fill!`, and `rmul!`, `lmul!`, `rdiv!` and `ldiv!` by a number or a
`UniformScaling`, keep ``x`` symmetric. `replace` and `replace!` visit each
entry of the lower triangle once, and their `count` counts those entries,
column by column. `triu`, `tril` and `sparse` return a `SparseMatrixCSC`.
`sort(x; dims)` and `accumulate(op, x)` without `dims` keep their dense
results. Allocating `lu`, `qr`, `cholesky` and `bunchkaufman` use dense copies.

Other functions that write an arbitrary array position by position are not
checked; apply them to `Matrix(x)`. Loops of scalar writes over every
position stay unchecked. Other writes through a view or reshape, such as
`copyto!` into a view and scalar writes through a view, stay unchecked too. Broadcasts and fills through other
nested wrappers, such as `vec(view(x, :, :))`, are also unchecked.
Two-argument `copyto!(x, itr)` from other non-array iterators is unchecked.
A Base function can write part of ``x`` before reaching a check. `rand!`
and `randn!` give a symmetric matrix but draw each off-diagonal entry twice.
`shuffle!` and `shuffle` give wrong values; use `Matrix(x)` instead.

Its coordinates are the
``p(p+1)/2`` entries ``(r, c)``, ``r \ge c``, of the lower triangle, numbered
column by column:

```math
k = \texttt{tril\_index}(p, r, c) = \frac{(c - 1)(2p - c + 2)}{2} + (r - c + 1).
```

[`tril_cartesian`](@ref) inverts this by solving a quadratic equation. Its
floating square root is exact for ``p < 2^{24}``, far beyond any matrix that
fits in memory, and it switches to an integer square root beyond that.
[`coordinates`](@ref)`(x)` is a vector view of the coordinates.

**The coordinate step on a symmetric matrix.** A coordinate ``(r, c)`` with
``r \ne c`` moves two entries of ``\Theta`` together. Take a penalty that
counts both triangles, ``\lambda\sum_{i,j}|\Theta_{ij}|``, as `NormL1` does on
the matrix. Let ``G`` be the gradient of ``f`` over symmetric matrices, and
``a`` the curvature along the direction ``E_{rc} + E_{cr}``. Along the
coordinate,

```math
\varphi(t) = 2G_{rc}(t - t_0) + \frac{a}{2}(t - t_0)^2 + 2\lambda|t| + \text{const},
\qquad
t^+ = S_{2\lambda/a}\Big(t_0 - \frac{2G_{rc}}{a}\Big).
```

That is `prox_entry(NormL1(λ), t₀ - (2/a) G[r, c], i, 2/a)`. On the diagonal
the factors of 2 disappear and the step is ``1/a``. So, as the
[`KronQuadratic`](@ref) example of the D-trace loss shows, off-diagonal
coordinates take the step ``2/a`` where the diagonal takes ``1/a``.

### A read-only matrix view

[`matrixview`](@ref)`(x)` returns a live, read-only ``p \times p`` view of a
`SymmetricSparseIterate`: ordinary matrix indices, both triangles, no copy.
Pass it to code that only reads a matrix. Writes to `x` show through it, and
nothing can be written through it or through a wrapper derived from it.
`mul!(y, matrixview(x), v)` costs ``O(p + \mathrm{nactive}(x))``. Its linear
indices are matrix indices, not coordinates: for ``p = 3``,
`coordinates(x)[4]` is the variable at ``(2, 2)`` and `matrixview(x)[4]` is the
entry at ``(1, 2)``. [`activevalues`](@ref)`(x)` gives the values of the active
coordinates in the order of [`activecoordinates`](@ref)`(x)`; like that list,
take it again after a coordinate becomes active or after `dropzeros!`.

### Symmetric parameters

Pass exactly symmetric parameter arrays when you use a `SymmetricSparseIterate`.
It stores only the lower triangle, the positions ``(r, c)`` with ``r \ge c``,
so different weights at ``(r, c)`` and ``(c, r)`` would make the answer depend
on which triangle is read. A check that finds different values throws an
`ArgumentError`.

A *per-entry parameter*, such as `W[r, c]`, assigns a weight or bound to
`x[r, c]`. Positions `(r, c)` and `(c, r)` form a mirrored pair.
The package symmetrizes computed results, not your parameters.

This weighted ℓ1 step stores an off-diagonal entry explicitly so that the
proximal call checks its weight.

```@example symmetric_parameters
using ProximalBase, LinearAlgebra

p = 3
x = SymmetricSparseIterate(p)
x[2, 1] = 4.0
W = [1.0 2.0 1.0; 2.0 1.0 1.0; 1.0 1.0 1.0]
y = prox(NormL1(1.0, W), x, 1.0)
@assert y[2, 1] == y[1, 2] == 2.0
@assert Matrix(y) == [0.0 2.0 0.0; 2.0 0.0 0.0; 0.0 0.0 0.0]
Matrix(y)
```

Changing just the upper weight breaks that agreement.

```@example symmetric_parameters
Wbad = copy(W)
Wbad[1, 2] = 6.0
try
    prox(NormL1(1.0, Wbad), x, 1.0)
    error("expected asymmetric weights to be rejected")
catch e
    @assert e isa ArgumentError
    showerror(stdout, e)
    println()
end
```

The repairs have different meanings. `Symmetric(Wbad, :L)` keeps the lower
weight of 2 at both positions. Averaging sets both weights to 4, changing the
penalty.

```@example symmetric_parameters
Wlower = Symmetric(Wbad, :L)
Wmean = (Wbad + Wbad') / 2
@assert Wlower[1, 2] == Wlower[2, 1] == 2.0
@assert Wmean[1, 2] == Wmean[2, 1] == 4.0
ylower = prox(NormL1(1.0, Wlower), x, 1.0)
ymean = prox(NormL1(1.0, Wmean), x, 1.0)
@assert Matrix(ylower) == Matrix(y)
@assert iszero(Matrix(ymean))
(Matrix(ylower), Matrix(ymean))
```

Choose the repair to match your model. Averaging is not a general repair for
patterns, bounds or labels: their meaning and domain must remain valid.
The sum in `(W + W') / 2` can also overflow even when both entries are finite.

Equality is exact, with no tolerance: `NaN` matches `NaN`, `0.0` matches
`-0.0`, and `1` matches `1.0`. Constructors separately check the parameter's
domain, such as real element types and nonnegative weights.

The table lists arrays indexed by matrix entry. Where a function accepts a
vector of length ``p^2``, it uses column-major order: `(c - 1)p + r` for
`(r, c)`. Usual size and axes requirements still apply.

| Function or argument | Per-entry parameter |
|:---|:---|
| `NormL1`, `NormL1Nonneg`, `ElasticNet`, `CappedL1`, `LogPenalty`, `NormLq`, `Berhu`, `CheckLoss`, `EpsInsensitiveLoss` | weights `w` |
| `HingeLoss`, `SqrHingeLoss` | weights `w`, labels `y` |
| `BernoulliLoss` | weights `w`, data `b` |
| `Tilt`, `LinearFunction`, `Regularize` | coefficients or center `c` |
| `IndBox`, `IndBinary` | bounds `l`, `u` |
| `IndPattern` | pattern `P` |
| `SqrNormL2` | center `y` |
| `KLDivergence` | reference `q` |
| `Translate` | shift `b` |
| array `Precompose` | scale `a` |
| `IndHyperplane`, `IndHalfspace`, `IndHyperslab` | normal `a` |
| `KronQuadratic` | linear coefficient `C` |
| proximal call with an array step | step at each entry |

Design matrices and image data have different indexing and are not covered
by this rule. Neither are arrays indexed by groups, ranks or eigenvalues.

#### What each call checks

A stored entry is a lower-triangle coordinate retained by the iterate,
including one whose value has returned to zero. Calls that check only these
entries cannot detect asymmetry elsewhere, so success does not certify the
whole parameter array.

This proximal call succeeds while `(3, 1)` is absent, then fails once that
entry is stored.

```@example symmetric_parameters
Wlater = copy(W)
Wlater[1, 3] = 3.0
f = NormL1(1.0, Wlater)
z = SymmetricSparseIterate(3)
z[2, 1] = 4.0
@assert Matrix(prox(f, z, 1.0)) == Matrix(y)
z[3, 1] = 4.0
try
    prox(f, z, 1.0)
    error("expected the newly stored pair to be checked")
catch e
    @assert e isa ArgumentError
    showerror(stdout, e)
    println()
end
```

Built-in separable functions such as weighted `NormL1` can compute values
and proximal steps using only stored entries when they preserve zero:
`fᵢ(0) = 0` and the scalar proximal operator maps zero to zero.
Some wrappers retain this behavior; others compare every pair.

Some calls compare every mirrored pair, including positions the iterate does
not store. Others compare only pairs at stored entries because those are all
they read, as with weighted `NormL1` above. The table identifies which
argument must be a `SymmetricSparseIterate` for these checks to apply.

| Operation | Checked when this argument is a `SymmetricSparseIterate` | Which pairs are compared |
|:---|:---|:---|
| `f(x)` | Input `x` | Every pair, or only stored pairs as with weighted `NormL1` |
| [`prox!`](@ref), [`value_and_prox!`](@ref) | Output | Every pair, or pairs stored in either input or output as with weighted `NormL1` |
| [`gradient!`](@ref), [`value_and_gradient!`](@ref) | Output | Every pair |
| [`gradient_mapping!`](@ref) | Output | Both functions' parameters, using the gradient and proximal rules above |
| Both forms of [`prox_coordinate!`](@ref) | Iterate `x` | The pair at the chosen coordinate |
| Four-argument [`kkt_violation`](@ref) | Iterate `x` | The pair at the chosen coordinate |
| [`hessian_vec!`](@ref) | Point `x`, even with dense output | Every pair |
| [`quadratic_model`](@ref), [`quadratic_model!`](@ref) | Point `x` | Every pair |
| [`coordinate_lipschitz!`](@ref) | Output `L` | Every pair |
| Separable [`lla_weights!`](@ref), [`lla_weights`](@ref) | Input `x`, even with dense output | Every pair |
| [`ProximalBase.check_symmetric_parameters`](@ref) | Iterate `x` | Every declared pair, also inside built-in wrappers and `SeparableSum` blocks |

Comparing only stored pairs in a proximal call requires both input and output
to be symmetric iterates. An array of per-entry steps is checked at the same
pairs as the function's parameters. For tuple arguments, these rules apply
to each symmetric block.

For a `SymmetricSparseIterate` input `x`:

- `prox(f, x)` returns a `SymmetricSparseIterate`, so it checks like `prox!`.
- `gradient(f, x)` returns a dense matrix; symmetric `x` alone triggers no check.
- `value_and_gradient(f, x)` returns a value and a dense gradient matrix;
  symmetric `x` alone triggers no check.
- `gradient_mapping(f, g, x, γ)` returns a dense matrix;
  symmetric `x` alone triggers no check.
- `lla_weights(f, x)` returns a dense matrix and checks every parameter pair
  because it checks `x`.

`kkt_violation(f, g, x, k)` uses the symmetric part of the gradient at the
chosen coordinate. It keeps equal entries exactly; otherwise it computes
`g[r, c] / 2 + g[c, r] / 2`. Dividing first avoids overflow to `Inf` when
adding two large finite entries. `+Inf` at one position and `-Inf` at its
mirror have no symmetric part and cause an `ArgumentError` at the chosen
coordinate. The call does not search the rest of `g` for such pairs.

To validate every declared parameter pair before a solver starts, call
`ProximalBase.check_symmetric_parameters(f, x)`. It returns `true` on success.
It does not replace arrays, remember earlier checks, or check separately supplied
steps. Ordinary calls still check; later mutation can invalidate validation. See
[Separable functions](@ref extending-separable) for declarations and solver use.

#### The argument of a proximal map

The argument `x` of `prox!` and `value_and_prox!` is not a parameter, and it is
not checked. For a symmetric ``Z`` and any ``V``,

```math
\|Z - V\|_F^2 = \|Z - (V + V^\top)/2\|_F^2 + \|(V - V^\top)/2\|_F^2,
```

so the proximal point over symmetric matrices at ``V`` is the proximal point at its
symmetric part, for every function. When the output is a `SymmetricSparseIterate`
and `x` is a square matrix of the same size that is not symmetric by type
(`Matrix`, `SparseMatrixCSC`, `SparseIterate`, ...), every built-in `prox!` and
`value_and_prox!` reads `x` through its symmetric part, computed entry by entry as
`x[r, c] / 2 + x[c, r] / 2` (equal entries are kept exactly, and the sum cannot
overflow). A `SymmetricSparseIterate`, a `Symmetric`, a `Hermitian`, a `Diagonal`
or a `SymTridiagonal` is used as it is, at no extra cost, and `Symmetric(x, :L)`
still selects the lower triangle. `+Inf` at one position and `-Inf` at its mirror give
`NaN`, as their sum does. For a tuple output, as in a `SeparableSum`, each block is
treated this way, so no block fails before another is written.

```@example symmetric_parameters
V = [0.0 1.0; 3.0 0.0]                   # not symmetric
y = SymmetricSparseIterate(2)
prox!(y, NormL1(0.5), V, 1.0)
@assert y[2, 1] == 1.5                   # the symmetric part has 2 off the diagonal
prox!(y, NormL1(0.5), Symmetric(V, :L), 1.0)
@assert y[2, 1] == 2.5                   # the lower triangle has 3
```

The spectral functions (`Spectral`, `IndPSD`, `TracePSD`) are defined on symmetric
matrices, so they decompose the symmetric part of a matrix that is not symmetric by
type for a dense output as well, instead of reading its upper triangle.
`gradient!`, `value_and_gradient!` and `hessian_vec!` of `KronQuadratic` write the
symmetric part of the result into a `SymmetricSparseIterate` output, also when
the point is a dense matrix that is not symmetric.

Weights, groups and steps that are not symmetric are parameters, not the argument,
and still cause an `ArgumentError`.

#### When no check happens

- Raw entry calls (`value_entry`, `prox_entry`, `lla_weight`, scalar
  `coordinate_lipschitz` and three-argument `kkt_violation`) and `restrict`
  have no symmetric iterate from which to determine mirrored positions.
  Validate the original function before using them in a symmetric solver.
- Scalars and `nothing` have no pair of array positions to compare.
  Diagonal entries mirror themselves and need no comparison.
- Arrays symmetric by construction, such as `Symmetric(W)`, skip comparison
  of their entries. The wrapper defines equal mirrored values by reading one
  triangle; it does not verify agreement between the triangles of `W`.
- Dense arrays, `SparseIterate` and `AtomIterate` do not trigger this
  symmetric-parameter check. For example, `gradient!` into a dense matrix
  does not check, even when `x` is a `SymmetricSparseIterate`.
- An override of a public oracle replaces the generic check. Its author must
  call `ProximalBase.check_symmetric_parameters`; declaring parameters alone
  does not run the check.

#### Cost and remaining limitations

Checking a dense parameter array adds work. An exploratory measurement of
`prox!` with dense weights at ``p = 1000`` took 1.6868 ms against 1.3059 ms
before checking, about 1.3 times as long
([measurement](https://github.com/mlakolar/ProximalBase.jl/pull/329)).
`Symmetric(W)` skips pair comparisons, but still has overhead. Performance
work is tracked in
[#374](https://github.com/mlakolar/ProximalBase.jl/issues/374).

Parameter validation does not guarantee every symmetric operation works.
`KronQuadratic` checks `C`; its `A`, `B`, and `LogDetLoss`'s `S` have separate
symmetry policies under [#269](https://github.com/mlakolar/ProximalBase.jl/issues/269).
`SphereConstrained` already checks its inner arrays, but its geometry for
symmetric coordinates remains under
[#336](https://github.com/mlakolar/ProximalBase.jl/issues/336).
Group, cone and slice policies remain under
[#342](https://github.com/mlakolar/ProximalBase.jl/issues/342): group proximal
operators already require mirrored positions to belong to the same group,
and group KKT calls reject symmetric coordinates.

## AtomIterate

An [`AtomIterate`](@ref) is a vector whose coordinates are groups of entries,
partitioning ``1{:}p``. Its active groups play the role of the stored entries
of a `SparseIterate`, and [`atom`](@ref)`(x, k)` is a view of `x` at group
``k``. Writes through it follow the rule of `x[i] = v`: a nonzero entry
activates the group, and a zero never deactivates it.
Block coordinate descent for the group lasso computes the trial point
``u = x_{G_k} - \nabla_{G_k} f(x)/L_k`` for group ``k``, with
``L_k = \|X_{G_k}\|_2^2`` for least squares. It then takes the block step
`prox_coordinate!(x, g, k, u, 1/L_k)`, which is block soft thresholding and
activates the group only when it ends nonzero. `AtomIterate(p, g)` builds an
iterate that shares the groups of a group penalty `g`.

An iterate and a group penalty whose groups are equal but were built separately
also take the active-group paths. The first check compares the two layouts in
full; the iterate then remembers the penalty's group array, so later checks take
constant time. `copy` and `similar` keep what the iterate remembers, `copyto!`
keeps the destination's, and an iterate built from a restricted penalty starts
with nothing remembered.

Group arrays and the indices inside them must not be mutated after a penalty
or an iterate is built. This also applies to arrays retained by inner
constructors.

## Tracking an active set

The active coordinates form a stored working set, which can contain entries
or groups that have returned to zero. We keep them during a sweep so that
the set being traversed does not change.

| Operation | Meaning |
|:----------|:--------|
| [`ncoordinates`](@ref)`(x)` | Number of possible coordinates |
| [`nactive`](@ref)`(x)` | Number of stored active coordinates |
| [`activecoordinates`](@ref)`(x)` | Indices of the active coordinates |
| [`activevalues`](@ref)`(x)` | Writable view of their values, in the same order |
| [`isactive`](@ref)`(x, k)` | Whether coordinate `k` is active |
| [`activate!`](@ref)`(x, k)` | Add a coordinate to the active set |
| `dropzeros!(x)` | Remove stored zero entries or zero groups |

Use [`kkt_violation`](@ref) to test excluded zero coordinates and
[`restrict`](@ref) to construct a supported restricted problem. The
[Coordinate descent](@ref) chapter explains the full optimality checks that
must accompany these storage operations.
