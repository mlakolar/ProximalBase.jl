# Symmetric parameters

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
| [`prox!`](@ref), [`value_and_prox!`](@ref) | Output, or the point `x` with a dense output | Every pair, or pairs stored in either input or output as with weighted `NormL1` |
| [`gradient!`](@ref), [`value_and_gradient!`](@ref) | Output, or the point `x` with a dense output | Every pair |
| Both forms of [`prox_coordinate!`](@ref) | Iterate `x` | The pair at the chosen coordinate |
| Four-argument [`kkt_violation`](@ref) | Iterate `x` | The pair at the chosen coordinate |
| [`hessian_vec!`](@ref) | Point `x`, even with dense output; the output, even at a dense point | Every pair |
| [`quadratic_model`](@ref), [`quadratic_model!`](@ref) | Point `x` | Every pair |
| [`coordinate_lipschitz!`](@ref) | Output `L` | Every pair |
| Separable [`lla_weights!`](@ref), [`lla_weights`](@ref) | Input `x`, even with dense output | Every pair |
| [`ProximalBase.check_symmetric_parameters`](@ref) | Iterate `x` | Every declared pair, also inside built-in wrappers and `SeparableSum` blocks |

Comparing only stored pairs in a proximal call requires both input and output
to be symmetric iterates. For tuple arguments, these rules apply to each symmetric
block. A `SymmetricSparseIterate` output takes a scalar step only; an array step is an
`ArgumentError` (see [The multiplicity of an off-diagonal variable](@ref)), and
`gradient_mapping` and `gradient_mapping!` have no method for an owner: pass `Matrix(x)`.

For a `SymmetricSparseIterate` input `x`:

- `prox(f, x)` returns a `SymmetricSparseIterate`, so it checks like `prox!`.
- `gradient(f, x)` returns a dense matrix; the owner `x` is checked all the same, and
  an asymmetric parameter is an `ArgumentError` (`LinearFunction([1.0 2; 3 1])` at an
  owner point throws).
- `value_and_gradient(f, x)` returns a value and a dense gradient matrix, and checks
  the owner `x` in the same way.
- `lla_weights(f, x)` returns a dense matrix and checks every parameter pair
  because it checks `x`.

`kkt_violation(f, g, x, k)` uses the symmetric part of the gradient at the
chosen coordinate. It keeps equal entries exactly; otherwise it computes
`g[r, c] / 2 + g[c, r] / 2`. Dividing first avoids overflow to `Inf` when
adding two large finite entries. `+Inf` at one position and `-Inf` at its
mirror have no symmetric part and cause an `ArgumentError` at the chosen
coordinate. The call does not search the rest of `g` for such pairs. After the check of the real
element types, it checks that `k` is one of `1:ncoordinates(x)` (a `BoundsError`
otherwise) and then that `g` has `p^2` entries, as a matrix or flat (a `DimensionMismatch`
otherwise).

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
`value_and_prox!` reads `x` through its symmetric part. Each pair is averaged as the correctly rounded `(a + b) / 2`
when the sum is finite (this keeps subnormal digits), and as `a / 2 + b / 2` when
the sum overflows; equal entries are kept exactly, and the result is exactly
symmetric. A `SymmetricSparseIterate`, a `Symmetric`, a `Hermitian`, a `Diagonal`
or a `SymTridiagonal` is used as it is, at no extra cost, and `Symmetric(x, :L)`
still selects the lower triangle. `+Inf` at one position and `-Inf` at its mirror give
`NaN`, as their sum does. For a tuple output, as in a `SeparableSum`, each block is
treated this way at every depth of nesting, so no block fails before another is written.
The rule holds for an argument on a device too, which is read on the host.

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

Weights and groups that are not symmetric are parameters, not the argument, and still
cause an `ArgumentError`. `Symmetric(x, :L)` is a choice of the caller, who selects the
triangle to read; the package never picks one for a parameter.

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
  symmetric-parameter check. For example, `gradient!` into a dense matrix at a
  dense point does not check. A call checks as soon as one of its iterate arguments
  is a `SymmetricSparseIterate`, as the point (`gradient!` into a dense matrix at
  an owner point throws for an asymmetric parameter) or as the output.
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
`SphereConstrained` checks its inner arrays and measures the sphere in the
Frobenius norm: an off-diagonal pair that carries the proximal point of the
fallback takes ``r/\sqrt 2`` in each of its entries (see
[Separable functions with constraints](@ref)).
Group, cone and slice policies remain under
[#342](https://github.com/mlakolar/ProximalBase.jl/issues/342): group proximal
operators already require mirrored positions to belong to the same group,
and group KKT calls reject symmetric coordinates.

See also [SymmetricSparseIterate](symmetric_sparse_iterate.md) for the coordinate representation and [Developer Documentation](developer/index.md) for implementation and contract ownership.
