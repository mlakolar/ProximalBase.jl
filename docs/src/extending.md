# Writing your own function

A function is a subtype of [`AbstractFunction`](@ref) that declares its traits
and implements the oracles they promise. Solvers written against the interface
then accept it like the built-in functions.

## Any function

| Trait | Implement | You get |
|:------|:----------|:--------|
| always | the value, `(f::MyF)(x)`, `Inf` outside the domain | |
| `isproximable` | [`prox!`](@ref)`(y, f, x, γ)`; `y` may be `x` | [`prox`](@ref), and [`value_and_prox!`](@ref) as `prox!` then `f(y)` |
| `issmooth` | [`value_and_gradient!`](@ref)`(g, f, x)` | [`gradient!`](@ref), [`gradient`](@ref), [`value_and_gradient`](@ref) |
| `isconvex` | | |

Override `value_and_prox!` when the proximal computation gives the value more
cheaply, and define [`lipschitz`](@ref) when a smooth function's constant is
known.

Values follow the rule in [Precision](@ref): `f(x)` and
`value_and_gradient!(g, f, x)` return `T = float(eltype(x))`, and
`value_and_prox!(y, f, x, γ)` returns `T` of `y`, whatever the precision of
the parameters. Convert a value computed with wider parameters to `T` before
returning it. The rule in [Overflow and underflow of values](@ref) is part of
the same contract: when an intermediate can leave the range of `T` although
the value is representable, check the ordinary result and recompute it in
scaled units. The traits are defined on the type:

```@example extending
using ProximalBase, LinearAlgebra

# the indicator of the box [-1, 1]ⁿ intersected with the hyperplane Σᵢ xᵢ = 0
struct CenteredBox <: AbstractFunction end
ProximalBase.isproximable(::Type{CenteredBox}) = true
ProximalBase.isconvex(::Type{CenteredBox}) = true
function (::CenteredBox)(x)
    T = float(eltype(x))
    all(abs.(x) .<= 1 + 1e-12) && abs(sum(x)) <= 1e-9 ? zero(T) : T(Inf)
end

# the projection: x = clamp.(v .- τ, -1, 1), with τ chosen by bisection so that Σᵢ xᵢ = 0
function ProximalBase.prox!(y::AbstractVector, ::CenteredBox, v::AbstractVector, ::Real)
    lo, hi = minimum(v) - 1, maximum(v) + 1
    for _ in 1:100
        τ = (lo + hi) / 2
        sum(t -> clamp(t - τ, -1, 1), v) > 0 ? (lo = τ) : (hi = τ)
    end
    y .= clamp.(v .- (lo + hi) / 2, -1, 1)
end

y = prox(CenteredBox(), [3.0, 0.2, -0.4, 0.1], 1.0)
(y, sum(y))
```

The projection uses the same idea as the projection onto the simplex: the
constraint ``\sum_i x_i = 0`` has a multiplier ``\tau``, and for a fixed
``\tau`` the box projection is separable. Here a bisection on ``\tau`` replaces
Condat's exact search.

## Separable functions

A separable function ``f(x) = \sum_i f_i(x_i)`` subtypes
[`SeparableFunction`](@ref) and implements two scalar functions:

- [`value_entry`](@ref)`(f, v, i)`, the value ``f_i(v)``;
- [`prox_entry`](@ref)`(f, v, i, γ)`, the proximal operator of ``\gamma f_i`` at
  ``v``.

Here ``i`` is the linear index of the entry, which lets parameters such as
weights depend on it. The value, [`prox!`](@ref) and
[`value_and_prox!`](@ref) then work on any array and on the three iterates, and
[`prox_coordinate!`](@ref) works on the iterates. Declare
[`iszeropreserving`](@ref) when ``f_i(0) = 0`` and the proximal operator maps
``0`` to ``0``. Then the value and the proximal operator on a sparse iterate
touch only its stored entries.

```@example extending
# Σᵢ wᵢ xᵢ² / 2: separable, with weights
struct WeightedSquare <: SeparableFunction
    w::Vector{Float64}
end
ProximalBase.value_entry(f::WeightedSquare, v, i) = f.w[i] * v^2 / 2
ProximalBase.prox_entry(f::WeightedSquare, v, i, γ) = v / (1 + γ * f.w[i])
ProximalBase.iszeropreserving(::WeightedSquare) = true
ProximalBase.isconvex(::Type{WeightedSquare}) = true
ProximalBase.check_parameters(f::WeightedSquare, x) =
  length(f.w) == length(x) || throw(DimensionMismatch("a parameter array has length $(length(f.w)) and the argument has length $(length(x))"))

f = WeightedSquare([1.0, 2.0, 4.0])
x = [1.0, 1.0, 1.0]
(f(x), prox(f, x, 0.5))
```

The generic methods add the entry values in the argument's precision, so
`f(Float32[1, 1, 1])` returns a `Float32` although the weights are `Float64`.
The optional [`ProximalBase.check_parameters`](@ref) method checks that the
weights fit the argument. The generic methods call it before writing any
output, so a mismatched argument leaves outputs and iterates untouched.
Before writing any output, `SeparableSum`, `SliceSum` and `FiberSum` check all
blocks, slices or fibers, including functions inside wrappers, against the
checks their own oracles run before writing. Earlier outputs can still be
written if a failure occurs during computation, if a check depends on the
step (such as `MoreauEnvelope`'s sum of its parameter and step), or if a
user-defined function's oracles reject an argument that its `check_parameters`
accepts. A few nested wrappers and structured matrices whose checks are not yet
complete, such as a box constraint inside another one, can also leave them written.

```@example extending
xs = SparseIterate(3)                          # a coordinate step on a sparse iterate
xs[2] = 1.0
prox_coordinate!(xs, f, 2, 0.5), nnz(xs)
```

On a `SymmetricSparseIterate`, the index ``i`` of entry ``(r, c)`` is
``(c - 1)p + r``. A user `SeparableFunction` must declare
[`ProximalBase.entry_parameters`](@ref), for example `(:w => f.w,)`, or `()`
when it has no per-entry arrays. Mirrored parameters must agree. The generic
oracles and built-in wrappers check this before writing. An override of a
public oracle should call [`ProximalBase.check_symmetric_parameters`](@ref)
itself. This check covers every pair, including unstored entries, and walks
built-in wrappers and `SeparableSum` tuples. It trusts the declaration to be
complete. Raw entry oracles and `restrict` have no iterate; solvers should
validate the original function before using them. Active-set solvers can also
use [`kkt_violation`](@ref) and [`restrict`](@ref).

## Testing a new function

[`ProximalBase.Testing`](@ref) has the checks that the package's own test
suite runs on every built-in function. With `symmetric = true`,
`test_separable` also validates the per-entry declaration and all mirrored
parameters. It does not construct asymmetric copies of user functions. The
checks are defined once the Test standard library is loaded, and they run
`@test` inside your `@testset`:

- [`test_prox`](@ref ProximalBase.Testing.test_prox): `prox`, `prox!` (also in
  place) and `value_and_prox!` agree, and the result is optimal (the
  subgradient inequality for a convex function, and no nearby point with a
  smaller proximal objective);
- [`test_gradient`](@ref ProximalBase.Testing.test_gradient): the gradient
  against central finite differences;
- [`test_separable`](@ref ProximalBase.Testing.test_separable): `prox_entry`
  against `prox!`, `value_entry` against the value, coordinate steps on the
  three iterates, and the O(nnz) paths when the function preserves zero;
- [`test_conjugate`](@ref ProximalBase.Testing.test_conjugate) and
  [`test_kkt`](@ref ProximalBase.Testing.test_kkt): Moreau's decomposition and
  the Fenchel–Young equality, and `kkt_violation` against the proximal operator;
- [`allocations`](@ref ProximalBase.Testing.allocations): the bytes a hot call
  allocates, measured inside a function after a warm-up call. The built-in
  allocation tests check these calls on Float32 and Float64 CPU arrays after
  warm-up. Check the methods and inputs you use on Julia 1.10 and on the
  current release, whose counts can differ.

```@example extending
using Test
const PT = ProximalBase.Testing
@testset "WeightedSquare" begin
    PT.test_traits(f; proximable = true, separable = true, convex = true)
    PT.test_prox(f, randn(3), 0.5)
    PT.test_separable(f, randn(3), 0.5)
    @test PT.allocations(prox!, similar(x), f, x, 0.5) == 0
end
nothing # hide
```

[`brute_force_prox`](@ref ProximalBase.Testing.brute_force_prox) is a
reference for a scalar proximal operator, convex or not.

## Running the package suite

From the package checkout, run `julia scripts/test.jl` after adding or changing
an implementation. This calls `Pkg.test()` with array bounds checking enabled
and installs the dependencies needed for the package tests. No clones of
other packages are required.
