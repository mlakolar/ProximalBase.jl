####################################
#
#  Checks for authors of new functions
#
####################################

"""
    ProximalBase.Testing

The checks that ProximalBase's own test suite runs on every function, for
authors of new ones: a [`SeparableFunction`](@ref) of your own, or the
functions that packages build on ProximalBase.

| Check | What it tests |
|:------|:--------------|
| [`test_prox`](@ref ProximalBase.Testing.test_prox) | the proximal operator: its forms agree, and the result is optimal |
| [`test_gradient`](@ref ProximalBase.Testing.test_gradient) | the gradient against finite differences |
| [`test_separable`](@ref ProximalBase.Testing.test_separable) | the scalar proximal operators against `prox!`, on every iterate |
| [`test_conjugate`](@ref ProximalBase.Testing.test_conjugate) | Moreau's decomposition and the Fenchel–Young equality |
| [`test_kkt`](@ref ProximalBase.Testing.test_kkt) | `kkt_violation` against the proximal operator |
| [`test_iterate`](@ref ProximalBase.Testing.test_iterate) | the bookkeeping of an iterate against its entries |
| [`test_traits`](@ref ProximalBase.Testing.test_traits) | the traits, on the instance and on the type |
| [`allocations`](@ref ProximalBase.Testing.allocations) | the bytes a call allocates, as solvers make it |
| [`brute_force_prox`](@ref ProximalBase.Testing.brute_force_prox) | a reference for scalar proximal operators |

The checks run `@test` inside the caller's `@testset`, so a failure names the
failing condition and its values. They are defined by a package extension on
the Test standard library: load it with `using Test`. Random points come from
the global random number generator, or from `rng`.

```julia
using Test, ProximalBase
const PT = ProximalBase.Testing
@testset "my function" begin
    f = MyPenalty(0.5)
    x = randn(10)
    PT.test_traits(f; proximable = true, separable = true, convex = true)
    PT.test_prox(f, x, 0.7)
    PT.test_separable(f, x, 0.7)
    PT.test_kkt(f, x, 0.7)
    @test PT.allocations(prox!, similar(x), f, x, 0.7) == 0
end
```
"""
module Testing

using ..ProximalBase

# `public` does not parse on Julia 1.10.
VERSION >= v"1.11" && Core.eval(@__MODULE__, Expr(:public, :test_prox, :test_gradient, :test_separable,
  :test_conjugate, :test_kkt, :test_iterate, :test_traits, :allocations, :brute_force_prox))

"""
    test_prox(f, x, γ; rtol = 1e-8, atol = 1e-10, local_points = true, symmetric = false, rng = nothing)

Test the proximal operator of `f` at `x` with step `γ`:

- `prox`, `prox!` and [`value_and_prox!`](@ref) agree, `prox!` returns its
  output, and the value is `f` at the result;
- `prox!` and `value_and_prox!` with the output as the input (`y === x`) give
  the same point, and `value_and_prox!` the same value, as with a separate output;
- the result `p` is optimal: no nearby point, nor the proximal point of
  another input, has a smaller proximal objective
  `f(z) + ‖z - x‖²/(2γ)`, and for a convex `f` the subgradient inequality
  `f(z) ≥ f(p) + ⟨(x - p)/γ, z - p⟩` holds at those points.

With `symmetric = true` the perturbations are symmetric matrices, for the
functions of symmetric matrices; with `local_points = false` only proximal
points of other inputs are compared (for functions whose domain is a thin set).
Returns `p`.
"""
function test_prox end

"""
    test_gradient(f, x; h = nothing, rtol = nothing, atol = nothing, directions = 3, rng = nothing)

Test the gradient of the smooth function `f` at `x` against central finite
differences along random directions, and [`gradient!`](@ref),
[`gradient`](@ref), [`value_and_gradient!`](@ref) and
[`value_and_gradient`](@ref) against each other and against `f(x)`. Returns the
gradient.

For Float64, the defaults are `h = 1e-6`, `rtol = 1e-5` and
`atol = 1e-6`. For other precisions they are multiplied by
`c = cbrt(eps(float(eltype(x))) / eps(Float64))`. For Float32 this gives
`h ≈ 8.1e-4`, `rtol ≈ 8.1e-3` and `atol ≈ 8.1e-4`. The absolute tolerance is
the larger of `atol * max(1, abs(fd))` and the finite difference's rounding
error allowance `8 eps(R) max|f(x ± hd)| / (2h)`, where `fd` is the finite
difference and `R = float(eltype(x))`. This allowance also applies when
`atol` is supplied, so a loss with a large value relative to its directional
derivative does not fail in Float32. Supplied keywords override the defaults.
Directions and perturbed points keep the floating element type of `x`.

In Float32, a small gradient error can go undetected when `|f|` is large
relative to the directional derivatives, because the comparison allows the
finite difference's rounding error. Increasing `directions` raises the chance
of detecting it.

**Tuples.** `x` may be a tuple of arrays or nested tuples (as for `SeparableSum`),
of different shapes but one floating type: `float(eltype(block))` must agree for
all blocks, so mixing Float32 and Float64 blocks throws an `ArgumentError`
before `f` is called. Model data and parameters may have other types. One
random direction, an array per block drawn from `rng` in block order, moves all
blocks; the difference is compared with the sum of the blockwise inner
products, with the defaults of an array of that type. Scalars and empty tuples
are rejected.
"""
function test_gradient end

"""
    test_separable(f, x, γ; groups = nothing, symmetric = false, rng = nothing)

Test a separable `f` at the vector or matrix `x` with step `γ`:

- [`prox_block`](@ref) agrees with `prox!` entry by entry, and for a
  [`SeparableFunction`](@ref) the terms [`value_entry`](@ref) sum to `f(x)`;
- [`prox_block`](@ref) followed by indexed writes to a [`SparseIterate`](@ref)
  (a copy of `x` with some entries set to zero) gives the proximal point;
- when [`iszeropreserving`](@ref)`(f)`, zero is fixed by every scalar proximal
  operator, and `prox!` on a `SparseIterate` agrees with the dense result;
- for a zero-preserving `SeparableFunction`, every `value_entry` at zero is 0;
- for any zero-preserving `f` with a standalone value, `f(0) == 0` exactly, and
  the value and [`value_and_prox!`](@ref) on a `SparseIterate` agree with the
  dense results (value and, into a fresh destination, written point). `f(0) == 0` is necessary for the per-entry contract
  ``f_i(0) = 0`` but not sufficient: terms could cancel, so only a
  `SeparableFunction` gets the per-entry check. A function is asked for a value only when the
  package's private `_has_value` is true and `f` has a method for the array.
  `_has_value` is false for a [`Conjugate`](@ref) whose value is not
  implemented, and for the package's own wrappers (`Scaled`, `Translate`,
  sums and similar) around one. It does not see inside a user-defined wrapper:
  that wrapper is asked for a value whenever it has a method, and skipped when
  it has none. The checks are independent. `f(0) == 0` needs a method for the
  dense zero. The `SparseIterate` checks need a method for the iterate and for
  the dense proximal point, so a value method for dense vectors only is tested
  at zero alone. When `f` has a value method for some arrays but none for the
  array at hand, the checks that need it are skipped and one `@info` message
  names the skipped checks and the rejected array type; a function without a
  value method is skipped silently. A value method that
  exists is called without a `try`, so an error in it is reported as an error;
- with `groups`, scalar `prox_block` calls on the entries of each group of a [`BlockIterate`](@ref)
  gives the proximal point;
- with `symmetric = true` and a square `x`, `prox_block` at the linear entry index of each
  coordinate of a [`SymmetricSparseIterate`](@ref) gives the proximal point of
  the symmetric part of `x`, which needs parameters that are equal at `(r, c)`
  and `(c, r)`.

For each iterate, also write the pure scalar results into an iterate with nothing stored or active, at the same input coordinates.
It gives the same dense proximal point, stores or activates no zero results,
and passes [`test_iterate`](@ref).
"""
function test_separable end

"""
    test_conjugate(f, x, γ; conjugate = Conjugate(f), rtol = 1e-8)

Test the convex `f` against its conjugate at `x` with step `γ`:

- Moreau's decomposition `x = prox_{γf}(x) + γ prox_{f*/γ}(x/γ)`, which tests
  a conjugate given with its own proximal operator (for `Conjugate(f)` it holds
  by construction);
- when the value of the conjugate is known, the Fenchel–Young equality
  `f(p) + f*(y) = ⟨p, y⟩` at `p = prox_{γf}(x)` and its subgradient
  `y = (x - p)/γ`, and the inequality `f(z) + f*(y) ≥ ⟨z, y⟩` at other points.
"""
function test_conjugate end

"""
    test_kkt(f, x, γ; trials = 5, rng = nothing)

Test [`kkt_violation`](@ref) against the proximal operator: from zero, a
proximal gradient step with a random gradient `g` keeps a coordinate at zero
exactly when its violation is `≤ 0`. The coordinates are the entries of `x`
for a separable `f`, and the groups for a [`BlockIterate`](@ref) `x` and a
group penalty. For a nonconvex `f`, take `γ` below its convexity limit, where
the scalar proximal operator at zero is the local one.
"""
function test_kkt end

"""
    test_iterate(x)

Test the bookkeeping of a [`BlockIterate`](@ref), [`SparseIterate`](@ref) or
[`SymmetricSparseIterate`](@ref) against its entries. Check the structural
lengths, the active or stored count, and the maps in both directions. For an
`BlockIterate`, also check the group indices and that inactive groups hold only
zeros.

Active groups and stored coordinates holding zeros are valid;
[`dropzeros!`](@ref) removes them. This check neither changes nor repairs `x`.
Returns `x`.
"""
function test_iterate end

"""
    test_traits(f; smooth = false, proximable = false, separable = false, convex = false)

Test the traits of `f`, on the instance and on its type.
"""
function test_traits end

"""
    allocations(f, args...)

The bytes allocated by `f(args...)`, measured inside a function after warming
the same measured entry, as a solver makes the call:
`allocations(prox!, y, f, x, γ) == 0` is the contract of the hot calls.
Counts differ between Julia versions; test on all
that you support.
"""
function allocations end

"""
    brute_force_prox(fᵢ, v, γ; lo, hi, n = 200_001)

The minimizer of `fᵢ(t) + (t - v)²/(2γ)` for a scalar function `fᵢ`, over a
fine grid on `[lo, hi]` refined by golden-section search around the best grid
point, and the minimum: `(t, value)`. A reference for scalar proximal
operators, nonconvex ones included.
"""
function brute_force_prox end

end
