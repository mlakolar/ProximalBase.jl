####################################
#
#  The function interface
#
####################################

"""
    AbstractFunction

Abstract supertype of the functions of ProximalBase: losses, penalties and
indicator functions. A function `f` is callable: `f(x)` is its value, `Inf`
outside its domain. Its traits say which oracles it provides:

| Trait | Oracles |
|:------|:--------|
| [`issmooth`](@ref) | [`gradient!`](@ref), [`value_and_gradient!`](@ref) |
| [`isproximable`](@ref) | [`prox!`](@ref), [`value_and_prox!`](@ref) |
| [`isseparable`](@ref) | [`prox_entry`](@ref), and [`prox_coordinate!`](@ref) on every iterate |
| [`hashessian`](@ref) | [`hessian_vec!`](@ref), [`quadratic_model`](@ref) |

A function can have several roles: [`SqrNormL2`](@ref), [`QuadraticFunction`](@ref),
[`LeastSquaresLoss`](@ref) and [`LogDetLoss`](@ref) are both smooth and
proximable. [`isconvex`](@ref) tells a solver whether convex theory applies.

To add a function, subtype `AbstractFunction`, make it callable, declare its
traits for its type, as in `ProximalBase.issmooth(::Type{<:MyLoss}) = true`, and
implement the oracles of those traits.
"""
abstract type AbstractFunction end

"""
    issmooth(f)

Whether `f` is differentiable on the interior of its domain, with
[`gradient!`](@ref) and [`value_and_gradient!`](@ref). The gradient need not be
globally Lipschitz: [`LogDetLoss`](@ref) is smooth on the positive definite
matrices, and its value is `Inf` elsewhere.
"""
issmooth(::Type{<:AbstractFunction}) = false
issmooth(f::AbstractFunction) = issmooth(typeof(f))

"""
    isproximable(f)

Whether `f` has an efficient proximal operator, computed by [`prox!`](@ref) and
[`value_and_prox!`](@ref).
"""
isproximable(::Type{<:AbstractFunction}) = false
isproximable(f::AbstractFunction) = isproximable(typeof(f))

"""
    isseparable(f)

Whether `f(x) = Σᵢ fᵢ(xᵢ)` is a sum over the entries of `x`. A separable `f` has
the scalar proximal operators [`prox_entry`](@ref), with which
[`prox_coordinate!`](@ref) updates a coordinate of any iterate, and
[`kkt_violation`](@ref) needs only the gradient.
"""
isseparable(::Type{<:AbstractFunction}) = false
isseparable(f::AbstractFunction) = isseparable(typeof(f))

"""
    iszeropreserving(f)

Whether the separable function `f` has `fᵢ(0) = 0` and
`prox_entry(f, 0, i, γ) = 0` for every entry `i` and every positive finite step
`γ`. Then sparse-iterate kernels may skip unstored zeros. This promises no
standalone value oracle. It is `false` for non-separable functions and unless
a function declares it. Wrappers forward it only where it holds for every
entry and every positive finite step.

The promise holds in exact arithmetic: for parameters or steps so extreme
that dense computations overflow (for example, a scale beyond
`floatmax(Float32)` with a `Float32` argument), paths over stored entries
return exact zeros where dense results can be `NaN`.

It is a property of the function, not of its type: the bounds of an
[`IndBox`](@ref) may exclude zero.
"""
iszeropreserving(::AbstractFunction) = false

"""
    isconvex(f)

Whether `f` is convex. For a nonconvex `f`, [`prox!`](@ref) returns one of the
minimizers that define the proximal operator, and accelerated methods such as
FISTA lose their guarantees.
"""
isconvex(::Type{<:AbstractFunction}) = false
isconvex(f::AbstractFunction) = isconvex(typeof(f))

####################################
#  Proximal operators
####################################

"""
    prox!(y, f, x, γ = 1)

Store the proximal operator

    prox_{γf}(x) = argmin_z  f(z) + ‖z - x‖² / (2γ)

in `y` and return `y`. `y` may be `x`. For a nonconvex `f` the result is one of
the minimizers.
"""
function prox! end
prox!(y, f::AbstractFunction, x) = prox!(y, f, x, 1)

"""
    prox(f, x, γ = 1)

Allocating version of [`prox!`](@ref).
"""
prox(f::AbstractFunction, x::AbstractArray, γ::Real = 1) = prox!(similar(x, float(eltype(x))), f, x, γ)
# a Symmetric or Hermitian wrapper does not accept off-diagonal writes: a dense result
prox(f::AbstractFunction, x::Union{Symmetric, Hermitian}, γ::Real = 1) =
  prox!(Matrix{float(eltype(x))}(undef, size(x)), f, x, γ)

"""
    value_and_prox!(y, f, x, γ = 1)

Store `prox_{γf}(x)` in `y`, as [`prox!`](@ref) does, and return the value at the
result. Many functions get the value from the computation of the proximal
operator: a nuclear norm from the singular values that it thresholds. For the
package's indicator functions, return `0` without checking membership. This
equals `f(y)` when the projection is exact, but is not a feasibility certificate
for an inexact projection such as [`IndIntersection`](@ref). Computing the value this
way can be cheaper than `prox!` followed by `f(y)`. The conjugates that
[`Conjugate`](@ref) lists as indicators also return `0`; another conjugate whose
value uses the Fenchel equality can return a small nonzero value even when it
represents an indicator.
"""
function value_and_prox!(y, f::AbstractFunction, x, γ::Real = 1)
  _checkwrite(value_and_prox!, f, y, x)
  prox!(y, f, x, γ)
  _isindicator(f) ? zero(_valtype(y)) : _value_at(f(y), y)
end

"""
    ProximalBase.check_parameters(f, x)

Check that the parameters of `f` fit the argument `x`. Throw `DimensionMismatch`
when they do not. The default for an [`AbstractFunction`](@ref) returns
`nothing`. Define this method for a function with parameter arrays; the
generic kernels call it before writing any output, including `x` when the
output aliases it.

For a [`SeparableFunction`](@ref) `WeightedSquare` with a weight vector `w`:

```julia
function ProximalBase.check_parameters(f::WeightedSquare, x)
  length(f.w) == length(x) || throw(DimensionMismatch("a parameter array has length \$(length(f.w)) and the argument has length \$(length(x))"))
  nothing
end
```
"""
check_parameters(::AbstractFunction, x) = nothing

# An indicator is zero at its projection. Built-ins check their parameter
# arrays privately; the fallback calls the public extension hook before writes.
_isindicator(::Type{<:AbstractFunction}) = false
_isindicator(f::AbstractFunction) = _isindicator(typeof(f))
_checkparams(f::AbstractFunction, x) = (check_parameters(f, x); true)
_argument_type(::Type{<:AbstractFunction}) = Any
_checkargument(f, x) = x isa _argument_type(typeof(f)) || throw(MethodError(f, (x,)))
function _checkwrite(op, f, y, x)
  if op === value_and_prox! || op === gradient!
    _checkparams(f, x)
    _checkargument(f, x)
  else
    _checkargument(f, x)
    _checkparams(f, x)
  end
  true
end
_checkvalue(f, x) = _checkparams(f, x)
_checkconjugate(f, y) = true

####################################
#  Gradients
####################################

"""
    gradient!(g, f, x)

Store the gradient of `f` at `x` in `g` and return `g`.
"""
gradient!(g, f::AbstractFunction, x) = (_checkwrite(gradient!, f, g, x); value_and_gradient!(g, f, x); g)

"""
    value_and_gradient!(g, f, x)

Store the gradient of `f` at `x` in `g` and return `f(x)`. Usually cheaper than
calling `f(x)` and [`gradient!`](@ref) separately.
"""
function value_and_gradient! end

# a dense array for the gradient at x, also when x is a sparse iterate
_gradient_buffer(x::AbstractArray) = similar(Array{float(eltype(x))}, axes(x))
_gradient_buffer(x::Tuple) = map(_gradient_buffer, x)

_prox_buffer(x::AbstractArray) = similar(x, float(eltype(x)))
_prox_buffer(x::Union{Symmetric, Hermitian}) = Matrix{float(eltype(x))}(undef, size(x))
_prox_buffer(x::Tuple) = map(_prox_buffer, x)

function _checksize(y::Tuple, x::Tuple)
  length(y) == length(x) || _block_size_error(length(y), length(x))
  map(_checksize, y, x)
  true
end
@noinline _block_size_error(n, m) = throw(DimensionMismatch("the output has $n blocks and the input $m"))

prox(f::AbstractFunction, x::Tuple, γ::Real = 1) = prox!(_prox_buffer(x), f, x, γ)

"""
    gradient(f, x)

Allocating version of [`gradient!`](@ref).
"""
gradient(f::AbstractFunction, x::AbstractArray) = gradient!(_gradient_buffer(x), f, x)
gradient(f::AbstractFunction, x::Tuple) = gradient!(_gradient_buffer(x), f, x)

"""
    value_and_gradient(f, x)

Allocating version of [`value_and_gradient!`](@ref): returns `(f(x), ∇f(x))`.
"""
function value_and_gradient(f::AbstractFunction, x::Union{AbstractArray, Tuple})
  g = _gradient_buffer(x)
  v = value_and_gradient!(g, f, x)
  v, g
end

####################################
#  Coordinate descent
####################################

"""
    prox_entry(f, v, i, γ)

For a separable `f(x) = Σᵢ fᵢ(xᵢ)` (see [`isseparable`](@ref)), the proximal
operator of `γfᵢ` at the scalar `v`: the value that a coordinate-descent step
gives entry `i` when the gradient step lands at `v`. `i` is a linear index of
the array, so that weights and bounds may depend on the entry; for a
[`SymmetricSparseIterate`](@ref), entry `(r, c)` has index `(c - 1)p + r`.
It touches no iterate: [`prox_coordinate!`](@ref) updates a coordinate of one in
place.
"""
function prox_entry end

"""
    prox_coordinate!(x, f, k, γ)

Apply the proximal operator of `γf` to coordinate `k` of the iterate `x` in
place, as in one step of coordinate descent, and return the new value of the
coordinate. The coordinates of a [`SparseIterate`](@ref) are its entries, those
of a [`SymmetricSparseIterate`](@ref) the entries of its lower triangle, and
those of an [`AtomIterate`](@ref) its groups, for which `prox_coordinate!`
returns a view of the group (see [`ncoordinates`](@ref)).

A separable `f` updates any coordinate through [`prox_entry`](@ref); a group
penalty such as [`GroupNormL2`](@ref) updates a group of an `AtomIterate` that
shares its groups. A coordinate that stays zero does not become stored or
active; one that becomes nonzero does.
"""
function prox_coordinate! end

@noinline _not_separable(f) =
  throw(ArgumentError("$(typeof(f)) is not separable: prox_coordinate! needs a separable function or a group penalty on matching groups"))

function prox_coordinate!(x::SparseIterate, f::AbstractFunction, k::Integer, γ::Real)
  isseparable(f) || _not_separable(f)
  _checkparams(f, x)
  v = prox_entry(f, x[k], k, γ)
  x[k] = v
  v
end

function prox_coordinate!(x::SymmetricSparseIterate, f::AbstractFunction, k::Integer, γ::Real)
  isseparable(f) || _not_separable(f)
  _checkparams(f, x)
  checkbounds(coordinates(x), k)
  r, c = tril_cartesian(x.p, k)
  v = prox_entry(f, _getstored(x, k), (c - 1) * x.p + r, γ)
  _setstored!(x, v, k)
  v
end

# group k, entry by entry; the group becomes active if it ends nonzero
function prox_coordinate!(x::AtomIterate, f::AbstractFunction, k::Integer, γ::Real)
  isseparable(f) || _not_separable(f)
  _checkparams(f, x)
  checkbounds(x.groups, k)
  nonzero = false
  @inbounds for i in x.groups[k]
    v = prox_entry(f, x.storage[i], i, γ)
    x.storage[i] = v
    nonzero |= !iszero(v)
  end
  nonzero && activate!(x, k)
  atom(x, k)
end

####################################
#  Optimality and active sets
####################################

"""
    kkt_violation(f, g, i)
    kkt_violation(f, g, x, k)

How much a coordinate at zero violates the optimality conditions of `h + f`,
where `g = ∇h(x)` is the gradient of the smooth part: a positive value means
that moving the coordinate away from zero decreases the objective, and a value
`≤ 0` that zero is optimal. For [`NormL1`](@ref) it is `|gᵢ| - λ0 wᵢ`.

For a separable `f` the condition involves only `gᵢ`, and the first form
suffices. The second form takes a coordinate `k` of the point `x`: for a
[`GroupNormL2`](@ref), the condition of an index involves the rest of its group,
and on an [`AtomIterate`](@ref) `k` is a group, whose violation is
`‖g[Gₖ]‖₂ - λ0 wₖ` (for a separable `f`, the largest over the group).
"""
function kkt_violation end

# for a separable function, the optimality of a coordinate at zero does not
# depend on the rest of x
kkt_violation(f::AbstractFunction, g::AbstractArray, ::AbstractArray, i::Integer) = kkt_violation(f, g, i)
function kkt_violation(f::AbstractFunction, g::AbstractArray, x::SymmetricSparseIterate, k::Integer)
  r, c = tril_cartesian(x.p, k)
  kkt_violation(f, g, (c - 1) * x.p + r)
end
# a group at zero violates optimality as much as its worst index
kkt_violation(f::AbstractFunction, g::AbstractArray, x::AtomIterate, k::Integer) =
  maximum(i -> kkt_violation(f, g, i), x.groups[k])

"""
    restrict(f, S)

The function of `x[S]` that `f` becomes on the vectors `x` supported on the
coordinates `S`, where supported: an active-set solver minimizes the restriction
over those coordinates. The restriction preserves the objective up to a finite
additive constant and, for a smooth function, has gradient `∇f(x)[S]`. Centered
[`SqrNormL2`](@ref), [`Regularize`](@ref), separable [`Translate`](@ref),
[`MoreauEnvelope`](@ref) and [`SqrDistL2`](@ref) may drop terms belonging to
excluded coordinates. These constants do not change the minimizer; evaluate the
original function when comparing full objective values.

The dropped terms must be finite at zero, or at `-b` under `Translate`;
otherwise every vector supported on `S` is infeasible, and `restrict` throws an
`ArgumentError`. So `restrict` of an [`IndBox`](@ref) throws when an excluded
coordinate's bounds exclude zero. `restrict` of [`NegLog`](@ref), or of an
`IndBox` with scalar bounds that exclude zero, throws for every `S`, even one
that drops nothing, because it does not know the dimension. Wrappers that
evaluate the dropped terms elsewhere, such as `Translate`, `MoreauEnvelope` and
`SqrDistL2`, can still restrict these functions. A nonseparable `Translate`
requires its translation to vanish outside `S`; otherwise restricting its
underlying function would change more than a constant, and the operation throws.
"""
function restrict end
# the restriction without the check that the dropped terms are finite at zero,
# for wrappers that evaluate those terms elsewhere
_restrict_unchecked(f::AbstractFunction, S::AbstractVector{<:Integer}) = restrict(f, S)
# the coordinates of 1:n outside S, through a mask so that ranges, duplicates
# and Boolean masks select as indexing does
function _excluded_coordinates(n::Integer, S::AbstractVector{<:Integer})
  selected = falses(n)
  selected[S] .= true
  findall(!, selected)
end

"""
    gradient_mapping!(G, f, g, x, γ)
    gradient_mapping(f, g, x, γ)

The gradient mapping `G = (x - prox_{γg}(x - γ∇f(x))) / γ` of the problem
`minimize f(x) + g(x)`, for a smooth `f` and a proximable `g`. When `f` and `g`
are convex it is zero exactly at the minimizers, for every `γ > 0`; with
`g = 0` it is `∇f(x)`. Its norm is a stopping rule that needs neither the
optimal value nor the structure of `g`, and a proximal gradient step from `x` is
`x - γG`. `G` must not be `x`; `gradient_mapping!` allocates nothing beyond
what `gradient!` and `prox!` do.
"""
function gradient_mapping!(G::AbstractArray, f::AbstractFunction, g::AbstractFunction, x::AbstractArray, γ::Real)
  G === x && throw(ArgumentError("G must not be x"))
  _checksize(G, x)
  _checkparams(f, x)
  _checkparams(g, x)
  gradient!(G, f, x)
  G .= x .- γ .* G
  prox!(G, g, G, γ)
  G .= (x .- G) ./ γ
end

gradient_mapping(f::AbstractFunction, g::AbstractFunction, x::AbstractArray, γ::Real) =
  gradient_mapping!(_gradient_buffer(x), f, g, x, γ)
@doc (@doc gradient_mapping!) gradient_mapping

"""
    lipschitz(f)

A Lipschitz constant `L` of the gradient of the smooth function `f`, so that a
gradient or proximal gradient step with step `1/L` decreases the objective.
It is exact for the closed forms (`SqrNormL2(λ)` gives `λ`). Dense matrices use
a full spectral calculation, such as `‖X‖₂²` for `LeastSquaresLoss`. Sparse and
other indexed matrices use the conservative bound `‖X‖₁ ‖X‖∞` for `‖X‖₂²`,
without forming a Gram matrix or densifying the input. Weighted losses apply
this calculation to `Diagonal(sqrt.(weights)) * X`.

The full dense calculation can cost more than an iterative estimate. For large
problems, [`opnorm2`](@ref) supplies a faster estimate with matrix products, but
it must be paired with backtracking or an independent bound when selecting a
step: an estimate can miss a leading eigenspace. `lipschitz` is `Inf` for a
smooth function whose gradient is not globally Lipschitz, such as `LogDetLoss`
and `PoissonLoss`, which call for a line search, and an error for a function
that is not smooth. Sums add the constants, `Scaled` scales them, and
`SeparableSum` takes their maximum.
"""
lipschitz(f::AbstractFunction) =
  throw(ArgumentError(issmooth(f) ? "no Lipschitz constant is known for the gradient of $(nameof(typeof(f)))" :
                                    "$(nameof(typeof(f))) is not smooth"))
