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

Store `prox_{γf}(x)` in `y`, as [`prox!`](@ref) does, and return `f(y)`, the
value at the result. Many functions get the value from the computation of the
proximal operator: a nuclear norm from the singular values that it thresholds,
an indicator function as `0`. This is then cheaper than `prox!` followed by
`f(y)`, and solvers that track the objective should call it.
"""
function value_and_prox!(y, f::AbstractFunction, x, γ::Real = 1)
  prox!(y, f, x, γ)
  _isindicator(f) ? zero(float(eltype(y))) : f(y)
end

# Internal traits: an indicator function is 0 at its projection, and the
# parameter arrays of a function (weights, bounds, centers) must have an entry
# for every entry of the argument, which _checkparams verifies in O(1).
_isindicator(::Type{<:AbstractFunction}) = false
_isindicator(f::AbstractFunction) = _isindicator(typeof(f))
_checkparams(::AbstractFunction, x) = true

####################################
#  Gradients
####################################

"""
    gradient!(g, f, x)

Store the gradient of `f` at `x` in `g` and return `g`.
"""
gradient!(g, f::AbstractFunction, x) = (value_and_gradient!(g, f, x); g)

"""
    value_and_gradient!(g, f, x)

Store the gradient of `f` at `x` in `g` and return `f(x)`. Usually cheaper than
calling `f(x)` and [`gradient!`](@ref) separately.
"""
function value_and_gradient! end

# a dense array for the gradient at x, also when x is a sparse iterate
_gradient_buffer(x::AbstractArray) = similar(Array{float(eltype(x))}, axes(x))

"""
    gradient(f, x)

Allocating version of [`gradient!`](@ref).
"""
gradient(f::AbstractFunction, x::AbstractArray) = gradient!(_gradient_buffer(x), f, x)

"""
    value_and_gradient(f, x)

Allocating version of [`value_and_gradient!`](@ref): returns `(f(x), ∇f(x))`.
"""
function value_and_gradient(f::AbstractFunction, x::AbstractArray)
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
  r, c = tril_cartesian(x.p, k)
  v = prox_entry(f, _getstored(x, k), (c - 1) * x.p + r, γ)
  _setstored!(x, v, k)
  v
end

# group k, entry by entry; the group becomes active if it ends nonzero
function prox_coordinate!(x::AtomIterate, f::AbstractFunction, k::Integer, γ::Real)
  isseparable(f) || _not_separable(f)
  _checkparams(f, x)
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
coordinates `S`, as a function of the same kind: an active-set solver minimizes
the restriction over the active coordinates `S`. At `x[S]` the restriction has
the value `f(x)` and the gradient `∇f(x)[S]`, except that [`SqrNormL2`](@ref)
with a center `y` drops the constant `Σⱼ∉S yⱼ²`.
"""
function restrict end
