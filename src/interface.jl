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
| [`isseparable`](@ref) | [`prox_block`](@ref) for scalar entries |
| [`hashessian`](@ref) | [`hessian_vec!`](@ref), [`quadratic_model`](@ref) |

For built-in functions, caller-supplied per-entry arrays must be exactly
symmetric at the pairs read when a proximal operator or gradient writes a
[`SymmetricSparseIterate`](@ref), or when a value, coordinate or second-order
oracle takes one. A value checks the pairs its kernel reads; kernels restricted
to stored entries check only stored pairs. Computed gradients in
[`kkt_violation`](@ref) use their symmetric part.

Checks that depend only on types, shapes and parameters run before any output
write; failures that depend on the step, argument values or computation may
leave outputs, and inputs that alias them, changed.

A function can have several roles: [`SqrNormL2`](@ref), [`QuadraticFunction`](@ref),
[`LeastSquaresLoss`](@ref) and [`LogDetLoss`](@ref) are both smooth and
proximable. [`isconvex`](@ref) tells a solver whether convex theory applies.

To add a function, subtype `AbstractFunction`, make it callable, declare its
traits for its type, as in `ProximalBase.issmooth(::Type{<:MyLoss}) = true`, and
implement the oracles of those traits.

See [Symmetric parameters](@ref) for which arrays must be symmetric and which calls check them.
"""
abstract type AbstractFunction end

# Where the oracles of the families that support it take either an array or the owner of a
# symmetric matrix. The owner is not an array: a method that is not written with this type, or with
# the owner, does not accept it.
const _ArrayOrOwner = Union{AbstractArray, SymmetricSparseIterate}

# A matrix, or the owner of one: the destination or the point of the oracles of the families that
# are defined on matrices and write whole results (a spectral function, a set of matrices)
const _MatrixOrOwner = Union{AbstractMatrix, SymmetricSparseIterate}

# the number of entries of the array, or of the matrix that the owner represents
_nentries(x::AbstractArray) = length(x)
_nentries(x::SymmetricSparseIterate) = x.p^2

function value_and_prox! end
function gradient! end
function hessian_vec! end
function quadratic_model end
function quadratic_model! end
function coordinate_lipschitz! end

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
the scalar proximal operators [`prox_block`](@ref), and
[`kkt_violation`](@ref) needs only the gradient.
"""
isseparable(::Type{<:AbstractFunction}) = false
isseparable(f::AbstractFunction) = isseparable(typeof(f))

"""
    iszeropreserving(f)

Whether the separable function `f` has `fᵢ(0) = 0` and
`prox_block(f, 0, i, γ) = 0` for every entry `i` and every positive finite step
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

in `y` and return `y`. For a nonconvex `f` the result is one of the minimizers.

The output may be the argument itself: `prox!(x, f, x, γ)` works for every
function. A [`SymmetricSparseIterate`](@ref) output takes a scalar step `γ`, the Frobenius
proximal operator of the symmetric matrix; an array or per-group step is an `ArgumentError`
before anything is written (use `Matrix(x)` or `matrixview(x)` with a dense output). Any other sharing of memory between `y` and `x` is not supported and
is not detected, even when the two cover the same entries: a call such as
`prox!(view(x, :), f, x, γ)`, or with a shifted or reversed view of `x`, can
return wrong values without an error. Copy the argument first, as in
`prox!(y, f, copy(x), γ)`. See O7 in
[Agreement between oracles](@ref numerics-oracles).
"""
function prox! end
prox!(y, f::AbstractFunction, x) = prox!(y, f, x, 1)

"""
    prox(f, x, γ = 1)

Allocating version of [`prox!`](@ref). For a [`SymmetricSparseIterate`](@ref) `x` the result is an
independent `SymmetricSparseIterate` of the float element type, and `γ` is a scalar.
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
package's indicator functions, return `0` without checking membership, except
where the stored output has a NaN that the value depends on, which gives NaN, as
for [`IndBallL0`](@ref), [`IndBallGroupL0`](@ref), [`IndBallRank`](@ref) and
`SingularValue(IndBallL0(r))` (the last two at a non-finite input that needs a
decomposition). `IndBallRank` with `r ≥ minimum(size(X))` copies its input and
returns the value of the copy: `Inf` at an infinite entry, and `0` when the only
non-finite entries are NaN. The other spectral indicators return `NaN` at an
input with a `NaN` entry and `Inf` at another non-finite input (E2). This equals `f(y)` when the projection is exact, but is
not a feasibility certificate
for an inexact projection such as [`IndIntersection`](@ref). Computing the value this
way can be cheaper than `prox!` followed by `f(y)`. The conjugates that
[`Conjugate`](@ref) lists as indicators follow the same convention; another conjugate whose
value uses the Fenchel equality can return a small nonzero value even when it
represents an indicator.

The output may be the argument itself, as for [`prox!`](@ref):
`value_and_prox!(x, f, x, γ)` stores the same point and returns the same value
as with a separate output.
Any other sharing of memory between `y` and `x` is not supported and is not
detected.
"""
value_and_prox!(y, f::AbstractFunction, x, γ::Real = 1) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(value_and_prox!, f, y, x, γ); _value_and_prox_body!(y, f, x, γ))
_value_and_prox_unchecked!(y, f::AbstractFunction, x, γ::Real = 1) =
  _isbuiltin(f) ? _value_and_prox_body!(y, f, x, γ) : value_and_prox!(y, f, x, γ)
function _value_and_prox_body!(y, f::AbstractFunction, x, γ::Real = 1)
  _checkwrite(value_and_prox!, f, y, x)
  _prox_unchecked!(y, f, x, γ)
  _isindicator(f) ? zero(_valtype(y)) : _value_at(_value_unchecked(f, y), y)
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
  length(f.w) == prod(size(x)) || throw(DimensionMismatch("a parameter array has length \$(length(f.w)) and the argument has \$(prod(size(x))) entries"))
  nothing
end
```
"""
check_parameters(::AbstractFunction, x) = nothing

"""
    ProximalBase.entry_parameters(f)

Declare the per-entry parameters of `f` as a tuple of `name => value` pairs,
for example `(:w => f.w,)`. Include arrays indexed like the argument;
`nothing` and scalar numbers are skipped. Other values throw `ArgumentError`.
Declare `()` when there are none. Each user [`SeparableFunction`](@ref) must
supply this method to work on a [`SymmetricSparseIterate`](@ref); its default
is `nothing` (undeclared). The declaration must include every per-entry array.
It describes only `f` itself, not the functions inside its wrappers.

The generic array and coordinate oracles check these parameters automatically.
An override of a public oracle replaces that boundary and should call
[`ProximalBase.check_symmetric_parameters`](@ref) when accepting a symmetric
iterate. Built-in wrappers also check their inner functions before writing.
Raw entry oracles and `restrict` have no iterate to check; validate the original
function before using them.

See [Symmetric parameters](@ref) for which arrays must be symmetric and which calls check them.
"""
function entry_parameters end

"""
    ProximalBase.check_symmetric_parameters(f, x)

Check every off-diagonal pair of each declared per-entry array of `f` and its
inner functions on a [`SymmetricSparseIterate`](@ref). Return `true`, or throw
`ArgumentError` for an undeclared user separable function, a malformed
declaration or an asymmetric pair. A tuple `x` pairs recursively with the
blocks of [`SeparableSum`](@ref); other iterates return `true`.

This checks all pairs, even those absent from the iterate. Coordinate oracles
check only their own pair. It does not check separately supplied step arrays
or certify arrays against later mutation. A user override of a public oracle
should call this check before writing; the declaration is trusted to be complete.

See [Symmetric parameters](@ref) for which arrays must be symmetric and which calls check them.
"""
function check_symmetric_parameters end

# An indicator is zero at its projection. Built-ins check their parameter
# arrays privately; the fallback calls the public extension hook before writes.
_isindicator(::Type{<:AbstractFunction}) = false
_isindicator(f::AbstractFunction) = _isindicator(typeof(f))
_counts_nonzeros(::AbstractFunction) = false
_scale_invariant(::AbstractFunction) = false
_checkparams(f::AbstractFunction, x) = (check_parameters(f, x); true)
_argument_type(::Type{<:AbstractFunction}) = Any
_checkargument(f, x) = x isa _argument_type(typeof(f)) || throw(MethodError(f, (x,)))
_checkgradient_output(g, x) = (g === x && throw(ArgumentError("the gradient must not overwrite x")); nothing)
# An owner output is also aliased by a matrix that shares its storage, such as its matrix view and the wrappers of
# that view, which a gradient written coordinate by coordinate would change before the point is read again
_checkgradient_output(g::SymmetricSparseIterate, x) =
  (_owner_shares(g, x) && throw(ArgumentError("the gradient must not overwrite x")); nothing)
_owner_shares(g::SymmetricSparseIterate, x) = g === x
_owner_shares(g::SymmetricSparseIterate, x::AbstractArray) = Base.mightalias(g, x)
_owner_shares(g::SymmetricSparseIterate, x::SymmetricSparseIterate) =
  g === x || (g.p > 0 && x.p > 0 && any(id -> id in Base.dataids(g), Base.dataids(x)))
# blocks pair up to the shorter tuple; the size checks report a length mismatch
@generated _checkgradient_output(g::Tuple, x::Tuple) =
  _blockwise(k -> :(_checkgradient_output(g[$k], x[$k])), min(fieldcount(g), fieldcount(x)), nothing)

function _checkwrite(op, f, y, x)
  _checkreal(f, x)
  _checkreal(f, y)
  (op === gradient! || op === value_and_gradient!) && _checkgradient_output(y, x)
  if op === value_and_prox! || op === gradient!
    _checkparams(f, x)
    _checkargument(f, x)
  else
    _checkargument(f, x)
    _checkparams(f, x)
  end
  true
end
# Scalar-step checks inspect types and shapes; transformed inputs have no values.
# A built-in function for which `_prox_supported` is false is rejected with the MethodError of the direct call;
# a block of a sum then raises it before any block is written. The trait is a function of the type, which is
# a constant where the type is known.
_prox_supported(f) = _prox_supported(typeof(f))
_prox_supported(::Type) = true
_checkstep(op, f, y, x, γ) = _prox_supported(f) || _prox_unsupported(y, f, x, γ)
# The arguments of the direct call: a placeholder of the step pass is replaced by the array that it stands for.
# A MethodError records the world age of its call, which the message of Julia 1.12 and later shows.
@static if isdefined(Base, :tls_world_age)
  _call_world() = Base.tls_world_age()
else
  _call_world() = typemax(UInt)
end
@noinline _prox_unsupported(y, f, x, γ) = throw(MethodError(prox!, (_unplaced(y), f, _unplaced(x), γ), _call_world()))
_unplaced(a) = a
_unplaced(a::Tuple) = map(_unplaced, a)
# The steps of an indicator projection, and of the penalized ball and sphere kernels, are
# the scalars in [0, Inf]; the comparison is written so that it rejects NaN.
@inline _checkstep_nonneg(γ::Real) = γ >= 0 || _throw_step(γ)
@noinline _throw_step(γ) = throw(ArgumentError("γ must be nonnegative, got $γ"))
struct _StepInput{T, N, A<:AbstractArray{T, N}} <: AbstractArray{T, N}
  array::A
end
Base.size(x::_StepInput) = size(x.array)
Base.axes(x::_StepInput) = axes(x.array)
_unplaced(a::_StepInput) = _unplaced(a.array)
_step_values(x::AbstractArray) = true
# A placeholder shows the storage of the array it stands for, never its values.
_projection_host(x::_StepInput) = _projection_host(x.array)
_step_values(x::_StepInput) = false
_step_values(x::SubArray) = _step_values(parent(x))

_checkvalue(f, x) = (_checkreal(f, x); _checkparams(f, x))
_checkconjugate(f, y) = (_checkreal(f, y); true)

####################################
#  Gradients
####################################

"""
    gradient!(g, f, x)

Store the gradient of `f` at `x` in `g` and return `g`.
`g` must not be `x` unless the function documents otherwise; built-in functions
throw an `ArgumentError`. A [`SymmetricSparseIterate`](@ref) `g` receives the symmetric part
of the gradient, the derivative over symmetric matrices.
"""
gradient!(g, f::AbstractFunction, x) =
  (_check_oracle_arrays(gradient!, f, g, x); _gradient_body!(g, f, x))
_gradient_unchecked!(g, f::AbstractFunction, x) =
  _isbuiltin(f) ? _gradient_body!(g, f, x) : gradient!(g, f, x)
function _gradient_body!(g, f::AbstractFunction, x)
  _checkwrite(gradient!, f, g, x)
  _value_and_gradient_unchecked!(g, f, x)
  g
end

"""
    value_and_gradient!(g, f, x)

Store the gradient of `f` at `x` in `g` and return `f(x)`. Usually cheaper than
calling `f(x)` and [`gradient!`](@ref) separately.
`g` must not be `x` unless the function documents otherwise; built-in functions
throw an `ArgumentError`.
"""
function value_and_gradient! end

# a dense array for the gradient at x, also when x is a sparse iterate or the owner of a symmetric matrix
_gradient_buffer(x::AbstractArray) = similar(Array{float(eltype(x))}, axes(x))
_gradient_buffer(x::SymmetricSparseIterate) = Matrix{float(eltype(x))}(undef, size(x))
@generated _gradient_buffer(x::Tuple) = _blocktuple(k -> :(_gradient_buffer(x[$k])), fieldcount(x))

_prox_buffer(x::AbstractArray) = similar(x, float(eltype(x)))
_prox_buffer(x::Union{Symmetric, Hermitian}) = Matrix{float(eltype(x))}(undef, size(x))
@generated _prox_buffer(x::Tuple) = _blocktuple(k -> :(_prox_buffer(x[$k])), fieldcount(x))

function _checksize(y::Tuple, x::Tuple)
  length(y) == length(x) || _block_size_error(length(y), length(x))
  _checksize_blocks(y, x)
end
@generated _checksize_blocks(y::Tuple, x::Tuple) = _blockwise(k -> :(_checksize(y[$k], x[$k])), fieldcount(y), true)
@noinline _block_size_error(n, m) = throw(DimensionMismatch("the output has $n blocks and the input $m"))

prox(f::AbstractFunction, x::Tuple, γ::Real = 1) = prox!(_prox_buffer(x), f, x, γ)

"""
    gradient(f, x)

Allocating version of [`gradient!`](@ref). At a [`SymmetricSparseIterate`](@ref) the gradient
is a new dense, exactly symmetric `Matrix`, not an iterate.
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
    prox_block(f, u, k, γ)
    prox_block!(out, f, u, k, γ)

Compute `prox_{γ fₖ}(u)` for one block of the penalty `f`, at the already
computed trial point `u`. For a separable function, `u` and the return value
are scalars and `k` is the underlying array's linear entry index. For a group
penalty, `k` is its group index, `u` is a compact vector in `f.groups[k]` order,
and the allocating form returns a vector. Coupled penalties without such a
decomposition do not support this operation.

`prox_block!` writes a compact group vector and returns `out`. It shares the
full-array proximal kernels and their step, weight, precision, NaN/Inf and tie
policies. Exact aliasing `out === u` is supported; other overlapping buffers
are rejected. Both buffers must be real vectors with one-based indexing and
exactly the group's length. **Caller precondition:** `out` must be independent
of iterate storage and solver caches. This includes public views such as
`activevalues(x)`, `coordinates(x)` and `blockview(x,k)`: they alias their owner
even when their parent is a plain vector. Ownership is not inferred or checked
from array types. Use the allocating form or an independent numerical buffer.
Group, shape, element-type, overlap and parameter checks precede writes;
a failing element read, conversion or user kernel may leave a partial output,
as with ordinary array writes. With this precondition, only the numerical
output is changed; the solver subsequently writes its iterate and updates caches.

**Index spaces:** a symmetric iterate coordinate is not a scalar penalty
entry. Translate `(r,c) = tril_cartesian(p,k)` to `i = (c-1)p+r` before calling
`prox_block(f, u, i, γ)`, and call [`ProximalBase.check_symmetric_parameters`](@ref) before
updating a symmetric iterate. The oracle itself has no owner and cannot check
pair symmetry. Group layouts of a penalty and an iterate must also be matched
by the caller. See [Symmetric parameters](@ref).

Custom [`SeparableFunction`](@ref) subtypes implement the scalar `prox_block`
method, alongside [`value_entry`](@ref). Full `prox!` uses that same method.
"""
function prox_block end
function prox_block! end
@doc (@doc prox_block) prox_block!

####################################
#  Optimality and active sets
####################################

"""
    kkt_violation(f, g, i)
    kkt_violation(f, g, x, k)

How much a coordinate at zero violates the optimality conditions of `h + f`,
where `g = ∇h(x)` is the gradient of the smooth part. For a
[`SparseIterate`](@ref) or a [`SymmetricSparseIterate`](@ref) `x`, the second form checks
the real element types, then that `k` is one of `1:ncoordinates(x)` (a `BoundsError`
otherwise), then that `g` has `length(x)` entries, or `p^2` for the symmetric iterate (a
`DimensionMismatch` otherwise); a matrix-shaped or flat `g` of that length is accepted. For an
[`BlockIterate`](@ref), `k` is a group, and the method is unchanged by these checks, and a group
penalty has its group layout validated where that applies. For other arrays `x` the second form
adds no check of `k` or `g` against `x`; the checks of the real element types and those made by
`f` and by indexing still apply. On a
[`SymmetricSparseIterate`](@ref), use the symmetric part of `g`; an opposing
pair of infinite entries has no symmetric part and throws `ArgumentError`.
A positive value means
that moving the coordinate away from zero decreases the objective, and a value
`≤ 0` that zero is optimal. For [`NormL1`](@ref) it is `|gᵢ| - λ0 wᵢ`.

For a separable `f` the condition involves only `gᵢ`, and the first form
suffices. The second form takes a coordinate `k` of the point `x`: for a
[`GroupNormL2`](@ref), the condition of an index involves the rest of its group,
and on a [`BlockIterate`](@ref) `k` is a group, whose violation is
`‖g[Gₖ]‖₂ - λ0 wₖ` (for a separable `f`, the largest over the group).

See [Symmetric parameters](@ref) for which arrays must be symmetric and which calls check them.
"""
function kkt_violation end

# for a separable function, the optimality of a coordinate at zero does not
# depend on the rest of x
kkt_violation(f::AbstractFunction, g::AbstractArray, x::AbstractArray, i::Integer) =
  (eltype(g) <: Real || _real_error(:kkt_violation); eltype(x) <: Real || _real_error(:kkt_violation);
   _check_kkt_coordinate(g, x, i);
   _isbuiltin(f) ? _kkt_violation(f, g, x, i) : kkt_violation(f, g, i))
function kkt_violation(f::AbstractFunction, g::AbstractArray, x::SymmetricSparseIterate, k::Integer)
  eltype(g) <: Real || _real_error(:kkt_violation)
  _check_kkt_coordinate(g, x, k)
  r, c = tril_cartesian(x.p, k)
  view = _SymmetricGradient(g, x.p)
  view[(c - 1) * x.p + r]  # reject an undefined symmetric part before dispatch
  _check_entry_arrays(f, x, _CoordinatePair(Int(k)))
  _kkt_violation(f, view, x, k)
end
# k must be a coordinate of x, tested before any conversion of k, and g must have the extent of the
# problem: length(x) for a SparseIterate, p^2 for a SymmetricSparseIterate; O(1), no scan of g
@inline function _check_kkt_coordinate(g::AbstractArray, x::Union{SparseIterate, SymmetricSparseIterate}, k::Integer)
  n = ncoordinates(x)
  (1 <= k) & (k <= n) || throw(BoundsError(Base.OneTo(n), k))
  length(g) == _gradient_extent(x) || _throw_kkt_extent(g, x)
  nothing
end
_check_kkt_coordinate(::AbstractArray, ::AbstractArray, ::Integer) = nothing
_gradient_extent(x::SparseIterate) = length(x.ind2nzval)
_gradient_extent(x::SymmetricSparseIterate) = x.p^2
@noinline _throw_kkt_extent(g, x) = throw(DimensionMismatch(
  "the gradient has length $(length(g)), but the $(nameof(typeof(x))) needs one of length $(_gradient_extent(x))"))
# a group at zero violates optimality as much as its worst index
kkt_violation(f::AbstractFunction, g::AbstractArray, x::BlockIterate, k::Integer) =
  (eltype(g) <: Real || _real_error(:kkt_violation);
   _isbuiltin(f) ? _kkt_violation(f, g, x, k) : maximum(i -> kkt_violation(f, g, i), x.groups[k]))

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
`MoreauEnvelope` and `SqrDistL2` restrict only a separable function or set,
respectively, and throw otherwise.

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
what `gradient!` and `prox!` do. Overlapping views of `G` and `x` are unsupported.
"""
function gradient_mapping!(G::AbstractArray, f::AbstractFunction, g::AbstractFunction, x::AbstractArray, γ::Real)
  eltype(G) <: Real && eltype(x) <: Real || _real_error(:gradient_mapping!)
  G === x && throw(ArgumentError("G must not be x"))
  _checksize(G, x)
  _checkparams(f, x)
  _checkparams(g, x)
  _checkwrite(gradient!, f, G, x)
  _checkwrite(prox!, g, G, G)
  _check_write_arrays(gradient!, f, G, x)
  _check_write_arrays(prox!, g, G, G)
  _gradient_unchecked!(G, f, x)
  G .= x .- γ .* G
  _prox_unchecked!(G, g, G, γ)
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

function _kkt_violation(f::AbstractFunction, g::AbstractArray, x::SymmetricSparseIterate, k::Integer)
  r, c = tril_cartesian(x.p, k)
  kkt_violation(f, g, (c - 1) * x.p + r)
end

_kkt_violation(f::AbstractFunction, g::AbstractArray, x::AbstractArray, i::Integer) =
  _isbuiltin(f) ? kkt_violation(f, g, i) : kkt_violation(f, g, x, i)
_kkt_violation(f::AbstractFunction, g::AbstractArray, x::BlockIterate, k::Integer) =
  _isbuiltin(f) ? maximum(i -> kkt_violation(f, g, i), x.groups[k]) : kkt_violation(f, g, x, k)

####################################
#  Linear minimization oracles
####################################

"""
    haslmo(C)

Whether the set `C` (an indicator function) has a linear minimization oracle,
[`lmo!`](@ref), on its documented input domain. It is `false` unless a set
declares it. Unlike the other traits it is a property of the instance, not the
type: [`IndBox`](@ref) has an oracle when every bound is finite, and
[`IndSimplex`](@ref), [`IndBallL1`](@ref) and [`IndBallL2`](@ref) when the radius
is finite, and the capped simplex `Constrained(IndBox(0, 1), IndHyperplane(a, k))`
when `a` is `1` or an array of ones and `k` is a nonnegative integer; a scalar `a`
does not know the dimension, so `0 ≤ k ≤ n` is checked by `lmo!`. It is also `true` for the matrix sets
`SingularValue(IndBallL1(r))` (the nuclear-norm ball), `Spectral(IndSimplex(r))` (the density matrices and,
with `equality = false`, the PSD trace ball) and the Fantope `Spectral(Constrained(IndBox(0, 1),
IndHyperplane(1, k)))`, under the same conditions on `r` and `k`. A `false` means that the oracle is unsupported, which is a different
statement from "the linear objective is unbounded below": an unbounded set can
have a finite minimum for some directions, and `lmo!` makes no promise about
sets outside the compact families it lists.
"""
haslmo(::AbstractFunction) = false

"""
    lmo!(s, C, g)

Store in `s` a minimizer of the linear function `s ↦ dot(g, s)` over the compact
set `C`, the linear minimization oracle of Frank–Wolfe methods, and return `s`.
`C` must have [`haslmo`](@ref); otherwise `lmo!` throws an `ArgumentError`
before writing. Supported sets: [`IndSimplex`](@ref) (equality and inequality),
[`IndBallL1`](@ref), [`IndBallL2`](@ref), [`IndBox`](@ref) and
[`IndBallLinf`](@ref) with finite bounds, and the capped simplex
`Constrained(IndBox(0, 1), IndHyperplane(1, k))` (the normal `1` or an array of
ones, an integer `0 ≤ k ≤ length(g)`). The matrix sets `SingularValue(IndBallL1(r))`,
`Spectral(IndSimplex(r))` and the Fantope have dense oracles for a real square (spectral sets) or
rectangular `g`, which write `-r u vᵀ`, `r v vᵀ` and `Q Qᵀ`, and [`lmo_atom`](@ref) returns the same
minimizers in factored form without the dense matrix; see there for their ties, symmetric-part
convention and exactness. A non-finite entry of such a `g` gives an all-`NaN` `s`. No other
`LinearConstrained` set has an oracle: other bounds, normals, halfspaces and hyperslabs do not.

| Set | `s` | Ties and degenerate cases |
|:----|:----|:--------------------------|
| `IndSimplex(r)` | `r eᵢ`, `i` the first index of the smallest `gᵢ` | with `equality = false`, `0` when `minᵢ gᵢ ≥ 0` (including a tie at `0`) |
| `IndBallL1(r)` | `-r sign(gᵢ) eᵢ`, `i` the first index of the largest `\\|gᵢ\\|` | `0` when `g = 0` |
| `IndBallL2(r)` | `-r g / ‖g‖₂` | `0` when `g = 0` |
| `IndBox(l, u)`, `IndBallLinf(r)` | `lᵢ` if `gᵢ > 0`, `uᵢ` if `gᵢ < 0` | `lᵢ` when `gᵢ = 0` |
| capped simplex, `k` | `sᵢ = 1` for the `k` smallest `gᵢ`, `0` elsewhere | the earliest indices among equal entries; `0` for `k = 0` and `1` for `k = n`, whatever `g` holds |

The capped simplex selects in `O(n)` expected time, without sorting, and compares
entries in their own type, so integers of any size keep their order. It keeps a
copy of `g` in the scratch of the set, which is reused by later calls and by
`Conjugate(C)`: use one set object per thread. An infeasible `k > length(g)`
is an `ArgumentError`, and `NaN` in `g` gives `NaN` in all of `s` unless `k` is `0` or `n`.

Arrays of any shape are vectors in linear (column-major) order, so "first
index" is the first in that order and a matrix has the Frobenius pairing. `g`
holds real numbers, and `s` must be an `Array`, or a view of one, of
floating-point numbers of the size of `g`. A radius or bound is rounded to the
element type of `s`; a finite one that overflows it is an `ArgumentError`, and one that underflows
to zero is accepted as zero. The
result is feasible up to that rounding. A radius of `0` gives `0`, and a box
coordinate with `lᵢ = uᵢ` gives that value, whatever `g` holds there. Zeros are
`+0.0`.

`s` may be `g`; a partial overlap, such as shifted views of one array, is not
supported. A `Symmetric` `g` is read as a dense matrix, and the result need not
be symmetric. Validation precedes every write: a size mismatch is a
`DimensionMismatch`, and an unsupported set, an invalid output or parameter, or
an empty simplex with `equality = true` and a radius above the tolerance of the
element type, an `ArgumentError`, with `s` unchanged.

A `NaN` in `g` gives `NaN` in every output entry that depends on it: all of
`s` for the simplex and the balls, entry `i` for the box. Infinite entries take
the limit where it is determined: the simplex, the ℓ1 ball and the box compare
entries, so `-Inf` wins the minimum and `±Inf` the largest magnitude; for the
ℓ2 ball one infinite entry gives the vertex `-r sign(gᵢ) eᵢ`, and two or more
give `NaN`. See [`prox!`](@ref) for the conventions the aliasing and the
non-finite rules follow.
"""
function lmo!(s, C::AbstractFunction, g)
  haslmo(C) || _lmo_unsupported(C)
  throw(MethodError(lmo!, (s, C, g)))
end

@noinline _lmo_unsupported(C) = throw(ArgumentError(
  "no linear minimization oracle for $(typeof(C)) with these parameters: haslmo(C) is false " *
  "(only compact sets listed in lmo! are supported; this does not say the linear objective is unbounded)"))

"""
    lmo(C, g)

Allocating version of [`lmo!`](@ref): the result is a dense `Array` of the size of `g` and
the element type `float(eltype(g))`, also for a sparse or structured `g`.
"""
lmo(C::AbstractFunction, g::AbstractArray) = lmo!(Array{float(eltype(g))}(undef, size(g)), C, g)
