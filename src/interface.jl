####################################
#
#  The function interface
#
####################################

"""
    AbstractFunction

Abstract supertype for mathematical functions with callable values, gradients
and proximal operators. Declare `issmooth`, `isproximable`, `isseparable` and
`isconvex` for the function's type. Mutating oracles take their output first.

Shapes, real element types and parameter extents are checked before predictable
output writes. The per-entry parameters of a function used on a
SymmetricSparseIterate (weights, centers, coefficients) must be exactly symmetric:
this is a precondition on the model that the oracles assume and do not compare.
Validate it once per model with `ProximalBase.check_symmetric_parameters`; custom
separable functions declare their arrays with `entry_parameters`. A failure during
numerical computation may leave the output changed.
"""
abstract type AbstractFunction end

# Where the oracles of the families that support it take either an array or a SymmetricSparseIterate.
# A symmetric iterate is not an array: a method that is not written with this type, or with
# SymmetricSparseIterate, does not accept it.
const _ArrayOrSymmetricIterate = Union{AbstractArray, SymmetricSparseIterate}

# the number of entries of the array, or of the matrix that the symmetric iterate represents
_nentries(x::AbstractArray) = length(x)
_nentries(x::SymmetricSparseIterate) = x.p^2

function value_and_prox! end
function gradient! end

"""
    issmooth(f)

Whether f provides a gradient and value_and_gradient! on the interior of its
domain. Declare the trait on its type; the instance query forwards to it.
"""
issmooth(::Type{<:AbstractFunction}) = false
issmooth(f::AbstractFunction) = issmooth(typeof(f))

"""
    isproximable(f)

Whether `f` has an efficient proximal operator, computed by `prox!` and
`value_and_prox!`.
"""
isproximable(::Type{<:AbstractFunction}) = false
isproximable(f::AbstractFunction) = isproximable(typeof(f))

"""
    isseparable(f)

Whether `f(x) = Σᵢ fᵢ(xᵢ)` is a sum over the entries of `x`. A separable `f` has
the scalar proximal operators `prox_block`, and
`kkt_violation` needs only the gradient.
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

It is a property of the function instance and its parameters.
"""
iszeropreserving(::AbstractFunction) = false

"""
    isconvex(f)

Whether `f` is convex. For a nonconvex `f`, `prox!` returns one of the
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
function. A `SymmetricSparseIterate` output takes a scalar step `γ`, the Frobenius
proximal operator of the symmetric matrix; an array or per-group step is an `ArgumentError`
before anything is written (use `Matrix(x)` or `matrixview(x)` with a dense output). Any other sharing of memory between `y` and `x` is not supported and
is not detected, even when the two cover the same entries: a call such as
`prox!(view(x, :), f, x, γ)`, or with a shifted or reversed view of `x`, can
return wrong values without an error. Copy the argument first, as in
`prox!(y, f, copy(x), γ)`. Supported operand shapes depend on the concrete function.
"""
function prox! end
prox!(y, f::AbstractFunction, x) = prox!(y, f, x, 1)

"""
    prox(f, x, γ = 1)

Allocating version of `prox!`. For a supported `SymmetricSparseIterate` `x` the result is an
independent `SymmetricSparseIterate` of the float element type, and `γ` is a scalar.
"""
prox(f::AbstractFunction, x::AbstractArray, γ::Real = 1) = prox!(similar(x, float(eltype(x))), f, x, γ)
# a Symmetric or Hermitian wrapper does not accept off-diagonal writes: a dense result
prox(f::AbstractFunction, x::Union{Symmetric, Hermitian}, γ::Real = 1) =
  prox!(Matrix{float(eltype(x))}(undef, size(x)), f, x, γ)

"""
    value_and_prox!(y, f, x, γ=1)

Store the proximal point in `y` and return the value at that stored result.
Exact input/output aliasing is supported. Other shared-memory views are not
supported. Shape and parameter checks precede writes; failures during numerical
computation may leave output changed.
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
when they do not. The default for an `AbstractFunction` returns
`nothing`. Define this method for a function with parameter arrays; the
generic kernels call it before writing any output, including `x` when the
output aliases it.

For a `SeparableFunction` `WeightedSquare` with a weight vector `w`:

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
Declare `()` when there are none. Each user `SeparableFunction` must
supply this method to work on a `SymmetricSparseIterate`; its default
is `nothing` (undeclared). The declaration must include every per-entry array.
It describes only `f` itself, not the functions inside its wrappers.

Within ProximalBase, only `ProximalBase.check_symmetric_parameters` reads this declaration. The
numerical oracles do not compare mirrored entries of these arrays: the caller or
solver validates the model, and an oracle given asymmetric parameters has no
guaranteed numerical meaning. Built-in wrappers expose their inner function's
arrays to the checker. Raw entry oracles and `restrict` have no iterate; validate the
original function before using them.

See Symmetric parameters for which arrays must be symmetric and who validates them.
"""
function entry_parameters end

"""
    ProximalBase.check_symmetric_parameters(f,x)

Validate caller-supplied per-entry parameters on a symmetric iterate. Every
declared array must be exactly symmetric at every off-diagonal pair, including
pairs the iterate does not store, recursively through `Scaled`; an asymmetric
array, or malformed or undeclared custom separable parameter metadata, throws
`ArgumentError`. Arrays symmetric by type, such as `Symmetric(W, :L)`, pass.
Other iterates return true.

The numerical oracles never call this check: validate the model at the solve or
workspace boundary, and again after changing a parameter array. A successful
check is not recorded anywhere and is not a replacement for `check_parameters`
or the shape checks of the oracles.
"""
function check_symmetric_parameters end

# An indicator is zero at its projection. Built-ins check their parameter
# arrays privately; the fallback calls the public extension hook before writes.
_isindicator(::Type{<:AbstractFunction}) = false
_isindicator(f::AbstractFunction) = _isindicator(typeof(f))
_checkparams(f::AbstractFunction, x) = (check_parameters(f, x); true)
_argument_type(::Type{<:AbstractFunction}) = Any
_checkargument(f, x) = x isa _argument_type(typeof(f)) || throw(MethodError(f, (x,)))
_checkgradient_output(g, x) = (g === x && throw(ArgumentError("the gradient must not overwrite x")); nothing)
# A symmetric iterate output is also aliased by a matrix that shares its storage, such as its matrix view and the wrappers of
# that view, which a gradient written coordinate by coordinate would change before the point is read again
_checkgradient_output(g::SymmetricSparseIterate, x) =
  (_symmetric_iterate_mightalias(g, x) && throw(ArgumentError("the gradient must not overwrite x")); nothing)
_symmetric_iterate_mightalias(g::SymmetricSparseIterate, x) = g === x
_symmetric_iterate_mightalias(g::SymmetricSparseIterate, x::AbstractArray) = Base.mightalias(g, x)
_symmetric_iterate_mightalias(g::SymmetricSparseIterate, x::SymmetricSparseIterate) =
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
_checkvalue(f, x) = (_checkreal(f, x); _checkparams(f, x))

####################################
#  Gradients
####################################

"""
    gradient!(g, f, x)

Store the gradient of `f` at `x` in `g` and return `g`.
`g` must not be `x` unless the function documents otherwise; built-in functions
throw an `ArgumentError`. A `SymmetricSparseIterate` `g` receives the symmetric part
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
calling `f(x)` and `gradient!` separately.
`g` must not be `x` unless the function documents otherwise; built-in functions
throw an `ArgumentError`.
"""
function value_and_gradient! end

# a dense array for the gradient at x, also when x is a sparse iterate or a SymmetricSparseIterate
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

Allocating version of `gradient!`; array arguments get dense floating output.
Supported symmetric-iterate gradients (SqrNormL2, LinearFunction and supported Scaled
leaves) return a dense symmetric Matrix. ZeroFunction has symmetric-iterate value/prox
but no symmetric-iterate gradient method. Traits do not supply missing operand methods.
"""
gradient(f::AbstractFunction, x::AbstractArray) = gradient!(_gradient_buffer(x), f, x)
gradient(f::AbstractFunction, x::Tuple) = gradient!(_gradient_buffer(x), f, x)

"""
    value_and_gradient(f, x)

Allocating version of `value_and_gradient!`: returns `(f(x), ∇f(x))`.
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
prox_block(f, v::Real, i::Integer, γ::Real)
    prox_block(f, u::AbstractVector, k::Integer, γ::Real)

Compute a pure mathematical proximal point at an already constructed trial.
Scalar methods use the underlying array's global linear entry index `i`;
retained built-ins are NormL1, SqrNormL2, LinearFunction, ZeroFunction and
supported Scaled leaves. Custom SeparableFunction subtypes implement this
scalar method together with value_entry. No iterate or cache is updated.

The vector form allocates a plain floating Vector of length(u) and delegates
to prox_block!: the built-in GroupNormL2 (group k of its groups, in their order)
or a custom implementation, whose group index and compact vector order that
extension defines. The generic wrapper adds no group, shape, parameter or
overlap validation of its own; GroupNormL2 validates in its prox_block!.
Implementing a block hook does not add a full-array prox or capability trait.

For symmetric iterate coordinate k, convert `(r,c)=tril_cartesian(p,k)` to
full-entry `i=(c-1)p+r` for a scalar hook and validate the original function
with ProximalBase.check_symmetric_parameters before using raw entry calls.
"""
function prox_block end
"""
    prox_block!(out, f, u, k, γ)

Pure compact-vector block proximal extension hook. A conforming custom method
writes the proximal point of its selected block into out and returns out.
Buffers follow that block's compact ordered indices. GroupNormL2 is the
built-in implementation; the allocating vector prox_block delegates here.

The extension validates one-based real vectors, group bounds, exact lengths,
parameters and supported step before predictable writes. Exact out===u is
supported; other overlap must be rejected by the extension. Failed reads,
conversions or user arithmetic may leave partial numerical output.

Caller precondition: out is independent of iterate storage and solver/model
caches, including activevalues, coordinates and blockview aliases. Ownership
is not inferred from buffer types. Compute in independent scratch, then write
public storage and update caller caches separately. No activation occurs here.
"""
function prox_block! end

####################################
#  Optimality and active sets
####################################

"""
kkt_violation(f, g, i)
    kkt_violation(f, g, x, k)

Signed screen of a coordinate assumed zero in `h+f`, with full smooth-part
gradient `g=∇h(x)`. A positive value means zero violates its condition;
a nonpositive value checks zero only, not stationarity of a nonzero coordinate.
NormL1 returns `abs(g[i])-λ0*w[i]`. Retained scalar screens also cover
ZeroFunction, LinearFunction, SqrNormL2 and supported Scaled leaves.

For SparseIterate and SymmetricSparseIterate, the four-argument form checks
real types, coordinate bounds (BoundsError), then full gradient entry count
(DimensionMismatch): length(x) or p², respectively. A flat or matrix-shaped
gradient of that count is accepted. The symmetric iterate uses the symmetric gradient pair;
opposing infinities in that gradient pair throw ArgumentError. The extent of
the function's parameters is checked, but their mirrored entries are not compared:
symmetric parameters are a precondition (see `check_symmetric_parameters`).

For BlockIterate, k is a group and retained scalar-screen functions return
the maximum screen over its ordered global entry indices. That path does not
add the same explicit full-gradient extent preflight. GroupNormL2 instead
returns the group screen `‖g[Gₖ]‖₂-λ0*wₖ` after checking that group k of x is
its group k and that g has length(x) entries; on an array point it takes an
entry index (see GroupNormL2), and its three-argument form throws. Other array points add
no coordinate/gradient extent check beyond function/indexing checks. Raw
three-argument calls have no symmetric iterate, so no parameter-symmetry preflight.
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
  _check_parameter_extent(f, x)
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

"""
    restrict(f,S)

Restrict a retained function's entry parameters to the integer index vector S.
NormL1 slices weights, centered SqrNormL2 slices its center, LinearFunction
slices coefficients, QuadraticFunction selects its principal submatrix and
linear term, vector-response LeastSquaresLoss selects design columns, and
Scaled forwards. For least squares and quadratics, omitted coordinates are fixed at zero.
Centered SqrNormL2 drops the omitted-coordinate constant λ*sum(y[j]^2)/2;
when finite, it changes values but not restricted minimizers/gradients/prox.
Add it back for full-value comparisons. A nonzero fixed point needs separately
derived residual/linear terms. Raw restriction has no iterate
on which to validate symmetric per-entry parameters; validate the original
function before using the result in a coordinate routine.
"""
function restrict end

"""
    lipschitz(f)

A global gradient Lipschitz bound for retained smooth functions. ZeroFunction
and LinearFunction return zero; SqrNormL2 returns its level. QuadraticFunction
uses a matrix operator-norm bound; LeastSquaresLoss bounds the squared design
norm. Dense matrices use their full spectral norm; other matrices use a safe
one/infinity norm bound. Scaled scales the inner bound. Unknown bounds throw.
The separate opnorm2 estimate may miss a leading eigenspace and needs an
independent bound or backtracking when choosing a step.
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

# a group at zero violates optimality as much as its worst index
kkt_violation(f::AbstractFunction, g::AbstractArray, x::BlockIterate, k::Integer) =
  (eltype(g) <: Real || _real_error(:kkt_violation);
   _isbuiltin(f) ? _kkt_violation(f, g, x, k) : maximum(i -> kkt_violation(f, g, i), x.groups[k]))

_kkt_violation(f::AbstractFunction, g::AbstractArray, x::BlockIterate, k::Integer) =
  _isbuiltin(f) ? maximum(i -> kkt_violation(f, g, i), x.groups[k]) : kkt_violation(f, g, x, k)
