####################################
#
#  The function interface
#
####################################

"""
    AbstractFunction

Supertype for mathematical functions with callable values, gradients and
proximal operators. Declare capabilities with [`issmooth`](@ref),
[`isproximable`](@ref), [`isseparable`](@ref) and [`isconvex`](@ref).
Mutating operations put the output first.

Built-in operations check supported real types, one-based axes, shapes and
parameter extents before predictable writes. Numerical failures can leave
partial output. Custom public methods must provide their own validation.

On a [`SymmetricSparseIterate`](@ref), per-entry weights, centers and coefficients
must be exactly symmetric. Oracles assume this model precondition; validate it
with [`ProximalBase.check_symmetric_parameters`](@ref). Custom separable functions
declare their arrays with [`ProximalBase.entry_parameters`](@ref).

See [operations](@ref operation-guide) and [array, precision and ownership
conventions](@ref numerical-usage).
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

Whether `f` provides [`gradient`](@ref) and [`value_and_gradient!`](@ref) on the interior of its
domain. Declare the trait on its type; the instance query forwards to it.
"""
issmooth(::Type{<:AbstractFunction}) = false
issmooth(f::AbstractFunction) = issmooth(typeof(f))

"""
    isproximable(f)

Whether `f` has an efficient proximal operator, computed by `prox!` and
[`value_and_prox!`](@ref).
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

Whether `f` is convex. `true` means the type guarantees convexity for every
instance that satisfies its documented preconditions. `false` means convexity
is not guaranteed; some instances may still be convex, for example a
`QuadraticFunction` with a positive semidefinite matrix. For a nonconvex `f`,
`prox!` returns one of the minimizers that define the proximal operator, and
accelerated methods such as FISTA lose their guarantees.
"""
isconvex(::Type{<:AbstractFunction}) = false
isconvex(f::AbstractFunction) = isconvex(typeof(f))

####################################
#  Proximal operators
####################################

"""
    prox!(y, f, x, γ = 1)

Write the proximal point of `f` at `x` into `y` and return `y`. For `γ > 0`,
this minimizes `f(z) + ‖z-x‖²/(2γ)` over `z`. A nonconvex function
may have several minimizers; a supported method returns one.

`y` must have the supported shape of `x`. Exact in-place use,
`prox!(x, f, x, γ)`, is supported. Other overlapping inputs and outputs,
including distinct views of the same entries, are unsupported and generally
undetected; copy the input first. A supported symmetric-iterate destination
uses the Frobenius metric and accepts only a scalar step; array or per-group
steps throw `ArgumentError` before writes. To use array steps with a symmetric
iterate, use a dense output and pass `Matrix(x)` or [`matrixview(x)`](@ref).

Use positive finite steps for the mathematical proximal map. Validation and
zero/non-finite endpoints depend on the function; see the
[Numerical contract](@ref numerics). For a new output use [`prox`](@ref); for
the value at the stored output use [`value_and_prox!`](@ref).
"""
function prox! end
prox!(y, f::AbstractFunction, x) = prox!(y, f, x, 1)

"""
    prox(f, x, γ = 1)

Return a newly allocated proximal point; see [`prox!`](@ref) for the definition
and step restrictions. Array results use floating-point element types.
`Symmetric`/`Hermitian` array wrappers produce a dense matrix. Supported
[`SymmetricSparseIterate`](@ref) calls produce an independent symmetric iterate
and require a scalar step.

# Examples
```jldoctest
julia> Tuple(prox(NormL1(0.5), [2.0, -1.0], 1.0))
(1.5, -0.5)
```
"""
prox(f::AbstractFunction, x::AbstractArray, γ::Real = 1) =
  (_check_one_based(x); prox!(similar(x, float(eltype(x))), f, x, γ))
# a Symmetric or Hermitian wrapper does not accept off-diagonal writes: a dense result
prox(f::AbstractFunction, x::Union{Symmetric, Hermitian}, γ::Real = 1) =
  (_check_one_based(x); prox!(Matrix{float(eltype(x))}(undef, size(x)), f, x, γ))

"""
    value_and_prox!(y, f, x, γ = 1)

Write the proximal point into `y` and return **`f(y)` at the stored output**,
including its rounding. The return value is a scalar, not `(value, y)`.

Exact `y === x` is supported; other overlapping views are unsupported.
Predictable shape and parameter checks precede writes, but a numerical failure
can leave partial output. Step and storage restrictions are those of
[`prox!`](@ref) and the concrete function.

# Examples
```jldoctest
julia> y = zeros(2); v = value_and_prox!(y, NormL1(0.5), [2.0, -1.0], 1.0);

julia> (v, Tuple(y))
(1.0, (1.5, -0.5))
```
"""
value_and_prox!(y, f::AbstractFunction, x, γ::Real = 1) =
  (x = _symmetric_argument(y, x); _check_fallback_arrays(value_and_prox!, f, y, x, γ); _value_and_prox_body!(y, f, x, γ))
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

See [symmetric parameters](@ref symmetric-parameter-guide) for the arrays
to declare and the model validation boundary.
"""
function entry_parameters end

"""
    ProximalBase.check_symmetric_parameters(f, x)

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

Write the gradient of `f` at `x` into `g` and return `g`.

Use a destination with the supported shape of `x` and independent storage.
Built-in functions reject `g === x` with `ArgumentError`; other overlapping
views are unsupported. Numerical failures can leave a partially written
destination. For supported methods, a [`SymmetricSparseIterate`](@ref)
destination receives the symmetric part of
the gradient, the derivative over symmetric matrices.

See [`gradient`](@ref) for an allocating call and
[`value_and_gradient!`](@ref) to obtain the value in the same call.
"""
gradient!(g, f::AbstractFunction, x) =
  (_check_fallback_arrays(gradient!, f, g, x); _gradient_body!(g, f, x))
_gradient_unchecked!(g, f::AbstractFunction, x) =
  _isbuiltin(f) ? _gradient_body!(g, f, x) : gradient!(g, f, x)
function _gradient_body!(g, f::AbstractFunction, x)
  _checkwrite(gradient!, f, g, x)
  _value_and_gradient_unchecked!(g, f, x)
  g
end

"""
    value_and_gradient!(g, f, x)

Write the gradient into `g` and return **the value at the input `x`**.
The return value is a scalar, not `(value, g)`.

The destination must be independent of the input and have its supported shape.
Built-in functions reject `g === x`; other overlapping views are unsupported.
As with [`gradient!`](@ref), numerical failures can leave partial output.

# Examples
```jldoctest
julia> g = zeros(2); v = value_and_gradient!(g, SqrNormL2(1.0), [2.0, -1.0]);

julia> (v, Tuple(g))
(2.5, (2.0, -1.0))
```
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

prox(f::AbstractFunction, x::Tuple, γ::Real = 1) = (_check_one_based(x); prox!(_prox_buffer(x), f, x, γ))

"""
    gradient(f, x)

Return a newly allocated gradient. Array inputs produce a dense floating-point
array with the same shape; see [`gradient!`](@ref) to reuse an output buffer.

Supported symmetric-iterate gradients (`SqrNormL2`, `LinearFunction` and
supported `Scaled` leaves) produce a dense symmetric matrix. `ZeroFunction`
has no symmetric-iterate gradient method. A capability trait does not supply
missing operand methods.

# Examples
```jldoctest
julia> Tuple(gradient(SqrNormL2(0.5), [2.0, -4.0]))
(1.0, -2.0)
```
"""
gradient(f::AbstractFunction, x::AbstractArray) = (_check_one_based(x); gradient!(_gradient_buffer(x), f, x))
gradient(f::AbstractFunction, x::Tuple) = (_check_one_based(x); gradient!(_gradient_buffer(x), f, x))

"""
    value_and_gradient(f, x)

Return `(f(x), ∇f(x))`, allocating the gradient as [`gradient`](@ref) does.
To reuse a destination, call [`value_and_gradient!`](@ref).

# Examples
```jldoctest
julia> value, grad = value_and_gradient(SqrNormL2(1.0), [2.0, -1.0]);

julia> (value, Tuple(grad))
(2.5, (2.0, -1.0))
```
"""
function value_and_gradient(f::AbstractFunction, x::Union{AbstractArray, Tuple})
  _check_one_based(x)
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

Compute a proximal point for one entry or compact block at the supplied trial,
without changing an iterate, active set or solver cache.

The scalar form uses the **global linear entry index** `i` to select parameters.
It is implemented by `NormL1`, `ElasticNet`, `SqrNormL2`, `LinearFunction`,
`ZeroFunction` and supported `Scaled` leaves. A custom [`SeparableFunction`](@ref)
implements it together with [`value_entry`](@ref).

The vector form allocates a plain floating `Vector` of `length(u)` and calls
[`prox_block!`](@ref). [`GroupNormL2`](@ref) interprets `k` as its group number
and `u` in that group's order. Custom methods define their own layout and must
validate it; the allocating wrapper adds no group, shape, parameter or overlap
checks. A block method alone does not supply full-array operations or traits.

For symmetric coordinate `k`, convert `(r,c) = tril_cartesian(p,k)` to the
full-entry index `i = (c-1)*p+r`. Validate the original model with
[`ProximalBase.check_symmetric_parameters`](@ref) before raw entry calls.

# Examples
```jldoctest
julia> prox_block(NormL1(0.5), 2.0, 1, 1.0)
1.5
```

See [coordinate and block operations](@ref block-proximal-oracle).
"""
function prox_block end
"""
    prox_block!(out, f, u, k, γ)

Write the proximal point of compact block `k` into `out` and return `out`.
[`GroupNormL2`](@ref) supplies the built-in implementation; custom functions
can implement this hook. Buffers follow the selected group's ordered indices.

A conforming method validates real one-based vectors, group bounds, exact
lengths, parameters and the supported step before predictable writes. Exact
`out === u` is supported; other overlap must be rejected by the implementation.
Failed reads, conversions or numerical computations may leave partial output.

**Caller precondition:** `out` must be independent of iterate storage and
solver/model caches, including `activevalues`, `coordinates` and `blockview`
aliases. Buffer types do not establish ownership. Compute in independent
scratch, then commit with a public storage writer and update caches separately.
This operation never activates an iterate coordinate.

See [compact block buffers](@ref block-proximal-oracle) for the full extension contract.
"""
function prox_block! end

####################################
#  Optimality and active sets
####################################

"""
    kkt_violation(f, g, i)
    kkt_violation(f, g, x, k)

Return a signed optimality screen for a coordinate **assumed zero** in `h + f`,
where `g = ∇h(x)` is the full smooth-part gradient. A positive result means
zero violates its condition. A nonpositive result does not certify a nonzero
coordinate or the entire point.

For `NormL1` the scalar screen is `abs(g[i]) - λ0*w[i]`; `ElasticNet` uses
`λ1` and returns `-Inf` for an infinite ridge level. Other supported scalar
screens are `ZeroFunction`, `LinearFunction`, `SqrNormL2` and `Scaled` leaves.

In the four-argument form, `k` is an entry index for a plain array or a
`SparseIterate`, an independent symmetric coordinate for a
`SymmetricSparseIterate`, or a group ID for a `BlockIterate`. Scalar
penalties take the maximum member screen on a block. [`GroupNormL2`](@ref)
uses `norm(g[Gₖ]) - λ0*wₖ` for an inactive group and checks the group layout.
For plain array `x`, however, `k` is an **entry index**: when its group is
nonzero, the screen is `abs(g[k])`, not the group norm. There is no
three-argument `GroupNormL2` screen.

Symmetric models require validated symmetric parameters. The gradient's chosen
symmetric pair must not contain opposing infinities. Gradient-length and bounds
checks differ by storage and function; see [screening a zero coordinate](@ref kkt-screens).

# Examples
```jldoctest
julia> kkt_violation(NormL1(0.5), [0.75, 0.25], 1)
0.25
```
"""
function kkt_violation end

# for a separable function, the optimality of a coordinate at zero does not
# depend on the rest of x
kkt_violation(f::AbstractFunction, g::AbstractArray, x::AbstractArray, i::Integer) =
  (_check_one_based(g, x); eltype(g) <: Real || _real_error(:kkt_violation); eltype(x) <: Real || _real_error(:kkt_violation);
   _check_kkt_coordinate(g, x, i);
   _isbuiltin(f) ? _kkt_violation(f, g, x, i) : kkt_violation(f, g, i))
function kkt_violation(f::AbstractFunction, g::AbstractArray, x::SymmetricSparseIterate, k::Integer)
  _check_one_based(g)
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
    restrict(f, S)

Restrict a retained function's entry parameters to the integer index vector S.
NormL1 and ElasticNet slice weights, centered SqrNormL2 slices its center, LinearFunction
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
  (_check_one_based(g); eltype(g) <: Real || _real_error(:kkt_violation);
   _isbuiltin(f) ? _kkt_violation(f, g, x, k) : maximum(i -> kkt_violation(f, g, i), x.groups[k]))

_kkt_violation(f::AbstractFunction, g::AbstractArray, x::BlockIterate, k::Integer) =
  _isbuiltin(f) ? maximum(i -> kkt_violation(f, g, i), x.groups[k]) : kkt_violation(f, g, x, k)
