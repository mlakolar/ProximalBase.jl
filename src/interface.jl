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
It is implemented by `NormL1`, `NormL1Nonneg`, `ElasticNet`, `SqrNormL2`, `LinearFunction`,
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
    kkt_violation(f, v, i)
    kkt_violation(f, v, x, k)
    kkt_violation(f, v, x)

Return a first-order optimality score of `h + f` from a vector `v` that the caller
supplies, usually `∇h(x)`; `v` is not recomputed. The score is raw: no scaling,
tolerance or stopping rule is applied, and it concerns stationarity, not global
optimality.

* `kkt_violation(f, v, i)` is the signed screen of coordinate `i` **assumed zero**.
  It reads no `x`. A positive value means that zero violates its condition.
* `kkt_violation(f, v, x, k)` is the score of the actual coordinate `k` of `x`
  (group `k` of a `BlockIterate`): the signed margin at a zero coordinate, and a
  nonnegative mismatch of stationarity at a nonzero one. A positive mismatch is not
  an instruction to set the coordinate to zero. *Since 0.6 this form reads `x`: at a
  nonzero coordinate it does not return the screen.*
* `kkt_violation(f, v, x)` is `max(0, max_k score_k)`, nonnegative, and zero of the
  working type for an empty `x`.

With `cᵢ = λ0*wᵢ` (`λ1*wᵢ` for `ElasticNet`), the scores of entry `i` at `xᵢ = 0` and
at `xᵢ ≠ 0` are, for the six functions that have all three forms:

* `NormL1`: `abs(vᵢ) - cᵢ` and `abs(vᵢ + cᵢ*sign(xᵢ))`.
* `NormL1Nonneg`: for finite inputs, `-vᵢ - cᵢ` at `xᵢ = 0`, and at `xᵢ ≠ 0` `abs(vᵢ + cᵢ)` for `xᵢ > 0` and
  `Inf` for `xᵢ < 0`, which is outside the domain `x ≥ 0`.
* `ElasticNet`: `abs(vᵢ) - cᵢ` (`-Inf` for an infinite ridge level) and
  `abs((vᵢ + λ2*xᵢ) + cᵢ*sign(xᵢ))`.
* `ZeroFunction`: `abs(vᵢ)` in both cases.
* `LinearFunction`: `abs(vᵢ + cᵢ)` in both cases, with `cᵢ` its coefficient entry.
* `SqrNormL2`: `abs(vᵢ + dᵢ)` in both cases, with `dᵢ` the entry `λ*xᵢ` or `λ*(xᵢ - yᵢ)` of the gradient at `x`,
  computed in the working type of the call. The screen `kkt_violation(f, v, i)` takes `xᵢ = 0` and does not use it.

`GroupNormL2` and `NormL2` score a block `G` with `c = λ*w_G` (`c = λ` for `NormL2`, whose block is all of `x`):
`norm(v_G) - c` at a zero block and `norm(v_G + c*x_G/norm(x_G))` at a nonzero one. The block of the entry `k` of an
array is its group (an entry in no group scores `abs(vₖ)`), and for a `BlockIterate` it is group `k`, whose layout must be
that of `f`. The value of `kkt_violation(f, v, x)` is then a maximum of block 2-norms and of those entries, not a global
∞-norm. `NormL2` has a full-point form, and its forms for an entry or a group, and those of `GroupNormL2` at a
`SymmetricSparseIterate`, throw an `ArgumentError`. At a nonzero block, in the ordinary range of the squares, the error is at most `(n + 8)*u*(norm(v_G) + c)`
to first order in `n*u`, with `n` the size of the block and `u = eps(W)/2` for the working type `W` of `v` and `x`; outside
that range the sums are scaled by an exponent, which extends the range and not the precision.

`NormLinf(λ)` has the full-point form (its forms for an entry or a group throw an `ArgumentError`) and returns the
Euclidean distance from `-v` to the subdifferential of `λ*maximum(abs, x)` at `x`. At `x = 0` that is the distance from `v`
to the ℓ1 ball of radius `λ`, which is `0` when `sum(abs, v) <= λ`. At `x ≠ 0` the face is the set of entries whose
magnitude equals the largest one after conversion to the working type, with no tolerance, so for a positive finite level
an exact tie can make the value discontinuous in `x`. A length `n` for which `n + 1` is not a finite number of the working
type (`Float16` from `n = 65519`) throws an `ArgumentError`.

`Scaled` with a finite positive factor `α` evaluates the existing score of its function (the functions above; `NormL2` in
the full-point form) on the lazy gradient `v/α`, with the rounding and range limits of that evaluation, and multiplies it by
`α`; underflow, overflow and `NaN` are not recovered. For a chain of wrappers whose factors do not multiply to a normal number, the product is applied to the gradient
of `LinearFunction` and `SqrNormL2` (the three-argument form too), and folded into the level of the other functions where
that gives a normal level. Where it does not, the three-argument form and a zero coordinate or group have a value, and a
nonzero point and the full-point form throw an `ArgumentError`. So does an infinite factor at a nonzero point and in
the full-point form. A function of this package without a score gives a `MethodError` under `Scaled`. A function
defined outside this package gets its three-argument value at a zero coordinate and an `ArgumentError` at a nonzero one,
unless it defines the method. Under `Scaled` it has its scaled three-argument value and that error also when it defines
a four-argument method of its own, because PB does not know that its score scales with `α`: a method for `Scaled{<:T}`
supplies the scaled forms.

# Examples
```jldoctest kkt_violation_example
julia> kkt_violation(NormL1(0.5), [0.75, 0.25], 1)
0.25

julia> g = NormL1(0.5); x = [0.0, 0.0, 2.0, -1.0]; v = [0.8, 0.2, 0.0, 0.5];

julia> kkt_violation(g, v, 3)  # the screen assumes x[3] = 0
-0.5

julia> kkt_violation(g, v, x, 3)  # x[3] = 2 is read: a mismatch
0.5

julia> kkt_violation(g, v, x)
0.5
```

Storage-specific checks, element types, non-finite values and the extension rules are in
[scoring a coordinate](@ref kkt-screens) and the [numerical contract](@ref numerics).
"""
function kkt_violation end

# The four-argument form reads the point. A function of this package takes its own method (_kkt_violation);
# a function defined outside it has the screen of the three-argument form at a zero coordinate (_kkt_custom).
# The method is marked for inlining, as the public four-argument methods of a Scaled are (functions/scaled.jl): inlining
# permits the compiler to hoist loop-invariant state of the unweighted cases out of the caller's loop; weighted coefficients
# remain entry-dependent.
@inline kkt_violation(f::AbstractFunction, g::AbstractArray, x::AbstractArray, i::Integer) =
  (_check_one_based(g, x); eltype(g) <: Real || _real_error(:kkt_violation); eltype(x) <: Real || _real_error(:kkt_violation);
   _check_kkt_coordinate(g, x, i);
   length(g) == length(x) || _gradient_length_error(g, x);
   _isbuiltin(f) ? _kkt_violation(f, g, x, i) : _kkt_custom(f, g, x, i))
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
NormL1, NormL1Nonneg and ElasticNet slice weights, centered SqrNormL2 slices its center, LinearFunction
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

# A function of this package without a point-aware method (or, below, a custom function) reaches these methods:
# the four-argument call has no value for it, and for a function defined outside the package the value is the
# screen of the three-argument form at a zero coordinate or group.
_kkt_violation(f::AbstractFunction, g::AbstractArray, x::AbstractArray, i::Integer) =
  _isbuiltin(f) ? throw(MethodError(kkt_violation, (f, g, x, i))) : kkt_violation(f, g, x, i)

# a group at zero violates optimality as much as its worst index
kkt_violation(f::AbstractFunction, g::AbstractArray, x::BlockIterate, k::Integer) =
  (_check_one_based(g); eltype(g) <: Real || _real_error(:kkt_violation);
   length(g) == length(x) || _gradient_length_error(g, x);
   _isbuiltin(f) ? _kkt_violation(f, g, x, k) : _kkt_custom(f, g, x, k))

_kkt_violation(f::AbstractFunction, g::AbstractArray, x::BlockIterate, k::Integer) =
  _isbuiltin(f) ? throw(MethodError(kkt_violation, (f, g, x, k))) : kkt_violation(f, g, x, k)

function _kkt_violation(f::AbstractFunction, g::AbstractArray, x::SymmetricSparseIterate, k::Integer)
  _isbuiltin(f) && throw(MethodError(kkt_violation, (f, g, x, k)))
  r, c = tril_cartesian(x.p, k)
  i = (c - 1) * x.p + r
  iszero(_getstored(x, k)) ? kkt_violation(f, g, i) : _kkt_custom_error(f, g, i)
end

# a function defined outside the package: its three-argument screen where the coordinate or group of x is
# zero, an error where it is not
function _kkt_custom(f::AbstractFunction, g::AbstractArray, x::AbstractArray, i::Integer)
  iszero(x[i]) && return kkt_violation(f, g, i)
  _kkt_custom_error(f, g, i)
end
function _kkt_custom(f::AbstractFunction, g::AbstractArray, x::BlockIterate, k::Integer)
  G = x.groups[k]
  all(j -> iszero(x[j]), G) && return maximum(i -> kkt_violation(f, g, i), G)
  _kkt_custom_error(f, g, first(G))
end
@noinline function _kkt_custom_error(f, g, i)
  hasmethod(kkt_violation, Tuple{typeof(f), typeof(g), typeof(i)}) || throw(MethodError(kkt_violation, (f, g, i)))
  name = nameof(typeof(f))
  throw(ArgumentError("kkt_violation(f, g, x, k) at a nonzero coordinate or group is not defined for $name: its " *
    "three-argument method is the screen of a coordinate assumed zero. Define " *
    "kkt_violation(f::$name, g::AbstractArray, x::AbstractArray, k::Integer), and a method for BlockIterate " *
    "and SymmetricSparseIterate where they are used"))
end

####################################
#  Point-aware scores of the entrywise functions
####################################

# Each of NormL1, NormL1Nonneg, ElasticNet, ZeroFunction, LinearFunction and SqrNormL2 supplies, for the floating type W of
# the call:
#   _kkt_type(f, V, X)                 W, from the element types of v and x
#   _kkt_state(f, W)                   the values that a call prepares once and reuses for its entries: the coefficient
#                                      of an unweighted NormL1, NormL1Nonneg or ElasticNet, a level
#   _kkt_natural(f, s, W, v, x, i)     the ordinary score of entry i from v and x in W, by the formulas of kkt_violation
#   _kkt_entry(f, s, W, v, x, i)       the score of entry i for the local form and the pass by entry: the inputs that
#                                      the formula of f uses are converted to W, and the rules for NaN and infinite
#                                      inputs are applied
#   _kkt_flag(f, x, r)                 0, or NaN for an entry that makes the natural result of the loop doubtful beyond a NaN
# NormL1, NormL1Nonneg and ElasticNet also supply
#   _kkt_entry_coefficient(f, s, W, i) the coefficient of entry i: the one held by the state of an unweighted function,
#                                      or the one that _kkt_coefficient forms from the level and the weight of the entry
# _kkt_coefficient (numerical_kernels.jl) is the numerical rule that forms a coefficient λw in a requested type, and
# _kkt_entry_coefficient is the lookup for one entry of a call, which returns the held value or applies that rule.
# _kkt_reduce computes the full-point score. When v and x are both an Array{W} with W Float32 or Float64 and
# _kkt_fast_ok(f, s) holds, it is a loop over the natural scores; otherwise it is the pass by entry of _kkt_entry
# (_kkt_scalar). The loop keeps a NaN score in its maximum, which is also the result the rules give for a NaN in an
# input that the formula uses; the entries that the loop does not decide (for example an infinite x, which changes the
# score of NormL1 and ElasticNet, or a non-finite natural score of SqrNormL2) and a NaN maximum send the call to the
# pass by entry.
@inline _kkt_type(f, ::Type{V}, ::Type{X}) where {V, X} = float(promote_type(V, X))
@inline _kkt_fast_ok(f, s) = true
@inline _kkt_flag(f, x, r) = x - x
@inline function _kkt_entry(f, s, ::Type{W}, v, x, i) where {W}
  vw = convert(W, v)
  xw = convert(W, x)
  _kkt_override(_kkt_natural(f, s, W, vw, xw, i), vw, xw)
end

# a NaN in the entries the formula reads gives NaN, and otherwise an infinite one gives Inf, replacing the natural score r
@inline function _kkt_override(r::W, v, x) where {W}
  (isnan(v) | isnan(x)) && return W(NaN)
  (isinf(v) | isinf(x)) && return W(Inf)
  r
end
@inline function _kkt_override(r::W, v) where {W}
  isnan(v) && return W(NaN)
  isinf(v) && return W(Inf)
  r
end

Base.@propagate_inbounds function _kkt_local(f, g::AbstractArray, x::AbstractArray, i::Integer)
  W = _kkt_type(f, eltype(g), eltype(x))
  _kkt_entry(f, _kkt_state(f, W), W, g[i], x[i], i)
end
function _kkt_block(f, g::AbstractArray, x::BlockIterate, k::Integer)
  G = x.groups[k]
  W = _kkt_type(f, eltype(g), eltype(x))
  s = _kkt_state(f, W)
  m = W(-Inf)
  for j in G
    m = max(m, _kkt_entry(f, s, W, g[j], x[j], j))
  end
  m
end
# coordinate k of the lower triangle: the symmetric gradient pair of g, the parameters at the lower position
function _kkt_symmetric(f, g::AbstractArray, x::SymmetricSparseIterate, k::Integer)
  r, c = tril_cartesian(x.p, k)
  j = (c - 1) * x.p + r
  W = _kkt_type(f, eltype(g), eltype(x))
  _kkt_entry(f, _kkt_state(f, W), W, g[j], _getstored(x, k), j)
end

# the full-point form: one validation and a pass over the entries; dense arrays of one hardware type can be followed by
# a pass by entry that applies the rules for non-finite values (see _kkt_reduce)
function _kkt_point(f, v::AbstractArray, x::AbstractArray)
  _check_one_based(v, x)
  eltype(v) <: Real || _real_error(:kkt_violation)
  eltype(x) <: Real || _real_error(:kkt_violation)
  length(v) == length(x) || _gradient_length_error(v, x)
  _checkparams(f, x)
  W = _kkt_type(f, eltype(v), eltype(x))
  _kkt_reduce(f, _kkt_state(f, W), W, v, x)
end
# max(0, max score) by entry, with the maximum that keeps a NaN
function _kkt_scalar(f, s, ::Type{W}, v, x) where {W}
  m = zero(W)
  for i in 1:length(x)
    m = max(m, _kkt_entry(f, s, W, v[i], x[i], i))
  end
  m
end
_kkt_reduce(f, s, ::Type{W}, v, x) where {W} = _kkt_scalar(f, s, W, v, x)
# Dense arrays of one hardware type: a loop over the natural scores. The running maximum from zero is the maximum
# of the scores and 0, and keeps a NaN. The flag sum is NaN when an entry is one that the loop does not decide, and
# then the pass by entry decides, as it does for a NaN maximum.
function _kkt_reduce(f, s, ::Type{W}, v::Array{W}, x::Array{W}) where {W<:Union{Float32, Float64}}
  _kkt_fast_ok(f, s) || return _kkt_scalar(f, s, W, v, x)
  m = zero(W)
  flag = zero(W)
  @inbounds @simd for i in 1:length(x)
    vi = v[i]
    xi = x[i]
    r = _kkt_natural(f, s, W, vi, xi, i)
    flag += _kkt_flag(f, xi, r)
    m = max(m, r)
  end
  isnan(m + flag) ? _kkt_scalar(f, s, W, v, x) : m
end

@noinline _kkt_symmetric_point_error(f) = throw(ArgumentError(
  "kkt_violation(f, v, x) is not defined for a SymmetricSparseIterate and $(nameof(typeof(f))): use the coordinate form kkt_violation(f, v, x, k)"))

# The methods of a function with the entry kernels above, for arrays, a BlockIterate and a SymmetricSparseIterate.
macro _kkt_entrywise(F)
  esc(quote
    _kkt_violation(f::$F, g::AbstractArray, x::AbstractArray, i::Integer) = _kkt_local(f, g, x, i)
    _kkt_violation(f::$F, g::AbstractArray, x::BlockIterate, k::Integer) = _kkt_block(f, g, x, k)
    _kkt_violation(f::$F, g::AbstractArray, x::SymmetricSparseIterate, k::Integer) = _kkt_symmetric(f, g, x, k)
    kkt_violation(f::$F, v::AbstractArray, x::AbstractArray) = _kkt_point(f, v, x)
    kkt_violation(f::$F, ::AbstractArray, ::SymmetricSparseIterate) = _kkt_symmetric_point_error(f)
  end)
end
