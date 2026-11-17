####################################
#  ℓ2 norm  λ ‖x‖₂
####################################

# The numerical kernels of NormL2 and GroupNormL2 (shrinkage, values, screens and their recovery of values
# out of range) are in numerical_kernels.jl; this file has the checks, the type and the oracles.

@noinline _l2_step_error(f, γ) = throw(ArgumentError("the step of $(nameof(typeof(f))) must be nonnegative, got $γ"))
# a negative or NaN step fails the comparison
@inline _check_l2_step(f, γ::Real) = (γ >= zero(γ) || _l2_step_error(f, γ); nothing)

@noinline _symmetric_prox_error(f) = throw(ArgumentError(
  "$(nameof(typeof(f))) has no proximal operator for a SymmetricSparseIterate; use a dense matrix such as Matrix(x)"))

####################################
#  NormL2
####################################

"""
    NormL2(λ = 1)

Euclidean-norm penalty `f(x) = λ‖x‖₂`; for a matrix this is the Frobenius norm.
The level must be nonnegative and not `NaN`. Its proximal operator shrinks the
whole argument: `max(1 - γλ/‖x‖₂, 0) * x`. Use [`GroupNormL2`](@ref) to
shrink separate groups.

Values accept real, one-based arrays and the three iterate types. Sparse and
symmetric values read only stored entries; symmetric off-diagonal coordinates
count twice in the squared norm. Proximal calls accept arrays of the same size,
a nonnegative scalar step and exact input/output identity.

# Examples
```jldoctest
julia> f = NormL2(1.0);

julia> f([3.0, 4.0])
5.0

julia> prox(f, [3.0, 4.0], 2.5)
2-element Vector{Float64}:
 1.5
 2.0
```

# Edge cases
- No proximal operator is provided for [`SymmetricSparseIterate`](@ref).
- Invalid shapes, real types, one-based axes and negative/`NaN` steps are checked
  before writes. Other overlapping input/output buffers are unsupported.
- `λ = Inf` fixes every output at `+0`, including at zero step and non-finite input.
- Otherwise, zero level or zero step gives the identity. At positive finite
  level and step, a `NaN` entry makes the entire result `NaN`.

See [ℓ₂ operations](@ref norm-l2-guide),
[group endpoint rules](@ref group-l2-endpoints) and the
[numerical contract](@ref numerics) for the remaining endpoints and range limits.
"""
struct NormL2{T<:Real} <: AbstractFunction
  λ::T

  NormL2{T}(λ::T) where {T<:Real} = (_checknonneg("λ", λ); new{T}(λ))
end
NormL2(λ::Real = 1) = NormL2{typeof(λ)}(λ)

isproximable(::Type{<:NormL2}) = true
isconvex(::Type{<:NormL2}) = true

(f::NormL2)(x::_ArrayOrSymmetricIterate) = (_check_one_based(x); _checkreal(f, x); _l2_value(f.λ, x))

function _check_l2_arrays(f, y::AbstractArray, x::AbstractArray, γ::Real)
  _checkreal(f, x)
  _checkreal(f, y)
  Base.require_one_based_indexing(y, x)
  _checksize(y, x)
  _check_l2_step(f, γ)
  nothing
end

prox!(y::AbstractArray, f::NormL2, x::AbstractArray, γ::Real) =
  (_check_l2_arrays(f, y, x, γ); _l2_shrink!(y, x, Base.OneTo(length(x)), f.λ, true, γ); y)
function value_and_prox!(y::AbstractArray, f::NormL2, x::AbstractArray, γ::Real)
  _check_l2_arrays(f, y, x, γ)
  G = Base.OneTo(length(x))
  s = _l2_shrink!(y, x, G, f.λ, true, γ)
  # a recomputed value reads y as the value does: its stored entries only for a sparse iterate
  s == _L2_RECOMPUTE ? _l2_value(f.λ, y) : _l2_fused_term(f.λ, true, y, G, s, _valtype(y))
end
# routes of wrappers such as Scaled, which preflight with _checkwrite first
_value_and_prox_unchecked!(y::AbstractArray, f::NormL2, x::AbstractArray, γ::Real) = value_and_prox!(y, f, x, γ)
