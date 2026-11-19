"""
    LogisticLoss(y, X; weights=nothing, offset=nothing)

Convex, smooth logistic loss with predictor `η = X*β + offset`, value
`sum(wᵢ * (log(1 + exp(ηᵢ)) - yᵢ*ηᵢ))` and gradient
`X' * (wᵢ * (σ(ηᵢ) - yᵢ))ᵢ`, where `σ` is the logistic sigmoid.
The defaults are unit weights and zero offset.

Use one-based strided vectors and a strided or `SparseMatrixCSC` design,
all of the same type (`Float32` or `Float64`). Responses must be finite in
`[0, 1]`, including on zero-weight rows. Weights must be finite and
nonnegative, and offsets finite. The constructor borrows these arrays.

The loss implements [`gradient!`](@ref) and [`value_and_gradient!`](@ref),
with no proximal operator. For caller-held predictors use
[`ProximalBase.value_from_predictor`](@ref) or
[`ProximalBase.value_and_gradient_from_predictor!`](@ref).
See [logistic regression](@ref Logistic-regression-loss) for ownership,
scaling and numerical limits.
"""
struct LogisticLoss{T,Y,X,W,O} <: AbstractFunction
  y::Y
  X::X
  weights::W
  offset::O
  tmp::Vector{T}

  function LogisticLoss(y::StridedVector{T}, X::Union{StridedMatrix{T},SparseMatrixCSC{T}};
                        weights=nothing, offset=nothing) where {T<:Union{Float32,Float64}}
    _check_one_based(y, X, weights, offset)
    n = size(X, 1)
    _ls_checklength(:y, y, n)
    if weights !== nothing
      weights isa StridedVector{T} || throw(ArgumentError("weights must be a strided vector of type $T"))
      _ls_checklength(:weights, weights, n)
      all(w -> isfinite(w) && w >= zero(T), weights) ||
        throw(ArgumentError("weights must be finite and nonnegative"))
    end
    if offset !== nothing
      offset isa StridedVector{T} || throw(ArgumentError("offset must be a strided vector of type $T"))
      _ls_checklength(:offset, offset, n)
      all(isfinite, offset) || throw(ArgumentError("offset must be finite"))
    end
    all(t -> zero(T) <= t <= one(T), y) || throw(ArgumentError("responses must be finite and in [0, 1]"))
    new{T,typeof(y),typeof(X),typeof(weights),typeof(offset)}(y, X, weights, offset, Vector{T}(undef, n))
  end
end

issmooth(::Type{<:LogisticLoss}) = true
isconvex(::Type{<:LogisticLoss}) = true

# A subnormal or zero exp(-s) can lose bits before a large weight rescales it.
# In this range log1p(t) and t/(1+t) round to t. Use quarter exponents so the
# individual exponentials are normal whenever the final tail is representable.
@noinline function _logistic_weighted_tail(w, s)
  q = exp(-s / 4)
  (((w * q) * q) * q) * q
end

# For y = 1/2, the slope magnitude is (1 - exp(-s)) / (2 * (1 + exp(-s))).
@noinline function _logistic_midpoint_slope(w::T, s::T, e::T) where {T}
  # Below eps(T), the relative error of the linear term s/4 is O(s^2).
  # Apply the weight first so dividing a subnormal predictor retains its contribution.
  s < eps(T) && return (w * s) / 4
  w * (-expm1(-s) / (2 * (one(T) + e)))
end

@inline function _logistic_terms(η::T, y, w, ::Val{V}, ::Val{D}) where {T,V,D}
  iszero(w) && return (zero(T), zero(T))
  positive = η > zero(T)
  # The flags specialize this loop. Fused calls share the unsigned
  # subtraction and apply its sign after weighting.
  fused = V && D
  c = fused ? (positive ? one(T) - y : y) : ifelse(positive, one(T) - y, y)
  s = abs(η)
  e = exp(-s)
  linear = zero(T)
  if V
    if fused
      linear = iszero(c) && isinf(s) ? zero(T) : c * s
    else
      linear = c * ifelse(iszero(c) & isinf(s), zero(T), s)
    end
  end
  if e < floatmin(T) && w > one(T)
    tail = _logistic_weighted_tail(w, s)
    v = V ? w * linear + tail : zero(T)
    wc = w * c
    d = D ? (fused ? wc - tail : ifelse(positive, wc - tail, tail - wc)) : zero(T)
  else
    v = V ? w * (linear + log1p(e)) : zero(T)
    if D && s < T(0.0625) && y == T(0.5) && !iszero(s)
      return v, copysign(_logistic_midpoint_slope(w, s, e), η)
    end
    q = e / (one(T) + e)
    d = D ? w * (fused ? c - q : ifelse(positive, c - q, q - c)) : zero(T)
  end
  return v, fused && !positive ? -d : d
end

@inline _logistic_weight(::Nothing, ::Type{T}, i) where {T} = one(T)
@inline _logistic_weight(w, ::Type, i) = w[i]

function _logistic_scan!(d, f::LogisticLoss{T}, η, value::Val{V}) where {T,V}
  total = zero(T)
  derivative = Val(d !== nothing)
  @inbounds for i in eachindex(η)
    v, di = _logistic_terms(η[i], f.y[i], _logistic_weight(f.weights, T, i), value, derivative)
    V && (total += v)
    d === nothing || (d[i] = di)
  end
  total
end

function _logistic_predict!(f, β)
  η = f.tmp
  if isempty(f.X)
    fill!(η, zero(eltype(η)))
  else
    mul!(η, f.X, β)
  end
  f.offset === nothing || (η .+= f.offset)
  η
end

function _logistic_transpose!(g, X, d)
  isempty(X) ? fill!(g, zero(eltype(g))) : mul!(g, transpose(X), d)
end

@inline function _logistic_check_output(out, name, f, input, input_name)
  _ls_overlap(out, input) && _ls_alias_error(name, input_name)
  _ls_overlap(out, f.X) && _ls_alias_error(name, "the design X")
  _ls_overlap(out, f.y) && _ls_alias_error(name, "the response y")
  f.weights !== nothing && _ls_overlap(out, f.weights) && _ls_alias_error(name, "the weights")
  f.offset !== nothing && _ls_overlap(out, f.offset) && _ls_alias_error(name, "the offset")
  _ls_overlap(out, f.tmp) && _ls_alias_error(name, "the scratch of the loss")
  nothing
end

function _checkvalue(f::LogisticLoss{T}, β::StridedVector{T}) where {T}
  _check_one_based(β)
  _ls_checklength(:β, β, size(f.X, 2))
  _ls_overlap(β, f.tmp) && throw(ArgumentError("coefficients might share memory with the scratch of the loss"))
  nothing
end

function _checkwrite(::Union{typeof(gradient!),typeof(value_and_gradient!)}, f::LogisticLoss{T},
                     g::StridedVector{T}, β::StridedVector{T}) where {T}
  _checkvalue(f, β)
  _check_one_based(g)
  _ls_checklength(:g, g, size(f.X, 2))
  _logistic_check_output(g, "g", f, β, "the coefficients β")
end

(f::LogisticLoss{T})(β::StridedVector{T}) where {T} =
  (_checkvalue(f, β); _logistic_scan!(nothing, f, _logistic_predict!(f, β), Val(true)))

function gradient!(g::StridedVector{T}, f::LogisticLoss{T}, β::StridedVector{T}) where {T}
  _checkwrite(gradient!, f, g, β)
  _gradient_unchecked!(g, f, β)
end

function _gradient_unchecked!(g::StridedVector{T}, f::LogisticLoss{T}, β::StridedVector{T}) where {T}
  η = _logistic_predict!(f, β)
  _logistic_scan!(η, f, η, Val(false))
  _logistic_transpose!(g, f.X, η)
end

function value_and_gradient!(g::StridedVector{T}, f::LogisticLoss{T}, β::StridedVector{T}) where {T}
  _checkwrite(value_and_gradient!, f, g, β)
  _value_and_gradient_unchecked!(g, f, β)
end

function _value_and_gradient_unchecked!(g::StridedVector{T}, f::LogisticLoss{T}, β::StridedVector{T}) where {T}
  η = _logistic_predict!(f, β)
  v = _logistic_scan!(η, f, η, Val(true))
  _logistic_transpose!(g, f.X, η)
  v
end

@inline function _logistic_check_predictor(f, η)
  _check_one_based(η)
  _ls_checklength(:η, η, length(f.y))
  _ls_overlap(η, f.tmp) && throw(ArgumentError("the predictor might share memory with the scratch of the loss"))
  nothing
end

function value_from_predictor(f::LogisticLoss{T}, η::StridedVector{T}) where {T}
  _logistic_check_predictor(f, η)
  _logistic_scan!(nothing, f, η, Val(true))
end

function value_and_gradient_from_predictor!(g::StridedVector{T}, d::StridedVector{T},
                                            f::LogisticLoss{T}, η::StridedVector{T}) where {T}
  _logistic_check_predictor(f, η)
  _check_one_based(g, d)
  _ls_checklength(:d, d, length(f.y))
  _ls_checklength(:g, g, size(f.X, 2))
  _ls_overlap(d, g) && _ls_alias_error("d", "the output g")
  _logistic_check_output(d, "d", f, η, "the predictor η")
  _logistic_check_output(g, "g", f, η, "the predictor η")
  v = _logistic_scan!(d, f, η, Val(true))
  _logistic_transpose!(g, f.X, d)
  v
end
