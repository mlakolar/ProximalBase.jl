# Pure compact group oracles. Scalar dispatches live with their functions.
_block_groups(f::AbstractFunction) = throw(ArgumentError("$(typeof(f)) has no group decomposition"))
_block_groups(f::AbstractGroupPenalty) = f.groups
_block_groups(f::Union{Scaled, Regularize}) = _block_groups(f.f)

# Keep the validated boundary visible to callers: in-place views and repeated
# small-block loops can eliminate/hoist checks without changing their contract.
@inline function _check_block_buffers(out, f, u, k)
  groups = _block_groups(f)
  checkbounds(groups, k)
  Base.require_one_based_indexing(out, u)
  _checkreal(f, out)
  _checkreal(f, u)
  n = length(groups[k])
  length(out) == n && length(u) == n || throw(DimensionMismatch("block $k needs input and output vectors of length $n"))
  out !== u && Base.mightalias(out, u) && throw(ArgumentError("prox_block! supports exact aliasing only"))
  _check_block_parameters(f, _model_type(out, u), k)
  nothing
end
_check_block_parameters(::AbstractFunction, ::Type, k) = nothing
_check_block_parameters(f::Scaled, T::Type, k) = _check_block_parameters(f.f, T, k)
function _check_block_parameters(f::Regularize, T::Type, k)
  G = _block_groups(f)[k]
  f.c === nothing || (eltype(f.c) <: Real || _real_error(:prox_block!);
                     checkbounds(f.c, G))
  _check_block_parameters(f.f, T, k)
end
function _check_block_parameters(f::_RadialGroupPenalty, T::Type, k)
  _infinite_radial_coefficient(f, T, k) || _group_scalar(f, T, k)
  nothing
end

function prox_block(f::AbstractFunction, u::AbstractVector, k::Integer, γ::Real)
  out = Vector{float(eltype(u))}(undef, length(u))
  prox_block!(out, f, u, k, γ)
end
@inline function prox_block!(out::AbstractVector, f::AbstractFunction, u::AbstractVector, k::Integer, γ::Real)
  _check_block_buffers(out, f, u, k)
  _prox_block!(out, f, u, k, γ)
  out
end

@inline function _prox_block!(out, f::HomogeneousGroupNorm, u, k, γ)
  T = _model_type(out, u)
  G = eachindex(u)
  if iszero(f.w[k]) || (iszero(f.λ0) && !isinf(f.w[k]))
    copyto!(out, u)
    return out
  end
  c = isinf(f.w[k]) ? _weight_product(T, γ, f.λ0, f.w[k]) : _positive_product(T, γ, f.λ0, f.w[k])
  n = _shrink_group!(f, out, u, G, c)
  if f isa GroupNormL2 && isnan(n)
    _fill_group!(out, u, G, nothing, _hard_fixed(f, γ, k) ? 0 : iszero(γ) ? 1 : 2)
  end
  out
end

@inline function _prox_block!(out, f::SparseGroupNormL2, u, k, γ)
  T = _model_type(out, u)
  G = eachindex(u)
  isinf(f.λ1) && return (_fixed_sparse_group!(out, G, nothing, T); out)
  c1 = _l1_threshold(T, γ, f.λ1)
  if iszero(f.w[k]) || (iszero(f.λ0) && !isinf(f.w[k]))
    _soft_group_only!(out, u, G, c1)
    return out
  end
  c2 = isinf(f.w[k]) ? _weight_product(T, γ, f.λ0, f.w[k]) : _positive_product(T, γ, f.λ0, f.w[k])
  n, _ = _sparse_shrink_group!(out, u, G, c1, c2)
  if isnan(n)
    if _hard_fixed_sparse(f, γ, k)
      _fixed_sparse_group!(out, G, nothing, T)
    elseif iszero(γ)
      _soft_group_only!(out, u, G, c1)
    else
      fill!(out, eltype(out)(NaN))
    end
  end
  out
end

@inline function _prox_block!(out, f::_RadialGroupPenalty, u, k, γ)
  T = _model_type(out, u)
  if _infinite_radial_coefficient(f, T, k)
    _shrink_group!(NormL2(one(T)), out, u, eachindex(u), _weight_product(T, γ, f.λ0, f.w[k]))
    return out
  end
  p = _group_scalar(f, T, k)
  iszero(p.λ) && return copyto!(out, u)
  _radial_shrink!(p, out, u, eachindex(u), convert(T, γ))
  out
end

function _prox_block!(out, s::Scaled, u, k, γ)
  if _nested_route(s, γ)
    f, step, _ = _nested_step(s, γ)
    return @noinline prox_block!(out, f, u, k, step)
  end
  @noinline prox_block!(out, s.f, u, k, s.α * γ)
end
function _prox_block!(out, r::Regularize, u, k, γ)
  T = float(eltype(out))
  a, b, step = _regularize_parameters(convert(T, γ), convert(T, r.ρ), _regularize_endpoint(r, γ))
  G = _block_groups(r)[k]
  @inbounds for j in eachindex(out, u)
    out[j] = r.c === nothing ? a * u[j] : _ridge_blend(u[j], r.c[G[j]], a, b)
  end
  # Keep wrapper numerical loops at their established boundaries: forcing the
  # leaf inline here changes SIMD rounding in sparse/radial group kernels.
  @noinline prox_block!(out, r.f, out, k, step)
end
