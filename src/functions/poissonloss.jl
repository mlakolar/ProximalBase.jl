"""
    PoissonLoss(y, X)

Convex, smooth Poisson regression loss with predictor `η = X*β`. The
mathematical objective is `sum(exp(ηᵢ) - yᵢ*ηᵢ)` with gradient
`X' * (exp(ηᵢ) - yᵢ)ᵢ`; the computed terms at an infinite predictor or an
overflowing intermediate follow the
[endpoint rules](@ref spec-poissonloss-endpoints). The constant `log Γ(yᵢ + 1)`
of the negative log-likelihood does not depend on `β` and is omitted. The value
is a sum over observations. For a positive number of observations `n` and a
finite unscaled sum, `(1/n) * f` gives the mean.

Use a `Vector{Float64}` of responses and a `Matrix{Float64}` design with
`size(X, 1) == length(y)`. Responses must be finite and nonnegative; a
noninteger response is accepted. The constructor borrows both arrays. It has
no weight or offset argument. A caller who forms `η = X*β + offset` evaluates
that model through [`ProximalBase.value_from_predictor`](@ref) and
[`ProximalBase.value_and_gradient_from_predictor!`](@ref), which take any
predictor that meets their buffer requirements.

The loss implements [`gradient!`](@ref) and [`value_and_gradient!`](@ref),
with no proximal operator. The API supplies no gradient bound for any
instance, so [`lipschitz`](@ref) throws; the curvature in `β` is unbounded for a
design with a nonzero row. A term whose `exp(ηᵢ)` or `yᵢ*ηᵢ` overflows is
computed as `±Inf`. See [Poisson regression](@ref Poisson-regression-loss) for
ownership, scaling and numerical limits.
"""
struct PoissonLoss <: AbstractFunction
  y::Vector{Float64}
  X::Matrix{Float64}
  tmp::Vector{Float64}

  function PoissonLoss(y::Vector{Float64}, X::Matrix{Float64})
    n = size(X, 1)
    _ls_checklength(:y, y, n)
    all(t -> isfinite(t) && t >= 0.0, y) || throw(ArgumentError("responses must be finite and nonnegative"))
    new(y, X, Vector{Float64}(undef, n))
  end
end

issmooth(::Type{<:PoissonLoss}) = true
isconvex(::Type{<:PoissonLoss}) = true

# The requested value term (V) and derivative (D) of one observation at the
# predictor η; a quantity that is not requested is returned as zero. The value term
# is e = exp(η) for a zero response or an infinite exponential, and e - y*η
# otherwise. ifelse evaluates both arms, so the selection of e keeps the NaN of an
# unselected 0*Inf (zero response, infinite η) or Inf - Inf (infinite exponential
# and product) out of the result. NaN in η gives NaN.
@inline function _poisson_terms(η::Float64, y::Float64, ::Val{V}, ::Val{D}) where {V,D}
  e = exp(η)
  v = V ? ifelse(iszero(y) | (e == Inf), e, e - y * η) : 0.0
  d = D ? e - y : 0.0
  v, d
end

function _poisson_scan!(d, f::PoissonLoss, η, value::Val{V}) where {V}
  total = 0.0
  derivative = Val(d !== nothing)
  y = f.y
  @inbounds for i in eachindex(η)
    v, di = _poisson_terms(η[i], y[i], value, derivative)
    V && (total += v)
    d === nothing || (d[i] = di)
  end
  total
end

function _poisson_predict!(f, β)
  η = f.tmp
  isempty(f.X) ? fill!(η, 0.0) : mul!(η, f.X, β)
  η
end

function _poisson_transpose!(g, X, d)
  isempty(X) ? fill!(g, 0.0) : mul!(g, transpose(X), d)
end

@inline function _poisson_check_output(out, name, f, input, input_name)
  _ls_overlap(out, input) && _ls_alias_error(name, input_name)
  _ls_overlap(out, f.X) && _ls_alias_error(name, "the design X")
  _ls_overlap(out, f.y) && _ls_alias_error(name, "the response y")
  _ls_overlap(out, f.tmp) && _ls_alias_error(name, "the scratch of the loss")
  nothing
end

function _checkvalue(f::PoissonLoss, β::StridedVector{Float64})
  _check_one_based(β)
  _ls_checklength(:β, β, size(f.X, 2))
  _ls_overlap(β, f.tmp) && throw(ArgumentError("coefficients might share memory with the scratch of the loss"))
  nothing
end

function _checkwrite(::Union{typeof(gradient!),typeof(value_and_gradient!)}, f::PoissonLoss,
                     g::StridedVector{Float64}, β::StridedVector{Float64})
  _checkvalue(f, β)
  _check_one_based(g)
  _ls_checklength(:g, g, size(f.X, 2))
  _poisson_check_output(g, "g", f, β, "the coefficients β")
end

(f::PoissonLoss)(β::StridedVector{Float64}) =
  (_checkvalue(f, β); _poisson_scan!(nothing, f, _poisson_predict!(f, β), Val(true)))

function gradient!(g::StridedVector{Float64}, f::PoissonLoss, β::StridedVector{Float64})
  _checkwrite(gradient!, f, g, β)
  _gradient_unchecked!(g, f, β)
end

function _gradient_unchecked!(g::StridedVector{Float64}, f::PoissonLoss, β::StridedVector{Float64})
  η = _poisson_predict!(f, β)
  _poisson_scan!(η, f, η, Val(false))
  _poisson_transpose!(g, f.X, η)
end

function value_and_gradient!(g::StridedVector{Float64}, f::PoissonLoss, β::StridedVector{Float64})
  _checkwrite(value_and_gradient!, f, g, β)
  _value_and_gradient_unchecked!(g, f, β)
end

function _value_and_gradient_unchecked!(g::StridedVector{Float64}, f::PoissonLoss, β::StridedVector{Float64})
  η = _poisson_predict!(f, β)
  v = _poisson_scan!(η, f, η, Val(true))
  _poisson_transpose!(g, f.X, η)
  v
end

@inline function _poisson_check_predictor(f, η)
  _check_one_based(η)
  _ls_checklength(:η, η, length(f.y))
  _ls_overlap(η, f.tmp) && throw(ArgumentError("the predictor might share memory with the scratch of the loss"))
  nothing
end

function value_from_predictor(f::PoissonLoss, η::StridedVector{Float64})
  _poisson_check_predictor(f, η)
  _poisson_scan!(nothing, f, η, Val(true))
end

function value_and_gradient_from_predictor!(g::StridedVector{Float64}, d::StridedVector{Float64},
                                            f::PoissonLoss, η::StridedVector{Float64})
  _poisson_check_predictor(f, η)
  _check_one_based(g, d)
  _ls_checklength(:d, d, length(f.y))
  _ls_checklength(:g, g, size(f.X, 2))
  _ls_overlap(d, g) && _ls_alias_error("d", "the output g")
  _poisson_check_output(d, "d", f, η, "the predictor η")
  _poisson_check_output(g, "g", f, η, "the predictor η")
  v = _poisson_scan!(d, f, η, Val(true))
  _poisson_transpose!(g, f.X, d)
  v
end
