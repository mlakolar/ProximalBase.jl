"""
    PoissonLoss(y, X)

Convex, smooth Poisson regression loss with the log link and predictor
`η = X*β`. The mathematical objective is `sum(exp(ηᵢ) - yᵢ*ηᵢ)` with gradient
`X' * d`, `dᵢ = exp(ηᵢ) - yᵢ`; the constant `log Γ(yᵢ + 1)` is omitted. Computed terms at
infinite or NaN predictors and under overflow follow the
[endpoint](@ref spec-poissonloss-endpoints) and [range](@ref spec-poissonloss-range)
clauses. The value is a sum; for `n = length(y) > 0` and a finite sum,
`(1/n) * f` gives the mean.

`y::Vector{Float64}` holds finite nonnegative responses, which may be
noninteger, and `X::Matrix{Float64}` has `length(y)` rows. The constructor
borrows both arrays and has no weight or offset argument; to use an offset, pass
`η = X*β + offset` to [`ProximalBase.value_from_predictor`](@ref) or
[`ProximalBase.value_and_gradient_from_predictor!`](@ref).

The loss implements [`gradient!`](@ref) and [`value_and_gradient!`](@ref) and
has no proximal operator. The API supplies no gradient Lipschitz bound;
[`lipschitz`](@ref) throws. See the [numerical contract](@ref spec-poissonloss).
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
# otherwise. ifelse evaluates both arms; selecting e keeps the NaN of an unselected
# 0*Inf (zero response, infinite η) or Inf - Inf (infinite exponential and product)
# out of the result. A NaN η makes each requested quantity NaN.
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
