####################################
#
#  Diagonal metrics: a step per entry, and per-coordinate curvatures
#
####################################

# In the metric of D = Diagonal(d), the proximal operator of a separable g is
#   argmin_z g(z) + Σᵢ dᵢ (zᵢ - xᵢ)² / 2,
# the scalar proximal operator with the step γᵢ = 1/dᵢ in every entry, and for a
# group norm a metric constant on each group keeps the closed form. A gradient
# method with the step 1/Lₖ in coordinate k, from the per-coordinate curvatures
# below, takes the scale of each column into account without standardizing X.

prox(f::AbstractFunction, x::AbstractArray, γ::AbstractArray) = prox!(similar(x, float(eltype(x))), f, x, γ)

function prox!(y::AbstractArray, ::ZeroFunction, x::AbstractArray, γ::AbstractArray)
  _checkparam(γ, x)
  _checksize(y, x)
  y === x ? y : copyto!(y, x)
end
value_and_prox!(y::AbstractArray, f::ZeroFunction, x::AbstractArray, γ::AbstractArray) =
  (prox!(y, f, x, γ); zero(float(eltype(y))))

# separable functions: the step of each entry; on a SymmetricSparseIterate the
# steps are a p × p array read at the lower triangle, so give a symmetric one.
# Scalar and array steps share the loops of separable.jl; _entry specializes
# to the scalar or to γ[i] without a branch inside the loop.
prox!(y::AbstractArray, f::SeparableFunction, x::AbstractArray, γ::AbstractArray) = _prox_sep!(y, f, x, γ)
value_and_prox!(y::AbstractArray, f::SeparableFunction, x::AbstractArray, γ::AbstractArray) =
  _value_and_prox_sep!(y, f, x, γ)

function prox!(y::AbstractArray, h::_ArrayPrecompose, x::AbstractArray, γ::AbstractArray)
  isseparable(h) || throw(ArgumentError("array steps for an array scale require a separable function"))
  _checkwrite(prox!, h, y, x)
  _checkparam(γ, x)
  _value_and_prox_array!(y, h, x, γ, false)
  y
end
function value_and_prox!(y::AbstractArray, h::_ArrayPrecompose, x::AbstractArray, γ::AbstractArray)
  isseparable(h) || throw(ArgumentError("array steps for an array scale require a separable function"))
  _checkwrite(value_and_prox!, h, y, x)
  _checkparam(γ, x)
  _value_and_prox_array!(y, h, x, γ, true)
end

# group penalties: a step per group, constant on the group, keeps the closed form
value_and_prox!(y::AbstractArray, f::AbstractGroupPenalty, x::AbstractArray, γ::AbstractVector) = _value_and_prox_groups!(y, f, x, γ)
prox!(y::AbstractArray, f::AbstractGroupPenalty, x::AbstractArray, γ::AbstractVector) = (value_and_prox!(y, f, x, γ); y)
value_and_prox!(y::AtomIterate, f::AbstractGroupPenalty, x::AtomIterate, γ::AbstractVector) = _value_and_prox_groups!(y, f, x, γ)

# the entrywise reciprocal of an array of steps, without allocating
struct _ReciprocalArray{T, N, A<:AbstractArray} <: AbstractArray{T, N}
  γ::A
end
_ReciprocalArray(γ::AbstractArray{T, N}) where {T, N} = _ReciprocalArray{float(T), N, typeof(γ)}(γ)
Base.size(a::_ReciprocalArray) = size(a.γ)
Base.IndexStyle(::Type{<:_ReciprocalArray}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(a::_ReciprocalArray{T}, i::Int) where {T} = inv(convert(T, a.γ[i]))

# functions built from functions pass the steps on, entry by entry
prox!(y, s::Scaled, x, γ::AbstractArray) = (_checkparams(s, x); prox!(y, s.f, x, _ScaledArray(γ, s.α)))
value_and_prox!(y, s::Scaled, x, γ::AbstractArray) =
  (_checkparams(s, x); s.α * value_and_prox!(y, s.f, x, _ScaledArray(γ, s.α)))

function prox!(y::AbstractArray, t::Tilt, x::AbstractArray, γ::AbstractArray)
  _checkwrite(prox!, t, y, x)
  _checkparam(γ, x)
  Base.Broadcast.check_broadcast_axes(axes(y), γ)
  y .= x .- γ .* t.c
  prox!(y, t.f, y, γ)
end
function value_and_prox!(y::AbstractArray, t::Tilt, x::AbstractArray, γ::AbstractArray)
  _checkwrite(value_and_prox!, t, y, x)
  _checkparam(γ, x)
  Base.Broadcast.check_broadcast_axes(axes(y), γ)
  y .= x .- γ .* t.c
  value_and_prox!(y, t.f, y, γ) + dot(t.c, y) + t.d
end

function prox!(y::AbstractArray, t::Translate, x::AbstractArray, γ::AbstractArray)
  _checkwrite(prox!, t, y, x)
  y .= x .- t.b
  prox!(y, t.f, y, γ)
  y .+= t.b
end
function value_and_prox!(y::AbstractArray, t::Translate, x::AbstractArray, γ::AbstractArray)
  _checkwrite(value_and_prox!, t, y, x)
  y .= x .- t.b
  v = value_and_prox!(y, t.f, y, γ)
  y .+= t.b
  v
end

# Moreau's identity in each entry: prox_{γᵢfᵢ*}(xᵢ) = xᵢ - γᵢ prox_{fᵢ/γᵢ}(xᵢ/γᵢ)
function prox!(y::AbstractArray, c::Conjugate, x::AbstractArray, γ::AbstractArray)
  _checksize(y, x)
  _checkparams(c, x)
  _checkparam(γ, x)
  _checkwrite(prox!, c, y, x)
  T = float(eltype(y))
  z = _array_buffer!(c.scratch, T, size(x))
  z .= x ./ γ
  prox!(z, c.f, z, _ReciprocalArray(γ))
  y .= x .- γ .* z
end

function value_and_prox!(y::AbstractArray, c::Conjugate, x::AbstractArray, γ::AbstractArray)
  _checkwrite(value_and_prox!, c, y, x)
  _isindicator(c) && (prox!(y, c, x, γ); return _zero_value(y))
  if _has_conjugate_value(c.f)
    prox!(y, c, x, γ)
    return _conjugate_value(c.f, y)
  end
  _checksize(y, x)
  _checkparam(γ, x)
  T = float(eltype(y))
  p = _array_buffer!(c.scratch, T, size(x))
  p .= x ./ γ
  fp = value_and_prox!(p, c.f, p, _ReciprocalArray(γ))
  y .= x .- γ .* p
  dot(p, y) - fp
end

####################################
#  sets in a diagonal metric
####################################

# The projection onto a hyperplane in the metric of Diagonal(1 ./ γ),
#   x - γ .* a (⟨a, x⟩ - b) / Σᵢ γᵢaᵢ²,
# and onto a halfspace the same step when x violates the inequality
function prox!(y::AbstractArray, f::Union{IndHyperplane, IndHalfspace}, x::AbstractArray, γ::AbstractArray)
  _checksize(y, x)
  _checkparams(f, x)
  _checkparam(γ, x)
  T = float(eltype(y))
  if _isscaled(f)
    a = _normalized_normal(f.a, f.s)
    g = _dot_normal(a, x) - _scaled_offset(f)
    f isa IndHalfspace && (g = max(g, zero(g)))
    t = convert(T, g / _metric_sqnorm(f, x, γ))
    return y .= x .- t .* (γ .* a)
  end
  g = _dot_normal(f.a, x) - f.b
  f isa IndHalfspace && (g = max(g, zero(g)))
  t = convert(T, g / _metric_sqnorm(f, x, γ))
  y .= x .- t .* γ .* f.a
end

function prox!(y::AbstractArray, f::IndHyperslab, x::AbstractArray, γ::AbstractArray)
  _checkwrite(prox!, f, y, x)
  _checkparams(f, x)
  _checkparam(γ, x)
  Base.Broadcast.check_broadcast_axes(axes(y), γ)
  _hyperslab_input(f, x)
  isempty(x) && return y
  g = _hyperslab_residual(f, x)
  iszero(g) && return (y === x ? y : copyto!(y, x))
  t = g / _metric_sqnorm(f, x, γ)
  if _isscaled(f)
    return _hyperslab_metric_output!(y, x, _normalized_normal(f.a, f.s), γ, t)
  end
  _hyperslab_metric_output!(y, x, f.a, γ, t)
end

# Inline so that passing the scaled normal does not allocate a view.
@inline function _hyperslab_metric_output!(y, x, a, γ, t::T) where {T}
  y .= x .- t .* (T.(γ) .* a)
end

# The simplex and the ℓ1 ball: max(xᵢ - γᵢν, 0) and shrink(xᵢ, γᵢν) for the
# level ν that meets the constraint
function prox!(y::AbstractArray, f::IndSimplex, x::AbstractArray, γ::AbstractArray)
  _checksize(y, x)
  _checkparam(γ, x)
  T = float(eltype(y))
  r = convert(T, f.r)
  if !f.equality
    s = zero(T)
    for v in x
      s += max(v, zero(v))
    end
    s <= r && return (y .= max.(x, zero(T)))
  end
  iszero(r) && return fill!(y, zero(eltype(y)))
  ν = _metric_simplex_level(x, γ, r, identity)
  y .= max.(x .- γ .* ν, zero(T))
end

function prox!(y::AbstractArray, f::IndBallL1, x::AbstractArray, γ::AbstractArray)
  _checksize(y, x)
  _checkparam(γ, x)
  T = float(eltype(y))
  r = convert(T, f.r)
  norm(x, 1) <= r && return (y === x ? y : copyto!(y, x))
  iszero(r) && return fill!(y, zero(eltype(y)))
  ν = _metric_simplex_level(x, γ, r, abs)
  y .= shrink.(x, γ .* ν)
end

value_and_prox!(y::AbstractArray, f::Union{IndHyperplane, IndHalfspace, IndHyperslab, IndSimplex, IndBallL1}, x::AbstractArray, γ::AbstractArray) =
  (prox!(y, f, x, γ); zero(float(eltype(y))))

# The level ν with Σᵢ max(uᵢ - γᵢν, 0) = r for uᵢ = h(xᵢ), by Newton's method
# from the left on this convex, decreasing, piecewise linear function. From
# ν = -∞, where every entry is active, each step solves for ν on the active set,
# which then only shrinks, so ν increases to the root; the steps end when the
# active set stays the same, at the exact root, as in Michelot's algorithm for
# the simplex. Rounding can only stop it a step early.
function _metric_simplex_level(x::AbstractArray, γ::AbstractArray, r::T, h::H) where {T, H}
  ν = T(-Inf)
  last = typemax(Int)
  while true
    s, w, k = zero(T), zero(T), 0
    @inbounds for i in eachindex(IndexLinear(), x)
      u, g = convert(T, h(x[i])), convert(T, γ[i])
      if u > g * ν
        s += u
        w += g
        k += 1
      end
    end
    (k >= last || k == 0) && return ν
    last = k
    ν = (s - r) / w
  end
end

####################################
#  per-coordinate curvatures
####################################

"""
    coordinate_lipschitz(f, k)

A Lipschitz constant `Lₖ` of the partial derivative `∂f/∂xₖ` of the smooth
function `f` along coordinate `k`, a linear index of its argument: the curvature
of `f` along the coordinate, for which a coordinate step `1/Lₖ` decreases `f`.
It is the diagonal entry of the Hessian for the quadratic losses, a bound on it
for [`GLMLoss`](@ref), and `Inf` where none exists:

| `f` | `Lₖ` |
|:----|:-----|
| [`LeastSquaresLoss`](@ref) | `‖X[:, j]‖²` for the row `j` of entry `k` |
| [`GLMLoss`](@ref) | `curvature(ℓ) ⋅ Σᵢ wᵢ X[i, k]²`; `Inf` for `PoissonLoss` |
| [`QuadraticFunction`](@ref) | `A[k, k]` |
| [`KronQuadratic`](@ref) | `A[r, r] B[c, c]` for entry `(r, c)` of `Θ` |
| [`SqrNormL2`](@ref)`(λ)` | `λ` |
| [`SqrHingeLoss`](@ref)`(y, w)` | `2wₖyₖ²` |
| [`CubeNormL2`](@ref)`(λ)` | `Inf` for `λ > 0`, `0` for `λ = 0` |
| [`LogDetLoss`](@ref) | `Inf` |

`Scaled` scales the constants, `Tilt` and `Translate` keep them, and `Sum` adds
them. [`coordinate_lipschitz!`](@ref) computes all of them.

These constants are what coordinate descent needs. A gradient method that moves
all coordinates at once can use them as a diagonal metric `D = Diagonal(L)`,
with the steps `γ = t ./ L` per entry in [`prox!`](@ref), but the diagonal of a
Hessian does not majorize it: the scalar `t` must come from a line search or be
`1/λmax(D^(-1/2) ∇²f D^(-1/2))`, which standardizing the columns of `X` amounts
to.
"""
coordinate_lipschitz(f::AbstractFunction, ::Integer) =
  throw(ArgumentError(issmooth(f) ? "no per-coordinate Lipschitz constant is known for $(nameof(typeof(f)))" :
                                    "$(nameof(typeof(f))) is not smooth"))

"""
    coordinate_lipschitz!(L, f)

Fill `L`, an array of the size of the argument of `f`, with the per-coordinate
Lipschitz constants [`coordinate_lipschitz`](@ref)`(f, k)`, and return `L`. It
allocates nothing.
"""
function coordinate_lipschitz!(L::AbstractArray, f::AbstractFunction)
  @inbounds for k in eachindex(IndexLinear(), L)
    L[k] = coordinate_lipschitz(f, k)
  end
  L
end

# Σᵢ wᵢ X[i, j]², over the stored entries of a sparse X
function _column_sumsq(X::AbstractMatrix, j::Integer, w = nothing)
  T = float(eltype(X))
  s = zero(T)
  @inbounds for i in axes(X, 1)
    s += convert(T, _entry(w, i)) * abs2(X[i, j])
  end
  s
end
function _column_sumsq(X::SparseMatrixCSC, j::Integer, w = nothing)
  T = float(eltype(X))
  s = zero(T)
  rows, vals = rowvals(X), nonzeros(X)
  @inbounds for t in nzrange(X, j)
    s += convert(T, _entry(w, rows[t])) * abs2(vals[t])
  end
  s
end

coordinate_lipschitz(::ZeroFunction, ::Integer) = 0.0
coordinate_lipschitz(f::LinearFunction, ::Integer) = zero(float(eltype(f.c)))
coordinate_lipschitz(f::SqrNormL2, ::Integer) = float(f.λ)
function coordinate_lipschitz(f::SqrDistL2, k::Integer)
  isseparable(f) && (iszero(f.λ) || isconvex(f)) && return float(f.λ)
  invoke(coordinate_lipschitz, Tuple{AbstractFunction, Integer}, f, k)
end

function coordinate_lipschitz(f::SqrHingeLoss, k::Integer)
  y, w = promote(float(_entry(f.y, k)), float(_entry(f.w, k)))
  _sqrhinge_slope(y, w, abs(y), 0)
end

function coordinate_lipschitz!(L::AbstractArray, f::SqrHingeLoss)
  _checkparams(f, L)
  @inbounds for k in eachindex(IndexLinear(), L)
    L[k] = coordinate_lipschitz(f, k)
  end
  L
end
coordinate_lipschitz(f::CubeNormL2, ::Integer) = lipschitz(f)

function coordinate_lipschitz(f::QuadraticFunction{T}, k::Integer) where {T}
  checkbounds(f.A, k, k)
  convert(T, f.A[k, k])
end

function coordinate_lipschitz(f::KronQuadratic, k::Integer)
  p, q = size(f)
  checkbounds(1:p*q, k)
  r, c = mod1(k, p), cld(k, p)
  f.A[r, r] * f.B[c, c]
end

function coordinate_lipschitz(f::LeastSquaresLoss, k::Integer)
  p = size(f.X, 2)
  checkbounds(1:p*size(f.Y, 2), k)
  _column_sumsq(f.X, mod1(k, p))
end

# the squared norm of each column of X once, for every column of β
function coordinate_lipschitz!(L::AbstractArray, f::LeastSquaresLoss)
  p, m = size(f.X, 2), size(f.Y, 2)
  length(L) == p * m || throw(DimensionMismatch("L has $(length(L)) entries and β has $(p * m)"))
  @inbounds for j in 1:p
    s = _column_sumsq(f.X, j)
    for c in 1:m
      L[(c - 1) * p + j] = s
    end
  end
  L
end

function coordinate_lipschitz(f::GLMLoss{L, T}, k::Integer) where {L, T}
  checkbounds(axes(f.X, 2), k)
  c = curvature(f.loss)
  isinf(c) ? T(Inf) : convert(T, c) * _column_sumsq(f.X, k, f.weights)
end

coordinate_lipschitz(::LogDetLoss{T}, ::Integer) where {T} = T(Inf)
Base.@propagate_inbounds function coordinate_lipschitz(f::BernoulliLoss, k::Integer)
  w = float(_entry(f.w, k))
  iszero(w) ? zero(w) : oftype(w, Inf)
end

coordinate_lipschitz(s::Scaled, k::Integer) = s.α * coordinate_lipschitz(s.f, k)
coordinate_lipschitz!(L::AbstractArray, s::Scaled) = (coordinate_lipschitz!(L, s.f); L .*= s.α)
coordinate_lipschitz(t::Union{Tilt, Translate}, k::Integer) = coordinate_lipschitz(t.f, k)
coordinate_lipschitz!(L::AbstractArray, t::Union{Tilt, Translate}) = coordinate_lipschitz!(L, t.f)
coordinate_lipschitz(s::Sum, k::Integer) = sum(f -> coordinate_lipschitz(f, k), s.fs)

function coordinate_lipschitz(r::Regularize, k::Integer)
  r.c === nothing || checkbounds(r.c, k)
  coordinate_lipschitz(r.f, k) + r.ρ
end
function coordinate_lipschitz!(L::AbstractArray, r::Regularize)
  _checkparam(r.c, L)
  coordinate_lipschitz!(L, r.f)
  L .+= r.ρ
end
coordinate_lipschitz(h::_ScalarPrecompose, k::Integer) = h.μ * coordinate_lipschitz(h.f, k)
coordinate_lipschitz!(L::AbstractArray, h::_ScalarPrecompose) =
  (coordinate_lipschitz!(L, h.f); L .*= h.μ)

Base.@propagate_inbounds function coordinate_lipschitz(h::_ArrayPrecompose, k::Integer)
  a = abs(float(h.a[k]))
  _positive_product(coordinate_lipschitz(h.f, k), a, a)
end
function coordinate_lipschitz!(L::AbstractArray, h::_ArrayPrecompose)
  _checkparams(h, L)
  @inbounds for k in eachindex(IndexLinear(), L)
    L[k] = coordinate_lipschitz(h, k)
  end
  L
end
