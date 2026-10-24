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

# separable functions: the step of each entry; on a SymmetricSparseIterate the
# steps are a p × p array read at the lower triangle, so give a symmetric one.
# The loops are those of separable.jl with γ[i] for γ, kept apart so that the
# scalar loops stay exactly as they are
prox!(y::AbstractArray, f::SeparableFunction, x::AbstractArray, γ::AbstractArray) = _prox_sep!(y, f, x, γ)
value_and_prox!(y::AbstractArray, f::SeparableFunction, x::AbstractArray, γ::AbstractArray) =
  _isindicator(f) ? (_prox_sep!(y, f, x, γ); zero(float(eltype(y)))) : _value_sep(f, _prox_sep!(y, f, x, γ))

function _prox_sep!(y::AbstractArray, f, x::AbstractArray, γ::AbstractArray)
  _checksize(y, x)
  _checkparams(f, x)
  _checkparam(γ, x)
  T = eltype(y)
  @inbounds for i in eachindex(IndexLinear(), y, x)
    y[i] = convert(T, prox_entry(f, x[i], i, γ[i]))
  end
  y
end

function _prox_sep!(y::SparseIterate, f, x::SparseIterate, γ::AbstractArray)
  _checksize(y, x)
  _checkparams(f, x)
  _checkparam(γ, x)
  iszeropreserving(f) || return invoke(_prox_sep!, Tuple{AbstractArray, Any, AbstractArray, AbstractArray}, y, f, x, γ)
  T = eltype(y)
  if y !== x
    @inbounds for k in 1:y.nnz
      y.nzval[k] = zero(T)
    end
  end
  @inbounds for k in 1:x.nnz
    i = x.nzval2ind[k]
    _setstored!(y, convert(T, prox_entry(f, x.nzval[k], i, γ[i])), i)
  end
  y
end

function _prox_sep!(y::SymmetricSparseIterate, f, x::AbstractMatrix, γ::AbstractArray)
  _checksize(y, x)
  _checkparams(f, x)
  _checkparam(γ, x)
  T = eltype(y)
  p = y.p
  if x isa SymmetricSparseIterate && iszeropreserving(f)
    if y !== x
      @inbounds for k in 1:y.nnz
        y.nzval[k] = zero(T)
      end
    end
    @inbounds for k in 1:x.nnz
      i = x.nzval2ind[k]
      r, c = tril_cartesian(p, i)
      j = (c - 1) * p + r
      _setstored!(y, convert(T, prox_entry(f, x.nzval[k], j, γ[j])), i)
    end
    return y
  end
  @inbounds for c in 1:p, r in c:p
    j = (c - 1) * p + r
    _setstored!(y, convert(T, prox_entry(f, x[r, c], j, γ[j])), tril_index(p, r, c))
  end
  y
end

# group norms: a step per group, constant on the group, which keeps the closed form
value_and_prox!(y::AbstractArray, f::AbstractGroupNorm, x::AbstractArray, γ::AbstractVector) = _value_and_prox_groups!(y, f, x, γ)
prox!(y::AbstractArray, f::AbstractGroupNorm, x::AbstractArray, γ::AbstractVector) = (_value_and_prox_groups!(y, f, x, γ); y)
value_and_prox!(y::AtomIterate, f::AbstractGroupNorm, x::AtomIterate, γ::AbstractVector) = _value_and_prox_groups!(y, f, x, γ)

# the entrywise reciprocal of an array of steps, without allocating
struct _ReciprocalArray{T, N, A<:AbstractArray} <: AbstractArray{T, N}
  γ::A
end
_ReciprocalArray(γ::AbstractArray{T, N}) where {T, N} = _ReciprocalArray{float(T), N, typeof(γ)}(γ)
Base.size(a::_ReciprocalArray) = size(a.γ)
Base.IndexStyle(::Type{<:_ReciprocalArray}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(a::_ReciprocalArray{T}, i::Int) where {T} = inv(convert(T, a.γ[i]))

# functions built from functions pass the steps on, entry by entry
prox!(y, s::Scaled, x, γ::AbstractArray) = prox!(y, s.f, x, _ScaledArray(γ, s.α))
value_and_prox!(y, s::Scaled, x, γ::AbstractArray) = s.α * value_and_prox!(y, s.f, x, _ScaledArray(γ, s.α))

function prox!(y::AbstractArray, t::Tilt, x::AbstractArray, γ::AbstractArray)
  _checksize(y, x)
  _checkparam(γ, x)
  y .= x .- γ .* t.c
  prox!(y, t.f, y, γ)
end
function value_and_prox!(y::AbstractArray, t::Tilt, x::AbstractArray, γ::AbstractArray)
  _checksize(y, x)
  _checkparam(γ, x)
  y .= x .- γ .* t.c
  value_and_prox!(y, t.f, y, γ) + dot(t.c, y) + t.d
end

function prox!(y::AbstractArray, t::Translate, x::AbstractArray, γ::AbstractArray)
  _checksize(y, x)
  y .= x .- t.b
  prox!(y, t.f, y, γ)
  y .+= t.b
end
function value_and_prox!(y::AbstractArray, t::Translate, x::AbstractArray, γ::AbstractArray)
  _checksize(y, x)
  y .= x .- t.b
  v = value_and_prox!(y, t.f, y, γ)
  y .+= t.b
  v
end

# Moreau's identity in each entry: prox_{γᵢfᵢ*}(xᵢ) = xᵢ - γᵢ prox_{fᵢ/γᵢ}(xᵢ/γᵢ)
function prox!(y::AbstractArray, c::Conjugate, x::AbstractArray, γ::AbstractArray)
  _checksize(y, x)
  _checkparam(γ, x)
  T = float(eltype(y))
  z = _array_buffer!(c.scratch, T, size(x))
  z .= x ./ γ
  prox!(z, c.f, z, _ReciprocalArray(γ))
  y .= x .- γ .* z
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
  checkbounds(f.X, 1, k)
  c = curvature(f.loss)
  isinf(c) ? T(Inf) : convert(T, c) * _column_sumsq(f.X, k, f.weights)
end

coordinate_lipschitz(::LogDetLoss{T}, ::Integer) where {T} = T(Inf)

coordinate_lipschitz(s::Scaled, k::Integer) = s.α * coordinate_lipschitz(s.f, k)
coordinate_lipschitz!(L::AbstractArray, s::Scaled) = (coordinate_lipschitz!(L, s.f); L .*= s.α)
coordinate_lipschitz(t::Union{Tilt, Translate}, k::Integer) = coordinate_lipschitz(t.f, k)
coordinate_lipschitz!(L::AbstractArray, t::Union{Tilt, Translate}) = coordinate_lipschitz!(L, t.f)
coordinate_lipschitz(s::Sum, k::Integer) = sum(f -> coordinate_lipschitz(f, k), s.fs)
