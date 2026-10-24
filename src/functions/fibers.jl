####################################
#
#  Functions of stacked arrays: slice by slice and fiber by fiber
#
####################################

# The joint estimation of K related matrices, such as the precision matrices of
# K classes or time points, works on an m × n × K array: SliceSum gives the
# function of each matrix, the smooth part, and FiberSum the penalty on each entry
# across the matrices. ProximalSolvers' ADMMSolver takes such an array.

# the k-th slice along the last dimension, and the fiber at I along it
_slice(x::AbstractArray{<:Any, N}, k::Integer) where {N} = view(x, ntuple(_ -> Colon(), Val(N - 1))..., k)
Base.@propagate_inbounds _fiber(x::AbstractArray, I::CartesianIndex) = view(x, I, :)

# y[j, i, :] = y[i, j, :]
function _mirror!(y::AbstractArray{<:Any, 3}, i::Int, j::Int)
  @inbounds for k in axes(y, 3)
    y[j, i, k] = y[i, j, k]
  end
  y
end

####################################
#  slices
####################################

"""
    SliceSum(f₁, …, f_K)
    SliceSum(fs::AbstractVector)

The function `Σₖ fₖ(xₖ)` of an array `x` whose slices along the last dimension
are `x₁, …, x_K`: the matrices `x[:, :, k]` of an `m × n × K` array, or the
columns of a matrix. The value, the gradient and the proximal operator act slice
by slice, on views, and allocate nothing beyond what the `fₖ` do. `SliceSum` has
the traits that all the `fₖ` share. The `fₖ` are given as arguments, or as a
vector when there are many of one type.

With `fₖ = LogDetLoss(Sₖ, nₖ/n)` it is the smooth part of the joint graphical
lasso of K classes, whose penalty on the entries across the classes is a
[`FiberSum`](@ref).
"""
struct SliceSum{F<:Union{Tuple, AbstractVector}} <: AbstractFunction
  fs::F

  function SliceSum{F}(fs::F) where {F<:Union{Tuple, AbstractVector}}
    isempty(fs) && throw(ArgumentError("SliceSum needs a function for each slice, got none"))
    new{F}(fs)
  end
end
SliceSum(fs::Union{Tuple, AbstractVector}) = SliceSum{typeof(fs)}(fs)
SliceSum(fs::AbstractFunction...) = SliceSum(fs)

for trait in (:issmooth, :isproximable, :isconvex)
  @eval $trait(::Type{SliceSum{F}}) where {F<:Tuple} = all($trait, fieldtypes(F))
  @eval $trait(::Type{SliceSum{F}}) where {F<:AbstractVector} = $trait(eltype(F))
end

function _checkslices(s::SliceSum, x::AbstractArray)
  size(x, ndims(x)) == length(s.fs) ||
    throw(DimensionMismatch("$(length(s.fs)) functions for $(size(x, ndims(x))) slices"))
end

# Σₖ fun(fₖ, k): unrolled over a tuple, so that each fₖ has its own type, and a
# loop over a vector
_sumslices(fun::G, fs::Tuple) where {G} = sum(ntuple(k -> fun(fs[k], k), Val(length(fs))))
function _sumslices(fun::G, fs::AbstractVector) where {G}
  v = fun(fs[1], 1)
  for k in 2:length(fs)
    v += fun(fs[k], k)
  end
  v
end

(s::SliceSum)(x::AbstractArray) = (_checkslices(s, x); _sumslices((f, k) -> f(_slice(x, k)), s.fs))

function value_and_prox!(y::AbstractArray, s::SliceSum, x::AbstractArray, γ::Real)
  _checksize(y, x)
  _checkslices(s, x)
  _sumslices((f, k) -> value_and_prox!(_slice(y, k), f, _slice(x, k), γ), s.fs)
end
prox!(y::AbstractArray, s::SliceSum, x::AbstractArray, γ::Real) = (value_and_prox!(y, s, x, γ); y)

function value_and_gradient!(g::AbstractArray, s::SliceSum, x::AbstractArray)
  _checksize(g, x)
  _checkslices(s, x)
  _sumslices((f, k) -> value_and_gradient!(_slice(g, k), f, _slice(x, k)), s.fs)
end

lipschitz(s::SliceSum) = maximum(lipschitz, s.fs)

####################################
#  fibers
####################################

"""
    FiberSum(f; diagonal = f, symmetric = false)

The function `Σᵢⱼ f(x[i, j, :])` of an `m × n × K` array `x`: a vector function
`f` of each fiber across the K slices, as in the penalties of K related
precision matrices that make them share a sparsity pattern or be close.
`diagonal` is the function of the fibers `x[i, i, :]` of square slices:
`nothing` leaves them unpenalized, and another function gives them their own
rule. The proximal operator applies the proximal operator of `f` to each fiber,
on views, and allocates nothing beyond what that of `f` does. On a matrix, the
fibers are its rows.

With `symmetric = true` the slices are taken to be symmetric, as precision
matrices are: only the fibers with `i ≥ j` are read, and the proximal operator
copies each result to the fiber `(j, i)`, which halves the work and keeps the
result exactly symmetric. The value counts each off-diagonal fiber twice.

With `x[:, :, k]` the precision matrix of class or time point `k`:

| penalty | `f` |
|---|---|
| a shared sparsity pattern | `NormL2(λ)` |
| the group graphical lasso | `SparseGroupNormL2(λ1, λ2, [1:K], [1.0])` |
| simultaneous selection | `NormLinf(λ)` |
| the fused graphical lasso of K classes, all pairs fused | `NormL1Fused(λ1, λ2)` |
| graphs over time with jumps, neighbors fused | `FusedLasso(λ1, λ2)` or `TotalVariation1D(λ)` |

and `diagonal = nothing` leaves the diagonal free, `diagonal =
NormL1Fused(0, λ2)` keeps only the fusion on it.
"""
struct FiberSum{F<:AbstractFunction, D<:Union{Nothing, AbstractFunction}} <: AbstractFunction
  f::F
  diagonal::D
  symmetric::Bool
end
FiberSum(f::AbstractFunction; diagonal::Union{Nothing, AbstractFunction} = f, symmetric::Bool = false) =
  FiberSum(f, diagonal, symmetric)

for trait in (:isproximable, :isconvex)
  @eval $trait(::Type{FiberSum{F, D}}) where {F, D} = $trait(F) && (D === Nothing || $trait(D))
end

_square_slices(x::AbstractArray, why) = ndims(x) == 3 && size(x, 1) == size(x, 2) ||
  throw(DimensionMismatch("$why needs square slices, an n × n × K array; got size $(size(x))"))

# whether the diagonal fibers take their own rule; that needs square slices
function _owndiagonal(fb::FiberSum, x::AbstractArray)
  fb.symmetric && _square_slices(x, "symmetric = true")
  fb.diagonal === fb.f && return false
  _square_slices(x, "a rule for the diagonal")
  true
end

# the fibers to read: all, or those with i ≥ j of symmetric slices
_fiberindices(fb::FiberSum, x::AbstractArray) = CartesianIndices(Base.front(axes(x)))

_isdiagonal(I::CartesianIndex{2}) = I[1] == I[2]
_isdiagonal(::CartesianIndex) = false

function (fb::FiberSum)(x::AbstractArray)
  own = _owndiagonal(fb, x)
  T = float(eltype(x))
  v = zero(T)
  for I in _fiberindices(fb, x)
    fb.symmetric && I[1] < I[2] && continue
    xv = @inbounds _fiber(x, I)
    if own && _isdiagonal(I)
      fb.diagonal === nothing || (v += fb.diagonal(xv))
    else
      w = fb.f(xv)
      v += fb.symmetric && !_isdiagonal(I) ? 2w : w
    end
  end
  v
end

value_and_prox!(y::AbstractArray, fb::FiberSum, x::AbstractArray, γ::Real) = _fiberprox!(y, fb, x, γ, Val(true))
prox!(y::AbstractArray, fb::FiberSum, x::AbstractArray, γ::Real) = (_fiberprox!(y, fb, x, γ, Val(false)); y)

_fiberprox!(y::AbstractArray, fb::FiberSum, x::AbstractArray, γ::Real, v::Val) = _fiberprox_views!(y, fb, x, γ, v)

# the proximal operator fiber by fiber, through views, and the value at the
# result if V
function _fiberprox_views!(y::AbstractArray, fb::FiberSum, x::AbstractArray, γ::Real, ::Val{V}) where {V}
  _checksize(y, x)
  own = _owndiagonal(fb, x)
  T = float(eltype(y))
  v = zero(T)
  for I in _fiberindices(fb, x)
    fb.symmetric && I[1] < I[2] && continue
    yv, xv = @inbounds(_fiber(y, I)), @inbounds(_fiber(x, I))     # I is in the axes of x and y
    if own && _isdiagonal(I)
      v += _prox_diagonal!(yv, fb.diagonal, xv, γ, T, Val(V))
    else
      w = V ? T(value_and_prox!(yv, fb.f, xv, γ)) : (prox!(yv, fb.f, xv, γ); zero(T))
      if fb.symmetric && !_isdiagonal(I)
        _mirror!(y, I[1], I[2])
        w *= 2
      end
      v += w
    end
  end
  v
end

_prox_diagonal!(y, ::Nothing, x, γ, ::Type{T}, ::Val) where {T} = (y === x || copyto!(y, x); zero(T))
_prox_diagonal!(y, d::AbstractFunction, x, γ, ::Type{T}, ::Val{true}) where {T} = T(value_and_prox!(y, d, x, γ))
_prox_diagonal!(y, d::AbstractFunction, x, γ, ::Type{T}, ::Val{false}) where {T} = (prox!(y, d, x, γ); zero(T))

# NormL1Fused on the pairs of two stacked matrices, the fused graphical lasso of
# two classes: the same proximal operator, entry by entry without views, which
# costs about as much per fiber as one call of _prox_l1fused. Other numbers of
# slices go through the views.
function _fiberprox!(y::AbstractArray{<:Any, 3}, fb::FiberSum{<:NormL1Fused, D}, x::AbstractArray{<:Any, 3},
                     γ::Real, ::Val{V}) where {D<:Union{Nothing, NormL1Fused}, V}
  _checksize(y, x)
  size(x, 3) == 2 || return _fiberprox_views!(y, fb, x, γ, Val(V))
  own = _owndiagonal(fb, x)
  T = float(eltype(y))
  a, b = firstindex(x, 3), firstindex(x, 3) + 1
  f, γT = fb.f, T(γ)
  v = zero(T)
  @inbounds for j in axes(x, 2), i in axes(x, 1)
    fb.symmetric && i < j && continue
    if own && i == j
      d = fb.diagonal
      if d === nothing
        y[i, i, a], y[i, i, b] = x[i, i, a], x[i, i, b]
      else
        u, w = _prox_l1fused(T(x[i, i, a]), T(x[i, i, b]), T(d.λ1), T(d.λ2), γT)
        y[i, i, a], y[i, i, b] = u, w
        V && (v += T(d.λ1) * (abs(u) + abs(w)) + T(d.λ2) * abs(u - w))
      end
    else
      u, w = _prox_l1fused(T(x[i, j, a]), T(x[i, j, b]), T(f.λ1), T(f.λ2), γT)
      y[i, j, a], y[i, j, b] = u, w
      c = fb.symmetric && i != j ? 2 : 1
      if c == 2
        y[j, i, a], y[j, i, b] = u, w
      end
      V && (v += c * (T(f.λ1) * (abs(u) + abs(w)) + T(f.λ2) * abs(u - w)))
    end
  end
  v
end
