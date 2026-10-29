"""
    reset_cache!(f)

Invalidate data-dependent caches of `f` and return `f`. Call this after changing
the entries of data stored by reference in a [`LeastSquaresLoss`](@ref) (`X`
or `Y`) or [`QuadraticFunction`](@ref) (`A`). Array sizes must stay unchanged;
construct a new function for different dimensions. The next proximal call
rebuilds the decomposition and may allocate; subsequent calls reuse it.

Call this on the outermost function you use: it descends through compositions,
and an affine-constrained loss also resets the cache of its constrained solve.
If several wrappers share the same loss, reset each wrapper after a data
change. Values and gradients read the data directly and need no reset.

An [`IndAffine`](@ref) must instead be reconstructed when its data change:
its factorization type depends on the rank of `A`. For an affine-constrained
loss, `reset_cache!` supports changes to the loss data with the set unchanged.
Other functions without data-dependent caches return themselves. Custom
functions that cache derived data should implement this method.
"""
reset_cache!(f::AbstractFunction) = f

function reset_cache!(f::LeastSquaresLoss{T, Ty, Tx, N}) where {T, Ty, Tx, N}
  size(f.tmp) == size(f.Y) && size(f.X, 1) == size(f.Y, 1) ||
    throw(DimensionMismatch("data sizes changed; construct a new LeastSquaresLoss"))
  f.svd_ready = false
  f.XtY = Array{T, N}(undef, ntuple(_ -> 0, N))
  f
end

function reset_cache!(f::QuadraticFunction)
  size(f.A) == (length(f.tmp), length(f.tmp)) &&
    (f.b === nothing || length(f.b) == length(f.tmp)) ||
    throw(DimensionMismatch("data sizes changed; construct a new QuadraticFunction"))
  empty!(f.Λ)
  f
end

function reset_cache!(c::AffineConstrained{T}) where {T}
  reset_cache!(c.f)
  empty!(c.t)
  c.N = Matrix{T}(undef, 0, 0)
  c.γ = T(NaN)
  c
end

reset_cache!(::IndAffine) = throw(ArgumentError("construct a new IndAffine after changing its data"))

for F in (:Scaled, :Tilt, :Translate, :Conjugate, :Precompose, :_ArrayPrecompose, :Regularize,
          :MoreauEnvelope, :SqrDistL2, :Spectral, :SingularValue, :BallConstrained,
          :SphereConstrained, :BoxConstrained, :LinearConstrained)
  @eval reset_cache!(f::$F) = (reset_cache!(f.f); f)
end

for F in (:Sum, :SeparableSum, :SliceSum)
  @eval reset_cache!(s::$F) = (foreach(reset_cache!, s.fs); s)
end

function reset_cache!(s::FiberSum)
  reset_cache!(s.f)
  s.diagonal === nothing || s.diagonal === s.f || reset_cache!(s.diagonal)
  s
end

function reset_cache!(s::IndIntersection)
  foreach(reset_cache!, s.sets)
  s.buffers = nothing
  s.iterations = 0
  s.converged = false
  s
end
