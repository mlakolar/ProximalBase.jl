"""
    reset_cache!(f)

Invalidate data-dependent caches after changing entries of LeastSquaresLoss
`X` or `Y`, or QuadraticFunction `A`, stored by reference. Array sizes must stay
unchanged; construct a new function for different dimensions. The next proximal
call rebuilds the decomposition. Values and gradients read data directly.
Call this on the outermost Scaled wrapper. Functions without caches return
themselves; custom functions with derived data should implement this hook.
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

reset_cache!(f::Scaled) = (reset_cache!(f.f); f)
