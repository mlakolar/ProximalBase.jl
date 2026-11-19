####################################
#
#  Least squares  ‖Y - Xβ‖² / 2
#
####################################

"""
    LeastSquaresLoss(Y, X)

Least-squares loss `f(β) = ‖Y-Xβ‖²/2` with gradient `X'*(X*β-Y)`.
`X` is an `n × p` design matrix. A length-`n` response `Y` uses length-`p`
coefficients; an `n × k` response uses `p × k` coefficients and the Frobenius norm.
Data and arguments must be real and one-based.

The loss is smooth, convex and proximable. Dense arrays, [`SparseIterate`](@ref)
and [`BlockIterate`](@ref) use their supported array products. The first proximal
call caches a thin SVD; later calls reuse it for different steps, including when
`p > n`.

# Examples
```jldoctest
julia> f = LeastSquaresLoss([1.0, -1.0], [1.0 0.0; 0.0 1.0]);

julia> f([3.0, 1.0])
4.0

julia> gradient(f, [3.0, 1.0])
2-element Vector{Float64}:
 2.0
 2.0
```

# Edge cases
- `X` and `Y` are retained by reference. After same-shape changes, call
  [`reset_cache!`](@ref) before another prox; reconstruct after dimension changes.
- Values, gradients and prox reuse private scratch: synchronize concurrent calls
  on one loss. Supplied-predictor oracles have separate ownership rules.
- No symmetric-iterate or scalar-coordinate prox is provided.
- Value range recovery does not guarantee recovery of matrix products or SVDs.

See [least-squares operations](@ref Least-squares-loss) and
[supplied-predictor oracles](@ref Least-squares-at-a-predictor-or-residual-that-the-caller-holds).
"""
mutable struct LeastSquaresLoss{T<:AbstractFloat, Ty<:AbstractVecOrMat, Tx<:AbstractMatrix, N, A<:AbstractArray{T, N}} <: AbstractFunction
  Y::Ty
  X::Tx
  tmp::A                  # Xβ - Y, next to Y; calls to f(β) do not allocate, except to
                          # recover a value whose product Xβ overflows (a copy of β)
  # the proximal operator: X = U Diagonal(√s2) V', computed on first use
  V::Matrix{T}
  s2::Vector{T}
  sUtY::Array{T, N}       # Diagonal(s) U'Y, the right-hand side in the V basis
  w::Array{T, N}
  output::Scratch         # dense output for a non-strided destination
  svd_ready::Bool

  function LeastSquaresLoss{T, Ty, Tx, N, A}(Y, X, tmp, V, s2, sUtY, w, output, svd_ready) where {T<:AbstractFloat,
      Ty<:AbstractVecOrMat, Tx<:AbstractMatrix, N, A<:AbstractArray{T, N}}
    _check_one_based(Y, X, tmp, V, s2, sUtY, w)
    all(a -> a === nothing || eltype(a) <: Real,
        (Y, X, tmp, V, s2, sUtY, w)) || _real_error(:LeastSquaresLoss)
    new{T, Ty, Tx, N, A}(Y, X, tmp, V, s2, sUtY, w, output, svd_ready)
  end
end
LeastSquaresLoss(Y::Ty, X::Tx, tmp::A, V::Matrix{T}, s2::Vector{T}, sUtY::Array{T, N},
    w::Array{T, N}, output::Scratch, svd_ready::Bool) where {T<:AbstractFloat, Ty<:AbstractVecOrMat,
    Tx<:AbstractMatrix, N, A<:AbstractArray{T, N}} =
  LeastSquaresLoss{T, Ty, Tx, N, A}(Y, X, tmp, V, s2, sUtY, w, output, svd_ready)

function LeastSquaresLoss(Y::AbstractVecOrMat, X::AbstractMatrix)
  eltype(Y) <: Real || _real_error(:LeastSquaresLoss)
  eltype(X) <: Real || _real_error(:LeastSquaresLoss)
  size(X, 1) == size(Y, 1) || throw(DimensionMismatch("X has $(size(X, 1)) rows and Y has $(size(Y, 1))"))
  T = float(promote_type(eltype(Y), eltype(X)))
  N = ndims(Y)
  tmp = zeros(T, size(Y))
  LeastSquaresLoss{T, typeof(Y), typeof(X), N, typeof(tmp)}(Y, X, tmp, Matrix{T}(undef, 0, 0), T[],
                                                            Array{T, N}(undef, ntuple(_ -> 0, N)),
                                                            Array{T, N}(undef, ntuple(_ -> 0, N)), Scratch(), false)
end

issmooth(::Type{<:LeastSquaresLoss}) = true
_argument_type(::Type{<:LeastSquaresLoss}) = AbstractVecOrMat
isproximable(::Type{<:LeastSquaresLoss}) = true
isconvex(::Type{<:LeastSquaresLoss}) = true

function (f::LeastSquaresLoss{T})(β::AbstractVecOrMat) where {T}
  _check_one_based(β)
  _checkreal(f, β)
  mul!(f.tmp, f.X, β)
  v = _ls_value(f, f.tmp, f.Y)
  _value_at(isfinite(v) ? v : _leastsquares_image_overflow(f, β, v), β)
end

function value_and_gradient!(g::AbstractVecOrMat, f::LeastSquaresLoss{T}, β::AbstractVecOrMat) where {T}
  _checkwrite(value_and_gradient!, f, g, β)
  mul!(f.tmp, f.X, β)
  v = _leastsquares_residual_gradient!(g, f.tmp, f.X, f.tmp, f.Y, f)
  _value_at(isfinite(v) ? v : _leastsquares_image_overflow(f, β, v), β)
end

# d = z - Y and g = X'd, with one product by X'. d may be the same array as z (the residual then replaces the
# predictor, as in the methods of the loss, which pass f.tmp for both); otherwise g and d must not overlap X, Y,
# z or each other, which the predictor oracles check before they call it.
@inline function _leastsquares_residual_gradient!(g, d, X, z, Y)
  d .= z .- Y
  mul!(g, transpose(X), d)
  g
end
# The same, returning also the value α‖d‖²/2 (_ls_value, with f the loss or its Scaled).
@inline function _leastsquares_residual_gradient!(g, d, X, z, Y, f)
  _leastsquares_residual_gradient!(g, d, X, z, Y)
  _ls_value(f, d, nothing)
end

# The value α Σ (xᵢ - yᵢ)² / 2 (α Σ xᵢ² / 2 for y === nothing) of the least-squares methods and oracles, with f the
# loss or its Scaled, α = 1 for a loss and the factor as supplied for a Scaled (_ls_base_factor), and y nothing or the
# response Y of the loss. The ordinary path below takes a strided Float32 or Float64 vector x with y === nothing or
# a strided vector y of the same element type. Other arguments take the squared-residual kernel _square_value_wide:
# matrices, arrays that are not strided, and mixed element types (a loss with Float32 Y and Float64 X has Float64
# scratch, so that f(β) passes a Float64 x and a Float32 y).
@inline _ls_value(f, x, y) = _square_value_wide(_ls_base_factor(f)[2], x, y)
@inline function _ls_value(f, x::StridedVector{T}, ::Nothing) where {T<:Union{Float32, Float64}}
  long = length(x) > _LS_ONE_LOOP_MAX
  _ls_square_value(f, x, nothing, long ? zero(T) : _simd_sumabs2(x), long)
end
@inline function _ls_value(f, x::StridedVector{T}, y::StridedVector{T}) where {T<:Union{Float32, Float64}}
  axes(x) == axes(y) || _ls_axes_error("the predictor", axes(x, 1), axes(y, 1))
  long = length(x) > _LS_ONE_LOOP_MAX
  _ls_square_value(f, x, _LSResponse(), long ? zero(T) : _ls_simd_sumabs2_diff(x, y), long)
end

# Σ (xᵢ - yᵢ)² for vectors with the same axes (an offset view has other axes than a vector of the same length; the
# loop reads y at the indices of x): one vectorizing loop up to _LS_ONE_LOOP_MAX entries, Julia's pairwise reduction
# over the indices above.
@inline function _ls_sumabs2_diff(x::StridedVector{T}, y::StridedVector{T}) where {T<:Union{Float32, Float64}}
  axes(x) == axes(y) || _ls_axes_error("the predictor", axes(x, 1), axes(y, 1))
  length(x) > _LS_ONE_LOOP_MAX ? _ls_sumabs2_diff_pairwise(x, y) : _ls_simd_sumabs2_diff(x, y)
end
@inline function _ls_simd_sumabs2_diff(x::StridedVector{T}, y::StridedVector{T}) where {T<:Union{Float32, Float64}}
  s = zero(T)
  @inbounds @simd for i in eachindex(x)
    s += abs2(x[i] - y[i])
  end
  s
end
_ls_sumabs2_diff_pairwise(x, y) = mapreduce(i -> @inbounds(abs2(x[i] - y[i])), +, eachindex(x))
@noinline _ls_axes_error(name, ax, ay) =
  throw(DimensionMismatch("$name has axes $ax and the response Y has axes $ay; they must be equal"))

# The ordinary path from the sum s of the n squares in one vectorizing loop, n = length(x) ≤ _LS_ONE_LOOP_MAX (long
# is n > _LS_ONE_LOOP_MAX, where s is not formed here). The kernel keeps the product α s / 2, formed in the promoted
# type R of α and T, when it is finite and s ≥ n floatmin(T)/eps(T) (_square_value_level). Here the product is kept
# when K ≤ s ≤ floatmax(T) for K = 2^63 floatmin(T)/eps(T) (_ls_ordinary_sum) and, for a Scaled, α s < floatmax(T)
# (_ls_product_fits; for a loss α = 1 and α s = s ≤ floatmax(T)). Then α s / 2 is finite, and K is at least
# n floatmin(T)/eps(T) for n ≤ typemax(Int) (Int to T rounds n to at most 2^63): where the test holds, the kernel's
# test holds too. The kept value α s / 2 is at most floatmax(T)/2: a Scaled keeps α s < floatmax(T), and a loss also
# keeps s = floatmax(T), whose value is exactly floatmax(T)/2. The computed sum of the n squares is within a relative
# γ_m of the exact sum of squares of the entries, γ_m = m u / (1 - m u) with u = eps(T)/2. For the loop, m = n + 2 for
# differences (the rounding of a difference enters its square twice, then the rounding of the square and n - 1
# additions) and m = n for the entries of a residual; n ≤ 2^17, so γ_m < 0.008 in Float32. For the pairwise reduction
# of a long vector, a term passes at most 1023 additions inside its block of at most 1024 entries and at most
# ⌈log2 n⌉ additions in the tree above the blocks, so m = 1026 + ⌈log2 n⌉ bounds it (1024 + ⌈log2 n⌉ for a
# residual). With the rounding of α s, the exact value is then below floatmax(T), so that a kept value is not a value
# that overflows T. Long vectors and every other sum, a NaN s or α s included, go to
# _ls_value_cold, the one call on this path.
@inline function _ls_square_value(f, x::StridedVector{T}, y, s::T, long::Bool) where {T<:Union{Float32, Float64}}
  _, α = _ls_base_factor(f)
  p = α * s
  (!long & _ls_ordinary_sum(s) & _ls_product_fits(f, p, T)) && return _value_at(p / 2, x)
  _ls_value_cold(f, x, y, s)
end
# A long vector: its sum by the pairwise reduction (_sumabs2_loop, _ls_sumabs2_diff), and the test above on it. Then
# the exceptional path.
@noinline function _ls_value_cold(f, x::StridedVector{T}, y, s::T) where {T}
  if length(x) > _LS_ONE_LOOP_MAX
    base, α = _ls_base_factor(f)
    s = y === nothing ? _sumabs2_loop(x) : _ls_sumabs2_diff(x, _ls_response(base, y))
    p = α * s
    (_ls_ordinary_sum(s) & _ls_product_fits(f, p, T)) && return _value_at(p / 2, x)
  end
  _ls_value_exceptional(f, x, y, s)
end
# K ≤ s ≤ floatmax(T) with K = 2^63 floatmin(T)/eps(T), on the bits of s: with m = precision(T) - 1 significand bits
# and the largest exponent E = exponent(floatmax(T)), the biased exponent of K is 64 + m (2^-907 for Float64, 2^-40
# for Float32) and that of floatmax(T) is 2E; the shifted bits of a zero or subnormal s, an infinite or NaN s and a
# negative s fall outside that range.
@inline function _ls_ordinary_sum(s::T) where {T<:Union{Float32, Float64}}
  m = precision(T) - 1
  e = reinterpret(Unsigned, s) >> m
  lo = 64 + m
  e - (lo % typeof(e)) <= (2 * exponent(floatmax(T)) - lo) % typeof(e)
end
# The second operand of the kernel on the exceptional path: nothing, or _LSResponse() for the response Y of the loss,
# which _ls_value_exceptional loads from f. The ordinary path then passes to the call no array that it loaded from a
# field of the loss, which would need a GC root in the caller.
struct _LSResponse end
_ls_response(base, ::Nothing) = nothing
_ls_response(base, ::_LSResponse) = base.Y
# After the check of the scale of an oracle (_ls_base_scale, which refuses the infinite factor that _ls_base_factor
# does not check): where the sum s is finite and α s is not below floatmax(T) (the value would be at least
# floatmax(T)/2), the squares are recomputed compensated with α folded in (_scaled_square_value), in the promoted type
# of α and T. The kernel's test (_square_value_level) would accept the same s again there, also where its error
# decides between a finite value and Inf. The other sums (not finite, or below K with α s below floatmax(T)) go to
# the kernel's test and recomputation.
@noinline function _ls_value_exceptional(f, x, y, s)
  base, α = _ls_base_scale(f)
  r = _ls_response(base, y)
  p = α * s
  (isfinite(s) && !(p < floatmax(eltype(x)))) && return _scaled_square_value(α, x, r, p / 2)
  _square_value_level(α, α, x, r, s, true)
end

# The value of f(β) or value_and_gradient! when the result v of the squared-residual kernel is not finite. f.tmp
# holds Xβ (f(β)) or Xβ - Y (value_and_gradient!). Where every entry of f.tmp is finite, v stands: the kernel has
# recomputed the squares of a finite residual, halving a difference that overflows. Where an entry is not finite
# although β, X and Y are, Xβ is formed again as 2^k X βs from βs = β 2^-k, with k chosen so that no entry of X βs
# can overflow. An entry 2^k (X βs)ᵢ that is representable meets Yᵢ in its own units, so that a small Yᵢ survives a
# product that cancels; the others are formed as (X βs)ᵢ - Yᵢ 2^-k in units of 2^k. The two sums of squares are added
# by their exponents. This allocates a copy of β and two arrays of the size of Y.
@noinline function _leastsquares_image_overflow(f::LeastSquaresLoss{T}, β, v) where {T}
  r, Y = f.tmp, f.Y
  _all_finite(r) && return v
  (_all_finite(β) && _all_finite(f.X) && _all_finite(Y)) || return v
  k = max(_product_sum_shift(T, _stored_maxabs(f.X), _stored_maxabs(β), size(f.X, 2), 3),
          _bound_exponent(_stored_maxabs(Y)) + 3 - _max_exponent(T))
  sc, up = _power_factors(T, -k), _power_factors(T, k)
  mul!(r, f.X, β .* sc[1] .* sc[2])
  lim = _scale_value(floatmax(T) / 4, sc)
  own = _leastsquares_own.(r, lim, Ref(up))                  # 2^k (X βs)ᵢ where representable
  Yo = _leastsquares_own_y.(r, Y, lim, T)
  r .= _leastsquares_big.(r, Y, lim, Ref(sc))
  acc = _expsum_add(_expsum_start(T), _leastsquares_squares(own, Yo)...)
  acc = _expsum_add(acc, _leastsquares_squares(r, nothing, 2k)...)
  _expsum_value(T, acc)
end
@inline _leastsquares_own(a::T, lim, up) where {T} = abs(a) <= lim ? _scale_value(a, up) : zero(T)
@inline _leastsquares_own_y(a, y, lim, ::Type{T}) where {T} = abs(a) <= lim ? convert(T, y) : zero(T)
@inline _leastsquares_big(a::T, y, lim, sc) where {T} = abs(a) <= lim ? zero(T) : a - _scale_value(convert(T, y), sc)
# 2^shift ‖x - y‖²/2 as a normalized significand and an exponent, for finite entries
function _leastsquares_squares(x, y, shift = 0)
  s, k = _residual_squares(x, y, eltype(x))
  m, e = _frexp(s)
  ex = _exponent_sum(eltype(x), e, 2k - 1 + shift)
  # _ExpSum stores an Int exponent. For these nonnegative terms, saturation
  # keeps a BigFloat exponent outside Int on the same side of MPFR's range.
  (m, Int(clamp(ex, typemin(Int), typemax(Int))))
end

function gradient!(g::AbstractVecOrMat, f::LeastSquaresLoss, β::AbstractVecOrMat)
  _checkwrite(gradient!, f, g, β)
  mul!(f.tmp, f.X, β)
  _leastsquares_residual_gradient!(g, f.tmp, f.X, f.tmp, f.Y)
end

function _checkwrite(::Union{typeof(gradient!), typeof(value_and_gradient!)}, f::LeastSquaresLoss, g::AbstractVecOrMat, β::AbstractVecOrMat)
  _check_one_based(g, β)
  _checkreal(f, β)
  _checkreal(f, g)
  _checkgradient_output(g, β)
  _checksize(g, β)
  # Preserve the dimension error of the product, which checks before writing.
  size(β, 1) == size(f.X, 2) && size(β, 2) == size(f.tmp, 2) || mul!(f.tmp, f.X, β)
end

function _checkwrite(op::Union{typeof(prox!), typeof(value_and_prox!)}, f::LeastSquaresLoss{T}, β::AbstractVecOrMat, v::AbstractVecOrMat) where {T}
  _check_one_based(β, v)
  _checkreal(f, v)
  _checkreal(f, β)
  _checksize(β, v)
  size(β, 1) == size(f.X, 2) || throw(DimensionMismatch("β must have $(size(f.X, 2)) rows"))
  ndims(β) == ndims(f.Y) && size(β, 2) == size(f.Y, 2) ||
    throw(DimensionMismatch("β must have the same number of responses as Y"))
  true
end

function _svd_cache!(f::LeastSquaresLoss{T, Ty, Tx, N}) where {T, Ty, Tx, N}
  if !f.svd_ready
    if isempty(f.X)
      # LAPACK's empty SVD may expose a square V even for a 0×p design.
      # Hessian consumers need the thin rank-zero factorization p×0.
      f.V = zeros(T, size(f.X, 2), 0)
      f.s2 = T[]
      dims = N == 1 ? (0,) : (0, size(f.Y, 2))
      f.sUtY = Array{T, N}(undef, dims)
      f.w = Array{T, N}(undef, dims)
      f.svd_ready = true
      return f
    end
    F = svd(Matrix{T}(f.X))
    f.V = F.V
    f.s2 = F.S .^ 2
    f.sUtY = Array{T, N}(transpose(F.U) * f.Y)
    f.sUtY .*= F.S
    f.w = N == 1 ? Array{T, N}(undef, length(f.s2)) : Array{T, N}(undef, length(f.s2), size(f.Y, 2))
    f.svd_ready = true
  end
  f
end

# In the V basis the solution is (s U'Y + ρ V'v)/(s² + ρ). When V is
# rectangular, keep v's null-space component with v + V(s U'Y - s² V'v)/(s² + ρ).
# Neither form subtracts terms of size γ X'Y at a large step γ.
function prox!(β::AbstractVecOrMat, f::LeastSquaresLoss{T}, v::AbstractVecOrMat, γ::Real) where {T}
  v = _symmetric_argument(β, v)
  _checkwrite(prox!, f, β, v)
  # With no observations or coefficients the loss is constant; avoid an empty SVD.
  (isempty(f.X) || isempty(β)) && return β === v ? β : copyto!(β, v)
  β isa StridedVecOrMat || return copyto!(β, prox!(_array_buffer!(f.output, T, size(β)), f, v, γ))
  _svd_cache!(f)
  ρ = inv(convert(T, γ))
  mul!(f.w, transpose(f.V), v)
  if size(f.V, 1) == size(f.V, 2)
    if convert(T, γ) <= one(T)
      # At small steps, avoid overflow in ρ or ρ V'v by scaling both sides by γ.
      s = convert(T, γ)
      @inbounds for j in axes(f.w, 2), i in axes(f.w, 1)
        f.w[i, j] = (s * f.sUtY[i, j] + f.w[i, j]) / (s * f.s2[i] + one(T))
      end
    else
      @inbounds for j in axes(f.w, 2), i in axes(f.w, 1)
        f.w[i, j] = (f.sUtY[i, j] + ρ * f.w[i, j]) / (f.s2[i] + ρ)
      end
    end
    mul!(β, f.V, f.w)
  else
    @inbounds for j in axes(f.w, 2), i in axes(f.w, 1)
      f.w[i, j] = (f.sUtY[i, j] - f.s2[i] * f.w[i, j]) / (f.s2[i] + ρ)
    end
    β === v || copyto!(β, v)
    mul!(β, f.V, f.w, one(T), one(T))
  end
end

restrict(f::LeastSquaresLoss{T, <:AbstractVector}, S::AbstractVector{<:Integer}) where {T} =
  LeastSquaresLoss(f.Y, f.X[:, S])

lipschitz(f::LeastSquaresLoss) = _opnorm2_upper(f.X)
