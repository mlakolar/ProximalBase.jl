####################################
#  elastic net
####################################

"""
    ElasticNet(λ1, λ2)
    ElasticNet(λ1, λ2, w)

The elastic-net penalty `f(x) = λ1 Σᵢ wᵢ |xᵢ| + (λ2/2) Σᵢ xᵢ²`, with levels
`λ1, λ2 ≥ 0` and nonnegative weights `w`, all `wᵢ = 1` when `w` is omitted.
Convex, separable and zero-preserving, not smooth. Its proximal operator is
`shrink(xᵢ, γλ1wᵢ) / (1 + γλ2)` in each entry: soft thresholding by the threshold
of `NormL1(λ1, w)`, then the ridge weight of `SqrNormL2(λ2)`, each formed by
the kernel of that function from the unrounded factors. The ridge kernel
multiplies by the rounded weight `1/(1 + γλ2)`, so a result can differ in its
last bits from a division by `1 + γλ2`: `prox_block(ElasticNet(1.0, 1.0), 3.0, 1, 0.5)`
is `1.6666666666666665`, where `2.5 / 1.5` is `1.6666666666666667`.
Values are the ℓ1 part, computed as `NormL1(λ1, w)` computes its value, plus
the ridge part, computed as `SqrNormL2(λ2)` computes its value, each with its
recovery of terms that leave the range (Overflow and underflow of values). On a
dense `Float32` or `Float64` `Array` with parameters of its type, a single pass
over the entries replaces the two when a test on its sum bounds its relative
error by `(n + 2) eps` for `n` entries, `n ≤ 1/(2 eps) - 2`. With both levels positive and finite, a
sum of the two parts below `2 floatmin` is recomputed with the terms of both parts added in one
exponent-sum accumulator and rounded together: `ElasticNet(2.0^-538, 1.0)([2.0^-537])` is `2.0^-1074`,
although each part rounds to zero on its own. A zero `λ2` gives the values of
`NormL1(λ1, w)`, and at steps `γ ≥ 0` its proximal points, bit for bit.

`w` has one weight per entry of the argument, paired by linear index, and is
stored by reference. On a `SymmetricSparseIterate` the weights are a `p × p`
matrix (or anything indexed by the linear indices of one), read at both
positions of an off-diagonal coordinate, and the value counts such a coordinate
twice. On a symmetric iterate the weights must be exactly symmetric, a
precondition that the oracles assume and do not compare; validate it with
`ProximalBase.check_symmetric_parameters`.

`prox`, `prox!` and `value_and_prox!` throw an `ArgumentError` for a negative
or `NaN` scalar step before anything is written. The entries of an array of
steps are not checked: they must be nonnegative, a precondition. This is an
exception to the requirement of issue #908 that negative and `NaN` steps fail
before writes: the check would add a pass over the steps to each such call
(issue #874).
`prox_block` does not check its step.

An infinite ridge level makes `f` the indicator of `{0}` (issue #908): at
every step `γ ≥ 0`, `γ = 0` included, each entry of the proximal point is
`+0`, also for `NaN` and `±Inf` inputs; the value is `0` at the zero vector,
`NaN` if an entry is `NaN`, and `Inf` otherwise; `kkt_violation` is `-Inf`.
An infinite `λ1` and infinite weights follow the rules of `NormL1`: an infinite
`λ1` fixes every positive-weight entry at `+0` at every `γ ≥ 0` (issue #665),
and an infinite weight maps a finite input to `+0`, with no fixed-zero
guarantee for `NaN` or `±Inf` inputs at a finite `λ1` (issues #853, #653). A
threshold that overflows from finite factors keeps the limitation of issue #866
for `±Inf` inputs.

Array steps are not supported with a `SymmetricSparseIterate` destination, nor
with a dense destination and a `SymmetricSparseIterate` argument; there is no
vector form of `prox_block`, and complex entries or weights are rejected.

See Symmetric parameters for which arrays must be symmetric and who validates them.
"""
struct ElasticNet{T<:Real, W<:Union{Nothing, AbstractArray}} <: SeparableFunction
  λ1::T
  λ2::T
  w::W
  # NormL1(λ1, w), built once: its constructor scans the weights, and the value reaches its kernels through it
  l1::NormL1{T, W}

  function ElasticNet{T, W}(λ1::T, λ2::T, w::W) where {T<:Real, W<:Union{Nothing, AbstractArray}}
    w === nothing || eltype(w) <: Real || _real_error(:ElasticNet)
    _checknonneg("λ1", λ1)
    _checknonneg("λ2", λ2)
    new{T, W}(λ1, λ2, w, NormL1(λ1, w))      # NormL1 checks that the weights are nonnegative
  end
end
function ElasticNet(λ1::Real, λ2::Real, w::Union{Nothing, AbstractArray} = nothing)
  λ1, λ2 = promote(λ1, λ2)
  ElasticNet{typeof(λ1), typeof(w)}(λ1, λ2, w)
end

isconvex(::Type{<:ElasticNet}) = true
iszeropreserving(::ElasticNet) = true
_checkparams(f::ElasticNet, x) = (_checkparam(f.w, x); true)

# The soft threshold of NormL1(λ1, w) (_l1_prox_threshold), then the ridge weight of SqrNormL2(λ2) as its
# prox_block applies it. A zero λ1 gives the threshold that the NormL1 kernel forms for it, Inf at an infinite
# weight and 0 otherwise, without the kernel's exceptional product, which a zero level reaches for every
# entry with a finite weight. An infinite λ2 at γ >= 0 gives +0 (#908, D2), whatever the ridge arithmetic gives (NaN at γ = 0, a
# NaN or ±Inf entry). The tests of λ1, λ2 and a scalar γ do not depend on the entry.
Base.@propagate_inbounds function prox_block(f::ElasticNet, v::T, i::Integer, γ::Real) where {T<:Real}
  S = float(T)
  z = _elastic_threshold(f, v, i, γ, S)
  r = _ridge_needs_scaling(S, γ, f.λ2) ? _ridge_apply(_ridge_scaled(S, γ, f.λ2), z) :
                                         _ridge_apply(_ridge_ordinary(S, γ, f.λ2), z)
  ifelse(_elastic_fixed(f, γ), zero(r), r)
end
Base.@propagate_inbounds function _elastic_threshold(f::ElasticNet, v, i, γ, ::Type{S}) where {S}
  w = _entry(f.w, i)
  iszero(f.λ1) ? shrink(v, ifelse(isinf(w), S(Inf), zero(S))) : first(_l1_prox_threshold(v, γ, f.λ1, w, S))
end
@inline _elastic_fixed(f::ElasticNet, γ) = isinf(f.λ2) & (γ >= 0)

# A scalar step on an array: the ridge weights of the step, formed once per call as the array prox of SqrNormL2
# forms them (ordinary or scaled), and the D2 test, so that the loops of separable.jl apply them after the soft
# threshold of each entry. For an argument of a concrete element type, the number computed for each entry is then
# prox_block(f, xᵢ, i, γ), bit for bit, before its conversion to the element type of the destination and its
# storage (a new SparseIterate output does not store a zero result): the same kernels with the same arguments.
# With the weights formed inside prox_block instead, the code generated by Julia 1.13 formed them again for each
# entry in the loops over the stored entries of the iterates, and in the dense loop for weights outside the
# ordinary band (a zero λ2 included), which did not vectorize (benchmark/elasticnet.jl). An argument whose element
# type is not concrete keeps prox_block.
struct _ElasticStep{F<:ElasticNet, R} <: SeparableFunction
  f::F
  ridge::R          # a _RidgeOrdinary or a _RidgeScaled of the step
  fixed::Bool       # _elastic_fixed(f, γ)
end
Base.@propagate_inbounds function prox_block(g::_ElasticStep, v::T, i::Integer, γ::Real) where {T<:Real}
  r = _ridge_apply(g.ridge, _elastic_threshold(g.f, v, i, γ, float(T)))
  ifelse(g.fixed, zero(r), r)
end
iszeropreserving(::_ElasticStep) = true
_checkparams(g::_ElasticStep, x) = _checkparams(g.f, x)

# The loops of separable.jl with the prepared step: the dense one, the stored entries of a sparse iterate, and
# a symmetric iterate output (a dense output from a symmetric argument reaches the first through matrixview).
# The checks of the function run before the step is formed, which needs real entries.
@inline function _elastic_prepared!(loop!::L, y, f::ElasticNet, x, γ::Real) where {L}
  _checkwrite(prox!, f, y, x)
  isconcretetype(eltype(x)) || return loop!(y, f, x, γ)
  S = float(eltype(x))
  fixed = _elastic_fixed(f, γ)
  _ridge_needs_scaling(S, γ, f.λ2) ? loop!(y, _ElasticStep(f, _ridge_scaled(S, γ, f.λ2), fixed), x, γ) :
                                     loop!(y, _ElasticStep(f, _ridge_ordinary(S, γ, f.λ2), fixed), x, γ)
end
_prox_sep!(y::AbstractArray, f::ElasticNet, x::AbstractArray, γ::Real) = _elastic_prepared!(_prox_sep_dense!, y, f, x, γ)
_prox_sep!(y::SparseIterate, f::ElasticNet, x::SparseIterate, γ::Real) = _elastic_prepared!(_prox_stored!, y, f, x, γ)
_prox_sep!(y::SymmetricSparseIterate, f::ElasticNet, x::Union{AbstractMatrix, SymmetricSparseIterate}, γ::Real) =
  _elastic_prepared!(_prox_coordinates!, y, f, x, γ)
# the loops of separable.jl for these arguments, also for an ElasticNet whose step is not prepared
_prox_stored!(y, f, x, γ) = invoke(_prox_sep!, Tuple{SparseIterate, Any, SparseIterate, Union{Real, AbstractArray}}, y, f, x, γ)
_prox_coordinates!(y, f, x, γ) =
  invoke(_prox_sep!, Tuple{SymmetricSparseIterate, Any, Union{AbstractMatrix, SymmetricSparseIterate}, Real}, y, f, x, γ)

# The term of entry i: the expression of value_entry(::NormL1) with λ1, plus λ2 v²/2. A nonzero term below 2 floatmin
# with both levels positive and finite is recomputed with its two parts rounded together (_elastic_entry_recomputed).
Base.@propagate_inbounds function value_entry(f::ElasticNet, v::Real, i::Integer)
  a = abs(float(v))
  # an absent weight is one in the entry's type, as for NormL1 (#684)
  w = f.w === nothing ? one(a) : _entry(f.w, i)
  e = _elastic_entry(f.λ1, f.λ2, w, a)
  _elastic_recompute(e, f.λ1, f.λ2) && !iszero(a) ? _elastic_entry_recomputed(f.λ1, f.λ2, w, a, e) : e
end
@inline function _elastic_entry(λ1, λ2, w, a)
  p = _positive_product(λ1, w, a)
  ifelse(isinf(w) & !iszero(a), oftype(p, Inf) * a, p) + _elastic_square(λ2, a)
end

# Whether a value or a term c, the sum of the two parts each rounded on its own, is recomputed: a Float32 or Float64
# c below 2 floatmin, with both levels positive and finite (one zero level leaves one part, rounded once).
@inline _elastic_recompute(c, λ1, λ2) =
  _underflow_checked(typeof(c)) && (c < 2 * floatmin(typeof(c))) & (λ1 > 0) & (λ2 > 0) & isfinite(λ1) & isfinite(λ2)

# The recomputation of a value or a term c of type T that _elastic_recompute selects. Each part of c rounds to T on its
# own, so c can lose half a spacing δ/2 (δ = eps(T) floatmin(T)) to each part of a total of that size:
# ElasticNet(2^-538, 1) at [2^-537] has two parts 2^-1075, which each round to 0, and the value 2^-1074. The
# recomputation adds the N terms λ1 wᵢ|xᵢ| and λ2 xᵢ²/2 of the entries that the separable loops visit (every entry of
# an array, the stored entries of a sparse iterate, the active groups of a block iterate, each stored coordinate of a
# symmetric iterate at both of its positions) to one exponent-sum accumulator (_ExpSum) in R, the promotion of T,
# Float64 and the factors, and rounds the sum to T once (_expsum_value), as the NormL1 recomputation does for its own
# terms (#373). A value whose stored entries are all zero is kept.
#
# Claim: with u = eps(R)/2 and γₘ = m u/(1 - m u), the result r and the exact value V satisfy
#   |r - V| ≤ (γ₃ + (γ²_{N-1} + N 2^-1070)(1 + γ₂)) V + h,
# h being half the spacing of T at r (δ/2 for r below 2 floatmin(T)), or the whole spacing where ldexp truncates below
# floatmin (Julia 1.10, T = Float64, #896).
# Proof. A term is formed from the significands of its two factors and of its level, with the exponents summed exactly,
# as (mₐ mᵦ) m_λ 2^e: two roundings in R, so it is within γ₂ of its exact value, and it is nonnegative; their sum S is
# within γ₂ V of V. The accumulator adds each term with an error-free transformation (Fast2Sum on ordered magnitudes)
# and sums the errors in plain floating point, then forms s + c once: the summation Sum2 of Ogita, Rump and Oishi
# (2005), within u S + γ²_{N-1} S of S for nonnegative terms. Moving the sum into the frame of a larger term, or a term
# into the frame of the sum, multiplies by a power of two, exactly except below the smallest subnormal of R: at most
# 2^-1073 frame units per term, while the sum is at least 2^-3 frame units (a term is a product of three significands
# in [1/2, 1)), so at most N 2^-1070 S in all. The exponent is applied exactly (_scaled_product with the factor 1
# multiplies by powers of two), and the result is rounded once to T, within h. The relative factors combine to at most
# γ₃ + (γ²_{N-1} + N 2^-1070)(1 + γ₂).  □
# The bound depends on N only through γ²_{N-1}, a second-order term, and N 2^-1070, below 2^-1000 for any N that an
# array can hold. It is below the worst case of c: a part of c that is recomputed (NormL1 from the same kind of
# accumulator, the ridge from scaled squares summed the same compensated way, in the type of the data) carries a
# relative term at least as large plus its own rounding to T, and c adds the two parts with one more rounding, so the
# bound of c is (γ₄ + γ²_{N-1}(1 + γ₄)) V, with a u at least that of R, plus three halves of a spacing; a part that
# keeps the sum of its ordinary loop has an error that grows with its number of terms. For Float32 data R is Float64,
# and for N ≤ 2^26 the relative term is below 2^-50 V: r is the rounding of V except within that distance of a tie.
@noinline function _elastic_entry_recomputed(λ1, λ2, w, a, e::T) where {T}
  R = promote_type(T, Float64, _factor_type(λ1), _factor_type(λ2), _factor_type(w), _factor_type(a))
  m1, e1 = _frexp(convert(R, λ1))
  m2, e2 = _frexp(convert(R, λ2))
  acc = _expsum_term(_expsum_start(R), w, a, m1, e1, 0)
  _expsum_value(T, _expsum_term(acc, a, a, m2, e2, -1))
end

# λ a²/2 for a = |v|. While a² is normal and λ a² is in [2 floatmin, floatmax], the two products are kept and
# the halving is exact. Otherwise: a zero λ or a zero a gives 0 (a zero λ also at a NaN or ±Inf entry), an
# infinite λ gives λ a, Inf or NaN (#908, D2), and other factors the bounded-product kernel, rounded once
# from scaled significands, so that a square that underflows or overflows is not lost before λ restores it.
@inline _elastic_square(λ::Real, a::Real) = _elastic_square(promote(float(λ), a)...)
@inline function _elastic_square(λ::T, a::T) where {T<:AbstractFloat}
  q = a * a
  p = λ * q
  ((floatmin(T) <= q) & (q <= floatmax(T)) & (2 * floatmin(T) <= p) & (p <= floatmax(T))) && return p / 2
  _elastic_square_exceptional(λ, a)
end
@noinline function _elastic_square_exceptional(λ::T, a::T) where {T}
  (iszero(λ) || iszero(a)) && return zero(T)
  isinf(λ) && return λ * a
  _scaled_product(λ, a, a, -1)
end

# The value on each representation: the ℓ1 part from the stored NormL1 (its dense fast paths and its recovery
# of terms below floatmin, #373) plus the ridge part from the squared-residual kernel (_square_value, which
# recomputes squares that underflow or overflow), then one addition. Both read the entries that the
# representation stores: every entry of a dense array, the stored entries of a sparse iterate, the storage of
# a block iterate, each stored coordinate of a symmetric iterate at both of its positions.
function _elastic_value(f::ElasticNet, x)
  l = _value_unchecked(f.l1, x)
  c = _value_at(l + _elastic_ridge_value(f.λ2, x), x)
  _elastic_recompute(c, f.λ1, f.λ2) ? _elastic_value_recomputed(f, x, c) : c
end

# The recomputation of _elastic_entry_recomputed for a value, over the entries of each representation.
@noinline function _elastic_value_recomputed(f::ElasticNet, x, c::T) where {T}
  iszero(c) && _all_zero(_stored_values(x)) && return c
  R = promote_type(T, Float64, _factor_type(f.λ1), _factor_type(f.λ2), _l1_factor_type(f.w), _l1_factor_type(_stored_values(x)))
  m1, e1 = _frexp(convert(R, f.λ1))
  m2, e2 = _frexp(convert(R, f.λ2))
  _expsum_value(T, _elastic_expsum(_expsum_start(R), f.w, m1, e1, m2, e2, x))
end
# Σ (λ1 wᵢ|xᵢ| + λ2 xᵢ²/2) added to the accumulator, the levels given by their significands and exponents
function _elastic_expsum(acc, w, m1, e1, m2, e2, x::AbstractArray)
  @inbounds for i in eachindex(IndexLinear(), x)
    a = abs(float(x[i]))
    acc = _expsum_term(_expsum_term(acc, _entry(w, i), a, m1, e1, 0), a, a, m2, e2, -1)
  end
  acc
end
function _elastic_expsum(acc, w, m1, e1, m2, e2, x::SparseIterate)
  @inbounds for k in 1:x.nnz
    a = abs(float(x.nzval[k]))
    acc = _expsum_term(_expsum_term(acc, _entry(w, x.nzval2ind[k]), a, m1, e1, 0), a, a, m2, e2, -1)
  end
  acc
end
function _elastic_expsum(acc, w, m1, e1, m2, e2, x::BlockIterate)
  @inbounds for g in 1:x.nactive, i in x.groups[x.active[g]]
    a = abs(float(x.storage[i]))
    acc = _expsum_term(_expsum_term(acc, _entry(w, i), a, m1, e1, 0), a, a, m2, e2, -1)
  end
  acc
end
# an off-diagonal coordinate: its ℓ1 terms with the weights of both positions, its square counted twice
function _elastic_expsum(acc, w, m1, e1, m2, e2, x::SymmetricSparseIterate)
  p = x.p
  @inbounds for k in 1:x.nnz
    r, c = tril_cartesian(p, x.nzval2ind[k])
    a = abs(float(x.nzval[k]))
    acc = _expsum_term(acc, _entry(w, (c - 1) * p + r), a, m1, e1, 0)
    r == c || (acc = _expsum_term(acc, _entry(w, (r - 1) * p + c), a, m1, e1, 0))
    acc = _expsum_term(acc, a, a, m2, e2, r == c ? -1 : 0)
  end
  acc
end

# λ/2 Σ xᵢ²: exactly 0 for a zero λ, also at NaN and ±Inf entries, as for SqrNormL2, and for an infinite λ the
# indicator of {0} (#908, D2), from the largest stored magnitude, never from Inf⋅0: 0 when every stored entry
# is a zero, NaN when one is NaN, Inf otherwise.
@inline function _elastic_ridge_value(λ, x)
  T = _valtype(x)
  iszero(λ) && return zero(T)
  isinf(λ) || return _square_value(λ, x, nothing)
  m = _stored_maxabs(x)
  iszero(m) ? zero(T) : isnan(m) ? T(NaN) : T(Inf)
end

_value_dense(f::ElasticNet, x::AbstractArray) = _elastic_value(f, x)

# A dense Float32 or Float64 Array, with levels and weights of its type: one fused pass, kept when the bound
# below holds, and otherwise the composed value. Each part of the composed value checks its own sum, so a
# part that the other dominates is recomputed (terms λ1 wᵢ|xᵢ| below floatmin beside a normal ridge part,
# squares below floatmin beside a normal ℓ1 part) where the sum of both is accurate already. A zero λ2 gives
# the value of the ℓ1 part, the value of NormL1(λ1, w).
function _value_dense(f::ElasticNet{T, <:Union{Nothing, Array{T}}}, x::Array{T}) where {T<:Union{Float32, Float64}}
  iszero(f.λ2) && return _value_unchecked(f.l1, x)
  s, kept = _elastic_fused_value(f, x)
  kept ? s : _elastic_value(f, x)
end

# The fused value s = fl(fl(λ1 s1) + fl(fl(λ2 s2)/2)) of the sums s1 = Σᵢ wᵢ|xᵢ| (wᵢ = 1 without weights) and
# s2 = Σᵢ xᵢ², both accumulated in one vectorized loop, and whether the n entries number at most 1/(2 eps) - 2
# (an integer comparison: 1/(2 eps) = 2^(precision - 2)) and s is finite with s ≥ 4 floatmin (n (λ1 + λ2) + 1).
# Then the relative error of s is at most (n + 2) eps. For more entries the loop is not run and the flag is false.
#
# Proof. Let u = eps/2, δ = eps floatmin (the spacing of the subnormal numbers, δ/2 = u floatmin), γₖ = k u/(1 - k u),
# W = Σ wᵢ aᵢ, Q = Σ aᵢ² with aᵢ = |xᵢ|, S = λ1 W + λ2 Q/2 the exact value; the length test gives (n + 2) u ≤ 1/4. Every
# intermediate is nonnegative: an overflow makes s Inf, and a zero times Inf makes it NaN, both excluded; so no
# operation overflows. A product rounds to p(1 + ε) + η with |ε| ≤ u and |η| ≤ δ/2 (η ≠ 0 only where the result is
# below floatmin), a sum to (a + b)(1 + ε), and a sum of n nonnegative terms, in any order, within γ_{n-1} of their sum.
# @simd may also contract a product into its addition (Julia 1.13 emits fma for s2, and for s1 with weights); an
# fma rounds acc + p once, to (acc + p)(1 + ε) + η, so each term still meets at most n roundings and each η at most
# n - 1 later ones, and the bounds below hold unchanged.
# The terms wᵢ aᵢ and aᵢ² give
#   |ŝ1 - W| ≤ γₙ W + (1 + γ_{n-1}) n δ/2,    |ŝ2 - Q| ≤ γₙ Q + (1 + γ_{n-1}) n δ/2.
# The products with λ1 and λ2, and the halving (exact unless subnormal, then within δ/2), give
#   |fl(λ1 ŝ1) - λ1 W| ≤ γ_{n+1} λ1 W + (1 + u)(1 + γ_{n-1}) λ1 n δ/2 + δ/2,
#   |fl(fl(λ2 ŝ2)/2) - λ2 Q/2| ≤ γ_{n+1} λ2 Q/2 + (1 + u)(1 + γ_{n-1}) (λ2/2) n δ/2 + 3δ/4,
# and the final sum |s - S| ≤ γ_{n+2} S + E with E = (1 + u)² (1 + γ_{n-1}) (λ1 + λ2/2) n δ/2 + (5/2)(1 + u) δ/2.
# The test gives S ≥ (s - E)/(1 + γ_{n+2}) ≥ (B - E)/(1 + γ_{n+2}), B = 4 floatmin (n (λ1 + λ2) + 1), so E ≤ u S when
# B ≥ E (1 + γ_{n+2} + u)/u. With δ/2 = u floatmin and γ_{n-1}, γ_{n+2} ≤ 1/3,
#   E (1 + γ_{n+2} + u)/u ≤ (4/3 + u) ((4/3)(1 + u)² (λ1 + λ2/2) n + (5/2)(1 + u)) floatmin
#                       < (1.78 λ1 n + 0.89 λ2 n + 3.34) floatmin,
# which B exceeds by a factor 1.19 or more, also as computed (four roundings). Then
#   |s - S| ≤ (γ_{n+2} + u) S ≤ ((4/3)(n + 2) + 1) u S ≤ 2 (n + 2) u S = (n + 2) eps S.  □
@inline function _elastic_fused_value(f::ElasticNet{T}, x::Array{T}) where {T}
  length(x) <= _elastic_fused_length(T) || return (zero(T), false)
  s1 = s2 = zero(T)
  @inbounds @simd for i in eachindex(IndexLinear(), x)
    a = abs(x[i])
    s1 += _elastic_weighted(f.w, i, a)
    s2 += a * a
  end
  λ1, λ2 = f.λ1, f.λ2
  s = λ1 * s1 + λ2 * s2 / 2
  s, (s <= floatmax(T)) & (s >= 4 * floatmin(T) * (T(length(x)) * (λ1 + λ2) + 1))
end
# 1/(2 eps) - 2 entries, the most for which the bound above is proved
@inline _elastic_fused_length(::Type{T}) where {T} = (1 << (precision(T) - 2)) - 2
@inline _elastic_weighted(::Nothing, i, a) = a
Base.@propagate_inbounds _elastic_weighted(w::AbstractArray, i, a) = w[i] * a

_value_sep(f::ElasticNet, x::SparseIterate) = (_checkparams(f, x); _elastic_value(f, x))
_value_sep(f::ElasticNet, x::BlockIterate) = (_checkparams(f, x); _elastic_value(f, x))
_value_sep(f::ElasticNet, x::SymmetricSparseIterate) = (_checkparams(f, x); _elastic_value(f, x))

# D1 (#908): the public array oracles refuse a negative or NaN scalar step before anything is written, and
# then take the checked separable route. prox_block and the routes inside ProximalBase do not test the step:
# Scaled hands its own step on (α γ, NaN for α = Inf at γ = 0, #311). Array steps are not read here.
@noinline _elastic_step_error(γ) = throw(ArgumentError("ElasticNet requires a nonnegative step γ, got $γ"))
@inline _check_elastic_step(γ::Real) = (γ >= 0 || _elastic_step_error(γ); nothing)
const _SeparableScalarStep = Tuple{_ArrayOrSymmetricIterate, SeparableFunction, _ArrayOrSymmetricIterate, Real}
prox!(y::_ArrayOrSymmetricIterate, f::ElasticNet, x::_ArrayOrSymmetricIterate, γ::Real) =
  (_check_elastic_step(γ); invoke(prox!, _SeparableScalarStep, y, f, x, γ))
value_and_prox!(y::_ArrayOrSymmetricIterate, f::ElasticNet, x::_ArrayOrSymmetricIterate, γ::Real) =
  (_check_elastic_step(γ); invoke(value_and_prox!, _SeparableScalarStep, y, f, x, γ))

# the ℓ1 screen of NormL1; an infinite ridge level fixes the coordinate at zero, which then never violates (D2)
function kkt_violation(f::ElasticNet, g::AbstractArray, i::Integer)
  eltype(g) <: Real || _real_error(:kkt_violation)
  v = abs(g[i]) - _kkt_coefficient(eltype(g), f.λ1, _entry(f.w, i))
  isinf(f.λ2) ? oftype(v, -Inf) : v
end
restrict(f::ElasticNet, S::AbstractVector{<:Integer}) = ElasticNet(f.λ1, f.λ2, _restrict_entries(f.w, S))
