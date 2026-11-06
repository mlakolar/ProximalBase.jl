module ZeroWeightNaNTest

# A zero coefficient times NaN in the values of weighted separable families (E4 before E2): a zero
# coordinate weight, or a zero level at a finite weight, leaves the coordinate free, so a NaN there
# does not reach the value of EpsInsensitiveLoss, Berhu, CappedL1 or NormLq. A NaN in a penalized
# coordinate stays NaN, an infinite weight keeps its constraint, and the raw parameters decide: a
# positive parameter whose conversion or product rounds to zero is no zero mask (R3).

using Test
using ProximalBase
using SparseArrays
using JLArrays: JLArray
using GPUArraysCore: allowscalar
import Adapt

const PB = ProximalBase

####################################
#  Reference values, from the documented formulas and independent of the package kernels
####################################

weight(f, i) = f.w === nothing ? 1 : f.w[i]
level(f::EpsInsensitiveLoss) = 1
level(f) = f.λ
# the unweighted term at t = |v| ≥ 0
shape(f::EpsInsensitiveLoss, t::T) where {T} = max(t - T(f.ε), zero(T))
shape(f::Berhu, t::T) where {T} = t <= T(f.M) ? t : (t^2 + T(f.M)^2) / (2 * T(f.M))
shape(f::CappedL1, t::T) where {T} = min(t, T(f.a))
shape(f::NormLq, t::T) where {T} = t^T(f.q)

free(λ, w) = iszero(w) || (iszero(λ) && !isinf(w))
function ref_term(f, v, i)
  T = float(typeof(v))
  λ, w = level(f), weight(f, i)
  free(λ, w) && return zero(T)
  φ = shape(f, abs(T(v)))
  isinf(w) && return isnan(φ) ? φ : iszero(φ) ? zero(T) : T(Inf)
  T(λ) * T(w) * φ
end
function ref_value(f, x)
  s = zero(float(eltype(x)))
  for i in eachindex(x)
    s += ref_term(f, x[i], i)
  end
  s
end

families(T, w) = (EpsInsensitiveLoss(T(0.25), w), Berhu(one(T), one(T), w), CappedL1(one(T), one(T), w),
                  NormLq(one(T), T(0.5), w))
leveled(T, λ, w) = (Berhu(T(λ), one(T), w), CappedL1(T(λ), one(T), w), NormLq(T(λ), T(0.5), w))

# the fused value with a separate output and aliased, and the stored output
function fused(f, x, γ)
  y = similar(x)
  v = value_and_prox!(y, f, x, γ)
  z = copy(x)
  va = value_and_prox!(z, f, z, γ)
  v, va, y, z
end

####################################
#  The issue's cases
####################################

@testset "[NaN, 1] with weights [0, 1], $T" for T in (Float32, Float64)
  w = T[0, 1]
  x = T[NaN, 1]
  # standalone value, fused value, stored output (unit step)
  for (F, value, stored) in ((w -> EpsInsensitiveLoss(T(0.2), w), one(T) - T(0.2), T[NaN, 0.2]),
                             (w -> Berhu(one(T), one(T), w), one(T), T[NaN, 0]),
                             (w -> CappedL1(one(T), one(T), w), one(T), T[NaN, 0]),
                             (w -> NormLq(one(T), T(0.5), w), one(T), T[NaN, 0]))
    f = F(w)
    @test f(x) === value
    @test PB.value_entry(f, T(NaN), 1) === zero(T)
    v, va, y, z = fused(f, x, one(T))
    @test v === zero(T) && va === zero(T)
    @test isequal(y, stored) && isequal(z, stored)
    # a positive weight at the NaN entry keeps NaN
    g = F(T[1, 0])
    @test isnan(g(x))
    @test isnan(fused(g, x, one(T))[1])
  end
end

@testset "a zero-weight NaN is free, a positive-weight NaN is NaN, $T" for T in (Float32, Float64)
  w = T[0, 1, 2, 0, 1]
  x0 = T[1, 0.25, 4, 1, 0.25]
  for f in families(T, w), nans in ([1], [4], [1, 4], [3], [5], [1, 2], [1, 4, 5], Int[])
    x = copy(x0); x[nans] .= NaN
    isfree = all(i -> iszero(w[i]), nans)
    expected = ref_value(f, x)
    @test isnan(expected) == !isfree
    @test isequal(f(x), expected)
    for γ in (T(0.5), one(T), T(3))
      v, va, y, z = fused(f, x, γ)
      @test isequal(y, prox(f, x, γ)) && isequal(z, y)
      r = ref_value(f, y)
      if isfree
        @test isapprox(v, r; rtol = 8eps(T)) && isapprox(va, r; rtol = 8eps(T))
      else
        @test isnan(v) && isnan(va)
      end
    end
  end
  # matrices with matrix weights
  W = T[0 1 2; 1 0 1]
  for f in families(T, W)
    X = T[NaN 1 4; 0.25 NaN 1]
    @test f(X) === ref_value(f, X)
    @test isfinite(f(X))
    @test isfinite(fused(f, X, one(T))[1])
    X[1, 2] = NaN
    @test isnan(f(X)) && isnan(fused(f, X, one(T))[1])
  end
end

####################################
#  Levels
####################################

@testset "zero level with finite weights, $T" for T in (Float32, Float64)
  for w in (nothing, T[2, 1]), f in leveled(T, 0, w),
      x in (T[NaN, 1], T[1, NaN], T[NaN, NaN], T[NaN, Inf], T[0, NaN])
    @test f(x) === zero(T)
    v, va, y, z = fused(f, x, one(T))
    @test v === zero(T) && va === zero(T)
    @test isequal(y, x) && isequal(z, x)
    @test PB.value_entry(f, T(NaN), 2) === zero(T)
  end
  for x in (T[NaN, 1], T[1, NaN], T[NaN, NaN])
    f = EpsInsensitiveLoss(T(0.2), T[0, 0])
    @test f(x) === zero(T)
    @test fused(f, x, one(T))[1] === zero(T)
  end
end

@testset "zero level with an infinite weight keeps its constraint, $T" for T in (Float32, Float64)
  for f in leveled(T, 0, T[Inf, 1])
    @test isnan(f(T[NaN, 1])) && isnan(f(T[NaN, 0]))
    @test f(T[0, NaN]) === zero(T)
    @test f(T[1, NaN]) === T(Inf)             # the free coordinate does not hide the constraint
    @test f(T[1, 1]) === T(Inf) && f(T[0, 1]) === zero(T)
    @test isnan(PB.value_entry(f, T(NaN), 1))
    v, va, y, z = fused(f, T[1, NaN], one(T))
    @test isequal(y, T[0, NaN]) && isequal(z, y)
    @test v === zero(T) && va === zero(T)
  end
  # zero scaling folds into a zero level and keeps the infinite weight
  for f in leveled(T, 1, T[Inf, 1])
    z = Scaled(f, zero(T))
    @test z isa typeof(f) && iszero(z.λ)
    @test z(T[0, NaN]) === zero(T)
    @test isnan(z(T[NaN, 0]))
  end
  # EpsInsensitiveLoss: an infinite weight beside a zero one
  f = EpsInsensitiveLoss(T(0.2), T[Inf, 0])
  @test f(T[0.1, NaN]) === zero(T)
  @test f(T[1, NaN]) === T(Inf)
  @test isnan(f(T[NaN, 0]))
end

@testset "infinite weights at a positive level, $T" for T in (Float32, Float64)
  for f in families(T, T[Inf, 1])
    @test isnan(f(T[NaN, 1]))
    @test isnan(f(T[1, NaN]))                  # NaN takes precedence over Inf
    @test isnan(f(T[0, NaN]))
    @test f(T[1, 1]) === T(Inf)
    @test f(T[0, 1]) === ref_value(f, T[0, 1])
  end
  for f in families(T, T[Inf, 0])
    @test f(T[0, NaN]) === zero(T)
    @test f(T[1, NaN]) === T(Inf)
    @test isnan(f(T[NaN, 1]))
  end
end

####################################
#  Conversions and parameter types
####################################

@testset "a positive parameter that rounds to zero is no zero mask" begin
  @test isnan(Berhu(1e-200, 1, [1e-200, 1])([NaN, 1]))
  @test isnan(CappedL1(1e-200, 1, [1e-200, 1])([NaN, 1]))
  @test isnan(NormLq(1e-200, 0.5, [1e-200, 1])([NaN, 1]))
  @test isnan(CappedL1(1e-46, 1)(Float32[NaN]))
  @test isnan(NormLq(1e-46, 0.5)(Float32[NaN]))
  @test isnan(Berhu(1e-46, 1)(Float32[NaN]))
  for f in families(Float64, [1e-50, 1.0])
    @test isnan(f(Float32[NaN, 1]))
    @test isnan(PB.value_entry(f, NaN32, 1))
  end
end

@testset "parameter types" begin
  # Float64 parameters with Float32 data: the value is Float32, the entry keeps its type
  for (f, value) in ((EpsInsensitiveLoss(0.2, [0.0, 1.0]), 1f0 - 0.2f0), (Berhu(1.0, 1.0, [0.0, 1.0]), 1f0),
                     (CappedL1(1.0, 1.0, [0.0, 1.0]), 1f0), (NormLq(1.0, 0.5, [0.0, 1.0]), 1f0))
    @test f(Float32[NaN, 1]) === value
    @test (@inferred PB.value_entry(f, NaN32, 1)) === zero(typeof(PB.value_entry(f, 1f0, 2)))
  end
  # integer and Bool weights, integer levels
  @test EpsInsensitiveLoss(0.2, [0, 1])([NaN, 1.0]) === 1.0 - 0.2
  @test EpsInsensitiveLoss(0.2, Bool[0, 1])([NaN, 1.0]) === 1.0 - 0.2
  @test Berhu(1, 1, [0, 1])([NaN, 1.0]) === 1.0
  @test CappedL1(1, 1, [0, 1])([NaN, 1.0]) === 1.0
  @test Berhu(0, 1)(Float32[NaN]) === 0f0
  @test CappedL1(0, 1, [0, 2])(Float32[NaN, NaN]) === 0f0
  @test isnan(Berhu(1, 1, [0, 1])([1.0, NaN]))
end

####################################
#  Finite inputs: the base formulas, frozen
####################################

frozen_coefficient(s, w) = (p = s * w; (s > zero(s)) & isfinite(s) ? p : ifelse(isnan(p) & !isnan(s), oftype(p, w), p))
frozen_weighted(p, φ) = ifelse(isnan(p) & !isnan(φ), zero(p), p)
entry(w, i) = w === nothing ? true : w[i]
function frozen_entry(f::EpsInsensitiveLoss, v::T, i) where {T}
  w = entry(f.w, i)
  φ = max(abs(v) - convert(float(T), f.ε), zero(float(T)))
  frozen_weighted(w * φ, φ)
end
function frozen_entry(f::Berhu, v::T, i) where {T}
  S = float(T)
  λ, w = S(f.λ), S(entry(f.w, i))
  c, M, t = frozen_coefficient(λ, w), S(f.M), abs(S(v))
  frozen_weighted(t <= M ? c * t : c * (t^2 + M^2) / (2M), t)
end
function frozen_entry(f::CappedL1, v::T, i) where {T}
  S = float(T)
  λ, w, m = S(f.λ), S(entry(f.w, i)), min(abs(S(v)), S(f.a))
  frozen_weighted(frozen_coefficient(λ, w) * m, m)
end
function frozen_entry(f::NormLq, v::T, i) where {T}
  S = float(T)
  λ, w, φ = S(f.λ), S(entry(f.w, i)), abs(S(v))^S(f.q)
  frozen_weighted(frozen_coefficient(λ, w) * φ, φ)
end

@testset "finite inputs are unchanged, $T" for T in (Float32, Float64)
  n = 64
  x = T[(-1)^k * T(k) / 7 for k in 1:n]
  x[1:8] .= T[0, Inf, -Inf, nextfloat(zero(T)), -floatmin(T), floatmax(T), -floatmax(T), T(0.25)]
  base_w = T[0.5 + mod(k, 5) / 4 for k in 1:n]
  for P in (T, Float64)
    ws = (nothing, P.(base_w), [k % 3 == 0 ? zero(P) : k % 7 == 0 ? P(Inf) : P(base_w[k]) for k in 1:n])
    for w in ws, λ in (0, 0.3, 1, Inf)
      fs = (EpsInsensitiveLoss(P(0.25), w), Berhu(P(λ), one(P), w), CappedL1(P(λ), one(P), w), NormLq(P(λ), P(0.5), w))
      for f in fs
        @test all(i -> PB.value_entry(f, x[i], i) === frozen_entry(f, x[i], i), 1:n)
        # the sparse iterate sums value_entry in storage order
        si = SparseIterate(x)
        s = zero(T)
        for k in 1:si.nnz
          s += convert(T, frozen_entry(f, si.nzval[k], si.nzval2ind[k]))
        end
        @test isequal(f(si), s)
      end
    end
  end
end

####################################
#  Iterates and device arrays
####################################

@testset "iterates, $T" for T in (Float32, Float64)
  for f in families(T, T[0, 1])
    expected = ref_value(f, T[0, 1])
    si = SparseIterate(T, 2); si[1] = NaN; si[2] = 1
    @test f(si) === expected
    y = SparseIterate(T, 2)
    v = value_and_prox!(y, f, si, one(T))
    @test isequal(Vector(y), prox(f, T[NaN, 1], one(T)))
    @test v === ref_value(f, Vector(y))
    z = copy(si)
    @test value_and_prox!(z, f, z, one(T)) === v
    a = AtomIterate(T, 2, [1:1, 2:2]); a[1] = NaN; a[2] = 1
    @test f(a) === expected
    # a positive weight at the NaN entry keeps NaN
    si[1], si[2] = 1, NaN
    @test isnan(f(si))
  end
  # SymmetricSparseIterate: a NaN on the diagonal with a zero weight, an off-diagonal coordinate
  # counted twice
  for f in families(T, T[0 1; 1 1])
    S = SymmetricSparseIterate(T, 2); S[1, 1] = NaN; S[2, 1] = 1
    expected = 2 * ref_term(f, one(T), 2)
    @test f(S) === expected
    y = SymmetricSparseIterate(T, 2)
    v = value_and_prox!(y, f, S, one(T))
    @test isfinite(v)
    @test v === ref_value(f, vec(Matrix(y)))
    S[1, 1], S[2, 2] = 1, NaN
    @test isnan(f(S))
  end
end

# Scaled recovers α f(x) from the terms in parts when f(x) overflows or underflows (R1); a free
# NaN entry (NormLq: also a free infinite one) contributes nothing there either.
@testset "Scaled recovery past a free non-finite entry, $T" for T in (Float32, Float64)
  big_ = T === Float64 ? (λ = 1e300, w = 1e10, α = 1e-20) : (λ = 1f30, w = 1f10, α = 1f-20)
  small = T === Float64 ? (λ = 1e-300, w = 1e-30, α = 1e30) : (λ = 1f-30, w = 1f-30, α = 1f30)
  for (λ, w2, α) in (big_, small), mk in ((λ, w) -> CappedL1(λ, one(T), w), (λ, w) -> NormLq(λ, T(0.5), w))
    g = Scaled(mk(λ, T[0, w2]), α)
    expected = g(T[0, 1])
    @test isfinite(expected) && !iszero(expected)
    @test expected ≈ T(big(λ) * big(w2) * big(α))
    sexpected = g(SparseIterate(T[0, 1]))
    @test sexpected ≈ expected
    bad = g.f isa NormLq ? (T(NaN), T(Inf), T(-Inf)) : (T(NaN),)
    for b in bad
      @test g(T[b, 1]) === expected
      @test g(SparseIterate(T[b, 1])) === sexpected
    end
    # a penalized NaN or infinite entry keeps its result
    h = Scaled(mk(λ, T[1, w2]), α)
    @test isnan(h(T[NaN, 1])) && isnan(h(SparseIterate(T[NaN, 1])))
    if g.f isa NormLq
      @test h(T[Inf, 1]) === T(Inf) && h(SparseIterate(T[Inf, 1])) === T(Inf)
    else
      @test h(T[Inf, 1]) === h(T[1, 1])
    end
  end
end

# At a penalized non-finite entry the recovery is declined, and the scaled value is the ordinary
# α f(x). The entry's own value cannot stand in for its term: with q = 2^-160 or 1e-50, which
# convert to zero in Float32, value_entry gives λ w NaN^0 = λ w, already including the level.
@testset "Scaled recovery declines at a penalized non-finite entry, $T" for T in (Float32, Float64)
  declined(g, x) = T(g.α * g.f(x))
  for xs in (identity, SparseIterate)
    # the q of the coefficient's overflow converts to zero in Float32
    g = Scaled(NormLq(1e30, 1e-50, T[1, 1e10]), 1e-30)
    for b in (T(NaN), T(Inf), T(-Inf))
      x = xs(T[b, 1])
      @test isequal(g(x), declined(g, x))
      T === Float32 && !isnan(b) && @test g(x) === Inf32
    end
    # a subnormal level, a q that converts to zero, Bool, Int and absent weights
    for w in (Bool[true], [1], nothing), b in (T(NaN), T(Inf), T(-Inf))
      g = Scaled(NormLq(2.0^-130, 2.0^-160, w), 2.0^100)
      x = xs(T[b])
      @test isequal(g(x), declined(g, x))
      T === Float32 && @test g(x) === 2.0f0^-30
      h = Scaled(CappedL1(2.0^-130, 1, w), 2.0^100)
      isnan(b) && @test isnan(h(xs(T[b])))
    end
    # the same parameters at a free entry: the term is skipped, the other entries are recovered
    for w in (Bool[false, true], [0, 1]), b in (T(NaN), T(Inf), T(-Inf))
      g = Scaled(NormLq(2.0^-130, 2.0^-160, w), 2.0^100)
      @test g(xs(T[b, 1])) === g(xs(T[0, 1]))
      h = Scaled(CappedL1(2.0^-130, 1, w), 2.0^100)
      isnan(b) && @test h(xs(T[b, 1])) === h(xs(T[0, 1]))
    end
  end
end

# An infinite ε makes EpsInsensitiveLoss zero on every finite argument, so its value does not
# depend on the entry: it is 0 at NaN and ±Inf, at every weight (an infinite weight restricts the
# entry to the zero set of the term, which is the whole line). The raw ε decides.
@testset "EpsInsensitiveLoss with infinite ε, $T" for T in (Float32, Float64)
  nan, inf = T(NaN), T(Inf)
  special = T[NaN, Inf, -Inf, 0, -0.0, 1, -2, floatmax(T), -floatmax(T)]
  for ε in (inf, Inf), w in (nothing, T[0, 1, Inf])
    f = EpsInsensitiveLoss(ε, w)
    for v in special, i in 1:3
      @test PB.value_entry(f, v, i) === zero(T)
    end
    # dense values
    for x in (T[NaN, Inf, -Inf], T[Inf, Inf, Inf], T[NaN, NaN, NaN], T[1, -2, 0], T[0, NaN, 1], T[-Inf, 1, NaN])
      @test f(x) === zero(T)
      @test f(x[1:3]) === zero(T)
    end
    # a long vector, past the blocks of the dense loop
    x = T[special[mod1(i, 9)] for i in 1:600]
    for k in (3, 100, 512, 513, 514, 600)
      f2 = EpsInsensitiveLoss(ε, w === nothing ? nothing : repeat(w, 200)[1:k])
      @test f2(x[1:k]) === zero(T)
      @test f2(fill(nan, k)) === zero(T)
    end
    # sparse iterates
    si = SparseIterate(T, 3)
    si[1] = nan; si[2] = inf
    @test f(si) === zero(T)
    fsym = EpsInsensitiveLoss(ε, w === nothing ? nothing : T[0 1; 1 Inf])
    @test fsym(SymmetricSparseIterate(T[NaN Inf; Inf -Inf])) === zero(T)
    # fused value, with a separate and the aliased output; the proximal map is the identity
    x = T[NaN, Inf, -Inf]
    v, va, y, z = fused(f, x, one(T))
    @test v === zero(T) && va === zero(T)
    @test isequal(y, x) && isequal(z, x)
    @test isequal(prox(f, x, one(T)), x)
    # Scaled
    g = Scaled(f, 2)
    @test g(x) == 0 && !isnan(g(x))
    @test g(SparseIterate(sparsevec(x))) == 0
  end
  # a weight type wider than the data
  T === Float32 && for w in (Float64[0, 1, Inf], nothing)
    f = EpsInsensitiveLoss(Inf, w)
    for v in special[1:3], i in 1:3
      @test PB.value_entry(f, v, i) === (w === nothing ? 0f0 : 0.0)
    end
  end
  # the conjugate: ε Σ|s| on the box |sᵢ| ≤ wᵢ, zero at the origin and Inf elsewhere in it
  for ε in (inf, Inf)
    c = Conjugate(EpsInsensitiveLoss(ε))
    @test c(zeros(T, 2)) === zero(T)
    @test c(T[0.5, 0]) === inf
    @test c(T[2, 0]) === inf
    y = fill(T(7), 2)
    v = value_and_prox!(y, c, zeros(T, 2), one(T))
    @test v === zero(T) && iszero(y)
    w = T[0, 1, Inf]
    cw = Conjugate(EpsInsensitiveLoss(ε, w))
    @test cw(zeros(T, 3)) === zero(T)
    @test cw(T[0, 0.5, 3]) === inf
    # an infinite weight does not bound its coordinate
    @test cw(T[0, 0, 5]) === inf
  end
  # a finite ε is unchanged: the entries NaN, Inf, -Inf, 1, 1/4, -2 and 5/4 at the weights 1 and
  # [0, 1, Inf]
  expected = (nothing => T[NaN Inf Inf 0.75 0 1.75 1; NaN Inf Inf 0.75 0 1.75 1; NaN Inf Inf 0.75 0 1.75 1],
              T[0, 1, Inf] => T[0 0 0 0 0 0 0; NaN Inf Inf 0.75 0 1.75 1; NaN Inf Inf Inf 0 Inf Inf])
  for (w, table) in expected
    f = EpsInsensitiveLoss(T(0.25), w)
    for (j, v) in enumerate(T[NaN, Inf, -Inf, 1, 0.25, -2, 1.25]), i in 1:3
      @test isequal(PB.value_entry(f, v, i), table[i, j])
    end
    @test f(T[0.25, 0, 0.25]) === zero(T)
    @test isnan(f(T[NaN, 1, 0])) === (w === nothing)
    @test f(T[0, 1.25, 0]) === T(1)
  end
end

@testset "JLArrays, $T" for T in (Float32, Float64)
  allowscalar(false)
  x = T[NaN, 1]
  for w in (T[0, 1], T[Inf, 1], T[1, 1]), f in (families(T, w)..., leveled(T, 0, w)...)
    fg = Adapt.adapt(JLArray, f)
    xg = JLArray(x)
    @test isequal(fg(xg), f(x))
    @test isequal(fg(xg), ref_value(f, x))
    yg = similar(xg)
    v = value_and_prox!(yg, fg, xg, one(T))
    @test isequal(v, fused(f, x, one(T))[1])
    @test isequal(Array(yg), prox(f, x, one(T)))
  end
  # EpsInsensitiveLoss with infinite ε is zero at NaN and ±Inf; NormL1Nonneg at NaN is NaN, and
  # at a negative entry Inf, whichever of the two comes with the other entries
  for w in (nothing, T[0, 1, Inf]), ε in (T(Inf), Inf)
    fg = Adapt.adapt(JLArray, EpsInsensitiveLoss(ε, w))
    for x in (T[NaN, Inf, -Inf], T[NaN, 1, 2], T[Inf, Inf, Inf], T[1, -2, 0])
      @test fg(JLArray(x)) === zero(T)
    end
    yg = similar(JLArray(T[NaN, Inf, -Inf]))
    @test value_and_prox!(yg, fg, JLArray(T[NaN, Inf, -Inf]), one(T)) === zero(T)
    @test isequal(Array(yg), T[NaN, Inf, -Inf])
  end
  for w in (nothing, T[0, 1, Inf], T[1, 2, 3]), λ in (0, 1, Inf)
    f = NormL1Nonneg(T(λ), w)
    fg = Adapt.adapt(JLArray, f)
    for x in (T[NaN, 1, 2], T[1, NaN, 0], T[NaN, -1, 2], T[-Inf, NaN, 0], T[NaN, NaN, NaN])
      @test isnan(fg(JLArray(x)))
      @test isnan(f(x))
    end
    for x in (T[-1, 1, 2], T[1, -Inf, 0], T[-1e-2, 0, 0])
      @test fg(JLArray(x)) === T(Inf) === f(x)
    end
    for x in (T[0, 0, 0], T[0.5, 0, 2], T[-PB._tol(T) / 2, 0, 0])
      @test fg(JLArray(x)) === f(x)
    end
    yg = similar(JLArray(T[NaN, 1, 2]))
    @test isnan(value_and_prox!(yg, fg, JLArray(T[NaN, 1, 2]), one(T)))
    @test isnan(Array(yg)[1])
  end
end

end
