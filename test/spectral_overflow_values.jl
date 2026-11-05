module SpectralOverflowValuesTest

# Standalone values of SingularValue, Spectral and Conjugate(SingularValue) at finite matrices
# whose spectrum overflows (#283, part 2b), and at NaN entries (E2). References are BigFloat
# formulas of f at an exact spectrum, or at the LAPACK spectrum of an unscaled matrix scaled
# exactly; homogeneity is an additional check only.

using Test, ProximalBase, LinearAlgebra
using ..TestUtils

const PB = ProximalBase
setprecision(BigFloat, 256)

# A5: provisional c = 64.
const C = 64

# The positive class (norms and absolute sums): relative 64 d eps(R)|ref|, with an absolute floor
# floatmin(R) for subnormal values (R4). A reference within the budget of floatmax accepts Inf.
function positive_ok(v, ref::BigFloat, ::Type{R}, d) where {R}
  v isa R || return false
  b = C * d * eps(R) * abs(ref) + floatmin(R)
  if abs(ref) + b >= floatmax(R)
    isinf(v) && return sign(v) == sign(ref)
    abs(ref) - b > floatmax(R) && return false
  end
  isfinite(v) && abs(big(v) - ref) <= b
end
# The signed class: absolute 64 p eps(R) λ ‖X‖₂^d.
function signed_ok(v, ref::BigFloat, ::Type{R}, scale::BigFloat, p) where {R}
  v isa R || return false
  b = C * p * eps(R) * scale + floatmin(R)
  if abs(ref) + b >= floatmax(R)
    isinf(v) && return sign(v) == sign(ref)
  end
  isfinite(v) && abs(big(v) - ref) <= b
end

# BigFloat values of f at a spectrum σ (singular values or eigenvalues), with the levels as stored.
topk(σ, k) = sum(sort(σ; rev = true)[1:k])
ref(f::NormL1{<:Real, Nothing}, σ) = big(f.λ0) * sum(abs, σ)
ref(f::NormL2, σ) = big(f.λ) * sqrt(sum(abs2, σ))
ref(f::NormLinf, σ) = big(f.λ) * maximum(abs, σ)
ref(f::SumLargest, σ) = big(f.λ) * topk(σ, f.k)
ref(f::SLOPE, σ) = sum(big.(f.λ) .* sort(abs.(σ); rev = true))
ref(f::NormL1Nonneg, σ) = big(f.λ0) * sum(σ)
ref(f::ElasticNet, σ) = big(f.λ1) * sum(abs, σ) + big(f.λ2) / 2 * sum(abs2, σ)
ref(f::SqrNormL2, σ) = big(f.λ) / 2 * sum(abs2, σ)
ref(f::CubeNormL2, σ) = big(f.λ) / 3 * sqrt(sum(abs2, σ))^3
ref(f::Scaled, σ) = big(f.α) * ref(f.f, σ)
ref(c::Conjugate, σ) = cref(c.f, σ)
cref(f::IndBallL1, σ) = big(f.r) * maximum(abs, σ)
cref(f::IndBallL2, σ) = big(f.r) * sqrt(sum(abs2, σ))
cref(f::IndBox, σ) = sum(v -> v >= 0 ? v * big(f.u) : v * big(f.l), σ)
cref(f::IndSimplex, σ) = big(f.r) * (f.equality ? maximum(σ) : max(maximum(σ), 0))
cref(f::SqrNormL2, σ) = sum(abs2, σ) / (2 * big(f.λ))
# Ky Fan: the box [0, u] with the halfspace Σw ≤ u k
function cref(c::PB.LinearConstrained, σ)
  u, k = big(c.f.u), big(c.set.b) / big(c.f.u)
  t = sort(max.(σ, 0); rev = true)
  n = floor(Int, k)
  u * (sum(t[1:n]) + (k - n) * (n < length(t) ? t[n + 1] : 0))
end
ref(c::PB.BoxConstrained, σ) = ref(c.f, σ)
# The support function of {w : l ≤ w ≤ u, Σw ≤ β} at y, in BigFloat: w = l + (u - l) t with
# 0 ≤ t ≤ 1 and Σt ≤ κ = (β - n l)/(u - l), so the value is l Σy plus u - l times the sum of the
# ⌊κ⌋ largest positive parts and the fraction of the next.
function kyfan_ref(l, u, β, y)
  n = length(y)
  l, u, β, y = big(l), big(u), big(β), big.(y)
  κ = clamp((β - n * l) / (u - l), 0, n)
  t = sort(max.(y, 0); rev = true)
  j = floor(Int, κ)
  l * sum(y) + (u - l) * (sum(t[1:j]; init = big(0)) + (j < n ? (κ - j) * t[j + 1] : 0))
end

degree(f) = 1
degree(::Union{SqrNormL2, Conjugate{<:SqrNormL2}}) = 2
degree(::CubeNormL2) = 3
degree(f::Scaled) = degree(f.f)
degree(f::ElasticNet) = iszero(f.λ2) ? 1 : 2

# exact spectra; Mx = floatmax
square_fill(T) = fill(floatmax(T), 2, 2)                     # σ = (2M, 0), λ = (0, 2M)
# R D Rᵀ with R = [3 -4; 4 3]/5 and D = diag(a, b), a = 25(3/4)2^e, b = 25(1/4)2^e: exact entries
function rotation(T)
  e = exponent(floatmax(T)) - 3
  X = T[10.75 6; 6 14.25] .* ldexp(one(T), e)
  X, [25 * big(3) / 4, 25 * big(1) / 4] .* big(2.0)^e      # eigenvalues = singular values (PD)
end

@testset "Spectral values at an overflowing spectrum" begin
  for T in (Float32, Float64)
    M = floatmax(T)
    A = square_fill(T)
    σA = [2big(M), big(0)]
    λA = [big(0), 2big(M)]
    X, σX = rotation(T)
    @test all(isfinite, X)
    @test isinf(svdvals(X)[1])
    q = T(1) / 4
    tiny = nextfloat(zero(T))
    # degree-2 levels whose value at an overflowing spectrum is finite are subnormal in T
    l2 = T === Float64 ? 1e-310 : ldexp(1f0, -140)
    levels = (q, tiny, ldexp(one(T), -100))

    @testset "degree one ($T)" begin
      for λ in levels, mk in (NormL1, NormL2, NormLinf, l -> ElasticNet(l, zero(T)), l -> SumLargest(2, l), NormL1Nonneg,
                              l -> Conjugate(IndBallL1(l)), l -> Conjugate(IndBallL2(l)))
        f = mk(λ)
        # a subnormal level of SumLargest, which has no fold, is recovered by its value parts (#608)
        for (Z, σ, λs) in ((A, σA, λA), (X, σX, reverse(σX)))
          @test positive_ok(SingularValue(f)(Z), ref(f, σ), T, 1)
          PB._ispermutationinvariant(f) && @test positive_ok(Spectral(f)(Z), ref(f, λs), T, 1)
        end
      end
      # support functions of a box and a simplex, at ordinary levels
      for f in (Conjugate(IndBallLinf(q)), Conjugate(IndBox(-T(1) / 2, q)), Conjugate(IndSimplex(q)),
                Conjugate(IndSimplex(q; equality = false)),
                Conjugate(Constrained(IndBox(zero(T), q), IndHalfspace(1, q))))
        @test positive_ok(SingularValue(f)(A), cref(f.f, σA), T, 1)
        @test positive_ok(SingularValue(f)(X), cref(f.f, σX), T, 1)
        @test positive_ok(Spectral(f)(X), cref(f.f, reverse(σX)), T, 1)
      end
      # SLOPE is absolutely symmetric; SingularValue only (Spectral rejects its weights)
      f = SLOPE(T[1 / 4, 1 / 8])
      @test positive_ok(SingularValue(f)(A), ref(f, σA), T, 1)
      @test positive_ok(SingularValue(f)(X), ref(f, σX), T, 1)
      # weighted NormL1 in SingularValue
      f = NormL1(q, T[1, 1 / 2])
      @test positive_ok(SingularValue(f)(X), big(q) * (σX[1] + σX[2] / 2), T, 1)
    end

    @testset "zero margin and mixed levels ($T)" begin
      # A Float64 level that the factor folds below the normal range of Float32.
      if T === Float32
        f = NormL1(2.0^-254)
        @test positive_ok(SingularValue(f)(A), big(2.0)^-254 * 2big(M), T, 1)
        @test positive_ok(SingularValue(NormL1(0.25))(A), big(0.25) * 2big(M), T, 1)
        @test positive_ok(SingularValue(SqrNormL2(2.0^-300))(A), big(2.0)^-300 / 2 * 4big(M)^2, T, 2)
        # degree three: only a wider level gives a finite value, which the value parts recover
        @test positive_ok(SingularValue(CubeNormL2(2.0^-300))(A), big(2.0)^-300 / 3 * (2big(M))^3, T, 3)
      end
    end

    @testset "degree two ($T)" begin
      for f in (SqrNormL2(l2), SqrNormL2(tiny), ElasticNet(zero(T), l2), ElasticNet(zero(T), tiny),
                ElasticNet(ldexp(one(T), -100), l2))
        for (Z, σ) in ((A, σA), (X, σX))
          @test positive_ok(SingularValue(f)(Z), ref(f, σ), T, 2)
          @test positive_ok(Spectral(f)(Z), ref(f, σ), T, 2)
        end
      end
      # the conjugate of the squared norm, (λ/2‖⋅‖²)* = ‖⋅‖²/(2λ)
      Y = fill(T(5) / 8 * M, 2, 2)
      σY = [5big(M) / 4, big(0)]
      for f in (Conjugate(SqrNormL2(M)), Conjugate(Scaled(SqrNormL2(M / 2), T(2))))
        @test positive_ok(SingularValue(f)(Y), sum(abs2, σY) / (2big(M)), T, 2)
        @test positive_ok(Spectral(f)(Y), sum(abs2, σY) / (2big(M)), T, 2)
      end
      @test positive_ok(Conjugate(SingularValue(SqrNormL2(M)))(Y), sum(abs2, σY) / (2big(M)), T, 2)
      if T === Float32
        @test positive_ok(Conjugate(SingularValue(SqrNormL2(2.0^200)))(A), 4big(M)^2 / (2big(2.0)^200), T, 2)
      end
    end

    @testset "scale chains ($T)" begin
      h = T === Float64 ? 600 : 100
      f = Scaled(Scaled(NormL1(one(T)), ldexp(one(T), -h)), ldexp(one(T), -h))
      @test positive_ok(SingularValue(f)(A), big(2.0)^(-2h) * 2big(M), T, 1)
      @test positive_ok(Spectral(f)(X), big(2.0)^(-2h) * sum(σX), T, 1)
      f = Scaled(NormL1(one(T)), tiny)
      @test positive_ok(SingularValue(f)(A), big(tiny) * 2big(M), T, 1)
      f = Scaled(Scaled(NormL1(M), ldexp(one(T), -h)), ldexp(one(T), -h))
      @test positive_ok(SingularValue(f)(X), big(M) * big(2.0)^(-2h) * sum(σX), T, 1)
      f = Scaled(ElasticNet(ldexp(one(T), -100), l2), T(1) / 2)
      @test positive_ok(SingularValue(f)(A), ref(f, σA), T, 2)
      # zero scales and levels stay exact zeros
      for f in (Scaled(NormL1(one(T)), zero(T)), NormL1(zero(T)), SqrNormL2(zero(T)), ElasticNet(zero(T), zero(T)))
        @test SingularValue(f)(A) === zero(T)
        @test Spectral(f)(A) === zero(T)
      end
    end

    @testset "signed spectra ($T)" begin
      B = [A zeros(T, 2, 2); zeros(T, 2, 2) -A]
      λB = [-2big(M), big(0), big(0), 2big(M)]
      for k in 1:4
        f = SumLargest(k, q)
        @test signed_ok(Spectral(f)(B), ref(f, λB), T, big(q) * 2big(M), 4)
      end
      f = SumLargest(2, q)
      @test signed_ok(Spectral(f)(-A), ref(f, -λA), T, big(q) * 2big(M), 2)
      @test Spectral(SumLargest(2))(-A) === -T(Inf)          # genuine overflow
      @test positive_ok(Spectral(NormL1(q))(-A), big(q) * 2big(M), T, 1)
      f = Conjugate(IndBox(-T(1) / 2, q))
      @test signed_ok(Spectral(f)(-A), cref(f.f, -λA), T, big(1) / 2 * 2big(M), 2)   # the floatmax boundary
      for f in (Conjugate(IndSimplex(q)), Conjugate(IndSimplex(q; equality = false)))
        @test signed_ok(Spectral(f)(-A), cref(f.f, -λA), T, big(q) * 2big(M), 2)
        @test signed_ok(Spectral(f)(B), cref(f.f, λB), T, big(q) * 2big(M), 4)
      end
      @test signed_ok(Spectral(SumLargest(1, q))(-X), -big(q) * σX[2], T, big(q) * σX[1], 2)
    end

    @testset "decided functions ($T)" begin
      sA = sum(σA)
      for (f, r, d) in ((Constrained(NormL1(q), IndBox(0, Inf)), big(q) * sA, 1),
                        (Constrained(SqrNormL2(l2), IndBox(0, Inf)), big(l2) / 2 * sum(abs2, σA), 2),
                        (Constrained(NormL1(q), IndHalfspace(-1, 0)), big(q) * sA, 1),
                        (Scaled(Constrained(NormL1(one(T)), IndBox(0, Inf)), q), big(q) * sA, 1),
                        (Constrained(Scaled(NormL1(one(T)), q), IndBox(0, Inf)), big(q) * sA, 1),
                        (Scaled(NormL1Nonneg(one(T)), q), big(q) * sA, 1),
                        (Scaled(Scaled(NormL1Nonneg(one(T)), ldexp(one(T), -60)), ldexp(one(T), 58)), big(q) * sA, 1))
        @test positive_ok(SingularValue(f)(A), r, T, d)
        @test positive_ok(Spectral(f)(A), r, T, d)
      end
      # infeasible points stay Inf, at overflowing and finite spectra
      @test Spectral(NormL1Nonneg(q))(-A) === T(Inf)
      @test Spectral(Constrained(NormL1(q), IndBox(0, Inf)))(-A) === T(Inf)
      @test SingularValue(Constrained(NormL1(q), IndSimplex(one(T))))(A) === T(Inf)
      @test Spectral(NormL1Nonneg(one(T)))(T[-1 0; 0 1]) === T(Inf)
      @test Spectral(Constrained(NormL1(one(T)), IndBox(0, Inf)))(T[-1 0; 0 1]) === T(Inf)
    end

    @testset "conjugates of scaled norms ($T)" begin
      for (f, g) in ((Conjugate(Scaled(NormL1(M), T(4))), Scaled(NormL1(M), T(4))),
                     (Conjugate(Scaled(NormL2(M), T(4))), Scaled(NormL2(M), T(4))),
                     (Conjugate(Scaled(NormLinf(M), T(4))), Scaled(NormLinf(M), T(4))))
        @test SingularValue(f)(A) === zero(T)
        @test Spectral(f)(A) === zero(T)
        @test Conjugate(SingularValue(g))(A) === zero(T)
        # the radius 4M is never formed: at fill(M, 8, 8), σ₁ = 8M exceeds it
        @test SingularValue(f)(fill(M, 8, 8)) === T(Inf)
        @test Conjugate(SingularValue(g))(fill(M, 8, 8)) === T(Inf)
      end
      # ordinary calls keep the conjugate's own test
      @test Conjugate(Scaled(NormL1(M), T(4)))([M, M]) === zero(T)
      @test SingularValue(Conjugate(Scaled(NormL1(one(T)), T(2))))(T[1 0; 0 1]) === zero(T)
      @test SingularValue(Conjugate(Scaled(NormL1(one(T)), T(2))))(T[3 0; 0 0]) === T(Inf)
    end

    @testset "Conjugate(SingularValue(g)) ($T)" begin
      for g in (IndBallL1(q), IndBallL2(q), IndBallLinf(q), IndSimplex(q), Scaled(IndBallL1(q), T(2)))
        h = g isa Scaled ? g.f : g           # a positive scale does not change an indicator
        @test positive_ok(Conjugate(SingularValue(g))(A), cref(h, σA), T, 1)
        @test positive_ok(Conjugate(SingularValue(g))(X), cref(h, σX), T, 1)
      end
      @test Conjugate(SingularValue(NormL1(q)))(A) === T(Inf)
    end

    @testset "scaled balls at extreme radii ($T)" begin
      # σ of Scaled(ι_C, α) is σ_C for α > 0, so the reference is r times the dual norm of the
      # spectrum, whatever the scales (subnormal, tiny or large) around the ball.
      radii = (tiny, floatmin(T), ldexp(one(T), -100), T === Float64 ? 1.234567e-318 : 1.2345f-42)
      scales = (T(2), T(1) / 3, ldexp(one(T), -60), tiny, floatmax(T) / 4 * 3)
      for r in radii, α in scales, (B, ref) in ((IndBallL1, (r, σ) -> big(r) * maximum(abs, σ)),
                                               (IndBallL2, (r, σ) -> big(r) * sqrt(sum(abs2, σ))))
        for (Z, σ, λs) in ((A, σA, λA), (X, σX, reverse(σX)))
          g = Scaled(B(r), α)
          @test positive_ok(Conjugate(SingularValue(g))(Z), ref(r, σ), T, 1)
          @test positive_ok(SingularValue(Conjugate(g))(Z), ref(r, σ), T, 1)
          @test positive_ok(Spectral(Conjugate(g))(Z), ref(r, λs), T, 1)
          @test positive_ok(SingularValue(Conjugate(Scaled(Scaled(B(r), α), T(2))))(Z), ref(r, σ), T, 1)
        end
      end
      if T === Float32
        # mixed: a Float64 radius, subnormal in Float32 and normal in Float64, at a Float32 matrix
        Y = fill(T(0.7) * M, 2, 2)
        σY = [2big(T(0.7) * M), big(0)]
        for r in (1.2345e-42, 1e-50), B in (IndBallL1, IndBallL2)
          @test positive_ok(Conjugate(SingularValue(Scaled(B(r), 2.0)))(Y), big(r) * σY[1], T, 1)
          @test positive_ok(SingularValue(Conjugate(Scaled(B(r), 3.0)))(Y), big(r) * σY[1], T, 1)
        end
      end
    end

    @testset "Ky Fan ($T)" begin
      kf = Constrained(IndBox(zero(T), q), IndHalfspace(1, q))
      @test positive_ok(Conjugate(kf)([M, M]), big(q) * big(M), T, 1)
      @test positive_ok(SingularValue(Conjugate(kf))(Matrix(M * I, 2, 2)), big(q) * big(M), T, 1)
      @test positive_ok(SingularValue(Conjugate(kf))(A), big(q) * 2big(M), T, 1)
      @test Conjugate(kf)(T[1, 2]) === T(1) / 2                      # ordinary, unchanged
      @test Conjugate(Constrained(IndBox(zero(T), zero(T)), IndHalfspace(1, zero(T))))([M, M]) === zero(T)
      # u - l and n l overflow at a finite argument: no InexactError
      @test Conjugate(Constrained(IndBox(-M, M), IndHalfspace(1, zero(T))))(T[1, 1]) === zero(T)
      capped = Constrained(IndBox(zero(T), q), IndHyperplane(1, q))
      @test positive_ok(Conjugate(capped)([M, M]), big(q) * big(M), T, 1)
      @test positive_ok(Conjugate(capped)([M, -M]), big(q) * big(M), T, 1)
      # Bounds about a thousand binades apart: each term at its own exponent, so the lower bound
      # and the offset are not flushed. The halfspace pins the sum at n l (κ = 0), or just above it
      # (0 < κ below the range).
      e = exponent(M)
      lo, hi = ldexp(one(T), -(e ÷ 8)), ldexp(one(T), 7e ÷ 8)
      for (l, u, β, y) in ((lo, hi, 2lo, [M, M]), (lo, M, 2lo, [M, M]), (lo, hi, 3lo, [M, M / 2]),
                           (lo, M, 3lo, [M, M / 2]), (lo, hi, 3lo, [M, -M / 2]),
                           (-M, M, zero(T), T[1, 1]), (-M, M, M, [M / 4, -M / 4, M / 8]))
        f = Conjugate(Constrained(IndBox(l, u), IndHalfspace(1, β)))
        @test positive_ok(f(y), kyfan_ref(l, u, β, y), T, 1)
      end
      # the reproducers, at the exponents of Float64
      if T === Float64
        @test positive_ok(Conjugate(Constrained(IndBox(1e-30, 1e300), IndHalfspace(1.0, 2e-30)))([M, M]),
                          kyfan_ref(1e-30, 1e300, 2e-30, [M, M]), T, 1)
        # a finite spectrum (M, M) whose sum overflows
        kf = Conjugate(Constrained(IndBox(1e-300, M), IndHalfspace(1.0, 2e-300)))
        @test positive_ok(SingularValue(kf)(Matrix(M * I, 2, 2)), kyfan_ref(1e-300, M, 2e-300, [M, M]), T, 1)
        kf = Conjugate(Constrained(IndBox(1e-300, 1e300), IndHalfspace(1.0, 3e-300)))
        @test positive_ok(SingularValue(kf)(Matrix(Diagonal([M, M / 2]))), kyfan_ref(1e-300, 1e300, 3e-300, [M, M / 2]), T, 1)
        @test positive_ok(Spectral(kf)(Matrix(Diagonal([M, M / 2]))), kyfan_ref(1e-300, 1e300, 3e-300, [M, M / 2]), T, 1)
        # O3: at a spectrum whose sum is finite the ordinary kernel is used, and its κ = 1e-600
        # underflows, which loses the term (u - l) κ max v (unchanged from master)
        @test_broken positive_ok(Spectral(kf)(Matrix(Diagonal([M, -M / 2]))), kyfan_ref(1e-300, 1e300, 3e-300, [M, -M / 2]), T, 1)
      end
      # A known departure from E3, kept until the follow-up on the vector kernels (O9): at an
      # infinite entry the kernel's value is NaN, where E3 gives Inf.
      @test isnan(Conjugate(Constrained(IndBox(-M, M), IndHalfspace(1, zero(T))))(T[Inf, 1]))
      @test isnan(Conjugate(Constrained(IndBox(zero(T), q), IndHalfspace(1, q)))(T[Inf, 1]))
    end

    @testset "rectangular and rank-deficient ($T)" begin
      for Z in (fill(M, 2, 3), fill(M, 3, 2))
        σ = [sqrt(big(6)) * big(M), big(0)]
        for f in (NormL1(q), NormL2(q), SqrNormL2(l2), Conjugate(IndBallL1(q)))
          @test positive_ok(SingularValue(f)(Z), ref(f, σ), T, degree(f))
        end
      end
    end

    @testset "LAPACK spectra of unscaled matrices ($T)" begin
      for (m, n, seed) in ((3, 3, 1), (5, 3, 2), (3, 5, 3))
        B = T.(0.5 .+ 0.5 .* abs.(sin.(seed .* (1:m) .+ (1:n)')))        # deterministic, entries in [0.5, 1)
        B ./= 2 * maximum(abs, B) / 1.9
        maximum(abs, B) < 1 || continue
        Z = ldexp.(B, exponent(M) + 1)
        @test all(isfinite, Z)
        σ = big.(svdvals(B)) .* big(2.0)^(exponent(M) + 1)
        σ[1] > big(M) || continue
        for f in (NormL1(q), NormL2(q), NormLinf(q), SumLargest(2, q), ElasticNet(q, zero(T)),
                  SqrNormL2(l2), Conjugate(IndBallL1(q)), Conjugate(IndSimplex(q)))
          @test positive_ok(SingularValue(f)(Z), ref(f, σ), T, degree(f))
        end
        if m == n
          S = Matrix(Symmetric(Z))
          λ = big.(eigvals(Symmetric(Matrix(Symmetric(B))))) .* big(2.0)^(exponent(M) + 1)
          for f in (NormL1(q), NormL2(q), SqrNormL2(l2))
            @test positive_ok(Spectral(f)(S), ref(f, λ), T, degree(f))
          end
          f = SumLargest(2, q)
          @test signed_ok(Spectral(f)(S), ref(f, λ), T, big(q) * maximum(abs, λ), n)
        end
        # homogeneity, an additional check: F(2^k B) = 2^k F(B)
        k = exponent(M) + 1
        @test isapprox(big(SingularValue(NormL1(q))(Z)), big(2.0)^k * big(SingularValue(NormL1(q))(B)); rtol = C * eps(T))
      end
    end

    @testset "genuine overflow and not covered ($T)" begin
      @test SingularValue(NormL1(one(T)))(A) === T(Inf)
      @test SingularValue(CubeNormL2(T === Float64 ? 2.0^-1000 : 2f0^-140))(A) === T(Inf)
      @test SingularValue(SqrNormL2(T(2)^-20))(A) === T(Inf)
      # not covered: the ordinary result is kept
      for f in (NormLq(q, T(1) / 2), LogPenalty(one(T), one(T)), Translate(NormL1(q), zeros(T, 2)),
                SqrNormL2(l2, ones(T, 2)), TotalVariation1D(q), IndSOC())
        @test SingularValue(f)(A) === T(Inf)
      end
      # levels outside the normal range of T without a fold or value parts: not covered (O3)
      Z = X
      for f in (Conjugate(IndSimplex(tiny)), Conjugate(IndBox(-tiny, tiny)))
        @test Spectral(f)(Z) === T(Inf)          # no wrong finite value: the ordinary result
      end
      # A known departure from E3, kept until the follow-up on the vector kernels (O9): the Ky Fan
      # kernel's value at the infinite eigenvalue is NaN (l Σy with l = 0), at a finite matrix.
      @test isnan(Spectral(Conjugate(Constrained(IndBox(zero(T), tiny), IndHalfspace(1, tiny))))(Z))
      # SLOPE with a subnormal level: its value parts (#608)
      @test positive_ok(SingularValue(SLOPE([tiny, zero(T)]))(Z), big(tiny) * σX[1], T, 1)
      @test positive_ok(SingularValue(SumLargest(1, tiny))(Z), big(tiny) * σX[1], T, 1)
    end

    @testset "workspace reuse and arguments ($T)" begin
      for F in (SingularValue(NormL1(q)), Spectral(SqrNormL2(l2)), SingularValue(ElasticNet(q, l2)),
                Spectral(Constrained(NormL1(q), IndBox(0, Inf))), Conjugate(SingularValue(IndBallL1(q))))
        P = T[2 1; 1 3]
        P0, A0 = copy(P), copy(A)
        a = F(P)
        b = F(A)
        c = F(P)
        @test a === c
        @test isfinite(b)
        @test P == P0 && A == A0
      end
    end

    @testset "non-finite entries ($T)" begin
      for F in (SingularValue(NormL1(q)), SingularValue(SqrNormL2(l2)), Spectral(NormL1(q)), Spectral(NormL1Nonneg(q)),
                Spectral(Constrained(NormL1(q), IndBox(0, Inf))), Conjugate(SingularValue(IndBallL1(q))),
                NuclearNorm(q), TracePSD(q), IndPSD(), IndBallRank(1), IndStiefel(), LogDetLoss(Matrix{T}(I, 2, 2)))
        for v in (T(NaN), T(Inf), -T(Inf)), Z in (T[v 0; 0 1], T[1 v; v 1], T[1 v; 0 1])
          @test isequal(F(Z), isnan(v) ? T(NaN) : T(Inf))
          F isa Conjugate && continue
          Y = similar(Z)
          @test isequal(value_and_prox!(Y, F, Z, one(T)), isnan(v) ? T(NaN) : T(Inf))
          # an aliased destination: the value is decided before it is written
          W = copy(Z)
          @test isequal(value_and_prox!(W, F, W, one(T)), isnan(v) ? T(NaN) : T(Inf))
        end
      end
    end
  end
end

end
