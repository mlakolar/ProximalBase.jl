module SpectralOverflowValuesTest

# Standalone values of SingularValue, Spectral and Conjugate(SingularValue) at finite matrices
# whose spectrum overflows (#283, part 2b), and at NaN entries (E2). References are BigFloat
# formulas of f at an exact spectrum, or at the LAPACK spectrum of an unscaled matrix scaled
# exactly; homogeneity is an additional check only.

using Test, ProximalBase, LinearAlgebra, Random
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

# Weighted functions (T1): a finite value is within the budget of its reference; a value that is
# not finite is the ordinary result, which the recovery keeps.
function weighted_ok(v, ref::BigFloat, ::Type{R}, p) where {R}
  v isa R || return false
  isfinite(v) || return true
  ref > big(floatmax(R)) && return false
  abs(big(v) - ref) <= C * p * eps(R) * ref + 4p * big(nextfloat(zero(R)))
end

# Warmed bytes of a call at a non-finite input, which never fills a cache of f (a cold cache).
nonfinite_bytes(F, Z) = (F(Z); F(Z); @allocated F(Z))

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
ref(c::PB.BoxConstrained, σ) = ref(c.f, σ)
# The support function of {l ≤ x ≤ u, a Σx ≤ b} (or = b) at y, in 4096-bit BigFloat, which the
# cancellation of its two terms needs: σ_C(y) = σ_{-C}(-y) turns a < 0 into a > 0.
function box_linear_ref(l, u, a, b, y, eq)
  setprecision(BigFloat, 4096) do
    n = length(y)
    l, u, b, y = big(l), big(u), big(b) / big(a), big.(y)
    if a < 0
      l, u, b, y = -u, -l, -b, -y
    end
    κ = clamp((b - n * l) / (u - l), 0, n)
    t = sort(eq ? y : max.(y, 0); rev = true)
    j = floor(Int, κ)
    BigFloat(l * sum(y) + (u - l) * (sum(t[1:j]; init = big(0)) + (j < n ? (κ - j) * t[j + 1] : 0)); precision = 256)
  end
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
                Conjugate(IndSimplex(q; equality = false)))
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

    @testset "box with a linear constraint, not recovered (#648) ($T)" begin
      # The support function of {l ≤ x ≤ u, a Σx ≤ b} (or = b), such as the Ky Fan norm, keeps the
      # kernel of master, which #648 owns: each case is broken against its BigFloat reference, and
      # master's current class (NaN, ±Inf, the 0.0 of #642, or an InexactError) is pinned.
      kf(l, u, a, b, eq) = Conjugate(Constrained(IndBox(l, u), eq ? IndHyperplane(a, b) : IndHalfspace(a, b)))
      vector_cases = T === Float64 ?
        ((1e-30, 1e300, 1.0, 2e-30, [M, M], :inf),           # bounds a thousand binades apart
         (-1e10, -1e-10, 1.0, -2e-10, [M, M], :nan),         # l Σy and (u - l) Σv cancel
         (-1e10, -1e-10, 1.0, -2e-10, [1.0, 1.0], :zero),    # #642, at ordinary scale
         (-1e12, -1e-3, 1.0, 2(-1e-3) - 3e-5, [M, M / 2], :nan),   # B4: κ rounds to n
         (-4.0, -2.0, 1.0, -5.0, [M / 8, M / 4], :inf),      # #647: fractional κ
         (-M, M, 1.0, 0.0, [1.0, 1.0], :inexact),            # u - l overflows: master throws
         (-1e10, -1e-10, 1.0, -1.5e10, [M, M, -M / 2], :nan),  # exact value 0
         (0.0, 0.25, 1.0, 0.25, [M, M], :nan)) :             # the Ky Fan 1-norm
        ((1f-5, 1f30, 1f0, 2f-5, [M, M], :inf),
         (-1f10, -1f-10, 1f0, -2f-10, [M, M], :nan),
         (-1f10, -1f-10, 1f0, -2f-10, [1f0, 1f0], :zero),
         (-1f12, -1f-3, 1f0, 2(-1f-3) - 3f-5, [M, M / 2], :nan),
         (-4f0, -2f0, 1f0, -5f0, [M / 8, M / 4], :inf),
         (-M, M, 1f0, 0f0, [1f0, 1f0], :inexact),
         (-8f9, -1f-10, 1f0, -1.2f10, [M, M, -M / 2], :nan),    # exact value 0 (#648)
         (0f0, 0.25f0, 1f0, 0.25f0, [M, M], :nan))
      # B6: box(-(1 - 2^-j), 1) with the halfspace Σx ≤ 2^-j at [M, M], condition 15 to 2e9
      js = T === Float64 ? (3:30) : (3:20)
      cases = (vector_cases..., ((-(1 - ldexp(one(T), -j)), one(T), one(T), ldexp(one(T), -j), [M, M], :nan) for j in js)...)
      for (l, u, a, b, y, class) in cases, eq in (false, true)
        f = kf(l, u, a, b, eq)
        r = box_linear_ref(l, u, a, b, y, eq)
        if class === :inexact
          @test_throws InexactError f(y)
          @test_broken (try positive_ok(f(y), r, T, 1) catch; false end)   # #648
          continue
        end
        v = f(y)
        @test_broken positive_ok(v, r, T, 1)                                  # #648
        class === :nan && @test isnan(v)
        class === :inf && @test v === copysign(T(Inf), r)
        class === :zero && @test v === zero(T)                                 # #642
      end
      # through the matrix wrappers, at overflowing and finite spectra
      kq = kf(zero(T), q, one(T), q, false)
      for (F, Z, r) in ((SingularValue(kq), Matrix(M * I, 2, 2), big(q) * big(M)),      # spectral.kyfan-finite
                        (SingularValue(kq), A, big(q) * 2big(M)),
                        (Spectral(kq), X, big(q) * σX[1]),
                        (SingularValue(kf(zero(T), q, one(T), q, true)), A, big(q) * 2big(M)))
        v = F(Z)
        @test_broken positive_ok(v, r, T, 1)                                  # #648
        @test isnan(v)
      end
      # #647 at ordinary scale: the fraction of κ selects in input order
      for eq in (false, true)
        v = Spectral(kf(zero(T), one(T), one(T), T(3) / 2, eq))(T[1 0; 0 2])
        @test_broken v == T(5) / 2                                             # #647, #648
        @test v == 2
      end
      if T === Float64
        F = SingularValue(kf(1e-300, M, 1.0, 2e-300, false))
        @test_broken positive_ok(F(Matrix(M * I, 2, 2)), box_linear_ref(1e-300, M, 1.0, 2e-300, [M, M], false), T, 1)   # #648
        @test F(Matrix(M * I, 2, 2)) === Inf
        # κ ≈ 1e-600 underflows at a finite spectrum
        F = Spectral(kf(1e-300, 1e300, 1.0, 3e-300, false))
        @test_broken positive_ok(F(Matrix(Diagonal([M, -M / 2]))), box_linear_ref(1e-300, 1e300, 1.0, 3e-300, [M, -M / 2], false), T, 1)   # #648
      end
      # ordinary values, unchanged
      @test kf(zero(T), q, one(T), q, false)(T[1, 2]) === T(1) / 2
      @test kf(zero(T), zero(T), one(T), zero(T), false)(T[1, 2]) === zero(T)
      # At an infinite entry the kernel gives NaN where E3 gives Inf, or throws (#648)
      @test isnan(kf(zero(T), q, one(T), q, false)(T[Inf, 1]))
      @test_throws InexactError kf(-M, M, one(T), zero(T), false)(T[Inf, 1])
    end

    @testset "weighted functions ($T)" begin
      # A product of the level and a weight can underflow although each factor is normal: a small
      # value at the scaled spectrum is recovered only by a fold into a normal level whose value is
      # normal, or exactly zero term by term; otherwise the ordinary result stays.
      E = exponent(floatmax(T)) - 3                       # σX = 25(3/4, 1/4) 2^E
      wsum(w, σ) = sum(big.(w) .* σ)
      # the examples of the review: no fold reaches the products, the ordinary result stays
      if T === Float64
        @test isnan(SingularValue(Scaled(NormL1(2.0^-600, fill(2.0^-600, 2)), 2.0^1000))(X))   # 25⋅2^820
        @test SingularValue(NormL1(4.0, fill(tiny, 2)))(X) === Inf                             # 100⋅2^-54
      else
        @test SingularValue(NormL1(4f0, fill(1e-50, 2)))(X) === Inf32                          # 2.1267647f-11
      end
      small = (fill(tiny, 2), fill(nextfloat(0.0), 2), fill(ldexp(1.0, -1060), 2))
      for w in small, F in (SingularValue(NormL1(T(4), w)), SingularValue(NormL1Nonneg(T(4), w)),
                            SingularValue(ElasticNet(T(4), zero(T), w)), SingularValue(ElasticNet(T(4), l2, w)),
                            SingularValue(Scaled(NormL1(T(4), w), T(2))),
                            SingularValue(Constrained(NormL1(T(4), w), IndBox(0, Inf))),
                            SingularValue(Constrained(NormL1(T(4), w), IndHalfspace(-one(T), zero(T)))),
                            SingularValue(Scaled(Constrained(NormL1Nonneg(T(4), w), IndBox(0, Inf)), T(2))))
        v = F(X)
        @test weighted_ok(v, 4 * wsum(w, σX), T, 2)
        @test !isfinite(v)                                  # the level 4 folds out of range
      end
      # recovered: the product of level and weight underflows at the scaled spectrum, the fold does not
      λ = ldexp(one(T), -20)
      for w in (fill(ldexp(one(T), 2 - exponent(floatmax(T))), 2), fill(ldexp(1.0, -150), 2), [ldexp(one(T), -110), zero(T)])
        r = big(λ) * wsum(w, σX)
        for f in (NormL1(λ, w), NormL1Nonneg(λ, w), ElasticNet(λ, zero(T), w), Constrained(NormL1(λ, w), IndBox(0, Inf)),
                  Constrained(NormL1(λ, w), IndHalfspace(-one(T), zero(T))))
          v = SingularValue(f)(X)
          @test isfinite(v) && weighted_ok(v, r, T, 2)
        end
        v = SingularValue(Scaled(NormL1(λ, w), T(1) / 4))(X)
        @test isfinite(v) && weighted_ok(v, r / 4, T, 2)
      end
      # exact zeros term by term: zero weights
      for f in (NormL1(q, zeros(T, 2)), NormL1Nonneg(q, zeros(T, 2)), ElasticNet(q, zero(T), zeros(T, 2)))
        @test SingularValue(f)(X) === zero(T)
        @test SingularValue(f)(A) === zero(T)
      end
      # a seeded sweep of levels, weights (in T and Float64) and outer scales at exact spectra:
      # σX, and √2 M twice for Q
      Q = T[1 1; -1 1] .* M
      σQ = fill(sqrt(big(2)) * big(M), 2)
      rng = MersenneTwister(384)
      nrec = 0
      for trial in 1:120, WT in (T, Float64)
        Z, σ = rand(rng, Bool) ? (X, σX) : (Q, σQ)
        e(S, lo) = ldexp(S(1 + rand(rng)), rand(rng, lo:exponent(floatmax(S))))
        w = [rand(rng) < 0.15 ? zero(WT) : e(WT, exponent(floatmin(WT)) - precision(WT) + 1) for _ in 1:2]
        λ, α = e(T, exponent(floatmin(T))), e(T, exponent(floatmin(T)))
        r = big(λ) * wsum(w, σ)
        for (f, s) in ((NormL1(λ, w), 1), (NormL1Nonneg(λ, w), 1), (ElasticNet(λ, zero(T), w), 1),
                       (Scaled(NormL1(λ, w), α), big(α)), (Constrained(NormL1(λ, w), IndBox(0, Inf)), 1))
          v = SingularValue(f)(Z)
          @test weighted_ok(v, s * r, T, 2)
          nrec += isfinite(v) && !iszero(r)
        end
      end
      @test nrec > 0
      # each homogeneous part of ElasticNet is recovered or declined on its own: here the linear
      # part cannot fold, so the whole value keeps the ordinary result
      T === Float32 && @test SingularValue(ElasticNet(2.0^-300, 2.0^-260, fill(2.0^-100, 2)))(X) === Inf32
    end

    @testset "infinite levels (decision 49) ($T)" begin
      # The domain of an infinite level is {0}: the value is 0 at the zero matrix and Inf at every
      # other one, also at an overflowing spectrum, where no recovery takes an infinite level; NaN at
      # a NaN entry (E2). An infinite level is not a constant function.
      for f in (NormL1(T(Inf)), NormL1(T(Inf), T[1, 0]), SLOPE(T[Inf, 1]), NormL1Fused(T(Inf), one(T)),
                Scaled(NormL1(T(Inf)), T(2)), Constrained(NormL1(T(Inf)), IndBox(0, Inf)))
        F = SingularValue(f)
        @test PB._constant_kind(f, 2, true) === nothing
        for Z in (A, X, T[1 0; 0 0], T[2 1; 1 3], T[Inf 0; 0 2])
          @test F(Z) === T(Inf)
        end
        @test F(zeros(T, 2, 2)) === zero(T)
        @test isnan(F(T[NaN 0; 0 2]))
        @test isnan(value_and_prox!(similar(A), F, T[NaN 0; 0 2], one(T)))
      end
      for f in (NormL1(T(Inf)), Scaled(NormL1(T(Inf)), T(2)))
        @test Spectral(f)(X) === T(Inf)
        @test Spectral(f)(zeros(T, 2, 2)) === zero(T)
        @test isnan(Spectral(f)(T[NaN 0; 0 2]))
      end
    end

    @testset "BigFloat levels ($T)" begin
      # A BigFloat level takes the recovery through frexp of a BigFloat, whose exponent is an Int32
      # on some platforms (#645): every exponent is converted before it enters a _ProjectionScale.
      @test positive_ok(SingularValue(NormL1(big(q)))(A), big(q) * 2big(M), T, 1)
      @test positive_ok(SingularValue(Conjugate(IndBallL1(big(q))))(A), big(q) * 2big(M), T, 1)
      @test positive_ok(Spectral(Constrained(NormL1(big(q)), IndBox(0, Inf)))(A), big(q) * 2big(M), T, 1)
      @test SingularValue(Conjugate(Scaled(NormL1(big(M)), big(4))))(A) === zero(T)
      @test SingularValue(Conjugate(Scaled(NormL2(big(M)), T(4))))(fill(M, 8, 8)) === T(Inf)
      Y = fill(T(5) / 8 * M, 2, 2)
      @test positive_ok(SingularValue(Conjugate(SqrNormL2(big(M))))(Y), (5big(M) / 4)^2 / (2big(M)), T, 2)
      @test positive_ok(SingularValue(SqrNormL2(big(l2)))(A), big(l2) / 2 * 4big(M)^2, T, 2)
      T === Float32 && @test positive_ok(SingularValue(CubeNormL2(big(2.0)^-300))(A), big(2.0)^-300 / 3 * (2big(M))^3, T, 3)
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
      # The kernel of a box with a linear constraint is not recovered (#648): at the overflowing
      # eigenvalue its value is NaN (l Σy with l = 0), at a finite matrix.
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
          # the fused value: E2, except IndBallRank(1), a counting function, which gives its value
          # at the all-NaN output (O2, #614)
          fused = F isa IndBallRank || isnan(v) ? T(NaN) : T(Inf)
          Y = similar(Z)
          @test isequal(value_and_prox!(Y, F, Z, one(T)), fused)
          # an aliased destination: the value is decided before it is written
          W = copy(Z)
          @test isequal(value_and_prox!(W, F, W, one(T)), fused)
        end
      end
      # E1: the recursive parameters of f come first, as at a finite input, also the inner
      # function of a LinearConstrained (whose value check is that of its set)
      F = SingularValue(Constrained(NormL1(one(T), ones(T, 3)), IndHalfspace(-one(T), zero(T))))
      for Z in (T[NaN 0; 0 2], T[Inf 0; 0 2], T[1 0; 0 2])
        @test_throws DimensionMismatch F(Z)
        Y = fill(T(7), 2, 2)
        @test_throws DimensionMismatch value_and_prox!(Y, F, Z, one(T))
        @test all(==(7), Y)
      end
      G = SingularValue(Constrained(NormL1(one(T), ones(T, 2)), IndHalfspace(-one(T), zero(T))))
      @test isnan(G(T[NaN 0; 0 2]))
      @test G(T[Inf 0; 0 2]) === T(Inf)
      # The structure check builds no temporaries: wrappers that check their argument through a
      # buffer take a placeholder of the transformed shape (0 bytes, as master at these inputs).
      P = Precompose(NormL1(), Matrix{T}(I, 2, 2), one(T))
      for f in (P, Precompose(NormL1(), T(2)), Translate(NormL1(q), zeros(T, 2)), Tilt(P, ones(T, 2)),
                Regularize(P, one(T)), MoreauEnvelope(P, one(T)), Scaled(Translate(NormL1(q), zeros(T, 2)), T(2)),
                Precompose(Translate(NormL1(), zeros(T, 2)), Matrix{T}(I, 2, 2), one(T)),
                Constrained(Translate(NormL1(q), zeros(T, 2)), IndHalfspace(-one(T), zero(T))))
        for Z in (T[NaN 0; 0 2], T[Inf 0; 0 2])
          F = SingularValue(f)
          @test isequal(F(Z), any(isnan, Z) ? T(NaN) : T(Inf))
          @test nonfinite_bytes(SingularValue(f), Z) == 0
        end
      end
      # and their structure still comes first
      for f in (Precompose(NormL1(), Matrix{T}(I, 3, 3), one(T)), Precompose(NormL1(one(T), ones(T, 3)), Matrix{T}(I, 2, 2), one(T)),
                Translate(NormL1(), zeros(T, 3)), Tilt(NormL1(), ones(T, 3)), Regularize(NormL1(), one(T), ones(T, 3)),
                MoreauEnvelope(Translate(NormL1(), zeros(T, 3)), one(T)))
        for Z in (T[NaN 0; 0 2], T[Inf 0; 0 2], T[1 0; 0 2])
          @test_throws DimensionMismatch SingularValue(f)(Z)
        end
      end
    end
  end
end

end
