module SpectralDomainTest

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using Random
using JLArrays
using GPUArraysCore: allowscalar

using ..TestUtils

const PB = ProximalBase

# the comparison IndBox made before its bounds could overflow
legacy_inbox(v, l, u, t) = (v >= l - t * max(one(t), abs(l))) & (v <= u + t * max(one(t), abs(u)))

@testset "non-finite entries are outside the domain" begin
  for T in (Float32, Float64)
    o = one(T)
    fs = (NuclearNorm(o), NuclearNorm(zero(T)), SingularValue(NormL1(o)), SingularValue(NormL2(o)),
          SingularValue(NormL0(o)), SingularValue(SqrNormL2(o)), SingularValue(Conjugate(NormL1(o))),
          Spectral(NormL1(o)), Spectral(SqrNormL2(o)), Spectral(NormL1Nonneg(o)), Spectral(NormL0(o)),
          Spectral(Conjugate(IndNonnegative())), Conjugate(SingularValue(NormLinf(o))), TracePSD(o))
    for X in (T[Inf 0; 0 1], T[-Inf 0; 0 1], T[NaN 0; 0 1], T[1 Inf; Inf 1], T[1 NaN; NaN 1])
      for f in fs
        @test f(X) === T(Inf)
      end
      @test Spectral(NormL1Nonneg(o))(X) === TracePSD(o)(X)
      @test SingularValue(NormL1(o))(X) === NuclearNorm(o)(X)
    end
    @test SingularValue(NormL0(o))(T[Inf 1; 0 1]) === T(Inf)        # not symmetric
    @test Spectral(Conjugate(IndNonnegative()))(fill(T(-1), 1, 1)) === zero(T)
    @test Spectral(Conjugate(IndNonnegative()))(fill(T(-Inf), 1, 1)) === T(Inf)
    @test SingularValue(Conjugate(NormL1(o)))(T[0.5 0; 0 0.5]) === zero(T)
    @test SingularValue(Conjugate(NormL1(o)))(T[2 0; 0 2]) === T(Inf)

    # the logical entries decide, not the storage behind them
    S = sparse(T[NaN 0; 0 1])
    @test NuclearNorm(o)(S) === T(Inf)
    @test Spectral(NormL1(o))(S) === T(Inf)
    P = fill(T(NaN), 3, 3)
    P[1:2, 1:2] .= T[2 0; 0 1]
    V = view(P, 1:2, 1:2)
    @test NuclearNorm(o)(V) == 3
    @test SingularValue(NormL0(o))(V) == 2
    @test Spectral(NormL1(o))(V) == 3
    B = T[2 1; NaN 1]
    @test Spectral(NormL1(o))(Symmetric(B)) ≈ 3
    @test Spectral(NormL1(o))(Symmetric(T[2 NaN; 1 1], :L)) ≈ 3
  end
end

@testset "conjugates that are indicators" begin
  marked = (Conjugate(NormL1(1.0)), Conjugate(NormL1(1.0, [1.0, 2.0])), Conjugate(NormL2(1.0)), Conjugate(NormLinf(1.0)),
            Conjugate(NuclearNorm(1.0)), Conjugate(GroupNormL2(1.0, [1:2, 3:4])), Conjugate(SLOPE([3.0, 2.0, 1.0, 0.5])),
            Conjugate(NormL1Nonneg(1.0)), Conjugate(TracePSD(1.0)), Conjugate(TotalVariation1D(1.0)),
            Conjugate(FusedLasso(1.0, 0.5)), Conjugate(NormL1Fused(1.0, 0.5)), Conjugate(CheckLoss(0.3)),
            Conjugate(ZeroFunction()), Conjugate(SumLargest(1)), Conjugate(IndNonnegative()), Conjugate(IndPSD()),
            Conjugate(IndIsotonic()), Conjugate(IndToeplitz()), Conjugate(IndSOC()),
            Conjugate(Scaled(NormL1(1.0), 2.0)), Conjugate(SingularValue(NormL1(1.0))), Conjugate(Spectral(NormL1(1.0))),
            Conjugate(SeparableSum(NormL1Nonneg(1.0), TotalVariation1D(1.0))), Conjugate(Conjugate(IndBox(0.0, 1.0))))
  for c in marked
    @test PB._isindicator(c)
  end
  # a linear function's conjugate is -d on its set; the others add a value
  for c in (Conjugate(LinearFunction([0.0], 2.0)), Conjugate(SqrNormL2()), Conjugate(IndBallL2(1.0)),
            Conjugate(Translate(NormL1(1.0), [1.0, 2.0])), Conjugate(SeparableSum(NormL1(1.0), SqrNormL2())),
            Conjugate(Conjugate(NormL1(1.0))))
    @test !PB._isindicator(c)
  end
  @test value_and_prox!(zeros(1), Conjugate(LinearFunction([0.0], 2.0)), [0.0], 1.0) === -2.0

  # Conjugate's own value_and_prox!: zero after the projection, also where the
  # Fenchel equality left a rounding error
  x = [3.0, -0.2, 1.0, 0.7]
  for c in (Conjugate(NormL1Nonneg(1.0)), Conjugate(TotalVariation1D(1.0)), Conjugate(FusedLasso(1.0, 0.5)),
            Conjugate(NormL1Fused(1.0, 0.5)), Conjugate(IndIsotonic()), Conjugate(SLOPE([3.0, 2.0, 1.0, 0.5])))
    y = zeros(4)
    @test value_and_prox!(y, c, x, 0.7) === 0.0
    @test y == prox(c, x, 0.7)
  end
  X = [3.0 0.1; 0.1 -0.2]
  for c in (Conjugate(NuclearNorm(1.0)), Conjugate(IndPSD()), Conjugate(TracePSD(1.0)), Conjugate(Spectral(NormL1(1.0))))
    Y = zeros(2, 2)
    @test value_and_prox!(Y, c, X, 0.7) === 0.0
    @test Y == prox(c, X, 0.7)
  end
  ct = Conjugate(SeparableSum(NormL1Nonneg(1.0), TotalVariation1D(1.0)))
  xt = ([3.0, -0.2], [3.0, -0.2, 1.0])
  yt = (zeros(2), zeros(3))
  @test value_and_prox!(yt, ct, xt, 0.7) === 0.0
  @test yt == prox(ct, xt, 0.7)
  y = zeros(4)
  @test value_and_prox!(y, Conjugate(NormL1Nonneg(1.0)), x, [0.5, 1.0, 2.0, 0.7]) === 0.0
  @test y == prox(Conjugate(NormL1Nonneg(1.0)), x, [0.5, 1.0, 2.0, 0.7])
  # the conjugate value itself stays as it was
  @test_throws ArgumentError Conjugate(TotalVariation1D(1.0))(zeros(4))
end

@testset "users of the indicator property" begin
  x = [3.0, -0.2, 1.0, 0.7]
  c = Conjugate(TotalVariation1D(1.0))
  # the generic value_and_prox!
  y = zeros(4)
  @test invoke(value_and_prox!, Tuple{Any, PB.AbstractFunction, Any, Real}, y, c, x, 0.7) === 0.0
  @test y == prox(c, x, 0.7)
  # the constrained wrappers that take a conjugate
  for g in (Constrained(Conjugate(NormL1Nonneg(1.0)), IndBox(-0.5, 0.5)),
            Constrained(Conjugate(NormL1Nonneg(1.0)), IndHyperplane(1.0, 0.5)))
    @test PB._isindicator(g)
    y = zeros(4)
    @test value_and_prox!(y, g, x, 0.7) === 0.0
    @test y ≈ prox(g, x, 0.7) rtol = 1e-12
  end
  # wrappers that read the property of their inner function
  for g in (PointwiseMinimum(c, Conjugate(NormL1Nonneg(1.0))), Precompose(c, 2.0))
    @test value_and_prox!(zeros(4), g, x, 0.7) === 0.0
  end
  @test value_and_prox!(zeros(2, 2), Spectral(Conjugate(NormL1Nonneg(1.0))), [3.0 0.1; 0.1 -0.2], 0.7) === 0.0
  # IndIntersection now takes these sets: projections work, and values exist where Conjugate(f)'s does
  C = IndIntersection(Conjugate(NormL1(1.0)), IndBallL2(1.0))
  p = prox(C, [3.0, 0.5])
  @test p ≈ [3.0, 0.5] / norm([3.0, 0.5]) rtol = 1e-8
  @test C([0.5, 0.5]) === 0.0
  @test C([2.0, 0.0]) === Inf
  @test value_and_prox!(zeros(2), C, [3.0, 0.5], 1.0) === 0.0
  @test_throws ArgumentError IndIntersection(Conjugate(TotalVariation1D(1.0)), IndBallL2(1.0))([0.5, 0.5, 0.0, 0.0])
  # a zero scale keeps an indicator
  @test Scaled(Conjugate(NormL1(1.0)), 0.0) isa Conjugate{<:NormL1}
  @test_throws ArgumentError Scaled(Conjugate(LinearFunction([1.0])), 0.0)
  # distances to the set
  @test SqrDistL2(Conjugate(NormL1(1.0)))([3.0, -0.2]) ≈ 2
  @test DistL2(Conjugate(NormL1(1.0)))([3.0, -0.2]) ≈ 2
end

@testset "box bounds that do not overflow" begin
  for T in (Float32, Float64)
    M = floatmax(T)
    @test IndBox(-M, M)(T[Inf]) === T(Inf)
    @test IndBox(-M, M)(T[-Inf]) === T(Inf)
    @test IndBox(-M, M)(T[M, -M]) === zero(T)
    @test IndBox(-M, M)(T[NaN]) === T(Inf)
    @test IndBallLinf(M)(T[Inf]) === T(Inf)
    @test IndBox(zero(T), M)(T[M, 0]) === zero(T)
    @test IndBox(zero(T), M)(T[Inf]) === T(Inf)
    @test IndBox(-M, M)(T[prevfloat(M)]) === zero(T)
    # infinite bounds keep the separable convention
    @test IndBox(-T(Inf), T(Inf))(T[Inf, -Inf]) === zero(T)
    @test IndNonnegative()(T[Inf]) === zero(T)
    # the entry oracle and an array of bounds
    @test PB.value_entry(IndBox(-M, M), T(Inf), 1) === T(Inf)
    @test IndBox(T[-M, 0], T[M, 1])(T[Inf, 0]) === T(Inf)
    @test IndBox(T[-M, 0], T[M, 1])(T[M, 1]) === zero(T)
  end

  # ordinary inputs give the legacy comparison bit for bit
  rng = MersenneTwister(196)
  for T in (Float32, Float64)
    t = PB._tol(T)
    x = T[randn(rng, T, 200) .* 3; T[Inf, -Inf, NaN, 1, -1, 2, 0]]
    l = randn(rng, T, 207) .- 1
    u = l .+ abs.(randn(rng, T, 207)) .* 2
    for (lb, ub) in ((-one(T), one(T)), (l, u), (zero(T), abs.(u)), (l, T(Inf)), (-T(Inf), T(Inf)), (-1, 2), (Int32(0), T(0.5)))
      f = IndBox(lb, ub)
      li(i) = lb isa AbstractArray ? lb[i] : lb
      ui(i) = ub isa AbstractArray ? ub[i] : ub
      member(i) = legacy_inbox(x[i], li(i), ui(i), t)
      for i in eachindex(x)
        want = member(i) ? zero(T) : T(Inf)
        @test PB.value_entry(f, x[i], i) === want
        @test IndBox(lb isa AbstractArray ? lb[i:i] : lb, ub isa AbstractArray ? ub[i:i] : ub)(x[i:i]) === want
      end
      # the vectorized loop over a whole array: the members, then one entry outside
      xm = [member(i) ? x[i] : T(li(i)) for i in eachindex(x)]
      @test f(xm) === zero(T)
      xm[150] = T(NaN)
      @test f(xm) === T(Inf)
    end
  end

  # the GPU path uses the same test
  allowscalar(false)
  for T in (Float32, Float64)
    M = floatmax(T)
    @test IndBox(-M, M)(JLArray(T[Inf])) === T(Inf)
    @test IndBox(-M, M)(JLArray(T[M, -M])) === zero(T)
    @test IndBox(-one(T), one(T))(JLArray(T[0.5, -1])) === zero(T)
    @test IndBox(-one(T), one(T))(JLArray(T[0.5, -2])) === T(Inf)
    @test IndBox(-T(Inf), T(Inf))(JLArray(T[Inf])) === zero(T)
  end
end

@testset "balls and boxes at an overflowing spectrum" begin
  for T in (Float32, Float64)
    M = floatmax(T)
    A = T[M M; M -M]                      # singular values √2 M, eigenvalues ±√2 M
    P = T[M M; M M]                       # eigenvalues 0 and 2M
    for X in (A, -A)
      @test SingularValue(IndBallLinf(M))(X) === T(Inf)
      @test SingularValue(IndBallL1(M))(X) === T(Inf)
      @test SingularValue(IndBallL2(M))(X) === T(Inf)
      @test SingularValue(IndBox(zero(T), M))(X) === T(Inf)
      @test Spectral(IndBox(-M, M))(X) === T(Inf)
      @test Spectral(IndBallLinf(M))(X) === T(Inf)
      @test SingularValue(IndBallL1(T(Inf)))(X) === zero(T)
      @test SingularValue(IndBallL2(T(Inf)))(X) === zero(T)
      @test SingularValue(IndBallLinf(T(Inf)))(X) === zero(T)
      @test Spectral(IndBox(-T(Inf), T(Inf)))(X) === zero(T)
      @test Spectral(IndBallL2(T(Inf)))(X) === zero(T)
    end
    @test Spectral(IndBox(zero(T), T(Inf)))(P) === zero(T)
    @test Spectral(IndBox(zero(T), T(Inf)))(-P) === T(Inf)
    @test Spectral(IndBox(-T(Inf), zero(T)))(-P) === zero(T)
    @test Spectral(IndBox(-T(Inf), zero(T)))(P) === T(Inf)
    # an isolated eigenvalue beside a huge positive block: -1 lies within the
    # tolerance √ε·2M, relative to the spectrum, as for IndPSD; -4√ε·M does not
    Q = T[M M 0; M M 0; 0 0 -1]
    @test IndPSD()(Q) === zero(T)
    @test Spectral(IndBox(zero(T), T(Inf)))(Q) === zero(T)
    @test Spectral(IndNonnegative())(Q) === zero(T)
    @test Spectral(IndBox(-one(T), T(Inf)))(Q) === zero(T)
    δ = 4 * sqrt(eps(T)) * M
    Q = T[M M 0; M M 0; 0 0 -δ]
    @test IndPSD()(Q) === T(Inf)
    @test Spectral(IndBox(zero(T), T(Inf)))(Q) === T(Inf)
    @test Spectral(IndNonnegative())(Q) === T(Inf)
    @test Spectral(IndBox(-δ, T(Inf)))(Q) === zero(T)
    # members at the largest finite spectrum
    @test SingularValue(IndBallLinf(M))(T[M 0; 0 M]) === zero(T)
    @test SingularValue(IndBallL2(M))(T[M 0; 0 0]) === zero(T)
    @test Spectral(IndBox(-M, M))(T[M 0; 0 -M]) === zero(T)
    # bounds of a wider type are compared in that type
    if T === Float32
      @test SingularValue(IndBallLinf(floatmax(Float64)))(A) === 0f0
      @test Spectral(IndBox(-floatmax(Float64), floatmax(Float64)))(A) === 0f0
      @test SingularValue(IndBallL1(4.0 * floatmax(Float32)))(A) === 0f0
      @test SingularValue(IndBallL1(2.0 * floatmax(Float32)))(A) === Inf32
    end
    # array bounds are checked against the spectrum, as at a finite one
    @test_throws DimensionMismatch SingularValue(IndBox(T[0], T[Inf]))(A)
    @test_throws DimensionMismatch SingularValue(IndBox(zeros(T, 5), fill(T(Inf), 5)))(A)
    @test SingularValue(IndBox(zeros(T, 2), fill(M, 2)))(A) === T(Inf)
    @test SingularValue(IndBox(zeros(T, 2), fill(T(Inf), 2)))(A) === zero(T)
    # empty inputs
    @test SingularValue(IndBallL1(one(T)))(zeros(T, 0, 0)) === zero(T)
    @test Spectral(IndBox(zero(T), one(T)))(zeros(T, 0, 0)) === zero(T)
  end
end

@testset "the zero mask in the subnormal range" begin
  for T in (Float32, Float64)
    o, u = one(T), nextfloat(zero(T))
    X = T[2048u 0; 0 u]
    @test SingularValue(NormL0(o))(X) === T(2)
    @test SingularValue(IndBallL0(1))(X) === T(Inf)
    @test SingularValue(IndBallL0(2))(X) === zero(T)
    @test SingularValue(NormL0(o))(T[2 0; 0 1] .* u) === T(2)
    # the mask is still relative: a singular value below √ε σ₁ counts as zero
    v = ldexp(o, exponent(floatmin(T)) + 2)
    @test SingularValue(NormL0(o))(T[v 0; 0 0]) === o
    @test SingularValue(NormL0(o))(T[v 0; 0 v * PB._tol(T) / 4]) === o
  end
  # the threshold rounds onto floatmin: 2^-1022 lies above √ε prevfloat(2^-996)
  s1, s2 = prevfloat(2.0^-996), 2.0^-1022
  @test s2 > big(PB._tol(Float64)) * big(s1)
  @test SingularValue(NormL0(1.0))([s1 0; 0 s2]) === 2.0
  @test SingularValue(IndBallL0(1))([s1 0; 0 s2]) === Inf
  # balls and boxes decide membership in the original units there
  X = [2.0^-1030 0; 0 2.0^-1031]
  @test SingularValue(IndBox(1e-3, 1.0))(X) === Inf
  @test SingularValue(IndBox(-1.0, -1e-3))(X) === Inf
  @test SingularValue(IndBox(0.0, 1.0))(X) === 0.0
  @test SingularValue(IndBallLinf(1e-300))(X) === 0.0
  @test SingularValue(IndBox(5f-4, 1f0))(Float32[2f0^-138 0; 0 2f0^-139]) === Inf32
  @test SingularValue(IndBox(0f0, 1f0))(Float32[2f0^-138 0; 0 2f0^-139]) === 0f0
end

@testset "the conjugate value of SingularValue applies the mask" begin
  for T in (Float32, Float64)
    τ = PB._tol(T)
    X = Matrix(Diagonal(T[1, τ / 2, τ / 2, τ / 2, τ / 2]))
    @test SingularValue(NormL1(one(T)))(X) === one(T)
    @test Conjugate(SingularValue(NormLinf(one(T))))(X) === zero(T)
    @test Conjugate(SingularValue(NormLinf(one(T))))(2X) === T(Inf)
    @test Conjugate(SingularValue(IndBallL1(one(T))))(X) === one(T)
  end
end

@testset "allocations on the ordinary and exceptional paths" begin
  M = floatmax(Float64)
  u = nextfloat(0.0)
  Xo = [2.0 0.5; 0.5 1.0]
  Xn = [NaN 0; 0 1.0]
  Xa = [M M; M -M]
  Xs = [2048u 0; 0 u]
  Xc = Matrix(Diagonal([1, PB._tol(Float64) / 2]))
  F = SingularValue(IndBallLinf(M))
  G = SingularValue(NormL0(1.0))
  H = Conjugate(SingularValue(NormLinf(1.0)))
  S = Spectral(IndBox(-M, M))
  N = NuclearNorm(1.0)
  ref = (F(Xo), G(Xo), H(Xo / 4), S(Xo), N(Xo))
  for X in (Xo, Xn, Xa, Xs, Xc)
    F(X); G(X); H(X); S(X); N(X)
    @test allocs(F, X) == 0
    @test allocs(G, X) == 0
    @test allocs(H, X) == 0
    @test allocs(S, X) == 0
    @test allocs(N, X) == 0
  end
  @test (F(Xo), G(Xo), H(Xo / 4), S(Xo), N(Xo)) === ref
  c = Conjugate(TotalVariation1D(1.0))
  x = [3.0, -0.2, 1.0, 0.7]
  y = zeros(4)
  value_and_prox!(y, c, x, 0.7)
  @test allocs(value_and_prox!, y, c, x, 0.7) == 0
  b = IndBox(-M, M)
  z = [Inf, 1.0]
  b(z)
  @test allocs(b, z) == 0
end

end
