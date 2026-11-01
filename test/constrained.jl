module ConstrainedTest

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using Random

using ..TestUtils

# The small closed-form cases, coordinate steps and errors are in constrained_basics.jl.

Random.seed!(1)

@testset "a separable function on a box" begin
  n = 8
  x = 2 .* randn(n)
  w = rand(n)
  l, u = -rand(n), rand(n)
  fs = (NormL1(0.3, w), ElasticNet(0.2, 0.5), NormL1Nonneg(0.2), SqrNormL2(2.),
        ZeroFunction(), LinearFunction(randn(n)), Scaled(NormL1(1.), 0.4),
        IndBox(-0.3, 0.6))
  for f in fs, C in (IndBox(-0.5, 1.), IndBox(l, u), IndNonnegative(), IndBallLinf(0.7))
    @testset "$(nameof(typeof(f))) on $(nameof(typeof(C)))" begin
      c = Constrained(f, C)
      check_traits(c; proximable = true, separable = true, convex = true)
      for γ in (0.4, 1.7)
        p = check_prox(c, x, γ)
        @test p ≈ clamp.(prox(f, x, γ), c.box.l, c.box.u)
        @test all(i -> prox_entry(c, x[i], i, γ) ≈ p[i], 1:n)
      end
    end
  end
end

@testset "zero-preserving wrappers on a box" begin
  fs = (Regularize(NormL1(1.), 1.),
        Precompose(NormL1(1.), -2.), Conjugate(NormL1(1.)), ZeroFunction(), SqrNormL2())
  for f in fs, xs in (SparseIterate([0., 3., 0., -2., 0.]), SparseIterate(5))
    c = Constrained(f, IndBox(-1., 1.))
    @test iszeropreserving(c)
    x = Vector(xs)
    @test c(xs) == c(x)
    for step in (0.7, fill(0.7, 5))
      expected = prox(c, x, step)
      out = SparseIterate([2., 0., 0., 0., 0.])
      @test prox!(out, c, xs, step) === out
      @test Vector(out) ≈ expected
      @test value_and_prox!(out, c, xs, step) ≈ c(expected)
      @test Vector(out) ≈ expected
      @test c(out) ≈ c(expected)
      @test isactive(out, 1) && out[1] == 0
      alias = copy(xs)
      @test prox!(alias, c, alias, step) === alias
      @test Vector(alias) ≈ expected
      alias = copy(xs)
      @test value_and_prox!(alias, c, alias, step) ≈ c(expected)
      @test Vector(alias) ≈ expected
      @test nnz(alias) == nnz(xs)
    end
  end
end

@testset "a separable function on a hyperplane or a halfspace" begin
  n = 10
  x = randn(n)
  @testset "general normals" begin
    a = randn(n)
    a[3] = 0.
    for f in (ElasticNet(0.1, 0.3), NormL1Nonneg(0.1, rand(n)))
      for (H, b) in ((IndHyperplane, 0.3), (IndHalfspace, -0.4))
        c = Constrained(f, H(a, b))
        z = check_prox(c, 2 .* x, 0.9)
        H === IndHyperplane ? (@test dot(a, z) ≈ b) : (@test dot(a, z) <= b + 1e-12)
        @test z[3] ≈ prox_entry(f, 2x[3], 3, 0.9)           # aᵢ = 0: independent of ν
      end
    end
  end

  @testset "Float32" begin
    x32 = Float32.(x)
    for c in (Constrained(IndBox(0, 1), IndHyperplane(1, 3)), Constrained(ElasticNet(0.1, 0.2), IndSimplex(2.)))
      z = prox(c, x32, 0.7f0)
      @test eltype(z) == Float32
      @test z ≈ prox(c, x, 0.7) atol = 1e-5
      @test isfinite(c(z))
    end
  end
end

# a separable function without known slopes, for the secant path of the search
struct NoSlope{F} <: SeparableFunction
  f::F
end
ProximalBase.entry_parameters(f::NoSlope) = ProximalBase.entry_parameters(f.f)
ProximalBase.prox_entry(g::NoSlope, v, i, γ) = prox_entry(g.f, v, i, γ)
ProximalBase.value_entry(g::NoSlope, v, i) = value_entry(g.f, v, i)
ProximalBase.isconvex(::Type{<:NoSlope}) = true

# and one that counts the passes of the search over x, and the entries whose
# slopes it computes
const CALLS = Ref(0)
const SLOPES = Ref(0)
struct Counted{F} <: SeparableFunction
  f::F
end
ProximalBase.entry_parameters(f::Counted) = ProximalBase.entry_parameters(f.f)
ProximalBase.prox_entry(g::Counted, v, i, γ) = (CALLS[] += 1; prox_entry(g.f, v, i, γ))
ProximalBase.value_entry(g::Counted, v, i) = value_entry(g.f, v, i)
ProximalBase.isconvex(::Type{<:Counted}) = true
ProximalBase._has_prox_slope(::Type{<:Counted{F}}) where {F} = ProximalBase._has_prox_slope(F)
ProximalBase._prox_entry_slope(g::Counted, v, i, γ) =
  (CALLS[] += 1; SLOPES[] += 1; ProximalBase._prox_entry_slope(g.f, v, i, γ))

@testset "the search starts from the last multiplier" begin
  n = 2000
  x = randn(n)
  noise = randn(n)
  for (f, H) in ((NormL1(0.5), IndHyperplane(1, 0)), (IndBox(0, 1), IndHyperplane(1, 10)),
                 (IndNonnegative(), IndHyperplane(1, n / 10)), (Constrained(NormL1(0.3), IndBox(-1, 1)), IndHyperplane(randn(n), 3.)),
                 (Constrained(ElasticNet(0.1, 1.), IndNonnegative()), IndHalfspace(1, 1.)), (NoSlope(NormL1(0.5)), IndHyperplane(1, 0)),
                 (NoSlope(IndBox(0, 1)), IndHyperplane(1, 10)))
    c = Constrained(f, H)
    for t in 1:10
      xt = x .+ (0.01 * t) .* noise
      @test prox(c, xt, 0.7) ≈ prox(Constrained(f, H), xt, 0.7) atol = 1e-10    # a fresh one starts from zero
    end
    xs = 5 .* randn(n)                                                          # a stale start
    @test prox(c, xs, 0.7) ≈ prox(Constrained(f, H), xs, 0.7) atol = 1e-10
  end

  # a halfspace that becomes inactive, and active again
  c = Constrained(NormL1(0.1), IndHalfspace(1, 1.))
  xa = abs.(randn(n)) .+ 0.5
  za = prox(c, xa, 1.)
  @test sum(za) ≈ 1 && c.last[] > 0
  xi = 0.01 .* randn(n)
  @test prox(c, xi, 1.) == prox(NormL1(0.1), xi, 1.)
  @test c.last[] == 0
  @test prox(c, xa, 1.) ≈ za

  # the passes of the search: many from zero, a few from the last multiplier
  y = similar(x)
  for (f, H) in ((IndNonnegative(), IndHyperplane(1, 1)), (NormL1(0.5), IndHyperplane(1, 0)), (IndBox(0, 1), IndHyperplane(1, n / 2)))
    c = Constrained(Counted(f), H)
    CALLS[] = SLOPES[] = 0
    prox!(y, c, x, 0.7)
    cold = CALLS[] / n
    @test SLOPES[] == 0                               # a first call computes no slopes
    CALLS[] = SLOPES[] = 0
    for t in 1:10
      prox!(y, c, x .+ (0.001 * t) .* noise, 0.7)
    end
    warm = CALLS[] / n / 10
    @test warm <= 5 && warm < cold
    @test SLOPES[] == 10n                             # the later ones in their first pass
  end

  # Float32 calls keep the multiplier too
  c = Constrained(NormL1(0.5f0), IndHyperplane(1, 0))
  x32 = Float32.(x)
  @test prox(c, x32, 0.7f0) ≈ prox(c, x32 .+ 0.0f0, 0.7f0)
  @test eltype(prox(c, x32, 0.7f0)) == Float32
end

# and one that also keeps the ranges of its scalar proximal operators, so that
# the search fixes entries; Counted hides them, and searches as before
struct Ranged{F} <: SeparableFunction
  f::F
end
ProximalBase.entry_parameters(f::Ranged) = ProximalBase.entry_parameters(f.f)
ProximalBase.prox_entry(g::Ranged, v, i, γ) = (CALLS[] += 1; prox_entry(g.f, v, i, γ))
ProximalBase.value_entry(g::Ranged, v, i) = value_entry(g.f, v, i)
ProximalBase.isconvex(::Type{<:Ranged}) = true
ProximalBase._has_prox_slope(::Type{<:Ranged{F}}) where {F} = ProximalBase._has_prox_slope(F)
ProximalBase._prox_entry_slope(g::Ranged, v, i, γ) =
  (CALLS[] += 1; SLOPES[] += 1; ProximalBase._prox_entry_slope(g.f, v, i, γ))
ProximalBase._has_prox_range(::Type{<:Ranged{F}}) where {F} = ProximalBase._has_prox_range(F)
ProximalBase._prox_range(g::Ranged, i::Integer, ::Type{S}) where {S} = ProximalBase._prox_range(g.f, i, S)
ProximalBase._has_uniform_range(::Type{<:Ranged{F}}) where {F} = ProximalBase._has_uniform_range(F)

@testset "variable fixing" begin
  # against the search without it, from zero, on problems whose root lies in
  # the tail of the entries, where it fixes most of them
  n = 20_000
  a = randn(n)
  a[1:7:end] .= 0                                   # a normal with zeros and both signs
  b = 0.3 .* randn(n)
  w, l, u = rand(n), -rand(n), rand(n)
  γv = 0.2 .+ rand(n)
  x = randn(n)
  cases = [(IndNonnegative(), IndHyperplane(1, 1), true),                           # the simplex
           (IndBox(0, 1), IndHyperplane(1, 10), true),                              # the capped simplex
           (Constrained(ElasticNet(0.1, 1.), IndNonnegative()), IndHyperplane(1, 1), true),
           (Constrained(ElasticNet(0.1, 1.), IndNonnegative()), IndHalfspace(1, 1), true),
           (NormL1Nonneg(0.1, w), IndHyperplane(abs.(a) .+ 0.1, 0.5), true),         # weights, a normal per entry
           (NormL1Nonneg(0.1, w), IndHyperplane(a, 0.3), false),                    # zeros and negative entries
           (IndBox(l, u), IndHyperplane(1, sum(l) + 0.5), true),                    # bounds per entry
           (Translate(IndBox(0, 1), b), IndHyperplane(1, sum(b) + 3), true),
           (Scaled(NormL1Nonneg(0.2), 2.), IndHyperplane(1, 2), true),
           (Tilt(IndBox(0, 1), b), IndHyperplane(1, 5), true),
           (Constrained(NormL1(0.3), IndBox(-1, 1)), IndHyperplane(-1, n - 5), true),  # at the top of [-1, 1]
           (IndNonnegative(), IndHyperplane(1, n / 10), false),                     # the root among the entries
           (IndBox(0, 1), IndHyperplane(1, n / 2), false)]
  y, yr = similar(x), similar(x)
  for (f, H, tail) in cases, γ in (0.7, γv)
    c, r = Constrained(Ranged(f), H), Constrained(Counted(f), H)
    CALLS[] = SLOPES[] = 0
    prox!(y, c, x, γ)
    fixed = CALLS[]
    @test SLOPES[] == 0
    CALLS[] = 0
    prox!(yr, r, x, γ)
    @test isapprox(y, yr; rtol = 1e-10)
    H isa IndHyperplane && @test abs(dot(H.a isa Real ? fill(H.a, n) : H.a, y) - H.b) <= 1e-10 * max(1, abs(H.b))
    tail && @test fixed < CALLS[] / 2
    # the calls that follow start from the last multiplier, as before
    xt = x .+ 0.01 .* randn(n)
    @test isapprox(prox!(y, c, xt, γ), prox!(yr, r, xt, γ); rtol = 1e-10)
  end

  # Float32, a matrix, and a small input, where it does not fix
  for (f, H) in ((IndNonnegative(), IndHyperplane(1, 1)), (IndBox(0, 1), IndHyperplane(1, 10)))
    x32 = Float32.(x)
    z = prox(Constrained(f, H), x32, 0.7f0)
    @test eltype(z) == Float32
    @test z ≈ prox(Constrained(f, H), x, 0.7) atol = 1e-5
    @test abs(sum(z) - H.b) <= 1e-4
    X = reshape(x, 200, 100)
    @test prox(Constrained(Ranged(f), H), X, 0.7) ≈ prox(Constrained(Counted(f), H), X, 0.7) rtol = 1e-10
    @test prox(Constrained(Ranged(f), H), x[1:100], 0.7) ≈ prox(Constrained(Counted(f), H), x[1:100], 0.7) rtol = 1e-10
  end

  # a call from zero allocates nothing once the buffers exist, and is inferred
  cold!(y, c, x, γ) = (c.last[] = NaN; prox!(y, c, x, γ))
  for c in (Constrained(IndNonnegative(), IndHyperplane(1, 1)), Constrained(NormL1Nonneg(0.1, w), IndHyperplane(abs.(a) .+ 0.1, 0.5)))
    @test allocs(cold!, y, c, x, 0.7) == 0
    @test allocs(cold!, y, c, x, γv) == 0
    c.last[] = NaN
    @inferred prox!(y, c, x, 0.7)
  end

  # the ranges of the scalar proximal operators, reached by prox_entry
  for f in (IndNonnegative(), NormL1Nonneg(0.2), NormL1Nonneg(0.2, w), IndBox(-0.5, 1), IndBox(l, u), Scaled(IndBox(0, 1), 2.),
            Tilt(IndNonnegative(), b), Translate(IndBox(0, 1), b), Constrained(NormL1(0.3), IndBox(-1, 1)),
            Constrained(NormL1Nonneg(0.3), IndBox(-1, 1)), Constrained(ElasticNet(0.1, 1.), IndNonnegative()))
    @test ProximalBase._has_prox_range(typeof(f))
    for i in (1, 2, 3)
      lo, hi = ProximalBase._prox_range(f, i, Float64)
      @test isfinite(lo) ? lo == prox_entry(f, -1e300, i, 0.7) : lo == -Inf
      @test isfinite(hi) ? hi == prox_entry(f, 1e300, i, 0.7) : hi == Inf
    end
  end
  @test ProximalBase._prox_range(NormL1(0.3), 1, Float32) === (-Inf32, Inf32)
  for f in (NormL1(0.3), ElasticNet(0.1, 1.), SqrNormL2(1.), ZeroFunction(), LinearFunction(b), MCP(1.))
    @test !ProximalBase._has_prox_range(typeof(f))
  end
  for (f, uniform) in ((IndNonnegative(), true), (NormL1Nonneg(0.2, w), true), (IndBox(-0.5, 1), true), (IndBox(l, u), false),
                       (Scaled(IndBox(0, 1), 2.), true), (Tilt(IndNonnegative(), b), true), (Translate(IndBox(0, 1), b), false),
                       (Constrained(NormL1(0.3), IndBox(-1, 1)), true), (Constrained(ElasticNet(0.1, 1.), IndBox(l, u)), false),
                       (Constrained(SqrNormL2(1.), IndNonnegative()), true), (Constrained(ZeroFunction(), IndNonnegative()), true),
                       (Constrained(LinearFunction(b), IndNonnegative()), true), (MCP(1.), false))
    @test ProximalBase._has_uniform_range(typeof(f)) == uniform
  end
end

a_value(f, x) = f(x)

@testset "the conjugate of a box with a linear constraint" begin
  n = 7
  y = randn(n)
  function top(v, κ)
    s = sort(v; rev = true)
    k = floor(Int, κ)
    sum(s[1:k]) + (k < length(v) ? (κ - k) * s[k + 1] : 0.0)
  end
  # on the capped simplex, the sum of the k largest entries, also for a fractional k
  for k in (0, 1, 3, 2.5, 7)
    @test Conjugate(Constrained(IndBox(0, 1), IndHyperplane(1, k)))(y) ≈ top(y, k) atol = 1e-14
    @test Conjugate(Constrained(IndBox(0, 1), IndHalfspace(1, k)))(y) ≈ top(max.(y, 0), k) atol = 1e-14
  end
  @test Conjugate(Constrained(IndBox(0, 1), IndHyperplane(1, 3)))(reshape(1:12, 3, 4)) ≈ 33
  # Fenchel–Young with equality at the proximal point, for other boxes, normals
  # and right-hand sides, and a lower bound on the sum
  for C in (Constrained(IndBox(-0.5, 2), IndHyperplane(2, 3)), Constrained(IndBox(-1, 1), IndHalfspace(-1.5, 2)),
            Constrained(IndBox(-1, 1), IndHalfspace(0.5, -1)), Constrained(IndBox(0.3, 0.3), IndHalfspace(1, 4)),
            Constrained(IndBox(-1, 1), IndHalfspace(1, 20)))
    S = Conjugate(C)
    x = 3 .* randn(n)
    for γ in (0.3, 1.)
      z = prox(S, x, γ)
      p = (x .- z) ./ γ                              # the projection of x/γ onto the set
      @test C(p) == 0
      @test S(z) ≈ dot(z, p) atol = 1e-10
    end
    check_prox(S, x, 0.7)
  end
  S = Conjugate(Constrained(IndBox(0, 1), IndHyperplane(1, 2.5)))
  @test S(Float32.(y)) isa Float32
  @test S(Float32.(y)) ≈ top(y, 2.5) rtol = 1e-5
  @test allocs(a_value, S, y) == 0
  @test_throws ArgumentError Conjugate(Constrained(IndBox(0, 1), IndHyperplane(1, 9)))(y)        # an empty set
  @test_throws ArgumentError Conjugate(Constrained(IndBox(0, 1), IndHalfspace(1, -1)))(y)
  @test_throws ArgumentError Conjugate(Constrained(IndBox(0, Inf), IndHyperplane(1, 1)))(y)      # unbounded
  @test_throws ArgumentError Conjugate(Constrained(IndBox(zeros(n), ones(n)), IndHyperplane(1, 2)))(y)
end

@testset "slopes of scalar proximal operators" begin
  n = 5
  for f in (ZeroFunction(), LinearFunction(randn(n)), SqrNormL2(2.), SqrNormL2(1., randn(n)), NormL1(0.3),
            NormL1(0.3, [0.5, 1., 1.5, 2., 0.2]), NormL1Nonneg(0.2), ElasticNet(0.1, 0.5), IndBox(-0.5, 1), IndNonnegative(),
            0.5 * NormL1(1.), Tilt(NormL1(0.3), [0.1, -0.3, 0.2, 0., 0.4]), Translate(NormL1(0.3), [0.1, -0.2, 0.3, 0., 0.5]),
            Constrained(NormL1(0.2), IndBox(-1, 1)))
    @test ProximalBase._has_prox_slope(typeof(f))
    for i in 1:n, v in (-2.3, -0.45, 0.05, 0.7, 1.9)          # away from the kinks
      z, dz = ProximalBase._prox_entry_slope(f, v, i, 0.8)
      @test z == prox_entry(f, v, i, 0.8)
      h = 1e-7
      @test dz ≈ (prox_entry(f, v + h, i, 0.8) - prox_entry(f, v - h, i, 0.8)) / 2h atol = 1e-6
    end
  end
  @test !ProximalBase._has_prox_slope(typeof(MCP(1.)))
  @test !ProximalBase._has_prox_slope(typeof(NoSlope(NormL1(1.))))
end

@testset "boxes intersecting closed and open scalar domains" begin
  for T in (Float32,)
    # Entropy has a closed domain at zero; the positive log barrier can instead
    # be constrained to an interval contained in its open domain.
    @test prox(Constrained(NegEntropy(one(T)), IndBox(zero(T), zero(T))), T[1]) == T[0]
    c = Constrained(NegLog(one(T)), IndBox(one(T), T(2)))
    x, out = T[-3, 0, 5], zeros(T, 3)
    expected = clamp.((x .+ sqrt.(x.^2 .+ 4)) ./ 2, one(T), T(2))
    @test prox(c, x) ≈ expected
    @test value_and_prox!(out, c, x, one(T)) ≈ -sum(log, expected)
    @test out ≈ expected
    @test all(>(zero(T)), out)
  end
end

@testset "boxes with mixed-precision touching endpoints" begin
  for T in (Float32, Float64)
    # Each bound rounds outward in T. The original domains touch at exactly
    # b, which the proximal output represents by T(b).
    b = T === Float32 ? 0.3 : big"0.1"
    for f in (IndBox(zero(b), b), Conjugate(NormL1(b)))
      c = Constrained(f, IndBox(b, b))
      @test prox(c, T[0]) == T[b]
      @test prox_entry(c, zero(T), 1, one(T)) == T(b)
      x = SparseIterate(T[0])
      prox_coordinate!(x, c, 1, one(T))
      @test Vector(x) == T[b]
    end
    c = Constrained(IndBox(zero(b), b), IndBox(b, b))
    @test ProximalBase._prox_entry_slope(c, zero(T), 1, one(T)) == (T(b), zero(T))
  end
end

@testset "variable fixing with extreme normals" begin
  rng = Xoshiro(140)
  n = 20_000
  for T in (Float32, Float64)
    e = round(Int, 0.6 * exponent(floatmax(T)))
    x = randn(rng, T, n)
    a = randn(rng, T, n)
    a[1:7:end] .= zero(T)
    positive = abs.(a) .+ T(0.5)
    γv = T(0.2) .+ rand(rng, T, n)
    cases = ((IndNonnegative(), one(T), one(T)),
             (IndBox(zero(T), one(T)), one(T), T(10)),
             (Constrained(NormL1(T(0.3)), IndBox(-one(T), one(T))), -one(T), -T(n - 5)),
             (NormL1Nonneg(T(0.1)), positive, T(0.5)),
             (IndBox(zero(T), one(T)), a, sum(min.(a, zero(T))) + one(T)))
    rtol, atol = T === Float64 ? (1e-10, 1e-12) : (1e-4, 1e-6)
    tol = T === Float64 ? 1e-10 : 16Float64(eps(T))
    for (f, normal, b) in cases, F in (IndHyperplane, IndHalfspace), γ in (T(0.7), γv)
      H = F(normal, b)
      r = Constrained(Ranged(f), H)
      CALLS[] = SLOPES[] = 0
      ref = prox(r, x, γ)
      unfixed = T === Float64 ? prox(Constrained(Counted(f), H), x, γ) : nothing
      for c in (ldexp(one(T), -e), ldexp(one(T), e))
        ac, bc = c .* normal, c * b
        @test ac ./ c == normal
        @test bc / c == b
        Hc = F(ac, bc)
        g = Constrained(Ranged(f), Hc)
        CALLS[] = SLOPES[] = 0
        y = prox(g, x, γ)                            # from a fresh multiplier
        @test SLOPES[] == 0
        @test g.live[6].buf[].compact
        @test g.live[6].buf[].n < n
        @test isapprox(y, ref; rtol, atol = atol * max(1, norm(x)))
        T === Float64 && @test isapprox(y, unfixed; rtol = 1e-10)
        m = normal isa Real ? abs(normal) : maximum(abs, normal)
        s = ldexp(1.0, -exponent(m))
        an, bn = Float64.(normal) .* s, Float64(b) * s
        products = an .* Float64.(y)
        gap = sum(products) - bn
        scale = sum(abs, products) + abs(bn)
        @test (F === IndHalfspace ? max(gap, zero(gap)) : abs(gap)) <= tol * scale
      end
    end
  end
end

end
