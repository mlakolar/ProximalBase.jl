module SqrHingeTest

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using Random
using ..TestUtils

const PB = ProximalBase
const PT = PB.Testing

Random.seed!(184)

@testset "SqrHingeLoss oracles" begin
  x = [-0.3, 0.2, 1.5, -1.4, 0., 2.]
  y = [1., -1., 1., -1., -1., 1.]
  yg = [2., -0.5, 1., -3., -2., 0.25]
  w = [0.7, 0., 1.2, 2., 0.3, 1.]
  for f in (SqrHingeLoss(), SqrHingeLoss(y), SqrHingeLoss(y, w),
            SqrHingeLoss(yg, w), SqrHingeLoss(nothing, w))
    yl = f.y === nothing ? ones(6) : f.y
    wl = f.w === nothing ? ones(6) : f.w
    @test f(x) ≈ sum(wl .* max.(0, 1 .- yl .* x).^2) rtol = 1e-12
    @test gradient(f, x) ≈ -2 .* wl .* yl .* max.(0, 1 .- yl .* x) rtol = 1e-12
    PT.test_traits(f; smooth = true, proximable = true, separable = true, convex = true)
    @test !iszeropreserving(f)
    @test !hashessian(f)
    PT.test_separable(f, x, 0.4; groups = 2)
    PT.test_gradient(f, x)
    PT.test_kkt(f, x, 0.4)
    PT.test_prox(f, x, 0.4)
    PT.test_conjugate(f, x, 0.4)
    @test lipschitz(f) ≈ maximum(2 .* wl .* yl.^2) rtol = 1e-12
    L = similar(x)
    @test coordinate_lipschitz!(L, f) === L
    @test L ≈ 2 .* wl .* yl.^2 rtol = 1e-12
    @test L == [coordinate_lipschitz(f, i) for i in 1:6]
    for γ in (1e-6, 0.4, 1e6)
      p = [wl[i] == 0 || yl[i] * x[i] >= 1 ? x[i] :
           (x[i] + 2γ * wl[i] * yl[i]) / (1 + 2γ * wl[i] * yl[i]^2) for i in 1:6]
      @test prox(f, x, γ) ≈ p rtol = 1e-12
    end
    g = 2 .* wl .* yl
    for i in 1:6
      @test kkt_violation(f, g, i) == 0
      for δ in (-0.25, 0.25)
        h = copy(g)
        h[i] += δ
        @test kkt_violation(f, h, i) ≈ abs(δ) atol = eps(Float64)
      end
    end
    S = [1, 3, 5]
    r = restrict(f, S)
    @test r(x[S]) ≈ sum(value_entry(f, x[i], i) for i in S)
    @test gradient(r, x[S]) ≈ gradient(f, x)[S]
  end
  @test lipschitz(SqrHingeLoss(nothing, zeros(6))) == 0
  for f in (SqrHingeLoss(Float64[]), SqrHingeLoss(nothing, Float32[]))
    @test lipschitz(f) == 0
  end

  for wl in (nothing, w)
    f = SqrHingeLoss(y, wl)
    glm = GLMLoss(PB.SquaredHinge(), y, Matrix{Float64}(I, 6, 6); weights = wl)
    @test f(x) ≈ glm(x) rtol = 1e-12
    @test gradient(f, x) ≈ gradient(glm, x) rtol = 1e-12
  end
end

@testset "brute force on moderate scales" begin
  # The minimizer lies between v and 1/y. Extra precision resolves the minimum
  # to 1e-8 even when the objective rounds flat in Float64.
  for y in (-2., -0.5, 1., 3.), w in (0., 0.7), v in (-0.4, 0., 1.5), γ in (0.1, 2.)
    f = SqrHingeLoss([y], [w])
    term(t) = big(w) * max(zero(t), 1 - big(y) * t)^2
    lo, hi = min(v, 1 / y) - 1, max(v, 1 / y) + 1
    p, _ = PT.brute_force_prox(term, big(v), big(γ); lo = big(lo), hi = big(hi), n = 129)
    @test prox_entry(f, v, 1, γ) ≈ Float64(p) atol = 1e-8 rtol = 0
  end
end

@testset "conjugate values" begin
  y = [2., -0.5, 1.]
  w = [0.7, 2., 0.]
  c = Conjugate(SqrHingeLoss(y, w))
  s = [-0.3, 0.4, 0.]
  @test c(s) ≈ sum(s[i] / y[i] + s[i]^2 / (4w[i] * y[i]^2) for i in 1:2) rtol = 1e-12
  @test c(zeros(3)) == 0
  @test c([0.1, 0., 0.]) == Inf
  @test c([0., -0.1, 0.]) == Inf
  @test c([0., 0., 0.1]) == Inf
  @test c([0., 0., -0.1]) == Inf
  @test Conjugate(SqrHingeLoss())([-1., -2., 0.]) == -1.75
  @test Conjugate(SqrHingeLoss(nothing, [0.]))([0.]) == 0
  @test Conjugate(SqrHingeLoss(nothing, [0.]))([-1.]) == Inf
  p = zeros(3)
  @test value_and_prox!(p, c, s, 0.4) ≈ c(p)
end

@testset "diagonal steps and aliasing" begin
  x = [-0.3, 0., 1.5, -1.4, 0.2, 2.]
  f = SqrHingeLoss([2., -0.5, 1., -3., -2., 0.25], [0.7, 0., 1.2, 2., 0.3, 1.])
  γ = [1e-6, 0.2, 0.5, 1., 3., 1e6]
  p = [prox_entry(f, x[i], i, γ[i]) for i in eachindex(x)]
  z = similar(x)
  @test prox!(z, f, x, γ) === z
  @test z ≈ p rtol = 1e-12
  @test value_and_prox!(z, f, x, γ) ≈ f(p) rtol = 1e-12
  @test z ≈ p rtol = 1e-12
  for step in (0.4, γ)
    expected = prox(f, x, step)
    z = copy(x)
    @test prox!(z, f, z, step) ≈ expected rtol = 1e-12
    z = copy(x)
    @test value_and_prox!(z, f, z, step) ≈ f(expected) rtol = 1e-12
    @test z ≈ expected rtol = 1e-12
  end
  z = copy(x)
  @test_throws ArgumentError value_and_gradient!(z, f, z)
  @test z == x
  z = copy(x)
  @test_throws ArgumentError gradient!(z, f, z)
  @test z == x
end

@testset "sparse and atom iterates" begin
  x = [0., -0.5, 0., 1.5, 0., 0.2]
  for f in (SqrHingeLoss(), SqrHingeLoss([1., -2., 0.5, 1., -1., 3.], [0., 1., 2., 0.5, 1., 3.]))
    for xs in (SparseIterate(x), AtomIterate(x, 2))
      @test f(xs) ≈ f(x)
      @test f(xs) ≈ sum(value_entry(f, x[i], i) for i in 1:6)
      @test gradient(f, xs) ≈ gradient(f, x)
      z = similar(x)
      @test prox!(z, f, xs, 0.4) ≈ prox(f, x, 0.4)
      zs = similar(xs)
      @test Vector(prox!(zs, f, xs, 0.4)) ≈ prox(f, x, 0.4)
      @test value_and_prox!(zs, f, xs, fill(0.4, 6)) ≈ f(prox(f, x, 0.4))
    end
  end
  @test nnz(prox(SqrHingeLoss(), SparseIterate(x), 0.4)) == length(x)
  X = [0. -0.5; -0.5 1.5]
  y, w = [1. -2.; -2. 0.5], [0.7 1.; 1. 2.]
  for f in (SqrHingeLoss(), SqrHingeLoss(y, w))
    xs = SymmetricSparseIterate(X)
    @test f(xs) ≈ f(X)
    @test Matrix(prox(f, xs, 0.4)) ≈ prox(f, X, 0.4)
    @test Matrix(gradient(f, xs)) ≈ gradient(f, X)
    PT.test_separable(f, X, 0.4; symmetric = true)
  end
end

@testset "extreme arithmetic" begin
  for (T, k, huge, small) in ((Float64, 600, 1023, 60), (Float32, 80, 127, 40))
    high, low = ldexp(one(T), k), ldexp(one(T), -k)
    h = ldexp(one(T), huge)
    cases = ((one(T), one(T), zero(T), ldexp(one(T), -small)),
             (one(T), one(T), -h, h),
             (high, one(T), high, one(T)),
             (high, low, zero(T), low),
             (low, high, zero(T), high),
             (nextfloat(zero(T)), one(T), zero(T), one(T)))
    for (y, w, v, γ) in cases
      f = SqrHingeLoss([y], [w])
      p = prox_entry(f, v, 1, γ)
      by, bw, bv, bg = big(y), big(w), big(v), big(γ)
      ref = by * bv >= 1 ? bv : (bv + 2bg * bw * by) / (1 + 2bg * bw * by^2)
      @test p isa T
      @test isfinite(p)
      @test p ≈ T(ref) rtol = 8eps(T) atol = 0
    end
    @test prox_entry(SqrHingeLoss([high]), high, 1, one(T)) == high
    f = SqrHingeLoss([one(T)], [low])
    x = T[-high]
    ref = big(low) * (1 + big(high))^2
    @test isfinite(f(x))
    @test f(x) ≈ T(ref) rtol = 8eps(T)
    @test gradient(f, x)[1] ≈ T(-2big(low) * (1 + big(high))) rtol = 8eps(T)
    g = similar(x)
    @test value_and_gradient!(g, f, x) == f(x)
    @test g == gradient(f, x)
    f = SqrHingeLoss([low], [high])
    @test coordinate_lipschitz(f, 1) == 2low
    @test lipschitz(f) == 2low
    c = Conjugate(SqrHingeLoss([one(T)], [high]))
    @test isfinite(c(T[-high]))
    @test c(T[-high]) ≈ T(-big(high) + big(high)^2 / (4big(high))) rtol = 8eps(T)
    f = SqrHingeLoss([high], [zero(T)])
    @test f(T[-h]) == 0
    @test gradient(f, T[-h]) == T[0]
    @test value_and_gradient!(g, f, T[-h]) == 0
    @test g == T[0]
    @test coordinate_lipschitz(f, 1) == lipschitz(f) == 0
    @test prox_entry(f, -h, 1, h) == -h
    # The error relation remains measurable before the point rounds to 1/y.
    y, w, v, γ = T(2), T(0.7), T(-0.4), T(100)
    p = prox_entry(SqrHingeLoss([y], [w]), v, 1, γ)
    err = abs(big(v) - inv(big(y))) / (1 + 2big(γ) * big(w) * big(y)^2)
    @test abs(big(p) - inv(big(y))) ≈ err rtol = 4eps(T) / T(err)
  end
  # Even an overflowing margin can have a finite weighted square and slope.
  y, w, x = 2.0^600, 2.0^-1070, -2.0^430
  f = SqrHingeLoss([y], [w])
  @test f([x]) ≈ Float64(big(w) * (1 - big(y) * big(x))^2) rtol = 1e-12
  @test gradient(f, [x])[1] ≈ Float64(-2big(w) * big(y) * (1 - big(y) * big(x))) rtol = 1e-12
  @test Conjugate(SqrHingeLoss([0.25], [2.0^1022]))([-2.0^1022]) == 0
end

@testset "precision" begin
  for x in (Float32[-0.2, 0., 1.5], Float64[-0.2, 0., 1.5], [-1, 0, 2], Float32[], Int[])
    n, T = length(x), float(eltype(x))
    for P in (Float32, Float64), G in (Float32, Float64)
      y, w = fill(P(-1), n), fill(P(0.7), n)
      for f in (SqrHingeLoss(), SqrHingeLoss(y, w), SqrHingeLoss(nothing, w))
        g = zeros(G, n)
        v = value_and_gradient!(g, f, x)
        @test v isa T
        @test f(x) isa T
        @test v ≈ f(x)
        @test g ≈ G.(gradient(f, x)) rtol = 8eps(Float32)
        @test prox(f, x, 0.4) isa Vector{T}
        z = zeros(G, n)
        @test prox!(z, f, x, 0.4) === z
        @test z ≈ G.(prox(f, x, 0.4)) rtol = 8eps(Float32)
        @test value_and_prox!(z, f, x, 0.4) isa G
      end
    end
  end
  f = SqrHingeLoss(Float32[1, -1], Float32[0.7, 1])
  @test @inferred(prox_entry(f, 0.2f0, 1, 0.3f0)) isa Float32
  @test @inferred(f(Float32[0.2, 0.3])) isa Float32
end

@testset "checks before writes" begin
  for y in ([0.], [Inf], [-Inf], [NaN], [1im])
    @test_throws ArgumentError SqrHingeLoss(y)
  end
  for w in ([-1.], [Inf], [-Inf], [NaN], [1im])
    @test_throws ArgumentError SqrHingeLoss(nothing, w)
  end
  x = [0.2, -0.3, 0.4]
  for f in (SqrHingeLoss([1., 2.]), SqrHingeLoss(nothing, [1., 2.]),
            SqrHingeLoss(ones(3), ones(2)))
    @test_throws DimensionMismatch f(x)
    @test_throws DimensionMismatch Conjugate(f)(x)
    for op in (gradient!, value_and_gradient!)
      g = fill(99., 3)
      @test_throws DimensionMismatch op(g, f, x)
      @test g == fill(99., 3)
    end
    for step in (0.4, ones(3)), op in (prox!, value_and_prox!)
      z = fill(99., 3)
      @test_throws DimensionMismatch op(z, f, x, step)
      @test z == fill(99., 3)
    end
    L = fill(99., 3)
    @test_throws DimensionMismatch coordinate_lipschitz!(L, f)
    @test L == fill(99., 3)
  end
  @test_throws DimensionMismatch lipschitz(SqrHingeLoss(ones(3), ones(2)))
  for op in (gradient!, value_and_gradient!)
    g = fill(99., 2)
    @test_throws DimensionMismatch op(g, SqrHingeLoss(), x)
    @test g == fill(99., 2)
    g = fill(99., 3, 1)
    @test_throws DimensionMismatch op(g, SqrHingeLoss(), x)
    @test g == fill(99., 3, 1)
  end
  @test_throws BoundsError prox_entry(SqrHingeLoss([1.]), 0., 2, 0.4)
  @test_throws BoundsError coordinate_lipschitz(SqrHingeLoss([1.]), 2)
  @test_throws BoundsError coordinate_lipschitz(SqrHingeLoss(nothing, [1.]), 2)
end

@testset "zero allocations" begin
  x = [-0.3, 0., 1.5, -1.4, 0.2, 2.]
  for f in (SqrHingeLoss(), SqrHingeLoss([2., -0.5, 1., -3., -2., 0.25], [0.7, 0., 1.2, 2., 0.3, 1.]))
    for z in (x, Float32.(x), SparseIterate(x))
      p, g = similar(z), similar(z)
      @test allocs(f, z) == 0
      @test allocs(prox!, p, f, z, 0.4) == 0
      @test allocs(value_and_prox!, p, f, z, 0.4) == 0
      @test allocs(gradient!, g, f, z) == 0
      @test allocs(value_and_gradient!, g, f, z) == 0
    end
    @test allocs(prox_coordinate!, SparseIterate(x), f, 2, 0.4) == 0
  end
end

end
