module DistanceTest

using Test
using ProximalBase
using LinearAlgebra
using Random

using ..TestUtils

const PB = ProximalBase
const PT = PB.Testing
const oracles = (prox!, value_and_prox!, gradient!, value_and_gradient!)
Random.seed!(178)

struct NoProjection <: AbstractFunction end
PB._isindicator(::Type{NoProjection}) = true

mutable struct CountingProjection{F<:AbstractFunction} <: AbstractFunction
  f::F
  calls::Int
  resets::Int
end
PB._isindicator(::Type{<:CountingProjection}) = true
PB.isproximable(::Type{<:CountingProjection}) = true
PB.isconvex(::Type{<:CountingProjection}) = true
function PB.prox!(y::AbstractArray, C::CountingProjection, x::AbstractArray, γ::Real)
  C.calls += 1
  prox!(y, C.f, x, γ)
end
PB.reset_cache!(C::CountingProjection) = (C.resets += 1; C)

function oracle!(op, y, f, x)
  @nospecialize op y f x
  op === prox! || op === value_and_prox! ? op(y, f, x, 0.7) : op(y, f, x)
end

function unchanged(op, f, point, E)
  @nospecialize op f point E
  x, y = deepcopy(point), deepcopy(point)
  before = deepcopy(y)
  @test_throws E oracle!(op, y, f, x)
  @test isequal(y, before)
  @test isequal(x, point)
end

# Every allowed support gives a quadratic minimizer; compare their objectives.
function subset_prox(x, k, λ, γ)
  best, objective = similar(x), Inf
  for mask in 0:(1 << length(x)) - 1
    count_ones(mask) <= k || continue
    p = [iszero(mask & (1 << (i - 1))) ? 0.0 : x[i] for i in eachindex(x)]
    y = [iszero(mask & (1 << (i - 1))) ? x[i] / (1 + γ * λ) : x[i] for i in eachindex(x)]
    value = λ / 2 * sum(abs2, y .- p) + sum(abs2, y .- x) / (2γ)
    if value < objective
      best, objective = y, value
    end
  end
  best, objective
end

@testset "SqrDistL2" begin
  x, λ, γ = [-2.3, 0.4, 3.1, -0.7], 1.7, 0.6
  @test SqrDistL2(IndBox(-1.0, 1.0)).λ == 1
  @test_throws ArgumentError SqrDistL2(NormL1())
  @test_throws ArgumentError SqrDistL2(NoProjection())
  for scale in (-1.0, Inf, -Inf, NaN)
    @test_throws ArgumentError SqrDistL2(IndBox(-1.0, 1.0), scale)
  end

  @testset "convex $(nameof(typeof(C)))" for C in (
      IndBox(-0.5, 1.2), IndBallL2(1.3), IndSimplex(1.4),
      IndAffine([1.0 1.0 0.0 0.0; 0.0 0.0 1.0 1.0], [0.5, -0.2]))
    f, e = SqrDistL2(C, λ), MoreauEnvelope(C, 1 / λ)
    check_traits(f; smooth = true, proximable = true, convex = true, separable = isseparable(C))
    @test f(x) ≈ e(x) rtol = 1e-12
    @test gradient(f, x) ≈ gradient(e, x) rtol = 1e-12
    @test prox(f, x, γ) ≈ prox(e, x, γ) rtol = 1e-12
    check_gradient(f, x)
    check_prox(f, x, γ)
    y, g = similar(x), similar(x)
    @test value_and_gradient!(g, f, x) ≈ f(x) rtol = 1e-12
    @test value_and_prox!(y, f, x, γ) ≈ f(y) rtol = 1e-12
    @test lipschitz(f) == λ
    @test_throws ArgumentError gradient!(x, f, x)
    @test_throws ArgumentError value_and_gradient!(x, f, x)
  end

  @testset "sparse supports" begin
    for n in (1, 5, 8), k in (0, 1, n, n + 1)
      z = collect(range(-2.7, 3.2; length = n + 1))[1:n]
      f = SqrDistL2(IndBallL0(k), λ)
      y, objective = subset_prox(z, k, λ, γ)
      check_traits(f; proximable = true)
      @test prox(f, z, γ) ≈ y rtol = 1e-12
      @test prox_objective(f, y, z, γ) ≈ objective rtol = 1e-12
      @test_throws ArgumentError lipschitz(f)
    end
    f = SqrDistL2(IndBallL0(2), λ)
    check_gradient(f, [4.2, -2.1, 0.3, -0.1])
    check_prox(f, x, γ)
    @test f([0.0, 2.0, 0.0, -1.0]) == 0
    @test prox(f, [0.0, 2.0, 0.0, -1.0], γ) == [0.0, 2.0, 0.0, -1.0]
  end

  @testset "nonconvex sets" begin
    groups = IndBallGroupL0(1, [1:2, 3:4])
    for C in (groups, IndSphereL2(1.4))
      f = SqrDistL2(C, λ)
      p = prox(C, x)
      check_traits(f; proximable = true)
      @test f(x) ≈ λ / 2 * sum(abs2, x .- p) rtol = 1e-12
      @test gradient(f, x) ≈ λ .* (x .- p) rtol = 1e-12
      @test prox(f, x, γ) ≈ (x .+ γ * λ .* p) ./ (1 + γ * λ) rtol = 1e-12
      check_gradient(f, x)
      check_prox(f, x, γ)
      @test_throws ArgumentError lipschitz(f)
    end
    X = [3.1 0.2; 0.4 1.0; -0.5 0.3]
    F = svd(X)
    p = F.U[:, 1:1] * Diagonal(F.S[1:1]) * F.Vt[1:1, :]
    f = SqrDistL2(IndBallRank(1), λ)
    @test f(X) ≈ λ / 2 * sum(abs2, X .- p) rtol = 1e-12
    @test prox(f, X, γ) ≈ (X .+ γ * λ .* p) ./ (1 + γ * λ) rtol = 1e-12
    check_gradient(f, X)
    check_prox(f, X, γ)
    check_traits(f; proximable = true)
    @test_throws ArgumentError lipschitz(f)
  end

  @testset "projection ties" begin
    for (C, z) in ((IndBallL0(1), [1.0, -1.0]),
                   (IndSphereL2(2.0), zeros(3)),
                   (IndBallRank(1), Matrix{Float64}(I, 2, 2)))
      f = SqrDistL2(C, λ)
      y = prox(f, z, γ)
      distance = C isa IndSphereL2 ? 2.0 : 1.0
      reference = λ * distance^2 / (2 * (1 + γ * λ))
      @test prox_objective(f, y, z, γ) ≈ reference rtol = 1e-12
      @test value_and_prox!(similar(z), f, z, γ) ≈ f(y) rtol = 1e-12
      @test !issmooth(f)
    end
    tied = [1.0, -1.0, 0.2]
    f = SqrDistL2(IndBallL0(1), λ)
    _, reference = subset_prox(tied, 1, λ, γ)
    @test prox_objective(f, prox(f, tied, γ), tied, γ) ≈ reference rtol = 1e-12
  end

  @testset "separable and restricted" begin
    for C in (IndBox(-0.5, 1.2), IndBox([0.5, -1.0, 0.2, -2.0], [2.0, 0.0, 3.0, 1.0]))
      f = SqrDistL2(C, λ)
      PT.test_separable(f, x, γ)
      PT.test_kkt(f, x, γ)
      @test iszeropreserving(f) == iszeropreserving(C)
      @test coordinate_lipschitz(f, 2) == λ
      @test coordinate_lipschitz!(similar(x), f) == fill(λ, length(x))
      S = [2, 4]
      restricted = restrict(f, S)
      supported = zeros(length(x))
      supported[S] = x[S]
      @test gradient(restricted, x[S]) ≈ gradient(f, supported)[S] rtol = 1e-12
      @test restricted(x[S]) - restricted(zeros(2)) ≈ f(supported) - f(zeros(4)) rtol = 1e-12
      g = λ .* prox(C, zeros(4))
      @test all(i -> iszero(kkt_violation(f, g, i)), eachindex(g))
    end
    coupled = SqrDistL2(IndHyperplane([1.0, 1.0], 1.0), λ)
    @test_throws ArgumentError restrict(coupled, [1])
    @test_throws ArgumentError coordinate_lipschitz(coupled, 1)
    @test_throws ArgumentError prox_entry(coupled, 0.0, 1, γ)
    @test_throws ArgumentError kkt_violation(coupled, zeros(2), 1)
    @test_throws MethodError prox(SqrDistL2(IndBox(-1.0, 1.0), λ), x, ones(4))
    @test_throws ArgumentError Spectral(SqrDistL2(IndBox(-1.0, 1.0), λ))
  end

  @testset "wrapped sets" begin
    for (C, z) in ((Translate(IndBallL0(1), ones(4)), x),
                   (Precompose(IndBox(-1.0, 1.0), 2.0), x),
                   (Precompose(IndBox(-1.0, 1.0), [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0], 1.0), x),
                   (Constrained(IndBox(0.0, 1.0), IndHyperplane(1.0, 1.5)), x),
                   (Spectral(IndBox(-0.5, 1.0)), [2.0 0.3; 0.3 -1.0]))
      f = SqrDistL2(C, λ)
      p = prox(C, z)
      @test f(z) ≈ λ / 2 * sum(abs2, z .- p) rtol = 1e-12
      @test prox(f, z, γ) ≈ (z .+ γ * λ .* p) ./ (1 + γ * λ) rtol = 1e-12
      check_gradient(f, z; directions = 1)
    end
  end

  @testset "zero scale, empty arrays and projection counts" begin
    C = CountingProjection(IndBox(0.5, 1.0), 0, 0)
    for scale in (0.0, λ)
      f = SqrDistL2(C, scale)
      expected = iszero(scale) ? 0 : 1
      before = C.calls
      @test isfinite(f(x))
      @test C.calls - before == expected
      for op in oracles
        before = C.calls
        y = fill(7.0, length(x))
        oracle!(op, y, f, x)
        @test C.calls - before == expected
        if iszero(scale)
          @test y == (op === prox! || op === value_and_prox! ? x : zeros(4))
        end
      end
      @test reset_cache!(f) === f
      @test C.resets == (iszero(scale) ? 1 : 2)
      @test Scaled(f, 0) isa ZeroFunction
    end
    for C in (IndBox(0.5, 1.0), IndBallL0(1), IndSphereL2()), scale in (0.0, λ)
      f = SqrDistL2(C, scale)
      for op in (prox!, value_and_prox!)
        z = copy(x)
        expected = prox(f, x, γ)
        result = op(z, f, z, γ)
        @test z ≈ expected rtol = 1e-12
        op === value_and_prox! && @test result ≈ f(z) rtol = 1e-12
      end
      if iszero(scale)
        @test f(x) == 0
        @test gradient(f, x) == zeros(4)
        @test lipschitz(f) == 0
      end
    end
    for C in (IndBox(-1.0, 1.0), IndBallL0(0)), scale in (0.0, λ)
      f = SqrDistL2(C, scale)
      @test f(Float64[]) == 0
      @test value_and_gradient!(Float64[], f, Float64[]) == 0
      @test value_and_prox!(Float64[], f, Float64[], γ) == 0
    end
    for scale in (0.0, λ)
      f = SqrDistL2(IndSphereL2(), scale)
      @test_throws ArgumentError f(Float64[])
      for op in oracles
        @test_throws ArgumentError oracle!(op, Float64[], f, Float64[])
      end
    end
  end

  @testset "checks before writes" begin
    for (C, z, E) in ((IndBallGroupL0(1, [1:3]), [2.0, 3.0], BoundsError),
                      (IndAffine(ones(1, 3), [1.0]), [2.0, 3.0], DimensionMismatch),
                      (Spectral(IndBox(-1.0, 1.0)), ones(2, 3), DimensionMismatch),
                      (IndBox(zeros(3), ones(3)), [2.0, 3.0], DimensionMismatch))
      for scale in (0.0, λ)
        f = SqrDistL2(C, scale)
        for op in oracles
          unchanged(op, f, z, E)
          unchanged(op, Tilt(f, ones(size(z))), z, E)
          unchanged(op, Translate(f, ones(size(z))), z, E)
          unchanged(op, SeparableSum(SqrNormL2(), f), ([2.0], z), E)
        end
        for op in (prox!, value_and_prox!)
          alias = copy(z)
          @test_throws E op(alias, f, alias, γ)
          @test alias == z
        end
      end
    end
    for op in oracles
      y = [7.0]
      @test_throws DimensionMismatch oracle!(op, y, SqrDistL2(IndBox(-1.0, 1.0)), x)
      @test y == [7.0]
    end
  end
end

@testset "distance precision and cache reuse" begin
  for T in (Float32, Float64)
    x = T[-2.3, 0.4, 3.1, -0.7]
    f = SqrDistL2(IndBox(-0.5, 1.2), 1.7)
    tol = T === Float32 ? 8eps(T) : 1e-12
    g, y = similar(x), similar(x)
    reference = T(1.7) / 2 * sum(abs2, x .- clamp.(x, T(-0.5), T(1.2)))
    @test f(x) isa T
    @test f(x) ≈ reference rtol = tol
    @test value_and_gradient!(g, f, x) isa T
    @test value_and_prox!(y, f, x, 0.6) isa T
    @test value_and_prox!(y, f, x, 0.6) ≈ f(y) rtol = tol
    check_gradient(f, x)
    check_prox(f, x, T(0.6); rtol = tol, atol = tol)
    for S in (Float64, Float32, Float64)
      output = zeros(S, length(x))
      gradient!(output, f, x)
      @test f.scratch.buf isa Vector{T}
      @test output ≈ gradient(f, x) rtol = max(tol, 8eps(S))
      value_and_gradient!(output, f, x)
      @test f.scratch.buf isa Vector{T}
      prox!(output, f, x, 0.6)
      @test f.scratch.buf isa Vector{S}
      @test output ≈ prox(f, x, 0.6) rtol = max(tol, 8eps(S))
      value_and_prox!(output, f, x, 0.6)
      @test f.scratch.buf isa Vector{S}
    end
  end
end

@testset "distance at extreme scales" begin
  for T in (Float32, Float64)
    large = T === Float32 ? T(1e30) : T(1e200)
    small = inv(large)
    tiny = T === Float32 ? T(1e-40) : T(1e-320)
    huge = T === Float32 ? T(1e20) : T(1e160)
    c = T(0.6) * floatmax(T)
    cases = ((zero(T), large, small), (zero(T), small, large),
             (zero(T), huge, tiny), (c, -c, inv(floatmax(T))))
    for (p, v, λ) in cases
      f, x = SqrDistL2(IndBox(p, p), λ), T[v]
      reference = T(BigFloat(λ) * (BigFloat(v) - BigFloat(p))^2 / 2)
      derivative = T(BigFloat(λ) * (BigFloat(v) - BigFloat(p)))
      @test isfinite(reference) && reference > 0
      @test f(x) ≈ reference rtol = 16eps(T)
      g = similar(x)
      @test value_and_gradient!(g, f, x) ≈ reference rtol = 16eps(T)
      @test g[1] ≈ derivative rtol = 8eps(T)
      @test gradient(f, x)[1] ≈ derivative rtol = 8eps(T)
      for alias in (false, true)
        input = copy(x)
        output = alias ? input : similar(x)
        value = value_and_prox!(output, f, input, one(T))
        exact = T(BigFloat(λ) * (BigFloat(output[1]) - BigFloat(p))^2 / 2)
        @test value ≈ exact rtol = 16eps(T)
      end
    end

    # Both a subnormal weight and an overflowing product can retain a
    # representable contribution after multiplication by a large endpoint.
    for (γ, λ, v, p) in ((huge, huge, floatmax(T) / 2, zero(T)),
                         (small, small, zero(T), floatmax(T) / 2),
                         (huge, huge, -c, c), (small, small, -c, c))
      f, x = SqrDistL2(IndBox(p, p), λ), T[v]
      product = BigFloat(γ) * BigFloat(λ)
      reference = T((BigFloat(v) + product * BigFloat(p)) / (1 + product))
      @test prox(f, x, γ)[1] ≈ reference rtol = 16eps(T)
      @test prox_entry(f, v, 1, γ) ≈ reference rtol = 16eps(T)
      y = copy(x)
      value = value_and_prox!(y, f, y, γ)
      @test y[1] ≈ reference rtol = 16eps(T)
      exact = T(BigFloat(λ) * (BigFloat(y[1]) - BigFloat(p))^2 / 2)
      @test value ≈ exact rtol = 16eps(T)
    end
    f = SqrDistL2(IndBox(c, c), one(T))
    @test prox(f, T[-c], one(T)) == T[0]
  end
end

a_value(f, x) = f(x)
a_grad!(g, f, x) = gradient!(g, f, x)
a_vgrad!(g, f, x) = value_and_gradient!(g, f, x)
a_prox!(y, f, x, γ) = prox!(y, f, x, γ)
a_vprox!(y, f, x, γ) = value_and_prox!(y, f, x, γ)

@testset "distance zero allocations" begin
  for T in (Float32, Float64), C in (IndBox(-0.5, 1.2), IndBallL0(3))
    x = T.(range(-3.1, 2.7; length = 12))
    y, g = similar(x), similar(x)
    f = SqrDistL2(C, T(1.7))
    @test allocs(a_value, f, x) == 0
    @test allocs(a_grad!, g, f, x) == 0
    @test allocs(a_vgrad!, g, f, x) == 0
    @test allocs(a_prox!, y, f, x, T(0.6)) == 0
    @test allocs(a_vprox!, y, f, x, T(0.6)) == 0
    @test allocs(a_prox!, x, f, x, T(0.6)) == 0
    @test allocs(a_vprox!, x, f, x, T(0.6)) == 0
    @inferred value_and_gradient!(g, f, x)
    @inferred value_and_prox!(y, f, x, T(0.6))
  end
end

end
