module DistL2Test

using Test
using ProximalBase
using LinearAlgebra
using Random
using Statistics

using ..TestUtils

const PB = ProximalBase
const oracles = (prox!, value_and_prox!)
Random.seed!(177)

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

# In two dimensions the simplex is a line segment with a scalar projection.
box_distance(x) = hypot(max(-0.5 - x[1], 0, x[1] - 1), max(-0.5 - x[2], 0, x[2] - 1))
ball_distance(x) = max(norm(x) - 0.7, 0)
function simplex_distance(x)
  t = clamp((x[1] - x[2] + 1) / 2, 0, 1)
  hypot(x[1] - t, x[2] - (1 - t))
end

# Solve the radial problem on every allowed support, without using a set oracle.
function support_objective(x, k, λ, γ)
  best = Inf
  for mask in 0:(1 << length(x)) - 1
    count_ones(mask) <= k || continue
    excluded = [iszero(mask & (1 << (i - 1))) for i in eachindex(x)]
    r = norm(x[excluded])
    s = iszero(r) ? 1.0 : max(1 - γ * λ / r, 0)
    y = [excluded[i] ? s * x[i] : x[i] for i in eachindex(x)]
    best = min(best, λ * norm(y[excluded]) + sum(abs2, y .- x) / (2γ))
  end
  best
end
sparse_distance(x, k) = norm(sort(abs.(x); rev = true)[min(k, length(x)) + 1:end])

function unchanged(op, f, point, E)
  @nospecialize op f point E
  for alias in (false, true)
    x = deepcopy(point)
    y = alias ? x : deepcopy(point)
    before = deepcopy(y)
    @test_throws E op(y, f, x, 0.7)
    @test isequal(y, before)
    @test isequal(x, point)
  end
end

@testset "DistL2 construction and traits" begin
  @test DistL2(IndBox(0, 0)).λ === 1
  @test_throws ArgumentError DistL2(NormL1())
  @test_throws ArgumentError DistL2(NoProjection())
  for λ in (-1.0, Inf, -Inf, NaN)
    @test_throws ArgumentError DistL2(IndBox(0, 0), λ)
  end
  for λ in (0, 1, 1.7), C in (IndBox(0, 1), IndBallL0(1))
    f = DistL2(C, λ)
    check_traits(f; proximable = true, convex = isconvex(C))
    @test !iszeropreserving(f)
    @test_throws ArgumentError lipschitz(f)
    @test_throws ArgumentError coordinate_lipschitz(f, 1)
    @test_throws ArgumentError coordinate_lipschitz!(zeros(2), f)
    @test_throws MethodError prox_entry(f, 0.0, 1, 0.7)
    @test_throws MethodError restrict(f, [1])
    @test_throws MethodError prox(f, ones(2), ones(2))
    @test_throws ArgumentError Spectral(f)
    @test_throws ArgumentError Constrained(f, IndBox(0, 1))
    @test Scaled(f, 0) isa ZeroFunction
  end
end

@testset "DistL2 at the origin and a common effect" begin
  x = [-2.3, 0.4, 3.1, -0.7]
  for C in (IndBox(0, 0), IndBallL2(0)), λ in (0, 1, 1.7), γ in (0.1, 0.7, 10.0)
    f, g = DistL2(C, λ), NormL2(λ)
    @test f(x) ≈ g(x) rtol = 1e-12
    @test prox(f, x, γ) ≈ prox(g, x, γ) rtol = 1e-12
    y = similar(x)
    v = value_and_prox!(y, f, x, γ)
    @test y ≈ prox(g, x, γ) rtol = 1e-12
    @test v ≈ g(y) rtol = 1e-12
    check_prox(f, x, γ)
  end
  A = [1.0 -1.0 0.0 0.0; 0.0 1.0 -1.0 0.0; 0.0 0.0 1.0 -1.0]
  f = DistL2(IndAffine(A, zeros(3)), 1.7)
  p = fill(mean(x), length(x))
  d = norm(x .- p)
  @test f(x) ≈ 1.7d rtol = 1e-12
  for γ in (0.2, 10.0)
    expected = d <= 1.7γ ? p : x .+ (1.7γ / d) .* (p .- x)
    @test prox(f, x, γ) ≈ expected rtol = 1e-12
    y = similar(x)
    @test value_and_prox!(y, f, x, γ) ≈ 1.7norm(y .- mean(y)) atol = 1e-14 rtol = 1e-12
    check_prox(f, x, γ)
  end
end

@testset "DistL2 independent convex distances and Cartesian search" begin
  x, λ = [-1.3, 1.8], 1.7
  for (C, distance) in ((IndBox(-0.5, 1), box_distance),
                        (IndBallL2(0.7), ball_distance),
                        (IndSimplex(1), simplex_distance)), γ in (0.2, 0.8, 3.0)
    f = DistL2(C, λ)
    @test f(x) ≈ λ * distance(x) rtol = 1e-12
    y = prox(f, x, γ)
    objective(z) = λ * distance(z) + sum(abs2, z .- x) / (2γ)
    grid = minimum(objective([a, b]) for a in -2:0.1:2, b in -2:0.1:2)
    @test objective(y) <= grid + 1e-12
    @test grid <= objective(y) + 0.15
    @test value_and_prox!(similar(x), f, x, γ) ≈ λ * distance(y) rtol = 1e-12 atol = 1e-14
    check_prox(f, x, γ)
  end
end

@testset "DistL2 nonconvex global objectives and ties" begin
  for x in ([3.1, -1.4, 0.3], [1.0, -1.0, 0.2]), k in (0, 1, 2, 3, 4), γ in (0.2, 1.0, 5.0)
    f = DistL2(IndBallL0(k), 1.7)
    y = prox(f, x, γ)
    objective = 1.7sparse_distance(y, k) + sum(abs2, y .- x) / (2γ)
    @test f(x) ≈ 1.7sparse_distance(x, k) rtol = 1e-12
    @test objective ≈ support_objective(x, k, 1.7, γ) rtol = 1e-12 atol = 1e-14
    @test value_and_prox!(similar(x), f, x, γ) ≈ 1.7sparse_distance(y, k) rtol = 1e-12 atol = 1e-14
  end
  for x in (zeros(3), [2.0, -0.3, 0.1]), γ in (0.2, 3.0)
    f = DistL2(IndSphereL2(1.4), 1.7)
    y = prox(f, x, γ)
    d = abs(norm(x) - 1.4)
    expected = d <= γ * 1.7 ? d^2 / (2γ) : 1.7d - γ * 1.7^2 / 2
    objective = 1.7abs(norm(y) - 1.4) + sum(abs2, y .- x) / (2γ)
    @test objective ≈ expected rtol = 1e-12 atol = 1e-14
    @test value_and_prox!(similar(x), f, x, γ) ≈ 1.7abs(norm(y) - 1.4) rtol = 1e-12 atol = 1e-14
  end
  for C in (IndBallGroupL0(1, [1:2, 3:4]), IndBallRank(1))
    x = C isa IndBallRank ? [3.0 0.0; 0.0 1.0] : [3.0, -2.0, 0.5, -0.1]
    check_prox(DistL2(C, 1.7), x, 0.7)
  end
end

@testset "DistL2 branches and rounded output values" begin
  for T in (Float32, Float64)
    f = DistL2(IndBox(0, 0), T(2))
    for d in (prevfloat(T(2)), T(2), nextfloat(T(2)))
      x = T[d]
      for op in oracles, alias in (false, true)
        input = copy(x)
        output = alias ? input : similar(x)
        v = op(output, f, input, one(T))
        @test output == T[max(d - T(2), 0)]
        op === value_and_prox! && @test v == T(2) * abs(output[1])
      end
    end
    c = T === Float32 ? T(2)^24 : T(2)^53
    f = DistL2(IndBox(c, c))
    for op in oracles, alias in (false, true)
      x = T[c + 2]
      y = alias ? x : similar(x)
      v = op(y, f, x, 1)
      @test y == T[c]
      op === value_and_prox! && @test v === zero(T)
      @test f(y) === zero(T)
    end
    f = DistL2(IndBox(T(1), T(1)))
    x = T[nextfloat(T(1))]
    @test f.f(x) == 0
    @test f(x) == eps(T)
  end
end

@testset "DistL2 zero scale, projection counts and empty arrays" begin
  x = [-2.3, 0.4, 3.1, -0.7]
  C = CountingProjection(IndBox(0.5, 1), 0, 0)
  for λ in (0, 1.7)
    f = DistL2(C, λ)
    expected = iszero(λ) ? 0 : 1
    before = C.calls
    @test isfinite(f(x))
    @test C.calls - before == expected
    for op in oracles, alias in (false, true), γ in (0.7, 10.0)
      z = copy(x)
      y = alias ? z : similar(z)
      before = C.calls
      v = op(y, f, z, γ)
      @test C.calls - before == expected
      if iszero(λ)
        @test y == x
        op === value_and_prox! && @test v === 0.0
      end
    end
    @test reset_cache!(f) === f
  end
  @test C.resets == 2
  f = DistL2(C)
  before = C.calls
  @test prox!(similar(x), f, x, 0) == x
  @test C.calls == before
  @test value_and_prox!(similar(x), f, x, 0) ≈ norm(x .- clamp.(x, 0.5, 1)) rtol = 1e-12
  @test C.calls == before + 1
  @test_throws ArgumentError reset_cache!(DistL2(IndAffine(ones(1, 2), [1.0])))
  for λ in (0, 1.7), C in (IndBox(-1, 1), IndBallL0(0), IndBallL2(0),
                            IndSimplex(0), IndSimplex(1; equality = false), IndAffine(zeros(0, 0), Float64[]))
    f = DistL2(C, λ)
    @test f(Float64[]) === 0.0
    @test prox!(Float64[], f, Float64[], 0.7) == Float64[]
    @test value_and_prox!(Float64[], f, Float64[], 0.7) === 0.0
  end
  for λ in (0, 1.7), C in (IndSphereL2(), IndSOC())
    f = DistL2(C, λ)
    @test_throws ArgumentError f(Float64[])
    for op in oracles
      unchanged(op, f, Float64[], ArgumentError)
    end
  end
  for λ in (0, 1.7), op in oracles
    f, z = DistL2(IndBox(0, 0), λ), copy(x)
    v = op(z, f, z, 0)
    @test z == x
    op === value_and_prox! && @test v ≈ λ * norm(x) rtol = 1e-12
  end
end

@testset "DistL2 checks before writes" begin
  for (C, x, E) in ((IndBox(zeros(3), ones(3)), [2.0, 3.0], DimensionMismatch),
                    (IndAffine(ones(1, 3), [1.0]), [2.0, 3.0], DimensionMismatch),
                    (Spectral(IndBox(-1, 1)), ones(2, 3), DimensionMismatch),
                    (IndBallGroupL0(1, [1:3]), [2.0, 3.0], BoundsError))
    for λ in (0, 1.7), op in oracles
      f = DistL2(C, λ)
      @test_throws E f(x)
      unchanged(op, f, x, E)
      unchanged(op, Tilt(f, ones(size(x))), x, E)
      unchanged(op, Translate(f, ones(size(x))), x, E)
      unchanged(op, SeparableSum(SqrNormL2(), f), ([2.0], x), E)
    end
  end
  for op in oracles, λ in (0, 1.7)
    x, y = [2.0, 3.0], [7.0]
    f = DistL2(IndBox(0, 1), λ)
    for g in (f, Tilt(f, ones(2)), Translate(f, ones(2)))
      @test_throws DimensionMismatch op(y, g, x, 0.7)
      @test y == [7.0]
      @test x == [2.0, 3.0]
    end
    output = ([5.0], y)
    @test_throws DimensionMismatch op(output, SeparableSum(SqrNormL2(), f), ([2.0], x), 0.7)
    @test output == ([5.0], [7.0])
  end
  @test_throws ArgumentError DistL2(IndBox(2, 1))
end

@testset "DistL2 precision and cache transitions" begin
  for T in (Float32, Float64)
    x = T[-2.3, 0.4, 3.1, -0.7]
    f = DistL2(IndBox(-0.5, 1.2), 1.7)
    tol = T === Float32 ? 8eps(T) : 1e-12
    @test f(x) isa T
    @test f.scratch.buf isa Vector{T}
    @test f(x) ≈ 1.7norm(x .- clamp.(x, T(-0.5), T(1.2))) rtol = tol
    for S in (Float64, Float32, Float64)
      y = zeros(S, length(x))
      @test prox!(y, f, x, 0.7) === y
      @test f.scratch.buf isa Vector{S}
      @test y ≈ prox(f, x, 0.7) rtol = max(tol, 8eps(S))
      v = value_and_prox!(y, f, x, 0.7)
      @test v isa S
      @test f.scratch.buf isa Vector{S}
      @test v ≈ f(y) rtol = max(tol, 8eps(S))
    end
    check_prox(f, x, T(0.7); rtol = tol, atol = tol)
    g = DistL2(IndBox(0, 0))
    @test g(x) isa T
    @test value_and_prox!(similar(x), g, x) isa T
  end
  f = DistL2(IndBox(0, 0), 1e100)
  x = Float32[3, 4]
  for op in oracles
    y = similar(x)
    v = op(y, f, x, 1e-100)
    @test y ≈ Float32[2.4, 3.2] rtol = 4eps(Float32)
    op === value_and_prox! && @test v === Float32(Inf)
  end
  f = DistL2(IndBox(0, 0), 1e-50)
  x = Float32[1e30]
  expected = Float32(BigFloat(f.λ) * BigFloat(x[1]))
  @test f(x) ≈ expected rtol = 4eps(Float32)
  y = similar(x)
  @test value_and_prox!(y, f, x, 1.0) ≈ expected rtol = 4eps(Float32)
  f = DistL2(IndBox(0, 0), 1e50)
  x = Float32[1e-30]
  expected = Float32(BigFloat(f.λ) * BigFloat(x[1]))
  @test f(x) ≈ expected rtol = 4eps(Float32)
  @test value_and_prox!(similar(x), f, x, 0) ≈ expected rtol = 4eps(Float32)
  @test prox(DistL2(IndBox(0, 0)), [3, 4], 1) ≈ [2.4, 3.2] rtol = 1e-12
end

@testset "DistL2 extreme distances and steps" begin
  for T in (Float32, Float64)
    c = T === Float32 ? T(2e38) : T(1e308)
    tiny = T === Float32 ? T(1e-38) : T(1e-308)
    step = T === Float32 ? T(1e-30) : T(1e-300)
    large = T === Float32 ? T(1e30) : T(1e300)
    maxval = floatmax(T)
    cases = ((T[c], T[-c], tiny, one(T)),
             (T[0, 0], T[0.75maxval, 0.75maxval], T(2), 0.52maxval),
             (T[c], T[0], one(T), step),
             (T[c], T[-c], one(T), c),
             (T[c], T[-c], T(1.5), c),
             (T[0], T[0.5maxval], large, large),
             (T[c], T[0], step, step),
             (T[0], T[large], tiny, one(T)))
    for (p, x, λ, γ) in cases
      f = DistL2(IndBox(p, p), λ)
      d = norm(BigFloat.(x) .- BigFloat.(p))
      t = BigFloat(γ) * BigFloat(λ)
      exact = d <= t ? BigFloat.(p) : BigFloat.(x) .+ (t / d) .* (BigFloat.(p) .- BigFloat.(x))
      reference = T.(exact)
      value = T(BigFloat(λ) * d)
      @test f(x) ≈ value rtol = 16eps(T)
      for op in oracles, alias in (false, true)
        input = copy(x)
        y = alias ? input : similar(x)
        v = op(y, f, input, γ)
        @test all(isfinite, y)
        @test y ≈ reference rtol = 16eps(T) atol = 0
        if op === value_and_prox!
          expected = T(BigFloat(λ) * norm(BigFloat.(y) .- BigFloat.(p)))
          @test v ≈ expected rtol = 16eps(T) atol = 0
        end
      end
    end
  end
  f = DistL2(IndBox(0, 0), 2)
  x, γ = [1.3e308, 1.3e308], 9e307
  d = norm(BigFloat.(x))
  reference = Float64.((1 - 2BigFloat(γ) / d) .* BigFloat.(x))
  @test prox(f, x, γ) ≈ reference rtol = 16eps(Float64)
  for T in (Float32, Float64), γ in (prevfloat(sqrt(T(2))), sqrt(T(2)), nextfloat(sqrt(T(2))))
    f, x = DistL2(IndBox(0, 0), one(T)), T[1, 1]
    a = max(1 - BigFloat(γ) / sqrt(BigFloat(2)), 0)
    reference = fill(T(a), 2)
    for op in oracles, alias in (false, true)
      z = copy(x)
      y = alias ? z : similar(z)
      op(y, f, z, γ)
      @test y ≈ reference rtol = 16eps(T) atol = 0
    end
  end
end

@testset "DistL2 non-finite arguments" begin
  for T in (Float32, Float64), v in (T(NaN), T(Inf), T(-Inf)), γ in (one(T), T(Inf))
    f, x = DistL2(IndBox(1, 1)), T[v]
    @test !isfinite(f(x))
    for op in oracles, alias in (false, true)
      z = copy(x)
      y = alias ? z : similar(x)
      result = op(y, f, z, γ)
      @test !isfinite(y[1])
      op === value_and_prox! && @test !isfinite(result)
    end
  end
end

@testset "DistL2 compositions" begin
  x, λ, γ = [-2.3, 0.4, 3.1, -0.7], 1.7, 0.7
  for C in (Translate(IndBallL0(1), ones(4)), Precompose(IndBox(-1, 1), 2),
            Constrained(IndBox(0, 1), IndHyperplane(1, 1.5)),
            IndIsotonic(), IndNonnegative())
    f = DistL2(C, λ)
    p = prox(C, x)
    d = norm(x .- p)
    @test f(x) ≈ λ * d rtol = 1e-12
    @test prox(f, x, γ) ≈ (d <= λ * γ ? p : x .+ (γ * λ / d) .* (p .- x)) rtol = 1e-12
  end
  X = [2.0 0.3; 0.3 -1.0]
  check_prox(DistL2(Spectral(IndBox(-0.5, 1)), λ), X, γ; symmetric = true)
  f = DistL2(IndBox(-0.5, 1), λ)
  for g in (Tilt(f, ones(4)), Translate(f, ones(4)), Scaled(f, 2))
    check_prox(g, x, γ)
  end
  f = DistL2(IndBox(0, 0), λ)
  c = Conjugate(f)
  @test prox(c, x, γ) ≈ prox(IndBallL2(λ), x) rtol = 1e-12
  y = similar(x)
  @test value_and_prox!(y, c, x, γ) ≈ 0 atol = 1e-14
  @test y ≈ prox(IndBallL2(λ), x) rtol = 1e-12
  @test_throws ArgumentError c(x)
  @test_throws ArgumentError Conjugate(DistL2(IndBallL0(1)))
  s = SeparableSum(f, DistL2(IndBox(1, 1)))
  z = (x, [2.0, -1.0])
  output = map(similar, z)
  @test value_and_prox!(output, s, z, γ) ≈ s(output) rtol = 1e-12
end

a_value(f, x) = f(x)
a_prox!(y, f, x, γ) = prox!(y, f, x, γ)
a_vprox!(y, f, x, γ) = value_and_prox!(y, f, x, γ)

@testset "DistL2 zero allocations and inference" begin
  for T in (Float32, Float64), C in (IndBox(-0.5, 1.2), IndBallL2(1), IndBallL0(3)), λ in (T(0), T(1.7))
    x = T.(range(-3.1, 2.7; length = 12))
    f = DistL2(C, λ)
    @test allocs(a_value, f, x) == 0
    @inferred f(x)
    for γ in (T(0.6), T(20)), op in (a_prox!, a_vprox!), alias in (false, true)
      z = copy(x)
      y = alias ? z : similar(z)
      @test allocs(op, y, f, z, γ) == 0
    end
    @inferred prox!(similar(x), f, x, T(0.6))
    @inferred value_and_prox!(similar(x), f, x, T(0.6))
  end
  for T in (Float32, Float64)
    c = T(0.6) * floatmax(T)
    tiny = T === Float32 ? T(1e-30) : T(1e-300)
    for (x, p, λ, γ) in ((T[-c], c, inv(c), c), (T[0], c, one(T), tiny),
                         (T[0], c, tiny, tiny), (T[-c], c, T(2), c),
                         (T[0.75floatmax(T), 0.75floatmax(T)], zero(T), T(2), T(0.52) * floatmax(T)))
      f = DistL2(IndBox(p, p), λ)
      @test allocs(a_value, f, x) == 0
      for op in (a_prox!, a_vprox!), alias in (false, true)
        z = copy(x)
        y = alias ? z : similar(z)
        @test allocs(op, y, f, z, γ) == 0
      end
    end
  end
end

end
