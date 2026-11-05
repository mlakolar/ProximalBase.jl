module PrecomposeArrayTest

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using Random

using ..TestUtils

const PB = ProximalBase
const PT = PB.Testing
Random.seed!(181)

mutable struct ArraySquare <: SeparableFunction
  w::Vector{Float64}
  resets::Int
end
ProximalBase.entry_parameters(f::ArraySquare) = (:w => f.w,)
PB.value_entry(f::ArraySquare, v::Real, i::Integer) = f.w[i] * v^2 / 2
PB.prox_entry(f::ArraySquare, v::Real, i::Integer, γ::Real) = v / (1 + γ * f.w[i])
PB.isconvex(::Type{ArraySquare}) = true
PB.iszeropreserving(::ArraySquare) = true
PB.check_parameters(f::ArraySquare, x) = (PB._checkparam(f.w, x); nothing)
PB.restrict(f::ArraySquare, S::AbstractVector{<:Integer}) = ArraySquare(f.w[S], 0)
PB.reset_cache!(f::ArraySquare) = (f.resets += 1; f)

@testset "constant scales and separable oracles" begin
  x = [-2.0, -0.3, 0.0, 0.7, 1.2, 3.0]
  for f in (NormL1(0.3), MCP(0.4), SCAD(0.4), SqrNormL2(0.7),
            MoreauEnvelope(NormL1(0.3), 0.8)), c in (2.0, -0.5)
    h, scalar = Precompose(f, fill(c, size(x))), Precompose(f, c)
    @test !(h isa Precompose)
    @test h(x) ≈ scalar(x) rtol = 1e-14
    @test prox(h, x, 0.2) ≈ prox(scalar, x, 0.2) rtol = 1e-14
    @test [prox_entry(h, x[i], i, 0.2) for i in eachindex(x)] ≈ prox(scalar, x, 0.2) rtol = 1e-14
    PT.test_traits(h; smooth = issmooth(f), proximable = true, separable = true, convex = isconvex(f))
    @test !hashessian(h)
    @test iszeropreserving(h) == iszeropreserving(f)
    PT.test_separable(h, x, 0.2; groups = [1:2, 3:6])
    PT.test_prox(h, x, 0.2)
    PT.test_kkt(h, x, 0.1)                  # γc² is below the nonconvex limits
    if issmooth(f)
      @test gradient(h, x) ≈ gradient(scalar, x) rtol = 1e-14
      PT.test_gradient(h, x)
    end
    if f isa SeparableFunction
      @test sum(i -> value_entry(h, x[i], i), eachindex(x)) ≈ h(x)
    else
      @test !applicable(value_entry, h, x[1], 1)
    end
  end
end

@testset "MCP identity and scalar references" begin
  λ, b = 0.6, 3.0
  s, x = [0.25, 0.8, 1.5, 3.0], [-2.0, -0.3, 0.7, 1.2]
  h = Precompose(MCP(λ, b), s)
  for i in eachindex(x)
    f = MCP(s[i] * λ, b / s[i]^2)
    @test value_entry(h, x[i], i) ≈ value_entry(f, x[i], 1) rtol = 1e-12
    for γ in (0.1, 2.0)
      @test prox_entry(h, x[i], i, γ) ≈ prox_entry(f, x[i], 1, γ) rtol = 1e-12
    end
  end
  a, x = [0.5, -1.0, 2.0, -0.75], [-2.0, -0.35, 0.4, 1.7]
  for f in (SCAD(0.6), CheckLoss(0.3), NormL1(0.4, [1.0, 0.5, 2.0, 0.7])), γ in (0.2, 4.0)
    h = Precompose(f, a)
    p = prox(h, x, γ)
    for i in eachindex(x)
      φ = t -> value_entry(f, a[i] * t, i)
      q, objective = brute_force_prox(φ, x[i], γ; lo = -8.0, hi = 8.0, n = 20_001)
      @test φ(p[i]) + (p[i] - x[i])^2 / (2γ) ≈ objective atol = 1e-8 rtol = 1e-12
      isconvex(f) && @test p[i] ≈ q atol = 1e-8 rtol = 1e-8
    end
  end
  # At a nonconvex tie, either minimizer has the correct objective.
  h = Precompose(SCAD(1.0), [-2.0])
  x, γ = sqrt(47.0) / 2, 2.5
  φ = t -> value_entry(h, t, 1)
  _, objective = brute_force_prox(φ, x, γ; lo = -5.0, hi = 5.0, n = 20_001)
  p = prox_entry(h, x, 1, γ)
  @test φ(p) + (p - x)^2 / (2γ) ≈ objective atol = 1e-8

  a, x, γ = [0.5, -1.0, 2.0, -0.75], [-2.0, -0.35, 0.4, 1.7], 0.4
  δ, λ = 0.7, 0.8
  huber = Precompose(MoreauEnvelope(NormL1(1.0), δ), a)
  square = Precompose(SqrNormL2(λ), a)
  for (h, φ) in ((huber, t -> abs(t) <= δ ? t^2 / (2δ) : abs(t) - δ / 2),
                 (square, t -> λ * t^2 / 2))
    p = prox(h, x, γ)
    @test h(x) ≈ sum(φ.(a .* x))
    for i in eachindex(x)
      q, objective = brute_force_prox(t -> φ(a[i] * t), x[i], γ; lo = -8.0, hi = 8.0, n = 20_001)
      @test φ(a[i] * p[i]) + (p[i] - x[i])^2 / (2γ) ≈ objective atol = 1e-8
      @test p[i] ≈ q atol = 1e-8 rtol = 1e-8
    end
    PT.test_gradient(h, x)
  end
  l, u = [-1.0, 0.2, -2.0, -0.3], [0.5, 2.0, 1.0, 1.2]
  h = Precompose(IndBox(l, u), a)
  expected = clamp.(x, min.(l ./ a, u ./ a), max.(l ./ a, u ./ a))
  @test prox(h, x, γ) ≈ expected atol = 1e-14
  @test value_and_prox!(similar(x), h, x, γ) == 0
  @test PB._isindicator(h)
  @test !iszeropreserving(h)
end

@testset "custom functions, wrappers, restriction and diagonal steps" begin
  a, x = [0.5, -1.0, 2.0, -0.75], [-2.0, 0.0, 0.4, 1.7]
  custom = ArraySquare([0.5, 1.0, 2.0, 0.3], 0)
  for f in (custom, SqrNormL2(0.7, [0.1, -0.2, 0.0, 0.5]),
            LinearFunction([0.2, -0.3, 0.5, -0.4], 1.0),
            Translate(Scaled(NormL1(0.4), 0.5), [0.1, 0.0, -0.2, 0.3]),
            MoreauEnvelope(Regularize(NormL1(0.4), 0.3), 0.8))
    h = Precompose(f, a)
    steps = [0.1, 0.4, 0.7, 0.2]
    p = [prox_entry(h, x[i], i, steps[i]) for i in eachindex(x)]
    @test prox(h, x, steps) ≈ p
    y = similar(x)
    @test value_and_prox!(y, h, x, steps) ≈ h(p)
    @test y ≈ p
    alias = copy(x)
    @test prox!(alias, h, alias, steps) ≈ p
    PT.test_separable(h, x, 0.3)
    if f isa SeparableFunction
      @test sum(i -> value_entry(h, x[i], i), eachindex(x)) ≈ h(x)
    end
    S = [1, 3]
    restricted = restrict(h, S)
    full = zeros(4); full[S] .= x[S]
    @test prox(restricted, x[S], 0.3) ≈ prox(h, full, 0.3)[S]
    @test restricted(x[S]) - restrict(f, S)(a[S] .* x[S]) ≈ 0 atol = 1e-14
    if issmooth(f)
      PT.test_gradient(h, x)
      @test gradient(restricted, x[S]) ≈ gradient(h, full)[S]
    end
  end
  h = Precompose(custom, a)
  @test h.a === a
  @test reset_cache!(h) === h
  @test custom.resets == 1

  f = SqrNormL2(0.7)
  h = Precompose(f, reshape(a, 2, 2))
  L = reshape(0.7 .* a.^2, 2, 2)
  @test coordinate_lipschitz!(zeros(2, 2), h) ≈ L
  @test [coordinate_lipschitz(h, k) for k in 1:4] ≈ vec(L)
  @test lipschitz(h) ≈ maximum(L)
  @test_throws BoundsError coordinate_lipschitz(h, 5)
  @test_throws DimensionMismatch coordinate_lipschitz!(zeros(4), h)
end

@testset "sparse and symmetric iterates" begin
  x, a = [0.0, -2.0, 0.0, 0.7, 0.0, 1.2], [0.5, -1.0, 2.0, -0.75, 1.5, 0.3]
  for f in (NormL1(0.3), MCP(0.4), SqrNormL2(0.7), MoreauEnvelope(NormL1(0.3), 0.8),
            IndBox(0.1, 1.0))
    h = Precompose(f, a)
    xs, ys = SparseIterate(x), SparseIterate(ones(6))
    p = prox(h, x, 0.2)
    @test Vector(prox!(ys, h, xs, 0.2)) ≈ p
    @test value_and_prox!(ys, h, xs, 0.2) ≈ h(p)
    @test h(xs) ≈ h(x)
    alias = copy(xs)
    @test value_and_prox!(alias, h, alias, 0.2) ≈ h(p)
    @test Vector(alias) ≈ p
    steps = fill(0.2, 6)
    @test Vector(prox(h, xs, steps)) ≈ p
  end
  h = Precompose(NormL1(0.3), a)
  xs, ys = SparseIterate(x), SparseIterate(zeros(6))
  prox!(ys, h, xs, 0.2)
  @test h.image.buf === nothing             # no dense image on the sparse proximal path
  @test nnz(ys) == nnz(xs)
  @test allocs(prox!, ys, h, xs, 0.2) == 0
  @test allocs(prox_coordinate!, xs, h, 2, 0.2) == 0

  X = [0.0 1.2 0.0; 1.2 -0.5 0.0; 0.0 0.0 2.0]
  A = [0.5 -1.0 2.0; -1.0 0.75 -0.5; 2.0 -0.5 1.5]
  for f in (NormL1(0.3), MCP(0.4), IndBox(0.1, 1.0))
    h = Precompose(f, A)
    PT.test_separable(h, X, 0.2; symmetric = true)
    xs, ys = SymmetricSparseIterate(X), SymmetricSparseIterate(ones(3, 3))
    p = prox(h, X, 0.2)
    @test Matrix(prox!(ys, h, xs, 0.2)) ≈ p
    @test value_and_prox!(ys, h, xs, 0.2) ≈ h(p)
    @test h(xs) ≈ h(X)
    @test Matrix(prox(h, xs, fill(0.2, 3, 3))) ≈ p
  end
end

@testset "constant scales on groups" begin
  groups = [1:2, [4, 6], 7:8]
  a = [-2.0, -2.0, 0.3, 0.5, -0.7, 0.5, -1.5, -1.5, 0.6, -0.2]
  x = [3.0, -4.0, 7.0, -1.0, 8.0, 2.0, 1.2, -0.3, 9.0, 10.0]
  γ = 0.4
  for F in (GroupNormL2, GroupNormLinf, GroupMCP, GroupSCAD)
    f = F(0.5, groups, [1.0, 0.7, 0.0])
    h = Precompose(f, a)
    steps = γ .* [a[first(G)]^2 for G in groups]
    expected = prox(f, a .* x, steps) ./ a
    @test h(x) ≈ f(a .* x)
    @test prox(h, x, γ) ≈ expected
    @test prox(h, x, γ)[[3, 5, 7, 8, 9, 10]] == x[[3, 5, 7, 8, 9, 10]]
    y = similar(x)
    @test value_and_prox!(y, h, x, γ) ≈ f(a .* expected)
    @test y ≈ expected
    alias = copy(x)
    @test prox!(alias, h, alias, γ) ≈ expected
    PT.test_traits(h; proximable = true, convex = isconvex(f))
    PT.test_prox(h, x, γ)
    g = collect(range(-1.0, 1.0; length = 10))
    for point in (zeros(10), x), i in eachindex(x)
      expected_kkt = abs(a[i]) * kkt_violation(f, g ./ a, a .* point, i)
      @test kkt_violation(h, g, point, i) ≈ expected_kkt
    end
    @test kkt_violation(h, g, zeros(10), 3) ≈ abs(g[3])
    @test_throws ArgumentError restrict(h, [1, 2])
    @test_throws ArgumentError prox_entry(h, 1.0, 1, γ)
    @test_throws ArgumentError kkt_violation(h, g, 1)
    @test_throws ArgumentError prox!(y, h, x, fill(γ, 10))
    @test_throws ArgumentError Precompose(f, [1.0; a[2:end]])
    @test allocs(prox!, y, h, x, γ) == 0
    @test allocs(value_and_prox!, y, h, x, γ) == 0
    @test allocs(kkt_violation, h, g, zeros(10), 1) == 0
    @inferred value_and_prox!(y, h, x, γ)
  end
  f = GroupNormL2(0.5, [1:2, 3:4])
  h = Precompose(f, [-2.0, -2.0, 0.5, 0.5])
  x = AtomIterate([1.0, -2.0, 0.0, 0.0], f)
  before = Vector(x)
  @test_throws ArgumentError prox_coordinate!(x, h, 1, 0.4)
  @test Vector(x) == before
  @test kkt_violation(h, ones(4), x, 1) ≈ 2 * kkt_violation(f, ones(4) ./ h.a, x, 1)
  PT.test_kkt(h, x, 0.2)
  @test allocs(kkt_violation, h, ones(4), x, 1) == 0
  @test Vector(prox(h, x, 0.4)) ≈ prox(h, before, 0.4)
  free = Precompose(GroupNormL2(0.5, [1:2]), [2.0, 2.0, 1e-300])
  @test kkt_violation(free, ones(3), zeros(3), 3) == 1.0
end

@testset "validation before writes" begin
  for a in ([0.0, 1.0], [NaN], [Inf], [-Inf])
    @test_throws ArgumentError Precompose(NormL1(), a)
  end
  @test_throws ArgumentError("Precompose requires real entries") Precompose(NormL1(), ComplexF64[1, 2])
  @test_throws ArgumentError Precompose(NormL2(), ones(2))
  x = reshape([1.0, -2.0, 3.0, -4.0], 2, 2)
  for h in (Precompose(NormL1(), ones(4)),
            Precompose(NormL1(1.0, ones(3)), ones(2, 2)),
            Precompose(SqrNormL2(1.0, ones(4)), ones(2, 2)),
            Precompose(Translate(NormL1(), ones(4)), ones(2, 2)))
    for op in (prox!, value_and_prox!, gradient!, value_and_gradient!), alias in (false, true)
      input = copy(x)
      y = alias ? input : fill(123.0, 2, 2)
      before = copy(y)
      if op === prox! || op === value_and_prox!
        @test_throws DimensionMismatch op(y, h, input, 0.4)
      elseif alias
        @test_throws ArgumentError op(y, h, input)
      else
        @test_throws DimensionMismatch op(y, h, input)
      end
      @test y == before
      @test input == x
    end
  end
  h = Precompose(NormL1(), ones(2, 2))
  @test_throws DimensionMismatch PB.check_parameters(h, ones(4))
  for op in (prox!, value_and_prox!)
    y = copy(x)
    @test_throws DimensionMismatch op(y, h, y, ones(3))
    @test y == x
  end
  xs = SparseIterate(x)
  bad = Precompose(NormL1(), ones(4))
  @test_throws DimensionMismatch prox_coordinate!(xs, bad, 1, 0.4)
  @test Matrix(xs) == x
end

@testset "precision, empty arrays and extreme scales" begin
  x, a = Float32[2, -0.3, 0.7, -1.2], Float32[0.5, -2, 1.5, -0.7]
  for f in (NormL1(0.3f0), MCP(0.4f0), SCAD(0.4f0), SqrNormL2(0.7f0),
            MoreauEnvelope(NormL1(0.3f0), 0.8f0))
    h = Precompose(f, a)
    p = prox(h, x, 0.3f0)
    @test eltype(p) == Float32
    @test p ≈ [prox_entry(h, x[i], i, 0.3f0) for i in eachindex(x)] rtol = 5e-7
    @test value_and_prox!(similar(x), h, x, 0.3f0) ≈ h(p) rtol = 5e-7
    @test Vector(prox(h, SparseIterate(x), 0.3f0)) ≈ p rtol = 5e-7
    @test allocs(prox!, similar(x), h, x, 0.3f0) == 0
    @test allocs(value_and_prox!, similar(x), h, x, 0.3f0) == 0
    if issmooth(f)
      @test eltype(gradient(h, x)) == Float32
    end
  end
  f = GroupNormL2(0.5f0, [1:2, 3:4])
  h = Precompose(f, Float32[-2, -2, 0.5, 0.5])
  @test eltype(prox(h, x, 0.3f0)) == Float32
  @test prox(h, x, 0.3f0) ≈ prox(f, h.a .* x, Float32[1.2, 0.075]) ./ h.a rtol = 5e-7
  for (X, A, Y, G) in ((Float32, Float32, Float64, Float64),
                       (Float64, Float32, Float32, Float64),
                       (Float32, Float64, Float32, Float32),
                       (Int, Int, Float64, Float64))
    x, a = X[2, -3], A[-2, 1]
    h = Precompose(NormL1(0.3), a)
    T, R = float(promote_type(X, A)), float(X)
    image = T.(a) .* T.(x)
    @test h(x) isa R
    @test h(x) == R(h.f(image))
    y, γ = zeros(Y, 2), G(0.4)
    expected = shrink.(Float64.(x), Float64(γ) .* 0.3 .* abs.(a))
    @test prox!(y, h, x, γ) === y
    @test y ≈ expected rtol = 5e-7
    @test eltype(prox(h, x, γ)) == float(X)
    v = value_and_prox!(y, h, x, γ)
    @test v isa Y
    @test v isa typeof(h(y))
    smooth = Precompose(SqrNormL2(0.7), a)
    @test gradient!(y, smooth, x) === y
    @test y ≈ 0.7 .* Float64.(a).^2 .* x rtol = 5e-7
    @test value_and_gradient!(y, smooth, x) isa R
  end
  h = Precompose(NormL1(), Float64[])
  @test h(Float64[]) == 0.0
  @test isempty(prox(h, Float64[], 0.4))
  @test value_and_prox!(Float64[], h, Float64[], Float64[]) == 0.0
  @test lipschitz(Precompose(SqrNormL2(), Float64[])) == 0.0
  @test isempty(coordinate_lipschitz!(Float64[], Precompose(SqrNormL2(), Float64[])))
  large = Precompose(SqrNormL2(2.0^-1000), [2.0^600])
  @test lipschitz(large) == 2.0^200
  @test coordinate_lipschitz(large, 1) == 2.0^200
  @test Precompose(NormL1(), [typemax(Int)])([2]) == Float64(typemax(Int)) * 2
  for (a, γ, x) in ((2.0^600, 2.0^-600, 2.0), (2.0^-600, 2.0^600, 1.0),
                    (1e150, 1e-150, 2.0), (-1e-150, 1e150, 2.0),
                    (1e-150, 1.0, 0.0), (1e150, 1.0, 0.0))
    h = Precompose(NormL1(), [a])
    reference = Float64(sign(BigFloat(x)) * max(abs(BigFloat(x)) - BigFloat(γ) * abs(BigFloat(a)), BigFloat(0)))
    @test prox_entry(h, x, 1, γ) ≈ reference rtol = 1e-14 atol = 0
    @test prox(h, [x], γ)[1] ≈ reference rtol = 1e-14 atol = 0
    @test prox(h, [x], [γ])[1] ≈ reference rtol = 1e-14 atol = 0
  end
  @test prox(Precompose(NormL1(), [2.0^600]), [2.0], 2.0^-600) == [1.0]
  @test prox(Precompose(NormL1(), [2.0^-600]), [1.0], 2.0^600) == [0.0]
  tiny = ldexp(1.0, -1070)
  h = Precompose(NormL1(0.25), [tiny])
  @test h([1.0]) == tiny / 4
  @test prox(h, [0.0], 1.0) == [0.0]
  @test kkt_violation(h, [tiny], 1) == tiny * 0.75
  for a in (2.0, -0.5)
    h = Precompose(NormL1(), [a, a, a])
    p = prox(h, [NaN, Inf, -Inf], 0.4)
    @test isnan(p[1])
    @test p[2:3] == [Inf, -Inf]
    @test isnan(h([NaN, Inf, -Inf]))
  end
end

@testset "inner functions without a standalone value" begin
  # Conjugate(ElasticNet(λ1, λ2))(v) = Σ max(|vᵢ| - λ1, 0)² / (2λ2), from the proximal point
  λ1, λ2 = 1.0, 0.5
  en(v) = sum(t -> max(abs(t) - λ1, 0)^2, v) / (2λ2)
  f = Conjugate(ElasticNet(λ1, λ2))
  @test !PB._has_value(f)
  a, x = [2.0, -0.5, 1.5, 3.0], [1.0, 2.0, -0.5, 0.4]
  h = Precompose(f, a)
  for γ in (0.5, [0.5, 0.2, 1.0, 0.3])
    p = prox(h, x, γ)
    y = fill(NaN, 4)
    @test value_and_prox!(y, h, x, γ) ≈ en(a .* p) rtol = 1e-14
    @test y == p
    z = copy(x)
    @test value_and_prox!(z, h, z, γ) ≈ en(a .* p) rtol = 1e-14
    @test z == p
    ys = SparseIterate(4)
    @test value_and_prox!(ys, h, SparseIterate(x), γ) ≈ en(a .* p) rtol = 1e-14
    @test Vector(ys) == p
  end
  # a constant scale agrees with the scalar scale, which already gave the value
  @test value_and_prox!(similar(x), Precompose(f, fill(2.0, 4)), x, 0.5) ≈
        value_and_prox!(similar(x), Precompose(f, 2.0), x, 0.5) rtol = 1e-14
  # Float32 keeps its precision
  h32 = Precompose(Conjugate(ElasticNet(1f0, 0.5f0)), Float32[2, -0.5, 1.5])
  y32 = similar(Float32[1, 2, -0.5])
  @test value_and_prox!(y32, h32, Float32[1, 2, -0.5], 0.5f0) isa Float32
  @test y32 ≈ Float32[0.6, 2, -0.5]
  # a symmetric iterate reads the symmetric part of the argument
  A, X = [2.0 1.0; 1.0 3.0], [1.0 5.0; 2.0 -0.5]
  hS = Precompose(f, A)
  P = prox(hS, (X + X') / 2, 0.5)
  Y = SymmetricSparseIterate(2)
  @test value_and_prox!(Y, hS, X, 0.5) ≈ en(A .* P) rtol = 1e-14
  @test Matrix(Y) ≈ P rtol = 1e-14
  # Symmetric(X, :L) selects the lower triangle
  PL = prox(hS, Symmetric(X, :L) |> Matrix, 0.5)
  @test value_and_prox!(Y, hS, Symmetric(X, :L), 0.5) ≈ en(A .* PL) rtol = 1e-14
  @test Matrix(Y) ≈ PL rtol = 1e-14
  # errors of the inner value still come before the output is written
  y = fill(7.0, 4)
  @test_throws DimensionMismatch value_and_prox!(y, h, x, [0.5, 0.2])
  @test y == fill(7.0, 4)
end

@testset "zero allocations after warming" begin
  x, a, steps = [2.0, -0.3, 0.7, -1.2], [0.5, -2.0, 1.5, -0.7], fill(0.3, 4)
  y = similar(x)
  for f in (NormL1(0.3), MCP(0.4), SqrNormL2(0.7), MoreauEnvelope(NormL1(0.3), 0.8))
    h = Precompose(f, a)
    @test allocs(h, x) == 0
    @test allocs(prox!, y, h, x, 0.3) == 0
    @test allocs(value_and_prox!, y, h, x, 0.3) == 0
    @test allocs(prox!, y, h, x, steps) == 0
    @test allocs(value_and_prox!, y, h, x, steps) == 0
    @test allocs(prox_entry, h, x[1], 1, 0.3) == 0
    @test allocs(kkt_violation, h, x, 1) == 0
    @test allocs(prox_coordinate!, SparseIterate(x), h, 1, 0.3) == 0
    @inferred prox_entry(h, x[1], 1, 0.3)
    @inferred value_and_prox!(y, h, x, 0.3)
    if issmooth(f)
      @test allocs(gradient!, y, h, x) == 0
      @test allocs(value_and_gradient!, y, h, x) == 0
    end
  end
  h = Precompose(SqrNormL2(0.7), a)
  @test allocs(coordinate_lipschitz, h, 2) == 0
  @test allocs(coordinate_lipschitz!, y, h) == 0
end

end
