module ConstrainedBasicsTest

# Constrained on small inputs: a separable function on a box and on a hyperplane or a
# halfspace, against closed forms and the optimality conditions, coordinate steps and errors.
# The function x constraint grids, the multiplier searches and the stress cases are in constrained.jl.

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using Random

using ..TestUtils

const PB = ProximalBase

Random.seed!(1)

# the projection onto the capped simplex {0 ≤ x ≤ u, Σᵢ xᵢ = k}, by bisection on
# the level τ of clamp.(x .- τ, 0, u)
function capped_simplex_proj(x, k, u = 1.0)
  lo, hi = minimum(x) - u, maximum(x)
  for _ in 1:200
    τ = (lo + hi) / 2
    sum(clamp.(x .- τ, 0, u)) > k ? (lo = τ) : (hi = τ)
  end
  clamp.(x .- (lo + hi) / 2, 0, u)
end

@testset "a separable function on a box" begin
  n = 8
  x = 2 .* randn(n)
  w = rand(n)
  l, u = -rand(n), rand(n)
  fs = (NormL1(0.3), SqrNormL2(1., randn(n)))
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

  # against the minimizer of the scalar problems over a grid of the interval
  c = Constrained(NormL1(0.3, w), IndBox(l, u))
  p = prox(c, x, 0.9)
  for i in 1:n
    t, = brute_force_prox(s -> l[i] <= s <= u[i] ? 0.3 * w[i] * abs(s) : Inf, x[i], 0.9; lo = l[i], hi = u[i])
    @test p[i] ≈ t atol = 1e-6
  end

  c = Constrained(NormL1(0.3), IndBox(-0.5, 1.))
  @test c([0.2, -0.5, 1.]) ≈ 0.3 * 1.7
  @test c([0.2, -0.6, 1.]) == Inf
  @test Constrained(IndBox(0, 2), IndBox(1, 3))([1.5, 1.]) == 0
  @test value_and_prox!(similar(x), Constrained(IndBox(0, 2), IndBox(1, 3)), x, 1.) == 0
  @test prox(Constrained(IndBox(0, 2), IndBox(1, 3)), x) ≈ clamp.(x, 1, 2)

  # a box under a box, and a scaled penalty
  @test prox(Constrained(Constrained(NormL1(0.3), IndBox(-1, 1)), IndBox(0, 2)), x, 0.5) ≈ clamp.(shrink.(x, 0.15), 0, 1)
  @test prox(Constrained(0.5 * NormL1(0.6), IndNonnegative()), x, 0.5) ≈ max.(shrink.(x, 0.15), 0)
end

@testset "coordinate steps on a box" begin
  c = Constrained(NormL1(0.3), IndBox(0, 1))
  v = [1.5, -0.2, 0., 0.4, 0., 2.]
  xs = SparseIterate(sparse(v))
  for k in 1:6
    @test prox_coordinate!(xs, c, k, 0.5) == clamp(shrink(v[k], 0.15), 0, 1)
    @test xs[k] == clamp(shrink(v[k], 0.15), 0, 1)
  end
  S = SymmetricSparseIterate(4)
  S[2, 1] = 0.7
  @test prox_coordinate!(S, c, tril_index(4, 2, 1), 1.) ≈ 0.4
  @test S[1, 2] ≈ 0.4
  xa = AtomIterate([1.5, -0.2, 0.3, 2.], [1:2, 3:4])
  @test prox_coordinate!(xa, c, 2, 1.) ≈ [0., 1.]
  @test isactive(xa, 2)

  @test iszeropreserving(Constrained(NormL1(1.), IndBox(-1, 1)))
  @test iszeropreserving(Constrained(Scaled(NormL1(1.), 2.), IndNonnegative()))
  @test iszeropreserving(Constrained(SqrNormL2(1.), IndNonnegative()))
  @test !iszeropreserving(Constrained(NormL1(1.), IndBox(0.5, 1)))
  @test !iszeropreserving(Constrained(LinearFunction(ones(3)), IndBox(-1, 1)))
  @test !iszeropreserving(Constrained(SqrNormL2(1., ones(3)), IndBox(-1, 1)))
  # on a sparse iterate: the stored entries only, or all of them when zero moves
  xs = SparseIterate(sparse([0., 2., 0., -1.]))
  ys = prox(Constrained(NormL1(0.5), IndBox(-1, 1)), xs, 1.)
  @test ys isa SparseIterate && Array(ys) ≈ [0., 1., 0., -0.5]
  @test Array(prox(Constrained(NormL1(0.5), IndBox(0.2, 1)), xs, 1.)) ≈ [0.2, 1., 0.2, 0.2]
  @test Constrained(NormL1(0.5), IndBox(-1, 1))(ys) ≈ 0.75
  @test Constrained(NormL1(0.5), IndBox(-1, 1))(xs) == Inf             # xs[2] = 2 is outside

  # the direction that improves on zero must be open
  c = Constrained(NormL1(1.), IndBox(0, 2))
  g = [-3., 3., 0.5, -0.5]
  @test kkt_violation(c, g, 1) ≈ 2.
  @test kkt_violation(c, g, 2) <= 0
  @test kkt_violation(c, g, 3) <= 0 && kkt_violation(c, g, 4) <= 0
  @test kkt_violation(c, g, xs, 1) ≈ 2.
  # for a linear term, the direction is that of gᵢ + cᵢ
  @test kkt_violation(Constrained(LinearFunction([-2.]), IndBox(-1, 0)), [0.5], 1) <= 0
  @test kkt_violation(Constrained(LinearFunction([-2.]), IndBox(-1, 1)), [0.5], 1) ≈ 1.5

  w, l, u = rand(8), -rand(8), rand(8)
  c = Constrained(NormL1(0.3, w), IndBox(l, u))
  x = randn(8)
  S = [2, 5, 7]
  @test prox(restrict(c, S), x[S], 0.6) ≈ prox(c, x, 0.6)[S]
end

@testset "zero-preserving wrappers on a box" begin
  fs = (Scaled(NormL1(1.), 2.),)
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
  @test !iszeropreserving(Constrained(Scaled(NormL1(1.), Inf), IndBox(-1., 1.)))
end

@testset "a separable function on a hyperplane or a halfspace" begin
  n = 10
  x = randn(n)
  @testset "simplices" begin
    for r in (0.5, 1., 3.), γ in (0.3, 2.)
      Δ = prox(IndSimplex(r), x)
      @test prox(Constrained(IndNonnegative(), IndHyperplane(1, r)), x, γ) ≈ Δ
      @test prox(Constrained(ZeroFunction(), IndSimplex(r)), x, γ) ≈ Δ
      @test prox(Constrained(IndNonnegative(), IndHalfspace(1, r)), x, γ) ≈ prox(IndSimplex(r; equality = false), x)
    end
    for k in (1, 3.5, 7), u in (1., 0.8)
      @test prox(Constrained(IndBox(0, u), IndHyperplane(1, k)), 3 .* x) ≈ capped_simplex_proj(3 .* x, k, u)
    end
  end

  @testset "closed forms" begin
    a, b = randn(n), 0.7
    @test prox(Constrained(ZeroFunction(), IndHyperplane(a, b)), x, 0.5) ≈ prox(IndHyperplane(a, b), x)
    # ridge on a hyperplane: z = (x - γνa) / (1 + γλ)
    λ, γ = 2., 0.6
    ν = (dot(a, x) - b * (1 + γ * λ)) / (γ * dot(a, a))
    c = Constrained(SqrNormL2(λ), IndHyperplane(a, b))
    @test prox(c, x, γ) ≈ (x .- γ * ν .* a) ./ (1 + γ * λ)
    check_prox(c, x, γ)
    # an inactive halfspace leaves the proximal point of f
    @test prox(Constrained(NormL1(0.3), IndHalfspace(1, 100)), x, 0.5) == prox(NormL1(0.3), x, 0.5)
  end

  @testset "the lasso penalty on coefficients that sum to zero" begin
    λ, γ = 0.4, 0.8
    c = Constrained(NormL1(λ), IndHyperplane(1, 0))
    check_traits(c; proximable = true, convex = true)
    z = check_prox(c, x, γ)
    @test abs(sum(z)) <= 1e-12 * norm(x, 1)
    # (x - z)/γ - ν ∈ λ ∂‖z‖₁ for one multiplier ν
    nz = findall(!iszero, z)
    ν = (x[nz[1]] - z[nz[1]]) / γ - λ * sign(z[nz[1]])
    @test all(i -> abs((x[i] - z[i]) / γ - ν - λ * sign(z[i])) <= 1e-10, nz)
    @test all(i -> abs(x[i] / γ - ν) <= λ + 1e-10, findall(iszero, z))
    @test c([1., -1., 0.]) ≈ 0.8
    @test c([1., -0.5, 0.]) == Inf
    # on a matrix, the entries sum to zero
    Z = prox(c, randn(3, 4), γ)
    @test size(Z) == (3, 4) && abs(sum(Z)) <= 1e-12
  end

  @testset "general normals" begin
    a = randn(n)
    a[3] = 0.
    for f in (Constrained(NormL1(0.2), IndBox(-1, 1)),)
      for (H, b) in ((IndHyperplane, 0.3), (IndHalfspace, -0.4))
        c = Constrained(f, H(a, b))
        z = check_prox(c, 2 .* x, 0.9)
        H === IndHyperplane ? (@test dot(a, z) ≈ b) : (@test dot(a, z) <= b + 1e-12)
        @test z[3] ≈ prox_entry(f, 2x[3], 3, 0.9)           # aᵢ = 0: independent of ν
      end
    end
    # a scalar normal is an array of it
    for H in (IndHyperplane, IndHalfspace)
      @test prox(Constrained(NormL1(0.3), H(2., 1.5)), x, 0.7) ≈ prox(Constrained(NormL1(0.3), H(fill(2., n), 1.5)), x, 0.7)
    end
    # a box after the linear constraint goes under it
    c1 = Constrained(Constrained(NormL1(0.2), IndHyperplane(1, 1)), IndBox(-1, 1))
    c2 = Constrained(Constrained(NormL1(0.2), IndBox(-1, 1)), IndHyperplane(1, 1))
    @test typeof(c1) == typeof(c2) && prox(c1, x, 0.5) == prox(c2, x, 0.5)
    @test Constrained(Constrained(NormL1(0.2), IndHyperplane(1, 1)), IndNonnegative()) isa typeof(
      Constrained(Constrained(NormL1(0.2), IndNonnegative()), IndHyperplane(1, 1)))
  end

  @testset "errors" begin
    @test_throws ArgumentError prox(Constrained(IndBox(0, 1), IndHyperplane(1, n + 1)), x)
    @test_throws ArgumentError prox(Constrained(IndNonnegative(), IndHalfspace(1, -1)), x)
    @test_throws ArgumentError Constrained(MCP(1.), IndHyperplane(1, 0))
    @test_throws ArgumentError Constrained(NormL0(1.), IndSimplex(1.))
    @test_throws ArgumentError Constrained(NormL2(1.), IndBox(0, 1))
    @test_throws ArgumentError Constrained(SqrNormL2(1.), IndBallL2(1.))
    @test_throws ArgumentError Constrained(Constrained(NormL1(1.), IndHyperplane(1, 0)), IndHyperplane(1, 1))
    @test_throws DimensionMismatch prox(Constrained(NormL1(1.), IndHyperplane(ones(3), 0)), x)
  end

  @testset "Float32" begin
    x32 = Float32.(x)
    for c in (Constrained(NormL1(0.3), IndBox(0, 1)), Constrained(NormL1(0.3), IndHyperplane(1, 0)))
      z = prox(c, x32, 0.7f0)
      @test eltype(z) == Float32
      @test z ≈ prox(c, x, 0.7) atol = 1e-5
      @test isfinite(c(z))
    end
  end
end

@testset "boxes intersecting closed and open scalar domains" begin
  for T in (Float64,)
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

# The value of the conjugate of ElasticNet(λ1, λ2, w) at y, from the explicit formula in BigFloat: for
# λ2 > 0, Σᵢ max(|yᵢ| - λ1 wᵢ, 0)² / (2λ2); for λ2 = 0, the indicator of |yᵢ| ≤ λ1 wᵢ, within the
# relative tolerance sqrt(eps) that the package gives to the values of indicators.
function elastic_conjugate_reference(λ1, λ2, w, y)
  S = BigFloat
  level(i) = S(λ1) * (w === nothing ? one(S) : S(w[i]))
  if λ2 > 0
    sum((max(abs(S(y[i])) - level(i), 0)^2 for i in eachindex(y)); init = zero(S)) / (2 * S(λ2))
  else
    tol = sqrt(S(eps(float(eltype(y)))))
    all(i -> abs(S(y[i])) - level(i) <= tol * max(1, level(i)), eachindex(y)) ? zero(S) : S(Inf)
  end
end

function check_elastic_value(v, T, ref)
  r = T(ref)
  @test v isa T
  if isinf(r)
    @test v == r
  elseif iszero(r)
    @test v == 0
  else
    @test isapprox(v, r; rtol = 1000 * eps(T))
  end
end

# A vector with indices o+1:o+n, as OffsetArrays provides, defined here to add no dependency.
struct OffsetVec{T} <: AbstractVector{T}
  data::Vector{T}
  offset::Int
end
Base.size(a::OffsetVec) = size(a.data)
Base.axes(a::OffsetVec) = (Base.IdentityUnitRange((1:length(a.data)) .+ a.offset),)
Base.IndexStyle(::Type{<:OffsetVec}) = IndexLinear()
Base.getindex(a::OffsetVec, i::Int) = a.data[i - a.offset]
Base.setindex!(a::OffsetVec, v, i::Int) = (a.data[i - a.offset] = v)
Base.similar(a::OffsetVec, ::Type{T}, dims::Dims) where {T} = OffsetVec(similar(a.data, T, dims), a.offset)
# Broadcasting over an OffsetVec (as IndBox does with array bounds) allocates through a style that the fixture owns.
Base.BroadcastStyle(::Type{<:OffsetVec}) = Broadcast.ArrayStyle{OffsetVec}()
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{OffsetVec}}, ::Type{T}) where {T}
  ax = axes(bc)[1]
  OffsetVec(Vector{T}(undef, length(ax)), first(ax) - 1)
end

a_value_and_prox!(y, f, x, γ) = value_and_prox!(y, f, x, γ)
a_prox!(y, f, x, γ) = prox!(y, f, x, γ)

@testset "value_and_prox! of the conjugate of an elastic net on a box" begin
  f = Conjugate(ElasticNet(1.0, 0.5))
  x = [5.0, -5.0, 0.5]

  @testset "the value is taken at the stored output" begin
    # clamped at ±2: the value at [2, -2, 0.5] is (1² + 1² + 0) / (2 0.5) = 2, while the
    # unclamped proximal point [3, -3, 0.5] has the value 8
    c = Constrained(f, IndBox(-2.0, 2.0))
    y = zeros(3)
    @test value_and_prox!(y, c, x, 0.5) === 2.0
    @test y == [2.0, -2.0, 0.5]
    @test y == prox(c, x, 0.5)
    @test value_and_prox!(zeros(3), Constrained(f, IndBox(-10.0, 10.0)), x, 0.5) === 8.0
    # the original box: the stored output is inside the domain, where the value is zero
    y = zeros(3)
    @test value_and_prox!(y, Constrained(f, IndBox(-1.0, 1.0)), [1.0, 2.0, -0.5], 0.5) === 0.0
    @test y == [1.0, 1.0, -0.5]
    z = copy(x)
    @test value_and_prox!(z, c, z, 0.5) === 2.0
    @test z == [2.0, -2.0, 0.5]
    @test value_and_prox!(zeros(3), c, x, [0.5, 0.5, 0.5]) === 2.0
    @test value_and_prox!(zeros(3), Constrained(f, IndNonnegative()), x, 0.5) === 4.0
    @test value_and_prox!(Float64[], c, Float64[], 0.5) === 0.0
    empty = Constrained(Conjugate(ElasticNet(1.0, 0.5, Float64[])), IndBox(-2.0, 2.0))
    @test value_and_prox!(Float64[], empty, Float64[], 0.5) === 0.0
  end

  @testset "standalone values stay unavailable" begin
    c = Constrained(f, IndBox(-2.0, 2.0))
    @test_throws ArgumentError f([1.0, 1.0, 0.0])
    @test_throws ArgumentError c([1.0, 1.0, 0.0])
    @test !PB._has_value(c)
    @test !PB._has_value(f)
    @test !PB._has_conjugate_value(ElasticNet(1.0, 0.5))
  end

  @testset "against the explicit formula" begin
    onbound = Set{Int}()
    for T in (Float32, Float64)
      xin = T[5, -5, 0.5, 1.2, -0.05, 0, 3]
      n = length(xin)
      weights = (nothing, T[1, 0, 2, 0.5, 3, 1, 0])
      lo, hi = -T[2, 1, 0.5, 3, 0.1, 1, 2], T[2, 1, 0.5, 3, 0.2, 1, 2]
      boxes = ((IndBox(T(-10), T(10)), fill(T(-10), n), fill(T(10), n)),
               (IndBox(T(-1), T(1)), fill(T(-1), n), fill(T(1), n)),
               (IndBox(lo, hi), lo, hi),
               (IndNonnegative(), zeros(T, n), fill(T(Inf), n)),
               (IndBox(T(0.25), T(0.25)), fill(T(0.25), n), fill(T(0.25), n)))
      steps = (T(0.5), T[0.5, 0.25, 1, 2, 0.75, 0.5, 1])
      # λ1 and the weights give exact levels, so that the reference does not depend on a rounded level
      for w in weights, (λ1, λ2) in ((T(0.25), T(0.7)), (T(0), T(0.7)), (T(0.25), T(0)), (T(0), T(0)), (T(0.25), T(1e-6))),
          (box, l, u) in boxes, γ in steps, alias in (false, true)
        c = Constrained(Conjugate(ElasticNet(λ1, λ2, w)), box)
        expected = prox!(similar(xin), c, xin, γ)
        y = alias ? copy(xin) : similar(xin)
        v = a_value_and_prox!(y, c, alias ? y : xin, γ)
        @test y == expected
        alias || @test xin == T[5, -5, 0.5, 1.2, -0.05, 0, 3]
        check_elastic_value(v, T, elastic_conjugate_reference(λ1, λ2, w, y))
        λ2 > 0 && push!(onbound, count(i -> y[i] == l[i] || y[i] == u[i], 1:n))
      end
    end
    # no entry, some entries and every entry sit on a bound
    @test 0 in onbound && 7 in onbound && any(k -> 0 < k < 7, onbound)
  end

  @testset "levels, bounds and the domain of the indicator" begin
    for T in (Float32, Float64)
      w = T[1, 2, 4]
      level = T(0.5) .* w                                       # 0.5, 1, 2 exactly
      big = fill(T(1e3), 3)
      # the stored output is the bound: exactly at the level and one float on each side
      for (y0, λ2) in ((level, T(0.5)),
                       (T[level[1], nextfloat(level[2]), prevfloat(level[3])], T(0.5)),
                       (T[prevfloat(level[1]), level[2], nextfloat(level[3])], T(1)),
                       (-T[level[1], nextfloat(level[2]), prevfloat(level[3])], T(2)))
        c = Constrained(Conjugate(ElasticNet(T(0.5), λ2, w)), IndBox(y0, y0))
        y = zeros(T, 3)
        v = value_and_prox!(y, c, big, T(1))
        @test y == y0
        check_elastic_value(v, T, elastic_conjugate_reference(T(0.5), λ2, w, y))
      end
      # λ2 = 0: the indicator of |yᵢ| ≤ λ1 wᵢ, with the tolerance of the NormL1 conjugate
      indicator = Conjugate(NormL1(T(0.5), w))
      for (y0, expected) in ((level, T(0)),
                             (T[level[1], nextfloat(level[2]), prevfloat(level[3])], T(0)),
                             (-nextfloat.(level), T(0)),
                             (level .* T(1 + 1e-3), T(Inf)),
                             (T[level[1], level[2], 3], T(Inf)),
                             (T[0, 0, 0], T(0)))
        c = Constrained(Conjugate(ElasticNet(T(0.5), T(0), w)), IndBox(y0, y0))
        y = zeros(T, 3)
        v = value_and_prox!(y, c, big, T(1))
        @test y == y0
        @test v === expected
        @test v === indicator(y)
      end
      # the box misses the domain of the indicator: the output is clamped there, and its value is +Inf;
      # with λ2 > 0 the value at the clamped output is finite
      c = Constrained(Conjugate(ElasticNet(T(0.5), T(0), w)), IndBox(T(5), T(6)))
      y = zeros(T, 3)
      @test value_and_prox!(y, c, T[0, 1, -1], T(1)) === T(Inf)
      @test y == fill(T(5), 3)
      @test y == prox(c, T[0, 1, -1], T(1))
      c = Constrained(Conjugate(ElasticNet(T(0.5), T(0.5), w)), IndBox(T(5), T(6)))
      v = value_and_prox!(y, c, T[0, 1, -1], T(1))
      @test y == fill(T(5), 3)
      @test v ≈ 45.25
    end
  end

  @testset "dense, sparse and symmetric arguments" begin
    c = Constrained(Conjugate(ElasticNet(0.25, 0.7, [1.0, 2.0, 0.0, 0.5, 3.0])), IndBox(-1.0, 1.0))
    xs = SparseIterate(sparse([0.0, 3.0, 0.0, -2.0, 0.0]))
    for γ in (0.7, [0.7, 0.5, 1.0, 0.7, 0.9])
      yd = zeros(5)
      vd = value_and_prox!(yd, c, Vector(xs), γ)
      out = SparseIterate(5)
      v = value_and_prox!(out, c, xs, γ)
      @test v == vd && Vector(out) == yd
      @test !Base.has_offset_axes(out) && PB._elastic_value_route(c, out, xs, γ)
      @test isapprox(v, Float64(elastic_conjugate_reference(0.25, 0.7, [1.0, 2.0, 0.0, 0.5, 3.0], yd)); rtol = 1e-13)
      alias = copy(xs)
      @test value_and_prox!(alias, c, alias, γ) == vd
      @test Vector(alias) == yd
    end
    # a zero stays inside the box, and a box that excludes it moves every entry
    for box in (IndNonnegative(), IndBox(0.2, 1.0)), f in (ElasticNet(0.25, 0.7), ElasticNet(0.25, 0.0))
      c = Constrained(Conjugate(f), box)
      yd = zeros(5)
      vd = value_and_prox!(yd, c, Vector(xs), 0.7)
      out = SparseIterate(5)
      @test value_and_prox!(out, c, xs, 0.7) == vd
      @test Vector(out) == yd
      @test isapprox(vd, Float64(elastic_conjugate_reference(0.25, f.λ2, nothing, yd)); rtol = 1e-13)
    end
    # the symmetric matrix: every entry, off-diagonals included
    M = [2.0 0.5 -3.0; 0.5 -1.0 4.0; -3.0 4.0 0.1]
    W = [1.0 2.0 0.0; 2.0 0.5 1.0; 0.0 1.0 3.0]
    for w in (nothing, W), (λ1, λ2) in ((0.25, 0.7), (0.25, 0.0)), γ in (0.7, fill(0.7, 3, 3))
      c = Constrained(Conjugate(ElasticNet(λ1, λ2, w)), IndBox(-1.0, 1.0))
      Yd = zeros(3, 3)
      vd = value_and_prox!(Yd, c, M, γ)
      S = SymmetricSparseIterate(3)
      v = value_and_prox!(S, c, M, γ)
      @test v == vd && Matrix(S) == Yd
      @test !Base.has_offset_axes(S) && PB._elastic_value_route(c, S, M, γ)
      @test Yd == Yd'
      @test isapprox(v, Float64(elastic_conjugate_reference(λ1, λ2, w, Yd)); rtol = 1e-13)
      alias = SymmetricSparseIterate(M)
      @test value_and_prox!(alias, c, alias, γ) == vd
      @test Matrix(alias) == Yd
    end
  end

  @testset "destinations and weights that do not start at one are refused before writing" begin
    refused(f, args...) = try
      f(args...)
      false
    catch e
      e isa ArgumentError && occursin("is not implemented", e.msg)
    end
    for off in (-1, 4), λ2 in (0.5, 0.0)
      c = Constrained(Conjugate(ElasticNet(1.0, λ2)), IndBox(-2.0, 2.0))
      xo = OffsetVec([5.0, -5.0], off)
      @test !Base.has_offset_axes([1.0]) && Base.has_offset_axes(xo)
      # the proximal operator still works, as without the value
      @test prox!(OffsetVec(zeros(2), off), c, xo, 0.5).data == (λ2 > 0 ? [2.0, -2.0] : [1.0, -1.0])
      y = OffsetVec([-77.0, -77.0], off)
      @test refused(value_and_prox!, y, c, xo, 0.5)
      @test y.data == [-77.0, -77.0]
      @test_throws ArgumentError value_and_prox!(y, c, [5.0, -5.0], 0.5)
      @test y.data == [-77.0, -77.0]
      # in place
      z = OffsetVec([5.0, -5.0], off)
      @test refused(value_and_prox!, z, c, z, 0.5)
      @test z.data == [5.0, -5.0]
      # weights with one-based or with offset indices
      for w in ([1.0, 3.0], OffsetVec([1.0, 3.0], off))
        cw = Constrained(Conjugate(ElasticNet(1.0, λ2, w)), IndBox(-2.0, 2.0))
        y = OffsetVec([-77.0, -77.0], off)
        @test refused(value_and_prox!, y, cw, xo, 0.5)
        @test y.data == [-77.0, -77.0]
        w isa OffsetVec && @test refused(value_and_prox!, fill(-77.0, 2), cw, [5.0, -5.0], 0.5)
      end
      # a block after an earlier block
      s = SeparableSum(SqrNormL2(2.0), c)
      dest, point = ([7.0], y), ([3.0], xo)
      before = (copy(dest[1]), copy(y.data))
      @test refused(value_and_prox!, dest, s, point, 0.5)
      @test dest[1] == before[1] && y.data == before[2]
      @test point[1] == [3.0] && xo.data == [5.0, -5.0]
    end
    # an overflowing value: finite on a one-based destination, refused before writing on an offset one
    for T in (Float64, Float32)
      b = floatmax(T)
      m = T(0.8) * b
      box = IndBox(m, m)
      c = Constrained(Conjugate(ElasticNet(T(0), b)), box)
      y = zeros(T, 2)
      v = value_and_prox!(y, c, zeros(T, 2), T(1))
      @test y == [m, m]
      @test v isa T && isapprox(v, T(BigFloat(m)^2 / BigFloat(b)); rtol = 1000 * eps(T))
      for off in (-1, 4)
        yo = OffsetVec(fill(T(-7), 2), off)
        @test refused(value_and_prox!, yo, c, OffsetVec(zeros(T, 2), off), T(1))
        @test yo.data == fill(T(-7), 2)
      end
    end
  end

  @testset "blocks of a separable sum" begin
    c = Constrained(f, IndBox(-2.0, 2.0))
    s = SeparableSum(SqrNormL2(2.0), c)
    point = ([3.0], x)
    dest = ([7.0], zeros(3))
    ref1, ref2 = zeros(1), zeros(3)
    expected = value_and_prox!(ref1, SqrNormL2(2.0), [3.0], 0.5) + value_and_prox!(ref2, c, x, 0.5)
    @test value_and_prox!(dest, s, point, 0.5) == expected
    @test dest == (ref1, ref2)
    @test point == ([3.0], [5.0, -5.0, 0.5])
  end

  @testset "allocations" begin
    x64 = [5.0, -5.0, 0.5]
    steps = (0.5, [0.5, 0.25, 1.0])
    for T in (Float32, Float64), γ in steps
      xt = T.(x64)
      step = γ isa Real ? T(γ) : T.(γ)
      for g in (ElasticNet(T(1), T(0.5)), ElasticNet(T(1), T(0.5), T[1, 0, 2]),
                ElasticNet(T(1), T(0)), ElasticNet(T(1), T(0), T[1, 0, 2]), ElasticNet(T(0), T(0.5)))
        for box in (IndBox(T(-2), T(2)), IndBox(fill(T(-2), 3), fill(T(2), 3)), IndNonnegative())
          c = Constrained(Conjugate(g), box)
          y = zeros(T, 3)
          @test allocs(value_and_prox!, y, c, xt, step) == 0
          z = copy(xt)
          @test allocs(value_and_prox!, z, c, z, step) == 0
          @test allocs(prox!, y, c, xt, step) == 0
        end
      end
    end
  end

  @testset "controls: values that were available" begin
    xc = [5.0, -5.0, 0.5, 1.2]
    for g in (NormL1(0.3), NormL1(0.3, [1.0, 0.0, 2.0, 0.5]), NormL1Nonneg(0.3), ElasticNet(0.3, 0.4),
              ElasticNet(0.3, 0.4, [1.0, 0.0, 2.0, 0.5]), SqrNormL2(0.7), SqrNormL2(0.7, [1.0, 0.0, 2.0, 0.5]),
              IndBox(-1.5, 1.5), Conjugate(NormL1(0.3)), Conjugate(SqrNormL2(0.7)), Conjugate(IndBox(-1.0, 1.0))),
        box in (IndBox(-1.0, 1.0), IndNonnegative()), γ in (0.6, [0.6, 0.3, 1.0, 0.8])
      c = Constrained(g, box)
      y = zeros(4)
      v = value_and_prox!(y, c, xc, γ)
      @test y == prox(c, xc, γ)
      @test v ≈ c(y)
      @test PB._has_value(c)
      @test allocs(value_and_prox!, y, c, xc, γ) == 0
    end
  end
end

# ----------------------------------------------------------------------------------------------------
# The allowlist of the fused value: the representations that take it, and those that keep the refusal.
# ----------------------------------------------------------------------------------------------------

# true when the call throws the refusal of a value that is not available
refused_value(f, args...) = try
  f(args...)
  false
catch e
  e isa ArgumentError && occursin("is not implemented", e.msg)
end
snap(a) = collect(a)
snap(a::AtomIterate) = copy(a.storage)
snap(t::Tuple) = map(snap, t)

function check_fused(c, mkdest, x, γ; alias = nothing)
  f = c.f.f
  T = eltype(mkdest())
  expected = prox!(mkdest(), c, x, γ)
  y = mkdest()
  v = value_and_prox!(y, c, x, γ)
  @test snap(y) == snap(expected)
  w = f.w === nothing ? nothing : collect(f.w)
  check_elastic_value(v, T, elastic_conjugate_reference(f.λ1, f.λ2, w, snap(y)))
  if alias !== nothing
    z = alias()
    v = value_and_prox!(z, c, z, γ)
    @test snap(z) == snap(expected)
    check_elastic_value(v, T, elastic_conjugate_reference(f.λ1, f.λ2, w, snap(z)))
  end
end

@testset "representations that take the fused value" begin
  for T in (Float32, Float64)
    vx = T[5, -5, 0.5, 1.2]
    mx = T[5 -5; 0.5 1.2]
    S3 = T[5 1 0; 1 -5 0; 0 0 0.5]
    N3 = T[5 2 0; 0 -5 0; 0 1 0.5]
    dense(mk, data) = () -> (y = mk(); copyto!(y, data); y)
    reps = (
      ("vector", () -> zeros(T, 4), vx, dense(() -> zeros(T, 4), vx), false),
      ("matrix", () -> zeros(T, 2, 2), mx, dense(() -> zeros(T, 2, 2), mx), false),
      ("range view", () -> view(zeros(T, 7), 2:5), vx, dense(() -> view(zeros(T, 7), 2:5), vx), false),
      ("strided view", () -> view(zeros(T, 8), 1:2:7), vx, dense(() -> view(zeros(T, 8), 1:2:7), vx), false),
      ("colon view", () -> view(zeros(T, 4, 2), :, 1), vx, dense(() -> view(zeros(T, 4, 2), :, 1), vx), false),
      ("scalar-index view", () -> view(zeros(T, 2, 4), 1, :), vx, dense(() -> view(zeros(T, 2, 4), 1, :), vx), false),
      ("view of a reshape", () -> view(reshape(zeros(T, 8), 2, 4), 1:2, 2:3), mx, dense(() -> view(reshape(zeros(T, 8), 2, 4), 1:2, 2:3), mx), false),
      ("reshape", () -> reshape(zeros(T, 4), 2, 2), mx, dense(() -> reshape(zeros(T, 4), 2, 2), mx), false),
      ("SparseIterate, Vector x", () -> SparseIterate(T, 4), vx, () -> SparseIterate(sparse(vx)), false),
      ("SparseIterate, SparseIterate x", () -> SparseIterate(T, 4), SparseIterate(sparse(vx)), () -> SparseIterate(sparse(vx)), false),
      ("SymmetricSparseIterate, symmetric x", () -> SymmetricSparseIterate(T, 3), S3, () -> SymmetricSparseIterate(S3), true),
      ("SymmetricSparseIterate, general x", () -> SymmetricSparseIterate(T, 3), N3, nothing, true),
      ("SymmetricSparseIterate, SymmetricSparseIterate x", () -> SymmetricSparseIterate(T, 3), SymmetricSparseIterate(S3), () -> SymmetricSparseIterate(S3), true),
      ("vector, Int x", () -> zeros(T, 4), [5, -5, 1, 2], nothing, false))
    # weights of the shape of the output, in each admitted element type and wrapper; symmetric for a symmetric output
    weight(kind, n, sym) = begin
      base = sym ? [1 2 0; 2 1 1; 0 1 3] : [1, 0, 2, 1, 3, 1, 0, 1, 2][1:n]
      kind === :f64 ? Float64.(base) : kind === :f32 ? Float32.(base) : Int.(base)
    end
    for (name, mkdest, x, mkalias, sym) in reps
      n = length(mkdest())
      sz = size(mkdest())
      @testset "$name, $T" begin
        for (λ1, λ2) in ((T(0.25), T(0.7)), (T(0.25), T(0))), kind in (:none, :f64, :f32, :int),
            box in (IndBox(T(-2), T(2)), IndBox(fill(T(-2), sz), fill(T(2), sz)), IndBox(fill(-2.0, sz), fill(2.0, sz)), IndNonnegative()),
            γ in (T(0.5), fill(T(0.5), sz))
          w = kind === :none ? nothing : weight(kind, n, sym)
          c = Constrained(Conjugate(ElasticNet(λ1, λ2, w)), box)
          check_fused(c, mkdest, x, γ; alias = mkalias)
        end
      end
    end

    @testset "parameter types on a vector, $T" begin
      xs = T[5, -5, 0.5, 1.2]
      mk = () -> zeros(T, 4)
      al = () -> copy(xs)
      wdense = [1.0, 0, 2, 0.5]
      levels = ((T(0.25), T(0.7)), (0.25, 0.7), (0.25f0, 0.7f0), (1, 1), (0.25, 0.0), (1, 0), (0.25f0, 0f0))
      weights = (nothing, wdense, Float32.(wdense), [1, 0, 2, 1], reshape(wdense, 2, 2), view([9.0, 1, 0, 2, 0.5, 9], 2:5),
                 reshape(wdense, 4, 1))
      for (λ1, λ2) in levels, w in weights
        check_fused(Constrained(Conjugate(ElasticNet(λ1, λ2, w)), IndBox(T(-2), T(2))), mk, xs, T(0.5); alias = al)
      end
      steps = (0.5, 0.5f0, 1, T(0.5), 1 // 2, [0.5, 0.25, 1, 2], Float32[0.5, 0.25, 1, 2], [1, 1, 2, 2], view([9.0, 0.5, 0.25, 1, 2, 9], 2:5))
      bounds = (IndBox(-2.0, 2.0), IndBox(-2, 2), IndBox(T(-2), T(2)), IndBox([-2.0, -1, -3, -2], [2.0, 1, 3, 2]),
                IndBox([-2, -1, -3, -2], [2, 1, 3, 2]), IndBox(Float32[-2, -1, -3, -2], Float32[2, 1, 3, 2]),
                IndBox(view([9.0, -2, -1, -3, -2, 9], 2:5), view([9.0, 2, 1, 3, 2, 9], 2:5)), IndNonnegative())
      for γ in steps, box in bounds, (λ1, λ2) in ((T(0.25), T(0.7)), (T(0.25), T(0)))
        check_fused(Constrained(Conjugate(ElasticNet(λ1, λ2)), box), mk, xs, γ; alias = al)
      end
    end

    @testset "ordinary, overflowing and underflowing values on every representation, $T" begin
      huge, tiny, λbig, λzero = T === Float32 ? (T(1e30), T(1e-10), T(1e30), T(1e30)) : (T(1e160), T(1e-100), T(1e100), T(1e200))
      for (name, mkdest, x, _, sym) in reps
        sz = size(mkdest())
        n = prod(sz)
        for (vals, λ1, λ2) in ((fill(huge, n), T(0), λbig), (fill(huge, n), T(0.5) * huge, λbig), (fill(huge, n), T(0), T(1)),
                               (fill(tiny, n), T(0), tiny), (fill(tiny, n), T(0), λzero), (sym ? T[min(i, j) for i in 1:sz[1], j in 1:sz[2]] : T.(1:n), T(0.5), T(0.25)))
          Y = reshape(vals, sz)
          c = Constrained(Conjugate(ElasticNet(λ1, λ2)), IndBox(Y, Y))
          check_fused(c, mkdest, x, T(1))
        end
      end
    end
  end
end

@testset "representations that keep the refusal" begin
  c0 = Constrained(Conjugate(ElasticNet(1.0, 0.5)), IndBox(-2.0, 2.0))
  x = [5.0, -5.0, 0.5]
  M3 = [5.0 1 0; 1 -5 0; 0 0 0.5]
  # destinations
  function refusal(c, mkdest, xdata, γ; mkalias = nothing)
    y = mkdest()
    before, xb = snap(y), snap(xdata)
    @test refused_value(value_and_prox!, y, c, xdata, γ)
    @test isequal(snap(y), before) && isequal(snap(xdata), xb)
    if γ isa Real
      s = SeparableSum(SqrNormL2(2.0), c)
      dest, point = ([7.0], mkdest()), ([3.0], xdata)
      db = snap(dest)
      @test refused_value(value_and_prox!, dest, s, point, γ)
      @test isequal(snap(dest), db) && isequal(snap(point), ([3.0], xb))
    end
    if mkalias !== nothing
      z = mkalias()
      zb = snap(z)
      @test refused_value(value_and_prox!, z, c, z, γ)
      @test isequal(snap(z), zb)
    end
  end
  for (name, mk, xdata, mkalias) in (("Int", () -> fill(-7, 3), x, () -> [5, -5, 1]),
                                     ("Rational", () -> fill(-7 // 1, 3), x, () -> [5 // 1, -5 // 1, 1 // 2]),
                                     ("Real", () -> Real[-7, -7, -7], x, () -> Real[5, -5, 1]),
                                     ("AbstractFloat", () -> AbstractFloat[-7f0, -7.0, -7f0], x, () -> AbstractFloat[5f0, -5.0, 1f0]),
                                     ("Union", () -> Union{Float32, Float64}[-7f0, -7.0, -7f0], x, () -> Union{Float32, Float64}[5f0, -5.0, 1f0]),
                                     ("BigFloat", () -> fill(big(-7.0), 3), x, () -> big.([5.0, -5.0, 0.5])),
                                     ("Bool", () -> fill(true, 3), x, () -> fill(true, 3)),
                                     ("Float16", () -> fill(Float16(-7), 3), x, () -> Float16[5, -5, 1]),
                                     ("offset", () -> OffsetVec(fill(-7.0, 3), -1), OffsetVec(copy(x), -1), () -> OffsetVec(copy(x), -1)))
    @testset "destination $name" begin
      refusal(c0, mk, xdata, 0.5; mkalias)
    end
  end
  # the proximal operator on an offset destination is as at the base
  @test prox!(OffsetVec(zeros(3), -1), c0, OffsetVec(copy(x), -1), 0.5).data == [2.0, -2.0, 0.5]
  @testset "destinations of other shapes" begin
    refusal(c0, () -> Symmetric(fill(-7.0, 3, 3)), M3, 0.5)
    refusal(c0, () -> transpose(fill(-7.0, 3, 3)), M3, 0.5)
    refusal(c0, () -> 1:3, x, 0.5)
    # a reshape of a view is no host array of the list (a reshape of an Array, and a view of either, are)
    refusal(c0, () -> reshape(view(fill(-7.0, 6), 2:5), 2, 2), [5.0 -5; 0.5 1], 0.5)
    refusal(Constrained(Conjugate(ElasticNet(1.0, 0.5, reshape(view([9.0, 1, 3, 1, 9], 2:4), 3, 1))), IndBox(-2.0, 2.0)),
            () -> fill(-7.0, 3), x, 0.5)
    refusal(c0, () -> AtomIterate(3, [1:3]), AtomIterate(copy(x), [1:3]), 0.5)
    refusal(c0, () -> sparsevec(fill(-7.0, 3)), sparsevec(x), 0.5)
    refusal(c0, () -> sparse(fill(-7.0, 2, 2)), sparse([5.0 -5; 0.5 1]), 0.5)
  end
  @testset "inputs" begin
    refusal(c0, () -> fill(-7.0, 3), OffsetVec(copy(x), -1), 0.5)
    refusal(c0, () -> fill(-7.0, 3), Real[5, -5.0, 1], 0.5)
    refusal(c0, () -> fill(-7.0, 3), Union{Float32, Float64}[5f0, -5.0, 1f0], 0.5)
    refusal(c0, () -> fill(-7.0, 3), 1:3, 0.5)
    refusal(c0, () -> SymmetricSparseIterate(3), Symmetric(M3), 0.5)
  end
  @testset "levels, weights and bounds" begin
    box = IndBox(-2.0, 2.0)
    for f in (ElasticNet(big(1.0), big(0.5)), ElasticNet(1 // 1, 1 // 2))
      refusal(Constrained(Conjugate(f), box), () -> fill(-7.0, 3), x, 0.5)
    end
    for w in (AbstractFloat[1f0, 3.0, 1.0], Real[1, 3.0, 1], Union{Float32, Float64}[1f0, 3.0, 1.0], OffsetVec([1.0, 3.0, 1.0], -1),
              [big(1.0), big(3.0), big(1.0)], [1 // 1, 3 // 1, 1 // 1])
      for λ2 in (0.5, 0.0)
        refusal(Constrained(Conjugate(ElasticNet(1.0, λ2, w)), box), () -> fill(-7.0, 3), x, 0.5;
                mkalias = () -> copy(x))
      end
    end
    # weights that share memory with the output, for both ridge levels
    for λ2 in (0.5, 0.0)
      yw = fill(0.75, 3)
      c = Constrained(Conjugate(ElasticNet(1.0, λ2, yw)), box)
      before = copy(yw)
      @test refused_value(value_and_prox!, yw, c, x, 0.5)
      @test yw == before
      vw = view(yw, 1:3)
      @test refused_value(value_and_prox!, vw, c, x, 0.5)
      @test yw == before
    end
    for (l, u) in ((OffsetVec([-2.0, -2.0, -2.0], -1), OffsetVec([2.0, 2.0, 2.0], -1)), (Real[-2, -2.0, -2], Real[2, 2.0, 2]))
      refusal(Constrained(Conjugate(ElasticNet(1.0, 0.5)), IndBox(l, u)), () -> fill(-7.0, 3), x, 0.5; mkalias = () -> copy(x))
    end
  end
  @testset "levels of an ElasticNet of an abstract type parameter" begin
    box = IndBox(-2.0, 2.0)
    # a stored level that is no machine number, in either position: refused, though the type parameter is Real
    for (l1, l2) in ((1.0, 1 // 2), (1.0, π), (1.0, big"0.5"), (1.0, big(0.5)), (1 // 1, 0.5), (π, 0.5), (big"1.0", 0.5))
      f = ElasticNet{Real, Nothing}(l1, l2, nothing)
      c = Constrained(Conjugate(f), box)
      refusal(c, () -> fill(-7.0, 3), x, 0.5; mkalias = () -> copy(x))
      # the proximal operator is not touched by the route: it runs for every one of these functions, as before
      y = zeros(3)
      @test prox!(y, c, x, 0.5) === y
      @test y == [2.0, -2.0, 0.5]
    end
    # machine-number levels are admitted, with the value checked at the stored output
    for T in (Float32, Float64), (l1, l2) in ((T(1), T(0.5)), (T(1), T(0)), (1.0, 0.5f0), (1f0, 0.5))
      check_fused(Constrained(Conjugate(ElasticNet(l1, l2)), IndBox(T(-2), T(2))), () -> zeros(T, 3), T.(x), T(0.5); alias = () -> T.(x))
    end
  end
  @testset "views whose selectors are not proven host ranges" begin
    # three entries chosen from a longer vector by each selector; every one of them is refused in every role
    selectors = (("index vector", v -> view(v, [4, 1, 2])),
                 ("repeated index", v -> view(v, [1, 1, 2])),
                 ("view of a view", v -> (vv = SubArray(view(v, 2:5), (1:3,)); @assert parent(vv) isa SubArray; vv)),
                 ("zero-step range", v -> view(v, StepRangeLen(2, 0, 3))))
    box = IndBox(-2.0, 2.0)
    c0 = Constrained(Conjugate(ElasticNet(1.0, 0.5)), box)
    ones6() = fill(1.0, 6)
    for (name, sel) in selectors
      @testset "$name" begin
        # as the output, the input, the bounds, the steps and the weights
        refusal(c0, () -> sel(fill(-7.0, 6)), x, 0.5; mkalias = () -> (y = sel(fill(-7.0, 6)); copyto!(y, x); y))
        refusal(c0, () -> fill(-7.0, 3), sel([5.0, -5.0, 0.5, 1.0, 2.0, 3.0]), 0.5)
        refusal(Constrained(Conjugate(ElasticNet(1.0, 0.5)), IndBox(sel(-2 .* ones6()), sel(2 .* ones6()))), () -> fill(-7.0, 3), x, 0.5;
                mkalias = () -> copy(x))
        refusal(Constrained(Conjugate(ElasticNet(1.0, 0.5)), IndBox(-2.0, sel(2 .* ones6()))), () -> fill(-7.0, 3), x, 0.5)
        refusal(c0, () -> fill(-7.0, 3), x, sel(0.5 .* ones6()); mkalias = () -> copy(x))
        refusal(Constrained(Conjugate(ElasticNet(1.0, 0.5, sel(ones6()))), box), () -> fill(-7.0, 3), x, 0.5; mkalias = () -> copy(x))
        refusal(Constrained(Conjugate(ElasticNet(1.0, 0.0, sel(ones6()))), box), () -> fill(-7.0, 3), x, 0.5)
      end
    end
    # a selector whose storage is the output: the review's reproducer, in each role (all run with bounds checking)
    bits(y) = copy(reinterpret(Int64, y))
    function aliased()
      y = collect(reinterpret(Float64, Int64[1, 2]))
      idx = view(reinterpret(Int64, y), 2:-1:1)
      y, idx
    end
    let (y, idx) = aliased()
      c = Constrained(Conjugate(ElasticNet(1.0, 0.5)), IndBox(-2.0, view([2.0, 2.0], idx)))
      before = bits(y)
      @test refused_value(value_and_prox!, y, c, [5.0, -5.0], 0.5)
      @test bits(y) == before
    end
    let (y, idx) = aliased()
      c = Constrained(Conjugate(ElasticNet(1.0, 0.5)), IndBox(view([-2.0, -2.0], idx), view([2.0, 2.0], idx)))
      before = bits(y)
      @test refused_value(value_and_prox!, y, c, [5.0, -5.0], 0.5)
      @test bits(y) == before
    end
    let (y, idx) = aliased()
      c = Constrained(Conjugate(ElasticNet(1.0, 0.5)), IndBox(-2.0, 2.0))
      before = bits(y)
      @test refused_value(value_and_prox!, y, c, view([5.0, -5.0], idx), 0.5)
      @test bits(y) == before
    end
    let (y, idx) = aliased()
      c = Constrained(Conjugate(ElasticNet(1.0, 0.5)), IndBox(-2.0, 2.0))
      before = bits(y)
      @test refused_value(value_and_prox!, y, c, [5.0, -5.0], view([0.5, 0.5], idx))
      @test bits(y) == before
    end
    let (y, idx) = aliased()
      c = Constrained(Conjugate(ElasticNet(1.0, 0.5, view([1.0, 1.0], idx))), IndBox(-2.0, 2.0))
      before = bits(y)
      @test refused_value(value_and_prox!, y, c, [5.0, -5.0], 0.5)
      @test bits(y) == before
    end
  end
  @testset "weight fields declared as unions" begin
    box = IndBox(-2.0, 2.0)
    for W in (Union{Nothing, Vector{Float64}}, Union{Nothing, AbstractVector}), wv in (nothing, [1.0, 3.0, 1.0])
      f = ElasticNet{Float64, W}(1.0, 0.5, wv)
      @test !PB._elastic_value_route(Constrained(Conjugate(f), box), zeros(3), x, 0.5)
      refusal(Constrained(Conjugate(f), box), () -> fill(-7.0, 3), x, 0.5; mkalias = () -> copy(x))
    end
    # concrete declarations take the route, whatever their type parameters are
    for W in (Nothing, Vector{Float64}, AbstractVector), (S, l) in ((Float64, (1.0, 0.5)), (Real, (1.0, 0.5)), (Float32, (1f0, 0.5f0)), (Int, (1, 1)))
      wv = W === Nothing ? nothing : [1.0, 3.0, 1.0]
      f = ElasticNet{S, W}(l..., wv)
      @test PB._elastic_value_route(Constrained(Conjugate(f), box), zeros(3), x, 0.5)
    end
  end
  @testset "array steps" begin
    for γ in (OffsetVec([0.5, 0.5, 0.5], -1), Real[0.5, 0.25, 1])
      refusal(c0, () -> fill(-7.0, 3), x, γ; mkalias = () -> copy(x))
      xs = SparseIterate(sparse(x))
      refusal(c0, () -> SparseIterate(3), xs, γ)
    end
    # the refusal comes from the step alone, with a good output and input: the generic refusal, not a shape error
    y = fill(-7.0, 3)
    @test_throws "is not implemented" value_and_prox!(y, c0, x, OffsetVec([0.5, 0.5, 0.5], -1))
    @test y == fill(-7.0, 3)
    @test_throws "is not implemented" PB._checkwrite(value_and_prox!, c0, y, x, OffsetVec([0.5, 0.5, 0.5], -1))
  end
  @testset "offset bounds in a later block" begin
    c = Constrained(Conjugate(ElasticNet(1.0, 0.5)), IndBox(OffsetVec([-2.0, -2.0, -2.0], -1), OffsetVec([2.0, 2.0, 2.0], -1)))
    s = SeparableSum(SqrNormL2(2.0), c)
    dest = ([7.0], fill(-7.0, 3))
    @test refused_value(value_and_prox!, dest, s, ([3.0], x), 0.5)
    @test dest == ([7.0], fill(-7.0, 3))
  end
end

# The fixtures above extend no behaviour of Base for other array types: allocation requests with offset axes are
# answered as in a fresh session, which is run here for comparison.
@testset "the OffsetVec fixture leaves Base.similar as it is" begin
  probe = """
    for T in (Vector{Float64}, Vector{Int}, BitVector, Array{Float32, 1}, typeof(view(zeros(3), 1:2)), Vector{Bool})
      r = try
        string(typeof(similar(T, (Base.IdentityUnitRange(1:3),))))
      catch e
        string(typeof(e))
      end
      println(T, " => ", r)
    end
    """
  fresh = readchomp(`$(Base.julia_cmd()) --startup-file=no -e $probe`)
  here = join((string(T, " => ", try
                        string(typeof(similar(T, (Base.IdentityUnitRange(1:3),))))
                      catch e
                        string(typeof(e))
                      end)
               for T in (Vector{Float64}, Vector{Int}, BitVector, Array{Float32, 1}, typeof(view(zeros(3), 1:2)), Vector{Bool})), "\n")
  @test here == fresh
  @test !occursin("OffsetVec", here)
  # the fixture's own allocation still works: for broadcasts over it and for similar of an instance
  a, b = OffsetVec([1.0, 2.0], -1), OffsetVec([2.0, 1.0], -1)
  m = a .<= b
  @test m isa OffsetVec{Bool} && axes(m) == axes(a) && m.data == [true, false]
  @test similar(a, Float32, (3,)) isa OffsetVec{Float32}
  @test all(a .<= OffsetVec([5.0, 5.0], -1))
  # every method of similar that this file defines has the fixture in its signature
  mine = filter(m -> m.module === @__MODULE__, collect(methods(similar)))
  @test length(mine) == 2
  @test all(m -> occursin("OffsetVec", string(m.sig)), mine)
end

end
