module ExpConeTest

using Test
using ProximalBase
using LinearAlgebra
using Random

using ..TestUtils

const PB = ProximalBase
const PT = ProximalBase.Testing
const rng = MersenneTwister(171)

# Independent high-precision inequalities, applied after normalizing each
# triple. The face test allows components rounded to zero at underflow.
function in_primal(x, tol)
  r, s, t = x
  (abs(s) <= tol && r <= tol && t >= -tol) ||
    (s > 0 && t >= -tol && t + tol > 0 && r <= s * log((t + tol) / s) + tol)
end
function in_polar(x, tol)
  r, s, t = x
  (abs(r) <= tol && s <= tol && t <= tol) ||
    (r > 0 && t <= tol && tol - t > 0 && s <= r * (1 + log((tol - t) / r)) + tol)
end

function check_moreau(v::AbstractVector{T}) where {T}
  p = prox(IndExpPrimal(), v)
  q = -prox(IndExpDual(), -v)
  scale = maximum(abs, BigFloat.(v))
  iszero(scale) && (scale = one(scale))
  x, a, b = BigFloat.(v) ./ scale, BigFloat.(p) ./ scale, BigFloat.(q) ./ scale
  tol = BigFloat(T === Float32 ? 1e-4 : 1e-10)
  allowance = 8BigFloat(nextfloat(zero(T))) / scale
  @test all(isfinite, p) && all(isfinite, q)
  @test in_primal(a, tol + allowance)
  @test in_polar(b, tol + allowance)
  @test maximum(abs.(x .- a .- b)) <= tol + allowance
  @test abs(dot(a, b)) <= tol + 4allowance
  @test IndExpPrimal()(p) === zero(T)
  @test IndExpDual()(-q) === zero(T)
  pp = prox(IndExpPrimal(), p)
  qq = prox(IndExpDual(), -q)
  @test maximum(abs.(BigFloat.(pp) .- BigFloat.(p))) / scale <= tol + allowance
  @test maximum(abs.(BigFloat.(qq) .+ BigFloat.(q))) / scale <= tol + allowance
  p, q
end

# Manufacture orthogonal primal and polar boundary points, without a root
# search or an optimizer. Positive combinations give their own projections.
function manufactured(T, rho, a, b; scale = 1)
  setprecision(BigFloat, 256) do
    h = BigFloat(rho)
    e = exp(h)
    p = BigFloat[h, 1, e]
    q = BigFloat[e, (1 - h) * e, -1]
    p .*= BigFloat(a) * BigFloat(scale) / maximum(abs, p)
    q .*= BigFloat(b) * BigFloat(scale) / maximum(abs, q)
    T.(p + q), p, q
  end
end

function check_manufactured(T, rho, a, b; scale = 1)
  x, p, q = manufactured(T, rho, a, b; scale)
  y, z = check_moreau(x)
  magnitude = max(maximum(abs, BigFloat.(x)), maximum(abs, p), maximum(abs, q))
  tol = BigFloat(T === Float32 ? 1e-4 : 1e-10)
  allowance = tol * magnitude + 8BigFloat(nextfloat(zero(T)))
  @test maximum(abs.(BigFloat.(y) .- p)) <= allowance
  @test maximum(abs.(BigFloat.(z) .- q)) <= allowance
  # Absolute errors are also bounded for the unit-scale reference cases.
  scale == 1 && (@test maximum(abs.(BigFloat.(y) .- p)) <= tol * max(1, magnitude))
end

@testset "Exponential cone known projections and membership" begin
  for T in (Float64, Float32)
    f, g = IndExpPrimal(), IndExpDual()
    for x in (T[0, 1, 2], T[0, 1, 1], T[-2, 0, 3], T[-2, 0, 0], zeros(T, 3))
      @test f(x) === zero(T)
      @test prox(f, x) == x
      check_moreau(x)
    end
    for x in (T[0, 2, 3], T[-1, 1, 1], T[0, 0, 3])
      @test g(x) === zero(T)
      @test prox(g, x) == x
    end
    for x in (T[-2, -3, 4], T[-2, -3, -4], T[0, -3, 4], T[-2, 0, -4])
      @test prox(f, x) == T[x[1], 0, max(x[3], 0)]
      check_moreau(x)
    end
    for x in (T[1, -2, -3], T[0, -2, -3], T[1, 0, -1])
      @test prox(f, x) == zeros(T, 3)
      @test Conjugate(f)(x) == 0
    end
    tol = PB._tol(T)
    @test f(T[tol / 2, 0, 1]) == 0
    @test isinf(f(T[2tol, 0, 1]))
    @test f(T[-1, -tol / 2, 1]) == 0
    @test isinf(f(T[-1, -2tol, 1]))
    @test g(T[tol / 2, 1, 1]) == 0
    @test isinf(g(T[2tol, 1, 1]))
    @test f(T[tol / 2, 1, 1]) == 0
    @test isinf(f(T[3tol, 1, 1]))
    @test g(T[-1, -1 - tol / 2, 1]) == 0
    @test isinf(g(T[-1, -1 - 3tol, 1]))
    # Tolerant values must not select an unchanged projection branch.
    x = T[2eps(T), 1, 1]
    @test f(x) == 0
    @test prox(f, x) != x
  end
  @test isinf(IndExpPrimal()([1e300, 1e-300, 1e300]))
  @test isinf(IndExpDual()([-1e-300, -1e300, 1e300]))
end

@testset "Exponential cone manufactured independent references" begin
  for T in (Float64, Float32)
    for rho in (-30, -8, -1, 0, 1, 2, 8, 30), (a, b) in ((1, 1), (0, 1), (1, 0), (0.01, 3), (3, 0.01))
      check_manufactured(T, rho, a, b)
    end
    for _ in 1:200
      check_manufactured(T, 24rand(rng) - 12, 0.1 + rand(rng), 0.1 + rand(rng))
    end
  end
end

@testset "Exponential cone Moreau characterization and boundary pieces" begin
  for T in (Float64, Float32)
    cases = [randn(rng, T, 3) for _ in 1:200]
    for delta in (zero(T), eps(T), -eps(T), T(1e-5), T(-1e-5)), sign in (-1, 1)
      append!(cases, [T[sign, delta, 1], T[delta, sign, 1], T[sign, 1, delta],
                      T[-1, delta, 0], T[delta, 0, 1], T[0, delta, sign]])
    end
    for x in cases
      p, q = check_moreau(x)
      for f in (IndExpPrimal(), IndExpDual())
        y = copy(x)
        prox!(y, f, y, T(0.7))
        @test y == prox(f, x)
        @test value_and_prox!(similar(x), f, x, 2.0) === zero(T)
      end
      @test prox(Conjugate(IndExpPrimal()), x) ≈ q rtol = 1e-4 atol = 1e-6
      if T === Float64
        @test prox(Conjugate(IndExpPrimal()), x) ≈ q rtol = 1e-10 atol = 1e-10
      end
      for f in (IndExpPrimal(), IndExpDual()), factor in (T(0.125), T(8))
        tol = T === Float64 ? 1e-10 : 1e-4
        @test prox(f, factor .* x) ≈ factor .* prox(f, x) rtol = tol atol = tol
      end
    end
  end
end

@testset "Exponential cone exponents and asymmetric magnitudes" begin
  for T in (Float64, Float32)
    large, small = T === Float64 ? (T(1e300), T(1e-300)) : (T(1e30), T(1e-30))
    M, tiny = floatmax(T) / 4, nextfloat(zero(T))
    cases = [T[0.01, -8, 8], T[-8, 0.01, -8], T[1, small, 1],
             T[large, small, large], T[-large, small, 0], T[small, 0, large],
             T[0, large, small], T[small, large, small], T[large, -small, 0],
             T[M, 0, 0], T[0, M, M], T[M, -M, M], T[-M, M, -M],
             T[tiny, 0, tiny], T[0, tiny, 0], T[-tiny, tiny, -tiny]]
    for rho in (-1e4, -700, 700, 1e4)
      append!(cases, [T[rho, 1, 1], T[rho, 1, -1]])
    end
    for x in copy(cases)[1:2], factor in (large, small)
      push!(cases, factor .* x)
    end
    for x in cases
      check_moreau(x)
      check_moreau(-x)
    end
    span = T === Float64 ? 300 : 30
    for _ in 1:100
      x = T[sign(randn(rng)) * 10.0^(span * (2rand(rng) - 1)) for _ in 1:3]
      @testset "$T asymmetric x=$x" begin
        check_moreau(x)
      end
    end
    for rho in (-1e4, -801, -700, 700, 801, 1e4), scale in (small, one(T), large / 4)
      check_manufactured(T, rho, 1, 1; scale)
    end
    for rho in (-700, -100, -10, 10, 100, 700), scale in (128tiny, 4096tiny)
      check_manufactured(T, rho, 1, 1; scale)
    end
    # Small representable components must survive beside a huge coordinate.
    x = T[small, 0, large]
    p, _ = check_moreau(x)
    @test p[1] > small / 2
    @test p[2] > 0
    @test prox(IndExpPrimal(), T[-large, small, 0])[2] == small
  end
end

@testset "Exponential cone groups and free entries" begin
  groups = [[3, 1, 2], [7, 5, 6]]
  x = [1.0, 1, 1, 17, 0, -1, 2, 19]
  for Constructor in (IndExpPrimal, IndExpDual)
    f = Constructor(groups)
    p = prox(f, x)
    @test p[[4, 8]] == x[[4, 8]]
    for G in groups
      @test p[G] == prox(Constructor(), x[G])
    end
    @test prox(f, reshape(x, 2, 4)) == reshape(p, 2, 4)
    @test f(reshape(p, 2, 4)) == 0
    @test prox(Constructor(), reshape(x[1:3], 1, 3)) == reshape(prox(Constructor(), x[1:3]), 1, 3)
    y = copy(x)
    @test prox!(y, f, y, 0.3) === y
    @test y == p
    q = prox(Conjugate(f), x)
    @test q[[4, 8]] == zeros(2)
    @test Conjugate(f)(q) == 0
    @test p + q ≈ x rtol = 1e-10
    z = copy(q)
    z[4] = 1
    @test isinf(Conjugate(f)(z))
    z[4] = PB._tol(Float64) / 2
    @test Conjugate(f)(z) == 0
    empty = Constructor(UnitRange{Int}[])
    @test empty(x) == 0
    @test prox(empty, x) == x
    @test empty(Float64[]) == 0
    @test prox(empty, Float64[]) == Float64[]
    @test Conjugate(empty)(zeros(8)) == 0
    @test isinf(Conjugate(empty)(x))
    @test prox(Conjugate(empty), x) == zeros(8)
    gap = Constructor([2:4])
    @test prox(gap, x)[[1, 5, 6, 7, 8]] == x[[1, 5, 6, 7, 8]]
    @test f.groups == groups
  end
  owned = IndExpPrimal(groups)
  groups[1][1] = 4
  push!(groups, [8, 9, 10])
  @test owned.groups == [[3, 1, 2], [7, 5, 6]]
  ranges = [1:3, 5:7]
  owned = IndExpDual(ranges)
  ranges[1] = 2:4
  @test owned.groups == [1:3, 5:7]
end

struct ZeroBased{T} <: AbstractVector{T}
  data::Vector{T}
end
Base.size(x::ZeroBased) = size(x.data)
Base.axes(x::ZeroBased) = (0:length(x.data)-1,)
Base.getindex(x::ZeroBased, i::Int) = x.data[i + 1]

@testset "Exponential cone contracts and validation before writes" begin
  for Constructor in (IndExpPrimal, IndExpDual)
    f = Constructor()
    for n in (0, 1, 2, 4), op in (prox!, value_and_prox!)
      x, y = ones(n), fill(17.0, n)
      @test_throws ArgumentError f(x)
      @test_throws ArgumentError op(y, f, x, 1.0)
      @test y == fill(17.0, n)
    end
    for groups in ([Int[]], [[1]], [[1, 2]], [1:4], [[1, 1, 2]], [[0, 1, 2]], [[-1, 2, 3]], [1:3, 3:5])
      @test_throws ArgumentError Constructor(groups)
    end
    bad = Constructor([1:3, 5:7])
    @test_throws BoundsError bad(ones(6))
    @test_throws BoundsError Conjugate(bad)(ones(6))
    for g in (bad, Scaled(bad, 2.0), Translate(bad, ones(6)), Tilt(bad, ones(6)), Conjugate(bad)),
        op in (prox!, value_and_prox!), alias in (false, true)
      x = ones(6)
      y = alias ? x : fill(17.0, 6)
      before = copy(y)
      @test_throws BoundsError op(y, g, x, 1.0)
      @test x == ones(6)
      @test y == before
    end
    for op in (prox!, value_and_prox!)
      x, y = [1.0, 1, 1], fill(17.0, 3)
      @test_throws MethodError op(y, f, x, ones(3))
      @test_throws ArgumentError("$(nameof(typeof(f))) requires real entries") op(y, f, complex.(x), 1.0)
      @test_throws ArgumentError op(y, f, ZeroBased(x), 1.0)
      @test y == fill(17.0, 3)
      @test_throws ArgumentError("$(nameof(typeof(f))) requires real entries") op(complex.(y), f, x, 1.0)
      @test_throws ArgumentError op(ZeroBased(y), f, x, 1.0)
      @test_throws DimensionMismatch op(reshape(y, 1, 3), f, x, 1.0)
    end
    @test_throws ArgumentError("$(nameof(typeof(f))) requires real entries") f(complex.(ones(3)))
    @test_throws ArgumentError f(ZeroBased(ones(3)))
    for T in (Float64, Float32), nonfinite in (T(NaN), T(Inf), T(-Inf)), i in 1:3
      x = T[1, 1, 1]
      x[i] = nonfinite
      @test isinf(f(x))
      @test isinf(Conjugate(f)(x))
      @test all(isnan, prox(f, x))
      y = copy(x)
      prox!(y, f, y, one(T))
      @test all(isnan, y)
      grouped = Constructor([1:3, 5:7])
      v = [x; T(Inf); T[0, 1, 2]]
      p = prox(grouped, v)
      @test all(isnan, p[1:3])
      @test p[4] == Inf
      @test p[5:7] == prox(f, v[5:7])
      @test isinf(grouped(v))
    end
    @test isinf(Constructor(UnitRange{Int}[])([Inf]))
    x = [typemin(Int), 0, 1]
    @test prox(f, x) == prox(f, float.(x))
    @test Conjugate(f)(x) == Conjugate(f)(float.(x))
  end
end

@testset "Exponential cone precision and generic helpers" begin
  for Constructor in (IndExpPrimal, IndExpDual),
      f in (Constructor(), Constructor([1:3]), Constructor([[3, 1, 2]]))
    check_traits(f; proximable = true, convex = true)
    @test !isseparable(f)
    @test PB._isindicator(f)
    x = [1.0, 1, 1]
    check_prox(f, x, 0.7; rng)
    PT.test_conjugate(f, x, 0.7; conjugate = Conjugate(f))
    for T in (Float32, Float64)
      v = T.(x)
      @test f(v) isa T
      @test Conjugate(f)(v) isa T
      @test prox(f, v) isa Vector{T}
      for S in (Float32, Float64)
        y = similar(v, S)
        @test prox!(y, f, v, 0.7) === y
        @test y ≈ S.(prox(f, Float64.(v))) rtol = 1e-4
        @test value_and_prox!(y, f, v, 0.7) isa S
        @test value_and_prox!(y, Conjugate(f), v, 0.7) isa S
      end
    end
    grouped = Constructor([1:3, 5:7])
    check_prox(grouped, randn(rng, 8), 0.8; rng)
  end
  @test prox!(zeros(Float32, 3), IndExpPrimal(), [1e100, -1e100, -1e100], 1) == zeros(Float32, 3)
end

function fresh_allocations(op::O, f::F, x::X) where {O, F, X}
  y = copy(x)
  op(y, f, y, 1)
  copyto!(y, x)
  @allocated op(y, f, y, 1)
  copyto!(y, x)
  @allocated op(y, f, y, 1)
end

function check_allocations(f::F, x::X) where {F, X}
  y = similar(x)
  for g in (f, Conjugate(f))
    @test allocs(g, x) == 0
    for op in (prox!, value_and_prox!)
      @test allocs(op, y, g, x, 1) == 0
      @test fresh_allocations(op, g, x) == 0
    end
  end
end

@testset "Exponential cone allocations on fresh inputs" begin
  for T in (Float64, Float32), Constructor in (IndExpPrimal, IndExpDual)
    for x in (T[0, 1, 2], T[1, -2, -3], T[-2, -3, 4], T[1, 1, 1],
              T[0.01, -8, 8], T[-8, 0.01, -8], T[floatmax(T) / 4, 0, 0],
              T[nextfloat(zero(T)), 0, nextfloat(zero(T))])
      check_allocations(Constructor(), x)
      for groups in ([1:3, 5:7], [[3, 1, 2], [7, 5, 6]])
        v = fill(T(17), 8)
        for G in groups
          v[G] = x
        end
        check_allocations(Constructor(groups), v)
        check_allocations(Constructor(groups), reshape(v, 2, 4))
      end
    end
  end
end

end
