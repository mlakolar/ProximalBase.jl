module MinimumTest

using Test
using ProximalBase
using LinearAlgebra
using Random

using ..TestUtils

const PB = ProximalBase
Random.seed!(179)

# Singleton domains give exact candidate points and independently chosen
# offsets, including deliberately invalid value oracles for exceptional cases.
struct MinimumPoint{T<:Real, V<:Real} <: AbstractFunction
  point::T
  offset::V
end
PB.isproximable(::Type{<:MinimumPoint}) = true
PB.isconvex(::Type{<:MinimumPoint}) = true
(f::MinimumPoint)(x::AbstractArray) = all(==(f.point), x) ? f.offset : oftype(float(f.offset), Inf)
PB.prox!(y::AbstractArray, f::MinimumPoint, x::AbstractArray, γ::Real) = fill!(y, f.point)
PB.value_and_prox!(y::AbstractArray, f::MinimumPoint, x::AbstractArray, γ::Real) =
  (fill!(y, f.point); f.offset)

struct MinimumTyped <: AbstractFunction end
PB.isproximable(::Type{MinimumTyped}) = true
(::MinimumTyped)(x::AbstractArray) = 0.0
PB.prox!(y::AbstractArray, ::MinimumTyped, x::AbstractArray, γ::Real) = copyto!(y, x)
function PB._checkwrite(op::Union{typeof(prox!), typeof(value_and_prox!)}, ::MinimumTyped, y::AbstractArray, x::AbstractArray)
  eltype(y) == Float64 || throw(ArgumentError("this test component requires a Float64 output"))
  true
end

function reference(fs, x, γ; T = eltype(x))
  candidates = map(fs) do f
    y = similar(x, T)
    v = value_and_prox!(y, f, x, γ)
    # The offset and energy can be over a thousand binary orders apart.
    score = setprecision(BigFloat, 4096) do
      BigFloat(v) + sum(abs2, BigFloat.(y) - BigFloat.(x)) / (2BigFloat(γ))
    end
    (y, v, score)
  end
  scores = map(last, candidates)
  winner = argmin(scores)
  candidates[winner]
end

function check_candidates(fs, x, γ)
  g = PointwiseMinimum(fs...)
  p, v, _ = reference(fs, x, γ)
  @test isequal(prox(g, x, γ), p)
  y = similar(x)
  @test value_and_prox!(y, g, x, γ) == v
  @test isequal(y, p)
  @test g(x) == minimum(f(x) for f in fs)
  z = copy(x)
  @test prox!(z, g, z, γ) === z
  @test isequal(z, p)
  z = copy(x)
  @test value_and_prox!(z, g, z, γ) == v
  @test isequal(z, p)
  g
end

@testset "PointwiseMinimum" begin
  @test_throws ArgumentError PointwiseMinimum()
  @test_throws ArgumentError PointwiseMinimum(Sum(NormL1(), NormL2()))
  @test_throws ArgumentError PointwiseMinimum(NormL1(), Sum(NormL1(), NormL2()))
  @test_throws MethodError PointwiseMinimum((NormL1(), NormL2()))

  @testset "unions of sets" begin
    for T in (Float32, Float64)
      fs = ((IndIsotonic(), IndIsotonic(decreasing = true)),
            (IndNonnegative(), IndBox(-Inf, 0)),
            (IndBox(T(-3), T(-1)), IndBox(T(1), T(2))),
            (Translate(IndBallL2(T(1)), T[-2, 0, 0, 0]),
             Translate(IndBallL2(T(0.5)), T[2, 0, 0, 0])))
      for sets in fs, x in (T[3, -1, 2, -2], T[-3, -1, 1, 2], T[2, 1, -1, -3], zeros(T, 4))
        g = check_candidates(sets, x, T(0.7))
        @test PB._isindicator(g)
        @test g(prox(g, x)) == zero(T)
        check_traits(g; proximable = true)
      end
    end
    g = PointwiseMinimum(IndNonnegative(), IndBox(-Inf, 0))
    check_prox(g, [2.0, -0.5, 1.0], 0.7)
    @test g([1.0, -1.0]) == Inf
  end

  @testset "penalties and singleton" begin
    for T in (Float32, Float64)
      fs = (Translate(NormL1(T(0.5)), T[-2, 0]),
            Translate(NormL1(T(0.5)), T[1, 2]),
            Translate(NormL1(T(0.5)), T[3, -1]))
      for x in (T[-3, 0], T[1, 3], T[4, -1], zeros(T, 2))
        check_candidates(fs, x, T(0.7))
      end
      for f in (NormL1(T(0.5)), IndIsotonic(), NormL0(T(0.3)))
        x = T[2, -1, 0.5]
        g = check_candidates((f,), x, T(0.7))
        @test g(x) === f(x)
        check_traits(g; proximable = true, convex = isconvex(f))
      end
    end
    g = PointwiseMinimum(Translate(NormL1(0.5), [-2.0, 0.0]),
                         Translate(NormL1(0.5), [1.0, 2.0]))
    check_prox(g, [-1.0, 0.3], 0.7)
    @test !PB._isindicator(g)
    @test_throws MethodError g(([1.0], [2.0]))
    @test_throws ArgumentError prox(g, ones(2), ones(2))
    @test_throws ArgumentError value_and_prox!(ones(2), g, ones(2), ones(2))
  end

  @testset "ties, offsets and heterogeneous winners" begin
    fs = (IndBox(-1, -1), IndBox(1, 1))
    for sets in (fs, reverse(fs))
      @test prox(PointwiseMinimum(sets...), [0.0]) == prox(first(sets), [0.0])
      check_candidates(sets, [0.0], 1.0)
    end
    # Neither distance alone nor the component value alone selects the branch.
    @test prox(PointwiseMinimum(MinimumPoint(1.0, 10.0), MinimumPoint(2.0, 0.0)), [0.0]) == [2.0]
    @test prox(PointwiseMinimum(MinimumPoint(1.0, 0.0), MinimumPoint(2.0, -0.5)), [0.0]) == [1.0]
    fs = (Translate(NormL1(2.0), [-3.0]), SqrNormL2(4.0), IndBox(3, 3))
    for (winner, x) in enumerate(([-3.0], [0.0], [3.0]))
      g = check_candidates(fs, x, 0.5)
      @test isequal(prox(g, x, 0.5), prox(fs[winner], x, 0.5))
    end
  end

  @testset "range-safe comparison" begin
    for a in (2.0^600, 2.0^-600), γ in (2.0^-500, 1.0, 2.0^500)
      for fs in ((IndBox(a, a), IndBox(a / 2, a / 2)),
                 (MinimumPoint(a, 0.0), MinimumPoint(a / 2, 0.0)),
                 (MinimumPoint(a, -a), MinimumPoint(a / 2, -a)))
        check_candidates(fs, [0.0], γ)
        check_candidates(reverse(fs), [0.0], γ)
      end
    end
    # Both the subtraction and the norm can exceed Float64's range.
    a = floatmax(Float64)
    for x in ([-a], fill(-a, 4))
      for fs in ((IndBox(a, a), IndBox(a / 2, a / 2)),
                 (MinimumPoint(a, 0.0), MinimumPoint(a / 2, 0.0)))
        check_candidates(fs, x, 1.0)
      end
    end
    # Large, signed offsets must participate in the comparison.
    for fs in ((MinimumPoint(2.0^500, -2.0^1000), MinimumPoint(2.0^499, -2.0^998)),
               (MinimumPoint(2.0^-500, -2.0^-1000), MinimumPoint(2.0^-499, -2.0^-998)))
      check_candidates(fs, [0.0], 1.0)
      check_candidates(reverse(fs), [0.0], 1.0)
    end
  end

  @testset "precision and buffer reuse" begin
    g = PointwiseMinimum(NormL1(1f0), LinearFunction(Float32[0], 1.75))
    for x in (Float32[2], Float32[0.5], Float32[3])
      y = similar(x)
      @test g(x) isa Float64
      @test value_and_prox!(y, g, x, 1f0) isa Float64
      @test value_and_prox!(y, g, x, 1f0) == g(y)
      @test isequal(y, first(reference(g.fs, x, 1f0)))
    end
    g = PointwiseMinimum(NormL1(0.5), SqrNormL2(2f0), ZeroFunction())
    for T in (Float32, Float64, Float32), dims in ((3,), (2, 3), (0,), (2, 2))
      x = randn(T, dims)
      for S in (Float64, Float32)
        y = similar(x, S)
        value_and_prox!(y, g, x, 0.7)
        @test eltype(y) == S
        @test g.best.buf isa Array{S, length(dims)}
        @test g.current.buf isa Array{S, length(dims)}
        @test size(g.best.buf) == dims
        @test isequal(y, first(reference(g.fs, x, 0.7; T = S)))
      end
    end
  end

  @testset "preflight before writes" begin
    bad = PointwiseMinimum(NormL1(), NormL1(1.0, ones(3)))
    for op in (prox!, value_and_prox!)
      x, y = ones(2), fill(17.0, 2)
      @test_throws DimensionMismatch op(y, bad, x, 1.0)
      @test y == fill(17.0, 2)
      @test_throws DimensionMismatch op(x, bad, x, 1.0)
      @test x == ones(2)
      @test_throws DimensionMismatch bad(x)
      @test_throws DimensionMismatch op(ones(3), PointwiseMinimum(NormL1()), x, 1.0)
      # Center shapes pass _checkparams but fail the component's _checkwrite.
      g = PointwiseMinimum(NormL1(), SqrNormL2(1.0, ones(1, 2)))
      for wrap in (identity, f -> Tilt(f, ones(2)), f -> Translate(f, ones(2)))
        y = fill(17.0, 2)
        @test_throws DimensionMismatch op(y, wrap(g), x, 1.0)
        @test y == fill(17.0, 2)
      end
      s = SeparableSum(NormL1(), g)
      ys = (fill(17.0, 2), fill(17.0, 2))
      @test_throws DimensionMismatch op(ys, s, (x, x), 1.0)
      @test all(y -> y == fill(17.0, 2), ys)
      # A valid outer shape must still check the candidate output precision.
      g = PointwiseMinimum(NormL1(), MinimumTyped())
      y = fill(17f0, 2)
      @test_throws ArgumentError op(y, g, x, 1.0)
      @test y == fill(17f0, 2)
    end
  end

  @testset "exceptional values and missing value oracles" begin
    for T in (Float32, Float64), x in (T[], zeros(T, 3))
      check_candidates((NormL1(T(0.5)), ZeroFunction()), x, one(T))
      check_candidates((IndBox(-1, 1), IndNonnegative()), x, one(T))
    end
    for fs in ((MinimumPoint(1.0, NaN), MinimumPoint(2.0, 0.0)),
               (MinimumPoint(1.0, Inf), MinimumPoint(2.0, 0.0)),
               (MinimumPoint(NaN, 0.0), MinimumPoint(2.0, 0.0)),
               (MinimumPoint(Inf, 0.0), MinimumPoint(2.0, 0.0)))
      for components in (fs, reverse(fs))
        g = PointwiseMinimum(components...)
        y = zeros(1)
        @test value_and_prox!(y, g, [0.0]) == 0.0
        @test y == [2.0]
      end
    end
    g = PointwiseMinimum(MinimumPoint(1.0, NaN), MinimumPoint(2.0, NaN))
    y = zeros(1)
    @test isnan(value_and_prox!(y, g, [0.0]))
    @test y == [1.0]
    @test isnan(g([2.0]))
    g = PointwiseMinimum(MinimumPoint(1.0, Inf), MinimumPoint(2.0, Inf))
    @test value_and_prox!(y, g, [0.0]) == Inf
    @test y == [1.0]
    g = PointwiseMinimum(IndBox(-1, -1), IndBox(1, 1))
    @test isnan(value_and_prox!(y, g, [NaN]))
    @test isequal(y, prox(first(g.fs), [NaN]))
    @test value_and_prox!(y, g, [Inf]) == 0.0
    @test y == [-1.0]
    g = PointwiseMinimum(Conjugate(ElasticNet(0.5, 0.2)), NormL1())
    @test_throws ArgumentError g([2.0])
    @test isfinite(value_and_prox!(y, g, [2.0]))
  end

  @testset "reset component caches" begin
    X, Y, x = randn(6, 2), randn(6), randn(2)
    f = LeastSquaresLoss(Y, X)
    g = PointwiseMinimum(f, SqrNormL2(0.5))
    prox(g, x, 0.7)
    @test f.svd_ready
    X .*= 1.2
    Y .+= 0.3
    @test reset_cache!(g) === g
    @test !f.svd_ready
    fresh = PointwiseMinimum(LeastSquaresLoss(Y, X), SqrNormL2(0.5))
    @test prox(g, x, 0.7) ≈ prox(fresh, x, 0.7)
  end

  @testset "zero steady-state allocations" begin
    for T in (Float32, Float64)
      x, y = T[2, -1, 0.5], zeros(T, 3)
      for g in (PointwiseMinimum(NormL1(T(0.5))),
                PointwiseMinimum(IndIsotonic(), IndIsotonic(decreasing = true)),
                PointwiseMinimum(NormL1(T(0.5)), SqrNormL2(T(2)), LinearFunction(zeros(T, 3), 1.75)))
        @test allocs(g, x) == 0
        @test allocs(prox!, y, g, x, T(0.7)) == 0
        @test allocs(value_and_prox!, y, g, x, T(0.7)) == 0
        @test allocs(prox!, y, g, y, T(0.7)) == 0
        @test allocs(value_and_prox!, y, g, y, T(0.7)) == 0
      end
    end
    for (a, x) in ((2.0^600, [0.0]), (2.0^-600, [0.0]),
                   (floatmax(Float64), [-floatmax(Float64)]))
      for g in (PointwiseMinimum(IndBox(a, a), IndBox(a / 2, a / 2)),
                PointwiseMinimum(MinimumPoint(a, -a), MinimumPoint(a / 2, -a)))
        y = similar(x)
        @test allocs(g, x) == 0
        @test allocs(prox!, y, g, x, 1.0) == 0
        @test allocs(value_and_prox!, y, g, x, 1.0) == 0
      end
    end
  end
end

end
