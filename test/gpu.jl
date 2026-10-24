module GPUTest

# The separable functions on GPU arrays, through JLArrays: GPUArrays' reference
# implementation of a GPU array, which runs on the CPU and, like a GPU,
# forbids reading or writing one entry at a time.

using Test
using ProximalBase
using LinearAlgebra
using Random
using JLArrays
using GPUArraysCore: allowscalar
import Adapt

Random.seed!(1)

# a separable function of one's own, a bits type: Σᵢ c xᵢ² / 2
struct ScaledSquare{T} <: SeparableFunction
  c::T
end
ProximalBase.value_entry(f::ScaledSquare, v, i) = f.c * v^2 / 2
ProximalBase.prox_entry(f::ScaledSquare, v, i, γ) = v / (1 + γ * f.c)
ProximalBase.iszeropreserving(::ScaledSquare) = true

# the values agree, including Inf for a point outside a set
same(a, b) = isinf(b) ? a == b : a ≈ b

@testset "separable functions on GPU arrays" begin
  allowscalar(false)
  n = 300
  x, w, l, u = 2 .* randn(n), 0.5 .+ rand(n), -rand(n), rand(n)
  for T in (Float64, Float32)
    xc = T.(x)
    xg = JLArray(xc)
    for f in (NormL1(0.3), NormL1(0.3, T.(w)), NormL1Nonneg(0.3), NormL1Nonneg(0.3, T.(w)),
              ElasticNet(0.3, 0.5), ElasticNet(0.3, 0.5, T.(w)), IndBox(-0.5, 0.7), IndBox(T.(l), T.(u)),
              IndBallLinf(0.4), IndNonnegative(), NormL0(0.2), MCP(0.3, 3.0), SCAD(0.3, 3.7), ScaledSquare(T(0.6)),
              SqrNormL2(0.7), SqrNormL2(0.7, T.(w)))
      @testset "$(nameof(typeof(f))), $T" begin
        fg = Adapt.adapt(JLArray, f)                       # its parameter arrays on the device
        yc = prox(f, xc, 0.8)
        yg = similar(xg)
        @test prox!(yg, fg, xg, 0.8) === yg
        @test eltype(yg) == T
        @test Array(yg) ≈ yc
        @test same(fg(xg), f(xc))
        @test same(fg(yg), f(yc))
        vc = value_and_prox!(similar(xc), f, xc, 0.8)
        @test same(value_and_prox!(yg, fg, xg, 0.8), vc)
        @test Array(yg) ≈ yc
        zg = copy(xg)
        @test Array(prox!(zg, fg, zg, 0.8)) ≈ yc        # in place
      end
    end
  end
  # matrices, and a point on each side of a box
  X = randn(6, 5)
  fw = NormL1(0.2, rand(6, 5))
  @test Adapt.adapt(JLArray, fw)(JLArray(X)) ≈ fw(X)
  @test Array(prox(Adapt.adapt(JLArray, IndBox(-0.1, 0.1)), JLArray(X))) ≈ clamp.(X, -0.1, 0.1)
  @test IndBox(0.0, 1.0)(JLArray([0.5, 1.0])) == 0
  @test IndBox(0.0, 1.0)(JLArray([0.5, 1.5])) == Inf
  # adapt converts the parameter arrays and keeps the constructors' checks
  @test Adapt.adapt(JLArray, NormL1(0.3, w)).w isa JLArray
  @test Adapt.adapt(JLArray, IndBox(l, u)).u isa JLArray
  @test Adapt.adapt(JLArray, NormL1(0.3)) == NormL1(0.3)
  @test_throws ArgumentError Adapt.adapt(JLArray, NormL1(0.3, -w))
end

@testset "norms, balls and the simplex on GPU arrays" begin
  allowscalar(false)
  n = 300
  x = 2 .* randn(n)
  for T in (Float64, Float32)
    xc = T.(x)
    xg = JLArray(xc)
    # each function also with a level or radius at which the result is zero or x
    for f in (NormL2(0.3), NormL2(100.), NormLinf(0.3), NormLinf(1e4), IndBallL2(1.), IndBallL2(1e3), IndBallL1(2.),
              IndBallL1(1e4), IndBallL1(0.), IndSimplex(1.), IndSimplex(3., equality = false), IndSimplex(1e4, equality = false),
              IndSimplex(1e4), IndSimplex(0.))
      @testset "$(nameof(typeof(f))), $T" begin
        yc = prox(f, xc, 0.8)
        yg = similar(xg)
        @test prox!(yg, f, xg, 0.8) === yg
        @test eltype(yg) == T
        @test Array(yg) ≈ yc
        @test same(f(xg), f(xc))
        @test same(f(yg), f(yc))
        vc = value_and_prox!(similar(xc), f, xc, 0.8)
        @test same(value_and_prox!(yg, f, xg, 0.8), vc)
        @test Array(yg) ≈ yc
        zg = copy(xg)
        @test Array(prox!(zg, f, zg, 0.8)) ≈ yc       # in place
      end
    end
  end
  # a matrix, and entries of one sign, where every entry stays active
  X = randn(6, 5)
  @test Array(prox(IndBallL1(1.), JLArray(X))) ≈ prox(IndBallL1(1.), X)
  @test Array(prox(IndSimplex(1.), JLArray(fill(0.5, 4)))) ≈ fill(0.25, 4)
  @test Array(prox(NormLinf(0.5), JLArray([3., -3., 3.]))) ≈ fill(3 - 0.5 / 3, 3) .* [1, -1, 1]
end

@testset "group norms on GPU arrays" begin
  allowscalar(false)
  # groups of sizes 1 to 26 in six size classes, out of order and not all
  # contiguous; x[2] and x[58:60] are in no group
  groups = [[1, 5, 9], 3:4, [6], 10:17, [8, 7], 18:30, [31], 32:57]
  n = 60
  x = 2 .* randn(n)
  x[3:4] .= 0                                         # a group at zero
  x[10:17] .*= 0.05                                   # groups that shrink to zero
  x[[8, 7]] .*= 0.05
  w = 0.5 .+ rand(length(groups))
  for T in (Float64, Float32)
    xc = T.(x)
    xg = JLArray(xc)
    # each function also with a level at which nothing or everything shrinks to zero
    for f in (GroupNormL2(0.3, groups), GroupNormL2(0.3, groups, w), GroupNormL2(0., groups), GroupNormL2(50., groups),
              GroupNormLinf(0.3, groups), GroupNormLinf(0.3, groups, w), GroupNormLinf(0., groups), GroupNormLinf(100., groups),
              SparseGroupNormL2(0.2, 0.3, groups), SparseGroupNormL2(0., 0.3, groups), SparseGroupNormL2(0.5, 1.5, groups, w),
              SparseGroupNormL2(0.2, 0., groups), IndBallGroupL0(3, groups), IndBallGroupL0(0, groups), IndBallGroupL0(20, groups))
      # with its groups on the device, and left on the host
      for (where, fg) in (("device", Adapt.adapt(JLArray, f)), ("host", f))
        @testset "$(nameof(typeof(f))) on the $where, $T" begin
          yc = prox(f, xc, 0.8)
          yg = similar(xg)
          @test prox!(yg, fg, xg, 0.8) === yg
          @test eltype(yg) == T
          @test Array(yg) ≈ yc
          @test same(fg(xg), f(xc))
          @test same(fg(yg), f(yc))
          vc = value_and_prox!(similar(xc), f, xc, 0.8)
          @test same(value_and_prox!(yg, fg, xg, 0.8), vc)
          @test Array(yg) ≈ yc
          zg = copy(xg)
          @test Array(prox!(zg, fg, zg, 0.8)) ≈ yc      # in place
        end
      end
    end
  end
  # the rows of a matrix: the multi-task group lasso
  X = randn(6, 5)
  rows = [i:6:30 for i in 1:6]
  for f in (GroupNormL2(0.4, rows), GroupNormLinf(0.4, rows), SparseGroupNormL2(0.2, 0.4, rows))
    fg = Adapt.adapt(JLArray, f)
    @test Array(prox(fg, JLArray(X), 0.7)) ≈ prox(f, X, 0.7)
    @test fg(JLArray(X)) ≈ f(X)
  end
  # the earlier groups first among ties, and at most k entries in a group
  f = IndBallGroupL0(2, [1:2, 3:4, 5:6])
  @test Array(prox(Adapt.adapt(JLArray, f), JLArray([1., 1., 1., 1., 2., 0.]))) == [1., 1., 0., 0., 2., 0.]
  fk = IndBallGroupL0(3, groups; k = 2)
  @test Adapt.adapt(JLArray, fk)(JLArray(x)) == fk(x) == Inf
  @test IndBallGroupL0(6, groups; k = 30)(JLArray(x)) == Inf          # seven nonzero groups
  @test IndBallGroupL0(7, groups; k = 30)(JLArray(x)) == 0
  @test_throws ArgumentError prox(Adapt.adapt(JLArray, fk), JLArray(x))
  # no groups, and groups that do not fit in x
  f0 = GroupNormL2(0.3, UnitRange{Int}[])
  @test f0(JLArray(x)) == 0
  @test Array(prox(f0, JLArray(x))) == x
  @test_throws BoundsError prox(GroupNormL2(0.3, groups), JLArray(randn(20)))
  @test_throws DimensionMismatch prox!(JLArray(randn(59)), GroupNormL2(0.3, groups), JLArray(x), 1.0)
  # adapt to the device and back; on host arrays the adapted function is the function
  f = SparseGroupNormL2(0.2, 0.3, groups)
  fg = Adapt.adapt(JLArray, f)
  @test Adapt.adapt(Array, fg) === f
  @test Adapt.adapt(Array, f) === f
  @test Array(prox(Adapt.adapt(JLArray, fg), JLArray(x))) ≈ prox(f, x)
  @test isproximable(fg) && isconvex(fg) && !ProximalBase._isindicator(fg)
  fl0 = Adapt.adapt(JLArray, IndBallGroupL0(2, groups))
  @test isproximable(fl0) && !isconvex(fl0) && ProximalBase._isindicator(fl0)
  @test fg(x) == f(x)
  @test prox(fg, x, 0.8) == prox(f, x, 0.8)
  @test value_and_prox!(similar(x), fg, x, 0.8) == value_and_prox!(similar(x), f, x, 0.8)
  @test value_and_prox!(JLArray(similar(x)), fl0, JLArray(x), 1.0) == 0
end

end
