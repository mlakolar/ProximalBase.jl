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

end
