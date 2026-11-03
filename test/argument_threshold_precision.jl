module ArgumentThresholdPrecisionTest
using Test, ProximalBase
using ..TestUtils

# The group kernels compare the norm of the argument with a threshold formed in the model
# precision of the call, the wider of the output's and the argument's (numerics T3), and round it
# once. A Float64 argument and a Float32 destination therefore give the Float64 result rounded
# once, not the result of a threshold rounded to Float32 first. One-element groups show the
# difference: the margin of x = 1 + 2^-28 over γ = 1 - 2^-28 is 2^-27.

const x1 = [1.0 + 2.0^-28]
const γ1 = 1.0 - 2.0^-28
const G1 = [1:1]

penalties() = [("GroupNormL2", GroupNormL2(1.0, G1, [1.0])),
               ("GroupNormLinf", GroupNormLinf(1.0, G1, [1.0])),
               ("SparseGroupNormL2", SparseGroupNormL2(1.0, 0.0, G1, [1.0])),
               ("GroupMCP", GroupMCP(1.0, G1, [1.0]; a = 3.0)),
               ("GroupSCAD", GroupSCAD(1.0, G1, [1.0]; a = 4.0))]

@testset "Float64 argument, Float32 destination" begin
  expected = Dict("GroupNormL2" => 2.0^-27, "GroupNormLinf" => 2.0^-27, "SparseGroupNormL2" => 2.0^-27,
                  "GroupMCP" => 1.1175871e-8, "GroupSCAD" => 2.0^-27)
  for (name, f) in penalties()
    @testset "$name" begin
      ref = prox(f, x1, γ1)                      # all Float64
      @test Float32(ref[1]) == Float32(expected[name])
      y = zeros(Float32, 1)
      prox!(y, f, x1, γ1)
      @test y == Float32.(ref)
      v = value_and_prox!(zeros(Float32, 1), f, x1, γ1)
      y2 = zeros(Float32, 1)
      @test value_and_prox!(y2, f, x1, γ1) === v
      @test y2 == Float32.(ref)
      # a vector of steps, and a longer group, take the same path
      y3 = zeros(Float32, 1)
      prox!(y3, f, x1, [γ1])
      @test y3 == y
    end
  end
end

@testset "Float32 argument, Float64 destination: no loss beyond the argument's norm" begin
  x = Float32[1.5, -2.5, 0.25]
  for f in (GroupNormL2(0.7, [1:2, 3:3]), GroupNormLinf(0.7, [1:2, 3:3]),
            SparseGroupNormL2(0.7, 0.2, [1:2, 3:3]), GroupMCP(0.7, [1:2, 3:3]), GroupSCAD(0.7, [1:2, 3:3]))
    y = zeros(Float64, 3)
    prox!(y, f, x, 0.9)
    @test y ≈ prox(f, Float64.(x), 0.9) rtol = 1e-6    # the norm stays in the argument's type
  end
end

@testset "the same type: the closed form of GroupNormL2" begin
  for T in (Float32, Float64)
    x = T[1.5, -0.5, 2.0]
    f = GroupNormL2(T(0.7), [1:2, 3:3])
    y = zeros(T, 3)
    prox!(y, f, x, T(0.9))
    c = T(0.9) * T(0.7) * sqrt(T(2))              # the factors formed in T, w = √|G|
    n1 = sqrt(x[1]^2 + x[2]^2)
    @test y[1:2] ≈ max(1 - c / n1, 0) .* x[1:2] rtol = 4eps(T)
    @test y[3] ≈ max(1 - T(0.9) * T(0.7) / abs(x[3]), 0) * x[3] rtol = 4eps(T)
  end
end

@testset "group sphere selects with the threshold of the group prox" begin
  # margins -0.5 and -0.5 + 2^-30 in Float64; a threshold rounded to Float32 first ties them
  f = GroupNormL2(1.0, [1:1, 2:2], [1.0 + 2.0^-30, 1.0])
  c = Constrained(f, IndSphereL2(2.0))
  x = [0.5, 0.5]
  @test prox(c, x, 1.0) == [0.0, 2.0]
  y = zeros(Float32, 2)
  prox!(y, c, x, 1.0)
  @test y == Float32[0, 2]
end
end
