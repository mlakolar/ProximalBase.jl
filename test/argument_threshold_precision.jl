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

@testset "a positive margin that rounds to zero in the destination" begin
  # the group sphere has no stored direction to scale: it takes the direction of x
  c = Constrained(GroupNormL2(1.0, [1:1], [1.0]), IndSphereL2(1.0))
  x = [2.0^-100]
  γ = 2.0^-100 * (1.0 - 2.0^-52)                  # the threshold is below the norm by 2^-152
  y = zeros(Float32, 1)
  prox!(y, c, x, γ)
  @test y == Float32[1.0]
  y = zeros(Float32, 1)
  @test value_and_prox!(y, c, x, γ) === 1f0
  @test y == Float32[1.0]
  # every positive group keeps its share, also when each shrinkage rounds to zero in the destination
  c3 = Constrained(GroupNormL2(1.0, [1:1, 2:2], [1.0, 1.0]), IndSphereL2(1.0))
  x3 = [2.0^-100, 2.0^-100]
  γ3 = 2.0^-100 * (1.0 - 2.0^-52)
  y3 = zeros(Float32, 2)
  prox!(y3, c3, x3, γ3)
  @test y3 ≈ Float32[inv(sqrt(2)), inv(sqrt(2))] rtol = 1e-6
  y3 = zeros(Float32, 2)
  v3 = value_and_prox!(y3, c3, x3, γ3)
  @test y3 ≈ Float32[inv(sqrt(2)), inv(sqrt(2))] rtol = 1e-6
  @test sqrt(sum(abs2, Float64.(y3))) ≈ 1 rtol = 1e-6
  @test v3 == c3.f(y3)                              # the value at the stored output
  # one group whose norm underflows in the destination keeps its direction
  c4 = Constrained(GroupNormL2(1.0, [1:2], [1.0]), IndSphereL2(1.0))
  y4 = zeros(Float32, 2)
  prox!(y4, c4, [3.0, 4.0] .* 2.0^-160, 5.0 * 2.0^-160 * (1.0 - 2.0^-52))
  @test y4 ≈ Float32[0.6, 0.8] rtol = 1e-6
  # the step rules of the sphere hold with the threshold formed in the model precision
  @test_throws ArgumentError prox!(zeros(Float32, 2), c4, [3.0, 4.0], Inf)
  @test_throws ArgumentError prox!(zeros(Float32, 2), c4, [3.0, 4.0], -1.0)
  # two groups: the sphere point lies in the group of the larger margin, with norm r
  c2 = Constrained(GroupNormL2(1.0, [1:2, 3:3], [1.0, 1.0]), IndSphereL2(2.0))
  x2 = [3.0 * 2.0^-100, -4.0 * 2.0^-100, 2.0^-101]
  y2 = zeros(Float32, 3)
  prox!(y2, c2, x2, 2.0^-100)
  @test y2 ≈ Float32[1.2, -1.6, 0] rtol = 1e-6
end

nrm2(y) = sqrt(sum(abs2, Float64.(y)))

# Every destination array takes the wide recovery: a view of a Float32 buffer, a strided one, a
# noncontiguous one. The shrinkages are all zero in Float32; the direction is analytic.
@testset "wide sphere recovery for any destination array" begin
  s = 2.0^-100
  destinations = (
    "Array" => n -> zeros(Float32, n),
    "full view" => n -> view(zeros(Float32, n), :),
    "strided view" => n -> view(zeros(Float32, 2n), 1:2:2n-1),
    "noncontiguous view" => n -> view(zeros(Float32, 3, 2n), 2, 1:2:2n-1),
    "reshaped view" => n -> vec(reshape(view(zeros(Float32, 2n), 1:n), 1, n)))
  # two positive margins, equal after rounding: the direction is x/‖x‖
  c = Constrained(GroupNormL2(1.0, [1:1, 2:2], [1.0, 1.0]), IndSphereL2(1.0))
  x = [s, s]
  γ = s * (1.0 - 2.0^-52)
  # three entries in two groups, a multicoordinate group among them: z ∝ x, y = r x/‖x‖
  c3 = Constrained(GroupNormL2(1.0, [1:2, 3:3], [1.0, 1.0]), IndSphereL2(2.0))
  # (a relative margin of 2^-50, so that the rounded norm of a group cannot reach the threshold)
  s3 = 2.0^-110
  x3 = [3s3, 4s3, 5s3]
  γ3 = 5s3 * (1.0 - 2.0^-50)
  expected3 = Float32.(2 .* [3, 4, 5] ./ sqrt(50.0))
  # one group of two entries whose norm underflows in Float32 (z has norm about 2^-210)
  c1 = Constrained(GroupNormL2(1.0, [1:2], [1.0]), IndSphereL2(1.0))
  x1 = [3.0, 4.0] .* 2.0^-160
  γ1 = 5.0 * 2.0^-160 * (1.0 - 2.0^-52)
  for (name, make) in destinations
    @testset "$name" begin
      y = make(2); prox!(y, c, x, γ)
      @test collect(y) ≈ Float32[inv(sqrt(2)), inv(sqrt(2))] rtol = 1e-6
      y = make(2); v = value_and_prox!(y, c, x, γ)
      @test collect(y) ≈ Float32[inv(sqrt(2)), inv(sqrt(2))] rtol = 1e-6
      @test nrm2(y) ≈ 1 rtol = 1e-6
      @test v == c.f(y)                                  # O2: the value at the stored output
      y = make(3); prox!(y, c3, x3, γ3)
      @test collect(y) ≈ expected3 rtol = 1e-6
      y = make(3); v = value_and_prox!(y, c3, x3, γ3)
      @test collect(y) ≈ expected3 rtol = 1e-6
      @test nrm2(y) ≈ 2 rtol = 1e-6
      @test v == c3.f(y)
      y = make(2); prox!(y, c1, x1, γ1)
      @test collect(y) ≈ Float32[0.6, 0.8] rtol = 1e-6
      y = make(2); v = value_and_prox!(y, c1, x1, γ1)
      @test collect(y) ≈ Float32[0.6, 0.8] rtol = 1e-6
      @test v == c1.f(y)
    end
  end
  # a shrinkage that stores in the subnormal range of Float32 has too few bits for a direction
  cs = Constrained(GroupNormL2(1.0, [1:2], [1.0]), IndSphereL2(1.0))
  xs = [0.6, 0.8] .* s
  γs = sqrt(xs[1]^2 + xs[2]^2) * (1 - 2.0^-48)    # z = [1.2, 1.6] ⋅ 2^-149 stores as [1, 2] ⋅ 2^-149
  for (name, make) in destinations
    y = make(2); prox!(y, cs, xs, γs)
    @test collect(y) ≈ Float32[0.6, 0.8] rtol = 1e-6
    y = make(2); v = value_and_prox!(y, cs, xs, γs)
    @test collect(y) ≈ Float32[0.6, 0.8] rtol = 1e-6
    @test v == cs.f(y)
  end
  # the same type is unchanged: no wide buffer, the kernel's own rounding
  yd = zeros(2); prox!(yd, c, [1.0, 1.0], 0.25)
  @test yd ≈ [inv(sqrt(2)), inv(sqrt(2))]
  yf = zeros(Float32, 2); prox!(yf, c, Float32[1, 1], 0.25f0)
  @test yf ≈ Float32[inv(sqrt(2)), inv(sqrt(2))] rtol = 1e-6
  # in place, with a view of the argument's own storage: the projection stays on the sphere
  b = [3.0, 4.0, 0.5]
  yv = view(b, 1:2)
  prox!(yv, c1, yv, 0.5)
  @test nrm2(yv) ≈ 1
  @test yv ≈ [0.6, 0.8]
end

@testset "value_and_prox! returns the value of the stored output (O2)" begin
  x = [2.0^-100]
  γ = 2.0^-225 * (1.0 - 2.0^-52)                  # the Float64 residual 2^-152 stores as zero
  for f in (GroupNormL2(2.0^125, [1:1], [1.0]), GroupNormLinf(2.0^125, [1:1], [1.0]),
            SparseGroupNormL2(2.0^125, 0.0, [1:1], [1.0]), SparseGroupNormL2(2.0^125, 1.0, [1:1], [1.0]),
            SparseGroupNormL2(0.0, 2.0^125, [1:1], [1.0]),
            GroupMCP(2.0^125, [1:1], [1.0]; a = 3.0), GroupSCAD(2.0^125, [1:1], [1.0]; a = 4.0))
    y = zeros(Float32, 1)
    v = value_and_prox!(y, f, x, γ)
    @test y == Float32[0]
    @test v === f(y) === 0f0
  end
  # a nonzero stored output keeps the value of the stored output
  f = GroupNormL2(3.0, [1:1], [1.0])
  y = zeros(Float32, 1)
  v = value_and_prox!(y, f, [1.0 + 2.0^-28], 1.0 - 2.0^-28)
  @test v == f(y)
end
end
