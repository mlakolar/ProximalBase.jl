module ThresholdPrecisionTest

using ..TestUtils: take_block_step!
using Test, ProximalBase
using ..TestUtils

# The threshold γλ0w is formed in the promotion of the working type and the factors' types and
# rounded once (numerics T3). Float32 factors widen exactly, so a penalty with Float32 factors
# must act on a Float64 argument exactly as the same penalty with Float64 factors does, also where
# the product overflows or underflows Float32.

active_set(x::AtomIterate) = Set(x.active[1:x.nactive])   # as in atom_iterate.jl

big32 = 2f0^80
tiny32 = 2f0^-80
groups = [1:2, 3:3]

# (label, Float32-factor penalty, Float64-factor penalty, γ)
function penalties(λ, w1, w2, a)
  λ64, w164, w264 = Float64(λ), Float64.(w1), Float64.(w2)
  [("NormL1", NormL1(λ, w1), NormL1(λ64, w164)),
   ("NormL1Nonneg", NormL1Nonneg(λ, w1), NormL1Nonneg(λ64, w164)),
   ("GroupNormL2", GroupNormL2(λ, groups, w2), GroupNormL2(λ64, groups, w264)),
   ("GroupNormL2 unit weights", GroupNormL2(λ, groups, ones(Float32, 2)), GroupNormL2(λ64, groups, ones(2))),
   ("GroupNormLinf", GroupNormLinf(λ, groups, w2), GroupNormLinf(λ64, groups, w264)),
   ("SparseGroupNormL2", SparseGroupNormL2(0f0, λ, groups, w2), SparseGroupNormL2(0.0, λ64, groups, w264)),
   ("GroupMCP", GroupMCP(λ, groups, w2; a = a), GroupMCP(λ64, groups, w264; a = a)),
   ("GroupSCAD", GroupSCAD(λ, groups, w2; a = 4 * a), GroupSCAD(λ64, groups, w264; a = 4 * a))]
end

function agree(f32, f64, x, γ)
  y32, y64 = similar(x), similar(x)
  @test prox(f32, x, γ) == prox(f64, x, γ)
  v32 = value_and_prox!(y32, f32, x, γ)
  v64 = value_and_prox!(y64, f64, x, γ)
  @test y32 == y64
  @test v32 == v64 || (isnan(v32) && isnan(v64))
end

@testset "threshold products in the working precision" begin
  @testset "the reported case" begin
    f = NormL1(big32, ones(Float32, 2))
    @test prox(f, [2.0^161, 1.0], big32) == [2.0^160, 0.0]
    @test prox(NormL1Nonneg(big32, ones(Float32, 2)), [2.0^161, 1.0], big32) == [2.0^160, 0.0]
    g = GroupNormL2(big32, [1:2], ones(Float32, 1))
    @test prox(g, [2.0^161, 1.0], big32) == [2.0^160, 0.5]
  end
  @testset "Float32 factors on a Float64 argument, overflow and underflow of Float32" begin
    for (λ, γ, x) in ((big32, big32, [2.0^161, 1.0, 2.0^159]), (big32, 1f0, [2.0^81, 1.0, 2.0^79]),
                      (tiny32, tiny32, [2.0^-161, 2.0^-170, 2.0^-159]), (tiny32, 1f0, [2.0^-79, 2.0^-90, 3.0^-60]),
                      (2f0, 1.5f0, [3.0, -1.0, 0.25]))
      for w2 in (ones(Float32, 2), Float32[big32, tiny32], Float32[0, Inf])
        for (label, f32, f64) in penalties(λ, ones(Float32, 3), w2, 3.0)
          @testset "$label λ=$λ γ=$γ w=$w2" begin
            agree(f32, f64, x, γ)
            agree(f32, f64, -x, γ)
          end
        end
      end
    end
  end
  @testset "Float32 destination, Float64 factors whose product overflows Float32" begin
    # γλ0 = 2^200 is Inf in Float32; times w = 2^-150 the threshold is 2^50
    γ, λ, w = 2.0^100, 2.0^100, 2.0^-150
    x = Float32[2^60, -2^60, 1]
    for f in (NormL1(λ, fill(w, 3)), NormL1Nonneg(λ, fill(w, 3)))
      @test prox(f, x, γ)[1] == Float32(2^60 - 2^50)
      @test prox(f, x, γ)[3] == 0
    end
    g = GroupNormL2(λ, [1:2, 3:3], fill(w, 2))
    y = prox(g, x, γ)
    @test y isa Vector{Float32}
    @test y == Float32.(prox(g, Float64.(x), γ))
  end
  @testset "per-entry weights, also through Constrained" begin
    # thresholds γλ0w = 2^240, 2^80, 2^160 (the first two overflow and underflow Float32 products)
    w = Float32[big32, tiny32, 1]
    x = [2.0^241, 1.0, 2.0^161]
    expected = [2.0^240, 0.0, 2.0^160]
    C = IndBox(-2.0^250, 2.0^250)
    for (f, g) in ((NormL1(big32, w), NormL1(Float64(big32), Float64.(w))),
                   (NormL1Nonneg(big32, w), NormL1Nonneg(Float64(big32), Float64.(w))))
      for h in (f, g, Constrained(f, C), Constrained(g, C))
        @test prox(h, x, big32) == expected
        y = similar(x); prox!(y, h, x, big32); @test y == expected
      end
    end
    @test prox(NormL1(big32, w), -x, big32) == -expected
    @test prox(Constrained(NormL1(big32, w), C), -x, big32) == -expected
    @test prox(NormL1Nonneg(big32, w), -x, big32) == zeros(3)
    @test prox(Constrained(NormL1Nonneg(big32, w), C), -x, big32) == zeros(3)
    # an infinite weight fixes the entry at zero, a zero weight leaves it alone
    @test prox(NormL1(big32, Float32[Inf, 0, 1]), x, big32) == [0.0, 1.0, 2.0^160]
  end
  @testset "sphere paths" begin
    x = [2.0^161, 1.0, 2.0^159]
    for (f32, f64) in ((NormL1(big32, ones(Float32, 3)), NormL1(Float64(big32), ones(3))),
                       (GroupNormL2(big32, groups, ones(Float32, 2)), GroupNormL2(Float64(big32), groups, ones(2))))
      s32 = ProximalBase.SphereConstrained(f32, IndSphereL2(1.0))
      s64 = ProximalBase.SphereConstrained(f64, IndSphereL2(1.0))
      @test prox(s32, x, big32) == prox(s64, x, big32)
    end
  end
end

# Analytic results for the groups [2^161, 0] and [-2^161] at the threshold γλ0w = 2^160, a product
# that overflows Float32 (2^80 · 2^80): the norms shrink to 2^160, and the MCP and SCAD kernels
# follow their piecewise formulas. (label, constructor, weights, step, expected first entry)
# within 4 ulps (the MCP formula divides by 2/3); zeros must be exact
close(a, b) = isapprox(a, b; rtol = 4eps(), atol = 0)

const GROUP_CASES = (
  ("GroupNormL2", (T, W) -> GroupNormL2(T(big32), [1:2, 3:3], W), ones(Float32, 2), big32, 2.0^160),
  ("GroupNormLinf", (T, W) -> GroupNormLinf(T(big32), [1:2, 3:3], W), ones(Float32, 2), big32, 2.0^160),
  ("SparseGroupNormL2", (T, W) -> SparseGroupNormL2(zero(T), T(big32), [1:2, 3:3], W), ones(Float32, 2), big32, 2.0^160),
  # MCP, a = 3, λ = 2^160, γ = 1: (|x| - λ) / (1 - γ/a) between λ and aλ
  ("GroupMCP", (T, W) -> GroupMCP(T(big32), [1:2, 3:3], W; a = 3.0), fill(big32, 2), 1f0, 1.5 * 2.0^160),
  # SCAD, a = 4: at |x| = (1 + γ)λ the soft threshold |x| - γλ applies
  ("GroupSCAD", (T, W) -> GroupSCAD(T(big32), [1:2, 3:3], W; a = 4.0), fill(big32, 2), 1f0, 2.0^160))

@testset "prox! with Float32 factors into a Float64 output" begin
  x = [2.0^161, 0.0, -2.0^161]
  for (label, make, W, γ, e) in GROUP_CASES
    @testset "$label" begin
      f = make(Float32, W)
      y = zeros(3)
      @test prox!(y, f, x, γ) === y
      @test close(y, [e, 0.0, -e])
      z = copy(x)
      prox!(z, f, z, γ)                                # in place
      @test z == y
      y2 = zeros(3)
      v = value_and_prox!(y2, f, x, γ)
      @test y2 == y
      @test v == f(y2)
    end
  end
  # a Float32 destination for wide inputs and parameters whose product overflows Float32: the
  # threshold is 2^100 · 2^100 · 2^-150 = 2^50 and the argument is 2^60
  x = [2.0^60, 0.0]
  for f in (GroupNormL2(2.0^100, [1:2], [2.0^-150]), GroupNormLinf(2.0^100, [1:2], [2.0^-150]),
            SparseGroupNormL2(0.0, 2.0^100, [1:2], [2.0^-150]))
    y = zeros(Float32, 2)
    prox!(y, f, x, 2.0^100)
    @test y == Float32[2^60 - 2^50, 0]
  end
end

@testset "AtomIterate with Float32 factors" begin
  groups = [1:2, 3:3]
  v = [2.0^161, 0.0, -2.0^161]
  for (label, make, W, γ, e) in GROUP_CASES
    @testset "$label" begin
      f = make(Float32, W)
      x = AtomIterate(v, groups)
      @test active_set(x) == Set([1, 2])
      y = AtomIterate(Float64, 3, groups)
      val = value_and_prox!(y, f, x, γ)
      @test close(Vector(y), [e, 0.0, -e])
      @test active_set(y) == Set([1, 2])
      @test val == f(y)
      z = AtomIterate(v, groups)                       # in place
      @test prox!(z, f, z, γ) === z
      @test close(Vector(z), [e, 0.0, -e])
      @test active_set(z) == Set([1, 2])
      # coordinate steps: an active group, then a group below the threshold, then an inactive one
      z = AtomIterate(v, groups)
      take_block_step!(z, f, 1, γ)
      @test close(Vector(z), [e, 0.0, -2.0^161]) && active_set(z) == Set([1, 2])
      take_block_step!(z, f, 2, γ)
      @test close(Vector(z), [e, 0.0, -e])
      small = AtomIterate([1.0, 0.0, 0.0], groups)
      take_block_step!(small, f, 1, γ)
      @test Vector(small) == zeros(3) && 2 ∉ active_set(small)
      take_block_step!(small, f, 2, γ)                 # an inactive group stays inactive
      @test Vector(small) == zeros(3) && 2 ∉ active_set(small)
    end
  end
end
end
