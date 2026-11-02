module ThresholdPrecisionTest
using Test, ProximalBase
using ..TestUtils

# The threshold γλ0w is formed in the promotion of the working type and the factors' types and
# rounded once (numerics T3). Float32 factors widen exactly, so a penalty with Float32 factors
# must act on a Float64 argument exactly as the same penalty with Float64 factors does, also where
# the product overflows or underflows Float32.

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
  @testset "Float32 argument, Float64 factors" begin
    for (label, f32, f64) in penalties(Float64(2f0^80), ones(Float32, 3), ones(Float32, 2), 3.0)
      @test prox(f64, Float32[2^100, 1, 2^90], 2.0^-60) isa Vector{Float32}
    end
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
end
