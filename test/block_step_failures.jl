module BlockStepFailuresTest
using Test, ProximalBase

struct Flaky <: AbstractVector{Float64} end
Base.size(::Flaky) = (2,)
Base.getindex(::Flaky, i::Int) = i == 1 ? 3. : error("boom")

@testset "block preflight before writes" begin
  f = GroupNormL2(1., [[1,3], [2,4]])
  for (u, k, E) in ((ones(3), 1, DimensionMismatch), (ones(2), 0, BoundsError),
                    (ones(2), 3, BoundsError), (ones(ComplexF64,2), 1, ArgumentError))
    out = fill(-7., 2)
    @test_throws E prox_block!(out, f, u, k, 1.)
    @test out == [-7.,-7.]
  end
  out = ones(ComplexF64, 2)
  @test_throws ArgumentError prox_block!(out, f, ones(2), 1, 1.)
  @test out == ones(ComplexF64, 2)
  buf = [1.,2.,3.]
  @test_throws ArgumentError prox_block!(view(buf,1:2), f, view(buf,2:3), 1, 1.)
  @test buf == [1.,2.,3.]
  for owner in (AtomIterate(2, [1:2]), SparseIterate(2))
    @test_throws ArgumentError prox_block!(owner, f, ones(2), 1, 1.)
    @test_throws ArgumentError prox_block!(view(owner, 1:2), f, ones(2), 1, 1.)
    @test nactive(owner) == 0
  end
  @test_throws MethodError prox_block(SLOPE([1.,.5]), 2., 1, 1.)
  @test_throws ArgumentError prox_block!(zeros(2), NormL1(), ones(2), 1, 1.)
end

@testset "element failures affect only the numerical buffer" begin
  f = Regularize(GroupNormL2(1., [1:2]), 1., Flaky())
  out = zeros(2)
  owner = AtomIterate(2, [1:2])
  @test_throws ErrorException("boom") prox_block!(out, f, zeros(2), 1, 1.)
  @test out == [1.5,0.]
  @test nactive(owner) == 0 && all(iszero, owner)
end
end
