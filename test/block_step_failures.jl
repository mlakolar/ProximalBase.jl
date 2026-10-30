module BlockStepFailuresTest

using Test
using ProximalBase

struct Flaky <: AbstractVector{Float64} end
Base.size(::Flaky) = (2,)
Base.getindex(::Flaky, i::Int) = i == 1 ? 3.0 : error("boom")

struct ThrowingGroupPenalty <: ProximalBase.AbstractGroupPenalty
  groups::Vector{UnitRange{Int}}
end
function ProximalBase.prox_coordinate!(x::AtomIterate, f::ThrowingGroupPenalty, k::Integer, γ::Real)
  ProximalBase._check_group_coordinate(f, x, k)
  x.storage[first(x.groups[k])] = 7.
  error("step boom")
end

function consistent(x::AtomIterate)
  act = collect(activecoordinates(x))
  length(unique(act)) == nactive(x) &&
    count(!iszero, x.position) == nactive(x) &&
    all(a -> x.position[act[a]] == a, eachindex(act)) &&
    all(k -> isactive(x, k) || all(iszero, x.storage[x.groups[k]]), eachindex(x.groups))
end

@testset "block step failure consistency" begin
  @testset "reshaped storage aliases" begin
    x = AtomIterate([1., 2., 3., 4.], [[2, 1], [3, 4]])
    y = copy(x)
    u = view(reshape(x.storage, 2, 2), :, 1)
    prox_coordinate!(y, NormL1(0.), 1, copy(u), 1.)
    prox_coordinate!(x, NormL1(0.), 1, u, 1.)
    @test x == y && x.storage == [2., 1., 3., 4.]
    @test activecoordinates(x) == activecoordinates(y) && consistent(x)
  end

  @testset "throwing regularization centers" begin
    for trial in (false, true)
      x = AtomIterate(2, [1:2])
      r = Regularize(GroupNormL2(1., x.groups), 1., Flaky())
      if trial
        @test_throws ErrorException("boom") prox_coordinate!(x, r, 1, zeros(2), 1.)
      else
        @test_throws ErrorException("boom") prox_coordinate!(x, r, 1, 1.)
      end
      @test x.storage == [1.5, 0.]
      @test isactive(x, 1) && consistent(x)
    end
  end

  @testset "throwing custom group step" begin
    x = AtomIterate(2, [1:2])
    r = Regularize(ThrowingGroupPenalty(x.groups), 1., ones(2))
    @test_throws ErrorException("step boom") prox_coordinate!(x, r, 1, 1.)
    @test x.storage == [7., 0.5]
    @test isactive(x, 1) && consistent(x)
  end

  @testset "nonfinite stored results activate" begin
    x = AtomIterate(Float32, 2, [1:2])
    f = GroupMCP(1f0, x.groups, Float32[1])
    prox_coordinate!(x, f, 1, [1e100, 0.0], 3f0)
    @test isnan(x.storage[1]) && iszero(x.storage[2])
    @test isactive(x, 1) && consistent(x)
  end
end

end
