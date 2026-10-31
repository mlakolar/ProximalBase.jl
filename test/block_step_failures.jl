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

  @testset "reference-valued trial points" begin
    x = AtomIterate(2, [1:2])
    f = GroupNormL2(1., x.groups)
    u = Vector{BigInt}(undef, 2)
    u[1] = big(3)
    @test_throws UndefRefError prox_coordinate!(x, f, 1, u, 1.)
    @test x.storage == [3., 0.]
    @test isactive(x, 1) && consistent(x)

    x = AtomIterate(2, [1:2])
    y = copy(x)
    u = BigFloat[3, 4]
    before = copy(u)
    result = prox_coordinate!(x, f, 1, u, 1.)
    reference = prox_coordinate!(y, f, 1, Float64.(u), 1.)
    @test result == reference && x == y
    @test activecoordinates(x) == activecoordinates(y) && consistent(x)
    @test u == before
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

  @testset "finite group results settle activation" begin
    for T in (Float32, Float64)
      gs = [1:2, 3:4, 5:6]
      fs = (GroupNormL2(one(T), gs), GroupNormLinf(one(T), gs),
            SparseGroupNormL2(T(0.2), one(T), gs), GroupMCP(one(T), gs), GroupSCAD(one(T), gs))
      for f in fs, h in (f, Regularize(f, one(T)))
        for (u, nonzero) in ((T[0.01, 0.01], false), (T[10, -10], true))
          x = AtomIterate(T, 6, gs)
          activate!(x, 3)
          activate!(x, 2)
          prox_coordinate!(x, h, 1, u, one(T))
          @test isactive(x, 1) == nonzero
          @test any(!iszero, atom(x, 1)) == nonzero
          @test activecoordinates(x) == (nonzero ? [3, 2, 1] : [3, 2])
          @test consistent(x)
        end
      end
    end
  end

  @testset "subnormal stored zeros stay inactive" begin
    q = nextfloat(0f0)
    x = AtomIterate(Float32, 4, [1:4])
    prox_coordinate!(x, GroupNormL2(3q, x.groups, Float32[1]), 1, fill(2q, 4), 1f0)
    @test all(iszero, x.storage) && !isactive(x, 1) && consistent(x)
    x = AtomIterate(Float32, 4, [1:4])
    r = Regularize(GroupNormL2(6q, x.groups, Float32[1]), 1f0, fill(4q, 4))
    prox_coordinate!(x, r, 1, zeros(Float32, 4), 1f0)
    @test all(iszero, x.storage) && !isactive(x, 1) && consistent(x)
  end

  @testset "nonfinite stored results activate" begin
    x = AtomIterate(Float32, 2, [1:2])
    f = GroupMCP(1f0, x.groups, Float32[1])
    prox_coordinate!(x, f, 1, [1e100, 0.0], 3f0)
    @test isnan(x.storage[1]) && iszero(x.storage[2])
    @test isactive(x, 1) && consistent(x)
  end
end

@testset "block steps check symmetric parameters before writing" begin
  A = ones(3, 3)
  A[2, 1] = nextfloat(1.)
  f = NormL1(.1, A)
  k = tril_index(3, 2, 1)
  x = SymmetricSparseIterate(3)
  e4 = try prox_coordinate!(copy(x), f, k, .2); nothing catch e; e end
  e5 = try prox_coordinate!(copy(x), f, k, 1., .2); nothing catch e; e end
  @test e4 isa ArgumentError && typeof(e5) == typeof(e4)
  @test sprint(showerror, e5) == sprint(showerror, e4)
  y = copy(x)
  @test_throws ArgumentError prox_coordinate!(y, f, k, 1., .2)
  @test Matrix(y) == Matrix(x) && nnz(y) == 0
end

end
