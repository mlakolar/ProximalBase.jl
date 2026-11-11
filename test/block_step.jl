module BlockStepTest
using Test, ProximalBase
using ..TestUtils

struct SquareBlocks <: ProximalBase.AbstractGroupPenalty
  groups::Vector{Vector{Int}}
end
function ProximalBase.prox_block!(out::AbstractVector, f::SquareBlocks, u::AbstractVector, k::Integer, γ::Real)
  checkbounds(f.groups,k)
  length(out) == length(u) == length(f.groups[k]) || throw(DimensionMismatch())
  out .= u ./ (1 + γ)
end

@testset "public group extension through wrappers" begin
  f = SquareBlocks([[1,3],[2,4]])
  u = [3.,-4.]
  @test prox_block(f,u,1,.5) ≈ u/1.5
  @test prox_block(Scaled(f,2.),u,1,.5) ≈ u/2
  @test prox_block(Regularize(f,1.),u,1,1.) ≈ u/3
  @test prox_block(Regularize(Scaled(f,2.),1.),u,1,1.) ≈ u/4
  nested = Regularize(Scaled(Regularize(f,.2),1.3),.4)
  # The extension is a unit quadratic, so all wrappers combine into one level.
  @test prox_block(nested,u,1,.7) ≈ u/(1 + .7*(1.3*1.2 + .4))
end

@testset "pure block oracle" begin
  @test !isdefined(ProximalBase, :prox_entry)
  @test !isdefined(ProximalBase, :prox_coordinate!)
  @test !isdefined(ProximalBase, :cdprox!)
  for T in (Float32, Float64), gs in ([1:2, 3:4], [[4, 2], [3, 1]])
    x = T[3, -4, 2, 8]
    fs = (GroupNormL2(T(.3), gs, T[1, 2]), GroupNormLinf(T(.3), gs, T[1, 2]),
          SparseGroupNormL2(T(.1), T(.3), gs, T[1, 2]),
          GroupMCP(T(.3), gs, T[1, 2]), GroupSCAD(T(.3), gs, T[1, 2]))
    for f in fs, h in (f, Scaled(f, T(2)), Regularize(f, T(.2)), Regularize(f, T(.2), T[1, 2, 3, 4]))
      reference = prox(h, x, T(.7))
      for k in eachindex(gs)
        u = x[gs[k]]
        before = copy(u)
        out = similar(u)
        @test prox_block!(out, h, u, k, T(.7)) === out
        @test out ≈ reference[gs[k]]
        @test u == before
        @test prox_block(h, u, k, T(.7)) ≈ out
        @test prox_block!(u, h, u, k, T(.7)) === u
        @test u ≈ out
      end
    end
  end
end

@testset "function entry versus symmetric coordinate" begin
  p = 3
  W = [1. 2. 3.; 2. 4. 5.; 3. 5. 6.]
  f = NormL1(.1, W)
  x = SymmetricSparseIterate(p)
  ProximalBase.check_symmetric_parameters(f, x)
  for k in 1:ncoordinates(x)
    r, c = tril_cartesian(p, k)
    i = (c - 1) * p + r
    v = prox_block(f, 2., i, 1.)
    @test v ≈ 2 - .1W[r,c]
    coordinates(x)[k] = v
  end
  @test Matrix(x) ≈ 2 .- .1W
  k = tril_index(3, 3, 2)
  @test k != 6
  @test prox_block(f, 2., k, 1.) != prox_block(f, 2., 6, 1.)
end

@testset "math and public storage writes" begin
  gs = [[1, 3], [2, 4]]
  f = GroupNormL2(1., gs, ones(2))
  x = BlockIterate(4, gs)
  activate!(x, 2)
  compact_owner = BlockIterate([3.,4.], [1:2])
  allocated = prox_block(f,compact_owner,1,1.)
  @test allocated isa Vector{Float64}
  @test Vector(compact_owner) == [3.,4.]
  out = zeros(2)
  for (u, active) in (([.1,.1], false), ([3.,4.], true), ([0.,0.], true))
    old = (Vector(x), collect(activecoordinates(x)))
    prox_block!(out, f, u, 1, 1.)
    @test (Vector(x), collect(activecoordinates(x))) == old
    copyto!(blockview(x, 1), out)
    @test isactive(x, 1) == active
    ProximalBase.Testing.test_iterate(x)
  end
end
@testset "independent extreme group reference" begin
  f = GroupNormL2(1e155, [1:2], [1.])
  @test prox_block(f, [1e155, 0.], 1, .5) ≈ [5e154, 0.]
  for n in (1, 2, 5, 32, 256)
    gs = [collect(2:2:2n), collect(1:2:2n)]
    x = [sin(Float64(i)) for i in 1:2n]
    for f in (GroupNormL2(.3, gs), GroupNormLinf(.3, gs),
              SparseGroupNormL2(.1,.3,gs), GroupMCP(.3,gs), GroupSCAD(.3,gs))
      u = x[gs[1]]
      out = zeros(Float32, n)
      full = zeros(Float32, length(x))
      prox!(full, f, x, .7)
      prox_block!(out, f, u, 1, .7)
      @test out ≈ full[gs[1]]
    end
  end
end

end
