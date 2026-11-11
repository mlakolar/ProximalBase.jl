module BlockIterateGridsTest

using ..TestUtils: take_block_step!

# Cases of block_iterate.jl that repeat one that stays there for another element type or group
# function: the mixed Float32 x Float64 triple and the Float32 strided views of the five-argument
# products, the group wrappers of GroupNormLinf and GroupSCAD, and the Float32 pass of the group view
# norms and dots. The rest of each testset is in block_iterate.jl; the helpers are copied from it.

using Test
using ProximalBase
using Random
using LinearAlgebra
using SparseArrays

Random.seed!(1)

# groups of unequal sizes, two not contiguous
groups = [1:3, [4, 7], 5:5, [8, 6], [9]]


function check_scaled_product(out, A, x, α, β)
  @nospecialize out A x α β
  expected = Array(out)
  mul!(expected, A, Array(x), α, β)
  @test parentmodule(which(mul!, typeof.((out, A, x, α, β)))) == ProximalBase
  @test mul!(out, A, x, α, β) === out
  @test out ≈ expected
end

function check_product_state(x, before)
  @nospecialize x before
  @test x.storage == before.storage
  @test activecoordinates(x) == activecoordinates(before)
  @test x.position == before.position
end

@testset "BlockIterate" begin

  @testset "five-argument products" begin
    n, p = 7, 9
    types = ((Float32, Float32, Float32), (Float32, Float64, Float64))
    for (Ta, Tx, To) in types, support in (:empty, :partial, :full), strided in (false, true)
      strided && !(Ta == Tx == To && Ta <: Real) && continue
      # the homogeneous Float32 triple is in block_iterate.jl, except its strided views
      Ta == Tx == To && !strided && continue
      @testset "$Ta × $Tx → $To, $support, views=$strided" begin
        A = strided ? view(randn(Ta, 2n, 2p), 1:2:2n, 1:2:2p) : randn(Ta, n, p)
        dense = randn(Tx, p)
        support == :empty && fill!(dense, 0)
        support == :partial && (dense[1:2:end] .= 0)
        for G in (groups,)
          x = BlockIterate(dense, G)
          support == :partial && activate!(x, length(G))
          before = copy(x)
          out = strided ? view(randn(To, 2n), 1:2:2n) : randn(To, n)
          initial = copy(out)
          S = Ta
          for α in S.((0, 1, 2.5)), β in S.((0, 1, -0.5))
            copyto!(out, initial)
            iszero(β) && fill!(out, NaN)
            check_scaled_product(out, A, x, α, β)
          end
          if support == :partial && !strided
            for (α, β) in ((true, false), (0, 0))
              fill!(out, NaN)
              check_scaled_product(out, A, x, α, β)
            end
          end
          check_product_state(x, before)
        end
      end
    end
  end

  @testset "group wrappers preserve a failed coordinate step" begin
    gs = [1:2, 3:4, 5:6]
    other = [[1, 3], [2, 5], [4, 6]]
    v = [3., -4., 0., 0., 0., 0.]
    for F in (GroupNormLinf, GroupSCAD)
      f = F === SparseGroupNormL2 ? F(0.2, 0.4, gs) : F(0.4, gs)
      bad = F === SparseGroupNormL2 ? F(0.2, 0.4, other) : F(0.4, other)
      for wrap in (h -> 2.0 * h, h -> Regularize(h, 0.3, ones(6)))
        x = BlockIterate(v, gs)
        activate!(x, 3)
        take_block_step!(x, wrap(f), 1, 0.7)
        reference = prox(wrap(f), v, 0.7)
        @test Vector(x)[gs[1]] ≈ reference[gs[1]]
        @test Vector(x)[3:6] == v[3:6]
        before = copy(x)
        @test_throws DimensionMismatch take_block_step!(x, wrap(bad), 1, 0.7)
        @test_throws DimensionMismatch kkt_violation(wrap(bad), ones(6), x, 1)
        @test x.storage == before.storage
        @test activecoordinates(x) == activecoordinates(before)
        @test x.position == before.position && x.layout_cache === f.groups
        take_block_step!(x, wrap(f), 1, 0.7)
        @test Vector(x)[gs[1]] ≈ prox(wrap(f), Vector(before), 0.7)[gs[1]]
      end
    end
  end

  @testset "group view norms and dots read storage" begin
    for T in (Float32,), m in (4, 64), gs in ([1:m, (m+1):2m], [collect(1:2:2m), collect(2:2:2m)])
      x = BlockIterate(randn(T, 2m), gs)
      v, w = blockview(x, 1), blockview(x, 2)
      rawv, raww = view(x.storage, gs[1]), view(x.storage, gs[2])
      z = randn(T, m)
      for p in (1, 2, Inf)
        @test norm(v, p) === norm(rawv, p)
      end
      @test norm(v) === norm(rawv)
      @test dot(v, z) === dot(rawv, z)
      @test dot(z, v) === dot(z, rawv)
      @test dot(v, w) === dot(rawv, raww)
    end
  end

end

end
