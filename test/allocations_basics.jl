module AllocationsBasicsTest

# Quick-tier zero-allocation checks of the hot calls a solver makes: block and coordinate steps on
# sparse, symmetric and atom iterates, prox!/value_and_prox! of NormL1 and GroupNormL2 (also on atom
# iterates with distinct group layouts), and value_and_gradient! of a least-squares loss. The
# exhaustive catalog of functions and inputs is in allocations.jl.

using Test
using ProximalBase
using SparseArrays
using Random

using ..TestUtils

Random.seed!(1)

a_value(f, x) = f(x)
a_prox!(y, f, x, γ) = prox!(y, f, x, γ)
a_vprox!(y, f, x, γ) = value_and_prox!(y, f, x, γ)
a_vgrad!(g, f, x) = value_and_gradient!(g, f, x)
a_grad!(g, f, x) = gradient!(g, f, x)
a_cd!(x, f, k, γ) = prox_coordinate!(x, f, k, γ)
a_kkt(f, g, x, k) = kkt_violation(f, g, x, k)
a_copyto!(y, x) = copyto!(y, x)

function zero_alloc_prox(f, x; γ = 0.7, value = true)
  y = similar(x)
  value && @test allocs(a_value, f, x) == 0
  @test allocs(a_prox!, y, f, x, γ) == 0
  @test allocs(a_vprox!, y, f, x, γ) == 0
  z = copy(x)
  @test allocs(a_prox!, z, f, z, γ) == 0                  # in place
end

function zero_alloc_grad(f, x)
  g = similar(Array(x))
  @test allocs(a_value, f, x) == 0
  @test allocs(a_vgrad!, g, f, x) == 0
  @test allocs(a_grad!, g, f, x) == 0
end

a_block!(x, f, k, u, γ) = prox_coordinate!(x, f, k, u, γ)

@testset "block steps allocate nothing" begin
  for x in (SparseIterate(zeros(4)),)
    @test allocs(a_block!, x, NormL1(0.3), 1, 3., 0.7) == 0
  end
  for gs in ([1:2, 3:4], [[1, 3], [2, 4]])
    x = AtomIterate(4, gs)
    u = [3., -4.]
    for f in (NormL1(0.3), GroupNormL2(0.4, gs))
      @test allocs(a_block!, x, f, 1, u, 0.7) == 0
    end
  end
end

@testset "distinct group layouts allocate nothing" begin
  ranges = [1:2, 3:5, 6:6, 7:9]
  layouts = (ranges, collect.(ranges))
  v = [3., -4., 0., 0., 0., 0., 2., -1., 0.5]
  for gs in layouts
    for f in (GroupNormL2(0.4, gs),)
      for xgroups in layouts, ygroups in layouts
        xa = AtomIterate(v, xgroups)
        activate!(xa, 3)
        ya = AtomIterate(ones(9), ygroups)
        @test xa.groups !== f.groups && ya.groups !== xa.groups && ya.groups !== f.groups
        @test allocs(a_value, f, xa) == 0
        @test allocs(a_prox!, ya, f, xa, 0.7) == 0
        @test allocs(a_copyto!, ya, xa) == 0
        alias = copy(xa)
        @test allocs(a_prox!, alias, f, alias, 0.7) == 0
        f isa IndBallGroupL0 && continue
        for step in (0.7, [0.4, 1.2, 0.3, 0.8])
          @test allocs(a_prox!, ya, f, xa, step) == 0
          @test allocs(a_vprox!, ya, f, xa, step) == 0
          alias = copy(xa)
          @test allocs(a_vprox!, alias, f, alias, step) == 0
        end
        @test allocs(a_cd!, copy(xa), f, 1, 0.7) == 0
        @test allocs(a_kkt, f, v, xa, 2) == 0
      end
    end
  end
end

@testset "zero allocations" begin
  # the setup of allocations.jl, in the same order, so that the inputs are the same draws
  n, p = 50, 20
  X = randn(n, p)
  y = randn(n)
  x = randn(p)
  xs = SparseIterate(sprandn(p, 0.3))
  w = rand(p)
  groups = [1:5, 6:10, 11:15, 16:20]
  xa = AtomIterate(randn(p), groups)

  @testset "separable penalties" begin
    for f in (NormL1(0.3),)
      @testset "$(nameof(typeof(f)))" begin
        zero_alloc_prox(f, x)
        zero_alloc_prox(f, Float32.(x))
        zero_alloc_prox(f, xs)
        xc = copy(xs)
        @test allocs(a_cd!, xc, f, 3, 0.7) == 0
        xc = copy(xa)
        @test allocs(a_cd!, xc, f, 2, 0.7) == 0
        f isa NormL0 || @test allocs(a_kkt, f, x, x, 4) == 0       # no optimality test at a jump
      end
    end
    # a SymmetricSparseIterate
    D = randn(8, 8); D = D + D'
    s = SymmetricSparseIterate(D)
    for f in (NormL1(0.2),)
      zero_alloc_prox(f, s)
      @test allocs(a_cd!, s, f, 5, 0.7) == 0
    end
  end

  @testset "norms" begin
    for f in (GroupNormL2(0.4, groups),)
      @testset "$(nameof(typeof(f)))" begin
        zero_alloc_prox(f, x)
        zero_alloc_prox(f, Float32.(x))
      end
    end
  end

  @testset "smooth functions" begin
    for (f, z) in ((LeastSquaresLoss(y, X), x),)
      @testset "$(nameof(typeof(f)))" begin
        zero_alloc_grad(f, z)
        isproximable(f) && zero_alloc_prox(f, z)
      end
    end
  end
end

end
