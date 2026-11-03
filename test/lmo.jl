module LMOTest

using Test
using LinearAlgebra
using Random
using ProximalBase

using ..TestUtils

Random.seed!(516)

vertices(::IndSimplex, n, r) = [[i == j ? r : 0.0 for j in 1:n] for i in 1:n]
vertices(f::IndBallL1, n, r) = [[i == j ? σ * r : 0.0 for j in 1:n] for i in 1:n for σ in (1.0, -1.0)]
box_vertices(l, u) = vec([collect(p) for p in Iterators.product(zip(l, u)...)])
a_lmo!(s, C, g) = lmo!(s, C, g)

@testset "traits" begin
  @test haslmo(IndSimplex(1.0)) && haslmo(IndBallL1(2)) && haslmo(IndBallL2(0.5))
  @test haslmo(IndBox(-1, 2)) && haslmo(IndBallLinf(1.0)) && haslmo(IndBox([0, 1.0], [1, 2.0]))
  @test !haslmo(IndSimplex(Inf)) && !haslmo(IndBallL1(Inf)) && !haslmo(IndBallL2(Inf))
  @test !haslmo(IndBox(0, Inf)) && !haslmo(IndBox([0, -Inf], [1, 1.0])) && !haslmo(IndBallLinf(Inf))
  @test !haslmo(IndNonnegative()) && !haslmo(NormL1(1.0)) && !haslmo(IndSphereL2(1.0)) && !haslmo(Scaled(IndBallL2(1.0), 2))
end

@testset "errors for unsupported sets leave s unchanged" begin
  s = fill(7.0, 3)
  for C in (IndNonnegative(), NormL1(1.0), IndSimplex(Inf), IndBox(0, Inf), IndBox([0, 0, -Inf], [1, 1, 1.0]), IndBallLinf(Inf))
    @test_throws ArgumentError lmo!(s, C, [1.0, 2, 3])
    @test s == fill(7.0, 3)
  end
  @test_throws MethodError lmo!(s, IndBallL2(1.0), (1.0, 2.0, 3.0))
end

@testset "feasibility and optimality against enumeration" begin
  for n in (1, 2, 4), r in (0.0, 0.7, 3.0), _ in 1:20
    g = randn(n)
    g[1] = round(g[1], digits = 1) # some ties and sign changes
    for (C, V) in ((IndSimplex(r), vertices(IndSimplex(r), n, r)), (IndBallL1(r), vertices(IndBallL1(r), n, r)))
      s = lmo(C, g)
      @test C(s) == 0
      best = minimum(dot(g, v) for v in V)
      if C isa IndSimplex || r == 0
        @test dot(g, s) ≈ (r == 0 ? 0.0 : best) atol = 1e-14
      else
        @test dot(g, s) ≈ best
      end
    end
    # inequality simplex also contains the origin
    C = IndSimplex(r; equality = false)
    s = lmo(C, g)
    @test C(s) == 0
    @test dot(g, s) ≈ minimum([[dot(g, v) for v in vertices(C, n, r)]; 0.0]) atol = 1e-14
    l, u = -rand(n), rand(n)
    C = IndBox(l, u)
    s = lmo(C, g)
    @test C(s) == 0
    @test dot(g, s) ≈ minimum(dot(g, v) for v in box_vertices(l, u))
    C = IndBallL2(r)
    s = lmo(C, g)
    @test C(s) == 0
    @test dot(g, s) ≈ -r * norm(g)
    C = IndBallLinf(r)
    @test dot(g, lmo(C, g)) ≈ -r * norm(g, 1)
  end
end

@testset "no worse than projected random feasible points" begin
  for n in (3, 10), C in (IndSimplex(1.5), IndSimplex(1.5; equality = false), IndBallL1(2.0), IndBallL2(2.0), IndBallLinf(0.5),
                          IndBox(-ones(n), 2 * ones(n)) )
    for _ in 1:30
      g = randn(n)
      s = lmo(C, g)
      x = prox(C, 3 .* randn(n))
      @test dot(g, s) <= dot(g, x) + 8eps() * norm(g, 1) * 3
    end
  end
end

@testset "ties, zero gradient, zero radius, degenerate bounds" begin
  @test lmo(IndSimplex(2.0), [1.0, 0.0, 0.0, 5.0]) == [0, 2, 0, 0]       # first index
  @test lmo(IndSimplex(2.0), zeros(3)) == [2, 0, 0]
  @test lmo(IndSimplex(2.0; equality = false), zeros(3)) == zeros(3)     # a tie at 0 goes to the origin
  @test lmo(IndSimplex(2.0; equality = false), [1.0, 0.0, 3.0]) == zeros(3)
  @test lmo(IndSimplex(2.0; equality = false), [1.0, -1.0, -1.0]) == [0, 2, 0]
  @test lmo(IndSimplex(0.0), [1.0, -1.0]) == zeros(2)
  @test lmo(IndBallL1(3.0), [1.0, -2.0, 2.0]) == [0, 3, 0]               # first of the largest |g|
  @test lmo(IndBallL1(3.0), [1.0, 2.0]) == [0, -3]
  @test lmo(IndBallL1(3.0), zeros(2)) == zeros(2)
  @test lmo(IndBallL1(0.0), [1.0, 2.0]) == zeros(2)
  @test lmo(IndBallL2(2.0), [3.0, 4.0]) ≈ [-1.2, -1.6]
  @test lmo(IndBallL2(2.0), zeros(2)) == zeros(2)
  @test lmo(IndBallL2(0.0), [1.0, 2.0]) == zeros(2)
  @test lmo(IndBox([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]), [0.0, 1.0, -1.0]) == [0, 0, 3]   # g = 0 goes to the lower bound
  @test lmo(IndBox(1.0, 1.0), [1.0, -1.0, 0.0]) == [1, 1, 1]
  @test lmo(IndBox([1.0, 2.0], [1.0, 5.0]), [-1.0, -1.0]) == [1, 5]
  @test lmo(IndBallLinf(0.0), [1.0, -1.0, 0.0]) == zeros(3)
  @test lmo(IndBallLinf(2), [1, -1, 0]) == [-2, 2, -2]
  # zeros are +0.0
  for (C, g) in ((IndBallLinf(0.0), [1.0, -1.0, 0.0]), (IndBallL1(1.0), zeros(2)), (IndBallL2(1.0), [0.0, 1.0, -0.0]),
                 (IndSimplex(1.0), [0.0, 1.0]), (IndBox(-0.0, 0.0), [1.0, -1.0]))
    @test all(x -> !iszero(x) || !signbit(x), lmo(C, g))
  end
  @test isempty(lmo(IndBallL2(1.0), Float64[])) && isempty(lmo(IndSimplex(1.0; equality = false), Float64[]))
  @test isempty(lmo(IndSimplex(0.0), Float64[]))
  @test_throws ArgumentError lmo(IndSimplex(1.0), Float64[])
end

@testset "element types, matrices, aliasing" begin
  for T in (Float32, Float64), C in (IndSimplex(T(1.5)), IndBallL1(T(2)), IndBallL2(T(2)), IndBallLinf(T(2)), IndBox(T(-1), T(3)))
    g = randn(T, 6)
    s = lmo(C, g)
    @test eltype(s) === T
    @test C(s) == 0
  end
  @test eltype(lmo(IndBallL2(1.0), Float32[1, 2])) === Float32
  @test eltype(lmo(IndBallL2(1.0), [1, 2])) === Float64
  @test lmo(IndBallL1(2), [1, -3]) == [0, 2]
  g = [1.0 -4.0; 2.0 3.0]
  @test lmo(IndBallL1(1.0), g) == [0 1.0; 0 0]
  @test lmo(IndSimplex(1.0), g) == [0 1.0; 0 0]       # the smallest entry, g[1, 2], is first in column order
  @test lmo(IndBallL2(1.0), g) ≈ -g / norm(g)
  S = Symmetric([1.0 -4.0; -4.0 3.0])
  @test lmo(IndBallL1(1.0), S) isa Matrix
  @test lmo(IndBallL1(1.0), S) == [0 0; 1.0 0]        # the tie of |g| between S[2,1] and S[1,2] goes to the first in linear order
  @test lmo(IndBallL2(1.0), view(g, :, 1)) ≈ -g[:, 1] / norm(g[:, 1])
  for C in (IndSimplex(1.5), IndSimplex(1.5; equality = false), IndBallL1(2.0), IndBallL2(2.0), IndBallLinf(0.5), IndBox(-ones(5), 2 * ones(5)))
    g = randn(5)
    want = lmo(C, g)
    x = copy(g)
    @test lmo!(x, C, x) === x
    @test x == want
  end
  out = zeros(3)
  @test lmo!(out, IndBallL1(1.0), [1.0, -2.0, 0.5]) === out
end

@testset "invalid inputs are rejected before writing" begin
  s = fill(7.0, 3)
  for C in (IndSimplex(1.0), IndBallL1(1.0), IndBallL2(1.0), IndBallLinf(1.0), IndBox(zeros(3), ones(3)))
    @test_throws DimensionMismatch lmo!(s, C, [1.0, 2.0])
    @test_throws DimensionMismatch lmo!(zeros(2), C, [1.0, 2.0, 3.0])
    @test_throws ArgumentError lmo!(s, C, [1.0, 2.0im, 3.0])
    @test_throws ArgumentError lmo!(zeros(Int, 3), C, [1.0, 2.0, 3.0])
    @test_throws ArgumentError lmo!(zeros(ComplexF64, 3), C, [1.0, 2.0, 3.0])
    @test s == fill(7.0, 3)
  end
  @test_throws DimensionMismatch lmo!(s, IndBox(zeros(2), ones(2)), [1.0, 2.0, 3.0])
  @test_throws DimensionMismatch lmo!(s, IndBox(0.0, ones(2)), [1.0, 2.0, 3.0])
  @test s == fill(7.0, 3)
  @test_throws ArgumentError lmo!(Symmetric(zeros(2, 2)), IndBallL1(1.0), ones(2, 2))
  @test_throws ArgumentError lmo!(SymmetricSparseIterate(2), IndBallL1(1.0), ones(2, 2))
  # a finite parameter that overflows the element type
  s32 = fill(7.0f0, 2)
  for C in (IndSimplex(1e39), IndBallL1(1e39), IndBallL2(1e39), IndBallLinf(1e39), IndBox([0.0, -1e39], [1.0, 1.0]), IndBox(0.0, 1e39))
    @test_throws ArgumentError lmo!(s32, C, [1.0f0, 2.0f0])
    @test s32 == fill(7.0f0, 2)
  end
  @test lmo!(zeros(2), IndBallL1(1e39), [1.0f0, 2.0f0]) == [0, -1e39]
  @test_throws ArgumentError lmo!(s, IndSimplex(Inf), [1.0, 2.0, 3.0])
end

@testset "NaN and infinite entries" begin
  nan = NaN
  for C in (IndSimplex(1.0), IndSimplex(1.0; equality = false), IndBallL1(1.0), IndBallL2(1.0))
    @test all(isnan, lmo(C, [1.0, nan, 2.0]))
    @test all(isnan, lmo(C, [nan, nan]))
  end
  @test isequal(lmo(IndBox([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]), [nan, 1.0, -1.0]), [nan, 0.0, 3.0])
  @test lmo(IndBallLinf(1.0), [nan, 1.0]) |> x -> isnan(x[1]) && x[2] == -1
  # parameters that remove the dependence on g
  @test lmo(IndBallL1(0.0), [nan, 1.0]) == zeros(2)
  @test lmo(IndBallL2(0.0), [nan, Inf]) == zeros(2)
  @test lmo(IndSimplex(0.0), [nan, 1.0]) == zeros(2)
  @test lmo(IndBox([1.0, 0.0], [1.0, 1.0]), [nan, 1.0]) == [1.0, 0.0]
  # infinite entries take the limit
  @test lmo(IndSimplex(2.0), [1.0, -Inf, -Inf]) == [0, 2, 0]
  @test lmo(IndSimplex(2.0), [Inf, Inf]) == [2, 0]
  @test lmo(IndSimplex(2.0; equality = false), [Inf, Inf]) == [0, 0]
  @test lmo(IndBallL1(2.0), [1.0, -Inf, 3.0]) == [0, 2, 0]
  @test lmo(IndBallL1(2.0), [Inf, -Inf]) == [-2, 0]
  @test lmo(IndBox(0.0, 1.0), [Inf, -Inf, 0.0]) == [0, 1, 0]
  @test lmo(IndBallL2(2.0), [Inf, 1.0, 3.0]) == [-2, 0, 0]
  @test lmo(IndBallL2(2.0), [1.0, -Inf]) == [0, 2]
  @test all(isnan, lmo(IndBallL2(2.0), [Inf, -Inf]))
  @test all(isnan, lmo(IndBallL2(2.0), [Inf, Inf, 1.0]))
end

@testset "L2 at extreme scales" begin
  big = floatmax(Float64)
  @test lmo(IndBallL2(1.0), [big, big]) ≈ fill(-1 / sqrt(2), 2)
  @test lmo(IndBallL2(1.0), [0.75big, -0.5big]) ≈ -[3, -2] / sqrt(13)
  @test lmo(IndBallL2(1.0), [5e-324, 5e-324]) ≈ fill(-1 / sqrt(2), 2)
  @test lmo(IndBallL2(1.0), [1e-200, 0.0]) ≈ [-1, 0]
  @test lmo(IndBallL2(1e300), [3.0, 4.0]) ≈ [-0.6e300, -0.8e300]
  @test lmo(IndBallL2(1e-300), [3.0, 4.0]) ≈ [-0.6e-300, -0.8e-300]
  @test lmo(IndBallL2(1.0f0), Float32[floatmax(Float32), floatmax(Float32)]) ≈ fill(-1 / sqrt(2.0f0), 2)
  @test norm(lmo(IndBallL2(2.0), randn(50) .* 1e150)) ≈ 2
end

@testset "no allocations after warm-up" begin
  for T in (Float32, Float64), n in (1, 100)
    g, s = randn(T, n), zeros(T, n)
    gi = rand(1:5, n)
    for C in (IndSimplex(T(1.5)), IndSimplex(T(1.5); equality = false), IndBallL1(T(2)), IndBallL2(T(2)), IndBallLinf(T(2)),
              IndBox(T(-1), T(3)), IndBox(-rand(T, n), rand(T, n)))
      @test allocs(a_lmo!, s, C, g) == 0
      @test allocs(a_lmo!, g, C, g) == 0
      @test allocs(a_lmo!, s, C, gi) == 0
    end
    @test allocs(a_lmo!, s, IndBallL2(T(2)), fill(floatmax(T), n)) == 0
    @test allocs(a_lmo!, s, IndBallL2(T(2)), fill(T(Inf), n)) == 0
    @test allocs(a_lmo!, s, IndBallL2(T(2)), fill(T(NaN), n)) == 0
  end
end

@testset "L2 at near-floatmax radii: the product must not overflow by rounding (review of #521)" begin
  r = floatmax(Float64)
  s = lmo(IndBallL2(r), [3.0])
  @test s == [-r]
  s = lmo(IndBallL2(r), [-3.0])
  @test s == [r]
  r32 = floatmax(Float32)
  s = lmo(IndBallL2(r32), Float32[25.0])
  @test s == Float32[-r32] && all(isfinite, s) && IndBallL2(r32)(s) == 0
  # mixed magnitudes: the large entry would overflow, the small one must stay accurate
  g = [3e307, 1e-300]
  s = lmo(IndBallL2(r), g)
  @test all(isfinite, s)
  @test s[1] ≈ -r rtol = 1e-15
  @test s[2] ≈ -r * 1e-300 / 3e307 rtol = 1e-14
  @test s[2] ≈ -5.992310449541053e-300 rtol = 1e-14
  s = lmo(IndBallL2(r), [1e-300, 1e300])
  @test s[1] ≈ -r * 1e-300 / 1e300 rtol = 1e-14
  @test s[2] ≈ -r rtol = 1e-15
  # the same through the overflowing-norm path, and mixed element types
  g = [1e308, 1e308, 1e-16]
  s = lmo(IndBallL2(1e300), g)
  @test s[3] ≈ -1e300 * 1e-16 / (sqrt(2) * 1e308) rtol = 1e-14
  s = zeros(2)
  lmo!(s, IndBallL2(1e300), Float32[1.0, 2.0])
  @test s ≈ -1e300 .* [1, 2] ./ sqrt(5.0) rtol = 1e-6
  # aliasing
  g = [3.0]
  @test lmo!(g, IndBallL2(r), g) == [-r]
  g = [3e307, 1e-300]
  want = lmo(IndBallL2(r), g)
  @test lmo!(g, IndBallL2(r), g) == want
  # no allocations, and the other special values are unchanged
  s = zeros(2)
  @test allocs(a_lmo!, s, IndBallL2(r), [3e307, 1e-300]) == 0
  @test allocs(a_lmo!, s, IndBallL2(r), [1e308, 1e308]) == 0
  @test all(isnan, lmo(IndBallL2(r), [NaN, 1.0]))
  @test lmo(IndBallL2(r), [Inf, 1.0]) == [-r, 0]
  @test lmo(IndBallL2(r), zeros(2)) == zeros(2)
  @test lmo(IndBallL2(1e-300), [3.0, 4.0]) ≈ [-0.6e-300, -0.8e-300]
end

@testset "the allocating form uses dense storage (review of #521)" begin
  using SparseArrays
  g = sparsevec([1, 3], [1.0, -2.0], 3)
  for C in (IndSimplex(1.0), IndSimplex(1.0; equality = false), IndBallL1(1.0), IndBallL2(1.0), IndBallLinf(1.0), IndBox(zeros(3), ones(3)))
    s = lmo(C, g)
    @test s isa Vector{Float64} && size(s) == (3,)
    @test s == lmo(C, Vector(g))
    s2 = zeros(3)
    @test lmo!(s2, C, g) === s2
    @test s2 == s
  end
  G = sprand(4, 3, 0.5)
  s = lmo(IndBallL1(2.0), G)
  @test s isa Matrix{Float64} && size(s) == (4, 3) && s == lmo(IndBallL1(2.0), Matrix(G))
  @test lmo(IndBallL2(1.0), sparsevec([2], Float32[3], 4)) isa Vector{Float32}
  @test lmo(IndSimplex(1.0), 1:3) isa Vector{Float64}
  @test lmo(IndSimplex(1.0), view(randn(Float32, 5), 2:4)) isa Vector{Float32}
  @test lmo(IndBallL1(1.0), Symmetric([1.0 -4; -4 3])) isa Matrix{Float64}
end

@testset "validation order: structure, then parameters, then values (E1, O6)" begin
  msg(f) = try; f(); "no error"; catch e; sprint(showerror, e); end
  s = fill(7.0, 3)
  # a size mismatch is reported before an unsupported set or an invalid parameter
  @test_throws DimensionMismatch lmo!(s, IndSimplex(Inf), [1.0, 2.0])
  @test_throws DimensionMismatch lmo!(s, IndBox(0, Inf), [1.0, 2.0])
  @test_throws DimensionMismatch lmo!(s, IndBox([0.0, 0.0], [1.0, Inf]), [1.0, 2.0, 3.0])
  @test_throws DimensionMismatch lmo!(s, IndBallL1(1e39), [1.0f0, 2.0f0])
  # an invalid type is reported before an unsupported set
  @test occursin("floating-point", msg(() -> lmo!(zeros(Int, 3), IndBallL1(Inf), [1.0, 2.0, 3.0])))
  @test occursin("real", lowercase(msg(() -> lmo!(s, IndBallL1(Inf), [1.0im, 2.0, 3.0]))))
  @test occursin("floating-point", msg(() -> lmo!(Symmetric(zeros(3, 3)), IndSimplex(Inf), zeros(3, 3))) )
  # then the set: unsupported before an unrepresentable parameter
  @test occursin("haslmo", msg(() -> lmo!(zeros(Float32, 2), IndBox([0.0, -Inf], [1e39, 1.0]), [1.0f0, 2.0f0])))
  @test s == fill(7.0, 3)
  # every combination leaves the output unchanged
  for C in (IndSimplex(Inf), IndBallL1(Inf), IndBox(0, Inf), IndNonnegative()), g in ([1.0, 2.0], [1.0im, 2.0, 3.0], Float64[])
    try; lmo!(s, C, g); catch; end
    @test s == fill(7.0, 3)
  end
end

end # module
