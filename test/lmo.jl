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

####################################
#  the capped simplex (#517)
####################################

capped(n, k) = Constrained(IndBox(0.0, 1.0), IndHyperplane(ones(n), k))
# the vertices by enumeration, independently of the selection of the oracle
function capped_vertices(n, k)
  V = Vector{Vector{Float64}}()
  for mask in 0:(2^n - 1)
    count_ones(mask) == k && push!(V, [Float64((mask >> (i - 1)) & 1) for i in 1:n])
  end
  V
end

@testset "capped simplex: traits" begin
  @test haslmo(capped(5, 2)) && haslmo(capped(5, 0)) && haslmo(capped(5, 5))
  @test haslmo(Constrained(IndBox(0, 1), IndHyperplane(1, 3))) && haslmo(Constrained(IndBox(0.0, 1.0), IndHyperplane(ones(Int, 4), 1)))
  # nothing else is recognized
  @test !haslmo(Constrained(IndBox(0.0, 1.0), IndHyperplane(ones(4), 2.5)))
  @test !haslmo(Constrained(IndBox(0.0, 1.0), IndHyperplane(ones(4), -1)))
  @test !haslmo(Constrained(IndBox(0.0, 1.0), IndHyperplane(2 * ones(4), 2)))
  @test !haslmo(Constrained(IndBox(0.0, 1.0), IndHyperplane([1.0, 1, 2, 1], 2)))
  @test !haslmo(Constrained(IndBox(0.0, 2.0), IndHyperplane(ones(4), 2)))
  @test !haslmo(Constrained(IndBox(-1.0, 1.0), IndHyperplane(ones(4), 2)))
  @test !haslmo(Constrained(IndBox(0.0, Inf), IndHyperplane(ones(4), 2)))
  @test !haslmo(Constrained(IndBox(0.0, 1.0), IndHalfspace(ones(4), 2)))
  @test !haslmo(Constrained(IndBox(0.0, 1.0), IndHyperslab(ones(4), 1.0, 2.0)))
  @test !haslmo(Constrained(IndBox(zeros(4), ones(4)), IndHyperplane(ones(4), 2)))
end

@testset "capped simplex: feasibility and optimality against enumeration" begin
  for n in (1, 2, 3, 5, 10), k in unique((0, 1, n - 1, n)), T in (Float64, Float32), kind in (:pos, :neg, :mixed, :zero, :ties)
    k < 0 && continue
    g = kind == :pos ? rand(T, n) .+ T(0.1) : kind == :neg ? -rand(T, n) .- T(0.1) : kind == :mixed ? randn(T, n) :
        kind == :zero ? zeros(T, n) : T.(rand(1:2, n))
    C = capped(n, k)
    s = lmo(C, g)
    @test s isa Vector{T}
    @test C(s) == 0
    @test all(x -> x == 0 || x == 1, s) && sum(s) == k
    @test dot(g, s) == minimum(dot(g, T.(v)) for v in capped_vertices(n, k))
    # the same through the in-place form and an aliased gradient
    s2 = fill(T(7), n)
    @test lmo!(s2, C, g) === s2 && s2 == s
    g2 = copy(g)
    @test lmo!(g2, C, g2) === g2 && g2 == s
  end
end

@testset "capped simplex: deterministic ties" begin
  C = capped(4, 2)
  @test lmo(C, [3.0, 1.0, 1.0, -2.0]) == [0, 1, 0, 1]
  # more entries at the threshold than remain: the earliest win
  @test lmo(capped(5, 2), [1.0, 1.0, 1.0, 1.0, 1.0]) == [1, 1, 0, 0, 0]
  @test lmo(capped(6, 3), [0.0, -1.0, 0.0, 0.0, -1.0, 0.0]) == [1, 1, 0, 0, 1, 0]
  @test lmo(capped(4, 3), [2.0, 2.0, 2.0, 2.0]) == [1, 1, 1, 0]
  # a signed zero is a tie
  @test lmo(capped(3, 1), [0.0, -0.0, 0.0]) == [1, 0, 0]
  # a scalar normal
  @test lmo(Constrained(IndBox(0, 1), IndHyperplane(1, 2)), [5.0, 4.0, 4.0, 4.0]) == [0, 1, 1, 0]
  # ±Inf compare as entries: tied infinities are resolved by index
  @test lmo(capped(4, 2), [-Inf, 1.0, -Inf, -Inf]) == [1, 0, 1, 0]
  @test lmo(capped(4, 3), [Inf, Inf, 0.0, Inf]) == [1, 1, 1, 0]
end

@testset "capped simplex: integer and matrix gradients" begin
  # distinct integers beyond Float64's exact range keep their order
  big = 2^60
  g = [big + 3, big + 1, big + 2, big + 1, big]
  C = capped(5, 2)
  @test lmo(C, g) == [0.0, 1.0, 0.0, 0.0, 1.0]
  @test Float64.(g) == fill(Float64(big), 5) # the floats cannot tell them apart
  @test lmo(capped(3, 1), [typemin(Int), typemax(Int), 0]) == [1, 0, 0]
  @test lmo(capped(3, 2), UInt8[3, 1, 2]) == [0, 1, 1]
  G = [3.0 -1.0 2.0; 0.5 4.0 -2.0]
  C = capped(6, 3)
  s = lmo(C, G)
  @test s isa Matrix{Float64} && size(s) == size(G) && s == reshape(lmo(C, vec(G)), 2, 3)
  @test sum(s) == 3 && dot(G, s) == minimum(dot(vec(G), v) for v in capped_vertices(6, 3))
  Gs = copy(G)
  @test lmo!(Gs, C, Gs) === Gs && Gs == s
  @test lmo(C, view(randn(10), 2:7)) isa Vector{Float64}
  S = zeros(Float32, 6)
  @test lmo!(view(S, 1:6), C, vec(G)) isa SubArray && S == vec(s)
end

@testset "capped simplex: errors leave the output unchanged" begin
  s = fill(7.0, 4)
  g = [1.0, 2.0, 3.0, 4.0]
  msg(f) = try; f(); "no error"; catch e; sprint(showerror, e); end
  for C in (Constrained(IndBox(0.0, 1.0), IndHyperplane(ones(4), 2.5)), Constrained(IndBox(0.0, 1.0), IndHyperplane(ones(4), -1)),
            Constrained(IndBox(0.0, 2.0), IndHyperplane(ones(4), 2)), Constrained(IndBox(0.0, 1.0), IndHyperplane([1.0, 1, 2, 1], 2)),
            Constrained(IndBox(0.0, 1.0), IndHalfspace(ones(4), 2)), Constrained(IndBox(0.0, 1.0), IndHyperplane(1, 5)))
    for gg in (g, g[1:3])
      try; lmo!(s, C, gg); catch; end
      @test s == fill(7.0, 4)
    end
  end
  @test_throws ArgumentError lmo!(s, Constrained(IndBox(0.0, 1.0), IndHyperplane(1, 5)), g)  # k > n
  @test_throws ArgumentError lmo!(s, Constrained(IndBox(0.0, 1.0), IndHyperplane(1.0, 2.5)), g)
  @test_throws ArgumentError lmo!(s, Constrained(IndBox(0.0, 2.0), IndHyperplane(ones(4), 2)), g)
  @test_throws DimensionMismatch lmo!(s, capped(4, 2), g[1:3])
  @test_throws DimensionMismatch lmo!(s, capped(3, 2), g)           # the normal has the wrong length
  @test_throws ArgumentError lmo!(s, capped(4, 2), [1.0im, 2, 3, 4])
  @test_throws ArgumentError lmo!(zeros(Int, 4), capped(4, 2), g)
  @test_throws ArgumentError lmo!(Symmetric(zeros(2, 2)), capped(4, 2), zeros(2, 2))
  @test_throws MethodError lmo!(s, capped(4, 2), (1.0, 2.0, 3.0, 4.0))
  # structure before the set before the parameters
  @test_throws DimensionMismatch lmo!(s, Constrained(IndBox(0.0, 2.0), IndHyperplane(ones(4), 2.5)), g[1:3])
  @test occursin("floating-point", msg(() -> lmo!(zeros(Int, 4), Constrained(IndBox(0.0, 2.0), IndHyperplane(ones(4), 2.5)), g)))
  @test occursin("haslmo", msg(() -> lmo!(s, Constrained(IndBox(0.0, 2.0), IndHyperplane(1, 9)), g)))
  @test occursin("no point", msg(() -> lmo!(s, Constrained(IndBox(0.0, 1.0), IndHyperplane(1, 9)), g)))
  @test s == fill(7.0, 4)
end

@testset "capped simplex: NaN and infinities (E2, E3)" begin
  C = capped(4, 2)
  @test all(isnan, lmo(C, [1.0, NaN, 3.0, 4.0]))
  @test all(isnan, lmo(C, [NaN, NaN, NaN, NaN]))
  @test lmo(C, [Inf, 2.0, 3.0, -Inf]) == [0, 1, 0, 1]
  @test lmo(C, [Inf, Inf, Inf, Inf]) == [1, 1, 0, 0]
  @test lmo(C, [-Inf, -Inf, -Inf, -Inf]) == [1, 1, 0, 0]
  # the minimizer of a degenerate set does not depend on g
  @test lmo(capped(3, 0), [NaN, Inf, -Inf]) == [0, 0, 0]
  @test lmo(capped(3, 3), [NaN, Inf, -Inf]) == [1, 1, 1]
  @test lmo(capped(3, 3), Float32[NaN, 1, 2]) isa Vector{Float32}
  g = [1.0, NaN, 3.0, 4.0]
  @test all(isnan, lmo!(g, C, g))
  @test lmo(Constrained(IndBox(0, 1), IndHyperplane(1, 0)), Float64[]) == Float64[]
end

alloc_lmo(s, C, g) = @allocated lmo!(s, C, g)
alloc_val(c, y) = @allocated c(y)

@testset "capped simplex: large n against a stable sort" begin
  # n > 600 reaches the sampling branch of the selection
  for n in (601, 2000), k in (1, 17, n ÷ 2, n - 3), _ in 1:3
    g = Float64.(rand(-30:30, n)); g[rand(1:n, 3)] .= [-Inf, Inf, -Inf]
    idx = sortperm(g)[1:k]
    ref = zeros(n); ref[idx] .= 1
    @test lmo(capped(n, k), g) == ref
  end
end

@testset "capped simplex: no allocation after warm-up" begin
  for T in (Float32, Float64), n in (10, 1000), k in (1, n ÷ 2, n - 1)
    C = capped(n, k)
    g = randn(T, n)
    s = zeros(T, n)
    lmo!(s, C, g); lmo!(g, C, g)
    @test alloc_lmo(s, C, g) == 0
    h = copy(g)
    @test alloc_lmo(h, C, h) == 0
  end
end

####################################
#  the support function (#517)
####################################

top_sum(y, k) = sum(sort(vec(y); rev = true)[1:k])

@testset "capped simplex: the support function" begin
  for n in (1, 2, 3, 5, 10), T in (Float64, Float32), scale in (1.0, 1e-8, 1e8), kind in (:mixed, :ties, :neg)
    y = kind == :mixed ? T.(scale .* randn(n)) : kind == :ties ? T.(scale .* rand(-1:1, n)) : T.(-scale .* (1 .+ rand(n)))
    for k in unique((0, 1, n - 1, n))
      C = capped(n, k)
      σ = Conjugate(C)(y)
      ref = k == 0 ? zero(T) : top_sum(Float64.(y), k)
      @test σ isa T
      @test σ ≈ ref rtol = 4 * eps(T) atol = 4 * eps(T) * scale
      k >= 1 && @test σ ≈ SumLargest(k)(y) rtol = 4 * eps(T) atol = 4 * eps(T) * scale
      k == n && @test σ ≈ sum(Float64.(y)) rtol = 4 * eps(T) atol = 4 * eps(T) * scale
    end
  end
  # the identity ⟨y, lmo(C, -y)⟩ agrees with the direct route
  for _ in 1:20
    n = rand(2:8); k = rand(1:n); y = randn(n)
    @test dot(y, lmo(capped(n, k), -y)) ≈ Conjugate(capped(n, k))(y)
  end
end

# a convex indicator with an oracle and no formula for its conjugate: [l, u]ⁿ
struct OracleCube <: ProximalBase.AbstractFunction
  l::Float64
  u::Float64
end
ProximalBase.isconvex(::Type{OracleCube}) = true
ProximalBase.isproximable(::Type{OracleCube}) = true
ProximalBase._isindicator(::Type{OracleCube}) = true
ProximalBase.haslmo(f::OracleCube) = isfinite(f.l) && isfinite(f.u)
(f::OracleCube)(x::AbstractArray) = all(v -> f.l <= v <= f.u, x) ? 0.0 : Inf
function ProximalBase.prox!(y::AbstractArray, f::OracleCube, x::AbstractArray, γ::Real)
  y .= clamp.(x, f.l, f.u)
end
function ProximalBase.lmo!(s::AbstractArray, f::OracleCube, g::AbstractArray)
  size(s) == size(g) || throw(DimensionMismatch())
  @inbounds for i in eachindex(s, g)
    s[i] = g[i] > 0 ? f.l : g[i] < 0 ? f.u : f.l
  end
  s
end
cube_support(f, y) = sum(v -> v >= 0 ? f.u * v : f.l * v, y)

@testset "support-function fallback" begin
  f = OracleCube(-1.0, 2.0)
  @test ProximalBase._has_conjugate_value(f) && ProximalBase._has_value(Conjugate(f))
  for T in (Float64, Float32), n in (1, 4, 20)
    y = randn(T, n)
    @test Conjugate(f)(y) isa T
    @test Conjugate(f)(y) ≈ cube_support(f, y) rtol = 4 * eps(T)
    @test Conjugate(f)(reshape(vcat(y, y), n, 2)) ≈ 2cube_support(f, y) rtol = 4 * eps(T)
  end
  @test Conjugate(f)(Int[1, -2, 3]) ≈ 2 + 2 + 6
  @test Conjugate(f)(Float64[]) == 0
  # wrappers
  y = randn(5)
  @test Conjugate(Scaled(f, 2.0))(y) ≈ 2 * cube_support(f, y / 2)
  @test Conjugate(Translate(f, fill(0.5, 5)))(y) ≈ cube_support(f, y) + 0.5 * sum(y)
  # an entry at an unattained end contributes nothing, also when it is infinite
  @test Conjugate(OracleCube(0.0, 1.0))([Inf, -Inf, 2.0]) == Inf
  @test Conjugate(OracleCube(0.0, 1.0))([-Inf, -1.0, -2.0]) == 0
  # extreme scales
  big = floatmax() / 4
  @test Conjugate(OracleCube(0.0, 1.0))([big, -big, big / 2]) == big + big / 2
  @test Conjugate(OracleCube(0.0, 1.0))([2big, 2big, 2big]) == Inf
  tiny = nextfloat(0.0)
  @test Conjugate(OracleCube(0.0, 1.0))([tiny, 3tiny, -tiny]) == 4tiny
  # cancellation: the compensated sum recovers what the plain one loses
  @test Conjugate(OracleCube(1.0, 2.0))([1e16, 0.5, -2e16]) == 1.0
  # value and proximal operator
  for γ in (0.3, 1.0, 5.0), x in (randn(6) .* 3, [4.0, -5.0, 0.0, 1.0, 2.0, -1.0])
    c = Conjugate(f)
    out = similar(x)
    v = value_and_prox!(out, c, x, γ)
    @test v ≈ c(out) rtol = 1e-12 atol = 1e-12
    @test out ≈ x - γ * clamp.(x / γ, f.l, f.u)
    # Fenchel: f*(y) + f(p) = ⟨p, y⟩ at p = (x - y)/γ
    p = (x - out) / γ
    @test v ≈ dot(p, out) atol = 1e-10
    z = copy(x)
    @test value_and_prox!(z, c, z, γ) ≈ v && z ≈ out
  end
  # the fallback is not given to a set without an oracle
  @test_throws ArgumentError Conjugate(OracleCube(0.0, Inf))([1.0])
  @test !ProximalBase._has_conjugate_value(OracleCube(0.0, Inf)) && !ProximalBase._has_value(Conjugate(OracleCube(0.0, Inf)))
  # specialized methods still take precedence, and unbounded sets keep their direct formulas
  @test Conjugate(IndSimplex(1.0))([1.0, 3.0, 2.0]) == 3.0
  @test Conjugate(IndSimplex(1.0; equality = false))([-1.0, -3.0]) == 0.0
  @test Conjugate(IndBallL1(2.0))([1.0, -3.0]) == 6.0
  @test Conjugate(IndBallL2(2.0))([3.0, 4.0]) == 10.0
  @test Conjugate(IndBox(-1.0, 2.0))([1.0, -3.0]) == 5.0
  @test Conjugate(IndBox(0.0, Inf))([-1.0, -3.0]) == 0.0 && Conjugate(IndBox(0.0, Inf))([1.0, -3.0]) == Inf
  @test Conjugate(IndNonnegative())([-1.0, -3.0]) == 0.0 && Conjugate(IndNonnegative())([1.0]) == Inf
  @test Conjugate(capped(4, 2))([1.0, 5.0, 3.0, 2.0]) == 8.0
  @test Conjugate(Constrained(IndBox(0.0, 1.0), IndHalfspace(ones(4), 2)))([1.0, 5.0, 3.0, -2.0]) == 8.0
  # value_and_prox! of a capped-simplex conjugate
  c = Conjugate(capped(5, 2)); x = randn(5); out = similar(x)
  @test value_and_prox!(out, c, x, 0.7) ≈ c(out)
end

@testset "support-function fallback: allocations" begin
  f = OracleCube(-1.0, 2.0)
  c = Conjugate(f)
  for T in (Float32, Float64), n in (10, 10000)
    y = randn(T, n)
    c(y)
    # neither a negated copy nor the minimizer is allocated
    @test alloc_val(c, y) == 0
  end
  C = Conjugate(capped(1000, 10)); y = randn(1000); C(y)
  @test alloc_val(C, y) == 0
end

####################################
#  the generic fallback under review (#530)
####################################

# an oracle that writes part of its destination, then fails
struct ThrowCube <: ProximalBase.AbstractFunction
  mode::Symbol
  n::Int
end
ProximalBase.isconvex(::Type{ThrowCube}) = true
ProximalBase.isproximable(::Type{ThrowCube}) = true
ProximalBase._isindicator(::Type{ThrowCube}) = true
ProximalBase.haslmo(::ThrowCube) = true
ProximalBase.prox!(y::AbstractArray, ::ThrowCube, x::AbstractArray, γ::Real) = (y .= clamp.(x, 0.0, 1.0))
function ProximalBase.lmo!(s::AbstractArray, f::ThrowCube, g::AbstractArray)
  length(g) == f.n || throw(DimensionMismatch("this set has dimension $(f.n)"))
  f.mode === :parameter && throw(ArgumentError("the parameters of the set are invalid"))
  for i in eachindex(s, g)
    s[i] = 42.0
    f.mode === :partial && i == length(s) ÷ 2 + 1 && throw(ErrorException("the oracle failed after writing"))
  end
  s
end

# an oracle that needs the matrix: column j is the interval [0, j]
struct ColumnCube <: ProximalBase.AbstractFunction end
ProximalBase.isconvex(::Type{ColumnCube}) = true
ProximalBase.isproximable(::Type{ColumnCube}) = true
ProximalBase._isindicator(::Type{ColumnCube}) = true
ProximalBase.haslmo(::ColumnCube) = true
function ProximalBase.prox!(y::AbstractArray, ::ColumnCube, x::AbstractArray, γ::Real)
  ndims(x) == 2 || throw(ArgumentError("ColumnCube needs a matrix"))
  for j in 1:size(x, 2), i in 1:size(x, 1)
    y[i, j] = clamp(x[i, j], 0.0, j)
  end
  y
end
function ProximalBase.lmo!(s::AbstractArray, ::ColumnCube, g::AbstractArray)
  ndims(g) == 2 && ndims(s) == 2 || throw(ArgumentError("ColumnCube needs a matrix"))
  for j in 1:size(g, 2), i in 1:size(g, 1)
    s[i, j] = g[i, j] < 0 ? Float64(j) : 0.0
  end
  s
end
column_support(y) = sum(j * max(y[i, j], 0.0) for i in 1:size(y, 1), j in 1:size(y, 2))

# an oracle that evaluates another conjugate with its own destination in use
struct NestedCube <: ProximalBase.AbstractFunction
  aux::Vector{Float64}
end
ProximalBase.isconvex(::Type{NestedCube}) = true
ProximalBase.isproximable(::Type{NestedCube}) = true
ProximalBase._isindicator(::Type{NestedCube}) = true
ProximalBase.haslmo(::NestedCube) = true
ProximalBase.prox!(y::AbstractArray, ::NestedCube, x::AbstractArray, γ::Real) = (y .= clamp.(x, 0.0, 1.0))
function ProximalBase.lmo!(s::AbstractArray, f::NestedCube, g::AbstractArray)
  for i in eachindex(s, g)
    s[i] = g[i] < 0 ? 1.0 : 0.0
  end
  Conjugate(OracleCube(-5.0, 5.0))(f.aux)   # evaluated while s holds the minimizer
  s
end

# the exact value, rounded once
function cube_exact(f, y)
  r = sum(v -> v >= 0 ? big(f.u) * big(v) : big(f.l) * big(v), y)
  T = float(eltype(y))
  T(r)
end

@testset "support fallback: overflow, underflow and cancellation" begin
  M = floatmax(Float64); tiny = nextfloat(0.0)
  f = OracleCube(1.0, 2.0)
  @test Conjugate(f)([M, -M]) == M
  @test Conjugate(OracleCube(0.0, 0.5))([tiny, tiny]) == tiny
  # mixed-sign large terms and an exact cancellation
  @test Conjugate(f)([M / 2, -2M / 2, M / 2]) == cube_exact(f, [M / 2, -2M / 2, M / 2])
  @test Conjugate(f)([1e16, 0.5, -2e16]) == 1.0
  @test Conjugate(f)([M, 0.25, -M, M / 4, -M / 4]) == Inf     # the exact value, 1.25 M, does not fit
  @test Conjugate(f)([M, 0.25, -M, -M / 4]) == cube_exact(f, [M, 0.25, -M, -M / 4])
  @test Conjugate(OracleCube(-1.0, 1.0))([M, -M, M, -M]) == cube_exact(OracleCube(-1.0, 1.0), [M, -M, M, -M])
  # small products that underflow before the sum
  g = OracleCube(0.0, 1e-160)
  y = fill(1e-160, 1000)
  @test Conjugate(g)(y) ≈ cube_exact(g, y) rtol = 4eps()
  y = [1e-160, 5e-324, 1e-160, 3.0e-300, 1e-160]
  @test Conjugate(g)(y) ≈ cube_exact(g, y) rtol = 4eps()
  @test Conjugate(OracleCube(0.0, 0.5))(fill(tiny, 4)) == 2tiny
  # a large and a small factor in different entries
  h = OracleCube(0.0, 1e-300)
  y = [1e300, 1e-300, 1e300]
  @test Conjugate(h)(y) ≈ cube_exact(h, y) rtol = 4eps()
  # Float32 has a wider range in the sum
  M32 = floatmax(Float32)
  c32 = Conjugate(OracleCube(1.0, 2.0))
  @test c32(Float32[M32, -M32]) === M32
  @test c32(Float32[1f-30, 1f-30]) isa Float32
  # a result that does not fit is infinite
  @test Conjugate(OracleCube(0.0, 2.0))([M, M]) == Inf
  @test Conjugate(OracleCube(0.0, 2.0))(Float32[M32, M32]) === Inf32
  @test Conjugate(OracleCube(0.0, 2.0))([-M, -M]) == 0.0
  # an infinite entry at a nonzero minimizer, and the opposite infinities
  @test Conjugate(OracleCube(0.0, 1.0))([Inf, 1.0]) == Inf
  @test Conjugate(OracleCube(-1.0, 1.0))([Inf, -Inf]) == Inf          # both terms are +Inf
  @test isnan(Conjugate(OracleCube(1.0, 2.0))([Inf, -Inf]))           # +Inf and -Inf
  for _ in 1:30
    n = rand(1:6); sc = 10.0^rand(-300:300)
    y = sc .* randn(n)
    fo = OracleCube(rand(), 1 + rand())
    ref = cube_exact(fo, y)
    got = Conjugate(fo)(y)
    @test isinf(ref) ? got == ref : isapprox(got, ref; rtol = 8eps(), atol = 8 * nextfloat(0.0))
  end
end

@testset "support fallback: NaN" begin
  f = OracleCube(0.0, 1.0)
  @test isnan(Conjugate(f)([NaN]))
  @test isnan(Conjugate(f)([1.0, NaN, -2.0]))
  @test isnan(Conjugate(f)(Float32[NaN, 1]))
  # a zero minimizer entry times an infinite one adds nothing
  @test Conjugate(f)([-Inf, 1.0]) == 1.0
  @test Conjugate(f)([-Inf, -Inf]) == 0.0
end

@testset "value_and_prox! of a fallback conjugate leaves the output unchanged on error" begin
  bits(v) = reinterpret(UInt64, copy(v))
  for mode in (:partial, :parameter), γ in (0.5, 2.0)
    f = ThrowCube(mode, 5)
    c = Conjugate(f)
    x = [0.3, -1.0, 2.0, 0.9, -0.2]
    out = fill(7.0, 5); before = bits(out)
    @test_throws Exception value_and_prox!(out, c, x, γ)
    @test bits(out) == before
    z = copy(x); zb = bits(z)
    @test_throws Exception value_and_prox!(z, c, z, γ)           # aliased
    @test bits(z) == zb
    @test_throws Exception c(x)
  end
  # invalid dimensions of the oracle, and a tuple input
  c = Conjugate(ThrowCube(:partial, 4))
  x = rand(5); out = fill(7.0, 5); before = bits(out)
  @test_throws DimensionMismatch value_and_prox!(out, c, x, 1.0)
  @test bits(out) == before
  @test_throws DimensionMismatch value_and_prox!(out, c, x, 2.0)
  # a structurally invalid output is rejected before anything is written
  @test_throws DimensionMismatch value_and_prox!(zeros(4), Conjugate(ThrowCube(:partial, 5)), x, 1.0)
  @test_throws ArgumentError value_and_prox!(zeros(ComplexF64, 5), Conjugate(ThrowCube(:partial, 5)), x, 1.0)
  # wrapped, and a separable sum of blocks
  w = Conjugate(Scaled(ThrowCube(:partial, 5), 2.0))
  out = fill(7.0, 5); before = bits(out)
  @test_throws Exception value_and_prox!(out, w, x, 1.0)
  @test bits(out) == before
  sc = Conjugate(SeparableSum(ThrowCube(:partial, 2), OracleCube(0.0, 1.0)))
  yt = (fill(7.0, 2), fill(7.0, 3)); xt = (rand(2), rand(3))
  @test_throws Exception value_and_prox!(yt, sc, xt, 1.0)
  @test yt == (fill(7.0, 2), fill(7.0, 3))
  # success still commits, in both forms
  f = OracleCube(-1.0, 2.0); c = Conjugate(f)
  x = randn(6) .* 3
  out = similar(x); v = value_and_prox!(out, c, x, 0.7)
  z = copy(x); @test value_and_prox!(z, c, z, 0.7) == v && z == out
  @test v ≈ c(out)
  st = (similar(xt[1]), similar(xt[2]))
  sf = Conjugate(SeparableSum(OracleCube(-1.0, 2.0), OracleCube(0.0, 1.0)))
  vt = value_and_prox!(st, sf, xt, 1.0)
  @test vt ≈ sf(st)
end

@testset "support fallback keeps the shape of the argument" begin
  c = Conjugate(ColumnCube())
  Y = randn(4, 3)
  @test c(Y) ≈ column_support(Y)
  @test c(view(Y, 1:2, :)) ≈ column_support(Y[1:2, :])
  @test c(Float32.(Y)) ≈ column_support(Float32.(Y)) rtol = 1e-5
  @test c(permutedims(Y)) ≈ column_support(permutedims(Y))
  X = randn(4, 3) .* 2
  out = similar(X)
  v = value_and_prox!(out, c, X, 0.8)
  @test v ≈ c(out) && v isa Float64
  @test out ≈ X .- 0.8 .* ((x, j) -> clamp(x / 0.8, 0.0, j)).(X, reshape(1:3, 1, 3))
  Z = copy(X)
  @test value_and_prox!(Z, c, Z, 0.8) ≈ v && Z ≈ out
  # a vector is not a matrix for this set
  @test_throws ArgumentError c(randn(4))
end

@testset "support fallback: nested conjugates keep the outer minimizer" begin
  f = NestedCube(randn(7))
  y = randn(7)
  @test Conjugate(f)(y) ≈ sum(max.(y, 0))
  Y = randn(7)
  out = similar(Y)
  @test value_and_prox!(out, Conjugate(f), Y, 1.0) ≈ Conjugate(f)(out)
  # a nested call of another size, and of a different element type
  @test Conjugate(NestedCube(randn(3)))(y) ≈ sum(max.(y, 0))
  @test Conjugate(NestedCube(randn(20)))(Float32.(y)) ≈ sum(max.(Float32.(y), 0))
end

@testset "support fallback: allocations after warm-up" begin
  c = Conjugate(OracleCube(-1.0, 2.0))
  for T in (Float32, Float64), n in (10, 1000, 100000)
    y = randn(T, n)
    c(y)
    @test alloc_val(c, y) == 0
  end
  cm = Conjugate(ColumnCube()); Y = randn(30, 4); cm(Y)
  @test alloc_val(cm, Y) == 0
  # the nested frame is also reused
  cn = Conjugate(NestedCube(randn(7))); yy = randn(7); cn(yy)
  @test alloc_val(cn, yy) == 0
end

@testset "capped simplex: the empty normal follows the constructor" begin
  # an empty array normal is the zero vector, which IndHyperplane rejects
  @test_throws ArgumentError IndHyperplane(ones(0), 0)
  # the scalar normal represents the zero-dimensional set
  C = Constrained(IndBox(0, 1), IndHyperplane(1, 0))
  @test haslmo(C) && lmo(C, Float64[]) == Float64[] && Conjugate(C)(Float64[]) == 0
  @test_throws ArgumentError lmo!(Float64[], Constrained(IndBox(0, 1), IndHyperplane(1, 1)), Float64[])
end

@testset "support fallback: further cases from the review" begin
  M = floatmax(Float64)
  f = OracleCube(1.0, 2.0)
  # the large group overflows on its own, but the total fits against the middle group
  y = [0.725M; fill(-4.9e-5 * M, 10000)]
  @test Conjugate(f)(y) ≈ cube_exact(f, y) rtol = 8eps()
  # other entry types and signed zeros
  g = OracleCube(0.0, 2.0)
  @test Conjugate(g)(Int[3, -1, 4]) == 14.0
  @test Conjugate(g)(Bool[true, false, true]) == 4.0
  @test Conjugate(g)(Float16[1.5, -2, 0.5]) isa Float16 && Conjugate(g)(Float16[1.5, -2, 0.5]) == 4
  @test Conjugate(g)([-0.0, 0.0]) == 0.0
  @test Conjugate(g)(BigFloat[1, -2, 3]) == 8
  @test isnan(Conjugate(g)(BigFloat[1, NaN]))
  # the frame stack returns to its depth after errors, and a failing call does not poison later ones
  d0 = ProximalBase._frames().depth
  for _ in 1:3
    @test_throws Exception Conjugate(ThrowCube(:partial, 5))(rand(5))
    @test_throws Exception value_and_prox!(zeros(5), Conjugate(ThrowCube(:partial, 5)), rand(5), 1.0)
  end
  @test ProximalBase._frames().depth == d0
  @test Conjugate(OracleCube(-1.0, 2.0))([1.0, -1.0]) == 3.0
  # wrappers and other outputs leave the output unchanged on error
  bits(v) = reinterpret(UInt64, copy(v))
  x = rand(5)
  for c in (Conjugate(Translate(ThrowCube(:partial, 5), fill(0.5, 5))), Conjugate(Scaled(Translate(ThrowCube(:partial, 5), fill(0.5, 5)), 2.0)))
    out = fill(7.0, 5); before = bits(out)
    @test_throws Exception value_and_prox!(out, c, x, 1.0)
    @test bits(out) == before
    big = fill(7.0, 8); bb = bits(big)
    @test_throws Exception value_and_prox!(view(big, 2:6), c, x, 1.0)
    @test bits(big) == bb
  end
  out = fill(ComplexF64(7), 5)
  @test_throws ArgumentError value_and_prox!(out, Conjugate(ThrowCube(:partial, 5)), x, 1.0)
  @test out == fill(ComplexF64(7), 5)
  # tasks do not share frames
  c = Conjugate(NestedCube(randn(7)))
  results = fetch.([Threads.@spawn(begin
    y = randn(7); ok = true
    for _ in 1:200
      ok &= c(y) ≈ sum(max.(y, 0))
    end
    ok
  end) for _ in 1:8])
  @test all(results)
end

end # module
