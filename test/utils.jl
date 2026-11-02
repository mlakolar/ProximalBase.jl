module UtilsTest

using Test
using ProximalBase
using LinearAlgebra
using Random
using SparseArrays

Random.seed!(1)

@testset "public squared operator norm" begin
  for T in (Float32, Float64), dims in ((7,3),(3,7),(4,4))
    X = randn(T,dims...)
    for A in (X,sparse(X))
      @test opnorm2(A) ≈ opnorm(X)^2 rtol=1e-5
      @test opnorm2(A) isa T
    end
  end
  @test opnorm2(zeros(3,4)) == 0
  @test opnorm2(zeros(0,3)) == 0
  @test opnorm2(zeros(3,0)) == 0
  @test opnorm2([1 0; 0 2]) ≈ 4
end

@testset "shrink" begin
  @test shrink(3.0, 1.0) == 2.0
  @test shrink(-2.0, 1.0) == -1.0
  @test shrink(0.5, 1.0) == 0.
  @test shrink(1.0, 1.0) === shrink(-1.0, 1.0) === shrink(-0.0, 0.0) === 0.0
  @test shrink(Inf, 1.0) == Inf && shrink(-Inf, 1.0) == -Inf
  @test isnan(shrink(NaN, 1.0))             # a NaN propagates instead of becoming zero
  @test shrink(3f0, 1f0) === 2f0
  @test shrink(3, 1) === 2.0                # mixed and integer arguments are promoted to floats
  v, c = randn(100), 0.7
  @test shrink.(v, c) == sign.(v) .* max.(abs.(v) .- c, 0)
end

@testset "quickselect" begin
  for n in (1, 2, 3, 10, 101, 1000), trial in 1:5
    v = trial == 5 ? rand(1:4, n) .+ 0.0 : randn(n)          # with many ties
    s = sort(v; rev = true)
    for k in unique([1, max(1, n ÷ 2), n])
      w = copy(v)
      @test ProximalBase._kth_largest!(w, k) == s[k]
      @test sort(w; rev = true) == s                          # a permutation of v
    end
  end
  w = randn(50)
  @test ProximalBase._kth_largest!(view(w, 11:30), 3) == sort(w[11:30]; rev = true)[3]
  @test_throws ArgumentError ProximalBase._kth_largest!(randn(3), 4)
  a(v, k) = @allocated ProximalBase._kth_largest!(v, k)
  v = randn(1000)
  a(v, 10)
  @test a(v, 10) == 0
end

# the threshold of the projection onto the simplex, by sorting
function simplex_threshold_sort(y, a)
  u = sort(y; rev = true)
  c, τ = 0.0, 0.0
  for (j, uj) in enumerate(u)
    c += uj
    t = (c - a) / j
    uj > t && (τ = t)
  end
  τ
end

@testset "projection onto the simplex" begin
  for n in (1, 2, 5, 50, 1000), a in (0.5, 1.0, 10.0)
    y = randn(n) .* 3
    buf = zeros(n)
    τ = ProximalBase._simplex_threshold!(buf, y, eachindex(y), a)
    @test τ ≈ simplex_threshold_sort(y, a)
    @test sum(max.(y .- τ, 0)) ≈ a
    # on |y|, the l1 ball
    τ = ProximalBase._simplex_threshold!(buf, y, eachindex(y), a, abs)
    @test sum(max.(abs.(y) .- τ, 0)) ≈ a
  end
  # ties and a single large entry
  @test ProximalBase._simplex_threshold!(zeros(4), ones(4), 1:4, 2.) ≈ 0.5
  @test ProximalBase._simplex_threshold!(zeros(3), [10., 0., 0.], 1:3, 1.) ≈ 9.
  # a matrix through its linear indices, and the indices of a group
  M = randn(4, 5)
  @test ProximalBase._simplex_threshold!(zeros(20), M, eachindex(IndexLinear(), M), 1.) ≈ simplex_threshold_sort(vec(M), 1.)
  G = [3, 7, 8, 15]
  @test ProximalBase._simplex_threshold!(zeros(4), M, G, 0.5, abs) ≈
        simplex_threshold_sort(abs.(M[G]), 0.5)
  @test ProximalBase._simplex_threshold!(zeros(1), Float64[], Int[], 1.) == 0.
  a_thr(buf, x) = @allocated ProximalBase._simplex_threshold!(buf, x, eachindex(x), 1., abs)
  y, buf = randn(1000), zeros(1000)
  a_thr(buf, y)
  @test a_thr(buf, y) == 0
end

@testset "lower triangular helpers" begin
  for p in (1, 2, 5, 13)
    rc = [(r, c) for c in 1:p for r in c:p]
    @test [tril_cartesian(p, k) for k in eachindex(rc)] == rc
    @test [tril_index(p, r, c) for (r, c) in rc] == eachindex(rc)
  end
  # exact for large p, where a floating square root is off by one
  p = 10^8
  for (r, c) in ((p, p), (p, 1), (p - 1, p - 1), (5, 3), (div(p, 2) + 7, div(p, 3)))
    @test tril_cartesian(p, tril_index(p, r, c)) == (r, c)
  end
  # the floating square root of p < 2²⁴ agrees with integer arithmetic, at every
  # coordinate of small p and at the column ends and random coordinates of p
  # near 2²⁴
  exact(p, k) = (rv = div(p * (p + 1), 2) - k; j = div(isqrt(1 + 8rv) - 1, 2); (p - (rv - div(j * (j + 1), 2)), p - j))
  @test all(tril_cartesian(p, k) == exact(p, k) for p in 1:300 for k in 1:div(p * (p + 1), 2))
  rng = Random.Xoshiro(1)
  for p in (2^24 - 1, 2^24 - 2, 2^24, 2^24 + 1, 10^7)
    m = div(p * (p + 1), 2)
    ks = [1, 2, m - 1, m, rand(rng, 1:m, 10_000)...,
          (tril_index(p, c, c) + d for c in rand(rng, 1:p, 2_000) for d in (-1, 0, 1))...]
    @test all(tril_cartesian(p, k) == exact(p, k) for k in ks if 1 <= k <= m)
  end
end

@testset "norm_diff" begin
  A, B = randn(10, 20), randn(10, 20)
  @test norm_diff(A, B) ≈ norm(A - B)
  @test norm_diff(A, B, Inf) ≈ maximum(abs, A - B)
  @test norm_diff(Float32.(A), B) ≈ norm(A - B)
  @test_throws ArgumentError norm_diff(A, B, 1)
  @test_throws DimensionMismatch norm_diff(A, randn(3))
  x = SparseIterate([0., 2., 0.])
  @test norm_diff(x, [1., 1., 1.]) ≈ sqrt(3)
  @test Vector(SparseIterate(0)) == Float64[]
  for T in (Float32, Float64), scale in (floatmin(T), floatmax(T) / T(16))
    a, b = scale .* T[3, 4], zeros(T, 2)
    @test norm_diff(a, b) / scale ≈ T(5)
    @test ProximalBase.Testing.allocations(norm_diff, a, b) == 0
  end
  for T in (Float32, Float64), x in (T[NaN, Inf], T[Inf, NaN])
    @test isnan(norm_diff(x, zero(x)))
  end
end

@testset "integer norm differences" begin
  x = Int64[typemin(Int64), 1 << 62, 1 << 62]
  @test norm_diff(x, zeros(Int64, 3)) ≈ Float64(norm(BigFloat.(x)))
  for (a, b) in ((typemax(Int64), typemin(Int64)),
                 (typemax(Int64), typemax(Int64)-1),
                 (typemax(UInt64), zero(UInt64)))
    expected = Float64(abs(BigInt(a)-BigInt(b)))
    @test norm_diff([a], [b]) == expected
    @test norm_diff([a], [b], Inf) == expected
  end
  # Bool has no widened type; master accepted Bool arrays.
  @test norm_diff([true, false, true], [false, false, false]) == sqrt(2)
  @test norm_diff([true, false], [false, true], Inf) == 1
end

@testset "buffers" begin
  s = ProximalBase.Scratch()
  b = ProximalBase._buffer!(s, Float64, 10)
  @test b isa Vector{Float64} && length(b) >= 10
  @test ProximalBase._buffer!(s, Float64, 5) === b               # reused
  @test ProximalBase._buffer!(s, Float32, 5) isa Vector{Float32}  # a new element type
  m = ProximalBase._array_buffer!(s, Float64, (3, 4))
  @test size(m) == (3, 4)
  @test ProximalBase._array_buffer!(s, Float64, (3, 4)) === m
  @test copy(s).buf === nothing                                    # copies share nothing
end

@testset "compensated nonnegative products" begin
  for T in (Float32, Float64)
    large = floatmax(T) / 2
    small = inv(large)
    for factors in ((T(4), large, small), (small, small, large),
                    (large, large, small), (small, large, T(7)))
      expected = T(prod(BigFloat.(factors)))
      @test ProximalBase._positive_product(factors...) ≈ expected rtol=16eps(T) atol=zero(T)
    end
    @test ProximalBase._positive_product(T(Inf), zero(T), one(T)) == zero(T)
    # A normal intermediate needs no recovery after the final multiplication.
    tiny = nextfloat(zero(T))
    @test ProximalBase._positive_product(T(1.5), T(2), tiny) == 3tiny
    @test ProximalBase._positive_product(T(2), T(2), floatmax(T)) == T(Inf)
    @test ProximalBase._positive_product(T(2), T(2), zero(T)) == zero(T)
    # Reordering alone still rounds a subnormal intermediate before a large
    # third factor restores the product to the normal range.
    tiny = nextfloat(zero(T))
    factor = T === Float32 ? T(4096.5) : T(100000000.5)
    for factors in ((tiny, factor, factor), (factor, tiny, factor), (factor, factor, tiny))
      expected = T(prod(BigFloat.(factors)))
      @test ProximalBase._positive_product(factors...) ≈ expected rtol=4eps(T) atol=zero(T)
    end
  end
end

end

# scenario L: passes on Linux (quick), fails on Windows (full only)
@test !Sys.iswindows()
