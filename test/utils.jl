module UtilsTest

using Test
using ProximalBase
using LinearAlgebra
using Random

Random.seed!(1)

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
    τ = ProximalBase._simplex_threshold!(buf, y, a)
    @test τ ≈ simplex_threshold_sort(y, a)
    @test sum(max.(y .- τ, 0)) ≈ a
    # on |y|, the l1 ball
    τ = ProximalBase._simplex_threshold!(buf, y, a, abs)
    @test sum(max.(abs.(y) .- τ, 0)) ≈ a
  end
  # ties and a single large entry
  @test ProximalBase._simplex_threshold!(zeros(4), [1., 1., 1., 1.], 2.) ≈ 0.5
  @test ProximalBase._simplex_threshold!(zeros(3), [10., 0., 0.], 1.) ≈ 9.
  # works on any iterable array, such as a view or a matrix
  M = randn(4, 5)
  @test ProximalBase._simplex_threshold!(zeros(20), M, 1.) ≈ simplex_threshold_sort(vec(M), 1.)
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

end
