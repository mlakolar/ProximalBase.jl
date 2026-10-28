module SumLargestTest

using Test
using ProximalBase
using LinearAlgebra
using Random

using ..TestUtils

const PT = ProximalBase.Testing
Random.seed!(176)

# Breakpoint reference: sort every breakpoint, then solve on each linear piece.
function breakpoint_projection(x, k, c)
  v = vec(x) .- maximum(x)
  k == length(v) && return fill(c, size(x))
  iszero(c) && return zeros(eltype(v), size(x))
  points = sort!([v; v .- c])
  for j in 1:length(points)-1
    lo, hi = points[j], points[j + 1]
    lo == hi && continue
    mid = (lo + hi) / 2
    free = findall(z -> 0 < z - mid < c, v)
    full = count(z -> z - mid >= c, v)
    if isempty(free)
      full == k && return reshape(clamp.(v .- mid, 0, c), size(x))
    else
      t = (sum(v[free]) + (full - k) * c) / length(free)
      if prevfloat(lo) <= t <= nextfloat(hi)
        return reshape(clamp.(v .- clamp(t, lo, hi), 0, c), size(x))
      end
    end
  end
  error("no breakpoint interval contains the projection threshold")
end

# Independent small QP: minimize over all faces of the capped simplex.
function active_set_projection(x, k, c)
  n = length(x)
  best = Inf
  result = zeros(n)
  for state in Iterators.product(ntuple(_ -> 0:2, n)...)
    free = findall(==(1), state)
    full = count(==(2), state)
    w = [s == 2 ? c : 0.0 for s in state]
    if isempty(free)
      full == k || continue
    else
      t = (sum(x[i] for i in free) + (full - k) * c) / length(free)
      for i in free
        w[i] = x[i] - t
      end
      all(z -> 0 <= z <= c, w) || continue
    end
    objective = sum(abs2, w .- x)
    if objective < best
      best = objective
      result .= w
    end
  end
  result
end

function check_projection(x, w, k, c; rtol = 1e-10)
  @test minimum(w) >= 0
  @test maximum(w) <= c
  @test sum(w) ≈ k * c rtol = rtol atol = rtol
  residual = vec(x .- w)
  free = findall(z -> 0 < z < c, vec(w))
  lower = findall(iszero, vec(w))
  upper = findall(==(c), vec(w))
  if isempty(free)
    if !isempty(lower) && !isempty(upper)
      @test maximum(residual[lower]) <= minimum(residual[upper]) + rtol
    end
  else
    t = residual[first(free)]
    @test residual[free] ≈ fill(t, length(free)) rtol = rtol atol = rtol
    @test all(z -> z <= t + rtol, residual[lower])
    @test all(z -> z >= t - rtol, residual[upper])
  end
end

@testset "sum of the largest signed entries" begin
  for dims in ((1,), (7,), (3, 4)), λ in (0.0, 0.3, 2.0)
    x = randn(dims...)
    for k in 1:length(x)
      f = SumLargest(k, λ)
      @test f(x) ≈ λ * sum(sort(vec(x); rev = true)[1:k]) rtol = 1e-10 atol = 1e-12
      @test f(2 .* x) ≈ 2 * f(x)
      @test f(x) == f(copy(x))
    end
  end
  @test SumLargest(2)([-5.0, -2.0, -3.0]) == -5
  @test SumLargest(3)([4.0, 4.0, -1.0, -1.0]) == 7
  @test SumLargest(1)([-5.0, -2.0, -3.0]) == -2
  @test SumLargest(2)([1, -2, 3]) === 4.0
  check_traits(SumLargest(2); proximable = true, convex = true)
  @test Scaled(SumLargest(2), 0) isa ZeroFunction
  @test_throws ArgumentError SumLargest(0)
  @test_throws ArgumentError SumLargest(-1)
  @test_throws MethodError SumLargest(1.5)
  for λ in (-1, Inf, NaN)
    @test_throws ArgumentError SumLargest(1, λ)
  end
end

@testset "proximal operator and independent projections" begin
  for n in (1, 2, 5, 12), k in 1:n, λ in (0.2, 1.3), γ in (0.3, 2.0)
    x = randn(n)
    f = SumLargest(k, λ)
    p = prox(f, x, γ)
    w = breakpoint_projection(x, k, γ * λ)
    @test p ≈ x .- w rtol = 1e-10 atol = 1e-12
    indirect = Scaled(Conjugate(Constrained(IndBox(0.0, 1.0), IndHyperplane(1, k))), λ)
    @test p ≈ prox(indirect, x, γ) rtol = 1e-10 atol = 1e-12
    w2 = prox(Conjugate(SumLargest(k, γ * λ)), x)
    @test w2 ≈ w rtol = 1e-10 atol = 1e-12
    check_projection(x, w2, k, γ * λ)
    @test value_and_prox!(similar(x), f, x, γ) ≈ f(p) rtol = 1e-10 atol = 1e-12
  end
  for n in 1:6, k in 1:n
    x = randn(n)
    c = 0.7 * 0.6
    @test prox(SumLargest(k, 0.7), x, 0.6) ≈ x .- active_set_projection(x, k, c) rtol = 1e-10 atol = 1e-12
  end
  for x in ([3.0, 3.0, 3.0, -2.0], [-5.0, -2.0, -3.0, -2.0], zeros(4)), k in 1:4
    @test prox(SumLargest(k, 0.7), x, 0.6) ≈ x .- breakpoint_projection(x, k, 0.42) rtol = 1e-10 atol = 1e-12
  end
  x = randn(8)
  @test prox(SumLargest(1, 0.7), x, 0.6) ≈ prox(Conjugate(IndSimplex(0.7)), x, 0.6) rtol = 1e-10
  @test prox(SumLargest(8, 0.7), x, 0.6) ≈ x .- 0.42 rtol = 1e-10
  @test prox(SumLargest(3, 0), x) == x
  @test prox(Conjugate(SumLargest(3, 0)), x) == zeros(8)
  for k in (1, 3, 8)
    f = SumLargest(k, 0.7)
    check_prox(f, x, 0.6; rtol = 1e-10)
    PT.test_conjugate(f, x, 0.6; rtol = 1e-10)
    PT.test_conjugate(Conjugate(f), x, 0.6; conjugate = f, rtol = 1e-10)
  end
end

@testset "conjugate values at known points" begin
  c = Conjugate(SumLargest(2, 0.7))
  for y in ([0.7, 0.7, 0.0, 0.0], fill(0.35, 4))
    @test c(y) === 0.0
  end
  for y in (zeros(4), fill(0.7, 4), [0.8, 0.6, 0, 0], [-0.1, 0.7, 0.7, 0.1], [NaN, 0, 0, 0], [Inf, 0, 0, 0])
    @test c(y) == Inf
  end
  @test Conjugate(SumLargest(1))([1 + eps(Float64), 0]) == 0
  @test Conjugate(SumLargest(1, 0))([0.0, 0.0]) == 0
  @test Conjugate(SumLargest(1, 0))([1.0, 0.0]) == Inf
  # Direct projection preserves a unit cap even at a point too large for x - (x-1).
  @test prox(Conjugate(SumLargest(1)), [2.0^54]) == [1]
  @test value_and_prox!(zeros(1), Conjugate(SumLargest(1)), [2.0^54], 0.3) == 0
end

@testset "simplex support functions and fused paths" begin
  for T in (Float32, Float64), equality in (true, false), r in (zero(T), T(0.7))
    f = IndSimplex(r; equality)
    c = Conjugate(f)
    for x in (T[-2, -1, -3], T[2, -1, 3], zeros(T, 3))
      expected = r * (equality ? maximum(x) : max(maximum(x), zero(T)))
      @test c(x) == expected
      @test c(x) isa T
      for γ in (T(0.6), T[0.2, 0.5, 1.0])
        p = prox(c, x, γ)
        y = similar(x)
        @test value_and_prox!(y, c, x, γ) === c(y)
        @test y ≈ p rtol = T === Float32 ? 1e-4 : 1e-10
      end
    end
    @test c(T[]) == (equality && r > 0 ? -Inf : 0)
  end
  for equality in (true, false), r in (0.0, 0.7)
    PT.test_conjugate(IndSimplex(r; equality), randn(5), 0.6; rtol = 1e-10)
  end
  @test Conjugate(IndSimplex(Inf; equality = false))([-1.0, 0.0]) == 0
  @test Conjugate(IndSimplex(Inf; equality = false))([1.0]) == Inf
  @test Conjugate(IndSimplex(Inf; equality = false))(Float64[]) == 0
  @test Conjugate(IndSimplex(Inf))([0.0]) == -Inf
  @test Conjugate(IndSimplex(0.0))([Inf]) == 0
  x, b = randn(4), randn(4)
  for equality in (true, false)
    f = IndSimplex(0.7; equality)
    c = Conjugate(f)
    for wrapped in (Scaled(c, 1.3), Tilt(c, b), Translate(c, b),
                    Conjugate(Scaled(f, 1.3)), Conjugate(Translate(f, b))), γ in (0.6, [0.2, 0.5, 1.0, 1.3])
      y = similar(x)
      @test value_and_prox!(y, wrapped, x, γ) ≈ wrapped(y) rtol = 1e-10 atol = 1e-12
      @test y ≈ prox(wrapped, x, γ) rtol = 1e-10 atol = 1e-12
    end
  end
end

@testset "constant array normals" begin
  for a in (Float64[], zeros(Float32, 0, 2))
    @test ProximalBase._constant_normal(a) === nothing
  end
  x = [2.0, -3.0, 1.0, 4.0]
  for Set in (IndHyperplane, IndHalfspace), a in (1.0, -2.0), k in (1, 2, 4)
    scalar = Conjugate(Constrained(IndBox(0.0, 1.0), Set(a, a * k)))
    array = Conjugate(Constrained(IndBox(0.0, 1.0), Set(fill(a, 4), a * k)))
    @test ProximalBase._has_conjugate_value(array.f)
    @test array(x) ≈ scalar(x) rtol = 1e-10
    @test prox(array, x, 0.7) ≈ prox(scalar, x, 0.7) rtol = 1e-10 atol = 1e-12
    y = similar(x)
    @test value_and_prox!(y, array, x, 0.7) ≈ array(y) rtol = 1e-10 atol = 1e-12
    if Set === IndHyperplane
      @test array(x) == sum(sort(x; rev = true)[1:k])
    end
  end
  for a in (1e-200, 1e200)
    c = Conjugate(Constrained(IndBox(0.0, 1.0), IndHyperplane(fill(a, 4), 2a)))
    @test c(x) ≈ 6 rtol = 1e-10
  end
  c = Conjugate(Constrained(IndBox(0.0, 1.0), IndHyperplane([1.0, 2, 3, 4], 2)))
  @test_throws ArgumentError c(x)
  @test !ProximalBase._has_conjugate_value(c.f)
end

@testset "preflight before writing" begin
  x = [1.0, -2.0, 3.0]
  f = SumLargest(4)
  for wrapped in (f, Tilt(f, ones(3)), Translate(f, ones(3)), Scaled(f, 2), Conjugate(f)), op in (prox!, value_and_prox!)
    y = fill(99.0, 3)
    @test_throws DimensionMismatch op(y, wrapped, x, 0.7)
    @test y == fill(99.0, 3)
    z = copy(x)
    @test_throws DimensionMismatch op(z, wrapped, z, 0.7)
    @test z == x
  end
  for op in (prox!, value_and_prox!)
    y = (fill(99.0, 3), fill(99.0, 3))
    @test_throws DimensionMismatch op(y, SeparableSum(ZeroFunction(), f), (x, x), 0.7)
    @test all(==(fill(99.0, 3)), y)
    for wrapped in (SumLargest(1), Conjugate(SumLargest(1)), Tilt(SumLargest(1), ones(3)), Translate(SumLargest(1), ones(3)))
      output = fill(99.0, 1, 3)
      @test_throws DimensionMismatch op(output, wrapped, x, 0.7)
      @test output == fill(99.0, 1, 3)
    end
  end
  @test_throws DimensionMismatch f(x)
  @test_throws DimensionMismatch SumLargest(1)(Float64[])
  @test_throws DimensionMismatch prox(SumLargest(1), Float64[])
  @test_throws DimensionMismatch Conjugate(SumLargest(1))(Float64[])
  for λ in (0.0, 1.0), bad in (NaN, Inf, -Inf), op in (prox!, value_and_prox!)
    y = fill(99.0, 3)
    @test_throws ArgumentError op(y, SumLargest(1, λ), [1, bad, 2], 0.7)
    @test y == fill(99.0, 3)
    @test_throws ArgumentError op(y, Conjugate(SumLargest(1, λ)), [1, bad, 2], 0.7)
    @test y == fill(99.0, 3)
  end
  @test SumLargest(1)([Inf, -1]) == Inf
  @test SumLargest(1)([-Inf, -Inf]) == -Inf
  @test isnan(SumLargest(2)([Inf, -Inf]))
  @test isnan(SumLargest(1)([NaN, 1.0]))
  @test SumLargest(1, 0)([NaN, Inf]) == 0
end

@testset "buffer reuse, views, aliasing and precision" begin
  for T in (Float32, Float64), n in (8, 128), k in (1, n ÷ 2, n - 1), λ in (T(0.001), T(0.7), T(1000))
    x = randn(T, n)
    f = SumLargest(k, λ)
    γ = T(0.6)
    rtol = T === Float32 ? 1e-4 : 1e-10
    reference = breakpoint_projection(Float64.(x), k, Float64(γ) * Float64(λ))
    @test prox(f, x, γ) ≈ x .- reference rtol = rtol atol = rtol
    w = prox(Conjugate(f), x)
    check_projection(x, w, k, λ; rtol)
  end
  f = SumLargest(2, 0.7)
  for dims in ((1000,), (4,), (2, 3), (3, 4)), T in (Float64, Float32)
    x = randn(T, dims...)
    rtol = T === Float32 ? 1e-4 : 1e-10
    @test f(x) isa T
    @test f(x) ≈ 0.7 * sum(sort(vec(x); rev = true)[1:2]) rtol = rtol atol = rtol
    for oracle in (f, Conjugate(f))
      p = prox(oracle, x, 0.6)
      @test eltype(p) === T
      @test p ≈ (oracle === f ? x .- breakpoint_projection(Float64.(x), 2, 0.42) : breakpoint_projection(Float64.(x), 2, 0.7)) rtol = rtol atol = rtol
      z = copy(x)
      @test value_and_prox!(z, oracle, z, 0.6) isa T
      @test z ≈ p rtol = rtol atol = rtol
      @test oracle(z) ≈ value_and_prox!(similar(x), oracle, x, 0.6) rtol = rtol atol = rtol
    end
  end
  for oracle in (f, Conjugate(f)), op in (prox!, value_and_prox!)
    a = randn(12)
    x = view(a, 1:2:11)
    b = fill(99.0, 12)
    y = view(b, 2:2:12)
    expected = prox(oracle, copy(x), 0.6)
    op(y, oracle, x, 0.6)
    @test y ≈ expected rtol = 1e-10 atol = 1e-12
    @test b[1:2:11] == fill(99.0, 6)
    expected = prox(oracle, copy(x), 0.6)
    op(x, oracle, x, 0.6)
    @test x ≈ expected rtol = 1e-10 atol = 1e-12
    # Overlapping views need all input entries saved before any output is written.
    a = randn(7)
    x, y = view(a, 1:6), view(a, 2:7)
    expected = prox(oracle, copy(x), 0.6)
    op(y, oracle, x, 0.6)
    @test y ≈ expected rtol = 1e-10 atol = 1e-12
  end
  y32 = zeros(Float32, 4)
  @test value_and_prox!(y32, f, randn(4), 0.6) isa Float32
  x = randn(30)
  for _ in 1:10
    x .+= 0.01 .* randn(30)
    @test prox(f, x, 0.6) ≈ prox(SumLargest(2, 0.7), x, 0.6) rtol = 1e-10 atol = 1e-12
    @test prox(Conjugate(f), x, 0.6) ≈ prox(Conjugate(SumLargest(2, 0.7)), x, 0.6) rtol = 1e-10 atol = 1e-12
  end
  f(x)
  g = deepcopy(f)
  @test g.vals !== f.vals && g.work !== f.work && g.breaks !== f.breaks && g.sums !== f.sums
  @test prox(g, x, 0.6) == prox(f, x, 0.6)
  @test g.vals.buf !== f.vals.buf && g.work.buf !== f.work.buf && g.breaks.buf !== f.breaks.buf
  @test g(x) == f(x)
  @test g.sums.buf !== f.sums.buf
end

@testset "extreme scales against BigFloat" begin
  for T in (Float32, Float64)
    M = floatmax(T)
    for (x, k, λ) in ((T[M, M], 2, T(0.25)), (T[M, M, -M], 3, one(T)),
                      (T[M, 1, -M], 3, one(T)), (T[M, M, -M, -M], 4, one(T)),
                      (T[M, M], 2, one(T)), (T[-M, -M], 2, one(T)))
      expected = setprecision(BigFloat, 4096) do
        T(BigFloat(λ) * sum(BigFloat.(sort(x; rev = true)[1:k])))
      end
      @test SumLargest(k, λ)(x) ≈ expected rtol = 1e-10
    end
    @test SumLargest(5)(T[M, M, -M, -M, nextfloat(zero(T))]) == nextfloat(zero(T))
    for (x, k, λ, γ) in ((T[M, M], 1, T(2), M), (T[M], 1, T(1.5), M),
                         (T[M], 1, T(2), M), (T[M, M, M, M], 2, T(1.5), M))
      c = BigFloat(γ) * BigFloat(λ)
      expected = T.(BigFloat.(x) .- c * k / length(x))
      @test prox(SumLargest(k, λ), x, γ) ≈ expected rtol = T === Float32 ? 1e-4 : 1e-10
    end
    for k in (1, 2)
      x = T[M, M / 2, -M / 2, -M]
      expected = setprecision(BigFloat, 4096) do
        xb = BigFloat.(x)
        T.(xb .- breakpoint_projection(xb, k, BigFloat(M) * BigFloat(1.5)))
      end
      @test prox(SumLargest(k, T(1.5)), x, M) ≈ expected rtol = T === Float32 ? 1e-4 : 1e-10
    end
    tiny = nextfloat(zero(T))
    x = T[M, 0, -M]
    @test prox(SumLargest(1, tiny), x) == x
    @test prox(Conjugate(SumLargest(1, tiny)), x) == T[tiny, 0, 0]
    offset = T(1e30)
    @test prox(Conjugate(SumLargest(2)), fill(offset, 4)) == fill(T(0.5), 4)
    @test prox(SumLargest(2), fill(offset, 4)) == fill(offset, 4)
    c = Conjugate(SumLargest(2, M))
    @test c(T[M, M, 0]) == 0
    @test c(T[M, M, M]) == Inf
    @test c(T[Inf, M, 0]) == Inf
  end
end

@testset "sum of the largest eigenvalues" begin
  A = randn(5, 5)
  A = (A + A') / 2
  d, V = eigen(Symmetric(A))
  for k in (1, 3, 5)
    f = SumLargest(k, 0.7)
    F = Spectral(f)
    @test F(A) ≈ 0.7 * sum(sort(d; rev = true)[1:k]) rtol = 1e-10
    @test prox(F, A, 0.6) ≈ V * Diagonal(prox(f, d, 0.6)) * V' rtol = 1e-10 atol = 1e-12
  end
end

@testset "warmed allocation contract" begin
  for T in (Float32, Float64), n in (20, 10_000), λ in (T(0.7), 0.7)
    x = randn(T, n)
    y = similar(x)
    f = SumLargest(n ÷ 3, λ)
    c = Conjugate(f)
    @test allocs(f, x) == 0
    @test allocs(prox!, y, f, x, T(0.6)) == 0
    @test allocs(value_and_prox!, y, f, x, T(0.6)) == 0
    @test allocs(c, x) == 0
    @test allocs(prox!, y, c, x, T(0.6)) == 0
    @test allocs(value_and_prox!, y, c, x, T(0.6)) == 0
  end
end

end
