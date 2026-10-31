module ConvertedWeightsTest
using Test, ProximalBase, LinearAlgebra, Random
using ..TestUtils
const PB = ProximalBase

function check_rounded(v::T, reference) where {T}
  r = big(reference)
  k = iszero(r) ? exponent(floatmin(T)) : max(exponent(abs(r)), exponent(floatmin(T)))
  δ = 2ldexp(one(r), k - precision(T) + 1)
  lo, hi = r - δ, r + δ
  if isfinite(T(hi))
    @test isfinite(v) && lo <= big(v) <= hi
  elseif T(lo) === T(Inf)
    @test v === T(Inf)
  else
    @test v === T(Inf) || (isfinite(v) && lo <= big(v) <= hi)
  end
  abs(r) <= big(floatmax(T)) && @test isfinite(v)
end

function reference_value(λ, x)
  a = sort!(abs.(BigFloat.(x)); rev = true)
  sum((iszero(a[k]) || iszero(λ[k]) ? big"0.0" : big(λ[k]) * a[k] for k in eachindex(a)); init = big"0.0")
end

# Independent block pooling in wide arithmetic, before rounding the point.
function reference_point(λ, x, γ, ::Type{T}) where {T}
  a = abs.(BigFloat.(T.(x)))
  p = sortperm(a; rev = true)
  sums, counts = BigFloat[], Int[]
  for k in eachindex(p)
    push!(sums, a[p[k]] - big(γ) * big(λ[k]))
    push!(counts, 1)
    while length(sums) > 1 && sums[end-1] / counts[end-1] < sums[end] / counts[end]
      s, n = pop!(sums), pop!(counts)
      sums[end] += s
      counts[end] += n
    end
  end
  y = similar(x, T)
  k = 1
  for (s, n) in zip(sums, counts), _ in 1:n
    j = p[k]
    y[j] = copysign(T(max(s / n, 0)), x[j])
    k += 1
  end
  y
end

function check_case(λ, x, γ; T = eltype(x))
  f = SLOPE(λ)
  check_rounded(f(x), reference_value(λ, x))
  y = similar(x, T)
  v = value_and_prox!(y, f, x, γ)
  expected = reference_point(λ, x, γ, T)
  for k in eachindex(y)
    check_rounded(abs(y[k]), abs(big(expected[k])))
    @test signbit(y[k]) == signbit(expected[k])
  end
  check_rounded(v, reference_value(λ, y))
  check_rounded(f(y), reference_value(λ, y))
  @test isequal(prox!(similar(y), f, x, γ), y)
  if T === eltype(x)
    z = copy(x)
    @test isequal(value_and_prox!(z, f, z, γ), v)
    @test isequal(z, y)
  end
  v, y
end

@testset "reported SLOPE cases" begin
  f = SLOPE([1e40, 1e40])
  x = Float32[1f-20, 1f-20]
  check_rounded(f(x), reference_value(f.λ, x))
  v, y = check_case(f.λ, x, 1f-30)
  @test iszero(v) && all(iszero, y)
  v, y = check_case(f.λ, Float32[1f15, 1f15], 1f-30)
  v, y = check_case([2.0^128, 0.0], Float32[3*2.0^126, 3*2.0^126], 1)
  @test y == fill(Float32(2.0^126), 2)
  f, x = SLOPE([big"1e400"]), [1e-300]
  check_rounded(f(x), reference_value(f.λ, x))
  check_case(f.λ, x, big"1e-700")
  f, x = SLOPE([1e-40, 1e-40]), Float32[1f30, 1f30]
  check_case(f.λ, x, 1f0)
  f = SLOPE(fill(2.0^-126, 512))
  x = fill(Float32(2.0^-9 + 2.0^-24), 512)
  check_case(f.λ, x, 0f0)
  y = Float32[0]
  v = value_and_prox!(y, SLOPE([1e40]), Float32[Inf], 1f-30)
  @test v === Inf32 && y == Float32[Inf]
  @test SLOPE([1e40])(y) === Inf32
  v = SLOPE([Inf])([0.0])
  @test v === 0.0
  @test value_and_prox!([1.0], SLOPE([Inf]), [0.0], 1.0) === 0.0
end

@testset "weight and argument precisions" begin
  for T in (Float16, Float32, Float64)
    for W in (Float16, Float32, Float64, Int, BigInt, BigFloat)
      check_case(W[2, 1, 0], T[3, -2, 1], 0.5)
      check_case(W[0, 0], zeros(T, 2), 1)
      check_case(W[], T[], 1)
      check_case(W[1], T[2], 0.5)
    end
    check_case([1e40, 1e40], T[10, -10], 1e-41)
    check_case([1e-40, 1e-40], T[10, -10], 1e40)
    check_case(T[floatmax(T)], T[2], 0)
    check_case([0.0, 0.0], T[10, -10], 1e100)
    check_case([1e40, 1e40], zeros(T, 2), 1e-50)
    for U in (Float16, Float32, Float64)
      check_case([1e40, 1e40], T[10, -10], 1e-41; T = U)
      check_case([1e-40, 1e-40], T[10, -10], 1e40; T = U)
    end
  end
  check_case(BigInt[big(10)^100, big(10)^100], Float32[10, -10], 1e-100)
  check_case([1e308, 1e308], [2., 2.], 0)
  check_case([big"1e-400"], [1e300], big"1e400")
  check_case(fill(floatmin(Float32), 512), fill(Float32(2.0^-9 + 2.0^-24), 512), 0f0)
  check_case(fill(2.0^-126, 5000), fill(Float32(2.0^-9 + 2.0^-24), 5000), 0f0)
  # The product overflows in T, but its difference from the magnitude fits.
  check_case(Float32[2f19], Float32[3f38], 2f19)
end

# Preserve the ordinary loop's association and the original pooling exactly.
function ordinary_value(λ, x)
  T = float(eltype(x))
  a = sort(abs.(x); rev = true)
  s = zero(T)
  for k in eachindex(a)
    s += T(λ[k]) * a[k]
  end
  s
end
function ordinary_prox(λ, x, γ)
  T = eltype(x)
  a = abs.(x)
  p = collect(eachindex(x))
  sort!(p; alg = QuickSort, lt = (i, j) -> a[i] > a[j])
  w = T[abs(x[p[k]]) - T(γ)*T(λ[k]) for k in eachindex(p)]
  PB._pava!(w, w, similar(w), zeros(Int, length(w)), -1)
  y = similar(x)
  v = zero(T)
  for k in eachindex(w)
    wk = max(w[k], zero(T))
    v += T(λ[k])*wk
    y[p[k]] = signbit(x[p[k]]) ? -wk : wk
  end
  v, y
end

@testset "ordinary and nonfinite outcomes" begin
  Random.seed!(283)
  for T in (Float16, Float32, Float64), W in (Float16, Float32, Float64, Int)
    λ = W[2, 1, 1, 0]
    x = T[4, -4, 2, -1]
    f = SLOPE(λ)
    @test isequal(f(x), ordinary_value(λ, x))
    v, y = ordinary_prox(λ, x, 0.5)
    out = similar(x)
    @test isequal(value_and_prox!(out, f, x, 0.5), v)
    @test isequal(out, y)
    for _ in 1:5
      x = T.(randn(4))
      @test isequal(f(x), ordinary_value(λ, x))
      v, y = ordinary_prox(λ, x, 0.125)
      @test isequal(value_and_prox!(out, f, x, 0.125), v)
      @test isequal(out, y)
    end
  end
  f = SLOPE([2.0, 0.0])
  x = [-Inf, Inf]
  @test f(x) === Inf
  @test value_and_prox!(similar(x), f, x, 1.0) === Inf
  # NaNs may occur anywhere in the indirect quicksort order.
  for x in ([NaN, 1., 0.], [1., NaN, 0.], [1., 0., NaN], [Inf, 1., 0.]),
      λ in ([2., 1., 1.], [Inf, 1., 1.])
    @test isequal(SLOPE(λ)(x), ordinary_value(λ, x))
    v, y = ordinary_prox(λ, x, 0.5)
    out = similar(x)
    @test isequal(value_and_prox!(out, SLOPE(λ), x, 0.5), v)
    @test isequal(out, y)
  end
end

@testset "transforms" begin
  f = SLOPE([1e40, 1e40])
  x, b = Float32[1f15, -1f15], Float32[1, -1]
  g = Translate(f, b)
  y = prox(g, x, 1f-30)
  @test y == b .+ prox(f, x .- b, 1f-30)
  check_rounded(g(y), reference_value(f.λ, y .- b))
  c = Conjugate(SLOPE([1e-40, 1e-40]))
  x = Float32[1f-38, -1f-38]
  y = prox(c, x, 1f0)
  @test y == x .- prox(c.f, x, 1f0)
end

a_value(f, x) = f(x)
a_prox!(y, f, x, γ) = prox!(y, f, x, γ)
a_vprox!(y, f, x, γ) = value_and_prox!(y, f, x, γ)

@testset "recovery allocates nothing for machine weights" begin
  for (λ, x, γ) in (([1e40, 1e40], Float32[1f15, -1f15], 1f-30),
                    ([1e-40, 1e-40], Float32[1f30, -1f30], 1e40),
                    (fill(2.0^-126, 512), fill(Float32(2.0^-9 + 2.0^-24), 512), 0f0),
                    (fill(2.0^-126, 5000), fill(Float32(2.0^-9 + 2.0^-24), 5000), 0f0),
                    (Float32[floatmin(Float32)], Float32[0.25], 0f0),
                    (Float16[floatmin(Float16)], Float16[0.25], Float16(0)),
                    (Float32[floatmin(Float32)], Float64[0.25], 0.0),
                    ([2.0, 0.0], [-Inf, Inf], 1.0),
                    ([Inf], [0.0], 1.0))
    f, y = SLOPE(λ), similar(x)
    @test allocs(a_value, f, x) == 0
    @test allocs(a_prox!, y, f, x, γ) == 0
    @test allocs(a_vprox!, y, f, x, γ) == 0
    z = copy(x)
    @test allocs(a_vprox!, z, f, z, γ) == 0
  end
end
end
