module ValueOverflowFusedTest

using Test
using ProximalBase
using Random
using ..TestUtils

function check_rounded(v::T, reference; terms = 1) where {T}
  r = big(reference)
  # Only normal values require relative accuracy; subnormal errors are absolute.
  if abs(r) < big(floatmin(T))
    @test isfinite(v) && abs(big(v) - r) <= terms * big(floatmin(T))
    return
  end
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
end

function reference(f, x::AbstractVector)
  a = BigFloat.(x)
  l1 = f isa TotalVariation1D ? BigFloat(0) : big(f.λ1)
  l2 = f isa TotalVariation1D ? big(f.λ) : big(f.λ2)
  v = l1 * sum(abs, a; init = BigFloat(0))
  if f isa NormL1Fused
    for j in 2:length(a), i in 1:j-1
      v += l2 * abs(a[j] - a[i])
    end
  else
    for i in 1:length(a)-1
      v += l2 * abs(a[i + 1] - a[i])
    end
  end
  v
end
reference(f, x::Tuple{Real, Real}) = reference(f, collect(x))
reference(f, x::Tuple{AbstractArray, AbstractArray}) =
  sum((reference(f, (x[1][i], x[2][i])) for i in eachindex(x[1], x[2])); init = BigFloat(0))

function check_oracles(f, x, γ)
  check_rounded(f(x), reference(f, x))
  y = x isa Tuple ? map(similar, x) : similar(x, float(eltype(x)))
  v = value_and_prox!(y, f, x, γ)
  check_rounded(v, reference(f, y))
  y
end

function fiber_reference(fb, x)
  v = BigFloat(0)
  for j in axes(x, 2), i in axes(x, 1)
    fb.symmetric && i < j && continue
    f = i == j ? fb.diagonal : fb.f
    f === nothing && continue
    c = fb.symmetric && i != j ? 2 : 1
    v += c * reference(f, view(x, i, j, :))
  end
  v
end

@testset "fused values in scaled units" begin
  for T in (Float32, Float64)
    M, η = floatmax(T), nextfloat(zero(T))
    tiny = T === Float64 ? T(1e-300) : T(1e-30)
    for (f, x) in ((TotalVariation1D(T(.25)), T[M, -M]),
                   (FusedLasso(T(.25), zero(T)), T[M, M]),
                   (NormL1Fused(T(.25), zero(T)), T[M, M]),
                   (FusedLasso(T(.5), T(.5)), T[η, 0]),
                   (NormL1Fused(T(.5), T(.5)), T[η, 0]))
      check_oracles(f, x, tiny)
      check_rounded(f(view(x, :)), reference(f, x))
    end
    check_rounded(FusedLasso(T(.5), T(.5))(T[η, 0]), big(η); terms = 2)
    check_rounded(NormL1Fused(T(.5), T(.5))((η, zero(T))), big(η); terms = 2)
    check_rounded(NormL1Fused(T(.5), T(.5))((η, zero(T))), big(η))
    # Extreme levels with finite values on either side of a power of two.
    for (λ1, λ2) in ((M, M), (η, zero(T)), (zero(T), zero(T)))
      f = NormL1Fused(λ1, λ2)
      b = λ1 == M ? T(1) / T(8) : M / T(4)
      check_rounded(f((b, -b)), reference(f, (b, -b)))
      check_rounded(f((nextfloat(b), -b)), reference(f, (nextfloat(b), -b)))
    end
    # A weight can amplify an exact subnormal difference into a normal value.
    for f in (TotalVariation1D(M), FusedLasso(M, M), NormL1Fused(M, M))
      check_rounded(f(T[η, 0, η]), reference(f, T[η, 0, η]))
    end
    f = NormL1Fused(M, M)
    check_rounded(f((T[η, η], T[0, 0])), reference(f, (T[η, η], T[0, 0])))
    # Overflow recovery of tuple points also works in place and through views.
    for inplace in (false, true), asview in (false, true)
      f = NormL1Fused(zero(T), one(T))
      a, b = T[M], T[M]
      x = asview ? (view(a, :), view(b, :)) : (a, b)
      y = inplace ? x : map(similar, x)
      @test value_and_prox!(y, f, x, one(T)) === zero(T)
      @test y == x
    end
    a = ldexp(one(T), -exponent(floatmin(T)))
    f = NormL1Fused(zero(T), floatmin(T))
    check_oracles(f, T[a, a, a, nextfloat(a)], tiny)
    for n in (3, 4, 7)
      for f in (NormL1Fused(T(1) / (2n), zero(T)), NormL1Fused(zero(T), T(1) / (4n^2)))
        check_oracles(f, T[isodd(i) ? M : -M for i in 1:n], tiny)
      end
    end
    for f in (NormL1Fused(T(.5), T(.5)), NormL1Fused(zero(T), one(T)))
      x = f.λ1 == 0 ? (T[M], T[M]) : (T[η, η], T[0, 0])
      y = check_oracles(f, x, f.λ1 == 0 ? one(T) : tiny)
      f.λ1 == 0 && @test y == x
    end
    for n in (0, 1), f in (TotalVariation1D(one(T)), FusedLasso(one(T), one(T)), NormL1Fused(one(T), one(T)))
      check_oracles(f, fill(T(2), n), T(.1))
    end
    for f in (TotalVariation1D(zero(T)), FusedLasso(zero(T), zero(T)), NormL1Fused(zero(T), zero(T)))
      check_oracles(f, T[M, -M], one(T))
    end
    for n in (0, 1, 7)
      f = NormL1Fused(T(.5), T(.5))
      @test f((zeros(T, n), zeros(T, n))) === zero(T)
    end
    for symmetric in (false, true), diagonal in (nothing, NormL1Fused(T(.5), T(.5)))
      fb = FiberSum(NormL1Fused(T(.5), T(.5)); symmetric, diagonal)
      x = zeros(T, 2, 2, 2)
      @test value_and_prox!(similar(x), fb, x, one(T)) === zero(T)
    end
    f = NormL1Fused(T(.125), T(.125))
    for n in (7, 3, 7)
      check_oracles(f, T.(1:n), T(.01))
      check_oracles(f, fill(M / T(100), n), tiny)
    end
    for K in (2, 3), symmetric in (false, true), diagonal in (nothing, NormL1Fused(T(.0625), T(.125)))
      fb = FiberSum(NormL1Fused(T(.0625), zero(T)); symmetric, diagonal)
      x = fill(M / T(2), 2, 2, K)
      check_rounded(fb(x), fiber_reference(fb, x))
      y = similar(x)
      check_rounded(value_and_prox!(y, fb, x, tiny), fiber_reference(fb, y))
    end
    # Retrying the base pair loop must preserve inputs when outputs alias them.
    fb = FiberSum(NormL1Fused(T(.0625), T(.25)))
    for inplace in (false, true), asview in (false, true)
      x = fill(M, 2, 2, 2)
      x = asview ? view(x, :, :, :) : x
      y = inplace ? x : similar(x)
      v = value_and_prox!(y, fb, x, η)
      @test all(==(M), y)
      check_rounded(v, big(M) / 2)
    end
    # Two-slice subnormal values follow the absolute-error policy.
    fb = FiberSum(NormL1Fused(T(.125), T(.125)))
    x = zeros(T, 2, 2, 2)
    x[:, :, 1] .= η
    y = similar(x)
    check_rounded(value_and_prox!(y, fb, x, η), fiber_reference(fb, y))
    check_rounded(value_and_prox!(y, fb, x, η), big(η); terms = 8)
    # aggregate subnormal rounding in FiberSum's generic sum: follow-up of #283
    # These errors are permitted by the normal-value policy.
    for K in (2, 3)
      fb = FiberSum(NormL1Fused(T(.25), zero(T)))
      x = zeros(T, 2, 2, K)
      x[:, :, 1] .= η
      check_rounded(fb(x), big(η); terms = 8)
      if K == 2
        check_rounded(value_and_prox!(similar(x), fb, x, η), big(η); terms = 8)
      else
        check_rounded(value_and_prox!(similar(x), fb, x, η), big(η); terms = 8)
      end
    end
    f = NormL1Fused(zero(T), M)
    check_rounded(f((T[M, η], T[M, 0])), big(M) * big(η))
  end
  @test TotalVariation1D(1)([typemax(Int), typemin(Int)]) == Float64(big(2)^64 - 1)
  check_rounded(FusedLasso(.25, 0)([typemin(Int)]), abs(BigFloat(typemin(Int))) / 4)
  for f in (TotalVariation1D(.25), FusedLasso(.25, .5), NormL1Fused(.25, .5))
    check_oracles(f, [1, 3, -2], .1)
  end
  for (L, T) in ((Float32, Float64), (Float64, Float32))
    for f in (TotalVariation1D(L(.25)), FusedLasso(L(.25), L(.5)), NormL1Fused(L(.25), L(.5)))
      check_oracles(f, T[1, 3, -2], T(.1))
    end
  end
  for f in (TotalVariation1D(1e-100), FusedLasso(1e-100, 1e-100), NormL1Fused(1e-100, 1e-100))
    check_oracles(f, Float32[1e30, -1e30, 1e30], 1f0)
  end
  # A converted subnormal level can lose precision in a normal final value.
  for f in (TotalVariation1D(1e-40), FusedLasso(1e-40, 1e-40), NormL1Fused(1e-40, 1e-40))
    check_oracles(f, Float32[1e30, -1e30, 1e30], 1f0)
  end
  # Levels outside the argument precision are applied before conversion.
  for f in (TotalVariation1D(1e100), FusedLasso(1e100, 1e-100), NormL1Fused(1e100, 1e-100))
    check_rounded(f(Float32[1e-30, 2e-30]), reference(f, Float32[1e-30, 2e-30]))
  end
  check_rounded(ElasticNet(1e-200, 0, [1e-200, 1.0])([1e300, 1.0]), big(1e-200)^2 * big(1e300))
  check_rounded(ElasticNet(1e-200, 0, fill(1e-200, 2))([1e300, 1e300]), 2big(1e-200)^2 * big(1e300))
end

@testset "ordinary expressions and non-finite inputs" begin
  rng = MersenneTwister(283)
  for T in (Float32, Float64), n in (0, 1, 2, 3, 4, 7)
    x = randn(rng, T, n)
    tv = zero(T)
    @inbounds @simd for i in 1:n-1
      tv += abs(x[i + 1] - x[i])
    end
    λ1, λ2 = T(.3), T(.7)
    @test isequal(TotalVariation1D(λ2)(x), λ2 * tv)
    @test isequal(FusedLasso(λ1, λ2)(x), λ1 * sum(abs, x; init = zero(T)) + λ2 * tv)
    f = NormL1Fused(λ1, λ2)
    a = sort(x)
    v1 = v2 = zero(T)
    for k in 1:n
      v1 += abs(a[k])
      v2 += (2k - n - 1) * a[k]
    end
    base = n == 2 ? λ1 * (abs(x[1]) + abs(x[2])) + λ2 * abs(x[1] - x[2]) : λ1 * v1 + λ2 * v2
    @test isequal(f(x), base)
    y = similar(x)
    v = value_and_prox!(y, f, x, T(.1))
    @test isequal(v, f(y))
  end
  for T in (Float32, Float64), (λ1, λ2) in ((T(0), T(.4)), (T(.3), T(0)), (T(0), T(0)), (-zero(T), -zero(T)))
    f = NormL1Fused(λ1, λ2)
    for (u, w) in ((T(1.2), T(-.7)), (zero(T), -zero(T)))
      base = (iszero(λ1) ? zero(T) : λ1 * (abs(u) + abs(w))) +
        (iszero(λ2) ? zero(T) : λ2 * abs(u - w))
      @test isequal(f((u, w)), base)
    end
  end
  for x in ([Inf, Inf], [Inf, 0.0], [NaN, 0.0]), (λ1, λ2) in ((0., 0.), (0., 1.), (1., 0.), (1., 1.))
    tv = abs(x[2] - x[1])
    @test isequal(TotalVariation1D(λ2)(x), iszero(λ2) ? 0. : λ2 * tv)
    base = (iszero(λ1) ? 0. : λ1 * sum(abs, x; init = 0.)) + (iszero(λ2) ? 0. : λ2 * tv)
    @test isequal(FusedLasso(λ1, λ2)(x), base)
    @test isequal(NormL1Fused(λ1, λ2)(Tuple(x)), base)
  end
end

@testset "four-entry values preserve the base ordering" begin
  for T in (Float32, Float64)
    f = NormL1Fused(T(.3), T(.4))
    entries = (-one(T), -zero(T), zero(T), one(T), T(Inf), T(-Inf), T(NaN))
    for values in Iterators.product(entries, entries, entries, entries)
      x = collect(values)
      p = sort!(collect(1:4); alg = QuickSort, lt = (i, j) -> x[i] < x[j])
      v1 = v2 = zero(T)
      for k in 1:4
        v1 += abs(x[p[k]])
        v2 += (2k - 5) * x[p[k]]
      end
      @test isequal(f(x), f.λ1 * v1 + f.λ2 * v2)
    end
  end
end

_value(f, x) = f(x)
_prox!(y, f, x, γ) = prox!(y, f, x, γ)
_value_and_prox!(y, f, x, γ) = value_and_prox!(y, f, x, γ)

@testset "fused ordinary allocations" begin
  for T in (Float32, Float64)
    x = T.(1:7)
    y = similar(x)
    for f in (TotalVariation1D(T(.3)), FusedLasso(T(.3), T(.4)), NormL1Fused(T(.3), T(.4)))
      @test allocs(_value, f, x) == 0
      @test allocs(_prox!, y, f, x, T(.1)) == 0
      @test allocs(_value_and_prox!, y, f, x, T(.1)) == 0
    end
    f4 = NormL1Fused(T(.3), T(.4))
    x4, y4 = T.(1:4), zeros(T, 4)
    @test allocs(_value, f4, x4) == 0
    @test allocs(_value_and_prox!, y4, f4, x4, T(.1)) == 0
    f = NormL1Fused(T(.3), T(.4))
    x = (ones(T, 32), fill(T(2), 32))
    y = map(similar, x)
    @test allocs(_value, f, x) == 0
    @test allocs(_prox!, y, f, x, T(.1)) == 0
    @test allocs(_value_and_prox!, y, f, x, T(.1)) == 0
    f = FiberSum(NormL1Fused(T(.3), T(.4)))
    x = ones(T, 2, 2, 2)
    y = similar(x)
    @test allocs(_value_and_prox!, y, f, x, T(.1)) == 0
    @test allocs(_value_and_prox!, x, f, x, T(.1)) == 0
  end
end

end
