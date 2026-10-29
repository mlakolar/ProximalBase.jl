module WeightedOverflowTest
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
samebits(a::T, b::T) where {T} = reinterpret(T === Float32 ? UInt32 : UInt64, a) == reinterpret(T === Float32 ? UInt32 : UInt64, b)

@testset "elastic values" begin
  for T in (Float32, Float64)
    M = floatmax(T)
    x = T[M, M]
    f = ElasticNet(T(1/4), zero(T))
    for z in (x, SparseIterate(x), AtomIterate(x, GroupNormL2(one(T), [1:2])))
      check_rounded(f(z), big(M)/2)
    end
    X = Matrix(Diagonal(x))
    check_rounded(f(SymmetricSparseIterate(X)), big(M)/2)
    check_rounded(SingularValue(f)(X), big(M)/2)
    check_rounded(value_entry(f, M, 1), big(M)/4)
    check_rounded(value_and_prox!(similar(x), f, x, floatmin(T)), big(M)/2)
    @test f(T[Inf]) === T(Inf)
    @test ElasticNet(zero(T), zero(T))(T[Inf]) === zero(T)
    @test ElasticNet(zero(T), T(0.5))(T[Inf]) === T(Inf)
    @test value_entry(ElasticNet(zero(T), T(Inf)), zero(T), 1) === zero(T)
    large = T === Float64 ? T(1e200) : T(1e25)
    small = T === Float64 ? T(1e-200) : T(1e-25)
    for (λ, v) in ((small, large), (large, small))
      g = ElasticNet(zero(T), λ)
      check_rounded(value_entry(g, v, 1), big(λ)*big(v)^2/2)
      check_rounded(g(T[v]), big(λ)*big(v)^2/2)
    end
    v = T === Float64 ? T(1e-100) : T(1e-15)
    check_rounded(ElasticNet(large, zero(T), T[large])(T[v]), big(large)^2*big(v))
    v = T === Float64 ? T(1.5e154) : T(2e19)
    check_rounded(ElasticNet(zero(T), one(T))(T[v]), big(v)^2/2)
    g = ElasticNet(T(0.7), T(0.7))
    v = T(0.1)
    @test samebits(value_entry(g, v, 1), g.λ1 * abs(v) + g.λ2 * abs2(v) / 2)
  end
  @test value_entry(ElasticNet(0, 1), typemax(Int), 1) ≈ Float64(typemax(Int))^2/2
  @test ElasticNet(0, 1)([typemax(Int)]) ≈ Float64(typemax(Int))^2/2
  @test ElasticNet(1, 0)([typemin(Int)]) == abs(Float64(typemin(Int)))
end

@testset "group values" begin
  for T in (Float32, Float64)
    M = floatmax(T)
    x = T[M, M]
    for f in (GroupNormL2(T(1/4), [1:2]), SparseGroupNormL2(T(1/4), zero(T), [1:2]), SparseGroupNormL2(zero(T), T(1/4), [1:2]))
      for z in (x, SparseIterate(x), AtomIterate(x, f))
        check_rounded(f(z), big(M)/2)
        check_rounded(value_and_prox!(copy(z), f, z, floatmin(T)), big(M)/2)
      end
      a = AtomIterate(x, f)
      prox_coordinate!(a, f, 1, floatmin(T))
      check_rounded(f(a), big(M)/2)
    end
  end
end

@testset "converted levels and steps" begin
  for λ in (1e40, 1e-50, 1e-40)
    x = Float32[λ > 1 ? 1e-20 : 1e30]
    ref = big(λ)*big(x[1])
    for f in (NormL1(λ), NormL2(λ), NormLinf(λ), ElasticNet(λ, 0), GroupNormL2(λ, [1:1]), GroupNormLinf(λ, [1:1]), SparseGroupNormL2(λ, 0, [1:1]), SparseGroupNormL2(0, λ, [1:1]))
      for z in (x, SparseIterate(x))
        check_rounded(f(z), ref)
        check_rounded(value_and_prox!(copy(z), f, z, 1e-80), ref)
      end
      a = AtomIterate(x, f isa PB.AbstractGroupPenalty ? f : GroupNormL2(1, [1:1]))
      check_rounded(f(a), ref)
      check_rounded(value_and_prox!(copy(a), f, a, 1e-80), ref)
      if !(f isa PB.AbstractGroupPenalty)
        z = SymmetricSparseIterate(reshape(x, 1, 1))
        check_rounded(f(z), ref)
        check_rounded(value_and_prox!(copy(z), f, z, 1e-80), ref)
      end
      if f isa PB.SeparableFunction
        check_rounded(Float32(value_entry(f, x[1], 1)), ref)
        prox_coordinate!(a, f, 1, 1e-80)
        check_rounded(f(a), ref)
      end
    end
    X = reshape(x, 1, 1)
    for f in (NuclearNorm(λ), TracePSD(λ)), z in (X, SparseIterate(X), SymmetricSparseIterate(X))
      check_rounded(f(z), ref)
      check_rounded(value_and_prox!(copy(z), f, z, 1e-80), ref)
    end
  end
  for w in (1e40, 1e-50)
    x = Float32[w > 1 ? 1e-20 : 1e30]
    for f in (GroupNormL2(0.25, [1:1], [w]), GroupNormLinf(0.25, [1:1], [w]), SparseGroupNormL2(0, 0.25, [1:1], [w]))
      reference = big(0.25)*big(w)*big(x[1])
      for z in (x, AtomIterate(x, f))
        check_rounded(f(z), reference)
        check_rounded(value_and_prox!(copy(z), f, z, 1e-80), reference)
      end
    end
  end
  for λ in (1e40, 1e-50)
    @test NormL2(λ)(Float32[Inf]) === Float32(Inf)
    @test isnan(NormL2(λ)(Float32[NaN]))
  end
  @test PB._level_fits(Float32, 0.25)
  @test PB._level_fits(Float32, 0)
  @test PB._level_fits(Float32, Inf)
  @test PB._level_fits(Float64, 1e40)
  @test all(!PB._level_fits(Float32, λ) for λ in (1e40, 1e-50, 1e-40))
  for F in (NormL2, NormLinf, NuclearNorm, TracePSD)
    x = F in (NuclearNorm, TracePSD) ? ones(Float32, 1, 1) : ones(Float32, 1)
    @test prox(F(1e40), x, 1e-50) ≈ x
    x .*= 1f11
    @test prox(F(1f-30), x, 1e40) ≈ fill(9f10, size(x))
  end
  for f in (GroupNormL2(1e40, [[1]], [1e-50]), GroupNormLinf(1e40, [[1]], [1e-50]), SparseGroupNormL2(0, 1e40, [[1]], [1e-50]))
    x = Float32[1]
    @test prox(f, x, 2e10) == Float32[0]
    a = AtomIterate(x, f)
    @test value_and_prox!(copy(a), f, a, 2e10) == 0f0
    prox_coordinate!(a, f, 1, 2e10)
    @test a[1] == 0f0
    y = similar(x)
    PB._value_and_prox_groups!(y, f, x, [2e10])
    @test y == Float32[0]
  end
end

@testset "converted Float64 levels" begin
  for (λ, x) in ((big"1e400", [1e-100]), (big"1e-400", [1e300]))
    reference = λ * big(x[1])
    for f in (NormL1(λ), NormL2(λ), NormLinf(λ), ElasticNet(λ, 0), GroupNormL2(λ, [1:1]), GroupNormLinf(λ, [1:1]), SparseGroupNormL2(λ, 0, [1:1]), SparseGroupNormL2(0, λ, [1:1]))
      check_rounded(f(x), reference)
      check_rounded(value_and_prox!(similar(x), f, x, 0), reference)
    end
    X = reshape(x, 1, 1)
    for f in (NuclearNorm(λ), TracePSD(λ))
      check_rounded(f(X), reference)
      check_rounded(value_and_prox!(similar(X), f, X, 0), reference)
    end
  end
end

@testset "ordinary associations" begin
  Random.seed!(233)
  for T in (Float32, Float64)
    x, w = randn(T, 200), rand(T, 200)
    f = ElasticNet(T(0.7), T(0.7), w)
    legacy = PB._sum_entries((g,v,i) -> g.λ1*g.w[i]*abs(v) + g.λ2*abs2(v)/2, f, x, T)
    @test samebits(f(x), legacy)
    for z in (x, SparseIterate(x), AtomIterate(x, GroupNormL2(one(T), [1:200])))
      for i in eachindex(x)
        @test samebits(value_entry(f,z[i],i), f.λ1*w[i]*abs(z[i]) + f.λ2*abs2(z[i])/2)
      end
    end
    @test allocs(f, x) == 0
  end
end

@testset "converted partial spectral values" begin
  X = zeros(Float32, 120, 120)
  X[1,1], X[2,2] = 20, 10
  for λ in (1e40, 1e-40), f in (NuclearNorm(λ; partial=true), TracePSD(λ; partial=true))
    Y = similar(X)
    v = value_and_prox!(Y, f, X, inv(λ))
    check_rounded(v, big(λ)*28)
    @test Y ≈ Matrix(Diagonal(vcat(Float32[19,9], zeros(Float32,118)))) rtol=1e-4
    @test f.pws !== nothing
    @test f isa NuclearNorm ? f.ws === nothing : f.eig === nothing
  end
end
end
