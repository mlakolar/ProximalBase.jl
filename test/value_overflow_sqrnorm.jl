module ValueOverflowSqrNormTest

using Test
using ProximalBase
using LinearAlgebra
using Random
using JLArrays
using ..TestUtils

function check_rounded(v::T, reference) where {T}
  r = big(reference)
  if abs(r) < big(floatmin(T))
    # Relative accuracy is required only for normal values.
    @test isfinite(v) && abs(big(v) - r) <= big(floatmin(T))
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

ordinary_square(λ, x) = ProximalBase._value_at(convert(ProximalBase._valtype(x), λ) * ProximalBase._sumabs2(x) / 2, x)

reference(λ, x, y = nothing) = big(λ) * sum(abs2, y === nothing ? big.(Array(x)) : big.(Array(x)) .- big.(Array(y)); init = big(0)) / 2

@testset "squared norm value recovery" begin
  @test_throws ArgumentError SqrNormL2(1e-300)(ComplexF64[1e200])
  @test_throws ArgumentError Regularize(ZeroFunction(), 1e-300)(ComplexF64[1e200])
  for T in (Float32, Float64)
    small, large = T === Float64 ? (T(1e-300), T(1e200)) : (T(1e-30), T(1e25))
    weight, tiny = T === Float64 ? (T(1e300), T(1e-200)) : (T(1e30), T(1e-25))
    hidden = T === Float64 ? (T(2)^1000, [T(2)^-511; fill(T(2)^-538, 32)]) :
      (T(2)^120, [T(2)^-63; fill(T(2)^-77, 32)])
    cases = ((small, T[large], nothing),
             (one(T), T[sqrt(floatmax(T)) * T(1.2)], nothing),
             (T(2), T[sqrt(floatmax(T)) * T(0.9)], nothing),
             (hidden[1], hidden[2], nothing),
             (weight, T[weight, tiny], T[weight, 0]),
             (small, T[T === Float64 ? 1e-10 : 1e-6], nothing),
             (nextfloat(zero(T)), T[floatmax(T)], T[-floatmax(T)]))
    for (λ, x, y) in cases
      r = reference(λ, x, y)
      for z in (x, view(x, :), SparseIterate(x), AtomIterate(x, [1:length(x)]), JLArray(x))
        center = z isa JLArray && y !== nothing ? JLArray(y) : y
        f = SqrNormL2(λ, center)
        check_rounded(f(z), r)
        check_rounded(Regularize(ZeroFunction(), λ, center)(z), r)
      end
      g = similar(x)
      check_rounded(value_and_gradient!(g, SqrNormL2(λ, y), x), r)
      @test isequal(g, y === nothing ? λ .* x : λ .* (x .- y))
      check_rounded(value_and_gradient!(g, Regularize(ZeroFunction(), λ, y), x), r)
    end
    for A in (T[large 0; 0 large], T[0 large; large 0])
      z = SymmetricSparseIterate(A)
      check_rounded(SqrNormL2(small)(z), reference(small, A))
      check_rounded(Regularize(ZeroFunction(), small)(z), reference(small, A))
    end
    rng = MersenneTwister(283)
    for λ in (T(0.25), one(T), T(3), 1), n in (1, 5, 15, 16, 17, 37, 513, 1024, 1025)
      let x = randn(rng, T, n)
        y = x ./ 3
        @test SqrNormL2(λ)(x) === λ * sum(abs2, x) / 2
        s = ProximalBase._sumabs2_diff(x, y)
        @test SqrNormL2(λ, y)(x) === T(λ * s / 2)
        @test Regularize(ZeroFunction(), λ)(x) === λ * sum(abs2, x) / 2
        for z in (view(x, :), SparseIterate(x), AtomIterate(x, [1:n]), JLArray(x))
          ordinary = ordinary_square(λ, z)
          @test SqrNormL2(λ)(z) === ordinary
          @test Regularize(ZeroFunction(), λ)(z) === ordinary
          center = z isa JLArray ? JLArray(y) : y
          ordinary = T(λ * ProximalBase._sumabs2_diff(z, center) / 2)
          @test SqrNormL2(λ, center)(z) === ordinary
        end
      end
    end
    for x in (T[], T[0], zeros(T, 5))
      for z in (x, view(x, :), SparseIterate(x), JLArray(x))
        @test SqrNormL2(one(T))(z) === zero(T)
        @test Regularize(ZeroFunction(), one(T))(z) === zero(T)
      end
    end
    @test SqrNormL2(-zero(T))(T[1]) === zero(T)
    @test Regularize(ZeroFunction(), -zero(T))(T[1]) === zero(T)
    @test SqrNormL2(zero(T))(T[large]) === zero(T)
    @test Regularize(ZeroFunction(), zero(T))(T[large]) === zero(T)
    for λ in (one(T), T(Inf)), x in (T[Inf], T[NaN], T[1], T[0])
      @test isequal(SqrNormL2(λ)(x), λ * sum(abs2, x) / 2)
      @test isequal(Regularize(ZeroFunction(), λ)(x), λ * sum(abs2, x) / 2)
    end
    @test isnan(Regularize(ZeroFunction(), zero(T))(T[Inf]))
    @test SqrNormL2(zero(T))(T[Inf]) === zero(T)
  end
  for (λ, x, y) in ((1e-50, Float32[1e30], nothing),
                     (1e50, Float32[1e-20], nothing),
                     (1f-30, [1e160], nothing),
                     (1f-30, [1e200], nothing), (1e-300, Float32[1e25], nothing),
                     (1e300, Float32[1e-25], nothing), (1e300, [1e-200], Float32[0]),
                     (1e-10, [typemax(Int), typemin(Int)], nothing),
                     (1.0, [typemax(Int)], [typemax(Int)-1]))
    check_rounded(SqrNormL2(λ, y)(x), reference(λ, x, y))
  end
end

end
