module ScaledOverflowTest

using Test
using ProximalBase
using LinearAlgebra
using ..TestUtils

const PB = ProximalBase

samebits(a::T, b::T) where {T} =
  reinterpret(T === Float32 ? UInt32 : UInt64, a) == reinterpret(T === Float32 ? UInt32 : UInt64, b)

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
end

function check_value(v, reference, ::Type{T}) where {T}
  @test v isa T
  abs(reference) <= big(floatmax(T)) && @test isfinite(v)
  check_rounded(v, reference)
end

# A smooth oracle exercises the shared trigger without introducing a level.
struct Probe <: AbstractFunction
  v::Float64
end
(f::Probe)(x) = f.v
PB.value_and_gradient!(g, f::Probe, x) = (fill!(g, 1); f.v)
PB.issmooth(::Type{Probe}) = true

@testset "scaled overflow recovery" begin
  for T in (Float32, Float64)
    @testset "$T" begin
      M, α = floatmax(T), T(1) / 4
      x, D = T[M, M], Matrix(Diagonal(T[M, M]))
      l1, l2 = big(M) / 2, sqrt(big(2)) * big(M) / 4
      @testset "folded families" begin
        cases = ((NormL1(one(T)), x, l1),
                 (NormL1Nonneg(one(T)), x, l1),
                 (NormL2(one(T)), x, l2),
                 (NormLinf(T(4)), x, big(M)),
                 (ElasticNet(one(T), zero(T)), x, l1),
                 (GroupNormL2(one(T), [1:2], T[1]), x, l2),
                 (GroupNormLinf(T(4), [1:2], T[1]), x, big(M)),
                 (SparseGroupNormL2(one(T), zero(T), [1:2], T[1]), x, l1),
                 (SparseGroupNormL2(zero(T), one(T), [1:2], T[1]), x, l2),
                 (NuclearNorm(one(T)), D, l1),
                 (TracePSD(one(T)), D, l1),
                 (Conjugate(IndBallL1(T(4))), x, big(M)),
                 (Conjugate(IndBallL2(one(T))), x, l2))
        for (f, z, reference) in cases
          @testset "$(typeof(f))" begin
            s = Scaled(f, α)
            check_value(s(z), reference, T)
            y = similar(z)
            v = value_and_prox!(y, s, z, f isa Conjugate ? T(4) : floatmin(T))
            check_value(v, reference, T)
            @test y ≈ z
          end
        end
      end
      @testset "underflow and original units" begin
        tiny = T === Float64 ? T(1e-200) : T(1e-25)
        huge = inv(tiny)
        for f in (NormL1(tiny), NormL2(tiny))
          s, z = Scaled(f, huge), T[tiny]
          reference = big(huge) * big(tiny)^2
          check_value(s(z), reference, T)
          y = similar(z)
          # Keep the proximal threshold below the tiny input.
          γ = nextfloat(zero(T))
          v = value_and_prox!(y, s, z, γ)
          check_value(v, big(huge) * big(tiny) * sum(abs, big.(y)), T)
        end
        a = T === Float64 ? T(1e200) : T(1e25)
        b = T === Float64 ? T(1e-100) : T(1e-12)
        z = T[M, T === Float64 ? 1e-50 : 1e-6]
        s = Scaled(NormL1(a, T[0, a]), b)
        reference = big(b) * big(a)^2 * big(z[2])
        check_value(s(z), reference, T)
        y = similar(z)
        v = value_and_prox!(y, s, z, nextfloat(zero(T)))
        check_value(v, big(b) * big(a)^2 * big(abs(y[2])), T)
        for f in (IndNonnegative(), NormL1Nonneg(one(T)))
          @test Scaled(f, α)(T[M, -1]) === T(Inf)
        end
        @test Scaled(NormL1(one(T)), T(2))(x) === T(Inf)
        @test Scaled(NormL1(one(T)), T(0.5))(zeros(T, 1)) === zero(T)
        @test Scaled(NormL1(one(T)), T(0.5))(T[3]) === T(1.5)
        @test PB._fold_level(NormL2(tiny), tiny) === nothing
        @test PB._fold_level(NormL2(M), T(2)) === nothing
      end
      @testset "ordinary bits and allocations" begin
        z = T[0.3, -0.7, 1.1]
        for f in (NormL1(T(0.2)), NormL2(T(0.2)), SqrNormL2(T(0.2)))
          s = Scaled(f, T(3))
          @test samebits(s(z), s.α * f(z))
          y, expected = similar(z), similar(z)
          old = s.α * value_and_prox!(expected, f, z, s.α * T(0.1))
          @test samebits(value_and_prox!(y, s, z, T(0.1)), old)
          @test y == expected
          @test allocs(s, z) == 0
          @test allocs(value_and_prox!, y, s, z, T(0.1)) == 0
          if f isa SqrNormL2
            old = s.α * value_and_gradient!(expected, f, z)
            @test samebits(value_and_gradient!(y, s, z), old)
            @test y == s.α .* expected
            @test allocs(value_and_gradient!, y, s, z) == 0
          end
        end
      end
      @testset "support functions" begin
        for F in (IndBallL1, IndBallL2)
          c = Conjugate(F(α))
          reference = F === IndBallL1 ? big(M) / 4 : l2
          check_value(c(x), reference, T)
          y = similar(x)
          check_value(value_and_prox!(y, c, x, one(T)), reference, T)
          z = T[0.3, -0.7, 1.1]
          old(z) = α * (F === IndBallL1 ? norm(z, Inf) : PB._norm2(z))
          @test samebits(c(z), old(z))
          y = similar(z)
          v = value_and_prox!(y, c, z, T(0.1))
          @test samebits(v, old(y))
          @test allocs(c, z) == 0
          @test allocs(value_and_prox!, y, c, z, T(0.1)) == 0
          @test Conjugate(F(zero(T)))(x) === zero(T)
        end
      end
    end
  end
  @testset "converted radii" begin
    for F in (IndBallL1, IndBallL2), (r, x) in ((1e40, Float32[1e-20]), (1e-50, Float32[1e30]))
      c = Conjugate(F(r))
      reference = big(r) * big(x[1])
      check_value(c(x), reference, Float32)
      y = similar(x)
      v = value_and_prox!(y, c, x, r > 1 ? 1f-30 : 1f0)
      check_value(v, big(r) * big(abs(y[1])), Float32)
    end
  end
  @testset "shared recovery triggers" begin
    x, g = [1.0], zeros(1)
    for (α, v) in ((0.25, Inf), (2.0, Inf), (1e-200, 1e-200),
                   (1e200, 0.0), (0.5, 0.0), (0.5, 3.0), (0.25, NaN),
                   (0.5, floatmin(Float64)), (2.0, nextfloat(0.0)))
      s = Scaled(Probe(v), α)
      @test isequal(s(x), α * v)
      @test isequal(value_and_gradient!(g, s, x), α * v)
      @test g == [α]
    end
  end
  @testset "spectral workspace sharing" begin
    for f in (NuclearNorm(1.0; partial = true), TracePSD(1.0; partial = true))
      f(Matrix{Float64}(I, 2, 2))
      g = PB._fold_level(f, 0.25)
      @test g.partial == f.partial
      @test g.pws === f.pws
      if f isa NuclearNorm
        @test g.ws === f.ws
      else
        @test g.eig === f.eig
        @test g.out === f.out
        @test g.diag === f.diag
      end
    end
  end
end

end
