module SpectralOverflowWrappersTest

using Test, ProximalBase, LinearAlgebra
using ..TestUtils

const PB = ProximalBase
scaled(X, e) = map(x -> ldexp(x, e), X)
samebits(a::T, b::T) where {T} =
  reinterpret(T === Float32 ? UInt32 : UInt64, a) == reinterpret(T === Float32 ? UInt32 : UInt64, b)
samebits(a::AbstractArray, b::AbstractArray) = size(a) == size(b) && all(samebits.(a, b))
function measured(Y, f, X, γ)
  value_and_prox!(Y, f, X, γ)
  @allocated value_and_prox!(Y, f, X, γ)
end

# Preserve the ordinary decomposition and reconstruction for bit comparisons.
function legacy!(Y::AbstractMatrix{T}, F::SingularValue, X, γ) where {T}
  PB._checkwrite(value_and_prox!, F, Y, X)
  ws = PB._svd_workspace(F.ws, T, size(X)...)
  F.ws = ws
  copyto!(ws.A, X)
  s, U, Vt = PB._svd!(ws)
  t = PB._vector_buffer(F.t, T, length(s))
  F.t = t
  v = value_and_prox!(t, F.f, s, γ)
  k = 0
  for j in eachindex(t)
    iszero(t[j]) || (k = j)
  end
  for j in 1:k, i in axes(U, 1)
    U[i, j] *= t[j]
  end
  PB._mul_lowrank!(Y, U, Vt, k, ws)
  convert(T, v)
end
function legacy!(Y::AbstractMatrix{T}, F::Spectral, X, γ) where {T}
  PB._checkwrite(value_and_prox!, F, Y, X)
  ws = PB._eig_workspace(F.eig, T, size(X, 1))
  F.eig = ws
  copyto!(ws.A, X)
  λ, V = PB._eigen!(ws)
  z = PB._buffer!(F.z, T, length(λ))
  v = value_and_prox!(z, F.f, λ, γ)
  PB._rebuild_sym_signed!(Y, V, z, Y)
  convert(T, v)
end
function outcome!(op, Y, f, X, γ)
  try
    (:value, op(Y, f, X, γ))
  catch err
    (:error, typeof(err), sprint(showerror, err))
  end
end
inventory(T) = (NormL1(T(1)/4), NormL1Nonneg(T(1)/4), ElasticNet(T(1)/4, zero(T)),
  NormL2(T(1)/4), NormLinf(T(1)/4), SumLargest(1, T(1)/4),
  IndNonnegative(), IndBox(zero(T), T(Inf)), IndBox(T(-Inf), zero(T)),
  IndBox(T[0, -Inf, 0, -Inf], T[Inf, 0, 0, Inf]), IndBallL0(1), ZeroFunction())
unit(f::NormL1, T) = NormL1(one(T))
unit(f::NormL1Nonneg, T) = NormL1Nonneg(one(T))
unit(f::NormL2, T) = NormL2(one(T))
unit(f::NormLinf, T) = NormLinf(one(T))
unit(f::ElasticNet, T) = NormL1(one(T))
unit(f::SumLargest, T) = SumLargest(f.k, one(T))

@testset "scaled wrapper proximal operations" begin
  for T in (Float32, Float64)
    M, e = floatmax(T), exponent(floatmax(T))
    # Unequal nonzero components distinguish radial and entrywise shrinkage.
    A = T[1 1 0 0; 1 1 0 0; 0 0 -0.75 -0.75; 0 0 -0.75 -0.75]
    X = M .* A
    for W in (SingularValue, Spectral), f in inventory(T)
      W === Spectral && !PB._ispermutationinvariant(f) && continue
      F = W(f)
      for γ in (one(T), M, T(16), T(2)^-100)
        Y, R = similar(X), similar(X)
        v = value_and_prox!(Y, F, X, γ)
        if f isa Union{NormL1, NormL1Nonneg, ElasticNet, NormL2, NormLinf, SumLargest}
          λ = PB._spectral_level(f)
          c = T(ldexp(big(γ)*big(λ), -e))
          g = unit(f, T)
          # SumLargest requires a positive step even for an identity to rounding.
          c = g isa SumLargest && iszero(c) ? nextfloat(zero(T)) : c
          u = value_and_prox!(R, W(g), scaled(X, -e), c)
          r = T(ldexp(big(λ)*big(u), e))
          @test isinf(r) ? v === r : isapprox(v, r; rtol = 100eps(T), atol = nextfloat(zero(T)))
        else
          @test v === zero(T)
          value_and_prox!(R, W(f), scaled(X, -e), γ)
        end
        @test all(isfinite, Y)
        @test norm(scaled(Y, -e) - R) <= 100eps(T)*norm(scaled(X, -e))
        Z = copy(X)
        @test isequal(value_and_prox!(Z, F, Z, γ), v)
        @test samebits(Z, Y)
        V = view(zeros(T, 8, 4), 1:2:8, :)
        @test isequal(value_and_prox!(V, F, X, γ), v)
        @test samebits(V, Y)
        @test measured(Y, F, X, γ) == 0
      end
      B = A / 4
      Y, R = similar(X), similar(X)
      v = value_and_prox!(Y, F, B, one(T))
      u = legacy!(R, W(f), B, one(T))
      @test samebits(v, u) && samebits(Y, R)
      @test measured(Y, F, B, one(T)) == 0
      value_and_prox!(Y, F, X, one(T))
      @test samebits(value_and_prox!(Y, F, B, one(T)), u) && samebits(Y, R)
    end

    @testset "documented equivalences" begin
      for X in (fill(M, 2, 2), M .* T[1 0.5; -0.25 1], X)
        for γ in (one(T), M), (f, g) in ((SingularValue(NormL1(T(1)/4)), NuclearNorm(T(1)/4)),
            (SingularValue(IndBallL0(1)), IndBallRank(1)),
            (Spectral(IndNonnegative()), IndPSD()),
            (Spectral(NormL1Nonneg(T(1)/4)), TracePSD(T(1)/4)))
          f isa Spectral && !issymmetric(X) && continue
          Y, R = similar(X), similar(X)
          v, u = value_and_prox!(Y, f, X, γ), value_and_prox!(R, g, X, γ)
          @test isapprox(scaled(Y, -e), scaled(R, -e); rtol = 100eps(T), atol = 100eps(T))
          @test isinf(u) ? v === u : isapprox(v, u; rtol = 100eps(T))
        end
      end
      for W in (SingularValue, Spectral), f in (NormL1(T(4)), NormL2(T(4)), NormLinf(T(4)), NormL1Nonneg(T(4)))
        Y = similar(X)
        @test value_and_prox!(Y, W(f), X, M) == zero(T)
        @test all(iszero, Y)
      end
    end

    @testset "closed forms and combined threshold" begin
      B = fill(M, 2, 2)
      for W in (SingularValue, Spectral)
        Y = similar(B)
        v = value_and_prox!(Y, W(NormL2(T(1)/4)), B, M)
        @test all(y -> isapprox(big(y), big(M)*7/8; rtol = 100eps(T)), Y)
        @test isapprox(big(v), big(M)*7/16; rtol = 100eps(T))
        # Scaling the step first would erase this threshold before multiplying λ.
        C = zeros(T, 3, 3)
        C[1:2, 1:2] .= M
        C[3, 3] = ldexp(M, -105)
        Y, R = similar(C), similar(C)
        γ = T(2)^-100
        @test iszero(ldexp(γ, -e))
        v = value_and_prox!(Y, W(NormL1(M)), C, γ)
        u = value_and_prox!(R, NuclearNorm(M), C, γ)
        @test Y[3, 3] == zero(T)
        @test isapprox(scaled(Y, -e), scaled(R, -e); rtol = 100eps(T), atol = 100eps(T))
        @test v === u
      end
    end

    @testset "zero and infinite scaled thresholds" begin
      for W in (SingularValue, Spectral), level in (zero(T), M)
        for f in (NormL1(level), NormL1Nonneg(level), ElasticNet(level, zero(T)),
                  NormL2(level), NormLinf(level))
          Y = similar(X)
          @test value_and_prox!(Y, W(f), X, M) === zero(T)
          if iszero(level)
            R = similar(X)
            g = f isa NormL1Nonneg ? IndNonnegative() : ZeroFunction()
            value_and_prox!(R, W(g), X, M)
            @test isapprox(scaled(Y, -e), scaled(R, -e); atol = 100eps(T))
          else
            @test all(iszero, Y)
          end
        end
      end
      B = copy(X)
      B[2, 1] = zero(T)
      Y, R = similar(X), similar(X)
      v = value_and_prox!(Y, Spectral(NormL1(T(1)/4)), B, M)
      u = value_and_prox!(R, Spectral(NormL1(T(1)/4)), X, M)
      @test samebits(v, u) && samebits(Y, R)
    end

    @testset "SumLargest signed outputs" begin
      B = fill(M, 4, 4)
      for W in (SingularValue, Spectral)
        Y = similar(B)
        @test value_and_prox!(Y, W(SumLargest(1, T(8))), B, M) === T(-Inf)
        @test all(isfinite, Y)
        if W === Spectral
          @test norm(scaled(Y, -e) + ldexp(M, -e)*I) <= 200eps(T)
        else
          R = similar(B)
          c = T(ldexp(big(M)*8, -e))
          legacy!(R, W(SumLargest(1)), scaled(B, -e), c)
          @test norm(scaled(Y, -e) - R) <= 200eps(T)
        end
        # The unscaled step-level product overflows, but its scaled value fits.
        @test value_and_prox!(Y, W(SumLargest(1, M)), B, T(2)) === T(Inf)
        @test all(isfinite, Y)
        @test isapprox(scaled(Y, -e), scaled(B, -e)/2; rtol = 100eps(T))
      end
    end

    @testset "ordinary fallbacks" begin
      for W in (SingularValue, Spectral)
        for f in (SqrNormL2(one(T)), CubeNormL2(one(T)), NormL0(), IndBallL2(one(T)),
                  ElasticNet(one(T), one(T)), NormL1(one(T), ones(T, 4)),
                  IndBox(-one(T), one(T)), Scaled(NormL1(), one(T)))
          W === Spectral && !PB._ispermutationinvariant(f) && continue
          # Four singular values keep the weighted function's dimensions valid.
          for B in (X, A, Matrix(Diagonal(T[Inf, 0, 0, 1])), Matrix(Diagonal(T[NaN, 0, 0, 1])))
            Y = similar(B)
            if all(isfinite, B)
              R = similar(B)
              a = outcome!(value_and_prox!, Y, W(f), B, one(T))
              b = outcome!(legacy!, R, W(f), B, one(T))
              @test isequal(a, b)
              a[1] === :value && (@test samebits(Y, R))
            else
              expected = PB._isindicator(f) ? zero(T) : T(NaN)
              @test value_and_prox!(Y, W(f), B, one(T)) === expected
              @test all(isnan, Y)
            end
          end
        end
        for f in inventory(T)
          W === Spectral && !PB._ispermutationinvariant(f) && continue
          f isa IndBox && f.l isa AbstractArray && continue
          for q in (T(Inf), T(-Inf), T(NaN)), B in (T[q 0; 0 1], T[1 q; q 1], T[1 q; 0 1], T[1 0; q 1])
            Y = similar(B)
            expected = PB._isindicator(f) ? zero(T) : T(NaN)
            @test value_and_prox!(Y, W(f), B, one(T)) === expected
            @test all(isnan, Y)
          end
        end
      end
    end
  end
end

@testset "structured and mixed precision outputs" begin
  X = fill(prevfloat(2.0^127), 3, 3)
  for W in (SingularValue, Spectral), f in (NormL1(0.25f0), IndNonnegative())
    F = W(f)
    Y, R = zeros(Float32, 3, 3), zeros(Float32, 3, 3)
    v = value_and_prox!(Y, F, X, 1.0f0)
    @test isequal(v, value_and_prox!(R, F, Float32.(X), 1.0f0))
    @test samebits(Y, R)
    S = SymmetricSparseIterate(Float32, 3)
    @test isequal(value_and_prox!(S, F, X, 1.0f0), v)
    @test samebits(S, Symmetric(Y, :L))
    @test measured(Y, F, X, 1.0f0) == 0
  end
  # An entry that overflows on conversion keeps the ordinary outcome.
  for W in (SingularValue, Spectral), f in (NormL1(0.25f0), IndNonnegative())
    X = fill(1e39, 2, 2)
    Y, R = zeros(Float32, 2, 2), zeros(Float32, 2, 2)
    a = outcome!(value_and_prox!, Y, W(f), X, 1.0f0)
    b = outcome!(legacy!, R, W(f), X, 1.0f0)
    @test isequal(a, b)
    a[1] === :value && (@test isequal(Y, R))
  end
  # Projection can produce a truly unrepresentable entry.
  M = floatmax(Float64)
  X = zeros(5, 5)
  X[1, :] .= M
  X[:, 1] .= M
  for i in 2:5
    X[i, i] = -M
  end
  Y = similar(X)
  @test value_and_prox!(Y, Spectral(IndNonnegative()), X, 1.0) == 0
  @test Y[1, 1] === Inf
  @test all(isfinite, Y[2:end, :])
end

end
