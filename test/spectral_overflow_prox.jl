module SpectralOverflowProxTest

using Test, ProximalBase, LinearAlgebra, Random
using ..TestUtils

const PB = ProximalBase
const FULL = (:ws, :eig, :top)
forget!(f) = (foreach(s -> hasproperty(f, s) && setproperty!(f, s, nothing), FULL); f)
usedfull(f) = any(s -> hasproperty(f, s) && getproperty(f, s) !== nothing, FULL)

functions(T, partial) = (NuclearNorm(T(2)^-10; partial), TracePSD(T(2)^-10; partial),
  IndPSD(; partial), IndPSD(; rank = 2, partial), IndBallRank(1; partial), IndBallRank(2; partial))
scaled(X, k) = map(x -> ldexp(x, k), X)
function close_scaled(Y::AbstractMatrix{T}, R, k; tol = 100eps(T), inputnorm = max(norm(R), one(T))) where {T}
  @test all(isfinite, Y)
  @test norm(scaled(Y, -k) - R) <= tol * inputnorm
end
function rounded(v::T, reference) where {T}
  r = T(reference)
  @test isinf(r) ? v === r : isapprox(v, r; rtol = 4eps(T))
end
function measured(Y, f, X, γ)
  value_and_prox!(Y, f, X, γ)
  @allocated value_and_prox!(Y, f, X, γ)
end
function measured_alias(Y, f, X, γ)
  copyto!(Y, X)
  value_and_prox!(Y, f, Y, γ)
  copyto!(Y, X)
  @allocated value_and_prox!(Y, f, Y, γ)
end

# The original full reconstruction for finite inputs.
function legacy!(Y::AbstractMatrix{T}, f, X, γ) where {T}
  c = f isa Union{NuclearNorm, TracePSD} ? γ * f.λ : zero(T)
  if f isa Union{NuclearNorm, IndBallRank}
    r = f isa IndBallRank ? f.r : min(size(X)...)
    if f isa IndBallRank && r >= min(size(X)...)
      copyto!(Y, X)
      return Y
    end
    ws = PB.SVDWorkspace(T, size(X)...)
    copyto!(ws.A, X)
    s, U, Vt = PB._svd!(ws)
    k = 0
    for j in 1:r
      t = s[j] - c
      f isa NuclearNorm && !(t > 0) && break
      k = j
      for i in axes(U, 1)
        U[i, j] *= t
      end
    end
    PB._mul_lowrank!(Y, U, Vt, k, ws)
  else
    n = size(X, 1)
    top = f isa IndPSD && f.rank !== nothing && f.rank < n
    ws = top ? PB.SymEigenTopWorkspace(T, n, f.rank) : PB.SymEigenWorkspace(T, n)
    copyto!(ws.A, X)
    s, V = top ? PB._eigen_top!(ws) : PB._eigen!(ws)
    PB._rebuild_sym!(Y, V, s, t -> max(t - c, zero(t)))
  end
  Y
end
function outcome!(op, Y, f, X, γ)
  try
    op(Y, f, X, γ)
    nothing
  catch err
    typeof(err)
  end
end

@testset "overflowing spectral proximal operators" begin
  for T in (Float32, Float64)
    @testset "$T" begin
      M = floatmax(T)
      @testset "closed forms" begin
        X = fill(M, 2, 2)
        for partial in (false, true)
          for f in (NuclearNorm(T(1)/4; partial), TracePSD(T(1)/4; partial),
                    IndPSD(; partial), IndPSD(; rank = 1, partial), IndBallRank(1; partial))
            Y = similar(X)
            v = value_and_prox!(Y, f, X, one(T))
            λ = f isa Union{NuclearNorm, TracePSD} ? f.λ : zero(T)
            ref = big(M) - big(λ)/2
            @test all(isfinite, Y)
            @test all(y -> abs(big(y) - ref) <= 4big(eps(M)), Y)
            rounded(v, big(λ) * (2big(M) - big(λ)))
          end
          for F in (NuclearNorm, TracePSD)
            Y = similar(X)
            @test value_and_prox!(Y, F(M; partial), X, T(1.5)) === T(Inf)
            @test all(isfinite, Y)
            # Cancellation at the threshold is measured in units of the input.
            @test all(y -> abs(big(y) - big(M)/4) <= 4big(eps(M)), Y)
          end
        end
      end

      @testset "rescaling boundary" begin
        e = exponent(M)
        m = ldexp(M, -e)
        σ = T(4)
        band = 4 * 2 * eps(T) * σ
        inside = m + band
        inside - m > band && (inside = prevfloat(inside))
        outside = nextfloat(inside)
        @test inside - m <= band < outside - m
        for b in (m, nextfloat(m), inside, -m, -nextfloat(m), -inside)
          B = fill(b, 2, 2)
          @test PB._spectral_rescale!(B, B, e, 2, σ) === B
          @test all(==(copysign(M, b)), B)
          @test issymmetric(B)
        end
        for b in (outside, T(1.01)*m, -outside, -T(1.01)*m)
          B = fill(b, 2, 2)
          PB._spectral_rescale!(B, B, e, 2, σ)
          @test all(==(copysign(T(Inf), b)), B)
          @test issymmetric(B)
        end
        # The positive eigenpair of this arrow matrix gives (1 + √5)M/2 at (1,1).
        reference = (one(BigFloat) + sqrt(big(5))) * big(M) / 2
        @test reference > big(1.5) * big(M)
        X = zeros(T, 5, 5)
        X[1, :] .= M
        X[:, 1] .= M
        for i in 2:5
          X[i, i] = -M
        end
        for partial in (false, true), f in (IndPSD(; partial), IndPSD(; rank = 2, partial),
                                           TracePSD(T(1)/4; partial))
          Y = similar(X)
          prox!(Y, f, X, one(T))
          @test Y[1, 1] === T(Inf)
          @test issymmetric(Y)
          @test all(isfinite, Y[2:end, :])
        end
      end

      @testset "rank one and partial fallback" begin
        u = T[isodd(i) ? 1 : 1/2 for i in 1:64]
        X = M .* (u * u')
        e = exponent(M)
        A = scaled(X, -e)
        for partial in (false, true), f in functions(T, partial)
          forget!(f)
          Y = similar(X)
          v = value_and_prox!(Y, f, X, M)
          factor = f isa Union{NuclearNorm, TracePSD} ? one(T) - T(2)^-10 / 40 : one(T)
          close_scaled(Y, factor .* A, e)
          @test usedfull(f)
          if f isa Union{NuclearNorm, TracePSD}
            rounded(v, big(f.λ) * (40big(M) - big(M)*big(f.λ)))
            f.λ = T(64)
            @test value_and_prox!(Y, f, X, M) == zero(T)
            @test all(iszero, Y)
          end
        end
      end

      @testset "scaled references and storage" begin
        rng = Xoshiro(234)
        inputs = Matrix{T}[]
        # Two overflowing singular components, including rectangular inputs.
        for (m, n) in ((3, 5), (64, 48), (5, 5), (64, 64))
          A = zeros(T, m, n)
          a, b = m ÷ 2, n ÷ 2
          A[1:a, 1:b] .= 1
          A[a+1:end, b+1:end] .= T(3)/4
          if m == n
            # A third component makes rank two a true truncation.
            A[end, :] .= 0
            A[:, end] .= 0
            A[end, end] = T(1)/2
          end
          push!(inputs, M .* A)
        end
        push!(inputs, (M * (T(3)/4)) .* T[1 1; 1 -1], fill(-M, 3, 3), fill(-M, 64, 64))
        for n in 2:5, rep in 1:3
          A = rand(rng, T, n, n) .* T(1)/8 .+ T(5)/8
          A = (A + A') / 2
          push!(inputs, M .* A)
        end
        # Mixed signs with a dominant singular value and finite output headroom.
        for (m, n) in ((4, 3), (3, 5), (5, 4), (4, 4)), rep in 1:3
          u = (T(3)/4 .+ rand(rng, T, m)/8) .* T[isodd(i) ? 1 : -1 for i in 1:m]
          v = (T(3)/4 .+ rand(rng, T, n)/8) .* T[isodd(i) ? -1 : 1 for i in 1:n]
          A = u * v' + randn(rng, T, m, n)/100
          @test opnorm(A) > one(T)
          @test !issymmetric(A)
          push!(inputs, M .* A)
        end
        # Positive and negative overflowing eigenvalues in separate blocks.
        for n in 4:6, rep in 1:3
          a = n ÷ 2
          u = T(4)/5 .+ rand(rng, T, a)/10
          v = T(4)/5 .+ rand(rng, T, n-a)/10
          A = randn(rng, T, n, n)/100
          A = (A + A')/2
          A[1:a, 1:a] .+= u * u'
          A[a+1:end, a+1:end] .-= v * v'
          θ = eigvals(Symmetric(A))
          @test first(θ) < -one(T) && last(θ) > one(T)
          push!(inputs, M .* A)
        end
        for X in inputs, partial in (false, true), f in functions(T, partial)
          symmetric = issymmetric(X)
          f isa Union{TracePSD, IndPSD} && !symmetric && continue
          e = exponent(M)
          A = scaled(X, -e)
          γ = M
          λ = f isa Union{NuclearNorm, TracePSD} ? f.λ : zero(T)
          c = T(ldexp(big(γ)*big(λ), -e))
          g = f isa NuclearNorm ? NuclearNorm(c) : f isa TracePSD ? TracePSD(c) :
            f isa IndBallRank ? IndBallRank(f.r) : IndPSD(; rank = f.rank)
          R = similar(X)
          prox!(R, g, A, one(T))
          Y = similar(X)
          value_and_prox!(Y, f, X, γ)
          close_scaled(Y, R, e; inputnorm = norm(A))
          symmetric && f isa Union{TracePSD, IndPSD} && (@test issymmetric(Y))
          # Full and partial calls preserve aliasing and strided output semantics.
          Z = copy(X)
          value_and_prox!(Z, f, Z, γ)
          @test isequal(Z, Y)
          n = size(X, 1)
          V = view(zeros(T, 2n, size(X, 2)), 1:2:2n, :)
          value_and_prox!(V, f, X, γ)
          @test isequal(V, Y)
          if partial && size(X, 1) >= 48
            @test usedfull(f)
          end
        end
      end

      @testset "zero singular columns in a scaled rank projection" begin
        X = zeros(T, 64, 64)
        f = IndBallRank(3; partial = true)
        ws = PB.SVDWorkspace(T, 64, 64)
        cache = PB._partial_svd_workspace(nothing, T, 64, 64, f.r)
        f.ws, f.pws = ws, cache
        Y = similar(X)
        # Exact zero weights isolate the retained-column rule from LAPACK rounding.
        PB._scaled_svd_prox!(Y, f, X, one(T), ws)
        @test ws.s[3] == zero(T)
        @test cache.rank == f.r
        @test all(iszero, view(ws.U, :, 3))
        @test cache.Y[:, 1:f.r] == ws.U[:, 1:f.r]
      end

      @testset "unchanged finite spectra and nonfinite proximal rules" begin
        rng = Xoshiro(235)
        B = randn(rng, T, 5, 5)
        for X in (B, B + B', B * B'), f in functions(T, false)
          Y, R = similar(X), similar(X)
          legacy!(R, f, X, one(T))
          prox!(Y, f, X, one(T))
          @test isequal(Y, R)
        end
        for v in (T(NaN), T(Inf), T(-Inf)), f in functions(T, false)
          for X in (T[v 0; 0 1], T[1 v; v 1])
            Y = similar(X)
            @test prox!(Y, f, X, one(T)) === Y
            identity = f isa IndBallRank && f.r >= minimum(size(X))
            @test isequal(Y, identity ? X : fill(T(NaN), size(X)))
          end
        end
      end

      @testset "warm starts and allocations" begin
        u = T[isodd(i) ? 1 : 1/2 for i in 1:64]
        X = M .* (u*u')
        A = zeros(T, 64, 64)
        A[1, 1] = T(4)
        for partial in (false, true), f in functions(T, partial)
          fresh() = f isa NuclearNorm ? NuclearNorm(f.λ; partial) : f isa TracePSD ? TracePSD(f.λ; partial) :
            f isa IndBallRank ? IndBallRank(f.r; partial) : IndPSD(; rank = f.rank, partial)
          Y, R = similar(X), similar(X)
          value_and_prox!(R, fresh(), X, M)
          for rep in 1:3
            value_and_prox!(Y, f, X, M)
            @test isequal(Y, R)
          end
          @test measured(Y, f, X, M) == 0
          @test measured_alias(Y, f, X, M) == 0
          value_and_prox!(Y, f, A, one(T))
          value_and_prox!(R, fresh(), A, one(T))
          @test norm(Y - R) <= 100PB._partial_rtol(T, 64) * norm(R)
        end
      end
    end
  end
end

@testset "mixed precision spectral overflow" begin
  T = Float32
  a = T(2)^127
  X = fill(prevfloat(2.0^127), 3, 3)
  @test Float32.(X) == fill(a, 3, 3)
  @test exponent(maximum(X)) != exponent(a)
  for partial in (false, true), γ in (one(T), a)
    for f in (NuclearNorm(T(1)/4; partial), TracePSD(T(1)/4; partial),
              IndPSD(; partial), IndPSD(; rank = 2, partial), IndBallRank(1; partial))
      Y = zeros(T, 3, 3)
      value = value_and_prox!(Y, f, X, γ)
      λ = f isa Union{NuclearNorm, TracePSD} ? f.λ : zero(T)
      reference = big(a) - big(γ)*big(λ)/3
      @test all(isfinite, Y)
      @test norm(big.(Y) .- reference) <= 100 * eps(T) * norm(big.(Float32.(X)))
      rounded(value, big(λ) * (3big(a) - big(γ)*big(λ)))
      R = similar(Y)
      preconverted = value_and_prox!(R, f, Float32.(X), γ)
      @test isequal(Y, R)
      @test isequal(value, preconverted)
      @test measured(Y, f, X, γ) == 0
    end
  end
  # Finite entries that overflow on conversion keep the legacy arithmetic.
  # Explicitly nonfinite inputs follow the decomposition fallback.
  X = fill(1e39, 2, 2)
  for partial in (false, true), f in (NuclearNorm(T(0.3); partial), TracePSD(T(0.3); partial),
                                     IndPSD(; partial), IndPSD(; rank = 2, partial), IndBallRank(1; partial))
    Y, R, B = zeros(T, 2, 2), zeros(T, 2, 2), zeros(T, 2, 2)
    value = value_and_prox!(Y, f, X, one(T))
    reference = value_and_prox!(R, f, Float32.(X), one(T))
    legacy!(B, f, X, one(T))
    @test reference === (PB._isindicator(f) ? zero(T) : T(NaN))
    @test all(isnan, R)
    @test isequal(Y, B)
  end
end

end
