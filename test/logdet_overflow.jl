module LogDetOverflowTest

using Test
using ProximalBase
using LinearAlgebra
using Random
using ..TestUtils

const PB = ProximalBase

# Keep the original arithmetic to detect changes on the ordinary path.
function legacy!(Y, f::LogDetLoss{T}, X, γ) where {T}
  r = sqrt(T(γ)) * sqrt(f.w)
  A = f.eig.A
  if r > one(T)
    A .= (((X ./ 2 .+ transpose(X) ./ 2) ./ r) ./ r) ./ 2 .- f.S ./ 2
    t, U = PB._eigen!(f.eig)
    ld = PB._rebuild_sym_out!(Y, U, t, s -> PB._logdet_scaled_root_half(s, r), f.C; logs = true)
  else
    A .= X ./ 4 .+ transpose(X) ./ 4 .- ((f.S ./ 2 .* r) .* r)
    t, U = PB._eigen!(f.eig)
    ld = PB._rebuild_sym_out!(Y, U, t, s -> PB._logbarrier_root_half(s, r), f.C; logs = true)
  end
  PB._value_at(PB._logdet_value(f, Y, ld), Y)
end

# For S = aI + bJ and X = MJ the two eigenspaces are known exactly.
function constant_reference(T, n, a, b, w, γ)
  setprecision(4096) do
    M = BigFloat(floatmax(T))
    step = BigFloat(γ) * BigFloat(w)
    root(t) = t >= 0 ? (t + sqrt(t*t + 4step))/2 : 2step/(sqrt(t*t + 4step) - t)
    z = root(n*M - step*(BigFloat(a) + n*BigFloat(b)))
    u = root(-step*BigFloat(a))
    diagonal = (z + (n-1)*u)/n
    offdiagonal = (z-u)/n
    ld = log(z) + (n-1)*log(u)
    value = BigFloat(w) * ((BigFloat(a) + n*BigFloat(b))*z + (n-1)*BigFloat(a)*u - ld)
    diagonal, offdiagonal, value
  end
end

@testset "LogDetLoss overflowing roots and spectra" begin
  for T in (Float32, Float64)
    M = floatmax(T)
    for n in (2, 3), (a, b) in ((one(T), zero(T)), (T(0.5), T(0.125)))
      S = a * Matrix{T}(I, n, n) + fill(b, n, n)
      for (w, γ) in ((T(0.25), T(4)), (one(T), one(T)), (one(T), T(4)))
        f = LogDetLoss(S, w)
        X = fill(M, n, n)
        Y = similar(X)
        v = value_and_prox!(Y, f, X, γ)
        d, o, reference = constant_reference(T, n, a, b, w, γ)
        @test all(isfinite, Y)
        @test isapprox(Y, fill(T(o), n, n); rtol = 32eps(T))
        @test isapprox(diag(Y), fill(T(d), n); rtol = 32eps(T))
        if isfinite(T(reference))
          @test isapprox(v, T(reference); rtol = 64eps(T))
        else
          @test v === T(Inf)
        end
        aliased = copy(X)
        @test isequal(value_and_prox!(aliased, f, aliased, γ), v)
        @test isequal(aliased, Y)
        @test allocs(value_and_prox!, Y, f, X, γ) == 0
      end
    end

    # The full shift overflows entrywise, but its positive part is representable.
    X = T[0 M; M 0]
    f = LogDetLoss(T[M -M; -M M], one(T))
    Y = similar(X)
    value_and_prox!(Y, f, X, one(T))
    @test all(isfinite, Y)
    @test isapprox(Y, fill(M/2, 2, 2); rtol = 32eps(T))

    # Keep small-root logs, and check the isolated component when resolvable.
    e = T === Float64 ? -500 : -60
    r = ldexp(one(T), e)
    X = zeros(T, 3, 3)
    X[1:2, 1:2] .= M
    f = LogDetLoss(zeros(T, 3, 3), r*r)
    Y = similar(X)
    v = value_and_prox!(Y, f, X, one(T))
    @test all(isfinite, Y)
    # The common reconstruction scale can discard this component, but not its log.
    @test Y[3, 3] >= 0
    @test isfinite(v)
    m, k = PB._logdet_root_parts(zero(T), exponent(M), r)
    @test ldexp(m, k) == r
    @test isapprox(log(m) + k*log(T(2)), log(r); rtol = 4eps(T))
    f = LogDetLoss(zeros(T, 3, 3), one(T))
    value_and_prox!(Y, f, X, one(T))
    @test isapprox(Y[3, 3], one(T); rtol = 8eps(T))
    for h in (-M, -4sqrt(M), -sqrt(M))
      m, k = PB._logdet_root_parts(h/M, exponent(M), one(T))
      reference = setprecision(4096) do
        half = BigFloat(h/M) * BigFloat(2)^exponent(M)
        inv(sqrt(half*half + 1) - half)
      end
      @test isapprox(ldexp(m, k), T(reference); rtol = 8eps(T))
    end

    # Symmetrization and output rounding also apply on the exceptional path.
    f = LogDetLoss(Matrix{T}(I, 2, 2), T(0.25))
    X = T[M M; M/2 M]
    Y, Z = zeros(Float64, 2, 2), zeros(Float64, 2, 2)
    v = value_and_prox!(Y, f, Float64.(X), T(4))
    u = value_and_prox!(Z, f, X/2 + X'/2, T(4))
    @test isequal(Y, Z)
    @test isequal(v, u)
    @test all(isfinite, Y)

    for Tx in (Float32, Float64), Ty in (Float32, Float64)
      magnitude = min(floatmax(T), floatmax(Tx), floatmax(Ty))
      X = fill(Tx(magnitude), 2, 2)
      f = LogDetLoss(Matrix{T}(I, 2, 2), T(0.25))
      Y, Z = zeros(Ty, 2, 2), zeros(T, 2, 2)
      value_and_prox!(Z, f, X, T(4))
      v = value_and_prox!(Y, f, X, T(4))
      @test Y == Ty.(Z)
      @test all(isfinite, Y)
      @test isapprox(v, Ty(BigFloat(magnitude)/2); rtol = 64max(eps(T), eps(Ty)))
      @test allocs(value_and_prox!, Y, f, X, T(4)) == 0
      @test prox!(Y, f, X, T(4)) === Y
      @test Y == Ty.(Z)
    end

    rng = Xoshiro(280)
    for n in (1, 2, 5), γ in (T(0.01), one(T), T(16)), w in (T(0.25), T(3))
      X = randn(rng, T, n, n)
      S = randn(rng, T, n, n)
      f = LogDetLoss(S*S', w)
      Y, Z = similar(X), similar(X)
      expected = legacy!(Z, f, X, γ)
      @test isequal(value_and_prox!(Y, f, X, γ), expected)
      @test isequal(Y, Z)
      @test allocs(value_and_prox!, Y, f, X, γ) == 0
    end
  end
end

end
