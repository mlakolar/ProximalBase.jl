module SpectralNonfiniteProxTest

using Test, ProximalBase, LinearAlgebra, SparseArrays, JLArrays
using GPUArraysCore: allowscalar

function spectral_functions(T)
  fs = ProximalBase.AbstractFunction[SingularValue(NormL1(one(T))), Spectral(NormL1(one(T)))]
  for partial in (false, true)
    append!(fs, [NuclearNorm(one(T); partial), NuclearNorm(zero(T); partial),
                 TracePSD(one(T); partial), TracePSD(zero(T); partial),
                 IndPSD(; partial), IndPSD(rank = 1; partial), IndPSD(rank = 2; partial),
                 IndBallRank(1; partial), IndBallRank(2; partial)])
  end
  fs
end

function rejected(f, X, name = nameof(typeof(f)); γ = 1.0, inplace = true)
  @nospecialize f X
  expected = ArgumentError("the proximal operator of $name needs finite entries")
  for op in (prox!, value_and_prox!)
    Y = fill(eltype(X)(17), size(X))
    before = copy(Y)
    @test_throws expected op(Y, f, X, γ)
    @test isequal(Y, before)
    if inplace
      A = copy(X)
      before = copy(A)
      @test_throws expected op(A, f, A, γ)
      @test isequal(A, before)
    end
  end
end

@testset "non-finite proximal inputs" begin
  for T in (Float32, Float64), bad in (Inf, -Inf, NaN), ij in ((1, 1), (1, 2), (2, 1))
    X = T[3 0; 0 1]
    X[ij...] = T(bad)
    S = SymmetricSparseIterate(T[3 0; 0 1])
    S[ij...] = T(bad)
    for A in (X, sparse(X), S), f in spectral_functions(T)
      rejected(f, A)
      if isconvex(f)
        rejected(Conjugate(f), A, nameof(typeof(f)))
      end
    end
  end
end

@testset "finite shortcuts and unsupported types" begin
  for T in (Float16, BigFloat), partial in (false, true)
    X = T[3 0; 0 1]
    for f in (NuclearNorm(0.0; partial), IndBallRank(2; partial)), op in (prox!, value_and_prox!)
      Y = fill(T(NaN), 2, 2)
      v = op(Y, f, X, 1.0)
      @test Y == X
      @test op === prox! ? v === Y : iszero(v)
      @test op === prox! ? op(X, f, X, 1.0) === X : iszero(op(X, f, X, 1.0))
    end
    Y = fill(T(NaN), 2, 2)
    before = copy(Y)
    @test_throws ArgumentError("TracePSD needs Float32 or Float64 entries, got $T") value_and_prox!(Y, TracePSD(0.0; partial), X, 1.0)
    @test isequal(Y, before)
  end
end

@testset "cold and warm workspaces" begin
  X = [3.0 0; 0 1]
  for f in spectral_functions(Float64)
    Y = fill(NaN, 2, 2)
    rejected(f, [NaN 0; 0 1])
    reference = prox(deepcopy(f), X, 1.0)
    @test all(isfinite, reference)
    for op in (prox!, value_and_prox!)
      fill!(Y, NaN)
      op(Y, f, X, 1.0)
      @test Y ≈ reference
      rejected(f, [Inf 0; 0 1])
      fill!(Y, NaN)
      op(Y, f, X, 1.0)
      @test Y ≈ reference
    end
  end
end

@testset "wrappers preserve finite behavior with stale buffers" begin
  makers = (() -> NuclearNorm(1.0), () -> SingularValue(NormL1(1.0)),
            () -> Spectral(NormL1(1.0)), () -> TracePSD(1.0), () -> IndPSD())
  wraps = (f -> Scaled(f, 2.0), f -> Tilt(f, fill(0.25, 2, 2)),
           f -> Translate(f, fill(0.25, 2, 2)), Conjugate, f -> Conjugate(Conjugate(f)))
  norm_outputs = ([1.0 0; 0 0],
                  [1.754191640756707 -0.21595085938862482; -0.2159508593886249 0.026584765647708106],
                  [2.004191640756707 0.03404914061137518; 0.034049140611375095 0.2765847656477081],
                  [1.0 0; 0 1], [2.0 0; 0 0])
  norm_values = (2.0, 2.1179950783112065, 1.780776406404415, 0.0, 2.0)
  psd_outputs = ([3.0 0; 0 1], [2.75 -0.25; -0.25 0.75], [3.0 0; 0 1], zeros(2, 2), [3.0 0; 0 1])
  psd_values = (0.0, 0.75, 0.0, 0.0, 0.0)
  for (i, make) in enumerate(makers), (j, wrap) in enumerate(wraps)
    f = make()
    g = wrap(f)
    expected = i == 5 ? psd_outputs[j] : norm_outputs[j]
    value = i == 5 ? psd_values[j] : norm_values[j]
    for warm in (false, true)
      warm && prox!(fill(NaN, 2, 2), g, [3.0 0; 0 1], 1.0)
      for bad in (Inf, -Inf, NaN)
        rejected(g, [bad 0; 0 1], nameof(typeof(f)))
      end
      Y = fill(NaN, 2, 2)
      @test value_and_prox!(Y, g, [3.0 0; 0 1], 1.0) ≈ value atol = 1e-12
      @test Y ≈ expected atol = 1e-12
      fill!(Y, NaN)
      @test prox!(Y, g, [3.0 0; 0 1], 1.0) === Y
      @test Y ≈ expected atol = 1e-12
    end
  end
end

@testset "logical symmetric entries" begin
  for uplo in (:U, :L), bad in (Inf, -Inf, NaN)
    A = [3.0 0; 0 1]
    A[uplo === :U ? CartesianIndex(2, 1) : CartesianIndex(1, 2)] = bad
    X = Symmetric(A, uplo)
    for f in spectral_functions(Float64)
      Y = fill(NaN, 2, 2)
      reference = prox(deepcopy(f), Matrix(X), 1.0)
      @test all(isfinite, reference)
      value_and_prox!(Y, f, X, 1.0)
      @test Y ≈ reference
      rejected(f, A)
    end
  end
end

@testset "partial paths on the issue matrix" begin
  for T in (Float32, Float64)
    u = T[isodd(i) ? 1 : 0.5 for i in 1:64]
    X = 3u * u' + T(0.01) * I
    for f in (NuclearNorm(T(0.25); partial = true), TracePSD(T(0.25); partial = true),
              IndPSD(partial = true), IndPSD(rank = 1, partial = true), IndBallRank(1; partial = true))
      for bad in (Inf, -Inf, NaN)
        A = copy(X)
        A[5, 5] = T(bad)
        rejected(f, A)
      end
    end
  end
end

@testset "device inputs without scalar indexing" begin
  allowscalar(false)
  for T in (Float32, Float64), f in spectral_functions(T)
    X = JLArray(T[3 0; 0 1])
    Y = fill(T(NaN), 2, 2)
    reference = prox(deepcopy(f), Array(X), 1.0)
    value_and_prox!(Y, f, X, 1.0)
    @test Y ≈ reference
    for bad in (Inf, -Inf, NaN)
      rejected(f, JLArray(T[bad 0; 0 1]); inplace = false)
    end
  end
end

end
