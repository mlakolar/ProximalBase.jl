module StiefelTest

using Test
using ProximalBase
using LinearAlgebra
using Random

using ..TestUtils

const PB = ProximalBase
const rng = MersenneTwister(168)

function orthonormal(::Type{T}, p, k) where {T}
  Matrix(qr(randn(rng, T, p, k)).Q)[:, 1:k]
end

function unchanged(op, f, point, E)
  for alias in (false, true)
    X = deepcopy(point)
    Y = alias ? X : deepcopy(point)
    before, input = deepcopy(Y), deepcopy(X)
    @test_throws E op(Y, f, X, 0.7)
    @test isequal(Y, before)
    @test isequal(X, input)
  end
end

@testset "IndStiefel interface and polar factor" begin
  f = IndStiefel()
  check_traits(f; proximable = true)
  @test PB._isindicator(f)
  @test_throws ArgumentError Conjugate(f)
  @test_throws ArgumentError MoreauEnvelope(f, 0.5)
  for T in (Float32, Float64)
    rtol = T === Float64 ? 1e-10 : 1e-4
    for (p, k) in ((9, 3), (4, 4), (20, 2))
      X = randn(rng, T, p, k)
      F = svd(X)
      P = prox(f, X)
      @test eltype(P) === T
      @test norm(P' * P - I) <= 100sqrt(T(k)) * eps(T)
      @test P ≈ F.U * F.Vt rtol = rtol
      @test f(P) === zero(T)
      for γ in (0.01, 1f0, 7 // 3)
        @test prox(f, X, γ) == P
      end
      v = value_and_prox!(similar(X), f, X, 0.3)
      @test v === zero(T)
    end
    X = orthonormal(T, 8, 3) * T(2)
    @test prox(f, X) ≈ X / T(2) rtol = rtol  # repeated singular values
    X = randn(rng, T, 7, 1)
    @test vec(prox(f, X)) ≈ prox(IndSphereL2(1), vec(X)) rtol = rtol
    check_prox(f, randn(rng, T, 7, 3), T(0.7);
      local_points = false, rtol, atol = eps(T), rng)
  end
  X = Matrix(Diagonal([-2.0, 3.0]))
  @test prox(f, X) ≈ Diagonal([-1.0, 1.0])
  @test det(prox(f, X)) ≈ -1.0
end

@testset "IndStiefel optimal value and rank deficiency" begin
  f = IndStiefel()
  U = [1.0 1 1; 1 -1 1; 1 1 -1; 1 -1 -1] / 2
  V = [0.0 -1 0; 1 0 0; 0 0 -1]
  for σ in ([4.0, 2.0, 0.5], [2.0, 2.0, 2.0], [3.0, 0.0, 0.0], zeros(3))
    X = U * Diagonal(σ) * V'
    P = prox(f, X)
    @test norm(P' * P - I) <= 100sqrt(3.0) * eps()
    @test sum(abs2, X - P) ≈ sum(abs2, σ .- 1) atol = 1e-12
    @test sum(P .* X) ≈ sum(σ) atol = 1e-12
    @test prox(f, X) == P
    @test f(P) == 0
    Y = copy(X)
    @test value_and_prox!(Y, f, Y, 0.8) === 0.0
    @test Y == P
  end
  for X in (Matrix(Diagonal([-2.0, 0.0])), zeros(5, 2))
    P = prox(f, X)
    @test norm(P' * P - I) <= 100sqrt(size(X, 2)) * eps()
    @test prox(f, X) == P
    @test sum(abs2, X - P) ≈ 2.0
  end
  @test prox(f, Matrix(Diagonal([-2.0, 0.0])))[1, 1] ≈ -1.0
  for T in (Float32, Float64)
    X = T[1 0 0; 0 0 0; 0 0 0; 0 0 0]
    P = prox(f, X)
    @test norm(P' * P - I) <= 100sqrt(T(3)) * eps(T)
    @test sum(abs2, X - P) ≈ T(2)
    @test prox(f, X) == P
  end

  X = randn(rng, 7, 3)
  P = prox(f, X)
  objective = sum(abs2, X - P)
  for _ in 1:100
    Q = orthonormal(Float64, 7, 3)
    @test sum(abs2, X - Q) >= objective - 1e-12
  end
  for _ in 1:10
    Q = prox(f, X + 0.1randn(rng, 7, 3))
    @test sum(abs2, X - Q) >= objective - 1e-12
  end
end

@testset "IndStiefel Procrustes" begin
  A, B = randn(rng, 8, 3), randn(rng, 8, 3)
  Q = prox(IndStiefel(), A' * B)
  objective = sum(abs2, A * Q - B)
  for _ in 1:100
    R = orthonormal(Float64, 3, 3)
    @test sum(abs2, A * R - B) >= objective - 1e-12
  end
end

@testset "IndStiefel membership" begin
  f = IndStiefel()
  for T in (Float32, Float64)
    k = 3
    t = PB._tol(T) * sqrt(T(k))
    X = Matrix{T}(I, 6, k)
    @test f(X) === zero(T)
    for scale in (T(0.25), T(2))
      D = copy(X)
      D[1, 1] = sqrt(one(T) + scale * t)
      O = copy(X)
      O[1, 2] = scale * t
      for Y in (D, O)
        expected = norm(Y' * Y - I) <= t ? zero(T) : T(Inf)
        @test expected === (scale < 1 ? zero(T) : T(Inf))
        @test f(Y) === expected
        @test f(X) === zero(T)        # every Gram entry is refreshed
        @test f(Y) === expected
      end
    end
    for v in (T(NaN), T(Inf), -T(Inf), floatmax(T))
      Y = copy(X)
      Y[1, 1] = v
      @test f(Y) === T(Inf)
      if !isfinite(v)
        for op in (prox!, value_and_prox!)
          Z = similar(Y)
          result = op(Z, f, Y, one(T))
          @test all(isnan, Z)
          op === value_and_prox! && @test result === zero(T)
        end
      end
    end
  end
  @test f([1 0; 0 1; 0 0]) === 0.0
  @test f(reshape([typemin(Int)], 1, 1)) === Inf
  @test f(ones(Float16, 1, 1)) === Float16(0)
  @test f(ones(BigFloat, 1, 1)) isa BigFloat
  @test iszero(f(ones(BigFloat, 1, 1)))
  for T in (Float32, Float64), p in (0, 4)
    X = zeros(T, p, 0)
    @test f(X) === zero(T)
    @test prox(f, X) == X
    @test value_and_prox!(X, f, X, 1) === zero(T)
  end
end

@testset "IndStiefel precision, scaling and views" begin
  f = IndStiefel()
  for scale in (1e-50, 1e100, nextfloat(0.0), floatmax(Float64))
    X = reshape([0.0, scale], 2, 1)
    Y = zeros(Float32, 2, 1)
    @test prox!(Y, f, X, 0.5) === Y
    # LAPACK's singular vectors have rounding error even for an axis direction.
    @test norm(Y - reshape(Float32[0, 1], 2, 1)) <= 100eps(Float32)
    @test value_and_prox!(Y, f, X, 0.5) === 0f0
  end
  for T in (Float32, Float64)
    X = reshape(T[0, nextfloat(zero(T))], 2, 1)
    @test prox(f, X) == reshape(T[0, 1], 2, 1)
    @test prox(f, zeros(T, 3, 2))' * prox(f, zeros(T, 3, 2)) ≈ I
  end
  X = [1 0; 0 -2; 0 0]
  Y = zeros(Float32, size(X))
  @test value_and_prox!(Y, f, X, 1.0) === 0f0
  @test Y ≈ Float32[1 0; 0 -1; 0 0]
  @test eltype(prox(f, X)) === Float64
  @test prox!(Y, f, BigFloat.(X), 1.0) ≈ Float32[1 0; 0 -1; 0 0]

  data = randn(rng, 12, 3)
  X = view(data, 1:2:12, :)
  expected = prox(f, Matrix(X))
  for rows in (2:2:12, [2, 4, 6, 8, 10, 12])
    storage = fill(17.0, 12, 3)
    Y = view(storage, rows, :)
    @test prox!(Y, f, X, 0.7) === Y
    @test Y ≈ expected
    @test all(==(17.0), storage[1:2:12, :])
    copyto!(Y, X)
    @test value_and_prox!(Y, f, Y, 0.7) === 0.0
    @test Y ≈ expected
    @test f(Y) === 0.0
  end
  Y = SparseIterate(6, 3)
  @test Matrix(prox!(Y, f, X, 0.7)) ≈ expected
  @test f(SparseIterate(expected)) === 0.0
end

@testset "IndStiefel checks before writes" begin
  f = IndStiefel()
  for (X, E) in ((zeros(2, 3), DimensionMismatch), (zeros(0, 1), DimensionMismatch),
                 (ones(3), ArgumentError), (ones(2, 2, 1), ArgumentError),
                 (ones(ComplexF64, 3, 2), ArgumentError("IndStiefel requires real entries")))
    @test_throws E f(X)
    for op in (prox!, value_and_prox!)
      unchanged(op, f, X, E)
      fs = SeparableSum(SqrNormL2(2.0), f)
      expected = eltype(X) <: Real ? E : ArgumentError("SeparableSum requires real entries")
      unchanged(op, fs, ([3.0], X), expected)
      for wrapper in (Tilt(f, zeros(size(X))), Translate(f, zeros(size(X))), Regularize(f, 0.5))
        expected = eltype(X) <: Real ? E : ArgumentError("$(nameof(typeof(wrapper))) requires real entries")
        unchanged(op, wrapper, X, expected)
      end
    end
  end
  @test_throws ArgumentError f(2.0)
  for op in (prox!, value_and_prox!)
    Y = fill(9.0, 4, 2)
    @test_throws ArgumentError op(Y, f, 2.0, 1)
    @test Y == fill(9.0, 4, 2)
    X = ones(4, 2)
    for Y in (fill(9.0, 2, 4), fill(9.0, 8))
      before = copy(Y)
      @test_throws DimensionMismatch op(Y, f, X, 1)
      @test Y == before
    end
    for T in (Float16, BigFloat, Int)
      Y = fill(T(9), 4, 2)
      before = copy(Y)
      @test_throws ArgumentError op(Y, f, X, 1)
      @test Y == before
    end
    Y = fill!(Matrix{Union{Float32, Float64}}(undef, 4, 2), 9.0)
    @test_throws ArgumentError op(Y, f, X, 1)
    @test all(==(9.0), Y)
  end
end

@testset "IndStiefel workspace and allocations" begin
  for T in (Float32, Float64)
    f = IndStiefel()
    X = randn(rng, T, 8, 3)
    Y = similar(X)
    f(X)
    input, gram = f.input.buf, f.gram.buf
    @test allocs(f, X) == 0
    @test f.input.buf === input && f.gram.buf === gram
    prox!(Y, f, X, T(0.7))
    ws = f.ws
    @test allocs(prox!, Y, f, X, T(0.7)) == 0
    @test allocs(value_and_prox!, Y, f, X, T(0.7)) == 0
    @test allocs(prox!, X, f, X, T(0.7)) == 0
    @test allocs(value_and_prox!, X, f, X, T(0.7)) == 0
    @test f.ws === ws
    Z = randn(rng, T, 7, 2)
    f(Z)
    @test f.input.buf !== input && f.gram.buf !== gram
    input, gram = f.input.buf, f.gram.buf
    f(Z)
    @test f.input.buf === input && f.gram.buf === gram
    prox!(similar(Z), f, Z, 1)
    @test f.ws !== ws
    ws = f.ws
    prox!(similar(Z), f, Z, 1)
    @test f.ws === ws
  end
end

end
