module SpectralFusedValuesTest

using Test, ProximalBase, LinearAlgebra, Random
using JLArrays
using GPUArraysCore: allowscalar
using ..TestUtils

# The value that value_and_prox! returns for SingularValue and Spectral of a function that counts
# nonzeros, and for IndBallRank, is the value at the matrix it stored (O2): the standalone value of
# that matrix, with its zero mask and rank threshold, after the reconstruction and the conversion
# to the destination's element type. The recheck itself is not observable, so each case compares
# the fused value with the standalone value at the stored matrix, as isequal.

const PB = ProximalBase

# the fused value, the standalone value at the stored matrix, and the stored matrix
function fused(f, X, ::Type{T}, γ = 1) where {T}
  Y = zeros(T, size(X)...)
  v = value_and_prox!(Y, f, X, γ)
  v, f(Y), Y
end

# whether the fused call stores what prox! stores, with the value at that matrix; a fresh function for
# each call, since a partial decomposition starts from the previous one
function consistent(make, X, ::Type{T}, γ = 1) where {T}
  v, fy, Y = fused(make(), X, T, γ)
  Y2 = zeros(T, size(X)...)
  prox!(Y2, make(), X, γ)
  isequal(v, fy) && isequal(Y, Y2)
end

rotation(rng, ::Type{T}, n) where {T} = Matrix(qr(randn(rng, T, n, n)).Q)

# the symmetric matrix Q Diagonal(s) Q'
function symmetric(rng, s::Vector{T}, rotate::Bool) where {T}
  n = length(s)
  Q = rotate ? rotation(rng, T, n) : Matrix{T}(I, n, n)
  X = Q * Diagonal(s) * Q'
  (X + X') / 2
end

@testset "large dynamic range: the stored matrix keeps 2, which the mask counts as zero (issue fixtures)" begin
  for wrapper in (SingularValue, Spectral), (T, big, small) in ((Float64, 2.0^60, 2.0^20), (Float32, 2f0^30, 2f0^10)),
      sign in (1, -1)
    X = Matrix(Diagonal(T[sign * big, 2, 0]))
    f = wrapper(NormL0(one(T)))
    v, fy, Y = fused(f, X, T)
    @test isequal(v, fy)
    @test v === one(T)
    @test isequal(Y, prox!(zeros(T, 3, 3), wrapper(NormL0(one(T))), X, 1))
    # the same with the smaller entry above the threshold
    X = Matrix(Diagonal(T[sign * small, 2, 0]))
    v, fy, Y = fused(wrapper(NormL0(one(T))), X, T)
    @test isequal(v, fy)
    @test v === T(2)
  end
  for wrapper in (SingularValue, Spectral), T in (Float32, Float64)
    X = Matrix(Diagonal(T[T(2)^(T == Float64 ? 60 : 30), 2, 0]))
    # IndBallL0 keeps one entry, so the stored matrix has rank one
    v, fy, _ = fused(wrapper(IndBallL0(1)), X, T)
    @test v === fy === zero(T)
    v, fy, _ = fused(wrapper(IndBallL0(2)), X, T)
    @test v === fy === zero(T)
  end
end

@testset "functions that count nonzeros inside wrappers" begin
  for wrapper in (SingularValue, Spectral), T in (Float32, Float64)
    X = Matrix(Diagonal(T[T(2)^(T == Float64 ? 60 : 30), 2, 0]))
    for f in (2 * NormL0(one(T)), 3 * NormL0(T(1 // 2)), Constrained(NormL0(T(1 // 1000)), IndBox(1, Inf)),
              Constrained(NormL0(T(1 // 1000)), IndBox(-Inf, Inf)))
      v, fy, _ = fused(wrapper(f), X, T)
      @test isequal(v, fy)
      @test v > 0 && isfinite(v)
    end
    # the value of the fixture of the issue's reproducer, at the box that holds every entry
    f = Constrained(NormL0(T(1 // 1000)), IndBox(1, Inf))
    v, fy, _ = fused(wrapper(f), X, T)
    @test v === fy === T(1 // 1000)
    # constant functions keep their value: no level, and a rank bound that every matrix meets
    for f in (NormL0(zero(T)), IndBallL0(3), IndBallL0(4))
      v, fy, _ = fused(wrapper(f), X, T)
      @test v === fy === zero(T)
    end
  end
end

@testset "boundary of the zero mask, diagonal and rotated" begin
  for T in (Float32, Float64), wrapper in (SingularValue, Spectral), rotate in (false, true)
    rng = MersenneTwister(564 + 10 * rotate + 100 * (T == Float64))
    tol = sqrt(eps(T))
    # σ₂ at the threshold: just below, at, just above, and within the reconstruction's rounding
    ratios = T[1 - 40eps(T), prevfloat(one(T)), one(T), nextfloat(one(T)), 1 + 40eps(T), 1 - 1 // 1000, 1 + 1 // 1000]
    bad = 0
    for _ in 1:12, ρ in ratios, g in (NormL0(T(1 // 10^30)), 2 * NormL0(T(1 // 10^30)))
      s = T[1, tol * ρ, tol * ρ / 4, 0, 0, 0]
      wrapper === Spectral && (s .*= T[1, -1, 1, 1, 1, 1])
      X = symmetric(rng, s, rotate)
      v, fy, _ = fused(wrapper(g), X, T)
      bad += !isequal(v, fy)
    end
    @test bad == 0
  end
end

@testset "boundary at a larger size and with negative entries of the spectrum" begin
  for T in (Float32, Float64), wrapper in (SingularValue, Spectral)
    rng = MersenneTwister(77)
    n = 40
    tol = sqrt(eps(T))
    bad = 0
    for ρ in T[1 - 1 // 1000, 1 - 10eps(T), 1, 1 + 10eps(T), 1 + 1 // 1000]
      s = zeros(T, n)
      s[1] = -1
      s[2] = tol * ρ
      s[3] = -tol * ρ / 2
      X = symmetric(rng, s, true)
      v, fy, _ = fused(wrapper(NormL0(T(1 // 10^30))), X, T)
      bad += !isequal(v, fy)
    end
    @test bad == 0
  end
end

@testset "zero entries of the proximal spectrum at a size where the band reaches the threshold" begin
  # In Float32 the band 4·n·ε·max|t| exceeds √ε·max|t| from n = 723; the zeros of t do not enter it,
  # and the value is still that of the stored matrix.
  rng = MersenneTwister(3)
  n = 800
  s = zeros(Float32, n)
  s[1:3] = Float32[3, 2, 1]
  X = symmetric(rng, s, true)
  for f in (SingularValue(NormL0(0.5f0)), Spectral(NormL0(0.5f0)))
    v, fy, Y = fused(f, X, Float32, 0.7)
    @test isequal(v, fy)
    @test v == 1.5f0
  end
end

@testset "mixed precision at normal range: a Float64 argument and a Float32 destination" begin
  tol32 = Float64(sqrt(eps(Float32)))
  for wrapper in (SingularValue, Spectral), g in (NormL0(1e-9), 2 * NormL0(1e-9)), δ in (0.999, 1.0, 1.001)
    X = Matrix(Diagonal([1, δ * tol32, 0]))
    v, fy, Y = fused(wrapper(g), X, Float32)
    @test v isa Float32
    @test isequal(v, fy)
    # rotated: the rounding of the argument to Float32 moves the spectrum
    Q = rotation(MersenneTwister(5), Float64, 3)
    Xr = Q * Matrix(Diagonal([1, δ * tol32, 0])) * Q'
    Xr = (Xr + Xr') / 2
    v, fy, Y = fused(wrapper(g), Xr, Float32)
    @test isequal(v, fy)
  end
end

@testset "IndBallRank: a subnormal rank-one argument stored in Float32" begin
  h = nextfloat(0f0)
  X = Float64(h) .* [1 5/8; 5/8 25/64]
  f = IndBallRank(1)
  v, fy, Y = fused(f, X, Float32)
  @test v isa Float32
  @test isequal(v, fy)
  @test v === Inf32                      # the rounded matrix has numerical rank two
  @test isequal(Y, prox!(zeros(Float32, 2, 2), IndBallRank(1), X, 1))
  # Float64 destination and normal range are exact
  v, fy, _ = fused(IndBallRank(1), X, Float64)
  @test v === fy === 0.0
  Xn = [1 5/8; 5/8 25/64]
  v, fy, _ = fused(IndBallRank(1), Xn, Float32)
  @test v === fy === 0f0
  # the other families keep their value at a subnormal spectrum
  for T in (Float32, Float64)
    u = nextfloat(zero(T))
    v = T[1, 5 / 8, 3 / 10]
    A = Float64(u) .* (v * v')
    for g in (IndBallRank(1), SingularValue(IndBallL0(1)), Spectral(IndBallL0(1)),
              SingularValue(IndBallL0(2)), Spectral(IndBallL0(2)))
      val, fy, _ = fused(g, A, T)
      @test isequal(val, fy)
    end
    # ranks two and three
    W = [1 0 0 0; 0 1 0 0; 0 0 1 0; 1 1 1 1] ./ 3
    for k in (2, 3)
      B = Float64(u) .* (W[:, 1:k] * W[:, 1:k]')
      B = (B + B') / 2
      for g in (IndBallRank(k), SingularValue(IndBallL0(k)), Spectral(IndBallL0(k)))
        val, fy, _ = fused(g, B, T)
        @test isequal(val, fy)
      end
    end
  end
end

@testset "IndBallRank with partial = true" begin
  rng = MersenneTwister(11)
  n = 24
  for T in (Float32, Float64)
    u = Float64(nextfloat(zero(T)))
    a = randn(rng, n)
    for scale in (u, 1.0)
      X = scale .* (a * a')
      make() = IndBallRank(1; partial = true)
      v, fy, Y = fused(make(), X, T)
      @test isequal(v, fy)
      @test isequal(v, fused(IndBallRank(1), X, T)[1])
      @test consistent(make, X, T)
    end
    # a rank-four argument with small noise, ordinary scale: the stored matrix has rank four
    U = Matrix(qr(randn(rng, n, 4)).Q)[:, 1:4]
    X = U * Diagonal([20.0, 16, 13, 10]) * U' + 1e-3 * Matrix(Symmetric(randn(rng, n, n)))
    for r in (3, 4)
      make() = IndBallRank(r; partial = true)
      v, fy, _ = fused(make(), X, T)
      @test isequal(v, fy)
    end
  end
end

@testset "overflowing spectra: the stored matrix is not finite" begin
  for T in (Float32, Float64)
    Z = fill(floatmax(T) / 2, 3, 3)
    for f in (SingularValue(NormL0(one(T))), Spectral(NormL0(one(T))), SingularValue(2 * NormL0(one(T))),
              Spectral(2 * NormL0(one(T))))
      v, fy, Y = fused(f, Z, T)
      @test isequal(v, fy)
      @test !isfinite(v)
    end
    # the proximal output of these is computed in scaled units: they keep their value
    for f in (SingularValue(IndBallL0(1)), Spectral(IndBallL0(1)), IndBallRank(1))
      v, fy, Y = fused(f, Z, T)
      @test v === fy === zero(T)
    end
  end
end

@testset "overflow-scaled output: a finite argument whose truncation has an entry above floatmax" begin
  # the spectrum overflows, the output is computed in scaled units and rescaled; an entry of the
  # rank-one part exceeds the range of the type and is stored as Inf, so that the value is that of
  # the stored matrix
  makers = (() -> IndBallRank(1), () -> IndBallRank(1; partial = true), () -> SingularValue(IndBallL0(1)),
            () -> Spectral(IndBallL0(1)))
  for T in (Float32, Float64), c in (1, 9 // 10)
    M = T(c) * floatmax(T)
    X = T[M M; M 0]
    for make in makers
      v, fy, Y = fused(make(), X, T)
      @test isequal(v, fy)
      all(isfinite, Y) || @test v === T(Inf)          # a matrix with an infinite entry
      @test consistent(make, X, T)
      # aliasing the destination and the argument
      Ya = copy(X)
      f = make()
      va = value_and_prox!(Ya, f, Ya, 1)
      @test isequal(va, v)
      @test isequal(va, f(Ya))
      @test isequal(Ya, Y)
    end
    # the proximal output is rank one in the other families too: an entry that fits keeps the value 0
    Xfit = T[M 0; 0 0]
    for make in makers
      v, fy, _ = fused(make(), Xfit, T)
      @test v === fy === zero(T)
    end
  end
end

@testset "IndBallRank: a finite argument that the destination cannot hold" begin
  X = [1e300 0.0; 0.0 1.0]
  # a rank bound that every matrix meets copies the argument; the copy overflows
  v, fy, Y = fused(IndBallRank(2), X, Float32)
  @test v === fy === Inf32
  @test isinf(Y[1])
  # a decomposition in Float32 of a matrix that does not fit
  v, fy, Y = fused(IndBallRank(1), X, Float32)
  @test isequal(v, fy)
  @test !isfinite(v)
  # a copy that fits, and a copy of a matrix with non-finite entries, keep their value
  v, fy, _ = fused(IndBallRank(2), [1e30 0.0; 0.0 1.0], Float32)
  @test v === fy === 0f0
  v, fy, _ = fused(IndBallRank(2), [1e300 0.0; 0.0 1.0], Float64)
  @test v === fy === 0.0
  v = value_and_prox!(zeros(Float32, 2, 2), IndBallRank(2), [Inf 0.0; 0.0 1.0], 1)
  @test v === Inf32
end

@testset "aliasing the destination and the argument" begin
  for T in (Float32, Float64)
    big = T(2)^(T == Float64 ? 60 : 30)
    h = nextfloat(zero(T))
    cases = (
      (() -> SingularValue(NormL0(one(T))), Matrix(Diagonal(T[big, 2, 0]))),
      (() -> Spectral(NormL0(one(T))), Matrix(Diagonal(T[-big, 2, 0]))),
      (() -> Spectral(NormL0(one(T))), Matrix(Diagonal(T[big, 2, 0]))),
      (() -> SingularValue(IndBallL0(1)), Float64(h) .* [1 5/8 3/10; 5/8 25/64 3/16; 3/10 3/16 9/100]),
      (() -> IndBallRank(1), Float64(h) .* [1 5/8 3/10; 5/8 25/64 3/16; 3/10 3/16 9/100]),
      (() -> IndBallRank(1), Float64(h) .* [1 5/8; 5/8 25/64]),
      (() -> IndBallRank(1; partial = true), Float64(h) .* (let a = collect(1.0:12.0) ./ 7; a * a' end)),
    )
    for (make, A) in cases
      X = T.(A)
      v0, f0, Y0 = fused(make(), X, T)
      Y = copy(X)
      f = make()
      v = value_and_prox!(Y, f, Y, 1)
      @test isequal(v, v0)
      @test isequal(v, f(Y))
      @test isequal(Y, Y0)
    end
  end
  # a destination of another element type cannot alias: the same call with the Float64 argument
  X = Matrix(Diagonal([2.0^60, 2, 0]))
  Y = copy(X)
  @test value_and_prox!(Y, SingularValue(NormL0(1.0)), Y, 1) === 1.0
end

@testset "prox! and value_and_prox! store the same matrix; ordinary cases" begin
  rng = MersenneTwister(2024)
  for T in (Float32, Float64)
    for (m, n) in ((8, 5), (5, 8), (6, 6))
      A = randn(rng, T, m, n)
      for make in (() -> SingularValue(NormL0(T(1 // 2))), () -> SingularValue(IndBallL0(2)),
                   () -> SingularValue(2 * NormL0(T(1 // 5))), () -> IndBallRank(2), () -> IndBallRank(3))
        @test consistent(make, A, T, T(7 // 10))
      end
    end
    for n in (4, 9)
      A = randn(rng, T, n, n)
      A = (A + A') / 2
      for make in (() -> Spectral(NormL0(T(1 // 2))), () -> Spectral(IndBallL0(2)), () -> Spectral(2 * NormL0(T(1 // 5))))
        @test consistent(make, A, T, T(7 // 10))
      end
    end
    # the fixtures of the issue
    h = nextfloat(0f0)
    @test consistent(() -> IndBallRank(1), Float64(h) .* [1 5/8; 5/8 25/64], Float32)
    @test consistent(() -> SingularValue(NormL0(one(T))), Matrix(Diagonal(T[2^30, 2, 0])), T)
    @test consistent(() -> Spectral(NormL0(one(T))), Matrix(Diagonal(T[2^30, 2, 0])), T)
  end
end

@testset "no write before a failed check" begin
  X = Matrix(Diagonal([3.0, 2.0, 1.0]))
  for f in (SingularValue(NormL0(1.0)), Spectral(NormL0(1.0)), IndBallRank(1), SingularValue(IndBallL0(1)))
    Y = fill(7.0, 3, 2)
    @test_throws DimensionMismatch value_and_prox!(Y, f, X, 1)
    @test all(==(7.0), Y)
    Y = fill(7 + 0im, 3, 3)
    @test_throws ArgumentError value_and_prox!(Y, f, X, 1)
    @test all(==(7 + 0im), Y)
    Y = fill(Float16(7), 3, 3)
    @test_throws ArgumentError value_and_prox!(Y, f, X, 1)
    @test all(==(Float16(7)), Y)
  end
end

@testset "non-finite arguments keep their values and outputs" begin
  for T in (Float32, Float64), x in (T(NaN), T(Inf), -T(Inf))
    X = T[x 0 0; 0 1 0; 0 0 1]
    Y = fill(T(7), 3, 3)
    @test isnan(value_and_prox!(Y, SingularValue(NormL0(one(T))), X, 1))
    @test all(isnan, Y)
    # the value of Spectral at the argument (E2): NaN at a NaN entry, Inf at an infinite one
    Y = fill(T(7), 3, 3)
    @test value_and_prox!(Y, Spectral(NormL0(one(T))), X, 1) === (isnan(x) ? T(NaN) : T(Inf))
    @test all(isnan, Y)
    Y = fill(T(7), 3, 3)
    @test isnan(value_and_prox!(Y, IndBallRank(1), X, 1))
    @test all(isnan, Y)
    # a constant counting function is zero at the all-NaN output
    Y = fill(T(7), 3, 3)
    @test value_and_prox!(Y, SingularValue(IndBallL0(3)), X, 1) === zero(T)
    @test all(isnan, Y)
  end
end

@testset "device arrays: the value at the stored matrix, with scalar indexing disallowed" begin
  allowscalar(false)
  for T in (Float32, Float64)
    big = T(2)^(T == Float64 ? 60 : 30)
    X = JLArray(Matrix(Diagonal(T[big, 2, 0])))
    for f in (SingularValue(NormL0(one(T))), Spectral(NormL0(one(T))))
      Y = similar(X)
      v = value_and_prox!(Y, f, X, 1)
      @test v === one(T)
      @test isequal(v, f(Y))
    end
    Z = JLArray(fill(floatmax(T) / 2, 3, 3))
    for f in (SingularValue(NormL0(one(T))), Spectral(NormL0(one(T))))
      Y = similar(Z)
      v = value_and_prox!(Y, f, Z, 1)
      @test !isfinite(v)
      @test isequal(v, f(Y))
    end
  end
  h = nextfloat(0f0)
  X = JLArray(Float64(h) .* [1 5/8; 5/8 25/64])
  Y = JLArray(zeros(Float32, 2, 2))
  v = value_and_prox!(Y, IndBallRank(1), X, 1)
  @test v === Inf32
  @test isequal(v, IndBallRank(1)(Y))
  X = JLArray([1e300 0.0; 0.0 1.0])
  Y = JLArray(zeros(Float32, 2, 2))
  @test value_and_prox!(Y, IndBallRank(2), X, 1) === Inf32
  # an overflow-scaled output with an entry above floatmax
  for T in (Float32, Float64), make in (() -> IndBallRank(1), () -> IndBallRank(1; partial = true),
                                        () -> SingularValue(IndBallL0(1)), () -> Spectral(IndBallL0(1)))
    M = floatmax(T)
    X = JLArray(T[M M; M 0])
    Y = similar(X)
    f = make()
    v = value_and_prox!(Y, f, X, 1)
    @test isequal(v, f(Y))
    @test !isfinite(v)
    Ya = similar(X); copyto!(Ya, X)
    va = value_and_prox!(Ya, make(), Ya, 1)
    @test isequal(va, v)
  end
end

end
