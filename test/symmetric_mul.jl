module SymmetricMulTest

# Products and reductions that have a SymmetricSparseIterate as an operand: mul! with the owner as
# the left operand, X * v, the products through matrixview(X), dot, norm, norm_diff and A * X * B.
# The owner is never the destination of a product: it is written with copyto!(X, M) from a dense
# matrix. The references are dense computations on the matrix that the owner represents, in
# BigFloat for the finite values, and explicit sums over the coefficients that are not zero for
# the non-finite ones.

using Test, ProximalBase, LinearAlgebra, SparseArrays, Random

# a symmetric matrix of the type T with a share `density` of nonzero lower-triangle entries
function randsym(rng, ::Type{T}, p, density = 0.4) where {T}
  M = zeros(T, p, p)
  for c in 1:p, r in c:p
    rand(rng) < density && (M[r, c] = M[c, r] = T(round(10randn(rng); digits = 2)))
  end
  M
end

# the owner of M, with an explicit zero active at each pair (r, c), r ≥ c, of `zs`
function owner(M, zs = ())
  x = SymmetricSparseIterate(M)
  for (r, c) in zs
    activate!(x, tril_index(size(M, 1), r, c))
  end
  x
end

# The sum of the absolute values of the terms of a product. Any order of summation of n terms is
# within n eps of the exact value times it, in the type of the destination.
nterms(M, B) = abs.(big.(M)) * abs.(big.(B))
within(y, ref, bound, ::Type{T}, n) where {T} =
  size(y) == size(ref) && all(abs.(big.(y) .- ref) .<= (n + 2) * big(eps(T)) .* bound)

# the products by the coefficients that are not zero: a zero coefficient contributes nothing
function termsum(M, B::AbstractVector, α = true)
  init = zero(promote_type(eltype(M), eltype(B)))
  [sum((B[j] * (α * M[i, j]) for j in axes(M, 2) if !iszero(M[i, j])); init) for i in axes(M, 1)]
end
termsum(M, B::AbstractMatrix, α = true) = hcat((termsum(M, B[:, j], α) for j in axes(B, 2))...)
dotterms(M, A) = sum((A[i, j] * M[i, j] for j in axes(M, 2), i in axes(M, 1) if !iszero(M[i, j]));
                     init = zero(promote_type(eltype(M), eltype(A))))
bilinear(M, a, b) = sum((a[i] * M[i, j] * b[j] for j in axes(M, 2), i in axes(M, 1) if !iszero(M[i, j]));
                        init = zero(promote_type(eltype(M), eltype(a), eltype(b))))

@testset "mul! with an owner as the left operand" begin
  @testset "finite values against a dense reference, $T" for T in (Float64, Float32)
    rng = MersenneTwister(13)
    for p in (0, 1, 2, 3, 5, 12, 40)
      M = randsym(rng, T, p)
      x = SymmetricSparseIterate(M)
      for U in (T, Float64, Float32)
        v = randn(rng, U, p)
        B = randn(rng, U, p, 3)
        # the three-argument forms overwrite the destination
        y = fill(T(NaN), p)
        @test mul!(y, x, v) === y
        @test within(y, big.(M) * big.(v), nterms(M, v), T, p)
        Y = fill(T(NaN), p, 3)
        @test mul!(Y, x, B) === Y
        @test within(Y, big.(M) * big.(B), nterms(M, B), T, p)
        # the allocating forms, and the transpose, which is the owner
        V = promote_type(T, U)
        z = x * v
        @test z isa Vector{V}
        @test within(z, big.(M) * big.(v), nterms(M, v), V, p)
        @test x' * v == z && transpose(x) * v == z
        Z = x * B
        @test Z isa Matrix{V}
        @test within(Z, big.(M) * big.(B), nterms(M, B), V, p)
        # the five-argument forms, with α and β of both kinds
        for (α, β) in ((1, 0), (2, 1), (-1.5, 0.5), (0, 1), (0, 0), (0, 2), (1, 1), (T(0.25), T(-2)))
          y0, Y0 = randn(rng, T, p), randn(rng, T, p, 3)
          y = copy(y0)
          @test mul!(y, x, v, α, β) === y
          @test within(y, big(α) * big.(M) * big.(v) + big(β) * big.(y0),
                       big(abs(α)) * nterms(M, v) + big(abs(β)) * abs.(big.(y0)), T, p)
          Y = copy(Y0)
          @test mul!(Y, x, B, α, β) === Y
          @test within(Y, big(α) * big.(M) * big.(B) + big(β) * big.(Y0),
                       big(abs(α)) * nterms(M, B) + big(abs(β)) * abs.(big.(Y0)), T, p)
        end
      end
    end
  end

  @testset "structured and strided right operands" begin
    rng = MersenneTwister(17)
    p = 6
    M = randsym(rng, Float64, p)
    x = SymmetricSparseIterate(M)
    N = randn(rng, p, p)
    for B in (Diagonal(randn(rng, p)), UpperTriangular(N), LowerTriangular(N),
              Bidiagonal(randn(rng, p), randn(rng, p - 1), :U), SymTridiagonal(randn(rng, p), randn(rng, p - 1)),
              Symmetric(N), transpose(N), view(N, :, 1:3), sparse(N))
      D = Matrix(B)
      for (α, β) in ((1, 0), (2.0, 0.5))
        C0 = randn(rng, p, size(D, 2))
        C = copy(C0)
        @test mul!(C, x, B, α, β) === C
        @test within(C, big(α) * big.(M) * big.(D) + big(β) * big.(C0),
                     big(α) * nterms(M, D) + big(β) * abs.(big.(C0)), Float64, p)
      end
      C = zeros(p, size(D, 2))
      mul!(C, x, B)
      @test within(C, big.(M) * big.(D), nterms(M, D), Float64, p)
    end
    # a strided destination and a strided vector
    big_y, big_v = zeros(2p), randn(rng, 2p)
    y, v = view(big_y, 1:2:2p), view(big_v, 2:2:2p)
    mul!(y, x, v)
    @test within(y, big.(M) * big.(v), nterms(M, v), Float64, p)
    # a Float32 destination, a Float64 owner and a Float64 vector
    y32 = zeros(Float32, p)
    mul!(y32, x, big_v[1:p])
    @test within(y32, big.(M) * big.(big_v[1:p]), nterms(M, big_v[1:p]), Float32, p)
  end

  @testset "α and β: the terms of the nonzero coefficients" begin
    # β = 0 does not read the destination, α = 0 does not read the right operand
    x = owner([2.0 3 0; 3 0 0; 0 0 5], [(3, 2)])
    nan = fill(NaN, 3)
    @test mul!(copy(nan), x, ones(3), 1.0, 0.0) == [5.0, 3.0, 5.0]
    @test mul!(fill(2.0, 3), x, nan, 0.0, 0.5) == fill(1.0, 3)
    @test mul!(copy(nan), x, nan, 0.0, 0.0) == zeros(3)
    @test isequal(mul!(fill(2.0, 3), x, nan, 0.0, NaN), fill(NaN, 3))
    # a non-finite α multiplies the nonzero coefficients only; the explicit zero at (3, 2) gives no NaN
    @test isequal(mul!(zeros(3), x, [1.0, 2.0, 0.0], Inf, 0.0), [Inf, Inf, NaN])   # row 3: 5 * Inf * 0
    @test isequal(mul!(zeros(3), x, ones(3), Inf, 0.0), fill(Inf, 3))
    @test isequal(mul!(zeros(3), x, ones(3), NaN, 0.0), fill(NaN, 3))
    @test isequal(mul!(ones(3), x, zeros(3), 2.0, Inf), fill(Inf, 3))
    @test isequal(mul!(zeros(3), x, zeros(3), 2.0, Inf), fill(NaN, 3))             # Inf * 0 in β C
    for (α, β) in ((2.0, 3.0), (-1.0, 0.0), (0.5, 1.0))
      y0, v = [1.0, -2.0, 4.0], [3.0, -1.0, 2.0]
      @test mul!(copy(y0), x, v, α, β) == α .* (Matrix(x) * v) .+ β .* y0
    end
  end

  @testset "output aliasing and shape errors" begin
    rng = MersenneTwister(19)
    x = SymmetricSparseIterate(randsym(rng, Float64, 4))
    B = randn(rng, 4, 4)
    for args in ((), (2.0, 1.0))
      B0 = copy(B)
      @test_throws ArgumentError mul!(B, x, B, args...)
      @test_throws ArgumentError mul!(view(B, :, :), x, B, args...)
      @test B == B0
      v = randn(rng, 4)
      v0 = copy(v)
      @test_throws ArgumentError mul!(v, x, v, args...)
      @test v == v0
    end
    # the storage of the owner is not a destination
    X1 = SymmetricSparseIterate(Float64, 1)
    X1[1, 1] = 2.0
    for args in ((), (1.0, 0.0))
      @test_throws ArgumentError mul!(coordinates(X1), X1, [3.0], args...)
      @test_throws ArgumentError mul!(reshape(coordinates(X1), 1, 1), X1, ones(1, 1), args...)
    end
    @test X1[1, 1] == 2.0
    # the coordinates of another owner are a destination
    Z = SymmetricSparseIterate(Float64, 1)
    mul!(coordinates(Z), X1, [3.0])
    @test Z[1, 1] == 6.0
    # a wrong shape leaves the destination unchanged
    for q in (3, 5), args in ((), (1.0, 0.0))
      C = fill(7.0, 4)
      @test_throws DimensionMismatch mul!(C, x, randn(rng, q), args...)
      @test C == fill(7.0, 4)
      @test_throws DimensionMismatch mul!(fill(7.0, q), x, randn(rng, 4), args...)
      C = fill(7.0, 4, 2)
      @test_throws DimensionMismatch mul!(C, x, randn(rng, q, 2), args...)
      @test_throws DimensionMismatch mul!(C, x, randn(rng, 4, 3), args...)
      @test C == fill(7.0, 4, 2)
      C = fill(7.0, q, 2)
      @test_throws DimensionMismatch mul!(C, x, randn(rng, 4, 2), args...)
      @test C == fill(7.0, q, 2)
    end
    @test_throws DimensionMismatch x * randn(rng, 3)
    @test_throws DimensionMismatch x * randn(rng, 3, 2)
    @test_throws ArgumentError mul!(zeros(ComplexF64, 4), x, ones(4))
    @test_throws ArgumentError mul!(zeros(4), x, ones(ComplexF64, 4))
    @test_throws ArgumentError mul!(zeros(4), x, ones(4), 1.0 + 1.0im, 0.0)
    @test_throws ArgumentError mul!(zeros(4, 2), x, ones(ComplexF64, 4, 2), 1.0, 0.0)
  end

  @testset "empty, one-dimensional and all-zero owners" begin
    x0 = SymmetricSparseIterate(Float64, 0)
    @test mul!(zeros(0), x0, zeros(0)) == zeros(0)
    @test mul!(zeros(0, 2), x0, zeros(0, 2), 2.0, 1.0) == zeros(0, 2)
    @test x0 * zeros(0) == zeros(0) && size(x0 * zeros(0, 3)) == (0, 3)
    @test dot(zeros(0), x0, zeros(0)) === 0.0 && dot(zeros(0, 0), x0) === 0.0
    @test norm(x0) === 0.0 && norm(x0, 1) === 0.0 && norm(x0, Inf) === 0.0
    @test norm_diff(x0, zeros(0, 0)) === 0.0
    @test Symmetric(zeros(0, 0)) * x0 * Symmetric(zeros(0, 0)) == zeros(0, 0)
    x1 = owner(fill(3.0, 1, 1))
    @test x1 * [2.0] == [6.0] && x1 * ones(1, 2) == fill(3.0, 1, 2)
    @test dot([2.0], x1, [5.0]) == 30.0 && dot(fill(2.0, 1, 1), x1) == 6.0
    @test norm(x1) == 3.0 && norm(x1, 1) == 3.0 && norm(x1, Inf) == 3.0
    @test Symmetric(fill(2.0, 1, 1)) * x1 * Symmetric(fill(5.0, 1, 1)) == fill(30.0, 1, 1)
    xz = owner(zeros(2, 2), [(2, 1)])
    @test nactive(xz) == 1
    @test xz * [Inf, NaN] == zeros(2) && dot([Inf, 1], xz, [1, NaN]) === 0.0
  end
end

# A coefficient of the owner that is zero, whether it is an explicit zero or absent, contributes
# nothing, even against a NaN or an infinity: the result with an explicit zero is the result after
# dropzeros!. A NaN coefficient is not zero, and a negative zero is.
@testset "strong zeros" begin
  @testset "the example, $T, $bad" for T in (Float64, Float32), bad in (Inf, -Inf, NaN)
    x = SymmetricSparseIterate(T, 3)
    x[1, 1] = T(2)
    activate!(x, tril_index(3, 2, 1))                       # an explicit zero at (2, 1)
    @test nactive(x) == 2 && isactive(x, tril_index(3, 2, 1))
    y = copy(x)
    dropzeros!(y)
    @test nactive(y) == 1
    v = T[bad, 1, 0]
    expected = T[T(2) * bad, 0, 0]                          # the terms of the coefficient 2 only
    for z in (x, y)
      @test isequal(mul!(zeros(T, 3), z, v), expected)
      @test isequal(mul!(zeros(T, 3), matrixview(z), v), expected)
      @test isequal(z * v, expected)
      @test isequal(matrixview(z) * v, expected)
      @test isequal(mul!(zeros(T, 3), z, v, T(1), T(0)), expected)
      @test isequal(mul!(zeros(T, 3), matrixview(z), v, T(1), T(0)), expected)
      @test isequal(mul!(ones(T, 3), z, v, T(2), T(1)), T[4bad + 1, 1, 1])
      @test isequal(mul!(ones(T, 3), matrixview(z), v, T(2), T(1)), T[4bad + 1, 1, 1])
      V = T[bad 1; 1 bad; 0 0]
      @test isequal(mul!(zeros(T, 3, 2), z, V), T[2bad 2; 0 0; 0 0])
      @test isequal(mul!(zeros(T, 3, 2), z, V, T(1), T(0)), T[2bad 2; 0 0; 0 0])
      @test isequal(z * V, T[2bad 2; 0 0; 0 0])
      # bad meets the explicit zero at (2, 1) in a and b and in M
      a, b = T[bad, 1, 1], ones(T, 3)
      @test isequal(dot(a, z, b), 2bad)
      @test isequal(dot(b, z, a), 2bad)
      @test isequal(dot(a, z, a), 2bad * bad)
      M = ones(T, 3, 3)
      M[2, 1] = bad
      @test isequal(dot(M, z), T(2)) && isequal(dot(z, M), T(2))
      @test isequal(dot(Matrix(M'), z), T(2)) && isequal(dot(z, Matrix(M')), T(2))
      M[1, 2] = bad
      @test isequal(dot(M, z), T(2))
      M[1, 1] = bad                                         # the coefficient 2 meets bad
      @test isequal(dot(M, z), 2bad)
    end
  end

  @testset "every pattern of three states for p = 3, $T, $bad" for (T, bad) in
      ((Float64, Inf), (Float64, NaN), (Float32, -Inf), (Float32, NaN))
    p, q = 3, 6
    coefs = T[2, 3, 5, 7, 11, 13]
    vs = (T[bad, 1, 2], T[1, bad, -1], T[3, -2, bad], T[bad, bad, 1])
    B = T[bad 1; 2 bad; 1 3]
    pairs = [(r, c) for c in 1:p for r in c:p]
    D = fill(T(bad), p, p)
    D[1, 1] = 1
    for code in 0:(3^q - 1)
      states = digits(code, base = 3, pad = q)     # 0 absent, 1 explicit zero, 2 nonzero
      M = zeros(T, p, p)
      for (k, (r, c)) in enumerate(pairs)
        states[k] == 2 && (M[r, c] = M[c, r] = coefs[k])
      end
      x = owner(M, [pairs[k] for k in 1:q if states[k] == 1])
      y = copy(x)
      dropzeros!(y)
      for v in vs
        expected = termsum(M, v)
        @test isequal(mul!(zeros(T, p), x, v), expected)
        @test isequal(mul!(zeros(T, p), matrixview(x), v), expected)
        @test isequal(x * v, expected) && isequal(matrixview(x) * v, expected)
        @test isequal(mul!(zeros(T, p), y, v), expected)
        @test isequal(mul!(zeros(T, p), x, v, T(2), T(0)), termsum(M, v, T(2)))
        @test isequal(mul!(zeros(T, p), matrixview(x), v, T(2), T(0)), termsum(M, v, T(2)))
        @test isequal(dot(v, x, v), bilinear(M, v, v)) && isequal(dot(v, y, v), bilinear(M, v, v))
        @test isequal(dot(reverse(v), x, v), bilinear(M, reverse(v), v))
        A = M .+ v
        @test isequal(dot(A, x), dotterms(M, A)) && isequal(dot(x, A), dotterms(M, A))
      end
      @test isequal(dot(D, x), dotterms(M, D)) && isequal(dot(D, y), dotterms(M, D))
      @test isequal(mul!(zeros(T, p, 2), x, B), termsum(M, B))
      @test isequal(mul!(zeros(T, p, 2), x, B, T(2), T(0)), termsum(M, B, T(2)))
      @test isequal(x * B, termsum(M, B))
    end
  end

  @testset "a NaN coefficient is not zero; a negative zero is" begin
    x = owner([2.0 0 0; 0 0 0; 0 0 0], [(2, 1), (3, 3)])
    coordinates(x)[tril_index(3, 2, 1)] = -0.0
    @test isactive(x, tril_index(3, 2, 1)) && iszero(x[2, 1]) && signbit(x[2, 1])
    v = [Inf, 1.0, Inf]
    @test isequal(x * v, [Inf, 0.0, 0.0])
    @test isequal(dot(v, x, v), Inf)
    @test dot(fill(Inf, 3, 3), x) == Inf
    coordinates(x)[tril_index(3, 2, 1)] = NaN
    @test isequal(x * [Inf, 1.0, 0.0], [NaN, NaN, 0.0])
    @test isequal(mul!(zeros(3, 1), x, reshape([1.0, 1.0, 0.0], 3, 1)), reshape([NaN, NaN, 0.0], 3, 1))
    @test isnan(dot([1.0, 1.0, 0.0], x, [1.0, 1.0, 0.0]))
    @test isnan(dot(ones(3, 3), x))
  end

  @testset "finite values with and without the explicit zeros" begin
    rng = MersenneTwister(23)
    for p in (1, 2, 5, 12, 40), T in (Float32, Float64)
      M = randsym(rng, T, p, 0.2)
      zs = [(r, c) for c in 1:p for r in c:p if iszero(M[r, c]) && rand(rng) < 0.5]
      x = owner(M, zs)
      y = copy(x)
      dropzeros!(y)
      @test nactive(x) == nactive(y) + length(zs) && Matrix(x) == Matrix(y)
      v, B = randn(rng, T, p), randn(rng, T, p, 2)
      @test x * v ≈ y * v && x * B ≈ y * B
      @test dot(v, x, v) ≈ dot(v, y, v)
      W = B[:, 1:1] * B[:, 2:2]'
      @test dot(W, x) ≈ dot(W, y)
    end
  end
end

@testset "dot of an owner and dense operands" begin
  @testset "off-diagonal entries count at both positions, $T" for T in (Float64, Float32)
    x = SymmetricSparseIterate(T, 3)
    x[2, 1] = T(3)
    M = T[1 2 3; 4 5 6; 7 8 9]
    @test dot(M, x) == T(3) * (M[2, 1] + M[1, 2]) && dot(x, M) == dot(M, x)
    @test dot(Matrix(M'), x) == T(3) * (M[1, 2] + M[2, 1])
    @test dot(Matrix(Symmetric(M)), x) == 6 * Symmetric(M)[2, 1]
    x[3, 3] = T(2)
    @test dot(M, x) == T(3) * (M[2, 1] + M[1, 2]) + 2M[3, 3]
    # the bilinear form against the full dense products, in both orders
    a, b = T[1, 2, 3], T[-1, 0, 2]
    @test dot(a, x, b) == dot(a, Matrix(x) * b)
    @test dot(b, x, a) == dot(a, x, b)
    # any vector: a column of a Symmetric matrix
    S = Symmetric(M)
    @test dot(view(S, :, 1), x, view(S, :, 2)) == dot(S[:, 1], Matrix(x) * S[:, 2])
    # an owner and an owner, through the matrix of one of them (there is no dot of two owners)
    @test dot(Matrix(x), x) == sum(abs2, Matrix(x))
  end

  @testset "the type of the result" begin
    x = SymmetricSparseIterate(Float32, 2)
    x[2, 1] = 3f0
    @test dot(ones(Float32, 2, 2), x) === 6f0
    @test dot(x, ones(Float32, 2, 2)) === 6f0
    @test dot(ones(Float32, 2), x, ones(Float32, 2)) === 6f0
    @test norm(x, 1) === 6f0
    @test dot(ones(2, 2), x) === 6.0                         # a wider matrix widens the result
    @test dot(Float32[1, 2], x, [1.0, 1.0]) === 9.0
    y = SymmetricSparseIterate(Float64, 2)
    y[2, 1] = 3.0
    @test dot(ones(Float32, 2, 2), y) === 6.0
  end

  @testset "random matrices against BigFloat, $T" for T in (Float64, Float32)
    rng = MersenneTwister(29)
    for p in (0, 1, 2, 3, 8, 30)
      M = randsym(rng, T, p)
      x = SymmetricSparseIterate(M)
      A, a, b = randn(rng, T, p, p), randn(rng, T, p), randn(rng, T, p)
      n = 4p * p + 8
      ref, bound = [sum(big.(A) .* big.(M))], [sum(abs.(big.(A)) .* abs.(big.(M)))]
      @test within([dot(A, x)], ref, bound, T, n) && within([dot(x, A)], ref, bound, T, n)
      @test within([dot(a, x, b)], [sum(big.(a) .* big.(M) .* big.(b)')],
                   [sum(abs.(big.(a)) .* abs.(big.(M)) .* abs.(big.(b))')], T, n)
      @test dot(a, x, b) isa T && dot(A, x) isa T
    end
  end

  @testset "shape and element type errors" begin
    x = owner([1.0 2; 2 3])
    @test_throws DimensionMismatch dot(ones(3, 3), x)
    @test_throws DimensionMismatch dot(x, ones(2, 3))
    @test_throws DimensionMismatch dot(ones(3), x, ones(2))
    @test_throws DimensionMismatch dot(ones(2), x, ones(3))
    @test_throws ArgumentError dot(ones(ComplexF64, 2), x, ones(2))
    @test_throws ArgumentError dot(ones(ComplexF64, 2, 2), x)
  end
end

@testset "norm and norm_diff of an owner count both triangles" begin
  @testset "one off-diagonal entry, $T" for T in (Float64, Float32)
    x = SymmetricSparseIterate(T, 2)
    x[2, 1] = T(3)
    @test norm(x) ≈ 3sqrt(T(2)) && norm(x, 2) ≈ 3sqrt(T(2))
    @test norm(x, 1) === T(6) && norm(x, Inf) === T(3)
    @test norm(coordinates(x)) === T(3)                      # the ordinary norm of the variables
    @test norm(x) isa T && norm(x, 1) isa T
    @test norm_diff(x, zeros(T, 2, 2)) ≈ 3sqrt(T(2)) && norm_diff(zeros(T, 2, 2), x, Inf) === T(3)
    @test norm_diff(x, x) === zero(T) && norm_diff(x, copy(x), Inf) === zero(T)
    @test norm_diff(x, zeros(T, 2, 2)) isa T
  end

  @testset "against the dense matrix, p = $p, $T" for T in (Float64, Float32), p in 0:3
    rng = MersenneTwister(31 + p)
    M, N = randsym(rng, T, p, 0.8), randsym(rng, T, p, 0.8)
    x, y = SymmetricSparseIterate(M), SymmetricSparseIterate(N)
    for q in (1, 2, Inf)
      @test norm(x, q) ≈ T(norm(big.(M), q)) rtol = 8eps(T)
      @test norm(x, q) isa T
    end
    @test norm(x) == norm(x, 2)
    for q in (0, 3, -Inf)
      @test isequal(norm(x, q), norm(M, q))
    end
    d = big.(M) - big.(N)
    @test norm_diff(x, y) ≈ T(norm(d)) rtol = 8eps(T)
    @test norm_diff(x, y, Inf) == T(norm(d, Inf))
    @test norm_diff(x, N) ≈ T(norm(d)) rtol = 8eps(T)
    @test norm_diff(M, y) ≈ T(norm(d)) rtol = 8eps(T)
    @test norm_diff(x, N, Inf) == T(norm(d, Inf)) && norm_diff(M, y, Inf) == T(norm(d, Inf))
    @test_throws ArgumentError norm_diff(x, y, 1)
    @test_throws DimensionMismatch norm_diff(x, zeros(T, p + 1, p + 1))
    @test_throws DimensionMismatch norm_diff(x, SymmetricSparseIterate(T, p + 1))
  end

  @testset "overflow and underflow are scaled, $T" for T in (Float64, Float32)
    for (value, kind) in ((floatmax(T) / 4, "overflow"), (2floatmin(T), "underflow"))
      M = T[value value 0; value 0 value; 0 value value]
      x = SymmetricSparseIterate(M)
      @test norm(x) ≈ T(norm(big.(M))) rtol = 16eps(T)
      @test isfinite(norm(x)) && !iszero(norm(x))
      @test norm(x, Inf) == value
      N = T[-value value 0; value 0 -value; 0 -value value]
      d = big.(M) - big.(N)
      @test norm_diff(x, SymmetricSparseIterate(N)) ≈ T(norm(d)) rtol = 16eps(T)
      @test norm_diff(x, N) ≈ T(norm(d)) rtol = 16eps(T)
      @test isfinite(norm_diff(x, N)) && !iszero(norm_diff(x, N))
    end
    # the sum of squares of this entry and its mirror overflows, the norm does not
    x = SymmetricSparseIterate(T, 2)
    x[2, 1] = floatmax(T) / 4
    @test norm(x) ≈ T(norm(big.(Matrix(x)))) rtol = 16eps(T)
    @test isfinite(norm(x))
  end

  @testset "non-finite entries" begin
    for bad in (Inf, -Inf)
      x = owner([1.0 bad; bad 2.0])
      @test norm(x) == Inf && norm(x, 1) == Inf && norm(x, Inf) == Inf
    end
    x = owner([1.0 NaN; NaN 2.0])
    @test isnan(norm(x)) && isnan(norm(x, 1)) && isnan(norm(x, Inf))
    @test isnan(norm(owner([Inf NaN; NaN 1.0])))
  end
end

@testset "A * X * B" begin
  rng = MersenneTwister(37)
  @testset "finite values against the dense product, $T" for T in (Float64, Float32)
    for p in (0, 1, 2, 3, 6, 15, 30), density in (0.05, 0.3, 0.9)
      M = randsym(rng, T, p, density)
      x = SymmetricSparseIterate(M)
      A, B = Symmetric(randn(rng, T, p, p)), Symmetric(randn(rng, T, p, p), :L)
      R = A * x * B
      @test R isa Matrix{T} && size(R) == (p, p)
      ref = big.(Matrix(A)) * big.(M) * big.(Matrix(B))
      bound = abs.(big.(Matrix(A))) * abs.(big.(M)) * abs.(big.(Matrix(B)))
      @test all(abs.(big.(R) .- ref) .<= (4p + 4) * big(eps(T)) .* bound)
    end
  end

  @testset "mixed element types" begin
    for (TA, TX, TB) in ((Float32, Float64, Float32), (Float64, Float32, Float64), (Float32, Float32, Float64))
      p = 7
      M = randsym(rng, TX, p, 0.3)
      x = SymmetricSparseIterate(M)
      A, B = Symmetric(randn(rng, TA, p, p)), Symmetric(randn(rng, TB, p, p))
      R = A * x * B
      @test R isa Matrix{promote_type(TA, TX, TB)}
      ref = big.(Matrix(A)) * big.(M) * big.(Matrix(B))
      bound = abs.(big.(Matrix(A))) * abs.(big.(M)) * abs.(big.(Matrix(B)))
      # an intermediate product in the narrowest type is rounded to it
      @test all(abs.(big.(R) .- ref) .<= (4p + 4) * big(max(eps(TA), eps(TX), eps(TB))) .* bound)
    end
  end

  @testset "entries of the product through dot" begin
    p = 9
    M = randsym(rng, Float64, p, 0.3)
    x = SymmetricSparseIterate(M)
    A, B = Symmetric(randn(rng, p, p)), Symmetric(randn(rng, p, p))
    R = Matrix(A) * M * Matrix(B)
    for (r, c) in ((1, 1), (2, 5), (9, 3))
      @test dot(view(A, :, r), x, view(B, :, c)) ≈ R[r, c]
    end
  end

  @testset "shape and element type errors" begin
    x = owner([1.0 2; 2 3])
    @test_throws DimensionMismatch Symmetric(ones(3, 3)) * x * Symmetric(ones(2, 2))
    @test_throws DimensionMismatch Symmetric(ones(2, 2)) * x * Symmetric(ones(3, 3))
    @test_throws ArgumentError Symmetric(ones(ComplexF64, 2, 2)) * x * Symmetric(ones(2, 2))
  end
end

@testset "no ambiguities of the products with an owner" begin
  names = (:mul!, :*, :dot, :norm, :norm_diff)
  ambiguities = filter(Test.detect_ambiguities(ProximalBase, LinearAlgebra; recursive = true)) do (a, b)
    (a.name in names || b.name in names) &&
      (occursin("SymmetricSparseIterate", string(a.sig)) || occursin("SymmetricSparseIterate", string(b.sig)))
  end
  @test isempty(ambiguities)
end

end
