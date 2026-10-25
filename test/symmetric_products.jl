module SymmetricProductsTest

using Test, ProximalBase, LinearAlgebra, SparseArrays, Random
using ..TestUtils

Random.seed!(35)

@testset "Symmetric products over active coordinates" begin
  for T in (Float32, Float64, ComplexF32, ComplexF64), uplo in (:U, :L)
    n = 12
    M = randn(T, n, n)
    A = Symmetric(M, uplo)
    dense = Matrix(A)
    # The ignored triangle may contain anything, including NaNs.
    for j in 1:n, i in 1:n
      ((uplo === :U && i > j) || (uplo === :L && i < j)) && (M[i, j] = T(NaN))
    end
    v = zeros(T, n)
    v[[1, 4, 12]] .= randn(T, 3)
    for x in (SparseIterate(v), AtomIterate(v, [[1, 4], [2, 5, 8], [3, 6, 7, 9, 10, 11, 12]]))
      out = fill(T(NaN), n)
      @test mul!(out, A, x) === out
      @test out ≈ dense * v
      @test A * x ≈ dense * v
      @test A * x isa Vector{T}
      @test allocs(mul!, out, A, x) == 0
      initial = randn(T, n)
      for (α, β) in ((T(0.7), T(-0.2)), (zero(T), zero(T)), (zero(T), T(0.3)))
        copyto!(out, initial)
        @test mul!(out, A, x, α, β) ≈ α .* (dense * v) .+ β .* initial
        @test allocs(mul!, out, A, x, α, β) == 0
      end
      @test mul!(fill(T(NaN), n), A, x, zero(T), zero(T)) == zeros(T, n)
      @test_throws DimensionMismatch mul!(zeros(T, n + 1), A, x)
    end
    B = zeros(T, n, 3)
    B[1, 1], B[4, 2], B[12, 3] = randn(T, 3)
    x = SparseIterate(B)
    out = zeros(T, n, 3)
    @test mul!(out, A, x) ≈ dense * B
    @test A * x ≈ dense * B
    @test A * x isa Matrix{T}
    @test allocs(mul!, out, A, x) == 0
    initial = copy(out)
    @test mul!(out, A, x, T(0.7), T(0.3)) ≈ T(0.7) .* (dense * B) .+ T(0.3) .* initial
    @test allocs(mul!, out, A, x, T(0.7), T(0.3)) == 0
    @test mul!(out, A, x, zero(T), zero(T)) == zeros(T, n, 3)
    @test_throws DimensionMismatch mul!(zeros(T, n, 2), A, x)
    @test_throws DimensionMismatch mul!(out, A, SparseIterate(zeros(T, n - 1, 3)))
    @test_throws DimensionMismatch mul!(zeros(T, n), A, SparseIterate(zeros(T, n - 1)))
    @test_throws DimensionMismatch mul!(zeros(T, n), A, AtomIterate(zeros(T, n - 1), [1:n-1]))
  end
  A = Symmetric(randn(6, 6))
  x = SparseIterate(Float32[1, 0, 2, 0, 0, 0])
  @test A * x ≈ Matrix(A) * Vector(x)
  @test eltype(A * x) == Float64
  @test mul!(zeros(6), A, SparseIterate(6)) == zeros(6)
  @test mul!(zeros(6), A, AtomIterate(6, 2)) == zeros(6)
end

@testset "strided destinations and mixed element types" begin
  for uplo in (:U, :L)
    data = randn(12, 12)
    A = Symmetric(view(data, 1:2:12, 2:2:12), uplo)
    values = Float32[1, 0, -2, 0, 0, 3]
    for x in (SparseIterate(values), AtomIterate(values, [1:3, 4:6]),
              SparseIterate(hcat(values, reverse(values))))
      out = ndims(x) == 1 ? view(zeros(12), 1:2:12) : view(zeros(12, 4), 1:2:12, 1:2:4)
      ref = Matrix(A) * Array(x)
      @test mul!(out, A, x) ≈ ref
      @test A * x ≈ ref
      @test eltype(A * x) == Float64
      initial = copy(out)
      @test mul!(out, A, x, 0.7f0, 0.3) ≈ 0.7f0 .* ref .+ 0.3 .* initial
      @test allocs(mul!, out, A, x) == 0
      @test allocs(mul!, out, A, x, 0.7f0, 0.3) == 0
    end
    @test mul!(zeros(6, 2), A, SparseIterate(zeros(Float32, 6, 2))) == zeros(6, 2)
  end
  A = Symmetric(Matrix{Float64}(I, 3, 3))
  values = Float32[0.1, 0.2, 0.3]
  for x in (SparseIterate(values), AtomIterate(values, [1:3]), SparseIterate(hcat(values, values)))
    ref = Float64(0.1f0) .* Float64.(Array(x))
    @test mul!(zeros(size(x)), A, x, 0.1f0, 0.) ≈ ref rtol = 4eps(Float64)
  end
end

@testset "raw symmetric quadratic has no hot allocations" begin
  for T in (Float32, Float64)
    X = randn(T, 20, 10)
    f = QuadraticFunction(Symmetric(X'X), randn(T, 10))
    v = zeros(T, 10); v[2] = T(0.5); v[7] = T(-1)
    for x in (SparseIterate(v), AtomIterate(v, [1:5, 6:10]))
      g = zeros(T, 10)
      @test f(x) ≈ f(v)
      @test gradient!(g, f, x) ≈ gradient(f, v)
      @test allocs(f, x) == 0
      @test allocs(value_and_gradient!, g, f, x) == 0
      @test allocs(gradient!, g, f, x) == 0
    end
  end
end

end
