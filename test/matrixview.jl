module MatrixViewTest

using Test, ProximalBase, LinearAlgebra, SparseArrays

# the example of the 0.6 contract (docs/design/symmetric-sparse-iterate-0.6.md)
@testset "matrixview: live, read-only, independent copy" begin
  X = SymmetricSparseIterate(Float64, 3)
  θ = coordinates(X)
  A = matrixview(X)
  @test size(A) == (3, 3)
  @test A isa AbstractMatrix{Float64}

  X[2, 1] = 3.0
  @test A[1, 2] == A[2, 1] == 3.0
  @test nactive(X) == 1
  @test sum(abs, θ) == 3.0
  @test NormL1(1.0)(X) == 6.0

  B = Matrix(X)
  θ[tril_index(3, 2, 1)] = 5.0
  @test A[1, 2] == 5.0 && B[1, 2] == 3.0
  @test Matrix(A) == Matrix(X) && Array(A) == Matrix(X)
  @test sparse(A) == sparse(X)
  @test all(A .== Matrix(X))
end

@testset "matrixview: indices are matrix indices, not coordinates" begin
  X = SymmetricSparseIterate(Float64, 3)
  X[2, 2] = 7.0
  θ, A = coordinates(X), matrixview(X)
  @test tril_cartesian(3, 4) == (2, 2)
  @test θ[4] == 7.0 && A[4] == 0.0 && A[5] == 7.0     # linear 4 is (1, 2), linear 5 is (2, 2)
  @test_throws BoundsError A[10]
  @test_throws BoundsError A[0, 1]
  @test_throws BoundsError A[1, 4]
  @test_throws BoundsError θ[7]
end

@testset "matrixview: nothing can be written through it" begin
  X = SymmetricSparseIterate(Float64, 3)
  X[2, 1] = 5.0
  A = matrixview(X)
  before = (Matrix(X), nactive(X), copy(activecoordinates(X)))
  unchanged() = (Matrix(X), nactive(X), copy(activecoordinates(X))) == before
  @test_throws Exception (A[1, 2] = 1.0)
  @test unchanged()
  @test_throws Exception (A[4] = 1.0)
  @test_throws Exception (view(A, 1:2, 1:2)[1, 2] = 1.0)
  @test_throws Exception (Symmetric(A)[1, 2] = 1.0)
  @test_throws Exception (reshape(A, 9)[4] = 1.0)
  @test_throws Exception (A .= 1.0)
  @test_throws Exception fill!(A, 0.0)
  @test_throws Exception copyto!(A, ones(3, 3))
  @test unchanged()
end

@testset "matrixview: allocating operations give ordinary writable arrays" begin
  X = SymmetricSparseIterate(Float64, 3)
  X[2, 1] = 5.0
  A = matrixview(X)
  @test similar(A) isa Matrix{Float64}
  @test similar(A, Float32) isa Matrix{Float32}
  S = A .+ 1
  @test S isa Matrix && S[1, 2] == 6.0
  S[1, 2] = 0.0
  @test X[1, 2] == 5.0 && A[1, 2] == 5.0
  C = copy(A)
  @test C isa Matrix
  C[2, 1] = -1.0
  @test X[2, 1] == 5.0
  @test collect(A) isa Matrix
  @test transpose(A) === A && adjoint(A) === A
end

@testset "matrixview: aliasing hooks" begin
  X = SymmetricSparseIterate(Float64, 3)
  A, θ = matrixview(X), coordinates(X)
  @test Base.mightalias(A, θ)
  @test Base.mightalias(A, A)
  @test !Base.mightalias(A, matrixview(SymmetricSparseIterate(Float64, 3)))
  @test !Base.mightalias(A, zeros(3, 3))
  Y = Base.unaliascopy(A)
  X[3, 3] = 2.0
  @test Y[3, 3] == 0.0 && A[3, 3] == 2.0
end

@testset "matrixview: product with a dense vector" begin
  for p in (0, 1, 2, 7)
    X = SymmetricSparseIterate(Float64, p)
    for _ in 1:min(3, p * (p + 1) ÷ 2)
      X[rand(1:p), rand(1:p)] = randn()
    end
    A = matrixview(X)
    v = randn(p)
    y = fill(NaN, p)
    mul!(y, A, v)
    @test y ≈ Matrix(X) * v
    @test A * v ≈ Matrix(X) * v
    @test A' * v ≈ Matrix(X) * v
    z = randn(p)
    z0 = copy(z)
    mul!(z, A, v, 2.0, 0.5)
    @test z ≈ 2 * (Matrix(X) * v) + 0.5 * z0
  end
end

@testset "matrixview: empty and one-dimensional owners" begin
  A0 = matrixview(SymmetricSparseIterate(Float64, 0))
  @test size(A0) == (0, 0) && Matrix(A0) == zeros(0, 0)
  X1 = SymmetricSparseIterate(Float64, 1)
  X1[1, 1] = 2.0
  A1 = matrixview(X1)
  @test A1 == [2.0;;] && coordinates(X1) == [2.0]
  @test_throws BoundsError A1[2, 1]
end

@testset "matrixview: show" begin
  X = SymmetricSparseIterate(Float64, 3)
  X[2, 1] = 3.0
  @test occursin("matrixview", sprint(show, MIME("text/plain"), matrixview(X)))
end

@testset "activevalues: stored values in the order of activecoordinates" begin
  for X in (SymmetricSparseIterate(Float64, 4), SparseIterate(Float64, 10))
    cx = X isa SymmetricSparseIterate ? coordinates(X) : X
    cx[3] = 5.0; cx[7] = -2.0; cx[2] = 0.5
    act, av = activecoordinates(X), activevalues(X)
    @test length(av) == nactive(X) == 3
    @test all(a -> av[a] == cx[act[a]], eachindex(act))
    @test count(!iszero, av) == 3

    # writes through the view reach X, and a zero keeps the coordinate active
    av[2] = 0.0
    @test cx[act[2]] == 0.0 && isactive(X, act[2]) && nactive(X) == 3
    av .*= 2
    @test all(a -> av[a] == cx[act[a]], eachindex(act))
    @test count(!iszero, av) == 2
    dropzeros!(X)
    @test nactive(X) == 2 && length(activevalues(X)) == 2
    @test all(a -> activevalues(X)[a] == cx[activecoordinates(X)[a]], 1:2)

    # an empty iterate
    @test isempty(activevalues(similar(X)))
  end
end

@testset "activevalues: replaces by-slot access to the stored arrays" begin
  X = SymmetricSparseIterate(Float64, 3)
  coordinates(X)[2] = 4.0
  coordinates(X)[6] = -1.0
  av = activevalues(X)
  @test av == X.nzval[1:nnz(X)]
  # a view taken earlier keeps its length when a coordinate becomes active
  coordinates(X)[1] = 9.0
  @test length(av) == 2 && length(activevalues(X)) == 3
end

end
