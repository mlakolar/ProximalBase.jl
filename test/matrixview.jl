module MatrixViewTest

using Test, ProximalBase, LinearAlgebra, SparseArrays, Random

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
  @test NormL1(1.0)(X) == 6.0                    # the symmetric iterate and its matrix count both positions
  @test NormL1(1.0)(A) == 6.0

  B = Matrix(X)
  θ[tril_index(3, 2, 1)] = 5.0
  @test A[1, 2] == 5.0 && B[1, 2] == 3.0
  @test Matrix(A) == Matrix(X) && Array(A) == Matrix(X)
  @test sparse(A) == sparse(Matrix(X))
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

# A write must be rejected *because* the view is read-only: the failure is a
# refused setindex!, not a bounds, conversion or unrelated error. The concrete
# type differs between Julia versions and wrappers, so the message is checked.
function rejected_as_read_only(f)
  err = try
    f()
    nothing
  catch e
    e
  end
  err !== nothing && !(err isa BoundsError) && !(err isa InexactError) &&
    occursin(r"setindex!|read-only|not defined", sprint(showerror, err))
end

@testset "matrixview: nothing can be written through it" begin
  X = SymmetricSparseIterate(Float64, 3)
  X[2, 1] = 5.0
  A = matrixview(X)
  before = (Matrix(X), nactive(X), copy(activecoordinates(X)))
  unchanged() = (Matrix(X), nactive(X), copy(activecoordinates(X))) == before

  writes = [
    "A[1, 2] = v"           => () -> (A[1, 2] = 1.0),
    "A[1, 1] = v"           => () -> (A[1, 1] = 1.0),
    "A[4] = v"              => () -> (A[4] = 1.0),
    "view"                  => () -> (view(A, 1:2, 1:2)[1, 2] = 1.0),
    "Symmetric, diagonal"   => () -> (Symmetric(A)[1, 1] = 1.0),
    "Symmetric(A, :L) .="   => () -> (Symmetric(A, :L)[2, 2] = 1.0),
    "reshape"               => () -> (reshape(A, 9)[4] = 1.0),
    "vec"                   => () -> (vec(A)[4] = 1.0),
    "vec .="                => () -> (vec(A) .= 1.0),
    "transpose"             => () -> (transpose(A)[1, 2] = 1.0),
    "transpose .="          => () -> (transpose(A) .= 1.0),
    "adjoint .="            => () -> (A' .= 1.0),
    "Transpose(A) .="       => () -> (Transpose(A) .= 1.0),
    "A .= "                 => () -> (A .= 1.0),
    "A .+= "                => () -> (A .+= 1.0),
    "fill!"                 => () -> fill!(A, 0.0),
    "copyto!"               => () -> copyto!(A, ones(3, 3)),
    "view .="               => () -> (view(A, :, :) .= 2.0),
  ]
  for (name, f) in writes
    @testset "$name" begin
      @test rejected_as_read_only(f)
      @test unchanged()
    end
  end
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

@testset "matrixview: output aliasing the matrix storage" begin
  # the one-dimensional case: clearing the output would clear the stored entry
  X = SymmetricSparseIterate(Float64, 1)
  X[1, 1] = 2.0
  y = coordinates(X)
  @test_throws ArgumentError mul!(y, matrixview(X), [3.0], 1.0, 0.0)
  @test_throws ArgumentError mul!(y, matrixview(X), [3.0])
  @test_throws ArgumentError mul!(y, X, [3.0], 1.0, 0.0)
  @test X[1, 1] == 2.0 && y == [2.0]
  # p = 2 with two stored entries: activevalues(Y) has the length of the vector
  Y = SymmetricSparseIterate(Float64, 2)
  Y[1, 1] = 1.0; Y[2, 1] = 4.0
  v = activevalues(Y)
  @test length(v) == 2
  @test_throws ArgumentError mul!(v, matrixview(Y), zeros(2), 1.0, 1.0)
  @test_throws ArgumentError mul!(v, matrixview(Y), zeros(2))
  @test Y[1, 1] == 1.0 && Y[2, 1] == 4.0
  # a different symmetric iterate is fine
  Z = SymmetricSparseIterate(Float64, 1)
  Z[1, 1] = 5.0
  out = zeros(1)
  mul!(out, matrixview(X), [3.0], 1.0, 0.0)
  @test out == [6.0]
  mul!(coordinates(Z), matrixview(X), [1.0], 1.0, 0.0)   # writes into another symmetric iterate: allowed
  @test Z[1, 1] == 2.0 && X[1, 1] == 2.0
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

# counts the reads and writes of a vector, to check how much of it a kernel touches
mutable struct CountingVector <: AbstractVector{Float64}
  data::Vector{Float64}
  ops::Int
end
CountingVector(data) = CountingVector(data, 0)
Base.size(v::CountingVector) = size(v.data)
Base.getindex(v::CountingVector, i::Int) = (v.ops += 1; v.data[i])
Base.setindex!(v::CountingVector, x, i::Int) = (v.ops += 1; v.data[i] = x)
Base.IndexStyle(::Type{<:CountingVector}) = IndexLinear()

@testset "matrixview: both products touch O(p + s) entries" begin
  s = 100
  for p in (200, 400, 800)
    X = SymmetricSparseIterate(Float64, p)
    for k in 1:s
      X[rand(1:p), rand(1:p)] = randn()
    end
    @test nactive(X) <= s
    A = matrixview(X)
    for call in (:three, :five)
      B, C = CountingVector(randn(p)), CountingVector(randn(p))
      call == :three ? mul!(C, A, B) : mul!(C, A, B, 2.0, 0.5)
      # a dense product reads B p² times; the stored-entry kernel at most twice per stored entry
      @test B.ops <= 2 * nactive(X)
      @test C.ops <= 4p + 4 * nactive(X)
    end
  end
end

@testset "matrixview: five-argument product semantics" begin
  for p in (0, 1, 2, 6)
    X = SymmetricSparseIterate(Float64, p)
    for _ in 1:min(4, p * (p + 1) ÷ 2)
      X[rand(1:p), rand(1:p)] = randn()
    end
    A = matrixview(X)
    D = Matrix(X)
    v, z = randn(p), randn(p)
    for α in (0.0, 1.0, -2.5, 3), β in (0.0, 1.0, 0.5, -1)
      y = copy(z)
      mul!(y, A, v, α, β)
      @test y ≈ α * (D * v) + β * z
      @test y ≈ mul!(copy(z), D, v, α, β)
    end
    # β = 0 does not read the old destination; α = 0 does not read B
    y = fill(NaN, p)
    mul!(y, A, v, 2.0, 0.0)
    @test y ≈ 2.0 * (D * v)
    y = copy(z)
    mul!(y, A, fill(NaN, p), 0.0, 2.0)
    @test y ≈ 2.0 * z
    # a mixed real type for the output and the vector
    y32, v32 = Float32.(z), Float32.(v)
    mul!(y32, A, v32, 2.0, 0.5)
    @test y32 isa Vector{Float32}
    @test isapprox(y32, Float32.(2 * (D * Float64.(v32)) + 0.5 * Float64.(z)); rtol = 1e-5, atol = 1e-6)
  end
  X = SymmetricSparseIterate(Float64, 3)
  X[2, 1] = 1.0
  A = matrixview(X)
  @test_throws DimensionMismatch mul!(zeros(3), A, zeros(2), 1.0, 0.0)
  @test_throws DimensionMismatch mul!(zeros(2), A, zeros(3), 1.0, 0.0)
  v = ones(3)
  @test_throws ArgumentError mul!(v, A, v, 1.0, 0.0)
  @test_throws ArgumentError mul!(v, A, v, 1.0, 1.0)
  @test_throws ArgumentError mul!(zeros(3), A, ones(3), 1.0 + 2.0im, 0.0)
end

# A coefficient of the symmetric iterate that is zero contributes nothing: a stored explicit zero does not
# multiply a non-finite entry of v, and dropzeros! leaves the product unchanged.
@testset "matrixview: non-finite vector and a stored zero" begin
  v = [Inf, 1.0, 0.0]
  for form in (:three, :five)
    X = SymmetricSparseIterate(Float64, 3)
    X[2, 1] = 1.0
    activate!(X, tril_index(3, 3, 1))              # a stored explicit zero at (3, 1)
    A = matrixview(X)
    run!() = (y = zeros(3); form == :three ? mul!(y, A, v) : mul!(y, A, v, 1.0, 0.0); y)
    before = run!()
    dropzeros!(X)
    after = run!()
    @test nactive(X) == 1
    @test after == [1.0, Inf, 0.0]
    @test isequal(before, after)                     # unchanged by compaction
  end
end

@testset "matrixview: empty and one-dimensional symmetric iterates" begin
  A0 = matrixview(SymmetricSparseIterate(Float64, 0))
  @test size(A0) == (0, 0) && Matrix(A0) == zeros(0, 0)
  X1 = SymmetricSparseIterate(Float64, 1)
  X1[1, 1] = 2.0
  A1 = matrixview(X1)
  @test A1 == [2.0;;] && coordinates(X1) == [2.0]
  @test_throws BoundsError A1[2, 1]
end

# the triangles of a matrix, from the definition: the entry (i, j) is kept when j - i >= k (triu) or j - i <= k (tril)
triu_reference(D, k) = [j - i >= k ? D[i, j] : zero(eltype(D)) for i in axes(D, 1), j in axes(D, 2)]
tril_reference(D, k) = [j - i <= k ? D[i, j] : zero(eltype(D)) for i in axes(D, 1), j in axes(D, 2)]

@testset "matrixview: sparse, triu and tril are sparse matrices of the nonzero entries" begin
  @testset "$T, p = $p" for T in (Float32, Float64), p in (0, 1, 2, 3, 6)
    rng = MersenneTwister(77 + p)
    q = p * (p + 1) ÷ 2
    X = SymmetricSparseIterate(T, p)
    # nonzero values, active zeros (positive and negative) and coordinates that are absent
    for k in randperm(rng, q)
      u = rand(rng)
      if u < 0.35
        continue
      elseif u < 0.55
        activate!(X, k)
      elseif u < 0.65
        activate!(X, k)
        coordinates(X)[k] = -zero(T)
      else
        coordinates(X)[k] = randn(rng, T)
      end
    end
    p >= 2 && activate!(X, 2)
    D = Matrix(X)
    A = matrixview(X)
    S = sparse(A)
    @test S isa SparseMatrixCSC{T, Int} && size(S) == (p, p) && S == sparse(D) && S == D
    @test nnz(S) == count(!iszero, D)                    # a stored zero is not an entry
    @test issymmetric(S)
    for k in -p-1:p+1
      U, L = triu(A, k), tril(A, k)
      @test U isa SparseMatrixCSC{T, Int} && L isa SparseMatrixCSC{T, Int}
      @test U == triu_reference(D, k) && L == tril_reference(D, k)
      @test nnz(U) == count(!iszero, triu_reference(D, k)) && nnz(L) == count(!iszero, tril_reference(D, k))
      -p + 1 <= k <= p + 1 && @test U == triu(D, k) && L == tril(D, k)
    end
    @test triu(A) == triu(D) && tril(A) == tril(D)
    @test triu(A, 1) + tril(A) == D                      # the strict upper triangle and the lower one
  end

  @testset "the cost is that of the active coordinates, not of the p² entries" begin
    p = 3000
    X = SymmetricSparseIterate(Float64, p)
    for (i, j) in ((5, 2), (700, 700), (3000, 1), (1500, 1499))
      X[i, j] = i + j / 10_000
    end
    activate!(X, tril_index(p, 10, 3))
    A = matrixview(X)
    for f in (A -> sparse(A), A -> triu(A), A -> tril(A, -1), A -> triu(A, 2))
      f(A)
      @test (@allocated f(A)) < 1_000_000              # the dense matrix has 72 MB
    end
    S = sparse(A)
    @test nnz(S) == 7 && S[2, 5] == S[5, 2] == 5 + 2 / 10_000 && S[1, 3000] == 3000 + 1 / 10_000
    @test nnz(triu(A, 1)) == 3 && nnz(tril(A, -1)) == 3 && nnz(triu(A)) == 4
  end
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
  @test av == X.nzval[1:nactive(X)]
  # a view taken earlier keeps its length when a coordinate becomes active
  coordinates(X)[1] = 9.0
  @test length(av) == 2 && length(activevalues(X)) == 3
end

end
