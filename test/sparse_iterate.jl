module SparseIterateTest

# The active-set, storage and copy invariants, and one product of each kind, run in the Quick tier
# from sparse_iterate_basics.jl; this file keeps the exhaustive product grids and the rest.

using Test
using ProximalBase
using Random
using LinearAlgebra
using SparseArrays

using Statistics

Random.seed!(1)

function check_scaled_product(out, A, x, α, β; specialized = true)
    @nospecialize out A x α β
    expected = Array(out)
    mul!(expected, A, Array(x), α, β)
    @test parentmodule(which(mul!, typeof.((out, A, x, α, β)))) == (specialized ? ProximalBase : LinearAlgebra)
    @test mul!(out, A, x, α, β) === out
    @test out ≈ expected
end

function check_product_state(x, before)
    @nospecialize x before
    @test Array(x) == Array(before)
    @test activecoordinates(x) == activecoordinates(before)
    @test x.nzval[1:nnz(x)] == before.nzval[1:nnz(before)]
    @test x.ind2nzval == before.ind2nzval
end

@testset "sparseIterate" begin

    @testset "show" begin
        io = IOBuffer()
        x = SparseIterate(3)

        show(io, MIME"text/plain"(), x)
        @test String(take!(io)) == "3-element $(typeof(x)) with 0 stored entries"

        x[1] = 1.
        show(io, MIME"text/plain"(), x)
        @test String(take!(io)) == "3-element $(typeof(x)) with 1 stored entry:\n  [1]  =  1.0"

        x[2] = 2.
        show(io, MIME"text/plain"(), x)
        @test String(take!(io)) == "3-element $(typeof(x)) with 2 stored entries:\n  [1]  =  1.0\n  [2]  =  2.0"

        show(io, x)
        @test String(take!(io)) == "  [1]  =  1.0\n  [2]  =  2.0"

        # a limited display shows the first and the last stored entries, however far apart
        v = SparseIterate(10_000)
        for i in 100:100:10_000
            v[i] = 1.
        end
        lines = split(sprint(show, v; context = (:limit => true, :displaysize => (24, 80))), '\n')
        @test length(lines) == 16               # 7 entries, ⋮ and 8 entries
        @test count(contains('⋮'), lines) == 1 && contains(lines[8], '⋮')
        @test startswith(lines[1], "  [100 ") && startswith(lines[end], "  [10000]")
        @test length(split(sprint(show, v), '\n')) == 100


        y = SparseIterate(2,3)

        show(io, MIME"text/plain"(), y)
        @test String(take!(io)) == "2×3 $(typeof(y)) with 0 stored entries"
        y[1,3] = 1.
        show(io, MIME"text/plain"(), y)
        @test String(take!(io)) == "2×3 $(typeof(y)) with 1 stored entry:\n  [1, 3]  =  1.0"

        y[2,2] = 2.
        show(io, MIME"text/plain"(), y)
        @test String(take!(io)) == "2×3 $(typeof(y)) with 2 stored entries:\n  [2, 2]  =  2.0\n  [1, 3]  =  1.0"
    end

    @testset "other" begin
        x = SparseIterate(3)
        x[3] = 2.
        @test ncoordinates(x) == 3
        @test nnz(x) == 1
        @test convert(Vector, x) == [0., 0., 2.]
        @test convert(Array, x) == [0., 0., 2.]
        @test Vector(x) == [0., 0., 2.]

        x = SparseIterate(10)
        xv = randn(5)
        for i=1:5
            x[i] = xv[i]
        end
        y = randn(10)
        @test dot(xv, y[1:5]) ≈ dot(x, y)
        @test dot(y[1:5], xv) ≈ dot(x, y)
        @test dot(x, x) ≈ dot(xv, xv)
        @test dot(x, SparseIterate(y)) ≈ dot(xv, y[1:5])

        # convert
        n, m = 10, 100
        M = sprandn(n, m, 0.2)
        x = SparseIterate(M)
        @test ncoordinates(x) == n*m
        @test Matrix(x) == Matrix(M)
        @test Array(x) == Matrix(M)

        # constructor
        @test typeof(SparseIterate(2, 3)) == SparseIterate{Float64, 2}

        # multiplication
        v = sprandn(50, 0.2)
        M = randn(30, 50)
        x = SparseIterate(v)

        @test mul!(zeros(30), M, x) == M*v

        @test transpose(M) * SparseIterate(sprandn(30, 0.2)) isa Vector{Float64}
        w = sprandn(30, 0.2)
        y = SparseIterate(w)
        @test transpose(M) * y ≈ transpose(M) * Vector(w)
        @test mul!(zeros(50), transpose(M), y) ≈ transpose(M) * Vector(w)
        @test mul!(fill(NaN, 50), transpose(M), y) ≈ transpose(M) * Vector(w)     # every entry is written
        @test mul!(fill(NaN, 50), transpose(M), SparseIterate(30)) == zeros(50)
        @test_throws DimensionMismatch transpose(M) * x
        @test_throws DimensionMismatch mul!(zeros(30), transpose(M), y)

        z = similar(x, Float32)
        @test z isa SparseIterate{Float32, 1}
        @test size(z) == size(x)
        @test nnz(z) == 0

        # M' takes the same sparse path as transpose(M) for real M
        @test which(*, typeof.((M', y))).module == ProximalBase
        @test M' * y ≈ transpose(M) * Vector(w)
        @test mul!(zeros(50), M', y) ≈ transpose(M) * Vector(w)
        @test_throws DimensionMismatch M' * x
    end

    @testset "products with rows and columns" begin
        A = randn(30, 50)
        x = SparseIterate(sprandn(50, 0.2))
        r = SparseIterate(sprandn(30, 0.3))
        for i in (1, 17, 30)
            @test dot(view(A, i, :), x) ≈ (A * Vector(x))[i]           # (A x)ᵢ in O(nnz)
            @test which(dot, typeof.((view(A, i, :), x))).module == ProximalBase
        end
        for j in (1, 50)
            @test dot(view(A, :, j), r) ≈ (A' * Vector(r))[j]
        end
        @test dot(x, view(A, 2, :)) ≈ dot(view(A, 2, :), x)
    end

    @testset "matrix of coefficients" begin
        A = randn(20, 8)
        B = SparseIterate(sprandn(8, 3, 0.4))
        @test mul!(fill(NaN, 20, 3), A, B) ≈ A * Matrix(B)
        @test_throws DimensionMismatch mul!(zeros(20, 2), A, B)

        Σx = Symmetric(randn(8, 8)); Σy = Symmetric(randn(3, 3))
        a, b = randn(8), randn(3)
        @test dot(a, B, b) ≈ dot(a, Matrix(B) * b)
        for X in (B, SparseIterate(sparse(randn(8, 3))))      # sparse and dense supports
            @test Σx * X * Σy ≈ Σx * Matrix(X) * Σy
        end
        X = SparseIterate(8, 3)
        X[2, 3] = 1.5
        @test dot(view(Matrix(Σx), :, 4), X, view(Matrix(Σy), :, 2)) ≈ (Σx * Matrix(X) * Σy)[4, 2]
    end

    @testset "five-argument products" begin
        n, p, q = 7, 32, 4
        types = ((Float32, Float32, Float32), (Float64, Float64, Float64),
                 (Float64, Float32, Float64), (Float32, Float64, Float64))
        for (Ta, Tx, To) in types, support in (:empty, :sparse, :partial, :full), strided in (false, true)
            strided && !(Ta == Tx == To && Ta <: Real) && continue
            @testset "$Ta × $Tx → $To, $support, views=$strided" begin
                A = strided ? view(randn(Ta, 2n, 2p), 1:2:2n, 1:2:2p) : randn(Ta, n, p)
                B = strided ? view(randn(Ta, 2p, 2n), 1:2:2p, 1:2:2n) : randn(Ta, p, n)
                for dims in ((p,), (p, q))
                    support == :sparse && length(dims) == 1 && continue
                    dense = randn(Tx, dims...)
                    support == :empty && fill!(dense, 0)
                    if support == :sparse
                        fill!(dense, 0)
                        dense[end] = one(Tx)
                    elseif support == :partial
                        if length(dims) == 1
                            dense[1:2:end] .= 0
                        else
                            dense[2:3:end] .= 0
                            dense[3:3:end] .= 0
                        end
                    end
                    x = SparseIterate(dense)
                    if length(dims) == 2 && support in (:sparse, :partial)
                        @test nnz(x) >= 1
                        @test (32 * nnz(x) <= length(x)) == (support == :sparse)
                    end
                    before = copy(x)
                    if length(dims) == 1
                        out = strided ? view(randn(To, 2n), 1:2:2n) : randn(To, n)
                        matrices = (A, transpose(B), adjoint(B))
                    else
                        out = strided ? view(randn(To, 2n, 2q), 1:2:2n, 1:2:2q) : randn(To, n, q)
                        matrices = (A,)
                    end
                    initial = copy(out)
                    S = Ta
                    for M in matrices, α in S.((0, 1, 2.5)), β in S.((0, 1, -0.5))
                        copyto!(out, initial)
                        iszero(β) && fill!(out, NaN)
                        check_scaled_product(out, M, x, α, β)
                    end
                    if support in (:sparse, :partial) && !strided
                        scalar_pairs = length(dims) == 2 ? ((true, false), (0, 0), (1, 0)) : ((true, false), (0, 0))
                        for M in matrices, (α, β) in scalar_pairs
                            fill!(out, NaN)
                            check_scaled_product(out, M, x, α, β)
                        end
                    end
                    check_product_state(x, before)
                end
            end
        end
    end

    @testset "five-argument product dimensions" begin
        A = zeros(3, 4)
        for α in (0.0, 2.5), β in (0.0, -0.5)
            out = fill(3., 3)
            @test_throws DimensionMismatch("A has second dimension 4, length of coef is 5") mul!(out, A, SparseIterate(5), α, β)
            @test out == fill(3., 3)
            @test_throws DimensionMismatch("A has first dimension 3, length of out is 2") mul!(zeros(2), A, SparseIterate(4), α, β)
            @test_throws DimensionMismatch("A has second dimension 4, coef has 5 rows") mul!(zeros(3, 2), A, SparseIterate(5, 2), α, β)
            @test_throws DimensionMismatch("out has size (2, 2)") mul!(zeros(2, 2), A, SparseIterate(4, 2), α, β)
            @test_throws DimensionMismatch("out has size (3, 1)") mul!(zeros(3, 1), A, SparseIterate(4, 2), α, β)
            for M in (transpose(A), adjoint(A))
                @test_throws DimensionMismatch("A has first dimension 3, length of coef is 4") mul!(zeros(4), M, SparseIterate(4), α, β)
                @test_throws DimensionMismatch("A has second dimension 4, length of out is 3") mul!(zeros(3), M, SparseIterate(3), α, β)
            end
        end
    end
end

@testset "sparseSymmetricIterate" begin

    @testset "generic functions agree with the dense matrix" begin
        # Base and LinearAlgebra read x through its indices, length and iteration
        p = 5
        A = sprandn(p, p, 0.4)
        M = Matrix(A + A') + 2p * I       # symmetric positive definite, with zeros
        S = SymmetricSparseIterate(sparse(M))
        b, B = randn(p), randn(p, 3)
        @test length(S) == p^2 == lastindex(S) && S[end] == M[end]
        @test [S[i] for i in 1:p^2] == vec(S) == S[:] == reshape(S, p^2) == vec(M)
        @test count(_ -> true, S) == p^2 && [v for v in S] == collect(S) == M
        @test mean(S) ≈ mean(M) && sum(abs2, S) ≈ sum(abs2, M) && sum(S; dims = 1) ≈ sum(M; dims = 1)
        @test extrema(S) == extrema(M) && findmax(S) == findmax(M)
        @test count(!iszero, S) == count(!iszero, M) && findall(!iszero, S) == findall(!iszero, M)
        @test norm(S) ≈ norm(M) && norm(S, 1) ≈ norm(M, 1) && norm(S, Inf) == norm(M, Inf)
        @test S == M && S ≈ M && isequal(S, M) && hash(S) == hash(M)
        @test Matrix{Float32}(S) == Float32.(M) && convert(Matrix{Float64}, S) == M
        @test copyto!(zeros(p, p), S) == M && Symmetric(S) == M && Diagonal(S) == Diagonal(M)
        @test S[2:4, 1:2] == M[2:4, 1:2] && S[:, 2] == M[:, 2] && view(S, 3, :) == M[3, :]
        @test S[S .> 0] == M[M .> 0] && S[CartesianIndex(3, 2)] == M[3, 2]
        @test permutedims(S) == M && hcat(S, S) == hcat(M, M) && [S S; S S] == [M M; M M]
        @test 2 .* S .+ 1 == 2 .* M .+ 1 && S + M == 2M && -S == -M && map(abs, S) == abs.(M)
        @test kron(S, S) == kron(M, M)
        # sparse, triu and tril build a SparseMatrixCSC from the stored entries
        @test sparse(S) isa SparseMatrixCSC && sparse(S) == M
        @test triu(S) == triu(M) && tril(S) == tril(M) && triu(S, 1) == triu(M, 1) && tril(S, -2) == tril(M, -2)
        # products
        @test S * b ≈ M * b && S' * b ≈ M * b && b' * S ≈ b' * M && transpose(b) * S ≈ transpose(b) * M
        @test S * Float32.(b) ≈ M * Float32.(b)
        @test S * B ≈ M * B && B' * S ≈ B' * M && S * S ≈ M * M && S * Float32.(B) ≈ M * Float32.(B)
        @test dot(S, M) ≈ dot(M, M) && dot(M, S) ≈ dot(M, M) && dot(S, S) ≈ dot(M, M) && dot(b, S, b) ≈ dot(b, M, b)
        # linear algebra
        @test tr(S) ≈ tr(M) && diag(S) == diag(M) && issymmetric(S) && !istriu(S)
        @test det(S) ≈ det(M) && logdet(S) ≈ logdet(M) && inv(S) ≈ inv(M) && S \ b ≈ M \ b
        @test eigvals(S) ≈ eigvals(M) && svdvals(S) ≈ svdvals(M) && cholesky(S).L ≈ cholesky(M).L
        @test opnorm(S) ≈ opnorm(M) && opnorm(S, 1) ≈ opnorm(M, 1) && exp(S) ≈ exp(M) && sqrt(S) ≈ sqrt(M)
    end

    @testset "show" begin
      s = SymmetricSparseIterate(3)
      @test sprint(show, MIME"text/plain"(), s) == "3×3 $(typeof(s)) with 0 stored entries in the lower triangle"
      s[3, 1] = 2.
      s[2, 2] = 1.
      @test sprint(show, MIME"text/plain"(), s) ==
        "3×3 $(typeof(s)) with 2 stored entries in the lower triangle:\n  [3, 1]  =  2.0\n  [2, 2]  =  1.0"
    end

    @testset "mul" begin
      n = 10
      M = sprandn(n, n, 0.1)
      M = (M + M') / 2 + spdiagm(0 => randn(n))   # diagonal entries exercise the r == c branch
      x = SymmetricSparseIterate(M)

      B = randn(10, 20)
      C = M * B
      CC = zeros(10, 20)
      @test mul!(CC, x, B) ≈ C

      U = UpperTriangular(randn(10, 10))
      CC = zeros(10, 10)
      @test mul!(CC, x, U) ≈ M * U

      b = randn(10)
      @test mul!(zeros(10), x, b) ≈ M * b
      @test_throws DimensionMismatch mul!(zeros(10), x, randn(5))

      # x is symmetric: x' and transpose(x) are x, and products from the left work
      @test transpose(x) === x && x' === x
      @test x' * b ≈ M * b
      @test b' * x ≈ b' * M
      @test transpose(b) * x ≈ transpose(b) * M
      # other element types take the sparse product too
      @test x * Float32.(b) ≈ M * Float32.(b)

      # a' X b, tr(A X), and A X B for symmetric A, B
      a = randn(n)
      @test dot(a, x, b) ≈ dot(a, Matrix(M), b)
      A = randn(n, n)
      @test dot(A, x) ≈ dot(A, Matrix(M)) && dot(x, A) ≈ dot(A, Matrix(M))
      Σx, Σy = Symmetric(randn(n, n)), Symmetric(randn(n, n))
      @test Σx * x * Σy ≈ Σx * Matrix(M) * Σy
      @test dot(view(Matrix(Σx), :, 3), x, view(Matrix(Σy), :, 7)) ≈ (Σx * Matrix(M) * Σy)[3, 7]
      # a sparse X uses only the rows where it has nonzeros
      y = SymmetricSparseIterate(n)
      y[4, 2] = 1.
      @test Σx * y * Σy ≈ Σx * Matrix(y) * Σy
    end
end


end
