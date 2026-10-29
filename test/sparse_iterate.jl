module SparseIterateTest


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

    @testset "issparse" begin
        @test issparse(SparseIterate(sparse(fill(1,5,5))))
        @test issparse(SparseIterate(fill(1,5,5)))
        @test issparse(SparseIterate(fill(1,5)))
    end

    @testset "dropzeros and nnz" begin
        x = SparseIterate(3)
        x[3] = 2.
        x[1] = 1.
        @test nnz(x) == 2
        x[1] = 0.
        @test nnz(x) == 2
        dropzeros!(x)
        @test nnz(x) == 1
        @test Vector(x) == [0., 0., 2.]

        x = SparseIterate(3)
        x[3] = 2.
        x[1] = 1.
        x[3] = 0.
        x[1] = 0.
        @test nnz(x) == 2
        dropzeros!(x)
        @test nnz(x) == 0
        @test Vector(x) == [0., 0., 0.]
    end

    @testset "activate!" begin
        x = SparseIterate(5)
        x[2] = 1.
        x[4] = 0.                       # assigning a zero does not store the entry
        @test nnz(x) == 1
        @test activate!(x, 4) === x
        @test nnz(x) == 2
        @test x[4] == 0.
        @test x.nzval2ind[1:nnz(x)] == [2, 4]
        @test x.nzval[1:nnz(x)] == [1., 0.]

        # activating a stored entry changes nothing
        activate!(x, 2)
        activate!(x, 4)
        @test nnz(x) == 2
        @test Vector(x) == [0., 1., 0., 0., 0.]

        # the activated entry is written in place, and dropzeros! removes it while zero
        x[4] = 3.
        @test nnz(x) == 2
        @test Vector(x) == [0., 1., 0., 3., 0.]
        x[4] = 0.
        dropzeros!(x)
        @test nnz(x) == 1
        @test x.nzval2ind[1:nnz(x)] == [2]

        @test_throws BoundsError activate!(x, 6)

        y = SparseIterate(Float32, 2, 3)
        activate!(y, 5)
        @test nnz(y) == 1
        @test y[1, 3] === 0f0
    end

    @testset "active coordinates" begin
        x = SparseIterate(6)
        x[4] = 1.
        x[2] = 3.
        activate!(x, 6)
        @test nactive(x) == 3
        @test activecoordinates(x) == [4, 2, 6]
        @test isactive(x, 2) && isactive(x, 6) && !isactive(x, 1)
        @test_throws BoundsError activecoordinates(x)[4]
        dropzeros!(x)
        @test activecoordinates(x) == [4, 2]
        x[4] = 0.
        dropzeros!(x)                   # the last active coordinate moves into the gap
        @test activecoordinates(x) == [2]
        @test ncoordinates(x) == 6
        @test coordinates(x) === x
    end

    @testset "setindex! converts" begin
        x = SparseIterate(3)
        x[1] = 1                        # an Int
        x[2] = 2f0                      # a Float32
        @test Vector(x) == [1., 2., 0.]
        x[1] = 0                        # a stored entry stays stored
        x[3] = 0                        # a zero is not stored
        @test nnz(x) == 2
        @test_throws InexactError (SparseIterate(Int, 2)[1] = 0.5)
        @test_throws BoundsError (x[4] = 1.)

        y = SparseIterate(2, 2)
        y[2, 1] = 3
        @test Matrix(y) == [0. 0.; 3. 0.]
    end

    @testset "fill! and norm" begin
        x = SparseIterate([0., 3., 0., -4.])
        @test norm(x) == 5.
        @test norm(x, 1) == 7. && norm(x, Inf) == 4. && norm(x, 0) == 2.
        @test norm(x, -Inf) == 0.                     # the entries that are not stored count
        @test fill!(x, 0) === x
        @test nnz(x) == 0 && Vector(x) == zeros(4)
        fill!(x, 2.)
        @test Vector(x) == fill(2., 4)
    end


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

    @testset "copy" begin
        v = sprand(20, 0.4)
        x = SparseIterate(v)
        y = copy(x)
        @test Vector(y) == Vector(v)
        @test Vector(y) == Vector(x)
        @test !(x === y)

        M = sprand(10, 20, 0.2)
        x = SparseIterate(M)
        y = copy(x)
        @test Matrix(y) == Matrix(M)
        @test Matrix(y) == Matrix(x)
        @test !(x === y)

        # copyto! into an iterate that holds other entries, via the sparse and the dense path
        for d in (0.02, 0.6)
            v, w = sprandn(200, d), sprandn(200, d)
            x, y = SparseIterate(v), SparseIterate(w)
            @test copyto!(x, y) === x
            @test Vector(x) == Vector(w)
            @test nnz(x) == nnz(y)
            @test Vector(copyto!(x, x)) == Vector(w)

            # entries set afterwards are stored correctly
            i = findfirst(iszero, Vector(w))
            x[i] = 7.
            @test x[i] == 7.
            @test nnz(x) == nnz(y) + 1
            @test Vector(y) == Vector(w)
        end
        @test_throws DimensionMismatch copyto!(SparseIterate(3), SparseIterate(4))
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

        # complex M' conjugates, so it must not use the transpose path
        Mc = complex.(M, randn(30, 50))
        @test which(*, typeof.((Mc', y))).module != ProximalBase
        @test Mc' * y ≈ Mc' * Vector(w)
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
                 (Float64, Float32, Float64), (Float32, Float64, Float64),
                 (ComplexF64, Float64, ComplexF64), (ComplexF64, ComplexF64, ComplexF64))
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
                    S = real(Ta)
                    for M in matrices, α in S.((0, 1, 2.5)), β in S.((0, 1, -0.5))
                        copyto!(out, initial)
                        iszero(β) && fill!(out, NaN)
                        specialized = !(M isa Adjoint && !(eltype(M) <: Real))
                        check_scaled_product(out, M, x, α, β; specialized = specialized)
                    end
                    if support in (:sparse, :partial) && !strided
                        scalar_pairs = length(dims) == 2 ? ((true, false), (0, 0), (1, 0)) : ((true, false), (0, 0))
                        for M in matrices, (α, β) in scalar_pairs
                            fill!(out, NaN)
                            specialized = !(M isa Adjoint && !(eltype(M) <: Real))
                            check_scaled_product(out, M, x, α, β; specialized = specialized)
                        end
                    end
                    if To <: Complex
                        for M in matrices
                            copyto!(out, initial)
                            specialized = !(M isa Adjoint && !(eltype(M) <: Real))
                            check_scaled_product(out, M, x, 1 + 0.5im, -0.5 + 0.25im; specialized = specialized)
                        end
                    end
                    check_product_state(x, before)
                end
            end
        end
    end

    @testset "five-argument stored zeros" begin
        A = [Inf 1. 3.; 2. 4. 5.]
        B = Matrix(transpose(A))
        x = SparseIterate([0., 2., 0.])
        activate!(x, 1)
        X = SparseIterate([0. 0.; 2. 0.; 0. -1.])
        activate!(X, 1)
        X_sparse = SparseIterate(3, 32)
        X_sparse[2, 1], X_sparse[3, 2] = 2., -1.
        activate!(X_sparse, 1)
        @test 32 * nnz(X_sparse) == length(X_sparse)
        @test 32 * nnz(X) > length(X)
        for (M, coef) in ((A, x), (A, X_sparse), (A, X), (transpose(B), x), (adjoint(B), x))
            before = copy(coef)
            dims = ndims(coef) == 1 ? (2,) : (2, size(coef, 2))
            reference = mul!(fill(NaN, dims...), M, coef)
            @test isnan(reference[1]) && reference[2] == 8.
            if ndims(coef) == 2
                @test reference[:, 2] == [-3., -5.]
                @test all(iszero, view(reference, :, 3:size(coef, 2)))
            end
            for (α, β) in ((1.0, 0.0), (2.5, -0.5))
                out = fill(iszero(β) ? NaN : 3., dims...)
                expected = iszero(β) ? α .* reference : α .* reference .+ β * 3.
                @test isequal(mul!(out, M, coef, α, β), expected)
            end
            @test mul!(fill(NaN, dims...), M, coef, 0.0, 0.0) == zeros(dims...)
            @test mul!(fill(3., dims...), M, coef, 0.0, -0.5) == fill(-1.5, dims...)
            @test all(isnan, mul!(fill(NaN, dims...), M, coef, 1.0, 1.0))
            check_product_state(coef, before)
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

    @testset "sesquilinear matrix products" begin
        for T in (ComplexF32, ComplexF64)
            a, b = T[1 + im, 2 - im], T[1, 2 + im]
            M = T[1 2; 0 3]
            X = SparseIterate(M)
            @test dot(a, X, b) ≈ dot(a, M, b)
            # A complex left vector must also be conjugated for real storage.
            @test dot(a, SparseIterate(real.(M)), b) ≈ dot(a, real.(M), b)
        end
    end

end

@testset "sparseSymmetricIterate" begin

    @testset "a symmetric matrix" begin
        s = SymmetricSparseIterate(3)
        s[2, 1] = 1                     # setindex! converts, as for SparseIterate
        s[1, 3] = 2f0
        @test s[1, 2] === 1. && s[3, 1] === 2.
        s[2, 2] = 5.
        M = [0. 1. 2.; 1. 5. 0.; 2. 0. 0.]
        @test Matrix(s) == M
        # linear indices, length and iteration follow the dense matrix
        @test length(s) == 9 && size(s) == (3, 3)
        @test [s[i] for i in 1:9] == vec(M)
        @test collect(s) == M
        @test vec(s) == vec(M)
        @test sum(s) == sum(M)
        @test s == M
        # an index outside the matrix does not alias an entry of the lower triangle
        @test_throws BoundsError s[4, 1]
        @test_throws BoundsError s[1, 4]
        @test_throws BoundsError (s[4, 1] = -1.)
        @test_throws BoundsError s[10]
        @test s[2, 2] == 5. && nnz(s) == 3
    end

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
        @test S * Float32.(b) ≈ M * Float32.(b) && S * complex.(b, b) ≈ M * complex.(b, b)
        @test S * B ≈ M * B && B' * S ≈ B' * M && S * S ≈ M * M && S * Float32.(B) ≈ M * Float32.(B)
        @test dot(S, M) ≈ dot(M, M) && dot(M, S) ≈ dot(M, M) && dot(S, S) ≈ dot(M, M) && dot(b, S, b) ≈ dot(b, M, b)
        # linear algebra
        @test tr(S) ≈ tr(M) && diag(S) == diag(M) && issymmetric(S) && !istriu(S)
        @test det(S) ≈ det(M) && logdet(S) ≈ logdet(M) && inv(S) ≈ inv(M) && S \ b ≈ M \ b
        @test eigvals(S) ≈ eigvals(M) && svdvals(S) ≈ svdvals(M) && cholesky(S).L ≈ cholesky(M).L
        @test opnorm(S) ≈ opnorm(M) && opnorm(S, 1) ≈ opnorm(M, 1) && exp(S) ≈ exp(M) && sqrt(S) ≈ sqrt(M)
    end

    @testset "writes update each entry once" begin
        p = 5
        A = sprandn(p, p, 0.4)
        S = SymmetricSparseIterate(sparse(A + A') + 2p * I)
        # a linear index is an entry of the dense matrix, and writing it writes
        # its mirror image
        S[7] = -1.                       # (2, 2)
        @test S[2, 2] == -1.
        S[2] = 3.                        # (2, 1), so (1, 2) too
        @test S[1, 2] == S[2, 1] == 3.
        @test_throws BoundsError S[0]
        @test_throws BoundsError S[p^2 + 1]
        @test_throws BoundsError (S[p^2 + 1] = 1.)
        # scaling and axpy update an off-diagonal entry once, not twice
        M = Matrix(S)
        @test Matrix(rmul!(copy(S), 2.)) == 2M && Matrix(lmul!(2., copy(S))) == 2M
        @test Matrix(rdiv!(copy(S), 2.)) == M / 2 && Matrix(ldiv!(2., copy(S))) == M / 2
        @test Matrix(axpy!(2., M, copy(S))) ≈ 3M && Matrix(axpby!(2., M, 3., copy(S))) ≈ 5M
        @test Matrix(axpy!(2., S, copy(S))) ≈ 3M
        @test_throws DimensionMismatch axpy!(2., vec(M), copy(S))
        @test Matrix(axpy!(0., fill(NaN, p, p), copy(S))) == M
        @test_throws DimensionMismatch axpy!(1., ones(3, 3), copy(S))
        @test_throws DimensionMismatch axpby!(1., ones(3, 3), 1., copy(S))
        # a stored zero stays stored, and a SparseIterate is scaled in O(nnz)
        x = SparseIterate(sparse([1., 0., 2.]))
        @test rmul!(x, 3.) === x && Vector(x) == [3., 0., 6.] && nnz(x) == 2
        @test Vector(ldiv!(3., x)) == [1., 0., 2.]
    end

    @testset "coordinates" begin
        p = 4
        s = SymmetricSparseIterate(p)
        s[3, 2] = 1.5
        s[1, 1] = -2.
        @test ncoordinates(s) == p * (p + 1) ÷ 2
        @test nactive(s) == 2
        @test activecoordinates(s) == [tril_index(p, 3, 2), tril_index(p, 1, 1)]
        @test isactive(s, tril_index(p, 3, 2)) && !isactive(s, tril_index(p, 2, 1))
        c = coordinates(s)
        @test c isa AbstractVector && length(c) == ncoordinates(s)
        @test c[tril_index(p, 3, 2)] == 1.5 && c[tril_index(p, 4, 1)] == 0.
        c[tril_index(p, 4, 1)] = 7.             # writes the entries (4, 1) and (1, 4)
        @test s[1, 4] == 7. && nnz(s) == 3
        @test_throws BoundsError c[ncoordinates(s) + 1]
        @test activate!(s, tril_index(p, 2, 2)) === s
        @test nactive(s) == 4 && s[2, 2] == 0.
        for k in 1:ncoordinates(s)
            r, cc = tril_cartesian(p, k)
            @test r >= cc && tril_index(p, r, cc) == k
        end
    end

    @testset "dropzeros and nnz" begin
        x = SymmetricSparseIterate(2)
        x[2, 2] = 2.
        x[1, 1] = 1.
        @test nnz(x) == 2
        x[1, 1] = 0.
        @test nnz(x) == 2
        dropzeros!(x)
        @test nnz(x) == 1
        @test Matrix(x) == [0.  0.; 0. 2.]
    end

    @testset "construct" begin
      n = 10
      M = sprandn(n, n, 0.1)
      M = (M + M') / 2
      x = SymmetricSparseIterate(M)
      @test ncoordinates(x) == n * (n+1) / 2
      @test Matrix(x) == Matrix(M)
      @test Array(x) == Matrix(M)
      @test typeof(SymmetricSparseIterate(2)) == SymmetricSparseIterate{Float64}
      # from a matrix that is not symmetric, only through Symmetric(D, :L)
      D = randn(4, 4)
      @test_throws ArgumentError SymmetricSparseIterate(D)
      @test_throws ArgumentError SymmetricSparseIterate(sparse(D))
      y = SymmetricSparseIterate(Symmetric(D, :L))
      @test Matrix(y) == Matrix(Symmetric(D, :L))
      @test nnz(y) == 10
      @test size(SymmetricSparseIterate(0)) == (0, 0)
      for p in (-1, -2)
        @test_throws ArgumentError SymmetricSparseIterate(p)
        @test_throws ArgumentError SymmetricSparseIterate(Float32, p)
      end
    end

    @testset "broadcasting and copyto!" begin
      D = randn(5, 5); D = D + D'
      s = SymmetricSparseIterate(D)
      s .= 2 .* s                      # each entry of the lower triangle once
      @test Matrix(s) ≈ 2D
      s .= abs.(s) .- 1
      @test Matrix(s) ≈ abs.(2D) .- 1
      t = similar(s)
      t .= s .+ D
      @test Matrix(t) ≈ abs.(2D) .- 1 .+ D
      # copyto! and broadcasting take a symmetric matrix, and one triangle of
      # a matrix that is not symmetric only through Symmetric
      E = randn(5, 5)
      F = E + E'
      @test_throws ArgumentError copyto!(t, E)
      @test Matrix(copyto!(t, Symmetric(E, :L))) == Matrix(Symmetric(E, :L))
      for source in (view(E, :, :), transpose(E), sparse(E), UpperTriangular(E))
        @test_throws ArgumentError copyto!(t, source)
      end
      for source in (view(F, :, :), transpose(F), sparse(F), Symmetric(sparse(E), :U))
        @test copyto!(t, source) === t
        @test Matrix(t) == Matrix(source)
      end
      for T in (Float64, ComplexF64)
        @test_throws ArgumentError copyto!(SymmetricSparseIterate(T, 5), SparseArrays.CHOLMOD.Dense(T.(E)))
        source = SparseArrays.CHOLMOD.Dense(T.(F))
        dest = SymmetricSparseIterate(T, 5)
        @test copyto!(dest, source) === dest
        @test Matrix(dest) == T.(F)
        # The differently typed destination takes CHOLMOD's untyped
        # intersection rather than its element-type-matched method.
        mixed = SymmetricSparseIterate(ComplexF64, 5)
        @test copyto!(mixed, source) === mixed
        @test Matrix(mixed) == ComplexF64.(F)
      end
      @test_throws DimensionMismatch copyto!(t, view(F, 1:4, 1:4))
      @test_throws ArgumentError t .= E
      t .= Symmetric(E, :L)
      @test Matrix(t) == Matrix(Symmetric(E, :L))
      fill!(t, 0)
      @test nnz(t) == 0
      fill!(t, 1.)
      @test Matrix(t) == ones(5, 5) && nnz(t) == 15
    end

    @testset "copyto! from sparse wrappers" begin
      A = sparse(ComplexF64[1+2im 3-4im; 5+6im 0])      # not symmetric
      B = sparse(ComplexF64[1+2im 3-4im; 3-4im 0])      # symmetric
      wraps = Any[transpose, adjoint]
      @static if isdefined(SparseArrays, :SparseMatrixCSCOrView)
        push!(wraps, M -> view(M, :, 1:2))
      end
      for wrap in wraps
        dest = fill!(SymmetricSparseIterate(ComplexF64, 2), 9)
        @test_throws ArgumentError copyto!(dest, wrap(A))
        @test Matrix(dest) == fill(9, 2, 2)
        source = wrap(B)
        @test copyto!(dest, source) === dest
        @test Matrix(dest) == Matrix(source)
        @test_throws DimensionMismatch copyto!(SymmetricSparseIterate(3), source)
      end
      @static if isdefined(SparseArrays, :AdjOrTransSparseVectorOrView)
        v = sparse(ComplexF64[2+3im])
        B = sparse(reshape(ComplexF64[2+3im], 1, 1))
        for parent in (v, view(v, :), view(B, :, 1)), wrap in (transpose, adjoint)
          source = wrap(parent)
          dest = SymmetricSparseIterate(ComplexF64, 1)
          @test copyto!(dest, source) === dest
          @test Matrix(dest) == Matrix(source)
          @test_throws DimensionMismatch copyto!(SymmetricSparseIterate(2), source)
        end
        @test_throws DimensionMismatch copyto!(SymmetricSparseIterate(1), transpose(sparse([1., 2.])))
      end
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
      @test x * complex.(b, b) ≈ M * complex.(b, b)

      # a' X b, tr(A X), and A X B for symmetric A, B
      a = randn(n)
      @test dot(a, x, b) ≈ dot(a, Matrix(M), b)
      for T in (ComplexF32, ComplexF64)
        ac, bc = T.(complex.(a, reverse(a))), T.(complex.(b, reverse(b)))
        @test dot(ac, x, bc) ≈ dot(ac, Matrix(M), bc)
        Mc = T[1 2 + im; 2 + im 3 - im]
        xc = SymmetricSparseIterate(Mc)
        @test dot(ac[1:2], xc, bc[1:2]) ≈ dot(ac[1:2], Mc, bc[1:2])
      end
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

    @testset "copy" begin
      n = 6
      M = sprandn(n, n, 0.4)
      M = (M + M') / 2
      x = SymmetricSparseIterate(M)

      y = similar(x)
      @test y isa SymmetricSparseIterate{Float64}
      @test size(y) == size(x)
      @test nnz(y) == 0
      @test similar(x, Float32) isa SymmetricSparseIterate{Float32}

      copyto!(y, x)
      @test Matrix(y) == Matrix(x)
      @test nnz(y) == nnz(x)
      @test_throws DimensionMismatch copyto!(SymmetricSparseIterate(3), x)
      z = copy(x)
      @test z isa SymmetricSparseIterate && Matrix(z) == Matrix(x)

      # into a destination that holds other entries (sparse path)
      x = SymmetricSparseIterate(20); x[3, 1] = 1.; x[5, 5] = 2.
      y = SymmetricSparseIterate(20); y[5, 5] = -1.; y[9, 2] = 4.
      copyto!(x, y)
      @test Matrix(x) == Matrix(y)
      @test nnz(x) == 2
      x[1, 3] = 6.
      @test x[3, 1] == 6.
      @test nnz(x) == 3
      @test Matrix(y)[3, 1] == 0.
    end
end


@testset "sparse triple products agree with dense products" begin
    rng = Xoshiro(20261025)
    for T in (Float32, Float64)
        m, n = 8, 10
        A = Symmetric(randn(rng, T, m, m))
        B = Symmetric(randn(rng, T, n, n))
        # Concentrated support exercises each multiplication order, and no
        # support verifies the empty-product dimensions. Stored zeros must
        # contribute neither a row nor a column to the restricted product.
        for (rows, columns) in (([2], [3, 7]), ([2, 6], [3]), (Int[], Int[]))
            dense = zeros(T, m, n)
            for i in rows, j in columns
                dense[i, j] = T(i - j)
            end
            sparse = SparseIterate(dense)
            activate!(sparse, 1)
            product = A * sparse * B
            @test product ≈ A * dense * B
            @test size(product) == (m, n)
            @test eltype(product) == T
            @test Matrix(sparse) == dense
        end
    end
end


end
