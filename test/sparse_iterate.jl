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

# The M1 reference: the sum over the coordinates where the iterate's coefficient is nonzero
# (where both iterates' are, for two). The finite entries are small integers, so every
# summation order gives the same result and isequal compares classes (FN-4); a zero of a
# dense operand is an IEEE zero.
strong(v, x) = sum((v[j] * x[j] for j in 1:length(x) if !iszero(x[j])); init = zero(promote_type(eltype(v), eltype(x))))
strong2(x, y) = sum((x[j] * y[j] for j in 1:length(x) if !iszero(x[j]) && !iszero(y[j]));
                    init = zero(promote_type(eltype(x), eltype(y))))

# The five-argument form into a reused out: β = 0 overwrites the NaN already in out, β = 1 keeps it.
function check_five(M, x, ref)
    T = eltype(ref)
    o = fill(T(NaN), size(ref)); o[end] = 1
    @test isequal(mul!(copy(o), M, x, 2, 0), 2 .* ref)
    @test isequal(mul!(copy(o), M, x, 2, 1), 2 .* ref .+ o)
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

    @testset "zero coefficients are strong (M1, FT12)" begin
        for T in (Float32, Float64), bad in T.((NaN, Inf, -Inf)), k in 1:5, stored in (true, false)
            # coefficients [0, 2, 0, -1, 0]: explicit zeros at 1 and 5, an unstored zero at 3
            x = SparseIterate(T, 5); x[2] = 2; x[4] = -1; activate!(x, 1); activate!(x, 5)
            stored || dropzeros!(x)
            before = copy(x)
            v = T[1, 2, 3, 4, 5]; v[k] = bad                          # a non-finite entry at k
            A = T[1 2 3 4 5; -1 0 2 1 3]; A[:, k] .= bad              # a non-finite column k
            ref = [strong(view(A, i, :), x) for i in 1:2]
            @test isequal(dot(v, x), strong(v, x)) && isequal(dot(x, v), strong(v, x))
            @test isequal(dot(view(A, 2, :), x), ref[2])
            for M in (A, transpose(permutedims(A)), permutedims(A)')
                @test isequal(M * x, ref)
                check_five(M, x, ref)
            end
            P = T[2 1 0 1 3; 1 2 1 0 1; 0 1 2 1 0; 1 0 1 2 1; 3 1 0 1 2]; P[k, :] .= bad; P[:, k] .= bad
            for uplo in (:U, :L)
                refP = [strong(view(P, i, :), x) for i in 1:5]
                @test isequal(Symmetric(P, uplo) * x, refP)
                check_five(Symmetric(P, uplo), x, refP)
            end
            # two iterates: y holds the non-finite entry alone, or with explicit zeros
            # everywhere else, so that dot walks y or x
            for padded in (false, true)
                y = SparseIterate(T, 5); y[k] = bad
                padded && foreach(j -> activate!(y, j), 1:5)
                @test isequal(dot(x, y), strong2(x, y)) && isequal(dot(y, x), strong2(x, y))
            end
            # a non-finite coefficient that meets a zero of a dense operand propagates
            w = copy(x); w[2] = bad
            @test isnan(dot(T[1, 0, 1, 1, 1], w)) && isequal(dot(T[1, 0, 1, 1, 1], w), strong(T[1, 0, 1, 1, 1], w))
            B = T[1 2 3 4 5; -1 0 2 1 3]                                # B[2, 2] == 0 meets w[2]
            @test isnan((B * w)[2]) && isequal(B * w, [strong(view(B, i, :), w) for i in 1:2])
            @test isequal(transpose(permutedims(B)) * w, B * w)
            check_product_state(x, before)
        end
    end

    @testset "matrix coefficients: zero coefficients are strong (M1, FT12)" begin
        for T in (Float32, Float64), bad in T.((NaN, Inf, -Inf)), p in (64, 3), k in 1:3, stored in (true, false)
            # explicit zeros at (1, 1) and (1, 2)
            X = SparseIterate(T, p, 2); X[2, 1] = 2; X[3, 2] = -1; activate!(X, 1); activate!(X, p + 1)
            stored || dropzeros!(X)
            @test (32 * nnz(X) <= length(X)) == (p == 64)           # the stored-entry walk for p = 64
            before = copy(X)
            Xd = Matrix(X)
            A = T.(reshape(0:2p-1, 2, p) .% 5 .- 2); A[:, k] .= bad
            ref = [strong(view(A, i, :), view(Xd, :, c)) for i in 1:2, c in 1:2]
            @test isequal(A * X, ref)                               # LinearAlgebra's * reaches this mul!
            @test parentmodule(which(mul!, typeof.((ref, A, X)))) == ProximalBase
            check_five(A, X, ref)
            P = T.(reshape(0:p*p-1, p, p) .% 3); P = P + P'; P[k, :] .= bad; P[:, k] .= bad
            for uplo in (:U, :L)
                refP = [strong(view(P, i, :), view(Xd, :, c)) for i in 1:p, c in 1:2]
                @test isequal(Symmetric(P, uplo) * X, refP)
                check_five(Symmetric(P, uplo), X, refP)
            end
            a = T.(1:p); a[k] = bad; b = T[1, -1]
            @test isequal(dot(a, X, b), sum((a[i] * Xd[i, j] * b[j] for i in 1:p, j in 1:2 if !iszero(Xd[i, j])); init = zero(T)))
            check_product_state(X, before)
        end
    end

    @testset "the zero test comes before α (M1)" begin
        x = SparseIterate(2); x[2] = 1e-300; activate!(x, 1)
        # α·c underflows to zero, and the nonzero coefficient still meets NaN
        @test isnan(only(mul!(zeros(1), [Inf NaN], x, 1e-300, 0)))
        @test isnan(only(mul!(zeros(1), transpose([Inf; NaN;;]), x, 1e-300, 0)))
        @test isnan(mul!(zeros(2), Symmetric([1.0 NaN; NaN 1.0]), x, 1e-300, 0)[1])
        for p in (64, 3)
            X = SparseIterate(p, 1); X[2, 1] = 1e-300; activate!(X, 1)
            @test isnan(only(mul!(zeros(1, 1), fill(NaN, 1, p), X, 1e-300, 0)))
            @test isnan(mul!(zeros(p, 1), Symmetric(fill(NaN, p, p)), X, 1e-300, 0)[1])
        end
        # α = Inf does not meet the explicit zero: no Inf · 0
        x[2] = 1
        @test mul!(zeros(1), [5.0 2.0], x, Inf, 0) == [Inf]
        @test mul!(zeros(2), Symmetric([5.0 2.0; 2.0 1.0]), x, Inf, 0) == [Inf, Inf]
        for p in (64, 3)
            X = SparseIterate(p, 1); X[2, 1] = 1; activate!(X, 1)
            @test mul!(zeros(1, 1), fill(2.0, 1, p), X, Inf, 0) == fill(Inf, 1, 1)
        end
    end

    @testset "an empty product ignores α (M1)" begin
        # no nonzero coefficient: out becomes β·out on every path, also for α = Inf or NaN
        A = [1.0 2.0 3.0; 4.0 5.0 6.0]
        for coef in (SparseIterate(3), (z = SparseIterate(3); activate!(z, 1); activate!(z, 3); z)),
            α in (Inf, NaN), β in (0.0, 1.0)
            o = [7.0, NaN]
            expected = iszero(β) ? zeros(2) : o
            @test isequal(mul!(copy(o), A, coef, α, β), expected)
            for M in (transpose(permutedims(A)), permutedims(A)')
                @test isequal(mul!(copy(o), M, coef, α, β), expected)
            end
            @test isequal(mul!([7.0, NaN, 1.0], Symmetric([1.0 2 3; 2 4 5; 3 5 6]), coef, α, β),
                          iszero(β) ? zeros(3) : [7.0, NaN, 1.0])
        end
    end

    @testset "finite overflow can depend on the order (FN-4, A1)" begin
        # The sum follows the walk, which takes the argument with fewer stored entries (the
        # first at a tie), so a finite partial sum can overflow in one order only. The values
        # are those of today and are allowed to differ (numerics M1 and A1).
        M = floatmax(Float64)
        x = SparseIterate([M, M, -M]); y = SparseIterate(3); y[1] = 1; y[3] = 1; y[2] = 1
        @test dot(x, y) == Inf
        @test dot(y, x) == M
    end
end

@testset "sparseSymmetricIterate" begin

    @testset "generic functions agree with the dense matrix" begin
        # Base and LinearAlgebra read matrixview(x) through its indices, length and iteration
        p = 5
        A = sprandn(p, p, 0.4)
        M = Matrix(A + A') + 2p * I       # symmetric positive definite, with zeros
        S = SymmetricSparseIterate(sparse(M))
        V = matrixview(S)
        b, B = randn(p), randn(p, 3)
        @test length(V) == p^2 == lastindex(V) && V[end] == M[end]
        @test [V[i] for i in 1:p^2] == vec(V) == V[:] == reshape(V, p^2) == vec(M)
        @test count(_ -> true, V) == p^2 && [v for v in V] == collect(V) == M
        @test mean(V) ≈ mean(M) && sum(abs2, V) ≈ sum(abs2, M) && sum(V; dims = 1) ≈ sum(M; dims = 1)
        @test extrema(V) == extrema(M) && findmax(V) == findmax(M)
        @test count(!iszero, V) == count(!iszero, M) && findall(!iszero, V) == findall(!iszero, M)
        @test norm(V) ≈ norm(M) && norm(V, 1) ≈ norm(M, 1) && norm(V, Inf) == norm(M, Inf)
        @test V == M && V ≈ M && isequal(V, M) && hash(V) == hash(M)
        @test Matrix{Float32}(V) == Float32.(M) && convert(Matrix{Float64}, V) == M
        @test copyto!(zeros(p, p), V) == M && Symmetric(V) == M && Diagonal(V) == Diagonal(M)
        @test V[2:4, 1:2] == M[2:4, 1:2] && V[:, 2] == M[:, 2] && view(V, 3, :) == M[3, :]
        @test V[V .> 0] == M[M .> 0] && V[CartesianIndex(3, 2)] == M[3, 2]
        @test permutedims(V) == M && hcat(V, V) == hcat(M, M) && [V V; V V] == [M M; M M]
        @test 2 .* V .+ 1 == 2 .* M .+ 1 && V + M == 2M && -V == -M && map(abs, V) == abs.(M)
        @test kron(V, V) == kron(M, M)
        # sparse, triu and tril of the view build a SparseMatrixCSC from the stored entries
        @test sparse(V) isa SparseMatrixCSC && sparse(V) == M
        @test triu(V) == triu(M) && tril(V) == tril(M) && triu(V, 1) == triu(M, 1) && tril(V, -2) == tril(M, -2)
        # products
        @test V * b ≈ M * b && V' * b ≈ M * b && b' * V ≈ b' * M && transpose(b) * V ≈ transpose(b) * M
        @test V * Float32.(b) ≈ M * Float32.(b)
        @test V * B ≈ M * B && B' * V ≈ B' * M && V * V ≈ M * M && V * Float32.(B) ≈ M * Float32.(B)
        @test dot(S, M) ≈ dot(M, M) && dot(M, S) ≈ dot(M, M) && dot(V, V) ≈ dot(M, M) && dot(b, S, b) ≈ dot(b, M, b)
        # linear algebra
        @test tr(V) ≈ tr(M) && diag(V) == diag(M) && issymmetric(V) && !istriu(V)
        @test det(V) ≈ det(M) && logdet(V) ≈ logdet(M) && inv(V) ≈ inv(M) && V \ b ≈ M \ b
        @test eigvals(V) ≈ eigvals(M) && svdvals(V) ≈ svdvals(M) && cholesky(V).L ≈ cholesky(M).L
        @test opnorm(V) ≈ opnorm(M) && opnorm(V, 1) ≈ opnorm(M, 1) && exp(V) ≈ exp(M) && sqrt(V) ≈ sqrt(M)
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

      # x is symmetric: x' and transpose(x) are x, and the products of its view work from the left
      @test transpose(x) === x && x' === x
      v = matrixview(x)
      @test v' * b ≈ M * b
      @test b' * v ≈ b' * M
      @test transpose(b) * v ≈ transpose(b) * M
      # other element types take the sparse product too
      @test v * Float32.(b) ≈ M * Float32.(b)

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

@testset "copyto! from an orthogonal factor into a matrix SparseIterate" begin
    # the full logical matrix of a factorization object (Matrix(Q) is the thin factor for a rectangular one)
    dense(Q) = lmul!(Q, Matrix{eltype(Q)}(I, size(Q)))
    state(x) = (Matrix(x), collect(activecoordinates(x)), collect(activevalues(x)), nactive(x))
    # nontrivial values, an explicit active zero at (1, 2) and a non-canonical activation order
    function populated(T, m, n)
        x = SparseIterate(T, m, n)
        x[m, n] = 3
        x[1, 1] = 5
        x[2, 1] = -2
        activate!(x, LinearIndices((m, n))[1, 2])
        x[2, 2] = 7
        j = LinearIndices((m, n))[1, 2]
        @test x.ind2nzval[j] != 0 && x[1, 2] == 0
        @test j in activecoordinates(x)
        return x
    end
    # every family accepts its adjoint on Julia 1.10 and 1.13 (probed on both), so every row has one
    function factors(A::AbstractMatrix)
        fs = Any[("qr", qr(A).Q), ("qr pivoted", qr(A, ColumnNorm()).Q), ("lq", lq(A).Q)]
        size(A, 1) == size(A, 2) && push!(fs, ("hessenberg", hessenberg(A).Q))
        return vcat(fs, Any[("adjoint of " * name, adjoint(Q)) for (name, Q) in fs])
    end
    for T in (Float32, Float64), (m, n) in ((4, 4), (5, 5), (4, 3), (3, 5)),
        (name, Q) in factors(randn(MersenneTwister(m + n), T, m, n))
        k = size(Q, 1)
        @test size(Q) == (k, k)
        tol = 10 * eps(T)
        @testset "$name, $T, from a $m x $n matrix" begin
            x = populated(T, k, k)
            ref = populated(T, k, k)
            copyto!(ref, dense(Q))
            @test copyto!(x, Q) === x
            @test Matrix(x) ≈ dense(Q) atol = tol
            @test state(x) == state(ref)
            # a coordinate that was active stays active, even with a zero value
            @test issubset(activecoordinates(populated(T, k, k)), activecoordinates(x))
            # the write does not depend on what x held before
            y = SparseIterate(T, k, k)
            copyto!(y, Q)
            @test Matrix(y) == Matrix(x)
            # the method is the package's own
            @test parentmodule(which(copyto!, typeof.((x, Q)))) == ProximalBase

            # failures leave x unchanged, and a later write works
            before = state(x)
            z = populated(T, k, k)
            zbefore = state(z)
            @test_throws DimensionMismatch copyto!(z, qr(randn(MersenneTwister(1), T, k + 1, k + 1)).Q)
            @test state(z) == zbefore
            @test copyto!(z, Q) === z
            @test state(z) == before
        end
    end

    @testset "a thin QR factor writes its full logical size" begin
        A = randn(MersenneTwister(2), 4, 3)
        Q = qr(A).Q
        @test size(Q) == (4, 4) && size(Matrix(Q)) == (4, 3)
        x = populated(Float64, 4, 4)
        copyto!(x, Q)
        @test Matrix(x) ≈ dense(Q)
        @test_throws DimensionMismatch copyto!(populated(Float64, 4, 3), Q)
    end

    @testset "a source of another element type is converted to that of x" begin
        Q = qr(randn(MersenneTwister(4), Float64, 4, 4)).Q
        x = populated(Float32, 4, 4)
        copyto!(x, Q)
        @test eltype(x) == Float32 && Matrix(x) == Float32.(dense(Q))
    end

    @testset "explicit zeros" begin
        T = Float64
        Q = hessenberg(randn(MersenneTwister(3), T, 4, 4)).Q
        @test dense(Q)[1, 2] == 0
        x = SparseIterate(T, 4, 4)
        x[1, 2] = 5
        copyto!(x, Q)
        j = LinearIndices((4, 4))[1, 2]
        @test j in activecoordinates(x) && x[1, 2] == 0
        # a zero in an absent coordinate does not activate it
        y = SparseIterate(T, 4, 4)
        copyto!(y, Q)
        @test nactive(y) == count(!iszero, dense(Q))
        @test !(j in activecoordinates(y))
    end

    @testset "failures write nothing" begin
        for T in (Float32, Float64)
            A = randn(MersenneTwister(5), T, 3, 3)
            C = randn(MersenneTwister(6), Complex{T}, 3, 3)
            W = randn(MersenneTwister(7), T, 4, 4)
            for Q in (qr(A).Q, lq(A).Q, hessenberg(A).Q, qr(A, ColumnNorm()).Q)
                x = populated(T, 3, 3)
                before = state(x)
                @test_throws ArgumentError copyto!(x, qr(C).Q)
                @test state(x) == before
                @test_throws DimensionMismatch copyto!(x, qr(W).Q)
                @test_throws DimensionMismatch copyto!(x, qr(W, ColumnNorm()).Q)
                @test_throws DimensionMismatch copyto!(x, lq(W).Q)
                @test_throws DimensionMismatch copyto!(x, hessenberg(W).Q)
                @test state(x) == before
                @test copyto!(x, Q) === x
                @test Matrix(x) ≈ dense(Q)
            end
        end
        # a destination that cannot hold the values
        A = randn(MersenneTwister(8), 3, 3)
        for Q in (qr(A).Q, lq(A).Q, hessenberg(A).Q, qr(A, ColumnNorm()).Q)
            xi = SparseIterate(Int, 3, 3)
            xi[3, 3] = 4
            activate!(xi, LinearIndices((3, 3))[1, 2])
            xi[2, 1] = 6
            j = LinearIndices((3, 3))[1, 2]
            @test j in activecoordinates(xi) && xi[1, 2] == 0
            before = state(xi)
            @test_throws InexactError copyto!(xi, Q)
            @test state(xi) == before
            @test copyto!(xi, Matrix{Int}(I, 3, 3)) === xi
        end
    end
end


end
