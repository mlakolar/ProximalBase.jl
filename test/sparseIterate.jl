module SparseIterateTest


using Test
using ProximalBase
using Random
using LinearAlgebra
using SparseArrays

Random.seed!(1)

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
      # from a dense matrix, its lower triangle
      D = randn(4, 4)
      y = SymmetricSparseIterate(D)
      @test Matrix(y) == Matrix(Symmetric(D, :L))
      @test nnz(y) == 10
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
      # copyto! and broadcasting from a matrix read its lower triangle
      E = randn(5, 5)
      @test Matrix(copyto!(t, E)) == Matrix(Symmetric(E, :L))
      t .= E
      @test Matrix(t) == Matrix(Symmetric(E, :L))
      fill!(t, 0)
      @test nnz(t) == 0
      fill!(t, 1.)
      @test Matrix(t) == ones(5, 5) && nnz(t) == 15
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

end
