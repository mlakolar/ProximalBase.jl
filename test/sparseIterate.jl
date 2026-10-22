module SparseIterateTest


using Test
using ProximalBase
using Random
using LinearAlgebra
using SparseArrays

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


    @testset "show" begin
        io = IOBuffer()
        x = SparseIterate(3)

        show(io, MIME"text/plain"(), x)
        @test String(take!(io)) == "3-element $(typeof(x)) with 0 stored entries"

        x[1] = 1.
        show(io, MIME"text/plain"(), x)
        @test String(take!(io)) == "3-element $(typeof(x)) with 1 stored entry:\n  [1]  =  1.0\n"

        x[2] = 2.
        show(io, MIME"text/plain"(), x)
        @test String(take!(io)) == "3-element $(typeof(x)) with 2 stored entries:\n  [1]  =  1.0\n  [2]  =  2.0\n"

        show(io, x)
        @test String(take!(io)) == "  [1]  =  1.0\n  [2]  =  2.0\n"


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
        @test numCoordinates(x) == 3
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

        # convert
        n, m = 10, 100
        M = sprandn(n, m, 0.2)
        x = SparseIterate(M)
        @test numCoordinates(x) == n*m
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

end

@testset "sparseSymmetricIterate" begin

    @testset "dropzeros and nnz" begin
        x = SymmetricSparseIterate(2)
        x[3] = 2.
        x[1] = 1.
        @test nnz(x) == 2
        x[1] = 0.
        @test nnz(x) == 2
        dropzeros!(x)
        @test nnz(x) == 1
        @test Matrix(x) == [0.  0.; 0. 2.]

        x = SymmetricSparseIterate(2)
        x[3] = 2.
        x[1] = 1.
        x[3] = 0.
        x[1] = 0.
        @test nnz(x) == 2
        dropzeros!(x)
        @test nnz(x) == 0
        @test Matrix(x) == [0. 0.; 0. 0.]
    end

    @testset "other" begin
      # convert
      n = 10
      M = sprandn(n, n, 0.1)
      M = (M + M') / 2
      x = SymmetricSparseIterate(M)
      @test numCoordinates(x) == n * (n+1) / 2
      @test Matrix(x) == Matrix(M)
      @test Array(x) == Matrix(M)

      # constructor
      @test typeof(SymmetricSparseIterate(2)) == SymmetricSparseIterate{Float64}
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
