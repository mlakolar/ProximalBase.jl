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


    @testset "show" begin
        io = IOBuffer()
        x = SparseIterate(3)

        show(io, MIME"text/plain"(), x)
        @test String(take!(io)) == "3-element ProximalBase.SparseIterate{Float64,1} with 0 stored entries"

        x[1] = 1.
        show(io, MIME"text/plain"(), x)
        @test String(take!(io)) == "3-element ProximalBase.SparseIterate{Float64,1} with 1 stored entry:\n  [1]  =  1.0\n"

        x[2] = 2.
        show(io, MIME"text/plain"(), x)
        @test String(take!(io)) == "3-element ProximalBase.SparseIterate{Float64,1} with 2 stored entries:\n  [1]  =  1.0\n  [2]  =  2.0\n"

        show(io, x)
        @test String(take!(io)) == "  [1]  =  1.0\n  [2]  =  2.0\n"


        y = SparseIterate(2,3)

        show(io, MIME"text/plain"(), y)
        @test String(take!(io)) == "2×3 ProximalBase.SparseIterate{Float64,2} with 0 stored entries"
        y[1,3] = 1.
        show(io, MIME"text/plain"(), y)
        @test String(take!(io)) == "2×3 ProximalBase.SparseIterate{Float64,2} with 1 stored entry:\n  [1, 3]  =  1.0"

        y[2,2] = 2.
        show(io, MIME"text/plain"(), y)
        @test String(take!(io)) == "2×3 ProximalBase.SparseIterate{Float64,2} with 2 stored entries:\n  [2, 2]  =  2.0\n  [1, 3]  =  1.0"
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
      M = (M + M') / 2
      x = SymmetricSparseIterate(M)
        
      B = randn(10, 20)
      C = M * B
      CC = zeros(10, 20)
      @test mul!(CC, x, B) == C
    end

end


@testset "AtomIterate" begin
    x = AtomIterate((2,3), 2, true)

    @test length(x) == 6
    @test size(x) == (2,3)
    @test IndexStyle(typeof(x)) == IndexLinear()

    x[:] = collect(1:6)
    y = [1. 3 5; 2 4 6]

    @test x.storage == y

    g = ProxL2(1., [10., 10.])
    out = prox(g, x)
    @test out.storage == zeros(2,3)
    out = prox(g, x, 1.)
    @test out.storage == zeros(2,3)

    g = ProxL2(1., [1., 10.])
    out = prox(g, x)
    @test out.storage[1,:] ≈ (1. - 1. / norm(y[1,:]))*y[1,:]
    @test out.storage[2,:] == zeros(3)

    g = ProxL2(1., [1., 1.])
    out = prox(g, x)
    @test out.storage[1,:] ≈ (1. - 1. / norm(y[1,:]))*y[1,:]
    @test out.storage[2,:] ≈ (1. - 1. / norm(y[2,:]))*y[2,:]

    @test_throws ArgumentError AtomIterate((2,3), 2, false)

    x = AtomIterate((2,3), 3, false)
    x[:] = collect(1:6)

    g = ProxL2(1., [1., 1.])
    @test_throws DimensionMismatch prox(g, x)

    g = ProxL2(1., [0., 1., 0.])
    out = prox(g, x)
    @test out.storage[:, 1] == y[:,1]
    @test out.storage[:, 2] ≈ (1. - 1. / norm(y[:,2]))*y[:,2]
    @test out.storage[:, 3] == y[:,3]
end


end
