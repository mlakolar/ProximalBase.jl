facts("sparseIterate") do

    context("dropzeros and nnz") do
        x = SparseIterate(3)
        x[3] = 2.
        x[1] = 1.
        @fact nnz(x) --> 2
        x[1] = 0.
        @fact nnz(x) --> 2
        dropzeros!(x)
        @fact nnz(x) --> 1
        @fact full(x) --> [0., 0., 2.]

        x = SparseIterate(3)
        x[3] = 2.
        x[1] = 1.
        x[3] = 0.
        x[1] = 0.
        @fact nnz(x) --> 2
        dropzeros!(x)
        @fact nnz(x) --> 0
        @fact full(x) --> [0., 0., 0.]
    end


    context("show") do
        io = IOBuffer()
        x = SparseIterate(3)

        show(io, MIME"text/plain"(), x)
        @fact String(take!(io)) == "3-element ProximalBase.SparseIterate{Float64,1} with 0 stored entries" --> true

        x[1] = 1.
        show(io, MIME"text/plain"(), x)
        @fact String(take!(io)) == "3-element ProximalBase.SparseIterate{Float64,1} with 1 stored entry:\n  [1]  =  1.0\n" --> true

        x[2] = 2.
        show(io, MIME"text/plain"(), x)
        @fact String(take!(io)) == "3-element ProximalBase.SparseIterate{Float64,1} with 2 stored entries:\n  [1]  =  1.0\n  [2]  =  2.0\n" --> true

        show(io, x)
        @fact String(take!(io)) == "  [1]  =  1.0\n  [2]  =  2.0\n" --> true


        y = SparseIterate(2,3)

        show(io, MIME"text/plain"(), y)
        @fact String(take!(io)) == "2×3 ProximalBase.SparseIterate{Float64,2} with 0 stored entries" --> true
        y[1,3] = 1.
        show(io, MIME"text/plain"(), y)
        @fact String(take!(io)) == "2×3 ProximalBase.SparseIterate{Float64,2} with 1 stored entry:\n  [1, 3]  =  1.0" --> true

        y[2,2] = 2.
        show(io, MIME"text/plain"(), y)
        @fact String(take!(io)) == "2×3 ProximalBase.SparseIterate{Float64,2} with 2 stored entries:\n  [2, 2]  =  2.0\n  [1, 3]  =  1.0" --> true
    end

    context("copy") do
        v = sprand(20, 0.4)
        x = convert(SparseIterate, v)
        y = copy(x)
        @fact full(y) --> full(v)
        @fact full(y) --> full(x)
        @fact x === y --> false

        M = sprand(10, 20, 0.2)
        x = convert(SparseIterate, M)
        y = copy(x)
        @fact full(y) --> full(M)
        @fact full(y) --> full(x)
        @fact x === y --> false
    end


    context("other") do
        x = SparseIterate(3)
        x[3] = 2.
        @fact numCoordinates(x) --> 3
        @fact nnz(x) --> 1
        @fact convert(Vector, x) --> [0., 0., 2.]
        @fact convert(Array, x) --> [0., 0., 2.]
        @fact full(x) --> [0., 0., 2.]

        x = SparseIterate(10)
        xv = randn(5)
        for i=1:5
            x[i] = xv[i]
        end
        y = randn(10)
        @fact dot(xv, y[1:5]) --> dot(x, y)
        @fact dot(y[1:5], xv) --> dot(x, y)

        # convert
        n, m = 10, 100
        M = sprandn(n, m, 0.2)
        x = convert(SparseIterate, M)
        @fact numCoordinates(x) --> n*m
        @fact full(x) --> full(M)
        @fact convert(Matrix, x) --> full(M)
        @fact convert(Array, x) --> full(M)

        # constructor
        @fact typeof(SparseIterate(2, 3)) --> SparseIterate{Float64, 2}

        # multiplication
        v = sprandn(50, 0.2)
        M = randn(30, 50)
        x = convert(SparseIterate, v)

        @fact A_mul_B!(zeros(30), M, x) --> M*v

    end

    facts("sparseSymmetricIterate") do

        context("dropzeros and nnz") do
            x = SymmetricSparseIterate(2)
            x[3] = 2.
            x[1] = 1.
            @fact nnz(x) --> 2
            x[1] = 0.
            @fact nnz(x) --> 2
            dropzeros!(x)
            @fact nnz(x) --> 1
            @fact full(x) --> [0.  0.; 0. 2.]

            x = SymmetricSparseIterate(2)
            x[3] = 2.
            x[1] = 1.
            x[3] = 0.
            x[1] = 0.
            @fact nnz(x) --> 2
            dropzeros!(x)
            @fact nnz(x) --> 0
            @fact full(x) --> [0. 0.; 0. 0.]
        end

        context("other") do
          # convert
          n = 10
          M = sprandn(n, n, 0.1)
          M = (M + M') / 2
          x = convert(SymmetricSparseIterate, M)
          @fact numCoordinates(x) --> n * (n+1) / 2
          @fact full(x) --> full(M)
          @fact convert(Matrix, x) --> full(M)
          @fact convert(Array, x) --> full(M)

          # constructor
          @fact typeof(SymmetricSparseIterate(2)) --> SymmetricSparseIterate{Float64}
        end

    end


    context("AtomIterate") do
        x = AtomIterate((2,3), 2, true)

        @fact length(x) --> 6
        @fact size(x) --> (2,3)
        @fact IndexStyle(typeof(x)) == IndexLinear() --> true

        x[:] = collect(1:6)
        y = [1. 3 5; 2 4 6]

        @fact x.storage --> y

        g = ProxL2(1., [10., 10.])
        out = prox(g, x)
        @fact out.storage --> zeros(2,3)
        out = prox(g, x, 1.)
        @fact out.storage --> zeros(2,3)

        g = ProxL2(1., [1., 10.])
        out = prox(g, x)
        @fact out.storage[1,:] --> roughly((1. - 1./vecnorm(y[1,:]))*y[1,:])
        @fact out.storage[2,:] --> zeros(3)

        g = ProxL2(1., [1., 1.])
        out = prox(g, x)
        @fact out.storage[1,:] --> roughly((1. - 1./vecnorm(y[1,:]))*y[1,:])
        @fact out.storage[2,:] --> roughly((1. - 1./vecnorm(y[2,:]))*y[2,:])

        @fact_throws ArgumentError AtomIterate((2,3), 2, false)

        x = AtomIterate((2,3), 3, false)
        x[:] = collect(1:6)

        g = ProxL2(1., [1., 1.])
        @fact_throws DimensionMismatch prox(g, x)

        g = ProxL2(1., [0., 1., 0.])
        out = prox(g, x)
        @fact out.storage[:, 1] --> y[:,1]
        @fact out.storage[:, 2] --> roughly((1. - 1./vecnorm(y[:,2]))*y[:,2])
        @fact out.storage[:, 3] --> y[:,3]
    end
end
