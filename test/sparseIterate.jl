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
    v = sprand(10, 0.2)
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
  end
end
