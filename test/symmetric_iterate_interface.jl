module SymmetricIterateInterfaceTest

# The surface of the owner of the variables of a symmetric matrix: what it has (T09), what it does
# not have, the one throwing axes method (D3), the rejected fills (#638), equality and hash of two
# owners (D2), the display, and the retained storage.

using Test, ProximalBase, LinearAlgebra, SparseArrays, Random

lower_positions(p) = [(r, c) for c in 1:p for r in c:p]

# everything that identifies the state of x, hidden storage included
state(x) = (x.p, x.nnz, copy(x.nzval), copy(x.nzval2ind), copy(x.ind2nzval))

# the dense matrix of the coordinate values v, in the order of the lower triangle
function dense_from(::Type{T}, v, p) where {T}
  D = zeros(T, p, p)
  for (k, (r, c)) in enumerate(lower_positions(p))
    D[r, c] = D[c, r] = v[k]
  end
  D
end

# an owner with the matrix of the values v (the zeros of v absent or active, at random, and the
# negative zeros active), its coordinates activated in a random order
function owner_from(rng, ::Type{T}, v, p) where {T}
  X = SymmetricSparseIterate(T, p)
  for k in randperm(rng, length(v))
    if iszero(v[k])
      (signbit(v[k]) || rand(rng) < 0.5) && activate!(X, k)
    end
    coordinates(X)[k] = v[k]
  end
  X
end

@testset "T09: the interface surface" begin
  @testset "what an owner has, $T, p = $p" for T in (Float32, Float64, Int), p in (0, 1, 3, 6)
    X = SymmetricSparseIterate(T, p)
    q = p * (p + 1) ÷ 2
    @test !(X isa AbstractArray) && !(X isa AbstractMatrix) && !(X isa AbstractVector)
    @test supertype(typeof(X)) === Any
    @test size(X) == (p, p) && size(X, 1) == p && size(X, 2) == p
    @test eltype(X) === T && eltype(typeof(X)) === T
    @test ncoordinates(X) == q && nactive(X) == 0
    @test X isa SymmetricSparseIterate{T}
  end

  @testset "complex element types are refused at construction" begin
    for T in (ComplexF32, ComplexF64)
      @test_throws ArgumentError("SymmetricSparseIterate requires real entries") SymmetricSparseIterate(T, 3)
      @test_throws ArgumentError("SymmetricSparseIterate requires real entries") SymmetricSparseIterate(zeros(T, 2, 2))
      @test_throws ArgumentError("SymmetricSparseIterate requires real entries") similar(SymmetricSparseIterate(2), T)
    end
  end

  @testset "what an owner does not have" begin
    for T in (Float32, Float64), p in (0, 1, 3)
      X = SymmetricSparseIterate(T, p)
      @test_throws MethodError length(X)
      @test_throws MethodError iterate(X)
      @test_throws MethodError eachindex(X)
      @test_throws MethodError firstindex(X)
      @test_throws MethodError lastindex(X)
      @test_throws MethodError map(identity, X)
      @test_throws MethodError sum(X)
      @test_throws MethodError (X .= 1)
      @test_throws MethodError (X .+ 1)
      @test_throws MethodError nnz(X)
      @test_throws MethodError ndims(X)
      @test_throws MethodError collect(X)
      @test_throws MethodError vec(X)
      @test_throws MethodError view(X, :, :)
      @test_throws MethodError similar(X, 2, 2)
      @test !hasmethod(length, Tuple{typeof(X)})
      @test !hasmethod(iterate, Tuple{typeof(X)})
      @test !hasmethod(nnz, Tuple{typeof(X)})
      for v in (0, 1, 1.5, NaN, 0im, 1 + 0im, 1 + 1im, T(2))
        @test_throws MethodError fill!(X, v)
      end
      @test_throws MethodError isapprox(X, copy(X))
    end
    # the vector and the matrix views keep what the owner does not have
    X = SymmetricSparseIterate(Float64, 3)
    @test length(coordinates(X)) == 6 && length(matrixview(X)) == 9
    @test axes(matrixview(X)) == (Base.OneTo(3), Base.OneTo(3))
    @test nnz(sparse(matrixview(X))) == 0
  end

  @testset "D3: the one throwing axes method" begin
    for p in (0, 1, 3)
      X = SymmetricSparseIterate(Float64, p)
      err = try
        axes(X)
        nothing
      catch e
        e
      end
      @test err isa ArgumentError
      @test occursin("matrixview", sprint(showerror, err)) && occursin("size", sprint(showerror, err))
    end
  end

  @testset "x[k] with one integer is ambiguous and refused" begin
    for p in (0, 1, 3), k in (1, 4, 100)
      X = SymmetricSparseIterate(Float64, p)
      err = try
        X[k]
        nothing
      catch e
        e
      end
      @test err isa ArgumentError
      msg = sprint(showerror, err)
      @test occursin("coordinates(x)[$k]", msg) && occursin("matrixview(x)[$k]", msg)
      @test_throws ArgumentError (X[k] = 1.0)
      @test_throws ArgumentError (X[k] = [1.0])
      @test_throws ArgumentError (X[k] = 1 + 0im)
    end
    X = SymmetricSparseIterate(Float64, 3)
    X[2, 1] = 3.0
    snapshot = state(X)
    @test_throws ArgumentError X[4]
    @test_throws ArgumentError (X[4] = 1.0)
    @test isequal(state(X), snapshot)
  end

  @testset "#638: every fill of the owner is refused and nothing changes" begin
    for T in (Float32, Float64)
      X = SymmetricSparseIterate(T, 3)
      X[2, 1] = 2
      X[3, 3] = 5
      activate!(X, 4)                          # an active zero
      snapshot = state(X)
      matrix, ids, vals = Matrix(X), collect(activecoordinates(X)), collect(activevalues(X))
      for v in (0, 1, 0im, 1 + 0im)
        @test_throws MethodError fill!(X, v)
        @test isequal(state(X), snapshot)
        @test Matrix(X) == matrix && collect(activecoordinates(X)) == ids && collect(activevalues(X)) == vals
      end
    end
  end

  @testset "FP-7: fill! of the coordinates writes the active values only" begin
    p = 5
    q = p * (p + 1) ÷ 2
    # a stale tail of the value array, which a pass over all q slots would overwrite
    X = SymmetricSparseIterate{Float64}(fill(7.0, q), zeros(Int, q), zeros(Int, q), 0, p)
    coordinates(X)[3] = 2.0
    activate!(X, 9)
    coordinates(X)[4] = -1.0
    @test nactive(X) == 3 && all(==(7.0), X.nzval[4:end])
    fill!(coordinates(X), 0)
    @test all(==(7.0), X.nzval[4:end])
    @test nactive(X) == 3 && all(iszero, activevalues(X)) && all(iszero, Matrix(X))
    dropzeros!(X)                              # the supported clear of the owner
    @test nactive(X) == 0 && all(iszero, Matrix(X)) && isempty(activecoordinates(X))
    @test all(k -> !isactive(X, k), 1:q)

    # an absent coordinate is not activated by a zero, an active one stores it
    Y = SymmetricSparseIterate(Float64, 3)
    Y[2, 1] = 1.0
    fill!(coordinates(Y), -0.0)
    @test nactive(Y) == 1 && isequal(activevalues(Y), [-0.0])
    # a nonzero value activates every coordinate, in coordinate order
    Z = SymmetricSparseIterate(Float32, 3)
    Z[3, 3] = 4.0
    fill!(coordinates(Z), 2)
    @test nactive(Z) == 6 && Matrix(Z) == fill(2.0f0, 3, 3)
    @test collect(activecoordinates(Z)) == [6, 1, 2, 3, 4, 5]
    # a value that does not convert changes nothing
    W = SymmetricSparseIterate(Float64, 3)
    W[2, 1] = 1.0
    snapshot = state(W)
    @test_throws InexactError fill!(coordinates(W), 1 + 1im)
    @test isequal(state(W), snapshot)
    fill!(coordinates(W), 0im)                 # a complex zero converts
    @test all(iszero, activevalues(W))
    # an owner with a fill of its own would have changed the other fills: the vector keeps vector semantics
    V = SymmetricSparseIterate(Float64, 2)
    fill!(coordinates(V), NaN)
    @test nactive(V) == 3 && all(isnan, activevalues(V))
  end
end

@testset "D2: equality and hash of two owners" begin
  @testset "an active zero is the absent coordinate" begin
    X = SymmetricSparseIterate(Float64, 3)
    activate!(X, 2)
    Y = SymmetricSparseIterate(Float64, 3)
    @test X == Y && isequal(X, Y) && hash(X) == hash(Y)
    @test Matrix(X) == Matrix(Y)
  end

  @testset "an active -0.0 differs from an absent +0.0 under isequal, not under ==" begin
    X = SymmetricSparseIterate(Float64, 3)
    X[2, 1] = 1.0
    X[2, 1] = -0.0                             # active, negative zero
    Y = SymmetricSparseIterate(Float64, 3)
    @test X == Y && Y == X
    @test !isequal(X, Y) && !isequal(Y, X)
    @test hash(X) != hash(Y)
    @test Matrix(X) == Matrix(Y) && !isequal(Matrix(X), Matrix(Y))
    # the same negative zero in both: equal in every sense
    Y[2, 1] = 3.0
    Y[2, 1] = -0.0
    @test X == Y && isequal(X, Y) && hash(X) == hash(Y)
  end

  @testset "NaN: isequal and not ==" begin
    X = SymmetricSparseIterate(Float64, 3)
    X[2, 1] = NaN
    Y = copy(X)
    @test isequal(X, Y) && !(X == Y) && hash(X) == hash(Y)
    @test isequal(X, X) && !(X == X)
    Z = SymmetricSparseIterate(Float64, 3)
    Z[2, 1] = reinterpret(Float64, 0x7ff8000000000123)    # another NaN
    @test isequal(X, Z) && hash(X) == hash(Z)
    W = SymmetricSparseIterate(Float64, 3)
    @test !isequal(X, W) && !(X == W)
  end

  @testset "different working sets and slot orders" begin
    X = SymmetricSparseIterate(Float64, 4)
    for k in (7, 2, 5)
      coordinates(X)[k] = Float64(k)
    end
    Y = SymmetricSparseIterate(Float64, 4)
    for k in (5, 7, 2)
      coordinates(Y)[k] = Float64(k)
    end
    activate!(Y, 10)                           # an extra active zero
    @test collect(activecoordinates(X)) != collect(activecoordinates(Y))
    @test X == Y && isequal(X, Y) && hash(X) == hash(Y)
    dropzeros!(Y)
    @test X == Y && isequal(X, Y) && hash(X) == hash(Y)
    Y[1, 1] = 1.0
    @test !(X == Y) && !isequal(X, Y)
  end

  @testset "copies" begin
    X = SymmetricSparseIterate(Float64, 4)
    X[2, 1] = 3.0
    activate!(X, 6)
    X[4, 4] = -1.5
    Y = copy(X)
    @test Y == X && isequal(Y, X) && hash(Y) == hash(X) && Y !== X
    Y[3, 3] = 1.0
    @test !(Y == X) && !isequal(Y, X)
    @test X[3, 3] == 0.0                        # independent
    @test copy(X) == X && X == X
  end

  @testset "sizes and element types" begin
    @test SymmetricSparseIterate(2) != SymmetricSparseIterate(3)
    @test !isequal(SymmetricSparseIterate(2), SymmetricSparseIterate(3))
    @test !isequal(SymmetricSparseIterate(0), SymmetricSparseIterate(1))
    @test SymmetricSparseIterate(0) == SymmetricSparseIterate(Float32, 0)
    @test hash(SymmetricSparseIterate(Float32, 0)) == hash(SymmetricSparseIterate(Float64, 0))
    X32 = SymmetricSparseIterate(Float32, 3)
    X64 = SymmetricSparseIterate(Float64, 3)
    XI = SymmetricSparseIterate(Int, 3)
    for X in (X32, X64, XI)
      X[2, 1] = 2
      X[3, 3] = 1
    end
    @test X32 == X64 == XI && isequal(X32, X64) && isequal(X64, XI)
    @test hash(X32) == hash(X64) == hash(XI)
    X32[3, 3] = 0.1f0
    @test !(X32 == X64) && !isequal(X32, X64)
  end

  @testset "no comparison with an array and no isapprox" begin
    X = SymmetricSparseIterate(Float64, 3)
    X[2, 1] = 1.0
    @test (X == Matrix(X)) === false && (Matrix(X) == X) === false
    @test !isequal(X, Matrix(X)) && !isequal(Matrix(X), X)
    @test (X == matrixview(X)) === false
    @test_throws MethodError isapprox(X, copy(X))
    @test Matrix(X) == matrixview(X) && isapprox(matrixview(X), Matrix(X))
  end

  @testset "against the dense matrices, $T1 and $T2, p = $p" for T1 in (Float32, Float64), T2 in (Float32, Float64, Int), p in (0, 1, 2, 4)
    rng = MersenneTwister(100p + 10 * (T1 === Float32) + (T2 === Float32) + 2 * (T2 === Int))
    q = p * (p + 1) ÷ 2
    menu = T2 === Int ? [0.0, -0.0, 1.0, -2.0, 4.0] : [0.0, -0.0, 1.5, -2.25, NaN, Inf, 0.5]
    for _ in 1:80
      vx = [rand(rng, menu) for _ in 1:q]
      vy = copy(vx)
      if q > 0 && rand(rng) < 0.5
        vy[rand(rng, 1:q)] = rand(rng, menu)
      end
      X = owner_from(rng, T1, vx, p)
      Y = owner_from(rng, T2, vy, p)
      DX, DY = dense_from(T1, vx, p), dense_from(T2, vy, p)
      @test isequal(Matrix(X), DX) && isequal(Matrix(Y), DY)
      same = isequal(DX, DY)
      @test isequal(X, Y) == same && isequal(Y, X) == same
      @test (X == Y) == (DX == DY) && (Y == X) == (DX == DY)
      same && @test hash(X) == hash(Y)
      @test isequal(X, copy(X)) && hash(X) == hash(copy(X))
    end
  end
end

@testset "show" begin
  @testset "two arguments: a short line, not the storage" begin
    p = 4
    X = SymmetricSparseIterate(Float64, p)
    for (k, v) in ((3, 1.0), (7, -2.5), (10, 4.0))
      coordinates(X)[k] = v
    end
    activate!(X, 5)
    for io in (sprint(show, X), string(X), "$X", repr(X))
      @test occursin("4×4", io) && occursin("SymmetricSparseIterate", io)
      @test occursin("4 stored entries", io)
      @test !occursin("nzval", io) && !occursin("[", io) && !occursin("-2.5", io)
      @test !occursin('\n', io) && length(io) < 100
    end
    @test occursin("1 stored entry", sprint(show, copyto!(SymmetricSparseIterate(2), [0 0; 0 1.0])))
    @test occursin("0 stored entries", sprint(show, SymmetricSparseIterate(3)))
    # in an error message
    err = try
      X[0, 1]
    catch e
      e
    end
    @test err isa BoundsError
    @test !occursin("nzval", sprint(showerror, err))
    msg = sprint(showerror, ArgumentError("bad $X"))
    @test occursin("4×4", msg) && length(msg) < 120
    # an element type and a dimension of its own
    @test occursin("3×3 ", sprint(show, SymmetricSparseIterate(Float32, 3)))
    @test occursin("Float32", sprint(show, SymmetricSparseIterate(Float32, 3)))
  end

  @testset "text/plain counts the active coordinates" begin
    s = SymmetricSparseIterate(3)
    @test sprint(show, MIME"text/plain"(), s) == "3×3 $(typeof(s)) with 0 stored entries in the lower triangle"
    s[3, 1] = 2.0
    s[2, 2] = 1.0
    @test sprint(show, MIME"text/plain"(), s) ==
      "3×3 $(typeof(s)) with 2 stored entries in the lower triangle:\n  [3, 1]  =  2.0\n  [2, 2]  =  1.0"
    activate!(s, 6)                            # a stored zero counts, as nactive does
    out = sprint(show, MIME"text/plain"(), s)
    @test occursin("with 3 stored entries", out) && nactive(s) == 3
    @test occursin("[3, 3]  =  0.0", out)
    @test occursin("SymmetricSparseIterate", sprint(show, MIME"text/plain"(), matrixview(s)))
  end
end

@testset "retained storage: one T array and two Int arrays of length q, and small headers" begin
  for T in (Float32, Float64), p in (0, 1, 3, 20, 40)
    q = p * (p + 1) ÷ 2
    X = SymmetricSparseIterate(T, p)
    @test abs(Base.summarysize(X) - (sizeof(T) + 2 * sizeof(Int)) * q) <= 1024
    @test length(X.nzval) == length(X.nzval2ind) == length(X.ind2nzval) == q
    # activation does not grow the storage
    for k in 1:q
      coordinates(X)[k] = 1
    end
    @test nactive(X) == q
    @test abs(Base.summarysize(X) - (sizeof(T) + 2 * sizeof(Int)) * q) <= 1024
  end
  # the views and the handles do not copy the storage, which has 6560 bytes of values at p = 40
  X = SymmetricSparseIterate(Float64, 40)
  coordinates(X)[7] = 1.0
  for f in (coordinates, matrixview, activevalues, activecoordinates)
    f(X)
    @test (@allocated f(X)) < 256
  end
end

end
