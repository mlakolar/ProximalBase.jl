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

# the owner of the table of absent operations, and the operands of its rows
absent_owner() = (X = SymmetricSparseIterate(Float64, 3); X[2, 1] = 2.0; X[3, 3] = 5.0; activate!(X, 4); X)
const AM = Float64[1 2 3; 2 4 5; 3 5 6]
const AD = Diagonal([1.0, 2.0, 3.0])
const AQ = qr(AM).Q
const AIDX = CartesianIndices((1:3, 1:3))

# the rows whose call, on some Julia versions, asks for axes(X) before it looks for a method
const AXES_ROWS = ("accumulate!", "permutedims!")

const ABSENT_READS = Pair{String, Function}[
  "length" => X -> length(X),
  "iterate" => X -> iterate(X),
  "eachindex" => X -> eachindex(X),
  "firstindex" => X -> firstindex(X),
  "lastindex" => X -> lastindex(X),
  "ndims" => X -> ndims(X),
  "nnz" => X -> nnz(X),
  "sum" => X -> sum(X),
  "maximum" => X -> maximum(X),
  "extrema" => X -> extrema(X),
  "any" => X -> any(isnan, X),
  "findall" => X -> findall(iszero, X),
  "map" => X -> map(identity, X),
  "collect" => X -> collect(X),
  "zero" => X -> zero(X),
  "one" => X -> one(X),
  "X + X" => X -> X + X,
  "2 * X" => X -> 2 * X,
  "X * X" => X -> X * X,
  "A * X" => X -> AM * X,
  "dot(X, X)" => X -> dot(X, X),
  "X .+ 1" => X -> X .+ 1,
  "broadcast" => X -> broadcast(+, X, 1),
  "isapprox" => X -> isapprox(X, copy(X)),
  "X[1:2, 1]" => X -> X[1:2, 1],
  "X[:, 1]" => X -> X[:, 1],
  "X[:]" => X -> X[:],
  "X[CartesianIndex(1, 2)]" => X -> X[CartesianIndex(1, 2)],
  "X[end]" => X -> X[end],
  "lu(X)" => X -> lu(X),
  "qr(X)" => X -> qr(X),
  "cholesky(X)" => X -> cholesky(X),
  "bunchkaufman(X)" => X -> bunchkaufman(X),
  "sort(X)" => X -> sort(X; dims = 1),
  "reverse(X)" => X -> reverse(X),
  "rot180(X)" => X -> rot180(X),
  "cumsum(X)" => X -> cumsum(X; dims = 1),
  "accumulate(op, X)" => X -> accumulate(+, X),
  "Symmetric(X)" => X -> Symmetric(X),
  "convert(Matrix, X)" => X -> convert(Matrix{Float64}, X),
  "Array{Float64}(X)" => X -> Array{Float64}(X),
  "similar(X, 2, 2)" => X -> similar(X, 2, 2),
  "view(X, :, :)" => X -> view(X, :, :),
  "view(X, 1:2, 1:2)" => X -> view(X, 1:2, 1:2),
  "view(X, :)" => X -> view(X, :),
  "reshape(X, 9)" => X -> reshape(X, 9),
  "vec(X)" => X -> vec(X),
]

const ABSENT_WRITES = Pair{String, Function}[
  "X .= 1" => X -> (X .= 1),
  "X .= A" => X -> (X .= AM),
  "X .+= 1" => X -> (X .+= 1),
  "fill!(X, v)" => X -> fill!(X, 1),
  "map!(f, X, A)" => X -> map!(identity, X, AM),
  "map!(f, X, A, B)" => X -> map!(+, X, AM, AM),
  "map!(f, X)" => X -> map!(identity, X),
  "replace!(f, X)" => X -> replace!(identity, X),
  "replace!(X, a => b)" => X -> replace!(X, 1.0 => 2.0),
  "replace(X, a => b)" => X -> replace(X, 1.0 => 2.0),
  "setindex!(X, A, :, :)" => X -> setindex!(X, AM, :, :),
  "setindex!(X, vec(A), :)" => X -> setindex!(X, vec(AM), :),
  "setindex!(X, v, 1:2, 1)" => X -> setindex!(X, [1.0, 2.0], 1:2, 1),
  "setindex!(X, v, 1:2)" => X -> setindex!(X, [1.0, 2.0], 1:2),
  "setindex!(X, A, CartesianIndices)" => X -> setindex!(X, AM, AIDX),
  "X[CartesianIndex(1, 2)] = v" => X -> (X[CartesianIndex(1, 2)] = 1.0),
  "fillband!" => X -> LinearAlgebra.fillband!(X, 1.0, 0, 0),
  "read!" => X -> read!(IOBuffer(zeros(UInt8, 100)), X),
  "readbytes!" => X -> readbytes!(IOBuffer(zeros(UInt8, 100)), X),
  "hex2bytes!" => X -> hex2bytes!(X, "01020304"),
  "get!" => X -> get!(X, AM, (1:3, 1:3), 7.0),
  "sortperm!" => X -> sortperm!(X, AM; dims = 1),
  "accumulate!" => X -> accumulate!(+, X, AM; dims = 1),
  "cumsum!" => X -> cumsum!(X, AM; dims = 1),
  "sum!" => X -> sum!(X, ones(3, 3, 2)),
  "maximum!" => X -> maximum!(X, ones(3, 3, 2)),
  "count!" => X -> count!(X, trues(3, 3, 2)),
  "findmax!" => X -> findmax!(X, fill(CartesianIndex(0, 0, 0), 3, 3), ones(3, 3, 2)),
]

const ABSENT_ALGEBRA = Pair{String, Function}[
  "reverse!" => X -> reverse!(X),
  "sort!" => X -> sort!(X; dims = 1),
  "kron!(X, A, B)" => X -> kron!(X, AM, ones(1, 1)),
  "kron!(X, A, s)" => X -> kron!(X, AM, 2.0),
  "triu!" => X -> triu!(X),
  "triu!(X, k)" => X -> triu!(X, -5),
  "tril!" => X -> tril!(X),
  "tril!(X, k)" => X -> tril!(X, 5),
  "lmul!(D, X)" => X -> lmul!(AD, X),
  "lmul!(A, X)" => X -> lmul!(AM, X),
  "lmul!(Q, X)" => X -> lmul!(AQ, X),
  "lmul!(2I, X)" => X -> lmul!(2I, X),
  "rmul!(X, D)" => X -> rmul!(X, AD),
  "rmul!(X, A)" => X -> rmul!(X, AM),
  "rmul!(X, Q)" => X -> rmul!(X, AQ),
  "rmul!(X, 2I)" => X -> rmul!(X, 2I),
  "ldiv!(D, X)" => X -> ldiv!(AD, X),
  "ldiv!(F, X)" => X -> ldiv!(lu(AM), X),
  "ldiv!(2I, X)" => X -> ldiv!(2I, X),
  "ldiv!(X, F, B)" => X -> ldiv!(X, lu(AM), AM),
  "rdiv!(X, D)" => X -> rdiv!(X, AD),
  "rdiv!(X, 2I)" => X -> rdiv!(X, 2I),
  "lu!" => X -> lu!(X),
  "qr!" => X -> qr!(X),
  "cholesky!" => X -> cholesky!(X),
  "bunchkaufman!" => X -> bunchkaufman!(X),
  "transpose!" => X -> transpose!(X, AM),
  "adjoint!" => X -> adjoint!(X, AM),
  "permutedims!" => X -> permutedims!(X, AM, (2, 1)),
  "axpy!" => X -> axpy!(2.0, AM, X),
  "axpby!" => X -> axpby!(2.0, AM, 3.0, X),
  "axpy! with index lists" => X -> axpy!(1.0, [1.0, 2.0], [1, 2], X, [2, 3]),
  "mul!(X, A, B)" => X -> mul!(X, AM, AM),
  "mul!(X, A, s)" => X -> mul!(X, AM, 2.0),
  "mul!(X, s, A)" => X -> mul!(X, 2.0, AM),
  "mul!(X, A, B, α, β)" => X -> mul!(X, AM, AM, 1.0, 0.0),
  "mul!(X, Y, Z)" => X -> mul!(X, absent_owner(), absent_owner()),
  "mul!(X, 'N', 'N', A, B, α, β)" => X -> mul!(X, 'N', 'N', AM, AM, 1.0, 0.0),
]
@static if isdefined(LinearAlgebra, :copytrito!)
  push!(ABSENT_ALGEBRA, "copytrito!" => X -> LinearAlgebra.copytrito!(X, AM, 'L'))
end

const ABSENT_COPIES = Pair{String, Function}[
  "copy!(X, A)" => X -> copy!(X, AM),
  "copy!(X, Y)" => X -> copy!(X, absent_owner()),
  "copyto!(X, offset, v)" => X -> copyto!(X, 1, vec(AM)),
  "copyto!(X, offset, v, offset)" => X -> copyto!(X, 1, vec(AM), 1),
  "copyto!(X, offset, v, offset, n)" => X -> copyto!(X, 1, vec(AM), 1, 9),
  "copyto!(X, CartesianIndices, A, CartesianIndices)" => X -> copyto!(X, AIDX, AM, AIDX),
  "copyto!(X, ranges, A, ranges)" => X -> copyto!(X, 1:3, 1:3, AM, 1:3, 1:3),
  "copy_transpose! with a flag" => X -> LinearAlgebra.copy_transpose!(X, 1:3, 1:3, 'T', AM, 1:3, 1:3),
  "copy_transpose!" => X -> LinearAlgebra.copy_transpose!(X, 1:3, 1:3, AM, 1:3, 1:3),
  "copyto!(X, tuple)" => X -> copyto!(X, (1.0, 2.0)),
  "copyto!(X, generator)" => X -> copyto!(X, (1.0 for _ in 1:9)),
  "copyto!(X, number)" => X -> copyto!(X, 1.0),
  "copyto!(X, 2I)" => X -> copyto!(X, 2I),
  "copyto!(X, Q)" => X -> copyto!(X, AQ),
  "copyto!(X, broadcast)" => X -> copyto!(X, Base.Broadcast.broadcasted(identity, AM)),
]
@static if isdefined(LinearAlgebra, :copy_adjoint!)
  push!(ABSENT_COPIES, "copy_adjoint!" => X -> LinearAlgebra.copy_adjoint!(X, 1:3, 1:3, AM, 1:3, 1:3))
end

const ABSENT_BUILDS = Pair{String, Function}[
  "sparse(X)" => X -> sparse(X),
  "triu(X)" => X -> triu(X),
  "tril(X)" => X -> tril(X),
  "triu(X, k)" => X -> triu(X, 1),
  "tril(X, k)" => X -> tril(X, -1),
]

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

  # One table of the array operations that an owner has no method for. Each call is a MethodError, and
  # a call that took the owner as its destination leaves it unchanged. The owner is 3 × 3, with an
  # active zero and a diagonal entry.
  @testset "operations that an owner does not have, $name" for (name, rows) in (
      ("reading and traversal", ABSENT_READS), ("writing and filling", ABSENT_WRITES),
      ("in-place algebra", ABSENT_ALGEBRA), ("copies and index writes", ABSENT_COPIES),
      ("matrices built from the owner", ABSENT_BUILDS))
    @testset "$row" for (row, f) in rows
      X = absent_owner()
      before = state(X)
      err = try
        f(X)
        nothing
      catch e
        e
      end
      if row in AXES_ROWS
        # Julia reaches axes(X) first, on some versions: the one throwing method of the owner
        @test err isa MethodError || (err isa ArgumentError && occursin("has no axes", sprint(showerror, err)))
      else
        @test err isa MethodError
      end
      @test isequal(state(X), before)
    end
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

@testset "scaling the owner by a number scales its coordinates, in place" begin
  ops = (("rmul!", (x, s) -> rmul!(x, s)), ("lmul!", (x, s) -> lmul!(s, x)),
         ("rdiv!", (x, s) -> rdiv!(x, s)), ("ldiv!", (x, s) -> ldiv!(s, x)))
  @testset "$name, $T, p = $p" for (name, op) in ops, T in (Float32, Float64), p in (0, 1, 3)
    S = T[r + c for r in 1:p, c in 1:p]
    for s in (T(2), T(-0.5), 4)
      x = copyto!(SymmetricSparseIterate(T, p), S)
      ids = collect(activecoordinates(x))
      expected = op(copy(S), s)                                  # the same call on the dense matrix
      @test op(x, s) === x
      @test Matrix(x) == expected
      @test collect(activecoordinates(x)) == ids                 # the working set is not changed
    end
    # a complex number is refused before anything is written
    x = copyto!(SymmetricSparseIterate(T, p), S)
    before = state(x)
    @test_throws ArgumentError op(x, 2im)
    @test isequal(state(x), before)
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

  @testset "the cost of a display: empty, small and large owners" begin
    E = SymmetricSparseIterate(0)
    @test sprint(show, E) == "0×0 $(typeof(E)) with 0 stored entries"
    @test sprint(show, MIME"text/plain"(), E) == "0×0 $(typeof(E)) with 0 stored entries in the lower triangle"
    @test !occursin("nnz", sprint(show, SymmetricSparseIterate(2))) &&
          !occursin("nnz", sprint(show, MIME"text/plain"(), SymmetricSparseIterate(2)))
    # three active coordinates whatever p: the display does not grow with the p² entries
    function owner3(p)
      X = SymmetricSparseIterate(Float64, p)
      X[2, 1] = 1.5
      X[p, p] = -2.0
      X[p, 1] = 4.0
      X
    end
    allocated(f, X) = (f(X); @allocated f(X))
    small, large = owner3(100), owner3(2000)
    for f in (X -> sprint(show, X), X -> string(X), X -> repr(X), X -> sprint(show, MIME"text/plain"(), X))
      @test allocated(f, large) <= allocated(f, small) + 64
      @test allocated(f, large) < 65_536
    end
    # a very large owner: the storage is not touched
    X = owner3(10_000)
    @test sprint(show, X) == "10000×10000 $(typeof(X)) with 3 stored entries"
    out = sprint(show, MIME"text/plain"(), X)
    @test startswith(out, "10000×10000 $(typeof(X)) with 3 stored entries in the lower triangle:")
    @test count('\n', out) == 3 && occursin("[10000, 10000]  =  -2.0", out)
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
