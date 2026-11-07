module SymmetricIterateIndexingTest

# The owner of the variables of a symmetric matrix: index spaces (T01), the two live views and the
# dense copy (T02), and active zeros, compaction and handles (T03, T12). Expected values come from
# a dense matrix that each test updates itself, and from the column-major order of the lower triangle
# written out by hand, not from a second path through the owner.

using Test, ProximalBase, LinearAlgebra, SparseArrays, Random

# the lower triangle in column-major order: the coordinate IDs 1, 2, ..., q
lower_positions(p) = [(r, c) for c in 1:p for r in c:p]

# everything that identifies the state of x, hidden storage included
state(x) = (x.p, x.nnz, copy(x.nzval), copy(x.nzval2ind), copy(x.ind2nzval))

# f() throws something other than a BoundsError: a refused setindex!, for a read-only view
function rejected(f)
  err = try
    f()
    nothing
  catch e
    e
  end
  err !== nothing && !(err isa BoundsError)
end

@testset "T01: index spaces and edges" begin
  @testset "p = 0" begin
    X = SymmetricSparseIterate(Float64, 0)
    @test size(X) == (0, 0) && ncoordinates(X) == 0 && nactive(X) == 0
    @test isempty(coordinates(X))
    @test Matrix(X) == zeros(0, 0) && size(Matrix(X)) == (0, 0)
    @test size(matrixview(X)) == (0, 0)
    @test_throws BoundsError X[1, 1]
    @test_throws BoundsError (X[1, 1] = 1.0)
  end

  @testset "p = 1" begin
    X = SymmetricSparseIterate(Float64, 1)
    X[1, 1] = 2.0
    @test coordinates(X) == [2.0] && matrixview(X) == [2.0;;]
    @test X[1, 1] == 2.0 && nactive(X) == 1
    @test_throws BoundsError X[2, 1]
    @test_throws BoundsError X[1, 2]
    @test_throws BoundsError coordinates(X)[2]
    @test_throws BoundsError (X[2, 1] = 1.0)
    @test_throws BoundsError matrixview(X)[2, 1]
  end

  @testset "p = 3" begin
    # by hand: (1,1), (2,1), (3,1), (2,2), (3,2), (3,3) are the IDs 1, ..., 6
    ids = [(1, 1) => 1, (2, 1) => 2, (3, 1) => 3, (2, 2) => 4, (3, 2) => 5, (3, 3) => 6]
    @test lower_positions(3) == first.(ids)
    for ((r, c), k) in ids
      @test tril_index(3, r, c) == k
      @test tril_cartesian(3, k) == (r, c)
    end
    X = SymmetricSparseIterate(Float64, 3)
    X[2, 2] = 7.0
    θ, A = coordinates(X), matrixview(X)
    @test θ[4] == 7.0                          # coordinate 4 is (2, 2)
    @test A[4] == 0.0 && A[5] == 7.0           # matrix linear index 4 is (1, 2), 5 is (2, 2)
    @test_throws BoundsError θ[7]
    @test_throws BoundsError θ[0]
    @test_throws BoundsError A[10]
    @test_throws BoundsError X[0, 1]
    @test_throws BoundsError X[1, 0]
    @test_throws BoundsError X[4, 1]
    @test_throws BoundsError X[1, 4]
    @test_throws BoundsError (X[0, 1] = 1.0)
    @test_throws BoundsError (X[3, 4] = 1.0)
    @test nactive(X) == 1                      # a failed write activates nothing

    # both triangles name one variable
    X[1, 3] = 5.0
    @test θ[3] == 5.0 && X[3, 1] == 5.0 && X[1, 3] == 5.0
    X[3, 1] = -1.0
    @test θ[3] == -1.0 && X[1, 3] == -1.0
    @test nactive(X) == 2
    @test A[1, 3] == A[3, 1] == -1.0
    # the same variable at matrix linear indices 3 and 7
    @test A[3] == A[7] == -1.0
  end

  @testset "the IDs follow the column-major order of the lower triangle, p = $p" for p in (2, 5, 8)
    positions = lower_positions(p)
    @test length(positions) == p * (p + 1) ÷ 2 == ncoordinates(SymmetricSparseIterate(p))
    for (k, (r, c)) in enumerate(positions)
      @test tril_index(p, r, c) == k && tril_cartesian(p, k) == (r, c)
    end
  end

  @testset "dense model of random writes, $T, p = $p" for T in (Float32, Float64), p in (0, 1, 2, 3, 7)
    rng = MersenneTwister(1000 + p)
    X = SymmetricSparseIterate(T, p)
    D = zeros(T, p, p)
    for _ in 1:(5 * p * p + 3)
      p == 0 && break
      i, j = rand(rng, 1:p), rand(rng, 1:p)
      v = rand(rng) < 0.3 ? zero(T) : T(randn(rng))
      X[i, j] = v
      D[i, j] = D[j, i] = v
      @test X[i, j] === v && X[j, i] === v
    end
    @test Matrix(X) == D && matrixview(X) == D
    @test collect(coordinates(X)) == [D[r, c] for (r, c) in lower_positions(p)]
    @test Array(X) == D && issymmetric(Matrix(X))
  end
end

@testset "T02: one variable, two entries, live and copy" begin
  X = SymmetricSparseIterate(Float64, 3)
  θ = coordinates(X)
  A = matrixview(X)
  @test size(A) == (3, 3) && A isa AbstractMatrix{Float64}
  @test θ isa AbstractVector{Float64} && length(θ) == 6

  X[2, 1] = 3.0
  @test A[1, 2] == A[2, 1] == 3.0
  @test nactive(X) == 1
  @test sum(abs, θ) == 3.0 && sum(abs, A) == 6.0     # the vector sees 3 once, the matrix twice

  B = Matrix(X)
  @test B isa Matrix{Float64} && B == [0 3 0; 3 0 0; 0 0 0]
  θ[tril_index(3, 2, 1)] = 5.0
  @test A[1, 2] == A[2, 1] == 5.0
  @test B[1, 2] == 3.0                               # the copy does not follow
  @test nactive(X) == 1
  @test θ == [0.0, 5.0, 0.0, 0.0, 0.0, 0.0]
  @test X[1, 2] == 5.0

  # a view is not a snapshot and Matrix(X), Array(X) are independent, exactly symmetric copies
  B[1, 1] = 9.0
  @test X[1, 1] == 0.0 && A[1, 1] == 0.0
  for C in (Matrix(X), Array(X))
    @test C isa Matrix{Float64} && issymmetric(C) && C == A
    C[3, 3] = 4.0
    @test X[3, 3] == 0.0
  end

  @testset "nothing can be written through the view or a wrapper" begin
    snapshot = (Matrix(X), nactive(X), copy(activecoordinates(X)))
    unchanged() = (Matrix(X), nactive(X), copy(activecoordinates(X))) == snapshot
    writes = [
      "A[1, 2]"         => () -> (A[1, 2] = 1.0),
      "A[4]"            => () -> (A[4] = 1.0),
      "view"            => () -> (view(A, 1:2, 1:2)[1, 2] = 1.0),
      "Symmetric(A)"    => () -> (Symmetric(A)[1, 1] = 1.0),
      "reshape"         => () -> (reshape(A, 9)[4] = 1.0),
      "vec"             => () -> (vec(A)[4] = 1.0),
      "transpose"       => () -> (transpose(A)[1, 2] = 1.0),
      "A .= "           => () -> (A .= 1.0),
      "fill!"           => () -> fill!(A, 0.0),
      "copyto!"         => () -> copyto!(A, ones(3, 3)),
    ]
    for (name, f) in writes
      @testset "$name" begin
        @test rejected(f)
        @test unchanged()
      end
    end
    @test X[2, 1] == 5.0 && nactive(X) == 1
  end

  @testset "allocating operations give ordinary writable arrays" begin
    @test similar(A) isa Matrix{Float64} && similar(A, Float32) isa Matrix{Float32}
    S = A .+ 1
    @test S isa Matrix{Float64} && S[1, 2] == 6.0
    S[1, 2] = 0.0
    @test X[1, 2] == 5.0
    C = copy(A)
    @test C isa Matrix{Float64}
    C[2, 1] = -1.0
    @test X[2, 1] == 5.0
    @test collect(A) isa Matrix{Float64}
  end

  @testset "alias discovery" begin
    @test Base.mightalias(A, θ)
    @test Base.mightalias(A, activevalues(X)) && Base.mightalias(A, activecoordinates(X))
    @test !Base.mightalias(A, matrixview(SymmetricSparseIterate(Float64, 3)))
    @test !Base.mightalias(A, zeros(3, 3))
    Y = Base.unaliascopy(A)
    X[3, 3] = 2.0
    @test Y[3, 3] == 0.0 && A[3, 3] == 2.0
  end

  @testset "live views of a state that changes, p = $p" for p in (1, 3, 6)
    rng = MersenneTwister(2000 + p)
    X = SymmetricSparseIterate(Float64, p)
    θ, A = coordinates(X), matrixview(X)
    D = zeros(p, p)
    for _ in 1:20
      r, c = rand(rng, 1:p), rand(rng, 1:p)
      v = randn(rng)
      if rand(rng) < 0.5
        X[r, c] = v
      else
        θ[tril_index(p, max(r, c), min(r, c))] = v
      end
      D[r, c] = D[c, r] = v
      @test A == D && Matrix(X) == D && θ == [D[a, b] for (a, b) in lower_positions(p)]
    end
  end
end

@testset "T03, T12: active zeros, compaction and handles" begin
  X = SymmetricSparseIterate(Float64, 3)
  θ = coordinates(X)

  X[3, 1] = 0.0                                   # absent: not activated
  @test nactive(X) == 0 && X[3, 1] == 0.0
  X[3, 1] = -0.0
  @test nactive(X) == 0
  activate!(X, 3)
  @test nactive(X) == 1 && isactive(X, 3) && θ[3] == 0.0
  X[2, 1] = 4.0
  @test nactive(X) == 2
  X[2, 1] = 0.0                                   # active: the zero is stored, the coordinate stays
  @test nactive(X) == 2 && isactive(X, 2) && θ[2] == 0.0

  act = activecoordinates(X)
  @test collect(act) == [3, 2]
  before = collect(act)
  for k in act                                    # a sweep that only updates active coordinates
    θ[k] = θ[k] + 1
  end
  @test collect(act) == before && nactive(X) == 2
  @test θ[3] == 1.0 && θ[2] == 1.0

  X[2, 2] = NaN
  @test isactive(X, 4) && isnan(θ[4]) && nactive(X) == 3
  X[3, 3] = Inf
  @test isactive(X, 6) && θ[6] == Inf

  @testset "activation order and dropzeros!" begin
    X = SymmetricSparseIterate(Float64, 3)
    activate!(X, 2)
    coordinates(X)[3] = 5.0
    coordinates(X)[6] = 7.0
    @test collect(activecoordinates(X)) == [2, 3, 6]      # in the order of activation
    @test collect(activevalues(X)) == [0.0, 5.0, 7.0]
    M = Matrix(X)
    dropzeros!(X)
    @test Set(activecoordinates(X)) == Set([3, 6]) && nactive(X) == 2
    @test Matrix(X) == M
    @test !isactive(X, 2) && isactive(X, 3) && isactive(X, 6)
    # the dropped slot is reused by the next activation
    X[2, 1] = 1.5
    @test isactive(X, 2) && nactive(X) == 3
    @test Matrix(X) == M + [0 1.5 0; 1.5 0 0; 0 0 0]
  end

  @testset "read-only IDs, mutable value handles" begin
    X = SymmetricSparseIterate(Float64, 3)
    coordinates(X)[3] = 5.0
    coordinates(X)[6] = 7.0
    ids, vals = activecoordinates(X), activevalues(X)
    @test ids isa AbstractVector{Int} && vals isa AbstractVector{Float64}
    snapshot = state(X)
    @test_throws ArgumentError (ids[1] = 2)
    @test_throws ArgumentError fill!(ids, 2)
    @test isequal(state(X), snapshot)
    # T12: the values pair with the IDs by slot
    @test length(vals) == nactive(X) == length(ids)
    @test all(a -> vals[a] == coordinates(X)[ids[a]], eachindex(ids))
    @test count(!iszero, vals) == 2
    @test collect(ids) == [3, 6]                    # (3, 1) then (3, 3)
    vals[1] = 0.0                                   # writing zero keeps the coordinate
    @test isactive(X, 3) && nactive(X) == 2 && X[3, 1] == 0.0
    @test count(!iszero, vals) == 1
    vals .*= 2
    @test X[3, 3] == 14.0 && X[3, 1] == 0.0
    vals[2] = 3.0
    @test X[3, 3] == 3.0 && matrixview(X)[3, 3] == 3.0   # a write through the handle reaches the matrix
  end

  @testset "a handle is not a snapshot, and a stale handle is memory safe" begin
    X = SymmetricSparseIterate(Float64, 4)
    for k in (5, 2, 9)
      coordinates(X)[k] = Float64(k)
    end
    ids, vals = activecoordinates(X), activevalues(X)
    @test length(ids) == length(vals) == 3
    coordinates(X)[1] = 1.0                         # a new activation: the old handles keep their length
    @test length(ids) == 3 && length(activecoordinates(X)) == 4
    @test length(vals) == 3 && length(activevalues(X)) == 4
    coordinates(X)[5] = 0.0
    dropzeros!(X)
    @test nactive(X) == 3
    # a handle taken before the compaction may show other slots, but reading it is in bounds
    stale = collect(ids)
    @test length(stale) == 3 && all(k -> 1 <= k <= ncoordinates(X), stale)
    @test all(isfinite, collect(vals))
    # reacquired handles describe the state
    @test Set(activecoordinates(X)) == Set([1, 2, 9])
    @test sum(activevalues(X)) == 12.0
  end

  @testset "Float32 narrowing of an absent and an active coordinate" begin
    X = SymmetricSparseIterate(Float32, 3)
    X[2, 1] = 1e-100                                # rounds to zero in Float32
    @test nactive(X) == 0 && X[2, 1] === 0.0f0
    coordinates(X)[2] = 1e-300
    @test nactive(X) == 0
    X[2, 1] = 1e-30                                 # representable
    @test nactive(X) == 1 && X[2, 1] === Float32(1e-30)
    X[2, 1] = 1e-100                                # an active coordinate stores the converted value
    @test nactive(X) == 1 && isactive(X, 2) && X[2, 1] === 0.0f0
    X[3, 1] = 1e300                                 # overflows to Inf, which is nonzero
    @test isactive(X, 3) && X[3, 1] === Inf32
    X[3, 3] = -1e-100                               # -0.0f0 is zero
    @test !isactive(X, 6)
    @test_throws InexactError (SymmetricSparseIterate(Int, 2)[1, 1] = 0.5)
    Y = SymmetricSparseIterate(Int, 2)
    Y[2, 1] = 3.0
    @test Y[1, 2] === 3 && isactive(Y, 2)
  end

  @testset "a failed write changes nothing" begin
    X = SymmetricSparseIterate(Float64, 3)
    X[2, 1] = 1.0
    activate!(X, 6)
    snapshot = state(X)
    @test_throws BoundsError (X[4, 1] = 2.0)
    @test_throws InexactError (X[1, 1] = 1 + 1im)
    @test isequal(state(X), snapshot)
  end
end

@testset "randomized activations, writes, compaction and imports against a dense model, $T, p = $p" for T in (Float32, Float64), p in (0, 1, 3, 6)
  rng = MersenneTwister(31 * p + (T === Float32 ? 1 : 2))
  q = p * (p + 1) ÷ 2
  pos = lower_positions(p)
  X = SymmetricSparseIterate(T, p)
  # the model: the dense matrix and the active IDs in order of activation
  D = zeros(T, p, p)
  active = Int[]
  ordered = true                                    # the slot order is still the order of activation
  model_write!(k, v) = begin
    t = convert(T, v)
    r, c = pos[k]
    if k in active
      D[r, c] = D[c, r] = t
    elseif !iszero(t)
      push!(active, k)
      D[r, c] = D[c, r] = t
    end
  end
  value() = (u = rand(rng); u < 0.25 ? 0.0 : u < 0.3 ? -0.0 : u < 0.35 ? NaN : u < 0.4 ? 1e-100 :
             u < 0.45 ? 1e300 : randn(rng))
  function check()
    @test isequal(Matrix(X), D)
    @test isequal(collect(coordinates(X)), [D[r, c] for (r, c) in pos])
    @test nactive(X) == length(active) && Set(activecoordinates(X)) == Set(active)
    ordered && @test collect(activecoordinates(X)) == active
    @test all(k -> isactive(X, k) == (k in active), 1:q)
    av = activevalues(X)
    @test all(a -> isequal(av[a], D[pos[activecoordinates(X)[a]]...]), eachindex(av))
  end
  check()
  for step in 1:60
    q == 0 && break
    op = rand(rng, 1:7)
    if op == 1                                     # a pair write
      i, j = rand(rng, 1:p), rand(rng, 1:p)
      v = value()
      X[i, j] = v
      model_write!(tril_index(p, max(i, j), min(i, j)), v)
    elseif op == 2                                 # a coordinate write
      k = rand(rng, 1:q)
      v = value()
      coordinates(X)[k] = v
      model_write!(k, v)
    elseif op == 3                                 # an explicit active zero
      k = rand(rng, 1:q)
      activate!(X, k)
      k in active || push!(active, k)
    elseif op == 4                                 # compaction: an active zero reads as an absent +0
      dropzeros!(X)
      filter!(k -> !iszero(D[pos[k]...]), active)
      for (k, (r, c)) in enumerate(pos)
        k in active || (D[r, c] = D[c, r] = zero(T))
      end
      ordered = false
    elseif op == 5                                 # a write through the handle of values
      if !isempty(active)
        a = rand(rng, 1:length(active))
        v = convert(T, value())
        k = activecoordinates(X)[a]
        activevalues(X)[a] = v
        r, c = pos[k]
        D[r, c] = D[c, r] = v
      end
    elseif op == 6                                 # a replacing import of a symmetric matrix
      M = zeros(Float64, p, p)
      for (r, c) in pos
        rand(rng) < 0.5 && continue
        M[r, c] = M[c, r] = value()
      end
      copyto!(X, M)
      D = map(v -> iszero(v) ? zero(T) : v, convert(Matrix{T}, M))     # a zero is absent: +0
      active = [k for (k, (r, c)) in enumerate(pos) if !iszero(D[r, c])]
      ordered = true
    else                                           # a copy is an independent state
      Y = copy(X)
      @test isequal(Matrix(Y), Matrix(X)) && collect(activecoordinates(Y)) == collect(activecoordinates(X))
      coordinates(Y)[rand(rng, 1:q)] = 1.0
      activate!(Y, rand(rng, 1:q))
    end
    check()
  end
end

end
