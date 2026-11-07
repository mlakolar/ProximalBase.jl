module SymmetricIterateCopiesTest

# Copies of the state of an owner of the variables of a symmetric matrix, the checked replacing import
# of a matrix, and the checks that run before anything is written: T10 (copies and import) and T06
# (narrowing and structural equality).

using Test, ProximalBase, LinearAlgebra, SparseArrays, Random

lower_positions(p) = [(r, c) for c in 1:p for r in c:p]

# everything that identifies the state of x, hidden storage included
state(x) = (x.p, x.nnz, copy(x.nzval), copy(x.nzval2ind), copy(x.ind2nzval))

# f(x) throws E, with msg in its message, and leaves x unchanged
function rejects(f::F, x, E, msg = nothing) where {F}
  before = state(x)
  err = try
    f(x)
    nothing
  catch e
    e
  end
  @test err isa E
  msg === nothing || @test occursin(msg, sprint(showerror, err))
  @test isequal(state(x), before)
end

# the active set [3, 2, 6] of a 3 × 3 owner, at the values 5, 0, 7
function sample(T = Float64)
  X = SymmetricSparseIterate(T, 3)
  X[3, 1] = 5
  activate!(X, 2)
  X[3, 3] = 7
  X
end

# a destination with an explicit zero and entries of its own
function destination(T = Float64, p = 4)
  x = SymmetricSparseIterate(T, p)
  x[4, 1] = 7
  activate!(x, tril_index(p, 3, 2))
  x[2, 2] = -1
  x
end

const ASYMMETRIC = "must be symmetric"

@testset "T10: copies and import" begin
  @testset "copy keeps the state and is independent" begin
    X = sample()
    @test collect(activecoordinates(X)) == [3, 2, 6]
    Y = copy(X)
    @test Y isa SymmetricSparseIterate{Float64} && Y !== X
    @test isequal(Matrix(Y), Matrix(X))
    @test collect(activecoordinates(Y)) == [3, 2, 6]          # the order of activation
    @test collect(activevalues(Y)) == [5.0, 0.0, 7.0]         # and the explicit zero
    @test isactive(Y, 2) && Y[2, 1] == 0.0
    snapshot = state(X)
    Y[2, 1] = 9.0
    activate!(Y, 1)
    coordinates(Y)[5] = 3.0
    dropzeros!(Y)
    @test isequal(state(X), snapshot)                         # writes and activations on the copy
    X[3, 2] = -4.0
    @test Y[3, 2] == 3.0 && X[3, 2] == -4.0                   # and on the original
    @test copy(SymmetricSparseIterate(Float32, 0)) isa SymmetricSparseIterate{Float32}
    Z = copy(X)
    @test collect(coordinates(Z)) == collect(coordinates(X))
  end

  @testset "similar is empty and has the same dimension" begin
    X = sample()
    S = similar(X)
    @test S isa SymmetricSparseIterate{Float64} && size(S) == (3, 3)
    @test nactive(S) == 0 && all(iszero, coordinates(S)) && all(iszero, Matrix(S))
    S32 = similar(X, Float32)
    @test S32 isa SymmetricSparseIterate{Float32} && size(S32) == (3, 3) && nactive(S32) == 0
    @test similar(X, Int) isa SymmetricSparseIterate{Int}
    @test_throws ArgumentError("SymmetricSparseIterate requires real entries") similar(X, ComplexF64)
    S[1, 1] = 1.0
    @test X[1, 1] == 0.0                                       # independent
    @test collect(activecoordinates(X)) == [3, 2, 6]
  end

  @testset "copyto! between owners is a state copy" begin
    X = sample()
    Y = SymmetricSparseIterate(Float64, 3)
    Y[1, 1] = 3.0
    activate!(Y, 5)
    @test copyto!(Y, X) === Y
    @test isequal(Matrix(Y), Matrix(X))
    @test collect(activecoordinates(Y)) == [3, 2, 6] && collect(activevalues(Y)) == [5.0, 0.0, 7.0]
    @test !isactive(Y, 1) && !isactive(Y, 5)                   # the previous working set is gone
    @test isactive(Y, 2) && Y[2, 1] == 0.0                     # the explicit zero travels

    # a different dimension rejects before writing
    W = SymmetricSparseIterate(Float64, 4)
    W[2, 2] = 1.0
    rejects(w -> copyto!(w, X), W, DimensionMismatch)
    rejects(y -> copyto!(y, SymmetricSparseIterate(Float64, 2)), Y, DimensionMismatch)

    # a copy of an owner onto itself does nothing
    snapshot = state(X)
    @test copyto!(X, X) === X
    @test isequal(state(X), snapshot)

    # the destination is independent of the source afterwards
    X[1, 1] = 8.0
    @test Y[1, 1] == 0.0
    # onto an owner with a working set of its own, small and large against q
    for p in (3, 20)
      rng = MersenneTwister(5p)
      q = p * (p + 1) ÷ 2
      S = SymmetricSparseIterate(Float64, p)
      for k in randperm(rng, q)[1:3]
        coordinates(S)[k] = randn(rng)
      end
      activate!(S, rand(rng, 1:q))
      for ndest in (0, 2, q)
        D = SymmetricSparseIterate(Float64, p)
        for k in randperm(rng, q)[1:ndest]
          coordinates(D)[k] = randn(rng)
        end
        copyto!(D, S)
        @test isequal(Matrix(D), Matrix(S)) && collect(activecoordinates(D)) == collect(activecoordinates(S))
        @test collect(activevalues(D)) == collect(activevalues(S))
        @test all(k -> isactive(D, k) == isactive(S, k), 1:q)
      end
    end
  end

  @testset "a state copy converts the active values only" begin
    # the source has a stale slot beyond its active set that does not convert to an Int
    p, q = 3, 6
    S = SymmetricSparseIterate{Float64}([2.0, 0.5, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], 1, p)
    @test nactive(S) == 1 && S.nzval[2] == 0.5
    D = SymmetricSparseIterate(Int, p)
    D[3, 3] = 9
    @test copyto!(D, S) === D
    @test Matrix(D) == [2 0 0; 0 0 0; 0 0 0] && collect(activecoordinates(D)) == [1]
    # a value that does not convert: the destination is unchanged, in values, set and hidden state
    T = SymmetricSparseIterate(Float64, p)
    T[2, 1] = 0.5
    T[3, 3] = 3.0
    D = SymmetricSparseIterate(Int, p)
    D[1, 1] = 4
    activate!(D, 5)
    snapshot = state(D)
    @test_throws InexactError copyto!(D, T)
    @test isequal(state(D), snapshot)
    # a mixed element type narrows the active values, and an active value that rounds to zero stays active
    X64 = SymmetricSparseIterate(Float64, p)
    X64[2, 1] = 1e-100
    X64[3, 3] = 2.5
    X32 = SymmetricSparseIterate(Float32, p)
    X32[1, 1] = 1.0
    copyto!(X32, X64)
    @test collect(activecoordinates(X32)) == [2, 6] && X32[2, 1] === 0.0f0 && X32[3, 3] === 2.5f0
    @test !isactive(X32, 1)
  end

  @testset "copyto! from a matrix replaces the working set, in increasing coordinate order" begin
    Y = sample()
    @test isactive(Y, 2) && Y[2, 1] == 0.0                      # the old explicit zero
    M = zeros(3, 3)
    M[3, 1] = M[1, 3] = 2.0                                      # one coordinate only
    @test copyto!(Y, M) === Y
    @test collect(activecoordinates(Y)) == [3] && Matrix(Y) == M
    @test !isactive(Y, 2) && !isactive(Y, 6)                    # the old explicit zero and the old values are gone

    # the working set is the nonzero support, in the order of the coordinates, whatever it held before
    rng = MersenneTwister(10)
    for p in (0, 1, 2, 5), T in (Float32, Float64)
      N = zeros(Float64, p, p)
      for (r, c) in lower_positions(p)
        rand(rng) < 0.5 && continue
        N[r, c] = N[c, r] = rand(rng, (-1.5, 0.25, 4.0, NaN, Inf))
      end
      Z = SymmetricSparseIterate(T, p)
      for k in randperm(rng, p * (p + 1) ÷ 2)                   # an unrelated working set
        activate!(Z, k)
      end
      copyto!(Z, N)
      expected = [k for (k, (r, c)) in enumerate(lower_positions(p)) if !iszero(N[r, c])]
      @test collect(activecoordinates(Z)) == expected
      @test isequal(Matrix(Z), convert(Matrix{T}, N))
    end

    # sources of other kinds
    F = Matrix(Symmetric(randn(rng, 4, 4)))
    for src in (F, view(F, :, :), transpose(F), Symmetric(F, :L), Symmetric(randn(rng, 4, 4), :U), sparse(F),
                transpose(sparse(F)), adjoint(sparse(F)))
      x = destination()
      @test copyto!(x, src) === x
      @test Matrix(x) == Matrix(src)
    end
    # a source of the element type of the owner and of the other types
    for T in (Float32, Int)
      x = copyto!(SymmetricSparseIterate(T, 2), [1.0 2.0; 2.0 3.0])
      @test eltype(x) == T && Matrix(x) == [1 2; 2 3] && collect(activecoordinates(x)) == [1, 2, 3]
    end
  end

  @testset "the constructor from a matrix" begin
    rng = MersenneTwister(11)
    F = Matrix(Symmetric(randn(rng, 4, 4)))
    F[3, 1] = F[1, 3] = 0.0
    X = SymmetricSparseIterate(F)
    @test X isa SymmetricSparseIterate{Float64} && Matrix(X) == F
    @test collect(activecoordinates(X)) == [k for (k, (r, c)) in enumerate(lower_positions(4)) if !iszero(F[r, c])]
    @test !isactive(X, 3)
    @test SymmetricSparseIterate(sparse(F)) == X && Matrix(SymmetricSparseIterate(sparse(F))) == F
    G = randn(rng, 3, 3)
    @test Matrix(SymmetricSparseIterate(Symmetric(G, :L))) == Matrix(Symmetric(G, :L))
    # the view of an owner is a source like any other, and the result is independent of it
    Y = SymmetricSparseIterate(matrixview(X))
    @test Y == X && Y !== X
    Y[1, 1] = 100.0
    @test X[1, 1] == F[1, 1]
    @test_throws ArgumentError SymmetricSparseIterate(randn(rng, 3, 3))
    @test_throws ArgumentError SymmetricSparseIterate(sparse(randn(rng, 3, 3)))
    @test_throws DimensionMismatch SymmetricSparseIterate(zeros(2, 3))
  end

  @testset "a source that shares memory with the owner is refused before anything is written" begin
    X = sample()
    A = matrixview(X)
    for src in (A, view(A, :, :), view(A, 1:3, 1:3), Symmetric(A), transpose(A), A', Symmetric(A, :L), reshape(A, 3, 3))
      rejects(x -> copyto!(x, src), X, ArgumentError, "shares memory")
    end
    # an unrelated copy of the same values is not an alias
    @test copyto!(X, Matrix(X)) === X
    @test collect(activecoordinates(X)) == [3, 6] && X[3, 1] == 5.0
    Y = sample()
    @test copyto!(Y, matrixview(sample())) === Y
    @test collect(activecoordinates(Y)) == [3, 6]
  end

  @testset "shape, dimension and element type are checked before writing" begin
    X = destination()
    F = Matrix(Symmetric(randn(MersenneTwister(12), 4, 4)))
    for src in (F[1:3, 1:3], zeros(4, 3), zeros(5, 5), [F zeros(4); zeros(1, 5)])
      rejects(x -> copyto!(x, src), X, DimensionMismatch)
    end
    rejects(x -> copyto!(x, ones(16)), X, DimensionMismatch)
    for src in (complex.(F), sparse(complex.(F)), Symmetric(complex.(F)), Hermitian(complex.(F)))
      rejects(x -> copyto!(x, src), X, ArgumentError, "requires real entries")
    end
    @test_throws DimensionMismatch SymmetricSparseIterate(zeros(4, 3))
  end

  @testset "the state copy does not change the working set of another owner" begin
    X = sample()
    Y = copy(X)
    Z = SymmetricSparseIterate(Float64, 3)
    copyto!(Z, Y)
    Y[1, 1] = 1.0
    @test collect(activecoordinates(Z)) == [3, 2, 6] && Z[1, 1] == 0.0
    @test collect(activecoordinates(X)) == [3, 2, 6]
  end
end

@testset "Base.mightalias and Base.unalias see through an owner" begin
  X = sample()
  A = matrixview(X)
  views = Any[A, coordinates(X), activevalues(X), activecoordinates(X), view(A, :, :), view(coordinates(X), 1:2),
              Symmetric(A), Symmetric(A, :L), transpose(A), A', reshape(A, 9), vec(A), view(A, 1:2, 1:2),
              view(activevalues(X), 1:2), reshape(coordinates(X), 2, 3), transpose(coordinates(X))]
  for a in views
    @test Base.mightalias(X, a) && Base.mightalias(a, X)
  end
  Y = sample()
  for a in (matrixview(Y), coordinates(Y), activevalues(Y), activecoordinates(Y), Symmetric(matrixview(Y)),
            Matrix(X), zeros(3), zeros(3, 3), view(zeros(3, 3), :, 1))
    @test !Base.mightalias(X, a) && !Base.mightalias(a, X)
  end
  # an empty array shares nothing
  E = SymmetricSparseIterate(Float64, 0)
  @test !Base.mightalias(E, matrixview(E)) && !Base.mightalias(coordinates(E), E)
  # unalias copies a view of the owner, and the copy does not follow later writes
  for f in (matrixview, coordinates, activevalues, activecoordinates)
    Z = sample()
    a = f(Z)
    b = Base.unalias(Z, a)
    @test b !== a && b == a && !Base.mightalias(Z, b)
    before = copy(b)
    Z[3, 1] = -1.0
    Z[2, 1] = 8.0
    @test b == before
    @test Base.unalias(Z, zeros(2)) isa Vector
  end
  M = zeros(3, 3)
  @test Base.unalias(X, M) === M
  # a view of another owner is returned as it is
  W = matrixview(Y)
  @test Base.unalias(X, W) === W
end

@testset "Base.unalias copies wrapped sources of an owner" begin
  wrappers = (
    "Symmetric" => A -> Symmetric(A), "Symmetric :L" => A -> Symmetric(A, :L), "Hermitian" => A -> Hermitian(A),
    "view of Symmetric" => A -> view(Symmetric(A), :, :), "view of Hermitian" => A -> view(Hermitian(A, :L), :, :),
    "Transpose" => A -> transpose(A), "Adjoint" => A -> A', "reshape" => A -> reshape(A, 9),
    "view of view" => A -> view(view(A, :, :), 1:2, 2:3),
    "nested view of Symmetric" => A -> view(view(Symmetric(A), :, :), 1:2, 1:2),
    "reshape of Symmetric" => A -> reshape(Symmetric(A), 9),
    "transpose of Symmetric" => A -> transpose(Symmetric(A)))
  for (name, wrap) in wrappers
    @testset "$name" begin
      Z = sample()
      a = wrap(matrixview(Z))
      @test Base.mightalias(Z, a)
      b = Base.unalias(Z, a)
      @test b !== a && b == a && !Base.mightalias(Z, b)
      a isa Union{Symmetric, Hermitian} && @test typeof(b) === typeof(a) && b.uplo == a.uplo
      before = copy(b)
      Z[3, 1] = -1.0
      Z[2, 1] = 8.0
      @test b == before
      # a source of another owner shares nothing, and is returned as it is
      W = wrap(matrixview(sample()))
      @test Base.unalias(Z, W) === W
      @test Base.unalias(Z, wrap(zeros(3, 3))) isa AbstractArray
    end
  end
  S = Symmetric(zeros(3, 3))
  @test Base.unalias(sample(), S) === S
end

@testset "Base.unalias copies Symmetric and Hermitian wrappers through derived parents" begin
  # a wrapper of a view, a reshape or a transpose of the matrix view, of the coordinates or of the active values,
  # in both nestings (Symmetric(view(A)) and view(Symmetric(A))), in both triangles
  for basekind in (:matrix, :coordinates, :activevalues)
    x = SymmetricSparseIterate(3)
    fill!(coordinates(x), 2.0)
    A = basekind === :matrix ? matrixview(x) :
        reshape(view(basekind === :coordinates ? coordinates(x) : activevalues(x), [1, 2, 2, 4]), 2, 2)
    bases = ("direct" => A, "view" => view(A, :, :), "nestedview" => view(view(A, :, :), :, :),
             "transpose" => Transpose(A), "adjoint" => Adjoint(A), "reshape" => reshape(vec(A), size(A)))
    for (basename, B) in bases
      sources = Any[("plain", B), ("view", view(B, :, :)), ("vec", vec(B)), ("transpose", Transpose(B)),
                    ("adjoint", Adjoint(B))]
      for tri in (:L, :U), F in (Symmetric, Hermitian)
        C = F(B, tri)
        push!(sources, ("$(nameof(F))-$tri", C), ("view-$(nameof(F))-$tri", view(C, :, :)),
              ("nestedview-$(nameof(F))-$tri", view(view(C, :, :), :, :)),
              ("reshape-$(nameof(F))-$tri", reshape(C, length(C))))
      end
      for (wrap, a) in sources
        label = "$basekind/$basename/$wrap"
        saved = copy(a)
        b = try
          Base.unalias(x, a)
        catch e
          e
        end
        @test b isa AbstractArray
        b isa AbstractArray || continue
        @test typeof(b) === typeof(a)
        @test isequal(b, a) && Base.mightalias(x, a) && !Base.mightalias(x, b)
        a isa Union{Symmetric, Hermitian} && @test b.uplo == a.uplo
        fill!(coordinates(x), 3.0)             # a later write of the owner is not seen by the copy
        @test isequal(b, saved)
        fill!(coordinates(x), 2.0)
      end
    end
  end
end

@testset "T06: narrowing and structural equality" begin
  ASYM = ArgumentError
  @testset "pairs equal in Float32 but not in Float64 are rejected before anything changes" begin
    M = Float64[1 1+1e-12; 1 1]
    @test Float32(M[1, 2]) == Float32(M[2, 1]) && M[1, 2] != M[2, 1]
    for T in (Float32, Float64)
      X32 = SymmetricSparseIterate(T, 2)
      X32[1, 1] = 5.0
      activate!(X32, 2)                                         # an explicit zero
      X32[2, 2] = 2.0
      X0 = copy(X32)
      err = try
        copyto!(X32, M)
        nothing
      catch e
        e
      end
      @test err isa ArgumentError && occursin(ASYMMETRIC, sprint(showerror, err))
      @test collect(activecoordinates(X32)) == collect(activecoordinates(X0))
      @test collect(activevalues(X32)) == collect(activevalues(X0))
      @test all(i -> all(j -> X32[i, j] == X0[i, j], 1:2), 1:2)
      @test isequal(state(X32), state(X0))
    end
    # the constructor too
    @test_throws ArgumentError SymmetricSparseIterate(M)
    # a pair that is equal in Float64 imports
    N = Float64[1 1.5; 1.5 1]
    X = SymmetricSparseIterate(Float32, 2)
    @test copyto!(X, N) === X && Matrix(X) == Float32.(N)
  end

  @testset "structural equality: NaN with NaN, +0 with -0, 1 with 1.0" begin
    x = SymmetricSparseIterate(2)
    @test isequal(Matrix(copyto!(x, [NaN NaN; NaN 1])), [NaN NaN; NaN 1])
    @test Matrix(copyto!(x, [1 -0.0; 0.0 1])) == [1 0; 0 1]
    @test !signbit(x[1, 2])                                     # the lower entry decides
    @test Matrix(copyto!(x, [2 1.0; 1 3])) == [2 1; 1 3]       # 1 and 1.0
    rejects(x -> copyto!(x, [1 NaN; 0.5 1]), x, ASYM, ASYMMETRIC)
    rejects(x -> copyto!(x, [1 1.0; NaN 1]), x, ASYM, ASYMMETRIC)
    rejects(x -> copyto!(x, [1 Inf; -Inf 1]), x, ASYM, ASYMMETRIC)
    @test Matrix(copyto!(x, [1 Inf; Inf 1])) == [1 Inf; Inf 1]
    # compared in the type of the source: equal in Float32, not in Float64
    rejects(x -> copyto!(x, [1 1 + 1e-12; 1 1]), SymmetricSparseIterate(Float32, 2), ASYM, ASYMMETRIC)
    # Int sources: an Int pair is symmetric when its entries are equal
    @test Matrix(copyto!(SymmetricSparseIterate(Float32, 2), [1 2; 2 3])) == [1 2; 2 3]
    rejects(x -> copyto!(x, [1 2; 3 4]), SymmetricSparseIterate(Float32, 2), ASYM, ASYMMETRIC)
    # a symmetric matrix that breaks the domain of a function is a fine source: the owner checks structure only
    @test Matrix(copyto!(SymmetricSparseIterate(2), [-1.0 -2.0; -2.0 -3.0])) == [-1 -2; -2 -3]
    # only the checks are atomic: a conversion error can leave x partly written
    @test_throws InexactError copyto!(SymmetricSparseIterate(Int, 2), [1 2; 2 3.5])
  end

  @testset "the asymmetric sources of every kind are rejected, the symmetric ones imported" begin
    rng = MersenneTwister(13)
    E = randn(rng, 4, 4)                                        # not symmetric
    F = E + transpose(E)                                        # exactly symmetric
    for A in (E, view(E, :, :), transpose(E), sparse(E), UpperTriangular(E), transpose(sparse(E)), adjoint(sparse(E)))
      rejects(x -> copyto!(x, A), destination(), ASYM, ASYMMETRIC)
    end
    for A in (Symmetric(E, :L), Symmetric(E, :U), F, view(F, :, :), sparse(F), transpose(F))
      x = destination()
      @test copyto!(x, A) === x
      @test Matrix(x) == Matrix(A)
    end
    rejects(x -> copyto!(x, SparseArrays.CHOLMOD.Dense(E)), destination(), ASYM, ASYMMETRIC)
    x = destination()
    @test copyto!(x, SparseArrays.CHOLMOD.Dense(F)) === x && Matrix(x) == F
    @test_throws ArgumentError SymmetricSparseIterate(E)
    @test_throws ArgumentError SymmetricSparseIterate(sparse(E))
    for (A, B) in ((Symmetric(E, :L), Symmetric(E, :L)), (Symmetric(E, :U), Symmetric(E, :U)),
                   (F, F), (sparse(F), F), (Symmetric(sparse(E), :L), Symmetric(E, :L)))
      @test Matrix(SymmetricSparseIterate(A)) == Matrix(B)
    end
  end

  @testset "the message names the pair with the largest difference" begin
    A = ones(4, 4)
    A[4, 1] = 2
    A[3, 2] = 5
    err = try
      copyto!(SymmetricSparseIterate(4), A)
    catch e
      e
    end
    @test err isa ArgumentError
    @test occursin("entry (3, 2) is 5.0 and entry (2, 3) is 1.0 (relative asymmetry 0.8)", sprint(showerror, err))
    @test occursin("Symmetric(A, :L)", sprint(showerror, err))
  end

  @testset "narrowing and activation on import and on write" begin
    X32 = SymmetricSparseIterate(Float32, 3)
    X32[2, 1] = 1e-100                                          # absent: stays absent
    @test nactive(X32) == 0 && X32[2, 1] === 0.0f0
    X32[2, 1] = 1e-30
    X32[2, 1] = 1e-100                                          # active: stores zero and stays
    @test nactive(X32) == 1 && X32[2, 1] === 0.0f0 && isactive(X32, 2)
    # an import of a Float64 matrix whose off-diagonal pair is 1e-100 has no such coordinate; 1e-30 is active
    for (v, active) in ((1e-100, false), (1e-30, true))
      M = zeros(3, 3)
      M[2, 1] = M[1, 2] = v
      Y = SymmetricSparseIterate(Float32, 3)
      activate!(Y, 2)
      copyto!(Y, M)
      @test isactive(Y, 2) == active && nactive(Y) == (active ? 1 : 0)
      @test Y[2, 1] === (active ? Float32(v) : 0.0f0)
    end
  end
end

end
