module SymmetricWritesTest

using Test, ProximalBase, LinearAlgebra, SparseArrays, Random

Random.seed!(191)

const ASYMMETRIC = "must be symmetric"
const PARTIAL = "cannot write part of a SymmetricSparseIterate"

double(v) = 2v
increment(v) = v + 1
vanish(v) = zero(v)

# everything a failed write must leave unchanged, in storage order
state(x) = (Matrix(x), nnz(x), copy(activecoordinates(x)), x.nzval[1:nnz(x)])

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

# a destination with a stored entry, a stored zero and a diagonal entry
function initial(T = Float64, p = 4)
  x = SymmetricSparseIterate(T, p)
  x[4, 1] = 7
  activate!(x, tril_index(p, 3, 2))
  x[2, 2] = -1
  x
end

function check_map(f::F, x, As::Vararg{AbstractArray, N}) where {F, N}
  snapshots = map(Matrix, As)
  reference = f.(snapshots...)
  @test reference == transpose(reference)
  y = copy(x)
  y .= f.(snapshots...)
  @test (@inferred map!(f, x, As...)) === x
  @test Matrix(x) == reference
  @test nnz(x) == nnz(y)
end

mutable struct CallCounter
  calls::Int
end
function (f::CallCounter)(vs...)
  f.calls += 1
  sum(vs) + 1
end

# Matching sizes need not mean matching axes.
struct ShiftedMatrix{T} <: AbstractMatrix{T}
  data::Matrix{T}
end
Base.size(A::ShiftedMatrix) = size(A.data)
Base.axes(A::ShiftedMatrix) = (0:size(A, 1)-1, 0:size(A, 2)-1)
Base.getindex(A::ShiftedMatrix, r::Int, c::Int) = A.data[r+1, c+1]

# The writes into x, each with its dense reference from the old value X of x
# and a matrix source A, and the error for a source with other axes (copy!
# checks the axes itself).
const WRITERS = (
  ("copyto!", (x, A) -> copyto!(x, A), (X, A) -> A, DimensionMismatch),
  ("copy!", (x, A) -> copy!(x, A), (X, A) -> A, ArgumentError),
  (".=", (x, A) -> (x .= A), (X, A) -> A, DimensionMismatch),
  (".+=", (x, A) -> (x .+= A), (X, A) -> X .+ A, DimensionMismatch),
  ("map!, one source", (x, A) -> map!(double, x, A), (X, A) -> 2 .* A, DimensionMismatch),
  ("map!, two sources", (x, A) -> map!(-, x, A, Symmetric(ones(4, 4))), (X, A) -> A .- 1, DimensionMismatch),
  ("map!, three sources", (x, A) -> map!(+, x, x, A, A), (X, A) -> X .+ A .+ A, DimensionMismatch),
  ("map!, four sources", (x, A) -> map!(+, x, A, x, x, A), (X, A) -> A .+ X .+ X .+ A, DimensionMismatch),
  ("axpy!", (x, A) -> axpy!(2, A, x), (X, A) -> X .+ 2 .* A, DimensionMismatch),
  ("axpby!", (x, A) -> axpby!(2, A, 3, x), (X, A) -> 2 .* A .+ 3 .* X, DimensionMismatch),
)

# Warm the exact measured call.
allocations(f::F, x::X) where {F, X} = (f(x); @allocated f(x))
allocations(f::F, x::X, A::TA) where {F, X, TA} = (f(x, A); @allocated f(x, A))
allocations(f::F, x::X, A::TA, B::TB) where {F, X, TA, TB} = (f(x, A, B); @allocated f(x, A, B))
allocations(f::F, x::X, A::TA, B::TB, C::TC) where {F, X, TA, TB, TC} =
  (f(x, A, B, C); @allocated f(x, A, B, C))
scale!(x) = (x .= 2 .* x)
shift!(x, A) = (x .= x .- A ./ 2)
outer!(x, v) = (x .= v .* v')
copy_into!(x, A) = copyto!(x, A)
map1!(x, A) = map!(double, x, A)
map2!(x, A, B) = map!(+, x, A, B)
map3!(x, A, B, C) = map!(+, x, A, B, C)
axpy2!(x, A) = axpy!(2.0, A, x)
axpby23!(x, A) = axpby!(2.0, A, 3.0, x)

@testset "Writes into a SymmetricSparseIterate" begin
  @testset "every entry point, $T" for T in (Float64, ComplexF64)
    E = randn(T, 4, 4)                     # not symmetric
    F = E + transpose(E)                   # exactly symmetric
    # every source type through copyto!, the path of matrix sources, and
    # representative ones through the other writers
    writers = T === Float64 ? WRITERS : WRITERS[[1, 3, 5, 10]]
    for (name, write!, reference, axes_error) in writers
      @testset "$name" begin
        all_sources = name == "copyto!"
        for A in (all_sources ? (E, view(E, :, :), transpose(E), sparse(E), UpperTriangular(E),
                                 transpose(sparse(E)), adjoint(sparse(E))) : (E, sparse(E)))
          rejects(x -> write!(x, A), initial(T), ArgumentError, ASYMMETRIC)
        end
        for A in (all_sources ? (Symmetric(E, :L), Symmetric(E, :U), F, view(F, :, :), sparse(F), transpose(F)) :
                                (Symmetric(E, :L), F, sparse(F)))
          x = initial(T)
          X = Matrix(x)
          @test write!(x, A) === x
          @test Matrix(x) == reference(X, Matrix(A))
        end
        for A in (vec(F), F[1:3, 1:3], [F zeros(T, 4); zeros(T, 1, 5)], ShiftedMatrix(F))
          rejects(x -> write!(x, A), initial(T), axes_error)
        end
      end
    end
    @test_throws ArgumentError SymmetricSparseIterate(E)
    @test_throws ArgumentError SymmetricSparseIterate(sparse(E))
    @test_throws DimensionMismatch SymmetricSparseIterate(ShiftedMatrix(F))
    for (A, B) in ((Symmetric(E, :L), Symmetric(E, :L)), (Symmetric(E, :U), Symmetric(E, :U)),
                   (F, F), (sparse(F), F), (Symmetric(sparse(E), :L), Symmetric(E, :L)))
      @test Matrix(SymmetricSparseIterate(A)) == Matrix(B)
    end
    rejects(x -> copyto!(x, SparseArrays.CHOLMOD.Dense(E)), initial(T), ArgumentError, ASYMMETRIC)
    x = initial(T)
    @test copyto!(x, SparseArrays.CHOLMOD.Dense(F)) === x && Matrix(x) == F
  end

  @testset "the message names the first pair" begin
    # (4, 1) comes first by columns, (3, 2) by rows
    A = ones(4, 4)
    A[4, 1] = 2
    A[3, 2] = 5
    err = try
      copyto!(SymmetricSparseIterate(4), A)
    catch e
      e
    end
    @test err isa ArgumentError
    @test occursin("entry (4, 1) is 2.0 and entry (1, 4) is 1.0", sprint(showerror, err))
    @test occursin("Symmetric(A, :L)", sprint(showerror, err))
  end

  @testset "values, not inputs" begin
    A = randn(4, 4)
    v, w = randn(4), randn(4)
    x = initial()
    @test (x .= A .+ A') === x && Matrix(x) == A + A'
    x .= v .* v'
    @test Matrix(x) == v * v'
    @test Matrix(map!(+, x, A, transpose(A))) == A + transpose(A)
    rejects(x -> (x .= v), initial(), ArgumentError, ASYMMETRIC)
    rejects(x -> (x .= v .* w'), initial(), ArgumentError, ASYMMETRIC)
    rejects(x -> (x .= x .+ v .* w'), initial(), ArgumentError, ASYMMETRIC)
    # -0.0 and 0.0 give one entry, but 1 ./ A tells them apart
    rejects(x -> (x .= 1 ./ [1 -0.0; 0.0 1]), SymmetricSparseIterate(2), ArgumentError, ASYMMETRIC)
    # leaves of every kind
    M = randn(4, 4); M = M + M'
    S = SymmetricSparseIterate(M)
    x = initial()
    @test Matrix(x .= Ref(2.0) .* S) == 2M
    @test Matrix(x .= fill(2.0) .* S) == 2M
    @test Matrix(x .= Diagonal(v) .+ S) == Diagonal(v) + M
    @test Matrix(x .= SymTridiagonal(v, w[1:3]) .- S) == SymTridiagonal(v, w[1:3]) - M
    @test Matrix(x .= sparse(M) .* 3) == 3M
    @test Matrix(x .= (1.0, 1.0, 1.0, 1.0) .+ M) == M .+ 1
    rejects(x -> (x .= (1.0, 2.0, 3.0, 4.0) .+ M), initial(), ArgumentError, ASYMMETRIC)
    rejects(x -> (x .= Bidiagonal(v, w[1:3], :U)), initial(), ArgumentError, ASYMMETRIC)
    rejects(x -> (x .= UpperTriangular(M)), initial(), ArgumentError, ASYMMETRIC)
    C = randn(ComplexF64, 4, 4); C = C + transpose(C)
    xc = SymmetricSparseIterate(ComplexF64, 4)
    @test Matrix(xc .= SymmetricSparseIterate(C)') == conj(C)
    @test Matrix(xc .= Symmetric(C)') == conj(C)
    rejects(x -> copyto!(x, Hermitian(C)), xc, ArgumentError, ASYMMETRIC)
    @test Matrix(copyto!(x, Hermitian(M))) == M
  end

  @testset "equality and conversion" begin
    x = SymmetricSparseIterate(2)
    @test isequal(Matrix(copyto!(x, [NaN NaN; NaN 1])), [NaN NaN; NaN 1])
    @test Matrix(copyto!(x, [1 -0.0; 0.0 1])) == [1 0; 0 1]
    @test !signbit(x[1, 2])                # the lower entry is stored
    rejects(x -> copyto!(x, [1 NaN; 0.5 1]), x, ArgumentError, ASYMMETRIC)
    # compared in the type of the source: equal in Float32, not in Float64
    rejects(x -> copyto!(x, [1 1 + 1e-12; 1 1]), SymmetricSparseIterate(Float32, 2), ArgumentError, ASYMMETRIC)
    for T in (ComplexF64, Float32, Int)
      x = copyto!(SymmetricSparseIterate(T, 2), [1.0 2.0; 2.0 3.0])
      @test eltype(x) == T && Matrix(x) == [1 2; 2 3]
    end
    # Only the checks are atomic: a conversion error can leave x partly
    # written, as in Base.
    @test_throws InexactError copyto!(SymmetricSparseIterate(Int, 2), [1 2; 2 3.5])
  end

  @testset "partial writes" begin
    A = [1.0 2.0; 2.0 4.0]
    CI = CartesianIndices((1:2, 1:2))
    for f in (x -> copyto!(x, 1, vec(A)), x -> copyto!(x, 1, vec(A), 1),
              x -> copyto!(x, 1, vec(A), 1, 4), x -> copyto!(x, CI, A, CI),
              x -> copyto!(x, CI, sparse(A), CI), x -> copyto!(x, 1:2, 1:2, A, 1:2, 1:2),
              x -> axpy!(1.0, [1.0, 2.0], [1, 2], x, [2, 3]))
      rejects(f, fill!(SymmetricSparseIterate(2), -1), ArgumentError, PARTIAL)
    end
    @test Matrix(copyto!(SymmetricSparseIterate(2), 2I)) == [2 0; 0 2]
  end

  @testset "#191: map! applies f once, $T" for T in (Float64, ComplexF64)
    x = SymmetricSparseIterate(Symmetric(randn(T, 4, 4), :L))
    check_map(double, x, x)
    check_map(+, x, x, x)
    before = Matrix(x)
    if VERSION >= v"1.12"
      @test map!(double, x) === x
      @test Matrix(x) == 2 .* before
    else
      @test_throws ArgumentError map!(double, x)
      @test Matrix(x) == before
    end
  end

  @testset "calls per entry, p = $p" for p in (0, 1, 4)
    for n in 1:4
      # symmetric by type, or scanned and found symmetric: once per entry
      for sources in (ntuple(_ -> SymmetricSparseIterate(p), n), ntuple(_ -> zeros(p, p), n))
        x = SymmetricSparseIterate(p)
        f = CallCounter(0)
        @test map!(f, x, sources...) === x
        @test f.calls == p * (p + 1) ÷ 2
        @test Matrix(x) == fill(1, p, p) && nnz(x) == p * (p + 1) ÷ 2
      end
    end
    # a value checked pair by pair: both positions first, then the lower ones
    A = randn(p, p)
    x = SymmetricSparseIterate(p)
    f = CallCounter(0)
    @test map!(f, x, A, transpose(A)) === x
    @test f.calls == p * (p - 1) + p * (p + 1) ÷ 2
  end

  @testset "sources" begin
    M = randn(4, 4); M = M + M'
    M[4, 1] = M[1, 4] = 0
    perm = [4, 2, 3, 1]
    for A in (SymmetricSparseIterate(M), M, sparse(M), view(M, perm, perm), view(M, :, :), Symmetric(M))
      check_map(double, initial(), A)
      check_map(+, initial(), A, M)
      check_map(+, initial(), A, sparse(M), A)
    end
    check_map(identity, SymmetricSparseIterate(ComplexF64, 4), M)
  end

  @testset "aliased sources" begin
    perm = [4, 2, 1, 3]
    x = SymmetricSparseIterate(Symmetric(randn(4, 4), :L))
    check_map(double, x, view(x, perm, perm))
    check_map(+, x, x, view(x, perm, perm))
    rejects(x -> map!(double, x, view(x, perm, :)), x, ArgumentError, ASYMMETRIC)
    rejects(x -> map!(double, x, reshape(x, 2, 8)), x, DimensionMismatch)
  end

  @testset "stored zeros" begin
    x = SymmetricSparseIterate(4)
    x[4, 1] = 3
    check_map(vanish, x, x)
    @test nnz(x) == 1
    @test isactive(x, tril_index(4, 4, 1)) && !isactive(x, tril_index(4, 3, 2))
    x = SymmetricSparseIterate(4)
    check_map(increment, x, zeros(4, 4))
    @test nnz(x) == 10
  end

  @testset "#236: aliased writers, $T" for T in (Float64, ComplexF64)
    writers = (
      ("copyto!", (x, A, B) -> copyto!(x, A), (X, A, B) -> A),
      ("copy!", (x, A, B) -> copy!(x, A), (X, A, B) -> A),
      (".=", (x, A, B) -> (x .= A), (X, A, B) -> A),
      ("identity broadcast", (x, A, B) -> (x .= identity.(A)), (X, A, B) -> A),
      (".+=", (x, A, B) -> (x .+= A), (X, A, B) -> X .+ A),
      ("map!, one source", (x, A, B) -> map!(double, x, A), (X, A, B) -> 2 .* A),
      ("map!, alias first", (x, A, B) -> map!(-, x, A, B), (X, A, B) -> A .- B),
      ("map!, alias second", (x, A, B) -> map!(-, x, B, A), (X, A, B) -> B .- A),
      ("map!, repeated alias", (x, A, B) -> map!(+, x, A, A), (X, A, B) -> A .+ A),
      ("map!, four sources", (x, A, B) -> map!(+, x, A, x, B, A),
        (X, A, B) -> A .+ X .+ B .+ A),
      ("map!, alias in middle", (x, A, B) -> map!(+, x, x, A, A, B),
        (X, A, B) -> X .+ A .+ A .+ B),
      ("axpy!", (x, A, B) -> axpy!(1, A, x), (X, A, B) -> X .+ A),
      ("axpby!", (x, A, B) -> axpby!(1, A, 2, x), (X, A, B) -> A .+ 2 .* X),
    )
    idx = [tril_index(3, max(4 - r, 4 - c), min(4 - r, 4 - c)) for c in 1:3 for r in 1:3]
    sources = (
      (2, x -> reshape(view(coordinates(x), [3, 2, 2, 1]), 2, 2)),
      (3, x -> reshape(view(coordinates(x), idx), 3, 3)),
      (2, x -> view(x, [2, 1], [2, 1])),
      (3, x -> view(x, 3:-1:1, 3:-1:1)),
    )
    for (p, source) in sources, (name, write!, reference) in writers
      @testset "$name, p = $p, $(typeof(source))" begin
        values = T.(1:p*(p+1)÷2)
        T === ComplexF64 && (values .*= 1 + 2im)
        x = SymmetricSparseIterate(T, p)
        coordinates(x) .= values
        A = source(x)
        B = ones(T, p, p)
        snapshots = map(Matrix, (x, A, B))
        expected = reference(snapshots...)
        @test write!(x, A, B) === x
        @test Matrix(x) == expected
      end
    end
  end

  @testset "#236: aliasing hooks, $T" for T in (Float64, ComplexF64)
    x = initial(T)
    c = coordinates(x)
    v = view(c, [10, 6, 6, 1])
    A = reshape(v, 2, 2)
    other = initial(T)
    @test all(id -> id in Base.dataids(v), Base.dataids(x))
    for source in (c, v, A, view(x, [4, 1], [4, 1]))
      @test Base.mightalias(x, source)
    end
    @test !Base.mightalias(x, reshape(view(coordinates(other), [10, 6, 6, 1]), 2, 2))
    copies = map(Base.unaliascopy, (c, v, A))
    snapshots = map(collect, (c, v, A))
    for (source, copied, snapshot) in zip((c, v, A), copies, snapshots)
      @test typeof(copied) === typeof(source)
      @test axes(copied) == axes(source)
      @test copied == snapshot
      @test !Base.mightalias(x, copied)
    end
    # copying coordinates preserves the active stored zero as well
    @test state(copies[1].x) == state(x)
    @test isactive(copies[1].x, tril_index(4, 3, 2))
    @test copies[1].x[3, 2] == 0
    fill!(x, 19)
    for (copied, snapshot) in zip(copies, snapshots)
      @test copied == snapshot
    end
  end

  @testset "#236: rejected aliased sources, $T" for T in (Float64, ComplexF64)
    for write! in ((x, A) -> copyto!(x, A), (x, A) -> (x .= A),
                   (x, A) -> map!(identity, x, A), (x, A) -> axpy!(1, A, x),
                   (x, A) -> axpby!(1, A, 2, x))
      x = SymmetricSparseIterate(T.([1 2; 2 3]))
      A = reshape(view(coordinates(x), [1, 2, 1, 3]), 2, 2)
      rejects(x -> write!(x, A), x, ArgumentError, ASYMMETRIC)
      A = reshape(view(coordinates(x), [3, 2, 2, 1]), 1, 4)
      rejects(x -> write!(x, A), x, DimensionMismatch)
    end
  end

  @testset "#236: nonaliased coordinates allocations, p = $p" for p in (2, 4)
    other = SymmetricSparseIterate(Symmetric(reshape(Float64.(1:p*p), p, p), :L))
    perm = p:-1:1
    indices = [tril_index(p, max(r, c), min(r, c)) for c in perm for r in perm]
    A = reshape(view(coordinates(other), indices), p, p)
    x = copy(other)
    @test !Base.mightalias(x, A)
    @test allocations(copy_into!, x, A) == 0
    @test allocations((x, A) -> (x .= A), x, A) == 0
    @test allocations(map1!, x, A) == 0
    @test allocations(axpy2!, x, A) == 0
    @test allocations(axpby23!, x, A) == 0
  end

  @testset "allocations" begin
    M = randn(4, 4); M = M + M'
    S = SymmetricSparseIterate(M)
    Sp = sparse(M)
    V = view(M, [4, 2, 3, 1], [4, 2, 3, 1])
    v = randn(4)
    x = copy(S)
    @test allocations(scale!, x) == 0
    @test allocations(shift!, x, M) == 0
    @test allocations(outer!, x, v) == 0
    for A in (S, M, Sp, V, view(M, :, :), Symmetric(M))
      @test allocations(copy_into!, x, A) == 0
      @test allocations(map1!, x, A) == 0
      @test allocations(map2!, x, A, M) == 0
      @test allocations(map3!, x, A, M, Sp) == 0
      @test allocations(axpy2!, x, A) == 0
      @test allocations(axpby23!, x, A) == 0
    end
  end
end

end
