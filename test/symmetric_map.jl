module SymmetricMapTest

using Test, ProximalBase, LinearAlgebra, SparseArrays, Random

Random.seed!(191)

double(v) = 2v
increment(v) = v + 1
vanish(v) = zero(v)
snapshot(A, x) = reshape(A isa AbstractMatrix ? Matrix(A) : collect(A), size(x))

function check_map(f::F, x, As::Vararg{AbstractArray, N}) where {F, N}
  snapshots = map(A -> snapshot(A, x), As)
  reference = Matrix(Symmetric(f.(snapshots...), :L))
  y = copy(x)
  y .= f.(snapshots...)
  @test (@inferred map!(f, x, As...)) === x
  @test Matrix(x) == reference
  @test Matrix(x) == Matrix(y)
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

# Warm the exact measured call, including its out-of-line entry.
map_allocations(f::F, x::X, A::TA) where {F, X, TA} =
  (@allocated map!(f, x, A); @allocated map!(f, x, A))
map_allocations(f::F, x::X, A::TA, B::TB) where {F, X, TA, TB} =
  (@allocated map!(f, x, A, B); @allocated map!(f, x, A, B))
map_allocations(f::F, x::X, A::TA, B::TB, C::TC) where {F, X, TA, TB, TC} =
  (@allocated map!(f, x, A, B, C); @allocated map!(f, x, A, B, C))

@testset "Symmetric map!" begin
  @testset "in place, $T" for T in (Float64, ComplexF64)
    x = SymmetricSparseIterate(randn(T, 4, 4))
    check_map(double, x, x)
    check_map(+, x, x, x)
    if VERSION >= v"1.12"
      before = snapshot(x, x)
      reference = Matrix(Symmetric(double.(before), :L))
      @test map!(double, x) === x
      @test Matrix(x) == reference
    end
  end

  @testset "each entry once, p = $p" for p in (0, 1, 4)
    for n in 1:4
      x = SymmetricSparseIterate(p)
      sources = ntuple(_ -> SymmetricSparseIterate(p), n)
      snapshots = map(A -> snapshot(A, x), sources)
      reference = Matrix(Symmetric((+).(snapshots...) .+ 1, :L))
      f = CallCounter(0)
      @test map!(f, x, sources...) === x
      @test f.calls == p * (p + 1) ÷ 2
      @test Matrix(x) == reference
      @test nnz(x) == p * (p + 1) ÷ 2
    end
  end

  @testset "source shapes and storage" begin
    dense = randn(4, 4)
    dense[4, 1] = 0
    symmetric = SymmetricSparseIterate(dense)
    dense_symmetric = Matrix(Symmetric(dense, :L))
    sparse_source = sparse(dense)
    wrapped = view(dense, [4, 2, 3, 1], :)
    contiguous = view(dense, :, :)
    rectangular = reshape(copy(dense), 2, 8)
    vector = vec(copy(dense))
    initial = SymmetricSparseIterate(4)
    initial[4, 1] = 7
    activate!(initial, tril_index(4, 3, 2))
    for A in (symmetric, dense_symmetric, dense, sparse_source, wrapped, contiguous, rectangular, vector)
      check_map(double, copy(initial), A)
    end
    check_map(+, copy(initial), dense, symmetric)
    check_map(+, copy(initial), dense, sparse_source, vector)
    check_map(+, copy(initial), wrapped, rectangular, vector, symmetric)
    check_map(identity, SymmetricSparseIterate(ComplexF64, 4), dense)

    x = copy(initial)
    for A in (dense, dense_symmetric, symmetric, sparse_source, wrapped, contiguous)
      @test map_allocations(double, x, A) == 0
      @test map_allocations(+, x, A, dense) == 0
      @test map_allocations(+, x, A, dense, dense) == 0
    end
    for A in (vector, rectangular)
      bytes = (map_allocations(double, x, A), map_allocations(+, x, A, dense),
               map_allocations(+, x, A, dense, dense))
      @info "Reshaped-source map! allocations (one, two, three sources)" size = size(A) bytes
    end
  end

  @testset "aliased views" begin
    perm = [4, 2, 1, 3]
    x = SymmetricSparseIterate(randn(4, 4))
    check_map(double, x, view(x, perm, :))
    check_map(+, x, x, view(x, perm, :))
    check_map(double, x, reshape(x, 2, 8))
    check_map(+, x, x, reshape(x, 2, 8))
    check_map(+, x, x, reshape(x, 2, 8), x)
    check_map(double, x, reshape(view(x, perm, :), 2, 8))
    check_map(+, x, x, reshape(view(x, perm, :), 2, 8), x, x)
  end

  @testset "stored zeros" begin
    x = SymmetricSparseIterate(4)
    x[4, 1] = 3
    stored = nnz(x)
    check_map(vanish, x, x)
    @test nnz(x) == stored == 1
    @test isactive(x, tril_index(4, 4, 1))
    @test !isactive(x, tril_index(4, 3, 2))
    x = SymmetricSparseIterate(4)
    check_map(increment, x, zeros(4, 4))
    @test nnz(x) == 10
  end

  @testset "Base iteration rules" begin
    x = fill!(SymmetricSparseIterate(3), -1)
    source = [10., 20.]
    before = copy(source)
    reference = snapshot(x, x)
    reference[1, 1] = before[1]
    reference[2, 1] = reference[1, 2] = before[2]
    @test map!(identity, x, source) === x
    @test Matrix(x) == reference == [10 20 -1; 20 -1 -1; -1 -1 -1]

    x = fill!(SymmetricSparseIterate(2), -1)
    before = snapshot(x, x)
    @test map!(identity, x, Float64[]) === x
    @test Matrix(x) == before

    x = fill!(SymmetricSparseIterate(2), -1)
    source = collect(1.0:9.0)
    before = reshape(copy(source[1:length(x)]), size(x))
    reference = Matrix(Symmetric(before, :U))
    @test map!(identity, x, source) === x
    @test Matrix(x) == reference == [1 3; 3 4]

    # Check the fallback for every arity, including a later short source.
    for sources in ((ones(4), [10., 20.]),
                    (ones(4), ones(4), [10., 20.]),
                    (ones(4), ones(4), ones(4), [10., 20.]))
      x = fill!(SymmetricSparseIterate(2), -1)
      snapshots = map(collect, sources)
      reference = snapshot(x, x)
      prefix = (+).(map(A -> A[1:2], snapshots)...)
      reference[1, 1] = prefix[1]
      reference[2, 1] = reference[1, 2] = prefix[2]
      @test map!(+, x, sources...) === x
      @test Matrix(x) == reference
    end

    source = ShiftedMatrix(reshape(collect(1.0:4.0), 2, 2))
    before = [source[r, c] for r in axes(source, 1), c in axes(source, 2)]
    for sources in ((source,), (zeros(2, 2), source),
                    (zeros(2, 2), zeros(2, 2), source),
                    (zeros(2, 2), zeros(2, 2), zeros(2, 2), source))
      x = fill!(SymmetricSparseIterate(2), -1)
      reference = Matrix(Symmetric(before, :U))
      @test map!(+, x, sources...) === x
      @test Matrix(x) == reference
    end
  end
end

end
