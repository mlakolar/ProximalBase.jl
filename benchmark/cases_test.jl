# Tests of the sparse operands of the `products` cases: each holds a fixed number of nonzeros
# at positions drawn from its own scenario subkey, the cases use exactly these operands, the
# dense operands, case IDs and `work` are as expected, and every reference check passes for
# repetitions 1-10.
#
#   julia --project=<benchmark environment> benchmark/cases_test.jl
#
# benchmark/check_only_test.jl includes this file, so CI runs it with the driver tests.
isdefined(@__MODULE__, :product_cases!) || include("run.jl")
using Test

# Nonzeros of a sparse product operand with N entries at a density, (N, density) => count.
# Written out rather than computed, so the expectations do not depend on the code under test.
const EXPECTED_NNZ = Dict((256, 0.01) => 3, (256, 0.2) => 51, (1024, 0.01) => 10, (1024, 0.2) => 205,
                          (4096, 0.01) => 41, (4096, 0.2) => 819, (512, 0.01) => 5, (512, 0.2) => 102,
                          (4096, 0.02) => 82, (4096, 0.05) => 205, (4096, 0.1) => 410)
# SHA-1 of the sorted case IDs of the products group (joined with newlines), computed at the
# PB-only storage rename (strided_block replaces the old storage label):
# still 108 extended IDs and 10 quick IDs, with identical operation coverage.
const EXTENDED_IDS_SHA1 = "4e12ee82880bc8d6cc81443cb92cb1e7e31f4572"
const QUICK_IDS_SHA1 = "b9832c414e3ed5e4414695682f85f730d7145ec6"

case_type(meta) = meta.dtype == "Float32" ? Float32 : Float64
case_density(meta) = parse(Float64, split(meta.layout, '_')[2])
case_dims(meta) = parse.(Int, split(meta.shape, 'x'))
case_kind(meta) = split(meta.name, '/')[1]

# The `work` of a products case: the nonzeros of its operand times the rows it multiplies.
function expected_work(meta)
    d, dims, kind = case_density(meta), case_dims(meta), case_kind(meta)
    k(N) = EXPECTED_NNZ[(N, d)]
    kind in ("symmetric_U", "symmetric_L") && return k(dims[1])
    kind in ("strided_vector", "strided_block") && return dims[1] * k(dims[2])
    kind == "strided_matrix" && return dims[1] * k(dims[2] * dims[3])
    kind == "transposed_vector" && return dims[2] * k(dims[1])
    kind == "switch_matrix" && return dims[1] * k(dims[2] * dims[3])
    error("unexpected products case $(meta.id)")
end

# A variable captured by the operation of case c: the closure's field `name`, or the one field
# that lowering renamed to `#<number>#name` (the destructured loop variable `x` of the strided
# cases is captured under such a name).
function captured(c, name::Symbol)
    op = c.operation
    fields = filter(f -> f === name || endswith(string(f), "#$name"), collect(fieldnames(typeof(op))))
    length(fields) == 1 || error("case $(c.meta.id): its operation captures no single variable `$name`")
    v = getfield(op, only(fields))
    v isa Core.Box && error("case $(c.meta.id): the captured variable `$name` is boxed")
    v
end

# The dense operand of case c as captured, and the first draw of its scenario RNG.
function dense_operand(c, rep)
    meta = c.meta
    T, d, dims, kind = case_type(meta), case_density(meta), case_dims(meta), case_kind(meta)
    if kind in ("symmetric_U", "symmetric_L")
        triangle = Symbol(kind[end])
        A = captured(c, :A)
        expected = randn(scenario_rng(rep, "products/$T/$(dims[1])/$d/$triangle"), T, dims[1], dims[1])
        return (parent(A), A.uplo), (expected, string(triangle)[1])
    elseif kind == "switch_matrix"
        expected = randn(scenario_rng(rep, "products/switch/$T/$(dims[1])/$d"), T, dims[1], dims[2])
        return captured(c, :A), expected
    else
        B = captured(c, :B)
        expected = randn(scenario_rng(rep, "products/strided/$T/$(dims[2])/$d"), T, dims[1], dims[2])
        return (kind == "transposed_vector" ? parent(B) : B), expected
    end
end

# Stored positions as linear indices, in storage order.
positions(S::SparseVector) = copy(S.nzind)
function positions(S::SparseMatrixCSC)
    I, J, _ = findnz(S)
    (J .- 1) .* size(S, 1) .+ I
end

@testset "fixed_count and sprandn_fixed" begin
    @test fixed_count(256, 0.01) == 3
    @test fixed_count(4096, 0.02) == 82
    @test fixed_count(10, 0.01) == 1
    @test fixed_count(10, 1) == 10
    @test fixed_count(1, 1.0) == 1
    for (N, d) in ((10, 0.0), (10, -0.1), (10, 1.5), (10, NaN), (0, 0.5), (-1, 0.5))
        @test_throws ArgumentError fixed_count(N, d)
    end
    @test_throws ArgumentError sprandn_fixed(Xoshiro(1), Float64, 10, 0.0)
    @test_throws ArgumentError sprandn_fixed(Xoshiro(1), Float64, 4, 3, 1.5)
    @test_throws ArgumentError sprandn_fixed(Xoshiro(1), Float64, 0, 0.5)
    shapes = (((256,), 0.01, 3), ((256,), 0.2, 51), ((1024, 4), 0.01, 41), ((1024, 4), 0.2, 819),
              ((1,), 1.0, 1), ((1, 1), 1.0, 1), ((5,), 1.0, 5))
    for T in (Float32, Float64), (dims, d, k) in shapes
        key = "cases_test/$T/$dims/$d"
        draws = [sprandn_fixed(scenario_rng(rep, key), T, dims..., d) for rep in 1:10]
        again = sprandn_fixed(scenario_rng(1, key), T, dims..., d)
        @test again == draws[1] && findnz(again) == findnz(draws[1])
        for S in draws
            @test S isa (length(dims) == 1 ? SparseVector{T,Int} : SparseMatrixCSC{T,Int})
            @test size(S) == dims
            @test nnz(S) == k && all(!iszero, nonzeros(S))
            @test all(>(0), diff(positions(S)))      # distinct and sorted
        end
        # Different repetitions give different positions, unless only one pattern exists.
        k < prod(dims) && @test allunique(positions.(draws))
    end
end

@testset "product operands with the case subkeys" begin
    n = 512
    operands = Tuple{String,DataType,Tuple,Float64,Int}[]    # subkey, eltype, size, density, nonzeros
    for T in (Float32, Float64), p in (256, 1024), d in (0.01, 0.2)
        for triangle in (:U, :L)
            push!(operands, ("products/$T/$p/$d/$triangle/xs", T, (p,), d, EXPECTED_NNZ[(p, d)]))
        end
        key = "products/strided/$T/$p/$d"
        push!(operands, (key * "/xs", T, (p,), d, EXPECTED_NNZ[(p, d)]))
        push!(operands, (key * "/xm", T, (p, 4), d, EXPECTED_NNZ[(4p, d)]))
        push!(operands, (key * "/xt", T, (n,), d, EXPECTED_NNZ[(n, d)]))
    end
    for (key, T, dims, d, k) in operands
        draws = [product_operand(rep, key, T, dims..., d) for rep in 1:10]
        @test all(rep -> product_operand(rep, key, T, dims..., d) == draws[rep], 1:10)
        @test all(S -> S isa (length(dims) == 1 ? SparseVector{T,Int} : SparseMatrixCSC{T,Int}), draws)
        @test all(S -> size(S) == dims && nnz(S) == k && all(!iszero, nonzeros(S)), draws)
        @test allunique(positions.(draws))
    end
    for T in (Float32, Float64), m in (64, 2048), d in (0.02, 0.05, 0.1)
        orders = Vector{Int}[]
        for rep in 1:10
            x, S = switch_operand(rep, T, m, d)
            x2, S2 = switch_operand(rep, T, m, d)
            order = collect(activecoordinates(x))
            @test S == S2 && order == collect(activecoordinates(x2))
            @test S isa SparseMatrixCSC{T,Int} && size(S) == (1024, 4)
            @test nnz(S) == EXPECTED_NNZ[(4096, d)] && all(!iszero, nonzeros(S))
            @test Array(x) == Array(S) && nactive(x) == nnz(S)
            # the five-argument product takes the stored-entry path exactly below density 1/32
            @test (32 * nactive(x) <= length(x)) == (d < 1 / 32)
            @test !issorted(order)                   # stored in random order
            push!(orders, order)
        end
        @test allunique(sort.(orders))               # positions differ across repetitions
    end
    # The cases use exactly these operands (and the stored order of the switch operand).
    for rep in 1:10
        cases = PerfCase[]
        product_cases!(cases, rep; quick=false)
        for c in cases
            meta = c.meta
            T, d, dims, kind = case_type(meta), case_density(meta), case_dims(meta), case_kind(meta)
            x = captured(c, :x)
            if kind in ("symmetric_U", "symmetric_L")
                S = product_operand(rep, "products/$T/$(dims[1])/$d/$(kind[end])/xs", T, dims[1], d)
            elseif kind == "switch_matrix"
                xs, S = switch_operand(rep, T, dims[1], d)
                @test (meta.id, collect(activecoordinates(x))) == (meta.id, collect(activecoordinates(xs)))
            else
                key = "products/strided/$T/$(dims[2])/$d"
                S = kind == "strided_matrix" ? product_operand(rep, key * "/xm", T, dims[2], 4, d) :
                    kind == "transposed_vector" ? product_operand(rep, key * "/xt", T, dims[1], d) :
                    product_operand(rep, key * "/xs", T, dims[2], d)
            end
            @test (meta.id, Array(x) == Array(S)) == (meta.id, true)
        end
        cases = nothing
    end
end

@testset "product cases, repetitions 1-10" begin
    for rep in 1:10
        cases = PerfCase[]
        product_cases!(cases, rep; quick=false)
        ids = sort([c.meta.id for c in cases])
        @test length(ids) == 108
        @test bytes2hex(sha1(join(ids, "\n"))) == EXTENDED_IDS_SHA1
        work = Dict(c.meta.id => c.meta.work for c in cases)
        # Regressions: these operands had no nonzero in some repetitions.
        @test work["products/symmetric_L/Float32/256/sparse_0.01/warm"] == 3
        @test work["products/strided_vector/mul!/Float64/512x256/sparse_0.01/warm"] == 3 * 512
        for c in cases
            @test (c.meta.id, c.meta.work) == (c.meta.id, expected_work(c.meta))
            @test (c.meta.id, (c.validate(); true)) == (c.meta.id, true)
            captured_operand, expected = dense_operand(c, rep)
            @test (c.meta.id, isequal(captured_operand, expected)) == (c.meta.id, true)
        end
        cases = nothing
        quick = PerfCase[]
        product_cases!(quick, rep; quick=true)
        ids = sort([c.meta.id for c in quick])
        @test length(ids) == 10
        @test bytes2hex(sha1(join(ids, "\n"))) == QUICK_IDS_SHA1
        @test all(c -> c.meta.work == expected_work(c.meta), quick)
        quick = nothing
    end
end

# ---- the helpers of the symmetric owner (a real owner, not a stand-in) ----

# The owner of p = 5 with positive, negative, off-diagonal, exact-zero and explicitly stored zero entries
# (the stored zero is at (4, 3)); its matrix is written out below.
function symmetric_owner(T)
    X = SymmetricSparseIterate(T, 5)
    X[1, 1] = 2; X[3, 1] = T(-1.5); X[2, 4] = T(0.5); X[5, 5] = -3
    activate!(X, only(k for k in 1:15 if ProximalBase.tril_cartesian(5, k) == (4, 3)))
    X
end
function symmetric_matrix(T)
    E = zeros(T, 5, 5)
    E[1, 1] = 2; E[3, 1] = E[1, 3] = -3 / 2; E[2, 4] = E[4, 2] = 1 / 2; E[5, 5] = -3
    E
end

@testset "ambient_work" begin
    for T in (Float32, Float64)
        X = symmetric_owner(T)
        @test ambient_work(X) == 25
        @test ambient_work(X) != ncoordinates(X) && ncoordinates(X) == 15
        @test ambient_work(SymmetricSparseIterate(T, 7)) == 49
        @test ambient_work(zeros(T, 3, 4)) == 12
        @test ambient_work(zeros(T, 7)) == 7
    end
end

@testset "dense of an owner" begin
    for T in (Float32, Float64)
        X = symmetric_owner(T)
        D = dense(X)
        @test D isa Matrix{T}
        @test D == symmetric_matrix(T)
        @test D[3, 1] == D[1, 3] == T(-1.5) && D[4, 2] == D[2, 4] == T(0.5)
        @test D[4, 3] == 0 && D[3, 4] == 0 && nactive(X) == 5     # the stored zero is active
        @test dense(zeros(T, 2, 2)) == zeros(T, 2, 2)
    end
end

@testset "finite_entries of an owner" begin
    for T in (Float32, Float64)
        X = symmetric_owner(T)
        @test finite_entries(X)
        @test size(owner_matrix(X)) == (5, 5)                     # the whole matrix is checked
        @test finite_entries(randn(T, 3)) && !finite_entries(T[1, NaN])
        for bad in (T(NaN), T(Inf), T(-Inf))
            Y = symmetric_owner(T); Y[5, 5] = bad                 # a diagonal entry
            @test !finite_entries(Y)
            Y = symmetric_owner(T); Y[2, 4] = bad                 # an off-diagonal entry, both triangles
            @test !finite_entries(Y) && count(!isfinite, Matrix(Y)) == 2
            Y = symmetric_owner(T); Y[4, 3] = bad                 # the stored zero
            @test !finite_entries(Y)
        end
    end
end

@testset "logical digest of an owner" begin
    for T in (Float32, Float64)
        X = symmetric_owner(T)
        @test digestable(X) && digest(X) isa UInt64
        h = digest(X)
        @test digest(X) == h && digest(copy(X)) == h
        Y = symmetric_owner(T); Y[1, 1] = 3                       # a stored value
        @test digest(Y) != h
        Y = symmetric_owner(T)                                    # an explicit zero; the matrix is unchanged
        activate!(Y, only(k for k in 1:15 if ProximalBase.tril_cartesian(5, k) == (5, 1)))
        @test Matrix(Y) == Matrix(X) && digest(Y) != h
        Y = symmetric_owner(T)                                    # 0.0 becomes -0.0
        a = only(a for (a, k) in enumerate(activecoordinates(Y)) if ProximalBase.tril_cartesian(5, k) == (4, 3))
        activevalues(Y)[a] = -zero(T)
        @test Matrix(Y) == Matrix(X) && digest(Y) != h
        @test digest(SymmetricSparseIterate(T, 5)) != h
    end
end

@testset "copy, similar and copyto! of an owner" begin
    for T in (Float32, Float64)
        X = symmetric_owner(T); h = digest(X)
        Y = copy(X); Y[1, 1] = 9
        @test digest(X) == h && Matrix(X) == symmetric_matrix(T)
        S = similar(X)
        @test S isa SymmetricSparseIterate{T} && size(S) == (5, 5) && nactive(S) == 0 && digest(X) == h
        Z = SymmetricSparseIterate(T, 5)
        copyto!(Z, X)
        @test sort(collect(activecoordinates(Z))) == sort(collect(activecoordinates(X)))
        @test nactive(Z) == 5 && count(iszero, activevalues(Z)) == 1     # the explicit zero is kept
        @test Matrix(Z) == symmetric_matrix(T) && digest(Z) == h
        Z[5, 5] = 7; Z[1, 2] = 1
        @test digest(X) == h && Matrix(X) == symmetric_matrix(T)
    end
end

@testset "clears of an owner through its coordinates" begin
    for T in (Float32, Float64)
        X = symmetric_owner(T)
        fill!(coordinates(X), 0)
        @test all(iszero, activevalues(X)) && all(iszero, Matrix(X)) && ncoordinates(X) == 15
        dropzeros!(X)
        @test nactive(X) == 0 && all(iszero, Matrix(X))
    end
end

