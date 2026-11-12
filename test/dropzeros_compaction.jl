module DropzerosCompactionTest

# dropzeros! on SparseIterate and SymmetricSparseIterate shares one compaction loop (#834). Each
# correctness-check call is compared with an independent copy of the previous swap-with-last loop, run on copies of
# the raw storage: the count, the live values and coordinate IDs in order, and the whole reverse map
# must agree. Stale slots past the count may differ and are not compared. The entries are checked
# against an independent dense reference, and the state is used afterwards (activation, writes,
# copies, clearing).

using Test
using ProximalBase
using Random
using SparseArrays: dropzeros!

# the loop that dropzeros! used on SymmetricSparseIterate before #834: a removed slot takes the
# value of the last slot, which is then checked again
function previous_dropzeros(nzval, nzval2ind, ind2nzval, nnz)
    nzval, nzval2ind, ind2nzval = copy(nzval), copy(nzval2ind), copy(ind2nzval)
    i = 1
    while i <= nnz
        if iszero(nzval[i])
            ind2nzval[nzval2ind[i]] = 0
            if i != nnz
                nzval[i] = nzval[nnz]
                ind2nzval[nzval2ind[nnz]] = i
                nzval2ind[i] = nzval2ind[nnz]
            end
            nnz -= 1
        else
            i += 1
        end
    end
    (nnz = nnz, values = nzval[1:nnz], ids = nzval2ind[1:nnz], reverse = ind2nzval)
end

live(x) = (nnz = nactive(x), values = collect(activevalues(x)),
           ids = collect(activecoordinates(x)), reverse = copy(x.ind2nzval))

# the entries as a dense array, built from the coordinate values only
dense(x::SparseIterate) = Array(x)
function dense(x::SymmetricSparseIterate{T}) where {T}
    p = size(x, 1)
    A = zeros(T, p, p)
    for k = 1:ncoordinates(x)
        r, c = tril_cartesian(p, k)
        A[r, c] = A[c, r] = coordinates(x)[k]
    end
    A
end

new_iterate(::Val{:sparse}, ::Type{T}, q) where {T} = SparseIterate(T, q)
function new_iterate(::Val{:symmetric}, ::Type{T}, q) where {T}
    p = 1
    while p * (p + 1) ÷ 2 < q
        p += 1
    end
    SymmetricSparseIterate(T, p)
end

# activate the coordinates in the given order with the given values (zeros stay stored)
function fill_working_set!(x, ids, vals)
    c = coordinates(x)
    for (k, v) in zip(ids, vals)
        activate!(x, k)
        c[k] = v
    end
    x
end

# run dropzeros! once and compare with the previous loop and with the dense reference
function check_drop!(x)
    expected = previous_dropzeros(x.nzval, x.nzval2ind, x.ind2nzval, nactive(x))
    before = dense(x)
    q = ncoordinates(x)
    @test dropzeros!(x) === x
    got = live(x)
    @test got.nnz == expected.nnz
    @test isequal(got.values, expected.values)
    @test got.ids == expected.ids
    @test got.reverse == expected.reverse
    # the working set is the stored nonzeros, the reverse map is consistent with it
    @test !any(iszero, got.values)
    @test all(x.ind2nzval[got.ids[s]] == s for s = 1:got.nnz)
    @test count(!iszero, x.ind2nzval) == got.nnz
    # entries are unchanged: equal values, NaN in the same places; a dropped -0.0 now reads 0
    after = dense(x)
    @test all(a == b || (isnan(a) && isnan(b)) for (a, b) in zip(after, before))
    @test all(isequal(after[i], before[i]) for i in eachindex(before) if !iszero(before[i]))
    x isa SymmetricSparseIterate && @test isequal(after, permutedims(after))
    @test all(isactive(x, k) == (x.ind2nzval[k] != 0) for k = 1:q)
    x
end

check_row(kind, ::Type{T}, q, ids, vals) where {T} =
    check_drop!(fill_working_set!(new_iterate(Val(kind), T, q), ids, vals))

drop_allocs(x) = @allocated dropzeros!(x)

@testset "dropzeros! shared compaction (#834)" begin
  for kind in (:sparse, :symmetric), T in (Float32, Float64)
    @testset "$kind $T" begin
      z, nz = zero(T), -zero(T)

      @testset "fixed patterns" begin
        check_row(kind, T, 6, Int[], T[])                                   # empty
        x = check_row(kind, T, 6, [3, 1, 4], T[z, nz, z])                   # all zero
        @test nactive(x) == 0 && all(iszero, x.ind2nzval)
        check_row(kind, T, 6, [3, 1, 4], T[1, 2, 3])                        # zero free
        check_row(kind, T, 6, [3, 1, 4], T[0, 2, 0])                        # partial
        check_row(kind, T, 8, 1:7, T[0, 0, 0, 1, 2, 3, 4])                  # leading run
        check_row(kind, T, 8, 1:7, T[1, 2, 0, 0, 0, 3, 4])                  # interior run
        check_row(kind, T, 8, 1:7, T[1, 2, 3, 4, 0, 0, 0])                  # trailing run
        check_row(kind, T, 8, 1:7, T[0, 0, 1, 0, 0, 2, 0])                  # runs at both ends
        check_row(kind, T, 8, [7, 3, 5, 1, 2], T[0, 4, 0, 0, 0])            # last nonzero is interior
        check_row(kind, T, 8, 1:7, T[0, 0, 0, 0, 0, 0, 1])                  # only the last slot survives
        check_row(kind, T, 8, 1:7, T[1, 0, 0, 0, 0, 0, 0])                  # only the first slot survives
        for v in (z, nz, T(5))                                              # a single active coordinate
            x = check_row(kind, T, 6, [4], T[v])
            @test nactive(x) == (iszero(v) ? 0 : 1)
        end
        # signed zeros are removed; NaN, Inf, -Inf and subnormals stay
        x = check_row(kind, T, 10, 1:9, T[NaN, nz, Inf, z, -Inf, floatmin(T) / 4, nz, -3, z])
        @test sort(collect(activecoordinates(x))) == [1, 3, 5, 6, 8]
        x = check_row(kind, T, 6, [3, 1, 4], T[nz, NaN, nz])
        @test collect(activecoordinates(x)) == [1] && isnan(coordinates(x)[1])
        if kind === :sparse                                                 # a matrix
          check_row(kind, T, 0, Int[], T[])
          check_drop!(fill_working_set!(SparseIterate(T, 3, 4), [5, 2, 9, 12], T[0, 1, 0, nz]))
          x = check_drop!(fill_working_set!(SparseIterate(T, 3, 4), [5, 2, 9], T[nz, 0, 0]))
          @test nactive(x) == 0
        end
      end

      if kind === :symmetric
        @testset "diagonal and off-diagonal coordinates" begin
          p = 5
          x = SymmetricSparseIterate(T, p)
          entries = [(3, 3, z), (4, 1, T(2)), (2, 2, T(-1)), (5, 2, nz), (1, 1, z), (5, 4, T(7)), (4, 3, z)]
          for (i, j, v) in entries
            activate!(x, tril_index(p, i, j))
            x[i, j] = v
          end
          @test x[1, 4] == 2 && x[2, 5] == 0 && nactive(x) == 7
          check_drop!(x)
          @test sort(collect(activecoordinates(x))) ==
                sort([tril_index(p, 4, 1), tril_index(p, 2, 2), tril_index(p, 5, 4)])
          @test x[4, 1] == x[1, 4] == 2 && x[2, 2] == -1 && x[5, 4] == x[4, 5] == 7
          @test x[3, 3] == 0 && x[5, 2] == x[2, 5] == 0 && x[3, 4] == 0
          # writing through the upper triangle reaches the same coordinate
          x[1, 4] = 0
          check_drop!(x)
          @test !isactive(x, tril_index(p, 4, 1)) && x[4, 1] == 0
        end
      end

      @testset "repeated compaction and later use" begin
        x = fill_working_set!(new_iterate(Val(kind), T, 12), [9, 2, 11, 5, 1, 7], T[0, 3, 0, 4, 0, 5])
        check_drop!(x)
        state = live(x)
        check_drop!(x)                                       # nothing left to drop
        @test live(x) == state
        # new zeros among the survivors, then compaction again
        coordinates(x)[collect(activecoordinates(x))[1]] = 0
        check_drop!(x)
        @test nactive(x) == 2
        # activation and writes after a compaction append to the working set
        k = findfirst(i -> !isactive(x, i), 1:ncoordinates(x))
        activate!(x, k)
        @test collect(activecoordinates(x))[end] == k && coordinates(x)[k] == 0
        coordinates(x)[k] = T(8)
        j = findlast(i -> !isactive(x, i), 1:ncoordinates(x))
        coordinates(x)[j] = T(-2)                            # a nonzero write activates
        @test collect(activecoordinates(x))[end-1:end] == [k, j]
        @test coordinates(x)[k] == 8 && coordinates(x)[j] == -2
        check_drop!(x)
        # copies after a compaction: copy, and copyto! by the sparse and the dense branch
        y = copy(x)
        @test live(y) == live(x) && isequal(dense(y), dense(x))
        for q in (64, 8)                                     # 4 (nnz + nnz) < q selects the sparse branch
          src = fill_working_set!(new_iterate(Val(kind), T, q), [1, 2, 3], T[0, 1, 0])
          check_drop!(src)
          dst = fill_working_set!(new_iterate(Val(kind), T, q), [3, 2], T[5, 0])
          @test copyto!(dst, src) === dst
          @test live(dst).ids == live(src).ids && isequal(live(dst).values, live(src).values)
          @test isequal(dense(dst), dense(src))
          check_drop!(dst)
        end
        # clearing: fill!(coordinates(x), 0) then dropzeros!
        fill!(coordinates(x), 0)
        check_drop!(x)
        @test nactive(x) == 0 && all(iszero, x.ind2nzval) && all(iszero, dense(x))
        coordinates(x)[3] = T(1)
        @test collect(activecoordinates(x)) == [3] && x.ind2nzval[3] == 1
      end

      @testset "randomized patterns" begin
        rng = Xoshiro(834)
        for frac in (0.0, 0.1, 0.5, 0.9, 1.0, -1.0), rep = 1:40
          q = rand(rng, (8, 37, 200))
          s = rand(rng, 0:q)
          f = frac < 0 ? rand(rng) : frac
          x = new_iterate(Val(kind), T, q)
          q = ncoordinates(x)
          # stale slots past the count: activate extra coordinates first and drop them
          extra = randperm(rng, q)[1:rand(rng, 0:min(5, q))]
          fill_working_set!(x, extra, [rand(rng, (z, T(9))) for _ in extra])
          fill!(coordinates(x), 0); dropzeros!(x)
          ids = randperm(rng, q)[1:s]
          vals = [rand(rng) < f ? rand(rng, (z, nz)) :
                  rand(rng) < 0.05 ? rand(rng, T[NaN, Inf, -Inf]) : T(randn(rng)) for _ = 1:s]
          check_drop!(fill_working_set!(x, ids, vals))
          # a second round of zeros on the survivors
          for k in collect(activecoordinates(x))
            rand(rng) < 0.3 && (coordinates(x)[k] = 0)
          end
          check_drop!(x)
        end
      end

      @testset "no allocation on warmed calls" begin
        mk(vals) = fill_working_set!(new_iterate(Val(kind), T, 40), 1:length(vals), vals)
        drop_allocs(mk(T[0, 1, 0, 2, 0]))                    # warm up on separate fixtures
        drop_allocs(mk(zeros(T, 5)))
        @test drop_allocs(mk(T[0, 1, 0, 2, 0, 3, 4, 0])) == 0
        @test drop_allocs(mk(zeros(T, 20))) == 0
        @test drop_allocs(mk(T[1, 2, 3])) == 0
        @test drop_allocs(mk(T[])) == 0
      end
    end
  end
end

end
