module SparseIterateBasicsTest

# Quick-tier coverage of SparseIterate and SymmetricSparseIterate: construction, value writes versus
# state copies, the active set and dropzeros!, symmetric coordinates and exact symmetry, aliasing of
# views, and one five-argument and one triple product. The exhaustive cases are in sparse_iterate.jl.

using Test
using ProximalBase
using Random
using LinearAlgebra
using SparseArrays

# a real number with no conversion to a floating-point type
struct NoFloat <: Real end
Base.iszero(::NoFloat) = false

# a real number whose conversion to Int succeeds once and then throws, to catch a second conversion
struct OnceReal <: Real
  count::Base.RefValue{Int}
end
Base.iszero(::OnceReal) = false
Base.convert(::Type{Int}, x::OnceReal) =
  (x.count[] += 1) == 1 ? 1 : throw(InexactError(:Int, Int, x))

Random.seed!(1)

function check_product_state(x, before)
    @nospecialize x before
    @test Array(x) == Array(before)
    @test activecoordinates(x) == activecoordinates(before)
    @test x.nzval[1:nnz(x)] == before.nzval[1:nnz(before)]
    @test x.ind2nzval == before.ind2nzval
end

@testset "sparseIterate" begin

    @testset "issparse" begin
        @test issparse(SparseIterate(sparse(fill(1,5,5))))
        @test issparse(SparseIterate(fill(1,5,5)))
        @test issparse(SparseIterate(fill(1,5)))
    end

    @testset "dropzeros and nnz" begin
        x = SparseIterate(3)
        x[3] = 2.
        x[1] = 1.
        @test nnz(x) == 2
        x[1] = 0.
        @test nnz(x) == 2
        dropzeros!(x)
        @test nnz(x) == 1
        @test Vector(x) == [0., 0., 2.]

        x = SparseIterate(3)
        x[3] = 2.
        x[1] = 1.
        x[3] = 0.
        x[1] = 0.
        @test nnz(x) == 2
        dropzeros!(x)
        @test nnz(x) == 0
        @test Vector(x) == [0., 0., 0.]
    end

    @testset "activate!" begin
        x = SparseIterate(5)
        x[2] = 1.
        x[4] = 0.                       # assigning a zero does not store the entry
        @test nnz(x) == 1
        @test activate!(x, 4) === x
        @test nnz(x) == 2
        @test x[4] == 0.
        @test x.nzval2ind[1:nnz(x)] == [2, 4]
        @test x.nzval[1:nnz(x)] == [1., 0.]

        # activating a stored entry changes nothing
        activate!(x, 2)
        activate!(x, 4)
        @test nnz(x) == 2
        @test Vector(x) == [0., 1., 0., 0., 0.]

        # the activated entry is written in place, and dropzeros! removes it while zero
        x[4] = 3.
        @test nnz(x) == 2
        @test Vector(x) == [0., 1., 0., 3., 0.]
        x[4] = 0.
        dropzeros!(x)
        @test nnz(x) == 1
        @test x.nzval2ind[1:nnz(x)] == [2]

        @test_throws BoundsError activate!(x, 6)

        y = SparseIterate(Float32, 2, 3)
        activate!(y, 5)
        @test nnz(y) == 1
        @test y[1, 3] === 0f0
    end

    @testset "active coordinates" begin
        x = SparseIterate(6)
        x[4] = 1.
        x[2] = 3.
        activate!(x, 6)
        @test nactive(x) == 3
        @test activecoordinates(x) == [4, 2, 6]
        @test isactive(x, 2) && isactive(x, 6) && !isactive(x, 1)
        @test_throws BoundsError activecoordinates(x)[4]
        dropzeros!(x)
        @test activecoordinates(x) == [4, 2]
        x[4] = 0.
        dropzeros!(x)                   # the last active coordinate moves into the gap
        @test activecoordinates(x) == [2]
        @test ncoordinates(x) == 6
        @test coordinates(x) === x
    end

    @testset "setindex! converts" begin
        x = SparseIterate(3)
        x[1] = 1                        # an Int
        x[2] = 2f0                      # a Float32
        @test Vector(x) == [1., 2., 0.]
        x[1] = 0                        # a stored entry stays stored
        x[3] = 0                        # a zero is not stored
        @test nnz(x) == 2
        @test_throws InexactError (SparseIterate(Int, 2)[1] = 0.5)
        @test_throws BoundsError (x[4] = 1.)

        y = SparseIterate(2, 2)
        y[2, 1] = 3
        @test Matrix(y) == [0. 0.; 3. 0.]
    end

    @testset "fill! and norm" begin
        x = SparseIterate([0., 3., 0., -4.])
        @test norm(x) == 5.
        @test norm(x, 1) == 7. && norm(x, Inf) == 4. && norm(x, 0) == 2.
        @test norm(x, -Inf) == 0.                     # the entries that are not stored count
        @test fill!(x, 0) === x
        @test nnz(x) == 0 && Vector(x) == zeros(4)
        fill!(x, 2.)
        @test Vector(x) == fill(2., 4)
    end


    @testset "copy" begin
        v = sprand(20, 0.4)
        x = SparseIterate(v)
        y = copy(x)
        @test Vector(y) == Vector(v)
        @test Vector(y) == Vector(x)
        @test !(x === y)

        M = sprand(10, 20, 0.2)
        x = SparseIterate(M)
        y = copy(x)
        @test Matrix(y) == Matrix(M)
        @test Matrix(y) == Matrix(x)
        @test !(x === y)

        # copyto! into an iterate that holds other entries, via the sparse and the dense path
        for d in (0.02, 0.6)
            v, w = sprandn(200, d), sprandn(200, d)
            x, y = SparseIterate(v), SparseIterate(w)
            @test copyto!(x, y) === x
            @test Vector(x) == Vector(w)
            @test nnz(x) == nnz(y)
            @test Vector(copyto!(x, x)) == Vector(w)

            # entries set afterwards are stored correctly
            i = findfirst(iszero, Vector(w))
            x[i] = 7.
            @test x[i] == 7.
            @test nnz(x) == nnz(y) + 1
            @test Vector(y) == Vector(w)
        end
        @test_throws DimensionMismatch copyto!(SparseIterate(3), SparseIterate(4))
    end

    @testset "a copy whose conversion throws leaves the destination unchanged" begin
        # the state of an iterate as the scalar reads, the active list and Vector see it
        snap(z) = (collect(z), isactive.(Ref(z), 1:length(z)), copy(activecoordinates(z)), Vector(z))
        # (n, filled): sparse branch when 4 * (nnz(d) + nnz(s)) < n, dense branch otherwise
        for (n, filled) in ((16, false), (32, true), (16, true), (4, false), (4, true))
            d = SparseIterate(Int, n)
            if filled
                d[n÷2] = 7; d[n] = 3
            end
            s = SparseIterate(n); s[1] = 1.0; s[2] = 1.5
            before = snap(d)
            @test_throws InexactError copyto!(d, s)
            @test snap(d) == before
            d[1] = 5                                   # still usable
            @test d[1] == 5 && isactive(d, 1)
        end

        # an integral source converts; stale values past nnz of the source do not matter
        s = SparseIterate(4); s[1] = 1.0; s[2] = 1.5; s[1] = 0; dropzeros!(s); s[2] = 1   # 1.5 stays in the unused slot
        d = SparseIterate(Int, 4)
        @test copyto!(d, s) === d && Vector(d) == [0, 1, 0, 0]
        s = SparseIterate(16); s[3] = 2.0
        @test Vector(copyto!(SparseIterate(Int, 16), s)) == Vector(s)

        # a source type with no conversion to the destination's: checked, not assumed to round
        r = SparseIterate(Real, 16); r[9] = 7; r[1] = 1; r[2] = NoFloat()
        f = SparseIterate(Float64, 16)
        f[5] = 4.0
        before = snap(f)
        @test_throws Exception copyto!(f, r)
        @test snap(f) == before

        # a fallible conversion runs once per value, before the destination changes
        for (n, filled) in ((16, false), (32, true), (4, true))
            c = Ref(0)
            r = SparseIterate(Real, n); r[1] = OnceReal(c)
            d = SparseIterate(Int, n)
            filled && (d[n] = 3)
            @test copyto!(d, r) === d && d[1] == 1 && c[] == 1
        end

        # SymmetricSparseIterate shares the copy
        S = SymmetricSparseIterate(5); S[2, 1] = 1.0; S[3, 3] = 0.5
        D = SymmetricSparseIterate(Int, 5); D[1, 1] = 4
        @test_throws InexactError copyto!(D, S)
        @test D[1, 1] == 4 && nactive(D) == 1 && Matrix(D) == Matrix(Diagonal([4, 0, 0, 0, 0]))
    end

    @testset "five-argument stored zeros" begin
        A = [Inf 1. 3.; 2. 4. 5.]
        B = Matrix(transpose(A))
        x = SparseIterate([0., 2., 0.])
        activate!(x, 1)
        X = SparseIterate([0. 0.; 2. 0.; 0. -1.])
        activate!(X, 1)
        X_sparse = SparseIterate(3, 32)
        X_sparse[2, 1], X_sparse[3, 2] = 2., -1.
        activate!(X_sparse, 1)
        @test 32 * nnz(X_sparse) == length(X_sparse)
        @test 32 * nnz(X) > length(X)
        for (M, coef) in ((A, x), (A, X_sparse), (A, X), (transpose(B), x), (adjoint(B), x))
            before = copy(coef)
            dims = ndims(coef) == 1 ? (2,) : (2, size(coef, 2))
            reference = mul!(fill(NaN, dims...), M, coef)
            @test isnan(reference[1]) && reference[2] == 8.
            if ndims(coef) == 2
                @test reference[:, 2] == [-3., -5.]
                @test all(iszero, view(reference, :, 3:size(coef, 2)))
            end
            for (α, β) in ((1.0, 0.0), (2.5, -0.5))
                out = fill(iszero(β) ? NaN : 3., dims...)
                expected = iszero(β) ? α .* reference : α .* reference .+ β * 3.
                @test isequal(mul!(out, M, coef, α, β), expected)
            end
            @test mul!(fill(NaN, dims...), M, coef, 0.0, 0.0) == zeros(dims...)
            @test mul!(fill(3., dims...), M, coef, 0.0, -0.5) == fill(-1.5, dims...)
            @test all(isnan, mul!(fill(NaN, dims...), M, coef, 1.0, 1.0))
            check_product_state(coef, before)
        end
    end
end

@testset "sparseSymmetricIterate" begin

    @testset "a symmetric matrix" begin
        s = SymmetricSparseIterate(3)
        s[2, 1] = 1                     # setindex! converts, as for SparseIterate
        s[1, 3] = 2f0
        @test s[1, 2] === 1. && s[3, 1] === 2.
        s[2, 2] = 5.
        M = [0. 1. 2.; 1. 5. 0.; 2. 0. 0.]
        @test Matrix(s) == M
        # linear indices, length and iteration follow the dense matrix
        @test length(s) == 9 && size(s) == (3, 3)
        @test [s[i] for i in 1:9] == vec(M)
        @test collect(s) == M
        @test vec(s) == vec(M)
        @test sum(s) == sum(M)
        @test s == M
        # an index outside the matrix does not alias an entry of the lower triangle
        @test_throws BoundsError s[4, 1]
        @test_throws BoundsError s[1, 4]
        @test_throws BoundsError (s[4, 1] = -1.)
        @test_throws BoundsError s[10]
        @test s[2, 2] == 5. && nnz(s) == 3
    end

    @testset "writes update each entry once" begin
        p = 5
        A = sprandn(p, p, 0.4)
        S = SymmetricSparseIterate(sparse(A + A') + 2p * I)
        # a linear index is an entry of the dense matrix, and writing it writes
        # its mirror image
        S[7] = -1.                       # (2, 2)
        @test S[2, 2] == -1.
        S[2] = 3.                        # (2, 1), so (1, 2) too
        @test S[1, 2] == S[2, 1] == 3.
        @test_throws BoundsError S[0]
        @test_throws BoundsError S[p^2 + 1]
        @test_throws BoundsError (S[p^2 + 1] = 1.)
        # scaling and axpy update an off-diagonal entry once, not twice
        M = Matrix(S)
        @test Matrix(rmul!(copy(S), 2.)) == 2M && Matrix(lmul!(2., copy(S))) == 2M
        @test Matrix(rdiv!(copy(S), 2.)) == M / 2 && Matrix(ldiv!(2., copy(S))) == M / 2
        @test Matrix(axpy!(2., M, copy(S))) ≈ 3M && Matrix(axpby!(2., M, 3., copy(S))) ≈ 5M
        @test Matrix(axpy!(2., S, copy(S))) ≈ 3M
        @test_throws DimensionMismatch axpy!(2., vec(M), copy(S))
        @test Matrix(axpy!(0., fill(NaN, p, p), copy(S))) == M
        @test_throws DimensionMismatch axpy!(1., ones(3, 3), copy(S))
        @test_throws DimensionMismatch axpby!(1., ones(3, 3), 1., copy(S))
        # a stored zero stays stored, and a SparseIterate is scaled in O(nnz)
        x = SparseIterate(sparse([1., 0., 2.]))
        @test rmul!(x, 3.) === x && Vector(x) == [3., 0., 6.] && nnz(x) == 2
        @test Vector(ldiv!(3., x)) == [1., 0., 2.]
    end

    @testset "coordinates" begin
        p = 4
        s = SymmetricSparseIterate(p)
        s[3, 2] = 1.5
        s[1, 1] = -2.
        @test ncoordinates(s) == p * (p + 1) ÷ 2
        @test nactive(s) == 2
        @test activecoordinates(s) == [tril_index(p, 3, 2), tril_index(p, 1, 1)]
        @test isactive(s, tril_index(p, 3, 2)) && !isactive(s, tril_index(p, 2, 1))
        c = coordinates(s)
        @test c isa AbstractVector && length(c) == ncoordinates(s)
        @test c[tril_index(p, 3, 2)] == 1.5 && c[tril_index(p, 4, 1)] == 0.
        c[tril_index(p, 4, 1)] = 7.             # writes the entries (4, 1) and (1, 4)
        @test s[1, 4] == 7. && nnz(s) == 3
        @test_throws BoundsError c[ncoordinates(s) + 1]
        @test activate!(s, tril_index(p, 2, 2)) === s
        @test nactive(s) == 4 && s[2, 2] == 0.
        for k in 1:ncoordinates(s)
            r, cc = tril_cartesian(p, k)
            @test r >= cc && tril_index(p, r, cc) == k
        end
    end

    @testset "dropzeros and nnz" begin
        x = SymmetricSparseIterate(2)
        x[2, 2] = 2.
        x[1, 1] = 1.
        @test nnz(x) == 2
        x[1, 1] = 0.
        @test nnz(x) == 2
        dropzeros!(x)
        @test nnz(x) == 1
        @test Matrix(x) == [0.  0.; 0. 2.]
    end

    @testset "construct" begin
      n = 10
      M = sprandn(n, n, 0.1)
      M = (M + M') / 2
      x = SymmetricSparseIterate(M)
      @test ncoordinates(x) == n * (n+1) / 2
      @test Matrix(x) == Matrix(M)
      @test Array(x) == Matrix(M)
      @test typeof(SymmetricSparseIterate(2)) == SymmetricSparseIterate{Float64}
      # from a matrix that is not symmetric, only through Symmetric(D, :L)
      D = randn(4, 4)
      @test_throws ArgumentError SymmetricSparseIterate(D)
      @test_throws ArgumentError SymmetricSparseIterate(sparse(D))
      y = SymmetricSparseIterate(Symmetric(D, :L))
      @test Matrix(y) == Matrix(Symmetric(D, :L))
      @test nnz(y) == 10
      @test size(SymmetricSparseIterate(0)) == (0, 0)
      for p in (-1, -2)
        @test_throws ArgumentError SymmetricSparseIterate(p)
        @test_throws ArgumentError SymmetricSparseIterate(Float32, p)
      end
    end

    @testset "broadcasting and copyto!" begin
      D = randn(5, 5); D = D + D'
      s = SymmetricSparseIterate(D)
      s .= 2 .* s                      # each entry of the lower triangle once
      @test Matrix(s) ≈ 2D
      s .= abs.(s) .- 1
      @test Matrix(s) ≈ abs.(2D) .- 1
      t = similar(s)
      t .= s .+ D
      @test Matrix(t) ≈ abs.(2D) .- 1 .+ D
      # copyto! and broadcasting take a symmetric matrix, and one triangle of
      # a matrix that is not symmetric only through Symmetric
      E = randn(5, 5)
      F = E + E'
      @test_throws ArgumentError copyto!(t, E)
      @test Matrix(copyto!(t, Symmetric(E, :L))) == Matrix(Symmetric(E, :L))
      for source in (view(E, :, :), transpose(E), sparse(E), UpperTriangular(E))
        @test_throws ArgumentError copyto!(t, source)
      end
      for source in (view(F, :, :), transpose(F), sparse(F), Symmetric(sparse(E), :U))
        @test copyto!(t, source) === t
        @test Matrix(t) == Matrix(source)
      end
      for T in (Float64,)
        @test_throws ArgumentError copyto!(SymmetricSparseIterate(T, 5), SparseArrays.CHOLMOD.Dense(T.(E)))
        source = SparseArrays.CHOLMOD.Dense(T.(F))
        dest = SymmetricSparseIterate(T, 5)
        @test copyto!(dest, source) === dest
        @test Matrix(dest) == T.(F)
        # The differently typed destination takes CHOLMOD's untyped
        # intersection rather than its element-type-matched method.
        mixed = SymmetricSparseIterate(Float32, 5)
        @test copyto!(mixed, source) === mixed
        @test Matrix(mixed) == Float32.(F)
      end
      @test_throws DimensionMismatch copyto!(t, view(F, 1:4, 1:4))
      @test_throws ArgumentError t .= E
      t .= Symmetric(E, :L)
      @test Matrix(t) == Matrix(Symmetric(E, :L))
      fill!(t, 0)
      @test nnz(t) == 0
      fill!(t, 1.)
      @test Matrix(t) == ones(5, 5) && nnz(t) == 15
    end

    @testset "copyto! from sparse wrappers" begin
      A = sparse(Float64[1 3; 5 0])      # not symmetric
      B = sparse(Float64[1 3; 3 0])      # symmetric
      wraps = Any[transpose, adjoint]
      @static if isdefined(SparseArrays, :SparseMatrixCSCOrView)
        push!(wraps, M -> view(M, :, 1:2))
      end
      for wrap in wraps
        dest = fill!(SymmetricSparseIterate(Float64, 2), 9)
        @test_throws ArgumentError copyto!(dest, wrap(A))
        @test Matrix(dest) == fill(9, 2, 2)
        source = wrap(B)
        @test copyto!(dest, source) === dest
        @test Matrix(dest) == Matrix(source)
        @test_throws DimensionMismatch copyto!(SymmetricSparseIterate(3), source)
      end
      @static if isdefined(SparseArrays, :AdjOrTransSparseVectorOrView)
        v = sparse(Float64[2])
        B = sparse(reshape(Float64[2], 1, 1))
        for parent in (v, view(v, :), view(B, :, 1)), wrap in (transpose, adjoint)
          source = wrap(parent)
          dest = SymmetricSparseIterate(Float64, 1)
          @test copyto!(dest, source) === dest
          @test Matrix(dest) == Matrix(source)
          @test_throws DimensionMismatch copyto!(SymmetricSparseIterate(2), source)
        end
        @test_throws DimensionMismatch copyto!(SymmetricSparseIterate(1), transpose(sparse([1., 2.])))
      end
    end

    @testset "copy" begin
      n = 6
      M = sprandn(n, n, 0.4)
      M = (M + M') / 2
      x = SymmetricSparseIterate(M)

      y = similar(x)
      @test y isa SymmetricSparseIterate{Float64}
      @test size(y) == size(x)
      @test nnz(y) == 0
      @test similar(x, Float32) isa SymmetricSparseIterate{Float32}

      copyto!(y, x)
      @test Matrix(y) == Matrix(x)
      @test nnz(y) == nnz(x)
      @test_throws DimensionMismatch copyto!(SymmetricSparseIterate(3), x)
      z = copy(x)
      @test z isa SymmetricSparseIterate && Matrix(z) == Matrix(x)

      # into a destination that holds other entries (sparse path)
      x = SymmetricSparseIterate(20); x[3, 1] = 1.; x[5, 5] = 2.
      y = SymmetricSparseIterate(20); y[5, 5] = -1.; y[9, 2] = 4.
      copyto!(x, y)
      @test Matrix(x) == Matrix(y)
      @test nnz(x) == 2
      x[1, 3] = 6.
      @test x[3, 1] == 6.
      @test nnz(x) == 3
      @test Matrix(y)[3, 1] == 0.
    end
end


@testset "sparse triple products agree with dense products" begin
    rng = Xoshiro(20261025)
    for T in (Float32, Float64)
        m, n = 8, 10
        A = Symmetric(randn(rng, T, m, m))
        B = Symmetric(randn(rng, T, n, n))
        # Concentrated support exercises each multiplication order, and no
        # support verifies the empty-product dimensions. Stored zeros must
        # contribute neither a row nor a column to the restricted product.
        for (rows, columns) in (([2], [3, 7]), ([2, 6], [3]), (Int[], Int[]))
            dense = zeros(T, m, n)
            for i in rows, j in columns
                dense[i, j] = T(i - j)
            end
            sparse = SparseIterate(dense)
            activate!(sparse, 1)
            product = A * sparse * B
            @test product ≈ A * dense * B
            @test size(product) == (m, n)
            @test eltype(product) == T
            @test Matrix(sparse) == dense
        end
    end
end

@testset "aliasing of the views of a SparseIterate" begin
  function alloc_unaliased(y, x)
    y .= x .+ 1
    copyto!(y, x)
    return nothing
  end
  function alloc_aliased(x)
    copyto!(x, activevalues(x))
    return nothing
  end
  for x in (SparseIterate(Float64, 6), SparseIterate(Float32, 6), SparseIterate(Float64, 3, 2))
    T = eltype(x)
    for i in (2, 1)   # the source starts as [2, 1]: the first write would overwrite the second source value
      x[i] = T(i)
    end
    @test Base.mightalias(x, activevalues(x))
    @test Base.mightalias(activevalues(x), x)
    @test Base.mightalias(x, activecoordinates(x))
    @test Base.mightalias(x, view(x, 1:2))
    @test Base.mightalias(x, reshape(x, :))
    @test all(id -> id in Base.dataids(x), Base.dataids(activevalues(x)))
    @test all(id -> id in Base.dataids(x), Base.dataids(activecoordinates(x)))
    y = similar(x)
    @test !Base.mightalias(x, y)
    @test !Base.mightalias(activevalues(x), y)
    @test !Base.mightalias(x, zeros(T, size(x)))

    # the source is a view of x: the result is the one for a private copy of the source
    expected = Array(x)
    copyto!(expected, copy(activevalues(x)))
    copyto!(x, activevalues(x))
    @test Array(x) == expected
  end

  # a broadcast whose source is a view of the destination
  x = SparseIterate(Float64, 3)
  for i in (3, 1, 2)
    x[i] = Float64(i)
  end
  expected = copy(activevalues(x)) .+ 0
  x .= activevalues(x) .+ 0
  @test Array(x) == expected

  # unaliased broadcasts and copies do not make a private copy of the source
  x = SparseIterate(Float64, 6)
  x[2] = 1.0
  y = SparseIterate(Float64, 6)
  alloc_unaliased(y, x)
  @test (@allocated alloc_unaliased(y, x)) == 0
  alloc_aliased(x)
  @test (@allocated alloc_aliased(x)) > 0
end

end
