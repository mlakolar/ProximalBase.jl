module SymmetricMutatorErrorsTest

using Test, ProximalBase, LinearAlgebra, SparseArrays, Random

state(x) = (Matrix(x), nnz(x), copy(activecoordinates(x)), x.nzval[1:nnz(x)])

@noinline function rejects(f, x, E, msg = nothing)
  Base.@nospecialize f x E msg
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

initial(T, p) = fill!(SymmetricSparseIterate(T, p), -1)
symmetric(p) = Float64[r + c for r in 1:p, c in 1:p]

@noinline function call_mutator(op, args...; kws...)
  Base.@nospecialize op args kws
  op(args...; kws...)
end

# Keep the reference writers behind a function barrier for abstract destinations.
@noinline function check_named_destination(make)
  Base.@nospecialize make
  A = [4.0 1; 2 5]
  P = [4.0 1; 1 5]
  @test Matrix(call_mutator(reverse!, make(A); dims = 1)) == [2 5; 4 1]
  if make === sparse
    @test Matrix(call_mutator(sort!, sparse(Bool[1 0; 0 1]); dims = 1)) == Bool[0 0; 1 1]
  else
    @test Matrix(call_mutator(sort!, make(A); dims = 1)) == [2 1; 4 5]
  end
  @test Matrix(call_mutator(accumulate!, +, make(A), A; dims = 1)) == [4 1; 6 6]
  @test Matrix(call_mutator(kron!, make(zeros(2, 2)), A, ones(1, 1))) == A
  @test Matrix(call_mutator(triu!, make(A))) == [4 1; 0 5]
  @test Matrix(call_mutator(tril!, make(A))) == [4 0; 2 5]
  D = Diagonal([2.0, 3.0])
  @test Matrix(call_mutator(lmul!, D, make(P))) == [8 2; 3 15]
  @test Matrix(call_mutator(rmul!, make(P), D)) == [8 3; 2 15]
  @test Matrix(call_mutator(ldiv!, D, make(P))) == [2 1/2; 1/3 5/3]
  @test Matrix(call_mutator(rdiv!, make(P), D)) == [2 1/3; 1/2 5/3]
  @test Matrix(call_mutator(ldiv!, make(zeros(2, 2)), D, P)) == [2 1/2; 1/3 5/3]
  @test Matrix(call_mutator(get!, make(zeros(2, 2)), A, (1:2, 1:2), 7.0)) == A
  @test Matrix(call_mutator(sortperm!, make(ones(Int, 2, 2)), A; dims = 1)) == [2 3; 1 4]
  @static if isdefined(LinearAlgebra, :copytrito!)
    B = make(zeros(2, 2))
    @test call_mutator(LinearAlgebra.copytrito!, B, A, 'L') === B
    @test B[2, 1] == A[2, 1]
  end
end

@testset "symmetric mutator errors" begin
  @testset "position-wise mutators" begin
    for T in DataType[Float64, Float32], p in Any[0, 1, 3]
      x = initial(T, p)
      A = ones(T, p, p)
      for dims in Any[:, 1, 2, (1, 2)]
        rejects(x -> reverse!(x; dims), x, ArgumentError, "reverse!")
      end
      for f in Any[x -> reverse(x), x -> reverse(x; dims = 1),
                x -> sort!(x; dims = 1), x -> accumulate!(+, x, A; dims = 1),
                x -> cumsum!(x, A; dims = 1), x -> cumprod!(x, A; dims = 2),
                x -> cumsum(x; dims = 1), x -> cumprod(x; dims = 2),
                x -> accumulate(max, x; dims = 1)]
        rejects(f, x, ArgumentError)
      end
      for (a, b) in Any[(A, ones(T, 1, 1)), (A, Diagonal(ones(T, 1))),
                     (Diagonal(ones(T, p)), ones(T, 1, 1)),
                     (Diagonal(ones(T, p)), Diagonal(ones(T, 1))),
                     (sparse(A), Diagonal(ones(T, 1))), (A, T(2)), (T(2), A)]
        rejects(x -> kron!(x, a, b), x, ArgumentError, "kron!")
      end
      @static if isdefined(LinearAlgebra, :copytrito!)
        for a in Any[A, sparse(A)], uplo in Any['L', 'U']
          rejects(x -> LinearAlgebra.copytrito!(x, a, uplo), x, ArgumentError, "copytrito!")
        end
      end
      for (f, boundary, bad) in Any[(triu!, 1 - p, 2 - p), (tril!, p - 1, p - 2)]
        before = state(x)
        @test f(x, boundary) === x
        @test isequal(state(x), before)
        if p == 0
          for k in Any[-3, 0, 3]
            @test f(x, k) === x
            @test isequal(state(x), before)
          end
        else
          rejects(x -> f(x, bad), x, ArgumentError, string(f))
        end
        if p <= 1
          @test f(x) === x
          @test isequal(state(x), before)
        else
          rejects(f, x, ArgumentError, string(f))
        end
      end
      for f in Any[lu!, qr!, cholesky!, bunchkaufman!]
        rejects(f, x, ArgumentError, string(f))
      end
      for pivot in Any[NoPivot(), RowMaximum(), RowNonZero(), Val(true), Val(false)]
        rejects(x -> lu!(x, pivot; check = false), x, ArgumentError, "lu!")
      end
      for pivot in Any[NoPivot(), ColumnNorm(), Val(true), Val(false)]
        rejects(x -> qr!(x, pivot), x, ArgumentError, "qr!")
      end
      for pivot in Any[NoPivot(), RowMaximum(), Val(true), Val(false)]
        rejects(x -> cholesky!(x, pivot; check = false), x, ArgumentError, "cholesky!")
      end
      for rook in Any[true, false]
        rejects(x -> bunchkaufman!(x, rook; check = false), x, ArgumentError, "bunchkaufman!")
      end
      for I in Any[1:p^2, collect(1:p^2), (1:p, 1:p), [collect(1:p), collect(1:p)]]
        rejects(x -> get!(x, A, I, T(7)), x, ArgumentError, "get!")
      end
      rejects(x -> sortperm!(x, A; dims = 1), initial(Int, p), ArgumentError, "sortperm!")
    end
  end

  @testset "matrix operands" begin
    for T in DataType[Float64, Float32]
      A = T[4 1 2; 2 5 1; 1 2 6]
      P = T[5 1 1; 1 5 1; 1 1 5]
      G = givens(T(1), T(2), 1, 2)[1]
      Q = qr(A, ColumnNorm()).Q
      ops = (A, Diagonal(T[2, 3, 4]), Bidiagonal(T[2, 3, 4], ones(T, 2), :U),
             UpperTriangular(A), LowerTriangular(A), Symmetric(P), SymmetricSparseIterate(P),
             sparse(A), lu(A), cholesky(P), qr(A), Q, adjoint(Q), G, adjoint(G))
      for op in ops
        x = initial(T, 3)
        for (f, call) in Any[("lmul!", x -> lmul!(op, x)), ("rmul!", x -> rmul!(x, op)),
                          ("ldiv!", x -> ldiv!(op, x)), ("rdiv!", x -> rdiv!(x, op)),
                          ("ldiv!", x -> ldiv!(x, op, P))]
          rejects(call, x, ArgumentError, f)
        end
      end
      # A symmetric solution still requires a dense destination for the solve.
      rejects(x -> ldiv!(x, lu(P), P), initial(T, 3), ArgumentError, "ldiv!")
      for p in Any[0, 1, 3], s in Any[T(2), T(2)I], f in Any[lmul!, rmul!, ldiv!, rdiv!]
        x = copyto!(SymmetricSparseIterate(T, p), symmetric(p))
        M = Matrix(x)
        left = f === lmul! || f === ldiv!
        expected = left ? f(s, M) : f(M, s)
        @test (left ? f(s, x) : f(x, s)) === x
        @test Matrix(x) == expected
      end
    end
  end

  @testset "reduction destinations" begin
    for T in DataType[Float64, Float32], p in Any[0, 1, 3]
      x = initial(T, p)
      A = ones(T, p, p, 2)
      for f in Any[sum!, prod!, maximum!, minimum!, all!, any!, extrema!]
        rejects(x -> f(x, A), x, ArgumentError, string(f))
        rejects(x -> f(identity, x, A; init = false), x, ArgumentError, string(f))
      end
      for a in Any[trues(p, p, 2), Base.Broadcast.broadcasted(>, A, 0)]
        rejects(x -> count!(x, a), x, ArgumentError, "count!")
        rejects(x -> count!(identity, x, a; init = false), x, ArgumentError, "count!")
      end
      for f in Any[findmax!, findmin!]
        inds = fill(CartesianIndex(0, 0, 0), p, p)
        before = copy(inds)
        rejects(x -> f(x, inds, A), x, ArgumentError, string(f))
        @test inds == before
      end
    end
  end

  @testset "orthogonal copies and messages" begin
    for T in DataType[Float64, Float32]
      for p in Any[0, 1]
        Q = qr(Matrix{T}(I, p, p)).Q
        x = initial(T, p)
        expected = Matrix{T}(I, p, p)
        lmul!(Q, expected)
        @test copyto!(x, Q) === x
        @test Matrix(x) == expected
      end
      Q = qr(reshape(T[0, 1, 0], 3, 1)).Q
      x = initial(T, 3)
      expected = Matrix{T}(I, 3, 3)
      lmul!(Q, expected)
      @test expected == transpose(expected)
      @test copyto!(x, Q) === x
      @test Matrix(x) == expected
      Q = qr(T[1 2 3; 4 5 6; 7 8 10]).Q
      rejects(x -> copyto!(x, Q), x, ArgumentError, "must be symmetric")
      rejects(x -> copyto!(x, adjoint(Q)), x, ArgumentError, "must be symmetric")
      rejects(x -> copyto!(x, (1.0,)), x, ArgumentError, "position by position")
      rejects(x -> (view(x, 1:2, 1) .= 7), x, ArgumentError, "fill! or broadcasting into a view")
      rejects(x -> (view(vec(x), 1:2) .= 7), x, ArgumentError, "fill! or broadcasting into a reshaped view")
    end
  end

  @testset "allocating operations" begin
    for T in DataType[Float64, Float32], p in Any[0, 1, 3]
      M = ones(T, p, p) + p * Matrix{T}(I, p, p)
      x = SymmetricSparseIterate(M)
      before = state(x)
      if p > 0
        @test sort(x; dims = 1) == sort(M; dims = 1)
      end
      @test accumulate(+, x) == accumulate(+, M)
      @test rot180(x) == rot180(M)
      @test map(sin, x) == map(sin, M)
      @test x .+ 1 == M .+ 1
      @test lu(x).L == lu(M).L
      @test lu(x).U == lu(M).U
      @test Matrix(qr(x).Q) ≈ Matrix(qr(M).Q)
      @test qr(x).R ≈ qr(M).R
      @test cholesky(x).L ≈ cholesky(M).L
      @test bunchkaufman(x) \ Matrix{T}(I, p, p) ≈ bunchkaufman(M) \ Matrix{T}(I, p, p)
      @test isequal(state(x), before)
    end
  end

  @testset "other named destinations" begin
    A = [4.0 1; 2 5]
    P = [4.0 1; 1 5]
    for make in Any[copy, sparse]
      check_named_destination(make)
    end
    for f in Any[sum!, prod!, maximum!, minimum!, all!, any!, extrema!]
      a = f === all! || f === any! ? trues(2, 2, 2) : ones(2, 2, 2)
      value = f === extrema! ? (1.0, 1.0) : f === all! || f === any! ? true : 1.0
      r = fill(value, 2, 2)
      @test f(r, a) === r
      @test f(identity, r, a) === r
      @test r == fill(f === sum! ? 2.0 : value, 2, 2)
    end
    r = zeros(Int, 2, 2)
    @test count!(r, trues(2, 2, 2)) === r
    @test count!(identity, r, trues(2, 2, 2)) === r
    @test r == fill(2, 2, 2)
    for f in Any[findmax!, findmin!]
      rv = zeros(2, 2)
      ri = fill(CartesianIndex(0, 0, 0), 2, 2)
      @test f(rv, ri, ones(2, 2, 2)) == (rv, ri)
      @test rv == ones(2, 2)
    end
    @test lu!(copy(A)).U == lu(A).U
    @test qr!(copy(A)).R ≈ qr(A).R
    @test cholesky!(copy(P)).L ≈ cholesky(P).L
    @test bunchkaufman!(copy(P)) \ ones(2) ≈ bunchkaufman(P) \ ones(2)
    Q = qr(A).Q
    r = zeros(2, 2)
    @test copyto!(r, Q) === r
    @test r ≈ Matrix(Q)
  end

end

end
