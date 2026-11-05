module SymmetricMutatorsTest

using Test, ProximalBase, LinearAlgebra, SparseArrays, Random
using JLArrays: JLArray
using GPUArraysCore: allowscalar

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

@noinline function matches_dense(f, x)
  Base.@nospecialize f x
  M = Matrix(x)
  @test f(M) === M
  @test f(x) === x
  @test Matrix(x) == M
end

@noinline function check_scalar_mul(op, x, A, s, scalars, left)
  Base.@nospecialize op x A s scalars left
  M = Matrix(x)
  if left
    @test op(M, s, A, scalars...) === M
    @test op(x, s, A, scalars...) === x
  else
    @test op(M, A, s, scalars...) === M
    @test op(x, A, s, scalars...) === x
  end
  @test Matrix(x) == M
end

initial(T, p) = fill!(SymmetricSparseIterate(T, p), -1)
symmetric(p) = Float64[r + c for r in 1:p, c in 1:p]

# Matching sizes do not imply matching axes.
struct ShiftedMatrix{T} <: AbstractMatrix{T}
  data::Matrix{T}
end
Base.size(A::ShiftedMatrix) = size(A.data)
Base.axes(A::ShiftedMatrix) = (0:size(A, 1)-1, 0:size(A, 2)-1)
Base.getindex(A::ShiftedMatrix, r::Int, c::Int) = A.data[r+1, c+1]

struct ScalarStyle <: Base.Broadcast.AbstractArrayStyle{0} end

@testset "symmetric mutators" begin
  @testset "indexed arrays" begin
    for T in DataType[Float64, Float32], p in Any[0, 1, 3]
      S = symmetric(p)
      for f in Any[x -> setindex!(x, S, :, :),
                x -> setindex!(x, S, 1:p, collect(1:p)),
                x -> setindex!(x, vec(S), :),
                x -> setindex!(x, vec(S), 1:p^2),
                x -> setindex!(x, S, CartesianIndices(x)),
                x -> setindex!(x, S, collect(CartesianIndices(x)))]
        x = initial(T, p)
        @test f(x) === x
        @test Matrix(x) == S
      end
      x = initial(T, p)
      rejects(x -> setindex!(x, zeros(p^2 + 1), :), x, DimensionMismatch)
      rejects(x -> setindex!(x, zeros(p, p + 1), :, :), x, DimensionMismatch)
      rejects(x -> setindex!(x, vec(S), trues(p, p)), x, ArgumentError)
    end
    for T in DataType[Float64, Float32]
      x = initial(T, 2)
      A = [1.0 1 + 1e-12; 1 4]
      for f in Any[x -> setindex!(x, A, :, :), x -> setindex!(x, A, 1:2, 1:2),
                x -> setindex!(x, vec(A), :), x -> setindex!(x, vec(A), 1:4),
                x -> setindex!(x, A, CartesianIndices(x))]
        rejects(f, x, ArgumentError, "must be symmetric")
      end
      for f in Any[x -> setindex!(x, ones(2, 2), [2, 1], [2, 1]),
                x -> setindex!(x, ones(2, 2), [1, 1], 1:2),
                x -> setindex!(x, [5, 6], 1:2, 1),
                x -> setindex!(x, [5, 6], 1, :),
                x -> setindex!(x, [5, 6], 2:3),
                x -> setindex!(x, [5, 6], [true, true], 1),
                x -> setindex!(x, [5], 1, 1, :),
                x -> setindex!(x, [5], CartesianIndex(1, 1), :),
                x -> setindex!(x, zeros(2, 2, 1), :, :, 1)]
        rejects(f, x, ArgumentError)
      end
      for f in Any[x -> setindex!(x, [5, 6], 4:5),
                x -> setindex!(x, ones(2, 2), [1, 3], 1:2),
                x -> setindex!(x, [5], [CartesianIndex(3, 1)])]
        rejects(f, x, BoundsError)
      end
      rejects(x -> setindex!(x, 5.0, 1:2, 1), x, ArgumentError)
      @test setindex!(x, 7, 1, 2) === x
      @test x[2, 1] == 7
      x[2] = 8
      @test x[1, 2] == 8
      x[CartesianIndex(1, 2)] = 9
      @test x[2, 1] == 9
    end
  end

  @testset "broadcast wrappers" begin
    for T in DataType[Float64, Float32], p in Any[0, 1, 3]
      S = symmetric(p)
      x = initial(T, p)
      for make in Any[x -> view(x, :, :), x -> view(x, 1:p, collect(1:p)),
                   x -> view(x, :), x -> vec(x), x -> reshape(x, 1, p^2),
                   x -> view(reshape(x, 1, p^2), :, :),
                   x -> view(reshape(x, 1, p^2), 1, :)]
        dest = make(x)
        vals = reshape(S, size(dest))
        @test broadcast!(identity, dest, vals) === dest
        @test Matrix(x) == S
        @test broadcast!(identity, dest, 3) === dest
        @test Matrix(x) == fill(3, p, p)
        @test fill!(dest, 4) === dest
        @test Matrix(x) == fill(4, p, p)
        rejects(x -> fill!(make(x), 1 + 2im), x, ArgumentError,
                "SymmetricSparseIterate requires real entries")
        copyto!(x, S)
        dest .+= 1
        @test Matrix(x) == S .+ 1
        dest .*= 2
        @test Matrix(x) == 2 .* (S .+ 1)
      end
      copyto!(x, S)
      x[:, :] .+= 1
      @test Matrix(x) == S .+ 1
      x[:] .+= 1
      @test Matrix(x) == S .+ 2
      vec(x) .*= 2
      @test Matrix(x) == 2 .* (S .+ 2)
      x[:] .= 7
      @test Matrix(x) == fill(7, p, p)
      # Structured and sparse broadcast styles take the same checked route.
      for A in Any[Diagonal(ones(p)), sparse(S)]
        dest = view(x, :, :)
        @test broadcast!(identity, dest, A) === dest
        @test Matrix(x) == Matrix(A)
      end
    end
    @testset "broadcast styles" begin
      x = initial(Float64, 2)
      for make in Any[x -> view(x, :, :), x -> view(x, :), x -> vec(x)]
        dest = make(x)
        for Style in Any[Nothing, ScalarStyle, Base.Broadcast.DefaultArrayStyle{0}]
          bc = Base.Broadcast.Broadcasted{Style}(identity, (Ref(7),), axes(dest))
          result = if Style === Nothing
            copyto!(dest, bc)
          else
            Base.Broadcast.materialize!(Style(), dest, bc)
          end
          @test result === dest
          @test Matrix(x) == fill(7, 2, 2)
        end
      end
    end
    for T in DataType[Float64, Float32]
      x = initial(T, 3)
      for f in Any[x -> (view(x, 1:2, 1) .= 7), x -> (x[1, :] .= [5, 6, 7]),
                x -> (view(x, 1:2, 1:2) .+= 1),
                x -> (view(x, [3, 2, 1], :) .= 7),
                x -> (view(x, [1, 1, 1], :) .= 7),
                x -> (view(vec(x), 2:4) .= 7),
                x -> (view(vec(x), 9:-1:1) .= 7),
                x -> (view(vec(x), [true; falses(8)]) .= 7)]
        rejects(f, x, ArgumentError)
      end
      for make in Any[x -> view(x, 1:2, 1), x -> view(x, 1:2, 1:2),
                   x -> view(vec(x), 2:4)]
        rejects(x -> fill!(make(x), 7), x, ArgumentError)
      end
      x = initial(T, 2)
      rejects(x -> (x .= 1 + 2im), x, ArgumentError,
              "SymmetricSparseIterate requires real entries")
      rejects(x -> (view(x, :, :) .= 1 + 2im), x, ArgumentError,
              "SymmetricSparseIterate requires real entries")
      A = [1.0 1 + 1e-12; 1 4]
      for make in Any[x -> view(x, :, :), x -> view(x, :), x -> vec(x),
                   x -> reshape(x, 1, 4), x -> view(reshape(x, 1, 4), :, :)]
        rejects(x -> broadcast!(identity, make(x), reshape(A, size(make(x)))), x,
                ArgumentError, "must be symmetric")
      end
    end
  end

  @testset "copies through wrappers" begin
    for T in DataType[Float64, Float32], p in Any[0, 1, 3]
      S = symmetric(p)
      A = S + triu(ones(p, p), 1)      # not symmetric for p > 1
      N = copy(S)
      p > 0 && (N[p, 1] = N[1, p] = NaN)
      for make in Any[x -> view(x, :, :), x -> view(x, 1:p, collect(1:p)),
                   x -> view(x, :), x -> vec(x), x -> reshape(x, 1, p^2),
                   x -> view(reshape(x, 1, p^2), :, :),
                   x -> view(reshape(x, 1, p^2), 1, :),
                   x -> view(view(x, :, :), :, :), x -> reshape(vec(x), p, p)]
        x = initial(T, p)
        dest = make(x)
        for src in Any[reshape(S, size(dest)), S, vec(S), sparse(S), Symmetric(A, :L),
                    SymmetricSparseIterate(S), Float32.(S)]
          fill!(x, -1)
          @test copyto!(dest, src) === dest
          @test Matrix(x) == S
        end
        fill!(x, -1)
        @test copyto!(dest, N) === dest
        @test isequal(Matrix(x), N)
        # sources that alias x, including one read in reverse
        copyto!(x, S)
        for src in Any[x, x', view(x, :, :), vec(x)]
          @test copyto!(dest, src) === dest
          @test Matrix(x) == S
        end
        @test copyto!(dest, view(x, p:-1:1, p:-1:1)) === dest
        @test Matrix(x) == rot180(S)
        copyto!(x, S)
        @test copy!(dest, reshape(S, size(dest))) === dest
        @test Matrix(x) == S
        fill!(x, -1)
        if p > 1
          for src in Any[reshape(A, size(dest)), A, vec(A), sparse(A)]
            rejects(x -> copyto!(make(x), src), x, ArgumentError, "must be symmetric")
          end
          rejects(x -> copy!(make(x), reshape(A, size(make(x)))), x, ArgumentError, "must be symmetric")
          rejects(x -> copyto!(make(x), fill(1 + 2im, p, p)), x, ArgumentError,
                  "SymmetricSparseIterate requires real entries")
        end
        for src in Any[ones(p^2 + 1), ones(p + 1, p + 1), zeros(0)]
          length(src) == p^2 && continue
          rejects(x -> copyto!(make(x), src), x, DimensionMismatch, "copyto! source has")
        end
        p > 0 && rejects(x -> copyto!(make(x), ones(p^2 - 1)), x, DimensionMismatch)
        for f in Any[x -> copyto!(make(x), 1, vec(S)), x -> copyto!(make(x), 1, vec(S), 1),
                  x -> copyto!(make(x), 1, vec(S), 1, p^2),
                  x -> copyto!(make(x), 1, (1.0 for _ in 1:p^2)),
                  x -> copyto!(make(x), 1, (1.0 for _ in 1:p^2), 1),
                  x -> copyto!(make(x), 1, (1.0 for _ in 1:p^2), 1, p^2)]
          rejects(f, x, ArgumentError, "copyto! with offsets")
        end
      end
      x = initial(T, p)
      @test copyto!(view(x, :, :), zeros(p^2)) isa SubArray
      @test Matrix(x) == zeros(p, p)
      # a one-dimensional copy! resizes for a source of another length, which a reshape cannot do
      fill!(x, -1)
      rejects(x -> copy!(vec(x), ones(p^2 + 1)), x, Exception)
      rejects(x -> copy!(view(x, :, :), vec(S)), x, ArgumentError, "same axes")
    end
    # An orthogonal factor is formed in a buffer before the checked write.
    for T in DataType[Float64, Float32]
      for p in Any[0, 1, 3]
        Q = qr(Matrix{T}(I, p, p)).Q
        for make in Any[x -> view(x, :, :), x -> view(view(x, :, :), :, :)]
          x = initial(T, p)
          dest = make(x)
          @test copyto!(dest, Q) === dest
          @test Matrix(x) == Matrix{T}(I, p, p)
        end
      end
      Q = qr(reshape(T[0, 1, 0], 3, 1)).Q
      expected = lmul!(Q, Matrix{T}(I, 3, 3))
      x = initial(T, 3)
      @test copyto!(view(x, 1:3, :), Q) isa SubArray
      @test Matrix(x) == expected
      Q = qr(T[1 2 3; 4 5 6; 7 8 10]).Q
      for make in Any[x -> view(x, :, :), x -> view(x, 1:3, collect(1:3)), x -> reshape(vec(x), 3, 3)]
        rejects(x -> copyto!(make(x), Q), x, ArgumentError, "must be symmetric")
      end
      rejects(x -> copyto!(view(x, 1:2, 1:2), Q), x, ArgumentError, "copyto! into a view")
      rejects(x -> copyto!(reshape(x, 1, 9), Q), x, DimensionMismatch)
    end
    @static if isdefined(SparseArrays, :CHOLMOD) && isdefined(SparseArrays.CHOLMOD, :Dense)
      S = symmetric(3)
      for T in DataType[Float64, Float32], make in Any[x -> view(x, :, :), x -> vec(x), x -> reshape(x, 1, 9)]
        x = initial(T, 3)
        dest = make(x)
        @test copyto!(dest, SparseArrays.CHOLMOD.Dense(S)) === dest
        @test Matrix(x) == S
        rejects(x -> copyto!(make(x), SparseArrays.CHOLMOD.Dense(S + triu(ones(3, 3), 1))), x,
                ArgumentError, "must be symmetric")
      end
    end
    # A device source behaves as it does in copyto!(x, A).
    S = symmetric(3)
    for T in DataType[Float64, Float32], src in Any[JLArray(T.(S)), JLArray(vec(T.(S)))]
      allowscalar(false)
      x = initial(T, 3)
      expected = try
        copyto!(x, reshape(src, 3, 3))
        nothing
      catch e
        typeof(e)
      end
      for make in Any[x -> view(x, :, :), x -> vec(x), x -> reshape(x, 1, 9)]
        if expected === nothing
          @test copyto!(make(x), src) isa AbstractArray
          @test Matrix(x) == S
          fill!(x, -1)
        else
          rejects(x -> copyto!(make(x), src), x, expected)
        end
      end
      for make in Any[x -> view(x, :, :), x -> vec(x)]
        fill!(x, -1)
        allowscalar(() -> copyto!(make(x), src))
        @test Matrix(x) == S
      end
    end
    # Partial destinations throw before writing, and dense destinations are unchanged.
    for T in DataType[Float64, Float32]
      x = initial(T, 3)
      for f in Any[x -> copyto!(view(x, 2:2, 1:1), fill(7.0, 1, 1)),
                x -> copyto!(view(x, 1:2, 1), [5.0, 6.0]),
                x -> copyto!(view(x, [3, 2, 1], :), symmetric(3)),
                x -> copyto!(view(x, 1:2, 1:2), sparse([1.0 2; 2 1])),
                x -> copy!(view(x, 1:2, 1:2), [1.0 2; 2 1]),
                x -> copyto!(view(vec(x), 2:4), [1.0, 2, 3]),
                x -> copyto!(view(vec(x), 9:-1:1), vec(symmetric(3))),
                x -> copyto!(view(x, 1, :), 1, [1.0, 2, 3], 1, 3)]
        rejects(f, x, ArgumentError)
      end
    end
    A = [1.0 3; 2 4]
    M = zeros(2, 2)
    @test copyto!(view(M, :, :), A) == A
    @test copyto!(vec(M), vec(2A)) == vec(2A)
    @test M == 2A
    @test copyto!(view(M, 2:2, 1:1), fill(7.0, 1, 1)) == fill(7.0, 1, 1)
    @test M == [2 6; 7 8]
  end

  @testset "transposition and permutation" begin
    for T in DataType[Float64, Float32], p in Any[0, 1, 3]
      S = symmetric(p)
      for f in Any[x -> transpose!(x, S), x -> adjoint!(x, S),
                x -> permutedims!(x, S, (1, 2)), x -> permutedims!(x, S, [1, 2]),
                x -> permutedims!(x, S, (2, 1)), x -> permutedims!(x, S, [2, 1])]
        x = initial(T, p)
        matches_dense(f, x)
      end
    end
    for T in DataType[Float64, Float32]
      x = initial(T, 2)
      A = [1.0 1 + 1e-12; 1 4]
      for f in Any[x -> transpose!(x, A), x -> adjoint!(x, A),
                x -> permutedims!(x, A, (1, 2)), x -> permutedims!(x, A, (2, 1))]
        rejects(f, x, ArgumentError, "must be symmetric")
      end
      for perm in Any[(1,), (1, 1), (0, 2), (2, 3), (1, 2, 3), [2, 2]]
        err = try
          permutedims!(zeros(2, 2), ones(2, 2), perm)
        catch e
          e
        end
        rejects(x -> permutedims!(x, ones(2, 2), perm), x, typeof(err))
      end
      for f in Any[x -> transpose!(x, ones(3, 2)), x -> adjoint!(x, ones(2, 3)),
                x -> transpose!(x, ones(4)), x -> adjoint!(x, ones(4))]
        rejects(f, x, DimensionMismatch)
      end
      rejects(x -> permutedims!(x, ones(4), (1,)), x, MethodError)
      y = initial(T, 1)
      @test transpose!(y, [3.0]) === y
      @test y[1, 1] == 3
      @test adjoint!(y, [4.0]) === y
      @test y[1, 1] == 4
    end
  end

  @testset "scalar multiplication" begin
    for T in DataType[Float64, Float32], p in Any[0, 1, 3], scalars in Any[(), (2, 3), (0, 0)]
      S = symmetric(p)
      for left in Any[true, false]
        check_scalar_mul(mul!, initial(T, p), S, 2.0, scalars, left)
      end
    end
    for T in DataType[Float64, Float32], scalars in Any[(), (2, 3), (0, 0)]
      x = initial(T, 2)
      for A in Any[ones(4), ones(1, 4), ones(2, 3), ShiftedMatrix(ones(2, 2))]
        rejects(x -> mul!(x, A, 2, scalars...), x, DimensionMismatch)
        rejects(x -> mul!(x, 2, A, scalars...), x, DimensionMismatch)
      end
    end
    for T in DataType[Float64, Float32], scalars in Any[(), (1, 1)]
      x = initial(T, 2)
      A = [1.0 1 + 1e-12; 1 4]
      rejects(x -> mul!(x, A, 2, scalars...), x, ArgumentError, "must be symmetric")
      rejects(x -> mul!(x, 2, A, scalars...), x, ArgumentError, "must be symmetric")
    end
  end

  @testset "iterator copies" begin
    for T in DataType[Float64, Float32], p in Any[0, 1, 3]
      x = initial(T, p)
      for src in Any[(1.0, 2.0, 3.0, 4.0), (i * 1.0 for i in 1:4), (1.0,), 1.0]
        for f in Any[x -> copyto!(x, src), x -> copyto!(x, 1, src),
                  x -> copyto!(x, 1, src, 1), x -> copyto!(x, 1, src, 1, 4)]
          rejects(f, x, ArgumentError, "copyto!")
        end
      end
      @test copyto!(x, 2I) === x
      @test Matrix(x) == Matrix(2.0I, p, p)
      S = symmetric(p)
      @test copyto!(x, S) === x
      @test Matrix(x) == S
      @test (x .= 3) === x
      @test Matrix(x) == fill(3, p, p)
      @test (x .= S) === x
      @test Matrix(x) == S
    end
  end

  @testset "other destinations" begin
    A = [1.0 3; 2 4]
    for make in Any[copy, sparse, SparseIterate, M -> view(copy(M), :, :)]
      M = make(zeros(2, 2))
      @test (view(M, :, :) .= A) == A
      @test setindex!(M, [5.0, 6.0], 1, :) === M
      @test Matrix(M) == [5 6; 2 4]
      @test (vec(M) .= [1.0, 2.0, 3.0, 4.0]) == vec(A)
      @test Matrix(M) == A
      @test transpose!(M, A) === M
      @test Matrix(M) == transpose(A)
      @test copyto!(M, (1.0, 2.0, 3.0, 4.0)) === M
      @test Matrix(M) == A
      @test mul!(M, [1.0, 2.0, 3.0, 4.0], 2.0) === M
      @test Matrix(M) == 2A
    end
  end
end

end
