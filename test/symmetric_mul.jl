module SymmetricMulTest

using Test, ProximalBase, LinearAlgebra, SparseArrays

state(x) = (Matrix(x), nnz(x), copy(activecoordinates(x)), x.nzval[1:nnz(x)])

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

# the dense reference accumulates before converting to the destination type
@noinline function check_mul(x, A, B, scalars...)
  Base.@nospecialize x A B scalars
  α, β = isempty(scalars) ? (true, false) : scalars
  R = Matrix(x)
  mul!(R, A, B, α, β)
  @test ProximalBase._issymmetric_exact(R, ProximalBase._symequal)
  @test mul!(x, A, B, scalars...) === x
  @test isequal(Matrix(x), R) || Matrix(x) == R
  x
end

@testset "mul! into a SymmetricSparseIterate" begin
  @testset "issue calls and scalars, $T" for T in (Float64, Float32)
    X = T[1 2 0; 0 1 1; 1 0 1; 2 1 1]
    for A in (Matrix(X'), X', transpose(X)), scalars in ((), (1, 1), (1, 2), (2, 0), (-1, 2), (true, false), (1.0, 0.0))
      check_mul(fill!(SymmetricSparseIterate(T, 3), 1), A, X, scalars...)
    end
  end

  @testset "zero scalars" begin
    for β in (0, 1, 2)
      x = SymmetricSparseIterate([1.0 0; 0 3])
      activate!(x, tril_index(2, 2, 1))
      before = state(x)
      R = Matrix(x)
      mul!(R, fill(NaN, 2, 2), ones(2, 2), 0.0, β)
      @test mul!(x, fill(NaN, 2, 2), ones(2, 2), 0.0, β) === x
      @test Matrix(x) == R
      if β == 0
        @test state(x) == state(fill!(SymmetricSparseIterate([1.0 0; 0 3]), 0))
      elseif β == 1
        @test state(x) == before
      end
    end
    check_mul(fill!(SymmetricSparseIterate(2), NaN), ones(2, 2), ones(2, 2), 1, 0)
  end

  @testset "sizes" begin
    for p in (0, 1, 2, 3, 5, 40)
      X = reshape(Float64.(mod.(1:3p, 3)), 3, p)
      check_mul(SymmetricSparseIterate(p), X', X)
      check_mul(SymmetricSparseIterate(p), zeros(p, 0), zeros(0, p))
    end
    for (A, B) in ((ones(3, 2), ones(2, 2)), (ones(2, 2), ones(2, 3)), (ones(2, 3), ones(2, 2)))
      for α in (0, 1)
        rejects(x -> mul!(x, A, B, α, 0), fill!(SymmetricSparseIterate(2), 1), DimensionMismatch)
      end
    end
  end

  @testset "rejected products and equality" begin
    A = [1.0 2; 2 3]
    D = Diagonal([1.0, 2])
    rejects(x -> mul!(x, Symmetric(A), Symmetric(Matrix(D))), SymmetricSparseIterate(A), ArgumentError,
            "entry (2, 1) is 2.0 and entry (1, 2) is 4.0")
    rejects(x -> mul!(x, x, D), SymmetricSparseIterate(A), ArgumentError)
    # diagonal multiplication preserves the NaN and signed-zero pairs
    I2 = Diagonal(ones(2))
    rejects(x -> mul!(x, [1.0 NaN; 0 1], I2), SymmetricSparseIterate(A), ArgumentError)
    check_mul(SymmetricSparseIterate(2), [1.0 NaN; NaN 1], I2)
    check_mul(SymmetricSparseIterate(2), [1.0 -0.0; 0.0 1], I2)
  end

  @testset "precision" begin
    A = Float32[1e8 1 -1e8; 1e8 1 -1e8]
    for (α, β) in ((1.0, 0.0), (true, false))
      x = check_mul(SymmetricSparseIterate(Float32, 2), A, ones(Float32, 3, 2), α, β)
      @test iszero(Matrix(x))
    end
    x = check_mul(SymmetricSparseIterate(Float64, 2), A, ones(Float32, 3, 2))
    @test Matrix(x) == ones(2, 2)
    check_mul(SymmetricSparseIterate(ones(Int, 2, 2)), Diagonal(ones(Int, 2)),
              ones(Int, 2, 2), 0.5, 0.5)
    # a conversion failure at the final write may leave a partial write
    @test_throws InexactError mul!(SymmetricSparseIterate(ones(Int, 2, 2)),
                                  Diagonal(ones(Int, 2)), ones(Int, 2, 2), 0.5, 0)
  end

  @testset "uniform scaling" begin
    A = [1.0 2; 0 1]
    M = [1.0 2; 2 3]
    for (L, R) in ((A, I), (I, A))
      rejects(x -> mul!(x, L, R, 1, 1),
              fill!(SymmetricSparseIterate(2), 1e20), ArgumentError)
    end
    for (L, R) in ((M, 2I), (2I, M))
      check_mul(fill!(SymmetricSparseIterate(2), 1), L, R, 2, 3)
      check_mul(fill!(SymmetricSparseIterate(2), NaN), L, R, 1, 0)
    end
  end

  @testset "operands and aliasing" begin
    M = [1.0 2; 2 3]
    I2 = Matrix{Float64}(I, 2, 2)
    for A in (Diagonal([1.0, 2]), Symmetric(M), sparse(M), M', transpose(M), SymmetricSparseIterate(M))
      check_mul(SymmetricSparseIterate(2), A, I2)
      check_mul(SymmetricSparseIterate(2), I2, A)
    end
    v = [1.0, 2]
    check_mul(SymmetricSparseIterate(2), v, v')
    for scalars in ((), (2, 1))
      x = SymmetricSparseIterate(M)
      check_mul(x, x, I2, scalars...)
      x = SymmetricSparseIterate(M)
      check_mul(x, I2, x, scalars...)
      x = SymmetricSparseIterate(M)
      check_mul(x, x, x, scalars...)
    end
  end

  @testset "structured dispatch resolvers" begin
    M = [1.0 2; 2 3]
    I2 = Matrix{Float64}(I, 2, 2)
    U = UpperTriangular(I2)
    D = Diagonal([1.0, 2])
    S = SymmetricSparseIterate(M)
    # each three-argument resolver, including the package operand methods
    for (A, B) in ((U, [2.0]), (U, M), (M, U), (U, U), (U, D), (D, U), (S, M), (S, U))
      if B isa AbstractVector
        rejects(x -> mul!(x, A, B), SymmetricSparseIterate(2), DimensionMismatch)
      else
        check_mul(SymmetricSparseIterate(2), A, B)
      end
    end
    # each five-argument resolver
    for (A, B) in ((U, [2.0]), (U, M), (M, U), (U, U), (U, D), (D, U),
                   (D, [2.0]), (D, I2), (I2, D), (D, D))
      if B isa AbstractVector
        rejects(x -> mul!(x, A, B, 1, 0), SymmetricSparseIterate(2), DimensionMismatch)
      else
        check_mul(SymmetricSparseIterate(2), A, B, 2, 1)
      end
    end
  end

  @testset "destination ambiguities" begin
    ambiguities = filter(Test.detect_ambiguities(ProximalBase, LinearAlgebra; recursive = true)) do (a, b)
      a.name === :mul! && b.name === :mul! &&
        typeintersect(typeintersect(a.sig, b.sig),
                      Tuple{typeof(mul!), SymmetricSparseIterate, Vararg{Any}}) !== Union{}
    end
    @test isempty(ambiguities)
  end
end

end
