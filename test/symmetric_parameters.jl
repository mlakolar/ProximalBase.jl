module SymmetricParametersTest

using Test, ProximalBase, LinearAlgebra, SparseArrays, Random

Random.seed!(191)

const ASYMMETRIC = "must be symmetric"

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

# Functions that write a parameter, or a product computed in floating point,
# into a SymmetricSparseIterate output.
@testset "Functions writing into a SymmetricSparseIterate" begin
  @testset "functions that write a parameter" begin
    A = [1.0 3.0; 2.0 4.0]
    xs = SymmetricSparseIterate([2.0 1.0; 1.0 2.0])
    for f in (LinearFunction(A), Tilt(SqrNormL2(), A))
      rejects(g -> gradient!(g, f, xs), fill!(SymmetricSparseIterate(2), -1), ArgumentError, ASYMMETRIC)
      rejects(y -> prox!(y, f, xs, 1.0), fill!(SymmetricSparseIterate(2), -1), ArgumentError, ASYMMETRIC)
    end
    @test Matrix(gradient!(SymmetricSparseIterate(2), LinearFunction(Symmetric(A, :L)), xs)) == [1 2; 2 4]
    rejects(g -> gradient!(g, LinearFunction(vec(A)), xs), SymmetricSparseIterate(2), DimensionMismatch)
    # a block sum checks every block before it writes one
    y = (fill!(SymmetricSparseIterate(2), -1), fill!(SymmetricSparseIterate(2), -1))
    for f in (Tilt(NormL1(1.0), A), Tilt(SqrNormL2(), A), LinearFunction(A), SqrNormL2(1.0, A),
              Translate(NormL1(1.0), A), Translate(SqrNormL2(), A), Precompose(SqrNormL2(), A))
      h = SeparableSum(SqrNormL2(), f)
      for op in (issmooth(f) ? (prox!, gradient!) : (prox!,))
        err = try
          op === prox! ? prox!(y, h, (copy(xs), copy(xs)), 1.0) : gradient!(y, h, (copy(xs), copy(xs)))
          nothing
        catch e
          e
        end
        @test err isa ArgumentError
        @test Matrix(y[1]) == Matrix(y[2]) == fill(-1, 2, 2)
      end
    end
    # entrywise steps are written too
    rejects(y -> prox!(y, Tilt(NormL1(1.0), [1.0 2.0; 2.0 1.0]), xs, A), fill!(SymmetricSparseIterate(2), -1),
            ArgumentError, ASYMMETRIC)
  end

  @testset "SVD-based prox of a symmetric matrix" begin
    B = randn(6, 6); B = B + B'
    for f in (() -> SingularValue(NormL1(0.4)), () -> NuclearNorm(0.4), () -> NuclearNorm(0.4; partial = true))
      Y = prox!(zeros(6, 6), f(), B, 1.0)
      S = prox!(SymmetricSparseIterate(6), f(), B, 1.0)
      @test Matrix(S) == Matrix(Symmetric(Y, :L))
    end
  end
end

end
