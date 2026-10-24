module CacheTest

using Test, ProximalBase, LinearAlgebra, Random
using ..TestUtils

Random.seed!(32)

@testset "least-squares cache lifecycle" begin
  for T in (Float32, Float64), (n, p) in ((16, 5), (5, 16)), m in (1, 3)
    X = randn(T, n, p)
    Y = m == 1 ? randn(T, n) : randn(T, n, m)
    v = m == 1 ? randn(T, p) : randn(T, p, m)
    f = LeastSquaresLoss(Y, X)
    prox(f, v, T(0.7))
    @test f.svd_ready && isempty(f.XtY)
    X .+= T(0.2) .* randn(T, n, p)
    Y .+= one(T)
    @test reset_cache!(f) === f
    @test !f.svd_ready && isempty(f.XtY)
    @test prox(f, v, T(0.7)) ≈ prox(LeastSquaresLoss(Y, X), v, T(0.7))
    out = similar(v)
    @test allocs(prox!, out, f, v, T(0.7)) == 0
  end
  Y = zeros(3)
  f = LeastSquaresLoss(Y, ones(3, 2))
  resize!(Y, 4)
  @test_throws DimensionMismatch reset_cache!(f)
end

@testset "quadratic and constrained caches" begin
  for T in (Float32, Float64)
    X, Y, v = randn(T, 16, 5), randn(T, 16), randn(T, 5)
    A, b = X'X, randn(T, 5)
    f = QuadraticFunction(A, b)
    prox(f, v, T(0.7))
    for i in axes(A, 1)
      A[i, i] += T(0.5)
    end
    b .+= one(T)
    @test reset_cache!(f) === f
    @test isempty(f.Λ)
    @test prox(f, v, T(0.7)) ≈ prox(QuadraticFunction(A, b), v, T(0.7))
    C = IndAffine(randn(T, 2, 5), zeros(T, 2))
    @test_throws ArgumentError reset_cache!(C)
    for loss in (f, LeastSquaresLoss(Y, X))
      c = Constrained(loss, C)
      prox(c, v, T(0.7))
      if loss isa LeastSquaresLoss
        @test loss.XtY ≈ X'Y
        X .*= T(1.2)
        Y .+= T(0.3)
      else
        A .*= T(1.2)
      end
      @test reset_cache!(c) === c
      @test isempty(c.t) && isnan(c.γ)
      fresh = loss isa LeastSquaresLoss ? LeastSquaresLoss(Y, X) : QuadraticFunction(A, b)
      @test prox(c, v, T(0.7)) ≈ prox(Constrained(fresh, C), v, T(0.7))
      @test allocs(prox!, similar(v), c, v, T(0.7)) == 0
    end
    resize!(b, 6)
    @test_throws DimensionMismatch reset_cache!(f)
  end
end

@testset "reset the outermost function" begin
  X, Y, v = randn(12, 4), randn(12), randn(4)
  for wrap in (f -> Scaled(f, 2.), f -> Tilt(f, ones(4)),
               f -> Translate(f, ones(4)), f -> Regularize(f, 0.5),
               f -> Precompose(f, -2.), f -> MoreauEnvelope(f, 0.4), f -> Conjugate(f))
    f = LeastSquaresLoss(Y, X)
    outer = wrap(f)
    prox(outer, v, 0.7)
    X .*= 1.01
    @test reset_cache!(outer) === outer
    @test !f.svd_ready
    @test prox(outer, v, 0.7) ≈ prox(wrap(LeastSquaresLoss(Y, X)), v, 0.7)
  end
  for wrap in (f -> Sum(f, SqrNormL2()), f -> SeparableSum(f, SqrNormL2()),
               f -> SliceSum(f, f))
    f = LeastSquaresLoss(Y, X)
    prox(f, v, 0.7)
    outer = wrap(f)
    @test reset_cache!(outer) === outer
    @test !f.svd_ready
  end
  for kind in (:nothing, :same, :other)
    f, d = LeastSquaresLoss(Y, X), LeastSquaresLoss(Y, X)
    prox(f, v, 0.7); prox(d, v, 0.7)
    outer = FiberSum(f; diagonal = kind === :nothing ? nothing : kind === :same ? f : d)
    @test reset_cache!(outer) === outer
    @test !f.svd_ready
    @test (kind === :other) == !d.svd_ready
  end
  @test reset_cache!(NormL1()) isa NormL1
  upper = ones(4)
  c = IndIntersection(IndBox(-ones(4), upper), IndBallL2(1.); warmstart = true)
  prox(c, v, 0.7)
  upper .= 0.1
  @test reset_cache!(c) === c
  @test c.buffers === nothing && c.iterations == 0 && !c.converged
  @test prox(c, v, 0.7) ≈ prox(IndIntersection(IndBox(-ones(4), upper), IndBallL2(1.)), v, 0.7)
end

end
