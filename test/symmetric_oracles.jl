module SymmetricOraclesTest

using Test, ProximalBase, LinearAlgebra

# the state of an owner: its values, its active coordinates and the arrays behind them
state(x) = (Matrix(x), nactive(x), copy(activecoordinates(x)), copy(x.nzval), copy(x.nzval2ind), copy(x.ind2nzval))

const C = [1.0 2.0; 3.0 4.0]                     # not symmetric
const I2 = Matrix(1.0I, 2, 2)
const smooth = ("LinearFunction" => LinearFunction(C), "SqrNormL2" => SqrNormL2(1.0, C),
                "KronQuadratic" => KronQuadratic(I2, I2, C), "SqrHingeLoss" => SqrHingeLoss(C),
                "BernoulliLoss" => BernoulliLoss([0.1 0.2; 0.3 0.4]))

@testset "the parameters are checked in every role of an owner" begin
  for (name, f) in smooth
    @testset "$name" begin
      x = SymmetricSparseIterate(fill(0.25, 2, 2))
      before = state(x)
      # an owner point: the allocating forms and a dense output
      @test_throws ArgumentError gradient(f, x)
      @test_throws ArgumentError value_and_gradient(f, x)
      G = fill(7.0, 2, 2)
      @test_throws ArgumentError gradient!(G, f, x)
      @test_throws ArgumentError value_and_gradient!(G, f, x)
      @test G == fill(7.0, 2, 2)
      @test isequal(state(x), before)
      # an owner output and a dense point
      g = SymmetricSparseIterate(fill(7.0, 2, 2))
      gbefore = state(g)
      @test_throws ArgumentError gradient!(g, f, fill(0.25, 2, 2))
      @test_throws ArgumentError value_and_gradient!(g, f, fill(0.25, 2, 2))
      @test isequal(state(g), gbefore)
    end
  end
  @testset "the Hessian product" begin
    for (name, f) in smooth[1:3]
      H = SymmetricSparseIterate(fill(7.0, 2, 2))
      before = state(H)
      @test_throws ArgumentError hessian_vec!(H, f, ones(2, 2), ones(2, 2))
      @test isequal(state(H), before)
      x = SymmetricSparseIterate(fill(0.25, 2, 2))
      @test_throws ArgumentError hessian_vec(f, x, ones(2, 2))
      Hd = fill(7.0, 2, 2)
      @test_throws ArgumentError hessian_vec!(Hd, f, x, ones(2, 2))
      @test Hd == fill(7.0, 2, 2)
    end
  end
  @testset "the proximal step with asymmetric weights" begin
    f = NormL1(0.1, C)
    x = SymmetricSparseIterate(fill(0.25, 2, 2))
    for op in (prox!, value_and_prox!)
      # a dense output at an owner point
      y = fill(7.0, 2, 2)
      @test_throws ArgumentError op(y, f, x, 0.2)
      @test y == fill(7.0, 2, 2)
      # an owner output at a dense point
      z = SymmetricSparseIterate(fill(7.0, 2, 2))
      before = state(z)
      @test_throws ArgumentError op(z, f, fill(0.25, 2, 2), 0.2)
      @test isequal(state(z), before)
    end
  end
  @testset "symmetric parameters are accepted in the same roles" begin
    S = [1.0 2.0; 2.0 4.0]
    x = SymmetricSparseIterate(fill(0.25, 2, 2))
    @test gradient(LinearFunction(S), x) == S
    @test gradient(SqrNormL2(1.0, S), x) == fill(0.25, 2, 2) - S
    @test Matrix(prox!(SymmetricSparseIterate(2), NormL1(0.1, S), x, 0.2)) ==
          prox(NormL1(0.1, S), fill(0.25, 2, 2), 0.2)
  end
end

@testset "the Hessian of SqrNormL2 is scaled in the model precision" begin
  for (level, entry) in ((1e40, 1e-40), (1e-300, 1e300))
    f = SqrNormL2(level)
    H = SymmetricSparseIterate(Float32, 2)
    x = SymmetricSparseIterate(Float64, 2)
    @test hessian_vec!(H, f, x, fill(entry, 2, 2)) === H
    @test Matrix(H) == fill(1.0f0, 2, 2)
    Hd = zeros(Float32, 2, 2)
    hessian_vec!(Hd, f, x, fill(entry, 2, 2))
    @test Hd == fill(1.0f0, 2, 2)
    # an owner direction
    V = SymmetricSparseIterate(fill(entry, 2, 2))
    H = SymmetricSparseIterate(Float32, 2)
    hessian_vec!(H, f, x, V)
    @test Matrix(H) == fill(1.0f0, 2, 2)
  end
end

@testset "a wider point or parameter meets a narrower owner in the wider precision" begin
  S = [1.0 0.3; 0.3 2.0]
  X = [0.2 0.7; 0.7 0.9]
  P = [0.1 0.4; 0.4 0.8]
  fs = (LinearFunction(S, 0.0), SqrNormL2(0.1, S), SqrNormL2(1e-3, P), KronQuadratic(I2, [2.0 0.5; 0.5 1.0], S),
        SqrHingeLoss(P), BernoulliLoss(P))
  for f in fs
    ref = gradient(f, X)
    ref = (ref + ref') / 2              # the derivative over symmetric matrices
    g = SymmetricSparseIterate(Float32, 2)
    gradient!(g, f, X)
    @test Matrix(g) == Float32.(ref)
    g = SymmetricSparseIterate(Float32, 2)
    @test value_and_gradient!(g, f, SymmetricSparseIterate(X)) ≈ f(X)
    @test Matrix(g) == Float32.(ref)
  end
  # a level that Float32 cannot hold
  g = SymmetricSparseIterate(Float32, 2)
  gradient!(g, SqrNormL2(1e40), fill(1e-40, 2, 2))
  @test Matrix(g) == fill(1.0f0, 2, 2)
end

@testset "a gradient written to an owner keeps its active zeros" begin
  function with_active_zeros()
    g = SymmetricSparseIterate(3)
    for (r, c) in ((1, 1), (2, 1), (3, 3))
      g[r, c] = 1.0
      g[r, c] = 0.0
    end
    @test nactive(g) == 3
    g
  end
  Z = zeros(3, 3)
  for f in (LinearFunction(Z), KronQuadratic(Matrix(1.0I, 3, 3), Matrix(1.0I, 3, 3), Z))
    g = with_active_zeros()
    ids = copy(activecoordinates(g))
    gradient!(g, f, Z)
    @test nactive(g) == 3 && activecoordinates(g) == ids && Matrix(g) == Z
    g = with_active_zeros()
    value_and_gradient!(g, f, Z)
    @test nactive(g) == 3 && activecoordinates(g) == ids && Matrix(g) == Z
  end
  g = with_active_zeros()
  hessian_vec!(g, KronQuadratic(Matrix(1.0I, 3, 3), Matrix(1.0I, 3, 3), Z), Z, Z)
  @test nactive(g) == 3
end

@testset "the projected gradient at the subnormal edge, $T" for T in (Float32, Float64)
  d = nextfloat(zero(T))
  x = T[0 5d; d 0]
  g = SymmetricSparseIterate(T, 2)
  gradient!(g, SqrNormL2(one(T)), x)
  @test g[2, 1] == 3d && g[1, 2] == 3d
  g = SymmetricSparseIterate(T, 2)
  gradient!(g, SqrNormL2(one(T)), T[0 d; 5d 0])
  @test g[2, 1] == 3d
end

end
