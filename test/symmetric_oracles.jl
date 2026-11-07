module SymmetricOraclesTest

using Test, ProximalBase, LinearAlgebra
using ..TestUtils: allocs

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

# ------------------------------------------------------------------------------------------------
# a kernel that runs over the matrix positions of an owner must read each mirrored pair once and write it once

@testset "a constrained ℓ1 step in place reads each pair once, $T" for T in (Float32, Float64)
  f = Constrained(NormL1(T(1)), IndBallL2(T(10)))
  A = T[1 3; 3 2]
  ref = sign.(A) .* max.(abs.(A) .- T(0.5), 0)
  for point in (:inplace, :view, :separate, :dense)
    y = SymmetricSparseIterate(A)
    x = point == :inplace ? y : point == :view ? matrixview(y) : point == :separate ? SymmetricSparseIterate(A) : A
    v = value_and_prox!(y, f, x, T(0.5))
    @test Matrix(y) == ref
    @test v == sum(abs, ref)
    y = SymmetricSparseIterate(A)
    x = point == :inplace ? y : point == :view ? matrixview(y) : point == :separate ? SymmetricSparseIterate(A) : A
    @test Matrix(prox!(y, f, x, T(0.5))) == ref
  end
  y = SymmetricSparseIterate(A)
  @test allocs(value_and_prox!, y, f, y, T(0.5)) == 0
end

@testset "IndBallL0 at every kind of point, $T" for T in (Float32, Float64)
  A = T[0 3; 3 0]
  for f in (IndBallL0(1), IndBallL0(1; diagonal = false))
    for kind in (:owner, :inplace, :view, :dense, :symmetric)
      y = SymmetricSparseIterate(A)
      x = kind == :owner ? SymmetricSparseIterate(A) : kind == :inplace ? y : kind == :view ? matrixview(y) :
          kind == :dense ? A : Symmetric(A, :L)
      @test Matrix(prox!(y, f, x, one(T))) == A
      y = SymmetricSparseIterate(A)
      x = kind == :owner ? SymmetricSparseIterate(A) : kind == :inplace ? y : kind == :view ? matrixview(y) :
          kind == :dense ? A : Symmetric(A, :L)
      @test value_and_prox!(y, f, x, one(T)) == 0
      @test Matrix(y) == A
    end
  end
  # a ranking in which the off-diagonal counts √2, over a point that is not symmetric
  B = T[1 2 0; 2 3 0.5; 0 0.5 1.2]
  ref = Matrix(prox!(SymmetricSparseIterate(T, 3), IndBallL0(2), SymmetricSparseIterate(B), one(T)))
  y = SymmetricSparseIterate(T, 3)
  @test Matrix(prox!(y, IndBallL0(2), B, one(T))) == ref
  y = SymmetricSparseIterate(T, 3)
  @test Matrix(prox!(y, IndBallL0(2), B + T[0 0.5 0; -0.5 0 0; 0 0 0], one(T))) == ref
end

@testset "a proximal value write keeps the active coordinates of the spectral steps, $T" for T in (Float32, Float64)
  function with_active_zeros()
    y = SymmetricSparseIterate(T, 2)
    fill!(coordinates(y), T(1))
    fill!(coordinates(y), T(0))
    y
  end
  for (f, X) in ((IndBallRank(1), zeros(T, 2, 2)), (IndBallRank(1), T[2 0; 0 0]), (NuclearNorm(T(0.5)), T[2 0; 0 0]),
                 (IndPSD(), zeros(T, 2, 2)), (TracePSD(T(0.5)), zeros(T, 2, 2)))
    y = with_active_zeros()
    ids = copy(activecoordinates(y))
    prox!(y, f, X, one(T))
    @test nactive(y) == 3 && activecoordinates(y) == ids
    y = with_active_zeros()
    value_and_prox!(y, f, X, one(T))
    @test nactive(y) == 3 && activecoordinates(y) == ids
  end
end

@testset "IndToeplitz at a symmetric dense point, $T" for T in (Float32, Float64)
  y = SymmetricSparseIterate(T, 1)
  @test Matrix(prox!(y, IndToeplitz(), reshape(T[2], 1, 1), one(T))) == reshape(T[2], 1, 1)
  S = T[2 1; 1 3]
  ref = prox(IndToeplitz(symmetric = true), S)
  for f in (IndToeplitz(), IndToeplitz(symmetric = true))
    y = SymmetricSparseIterate(T, 2)
    @test Matrix(prox!(y, f, S, one(T))) == ref
    y = SymmetricSparseIterate(T, 2)
    @test Matrix(prox!(y, f, Symmetric(S, :L), one(T))) == ref
    y = SymmetricSparseIterate(T, 2)
    @test Matrix(prox!(y, f, SymmetricSparseIterate(S), one(T))) == ref
  end
  # a point that is not symmetric, with a set that is not, has a result that an owner cannot hold
  @test_throws ArgumentError prox!(SymmetricSparseIterate(T, 2), IndToeplitz(), T[1 2; 3 4], one(T))
end

# ------------------------------------------------------------------------------------------------
# the families of the previous step, through the writable matrix of the owner

function audit_catalog(::Type{T}) where {T}
  groups = [[1], [2, 4], [3, 7], [5], [6, 8], [9]]      # closed under transposition, for p = 3
  S = T[1 2 3; 2 4 5; 3 5 6]
  a = T[1 0.5 -1; 0.5 2 0.25; -1 0.25 1.5]
  Any[
    ("NuclearNorm", NuclearNorm(T(0.4)), T(1)), ("IndBallRank", IndBallRank(2), T(1)), ("IndPSD", IndPSD(), T(1)),
    ("IndPSD rank", IndPSD(rank = 2), T(1)), ("TracePSD", TracePSD(T(0.4)), T(1)),
    ("SingularValue", SingularValue(NormL1(T(0.4))), T(1)), ("Spectral", Spectral(NormL1(T(0.4))), T(1)),
    ("NormL2", NormL2(T(0.3)), T(0.7)),
    ("GroupNormL2", GroupNormL2(T(0.3), groups), T(0.7)), ("GroupNormLinf", GroupNormLinf(T(0.3), groups), T(0.7)),
    ("SparseGroupNormL2", SparseGroupNormL2(T(0.2), T(0.3), groups), T(0.7)),
    ("GroupMCP", GroupMCP(T(0.3), groups; a = T(2)), T(0.7)), ("GroupSCAD", GroupSCAD(T(0.3), groups; a = T(3.7)), T(0.7)),
    ("ball NormL1", Constrained(NormL1(T(0.3)), IndBallL2(T(2))), T(0.7)),
    ("ball NormL1 inactive", Constrained(NormL1(T(0.3)), IndBallL2(T(20))), T(0.7)),
    ("sphere NormL1", Constrained(NormL1(T(0.3)), IndSphereL2(T(2))), T(0.7)),
    ("ball GroupNormL2", Constrained(GroupNormL2(T(0.3), groups), IndBallL2(T(2))), T(0.7)),
    ("sphere GroupNormL2", Constrained(GroupNormL2(T(0.3), groups), IndSphereL2(T(2))), T(0.7)),
    ("IndBallL1", IndBallL1(T(3)), T(1)), ("IndBallL2", IndBallL2(T(3)), T(1)), ("IndSphereL2", IndSphereL2(T(3)), T(1)),
    ("IndSimplex", IndSimplex(T(10); equality = false), T(1)),
    ("IndBallL0", IndBallL0(3), T(1)), ("IndBallL0 free diagonal", IndBallL0(2; diagonal = false), T(1)),
    ("IndHyperplane", IndHyperplane(a, T(0.5)), T(1)), ("IndHalfspace", IndHalfspace(a, T(0.5)), T(1)),
    ("IndHyperslab", IndHyperslab(a, T(-0.5), T(0.5)), T(1)),
    ("IndToeplitz", IndToeplitz(), T(1)), ("IndToeplitz symmetric", IndToeplitz(symmetric = true), T(1)),
  ]
end

dense_reference(f, X, γ) = prox(f, X, γ)
dense_reference(f::IndBallL0, X, γ) = prox(f, Symmetric(X, :L), γ)

@testset "$name at every kind of point, $T" for T in (Float32, Float64), (name, f, γ) in audit_catalog(T)
  X = T[1.5 -0.8 0.3; -0.8 0.6 -1.1; 0.3 -1.1 2.2]
  ref = Matrix(dense_reference(f, X, γ))
  ref = (ref + ref') / 2
  rtol = T == Float32 ? 2e-4 : 1e-10
  atol = T == Float32 ? 2e-4 : 1e-10
  for kind in (:inplace, :view, :owner, :dense, :symmetric)
    y = SymmetricSparseIterate(X)
    x = kind == :inplace ? y : kind == :view ? matrixview(y) : kind == :owner ? SymmetricSparseIterate(X) :
        kind == :dense ? X : Symmetric(X, :L)
    kind == :owner && (y = SymmetricSparseIterate(T, 3))
    @test prox!(y, f, x, γ) === y
    @test isapprox(Matrix(y), ref; rtol, atol)
    y = SymmetricSparseIterate(X)
    x = kind == :inplace ? y : kind == :view ? matrixview(y) : kind == :owner ? SymmetricSparseIterate(X) :
        kind == :dense ? X : Symmetric(X, :L)
    kind == :owner && (y = SymmetricSparseIterate(T, 3))
    # the value of IndToeplitz at an owner is not defined, so the joint call has no owner destination
    f isa IndToeplitz && continue
    value_and_prox!(y, f, x, γ)
    @test isapprox(Matrix(y), ref; rtol, atol)
  end
end

@testset "an owner destination keeps its active coordinates at an all-zero result, $T" for T in (Float32, Float64)
  function with_active_zeros()
    y = SymmetricSparseIterate(T, 3)
    fill!(coordinates(y), T(1))
    fill!(coordinates(y), T(0))
    y
  end
  Z = zeros(T, 3, 3)
  for (name, f, γ) in audit_catalog(T)
    # the sphere and the hyperplane have no all-zero projection of a zero point
    (occursin("sphere", lowercase(name)) || name == "IndHyperplane") && continue
    y = with_active_zeros()
    ids = copy(activecoordinates(y))
    try
      prox!(y, f, Z, γ)
    catch err
      err isa ArgumentError || rethrow()
      continue
    end
    @test nactive(y) == length(ids)
    @test activecoordinates(y) == ids
  end
end

end
