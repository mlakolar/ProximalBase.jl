module SymmetricWrappersTest

# The wrappers (Scaled, Tilt, Translate, Regularize, Precompose, MoreauEnvelope, SqrDistL2, DistL2, Conjugate,
# BoxConstrained, Sum) and the proximal operators of ZeroFunction, CubeNormL2, SumLargest and IndBallGroupL0 on a
# SymmetricSparseIterate, against the dense computation on the matrix that it represents.

using Test, ProximalBase, LinearAlgebra, Random
using ..TestUtils: allocs

const PB = ProximalBase
state(x) = (Matrix(x), nactive(x), copy(activecoordinates(x)), copy(x.nzval), copy(x.nzval2ind), copy(x.ind2nzval))

const X = [1.5 -0.8 0.3; -0.8 0.6 -1.1; 0.3 -1.1 2.2]
const C = [1.0 0.5 -1.0; 0.5 2.0 0.25; -1.0 0.25 1.5]
const B = [0.1 0.2 0.3; 0.2 0.4 0.5; 0.3 0.5 0.6]
const A = [1.5 0.5 -1.0; 0.5 2.0 0.25; -1.0 0.25 -1.25]
const GROUPS = [[1], [2, 4], [3, 7], [5], [6, 8], [9]]      # closed under transposition, for p = 3

function proxcases(::Type{T}) where {T}
  Any[
    "Scaled" => Scaled(NormL1(T(0.3)), T(2)),
    "Tilt" => Tilt(NormL1(T(0.3)), T.(C), T(0.1)),
    "Translate" => Translate(NormL1(T(0.3)), T.(B)),
    "Regularize" => Regularize(NormL1(T(0.3)), T(0.5), T.(C)),
    "Precompose scalar" => Precompose(NormL1(T(0.3)), T(2)),
    "Precompose array" => Precompose(NormL1(T(0.3)), T.(A)),
    "MoreauEnvelope" => MoreauEnvelope(NormL1(T(0.3)), T(0.5)),
    "SqrDistL2" => SqrDistL2(IndBox(-T(1), T(1)), T(0.7)),
    "DistL2" => DistL2(IndBox(-T(1), T(1)), T(0.7)),
    "BoxConstrained" => Constrained(NormL1(T(0.3)), IndBox(-T(1), T(1))),
    "Conjugate" => Conjugate(NormL1(T(0.3))),
    "Conjugate of SumLargest" => Conjugate(SumLargest(2, T(0.3))),
    "Conjugate of a conjugate" => Conjugate(Conjugate(NormL1(T(0.3)))),
    "Scaled Tilt Translate" => Scaled(Tilt(Translate(NormL1(T(0.3)), T.(B)), T.(C), T(0.1)), T(1.5)),
    "ZeroFunction" => ZeroFunction(),
    "CubeNormL2" => CubeNormL2(T(0.3)),
    "SumLargest" => SumLargest(2, T(0.3)),
    "IndBallGroupL0" => IndBallGroupL0(2, GROUPS),
  ]
end

point(kind, T, x) = kind == :dense ? T.(X) : kind == :symmetric ? Symmetric(T.(X), :L) : kind == :owner ? SymmetricSparseIterate(T.(X)) :
                    kind == :view ? matrixview(x) : x

@testset "$name at every kind of point, $T" for T in (Float32, Float64), (name, f) in proxcases(T)
  rtol = T == Float32 ? 2e-4 : 1e-10
  ref = Matrix(prox(f, T.(X), T(0.7)))
  fy = try f(ref) catch e; e isa ArgumentError ? nothing : rethrow() end
  for kind in (:dense, :symmetric, :owner, :inplace, :view)
    y = kind in (:inplace, :view) ? SymmetricSparseIterate(T.(X)) : SymmetricSparseIterate(T, 3)
    x = kind == :inplace ? y : point(kind, T, y)
    @test prox!(y, f, x, T(0.7)) === y
    @test isapprox(Matrix(y), ref; rtol, atol = rtol)
    y = kind in (:inplace, :view) ? SymmetricSparseIterate(T.(X)) : SymmetricSparseIterate(T, 3)
    x = kind == :inplace ? y : point(kind, T, y)
    v = value_and_prox!(y, f, x, T(0.7))
    @test isapprox(Matrix(y), ref; rtol, atol = rtol)
    fy === nothing || @test isapprox(v, fy; rtol = 10rtol, atol = 10rtol)
  end
  # an owner point with a dense output (not defined for the families whose proximal operator is an owner's only), and
  # the allocating form
  if !(f isa Union{ZeroFunction, CubeNormL2, SumLargest, IndBallGroupL0})
    z = zeros(T, 3, 3)
    @test prox!(z, f, SymmetricSparseIterate(T.(X)), T(0.7)) === z
    @test isapprox(z, ref; rtol, atol = rtol)
  end
  p = prox(f, SymmetricSparseIterate(T.(X)), T(0.7))
  @test p isa SymmetricSparseIterate && isapprox(Matrix(p), ref; rtol, atol = rtol)
  # an array step is not supported at an owner, which is left as it was
  y = SymmetricSparseIterate(T.(X)); before = state(y)
  @test_throws ArgumentError prox!(y, f, T.(X), fill(T(0.7), 3, 3))
  @test isequal(state(y), before)
end

@testset "a Float32 owner takes a Float64 point through the wrappers" begin
  for (name, f) in proxcases(Float64)
    name in ("ZeroFunction", "CubeNormL2", "SumLargest", "IndBallGroupL0") && continue
    y = SymmetricSparseIterate(Float32, 3)
    d = zeros(Float32, 3, 3)
    prox!(y, f, X, 0.7)
    prox!(d, f, X, 0.7)
    @test Matrix(y) == d
  end
end

function smoothcases(::Type{T}) where {T}
  P = Matrix{T}(I, 3, 3)
  Any[
    "Scaled" => Scaled(SqrNormL2(T(2), T.(C)), T(3)),
    "Tilt" => Tilt(SqrNormL2(T(1.5)), T.(C), T(0.2)),
    "Translate" => Translate(SqrNormL2(T(1.5)), T.(B)),
    "Regularize" => Regularize(LinearFunction(T.(C)), T(0.5), T.(B)),
    "Precompose scalar" => Precompose(SqrNormL2(T(1.5)), -T(2)),
    "Precompose array" => Precompose(SqrNormL2(T(1.5)), T.(A)),
    "Sum" => Sum(SqrNormL2(T(1.5)), LinearFunction(T.(C)), KronQuadratic(P, P, T.(B))),
    "Scaled Sum" => Scaled(Sum(SqrNormL2(T(1.5)), Tilt(SqrNormL2(T(0.5)), T.(C))), T(0.5)),
  ]
end

@testset "$name: value, gradient and second order at an owner, $T" for T in (Float32, Float64), (name, f) in smoothcases(T)
  rtol = T == Float32 ? 2e-4 : 1e-10
  x = SymmetricSparseIterate(T.(X))
  Xd = T.(X)
  @test isapprox(f(x), f(Xd); rtol)
  g = gradient(f, Xd)
  @test isapprox(gradient(f, x), g; rtol)
  @test gradient(f, x) isa Matrix && issymmetric(gradient(f, x))
  v, gv = value_and_gradient(f, x)
  @test isapprox(v, f(Xd); rtol) && isapprox(gv, g; rtol)
  # an owner output takes the symmetric part, and keeps the coordinates that it has
  out = SymmetricSparseIterate(T, 3)
  @test gradient!(out, f, x) === out
  @test isapprox(Matrix(out), g; rtol)
  out = SymmetricSparseIterate(T, 3)
  @test isapprox(value_and_gradient!(out, f, Xd), f(Xd); rtol)
  @test isapprox(Matrix(out), g; rtol)
  # a point that is not symmetric: the symmetric part of the gradient at it
  Xa = Xd + T.([0 0.5 0; -0.5 0 0.25; 0 -0.25 0])
  out = SymmetricSparseIterate(T, 3)
  gradient!(out, f, Xa)
  ga = gradient(f, Xa)
  @test isapprox(Matrix(out), (ga + ga') / 2; rtol, atol = rtol)
  # the aliased output is refused
  @test_throws ArgumentError gradient!(x, f, x)
  # the product of the Hessian with a direction
  V = T.([0.3 -0.2 0.1; -0.2 0.5 0.4; 0.1 0.4 -0.6])
  if hashessian(f)
    H = hessian_vec(f, Xd, V)
    @test isapprox(hessian_vec(f, x, SymmetricSparseIterate(V)), H; rtol)
    Hout = SymmetricSparseIterate(T, 3)
    @test hessian_vec!(Hout, f, x, V) === Hout
    @test isapprox(Matrix(Hout), (H + H') / 2; rtol, atol = rtol)
  end
  # the constants of the coordinates
  L = SymmetricSparseIterate(T, 3)
  Ld = zeros(T, 3, 3)
  coordinate_lipschitz!(Ld, f)
  @test coordinate_lipschitz!(L, f) === L
  @test isapprox(Matrix(L), Ld; rtol)
end

@testset "Regularize adds its level times the direction in the model precision" begin
  # a Float64 level that a Float32 model cannot hold, against the exact product rounded once to the output
  for ρ in (1e40, 1e-40, 3.0), (TH, Tx, Tv) in ((Float32, Float64, Float64), (Float32, Float32, Float64),
                                                (Float64, Float32, Float32), (Float32, Float32, Float32))
    f = Regularize(LinearFunction(zeros(2, 2)), ρ)
    H = SymmetricSparseIterate(TH, 2)
    x = SymmetricSparseIterate(Tx, 2)
    v = fill(Tv(1e-30), 2, 2)
    hessian_vec!(H, f, x, v)
    W = promote_type(TH, Tx, Tv)
    level = isfinite(W(ρ)) && W(ρ) >= floatmin(W) ? W(ρ) : ρ
    ref = setprecision(BigFloat, 512) do
      TH(BigFloat(level) * BigFloat(v[1]))
    end
    @test Matrix(H) == fill(ref, 2, 2)
  end
end

@testset "rejection before any write, in the wrapper tree" begin
  W = ones(3, 3); W[2, 1] = 2.0                  # not symmetric
  inner = NormL1(0.1, W)
  wrappers = (Scaled(inner, 2.0), Tilt(inner, zeros(3, 3)), Translate(inner, zeros(3, 3)),
              Regularize(inner, 1.0), Precompose(inner, 2.0), MoreauEnvelope(inner, 0.5), Conjugate(inner),
              Constrained(inner, IndBox(-1.0, 1.0)), SqrDistL2(IndBox(-W, 2.0), 1.0), DistL2(IndBox(-W, 2.0), 1.0),
              Scaled(Conjugate(Conjugate(inner)), 2.0), Conjugate(Scaled(inner, 2.0)),
              Precompose(NormL1(0.1), W), Tilt(NormL1(0.1), W), Translate(NormL1(0.1), W), Regularize(NormL1(0.1), 1.0, W))
  x = SymmetricSparseIterate(fill(0.5, 3, 3))
  for f in wrappers, op in (prox!, value_and_prox!)
    y = SymmetricSparseIterate(fill(7.0, 3, 3)); before = state(y)
    @test_throws ArgumentError op(y, f, x, 0.5)
    @test isequal(state(y), before)
    z = SymmetricSparseIterate(fill(7.0, 3, 3)); before = state(z)
    @test_throws ArgumentError op(z, f, fill(0.5, 3, 3), 0.5)
    @test isequal(state(z), before)
    d = fill(7.0, 3, 3)
    @test_throws ArgumentError op(d, f, x, 0.5)
    @test d == fill(7.0, 3, 3)
  end
  # a sum rejects before the first function writes
  s = Sum(LinearFunction(ones(3, 3)), Tilt(SqrNormL2(1.0), W))
  g = SymmetricSparseIterate(fill(7.0, 3, 3)); before = state(g)
  @test_throws ArgumentError gradient!(g, s, x)
  @test isequal(state(g), before)
end

@testset "a conjugate commits to an owner only a symmetric result" begin
  groups = [[1], [2], [3], [4]]                       # not closed under transposition, for p = 2
  w = [1.0, 0.25, 0.5, 1.0]
  f = Conjugate(GroupNormLinf(1.0, groups, w))
  y = SymmetricSparseIterate(fill(7.0, 2, 2)); before = state(y)
  @test_throws ArgumentError prox!(y, f, ones(2, 2), 1.0)
  @test isequal(state(y), before)
end

# the points of a call at an owner y: a dense matrix, an owner, the matrix view of the output (y holds the
# matrix), the view of another owner and a Symmetric wrapper
function owner_point(kind, A, y)
  kind == "dense" ? A : kind == "owner" ? SymmetricSparseIterate(A) : kind == "view of the output" ? matrixview(y) :
  kind == "view" ? matrixview(SymmetricSparseIterate(A)) : Symmetric(A, :L)
end

@testset "a spectral result reaches an owner up to rounding, $T, p = $p" for T in (Float32, Float64), p in (3, 6)
  rng = MersenneTwister(100p + sizeof(T))
  A = randn(rng, T, p, p); A = (A + A') / 2
  rtol = T == Float32 ? 1e-3 : 1e-10
  q = p * (p + 1) ÷ 2
  wrappers = ("Conjugate of NuclearNorm" => Conjugate(NuclearNorm(T(0.4))),
              "MoreauEnvelope of NuclearNorm" => MoreauEnvelope(NuclearNorm(T(0.4)), T(0.5)),
              "Precompose of NuclearNorm" => Precompose(NuclearNorm(T(0.4)), T(1)),
              "SqrDistL2 of IndBallRank" => SqrDistL2(IndBallRank(1), T(0.7)),
              "DistL2 of IndBallRank" => DistL2(IndBallRank(1), T(0.7)))
  @testset "$name" for (name, f) in wrappers, op in (prox!, value_and_prox!)
    ref = zeros(T, p, p)
    vref = op(ref, f, A, T(0.5))
    @testset "$kind" for kind in ("dense", "view of the output", "view", "Symmetric", "owner")
      y = kind == "view of the output" ? SymmetricSparseIterate(A) : SymmetricSparseIterate(T, p)
      x = owner_point(kind, A, y)
      # the owner as the point of a wrapper of a spectral function is not defined, but for the conjugate and the scalar
      # precomposition, which read its matrix (the check of the inner function reads a matrix)
      if kind == "owner" && !(f isa Union{Conjugate, Precompose})
        @test_throws MethodError op(y, f, x, T(0.5))
        continue
      end
      v = op(y, f, x, T(0.5))
      @test isapprox(Matrix(y), ref; rtol, atol = rtol * norm(ref))
      @test Matrix(y) == Matrix(y)' && nactive(y) == q
      op === value_and_prox! && @test isapprox(v, vref; rtol = 10rtol, atol = 10rtol)
    end
  end
end

@testset "a wrapper refuses a dense result that is not symmetric before any write, $T" for T in (Float32, Float64)
  g = GroupNormL2(T(1), [[1], [2], [3], [4]], T[0.1, 0.5, 2.0, 0.1])     # the groups are not closed under transposition
  C = IndBallGroupL0(0, [[2]])
  wrappers = ("MoreauEnvelope" => MoreauEnvelope(g, T(0.5)), "Precompose" => Precompose(g, T(1)),
              "SqrDistL2" => SqrDistL2(C, T(1)), "DistL2" => DistL2(C, T(1)))
  A = ones(T, 2, 2)
  @testset "$name" for (name, f) in wrappers, op in (prox!, value_and_prox!)
    @testset "$kind" for kind in ("dense", "owner", "view of the output", "view", "Symmetric")
      y = SymmetricSparseIterate(A)
      if kind != "view of the output"
        y = SymmetricSparseIterate(T, 2)
        coordinates(y)[1] = T(5); coordinates(y)[3] = T(-2)     # some active coordinates, one inactive
      end
      before = state(y)
      x = owner_point(kind, A, y)
      # an owner point is not defined for the wrappers whose inner function has no owner point (MethodError)
      @test_throws (kind == "owner" && !(f isa Precompose) ? MethodError : ArgumentError) op(y, f, x, T(0.5))
      @test isequal(state(y), before)
    end
    # a dense output takes the result as it is
    d = zeros(T, 2, 2)
    @test op(d, f, A, T(0.5)) isa Union{Matrix, Real}
  end
  # the same inner function directly at an owner is refused as well
  y = SymmetricSparseIterate(ones(T, 2, 2)); before = state(y)
  @test_throws ArgumentError prox!(y, g, ones(T, 2, 2), T(0.5))
  @test_throws ArgumentError prox!(y, C, ones(T, 2, 2), T(0.5))
  @test isequal(state(y), before)
end

@testset "the entry wrappers allocate nothing after warm-up" begin
  x = SymmetricSparseIterate(X)
  for f in (Scaled(NormL1(0.3), 2.0), Constrained(NormL1(0.3), IndBox(-1.0, 1.0)), Scaled(Scaled(NormL1(0.3), 2.0), 0.5))
    y = SymmetricSparseIterate(X)
    prox!(y, f, x, 0.7)
    @test allocs(prox!, y, f, x, 0.7) == 0
    value_and_prox!(y, f, x, 0.7)
    @test allocs(value_and_prox!, y, f, x, 0.7) == 0
    @test allocs(prox!, y, f, y, 0.7) == 0
  end
end

end
