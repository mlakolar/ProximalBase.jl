module SymmetricWrappersTest

# The wrappers (Scaled, Tilt, Translate, Regularize, Precompose, MoreauEnvelope, SqrDistL2, DistL2, Conjugate,
# BoxConstrained, Sum) and the proximal operators of ZeroFunction, CubeNormL2, SumLargest and IndBallGroupL0 on a
# SymmetricSparseIterate, against the dense computation on the matrix that it represents.

using Test, ProximalBase, LinearAlgebra, Random
using ..TestUtils: allocs

const PB = ProximalBase
# the set of the one skew-symmetric matrix with entries 1 below and -1 above the diagonal: its projection is that matrix
struct SkewSet <: PB.AbstractFunction end
PB._isindicator(::Type{SkewSet}) = true
PB.isproximable(::Type{SkewSet}) = true
PB.isconvex(::Type{SkewSet}) = true
function PB.prox!(y::AbstractMatrix, ::SkewSet, x::AbstractMatrix, γ::Real)
  for c in axes(y, 2), r in axes(y, 1)
    y[r, c] = r > c ? one(eltype(y)) : r < c ? -one(eltype(y)) : zero(eltype(y))
  end
  y
end
PB.value_and_prox!(y::AbstractMatrix, f::SkewSet, x::AbstractMatrix, γ::Real) = (PB.prox!(y, f, x, γ); zero(eltype(y)))
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

@testset "Regularize forms its gradient in the model precision, at both extremes" begin
  # the point makes the model Float64, the output is Float32: the level and the entries meet before the output
  for (ρ, xv) in ((1e40, 1e-40), (1e-300, 1e300)), c in (nothing, fill(0.0, 2, 2))
    f = c === nothing ? Regularize(LinearFunction(zeros(2, 2)), ρ) : Regularize(LinearFunction(zeros(2, 2)), ρ, c)
    x = SymmetricSparseIterate(fill(xv, 2, 2))
    ones32 = fill(1f0, 2, 2)
    g = SymmetricSparseIterate(Float32, 2)
    @test gradient!(g, f, x) === g
    @test Matrix(g) == ones32
    g = SymmetricSparseIterate(Float32, 2)
    @test value_and_gradient!(g, f, x) isa Real
    @test Matrix(g) == ones32
    d = zeros(Float32, 2, 2)
    @test gradient!(d, f, x) === d
    @test d == ones32
    d = zeros(Float32, 2, 2)
    @test value_and_gradient!(d, f, x) isa Real
    @test d == ones32
    # a dense point with an owner output
    g = SymmetricSparseIterate(Float32, 2)
    gradient!(g, f, fill(xv, 2, 2))
    @test Matrix(g) == ones32
    # the Hessian product: the direction is the point, whatever the kind of the roles
    f0 = Regularize(LinearFunction(zeros(2, 2)), ρ)
    v = SymmetricSparseIterate(fill(xv, 2, 2))
    for (Hv, point, dir) in ((zeros(Float32, 2, 2), x, v), (zeros(Float32, 2, 2), x, fill(xv, 2, 2)),
                             (zeros(Float32, 2, 2), fill(xv, 2, 2), v), (SymmetricSparseIterate(Float32, 2), x, v),
                             (SymmetricSparseIterate(Float32, 2), fill(xv, 2, 2), fill(xv, 2, 2)))
      hessian_vec!(Hv, f0, point, dir)
      @test Matrix(Hv) == ones32
    end
  end
  # the entry is added to the gradient of the inner function, and rounded to the output once
  f = Regularize(SqrNormL2(1.0), 1e40)
  g = SymmetricSparseIterate(Float32, 2)
  gradient!(g, f, SymmetricSparseIterate(fill(1e-40, 2, 2)))
  @test Matrix(g) == fill(1f0, 2, 2)
  # a Float32 level that Float32 holds, and a center of the output's width
  f = Regularize(LinearFunction(zeros(2, 2)), 3f0, fill(1f0, 2, 2))
  g = SymmetricSparseIterate(Float32, 2)
  gradient!(g, f, SymmetricSparseIterate(fill(2f0, 2, 2)))
  @test Matrix(g) == fill(3f0, 2, 2)
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

# The nuclear norm is the one spectral function whose wrappers (Conjugate, MoreauEnvelope, the scalar Precompose)
# store the lower triangle of the dense result at an owner without the pair test: its proximal point at a symmetric
# point is unique and symmetric, and the two triangles of the computed matrix differ by rounding.
@testset "a nuclear-norm result reaches an owner up to rounding, $T, p = $p" for T in (Float32, Float64), p in (3, 6)
  rng = MersenneTwister(100p + sizeof(T))
  A = randn(rng, T, p, p); A = (A + A') / 2
  rtol = T == Float32 ? 1e-3 : 1e-10
  q = p * (p + 1) ÷ 2
  wrappers = ("Conjugate of NuclearNorm" => Conjugate(NuclearNorm(T(0.4))),
              "Conjugate of a partial NuclearNorm" => Conjugate(NuclearNorm(T(0.4); partial = true)),
              "MoreauEnvelope of NuclearNorm" => MoreauEnvelope(NuclearNorm(T(0.4)), T(0.5)),
              "Precompose of NuclearNorm" => Precompose(NuclearNorm(T(0.4)), T(1)))
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

# The dense result of the inner function of a wrapper is stored from its lower triangle only when it is exactly
# symmetric (every pair equal, NaNs, infinities and signed zeros included): the check has no tolerance, so a
# result that is asymmetric by any amount, at any size, is refused before the owner is written.
@testset "a wrapper refuses a dense result that is asymmetric at any size, Float32" begin
  # the inner projection keeps the lower triangle and zeros the upper one: the restricted proximal solution of
  # the sum of the square distance and the step is 2 / (2 + γ) = 0.8, not the 1 of the lower entries
  p = 2048
  A = ones(Float32, p, p)
  upper = [r + (c - 1) * p for c in 1:p for r in 1:c-1]
  @testset "$(nameof(typeof(f))), $op" for f in (SqrDistL2(IndBallGroupL0(0, [upper]), 1f0), DistL2(IndBallGroupL0(0, [upper]), 1f0)),
                                          op in (prox!, value_and_prox!)
    y = SymmetricSparseIterate(Float32, p)
    @test_throws ArgumentError op(y, f, A, 0.5f0)
    @test nactive(y) == 0 && y[2, 1] == 0 && y[p, p] == 0
  end
  # a skew-symmetric result (lower entries 1, upper entries -1: the maximal relative asymmetry) at p = 2050, from an
  # inner set that is a single skew matrix
  p = 2050
  A = ones(Float32, p, p)
  @testset "skew result, $(nameof(typeof(f))), $op" for f in (SqrDistL2(SkewSet(), 1f0), MoreauEnvelope(SkewSet(), 0.5f0)),
                                                       op in (prox!, value_and_prox!)
    y = SymmetricSparseIterate(Float32, p)
    @test_throws ArgumentError op(y, f, A, 0.5f0)
    @test nactive(y) == 0 && y[2, 1] == 0
  end
end

# the point of the restricted problem x -> (nuclear norm), at scales that make the dense computation cancel
@testset "the nuclear-norm wrappers at an owner agree with their references at every scale, $T" for T in (Float32, Float64)
  rng = MersenneTwister(2020 + sizeof(T))
  # reference of the proximal point of c ‖·‖_* at a symmetric matrix, in Float64: the eigenvalues, soft-thresholded
  spectral(A, g) = (E = eigen(Symmetric(Float64.(A))); R = E.vectors * Diagonal(g.(E.values)) * E.vectors'; (R + R') / 2)
  λc, γ, t, λe = T(0.35), T(0.3), T(0.4), T(0.4)
  worst = 0.0
  for p in 1:20, scale in (0.01, 1, 100)
    B = randn(rng, T, p, p); B = (B + B') / 2
    A = T(scale) .* B
    nA = opnorm(Float64.(A))
    # The conjugate of the nuclear norm is the projection onto the spectral-norm ball: the eigenvalues clipped to
    # [-λ, λ]. The error of a backward stable eigendecomposition is a few eps ‖A‖₂; the bound is 32 eps(T) max(‖A‖₂, λ)
    # in every entry (the largest ratio of the sweep is below 11)
    for op in (prox!, value_and_prox!)
      y = SymmetricSparseIterate(T, p)
      op(y, Conjugate(NuclearNorm(λc)), A, γ)
      err = maximum(abs.(Float64.(Matrix(y)) .- spectral(A, e -> clamp(e, -λc, λc))))
      @test err <= 32 * eps(T) * max(nA, Float64(λc))
      @test Matrix(y) == Matrix(y)' && nactive(y) == p * (p + 1) ÷ 2
      worst = max(worst, err / (eps(T) * max(nA, Float64(λc))))
    end
    # the envelope: x + t / (γ + t) (P - x) with P the eigenvalues soft-thresholded at (γ + t) λ
    env = MoreauEnvelope(NuclearNorm(λe), γ)
    y = SymmetricSparseIterate(T, p)
    prox!(y, env, A, t)
    P = spectral(A, e -> sign(e) * max(abs(e) - Float64(γ + t) * Float64(λe), 0))
    Y = Float64.(A) .+ Float64(t / (γ + t)) .* (P .- Float64.(A))
    @test maximum(abs.(Float64.(Matrix(y)) .- Y)) <= 32 * eps(T) * max(nA, 1.0)
    # the envelope and the scalar precomposition store the lower triangle of the dense call on the matrix of the point
    for f in (env, Precompose(NuclearNorm(λe), T(1.5)))
      for point in (A, SymmetricSparseIterate(A), matrixview(SymmetricSparseIterate(A)))
        point isa SymmetricSparseIterate && f === env && continue            # no owner point for the envelope
        y = SymmetricSparseIterate(T, p)
        prox!(y, f, point, t)
        ref = zeros(T, p, p)
        prox!(ref, f, Matrix(A), t)
        @test Matrix(y) == Matrix(Symmetric(ref, :L))
      end
    end
  end
  @test worst < 32
  # the point that the review of the check reports: the dense computation subtracts quantities of size 100
  A = T(100) .* T[1 0.3; 0.3 -0.8]
  expected = spectral(A, e -> clamp(e, -0.35, 0.35))
  for point in (A, SymmetricSparseIterate(A), matrixview(SymmetricSparseIterate(A))), op in (prox!, value_and_prox!)
    y = SymmetricSparseIterate(T, 2)
    op(y, Conjugate(NuclearNorm(T(0.35))), point, T(0.3))
    @test maximum(abs.(Float64.(Matrix(y)) .- expected)) <= 32 * eps(T) * 106
    @test Matrix(y) == Matrix(y)' && nactive(y) == 3
  end
end

# Nothing else is derived: the wrappers of a nuclear norm (Scaled, Translate, Regularize, a conjugate, nested
# inside another wrapper) and every other spectral function take the exact check. Each call is refused before the
# owner is written, or the inner result is exactly symmetric and the owner stores the lower triangle of the dense
# result; some of the calls are refused.
@testset "the exact check applies to every other combination, $T" for T in (Float32, Float64)
  nuc(c) = NuclearNorm(T(c))
  combos = ("Conjugate(Scaled(NuclearNorm))" => Conjugate(Scaled(nuc(0.4), T(2))),
            "Conjugate(Translate(NuclearNorm))" => Conjugate(Translate(nuc(0.4), zeros(T, 3, 3))),
            "Conjugate(Regularize(NuclearNorm))" => Conjugate(Regularize(nuc(0.4), T(0.5))),
            "MoreauEnvelope(Scaled(NuclearNorm))" => MoreauEnvelope(Scaled(nuc(0.4), T(2)), T(0.5)),
            "MoreauEnvelope(Conjugate(NuclearNorm))" => MoreauEnvelope(Conjugate(nuc(0.4)), T(0.5)),
            "Precompose(Scaled(NuclearNorm))" => Precompose(Scaled(nuc(0.4), T(2)), T(1.5)))
  # The Float32 entries of the three names in `moved` run in symmetric_wrappers_grids.jl.
  moved = ("Conjugate(Translate(NuclearNorm))", "Conjugate(Regularize(NuclearNorm))", "MoreauEnvelope(Conjugate(NuclearNorm))")
  kept = T === Float32 ? filter(c -> !(first(c) in moved), combos) : combos
  @testset "$name, $op" for (name, f) in kept, op in (prox!, value_and_prox!)
    refused = 0
    for trial in 1:12
      rng = MersenneTwister(1000trial + sizeof(T))
      A = randn(rng, T, 3, 3); A = (A + A') / 2
      y = SymmetricSparseIterate(T, 3)
      coordinates(y)[1] = T(5); coordinates(y)[3] = T(-2)
      before = state(y)
      ref = zeros(T, 3, 3)
      op(ref, f, A, T(0.5))
      stored = try op(y, f, A, T(0.5)); true catch e; e isa ArgumentError ? false : rethrow() end
      if stored
        @test Matrix(y) == Matrix(Symmetric(ref, :L))
      else
        refused += 1
        @test isequal(state(y), before)
      end
    end
    @test refused > 0
  end
end

# At a tie of the singular values the projection onto the rank-1 matrices is not unique, and the dense result at a
# symmetric point need not be symmetric. The rank balls take the exact check and nothing else: an exactly symmetric
# dense result is a minimizer of the restricted problem and is stored; any other is refused with the owner unchanged.
# The lower triangle of an asymmetric result is never stored. Outcomes recorded for the points below:
# the diagonal tie is stored by all three wrappers, its rotation is refused by all three.
@testset "rank-ball ties: the dense minimizer is stored exactly or the call is refused, $T" for T in (Float32, Float64)
  θ = T(0.3)
  Q = T[cos(θ) -sin(θ); sin(θ) cos(θ)]
  D = T[1 0; 0 -1]
  points = ("diagonal" => D, "rotated" => (S = Q * D * Q'; (S + S') / 2))
  γ, λ, μ = T(0.5), T(0.7), T(0.8)
  # the value of each wrapper at the symmetric matrix U (Eckart-Young for the distance to the rank-1 matrices)
  dist2(U) = sum(abs2, svdvals(Float64.(U))[2:end])
  objective(::SqrDistL2, U, X) = Float64(λ) / 2 * dist2(U) + sum(abs2, Float64.(U) .- Float64.(X)) / (2 * Float64(γ))
  objective(::DistL2, U, X) = Float64(λ) * sqrt(dist2(U)) + sum(abs2, Float64.(U) .- Float64.(X)) / (2 * Float64(γ))
  # the indicator of rank(μ U) ≤ 1: feasible up to the rounding of the stored matrix, else infinite
  function objective(::Precompose, U, X)
    s = svdvals(Float64.(U))
    s[2] <= 64 * eps(T) * s[1] ? sum(abs2, Float64.(U) .- Float64.(X)) / (2 * Float64(γ)) : Inf
  end
  outcome = Dict("diagonal" => :stored, "rotated" => :refused)
  @testset "$name, $label, $op" for (name, f) in ("SqrDistL2" => SqrDistL2(IndBallRank(1), λ), "DistL2" => DistL2(IndBallRank(1), λ),
                                                  "Precompose" => Precompose(IndBallRank(1), μ)),
                                    (label, X) in points, op in (prox!, value_and_prox!)
    y = SymmetricSparseIterate(T, 2)
    coordinates(y)[1] = T(5); coordinates(y)[2] = T(-2)
    before = state(y)
    ref = zeros(T, 2, 2)
    op(ref, f, X, γ)
    stored = try op(y, f, X, γ); true catch e; e isa ArgumentError ? false : rethrow() end
    if stored
      @test ref == ref'                                 # only an exactly symmetric dense result is stored
      @test Matrix(y) == ref
      @test objective(f, Matrix(y), X) <= objective(f, ref, X) * (1 + 8 * eps(T)) + 8 * eps(T)
    else
      @test isequal(state(y), before)
    end
    @test (stored ? :stored : :refused) == outcome[label]
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

@testset "a rejection-only owner role ends in a MethodError, never in recursion, $T" for T in (Float32, Float64)
  W = ones(T, 2, 2); W[2, 1] = 2                       # not symmetric
  valid = ("PointwiseMinimum" => PointwiseMinimum(NormL1(T(0.2)), SqrNormL2(T(0.3))),
           "IndIntersection" => IndIntersection(IndBox(-T(1), T(1)), IndBallL2(T(1))),
           "LinearConstrained" => Constrained(NormL1(T(0.2)), IndHalfspace(ones(T, 2, 2), T(1))))
  invalid = ("PointwiseMinimum" => PointwiseMinimum(NormL1(T(0.2), W), SqrNormL2(T(0.3))),
             "IndIntersection" => IndIntersection(IndBox(-W, T(1)), IndBallL2(T(1))),
             "LinearConstrained" => Constrained(NormL1(T(0.2)), IndHalfspace(W, T(1))))
  X = T[1 0.5; 0.5 2]
  roles = ("owner output, owner point" => () -> (SymmetricSparseIterate(T, 2), SymmetricSparseIterate(X)),
           "owner output, dense point" => () -> (SymmetricSparseIterate(T, 2), X),
           "owner output, matrix view" => () -> (y = SymmetricSparseIterate(X); (y, matrixview(y))),
           "dense output, owner point" => () -> (zeros(T, 2, 2), SymmetricSparseIterate(X)))
  snapshot(y::SymmetricSparseIterate) = state(y)
  snapshot(y) = copy(y)
  @testset "$name, $role, $op" for ((name, f), (_, bad)) in zip(valid, invalid), (role, make) in roles, op in (prox!, value_and_prox!)
    y, x = make()
    before = snapshot(y)
    err = try op(y, f, x, T(0.2)); nothing catch e; e end
    @test err isa MethodError && err.f === op
    @test isequal(snapshot(y), before)
    y, x = make()
    before = snapshot(y)
    @test_throws ArgumentError op(y, bad, x, T(0.2))
    @test isequal(snapshot(y), before)
  end
  # the dense operations are untouched
  for (_, f) in valid
    @test prox!(zeros(T, 2, 2), f, X, T(0.2)) isa Matrix
  end
end

# The block preflight of a separable sum asks, after the parameters are validated and before any block is written,
# whether the owner role of each block has a method: the three rejection-only functions end in the MethodError of
# the operation, with every block, owner or dense, as it was.
@testset "a separable sum with a rejection-only owner block writes no block, $T" for T in (Float32, Float64)
  mkowner(v) = SymmetricSparseIterate(fill(T(v), 2, 2))
  point() = ones(T, 2, 2)
  # The testsets of the Float32 pass before the supported blocks run in symmetric_wrappers_grids.jl.
  if T === Float64
    W = ones(T, 2, 2); W[2, 1] = 2                       # not symmetric
    rejecting = ("PointwiseMinimum" => PointwiseMinimum(NormL1(T(0.2)), SqrNormL2(T(0.3))),
                 "IndIntersection" => IndIntersection(IndBox(-T(1), T(1)), IndBallL2(T(1))),
                 "LinearConstrained" => Constrained(NormL1(T(0.2)), IndHalfspace(ones(T, 2, 2), T(1))))
    invalid = ("PointwiseMinimum" => PointwiseMinimum(NormL1(T(0.2), W), SqrNormL2(T(0.3))),
               "IndIntersection" => IndIntersection(IndBox(-W, T(1)), IndBallL2(T(1))),
               "LinearConstrained" => Constrained(NormL1(T(0.2)), IndHalfspace(W, T(1))))
    tstate(t::Tuple) = map(tstate, t)
    tstate(y::SymmetricSparseIterate) = state(y)
    tstate(y) = copy(y)
    # the outputs and the points of a call, as a tuple of two or of a block and a nested pair
    roles = ("owner outputs, dense points" => () -> ((mkowner(1), mkowner(2)), (point(), point())),
             "dense outputs, owner points" => () -> ((fill(T(3), 2, 2), fill(T(4), 2, 2)), (mkowner(1), mkowner(2))),
             "owner and dense outputs, dense points" => () -> ((mkowner(1), fill(T(4), 2, 2)), (point(), point())))
    @testset "$name, $role, $where, $op" for (name, bad) in rejecting, (role, make) in roles,
                                             where in ("second", "first", "nested"), op in (prox!, value_and_prox!)
      nl1 = NormL1(T(0.5))
      y, x = make()                                       # the outputs and the points of two blocks
      if where == "second"
        f, ry, rx = SeparableSum(nl1, bad), y[2], x[2]
      elseif where == "first"
        f, ry, rx = SeparableSum(bad, nl1), y[1], x[1]
      else                                                # (a, (b, c)) with the rejecting block last, of the kind of b
        ey = y[2] isa SymmetricSparseIterate ? mkowner(5) : fill(T(5), 2, 2)
        ex = x[2] isa SymmetricSparseIterate ? mkowner(6) : point()
        f, ry, rx = SeparableSum(nl1, SeparableSum(nl1, bad)), ey, ex
        y, x = (y[1], (y[2], ey)), (x[1], (x[2], ex))
      end
      before = tstate(y)
      err = try op(y, f, x, T(0.5)); nothing catch e; e end
      if ry isa SymmetricSparseIterate || rx isa SymmetricSparseIterate
        @test err isa MethodError && err.f === op
        @test isequal(tstate(y), before)
      else                                                # a role without an owner is a dense call, which is supported
        @test err === nothing
      end
    end
    # the wrappers that forward to a rejection-only function (Scaled, Translate, Tilt, nested) end in its MethodError too
    forwarding = (("Scaled", f -> Scaled(f, T(2))), ("Translate", f -> Translate(f, zeros(T, 2, 2))),
                  ("Tilt", f -> Tilt(f, T[0.1 0.2; 0.2 0.3])), ("Scaled of Translate", f -> Scaled(Translate(f, zeros(T, 2, 2)), T(2))),
                  ("Tilt of Scaled", f -> Tilt(Scaled(f, T(2)), T[0.1 0.2; 0.2 0.3])))
    @testset "$wname of $name, $role, $where, $op" for (wname, wrap) in forwarding, (name, bad) in rejecting,
                                                       (role, make) in roles[1:2], where in ("second", "first"), op in (prox!, value_and_prox!)
      nl1 = NormL1(T(0.5))
      f = where == "second" ? SeparableSum(nl1, wrap(bad)) : SeparableSum(wrap(bad), nl1)
      y, x = make()
      before = tstate(y)
      err = try op(y, f, x, T(0.5)); nothing catch e; e end
      @test err isa MethodError && err.f === op
      @test isequal(tstate(y), before)
    end
    # a scaled sum of blocks as a block: the preflight descends to its leaves, nested sums and scales included
    @testset "scaled sum of blocks, $name, $role, $op" for (name, bad) in rejecting, (role, make) in roles[1:2], op in (prox!, value_and_prox!)
      nl1 = NormL1(T(0.5))
      extra(t, v) = t isa SymmetricSparseIterate ? mkowner(v) : fill(T(v), 2, 2)
      for (f, depth) in ((SeparableSum(nl1, Scaled(SeparableSum(nl1, bad), T(2))), 1),
                         (SeparableSum(nl1, Scaled(Scaled(SeparableSum(nl1, bad), T(2)), T(3))), 1),
                         (SeparableSum(nl1, Scaled(SeparableSum(nl1, Scaled(SeparableSum(nl1, bad), T(2))), T(3))), 2))
        y, x = make()                                    # (a, b) and its point; the nested pair holds the rejecting block last
        y = depth == 1 ? (y[1], (y[2], extra(y[2], 5))) : (y[1], (y[2], (extra(y[2], 5), extra(y[2], 6))))
        x = depth == 1 ? (x[1], (x[2], extra(x[2], 5))) : (x[1], (x[2], (extra(x[2], 5), extra(x[2], 6))))
        before = tstate(y)
        err = try op(y, f, x, T(0.5)); nothing catch e; e end
        @test err isa MethodError && err.f === op
        @test isequal(tstate(y), before)
      end
    end
    # parameter validation wins over the MethodError
    @testset "invalid parameters under $wname, $op" for (wname, wrap) in forwarding, op in (prox!, value_and_prox!)
      for (_, bad) in invalid
        y = (mkowner(1), mkowner(2)); before = tstate(y)
        @test_throws ArgumentError op(y, SeparableSum(NormL1(T(0.5)), wrap(bad)), (point(), point()), T(0.5))
        @test isequal(tstate(y), before)
      end
    end
    # a block with invalid parameters is refused by the parameter check, whichever block is rejection-only
    @testset "invalid parameters, $name, $op" for ((name, bad), (_, good)) in zip(invalid, rejecting), op in (prox!, value_and_prox!)
      for f in (SeparableSum(NormL1(T(0.5), W), good), SeparableSum(NormL1(T(0.5)), bad), SeparableSum(bad, NormL1(T(0.5))))
        y = (mkowner(1), mkowner(2)); before = tstate(y)
        @test_throws ArgumentError op(y, f, (point(), point()), T(0.5))
        @test isequal(tstate(y), before)
      end
    end
  end
  # a tuple of supported blocks is unchanged: the owner holds the dense result, nested or not
  @testset "supported blocks, $op" for op in (prox!, value_and_prox!)
    for f in (SeparableSum(NormL1(T(0.5)), SqrNormL2(T(0.3))), SeparableSum(NormL1(T(0.5)), SeparableSum(SqrNormL2(T(0.3)), NormL1(T(0.1)))))
      nested = f.fs[2] isa SeparableSum
      y = nested ? (mkowner(1), (mkowner(2), mkowner(3))) : (mkowner(1), mkowner(2))
      d = nested ? (ones(T, 2, 2), (ones(T, 2, 2), ones(T, 2, 2))) : (ones(T, 2, 2), ones(T, 2, 2))
      x = nested ? (point(), (point(), point())) : (point(), point())
      op(y, f, x, T(0.5)); op(d, f, x, T(0.5))
      @test Matrix(y[1]) == d[1]
      @test nested ? (Matrix(y[2][1]) == d[2][1] && Matrix(y[2][2]) == d[2][2]) : Matrix(y[2]) == d[2]
    end
  end
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
