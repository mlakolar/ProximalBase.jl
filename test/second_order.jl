module SecondOrderTest

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using Random

using ..TestUtils

Random.seed!(1)

const PB = ProximalBase

# central differences of the gradient along v
fd_hessian_vec(f, x, v; h = 1e-6) = (gradient(f, x .+ h .* v) .- gradient(f, x .- h .* v)) ./ (2h)

# hessian_vec! against central differences of the gradient, its allocating
# version, and the symmetry of the Hessian; with symmetric = true the
# directions are symmetric matrices
function check_hessian(f, x; rtol = 1e-5, symmetric = false)
  sym(w) = symmetric ? (w .+ w') ./ 2 : w
  @test hashessian(f)
  @test hashessian(typeof(f))
  xd = Array(x)
  v, u = sym(randn(size(xd)...)), sym(randn(size(xd)...))
  Hv = hessian_vec(f, x, v)
  @test size(Hv) == size(x)
  Hv2 = similar(Hv)
  @test hessian_vec!(Hv2, f, x, v) === Hv2
  @test Hv2 ≈ Hv
  @test Hv ≈ fd_hessian_vec(f, xd, v) rtol = rtol atol = 1e-6 * max(1, norm(Hv))
  @test dot(u, Hv) ≈ dot(v, hessian_vec(f, x, u)) rtol = 1e-8
  Hv
end

# the model q of f at x has the gradient and the Hessian of f at x, and its
# change from x is the second-order expansion of f
function check_model(f, x; rtol = 1e-8, symmetric = false)
  sym(w) = symmetric ? (w .+ w') ./ 2 : w
  q = quadratic_model(f, x)
  @test issmooth(q) && hashessian(q)
  xd = Array(x)
  g = gradient(f, x)
  @test gradient(q, x) ≈ g rtol = rtol
  v = sym(randn(size(xd)...))
  @test hessian_vec(q, x, v) ≈ hessian_vec(f, x, v) rtol = rtol
  Δ = sym(randn(size(xd)...)) ./ 10
  @test q(xd .+ Δ) - q(xd) ≈ dot(g, Δ) + dot(Δ, hessian_vec(f, x, Δ)) / 2 rtol = sqrt(rtol)
  q
end

@testset "quadratics are their own models" begin
  n, p = 30, 8
  X, y, x = randn(n, p), randn(n), randn(p)
  A = Matrix(Symmetric(X'X / n))
  for f in (ZeroFunction(), LinearFunction(randn(p), 1.), SqrNormL2(2.), SqrNormL2(0.5, randn(p)),
            QuadraticFunction(A, randn(p), 3.), LeastSquaresLoss(y, X))
    @testset "$(nameof(typeof(f)))" begin
      Hv = check_hessian(f, x)
      @test quadratic_model(f, x) === f
      @test quadratic_model!(f, f, randn(p)) === f
      check_model(f, x)
    end
  end
  v = randn(p)
  @test hessian_vec(ZeroFunction(), x, v) == zeros(p)
  @test hessian_vec(SqrNormL2(2.), x, v) ≈ 2v
  @test hessian_vec(QuadraticFunction(A), x, v) ≈ A * v
  @test hessian_vec(LeastSquaresLoss(y, X), x, v) ≈ X' * (X * v)

  # a matrix response, and sparse points and directions
  Y = randn(n, 3)
  fm = LeastSquaresLoss(Y, X)
  check_hessian(fm, randn(p, 3))
  @test hessian_vec(fm, randn(p, 3), ones(p, 3)) ≈ X' * (X * ones(p, 3))
  xs, vs = SparseIterate(sprandn(p, 0.4)), SparseIterate(sprandn(p, 0.4))
  @test hessian_vec(LeastSquaresLoss(y, X), xs, vs) ≈ X' * (X * Vector(vs))
  @test hessian_vec(LeastSquaresLoss(y, X), x, AtomIterate(Vector(vs), [1:4, 5:8])) ≈ X' * (X * Vector(vs))

  @testset "KronQuadratic" begin
    B = randn(5, 5); Bq = Matrix(Symmetric(B'B / 5))
    C = randn(p, 5)
    f = KronQuadratic(A, Bq, C)
    Θ = randn(p, 5)
    Hv = check_hessian(f, Θ)
    V = randn(p, 5)
    @test hessian_vec(f, Θ, V) ≈ A * V * Bq
    @test hessian_vec(f, vec(Θ), vec(V)) ≈ vec(A * V * Bq)
    @test hessian_vec(f, SparseIterate(sparse(Θ)), V) ≈ A * V * Bq
    check_model(f, Θ)
    # over symmetric matrices the product is symmetrized, as the gradient is
    Bp = randn(p, p)
    fs = KronQuadratic(A, Bp'Bp / p, Matrix(Symmetric(randn(p, p))))
    Θs = SymmetricSparseIterate(Matrix(Symmetric(randn(p, p))))
    Vs = Matrix(Symmetric(randn(p, p)))
    Hs = hessian_vec(fs, Θs, Vs)
    @test Hs ≈ (fs.A * Vs * fs.B + fs.B * Vs * fs.A) / 2
    @test issymmetric(Hs)
  end
end

@testset "generalized linear models" begin
  n, p = 60, 6
  X = randn(n, p)
  x = randn(p) ./ 3
  w, o = rand(n) .+ 0.5, randn(n) ./ 4
  fs = [LogisticLoss(rand(0:1, n), X), LogisticLoss(rand(0:1, n), X; weights = w, offset = o),
        PoissonLoss(rand(0:4, n), X), PoissonLoss(rand(0:4, n), X; weights = w, offset = o),
        HuberLoss(randn(n), X, 0.5), GLMLoss(PB.Squared(), randn(n), X; weights = w)]
  for f in fs
    @testset "$(nameof(typeof(f.loss)))" begin
      check_hessian(f, x)
      # the Huber model has the curvature √eps where ℓ'' vanishes
      check_model(f, x; rtol = f.loss isa PB.Huber ? 1e-6 : 1e-8)
      check_hessian(f, SparseIterate(sparse(x)))
    end
  end

  @testset "the IRLS model" begin
    f = LogisticLoss(rand(0:1, n), X; weights = w, offset = o)
    q = quadratic_model(f, x)
    @test q isa GLMLoss{PB.Squared}
    η = X * x .+ o
    μ = 1 ./ (1 .+ exp.(-η))
    @test q.weights ≈ w .* μ .* (1 .- μ)
    @test q.y ≈ η .- (μ .- f.y) ./ (μ .* (1 .- μ))
    @test q.X === f.X && q.offset === f.offset
    # in place, at another point, the model at that point
    x2 = randn(p) ./ 3
    @test quadratic_model!(q, f, x2) === q
    q2 = quadratic_model(f, x2)
    @test q.y ≈ q2.y && q.weights ≈ q2.weights
    @test_throws ArgumentError quadratic_model!(quadratic_model(LogisticLoss(rand(0:1, n), randn(n, p)), x), f, x)
    # the gradient of the model is exact where the curvature vanishes
    fh = HuberLoss(10 .* randn(n), X, 0.1)
    @test gradient(quadratic_model(fh, x), x) ≈ gradient(fh, x)
  end

  @testset "Float32" begin
    f = LogisticLoss(Float32.(rand(0:1, n)), Float32.(X))
    x32 = Float32.(x)
    @test hessian_vec(f, x32, ones(Float32, p)) isa Vector{Float32}
    q = quadratic_model(f, x32)
    @test eltype(q.y) == Float32
    @test gradient(q, x32) ≈ gradient(f, x32) rtol = 1f-4
  end
end

@testset "LogDetLoss" begin
  p = 6
  B = randn(3p, p)
  S = Matrix(Symmetric(B'B / 3p))
  X = Matrix(Symmetric(inv(S) + 0.3I))
  for w in (1.0, 0.4)
    f = LogDetLoss(S, w)
    @testset "w = $w" begin
      check_hessian(f, X; symmetric = true)
      V = Matrix(Symmetric(randn(p, p)))
      W = inv(X)
      @test hessian_vec(f, X, V) ≈ w * W * V * W
      @test hessian_vec(f, SymmetricSparseIterate(X), SymmetricSparseIterate(V)) ≈ w * W * V * W

      q = check_model(f, X; symmetric = true)
      @test q isa KronQuadratic
      @test q.A ≈ w * W && q.B ≈ W && q.C ≈ w * (S - 2W)
      @test issymmetric(q.A) && issymmetric(q.B) && issymmetric(q.C)
      # the expansion in the step Δ = D - X, w(tr((S - W)Δ) + tr(WΔWΔ)/2), exactly
      for _ in 1:3
        Δ = Matrix(Symmetric(randn(p, p)))
        @test q(X + Δ) - q(X) ≈ w * (tr((S - W) * Δ) + tr(W * Δ * W * Δ) / 2) rtol = 1e-10
      end
      # on a symmetric iterate the model has the gradient of f
      @test gradient(q, SymmetricSparseIterate(X)) ≈ gradient(f, X)
      # without a penalty, the minimizer of the model is Newton's step 2X - XSX
      @test norm(gradient(q, 2X - X * S * X)) <= 1e-10 * norm(S)
      # in place, at another point
      X2 = Matrix(Symmetric(X + 0.1 * Matrix(Symmetric(randn(p, p))) / sqrt(p)))
      @test isposdef(X2)
      @test quadratic_model!(q, f, X2) === q
      q2 = quadratic_model(f, X2)
      @test q.A ≈ q2.A && q.B ≈ q2.B && q.C ≈ q2.C
    end
  end

  f = LogDetLoss(S)
  # off the cone: no Hessian, no model
  N = Matrix(Diagonal([-1.0; ones(p - 1)]))
  @test all(isnan, hessian_vec(f, N, Matrix(1.0I, p, p)))
  @test_throws ArgumentError quadratic_model(f, N)
  @test_throws ArgumentError hessian_vec!(zeros(Float32, p, p), f, X, X)
  # an output whose columns are not contiguous (issue #214)
  V = Matrix(Symmetric(randn(p, p)))
  Hv = view(zeros(2p, p), 1:2:2p, :)
  @test hessian_vec!(Hv, f, X, V) === Hv
  @test Hv ≈ hessian_vec(f, X, V)
  @test all(iszero, parent(Hv)[2:2:end, :])

  @testset "Float32" begin
    f32 = LogDetLoss(Float32.(S))
    X32 = Float32.(X)
    V32 = Float32.(Matrix(Symmetric(randn(p, p))))
    H32 = hessian_vec(f32, X32, V32)
    @test H32 isa Matrix{Float32}
    @test H32 ≈ Float32.(hessian_vec(f, X, Float64.(V32))) rtol = 1f-3
    q32 = quadratic_model(f32, X32)
    @test eltype(q32.A) == Float32
  end
end

@testset "functions built from functions" begin
  n, p = 40, 5
  X, y = randn(n, p), randn(n)
  x, c = randn(p) ./ 3, randn(p)
  fl = LogisticLoss(rand(0:1, n), X)
  for f in (Scaled(fl, 0.5), Tilt(fl, c, 2.), Translate(fl, c), Sum(fl, SqrNormL2(0.5)),
            Sum(LeastSquaresLoss(y, X), fl, LinearFunction(c)), (1 / n) * LeastSquaresLoss(y, X))
    @testset "$(nameof(typeof(f)))" begin
      check_hessian(f, x)
      check_model(f, x)
      # in place at another point
      q = quadratic_model(f, x)
      x2 = randn(p) ./ 3
      @test quadratic_model!(q, f, x2) === q
      @test gradient(q, x2) ≈ gradient(f, x2)
      @test hessian_vec(q, x2, c) ≈ hessian_vec(f, x2, c)
    end
  end
  @test hessian_vec(Scaled(fl, 0.5), x, c) ≈ 0.5 * hessian_vec(fl, x, c)
  @test hessian_vec(Translate(fl, c), x, c) ≈ hessian_vec(fl, x - c, c)
  @test quadratic_model(Scaled(LeastSquaresLoss(y, X), 2.), x) isa Scaled{<:LeastSquaresLoss}

  # blocks
  s = SeparableSum(LeastSquaresLoss(y, X), SqrNormL2(2.))
  @test hashessian(s)
  xb, vb = (randn(p), randn(3)), (randn(p), randn(3))
  Hb = hessian_vec(s, xb, vb)
  @test Hb[1] ≈ X' * (X * vb[1]) && Hb[2] ≈ 2vb[2]
  qs = quadratic_model(s, xb)
  @test qs isa SeparableSum
  @test quadratic_model!(qs, s, xb) === qs

  s = SeparableSum(SqrNormL2(), SqrNormL2(2))
  point, direction = ([1., 2.], [3.]), ([2., -1.], [4.])
  out = map(similar, point)
  expected = map((f, xk, vk) -> hessian_vec(2f, xk, vk), s.fs, point, direction)
  @test hessian_vec!(out, Scaled(s, 2), point, direction) === out
  @test all(map(isapprox, out, expected))
  @test all(map(isapprox, hessian_vec(Scaled(s, 2), point, direction), expected))
  @test allocs(hessian_vec!, out, Scaled(s, 2), point, direction) == 0

  nested = Scaled(SeparableSum(s, SqrNormL2(3)), 2)
  S = Symmetric([1. 2.; 2. 3.])
  for matrix in (S,)
    product = hessian_vec(nested, (point, matrix), (direction, matrix))
    @test all(map(isapprox, product[1], expected))
    @test product[2] isa Matrix
    @test product[2] ≈ 6matrix
  end
  # Later invalid blocks and directions are checked before the first product.
  for f in (SeparableSum(SqrNormL2(), SqrNormL2(1., ones(2))),
            Scaled(SeparableSum(SqrNormL2(), SqrNormL2(1., ones(2))), 2))
    out = map(copy, point)
    @test_throws DimensionMismatch hessian_vec!(out, f, point, direction)
    @test out == point
  end
  out = map(copy, point)
  @test_throws DimensionMismatch hessian_vec!(out, s, point, (direction[1], ones(2)))
  @test out == point

  # no Hessian
  @test !hashessian(NormL1(1.))
  @test !hashessian(Sum(fl, NormL1(1.)))
  @test hashessian(LogisticLoss)
  @test_throws ArgumentError hessian_vec(NormL1(1.), x, x)
  @test_throws ArgumentError quadratic_model(NormL1(1.), x)
  @test_throws DimensionMismatch hessian_vec(fl, x, randn(p + 1))
end

a_hv!(Hv, f, x, v) = hessian_vec!(Hv, f, x, v)

# a user function with its own Hessian: ½‖x‖², whose Hessian-vector product is v
struct UserHalfNorm <: AbstractFunction end
(::UserHalfNorm)(x) = sum(abs2, x) / 2
PB.hessian_vec!(Hv, ::UserHalfNorm, x, v) = copyto!(Hv, v)

blocks_approx(x::AbstractArray, y::AbstractArray, rtol = sqrt(eps(Float64))) = isapprox(x, y; rtol)
blocks_approx(x::Tuple, y::Tuple, rtol = sqrt(eps(Float64))) = all(map((a, b) -> blocks_approx(a, b, rtol), x, y))
fresh(x::AbstractArray) = similar(x)
fresh(x::Tuple) = map(fresh, x)
along(x::AbstractArray, v::AbstractArray, h) = x .+ h .* v
along(x::Tuple, v::Tuple, h) = map((a, b) -> along(a, b, h), x, v)
combine(a::AbstractArray, b::AbstractArray, w) = (a .- b) ./ w
combine(a::Tuple, b::Tuple, w) = map((u, t) -> combine(u, t, w), a, b)
tdot(a::AbstractArray, b::AbstractArray) = dot(a, b)
tdot(a::Tuple, b::Tuple) = sum(map(tdot, a, b))

@testset "Sum on tuples" begin
  # the reproduction of #159
  s = SeparableSum(SqrNormL2(), SqrNormL2(2))
  f = Sum(s, s)
  x, v = ([1.0, 2.0], [3.0]), ([2.0, -1.0], [5.0])
  @test hashessian(f)
  Hv = map(similar, v)
  @test hessian_vec!(Hv, f, x, v) === Hv
  @test Hv[1] ≈ [4.0, -2.0] && Hv[2] ≈ [20.0]
  @test blocks_approx(hessian_vec(f, x, v), Hv)

  # independent references: SqrNormL2(λ) has Hessian λI and ‖Xβ - y‖²/2 has X'X
  rng = Xoshiro(11)
  X, y = randn(rng, 9, 3), randn(rng, 9)
  c = [0.5, -1.0]
  a = SeparableSum(LeastSquaresLoss(y, X), SqrNormL2(2.))
  b = SeparableSum(SqrNormL2(0.3), SqrNormL2(1.5, c))
  z, w = (randn(rng, 3), randn(rng, 2)), (randn(rng, 3), randn(rng, 2))
  Ha = (X' * (X * w[1]), 2 .* w[2])
  Hb = (0.3 .* w[1], 1.5 .* w[2])
  @test blocks_approx(hessian_vec(Sum(a, b), z, w), map(+, Ha, Hb))
  @test blocks_approx(hessian_vec(Sum(b, a), z, w), map(+, Ha, Hb))
  @test blocks_approx(hessian_vec(Sum(a), z, w), Ha)
  @test blocks_approx(hessian_vec(Sum(a, b, Scaled(a, 0.5)), z, w), map((p, q, r) -> p + q + r / 2, Ha, Hb, Ha))
  @test blocks_approx(hessian_vec(Scaled(Sum(a, b), 3), z, w), map(t -> 3 .* t, map(+, Ha, Hb)))
  # central differences of the gradient along w, with the tuple structure kept
  for h in (Sum(a, b), Sum(a, Sum(b, a)), Scaled(Sum(a, b), 2))
    δ = 1e-6
    fd = combine(gradient(h, along(z, w, δ)), gradient(h, along(z, w, -δ)), 2δ)
    @test blocks_approx(hessian_vec(h, z, w), fd, 1e-6)
  end

  # blocks of different sizes, matrices and nested tuples; nested and repeated sums
  s3 = SeparableSum(SqrNormL2(), SqrNormL2(2), SqrNormL2(3))
  x3 = ([1.0, 2.0], [3.0 4.0 5.0; 6.0 7.0 8.0], [9.0])
  v3 = ([1.0, -1.0], [1.0 0.0 2.0; 0.0 1.0 3.0], [2.0])
  @test blocks_approx(hessian_vec(Sum(s3, s3), x3, v3), map((vk, k) -> 2k .* vk, v3, (1, 2, 3)))
  @test blocks_approx(hessian_vec(Sum(Sum(s3, s3), s3), x3, v3), map((vk, k) -> 3k .* vk, v3, (1, 2, 3)))
  nested = SeparableSum(s, SeparableSum(SqrNormL2(), SqrNormL2(2)))
  xn, vn = (x, ([4.0 5.0], [6.0, 7.0])), (v, ([1.0 2.0], [3.0, 4.0]))
  Hn = hessian_vec(Sum(nested, Scaled(nested, 3)), xn, vn)
  @test Hn[1][1] ≈ 4 .* vn[1][1] && Hn[1][2] ≈ 8 .* vn[1][2]
  @test Hn[2][1] ≈ 4 .* vn[2][1] && Hn[2][2] ≈ 8 .* vn[2][2]
  H1 = fresh(xn)
  @test hessian_vec!(H1, Sum(nested), xn, vn) === H1
  @test blocks_approx(H1, (map(*, (1, 2), vn[1]), map(*, (1, 2), vn[2])))
  # a user function that supplies its own Hessian is not rejected before any write
  @test !hashessian(Sum(SqrNormL2(), UserHalfNorm()))
  @test hessian_vec(Sum(SqrNormL2(2.), UserHalfNorm()), [1.0, 2.0], [3.0, 4.0]) ≈ [9.0, 12.0]

  # Float32 variables, also with Float64 model data
  x32, v32 = map(t -> Float32.(t), x), map(t -> Float32.(t), v)
  H32 = hessian_vec(f, x32, v32)
  @test H32 isa Tuple{Vector{Float32}, Vector{Float32}} && H32 == (Float32[4, -2], Float32[20])
  z32, w32 = map(t -> Float32.(t), z), map(t -> Float32.(t), w)
  Hm = hessian_vec(Sum(a, b), z32, w32)
  @test Hm isa Tuple{Vector{Float32}, Vector{Float32}}
  @test blocks_approx(Hm, hessian_vec(Sum(a, b), map(t -> Float64.(t), z32), map(t -> Float64.(t), w32)), 1e-4)

  # the buffer is reused while the shapes agree and renewed when they change
  h = Sum(s, s, Scaled(s, 2))
  out = fresh(x)
  hessian_vec!(out, h, x, v); buffer = h.scratch.buf
  hessian_vec!(out, h, x, v)
  @test h.scratch.buf === buffer
  @test all(bk -> all(o -> bk !== o, out) && all(o -> bk !== o, v) && all(o -> bk !== o, x), buffer[3])
  x2, v2 = ([1.0, 2.0, 3.0], [4.0]), ([1.0, 1.0, 1.0], [2.0])
  @test blocks_approx(hessian_vec!(fresh(x2), h, x2, v2), (4 .* v2[1], 8 .* v2[2]))
  @test h.scratch.buf !== buffer
  @test blocks_approx(hessian_vec!(out, h, x, v), (4 .* v[1], 8 .* v[2]))

  # many blocks and warm calls
  many = SeparableSum(ntuple(_ -> SqrNormL2(), 40)...)
  xm = ntuple(i -> fill(Float64(i), 2), 40)
  fm = Sum(many, many)
  Hm = fresh(xm)
  @test blocks_approx(hessian_vec!(Hm, fm, xm, xm), map(t -> 2 .* t, xm))
  @test allocs(a_hv!, Hm, fm, xm, xm) == 0
  @test allocs(a_hv!, out, h, x, v) == 0
  @test allocs(a_hv!, fresh(xn), Sum(nested, nested), xn, vn) == 0
  @inferred hessian_vec!(out, h, x, v)
  @inferred hessian_vec!(Hm, fm, xm, xm)
end

@testset "quadratic models of Sum on tuples" begin
  rng = Xoshiro(5)
  X1, X2 = randn(rng, 30, 3), randn(rng, 25, 2)
  l1 = LogisticLoss(Float64.(rand(rng, Bool, 30)), X1)
  l2 = LeastSquaresLoss(randn(rng, 25), X2)
  sep = SeparableSum(l1, l2)
  f = Sum(sep, SeparableSum(SqrNormL2(0.5), SqrNormL2(0.2, [1.0, -1.0])), Scaled(sep, 0.5))
  x, x′ = (randn(rng, 3) ./ 3, randn(rng, 2)), (randn(rng, 3) ./ 3, randn(rng, 2))
  v = (randn(rng, 3), randn(rng, 2))
  @test hashessian(f)
  q = quadratic_model(f, x)
  @test q isa Sum && issmooth(q) && hashessian(q)
  @test blocks_approx(gradient(q, x), gradient(f, x), 1e-10)
  @test blocks_approx(hessian_vec(q, x′, v), hessian_vec(f, x, v), 1e-10)
  Δ = map(t -> t ./ 10, v)
  g = gradient(f, x)
  @test q(along(x, Δ, 1.0)) - q(x) ≈ tdot(g, Δ) + tdot(Δ, hessian_vec(f, x, Δ)) / 2 rtol = 1e-8
  @test quadratic_model!(q, f, x′) === q
  @test blocks_approx(gradient(q, x′), gradient(f, x′), 1e-10)
  @test blocks_approx(hessian_vec(q, x, v), hessian_vec(f, x′, v), 1e-10)
  @test quadratic_model(Scaled(f, 2), x) isa Scaled{<:Sum}
  # parts are counted before any of them is updated
  @test_throws DimensionMismatch quadratic_model!(Sum(sep), f, x)
  @test_throws DimensionMismatch quadratic_model!(q, Sum(sep, sep), x)
  # a model of a function without a Hessian is refused
  @test_throws ArgumentError quadratic_model(Sum(sep, SeparableSum(NormL1(1.), SqrNormL2())), x)
end

@testset "Sum on tuples checks before writing" begin
  s = SeparableSum(SqrNormL2(), SqrNormL2(2))
  centered = SeparableSum(SqrNormL2(), SqrNormL2(1., ones(2)))     # the center does not fit the second block
  x, v = ([1.0, 2.0], [3.0]), ([2.0, -1.0], [5.0])
  sentinel = ([9.0, 8.0], [7.0])
  # an invalid final summand, however many valid ones precede it
  for f in (Sum(s, centered), Sum(s, s, centered), Sum(s, Sum(s, centered)), Sum(s, Scaled(centered, 2)))
    out = map(copy, sentinel)
    @test_throws DimensionMismatch hessian_vec!(out, f, x, v)
    @test out == sentinel
  end
  # an invalid final block of a long tuple, and a block of a nested tuple
  s3 = SeparableSum(SqrNormL2(), SqrNormL2(2), SqrNormL2(3))
  bad3 = SeparableSum(SqrNormL2(), SqrNormL2(2), SqrNormL2(1., ones(2)))
  x3, v3 = ([1.0, 2.0], [3.0], [4.0, 5.0, 6.0]), ([1.0, 2.0], [3.0], [4.0, 5.0, 6.0])
  out = map(t -> fill(-1.0, size(t)), x3)
  @test_throws DimensionMismatch hessian_vec!(out, Sum(s3, s3, bad3), x3, v3)
  @test all(o -> all(==(-1.0), o), out)
  nested = SeparableSum(s, SeparableSum(SqrNormL2(), SqrNormL2(2)))
  nested_bad = SeparableSum(s, SeparableSum(SqrNormL2(), SqrNormL2(1., ones(2))))
  xn = (x, ([4.0, 5.0], [6.0]))
  out = map(xk -> map(t -> fill(-1.0, size(t)), xk), xn)
  @test_throws DimensionMismatch hessian_vec!(out, Sum(nested, nested, nested_bad), xn, xn)
  @test all(o -> all(t -> all(==(-1.0), t), o), out)
  # a later function that needs another shape, in a block and for an array
  kron = KronQuadratic(Matrix(1.0I, 3, 3), Matrix(1.0I, 3, 3), zeros(3, 3))   # for 3 × 3 blocks
  xk, vk = (x[1], [1.0 2.0; 3.0 4.0]), (v[1], [1.0 0.0; 0.0 1.0])
  out = (zeros(2), zeros(2, 2)); copy_out = map(copy, out)
  @test_throws DimensionMismatch hessian_vec!(out, Sum(SeparableSum(SqrNormL2(), SqrNormL2()), SeparableSum(SqrNormL2(), kron)), xk, vk)
  @test out == copy_out
  A = [1.0 2.0; 3.0 4.0]
  out = fill(-1.0, 2, 2)
  @test_throws DimensionMismatch hessian_vec!(out, Sum(SqrNormL2(), kron), A, A)
  @test all(==(-1.0), out)
  # a later built-in whose product cannot take the argument: every built-in with a Hessian,
  # for an array and for a block of a tuple
  rng = Xoshiro(3)
  mismatched = (
    "QuadraticFunction" => (QuadraticFunction(Matrix(1.0I, 3, 3)), [1.0, 2.0], [3.0, 4.0]),
    "LeastSquaresLoss" => (LeastSquaresLoss(randn(rng, 6), randn(rng, 6, 3)), [1.0, 2.0], [3.0, 4.0]),
    "LogisticLoss" => (LogisticLoss(Float64.(rand(rng, Bool, 6)), randn(rng, 6, 3)), [1.0, 2.0], [3.0, 4.0]),
    "PoissonLoss" => (PoissonLoss(rand(rng, 0:3, 6), randn(rng, 6, 3)), [1.0, 2.0], [3.0, 4.0]),
    "LogDetLoss" => (LogDetLoss(Matrix(1.0I, 3, 3)), Matrix(1.0I, 2, 2), Matrix(2.0I, 2, 2)),
    "Precompose" => (Precompose(SqrNormL2(), Matrix(1.0I, 3, 4), 1.0), [1.0, 2.0], [3.0, 4.0]))   # A A' = I
  for (name, (q, xq, vq)) in mismatched
    @testset "$name" begin
      out = fill(-7.0, size(xq))
      @test_throws DimensionMismatch hessian_vec!(out, Sum(SqrNormL2(), q), xq, vq)
      @test all(==(-7.0), out)
      @test_throws DimensionMismatch hessian_vec!(out, Sum(SqrNormL2(), Scaled(q, 2)), xq, vq)
      @test all(==(-7.0), out)
      # the same function as a block of a tuple, after a block that is written first
      xt, vt = (x[1], xq), (v[1], vq)
      outt = (fill(-7.0, size(x[1])), fill(-7.0, size(xq)))
      @test_throws DimensionMismatch hessian_vec!(outt, Sum(SeparableSum(SqrNormL2(), SqrNormL2()), SeparableSum(SqrNormL2(), q)), xt, vt)
      @test all(==(-7.0), outt[1]) && all(==(-7.0), outt[2])
      outt = (fill(-7.0, size(x[1])), fill(-7.0, size(xq)))
      @test_throws DimensionMismatch hessian_vec!(outt, SeparableSum(SqrNormL2(), q), xt, vt)
      @test all(==(-7.0), outt[1]) && all(==(-7.0), outt[2])
    end
  end
  # a later function whose inner function does not fit the image of a Precompose
  inner = Precompose(QuadraticFunction(Matrix(1.0I, 3, 3)), Matrix(1.0I, 2, 2), 1.0)   # an image of size 2, a 3 × 3 quadratic
  for q in (inner, Scaled(inner, 2), Regularize(inner, 0.5), Tilt(inner, [1.0, 1.0], 0.0))
    out = fill(-7.0, 2)
    @test_throws DimensionMismatch hessian_vec!(out, Sum(SqrNormL2(), q), [1.0, 2.0], [3.0, 4.0])
    @test all(==(-7.0), out)
    outt = (fill(-7.0, 2), fill(-7.0, 2))
    @test_throws DimensionMismatch hessian_vec!(outt, Sum(SeparableSum(SqrNormL2(), SqrNormL2()), SeparableSum(SqrNormL2(), q)),
                                                ([1.0, 2.0], [1.0, 2.0]), ([3.0, 4.0], [3.0, 4.0]))
    @test all(==(-7.0), outt[1]) && all(==(-7.0), outt[2])
  end
  # LogDetLoss writes a dense matrix of its own precision, found before an earlier function writes
  logdet = LogDetLoss(Matrix(1.0I, 2, 2))
  Xl, Vl = Matrix(1.0I, 2, 2), Matrix(2.0I, 2, 2)
  for wrong in (fill(-7f0, 2, 2), Symmetric(fill(-7.0, 2, 2)))
    before = copy(wrong)
    @test_throws ArgumentError hessian_vec!(wrong, Sum(SqrNormL2(), logdet), Xl, Vl)
    @test wrong == before
    @test_throws ArgumentError hessian_vec!(wrong, logdet, Xl, Vl)
  end
  right = fill(-7.0, 2, 2)
  @test hessian_vec!(right, Sum(SqrNormL2(), logdet), Xl, Vl) === right
  @test right ≈ Vl .+ hessian_vec(logdet, Xl, Vl)
  # an output that refuses off-diagonal writes cannot take a sum: found before the first function writes
  for wrap in (Symmetric, Hermitian)
    wrongout = wrap(fill(-7.0, 2, 2)); before = copy(wrongout)
    @test_throws ArgumentError hessian_vec!(wrongout, Sum(ZeroFunction(), ZeroFunction()), Xl, Vl)
    @test wrongout == before
    tupleout = (fill(-7.0, 2), wrap(fill(-7.0, 2, 2)))
    @test_throws ArgumentError hessian_vec!(tupleout, Sum(SeparableSum(SqrNormL2(), SqrNormL2()), SeparableSum(SqrNormL2(), SqrNormL2())),
                                            ([1.0, 2.0], Xl), ([3.0, 4.0], Vl))
    @test all(==(-7.0), tupleout[1]) && tupleout[2] == before
  end
  # a built-in later function without a Hessian
  nohessian = SeparableSum(SqrNormL2(), NormL1(1.))
  @test !hashessian(Sum(s, nohessian))
  out = map(copy, sentinel)
  @test_throws ArgumentError hessian_vec!(out, Sum(s, s, nohessian), x, v)
  @test out == sentinel
  out = [-1.0, -1.0]
  @test_throws ArgumentError hessian_vec!(out, Sum(SqrNormL2(), NormL1(1.)), [1.0, 2.0], [1.0, 1.0])
  @test out == [-1.0, -1.0]
  # the output must not be the point or the direction, nor one block of either (#324); x === v is allowed
  for f in (Sum(s, s), Sum(s, Scaled(s, 2)), Scaled(Sum(s, s), 3))
    xc, vc = map(copy, x), map(copy, v)
    @test_throws ArgumentError hessian_vec!(xc, f, xc, vc)
    @test_throws ArgumentError hessian_vec!(vc, f, xc, vc)
    @test xc == x && vc == v
    mixed = (similar(x[1]), vc[2])
    @test_throws ArgumentError hessian_vec!(mixed, f, xc, vc)
    @test vc == v
    mixed = (xc[1], similar(x[2]))
    @test_throws ArgumentError hessian_vec!(mixed, f, xc, vc)
    @test xc == x
    @test blocks_approx(hessian_vec(f, x, x), hessian_vec(f, x, map(copy, x)))
  end
  nestedpoint = (x, ([4.0, 5.0], [6.0]))
  nestedout = (map(copy, x), nestedpoint[2])
  @test_throws ArgumentError hessian_vec!(nestedout, Sum(nested, nested), nestedpoint, nestedpoint)
  @test nestedpoint[2] == ([4.0, 5.0], [6.0])
  # wrong arity and block shapes of the output and the direction
  for bad in ((similar(x[1]),), (similar(x[1]), similar(x[2]), similar(x[2])), (similar(x[1]), zeros(2)), (zeros(3), similar(x[2])))
    @test_throws DimensionMismatch hessian_vec!(bad, Sum(s, s), x, v)
  end
  out = map(copy, sentinel)
  for bad in ((v[1],), (v[1], v[2], v[2]), (v[1], [1.0, 2.0]), ([1.0], v[2]))
    @test_throws DimensionMismatch hessian_vec!(out, Sum(s, s), x, bad)
    @test out == sentinel
  end
  @test_throws DimensionMismatch hessian_vec!(out, Sum(s, s), (x[1],), v)
  @test out == sentinel
end

a_qm!(q, f, x) = quadratic_model!(q, f, x)

@testset "zero allocations and inference" begin
  n, p = 50, 10
  X, y = randn(n, p), randn(n)
  x, v = randn(p) ./ 3, randn(p)
  B = randn(2n, p)
  S = Matrix(Symmetric(B'B / 2n))
  P = Matrix(Symmetric(inv(S) + 0.3I))
  V = Matrix(Symmetric(randn(p, p)))
  Hv, HV = similar(x), similar(P)
  fl = LogisticLoss(rand(0:1, n), X; weights = rand(n), offset = randn(n))
  for f in (SqrNormL2(2.), QuadraticFunction(X'X, randn(p)), LeastSquaresLoss(y, X), fl,
            PoissonLoss(rand(0:3, n), X), HuberLoss(y, X, 0.5), Scaled(fl, 0.5), Tilt(fl, randn(p)),
            Translate(fl, randn(p)), Sum(LeastSquaresLoss(y, X), SqrNormL2(0.5)))
    @testset "$(nameof(typeof(f)))" begin
      @test allocs(a_hv!, Hv, f, x, v) == 0
      @inferred hessian_vec!(Hv, f, x, v)
      q = quadratic_model(f, x)
      @test allocs(a_qm!, q, f, x) == 0
      @inferred quadratic_model!(q, f, x)
    end
  end
  xs = SparseIterate(sprandn(p, 0.3))
  @test allocs(a_hv!, Hv, LeastSquaresLoss(y, X), xs, xs) == 0
  @test allocs(a_hv!, Hv, fl, xs, xs) == 0

  f = LogDetLoss(S)
  @test allocs(a_hv!, HV, f, P, V) == 0
  @inferred hessian_vec!(HV, f, P, V)
  q = quadratic_model(f, P)
  @test allocs(a_qm!, q, f, P) == 0
  @inferred quadratic_model!(q, f, P)
  f32 = LogDetLoss(Float32.(S))
  @test allocs(a_hv!, Float32.(HV), f32, Float32.(P), Float32.(V)) == 0

  fk = KronQuadratic(S, P, Matrix(Symmetric(randn(p, p))))
  @test allocs(a_hv!, HV, fk, P, V) == 0
  @test allocs(a_hv!, HV, fk, SymmetricSparseIterate(P), V) == 0
end


# Tilt and Translate on tuples (#159): references apply the Hessians of the blocks to arrays.
@testset "Tilt and Translate Hessians on tuples" begin
  rng = Xoshiro(21)
  X, y = randn(rng, 9, 3), randn(rng, 9)
  f = SeparableSum(LeastSquaresLoss(y, X), SqrNormL2(2.0))
  x, v = (randn(rng, 3), randn(rng, 2, 2)), (randn(rng, 3), randn(rng, 2, 2))
  c = (randn(rng, 3), randn(rng, 2, 2))
  b = (randn(rng, 3), randn(rng, 2, 2))
  expected = (X' * (X * v[1]), 2.0 .* v[2])
  # each function with the factor that multiplies the Hessian of f
  for (g, scale) in ((Tilt(f, c, 1.5), 1.0), (Translate(f, b), 1.0), (Tilt(Translate(f, b), c, 0.5), 1.0),
                     (Translate(Tilt(f, c, 0.5), b), 1.0), (Scaled(Tilt(f, c, 1.0), 3.0), 3.0),
                     (Tilt(Scaled(f, 3.0), c), 3.0), (Translate(Sum(f, f), b), 2.0),
                     (Tilt(Sum(f, Scaled(f, 2)), c), 3.0))
    @test hashessian(g)
    Hv = fresh(x)
    @test hessian_vec!(Hv, g, x, v) === Hv
    @test blocks_approx(Hv, map(t -> scale .* t, expected))
    @test blocks_approx(hessian_vec(g, x, v), Hv)
    # central differences of the gradient along v, with the structure of x kept
    δ = 1e-6
    fd = combine(gradient(g, along(x, v, δ)), gradient(g, along(x, v, -δ)), 2δ)
    @test blocks_approx(Hv, fd, 1e-6)
    # x and v may be the same tuple when Hv is separate
    Hx = fresh(x)
    @test hessian_vec!(Hx, g, x, x) === Hx
    @test blocks_approx(Hx, map(t -> scale .* t, (X' * (X * x[1]), 2.0 .* x[2])))
  end
  # blocks of different sizes, nested tuples and long tuples
  nf = SeparableSum(SeparableSum(SqrNormL2(), SqrNormL2(2)), SeparableSum(SqrNormL2(3), SqrNormL2(4)))
  xn = (([1.0, 2.0], [3.0]), ([4.0 5.0; 6.0 7.0], [8.0, -9.0]))
  vn = (([1.0, -1.0], [2.0]), ([1.0 0.0; 0.0 1.0], [3.0, 4.0]))
  cn = (([0.5, 0.5], [1.0]), ([1.0 0.0; 0.0 1.0], [0.1, -0.1]))
  weights = ((1, 2), (3, 4))
  refn = ((weights[1][1] .* vn[1][1], weights[1][2] .* vn[1][2]), (weights[2][1] .* vn[2][1], weights[2][2] .* vn[2][2]))
  for g in (Tilt(nf, cn, 1.0), Translate(nf, cn))
    Hn = fresh(xn)
    @test hessian_vec!(Hn, g, xn, vn) === Hn
    @test blocks_approx(Hn, refn)
  end
  many = SeparableSum(ntuple(k -> SqrNormL2(k), 40)...)
  xm = ntuple(k -> fill(Float64(k), 2), 40); vm = ntuple(k -> fill(1.0, 2), 40); cm = ntuple(k -> fill(0.1k, 2), 40)
  for g in (Tilt(many, cm, 0.0), Translate(many, cm))
    Hm = fresh(xm)
    @test hessian_vec!(Hm, g, xm, vm) === Hm
    @test blocks_approx(Hm, ntuple(k -> fill(Float64(k), 2), 40))
    hv!(Hv, g, x, v) = hessian_vec!(Hv, g, x, v)
    @test allocs(hv!, Hm, g, xm, vm) == 0
    @inferred hessian_vec!(Hm, g, xm, vm)
  end
  # Float32 variables with Float64 data
  x32, v32 = map(a -> Float32.(a), x), map(a -> Float32.(a), v)
  for g in (Tilt(f, c, 1.5), Translate(f, b))
    H32 = map(a -> zeros(Float32, size(a)), x)
    @test hessian_vec!(H32, g, x32, v32) === H32
    @test H32 isa Tuple{Vector{Float32}, Matrix{Float32}}
    @test blocks_approx(H32, map(t -> Float32.(t), expected), 1e-4)
  end
end

@testset "Tilt and Translate Hessians on tuples check before writing" begin
  s = SeparableSum(SqrNormL2(), SqrNormL2(2))
  x, v = ([1.0, 2.0], [3.0 4.0; 5.0 6.0]), ([0.5, 1.0], [1.0 0.0; 0.0 1.0])
  ok = ([0.1, 0.2], [0.3 0.4; 0.5 0.6])
  bad = ([0.1, 0.2], [0.3, 0.4, 0.5, 0.6])
  sentinel = ([9.0, 8.0], [7.0 7.0; 7.0 7.0])
  fresh_out() = map(copy, sentinel)
  for mk in (p -> Tilt(s, p, 0.0), p -> Translate(s, p))
    # parameters that do not fit, in the final block, in the arity and in the nesting
    for param in (bad, (ok[1],), (ok..., [1.0]), (ok[1], (ok[2],)), (ok[1], zeros(2, 3)))
      out = fresh_out()
      @test_throws DimensionMismatch hessian_vec!(out, mk(param), x, v)
      @test out == sentinel
    end
    # the output and the direction
    for badout in ((sentinel[1],), (sentinel[1], sentinel[2], sentinel[2]), (sentinel[1], zeros(4)), (zeros(3), sentinel[2]))
      out = map(copy, badout)
      @test_throws DimensionMismatch hessian_vec!(out, mk(ok), x, v)
      @test out == map(copy, badout)
    end
    out = fresh_out()
    @test_throws DimensionMismatch hessian_vec!(out, mk(ok), x, (v[1], zeros(4)))
    @test_throws DimensionMismatch hessian_vec!(out, mk(ok), x, (v[1],))
    @test out == sentinel
    # an inner function that fails in a later block: a parameter, and a block without a Hessian
    later = SeparableSum(SqrNormL2(), SqrNormL2(1.0, ones(3)))
    out = fresh_out()
    @test_throws DimensionMismatch hessian_vec!(out, mk(ok) isa Tilt ? Tilt(later, ok, 0.0) : Translate(later, ok), x, v)
    @test out == sentinel
    nohessian = SeparableSum(SqrNormL2(), NormL1(0.1))
    out = fresh_out()
    @test_throws ArgumentError hessian_vec!(out, mk(ok) isa Tilt ? Tilt(nohessian, ok, 0.0) : Translate(nohessian, ok), x, v)
    @test out == sentinel
    # a nested inner wrapper
    out = fresh_out()
    inner = mk(ok) isa Tilt ? Translate(Tilt(s, bad, 0.0), ok) : Tilt(Translate(s, bad), ok, 0.0)
    @test_throws DimensionMismatch hessian_vec!(out, inner, x, v)
    @test out == sentinel
    out = fresh_out()
    inner = mk(ok) isa Tilt ? Scaled(Tilt(nohessian, ok, 0.0), 2.0) : Scaled(Translate(nohessian, ok), 2.0)
    @test_throws ArgumentError hessian_vec!(out, inner, x, v)
    @test out == sentinel
    # Hv must not be x or v, nor a block of them; x and v may be the same
    xa, va = map(copy, x), map(copy, v)
    @test_throws ArgumentError hessian_vec!(xa, mk(ok), xa, va)
    @test_throws ArgumentError hessian_vec!(va, mk(ok), xa, va)
    @test xa == x && va == v
    for outs in ((similar(x[1]), x[2]), (x[1], similar(x[2])), (similar(v[1]), v[2]), (v[1], similar(v[2])))
      keep = map(copy, outs)
      @test_throws ArgumentError hessian_vec!(outs, mk(ok), x, v)
      @test x == ([1.0, 2.0], [3.0 4.0; 5.0 6.0]) && v == ([0.5, 1.0], [1.0 0.0; 0.0 1.0])
    end
    out = fresh(x)
    @test hessian_vec!(out, mk(ok), x, x) === out
    # also when the function is wrapped or nested
    @test_throws ArgumentError hessian_vec!(xa, Scaled(mk(ok), 2.0), xa, va)
    @test_throws ArgumentError hessian_vec!(va, Sum(mk(ok), mk(ok)), xa, va)
    @test xa == x && va == v
  end
end

# a smooth function that counts its Hessian products
mutable struct CountHess <: AbstractFunction
  calls::Int
end
ProximalBase.issmooth(::Type{CountHess}) = true
ProximalBase.hashessian(::Type{CountHess}) = true
ProximalBase.hessian_vec!(Hv::AbstractArray, f::CountHess, x::AbstractArray, v::AbstractArray) = (f.calls += 1; copyto!(Hv, v))

@testset "Tilt and Translate Hessians on tuples refuse restricted outputs before writing" begin
  # output blocks that cannot take every entry; the type decides, no entry is read
  restricted = (
    "Diagonal" => () -> Diagonal(fill(-7.0, 2)),
    "UpperTriangular" => () -> UpperTriangular(fill(-7.0, 2, 2)),
    "LowerTriangular" => () -> LowerTriangular(fill(-7.0, 2, 2)),
    "UnitUpperTriangular" => () -> UnitUpperTriangular(fill(-7.0, 2, 2)),
    "UnitLowerTriangular" => () -> UnitLowerTriangular(fill(-7.0, 2, 2)),
    "Bidiagonal" => () -> Bidiagonal(fill(-7.0, 2), fill(-7.0, 1), :U),
    "Tridiagonal" => () -> Tridiagonal(fill(-7.0, 1), fill(-7.0, 2), fill(-7.0, 1)),
    "SymTridiagonal" => () -> SymTridiagonal(fill(-7.0, 2), fill(-7.0, 1)),
    "UpperHessenberg" => () -> UpperHessenberg(fill(-7.0, 2, 2)),
    "Symmetric" => () -> Symmetric(fill(-7.0, 2, 2)),
    "Hermitian" => () -> Hermitian(fill(-7.0, 2, 2)))
  snap(y::AbstractArray) = Array(y)
  snap(y::Tuple) = map(snap, y)
  same(a, b) = isequal(a, b)
  same(a::Tuple, b::Tuple) = length(a) == length(b) && all(map(same, a, b))
  f = SeparableSum(SqrNormL2(), SqrNormL2())
  x = ([1.0, 2.0], [1.0 2.0; 3.0 4.0])
  v = ([2.0, 3.0], [4.0 5.0; 6.0 7.0])
  c = (zeros(2), zeros(2, 2))
  wrappers(g, c) = ("Tilt" => Tilt(g, c, 0.0), "Translate" => Translate(g, c))

  @testset "$name, $W: direct, in the final block" for (name, make) in restricted, (W, w) in wrappers(f, c)
    Hv = (fill(-7.0, 2), make()); before = snap(Hv)
    @test_throws ArgumentError hessian_vec!(Hv, w, x, v)
    @test same(snap(Hv), before)
    # also when the restricted block comes first
    sq = (x[2], x[2]); sqv = (v[2], v[2])
    w2 = W == "Tilt" ? Tilt(f, (zeros(2, 2), zeros(2, 2)), 0.0) : Translate(f, (zeros(2, 2), zeros(2, 2)))
    Hv = (make(), fill(-7.0, 2, 2)); before = snap(Hv)
    @test_throws ArgumentError hessian_vec!(Hv, w2, sq, sqv)
    @test same(snap(Hv), before)
  end

  # the inner function is not run: the rejected output reaches none of its products
  counted = SeparableSum(CountHess(0), CountHess(0))
  cc = (zeros(2), zeros(2, 2))
  for (name, make) in restricted, (W, w) in wrappers(counted, cc)
    Hv = (fill(-7.0, 2), make()); before = snap(Hv)
    @test_throws ArgumentError hessian_vec!(Hv, w, x, v)
    @test same(snap(Hv), before)
    @test all(g -> g.calls == 0, counted.fs)
  end
  # the counting function does work with an ordinary output
  for (W, w) in wrappers(counted, cc)
    Hv = (fill(-7.0, 2), fill(-7.0, 2, 2))
    @test hessian_vec!(Hv, w, x, v) === Hv
    @test Hv == v && all(g -> g.calls == 1, counted.fs)
    foreach(g -> g.calls = 0, counted.fs)
  end

  # nested tuples: the restricted leaf is last, and every earlier leaf stays as it was
  nf = SeparableSum(SeparableSum(SqrNormL2(), SqrNormL2()), SeparableSum(SqrNormL2(), SqrNormL2()))
  nx = (([1.0, 2.0], [3.0]), ([4.0, 5.0], [1.0 2.0; 3.0 4.0]))
  nv = (([2.0, 3.0], [1.0]), ([0.5, 0.5], [4.0 5.0; 6.0 7.0]))
  nc = (([0.1, 0.1], [0.2]), ([0.3, 0.3], zeros(2, 2)))
  for (name, make) in restricted, (W, w) in wrappers(nf, nc)
    Hv = ((fill(-7.0, 2), fill(-7.0, 1)), (fill(-7.0, 2), make())); before = snap(Hv)
    @test_throws ArgumentError hessian_vec!(Hv, w, nx, nv)
    @test same(snap(Hv), before)
  end

  # an enclosing Scaled finds it
  for (name, make) in restricted, (W, w) in wrappers(counted, cc)
    Hv = (fill(-7.0, 2), make()); before = snap(Hv)
    @test_throws ArgumentError hessian_vec!(Hv, Scaled(w, 2.0), x, v)
    @test same(snap(Hv), before)
    @test all(g -> g.calls == 0, counted.fs)
  end

  # an enclosing Sum finds it before its first summand is evaluated, whichever the order
  for (name, make) in restricted, (W, w) in wrappers(counted, cc)
    first_ok = SeparableSum(CountHess(0), CountHess(0))
    for g in (Sum(first_ok, w), Sum(w, first_ok), Sum(first_ok, Scaled(w, 3.0), first_ok))
      Hv = (fill(-7.0, 2), make()); before = snap(Hv)
      @test_throws ArgumentError hessian_vec!(Hv, g, x, v)
      @test same(snap(Hv), before)
      @test all(h -> h.calls == 0, first_ok.fs)
    end
  end

  # an enclosing SeparableSum: the first block is valid and counted, the second is a tuple wrapper
  for (name, make) in restricted, mk in (cx -> Tilt(f, cx, 0.0), cx -> Translate(f, cx))
    first_ok = CountHess(0)
    sep = SeparableSum(first_ok, mk(c))
    sx, sv = ([1.0, 2.0], x), ([1.0, 1.0], v)
    Hv = (fill(-7.0, 2), (fill(-7.0, 2), make())); before = snap(Hv)
    @test_throws ArgumentError hessian_vec!(Hv, sep, sx, sv)
    @test same(snap(Hv), before) && first_ok.calls == 0
    # and one more level, the wrapper in the middle
    deep = SeparableSum(SeparableSum(CountHess(0), mk(c)), CountHess(0))
    dx, dv = (sx, [3.0]), (sv, [1.0])
    Hv = ((fill(-7.0, 2), (fill(-7.0, 2), make())), fill(-7.0, 1)); before = snap(Hv)
    @test_throws ArgumentError hessian_vec!(Hv, deep, dx, dv)
    @test same(snap(Hv), before)
  end

  # the precedence of the errors: a shape error keeps its DimensionMismatch, whatever the output types
  for (W, w) in wrappers(f, c)
    Hv = (fill(-7.0, 2), Diagonal(fill(-7.0, 2))); before = snap(Hv)
    @test_throws DimensionMismatch hessian_vec!(Hv, w, x, (v[1],))                      # direction arity
    @test_throws DimensionMismatch hessian_vec!(Hv, w, x, (v[1], zeros(3)))             # direction size
    @test_throws DimensionMismatch hessian_vec!((Hv..., fill(0.0, 2)), w, x, v)         # output arity
    @test same(snap(Hv), before)
  end
  Hv = (fill(-7.0, 2), Diagonal(fill(-7.0, 2)))
  @test_throws DimensionMismatch hessian_vec!(Hv, Tilt(f, (c[1], zeros(2, 3)), 0.0), x, v)    # parameter size
  @test_throws DimensionMismatch hessian_vec!(Hv, Translate(f, (c[1], zeros(2, 3))), x, v)
  @test same(snap(Hv), (fill(-7.0, 2), Array(Diagonal(fill(-7.0, 2)))))

  # ordinary dense outputs still work, against an independent reference
  g = SeparableSum(SqrNormL2(2.0), SqrNormL2(3.0))
  expected = (2.0 .* v[1], 3.0 .* v[2])
  for (W, w) in wrappers(g, c)
    Hv = map(similar, x)
    @test hessian_vec!(Hv, w, x, v) === Hv
    @test Hv == expected
    @test hessian_vec!(Hv, Scaled(w, 2.0), x, v) === Hv && Hv == map(t -> 2.0 .* t, expected)
    @test hessian_vec!(map(similar, x), Sum(w, w), x, v) == map(t -> 2.0 .* t, expected)
    Hx = map(similar, x)                                       # x and v may be one tuple when Hv is separate
    @test hessian_vec!(Hx, w, x, x) === Hx && Hx == (2.0 .* x[1], 3.0 .* x[2])
    # a Symmetric output block is rejected (it cannot take the asymmetric product), a dense one is not
    Hs = (similar(x[1]), Symmetric(fill(0.0, 2, 2)))
    @test_throws ArgumentError hessian_vec!(Hs, w, x, v)
  end

  # the aliasing rules are unchanged
  for (W, w) in wrappers(f, c)
    xa, va = map(copy, x), map(copy, v)
    @test_throws ArgumentError hessian_vec!(xa, w, xa, va)
    @test_throws ArgumentError hessian_vec!(va, w, xa, va)
    @test_throws ArgumentError hessian_vec!((similar(xa[1]), xa[2]), w, xa, va)
    @test_throws ArgumentError hessian_vec!((va[1], similar(va[2])), w, xa, va)
    @test xa == x && va == v
  end

  # an array output is not looked at by the tuple check: the array preflight is as it was
  for w in (Tilt(SqrNormL2(), zeros(2, 2), 0.0), Translate(SqrNormL2(), zeros(2, 2)))
    @test PB._check_hessian_shape(w, ones(2, 2), Diagonal(ones(2)), ones(2, 2)) === true
  end

  # no allocation in the preflight or the product of a warmed call, and inference
  hv!(Hv, g, x, v) = hessian_vec!(Hv, g, x, v)
  for (W, w) in wrappers(g, c)
    Hv = map(similar, x)
    @test allocs(hv!, Hv, w, x, v) == 0
    @test allocs(PB._check_second_order_arrays, w, x, Hv, v) == 0
    @inferred hessian_vec!(Hv, w, x, v)
  end
  many = SeparableSum(ntuple(k -> SqrNormL2(k), 40)...)
  xm = ntuple(k -> fill(Float64(k), 2), 40); vm = ntuple(k -> fill(1.0, 2), 40); cm = ntuple(k -> fill(0.1k, 2), 40)
  for (W, w) in wrappers(many, cm)
    Hm = fresh(xm)
    @test hessian_vec!(Hm, w, xm, vm) === Hm
    @test allocs(hv!, Hm, w, xm, vm) == 0
    @test allocs(PB._check_second_order_arrays, w, xm, Hm, vm) == 0
  end
  four = SeparableSum(SqrNormL2(), SqrNormL2(2), SqrNormL2(3), SqrNormL2(4))
  x4, v4, c4 = ntuple(k -> fill(Float64(k), 3), 4), ntuple(k -> fill(1.0, 3), 4), ntuple(k -> fill(0.5, 3), 4)
  for (W, w) in wrappers(four, c4)
    H4 = fresh(x4)
    @test hessian_vec!(H4, w, x4, v4) === H4
    @test allocs(hv!, H4, w, x4, v4) == 0
  end
  for (W, w) in wrappers(nf, nc)
    Hn = fresh(nx)
    @test hessian_vec!(Hn, w, nx, nv) === Hn
    @test allocs(hv!, Hn, w, nx, nv) == 0
  end
end

@testset "quadratic models of Tilt and Translate on tuples" begin
  rng = Xoshiro(31)
  X1, X2 = randn(rng, 30, 3), randn(rng, 25, 2)
  l1 = LogisticLoss(Float64.(rand(rng, Bool, 30)), X1)
  l2 = LeastSquaresLoss(randn(rng, 25), X2)
  sep = SeparableSum(l1, l2)
  x, x′ = (randn(rng, 3) ./ 3, randn(rng, 2)), (randn(rng, 3) ./ 3, randn(rng, 2))
  v = (randn(rng, 3), randn(rng, 2))
  c, b = (randn(rng, 3), randn(rng, 2)), (randn(rng, 3) ./ 3, randn(rng, 2))
  d = 0.7
  Δ = map(t -> t ./ 10, v)
  for (f, shift) in ((Tilt(sep, c, d), nothing), (Translate(sep, b), b))
    @test hashessian(f)
    q = quadratic_model(f, x)
    @test q isa (f isa Tilt ? Tilt : Translate) && issmooth(q) && hashessian(q)
    @test f isa Tilt ? (q.c === f.c && q.d == f.d) : q.b === f.b
    @test blocks_approx(gradient(q, x), gradient(f, x), 1e-10)
    @test blocks_approx(hessian_vec(q, x′, v), hessian_vec(f, x, v), 1e-10)
    g = gradient(f, x)
    @test q(along(x, Δ, 1.0)) - q(x) ≈ tdot(g, Δ) + tdot(Δ, hessian_vec(f, x, Δ)) / 2 rtol = 1e-8
    # updated at a second point, in place
    @test quadratic_model!(q, f, x′) === q
    @test blocks_approx(gradient(q, x′), gradient(f, x′), 1e-10)
    @test blocks_approx(hessian_vec(q, x, v), hessian_vec(f, x′, v), 1e-10)
    @test q(along(x′, Δ, 1.0)) - q(x′) ≈ tdot(gradient(f, x′), Δ) + tdot(Δ, hessian_vec(f, x′, Δ)) / 2 rtol = 1e-8
    # back to the first point
    quadratic_model!(q, f, x)
    @test blocks_approx(gradient(q, x), gradient(f, x), 1e-10)
  end
  # the expected construction: the model of the inner function at x (- b), wrapped again
  qt, qu = quadratic_model(Tilt(sep, c, d), x), quadratic_model(Translate(sep, b), x)
  innerq = quadratic_model(sep, map(-, x, b))
  @test blocks_approx(gradient(qu, x), gradient(innerq, map(-, x, b)), 1e-10)
  @test qt.f isa SeparableSum && qu.f isa SeparableSum
  # nested tuples and the wrappers around them
  nested = SeparableSum(sep, SeparableSum(SqrNormL2(0.5), SqrNormL2(2.0)))
  xn = (x, (randn(rng, 4), randn(rng, 2, 2))); vn = (v, (randn(rng, 4), randn(rng, 2, 2)))
  cn = (c, (randn(rng, 4), randn(rng, 2, 2)))
  for g in (Tilt(nested, cn, 1.0), Translate(nested, cn), Scaled(Translate(nested, cn), 2.0), Tilt(Translate(nested, cn), cn))
    q = quadratic_model(g, xn)
    @test blocks_approx(gradient(q, xn), gradient(g, xn), 1e-10)
    @test blocks_approx(hessian_vec(q, xn, vn), hessian_vec(g, xn, vn), 1e-10)
    @test quadratic_model!(q, g, xn) === q
  end
  # invalid parameters are rejected, and the model of a function without a Hessian
  badc = (c[1], zeros(3))
  @test_throws DimensionMismatch quadratic_model(Tilt(sep, badc, 0.0), x)
  @test_throws DimensionMismatch quadratic_model(Translate(sep, badc), x)
  @test_throws DimensionMismatch quadratic_model(Translate(sep, (c[1],)), x)
  @test_throws DimensionMismatch quadratic_model!(quadratic_model(Tilt(sep, c, 0.0), x), Tilt(sep, badc, 0.0), x)
  @test_throws DimensionMismatch quadratic_model!(quadratic_model(Translate(sep, b), x), Translate(sep, badc), x)
  @test_throws ArgumentError quadratic_model(Tilt(SeparableSum(NormL1(1.0), SqrNormL2()), c, 0.0), x)
  @test_throws ArgumentError quadratic_model(Translate(SeparableSum(NormL1(1.0), SqrNormL2()), b), x)
end

end
