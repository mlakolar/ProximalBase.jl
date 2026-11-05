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

# The exact IRLS pair of one observation in BigFloat: the coefficient w max(exp(η), √eps)
# and the working response η - (exp(η) - y)/max(exp(η), √eps), with the floor of the
# working precision. It uses nothing of GLMLoss, whose own gradient is under recovery.
function irls_reference(::Type{T}, η, w, y) where {T}
  setprecision(BigFloat, 512) do
    e = exp(big(η))
    h = max(e, big(sqrt(eps(T))))
    big(w) * h, big(η) - (e - big(y)) / h, big(w) * (e - big(y)), big(w) * h
  end
end

# the unchanged ordinary computation of the model, as before the staged update
function ordinary_model(f::GLMLoss{L, T}, β) where {L, T}
  η = f.X * β
  f.offset === nothing || (η .+= f.offset)
  v, z = zeros(T, length(η)), zeros(T, length(η))
  for i in eachindex(η)
    w = f.weights === nothing ? one(T) : convert(T, f.weights[i])
    iszero(w) && continue
    d1 = convert(T, PB.deriv(f.loss, η[i], f.y[i]))
    d2 = max(convert(T, PB.deriv2(f.loss, η[i], f.y[i])), sqrt(eps(T)))
    v[i] = w * d2
    z[i] = η[i] - d1 / d2
  end
  v, z
end

near(v, ref, k, T = typeof(v)) = abs(big(v) - ref) <= k * big(eps(abs(T(ref))))
split_allocations(x) = (PB._exp_split(x); @allocated PB._exp_split(x))

@testset "GLMLoss quadratic models at large Poisson predictors (#317)" begin
  poisson(T, η, w, y) = PoissonLoss(T[y], ones(T, 1, 1); weights = T[w])

  @testset "the pair, the gradient and the Hessian against exact references" begin
    # (η, w, y), exceptional and ordinary; 710 with w = 1/8 has exp(710) > floatmax
    for T in (Float64, Float32)
      big_η = T === Float64 ? 710 : 89
      cases = [(0, 1, 0), (0, 1, 3), (T === Float64 ? 700 : 88, 1, 0), (T === Float64 ? 700 : 88, 1, 2),
               (big_η, 1 / 8, 0), (big_η, 1 / 8, 5), (big_η + 10, 2.0^-20, 0), (big_η, 2.0^-40, 3),
               (-30, 1, 2), (T === Float64 ? -800 : -100, 1, 3 // 2)]
      for (η, w, y) in cases
        f = poisson(T, η, w, y)
        q = quadratic_model(f, T[η])
        v, z, g, H = irls_reference(T, T(η), T(w), T(y))
        @test near(only(q.weights), v, 8, T)
        @test only(q.y) ≈ z rtol = 20 * eps(T)
        # the gradient of the model at the point is the gradient of f: w (exp(η) - y)
        @test only(gradient(q, T[η])) ≈ g rtol = 1000 * eps(T)
        # and its Hessian is the effective coefficient (X = 1)
        @test near(only(hessian_vec(q, T[η], T[1])), H, 8, T)
        @test isfinite(only(q.weights)) && isfinite(only(q.y))
      end
    end
    # the working response of the issue
    q = quadratic_model(poisson(Float64, 710, 1 / 8, 0), [710.0])
    @test only(q.y) == 709
    @test near(only(q.weights), exp(big(710)) / 8, 8)
    # a deep recovery: w = 2^-1000 at η = 1400, exp(1400) ≈ 2^2019.7
    q = quadratic_model(poisson(Float64, 1400, 2.0^-1000, 0), [1400.0])
    @test near(only(q.weights), exp(big(1400)) * big(2)^-1000, 8)
    @test only(q.y) == 1399
  end

  @testset "unrepresentable coefficients raise PrecisionError" begin
    for (η, w) in ((710, 1.0), (710, 3.0), (800, 1.0), (700, 1e10), (1000, 1e-100), (1400, 2.0^-990))
      f = poisson(Float64, η, w, 0)
      @test_throws PrecisionError quadratic_model(f, [Float64(η)])
      @test_throws ProximalBase.PrecisionError quadratic_model!(quadratic_model(f, [0.0]), f, [Float64(η)])
    end
    @test_throws PrecisionError quadratic_model(poisson(Float32, 89, 1, 0), Float32[89])
    @test_throws PrecisionError quadratic_model(poisson(Float32, 100, 2.0^-10, 0), Float32[100])
    e = try quadratic_model(poisson(Float64, 710, 1, 0), [710.0]); nothing catch e e end
    @test e isa PrecisionError
    @test occursin("observation 1", e.msg) && occursin("unchanged", e.msg)
    # non-finite inputs propagate as before, they are not a range failure
    @test quadratic_model(poisson(Float64, 0, 1, 0), [NaN]) isa GLMLoss
    @test quadratic_model(poisson(Float64, 0, 1, 0), [Inf]) isa GLMLoss
    fw = poisson(Float64, 0, 1, 0); fw.weights[1] = NaN
    @test all(isnan, quadratic_model(fw, [0.0]).weights)
    # a response y exp(-η) that leaves Float32 with a Float64 y is not dropped
    f32 = GLMLoss(PB.Poisson(), [1e300], ones(Float32, 1, 1), [1e-10], nothing, zeros(Float32, 1), zeros(Float32, 1))
    @test_throws PrecisionError quadratic_model(f32, Float32[110])
    # a model that uses the buffers of the loss crosswise is refused before anything is written
    fa = PoissonLoss([0.0, 1.0], ones(2, 1))
    qa = GLMLoss(PB.Squared(), fa.d, fa.X, fa.η, nothing, zeros(2), zeros(2))
    @test_throws ArgumentError quadratic_model!(qa, fa, [0.5])
    # a coefficient that rounds to zero: the gradient w ℓ' would be lost
    @test_throws PrecisionError quadratic_model(poisson(Float64, 0, 5e-324, 2), [-30.0])
    @test_throws PrecisionError quadratic_model(LogisticLoss([1.0], ones(1, 1); weights = [5e-324]), [0.0])
    # a subnormal coefficient is a coefficient
    @test only(quadratic_model(poisson(Float64, 0, 5e-324, 2), [0.0]).weights) == 5e-324
  end

  @testset "the representability boundary" begin
    for T in (Float64, Float32)
      ηmax = T(log(big(floatmax(T))))
      inside, outside = 0, 0
      for η in (prevfloat(ηmax, 3), prevfloat(ηmax), ηmax, nextfloat(ηmax), nextfloat(ηmax, 3))
        r = exp(big(η)) / floatmax(T)
        abs(r - 1) < 64 * eps(T) && continue           # within the error of the computation
        f = poisson(T, η, 1, 0)
        if r < 1
          inside += 1
          q = quadratic_model(f, T[η])
          @test near(only(q.weights), exp(big(η)), 8, T)
        else
          outside += 1
          @test_throws PrecisionError quadratic_model(f, T[η])
        end
      end
      @test inside >= 1 && outside >= 1
      # weights scale the boundary
      for k in (-3, 0, 3)
        w = T(2)^k
        η = T(ηmax + log(T(2)) * (-k) + T(0.5))        # w exp(η) ≈ √e floatmax
        @test_throws PrecisionError quadratic_model(poisson(T, η, w, 0), T[η])
        η = η - 1                                      # w exp(η) ≈ floatmax/√e
        q = quadratic_model(poisson(T, η, w, 0), T[η])
        @test near(only(q.weights), big(w) * exp(big(η)), 8, T)
      end
    end
  end

  @testset "zero weights" begin
    f = PoissonLoss([0.0, 1.0], [1.0; 1.0;;]; weights = [0.0, 1.0])
    q = quadratic_model(f, [710.0 - 700])             # η = 10: ordinary
    @test q.weights[1] == 0 && q.y[1] == 0
    # a zero-weight observation at an overflowing predictor takes no part
    f = PoissonLoss([0.0, 1.0], [710.0 0.0; 0.0 1.0]; weights = [0.0, 1.0])
    q = quadratic_model(f, [1.0, 0.0])
    @test q.weights == [0.0, 1.0] && q.y == [0.0, -0.0 - (1.0 - 1.0) / 1.0]
    q = quadratic_model(PoissonLoss([0.0], ones(1, 1); weights = [0.0]), [710.0])
    @test q.weights == [0.0] && q.y == [0.0]
  end

  @testset "atomic updates" begin
    allfields(a, b) = all(isequal(getfield(a, k), getfield(b, k)) for k in fieldnames(typeof(a)))
    # failure at the last, the middle and the first of three observations
    for (X, β0, β1, bad) in (([1.0 0.0; 0.5 0.5; 0.0 720.0], [0.0, 0.0], [1.0, 1.0], 3),
                              ([1.0 0.0; 720.0 0.0; 0.0 1.0], [0.0, 0.0], [1.0, 0.5], 2),
                              ([720.0 0.0; 1.0 0.0; 0.0 1.0], [0.0, 0.0], [1.0, 0.5], 1))
      f = PoissonLoss([0.0, 1.0, 2.0], X; weights = [1.0, 0.5, 2.0], offset = [0.0, 0.1, 0.2])
      q = quadratic_model(f, β0)
      before = deepcopy(q)
      y0, w0 = q.y, q.weights
      e = try quadratic_model!(q, f, β1); nothing catch e e end
      @test e isa PrecisionError
      @test occursin("observation $bad", e.msg)
      @test allfields(q, before)
      @test q.y === y0 && q.weights === w0                      # the same arrays, the same contents
      # a valid update afterwards is the model at that point
      β2 = [0.4, 0.3]
      @test quadratic_model!(q, f, β2) === q
      q2 = quadratic_model(f, β2)
      @test q.y == q2.y && q.weights == q2.weights
      @test q.y === y0 && q.weights === w0
      v, z = ordinary_model(f, β2)
      @test q.y == z && q.weights == v
    end
    # the allocating call fails the same way
    f = PoissonLoss([0.0, 0.0], reshape([1.0, 710.0], 2, 1))
    @test_throws PrecisionError quadratic_model(f, [1.0])
    # a failure after an earlier failure, then success, repeatedly
    q = quadratic_model(f, [0.0])
    for β in ([1.0], [0.0], [0.5], [1.0], [0.25])
      before = deepcopy(q)
      if β[1] == 1.0
        @test_throws PrecisionError quadratic_model!(q, f, β)
        @test allfields(q, before)
      else
        @test quadratic_model!(q, f, β) === q
        @test q.weights == quadratic_model(f, β).weights
      end
    end
  end

  @testset "no change where nothing leaves the range" begin
    Random.seed!(317)
    n, p = 30, 4
    X = randn(n, p)
    β = randn(p) ./ 3
    w = rand(n); w[3] = 0
    o = randn(n) ./ 4
    for T in (Float64, Float32)
      X_, β_, w_, o_ = T.(X), T.(β), T.(w), T.(o)
      for f in (LogisticLoss(T.(rand(0:1, n)), X_; weights = w_, offset = o_),
                PoissonLoss(T.(rand(0:4, n)), X_; weights = w_, offset = o_),
                PoissonLoss(T.(rand(0:4, n)), X_),
                HuberLoss(T.(10 .* randn(n)), X_, T(0.1); weights = w_),
                HuberLoss(T.(randn(n)), X_, T(0.5)))
        q = quadratic_model(f, β_)
        v, z = ordinary_model(f, β_)
        @test q.weights == v && q.y == z                       # the same bits as the ordinary computation
        @test quadratic_model!(q, f, 2 .* β_) === q
        v, z = ordinary_model(f, 2 .* β_)
        @test q.weights == v && q.y == z
      end
      # the model's gradient is exact where the Huber curvature is floored
      fh = HuberLoss(T.(10 .* randn(n)), X_, T(0.1))
      @test gradient(quadratic_model(fh, β_), β_) ≈ gradient(fh, β_) rtol = 100 * eps(T)
    end
    # Poisson at moderate and large predictors, an ordinary X, offsets: the model's gradient
    f = PoissonLoss(rand(0:4, n), X; weights = w, offset = o)
    for s in (0.0, 1.0, 100.0)
      b = s .* sign.(β) .+ β .* (s == 0 ? 1 : 0)
      η = X * b .+ o
      maximum(η) < 700 || continue
      g = X' * (w .* (exp.(η) .- f.y))
      @test gradient(quadratic_model(f, b), b) ≈ g rtol = 1e-9
    end
  end
end

@testset "GLMLoss Poisson models: range recovery with exp(η) finite, the exponent bound, aliasing (#317)" begin
  poisson(T, η, w, y) = PoissonLoss(T[y], ones(T, 1, 1); weights = T[w])
  representable(T, η, w) = big(w) * exp(big(η)) < big(floatmax(T)) * (1 + big(2.0)^(-precision(T)))

  @testset "exp(η) is finite and the response or the coefficient is not formed by the ordinary expressions" begin
    for T in (Float64, Float32)
      η = T === Float64 ? T(700) : T(88)
      w = T === Float64 ? T(1e-306) : T(1e-36)
      for y in (-floatmax(T), -floatmax(T) / 2, -floatmax(T) / 3, floatmax(T), T(5))
        f = poisson(T, η, w, y)
        v, z, g, H = irls_reference(T, η, w, y)
        q = quadratic_model(f, T[η])
        @test near(only(q.weights), v, 8, T)
        @test only(q.y) ≈ z rtol = 20 * eps(T)
        @test only(gradient(q, T[η])) ≈ g rtol = 1000 * eps(T)
        @test near(only(hessian_vec(q, T[η], T[1])), H, 8, T)
        # the ordinary expressions leave the range for a large negative y
        y == -floatmax(T) && @test !isfinite(only(ordinary_model(f, T[η])[2]))
      end
    end
    # η = 700, y = -floatmax, w = 1e-306 of the report
    q = quadratic_model(poisson(Float64, 700, 1e-306, -floatmax(Float64)), [700.0])
    @test isfinite(only(q.y)) && isfinite(only(q.weights))
    v, z, g, H = irls_reference(Float64, 700.0, 1e-306, -floatmax(Float64))
    @test only(q.y) ≈ z rtol = 1e-14
    @test near(only(q.weights), v, 8)
    # the effective-coefficient boundary w exp(η) = floatmax at η = 700, both sides
    for T in (Float64, Float32)
      η = T === Float64 ? T(700) : T(88)
      wlim = big(floatmax(T)) / exp(big(η))
      for (sgn, ok) in ((-1, true), (1, false))
        w = T(wlim * (1 + sgn * big(10)^-6))
        f = poisson(T, η, w, 0)
        if ok
          q = quadratic_model(f, T[η])
          @test near(only(q.weights), big(w) * exp(big(η)), 8, T)
          @test only(q.y) == η - 1
        else
          @test_throws PrecisionError quadratic_model(f, T[η])
        end
      end
      # the response boundary at η = -1: z = η - 1 + y exp(1) leaves the range with |y|
      ylim = big(floatmax(T)) / exp(big(1))
      for s in (1, -1), (sgn, ok) in ((-1, true), (1, false))
        y = T(s * ylim * (1 + sgn * big(10)^-6))
        f = poisson(T, -1, 1, y)
        if ok
          q = quadratic_model(f, T[-1])
          _, z, = irls_reference(T, T(-1), T(1), y)
          @test only(q.y) ≈ z rtol = 20 * eps(T)
        else
          @test_throws PrecisionError quadratic_model(f, T[-1])
        end
      end
    end
    # bit patterns of the ordinary path: no recovery where the ordinary expressions are finite
    for T in (Float64, Float32), η in (T(-30), T(-5), T(0), T(3), T(50)), y in (T(0), T(2.5), T(1e10), -T(1e10))
      f = poisson(T, η, T(0.7), y)
      v, z = ordinary_model(f, T[η])
      q = quadratic_model(f, T[η])
      @test only(q.weights) === only(v) && only(q.y) === only(z)
    end
  end

  @testset "the exponent bound: no conversion of a huge exponent" begin
    # the shared helper: exp(x) = m 2^j, Int exponent, nothing converted beyond the bound
    @test PB._exp_split(1e20) == (Inf, 0) && PB._exp_split(-1e100) == (0.0, 0)
    @test PB._exp_split(1f20) == (Inf32, 0) && PB._exp_split(-floatmax(Float32)) == (0f0, 0)
    @test PB._exp_split(floatmax(Float64)) == (Inf, 0) && PB._exp_split(-floatmax(Float64)) == (0.0, 0)
    @test PB._exp_split(Inf) == (Inf, 0) && PB._exp_split(-Inf) == (0.0, 0)
    @test isnan(first(PB._exp_split(NaN))) && isnan(first(PB._exp_split(NaN32)))
    for T in (Float64, Float32)
      bound = PB._exp_bound(T)
      @test @inferred(PB._exp_split(T(3))) isa Tuple{T, Int}
      @test @inferred(PB._exp_split(T(1e30))) isa Tuple{T, Int}
      @test split_allocations(T(710)) == 0
      # beyond the bound no weight recovers the product: above it the smallest weight overflows,
      # below it the largest weight rounds to zero (half the smallest subnormal)
      @test big(nextfloat(zero(T))) * exp(big(bound)) > big(floatmax(T)) * (1 + big(2.0)^(-precision(T)))
      @test big(floatmax(T)) * exp(-big(bound)) < big(nextfloat(zero(T))) / 2
      @test PB._exp_split(nextfloat(bound)) == (T(Inf), 0) && PB._exp_split(-nextfloat(bound)) == (zero(T), 0)
      # at the bound the exponent is small and the reduction is the usual one, within 2 ulp
      for x in (bound, -bound, T(0), T(1), T(-1), T(0.5) * bound, T(-0.5) * bound, T(0.9) * bound, T(1e-5))
        m, j = PB._exp_split(x)
        @test abs(j) < 2^12
        @test abs(big(m) * big(2.0)^j - exp(big(x))) <= 2 * big(eps(T)) * exp(big(x))
      end
    end
    @test PB._exp_split(big(10.0)) == frexp(exp(big(10.0)))     # the fallback for another type
    for (T, ηs) in ((Float64, (1e20, 1e100, 1e300, floatmax(Float64), 2000.0, 1456.0)),
                    (Float32, (1f10, 1f20, 1f38, floatmax(Float32), 300f0, 194f0)))
      for η in ηs, w in (one(T), nextfloat(zero(T)), floatmax(T))
        f = poisson(T, η, w, 0)
        @test_throws PrecisionError quadratic_model(f, T[η])
        q = quadratic_model(f, T[0])
        before = deepcopy(q)
        @test_throws PrecisionError quadratic_model!(q, f, T[η])
        @test q.y == before.y && q.weights == before.weights
      end
      # the smallest weight reaches the farthest: both sides of the boundary
      wmin = nextfloat(zero(T))
      for η in (T === Float64 ? (1453.0, 1454.0, 1455.0) : (190f0, 191f0, 192f0, 193f0))
        f = poisson(T, η, wmin, 0)
        if representable(T, η, wmin)
          q = quadratic_model(f, T[η])
          @test near(only(q.weights), big(wmin) * exp(big(η)), 8, T)
          @test only(q.y) == η - 1
        else
          @test_throws PrecisionError quadratic_model(f, T[η])
        end
      end
      # a very negative predictor floors, with no exponent to convert
      for η in (T(-1e20), -floatmax(T))
        q = quadratic_model(poisson(T, η, 1, 1), T[η])
        @test only(q.weights) == sqrt(eps(T)) && isfinite(only(q.y))
      end
    end
  end

  @testset "a permitted non-finite observation does not hide an unrepresentable one" begin
    allfields(a, b) = all(isequal(getfield(a, k), getfield(b, k)) for k in fieldnames(typeof(a)))
    # observation 1 has a NaN response (propagates), observation 2 an unrepresentable coefficient
    f = PoissonLoss([NaN, 0.0], reshape([1.0, 710.0], 2, 1))
    q = quadratic_model(f, [0.0])
    @test isnan(q.y[1]) && q.weights == [1.0, 1.0]
    before = deepcopy(q)
    e = try quadratic_model!(q, f, [1.0]); nothing catch e e end
    @test e isa PrecisionError && occursin("observation 2", e.msg)
    @test allfields(q, before)
    @test quadratic_model!(q, f, [0.25]) === q
    q2 = quadratic_model(f, [0.25])
    @test isequal(q.y, q2.y) && q.weights == q2.weights
    # non-finite data alone propagate, with no PrecisionError, in each of its forms
    for f in (PoissonLoss([NaN, 0.0], ones(2, 1)), PoissonLoss([Inf, 0.0], ones(2, 1)), PoissonLoss([0.0, 1.0], [Inf; 1.0;;]))
      q = quadratic_model(f, [0.5])
      @test q isa GLMLoss && !(all(isfinite, q.y) && all(isfinite, q.weights))
      @test quadratic_model!(q, f, [0.25]) === q
    end
    # a non-finite response at a predictor where exp overflows (or the coefficient is not representable):
    # the data are not finite, so the update returns and the response is not finite; no PrecisionError,
    # and the in-place update equals the allocating one
    for T in (Float64, Float32), η in (T(89), T(710), T(1e4)), y in (T(NaN), T(Inf), T(-Inf)), w in (T(1), T(1 / 8))
      f = PoissonLoss(T[y], ones(T, 1, 1); weights = T[w], offset = T[η])
      q = quadratic_model(f, T[0])
      @test !isfinite(q.y[1])
      @test quadratic_model!(q, f, T[0]) === q
      @test isequal(q.y, quadratic_model(f, T[0]).y) && isequal(q.weights, quadratic_model(f, T[0]).weights)
    end
    # zero weights: a zero-weight observation with a non-finite or overflowing predictor is zero, whatever its data
    f = PoissonLoss([NaN, 0.0, 1.0], [1.0; 720.0; 1.0;;]; weights = [0.0, 0.0, 1.0])
    q = quadratic_model(f, [1.0])
    @test q.weights[1:2] == [0.0, 0.0] && q.y[1:2] == [0.0, 0.0] && isfinite(q.y[3]) && q.weights[3] == exp(1.0)
    # all weights zero at any predictor
    q = quadratic_model(PoissonLoss([0.0, 0.0], reshape([1.0, 1e20], 2, 1); weights = zeros(2)), [1.0])
    @test q.weights == [0.0, 0.0] && q.y == [0.0, 0.0]
  end

  @testset "the exceptional cases of the report: coefficient, response, gradient and Hessian against BigFloat" begin
    for (T, η, w, y) in ((Float64, 700, 1e-306, -floatmax(Float64)), (Float64, 710, 1 / 8, 0), (Float64, 1400, 2.0^-1000, 0),
                         (Float64, 1400, 2.0^-1000, 3), (Float64, 1454, 5e-324, 0), (Float64, 710, 2.0^-40, 3),
                         (Float32, 88, 1f-36, -floatmax(Float32)), (Float32, 89, 1 / 8, 0), (Float32, 190, 2f0^-149, 0))
      f = poisson(T, η, w, y)
      q = quadratic_model(f, T[η])
      v, z, g, H = irls_reference(T, T(η), T(w), T(y))
      @test near(only(q.weights), v, 8, T)
      @test only(q.y) ≈ z rtol = 20 * eps(T)
      @test only(gradient(q, T[η])) ≈ g rtol = 1000 * eps(T)
      @test near(only(hessian_vec(q, T[η], T[1])), H, 8, T)
    end
    # unrepresentable: PrecisionError, never clamped
    for (T, η, w) in ((Float64, 710, 1.0), (Float64, 1400, 2.0^-990), (Float32, 89, 1.0))
      @test_throws PrecisionError quadratic_model(poisson(T, η, w, 0), T[η])
    end
    # the floor and underflow to zero: a coefficient that is subnormal is kept, one that rounds to zero is not
    @test only(quadratic_model(poisson(Float64, -800, 1e-310, 1), [-800.0]).weights) == 1e-310 * sqrt(eps())
    @test_throws PrecisionError quadratic_model(poisson(Float64, -800, 1e-320, 1), [-800.0])
    # normal controls: Logistic and Huber agree with the ordinary computation bit for bit
    n, p = 20, 3
    X, β = randn(n, p), randn(p) ./ 3
    for f in (LogisticLoss(rand(0:1, n), X), HuberLoss(10 .* randn(n), X, 0.1))
      q = quadratic_model(f, β)
      v, z = ordinary_model(f, β)
      @test q.weights == v && q.y == z
    end
  end

  @testset "below the floor: a response whose terms leave the range while their sum does not (review of #544)" begin
    for T in (Float64, Float32)
      h = sqrt(eps(T))
      η = -floatmax(T)
      # η + y/√eps with y/√eps above floatmax: the sum is representable (0.5 floatmax, 0.75 floatmax, 0, ...)
      for c in (T(1.5), T(1.75), T(1), T(1.25), T(1.999))
        y = c * (floatmax(T) * h)
        f = poisson(T, η, 1, y)
        q = quadratic_model(f, T[η])
        v, z, = irls_reference(T, η, T(1), y)
        @test isfinite(only(q.y)) && only(q.weights) == h
        @test isapprox(only(q.y), T(z); atol = 4 * eps(T) * floatmax(T), rtol = 0)
        # the in-place update, from a model at another response, gives the same pair
        f2 = poisson(T, η, 1, T(0.5) * (floatmax(T) * h))
        qq = quadratic_model(f2, T[η])
        f2.y[1] = y
        before_y = copy(qq.y)
        @test quadratic_model!(qq, f2, T[η]) === qq
        @test before_y != qq.y && only(qq.y) == only(q.y) && only(qq.weights) == only(q.weights)
        c > 1 && @test only(ordinary_model(f, T[η])[2]) == T(Inf)       # the direct expression overflows
      end
      # the classification at the upper boundary is that of the exactly rounded sum, on both sides
      let rng = MersenneTwister(544 + sizeof(T))
        mism = 0
        for _ in 1:3000
          η = -floatmax(T) * (T(0.02) + T(0.9) * rand(rng, T))
          # y so that η + y/√eps is within a few 1e-9 (Float32) or 1e-15 (Float64) of floatmax, either side
          tgt = big(floatmax(T)) * (1 + (rand(rng) - 0.5) * 8 * big(eps(T)))
          y = T((tgt - big(η)) * big(h))
          ref = setprecision(BigFloat, 512) do
            e = exp(big(η)); T(big(η) - (e - big(y)) / big(h))
          end
          f = poisson(T, η, 1, y)
          got = try only(quadratic_model(f, T[η]).y) catch err; err end
          ok = isfinite(ref) ? (got isa T && got == ref) : got isa PrecisionError
          mism += !ok
        end
        @test mism == 0
      end
      # the sum itself does not fit: PrecisionError, and the model is unchanged
      for c in (T(2.5), T(3))
        y = c * (floatmax(T) * h)
        f = poisson(T, η, 1, y)
        q = quadratic_model(f, T[0])
        before = deepcopy(q)
        e = try quadratic_model!(q, f, T[η]); nothing catch e e end
        @test e isa PrecisionError && occursin("observation 1", e.msg)
        @test q.y == before.y && q.weights == before.weights
        @test_throws PrecisionError quadratic_model(f, T[η])
      end
      # the negative side
      y = -T(1.5) * (floatmax(T) * h)
      @test_throws PrecisionError quadratic_model(poisson(T, η, 1, y), T[η])
      # an ordinary finite pair below the floor is unchanged
      f = poisson(T, -40, 1, 2)
      @test quadratic_model(f, T[-40]).y == ordinary_model(f, T[-40])[2]
    end
    # an infinite or NaN predictor, response and weight still propagate
    for bad in (-Inf, NaN)
      q = quadratic_model(poisson(Float64, 0, 1, 1), [bad])
      @test !isfinite(only(q.y))
    end
    for y in (Inf, NaN, -Inf)
      @test !isfinite(only(quadratic_model(poisson(Float64, -floatmax(Float64), 1, y), [-floatmax(Float64)]).y))
    end
  end

  @testset "a subnormal coefficient of the recovery is rounded once; no global BigFloat state (review of #544)" begin
    exp_split = ProximalBase._exp_split
    # the report: Float32 scratch, Float64 response out of Float32's range, so that the response is recovered
    η, w, y = 2.2637296f0, 9.17786f-40, 3.7431058130238175e38
    f = GLMLoss(ProximalBase.Poisson(), [y], ones(Float32, 1, 1), Float32[w], nothing, zeros(Float32, 1), zeros(Float32, 1))
    m, j = exp_split(η)
    ref = setprecision(BigFloat, 256) do; Float32(ldexp(big(w) * big(m), j)); end
    @test only(quadratic_model(f, Float32[η]).weights) == ref
    let rng = MersenneTwister(5440)
      bad = 0
      for _ in 1:600
        # Float32 loss, Float64 response beyond floatmax(Float32): the response is recovered and finite where e^η > 1.1
        T = Float32
        ηT = T(0.2 + 5.8 * rand(rng)); wT = T(1.2e-38 * rand(rng) * 10.0^(-rand(rng) * 7))
        iszero(wT) && continue
        fT = GLMLoss(ProximalBase.Poisson(), [3.7e38], ones(T, 1, 1), T[wT], nothing, zeros(T, 1), zeros(T, 1))
        mT, jT = exp_split(ηT)
        refT = setprecision(BigFloat, 256) do; T(ldexp(big(wT) * big(mT), jT)); end
        got = try only(quadratic_model(fT, T[ηT]).weights) catch err; err end
        bad += !(refT == 0 ? got isa PrecisionError : got == refT)
      end
      @test bad == 0
    end
    # the floor response uses no BigFloat: the global precision is not touched, and nothing allocates
    for ambient in (32, 64, 512)
      setprecision(BigFloat, ambient) do
        for T in (Float32, Float64)
          η = -floatmax(T); y = T(1.5) * (floatmax(T) * sqrt(eps(T)))
          f = poisson(T, η, 1, y)
          b = T[η]
          q = quadratic_model(f, b)
          @test precision(BigFloat) == ambient
          quadratic_model!(q, f, b)
          @test (@allocated quadratic_model!(q, f, b)) == 0
          @test isfinite(only(q.y))
        end
      end
    end
  end

  @testset "Poisson coefficient w exp(η) at the overflow threshold: classification and correct rounding (round 7 of #544)" begin
    # The coefficient of one observation with y = 0 (so the response η - 1 is always finite) is the correctly
    # rounded w exp(η): finite (floatmax included) when w exp(η) < M = floatmax + ulp(floatmax)/2, and a
    # PrecisionError, the model unchanged, when w exp(η) ≥ M. The references below are BigFloat at an explicit
    # precision, written from this statement and nothing of the implementation.
    bigprec(f, p) = setprecision(f, BigFloat, p)
    threshold(::Type{T}, p) where {T} = bigprec(p) do; BigFloat(floatmax(T)) + BigFloat(eps(floatmax(T))) / 2; end
    wexp_ref(::Type{T}, w, η, p) where {T} = bigprec(p) do; T(BigFloat(w) * exp(BigFloat(η))); end
    one_obs(::Type{T}, w) where {T} = PoissonLoss(T[0], ones(T, 1, 1); weights = T[w])
    outcome(f, ::Type{T}, η) where {T} = try only(quadratic_model(f, T[η]).weights) catch err; err end
    allfields_equal(a, b) = all(isequal(getfield(a, k), getfield(b, k)) for k in fieldnames(typeof(a)))
    # name, η, w, whether the coefficient is finite (as the references say)
    named = [("A recovery, F32, finite", 154.59f0, 2.4788888f-29, true),
             ("B recovery, F32, overflow", 93f0, 0.013882019f0, false),
             ("C recovery, F64, finite", 719.0, 9.930773381465837e-5, true),
             ("D recovery, F64, overflow", 710.0, 0.8046989017574928, false),
             ("E ordinary, F64, overflow", nextfloat(2.0^-54), floatmax(Float64), false),
             ("F ordinary, F64, finite", 2.0^-54, floatmax(Float64), true),
             ("G ordinary, F32, overflow", nextfloat(2f0^-25), floatmax(Float32), false),
             ("H ordinary, F32, finite", 2f0^-25, floatmax(Float32), true)]
    near_lists = Dict{DataType, Vector{Any}}()          # filled by the sweep: (η, w, |w exp(η)/M - 1|)

    @testset "the threshold M of the references" begin
      # M is the midpoint of floatmax and 2^emax: 2^1024 (1 - 2^-54) and 2^128 (1 - 2^-25); it rounds to Inf, its predecessor does not
      m64 = bigprec(256) do; big(2.0)^1024 * (1 - big(2.0)^-54); end
      m32 = bigprec(256) do; big(2.0)^128 * (1 - big(2.0)^-25); end
      @test threshold(Float64, 256) == m64
      @test threshold(Float32, 256) == m32
      for T in (Float32, Float64)
        Mb = threshold(T, 256)
        below = bigprec(256) do; prevfloat(Mb); end
        inside = bigprec(256) do; BigFloat(floatmax(T)) * (1 + big(2.0)^-(precision(T) + 4)); end
        @test isinf(T(Mb)) && T(below) === floatmax(T) && T(inside) === floatmax(T)
      end
    end

    @testset "minimal regressions A-H: $(case[1])" for case in named
      name, η, w, fin = case
      T = typeof(η)
      r1, r2 = wexp_ref(T, w, η, 1024), wexp_ref(T, w, η, 2048)
      @test isequal(r1, r2)
      @test isfinite(r1) == fin
      f = one_obs(T, w)
      q = quadratic_model(f, T[0])                      # a valid model, at η = 0
      before = deepcopy(q)
      if fin
        m = quadratic_model(f, T[η])
        @test only(m.weights) === r1 && isfinite(only(m.weights)) && isfinite(only(m.y))
        @test only(m.y) == η - one(T)                   # η - ℓ'/ℓ'' = η - 1
        @test quadratic_model!(q, f, T[η]) === q
        @test only(q.weights) === r1 && isfinite(only(q.weights)) && isfinite(only(q.y))
        @test isequal(q.y, m.y) && isequal(q.weights, m.weights)
      else
        @test_throws PrecisionError quadratic_model(f, T[η])
        @test_throws PrecisionError quadratic_model!(q, f, T[η])
        e1 = try quadratic_model(f, T[η]); nothing catch e e end
        e2 = try quadratic_model!(q, f, T[η]); nothing catch e e end
        for e in (e1, e2)
          @test e isa PrecisionError && occursin("observation 1", sprint(showerror, e))
        end
        @test allfields_equal(q, before)
        # a later valid update succeeds: at η = 0 the coefficient is w itself
        @test quadratic_model!(q, f, T[0]) === q
        @test only(q.weights) === w && only(q.y) == -one(T)
        @test quadratic_model!(q, f, T[-1]) === q
        @test isapprox(only(q.weights), T(wexp_ref(Float64, w, T(-1), 256)); rtol = 8 * eps(T))
      end
    end

    @testset "a failure at position 1, 2 or 3 of three observations is atomic" begin
      for (T, η, w) in ((Float64, 710.0, 0.8046989017574928), (Float32, 93f0, 0.013882019f0))   # D and B
        @test !isfinite(wexp_ref(T, w, η, 1024))
        for pos in 1:3
          ws = T[w / 4, w / 16, w / 4]                    # the others are in the recovery range and representable
          ws[pos] = w
          f = PoissonLoss(zeros(T, 3), ones(T, 3, 1); weights = ws)
          for k in setdiff(1:3, pos)
            @test isfinite(wexp_ref(T, ws[k], η, 1024))
          end
          q = quadratic_model(f, T[0])
          before = deepcopy(q)
          y0, w0 = q.y, q.weights
          @test_throws PrecisionError quadratic_model(f, T[η])
          e1 = try quadratic_model(f, T[η]); nothing catch e e end
          e2 = try quadratic_model!(q, f, T[η]); nothing catch e e end
          for e in (e1, e2)
            @test e isa PrecisionError && occursin("observation $pos", sprint(showerror, e))
          end
          @test allfields_equal(q, before)
          @test q.y === y0 && q.weights === w0
          β = T[0.3]
          @test quadratic_model!(q, f, β) === q
          q2 = quadratic_model(f, β)
          @test isequal(q.y, q2.y) && isequal(q.weights, q2.weights) && q.y === y0 && q.weights === w0
          @test all(isapprox.(q.weights, ws .* exp(T(0.3)); rtol = 8 * eps(T)))
        end
      end
    end

    @testset "boundary sweep, $T" for T in (Float32, Float64)
      fm = floatmax(T)
      ηexp = Float64(log(fm))                           # exp(η) overflows just above
      bound = Float64(PB._exp_bound(T))
      rng = MersenneTwister(317_544 + sizeof(T))
      regions = ((:ordinary, 3000, () -> T(ηexp * (1 - 1e-4) * rand(rng))),
                 (:above_exp, 2500, () -> T(ηexp + 20 * rand(rng))),
                 (:large, 2500, () -> T(min(bound, ηexp + 20 + (bound - ηexp - 20) * rand(rng)))),
                 (:tiny, 3000, () -> T((rand(rng, Bool) ? 1 : -1) * 2.0^(-20 - 40 * rand(rng)))))
      st = Dict{Symbol, Int}()
      bump(k) = (st[k] = get(st, k, 0) + 1; nothing)
      failures = Any[]                                   # the first few, for the message
      near = Any[]
      f = one_obs(T, one(T))
      q = quadratic_model(f, T[0])
      t0 = time()
      for (region, neta, gen) in regions, _ in 1:neta
        η = gen()
        # the weights around M/exp(η) and floatmax/exp(η), and some clearly below or beyond, with the exact class
        ws, classes, dists, r1 = bigprec(1024) do
          e = exp(BigFloat(η)); Mb = threshold(T, 1024); fmb = BigFloat(fm)
          set = T[]
          for b in (T(Mb / e), T(fmb / e))
            (isfinite(b) && b > 0) || continue
            down = up = b
            push!(set, b)
            for _ in 1:2
              down = prevfloat(down); down > 0 && push!(set, down)
              up = nextfloat(up); isfinite(up) && push!(set, up)
            end
          end
          for x in (Mb / e * 1.0005, Mb / e * 1.01, fmb / e * 0.9995, fmb / e * 0.9985, fmb / e * 0.99)
            b = T(x); (isfinite(b) && b > 0) && push!(set, b)
          end
          unique!(set)
          Ps = [BigFloat(w) * e for w in set]
          cls = map(Ps) do P
            P < 0.999 * fmb && return :clearly_below
            P <= fmb && return :below_floatmax
            P < Mb && return :between_floatmax_and_M
            P < Mb * (1 + big(2.0)^-24) && return :just_above_M
            :beyond_M
          end
          set, cls, [Float64(abs(P / Mb - 1)) for P in Ps], [T(P) for P in Ps]
        end
        isempty(ws) && continue
        r2 = bigprec(2048) do; e = exp(BigFloat(η)); [T(BigFloat(w) * e) for w in ws]; end
        bump(:eta); bump(region)
        β = T[η]
        for (i, w) in enumerate(ws)
          bump(:cases)
          isequal(r1[i], r2[i]) || (bump(:reference_disagreements); continue)
          ref, cl = r1[i], classes[i]
          bump(cl); push!(near, (η, w, dists[i]))
          f.weights[1] = w
          got = outcome(f, T, η)
          if (got isa T && isfinite(got)) != isfinite(ref) || (!isfinite(ref) && !(got isa PrecisionError))
            bump(:classification_mismatches)
            length(failures) < 5 && push!(failures, (T, η, w, ref, got))
          elseif isfinite(ref)
            # below 0.999 floatmax (with a margin of a few ulps) the first tier is accurate to a few ulps; above it, correctly rounded
            if cl === :clearly_below && ref < 0.999 * fm * (1 + 64 * eps(T))
              isapprox(got, ref; rtol = 8 * eps(T)) || (bump(:accuracy_mismatches); length(failures) < 5 && push!(failures, (T, η, w, ref, got)))
            else
              got === ref || (bump(:value_mismatches); length(failures) < 5 && push!(failures, (T, η, w, ref, got)))
            end
          end
          if st[:cases] % 50 == 0                        # the same case through the in-place update
            bump(:update_cases)
            snap = deepcopy(q)
            got2 = try (quadratic_model!(q, f, β) === q ? only(q.weights) : :not_the_same_object) catch err; err end
            if got2 isa PrecisionError
              allfields_equal(q, snap) || bump(:update_atomicity_failures)
              got isa PrecisionError && occursin("observation 1", got.msg) && occursin("observation 1", got2.msg) || bump(:message_mismatches)
            end
            isequal(got, got2) || got isa PrecisionError && got2 isa PrecisionError || bump(:update_mismatches)
          end
        end
      end
      elapsed = time() - t0
      sort!(near; by = last)
      near_lists[T] = near
      @info "sweep $T: $(st[:eta]) η, $(st[:cases]) weights ($(sort!(collect(st); by = first))), $(round(elapsed; digits = 1)) s" failures
      @test get(st, :eta, 0) >= 10_000
      for k in (:ordinary, :above_exp, :large, :tiny)
        @test get(st, k, 0) >= 2000
      end
      for k in (:clearly_below, :below_floatmax, :between_floatmax_and_M, :just_above_M, :beyond_M)
        @test get(st, k, 0) >= 100
      end
      @test get(st, :classification_mismatches, 0) == 0
      @test get(st, :value_mismatches, 0) == 0
      @test get(st, :accuracy_mismatches, 0) == 0
      @test get(st, :reference_disagreements, 0) == 0
      @test get(st, :update_cases, 0) >= 1000
      @test get(st, :update_mismatches, 0) == 0
      @test get(st, :update_atomicity_failures, 0) == 0
      @test get(st, :message_mismatches, 0) == 0
    end

    @testset "the references are stable in the precision: the named cases and the sweep cases nearest to M" begin
      for (name, η, w, _) in named
        T = typeof(η)
        r = [wexp_ref(T, w, η, p) for p in (512, 1024, 2048)]
        @test isequal(r[1], r[2]) && isequal(r[2], r[3])
      end
      for T in (Float32, Float64)
        @test length(near_lists[T]) >= 10_000
        nearest = near_lists[T][1:200]
        @test last(nearest[end]) < 1e-3
        bad = count(nearest) do (η, w, _)
          r = [wexp_ref(T, w, η, p) for p in (512, 1024, 2048)]
          !(isequal(r[1], r[2]) && isequal(r[2], r[3]))
        end
        @test bad == 0
      end
    end

    @testset "the result does not depend on the ambient BigFloat precision" begin
      for p in (8, 64, 4096)
        setprecision(BigFloat, p) do
          for (name, η, w, fin) in named
            T = typeof(η)
            expected = fin ? wexp_ref(T, w, η, 1024) : PrecisionError
            f = one_obs(T, w)
            got = outcome(f, T, η)
            @test fin ? got === expected : got isa PrecisionError
            @test precision(BigFloat) == p
            q = quadratic_model(f, T[0])
            got2 = try quadratic_model!(q, f, T[η]); only(q.weights) catch err; err end
            @test fin ? got2 === expected : got2 isa PrecisionError
            @test precision(BigFloat) == p
          end
        end
      end
    end

    @testset "concurrent tasks: certified results with an unusual ambient precision seen by every task" begin
      # the cases A-D and the nearest-to-M cases of the sweep, with their sequential results
      cases = Tuple{DataType, Any, Any}[(typeof(η), η, w) for (_, η, w, _) in named[1:4]]
      for T in (Float64, Float32), (η, w, _) in near_lists[T][1:12]
        push!(cases, (T, η, w))
      end
      expected = [outcome(one_obs(T, w), T, η) for (T, η, w) in cases]
      @test count(x -> x isa PrecisionError, expected) >= 4 && count(x -> x isa PrecisionError, expected) < length(expected)
      # the sequential results are the correctly classified, correctly rounded values
      for (i, (T, η, w)) in enumerate(cases)
        r = wexp_ref(T, w, η, 1024)
        @test isfinite(r) ? expected[i] === r : expected[i] isa PrecisionError
      end
      prior = precision(BigFloat)
      third77 = bigprec(77) do; BigFloat(1) / 3; end
      third53 = bigprec(53) do; BigFloat(1) / 3; end
      @test third77 != third53                                             # the observer can tell 77 bits from 53
      wrong, bad_prec, bad_third, seen, done = (Threads.Atomic{Int}(0) for _ in 1:5)
      try
        setprecision(BigFloat, 77)
        observers = [Threads.@spawn begin
          while done[] == 0
            Threads.atomic_add!(seen, 1)
            precision(BigFloat(1)) == 77 || Threads.atomic_add!(bad_prec, 1)
            BigFloat(1) / 3 == third77 || Threads.atomic_add!(bad_third, 1)
            yield()
          end
        end for _ in 1:2]
        workers = [Threads.@spawn begin
          for _ in 1:10, (i, (T, η, w)) in enumerate(cases)
            f = one_obs(T, w)                              # each task has its own loss and model
            isequal(outcome(f, T, η), expected[i]) || Threads.atomic_add!(wrong, 1)
            q = quadratic_model(f, T[0])
            got = try quadratic_model!(q, f, T[η]); only(q.weights) catch err; err end
            isequal(got, expected[i]) || Threads.atomic_add!(wrong, 1)
            yield()
          end
        end for _ in 1:8]
        foreach(wait, workers)
        Threads.atomic_xchg!(done, 1)
        foreach(wait, observers)
      finally
        setprecision(BigFloat, prior)
      end
      @test precision(BigFloat) == prior
      @test seen[] > 0 && wrong[] == 0 && bad_prec[] == 0 && bad_third[] == 0
    end

    @testset "audit: the subnormal coefficient, explicit cases below, at and above floatmin, the smallest subnormal, underflow to zero" begin
      # Near floatmin the contract is the single rounding of the product of the weight and the reduced exponential
      # m 2^j (the library's own exponential, a few ulps from exp(η)): the references round w m 2^j once, from BigFloat.
      # Float32 loss with a Float64 response beyond floatmax(Float32): e^η > 1.1 recovers the response and the coefficient.
      T = Float32
      recovered(w) = GLMLoss(ProximalBase.Poisson(), [3.7e38], ones(T, 1, 1), T[w], nothing, zeros(T, 1), zeros(T, 1))
      seen = Dict{Symbol, Int}()
      bad = 0
      for η in T.(range(0.2, 6; length = 60))
        m, j = PB._exp_split(η)
        P1(w) = bigprec(512) do; ldexp(BigFloat(w) * BigFloat(m), j); end
        w0 = bigprec(512) do; T(BigFloat(floatmin(T)) / (BigFloat(m) * big(2.0)^j)); end
        ws = T[nextfloat(zero(T)), T(2) * nextfloat(zero(T)), T(3) * nextfloat(zero(T))]
        for k in -3:3
          w = w0; for _ in 1:abs(k); w = k < 0 ? prevfloat(w) : nextfloat(w); end
          push!(ws, w)
        end
        for w in ws
          iszero(w) && continue
          P = P1(w)
          ref = bigprec(512) do; T(P); end
          cl = ref == nextfloat(zero(T)) ? :smallest_subnormal : ref == floatmin(T) ? (P < floatmin(T) ? :rounds_up_to_floatmin : :at_floatmin) :
               ref < floatmin(T) ? :subnormal : :above_floatmin
          seen[cl] = get(seen, cl, 0) + 1
          got = outcome(recovered(w), T, η)
          bad += !(got === ref)
        end
      end
      @test bad == 0
      for k in (:smallest_subnormal, :rounds_up_to_floatmin, :at_floatmin, :subnormal, :above_floatmin)
        @test get(seen, k, 0) >= 1
      end
      # the ordinary path, both types: the product w exp(η) is rounded once, and a product that rounds to zero is a PrecisionError
      for T in (Float32, Float64), η in T[-0.5, -1, -3, -6], w in (nextfloat(zero(T)), T(3) * nextfloat(zero(T)), floatmin(T) / 4, floatmin(T))
        ref = bigprec(512) do; T(BigFloat(w) * BigFloat(exp(η))); end
        got = outcome(one_obs(T, w), T, η)
        @test iszero(ref) ? got isa PrecisionError : got === ref
      end
      @test iszero(wexp_ref(Float32, nextfloat(0f0), -1f0, 512)) && iszero(wexp_ref(Float64, nextfloat(0.0), -1.0, 512))
      @test wexp_ref(Float32, nextfloat(0f0), -0.5f0, 512) === nextfloat(0f0)
    end
  end

  @testset "the floor response: mixed precision at the overflow midpoint, no NaN, the floatmin boundary of the coefficient (review of #544)" begin
    ref_sum(T, η, e, y, h) = setprecision(BigFloat, 512) do
      T(big(η) - (big(e) - big(y)) / big(h))
    end
    # the reports: a Float64 response with Float32 scratch whose exact response lies 2^-55 below the overflow midpoint
    let η = -7.440598f35, y = 1.1774515996114581e35
      f = GLMLoss(ProximalBase.Poisson(), [y], ones(Float32, 1, 1), nothing, nothing, zeros(Float32, 1), zeros(Float32, 1))
      @test only(quadratic_model(f, Float32[η]).y) == floatmax(Float32)
      q = quadratic_model(f, Float32[0]); @test only(quadratic_model!(q, f, Float32[η]).y) == floatmax(Float32)
    end
    # sweeps of the boundary and of the overflow midpoint floatmax(1 + 2^-p), same type and mixed
    for (T, Y) in ((Float32, Float32), (Float32, Float64), (Float64, Float64))
      h = sqrt(eps(T))
      rng = MersenneTwister(5441 + sizeof(T) + 7 * sizeof(Y))
      mism = 0
      for k in 1:4000
        η = -floatmax(T) * (T(0.02) + T(0.9) * rand(rng, T))
        centre = isodd(k) ? big(floatmax(T)) : big(floatmax(T)) * (1 + big(2.0)^-precision(T))   # boundary, midpoint
        width = k % 4 < 2 ? 8 * big(eps(T)) : 1e-14 * big(1.0)                                    # near, and very near
        tgt = centre * (1 + (rand(rng) - 0.5) * width)
        y = Y((tgt - big(η)) * big(h))
        f = GLMLoss(ProximalBase.Poisson(), [y], ones(T, 1, 1), nothing, nothing, zeros(T, 1), zeros(T, 1))
        ref = ref_sum(T, η, exp(big(η)), y, h)
        got = try only(quadratic_model(f, T[η]).y) catch err; err end
        mism += !(isfinite(ref) ? (got isa T && got == ref) : got isa PrecisionError)
      end
      @test mism == 0
    end
    # no NaN from finite input, any sign, magnitude and precision; Inf exactly where the sum overflows
    for T in (Float32, Float64, Float16)
      h = sqrt(eps(T)); fm = floatmax(T)
      for η in (-fm, -fm / 2, T(-3), zero(T), T(1), fm / 2, fm), c in (0, 0.25, 1, 1.5, 2.5, 3, 5), sg in (1, -1), e in (zero(T), h / 2, h)
        y = T(sg) * T(c) * (fm * h)
        z = ProximalBase._floored_response(η, e, y, h)
        @test !isnan(z)
        if T != Float16 && isfinite(y)
          ref = ref_sum(T, η, e, y, h)
          @test isfinite(z) == isfinite(ref)
          isfinite(ref) && @test abs(big(z) - big(ref)) <= 4 * eps(T) * big(fm)
        end
      end
    end
    # a response that is not a float is converted to the working float type first, as e - y is (numerics T3):
    # the model equals that of the converted response, for integers and rationals, in both API calls
    for (η, y0) in ((-1.8697808f38, Int128(182045609535131808700852906315186176)),
                    (-2.3926112f38, Int128(200097226578706225858804805414092800)), (-3.0f0, Int128(10)^30))
      for y in (y0, y0 // Int128(1), Int64(7))
        f = PoissonLoss([y], ones(Float32, 1, 1)); fc = PoissonLoss([Float32(y)], ones(Float32, 1, 1))
        get(f) = try only(quadratic_model(f, Float32[η]).y) catch err; nameof(typeof(err)) end
        @test isequal(get(f), get(fc))
        # the in-place update: the same value or the same error, and the model unchanged on an error
        upd(f) = (q = quadratic_model(f, Float32[0]); before = deepcopy(q);
                  r = try only(quadratic_model!(q, f, Float32[η]).y) catch err; nameof(typeof(err)) end;
                  r isa Symbol && @test all(isequal(getfield(q, k), getfield(before, k)) for k in fieldnames(typeof(q)));
                  r)
        @test isequal(upd(f), upd(fc)) && isequal(upd(f), get(f))
      end
    end
    # a BigFloat response is computed at the current BigFloat precision and delivered in the loss's type
    # The response is z = η + (y - e)/h at η = -floatmax/3 (so e = exp(η) = 0 and the curvature is the floor h = √eps),
    # with y = c floatmax h formed in BigFloat, where it is finite: y/h = c floatmax does not fit in T, but
    # z = (c - 1/3) floatmax does for c = 1.2 (z = 0.867 floatmax, a cancelling pair) and does not for c = 1.4
    # (z = 1.067 floatmax: PrecisionError). The reference is the definition at the current precision: the
    # BigFloat arithmetic of the expression at the precision `prec`, delivered in T (numerics T7). It equals the
    # 1024-bit value for these data (asserted, so that the reference does not depend on the rounding of the steps).
    for T in (Float32, Float64), prec in (64, 128, 256)
      setprecision(BigFloat, prec) do
        h = sqrt(eps(T)); M = floatmax(T); η = -M / 3; e = exp(η)
        for (c, fits) in ((1.2, true), (1.4, false))
          y = BigFloat(c) * BigFloat(M) * BigFloat(h)
          @test isfinite(y) && precision(y) == prec
          @test !isfinite(T(y / h))                                     # the term y/h alone does not fit
          ref = T(η + (y - e) / h)
          ref1024 = setprecision(BigFloat, 1024) do; T(η + (y - e) / h); end
          @test isequal(ref, ref1024) && isfinite(ref) == fits
          f = GLMLoss(ProximalBase.Poisson(), [y], ones(T, 1, 1), nothing, nothing, zeros(T, 1), zeros(T, 1))
          q = quadratic_model(f, T[0]); before = deepcopy(q)            # a valid model, at η = 0
          if fits
            m = quadratic_model(f, T[η])
            @test only(m.y) === ref && only(m.weights) === h            # the floor h: e = 0
            @test quadratic_model!(q, f, T[η]) === q
            @test only(q.y) === ref && only(q.weights) === h
          else
            @test_throws PrecisionError quadratic_model(f, T[η])
            @test_throws PrecisionError quadratic_model!(q, f, T[η])
            @test all(isequal(getfield(q, k), getfield(before, k)) for k in fieldnames(typeof(q)))
          end
          @test precision(BigFloat) == prec
        end
      end
    end
    # the coefficient at the floatmin boundary is rounded once as well (the reported case, and a grid)
    let η = 1.105095f0, w = 3.892995f-39, exp_split = ProximalBase._exp_split
      f = GLMLoss(ProximalBase.Poisson(), [1.1 * Float64(floatmax(Float32))], ones(Float32, 1, 1), Float32[w], nothing, zeros(Float32, 1), zeros(Float32, 1))
      m, j = exp_split(η)
      ref = setprecision(BigFloat, 512) do; Float32(ldexp(big(w) * big(m), j)); end
      @test only(quadratic_model(f, Float32[η]).weights) === ref
      rng = MersenneTwister(5442); bad = 0
      for _ in 1:3000
        ηT = Float32(0.2 + 5.8 * rand(rng))
        wT = Float32(floatmin(Float32) / exp(ηT) * (1 + (rand(rng) - 0.5) * 1e-5))
        fT = GLMLoss(ProximalBase.Poisson(), [3.7e38], ones(Float32, 1, 1), Float32[wT], nothing, zeros(Float32, 1), zeros(Float32, 1))
        mT, jT = exp_split(ηT)
        refT = setprecision(BigFloat, 512) do; Float32(ldexp(big(wT) * big(mT), jT)); end
        bad += only(quadratic_model(fT, Float32[ηT]).weights) !== refT
      end
      @test bad == 0
    end
  end

  @testset "non-finite data: the ordinary response, offsets of allocating models, allocation (review of #544)" begin
    # a non-finite weight alone leaves the response of the ordinary computation (bit for bit)
    for T in (Float32, Float64), bad in (T(NaN), T(Inf)), η in (T(1e-8), T(1), T(-3), T(30))
      f = PoissonLoss(T[1], ones(T, 1, 1); weights = T[1])
      f.weights[1] = bad
      q = quadratic_model(f, T[η])
      @test isequal(q.y, ordinary_model(f, T[η])[2])
      @test !isfinite(only(q.weights))
      @test isequal(quadratic_model!(q, f, T[η]).y, ordinary_model(f, T[η])[2])
    end
    # an overflowing predictor with a non-finite weight: no recovery, as before
    for T in (Float32, Float64), bad in (T(NaN), T(Inf))
      f = PoissonLoss(T[1], ones(T, 1, 1); weights = T[1])
      f.weights[1] = bad
      η = T === Float64 ? T(710) : T(89)
      @test !isfinite(only(quadratic_model(f, T[η]).y))
    end
    # an offset made non-finite after the construction: the update of a model propagates it,
    # the allocating call revalidates the offset (documented)
    for T in (Float32, Float64), bad in (T(NaN), T(Inf), -T(Inf))
      f = PoissonLoss(T[0], ones(T, 1, 1); offset = T[0])
      q = quadratic_model(f, T[0])
      f.offset[1] = bad
      @test quadratic_model!(q, f, T[0]) === q
      @test !isfinite(only(q.y)) || !isfinite(only(q.weights))
      @test_throws ArgumentError quadratic_model(f, T[0])
    end
    # allocation: none for finite data (also at a large predictor), and a successful update with a
    # non-finite observation checks the staged values in a new vector (documented)
    alloc(f, q, b) = (quadratic_model!(q, f, b); @allocated quadratic_model!(q, f, b))
    for n in (1, 10, 100)
      f = PoissonLoss(zeros(n), ones(n, 1))
      q = quadratic_model(f, [0.0])
      @test alloc(f, q, [0.0]) == 0
      fr = PoissonLoss(zeros(n), ones(n, 1); weights = fill(1 / 8, n))     # recovery at η = 710
      qr = quadratic_model(fr, [710.0])
      @test alloc(fr, qr, [710.0]) == 0 && only(unique(qr.y)) == 709.0
      f.y[1] = NaN
      @test quadratic_model!(q, f, [0.0]) === q
      @test alloc(f, q, [0.0]) > 0
    end
  end

  @testset "known limitation: the working response cannot carry l'/l'' at a predictor of 1e30" begin
    # at η = -1e30 the pair (v, z) = (w √eps, η + y/√eps) rounds η - z to 0: the model's gradient is 0, not w(exp(η) - y);
    # the same before this change (a separate numerical-contract issue)
    f = PoissonLoss([1.0], ones(1, 1))
    @test only(gradient(f, [-1e10])) == -1
    @test only(gradient(quadratic_model(f, [-1e10]), [-1e10])) ≈ -1
    @test_broken only(gradient(quadratic_model(f, [-1e30]), [-1e30])) ≈ -1
  end

  @testset "the arrays of the model are not the buffers of the loss" begin
    allfields(a, b) = all(isequal(getfield(a, k), getfield(b, k)) for k in fieldnames(typeof(a)))
    X = [1.0 0.0; 720.0 0.0; 0.0 1.0]               # the update at β = (1, 0.5) fails at observation 2
    for (yb, wb) in ((:η, :own), (:d, :own), (:own, :η), (:own, :d), (:η, :d), (:d, :η))
      f = PoissonLoss([0.0, 1.0, 2.0], X)
      pick(b) = b === :η ? f.η : b === :d ? f.d : zeros(3)
      y, w = pick(yb), pick(wb)
      fill!(y, -3.0); fill!(w, 4.0)                      # sentinels: the update would overwrite them
      q = GLMLoss(PB.Squared(), y, X; weights = w)
      @test q.y === y && q.weights === w
      before = deepcopy(q)
      e = try quadratic_model!(q, f, [1.0, 0.5]); nothing catch e e end
      @test e isa ArgumentError                          # refused up front, not a PrecisionError or a corrupted model
      @test allfields(q, before)
      @test q.y === y && q.weights === w
      @test all(==(-3.0), q.y) && all(==(4.0), q.weights)
    end
    # the two arrays of the model are one, or one of them is an array of the loss's data: refused up front,
    # and neither the loss nor the model changes
    f = PoissonLoss([0.0, 1.0, 2.0], X; weights = [1.0, 0.5, 2.0], offset = [0.0, 0.1, 0.2])
    shared = fill(2.0, 3)
    for (name, yq, wq) in (("q.y === q.weights", shared, shared), ("q.y === f.y", f.y, zeros(3)),
                           ("q.weights === f.y", zeros(3), f.y), ("q.y === f.weights", f.weights, zeros(3)),
                           ("q.weights === f.weights", zeros(3), f.weights), ("q.y === f.offset", f.offset, zeros(3)),
                           ("q.weights === f.offset", zeros(3), f.offset))
      q = GLMLoss(PB.Squared(), yq, X; weights = wq, offset = f.offset)
      @test q.X === f.X && q.offset === f.offset
      qbefore = deepcopy(q)
      fdata = (copy(f.y), copy(f.weights), copy(f.offset))
      e = try quadratic_model!(q, f, [0.4, 0.3]); nothing catch e e end
      @test e isa ArgumentError
      @test allfields(q, qbefore)
      @test (f.y, f.weights, f.offset) == fdata
    end
    # a model with its own arrays is unaffected
    f = PoissonLoss([0.0, 1.0, 2.0], X)
    q = quadratic_model(f, [0.0, 0.0])
    @test quadratic_model!(q, f, [0.1, 0.2]) === q
  end

  @testset "the scratch buffers η and d of a loss are two arrays, and not its data" begin
    # With η === d the staged pair (v, z) = (1, -1) of a Poisson observation at η = 0, y = 0
    # was written into one array and the model got the weight -1, which passed the check.
    X = ones(1, 1)
    mk(y, w, o, η, d) = GLMLoss(PB.Poisson(), y, X, w, o, η, d)   # the positional constructor
    buf = zeros(1)
    @test_throws ArgumentError mk([0.0], [1.0], nothing, buf, buf)
    @test_throws ArgumentError mk([0.0], [1.0], [0.0], buf, buf)
    for (name, which) in (("y", 1), ("weights", 2), ("offset", 3))
      for (buffer, other) in (("η", 1), ("d", 2))
        data = (zeros(1), ones(1), zeros(1))
        sc = data[which]
        e = try
          mk(data[1], data[2], data[3], other == 1 ? sc : zeros(1), other == 2 ? sc : zeros(1))
          nothing
        catch e
          e
        end
        @test e isa ArgumentError
      end
    end
    # separate arrays are accepted; so are equal ones, and data shared between y, weights and offset,
    # which the update only reads
    f = mk([0.0], [1.0], [0.0], zeros(1), zeros(1))
    @test f.η !== f.d
    shared = ones(1)
    @test mk(shared, shared, shared, zeros(1), zeros(1)) isa GLMLoss
    @test mk([0.0], [1.0], nothing, zeros(1), zeros(1)) isa GLMLoss
    # the model of the Poisson observation at η = 0, y = 0, w = 1 is (v, z) = (1, -1)
    q = quadratic_model(f, [0.0])
    @test q.weights == [1.0] && q.y == [-1.0]
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
