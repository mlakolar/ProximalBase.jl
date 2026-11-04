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

end
