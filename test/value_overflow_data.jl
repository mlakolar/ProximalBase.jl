module ValueOverflowDataTest

# Values of the functions that hold data when a matrix product, a quadratic form
# or a dot leaves the floating-point range at a representable value (R1, #358).

using Test
using ProximalBase
using LinearAlgebra
using Random
using SparseArrays
using JLArrays

const PB = ProximalBase

# v within k units in the last place of T at the exact value r.
function within(v::T, r, k = 2) where {T<:AbstractFloat}
  r = big(r)
  iszero(r) && return iszero(v)
  isfinite(v) && abs(big(v) - r) <= k * ldexp(big(1.0), exponent(r) - precision(T) + 1)
end
# v within k eps(T) of the scale s of the computation, which accounts for cancellation.
near_scale(v::T, r, s, k = 8) where {T<:AbstractFloat} = isfinite(v) && abs(big(v) - big(r)) <= k * big(eps(T)) * big(s)

# the base's ordinary least-squares loop, on a given product tmp = Xβ
function plain_leastsquares(Y, X, β, tmp = X * β)
  v = zero(eltype(tmp))
  @inbounds @simd for i in eachindex(tmp, Y)
    v += abs2(Y[i] - tmp[i])
  end
  v / 2
end

# in a function, so that the value returned from a global is not boxed (Julia 1.10)
allocated_value(f, x) = (f(x); @allocated f(x))
pow2_three(::Type{T}, E, a, b, c) where {T} = PB._pow2_weighted(T, one(T), (a, 2E), (b, 2E), (c, 0))
allocated_pow2(::Type{T}, E, a, b, c) where {T} = (pow2_three(T, E, a, b, c); @allocated pow2_three(T, E, a, b, c))

@testset "data function values that leave the range" begin
  for T in (Float32, Float64)
    M = floatmax(T)
    E = exponent(M)
    root = T(1.2) * sqrt(M)                          # its square overflows, half of it does not

    @testset "LeastSquaresLoss $T" begin
      X = reshape(T[M, -M], 1, 2)
      f = LeastSquaresLoss(T[0], X)
      @test f(T[2, 2]) === zero(T)                   # B01: X β = 2M - 2M
      g = zeros(T, 2)
      @test value_and_gradient!(g, f, T[2, 2]) === zero(T)
      @test isequal(g, gradient!(zeros(T, 2), f, T[2, 2]))
      # a residual of 2 under products that overflow
      X3 = reshape(T[M, -M, 1], 1, 3)
      f3 = LeastSquaresLoss(T[1], X3)
      for β in (T[2, 2, 3], SparseIterate(T[2, 2, 3]), AtomIterate(T[2, 2, 3], [1:3]))
        @test within(f3(β), 2)
      end
      @test within(value_and_gradient!(zeros(T, 3), f3, T[2, 2, 3]), 2)
      # multiple responses
      Y2 = T[1 0]
      β2 = T[2 1; 2 1; 3 0]
      @test within(LeastSquaresLoss(Y2, X3)(β2), 2)
      @test within(value_and_gradient!(zeros(T, 3, 2), LeastSquaresLoss(Y2, X3), β2), 2)
      # finite residual whose square overflows
      @test within(LeastSquaresLoss(T[0], ones(T, 1, 1))([root]), big(root)^2 / 2)
      @test within(value_and_gradient!(zeros(T, 1), LeastSquaresLoss(T[0], ones(T, 1, 1)), [root]), big(root)^2 / 2)
      # a small response survives a product that cancels in the recovery
      P = ldexp(one(T), E - 23)
      XP = reshape(T[P, -P], 1, 2)
      for y in (ldexp(one(T), -40), T(3) * sqrt(floatmin(T)), one(T))     # y²/2 is normal
        @test within(LeastSquaresLoss(T[y], XP)(T[P, P]), big(y)^2 / 2)
        @test within(value_and_gradient!(zeros(T, 2), LeastSquaresLoss(T[y], XP), T[P, P]), big(y)^2 / 2)
        @test within(LeastSquaresLoss(JLArray(T[y]), JLArray(XP))(JLArray(T[P, P])), big(y)^2 / 2)
      end
      # one entry cancels to a small residual, another stays large
      XB = T[P -P 0; 0 0 1]
      y = ldexp(one(T), -40)
      @test within(LeastSquaresLoss(T[y, 0], XB)(T[P, P, root / 2]), (big(y)^2 + big(root / 2)^2) / 2)
      # exact fit, non-finite inputs and the device path
      @test LeastSquaresLoss(T[1, 2], T[1 0; 0 1])(T[1, 2]) === zero(T)
      @test isnan(f(T[NaN, 2]))
      @test isequal(f(T[Inf, 2]), plain_leastsquares(T[0], X, T[Inf, 2]))
      @test isequal(LeastSquaresLoss(T[Inf], X3)(T[2, 2, 3]), plain_leastsquares(T[Inf], X3, T[2, 2, 3]))
      # JLArrays disallow scalar indexing outside interactive sessions
      fd = LeastSquaresLoss(JLArray(T[1]), JLArray(X3))
      @test within(fd(JLArray(T[2, 2, 3])), 2)
      @test within(value_and_gradient!(JLArray(zeros(T, 3)), fd, JLArray(T[2, 2, 3])), 2)
      @test within(LeastSquaresLoss(JLArray(T[0]), JLArray(ones(T, 1, 1)))(JLArray([root])), big(root)^2 / 2)
      @test LeastSquaresLoss(JLArray(T[1]), JLArray(ones(T, 1, 1)))(JLArray(T[1])) === zero(T)
      # ordinary inputs keep the ordinary computation
      for (n, p) in ((5, 3), (40, 17))
        Xr, Yr, βr = T.(sin.(1:n) .* cos.((1:p)')), T.(cos.(1:n)), T.(sin.(1:p) ./ 3)
        fr = LeastSquaresLoss(Yr, Xr)
        v = fr(βr)
        # on the product the call stored: BLAS may round X * β and mul! differently
        @test v === plain_leastsquares(Yr, Xr, βr, fr.tmp)
      end
    end

    @testset "QuadraticFunction $T" begin
      A = ones(T, 1, 1)
      f = QuadraticFunction(A)
      r = big(root)^2 / 2
      @test within(f([root]), r)                     # B02
      g = zeros(T, 1)
      @test within(value_and_gradient!(g, f, [root]), r)
      @test isequal(g, gradient!(zeros(T, 1), f, [root]))
      # the quadratic and the linear term both overflow and cancel
      x = ldexp(one(T), E ÷ 2 + 1)
      fl = QuadraticFunction(A, [-x / 2], one(T))
      @test fl([x]) === one(T)
      @test value_and_gradient!(zeros(T, 1), fl, [x]) === one(T)
      # a constant far below the exponent of the cancelling terms (review of f302de59)
      xc = ldexp(one(T), T === Float32 ? 100 : 600)
      for cc in (one(T), T(-0.75), floatmin(T))
        fc = QuadraticFunction(A, [-xc / 2], cc)
        @test fc([xc]) === cc
        gc = zeros(T, 1)
        @test value_and_gradient!(gc, fc, [xc]) === cc
        @test isequal(gc, gradient!(zeros(T, 1), fc, [xc]))
        @test fc(SparseIterate([xc])) === cc
      end
      # a Float32 function at a Float64 argument: A x overflows the Float32 buffer, the value is Float64
      T === Float32 && @test QuadraticFunction(fill(0.25f0, 1, 1), [-2f0^127], 1f0)([2.0^130]) === 1.0
      # a large matrix, a large linear term and a 2 × 2 form
      @test within(QuadraticFunction(fill(M, 1, 1))(T[2^-10]), big(M) * big(2.0)^-20 / 2)
      A2 = T[2 1; 1 2]
      b2 = T[1, -1] .* (root / 8)
      x2 = T(0.6) .* [root, -root]                   # x'Ax overflows, x'Ax/2 does not
      ref = (big.(x2)' * big.(A2) * big.(x2)) / 2 + big.(b2)' * big.(x2)
      scale = (abs.(big.(x2))' * big.(A2) * abs.(big.(x2))) / 2 + abs.(big.(b2))' * abs.(big.(x2))
      @test near_scale(QuadraticFunction(A2, b2)(x2), ref, scale)
      @test near_scale(QuadraticFunction(A2, b2)(SparseIterate(x2)), ref, scale)
      # a Float32 function at a Float64 argument: the products are stored in Float32
      T === Float32 && @test isapprox(QuadraticFunction(fill(3f38, 2, 2))([1e10, 1e10]), 4 * big(3f38) * big(1e10)^2 / 2;
                                      rtol = 4eps(Float32))
      # the recovery allocates nothing
      @test allocated_value(f, [root]) == 0
      # non-finite inputs and ordinary inputs
      @test isnan(f(T[NaN]))
      @test f(T[Inf]) === T(Inf)
      Ar = T[3 1 0; 1 3 1; 0 1 3]
      br, xo = T[1, -2, 0.5], T[0.3, -1.1, 2]
      @test QuadraticFunction(Ar, br, T(0.25))(xo) === dot(xo, Ar * xo) / 2 + T(0.25) + dot(br, xo)
    end

    @testset "KronQuadratic $T" begin
      f = KronQuadratic(ones(T, 1, 1), ones(T, 1, 1))
      r = big(root)^2 / 2
      @test within(f(fill(root, 1, 1)), r)          # B03
      G = zeros(T, 1, 1)
      @test within(value_and_gradient!(G, f, fill(root, 1, 1)), r)
      @test isequal(G, gradient!(zeros(T, 1, 1), f, fill(root, 1, 1)))
      # large A and B: D must not be pushed into the subnormal range
      t = ldexp(one(T), -(2E ÷ 3))
      @test within(KronQuadratic(fill(M, 1, 1), fill(M, 1, 1))(fill(t, 1, 1)), big(M)^2 * big(t)^2 / 2)
      # the quadratic and the linear term cancel
      x = ldexp(one(T), E ÷ 2 + 1)
      @test KronQuadratic(ones(T, 1, 1), ones(T, 1, 1), fill(-x / 2, 1, 1))(fill(x, 1, 1)) === zero(T)
      # a symmetric iterate and a 2 × 3 matrix
      θ = T(0.6) * root
      Θs = SymmetricSparseIterate(T[θ 0; 0 θ])
      @test within(KronQuadratic(Matrix{T}(I, 2, 2), Matrix{T}(I, 2, 2))(Θs), big(θ)^2)
      A, B = T[2 1; 1 2], T[1 0 0; 0 2 0; 0 0 1]
      Θ = T[root / 2 0 1; 0 root / 4 0]
      bigΘ = big.(Θ)
      ref = dot(bigΘ, big.(A) * bigΘ * big.(B)) / 2
      scale = dot(abs.(bigΘ), big.(A) * abs.(bigΘ) * big.(B)) / 2
      fk = KronQuadratic(A, B)
      @test near_scale(fk(Θ), ref, scale)
      @test allocated_value(fk, Θ) == 0
      @test isnan(fk(T[NaN 0 0; 0 0 0]))
      Θo = T[0.5 -1 2; 1 0.25 -0.5]
      D = copy(Θo)
      @test fk(Θo) === dot(D, (A * D) * B) / 2
    end

    @testset "LinearFunction $T" begin
      @test LinearFunction(T[2], -M)(T[M]) === M     # B04
      @test value_and_gradient!(zeros(T, 1), LinearFunction(T[2], -M), T[M]) === M
      @test LinearFunction(T[2, -2])(T[M, M]) === zero(T)
      # a moderate result under products that overflow, in the scale of the computation
      c, x, d = T[2, -2, 3], T[M, M, 1], one(T)
      @test near_scale(LinearFunction(c, d)(x), 4, 4big(M))
      # a small d survives a dot that cancels
      P = ldexp(one(T), E - 23)
      for d in (T(1e-20), floatmin(T) * T(2)^40, -one(T))
        @test LinearFunction(T[P, P], d)(T[P, -P]) === d
      end
      # argument and coefficient types
      @test LinearFunction(T[2, 0], -M)(AtomIterate(T[M, 0], [1:2])) === M
      @test LinearFunction(T[2, 0], -M)(SparseIterate(T[M, 0])) === M
      @test LinearFunction(T[2 0; 0 0], -M)(SymmetricSparseIterate(T[M 0; 0 0])) === M
      @test LinearFunction(T[0 2; 2 0])(SymmetricSparseIterate(T[0 M; M 0])) === T(Inf)   # 4M overflows
      @test LinearFunction(T[0 1; 1 0], -M)(SymmetricSparseIterate(T[0 M; M 0])) === M
      # an off-diagonal coefficient at floatmax: its doubled weight overflows on its own (review of f302de59)
      Cm = T[0 M; M 0]
      Xh = SymmetricSparseIterate(T[0 0.5; 0.5 0])
      @test LinearFunction(Cm)(Xh) === M
      @test LinearFunction(Cm, -M / 2)(Xh) === M / 2
      @test LinearFunction(T[M 0; 0 M], -M)(SymmetricSparseIterate(T[1 0; 0 1])) === M   # diagonal entries count once
      @test LinearFunction(Cm)(SymmetricSparseIterate(T[0 1; 1 0])) === T(Inf)            # 2M is out of range
      @test value_and_gradient!(zeros(T, 2, 2), LinearFunction(Cm), Xh) === M
      @test allocated_value(LinearFunction(Cm), Xh) == 0
      T === Float64 && @test LinearFunction(Float32[0 floatmax(Float32); floatmax(Float32) 0])(
        SymmetricSparseIterate([0 0.5; 0.5 0])) === Float64(floatmax(Float32))
      @test isnan(LinearFunction(Cm)(SymmetricSparseIterate(T[0 NaN; NaN 0])))
      @test LinearFunction(sparse(T[2, 0]), -M)(T[M, 1]) === M
      @test LinearFunction(Diagonal(T[2, 1]), -M)(Diagonal(T[M, 0])) === M
      @test LinearFunction(T[2, 0], -M)(sparse(T[M, 0])) === M
      # wrapped sparse coefficients: the recovery reads the stored entries of the parent
      @test LinearFunction(sparse(T[2 0; 1 0])', -M)(T[M 0; 0 0]) === M
      @test LinearFunction(transpose(sparse(T[2 0; 1 0])), -M)(T[M 0; 0 0]) === M
      @test isnan(LinearFunction(sparse(T[2 0; NaN 0])', -M)(T[M 0; 0 0]))
      @test LinearFunction(JLArray(T[2, -2]))(JLArray(T[M, M])) === zero(T)
      # mixed precision: Float64 coefficients with a Float32 argument give a Float32 value (T1)
      T === Float32 && @test LinearFunction([2.0], -Float64(M))(T[M]) === M
      T === Float64 && @test LinearFunction(Float32[2], -M)([M]) === M
      # non-finite inputs and ordinary inputs
      @test isnan(LinearFunction(T[2], -M)(T[NaN]))
      @test LinearFunction(T[2], zero(T))(T[Inf]) === T(Inf)
      co, xo = T[1, -2, 0.5], T[0.3, -1.1, 2]
      @test LinearFunction(co, T(0.25))(xo) === dot(co, xo) + T(0.25)
    end
  end

  @testset "LogDetLoss: trace products that underflow before the weight" begin
    # B16: dot(S, X) = 2^-1200 underflows; the weight 2^1000 lifts it to 2^-200
    f = LogDetLoss(Diagonal([2.0^-600, 0.0]), 2.0^1000)
    X = Diagonal([2.0^-600, 2.0^600])
    @test f(X) === 2.0^-200
    @test value_and_gradient!(zeros(2, 2), f, Matrix(X)) === 2.0^-200
    # a subnormal dot is finite, normal after the weight, but has lost bits
    a = 1.1 * 2.0^-500
    f2 = LogDetLoss(Diagonal([a, 0.0]), 2.0^1000)
    @test within(f2(Diagonal([2.0^-570, 2.0^570])), big(a) * big(2.0)^-570 * big(2.0)^1000)
    # Float32: dot 2^-152, weight 2^100
    f32 = LogDetLoss(Diagonal(Float32[2.0^-76, 0]), 2f0^100)
    ld = 2 * (log(2f0^-38) + log(2f0^38))
    @test within(f32(Diagonal(Float32[2.0^-76, 2.0^76])), 2f0^100 * (big(2.0)^-152 - big(ld)))
    # a weight above one with an ordinary difference keeps the ordinary value
    S, Xo = [2.0 0.5; 0.5 1.0], [1.5 0.25; 0.25 2.0]
    w = 3.0
    @test LogDetLoss(S, w)(Xo) ≈ w * (dot(S, Xo) - logdet(Xo)) rtol = 1e-14
    # the third part of the split dot: products below floatmin·2^(p+1)
    big_, small, tiny = PB._split_dot([2.0^-600, 3.0], [2.0^-600, 1.0], (1.0, 1.0), (1.0, 1.0))
    @test (big_, small) == (0.0, 3.0)
    @test tiny == ldexp(1.0, -1200 + PB._tiny_unit(Float64))
    @test PB._split_dot([0.0, 2.0^-600], [5.0, 0.0], (1.0, 1.0), (1.0, 1.0)) == (0.0, 0.0, 0.0)
  end

  @testset "_pow2_sum: large terms cancel before smaller ones are added" begin
    for T in (Float32, Float64)
      E = T === Float32 ? 100 : 600
      # x²/2 in units of 2^(2k), -x²/2 in units of 2^k, and a constant in units of 1
      @test PB._pow2_sum(T, (T(0.5), 2E), (T(-0.5), 2E), (one(T), 0)) === one(T)
      @test PB._pow2_sum(T, (one(T), 0), (T(0.5), 2E), (T(-0.5), 2E)) === one(T)      # any order
      @test PB._pow2_sum(T, (T(0.5), E), (T(-1), E - 1), (T(3), -E)) === ldexp(T(3), -E)
      # partial cancellation leaves a term in the middle, then the small one
      @test PB._pow2_sum(T, (T(1.5), 2E), (T(-1), 2E), (T(-0.5), 2E), (T(0.25), 0), (T(0.5), 1)) === T(1.25)
      @test PB._pow2_weighted(T, T(2)^E, (one(T), E), (-one(T), E), (one(T), -E)) === one(T)  # weight last
      @test PB._pow2_weighted(T, T(3), (T(0.5), 2E), (T(-0.5), 2E)) === zero(T)
      @test PB._pow2_sum(T, (T(2), E), (one(T), -E)) === ldexp(T(2), E) + ldexp(one(T), -E) # ordinary rounding
      # non-finite terms give the ordinary sum
      @test isnan(PB._pow2_sum(T, (T(NaN), 0), (one(T), E)))
      @test PB._pow2_sum(T, (T(Inf), 0), (one(T), E)) === T(Inf)
      @test isnan(PB._pow2_sum(T, (T(Inf), 0), (-T(Inf), 0)))
      @test PB._pow2_sum(T) === zero(T)
      # mixed term types are converted, and nothing is allocated
      @test PB._pow2_sum(T, (0.5f0, 2E), (-0.5, 2E), (1, 0)) === one(T)
      @test allocated_pow2(T, E, T(0.5), T(-0.5), one(T)) == 0
    end
  end
  # Underflow before a large argument entry multiplies the intermediate (R1, #520).
  # Accuracy classes used below: "ulps" is `within` (relative, a few units in the last place,
  # for sums without cancellation); "scale" is `near_scale` (error against the cancellation-aware
  # scale of the computation); "exact" is `===`.
  @testset "values whose intermediates underflow (#520)" begin
    big_q(A, x) = dot(big.(x), big.(A) * big.(x)) / 2
    big_k(A, B, Θ) = dot(big.(Θ), big.(A) * big.(Θ) * big.(B)) / 2
    # the scale of the same computation with absolute values
    abs_q(A, x) = dot(abs.(big.(x)), abs.(big.(A)) * abs.(big.(x))) / 2
    abs_k(A, B, Θ) = dot(abs.(big.(Θ)), abs.(big.(A)) * abs.(big.(Θ)) * abs.(big.(B))) / 2

    for (T, k) in ((Float32, 16), (Float64, 28))
      a = nextfloat(zero(T))
      x = T[ldexp(one(T), k) + T(0.25)]
      A = reshape(T[a], 1, 1)
      f = QuadraticFunction(A)
      ref = big_q(A, x)

      @testset "QuadraticFunction $T: A x is subnormal" begin
        @test within(f(x), ref)                                          # ulps
        g = zeros(T, 1)
        @test within(value_and_gradient!(g, f, x), ref)
        @test isequal(g, gradient!(zeros(T, 1), f, x))                   # the gradient keeps its semantics
        @test within(f(-x), ref)                                         # signs
        @test within(f(SparseIterate(x)), ref)                           # iterate representations
        @test within(f(view([zero(T); x], 2:2)), ref)
        @test within(value_and_gradient!(zeros(T, 1), f, SparseIterate(x)), ref)
        xc = copy(x)
        gc = zeros(T, 1)
        @test_throws ArgumentError value_and_gradient!(xc, f, xc)       # aliasing is still refused
        @test xc == x && A == reshape(T[a], 1, 1)                        # no input is modified
        @test allocated_value(f, x) == 0
      end

      @testset "KronQuadratic $T: stage-one product subnormal, amplified by B" begin
        kb = T === Float32 ? 60 : 500
        Ak, Bk = reshape(T[a], 1, 1), reshape(T[ldexp(one(T), kb)], 1, 1)
        Θ = reshape(x, 1, 1)
        fk = KronQuadratic(Ak, Bk)
        r = big_k(Ak, Bk, Θ)
        @test within(fk(Θ), r)
        @test within(value_and_gradient!(zeros(T, 1, 1), fk, Θ), r)
      end
    end

    @testset "QuadraticFunction: the historical probe (indefinite A, #520 discovery)" begin
      a = 1.37 * 2.0^-970
      A = [0 a; a 0]                                  # not PSD: kept as evidence only
      x = [1.23 * 2.0^1000, 1.11 * 2.0^-60]
      @test within(QuadraticFunction(A)(x), big_q(A, x))
    end

    @testset "KronQuadratic: complete intermediate underflow (A D B is exactly zero)" begin
      for (T, ea, eb, et, er) in ((Float64, -900, -1000, 800, -301), (Float32, -110, -125, 80, -76))
        A = reshape(T[ldexp(one(T), ea)], 1, 1)
        B = reshape(T[ldexp(one(T), eb)], 1, 1)
        Θ = reshape(T[ldexp(one(T), et)], 1, 1)
        f = KronQuadratic(A, B)
        @test (A * Θ) * B == zeros(T, 1, 1)           # the ordinary products underflow
        @test f(Θ) === ldexp(one(T), er)                                  # exact
        G = zeros(T, 1, 1)
        @test value_and_gradient!(G, f, Θ) === ldexp(one(T), er)
        @test isequal(G, gradient!(zeros(T, 1, 1), f, Θ))
        @test f(-Θ) === ldexp(one(T), er)
        @test f(vec(Θ)) === ldexp(one(T), er)
        @test f(SparseIterate(Θ)) === ldexp(one(T), er)
        @test f(SymmetricSparseIterate(Θ)) === ldexp(one(T), er)
        @test KronQuadratic(A, B, zeros(T, 1, 1))(Θ) === ldexp(one(T), er)
        @test allocated_value(f, Θ) == 0
        # the linear term survives the recovery
        C = fill(one(T), 1, 1)
        fC = KronQuadratic(A, B, C)
        @test within(fC(Θ), big(ldexp(one(T), er)) + big(ldexp(one(T), et)))
      end
    end

    @testset "KronQuadratic: a small survivor after the linear term cancels the quadratic one" begin
      # v = Θ₁²AB₁₁/2 + ⟨C,Θ⟩ with ⟨C,Θ⟩ = -(1 - 2^-30) 2^-301: only 2^-331 is left
      A = fill(2.0^-900, 1, 1)
      B = Matrix{Float64}(2.0^-1000 * I, 2, 2)
      Θ = [2.0^800 2.0^-300]
      C = [0.0 -0.5 * (1 - 2.0^-30)]
      f = KronQuadratic(A, B, C)
      r = big_k(A, B, Θ) + dot(big.(C), big.(Θ))
      @test r == big(2.0)^-331                         # the data: the survivor is exact
      @test f(Θ) === 2.0^-331
      @test value_and_gradient!(zeros(1, 2), f, Θ) === 2.0^-331
    end

    @testset "KronQuadratic: rectangular Θ, PSD A and B, mixed magnitudes" begin
      for T in (Float32, Float64)
        s = T === Float32 ? (-110, -125, 80) : (-900, -1000, 800)   # exponents of A, B and Θ (A Θ B underflows)
        A = ldexp.(T[2 1; 1 2], s[1])
        B = ldexp.(T[2 1 0; 1 2 1; 0 1 2], s[2])
        Θ = ldexp.(T[1 -2 0; 0.5 1 3], s[3])
        f = KronQuadratic(A, B)
        @test near_scale(f(Θ), big_k(A, B, Θ), abs_k(A, B, Θ))        # scale
        @test near_scale(value_and_gradient!(zeros(T, 2, 3), f, Θ), big_k(A, B, Θ), abs_k(A, B, Θ))
        @test near_scale(f(vec(Θ)), big_k(A, B, Θ), abs_k(A, B, Θ))
        @test minimum(abs, (A * Θ) * B) < floatmin(T)                  # the ordinary products leave the normal range
      end
    end

    @testset "KronQuadratic: a SymmetricSparseIterate, and mixed precision" begin
      A = ldexp.([2.0 1; 1 2], -900)
      B = ldexp.([1.0 0; 0 3], -1000)
      Θ = ldexp.([1.0 0.25; 0.25 2], 800)
      f = KronQuadratic(A, B)
      r, s = big_k(A, B, Θ), abs_k(A, B, Θ)
      @test near_scale(f(SymmetricSparseIterate(Θ)), r, s)
      # a Float32 function at a Float64 argument: the buffers are Float32
      A32 = reshape(Float32[ldexp(1f0, -110)], 1, 1)
      B32 = reshape(Float32[ldexp(1f0, -125)], 1, 1)
      f32 = KronQuadratic(A32, B32)
      @test f32(reshape([ldexp(1.0, 80)], 1, 1)) === ldexp(1.0, -76)
      # a Float64 function at a Float32 argument is delivered in Float32 (no underflow in its buffers)
      f64 = KronQuadratic(reshape([ldexp(1.0, -110)], 1, 1), reshape([ldexp(1.0, -125)], 1, 1))
      @test f64(reshape(Float32[ldexp(1f0, 80)], 1, 1)) === ldexp(1f0, -76)
    end

    @testset "rounding of the final dots, and lossy scalings (review of #520)" begin
      # n products x[i] (A x)[i] = 1.5 2^-1074 each tie up to 2^-1073: the error grows with n and does not scale with max|x|
      n = 64
      x = fill(2.0^-537, n); x[1] = 2.0^-300
      A = zeros(n, n); for i in 2:n; A[i, i] = 1.5; end; A[1, 1] = 2.0^-421
      @test near_scale(QuadraticFunction(A)(x), big_q(A, x), abs_q(A, x))
      @test near_scale(value_and_gradient!(zeros(n), QuadraticFunction(A), x), big_q(A, x), abs_q(A, x))
      @test near_scale(QuadraticFunction(Diagonal(diag(A)))(x), big_q(A, x), abs_q(A, x))
      p = 16
      Ak = Matrix(1.5I, p, p); Ak[1, 1] = 2.0^-421
      Θk = fill(2.0^-537, p, p); Θk[1, 1] = 2.0^-300
      fk = KronQuadratic(Ak, Matrix{Float64}(I, p, p))
      @test near_scale(fk(Θk), big_k(Ak, Matrix{Float64}(I, p, p), Θk), abs_k(Ak, Matrix{Float64}(I, p, p), Θk))
      @test near_scale(value_and_gradient!(zeros(p, p), fk, Θk), big_k(Ak, Matrix{Float64}(I, p, p), Θk), abs_k(Ak, Matrix{Float64}(I, p, p), Θk))
      # a huge entry of A scales Θ into the subnormal range: the scaled operand loses entries, the exact tier runs
      let H = 2.0^1000
        B = [0 H 0; H 1 2.0^80; 0 2.0^80 1]
        Θ = [0 2.0^50 1.37 * 2.0^-30]
        f = KronQuadratic(fill(1.0, 1, 1), B)
        @test isnan(dot(Θ, (fill(1.0, 1, 1) * Θ) * B) / 2)               # the ordinary value
        @test near_scale(f(Θ), big_k(fill(1.0, 1, 1), B, Θ), abs_k(fill(1.0, 1, 1), B, Θ))
      end
      let A = zeros(4, 4), B = zeros(3, 3), Θ = zeros(4, 3)
        A[1, 3] = A[3, 1] = 2.0^480; A[1, 4] = A[4, 1] = -2.0^920
        B[2, 3] = B[3, 2] = 2.0^56
        Θ[1, 2] = 2.0^80; Θ[3, 3] = 1.37 * 2.0^-32
        f = KronQuadratic(A, B)
        @test near_scale(f(Θ), big_k(A, B, Θ), abs_k(A, B, Θ))
      end
      # an overflow whose terms meet exact zeros: the ordinary value is NaN, the true one is finite
      let A = Float32[6.9088907f-15 1.4637222f22 7.66848f-39 -1.3283052f-32 0.0 7.732994f-28 0.0;
                      -2.527356f23 0.0 2.7910939f29 -55866.508 -6.0957953f-12 2.0488126f8 -8.1965946f17;
                      4.0f-45 -0.0002509096 -3.0f-45 0.0 3.0f-45 3.0f-45 -1.0f-45;
                      -1.0f-45 -1751.5917 0.0 1.0f-45 1.0f-45 1.0f-45 8.0922083f-38;
                      -1.0f-45 -6.202017f-12 1.0f-45 1.0f-45 -1.0f-45 1.0f-45 8.006708f-35;
                      1.2721369f-27 5482.0356 -1.0f-45 -3.0f-45 0.0 3.0f-45 0.0;
                      0.0 2.4653365f27 1.0f-45 0.0 0.0 -1.6910727f-35 2.1856801f-26]
        x = Float32[1.4967584f-13, -0.0, -4.0337943f12, 26206.658, -8.882388f20, 859.72455, 0.0]
        @test isnan(dot(x, A * x) / 2)
        @test near_scale(QuadraticFunction(A)(x), big_q(A, x), abs_q(A, x))
        @test near_scale(QuadraticFunction(sparse(A))(x), big_q(A, x), abs_q(A, x))
      end
    end

    @testset "KronQuadratic: tiers of the recovery" begin
      # A tiny entry of A against a huge entry of Θ and a large one against a tiny one (found by search): one
      # scale per stage cannot hold both, the scaled result fails its check and the exact tier runs (it allocates)
      pinned = (
        (Float32[5.4559455f-38, 0.0035407313, 0.97407764], 3.3879756e-14, Float32[2.6343046f11, 1.110209f-12, 1.0143073f-16]),
        (Float32[4f-45, 0.6805282, 6.81f-43], 1.1492f-41, Float32[1.0009612f30, 7.999302f-19, 8.481519f13]),
        ([8.569248646228655e-219, 1.4956961140376902e-75, 1.2283957343234676e-182], 3.297208433286644e-258,
         [3.812425966616462e112, 6.540330316083966e-53, 1.0238183678489816e124]),
        ([1.8886401986653776e-153, 1.0e-323, 8.077213551578974e-54], 1.204540084984646e-276,
         [1.6492855484872404e-48, 4.347812028042161e282, 2.697153238521265e129]),
        ([4.945607044100949e-278, 7.4337e-320, 3.713225189883908e-147], 1.596082277630662e-283,
         [9.064938250452867e67, 1.2388865030405628e7, 1.4515369764334937e71]),
      )
      tier2 = Int[]
      for (d, b, θ) in pinned
        T = eltype(d)
        A = Matrix(Diagonal(d))
        B = fill(T(b), 1, 1)
        Θ = reshape(θ, 3, 1)
        f = KronQuadratic(A, B)
        r = big_k(A, B, Θ)
        @test floatmin(T) <= abs(r) <= floatmax(T)
        @test within(f(Θ), r, 3)                                            # ulps (sum of positive terms)
        f(Θ); push!(tier2, @allocated f(Θ))                                 # the exact tier allocates
        @test within(value_and_gradient!(zeros(T, 3, 1), f, Θ), r, 3)
        @test within(f(vec(Θ)), r, 3)
        @test within(f(SparseIterate(Θ)), r, 3)
        @test within(ProximalBase._expsum_value(T, (copyto!(f.D, Θ); ProximalBase._kron_exact_sum(f, Float64))), r, 3)
      end
      @test count(>(0), tier2) >= 1
      # the scaled-result check itself: it fails for a result below the bound, passes above it
      @test !ProximalBase._kron_scaled_reliable(Float64, 0.0, 2, 2, 0, 0, 0)
      @test ProximalBase._kron_scaled_reliable(Float64, 1.0, 2, 2, 0, 0, 0)
      # a threshold that would underflow in floatmin * max|Θ| is still evaluated (review of the first version)
      A = [6.73812617482326e-198 2.918153251204995e-124; 2.918153251204995e-124 7.378596022339728e-212]
      B = [-7.011171964901988e-148 -2.949589487225788e86; -2.949589487225788e86 -3.611665e-317]
      Θ = [2.2740868514538223e-225 1.3637911615273998e-275; 1.3637911615273998e-275 6.166992690142779e-19]
      fr = KronQuadratic(A, B)
      @test near_scale(fr(Θ), big_k(A, B, Θ), abs_k(A, B, Θ))
      @test within(fr(Θ), big_k(A, B, Θ), 4)
      @test ProximalBase._underflow_threshold(Float64, 1e87, 6e-19) > 1e-250
    end

    @testset "QuadraticFunction: PSD data with ordinary and underflow-risk entries" begin
      for (T, k, lo) in ((Float32, 16, -140), (Float64, 28, -1060))
        a = nextfloat(zero(T))
        d = T[1, a, 0, ldexp(one(T), lo), 3]
        xs = T[ldexp(one(T), lo ÷ 2 - 10), ldexp(one(T), k) + T(0.25), 0, ldexp(one(T), k - 3) + T(0.5), ldexp(one(T), lo ÷ 2)]
        A = Matrix(Diagonal(d))
        ref = big_q(A, xs)
        @test within(QuadraticFunction(A)(xs), ref, 4)
        @test within(QuadraticFunction(Diagonal(d))(xs), ref, 4)
        @test within(QuadraticFunction(sparse(Diagonal(d)))(xs), ref, 4)
        @test within(QuadraticFunction(Symmetric(A))(xs), ref, 4)
        @test within(QuadraticFunction(A)(SparseIterate(xs)), ref, 4)
        @test within(value_and_gradient!(zeros(T, 5), QuadraticFunction(A), xs), ref, 4)
        # a dense rank-one PSD matrix of subnormal entries
        n = 6
        Ar = fill(a, n, n)
        xr = [ldexp(one(T), k + mod(i, 3)) + T(i) / 8 for i in 1:n]
        @test within(QuadraticFunction(Ar)(xr), big_q(Ar, xr), 4)
        @test within(QuadraticFunction(Ar + Diagonal(fill(a, n)))(xr), big_q(Ar + Diagonal(fill(a, n)), xr), 4)
        # nonzero linear term and constant, comparable to the quadratic term
        b = T[3a; zeros(T, 4)]
        cc = ldexp(one(T), k === 28 ? -1020 : -126)
        fb = QuadraticFunction(A, b, cc)
        refb = big_q(A, xs) + dot(big.(b), big.(xs)) + big(cc)
        @test within(fb(xs), refb, 4)
        @test within(value_and_gradient!(zeros(T, 5), fb, xs), refb, 4)
      end
    end

    @testset "QuadraticFunction: mixed precision" begin
      a = nextfloat(0f0)
      f32 = QuadraticFunction(reshape([a], 1, 1))
      x = [2.0^16 + 0.25]                                   # a Float64 argument, Float32 buffers
      @test within(f32(x), big(a) * big(x[1])^2 / 2)
      g = zeros(1)
      @test within(value_and_gradient!(g, f32, x), big(a) * big(x[1])^2 / 2)
      # a Float64 function at a Float32 argument
      f64 = QuadraticFunction(reshape([nextfloat(0.0)], 1, 1))
      v = f64(Float32[2f0^28])                              # 2^-1019 is not a Float32
      @test v === 0f0
    end

    @testset "values that underflow nothing take the ordinary path" begin
      for T in (Float32, Float64)
        Ao = T[3 1 0; 1 3 1; 0 1 3]
        xo = T[0.3, -1.1, 2]
        fo = QuadraticFunction(Ao, T[1, -2, 0.5], T(0.25))
        @test !PB._quadratic_underflow_risk(fo, xo, fo(xo), T)
        @test !PB._quadratic_underflow_risk(QuadraticFunction(Ao), xo, QuadraticFunction(Ao)(xo), T)
        # exact zeros: sparse and diagonal data with a zero block
        Az = sparse(Diagonal(T[2, 0, 5]))
        xz = T[1, 7, 1e3]
        fz = QuadraticFunction(Az)
        @test fz(xz) === dot(xz, Az * xz) / 2
        @test !PB._quadratic_underflow_risk(fz, xz, fz(xz), T)
        @test QuadraticFunction(zeros(T, 2, 2))(T[1, 2]) === zero(T)       # exact zero stays exact
        @test QuadraticFunction(Ao)(zeros(T, 3)) === zero(T)
        @test !PB._quadratic_underflow_risk(QuadraticFunction(Ao), zeros(T, 3), zero(T), T)
        # Kron: ordinary, zeros in D, sparse Θ
        Ak, Bk = T[2 1; 1 2], T[1 0; 0 3]
        Θo = T[0.5 -1; 2 0]
        fk = KronQuadratic(Ak, Bk)
        @test fk(Θo) === dot(Θo, (Ak * Θo) * Bk) / 2
        copyto!(fk.D, Θo)
        @test !PB._kron_underflow_risk(fk, fk(Θo))
        @test KronQuadratic(Ak, Bk)(zeros(T, 2, 2)) === zero(T)
        # non-finite inputs follow the ordinary expression
        for xn in (T[NaN, 0, 0], T[Inf, 0, 0], T[-Inf, 1, 2], T[Inf, Inf, Inf])
          @test isequal(QuadraticFunction(Ao)(xn), dot(xn, Ao * xn) / 2)
        end
        for Θn in (T[NaN 0; 0 0], T[Inf 0; 0 0], T[Inf 1; 2 -Inf])
          @test isequal(KronQuadratic(Ak, Bk)(Θn), dot(Θn, (Ak * Θn) * Bk) / 2)
        end
      end
    end

    @testset "the stage-one shortcut of the Kron trigger (second review of #520)" begin
      # Julia's exponent has a significand in [1, 2): |a| |d| >= 2^(ea + ed), so ea + ed >= exponent(floatmin)
      # rules out a subnormal product of two entries (frexp's [1/2, 1) would need a margin of 2).
      for T in (Float32, Float64)
        emin = exponent(floatmin(T))
        rng = Xoshiro(5200)
        for _ in 1:20_000
          ea, ed = rand(rng, -3:3), rand(rng, emin-3:emin+3)
          a = ldexp(T(1) + rand(rng, T) * (1 - eps(T)), ea) * rand(rng, (T(1), T(-1)))
          d = ldexp(T(1) + rand(rng, T) * (1 - eps(T)), ed)
          (exponent(a) + exponent(d) >= emin) && @test abs(a * d) >= floatmin(T)
        end
        @test exponent(T(0.5)) + exponent(floatmin(T)) < emin       # the reviewer's example: not taken
        p, q = 1, 4                                                  # p (p + q) < q^2: decided from exponents
        B = Matrix(T(2)^(emin ÷ 2) * I, q, q)
        amp(a, d) = (f = KronQuadratic(fill(T(a), 1, 1), B); f.D .= reshape(T.(d), 1, :); PB._stage_one_amplification(f))
        @test amp(1, [floatmin(T), 1, 1, 1]) == 0                    # ea + ed == exponent(floatmin): product normal
        @test amp(T(1.75), [floatmin(T), 1, 1, 1]) == 0              # 1.75 floatmin is normal
        @test amp(T(0.5), [floatmin(T), 1, 1, 1]) == maximum(B)      # 2^-1 floatmin is subnormal: not taken
        @test amp(T(0.875), [T(1.5) * floatmin(T), 1, 1, 1]) == maximum(B)   # conservative at ea + ed = emin - 1
        @test amp(1, [0, 0, 1, 0]) == 0                              # zeros are skipped (their products are exact)
        @test amp(1, [prevfloat(floatmin(T)), 1, 1, 1]) == maximum(B)
      end
      # cancellation in A D (exact, since a sum that is subnormal is exact) and a large B, with the
      # shortcut taken: the value keeps its error against the scale of the terms
      for (T, ed, eb) in ((Float32, -126, 90), (Float64, -1022, 900))
        rng = Xoshiro(5201)
        p, q = 2, 8
        A = T[1 1; 1 1]
        D1 = [ldexp(T(1) + rand(rng, T) / 2, ed + rand(rng, 0:3)) for _ in 1:q]
        D = vcat(D1', -D1' .* (1 - eps(T)) .* T[(j % 3 == 0 ? 1 : 1 - eps(T)) for _ in 1:1, j in 1:q])
        Bm = ldexp.(Matrix(Symmetric(rand(rng, T, q, q))) + T(q) * I, eb)
        f = KronQuadratic(A, Bm)
        @test p * (p + q) < q * q
        r, sc = big_k(A, Bm, D), abs_k(A, Bm, D)
        @test near_scale(f(D), r, sc)
        @test near_scale(value_and_gradient!(zeros(T, p, q), f, D), r, sc)
        D0 = vcat(D1', -D1')                                       # A D = 0: the value is exactly 0
        @test f(D0) === zero(T)
      end
    end

    @testset "the bit-pattern helpers _maximum_abs and _min_nonzero_exponent (second review of #520)" begin
      for T in (Float32, Float64)
        fm, fx, dm = floatmin(T), floatmax(T), nextfloat(zero(T))
        @test PB._maximum_abs(T[]) === zero(T)
        @test PB._maximum_abs(T[0, -0.0, 0]) === zero(T)               # signed zeros
        @test PB._maximum_abs(T[-3, 2, 1]) === T(3)
        @test PB._maximum_abs(T[prevfloat(fm), -fm]) === fm             # both sides of floatmin
        @test PB._maximum_abs(T[-prevfloat(fm), 0]) === prevfloat(fm)
        @test PB._maximum_abs(T[-dm, 0]) === dm                          # subnormals
        @test PB._maximum_abs(T[fx, -fx, 1]) === fx
        @test PB._maximum_abs(T[1, -Inf]) === T(Inf)
        @test PB._maximum_abs(T[-Inf, 1, NaN]) |> isnan                  # a NaN is kept, wherever it is
        @test PB._maximum_abs(T[NaN, 1]) |> isnan
        @test PB._maximum_abs(T[1, 2, -NaN]) |> isnan
        @test isequal(PB._maximum_abs(T[NaN, 1]), maximum(abs, T[NaN, 1]))
        @test PB._maximum_abs(sparse(T[0 -5; 2 0])) === T(5)             # stored values only
        @test PB._maximum_abs(view(T[1 9; 2 8; 3 7], 1:2, 1:1)) === T(2)  # views
        @test PB._maximum_abs(view(T[1, 9, 2, 8, 3, 7], 1:2:5)) === T(3)
        @test PB._maximum_abs(T[1 2; 3 4]') === T(4)
        # the float fallback is unchanged
        @test PB._maximum_abs(Float16[-3, 2]) === Float16(3)
        @test PB._maximum_abs([1, -3]) === 3.0
        @test PB._maximum_abs(BigFloat[-3, 2]) == 3

        ef = exponent(fx)
        @test PB._min_nonzero_exponent(T[]) == ef                        # nothing: exponent(floatmax)
        @test PB._min_nonzero_exponent(T[0, -0.0]) == ef
        @test PB._min_nonzero_exponent(T[3, -0.5, 0]) == -1
        @test PB._min_nonzero_exponent(T[fm]) == exponent(fm)            # both sides of floatmin
        @test PB._min_nonzero_exponent(T[prevfloat(fm)]) == exponent(fm) - 1
        @test PB._min_nonzero_exponent(T[-dm]) == exponent(dm)           # subnormals
        @test PB._min_nonzero_exponent(T[fx, -fx]) == ef
        @test PB._min_nonzero_exponent(T[Inf, 4]) == 2                   # an Inf or a NaN is not a minimum
        @test PB._min_nonzero_exponent(T[NaN, 2]) == 1                   # and never reaches exponent
        @test PB._min_nonzero_exponent(T[NaN]) == ef
        @test PB._min_nonzero_exponent(T[-Inf, 0]) == ef
        @test PB._min_nonzero_exponent(sparse(T[0 -5; 2 0])) == exponent(T(2))  # nonzero entries
        @test PB._min_nonzero_exponent(view(T[1 9; 4 8], 2:2, 1:1)) == 2
        # the generic method (Float16) keeps its behavior on finite data
        @test PB._min_nonzero_exponent(Float16[8, -2, 0]) == 1
        # nothing is allocated
        v = rand(T, 64)
        @test allocated_value(PB._maximum_abs, v) == 0
        @test allocated_value(PB._min_nonzero_exponent, v) == 0
      end
    end

    @testset "no recovery when no product can be subnormal (second review of #520)" begin
      for T in (Float32, Float64)
        emin = exponent(floatmin(T))
        # A = 0, and an argument in the null space of A: the value is exactly zero at a nonzero x, below
        # the worst-case threshold, and the exponent test shows that nothing was lost
        for A in (zeros(T, 3, 3), T[1 -1 0; -1 1 0; 0 0 0], sparse(zeros(T, 3, 3)), Diagonal(zeros(T, 3)))
          f = QuadraticFunction(A)
          x = T[1, 1, 3]
          x = A isa Matrix{T} && A[1, 1] == 1 ? T[1, 1, 3] : x
          v = f(x)
          @test v === zero(T)
          @test PB._quadratic_underflow_risk(f, x, v, T)
          @test PB._quadratic_products_normal(f, x, v, T)
          @test value_and_gradient!(zeros(T, 3), f, x) === zero(T)
          fb = QuadraticFunction(A, T[0, 0, 0])
          @test PB._quadratic_products_normal(fb, x, v, T)
        end
        # a subnormal product of A and x: the recovery is needed
        f = QuadraticFunction(reshape(T[nextfloat(zero(T))], 1, 1))
        x = T[ldexp(one(T), T === Float32 ? 16 : 28)]
        @test !PB._quadratic_products_normal(f, x, f(x), T)
        # a subnormal product in the final dot, with normal products in A x
        f = QuadraticFunction(reshape(T[ldexp(one(T), emin ÷ 2)], 1, 1), T[ldexp(one(T), emin ÷ 2 - 1)])
        x = T[ldexp(one(T), emin - emin ÷ 2 - 1)]
        @test !PB._quadratic_products_normal(f, x, f(x), T)
        # a wider argument than the buffers is rounded elsewhere: not covered by the test
        W = T === Float32 ? Float64 : BigFloat
        T === Float32 && (@test !PB._quadratic_products_normal(QuadraticFunction(zeros(T, 2, 2)), W[1, 1], zero(T), T))
        # Kron: the same, on A D, (A D) B, D W and C D
        for (A, B) in ((zeros(T, 2, 2), T[1 0; 0 1]), (T[1 -1; -1 1], T[1 -1; -1 1]), (T[1 0; 0 1], zeros(T, 2, 2)))
          fk = KronQuadratic(A, B)
          Θ = T[1 2; 3 5]
          vk = fk(Θ)
          @test isfinite(vk)
          @test PB._kron_products_normal(fk, vk)
          @test within(vk, big_k(A, B, Θ)) || iszero(big_k(A, B, Θ))
        end
        fk = KronQuadratic(reshape(T[nextfloat(zero(T))], 1, 1), reshape(T[ldexp(one(T), emin ÷ 2)], 1, 1))
        Θk = reshape(T[ldexp(one(T), T === Float32 ? 16 : 28)], 1, 1)
        @test !PB._kron_products_normal(fk, fk(Θk))
      end
    end

    @testset "the fused product and maximum for a diagonal A (second review of #520)" begin
      for T in (Float32, Float64)
        rng = Xoshiro(5202)
        for n in (1, 2, 7, 64, 129)
          d, x = randn(rng, T, n), randn(rng, T, n)
          y, y2 = zeros(T, n), zeros(T, n)
          @test PB._mul_maxabs!(y, Diagonal(d), x) === maximum(abs, x)
          mul!(y2, Diagonal(d), x)
          @test y == y2                                                 # the same products
          xv = view([zero(T); x; zero(T)], 2:n+1)                       # a strided view
          @test PB._mul_maxabs!(fill!(y, 0), Diagonal(d), xv) === maximum(abs, x) && y == y2
        end
        x = T[1, -Inf, NaN]
        @test isnan(PB._mul_maxabs!(zeros(T, 3), Diagonal(ones(T, 3)), x))
        # other types and layouts fall back to mul! and the maximum of the stored entries
        W = T === Float32 ? Float64 : Float32
        y = zeros(T, 2)
        @test PB._mul_maxabs!(y, Diagonal(T[2, 3]), W[1, -4]) == 4 && y == T[2, -12]
        @test PB._mul_maxabs!(zeros(T, 2), T[1 2; 3 4], T[1, -5]) === T(5)
        f = QuadraticFunction(Diagonal(T[2, 3]), T[1, 1], T(1))
        @test f(T[1, -2]) == (T(2) * 1 + T(3) * 4) / 2 + (1 - 2) + 1
      end
    end

    @testset "a wider argument than the coefficients is rounded into the buffers (known limitation)" begin
      # The Float32 buffer of a Float32 KronQuadratic receives a Float64 argument rounded to Float32, which
      # is zero here; T3 asks for the argument not to be rounded below the model precision. This is the
      # behavior before #520 too; the trigger sees max|D| = 0 and cannot recover it.
      # Tracked in #584; keep this @test_broken until that issue is fixed.
      A = fill(ldexp(1f0, -100), 1, 1)
      f = KronQuadratic(A, copy(A))
      Θ = fill(ldexp(1.0, -200), 1, 1)
      @test_broken isequal(f(Θ), ldexp(1.0, -601))
    end

    @testset "_pow2_sum cancellation, as the underflow path will use it" begin
      for T in (Float32, Float64)
        E = T === Float32 ? 30 : 400
        base = ((T(0.5), 2E), (T(-0.5), 2E), (T(0.25), -E))
        for perm in ((1, 2, 3), (3, 1, 2), (2, 3, 1), (3, 2, 1))
          @test PB._pow2_sum(T, base[perm[1]], base[perm[2]], base[perm[3]]) === ldexp(T(0.25), -E)
        end
      end
    end
  end

end

end
