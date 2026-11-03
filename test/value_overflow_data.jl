module ValueOverflowDataTest

# Values of the functions that hold data when a matrix product, a quadratic form
# or a dot leaves the floating-point range at a representable value (R1, #358).

using Test
using ProximalBase
using LinearAlgebra
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
        # On the product the call stored: BLAS may round X * β and mul! differently.
        # The value is not compared bit for bit with plain_leastsquares: two @simd
        # reductions may add in different orders, which differ by a unit in the last
        # place on some CPUs (#537). The guard keeps the ordinary sum exactly ...
        s = 2 * plain_leastsquares(Yr, Xr, βr, fr.tmp)
        @test PB._leastsquares_value(fr, βr, s, true) === s / 2
        # ... and the value has the accuracy of the ordinary computation (R1): one
        # rounding per difference and per square and n - 1 additions in any order,
        # (n + 2)u relative to first order, u = eps(T)/2, as the terms are nonnegative.
        exact = sum(abs2, big.(Yr) .- big.(fr.tmp)) / 2
        @test isfinite(v) && abs(big(v) - exact) <= (n + 3) * big(eps(T)) / 2 * exact
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
end

end
