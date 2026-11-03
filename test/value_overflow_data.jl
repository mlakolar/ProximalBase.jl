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

# the base's ordinary least-squares loop
function plain_leastsquares(Y, X, β)
  tmp = X * β
  v = zero(eltype(tmp))
  @inbounds @simd for i in eachindex(tmp, Y)
    v += abs2(Y[i] - tmp[i])
  end
  v / 2
end

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
        @test LeastSquaresLoss(Yr, Xr)(βr) === plain_leastsquares(Yr, Xr, βr)
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
      # a large matrix, a large linear term and a 2 × 2 form
      @test within(QuadraticFunction(fill(M, 1, 1))(T[2^-10]), big(M) * big(2.0)^-20 / 2)
      A2 = T[2 1; 1 2]
      b2 = T[1, -1] .* (root / 8)
      x2 = T(0.6) .* [root, -root]                   # x'Ax overflows, x'Ax/2 does not
      ref = (big.(x2)' * big.(A2) * big.(x2)) / 2 + big.(b2)' * big.(x2)
      scale = (abs.(big.(x2))' * big.(A2) * abs.(big.(x2))) / 2 + abs.(big.(b2))' * abs.(big.(x2))
      @test near_scale(QuadraticFunction(A2, b2)(x2), ref, scale)
      @test near_scale(QuadraticFunction(A2, b2)(SparseIterate(x2)), ref, scale)
      # the recovery allocates nothing
      xr = [root]
      f(xr)
      @test (@allocated f(xr)) == 0
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
      fk(Θ)
      @test (@allocated fk(Θ)) == 0
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
      # argument and coefficient types
      @test LinearFunction(T[2, 0], -M)(SparseIterate(T[M, 0])) === M
      @test LinearFunction(T[2 0; 0 0], -M)(SymmetricSparseIterate(T[M 0; 0 0])) === M
      @test LinearFunction(T[0 2; 2 0])(SymmetricSparseIterate(T[0 M; M 0])) === T(Inf)   # 4M overflows
      @test LinearFunction(T[0 1; 1 0], -M)(SymmetricSparseIterate(T[0 M; M 0])) === M
      @test LinearFunction(sparse(T[2, 0]), -M)(T[M, 1]) === M
      @test LinearFunction(Diagonal(T[2, 1]), -M)(Diagonal(T[M, 0])) === M
      @test LinearFunction(T[2, 0], -M)(sparse(T[M, 0])) === M
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
end

end
