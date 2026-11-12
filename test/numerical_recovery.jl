module NumericalRecoveryTest

# Audit of the live overflow/underflow/cancellation recovery paths of src/ (docs/audit/numerical-kernels-file/
# recovery-coverage.md has the inventory and the coverage evidence). Every case reaches an exceptional
# branch of the kernel under test: an intermediate product, difference, square or partial sum leaves the
# range of Float32/Float64 (or cancels) while the final value is representable; or an input is a genuine
# NaN/Inf, an exact or signed zero, or a subnormal. The oracle is BigFloat evaluated directly from the
# mathematical formula on the exact (Float32/Float64) data; the package never runs in BigFloat here.
#
# Accuracy is stated in units in the last place (ulp) of the correctly rounded reference, with the bound each
# algorithm gives: a recomputation from significands rounds a handful of times (<= 2-4 ulp; the documented
# contract of the three-factor sparse/symmetric terms is one neighbouring float for a normal term and one
# spacing for a subnormal one), a compensated sum is accurate to the unit roundoff of its result, sums of
# terms in storage order carry an error relative to the sum of the magnitudes of the terms (as docs/src/numerics.md
# says). The retained
# frozen ridge references are in ridge_precision.jl, the primitive kernels in numerical_kernels.jl.
#
# Each test function builds its own seeded Xoshiro; the cases that need a long random sample say so.

using Test, ProximalBase, Random, SparseArrays, LinearAlgebra
const PB = ProximalBase

pow2(T, k) = ldexp(one(T), k)
emax(T) = exponent(floatmax(T))
emin(T) = exponent(floatmin(T))
# |a - r| within n units in the last place of the rounded reference (subnormal units included, so a reference that
# rounds to zero allows n subnormal units); a reference that rounds to an infinity must be reproduced exactly
function within(a::T, r::BigFloat, n) where {T}
  t = T(r)
  isfinite(t) || return a == t
  isfinite(a) && abs(big(a) - r) <= n * eps(abs(t))
end
relerr(a::BigFloat, r::BigFloat) = iszero(r) ? abs(a) : abs(a - r) / abs(r)
# a random significand in [1/2, 1) with a random sign, times 2^e
signif(rng, T, e) = (rand(rng, Bool) ? -1 : 1) * ldexp(T(0.5) + T(rand(rng)) / 2, e)
# the value m 2^e of a significand/exponent pair, exactly
bigparts(m, e) = big(m) * big(2.0)^e
# the exact leaf values and the soft threshold, in BigFloat
l1val(λ, w, x) = big(λ) * sum((w === nothing ? ones(BigFloat, length(x)) : big.(w)) .* abs.(big.(x)))
sqval(λ, y, x) = big(λ) / 2 * sum((big.(x) .- (y === nothing ? zeros(BigFloat, size(x)) : big.(y))) .^ 2)
shrinkb(v, t) = sign(v) * max(abs(v) - t, 0)
accbig(a) = bigparts(a.s, a.ex) + bigparts(a.c, a.ex)   # the sum held by an _ExpSum, exactly
# the exact value x'Ax/2 + b'x + c of the data, in BigFloat
qref_simple(A, b, c, x) = (xb = big.(x); xb' * big.(Matrix(A)) * xb / 2 + (b === nothing ? big(0) : sum(big.(b) .* xb)) + big(c))

# a custom separable function without levels to fold: Σ wᵢ xᵢ² / 2 (the example of the SeparableFunction docstring)
struct WeightedSquare{W<:AbstractVector} <: SeparableFunction
  w::W
end
PB.value_entry(f::WeightedSquare, v, i) = f.w[i] * v^2 / 2
PB.prox_block(f::WeightedSquare, v, i, γ) = v / (1 + γ * f.w[i])
PB.iszeropreserving(::WeightedSquare) = true
PB.isconvex(::Type{<:WeightedSquare}) = true
PB.entry_parameters(f::WeightedSquare) = (:w => f.w,)
function PB.check_parameters(f::WeightedSquare, x)
  length(f.w) == prod(size(x)) || throw(DimensionMismatch("a parameter array has length $(length(f.w)) and the argument has $(prod(size(x))) entries"))
  nothing
end

function test_expsum_accumulator(::Type{T}) where {T<:AbstractFloat}
  E, e0 = emax(T), emin(T)
  δ = nextfloat(zero(T))
  precT = precision(T)
  rng = Xoshiro(1000)
  @testset "exponent-sum accumulator (_ExpSum)" begin
    R = Float64
    acc0 = PB._expsum_start(R)
    @test PB._expsum_value(T, acc0) === zero(T)
    # a zero term does not move the accumulator; the first term fixes the frame
    @test PB._expsum_add(acc0, 0.0, 77) === acc0
    a1 = PB._expsum_add(acc0, 0.75, -40)
    @test a1 === PB._ExpSum{R}(0.75, 0.0, -40)
    @test PB._expsum_add(a1, 0.0, 99) === a1
    # a term in a smaller frame is scaled down; one in a larger frame rescales the sum (the frame of the
    # larger exponent), and the value is exact where no bit falls below the frame
    a2 = PB._expsum_add(a1, 0.5, -45)
    @test a2.ex == -40 && bigparts(a2.s, a2.ex) + bigparts(a2.c, a2.ex) == big(0.75) * big(2.0)^-40 + big(0.5) * big(2.0)^-45
    a3 = PB._expsum_add(a2, -0.625, -38)
    @test a3.ex == -38
    @test bigparts(a3.s, a3.ex) + bigparts(a3.c, a3.ex) == big(0.75) * big(2.0)^-40 + big(0.5) * big(2.0)^-45 - big(0.625) * big(2.0)^-38
    # compensation: a small term survives a cancellation of the larger ones that absorbed it
    c = PB._expsum_add(PB._expsum_add(PB._expsum_add(acc0, 0.5, 600), 0.5 * 2.0^-70, 600), -0.5, 600)
    @test iszero(c.s) && c.c == 0.5 * 2.0^-70
    @test PB._expsum_value(Float64, c) == ldexp(0.5 * 2.0^-70, 600)
    @test PB._expsum_value(Float64, c) === ldexp(0.5, 530)
    # the value leaves the range only at the final conversion: Inf, 0, subnormal; exact zero sum is +0
    @test PB._expsum_value(Float32, PB._expsum_add(acc0, 0.5, 300)) === Inf32
    @test PB._expsum_value(Float32, PB._expsum_add(acc0, -0.5, 300)) === -Inf32
    @test PB._expsum_value(Float32, PB._expsum_add(acc0, 0.5, -300)) === 0.0f0
    @test PB._expsum_value(Float32, PB._expsum_add(acc0, 0.75, -148)) === Float32(ldexp(big(0.75), -148))
    @test PB._expsum_value(Float64, PB._expsum_add(PB._expsum_add(acc0, 0.75, 5), -0.75, 5)) === 0.0
    # many terms of widely different exponents: the sum of the terms, with the accuracy of a compensated sum
    for n in (2, 5, 40)
      es = rand(rng, -900:900, n)
      ms = [signif(rng, Float64, 0) for _ in 1:n]
      acc = acc0
      for i in 1:n
        acc = PB._expsum_add(acc, ms[i], es[i])
      end
      r = sum(bigparts(ms[i], es[i]) for i in 1:n)
      @test relerr(accbig(acc), r) <= 4 * eps(Float64)
    end
    # product, term and dot: significands multiplied (a b), exponents added with the power k
    for _ in 1:200
      a, b = signif(rng, T, rand(rng, e0:E)), signif(rng, T, rand(rng, e0:E))
      k = rand(rng, -300:300)
      acc = PB._expsum_product(acc0, a, b, k)
      @test relerr(bigparts(acc.s, acc.ex), big(a) * big(b) * big(2.0)^k) <= 2 * eps(Float64)
    end
    @test PB._expsum_product(acc0, zero(T), T(3)) === acc0
    @test PB._expsum_product(acc0, T(3), -zero(T)) === acc0
    accd = PB._expsum_dot(acc0, T[1, 2, -3], T[4, -5, 6], 3)
    @test PB._expsum_value(Float64, accd) == (4 - 10 - 18) * 8.0
    @test_throws DimensionMismatch PB._expsum_dot(acc0, T[1, 2], T[1, 2, 3], 0)
    # an overflowing dot with cancellation and a small remainder
    big_ = pow2(T, E - 2)
    accd = PB._expsum_dot(acc0, T[big_, big_, 1], T[big_, -big_, T(3)], 0)
    @test PB._expsum_value(Float64, accd) == 3.0
    # the quadratic form of a dense, sparse and diagonal matrix: Σ aᵢⱼ xᵢ xⱼ 2^k, from products that would
    # overflow or underflow in T, with exact zeros skipped
    for (A, x) in (
        (T[1 -1 0; -1 1 2; 0 2 3] .* pow2(T, E ÷ 2), T[pow2(T, E ÷ 2), pow2(T, E ÷ 2), 0]),
        (T[1 2 0; 2 -4 1; 0 1 5] .* pow2(T, e0 ÷ 2), T[pow2(T, e0 ÷ 2 - 3), 0, -pow2(T, e0 ÷ 2 - 2)]),
        (T[3 1; 1 2] .* T(1.1), T[T(0.7), T(-2.3)]))
      r = sum(big(A[i, j]) * big(x[i]) * big(x[j]) for i in 1:3, j in 1:3 if i <= size(A, 1) && j <= size(A, 2) && i <= length(x) && j <= length(x))
      for M in (A, sparse(A), Matrix(A))
        accq = PB._expsum_quadratic(PB._expsum_start(R), M, x, -1)
        @test relerr(accbig(accq), r / 2) <= 8 * eps(Float64)
      end
    end
    d = T[3, 0, -2] .* pow2(T, e0 ÷ 2)
    xd = T[pow2(T, e0 ÷ 2 - 1), 5, pow2(T, e0 ÷ 2 - 2)]
    accq = PB._expsum_quadratic(PB._expsum_start(R), Diagonal(d), xd, 0)
    @test relerr(accbig(accq), sum(big(d[i]) * big(xd[i])^2 for i in 1:3)) <= 4 * eps(Float64)
  end
end

function test_quadratic_value(::Type{T}) where {T<:AbstractFloat}
  E, e0 = emax(T), emin(T)
  δ = nextfloat(zero(T))
  precT = precision(T)
  rng = Xoshiro(1001)
  @testset "quadratic value: overflow, cancellation and underflow recovery" begin
    qref = qref_simple
    s = E ÷ 2 + 4
    # (1) the dot x'(Ax) overflows alone, and b'x cancels it exactly: v = c. Integer entries times powers
    # of two keep every product exact, so the cancellation is exact in the reference too.
    for trial in 1:12
      n = rand(rng, 2:5)
      Ai = rand(rng, -6:6, n, n); Ai = Ai + Ai'
      xe = rand(rng, s:s+2, n)
      xs = [rand(rng, Bool) ? pow2(T, e) : -pow2(T, e) for e in xe]
      A = T.(Ai) .* pow2(T, 4)
      Ax = big.(A) * big.(xs)
      b = T.(-Ax / 2)
      @test big.(b) == -Ax / 2                      # exact: the products are integers times a power of two
      c = signif(rng, T, rand(rng, -5:5))
      f = QuadraticFunction(A, b, c)
      r = qref(A, b, c, xs)
      @test r == big(c)
      @test isfinite(sum(xs' * (A * xs)) / 2) == false       # the ordinary dot is not finite
      v = f(xs)
      @test within(v, r, 1)
      g = zeros(T, n)
      @test value_and_gradient!(g, f, xs) === v          # the fused oracle takes the same recovery
      @test g == gradient!(zeros(T, n), f, xs)
    end
    # (2) A x overflows: a null-space direction of a signed matrix, whose partial sums overflow (Inf - Inf = NaN, or
    # Inf, as the BLAS kernel of the machine orders and fuses them)
    p = pow2(T, E - s + 5)
    Anull = [p -p; -p p]
    xs = T[pow2(T, s), pow2(T, s)]
    @test !isfinite((Anull * xs)[1])
    b2, c2 = T[1, 2], T(3)
    f = QuadraticFunction(Anull, b2, c2)
    @test within(f(xs), qref(Anull, b2, c2, xs), 1)
    @test within(f(xs), big(3) * big(pow2(T, s)) + 3, 1)
    f0 = QuadraticFunction(Anull)                       # no b, no c: the quadratic form alone is exactly zero
    @test f0(xs) === zero(T)
    @test QuadraticFunction(Anull, nothing, T(5))(xs) === T(5)
    # a true overflow stays Inf (the value is out of range), and stays -Inf with the sign of the value
    fo = QuadraticFunction(T[1 0; 0 1] .* pow2(T, 8), nothing, 0)
    @test fo(T[pow2(T, E - 3), 0]) === T(Inf)
    @test QuadraticFunction(T[-1 0; 0 -1] .* pow2(T, 8), nothing, 0)(T[pow2(T, E - 3), 0]) === -T(Inf)
    # (3) underflow: the products x'(Ax) are subnormal while A x is normal; the value is representable (subnormal)
    # and the ordinary dot has lost bits; the recovered value is rounded once
    lost = 0
    rng = Xoshiro(3)
    for trial in 1:40
      n = rand(rng, 3:6)
      a = e0 ÷ 2 - rand(rng, 10:14)                      # |A| ~ 2^a, and the entries of x ~ 2^xe, so that the value is a few subnormal units
      xe = (e0 - precT + 5 - a) ÷ 2
      A = [signif(rng, T, a) for _ in 1:n, _ in 1:n]; A = A + A'
      xs = [signif(rng, T, xe) for _ in 1:n]
      r = qref(A, nothing, 0, xs)
      @test big(2.0)^(e0 - precT - 1) < abs(r) < big(2.0)^(e0 - precT + 12)   # a few to a few thousand subnormal units
      f = QuadraticFunction(A)
      v = f(xs)
      @test abs(big(v) - r) <= 0.51 * eps(abs(T(r)))                # rounded once (half a unit, up to the tie margin)
      lost += abs(big(dot(xs, A * xs) / 2) - r) > 0.51 * eps(abs(T(r)))   # the ordinary evaluation rounded each product
      g = zeros(T, n)
      @test value_and_gradient!(g, f, xs) === v
    end
    @test lost >= 1                                                  # and it is not a rarity of the seed alone: see the cancellation case below
    # with b and c: the subnormal products of b'x do not matter next to a normal c, the value is the ordinary one
    A = T[1 0; 0 1] .* pow2(T, e0 ÷ 2); xs = T[pow2(T, e0 ÷ 4), pow2(T, e0 ÷ 4 - 1)]
    @test within(QuadraticFunction(A, T[1, 1] .* pow2(T, e0 ÷ 2), T(2))(xs), qref(A, T[1, 1] .* pow2(T, e0 ÷ 2), 2, xs), 1)
    # sparse and diagonal matrices take their own product kernels and stored-entry loops, the same recovery
    a = e0 ÷ 2 - 12; xe = (e0 - 8 - a) ÷ 2
    Ad = Diagonal(T[signif(rng, T, a), signif(rng, T, a - 1), signif(rng, T, a + 1)])
    xs = T[signif(rng, T, xe), signif(rng, T, xe + 1), signif(rng, T, xe - 1)]
    Asp = sparse(T[signif(rng, T, a) 0 signif(rng, T, a - 2); 0 0 0; signif(rng, T, a - 2) 0 signif(rng, T, a)])
    for M in (Ad, Asp)
      f = QuadraticFunction(M)
      r = qref(M, nothing, 0, xs)
      @test within(f(xs), r, 1)
      g = zeros(T, 3)
      @test value_and_gradient!(g, f, xs) === f(xs)
      @test g == gradient!(zeros(T, 3), f, xs)
    end
    # a sparse iterate argument: only the stored entries are read, an unstored coordinate is an exact zero
    xsp = SparseIterate(T, 3); xsp[1] = xs[1]; xsp[3] = xs[3]
    f = QuadraticFunction(Matrix(Ad)); xd = T[xs[1], 0, xs[3]]
    @test f(xsp) === f(xd)
    @test within(f(xsp), qref(Ad, nothing, 0, xd), 1)
  end
end

function test_quadratic_guards(::Type{T}) where {T<:AbstractFloat}
  E, e0 = emax(T), emin(T)
  δ = nextfloat(zero(T))
  precT = precision(T)
  rng = Xoshiro(1002)
  @testset "quadratic recovery trigger and guards" begin
    n = 2
    A0 = T[1 -1; -1 1]
    # an exact zero of a null-space argument, with every product normal: the ordinary value 0 is exact and the
    # underflow test fires but the products are normal, so nothing is recomputed (the result is the same zero)
    f = QuadraticFunction(A0)
    x = T[3, 3]
    v = f(x)
    @test v === zero(T)
    @test PB._quadratic_underflow_risk(f, x, v, T)
    @test PB._quadratic_products_normal(f, x, v, T)
    @test PB._quadratic_checked(f, x, v, T) === v
    # subnormal products (A and x small): the same test does not allow keeping the ordinary value
    fs = QuadraticFunction(A0 .* pow2(T, e0 ÷ 2))
    xs = T[1, 1] .* pow2(T, e0 ÷ 2 - 2)
    vs = fs(xs)
    @test PB._quadratic_underflow_risk(fs, xs, vs, T)
    @test !PB._quadratic_products_normal(fs, xs, vs, T)
    # no risk at an exact zero of x, or when |v| is large compared with the underflow threshold
    @test !PB._quadratic_underflow_risk(f, zeros(T, 2), zero(T), T)
    @test !PB._quadratic_underflow_risk(f, T[1, 2], T(100), T)
    @test PB._quadratic_underflow_risk(f, T[1, 2], zero(T), T)
    # the threshold: kappa floatmin max|x| + N floatmin, in Float64, Inf when it overflows
    @test PB._underflow_threshold(T, 3, T(5), 7) == (3.0 * Float64(floatmin(T))) * 5.0 + 7 * Float64(floatmin(T))
    @test PB._underflow_threshold(T, 2^40, T(1), 0) == 2.0^40 * Float64(floatmin(T))
    @test PB._underflow_threshold(T, 1e308, floatmax(T), 0) === Inf
    # the smallest nonzero exponent over the stored entries (the bit-pattern kernel and the generic one)
    @test PB._min_nonzero_exponent(T[0, 3, -0.375, floatmax(T), T(Inf)]) == -2
    @test PB._min_nonzero_exponent(T[0, T(0), -zero(T)]) == emax(T)
    @test PB._min_nonzero_exponent(T[floatmax(T), 0]) == emax(T)
    @test PB._min_nonzero_exponent(T[T(NaN), 4]) == 2
    @test PB._min_nonzero_exponent([δ, 1, 2]) == exponent(δ) == emin(T) - precision(T) + 1
    @test PB._min_nonzero_exponent(Diagonal(T[0, 8, 0.5])) == -1
    @test PB._min_nonzero_exponent(sparse(T[0 0.25; 0 4])) == -2
    @test PB._min_nonzero_exponent(BigFloat[3, 0.1]) == -4                  # the generic method (no bit patterns)
    @test PB._stored_minexp(transpose(sparse(T[0 0.25; 0 4]))) == -2
    # genuine NaN/Inf data keep the ordinary result: no recomputation hides them
    fi = QuadraticFunction(T[2 1; 1 3], T[1, 1], T(1))
    @test fi(T[Inf, 1]) === T(Inf)
    @test isnan(fi(T[NaN, 1]))
    @test isnan(QuadraticFunction(T[Inf 0; 0 1])(T[0, 1])) && QuadraticFunction(T[Inf 0; 0 1])(T[1, 1]) === T(Inf)    # Inf*0 is a genuine NaN
    @test QuadraticFunction(T[2 0; 0 3], nothing, T(Inf))(T[1, 1]) === T(Inf)
    @test isnan(QuadraticFunction(T[2 0; 0 3], T[NaN, 1], T(1))(T[1, 1]))
    @test isnan(QuadraticFunction(T[1 0; 0 1])(T[Inf, -Inf]))                      # 0 * Inf in the product A x
    # the recovery itself leaves the ordinary result where the data are not finite (a direct call)
    @test PB._quadratic_value_recover(fi, T[Inf, 1], T(7)) === T(7)
    @test PB._quadratic_value_recover(QuadraticFunction(T[Inf 0; 0 1], T[1, 1], T(1)), T[1, 1], T(7)) === T(7)
    @test PB._quadratic_value_recover(QuadraticFunction(T[1 0; 0 1], T[Inf, 1], T(1)), T[1, 1], T(7)) === T(7)
    @test PB._quadratic_value_recover(QuadraticFunction(T[1 0; 0 1], T[1, 1], T(Inf)), T[1, 1], T(7)) === T(7)
    # signed zeros: x = ±0 gives +0 or the exact value of b and c; a negative zero constant keeps a zero value
    @test QuadraticFunction(T[1 2; 2 3])(T[0, -0.0]) === zero(T)
    @test QuadraticFunction(T[1 2; 2 3], T[1, 1], T(2))(T[0, -0.0]) === T(2)
    @test QuadraticFunction(T[1 2; 2 3], nothing, -zero(T))(T[0, -0.0]) == 0
  end
end

function test_quadratic_oracles(::Type{T}) where {T<:AbstractFloat}
  E, e0 = emax(T), emin(T)
  δ = nextfloat(zero(T))
  precT = precision(T)
  rng = Xoshiro(1003)
  @testset "quadratic oracles: fused and standalone, eigendecomposition prox" begin
    rng = Xoshiro(5)
    n = 4
    Q = Matrix(qr(randn(rng, n, n)).Q)
    Λ = T[0.5, 1, 1.5, 2]
    A = T.(Q * Diagonal(Λ) * Q'); A = (A + A') / 2
    b = T.(randn(rng, n)); c = T(0.25)
    f = QuadraticFunction(A, b, c)
    x = T.(randn(rng, n))
    g1, g2 = zeros(T, n), zeros(T, n)
    @test value_and_gradient!(g1, f, x) === f(x)
    @test g1 == gradient!(g2, f, x)
    @test within(f(x), qref_simple(A, b, c, x), 8)
    # the gradient Ax + b to the accuracy of a dot product of length n: n eps |A||x|
    gref = big.(A) * big.(x) + big.(b)
    @test all(abs.(big.(g1) .- gref) .<= n * eps(T) .* (abs.(big.(A)) * abs.(big.(x)) .+ abs.(big.(b))))
    # a gradient of another element type: the fused oracle returns the standalone value and the standalone gradient
    gw = zeros(Float64, n)
    T === Float32 && (@test value_and_gradient!(gw, f, x) == f(x); @test gw == gradient!(zeros(Float64, n), f, x))
    # prox: (I + γA) z = x - γb, solved in BigFloat; the eigendecomposition is computed once, on first use
    for γ in (T(1e-3), T(1), T(50))
      reset_cache!(f)
      z = prox!(zeros(T, n), f, x, γ)
      zb = (Matrix{BigFloat}(I, n, n) + big(γ) * big.(A)) \ (big.(x) - big(γ) * big.(b))
      @test all(abs.(big.(z) .- zb) .<= 30 * n * eps(T) * maximum(abs, zb))
      @test prox!(zeros(T, n), f, x, γ) == z                               # the cached factorization
      xa = copy(x)
      @test prox!(xa, f, xa, γ) == z                                       # exact input/output aliasing
      y = zeros(T, n)
      v = value_and_prox!(y, f, x, γ)
      @test y == z && v === f(y)                                           # the value is that at the stored point
      @test prox(f, x, γ) == z
    end
    # the prox without b is the plain resolvent; a zero-matrix function has the identity prox
    f0 = QuadraticFunction(zeros(T, n, n))
    @test prox!(zeros(T, n), f0, x, T(7)) ≈ x rtol = 10 * eps(T)
  end
end

function test_scaled_values(::Type{T}) where {T<:AbstractFloat}
  E, e0 = emax(T), emin(T)
  δ = nextfloat(zero(T))
  precT = precision(T)
  rng = Xoshiro(1004)
  @testset "Scaled values: overflow, underflow and exact zeros through the factor" begin
    p = E ÷ 2 + 20                                   # two factors 2^±p have a product 2^±(E+39) outside the range
    α1 = α2 = pow2(T, -p)
    β1 = β2 = pow2(T, p)
    # single factor, the leaf value overflows and the factor is below one: recovered from the value parts (SqrNormL2)
    f = SqrNormL2(one(T)); x = T[pow2(T, E ÷ 2 + 3), -pow2(T, E ÷ 2 + 2)]
    α = pow2(T, -(E - 3)) * T(1.3)
    @test f(x) === T(Inf)
    @test within(Scaled(f, α)(x), big(α) * sqval(1, nothing, x), 2)
    @test within((α * f)(x), big(α) * sqval(1, nothing, x), 2)          # α*f and f*α construct Scaled
    @test (f * α)(x) === (α * f)(x)
    # the factor above one and the leaf value zero or subnormal: restored from the parts, to the accuracy of the leaf
    for xe in (e0 ÷ 2 - 20, e0 ÷ 2 - 5, e0 ÷ 2 - 1)
      x = T[signif(rng, T, xe)]; α = pow2(T, E - 10) * T(1.7)
      @test abs(f(x)) < floatmin(T)
      r = big(α) * sqval(1, nothing, x)
      @test within(Scaled(f, α)(x), r, 4)
    end
    # an exactly zero value stays exactly zero (+0), however large the factor is; no recomputed rounding noise
    @test Scaled(f, pow2(T, E - 10))(zeros(T, 3)) === zero(T)
    @test Scaled(SqrNormL2(T(3), T[2, 2]), pow2(T, E - 10))(T[2, 2]) === zero(T)
    @test Scaled(SqrNormL2(zero(T)), pow2(T, E - 10))(T[1, 2]) === zero(T)
    # NormL1, single factor: the level absorbs the factor
    f1 = NormL1(pow2(T, E - 20)); x = T[pow2(T, 60), -pow2(T, 59)]
    @test f1(x) === T(Inf)
    α = pow2(T, -(E - 30)) * T(1.1)
    @test within(Scaled(f1, α)(x), big(α) * l1val(pow2(T, E - 20), nothing, x), 2)
    # nested factors whose product is not a normal number: the chain keeps the exponent sum
    s1 = Scaled(Scaled(f1, α1), α2)
    @test s1 isa Scaled{<:Scaled} && PB._chained(s1)
    @test !(floatmin(T) <= α1 * α2 <= floatmax(T))
    @test within(s1(x), big(α1) * big(α2) * l1val(pow2(T, E - 20), nothing, x), 1)           # fold into the level
    # (b) no level can hold the factor: the level is small and the weights are large (partial fold, window shift)
    fw = NormL1(pow2(T, -10), T[pow2(T, E - 5), pow2(T, E - 6)]); xw = T[pow2(T, 30) * T(1.3), -pow2(T, 29) * T(1.7)]
    @test fw(xw) === T(Inf)
    @test within(Scaled(Scaled(fw, α1), α2)(xw), big(α1) * big(α2) * l1val(pow2(T, -10), fw.w, xw), 2)
    # (c) the factor is above one and the leaf value underflows to zero: fold into the level gives the value
    fu = NormL1(pow2(T, -(E ÷ 2)), T[pow2(T, -(E - 5))]); xu = T[pow2(T, -(E ÷ 2)) * T(1.1)]
    @test fu(xu) === zero(T)
    @test within(Scaled(Scaled(fu, β1), β2)(xu), big(β1) * big(β2) * l1val(fu.λ0, fu.w, xu), 1)
    # (c2) the level cannot hold the factor (it overflows) and the leaf value is subnormal: the partial fold
    fv = NormL1(pow2(T, E ÷ 2), T[pow2(T, -(E - 5))]); xv = T[pow2(T, -(E ÷ 2) - 10) * T(1.9)]
    @test 0 < fv(xv) < floatmin(T)
    @test within(Scaled(Scaled(fv, β1), β2)(xv), big(β1) * big(β2) * l1val(fv.λ0, fv.w, xv), 2)
    # a result that is representable only as the product of the exponents: exact zero stays zero
    @test Scaled(Scaled(fv, β1), β2)(T[0]) === zero(T)
    @test Scaled(Scaled(fu, β1), β2)(T[-0.0]) === zero(T)
    # SqrNormL2 under a chain: overflow below one and underflow above one, from the retained exponent
    y = T[1, -2]; x = T[pow2(T, E - 3), pow2(T, E - 4)]
    fq = SqrNormL2(T(1.5), y)
    @test fq(x) === T(Inf)
    @test within(Scaled(Scaled(fq, α1), α2)(x), big(α1) * big(α2) * sqval(T(1.5), y, x), 2)
    xs = T[pow2(T, e0 ÷ 2 - 3), -pow2(T, e0 ÷ 2 - 4)] * T(1.3)
    @test within(Scaled(Scaled(SqrNormL2(T(1.5)), β1), β2)(xs), big(β1) * big(β2) * sqval(T(1.5), nothing, xs), 4)
    @test Scaled(Scaled(SqrNormL2(T(1.5)), β1), β2)(zeros(T, 2)) === zero(T)
    # a wrapper in a wrapper with an ordinary product of its factors (not merged): the parts of the inner one are used
    inner = Scaled(SqrNormL2(one(T)), pow2(T, 5))
    outer = Scaled{typeof(inner), T}(inner, pow2(T, -(E - 10)))
    xo = T[pow2(T, E ÷ 2 + 4)]
    @test inner(xo) === T(Inf) && !PB._chained(outer)
    @test within(outer(xo), big(pow2(T, -(E - 10))) * big(pow2(T, 5)) * sqval(1, nothing, xo), 4)
    # infinite factor: Inf*0 is 0 where the function is certainly zero (a norm at the zero vector), Inf elsewhere
    @test Scaled(NormL1(T(2)), T(Inf))(zeros(T, 2)) === zero(T)
    @test Scaled(NormL1(T(2)), T(Inf))(T[0, 1]) === T(Inf)
    @test Scaled(NormL1(T(2), T[0, 3]), T(Inf))(T[5, 0]) === zero(T)       # the unpenalized entry does not count
    @test Scaled(NormL1(T(2), T[1, 0]), T(Inf))(T[0, 5]) === zero(T)
    @test Scaled(NormL1(T(2), T[1, Inf]), T(Inf))(T[0, 5]) === T(Inf)
    # genuine NaN/Inf data under a factor
    @test isnan(Scaled(NormL1(T(2)), pow2(T, -E))(T[NaN, 1]))
    @test Scaled(NormL1(T(2)), pow2(T, -E))(T[Inf, 1]) === T(Inf)
    @test isnan(Scaled(Scaled(fu, β1), β2)(T[NaN]))
    @test Scaled(Scaled(f1, α1), α2)(T[Inf, 1]) === T(Inf)
    # (d) the partial fold evaluates at the end of its window and recentres a value near the top of the range (precT + 16 binades)
    pα = E - (T === Float64 ? 23 : 7)
    e1 = 2pα - E - 8
    fe = NormL1(pow2(T, -e1) * T(1.1))                                  # α λ0 > floatmax: no level holds the factor, and w = 1 is not scaled
    xe = T[pow2(T, e0 - 3 + e1) * T(1.3)]
    se = Scaled(Scaled(fe, pow2(T, pα)), pow2(T, pα))
    @test PB._chained(se) && !(floatmin(T) <= pow2(T, pα) * pow2(T, pα) <= floatmax(T))
    @test 0 < fe(xe) < floatmin(T)
    @test PB._level_window(fe, T, T) !== nothing
    @test within(se(xe), big(pow2(T, pα))^2 * l1val(fe.λ0, nothing, xe), 3)
    # an infinite level in a chain: no window, the product with the factors stays infinite; a non-IEEE level is read in its own type
    @test Scaled(Scaled(NormL1(T(Inf)), α1), α2)(T[1]) === T(Inf)
    @test PB._level_window(NormL1(T(Inf)), T, T) === nothing
    Wl = T === Float32 ? Float64 : BigFloat
    fbl = NormL1(Wl(2)^(T === Float32 ? 200 : 1500))
    @test within(Scaled(Scaled(fbl, α1), α2)(T[pow2(T, E ÷ 2)]), big(α1) * big(α2) * l1val(fbl.λ0, nothing, T[pow2(T, E ÷ 2)]), 3)
    # a rational factor is not an IEEE float: it folds as it is, below one, into the level
    @test Scaled(NormL1(pow2(T, E - 2)), 1 // 4)(T[8]) === pow2(T, E - 1)
    @test PB._value_fold_scale(1 // 4) === 1 // 4 && PB._partial_scale(1 // 4) === nothing
    @test PB._value_fold_scale(T(Inf)) === T(Inf)
    @test PB._partial_plan(T(0.5), T(Inf))[2] == -1 && PB._partial_plan(T(2), zero(T))[2] == 1 && PB._partial_plan(1 // 2, T(Inf)) == (nothing, 0)
  end
end

function test_scaled_oracles(::Type{T}) where {T<:AbstractFloat}
  E, e0 = emax(T), emin(T)
  δ = nextfloat(zero(T))
  precT = precision(T)
  rng = Xoshiro(1005)
  @testset "Scaled oracles on factor chains: gradient, prox, fused and block forms" begin
    rng = Xoshiro(11)
    p = E ÷ 2 + 20
    α1 = α2 = pow2(T, -p)
    # gradient of a chain: λ(x - y) times the product of the factors, formed with the exponent last
    y = T[1, -2, 0.5]; x = T[pow2(T, E - 5) * T(1.3), -pow2(T, E - 6) * T(1.1), 3]
    λ = T(1.5)
    s = Scaled(Scaled(SqrNormL2(λ, y), α1), α2)
    gref = big(α1) * big(α2) * big(λ) .* (big.(x) .- big.(y))
    g = gradient!(zeros(T, 3), s, x)
    @test all(within(g[i], gref[i], 3) for i in 1:3)
    g2 = zeros(T, 3)
    v = value_and_gradient!(g2, s, x)
    @test g2 == g                                                             # the fused gradient is the standalone one
    @test v === s(x)                                                          # and its value is the standalone value
    @test within(v, big(α1) * big(α2) * sqval(λ, y, x), 2)
    # a symmetric owner: the stored coordinates are scaled once each
    xo = SymmetricSparseIterate(T, 3); xo[2, 1] = x[1]; xo[3, 3] = x[2]
    go = SymmetricSparseIterate(T, 3)
    so = Scaled(Scaled(SqrNormL2(λ), α1), α2)
    gradient!(go, so, xo)
    xm = Matrix(xo)
    @test all(within(go[r, c], big(α1) * big(α2) * big(λ) * big(xm[r, c]), 3) for r in 1:3, c in 1:3)
    @test within(so(xo), big(α1) * big(α2) * sqval(λ, nothing, xm), 2)       # the owner counts the off-diagonal entry twice
    # a single factor, an ordinary Scaled, scales the gradient and the value as plain products
    s1 = Scaled(SqrNormL2(λ, y), T(3))
    @test gradient!(zeros(T, 3), s1, x) == T(3) .* gradient!(zeros(T, 3), SqrNormL2(λ, y), x)
    # the prox of a chain with the SqrNormL2 leaf: the step is the product αγ, formed with the exponent last
    xs = T[3, 5, -2]
    γ = pow2(T, E - 5)
    τ = big(α1) * big(α2) * big(γ) * big(λ)
    zref = (big.(xs) .+ τ .* big.(y)) ./ (1 + τ)
    z = prox!(zeros(T, 3), s, xs, γ)
    @test all(within(z[i], zref[i], 4) for i in 1:3)
    @test z == T[prox_block(s, xs[i], i, γ) for i in 1:3]                      # prox_block takes the same route
    xa = copy(xs)
    @test prox!(xa, s, xa, γ) == z                                              # exact aliasing
    yv = zeros(T, 3)
    vv = value_and_prox!(yv, s, xs, γ)
    @test yv == z && vv === s(yv)                                              # the value is that at the stored point
    @test within(vv, big(α1) * big(α2) * sqval(λ, y, z), 4)
    # the prox of a chain with a NormL1 leaf whose level absorbs the factors
    f1 = NormL1(pow2(T, E - 20)); sl = Scaled(Scaled(f1, α1), α2)
    γ = pow2(T, -30)
    t = big(α1) * big(α2) * big(γ) * big(pow2(T, E - 20))
    xl = T.(big.(T[3, -0.5, 7]) .* t)
    zl = prox!(zeros(T, 3), sl, xl, γ)
    @test all(within(zl[i], shrinkb(big(xl[i]), t), 2) for i in 1:3)
    @test zl == T[prox_block(sl, xl[i], i, γ) for i in 1:3]
    yl = zeros(T, 3)
    @test value_and_prox!(yl, sl, xl, γ) |> v -> within(v, big(α1) * big(α2) * l1val(pow2(T, E - 20), nothing, yl), 2)
    @test yl == zl
    # a single factor and a step whose product leaves the range: the chain route forms the coefficient from both
    for (αf, γf, λf) in ((pow2(T, E - 10), pow2(T, 30), pow2(T, -(E - 20))), (pow2(T, -(E - 10)), pow2(T, -30), pow2(T, E - 20)))
      sf = Scaled(NormL1(λf), αf)
      @test !(floatmin(T) <= αf * γf <= floatmax(T))
      tf = big(αf) * big(γf) * big(λf)
      xf = T.(big.(T[2, -3.5, 0.75]) .* tf)
      zf = prox!(zeros(T, 3), sf, xf, γf)
      @test all(within(zf[i], shrinkb(big(xf[i]), tf), 2) for i in 1:3)
      @test zf == T[prox_block(sf, xf[i], i, γf) for i in 1:3]
      yf = zeros(T, 3)
      @test within(value_and_prox!(yf, sf, xf, γf), big(αf) * l1val(λf, nothing, yf), 2) && yf == zf
    end
    # a function with an infinite level is fixed at zero whatever the factors and the step are
    sh = Scaled(Scaled(NormL1(T(Inf)), α1), α2)
    @test prox!(zeros(T, 2), sh, T[3, -4], T(1)) == T[0, 0]
    @test value_and_prox!(zeros(T, 2), sh, T[3, -4], T(1)) === zero(T)
    @test prox!(zeros(T, 2), Scaled(NormL1(T(Inf), T[1, 0]), T(7)), T[3, -4], T(2)) == T[0, -4]
    # the lipschitz constant and the kkt violation of a chain, with the exponent last
    @test lipschitz(Scaled(Scaled(SqrNormL2(λ), α1), α2)) == PB._apply_scale(PB._scaled_chain(Scaled(Scaled(SqrNormL2(λ), α1), α2))[2], λ)
    @test within(lipschitz(Scaled(Scaled(SqrNormL2(pow2(T, E - 5)), pow2(T, -p)), pow2(T, -p))), big(pow2(T, E - 5)) * big(α1) * big(α2), 2)
    cl = big(α1) * big(α2) * big(pow2(T, E - 20))                   # the folded level of the chain: kkt |g| - level
    gk = T.(big.(T[4, -5, 1.5]) .* cl)
    skk = Scaled(Scaled(f1, α1), α2)
    @test all(within(kkt_violation(skk, gk, i), abs(big(gk[i])) - cl, 2) for i in 1:3)
    # no level folds: the violation of the function at g / (the factors), times the factors
    fk = NormL1(pow2(T, -10), T[pow2(T, E - 5), pow2(T, E - 6), pow2(T, E - 7)])
    ck = big(α1) * big(α2) * big(pow2(T, -10)) .* big.(fk.w)
    gk = T.(big.(T[4, -5, 1.5]) .* ck)
    skk = Scaled(Scaled(fk, α1), α2)
    @test all(within(kkt_violation(skk, gk, i), abs(big(gk[i])) - ck[i], 4) for i in 1:3)
    # the same through the block-coordinate form on an array and on a block iterate
    @test all(within(PB._kkt_violation(skk, gk, zeros(T, 3), i), abs(big(gk[i])) - ck[i], 4) for i in 1:3)
  end
end

function test_scaled_array_steps(::Type{T}) where {T<:AbstractFloat}
  E, e0 = emax(T), emin(T)
  δ = nextfloat(zero(T))
  precT = precision(T)
  rng = Xoshiro(1006)
  @testset "Scaled array steps: folded, shifted and rounded routes" begin
    q = E ÷ 4 + 30                                    # a factor outside the band 2^±(E÷4) takes the array route
    x = T[3, -5, 0.25]; γ = T[1, 2, 4]
    # folded: the level holds the whole factor
    α = pow2(T, q); λ0 = pow2(T, -q) * T(1.3)
    f = NormL1(λ0); s = Scaled(f, α)
    @test PB._array_route(s, x, γ)
    @test PB._array_fold(f, PB._scale_start(α), γ)[3] === :folded
    t = big(α) * big(λ0) .* big.(γ)
    z = prox!(zeros(T, 3), s, x, γ)
    @test all(within(z[i], shrinkb(big(x[i]), t[i]), 2) for i in 1:3)
    y = zeros(T, 3)
    @test within(value_and_prox!(y, s, x, γ), big(α) * l1val(λ0, nothing, y), 2) && y == z
    # a factor inside the band is a plain product with each step (no chain)
    sm = Scaled(NormL1(T(0.5)), T(3))
    @test !PB._array_route(sm, x, γ)
    @test prox!(zeros(T, 3), sm, x, γ) == T[prox_block(sm, x[i], i, γ[i]) for i in 1:3]
    # shifted: the level cannot hold the factor (it overflows), a power of two moves from the steps to the level
    λ0 = pow2(T, E - q + 10) * T(1.3)
    f = NormL1(λ0); s = Scaled(f, α)
    γs = T[pow2(T, -q - 5), pow2(T, -q - 6), pow2(T, -q - 7)] * T(1.1)
    @test PB._array_fold(f, PB._scale_start(α), γs)[3] === :shifted
    t = big(α) * big(λ0) .* big.(γs)
    xs = T.(big.(T[3, -5, 0.25]) .* t)
    z = prox!(zeros(T, 3), s, xs, γs)
    @test all(within(z[i], shrinkb(big(xs[i]), t[i]), 2) for i in 1:3)
    y = zeros(T, 3)
    v = value_and_prox!(y, s, xs, γs)
    @test y == z && within(v, big(α) * l1val(λ0, nothing, y), 2)                 # the value of the chain, not of the shifted function
    # rounded: a function without levels to fold takes each step multiplied by the chain with the exponent last
    pp = E ÷ 2 + 20; a1 = a2 = pow2(T, -pp)
    w = T[1, 2, 3] .* pow2(T, E ÷ 2)
    fr = WeightedSquare(w)
    sr = Scaled(Scaled(fr, a1), a2)
    γr = T[pow2(T, E - 1) * T(1.1), pow2(T, E - 2) * T(1.3), pow2(T, E - 3) * T(1.7)]
    @test PB._array_fold(fr, PB._scaled_chain(sr)[2], γr)[3] === :rounded
    xr = T[3, -5, 0.25]
    τ = big(a1) * big(a2) .* big.(γr)
    zr = prox!(zeros(T, 3), sr, xr, γr)
    @test all(within(zr[i], big(xr[i]) / (1 + τ[i] * big(w[i])), 4) for i in 1:3)
    yr = zeros(T, 3)
    vr = value_and_prox!(yr, sr, xr, γr)
    @test yr == zr
    @test within(vr, big(a1) * big(a2) * sum(big.(w) .* big.(yr) .^ 2) / 2, 4)
    # a stored infinite level fixes every entry, the steps are the caller's array
    @test PB._array_fold(NormL1(T(Inf)), PB._scale_start(α), γ)[3] === :folded
    @test prox!(zeros(T, 3), Scaled(NormL1(T(Inf)), α), x, γ) == zeros(T, 3)
    # an empty step array has nothing to shift; no positive finite step has no shift either
    @test PB._shifted_fold(NormL1(T(2)), PB._scale_start(α), T[]) === nothing
    @test PB._shifted_fold(NormL1(T(2)), PB._scale_start(α), T[0, Inf, -1]) === nothing
    # a chain with an array step
    pp = E ÷ 2 + 20; a1 = a2 = pow2(T, -pp)
    sc = Scaled(Scaled(NormL1(pow2(T, E - 20)), a1), a2)
    tc = big(a1) * big(a2) * big(pow2(T, E - 20)) .* big.(γ)
    xc = T.(big.(T[3, -5, 0.25]) .* tc)
    zc = prox!(zeros(T, 3), sc, xc, γ)
    @test all(within(zc[i], shrinkb(big(xc[i]), tc[i]), 2) for i in 1:3)
    # separable and zero functions under an array step; the steps must match
    @test prox!(zeros(T, 3), ZeroFunction(), x, γ) == x
    @test value_and_prox!(zeros(T, 3), ZeroFunction(), x, γ) === zero(T)
    @test_throws DimensionMismatch prox!(zeros(T, 3), NormL1(T(1)), x, T[1, 2])
  end
end

function test_normL1_values(::Type{T}) where {T<:AbstractFloat}
  E, e0 = emax(T), emin(T)
  δ = nextfloat(zero(T))
  precT = precision(T)
  rng = Xoshiro(1007)
  @testset "NormL1 values: levels and weights outside the range, infinite and zero parameters" begin
    rng = Xoshiro(21)
    # weighted dense values: the level times the weight overflows or underflows, the final value is representable
    for (λ0, w, xe) in ((pow2(T, E - 10), T[pow2(T, 30), pow2(T, 29)], -100), (pow2(T, -(E - 10)), T[pow2(T, -30), pow2(T, -31)], 100))
      x = T[signif(rng, T, xe), signif(rng, T, xe - 1)]
      f = NormL1(λ0 * T(1.3), w)
      @test !(floatmin(T) <= f.λ0 * w[1] <= floatmax(T))
      @test within(f(x), l1val(f.λ0, w, x), 2)
      xsp = SparseIterate(T, 2); xsp[1] = x[1]; xsp[2] = x[2]
      @test f(xsp) === f(x)                                  # the stored entries take the same value entries
    end
    # the value entry and the positive product, where the factors leave the range and the final value is representable
    @test value_entry(NormL1(pow2(T, E - 10), T[pow2(T, 30)]), T(-3), 1) === T(Inf)          # 3 * 2^(E + 20): out of range, Inf
    @test within(value_entry(NormL1(pow2(T, E - 10), T[pow2(T, 30)]), pow2(T, -100) * T(-1.5), 1), big(2.0)^(E - 80) * 1.5, 1)
    @test within(PB._positive_product(pow2(T, E - 10), pow2(T, 30), pow2(T, -100)), big(2.0)^(E - 80), 1)
    # an infinite weight gives Inf away from zero and the zero entry is free; a zero weight is unpenalized at any entry
    fw = NormL1(T(2), T[Inf, 1, 0])
    @test fw(T[0, 3, Inf]) === T(6)
    @test fw(T[1, 3, 5]) === T(Inf)
    @test NormL1(T(0), T[Inf, 1])(T[1, 3]) === T(Inf) && NormL1(T(0), T[Inf, 1])(T[0, 3]) === zero(T)    # an infinite weight beats a zero level
    @test NormL1(T(Inf), T[0, 1])(T[5, 0]) === zero(T)
    # genuine NaN and Inf entries; the unweighted value, the zero level and the infinite level
    @test isnan(NormL1(T(2))(T[NaN, 1])) && NormL1(T(2))(T[Inf, -1]) === T(Inf)
    @test NormL1(zero(T))(T[NaN, Inf]) === zero(T)                       # the documented zero level
    @test NormL1(T(Inf))(zeros(T, 3)) === zero(T) && NormL1(T(Inf))(T[0, 1]) === T(Inf)
    @test isnan(NormL1(T(Inf))(T[NaN, 0]))
    @test isnan(NormL1(T(2), T[1, 1])(T[NaN, 1])) && NormL1(T(2), T[1, 1])(T[Inf, 1]) === T(Inf)
    @test NormL1(T(2))(T[-0.0, 0.0]) === zero(T)                          # signed zeros: |-0| = +0
    # a level outside the argument's precision is applied before the final conversion (a wider level)
    W = T === Float32 ? Float64 : BigFloat
    for (λ, xe) in ((W(2)^(T === Float32 ? 200 : 1500), -(T === Float32 ? 100 : 1000)), (W(2)^-(T === Float32 ? 200 : 1500), T === Float32 ? 100 : 1000))
      x = T[signif(rng, T, xe), signif(rng, T, xe - 1), 0, signif(rng, T, xe + 1)]
      @test !PB._level_fits(T, λ)
      @test within(NormL1(λ)(x), big(λ) * sum(abs, big.(x)), 2)
      @test within(PB._converted_l1_value(λ, x), big(λ) * sum(abs, big.(x)), 2)
    end
    @test NormL1(W(2)^(T === Float32 ? 200 : 1500))(T[Inf, 1]) === T(Inf)
    @test isnan(NormL1(W(2)^(T === Float32 ? 200 : 1500))(T[NaN, 1]))
    @test NormL1(W(2)^(T === Float32 ? 200 : 1500))(zeros(T, 2)) === zero(T)
    # the scaled sum with its compensated accumulation, and the dot of a coefficient array with scaled entries
    xs = T[1, -2, 3] .* pow2(T, 40)
    sf = PB._power_factors(T, -40)
    @test PB._scaled_value_sum(xs, sf, abs) == T(6)
    @test PB._scaled_value_sum(T[1, 1e-8, -1], PB._power_factors(T, 0)) ≈ T(1e-8) rtol = 4 * eps(T)         # the small term survives the cancellation
    @test PB._scaled_dot(T[1, 2, 3], T[4, 5, 6] .* pow2(T, 40), sf) == T(32)
    @test PB._scaled_dot(2, T[4, 5, 6], PB._power_factors(T, 0)) == T(30)                    # an integer coefficient
    @test_throws DimensionMismatch PB._scaled_dot(T[1, 2], T[1, 2, 3], sf)
  end
end

function test_normL1_prox(::Type{T}) where {T<:AbstractFloat}
  E, e0 = emax(T), emin(T)
  δ = nextfloat(zero(T))
  precT = precision(T)
  rng = Xoshiro(1008)
  @testset "NormL1 prox thresholds and the weight kernels" begin
    rng = Xoshiro(23)
    # thresholds γ λ w whose partial products leave the range, formed once from the bounded factors
    for (γ, λ0, w) in ((pow2(T, E - 10), pow2(T, 20), pow2(T, -40)), (pow2(T, -(E - 10)), pow2(T, -20), pow2(T, 40)),
                       (pow2(T, E - 5), pow2(T, E - 5), pow2(T, -(2E - 20))), (pow2(T, 3), pow2(T, -(E - 4)), pow2(T, 0)))
      γ, λ0 = γ * T(1.3), λ0 * T(1.7)
      t = big(γ) * big(λ0) * big(w)
      (isfinite(T(t)) && T(t) >= floatmin(T) || T(t) < floatmin(T)) || continue
      f = NormL1(λ0, T[w, w])
      x = T[T(t) * 4, -T(t) * 2.5]
      z = prox!(zeros(T, 2), f, x, γ)
      @test all(within(z[i], shrinkb(big(x[i]), t), 2) for i in 1:2)
      @test z == T[prox_block(f, x[i], i, γ) for i in 1:2]
      xa = copy(x)
      @test prox!(xa, f, xa, γ) == z
      y = zeros(T, 2)
      v = value_and_prox!(y, f, x, γ)
      @test y == z && v === f(y)
    end
    # the threshold itself, rounded once to the destination type from the unrounded factors
    @test within(PB._weight_product(T, pow2(T, E - 10) * T(1.3), pow2(T, 20) * T(1.7), pow2(T, -40)), big(pow2(T, E - 10) * T(1.3)) * big(pow2(T, 20) * T(1.7)) * big(pow2(T, -40)), 2)
    @test PB._weight_product(T(2), T(3), T(Inf)) === T(Inf)
    @test PB._weight_product(T(0), T(3), T(Inf)) === T(Inf)                    # an infinite weight even at a zero step
    @test PB._weight_product(T(0), T(3), T(2)) === zero(T)
    @test PB._weight_product(T, T(2), T(3), true) === T(6)                        # the weight of an unweighted entry
    @test PB._weight_product(Float32, 2.0, 3.0, 5.0) === 30.0f0
    @test PB._weight_product(Float32, 2.0^100, 2.0^100, 2.0^-190) === Float32(2.0^10)    # formed in Float64, rounded once
    # zero, infinite and NaN steps, levels and weights
    @test prox(NormL1(zero(T), T[Inf]), T[3], T(1)) == T[0]                    # an infinite weight thresholds a finite trial to zero at zero level
    @test prox(NormL1(T(Inf), T[0]), T[3], T(1)) == T[3]                       # a zero weight is unpenalized at every level
    @test prox(NormL1(T(Inf)), T[NaN, Inf, 3], T(1)) == T[0, 0, 0]             # an infinite level at a positive step fixes every entry
    @test isnan(prox(NormL1(T(1), T[Inf]), T[NaN], T(1))[1])
    @test prox(NormL1(T(2), T[Inf]), T[3], zero(T)) == T[0]
    @test prox(NormL1(T(2)), T[3, -4], zero(T)) == T[3, -4]                    # a zero step is the identity
    @test prox(NormL1(zero(T)), T[3, -4], T(5)) == T[3, -4]
    @test isnan(prox(NormL1(T(1)), T[NaN], T(1))[1]) && prox(NormL1(T(1)), T[Inf, -Inf], T(1)) == T[Inf, -Inf]
    # signed zeros: shrink(±0) is a zero of either sign by the sum of the two branches; the value is +0
    zs = prox(NormL1(T(1)), T[0, -0.0], T(1))
    @test iszero(zs[1]) && iszero(zs[2])
    # sparse iterates: the stored entries only, the same thresholds
    f = NormL1(pow2(T, 20) * T(1.7), T[pow2(T, -40), pow2(T, -40), pow2(T, -40)])
    γ = pow2(T, E - 10) * T(1.3)
    t = big(γ) * big(f.λ0) * big(pow2(T, -40))
    xsp = SparseIterate(T, 3); xsp[1] = T(t) * 4; xsp[3] = -T(t) * 2.5
    ysp = SparseIterate(T, 3)
    prox!(ysp, f, xsp, γ)
    @test within(ysp[1], shrinkb(big(xsp[1]), t), 2) && ysp[2] === zero(T) && within(ysp[3], shrinkb(big(xsp[3]), t), 2)
    @test ysp == prox!(zeros(T, 3), f, Vector(xsp), γ)
    # the coefficients of the kkt violations: zero level against an infinite weight, and the plain products
    @test PB._weight_coefficient(zero(T), T(Inf)) === T(Inf) && PB._weight_coefficient(zero(T), T(2)) === zero(T)
    @test PB._weight_coefficient(T(Inf), zero(T)) === zero(T) && isnan(PB._weight_coefficient(T(NaN), zero(T)))
    @test PB._weight_coefficient(-T(1), T(2)) === -T(2) && PB._weight_coefficient(T(2), T(3)) === T(6)
    @test PB._kkt_coefficient(T, 2.0, 3.0) === T(6) && PB._kkt_coefficient(Int, 2, 3) == 6
    @test kkt_violation(NormL1(zero(T), T[Inf]), T[3], 1) === -T(Inf)
    @test kkt_violation(NormL1(T(2), T[3]), T[7], 1) === T(1)
    @test kkt_violation(NormL1(T(2), T[0]), T[-7], 1) === T(7)
    @test within(kkt_violation(NormL1(pow2(T, E - 10) * T(1.3), T[4]), T[-pow2(T, E - 3)], 1), big(pow2(T, E - 3)) - big(pow2(T, E - 10) * T(1.3)) * 4, 1)
    @test kkt_violation(NormL1(pow2(T, E - 3) * T(1.3), T[16]), T[-pow2(T, E - 3)], 1) === -T(Inf)      # the coefficient itself is out of range
    # a zero or NaN value times an infinite or zero coefficient
    @test PB._weighted_value(T(NaN), T(2)) === zero(T) && isnan(PB._weighted_value(T(NaN), T(NaN))) && PB._weighted_value(T(3), T(2)) === T(3)
  end
end

function test_pow2_sums(::Type{T}) where {T<:AbstractFloat}
  E, e0 = emax(T), emin(T)
  δ = nextfloat(zero(T))
  precT = precision(T)
  rng = Xoshiro(1009)
  @testset "power-of-two sums (_pow2_weighted) and the recomputed-dot shift" begin
    rng = Xoshiro(31)
    # w Σ mᵢ 2^eᵢ over terms in different units, added from the largest exponent down so that terms that
    # cancel do so before smaller ones are aligned; one rounding at the end
    function ref_terms(w, terms)
      big(w) * sum(big(t[1]) * big(2.0)^t[2] for t in terms; init = big(0))
    end
    for N in (1, 2, 3, 4), _ in 1:200
      terms = ntuple(_ -> (signif(rng, T, rand(rng, -3:3)), rand(rng, -E:E)), N)
      w = abs(signif(rng, T, rand(rng, -E ÷ 2:E ÷ 2)))
      r = ref_terms(w, terms)
      abs(r) > floatmax(T) || abs(r) < nextfloat(zero(T)) && continue
      @test within(PB._pow2_weighted(T, w, terms...), r, 3)
      @test within(PB._pow2_sum(T, terms...), ref_terms(1, terms), 3)
    end
    # exact cancellation of the large terms before a small one is aligned: no digit of the remainder is lost
    big_ = signif(rng, T, 0)
    terms = ((big_, E - 1), (T(0.75), -100), (-big_, E - 1))
    @test PB._pow2_sum(T, terms...) === T(0.75) * pow2(T, -100)
    terms = ((T(0.75), 100), (-T(0.75), 100), (T(0.625), -50))
    @test PB._pow2_sum(T, terms...) === T(0.625) * pow2(T, -50)
    # an exact zero sum is +0 times the weight; zero terms are skipped; a lone term is scaled
    @test PB._pow2_sum(T, (T(0.75), 5), (-T(0.75), 5)) === zero(T)
    @test PB._pow2_sum(T, (zero(T), 5), (T(0.75), 3)) === T(6)
    @test PB._pow2_weighted(T, T(2), (T(0.75), 5), (-T(0.75), 5)) === zero(T)
    @test PB._pow2_weighted(T, T(2), (T(0.5), 1)) === T(2)
    # the result leaves the range only at the final conversion
    @test PB._pow2_sum(T, (T(0.5), E + 10)) === T(Inf) && PB._pow2_sum(T, (-T(0.5), E + 10)) === -T(Inf)
    @test PB._pow2_sum(T, (T(0.5), e0 - precT - 5)) === zero(T)
    @test within(PB._pow2_sum(T, (T(0.75), e0 - 2)), big(0.75) * big(2.0)^(e0 - 2), 1)           # a subnormal result, rounded once
    # genuine Inf and NaN terms give the ordinary sum
    @test PB._pow2_sum(T, (T(Inf), 5), (T(0.5), 2)) === T(Inf)
    @test isnan(PB._pow2_sum(T, (T(Inf), 5), (-T(Inf), 2)))
    @test isnan(PB._pow2_weighted(T, T(3), (T(NaN), 5), (T(0.5), 2)))
    # BigFloat terms whose exponents leave Int in the frame arithmetic: the large terms cancel, the small one stays
    Bs = PB._pow2_sum(BigFloat, (big(0.75), 2^61), (big(0.625), -2^40), (-big(0.75), 2^61))
    @test Bs == Base.ldexp(big(0.625), -2^40)
    @test PB._pow2_sum(BigFloat, (big(0.75), 2^61), (big(0.5), 2^61)) == Base.ldexp(big(1.25), 2^61)
    # the units of a recomputed dot: bound exponent, ceil(log2 n), and the shift
    @test PB._bound_exponent(T(3)) == 2 && PB._bound_exponent(zero(T)) == 0 && PB._bound_exponent(T(Inf)) == 0 && PB._bound_exponent(T(NaN)) == 0
    @test PB._log2_ceil(1) == 0 && PB._log2_ceil(2) == 1 && PB._log2_ceil(3) == 2 && PB._log2_ceil(1024) == 10 && PB._log2_ceil(1025) == 11
    @test PB._max_exponent(T) == E
    @test PB._product_sum_shift(T, pow2(T, E - 1), pow2(T, E - 1), 4, 2) == (E) + (E) + 2 + 2 - E
    @test PB._product_sum_shift(T, zero(T), pow2(T, E - 1), 4, 2) == max(0, 0 + E + 2 + 2 - E)
  end
end

function test_square_consumers(::Type{T}) where {T<:AbstractFloat}
  E, e0 = emax(T), emin(T)
  δ = nextfloat(zero(T))
  precT = precision(T)
  rng = Xoshiro(1010)
  @testset "squared-residual consumers: the exceptional branches of the value" begin
    rng = Xoshiro(33)
    W = T
    # a level that is infinite keeps the ordinary value; a zero level the zero value, also at non-finite entries
    @test SqrNormL2(T(Inf))(T[1, 2]) === T(Inf)
    @test isnan(SqrNormL2(T(Inf))(T[0, 0]))                                  # Inf * 0 at the exact zero vector
    @test SqrNormL2(zero(T))(T[NaN, Inf]) === zero(T)
    @test isnan(PB._square_value(zero(T), T[NaN, 1], nothing))                 # the primitive keeps the ordinary 0 * NaN; the oracle returns zero above
    @test PB._square_value(zero(T), T[2, 1], nothing) === zero(T)
    # a level below floatmin(T): the scaled recomputation applies it to the squares, with the rounding of one product
    λs = pow2(T, e0 - 10)
    x = T[pow2(T, 40), -pow2(T, 41) * T(1.3)]
    @test within(PB._square_value(λs, x, nothing), big(λs) * sum(abs2, big.(x)) / 2, 2)
    @test within(SqrNormL2(λs)(x), big(λs) * sum(abs2, big.(x)) / 2, 2)
    # an exact zero (x == 0) with a subnormal level, and an exact fit against a center
    @test SqrNormL2(λs)(zeros(T, 3)) === zero(T)
    @test SqrNormL2(λs, T[1, 2])(T[1, 2]) === zero(T)
    @test SqrNormL2(T(3), T[pow2(T, E - 3), 2])(T[pow2(T, E - 3), 2]) === zero(T)
    # integer arguments and centers go through the widened integer difference
    @test within(SqrNormL2(T(2), [typemin(Int)])([typemax(Int)]), big(2) * (big(typemax(Int)) - big(typemin(Int)))^2 / 2, 2)
    @test SqrNormL2(T(2))([3, 4]) == 25
    # block iterates reduce over their storage
    bx = BlockIterate(T[pow2(T, E - 3), 1, pow2(T, E - 4)], [[1, 2], [3]])
    @test SqrNormL2(T(1))(bx) === T(Inf)
    @test within(SqrNormL2(pow2(T, -E + 5))(bx), big(pow2(T, -E + 5)) * sum(abs2, big.(bx.storage)) / 2, 2)
  end
end

function test_sqrnorml2_oracles(::Type{T}) where {T<:AbstractFloat}
  E, e0 = emax(T), emin(T)
  δ = nextfloat(zero(T))
  precT = precision(T)
  rng = Xoshiro(1011)
  @testset "SqrNormL2 oracles: ridge weights, owners, wide levels and centers" begin
    rng = Xoshiro(41)
    ridge(x, c, t) = (big(x) + t * (c === nothing ? big(0) : big(c))) / (1 + t)
    # the prox weights a = 1/(1+t), b = t/(1+t) for t = γλ below, in and above the ordinary band (the scaled weights
    # carry the smaller one as a significand and an exponent): dense, coordinate, owner and aliased forms
    p = 3
    xo = SymmetricSparseIterate(T, p); xo[2, 1] = pow2(T, E - 3) * T(1.3); xo[3, 3] = -pow2(T, E - 4) * T(1.7); xo[1, 1] = T(3)
    xd = Matrix(xo)
    co = T[1 2 3; 2 -5 7; 3 7 pow2(T, E - 2)]                                    # a symmetric center
    for (γ, λ) in ((pow2(T, E ÷ 2 + 20), pow2(T, E ÷ 2 + 20)), (T(0.5), T(3)), (pow2(T, -E ÷ 2 - 30), pow2(T, -E ÷ 2 - 30)),
                    (pow2(T, E - 1), pow2(T, -E + 3)), (pow2(T, -E + 3), pow2(T, E - 1)))
      t = big(γ) * big(λ)
      for y in (nothing, co)
        f = SqrNormL2(λ, y)
        # the dense output reads the matrix of the owner
        zd = prox!(zeros(T, p, p), f, xo, γ)
        @test all(within(zd[r, c], ridge(xd[r, c], y === nothing ? nothing : y[r, c], t), 4) for r in 1:p, c in 1:p)
        # an owner output holds each coordinate once, the lower triangle
        zo = SymmetricSparseIterate(T, p)
        prox!(zo, f, xo, γ)
        @test Matrix(zo) == zd
        # the in-place forms, with exact aliasing of owner and of dense array
        xa = copy(xo)
        prox!(xa, f, xa, γ)
        @test Matrix(xa) == zd
        xb = copy(xd)
        prox!(xb, f, xb, γ)
        @test xb == zd
        # the coordinate form takes the same route
        xv = vec(xd)
        yv = y === nothing ? nothing : vec(y)
        fv = SqrNormL2(λ, yv)
        @test T[prox_block(fv, xv[i], i, γ) for i in eachindex(xv)] == prox!(zeros(T, p * p), fv, xv, γ)
        # the fused oracle returns the value at the stored point
        zf = zeros(T, p, p)
        @test value_and_prox!(zf, f, xd, γ) === f(zf) && zf == zd
      end
    end
    # a Float64 level with a Float32 model, and a BigFloat level with a Float64 model (outside the range of T): the level is
    # applied before the final rounding, for the dense and owner gradient and for the value
    W = T === Float32 ? Float64 : BigFloat
    λw = W(2)^(T === Float32 ? 200 : 1500)
    xe = -(T === Float32 ? 100 : 1000)
    xs = T[signif(rng, T, xe), signif(rng, T, xe + 1), 0]
    @test !PB._level_fits(T, λw)
    gd = gradient!(zeros(T, 3), SqrNormL2(λw), xs)
    @test all(within(gd[i], big(λw) * big(xs[i]), 2) for i in 1:3)
    # the entry that overflows in the model precision is Inf; the others are exact
    xw = T[signif(rng, T, xe), 1, -2]
    gw = gradient!(zeros(T, 3), SqrNormL2(λw), xw)
    @test within(gw[1], big(λw) * big(xw[1]), 2) && gw[2] === T(Inf) && gw[3] === -T(Inf)
    xo2 = SymmetricSparseIterate(T, 2); xo2[2, 1] = signif(rng, T, xe); xo2[1, 1] = signif(rng, T, xe + 1)
    go = SymmetricSparseIterate(T, 2)
    gradient!(go, SqrNormL2(λw), xo2)
    @test all(within(go[r, c], big(λw) * big(Matrix(xo2)[r, c]), 2) for r in 1:2, c in 1:2)
    @test Matrix(go) == gradient!(zeros(T, 2, 2), SqrNormL2(λw), xo2)
    @test within(SqrNormL2(λw)(xs), big(λw) * sum(abs2, big.(xs)) / 2, 2)
    # a center wider than the model precision and outside its range: the entry is computed in the wider type
    Wc = T === Float32 ? Float64 : BigFloat
    cw = Wc[Wc(2)^(T === Float32 ? 200 : 1100), 3, -Wc(2)^-(T === Float32 ? 200 : 1100)]
    xv = T[1, 2, 3]
    for (λf, γf) in ((T === Float32 ? 2.0^-100 : 2.0^-80, T(1)), (T(1), T(1)))
      f = SqrNormL2(λf, cw)
      t = big(γf) * big(λf)
      z = prox!(zeros(T, 3), f, xv, γf)
      @test all(within(z[i], ridge(xv[i], cw[i], t), 4) for i in 1:3)
      @test z == T[prox_block(f, xv[i], i, γf) for i in 1:3]
      gc = gradient!(zeros(T, 3), f, xv)
      @test all(within(gc[i], big(λf) * (big(xv[i]) - big(cw[i])), 3) for i in 1:3)
    end
    # a Float64 center entry out of the Float32 range: the value reads it unrounded where rounding leaves the range
    if T === Float32
      cc = Float64[2.0^200, 3.0, 2.0^-200]
      fc = SqrNormL2(2.0^-300, cc)
      @test within(fc(T[1, 2, 3]), big(2.0)^-300 * sum(abs2, big.(T[1, 2, 3]) .- big.(cc)) / 2, 4)
    end
    # gradient on a sparse owner and on a block iterate argument, in place of the dense one
    bx = BlockIterate(T[pow2(T, E - 3), 1, pow2(T, E - 4)], [[1, 2], [3]])
    @test gradient!(zeros(T, 3), SqrNormL2(pow2(T, -E + 6)), bx) == gradient!(zeros(T, 3), SqrNormL2(pow2(T, -E + 6)), bx.storage)
    # the level that is exactly zero and a center equal to the argument
    @test gradient!(zeros(T, 2), SqrNormL2(zero(T)), T[3, 4]) == T[0, 0]
    @test prox!(zeros(T, 2), SqrNormL2(zero(T)), T[3, 4], T(7)) == T[3, 4]
    @test prox!(zeros(T, 2), SqrNormL2(T(2), T[3, 4]), T[3, 4], T(7)) == T[3, 4]
    # infinite level and positive step: the prox is the center (weight 1 on the center); the gradient is IEEE
    @test prox!(zeros(T, 2), SqrNormL2(T(Inf), T[3, 4]), T[1, 2], T(7)) == T[3, 4]
    # an owner output with a center wider than the model and outside its range
    Wco = T === Float32 ? Float64 : BigFloat
    cbig = Matrix{Wco}([Wco(2)^(T === Float32 ? 200 : 1100) 3; 3 -Wco(2)^-(T === Float32 ? 200 : 1100)])
    xob = SymmetricSparseIterate(T, 2); xob[1, 1] = T(1); xob[2, 1] = T(2); xob[2, 2] = T(3)
    λb = T === Float32 ? 2.0^-100 : 2.0^-80
    zob = SymmetricSparseIterate(T, 2)
    prox!(zob, SqrNormL2(λb, cbig), xob, T(1))
    tb = big(λb)
    xbm = Matrix(xob)
    @test all(within(zob[r, c], ridge(xbm[r, c], cbig[r, c], tb), 4) for r in 1:2, c in 1:2)
    @test Matrix(zob) == prox!(zeros(T, 2, 2), SqrNormL2(λb, cbig), xbm, T(1))
  end
end

function test_leastsquares(::Type{T}) where {T<:AbstractFloat}
  E, e0 = emax(T), emin(T)
  δ = nextfloat(zero(T))
  precT = precision(T)
  rng = Xoshiro(1012)
  @testset "LeastSquaresLoss: SVD prox, multi-response data and iterate arguments" begin
    rng = Xoshiro(51)
    # (X'X + I/γ) β = X'Y + v/γ, solved in BigFloat; the SVD is computed once; square and rectangular V, steps on both sides of 1
    for (n, p) in ((6, 4), (4, 4), (3, 6))
      X = T.(randn(rng, n, p)); Y = T.(randn(rng, n)); v = T.(randn(rng, p))
      Xb = big.(X)
      κ = cond(Float64.(X))
      f = LeastSquaresLoss(Y, X)
      for γ in (T(1e-3), T(1), T(1e3), pow2(T, e0 + 10), T(1e30))
        β = prox!(zeros(T, p), f, v, γ)
        βref = isfinite(1 / big(γ)) ? ((Xb' * Xb + I / big(γ)) \ (Xb' * big.(Y) + big.(v) / big(γ))) : big.(v)
        tol = 400 * eps(T) * max(1, κ^2) * max(1, maximum(abs, βref))
        @test all(abs.(big.(β) .- βref) .<= tol)
        va = copy(v)
        @test prox!(va, f, va, γ) == β                            # exact aliasing
        bf = zeros(T, p)
        @test value_and_prox!(bf, f, v, γ) === f(bf) && bf == β    # the value at the stored point
      end
      # multi-response Y: one column per response, the Frobenius loss
      Y2 = T.(randn(rng, n, 2)); V2 = T.(randn(rng, p, 2))
      f2 = LeastSquaresLoss(Y2, X)
      for γ in (T(0.05), T(40))
        B2 = prox!(zeros(T, p, 2), f2, V2, γ)
        B2ref = (Xb' * Xb + I / big(γ)) \ (Xb' * big.(Y2) + big.(V2) / big(γ))
        @test all(abs.(big.(B2) .- B2ref) .<= 400 * eps(T) * max(1, κ^2) * max(1, maximum(abs, B2ref)))
      end
      # the fused and the standalone value sum the same residual in different orders: both are within a few roundings
      r2 = sum(abs2, big.(Y2) .- Xb * big.(V2)) / 2
      @test relerr(big(value_and_gradient!(zeros(T, p, 2), f2, V2)), r2) <= 2 * (n + p) * eps(T) * (1 + κ^2)
      @test relerr(big(f2(V2)), r2) <= 2 * (n + p) * eps(T) * (1 + κ^2)
    end
    # no observations or no coefficients: the loss is constant and the prox is the identity
    fe = LeastSquaresLoss(T[], zeros(T, 0, 3))
    @test prox!(zeros(T, 3), fe, T[1, 2, 3], T(2)) == T[1, 2, 3]
    @test fe(zeros(T, 3)) === zero(T)
    @test PB._svd_cache!(fe).svd_ready && size(fe.V) == (3, 0)
    # the value on an overflowing multi-response residual: Y - Xβ overflows entrywise, the squares are summed in units of 2^k
    X = T[1 0; 0 1]; Y = T[pow2(T, E - 2) pow2(T, E - 3); -pow2(T, E - 2) 1]
    β = T[-pow2(T, E - 2) 0; pow2(T, E - 2) 0]
    f = LeastSquaresLoss(Y, X)
    r = sum(abs2, big.(Y) .- big.(X) * big.(β)) / 2
    @test within(f(β), r, 3)
    @test within(value_and_gradient!(similar(β), f, β), r, 3)
    # a sparse iterate and a block iterate as the argument: the stored products, the same recovery
    h = E ÷ 2
    Xo = T[pow2(T, h + 4) pow2(T, h + 4); 1 1]; Yo = T[3, -2]
    βo = T[pow2(T, h + 4), -pow2(T, h + 4)]                       # X β = 0 with overflowing partial products
    fo = LeastSquaresLoss(Yo, Xo)
    ro = sum(abs2, big.(Yo) .- big.(Xo) * big.(βo)) / 2
    @test within(fo(βo), ro, 3)
    bs = SparseIterate(T, 2); bs[1] = βo[1]; bs[2] = βo[2]
    @test fo(bs) === fo(βo)
    bb = BlockIterate(copy(βo), [[1], [2]])
    @test within(fo(bb), ro, 3)
    # the gradient has no recovery: a genuine NaN/Inf keeps its IEEE meaning in value and gradient
    fn = LeastSquaresLoss(T[1, 2], T[1 1; 1 1])
    @test isnan(fn(T[NaN, 0])) && fn(T[Inf, 1]) === T(Inf)
    @test isnan(gradient!(zeros(T, 2), fn, T[NaN, 0])[1])
  end
end

function test_linear_function(::Type{T}) where {T<:AbstractFloat}
  E, e0 = emax(T), emin(T)
  δ = nextfloat(zero(T))
  precT = precision(T)
  rng = Xoshiro(1013)
  @testset "LinearFunction: overflowing dots, owners and iterates" begin
    rng = Xoshiro(53)
    h = E ÷ 2
    a = pow2(T, h + 4)
    # a dense dot that overflows in a partial sum, with the constant in its own units
    c = T[a, a, 1]; x = T[a, -a, 1]
    for d in (T(1), pow2(T, e0), T(-0.375))
      f = LinearFunction(c, d)
      r = sum(big.(c) .* big.(x)) + big(d)
      @test within(f(x), r, 1)
      g = zeros(T, 3)
      @test value_and_gradient!(g, f, x) === f(x) && g == c
      @test prox!(zeros(T, 3), f, x, T(2)) == x .- T(2) .* c
    end
    # a view takes the copy-and-scale dot; a sparse iterate and a block iterate their stored entries
    f = LinearFunction(c, T(1))
    @test f(view(x, :)) === f(x)
    xsp = SparseIterate(T, 3); xsp[1] = x[1]; xsp[2] = x[2]; xsp[3] = x[3]
    @test f(xsp) === f(x)
    @test f(BlockIterate(copy(x), [[1, 2], [3]])) === f(x)
    # a symmetric owner counts an off-diagonal coordinate at both positions; c is a p × p array indexed by position
    p = 3
    xo = SymmetricSparseIterate(T, p); xo[2, 1] = a; xo[3, 1] = -a; xo[3, 3] = T(2)
    cm = fill(a, p, p); cm[3, 3] = T(5)
    xm = Matrix(xo)
    for d in (T(1), T(0))
      fo = LinearFunction(cm, d)
      @test within(fo(xo), sum(big.(cm) .* big.(xm)) + big(d), 1)
      @test fo(xo) === fo(xm)                                          # the owner and the matrix it represents
    end
    # a function of a vector whose argument is a matrix of the same length: the elementwise prox, the shapes differ
    fl = LinearFunction(T[1, 2, 3, 4])
    @test prox!(zeros(T, 2, 2), fl, T[1 2; 3 4], T(2)) == reshape(vec(T[1 2; 3 4]) .- T(2) .* T[1, 2, 3, 4], 2, 2)
    # an owner output: each coordinate once
    po = SymmetricSparseIterate(T, 2)
    cs = T[1 2; 2 3]
    prox!(po, LinearFunction(cs), SymmetricSparseIterate(T, 2), T(2))
    @test Matrix(po) == -T(2) .* cs
    # genuine Inf and NaN
    @test LinearFunction(T[1, 1], zero(T))(T[Inf, 1]) === T(Inf)
    @test isnan(LinearFunction(T[1, 1], zero(T))(T[Inf, -Inf]))
    @test isnan(LinearFunction(T[NaN, 1], zero(T))(T[1, 1]))
    @test LinearFunction(T[1, 2], T(Inf))(T[1, 1]) === T(Inf)
    @test LinearFunction(T[1, 1], zero(T))(T[0, -0.0]) === zero(T)
    # the fused oracle on an owner reads the stored coordinates through _symmetric_matrix_dot (c need not be checked symmetric there)
    for d in (T(1), T(0))
      fo = LinearFunction(cm, d)
      go = SymmetricSparseIterate(T, p)
      @test within(value_and_gradient!(go, fo, xo), sum(big.(cm) .* big.(xm)) + big(d), 1)
      @test value_and_gradient!(SymmetricSparseIterate(T, p), fo, xo) === fo(xo)
      @test Matrix(go) == cm
    end
    # the scaled dots of the stored entries, for a sparse iterate, an owner (an off-diagonal entry reads c at both positions)
    # and a block iterate (the scaled copy): Σ cᵢ xᵢ 2^-k
    sf = PB._power_factors(T, -(h + 4))
    @test within(PB._linear_scaled_dot(cm, xo, sf), sum(big.(cm) .* big.(xm)) * big(2.0)^-(h + 4), 2)
    xsp3 = SparseIterate(T, 3); xsp3[1] = a; xsp3[3] = -a
    @test within(PB._scaled_dot(T[1, 2, 3], xsp3, sf), (big(a) - 3 * big(a)) * big(2.0)^-(h + 4), 1)
    @test PB._linear_scaled_dot(T[1, 2, 3], BlockIterate(T[a, 0, -a], [[1, 2], [3]]), sf) == PB._scaled_dot(T[1, 2, 3], T[a, 0, -a], sf)
  end
end

function test_norms(::Type{T}) where {T<:AbstractFloat}
  E, e0 = emax(T), emin(T)
  δ = nextfloat(zero(T))
  precT = precision(T)
  rng = Xoshiro(1014)
  @testset "norms: the symmetric owner multiplicity and the norm_diff hypot fallback" begin
    rng = Xoshiro(61)
    # the Frobenius norm of the matrix an owner represents: an off-diagonal coordinate counts twice, no square overflows
    for (e1, e2, e3) in ((E - 3, E - 4, E - 4), (e0 + 5, e0 + 3, e0 + 4), (0, 3, -2))
      xo = SymmetricSparseIterate(T, 3)
      xo[1, 1] = signif(rng, T, e1); xo[2, 1] = signif(rng, T, e2); xo[3, 2] = signif(rng, T, e3)
      xm = Matrix(xo)
      r = sqrt(sum(abs2, big.(xm)))
      @test within(norm(xo), r, 6)
      @test within(norm(xo, 2), r, 6)
      @test within(PB._norm2_scaled(xo), r, 6)
      @test norm(xo) ≈ norm(xm) rtol = 8 * eps(T)
      # 1- and Inf-norms: the off-diagonal entries count twice in the 1-norm and once in the Inf-norm
      @test within(norm(xo, 1), sum(abs, big.(xm)), 4)
      @test norm(xo, Inf) == maximum(abs, xm)
      @test within(norm(xo, 3), cbrt(sum(abs.(big.(xm)) .^ 3)), 8)       # another p reads the matrix the owner represents
    end
    # an empty owner and genuine NaN/Inf coordinates
    @test norm(SymmetricSparseIterate(T, 3)) === zero(T)
    xn = SymmetricSparseIterate(T, 2); xn[1, 1] = T(NaN); xn[2, 1] = T(3)
    @test isnan(norm(xn))
    xi = SymmetricSparseIterate(T, 2); xi[2, 1] = T(Inf); xi[2, 2] = T(3)
    @test norm(xi) === T(Inf)
    xi[2, 2] = T(NaN)
    @test isnan(norm(xi))
    # a block iterate takes the norm of its storage
    @test PB._norm2_scaled(BlockIterate(T[pow2(T, E - 2), pow2(T, E - 2), 3], [[1], [2, 3]])) ≈ sqrt(T(2)) * pow2(T, E - 2) rtol = 4 * eps(T)
    # norm_diff: the sum of squares overflows, underflows, or both arrays are equal; hypot accumulates without squaring
    for (A, B) in ((T[pow2(T, E - 2), 1, 5], T[-pow2(T, E - 2), 0, 5]),
                   (T[pow2(T, e0 ÷ 2 - 3) * T(1.3), 0], T[0, pow2(T, e0 ÷ 2 - 4)]),
                   (T[pow2(T, e0 - 10), -pow2(T, e0 - 12)], zeros(T, 2)))
      r = sqrt(sum(abs2, big.(A) .- big.(B)))
      @test within(norm_diff(A, B), r, 4)
      @test norm_diff(A, B, Inf) == maximum(abs.(big.(A) .- big.(B)))
    end
    @test norm_diff(T[1, 2, 3], T[1, 2, 3]) === zero(T)
    @test norm_diff(T[pow2(T, E - 2), -pow2(T, e0 + 3)], T[pow2(T, E - 2), -pow2(T, e0 + 3)]) === zero(T)
    @test norm_diff(zeros(T, 2, 2), zeros(T, 2, 2)) === zero(T)
    @test isnan(norm_diff(T[1, NaN], T[1, 2])) && isnan(norm_diff(T[Inf, NaN], T[1, 2]))
    @test norm_diff(T[Inf, 1], T[1, 2]) === T(Inf)
    @test norm_diff(T[Inf, 1], T[Inf, 2]) |> isnan                         # Inf - Inf is a genuine NaN
    @test norm_diff(T[3, 4], T[0, 0]) === T(5)
    @test norm_diff(T[1, -2], [3, 5], Inf) === T(7)
    # integers take a widened difference; mixed integer/float arrays take the float type
    @test norm_diff([typemax(Int), 0], [typemin(Int), 0]) ≈ 2.0^64 rtol = eps(Float64)
    @test norm_diff([typemin(Int8), 5], [typemax(Int8), 5], Inf) == 255
    @test norm_diff(T[1.5, 2], [1, 2]) === T(0.5)
    # the owner overloads read the matrices that the owners represent
    xa = SymmetricSparseIterate(T, 3); xa[2, 1] = pow2(T, E - 3); xa[3, 3] = T(2)
    xb = SymmetricSparseIterate(T, 3); xb[2, 1] = -pow2(T, E - 3)
    r = sqrt(sum(abs2, big.(Matrix(xa)) .- big.(Matrix(xb))))
    @test within(norm_diff(xa, xb), r, 6)
    @test norm_diff(xa, Matrix(xb)) == norm_diff(xa, xb) == norm_diff(Matrix(xa), xb)
    # argument errors
    @test_throws ArgumentError norm_diff(T[1], T[2], 3)
    @test_throws DimensionMismatch norm_diff(T[1, 2], T[1])
    @test_throws ArgumentError norm_diff([1 + 1im], [1])
  end
end

function test_sparse_products(::Type{T}) where {T<:AbstractFloat}
  E, e0 = emax(T), emin(T)
  δ = nextfloat(zero(T))
  precT = precision(T)
  rng = Xoshiro(1015)
  @testset "sparse and symmetric matrix products: overflowing terms, cancellation and non-finite factors" begin
    rng = Xoshiro(71)
    a = pow2(T, E ÷ 2 + 2); b = pow2(T, -(E ÷ 2 + 2))
    # each a x overflows and the signed partial sums overflow before the column sums cancel; the terms a x b are representable
    X = SparseIterate(T, 2, 2); X[1, 1] = a; X[2, 1] = -a; X[1, 2] = a * T(1.5); X[2, 2] = a * T(0.625)
    A = Symmetric(fill(a, 2, 2)); B = Symmetric(Matrix(Diagonal(fill(b, 2))))
    C = A * X * B
    Cref = big.(Matrix(A)) * big.(Matrix(X)) * big.(Matrix(B))
    @test all(within(C[i, j], Cref[i, j], 4) for i in 1:2, j in 1:2)
    @test C[1, 1] === zero(T) && C[2, 1] === zero(T)                        # an exactly zero sum is +0.0
    @test all(isnan, Matrix(A) * Matrix(X) * Matrix(B))                     # the ordinary dense product fails
    # terms with random exponents: every term is representable, the sums are not formed in range
    for trial in 1:20
      m, n = 3, 3
      Ad = Matrix(Symmetric(T[signif(rng, T, rand(rng, E ÷ 2 - 3:E ÷ 2 + 2)) for _ in 1:m, _ in 1:m]))
      Bd = Matrix(Symmetric(T[signif(rng, T, rand(rng, -(E ÷ 2 + 2):-(E ÷ 2) + 3)) for _ in 1:n, _ in 1:n]))
      Xr = SparseIterate(T, m, n)
      for k in 1:4
        Xr[rand(rng, 1:m), rand(rng, 1:n)] = signif(rng, T, rand(rng, E ÷ 2 - 3:E ÷ 2 + 2))
      end
      Cr = Symmetric(Ad) * Xr * Symmetric(Bd)
      Cref = big.(Ad) * big.(Matrix(Xr)) * big.(Bd)
      mag = abs.(big.(Ad)) * abs.(big.(Matrix(Xr))) * abs.(big.(Bd))          # the terms are summed in storage order: the error is relative to Σ|terms|
      @test all(abs(big(Cr[i, j]) - Cref[i, j]) <= 12 * eps(T) * mag[i, j] for i in 1:m, j in 1:n)
    end
    # a symmetric owner: an off-diagonal coordinate is the two entries (r, c) and (c, r), each a separate term
    Xs = SymmetricSparseIterate(T, 3); Xs[1, 1] = a; Xs[2, 1] = -a; Xs[3, 3] = a * T(0.75); Xs[3, 2] = a * T(1.25)
    As = Symmetric(T[a a a; a -a a; a a a]); Bs = Symmetric(T[b 0 b; 0 b 0; b 0 -b])
    Cs = As * Xs * Bs
    Csref = big.(Matrix(As)) * big.(Matrix(Xs)) * big.(Matrix(Bs))
    @test all(within(Cs[i, j], Csref[i, j], 6) for i in 1:3, j in 1:3)
    @test Cs isa Matrix{T}
    # an exactly zero owner or iterate gives the zero matrix without reading A or B (not even a NaN there)
    @test A * SparseIterate(T, 2, 2) * B == zeros(T, 2, 2)
    @test Symmetric(fill(T(NaN), 2, 2)) * SparseIterate(T, 2, 2) * Symmetric(fill(T(Inf), 2, 2)) == zeros(T, 2, 2)
    @test Symmetric(fill(T(NaN), 3, 3)) * SymmetricSparseIterate(T, 3) * Symmetric(fill(T(Inf), 3, 3)) == zeros(T, 3, 3)
    # genuine Inf and NaN factors keep the IEEE outcome: no finite value replaces them, and a zero against Inf is NaN
    Xi = SparseIterate(T, 2, 2); Xi[1, 1] = T(Inf); Xi[2, 2] = T(3)
    Ci = Symmetric(T[1 2; 2 3]) * Xi * Symmetric(T[0 1; 1 1])
    @test isnan(Ci[1, 1]) && Ci[1, 2] === T(Inf) && isnan(Ci[2, 1]) && Ci[2, 2] === T(Inf)
    Cz = Symmetric(T[0 0; 0 3]) * Xi * Symmetric(T[1 1; 1 1])
    @test all(isnan, Cz)
    Xnan = SymmetricSparseIterate(T, 2); Xnan[1, 1] = T(NaN)
    @test isnan((Symmetric(T[1 2; 2 3]) * Xnan * Symmetric(T[1 1; 1 1]))[1, 1])
    Xinf = SymmetricSparseIterate(T, 2); Xinf[2, 1] = T(-Inf)
    Cinf = Symmetric(T[1 2; 2 3]) * Xinf * Symmetric(T[1 1; 1 1])
    @test all(!isfinite, Cinf)
    # the rescue of one entry directly: scaled by the largest exponent among its terms; nothing when a factor is not finite
    Cd = fill(T(Inf), 2, 2)
    @test PB._rescue_entry!(Cd, Matrix(A), X, Matrix(B), 1, 2) === nothing
    @test within(Cd[1, 2], Cref[1, 2] * 0 + big.(Matrix(A))[1, :]' * big.(Matrix(X)) * big.(Matrix(B))[:, 2], 4)
    Cd = fill(T(NaN), 2, 2)
    Afin = Matrix(A); Afin[1, 1] = T(Inf)
    @test PB._rescue_entry!(Cd, Afin, X, Matrix(B), 1, 1) === nothing && isnan(Cd[1, 1])
    @test PB._rescue_nonfinite!(copy(C), Matrix(A), X, Matrix(B)) == C
    # the three-factor term: NaN, zero against infinity, signed infinities, and a zero factor
    @test isnan(PB._triple(T(NaN), T(2), T(3))) && isnan(PB._triple(T(0), T(Inf), T(3))) && isnan(PB._triple(T(2), T(3), T(Inf) * 0))
    @test PB._triple(T(Inf), T(-2), T(3)) === -T(Inf) && PB._triple(T(Inf), T(-2), T(-3)) === T(Inf)
    @test PB._triple(T(0), T(2), T(3)) === zero(T) && PB._triple(T(2), -T(0), T(3)) === zero(T)
    for _ in 1:2000
      x1, x2, x3 = signif(rng, T, rand(rng, -E:E)), signif(rng, T, rand(rng, -E:E)), signif(rng, T, rand(rng, -E:E))
      r = big(x1) * big(x2) * big(x3)
      (abs(r) > floatmax(T) || abs(r) < nextfloat(zero(T))) && continue
      @test within(PB._triple(x1, x2, x3), r, 2)                          # within one neighbour of the rounded exact term
    end
    @test within(PB._triple(pow2(T, e0 ÷ 2) * T(1.3), pow2(T, e0 ÷ 2 - 3) * T(1.7), pow2(T, -e0 ÷ 2 + 2) * T(1.1)),
                 big(pow2(T, e0 ÷ 2) * T(1.3)) * big(pow2(T, e0 ÷ 2 - 3) * T(1.7)) * big(pow2(T, -e0 ÷ 2 + 2) * T(1.1)), 1)   # a subnormal term
    # sums of subnormal terms next to a large one, in storage order
    Xt = SparseIterate(T, 1, 1); Xt[1, 1] = pow2(T, e0 ÷ 2 - 3) * T(1.7)
    At = Symmetric(fill(pow2(T, e0 ÷ 2) * T(1.3), 1, 1)); Bt = Symmetric(fill(pow2(T, -e0 ÷ 2 + 2) * T(1.1), 1, 1))
    @test within((At * Xt * Bt)[1, 1], big(At[1, 1]) * big(Xt[1, 1]) * big(Bt[1, 1]), 1)
  end
end

function test_symmetric_part(::Type{T}) where {T<:AbstractFloat}
  E, e0 = emax(T), emin(T)
  δ = nextfloat(zero(T))
  precT = precision(T)
  rng = Xoshiro(1016)
  @testset "symmetric part of an asymmetric argument: the mean and the tiny-α product" begin
    rng = Xoshiro(81)
    # (a + b)/2 correctly rounded where the sum is finite (subnormal digits kept), halves first where it overflows
    @test within(PB._symmetric_mean(floatmax(T), floatmax(T) / 2), big(floatmax(T)) * 3 / 4, 1)             # the sum overflows
    @test PB._symmetric_mean(floatmax(T), -floatmax(T)) === zero(T)
    @test PB._symmetric_mean(3δ, 4δ) === T(big(7) * big(δ) / 2)                  # a tie rounds to even: 4δ
    @test PB._symmetric_mean(δ, zero(T)) === T(big(δ) / 2)                         # half the smallest subnormal ties to even
    @test isnan(PB._symmetric_mean(T(Inf), -T(Inf)))
    @test PB._symmetric_mean(T(Inf), T(3)) === T(Inf)
    @test isnan(PB._symmetric_mean(T(NaN), T(3)))
    for _ in 1:300
      u, v = signif(rng, T, rand(rng, e0:E)), signif(rng, T, rand(rng, e0:E))
      @test within(PB._symmetric_mean(u, v), (big(u) + big(v)) / 2, 1)
      @test PB._symmetric_mean(u, v) === PB._symmetric_mean(v, u)                  # exactly symmetric
    end
    # the lazily read symmetric part: equal pairs (NaN matches NaN, ±0 agree) are returned as they are
    V = T[1 2 NaN; 4 5 6; NaN 8 9]
    S = PB._SymmetricPart(V)
    @test S[1, 2] == 3 && S[2, 1] == 3 && isnan(S[1, 3]) && isnan(S[3, 1]) && S[2, 2] === T(5)
    @test isequal(collect(S), collect(S')) && transpose(S) === S
    @test iszero(PB._SymmetricPart(T[0.0 -0.0; 0.0 1])[1, 2]) && iszero(PB._SymmetricPart(T[0.0 -0.0; 0.0 1])[2, 1])
    # an owner output reads the argument through its symmetric part
    xo = T[pow2(T, E - 2) 5; -pow2(T, E - 2) 7]
    zo = SymmetricSparseIterate(T, 2)
    prox!(zo, ZeroFunction(), xo, T(1))
    @test zo[1, 1] === xo[1, 1] && zo[2, 2] === xo[2, 2]
    @test within(zo[2, 1], (big(xo[2, 1]) + big(xo[1, 2])) / 2, 1)
    xo2 = T[floatmax(T) 1; floatmax(T) / 2 0]
    zo2 = SymmetricSparseIterate(T, 2)
    prox!(zo2, ZeroFunction(), T[1 floatmax(T); floatmax(T) / 2 0], T(1))
    @test within(zo2[2, 1], (big(floatmax(T)) + big(floatmax(T) / 2)) / 2, 1)
    # the product with a tiny α: halving α would lose digits, so each term is formed from significands (rounded once)
    A = T[pow2(T, E ÷ 2) 3; 5 -pow2(T, E ÷ 2 - 3)] .* T(1.1)
    B = T[1 2; 3 4] .* pow2(T, E ÷ 2 - 10)
    for α in (T(1.3) * pow2(T, e0 - precT + 4), pow2(T, e0 - 4), T(0.7))
      for β in (zero(T), T(2))
        C = T[1 2; 3 4] .* pow2(T, -E ÷ 2)
        C0 = copy(C)
        mul!(C, PB._SymmetricPart(A), B, α, β)
        Sm = (big.(A) + big.(A)') / 2
        ref = big(α) * Sm * big.(B) + big(β) * big.(C0)
        mag = abs(big(α)) * abs.(Sm) * abs.(big.(B)) + abs(big(β)) * abs.(big.(C0))
        @test all(abs(big(C[i, j]) - ref[i, j]) <= 4 * eps(T) * mag[i, j] + eps(abs(T(ref[i, j]))) for i in 1:2, j in 1:2)
      end
    end
    # the vector form
    y = zeros(T, 2); bv = T[pow2(T, E ÷ 2 - 10), 1]
    α = T(1.3) * pow2(T, e0 - precT + 4)
    mul!(y, PB._SymmetricPart(A), bv, α, zero(T))
    refv = big(α) * ((big.(A) + big.(A)') / 2) * big.(bv)
    @test all(abs(big(y[i]) - refv[i]) <= 4 * eps(T) * abs(big(α)) * ((abs.((big.(A) + big.(A)') / 2)) * abs.(big.(bv)))[i] + eps(abs(T(refv[i]))) for i in 1:2)
    @test_throws DimensionMismatch PB._scaled_symmetric_mul!(zeros(T, 3, 2), PB._SymmetricPart(A), B, α, zero(T))
    # the terms of the tiny-α product (_scaled_product with k = 0): rounded once for finite factors, ordinary for non-finite, zero for zero
    @test within(PB._scaled_product(pow2(T, E - 2) * T(1.3), pow2(T, E - 2) * T(1.7), pow2(T, -(E - 4)) * T(1.1), 0), big(pow2(T, E - 2) * T(1.3)) * big(pow2(T, E - 2) * T(1.7)) * big(pow2(T, -(E - 4)) * T(1.1)), 1)
    @test PB._scaled_product(zero(T), T(Inf), T(2), 0) === zero(T) && PB._scaled_product(T(Inf), T(2), T(3), 0) === T(Inf) && isnan(PB._scaled_product(T(NaN), T(2), T(3), 0))
  end
end

function test_scaled_single_factors(::Type{T}) where {T<:AbstractFloat}
  E, e0 = emax(T), emin(T)
  δ = nextfloat(zero(T))
  precT = precision(T)
  rng = Xoshiro(1017)
  @testset "Scaled single factors, zero and infinite scales, aliases and nested wrappers" begin
    rng = Xoshiro(91)
    # a zero scale simplifies only where the data and parameters are finite
    @test Scaled(NormL1(T(2)), 0) isa ZeroFunction && Scaled(NormL1(T(2), T[1, 2]), zero(T)) isa ZeroFunction
    @test Scaled(SqrNormL2(T(2), T[1, 2]), 0) isa ZeroFunction && Scaled(LinearFunction(T[1, 2], T(3)), 0) isa ZeroFunction
    @test Scaled(QuadraticFunction(T[1 0; 0 1]), 0) isa ZeroFunction && Scaled(LeastSquaresLoss(T[1, 2], T[1 0; 0 1]), 0) isa ZeroFunction
    @test Scaled(Scaled(NormL1(T(2)), T(3)), 0) isa ZeroFunction
    @test_throws ArgumentError Scaled(NormL1(T(Inf)), 0)
    @test_throws ArgumentError Scaled(SqrNormL2(T(2), T[Inf, 1]), 0)
    @test_throws ArgumentError Scaled(SqrNormL2(T(Inf)), 0)
    @test_throws ArgumentError Scaled(LinearFunction(T[1, NaN]), 0)
    @test_throws ArgumentError Scaled(QuadraticFunction(T[1 0; 0 NaN]), 0)
    @test_throws ArgumentError Scaled(LeastSquaresLoss(T[Inf, 2], T[1 0; 0 1]), 0)
    @test_throws ArgumentError Scaled(NormL1(T(1)), -1)
    @test Scaled(NormL1(zero(T), T[Inf, 1]), 0) isa NormL1 && Scaled(NormL1(zero(T), T[Inf, 1]), 0)(T[1, 1]) === T(Inf)   # an infinite weight survives the zero scale
    # an ordinary single factor: the value, the gradient and the fused forms are plain products
    f = SqrNormL2(T(1.5), T[1, -2]); s = Scaled(f, T(3))
    x = T[4, 5]
    @test s(x) === T(3) * f(x)
    g = zeros(T, 2)
    @test value_and_gradient!(g, s, x) === s(x) && g == T(3) .* gradient!(zeros(T, 2), f, x)
    yv = zeros(T, 2)
    @test value_and_prox!(yv, s, x, T(0.5)) === s(yv) && yv == prox!(zeros(T, 2), f, x, T(1.5))
    xo = SymmetricSparseIterate(T, 2); xo[2, 1] = T(2); xo[2, 2] = T(3)
    go = SymmetricSparseIterate(T, 2)
    gradient!(go, Scaled(SqrNormL2(T(1.5)), T(3)), xo)
    @test Matrix(go) == T(4.5) .* Matrix(xo)
    @test value_and_gradient!(SymmetricSparseIterate(T, 2), Scaled(SqrNormL2(T(1.5)), T(3)), xo) === Scaled(SqrNormL2(T(1.5)), T(3))(xo)
    # an array step in the band is a plain product; its value at the stored point recovers an overflowing product (α < 1)
    fl = NormL1(pow2(T, E - 1))
    sl = Scaled(fl, pow2(T, -5))
    γz = zeros(T, 3)
    xl = T[pow2(T, 1), -pow2(T, 2), pow2(T, 0)]
    @test !PB._array_route(sl, xl, γz)
    @test fl(xl) === T(Inf)
    yl = zeros(T, 3)
    vl = value_and_prox!(yl, sl, xl, γz)
    @test yl == xl && within(vl, big(pow2(T, -5)) * l1val(pow2(T, E - 1), nothing, xl), 2)
    @test PB._scaled_value_overflow(fl, pow2(T, -5), xl, T(Inf)) |> v -> within(v, big(pow2(T, -5)) * l1val(pow2(T, E - 1), nothing, xl), 2)
    # an indicator ignores its scale: the value at a projection is exactly 0 (or Inf), an infinite factor included
    @test PB._indicator_product(ZeroFunction(), T(NaN), T(0)) |> isnan
    # an infinite factor with an array step: the steps become infinite, the value at the zero output is 0 (Inf * 0 = 0 at a zero function)
    si = Scaled(NormL1(T(2)), T(Inf))
    γp = T[1, 2, 3]
    @test PB._array_route(si, xl, γp) && PB._infinite_alone(si)
    zi = prox!(zeros(T, 3), si, T[3, -4, 0.5], γp)
    @test zi == zeros(T, 3)
    yi = zeros(T, 3)
    @test value_and_prox!(yi, si, T[3, -4, 0.5], γp) === zero(T) && yi == zeros(T, 3)
    # a stored infinite level fixes the nested chain's function at zero whatever the factors
    @test PB._hard_scale(T(3)) && !PB._hard_scale(T(Inf)) && !PB._hard_scale(zero(T))
    @test PB._has_infinite_level(Scaled(NormL1(T(Inf)), T(2))) && !PB._has_infinite_level(NormL1(T(2)))
    # the center of a SqrNormL2 may be the array that receives the gradient: its parts must not be read after the write
    yc = T[1, -2]
    fa = SqrNormL2(T(1), yc)
    @test PB._parameter_alias(fa, yc) && !PB._parameter_alias(fa, copy(yc)) && PB._parameter_alias(Scaled(fa, T(2)), yc)
    @test !PB._parameter_alias(SqrNormL2(T(1)), yc)
    # a nested wrapper with an ordinary product of factors, over a function with parts: the exact zero and the zero exponent
    inner = Scaled(SqrNormL2(one(T)), pow2(T, 5))
    outer = Scaled{typeof(inner), T}(inner, pow2(T, E - 10))
    @test outer(zeros(T, 2)) === zero(T)
    xt = T[signif(rng, T, e0 ÷ 2 - 3), signif(rng, T, e0 ÷ 2 - 4)]
    @test within(outer(xt), big(pow2(T, E - 10)) * big(pow2(T, 5)) * sqval(1, nothing, xt), 4)
    @test PB._zero_value_exponent(inner, xt) isa Int && PB._zero_value_exponent(Scaled(SqrNormL2(zero(T)), T(2)), xt) isa Int
    @test PB._zero_value_exponent(Scaled(ZeroFunction(), T(2)), xt) === nothing
    # the retained exponent of a chain: significand, exponent, and the inverse
    a = PB._scale_start(pow2(T, -(E ÷ 2)) * T(1.5))
    @test a.m * ldexp(one(T), a.e) == pow2(T, -(E ÷ 2)) * T(1.5) && 0.5 <= a.m < 1
    ai = PB._scale_inverse(a)
    @test 0.5 <= ai.m < 1 && relerr(big(ai.m) * big(2.0)^ai.e, 1 / (big(pow2(T, -(E ÷ 2))) * big(T(1.5)))) <= 2 * eps(T)
    @test PB._scale_inverse(T(4)) === T(0.25)
    @test PB._scale_below_one(a) && !PB._scale_above_one(a) && PB._scale_above_one(PB._scale_start(T(3))) && !PB._scale_above_one(PB._scale_start(T(1)))
    @test within(PB._apply_scale(a, T(3)), big(3) * big(pow2(T, -(E ÷ 2))) * big(T(1.5)), 1)
    @test PB._apply_scale(a, zero(T)) === zero(T) && PB._apply_scale(a, T(Inf)) === T(Inf) && isnan(PB._apply_scale(a, T(NaN)))
    @test PB._projection_accumulate(T(1), T(5)) === T(5) && PB._projection_accumulate(T(0.5), zero(T)) === zero(T)
    @test PB._projection_accumulate(pow2(T, -E ÷ 2 - 10), pow2(T, -E ÷ 2 - 10)) isa PB._ProjectionScale
    @test PB._projection_accumulate(T(2), T(3)) === T(6)
    @test PB._projection_accumulate(T(Inf), T(3)) === T(Inf) && PB._projection_accumulate(PB._scale_start(T(2)), zero(T)) === zero(T)
    @test PB._projection_accumulate(PB._scale_start(T(2)), T(Inf)) === T(Inf)
    # a factor that is not an IEEE float is folded as it is (Int, BigFloat): no retained exponent
    @test Scaled(NormL1(T(2)), 3)(T[1, -2]) === T(18)
    @test within(Scaled(NormL1(T(2)), big(3))(T[1, -2]), big(18.0), 1)
    @test Scaled(ZeroFunction(), 0) isa ZeroFunction
  end
end

function test_recorded_defects(::Type{T}) where {T<:AbstractFloat}
  E, e0 = emax(T), emin(T)
  δ = nextfloat(zero(T))
  precT = precision(T)
  rng = Xoshiro(1018)
  @testset "recorded defects (see the audit document): @test_broken against the independent reference" begin
    pp = E ÷ 2 + 20
    # D1: the prox of a nested chain over a weighted NormL1 loses the threshold when the level cannot hold α γ λ and the weights
    # compensate. The value of the same chain recovers it (partial fold) and so does the array-step route (_shifted_fold).
    # Underflow side: α1 α2 γ λ0 is below the range (zero, or a subnormal level with a few bits), α1 α2 γ λ0 w is representable.
    f = NormL1(T(1.3) * pow2(T, 20), T[pow2(T, E - 10)])
    s = Scaled(Scaled(f, pow2(T, -pp)), pow2(T, -pp))
    γ = pow2(T, -30)
    t = big(pow2(T, -pp))^2 * big(γ) * big(f.λ0) * big(f.w[1])
    x = T[3 * pow2(T, -50)]
    @test floatmin(T) < T(t) < x[1]
    @test within(s(T[1]), big(pow2(T, -pp))^2 * l1val(f.λ0, f.w, T[1]), 2)                                 # the value is right
    @test_broken within(prox!(zeros(T, 1), s, x, γ)[1], shrinkb(big(x[1]), t), 4)
    @test_broken within(prox_block(s, x[1], 1, γ), shrinkb(big(x[1]), t), 4)
    @test within(prox!(zeros(T, 1), s, x, T[γ])[1], shrinkb(big(x[1]), t), 4)                              # the array step is right
    # Overflow side: the chain factor and the step multiply to a number beyond the range, the weight brings it back.
    fo = NormL1(T(1.3) * pow2(T, E - 20), T[pow2(T, -(E - 10) - 20)])
    so = Scaled(Scaled(fo, pow2(T, pp)), pow2(T, pp))
    γo = pow2(T, -30)
    to = big(pow2(T, pp))^2 * big(γo) * big(fo.λ0) * big(fo.w[1])
    xo = T[3 * T(to)]
    @test isfinite(T(to)) && isfinite(xo[1])
    @test_broken within(prox!(zeros(T, 1), so, xo, γo)[1], shrinkb(big(xo[1]), to), 4)
    # D2: an infinite entry under a weighted level whose product underflows is NaN, where the unweighted value (and the exact
    # product of the positive finite factors with +Inf) is +Inf
    fu = NormL1(pow2(T, -(E ÷ 2)), T[pow2(T, -(E - 5))])
    @test NormL1(pow2(T, -(E ÷ 2)))(T[Inf]) === T(Inf)
    @test_broken fu(T[Inf]) === T(Inf)
    @test_broken PB._positive_product(pow2(T, -(E ÷ 2)), pow2(T, -(E - 5)), T(Inf)) === T(Inf)
  end
end

setprecision(BigFloat, 2400) do
@testset "numerical recovery $T" for T in (Float64, Float32)
  test_expsum_accumulator(T)   # exponent-sum accumulator (_ExpSum)
  test_quadratic_value(T)   # quadratic value: overflow, cancellation and underflow recovery
  test_quadratic_guards(T)   # quadratic recovery trigger and guards
  test_quadratic_oracles(T)   # quadratic oracles: fused and standalone, eigendecomposition prox
  test_scaled_values(T)   # Scaled values: overflow, underflow and exact zeros through the factor
  test_scaled_oracles(T)   # Scaled oracles on factor chains: gradient, prox, fused and block forms
  test_scaled_array_steps(T)   # Scaled array steps: folded, shifted and rounded routes
  test_normL1_values(T)   # NormL1 values: levels and weights outside the range, infinite and zero parameters
  test_normL1_prox(T)   # NormL1 prox thresholds and the weight kernels
  test_pow2_sums(T)   # power-of-two sums (_pow2_weighted) and the recomputed-dot shift
  test_square_consumers(T)   # squared-residual consumers: the exceptional branches of the value
  test_sqrnorml2_oracles(T)   # SqrNormL2 oracles: ridge weights, owners, wide levels and centers
  test_leastsquares(T)   # LeastSquaresLoss: SVD prox, multi-response data and iterate arguments
  test_linear_function(T)   # LinearFunction: overflowing dots, owners and iterates
  test_norms(T)   # norms: the symmetric owner multiplicity and the norm_diff hypot fallback
  test_sparse_products(T)   # sparse and symmetric matrix products: overflowing terms, cancellation and non-finite factors
  test_symmetric_part(T)   # symmetric part of an asymmetric argument: the mean and the tiny-α product
  test_scaled_single_factors(T)   # Scaled single factors, zero and infinite scales, aliases and nested wrappers
  test_recorded_defects(T)   # recorded defects (see the audit document): @test_broken against the independent reference
end
end

end
