module NumericalKernelsTest

# The shared numerical kernels of src/numerical_kernels.jl (issue #865), tested at the primitive level in
# Float32 and Float64: finite inputs whose intermediate products, differences or squares leave the
# range while the result is representable, underflow restored by a later factor, genuine NaN/Inf
# inputs, exact zeros and signed zeros. BigFloat is the test oracle only; the kernels never use it.

using Test, ProximalBase, Random, SparseArrays
using LinearAlgebra: norm
const PB = ProximalBase

pow2(T, k) = ldexp(one(T), k)
emax(T) = exponent(floatmax(T))
emin(T) = exponent(floatmin(T))
# |a - r| within n units in the last place of the rounded reference (subnormal units included)
function within(a::T, r::BigFloat, n) where {T}
  t = T(r)
  isfinite(t) || return a == t
  abs(big(a) - r) <= n * eps(abs(t))
end
relerr(a::BigFloat, r::BigFloat) = iszero(r) ? abs(a) : abs(a - r) / abs(r)
# a random significand in [1/2, 1) with a random sign, times 2^e
signif(rng, T, e) = (rand(rng, Bool) ? -1 : 1) * ldexp(T(0.5) + T(rand(rng)) / 2, e)
# equal entry by entry under ===, so that signed zeros and NaN bit patterns count
same(a, b) = length(a) == length(b) && all(i -> a[i] === b[i], eachindex(a))
# sums over an indexed vector, by iteration and by index
function iter_sum(v)
  s = zero(eltype(v))
  for a in v
    s += a
  end
  s
end
function index_sum(v)
  s = zero(eltype(v))
  for j in eachindex(v)
    s += v[j]
  end
  s
end
# the allocations of a second call
alloc(f::F, a::A) where {F, A} = (f(a); @allocated f(a))
# BigFloats are the same when sign, precision and value agree (NaN with NaN)
bigsame(a::BigFloat, b::BigFloat) = signbit(a) == signbit(b) && precision(a) == precision(b) &&
  ((isnan(a) && isnan(b)) || (!isnan(a) && !isnan(b) && a == b))

setprecision(BigFloat, 1200) do
@testset "numerical kernels $T" for T in (Float64, Float32)
  rng = Xoshiro(865)
  E, e0 = emax(T), emin(T)

  @testset "bounded-product kernel" begin
    # finite factors whose partial products leave the range, with a representable final product
    for _ in 1:2000
      ea, eb = rand(rng, -E:E), rand(rng, -E:E)
      ec = rand(rng, -E:E)
      k = rand(rng, -2E:2E)
      a, b, c = signif(rng, T, ea), signif(rng, T, eb), signif(rng, T, ec)
      r = big(a) * big(b) * big(c) * big(2.0)^k
      (abs(r) > floatmax(T) || abs(r) < nextfloat(zero(T))) && continue
      @test within(PB._scaled_product(a, b, c, k), r, 2)
      w = abs(a)
      @test within(PB._scaled_product(w, b, k), big(w) * big(b) * big(2.0)^k, 2)
      k == 0 && @test within(PB._exceptional_product(a, b, c), r, 2)
    end
    # the ordinary product is kept while ab is normal; outside, a later factor restores the range
    a, b = T(3), T(5) / 7
    @test PB._positive_product(a, b, T(11)) === (a * b) * T(11)
    @test PB._positive_product(pow2(T, -E + 4), pow2(T, -E + 4), pow2(T, E)) === pow2(T, -E + 8)
    @test PB._positive_product(pow2(T, E), pow2(T, E), pow2(T, -E - 3)) === pow2(T, E - 3)
    # a subnormal final product: no later factor, rounded once from the scaled significands
    @test within(PB._positive_product(pow2(T, e0), T(0.75), T(0.5)), big(0.375) * big(2.0)^e0, 1)
    # zero factors: +0 from the three-factor kernel (a zero beats an infinite factor), w v from the weighted one
    @test PB._scaled_product(zero(T), T(-3), T(2), 5) === zero(T)
    @test PB._scaled_product(T(Inf), zero(T), T(2), 5) === zero(T)
    @test PB._exceptional_product(T(Inf), zero(T), T(2)) === zero(T)
    @test PB._scaled_product(T(2), -zero(T), 5) === -zero(T)
    @test PB._scaled_product(zero(T), T(-3), 5) === -zero(T)
    # genuine NaN and Inf factors give the ordinary product
    @test isnan(PB._scaled_product(T(NaN), T(2), T(3), 0))
    @test PB._scaled_product(T(-Inf), T(2), T(3), -10) === T(-Inf)
    @test PB._scaled_product(T(Inf), T(-2), 7) === T(-Inf)
    @test isnan(PB._scaled_product(T(2), T(NaN), 7))
    # the retained-exponent variant agrees with the kernel where the exponent does not wrap
    for _ in 1:200
      w, v, k = abs(signif(rng, T, rand(rng, -E:E))), signif(rng, T, rand(rng, -E:E)), rand(rng, -E:E)
      @test PB._scaled_product_retained(w, v, k) === PB._scaled_product(w, v, k)
    end
  end

  @testset "halved difference" begin
    # a finite difference is exact units (h = 0) and bitwise the ordinary one
    for _ in 1:500
      a, b = signif(rng, T, rand(rng, e0:E+1)), signif(rng, T, rand(rng, e0:E+1))
      d, h = PB._halved_difference(a, b, T)
      if isfinite(a - b)
        @test (d, h) === (a - b, 0)
      else
        # an overflowing difference of finite operands: a/2 - b/2, one binade up, finite
        @test h == 1 && isfinite(d)
        @test within(d, (big(a) - big(b)) / 2, 1)
      end
    end
    d, h = PB._halved_difference(floatmax(T), -floatmax(T), T)
    @test (d, h) === (floatmax(T), 1)
    @test PB._residual_exponent(floatmax(T), -floatmax(T), T) == E + 1
    # cancellation keeps its units: the difference is formed before any scaling
    @test PB._halved_difference(nextfloat(T(1)), T(1), T) === (eps(T), 0)
    # zeros, and genuine Inf/NaN operands, which the callers screen before using the result
    @test PB._residual_exponent(T(3), T(3), T) == typemin(Int)
    @test PB._residual_exponent(T(Inf), T(3), T) == typemax(Int)
    @test PB._residual_exponent(T(3), T(NaN), T) == typemax(Int)
    # integer operands take a widened integer difference
    @test PB._halved_difference(typemin(Int), typemax(Int), Float64) === (-18446744073709551615.0, 0)
  end

  @testset "squared-residual kernel" begin
    ref(x, y) = sum(abs2, big.(collect(x)) .- (y === nothing ? 0 : big.(y)); init = big(0))
    parts_value(q) = big(q[1]) * big(2.0)^(2q[2])
    scales = (e0 + 2, e0 ÷ 2 - 3, -20, 0, 20, E ÷ 2 + 3, E - 1, E, E + 1)   # signif(·, E + 1) is the top binade
    for _ in 1:400, n in (1, 4, 13)
      x = T[signif(rng, T, rand(rng, scales)) for _ in 1:n]
      y = rand(rng, Bool) ? nothing : T[signif(rng, T, rand(rng, scales)) for _ in 1:n]
      q = PB._residual_squares(x, y, T)
      @test q isa Tuple{T, Int}
      @test relerr(parts_value(q), ref(x, y)) <= (n + 4) * eps(T)
    end
    # every square overflows; every square underflows; the differences overflow
    for (x, y) in ((fill(pow2(T, E - 2), 3), nothing), (fill(pow2(T, e0 + 1), 3), nothing),
                   ([floatmax(T), T(1)], [-floatmax(T), T(1)]),
                   ([T(1), pow2(T, -E)], [nextfloat(T(1)), zero(T)]))   # cancellation beside a tiny residual
      q = PB._residual_squares(x, y, T)
      @test relerr(parts_value(q), ref(x, y)) <= 6eps(T)
    end
    # stored entries only, with the logical multiplicity of a symmetric iterate
    xs = PB.SparseIterate(T, 9)
    xs[2], xs[7] = pow2(T, E - 1), T(-3)
    @test relerr(parts_value(PB._residual_squares(xs, nothing, T)), ref(Vector(xs), nothing)) <= 4eps(T)
    xm = PB.SymmetricSparseIterate(T, 3)
    xm[2, 1], xm[3, 3] = pow2(T, e0), T(-0.75) * pow2(T, e0)
    @test relerr(parts_value(PB._residual_squares(xm, nothing, T)), ref(Matrix(xm), nothing)) <= 4eps(T)
    # exact zero residuals; genuine Inf/NaN entries give `nothing` (the ordinary result stands)
    @test PB._residual_squares(zeros(T, 3), nothing, T) === (zero(T), 0)
    @test PB._residual_squares(T[1, -2], T[1, -2], T) === (zero(T), 0)
    @test PB._residual_squares(T[1, Inf], nothing, T) === nothing
    @test PB._residual_squares(T[1, 2], T[NaN, 2], T) === nothing
  end

  @testset "indexed values" begin
    # the entries x[G[j]] of a parent read without a copy: equal to x[G] under === for a Vector, a
    # Matrix (linear indices), a SparseIterate and a BlockIterate, with a range, a permuted vector and a
    # noncontiguous vector as groups
    n = 12
    d = T[signif(rng, T, rand(rng, -20:20)) for _ in 1:n]
    d[3], d[6], d[8] = -zero(T), T(NaN), T(-Inf)
    xs = PB.SparseIterate(T, n)
    for i in (2, 3, 6, 9, 12)
      xs[i] = d[i]
    end
    xb = PB.BlockIterate(copy(d), 3)
    parents = (vector = d, matrix = reshape(copy(d), 3, 4), sparse = xs, block = xb)
    groups = (2:9, 5:5, [7, 3, 12, 1, 5], [11, 2, 9, 4], Int[])
    for (name, parent) in pairs(parents), G in groups
      v = PB._IndexedValues(parent, G)
      ref = [parent[i] for i in G]
      @test v isa AbstractVector{T} && length(v) == length(G) && size(v) == (length(G),)
      @test same([v[j] for j in 1:length(v)], ref)
      @test same([a for a in v], ref)
      @test same(collect(v), ref)
      @test same(collect(PB._stored_values(v)), ref)
      @test all(j -> PB._square_multiplicity(v, j) == 1, eachindex(v))
    end
    # a warmed sum allocates nothing, by iteration, by index and through sum
    for parent in (parents.vector, parents.matrix), G in (2:9, [7, 3, 12, 1, 5])
      v = PB._IndexedValues(copy(parent), G)
      @test alloc(iter_sum, v) == 0
      @test alloc(index_sum, v) == 0
      @test alloc(sum, v) == 0
    end
    # with an element type the entries are rounded to it: the entries of a wider parent as the loops in that type read
    # them, a group that is zero only after the rounding is zero, and the rounding does not allocate
    if T === Float64
      tiny = fill(-prevfloat(2.0^-150), 6)                     # below half the smallest Float32
      tiny[3] = -0.0
      Gt = [5, 3, 1, 6]
      for S in (Float32, Float16)
        v = PB._IndexedValues(tiny, Gt, S)
        @test eltype(v) === S && length(v) == 4
        @test same(collect(v), S.(tiny[Gt])) && same([a for a in v], S.(tiny[Gt])) && same([v[j] for j in 1:4], S.(tiny[Gt]))
        @test PB._all_zero(v) && PB._l2_all_zero(tiny, Gt, S)
        @test alloc(iter_sum, v) == 0 && alloc(index_sum, v) == 0
      end
      @test !PB._all_zero(PB._IndexedValues(tiny, Gt)) && !PB._l2_all_zero(tiny, Gt, Float64)
      # the entries of Float32 and larger magnitude are not zero in Float32, one entry that is not rounds the group
      mixed = copy(tiny); mixed[6] = 2.0^-140
      @test !PB._l2_all_zero(mixed, Gt, Float32) && PB._l2_all_zero(mixed, [1, 2, 3], Float32)
      # the same through a matrix parent and a vector group
      @test PB._l2_all_zero(reshape(copy(tiny), 2, 3), [1, 4, 6, 3], Float32)
      # NaN, an infinity and an overflow in the rounding are not zero
      for bad in (NaN, Inf, -Inf, 1e300)
        w = copy(tiny); w[2] = bad
        @test !PB._l2_all_zero(w, [1, 2, 3], Float32) && !PB._l2_all_zero(w, [1, 2, 3], Float16)
      end
    end
    # an index outside 1:length(G) is a BoundsError when bounds checks are on
    if Base.JLOptions().check_bounds != 2
      for G in (2:9, [7, 3, 12, 1, 5])
        v = PB._IndexedValues(d, G)
        @test_throws BoundsError v[0]
        @test_throws BoundsError v[length(G) + 1]
      end
    end
  end

  @testset "exponent codes" begin
    se(v) = PB._square_exponent(v, nothing, T)
    @test PB._EXPONENT_NAN == typemax(Int) && PB._EXPONENT_INF == typemax(Int) - 1 && PB._EXPONENT_ZERO == typemin(Int)
    # the NaN code beats the Inf code at either order and any position; an infinite entry gives the Inf code
    @test se(T[Inf, 1]) == PB._EXPONENT_INF
    @test se(T[1, Inf]) == PB._EXPONENT_INF
    @test se(T[-Inf, 2]) == PB._EXPONENT_INF
    @test se(T[Inf, NaN]) == PB._EXPONENT_NAN
    @test se(T[NaN, Inf]) == PB._EXPONENT_NAN
    @test se(T[Inf, 1, 5, NaN, -Inf]) == PB._EXPONENT_NAN
    @test se(T[NaN, 1, 5, Inf]) == PB._EXPONENT_NAN
    @test se(T[1, NaN]) == PB._EXPONENT_NAN
    # exactly zero entries, signed zeros included, give the zero code; otherwise the largest exponent
    @test se(T[0.0, -0.0]) == PB._EXPONENT_ZERO
    @test se(T[]) == PB._EXPONENT_ZERO
    @test se(T[1.0, 3.0]) == 1
    @test se(T[-0.0, 0.75, -0.5]) == -1
    @test se(T[floatmax(T), 1]) == E
    # with a second operand a residual that is not finite is not classified: NaN, Inf and Inf - Inf all give
    # the code of the base, typemax(Int)
    @test PB._EXPONENT_NONFINITE == typemax(Int) && PB._EXPONENT_NONFINITE >= PB._EXPONENT_INF
    for (a, b) in ((T(Inf), T(Inf)), (T(Inf), T(-Inf)), (T(Inf), T(1)), (T(1), T(-Inf)), (T(NaN), T(Inf)), (T(NaN), T(1)), (T(1), T(NaN)))
      @test PB._residual_exponent(a, b, T) == typemax(Int)
    end
    @test PB._square_exponent(T[Inf, 1], T[Inf, 0], T) == typemax(Int)
    @test PB._square_exponent(T[Inf, 1], T[0, 0], T) == typemax(Int)
    @test PB._square_exponent(T[1, Inf], T[1, -Inf], T) == typemax(Int)
    @test PB._square_exponent(T[3, 3], T[3, 3], T) == PB._EXPONENT_ZERO
    @test PB._square_exponent(T[3, 5], T[1, 1], T) == exponent(T(4))
    # the squared residuals are `nothing` for both codes, and a pair for a finite group
    for v in (T[Inf, 1], T[-Inf, 2], T[Inf, NaN], T[NaN, Inf], T[NaN, 1])
      @test PB._residual_squares(v, nothing, T) === nothing
    end
    @test PB._residual_squares(T[Inf, 1], T[Inf, 0], T) === nothing
    @test PB._residual_squares(T[Inf, 1], T[0, 0], T) === nothing
    @test PB._residual_squares(T[0.0, -0.0], nothing, T) === (zero(T), 0)
    @test PB._residual_squares(T[1.0, 3.0], nothing, T) isa Tuple{T, Int}
  end

  @testset "exponent of the entries (y = 0)" begin
    # the method for y = 0 gives the exponent of the general method with y = 0 for the finite data below (random
    # values over all binades, with signed zeros, subnormals and entries near floatmax) and the same class: the
    # general code for an entry that is not finite is typemax(Int), which is the NaN code or the Inf code here
    es = exponent(nextfloat(zero(T)))
    specials = T[-zero(T), zero(T), nextfloat(zero(T)), -nextfloat(zero(T)), prevfloat(floatmin(T)), floatmin(T), floatmax(T), -floatmax(T),
                 prevfloat(floatmax(T)), T(1), T(-1), T(0.75)]
    for R in unique((T, Float64)), trial in 1:600
      n = rand(rng, 1:17)
      x = T[signif(rng, T, rand(rng, es:E + 1)) for _ in 1:n]
      for _ in 1:rand(rng, 0:4)
        x[rand(rng, 1:n)] = rand(rng, specials)
      end
      @test PB._square_exponent(x, nothing, R) == PB._square_exponent(x, zeros(R, n), R)
      # one entry that is not finite: the class decides, and a NaN wins at any position
      p = rand(rng, 1:n)
      for (bad, code) in ((T(Inf), PB._EXPONENT_INF), (-T(Inf), PB._EXPONENT_INF), (T(NaN), PB._EXPONENT_NAN), (-T(NaN), PB._EXPONENT_NAN))
        y = copy(x)
        y[p] = bad
        @test PB._square_exponent(y, zeros(R, n), R) == typemax(Int)
        @test PB._square_exponent(y, nothing, R) == code
        q = rand(rng, 1:n)
        y[q] = T(NaN)
        @test PB._square_exponent(y, nothing, R) == PB._EXPONENT_NAN
        y[q] = T(Inf)
        @test PB._square_exponent(y, nothing, R) == (any(isnan, y) ? PB._EXPONENT_NAN : PB._EXPONENT_INF)
      end
    end
    for R in unique((T, Float64)), x in (T[-0.0, 0.0], T[], T[floatmax(T), -floatmax(T)], fill(nextfloat(zero(T)), 5), T[3], T[-0.0, 0.75, -0.5],
                                          T[Inf, NaN], T[NaN, Inf], T[Inf, Inf], T[-Inf, 2], T[2, -Inf, NaN, 4])
      g = PB._square_exponent(x, zeros(R, length(x)), R)
      v = PB._square_exponent(x, nothing, R)
      all(isfinite, x) ? @test(v == g) : @test(g == typemax(Int) && v >= PB._EXPONENT_INF)
    end
    # the adapter, in the entries' type and rounded to another, and the stored values of sparse iterates
    xd = T[signif(rng, T, rand(rng, es:E + 1)) for _ in 1:10]
    G = [7, 3, 10, 1, 5, 9]
    for R in unique((T, Float64))
      @test PB._square_exponent(PB._IndexedValues(xd, G), nothing, R) == PB._square_exponent(xd[G], zeros(R, length(G)), R)
    end
    xs = PB.SparseIterate(T, 10)
    xs[2], xs[5], xs[8] = floatmax(T), T(-3), -zero(T)
    @test PB._square_exponent(xs, nothing, Float64) == PB._square_exponent(xs.nzval[1:xs.nnz], zeros(xs.nnz), Float64)
    S = PB.SymmetricSparseIterate(T, 4)
    S[1, 1], S[3, 2], S[4, 1] = T(2), T(-6), -zero(T)
    @test PB._square_exponent(S, nothing, Float64) == PB._square_exponent(S.nzval[1:S.nnz], zeros(S.nnz), Float64)
    S[2, 2] = T(Inf)
    @test PB._square_exponent(S, nothing, Float64) == PB._EXPONENT_INF
    S[1, 1] = T(NaN)
    @test PB._square_exponent(S, nothing, Float64) == PB._EXPONENT_NAN
  end

  @testset "two-term power-of-two sum" begin
    # the method for two terms performs the operations of the generic method: the same bits for cancellation to zero,
    # equal exponents, exponents far apart, either sign, zero terms, non-finite terms and a weight w
    generic(R, w, t1, t2) = invoke(PB._pow2_weighted, Tuple{Type, Any, Vararg{Tuple{Any, Int}}}, R, w, t1, t2)
    es = exponent(nextfloat(zero(T)))
    for R in unique((T, Float64))
      terms = Tuple[]
      for _ in 1:2500
        m1 = signif(rng, R, rand(rng, -3:3)) * (rand(rng) < 0.1 ? 3 : 1)
        k = rand(rng, (0, 0, 1, 2, 30, 200, -200, 2 * E, -2 * E, E + 30, -(E + 30), 3 * E))
        m2 = rand(rng) < 0.25 ? -m1 : rand(rng) < 0.25 ? m1 : signif(rng, R, rand(rng, -3:3))
        push!(terms, ((m1, rand(rng, -E:E)), (m2, rand(rng, -E:E) + k)))
      end
      # exponents that differ by at most 3, either order
      for _ in 1:1500
        e = rand(rng, -E:E)
        push!(terms, ((signif(rng, R, rand(rng, -2:2)), e), (signif(rng, R, rand(rng, -2:2)), e + rand(rng, -3:3))))
      end
      # exactly cancelling terms, with unnormalized significands, and a remainder that is below the frame
      for a in (R(0.75), R(1.5), R(3), R(-0.625)), ea in (-40, 0, 7, E + 3)
        push!(terms, ((a, ea), (-a, ea)))
        push!(terms, ((a, ea), (-a * 2, ea - 1)))
        push!(terms, ((a, ea), (nextfloat(-a), ea)))
        push!(terms, ((a, ea), (zero(R), 0)))
        push!(terms, ((zero(R), 0), (a, ea)))
        push!(terms, ((zero(R), 0), (-zero(R), 5)))
        push!(terms, ((R(Inf), ea), (a, 3)))
        push!(terms, ((a, 3), (R(NaN), ea)))
        push!(terms, ((R(Inf), 0), (-R(Inf), 0)))
        push!(terms, ((R(-Inf), 0), (-R(Inf), 4)))
      end
      for (t1, t2) in terms, w in (one(R), R(3) / 7, R(2)^(E ÷ 2), zero(R), R(Inf))
        t1 = (t1[1], Int(t1[2])); t2 = (t2[1], Int(t2[2]))
        a = PB._pow2_weighted(R, w, t1, t2)
        b = generic(R, w, t1, t2)
        @test a === b
      end
      @test PB._pow2_sum(R, (R(3), 5), (R(1), 5)) === PB._pow2_weighted(R, one(R), (R(3), 5), (R(1), 5))
      # and against the exact value, so that the generic method is not the only reference
      for (t1, t2) in terms[1:300]
        all(isfinite, (t1[1], t2[1])) || continue
        r = big(t1[1]) * big(2.0)^t1[2] + big(t2[1]) * big(2.0)^t2[2]
        (isfinite(R(r)) && (iszero(r) || abs(r) >= big(nextfloat(zero(R))) * 4)) || continue
        @test within(PB._pow2_sum(R, t1, t2), r, 2)
      end
    end
  end

  @testset "scaled sum without a second operand" begin
    # y === nothing gives the bits of the general method with y = 0, for finite data over many binades,
    # signed zeros, subnormals and entries near floatmax, in the entries' type and in Float64
    es = exponent(nextfloat(zero(T)))
    specials = T[-zero(T), zero(T), nextfloat(zero(T)), -nextfloat(zero(T)), prevfloat(floatmin(T)), floatmax(T), -floatmax(T),
                 prevfloat(floatmax(T)), floatmin(T)]
    for R in unique((T, Float64)), trial in 1:400
      n = rand(rng, 1:17)
      x = T[signif(rng, T, rand(rng, es:E + 1)) for _ in 1:n]
      for _ in 1:rand(rng, 0:3)
        x[rand(rng, 1:n)] = rand(rng, specials)
      end
      k = PB._square_exponent(x, nothing, R)
      k == PB._EXPONENT_ZERO && continue
      for kk in (k, k + 1, k + 20)
        @test PB._square_scaled_sum(x, nothing, kk, R) === PB._square_scaled_sum(x, zeros(R, n), kk, R)
      end
    end
    for R in unique((T, Float64))
      for x in (T[-0.0, 0.0, -0.0, 1], T[floatmax(T), -floatmax(T), floatmax(T) / 2], fill(nextfloat(zero(T)), 5), T[3])
        k = PB._square_exponent(x, nothing, R)
        @test PB._square_scaled_sum(x, nothing, k, R) === PB._square_scaled_sum(x, zeros(R, length(x)), k, R)
      end
    end
    # the adapter and the stored values of a sparse iterate are read in order, as a vector of the same entries
    xd = T[signif(rng, T, rand(rng, es:E + 1)) for _ in 1:10]
    G = [7, 3, 10, 1, 5, 9]
    k = PB._square_exponent(PB._IndexedValues(xd, G), nothing, Float64)
    @test PB._square_scaled_sum(PB._IndexedValues(xd, G), nothing, k, Float64) === PB._square_scaled_sum(xd[G], zeros(length(G)), k, Float64)
    xs = PB.SparseIterate(T, 10)
    xs[2], xs[5], xs[8] = floatmax(T), T(-3), -zero(T)
    vals = xs.nzval[1:xs.nnz]
    k = PB._square_exponent(xs, nothing, Float64)
    @test PB._square_scaled_sum(xs, nothing, k, Float64) === PB._square_scaled_sum(vals, zeros(length(vals)), k, Float64)
  end

  @testset "scaled norm with a scale" begin
    cases = ((fill(pow2(T, E - 2), 4), nothing), (fill(pow2(T, e0 + 1), 3), nothing),
             (T[T(0.6) * floatmax(T), 1], T[T(-0.1) * floatmax(T), 1]),
             (T[3, -4, 12] .* pow2(T, E - 6), nothing), (T[3, -4, 12] .* pow2(T, e0 - 12), nothing),
             (T[1, 1, 1], T[1, 1, 1 + eps(T)]))
    for (x, y) in cases, R in unique((T, Float64))
      q = PB._residual_squares(x, y, R)
      @test q isa Tuple{R, Int}
      m, k = q
      s = zero(R)   # the ordinary sum is used only in the branch of a residual that is not finite
      # the default scale is the norm of the baseline, one rounding of √m 2^k
      @test PB._scaled_norm(x, y, s) === PB._scaled_product(sqrt(m), one(R), k)
      @test PB._scaled_norm(x, y, s, one(R)) === PB._scaled_product(one(R), sqrt(m), k)
      R === T && y !== nothing && @test norm_diff(x, y) === PB._scaled_product(sqrt(m), one(R), k)
      # a scale λ multiplies the norm in the same rounding
      for λ in (R(3) / 7, pow2(R, -E ÷ 2), pow2(R, E ÷ 2), zero(R), R(Inf))
        @test PB._scaled_norm(x, y, s, λ) === PB._scaled_product(λ, sqrt(m), k)
      end
      r = sqrt(sum(abs2, big.(x) .- (y === nothing ? 0 : big.(y))))
      for λ in (R(3) / 7, R(2))
        @test within(PB._scaled_norm(x, y, s, λ), big(λ) * r, 2)
      end
    end
    # an entry that is not finite keeps the ordinary result: the scale times √s
    for R in unique((T, Float64)), λ in (R(2), R(0.25), one(R)), (x, y, s) in ((T[Inf, 1], nothing, R(Inf)), (T[-Inf, 1], T[0, 1], R(Inf)))
      @test PB._scaled_norm(x, y, s, λ) === λ * sqrt(s) === R(Inf)
      @test PB._scaled_norm(x, y, s) === sqrt(s)
    end
    for R in unique((T, Float64)), λ in (R(2), zero(R), one(R)), (x, y) in ((T[NaN, 1], nothing), (T[Inf], T[Inf]), (T[Inf, NaN], nothing))
      @test isnan(PB._scaled_norm(x, y, R(NaN), λ)) && isnan(λ * sqrt(R(NaN)))
    end
    @test isnan(PB._scaled_norm(T[Inf, 1], nothing, Float64(Inf), 0.0))      # 0 * Inf
  end

  @testset "threshold product" begin
    for P in unique((T, Float64))
      R = promote_type(T, P)
      Er, er = emax(R), emin(R)
      expected(γ, λ, w) = convert(T, (convert(R, γ) * convert(R, λ)) * convert(R, w))
      # ordinary range: the product γλw rounded once to T, formed in the working type, and a true flag
      for _ in 1:500
        γ, λ, w = (abs(signif(rng, P, rand(rng, -30:30))) for _ in 1:3)
        c, ok = PB._threshold_product(T, γ, λ, w)
        @test c === expected(γ, λ, w) && ok
      end
      # γλ = 2^160 is normal in Float64 and overflows Float32: the flag follows the working type, while c is
      # rounded once to T (Inf for Float32 data)
      γ, λ, w = pow2(R, 100), pow2(R, 60), pow2(R, 30)
      c, ok = PB._threshold_product(T, γ, λ, w)
      @test c === expected(γ, λ, w) && ok == (R !== Float32)
      # the flag is the range test of γλ in R: floatmin and floatmax are in range, outside it is false
      for (γ, λ, w, flag) in ((pow2(R, er), one(R), R(3), true), (floatmax(R), one(R), R(1) / 4, true),
                              (floatmax(R), R(4), R(1) / 8, false), (floatmax(R), floatmax(R), R(1), false),
                              (pow2(R, er), R(1) / 4, R(8), false), (prevfloat(floatmin(R)), one(R), R(2), false),
                              (zero(R), R(3), R(2), false), (R(3), zero(R), R(2), false),
                              (pow2(R, -Er + 4), pow2(R, -Er + 4), pow2(R, Er), false),
                              (pow2(R, Er - 2), pow2(R, 3), pow2(R, -Er), false))
        c, ok = PB._threshold_product(T, P(γ), P(λ), P(w))
        @test c === expected(P(γ), P(λ), P(w)) && ok == flag
      end
    end
    # the weight true of an unweighted entry does not widen the type
    c, ok = PB._threshold_product(T, T(0.5), T(3), true)
    @test c === T(1.5) && ok && c isa T
  end

  @testset "kept sums" begin
    # a sum of n terms below n floatmin may have lost more than ordinary rounding (#373); NaN and Inf are kept
    n = 5
    @test PB._sum_kept(T(1), n) && PB._sum_kept(n * floatmin(T), n)
    @test !PB._sum_kept(prevfloat(n * floatmin(T)), n) && !PB._sum_kept(zero(T), n)
    @test PB._sum_kept(T(NaN), n) && PB._sum_kept(T(Inf), n)
  end

  @testset "ordinary sum-of-squares trigger" begin
    n = 7
    t = n * (floatmin(T) / eps(T))
    @test PB._squares_normal(t, n, T)
    @test !PB._squares_normal(prevfloat(t), n, T)
    @test PB._squares_normal(T(1), n, T)
  end

  @testset "scaled square value and parts" begin
    # λ/2 Σ (xᵢ - yᵢ)² where the squares overflow or underflow and λ restores the range
    for (λ, x, y) in ((pow2(T, -E + 2), fill(pow2(T, E - 2), 4), nothing),
                      (pow2(T, E - 4), fill(pow2(T, e0 ÷ 2 - 2), 4), nothing),
                      (pow2(T, -E), [floatmax(T), T(2)], [-floatmax(T), T(1)]))
      r = big(λ) * sum(abs2, big.(x) .- (y === nothing ? 0 : big.(y))) / 2
      @test within(PB._square_value(λ, x, y), r, 2)
      m, e = PB._square_parts(λ, x, y)
      @test 0.5 <= abs(m) < 1
      @test relerr(big(m) * big(2.0)^e, r) <= 4eps(T)
    end
    # genuine nonfinite entries keep the ordinary value; a zero or infinite level has no parts
    @test PB._square_value(T(1), T[Inf, 1], nothing) === T(Inf)
    @test isnan(PB._square_value(T(1), T[NaN, 1], nothing))
    @test PB._square_parts(T(1), T[Inf, 1], nothing) === nothing
    @test PB._square_parts(zero(T), T[1, 1], nothing) === nothing
    @test PB._square_parts(T(Inf), T[1, 1], nothing) === nothing
    @test PB._square_parts(T(2), zeros(T, 2), nothing) === (zero(T), 0)
  end

  @testset "SqrNormL2 on the shared kernels" begin
    # the gradient λ(x - c) where x - c overflows and λ restores the range; the value and the fused
    # oracle on the same data
    for _ in 1:300
      x, c = abs(signif(rng, T, E + 1)), -abs(signif(rng, T, rand(rng, E:E+1)))   # x - c overflows
      λ = abs(signif(rng, T, rand(rng, -E:-2)))
      r = big(λ) * (big(x) - big(c))
      abs(r) <= floatmax(T) || continue
      f = SqrNormL2(λ, [c])
      g = gradient!(zeros(T, 1), f, [x])
      @test within(g[1], r, 2)
      g2 = zeros(T, 1)
      v = value_and_gradient!(g2, f, [x])
      @test g2 == g && v === f([x])
      @test within(v, big(λ) * (big(x) - big(c))^2 / 2, 3)
    end
    # a genuine infinite level keeps the IEEE gradient λ(x - c), also for a subnormal residual,
    # which halving would have rounded to zero (NaN on master)
    δ = nextfloat(zero(T))
    g = gradient!(zeros(T, 3), SqrNormL2(T(Inf), T[0, 0, 1]), T[-δ, 3δ, 1])
    @test g[1] === T(-Inf) && g[2] === T(Inf) && isnan(g[3])
    # centers are rounded once as the value rounds them, in the value and in the value parts:
    # an entry that leaves T's range on rounding is read unrounded, the others are rounded to T
    W = T === Float32 ? Float64 : BigFloat
    # the second entry rounds to exactly 1 in T: read unrounded, its residual would dominate the value
    c = W[ldexp(W(1), e0 - 4), 1 + ldexp(W(1), -(precision(T) + 20))]
    λ = pow2(T, E)
    x = T[0, 1]
    cr = [big(c[1]), big(T(c[2]))]
    ref = big(λ) * sum(abs2, big.(x) .- cr) / 2
    f = SqrNormL2(λ, c)
    @test within(f(x), ref, 2)
    α = pow2(T, -E ÷ 2)
    @test within(Scaled(f, α)(x), big(α) * ref, 2)
    λs = pow2(T, -E + 2)
    xs = T[0, 0]
    ref_small = big(λs) * sum(abs2, big.(xs) .- cr) / 2
    @test within(Scaled(SqrNormL2(λs, c), pow2(T, E - 1))(xs), big(pow2(T, E - 1)) * ref_small, 2)
    # stored iterates: O(nnz) reductions with the logical multiplicity, on the recovery path
    xsp = SparseIterate(T, 6)
    xsp[2], xsp[5] = pow2(T, E - 1), T(-3)
    xm = SymmetricSparseIterate(T, 3)
    xm[2, 1], xm[3, 3] = pow2(T, E - 1), T(-0.75) * pow2(T, E - 1)
    λt = pow2(T, -E + 1)
    for z in (xsp, xm)
      dense = z isa SparseIterate ? Vector(z) : Matrix(z)
      @test within(SqrNormL2(λt)(z), big(λt) * sum(abs2, big.(dense)) / 2, 2)
      @test SqrNormL2(λt)(z) === SqrNormL2(λt)(dense)
    end
    # prox: the coordinate and array methods, and the in-place alias, agree at exceptional scales
    for (γ, λp) in ((pow2(T, E - 1), pow2(T, E - 1)), (pow2(T, -E), pow2(T, e0)), (T(1), T(3)))
      x = T[pow2(T, E), -pow2(T, e0), T(0.3)]
      cc = T[-pow2(T, E - 1), pow2(T, e0), T(1)]
      for f in (SqrNormL2(λp), SqrNormL2(λp, cc))
        z = prox!(similar(x), f, x, γ)
        @test z == [prox_block(f, x[i], i, γ) for i in eachindex(x)]
        xa = copy(x)
        @test prox!(xa, f, xa, γ) == z
      end
    end
  end

  @testset "recomputed-dot units" begin
    # no partial sum of n products, scaled by 2^-k, can reach 2^(emax - margin)
    for _ in 1:500
      ma, mb = abs(signif(rng, T, rand(rng, e0:E))), abs(signif(rng, T, rand(rng, e0:E)))
      n, margin = rand(rng, 1:1000), rand(rng, 2:3)
      k = PB._product_sum_shift(T, ma, mb, n, margin)
      @test k >= 0
      @test big(n) * big(ma) * big(mb) * big(2.0)^(-k) < big(2.0)^(E + 1 - margin)
      k > 0 && @test big(n) * big(ma) * big(mb) * big(2.0)^(1 - k) >= big(2.0)^(E - 1 - margin - 2)
    end
    @test PB._product_sum_shift(T, T(1), T(1), 1, 2) == 0
    @test PB._product_sum_shift(T, T(Inf), T(1), 10, 2) == 0
  end

  @testset "LeastSquaresLoss and LinearFunction on the shared kernels" begin
    ref(Y, X, β) = sum(abs2, big.(Y) .- big.(X) * big.(β)) / 2
    h = E ÷ 2
    cases = Any[
      # residual squares overflow, value representable only as Inf or not: the reference decides
      (T[pow2(T, h + 2), 1], T[1 0; 0 1], T[0, 0]),
      # residual squares underflow, value restored by summing in power-of-two units
      (T[pow2(T, e0 ÷ 2 - 4), -pow2(T, e0 ÷ 2 - 3)], T[1 0; 0 1], T[0, 0]),
      # Y - Xβ overflows while Xβ is finite (halved residual)
      (T[floatmax(T), 1], T[1 0; 0 1], T[-floatmax(T) / 2, 0]),
      # Xβ overflows in its partial products although Xβ = 0 and the value is representable
      (T[3, -2], T[pow2(T, h + 4) pow2(T, h + 4); 1 1], T[pow2(T, h + 4), -pow2(T, h + 4)]),
      # an exact fit
      (T[pow2(T, E - 3), 1], T[1 0; 0 1], T[pow2(T, E - 3), 1]),
    ]
    for (Y, X, β) in cases
      f = LeastSquaresLoss(Y, X)
      r = ref(Y, X, β)
      v = f(β)
      @test within(v, r, 3)
      g = similar(β)
      @test within(value_and_gradient!(g, f, β), r, 3)
      @test isequal(g, gradient!(similar(β), f, β))   # the gradient itself has no recovery path
      bs = SparseIterate(T, length(β))
      for i in eachindex(β)
        iszero(β[i]) || (bs[i] = β[i])
      end
      @test f(bs) === v
    end
    # genuine NaN/Inf data keep the ordinary result
    f = LeastSquaresLoss(T[NaN, 1], T[1 0; 0 1])
    @test isnan(f(T[1, 1]))
    @test LeastSquaresLoss(T[Inf, 1], T[1 0; 0 1])(T[1, 1]) === T(Inf)
    @test isnan(LeastSquaresLoss(T[1, 1], T[1 0; 0 1])(T[Inf, -Inf]) - T(Inf)) # Inf or NaN, never finite
    # LinearFunction: an overflowing dot at a representable value, with d in its own units
    a = pow2(T, h + 4)
    for (c, x, d) in ((T[a, a, 1], T[a, -a, 1], T(1)), (T[floatmax(T), floatmax(T)], T[1, -1], T(0.25)),
                      (T[a, a], T[a, -a], pow2(T, e0)))
      rr = sum(big.(c) .* big.(x)) + big(d)
      @test within(LinearFunction(c, d)(x), rr, 1)
    end
    @test LinearFunction(T[1, 1], T(0))(T[Inf, 1]) === T(Inf)
    @test isnan(LinearFunction(T[1, 1], T(0))(T[Inf, -Inf]))
  end

  @testset "Euclidean norm recovery" begin
    function owner(vals, p)
      S = SymmetricSparseIterate(T, p)
      k = 0
      for c in 1:p, r in c:p
        k += 1
        k <= length(vals) && (S[r, c] = vals[k])
      end
      S
    end
    frob(S, p) = sqrt(sum(big(S[r, c])^2 for r in 1:p, c in 1:p; init = big(0.0)))
    # residuals whose squares overflow, underflow or are subnormal, and mixed scales; the
    # recomputation (a sum of squares that is not a finite normal number) is within 2 ulps
    es = exponent(nextfloat(zero(T)))
    for (lo, hi) in ((E ÷ 2 + 1, E - 4), (e0 ÷ 2 - 20, e0 ÷ 2 - 1), (es + 3, e0 - 1), (e0 + 2, E - 6))
      for _ in 1:300
        n = rand(rng, 1:30)
        x = T[signif(rng, T, rand(rng, lo:hi)) for _ in 1:n]
        y = T[signif(rng, T, rand(rng, lo:hi)) for _ in 1:n]
        rand(rng) < 0.2 && (y .= nextfloat.(x, rand(rng, -3:3)))    # cancellation
        r = sqrt(sum((big.(x) .- big.(y)) .^ 2))
        s = sum(abs2, x .- y)
        (isfinite(s) && s >= floatmin(T)) || @test within(norm_diff(x, y), r, 2)
        p = rand(rng, 1:7)
        S = owner(x[1:min(n, p * (p + 1) ÷ 2)], p)
        # every scale, O(nnz); the ordinary sum of m squares adds up to (m - 1)/4 ulps to the norm
        @test within(norm(S), frob(S, p), 2 + S.nnz / 4)
      end
    end
    # one residual whose square overflows, a finite norm; an overflowing difference is a norm beyond range
    fm = floatmax(T)
    @test within(norm_diff(T[0.6fm, 0], T[-0.1fm, 0]), big(0.7) * big(fm), 1)
    @test norm_diff(T[0.75fm], T[-0.5fm]) === T(Inf)
    S = owner(T[0, pow2(T, E - 2), 0], 2)                            # an off-diagonal entry counts twice
    @test within(norm(S), sqrt(big(2.0)) * big(pow2(T, E - 2)), 1)
    S = owner(T[pow2(T, es), -pow2(T, es + 1), pow2(T, es)], 2)       # subnormal entries only
    @test within(norm(S), sqrt(big(1.0) + 2 * 4 + 1) * big(pow2(T, es)), 1)
    @test (@allocated norm(S)) == 0
    # genuine NaN/Inf keep the ordinary result; exact and signed zeros give +0
    @test isnan(norm_diff(T[NaN, 1], T[0, 1])) && isnan(norm_diff(T[Inf, NaN], T[0, 0]))
    @test isnan(norm_diff(T[Inf], T[Inf]))                              # Inf - Inf
    @test norm_diff(T[Inf, pow2(T, e0 - 4)], T[0, 0]) === T(Inf)
    @test norm_diff(T[-0.0, 0], T[0, -0.0]) === zero(T)
    @test norm_diff(T[pow2(T, es), 1], T[pow2(T, es), 1]) === zero(T)
    @test isnan(norm(owner(T[pow2(T, E - 1), NaN], 2))) && norm(owner(T[Inf, pow2(T, es)], 2)) === T(Inf)
    @test norm(owner(T[-0.0, -0.0], 2)) === zero(T) && norm(SymmetricSparseIterate(T, 3)) === zero(T)
  end
end
end


# #767: the retained exponent of a _ProjectionScale is an Int that may already have wrapped. The
# retained products sum the exponents in Int, modulo 2^64, so they are right whenever the final
# exponent fits in Int. The ordinary kernel widens BigFloat exponents to Int128 and would read the
# wrapped power at face value. MPFR exponents near -2^62 let two factors carry the wrap back.
setprecision(BigFloat, 256) do
@testset "two-term power-of-two sum, BigFloat exponents beyond Int64" begin
  # MPFR exponents reach 2^62 - 1: the frame exponents of the sum are taken in Int128, as in the generic method
  generic(R, w, t1, t2) = invoke(PB._pow2_weighted, Tuple{Type, Any, Vararg{Tuple{Any, Int}}}, R, w, t1, t2)
  big_e = (2^62 - 3, 2^61, 2^40, 5, 0, -5, -2^40, -2^61, -(2^62 - 3))
  ms = (BigFloat(0.75), BigFloat(-0.75), BigFloat(1.5), BigFloat(3) / 7, BigFloat(0))
  count = 0
  for m1 in ms, m2 in ms, e1 in big_e, e2 in big_e, w in (BigFloat(1), BigFloat(0.3))
    t1, t2 = (m1, e1), (m2, e2)
    @test bigsame(PB._pow2_weighted(BigFloat, w, t1, t2), generic(BigFloat, w, t1, t2))
    count += 1
  end
  @test count == 5 * 5 * 9 * 9 * 2
  # a value in range through terms with exponents beyond Int64 in sum, and the overflow of the weight
  big2 = BigFloat(2)
  @test PB._pow2_sum(BigFloat, (BigFloat(0.5), 2^62 - 3), (BigFloat(-0.5), 2^62 - 3)) == 0
  v = PB._pow2_sum(BigFloat, (BigFloat(0.75), 2^62 - 3), (BigFloat(0.75), -(2^62 - 3)))
  @test v == BigFloat(0.75) * big2^(2^62 - 3)
end

@testset "two-term power-of-two sum under directed rounding (BigFloat)" begin
  # Only the IEEE types skip the shift of a term far below the sum (they round to nearest only). A BigFloat does the
  # sum of the generic method in every rounding mode: under RoundUp, RoundDown and RoundToZero a term below half an ulp
  # still moves the result. The precisions and gaps below straddle the skip bound 2 precision for the types that have it.
  generic(R, w, t1, t2) = invoke(PB._pow2_weighted, Tuple{Type, Any, Vararg{Tuple{Any, Int}}}, R, w, t1, t2)
  ntests = 0
  for p in (256, 512), mode in (RoundNearest, RoundUp, RoundDown, RoundToZero)
    setprecision(BigFloat, p) do
      setrounding(BigFloat, mode) do
        for gap in (1, 30, 255, 256, 257, 511, 512, 513, 600, 1023, 1024, 1025, 1112, 4000), sgn in (1, -1),
            (m1, m2) in ((BigFloat(0.75), BigFloat(0.75)), (BigFloat(0.75), BigFloat(0.5)), (BigFloat(1) / 3, BigFloat(2) / 3)),
            e in (0, 7, -50, 2^40), w in (BigFloat(1), BigFloat(3) / 7)
          t1, t2 = (m1, e), (sgn * m2, e - gap)
          for (u, v) in ((t1, t2), (t2, t1), ((-m1, e), t2))
            a = PB._pow2_weighted(BigFloat, w, u, v)
            ntests += 1
            @test bigsame(a, generic(BigFloat, w, u, v))
          end
        end
        # the example of the review: 0.75 + 0.75 2^-600 at 256 bits is above 0.75 under RoundUp, and
        # 0.75 - 0.75 2^-600 is below it under RoundDown and RoundToZero
        if p == 256
          up = PB._pow2_sum(BigFloat, (BigFloat(0.75), 0), (BigFloat(0.75), -600))
          dn = PB._pow2_sum(BigFloat, (BigFloat(0.75), 0), (BigFloat(-0.75), -600))
          @test up == (mode === RoundUp ? nextfloat(BigFloat(0.75)) : BigFloat(0.75))
          @test dn == (mode in (RoundDown, RoundToZero) ? prevfloat(BigFloat(0.75)) : BigFloat(0.75))
        end
      end
    end
  end
  @test ntests == 2 * 4 * 14 * 2 * 3 * 4 * 2 * 3
  # significands stored above the current precision: frexp rounds, and the frexp exponent of the second term can exceed
  # that of the first (D > E); the sum is still that of the generic method, in every mode
  v = setprecision(() -> 1 - ldexp(BigFloat(1), -258), BigFloat, 768)
  @test precision(v) == 768
  for mode in (RoundNearest, RoundUp, RoundDown, RoundToZero)
    setprecision(BigFloat, 256) do
      setrounding(BigFloat, mode) do
        for (t1, t2) in (((BigFloat(0.75), 0), (v, 0)), ((v, 0), (BigFloat(0.75), 0)), ((v, 3), (-v, 3)), ((v, 0), (ldexp(v, -300), 0)))
          @test bigsame(PB._pow2_weighted(BigFloat, BigFloat(1), t1, t2), generic(BigFloat, BigFloat(1), t1, t2))
        end
      end
    end
  end
  # the public caller of the review: the group screen ‖g‖ - λw with λw 2^-600 of ‖g‖, g = 2^e, e = -2^61: the exact
  # value is g (1 - 2^-600), which rounds to g to nearest and upward, and to prevfloat(g) downward and toward zero
  for (mode, down) in ((RoundNearest, false), (RoundUp, false), (RoundDown, true), (RoundToZero, true)), p in (256, 512)
    setprecision(BigFloat, p) do
      setrounding(BigFloat, mode) do
        e = -2^61
        g = [ldexp(BigFloat(1), e)]
        f = GroupNormL2(ldexp(BigFloat(1), e - (p == 256 ? 600 : 1112)), [1:1], [BigFloat(1)])
        r = kkt_violation(f, g, zeros(BigFloat, 1), 1)
        @test r == (down ? prevfloat(g[1]) : g[1])
      end
    end
  end
  # the IEEE types have no rounding mode to set (the skip is exact for them): there is no setrounding method
  @test_throws MethodError setrounding(() -> 1.0, Float64, RoundUp)
end

@testset "retained exponent across Int wraparound (#767)" begin
  E = exponent(floatmax(BigFloat))
  a, b = ldexp(big(0.75), -E + 10), ldexp(big(0.625), -E + 10)
  (ma, ea), (mb, eb) = PB._frexp(a), PB._frexp(b)
  # a small third factor gives negative final exponents with the same wrapped power
  for d in (0, 40, 1000), c in (ldexp(big(0.875), 3), ldexp(big(0.875), -3000))
    mc, ec = PB._frexp(c)
    K = Int128(typemax(Int)) + 1 + d          # the true power: not an Int
    kw = K % Int                              # as a _ProjectionScale stores it, wrapped
    final3, final2 = Int128(ea) + eb + ec + K, Int128(ea) + eb + 1 + K   # c = 1 has exponent 1
    # the intended boundary: the power is not an Int, the widened sum of the stored exponents
    # leaves Int (so widening gives the wrong exponent), and the true final exponent fits in Int
    inint(e) = typemin(Int) <= e <= typemax(Int)
    @test !inint(K)
    @test !inint(Int128(ea) + eb + ec + kw) && !inint(Int128(ea) + eb + 1 + kw)
    @test inint(final3) && inint(final2)
    ref3 = (ma * mb * mc) * big(2.0)^Int(final3)
    ref2 = (ma * mb / 2) * big(2.0)^Int(final2)
    @test PB._scaled_product_retained(a, b, c, kw) == ref3
    @test PB._scaled_product_retained(a, b, kw) == ref2
    # the live caller: a scale m 2^e applied to parts whose exponent is added to the retained one
    @test PB._apply_parts(PB._ProjectionScale(a, kw), b, 0) == ldexp(a, -ea) * ldexp(b, -eb) * big(2.0)^Int(Int128(ea) + eb + K)
    # widening the wrapped power, as the ordinary kernel does, loses the value
    @test PB._scaled_product(a, b, c, kw) != ref3
  end
end
end

end
