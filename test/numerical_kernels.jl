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
# a vector with the axis 0:n-1 whose getindex always checks its bounds
struct ZeroBasedVector{T} <: AbstractVector{T}
  p::Vector{T}
end
Base.axes(v::ZeroBasedVector) = (Base.IdentityUnitRange(0:length(v.p)-1),)
Base.size(v::ZeroBasedVector) = (length(v.p),)
Base.IndexStyle(::Type{<:ZeroBasedVector}) = IndexLinear()
function Base.getindex(v::ZeroBasedVector, i::Int)
  checkbounds(v, i)
  v.p[i + 1]
end
# a vector whose axis ends at typemax(Int), with a getindex that always checks its bounds
struct EndAxis{T} <: AbstractVector{T}
  data::Vector{T}
end
Base.size(x::EndAxis) = size(x.data)
Base.axes(x::EndAxis) = (Base.IdentityUnitRange((typemax(Int) - length(x.data) + 1):typemax(Int)),)
Base.IndexStyle(::Type{<:EndAxis}) = IndexLinear()
function Base.getindex(x::EndAxis, i::Int)
  checkbounds(x, i)
  x.data[i - firstindex(x) + 1]
end
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

  @testset "two-term exponent sum" begin
    p = precision(T)
    es = exponent(nextfloat(zero(T)))
    # Both orders of finite terms, including zeros, exact and near cancellation,
    # and results at the normal, subnormal and overflow boundaries.
    cases = (
      ((zero(T), E), (-zero(T), es)),
      ((zero(T), E), (T(0.75), 3)),
      ((T(0.75), E + 2), (-T(0.75), E + 2)),
      ((T(0.5), E + 2), (-prevfloat(one(T)), E + 1)),
      ((T(0.75), 3), (T(0.75), 3 - 2E)),
      ((T(-0.75), 3), (T(0.75), 3 - 2E)),
      ((T(0.5), es), (T(0.5), es)),
      ((T(0.5), es - 1), (T(0.5), es - 1)),
      ((T(0.5), e0), (T(0.5), e0)),
      ((prevfloat(one(T)), e0), (zero(T), 0)),
      ((prevfloat(one(T)), E + 1), (zero(T), 0)),
      ((prevfloat(one(T)), E + 1), (T(0.5), E + 1 - p)),
      ((T(0.5), E + 1), (T(0.5), E + 1)),
    )
    setprecision(BigFloat, 4096) do
      check_sum(t1, t2) = begin
        acc = PB._expsum_add(PB._expsum_start(T), t1...)
        acc = PB._expsum_add(acc, t2...)
        ref = ldexp(BigFloat(t1[1]), t1[2]) + ldexp(BigFloat(t2[1]), t2[2])
        @test PB._expsum_value(T, acc) === T(ref)
      end
      for (t1, t2) in cases
        check_sum(t1, t2)
        check_sum(t2, t1)
      end
      rng_sum = MersenneTwister(31)
      for _ in 1:300
        e = rand(rng_sum, e0 + p:E - 1)
        m1, m2 = signif(rng_sum, T, 0), signif(rng_sum, T, 0)
        check_sum((m1, e), (m2, e + rand(rng_sum, -2E:1)))
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

  @testset "shrinkage kernel at exceptional parameters and values" begin
    # data whose sum of squares is not a finite normal number (overflowing, underflowing, subnormal, NaN or Inf entries):
    # with any step, level and weight, positive and finite or not, the shrinkage is the cold one, so it gives the bits of
    # the cold route (the exits that the parameters decide, _l2_shrink_fixed!, then the values, _l2_shrink_values!) in its
    # code and in the group it writes, with the exits called (the default) or inlined (prox_block!)
    fm, fn = floatmax(T), floatmin(T)
    params = (zero(T), 4fn, T(0.7), fm / 4, T(Inf))
    datas = (T[pow2(T, E - 3), -pow2(T, E - 4), 0, 3], T[pow2(T, e0 + 2), pow2(T, e0 + 3), 0], T[1, NaN, 2], T[1, Inf, 2],
             T[1, -Inf, NaN], fill(nextfloat(zero(T)), 4), zeros(T, 3))
    cold!(y, x, G, λ, w, γ) = (iszero(w) | isinf(w) | isinf(λ) | iszero(λ) | iszero(γ) | isinf(γ)) ?
      PB._l2_shrink_fixed!(y, x, G, λ, w, γ) : PB._l2_shrink_values!(y, x, G, λ, w, γ)
    ntests = 0
    for x in datas, γ in params, λ in params, w in params
      G = 1:length(x)
      y1, y2, y3 = fill(T(7), length(x)), fill(T(7), length(x)), fill(T(7), length(x))
      @test PB._l2_shrink!(y1, x, G, λ, w, γ) === T(cold!(y2, x, G, λ, w, γ))
      @test same(y1, y2)
      @test PB._l2_shrink!(y3, x, G, λ, w, γ, PB._l2_shrink_fixed!) === T(cold!(fill(T(7), length(x)), x, G, λ, w, γ))
      @test same(y3, y2)
      z1, z2 = copy(x), copy(x)
      @test PB._l2_shrink!(z1, z1, G, λ, w, γ) === T(cold!(z2, z2, G, λ, w, γ))
      @test same(z1, z2)
      ntests += 1
    end
    @test ntests == length(datas) * 5^3
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
    # The value and the value parts use the center rounded to T.
    W = T === Float32 ? Float64 : BigFloat
    # the second entry rounds to exactly 1 in T: read unrounded, its residual would dominate the value
    c = W[ldexp(W(1), e0 - 4), 1 + ldexp(W(1), -(precision(T) + 20))]
    λ = pow2(T, E)
    x = T[0, 1]
    cr = big.(T.(c))
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
    @test alloc(norm, S) == 0    # behind a function barrier: inline, 1.10 counts the boxed result
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


setprecision(BigFloat, 256) do
@testset "group screen under directed rounding (BigFloat)" begin
  # the group screen ‖g‖ - λw through the public kkt_violation, with g = 2^e, e = -((emax + 1) ÷ 2) for
  # emax = Base.MPFR.get_emax() (-2^61 with a 64-bit C `long`), and λw = 2^-600 ‖g‖ at precision 256 and
  # 2^-1112 ‖g‖ at precision 512: the exact value is g (1 - 2^-600) or g (1 - 2^-1112) respectively, which rounds to g
  # to nearest and upward, and to prevfloat(g) downward and toward zero
  for (mode, down) in ((RoundNearest, false), (RoundUp, false), (RoundDown, true), (RoundToZero, true)), p in (256, 512)
    setprecision(BigFloat, p) do
      setrounding(BigFloat, mode) do
        e = -((Int(Base.MPFR.get_emax()) + 1) ÷ 2)
        g = [ldexp(BigFloat(1), e)]
        f = GroupNormL2(ldexp(BigFloat(1), e - (p == 256 ? 600 : 1112)), [1:1], [BigFloat(1)])
        r = kkt_violation(f, g, zeros(BigFloat, 1), 1)
        @test r == (down ? prevfloat(g[1]) : g[1])
      end
    end
  end
end

# #767: the retained exponent of a _ProjectionScale is an Int that may already have wrapped. The
# retained products sum the exponents in Int, modulo 2^64, so they are right whenever the final
# exponent fits in Int. The ordinary kernel widens BigFloat exponents to Int128 and would read the
# wrapped power at face value. MPFR exponents near -2^62 let two factors carry the wrap back.
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
# The fast paths of the group kernels agree with the definitions they shorten.
@testset "group kernel fast paths: $T" for T in (Float32, Float64)
  rng = MersenneTwister(901)
  @testset "powers of two from bits equal the split ldexp factors" begin
    for k in -1200:1200
      @test PB._power_factors(T, k) === PB._power_factors_split(T, k)
    end
  end
  @testset "all-zero scan in blocks" begin
    for n in (0, 1, 5, 511, 512, 513, 1024, 1500, 5000)
      z = zeros(T, n)
      @test PB._all_zero_early(z) == PB._all_zero(z) == true
      n == 0 && continue
      for pos in unique((1, n, (n + 1) ÷ 2, min(n, 513), min(n, 512))), v in (T(1), -T(1), T(NaN), T(Inf), nextfloat(zero(T)), -zero(T))
        w = copy(z); w[pos] = v
        @test PB._all_zero_early(w) == PB._all_zero(w) == (v === -zero(T))
      end
    end
    @test PB._all_zero_early(Int[0, 0]) && !PB._all_zero_early(Int[0, 1])
  end
  @testset "all-zero scans on axes that do not start at one" begin
    # the scans read the indices of the axes: an index outside them throws (the check below is not elided by @inbounds)
    for n in (1, 3, 513), pos in unique((0, n - 1))
      v = zeros(T, n); v[pos + 1] = nextfloat(zero(T))
      z = ZeroBasedVector(v)
      @test !PB._all_zero(z) && !PB._all_zero_early(z)
      @test PB._all_zero(ZeroBasedVector(zeros(T, n))) && PB._all_zero_early(ZeroBasedVector(zeros(T, n)))
    end
    # the public values refuse an array whose axes do not start at 1, before any scan
    x = ZeroBasedVector(T[floatmin(T), 0, 0])
    @test_throws ArgumentError NormL2(one(T))(x)
    @test_throws ArgumentError NormL1(floatmin(T))(x)
    @test_throws ArgumentError NormL1(floatmin(T))(ZeroBasedVector(zeros(T, 3)))
  end
  @testset "all-zero scans on an axis that ends at typemax(Int)" begin
    # no block end or block start passes the last index, typemax(Int) here: an index outside the axis throws
    for n in (1, 512, 513, 1025)
      z = EndAxis(zeros(T, n))
      @test PB._all_zero(z) && PB._all_zero_early(z)
      # one value at the block boundaries 512 | 513 and 1024 | 1025 of the stored positions, and at the ends
      for pos in unique(filter(<=(n), (1, 512, 513, 1024, 1025, n))), v in (nextfloat(zero(T)), -zero(T))
        d = zeros(T, n); d[pos] = v
        w = EndAxis(d)
        @test PB._all_zero_early(w) == PB._all_zero(w) == (v === -zero(T))
      end
    end
  end
  @testset "exponent of a group from the largest magnitude" begin
    R = Float64
    for n in (1, 2, 5, 17, 64), trial in 1:200
      x = [signif(rng, T, rand(rng, emin(T)-20:emax(T))) for _ in 1:n]
      rand(rng) < 0.2 && (x[rand(rng, 1:n)] = zero(T))
      rand(rng) < 0.1 && (x .= zero(T))
      rand(rng) < 0.1 && (x[rand(rng, 1:n)] = T(Inf))
      rand(rng) < 0.1 && (x[rand(rng, 1:n)] = T(NaN))
      G = randperm(rng, n)
      v = PB._IndexedValues(x, G)
      @test PB._square_exponent(v, nothing, R) == PB._square_exponent_scan(v, R)
      @test PB._square_exponent(x, nothing, R) == PB._square_exponent_scan(x, R)
    end
  end
end

@testset "group value of unpenalized groups" begin
  # zero weights: the term of an unpenalized group is +0, and a sum below K floatmin (K groups) is recomputed
  for T in (Float32, Float64)
    K = 600
    groups = [(2j - 1):(2j) for j in 1:K]
    x = ones(T, 2K)
    f0 = GroupNormL2(T(0.6), groups, zeros(T, K))
    @test f0(x) === zero(T)
    @test value_and_prox!(similar(x), f0, x, T(0.8)) === zero(T)
    # one penalized group of a tiny norm among zero weights, and an unpenalized group with a large entry: the value is
    # within 2 units of the BigFloat reference
    w = zeros(T, K); w[7] = 1
    for e in (emin(T) - 3, emin(T) + 2, emin(T) + 12)
      y = zeros(T, 2K); y[13] = pow2(T, e); y[14] = pow2(T, e)       # group 7
      y[1] = T(3)                                                     # an unpenalized group with a large entry
      f = GroupNormL2(T(0.5), groups, w)
      ref = 0.5 * big(1.0) * sqrt(big(y[13])^2 + big(y[14])^2)
      @test within(f(y), ref, 2)
    end
    # the penalized weights all positive again, with the data below K floatmin: the recomputation still runs
    f = GroupNormL2(T(0.5), groups, ones(T, K))
    y = zeros(T, 2K); y[1] = pow2(T, emin(T) - 3)
    @test within(f(y), 0.5 * big(y[1]), 2)
  end
end

@testset "NormL2 value at NaN entries keeps the propagated NaN: $T" for T in (Float32, Float64)
  # at the positive finite levels tested below the value is λ √s from the ordinary sum of squares s, rounded as the
  # ordinary path rounds it: a NaN keeps its sign and payload, from the smallest of these levels to the largest; an
  # infinite level gives T(NaN), and a zero level gives +0 before the sum is formed
  qnan(b) = reinterpret(T, T === Float64 ? UInt64(b) : UInt32(b))
  nans = (-T(NaN), T === Float64 ? qnan(0x7ff8000000000123) : qnan(0x7fc00123), T === Float64 ? qnan(0xfff8000000000123) : qnan(0xffc00123), T(NaN))
  for v in nans, λ in (T(0.5), one(T), floatmax(T), nextfloat(zero(T)), 1e300, 2.0^-1074), rest in (T[1, 2], T[floatmax(T), 0], T[nextfloat(zero(T)), -0.0])
    for pos in 1:3
      x = copy(rest); insert!(x, pos, v)
      R = promote_type(Float64, T, typeof(λ))
      expected = convert(T, convert(R, λ) * sqrt(convert(R, sum(abs2, x))))
      @test NormL2(λ)(x) === expected
      @test isnan(expected) && signbit(expected) == signbit(v)
    end
    @test NormL2(T(Inf))(T[v, 1]) === T(NaN)
    @test NormL2(zero(T))(T[v, 1]) === zero(T)
  end
  @test NormL2(one(T))(T[1, -Inf]) === T(Inf) && NormL2(T(Inf))(T[1, -Inf]) === T(Inf)
end

@testset "group value below K floatmin with zero-weight groups: $T" for T in (Float32, Float64)
  # K counts every group, penalized or not. A value from npen floatmin (npen penalized groups) up to just below
  # K floatmin is recomputed with the terms' exponents. For positive finite λ and weights stored in T and finite
  # entries in the penalized groups, the recomputed value has about five roundings of u = eps(Float64)/2 against the
  # BigFloat value: the squares and their compensated sum, the sqrt, the product with the weight, the rounding of the
  # accumulated sum, and the product with λ. For Float64 data that is a relative error of at most 5u + O(K u²), up to
  # 5 units in the last place near the top of a binade, and the test allows 3 eps(T) relative. For Float32 data the
  # recovery runs in Float64 and the one rounding to Float32 dominates, so the test allows one unit in the last
  # place. Two edges lie outside this bound. (a) For Float64 data a recomputed value just below floatmin is rounded
  # into the subnormal range by the final scaling, by up to one subnormal spacing, which the Float64 bound adds.
  # (b) A sum within a few eps of K floatmin can be kept by the ordinary path, whose error this bound does not
  # cover, so the window stops at K floatmin (1 - 16 eps(T)). The unpenalized groups hold tiny, large and
  # non-finite entries, which add nothing.
  rng = MersenneTwister(373)
  K = 8
  groups = [1:1, 2:3, 4:6, 7:7, 8:9, 10:12, 13:13, 14:15]
  in_window(ref, npen) = npen * big(floatmin(T)) <= ref < K * big(floatmin(T)) * (1 - 16 * big(eps(T)))
  recomputed_ok(v, ref) =
    T === Float64 ? abs(big(v) - ref) <= 3 * big(eps(T)) * ref + big(nextfloat(zero(T))) : within(v, ref, 1)
  inwindow = 0
  for trial in 1:3000
    w = T[rand(rng, Bool) ? 0 : T(0.5) + rand(rng, T) * 2 for _ in 1:K]
    npen = count(!iszero, w)
    npen == 0 && continue
    x = [ldexp(T(1) + rand(rng, T), emin(T) - rand(rng, -2:precision(T) + 2)) * rand(rng, (-1, 1)) for _ in 1:15]
    for k in 1:K
      iszero(w[k]) || continue
      for i in groups[k]
        x[i] = rand(rng, (zero(T), T(3), floatmax(T), T(NaN), -T(Inf), nextfloat(zero(T))))
      end
    end
    λ = T(rand(rng, (0.9, 0.5, 2.0)))
    f = GroupNormL2(λ, groups, w)
    ref = setprecision(BigFloat, 256) do
      big(λ) * sum(iszero(w[k]) ? big(0) : big(w[k]) * sqrt(sum(big(x[i])^2 for i in groups[k])) for k in 1:K)
    end
    in_window(ref, npen) || continue
    inwindow += 1
    @test recomputed_ok(f(x), ref)
    @test recomputed_ok(value_and_prox!(similar(x), f, x, zero(T)), ref)      # the identity, then the fused value
  end
  @test inwindow >= 100
  if T === Float64
    let
      # an input whose recomputed value is more than 2 ulps from the reference, within the derived bound
      λ = reinterpret(Float64, 0x3feccccccccccccd)
      w = reinterpret.(Float64, UInt64[0x0000000000000000, 0x0000000000000000, 0x3ffe2c98581cf65e, 0x4001df64c872c753,
        0x0000000000000000, 0x3ffeff0492236dfa, 0x0000000000000000, 0x4000fea62bb673e9])
      x = reinterpret.(Float64, UInt64[0x7ff8000000000000, 0x0000000000000000, 0x4008000000000000, 0x803157df512b25c7,
        0x800000000067d9ce, 0x0000afb23a3d71ce, 0x0001e2242014c2b7, 0x0000000000000000, 0x4008000000000000,
        0x8000021d366386ce, 0x00000000000009dc, 0x0000000000001fa9, 0xfff0000000000000, 0x00000000000090f6,
        0x8000000000000efe])
      f = GroupNormL2(λ, groups, w)
      ref = big(λ) * sum(iszero(w[k]) ? big(0) : big(w[k]) * sqrt(sum(big(x[i])^2 for i in groups[k])) for k in 1:K)
      @test in_window(ref, count(!iszero, w))
      @test recomputed_ok(f(x), ref)
      y = similar(x)
      @test recomputed_ok(value_and_prox!(y, f, x, 0.0), ref)
      @test isequal(y, x)
    end
  end
  # two cases where the sum that a threshold of npen floatmin would keep (the code keeps the ordinary sum at K floatmin
  # or above) and the recomputed value differ in the last place: the bits of the recomputed value
  g4 = [1:1, 2:2, 3:3, 4:4]
  if T === Float64
    x = [-2.906061108421734e-308, -9.19875727328223e-308, -0.0, 0.0]
    @test GroupNormL2(0.9, g4, [3.0, 0, 0, 0])(x) === 7.846364992738683e-308
  else
    x = Float32[2.009546f-39, -2.59f-43, 5.488473f-39, 3.1271334f-38]
    @test GroupNormL2(0.9f0, g4, Float32[0, 0, 1, 0.75])(x) === 2.6047776f-38
  end
end

@testset "ℓ2 shrinkage of a group with a zero sum of squares: $T" for T in (Float32, Float64)
  # A group whose sum of squares in T is exactly zero: a zero group (entries +0 and -0), or a nonzero group every square
  # of which rounds to zero. At a threshold c = γλw with c² ≥ |G| floatmin/eps the exact norm is far below c and the
  # proximal point is the zero group; the shortcut takes it where c ≥ |G| 2^-485 (Float64) or |G| 2^-51 (Float32),
  # above |G| √(floatmin/eps); at a smaller threshold the scan for the zero group and the cold path decide. In
  # every case _l2_shrink! must write the group and return the code of the cold route (_l2_shrink_values!: the
  # parameters below are positive and finite), bit for bit, called or inlined (prox_block!), in place or not, through a
  # range or a permuted index vector of a longer array whose other entries are large. Where the exact norm is above a
  # tiny threshold the result is the nonzero shrinkage, also against a BigFloat reference.
  rng = Xoshiro(921)
  hexp(S) = exponent(nextfloat(zero(S))) ÷ 2 - 2      # entries in [2^(h-1), 2^h): squares below nextfloat(0)/2 in S
  bound(S, n) = n * (floatmin(S) / eps(S))              # |G| floatmin/eps, the bound of the exact sum of squares
  # the threshold of the shortcut: c at or above it is the zero group without a scan
  tbound(S, n) = S === Float64 ? n * 0x1p-485 : n * Float32(0x1p-51)
  function cold(x, G, λ, w, γ, ::Type{S}) where {S}
    y = fill(S(7), length(x))
    code = PB._l2_shrink_values!(y, x, G, λ, w, γ)
    y, S(code)
  end
  # _l2_shrink! against the cold route, in all its call forms; returns the code
  function check_cold(x, G, λ, w, γ, ::Type{S}) where {S}
    yc, code = cold(x, G, λ, w, γ, S)
    y1, y2 = fill(S(7), length(x)), fill(S(7), length(x))
    @test PB._l2_shrink!(y1, x, G, λ, w, γ) === code
    @test PB._l2_shrink!(y2, x, G, λ, w, γ, PB._l2_shrink_fixed!) === code
    @test same(y1, yc) && same(y2, yc)
    if eltype(x) === S
      z = copy(x)
      zc = copy(x)
      @test PB._l2_shrink!(z, z, G, λ, w, γ) === S(PB._l2_shrink_values!(zc, zc, G, λ, w, γ)) === code
      @test same(z, zc)
    end
    code, y1
  end
  # a longer array with large entries outside the group: a range G, or a permuted index vector
  function embed(v, layout)
    n = length(v)
    layout === :range && return v, 1:n
    L = 2n + 3
    x = fill(eltype(v)(3), L)
    G = randperm(rng, L)[1:n]
    x[G] = v
    x, G
  end
  sizes = (1, 2, 5, 32, 1024)
  h = hexp(T)
  for n in sizes, layout in (:range, :index)
    zero_group = [rand(rng, (zero(T), -zero(T))) for _ in 1:n]
    tiny = [signif(rng, T, h) for _ in 1:n]
    @test all(iszero, abs2.(tiny)) && iszero(sum(abs2, tiny)) && !any(iszero, tiny)
    nrm = setprecision(() -> sqrt(sum(big.(tiny) .^ 2)), BigFloat, 256)
    # ordinary thresholds (c² at or above the bound, also where it overflows): the zero group, with the zero code
    for v in (zero_group, tiny), (γ, λ, w) in ((T(1), T(0.5), T(1)), (T(0.5), T(3), sqrt(T(n))), (T(1), floatmax(T) / 2, T(1)),
                                                (T(1), floatmax(T), T(1)), (T(1), T(0.5), true))
      x, G = embed(v, layout)
      c = γ * λ * w
      @test floatmin(T) <= c <= floatmax(T) && PB._l2_threshold_decides(c, n)
      code, y = check_cold(x, G, λ, w, γ, T)
      @test code === T(PB._L2_ZERO) && all(i -> y[i] === zero(T), G)
    end
    # the threshold at the bound of the shortcut: the two sides of c ≥ tbound(T, |G|), a few floats each way, and the
    # smallest ordinary threshold floatmin; the zero group, from the cold route below the bound, except for the tiny
    # group at floatmin, which is below its norm
    c0 = tbound(T, n)
    sides = Set{Bool}()
    for c in (floatmin(T), prevfloat(c0, 2), prevfloat(c0), c0, nextfloat(c0), nextfloat(c0, 2)), v in (zero_group, tiny)
      x, G = embed(v, layout)
      push!(sides, PB._l2_threshold_decides(c, n))
      @test PB._l2_threshold_decides(c, n) == (c >= c0)
      code, y = check_cold(x, G, c, T(1), T(1), T)
      if v === tiny && big(c) < nrm
        @test c == floatmin(T) && code === T(PB._L2_RECOMPUTE) && all(i -> !iszero(y[i]), G)
      else
        @test code === T(PB._L2_ZERO) && all(i -> y[i] === zero(T), G)
      end
    end
    @test sides == Set((false, true))
    # a tiny threshold below the norm of the tiny group (c² far below the bound): the scan and the cold path, which give
    # the nonzero shrinkage (1 - c/‖x‖) x and the recompute code, within a few eps of the BigFloat reference
    for r in (0.5, 0.875)
      c = T(r * nrm)
      @test floatmin(T) <= c && !PB._l2_threshold_decides(c, n)
      x, G = embed(tiny, layout)
      code, y = check_cold(x, G, c, T(1), T(1), T)
      @test code === T(PB._L2_RECOMPUTE)
      ref = setprecision(() -> (1 - big(c) / nrm) .* big.(tiny), BigFloat, 256)
      m = maximum(abs, tiny)
      @test all(j -> abs(big(y[G[j]]) - ref[j]) <= 4 * big(eps(T)) * big(m), 1:n)
      @test all(j -> !iszero(y[G[j]]), 1:n)
      # the zero group at the same threshold: the scan finds it
      xz, Gz = embed(zero_group, layout)
      code, y = check_cold(xz, Gz, c, T(1), T(1), T)
      @test code === T(PB._L2_ZERO) && all(i -> y[i] === zero(T), Gz)
    end
  end

  # Mixed precision: the sum of squares is formed from the entries converted to the output type S, and the cold route
  # reads the entries in their own type (at least Float64). A wider x whose entries convert to S with squares that
  # round to zero, entries that convert to subnormals or to zero in S, -0, and for BigFloat entries beyond the range
  # of S: at ordinary thresholds the zero group, at a tiny one below the norm of the entries as stored the cold
  # shrinkage, both bit for bit with the cold route; a narrower x has no zero sum except at the zero group.
  for (S, X) in ((Float32, Float64), (Float32, BigFloat), (Float64, BigFloat))
    T === S || continue
    hS = hexp(S)
    setprecision(BigFloat, 256) do
      for n in sizes, layout in (:range, :index)
        v = [X(signif(rng, S, hS)) * (1 + X(2)^-(precision(S) + 3)) for _ in 1:n]
        n >= 2 && (v[2] = ldexp(X(1), exponent(nextfloat(zero(S))) + 3))      # a subnormal in S
        n >= 5 && (v[3] = ldexp(X(1), exponent(nextfloat(zero(S))) - 3))      # rounds to zero in S
        n >= 5 && (v[4] = -zero(X))
        X === BigFloat && n >= 5 && (v[5] = -ldexp(X(1), 4 * exponent(nextfloat(zero(S)))))
        @test all(a -> iszero(abs2(S(a))), v) && !all(iszero, v)
        nrm = sqrt(sum(big.(v) .^ 2))
        x, G = embed(v, layout)
        for (γ, λ, w) in ((S(1), S(0.5), S(1)), (1.0, 0.5, 1.0), (S(1), S(0.5), true))
          code, y = check_cold(x, G, λ, w, γ, S)
          @test code === S(PB._L2_ZERO) && all(i -> y[i] === zero(S), G)
        end
        c0 = tbound(S, n)
        sides = Set{Bool}()
        for c in (prevfloat(c0, 2), prevfloat(c0), c0, nextfloat(c0), nextfloat(c0, 2))
          push!(sides, PB._l2_threshold_decides(c, n))
          code, y = check_cold(x, G, c, S(1), S(1), S)
          @test code === S(PB._L2_ZERO) && all(i -> y[i] === zero(S), G)
        end
        @test sides == Set((false, true))
        c = S(nrm / 2)
        @test floatmin(S) <= c && !PB._l2_threshold_decides(c, n)
        code, y = check_cold(x, G, c, S(1), S(1), S)
        @test code === S(PB._L2_RECOMPUTE)
        ref = (1 - big(c) / nrm) .* big.(v)
        @test all(j -> abs(big(y[G[j]]) - ref[j]) <= 4 * big(eps(S)) * maximum(abs, big.(v)) + big(nextfloat(zero(S))), 1:n)
      end
      # a narrower x: the zero group only (a nonzero Float32 or Float64 entry has a nonzero square in the wider type)
      if S === Float64
        x = Float32[0, -0.0, 0]
        code, y = check_cold(x, 1:3, 0.5, 1.0, 1.0, Float64)
        @test code === Float64(PB._L2_ZERO) && all(a -> a === 0.0, y)
      end
    end
  end

  # A Float16 output, where the bound overflows for long groups and the zero group is found by the scan: the cold route,
  # bit for bit, at ordinary thresholds (c² = Inf at c = 300), at floatmin and at the bound
  if T === Float32
    for n in sizes, X in (Float16, Float64), layout in (:range, :index)
      tiny = X.([signif(rng, Float16, hexp(Float16)) for _ in 1:n])
      zero_group = [rand(rng, (zero(X), -zero(X))) for _ in 1:n]
      @test all(a -> iszero(abs2(Float16(a))), tiny)
      for v in (tiny, zero_group), c in (Float16(0.5), Float16(300), floatmin(Float16), sqrt(bound(Float16, n)))
        x, G = embed(v, layout)
        code, y = check_cold(x, G, c, Float16(1), Float16(1), Float16)
        v === zero_group && @test code === Float16(PB._L2_ZERO) && all(i -> y[i] === zero(Float16), G)
      end
    end
  end
end

# The test of the zero-group shortcut: c ≥ n 2^-485 for Float64 and c ≥ n 2^-51 for Float32, one product and one
# comparison (2^-485 = √(floatmin/eps) of Float64, and 2^-51 ≥ √(floatmin/eps) = 2^-51.5 of Float32); false for
# other types, which scan the group
@testset "threshold of the zero-group shortcut" begin
  for (T, e) in ((Float64, -485), (Float32, -51))
    @test ldexp(one(T), e) >= sqrt(floatmin(T) / eps(T)) && ldexp(one(T), e - 1) < sqrt(floatmin(T) / eps(T))
    for n in (1, 5, 1024, 2^30)
      b = n * ldexp(one(T), e)
      @test b isa T && isfinite(b) && b >= floatmin(T)
      @test PB._l2_threshold_decides(b, n) && PB._l2_threshold_decides(nextfloat(b), n)
      @test !PB._l2_threshold_decides(prevfloat(b), n) && !PB._l2_threshold_decides(prevfloat(b, 3), n)
      @test PB._l2_threshold_decides(floatmax(T), n) && PB._l2_threshold_decides(T(Inf), n)
      @test !PB._l2_threshold_decides(floatmin(T), n) && !PB._l2_threshold_decides(zero(T), n)
      @test !PB._l2_threshold_decides(T(NaN), n)
      # c² ≥ n floatmin/eps wherever the shortcut is taken
      @test big(b)^2 >= n * big(floatmin(T)) / big(eps(T))
    end
  end
  for n in (1, 5, 1024, 2^30)
    @test !PB._l2_threshold_decides(Float16(1), n) && !PB._l2_threshold_decides(floatmax(Float16), n)
    @test !PB._l2_threshold_decides(big(1.0), n) && !PB._l2_threshold_decides(big(2.0)^100, n)
  end
end

end

# The unweighted NormL1 value of an Array of Float32 or Float64 with 16 to 1024 entries sums |xᵢ| in one vectorizing
# loop (_value_loop); shorter and longer arrays keep Base's sum, as on master. Views take the generic route of the
# other arrays and never reach the loop. The value must be within 4n eps(T) λ0 Σ|xᵢ| of λ0 Σ|xᵢ| summed in BigFloat
# (λ0 rounded to T, as the value uses it), and outside the window it must be master's convert(T, λ0) * sum(abs, x).
@testset "unweighted NormL1 value: one loop for 16 to 1024 entries ($T)" for T in (Float32, Float64)
  rng = Xoshiro(1024)
  lo, hi = PB._VALUE_LOOP_MIN, PB._VALUE_LOOP_MAX
  @test (lo, hi) == (16, 1024)
  l1_ref(λ, x) = setprecision(BigFloat, 512) do
    big(convert(T, λ)) * sum(a -> abs(big(a)), x; init = big(0))
  end
  l1_within(v, λ, x) = (r = l1_ref(λ, x); abs(big(v) - r) <= 4 * length(x) * eps(T) * r)
  levels = T === Float32 ? (one(T), T(0.3), 0.3) : (one(T), T(0.3))
  for n in (0, 1, 15, 16, 17, 100, 1023, 1024, 1025, 2000), λ in levels
    f = NormL1(λ)
    for x in (randn(rng, T, n), fill(T(0.1), n), fill(-T(1.1), n), ldexp.(randn(rng, T, n), -40))
      v = f(x)
      @test v isa T
      @test l1_within(v, λ, x)
      if lo <= n <= hi
        @test v === convert(T, λ) * PB._value_loop(abs, x)
      else
        @test v === convert(T, λ) * sum(abs, x)                 # master's value, bit for bit
      end
      n == 0 && @test v === zero(T)
    end
  end
  # a sum of small integers is exact in any order: every entry counts once, at every length of the remainder loop
  for n in vcat(0:70, [1023, 1024, 1025])
    x = T.(rand(rng, -8:8, n))
    @test NormL1(one(T))(x) === T(sum(abs, Int.(x); init = 0))
    @test NormL1(T(2))(x) === T(2 * sum(abs, Int.(x); init = 0))
  end
  # matrices in and out of the window; views of a length in the window take the generic route (not the loop)
  for dims in ((4, 4), (16, 16), (32, 32), (33, 32), (3, 5)), λ in levels
    M = randn(rng, T, dims...)
    f = NormL1(λ)
    @test l1_within(f(M), λ, M)
    @test f(M) === convert(T, λ) * (lo <= length(M) <= hi ? PB._value_loop(abs, M) : sum(abs, M))
    @test l1_within(f(view(M, :, 2)), λ, M[:, 2])
    @test l1_within(f(view(vec(M), 1:2:length(M))), λ, vec(M)[1:2:end])
  end
  @test which(NormL1(T(1)), Tuple{typeof(view(zeros(T, 100), 1:2:100))}) !== which(NormL1(T(1)), Tuple{Vector{T}})
  # NaN and ±Inf entries propagate, at both ends, in and out of the window
  for n in (1, 15, 16, 100, 1024, 1025), at in unique((1, n)), λ in (T(0.3), T(Inf))
    f = NormL1(λ)
    x = randn(rng, T, n)
    x[at] = T(NaN); @test isnan(f(x))
    x[at] = T(Inf); @test f(x) === T(Inf)
    x[at] = -T(Inf); @test f(x) === T(Inf)
    if n > 1
      x[n + 1 - at] = T(Inf); x[at] = -T(Inf); @test f(x) === T(Inf)
      x[n + 1 - at] = T(NaN); @test isnan(f(x))
    end
  end
  # a sum that overflows with finite entries is recovered from the entries (Inf where the value overflows, the
  # value where it is finite), at both ends of the window
  for n in (15, 16, 100, 1024, 1025)
    x = fill(floatmax(T) / 4, n)
    @test NormL1(one(T))(x) === T(Inf)
    for λ in (T(2) / n, T(1) / (8n))
      v = NormL1(λ)(x)
      r = l1_ref(λ, x)
      @test isfinite(v) == (r <= floatmax(T)) && (!isfinite(v) || abs(big(v) - r) <= 4 * n * eps(T) * r)
    end
  end
  # an infinite level at the zero vector is 0, elsewhere Inf; a zero level is 0, NaN entries included
  for n in (0, 1, 15, 16, 100, 1024, 1025)
    @test NormL1(T(Inf))(zeros(T, n)) === zero(T)
    @test NormL1(T(Inf))(fill(-zero(T), n)) === zero(T)
    @test NormL1(0.0)(zeros(T, n)) === zero(T)
    @test NormL1(zero(T))(fill(T(NaN), n)) === zero(T)
    n > 0 && @test NormL1(T(Inf))(vcat(zeros(T, n - 1), nextfloat(zero(T)))) === T(Inf)
  end
  # no allocation, in and out of the window
  for n in (7, 100, 1024, 2000)
    x = randn(rng, T, n)
    @test alloc(NormL1(T(0.3)), x) == 0
    @test alloc(NormL1(T(0.3)), reshape(x, 1, n)) == 0
  end
end

# The reference of the soft-thresholding tests below: the formula of shrink, written out here. A result must have its
# bits, NaN in either operand, signed zeros, infinities and subnormals included (any NaN for a NaN).
shrink_formula(v::T, c::T) where {T<:AbstractFloat} = max(v - c, zero(T)) + min(v + c, zero(T))
shrink_same(a, b) = isnan(b) ? isnan(a) : a === b
# the number of pairs of vs × cs where shrink and the formula differ, and the first such pair
function shrink_mismatches(vs::AbstractVector{T}, cs::AbstractVector{T}) where {T}
  bad, first_bad = 0, nothing
  for c in cs, v in vs
    if !shrink_same(shrink(v, c), shrink_formula(v, c))
      bad += 1
      first_bad === nothing && (first_bad = (v, c))
    end
  end
  bad, first_bad
end
function shrink_pair_mismatches(vs::AbstractVector{T}, cs::AbstractVector{T}) where {T}
  bad, first_bad = 0, nothing
  for i in eachindex(vs, cs)
    v, c = vs[i], cs[i]
    if !shrink_same(shrink(v, c), shrink_formula(v, c))
      bad += 1
      first_bad === nothing && (first_bad = (v, c))
    end
  end
  bad, first_bad
end
# zeros, the ends of the subnormal and normal ranges, values near one, infinities and NaNs of either sign (with a payload)
function shrink_edges(::Type{T}) where {T<:AbstractFloat}
  U = Base.uinttype(T)
  pos = T[0, nextfloat(zero(T)), 3 * nextfloat(zero(T)), prevfloat(floatmin(T)), floatmin(T), nextfloat(floatmin(T)),
          eps(T), prevfloat(one(T)), 1, nextfloat(one(T)), 2, 3, prevfloat(floatmax(T)), floatmax(T), Inf, NaN,
          reinterpret(T, reinterpret(U, T(NaN)) | one(U))]
  vcat(pos, -pos)
end

@testset "soft thresholding of other types" begin
  # under a directed rounding mode v - c is -0 at v = c, and max orders the zeros
  setprecision(BigFloat, 256) do
    for mode in (RoundDown, RoundNearest, RoundUp, RoundToZero)
      r = setrounding(() -> shrink(big(1.0), big(1.0)), BigFloat, mode)
      @test iszero(r) && !signbit(r)
      # the generic formula in every mode (at v = -c it is -0 under RoundDown, from min(-0, +0))
      for (v, c) in ((1.0, 1.0), (-1.0, 1.0), (3.0, 1.0), (-3.0, 1.0), (0.5, 1.0), (-0.0, 0.0), (0.0, -0.0))
        @test bigsame(setrounding(() -> shrink(big(v), big(c)), BigFloat, mode),
                      setrounding(() -> shrink_formula(big(v), big(c)), BigFloat, mode))
      end
    end
    @test shrink(big(3.0), big(1.0)) == 2 && shrink(big(-3.0), big(1.0)) == -2
  end
  # mixed and integer arguments promote to a common floating-point type, as before
  @test shrink(1, 0.5) === 0.5
  @test shrink(1f0, 0.25) === 0.75
  @test shrink(-3, 1) === -2.0
  @test shrink(1, 2) === 0.0 && shrink(-1, 2) === 0.0
  @test shrink(Float16(1), 0.5f0) === 0.5f0
  @test shrink(Float16(-3), Float16(1)) === Float16(-2)
  @test shrink(2, 1//2) === 1.5
end

# The ordinary branch of the NormL1 proximal operator soft-thresholds with _shrink_nonneg: v minus its projection onto
# [-c, c], plus +0. Its threshold γλ0w is a product of nonnegative factors, and for every v and every c that is not
# negative (+0, -0, positive, +Inf and every NaN) the result must have the bits of the formula of shrink (any NaN for
# a NaN). At a negative c the two differ.
nonneg(c) = !(c < 0)
function nonneg_mismatches(vs::AbstractVector{T}, cs::AbstractVector{T}) where {T}
  bad, first_bad = 0, nothing
  for c in cs, v in vs
    if !shrink_same(PB._shrink_nonneg(v, c), shrink_formula(v, c))
      bad += 1
      first_bad === nothing && (first_bad = (v, c))
    end
  end
  bad, first_bad
end
function nonneg_pair_mismatches(vs::AbstractVector{T}, cs::AbstractVector{T}) where {T}
  bad, first_bad = 0, nothing
  for i in eachindex(vs, cs)
    v, c = vs[i], cs[i]
    if !shrink_same(PB._shrink_nonneg(v, c), shrink_formula(v, c))
      bad += 1
      first_bad === nothing && (first_bad = (v, c))
    end
  end
  bad, first_bad
end
# pairs that the tests name explicitly: signed zeros, infinities at an infinite threshold, NaN in either operand
function nonneg_named(::Type{T}) where {T}
  U = Base.uinttype(T)
  nan2 = reinterpret(T, reinterpret(U, T(NaN)) | one(U))
  T[-0.0, -0.0, 0.0, Inf, -Inf, NaN, -T(NaN), 1, -1, 0.0, nan2, -nan2],
  T[0.0, -0.0, -0.0, Inf, Inf, 1, 1, NaN, -T(NaN), NaN, 0.0, Inf]
end

@testset "soft thresholding at a nonnegative threshold: Float16, every v" begin
  U = UInt16
  vs = [reinterpret(Float16, b) for b in typemin(U):typemax(U)]
  cs = filter(nonneg, unique(b -> reinterpret(U, b),
                             vcat([reinterpret(Float16, b) for b in typemin(U):typemax(U) if b % 97 == 0], shrink_edges(Float16))))
  @test length(vs) == 65536 && length(cs) > 350 && all(nonneg, cs)
  @test any(c -> c === -zero(Float16), cs) && any(c -> isnan(c) && signbit(c), cs) && any(c -> isnan(c) && !signbit(c), cs)
  bad, first_bad = nonneg_mismatches(vs, cs)
  @test bad == 0
  bad == 0 || @info "first Float16 pair that differs" first_bad
  @test nonneg_pair_mismatches(nonneg_named(Float16)...)[1] == 0
end

@testset "soft thresholding at a nonnegative threshold: $T" for T in (Float32, Float64)
  rng = Xoshiro(197)
  U = Base.uinttype(T)
  E = shrink_edges(T)
  C = filter(nonneg, E)
  bad, first_bad = nonneg_mismatches(E, C)
  @test bad == 0
  bad == 0 || @info "first $T edge pair that differs" first_bad
  # random bit patterns, the threshold as the absolute value of one (NaN thresholds keep their payload)
  vs = [reinterpret(T, rand(rng, U)) for _ in 1:10^6]
  cs = [abs(reinterpret(T, rand(rng, U))) for _ in 1:10^6]
  bad, first_bad = nonneg_pair_mismatches(vs, cs)
  @test bad == 0
  bad == 0 || @info "first random $T pair that differs" first_bad
  # random magnitudes against the edges, and thresholds within a few units in the last place of |v|
  ms = [rand(rng, (-1, 1)) * ldexp(one(T) + rand(rng, T), rand(rng, exponent(floatmin(T)) - precision(T):exponent(floatmax(T))))
        for _ in 1:2000]
  @test nonneg_mismatches(ms, C)[1] == 0 && nonneg_mismatches(E, abs.(ms))[1] == 0
  vn, cn = T[], T[]
  for v in vcat(ms, E), k in -3:3
    a = abs(v)
    c = k < 0 ? prevfloat(a, -k) : nextfloat(a, k)
    nonneg(c) || continue
    push!(vn, v, -v)
    push!(cn, c, c)
  end
  bad, first_bad = nonneg_pair_mismatches(vn, cn)
  @test bad == 0
  bad == 0 || @info "first near-threshold $T pair that differs" first_bad
  # the named pairs, and the values at the signed zeros and infinities
  @test nonneg_pair_mismatches(nonneg_named(T)...)[1] == 0
  @test PB._shrink_nonneg(-zero(T), zero(T)) === zero(T)
  @test PB._shrink_nonneg(-zero(T), -zero(T)) === zero(T)
  @test PB._shrink_nonneg(zero(T), -zero(T)) === zero(T)
  @test isnan(PB._shrink_nonneg(T(Inf), T(Inf))) && isnan(PB._shrink_nonneg(-T(Inf), T(Inf)))
  @test isnan(PB._shrink_nonneg(T(NaN), one(T))) && isnan(PB._shrink_nonneg(one(T), T(NaN)))
  @test PB._shrink_nonneg(T(Inf), one(T)) === T(Inf) && PB._shrink_nonneg(-T(Inf), one(T)) === -T(Inf)
end

@testset "soft thresholding at a nonnegative threshold: other types, and a negative threshold" begin
  # BigFloat forwards to shrink, in every rounding mode
  setprecision(BigFloat, 256) do
    for mode in (RoundDown, RoundNearest, RoundUp, RoundToZero)
      for (v, c) in ((1.0, 1.0), (-1.0, 1.0), (3.0, 1.0), (-3.0, 1.0), (0.5, 1.0), (-0.0, 0.0), (0.0, -0.0), (0.5, -1.0))
        @test bigsame(setrounding(() -> PB._shrink_nonneg(big(v), big(c)), BigFloat, mode),
                      setrounding(() -> shrink(big(v), big(c)), BigFloat, mode))
      end
    end
  end
  # mixed and integer arguments promote as shrink does
  for (v, c) in ((1, 0.5), (1f0, 0.25), (-3, 1), (1, 2), (-1, 2), (Float16(1), 0.5f0), (Float16(-3), Float16(1)), (2, 1//2))
    @test PB._shrink_nonneg(v, c) === shrink(v, c)
  end
  # at a negative threshold the projection onto [-c, c] is not soft thresholding: 0.5 - (-1) = 1.5, where shrink gives
  # (0.5 + 1) + (0.5 - 1) = 1. Neither can replace the other.
  for T in (Float16, Float32, Float64)
    @test PB._shrink_nonneg(T(0.5), T(-1)) === T(1.5)
    @test shrink(T(0.5), T(-1)) === T(1)
  end
end

@testset "NormL1 ordinary branch: a free -0 and a weight changed to NaN ($T)" for T in (Float32, Float64)
  # a zero weight at a -0 entry and a normal γλ0: the ordinary branch at a zero threshold gives +0, as on master,
  # through every dense entry point
  f = NormL1(T(1), T[0])
  @test prox(f, T[-0.0], T(1))[1] === zero(T)
  @test PB.prox_block(f, -zero(T), 1, T(1)) === zero(T)
  @test prox(f, T[-0.0], T[1])[1] === zero(T)
  y = T[7]; value_and_prox!(y, f, T[-0.0], T(1)); @test y[1] === zero(T)
  z = T[-0.0]; prox!(z, f, z, T(1)); @test z[1] === zero(T)
  @test prox(NormL1(1.0, [0.0]), [-0.0], 1.0)[1] === 0.0
  @test prox(NormL1(T(2), T[0, 1, 0]), T[-0.0, -0.0, 0.0], T(0.5)) |> r -> all(a -> a === zero(T), r)
  # weights changed to NaN after construction: the threshold is NaN and the entry NaN (master gives the same); the
  # other entries keep their soft thresholding
  w = T[1, 2, 0.5, 1]
  f = NormL1(T(0.5), w)
  x = T[3, -3, 0.25, -0.0]
  w[2] = T(NaN); w[4] = -T(NaN)
  for γ in (T(1), T[1, 1, 1, 1])
    r = prox(f, x, γ)
    @test isnan(r[2]) && isnan(r[4]) && r[1] === T(2.5) && r[3] === T(0)
    r = prox!(copy(x), f, copy(x), γ)
    @test isnan(r[2]) && isnan(r[4]) && r[1] === T(2.5) && r[3] === T(0)
  end
end

end
