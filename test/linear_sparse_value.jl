module LinearSparseValueTest
using Test, ProximalBase, LinearAlgebra, SparseArrays, Random
const PB = ProximalBase

# The value f(x) of f = LinearFunction(c, d) at a matrix SparseIterate x (#810): the sum of c[i] * x[i] over the
# stored entries of x whose value is nonzero (D1: an entry that is not stored, or is stored as +0 or -0, adds
# nothing, even against a NaN or infinite coefficient; a nonzero entry follows ordinary arithmetic), then d. When
# the ordinary value is not finite and d, the nonzero entries and the coefficients at them are finite, the value
# is recomputed from exponent-separated terms (R1). A Diagonal coefficient array is read at the diagonal of x only.
# The references below are computed from the stored entries in BigFloat, without the package's value.

# Writing a nonzero first keeps the coordinate stored when its value is zero.
function make_iterate(T, p, entries; q = p)
  x = SparseIterate(T, p, q)
  for (r, c, v) in entries
    x[r, c] = one(T)
    x[r, c] = v
  end
  x
end
linear_of(p, r, c) = (c - 1) * p + r
state(x::AbstractArray) = copy(x)
state(x::SparseIterate) = (Array(x), collect(activecoordinates(x)), nactive(x))

# ⟨c, x⟩ + d over the stored nonzero entries (the diagonal ones for a Diagonal c), in BigFloat of enough precision
# for the exponents of Float32 and Float64 products; n = max(the number of those entries, 1), Σ|cᵢxᵢ| + |d|, and the
# R1 bound 4 n eps(T) (Σ|cᵢxᵢ| + |d|) + n floatmin(T) eps(T) of a value of type T
function reference(c, d, x, T; diagonal = false)
  setprecision(BigFloat, 4400) do
    p = size(x, 1)
    terms = BigFloat[]
    for (i, v) in zip(collect(activecoordinates(x)), collect(activevalues(x)))
      iszero(v) && continue
      diagonal && (i - 1) % p != (i - 1) ÷ p && continue
      push!(terms, big(diagonal ? c[(i - 1) % p + 1] : c[i]) * big(v))
    end
    r = sum(terms; init = big(0)) + big(d)
    sabs = sum(abs, terms; init = big(0)) + abs(big(d))
    n = max(length(terms), 1)
    (value = r, n = n, sabs = sabs, bound = 4n * big(eps(T)) * sabs + n * big(floatmin(T)) * big(eps(T)))
  end
end
reference(f::LinearFunction, x, T; kw...) = reference(f.c isa Diagonal ? f.c.diag : f.c, f.d, x, T; diagonal = f.c isa Diagonal, kw...)

# the smallest value that rounds to Inf
overflow_threshold(T) = big(floatmax(T)) + (big(floatmax(T)) - big(prevfloat(floatmax(T)))) / 2
# v is the value of the reference r: finite and within the bound when the bound keeps r in range, ±Inf of the sign of r
# when r is beyond the range by more than the bound, and either between
function agrees(v, ref, T)
  r, bound = ref.value, ref.bound
  if abs(r) + bound <= big(floatmax(T))
    return isfinite(v) && abs(big(v) - r) <= bound
  elseif abs(r) - bound > overflow_threshold(T)
    return v == (r > 0 ? T(Inf) : -T(Inf))
  end
  true
end
# f(x) at a matrix SparseIterate
check(f, x, T) = agrees(f(x), reference(f, x, T), T)

# a random magnitude 2^e (0.5 + u) with a random sign, e in lo:hi
function rand_mag(rng, T, lo, hi)
  s = rand(rng, Bool) ? one(T) : -one(T)
  s * ldexp(T(0.5 + rand(rng)), rand(rng, lo:hi))
end
# a random iterate and coefficient array of size p × p, k entries stored at random positions, a fraction of them stored
# as ±0, with magnitudes 2^e, e in lox:hix for the iterate and in loc:hic for the coefficients
function random_pair(rng, T, p, k, lox, hix, loc, hic; zeros = 0.25)
  pos = randperm(rng, p^2)[1:k]
  x = SparseIterate(T, p, p)
  for i in pos
    x[i] = one(T)
    x[i] = rand(rng) < zeros ? (rand(rng, Bool) ? zero(T) : -zero(T)) : rand_mag(rng, T, lox, hix)
  end
  c = T[rand_mag(rng, T, loc, hic) for _ in 1:p, _ in 1:p]
  x, c
end

# Terms with separated exponents, E = exponent(floatmax(T)): the iterate entry has the exponent ex, uniform in
# exponent(floatmin(T)):E-1, and its coefficient the exponent clamp(ep - ex, exponent(floatmin(T)), E-1) with
# ep = E + (-8:1), so a product has the exponent ep unless the clamp acts (ex < ep - (E - 1) or ep - ex < exponent(floatmin(T))),
# and then a smaller one. With `cancel`, a second entry (another coefficient and an iterate entry, whose product is
# the negative of the first to rounding) is added. Positions on the diagonal when `diagonal`.
function separated_pair(rng, T, p; diagonal, cancel)
  E, Emin = exponent(floatmax(T)), exponent(floatmin(T))
  sgn() = rand(rng, Bool) ? one(T) : -one(T)
  x = SparseIterate(T, p, p)
  c = ones(T, p, p)
  slots = diagonal ? [(j, j) for j in randperm(rng, p)] : [(r, k) for (r, k) in Iterators.product(1:p, 1:p)][randperm(rng, p^2)]
  next = 1
  for _ in 1:rand(rng, 1:(cancel ? 3 : 5))
    ex = rand(rng, Emin:E-1)
    ep = E + rand(rng, -8:1)
    ec = clamp(ep - ex, Emin, E - 1)
    xv, cv = sgn() * ldexp(T(0.5 + rand(rng)), ex), sgn() * ldexp(T(0.5 + rand(rng)), ec)
    r, k = slots[next]; next += 1
    x[r, k] = one(T); x[r, k] = xv; c[r, k] = cv
    cancel || continue
    ex2 = rand(rng, max(Emin, ep - E + 1):E-1)
    c2 = sgn() * ldexp(T(0.5 + rand(rng)), clamp(ep - ex2, Emin, E - 1))
    x2 = T(-(big(cv) * big(xv) / big(c2)))
    isfinite(x2) || continue
    r, k = slots[next]; next += 1
    x[r, k] = one(T); x[r, k] = x2; c[r, k] = c2
  end
  x, diagonal ? Diagonal(T[c[j, j] for j in 1:p]) : c
end


# Terms x c that cancel exactly in pairs, for the diagonal of a p × p iterate, x of type Tx and c of type Tc: a class has
# the product exponent pe = E + 1 - rand(0:spread), E = exponent(floatmax(promote_type(Tx, Tc))), the product of the
# significands being small integers over 4 (so every product is exact in Float64), and each pair is (x, c), (x, -c).
# Returns the entries (x, c) in a random order, or nothing if a factor is out of the range of its type.
function cancelling_terms(rng, Tx, Tc, spread, nclass, maxpairs)
  E = exponent(floatmax(promote_type(Tx, Tc)))
  sig() = rand(rng, (4, 5, 6, 7)) / 4
  terms = Tuple{Tx, Tc}[]
  for _ in 1:nclass
    pe = E + 1 - rand(rng, 0:spread)
    for _ in 1:rand(rng, 1:maxpairs)
      xe = rand(rng, 1:20)
      x, c = ldexp(Tx(sig()), xe), ldexp(Tc(sig()), pe - xe)
      (isfinite(x) && isfinite(c)) || return nothing
      push!(terms, (x, c), (x, -c))
    end
  end
  terms[randperm(rng, length(terms))]
end
# an iterate with the entries xs on the diagonal, activated in the order of the vector, and the coefficients cs
function diagonal_iterate(Tx, p, xs)
  x = SparseIterate(Tx, p, p)
  for (j, v) in enumerate(xs)
    x[j, j] = one(Tx); x[j, j] = v
  end
  x
end
# all 24 orders of 1:4
const ORDERS4 = [[a, b, c, d] for a in 1:4 for b in 1:4 for c in 1:4 for d in 1:4 if allunique((a, b, c, d))]

# allocations of a warmed call
allocated(f, x) = (f(x); f(x); @allocated f(x))

@testset "LinearFunction value at a matrix SparseIterate" begin
  @testset "Strong zeros (D1) ($T)" for T in (Float32, Float64)
    p = 100
    mk() = make_iterate(T, p, [(1, 1, T(2)), (2, 2, zero(T)), (3, 3, -zero(T))])
    x = mk()
    @test nactive(x) == 3 && isactive(x, 1) && isactive(x, 102) && isactive(x, 203)   # the two zeros are stored
    @test signbit(x[3, 3]) && !signbit(x[2, 2]) && iszero(x[2, 2]) && iszero(x[3, 3])
    @test !isactive(x, linear_of(p, 4, 4))
    for bad in (T(NaN), T(Inf), -T(Inf)), (r, c) in ((4, 4), (2, 2), (3, 3))
      a = ones(T, p, p); a[r, c] = bad
      f = LinearFunction(a, T(0.25))
      @test f(x) === T(2.25)
      @test isnan(f(Matrix(x)))                 # the dense value diverges here: the generic dot of a dense c visits all positions
      @test check(f, x, T)
      @test LinearFunction(a)(x) === T(2)       # d omitted
    end
    # all the coefficients at zeros nonfinite
    a = fill(T(NaN), p, p); a[1, 1] = T(3)
    @test LinearFunction(a, T(-0.5))(x) === T(5.5)
    # a participating entry keeps ordinary arithmetic
    for (cv, expected) in ((T(NaN), T(NaN)), (T(Inf), T(Inf)), (-T(Inf), -T(Inf)), (zero(T), T(0.25)), (T(1.5), T(3.25)))
      a = ones(T, p, p); a[1, 1] = cv
      @test isequal(LinearFunction(a, T(0.25))(x), expected)
    end
    a = ones(T, p, p); a[1, 1] = T(Inf)
    y = make_iterate(T, p, [(1, 1, -T(2))])
    @test LinearFunction(a, T(0.25))(y) === -T(Inf)
    # x holds Inf or NaN
    for (entries, expected) in (([(1, 1, T(Inf))], T(Inf)), ([(1, 1, -T(Inf))], -T(Inf)),
                                ([(1, 1, T(Inf)), (2, 1, -T(Inf))], T(NaN)), ([(1, 1, T(Inf)), (2, 1, T(Inf))], T(Inf)),
                                ([(1, 1, T(NaN))], T(NaN)), ([(1, 1, T(1)), (5, 5, T(NaN)), (2, 2, zero(T))], T(NaN)))
      f = LinearFunction(ones(T, p, p), T(0.25))
      @test isequal(f(make_iterate(T, p, entries)), expected)
    end
    # Inf against a zero coefficient is NaN
    a = ones(T, p, p); a[1, 1] = zero(T)
    @test isnan(LinearFunction(a)(make_iterate(T, p, [(1, 1, T(Inf))])))
    # an empty iterate: the constant, also when the coefficients are NaN or infinite
    for (cv, dv) in ((T(NaN), T(0.5)), (T(Inf), zero(T)), (-T(Inf), T(-3)))
      v = LinearFunction(fill(cv, p, p), dv)(SparseIterate(T, p, p))
      @test v === dv && typeof(v) === T
    end
    # an iterate with only stored zeros
    only_zeros = make_iterate(T, p, [(1, 1, zero(T)), (7, 8, -zero(T))])
    @test nactive(only_zeros) == 2
    @test LinearFunction(fill(T(NaN), p, p), T(1.5))(only_zeros) === T(1.5)
    # a rectangular iterate
    xr = make_iterate(T, 3, [(1, 1, T(2)), (2, 4, T(-1)), (3, 2, zero(T))]; q = 5)
    ar = ones(T, 3, 5); ar[3, 2] = T(NaN); ar[1, 5] = T(Inf)
    @test LinearFunction(ar, T(0.25))(xr) === T(1.25)
  end

  @testset "Recovery of an overflowing value ($T)" for T in (Float32, Float64)
    p = 100
    huge, d = T == Float64 ? (T(1e308), T(-1.5e308)) : (T(2e38), T(-3e38))
    x = make_iterate(T, p, [(1, 1, huge), (2, 1, huge)])
    ref = reference(ones(T, p, p), d, x, T)
    expected = T(ref.value)
    @test isfinite(expected) && ref.value > 0
    # finite coefficients; the Inf, NaN and -Inf at zeros of x (unstored, stored as ±0)
    @test agrees(LinearFunction(ones(T, p, p), d)(x), ref, T)
    @test LinearFunction(ones(T, p, p), d)(x) ≈ expected rtol = 12 * eps(T)
    xz = make_iterate(T, p, [(1, 1, huge), (2, 1, huge), (3, 3, zero(T)), (4, 4, -zero(T))])
    for bad in (T(Inf), T(NaN), -T(Inf)), (r, c) in ((5, 5), (3, 3), (4, 4))
      a = ones(T, p, p); a[r, c] = bad
      for z in (x, xz)
        v = LinearFunction(a, d)(z)
        @test isfinite(v) && agrees(v, ref, T)
        @test v ≈ expected rtol = 12 * eps(T)
      end
    end
    # d keeps its own units, and the sign
    for (dv, ok) in ((-3huge / 2, true), (-huge, true), (T(1), true), (T(0), true))
      f = LinearFunction(ones(T, p, p), dv)
      @test check(f, x, T)
    end
    @test LinearFunction(ones(T, p, p), T(0.5))(make_iterate(T, p, [(1, 1, huge), (2, 1, huge), (3, 1, -huge), (4, 1, -huge)])) === T(0.5)
    # the same worked values of the issue
    if T == Float64
      @test LinearFunction(ones(T, p, p), d)(x) === 5.0e307
      a = ones(p, p); a[5, 5] = Inf
      @test LinearFunction(a, d)(x) === 5.0e307
    else
      a = ones(T, p, p); a[7, 7] = NaN32
      @test LinearFunction(a, d)(x) ≈ 1.0f38 rtol = 1.0f-6
    end
    # a participating coefficient or entry that is not finite keeps the ordinary value
    a = ones(T, p, p); a[1, 1] = T(Inf)
    @test LinearFunction(a, d)(x) === T(Inf)
    a[1, 1] = T(NaN)
    @test isnan(LinearFunction(a, d)(x))
    a = ones(T, p, p); a[2, 1] = -T(Inf)
    @test LinearFunction(a, d)(x) === -T(Inf)
    xi = make_iterate(T, p, [(1, 1, huge), (2, 1, T(Inf))])
    @test LinearFunction(ones(T, p, p), d)(xi) === T(Inf)
    xn = make_iterate(T, p, [(1, 1, huge), (2, 1, huge), (3, 1, T(NaN))])
    @test isnan(LinearFunction(ones(T, p, p), d)(xn))
    # the nonfinite constant: NaN, as on the dense path, and unchanged
    xneg = make_iterate(T, p, [(1, 1, -huge), (2, 1, -huge)])
    for (xx, dv) in ((xneg, T(Inf)), (x, -T(Inf)))
      f = LinearFunction(ones(T, p, p), dv)
      @test isnan(f(xx))
      @test isnan(f(Matrix(xx)))
    end
    @test LinearFunction(ones(T, p, p), T(Inf))(x) === T(Inf)            # the dot is Inf too
    @test LinearFunction(ones(T, p, p), T(Inf))(make_iterate(T, p, [(1, 1, T(1))])) === T(Inf)
    # a value beyond the range is Inf of its sign
    @test LinearFunction(ones(T, p, p), huge)(x) === T(Inf)
    @test LinearFunction(ones(T, p, p), -huge)(xneg) === -T(Inf)
  end

  @testset "Cancellation, boundary and small values ($T)" for T in (Float32, Float64)
    p = 20
    emax = exponent(floatmax(T))
    M = emax + 1                                            # 2^M = 2 * 2^emax is the first power beyond the range
    margin = T == Float64 ? 40 : 16                         # a relative 2^-margin exceeds the R1 bound of 3 terms
    h = ldexp(one(T), emax)                                 # 2^(M-1)
    ones_p = ones(T, p, p)
    f0 = LinearFunction(ones_p, zero(T))
    # a dot of exactly zero from large entries, against a small d
    four = make_iterate(T, p, [(1, 1, ldexp(one(T), emax)), (2, 1, ldexp(one(T), emax)), (3, 1, -ldexp(one(T), emax)), (4, 1, -ldexp(one(T), emax))])
    small = T == Float64 ? T(1e-300) : T(1e-30)
    @test LinearFunction(ones_p, small)(four) === small
    @test LinearFunction(ones_p, -small)(four) === -small
    @test LinearFunction(ones_p, zero(T))(four) === zero(T)
    # the overflow boundary: 2^emax + 2^emax - 2^(M - precision(T)) is floatmax(T) (M = emax + 1)
    pair(extra) = make_iterate(T, p, [(1, 1, h), (2, 1, h), (3, 1, extra)])
    exact = -ldexp(one(T), emax + 1 - precision(T))        # 2^M - 2^(M-p) = floatmax(T)
    @test f0(pair(exact)) === floatmax(T)
    @test f0(make_iterate(T, p, [(1, 1, -h), (2, 1, -h), (3, 1, -exact)])) === -floatmax(T)
    below = -ldexp(one(T), M - margin)                      # the true value is below floatmax by a relative 2^-margin
    above = ldexp(one(T), M - margin)                       # and above 2^M
    @test isfinite(f0(pair(below))) && check(f0, pair(below), T)
    @test f0(pair(below)) ≈ ldexp(T(2) - ldexp(one(T), 1 - margin), emax) rtol = 12 * eps(T)
    @test f0(pair(above)) === T(Inf)
    @test isfinite(f0(make_iterate(T, p, [(1, 1, -h), (2, 1, -h), (3, 1, -below)])))
    @test f0(make_iterate(T, p, [(1, 1, -h), (2, 1, -h), (3, 1, -above)])) === -T(Inf)
    @test f0(make_iterate(T, p, [(1, 1, h), (2, 1, h)])) === T(Inf)             # 2^M exactly
    # with c: a coefficient scales the entry, a product beyond the range is recovered when the sum is inside
    a = ones(T, p, p); a[1, 1] = T(4); a[2, 1] = T(-4); a[3, 1] = T(2)
    x = make_iterate(T, p, [(1, 1, h), (2, 1, h), (3, 1, h / 2)])
    @test LinearFunction(a, zero(T))(x) === h                                    # 4h - 4h + h = h
    # entries and coefficients in the binades near the top of the range: BigFloat references
    rng = Xoshiro(810_001 + (T == Float32))
    for _ in 1:40
      pairs = [(rand_mag(rng, T, emax - 4, emax - 1), rand_mag(rng, T, 0, 3)) for _ in 1:4]   # products beyond the range
      xs, cs = [u for (u, _) in pairs], [w for (_, w) in pairs]
      pos = [(1, 1), (2, 1), (3, 2), (5, 5)]
      a = ones(T, p, p); x = SparseIterate(T, p, p)
      for ((r, c), u, w) in zip(pos, xs, cs)
        a[r, c] = w; x[r, c] = u
      end
      # a second group that nearly cancels the first: x' = -(c x) / c'
      for ((r, c), (r2, c2)) in zip(pos, [(7, 1), (8, 2), (9, 3), (10, 4)])
        a[r2, c2] = abs(a[r, c]) * T(0.5 + rand(rng))
        x[r2, c2] = T(-(big(a[r, c]) * big(x[r, c]) / big(a[r2, c2])))
      end
      for dv in (zero(T), T(1), -T(2)^emax)
        @test check(LinearFunction(a, dv), x, T)
      end
    end
    # products that underflow, subnormal entries and results
    emin = exponent(nextfloat(zero(T)))
    for _ in 1:40
      k = rand(rng, 1:6)
      pos = randperm(rng, p^2)[1:k]
      x = SparseIterate(T, p, p); a = ones(T, p, p)
      for i in pos
        x[i] = rand_mag(rng, T, emin, emin + 20 + rand(rng, 0:80))
        a[i] = rand_mag(rng, T, -60, 0)                                              # products of different sizes, many below floatmin
      end
      for dv in (zero(T), floatmin(T) / 4, T(1), -floatmin(T))
        f = LinearFunction(a, dv)
        r = reference(f, x, T)
        @test agrees(f(x), r, T)
      end
    end
    sub = nextfloat(zero(T))
    xs = make_iterate(T, p, [(1, 1, sub), (2, 1, T(3) * sub), (3, 1, -sub)])
    @test f0(xs) === T(3) * sub
    @test LinearFunction(ones_p, T(2) * sub)(xs) === T(5) * sub
    @test f0(make_iterate(T, p, [(1, 1, sub)])) === sub
    # a small value beside large ones that cancel: the value is within the R1 bound
    xb = make_iterate(T, p, [(1, 1, h), (2, 1, h), (3, 1, -h), (4, 1, -h), (5, 5, small)])
    @test check(f0, xb, T)
  end

  @testset "Random matrices and densities ($T)" for T in (Float32, Float64)
    rng = Xoshiro(810_100 + (T == Float32))
    for p in (3, 8, 17), k in unique(clamp.((1, 2, p, p^2 ÷ 4, p^2 ÷ 2, p^2 - 1, p^2), 1, p^2))
      for (lox, hix, loc, hic) in ((-3, 3, -3, 3), (0, 6, -2, 2))
        x, a = random_pair(rng, T, p, k, lox, hix, loc, hic)
        d = T(rand(rng) - 0.5)
        f = LinearFunction(a, d)
        r = reference(f, x, T)
        v = f(x)
        @test agrees(v, r, T) && isfinite(v)
        # the dense value agrees where the coefficients at zeros are finite
        vd = f(Matrix(x))
        rd = (setprecision(BigFloat, 4400) do
                sum(big.(a) .* big.(Matrix(x))) + big(d)
              end)
        @test abs(big(v) - big(vd)) <= r.bound + 4 * length(a) * big(eps(T)) * (r.sabs) + length(a) * big(floatmin(T)) * big(eps(T))
        @test abs(big(vd) - rd) <= 4 * length(a) * big(eps(T)) * r.sabs + length(a) * big(floatmin(T)) * big(eps(T))
        # nonfinite coefficients at zeros of x leave the value as is; a stored-order permutation of the same data too
        a2 = copy(a)
        for i in 1:length(a2)
          (!isactive(x, i) || iszero(x[i])) && rand(rng) < 0.5 && (a2[i] = rand(rng, (T(NaN), T(Inf), -T(Inf))))
        end
        @test LinearFunction(a2, d)(x) === v
      end
    end
    # a shuffled activation order of the same entries
    p = 9
    x, a = random_pair(rng, T, p, 40, -2, 2, -2, 2)
    ids, vals = collect(activecoordinates(x)), collect(activevalues(x))
    y = SparseIterate(T, p, p)
    for j in randperm(rng, length(ids))
      y[ids[j]] = one(T); y[ids[j]] = vals[j]
    end
    f = LinearFunction(a, T(0.5))
    @test agrees(f(y), reference(f, y, T), T) && agrees(f(x), reference(f, x, T), T)
    @test abs(big(f(x)) - big(f(y))) <= 2 * reference(f, x, T).bound
    # random magnitudes near the range: the classification of the reference decides
    for _ in 1:60
      p = 6
      x, a = random_pair(rng, T, p, rand(rng, 1:12), exponent(floatmax(T)) - 6, exponent(floatmax(T)) - 1, 0, 2)
      for dv in (zero(T), T(1), floatmax(T) / 2, -floatmax(T) / 2)
        f = LinearFunction(a, dv)
        @test check(f, x, T)
      end
    end
  end

  @testset "Mixed element types" begin
    p = 12
    rng = Xoshiro(810_200)
    for (Tc, Tx, Td) in ((Float64, Float32, Float32), (Float32, Float64, Float64), (Float64, Float64, Float32), (Int, Float64, Int), (Float32, Float32, Int))
      x, _ = random_pair(rng, Tx, p, 20, -2, 2, 0, 0)
      a = Tc <: Integer ? rand(rng, Tc(-3):Tc(3), p, p) : randn(rng, Tc, p, p)
      f = LinearFunction(a, Td(2))
      r = reference(f, x, Tx)
      @test abs(big(f(x)) - r.value) <= 4 * r.n * big(eps(Tx)) * r.sabs * 2
      @test typeof(f(x)) === Tx
    end
  end

  @testset "Coefficient storage" begin
    p = 10
    rng = Xoshiro(810_300)
    for T in (Float32, Float64)
      x, a = random_pair(rng, T, p, 25, -2, 2, -2, 2)
      d = T(0.25)
      for c in (sparse(a), a', transpose(a), view(a, :, :), view(randn(rng, T, p + 3, p + 2), 2:p+1, 1:p), copy(a'))
        f = LinearFunction(c, d)
        @test agrees(f(x), reference(c, d, x, T), T)
      end
      # a sparse coefficient matrix with a stored zero
      s = sparse(a); s[1, 1] = 0
      @test agrees(LinearFunction(s, d)(x), reference(s, d, x, T), T)
    end
  end

  @testset "Diagonal coefficients ($T)" for T in (Float32, Float64)
    p = 30
    x = make_iterate(T, p, [(1, 1, T(2)), (2, 2, zero(T)), (5, 5, -zero(T)), (1, 2, T(NaN)), (9, 3, T(4))])
    v = ones(T, p); v[2] = T(NaN); v[5] = T(Inf); v[4] = T(NaN)           # NaN at the stored zero, Inf at -0, NaN at an implicit zero
    f = LinearFunction(Diagonal(v), T(0.25))
    @test f(x) === T(2.25)                                                  # the NaN of x off the diagonal is ignored
    @test isnan(f(Matrix(x)))                                               # the dense dot of a Diagonal reads the diagonal of x only: NaN at the stored zero
    # off-diagonal NaN alone, finite coefficients
    @test LinearFunction(Diagonal(ones(T, p)), T(0.25))(make_iterate(T, p, [(1, 2, T(NaN)), (7, 3, T(1))])) === T(0.25)
    @test LinearFunction(Diagonal(ones(T, p)), T(0.25))(make_iterate(T, p, [(1, 1, T(1)), (1, 2, T(NaN))])) === T(1.25)
    # a nonzero diagonal entry keeps ordinary arithmetic
    v2 = ones(T, p); v2[1] = T(NaN)
    @test isnan(LinearFunction(Diagonal(v2), T(0.25))(x))
    # recovery
    huge, d = T == Float64 ? (T(1e308), T(-1.5e308)) : (T(2e38), T(-3e38))
    xb = make_iterate(T, p, [(1, 1, huge), (2, 2, huge), (3, 3, zero(T)), (4, 4, -zero(T)), (1, 2, T(NaN)), (6, 5, T(Inf))])
    vb = ones(T, p); vb[3] = T(Inf); vb[4] = T(NaN); vb[7] = T(NaN)
    fb = LinearFunction(Diagonal(vb), d)
    rb = reference(fb, xb, T)
    @test isfinite(fb(xb)) && agrees(fb(xb), rb, T)
    @test fb(xb) ≈ T(rb.value) rtol = 12 * eps(T)
    vb[2] = T(Inf)                                                          # a participating coefficient: the ordinary value
    @test LinearFunction(Diagonal(vb), d)(xb) === T(Inf)
    # random diagonals and iterates
    rng = Xoshiro(810_400 + (T == Float32))
    for _ in 1:30
      k = rand(rng, 1:p^2 ÷ 8)
      xr, _ = random_pair(rng, T, p, k, -2, 2, 0, 0)
      for j in rand(rng, 1:p, 4)                                           # put some entries on the diagonal
        xr[j, j] = rand_mag(rng, T, -2, 2)
      end
      dg = T[rand_mag(rng, T, -2, 2) for _ in 1:p]
      for dv in (zero(T), T(0.5))
        f = LinearFunction(Diagonal(dg), dv)
        @test agrees(f(xr), reference(f, xr, T), T)
        @test abs(big(f(xr)) - big(f(Matrix(xr)))) <= 8 * p * big(eps(T)) * (reference(f, xr, T).sabs + 1)
      end
    end
    # the large diagonal entries
    for _ in 1:20
      xr = SparseIterate(T, p, p)
      dg = ones(T, p)
      for j in 1:p
        rand(rng) < 0.5 && (xr[j, j] = rand_mag(rng, T, exponent(floatmax(T)) - 4, exponent(floatmax(T)) - 1))
        rand(rng) < 0.2 && (xr[j, 1+mod(j, p)] = T(NaN))
      end
      for dv in (zero(T), floatmax(T) / 2, -floatmax(T) / 2)
        f = LinearFunction(Diagonal(dg), dv)
        @test agrees(f(xr), reference(f, xr, T), T)
      end
    end
  end

  @testset "Allocations of a warmed call ($T)" for T in (Float32, Float64)
    p = 50
    huge, d = T == Float64 ? (T(1e308), T(-1.5e308)) : (T(2e38), T(-3e38))
    x = make_iterate(T, p, [(1, 1, T(2)), (2, 1, T(0.5)), (3, 3, zero(T))])
    xb = make_iterate(T, p, [(1, 1, huge), (2, 1, huge), (3, 3, zero(T))])
    a = ones(T, p, p); a[3, 3] = T(NaN); a[4, 4] = T(Inf)
    @test allocated(LinearFunction(a, T(0.25)), x) == 0
    @test allocated(LinearFunction(a, d), xb) == 0
    @test isfinite(LinearFunction(a, d)(xb))
    dg = ones(T, p); dg[3] = T(NaN)
    @test allocated(LinearFunction(Diagonal(dg), T(0.25)), x) == 0
    @test allocated(LinearFunction(Diagonal(dg), d), xb) == 0
    full = make_iterate(T, p, [(r, c, T(r + c)) for c in 1:p for r in 1:p])
    @test allocated(LinearFunction(a, T(0.25)), full) == 0
  end

  @testset "Oracles agree with the value ($T)" for T in (Float32, Float64)
    p = 7
    rng = Xoshiro(810_500 + (T == Float32))
    x, a = random_pair(rng, T, p, 18, -1, 1, -1, 1)
    a[findfirst(i -> !isactive(x, i), 1:p^2)] = T(NaN)            # a NaN coefficient at an unstored entry
    d = T(0.625); γ = T(0.375)
    f = LinearFunction(a, d)
    v = f(x)
    @test isfinite(v) && agrees(v, reference(f, x, T), T)
    dense_out(y) = Matrix(y)
    # gradient and value_and_gradient into dense and iterate outputs: the coefficients
    g = fill(T(7), p, p)
    @test gradient!(g, f, x) === g && isequal(g, a)
    g = fill(T(7), p, p)
    @test isequal(value_and_gradient!(g, f, x), v) && isequal(g, a)
    gi = SparseIterate(T, p, p)
    @test isequal(value_and_gradient!(gi, f, x), v) && isequal(Matrix(gi), a)
    @test isequal(Matrix(gradient!(SparseIterate(T, p, p), f, x)), a)
    gv, gg = value_and_gradient(f, x)
    @test isequal(gv, v) && isequal(Array(gg), a)
    @test isequal(Array(gradient(f, x)), a)
    # prox: x - γc, a NaN coefficient gives a NaN entry (the proximal map is unchanged)
    expected = Matrix(x) .- γ .* a
    y = zeros(T, p, p)
    @test prox!(y, f, x, γ) === y && isequal(y, expected)
    @test isequal(Matrix(prox(f, x, γ)), expected)
    yi = SparseIterate(T, p, p)
    @test prox!(yi, f, x, γ) === yi && isequal(Matrix(yi), expected)
    # finite coefficients: the value at the proximal point, through value_and_prox!
    a = copy(a); a[isnan.(a)] .= T(1)
    f = LinearFunction(a, d)
    expected = Matrix(x) .- γ .* a
    for y in (zeros(T, p, p), SparseIterate(T, p, p))
      val = value_and_prox!(y, f, x, γ)
      @test isapprox(Matrix(y), expected; rtol = 4 * eps(T))
      @test abs(big(val) - (sum(big.(a) .* big.(Matrix(y))) + big(d))) <= 4 * length(a) * big(eps(T)) * (sum(abs, big.(a) .* big.(Matrix(y))) + abs(big(d)))
    end
    @test isequal(Matrix(prox(f, x, γ)), Matrix(x) .- γ .* a)
    # Scaled: the value is α f(x), also with a NaN coefficient at a zero; a leaf value out of range stays Inf
    f = LinearFunction([T(1) T(NaN); T(1) T(1)], d)
    x2 = make_iterate(T, 2, [(1, 1, T(2))])
    for α in (T(2), T(0.5), T(3))
      s = α * f
      @test s(x2) ≈ α * f(x2) rtol = 4 * eps(T)
      @test isfinite(s(x2))
    end
    huge = T == Float64 ? T(1e308) : T(2e38)
    fb = LinearFunction(ones(T, 2, 2), zero(T))
    xb = make_iterate(T, 2, [(1, 1, huge), (2, 1, huge)])
    @test fb(xb) === T(Inf)
    @test (T(0.25) * fb)(xb) === T(Inf)
  end

  @testset "Shapes are checked before any read or write ($T)" for T in (Float32, Float64)
    d = T(0.5); γ = T(0.25)
    A, v6 = ones(T, 2, 3), ones(T, 6)
    cases = ((LinearFunction(A, d), SparseIterate(T, 3, 2)), (LinearFunction(v6, d), SparseIterate(T, 3, 2)),
             (LinearFunction(ones(T, 3, 2), d), SparseIterate(T, 2, 3)), (LinearFunction(sparse(A), d), SparseIterate(T, 3, 2)),
             (LinearFunction(Diagonal(ones(T, 3)), d), SparseIterate(T, 1, 9)))
    for (f, x) in cases
      x[1] = T(1)
      xs = state(x)
      for h in (f, T(2) * f)
        @test_throws DimensionMismatch h(x)
        @test_throws DimensionMismatch prox(h, x, γ)
        @test_throws DimensionMismatch gradient(h, x)
        @test_throws DimensionMismatch value_and_gradient(h, x)
        for out in (fill(T(7), size(x)), SparseIterate(T, size(x)...))
          before = state(out)
          for op! in (y -> gradient!(y, h, x), y -> value_and_gradient!(y, h, x), y -> prox!(y, h, x, γ), y -> value_and_prox!(y, h, x, γ))
            @test_throws DimensionMismatch op!(out)
            @test state(out) == before
          end
        end
      end
      @test state(x) == xs
    end
  end

  @testset "Other arguments keep their value ($T)" for T in (Float32, Float64)
    p = 8
    rng = Xoshiro(810_600 + (T == Float32))
    d = T(0.25)
    # dense matrices and views: dot(c, x) + d, and NaN at a zero stays NaN
    X = randn(rng, T, p, p); X[2, 2] = zero(T)
    c = randn(rng, T, p, p)
    @test isequal(LinearFunction(c, d)(X), dot(c, X) + d)
    c2 = copy(c); c2[2, 2] = T(NaN)
    @test isnan(LinearFunction(c2, d)(X))
    @test isequal(LinearFunction(c, d)(view(X, :, :)), dot(c, view(X, :, :)) + d)
    # vector SparseIterate: the stored-entry dot with strong zeros
    xv = SparseIterate(T, 12)
    xv[3] = T(2); xv[5] = T(1); xv[5] = zero(T); xv[9] = T(-1.5)
    cv = randn(rng, T, 12); cv[5] = T(NaN); cv[7] = T(Inf)
    @test isequal(LinearFunction(cv, d)(xv), dot(cv, xv) + d)
    @test isfinite(LinearFunction(cv, d)(xv))
    # a symmetric iterate: the matrix it represents, with the terms of stored zeros formed
    si = SymmetricSparseIterate(T, p)
    for c in 1:p, r in c:p
      isodd(r + c) || (si[r, c] = randn(rng, T))
    end
    cs = randn(rng, T, p, p); cs = cs + cs'
    f = LinearFunction(cs, d)
    ref = sum(big.(cs) .* big.(Matrix(si))) + big(d)
    @test abs(big(f(si)) - ref) <= 4 * p^2 * big(eps(T)) * (sum(abs, big.(cs) .* big.(Matrix(si))) + abs(big(d)))
    # a stored zero of the symmetric iterate against a NaN coefficient: its term is formed
    sz = SymmetricSparseIterate(T, p)
    sz[2, 1] = T(1); sz[2, 1] = zero(T); sz[1, 1] = T(1)
    csz = ones(T, p, p); csz[2, 1] = T(NaN); csz[1, 2] = T(NaN)
    @test isnan(LinearFunction(csz, d)(sz))
    # dot(c, x) of a matrix iterate and a dense c that is not a Diagonal is the generic dot, which forms c[i] * 0 at the zeros
    xm = make_iterate(T, p, [(1, 1, T(2)), (3, 3, T(-1))])
    am = ones(T, p, p); am[4, 4] = T(NaN)
    @test isnan(dot(am, xm) + d)
    @test LinearFunction(am, d)(xm) === T(1.25)
  end

  @testset "Widely separated exponents ($T)" for T in (Float32, Float64)
    E = exponent(floatmax(T))
    h = ldexp(one(T), E)
    e = T == Float64 ? -46 : -17
    # a small entry times the largest coefficient, beside two large entries that overflow: the exact value
    # 2^(E+1) - floatmax(T) 2^e is inside the range by more than the R1 bound, for a Diagonal and for a dense c
    cs, vs = T[1, 1, floatmax(T)], T[h, h, -ldexp(one(T), e)]
    for diagonal in (true, false)
      x = SparseIterate(T, 3, 3)
      c = diagonal ? Diagonal(cs) : ones(T, 3, 3)
      for j in 1:3
        x[j, j] = vs[j]
        diagonal || (c[j, j] = cs[j])
      end
      f = LinearFunction(c, zero(T))
      ref = reference(f, x, T)
      v = f(x)
      @test isfinite(v) && agrees(v, ref, T)
      @test big(floatmax(T)) - ref.value > ref.bound               # the reference is finite by more than the bound
      @test T(ref.value) == (T == Float64 ? 1.7976931348622904e308 : 3.4027977f38)
      @test abs(big(v) - ref.value) <= ref.bound
      # with d of other sizes, in its own units
      for dv in (zero(T), T(1), -h, h / 4)
        g = LinearFunction(c, dv)
        @test check(g, x, T)
      end
    end
    # the cancellation probe: two products of size 2^(E+1) that cancel and a third product floatmax(T) 2^e, about 2^(E+1+e)
    a, tiny = ldexp(one(T), T == Float64 ? 512 : 64), ldexp(one(T), e)
    cs, vs = T[a, -a, floatmax(T), 0], T[a, a, tiny, floatmax(T)]
    for diagonal in (false, true)
      x = diagonal ? make_iterate(T, 4, [(j, j, vs[j]) for j in 1:4]) : make_iterate(T, 2, [(1, 1, vs[1]), (2, 1, vs[2]), (1, 2, vs[3]), (2, 2, vs[4])])
      c = diagonal ? Diagonal(cs) : reshape(cs, 2, 2)
      f = LinearFunction(c, zero(T))
      ref = reference(f, x, T)
      v = f(x)
      @test !iszero(v) && isfinite(v) && agrees(v, ref, T)
      @test Float64(ref.value) == (T == Float64 ? 2.554675596204441e294 : 2.596148274524909e33)
      @test abs(big(v) - ref.value) <= ref.bound
    end
    # families: separated exponents, with and without cancellation, d zero, tiny, one, large
    rng = Xoshiro(810_700 + (T == Float32))
    for diagonal in (false, true), cancel in (false, true), _ in 1:150
      x, c = separated_pair(rng, T, 7; diagonal = diagonal, cancel = cancel)
      for dv in (zero(T), ldexp(T(0.7), exponent(nextfloat(zero(T))) + 6), T(1), ldexp(T(-1.3), E - 3), ldexp(T(1.1), E - 1))
        @test check(LinearFunction(c, dv), x, T)
      end
    end
  end

  @testset "Zeros made by activation and by arithmetic ($T)" for T in (Float32, Float64)
    p = 6
    x = SparseIterate(T, p, p)
    x[1, 1] = T(2)
    activate!(x, linear_of(p, 2, 2)); activate!(x, linear_of(p, 3, 3))
    x[4, 4] = T(3); x[4, 4] -= T(3)                     # a zero made by arithmetic
    x[5, 1] = T(-2); x[5, 1] += T(2)
    @test nactive(x) == 5 && all(isactive(x, linear_of(p, r, c)) for (r, c) in ((1, 1), (2, 2), (3, 3), (4, 4), (5, 1)))
    @test count(iszero, activevalues(x)) == 4
    c = ones(T, p, p)
    for (i, (r, k)) in enumerate(((2, 2), (3, 3), (4, 4), (5, 1), (6, 6)))
      c[r, k] = (T(NaN), T(Inf), -T(Inf))[mod1(i, 3)]
    end
    @test LinearFunction(c, T(0.25))(x) === T(2.25)
    @test isnan(LinearFunction(c, T(0.25))(Matrix(x)))
    # a recovered value with the activated zeros
    huge, d = T == Float64 ? (T(1e308), T(-1.5e308)) : (T(2e38), T(-3e38))
    y = SparseIterate(T, p, p)
    y[1, 1] = huge; y[2, 1] = huge
    activate!(y, linear_of(p, 3, 3)); activate!(y, linear_of(p, 4, 4))
    @test nactive(y) == 4
    f = LinearFunction(c, d)
    @test isfinite(f(y)) && agrees(f(y), reference(f, y, T), T)
  end

  @testset "Nested Scaled ($T)" for T in (Float32, Float64)
    E = exponent(floatmax(T))
    A = round(Int, 0.6 * (E + 1))
    p = 4
    near(v, r) = abs(big(v) - r) <= 8 * big(eps(T)) * abs(r) + big(floatmin(T)) * big(eps(T))
    # a leaf value of size 2^(E - 23), and scales whose product is a normal number, one that underflows, one that overflows
    L = E - 23
    x = make_iterate(T, p, [(1, 1, ldexp(one(T), L))]); f = LinearFunction(ones(T, p, p), zero(T))
    leaf = f(x)
    @test leaf == ldexp(one(T), L)
    for (α, β) in ((T(2), T(0.5)), (ldexp(one(T), -A), ldexp(one(T), -A)), (ldexp(one(T), A), ldexp(one(T), -A)), (ldexp(T(1.5), -A), ldexp(T(1.25), -A)))
      v = Scaled(Scaled(f, α), β)(x)
      @test near(v, big(leaf) * big(α) * big(β))
      @test isfinite(v) && !iszero(v)
    end
    xs = make_iterate(T, p, [(1, 1, ldexp(one(T), -L))]); leaf = f(xs)
    v = Scaled(Scaled(f, ldexp(one(T), A)), ldexp(one(T), A))(xs)       # the product of the scales overflows
    @test near(v, big(leaf) * big(ldexp(one(T), A))^2) && isfinite(v)
    # a leaf recovered from an overflowing dot, scaled
    huge, d = T == Float64 ? (T(1e308), T(-1.5e308)) : (T(2e38), T(-3e38))
    xb = make_iterate(T, p, [(1, 1, huge), (2, 1, huge)])
    fb = LinearFunction(ones(T, p, p), d)
    lb = fb(xb)
    @test isfinite(lb)
    for (α, β) in ((T(0.5), T(0.25)), (T(2), T(0.125)))
      @test near(Scaled(Scaled(fb, α), β)(xb), big(lb) * big(α) * big(β))
    end
    @test near((T(0.5) * (T(2) * fb))(xb), big(lb))
  end

  @testset "Order-dependent classes ($T)" for T in (Float32, Float64)
    h = ldexp(one(T), exponent(floatmax(T)))
    # finite coefficients, and an overflow before an infinite entry of the other sign: the order of the summation
    # decides the class (NaN, or -Inf), and the dense dot takes the column-major order (-Inf)
    x = SparseIterate(T, 2, 2)
    x[2] = h; x[3] = h; x[1] = -T(Inf)
    f = LinearFunction(ones(T, 2, 2), zero(T))
    @test isnan(f(x)) || f(x) === -T(Inf)
    @test f(Matrix(x)) === -T(Inf) && dot(f.c, x) === -T(Inf)
    # the same with a Diagonal coefficient array
    xd = SparseIterate(T, 3, 3)
    xd[2, 2] = h; xd[3, 3] = h; xd[1, 1] = -T(Inf)
    fd = LinearFunction(Diagonal(ones(T, 3)), zero(T))
    @test isnan(fd(xd)) || fd(xd) === -T(Inf)
  end

  # The main loop of the ordinary sum has no branch on the value of an entry (the search for the first stored nonzero
  # entry does): a zero entry takes the coefficient index of the first stored nonzero entry and its term is dropped by a
  # select.
  @testset "Select on a zero entry ($T)" for T in (Float32, Float64)
    p = 40
    sums = (a, x, T) -> PB._linear_stored_dot(a, x)
    variants = (sums,)
    # the first stored nonzero entry (1,1) follows stored zeros and carries Inf, -Inf or NaN, or the finite 3 or 0: only
    # its own term counts, and the coefficients at the zeros are NaN and infinite
    for cv in (T(Inf), -T(Inf), T(NaN), T(3), zero(T))
      x = make_iterate(T, p, [(2, 2, zero(T)), (1, 1, T(2)), (3, 3, -zero(T)), (5, 5, zero(T))])
      a = ones(T, p, p); a[1, 1] = cv
      a[2, 2] = T(NaN); a[3, 3] = T(Inf); a[5, 5] = -T(Inf); a[7, 7] = T(NaN)
      expected = T(2) * cv
      for dot_variant in variants
        @test isequal(dot_variant(a, x, T), expected)
      end
      @test isequal(LinearFunction(a, T(0.25))(x), expected + T(0.25))
      # a second participating entry with the opposite infinity: ordinary arithmetic
      y = make_iterate(T, p, [(2, 2, zero(T)), (1, 1, T(2)), (4, 4, T(1))])
      b = copy(a); b[4, 4] = -T(Inf)
      for dot_variant in variants
        @test isequal(dot_variant(b, y, T), T(2) * cv + -T(Inf))
      end
    end
    # no nonzero stored entry: the sum is +0, also when the coefficients are NaN
    for x in (SparseIterate(T, p, p), make_iterate(T, p, [(1, 1, zero(T)), (2, 2, -zero(T))]))
      a = fill(T(NaN), p, p)
      for dot_variant in variants
        @test dot_variant(a, x, T) === zero(T)
      end
    end
    # random iterates of densities from one entry to fully stored, with stored zeros, and nonfinite coefficients at zeros: the sum agrees with the reference
    rng = Xoshiro(810_800 + (T == Float32))
    for p in (3, 9, 17), k in unique(clamp.((1, 2, p, p^2 ÷ 4, p^2 ÷ 3, p^2 ÷ 2, p^2 - 1, p^2), 1, p^2))
      x, a = random_pair(rng, T, p, k, -3, 3, -3, 3)
      ref = reference(a, zero(T), x, T)
      a2 = copy(a)
      for i in 1:length(a2)
        (!isactive(x, i) || iszero(x[i])) && (a2[i] = rand(rng, (T(NaN), T(Inf), -T(Inf))))
      end
      for dot_variant in variants
        v = dot_variant(a, x, T)
        @test abs(big(v) - ref.value) <= ref.bound
        @test isequal(dot_variant(a2, x, T), v)
      end
      f = LinearFunction(a2, T(0.5))
      @test agrees(f(x), reference(f, x, T), T)
    end
  end

  # A small d after a dot that cancels (the dot is recomputed from exponent-separated terms because it overflows): the
  # recovery collapses the pair of the compensated sum before d enters, so an exactly cancelling dot gives d exactly.
  @testset "A small d after a cancelling dot ($T)" for T in (Float32, Float64)
    for Tc in (T === Float32 ? (Float32, Float64) : (Float64,))
      R = promote_type(T, Tc)
      E = exponent(floatmax(R))
      xexp = T === Float32 ? 64 : 512
      xa = ldexp(one(T), xexp)
      # the reported case: products a^2 = 2^(E+1) (overflow), a, -a^2, -a with a = 2^(E+1)/2 for uniform types; for a
      # Float32 iterate with Float64 coefficients, coefficients 2^(E+1-xexp) and 2^((E+1)/2-xexp)
      cbig, csmall = ldexp(one(Tc), E + 1 - xexp), ldexp(one(Tc), (E + 1) ÷ 2 - xexp)
      cs = Tc[cbig, csmall, -cbig, -csmall]
      for d in (T(0.25), -T(0.25), T(3), ldexp(T(1.5), -20)), diagonal in (true, false), order in ORDERS4
        x = SparseIterate(T, 4, 4)                                      # activated in the order `order`
        for j in order
          x[j, j] = one(T); x[j, j] = xa
        end
        c = diagonal ? Diagonal(cs) : Matrix(Diagonal(cs))
        f = LinearFunction(c, d)
        @test !isfinite(PB._linear_stored_dot(c, x) + d)                  # the ordinary value overflows: the recovery runs
        @test f(x) === d
      end
      # the coefficient order is also permuted (the products cancel in another order)
      x = SparseIterate(T, 4, 4)
      for j in 1:4
        x[j, j] = one(T); x[j, j] = xa
      end
      for perm in ORDERS4, d in (T(0.25), -T(0.25))
        @test LinearFunction(Diagonal(cs[perm]), d)(x) === d
        @test LinearFunction(Matrix(Diagonal(cs[perm])), d)(x) === d
      end
      # exactly cancelling sets with more terms, in a random order: d exactly, ordinary value overflowing
      rng = Xoshiro(810_900 + 7 * (T === Float32) + (Tc === Float32))
      counted = 0
      for _ in 1:2000
        terms = cancelling_terms(rng, T, Tc, 100, rand(rng, 1:4), 3)
        terms === nothing && continue
        p = length(terms)
        x = diagonal_iterate(T, p, first.(terms)); cd = Tc[t[2] for t in terms]
        for d in (T(0.25), -T(0.25))
          isfinite(PB._linear_diagonal_dot(cd, x) + d) && continue
          counted += 1
          @test LinearFunction(Diagonal(cd), d)(x) === d
          @test LinearFunction(Matrix(Diagonal(cd)), d)(x) === d
        end
      end
      @test counted >= 40
    end
  end

  # Near cancellation: a cancelling set and one more product delta, beside d. The value is within the R1 bound of
  # delta + d, and whenever the error budget of the argument is small relative to |delta| + |d| it is also within that
  # budget: 4 eps(T) (|delta| + |d|) for the roundings of delta and d, 4 n eps(A)^2 sum|terms| for the compensated sum
  # (A = Float64 the type of the accumulator) and 2 n 2^(ex - 1074) for the terms below the frame of the largest
  # product (2^ex, ex <= E + 3 here), which the accumulator drops. The budget is not small for a delta much smaller than the terms.
  @testset "Near cancellation ($T)" for T in (Float32, Float64)
    for Tc in (T,)        # a mixed pair would need |delta| below floatmax(Float32) beside terms of size 2^1024: no relative claim
      R = promote_type(T, Tc)
      E = exponent(floatmax(R))
      rng = Xoshiro(810_950 + 7 * (T === Float32) + (Tc === Float32))
      within = 0
      for _ in 1:500
        terms = cancelling_terms(rng, T, Tc, 20, rand(rng, 1:3), 2)
        terms === nothing && continue
        # the extra product delta = (s1 2^k)(s2 2^l)-style with a significand of at most 4 bits and an exponent so that
        # |delta| is far above the budget of the large terms; d of a size from far below to far above delta
        k = T === Float32 ? rand(rng, 70:120) : rand(rng, 985:1000)
        dk = min(k + rand(rng, -30:30), T === Float32 ? 120 : 1000)
        xe = rand(rng, 1:20)
        δx, δc = ldexp(T(rand(rng, 1:7)), xe), ldexp(Tc((rand(rng, Bool) ? 1 : -1) * rand(rng, 1:7)), k - xe)
        push!(terms, (δx, δc))
        p = length(terms)
        x = diagonal_iterate(T, p, first.(terms)); cd = Tc[t[2] for t in terms]
        d = (rand(rng, Bool) ? one(T) : -one(T)) * ldexp(T(rand(rng, 1:7)), dk)
        isfinite(PB._linear_diagonal_dot(cd, x) + d) && continue
        for c in (Diagonal(cd), Matrix(Diagonal(cd)))
          f = LinearFunction(c, d)
          ref = reference(f, x, T)
          v = f(x)
          @test isfinite(v) && abs(big(v) - ref.value) <= ref.bound                 # R1
          δ = big(δx) * big(δc)
          sabs = sum(abs(big(a) * big(b)) for (a, b) in terms)
          budget = 4 * big(eps(T)) * (abs(δ) + abs(big(d))) + 4 * length(terms) * big(eps(Float64))^2 * sabs +
                   2 * length(terms) * ldexp(big(1.0), E + 3 - 1074)
          if budget <= 16 * big(eps(T)) * (abs(δ) + abs(big(d)))
            within += 1
            @test abs(big(v) - (δ + big(d))) <= budget
          end
        end
      end
      @test within >= 100
    end
  end
end

end
