module LinearSparseValueTest
using Test, ProximalBase, LinearAlgebra, SparseArrays, Random
const PB = ProximalBase

# The value f(x) of f = LinearFunction(c, d) at a matrix SparseIterate x (#810): the sum of c[i] * x[i] over the
# stored entries of x whose value is nonzero (D1: an entry that is not stored, or is stored as +0 or -0, adds
# nothing, even against a NaN or infinite coefficient; a nonzero entry follows ordinary arithmetic), then d. When
# the ordinary value is not finite and d, the nonzero entries and the coefficients at them are finite, the value
# is recomputed in units of a power of two (R1). A Diagonal coefficient array is read at the diagonal of x only.
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
# for the exponents of Float32 and Float64 products; the number n of those entries, Σ|cᵢxᵢ| + |d|, and the R1 bound
# 4 n eps(T) (Σ|cᵢxᵢ| + |d|) + n floatmin(T) eps(T) of a value of type T
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
      @test isnan(f(Matrix(x)))                 # the dense value divergence: dot over every position
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
    # a small value beside a large one: the small one is within the R1 bound, not necessarily kept
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
    # dot(c, x) of a matrix iterate is the generic dot: f(x) agrees with it up to rounding unless a coefficient at a zero is not finite
    xm = make_iterate(T, p, [(1, 1, T(2)), (3, 3, T(-1))])
    am = ones(T, p, p); am[4, 4] = T(NaN)
    @test isnan(dot(am, xm) + d)
    @test LinearFunction(am, d)(xm) === T(1.25)
  end
end

end
