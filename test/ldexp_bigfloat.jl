module LdexpBigFloatTest

# ProximalBase defines its own ldexp (src/ProximalBase.jl): Base's ldexp(::BigFloat, n) is the exact mpfr_mul_2si only for a
# Clong n, and where a C long is 32 bits (Windows) an Int shift is x * exp2(BigFloat(n)), which overflows when 2^n alone is
# out of range and rounds n at a low precision (#747). This file tests the contract of the module's function and the results
# that depended on it, against references built from Clong-typed Base shifts and a high-precision scaled sum, with no
# package helper. The cases that need the exponent range of Windows (2^30 - 1) run under it on every platform.

using Test, ProximalBase, JLArrays      # JLArrays loads the GPU extension

const PB = ProximalBase

# Run f with the MPFR exponent range [emin, emax] and restore the previous range. Everything that depends on the range
# (floatmax(BigFloat), the numbers) must be built inside f.
function with_exponent_range(f, emin, emax)
  old_min, old_max = Base.MPFR.get_emin(), Base.MPFR.get_emax()
  try
    Base.MPFR.set_emax!(emax)
    Base.MPFR.set_emin!(emin)
    return f()
  finally
    Base.MPFR.set_emin!(old_min)
    Base.MPFR.set_emax!(old_max)
  end
end
const WIN = 2^30 - 1
windows_range(f) = with_exponent_range(f, -WIN, WIN)
default_range(f) = f()

exp_clong(n) = Clong(clamp(n, typemin(Clong), typemax(Clong)))
function outcome(g)
  try
    return (:ok, g())
  catch err
    return (:err, typeof(err))
  end
end
same(a, b) = a[1] === b[1] && (a[1] === :err ? a[2] === b[2] :
  (typeof(a[2]) === typeof(b[2]) && isequal(a[2], b[2]) && (!(a[2] isa AbstractFloat) || signbit(a[2]) == signbit(b[2]))))

####################################
#  the contract of ProximalBase.ldexp
####################################

@testset "ProximalBase.ldexp is a function of the package and leaves Base's untouched" begin
  @test PB.ldexp !== Base.ldexp
  @test parentmodule(PB.ldexp) === PB
  @test Core.eval(PB, :ldexp) === PB.ldexp
  @test !any(m -> m.module === PB, methods(Base.ldexp))
  @test all(m -> m.module === PB, methods(PB.ldexp))
end

@testset "ProximalBase.ldexp delegates to Base for the other types" begin
  ns = Any[0, 1, -1, 2, 5, -5, 127, -127, 1023, 1100, -1100, 2000, -2000, 10^6, -10^6, 2^40, -2^40]
  for I in (Int8, Int16, Int32, Int64, Int128, UInt8, UInt16, UInt32, UInt64)
    push!(ns, typemax(I), typemin(I))
  end
  push!(ns, Int8(7), UInt8(3), BigInt(5), BigInt(2)^70, -BigInt(2)^70, Int128(2)^80)
  for T in (Float16, Float32, Float64)
    xs = T[zero(T), -zero(T), one(T), -one(T), T(1.5), T(0.375), floatmin(T), nextfloat(zero(T)), prevfloat(floatmin(T)),
           floatmax(T), -floatmax(T), T(NaN), T(Inf), -T(Inf)]
    bad = [(x, n) for x in xs for n in ns if !same(outcome(() -> PB.ldexp(x, n)), outcome(() -> Base.ldexp(x, n)))]
    @test isempty(bad)
  end
end

@testset "ProximalBase.ldexp on a BigFloat is the exact Clong shift" begin
  emax = Base.MPFR.get_emax()
  for p in (2, 5, 53, 256)
    setprecision(BigFloat, p) do
      xs = BigFloat[big(0.75), -big(0.75), one(BigFloat), -one(BigFloat), big(0.375), floatmax(BigFloat), -floatmax(BigFloat),
                    nextfloat(zero(BigFloat)), big(NaN), big(Inf), -big(Inf), zero(BigFloat), -zero(BigFloat)]
      shifts = Any[0, 1, -1, 3, 1000, -1000, emax - 1, emax, emax + 1, -emax, -emax - 1, 2^40, -2^40, typemax(Int), typemin(Int),
                   Int8(7), Int128(emax) + 5, -Int128(emax) - 5, BigInt(2)^80, -BigInt(2)^80, typemax(UInt64)]
      @test isempty([(x, n) for x in xs for n in shifts
                     if !same(outcome(() -> PB.ldexp(x, n)), outcome(() -> Base.ldexp(x, exp_clong(n))))])
      o = one(BigFloat)
      # saturation: past the range a shift overflows or underflows, and the sign is kept
      @test PB.ldexp(o, typemax(Int)) == Inf && PB.ldexp(-o, typemax(Int)) == -Inf
      lo, nlo = PB.ldexp(o, typemin(Int)), PB.ldexp(-o, typemin(Int))
      @test iszero(lo) && !signbit(lo) && iszero(nlo) && signbit(nlo)
      # an Int shift that is not a Clong where a C long is 32 bits: 0.375·2^(emax+1) is finite
      @test PB.ldexp(o, emax - 1) == Base.ldexp(o, Clong(emax - 1)) && PB.ldexp(o, emax) == Inf
      @test isfinite(PB.ldexp(big(0.375), Int(emax + 1)))
    end
  end
  # the precision of the argument is not the ambient one: the result is that of Base's exact shift
  x = BigFloat(0.7; precision = 60)
  setprecision(BigFloat, 4) do
    a = PB.ldexp(x, 3)
    @test isequal(a, Base.ldexp(x, Clong(3))) && precision(a) == precision(Base.ldexp(x, Clong(3)))
  end
end

####################################
#  the references: a high-precision scaled sum, shifts typed Clong, no package helper
####################################

# λ1 Σ|xᵢ| as r·2^E for BigFloat entries x: E = exponent(max|xᵢ|), r a 4000-bit value of the scaled sum (the scaled entries
# and λ1 are exact; an entry so far below the largest that scaling underflows it would be dropped, far below any envelope here)
function refscaled(x, λ1)
  E = exponent(maximum(abs, x))
  r = setprecision(BigFloat, 4000) do
    BigFloat(λ1) * sum(a -> abs(Base.ldexp(big(a), Clong(-E))), x)
  end
  r, E
end
# y within a relative envelope 2·2^(1-p)|r| of r·2^E (up to four ulps; at p = 2 it is the interval (0, 2r)), finite and positive
function refwithin(y, r, E, p)
  (y isa BigFloat && isfinite(y) && y > 0) || return false
  setprecision(BigFloat, 4000) do
    abs(Base.ldexp(big(y), Clong(-E)) - r) <= 2 * Base.ldexp(abs(r), Clong(1 - p))
  end
end
# whether r·2^E overflows when rounded once to the working precision p (r < 16, so E is large wherever it can)
function refoverflows(r, E, p)
  exponent(r) + E <= Base.MPFR.get_emax() - 2 && return false
  BigFloat(r, RoundNearest; precision = p) > Base.ldexp(floatmax(BigFloat), Clong(-E))
end
attempt(g) = try g() catch err; err end

####################################
#  the reproducer of #747 and the three review reproducers
####################################

@testset "NormL1Fused(0.25, Inf) at fill(floatmax, 3): the value against the scaled reference (256 bits)" begin
  setprecision(BigFloat, 256) do
    M = floatmax(BigFloat)
    for s in (1, -1)
      x = fill(s * M, 3)
      f = NormL1Fused(big(0.25), big(Inf))
      want = BigFloat(0.75) * M
      r, E = refscaled(x, f.λ1)
      @test isfinite(want) && refwithin(want, r, E, 256)
      @test f(x) == want
      # value_and_prox! returns the value of the reference, not f of its own output
      out = similar(x)
      v = value_and_prox!(out, f, x, one(BigFloat))
      @test v == want && all(==(s * M), out)
      a = copy(x)
      w = value_and_prox!(a, f, a, one(BigFloat))
      @test w == want && all(==(s * M), a)
    end
  end
end

# The class of x = [head(s); tail] for the cones, from the scaled entries in 4000 bits, with heads and tails positive: :in
# where x is inside with a margin (t ≥ 1.5‖tail‖; 2uv ≥ 1.5 ‖tail‖²), :out where it is outside by more than the tolerance of
# the package allows (‖tail‖ ≥ 3t; ‖tail‖² ≥ 8 (u + v)²), else :edge, which is not asserted.
function cone_class(kind, x)
  E = exponent(maximum(abs, x))
  setprecision(BigFloat, 4000) do
    s = [Base.ldexp(big(a), Clong(-E)) for a in x]
    if kind === :soc
      t, n2 = s[1], sum(a -> a * a, s[2:end])
      t * t >= 2.25 * n2 && return :in
      n2 >= 9 * t * t && return :out
    else
      u, v, n2 = s[1], s[2], sum(a -> a * a, s[3:end])
      2 * u * v >= 1.5 * n2 && return :in
      n2 >= 8 * (u + v)^2 && return :out
    end
    :edge
  end
end

@testset "cones at BigFloat scales near 2^35 and floatmax, also with the exponent range of Windows" begin
  for rng in (default_range, windows_range)
    rng() do
      setprecision(BigFloat, 5) do
        a = Base.ldexp(one(BigFloat), Clong(35))
        @test iszero(IndRotatedSOC()(fill(a, 3)))
        M = floatmax(BigFloat)
        z = fill(M, 2)
        @test iszero(IndSOC()(z))
        @test prox(IndSOC(), z) == z
      end
      for p in (rng === default_range ? (5, 256) : (2, 5, 53, 256))
        setprecision(BigFloat, p) do
          M = floatmax(BigFloat)
          pool = BigFloat[one(BigFloat), Base.ldexp(one(BigFloat), Clong(35)), Base.ldexp(big(1.5), Clong(35)), M]
          bad = String[]
          ncase = 0
          for t in pool, a in pool, b in pool
            cls = cone_class(:soc, [t, a, b])
            if cls !== :edge
              ncase += 1
              isinf(IndSOC()([t, a, b])) == (cls === :out) || push!(bad, "soc p=$p $t $a $b $cls")
            end
            cls = cone_class(:rsoc, [t, a, b])
            if cls !== :edge
              ncase += 1
              isinf(IndRotatedSOC()([t, a, b])) == (cls === :out) || push!(bad, "rsoc p=$p $t $a $b $cls")
            end
          end
          @test isempty(bad) && ncase > 40
          isempty(bad) || foreach(println, first(bad, 5))
        end
      end
    end
  end
end

# Scaled, nested Scaled, Scaled(Translate(0)) and Translate(Scaled) of NormL1Fused(λ1, λ2) at fill(floatmax, n): the unscaled
# value overflows for λ1 = 1, so the wrappers restore a value or its parts with the scale α = m·2^a. The range of Windows
# makes the factors' product of the nested chains overflow where the shifts are those of the reviews.
function scaled_grid(p)
  emax = Base.MPFR.get_emax()
  q = emax ÷ 64
  bad = String[]
  ncase = 0
  M = floatmax(BigFloat)
  for λ2 in (0.0, 0.5), λ1 in (1.0, 0.25), n in (2, 3), (m, a) in ((1.0, -16_777_217), (1.0, -q), (1.0, q), (0.75, -q), (1.0, 0))
    x = fill(M, n)
    g = NormL1Fused(BigFloat(λ1), BigFloat(λ2))
    α = Base.ldexp(BigFloat(m), Clong(a))
    r, E = refscaled(x, BigFloat(λ1))
    r2 = setprecision(BigFloat, 4000) do
      r * BigFloat(m)
    end
    E2 = E + a
    over = refoverflows(r2, E2, p)
    h1, h2 = Base.ldexp(BigFloat(m), Clong(a ÷ 2)), Base.ldexp(one(BigFloat), Clong(a - a ÷ 2))
    for (name, h) in (("Scaled", Scaled(g, α)), ("Scaled(Scaled)", Scaled(Scaled(g, h1), h2)),
                      ("Scaled(Translate)", Scaled(Translate(g, zeros(BigFloat, n)), α)),
                      ("Translate(Scaled)", Translate(Scaled(g, α), zeros(BigFloat, n))))
      ncase += 1
      y = attempt(() -> h(x))
      ok = over ? (y isa BigFloat && y == Inf) : refwithin(y, r2, E2, p)
      ok || push!(bad, "p=$p λ1=$λ1 λ2=$λ2 n=$n α=$m·2^$a $name = $(y isa BigFloat ? repr(y) : typeof(y)) (overflows: $over)")
    end
  end
  bad, ncase
end

@testset "Scaled and Translate of NormL1Fused: restored values against the reference, also with the exponent range of Windows" begin
  for rng in (default_range, windows_range), p in (rng === default_range ? (5, 256) : (2, 5, 53, 256))
    rng() do
      setprecision(BigFloat, p) do
        bad, ncase = scaled_grid(p)
        @test isempty(bad) && ncase == 2 * 2 * 2 * 5 * 4
        isempty(bad) || foreach(println, first(bad, 6))
      end
    end
  end
  # the review's parts case: unscaled value 2·floatmax overflows, α = 2^-16777217
  windows_range() do
    setprecision(BigFloat, 5) do
      M = floatmax(BigFloat)
      f = Scaled(NormL1Fused(one(BigFloat), zero(BigFloat)), Base.ldexp(one(BigFloat), Clong(-16_777_217)))
      y = f(fill(M, 2))
      ref = Base.ldexp(M, Clong(-16_777_216))
      @test isfinite(ref) && isfinite(y) && abs(y - ref) <= Base.ldexp(abs(ref), Clong(2 - 5))
    end
  end
end

@testset "a nested Scaled chain whose factors' product overflows: the review's reproducer" begin
  windows_range() do
    setprecision(BigFloat, 5) do
      c = Base.ldexp(one(BigFloat), Clong(-33_554_432))
      a = Base.ldexp(one(BigFloat), Clong(545_259_518))
      b = Base.ldexp(one(BigFloat), Clong(545_259_519))
      @test !isfinite(a * b)             # the product of the factors overflows, so the wrappers stay nested
      g = NormL1Fused(one(BigFloat), BigFloat(Inf))
      y = Scaled(Scaled(g, a), b)(fill(c, 2))
      ref = Base.ldexp(one(BigFloat), Clong(1_056_964_606))
      @test isfinite(ref) && isfinite(y) && abs(y - ref) <= Base.ldexp(abs(ref), Clong(2 - 5))
    end
  end
  # the same chain with the range of the platform
  setprecision(BigFloat, 5) do
    c = Base.ldexp(one(BigFloat), Clong(-33_554_432))
    a, b = Base.ldexp(one(BigFloat), Clong(545_259_518)), Base.ldexp(one(BigFloat), Clong(545_259_519))
    y = Scaled(Scaled(NormL1Fused(one(BigFloat), BigFloat(Inf)), a), b)(fill(c, 2))
    ref = Base.ldexp(one(BigFloat), Clong(1_056_964_606))
    @test isfinite(y) && abs(y - ref) <= Base.ldexp(abs(ref), Clong(2 - 5))
  end
end

####################################
#  exponent sums beyond an Int32 (frexp of a BigFloat is a Clong, an Int32 where a C long is 32 bits)
####################################

# _exceptional_product adds the three binary exponents of its factors: as raw Int32 they wrap past ±2^31 although the factors
# are finite and the product overflows or underflows. With the exponent range of Windows (|exponent| up to 2^30 - 1) the sum of
# three exponents reaches 3·2^30.
@testset "_exceptional_product: the sum of three exponents beyond the Int32 range" begin
  windows_range() do
    for p in (2, 5, 256)
      setprecision(BigFloat, p) do
        emax = Base.MPFR.get_emax()
        sh(k) = Base.ldexp(one(BigFloat), Clong(k))
        for s in (1, -1)
          # the product underflows (the exponents sum to about -2^31 - 1000)
          a, b, c = s * sh(-1000), sh(-(emax - 10)), sh(-(emax - 10))
          y = PB._exceptional_product(a, b, c, a * b * c)
          @test iszero(y) && signbit(y) == (s < 0)
          # the product overflows (the exponents sum to about 3·2^30)
          a, b, c = s * sh(emax - 2), sh(emax - 2), sh(emax - 2)
          y = PB._exceptional_product(a, b, c, a * b * c)
          @test y == s * Inf
          # in between, the product is representable: 2^(emax - 2) · 2^(emax - 2) · 2^(-(emax - 2)) = 2^(emax - 2)
          a, b, c = s * sh(emax - 2), sh(emax - 2), sh(-(emax - 2))
          @test PB._exceptional_product(a, b, c, a * b * c) == s * sh(emax - 2)
        end
      end
    end
  end
end

# The round-1 cone reproducers of the differential: the projection onto the rotated cone {2uv ≥ ‖w‖², u, v ≥ 0} of
# [-floatmax, 2^-35, 1] and of [2^-35, -floatmax, 1], with the exponent range of Windows. In the frame of the cone
# (x1, x2) = ((u + v), (u - v))/√2 the exact projection is λ (1, x2/s, w/s)/ with λ = (x1 + s)/2 and s = √(x2² + w²);
# for u = -floatmax, v = ε this gives v' = ε (1 + O(2^(-2 emax))), u' = ε/2 · 2^(-2 emax) and w' = ε/floatmax, the last two
# below the smallest BigFloat of that range, so they are zero, and the output is (0, ε, 0).
@testset "IndRotatedSOC at [-floatmax, 2^-35, 1] and [2^-35, -floatmax, 1]: finite, feasible, equal to the reference" begin
  windows_range() do
    for p in (2, 5, 53, 256)
      setprecision(BigFloat, p) do
        M = floatmax(BigFloat)
        ε = Base.ldexp(one(BigFloat), Clong(-35))
        o = one(BigFloat)
        for (x, big_head) in (([-M, ε, o], 1), ([ε, -M, o], 2))
          small_head = 3 - big_head
          y = prox(IndRotatedSOC(), x, o)
          out = similar(x)
          v = value_and_prox!(out, IndRotatedSOC(), x, o)
          @test all(isfinite, y) && isequal(y, out) && iszero(v)
          # the reference: the head that was negative is 0 (its exact value underflows), the other head is ε, the tail 0
          @test iszero(y[big_head]) && iszero(y[3])
          @test refwithin(y[small_head], one(BigFloat), -35, p)
          # feasibility, in 4000 bits
          @test setprecision(BigFloat, 4000) do
            big(y[1]) >= 0 && big(y[2]) >= 0 && 2 * big(y[1]) * big(y[2]) >= big(y[3])^2
          end
          # and the package accepts its own output
          @test iszero(IndRotatedSOC()(y))
        end
      end
    end
  end
end

# QuadraticFunction at [floatmax, -tiny, 1] (the differential's `x=5,-7,1`): ½x'Ax + b'x + c = floatmax² + floatmax + 4 + ... is
# about 2^(2 emax), far above floatmax, so the correctly rounded value is +Inf. Its terms are summed as significands and
# exponents; their exponents were added as raw frexp exponents, Int32 where a C long is 32 bits: for 2 x1² at the exponent
# range of Windows 2 + emax + emax = 2^31 wrapped to -2^31, the term was lost and the value came out as about floatmax.
# The gradient A x + b is (2 floatmax + ..., floatmax + 1 + 2 - 2 tiny, 5 - tiny) = (Inf, floatmax, 5), the last entry
# rounded to p bits.
@testset "QuadraticFunction at an overflowing value: Inf, and the gradient of the reference (exponent range of Windows)" begin
  windows_range() do
    for p in (2, 3, 5, 53)
      setprecision(BigFloat, p) do
        M = floatmax(BigFloat)
        tiny = Base.ldexp(one(BigFloat), Clong(Base.MPFR.get_emin() + 8))
        f = QuadraticFunction(BigFloat[2 1 0; 1 2 1; 0 1 2], BigFloat[1, 2, 3], one(BigFloat))
        x = [M, -tiny, one(BigFloat)]
        @test f(x) == Inf
        g = similar(x)
        v = value_and_gradient!(g, f, x)
        r5 = setprecision(BigFloat, 4000) do
          BigFloat(5)
        end
        @test v == Inf && g[1] == Inf && g[2] == M && refwithin(g[3], r5, 0, p)
      end
    end
  end
end

####################################
#  the GPU extension calls the same ldexp
####################################

@testset "the GPU extension's level split of a BigFloat level (_group_levels) uses the exact shift" begin
  ext = Base.get_extension(PB, :ProximalBaseGPUArraysCoreExt)
  @test ext !== nothing
  if ext !== nothing
    # γλ = 2^250 does not fit Float32: it is split into two factors 2^125, which need the shifts 125 and 126 of a BigFloat
    for p in (2, 3, 5, 16, 53)
      setprecision(BigFloat, p) do
        γ, λ = Base.ldexp(one(BigFloat), Clong(200)), Base.ldexp(one(BigFloat), Clong(50))
        step, level = ext._group_levels(Float32, γ, λ)
        @test step === Base.ldexp(1.0f0, 125) && level === Base.ldexp(1.0f0, 125)
      end
    end
  end
end

end
