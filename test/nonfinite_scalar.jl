module NonfiniteScalarTest

# NaN in a scalar penalty or loss (#561, E2): the entry-wise proximal operators and values of
# LogPenalty, CheckLoss, HingeLoss, MCP and SCAD return NaN where the result depends on the NaN
# entry, and compute normally where it does not (zero level or weight: free, E4; infinite weight
# or a constant map at a step of Inf: fixed).

using Test
using ProximalBase
using SparseArrays
using JLArrays
using GPUArraysCore: allowscalar
import Adapt
using ..TestUtils

const PB = ProximalBase

# One case per function: its name, the function, the steps and the out1 output entry at the NaN
# entry of x = [NaN, 1.5] (:nan, or :zero for a fixed or constant coordinate), and whether
# the value term of the out1 entry is free (E4: zero level or zero weight).
function cases(T)
  w0, w∞ = T[0, 1], T[Inf, 1]
  inf = T(Inf)
  steps = T[0, 0.5, 1]
  [("LogPenalty(1,1)", LogPenalty(T(1), T(1)), [steps; 5; 2.9], :nan, false),
   ("LogPenalty(1,1) γ=Inf", LogPenalty(T(1), T(1)), [inf], :zero, false),
   ("LogPenalty(0,1)", LogPenalty(T(0), T(1)), [steps; 5; inf], :nan, true),
   ("LogPenalty(1,1,[0,1])", LogPenalty(T(1), T(1), w0), [steps; 5; inf], :nan, true),
   ("LogPenalty(1,1,[Inf,1])", LogPenalty(T(1), T(1), w∞), [steps; 5; inf], :zero, false),
   ("LogPenalty(0,1,[Inf,1])", LogPenalty(T(0), T(1), w∞), [steps; inf], :zero, false),
   ("CheckLoss(.5)", CheckLoss(T(0.5)), [steps; 5], :nan, false),
   ("CheckLoss(.5) γ=Inf", CheckLoss(T(0.5)), [inf], :zero, false),
   ("CheckLoss(.5,[0,1])", CheckLoss(T(0.5), w0), [steps; 5; inf], :nan, true),
   ("CheckLoss(.5,[Inf,1])", CheckLoss(T(0.5), w∞), [steps; 5; inf], :zero, false),
   ("HingeLoss()", HingeLoss(), [steps; 5; inf], :nan, false),
   ("HingeLoss([1,1])", HingeLoss(T[1, 1]), [steps; 5; inf], :nan, false),
   ("HingeLoss([-1,2])", HingeLoss(T[-1, 2]), [steps; 5; inf], :nan, false),
   ("HingeLoss([1,1],[0,1])", HingeLoss(T[1, 1], w0), [steps; 5; inf], :nan, true),
   ("HingeLoss([-1,2],[Inf,1])", HingeLoss(T[-1, 2], w∞), [steps; 5; inf], :nan, false),
   ("MCP(1,3)", MCP(T(1), T(3)), [steps; 2.9; 3; 5], :nan, false),
   ("MCP(1,3) γ=Inf", MCP(T(1), T(3)), [inf], :zero, false),
   ("MCP(0,3)", MCP(T(0), T(3)), [steps; 2.9; 3; 5; inf], :nan, true),
   ("SCAD(1,3.7)", SCAD(T(1), T(3.7)), [steps; 2.6; 2.7; 5], :nan, false),
   ("SCAD(0,3.7)", SCAD(T(0), T(3.7)), [steps; 2.7; 5; inf], :nan, true)]
end

const NaNx = [NaN, 1.5]

@testset "$T" for T in (Float32, Float64)
  x = T.(NaNx)
  @testset "$name" for (name, f, steps, out1, free) in cases(T)
    for γ in steps
      γ = T(γ)
      # the second entry is finite and independent of the out1
      ref2 = PB.prox_block(f, x[2], 2, γ)
      @test isfinite(ref2)
      check(y) = out1 === :nan ? (@test isnan(y[1]); @test isequal(y[2], ref2)) :
                                  (@test isequal(y[1], zero(T)); @test isequal(y[2], ref2))
      out1 === :nan ? @test(isnan(PB.prox_block(f, x[1], 1, γ))) :
                       @test(isequal(PB.prox_block(f, x[1], 1, γ), zero(T)))
      y = fill(T(7), 2)                           # a sentinel the call must overwrite
      @test prox!(y, f, x, γ) === y
      check(y)
      z = copy(x)                                 # aliasing
      prox!(z, f, z, γ)
      check(z)
      @test eltype(prox(f, x, γ)) == T
      # value_and_prox!: the value at the stored output
      y = fill(T(7), 2)
      v = value_and_prox!(y, f, x, γ)
      check(y)
      @test v isa T
      @test isequal(v, f(y))
      free || out1 === :zero ? @test(isfinite(v)) : @test(isnan(v))
      z = copy(x)
      @test isequal(value_and_prox!(z, f, z, γ), v)
      @test isequal(z, y)
    end
    # the standalone value: free (zero level or weight) or dependent on the NaN entry
    v = f(x)
    if free
      @test isequal(v, PB.value_entry(f, x[2], 2))
      @test isfinite(v)
    else
      @test isnan(v)
    end
    @test isnan(PB.value_entry(f, T(NaN), 1)) == !free
  end
end

@testset "one-entry values and levels" begin
  @test isnan(MCP(1, 3)([NaN])) && isnan(MCP(1f0, 3f0)(Float32[NaN]))
  @test isnan(SCAD(1, 3.7)([NaN])) && isnan(SCAD(1f0, 3.7f0)(Float32[NaN]))
  @test MCP(0, 3)([NaN]) === 0.0 && SCAD(0, 3.7)([NaN]) === 0.0
  @test LogPenalty(0, 1)([NaN, 1.5]) === 0.0
  @test LogPenalty(1, 1, [0, 1])([NaN, 1.0]) == log(2)
  @test CheckLoss(0.5, [0, 1])([NaN, 1.0]) == 0.5
  @test HingeLoss([1.0, 1.0], [0, 1])([NaN, 0.0]) == 1.0
  @test HingeLoss(nothing, [0, 1])([NaN, 0.0]) == 1.0
  # an infinite weight is the indicator of xᵢ = 0 (CheckLoss, LogPenalty) or of yᵢxᵢ ≥ 1
  # (HingeLoss), which depends on xᵢ
  @test isnan(CheckLoss(0.5, [Inf, 1])([NaN, 1.0]))
  @test isnan(LogPenalty(1, 1, [Inf, 1])([NaN, 1.0]))
  @test isnan(HingeLoss([1.0, 1.0], [Inf, 1])([NaN, 0.0]))
  # the value at the stored output is finite where the output is fixed
  y = zeros(2)
  @test value_and_prox!(y, LogPenalty(1, 1, [Inf, 1]), [NaN, 1.0], 1.0) == 0
  @test y == [0.0, 0.0]
end

@testset "value of a zero weight has the type of the other weights" begin
  args = Float32[0.5, -2, 0, -0.0, NaN, Inf, -Inf]
  ws = [0.0, 2.0, Inf]
  # the value, widened to Float64, is bitwise the reference (NaN payloads aside)
  same(a, b) = (a = Float64(a); a === b || (isnan(a) && isnan(b)))
  # φ is formed in the type of the argument; a zero weight leaves the entry free (E4), an
  # infinite weight is the indicator of φ = 0
  weighted(w, φ) = iszero(w) ? 0.0 : (isinf(w) && iszero(φ) ? 0.0 : w * Float64(φ))
  τ = Float32(0.3)
  @testset "CheckLoss" begin
    f = CheckLoss(0.3, ws)
    for (i, w) in enumerate(ws), v in args
      φ = v >= 0 ? τ * v : (τ - 1) * v
      @test @inferred(PB.value_entry(f, v, i)) isa Float64
      @test same(PB.value_entry(f, v, i), weighted(w, φ))
    end
    @test same(PB.value_entry(f, -0.0f0, 2), -0.0)
    @test same(PB.value_entry(f, -0.0f0, 3), 0.0)
    @test same(PB.value_entry(f, NaN32, 1), 0.0)
  end
  @testset "HingeLoss" begin
    f = HingeLoss(nothing, ws)
    for (i, w) in enumerate(ws), v in args
      φ = max(1 - 1 * v, 0)
      @test @inferred(PB.value_entry(f, v, i)) isa Float64
      @test same(PB.value_entry(f, v, i), weighted(w, φ))
    end
    @test same(PB.value_entry(f, Inf32, 2), 0.0)
    @test same(PB.value_entry(f, Inf32, 3), 0.0)
    @test same(PB.value_entry(f, NaN32, 1), 0.0)
  end
  @testset "BigFloat weights on Float64 data" begin
    fc = CheckLoss(0.3, BigFloat[0, 2])
    fh = HingeLoss(nothing, BigFloat[0, 2])
    for f in (fc, fh), i in 1:2, v in (0.5, -2.0, NaN, Inf)
      @test @inferred(PB.value_entry(f, v, i)) isa BigFloat
    end
    @test PB.value_entry(fc, NaN, 1) == 0
    @test isnan(PB.value_entry(fc, NaN, 2))
    @test PB.value_entry(fh, NaN, 1) == 0
    @test isnan(PB.value_entry(fh, NaN, 2))
  end
  @testset "NormL1Nonneg has one type at every entry" begin
    # in the domain, out of it and at NaN; the result type is that of the in-domain value, so
    # unweighted functions and wider levels or weights give Float64 at Float32 entries
    for (f, R) in ((NormL1Nonneg(1f0), Float64), (NormL1Nonneg(1f0, Float32[0, 2, Inf]), Float32),
                   (NormL1Nonneg(), Float64), (NormL1Nonneg(1.0), Float64), (NormL1Nonneg(0), Float64),
                   (NormL1Nonneg(1.0, Float64[0.0, 2.0, Inf]), Float64), (NormL1Nonneg(1.0, Float32[1, 2, 3]), Float64))
      for v in args, i in 1:3
        @test @inferred(PB.value_entry(f, v, i)) isa R
      end
      @test @inferred(PB.value_entry(f, -1f0, 1)) === R(Inf)
      @test isnan(@inferred(PB.value_entry(f, NaN32, 1)))
    end
    # the dense values stay in the data type
    for f in (NormL1Nonneg(1f0), NormL1Nonneg(), NormL1Nonneg(1.0, Float64[1.0, 2.0, 3.0]))
      @test f(Float32[1, -1, 0]) === Inf32
      @test f(Float32[1, 2, 0]) isa Float32
      @test f(Float32[1, NaN, 0]) isa Float32 && isnan(f(Float32[1, NaN, 0]))
    end
  end
  @testset "controls" begin
    for v in args, i in 1:3
      @test @inferred(PB.value_entry(EpsInsensitiveLoss(0.1, ws), v, i)) isa Float64
    end
    for v in args, i in 1:2
      @test @inferred(PB.value_entry(SqrHingeLoss(nothing, [0.0, 2.0]), v, i)) isa Float64
    end
    for v in args
      @test @inferred(PB.value_entry(CheckLoss(0.3), v, 1)) isa Float32
      @test @inferred(PB.value_entry(HingeLoss(), v, 1)) isa Float32
    end
  end
  @testset "dense, sparse and value_and_prox! results are Float32" begin
    x = Float32[NaN, -1, 0]
    for f in (CheckLoss(0.3, ws), HingeLoss(nothing, ws))
      @test f(x) isa Float32
      @test f(Float32[NaN, 0.5, 2]) isa Float32
      si = SparseIterate(Float32, 3)
      si[1] = NaN32; si[2] = -1f0
      @test f(si) isa Float32
      @test value_and_prox!(similar(x), f, x, 1f0) isa Float32
    end
    @test HingeLoss(nothing, [0.0, 2.0])(Float32[NaN, -1]) === 4f0
    @test CheckLoss(0.3, [0.0, 2.0])(Float32[NaN, -1]) === Float32(2.0 * Float64((τ - 1) * -1f0))
  end
end

@testset "zero level is the identity at every step (finite entries)" begin
  x = [2.0, -1.0, 0.5, 0.0, 5.0]
  for f in (MCP(0, 3), SCAD(0, 3.7), LogPenalty(0, 1), LogPenalty(0.0, 0.5)), γ in (0.0, 0.3, 1.0, 3.0, 10.0, Inf)
    @test prox(f, x, γ) == x
    @test f(x) == 0
  end
  for γ in (0.3, 10.0, Inf)
    @test prox(LogPenalty(1, 1, [0, 1, 0, 1, 0]), x, γ)[[1, 3, 5]] == x[[1, 3, 5]]
  end
end

@testset "finite inputs are unchanged" begin
  x = [-3.0, -1.2, -1.0, -0.4, 0.0, 0.7, 1.0, 2.9, 3.0, 4.0, 11.0]
  # CheckLoss and HingeLoss: the definitions
  for τ in (0.3, 0.5), γ in (0.0, 0.5, 1.0, 4.0), w in (1.0, 0.0, 2.5)
    s = γ * w
    ref = [u > s * τ ? u - s * τ : u < -s * (1 - τ) ? u + s * (1 - τ) : 0.0 for u in x]
    @test prox(CheckLoss(τ, fill(w, length(x))), x, γ) == ref
    w == 1.0 && @test prox(CheckLoss(τ), x, γ) == ref
  end
  for γ in (0.0, 0.5, 1.0, 4.0), (y, w) in ((1.0, 1.0), (-1.0, 2.0), (2.0, 0.5), (0.5, 0.0))
    s = γ * w * y^2
    ref = [(let m = y * u; (m >= 1 ? m : m < 1 - s ? m + s : 1.0) / y end) for u in x]
    @test prox(HingeLoss(fill(y, length(x)), fill(w, length(x))), x, γ) == ref
  end
  @test prox(HingeLoss(), [2.0, 0.9, 0.5, -1.0], 0.3) ≈ [2.0, 1.0, 0.8, -0.7]
  @test prox(HingeLoss(), [0.3, 0.5], Inf) == [1.0, 1.0]
  @test prox(HingeLoss([1.0, 1.0], [Inf, 1.0]), [0.3, 0.5], 1.0) == [1.0, 1.0]
  @test HingeLoss([1.0, 1.0], [Inf, 1.0])([0.3, 0.5]) == Inf
  @test HingeLoss([1.0, 1.0], [Inf, 1.0])([1.3, 0.5]) == 0.5
  @test CheckLoss(0.5, [Inf, 1.0])([0.0, 1.0]) == 0.5
  @test prox(CheckLoss(0.5, [Inf, 1.0]), [0.3, 1.0], 1.0) == [0.0, 0.5]
  # MCP and SCAD values at the breakpoints, and the proximal maps on both sides of the switch
  m, sc = MCP(1.0, 3.0), SCAD(1.0, 3.7)
  @test m([0.0, 1.0, 3.0, 4.0]) ≈ 0 + (1 - 1 / 6) + 1.5 + 1.5
  @test sc([0.0, 1.0, 3.7, 5.0]) ≈ 0 + 1 + 2.35 + 2.35
  @test m([-3.0]) == m([3.0]) == 1.5 && sc([-3.7]) == sc([3.7]) == 2.35
  @test prox(m, [0.1, 1.5, 4.0, -4.0], 1.0) ≈ [0.0, 0.75, 4.0, -4.0]
  @test prox(m, [0.1, 5.0, 1.0, -1.0], 5.0) == [0.0, 5.0, 0.0, 0.0]
  @test prox(m, [0.1, 1.5, 5.0], Inf) == zeros(3)
  @test prox(sc, [0.1, 1.5, 6.0], 1.0) ≈ [0.0, 0.5, 6.0]
  @test prox(sc, [0.1, 6.0, 0.3], 5.0) == [0.0, 6.0, 0.0]
  # the larger root of z² - 5z - 5 = 0 at t = 6
  @test prox(LogPenalty(1.0, 1.0), [0.1, 1.5, 6.0], 1.0) ≈ [0.0, 1.0, (5 + sqrt(45)) / 2]
  @test prox(LogPenalty(1.0, 1.0), [1.5, 0.2], Inf) == zeros(2)
  @test LogPenalty(1.0, 1.0, [2.0, 3.0])([1.0, 2.0]) ≈ 2log(2) + 3log(3)
  @test CheckLoss(0.5, [2.0, 3.0])([1.0, -2.0]) == 1.0 + 3.0
end

# the three ways to call prox for one function and entry: the entry oracle, a caller-owned output
# with a sentinel, and the fused call, which also reports the value at the stored output
function prox_paths(f, x, γ)
  T = eltype(x)
  y = fill(T(7), length(x))
  prox!(y, f, x, γ)
  z = copy(x)
  prox!(z, f, z, γ)
  yv = fill(T(7), length(x))
  v = value_and_prox!(yv, f, x, γ)
  w = copy(x)
  vw = value_and_prox!(w, f, w, γ)
  @test isequal(y, z) && isequal(y, yv) && isequal(y, w) && isequal(v, vw)
  @test isequal(y[1], PB.prox_block(f, x[1], 1, γ))
  (y, v)
end

# A finite parameter never becomes an endpoint or a hard weight through the rounding of a
# product: γλw that overflows to Inf is no constant map, so a NaN entry stays NaN (Float32 and
# Float64); only the step γ = Inf and an infinite weight are endpoints.
@testset "overflowing finite factors are no endpoints, $T" for T in (Float32, Float64)
  big = T(floatmax(T) / 4)
  x = T[NaN]
  for (name, f, γ) in (("LogPenalty level", LogPenalty(big, T(1)), T(8)),
                       ("LogPenalty weight", LogPenalty(T(1), T(1), T[big]), T(8)),
                       ("LogPenalty step", LogPenalty(T(1), T(1)), big * 4),
                       ("CheckLoss weight", CheckLoss(T(0.5), T[big]), T(8)),
                       ("CheckLoss step", CheckLoss(T(0.5)), big * 4),
                       ("HingeLoss weight", HingeLoss(nothing, T[big]), T(8)),
                       ("MCP step", MCP(T(1), T(3)), big * 4),
                       ("SCAD step", SCAD(T(1), T(3.7)), big * 4))
    y, v = prox_paths(f, x, γ)
    @test isnan(y[1])
    @test isnan(v)
  end
  # Float64 factors that overflow in Float32 only: the working type does not decide
  @test isnan(prox(CheckLoss(T(0.5)), x, 1e300)[1])
  @test isnan(prox(MCP(T(1), T(3)), x, 1e300)[1])
  @test isnan(prox(LogPenalty(T(1), T(1)), x, 1e300)[1])
  # the genuine endpoints keep their meaning
  for f in (LogPenalty(T(1), T(1)), CheckLoss(T(0.5)), MCP(T(1), T(3)))
    y, v = prox_paths(f, x, T(Inf))
    @test y[1] === zero(T) && v === zero(T)
  end
  for f in (LogPenalty(T(1), T(1), T[Inf]), CheckLoss(T(0.5), T[Inf]))
    y, v = prox_paths(f, x, T(8))
    @test y[1] === zero(T) && v === zero(T)
  end
  # and finite entries are as before: an overflowing coefficient thresholds everything
  @test prox(CheckLoss(T(0.5), T[big]), T[1.5], T(8)) == T[0]
end

# E4 at a zero level: the identity at every finite step, including entries that are tiny (an
# underflowing penalty), and the sign of zero (base: 0 for tiny entries at steps past the switch)
@testset "zero level: tiny entries and signed zero, $T" for T in (Float32, Float64)
  tiny = nextfloat(zero(T))
  x = T[tiny, -tiny, 0, -0, 1e-30, -1e-30, 1e-44, 3, -3]
  x[3] = 0; x[4] = -0.0
  for f in (MCP(T(0), T(3)), SCAD(T(0), T(3.7))), γ in T[5, 100, 1e10, Inf]
    y, v = prox_paths(f, x, γ)
    @test isequal(y, x)                         # isequal tells -0.0 from 0.0
    @test signbit(y[4]) && !signbit(y[3])
    @test v === zero(T)
    @test all(i -> isequal(PB.prox_block(f, x[i], i, γ), x[i]), eachindex(x))
  end
  # a level of zero but a NaN entry: the identity, NaN copied
  for f in (MCP(T(0), T(3)), SCAD(T(0), T(3.7)))
    @test isnan(prox_paths(f, T[NaN], T(5))[1][1])
    @test prox_paths(f, T[NaN], T(5))[2] === zero(T)
  end
  # a positive level at a tiny entry is still thresholded to zero at these steps
  for f in (MCP(T(1), T(3)), SCAD(T(1), T(3.7)))
    @test iszero(prox(f, T[tiny, -tiny], T(5)))
  end
end
@testset "zero level: a Float64 entry whose square underflows (control)" begin
  x = [1e-300, -1e-300, 1e-160, 2.0]
  for f in (MCP(0.0, 3.0), SCAD(0.0, 3.7)), γ in (5.0, 100.0)
    @test isequal(prox_paths(f, x, γ)[1], x)
  end
end

# LogPenalty(λ, ε = Inf) is the zero function on finite arguments (log1p(|x|/Inf) = 0): the value
# is 0, NaN included, and the proximal map is the identity at every step, including γ = Inf; an
# infinite coordinate weight keeps its hard meaning, a zero weight stays free.
@testset "LogPenalty with an infinite ε, $T" for T in (Float32, Float64)
  f = LogPenalty(T(1), T(Inf))
  @test f(T[0, 1, -100]) === zero(T)
  @test f(T[NaN]) === zero(T)
  @test f(T[NaN, 1.5]) === zero(T)
  for γ in T[0, 0.5, 1, 100, Inf]
    for x in (T[NaN, 1.5, -2, 0, 1e30], T[0.3, -3])
      y, v = prox_paths(f, x, γ)
      @test isequal(y, x)
      @test v === zero(T)
    end
  end
  # free and hard weights
  g = LogPenalty(T(1), T(Inf), T[0, Inf, 2])
  x = T[NaN, NaN, NaN]
  @test g(T[NaN, 0, NaN]) === zero(T)
  @test isnan(g(T[0, NaN, 0]))                  # the indicator of x₂ = 0 depends on x₂
  @test g(T[1, 0, 3]) === zero(T) && g(T[1, 1, 3]) === T(Inf)
  for γ in T[0.5, 1, Inf]
    y, v = prox_paths(g, x, γ)
    @test isnan(y[1]) && y[2] === zero(T) && isnan(y[3])
    @test v === zero(T)                         # the value at the stored output: zero function, y₂ = 0
    @test isequal(prox(g, T[1, 1, 1], γ), T[1, 0, 1])
  end
  # the level stays positive: zero level and infinite ε agree
  @test isequal(prox(LogPenalty(T(0), T(1)), T[NaN, 2], T(Inf)), T[NaN, 2])
end


# A valid quantile (inside (0, 1) in its own type) may round to 1 in the working type, which makes
# the threshold s(1 - τ) of a hard weight or an infinite step 0⋅Inf = NaN: the finite entries are
# still fixed at 0 (E4), as are NaN entries of a hard weight or an infinite step, while a finite
# product that overflows is no endpoint and keeps a NaN entry NaN. Mixed precision: a Float64
# quantile with Float32 data, and a BigFloat quantile with Float64 data.
@testset "CheckLoss quantile that rounds to one in the working type" begin
  for (T, tau) in ((Float32, prevfloat(1.0)), (Float64, prevfloat(BigFloat(1))), (Float32, nextfloat(0.0)))
    @test 0 < tau < 1
    @test T(tau) == one(T) || T(tau) == zero(T)
    x = T[1.5, -1.5, 0.0, 1e30, -1e30]
    for (label, w, γ) in (("hard weight", Inf, 1.0), ("infinite step", 1.0, Inf),
                          ("zero weight", 0.0, 1.0), ("finite coefficient overflow", big(floatmax(T)) * 4, 1.0))
      free = label == "zero weight"
      # finite entries: fixed at 0 (hard weight, infinite step, overflow) or free (zero weight)
      y, v = prox_paths(CheckLoss(tau, fill(w, length(x))), x, γ)
      if label == "finite coefficient overflow"
        # The thresholds γτw and γ(1-τ)w are formed in the promotion of the factors and rounded once to T
        # (coefficient contract T3), not from a quantile first rounded to T: where such a threshold is finite
        # in T it is an endpoint (e.g. 1 - τ = 2^-53 keeps γ(1-τ)w ≈ 1e23 finite), and where it overflows
        # or underflows it is the rounded value, Inf or 0.
        hi, lo = convert(T, big(γ) * big(tau) * big(w)), convert(T, big(γ) * (1 - big(tau)) * big(w))
        @test isequal(y, T[u > hi ? u - hi : (u < -lo ? u + lo : zero(T)) for u in x])
      else
        @test isequal(y, free ? x : zeros(T, length(x)))
        @test iszero(v)
      end
      # a NaN entry: 0 for a hard weight or an infinite step, NaN (copied) for a zero weight, whose
      # value stays 0, and NaN for the finite overflow, whose stored output makes the value NaN
      y, v = prox_paths(CheckLoss(tau, [w]), T[NaN], γ)
      if label == "hard weight" || label == "infinite step"
        @test y[1] === zero(T) && v === zero(T)
      elseif free
        @test isnan(y[1]) && v === zero(T)
      else
        @test isnan(y[1]) && isnan(v)
      end
    end
  end
end

# LogPenalty(Inf, ε) has the domain {0}: its proximal map is identically 0 on finite arguments at
# every positive finite step, so 0 at a NaN entry too (entry, stored, aliased and fused); the
# standalone value stays NaN there (it is the indicator of x = 0, which depends on x). A finite
# level whose coefficient overflows is another case and keeps NaN.
@testset "LogPenalty with an infinite level, $T" for T in (Float32, Float64)
  for γ in T[0.5, 5, 100]
    f = LogPenalty(T(Inf), T(1))
    @test f(T[0]) === zero(T) && f(T[1]) === T(Inf) && isnan(f(T[NaN]))
    @test iszero(prox(f, T[-10, -1, 0, 1, 10], γ))
    y, v = prox_paths(f, T[NaN, 2], γ)
    @test y[1] === zero(T) && y[2] === zero(T) && v === zero(T)
    # a zero weight stays free, an infinite weight stays fixed
    g = LogPenalty(T(Inf), T(1), T[0, 1])
    y, v = prox_paths(g, T[NaN, NaN], γ)
    @test isnan(y[1]) && y[2] === zero(T)
    @test isnan(prox(LogPenalty(floatmax(Float64), 1.0), T[NaN], γ)[1])    # a finite level is no endpoint
  end
  @test isnan(prox(LogPenalty(T(floatmax(T) / 4), T(1)), T[NaN], T(8))[1])
end

# MCP beyond the switch: a step of Inf or an infinite level is a constant zero; a zero level the
# identity; a finite level whose aλ overflows in the working type is no endpoint; infinite entries
# keep the result of the base.
@testset "MCP beyond the switch: constant maps, overflow and infinite entries, $T" for T in (Float32, Float64)
  f = MCP(T(1), T(3))
  for γ in T[3, 5]
    y, v = prox_paths(f, T[NaN, 0.1, 5, -5], γ)
    @test isnan(y[1]) && y[2] === zero(T) && y[3] == 5 && y[4] == -5 && isnan(v)
  end
  for γ in T[3, 5, 100]
    @test iszero(prox(f, T[Inf, -Inf], γ))          # an infinite entry keeps the result of the base
    @test isnan(prox_paths(f, T[NaN], γ)[1][1])
  end
  y, v = prox_paths(f, T[NaN, 2], T(Inf))
  @test iszero(y) && v === zero(T)
  y, v = prox_paths(MCP(T(Inf), T(3)), T[NaN, 2], T(5))
  @test iszero(y)                                   # (the value at 0 of an infinite level is Inf⋅0 = NaN, as at base)
  big = T(floatmax(T) / 2)                          # a·λ overflows in T, λ and a are finite
  y, v = prox_paths(MCP(big, T(3)), T[NaN, 1], T(5))
  @test isnan(y[1]) && y[2] === zero(T) && isnan(v)
end

# The objective of MCP beyond the switch overflows (λ² is Inf) while aλ is finite: the strict
# comparison of the base selects 0, which is also the global minimizer at these small entries; a
# NaN made by the arithmetic of the objective is not a NaN entry. The same for an infinite entry at
# a level near floatmax, and for the GroupMCP caller of the scalar kernel.
@testset "MCP beyond the switch: overflowing objective, finite entries, $T" for (T, level) in ((Float32, 1f20), (Float64, 1e200))
  @test isfinite(T(3) * T(level)) && isinf(T(level)^2)
  x = T[1, -1, 0.1, -0.1]
  for γ in T[3, 5, 100]
    f = MCP(T(level), T(3))
    y, v = prox_paths(f, x, γ)
    @test iszero(y) && v === zero(T)
    @test PB.prox_block(f, x[1], 1, γ) === zero(T)
    y, v = prox_paths(f, T[NaN, 1], γ)
    @test isnan(y[1]) && y[2] === zero(T) && isnan(v)       # a NaN entry still gives NaN
  end
  g = GroupMCP(T(level), [1:2])
  y = fill(T(7), 2)
  v = value_and_prox!(y, g, T[1, 0], T(5))
  @test iszero(y) && v === zero(T)
  z = T[1, 0]
  prox!(z, g, z, T(5))
  @test iszero(z)
  # an infinite entry at a level near floatmax keeps the result of the base
  y, v = prox_paths(MCP(floatmax(T), one(T)), T[Inf, -Inf], T(5))
  @test iszero(y)
end


si_free(f) = (si = SparseIterate(2); si[1] = NaN; si[2] = 1.5; si)

@testset "iterates" begin
  for (name, f, steps, out1, free) in cases(Float64)
    for γ in steps
      si = SparseIterate(2); si[1] = NaN; si[2] = 1.5
      y = SparseIterate(2)
      prox!(y, f, si, γ)
      out1 === :nan ? @test(isnan(y[1])) : @test(y[1] == 0)
      @test isequal(y[2], PB.prox_block(f, 1.5, 2, γ))
      z = copy(si)
      prox!(z, f, z, γ)
      @test isequal(Vector(z), Vector(y))
      v = value_and_prox!(SparseIterate(2), f, si, γ)
      free || out1 === :zero ? @test(isfinite(v)) : @test(isnan(v))
    end
    free ? @test(isfinite(f(si_free(f)))) : @test(isnan(f(si_free(f))))
  end
  # AtomIterate, stored entries only
  for f in (CheckLoss(0.5), LogPenalty(1, 1), MCP(1, 3), SCAD(1, 3.7), HingeLoss(), MCP(0, 3), LogPenalty(0, 1))
    a = AtomIterate(2, [1:1, 2:2]); a[1] = NaN; a[2] = 1.5
    y = AtomIterate(2, [1:1, 2:2])
    prox!(y, f, a, 1.0)
    @test isnan(y[1]) && isequal(y[2], PB.prox_block(f, 1.5, 2, 1.0))
    @test isnan(f(a)) == !(f isa Union{MCP, LogPenalty} && f.λ == 0)
  end
  # SymmetricSparseIterate: a NaN on the diagonal and an off-diagonal coordinate
  for f in (CheckLoss(0.5), LogPenalty(1, 1), MCP(1, 3), SCAD(1, 3.7), HingeLoss(), MCP(0, 3), LogPenalty(0, 1))
    S = SymmetricSparseIterate(2); S[1, 1] = NaN; S[2, 1] = 1.5
    y = SymmetricSparseIterate(2)
    prox!(y, f, S, 1.0)
    r = PB.prox_block(f, 1.5, 2, 1.0)
    @test isnan(y[1, 1]) && isequal(y[2, 1], r) && isequal(y[1, 2], r)
    v = value_and_prox!(SymmetricSparseIterate(2), f, S, 1.0)
    @test isnan(v) == !(f isa Union{MCP, LogPenalty} && f.λ == 0)
  end
end

@testset "JLArrays" begin
  allowscalar(false)
  for T in (Float32, Float64), (f, expected) in ((LogPenalty(T(1), T(1)), NaN), (CheckLoss(T(0.5)), NaN), (HingeLoss(), NaN),
                                                 (MCP(T(1), T(3)), NaN), (SCAD(T(1), T(3.7)), NaN),
                                                 (MCP(T(0), T(3)), 0.0), (LogPenalty(T(0), T(1)), 0.0))
    x = T[NaN, 1.5]
    xg = JLArray(x)
    for γ in (T(0.5), T(5))
      yg = similar(xg)
      prox!(yg, f, xg, γ)
      @test isequal(Array(yg), prox(f, x, γ))
      @test isnan(Array(yg)[1])
      zg = copy(xg)
      prox!(zg, f, zg, γ)
      @test isequal(Array(zg), Array(yg))
    end
    @test isequal(f(xg), f(x))
    @test isnan(f(xg)) == isnan(expected)
  end
  # weighted: the device parameters
  for f in (CheckLoss(0.5, [0.0, 1.0]), CheckLoss(0.5, [Inf, 1.0]), CheckLoss(0.5, [1.0, 1.0]))
    fg = Adapt.adapt(JLArray, f)
    xg = JLArray([NaN, 1.5])
    yg = similar(xg)
    prox!(yg, fg, xg, 1.0)
    @test isequal(Array(yg), prox(f, [NaN, 1.5], 1.0))
    @test isequal(fg(xg), f([NaN, 1.5]))
  end
end

end
