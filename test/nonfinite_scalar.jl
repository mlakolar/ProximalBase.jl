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
      ref2 = PB.prox_entry(f, x[2], 2, γ)
      @test isfinite(ref2)
      check(y) = out1 === :nan ? (@test isnan(y[1]); @test isequal(y[2], ref2)) :
                                  (@test isequal(y[1], zero(T)); @test isequal(y[2], ref2))
      out1 === :nan ? @test(isnan(PB.prox_entry(f, x[1], 1, γ))) :
                       @test(isequal(PB.prox_entry(f, x[1], 1, γ), zero(T)))
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

si_free(f) = (si = SparseIterate(2); si[1] = NaN; si[2] = 1.5; si)

@testset "iterates" begin
  for (name, f, steps, out1, free) in cases(Float64)
    for γ in steps
      si = SparseIterate(2); si[1] = NaN; si[2] = 1.5
      y = SparseIterate(2)
      prox!(y, f, si, γ)
      out1 === :nan ? @test(isnan(y[1])) : @test(y[1] == 0)
      @test isequal(y[2], PB.prox_entry(f, 1.5, 2, γ))
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
    @test isnan(y[1]) && isequal(y[2], PB.prox_entry(f, 1.5, 2, 1.0))
    @test isnan(f(a)) == !(f isa Union{MCP, LogPenalty} && f.λ == 0)
  end
  # SymmetricSparseIterate: a NaN on the diagonal and an off-diagonal coordinate
  for f in (CheckLoss(0.5), LogPenalty(1, 1), MCP(1, 3), SCAD(1, 3.7), HingeLoss(), MCP(0, 3), LogPenalty(0, 1))
    S = SymmetricSparseIterate(2); S[1, 1] = NaN; S[2, 1] = 1.5
    y = SymmetricSparseIterate(2)
    prox!(y, f, S, 1.0)
    r = PB.prox_entry(f, 1.5, 2, 1.0)
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
