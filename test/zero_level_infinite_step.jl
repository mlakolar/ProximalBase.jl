module ZeroLevelInfiniteStepTest

using ..TestUtils: take_block_step!

using Test, ProximalBase, LinearAlgebra, SparseArrays
using JLArrays: JLArray
using GPUArraysCore: allowscalar
using Adapt: adapt

const PB = ProximalBase

# A zero level at the step γ = Inf for CappedL1, SqrNormL2 and Regularize (P1, E4). The product γ⋅0 is
# Inf⋅0 where the code forms it from the converted parameters; the level, tested as it is given, decides
# instead, and the step then uses what every finite step already produces: the coefficient 0 of
# CappedL1 (Inf at an infinite weight, which fixes its entry), the ridge weights (1, 0), and for
# Regularize the step γ handed to its function. So at a zero level γ = Inf equals the call at γ = 1,
# and Regularize(f, 0) equals f, bitwise: the signs of zeros, NaN and ±Inf included (an infinite weight
# gives NaN at a NaN entry, as at every finite step). Finite steps and positive levels are unchanged;
# the positive-level controls are pinned at the values before the change. NaN is compared with isequal.
# The infinite step is the only step that changes: a level or a step of either zero sign, a NaN step and a
# negative infinite step keep the arithmetic of the base. The frozen section pins literal values taken
# at the base, for both zero signs of the level and of the step, so that a change shared by the finite
# path and the step 1 cannot hide in a reference computed by the code under test.

same(a, b) = isequal(a, b)

# the class of a value: NaN, an infinity of either sign, or finite
value_class(v) = isnan(v) ? :nan : isinf(v) ? (v > 0 ? :pinf : :ninf) : :finite

# the first error of a call, for the comparison of error types and messages
caught(f) = try; f(); nothing; catch e; e; end

# the entries: both sides of the cap a = 1 and of its half, signed zeros, NaN and ±Inf
entries(::Type{T}) where {T} = T[0.5, -0.5, 1, -1, 2, -2, 0, -0.0, NaN, Inf, -Inf]
# the weights of an entry class, by position: zero, one, infinite and others
weightsof(::Type{T}, n) where {T} = Dict(
  :none => nothing,
  :ones => fill(T(1), n),
  :normal => T[0.5, 2, 1, 4][mod1.(1:n, 4)],
  :zero => fill(T(0), n),
  :infinite => fill(T(Inf), n),
  :mixed => T[0, 1, Inf, 0.5][mod1.(1:n, 4)],
)
const WEIGHTS = (:none, :ones, :normal, :zero, :infinite, :mixed)

# every route on the host of f at the step γ against f at the step ref: the proximal point, prox!
# (also with y === x), prox_block and the fused value, which is the value at the stored point
function check_routes(f, x::AbstractVector{T}, γ, ref) where {T}
  want = prox(f, x, ref)
  @test same(prox(f, x, γ), want)
  y = fill(T(7), length(x))
  @test prox!(y, f, x, γ) === y && same(y, want)
  z = copy(x)
  @test prox!(z, f, z, γ) === z && same(z, want)
  entry(s, i) = s isa AbstractArray ? s[i] : s
  @test same([prox_block(f, x[i], i, entry(γ, i)) for i in eachindex(x)],
             [prox_block(f, x[i], i, entry(ref, i)) for i in eachindex(x)])
  y = fill(T(7), length(x))
  v = value_and_prox!(y, f, x, γ)
  yr = fill(T(7), length(x))
  vr = value_and_prox!(yr, f, x, ref)
  @test same(y, yr) && same(v, vr)
  f isa PB.BoxConstrained || @test value_class(v) == value_class(f(y))   # the fused value is the value at the stored point
  z = copy(x)
  v = value_and_prox!(z, f, z, γ)
  @test same(z, yr) && same(v, vr)
end

####################################
#  CappedL1
####################################

@testset "CappedL1 with a zero level, $T" for T in (Float32, Float64)
  x = entries(T)
  n = length(x)
  Inf_ = T(Inf)
  @testset "weights $wc" for wc in WEIGHTS
    w = weightsof(T, n)[wc]
    for (λ, a) in ((T(0), T(1)), (0, 1), (0.0, 1.0), (T(-0.0), T(1)), (-0.0, 1.0)), wrap in (identity, f -> Scaled(f, 0))
      check_routes(wrap(CappedL1(λ, a, w)), x, Inf_, one(T))
    end
    # a factor 0 folds a positive level into a zero one
    check_routes(Scaled(CappedL1(T(1), T(1), w), 0), x, Inf_, one(T))
  end
  # every weight class meets every argument
  check_routes(CappedL1(T(0), T(1), repeat(T[0, 1, Inf, 0.5], inner = n)), repeat(x, outer = 4), Inf_, one(T))
  @testset "the named cases" begin
    @test same(prox(CappedL1(T(0), T(1), T[0, 1, Inf]), T[0.5, -0.25, 2], Inf_), T[0.5, -0.25, 0])
    @test same(prox(CappedL1(T(0), T(1), T[0, 1, Inf]), T[0.5, -0.25, 2], one(T)), T[0.5, -0.25, 0])
    w = fill(T(Inf), 4)
    @test same(prox(CappedL1(T(0), T(1), w), T[NaN, Inf, -Inf, -2], Inf_), T[NaN, NaN, NaN, -0.0])
    @test same(prox(CappedL1(T(0), T(1), w), T[NaN, Inf, -Inf, -2], one(T)), T[NaN, NaN, NaN, -0.0])
    # no weight: the zero function, a copy (-0 as at every step)
    @test same(prox(CappedL1(T(0), T(1)), T[2, -1, 0.5, NaN], Inf_), T[2, -1, 0.5, NaN])
    @test same(prox(CappedL1(T(0), T(1), T[1, 1, 1]), T[2, -1, 0.5], T[Inf, 1, Inf]), T[2, -1, 0.5])
    # array steps mixing Inf and finite entries
    f = CappedL1(T(0), T(1), T[0, 1, Inf, 0.5])
    y = T[0.5, -2, 2, 0.25]
    @test same(prox(f, y, T[Inf, Inf, Inf, Inf]), prox(f, y, one(T)))
    @test same(prox(f, y, T[Inf, 1, Inf, 1]), prox(f, y, one(T)))
    # an Int argument gives a float, at the step Inf as at the step 1
    @test prox_block(CappedL1(0, 1), 2, 1, Inf) === 2.0
    @test prox_block(CappedL1(0, 1, [Inf]), 2, 1, Inf) === 0.0
    @test prox_block(CappedL1(0, 1), 2, 1, 1) === prox_block(CappedL1(0, 1), 2, 1, Inf)
  end
end

####################################
#  SqrNormL2
####################################

@testset "SqrNormL2 with a zero level, $T" for T in (Float32, Float64)
  x = entries(T)
  n = length(x)
  Inf_ = T(Inf)
  centers = T[0.25, -3, 2, 0, 1, -1, 5, 0.5, 0, 2, -2]
  for λ in (T(0), 0, 0.0, T(-0.0), -0.0)
    for f in (SqrNormL2(λ), SqrNormL2(λ, centers), Constrained(SqrNormL2(λ), IndBox(T(0), T(1))),
              Constrained(SqrNormL2(λ, centers), IndBox(T(0), T(1))),
              Scaled(SqrNormL2(1), 0))
      check_routes(f, x, Inf_, one(T))
    end
    # the value is 0 at a finite point
    y = fill(T(7), 3)
    @test iszero(value_and_prox!(y, SqrNormL2(λ), T[2, -1, 0.5], Inf_)) && y == T[2, -1, 0.5]
  end
  @test same(prox(SqrNormL2(T(0)), T[2, -1, 0.5], Inf_), T[2, -1, 0.5])
  @test same(prox(SqrNormL2(T(0), T[1, 1, 1]), T[2, -1, 0.5], Inf_), T[2, -1, 0.5])
  @test same(prox(Constrained(SqrNormL2(T(0)), IndBox(T(0), T(1))), T[2, -1, 0.5], Inf_), T[1, 0, 0.5])
  @test prox_block(SqrNormL2(0), 2, 1, Inf) === 2.0
  @test prox_block(SqrNormL2(0, [1.0]), 2, 1, Inf) === 2.0
  # the weights of the ridge, which the slope of the box route and the entries share
  @test PB._ridge_ordinary(T, Inf_, T(0)) === PB._ridge_ordinary(T, one(T), T(0))
  @test PB._ridge_ordinary(T, Inf, 0) === PB._ridge_ordinary(T, 1, 0)
  @test PB._ridge_ordinary(T, Inf_, T(-0.0)) === PB._ridge_ordinary(T, one(T), T(-0.0))
  @test signbit(PB._ridge_ordinary(T, Inf_, T(-0.0)).b) && !signbit(PB._ridge_ordinary(T, Inf_, T(0)).b)
  # every other step keeps the arithmetic of the product: a NaN or a negative infinite step is no endpoint
  @test isnan(PB._ridge_ordinary(T, T(NaN), T(0)).a) && isnan(PB._ridge_ordinary(T, T(NaN), T(-0.0)).a)
  @test isnan(PB._ridge_ordinary(T, T(-Inf), T(0)).a)
end

####################################
#  Regularize
####################################

# Regularize(f, 0[, c]) at the step γ against f at γ: the proximal point, prox!, prox!(y, r, y), the
# fused value, and prox_block for a separable f. A center c enters as the point (v + 0⋅c)/(1 + 0), which
# is v for a finite entry other than -0 (the blend gives +0 there, at every step), so the arguments of a
# centered case are finite and have no -0. The fused value is checked where the stored point is finite:
# r(y) adds the ridge 0⋅y², which is NaN at an infinite entry at every step.
function check_regularize(f, x::AbstractVector{T}, γ; c = nothing, ρs = (0, T(0), 0.0, T(-0.0), -0.0)) where {T}
  want = prox(f, x, γ)
  yf = fill(T(7), length(x))
  vf = value_and_prox!(yf, f, x, γ)
  entry(s, i) = s isa AbstractArray ? s[i] : s
  for ρ in ρs
    r = Regularize(f, ρ, c)
    @test same(prox(r, x, γ), want)
    y = fill(T(7), length(x))
    @test prox!(y, r, x, γ) === y && same(y, want)
    z = copy(x)
    @test prox!(z, r, z, γ) === z && same(z, want)
    y = fill(T(7), length(x))
    v = value_and_prox!(y, r, x, γ)
    @test same(y, yf)
    all(isfinite, y) && @test same(v, vf)
    @test value_class(v) == value_class(r(y))            # the fused value is the value at the stored point
    z = copy(x)
    v = value_and_prox!(z, r, z, γ)
    @test same(z, yf)
    all(isfinite, z) && @test same(v, vf)
    if PB.isseparable(f)
      @test same([prox_block(r, x[i], i, entry(γ, i)) for i in eachindex(x)],
                 [prox_block(f, x[i], i, entry(γ, i)) for i in eachindex(x)])
    end
  end
end

@testset "Regularize with a zero level, $T" for T in (Float32, Float64)
  Inf_ = T(Inf)
  x = T[2, -3, 0.5, 0, -0.0, 1, -1, 0.25]
  xn = T[2, NaN, Inf, -Inf, -0.0, 0, 1, NaN]
  xc = T[2, -3, 0.5, 0, 0, 1, -1, 0.25]                 # for a center: no -0
  c = T[1, -2, 0.5, 3, 0, -1, 1, 2]
  box = IndBox(T(0), T(1))
  fs = (ZeroFunction(), NormL1(T(1)), NormL1(T(0.5), T[0, 1, Inf, 2, 1, 1, 0.5, 4]), NormL2(T(1)), box,
        IndNonnegative(), Constrained(NormL1(T(1)), IndBallL2(T(1))), CappedL1(T(0), T(1)),
        CappedL1(T(1), T(1), T[0, 1, Inf, 2, 1, 1, 0.5, 4]), SqrNormL2(T(0)), SqrNormL2(T(1)))
  # Float32 keeps five of the eleven functions; the other six are in zero_level_infinite_step_grids.jl
  @testset "$(nameof(typeof(f)))" for f in (T === Float32 ? Any[fs[i] for i in (1, 2, 4, 5, 11)] : fs)
    # a NaN or infinite entry is the argument of f itself, the weights (1, 0) leave it unchanged
    check_regularize(f, x, Inf_)
    check_regularize(f, xn, Inf_)
    check_regularize(f, xc, Inf_; c = c)
    check_regularize(f, x, one(T))               # finite steps, as before
    check_regularize(f, xc, T(0.5); c = c)
  end
  @testset "bitwise as f" begin
    @test same(prox(Regularize(NormL1(T(1)), 0), T[3, -2, NaN, Inf, -0.0], Inf_), prox(NormL1(T(1)), T[3, -2, NaN, Inf, -0.0], Inf_))
    @test same(prox(Regularize(NormL1(T(1)), 0), T[3, -2], Inf_), T[0, 0])
    @test same(prox(Regularize(ZeroFunction(), 0), T[3, -2, NaN, Inf, -0.0], Inf_), prox(ZeroFunction(), T[3, -2, NaN, Inf, -0.0], Inf_))
    @test same(prox(Regularize(ZeroFunction(), 0), T[3, -2], Inf_), T[3, -2])
    @test same(prox(Regularize(box, 0), T[3, -2, 0.5], Inf_), T[1, 0, 0.5])
    @test same(prox(Regularize(Constrained(NormL1(T(1)), IndBallL2(T(1))), 0), T[3, -2], Inf_), T[0, 0])
  end
  @testset "a function that rejects an infinite step" begin
    f = Constrained(NormL1(T(1)), IndSphereL2(T(1)))
    x2 = T[3, -2]
    ef = caught(() -> prox(f, x2, Inf_))
    @test ef isa ArgumentError
    r = Regularize(f, 0)
    for run in (y -> prox!(y, r, x2, Inf_), y -> prox!(y, r, y, Inf_), y -> value_and_prox!(y, r, x2, Inf_))
      y = T[7, 8]
      er = caught(() -> run(y))
      @test er isa ArgumentError && er.msg == ef.msg
      @test y == T[7, 8]
    end
    er = caught(() -> prox(r, x2, Inf_))
    @test er isa ArgumentError && er.msg == ef.msg
    @test same(prox(r, x2, one(T)), prox(f, x2, one(T)))
  end
  @testset "array steps" begin
    # Float32 keeps the first, second and fourth function; the others are in zero_level_infinite_step_grids.jl
    for (i, f) in enumerate((ZeroFunction(), NormL1(T(1)), NormL1(T(0.5), T[0, 1, Inf, 2, 1, 1, 0.5, 4]), box, IndNonnegative(),
                             CappedL1(T(1), T(1), T[0, 1, Inf, 2, 1, 1, 0.5, 4])))
      T === Float32 && !(i in (1, 2, 4)) && continue
      for γ in (T[Inf, 1, 0.5, Inf, 1, Inf, 2, 0.25], fill(Inf_, 8), T[Inf, 1, 0.5, 0, 1, 0, 2, 0.25])
        check_regularize(f, x, γ)
        check_regularize(f, xc, γ; c = c)
      end
    end
    # the two-entry example
    @test same(prox(Regularize(NormL1(T(1)), 0), T[3, -2], T[Inf, 1]), T[0, -1])
    @test same(prox(Regularize(NormL1(T(1)), 0, T[1, 1]), T[3, -2], T[Inf, 1]), T[0, -1])
    @test same(prox(Regularize(NormL1(T(1)), 0), T[3, -2, NaN], T[Inf, 1, Inf]), prox(NormL1(T(1)), T[3, -2, NaN], T[Inf, 1, Inf]))
  end
  @testset "per-group steps and the coordinate step" begin
    groups = [1:2, 3:5, 6:8]
    # Float32 keeps the first function; the others are in zero_level_infinite_step_grids.jl
    for (i, f) in enumerate((GroupNormL2(T(0.5), groups), GroupNormL2(T(0.5), groups, T[0, 1, Inf]),
                             GroupNormLinf(T(0.5), groups), SparseGroupNormL2(T(0.2), T(0.4), groups)))
      T === Float32 && i > 1 && continue
      for γ in (T[Inf, 1, 0.5], T[Inf, Inf, Inf], T[1, Inf, 2], T[0.5, 1, 2])
        check_regularize(f, x, γ)
        check_regularize(f, xc, γ; c = c)
      end
      for k in 1:3, centered in (false, true)
        rc = centered ? c : nothing
        for start in (centered ? xc : x, T[0, 0, 0, 0, 0, 1, -1, 0.25])
          for γ in (Inf_, one(T))
            xf = AtomIterate(copy(start), groups)
            xr = AtomIterate(copy(start), groups)
            take_block_step!(xf, f, k, γ)
            take_block_step!(xr, Regularize(f, 0, rc), k, γ)
            @test same(Vector(xr), Vector(xf))
            @test activecoordinates(xr) == activecoordinates(xf)
            @test same(xr.storage, xf.storage)
          end
        end
      end
    end
  end
  @testset "wrappers" begin
    xs = (T[3, -2, 0.5], T[1, -2])
    sep = SeparableSum(Regularize(NormL1(T(1)), 0), Regularize(ZeroFunction(), T(0)))
    ref = SeparableSum(NormL1(T(1)), ZeroFunction())
    @test same(prox(sep, xs, Inf_), prox(ref, xs, Inf_))
    @test same(prox(sep, xs, Inf_), (T[0, 0, 0], T[1, -2]))
    for k in (2, 0.5), c in (nothing, T[1, 1, 1])
      xk = T[3, -2, 0.5]
      @test same(prox(Scaled(Regularize(NormL1(T(1)), 0, c), k), xk, Inf_), prox(Scaled(NormL1(T(1)), k), xk, Inf_))
      @test same(prox(Scaled(Regularize(box, T(0), c), k), xk, Inf_), prox(Scaled(box, k), xk, Inf_))
    end
    @test same(prox(Scaled(Regularize(NormL1(T(1)), 0), 2), T[3, -2], Inf_), T[0, 0])
    # the sum with a function whose step it shares
    g = Regularize(NormL1(T(1)), 0)
    @test same(prox(Translate(g, T[1, 1]), T[3, -2], Inf_), prox(Translate(NormL1(T(1)), T[1, 1]), T[3, -2], Inf_))
  end
  @testset "argument and step types" begin
    # Float32 input with a Float64 output, and Float64 parameters with a Float32 argument
    for ρ in (0, 0.0, Float32(0))
      y = zeros(Float64, 2)
      prox!(y, Regularize(NormL1(1), ρ), Float32[3, -2], Inf)
      @test same(y, [0.0, 0.0])
      y = zeros(Float64, 3)
      v = value_and_prox!(y, Regularize(ZeroFunction(), ρ), Float32[3, -2, 0.5], Inf)
      @test same(y, [3.0, -2.0, 0.5]) && iszero(v)
      @test same(prox(Regularize(NormL1(1.0), ρ), Float32[3, -2], Float32(Inf)), Float32[0, 0])
    end
    # an Int argument gives a float
    @test prox_block(Regularize(NormL1(1), 0), 2, 1, Inf) === 0.0
    @test prox_block(Regularize(ZeroFunction(), 0), 2, 1, Inf) === 2.0
    @test prox_block(Regularize(NormL1(1), 0), 2, 1, 1) === 1.0
    # an integer output that cannot hold the point of the second group stops before it writes
    y = fill(17, 4)
    r = Regularize(GroupNormL2(0.5, [1:2, 3:4]), 0.0)
    @test_throws InexactError prox!(y, r, [3, 4, 3, 4], [Inf, 0.9])
    @test y == fill(17, 4)
  end
  @testset "the level is decided as given" begin
    # a level that rounds to zero in Float32 is no zero level: at Inf the result is not the
    # identity of f = ZeroFunction (a false finite point, which E4 and R3 forbid)
    f = Regularize(ZeroFunction(), 1e-300)
    xs = Float32[3, -2]
    @test !same(prox(f, xs, Inf), xs)
    @test !same(prox(f, xs, Float32(Inf)), xs)
    y = Float32[7, 8]
    prox!(y, f, xs, Inf)
    @test !same(y, xs)
    @test same(prox(f, Float64[3, -2], Inf), [0.0, -0.0])
    @test same(prox(Regularize(ZeroFunction(), 0), xs, Inf), xs)
    # and at a zero level of another type
    @test same(prox(Regularize(ZeroFunction(), 0.0f0), Float64[3, -2], Inf), [3.0, -2.0])
    @test same(prox(Regularize(ZeroFunction(), 0.0), xs, Inf), xs)
    @test same(prox(Regularize(ZeroFunction(), big(0.0)), Float64[3, -2], Inf), [3.0, -2.0])
  end
end

####################################
#  Frozen base values
####################################

# Literal results taken at the base commit, for the levels +0 and -0 and the steps +0, -0, 1, NaN, -Inf and
# Inf, on the arguments FX (and the centers FC, the weights FW). At the step Inf they are the results of the
# step 1 (for Regularize: the shrunk point of the step 1, then f at Inf), with the signs of zeros. Every
# other step keeps the arithmetic of the base, a NaN step included: it is neither rejected nor read as a
# step. Each route must give them: prox, prox!, the fused point, prox_block, array steps, per-group steps
# and the coordinate step.
const FX = (0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf)          # the arguments
const FC = (0.25, -3.0, 2.0, 0.0, 1.0, -1.0, 5.0, 0.5, 0.0)           # the centers
const FW = (0.0, 1.0, Inf, 0.5, 1.0, 0.0, 2.0, Inf, 1.0)              # the weights
const FGROUPS = [1:2, 3:4]
const FGX = (3.0, 4.0, 0.0, -0.0)
const FGC = (1.0, -1.0, 0.0, 2.0)

# the exception of a call, or its value
outcome(f) = try f() catch e; e end

function frozen_function(::Type{T}, family, kind, centered, level) where {T}
  c = centered ? T[FC...] : nothing
  if family === :capped
    CappedL1(T(level), T(1), kind === :weighted ? T[FW...] : nothing)
  elseif family === :sqr
    f = SqrNormL2(T(level), c)
    kind === :box ? Constrained(f, IndBox(T(0), T(1))) : f
  else
    Regularize(kind === :zero ? ZeroFunction() : NormL1(T(1)), T(level), c)
  end
end

# every route of f at the step γ on x, with γ as given (a Float64 step on Float32 data is not converted
# before the oracle sees it): the point of prox, the entries one by one, and array steps
function frozen_routes(f, x::Vector{T}, γ, array) where {T}
  n = length(x)
  r = Pair{Symbol, Any}[:prox => outcome(() -> prox(f, x, γ)),
                        :inplace => outcome(() -> (z = copy(x); prox!(z, f, z, γ); z)),
                        :fused => outcome(() -> (z = similar(x); value_and_prox!(z, f, x, γ); z)),
                        :entry => outcome(() -> [prox_block(f, x[i], i, γ) for i in 1:n])]
  array && push!(r, :array => outcome(() -> prox(f, x, fill(γ, n))),
                    :arrayfused => outcome(() -> (z = similar(x); value_and_prox!(z, f, x, fill(γ, n)); z)))
  r
end

# the group routes of Regularize(GroupNormL2, level[, c]): the point of prox, per-group steps, and the
# coordinate step of each group on an AtomIterate (the other groups stay as they are)
function frozen_group_routes(::Type{T}, centered, level, γ) where {T}
  f = GroupNormL2(T(2.5), FGROUPS, T[1, 0])
  r = Regularize(f, T(level), centered ? T[FGC...] : nothing)
  x = T[FGX...]
  res = Pair{Symbol, Any}[:prox => outcome(() -> prox(r, x, γ)),
                          :steps => outcome(() -> prox(r, x, fill(γ, 2)))]
  for k in 1:2
    push!(res, Symbol(:coord, k) => outcome(() -> (xi = AtomIterate(copy(x), FGROUPS); take_block_step!(xi, r, k, γ); Vector(xi))))
  end
  res
end

const FROZEN = [
  (:capped, :plain, false, 0.0, 0.0) => [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf],
  (:capped, :plain, false, 0.0, -0.0) => [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf],
  (:capped, :plain, false, 0.0, 1.0) => [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf],
  (:capped, :plain, false, 0.0, NaN) => [1.0, -1.0, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf],
  (:capped, :plain, false, 0.0, -Inf) => [1.0, -1.0, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf],
  (:capped, :plain, false, 0.0, Inf) => [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf],
  (:capped, :plain, false, -0.0, 0.0) => [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf],
  (:capped, :plain, false, -0.0, -0.0) => [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf],
  (:capped, :plain, false, -0.0, 1.0) => [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf],
  (:capped, :plain, false, -0.0, NaN) => [1.0, -1.0, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf],
  (:capped, :plain, false, -0.0, -Inf) => [1.0, -1.0, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf],
  (:capped, :plain, false, -0.0, Inf) => [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf],
  (:capped, :weighted, false, 0.0, 0.0) => [0.5, -0.5, 0.0, -2.0, 0.0, -0.0, NaN, NaN, -Inf],
  (:capped, :weighted, false, 0.0, -0.0) => [0.5, -0.5, 0.0, -2.0, 0.0, -0.0, NaN, NaN, -Inf],
  (:capped, :weighted, false, 0.0, 1.0) => [0.5, -0.5, 0.0, -2.0, 0.0, -0.0, NaN, NaN, -Inf],
  (:capped, :weighted, false, 0.0, NaN) => [1.0, -1.0, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf],
  (:capped, :weighted, false, 0.0, -Inf) => [1.0, -1.0, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf],
  (:capped, :weighted, false, 0.0, Inf) => [0.5, -0.5, 0.0, -2.0, 0.0, -0.0, NaN, NaN, -Inf],
  (:capped, :weighted, false, -0.0, 0.0) => [0.5, -0.5, 0.0, -2.0, 0.0, -0.0, NaN, NaN, -Inf],
  (:capped, :weighted, false, -0.0, -0.0) => [0.5, -0.5, 0.0, -2.0, 0.0, -0.0, NaN, NaN, -Inf],
  (:capped, :weighted, false, -0.0, 1.0) => [0.5, -0.5, 0.0, -2.0, 0.0, -0.0, NaN, NaN, -Inf],
  (:capped, :weighted, false, -0.0, NaN) => [1.0, -1.0, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf],
  (:capped, :weighted, false, -0.0, -Inf) => [1.0, -1.0, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf],
  (:capped, :weighted, false, -0.0, Inf) => [0.5, -0.5, 0.0, -2.0, 0.0, -0.0, NaN, NaN, -Inf],
  (:sqr, :plain, false, 0.0, 0.0) => [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf],
  (:sqr, :plain, false, 0.0, -0.0) => [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf],
  (:sqr, :plain, false, 0.0, 1.0) => [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf],
  (:sqr, :plain, false, 0.0, NaN) => [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
  (:sqr, :plain, false, 0.0, -Inf) => [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
  (:sqr, :plain, false, 0.0, Inf) => [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf],
  (:sqr, :plain, false, -0.0, 0.0) => [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf],
  (:sqr, :plain, false, -0.0, -0.0) => [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf],
  (:sqr, :plain, false, -0.0, 1.0) => [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf],
  (:sqr, :plain, false, -0.0, NaN) => [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
  (:sqr, :plain, false, -0.0, -Inf) => [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
  (:sqr, :plain, false, -0.0, Inf) => [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf],
  (:sqr, :plain, true, 0.0, 0.0) => [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, NaN, -Inf],
  (:sqr, :plain, true, 0.0, -0.0) => [0.5, -0.5, 2.0, -2.0, 0.0, 0.0, NaN, NaN, -Inf],
  (:sqr, :plain, true, 0.0, 1.0) => [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, NaN, -Inf],
  (:sqr, :plain, true, 0.0, NaN) => [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
  (:sqr, :plain, true, 0.0, -Inf) => [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
  (:sqr, :plain, true, 0.0, Inf) => [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, NaN, -Inf],
  (:sqr, :plain, true, -0.0, 0.0) => [0.5, -0.5, 2.0, -2.0, 0.0, 0.0, NaN, NaN, -Inf],
  (:sqr, :plain, true, -0.0, -0.0) => [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, NaN, -Inf],
  (:sqr, :plain, true, -0.0, 1.0) => [0.5, -0.5, 2.0, -2.0, 0.0, 0.0, NaN, NaN, -Inf],
  (:sqr, :plain, true, -0.0, NaN) => [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
  (:sqr, :plain, true, -0.0, -Inf) => [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
  (:sqr, :plain, true, -0.0, Inf) => [0.5, -0.5, 2.0, -2.0, 0.0, 0.0, NaN, NaN, -Inf],
  (:sqr, :box, false, 0.0, 0.0) => [0.5, 0.0, 1.0, 0.0, 0.0, -0.0, NaN, 1.0, 0.0],
  (:sqr, :box, false, 0.0, -0.0) => [0.5, 0.0, 1.0, 0.0, 0.0, -0.0, NaN, 1.0, 0.0],
  (:sqr, :box, false, 0.0, 1.0) => [0.5, 0.0, 1.0, 0.0, 0.0, -0.0, NaN, 1.0, 0.0],
  (:sqr, :box, false, 0.0, NaN) => [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
  (:sqr, :box, false, 0.0, -Inf) => [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
  (:sqr, :box, false, 0.0, Inf) => [0.5, 0.0, 1.0, 0.0, 0.0, -0.0, NaN, 1.0, 0.0],
  (:sqr, :box, false, -0.0, 0.0) => [0.5, 0.0, 1.0, 0.0, 0.0, -0.0, NaN, 1.0, 0.0],
  (:sqr, :box, false, -0.0, -0.0) => [0.5, 0.0, 1.0, 0.0, 0.0, -0.0, NaN, 1.0, 0.0],
  (:sqr, :box, false, -0.0, 1.0) => [0.5, 0.0, 1.0, 0.0, 0.0, -0.0, NaN, 1.0, 0.0],
  (:sqr, :box, false, -0.0, NaN) => [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
  (:sqr, :box, false, -0.0, -Inf) => [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
  (:sqr, :box, false, -0.0, Inf) => [0.5, 0.0, 1.0, 0.0, 0.0, -0.0, NaN, 1.0, 0.0],
  (:sqr, :box, true, 0.0, 0.0) => [0.5, 0.0, 1.0, 0.0, 0.0, -0.0, NaN, NaN, 0.0],
  (:sqr, :box, true, 0.0, -0.0) => [0.5, 0.0, 1.0, 0.0, 0.0, 0.0, NaN, NaN, 0.0],
  (:sqr, :box, true, 0.0, 1.0) => [0.5, 0.0, 1.0, 0.0, 0.0, -0.0, NaN, NaN, 0.0],
  (:sqr, :box, true, 0.0, NaN) => [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
  (:sqr, :box, true, 0.0, -Inf) => [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
  (:sqr, :box, true, 0.0, Inf) => [0.5, 0.0, 1.0, 0.0, 0.0, -0.0, NaN, NaN, 0.0],
  (:sqr, :box, true, -0.0, 0.0) => [0.5, 0.0, 1.0, 0.0, 0.0, 0.0, NaN, NaN, 0.0],
  (:sqr, :box, true, -0.0, -0.0) => [0.5, 0.0, 1.0, 0.0, 0.0, -0.0, NaN, NaN, 0.0],
  (:sqr, :box, true, -0.0, 1.0) => [0.5, 0.0, 1.0, 0.0, 0.0, 0.0, NaN, NaN, 0.0],
  (:sqr, :box, true, -0.0, NaN) => [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
  (:sqr, :box, true, -0.0, -Inf) => [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
  (:sqr, :box, true, -0.0, Inf) => [0.5, 0.0, 1.0, 0.0, 0.0, 0.0, NaN, NaN, 0.0],
  (:reg, :zero, false, 0.0, 0.0) => [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf],
  (:reg, :zero, false, 0.0, -0.0) => [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf],
  (:reg, :zero, false, 0.0, 1.0) => [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf],
  (:reg, :zero, false, 0.0, NaN) => [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
  (:reg, :zero, false, 0.0, -Inf) => [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
  (:reg, :zero, false, 0.0, Inf) => [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf],
  (:reg, :zero, false, -0.0, 0.0) => [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf],
  (:reg, :zero, false, -0.0, -0.0) => [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf],
  (:reg, :zero, false, -0.0, 1.0) => [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf],
  (:reg, :zero, false, -0.0, NaN) => [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
  (:reg, :zero, false, -0.0, -Inf) => [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
  (:reg, :zero, false, -0.0, Inf) => [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf],
  (:reg, :zero, true, 0.0, 0.0) => [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, NaN, -Inf],
  (:reg, :zero, true, 0.0, -0.0) => [0.5, -0.5, 2.0, -2.0, 0.0, 0.0, NaN, NaN, -Inf],
  (:reg, :zero, true, 0.0, 1.0) => [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, NaN, -Inf],
  (:reg, :zero, true, 0.0, NaN) => [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
  (:reg, :zero, true, 0.0, -Inf) => [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
  (:reg, :zero, true, 0.0, Inf) => [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, NaN, -Inf],
  (:reg, :zero, true, -0.0, 0.0) => [0.5, -0.5, 2.0, -2.0, 0.0, 0.0, NaN, NaN, -Inf],
  (:reg, :zero, true, -0.0, -0.0) => [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, NaN, -Inf],
  (:reg, :zero, true, -0.0, 1.0) => [0.5, -0.5, 2.0, -2.0, 0.0, 0.0, NaN, NaN, -Inf],
  (:reg, :zero, true, -0.0, NaN) => [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
  (:reg, :zero, true, -0.0, -Inf) => [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
  (:reg, :zero, true, -0.0, Inf) => [0.5, -0.5, 2.0, -2.0, 0.0, 0.0, NaN, NaN, -Inf],
  (:reg, :l1, false, 0.0, 0.0) => [0.5, -0.5, 2.0, -2.0, 0.0, 0.0, NaN, Inf, -Inf],
  (:reg, :l1, false, 0.0, -0.0) => [0.5, -0.5, 2.0, -2.0, 0.0, 0.0, NaN, Inf, -Inf],
  (:reg, :l1, false, 0.0, 1.0) => [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, NaN, Inf, -Inf],
  (:reg, :l1, false, 0.0, NaN) => [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
  (:reg, :l1, false, 0.0, -Inf) => [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
  (:reg, :l1, false, 0.0, Inf) => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, NaN, NaN, NaN],
  (:reg, :l1, false, -0.0, 0.0) => [0.5, -0.5, 2.0, -2.0, 0.0, 0.0, NaN, Inf, -Inf],
  (:reg, :l1, false, -0.0, -0.0) => [0.5, -0.5, 2.0, -2.0, 0.0, 0.0, NaN, Inf, -Inf],
  (:reg, :l1, false, -0.0, 1.0) => [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, NaN, Inf, -Inf],
  (:reg, :l1, false, -0.0, NaN) => [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
  (:reg, :l1, false, -0.0, -Inf) => [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
  (:reg, :l1, false, -0.0, Inf) => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, NaN, NaN, NaN],
  (:reg, :l1, true, 0.0, 0.0) => [0.5, -0.5, 2.0, -2.0, 0.0, 0.0, NaN, NaN, -Inf],
  (:reg, :l1, true, 0.0, -0.0) => [0.5, -0.5, 2.0, -2.0, 0.0, 0.0, NaN, NaN, -Inf],
  (:reg, :l1, true, 0.0, 1.0) => [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, NaN, NaN, -Inf],
  (:reg, :l1, true, 0.0, NaN) => [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
  (:reg, :l1, true, 0.0, -Inf) => [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
  (:reg, :l1, true, 0.0, Inf) => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, NaN, NaN, NaN],
  (:reg, :l1, true, -0.0, 0.0) => [0.5, -0.5, 2.0, -2.0, 0.0, 0.0, NaN, NaN, -Inf],
  (:reg, :l1, true, -0.0, -0.0) => [0.5, -0.5, 2.0, -2.0, 0.0, 0.0, NaN, NaN, -Inf],
  (:reg, :l1, true, -0.0, 1.0) => [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, NaN, NaN, -Inf],
  (:reg, :l1, true, -0.0, NaN) => [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
  (:reg, :l1, true, -0.0, -Inf) => [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
  (:reg, :l1, true, -0.0, Inf) => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, NaN, NaN, NaN],
]

const FROZEN_GROUPS = [
  (:group, false, 0.0, 0.0) => (prox = [3.0, 4.0, 0.0, -0.0], coord1 = [3.0, 4.0, 0.0, -0.0], coord2 = [3.0, 4.0, 0.0, -0.0]),
  (:group, false, 0.0, -0.0) => (prox = [3.0, 4.0, 0.0, -0.0], coord1 = [3.0, 4.0, 0.0, -0.0], coord2 = [3.0, 4.0, 0.0, -0.0]),
  (:group, false, 0.0, 1.0) => (prox = [1.5, 2.0, 0.0, -0.0], coord1 = [1.5, 2.0, 0.0, -0.0], coord2 = [3.0, 4.0, 0.0, -0.0]),
  (:group, false, 0.0, NaN) => (prox = [NaN, NaN, NaN, NaN], coord1 = [NaN, NaN, 0.0, -0.0], coord2 = [3.0, 4.0, NaN, NaN]),
  (:group, false, 0.0, -Inf) => (prox = [NaN, NaN, NaN, NaN], coord1 = [NaN, NaN, 0.0, -0.0], coord2 = [3.0, 4.0, NaN, NaN]),
  (:group, false, 0.0, Inf) => (prox = [0.0, 0.0, 0.0, -0.0], coord1 = [0.0, 0.0, 0.0, -0.0], coord2 = [3.0, 4.0, 0.0, -0.0]),
  (:group, false, -0.0, 0.0) => (prox = [3.0, 4.0, 0.0, -0.0], coord1 = [3.0, 4.0, 0.0, -0.0], coord2 = [3.0, 4.0, 0.0, -0.0]),
  (:group, false, -0.0, -0.0) => (prox = [3.0, 4.0, 0.0, -0.0], coord1 = [3.0, 4.0, 0.0, -0.0], coord2 = [3.0, 4.0, 0.0, -0.0]),
  (:group, false, -0.0, 1.0) => (prox = [1.5, 2.0, 0.0, -0.0], coord1 = [1.5, 2.0, 0.0, -0.0], coord2 = [3.0, 4.0, 0.0, -0.0]),
  (:group, false, -0.0, NaN) => (prox = [NaN, NaN, NaN, NaN], coord1 = [NaN, NaN, 0.0, -0.0], coord2 = [3.0, 4.0, NaN, NaN]),
  (:group, false, -0.0, -Inf) => (prox = [NaN, NaN, NaN, NaN], coord1 = [NaN, NaN, 0.0, -0.0], coord2 = [3.0, 4.0, NaN, NaN]),
  (:group, false, -0.0, Inf) => (prox = [0.0, 0.0, 0.0, -0.0], coord1 = [0.0, 0.0, 0.0, -0.0], coord2 = [3.0, 4.0, 0.0, -0.0]),
  (:group, true, 0.0, 0.0) => (prox = [3.0, 4.0, 0.0, 0.0], coord1 = [3.0, 4.0, 0.0, -0.0], coord2 = [3.0, 4.0, 0.0, 0.0]),
  (:group, true, 0.0, -0.0) => (prox = [3.0, 4.0, 0.0, -0.0], coord1 = [3.0, 4.0, 0.0, -0.0], coord2 = [3.0, 4.0, 0.0, -0.0]),
  (:group, true, 0.0, 1.0) => (prox = [1.5, 2.0, 0.0, 0.0], coord1 = [1.5, 2.0, 0.0, -0.0], coord2 = [3.0, 4.0, 0.0, 0.0]),
  (:group, true, 0.0, NaN) => (prox = [NaN, NaN, NaN, NaN], coord1 = [NaN, NaN, 0.0, -0.0], coord2 = [3.0, 4.0, NaN, NaN]),
  (:group, true, 0.0, -Inf) => (prox = [NaN, NaN, NaN, NaN], coord1 = [NaN, NaN, 0.0, -0.0], coord2 = [3.0, 4.0, NaN, NaN]),
  (:group, true, 0.0, Inf) => (prox = [0.0, 0.0, 0.0, 0.0], coord1 = [0.0, 0.0, 0.0, -0.0], coord2 = [3.0, 4.0, 0.0, 0.0]),
  (:group, true, -0.0, 0.0) => (prox = [3.0, 4.0, 0.0, -0.0], coord1 = [3.0, 4.0, 0.0, -0.0], coord2 = [3.0, 4.0, 0.0, -0.0]),
  (:group, true, -0.0, -0.0) => (prox = [3.0, 4.0, 0.0, 0.0], coord1 = [3.0, 4.0, 0.0, -0.0], coord2 = [3.0, 4.0, 0.0, 0.0]),
  (:group, true, -0.0, 1.0) => (prox = [1.5, 2.0, 0.0, -0.0], coord1 = [1.5, 2.0, 0.0, -0.0], coord2 = [3.0, 4.0, 0.0, -0.0]),
  (:group, true, -0.0, NaN) => (prox = [NaN, NaN, NaN, NaN], coord1 = [NaN, NaN, 0.0, -0.0], coord2 = [3.0, 4.0, NaN, NaN]),
  (:group, true, -0.0, -Inf) => (prox = [NaN, NaN, NaN, NaN], coord1 = [NaN, NaN, 0.0, -0.0], coord2 = [3.0, 4.0, NaN, NaN]),
  (:group, true, -0.0, Inf) => (prox = [0.0, 0.0, 0.0, -0.0], coord1 = [0.0, 0.0, 0.0, -0.0], coord2 = [3.0, 4.0, 0.0, -0.0]),
]

# A finite step beyond the range of Float32, on Float32 data (a Float64 step, as given): the base reads
# it as Inf after the conversion and keeps the arithmetic of the formulas, so it is no endpoint. The
# step -1e100 is invalid and keeps the base arithmetic. Float64 data keeps 1e100 finite. The values of
# CappedL1 and SqrNormL2 do not depend on the element type, those of Regularize do.
const FROZEN_OVERFLOW = [
  (:capped, :plain, false, 0.0, 1.0e100) => (f32 = [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf], f64 = [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf]),
  (:capped, :plain, false, 0.0, -1.0e100) => (f32 = [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf], f64 = [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf]),
  (:capped, :plain, false, -0.0, 1.0e100) => (f32 = [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf], f64 = [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf]),
  (:capped, :plain, false, -0.0, -1.0e100) => (f32 = [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf], f64 = [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf]),
  (:capped, :weighted, false, 0.0, 1.0e100) => (f32 = [0.5, -0.5, 0.0, -2.0, 0.0, -0.0, NaN, NaN, -Inf], f64 = [0.5, -0.5, 0.0, -2.0, 0.0, -0.0, NaN, NaN, -Inf]),
  (:capped, :weighted, false, 0.0, -1.0e100) => (f32 = [0.5, -0.5, 0.0, -2.0, 0.0, -0.0, NaN, NaN, -Inf], f64 = [0.5, -0.5, 0.0, -2.0, 0.0, -0.0, NaN, NaN, -Inf]),
  (:capped, :weighted, false, -0.0, 1.0e100) => (f32 = [0.5, -0.5, 0.0, -2.0, 0.0, -0.0, NaN, NaN, -Inf], f64 = [0.5, -0.5, 0.0, -2.0, 0.0, -0.0, NaN, NaN, -Inf]),
  (:capped, :weighted, false, -0.0, -1.0e100) => (f32 = [0.5, -0.5, 0.0, -2.0, 0.0, -0.0, NaN, NaN, -Inf], f64 = [0.5, -0.5, 0.0, -2.0, 0.0, -0.0, NaN, NaN, -Inf]),
  (:sqr, :plain, false, 0.0, 1.0e100) => (f32 = [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf], f64 = [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf]),
  (:sqr, :plain, false, 0.0, -1.0e100) => (f32 = [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf], f64 = [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf]),
  (:sqr, :plain, false, -0.0, 1.0e100) => (f32 = [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf], f64 = [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf]),
  (:sqr, :plain, false, -0.0, -1.0e100) => (f32 = [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf], f64 = [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf]),
  (:sqr, :plain, true, 0.0, 1.0e100) => (f32 = [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, NaN, -Inf], f64 = [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, NaN, -Inf]),
  (:sqr, :plain, true, 0.0, -1.0e100) => (f32 = [0.5, -0.5, 2.0, -2.0, 0.0, 0.0, NaN, NaN, -Inf], f64 = [0.5, -0.5, 2.0, -2.0, 0.0, 0.0, NaN, NaN, -Inf]),
  (:sqr, :plain, true, -0.0, 1.0e100) => (f32 = [0.5, -0.5, 2.0, -2.0, 0.0, 0.0, NaN, NaN, -Inf], f64 = [0.5, -0.5, 2.0, -2.0, 0.0, 0.0, NaN, NaN, -Inf]),
  (:sqr, :plain, true, -0.0, -1.0e100) => (f32 = [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, NaN, -Inf], f64 = [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, NaN, -Inf]),
  (:sqr, :box, false, 0.0, 1.0e100) => (f32 = [0.5, 0.0, 1.0, 0.0, 0.0, -0.0, NaN, 1.0, 0.0], f64 = [0.5, 0.0, 1.0, 0.0, 0.0, -0.0, NaN, 1.0, 0.0]),
  (:sqr, :box, false, 0.0, -1.0e100) => (f32 = [0.5, 0.0, 1.0, 0.0, 0.0, -0.0, NaN, 1.0, 0.0], f64 = [0.5, 0.0, 1.0, 0.0, 0.0, -0.0, NaN, 1.0, 0.0]),
  (:sqr, :box, false, -0.0, 1.0e100) => (f32 = [0.5, 0.0, 1.0, 0.0, 0.0, -0.0, NaN, 1.0, 0.0], f64 = [0.5, 0.0, 1.0, 0.0, 0.0, -0.0, NaN, 1.0, 0.0]),
  (:sqr, :box, false, -0.0, -1.0e100) => (f32 = [0.5, 0.0, 1.0, 0.0, 0.0, -0.0, NaN, 1.0, 0.0], f64 = [0.5, 0.0, 1.0, 0.0, 0.0, -0.0, NaN, 1.0, 0.0]),
  (:sqr, :box, true, 0.0, 1.0e100) => (f32 = [0.5, 0.0, 1.0, 0.0, 0.0, -0.0, NaN, NaN, 0.0], f64 = [0.5, 0.0, 1.0, 0.0, 0.0, -0.0, NaN, NaN, 0.0]),
  (:sqr, :box, true, 0.0, -1.0e100) => (f32 = [0.5, 0.0, 1.0, 0.0, 0.0, 0.0, NaN, NaN, 0.0], f64 = [0.5, 0.0, 1.0, 0.0, 0.0, 0.0, NaN, NaN, 0.0]),
  (:sqr, :box, true, -0.0, 1.0e100) => (f32 = [0.5, 0.0, 1.0, 0.0, 0.0, 0.0, NaN, NaN, 0.0], f64 = [0.5, 0.0, 1.0, 0.0, 0.0, 0.0, NaN, NaN, 0.0]),
  (:sqr, :box, true, -0.0, -1.0e100) => (f32 = [0.5, 0.0, 1.0, 0.0, 0.0, -0.0, NaN, NaN, 0.0], f64 = [0.5, 0.0, 1.0, 0.0, 0.0, -0.0, NaN, NaN, 0.0]),
  (:reg, :zero, false, 0.0, 1.0e100) => (f32 = [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN], f64 = [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf]),
  (:reg, :zero, false, 0.0, -1.0e100) => (f32 = [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN], f64 = [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf]),
  (:reg, :zero, false, -0.0, 1.0e100) => (f32 = [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN], f64 = [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf]),
  (:reg, :zero, false, -0.0, -1.0e100) => (f32 = [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN], f64 = [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, Inf, -Inf]),
  (:reg, :zero, true, 0.0, 1.0e100) => (f32 = [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN], f64 = [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, NaN, -Inf]),
  (:reg, :zero, true, 0.0, -1.0e100) => (f32 = [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN], f64 = [0.5, -0.5, 2.0, -2.0, 0.0, 0.0, NaN, NaN, -Inf]),
  (:reg, :zero, true, -0.0, 1.0e100) => (f32 = [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN], f64 = [0.5, -0.5, 2.0, -2.0, 0.0, 0.0, NaN, NaN, -Inf]),
  (:reg, :zero, true, -0.0, -1.0e100) => (f32 = [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN], f64 = [0.5, -0.5, 2.0, -2.0, 0.0, -0.0, NaN, NaN, -Inf]),
  (:reg, :l1, false, 0.0, 1.0e100) => (f32 = [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN], f64 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, NaN, Inf, -Inf]),
  (:reg, :l1, false, 0.0, -1.0e100) => (f32 = [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN], f64 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, NaN, Inf, -Inf]),
  (:reg, :l1, false, -0.0, 1.0e100) => (f32 = [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN], f64 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, NaN, Inf, -Inf]),
  (:reg, :l1, false, -0.0, -1.0e100) => (f32 = [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN], f64 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, NaN, Inf, -Inf]),
  (:reg, :l1, true, 0.0, 1.0e100) => (f32 = [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN], f64 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, NaN, NaN, -Inf]),
  (:reg, :l1, true, 0.0, -1.0e100) => (f32 = [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN], f64 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, NaN, NaN, -Inf]),
  (:reg, :l1, true, -0.0, 1.0e100) => (f32 = [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN], f64 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, NaN, NaN, -Inf]),
  (:reg, :l1, true, -0.0, -1.0e100) => (f32 = [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN], f64 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, NaN, NaN, -Inf]),
]

const FROZEN_OVERFLOW_GROUPS = [
  (:group, false, 0.0, 1.0e100) => (f32 = (prox = [NaN, NaN, NaN, NaN], coord1 = [NaN, NaN, 0.0, -0.0], coord2 = [3.0, 4.0, NaN, NaN]), f64 = (prox = [0.0, 0.0, 0.0, -0.0], coord1 = [0.0, 0.0, 0.0, -0.0], coord2 = [3.0, 4.0, 0.0, -0.0])),
  (:group, false, 0.0, -1.0e100) => (f32 = (prox = [NaN, NaN, NaN, NaN], coord1 = [NaN, NaN, 0.0, -0.0], coord2 = [3.0, 4.0, NaN, NaN]), f64 = (prox = [1.5000000000000001e100, 2.0e100, 0.0, -0.0], coord1 = [1.5000000000000001e100, 2.0e100, 0.0, -0.0], coord2 = [3.0, 4.0, 0.0, -0.0])),
  (:group, false, -0.0, 1.0e100) => (f32 = (prox = [NaN, NaN, NaN, NaN], coord1 = [NaN, NaN, 0.0, -0.0], coord2 = [3.0, 4.0, NaN, NaN]), f64 = (prox = [0.0, 0.0, 0.0, -0.0], coord1 = [0.0, 0.0, 0.0, -0.0], coord2 = [3.0, 4.0, 0.0, -0.0])),
  (:group, false, -0.0, -1.0e100) => (f32 = (prox = [NaN, NaN, NaN, NaN], coord1 = [NaN, NaN, 0.0, -0.0], coord2 = [3.0, 4.0, NaN, NaN]), f64 = (prox = [1.5000000000000001e100, 2.0e100, 0.0, -0.0], coord1 = [1.5000000000000001e100, 2.0e100, 0.0, -0.0], coord2 = [3.0, 4.0, 0.0, -0.0])),
  (:group, true, 0.0, 1.0e100) => (f32 = (prox = [NaN, NaN, NaN, NaN], coord1 = [NaN, NaN, 0.0, -0.0], coord2 = [3.0, 4.0, NaN, NaN]), f64 = (prox = [0.0, 0.0, 0.0, 0.0], coord1 = [0.0, 0.0, 0.0, -0.0], coord2 = [3.0, 4.0, 0.0, 0.0])),
  (:group, true, 0.0, -1.0e100) => (f32 = (prox = [NaN, NaN, NaN, NaN], coord1 = [NaN, NaN, 0.0, -0.0], coord2 = [3.0, 4.0, NaN, NaN]), f64 = (prox = [1.5000000000000001e100, 2.0e100, 0.0, -0.0], coord1 = [1.5000000000000001e100, 2.0e100, 0.0, -0.0], coord2 = [3.0, 4.0, 0.0, -0.0])),
  (:group, true, -0.0, 1.0e100) => (f32 = (prox = [NaN, NaN, NaN, NaN], coord1 = [NaN, NaN, 0.0, -0.0], coord2 = [3.0, 4.0, NaN, NaN]), f64 = (prox = [0.0, 0.0, 0.0, -0.0], coord1 = [0.0, 0.0, 0.0, -0.0], coord2 = [3.0, 4.0, 0.0, -0.0])),
  (:group, true, -0.0, -1.0e100) => (f32 = (prox = [NaN, NaN, NaN, NaN], coord1 = [NaN, NaN, 0.0, -0.0], coord2 = [3.0, 4.0, NaN, NaN]), f64 = (prox = [1.5000000000000001e100, 2.0e100, 0.0, 0.0], coord1 = [1.5000000000000001e100, 2.0e100, 0.0, -0.0], coord2 = [3.0, 4.0, 0.0, 0.0])),
]

@testset "frozen base values, $T" for T in (Float32, Float64)
  x = T[FX...]
  for ((family, kind, centered, level, γ), want) in FROZEN
    f = frozen_function(T, family, kind, centered, level)
    for (route, got) in frozen_routes(f, x, γ, family !== :sqr)
      ok = same(got, T.(want))
      ok || @info "frozen value differs" family kind centered level γ route got want
      @test ok
    end
  end
  for ((_, centered, level, γ), want) in FROZEN_GROUPS
    for (route, got) in frozen_group_routes(T, centered, level, γ)
      w = route === :coord1 ? want.coord1 : route === :coord2 ? want.coord2 : want.prox
      ok = same(got, T.(w))
      ok || @info "frozen group value differs" centered level γ route got w
      @test ok
    end
  end
  # the steps beyond the range of Float32: each element type has its own base values
  for ((family, kind, centered, level, γ), want) in FROZEN_OVERFLOW
    f = frozen_function(T, family, kind, centered, level)
    w = T === Float32 ? want.f32 : want.f64
    for (route, got) in frozen_routes(f, x, γ, family !== :sqr)
      ok = same(got, T.(w))
      ok || @info "frozen overflow value differs" family kind centered level γ route got w
      @test ok
    end
  end
  for ((_, centered, level, γ), want0) in FROZEN_OVERFLOW_GROUPS
    want = T === Float32 ? want0.f32 : want0.f64
    for (route, got) in frozen_group_routes(T, centered, level, γ)
      w = route === :coord1 ? want.coord1 : route === :coord2 ? want.coord2 : want.prox
      ok = same(got, T.(w))
      ok || @info "frozen overflow group value differs" centered level γ route got w
      @test ok
    end
  end
end

# the inputs that show the two ways to go wrong, by name (the base values)
@testset "signed zeros and NaN steps, named inputs, $T" for T in (Float32, Float64)
  z, nz, N, Inf_ = zero(T), -zero(T), T(NaN), T(Inf)
  # a level of negative zero at the step 1: the sign of the product is kept, on every route
  @test same(prox(SqrNormL2(nz, T[0]), T[-0.0], one(T)), T[-0.0])
  r = Regularize(ZeroFunction(), nz, T[0])
  @test same(prox(r, T[-0.0], one(T)), T[-0.0])
  @test same(prox_block(r, T(-0.0), 1, one(T)), T(-0.0))
  @test same(prox(r, T[-0.0], T[1]), T[-0.0])
  # a level of positive zero at the step -0
  @test same(prox(SqrNormL2(z, T[0]), T[-0.0], nz), T[-0.0])
  r = Regularize(ZeroFunction(), z, T[0])
  @test same(prox(r, T[-0.0], nz), T[-0.0])
  @test same(prox_block(r, T(-0.0), 1, nz), T(-0.0))
  @test same(prox(r, T[-0.0], T[-0.0]), T[-0.0])
  # and the same at the endpoint, where the sign is that of the level
  @test same(prox(SqrNormL2(nz, T[0]), T[-0.0], Inf_), T[-0.0])
  @test same(prox(Regularize(ZeroFunction(), nz, T[0]), T[-0.0], Inf_), T[-0.0])
  @test same(prox_block(Regularize(ZeroFunction(), nz, T[0]), T(-0.0), 1, Inf_), T(-0.0))
  # a finite step beyond the range of the data type is no endpoint, whatever the zero sign of the level
  for ρ in (z, nz)
    r = Regularize(ZeroFunction(), ρ)
    big = 1e100                                       # a Float64 step, as given
    want = T === Float32 ? T[NaN] : T[0.5]
    @test same(prox(r, T[0.5], big), want)
    @test same(prox(r, T[0.5], [big]), want)
    @test same([prox_block(r, T(0.5), 1, big)], want)
    @test same(prox(Regularize(GroupNormL2(T(0.5), [1:1], T[0]), ρ), T[0.5], [big]), want)
    xi = AtomIterate(T[0.5], [1:1])
    take_block_step!(xi, Regularize(GroupNormL2(T(0.5), [1:1], T[0]), ρ), 1, big)
    @test same(Vector(xi), want)
    @test same(prox(r, T[0.5], Inf), T[0.5])          # next to a true Inf
  end
  # a NaN step is neither rejected nor read as a step: the result is that of the base
  for (f, want) in ((CappedL1(z, T(1)), T[1]), (SqrNormL2(z), T[NaN]), (Regularize(ZeroFunction(), z), T[NaN]),
                    (CappedL1(nz, T(1)), T[1]), (SqrNormL2(nz), T[NaN]), (Regularize(ZeroFunction(), nz), T[NaN]))
    y = T[7]
    @test !(outcome(() -> prox!(y, f, T[0.5], N)) isa Exception)
    @test same(y, want)
    @test same([prox_block(f, T(0.5), 1, N)], want)
  end
end

####################################
#  Positive levels and finite steps: unchanged
####################################

# pinned at the values before the change (Float64 and Float32 alike: the arguments, the levels and the
# results are dyadic or exact in both)
@testset "positive levels, $T" for T in (Float32, Float64)
  x = entries(T)
  N = T(NaN)
  I = T(Inf)
  # CappedL1(1, 1): the steps 1 and Inf
  f = CappedL1(T(1), T(1))
  @test same(prox(f, x, one(T)), T[0, -0.0, 0, -0.0, 2, -2, 0, -0.0, N, I, -I])
  @test same(prox(f, x, I), T[0, -0.0, 0, -0.0, 0, -0.0, 0, -0.0, N, N, N])
  # SqrNormL2(1)
  f = SqrNormL2(T(1))
  @test same(prox(f, x, one(T)), T[0.25, -0.25, 0.5, -0.5, 1, -1, 0, -0.0, N, I, -I])
  @test same(prox(f, x, I), T[0, -0.0, 0, -0.0, 0, -0.0, 0, -0.0, N, N, N])
  # Regularize(NormL1(1), 1)
  f = Regularize(NormL1(T(1)), 1)
  @test same(prox(f, x, one(T)), T[0, 0, 0, 0, 0.5, -0.5, 0, 0, N, I, -I])
  @test same(prox(f, x, I), T[0, 0, 0, 0, 0, 0, 0, 0, N, N, N])
  # a weighted level and a centered ridge
  f = CappedL1(T(1), T(1), T[0, 1, Inf, 0.5, 1, 2, 1, 1, 1, 0.5, 1])
  @test same(prox(f, x, one(T)), T[0.5, -0.0, 0, -0.5, 2, -0.0, 0, -0.0, N, I, -I])
  @test same(prox(f, x, I), T[0.5, -0.0, 0, -0.0, 0, -0.0, 0, -0.0, N, N, N])
  c = T[0.25, -3, 2, 0, 1, -1, 5, 0.5, 0, 2, -2]
  y = T[2, -1, 0.5, 0.25, 3, -3, 0, 1]
  f = SqrNormL2(T(1), c[1:8])
  @test same(prox(f, y, one(T)), T[1.125, -2, 1.25, 0.125, 2, -2, 2.5, 0.75])
  @test same(prox(f, y, I), c[1:8])
  # the zero step
  @test same(prox(CappedL1(T(1), T(1)), x, zero(T)), x)
  @test same(prox(SqrNormL2(T(1)), x, zero(T)), x)
end

####################################
#  Device arrays
####################################

@testset "device arrays, $T" for T in (Float32, Float64)
  allowscalar(false)
  x = T[0.5, -2, 2, -0.25, 1, 0, NaN, Inf]
  w = T[0, 1, Inf, 0.5, 1, Inf, 1, 0]
  Inf_ = T(Inf)
  # CappedL1: the coefficient is decided in prox_block, on the device as on the host
  for f in (CappedL1(T(0), T(1)), CappedL1(T(0), T(1), w))
    fd = adapt(JLArray, f)
    want = prox(f, x, one(T))
    @test same(Array(prox(fd, JLArray(x), Inf_)), want)
    @test same(Array(prox(fd, JLArray(x), JLArray(fill(Inf_, length(x))))), want)
    z = JLArray(copy(x))
    @test prox!(z, fd, z, Inf_) === z && same(Array(z), want)
    y = JLArray(fill(T(7), length(x)))
    v = value_and_prox!(y, fd, JLArray(x), Inf_)
    vh = f(Array(y))
    @test same(Array(y), want) && (same(v, vh) || (isfinite(vh) && v ≈ vh))
  end
  # SqrNormL2: the weights are decided on the host
  for f in (SqrNormL2(T(0)), SqrNormL2(T(0), T[1, 2, 3, 4, 5, 6, 7, 8]))
    fd = adapt(JLArray, f)
    want = prox(f, x, one(T))
    @test same(Array(prox(fd, JLArray(x), Inf_)), want)
    z = JLArray(copy(x))
    @test prox!(z, fd, z, Inf_) === z && same(Array(z), want)
  end
  # Regularize, whose copy on the device has no scratch buffer: no element is read on the host
  xf = T[0.5, -2, 2, -0.25, 1, 0, 3, -1]
  cf = T[1, -2, 0.5, 3, 0, -1, 1, 2]
  for (f, c) in ((NormL1(T(1)), nothing), (NormL1(T(1)), cf), (ZeroFunction(), nothing), (IndBox(T(0), T(1)), cf))
    r = Regularize(f, 0, c)
    rd = adapt(JLArray, r)
    want = prox(f, xf, Inf_)
    @test same(Array(prox(rd, JLArray(xf), Inf_)), want)
    z = JLArray(copy(xf))
    @test prox!(z, rd, z, Inf_) === z && same(Array(z), want)
    y = JLArray(fill(T(7), length(xf)))
    v = value_and_prox!(y, rd, JLArray(xf), Inf_)
    vh = r(Array(y))                              # the host value; a device sum may round otherwise
    @test same(Array(y), want) && (same(v, vh) || (isfinite(vh) && v ≈ vh))
  end
  @test same(Array(prox(adapt(JLArray, Regularize(NormL1(T(1)), 0)), JLArray(x), Inf_)), prox(NormL1(T(1)), x, Inf_))
end

end
