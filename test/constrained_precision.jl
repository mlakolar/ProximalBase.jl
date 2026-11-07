module ConstrainedPrecisionTest

# Accuracy and failure behaviour of the multiplier search of Constrained on a hyperplane, a halfspace and a
# hyperslab (issue 151), against an independent BigFloat oracle of each call's effective problem: model
# precision W from the destination, the argument and the data (T3), steps, levels and bounds rounded once
# to W, then exact arithmetic. The oracle follows the mathematics (a root of a nonincreasing piecewise-affine
# function, bracketed and bisected), not the implementation, and calls no internal helper of the package.
#
# The gates are the pre-registered ones: ‖y − z*‖/‖z*‖ ≤ 16 eps of the output type on the recoverable
# fixtures and the small decision cases, ≤ 8 eps on the ordinary controls. The measured error, in eps, is in
# the name of each testset. The n = 20 000 and 100 000 grids are in constrained_precision_large.jl.
# Nothing here is marked broken: a case the package gets wrong fails.

using Test
using ProximalBase
using Random

using ..TestUtils

# ---------------------------------------------------------------------------------------------------
# BEGIN SHARED. The text between BEGIN SHARED and END SHARED is identical in constrained_precision.jl
# and constrained_precision_large.jl (a tier file cannot share code with another one without a helper
# in tiers.toml); constrained_precision.jl checks that the two copies agree.
#
# The oracle. A call of Constrained(f, set) on x with steps γ solves, in the model precision W (T3),
#     minimize  f(z) + ‖z − x‖²_{1/γ} / 2   subject to   lb ≤ ⟨a, z⟩ ≤ ub     (lb = ub: a hyperplane,
#     lb = -Inf: a halfspace)
# for a separable f whose scalar proximal map is  z = clamp(κ·soft(v, θ), lo, hi),  v = x − γ ν a.
# The parameters of the call (steps, levels, bounds) are rounded once to W, the combined thresholds
# θ = fl_W(γ·λ₁) and t = fl_W(γ·λ₂) are formed in W, and κ = 1/(1 + t); data (x, a, offsets) are
# exact. Everything after that is exact arithmetic in 256-bit BigFloat. The multiplier ν* is the root
# of the nonincreasing σ(ν) = Σᵢ aᵢ zᵢ(ν): the case analysis of a slab is σ(0) < lb (root of σ − lb,
# ν < 0), σ(0) > ub (root of σ − ub, ν > 0), else ν = 0. The root is found by bisection on the exact
# piecewise-affine σ: every entry whose affine piece is the same at both ends of the bracket is summed
# once, so a pass costs O(live entries).
# ---------------------------------------------------------------------------------------------------

soft(v, θ) = v > θ ? v - θ : (v < -θ ? v + θ : zero(v))

# the effective problem: arrays hold exactly W-representable values (or the data), widened to Float64
struct Model
  W::DataType
  x::Vector{Float64}
  a::Vector{Float64}
  γ::Vector{Float64}
  θ::Vector{Float64}
  t::Vector{Float64}
  λ1::Float64
  λ2::Float64
  lo::Float64
  hi::Float64
  lb::Float64
  ub::Float64
end

_narrow(λ, ::Type{W}) where {W} = λ isa Integer || sizeof(typeof(λ)) <= sizeof(W)

# x, a, lb, ub are data; γ, λ1, λ2, lo, hi are parameters rounded once to W. A level wider than W
# would need the combined-threshold convention of T3, which no fixture here relies on.
function model(::Type{W}, x, a, γ, lb, ub; λ1 = 0, λ2 = 0, lo = -Inf, hi = Inf) where {W<:AbstractFloat}
  _narrow(λ1, W) && _narrow(λ2, W) || throw(ArgumentError("a level wider than the model precision"))
  n = length(x)
  γW = γ isa Real ? fill(W(γ), n) : W.(γ)
  l1, l2 = W(λ1), W(λ2)
  Model(W, Float64.(x), a isa Real ? fill(Float64(a), n) : Float64.(a), Float64.(γW),
        Float64.(W.(γW .* l1)), Float64.(W.(γW .* l2)), Float64(l1), Float64(l2),
        Float64(W(lo)), Float64(W(hi)), Float64(lb), Float64(ub))
end

# the BigFloat arrays and scalar maps of a model; build and use inside setprecision
struct Big
  X::Vector{BigFloat}
  A::Vector{BigFloat}
  G::Vector{BigFloat}
  Θ::Vector{BigFloat}
  K::Vector{BigFloat}
  lo::BigFloat
  hi::BigFloat
end
bigmodel(m::Model) = Big(BigFloat.(m.x), BigFloat.(m.a), BigFloat.(m.γ), BigFloat.(m.θ),
                    [inv(1 + BigFloat(t)) for t in m.t], BigFloat(m.lo), BigFloat(m.hi))
vof(B::Big, i, ν) = B.X[i] - B.G[i] * B.A[i] * ν
zof(B::Big, i, v) = clamp(B.K[i] * soft(v, B.Θ[i]), B.lo, B.hi)
σof(B::Big, ν) = (s = zero(BigFloat); for i in eachindex(B.X); iszero(B.A[i]) || (s += B.A[i] * zof(B, i, vof(B, i, ν))); end; s)

# which affine piece of the scalar map the argument v lies on: a code 0:8 of (branch of soft, clamp class)
function region(B::Big, i, v)
  θ = B.Θ[i]
  s = v > θ ? 1 : (v < -θ ? -1 : 0)
  z = B.K[i] * (s == 1 ? v - θ : (s == -1 ? v + θ : zero(v)))
  c = z < B.lo ? -1 : (z > B.hi ? 1 : 0)
  3 * (s + 1) + (c + 1)
end
# z = m v + q on that piece
function affine(B::Big, i, code)
  s, c = div(code, 3) - 1, rem(code, 3) - 1
  c == -1 && return (zero(B.lo), B.lo)
  c == 1 && return (zero(B.hi), B.hi)
  s == 0 && return (zero(B.lo), zero(B.lo))
  (B.K[i], -s * B.K[i] * B.Θ[i])
end

# the entries in idx that stay on one piece over [l, h] contribute acc + slope·ν; the others are returned
function classify(B::Big, idx, l, h)
  acc, slope, S = zero(BigFloat), zero(BigFloat), Int[]
  for i in idx
    iszero(B.A[i]) && continue
    r1, r2 = region(B, i, vof(B, i, l)), region(B, i, vof(B, i, h))
    if r1 == r2
      mi, qi = affine(B, i, r1)
      acc += B.A[i] * ((iszero(mi) ? mi : mi * B.X[i]) + qi)
      slope -= mi * B.A[i]^2 * B.G[i]
    else
      push!(S, i)
    end
  end
  acc, slope, S
end
ψof(B::Big, acc, slope, S, target, ν) =
  acc + slope * ν - target + sum((B.A[i] * zof(B, i, vof(B, i, ν)) for i in S); init = zero(BigFloat))

# the root of σ(ν) = target near the Float64 root of the same σ
function rootof(m::Model, B::Big, target)
  n = length(m.x)
  Kf = [1 / (1 + t) for t in m.t]
  σf(ν) = (s = 0.0; for i in 1:n; a = m.a[i]; iszero(a) && continue;
           s += a * clamp(Kf[i] * soft(m.x[i] - m.γ[i] * a * ν, m.θ[i]), m.lo, m.hi); end; s)
  f0 = σf(0.0) - target
  ν0 = 0.0
  if f0 != 0
    l, h, k = 0.0, (f0 > 0 ? 1.0 : -1.0), 0
    while sign(σf(h) - target) == sign(f0)
      l, h, k = h, 2h, k + 1
      k > 600 && error("oracle: no bracket (infeasible?)")
    end
    for _ in 1:200
      mid = (l + h) / 2
      (mid == l || mid == h) && break
      sign(σf(mid) - target) == sign(f0) ? (l = mid) : (h = mid)
    end
    ν0 = (l + h) / 2
  end
  δ = ldexp(1 + abs(ν0), -30)
  for _ in 1:14
    l, h = BigFloat(ν0) - δ, BigFloat(ν0) + δ
    acc, slope, S = classify(B, 1:n, l, h)
    if ψof(B, acc, slope, S, target, l) >= 0 && ψof(B, acc, slope, S, target, h) <= 0
      for _ in 1:2000
        if isempty(S)
          iszero(slope) && return (l + h) / 2
          return clamp((target - acc) / slope, l, h)
        end
        mid = (l + h) / 2
        p = ψof(B, acc, slope, S, target, mid)
        iszero(p) && return mid
        p > 0 ? (l = mid) : (h = mid)
        h - l <= ldexp(1 + abs(l), -200) && return (l + h) / 2
        a2, s2, S = classify(B, S, l, h)
        acc += a2
        slope += s2
      end
      error("oracle: no convergence")
    end
    δ *= 8
  end
  error("oracle: no bracket (infeasible?)")
end

# (z*, ν*, face) of the call, z* exact (256 bits); face is :inactive, :lower or :upper
function oracle(m::Model)
  setprecision(BigFloat, 256) do
    B = bigmodel(m)
    s0 = σof(B, zero(BigFloat))
    lb, ub = BigFloat(m.lb), BigFloat(m.ub)
    face, ν = s0 < lb ? (:lower, rootof(m, B, lb)) : s0 > ub ? (:upper, rootof(m, B, ub)) : (:inactive, zero(BigFloat))
    [zof(B, i, vof(B, i, ν)) for i in eachindex(B.X)], ν, face
  end
end
# σ(ν) and z(ν) of the model at any multiplier
sigma(m::Model, ν) = setprecision(() -> σof(bigmodel(m), BigFloat(ν)), BigFloat, 256)
zat(m::Model, ν) = setprecision(BigFloat, 256) do
  B = bigmodel(m)
  [zof(B, i, vof(B, i, BigFloat(ν))) for i in eachindex(B.X)]
end
# ⟨a, y⟩ of a stored output, exactly
dota(m::Model, y) = setprecision(BigFloat, 256) do
  s = zero(BigFloat)
  for i in eachindex(y)
    s += BigFloat(m.a[i]) * BigFloat(y[i])
  end
  s
end
# f(y) of the separable part, from the rounded levels (0 for an indicator)
fval(m::Model, y) = setprecision(BigFloat, 256) do
  s1, s2, l1, l2 = zero(BigFloat), zero(BigFloat), BigFloat(m.λ1), BigFloat(m.λ2)
  for v in y
    s1 += abs(BigFloat(v))
    s2 += BigFloat(v)^2
  end
  l1 * s1 + l2 * s2 / 2
end

# ‖y − z‖/‖z‖ in units of eps(T); with γ, in the norm of Diagonal(1 ./ γ). z = 0 asks for y = 0.
function relerr(y, z::Vector{BigFloat}, ::Type{T}; γ = nothing) where {T}
  setprecision(BigFloat, 256) do
    num, den = zero(BigFloat), zero(BigFloat)
    for i in eachindex(z)
      w = γ === nothing ? one(BigFloat) : 1 / BigFloat(γ[i])
      d = BigFloat(y[i]) - z[i]
      num += w * d * d
      den += w * z[i] * z[i]
    end
    iszero(den) && return iszero(num) ? 0.0 : Inf
    Float64(sqrt(num / den) / eps(T))
  end
end
# max |yᵢ − zᵢ| in units of the spacing of T at zᵢ
function maxulps(y, z::Vector{BigFloat}, ::Type{T}) where {T}
  setprecision(BigFloat, 256) do
    worst = 0.0
    for i in eachindex(z)
      worst = max(worst, Float64(abs(BigFloat(y[i]) - z[i]) / max(BigFloat(eps(T(z[i]))), BigFloat(floatmin(T)))))
    end
    worst
  end
end

# a call and its effective problem(s): mdls(x) lists the models the call may legitimately be solving
# (one, except where the model precision of the call is not decided: then the call is judged against each)
struct Case
  name::String
  T::DataType
  mk::Function
  x::Vector
  γ::Any
  mdls::Function
end

const SENTINEL = -123.25
const GATE_CONTROL = 8      # pre-registered, in eps of the output type: ‖y − z*‖/‖z*‖ on the ordinary controls
const GATE_FIXTURE = 16     # the same on the recoverable fixtures and the small decision cases
# The ordinary controls in Float64 have their own knob: the Float64 search keeps its residual acceptance
# (the plan certifies the Float32 model only), and the base is 8.1 to 12.4 eps(Float64) wrong on the
# per-entry ElasticNet controls at n = 20 000 to 100 000, above the pre-registered 8. It is 8 here until
# the maintainer decides between a Float64 certification and a wider Float64 gate.
const GATE_CONTROL_F64 = 8
control_gate(::Type{T}) where {T} = T === Float32 ? GATE_CONTROL : GATE_CONTROL_F64

# ‖y − z‖/‖z‖ in units of eps(T), and in the norm of Diagonal(1 ./ γ) (NaN without γ); z = 0 asks for y = 0
function errs(y, z::Vector{BigFloat}, ::Type{T}, γ = nothing) where {T}
  setprecision(BigFloat, 256) do
    num, den, numw, denw = zero(BigFloat), zero(BigFloat), zero(BigFloat), zero(BigFloat)
    for i in eachindex(z)
      d = BigFloat(y[i]) - z[i]
      num += d * d
      den += z[i] * z[i]
      if γ !== nothing
        w = 1 / BigFloat(γ[i])
        numw += w * d * d
        denw += w * z[i] * z[i]
      end
    end
    ratio(p, q) = iszero(q) ? (iszero(p) ? 0.0 : Inf) : Float64(sqrt(p / q) / eps(T))
    ratio(num, den), γ === nothing ? NaN : ratio(numw, denw)
  end
end

# the oracle points of a call at x, one per model the call may be solving
refs(case::Case, x) = [(m, oracle(m)...) for m in case.mdls(x)]

# one call on a function object, judged against the oracle; err holds an exception the call raised. The
# oracle points of x may be passed in (ref), as the repeated calls of a sequence share them.
function measure(case::Case, c; x = case.x, op::Symbol = :prox, alias::Bool = false, ref = nothing)
  T = case.T
  y = alias ? copy(x) : fill(T(SENTINEL), length(x))
  val = nothing
  err = nothing
  try
    src = alias ? y : x
    if op === :prox
      prox!(y, c, src, case.γ)
    else
      val = value_and_prox!(y, c, src, case.γ)
    end
  catch e
    err = e
  end
  err === nothing || return (; err, e2 = Inf, ew = Inf, y, val, z = BigFloat[], ν = BigFloat(NaN), face = :none, m = nothing)
  best = nothing
  for (m, z, ν, face) in (ref === nothing ? refs(case, x) : ref)
    e2, ew = errs(y, z, T, case.γ isa AbstractVector ? m.γ : nothing)
    isnan(ew) && (ew = e2)
    (best === nothing || e2 < best.e2) && (best = (; err, e2, ew, y, val, z, ν, face, m))
  end
  best
end

# the measured error in the name of a testset, so that a passing run shows its margin
function expect(label, r, gate; weighted = false)
  @testset "$label: $(r.err === nothing ? round(r.e2; digits = 2) : nameof(typeof(r.err))) eps" begin
    @test r.err === nothing
    @test r.e2 <= gate
    weighted && @test r.ew <= gate
  end
end

# value_and_prox!: the value is f at the stored output, from an independent evaluation of f. The sum of
# n terms has a roundoff of the order of eps·sqrt(n), so the tolerance is (8 + 2 sqrt(n)) eps relative.
function check_value(label, r, case)
  r.err === nothing || return
  n = length(r.y)
  ref = fval(r.m, r.y)
  tol = (8 + 2 * sqrt(n)) * eps(case.T)
  @testset "$label: value" begin
    @test r.val isa Real
    ok = setprecision(BigFloat, 256) do
      iszero(ref) ? iszero(r.val) : abs(BigFloat(r.val) - ref) <= BigFloat(tol) * ref
    end
    @test ok
  end
end

# a call sequence: cold, warm, changed argument, stale multipliers and slopes, fused value, aliasing.
# full = false keeps the cold, warm, one stale multiplier, one stale slope and the fused cold call.
function sequence(case::Case, gate; full::Bool = true)
  w = case.γ isa AbstractVector
  c = case.mk()
  ref = refs(case, case.x)
  expect("$(case.name) cold", measure(case, c; ref), gate; weighted = w)
  expect("$(case.name) warm", measure(case, c; ref), gate; weighted = w)
  if full
    x2 = case.x .+ case.T(0.02) .* randn(Xoshiro(7), case.T, length(case.x))
    expect("$(case.name) changed x", measure(case, c; x = x2), gate; weighted = w)
  end
  # stale multipliers, left by a very different problem or by a call in another precision
  for last in (full ? (1e6, -1e6, NaN) : (1e6,))
    c.last[] = last
    expect("$(case.name) stale multiplier $last", measure(case, c; ref), gate; weighted = w)
  end
  # stale slopes (the slope of the last secant is a seed too): 100 times too flat and too steep, and NaN
  cs = case.mk()
  measure(case, cs; ref)
  s0 = cs.slope[]
  for f in (full ? (1e-2, 1e2, NaN) : (1e-2,))
    cs.slope[] = s0 * f
    expect("$(case.name) stale slope × $f", measure(case, cs; ref), gate; weighted = w)
  end
  r = measure(case, case.mk(); op = :vprox, ref)
  expect("$(case.name) value_and_prox! cold", r, gate; weighted = w)
  check_value("$(case.name) value_and_prox! cold", r, case)
  if full
    r = measure(case, c; op = :vprox, ref)
    expect("$(case.name) value_and_prox! warm", r, gate; weighted = w)
    check_value("$(case.name) value_and_prox! warm", r, case)
    expect("$(case.name) prox!(x, c, x)", measure(case, case.mk(); alias = true, ref), gate; weighted = w)
    r = measure(case, case.mk(); op = :vprox, alias = true, ref)
    expect("$(case.name) value_and_prox!(x, c, x)", r, gate; weighted = w)
    check_value("$(case.name) value_and_prox!(x, c, x)", r, case)
  end
  nothing
end

# the inputs of the n = 20 000 fixtures of issue 151: x = randn(Float32), a = randn with every 7th entry
# zero, steps 0.2 + rand; the offset sits Σ min(a, 0) + 1 above the lowest value of ⟨a, z⟩ on [0, 1]ⁿ
function flat_data(n::Integer = 20_000; seed = 151)
  rng = Xoshiro(seed)
  x = randn(rng, Float32, n)
  a64 = randn(rng, n)
  a64[1:7:end] .= 0
  γ64 = 0.2 .+ rand(rng, n)
  a32 = Float32.(a64)
  (; n, x, a64, a32, γ64, γ32 = Float32.(γ64),
     b64 = sum(min.(a64, 0)) + 1, b32 = Float32(sum(min.(Float64.(a32), 0)) + 1),
     b32w = sum(min.(Float64.(a32), 0)) + 1)
end

const CONTROL_KINDS = (:capped, :capped_half, :capped_slab_lo, :capped_slab_hi, :lasso, :enet_simplex,
                       :enet_strict, :posl1, :signed)

# the ordinary problems: all data and parameters in T, scalar or per-entry steps; box builds the box
# of the capped simplex (a wrapper of IndBox for the tests of a function without a recovery hook)
function control(kind::Symbol, ::Type{T}, n::Integer, steps::Symbol; box = IndBox) where {T<:AbstractFloat}
  rng = Xoshiro(10_007 * n + 31 * findfirst(==(kind), CONTROL_KINDS) + (T === Float32 ? 1 : 2))
  x = randn(rng, T, n)
  a = randn(rng, T, n)
  a[1:7:end] .= 0
  apos = abs.(randn(rng, T, n)) .+ T(0.1)
  γ = steps === :scalar ? T(0.5) : T(0.2) .+ rand(rng, T, n)
  h = T(n) / 2
  q = T(n) / 4
  r = T(n) / 10
  mk, mdl = if kind === :capped
    () -> Constrained(box(T(0), T(1)), IndHyperplane(T(1), h)),
    x -> [model(T, x, 1, γ, h, h; lo = 0, hi = 1)]
  elseif kind === :capped_half
    () -> Constrained(box(T(0), T(1)), IndHalfspace(T(1), q)),
    x -> [model(T, x, 1, γ, -Inf, q; lo = 0, hi = 1)]
  elseif kind === :capped_slab_hi
    () -> Constrained(box(T(0), T(1)), IndHyperslab(T(1), T(n) / 8, q)),
    x -> [model(T, x, 1, γ, T(n) / 8, q; lo = 0, hi = 1)]
  elseif kind === :capped_slab_lo
    () -> Constrained(box(T(0), T(1)), IndHyperslab(T(1), h, T(3n) / 4)),
    x -> [model(T, x, 1, γ, h, T(3n) / 4; lo = 0, hi = 1)]
  elseif kind === :lasso
    () -> Constrained(NormL1(T(0.3)), IndHyperplane(T(1), T(0))),
    x -> [model(T, x, 1, γ, 0, 0; λ1 = T(0.3))]
  elseif kind === :enet_simplex
    # the normal of IndSimplex is the integer 1: whether the call is a model in T or in Float64 is
    # decided by the package, so the call is judged against both
    () -> Constrained(ElasticNet(T(0.3), T(0.4)), IndSimplex(r)),
    x -> unique(m -> m.W, [model(T, x, 1, γ, r, r; λ1 = T(0.3), λ2 = T(0.4), lo = 0),
                           model(Float64, x, 1, γ, r, r; λ1 = T(0.3), λ2 = T(0.4), lo = 0)])
  elseif kind === :enet_strict
    () -> Constrained(Constrained(ElasticNet(T(0.3), T(0.4)), IndNonnegative()), IndHyperplane(T(1), r)),
    x -> [model(T, x, 1, γ, r, r; λ1 = T(0.3), λ2 = T(0.4), lo = 0)]
  elseif kind === :posl1
    () -> Constrained(NormL1Nonneg(T(0.1)), IndHyperplane(apos, T(0.5))),
    x -> [model(T, x, apos, γ, T(0.5), T(0.5); λ1 = T(0.1), lo = 0)]
  elseif kind === :signed
    () -> Constrained(NormL1(T(0.1)), IndHyperplane(a, T(0.5))),
    x -> [model(T, x, a, γ, T(0.5), T(0.5); λ1 = T(0.1))]
  else
    throw(ArgumentError("unknown control $kind"))
  end
  Case("$kind $T n=$n $steps", T, mk, x, γ, mdl)
end

# ---------------------------------------------------------------------------------------------------
# the fixtures of issue 151 (n = 20 000, seed 151): x = randn(Float32), a = randn with every 7th entry 0,
# steps 0.7 or 0.2 + rand. Each model precision is its own family:
#   (a)  all Float32: x, y, normal, offset, steps, levels
#   (b1) Float32 x and y, Float64 normal and offset: the model is Float64 (the normal and the offset are data)
#   (b2) Float32 x and y, Float32 normal, Float64 offset: Float64 again
#   (c)  all Float64
#   (d)  all Float32 data with Float64 steps: a step is a parameter, rounded once to Float32
# ---------------------------------------------------------------------------------------------------

# signed box: IndBox(0, 1) on IndHyperplane(a, Σ min(a, 0) + 1)
function signed_box(d, fam::Symbol, steps::Symbol)
  scalar = steps === :scalar
  name = "signed box ($fam) $steps"
  box = IndBox(0, 1)
  if fam === :a
    γ = scalar ? 0.7f0 : d.γ32
    Case(name, Float32, () -> Constrained(box, IndHyperplane(d.a32, d.b32)), d.x, γ,
         x -> [model(Float32, x, d.a32, γ, d.b32, d.b32; lo = 0, hi = 1)])
  elseif fam === :b1
    γ = scalar ? 0.7 : d.γ64
    Case(name, Float32, () -> Constrained(box, IndHyperplane(d.a64, d.b64)), d.x, γ,
         x -> [model(Float64, x, d.a64, γ, d.b64, d.b64; lo = 0, hi = 1)])
  elseif fam === :b2
    γ = scalar ? 0.7 : d.γ64
    Case(name, Float32, () -> Constrained(box, IndHyperplane(d.a32, d.b32w)), d.x, γ,
         x -> [model(Float64, x, d.a32, γ, d.b32w, d.b32w; lo = 0, hi = 1)])
  elseif fam === :c
    γ = scalar ? 0.7 : d.γ64
    Case(name, Float64, () -> Constrained(box, IndHyperplane(d.a64, d.b64)), Float64.(d.x), γ,
         x -> [model(Float64, x, d.a64, γ, d.b64, d.b64; lo = 0, hi = 1)])
  else  # :d
    γ = scalar ? 0.7 : d.γ64
    Case(name, Float32, () -> Constrained(box, IndHyperplane(d.a32, d.b32)), d.x, γ,
         x -> [model(Float32, x, d.a32, γ, d.b32, d.b32; lo = 0, hi = 1)])
  end
end

# the ℓ1 penalty on a box under the constraint -Σ z = n - 5, normal -1 and offset n - 5
function l1_box(d, fam::Symbol, steps::Symbol)
  scalar = steps === :scalar
  n = d.n
  name = "L1 on a box ($fam) $steps"
  if fam === :a
    γ = scalar ? 0.7f0 : d.γ32
    Case(name, Float32, () -> Constrained(Constrained(NormL1(0.3f0), IndBox(-1f0, 1f0)), IndHyperplane(-1f0, Float32(n - 5))), d.x, γ,
         x -> [model(Float32, x, -1, γ, n - 5, n - 5; λ1 = 0.3f0, lo = -1, hi = 1)])
  elseif fam === :b
    γ = scalar ? 0.7 : d.γ64
    Case(name, Float32, () -> Constrained(Constrained(NormL1(0.3), IndBox(-1.0, 1.0)), IndHyperplane(-1.0, Float64(n - 5))), d.x, γ,
         x -> [model(Float64, x, -1, γ, n - 5, n - 5; λ1 = 0.3, lo = -1, hi = 1)])
  else  # :c
    γ = scalar ? 0.7 : d.γ64
    Case(name, Float64, () -> Constrained(Constrained(NormL1(0.3), IndBox(-1.0, 1.0)), IndHyperplane(-1.0, Float64(n - 5))), Float64.(d.x), γ,
         x -> [model(Float64, x, -1, γ, n - 5, n - 5; λ1 = 0.3, lo = -1, hi = 1)])
  end
end

# END SHARED

unchanged(y, T) = all(==(T(SENTINEL)), y)
raised(f) = try
  f()
  nothing
catch e
  e
end

# ---------------------------------------------------------------------------------------------------
# the two copies of the shared text agree
# ---------------------------------------------------------------------------------------------------
@testset "the shared oracle text is the same in constrained_precision_large.jl" begin
  shared(path) = (s = read(path, String); s[first(findfirst("# BEGIN SHARED", s)):last(findfirst("# END SHARED", s))])
  @test shared(@__FILE__) == shared(joinpath(@__DIR__, "constrained_precision_large.jl"))
end

# ---------------------------------------------------------------------------------------------------
# the oracle against closed forms (exact to 2^-230), so that every comparison below rests on it
# ---------------------------------------------------------------------------------------------------
# a root at a kink is found to a bracket of 2^-200, so a solution with an entry on its threshold is
# exact to about 2^-190; the others to 2^-230
exact(z, ref; bits = 230) = setprecision(BigFloat, 256) do
  maximum(abs.(z .- ref)) <= ldexp(BigFloat(1), -bits)
end

@testset "oracle against closed forms" begin
  big256(f) = setprecision(f, BigFloat, 256)
  # zero-sum lasso, x = (4, 1, 0, -1), λ = 1/2, unit step: ν = 7/6, z = (7/3, 0, -2/3, -5/3)
  m = model(Float64, [4.0, 1, 0, -1], 1, 1.0, 0, 0; λ1 = 0.5)
  z, ν, face = oracle(m)
  @test exact(z, big256(() -> BigFloat[7, 0, -2, -5] ./ [3, 1, 3, 3]))
  @test big256(() -> abs(ν - BigFloat(7) / 6) < ldexp(BigFloat(1), -230))
  # capped simplex, x = (2, 1/2, 1/2, -1) with sum 5/2: ν = -1/4, z = (1, 3/4, 3/4, 0)
  z, ν, face = oracle(model(Float64, [2.0, 0.5, 0.5, -1], 1, 1.0, 2.5, 2.5; lo = 0, hi = 1))
  @test exact(z, BigFloat[1, 0.75, 0.75, 0]) && ν == -0.25 && face === :lower
  # elastic net with sum 1/2 on x = (2, 1), λ = (1/2, 1): ν = 1/2, z = (1/2, 0)
  z, ν, face = oracle(model(Float64, [2.0, 1], 1, 1.0, 0.5, 0.5; λ1 = 0.5, λ2 = 1, lo = 0))
  @test exact(z, BigFloat[0.5, 0]; bits = 190) && face === :upper
  @test setprecision(() -> abs(ν - BigFloat(1) / 2) <= ldexp(BigFloat(1), -190), BigFloat, 256)
  # a slab holding σ(0), and the upper and lower bounds, are the closed-form projections
  rng = Xoshiro(1)
  for W in (Float32, Float64), per in (false, true)
    n = 7
    x, a = randn(rng, W, n), randn(rng, W, n)
    a[3] = 0
    γ = per ? W(0.2) .+ rand(rng, W, n) : W(0.7)
    for (lb, ub, face0) in ((W(1.5), W(1.5), nothing), (-Inf, W(-0.5), nothing), (W(0.25), W(0.5), nothing), (-Inf, W(3), :inactive))
      z, ν, face = oracle(model(W, x, a, γ, lb, ub))
      ref = big256() do
        A, X, Γ = BigFloat.(a), BigFloat.(x), BigFloat.(γ isa Real ? fill(γ, n) : γ)
        s0 = sum(A .* X)
        target = s0 < lb ? BigFloat(lb) : s0 > ub ? BigFloat(ub) : s0
        X .- (sum(A .* X) - target) / sum(Γ .* A .^ 2) .* Γ .* A
      end
      @test exact(z, ref)
      face0 === nothing || @test face === face0
    end
    # a ridge: z = κ (x - γ ν a) with κ = 1/(1 + γλ)
    λ = W(0.5)
    z, ν, face = oracle(model(W, x, a, γ, 0.5, 0.5; λ2 = λ))
    ref = big256() do
      A, X, Γ = BigFloat.(a), BigFloat.(x), BigFloat.(γ isa Real ? fill(γ, n) : γ)
      κ = [inv(1 + BigFloat(W(Γ[i] * λ))) for i in 1:n]
      νr = (sum(A .* κ .* X) - BigFloat(1) / 2) / sum(κ .* Γ .* A .^ 2)
      κ .* (X .- νr .* Γ .* A)
    end
    @test exact(z, ref)
  end
  # the exact residual of a hard fixture, and the rounded solution of the existing near-face case
  d = flat_data(2_000)
  m = model(Float32, d.x, d.a32, 0.7f0, d.b32, d.b32; lo = 0, hi = 1)
  z, ν, face = oracle(m)
  @test face === :upper
  @test big256(() -> abs(dota(m, z) - BigFloat(d.b32)) < ldexp(BigFloat(1), -200))
  m = model(Float32, Float32[1, 1, -1], 1, 1f0, 0, 2f0 - ldexp(1f0, -22); lo = 0, hi = 1)
  z, ν, face = oracle(m)
  @test exact(z, BigFloat[1 - ldexp(1.0, -23), 1 - ldexp(1.0, -23), 0])
end

# the fixture of issue 151 at sizes that fit this tier: without certification the Float32 search is 15 to
# 475 eps(Float32) wrong on these seeds (n = 1000 with seeds 151 and 152, n = 4000 with seed 153), the
# error depending on the reduction order of the host
@testset "signed box on a hyperplane at n = 1000 and 4000" begin
  for (n, seed) in ((1000, 151), (1000, 152), (4000, 153)), fam in (:a, :b1, :d), steps in (:scalar, :entry)
    case = signed_box(flat_data(n; seed), fam, steps)
    sequence(Case("n=$n seed=$seed " * case.name, case.T, case.mk, case.x, case.γ, case.mdls), GATE_FIXTURE; full = false)
  end
end

@testset "controls" begin
  # n = 4095 has no variable fixing; from 4096 a cold call may fix entries; 4097 is odd
  for n in (256, 4095, 4096, 4097), T in (Float32, Float64), steps in (:scalar, :entry)
    @testset "n = $n $T $steps" begin
      for kind in CONTROL_KINDS
        # the halfspace and the slabs need only a large and a small size
        kind in (:capped_half, :capped_slab_lo, :capped_slab_hi) && !(n in (256, 4097)) && continue
        sequence(control(kind, T, n, steps), control_gate(T))
      end
    end
  end
end

# ---------------------------------------------------------------------------------------------------
# decisions: active and inactive halfspaces, both hyperslab bounds and the interior, from large margins
# to a few ulps of the exact σ(0) (the sum at the unconstrained proximal point)
# ---------------------------------------------------------------------------------------------------
shiftb(b0::T, δ::Integer) where {T} = nextfloat(b0, δ)
shiftb(b0::T, δ::AbstractFloat) where {T} = T(b0 + δ)

const DEC_X = Dict(:box => [1.3, 0.35, 0.3, 0.7, -0.4, 0.15, 0.5, 0.2],
                   :l1box => [0.9, -0.4, 0.3, 1.7, -1.8, 0.15, 0.5, -0.2],
                   :l1 => [0.9, -0.4, 0.3, 1.7, -1.8, 0.15, 0.5, -0.2])
const DEC_A = [1, -1.5, 0.5, 2, -0.75, 0, 1, -1]
const DEC_G = [0.5, 0.9, 0.3, 1.1, 0.7, 0.2, 0.6, 0.8]

function decision_function(fam::Symbol, ::Type{T}) where {T}
  fam === :box && return IndBox(T(0), T(1)), T(0), T(1), T(0)
  fam === :l1box && return Constrained(NormL1(T(0.3)), IndBox(T(-1), T(1))), T(-1), T(1), T(0.3)
  NormL1(T(0.3)), T(-Inf), T(Inf), T(0.3)
end

@testset "decisions near the faces" begin
  margins = (-0.5, -2.0^-8, -4096, -64, -8, -2, -1, 0, 1, 2, 8, 64, 4096, 2.0^-8, 0.5)
  for T in (Float32, Float64), fam in (:box, :l1box, :l1), steps in (:scalar, :entry)
    x, a = T.(DEC_X[fam]), T.(DEC_A)
    γ = steps === :scalar ? T(0.6) : T.(DEC_G)
    f, lo, hi, λ1 = decision_function(fam, T)
    s0 = sigma(model(T, x, a, γ, 0, 0; λ1, lo, hi), 0)
    b0 = T(s0)
    # each design: the set, its offsets, and which face the exact point must select for a large margin
    designs = [
      (:plane, δ -> (b = shiftb(b0, δ); (IndHyperplane(a, b), b, b, δ isa AbstractFloat ? (δ > 0 ? :lower : :upper) : nothing))),
      (:half, δ -> (b = shiftb(b0, δ); (IndHalfspace(a, b), -Inf, b, δ isa AbstractFloat ? (δ > 0 ? :inactive : :upper) : nothing))),
      (:slab_lo, δ -> (l = shiftb(b0, δ); u = shiftb(b0, 5.0); (IndHyperslab(a, l, u), l, u, δ isa AbstractFloat ? (δ > 0 ? :lower : :inactive) : nothing))),
      (:slab_hi, δ -> (u = shiftb(b0, δ); l = shiftb(b0, -5.0); (IndHyperslab(a, l, u), l, u, δ isa AbstractFloat ? (δ < 0 ? :upper : :inactive) : nothing))),
    ]
    @testset "$T $fam $steps" begin
      for (kind, design) in designs, δ in margins
        set, lb, ub, face = design(δ)
        case = Case("$kind δ=$δ", T, () -> Constrained(f, set), x, γ, x -> [model(T, x, a, γ, lb, ub; λ1, lo, hi)])
        r = measure(case, case.mk())
        expect("$T $fam $steps $(case.name)", r, GATE_FIXTURE; weighted = steps === :entry)
        if r.err === nothing && face !== nothing && abs(δ) >= 2.0^-8
          # a large margin: the exact face is the designed one, and the output is on it
          d0 = relerr(r.y, zat(r.m, 0), T)
          @test r.face === face
          @test face === :inactive ? d0 <= GATE_FIXTURE : d0 > 64
        end
      end
      # a slab around σ(0): the faces are within a few ulps of each other, or both far
      for k in (0, 1, 2, 8, 64)
        l, u = shiftb(b0, -k), shiftb(b0, k)
        case = Case("slab ±$k ulps", T, () -> Constrained(f, IndHyperslab(a, l, u)), x, γ, x -> [model(T, x, a, γ, l, u; λ1, lo, hi)])
        expect("$T $fam $steps $(case.name)", measure(case, case.mk()), GATE_FIXTURE)
      end
      l, u = shiftb(b0, -0.5), shiftb(b0, 0.5)
      case = Case("slab ±0.5", T, () -> Constrained(f, IndHyperslab(a, l, u)), x, γ, x -> [model(T, x, a, γ, l, u; λ1, lo, hi)])
      r = measure(case, case.mk())
      expect("$T $fam $steps $(case.name)", r, GATE_FIXTURE)
      r.err === nothing && @test r.face === :inactive
    end
  end
end

# ---------------------------------------------------------------------------------------------------
# exact and special cases
# ---------------------------------------------------------------------------------------------------
@testset "near-face cases: box [0, 1], b = 2 - 2^-j, x = (1, 1, -1)" begin
  # j = p - 2 is the existing case of the conformance suite (linear_sets.certify): the two free entries are
  # 1 - 2^-(j+1) exactly, one step of eps below the representable neighbour of 1, and the third is fixed at 0
  for T in (Float32, Float64), kind in (:plane, :half, :slab), steps in (:scalar, :entry), j in precision(T) .- (2, 3, 4, 5)
    b = T(2) - ldexp(one(T), -j)
    set = kind === :plane ? IndHyperplane(T(1), b) : kind === :half ? IndHalfspace(T(1), b) : IndHyperslab(T(1), T(0), b)
    lb, ub = kind === :plane ? (b, b) : kind === :half ? (-Inf, b) : (T(0), b)
    γ = steps === :scalar ? one(T) : ones(T, 3)
    case = Case("$T $kind $steps j=$j", T, () -> Constrained(IndBox(T(0), T(1)), set), T[1, 1, -1], γ,
                x -> [model(T, x, 1, γ, lb, ub; lo = 0, hi = 1)])
    r = measure(case, case.mk())
    z1 = 1 - ldexp(1.0, -(j + 1))
    @testset "$(case.name)" begin
      @test r.err === nothing
      if r.err === nothing
        @test exact(r.z, BigFloat[z1, z1, 0])                 # the oracle's point
        # within one ulp of it, componentwise, in Float32, which the search certifies; the Float64 search keeps its
        # residual acceptance (G5), so a Float64 point is only required to be within 16 ulps
        @test maxulps(r.y, r.z, T) <= (T === Float32 ? 1 : 16)
        @test r.y[3] == 0
        T === Float32 && @test r.y[1] != 1 && r.y[2] != 1
      end
    end
  end
end

@testset "an iterate already on the constraint is returned, never rejected" begin
  rng = Xoshiro(3)
  for T in (Float32, Float64), steps in (:scalar, :entry), fam in (:box, :l1box)
    # dyadic data: every operation of the pass at ν = 0 is exact
    n = 12
    x = fam === :box ? T.(rand(rng, 0:16, n)) ./ 16 : T.(rand(rng, -32:32, n)) ./ 16
    γ = steps === :scalar ? T(0.5) : T[1, 0.5, 0.25, 2, 0.5, 1, 0.25, 2, 1, 0.5, 0.25, 1]
    f = fam === :box ? IndBox(T(0), T(1)) : Constrained(NormL1(T(0.25)), IndBox(T(-1), T(1)))
    lo, hi, λ1 = fam === :box ? (T(0), T(1), T(0)) : (T(-1), T(1), T(0.25))
    m0 = model(T, x, 1, γ, 0, 0; λ1, lo, hi)
    z0 = zat(m0, 0)
    s0 = sigma(m0, 0)
    b = T(s0)
    @test BigFloat(b) == s0                                    # exactly representable
    expected = T.(z0)
    for (kind, set, lb, ub) in (("hyperplane", IndHyperplane(T(1), b), b, b),
                                ("halfspace", IndHalfspace(T(1), b), -Inf, b),
                                ("slab, upper bound", IndHyperslab(T(1), b - 1, b), b - 1, b),
                                ("slab, lower bound", IndHyperslab(T(1), b, b + 1), b, b + 1),
                                ("slab, l = u", IndHyperslab(T(1), b, b), b, b))
      @testset "$T $fam $steps $kind" begin
        c = Constrained(f, set)
        y = fill(T(SENTINEL), n)
        @test (try prox!(y, c, x, γ); true catch; false end) && y == expected
        y = fill(T(SENTINEL), n)
        @test (try value_and_prox!(y, c, x, γ); true catch; false end) && y == expected
        y = copy(x)
        @test (try prox!(y, c, y, γ); true catch; false end) && y == expected
      end
    end
  end
end

@testset "ties of the soft threshold" begin
  for T in (Float32, Float64), steps in (:scalar, :entry)
    f = Constrained(NormL1(T(0.25)), IndBox(T(-1), T(1)))
    γ = steps === :scalar ? one(T) : ones(T, 6)
    # inactive: |x| = γλ gives 0 on either side, x = 1.5 is clamped after the shrink
    x = T[0.25, -0.25, 0.75, -0.75, 0, 1.5]
    for set in (IndHalfspace(T(1), T(5)), IndHyperslab(T(1), T(-5), T(5)))
      y = fill(T(SENTINEL), 6)
      prox!(y, Constrained(f, set), x, γ)
      @test y == T[0, 0, 0.5, -0.5, 0, 1]
    end
    # at the root: ν = 1/4 puts two entries exactly on their thresholds, the sum is 1/2
    x = T[0.5, 1.0, 0.0]
    γ3 = steps === :scalar ? one(T) : ones(T, 3)
    case = Case("$T $steps root on two kinks", T, () -> Constrained(NormL1(T(0.25)), IndHyperplane(T(1), T(0.5))), x, γ3,
                x -> [model(T, x, 1, γ3, T(0.5), T(0.5); λ1 = T(0.25))])
    r = measure(case, case.mk())
    expect(case.name, r, GATE_FIXTURE)
    @test r.err === nothing && exact(r.z, BigFloat[0, 0.5, 0]; bits = 190)
  end
end

@testset "an all-zero lasso solution" begin
  rng = Xoshiro(4)
  for T in (Float32, Float64), steps in (:scalar, :entry), n in (8, 256, 4097)
    x = randn(rng, T, n)
    γ = steps === :scalar ? T(0.7) : T(0.2) .+ rand(rng, T, n)
    f = Constrained(NormL1(T(100)), IndBox(T(-1e3), T(1e3)))
    for set in (IndHyperplane(T(1), T(0)), IndHalfspace(T(1), T(0)), IndHalfspace(T(1), T(1)), IndHyperslab(T(1), T(-1), T(1)),
                IndHyperslab(T(1), T(0), T(0)))
      @testset "$T $steps n=$n $(nameof(typeof(set)))" begin
        y = fill(T(SENTINEL), n)
        prox!(y, Constrained(f, set), x, γ)
        @test all(iszero, y)
        y = fill(T(SENTINEL), n)
        @test iszero(value_and_prox!(y, Constrained(f, set), x, γ)) && all(iszero, y)
      end
    end
  end
end

@testset "a root next to a kink" begin
  rng = Xoshiro(5)
  for T in (Float32, Float64), steps in (:scalar, :entry), n in (3, 64)
    x = randn(rng, T, n)
    x[1] += 2
    γ = steps === :scalar ? T(1) : T(0.5) .+ rand(rng, T, n)
    θ = T(0.25)
    m0 = model(T, x, 1, γ, 0, 0; λ1 = θ)
    # the entry 1 is at its threshold when x₁ − γ₁ν = λγ₁, i.e. ν = x₁/γ₁ − λ
    νk = T(x[1] / (γ isa Real ? γ : γ[1]) - θ)
    b0 = T(sigma(m0, νk))
    for k in (-3, -1, 0, 1, 3)
      b = nextfloat(b0, k)
      case = Case("$T $steps n=$n kink, $k ulps", T, () -> Constrained(NormL1(θ), IndHyperplane(T(1), b)), x, γ,
                  x -> [model(T, x, 1, γ, b, b; λ1 = θ)])
      expect(case.name, measure(case, case.mk()), GATE_FIXTURE)
    end
  end
end

@testset "scaled equivalent constraints" begin
  for T in (Float32, Float64), steps in (:scalar, :entry), fam in (:box, :l1box)
    x, a = T.(DEC_X[fam]), T.(DEC_A)
    γ = steps === :scalar ? T(0.6) : T.(DEC_G)
    f, lo, hi, λ1 = decision_function(fam, T)
    s0 = sigma(model(T, x, a, γ, 0, 0; λ1, lo, hi), 0)
    b0 = T(s0)
    for (kind, lb, ub) in ((:plane, b0 + T(0.25), b0 + T(0.25)), (:half, T(-Inf), b0 - T(0.25)), (:slab, b0 + T(0.125), b0 + T(1)))
      for s in (T(2)^-10, T(2)^7, T(3))
        as = T.(s .* a)
        lbs, ubs = T(s * lb), T(s * ub)
        set = kind === :plane ? IndHyperplane(as, lbs) : kind === :half ? IndHalfspace(as, ubs) : IndHyperslab(as, lbs, ubs)
        case = Case("$T $fam $steps $kind scaled by $s", T, () -> Constrained(f, set), x, γ,
                    x -> [model(T, x, as, γ, lbs, ubs; λ1, lo, hi)])
        r = measure(case, case.mk())
        expect(case.name, r, GATE_FIXTURE)
        if r.err === nothing && s != 3
          # a power of two leaves the constraint, and so the exact point, unchanged
          z0, = oracle(model(T, x, a, γ, lb, ub; λ1, lo, hi))
          @test relerr(r.y, z0, T) <= GATE_FIXTURE
        end
      end
    end
  end
end

@testset "one entry, and none" begin
  for T in (Float32, Float64), steps in (:scalar, :entry)
    γ = steps === :scalar ? T(0.5) : T[0.5]
    case = Case("$T $steps n=1", T, () -> Constrained(IndBox(T(0), T(1)), IndHyperplane(T(2), T(1))), T[0.9], γ,
                x -> [model(T, x, 2, γ, T(1), T(1); lo = 0, hi = 1)])
    expect(case.name, measure(case, case.mk()), GATE_FIXTURE)
    y = T[]
    @test raised(() -> prox!(y, Constrained(IndBox(T(0), T(1)), IndHyperplane(T(1), T(0))), T[], steps === :scalar ? T(0.5) : T[])) === nothing
    @test_throws ArgumentError prox!(T[], Constrained(IndBox(T(0), T(1)), IndHyperplane(T(1), T(1))), T[], steps === :scalar ? T(0.5) : T[])
  end
end

# ---------------------------------------------------------------------------------------------------
# the model of a call (T3): normal and offset are data, steps and levels are parameters
# ---------------------------------------------------------------------------------------------------
@testset "model boundary" begin
  # decision 46(b): Float64 normal (1, 1 + 2^-25, 1), offset 0, Float32 point (2^25, -2^25, 0), Float32 output:
  # residual -1 with the stored normal and third coordinate about 1/3 (the rounded normal gives 0)
  a = [1.0, 1 + ldexp(1.0, -25), 1.0]
  x = Float32[ldexp(1f0, 25), -ldexp(1f0, 25), 0]
  for steps in (:scalar, :entry)
    γ = steps === :scalar ? 1f0 : ones(Float32, 3)
    case = Case("46(b) $steps", Float32, () -> Constrained(NormL1(0f0), IndHyperplane(a, 0.0)), x, γ,
                x -> [model(Float64, x, a, γ, 0.0, 0.0; λ1 = 0f0)])
    r = measure(case, case.mk())
    @testset "$(case.name): $(r.err === nothing ? round(maxulps(r.y, r.z, Float32); digits = 2) : nameof(typeof(r.err))) ulps" begin
      @test r.err === nothing
      if r.err === nothing
        @test maxulps(r.y, r.z, Float32) <= GATE_FIXTURE      # componentwise: x is 2^25 times larger than the answer
        @test abs(r.y[3] - 1 / 3) <= ldexp(1.0, -20)
      end
    end
  end
  # the offset is data too: a Float32 normal with a Float64 offset that a Float32 cannot hold
  for steps in (:scalar, :entry)
    γ = steps === :scalar ? 1f0 : ones(Float32, 3)
    b = ldexp(1.0, 24) + 1 / 3
    x = Float32[ldexp(1f0, 24), 0, 0]
    case = Case("Float64 offset $steps", Float32, () -> Constrained(NormL1(0f0), IndHyperplane(Float32[1, 1, 1], b)), x, γ,
                x -> [model(Float64, x, 1, γ, b, b; λ1 = 0f0)])
    r = measure(case, case.mk())
    @testset "$(case.name): $(r.err === nothing ? round(maxulps(r.y, r.z, Float32); digits = 2) : nameof(typeof(r.err))) ulps" begin
      @test r.err === nothing
      r.err === nothing && @test maxulps(r.y, r.z, Float32) <= GATE_FIXTURE && abs(r.y[3] - 1 / 9) <= ldexp(1.0, -20)
    end
  end
  # the same data in Float64 is the same model; a Float64 step on Float32 data is rounded once to Float32
  for steps in (:scalar, :entry)
    n = 64
    rng = Xoshiro(6)
    x = randn(rng, Float32, n)
    a = randn(rng, Float32, n)
    b = 3.0f0
    γ = steps === :scalar ? 0.1 : 0.1 .+ rand(rng, n)
    case = Case("Float64 step on a Float32 model $steps", Float32,
                () -> Constrained(Constrained(NormL1(0.3f0), IndBox(-1f0, 1f0)), IndHyperplane(a, b)), x, γ,
                x -> [model(Float32, x, a, γ, b, b; λ1 = 0.3f0, lo = -1, hi = 1)])
    expect(case.name, measure(case, case.mk()), GATE_FIXTURE; weighted = steps === :entry)
    x64, a64 = Float64.(x), Float64.(a)
    case = Case("Float64 $steps", Float64,
                () -> Constrained(Constrained(NormL1(0.3), IndBox(-1.0, 1.0)), IndHyperplane(a64, 3.0)), x64, γ,
                x -> [model(Float64, x, a64, γ, 3.0, 3.0; λ1 = 0.3, lo = -1, hi = 1)])
    expect(case.name, measure(case, case.mk()), GATE_FIXTURE; weighted = steps === :entry)
  end
  # integer-literal normals: behaviour recording. The call is judged against the oracle of the model it
  # uses, Float32 or Float64; which of the two a call with a literal normal chooses is not asserted here
  for T in (Float32, Float64), steps in (:scalar, :entry), k in (20, 32)
    n = 64
    x = randn(Xoshiro(8), T, n)
    γ = steps === :scalar ? T(0.5) : T(0.2) .+ rand(Xoshiro(9), T, n)
    case = Case("integer normal $T $steps k=$k", T, () -> Constrained(IndBox(T(0), T(1)), IndHyperplane(1, k)), x, γ,
                x -> unique(m -> m.W, [model(T, x, 1, γ, k, k; lo = 0, hi = 1), model(Float64, x, 1, γ, k, k; lo = 0, hi = 1)]))
    expect(case.name, measure(case, case.mk()), GATE_FIXTURE; weighted = steps === :entry)
  end
  # one function object, called in two precisions in turn: a warm start from the other call is only a seed
  rng = Xoshiro(10)
  n = 300
  a = randn(rng, Float32, n)
  b = Float32(sum(min.(Float64.(a), 0)) + 5)
  c = Constrained(IndBox(0, 1), IndHyperplane(a, b))
  for (T, order) in ((Float32, 1), (Float64, 2), (Float32, 3), (Float64, 4)), steps in (:scalar, :entry)
    x = randn(Xoshiro(100 + order), T, n)
    γ = steps === :scalar ? T(0.5) : T(0.2) .+ rand(Xoshiro(200 + order), T, n)
    case = Case("shared object, call $order in $T $steps", T, () -> c, x, γ, x -> [model(T, x, a, γ, b, b; lo = 0, hi = 1)])
    expect(case.name, measure(case, c), GATE_FIXTURE; weighted = steps === :entry)
  end
end

# ---------------------------------------------------------------------------------------------------
# failure safety (O6): nothing is written on an error, and the function object stays usable
# ---------------------------------------------------------------------------------------------------
# a separable convex function that no wide-recovery hook exists for: it forwards the entry maps and the
# traits of the multiplier search to the wrapped function, nothing else. A trait that the search gains
# later (a rounding-error hook, say) is forwarded here too, but a wide-recovery method is not defined for
# it: a call whose native certificate fails must then raise PrecisionError.
struct NoWide{F} <: ProximalBase.SeparableFunction
  f::F
end
ProximalBase.prox_entry(g::NoWide, v, i, γ) = prox_entry(g.f, v, i, γ)
ProximalBase.value_entry(g::NoWide, v, i) = value_entry(g.f, v, i)
ProximalBase.isconvex(::Type{<:NoWide}) = true
ProximalBase._has_prox_slope(::Type{<:NoWide{F}}) where {F} = ProximalBase._has_prox_slope(F)
ProximalBase._prox_entry_slope(g::NoWide, v, i, γ) = ProximalBase._prox_entry_slope(g.f, v, i, γ)
ProximalBase._has_prox_range(::Type{<:NoWide{F}}) where {F} = ProximalBase._has_prox_range(F)
ProximalBase._prox_range(g::NoWide, i::Integer, ::Type{S}) where {S} = ProximalBase._prox_range(g.f, i, S)
ProximalBase._has_uniform_range(::Type{<:NoWide{F}}) where {F} = ProximalBase._has_uniform_range(F)

@testset "a proven infeasible constraint: ArgumentError naming the domain, nothing written, the object usable" begin
  for T in (Float32, Float64), steps in (:scalar, :entry)
    n = 6
    x = randn(Xoshiro(11), T, n)
    γ = steps === :scalar ? T(0.5) : T(0.2) .+ rand(Xoshiro(12), T, n)
    # a scalar normal: the sum of n entries of [0, 1] cannot be n + 1, but that of n + 3 entries can
    c = Constrained(IndBox(T(0), T(1)), IndHyperplane(T(1), T(n + 1)))
    y = fill(T(SENTINEL), n)
    e = raised(() -> prox!(y, c, x, γ))
    @test e isa ArgumentError && occursin("misses the domain", sprint(showerror, e))
    @test unchanged(y, T)
    e = raised(() -> value_and_prox!(y, c, x, γ))
    @test e isa ArgumentError && occursin("misses the domain", sprint(showerror, e))
    @test unchanged(y, T)
    xa = copy(x)
    e = raised(() -> prox!(xa, c, xa, γ))
    @test e isa ArgumentError && occursin("misses the domain", sprint(showerror, e))
    @test xa == x
    xl = randn(Xoshiro(13), T, n + 3)
    γl = steps === :scalar ? γ : T(0.2) .+ rand(Xoshiro(14), T, n + 3)
    case = Case("$T $steps n+3 after the failure", T, () -> c, xl, γl, x -> [model(T, x, 1, γl, T(n + 1), T(n + 1); lo = 0, hi = 1)])
    expect(case.name, measure(case, c), GATE_FIXTURE; weighted = steps === :entry)
    # vector normals and the other sets: the same error, and the same error again from the same object
    a = T[1, 2, 3, 0, 4, 5]
    for (label, set) in (("hyperplane above the range", IndHyperplane(a, T(16))), ("hyperplane below the range", IndHyperplane(a, T(-1))),
                         ("halfspace below the range", IndHalfspace(a, T(-1))), ("hyperslab above the range", IndHyperslab(a, T(20), T(30))),
                         ("hyperslab below the range", IndHyperslab(a, T(-30), T(-20))))
      c = Constrained(IndBox(T(0), T(1)), set)
      for rep in 1:2
        y = fill(T(SENTINEL), n)
        e = raised(() -> prox!(y, c, x, γ))
        @test e isa ArgumentError && occursin("misses the domain", sprint(showerror, e))
        @test unchanged(y, T)
      end
    end
  end
  # an all-zero normal has no hyperplane: it is refused when the set is built
  @test raised(() -> IndHyperplane(zeros(3), 1.0)) isa ArgumentError
  @test raised(() -> IndHyperslab(zeros(3), -1.0, 1.0)) isa ArgumentError
end

@testset "invalid array steps and sizes: their own errors, nothing written" begin
  for T in (Float32, Float64)
    n = 6
    x = randn(Xoshiro(15), T, n)
    c = Constrained(IndBox(T(0), T(1)), IndHyperplane(T(1), T(2)))
    for op in (prox!, value_and_prox!)
      for bad in (T(0), T(-1), T(NaN))
        γ = fill(T(0.5), n)
        γ[n] = bad
        y = fill(T(SENTINEL), n)
        @test raised(() -> op(y, c, x, γ)) isa ArgumentError
        @test unchanged(y, T)
      end
      y = fill(T(SENTINEL), n)
      @test raised(() -> op(y, c, x, fill(T(0.5), 2))) isa DimensionMismatch
      @test unchanged(y, T)
      for ny in (n - 1, n + 1)
        y = fill(T(SENTINEL), ny)
        @test raised(() -> op(y, c, x, T(0.5))) isa DimensionMismatch
        @test unchanged(y, T)
      end
    end
  end
end

# P1 (docs/src/numerics.md): negative and NaN steps raise an ArgumentError before anything is written.
# A zero or infinite scalar step is not tested: it is an endpoint only where the function names one.
@testset "P1: negative and NaN scalar steps raise before writing" begin
  for T in (Float32, Float64), op in (prox!, value_and_prox!), γ in (T(-1), T(NaN))
    n = 6
    x = randn(Xoshiro(15), T, n)
    for c in (Constrained(IndBox(T(0), T(1)), IndHyperplane(T(1), T(2))), Constrained(NormL1(T(0.3)), IndHyperplane(T(1), T(0))))
      y = fill(T(SENTINEL), n)
      @test raised(() -> op(y, c, x, γ)) isa ArgumentError
      @test unchanged(y, T)
    end
  end
end

@testset "NaN and Inf in the argument" begin
  for T in (Float32, Float64), steps in (:scalar, :entry)
    n = 12
    rng = Xoshiro(16)
    x = randn(rng, T, n)
    a = T[1, -2, 3, 0.5, -4, 5, 1, -1, 2, 0, 1, -3]
    γ = steps === :scalar ? T(0.5) : T(0.2) .+ rand(rng, T, n)
    f = Constrained(NormL1(T(0.1)), IndBox(T(-1), T(1)))
    # NaN is a NaN result, as before, and the object works afterwards
    c = Constrained(f, IndHyperplane(a, T(2)))
    xn = copy(x)
    xn[3] = T(NaN)
    y = fill(T(SENTINEL), n)
    e = raised(() -> prox!(y, c, xn, γ))
    @test e === nothing && all(isnan, y)
    ok = Case("$T $steps after a NaN call", T, () -> c, x, γ, x -> [model(T, x, a, γ, T(2), T(2); λ1 = T(0.1), lo = -1, hi = 1)])
    expect(ok.name, measure(ok, c), GATE_FIXTURE; weighted = steps === :entry)
    # ±Inf entries are clamped by the box: the oracle point, never a PrecisionError
    xi = copy(x)
    xi[2], xi[5], xi[9] = T(Inf), T(-Inf), T(Inf)
    for (label, ff) in (("l1 on a box", f), ("box", IndBox(T(-1), T(1))))
      λ1 = ff === f ? T(0.1) : T(0)
      case = Case("$T $steps ±Inf entries, $label", T, () -> Constrained(ff, IndHyperplane(a, T(0.5))), xi, γ,
                  x -> [model(T, x, a, γ, T(0.5), T(0.5); λ1, lo = -1, hi = 1)])
      r = measure(case, case.mk())
      expect(case.name, r, GATE_FIXTURE; weighted = steps === :entry)
      @test !(r.err isa ProximalBase.PrecisionError)
    end
  end
end

@testset "an uncertifiable result: PrecisionError before writing, the object reusable" begin
  d = flat_data()            # n = 20 000, seed 151: 482 and 381 eps(Float32) wrong without certification
  T = Float32
  for steps in (:scalar, :entry)
    γ = steps === :scalar ? 0.7f0 : d.γ32
    c = Constrained(NoWide(IndBox(0, 1)), IndHyperplane(d.a32, d.b32))
    y = fill(T(SENTINEL), d.n)
    @test raised(() -> prox!(y, c, d.x, γ)) isa ProximalBase.PrecisionError
    @test unchanged(y, T)
    @test raised(() -> value_and_prox!(y, c, d.x, γ)) isa ProximalBase.PrecisionError
    @test unchanged(y, T)
    xa = copy(d.x)
    @test raised(() -> prox!(xa, c, xa, γ)) isa ProximalBase.PrecisionError
    @test xa == d.x
    @test raised(() -> prox!(y, c, d.x, γ)) isa ProximalBase.PrecisionError      # again: nothing was poisoned
    @test unchanged(y, T)
    # the upper bound of a slab is the same problem; the interior of the slab is certified without a search
    slab = Constrained(NoWide(IndBox(0, 1)), IndHyperslab(d.a32, d.b32 - 2f0, d.b32))
    @test raised(() -> prox!(y, slab, d.x, γ)) isa ProximalBase.PrecisionError
    @test unchanged(y, T)
    xin = Float32.(d.a32 .< 0)                  # a point of the box with ⟨a, x⟩ = Σ min(a, 0), inside the slab
    @test raised(() -> prox!(y, slab, xin, γ)) === nothing
    @test y == xin
    @test raised(() -> prox!(y, slab, d.x, γ)) isa ProximalBase.PrecisionError      # and the failure is repeatable
    @test raised(() -> prox!(y, slab, xin, γ)) === nothing && y == xin
  end
  # without the hook an ordinary problem still certifies natively
  for T in (Float32, Float64), steps in (:scalar, :entry)
    case = control(:capped, T, 4096, steps; box = (l, u) -> NoWide(IndBox(l, u)))
    expect(case.name * " (no wide hook)", measure(case, case.mk()), control_gate(T); weighted = steps === :entry)
  end
end

# ---------------------------------------------------------------------------------------------------
# allocations: zero bytes after warm-up, cold (multiplier forgotten) and warm
# ---------------------------------------------------------------------------------------------------
a_prox!(y, c, x, γ) = prox!(y, c, x, γ)
a_vprox!(y, c, x, γ) = value_and_prox!(y, c, x, γ)
a_cold_prox!(y, c, x, γ) = (c.last[] = NaN; c.slope[] = NaN; prox!(y, c, x, γ))
a_cold_vprox!(y, c, x, γ) = (c.last[] = NaN; c.slope[] = NaN; value_and_prox!(y, c, x, γ))

@testset "allocations" begin
  for kind in (:capped, :capped_half, :capped_slab_hi, :lasso, :enet_strict, :posl1, :signed),
      T in (Float32, Float64), steps in (:scalar, :entry), n in (256, 4096)
    case = control(kind, T, n, steps)
    c, y = case.mk(), similar(case.x)
    @testset "$(case.name)" begin
      @test allocs(a_prox!, y, c, case.x, case.γ) == 0
      @test allocs(a_vprox!, y, c, case.x, case.γ) == 0
      @test allocs(a_cold_prox!, y, c, case.x, case.γ) == 0
      @test allocs(a_cold_vprox!, y, c, case.x, case.γ) == 0
      z = copy(case.x)
      @test allocs(a_prox!, z, c, z, case.γ) == 0
    end
  end
end

end
