module ValueOverflowGLMTest

# Values and gradients of GLMLoss, weighted and with offsets, when an intermediate
# (a predictor, a residual, exp(η), a square, the unweighted loss, or d = w ℓ') leaves the
# floating-point range at a representable result (R1, R2, T1-T3, E2; #358 part B).
#
# Every case has a reference written from the mathematical definition of the loss, in BigFloat
# at 2560 bits (512 where that is enough), from the exact data (see `reference`). The same table runs as tests, or, with
# GLM_BASELINE=1 in the environment, as a table of outcomes (correct / finite-but-wrong /
# non-finite) for f, value_and_gradient! and gradient!.

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using JLArrays
using Random

const PB = ProximalBase

# 5 to 100 terms (m, e) for Σ m 2^e: pairs that cancel exactly beyond the range, pairs that differ
# by one ulp beyond it (residue in range), and terms in range, at random separations
function random_pow2_terms(rng, ::Type{T}) where {T}
  n = rand(rng, 5:100)
  hi, mid, lo = T === Float32 ? (200, 140, 120) : (1100, 1060, 1015)
  spread = rand(rng, (0, 10, 40, 100))
  ms, es = T[], Int[]
  while length(ms) < n
    m = T(rand(rng) / 2 + 0.5) * rand(rng, (-1, 1))
    r = rand(rng)
    if r < 0.3 && length(ms) <= n - 2
      e = hi - rand(rng, 0:spread)
      append!(ms, (m, -m)); append!(es, (e, e))
    elseif r < 0.45 && length(ms) <= n - 2
      e = mid - rand(rng, 0:min(spread, 20))
      append!(ms, (m, -nextfloat(m))); append!(es, (e, e))
    else
      push!(ms, m); push!(es, lo - rand(rng, 0:spread))
    end
  end
  p = randperm(rng, n)
  ms[p], es[p]
end

# a user's loss, with the documented interface only: |η - y|
struct AbsLoss <: PB.ScalarLoss end
(::AbsLoss)(η::Real, y::Real) = abs(η - y)
PB.deriv(::AbsLoss, η::Real, y::Real) = sign(η - y)
PB.deriv2(::AbsLoss, η::Real, y::Real) = zero(float(η))
PB.curvature(::AbsLoss) = Inf
# a user's loss defined only for y ≥ 0: (η - y)²/2, throwing elsewhere
struct ThrowsBelowZero <: PB.ScalarLoss end
(::ThrowsBelowZero)(η::Real, y::Real) = y < 0 ? throw(DomainError(y)) : abs2(η - y) / 2
PB.deriv(::ThrowsBelowZero, η::Real, y::Real) = y < 0 ? throw(DomainError(y)) : η - y
PB.deriv2(::ThrowsBelowZero, η::Real, y::Real) = one(float(η))
PB.curvature(::ThrowsBelowZero) = 1.0
const BASELINE = get(ENV, "GLM_BASELINE", "") == "1"

# ---------------------------------------------------------------------------------------------
# Helpers in the style of value_overflow_data.jl
# ---------------------------------------------------------------------------------------------

# v within k units in the last place of its type at the exact value r; below floatmin, within floatmin
function within(v::T, r, k = 2) where {T<:AbstractFloat}
  r = big(r)
  isfinite(v) || return false
  iszero(r) && return iszero(v)
  abs(r) < floatmin(T) && return abs(big(v) - r) <= big(floatmin(T))
  abs(big(v) - r) <= k * ldexp(big(1.0), exponent(r) - precision(T) + 1)
end
# v within k eps(T) of the scale s of the computation, which accounts for cancellation
near_scale(v::T, r, s, k = 8) where {T<:AbstractFloat} = isfinite(v) && abs(big(v) - big(r)) <= k * big(eps(T)) * big(s)

# in a function, so that the value returned from a global is not boxed (Julia 1.10)
allocated_value(f, x) = (f(x); @allocated f(x))
allocated_vg(g, f, x) = (value_and_gradient!(g, f, x); @allocated value_and_gradient!(g, f, x))
allocated_grad(g, f, x) = (gradient!(g, f, x); @allocated gradient!(g, f, x))

wsub_(T) = T === Float32 ? ldexp(one(T), -149) : ldexp(one(T), -1060)

# ---------------------------------------------------------------------------------------------
# Scalar losses with their references. Each reference is written from the definition of the
# loss (the docstring), in BigFloat, with the parameters rounded to T once (T3). ℓ is the
# value, dℓ the derivative with respect to η.
# ---------------------------------------------------------------------------------------------

struct LossSpec
  name::String
  make::Function      # T -> ScalarLoss
  ℓ::Function         # (T, η, y) -> BigFloat
  dℓ::Function
end

# log(1 + e^η) - yη = (1 - y)η + log(1 + e^-η) for η > 0 and log(1 + e^η) - yη otherwise, an identity
# with no cancellation, so that tails far below the precision of η are kept; the exponential term
# is exactly 0 beyond 1e9
_logistic(η, y) = η > 0 ? (1 - y) * η + (η > 1e9 ? zero(η) : log1p(exp(-η))) : (η < -1e9 ? zero(η) : log1p(exp(η))) - y * η
_bexp(η) = η > 1e9 ? big(Inf) : (η < -1e9 ? zero(η) : exp(η))
# the minimizer of ρ_τ(s) + (s - u)²/(2h): soft-thresholding of u at hτ and -h(1 - τ)
_ρ(τ, s) = s >= 0 ? τ * s : (τ - 1) * s
function _moreau(τ, h, u)
  s = u > h * τ ? u - h * τ : (u < -h * (1 - τ) ? u + h * (1 - τ) : zero(u))
  (_ρ(τ, s) + (s - u)^2 / (2h), (u - s) / h)          # value, derivative in u
end
# ∫|a - s| K(s) ds over [-1, 1] for the kernel K, from the antiderivatives F0 of K and F1 of sK:
# 2a F0(a) - 2 F1(a) - a, and its derivative 2 F0(a) - 1
function _kabs(kernel, a)
  if kernel === :uniform
    F0, F1 = (a + 1) / 2, (a^2 - 1) / 4
  else
    F0, F1 = 3 * (a - a^3 / 3 + big(2) / 3) / 4, 3 * (a^2 / 2 - a^4 / 4 - big(1) / 4) / 4
  end
  (2a * F0 - 2F1 - a, 2F0 - 1)
end
# ∫ρ_τ(u - t)K(t/h)/h dt, with ρ_τ(v) = (τ - 1/2)v + |v|/2
function _conv(kernel, τ, h, u)
  u >= h && return (τ * u, τ)
  u <= -h && return ((τ - 1) * u, τ - 1)
  m, dm = _kabs(kernel, u / h)
  ((τ - big(1) / 2) * u + h * m / 2, (τ - big(1) / 2) + dm / 2)
end

const SQUARED = LossSpec("Squared", T -> PB.Squared(), (T, η, y) -> (η - y)^2 / 2, (T, η, y) -> η - y)
const LOGISTIC = LossSpec("Logistic", T -> PB.Logistic(), (T, η, y) -> _logistic(η, y),
  (T, η, y) -> η > 0 ? (1 - y) - (η > 1e9 ? zero(η) : (e = exp(-η); e / (1 + e))) :
               (η < -1e9 ? -y : (e = exp(η); e / (1 + e) - y)))
const POISSON = LossSpec("Poisson", T -> PB.Poisson(), (T, η, y) -> _bexp(η) - (iszero(y) && abs(η) > 1e9 ? zero(η) : y * η),
  (T, η, y) -> _bexp(η) - y)
HUBER(δ) = LossSpec("Huber($δ)", T -> PB.Huber(T(δ)),
  (T, η, y) -> (d = big(T(δ)); r = η - y; abs(r) <= d ? r^2 / 2 : d * (abs(r) - d / 2)),
  (T, η, y) -> (d = big(T(δ)); r = η - y; abs(r) <= d ? r : d * sign(r)))
const SQHINGE = LossSpec("SquaredHinge", T -> PB.SquaredHinge(), (T, η, y) -> max(1 - y * η, zero(η))^2,
  (T, η, y) -> -2y * max(1 - y * η, zero(η)))
SMOOTHCHECK(τ, h) = LossSpec("SmoothCheck($τ,$h)", T -> PB.SmoothCheck(T(τ), T(h)),
  (T, η, y) -> _moreau(big(T(τ)), big(T(h)), y - η)[1],
  (T, η, y) -> -_moreau(big(T(τ)), big(T(h)), y - η)[2])
CONV(kernel, τ, h) = LossSpec("Conv-$kernel($τ,$h)", T -> PB.ConvolutionCheck(T(τ), T(h); kernel),
  (T, η, y) -> _conv(kernel, big(T(τ)), big(T(h)), y - η)[1],
  (T, η, y) -> -_conv(kernel, big(T(τ)), big(T(h)), y - η)[2])

# ---------------------------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------------------------

# T: element type of y, X, weights, offset; B: element type of β (the result is float(B));
# wrap: how β is passed (dense, SparseIterate, BlockIterate); status: :assert, or :info for the
# decisions that the contract leaves open (reported, not asserted).
struct GCase
  group::String
  name::String
  T::DataType
  B::DataType
  spec::LossSpec
  y::Vector
  X::AbstractMatrix
  β::Vector
  w::Union{Nothing, Vector}
  off::Union{Nothing, Vector}
  wrap::Any
  status::Symbol
end
function GCase(group, name, T, spec, y, X, β; B = T, w = nothing, off = nothing, wrap = identity, status = :assert, wT = T)
  GCase(group, name, T, B, spec, Vector{T}(y), X isa SparseMatrixCSC ? SparseMatrixCSC{T}(X) : Matrix{T}(X), Vector{B}(β),
        w === nothing ? nothing : Vector{wT}(w), off === nothing ? nothing : Vector{T}(off), wrap, status)
end

build(c::GCase) = GLMLoss(c.spec.make(c.T), c.y, c.X; weights = c.w, offset = c.off)

# The exact value and gradient, term by term. Zero weights and zero entries of X contribute
# nothing, whatever the (possibly infinite) loss and derivative there.
function reference(c::GCase)
  # 2560 bits: a residual near 2floatmax(Float64) and a bandwidth near 1 must differ exactly
  setprecision(BigFloat, 2560) do
    n = size(c.X, 1)
    Xb, βb = big.(Matrix(c.X)), big.(c.β)
    η = Xb * βb
    c.off === nothing || (η .+= big.(c.off))
    wb = c.w === nothing ? ones(BigFloat, n) : big.(c.w)
    yb = big.(c.y)
    vals = BigFloat[iszero(wb[i]) ? zero(BigFloat) : wb[i] * c.spec.ℓ(c.T, η[i], yb[i]) for i in 1:n]
    ders = BigFloat[iszero(wb[i]) ? zero(BigFloat) : wb[i] * c.spec.dℓ(c.T, η[i], yb[i]) for i in 1:n]
    terms = BigFloat[iszero(Xb[i, j]) || iszero(ders[i]) ? zero(BigFloat) : Xb[i, j] * ders[i] for j in axes(Xb, 2), i in 1:n]
    (value = sum(vals), nvalue = count(!iszero, vals), grad = vec(sum(terms; dims = 2)), terms = terms, ders = ders)
  end
end

# the tolerance of each entry: the terms of one sign are summed with a few roundings, so k ulps;
# terms of both signs can cancel, so the error is k eps of the sum of their magnitudes
function gradient_tolerance(terms, j)
  t = filter(!iszero, terms[j, :])
  (all(>(0), t) || all(<(0), t)) ? (2length(t) + 2, nothing) : (8, sum(abs, t))
end

# :correct, :wrong (finite but wrong, or a finite value for an overflowing one), or :nonfinite
function status(v::R, ref, k, scale = nothing) where {R<:AbstractFloat}
  r = R(ref)
  if isinf(r)
    return v === r ? :correct : isfinite(v) ? :wrong : :nonfinite
  end
  isfinite(v) || return :nonfinite
  ok = abs(r) < floatmin(R) ? abs(big(v) - big(ref)) <= big(floatmin(R)) :
       scale === nothing ? within(v, ref, k) : near_scale(v, ref, scale, k)
  ok ? :correct : :wrong
end
worst(a, b) = a === :correct ? b : (b === :correct ? a : (a === :nonfinite || b === :nonfinite ? :nonfinite : :wrong))
function value_status(c::GCase, v, ref)
  k = max(2ref.nvalue + 2, c.spec.name[1:4] == "Conv" ? 16 : 0)
  status(v, ref.value, k)
end
function gradient_status(c::GCase, g, ref)
  s = :correct
  for j in eachindex(g)
    k, scale = gradient_tolerance(ref.terms, j)
    s = worst(s, status(g[j], ref.grad[j], k, scale))
  end
  s
end

# the outcomes of one case: f, value_and_gradient!, gradient!
function run_case(c::GCase)
  f = build(c)
  R = float(c.B)
  ref = reference(c)
  out = Dict{Symbol, Any}()
  try
    v = f(c.wrap(copy(c.β)))
    out[:f] = (v isa R ? value_status(c, v, ref) : :type, v)
  catch err
    out[:f] = (:error, err)
  end
  try
    g = zeros(R, length(c.β))
    v = value_and_gradient!(g, f, c.wrap(copy(c.β)))
    out[:vgv] = (v isa R ? value_status(c, v, ref) : :type, v)
    out[:vgg] = (gradient_status(c, g, ref), copy(g))
  catch err
    out[:vgv] = (:error, err); out[:vgg] = (:error, err)
  end
  try
    g = zeros(R, length(c.β))
    gradient!(g, f, c.wrap(copy(c.β)))
    out[:g] = (gradient_status(c, g, ref), copy(g))
  catch err
    out[:g] = (:error, err)
  end
  out, ref
end

# ---------------------------------------------------------------------------------------------
# The case table
# ---------------------------------------------------------------------------------------------

function cases(T)
  M = floatmax(T)
  F32 = T === Float32
  lo = F32 ? T(1e-35) : T(1e-300)               # a weight that brings an overflowing loss back
  hi = F32 ? T(1e35) : T(1e300)                 # a weight that brings an underflowing loss back
  Bη = F32 ? T(1e20) : T(1e200)                 # its square overflows
  rq = F32 ? T(1e20) : T(1e160)                 # its square halved overflows (Huber, quadratic zone)
  tn = F32 ? T(1e-25) : T(1e-200)               # its square underflows
  wsub = F32 ? ldexp(one(T), -149) : ldexp(one(T), -1060)   # the smallest weights: powers of two, exact
  tailη = F32 ? [T(120), T(90)] : [T(800), T(725)]  # e^-η underflows; e^-η is subnormal and has lost bits
  cs = GCase[]
  add!(args...; kw...) = push!(cs, GCase(args...; kw...))
  one1 = ones(T, 1, 1)

  # (group, spec, [(name, y, η, w)]): single observations, X = 1, β = η
  groups = [
    ("Squared", SQUARED, [
      ("A overflow: η²/2 = Inf, w tiny", 0, Bη, lo),
      ("B tail: η²/2 underflows, w huge", 0, tn, hi),
      ("weight one, value overflows", 0, Bη, 1),
      ("weight one, ordinary", T(0.5), T(1.5), 1)]),
    ("Logistic", LOGISTIC, [
      ("A overflow: 2η, y = -1, w = 1/4", -1, M, T(0.25)),
      ("A' overflow: y = -1, w = 1", -1, M, 1),
      ("control: y = 0, large η, w tiny", 0, Bη, lo),
      ("B tail y = 1, η>0, e^-η underflows", 1, tailη[1], hi),
      ("B tail y = 1, η>0, e^-η subnormal", 1, tailη[2], hi),
      ("B tail y = 0, η<0, underflows", 0, -tailη[1], hi),
      ("B tail y = 0, η<0, subnormal", 0, -tailη[2], hi),
      ("weight one, ordinary", T(0.3), T(0.7), 1)]),
    ("Poisson", POISSON, [
      ("A overflow: exp(η) = Inf, w tiny", 0, F32 ? T(100) : T(1000), lo),
      ("A overflow with y = 2", 2, F32 ? T(100) : T(1000), lo),
      ("A' tiny subnormal weight", 0, F32 ? T(110) : T(1000), F32 ? ldexp(one(T), -149) : T(5e-324)),
      ("weight one, value overflows (stays Inf)", 0, F32 ? T(100) : T(1000), 1),
      ("B tail y = 0, e^η underflows", 0, -tailη[1], hi),
      ("B tail y = 0, e^η subnormal", 0, -tailη[2], hi),
      ("control: y = 1, η very negative", 1, -T(1e5), 1),
      ("weight one, ordinary", T(0.5), T(0.4), 1)]),
    ("Huber quadratic", HUBER(2rq), [
      ("A overflow: r²/2 = Inf, w tiny", 0, rq, lo),
      ("B tail: r²/2 underflows, w huge", 0, tn, hi),
      ("weight one, value overflows", 0, rq, 1),
      ("weight one, ordinary", T(0.5), T(1.5), 1)]),
    ("Huber linear", HUBER(Bη), [
      ("A overflow: δ(|r| - δ/2) = Inf, w tiny", 0, F32 ? T(1e30) : T(1e300), lo),
      ("A overflow, r < 0", 0, F32 ? T(-1e30) : T(-1e300), lo),
      ("weight one, overflows", 0, F32 ? T(1e30) : T(1e300), 1)]),
    ("Huber subnormal", HUBER(F32 ? T(1e-20) : T(1e-160)), [
      ("B tail: r²/2 subnormal at r = δ, w huge", 0, F32 ? T(1e-20) : T(1e-160), hi)]),
    ("SquaredHinge y=1", SQHINGE, [
      ("A overflow: m² = Inf, w tiny", 1, -Bη, lo),
      ("A overflow, y = -1", -1, Bη, lo),
      ("weight one, overflows", 1, -Bη, 1),
      ("control: small margin, w huge", 1, 1 - ldexp(one(T), -30), hi),
      ("zero margin side", 1, T(2), 1),
      ("weight one, ordinary", 1, T(0.25), 1)]),
    ("SquaredHinge d overflows", SQHINGE, [
      ("C gradient overflow: d = w(-2ym), w = 1/4", 1, -M, T(0.25))]),
    ("SmoothCheck", SMOOTHCHECK(0.25, 0.5), [
      ("A overflow: u = y - η = -2M, w = 1/2", -M, M, T(0.5)),
      ("A overflow, u = +2M", M, -M, T(0.5)),
      ("weight one, overflows", -M, M, 1),
      ("weight one, ordinary", T(0.1), T(0.3), 1)]),
    ("SmoothCheck quadratic zone", SMOOTHCHECK(0.3, 10 * Bη), [
      ("C intermediate: u² = Inf, u²/2h normal", 0, -Bη, 1),
      ("B tail: u²/2h underflows, w huge", tn, 0, hi)]),
    ("SmoothCheck tau tiny", SMOOTHCHECK(1e-10, F32 ? 1e-30 : 1e-300), [
      ("B tail: τu - hτ²/2 subnormal, w huge", F32 ? T(1e-30) : T(1e-300), 0, hi)]),
    ("SmoothCheck tau=0.5", SMOOTHCHECK(0.5, 0.5), [
      ("weight one, ordinary", T(0.1), T(0.3), 1)]),
  ]
  for kernel in (:epanechnikov, :uniform)
    push!(groups,
      ("Conv-$kernel", CONV(kernel, 0.25, 0.5), [
        ("A overflow: u = -2M, w = 1/2", -M, M, T(0.5)),
        ("A overflow, u = +2M", M, -M, T(0.5)),
        ("weight one, overflows", -M, M, 1),
        ("weight one, inside the kernel", T(0.1), T(0.3), 1)]),
      ("Conv-$kernel tau tiny", CONV(kernel, 1e-10, F32 ? 1e-30 : 1e-300), [
        ("B tail: τu subnormal, w huge", F32 ? T(2e-30) : T(2e-300), 0, hi)]))
  end

  for (gname, spec, obs) in groups
    for (name, y, η, w) in obs
      add!(gname, name, T, spec, [y], one1, [η]; w = [w])
    end
    # an infinite derivative would turn the zeros of X = I into NaN in X'd (0 ⋅ Inf, A1): the cases that
    # truly overflow stay single observations
    obs = filter(o -> !(occursin("weight one", o[1]) && occursin("overflows", o[1])), obs)
    # all together: X = I, each observation its own predictor, a zero weight beside an Inf-producing η
    n = length(obs)
    add!(gname, "multiple scales: all observations, X = I", T, spec, [o[2] for o in obs], Matrix{T}(I, n, n),
         [o[3] for o in obs]; w = [o[4] for o in obs])
    add!(gname, "multiple scales plus zero weight at η = Inf", T, spec, [[o[2] for o in obs]; T(0.5)],
         [Matrix{T}(I, n, n) zeros(T, n); zeros(T, 1, n) T(M)], [[o[3] for o in obs]; T(M)]; w = [[o[4] for o in obs]; 0])
    # the same, with η shifted into an offset (β = 0): η = offset exactly
    add!(gname, "multiple scales, η in the offset", T, spec, [o[2] for o in obs], Matrix{T}(I, n, n),
         zeros(T, n); w = [o[4] for o in obs], off = [o[3] for o in obs])
  end

  # ---- zero weight beside an Inf-producing predictor (two observations; the first has w = 0) -------
  for (gname, spec, y1, y2) in (("Squared", SQUARED, 0, 1), ("Logistic", LOGISTIC, 1, 0), ("Poisson", POISSON, 0, 1),
                                ("Huber", HUBER(1.0), 0, 1), ("SquaredHinge", SQHINGE, 1, 1),
                                ("SmoothCheck", SMOOTHCHECK(0.3, 0.2), 0, 1), ("Conv-epanechnikov", CONV(:epanechnikov, 0.3, 0.2), 0, 1))
    # η₁ = M + M overflows; w₁ = 0
    add!(gname, "zero weight, η overflows to Inf", T, spec, [y1, y2], T[M M; 0 1], [1, T(0.5)]; w = [0, T(1.5)])
    add!(gname, "zero weight, η overflows to -Inf", T, spec, [y1, y2], T[-M -M; 0 1], [1, T(0.5)]; w = [0, T(1.5)])
    add!(gname, "zero weight, offset", T, spec, [y1, y2], T[1 0; 0 1], [T(0.5), T(0.5)]; w = [0, T(1.5)], off = [M, 0])
  end

  # ---- cancellation: the sums of the observations' d_i X_ij overflow or cancel ------------------------
  # X'd is the output of a matrix product (A1 of the numerical contract), outside the GLM's own oracles:
  # reported, not asserted
  A = F32 ? ldexp(one(T), 20) : ldexp(one(T), 40)       # A d overflows for d ~ 1e300 (1e35)
  eta0 = zeros(T, 1)
  for (gname, spec, ys) in (("Squared", SQUARED, [1, -1]), ("Squared asymmetric", SQUARED, [1, -1 + ldexp(one(T), -20)]),
                            ("Logistic", LOGISTIC, [0, 1]), ("Logistic asymmetric", LOGISTIC, [0, 1 - ldexp(one(T), -20)]),
                            ("Poisson", POISSON, [0, 2]), ("Huber", HUBER(2.0), [1, -1]), ("SquaredHinge", SQHINGE, [1, -1]),
                            ("SmoothCheck", SMOOTHCHECK(0.5, 0.5), [0.25, -0.25]),
                            ("Conv-uniform", CONV(:uniform, 0.5, 0.5), [0.25, -0.25]),
                            ("Conv-epanechnikov", CONV(:epanechnikov, 0.5, 0.5), [0.25, -0.25]))
    add!(gname, "gradient cancels: d = ±w/2, X = ±A, products overflow", T, spec, ys, reshape([A, A], 2, 1), eta0; w = [hi, hi],
         status = :info)
  end
  # η the same for both rows, d_i = w(η - y_i) of size M/2 that cancel; the products A d overflow
  add!("Squared", "gradient overflows in X'd, cancels", T, SQUARED, [-M / 2, M / 2], reshape([A, A], 2, 1), eta0; w = [1, 1],
       status = :info)

  # ---- predictors that cancel or overflow -----------------------------------------------------------------
  # per loss a representative parameter set and response
  preds = [("Squared", SQUARED, T(0.5)), ("Logistic", LOGISTIC, T(0.5)), ("Poisson", POISSON, T(0.5)),
           ("Huber", HUBER(1.0), T(0.5)), ("SquaredHinge", SQHINGE, one(T)), ("SmoothCheck", SMOOTHCHECK(0.3, 0.5), T(0.5)),
           ("Conv-epanechnikov", CONV(:epanechnikov, 0.3, 0.5), T(0.5)), ("Conv-uniform", CONV(:uniform, 0.3, 0.5), T(0.5))]
  a8 = T(0.8) * M
  for (gname, spec, y0) in preds, w in (T(1.5), wsub)
    tag = w == wsub ? ", w tiny" : ""
    # X = [M -M], β = [2, 2]: η = 0
    add!(gname, "predictor X=[M -M], β=[2,2], η = 0$tag", T, spec, [y0], reshape(T[M, -M], 1, 2), T[2, 2]; w = [w])
    # X = [M -M 1], β = [2, 2, 3]: η = 3
    add!(gname, "predictor X=[M -M 1], β=[2,2,3], η = 3$tag", T, spec, [y0], reshape(T[M, -M, 1], 1, 3), T[2, 2, 3]; w = [w])
    # cancellation with an offset: 0.8M + 0.8M - M = 0.6M (up to the rounding of 0.8M, which is exact data)
    add!(gname, "predictor offset: 0.8M·1 + 0.8M·1 - M$tag", T, spec, [y0], reshape(T[a8, a8], 1, 2), T[1, 1]; w = [w], off = [-M])
    # the predictor itself overflows: η = 2M
    add!(gname, "predictor truly overflows: η = 2M$tag", T, spec, [y0], reshape(T[M, M], 1, 2), T[1, 1]; w = [w])
    add!(gname, "predictor truly overflows: η = -2M$tag", T, spec, [y0], reshape(T[-M, -M], 1, 2), T[1, 1]; w = [w])
    # sparse X and a sparse iterate, with two observations
    Xs = sparse(T[M -M 0; 0 0 1])
    add!(gname, "sparse X, cancelling row$tag", T, spec, [y0, y0], Xs, T[2, 2, 5]; w = [w, 2])
    add!(gname, "SparseIterate β$tag", T, spec, [y0], reshape(T[M, -M, 1], 1, 3), T[2, 2, 3]; w = [w], wrap = SparseIterate)
    add!(gname, "BlockIterate β$tag", T, spec, [y0], reshape(T[M, -M, 1], 1, 3), T[2, 2, 3]; w = [w], wrap = b -> BlockIterate(b, [1:3]))
    add!(gname, "SparseIterate β with zero entries, sparse X$tag", T, spec, [y0, y0], Xs, T[2, 2, 0]; w = [w, 2], wrap = SparseIterate)
  end
  # the gradient entry itself overflows: X'd with d = 2 and X = M; also the weight 4 of the sparse case
  add!("Squared", "gradient truly overflows (sparse X)", T, SQUARED, [T(0.5), T(1)], sparse(T[M -M 0; 0 0 1]), T[2, 2, 5]; w = [T(4), 2])

  # ---- finite inputs whose intermediates become NaN (the true values are finite or Inf) -----------------
  add!("Logistic", "finite β, η overflows to +Inf, y = 1: value 0 (0·Inf)", T, LOGISTIC, [1], reshape(T[M, M], 1, 2), T[1, 1]; w = [1])
  add!("Logistic", "finite β, η overflows to -Inf, y = 0: value 0", T, LOGISTIC, [0], reshape(T[-M, -M], 1, 2), T[1, 1]; w = [1])
  add!("Poisson", "finite β, η overflows to -Inf, y = 0: value 0 (0·Inf)", T, POISSON, [0], reshape(T[-M, -M], 1, 2), T[1, 1]; w = [1])
  add!("Poisson", "finite β, η overflows to +Inf, y = 2: value Inf (Inf - Inf)", T, POISSON, [2], reshape(T[M, M], 1, 2), T[1, 1]; w = [1])
  add!("Poisson", "finite β, η = 1000 (Inf - y·η), y = 2, w = 1", T, POISSON, [2], one1, [F32 ? T(100) : T(1000)]; w = [1])

  # ---- infinite β: the loss at η = ±Inf, where the limit does not depend on the path -----------------------
  for (gname, spec, y, η) in (("Squared", SQUARED, T(0.5), T(Inf)), ("Squared", SQUARED, T(0.5), T(-Inf)),
                              ("Logistic", LOGISTIC, T(0.5), T(Inf)), ("Logistic", LOGISTIC, T(0.5), T(-Inf)),
                              ("Poisson", POISSON, T(0.5), T(Inf)), ("Huber", HUBER(1.0), T(0.5), T(Inf)),
                              ("SquaredHinge", SQHINGE, one(T), T(-Inf)), ("SmoothCheck", SMOOTHCHECK(0.3, 0.5), T(0.5), T(-Inf)),
                              ("Conv-epanechnikov", CONV(:epanechnikov, 0.3, 0.5), T(0.5), T(Inf)))
    add!(gname, "β = $η, y = $y (infinite input)", T, spec, [y], one1, [η]; w = [T(1.5)], status = :info)
  end
  # joint limits that depend on the path: reported only
  add!("Logistic", "β = +Inf, y = 1 (0·Inf: path dependent)", T, LOGISTIC, [1], one1, [T(Inf)]; w = [1], status = :info)
  add!("Poisson", "β = -Inf, y = 0 (0·Inf: path dependent)", T, POISSON, [0], one1, [T(-Inf)]; w = [1], status = :info)
  # 0·Inf in X'd: d₁ overflows, X₁₂ = 0, and the true entry g₂ is finite
  add!("Poisson", "d overflows, X has a zero in that row: g₂ = d₂ (0·Inf)", T, POISSON, [0, 0], T[1 0; 0 1],
       [F32 ? T(100) : T(1000), 0]; w = [1, 1], status = :info)

  # ---- mixed precision: results of float(eltype β), the model precision is the widest (T3) ------------------
  if T === Float32
    # Float32 data, Float64 β: η = 1e39 does not fit in the Float32 buffer; the value is a Float64. The model
    # precision of data functions (decision 6 of the numerical contract) decides these: reported, not asserted
    add!("Squared", "Float32 data, Float64 β: η = 1e39, w = 2^-100", Float32, SQUARED, [0f0], ones(Float32, 1, 1), [1e39]; B = Float64, w = [ldexp(1f0, -100)], status = :info)
    add!("Poisson", "Float32 data, Float64 β: exp(100) overflows Float32", Float32, POISSON, [0f0], ones(Float32, 1, 1), [100.0]; B = Float64, w = [1f-35], status = :info)
    add!("Logistic", "Float32 data, Float64 β: tail e^-120 underflows Float32, w = 1e35", Float32, LOGISTIC, [1f0], ones(Float32, 1, 1), [120.0]; B = Float64, w = [1f35], status = :info)
    add!("Squared", "Float32 data, Float64 β: X β cancels (M32, -M32)", Float32, SQUARED, [0.5f0], reshape(Float32[floatmax(Float32), -floatmax(Float32)], 1, 2), [2.0, 2.0]; B = Float64, w = [1f0], status = :info)
    # Float64 weights that are not Float32 (T3: weights are data, the model precision is Float64)
    add!("Poisson", "Float32 data, Float64 β, Float64 weight 1e-300", Float32, POISSON, [0f0], ones(Float32, 1, 1), [1000.0]; B = Float64, w = [1e-300], wT = Float64, status = :info)
  else
    # Float64 data, Float32 β: the value is a Float32, computed from the Float64 data
    add!("Poisson", "Float64 data, Float32 β: exp(40)·1e-30", Float64, POISSON, [0.0], ones(1, 1), Float32[40]; B = Float32, w = [1e-30])
    add!("Poisson", "Float64 data, Float32 β: value 7e38 > floatmax(Float32) is Inf", Float64, POISSON, [0.0], ones(1, 1), Float32[101]; B = Float32, w = [1e-5])
    add!("Poisson", "Float64 data, Float32 β: value 2.7e38 is normal", Float64, POISSON, [0.0], ones(1, 1), Float32[100]; B = Float32, w = [1e-5])
    add!("Logistic", "Float64 data, Float32 β: tail e^-790 underflows Float64, w = 1e308; value 3e-35", Float64, LOGISTIC, [1.0], ones(1, 1), Float32[790]; B = Float32, w = [1e308])
    add!("Squared", "Float64 data, Float32 β: value 1e60 overflows Float32 (stays Inf)", Float64, SQUARED, [0.0], ones(1, 1), Float32[1e30]; B = Float32, w = [1.0])
  end
  cs
end

# the cases that run on the device: (spec, y, X, β, w, offset)
function devcases(T)
  F32 = T === Float32
  M = floatmax(T)
  lo, hi = F32 ? T(1e-35) : T(1e-300), F32 ? T(1e35) : T(1e300)
  [
      (POISSON, T[0], ones(T, 1, 1), T[F32 ? 100 : 1000], T[lo], nothing),                     # B11a: exp overflows, w tiny
      (LOGISTIC, T[1], ones(T, 1, 1), T[F32 ? 120 : 800], T[hi], nothing),                      # B11b: the tail, w huge
      (SQHINGE, T[1], ones(T, 1, 1), T[F32 ? -1e20 : -1e200], T[lo], nothing),                  # B11c: m² overflows
      (SQUARED, T[0.5], reshape(T[M, -M, 1], 1, 3), T[2, 2, 3], T[1.5], nothing),               # Xβ cancels
      (HUBER(1.0), T[0.5], reshape(T[M, -M, 1], 1, 3), T[2, 2, 3], T[wsub_(T)], nothing),
      (LOGISTIC, T[0], reshape(T[0.8M, 0.8M], 1, 2), T[1, 1], T[1], T[-M]),                     # cancellation with an offset
      (SMOOTHCHECK(0.25, 0.5), T[-M], ones(T, 1, 1), T[M], T[0.5], nothing),                    # u = y - η overflows
      (POISSON, T[0, 0], T[1 0; 0 1], T[F32 ? 100 : 1000, 0], T[lo, 0], nothing),               # two scales
      (POISSON, T[0, 0], T[1 0; 0 1], T[F32 ? 100 : 1000, 0], T[0, 1], nothing),                # a zero weight beside an overflow
      (CONV(:epanechnikov, 1e-10, F32 ? 1e-30 : 1e-300), T[F32 ? 2e-30 : 2e-300], ones(T, 1, 1), T[0], T[hi], nothing),
  ]
end

function device_case(T, (spec, y, X, β, w, off))
  c = GCase("jl", "device", T, spec, y, X, β; w = w, off = off)
  f = GLMLoss(spec.make(T), JLArray(c.y), JLArray(c.X); weights = JLArray(c.w), offset = off === nothing ? nothing : JLArray(c.off))
  c, f
end

# ---------------------------------------------------------------------------------------------
# Baseline table
# ---------------------------------------------------------------------------------------------

function baseline()
  tally = Dict{Tuple{Symbol, Symbol}, Int}()
  rows = String[]
  for T in (Float32, Float64), c in cases(T)
    out, _ = run_case(c)
    for key in (:f, :vgv, :vgg, :g)
      s = out[key][1]
      tally[(c.status === :info ? :info : :assert, s)] = get(tally, (c.status === :info ? :info : :assert, s), 0) + 1
      tally[(key, s)] = get(tally, (key, s), 0) + (c.status === :assert ? 1 : 0)
    end
    st = [out[k][1] for k in (:f, :vgv, :vgg, :g)]
    push!(rows, string(rpad(string(T), 8), rpad(c.group, 28), rpad(c.name, 70), join(rpad.(string.(st), 11)), c.status === :info ? " (info)" : ""))
  end
  for T in (Float32, Float64), dc in devcases(T)
    c, f = device_case(T, dc)
    ref = reference(c)
    βd() = JLArray(copy(c.β))
    st = Symbol[]
    try push!(st, value_status(c, f(βd()), ref)) catch; push!(st, :error) end
    try
      gd = JLArray(zeros(T, length(c.β)))
      push!(st, value_status(c, value_and_gradient!(gd, f, βd()), ref)); push!(st, gradient_status(c, Array(gd), ref))
    catch
      push!(st, :error, :error)
    end
    try
      gd = JLArray(zeros(T, length(c.β))); gradient!(gd, f, βd()); push!(st, gradient_status(c, Array(gd), ref))
    catch
      push!(st, :error)
    end
    push!(rows, string(rpad(string(T), 8), rpad("JLArray " * dc[1].name, 28), rpad("n = $(length(dc[2]))", 70), join(rpad.(string.(st[[1, 2, 3, 4]]), 11))))
    for (key, s_) in zip((:f, :vgv, :vgg, :g), st)
      tally[(key, s_)] = get(tally, (key, s_), 0) + 1
    end
  end
  println.(rows)
  println()
  for key in (:f, :vgv, :vgg, :g), s in (:correct, :wrong, :nonfinite, :error, :type)
    haskey(tally, (key, s)) && println(rpad(string(key), 5), rpad(string(s), 10), tally[(key, s)])
  end
  println("cases: ", sum(length(cases(T)) for T in (Float32, Float64)), " of which info: ",
          sum(count(c -> c.status === :info, cases(T)) for T in (Float32, Float64)))
end

# ---------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------

function test_case(c::GCase)
  f = build(c)
  R = float(c.B)
  ref = reference(c)
  β() = c.wrap(copy(c.β))
  v = f(β())
  @test v isa R
  @test value_status(c, v, ref) === :correct
  g1 = zeros(R, length(c.β))
  v1 = value_and_gradient!(g1, f, β())
  @test v1 isa R
  @test value_status(c, v1, ref) === :correct
  g2 = zeros(R, length(c.β))
  @test gradient!(g2, f, β()) === g2
  @test isequal(g1, g2)
  # A derivative w ℓ' below floatmin is stored with the bits it has, and X'd can
  # bring it back into the normal range: an intermediate of the product X'd (A1 of
  # the numerical contract), outside these oracles
  any(d -> !iszero(d) && abs(d) < floatmin(c.T), ref.ders) && return
  @test gradient_status(c, g1, ref) === :correct
  @test gradient_status(c, g2, ref) === :correct
end

# The Float32 storage cases of Logistic, Huber, SmoothCheck and the two Conv losses (sparse X, SparseIterate β and
# BlockIterate β, with the two weights of each), and the three of those cases of Poisson that are not the SparseIterate
# β with zero entries, run in value_overflow_glm_grids.jl. The Float64 pass, the dense Float32 cases of these losses and
# all the Float32 cases of Squared and SquaredHinge stay here.
function in_grids_file(c::GCase)
  c.T === Float32 || return false
  storage = startswith(c.name, "sparse X, cancelling row") || startswith(c.name, "SparseIterate β") ||
            startswith(c.name, "BlockIterate β")
  storage && (c.group in ("Logistic", "Huber", "SmoothCheck", "Conv-epanechnikov", "Conv-uniform") ||
              (c.group == "Poisson" && !startswith(c.name, "SparseIterate β with zero entries")))
end

if !BASELINE
@testset "GLMLoss values and gradients that leave the range" begin
  # the references agree with the package on ordinary inputs (a check of the table itself)
  @testset "references on ordinary inputs" begin
    for T in (Float64,), spec in (SQUARED, LOGISTIC, POISSON, HUBER(0.7), SQHINGE, SMOOTHCHECK(0.3, 0.5),
                                  CONV(:uniform, 0.3, 0.5), CONV(:epanechnikov, 0.7, 0.4))
      ℓ = spec.make(T)
      for η in (-3.0, -0.4, 0.0, 0.2, 0.45, 2.5), y in (0.0, 0.5, 1.0, -1.0)
        c = GCase("ref", "ordinary", T, spec, [y], ones(T, 1, 1), [η])
        ref = reference(c)
        @test isapprox(ℓ(η, y), Float64(ref.value); rtol = 1e-13, atol = 1e-15)
        @test isapprox(PB.deriv(ℓ, η, y), Float64(ref.grad[1]); rtol = 1e-13, atol = 1e-15)
      end
    end
  end

  for T in (Float32, Float64)
    @testset "$T" begin
      for c in cases(T)
        c.status === :info && continue
        in_grids_file(c) && continue
        @testset "$(c.group): $(c.name)" begin
          test_case(c)
        end
      end
    end
  end

  # ordinary inputs keep the ordinary computation: bitwise the loop over the stored predictor
  @testset "ordinary inputs keep the ordinary computation" begin
    for T in (Float32, Float64), spec in (SQUARED, LOGISTIC, POISSON, HUBER(0.7), SQHINGE, SMOOTHCHECK(0.3, 0.5), CONV(:epanechnikov, 0.3, 0.5))
      n, p = 30, 4
      X = T.(sin.(1:n) .* cos.((1:p)'))
      y = T.(abs.(cos.(1:n)))
      w = T.(0.5 .+ abs.(sin.(3 .* (1:n))))
      β = T.(sin.(1:p) ./ 3)
      f = GLMLoss(spec.make(T), y, X; weights = w, offset = T.(0.1 .* cos.(1:n)))
      v = f(β)
      ℓ = f.loss
      plain = zero(T)
      for i in 1:n
        plain += w[i] * ℓ(f.η[i], y[i])
      end
      @test v === plain
      g = zeros(T, p)
      @test value_and_gradient!(g, f, β) === plain
      @test isequal(gradient!(zeros(T, p), f, β), g)
    end
  end

  # the recovery of the terms allocates nothing
  @testset "allocations" begin
    for T in (Float32, Float64)
      F32 = T === Float32
      f = GLMLoss(PB.Poisson(), T[0], ones(T, 1, 1); weights = T[F32 ? 1e-35 : 1e-300])
      β = T[F32 ? 100 : 1000]
      g = zeros(T, 1)
      @test allocated_value(f, β) == 0
      @test allocated_vg(g, f, β) == 0
      @test allocated_grad(g, f, β) == 0
      f2 = GLMLoss(PB.Squared(), T[1], reshape(T[floatmax(T), -floatmax(T), 1], 1, 3); weights = T[0.5])
      β2 = T[2, 2, 3]
      g2 = zeros(T, 3)
      # the predictor recovery copies β, scaled, and allocates nothing else
      @test allocated_value(f2, β2) <= 64 + 2sizeof(β2)
      @test allocated_vg(g2, f2, β2) <= 64 + 2sizeof(β2)
    end
  end

  # ordinary calls allocate nothing, weighted and unweighted
  @testset "ordinary calls allocate nothing" begin
    for T in (Float32, Float64), spec in (SQUARED, LOGISTIC, POISSON, HUBER(0.7), SQHINGE, SMOOTHCHECK(0.3, 0.5), CONV(:epanechnikov, 0.3, 0.5))
      n, p = 20, 3
      X = T.(sin.(1:n) .* cos.((1:p)'))
      y = T.(abs.(cos.(1:n)))
      β = T.(sin.(1:p) ./ 3)
      for kw in ((;), (weights = T.(0.5 .+ 2 .* abs.(sin.(3 .* (1:n)))), offset = T.(0.1 .* cos.(1:n))))
        f = GLMLoss(spec.make(T), y, X; kw...)
        g = zeros(T, p)
        @test allocated_value(f, β) == 0
        @test allocated_vg(g, f, β) == 0
        @test allocated_grad(g, f, β) == 0
      end
    end
  end

  # the recovery leaves no state behind: an ordinary call after a recovered one is unchanged
  @testset "repeated calls" begin
    for T in (Float32, Float64)
      M = floatmax(T)
      X = T[M -M 0; 0 1 1; 1 0 2]
      f = GLMLoss(PB.Logistic(), T[1, 0, 1], X; weights = T[1.5, 2, 0.5])
      βo = T[0.1, 0.2, -0.3]
      vo = f(βo)
      go = gradient!(zeros(T, 3), f, βo)
      @test isfinite(f(T[2, 2, 1]))                  # the first row cancels: recovered
      @test isfinite(value_and_gradient!(zeros(T, 3), f, T[2, 2, 1]))
      @test f(βo) === vo
      @test isequal(gradient!(zeros(T, 3), f, βo), go)
      gv = zeros(T, 3)
      @test value_and_gradient!(gv, f, βo) === vo && isequal(gv, go)
    end
  end

  # a sparse design whose cancelling row sits next to a zero-weight row with an infinite predictor
  @testset "sparse design, a recovered row beside a zero weight" begin
    for T in (Float32, Float64)
      M = floatmax(T)
      X = sparse(T[M -M 0; M M 0; 0 0 1])
      f = GLMLoss(PB.Squared(), T[0.5, 0, 1], X; weights = T[2, 0, 1])
      β = T[2, 2, 3]
      ref = 2 * (big(T(0.5)))^2 / 2 + (3 - 1)^2 / 2
      @test within(f(β), ref, 4)
      g = zeros(T, 3)
      @test within(value_and_gradient!(g, f, β), ref, 4)
      @test isequal(g, gradient!(zeros(T, 3), f, β))
      @test g == T[-2 * T(0.5) * M, 2 * T(0.5) * M, 2]
    end
  end

  # review round 1 (numerics): the predictor recovery in the wider type of β; an underflow inside
  # SmoothCheck at any weight; a subnormal residual in the derivative of ConvolutionCheck
  @testset "review findings" begin
    # Float32 data, a Float64 β far beyond floatmax(Float32): the predictor truly overflows
    @test GLMLoss(PB.Squared(), Float32[0], fill(1f38, 1, 1))([1e100]) === Inf
    @test GLMLoss(PB.Poisson(), Float32[0], ones(Float32, 1, 1))([1e300]) === Inf
    g = zeros(1)
    @test value_and_gradient!(g, GLMLoss(PB.Squared(), Float32[0], fill(1f38, 1, 1)), [1e100]) === Inf
    @test g[1] === Inf
    gl = gradient!(zeros(1), GLMLoss(PB.Logistic(), Float32[0], fill(1f38, 1, 1)), [1e100])
    @test gl[1] === 1f38 * 1.0
    # SmoothCheck: u² underflows inside the loss, u²/(2h) is normal; also at w ≤ 1, and with 2h = Inf
    for w in (0.5, 1.0, 1.0000001)
      f = GLMLoss(PB.SmoothCheck(0.5, 1e-150), [0.0], ones(1, 1); weights = [w])
      @test within(f([1e-170]), big(w) * big(1e-170)^2 / (2 * big(1e-150)), 4)
      @test within(value_and_gradient!(zeros(1), f, [1e-170]), big(w) * big(1e-170)^2 / (2 * big(1e-150)), 4)
    end
    fh = GLMLoss(PB.SmoothCheck(0.5, 1.5e308), [0.0], ones(1, 1))
    @test within(fh([1e150]), big(1e150)^2 / (2 * big(1.5e308)), 4)
    @test within(GLMLoss(PB.SmoothCheck(0.5f0, 1f-20), Float32[0], ones(Float32, 1, 1))(Float32[1f-25]),
                 big(1f-25)^2 / (2 * big(1f-20)), 4)
    # ConvolutionCheck: the derivative at a subnormal residual, τ = 1/2, lifted by the weight
    for (T, η, w) in ((Float64, -1e-320, 1e300), (Float32, -1f-44, 1f30)), kernel in (:uniform, :epanechnikov)
      h = T(3)
      f = GLMLoss(PB.ConvolutionCheck(T(0.5), h; kernel), T[0], ones(T, 1, 1); weights = T[w])
      c = kernel === :uniform ? big(1) / 2 : big(3) / 4      # dℓ/dη = -m′(u/h)/2 with m′(s) ≈ s or 3s/2, u = -η
      ref = big(w) * c * big(η) / big(h)
      g = gradient!(zeros(T, 1), f, T[η])
      @test within(g[1], ref, 4)
      g2 = zeros(T, 1)
      value_and_gradient!(g2, f, T[η])
      @test isequal(g2, g)
    end
  end

  # Terms of different observations beyond floatmax that cancel (review of 066e1eb3): the sum of the
  # recomputed terms stays in power-of-two units until the end. The problem is ill-conditioned beyond
  # 1/eps (|terms| ≈ 5.4e308, value ≈ 7.0e291), so R1's "accuracy of the ordinary computation" is the
  # cancellation scale: the result equals the ordinary computation of the same problem scaled down by
  # an exact power of two, where nothing leaves the range.
  @testset "terms beyond floatmax that cancel between observations" begin
    for T in (Float64, Float32)
      w1 = T === Float64 ? 1e308 : 2f38
      w2 = setprecision(BigFloat, 512) do
        T(big(w1) * exp(big(1)) / (6 - exp(big(1))))
      end
      ref, scale = setprecision(BigFloat, 512) do
        t1, t2 = big(w1) * exp(big(1)), big(w2) * (exp(big(1)) - 6)
        t1 + t2, abs(t1) + abs(t2)
      end
      f = PoissonLoss(T[0, 6], ones(T, 2, 1); weights = T[w1, w2])
      v = f(T[1])
      @test v isa T && isfinite(v)
      @test near_scale(v, ref, scale)
      g1 = zeros(T, 1)
      @test value_and_gradient!(g1, f, T[1]) === v
      # the same problem with the weights scaled by 2^-60 takes the ordinary path
      fs = PoissonLoss(T[0, 6], ones(T, 2, 1); weights = T[w1, w2] .* ldexp(one(T), -60))
      @test ldexp(fs(T[1]), 60) === v
      # each wᵢℓ'ᵢ is beyond floatmax by itself: d is not representable and X'd is NaN, an issue of the
      # product (A1, #542); both gradient calls agree
      g2 = zeros(T, 1)
      gradient!(g2, f, T[1])
      @test isequal(g1, g2)
      # on the device
      fd = GLMLoss(PB.Poisson(), JLArray(T[0, 6]), JLArray(ones(T, 2, 1)); weights = JLArray(T[w1, w2]))
      @test fd(JLArray(T[1])) === v
      @test value_and_gradient!(JLArray(zeros(T, 1)), fd, JLArray(T[1])) === v
    end
    # finite terms whose partial sums overflow (y outside [0, 1]): +M, +M and -M in any order
    M = floatmax(Float64)
    obs = [(2.0, -M / 2), (-1.0, M / 2), (3.0, M / 2)]        # terms M, M and -M
    for perm in ([1, 2, 3], [1, 3, 2], [3, 1, 2])
      yy, ηs = first.(obs[perm]), last.(obs[perm])
      fl = LogisticLoss(yy, Matrix(1.0I, 3, 3))
      t = setprecision(BigFloat, 2560) do
        sum(_logistic(big(η), big(y)) for (y, η) in zip(yy, ηs))
      end
      @test within(fl(ηs), t, 8)
    end
  end

  @testset "sums in power-of-two units" begin
    pair(m, e) = (m, e)
    for T in (Float32, Float64)
      E = T === Float32 ? 200 : 1100
      # large terms that cancel exactly, then a small one, in any order
      for (ms, es) in ((T[0.5, -0.5, 0.75], [E, E, 0]), (T[0.75, 0.5, -0.5], [0, E, E]), (T[0.5, 0.75, -0.5], [E, 0, E]))
        @test PB._pow2_broadcast_sum(T, pair, ms, es) === T(0.75)
      end
      # a small term far below, beside a partial cancellation
      @test PB._pow2_broadcast_sum(T, pair, T[0.75, -0.5, 0.5], [E, E, -E]) === ldexp(T(0.5), E - 2)
      @test PB._pow2_broadcast_sum(T, pair, T[0, 0], [0, 0]) === zero(T)
      @test PB._pow2_broadcast_sum(T, pair, T[0.5, -0.5], [E, E]) === zero(T)
      @test PB._pow2_broadcast_sum(T, pair, T[0.5, 0.5], [E, 3]) === T(Inf)          # truly beyond floatmax
      @test PB._pow2_broadcast_sum(T, pair, T[Inf, 0.5], [0, E]) === T(Inf)
      @test isnan(PB._pow2_broadcast_sum(T, pair, T[Inf, -Inf], [0, 0]))
      @test isnan(PB._pow2_broadcast_sum(T, pair, T[NaN, 0.5], [0, 0]))
    end
    # large terms of different significands that cancel only with their rounding errors carried along
    # (review of 262ce757)
    ms32 = Float32[0.98000276, 0.6838328, -0.98000276, -0.94564354, -0.6838328]
    @test PB._pow2_broadcast_sum(Float32, pair, ms32, [300, 300, 300, 100, 300]) === ldexp(-0.94564354f0, 100)
    ms64 = [-0.8091040536903062, 0.7714371019247288, -0.7378713589447876, -0.7714371019247288, 0.7378713589447876]
    @test PB._pow2_broadcast_sum(Float64, pair, ms64, [-995, 2003, 1998, 2003, 1998]) === ldexp(-0.8091040536903062, -995)
    # the documented underflow limit: a term below the subnormal range in the small class's units
    # (review of d4333015; exact sum 786432)
    @test PB._pow2_broadcast_sum(Float32, pair, Float32[0.75, -0.75, 0.5, -0.5, 0.75], [300, 300, 200, 200, 20]) === 0f0
    # random sums against BigFloat: exact pairs beyond the range, near-exact pairs whose residue is
    # in range, and terms in range, in random order; error within eps |sum| plus the compensated
    # summation's bound, an exactly cancelling sum is zero, and a sum beyond floatmax is infinite
    # many terms: the zero rule is off beyond 2n u = 1/8, so a sum that does not cancel is kept
    # (n near 1/(2 eps) once returned 0; review of b8d09379)
    for n in (1_000_000, 3_600_000, 4_000_000)
      @test PB._pow2_broadcast_sum(Float32, pair, fill(0.75f0, n), fill(100, n)) === ldexp(Float32(0.75n), 100)
    end
    rng = Random.Xoshiro(358)
    for T in (Float32, Float64), trial in 1:400
      ms, es = random_pow2_terms(rng, T)
      s = PB._pow2_broadcast_sum(T, pair, ms, es)
      exact, A = setprecision(BigFloat, 4096) do
        sum(ldexp(big(m), e) for (m, e) in zip(ms, es)), sum(abs(ldexp(big(m), e)) for (m, e) in zip(ms, es))
      end
      g = length(ms) * eps(T)
      if !isfinite(T(exact))
        @test s === T(exact)
      elseif iszero(exact)
        @test s === zero(T)
      else
        @test abs(big(s) - exact) <= 2eps(T) * abs(exact) + 6g^2 * A
      end
    end
  end

  # a user's loss keeps the ordinary term w ℓ(η, y), also when the predictor is recomputed
  @testset "a user's ScalarLoss" begin
    for T in (Float32, Float64)
      M = floatmax(T)
      f = GLMLoss(AbsLoss(), T[0.5, 1], T[M -M 1; 0 0 1]; weights = T[3, 2])
      β = T[2, 2, 3]
      # row 1 cancels: η = 3, recovered; row 2: η = 3
      @test f(β) === T(3) * T(2.5) + T(2) * T(2)
      g = zeros(T, 3)
      @test value_and_gradient!(g, f, β) === f(β)
      @test g == T[3M, -3M, 3 + 2]
      @test isequal(gradient!(zeros(T, 3), f, β), g)
      # ordinary inputs: the plain weighted sum
      βo = T[0.1, 0.2, 0.3]
      @test f(βo) === T(3) * abs(f.η[1] - T(0.5)) + T(2) * abs(f.η[2] - T(1))
      # a tiny term of a user's loss is not recomputed (no internal methods): the ordinary product
      ft = GLMLoss(AbsLoss(), T[0], ones(T, 1, 1); weights = T[T === Float32 ? 1f30 : 1e300])
      @test ft(T[nextfloat(zero(T))]) === ft.weights[1] * nextfloat(zero(T))
    end
  end

  # ConvolutionCheck inside the kernel zone, with a value below floatmin lifted by the weight
  @testset "ConvolutionCheck kernel zone" begin
    for kernel in (:uniform, :epanechnikov), (T, h, w) in ((Float64, 1e-310, 1e300), (Float32, 1f-40, 1f30))
      f = GLMLoss(PB.ConvolutionCheck(T(0.5), h; kernel), T[0], ones(T, 1, 1); weights = T[w])
      u = h / 2                                      # y - η with y = 0, η = -h/2
      ref = setprecision(BigFloat, 2560) do
        big(w) * _conv(kernel, big(T(0.5)), big(h), big(u))[1]
      end
      @test within(f(T[-u]), ref, 8)
    end
  end

  @testset "exponent split in other float types" begin
    m, j = PB._exp_split(big(3.0))
    @test m * big(2.0)^j ≈ exp(big(3.0))
  end

  # NaN inputs propagate (E2), to the value and to every entry of the gradient that depends on β
  @testset "NaN propagates" begin
    for T in (Float32, Float64), spec in (SQUARED, LOGISTIC, POISSON, HUBER(0.7), SQHINGE, SMOOTHCHECK(0.3, 0.5), CONV(:uniform, 0.3, 0.5))
      f = GLMLoss(spec.make(T), T[0.5, 1], ones(T, 2, 2); weights = T[1.5, 0.5])
      β = T[NaN, 1]
      @test isnan(f(β))
      g = zeros(T, 2)
      @test isnan(value_and_gradient!(g, f, β)) && all(isnan, g)
      g = zeros(T, 2)
      gradient!(g, f, β)
      @test all(isnan, g)
      # a NaN entry in an offset or a response is rejected or propagates: here a NaN response
      fy = GLMLoss(spec.make(T), T[NaN, 1], ones(T, 2, 2); weights = T[1.5, 0.5])
      @test isnan(fy(T[0.2, 0.1]))
    end
  end

  # The ordinary loops leave out a test that the sum or another test implies, by traits of the
  # built-in losses (`_value_traps`, `_deriv_traps`, `_deriv_in_value`). First the properties
  # the traits claim, at the extreme inputs; then the loops against the recovery run on every call.
  @testset "tests implied at compile time" begin
    specs = (SQUARED, LOGISTIC, POISSON, HUBER(0.7), SQHINGE, SMOOTHCHECK(0.3, 0.5), CONV(:uniform, 0.3, 0.5),
             CONV(:epanechnikov, 0.3, 0.5))
    for T in (Float32, Float64)
      M = floatmax(T)
      ys = T[0, 1, -1, 0.5, 3, M, -M, nextfloat(zero(T)), -1e-30]
      ηs = T[0, 1, -1, 2, -2, 50, -50, 89, -89, 710, -710, 1e10, -1e10, M / 2, -M / 2, M, -M, nextfloat(zero(T))]
      for spec in specs
        ℓ = spec.make(T)
        @testset "$(spec.name), $T: trait properties" begin
          if PB._value_traps(ℓ)
            for η in T[Inf, -Inf, NaN], y in ys
              @test !isfinite(ℓ(η, y)) && !isfinite(PB.value_and_deriv(ℓ, η, y)[1])
            end
          end
          if PB._deriv_traps(ℓ)
            for η in T[Inf, -Inf, NaN], y in ys
              @test !isfinite(PB.deriv(ℓ, η, y)) && !isfinite(PB.value_and_deriv(ℓ, η, y)[2])
            end
          end
          if PB._deriv_in_value(ℓ)
            for η in ηs, y in ys, sη in (1, -1)
              v, dv = PB.value_and_deriv(ℓ, sη * η, y)
              @test isfinite(dv) || !isfinite(v)
              @test isfinite(PB.deriv(ℓ, sη * η, y)) || !isfinite(ℓ(sη * η, y))
            end
          end
        end
      end
      # why SquaredHinge and a user's loss keep every test: a finite value at an infinite
      # predictor, and a derivative that overflows where the value does not
      @test !PB._value_traps(PB.SquaredHinge()) && !PB._deriv_in_value(PB.SquaredHinge())
      @test PB.SquaredHinge()(T(Inf), one(T)) === zero(T)
      v, dv = PB.value_and_deriv(PB.SquaredHinge(), zero(T), M)
      @test isfinite(v) && !isfinite(dv)                   # 1 and -2 floatmax
      @test !PB._value_traps(AbsLoss()) && !PB._deriv_traps(AbsLoss()) && !PB._deriv_in_value(AbsLoss())
      @test !PB._pure_everywhere(AbsLoss())
      # the trait properties at extreme loss parameters (bandwidths and δ subnormal, huge, Inf)
      tiny, huge = nextfloat(zero(T)), M / 4
      for ℓ in (PB.Huber(tiny), PB.Huber(huge), PB.Huber(T(Inf)), PB.SmoothCheck(T(0.3), tiny), PB.SmoothCheck(T(0.3), huge),
                PB.SmoothCheck(prevfloat(one(T)), T(Inf)), PB.ConvolutionCheck(T(0.3), tiny), PB.ConvolutionCheck(T(0.7), T(Inf)),
                PB.ConvolutionCheck(T(0.3), huge; kernel = :uniform))
        for η in T[Inf, -Inf, NaN], y in ys
          @test !isfinite(ℓ(η, y))
        end
        for η in ηs, y in ys, sη in (1, -1)
          v, dv = PB.value_and_deriv(ℓ, sη * η, y)
          @test isfinite(dv) || !isfinite(v)
        end
      end
    end

    # a user's loss that throws where its weight is zero: no loop calls it there
    fz = GLMLoss(ThrowsBelowZero(), [-1.0, 0.5], [1.0 0; 0 1]; weights = [0.0, 2.0])
    @test fz([0.25, 1.0]) == 2 * 0.125
    gz = zeros(2)
    @test value_and_gradient!(gz, fz, [0.25, 1.0]) == 2 * 0.125 && gz == [0, 1]
    @test gradient!(zeros(2), fz, [0.25, 1.0]) == [0, 1]
    # and not in the recovery either: a predictor that overflows to Inf (exactly 0) at a weighted row
    M = floatmax(Float64)
    fz = GLMLoss(ThrowsBelowZero(), [0.5, -1.0], [M -M/2 -M; 0 0 1]; weights = [1.0, 0.0])
    β = [2.0, 2.0, 1.0]
    @test fz(β) == 0.125
    gz = zeros(3)
    @test value_and_gradient!(gz, fz, β) == 0.125 && gz == [-M / 2, M / 4, M / 2]
    @test gradient!(zeros(3), fz, β) == [-M / 2, M / 4, M / 2]

    # a predictor that overflows to ±Inf where the exact one is 0 (row [M, -M/2, -M] or its negative,
    # β = [2, 2, 1]: the products 2M = Inf, -M, -M): a finite ordinary value (SquaredHinge) or
    # derivative (Poisson, Logistic, Huber) at the wrong predictor must still be recomputed
    for T in (Float32, Float64)
      M = floatmax(T)
      β = T[2, 2, 1]
      Xp, Xn = T[M -M/2 -M; 0 0 0.5], T[-M M/2 M; 0 0 0.5]
      @test isinf((Xp * β)[1]) && isinf((Xn * β)[1])      # the case needs the overflow, not NaN
      # SquaredHinge, y = 1: ℓ(0) = 1 and ℓ'(0) = -2, ℓ(0.5) = 1/4 and ℓ'(0.5) = -1, weights 1/4 and 1
      f = GLMLoss(PB.SquaredHinge(), T[1, 1], Xp; weights = T[0.25, 1])
      @test f(β) === T(0.5)
      g1, g2 = zeros(T, 3), zeros(T, 3)
      @test value_and_gradient!(g1, f, β) === T(0.5)
      gradient!(g2, f, β)
      @test g1 == g2 == T[-M / 2, M / 4, M / 2]
      # Poisson, y = 0, at the computed -Inf: ℓ'(0) = 1 (the ordinary e^-Inf = 0); ℓ'(0.5) = e^0.5
      f = GLMLoss(PB.Poisson(), T[0, 0], Xn)
      g1, g2 = zeros(T, 3), zeros(T, 3)
      gradient!(g1, f, β)
      value_and_gradient!(g2, f, β)
      @test g1 == g2 == T[-M, M / 2, M + exp(T(0.5)) / 2]
      @test f(β) === one(T) + exp(T(0.5))
      # Logistic, y = 0, at the computed Inf: ℓ'(0) = 1/2 (the ordinary 1); Huber(1), y = 0.25 at η = 0:
      # ℓ'(0) = -1/4 (the ordinary δ = 1)
      for (ℓ, y, d1) in ((PB.Logistic(), T[0, 0], T(0.5)), (PB.Huber(one(T)), T[0.25, 0], T(-0.25)))
        f = GLMLoss(ℓ, y, Xp)
        g1, g2 = zeros(T, 3), zeros(T, 3)
        gradient!(g1, f, β)
        value_and_gradient!(g2, f, β)
        d2 = PB.deriv(ℓ, T(0.5), y[2])
        @test g1 == g2 == T[M * d1, -M / 2 * d1, -M * d1 + T(0.5) * d2]
      end
    end

    # random small problems with predictors that overflow (to ±Inf, or NaN through Inf - Inf),
    # logistic and Poisson tails, zero weights, weights above 1 and extreme responses: the loops
    # return what the recovery, run on every call, returns
    same(a, b) = isequal(a, b) || a == b
    rng = Random.Xoshiro(3580)
    for T in (Float32, Float64), spec in (specs..., (name = "user", make = T -> AbsLoss())), trial in 1:150
      M = floatmax(T)
      n, p = 6, 3
      ℓ = spec.make(T)
      pool = T[1, -1, 0.5, 30, -30, T === Float32 ? 100 : 800, T === Float32 ? -100 : -800, M / 2, -M / 2, M, -M]
      X = [rand(rng) < 0.6 ? T(randn(rng)) : rand(rng, pool) for _ in 1:n, _ in 1:p]
      β = T[rand(rng, (1, -1, 2, -2, 0.5)) for _ in 1:p]
      y = ℓ isa PB.Logistic ? T.(rand(rng, 0:1, n)) : ℓ isa PB.Poisson ? T.(rand(rng, (0, 1, 3), n)) :
          ℓ isa PB.SquaredHinge ? T.(rand(rng, (-1, 1), n)) : T[rand(rng) < 0.8 ? T(randn(rng)) : rand(rng, (M, -M, T(1e30))) for _ in 1:n]
      wpool = T[0, 0.3, 1, 3, T(1e20)]
      kw = rand(rng) < 0.25 ? (;) : (weights = [rand(rng, wpool) for _ in 1:n],)
      rand(rng) < 0.3 && (kw = (; kw..., offset = [rand(rng) < 0.7 ? zero(T) : rand(rng, pool) for _ in 1:n]))
      f = GLMLoss(ℓ, y, X; kw...)
      v1 = f(β)
      g1 = gradient!(zeros(T, p), f, β)
      g2 = zeros(T, p)
      v2 = value_and_gradient!(g2, f, β)
      fr = GLMLoss(ℓ, y, X; kw...)      # the recovery rewrites the buffers: a fresh copy
      PB._linear_predictor!(fr, β)
      vr = PB._glm_recover!(fr, β, Val(true), Val(true))
      gr = zeros(T, p)
      mul!(gr, transpose(fr.X), fr.d)
      @test same(v1, vr) && same(v2, vr)
      @test all(same.(g1, gr)) && all(same.(g2, gr))
    end
  end
end

# the device path: JLArrays with scalar indexing disallowed (as in value_overflow_data.jl)
@testset "GLMLoss values that leave the range on JLArrays" begin
  for T in (Float32, Float64)
    F32 = T === Float32
    M = floatmax(T)
    lo, hi = F32 ? T(1e-35) : T(1e-300), F32 ? T(1e35) : T(1e300)
    for dc in devcases(T)
      spec, y, β = dc[1], dc[2], dc[4]
      c, f = device_case(T, dc)
      ref = reference(c)
      βd = JLArray(copy(β))
      @testset "$(spec.name), $T, n = $(length(y))" begin
        @test value_status(c, f(βd), ref) === :correct
        gd = JLArray(zeros(T, length(β)))
        @test value_status(c, value_and_gradient!(gd, f, βd), ref) === :correct
        @test gradient_status(c, Array(gd), ref) === :correct
        gd2 = JLArray(zeros(T, length(β)))
        gradient!(gd2, f, βd)
        @test gradient_status(c, Array(gd2), ref) === :correct
      end
    end
  end
end

@testset "the combined tests of a term: NaN exactly when a test fails" begin
  # (0 ⋅ η) ⋅ p and the lift select decide as the separate tests: the mark is NaN
  # exactly when η or p is not finite (where tested) or a lost value is lifted
  for T in (Float64, Float32)
    xs = T[0, -0.0, 1, -3, floatmax(T), -floatmax(T), nextfloat(zero(T)), Inf, -Inf, NaN]
    M = floatmax(T)
    for ℓ in (PB.Squared(), PB.Huber(1.0), PB.Logistic(), PB.Poisson(), PB.SquaredHinge(),
              PB.SmoothCheck(0.3, 0.2), PB.ConvolutionCheck(0.3, 0.2), ThrowsBelowZero())
      # a mark is NaN, or a zero that leaves the accumulator finite
      mark_ok(m, nan) = isnan(m) == nan && (nan || iszero(m))
      vs = T[1, floatmin(T) / 4, 0]
      bad = []   # (mark, inputs) that disagree, one @test per loss and type
      for η in xs, p in xs, w in T[0.5, 1, 2, M], v in vs, dv in vs, y in T[0, 1, 2]
        vlift = PB._value_lifts(ℓ, w) & PB._lost_value(ℓ, v, η, y)
        dlift = (w > 1) & PB._lost_deriv(ℓ, dv, η, y)
        ηbad, pbad = !isfinite(η), !isfinite(p)
        mark_ok(PB._value_mark(ℓ, w, v, η, y), vlift | (ηbad & !PB._value_traps(ℓ))) ||
          push!(bad, (:value, η, w, v, y))
        mark_ok(PB._gradient_mark(ℓ, w, dv, p, η, y), dlift | pbad | (ηbad & !PB._deriv_traps(ℓ))) ||
          push!(bad, (:gradient, η, p, w, dv, y))
        for unweighted in (true, false)
          dtest = !(unweighted & PB._deriv_in_value(ℓ) & PB._value_traps(ℓ))
          mark_ok(PB._value_deriv_mark(ℓ, w, v, dv, p, η, y, unweighted),
                  vlift | dlift | (ηbad & !PB._value_traps(ℓ)) | (dtest & pbad)) ||
            push!(bad, (:value_and_gradient, unweighted, η, p, w, v, dv, y))
        end
      end
      @test isempty(bad)
    end
  end
end
# SmoothCheck in forms whose intermediates stay in range (review of 5344abd9): the value
# and derivative against the formula in BigFloat, at extreme and ordinary parameters
function smoothcheck_reference(τ, h, u)
  τ, h, u = big(τ), big(h), big(u)
  u >= h * τ && return τ * u - h * τ^2 / 2, -τ
  u <= -h * (1 - τ) && return (τ - 1) * u - h * (1 - τ)^2 / 2, 1 - τ
  u^2 / (2h), -u / h
end
function smoothcheck_cases(::Type{T}) where {T}
  F32 = T === Float32
  [  # (τ, h, u): the branch, and the intermediate that left the range before
    (T(0.5), F32 ? T(1e-15) : T(1e-150), F32 ? T(1e-22) : T(1e-160)),     # central: u² underflowed
    (T(0.3), floatmax(T) / 2, F32 ? T(1e10) : T(1e100)),                  # central: 2h overflowed
    (T(0.3), floatmax(T), F32 ? T(1e18) : T(1e150)),                      # central, h = floatmax
    (F32 ? T(1e-30) : T(1e-200), F32 ? T(1e30) : T(1e300), F32 ? T(2) : T(2e100)),   # upper: τ² underflowed
    (F32 ? T(1e-30) : T(1e-200), floatmax(T), F32 ? T(1e9) : T(1e110)),   # upper, h = floatmax
    (prevfloat(one(T)), floatmax(T) / 4, -floatmax(T) / 2),               # lower, 1 - τ one ulp
    (T(0.3), one(T), T(0.2)), (T(0.3), one(T), T(-0.5)), (T(0.3), one(T), T(0.05)),  # ordinary
    (T(0.9), T(2), T(-3)), (T(0.1), T(1e-3), T(7)),
  ]
end
@testset "SmoothCheck: value and derivative without intermediates out of range" begin
  setprecision(BigFloat, 512) do
    for T in (Float32, Float64), (τ, h, u) in smoothcheck_cases(T)
      ℓ = PB.SmoothCheck(τ, h)
      vr, dr = smoothcheck_reference(τ, h, u)
      @test isfinite(vr) && T(vr) > floatmin(T)
      # direct scalar evaluation (η = 0, y = u)
      v, dv = PB.value_and_deriv(ℓ, zero(T), u)
      @test abs(big(v) - vr) <= 3eps(T) * abs(vr)
      @test abs(big(dv) - dr) <= 3eps(T) * abs(dr)
      @test ℓ(zero(T), u) === v && PB.deriv(ℓ, zero(T), u) === dv
      # through GLMLoss: y = 0, X = 1, β = -u, so that the residual is u exactly
      for wv in (nothing, zero(T), T(0.5), one(T), T(2))
        wvec = wv === nothing ? nothing : T[wv]
        wb = wv === nothing ? big(1) : big(wv)
        for dev in (false, true)
          A = dev ? JLArray : identity
          f = GLMLoss(ℓ, A(T[0]), A(ones(T, 1, 1)); weights = wvec === nothing ? nothing : A(wvec))
          β = A(T[-u])
          g1, g2 = A(zeros(T, 1)), A(zeros(T, 1))
          val, vg = f(β), value_and_gradient!(g1, f, β)
          gradient!(g2, f, β)
          if iszero(wb)
            @test val === zero(T) && vg === zero(T) && all(iszero, Array(g1)) && all(iszero, Array(g2))
          else
            @test abs(big(val) - wb * vr) <= 3eps(T) * abs(wb * vr) && vg === val
            @test abs(big(Array(g1)[1]) - wb * dr) <= 3eps(T) * abs(wb * dr)   # X = 1: the gradient is w ℓ'(η)
            @test Array(g2) == Array(g1)
          end
        end
      end
    end
  end
end

# Inactive observations (weight zero) take no part, whatever their row holds (review of
# 5344abd9): the oracles equal those of the problem without them, on the CPU and on JLArrays
@testset "inactive observations with non-finite rows do not disable the recovery" begin
  for T in (Float32, Float64)
    F32 = T === Float32
    tiny, huge, big_b = F32 ? T(1e-30) : T(1e-300), F32 ? T(1e35) : T(1e300), F32 ? T(100) : T(1000)
    for (ℓ, yact, bact) in ((PB.Poisson(), T(0), big_b),            # exp(η) overflows before w: recovery
                            (PB.Logistic(), T(1), F32 ? T(120) : T(800)),   # the tail underflows, w lifts it
                            (PB.Squared(), T(0.5), T(2)),           # ordinary: no recovery needed
                            (PB.SmoothCheck(0.3, 0.2), T(0), T(3)),
                            (AbsLoss(), T(0), T(2)))                # a user's loss: the branch loops
      wact = ℓ isa PB.Poisson ? tiny : ℓ isa PB.Logistic ? huge : T(1.5)
      # the active problem alone: rows 2 and 3, columns 1 and 2
      Xa, ya, wa, oa = T[1 0.5; 0.25 1], T[yact, yact], T[wact, wact], T[0, 0]
      β = T[bact, 0]
      for bad in (:y_nan, :y_inf, :offset_nan, :offset_inf, :x_nan, :x_inf)
        X = vcat(T[0 0], Xa); y = vcat(T[0], ya); w = vcat(T[0], wa); o = vcat(T[0], oa)
        bad === :y_nan && (y[1] = T(NaN)); bad === :y_inf && (y[1] = T(Inf))
        # the constructor requires a finite offset: a non-finite one only by mutation
        obad = copy(o)
        bad === :offset_nan && (obad[1] = T(NaN)); bad === :offset_inf && (obad[1] = T(-Inf))
        bad === :x_nan && (X[1, 2] = T(NaN)); bad === :x_inf && (X[1, 1] = T(Inf))
        for dev in (false, true)
          A = dev ? JLArray : identity
          fa = GLMLoss(ℓ, A(ya), A(Xa); weights = A(wa), offset = A(oa))
          f = GLMLoss(ℓ, A(y), A(X); weights = A(w), offset = A(copy(o)))   # A = identity would share o
          copyto!(f.offset, A(obad))
          ga, g1, g2 = A(zeros(T, 2)), A(zeros(T, 2)), A(zeros(T, 2))
          va = value_and_gradient!(ga, fa, A(β))
          @testset "$(nameof(typeof(ℓ))) $T $bad $(dev ? "JLArray" : "CPU")" begin
            @test isfinite(va) && all(isfinite, Array(ga))
            @test f(A(β)) ≈ va rtol = 4eps(T)
            @test value_and_gradient!(g1, f, A(β)) ≈ va rtol = 4eps(T)
            @test Array(g1) ≈ Array(ga) rtol = 4eps(T)
            gradient!(g2, f, A(β))
            @test Array(g2) ≈ Array(ga) rtol = 4eps(T)
          end
        end
        # the weight is read at each call: activated, the non-finite row counts (E2)
        f = GLMLoss(ℓ, y, X; weights = copy(w), offset = copy(o))
        copyto!(f.offset, obad)
        f.weights[1] = one(T)
        g = zeros(T, 2)
        @test !isfinite(f(β)) && !isfinite(value_and_gradient!(g, f, β))
        f.weights[1] = zero(T)
        @test isfinite(value_and_gradient!(g, f, β))
      end
    end
    # an active non-finite response still gives the ordinary non-finite result (E2)
    f = GLMLoss(PB.Poisson(), T[NaN, 0], T[0 1; 1 0]; weights = T[1, tiny])
    @test isnan(f(T[big_b, 1]))
    # the ordinary path, inactive observations included, does not allocate
    f = GLMLoss(PB.Logistic(), T[0, 1, 1], T[1 0.5; 0.25 1; 2 1]; weights = T[0, 1, 2])
    β, g = T[0.3, -0.2], zeros(T, 2)
    @test allocated_value(f, β) == 0 && allocated_vg(g, f, β) == 0 && allocated_grad(g, f, β) == 0
  end
end
# The inactive-observation semantics over positions, several rows, all weights zero, sparse X
# and a loss that throws when called on an inactive row (the reference is the problem with the
# inactive rows removed)
@testset "inactive observations: positions, several rows, sparse X, a throwing loss" begin
  for T in (Float32, Float64)
    F32 = T === Float32
    tiny, big_b = F32 ? T(1e-30) : T(1e-300), F32 ? T(100) : T(1000)
    # two active rows: one overflows exp(η) before the tiny weight (recovery), one ordinary
    Xa, ya, wa = T[1 0.5; 0.25 1], T[0, 1], T[tiny, 2]
    β = T[big_b, -1]
    inactive_rows = (T[NaN Inf], T[-Inf NaN], T[Inf -Inf])
    inactive_y = (T(NaN), T(Inf), T(-Inf))
    for layout in (:before, :between, :after, :several)
      rows = layout === :before ? [0, 1, 2] : layout === :between ? [1, 0, 2] :
             layout === :after ? [1, 2, 0] : [0, 1, 0, 0, 2, 0]
      X = zeros(T, length(rows), 2); y = zeros(T, length(rows)); w = zeros(T, length(rows))
      bad = 0
      for (i, r) in enumerate(rows)
        if r == 0
          bad += 1
          X[i, :] = inactive_rows[mod1(bad, 3)]; y[i] = inactive_y[mod1(bad, 3)]
        else
          X[i, :] = Xa[r, :]; y[i] = ya[r]; w[i] = wa[r]
        end
      end
      for sparse_x in (false, true), ℓ in (PB.Poisson(), ThrowsBelowZero())
        ℓ isa ThrowsBelowZero && (y2 = copy(y); y2[w .== 0] .= -1)   # a call on an inactive row throws
        yy = ℓ isa ThrowsBelowZero ? y2 : y
        XX = sparse_x ? sparse(X) : X
        XXa = sparse_x ? sparse(Xa) : Xa
        fa = GLMLoss(ℓ, ya, XXa; weights = wa)
        f = GLMLoss(ℓ, yy, XX; weights = copy(w))
        ga, g1, g2 = zeros(T, 2), zeros(T, 2), zeros(T, 2)
        va = value_and_gradient!(ga, fa, β)
        @testset "$T $layout sparse=$sparse_x $(nameof(typeof(ℓ)))" begin
          @test isfinite(va) && all(isfinite, ga)
          @test f(β) ≈ va rtol = 4eps(T)
          @test value_and_gradient!(g1, f, β) ≈ va rtol = 4eps(T)
          @test g1 ≈ ga rtol = 4eps(T)
          gradient!(g2, f, β)
          @test g2 ≈ ga rtol = 4eps(T)
        end
      end
      # JLArrays (dense X)
      fa = GLMLoss(PB.Poisson(), JLArray(ya), JLArray(Xa); weights = JLArray(wa))
      f = GLMLoss(PB.Poisson(), JLArray(y), JLArray(X); weights = JLArray(w))
      ga, g1 = JLArray(zeros(T, 2)), JLArray(zeros(T, 2))
      va = value_and_gradient!(ga, fa, JLArray(β))
      @test f(JLArray(β)) ≈ va rtol = 4eps(T)
      @test value_and_gradient!(g1, f, JLArray(β)) ≈ va rtol = 4eps(T)
      @test Array(g1) ≈ Array(ga) rtol = 4eps(T)
      gradient!(g1, f, JLArray(β))
      @test Array(g1) ≈ Array(ga) rtol = 4eps(T)
    end
    # the review's example (comment 6646213238): the active value and gradient w e^η, in BigFloat
    for dev in (false, true)
      A = dev ? JLArray : identity
      wt, bt = F32 ? (T(1e-30), T(100)) : (T(1e-300), T(1000))
      f = PB.PoissonLoss(A(T[NaN, 0]), A(reshape(T[0, 1], 2, 1)); weights = A(T[0, wt]))
      ref = T(big(wt) * exp(big(bt)))
      g = A(zeros(T, 1))
      @test f(A(T[bt])) ≈ ref rtol = 4eps(T)
      @test value_and_gradient!(g, f, A(T[bt])) ≈ ref rtol = 4eps(T)
      @test Array(g)[1] ≈ ref rtol = 4eps(T)
      gradient!(g, f, A(T[bt]))
      @test Array(g)[1] ≈ ref rtol = 4eps(T)
    end
    # all weights zero: value 0, gradient 0, whatever the rows hold, and the loss is not called
    for ℓ in (PB.Poisson(), ThrowsBelowZero()), sparse_x in (false, true)
      X = T[NaN 1; Inf -Inf]; f = GLMLoss(ℓ, T[-1, NaN], sparse_x ? sparse(X) : X; weights = T[0, 0])
      g = fill(T(7), 2)
      @test f(β) === zero(T)
      @test value_and_gradient!(g, f, β) === zero(T) && g == zeros(T, 2)
      fill!(g, T(7)); gradient!(g, f, β)
      @test g == zeros(T, 2)
    end
  end
end
# The witnesses of the numerical review of 1d3677e8 (comment 6647339330)
@testset "inactive rows with a sparse β, weights that round to zero, SmoothCheck thresholds" begin
  setprecision(BigFloat, 2560) do
    for T in (Float32, Float64)
      # 1. a zero coefficient of a sparse β skips the column of the inactive non-finite entry,
      # so that η is finite there: X'd must still leave that row out
      for ℓ in (PB.Squared(), ThrowsBelowZero()), bad in (T(NaN), T(Inf), T(-Inf)),
          mk in (identity, sparse), bmk in (SparseIterate, sparse), wz in (zero(T), -zero(T))
        X = mk(T[1 0; 0 bad])
        f = GLMLoss(ℓ, T[0, 0], X; weights = T[1, wz])
        b = bmk(T[2, 0])
        g1, g2 = zeros(T, 2), zeros(T, 2)
        @test f(b) == 2 && value_and_gradient!(g1, f, b) == 2
        gradient!(g2, f, b)
        @test g1 == T[2, 0] && g2 == T[2, 0]
      end
      # JLArrays, dense β exposing the row, and the same reduced reference
      f = GLMLoss(PB.Squared(), JLArray(T[0, 0]), JLArray(T[1 0; 0 NaN]); weights = JLArray(T[1, 0]))
      g = JLArray(zeros(T, 2))
      @test value_and_gradient!(g, f, JLArray(T[2, 0])) == 2 && Array(g) == T[2, 0]
    end
    # 2. a positive weight that is zero in the working precision is inactive everywhere
    for dev in (false, true)
      A = dev ? JLArray : identity
      f = PB.PoissonLoss(A(Float32[0, NaN]), A(Float32[1 0; 0 1]); weights = A(Float64[1e-30, 1e-300]))
      b, g = A(Float32[100, 0]), A(zeros(Float32, 2))
      ref = Float32(big(1e-30) * exp(big(100)))
      @test f(b) ≈ ref rtol = 4eps(Float32)
      @test value_and_gradient!(g, f, b) ≈ ref rtol = 4eps(Float32)
      @test Array(g)[1] ≈ ref rtol = 4eps(Float32)
      @test Array(g)[2] == 0
      gradient!(g, f, b)
      @test Array(g)[1] ≈ ref rtol = 4eps(Float32)
      @test Array(g)[2] == 0
      f = GLMLoss(PB.Squared(), A(Float32[0, 0]), A(Float32[1 0; 0 NaN]); weights = A(Float64[1, 1e-300]))
      g = A(zeros(Float32, 2))
      @test value_and_gradient!(g, f, A(Float32[2, 0])) == 2 && Array(g) == Float32[2, 0]
      gradient!(g, f, A(Float32[2, 0]))
      @test Array(g) == Float32[2, 0]
    end
    # 3. SmoothCheck: the branch is not misplaced by an underflowing hτ, and a lifted
    # subnormal value or derivative is recomputed with its linear correction
    for T in (Float32, Float64)
      s = nextfloat(zero(T))
      @test PB.value_and_deriv(PB.SmoothCheck(T(0.5), s), zero(T), zero(T)) == (0, 0)   # exact fit
      @test PB.value_and_deriv(PB.SmoothCheck(T(0.3), 4s), zero(T), s)[2] == T(-0.25)
      w = floatmax(T)
      for (τ, h, u) in ((T(0.9), s, s), (T(0.1), s, -s), (4s, T(0.3), s), (T(0.5), 8s, 2s))
        ref_v, ref_d = smoothcheck_reference(τ, h, u)
        for dev in (false, true)
          A = dev ? JLArray : identity
          f = GLMLoss(PB.SmoothCheck(τ, h), A(T[0]), A(ones(T, 1, 1)); weights = A(T[w]))
          b, g1, g2 = A(T[-u]), A(zeros(T, 1)), A(zeros(T, 1))
          v1, v2 = f(b), value_and_gradient!(g1, f, b)
          gradient!(g2, f, b)
          rv, rd = big(w) * ref_v, big(w) * ref_d
          # (the weighted value of (4s, 0.3, s) is below the smallest subnormal: 0)
          @test v1 == v2 && abs(big(v1) - rv) <= 4eps(T) * abs(rv) + big(nextfloat(zero(T))) / 2
          @test abs(big(Array(g1)[1]) - rd) <= 4eps(T) * abs(rd) && Array(g2) == Array(g1)
        end
      end
    end
  end
end
end # !BASELINE

BASELINE && baseline()

end
