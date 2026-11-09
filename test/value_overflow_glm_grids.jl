module ValueOverflowGLMGridsTest

# The Float32 storage cases of "GLMLoss values and gradients that leave the range" that value_overflow_glm.jl
# leaves to its Float64 pass and to the dense Float32 cases: for Logistic, Huber(1.0), SmoothCheck(0.3, 0.5) and the
# two Conv losses (epanechnikov and uniform), the cases with a sparse X, a SparseIterate β, an AtomIterate β and a
# SparseIterate β with zero entries on a sparse X, each with the weights 1.5 and 2^-149 of the predictor table; and
# for Poisson the first three of those four (the cases named "sparse X, cancelling row", "SparseIterate β" and "AtomIterate β").
# value_overflow_glm.jl keeps the Float64 pass, the dense Float32 cases of these losses, the SparseIterate β with
# zero entries of Poisson, and all the Float32 cases of Squared and SquaredHinge. The loss references, the helpers, `GCase`, `reference`, the status functions, `test_case` and the
# lines that build these cases are copied from it; the case table keeps only the predictor loop of those cases.
# The fixtures are the same as in value_overflow_glm.jl. No case here draws random numbers, and the generators of
# value_overflow_glm.jl are created inside the testsets that use them, so the draws of the cases that stay there
# do not change.

using Test
using ProximalBase
using SparseArrays

const PB = ProximalBase

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

const LOGISTIC = LossSpec("Logistic", T -> PB.Logistic(), (T, η, y) -> _logistic(η, y),
  (T, η, y) -> η > 0 ? (1 - y) - (η > 1e9 ? zero(η) : (e = exp(-η); e / (1 + e))) :
               (η < -1e9 ? -y : (e = exp(η); e / (1 + e) - y)))
const POISSON = LossSpec("Poisson", T -> PB.Poisson(), (T, η, y) -> _bexp(η) - (iszero(y) && abs(η) > 1e9 ? zero(η) : y * η),
  (T, η, y) -> _bexp(η) - y)
HUBER(δ) = LossSpec("Huber($δ)", T -> PB.Huber(T(δ)),
  (T, η, y) -> (d = big(T(δ)); r = η - y; abs(r) <= d ? r^2 / 2 : d * (abs(r) - d / 2)),
  (T, η, y) -> (d = big(T(δ)); r = η - y; abs(r) <= d ? r : d * sign(r)))
SMOOTHCHECK(τ, h) = LossSpec("SmoothCheck($τ,$h)", T -> PB.SmoothCheck(T(τ), T(h)),
  (T, η, y) -> _moreau(big(T(τ)), big(T(h)), y - η)[1],
  (T, η, y) -> -_moreau(big(T(τ)), big(T(h)), y - η)[2])
CONV(kernel, τ, h) = LossSpec("Conv-$kernel($τ,$h)", T -> PB.ConvolutionCheck(T(τ), T(h); kernel),
  (T, η, y) -> _conv(kernel, big(T(τ)), big(T(h)), y - η)[1],
  (T, η, y) -> -_conv(kernel, big(T(τ)), big(T(h)), y - η)[2])


# T: element type of y, X, weights, offset; B: element type of β (the result is float(B));
# wrap: how β is passed (dense, SparseIterate, AtomIterate); status: :assert, or :info for the
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

# The Float32 storage cases of the predictor table of cases(T) in value_overflow_glm.jl: the loop over the
# losses and the two weights, restricted to the losses of this file and to the four storage cases.
function cases(T)
  M = floatmax(T)
  F32 = T === Float32
  wsub = F32 ? ldexp(one(T), -149) : ldexp(one(T), -1060)   # the smallest weights: powers of two, exact
  cs = GCase[]
  add!(args...; kw...) = push!(cs, GCase(args...; kw...))

  # per loss a representative parameter set and response
  preds = [("Logistic", LOGISTIC, T(0.5)), ("Poisson", POISSON, T(0.5)),
           ("Huber", HUBER(1.0), T(0.5)), ("SmoothCheck", SMOOTHCHECK(0.3, 0.5), T(0.5)),
           ("Conv-epanechnikov", CONV(:epanechnikov, 0.3, 0.5), T(0.5)), ("Conv-uniform", CONV(:uniform, 0.3, 0.5), T(0.5))]
  for (gname, spec, y0) in preds, w in (T(1.5), wsub)
    tag = w == wsub ? ", w tiny" : ""
    # sparse X and a sparse iterate, with two observations
    Xs = sparse(T[M -M 0; 0 0 1])
    add!(gname, "sparse X, cancelling row$tag", T, spec, [y0, y0], Xs, T[2, 2, 5]; w = [w, 2])
    add!(gname, "SparseIterate β$tag", T, spec, [y0], reshape(T[M, -M, 1], 1, 3), T[2, 2, 3]; w = [w], wrap = SparseIterate)
    add!(gname, "AtomIterate β$tag", T, spec, [y0], reshape(T[M, -M, 1], 1, 3), T[2, 2, 3]; w = [w], wrap = b -> AtomIterate(b, [1:3]))
    # the SparseIterate β with zero entries of Poisson stays in value_overflow_glm.jl
    gname == "Poisson" || add!(gname, "SparseIterate β with zero entries, sparse X$tag", T, spec, [y0, y0], Xs, T[2, 2, 0]; w = [w, 2], wrap = SparseIterate)
  end
  cs
end

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

@testset "GLMLoss values and gradients that leave the range" begin
  for T in (Float32,)
    @testset "$T" begin
      for c in cases(T)
        @testset "$(c.group): $(c.name)" begin
          test_case(c)
        end
      end
    end
  end
end

end
