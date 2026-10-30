module WeightStepsTest
using Test, ProximalBase, Random
const PB = ProximalBase

# Frozen coefficient and entry formulas (546dc86, and Berhu since #289), independent of the package kernels.
entry(w::Nothing, i) = 1
entry(w::AbstractArray, i) = w[i]
shrink_ref(v::T, c::T) where {T<:AbstractFloat} = max(v - c, zero(T)) + min(v + c, zero(T))
function coefficient(s, w)
  p = s * w
  (s > zero(s)) & isfinite(s) && return p
  ifelse(isnan(p) & !isnan(s), oftype(p, w), p)
end
coefficient(γ, λ, w) = coefficient(γ * λ, w)

function reference_entry(f::ElasticNet, v::T, i::Integer, γ::Real) where {T<:Real}
  S = float(T)
  shrink_ref(v, convert(S, coefficient(γ, f.λ1, entry(f.w, i)))) / (1 + convert(S, γ * f.λ2))
end

# as since #289: an identity prox for a zero step, level or finite weight
function reference_entry(f::Berhu, v::T, i::Integer, γ::Real) where {T<:Real}
  S = float(T)
  step, λ, w = S(γ), S(f.λ), S(entry(f.w, i))
  !isinf(w) && (iszero(step) || iszero(λ) || iszero(w)) && return S(v)
  c = coefficient(step, λ, w)
  M, t = S(f.M), abs(S(v))
  t <= M + c ? shrink_ref(S(v), c) : S(v) * M / (M + c)
end

function reference_entry(f::CheckLoss, v::T, i::Integer, γ::Real) where {T<:Real}
  S = float(T)
  w = entry(f.w, i)
  s, τ, u = convert(S, coefficient(γ, w)), convert(S, f.τ), S(v)
  u > s * τ ? u - s * τ : (u < -s * (1 - τ) ? u + s * (1 - τ) : zero(S))
end

function reference_entry(f::EpsInsensitiveLoss, v::T, i::Integer, γ::Real) where {T<:Real}
  S = float(T)
  w = entry(f.w, i)
  s, ε, u = convert(S, coefficient(γ, w)), convert(S, f.ε), S(v)
  a = abs(u)
  a <= ε ? u : (a <= ε + s ? copysign(ε, u) : u - copysign(s, u))
end

function reference_entry(f::HingeLoss, v::T, i::Integer, γ::Real) where {T<:Real}
  S = float(T)
  y = convert(S, entry(f.y, i))
  w = entry(f.w, i)
  s = convert(S, coefficient(γ, w)) * y^2
  u = y * S(v)
  (u >= 1 ? u : (u < 1 - s ? u + s : one(S))) / y
end

function reference_entry(f::CappedL1, v::T, i::Integer, γ::Real) where {T<:Real}
  S = float(T)
  c = coefficient(S(γ), S(f.λ), S(entry(f.w, i)))
  a, t = S(f.a), abs(S(v))
  z1 = min(max(t - c, zero(S)), a)          # the ℓ1 piece |z| ≤ a: soft thresholding, clipped at a
  z2 = max(t, a)                             # the constant piece |z| ≥ a
  # γ times the objective at each; a tie goes to the smaller entry, and an
  # infinite coefficient (z1 = 0, where h1 is Inf⋅0) to z1
  h1 = c * z1 + (z1 - t)^2 / 2
  h2 = c * a + (z2 - t)^2 / 2
  sign(S(v)) * ifelse(isinf(c) | (h1 <= h2), z1, z2)
end

function families(T, w; λ=T(0.3))
  (ElasticNet(λ, T(0.4), w), Berhu(λ, one(T), w), CheckLoss(T(0.3), w),
   EpsInsensitiveLoss(T(0.3), w), HingeLoss(nothing, w), CappedL1(λ, one(T), w))
end
samebits(a::T, b::T) where {T} = (isnan(a) && isnan(b)) ||
  reinterpret(T === Float32 ? UInt32 : UInt64, a) == reinterpret(T === Float32 ? UInt32 : UInt64, b)
reference(f, x, γ) = [reference_entry(f, x[i], i, γ[i]) for i in eachindex(x)]
function check_case(f, x, γ)
  expected = reshape(eltype(x).(reference(f, x, γ)), size(x))
  y = similar(x)
  @test prox!(y, f, x, γ) === y
  @test all(samebits.(y, expected))
  z = copy(x)
  @test prox!(z, f, z, γ) === z
  @test all(samebits.(z, expected))
end

# A reference separable function forces the original entrywise and wrapper paths.
struct Frozen{F} <: PB.SeparableFunction
  f::F
end
PB.prox_entry(f::Frozen, v, i, γ) = reference_entry(f.f, v, i, γ)
PB.isconvex(::Type{<:Frozen}) = true
PB.iszeropreserving(f::Frozen) = PB.iszeropreserving(f.f)
PB._checkparams(f::Frozen, x) = PB._checkparams(f.f, x)
PB._conjugate_isindicator(::Type{Frozen{F}}) where {F} = PB._conjugate_isindicator(F)

@testset "weighted array steps" begin
  rng = MersenneTwister(252)
  for T in (Float32, Float64)
    x = vcat(randn(rng, T, 24), T[0, -0.0, nextfloat(zero(T)), -nextfloat(zero(T)), Inf, -Inf, NaN, floatmax(T)])
    w = T(0.5) .+ rand(rng, T, length(x))
    ordinary = T(0.2) .+ rand(rng, T, length(x))
    for weights in (w, nothing, [i == 1 ? zero(T) : i == 2 ? T(Inf) : w[i] for i in eachindex(w)])
      for f in families(T, weights)
        check_case(f, x, ordinary)
        for bad in (zero(T), -one(T), nextfloat(zero(T)), T(Inf), T(NaN), floatmax(T))
          steps = copy(ordinary); steps[3] = bad
          check_case(f, x, steps)
        end
        X = reshape(copy(x), 4, 8); G = reshape(copy(ordinary), 4, 8)
        check_case(f, X, G)
        check_case(f, x, Float64.(ordinary))
      end
    end
    for λ in (zero(T), T(Inf), T(0.5), T(2))
      for f in families(T, w; λ)
        for steps in (fill(nextfloat(zero(T)), length(x)), fill(floatmax(T), length(x)))
          check_case(f, x, steps)
        end
      end
    end
    # Mixed factors must retain their conversion order.
    for f in (families(Float64, Float64.(w))..., families(T, Float64.(w))...)
      check_case(f, x, ordinary)
    end
    for f in families(T, T[])
      check_case(f, T[], T[])
    end
    for f in families(T, w)
      z = randn(rng, T, length(x))
      @test all(samebits.(prox(Scaled(f, T(2)), z, ordinary), prox(Scaled(Frozen(f), T(2)), z, ordinary)))
      if !(f isa CappedL1)
        a = prox(Conjugate(f), z, ordinary)
        b = prox(Conjugate(Frozen(f)), z, ordinary)
        if f isa CheckLoss
          # Direct clipping avoids Moreau cancellation; compare both to the box.
          setprecision(BigFloat, 4096) do
            ref = [clamp(BigFloat(z[i]), -BigFloat(entry(f.w, i)) * (1 - BigFloat(f.τ)),
                         BigFloat(entry(f.w, i)) * BigFloat(f.τ)) for i in eachindex(z)]
            direct, moreau = abs.(BigFloat.(a) .- ref), abs.(BigFloat.(b) .- ref)
            @test all(direct .<= 4BigFloat(eps(T)) .* abs.(ref) .+ 2BigFloat(nextfloat(zero(T))))
            @test maximum(direct) <= maximum(moreau)
            @test sum(direct) <= sum(moreau)
          end
        else
          @test all(samebits.(a, b))
        end
      end
    end
    # Cross every exceptional input with zero and infinite weights.
    for weight in (zero(T), T(Inf)), f in families(T, fill(weight, length(x)))
      check_case(f, x, ordinary)
    end
    for f in families(T, w)
      expected = reference(f, x, ordinary)
      for z in (SparseIterate(x), AtomIterate(x, GroupNormL2(one(T), [1:length(x)])))
        y = copy(z)
        reference_y = copy(z)
        prox!(reference_y, Frozen(f), z, ordinary)
        prox!(y, f, z, ordinary)
        @test all(samebits.(Array(y), Array(reference_y)))
      end
      y = Float64.(x)
      prox!(y, f, x, ordinary)
      @test all(samebits.(y, Float64.(expected)))
    end
    for λ2 in (zero(T), T(Inf), nextfloat(zero(T)))
      check_case(ElasticNet(T(0.3), λ2, w), x, ordinary)
    end
    labels = T[isodd(i) ? -2 : 3 for i in eachindex(x)]
    check_case(HingeLoss(labels, w), x, ordinary)
    check_case(HingeLoss(Int.(labels), w), x, ordinary)
  end
  @test_throws DimensionMismatch prox!(zeros(2), ElasticNet(0.3, 0.4, ones(2)), ones(2), ones(3))
  @test_throws DimensionMismatch prox!(zeros(3), ElasticNet(0.3, 0.4, ones(2)), ones(2), ones(2))
end
end
