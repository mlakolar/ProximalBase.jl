module ElasticNetZeroLevelsTest

using Test, ProximalBase, LinearAlgebra, SparseArrays
using JLArrays: JLArray
using GPUArraysCore: allowscalar, @allowscalar
using Adapt: adapt

const PB = ProximalBase

# ElasticNet with zero levels and weights at every admitted step (P1, E4; #594). At γ = Inf each part
# is decided from its parameters: a zero level forms no product γ·0. With no active part the map is
# the identity, NaN and ±Inf included; with an active part it is 0 at every finite argument, and a
# NaN or infinite entry gives NaN, as at the base (no joint limit, E1). An infinite weight fixes its
# entry as at every finite step. Finite steps are unchanged. NaN is compared with isequal.

same(a, b) = isequal(a, b)

# The proximal point of γ(λ1 w|z| + λ2 z²/2) at v, from the definition and the rules above. The test
# levels, weights and steps are dyadic, so the threshold and the ridge factor are exact and the one
# rounding is the final division.
function ref_prox(v::T, λ1, λ2, w, γ) where {T<:AbstractFloat}
  iszero(γ) && (isinf(λ1) || isinf(λ2)) && return T(NaN)      # master's 0·Inf at γ = 0, pinned
  if isinf(w)                                   # the entry is fixed at zero; NaN and ±Inf stay NaN
    return isfinite(v) ? zero(T) : T(NaN)
  end
  l1 = λ1 > 0 && w > 0
  l2 = λ2 > 0
  if isinf(γ)
    if !(l1 || l2)                              # the zero function: a copy (-0 gives +0, as at every step)
      return iszero(v) ? zero(T) : v
    end
    isfinite(v) || return T(NaN)
    return l1 || iszero(v) ? zero(T) : copysign(zero(T), v)
  end
  isnan(v) && return v
  iszero(v) && return zero(T)
  c = l1 ? T(γ) * T(λ1) * T(w) : zero(T)
  d = one(T) + T(γ) * T(λ2)
  if isinf(c)
    isfinite(v) || return T(NaN)
    z = zero(T)
  else
    a = abs(v) - c
    z = a > 0 ? copysign(a, v) : zero(T)
  end
  z / d
end
# its slope at v (either side at a kink, as _prox_entry_slope); finite levels
function ref_slope(v::T, λ1, λ2, w, γ) where {T<:AbstractFloat}
  isinf(w) && return zero(T)
  l1 = λ1 > 0 && w > 0
  l2 = λ2 > 0
  isinf(γ) && return !(l1 || l2) && abs(v) > 0 ? one(T) : zero(T)
  c = l1 ? T(γ) * T(λ1) * T(w) : zero(T)
  abs(v) > c ? inv(one(T) + T(γ) * T(λ2)) : zero(T)
end

entries(::Type{T}) where {T} = T[2, -1, 0.25, 0, -0.0, Inf, -Inf, NaN]
const LEVELS = ((0, 0), (0.5, 0), (0, 1), (0.5, 1))
const STEPS = (0, 0.5, 1, Inf)

enet(::Type{T}, λ1, λ2, w) where {T} = w === nothing ? ElasticNet(T(λ1), T(λ2)) : ElasticNet(T(λ1), T(λ2), w)
weightof(w, i) = w === nothing ? 1 : w[i]

# every route on the CPU against the reference
function check_routes(f, x::AbstractVector{T}, γ, want) where {T}
  @test same(prox(f, x, γ), want)
  y = fill(T(7), length(x))
  @test prox!(y, f, x, γ) === y && same(y, want)
  z = copy(x)
  @test prox!(z, f, z, γ) === z && same(z, want)
  @test same([prox_entry(f, x[i], i, γ isa AbstractArray ? γ[i] : γ) for i in eachindex(x)], want)
  y = fill(T(7), length(x))
  v = value_and_prox!(y, f, x, γ)                  # the value at the stored output (O2)
  @test same(y, want) && same(v, f(y))
  z = copy(x)
  v = value_and_prox!(z, f, z, γ)
  @test same(z, want) && same(v, f(z))
end

@testset "ElasticNet levels, weights and steps, $T" for T in (Float32, Float64)
  e = entries(T)
  n = length(e)
  @testset "λ = $((λ1, λ2)), w = $wc" for (λ1, λ2) in LEVELS, wc in (nothing, 0, 0.5, 1, Inf, :mixed)
    if wc === :mixed                              # every entry meets every weight class
      x = repeat(e, inner = 4)
      w = repeat(T[0, 0.5, Inf, 1], outer = n)
    else
      x = e
      w = wc === nothing ? nothing : fill(T(wc), n)
    end
    f = enet(T, λ1, λ2, w)
    for γ in STEPS
      want = [ref_prox(x[i], λ1, λ2, weightof(w, i), γ) for i in eachindex(x)]
      check_routes(f, x, T(γ), want)
      for (i, v) in enumerate(x)
        z, dz = PB._prox_entry_slope(f, v, i, T(γ))
        @test same(z, want[i])
        γ > 0 && @test dz === ref_slope(v, λ1, λ2, weightof(w, i), γ)
      end
    end
    # array steps mixing 0, finite and Inf
    xs = repeat(x, inner = 4)
    γs = repeat(T[0, 0.5, Inf, 1], outer = length(x))
    fs = enet(T, λ1, λ2, w === nothing ? nothing : repeat(w, inner = 4))
    want = [ref_prox(xs[i], λ1, λ2, weightof(fs.w, i), γs[i]) for i in eachindex(xs)]
    check_routes(fs, xs, γs, want)
  end
end

@testset "infinite levels, $T" for T in (Float32, Float64)
  # γ = Inf: an infinite level is an active part; γ = 0 and finite steps keep master's results
  e = entries(T)
  for (λ1, λ2) in ((Inf, 0), (0, Inf), (Inf, 1), (1, Inf)), w in (nothing, T[0], T[1], T[Inf])
    for γ in (0, 1, Inf)
      f = enet(T, λ1, λ2, w === nothing ? nothing : fill(w[1], length(e)))
      want = [ref_prox(v, λ1, λ2, w === nothing ? 1 : w[1], γ) for v in e]
      @test same(prox(f, e, T(γ)), want)
    end
  end
end

@testset "the named cases, $T" for T in (Float32, Float64)
  Inf_ = T(Inf)
  @test same(prox(ElasticNet(T(0), T(0)), T[2, -1], Inf_), T[2, -1])
  @test same(prox(ElasticNet(T(0), T(0)), T[2, -1, NaN], Inf_), T[2, -1, NaN])
  @test same(prox(ElasticNet(T(0), T(0)), T[Inf, -Inf], Inf_), T[Inf, -Inf])
  @test same(prox_entry(ElasticNet(T(0), T(0)), T(2), 1, Inf_), T(2))
  @test same(prox(ElasticNet(T(1), T(0)), T[2, -1], Inf_), T[0, 0])
  @test same(prox(ElasticNet(T(0), T(1)), T[2, -1], Inf_), T[0, -0.0])
  @test same(prox(ElasticNet(T(1), T(0), T[0, 1]), T[2, 3], Inf_), T[2, 0])
  @test same(prox(ElasticNet(T(0), T(0), T[Inf, 1]), T[2, 3], Inf_), T[0, 3])
  @test same(prox(ElasticNet(T(0), T(0)), T[2, 3], T[Inf, 1]), T[2, 3])
  @test same(prox(ElasticNet(T(1), T(1)), T[2, NaN], Inf_), T[0, NaN])
  @test same(prox(ElasticNet(T(1), T(1)), T[Inf, NaN], Inf_), T[NaN, NaN])
  @test same(prox(ElasticNet(T(1), T(0)), T[NaN], Inf_), T[NaN])
  @test same(prox(ElasticNet(T(1), T(1), T[Inf, Inf]), T[NaN, -Inf], Inf_), T[NaN, NaN])
  @test same(prox(ElasticNet(T(0), T(0)), T[-0.0], Inf_), T[0])             # as at every finite step
  @test same(prox(ElasticNet(T(0), T(0)), T[-0.0], one(T)), T[0])
  for γ in (zero(T), one(T))
    @test same(prox(ElasticNet(T(0), T(0)), T[2, -1, NaN], γ), T[2, -1, NaN])
  end
  y = fill(T(7), 2)
  @test same(value_and_prox!(y, ElasticNet(T(0), T(0)), T[2, NaN], Inf_), ElasticNet(T(0), T(0))(T[2, NaN]))
  @test same(y, T[2, NaN])
end

@testset "slopes and envelopes, $T" for T in (Float32, Float64)
  for (λ1, λ2) in LEVELS, w in (nothing, T[0], T[0.5], T[Inf]), v in T[2, -1, 0, -0.0]
    f = enet(T, λ1, λ2, w)
    wi = w === nothing ? 1 : w[1]
    for γ in T[0.5, 1, Inf]
      @test PB._prox_slope(f, v, γ) === ref_slope(v, λ1, λ2, wi, γ)
    end
    @test PB.deriv2(PB.Smoothed(f, T(Inf)), v, zero(T)) === zero(T)
    @test same(PB.Smoothed(f, T(Inf))(v, zero(T)), zero(T))
  end
  for (λ1, λ2) in LEVELS, w in (nothing, T[0, 1], T[Inf, 0.5])
    f = enet(T, λ1, λ2, w)
    e = MoreauEnvelope(f, T(Inf))
    x = T[2, -1]
    @test same(e(x), zero(T))
    g = gradient(e, x)
    @test all(iszero, g) && !any(isnan, g)
  end
end

@testset "box constraints, $T" for T in (Float32, Float64)
  e = entries(T)
  n = length(e)
  for (λ1, λ2) in LEVELS, w in (nothing, fill(T(0), n), fill(T(Inf), n)), γ in (0.5, Inf)
    f = enet(T, λ1, λ2, w)
    want = [ref_prox(v, λ1, λ2, weightof(w, i), γ) for (i, v) in enumerate(e)]
    @test same(prox(Constrained(f, IndBox(T(-1), T(1))), e, T(γ)), clamp.(want, T(-1), T(1)))
    @test same(prox(Constrained(f, IndNonnegative()), e, T(γ)), clamp.(want, T(0), T(Inf)))
  end
end

@testset "wrappers and iterates, $T" for T in (Float32, Float64)
  Inf_ = T(Inf)
  x = T[2, -1, NaN]
  @test same(prox(Scaled(ElasticNet(T(1), T(1), T[Inf, 1, 1]), zero(T)), x, Inf_), T[0, -1, NaN])
  @test same(prox(T(0.5) * ElasticNet(T(0), T(0)), x, Inf_), x)
  @test same(prox(T(0.5) * ElasticNet(T(1), T(0)), x, Inf_), T[0, 0, NaN])
  b = T[1, 1, 1]
  @test same(prox(Translate(ElasticNet(T(0), T(0)), b), x, Inf_), x)
  @test same(prox(Translate(ElasticNet(T(1), T(0)), b), x, Inf_), T[1, 1, NaN])

  xi = SparseIterate(sparsevec(T[2, 0, -1]))
  @test same(Vector(prox(ElasticNet(T(0), T(0)), xi, Inf_)), T[2, 0, -1])
  @test same(Vector(prox(ElasticNet(T(1), T(0), T[0, 1, 1]), xi, Inf_)), T[2, 0, 0])
  @test Vector(prox(ElasticNet(T(0), T(1)), xi, Inf_)) == zeros(T, 3)
  S = SymmetricSparseIterate(T[2 1; 1 -3])
  @test Matrix(prox(ElasticNet(T(0), T(0)), S, Inf_)) == T[2 1; 1 -3]
  @test Matrix(prox(ElasticNet(T(0), T(1)), S, Inf_)) == zeros(T, 2, 2)

  X = T[3 1; 1 2]
  @test prox(SingularValue(ElasticNet(T(0), T(0))), X, Inf_) ≈ X
  @test prox(SingularValue(ElasticNet(T(1), T(0))), X, Inf_) == zeros(T, 2, 2)
  @test prox(Spectral(ElasticNet(T(0), T(0))), X, Inf_) ≈ X
  @test prox(Spectral(ElasticNet(T(0), T(1))), X, Inf_) == zeros(T, 2, 2)
end

@testset "Tilt at an infinite step, $T" for T in (Float32, Float64)
  # Tilt passes x - γc, which is NaN at γ = Inf for a NaN or infinite entry (and for every entry when
  # c = 0); an active part keeps that NaN, as at the base, and gives no finite point (E2, O2)
  Inf_ = T(Inf)
  for c in (2, 1, 0), x in (T[NaN], T[Inf])
    y = fill(T(7), 1)
    v = value_and_prox!(y, Tilt(ElasticNet(T(1), T(0)), T[c]), x, Inf_)
    @test isnan(v) && same(y, T[NaN])
  end
  @test same(prox(Tilt(ElasticNet(T(1), T(0)), fill(T(1), 3)), T[NaN, -1, 2], Inf_), fill(T(NaN), 3))
  @test same(prox(Tilt(ElasticNet(T(1), T(1)), fill(T(0), 3)), T[NaN, -1, 2], Inf_), fill(T(NaN), 3))
  x = T[NaN, -1, 2, Inf, -Inf]
  @test same(prox(Constrained(Tilt(ElasticNet(T(1), T(0)), fill(T(1), 5)), IndBox(T(-1), T(1))), x, Inf_), fill(T(NaN), 5))
end

# A negative step is not admitted (P1); until it is rejected (#311) a −Inf step, scalar or as an
# array entry, keeps the base's arithmetic and never takes the γ = +Inf rule: the coefficient γλ1w
# as an IEEE product (0 for a zero weight under λ1 > 0), shrink(v, c)/(1 + γλ2), and the slope
# abs(v) > c ? 1/(1 + γλ2) : 0. These pin the base's results.
function ref_negative(v::T, λ1, λ2, w, γ) where {T<:AbstractFloat}
  c = w == 0 && λ1 > 0 ? zero(T) : T(γ) * T(λ1) * T(w)
  d = one(T) + T(γ) * T(λ2)
  z = (max(v - c, zero(T)) + min(v + c, zero(T))) / d
  z, abs(v) > c ? inv(d) : zero(T)
end

@testset "negative infinite steps, $T" for T in (Float32, Float64)
  allowscalar(false)
  x = T[2, -3, 0.25, 0, NaN, Inf]
  ref(v, λ1, λ2, w, γ) = γ < 0 ? ref_negative(v, λ1, λ2, w, γ) : (ref_prox(v, λ1, λ2, w, γ), ref_slope(v, λ1, λ2, w, γ))
  for (λ1, λ2) in LEVELS, w in (nothing, T[0, 1, Inf, 1, 0.5, 1])
    f = enet(T, λ1, λ2, w)
    fd = adapt(JLArray, f)
    for γ in (T(-Inf), T[0.5, -Inf, -Inf, 1, -Inf, 0.5])
      γi(i) = γ isa AbstractArray ? γ[i] : γ
      r = [ref(x[i], λ1, λ2, weightof(w, i), γi(i)) for i in eachindex(x)]
      want = first.(r)
      check_routes(f, x, γ, want)
      for i in eachindex(x)
        @test same(PB._prox_entry_slope(f, x[i], i, γi(i)), r[i])
      end
      γd = γ isa AbstractArray ? JLArray(γ) : γ
      @test same(Array(prox(fd, JLArray(x), γd)), want)
      z = JLArray(copy(x))
      @test prox!(z, fd, z, γd) === z && same(Array(z), want)
      y = JLArray(fill(T(7), length(x)))
      v = value_and_prox!(y, fd, JLArray(x), γd)
      @test same(Array(y), want) && isnan(v)
    end
    if w === nothing
      for v in T[2, -3, 0, NaN]
        @test same(PB._prox_slope(f, v, T(-Inf)), last(ref_negative(v, λ1, λ2, 1, -Inf)))
      end
    end
  end
  y = fill(T(7), 2)
  v = value_and_prox!(y, ElasticNet(T(1), T(0)), T[2, -3], T(-Inf))
  @test isnan(v) && same(y, T[NaN, NaN])
  y = fill(T(7), 2)
  v = value_and_prox!(y, ElasticNet(T(1), T(0)), T[2, -3], T[0.5, -Inf])
  @test isnan(v) && same(y, T[1.5, NaN])
end

@testset "argument and step types" begin
  @test same(prox(ElasticNet(0, 0), [2, -1], Inf), [2.0, -1.0])
  @test same(prox(ElasticNet(1, 0), [2, -1], Inf), [0.0, 0.0])
  @test same(prox_entry(ElasticNet(0, 0), 2, 1, Inf), 2.0)
  @test same(prox(ElasticNet(0.0, 0.0), big.([2.0, -1.0]), Inf), big.([2.0, -1.0]))
  @test same(prox(ElasticNet(big(0.0), big(1.0)), [2.0, NaN], Inf), [0.0, NaN])
  @test same(prox(ElasticNet(0.0, 1.0), [2.0, NaN], Float16(Inf)), [0.0, NaN])
  @test same(prox(ElasticNet(0.0, 0.0), [2.0, NaN], Float16(Inf)), [2.0, NaN])
  @test same(prox(ElasticNet(1f0, 0f0), Float32[2, NaN], Inf), Float32[0, NaN])
  @test same(prox(ElasticNet(0f0, 0f0), Float16[2, NaN], Inf), Float16[2, NaN])
  @test same(prox!(zeros(Float32, 2), ElasticNet(0.0, 0.0), [2.0, -1.0], Inf), Float32[2, -1])
end

@testset "finite steps keep the ridge recovery" begin
  # γλ2 overflows Float32 although both factors are finite; the scaled ridge still applies it
  y = prox(ElasticNet(0f0, 1f20), Float32[1f30, -1f30], 1f20)
  want = Float32(big(1f30) / (1 + big(1f20) * big(1f20)))
  @test y ≈ [want, -want] rtol = 2eps(Float32)
  @test same(prox(ElasticNet(0.0, 1e300), [2.0, -1.0], 1e300), [0.0, -0.0])
end

@testset "device arrays, $T" for T in (Float32, Float64)
  allowscalar(false)
  x = T[2, -3, NaN, Inf, -0.0, 0.25]
  for (λ1, λ2) in ((0, 0), (1, 0), (0, 1), (1, 1), (Inf, 0)), w in (nothing, T[0, 1, Inf, 0.5, 1, 0], [0, 1, Inf, 0.5, 1, 0])
    f = enet(T, λ1, λ2, w)
    fd = adapt(JLArray, f)
    for γ in (T(Inf), T[Inf, 0, 1, Inf, Inf, 0.5], T[Inf, Inf, Inf, Inf, Inf, Inf])
      want = [ref_prox(x[i], λ1, λ2, weightof(w, i), γ isa AbstractArray ? γ[i] : γ) for i in eachindex(x)]
      γd = γ isa AbstractArray ? JLArray(γ) : γ
      @test same(prox(f, x, γ), want)
      @test same(Array(prox(fd, JLArray(x), γd)), want)
      z = JLArray(copy(x))
      @test prox!(z, fd, z, γd) === z && same(Array(z), want)
      y = JLArray(fill(T(7), length(x)))
      v = value_and_prox!(y, fd, JLArray(x), γd)
      vh = f(Array(y))                              # the host value; a device sum may round otherwise
      @test same(Array(y), want) && (same(v, vh) || (isfinite(vh) && v ≈ vh))
    end
  end
end

# The fused value of the conjugate of ElasticNet on a box is taken for host arrays only; a device array keeps the
# refusal of a value that is not available, before anything is written.
@testset "the conjugate of an elastic net on a box refuses device arrays before writing" begin
  allowscalar(false)
  for T in (Float32, Float64), (λ1, λ2) in ((T(1), T(0.5)), (T(1), T(0)))
    c = Constrained(Conjugate(ElasticNet(λ1, λ2)), IndBox(T(-2), T(2)))
    x = JLArray(T[5, -5, 0.5])
    y = JLArray(fill(T(-7), 3))
    @test_throws "is not implemented" value_and_prox!(y, c, x, T(0.5))
    @test Array(y) == fill(T(-7), 3) && Array(x) == T[5, -5, 0.5]
    z = JLArray(T[5, -5, 0.5])
    @test_throws "is not implemented" value_and_prox!(z, c, z, T(0.5))
    @test Array(z) == T[5, -5, 0.5]
  end
end

# A view whose index vector is a device array has host storage for its data and device storage for its selector:
# in any role, the fused value refuses it before anything is indexed or written, with scalar indexing disabled.
@testset "views with device selectors are refused before any index is read" begin
  for T in (Float32, Float64)
    sel(v) = @allowscalar view(v, JLArray([1, 2, 3]))
    plain = (y = fill(T(-7), 3), x = T[5, -5, 0.5], γ = T(0.5))
    function attempt(role, c, y, x, γ)
      allowscalar(false)
      before = @allowscalar collect(y)
      @test_throws "is not implemented" value_and_prox!(y, c, x, γ)
      @test (@allowscalar collect(y)) == before
    end
    ones6 = ones(T, 4)
    box = IndBox(T(-2), T(2))
    f = ElasticNet(T(1), T(0.5))
    c0 = Constrained(Conjugate(f), box)
    attempt(:output, c0, sel(fill(T(-7), 4)), plain.x, plain.γ)
    attempt(:input, c0, copy(plain.y), sel(T[5, -5, 0.5, 1]), plain.γ)
    attempt(:step, c0, copy(plain.y), plain.x, sel(fill(T(0.5), 4)))
    attempt(:lower, Constrained(Conjugate(f), @allowscalar IndBox(sel(fill(T(-2), 4)), T(2))), copy(plain.y), plain.x, plain.γ)
    attempt(:upper, Constrained(Conjugate(f), @allowscalar IndBox(T(-2), sel(fill(T(2), 4)))), copy(plain.y), plain.x, plain.γ)
    for λ2 in (T(0.5), T(0))
      attempt(:weights, Constrained(Conjugate(@allowscalar ElasticNet(T(1), λ2, sel(copy(ones6)))), box), copy(plain.y), plain.x, plain.γ)
    end
  end
end

end
