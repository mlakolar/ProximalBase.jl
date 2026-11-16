using Random
@noinline call(op::F) where {F} = (op(); nothing)
function bench(op::F; reps = 41) where {F}
  call(op); t0 = time_ns(); call(op); dt = max(time_ns() - t0, 1); evals = max(1, round(Int, 2e5 / dt)); best = Inf
  for _ in 1:reps
    t = time_ns(); for _ in 1:evals; call(op); end; best = min(best, (time_ns() - t) / evals)
  end
  best
end
_selmax(a, b) = ifelse(b > a, b, a); _selmin(a, b) = ifelse(b < a, b, a)
utype(::Type{Float64}) = UInt64; utype(::Type{Float32}) = UInt32
stype(::Type{Float64}) = Int64; stype(::Type{Float32}) = Int32
function w_floor(x::Vector{T}, λ::T, w::Vector{T}) where {T}
  s = zero(T)
  @inbounds @simd for i in eachindex(x, w); s += (λ * w[i]) * abs(x[i]); end
  s
end
function w_chain(x::Vector{T}, λ::T, w::Vector{T}) where {T}         # ProximalBase's loop: Bool flag, chained comparison
  s = zero(T); ok = true
  @inbounds @simd for i in eachindex(x, w)
    a = λ * w[i]; s += a * abs(x[i]); ok &= (floatmin(T) <= a <= floatmax(T))
  end
  s, ok
end
function w_and(x::Vector{T}, λ::T, w::Vector{T}) where {T}           # Bool flag, both comparisons evaluated
  s = zero(T); ok = true
  @inbounds @simd for i in eachindex(x, w)
    a = λ * w[i]; s += a * abs(x[i]); ok &= (floatmin(T) <= a) & (a <= floatmax(T))
  end
  s, ok
end
function w_mask(x::Vector{T}, λ::T, w::Vector{T}) where {T}          # a mask of the lane width, OR-accumulated
  U = utype(T); s = zero(T); bad = zero(U)
  @inbounds @simd for i in eachindex(x, w)
    a = λ * w[i]; s += a * abs(x[i]); bad |= ifelse((floatmin(T) <= a) & (a <= floatmax(T)), zero(U), ~zero(U))
  end
  s, iszero(bad)
end
function w_intmin(x::Vector{T}, λ::T, w::Vector{T}) where {T}        # lower end by a signed integer minimum of the bits; upper end by isfinite(s)
  I = stype(T); s = zero(T); lo = typemax(I)
  @inbounds @simd for i in eachindex(x, w)
    a = λ * w[i]; s += a * abs(x[i]); lo = min(lo, reinterpret(I, a))
  end
  s, (lo >= reinterpret(I, floatmin(T))) & isfinite(s)
end
function w_exp(x::Vector{T}, λ::T, w::Vector{T}) where {T}           # exponent field + 1, masked: 0 for Inf/NaN, 1 unit for zero/subnormal
  I = stype(T); s = zero(T); lo = typemax(I)
  one_e = reinterpret(I, floatmin(T)); mask = reinterpret(I, T(Inf))
  @inbounds @simd for i in eachindex(x, w)
    a = λ * w[i]; s += a * abs(x[i]); lo = min(lo, (reinterpret(I, a) + one_e) & mask)
  end
  s, lo >= 2 * one_e
end
function w_wpass(x::Vector{T}, λ::T, w::Vector{T}) where {T}         # the weights' range in a pass of their own (integer min and max), then the bare sum
  I = stype(T); lo = typemax(I); hi = typemin(I)
  @inbounds @simd for i in eachindex(w); b = reinterpret(I, λ * w[i]); lo = min(lo, b); hi = max(hi, b); end
  w_floor(x, λ, w), (lo >= reinterpret(I, floatmin(T))) & (hi <= reinterpret(I, floatmax(T)))
end
function w_int2(x::Vector{T}, λ::T, w::Vector{T}) where {T}          # both ends by integer min and max of the bits, in the loop
  I = stype(T); s = zero(T); lo = typemax(I); hi = typemin(I)
  @inbounds @simd for i in eachindex(x, w)
    a = λ * w[i]; s += a * abs(x[i]); b = reinterpret(I, a); lo = min(lo, b); hi = max(hi, b)
  end
  s, (lo >= reinterpret(I, floatmin(T))) & (hi <= reinterpret(I, floatmax(T)))
end
function w_umax(x::Vector{T}, λ::T, w::Vector{T}) where {T}          # one unsigned maximum of the bits shifted by those of floatmin
  U = utype(T); s = zero(T); m = zero(U); base = reinterpret(U, floatmin(T))
  @inbounds @simd for i in eachindex(x, w)
    a = λ * w[i]; s += a * abs(x[i]); m = max(m, reinterpret(U, a) - base)
  end
  s, m <= reinterpret(U, floatmax(T)) - base
end
st(v::T, c::T) where {T} = _selmax(v - c, zero(T)) + _selmin(v + c, zero(T))
@noinline cold(v, g, λ, w) = st(v, (g * λ) * w)
function g_branchy!(y::Vector{T}, x, γ, λ::T, w) where {T}
  @inbounds for i in eachindex(y, x)
    ab = γ[i] * λ
    y[i] = (floatmin(T) <= ab <= floatmax(T)) ? st(x[i], ab * w[i]) : cold(x[i], γ[i], λ, w[i])
  end
  y
end
function steps_and(γ::Vector{T}, λ::T) where {T}
  ok = true
  @inbounds @simd for i in eachindex(γ); ab = γ[i] * λ; ok &= (floatmin(T) <= ab) & (ab <= floatmax(T)); end
  ok
end
function steps_mask(γ::Vector{T}, λ::T) where {T}
  U = utype(T); bad = zero(U)
  @inbounds @simd for i in eachindex(γ); ab = γ[i] * λ; bad |= ifelse((floatmin(T) <= ab) & (ab <= floatmax(T)), zero(U), ~zero(U)); end
  iszero(bad)
end
function steps_int(γ::Vector{T}, λ::T) where {T}
  I = stype(T); lo = typemax(I); hi = typemin(I)
  @inbounds @simd for i in eachindex(γ); b = reinterpret(I, γ[i] * λ); lo = min(lo, b); hi = max(hi, b); end
  (lo >= reinterpret(I, floatmin(T))) & (hi <= reinterpret(I, floatmax(T)))
end
function g_prescan!(scan::S, y::Vector{T}, x, γ, λ::T, w) where {S, T}
  scan(γ, λ) || return g_branchy!(y, x, γ, λ, w)
  @inbounds @simd for i in eachindex(y, x); y[i] = st(x[i], (γ[i] * λ) * w[i]); end
  y
end
function g_floor!(y::Vector{T}, x, γ, λ::T, w) where {T}
  @inbounds @simd for i in eachindex(y, x); y[i] = st(x[i], (γ[i] * λ) * w[i]); end
  y
end
for T in (Float64, Float32), n in (128, 1024, 16384, 1048576)
  rng = MersenneTwister(n); x = randn(rng, T, n); y = similar(x); w = T(0.5) .+ rand(rng, T, n); γ = T(0.5) .+ rand(rng, T, n); λ = T(0.6)
  r = Ref((zero(T), true)); r0 = Ref(zero(T))
  f0 = bench(() -> (r0[] = w_floor(x, λ, w)))
  p(g) = (g(x, λ, w)[2] || error("flag"); round(bench(() -> (r[] = g(x, λ, w))) / f0, digits = 2))
  g0 = bench(() -> g_floor!(y, x, γ, λ, w))
  q(g) = round(bench(() -> g_prescan!(g, y, x, γ, λ, w)) / g0, digits = 2)
  println(T, " n=", n, " value/bare (", round(f0, digits = 1), "ns): chain ", p(w_chain), " and ", p(w_and), " mask ", p(w_mask), " intmin ", p(w_intmin), " exp ", p(w_exp), " wpass ", p(w_wpass), " int2 ", p(w_int2), " umax ", p(w_umax),
          " || steps/bare (", round(g0, digits = 1), "ns): branchy ", round(bench(() -> g_branchy!(y, x, γ, λ, w)) / g0, digits = 2), " prescan: and ", q(steps_and), " mask ", q(steps_mask), " int ", q(steps_int))
end
