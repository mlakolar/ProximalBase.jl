# Cost of an OR of the bit patterns fused into the sum of squares of a group (to tell a zero group in one pass)
using Random
@noinline call(op::F) where {F} = (op(); nothing)
function bench(op::F; reps = 41) where {F}
  call(op); t0 = time_ns(); call(op); dt = max(time_ns() - t0, 1); evals = max(1, round(Int, 3e5 / dt)); best = Inf
  for _ in 1:reps
    t = time_ns(); for _ in 1:evals; call(op); end; best = min(best, (time_ns() - t) / evals)
  end
  best
end
@inline function ss(x::Vector{T}, G) where {T}
  s = zero(T)
  @inbounds @simd for i in G; s += abs2(x[i]); end
  s
end
@inline function ssor(x::Vector{Float64}, G)
  s = 0.0; m = zero(UInt64)
  @inbounds @simd for i in G; v = x[i]; s += abs2(v); m |= reinterpret(UInt64, v); end
  s, m
end
@inline function ssor(x::Vector{Float32}, G)
  s = 0f0; m = zero(UInt32)
  @inbounds @simd for i in G; v = x[i]; s += abs2(v); m |= reinterpret(UInt32, v); end
  s, m
end
function value_plain(x::Vector{T}, groups, w, λ) where {T}
  v = zero(T)
  @inbounds for k in eachindex(groups); v += (λ * w[k]) * sqrt(ss(x, groups[k])); end
  v
end
function value_or(x::Vector{T}, groups, w, λ) where {T}
  v = zero(T); any = zero(T === Float64 ? UInt64 : UInt32)
  @inbounds for k in eachindex(groups)
    s, m = ssor(x, groups[k]); any |= ifelse(iszero(s), m, zero(m)); v += (λ * w[k]) * sqrt(s)
  end
  v + T(any & 0)
end
for T in (Float64, Float32), m in (2, 5, 10, 32, 128, 1024), layout in (:range, :perm)
  Kg = 4096 ÷ m; rng = MersenneTwister(m); x = randn(rng, T, Kg * m)
  order = layout === :perm ? randperm(rng, Kg * m) : collect(1:Kg*m)
  groups = layout === :perm ? [order[(k-1)*m+1:k*m] for k in 1:Kg] : [(k-1)*m+1:k*m for k in 1:Kg]
  w = fill(T(1), Kg); r = Ref(zero(T)); λ = T(0.6)
  a = bench(() -> (r[] = value_plain(x, groups, w, λ))); b = bench(() -> (r[] = value_or(x, groups, w, λ)))
  println(VERSION, " ", T, " m=", m, " ", layout, ": plain ", round(a / Kg, digits = 2), " ns/group, with OR ", round(b / Kg, digits = 2), " ns/group, ratio ", round(b / a, digits = 3))
end
