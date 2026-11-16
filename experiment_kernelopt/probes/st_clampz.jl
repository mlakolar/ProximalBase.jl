# v minus its projection onto [-c, c], plus +0: the bits of max(v - c, 0) + min(v + c, 0) for every v and every
# threshold c that is not negative (+0, -0, positive, Inf, NaN)?
using InteractiveUtils, Random
st_maxmin(v::T, c::T) where {T} = max(v - c, zero(T)) + min(v + c, zero(T))
@inline function st_clampz(v::T, c::T) where {T}
  lo = -c
  m = ifelse(v > lo, v, lo)          # the hardware maximum of (v, -c): -c when v is NaN or c is NaN
  p = ifelse(m < c, m, c)            # the hardware minimum of (m, c): c when either is NaN
  (v - p) + zero(T)                  # -0 (from v = -0 at a zero threshold) becomes +0; nothing else changes
end
function exhaustive16()
  bad = 0; nanbits = 0; first = nothing; nc = 0
  for cu in 0x0000:0xffff
    c = reinterpret(Float16, cu)
    c < 0 && continue
    nc += 1
    for vu in 0x0000:0xffff
      v = reinterpret(Float16, vu)
      a, b = st_maxmin(v, c), st_clampz(v, c)
      a === b && continue
      if isnan(a) && isnan(b); nanbits += 1 else bad += 1; first === nothing && (first = (v, c, a, b)); end
    end
  end
  println(VERSION, " Float16: every v × ", nc, " thresholds that are not negative: value differences ", bad, ", NaN results with other bits ", nanbits, ", first ", first)
end
exhaustive16()
function check(::Type{T}, nrand) where {T}
  U = T === Float64 ? UInt64 : UInt32
  e = T[0, -0.0, floatmin(T), nextfloat(zero(T)), prevfloat(floatmin(T)), 1, prevfloat(one(T)), nextfloat(one(T)), eps(T), floatmax(T), prevfloat(floatmax(T)),
        floatmax(T) / 2, sqrt(floatmax(T)), Inf, NaN, -T(NaN), reinterpret(T, reinterpret(U, T(NaN)) | U(1)), 3, T(0.1)]
  vs = vcat(e, -e); cs = [c for c in vs if !(c < 0)]
  bad = 0; first = nothing
  cmp(v, c) = (a = st_maxmin(v, c); b = st_clampz(v, c); (a === b || (isnan(a) && isnan(b))) || (bad += 1; first === nothing && (first = (v, c, a, b))))
  for v in vs, c in cs; cmp(v, c); end
  rng = Xoshiro(11)
  for _ in 1:nrand
    v = reinterpret(T, rand(rng, U)); c = abs(reinterpret(T, rand(rng, U)))
    cmp(v, c)
    c2 = abs(v); for _ in 1:rand(rng, 0:3); c2 = rand(rng, Bool) ? nextfloat(c2) : prevfloat(c2); end
    c2 < 0 || cmp(v, c2)
  end
  println(VERSION, " ", T, ": grid and ", 2nrand, " random pairs, c not negative: differences ", bad, ", first ", first)
end
check(Float32, 100_000_000); check(Float64, 100_000_000)
function loop!(st::F, y, x, c) where {F}
  @inbounds @simd for i in eachindex(y, x); y[i] = st(x[i], c); end
  y
end
for st in (st_maxmin, st_clampz)
  io = IOBuffer(); code_native(io, loop!, (typeof(st), Vector{Float64}, Vector{Float64}, Float64); syntax = :intel, debuginfo = :none)
  ops = [first(split(strip(l))) for l in split(String(take!(io)), '\n') if occursin("ymm", l)]
  println(VERSION, " ", nameof(st), " ymm instructions: ", join(["$k×$(count(==(k), ops))" for k in unique(ops)], " "))
end
x = randn(1024); y = similar(x)
for st in (st_maxmin, st_clampz)
  loop!(st, y, x, 0.5); best = Inf
  for _ in 1:200; t = time_ns(); for _ in 1:2000; loop!(st, y, x, 0.5); end; best = min(best, (time_ns() - t) / 2000); end
  println(VERSION, " ", nameof(st), " n=1024: ", round(best, digits = 1), " ns")
end
