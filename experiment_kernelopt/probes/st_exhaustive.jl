# Bit-for-bit comparison of soft-threshold forms with ProximalBase's shrink.
st_maxmin(v::T, c::T) where {T} = max(v - c, zero(T)) + min(v + c, zero(T))
_selmax(a, b) = ifelse(b > a, b, a)
_selmin(a, b) = ifelse(b < a, b, a)
st_selmaxmin(v::T, c::T) where {T} = _selmax(v - c, zero(T)) + _selmin(v + c, zero(T))
st_clamp(v::T, c::T) where {T} = v - clamp(v, -c, c)
st_clampsel(v::T, c::T) where {T} = v - ifelse(v > c, c, ifelse(v < -c, -c, v))
st_copysignsel(v::T, c::T) where {T} = copysign(_selmax(abs(v) - c, zero(T)), v)
same(a, b) = (a === b) || (isnan(a) && isnan(b))
function exhaustive16(f::F, cs) where {F}
  nbits = 0; nnan = 0; first = nothing
  for cu in cs
    c = reinterpret(Float16, cu)
    for vu in 0x0000:0xffff
      v = reinterpret(Float16, vu)
      a, b = st_maxmin(v, c), f(v, c)
      if !(a === b)
        if isnan(a) && isnan(b); nnan += 1
        else nbits += 1; first === nothing && (first = (v, c, a, b)); end
      end
    end
  end
  (nbits, nnan, first)
end
allc = 0x0000:0xffff
nonneg = [u for u in allc if !signbit(reinterpret(Float16, u)) && !isnan(reinterpret(Float16, u))]   # c ≥ 0, +0 included
for (name, f) in (("selmaxmin", st_selmaxmin), ("clamp", st_clamp), ("clampsel", st_clampsel), ("copysignsel", st_copysignsel))
  r_all = exhaustive16(f, allc)
  r_nn = exhaustive16(f, nonneg)
  println(rpad(name, 12), " all c (2^32 pairs): value diffs ", r_all[1], ", NaN-bit diffs ", r_all[2], ", first ", r_all[3])
  println(rpad("", 12), " c >= 0 (non-NaN):   value diffs ", r_nn[1], ", NaN-bit diffs ", r_nn[2], ", first ", r_nn[3])
end
# Float32 / Float64: edge grid crossed with itself, and random bit patterns
function grid(::Type{T}) where {T}
  U = T === Float64 ? UInt64 : UInt32
  e = T[0, floatmin(T), nextfloat(zero(T)), prevfloat(floatmin(T)), 1, prevfloat(one(T)), nextfloat(one(T)), eps(T), floatmax(T),
        prevfloat(floatmax(T)), floatmax(T) / 2, sqrt(floatmax(T)), sqrt(floatmin(T)), T(Inf), T(NaN), reinterpret(T, reinterpret(U, T(NaN)) | U(1)), 3, T(0.1)]
  vcat(e, -e)
end
function check(f::F, ::Type{T}, nrand) where {F, T}
  U = T === Float64 ? UInt64 : UInt32
  g = grid(T); bad = 0; nan = 0; first = nothing
  cmp(v, c) = (a = st_maxmin(v, c); b = f(v, c); a === b ? 0 : (isnan(a) && isnan(b)) ? 1 : (first === nothing && (first = (v, c, a, b)); 2))
  for v in g, c in g
    r = cmp(v, c); bad += r == 2; nan += r == 1
  end
  rng = Random.Xoshiro(7)
  for _ in 1:nrand
    v = reinterpret(T, rand(rng, U)); c = reinterpret(T, rand(rng, U))
    r = cmp(v, c); bad += r == 2; nan += r == 1
    # near the threshold: c a few ulps from |v|
    c2 = abs(v); for _ in 1:rand(rng, 0:3); c2 = rand(rng, Bool) ? nextfloat(c2) : prevfloat(c2); end
    r = cmp(v, c2); bad += r == 2; nan += r == 1
  end
  (bad, nan, first)
end
using Random
for T in (Float32, Float64), (name, f) in (("selmaxmin", st_selmaxmin), ("clamp", st_clamp), ("clampsel", st_clampsel))
  println(T, " ", rpad(name, 10), " grid + 2e8 random pairs: (value diffs, NaN-bit diffs, first) = ", check(f, T, 100_000_000))
end
