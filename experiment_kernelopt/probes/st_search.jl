# Which arrangement of the two selects of "v minus its projection onto [-c, c]" has the bits of
# max(v - c, 0) + min(v + c, 0) for every v and every threshold that is not negative (±0, positive, Inf, NaN)?
st_maxmin(v::T, c::T) where {T} = max(v - c, zero(T)) + min(v + c, zero(T))
const LOS = (("-c", c -> -c), ("0-c", c -> zero(c) - c))
const MAXS = (("v>lo?v:lo", (v, lo) -> ifelse(v > lo, v, lo)), ("lo>v?lo:v", (v, lo) -> ifelse(lo > v, lo, v)),
              ("v<lo?lo:v", (v, lo) -> ifelse(v < lo, lo, v)), ("lo<v?v:lo", (v, lo) -> ifelse(lo < v, v, lo)))
const MINS = (("m<c?m:c", (m, c) -> ifelse(m < c, m, c)), ("c<m?c:m", (m, c) -> ifelse(c < m, c, m)),
              ("m>c?c:m", (m, c) -> ifelse(m > c, c, m)), ("c>m?m:c", (m, c) -> ifelse(c > m, m, c)))
function count_diffs(lo::L, mx::A, mn::B) where {L, A, B}
  bad = 0; first = nothing
  for cu in 0x0000:0xffff
    c = reinterpret(Float16, cu)
    c < 0 && continue
    l = lo(c)
    for vu in 0x0000:0xffff
      v = reinterpret(Float16, vu)
      a = st_maxmin(v, c); b = v - mn(mx(v, l), c)
      (a === b || (isnan(a) && isnan(b))) && continue
      bad += 1; first === nothing && (first = (v, c, a, b))
    end
  end
  bad, first
end
for (ln, lo) in LOS, (an, mx) in MAXS, (bn, mn) in MINS
  bad, first = count_diffs(lo, mx, mn)
  println(rpad("lo = $ln; m = $an; p = $bn", 48), " differences: ", bad, bad == 0 ? "   <== identical" : "   first $first")
end
