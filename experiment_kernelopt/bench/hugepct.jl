# The percentage of the 2 MiB pages that hold [a, a + len) that is mapped as huge pages, or -1 when it cannot be
# measured. /proc/self/smaps reports AnonHugePages per mapping, and an array is usually a small part of one, so the
# mappings are first split at lo and hi, the multiples of 2 MiB around the array, by an advice that changes nothing
# else (MADV_DONTDUMP, undone by MADV_DODUMP); no huge page is split at a multiple of 2 MiB. The denominator is the
# whole of those 2 MiB pages, not the bytes of the array: 100 and 0 are exact statements about the array, a value
# between them says that some of its 2 MiB pages are huge and some are not. -1: a mapping there already has the
# DONTDUMP flag (the split and the restoration would be wrong), or the mappings found after the split do not cover
# the array.
const HP = UInt(2 << 20)
advise(lo::UInt, len, advice) = ccall(:madvise, Cint, (Ptr{Cvoid}, Csize_t, Cint), Ptr{Cvoid}(lo), len, advice)
# (start, end, AnonHugePages kB, DONTDUMP flag) of every mapping of the process
function mappings()
  out = Tuple{UInt, UInt, Int, Bool}[]
  for l in eachline("/proc/self/smaps")
    m = match(r"^([0-9a-f]+)-([0-9a-f]+) ", l)
    if m !== nothing
      push!(out, (parse(UInt, m.captures[1]; base = 16), parse(UInt, m.captures[2]; base = 16), 0, false))
    elseif startswith(l, "AnonHugePages:")
      t = out[end]; out[end] = (t[1], t[2], parse(Int, split(l)[2]), t[4])
    elseif startswith(l, "VmFlags:")
      t = out[end]; out[end] = (t[1], t[2], t[3], "dd" in split(l)[2:end])
    end
  end
  out
end
function hugepct(a::UInt, len)
  lo = a & ~(HP - 1); hi = (a + len + HP - 1) & ~(HP - 1)
  any(m -> m[1] < hi && lo < m[2] && m[4], mappings()) && return -1
  advise(lo, hi - lo, 16)
  inside = filter(m -> lo <= m[1] && m[2] <= hi, mappings())
  advise(lo, hi - lo, 17)
  covered = 0
  for m in inside; covered += max(0, Int(min(m[2], a + len)) - Int(max(m[1], a))); end
  covered == len || return -1
  kb = 0; for m in inside; kb += m[3]; end
  round(Int, 100 * kb * 1024 / (hi - lo))
end
hugepct(x::Array) = hugepct(UInt(pointer(x)), sizeof(x))
# MADV_HUGEPAGE on the 4 KiB pages that hold [a, a + len)
hugeadvise(a::UInt, len) = (lo = a & ~UInt(4095); advise(lo, ((a + len + 4095) & ~UInt(4095)) - lo, 14))
