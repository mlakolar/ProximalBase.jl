using InteractiveUtils
st_maxmin(v::T, c::T) where {T} = max(v - c, zero(T)) + min(v + c, zero(T))
_selmax(a, b) = ifelse(b > a, b, a)
_selmin(a, b) = ifelse(b < a, b, a)
st_selmaxmin(v::T, c::T) where {T} = _selmax(v - c, zero(T)) + _selmin(v + c, zero(T))
function loop!(st::F, y, x, c) where {F}
  @inbounds @simd for i in eachindex(y, x)
    y[i] = st(x[i], c)
  end
  y
end
# the instructions of the widest vector loop body: the block that ends in the backward jump after the first ymm store
function body(st, T)
  io = IOBuffer(); code_native(io, loop!, (typeof(st), Vector{T}, Vector{T}, T); syntax = :intel, debuginfo = :none)
  lines = [strip(l) for l in split(String(take!(io)), '\n')]
  lines = filter(l -> !isempty(l) && !startswith(l, ";") && !startswith(l, "."), lines)
  # collect maximal runs of ymm/zmm instructions between labels/jumps
  best = String[]; cur = String[]
  for l in lines
    if occursin(r"^(j|L\d|\w+:)", l) || endswith(l, ":")
      n = count(x -> occursin(r"[yz]mm", x), cur)
      n > count(x -> occursin(r"[yz]mm", x), best) && (best = copy(cur))
      empty!(cur)
    else
      push!(cur, l)
    end
  end
  best
end
for T in (Float64,), st in (st_maxmin, st_selmaxmin)
  b = body(st, T)
  ops = [first(split(l)) for l in b if occursin(r"[yz]mm", l)]
  stores = count(l -> occursin(r"^vmovup[sd]\s+\w+ ptr", l), b)
  println(VERSION, " ", nameof(st), " ", T, ": ", length(ops), " vector instructions for ", stores, " stored vectors -> ",
          round(length(ops) / max(stores, 1), digits = 2), " per vector; mix: ", join(["$k×$(count(==(k), ops))" for k in unique(ops)], " "))
end
# scalar form, as a coordinate-descent step calls it
for st in (st_maxmin, st_selmaxmin)
  io = IOBuffer(); code_native(io, st, (Float64, Float64); syntax = :intel, debuginfo = :none)
  l = [strip(x) for x in split(String(take!(io)), '\n')]
  l = filter(x -> !isempty(x) && !startswith(x, ";") && !startswith(x, ".") && !endswith(x, ":") && !occursin(r"^(push|pop|mov\s+rbp|ret|nop)", x), l)
  println(VERSION, " scalar ", nameof(st), ": ", length(l), " instructions: ", join(first.(split.(l)), " "))
end
