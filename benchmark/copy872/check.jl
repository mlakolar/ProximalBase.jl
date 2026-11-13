# Independent reference check of whole-iterate BlockIterate copying for one source tree.
# usage: julia --check-bounds=yes check.jl SOURCE_ROOT NTRIALS SEED
# The reference model is written from the documented contract (dense values plus an
# ordered active list), not from any implementation under test.
using Random
const ROOT, NTRIALS, SEED = ARGS[1], parse(Int, ARGS[2]), parse(Int, ARGS[3])
Base.include(Main, joinpath(ROOT, "src", "ProximalBase.jl"))
const PB = Main.ProximalBase

failures = Ref(0); checks = Ref(0)
macro chk(ex, msg)
  quote
    checks[] += 1
    if !($(esc(ex)))
      failures[] += 1
      failures[] <= 20 && println("FAIL: ", $(esc(msg)))
    end
  end
end

function layouts(rng, n)
  b = rand(rng, [1, 1, 2, 3, 4, 8])
  n % b == 0 || (b = 1)
  K = n ÷ b
  kind = rand(rng, n == 1 ? (1:5) : (1:6))
  kind == 1 && return [(k-1)*b+1:k*b for k in 1:K], "contig b$b"
  kind == 2 && return [k:K:n for k in 1:K], "strided b$b"
  kind == 3 && (p = randperm(rng, n); return [p[(k-1)*b+1:k*b] for k in 1:K], "arbitrary b$b")
  kind == 4 && return [Int32((k-1)*b+1):Int32(k*b) for k in 1:K], "int32 contig b$b"
  kind == 5 && return [Int32(k):Int32(K):Int32(n) for k in 1:K], "int32 strided b$b"
  # uneven groups
  cuts = sort!(unique!(vcat(0, rand(rng, 1:n-1, rand(rng, 0:n-1)), n)))
  p = randperm(rng, n)
  return [p[cuts[j]+1:cuts[j+1]] for j in 1:length(cuts)-1], "uneven"
end

randval(rng, ::Type{T}) where {T} = rand(rng) < 0.1 ? rand(rng, T[-0.0, 0.0, Inf, -Inf, NaN]) : T(round(randn(rng) * 8) / 4)
randval(rng, ::Type{BigFloat}) = BigFloat(randval(rng, Float64))
randval(rng, ::Type{Int}) = rand(rng, -3:3)

# Write values through the public API in a random order of groups; return the model
# (values, active order). A zero write keeps a group inactive unless activated explicitly.
function populate!(rng, x, gs, frac; inactive_negzero = true)
  T = eltype(x); K = length(gs)
  ks = shuffle(rng, 1:K)[1:round(Int, frac * K)]
  for k in ks
    if rand(rng) < 0.15
      PB.activate!(x, k)                     # an active group, possibly all zeros
      for i in gs[k]; x[i] = rand(rng) < 0.5 ? -zero(T) : zero(T); end
    else
      for i in gs[k]; x[i] = randval(rng, T); end
      all(i -> iszero(x[i]), gs[k]) && PB.activate!(x, k)
    end
  end
  if inactive_negzero && T <: AbstractFloat
    for k in 1:K
      PB.isactive(x, k) && continue
      rand(rng) < 0.5 && (x[first(gs[k])] = -zero(T))   # stays inactive: a zero write
    end
  end
  x
end

snapshot(x) = (vals = copy(x.storage), ids = collect(PB.activecoordinates(x)), pos = copy(x.position),
               nact = PB.nactive(x), storage = x.storage, groups = x.groups, cache = x.layout_cache,
               active = x.active, position = x.position)

function check_state(x, vals, ids, tag)
  @chk isequal(x.storage, vals) "$tag values"
  @chk collect(PB.activecoordinates(x)) == ids "$tag ids"
  @chk PB.nactive(x) == length(ids) "$tag nactive"
  ok = true
  for k in eachindex(x.groups)
    p = findfirst(==(k), ids)
    ok &= x.position[k] == (p === nothing ? 0 : p)
    ok &= PB.isactive(x, k) == (p !== nothing)
  end
  @chk ok "$tag positions"
end

# expected result of copyto!(dest, src) with the same layout and a nonthrowing conversion
function expected_same(d0, s0, gs, ::Type{T}) where {T}
  vals = copy(d0.vals)
  ids = copy(s0.ids)
  for k in d0.ids
    k in ids && continue
    for i in gs[k]; vals[i] = zero(T); end
  end
  for k in ids, i in gs[k]
    vals[i] = convert(T, s0.vals[i])
  end
  vals, ids
end

# expected result of the generic fallback: entry writes in linear order
function expected_generic(d0, svals, gs_dest, ::Type{T}) where {T}
  vals = copy(d0.vals); ids = copy(d0.ids)
  gof = zeros(Int, length(vals)); for k in eachindex(gs_dest), i in gs_dest[k]; gof[i] = k; end
  for i in eachindex(svals)
    v = convert(T, svals[i]); vals[i] = v
    iszero(v) || (gof[i] in ids || push!(ids, gof[i]))
  end
  vals, ids
end

function trial(rng, t)
  n = rand(rng, [1, 2, 3, 4, 6, 8, 12, 16, 24, 48])
  gs, lname = layouts(rng, n)
  K = length(gs)
  pair = rand(rng, [(Float64, Float64), (Float32, Float32), (Float32, Float64), (Float64, Float32),
                    (BigFloat, BigFloat), (Float64, Int)])
  T, S = pair
  fs = rand(rng, [0.0, 0.0, 0.25, 0.5, 1.0, 1.0, 1.0, rand(rng)])
  fd = rand(rng, [0.0, 0.5, 1.0, rand(rng)])
  src = populate!(rng, PB.BlockIterate(S, n, gs), gs, fs)
  own = rand(rng, (:shared, :independent, :cached, :self, :different))
  tag = "trial $t $lname n$n $T<-$S $own fs=$fs fd=$fd"
  if own == :self
    s0 = snapshot(src)
    r = rand(rng, Bool) ? copyto!(src, src) : copy!(src, src)
    @chk r === src "$tag self return"
    check_state(src, s0.vals, s0.ids, tag * " self")
    return
  end
  if own == :different
    gd = K > 1 ? circshift(gs, 1) : [collect(gs[1])]
    gd == gs && (gd = [collect(g) for g in reverse(gs)])
    dest = populate!(rng, PB.BlockIterate(T, n, gd), gd, fd)
    d0 = snapshot(dest)
    gd == gs && return            # an equal layout (K == 1): not a different-layout case
    copyto!(dest, src)
    v, ids = expected_generic(d0, src.storage, gd, T)
    check_state(dest, v, ids, tag)
    return
  end
  dest = own == :shared ? similar(src, T) : PB.BlockIterate(T, n, gs)
  populate!(rng, dest, gs, fd)
  # distinct penalty caches must survive
  src.layout_cache = deepcopy(gs); dest.layout_cache = deepcopy(gs)
  own == :cached && PB.samegroups(dest, src)
  s0 = snapshot(src); d0 = snapshot(dest)
  sobj = copy(src.storage)               # element objects of src (identity for BigFloat)
  dobj = copy(dest.storage)
  r = rand(rng, Bool) ? copyto!(dest, src) : copy!(dest, src)
  @chk r === dest "$tag return"
  v, ids = expected_same(d0, s0, gs, T)
  check_state(dest, v, ids, tag)
  check_state(src, s0.vals, s0.ids, tag * " src unchanged")
  @chk dest.storage === d0.storage && dest.active === d0.active && dest.position === d0.position "$tag owned arrays"
  @chk dest.groups === d0.groups && dest.layout_cache === d0.cache && src.layout_cache === s0.cache "$tag layout/cache identity"
  @chk dest.storage !== src.storage && !Base.mightalias(dest, src) "$tag no aliasing"
  if T === BigFloat
    ok = true
    for k in 1:K
      if k in s0.ids
        ok &= all(i -> dest.storage[i] === sobj[i], gs[k])
      elseif k in d0.ids
        g = collect(gs[k]); ok &= all(i -> dest.storage[i] === dest.storage[g[1]], g)
        ok &= dest.storage[g[1]] !== dobj[g[1]] && isequal(dest.storage[g[1]], zero(T))   # one fresh +0 per group
      else
        ok &= all(i -> dest.storage[i] === dobj[i], gs[k])   # untouched objects
      end
    end
    @chk ok "$tag BigFloat identity"
  end
  # subsequent writes and pruning follow the model
  if K > 0
    k = rand(rng, 1:K); i = first(gs[k]); w = T(3)
    dest[i] = w; v[i] = w; k in ids || push!(ids, k)
    check_state(dest, v, ids, tag * " after write")
  end
end

# fallible conversion: preflight leaves the destination unchanged
function trial_throw(rng, t)
  gs = rand(rng, ([1:3, 4:6, 7:9], [1:3:9, 2:3:9, 3:3:9], [[7, 1, 4], [8, 2, 5], [9, 3, 6]], [k:k for k in 1:9]))
  src = PB.BlockIterate(Float64, 9, gs)
  for k in shuffle(rng, 1:length(gs)), i in gs[k]; src[i] = rand(rng, (1.0, 2.0, 0.0)); end
  bad = rand(rng, 1:9); src[bad] = 0.5
  dest = PB.BlockIterate(Int, 9, gs); populate!(rng, dest, gs, rand(rng))
  dest.layout_cache = deepcopy(gs)
  d0 = snapshot(dest)
  threw = try; copyto!(dest, src); false; catch e; e isa InexactError; end
  @chk threw "throw trial $t: InexactError"
  check_state(dest, d0.vals, d0.ids, "throw trial $t unchanged")
  @chk dest.layout_cache === d0.cache "throw trial $t cache"
end

rng = Xoshiro(SEED)
for t in 1:NTRIALS
  trial(rng, t)
  t % 10 == 0 && trial_throw(rng, t)
end
# the counterexample of the issue: an inactive -0.0 in src, a destination-only group
let gs = [1:1, 2:2], src = PB.BlockIterate(Float64, 2, gs), dest = PB.BlockIterate(Float64, 2, gs)
  src[1] = 1.0; src[2] = -0.0; dest[2] = 2.0
  copyto!(dest, src)
  @chk isequal(dest.storage, [1.0, 0.0]) && collect(PB.activecoordinates(dest)) == [1] "issue counterexample"
end
# zero-length iterate, DimensionMismatch
let z = PB.BlockIterate(Float64, 0, UnitRange{Int}[]), z2 = PB.BlockIterate(Float64, 0, UnitRange{Int}[])
  @chk copyto!(z, z2) === z && PB.nactive(z) == 0 "zero-length"
end
let d = PB.BlockIterate(Float64, 4, [1:2, 3:4]); d[1] = 1.0; d0 = snapshot(d)
  threw = try; copyto!(d, PB.BlockIterate(Float64, 6, 3)); false; catch e; e isa DimensionMismatch; end
  @chk threw "length mismatch throws"
  check_state(d, d0.vals, d0.ids, "length mismatch unchanged")
end
println("source=", ROOT, " checks=", checks[], " failures=", failures[])
exit(failures[] == 0 ? 0 : 1)
