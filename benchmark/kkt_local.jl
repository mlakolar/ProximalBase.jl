# Paired timing of the three- and four-argument kkt_violation (the screen of a coordinate assumed zero and the form that reads
# the point) between source trees compiled as modules of one process. Written for issue #980 and meant for a change that
# touches kkt_violation, its entry kernels or the way callers inline them.
#
#   PB_BASE=<root of the tree before the change> [PB_HEAD=<root of the tree under test>] \
#     [PB_HISTORY="name=<root>,name=<root>"] [PB_ROUNDS=15] [PB_GROUPS=ADCFBE] [PB_FILTER=<regular expression>] \
#     [PB_REVERSE=1] [PB_MODE=ir] [PB_CHECK_ONLY=1] \
#     julia --threads=1 --project=benchmark benchmark/kkt_local.jl
#
# A root is a directory that holds src/ProximalBase.jl. PB_BASE is required; PB_HEAD is the package root that contains this
# script when it is not given; PB_HISTORY lists earlier trees in the given order (a measurement of #980 names two of them,
# pre and stagea). The trees of a process, in this order, are the history trees, base, basectl (a second, separately
# compiled copy of base), head and headctl (a second copy of head). The two copies are the A/A controls: a row of base
# against basectl, or of head against headctl, shows how far two compilations of the same source differ in that row.
#
# A case is timed on a pool of fresh (v, x) pairs, in a loop that calls the form for the coordinates of the case. The time
# of a round is the minimum over batches of the time of one pass over the pool, per call; the rounds rotate the order of the
# trees. PB_ROUNDS is the number of rounds (default 15), PB_GROUPS selects groups of cases (the letters A to F; the groups
# run in the order A, D, C, F, B, E), PB_FILTER a regular expression on the case label, and PB_REVERSE=1 compiles and times
# the trees in the reversed order (the second process of a measurement). The groups are
#   A  zero coordinates of NormL1, ElasticNet (unweighted and weighted), SqrNormL2 (plain and centered), ZeroFunction,
#      LinearFunction and Scaled of NormL1, ElasticNet and SqrNormL2, in Float64 and Float32: the four-argument form at array,
#      view and SparseIterate points and the three-argument form at array points, with the results summed, stored, and
#      each call behind a function barrier
#   B  all coordinates of points with 50%, 90% and 99% zeros (lengths 2000 and 10^6), and the scan of the inactive
#      coordinates in the form of the loop that an active-set solver used (the largest score and the list of the positive
#      ones), with the four-argument form and, at array points, the three-argument form
#   C  GroupNormL2 and Scaled of it at zero groups of four entries, at array and BlockIterate points
#   D  where the work of the four-argument form comes from: the unweighted NormL1 at array points against four reference
#      kernels written in this file (the screen, the point read with a selection, that with the tests for non-finite values,
#      and one test of finiteness with the rules in a function that is not inlined), reported and not a design
#   E  one optimality check of an active-set design: the gradient X'r of a dense n x p design followed by the scan of the
#      inactive coordinates of a SparseIterate, and each of the two parts alone
#   F  a zero group of a BlockIterate and a zero coordinate of a SymmetricSparseIterate under the entrywise functions
# The rows over all coordinates of Scaled NormL1 in group B are left out: a history tree from before the methods of Scaled
# for a nonzero coordinate refuses a Scaled function there. A history tree that fails a case gives a SKIP line for it.
#
# PB_MODE=ir compares compiled code instead of timing: for the loop of each case of the selected groups (E excluded) and
# each tree the script takes the optimized LLVM text, replaces the names of the tree modules, numbered symbols and
# addresses, and counts lines, calls and calls whose text names kkt_violation. A tree whose text equals that of base is
# marked same, otherwise differs. Two trees whose loops compile to the same code are meant to give the same text, and equal
# text means that the loop is the same code for them, not that the machine code is placed alike.
#
# PB_CHECK_ONLY=1 runs the value comparison and the allocation measurement, and no timing rounds, for the cases
# of the selected groups (E excluded), and exits with status 1 when the values of base, basectl, head and headctl differ in
# a row, when one of the four trees allocates in a warmed pass, or when one of the four fails a case. The history trees
# are compared and reported and do not fail the run. The last line is CHECK PASS or CHECK FAIL.
#
# Output lines, tab separated:
#   ENV  the Julia version, the CPU, the settings, the trees (name, root, HEAD of a git checkout) and the load average
#   RAW  group, label, tree, calls per pass, bytes allocated in a warmed pass, and the time in ns per call of each round
#   VAL  group, label, whether the values of all trees agree, expected-equal or may-differ (the rows over all
#        coordinates of groups B and D, where a history tree or a reference kernel can give another value at a nonzero
#        coordinate), the values of the trees, and whether the four checked trees agree
#   IR   group, label, and per tree name=lines:calls:kkt_calls:same|differs (relative to base)
#   SKIP a case that a tree could not run, with the first words of its error
#   CHECK  one line per failed check, then CHECK PASS or CHECK FAIL (PB_CHECK_ONLY=1)
# benchmark/summarize_kkt_local.py reads these lines. Judge a row against its two controls: rows of one loop shape and one
# function move together, so a difference of the size of the controls in a single row is not evidence, and a second process
# with PB_REVERSE=1 is the check. All trees run in one thread, with the bounds checks of the command line.
using Random, LinearAlgebra, InteractiveUtils
Threads.nthreads() == 1 || error("one Julia thread required")
BLAS.set_num_threads(1)

const PKGROOT = dirname(@__DIR__)
function sourceroot(s)
  root = abspath(strip(String(s)))
  isfile(joinpath(root, "src", "ProximalBase.jl")) || error("not a source root (no src/ProximalBase.jl): ", root)
  root
end
haskey(ENV, "PB_BASE") || error("PB_BASE, a source root of the tree before the change, is required")
const SPEC = Tuple{String, String}[]
for item in split(get(ENV, "PB_HISTORY", ""), ","; keepempty = false)
  kv = split(item, "="; limit = 2)
  length(kv) == 2 || error("PB_HISTORY entries are name=root, got ", item)
  name = String(strip(kv[1]))
  name in ("base", "basectl", "head", "headctl") && error("PB_HISTORY cannot use the name ", name)
  push!(SPEC, (name, sourceroot(kv[2])))
end
let baseroot = sourceroot(ENV["PB_BASE"]), headroot = sourceroot(get(ENV, "PB_HEAD", PKGROOT))
  push!(SPEC, ("base", baseroot), ("basectl", baseroot), ("head", headroot), ("headctl", headroot))
end
const ROUNDS = parse(Int, get(ENV, "PB_ROUNDS", "15"))
const GROUPS = get(ENV, "PB_GROUPS", "ADCFBE")
const FILTER = Regex(get(ENV, "PB_FILTER", ""))
const MODE = get(ENV, "PB_MODE", "time")                 # "time", or "ir": compare the optimized LLVM code of the loops, no timing
const REVERSE = get(ENV, "PB_REVERSE", "0") == "1"        # reversed order of compilation and of the trees within a round
const CHECK_ONLY = get(ENV, "PB_CHECK_ONLY", "0") == "1"  # no timing rounds
MODE in ("time", "ir") || error("PB_MODE is time or ir")
(MODE == "ir" && CHECK_ONLY) && error("PB_MODE=ir and PB_CHECK_ONLY=1 are separate runs")

const NAMES = Tuple(first.(SPEC))
const TMODS = Module[Module(Symbol("PBT", j)) for j in eachindex(SPEC)]
for j in (REVERSE ? reverse(eachindex(SPEC)) : eachindex(SPEC))
  Base.include(TMODS[j], joinpath(SPEC[j][2], "src", "ProximalBase.jl"))
end
const MODS = Tuple(m.ProximalBase for m in TMODS)
const IBASE = findfirst(==("base"), NAMES)
const CORENAMES = ("base", "basectl", "head", "headctl")
iscore(name) = name in CORENAMES

cpumodel() = try
  first(l for l in eachline("/proc/cpuinfo") if startswith(l, "model name"))
catch
  "unknown"
end
loadavg() = try read("/proc/loadavg", String) catch; "unknown" end
function treehead(root)
  (isdir(joinpath(root, ".git")) || isfile(joinpath(root, ".git"))) || return "no git"
  try
    strip(read(pipeline(`git -C $root rev-parse HEAD`; stderr = devnull), String))
  catch
    "no git"
  end
end
println("ENV julia=", VERSION, " cpu_name=", Sys.CPU_NAME, " ", cpumodel(), " threads=", Threads.nthreads(),
        " blas=", BLAS.get_num_threads(), " rounds=", ROUNDS, " checkbounds=", Base.JLOptions().check_bounds, " mode=", MODE,
        " reverse=", REVERSE, " check_only=", CHECK_ONLY, " groups=", GROUPS, " filter=", FILTER.pattern)
println("ENV trees: ", join(("$(n) $(r) $(treehead(r))" for (n, r) in SPEC), "; "))
println("ENV loadavg start: ", strip(loadavg()))

# ---- the call forms -----------------------------------------------------------------------------------------------------
struct Four{K}; k::K; end
struct Three{K}; k::K; end
@inline (c::Four)(f, v, x, i) = c.k(f, v, x, i)
@inline (c::Three)(f, v, x, i) = c.k(f, v, i)

# ---- reference kernels outside the package (unweighted NormL1 on arrays; f is the coefficient) ---------------------------
# the screen: x is not read
@inline ref_screen(c::T, v::Vector{T}, x::Vector{T}, i::Integer) where {T} = abs(v[i]) - c
# x is read and the score is selected; no rules for non-finite values
@inline function ref_point(c::T, v::Vector{T}, x::Vector{T}, i::Integer) where {T}
  length(v) == length(x) || throw(DimensionMismatch("lengths"))
  vi = v[i]; xi = x[i]
  ifelse(iszero(xi), abs(vi) - c, abs(vi + c * sign(xi)))
end
# ref_point and the four tests for non-finite values, as the kernel of the package writes them
@inline function ref_rules(c::T, v::Vector{T}, x::Vector{T}, i::Integer) where {T}
  length(v) == length(x) || throw(DimensionMismatch("lengths"))
  vi = v[i]; xi = x[i]
  r = ifelse(iszero(xi), abs(vi) - c, abs(vi + c * sign(xi)))
  (isnan(vi) | isnan(xi)) && return T(NaN)
  (isinf(vi) | isinf(xi)) && return T(Inf)
  r
end
# ref_point and one test of finiteness; the rules are in a function that is not inlined
@noinline ref_nonfinite(v::T, x::T) where {T} = (isnan(v) | isnan(x)) ? T(NaN) : T(Inf)
@inline function ref_cold(c::T, v::Vector{T}, x::Vector{T}, i::Integer) where {T}
  length(v) == length(x) || throw(DimensionMismatch("lengths"))
  vi = v[i]; xi = x[i]
  r = ifelse(iszero(xi), abs(vi) - c, abs(vi + c * sign(xi)))
  t = (vi - vi) + (xi - xi)
  t == 0 ? r : ref_nonfinite(vi, xi)
end

# ---- loop shapes: (kkt, f, v, x, idx, act) -> a number --------------------------------------------------------------------
@noinline function lsum(kkt::K, f::F, v::Vector{T}, x::X, idx::Vector{Int}, act::A) where {K, F, T, X, A}
  s = zero(T)
  @inbounds for i in idx
    s += kkt(f, v, x, i)
  end
  s
end
const OUT64 = zeros(Float64, 1 << 21); const OUT32 = zeros(Float32, 1 << 21)
out(::Type{Float64}) = OUT64; out(::Type{Float32}) = OUT32
@noinline function lstore(kkt::K, f::F, v::Vector{T}, x::X, idx::Vector{Int}, act::A) where {K, F, T, X, A}
  o = out(T)
  @inbounds for j in eachindex(idx)
    o[j] = kkt(f, v, x, idx[j])
  end
  @inbounds o[1] + o[length(idx)]
end
@noinline callb(kkt::K, f::F, v, x, i) where {K, F} = kkt(f, v, x, i)
@noinline function lbarrier(kkt::K, f::F, v::Vector{T}, x::X, idx::Vector{Int}, act::A) where {K, F, T, X, A}
  s = zero(T)
  @inbounds for i in idx
    s += callb(kkt, f, v, x, i)
  end
  s
end
# all coordinates, zero or not
@noinline function lall(kkt::K, f::F, v::Vector{T}, x::X, idx::Vector{Int}, act::A) where {K, F, T, X, A}
  s = zero(T)
  for i in 1:length(v)
    s += kkt(f, v, x, i)
  end
  s
end
# the loop of an active-set solver: the coordinates outside the active set are scored, the largest score is kept and the
# coordinates with a positive score are collected
const VIOL = Int[]; const VAL64 = Float64[]; const VAL32 = Float32[]
vals(::Type{Float64}) = VAL64; vals(::Type{Float32}) = VAL32
@noinline function lscan(kkt::K, f::F, v::Vector{T}, x::X, idx::Vector{Int}, act::A) where {K, F, T, X, A}
  viol = VIOL; vl = vals(T)
  empty!(viol); empty!(vl)
  m = T(-Inf)
  for k in 1:length(v)
    act(x, k) && continue
    s = convert(T, kkt(f, v, x, k))
    m = max(m, s)
    if s > 0
      push!(viol, k); push!(vl, s)
    end
  end
  m + length(viol)
end
vecactive(x::AbstractVector, k) = !iszero(@inbounds x[k])

@noinline function batch(loop::L, kkt::K, f::F, pool::P, act::A) where {L, K, F, P, A}
  s = 0.0
  for (v, x, idx) in pool
    s += loop(kkt, f, v, x, idx, act)
  end
  s
end
const SINK = Ref(0.0)
@noinline function timeit(loop::L, kkt::K, f::F, pool::P, act::A, ncalls::Int, batches::Int) where {L, K, F, P, A}
  best = typemax(UInt64)
  for _ in 1:batches
    t = time_ns(); s = batch(loop, kkt, f, pool, act); dt = time_ns() - t
    SINK[] += s; best = min(best, dt)
  end
  best / ncalls
end
@noinline allocs(loop::L, kkt::K, f::F, pool::P, act::A) where {L, K, F, P, A} = @allocated batch(loop, kkt, f, pool, act)

# a participant: name, call form, function, pool, activity test
struct Part
  name::String
  kkt::Any
  f::Any
  pool::Any
  act::Any
end
median_(a) = (s = sort(a); n = length(s); isodd(n) ? s[(n + 1) >> 1] : (s[n >> 1] + s[(n >> 1) + 1]) / 2)
sameval(a, b) = a == b || (isnan(a) && isnan(b))
const FAILS = String[]
function runcase(group, label, parts::Vector{Part}, loop, ncalls; batches = 25, samevalues = true)
  occursin(FILTER, label) || return
  MODE == "ir" && return ircase(group, label, parts, loop)
  n = length(parts)
  valsum = Vector{Float64}(undef, n)
  for j in 1:n
    p = parts[j]
    try
      valsum[j] = batch(loop, p.kkt, p.f, p.pool, p.act)
      CHECK_ONLY || timeit(loop, p.kkt, p.f, p.pool, p.act, ncalls, 3)
    catch err
      println("SKIP\t", group, "\t", label, "\t[", p.name, "] ", sprint(showerror, err)[1:min(end, 160)])
      iscore(p.name) && push!(FAILS, string(group, " ", label, ": ", p.name, " failed the case"))
      return
    end
  end
  al = map(p -> (allocs(loop, p.kkt, p.f, p.pool, p.act); allocs(loop, p.kkt, p.f, p.pool, p.act)), parts)
  core = [j for j in 1:n if iscore(parts[j].name)]
  same = all(v -> sameval(v, valsum[1]), valsum)
  samecore = all(j -> sameval(valsum[j], valsum[first(core)]), core)
  if CHECK_ONLY
    samecore || push!(FAILS, string(group, " ", label, ": the values of the four checked trees differ"))
    any(j -> al[j] > 0, core) && push!(FAILS, string(group, " ", label, ": a checked tree allocates (",
                                                      join((string(parts[j].name, "=", al[j]) for j in core), ", "), " bytes)"))
  else
    R = zeros(ROUNDS, n)
    for r in 1:ROUNDS, o in 0:(n - 1)
      j = REVERSE ? n + 1 - mod1(r + o, n) : mod1(r + o, n)
      p = parts[j]
      R[r, j] = timeit(loop, p.kkt, p.f, p.pool, p.act, ncalls, batches)
    end
    for j in 1:n
      println("RAW\t", group, "\t", label, "\t", parts[j].name, "\t", ncalls, "\t", al[j], "\t",
              join((string(round(R[r, j], digits = 4)) for r in 1:ROUNDS), ","))
    end
  end
  println("VAL\t", group, "\t", label, "\t", same, "\t", samevalues ? "expected-equal" : "may-differ", "\t",
          join(valsum, ","), "\t", samecore)
  print(rpad(string(group, " ", label), 58), CHECK_ONLY ? " check" : " ns/call")
  if !CHECK_ONLY
    for j in 1:n
      print(" ", parts[j].name, "=", round(median_(R[:, j]), digits = 2))
    end
  end
  println(" alloc=", maximum(al), " same=", same)
  flush(stdout)
end

# the optimized LLVM code of the loop for one tree, with the names of the tree modules, numbered symbols and addresses
# replaced; two trees whose loops compile to the same code are meant to give the same text
function irtext(loop, p::Part)
  v, x, idx = first(p.pool)
  tt = Tuple{typeof(p.kkt), typeof(p.f), typeof(v), typeof(x), typeof(idx), typeof(p.act)}
  s = sprint(io -> code_llvm(io, loop, tt; debuginfo = :none))
  replace(s, r"PBT\d+" => "TREE", r"_\d+" => "_N", r"#\d+" => "#N", r"\b\d{7,}\b" => "PTR")
end
function ircase(group, label, parts, loop)
  local txt
  try
    txt = map(p -> irtext(loop, p), parts)
  catch err
    println("SKIP\t", group, "\t", label, "\t", sprint(showerror, err)[1:min(end, 160)])
    return
  end
  ib = findfirst(p -> p.name == "base", parts)
  desc = map(eachindex(parts)) do j
    ls = split(txt[j], '\n')
    calls = count(l -> occursin(r"\bcall\b", l) && occursin(r"@\"?j(ulia|fptr|lsys)?_", l), ls)
    kcalls = count(l -> occursin(r"\bcall\b", l) && occursin("kkt_violation", l), ls)
    string(parts[j].name, "=", length(ls), ":", calls, ":", kcalls, ":", ib === nothing ? "?" : txt[j] == txt[ib] ? "same" : "differs")
  end
  println("IR\t", group, "\t", label, "\t", join(desc, ";"))
  flush(stdout)
end

# ---- inputs ---------------------------------------------------------------------------------------------------------------
function densepair(::Type{T}, p, zfrac, seed) where {T}
  rng = MersenneTwister(seed)
  x = randn(rng, T, p); x[rand(rng, p) .< zfrac] .= 0
  v = randn(rng, T, p)
  v, x
end
point(M, kind, x) = kind === :vector ? x : kind === :sparse ? M.SparseIterate(x) : kind === :view ? view(x, 1:length(x)) : error("kind")
actfun(M, kind) = kind === :sparse ? M.isactive : vecactive
# the dense pairs of a case are built once and shared by the trees: an array point is the same array for each tree, and an
# iterate is built from the same values by each tree's own constructor (v is shared)
const DENSE = Dict{Any, Any}()
function densepool(::Type{T}, p, zfrac, npairs) where {T}
  key = (T, p, zfrac, npairs)
  if !haskey(DENSE, key)
    empty!(DENSE)
    DENSE[key] = [densepair(T, p, zfrac, 1000 + s) for s in 1:npairs]
  end
  DENSE[key]::Vector{Tuple{Vector{T}, Vector{T}}}
end
function mkpool(M, ::Type{T}, kind, p, zfrac, npairs; zeros_only = true) where {T}
  map(densepool(T, p, zfrac, npairs)) do (v, x)
    (v, point(M, kind, x), zeros_only ? findall(iszero, x) : Int[])
  end
end
function makef(M, name, ::Type{T}, p; level = T(0.5)) where {T}
  rng = MersenneTwister(77)
  w = T(0.5) .+ rand(rng, T, p)
  y = randn(rng, T, p)
  name === :NormL1 ? M.NormL1(level) :
  name === :NormL1w ? M.NormL1(level, w) :
  name === :ElasticNet ? M.ElasticNet(level, T(0.25)) :
  name === :ElasticNetw ? M.ElasticNet(level, T(0.25), w) :
  name === :SqrNormL2 ? M.SqrNormL2(T(0.5)) :
  name === :SqrNormL2c ? M.SqrNormL2(T(0.5), y) :
  name === :ZeroFunction ? M.ZeroFunction() :
  name === :LinearFunction ? M.LinearFunction(y) :
  name === :ScaledNormL1 ? M.Scaled(M.NormL1(level), T(2)) :
  name === :ScaledElasticNet ? M.Scaled(M.ElasticNet(level, T(0.25)), T(2)) :
  name === :ScaledSqrNormL2 ? M.Scaled(M.SqrNormL2(T(0.5)), T(2)) : error("name")
end
treeparts(form, name, ::Type{T}, kind, p, zfrac, npairs; zeros_only = true, level = T(0.5)) where {T} =
  [Part(NAMES[j], form(MODS[j].kkt_violation), makef(MODS[j], name, T, p; level = level),
        mkpool(MODS[j], T, kind, p, zfrac, npairs; zeros_only = zeros_only), actfun(MODS[j], kind)) for j in 1:length(MODS)]

# ---- group A: zero coordinates, the three loop shapes ------------------------------------------------------------------------
function groupA()
  p = 2000
  for T in (Float64, Float32), name in (:NormL1, :NormL1w, :ElasticNet, :ElasticNetw, :SqrNormL2, :SqrNormL2c, :ZeroFunction,
                                         :LinearFunction, :ScaledNormL1, :ScaledElasticNet, :ScaledSqrNormL2)
    for (kind, forms) in ((:vector, ((Four, "4"), (Three, "3"))), (:sparse, ((Four, "4"),)), (:view, ((Four, "4"),)))
      for (form, fl) in forms, (loop, sl) in ((lsum, "sum"), (lstore, "store"), (lbarrier, "barrier"))
        label = "$name $T $kind $(fl)arg $sl zero p=$p"
        occursin(FILTER, label) || continue
        parts = treeparts(form, name, T, kind, p, 0.5, 12)
        ncalls = sum(length(t[3]) for t in parts[1].pool)
        runcase("A", label, parts, loop, ncalls)
      end
    end
  end
end
# ---- group F: the other point types: a group of four zero entries of a BlockIterate under an entrywise function, and a zero
# coordinate of a SymmetricSparseIterate ------------------------------------------------------------------------------------
function groupF()
  p = 2000; g = 4
  groups = [collect(((k - 1) * g + 1):(k * g)) for k in 1:(p ÷ g)]
  for T in (Float64, Float32), name in (:NormL1, :NormL1w, :ElasticNet, :ElasticNetw, :SqrNormL2)
    parts = map(1:length(MODS)) do j
      M = MODS[j]
      pool = map(1:12) do s
        rng = MersenneTwister(2000 + s)
        x = randn(rng, T, p)
        zg = findall(rand(rng, length(groups)) .< 0.5)
        for k in zg; x[groups[k]] .= 0; end
        (randn(rng, T, p), M.BlockIterate(x, groups), zg)
      end
      Part(NAMES[j], Four(M.kkt_violation), makef(M, name, T, p), pool, vecactive)
    end
    ncalls = sum(length(t[3]) for t in parts[1].pool)
    for (loop, sl) in ((lsum, "sum"), (lbarrier, "barrier"))
      runcase("F", "$name $T block 4arg $sl zero g=$g", parts, loop, ncalls)
    end
  end
  q = 60; nc = q * (q + 1) ÷ 2
  for T in (Float64, Float32), name in (:NormL1, :ElasticNet, :SqrNormL2)
    parts = map(1:length(MODS)) do j
      M = MODS[j]
      pool = map(1:12) do s
        rng = MersenneTwister(3000 + s)
        X = randn(rng, T, q, q); X = X + X'
        Z = rand(rng, q, q) .< 0.5; Z = Z .& Z'
        X[Z] .= 0
        zc = [k for k in 1:nc if (rc = M.tril_cartesian(q, k); iszero(X[rc[1], rc[2]]))]
        (randn(rng, T, q * q), M.SymmetricSparseIterate(X), zc)
      end
      Part(NAMES[j], Four(M.kkt_violation), makef(M, name, T, q * q), pool, vecactive)
    end
    ncalls = sum(length(t[3]) for t in parts[1].pool)
    for (loop, sl) in ((lsum, "sum"), (lbarrier, "barrier"))
      runcase("F", "$name $T symmetric 4arg $sl zero q=$q", parts, loop, ncalls)
    end
  end
end
# ---- group B: all coordinates of points with a share of zeros, and the scan of the inactive coordinates ----------------------
function groupB()
  for T in (Float64, Float32), name in (:NormL1, :ElasticNet, :SqrNormL2, :ScaledNormL1)
    for kind in (:vector, :sparse), (p, npairs) in ((2000, 12), (1_000_000, 3)), zfrac in (0.5, 0.9, 0.99)
      p == 1_000_000 && zfrac == 0.5 && continue
      # the trees before the scaled method for a nonzero coordinate refuse a Scaled function there: no row over all coordinates for it
      if name !== :ScaledNormL1
        label = "$name $T $kind 4arg all z=$zfrac p=$p"
        if occursin(FILTER, label)
          parts = treeparts(Four, name, T, kind, p, zfrac, npairs; zeros_only = false)
          runcase("B", label, parts, lall, p * npairs; samevalues = false)
        end
      end
      # the scan: a level that leaves about 1% of the scores positive
      label = "$name $T $kind 4arg scan z=$zfrac p=$p"
      if occursin(FILTER, label)
        parts = treeparts(Four, name, T, kind, p, zfrac, npairs; zeros_only = false, level = T(2.5))
        ninact = sum(count(iszero, t[2]) for t in densepool(T, p, zfrac, npairs))
        runcase("B", label, parts, lscan, ninact)
      end
      label = "$name $T $kind 3arg scan z=$zfrac p=$p"
      if kind === :vector && occursin(FILTER, label)
        parts = treeparts(Three, name, T, kind, p, zfrac, npairs; zeros_only = false, level = T(2.5))
        ninact = sum(count(iszero, t[2]) for t in densepool(T, p, zfrac, npairs))
        runcase("B", label, parts, lscan, ninact)
      end
    end
  end
end
# ---- group C: GroupNormL2 at zero groups of four entries -------------------------------------------------------------------
function groupC()
  p = 2000; g = 4
  groups = [collect(((k - 1) * g + 1):(k * g)) for k in 1:(p ÷ g)]
  for T in (Float64, Float32), scaled in (false, true), kind in (:vector, :block)
    parts = map(1:length(MODS)) do j
      M = MODS[j]
      f0 = M.GroupNormL2(T(0.5), groups)
      f = scaled ? M.Scaled(f0, T(2)) : f0
      pool = map(1:12) do s
        rng = MersenneTwister(2000 + s)
        x = randn(rng, T, p)
        zg = findall(rand(rng, length(groups)) .< 0.5)
        for k in zg; x[groups[k]] .= 0; end
        v = randn(rng, T, p)
        kind === :vector ? (v, x, [first(groups[k]) for k in zg]) : (v, M.BlockIterate(x, groups), zg)
      end
      Part(NAMES[j], Four(M.kkt_violation), f, pool, vecactive)
    end
    ncalls = sum(length(t[3]) for t in parts[1].pool)
    for (loop, sl) in ((lsum, "sum"), (lbarrier, "barrier"))
      runcase("C", "$(scaled ? "Scaled" : "")GroupNormL2 $T $kind 4arg $sl zero g=$g", parts, loop, ncalls)
    end
  end
end
# ---- group D: where the work of the unweighted NormL1 kernel comes from -----------------------------------------------------
function groupD()
  p = 2000
  for T in (Float64, Float32), (loop, sl) in ((lsum, "sum"), (lstore, "store"), (lall, "all"))
    zo = loop !== lall
    pool = mkpool(MODS[IBASE], T, :vector, p, 0.5, 12; zeros_only = zo)
    c = T(0.5)
    parts = Part[Part("ref_screen", ref_screen, c, pool, vecactive), Part("ref_point", ref_point, c, pool, vecactive),
                 Part("ref_rules", ref_rules, c, pool, vecactive), Part("ref_cold", ref_cold, c, pool, vecactive)]
    for j in 1:length(MODS)
      NAMES[j] in ("pre", "base", "basectl", "head", "headctl") || continue
      push!(parts, Part(NAMES[j], Four(MODS[j].kkt_violation), makef(MODS[j], :NormL1, T, p), pool, vecactive))
    end
    ncalls = zo ? sum(length(t[3]) for t in pool) : 12p
    runcase("D", "NormL1 $T vector kernels $sl p=$p", parts, loop, ncalls; samevalues = zo)
  end
end
# ---- group E: one optimality check of the active-set design: the gradient X'r and the scan of the inactive coordinates ---------
@noinline function lcheck(kkt::K, f::F, v::Vector{T}, x::X, idx::Vector{Int}, act::A) where {K, F, T, X, A}
  mul!(v, transpose(XMAT[]::Matrix{T}), RES[]::Vector{T})
  lscan(kkt, f, v, x, idx, act)
end
@noinline function lgrad(kkt::K, f::F, v::Vector{T}, x::X, idx::Vector{Int}, act::A) where {K, F, T, X, A}
  mul!(v, transpose(XMAT[]::Matrix{T}), RES[]::Vector{T})
  v[1]
end
const XMAT = Ref{Any}(); const RES = Ref{Any}()
function groupE()
  for T in (Float64, Float32), (n, p) in ((50, 2000), (200, 2000), (200, 20000), (1000, 20000), (100, 200000))
    rng = MersenneTwister(31)
    X = randn(rng, T, n, p); r = randn(rng, T, n) ./ T(sqrt(n))
    XMAT[] = X; RES[] = r
    xd = zeros(T, p); xd[randperm(rng, p)[1:20]] .= randn(rng, T, 20)
    vbuf = zeros(T, p)
    for name in (:NormL1, :ElasticNet)
      parts = map(1:length(MODS)) do j
        M = MODS[j]
        Part(NAMES[j], Four(M.kkt_violation), makef(M, name, T, p; level = T(2.5)), [(vbuf, M.SparseIterate(xd), Int[])], M.isactive)
      end
      for (loop, sl) in ((lcheck, "check"), (lscan, "scanonly"), (lgrad, "gradonly"))
        mul!(vbuf, transpose(X), r)
        runcase("E", "$name $T sparse $sl n=$n p=$p", parts, loop, p - 20; batches = 9)
      end
    end
  end
end

for g in GROUPS
  g in "ABCDEF" || error("PB_GROUPS holds the letters A to F, got ", GROUPS)
end
'A' in GROUPS && groupA()
'D' in GROUPS && groupD()
'C' in GROUPS && groupC()
'F' in GROUPS && groupF()
'B' in GROUPS && groupB()
'E' in GROUPS && MODE != "ir" && !CHECK_ONLY && groupE()
println("ENV loadavg end: ", strip(loadavg()))
println("DONE sink=", SINK[])
if CHECK_ONLY
  for f in FAILS
    println("CHECK\t", f)
  end
  println(isempty(FAILS) ? "CHECK PASS" : "CHECK FAIL")
  isempty(FAILS) || exit(1)
end
