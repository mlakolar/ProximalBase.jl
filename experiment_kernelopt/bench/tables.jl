# Markdown tables from result directories of run.jl.
#   julia bench/tables.jl trees  [--filter=regex] DIR...   tree-against-reference summary, one process per directory
#   julia bench/tables.jl cases  [--filter=regex] [--impls=a,b,c] DIR...   one row per case, pooled ratios of the implementations
# A ratio is the median over rounds of the paired per-round ratio to the case's reference implementation. "pooled" takes
# all rounds of all directories; "lo".."hi" is the range of the per-directory (per-process) medians.
using Statistics, Printf
mode = ARGS[1]
dirs = filter(a -> !startswith(a, "--"), ARGS[2:end])
opt(name) = (i = findfirst(a -> startswith(a, "--$name="), ARGS); i === nothing ? nothing : String(split(ARGS[i], '=', limit = 2)[2]))
rx = opt("filter") === nothing ? r"" : Regex(opt("filter"))

# the directories must be one stratum: the same Julia version, CPU model, reference and tree commits
function stratum(d)
  kv = Dict{String, String}()
  for l in eachline(joinpath(d, "provenance.txt"))
    occursin(" = ", l) || continue
    k, v = split(l, " = ", limit = 2)
    (k in ("julia", "cpu_model", "KO_REF", "KO_BATCH_MS", "KO_SAMPLES") || startswith(k, "tree.")) && (kv[k] = startswith(k, "tree.") ? String(last(split(v, " @ "))) : String(v))
  end
  kv
end
if get(ENV, "KO_ALLOW_MIXED", "0") != "1" && length(dirs) > 1
  s1 = stratum(dirs[1])
  for d in dirs[2:end]
    stratum(d) == s1 || error("$(d) is not in the stratum of $(dirs[1]): $(setdiff(collect(stratum(d)), collect(s1)))")
  end
end
bad = Set{Tuple{String, String}}()
times = Dict{Tuple{String, String}, Dict{Tuple{Int, Int}, Float64}}()
refof = Dict{String, String}(); allocs = Dict{Tuple{String, String}, String}()
caseorder = String[]; implorder = String[]
for (di, d) in enumerate(dirs)
  for (k, line) in enumerate(eachline(joinpath(d, "meta.csv")))
    k == 1 && continue
    f = split(line, ',')
    c, im = String(f[1]), String(f[4])
    haskey(refof, c) || (push!(caseorder, c); refof[c] = String(f[3]))
    im in implorder || push!(implorder, im)
    allocs[(c, im)] = String(f[7])
    f[11] == "ok" || push!(bad, (c, im))
  end
  for (k, line) in enumerate(eachline(joinpath(d, "raw.csv")))
    k == 1 && continue
    f = split(line, ',')
    get!(times, (String(f[1]), String(f[2])), Dict{Tuple{Int, Int}, Float64}())[(di, parse(Int, f[3]))] = parse(Float64, f[5])
  end
end
filter!(c -> occursin(rx, c), caseorder)
# pooled median ratio, per-directory medians, median time
function stats(c, im)
  (haskey(times, (c, im)) && !((c, im) in bad)) || return nothing
  t, tr = times[(c, im)], times[(c, refof[c])]
  ks = collect(intersect(keys(t), keys(tr)))
  isempty(ks) && return nothing
  per = [median([t[k] / tr[k] for k in ks if k[1] == di]) for di in unique(first.(ks))]
  (pooled = median([t[k] / tr[k] for k in ks]), lo = minimum(per), hi = maximum(per), ns = median(collect(values(t))), nproc = length(per))
end
geomean(v) = exp(mean(log.(v)))
# family/op/type of a case id, e.g. "l1/prox_w/Float64"
group(c) = join(split(c, '/')[1:3], '/')

if mode == "cases"
  impls = opt("impls") === nothing ? implorder : String.(split(opt("impls"), ','))
  println("| case | reference ns | ", join(impls, " | "), " |"); println("|:--|--:|", repeat("--:|", length(impls)))
  for c in caseorder
    r = stats(c, refof[c]); r === nothing && continue
    cells = String[]
    for im in impls
      s = stats(c, im)
      push!(cells, s === nothing ? "" : @sprintf("%.2f", s.pooled) * (get(allocs, (c, im), "0") == "0" ? "" : "ᵃ"))
    end
    @printf("| %s | %.1f | %s |\n", c, r.ns, join(cells, " | "))
  end
elseif mode == "trees"
  ref = refof[first(caseorder)]; ctl = ref * "C"
  treeimpls = [im for im in implorder if im != ref && !startswith(im, "k_") && !startswith(im, "kC_") && im != "po"]
  groups = unique(group.(caseorder))
  println("Reference `", ref, "`; ", length(dirs), " processes; ", length(caseorder), " cases.\n")
  println("**Geometric mean of the pooled ratios by case group** (minimum and maximum pooled ratio of a case in the group in brackets)\n")
  println("| group | cases | ", join(treeimpls, " | "), " |"); println("|:--|--:|", repeat(":--|", length(treeimpls)))
  for g in groups
    cs = [c for c in caseorder if group(c) == g]
    cells = String[]
    for im in treeimpls
      v = [s.pooled for s in (stats(c, im) for c in cs) if s !== nothing]
      push!(cells, isempty(v) ? "" : @sprintf("%.3f [%.2f, %.2f]", geomean(v), minimum(v), maximum(v)))
    end
    @printf("| %s | %d | %s |\n", g, length(cs), join(cells, " | "))
  end
  println("\n**Decision by process** (a process is one fresh Julia process with its own compilation order; a case counts as slower when its median ratio is above 1.05 in every process and as faster when it is below 0.95 in every process; `unresolved` are those of them whose A/A control leaves [0.98, 1.02] in some process)\n")
  println("| tree | cases | slower in every process | of them unresolved | faster in every process | of them unresolved | worst slower case: per-process range; control range |"); println("|:--|--:|--:|--:|--:|--:|:--|")
  ctlok(c) = (k = stats(c, ctl); k !== nothing && k.lo >= 0.98 && k.hi <= 1.02)
  for im in treeimpls
    slow = Tuple{Float64, String}[]; fast = 0; n = 0; su = 0; fu = 0
    for c in caseorder
      s = stats(c, im); s === nothing && continue
      n += 1
      if s.lo > 1.05
        push!(slow, (s.lo, c)); ctlok(c) || (su += 1)
      elseif s.hi < 0.95
        fast += 1; ctlok(c) || (fu += 1)
      end
    end
    sort!(slow; rev = true)
    worst = isempty(slow) ? "" : begin
      c = slow[1][2]; s = stats(c, im); k = stats(c, ctl)
      @sprintf("%s [%.3f, %.3f]; control [%.3f, %.3f]", c, s.lo, s.hi, k.lo, k.hi)
    end
    @printf("| %s | %d | %d | %d | %d | %d | %s |\n", im, n, length(slow), su, fast, fu, worst)
  end
  if opt("list") !== nothing
    im = opt("list")
    println("\n**All cases of `", im, "` slower than 1.05 in every process**\n\n| case | ref ns | pooled | process range | control pooled | control range |\n|:--|--:|--:|:--|--:|:--|")
    for c in caseorder
      s = stats(c, im); s === nothing && continue
      s.lo > 1.05 || continue
      k = stats(c, ctl); r = stats(c, ref)
      @printf("| %s | %.1f | %.3f | [%.3f, %.3f] | %.3f | [%.3f, %.3f] |\n", c, r.ns, s.pooled, s.lo, s.hi, k.pooled, k.lo, k.hi)
    end
  end
end
