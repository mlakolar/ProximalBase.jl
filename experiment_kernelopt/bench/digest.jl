# Compact digest of a formal campaign: one column per stratum (CPU kind × Julia version, its processes pooled), one row
# per (tree, case group) for the groups that the tree is meant to change, and per tree the number of other cases that
# are slower than 1.05 in every process of a stratum while the A/A control stays within [0.98, 1.02].
#   julia bench/digest.jl <results dir of the campaign> [--trees=A2,B] [--sizes=regex on the case id]
using Statistics, Printf
root = ARGS[1]
opt(name, d) = (i = findfirst(a -> startswith(a, "--$name="), ARGS); i === nothing ? d : String(split(ARGS[i], '=', limit = 2)[2]))
sizerx = Regex(opt("sizes", ""))
TARGETS = Dict("A" => r"^l1/(prox|prox_w|prox_g|prox_gw|valprox|valprox_w)/", "A2" => r"^l1/(prox|prox_w|prox_g|prox_gw|valprox|valprox_w)/",
               "B" => r"^l1/(prox_g|prox_gw)/", "C" => r"^l1/value/", "C2" => r"^l2/value/", "D" => r"^(l2|group)/(prox|valprox|block_sweep)/.*(zero|sparse)",
               "D2" => r"^(l2|group)/(prox|valprox|value|block_sweep)/.*(zero|sparse)", "P155" => r"^l1/(value_w|valprox_w)/", "F" => r"^group/block_sweep/",
               "ALL" => r"^(l1|l2|group)/", "MINE" => r"^(l1|l2|group)/",
               "A3" => r"^(l1/(prox|prox_w|valprox|valprox_w)|enet/(prox|prox_w|prox_g|valprox|block_loop))/", "V" => r"^(l1|l2)/value/",
               "D3" => r"^(l2|group)/(prox|valprox|block_sweep)/.*(zero|sparse)",
               "B2" => r"^l1/(prox_g|prox_gw)/", "B925" => r"^l1/(prox_g|prox_gw)/",
               "FINAL" => r"^((l1|l2|group)/(prox|prox_w|valprox|valprox_w|value|block_sweep)|enet/(prox|prox_w|prox_g|valprox|block_loop))/",
               "FINAL2" => r"^(l1|l2|group)/(prox|prox_w|valprox|valprox_w|value|block_sweep)/")
dirs = sort(filter(d -> isfile(joinpath(root, d, "DONE")) && isfile(joinpath(root, d, "raw.csv")), readdir(root)))
strata = Dict{String, Vector{String}}()
for d in dirs
  m = match(r"^(.*)-p\d+$", d); m === nothing && continue
  push!(get!(strata, m.captures[1], String[]), d)
end
snames = sort(collect(keys(strata)))
function load(ds)
  t = Dict{Tuple{String, String}, Dict{Int, Vector{Float64}}}(); ref = ""; bad = Set{Tuple{String, String}}(); cases = String[]
  for (di, d) in enumerate(ds)
    for (k, l) in enumerate(eachline(joinpath(root, d, "meta.csv")))
      k == 1 && continue; f = split(l, ',')
      ref = String(f[3]); f[1] in cases || push!(cases, String(f[1])); f[11] == "ok" || push!(bad, (String(f[1]), String(f[4])))
    end
    cur = Dict{Tuple{String, Int}, Dict{String, Float64}}()
    for (k, l) in enumerate(eachline(joinpath(root, d, "raw.csv")))
      k == 1 && continue; f = split(l, ',')
      get!(cur, (String(f[1]), parse(Int, f[3])), Dict{String, Float64}())[String(f[2])] = parse(Float64, f[5])
    end
    for ((c, _), row) in cur
      haskey(row, ref) || continue
      for (im, v) in row; push!(get!(get!(t, (c, im), Dict{Int, Vector{Float64}}()), di, Float64[]), v / row[ref]); end
    end
  end
  per = Dict(k => [median(v) for (_, v) in sort(collect(d); by = first)] for (k, d) in t if !(k in bad))
  per, ref, cases
end
data = Dict(s => load(strata[s]) for s in snames)
ref = data[snames[1]][2]; ctl = ref * "C"
trees = opt("trees", "") == "" ? sort([String(im) for im in unique(last.(keys(data[snames[1]][1]))) if im != ref && im != ctl]) : String.(split(opt("trees", ""), ','))
geo(v) = exp(mean(log.(v)))
grp(c) = join(split(c, '/')[1:3], '/')
println("Strata (processes): ", join(["$s ($(length(strata[s])))" for s in snames], ", "), "\n")
println("| tree | case group | ", join(snames, " | "), " |"); println("|:--|:--|", repeat(":--|", length(snames)))
for tr in trees
  rx = get(TARGETS, tr, r".")
  groups = unique(grp(c) for s in snames for c in data[s][3] if occursin(rx, c) && occursin(sizerx, c))
  for g in groups
    cells = String[]
    for s in snames
      per, _, cases = data[s]
      cs = [c for c in cases if grp(c) == g && occursin(rx, c) && occursin(sizerx, c) && haskey(per, (c, tr))]
      if isempty(cs); push!(cells, ""); continue; end
      meds = [median(per[(c, tr)]) for c in cs]
      push!(cells, @sprintf("%.2f [%.2f–%.2f]", geo(meds), minimum(minimum(per[(c, tr)]) for c in cs), maximum(maximum(per[(c, tr)]) for c in cs)))
    end
    println("| ", tr, " | ", g, " (", count(c -> grp(c) == g && occursin(rx, c) && occursin(sizerx, c), data[snames[1]][3]), ") | ", join(cells, " | "), " |")
  end
end
println("\nCases slower than 1.05 in every process of the stratum with the control within [0.98, 1.02] in every process (and the cases faster than 0.95 likewise), over all ", length(data[snames[1]][3]), " cases\n")
println("| tree | ", join(snames, " | "), " |"); println("|:--|", repeat(":--|", length(snames)))
detail = String[]
for tr in vcat(ctl, trees)
  cells = String[]
  for s in snames
    per, _, cases = data[s]
    slow = String[]; fast = 0; unres = 0
    for c in cases
      haskey(per, (c, tr)) || continue
      v = per[(c, tr)]; k = get(per, (c, ctl), [1.0])
      ok = minimum(k) >= 0.98 && maximum(k) <= 1.02
      if minimum(v) > 1.05; ok ? push!(slow, @sprintf("%s %.2f–%.2f", c, minimum(v), maximum(v))) : (unres += 1)
      elseif maximum(v) < 0.95 && ok; fast += 1; end
    end
    push!(cells, "$(length(slow)) slower, $fast faster ($unres unresolved)")
    isempty(slow) || push!(detail, "- `$tr` on $s: " * join(slow[1:min(end, 6)], "; ") * (length(slow) > 6 ? "; … ($(length(slow)) in all)" : ""))
  end
  println("| ", tr, " | ", join(cells, " | "), " |")
end
isempty(detail) || println("\nSlower cases:\n", join(detail, "\n"))
