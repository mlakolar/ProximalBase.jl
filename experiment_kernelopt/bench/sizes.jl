# One row per case, one column per stratum: the median over processes of the per-process median ratio of a tree to the
# reference, with the range over the processes, and the range of the A/A control.
#   julia bench/sizes.jl <results dir> <tree> <regex on the case id>
using Statistics, Printf
root, tree, rx = ARGS[1], ARGS[2], Regex(ARGS[3])
dirs = sort(filter(d -> isfile(joinpath(root, d, "DONE")), readdir(root)))
strata = Dict{String, Vector{String}}()
for d in dirs; m = match(r"^(.*)-p\d+$", d); m === nothing || push!(get!(strata, m.captures[1], String[]), d); end
snames = sort(collect(keys(strata)))
function load(ds)
  per = Dict{Tuple{String, String}, Vector{Float64}}(); cases = String[]; ref = ""
  for d in ds
    for (k, l) in enumerate(eachline(joinpath(root, d, "meta.csv"))); k == 1 && continue; f = split(l, ','); ref = String(f[3]); (occursin(rx, f[1]) && !(f[1] in cases)) && push!(cases, String(f[1])); end
    cur = Dict{Tuple{String, Int}, Dict{String, Float64}}()
    for (k, l) in enumerate(eachline(joinpath(root, d, "raw.csv"))); k == 1 && continue; f = split(l, ','); occursin(rx, f[1]) || continue; get!(cur, (String(f[1]), parse(Int, f[3])), Dict{String, Float64}())[String(f[2])] = parse(Float64, f[5]); end
    acc = Dict{Tuple{String, String}, Vector{Float64}}()
    for ((c, _), row) in cur, (im, v) in row; haskey(row, ref) && push!(get!(acc, (c, im), Float64[]), v / row[ref]); end
    for (k, v) in acc; push!(get!(per, k, Float64[]), median(v)); end
  end
  per, cases, ref
end
data = Dict(s => load(strata[s]) for s in snames)
ref = data[snames[1]][3]; cases = data[snames[1]][2]
println("Tree `", tree, "` against `", ref, "`; cell: median over processes [range over processes]; * marks a case whose A/A control leaves [0.98, 1.02] in some process\n")
println("| case | ", join(["$s ($(length(strata[s])))" for s in snames], " | "), " |"); println("|:--|", repeat(":--|", length(snames)))
for c in cases
  cells = String[]
  for s in snames
    per = data[s][1]
    if !haskey(per, (c, tree)); push!(cells, ""); continue; end
    v = per[(c, tree)]; k = get(per, (c, ref * "C"), [1.0])
    push!(cells, @sprintf("%.2f [%.2f–%.2f]%s", median(v), minimum(v), maximum(v), (minimum(k) < 0.98 || maximum(k) > 1.02) ? "*" : ""))
  end
  println("| ", c, " | ", join(cells, " | "), " |")
end
