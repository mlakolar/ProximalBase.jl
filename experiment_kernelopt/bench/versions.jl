# Julia-version comparison from two result directories of the same node and suites:
#   julia bench/versions.jl [--filter=regex] [--min=1.25] DIR_NEW DIR_OLD
# For every case: the reference implementation's time in each version and their ratio old/new, next to the same ratio
# for a named bare kernel where the case has one. Lists the cases whose ratio is beyond --min either way.
using Statistics, Printf
dirs = filter(a -> !startswith(a, "--"), ARGS)
opt(name, d) = (i = findfirst(a -> startswith(a, "--$name="), ARGS); i === nothing ? d : String(split(ARGS[i], '=', limit = 2)[2]))
rx = Regex(opt("filter", "")); thr = parse(Float64, opt("min", "1.25"))
function load(d)
  t = Dict{Tuple{String, String}, Vector{Float64}}(); ref = Dict{String, String}(); order = String[]
  for (k, l) in enumerate(eachline(joinpath(d, "meta.csv"))); k == 1 && continue; f = split(l, ','); haskey(ref, f[1]) || (push!(order, String(f[1])); ref[String(f[1])] = String(f[3])); end
  for (k, l) in enumerate(eachline(joinpath(d, "raw.csv"))); k == 1 && continue; f = split(l, ','); push!(get!(t, (String(f[1]), String(f[2])), Float64[]), parse(Float64, f[5])); end
  Dict(k => median(v) for (k, v) in t), ref, order
end
tn, ref, order = load(dirs[1]); to, _, _ = load(dirs[2])
bare = ["k_st_maxmin", "k_simd", "k_two_st_maxmin", "k_direct", "k_norm_simd", "k_halfsumsq_simd", "k_three", "k_chol"]
println("| case | PB on new (ns) | PB on old (ns) | old/new PB | old/new bare kernel |"); println("|:--|--:|--:|--:|--:|")
rs = Float64[]
for c in order
  occursin(rx, c) || continue
  r = ref[c]; haskey(to, (c, r)) || continue
  q = to[(c, r)] / tn[(c, r)]; push!(rs, q)
  b = findfirst(k -> haskey(tn, (c, k)) && haskey(to, (c, k)), bare)
  qb = b === nothing ? NaN : to[(c, bare[b])] / tn[(c, bare[b])]
  (q > thr || q < 1 / thr) && @printf("| %s | %.1f | %.1f | **%.2f** | %s |\n", c, tn[(c, r)], to[(c, r)], q, isnan(qb) ? "" : @sprintf("%.2f", qb))
end
@printf("\n%d cases; old/new for the PB reference: geometric mean %.3f, median %.3f, min %.2f, max %.2f; %d cases above %.2f, %d below %.2f\n",
        length(rs), exp(mean(log.(rs))), median(rs), minimum(rs), maximum(rs), count(>(thr), rs), thr, count(<(1 / thr), rs), 1 / thr)
