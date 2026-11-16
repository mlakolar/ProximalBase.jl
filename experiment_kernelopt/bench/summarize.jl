# Summary of one or more result directories of run.jl (pooled when they hold the same cases):
#   julia bench/summarize.jl [--filter=regex] [--csv=out.csv] DIR...
# For every case and implementation: the median time, the median of the paired per-round ratios to the case's
# reference implementation, and the 10% and 90% quantiles of those ratios.
using Statistics, Printf
dirs = filter(a -> !startswith(a, "--"), ARGS)
opt(name) = (i = findfirst(a -> startswith(a, "--$name="), ARGS); i === nothing ? nothing : split(ARGS[i], '=', limit = 2)[2])
rx = opt("filter") === nothing ? r"" : Regex(opt("filter"))
times = Dict{Tuple{String, String}, Dict{Tuple{Int, Int}, Float64}}()      # (case, impl) => (dir, round) => ns
meta = Dict{Tuple{String, String}, Vector{String}}()
order = Tuple{String, String}[]
for (di, d) in enumerate(dirs)
  for (k, line) in enumerate(eachline(joinpath(d, "meta.csv")))
    k == 1 && continue
    f = split(line, ',')
    key = (String(f[1]), String(f[4]))
    haskey(meta, key) || (push!(order, key); meta[key] = String.(f))
  end
  for (k, line) in enumerate(eachline(joinpath(d, "raw.csv")))
    k == 1 && continue
    f = split(line, ',')
    get!(times, (String(f[1]), String(f[2])), Dict{Tuple{Int, Int}, Float64}())[(di, parse(Int, f[3]))] = parse(Float64, f[5])
  end
end
csv = opt("csv") === nothing ? nothing : open(opt("csv"), "w")
csv === nothing || println(csv, "case,impl,level,median_ns,ratio,ratio_q10,ratio_q90,alloc_bytes,rounds,ref")
lastcase = ""
for key in order
  case, impl = key
  occursin(rx, case) || continue
  haskey(times, key) || continue
  m = meta[key]
  ref = m[3]
  t, tr = times[key], times[(case, ref)]
  ks = sort!(collect(intersect(keys(t), keys(tr))))
  ratios = [t[k] / tr[k] for k in ks]
  med = median(collect(values(t)))
  q10, q50, q90 = quantile(ratios, 0.1), median(ratios), quantile(ratios, 0.9)
  if case != lastcase
    global lastcase = case
    @printf("\n%s   (ref %s, %d rounds)\n", case, ref, length(ks))
  end
  @printf("  %-22s L%s %12.2f ns  %6.3f  [%5.3f, %5.3f]%s\n", impl, m[5], med, q50, q10, q90, m[7] == "0" ? "" : "  alloc " * m[7])
  csv === nothing || @printf(csv, "%s,%s,%s,%.4f,%.4f,%.4f,%.4f,%s,%d,%s\n", case, impl, m[5], med, q50, q10, q90, m[7], length(ks), ref)
end
csv === nothing || close(csv)
