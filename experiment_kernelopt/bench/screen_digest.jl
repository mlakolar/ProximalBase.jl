# Condensed tables of a screen campaign: rows are (operation, implementation), columns are strata (one result
# directory each, labelled NAME=DIR); a cell is the geometric mean over the selected cases of the median per-round
# ratio of the implementation's time to ProximalBase's. The reference row shows ProximalBase's time range in ns.
#   julia bench/screen_digest.jl <family: l1|l2|group|ls> NAME=DIR...
using Statistics, Printf
fam = ARGS[1]
strata = [(String(split(a, '=')[1]), String(split(a, '=')[2])) for a in ARGS[2:end]]
function load(d)
  t = Dict{Tuple{String, String}, Dict{Int, Float64}}(); ref = Dict{String, String}(); order = String[]; bad = Set{Tuple{String, String}}()
  for (k, l) in enumerate(eachline(joinpath(d, "meta.csv")))
    k == 1 && continue; f = split(l, ',')
    haskey(ref, f[1]) || (push!(order, String(f[1])); ref[String(f[1])] = String(f[3])); f[11] == "ok" || push!(bad, (String(f[1]), String(f[4])))
  end
  for (k, l) in enumerate(eachline(joinpath(d, "raw.csv"))); k == 1 && continue; f = split(l, ','); get!(t, (String(f[1]), String(f[2])), Dict{Int, Float64}())[parse(Int, f[3])] = parse(Float64, f[5]); end
  (t, ref, order, bad)
end
data = [load(d) for (_, d) in strata]
function ratio(D, c, im)
  t, ref, _, bad = D
  (haskey(t, (c, im)) && !((c, im) in bad)) || return NaN
  median([t[(c, im)][r] / t[(c, ref[c])][r] for r in keys(t[(c, im)]) if haskey(t[(c, ref[c])], r)])
end
refns(D, c) = median(collect(values(D[1][(c, D[2][c])])))
geo(v) = isempty(v) ? NaN : exp(mean(log.(v)))
# rows: (label, regex selecting the cases, implementations)
SPEC = Dict(
  "l1" => [("prox, scalar step, n = 128…16384", r"^l1/prox/T/n(128|1024|16384)/normal", ["k_st_maxmin", "k_st_selmaxmin", "k_st_clampz", "k_st_clamp", "po"]),
           ("prox, weights", r"^l1/prox_w/T/n(128|1024|16384)/normal", ["k_st_maxmin", "k_st_clampz", "po"]),
           ("prox, step per entry", r"^l1/prox_g/T/n(128|1024|16384)/normal", ["k_st_maxmin", "k_st_clampz"]),
           ("prox, step per entry and weights", r"^l1/prox_gw/T/n(128|1024|16384)/normal", ["k_st_maxmin", "k_st_clampz"]),
           ("value", r"^l1/value/T/n(128|1024|16384)/normal", ["k_simd", "k_base", "k_blas", "po"]),
           ("value, weights", r"^l1/value_w/T/n(128|1024|16384)/normal", ["k_simd", "po"]),
           ("value and prox", r"^l1/valprox/T/n(128|1024|16384)/normal", ["k_two_st_maxmin", "k_fused_st_maxmin", "k_fused_st_clamp", "po"]),
           ("value and prox, weights", r"^l1/valprox_w/T/n(128|1024|16384)/normal", ["k_two_st_maxmin", "k_fused_st_maxmin", "po"]),
           ("prox, n = 2^20 (out of cache)", r"^l1/prox/T/n1048576/normal", ["k_st_maxmin", "k_st_clampz", "po"])],
  "l2" => [("value, n = 2…10", r"^l2/value/T/n(2|5|10)/shrink", ["k_norm_simd", "k_norm_la", "k_norm_blas", "k_norm_blue", "po"]),
           ("value, n = 32…1024", r"^l2/value/T/n(32|128|1024)/shrink", ["k_norm_simd", "k_norm_base", "k_norm_la", "k_norm_blas", "k_norm_dot", "k_norm_blue", "po"]),
           ("value, n = 65536", r"^l2/value/T/n65536/shrink", ["k_norm_simd", "k_norm_la", "k_norm_blas", "k_norm_blue", "po"]),
           ("value of the zero vector, n = 32…1024", r"^l2/value/T/n(32|128|1024)/zero", ["k_norm_simd", "k_norm_la"]),
           ("prox, n = 2…10", r"^l2/prox/T/n(2|5|10)/shrink", ["k_direct", "k_norm", "po"]),
           ("prox, n = 32…1024", r"^l2/prox/T/n(32|128|1024)/shrink", ["k_direct", "k_norm", "k_nrm2", "po"]),
           ("prox of the zero vector, n = 32…1024", r"^l2/prox/T/n(32|128|1024)/zero", ["k_direct", "po"]),
           ("value and prox, n = 32…1024", r"^l2/valprox/T/n(32|128|1024)/shrink", ["k_direct", "po"])],
  "group" => vcat([("$op, groups of $m, $lay, $pat", Regex("^group/$op/T/m$m/$lay/$pat\$"), impls)
                   for (op, impls) in (("prox", ["k_direct", "po"]), ("valprox", ["k_direct", "po"]), ("value", ["k_direct", "po"]), ("block_sweep", ["k_direct"]))
                   for m in (2, 5, 32, 1024) for (lay, pat) in (("range", "allshrink"), ("range", "sparse"), ("perm", "allshrink")) if !(op == "block_sweep" && lay == "perm")]),
  "ls" => [("value at a residual, n = 100…100000", r"^ls/value_res/T/n(100|1000|10000|100000)$", ["k_halfsumsq_simd", "k_halfsumsq_base", "k_halfsumsq_dot", "k_halfsumsq_nrm2"]),
           ("value at a predictor, n = 100…100000", r"^ls/value_pred/T/n(100|1000|10000|100000)$", ["k_simd"]),
           ("value at β (one product with X)", r"^ls/value/T/", ["k_direct", "po"]),
           ("value and gradient at β", r"^ls/valgrad/T/", ["k_three", "k_fused", "k_gemv", "po"]),
           ("value and gradient at a predictor", r"^ls/valgrad_pred/T/", ["k_three", "k_fused"]),
           ("prox, fixed step, tall X", r"^ls/prox_fixed/T/(200x50|1000x200|2000x500)$", ["k_chol", "po"]),
           ("prox, fixed step, wide X (200×1000)", r"^ls/prox_fixed/T/200x1000$", ["k_chol", "po"]),
           ("prox, the step changes at every call", r"^ls/prox_vary/T/", ["k_chol", "po"]),
           ("construction and first prox", r"^ls/prox_first/T/", ["k_chol", "po"])])
println("| operation | implementation | ", join(first.(strata), " | "), " |"); println("|:--|:--|", repeat("--:|", length(strata)))
for T in ("Float64", "Float32"), (label, rx0, impls) in SPEC[fam]
  rx = Regex(replace(rx0.pattern, "/T/" => "/$T/"))
  cells = String[]
  for D in data
    cs = [c for c in D[3] if occursin(rx, c)]
    isempty(cs) ? push!(cells, "") : (v = [refns(D, c) for c in cs]; push!(cells, length(cs) == 1 ? @sprintf("%.3g ns", v[1]) : @sprintf("%.3g–%.3g ns", minimum(v), maximum(v))))
  end
  println("| **", T, " ", label, "** | ProximalBase | ", join(cells, " | "), " |")
  for im in impls
    cells = String[]
    for D in data
      v = filter(!isnan, [ratio(D, c, im) for c in D[3] if occursin(rx, c)])
      push!(cells, isempty(v) ? "" : @sprintf("%.2f", geo(v)))
    end
    all(isempty, cells) || println("| | ", im, " | ", join(cells, " | "), " |")
  end
end
