# Accuracy evaluator of the wide least-squares study (#1103). It reads the prox outputs that harness.jl wrote for
# the primary roles (group outputs) and the inputs of generate.jl, checks the bytes of each file against its SHA-256,
# and evaluates, independently of the implementations and in BigFloat, from X, y, v and γ of each case:
#   eta             the stationarity residual
#                   ‖X'(Xβ - y) + ρ(β - v)‖∞ / (‖X'‖∞‖Xβ‖∞ + ‖X'y‖∞ + ρ(‖β‖∞ + ‖v‖∞)), ρ = 1/big(T(γ)), 256 bits,
#   pass            η ≤ 1e3·eps(T), decided in BigFloat before η is rounded to Float64 for the file,
#   forward_error   for n ≤ PB_FORWARD_MAX_N (default 500): ‖β - β*‖∞/‖β*‖∞ against the solution
#                   β* = v + X'a* of the dual system (XX' + ρI)a* = y - Xv in 128-bit BigFloat (Cholesky); the
#                   absolute error ‖β - β*‖∞ when β* = 0. A diagnostic; it is not tested.
# Conventions: an output that the harness recorded as a failure (outputs-failures.tsv) has η = Inf, pass = false and
# the status `error: <message>`; an output with a nonfinite entry and no recorded failure has η = Inf, pass = false and
# the status `nonfinite`; the others have the status `ok`. For n ≤ PB_FORWARD_MAX_N the forward error of the first two
# kinds is Inf; for larger n the forward error of every output is NaN (not evaluated).
#
#   PB_INPUTS=<inputs dir> PB_OUTPUTS=<the outputs-<julia> dir of harness.jl> PB_SOURCE_SHA=<sha> PB_HARNESS_SHA=<sha> \
#   [PB_FILTER=<regex on case ids>] [PB_ACC_TAG=all] [PB_FORWARD_MAX_N=500] \
#   julia --threads=1 --project=. docs/audit/wide-leastsquares-solves-1103/accuracy.jl
#
# Before scoring anything, a preflight checks the outputs index and the failures index together: their formats,
# columns and identities (source, harness, manifest digest and Julia version agree between them, the manifest digest is
# that of the inputs, and PB_SOURCE_SHA and PB_HARNESS_SHA equal the source and harness of the outputs; outside
# PB_CHECK_ONLY both are required), one row per (case, role) with a case of the manifest, a primary role and the four
# primary roles for each case, the element type, steps, number of inputs and shape of the manifest, the count nfail
# against the failure rows of that (case, role), failure keys in range and not repeated, and then the size and SHA-256
# of every output file. It lists every problem it finds and stops before any score is written. The result is
# accuracy-<tag>.tsv in the outputs directory: a header of `# key: value` lines (with the source, harness and input
# identities of the outputs, the SHA-256 of the outputs index and PB_FILTER), one row per (case, role, step, input), and
# the closing line `# complete: yes`.
# With PB_CHECK_ONLY=1 the script first checks its column sweeps against the dense BigFloat evaluation eta_exact and a dense
# dual solve on a small random problem, with and without the in-place operation, and the in-place operation against
# Base.fma under a scoped rounding mode, and prints the effective cost of the whole η evaluation per multiply-add of its
# two main sweeps; it then evaluates PB_OUTPUTS when that is set. The sweeps use the MPFR fused multiply-add (a ccall into
# libmpfr, with the rounding mode in effect) when it works and plain BigFloat arithmetic otherwise; PB_ACC_NOINPLACE=1
# selects the plain arithmetic.
ENV["PB_LOAD_PACKAGE"] = "0"
include(joinpath(@__DIR__, "common.jl"))

const INPLACE = Ref(false)

# acc := acc + x*y by one MPFR fused multiply-add into acc (the result has the precision of acc)
@inline function mpfr_fma!(acc::BigFloat, x::BigFloat, y::BigFloat)
    ccall((:mpfr_fma, Base.MPFR.libmpfr), Int32, (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat},
          Base.MPFR.MPFRRoundingMode), acc, x, y, acc, Base.MPFR.rounding_raw(BigFloat))
    acc
end
try
    z = mpfr_fma!(BigFloat(5), BigFloat(2), BigFloat(3))
    INPLACE[] = z == 11
catch
end
env_flag("PB_ACC_NOINPLACE") && (INPLACE[] = false)

# acc + x*y; with the in-place option the result is acc itself (one rounding instead of two)
@inline acc_mul(acc::BigFloat, x::BigFloat, y::BigFloat, tmp::BigFloat) = INPLACE[] ? mpfr_fma!(acc, x, y) : acc + x * y

# η of the columns of B (p × N) and whether η ≤ 1e3·eps(T) (decided in BigFloat): column m is an output for the step
# γs[γidx[m]] and the input V[:, vidx[m]]; a column with skip[m] or with a nonfinite entry has η = Inf and fails
function eta_columns(X::Matrix{T}, y::Vector{T}, V::Matrix{T}, B::Matrix{T}, γs::Vector{T}, γidx::Vector{Int},
                     vidx::Vector{Int}; skip::AbstractVector{Bool} = falses(size(B, 2)), prec::Int = 256) where {T<:AbstractFloat}
    n, p = size(X)
    N = size(B, 2)
    finite = [!skip[m] && all(isfinite, view(B, :, m)) for m in 1:N]
    η = fill(Inf, N)
    pass = fill(false, N)
    setprecision(BigFloat, prec) do
        ρs = [1 / big(γs[γidx[m]]) for m in 1:N]
        thr = big(1000) * big(eps(T))
        R = [BigFloat(0) for i in 1:n, m in 1:N]
        xcol = [BigFloat(0) for i in 1:n]
        tmp = BigFloat(0)
        for j in 1:p
            for i in 1:n
                xcol[i] = BigFloat(X[i, j])
            end
            for m in 1:N
                finite[m] || continue
                b = B[j, m]
                iszero(b) && continue
                bb = BigFloat(b)
                for i in 1:n
                    R[i, m] = acc_mul(R[i, m], xcol[i], bb, tmp)
                end
            end
        end
        xβinf = [finite[m] ? maximum(abs.(view(R, :, m))) : BigFloat(Inf) for m in 1:N]
        yb = BigFloat.(y)
        for m in 1:N, i in 1:n
            R[i, m] = R[i, m] - yb[i]
        end
        num = [BigFloat(0) for m in 1:N]
        Xt_inf, Xty_inf = BigFloat(0), BigFloat(0)
        for j in 1:p
            for i in 1:n
                xcol[i] = BigFloat(X[i, j])
            end
            cs, xy = BigFloat(0), BigFloat(0)
            for i in 1:n
                cs += abs(xcol[i])
                xy = acc_mul(xy, xcol[i], yb[i], tmp)
            end
            cs > Xt_inf && (Xt_inf = cs)
            abs(xy) > Xty_inf && (Xty_inf = abs(xy))
            for m in 1:N
                finite[m] || continue
                s = BigFloat(0)
                for i in 1:n
                    s = acc_mul(s, xcol[i], R[i, m], tmp)
                end
                s += ρs[m] * (BigFloat(B[j, m]) - BigFloat(V[j, vidx[m]]))
                a = abs(s)
                a > num[m] && (num[m] = a)
            end
        end
        for m in 1:N
            finite[m] || continue
            binf, vinf = BigFloat(maximum(abs, view(B, :, m))), BigFloat(maximum(abs, view(V, :, vidx[m])))
            den = Xt_inf * xβinf[m] + Xty_inf + ρs[m] * (binf + vinf)
            η[m] = Float64(iszero(den) ? (iszero(num[m]) ? BigFloat(0) : BigFloat(Inf)) : num[m] / den)
            pass[m] = iszero(den) ? iszero(num[m]) : num[m] <= thr * den
        end
    end
    η, pass
end

# the forward errors of the columns of B against the 128-bit solution of the dual system (columns as in eta_columns):
# relative to ‖β*‖∞, absolute when β* = 0; Inf for a skipped or nonfinite column
function forward_columns(X::Matrix{T}, y::Vector{T}, V::Matrix{T}, B::Matrix{T}, γs::Vector{T}, γidx::Vector{Int},
                         vidx::Vector{Int}; skip::AbstractVector{Bool} = falses(size(B, 2)), prec::Int = 128) where {T<:AbstractFloat}
    n, p = size(X)
    N = size(B, 2)
    err = fill(Inf, N)
    any(m -> !skip[m] && all(isfinite, view(B, :, m)), 1:N) || return err
    setprecision(BigFloat, prec) do
        yb = BigFloat.(y)
        xcol = [BigFloat(0) for i in 1:n]
        tmp = BigFloat(0)
        G = [BigFloat(0) for i in 1:n, j in 1:n]
        for j in 1:p
            for i in 1:n
                xcol[i] = BigFloat(X[i, j])
            end
            for l in 1:n, i in l:n
                G[i, l] = acc_mul(G[i, l], xcol[i], xcol[l], tmp)
            end
        end
        for l in 1:n, i in 1:l-1
            G[i, l] = G[l, i]
        end
        ks = sort!(unique(vidx))
        Xv = Dict(k => [BigFloat(0) for i in 1:n] for k in ks)
        for j in 1:p
            for i in 1:n
                xcol[i] = BigFloat(X[i, j])
            end
            for k in ks
                vj = BigFloat(V[j, k])
                iszero(vj) && continue
                for i in 1:n
                    Xv[k][i] = acc_mul(Xv[k][i], xcol[i], vj, tmp)
                end
            end
        end
        for gi in sort!(unique(γidx))
            ρ = 1 / big(γs[gi])
            M = copy(G)
            for i in 1:n
                M[i, i] += ρ
            end
            F = cholesky(Symmetric(M))
            for k in ks
                cols = [m for m in 1:N if γidx[m] == gi && vidx[m] == k]
                isempty(cols) && continue
                a = F \ (yb .- Xv[k])
                β = BigFloat.(V[:, k])
                for j in 1:p
                    for i in 1:n
                        xcol[i] = BigFloat(X[i, j])
                    end
                    s = BigFloat(0)
                    for i in 1:n
                        s = acc_mul(s, xcol[i], a[i], tmp)
                    end
                    β[j] += s
                end
                βinf = maximum(abs, β)
                for m in cols
                    (skip[m] || !all(isfinite, view(B, :, m))) && continue
                    d = maximum(abs(BigFloat(B[j, m]) - β[j]) for j in 1:p)
                    err[m] = Float64(iszero(βinf) ? d : d / βinf)
                end
            end
        end
    end
    err
end

const PRIMARY_ROLES = ("S", "C_B", "C_D", "C_W")
const OUTPUT_COLUMNS = ["case", "role", "file", "eltype", "shape", "sha256", "nfail", "copy_identical", "acc_gammas", "nv"]
const FAILURE_COLUMNS = ["case", "role", "gamma_index", "vindex", "message"]

# The preflight of the outputs directory (see the header). It returns the header and rows of the index, the failures
# keyed by (case, role, step index, input index) and the SHA-256 of the index; it throws with the list of every problem.
function preflight(m::Manifest, outdir::String)
    problems = String[]
    bad(msg) = push!(problems, msg)
    ipath, fpath = joinpath(outdir, "outputs-index.tsv"), joinpath(outdir, "outputs-failures.tsv")
    isfile(ipath) || error("missing $ipath")
    isfile(fpath) || error("missing $fpath")
    ibytes = read(ipath)
    header, cols, rows = parse_tsv(String(ibytes), ipath)
    fheader, fcols, frows = read_tsv(fpath)
    get(header, "format", "") == "pb1103-outputs-2" || bad("outputs-index.tsv has the format $(get(header, "format", "<absent>"))")
    get(fheader, "format", "") == "pb1103-outputs-failures-1" || bad("outputs-failures.tsv has the format $(get(fheader, "format", "<absent>"))")
    cols == OUTPUT_COLUMNS || bad("outputs-index.tsv has the columns $(join(cols, ','))")
    fcols == FAILURE_COLUMNS || bad("outputs-failures.tsv has the columns $(join(fcols, ','))")
    for k in ("source_sha", "harness_sha", "inputs_index_sha256", "julia")
        get(fheader, k, "<absent>") == get(header, k, "<absent>") || bad("outputs-failures.tsv and outputs-index.tsv differ in $k")
    end
    get(header, "inputs_index_sha256", "") == m.sha256 ||
        bad("the outputs were made from the manifest $(get(header, "inputs_index_sha256", "?")), not from $(m.sha256)")
    required = !env_flag("PB_CHECK_ONLY")
    for (var, key) in (("PB_SOURCE_SHA", "source_sha"), ("PB_HARNESS_SHA", "harness_sha"))
        want = env_sha(var; required)
        got = get(header, key, "<absent>")
        want == got || bad("$var $want differs from the $key $got of the outputs")
    end
    isempty(problems) || error("accuracy.jl refuses the outputs before scoring:\n  " * join(problems, "\n  "))
    byid = Dict(c.id => c for c in m.cases)
    seen = Dict{Tuple{String,String},Int}()
    for (i, r) in enumerate(rows)
        where_ = "outputs-index.tsv row $i ($(r["case"]), $(r["role"]))"
        c = get(byid, r["case"], nothing)
        c === nothing && (bad("$where_: the case is not in the manifest"); continue)
        r["role"] in PRIMARY_ROLES || (bad("$where_: not a primary role"); continue)
        key = (r["case"], r["role"])
        haskey(seen, key) && (bad("$where_: repeats row $(seen[key])"); continue)
        seen[key] = i
        K = size_of_v(c)
        r["eltype"] == string(c.T) || bad("$where_: element type $(r["eltype"]), the manifest has $(c.T)")
        r["acc_gammas"] == join(repr.(c.acc_gammas), ",") || bad("$where_: other steps than the manifest")
        tryparse(Int, r["nv"]) == K || bad("$where_: $(r["nv"]) inputs, the manifest has $K")
        r["shape"] == string(c.p, "x", K * length(c.acc_gammas)) || bad("$where_: shape $(r["shape"])")
        n = tryparse(Int, r["nfail"])
        n !== nothing && n >= 0 || bad("$where_: nfail $(r["nfail"])")
        r["copy_identical"] in ("true", "false") || bad("$where_: copy_identical $(r["copy_identical"])")
    end
    for id in unique(k[1] for k in keys(seen)), role in PRIMARY_ROLES
        haskey(seen, (id, role)) || bad("outputs-index.tsv: no row of $role for $id")
    end
    failures = Dict{Tuple{String,String,Int,Int},String}()
    counts = Dict{Tuple{String,String},Int}()
    for (i, f) in enumerate(frows)
        where_ = "outputs-failures.tsv row $i ($(f["case"]), $(f["role"]))"
        key2 = (f["case"], f["role"])
        haskey(seen, key2) || (bad("$where_: no such row in outputs-index.tsv"); continue)
        c = byid[f["case"]]
        gi, k = tryparse(Int, f["gamma_index"]), tryparse(Int, f["vindex"])
        (gi !== nothing && 1 <= gi <= length(c.acc_gammas) && k !== nothing && 1 <= k <= size_of_v(c)) ||
            (bad("$where_: step index $(f["gamma_index"]) or input index $(f["vindex"]) out of range"); continue)
        key = (f["case"], f["role"], gi, k)
        haskey(failures, key) && (bad("$where_: repeats the failure key $key"); continue)
        failures[key] = f["message"]
        counts[key2] = get(counts, key2, 0) + 1
    end
    for (key2, i) in seen
        n = tryparse(Int, rows[i]["nfail"])
        n === nothing || n == get(counts, key2, 0) ||
            bad("outputs-index.tsv row $i ($(key2[1]), $(key2[2])): nfail $n but $(get(counts, key2, 0)) failure rows")
    end
    if isempty(problems)
        for (i, r) in enumerate(rows)
            c = byid[r["case"]]
            path = joinpath(outdir, r["file"])
            isfile(path) || (bad("output file $(r["file"]) is missing"); continue)
            bytes = read(path)
            want = c.p * size_of_v(c) * length(c.acc_gammas) * sizeof(c.T)
            length(bytes) == want || bad("output file $(r["file"]) has $(length(bytes)) bytes, expected $want")
            bytes2hex(sha256(bytes)) == r["sha256"] || bad("output file $(r["file"]): SHA-256 mismatch")
        end
    end
    isempty(problems) || error("accuracy.jl refuses the outputs before scoring:\n  " * join(problems, "\n  "))
    header, rows, failures, bytes2hex(sha256(ibytes))
end
size_of_v(c::Case) = c.arrays["V"].shape[2]

function evaluate(m::Manifest, outdir::String)
    header, rows, failures, index_sha = preflight(m, outdir)
    rx = Regex(env_string("PB_FILTER", ""))
    fwd_max = env_int("PB_FORWARD_MAX_N", 500)
    tag = env_string("PB_ACC_TAG", "all")
    byid = Dict(c.id => c for c in m.cases)
    ids = unique([r["case"] for r in rows])
    open(joinpath(outdir, "accuracy-$tag.tsv"), "w") do io
        for (k, v) in [("format", "pb1103-accuracy-3"), ("julia", get(header, "julia", "unknown")),
                       ("outputs_order", get(header, "order", "unknown")), ("source_sha", get(header, "source_sha", "unset")),
                       ("harness_sha", get(header, "harness_sha", "unset")), ("inputs_index_sha256", m.sha256),
                       ("outputs_index_sha256", index_sha), ("filter", rx.pattern),
                       ("forward_max_n", fwd_max), ("inplace_bigfloat", INPLACE[]), ("eta_precision_bits", 256),
                       ("forward_precision_bits", 128), ("started_utc", Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ"))]
            println(io, "# ", k, ": ", v)
        end
        println(io, join(["case", "eltype", "kind", "role", "gamma_index", "gamma", "vindex", "eta", "forward_error", "pass", "status",
                          "copy_identical"], '\t'))
        for id in ids
            occursin(rx, id) || continue
            c = byid[id]
            t0 = time()
            X, y, V = read_input(m, c, "X"), read_input(m, c, "y"), read_input(m, c, "V")
            T = c.T
            γs = T.(c.acc_gammas)
            K = size(V, 2)
            crow = [r for r in rows if r["case"] == id]
            Bs, γidx, vidx, labels, skip = Matrix{T}[], Int[], Int[], Tuple{String,String,String}[], Bool[]
            for r in crow
                shape = parse.(Int, split(r["shape"], 'x'))
                B = read_verified(joinpath(outdir, r["file"]), r["sha256"], T, shape)
                push!(Bs, B)
                for gi in eachindex(γs), k in 1:K
                    msg = get(failures, (id, r["role"], gi, k), nothing)
                    col = view(B, :, (gi - 1) * K + k)
                    status = msg !== nothing ? "error: " * msg : all(isfinite, col) ? "ok" : "nonfinite"
                    push!(γidx, gi)
                    push!(vidx, k)
                    push!(skip, msg !== nothing)
                    push!(labels, (r["role"], status, r["copy_identical"]))
                end
            end
            Ball = reduce(hcat, Bs)
            η, pass = eta_columns(X, y, V, Ball, γs, γidx, vidx; skip = skip)
            fe = c.n <= fwd_max ? forward_columns(X, y, V, Ball, γs, γidx, vidx; skip = skip) : fill(NaN, length(η))
            for m in eachindex(η)
                println(io, join([id, string(T), c.kind, labels[m][1], γidx[m], repr(c.acc_gammas[γidx[m]]), vidx[m], η[m], fe[m],
                                  pass[m], labels[m][2], labels[m][3]], '\t'))
            end
            flush(io)
            println("evaluated ", id, " (", length(η), " outputs, ", round(time() - t0; digits = 1), " s)")
        end
        println(io, "# complete: yes")
    end
end

# the in-place multiply-add against Base.fma at 3 bits under the scoped rounding mode RoundUp
function check_rounding_scope()
    INPLACE[] || return
    setprecision(BigFloat, 3) do
        setrounding(BigFloat, RoundUp) do
            a, x, y = BigFloat(0.125), BigFloat(1), BigFloat(1)
            ref = fma(x, y, a)
            got = mpfr_fma!(BigFloat(a), x, y)
            got == ref || error("the in-place multiply-add ignores the scoped rounding mode: $got against $ref")
            ref == BigFloat(1.25) || error("the scoped rounding check expects 1.25 from fma, got $ref")
        end
    end
end

# column sweeps against the dense evaluations on a small random problem, with and without in-place operations
function self_check()
    println("== accuracy.jl self-check; in-place MPFR multiply-add available: ", INPLACE[])
    check_rounding_scope()
    saved = INPLACE[]
    for T in (Float64, Float32), inplace in unique([saved, false])
        INPLACE[] = inplace
        rng = MersenneTwister(7)
        n, p = 9, 24
        X = randn(rng, T, n, p)
        y = randn(rng, T, n)
        V = randn(rng, T, p, 2)
        γs = T[1e-3, 1.0, 1e3]
        B = zeros(T, p, 8)
        γidx, vidx = Int[], Int[]
        for gi in 1:3, k in 1:2
            ρ = inv(γs[gi])
            col = (gi - 1) * 2 + k
            B[:, col] = (X' * X + ρ * I) \ (X' * y + ρ * V[:, k])
            B[1, col] += T(1e-3) * T(col)            # a perturbation, so that η is well above the rounding level
            push!(γidx, gi)
            push!(vidx, k)
        end
        B[:, 7] .= T(NaN)                             # a nonfinite output
        push!(γidx, 3); push!(vidx, 1)
        ρ1 = inv(γs[2])
        B[:, 8] = (X' * X + ρ1 * I) \ (X' * y + ρ1 * V[:, 1])   # an accurate output at γ = 1
        push!(γidx, 2); push!(vidx, 1)
        η, pass = eta_columns(X, y, V, B, γs, γidx, vidx)
        for m in (1:6..., 8)
            ref = eta_exact(X, y, V[:, vidx[m]], γs[γidx[m]], B[:, m])
            abs(η[m] - ref) <= 1e-12 * ref || error("eta_columns differs from eta_exact: $(η[m]) against $ref (T $T, in place $inplace)")
            pass[m] == (ref <= 1e3 * eps(T)) || error("the pass flag of column $m disagrees with the threshold")
        end
        all(!, pass[1:7]) && pass[8] || error("the pass flags are wrong: $pass")
        isinf(η[7]) || error("a nonfinite output must have η = Inf")
        η2, pass2 = eta_columns(X, y, V, B, γs, γidx, vidx; skip = [m == 8 for m in 1:8])
        isinf(η2[8]) && !pass2[8] && η2[1] == η[1] || error("a skipped column must have η = Inf and fail")
        fe = forward_columns(X, y, V, B, γs, γidx, vidx)
        for m in (1:5..., 8)
            setprecision(BigFloat, 128) do
                Xb, yb, Vb = BigFloat.(X), BigFloat.(y), BigFloat.(V)
                ρ = 1 / big(γs[γidx[m]])
                v = Vb[:, vidx[m]]
                a = (Xb * Xb' + ρ * I) \ (yb - Xb * v)
                β = v + Xb' * a
                ref = Float64(maximum(abs.(BigFloat.(B[:, m]) - β)) / maximum(abs.(β)))
                abs(fe[m] - ref) <= 1e-9 * ref || error("forward_columns differs from the dense solve: $(fe[m]) against $ref")
            end
        end
        println(T, ", in place ", inplace, ": η, the pass flags and the forward errors agree with the dense BigFloat evaluation")
    end
    INPLACE[] = saved
    rng = MersenneTwister(8)
    X, y, V = randn(rng, 100, 400), randn(rng, 100), randn(rng, 400, 2)
    B = randn(rng, 400, 8)
    eta_columns(X, y, V, B[:, 1:1], [1.0], [1], [1])
    t = @elapsed eta_columns(X, y, V, B, [1.0, 2.0, 3.0, 4.0], repeat([1, 2, 3, 4], 2), repeat([1, 2], 4))
    println("eta_columns: effective ", round(t / (2 * 100 * 400 * 8) * 1e9; digits = 1),
            " ns per multiply-add of its two main sweeps, for the whole evaluation (in place ", INPLACE[], ")")
    println("accuracy.jl self-check passed")
end

function main()
    env_flag("PB_CHECK_ONLY") && self_check()
    outdir = env_string("PB_OUTPUTS", "")
    if isempty(outdir)
        env_flag("PB_CHECK_ONLY") || error("PB_OUTPUTS names the outputs directory of harness.jl")
        println("no PB_OUTPUTS: only the self-check ran")
        return
    end
    m = load_manifest(env_string("PB_INPUTS", ""))
    evaluate(m, outdir)
    println("accuracy.jl done")
end

main()
