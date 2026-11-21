# Accuracy evaluator of the wide least-squares study (#1103). It reads the prox outputs that harness.jl wrote for
# the primary roles (group outputs) and the inputs of generate.jl, checks the bytes of each file against its SHA-256,
# and evaluates, independently of the implementations and in BigFloat, from X, y, v and γ of each case:
#   eta             the stationarity residual
#                   ‖X'(Xβ - y) + ρ(β - v)‖∞ / (‖X'‖∞‖Xβ‖∞ + ‖X'y‖∞ + ρ(‖β‖∞ + ‖v‖∞)), ρ = 1/big(T(γ)), 256 bits,
#   pass            η ≤ 1e3·eps(T), decided in BigFloat before η is rounded to Float64 for the file,
#   forward_error   for n ≤ PB_FORWARD_MAX_N (default 500): ‖β - β*‖∞/‖β*‖∞ against the solution
#                   β* = v + X'a* of the dual system (XX' + ρI)a* = y - Xv in 128-bit BigFloat (Cholesky); the
#                   absolute error ‖β - β*‖∞ when β* = 0. A diagnostic; it is not tested.
# An output with a nonfinite entry, or that the harness recorded as a failure (outputs-failures.tsv), has η = Inf,
# forward_error = Inf and pass = false, with the status of the failure.
#
#   PB_INPUTS=<inputs dir> PB_OUTPUTS=<the outputs-<julia> dir of harness.jl> PB_HARNESS_SHA=<sha> \
#   [PB_FILTER=<regex on case ids>] [PB_ACC_TAG=all] [PB_FORWARD_MAX_N=500] \
#   julia --threads=1 --project=. docs/audit/wide-leastsquares-solves-1103/accuracy.jl
#
# The inputs must be the manifest that the outputs were made from (the SHA-256 of inputs-index.tsv recorded in the outputs
# index), and the element type, steps and number of inputs of each output file must agree with it; a mismatch stops the
# run before any scoring. PB_HARNESS_SHA must equal the harness identity of the outputs unless both are unset (PB_CHECK_ONLY).
# The result is accuracy-<tag>.tsv in the outputs directory: a header of `# key: value` lines (with the source, harness and
# input identities of the outputs), one row per (case, role, step, input), and the closing line `# complete: yes`.
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

function evaluate(m::Manifest, outdir::String)
    header, _, rows = read_tsv(joinpath(outdir, "outputs-index.tsv"))
    fheader, _, frows = read_tsv(joinpath(outdir, "outputs-failures.tsv"))
    get(header, "inputs_index_sha256", "") == m.sha256 ||
        error("the outputs were made from the manifest $(get(header, "inputs_index_sha256", "?")), not from $(m.sha256)")
    for k in ("source_sha", "harness_sha", "inputs_index_sha256", "julia")
        get(fheader, k, "") == get(header, k, "?") || error("outputs-failures.tsv and outputs-index.tsv differ in $k")
    end
    out_harness = get(header, "harness_sha", "unset")
    env_harness = env_sha("PB_HARNESS_SHA"; required = !env_flag("PB_CHECK_ONLY"))
    env_harness == out_harness || error("PB_HARNESS_SHA $env_harness differs from the harness identity $out_harness of the outputs")
    rx = Regex(env_string("PB_FILTER", ""))
    fwd_max = env_int("PB_FORWARD_MAX_N", 500)
    tag = env_string("PB_ACC_TAG", "all")
    byid = Dict(c.id => c for c in m.cases)
    ids = unique([r["case"] for r in rows])
    for id in ids
        haskey(byid, id) || error("the outputs hold the case $id, which the manifest does not")
    end
    failures = Dict{Tuple{String,String,Int,Int},String}((f["case"], f["role"], parse(Int, f["gamma_index"]), parse(Int, f["vindex"])) =>
                                                          f["message"] for f in frows)
    open(joinpath(outdir, "accuracy-$tag.tsv"), "w") do io
        for (k, v) in [("format", "pb1103-accuracy-2"), ("julia", get(header, "julia", "unknown")),
                       ("outputs_order", get(header, "order", "unknown")), ("source_sha", get(header, "source_sha", "unset")),
                       ("harness_sha", out_harness), ("inputs_index_sha256", m.sha256), ("filter", rx.pattern),
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
                r["eltype"] == string(T) || error("output $(r["file"]) is $(r["eltype"]); the case $id is $T")
                r["acc_gammas"] == join(repr.(c.acc_gammas), ",") || error("output $(r["file"]) has other steps than the manifest")
                parse(Int, r["nv"]) == K || error("output $(r["file"]) has another number of inputs than the manifest")
                shape = parse.(Int, split(r["shape"], 'x'))
                shape == [c.p, K * length(γs)] || error("output $(r["file"]) has the wrong shape")
                B = read_verified(joinpath(outdir, r["file"]), r["sha256"], T, shape)
                push!(Bs, B)
                for gi in eachindex(γs), k in 1:K
                    msg = get(failures, (id, r["role"], gi, k), nothing)
                    push!(γidx, gi)
                    push!(vidx, k)
                    push!(skip, msg !== nothing)
                    push!(labels, (r["role"], msg === nothing ? "ok" : "error: " * msg, r["copy_identical"]))
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
