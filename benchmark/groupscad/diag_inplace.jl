# Diagnosis of the in-place cost of GroupSCAD for large groups, and of the small-group value, in one process.
#
#   PB_SOURCE=<candidate tree> PB_LEGACY=<archived tree> PB_REFERENCE=<tree with GroupNormL2> [PB_VARIANT=<value variant tree>] \
#   PB_OUTPUT=<dir> julia --startup-file=no --threads=1 --check-bounds=no --project=benchmark benchmark/groupscad/diag_inplace.jl
#
# The trees need a src/ folder (git archive <sha> src). PB_LEGACY is the archived implementation (eff21b6d7d33c622313122f46c6ee71693c101ee), PB_REFERENCE any tree
# with the shared GroupNormL2 (the base of the pull request), PB_VARIANT (optional) a copy of the candidate whose ordinary value
# path calls _scad_stored_value instead of _scad_value. Modules: C (candidate), CC (a second compilation of the candidate,
# the A/A control), C3 (a third compilation), L (legacy), R (reference), V (variant), W (in-place variant). PB_REVERSE=1 reverses
# the load order.
#
# Cases: the input generation of run.jl (4096 entries, level 3/4, shape 3.7, unit weights, 8 cycled inputs per case, steps
# below, at and above a - 1, regimes zero, soft, middle, kept, contiguous ranges and permuted index vectors, Float32 and
# Float64). Section 1 (group sizes 512 and 4096) is about the proximal map in place; section 2 (group sizes 1, 4 and 32)
# about the value and the fused value. PB_FILTER selects cases by a regular expression on their ids.
#
# For each case the script prints
#   * the branch that the candidate takes for each group of the 8 inputs, counted over the inputs and the groups: copy (the
#     scalar kernel returns the norm), zero, scale, or cold (the ordinary guard fails);
#   * the absolute median of each probe in ns and ns per entry (the median over the repetitions of the median of a BenchmarkTools
#     run: evals = 1, the setup restores x from the input pool, so that the data are rewritten before each sample);
#   * the paired ratios of probes, as the median over the repetitions with the minimum and maximum.
# Probes are timed in a rotating order that is reversed in alternate repetitions (PB_REPS repetitions, default 8, at least 8;
# PB_SAMPLES samples per run, default 200; PB_SECONDS cap, default 0.05). Section 1 probes:
#   X_inplace, XC_inplace (A/A), L_inplace, R_inplace   prox!(x, f, x, γ) of the candidate, its second compilation, the legacy
#                                                       tree and GroupNormL2
#   X3_inplace                                          a third compilation of the candidate (A/A: the spread among compilations)
#   W_inplace                                           (PB_VARIANT_INPLACE) the candidate with an in-place scaling loop over one array
#                                                       when y === x (its output is compared bit for bit with X on every input)
#   X_outofplace, L_outofplace                          prox!(y, f, x, γ)
#   c_sumabs2            the sum of squares of every group (the shared _indexed_sumabs2)
#   c_l2scale_inplace    _l2_scale!(x, x, G, q) for every group (the shared @simd loop; q = t/r of the first input, per group)
#   c_plain_inplace      @inbounds for i in G; x[i] = q * x[i]; end
#   c_plain_simd_inplace the same loop with @simd
#   c_l2scale_hidden_inplace, c_plain_hidden_inplace, c_plain_simd_hidden_inplace
#                        the same in-place loops behind a @noinline call, so that the compiler does not know that y === x (as in
#                        prox!(x, f, x, γ)), where a vectorized loop needs a runtime alias test
#   X_loop_inplace       the candidate's loop over the groups (_gscad_prox_loop!) called directly, without the argument checks
#   c_l2scale_outofplace _l2_scale!(y, x, G, q)
#   c_scalar             the scalar kernel _scad_magnitude(r, λ, a, γ) over 4096 radii near the target norm (in ns per call)
# Section 2 probes: X_value, XC_value, V_value, L_value, R_value (f(x)) and X_vap, V_vap, L_vap, R_vap (value_and_prox!(y, f, x, γ)).
# Files in PB_OUTPUT: provenance.txt, branches.csv, times.csv (case, probe, repetition, median_ns) and ratios.txt.
using BenchmarkTools, Dates, Random, Statistics, Printf

Threads.nthreads() == 1 || error("one Julia thread required")
const SOURCE = abspath(ENV["PB_SOURCE"])
const LEGACY = abspath(ENV["PB_LEGACY"])
const REFERENCE = abspath(ENV["PB_REFERENCE"])
const VARIANT = haskey(ENV, "PB_VARIANT") ? abspath(ENV["PB_VARIANT"]) : nothing
const VARIANT_INPLACE = haskey(ENV, "PB_VARIANT_INPLACE") ? abspath(ENV["PB_VARIANT_INPLACE"]) : nothing
const REVERSE = get(ENV, "PB_REVERSE", "0") == "1"
const OUTDIR = get(ENV, "PB_OUTPUT", mktempdir())

module C end
module CC end
module L end
module R end
module V end
module W end
module C3 end
let mods = [(C, SOURCE), (CC, SOURCE), (C3, SOURCE), (L, LEGACY), (R, REFERENCE)]
    VARIANT === nothing || push!(mods, (V, VARIANT))
    VARIANT_INPLACE === nothing || push!(mods, (W, VARIANT_INPLACE))
    for (m, root) in (REVERSE ? reverse(mods) : mods)
        Base.include(m, joinpath(root, "src", "ProximalBase.jl"))
    end
end

const TOTAL = 4096
const STEPS = ((:below, 1 // 2), (:at, 1 // 1), (:above, 3 // 2))        # multiples of a - 1
const REGIMES = (:zero, :soft, :middle, :kept)
const SHAPE = 37 // 10
const LEVEL = 3 // 4
const NINPUTS = 8
const REPS = max(8, parse(Int, get(ENV, "PB_REPS", "8")))
const SAMPLES = parse(Int, get(ENV, "PB_SAMPLES", "200"))
const SECONDS = parse(Float64, get(ENV, "PB_SECONDS", "0.05"))
const RETRIES = 3
const FILTER = Regex(get(ENV, "PB_FILTER", ""))

# the generation of benchmark/groupscad/run.jl (the same functions)
function group_norm(regime, γ, a, λ)
    convex = γ < a - 1
    keep_cut = convex ? a * λ : max(a * λ, (a + 1 + γ) * λ / 2, γ * λ, λ * sqrt(γ * (a + 1)))
    regime === :zero ? γ * λ / 2 :
    regime === :soft ? (convex ? (γ + 1 / 2) * λ : (γ + (a + 1 + γ) / 2) * λ / 2) :
    regime === :middle ? (convex ? ((1 + γ) + a) * λ / 2 : (1 + a) * λ / 2) : 2keep_cut
end

function makedata(::Type{T}, m, layout, step, regime, ninputs) where {T}
    rng = MersenneTwister(1070 + 7m + 3(layout === :perm) + 11ninputs)
    K = TOTAL ÷ m
    order = layout === :perm ? randperm(rng, TOTAL) : collect(1:TOTAL)
    groups = layout === :perm ? [order[(k-1)*m+1:k*m] for k in 1:K] : [(k-1)*m+1:k*m for k in 1:K]
    a, λ0 = T(SHAPE), T(LEVEL)
    γ = (a - 1) * T(step)
    w = ones(T, K)
    inputs = Vector{Vector{T}}(undef, ninputs)
    for j in 1:ninputs
        x = Vector{T}(undef, TOTAL)
        for (k, G) in enumerate(groups)
            r = group_norm(regime, Float64(γ), Float64(a), Float64(λ0))
            d = randn(rng, m)
            d ./= sqrt(sum(abs2, d))
            x[G] .= T.(r .* d)
        end
        inputs[j] = x
    end
    (; groups, inputs, a, λ0, γ, w, r = group_norm(regime, Float64(γ), Float64(a), Float64(λ0)))
end

# the branch of the candidate for one group: :copy, :zero, :scale, or :cold when the ordinary guard fails
function branch(PB, f, x, G, k, γ::T) where {T}
    s = PB._indexed_sumabs2(x, G, T)
    λ = f.λ0 * f.w[k]
    ok = (zero(T) < γ < T(Inf)) && λ >= PB._gscad_level_min(T) && isfinite(f.a * λ) && isfinite(s) && PB._squares_normal(s, length(G), T)
    ok || return :cold
    r = sqrt(s)
    t = PB._scad_magnitude(r, λ, f.a, γ)
    t == r ? :copy : iszero(t) ? :zero : :scale
end

function timed_run(b)
    for attempt in 1:RETRIES
        try
            t = run(b)
            length(t.times) >= 3 && return median(t).time      # fewer samples: a stalled first sample on Julia 1.13
        catch e
            e isa InexactError || rethrow()      # the sizehint! error of a first sample longer than the seconds cap (see run.jl)
        end
    end
    NaN
end

function prepare(op, setup)
    for _ in 1:20
        setup()
        op()
    end
    @benchmarkable $op() setup = ($setup()) evals = 1 samples = SAMPLES seconds = SECONDS gctrial = false gcsample = false
end

# Calls that the compiler cannot see through, so that it does not know that y and x are the same array (as in prox!(x, f, x, γ),
# where the loops are compiled with unrelated y and x and the vectorized loop needs a runtime alias test).
@noinline hidden_l2scale!(y, x, G, q) = C.ProximalBase._l2_scale!(y, x, G, q)
@noinline function hidden_plain!(y, x, G, q)
    @inbounds for i in G
        y[i] = q * x[i]
    end
end
@noinline function hidden_plain_simd!(y, x, G, q)
    @inbounds @simd for i in G
        y[i] = q * x[i]
    end
end

# Probes of one case on the typed data. Each probe is a named (op, setup) pair; the setups cycle the input pool outside the timing.
function probes(d, fC, fCC, fL, fR, fV, fC3, fW, ::Val{section}) where {section}
    T = eltype(d.inputs[1])
    inputs, γ, groups = d.inputs, d.γ, d.groups
    x = copy(inputs[1]); y = similar(x)
    cur = Ref(0)
    restore!() = (cur[] = mod1(cur[] + 1, NINPUTS); copyto!(x, inputs[cur[]]); nothing)
    val = Ref(zero(T))
    list = Pair{String, Any}[]
    add!(name, op, setup = restore!) = push!(list, name => (op, setup))
    if section == 1
        add!("X_inplace", () -> (C.ProximalBase.prox!(x, fC, x, γ); nothing))
        add!("XC_inplace", () -> (CC.ProximalBase.prox!(x, fCC, x, γ); nothing))
        add!("X3_inplace", () -> (C3.ProximalBase.prox!(x, fC3, x, γ); nothing))
        fW === nothing || add!("W_inplace", () -> (W.ProximalBase.prox!(x, fW, x, γ); nothing))
        add!("L_inplace", () -> (L.ProximalBase.prox!(x, fL, x, γ); nothing))
        add!("R_inplace", () -> (R.ProximalBase.prox!(x, fR, x, γ); nothing))
        add!("X_outofplace", () -> (C.ProximalBase.prox!(y, fC, x, γ); nothing))
        add!("L_outofplace", () -> (L.ProximalBase.prox!(y, fL, x, γ); nothing))
        # per-group q = t/r of the first input, as the candidate computes it
        q = T[]
        rs = T[]
        for (k, G) in enumerate(groups)
            s = C.ProximalBase._indexed_sumabs2(inputs[1], G, T)
            r = sqrt(s); λ = fC.λ0 * fC.w[k]
            t = C.ProximalBase._scad_magnitude(r, λ, fC.a, γ)
            push!(q, iszero(r) ? zero(T) : t / r)
            push!(rs, r)
        end
        acc = Ref(zero(T))
        add!("c_sumabs2", () -> begin
            s = zero(T)
            for G in groups
                s += C.ProximalBase._indexed_sumabs2(x, G, T)
            end
            acc[] = s
            nothing
        end)
        add!("c_l2scale_inplace", () -> begin
            for k in eachindex(groups)
                C.ProximalBase._l2_scale!(x, x, groups[k], q[k])
            end
            nothing
        end)
        add!("c_plain_inplace", () -> begin
            for k in eachindex(groups)
                qk = q[k]
                @inbounds for i in groups[k]
                    x[i] = qk * x[i]
                end
            end
            nothing
        end)
        add!("c_plain_simd_inplace", () -> begin
            for k in eachindex(groups)
                qk = q[k]
                @inbounds @simd for i in groups[k]
                    x[i] = qk * x[i]
                end
            end
            nothing
        end)
        add!("c_l2scale_hidden_inplace", () -> begin
            for k in eachindex(groups)
                hidden_l2scale!(x, x, groups[k], q[k])
            end
            nothing
        end)
        add!("c_plain_hidden_inplace", () -> begin
            for k in eachindex(groups)
                hidden_plain!(x, x, groups[k], q[k])
            end
            nothing
        end)
        add!("c_plain_simd_hidden_inplace", () -> begin
            for k in eachindex(groups)
                hidden_plain_simd!(x, x, groups[k], q[k])
            end
            nothing
        end)
        add!("X_loop_inplace", () -> (C.ProximalBase._gscad_prox_loop!(x, fC, x, γ, Val(true), Val(false)); nothing))
        add!("c_l2scale_outofplace", () -> begin
            for k in eachindex(groups)
                C.ProximalBase._l2_scale!(y, x, groups[k], q[k])
            end
            nothing
        end)
        # the scalar kernel at 4096 radii that differ in the last digits of the target norm of the regime
        rng = MersenneTwister(5)
        radii = T[T(d.r) * (one(T) + T(1e-3) * (2rand(rng, T) - one(T))) for _ in 1:TOTAL]
        λ = fC.λ0 * fC.w[1]
        add!("c_scalar", () -> begin
            s = zero(T)
            for i in eachindex(radii)
                @inbounds s += C.ProximalBase._scad_magnitude(radii[i], λ, fC.a, γ)
            end
            acc[] = s
            nothing
        end, () -> nothing)
    else
        add!("X_value", () -> (val[] = fC(x); nothing))
        add!("XC_value", () -> (val[] = fCC(x); nothing))
        fV === nothing || add!("V_value", () -> (val[] = fV(x); nothing))
        add!("L_value", () -> (val[] = fL(x); nothing))
        add!("R_value", () -> (val[] = fR(x); nothing))
        add!("X_vap", () -> (val[] = C.ProximalBase.value_and_prox!(y, fC, x, γ); nothing))
        fV === nothing || add!("V_vap", () -> (val[] = V.ProximalBase.value_and_prox!(y, fV, x, γ); nothing))
        add!("L_vap", () -> (val[] = L.ProximalBase.value_and_prox!(y, fL, x, γ); nothing))
        add!("R_vap", () -> (val[] = R.ProximalBase.value_and_prox!(y, fR, x, γ); nothing))
    end
    list
end

const PAIRS1 = (("X3_inplace", "X_inplace"), ("W_inplace", "X_inplace"), ("W_inplace", "L_inplace"), ("W_inplace", "R_inplace"), ("X_inplace", "X_loop_inplace"), ("c_l2scale_hidden_inplace", "c_l2scale_inplace"), ("c_plain_hidden_inplace", "c_plain_inplace"),
                ("c_plain_simd_hidden_inplace", "c_plain_hidden_inplace"), ("c_l2scale_hidden_inplace", "c_plain_hidden_inplace"),
                ("X_inplace", "X_outofplace"), ("L_inplace", "L_outofplace"),("X_inplace", "R_inplace"), ("X_inplace", "L_inplace"), ("XC_inplace", "X_inplace"), ("X_outofplace", "L_outofplace"),
                ("c_l2scale_inplace", "c_plain_inplace"), ("c_plain_simd_inplace", "c_plain_inplace"), ("c_l2scale_inplace", "c_plain_simd_inplace"),
                ("c_l2scale_inplace", "c_l2scale_outofplace"), ("c_sumabs2", "X_inplace"), ("c_l2scale_inplace", "X_inplace"))
const PAIRS2 = (("X_value", "L_value"), ("V_value", "L_value"), ("V_value", "X_value"), ("XC_value", "X_value"), ("X_value", "R_value"),
                ("X_vap", "L_vap"), ("V_vap", "L_vap"), ("V_vap", "X_vap"), ("X_vap", "R_vap"))

function inplace_equal(d, fC, fW)
    all(d.inputs) do u
        x1, x2 = copy(u), copy(u)
        C.ProximalBase.prox!(x1, fC, x1, d.γ)
        W.ProximalBase.prox!(x2, fW, x2, d.γ)
        isequal(x1, x2)
    end
end

function value_equal(d, fC, fV)
    all(d.inputs) do u
        y1, y2 = similar(u), similar(u)
        v1 = C.ProximalBase.value_and_prox!(y1, fC, u, d.γ)
        v2 = V.ProximalBase.value_and_prox!(y2, fV, u, d.γ)
        isequal(fC(u), fV(u)) && isequal(v1, v2) && isequal(y1, y2)
    end
end

function run_case(io_times, io_ratios, id, d, nentries, section)
    T = eltype(d.inputs[1])
    mk(PB) = PB.GroupSCAD(d.λ0, d.groups, d.w; a = d.a)
    fC, fCC, fL, fR = mk(C.ProximalBase), mk(CC.ProximalBase), mk(L.ProximalBase), R.ProximalBase.GroupNormL2(d.λ0, d.groups, d.w)
    fV = VARIANT === nothing ? nothing : mk(V.ProximalBase)
    fC3 = mk(C3.ProximalBase)
    fW = VARIANT_INPLACE === nothing ? nothing : mk(W.ProximalBase)
    section == 2 && fV !== nothing && println("   value and fused value of V bitwise equal to X on every input: ", value_equal(d, fC, fV))
    section == 1 && fW !== nothing && println("   in-place output of W bitwise equal to X on every input: ", inplace_equal(d, fC, fW))
    list = probes(d, fC, fCC, fL, fR, fV, fC3, fW, Val(section))
    names = first.(list)
    bench = Dict(name => prepare(op...) for (name, op) in list)
    times = Dict(name => Float64[] for name in names)
    for rep in 1:REPS
        ord = circshift(names, rep)
        isodd(rep) || reverse!(ord)
        for name in ord
            t = timed_run(bench[name])
            push!(times[name], t)
            println(io_times, id, ',', name, ',', rep, ',', t)
        end
    end
    println("== ", id)
    for name in names
        med = median(filter(!isnan, times[name]))
        per = name == "c_scalar" ? med / TOTAL : med / nentries
        @printf("   %-22s %10.1f ns   %8.3f ns/%s\n", name, med, per, name == "c_scalar" ? "call" : "entry")
    end
    for (a, b) in (section == 1 ? PAIRS1 : PAIRS2)
        haskey(times, a) && haskey(times, b) || continue
        ratios = times[a] ./ times[b]
        ratios = filter(!isnan, ratios)
        line = @sprintf("   %-22s / %-22s median %.3f  [min %.3f, max %.3f]", a, b, median(ratios), minimum(ratios), maximum(ratios))
        println(line)
        println(io_ratios, id, "  ", line)
    end
    flush(stdout)
end

function main()
    mkpath(OUTDIR)
    sha(root) = try readchomp(Cmd(`git rev-parse HEAD`; dir = root)) catch; "not a git checkout" end
    open(joinpath(OUTDIR, "provenance.txt"), "w") do io
        println(io, "UTC: ", now(UTC))
        println(io, "Julia: ", VERSION, ", Julia commit: ", Base.GIT_VERSION_INFO.commit, ", ", Base.julia_cmd())
        println(io, "CPU: ", first(Sys.cpu_info()).model, ", architecture: ", Sys.ARCH, ", Julia CPU target: ", Sys.CPU_NAME)
        println(io, "Host: ", gethostname(), ", Slurm job: ", get(ENV, "SLURM_JOB_ID", "none"), ", threads: ", Threads.nthreads(), ", bounds: ", Base.JLOptions().check_bounds)
        println(io, "BenchmarkTools: ", Base.pkgversion(BenchmarkTools))
        println(io, "Candidate (C, CC): ", SOURCE, " at ", sha(SOURCE), "; legacy (L): ", LEGACY, " at ", sha(LEGACY), "; reference (R): ", REFERENCE,
                " at ", sha(REFERENCE), "; variant (V): ", VARIANT === nothing ? "none" : VARIANT)
        println(io, "Load order: ", REVERSE ? "reversed" : "C, CC, L, R, V")
        println(io, "Repetitions: ", REPS, ", samples per run: ", SAMPLES, ", seconds cap: ", SECONDS, ", inputs per case: ", NINPUTS, ", filter: ", FILTER)
        println(io, "Input configuration: entries ", TOTAL, ", shape ", SHAPE, ", level ", LEVEL, ", steps ", STEPS, ", regimes ", REGIMES)
    end
    println(read(joinpath(OUTDIR, "provenance.txt"), String))
    open(joinpath(OUTDIR, "times.csv"), "w") do io_times
        open(joinpath(OUTDIR, "ratios.txt"), "w") do io_ratios
            open(joinpath(OUTDIR, "branches.csv"), "w") do io_br
                println(io_times, "case,probe,repetition,median_ns")
                println(io_br, "case,copy,zero,scale,cold")
                for (section, sizes) in ((1, (512, 4096)), (2, (1, 4, 32)))
                    for T in (Float32, Float64), m in sizes, layout in (:range, :perm), (sname, step) in STEPS, regime in REGIMES
                        id = "$T/m$m/$layout/$sname/$regime/$(section == 1 ? "inplace" : "value")"
                        occursin(FILTER, id) || continue
                        d = makedata(T, m, layout, step, regime, NINPUTS)
                        fC = C.ProximalBase.GroupSCAD(d.λ0, d.groups, d.w; a = d.a)
                        counts = Dict(:copy => 0, :zero => 0, :scale => 0, :cold => 0)
                        for x in d.inputs, (k, G) in enumerate(d.groups)
                            counts[branch(C.ProximalBase, fC, x, G, k, d.γ)] += 1
                        end
                        println(io_br, id, ',', counts[:copy], ',', counts[:zero], ',', counts[:scale], ',', counts[:cold])
                        @printf("-- %s branches over %d inputs x %d groups: copy=%d zero=%d scale=%d cold=%d\n", id, NINPUTS,
                                length(d.groups), counts[:copy], counts[:zero], counts[:scale], counts[:cold])
                        run_case(io_times, io_ratios, id, d, TOTAL, section)
                        flush(io_times); flush(io_ratios); flush(io_br)
                    end
                end
            end
        end
    end
end

main()
