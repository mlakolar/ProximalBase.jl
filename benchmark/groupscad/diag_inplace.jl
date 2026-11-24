# Diagnosis of the in-place cost of GroupSCAD for large groups, of the small-group value, and of the compact block sweeps, in
# one process.
#
#   PB_SOURCE=<baseline tree> PB_LEGACY=<archived tree> PB_REFERENCE=<tree with GroupNormL2> \
#   [PB_VARIANT=<tree for the V probes>] [PB_VARIANT_INPLACE=<tree for the W probes>] PB_OUTPUT=<dir> \
#   julia --startup-file=no --threads=1 --check-bounds=no --project=benchmark benchmark/groupscad/diag_inplace.jl
#
# (diag.sh runs it from a package root and makes missing variant trees.) The trees are folders with a src/ folder
# (git archive <sha> src | tar -x -C <tree>). PB_LEGACY is the archived implementation (eff21b6d7d33c622313122f46c6ee71693c101ee),
# PB_REFERENCE any tree with the shared GroupNormL2 (the base of the pull request). The two variant flags are independent: the V
# probes use the tree of PB_VARIANT and the W probes (sections 1 and 3) that of PB_VARIANT_INPLACE; a probe whose flag is unset
# is not run. Modules: C (PB_SOURCE), CC and C3 (a second and a third compilation of it: A/A controls), L (legacy), R
# (reference), V (PB_VARIANT), W and W2 (PB_VARIANT_INPLACE and a second compilation of it). PB_REVERSE=1 reverses the load
# order; the forward order is C, CC, C3, L, R, then V, W and W2 where supplied, and the order actually used is written to
# provenance.txt together with a digest of each tree.
#
# Comparing two commits: PB_SOURCE is the baseline tree and the new tree (the head of the pull request, or any tree of this
# family) is passed as both PB_VARIANT and PB_VARIANT_INPLACE: V_value, V_vap and W_inplace are then the new tree's value, fused
# value and in-place proximal map, and the W sweeps (section 3) its compact blocks, next to X (baseline), L (legacy) and R
# (GroupNormL2); a head whose compact route does not use the in-place flag has the baseline's out-of-place loops there, so that
# W/C of the compact sweeps then compares equal code. The lines "bitwise equal" compare the new tree's values and in-place points
# with the baseline's. To measure the two experimental variants of make_variants.py against a baseline instead, give a baseline whose source has the pattern that
# make_variants.py edits (the tree of 9d2fefc71 does; the head does not, for the in-place variant).
#
# Cases: the input generation of run.jl (4096 entries, level 3/4, shape 3.7, unit weights, 8 cycled inputs per case, steps
# below, at and above a - 1, regimes zero, soft, middle, kept, contiguous ranges and permuted index vectors, Float32 and
# Float64), and in sections 1 and 2 a fifth regime, mixed, in which each group of each input draws its regime at random
# (seeded) from the four; the timings of the mixed cases are compared with those of the regimes alone to judge whether
# branches that are not predictable matter (that is an inference from the timings; the script does not count mispredictions).
# Section 1 (group sizes 512 and 4096) is about the proximal map in place; section 2 (group sizes 1, 4 and 32) about the value
# and the fused value; section 3 (group sizes 1, 4, 32 and 512, steps below and above a - 1, the four regimes) about the compact
# block sweeps: prox_block!(u_k, f, u_k, k, γ) over every group with out === u (the block_sweep_inplace operation of run.jl) and
# prox_block!(out_k, f, u_k, k, γ) into separate outputs (block_sweep). PB_FILTER selects cases by a regular expression on their
# ids (the last part of an id is inplace, value or blocks).
#
# For each case the script prints
#   * the branch that the baseline takes for each group of the 8 inputs, counted over the inputs and the groups: copy (the
#     scalar kernel returns the norm), zero, scale, or cold (the ordinary guard fails);
#   * the absolute median of each probe in ns and ns per entry (the median over the repetitions of the median of a BenchmarkTools
#     run: evals = 1, the setup restores the data from the input pool, so that they are rewritten before each sample);
#   * the paired ratios of probes, as the median over the repetitions with the minimum and maximum.
# Probes are timed in a rotating order that is reversed in alternate repetitions (PB_REPS repetitions, default 8, at least 8;
# PB_SAMPLES samples per run, default 200, at least 3; PB_SECONDS cap, default 0.05). A run that throws the InexactError of
# sizehint! (see run.jl: a first sample longer than the cap; the stack has a sizehint! frame and a BenchmarkTools frame) or has
# fewer than three samples (whatever the reason) is repeated, up to 3 attempts; BenchmarkTools 1.8.0 is the
# version of the campaigns; a probe that fails in a repetition is recorded as NaN and left out of the medians. Section 1 probes:
#   X_inplace, XC_inplace (A/A), L_inplace, R_inplace   prox!(x, f, x, γ) of the baseline, its second compilation, the legacy
#                                                       tree and GroupNormL2
#   X3_inplace                                          a third compilation of the baseline (A/A: the spread among compilations)
#   W_inplace                                           (PB_VARIANT_INPLACE) prox!(x, f, x, γ) of that tree (its output is compared
#                                                       bit for bit with X on every input)
#   X_outofplace, L_outofplace                          prox!(y, f, x, γ)
#   c_sumabs2            the sum of squares of every group (the shared _indexed_sumabs2)
#   c_l2scale_inplace    _l2_scale!(x, x, G, q) for every group (the shared @simd loop; q = t/r of the first input, per group)
#   c_plain_inplace      @inbounds for i in G; x[i] = q * x[i]; end
#   c_plain_simd_inplace the same loop with @simd
#   c_l2scale_hidden_inplace, c_plain_hidden_inplace, c_plain_simd_hidden_inplace
#                        the same in-place loops behind a @noinline call, so that the compiler does not know that y === x (as with
#                        two argument names in prox!(x, f, x, γ)); a vectorized loop with distinct argument names can need an
#                        overlap test at run time, which fails for one array
#   X_loop_inplace       the baseline's loop over the groups (_gscad_prox_loop!) called directly, without the argument checks (for a
#                        tree with the in-place flag, with Val(true), Val(false), Val(true))
#   c_l2scale_outofplace _l2_scale!(y, x, G, q)
#   c_scalar             the scalar kernel _scad_magnitude(r, λ, a, γ) over 4096 radii within 0.1% of the target norm of the regime (the
#                        soft target for a mixed case), in ns per call
# Section 2 probes: X_value, XC_value, V_value, L_value, R_value (f(x)) and X_vap, V_vap, L_vap, R_vap (value_and_prox!(y, f, x, γ)).
# Section 3 probes: C_sweep_inplace, CC_sweep_inplace, W_sweep_inplace, W2_sweep_inplace (the compact sweep with out === u of the
# baseline, its second compilation, the PB_VARIANT_INPLACE tree and its second compilation) and C_sweep, CC_sweep, W_sweep, W2_sweep
# (the sweep into separate outputs); ratios W/C, W2/CC, CC/C and W2/W of each.
# Files in PB_OUTPUT: provenance.txt, branches.csv, times.csv (case, probe, repetition, median_ns) and ratios.txt.
using BenchmarkTools, Dates, Random, SHA, Statistics, Printf

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
module W2 end
module C3 end
const LOAD_ORDER = String[]
let mods = [("C", C, SOURCE), ("CC", CC, SOURCE), ("C3", C3, SOURCE), ("L", L, LEGACY), ("R", R, REFERENCE)]
    VARIANT === nothing || push!(mods, ("V", V, VARIANT))
    VARIANT_INPLACE === nothing || push!(mods, ("W", W, VARIANT_INPLACE), ("W2", W2, VARIANT_INPLACE))
    for (name, m, root) in (REVERSE ? reverse(mods) : mods)
        Base.include(m, joinpath(root, "src", "ProximalBase.jl"))
        push!(LOAD_ORDER, name)
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
SAMPLES >= 3 || error("PB_SAMPLES must be at least 3")
REPS >= 8 || error("PB_REPS must be at least 8")
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
            # :mixed draws the regime of each group (and input) at random from the four regimes
            reg = regime === :mixed ? rand(rng, (:zero, :soft, :middle, :kept)) : regime
            r = group_norm(reg, Float64(γ), Float64(a), Float64(λ0))
            d = randn(rng, m)
            d ./= sqrt(sum(abs2, d))
            x[G] .= T.(r .* d)
        end
        inputs[j] = x
    end
    (; groups, inputs, a, λ0, γ, w, r = group_norm(regime === :mixed ? :soft : regime, Float64(γ), Float64(a), Float64(λ0)))
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

# sha256 over the relative paths and the bytes of the files below dir: the identity of an archived tree
function tree_digest(dir)
    isdir(dir) || return "missing"
    ctx = SHA.SHA256_CTX()
    for p in sort!([joinpath(r, f) for (r, _, fs) in walkdir(dir) for f in fs])
        SHA.update!(ctx, codeunits(relpath(p, dir) * "\0"))
        SHA.update!(ctx, read(p))
    end
    bytes2hex(SHA.digest!(ctx))
end

function timed_run(b)
    for attempt in 1:RETRIES
        try
            t = run(b)
            length(t.times) >= 3 && return median(t).time      # fewer than three samples: repeated (see run.jl)
        catch e
            st = stacktrace(catch_backtrace())
            e isa InexactError && any(fr -> fr.func === :sizehint!, st) && any(fr -> occursin("BenchmarkTools", string(fr.file)), st) || rethrow()   # see run.jl
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

# The compact sweeps of a tree: with out === u (the buffers Wk are restored by the setup) and into separate outputs. A function
# of the typed prox_block! so that the closures call it directly.
function block_sweeps(pb!::P, f, Wk, outs, U, cur, γ) where {P}
    K = length(Wk)
    inplace = () -> begin
        for k in 1:K
            @inbounds pb!(Wk[k], f, Wk[k], k, γ)
        end
        nothing
    end
    outofplace = () -> begin
        Uc = U[cur[]]
        for k in 1:K
            @inbounds pb!(outs[k], f, Uc[k], k, γ)
        end
        nothing
    end
    inplace, outofplace
end

function block_equal(d, fC, fW)
    all(d.inputs) do u
        ok = true
        for (k, G) in enumerate(d.groups)
            a, b = u[G], u[G]
            C.ProximalBase.prox_block!(a, fC, a, k, d.γ)
            W.ProximalBase.prox_block!(b, fW, b, k, d.γ)
            ok &= isequal(a, b)
        end
        ok
    end
end

# Probes of one case on the typed data. Each probe is a named (op, setup) pair; the setups cycle the input pool outside the timing.
function probes(d, fC, fCC, fL, fR, fV, fC3, fW, fW2, ::Val{section}) where {section}
    T = eltype(d.inputs[1])
    inputs, γ, groups = d.inputs, d.γ, d.groups
    x = copy(inputs[1]); y = similar(x)
    cur = Ref(0)
    restore!() = (cur[] = mod1(cur[] + 1, NINPUTS); copyto!(x, inputs[cur[]]); nothing)
    val = Ref(zero(T))
    list = Pair{String, Any}[]
    add!(name, op, setup = restore!) = push!(list, name => (op, setup))
    if section == 3
        K = length(groups)
        U = [[inputs[j][G] for G in groups] for j in 1:NINPUTS]
        Wk = [similar(u) for u in U[1]]
        outs = [similar(u) for u in U[1]]
        advance!() = (cur[] = mod1(cur[] + 1, NINPUTS); nothing)
        restore_blocks!() = (cur[] = mod1(cur[] + 1, NINPUTS); Uc = U[cur[]]; foreach(k -> copyto!(Wk[k], Uc[k]), 1:K); nothing)
        for (tag, PB, f) in (("C", C, fC), ("CC", CC, fCC), ("W", W, fW), ("W2", W2, fW2))
            f === nothing && continue
            ip, op = block_sweeps(getfield(PB, :ProximalBase).prox_block!, f, Wk, outs, U, cur, γ)
            add!("$(tag)_sweep_inplace", ip, restore_blocks!)
            add!("$(tag)_sweep", op, advance!)
        end
    elseif section == 1
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
        if any(m -> m.nargs == 8, methods(C.ProximalBase._gscad_prox_loop!))      # the in-place flag is a seventh argument
            add!("X_loop_inplace", () -> (C.ProximalBase._gscad_prox_loop!(x, fC, x, γ, Val(true), Val(false), Val(true)); nothing))
        else
            add!("X_loop_inplace", () -> (C.ProximalBase._gscad_prox_loop!(x, fC, x, γ, Val(true), Val(false)); nothing))
        end
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
const PAIRS3 = (("W_sweep_inplace", "C_sweep_inplace"), ("W2_sweep_inplace", "CC_sweep_inplace"), ("CC_sweep_inplace", "C_sweep_inplace"),
                ("W2_sweep_inplace", "W_sweep_inplace"), ("W_sweep", "C_sweep"), ("W2_sweep", "CC_sweep"), ("CC_sweep", "C_sweep"),
                ("W2_sweep", "W_sweep"))
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
    fW2 = VARIANT_INPLACE === nothing ? nothing : mk(W2.ProximalBase)
    section == 2 && fV !== nothing && println("   value and fused value of V bitwise equal to X on every input: ", value_equal(d, fC, fV))
    section == 1 && fW !== nothing && println("   in-place output of W bitwise equal to X on every input: ", inplace_equal(d, fC, fW))
    section == 3 && fW !== nothing && println("   compact in-place output of W bitwise equal to C on every input: ", block_equal(d, fC, fW))
    list = probes(d, fC, fCC, fL, fR, fV, fC3, fW, fW2, Val(section))
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
        good = filter(!isnan, times[name])
        if isempty(good)                         # every attempt of every repetition failed: report it, do not abort the run
            @printf("   %-22s %10s ns   (no timing: every trial failed)\n", name, "NaN")
            continue
        end
        med = median(good)
        per = name == "c_scalar" ? med / TOTAL : med / nentries
        @printf("   %-22s %10.1f ns   %8.3f ns/%s\n", name, med, per, name == "c_scalar" ? "call" : "entry")
    end
    for (a, b) in (section == 1 ? PAIRS1 : section == 2 ? PAIRS2 : PAIRS3)
        haskey(times, a) && haskey(times, b) || continue
        ratios = times[a] ./ times[b]
        ratios = filter(!isnan, ratios)
        if isempty(ratios)                       # a probe of the pair has no timing in any repetition
            line = @sprintf("   %-22s / %-22s no ratio (no repetition has both timings)", a, b)
            println(line)
            println(io_ratios, id, "  ", line)
            continue
        end
        line = @sprintf("   %-22s / %-22s median %.3f  [min %.3f, max %.3f]", a, b, median(ratios), minimum(ratios), maximum(ratios))
        println(line)
        println(io_ratios, id, "  ", line)
    end
    flush(stdout)
end

function main()
    mkpath(OUTDIR)
    sha(root) = try readchomp(pipeline(Cmd(`git rev-parse HEAD`; dir = root); stderr = devnull)) catch; "not a git checkout" end
    # the git revision where there is one, and always the sha256 of the paths and bytes of src/ (a git archive has no revision)
    identity(root) = string(sha(root), ", src/ sha256 ", tree_digest(joinpath(root, "src")))
    open(joinpath(OUTDIR, "provenance.txt"), "w") do io
        println(io, "UTC: ", now(UTC))
        println(io, "Julia: ", VERSION, ", Julia commit: ", Base.GIT_VERSION_INFO.commit, ", ", Base.julia_cmd())
        println(io, "CPU: ", first(Sys.cpu_info()).model, ", architecture: ", Sys.ARCH, ", Julia CPU target: ", Sys.CPU_NAME)
        println(io, "Host: ", gethostname(), ", Slurm job: ", get(ENV, "SLURM_JOB_ID", "none"), ", threads: ", Threads.nthreads(), ", bounds: ", Base.JLOptions().check_bounds)
        println(io, "BenchmarkTools: ", Base.pkgversion(BenchmarkTools))
        println(io, "Baseline (C, CC, C3): ", SOURCE, " at ", identity(SOURCE))
        println(io, "Legacy (L): ", LEGACY, " at ", identity(LEGACY))
        println(io, "Reference (R): ", REFERENCE, " at ", identity(REFERENCE))
        println(io, "Variant V: ", VARIANT === nothing ? "none" : string(VARIANT, " at ", identity(VARIANT)))
        println(io, "Variant W, W2: ", VARIANT_INPLACE === nothing ? "none" : string(VARIANT_INPLACE, " at ", identity(VARIANT_INPLACE)))
        println(io, "Harness: ", @__FILE__, ", sha256 ", bytes2hex(sha256(read(@__FILE__))))
        println(io, "Module load order used: ", join(LOAD_ORDER, ", "))
        println(io, "Repetitions: ", REPS, ", samples per run: ", SAMPLES, ", seconds cap: ", SECONDS, ", inputs per case: ", NINPUTS, ", filter: ", FILTER)
        println(io, "Input configuration: entries ", TOTAL, ", shape ", SHAPE, ", level ", LEVEL, ", steps ", STEPS, ", regimes ", REGIMES)
    end
    println(read(joinpath(OUTDIR, "provenance.txt"), String))
    open(joinpath(OUTDIR, "times.csv"), "w") do io_times
        open(joinpath(OUTDIR, "ratios.txt"), "w") do io_ratios
            open(joinpath(OUTDIR, "branches.csv"), "w") do io_br
                println(io_times, "case,probe,repetition,median_ns")
                println(io_br, "case,copy,zero,scale,cold")
                for (section, sizes) in ((1, (512, 4096)), (2, (1, 4, 32)), (3, (1, 4, 32, 512)))
                    steps = section == 3 ? filter(s -> s[1] != :at, collect(STEPS)) : collect(STEPS)
                    regimes = section == 3 ? REGIMES : (REGIMES..., :mixed)
                    section == 3 && VARIANT_INPLACE === nothing && continue          # the sweeps compare the baseline with the W tree
                    for T in (Float32, Float64), m in sizes, layout in (:range, :perm), (sname, step) in steps, regime in regimes
                        id = "$T/m$m/$layout/$sname/$regime/$(("inplace", "value", "blocks")[section])"
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
