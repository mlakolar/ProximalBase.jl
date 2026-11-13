# Common screening benchmark of the #901 architecture experiment (orchestrator-owned).
#
#   PB_TREES="base=/path/base;ref=/path/ref;A=/path/A" PB_OUTPUT=<dir> [PB_ROUNDS=11] [PB_FILTER=<regex>]
#   [PB_SHARD=i/n] [PB_TARGET_NS=200000] [PB_SAMPLES=5] [PB_SEED=901]
#     julia --threads=1 --check-bounds=no --startup-file=no --project=<benchmark env> screen901.jl
#
# The first tree is role X, the denominator of every ratio. It is compiled twice: X and the A/A control
# "X#aa". Every other tree is compiled once. Each tree is a separate module (Base.include of
# src/ProximalBase.jl). Inputs and output buffers are shared by all roles; function objects and iterates are
# built per role from the same values. Each round times every role in a rotated order (round r starts at
# role r mod nroles). A role's time in a round is the median of PB_SAMPLES samples of `evals` calls, with
# `evals` calibrated once per case on X so that a sample takes about PB_TARGET_NS. Allocations are measured
# per role after warm-up. A coarse sanity check (same NaN/Inf class, relative difference ≤ 1e-3 of the
# scale of the output) flags a broken candidate; numerical accuracy is accuracy901.jl's job.
#
# Output: <dir>/raw.csv (case,role,round,ns), <dir>/alloc.csv (case,role,bytes), <dir>/cases.csv,
# <dir>/sanity.csv, <dir>/provenance.txt. Summarize with summarize901.py.
using Random, Statistics, Dates, Printf, LinearAlgebra, SparseArrays
Threads.nthreads() == 1 || error("one Julia thread required")

const TREES = [let (l, p) = split(s, "="; limit = 2); (String(strip(l)), abspath(strip(p))) end
               for s in split(ENV["PB_TREES"], ";") if !isempty(strip(s))]
const OUT = ENV["PB_OUTPUT"]; mkpath(OUT)
const ROUNDS = parse(Int, get(ENV, "PB_ROUNDS", "11"))
const SAMPLES = parse(Int, get(ENV, "PB_SAMPLES", "5"))
const TARGET_NS = parse(Float64, get(ENV, "PB_TARGET_NS", "200000"))
const FILTER = Regex(get(ENV, "PB_FILTER", "."))
const SHARD = let s = split(get(ENV, "PB_SHARD", "1/1"), "/"); (parse(Int, s[1]), parse(Int, s[2])) end
const SEED = parse(Int, get(ENV, "PB_SEED", "901"))

# roles: X, X#aa, then the other trees
role_specs = [(TREES[1][1], TREES[1][2]), (TREES[1][1] * "#aa", TREES[1][2]), TREES[2:end]...]
const ROLES = String[r[1] for r in role_specs]
const PBS = Module[]
for (i, (label, root)) in enumerate(role_specs)
    m = Module(Symbol("Role", i))
    Base.include(m, joinpath(root, "src", "ProximalBase.jl"))
    push!(PBS, Base.invokelatest(getfield, m, :ProximalBase))
end
const NR = length(ROLES)

open(joinpath(OUT, "provenance.txt"), "w") do io
    println(io, "date ", now())
    println(io, "host ", gethostname())
    println(io, "julia ", VERSION)
    println(io, "cpu ", Sys.cpu_info()[1].model)
    println(io, "affinity ", get(ENV, "PB_CORE", "?"), " slurm_job ", get(ENV, "SLURM_JOB_ID", "-"))
    for (label, root) in TREES
        shafile = joinpath(root, "SHA")
        sha = isfile(shafile) ? readchomp(shafile) : try readchomp(`git -C $root rev-parse HEAD`) catch; "?" end
        dirty = isfile(shafile) ? "" : try isempty(readchomp(`git -C $root status --porcelain --untracked-files=no -- src`)) ? "" : " DIRTY" catch; "" end
        println(io, "tree ", label, " ", root, " ", sha, dirty)
    end
    println(io, "rounds ", ROUNDS, " samples ", SAMPLES, " target_ns ", TARGET_NS, " filter ", FILTER.pattern, " shard ", SHARD, " seed ", SEED)
    println(io, "roles ", join(ROLES, ","))
end

####################################
#  Cases
####################################

struct Case
    id::String
    n::Int
    build::Function      # PB -> (op = closure, out = closure returning outputs for the sanity check)
end
cases = Case[]
addcase!(id, n, build) = push!(cases, Case(id, n, build))

bindcall(fn, args...) = () -> (fn(args...); nothing)
bindstore(out, fn, args...) = () -> (out[1] = fn(args...); nothing)
bindvalue(out, f, x) = () -> (out[1] = f(x); nothing)
bindsweep_block(pb!, outs, f, U, γ) = () -> (for k in eachindex(U); pb!(outs[k], f, U[k], k, γ); end; nothing)
bindsweep_kkt(kkt, kout, f, g, xb) = () -> (for k in eachindex(kout); kout[k] = kkt(f, g, xb, k); end; nothing)

p2(::Type{T}, e) where {T} = ldexp(one(T), e)
emaxT(::Type{T}) where {T} = exponent(floatmax(T))
eminT(::Type{T}) where {T} = exponent(floatmin(T))

# data of a group regime: x (length N), level, weights, step
function group_data(::Type{T}, regime, m, K, rng) where {T}
    N = m * K
    x = randn(rng, T, N)
    w = T[sqrt(T(m)) * T(0.5 + rand(rng)) for _ in 1:K]
    γ = T(0.8)
    # ord: level so that about half of the groups shrink to zero (threshold ≈ median norm)
    λ0 = T(1.0 / 0.8)
    emax, emin = emaxT(T), eminT(T)
    if regime === :ord
    elseif regime === :w0
        w .= zero(T)
    elseif regime === :wmix                   # a quarter unpenalized, the rest ordinary
        for k in 1:4:K; w[k] = zero(T); end
    elseif regime === :zgrp                   # a quarter of the groups exactly zero
        for k in 1:4:K; x[(k-1)*m+1:k*m] .= zero(T); end
    elseif regime === :over                   # sum of squares overflows, norm representable
        x .*= p2(T, emax ÷ 2 + 2); λ0 = p2(T, emax ÷ 2 + 2) / T(0.8)
    elseif regime === :under                  # sum of squares underflows
        x .*= p2(T, emin ÷ 2 - 4); λ0 = p2(T, emin ÷ 2 - 4) / T(0.8)
    elseif regime === :thr_over               # γλ0w overflows; groups are fixed at zero
        γ = p2(T, emax ÷ 2 + 1); λ0 = p2(T, emax ÷ 2 + 1)
    elseif regime === :thr_under              # γλ0w underflows; groups are (almost) unchanged
        γ = p2(T, emin ÷ 2 - 1); λ0 = p2(T, emin ÷ 2 - 1)
    elseif regime === :inf                    # a quarter of the groups hold an Inf entry
        for k in 1:4:K; x[(k-1)*m+1] = T(Inf); end
    elseif regime === :nan                    # a quarter of the groups hold a NaN entry
        for k in 1:4:K; x[(k-1)*m+1] = T(NaN); end
    elseif regime === :lamInf
        λ0 = T(Inf)
    else
        error("regime $regime")
    end
    (x = x, λ0 = λ0, w = w, γ = γ)
end

function group_layout(m, K, layout, rng)
    layout === :range && return [((k-1)*m+1):(k*m) for k in 1:K]
    perm = randperm(rng, m * K)
    [sort!(perm[((k-1)*m+1):(k*m)]) for k in 1:K]
end

const GROUP_REGIMES = (:ord, :w0, :wmix, :zgrp, :over, :under, :thr_over, :thr_under, :inf, :nan, :lamInf)
const GROUP_SHAPES = ((1, 4096, :range), (2, 2048, :range), (5, 819, :range), (5, 819, :perm),
                      (32, 128, :range), (1024, 4, :range), (1024, 4, :perm))
const FULL_OPS_REGIMES = (:ord, :w0, :over, :under)

for T in (Float64, Float32), (m, K, layout) in GROUP_SHAPES, regime in GROUP_REGIMES
    rng = MersenneTwister(hash((SEED, T, m, K, layout, regime)))
    d = group_data(T, regime, m, K, rng)
    groups = group_layout(m, K, layout, rng)
    N = m * K
    y = similar(d.x); out1 = zeros(T, 1)
    tag = "Group/$T/m$m/$layout/$regime"
    mkf(PB) = PB.GroupNormL2(d.λ0, groups, d.w)
    addcase!("$tag/value", N, PB -> (f = mkf(PB);
        (op = bindvalue(out1, f, d.x), out = () -> f(d.x))))
    addcase!("$tag/prox", N, PB -> (f = mkf(PB);
        (op = bindcall(PB.prox!, y, f, d.x, d.γ), out = () -> (PB.prox!(y, f, d.x, d.γ); copy(y)))))
    addcase!("$tag/fused", N, PB -> (f = mkf(PB);
        (op = bindstore(out1, PB.value_and_prox!, y, f, d.x, d.γ), out = () -> (v = PB.value_and_prox!(y, f, d.x, d.γ); vcat(v, y)))))
    if regime in FULL_OPS_REGIMES && layout === :range
        U = [d.x[G] for G in groups]; outs = [similar(u) for u in U]
        addcase!("$tag/block", N, PB -> (f = mkf(PB);
            (op = bindsweep_block(PB.prox_block!, outs, f, U, d.γ),
             out = () -> (for k in 1:K; PB.prox_block!(outs[k], f, U[k], k, d.γ); end; reduce(vcat, outs)))))
        kout = zeros(T, K)
        addcase!("$tag/kkt", N, PB -> (f = mkf(PB); xb = PB.BlockIterate(zeros(T, N), groups);
            (op = bindsweep_kkt(PB.kkt_violation, kout, f, d.x, xb), out = () -> [PB.kkt_violation(f, d.x, xb, k) for k in 1:K])))
    end
end

# NormL2 (one group, weight one)
const NORM_REGIMES = (:ord, :zero, :over, :under, :thr_under, :inf, :nan)
for T in (Float64, Float32), n in (1, 2, 5, 32, 1024, 65536), regime in NORM_REGIMES
    rng = MersenneTwister(hash((SEED, T, n, regime, :norm)))
    x = randn(rng, T, n); λ = T(0.25) * sqrt(T(n)); γ = T(0.8)
    emax, emin = emaxT(T), eminT(T)
    regime === :zero && (x .= zero(T))
    regime === :over && (x .*= p2(T, emax ÷ 2 + 2); λ *= p2(T, emax ÷ 2 + 2))
    regime === :under && (x .*= p2(T, emin ÷ 2 - 4); λ *= p2(T, emin ÷ 2 - 4))
    regime === :thr_under && (λ = p2(T, emin ÷ 2 - 1); γ = p2(T, emin ÷ 2 - 1))
    regime === :inf && (x[1] = T(Inf))
    regime === :nan && (x[1] = T(NaN))
    y = similar(x); out1 = zeros(T, 1)
    tag = "Norm/$T/n$n/$regime"
    addcase!("$tag/value", n, PB -> (f = PB.NormL2(λ); (op = bindvalue(out1, f, x), out = () -> f(x))))
    addcase!("$tag/prox", n, PB -> (f = PB.NormL2(λ);
        (op = bindcall(PB.prox!, y, f, x, γ), out = () -> (PB.prox!(y, f, x, γ); copy(y)))))
    addcase!("$tag/fused", n, PB -> (f = PB.NormL2(λ);
        (op = bindstore(out1, PB.value_and_prox!, y, f, x, γ), out = () -> (v = PB.value_and_prox!(y, f, x, γ); vcat(v, y)))))
end

# Consumers of the shared kernels (smoke coverage: SqrNormL2, norm_diff, NormL1)
for T in (Float64, Float32), n in (32, 1024), regime in (:ord, :over)
    rng = MersenneTwister(hash((SEED, T, n, regime, :consumers)))
    x = randn(rng, T, n); c = randn(rng, T, n)
    regime === :over && (x .*= p2(T, emaxT(T) ÷ 2 + 2); c .*= p2(T, emaxT(T) ÷ 2 + 2))
    out1 = zeros(T, 1); y = similar(x)
    addcase!("SqrNormL2/$T/n$n/$regime/value", n, PB -> (f = PB.SqrNormL2(T(0.7), c); (op = bindvalue(out1, f, x), out = () -> f(x))))
    addcase!("SqrNormL2/$T/n$n/$regime/prox", n, PB -> (f = PB.SqrNormL2(T(0.7), c);
        (op = bindcall(PB.prox!, y, f, x, T(0.8)), out = () -> (PB.prox!(y, f, x, T(0.8)); copy(y)))))
    addcase!("norm_diff/$T/n$n/$regime/value", n, PB -> (op = bindstore(out1, PB.norm_diff, x, c), out = () -> PB.norm_diff(x, c)))
    w = T[0.5 + rand(rng) for _ in 1:n]
    addcase!("NormL1/$T/n$n/$regime/value", n, PB -> (f = PB.NormL1(T(0.3), w); (op = bindvalue(out1, f, x), out = () -> f(x))))
end

selected = [c for c in cases if occursin(FILTER, c.id)]
selected = [c for (i, c) in enumerate(selected) if mod1(i, SHARD[2]) == SHARD[1]]
println("cases: ", length(cases), " selected: ", length(selected), " roles: ", join(ROLES, ","))
flush(stdout)

####################################
#  Timing
####################################

@noinline function runmany(op::F, evals::Int) where {F}
    for _ in 1:evals; op(); end
    nothing
end
@noinline function sample_ns(op::F, evals::Int) where {F}
    t0 = time_ns(); runmany(op, evals); t1 = time_ns()
    (t1 - t0) / evals
end
function calibrate(op)
    runmany(op, 3)
    evals = 1
    while true
        t = sample_ns(op, evals) * evals
        t >= TARGET_NS / 4 && return max(1, round(Int, evals * TARGET_NS / t))
        evals *= 4
        evals > 10^8 && return evals
    end
end
@noinline function allocs(op::F) where {F}
    op(); op()
    @allocated op()
end

classof(v::Real) = isnan(v) ? 2 : isinf(v) ? 3 : 1
function sane(a, b)
    a isa Real && (a = [a]); b isa Real && (b = [b])
    length(a) == length(b) || return false
    sc = maximum(v -> isfinite(v) ? abs(v) : zero(v), a; init = zero(eltype(a)))
    all(i -> classof(a[i]) == classof(b[i]) && (!isfinite(a[i]) || abs(a[i] - b[i]) <= 1e-3 * sc + 4 * nextfloat(zero(sc))), eachindex(a, b))
end

raw = open(joinpath(OUT, "raw.csv"), "w"); println(raw, "case,role,round,ns")
alc = open(joinpath(OUT, "alloc.csv"), "w"); println(alc, "case,role,bytes")
cio = open(joinpath(OUT, "cases.csv"), "w"); println(cio, "case,n,evals")
san = open(joinpath(OUT, "sanity.csv"), "w"); println(san, "case,role,ok")
t_start = time()
for (ci, c) in enumerate(selected)
    built = [c.build(PB) for PB in PBS]
    ops = [b.op for b in built]
    ref_out = built[1].out()
    for r in 2:NR
        ok = try sane(ref_out, built[r].out()) catch err; false end
        println(san, c.id, ",", ROLES[r], ",", ok)
        ok || @warn "sanity mismatch" c.id ROLES[r]
    end
    for r in 1:NR
        println(alc, c.id, ",", ROLES[r], ",", allocs(ops[r]))
    end
    evals = calibrate(ops[1])
    println(cio, c.id, ",", c.n, ",", evals)
    for r in 1:NR; runmany(ops[r], evals); end          # warm-up of every role
    for round in 1:ROUNDS
        for j in 0:NR-1
            r = mod(round + j, NR) + 1
            op = ops[r]
            s = Float64[sample_ns(op, evals) for _ in 1:SAMPLES]
            println(raw, c.id, ",", ROLES[r], ",", round, ",", median(s))
        end
    end
    flush(raw); flush(alc); flush(cio); flush(san)
    if ci % 25 == 0
        @printf("%d/%d cases, %.0f s\n", ci, length(selected), time() - t_start); flush(stdout)
    end
end
close(raw); close(alc); close(cio); close(san)
open(joinpath(OUT, "provenance.txt"), "a") do io
    @printf(io, "elapsed_s %.1f\n", time() - t_start)
    println(io, "done ", now())
end
println("done in ", round(time() - t_start; digits = 1), " s")
