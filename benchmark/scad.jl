# Paired SCAD benchmark for issue #953. Use one exclusive node and one Julia/BLAS thread.
#
# PB_ARCHIVE and PB_CANDIDATE point to separate clean worktrees. Set PB_PREOPT
# instead of PB_ARCHIVE to compare against the pre-optimization reference.
# Activate the candidate's instantiated benchmark environment:
#
# PB_ARCHIVE=... PB_CANDIDATE=... PB_OUTPUT=... \
#   julia --startup-file=no --threads=1 --check-bounds=no --project=benchmark benchmark/scad.jl
#
# Rerun in a fresh process with PB_REVERSE=1 and a different PB_OUTPUT directory.
# PB_CHECK_ONLY=1 checks the outputs without timing. PB_FILTER selects case IDs.
# PB_ROUNDS=1 PB_SAMPLES=16 is for a short pilot.
using BenchmarkTools, Dates, LinearAlgebra, Random, SHA, Statistics

Threads.nthreads() == 1 || error("one Julia thread required")
BLAS.set_num_threads(1)

const COMPARISON = haskey(ENV, "PB_PREOPT") ? :preopt : :archive
const REFERENCE = abspath(ENV[COMPARISON === :preopt ? "PB_PREOPT" : "PB_ARCHIVE"])
const CANDIDATE = abspath(ENV["PB_CANDIDATE"])
const REVERSED = get(ENV, "PB_REVERSE", "0") == "1"
const SIZES = (8, 1024, 65536)
const CHECK_ONLY = get(ENV, "PB_CHECK_ONLY", "0") == "1"

module ReferenceSource end
module CandidateSource end
module ControlSource end

sources = ((ReferenceSource, REFERENCE), (CandidateSource, CANDIDATE))
for (m, root) in (REVERSED ? reverse(sources) : sources)
    Base.include(m, joinpath(root, "src", "ProximalBase.jl"))
end
Base.include(ControlSource, joinpath(CANDIDATE, "src", "ProximalBase.jl"))

# With gamma = 4 the global switch is at 2.125 for the chosen parameters.
# Legacy subtracts nearly equal complete costs to choose between zero and the
# first-piece minimum at 2. Exclude both transition neighborhoods from the
# equivalent benchmark dataset. Switches are tested separately.
function magnitudes(rng, ::Type{T}, n) where {T}
    x = T(8) .* rand(rng, T, n) .- T(4)
    for i in eachindex(x)
        if abs(abs(x[i]) - T(2.125)) < T(0.025)
            x[i] = copysign(T(2.07), x[i])
        elseif abs(abs(x[i]) - T(2)) < T(0.004)
            x[i] = copysign(T(1.99), x[i])
        end
    end
    x
end

function make_data(::Type{T}) where {T}
    rng = MersenneTwister(953)
    dense = Dict(n => begin
        x = magnitudes(rng, T, n)
        steps = T(0.25) .+ rand(rng, T, n)
        mixed = T[ (T(0.75), T(2.45), T(4))[mod1(i, 3)] for i in 1:n ]
        (; x, steps, mixed, out = similar(x))
    end for n in SIZES)
    iterates = Dict(n => begin
        sparse = copy(dense[n].x)
        for i in eachindex(sparse)
            (i % 10 == 1) || (sparse[i] = zero(T))
        end
        groups = [collect(k:16:n) for k in 1:16]
        block = copy(dense[n].x)
        for k in 2:2:16
            block[groups[k]] .= zero(T)
        end
        p = isqrt(n)
        symmetric = [(r, c, magnitudes(rng, T, 1)[1]) for c in 1:p for r in c:p if rand(rng) < 0.1]
        (; sparse, block, groups, p, symmetric)
    end for n in (1024, 65536))
    (; dense, iterates)
end

function scalar_loop!(kernel, out, f, x, γ)
    @inbounds for i in eachindex(x, out)
        out[i] = kernel(f, x[i], i, γ)
    end
    nothing
end

# Resolve callables while building the probes, then close over concrete types.
loop_case(kernel, out, f, x, γ) =
    (() -> scalar_loop!(kernel, out, f, x, γ),
     () -> (scalar_loop!(kernel, out, f, x, γ); copy(out)))
prox_case(px, out, f, x, γ, snapshot) =
    (() -> (px(out, f, x, γ); nothing),
     () -> (px(out, f, x, γ); snapshot(out)))
prox_case(px, out, f, x, snapshot) =
    (() -> (px(out, f, x); nothing),
     () -> (px(out, f, x); snapshot(out)))
function allocating_prox_case(prox, f, x, steps...)
    result = Ref{typeof(x)}()
    run() = prox(f, x, steps...)
    function check()
        out = run()
        typeof(out) === typeof(x) || error("allocating prox changed the array type")
        out
    end
    (() -> (result[] = run(); nothing), check)
end
function value_case(f, x, ::Type{T}) where {T}
    result = Ref{T}()
    (() -> (result[] = f(x); nothing), () -> f(x))
end
function fused_case(vp, out, f, x, γ, snapshot, ::Type{T}) where {T}
    result = Ref{T}()
    (() -> (result[] = vp(out, f, x, γ); nothing),
     () -> (v = vp(out, f, x, γ); (v, snapshot(out))))
end

function make_probes(PB, norm_module, ::Type{T}, data) where {T}
    probes = NamedTuple[]
    function add(id, n, kind, (op, check); comparison = COMPARISON)
        push!(probes, (; id = "$T/$id", n, kind, op, check, comparison))
    end

    λ, a = T(0.5), T(3.5)
    small, near, large = T(0.75), T(2.45), T(4)
    f = PB.SCAD(λ, a)
    zero_level = PB.SCAD(zero(T), a)
    default_shape = ("default_shape", PB.SCAD(λ))
    default_models = T === Float32 ? (default_shape, ("typed_shape", PB.SCAD(λ, T(3.7)))) : (default_shape,)
    pa, px, vp, pb = PB.prox, PB.prox!, PB.value_and_prox!, PB.prox_block
    for n in SIZES
        d = data.dense[n]
        for (label, γ) in (("below", small), ("near", near), ("above", large))
            add("scalar/prox_block/$label/n$n", n, :prox, loop_case(pb, d.out, f, d.x, γ))
            add("dense/prox/$label/n$n", n, :prox, prox_case(px, d.out, f, d.x, γ, copy))
            add("dense/fused/$label/n$n", n, :fused, fused_case(vp, d.out, f, d.x, γ, copy, T))
        end
        add("dense/value/n$n", n, :value, value_case(f, d.x, T))
        add("dense/prox/zero/n$n", n, :prox, prox_case(px, d.out, zero_level, d.x, small, copy))
        add("dense/fused/zero/n$n", n, :fused, fused_case(vp, d.out, zero_level, d.x, large, copy, T))
        add("dense/value/zero/n$n", n, :value, value_case(zero_level, d.x, T))
        for (label, γ) in (("entry_below", d.steps), ("entry_mixed", d.mixed))
            add("dense/prox/$label/n$n", n, :prox, prox_case(px, d.out, f, d.x, γ, copy))
            add("dense/fused/$label/n$n", n, :fused, fused_case(vp, d.out, f, d.x, γ, copy, T))
        end
        for (label, model) in default_models
            id = "default_call/$label"
            add("$id/alloc_prox/omitted/n$n", n, :prox, allocating_prox_case(pa, model, d.x))
            add("$id/alloc_prox/typed_step/n$n", n, :prox, allocating_prox_case(pa, model, d.x, one(T)))
            add("$id/prox/omitted/n$n", n, :prox, prox_case(px, d.out, model, d.x, copy))
            add("$id/prox/typed_step/n$n", n, :prox, prox_case(px, d.out, model, d.x, one(T), copy))
        end
        if n == 1024
            # The reference role times current NormL1 here, for context only.
            nf = norm_module === PB ? f : norm_module.NormL1(λ)
            add("context/dense/prox/n$n", n, :prox,
                prox_case(norm_module.prox!, d.out, nf, d.x, small, copy); comparison = :norml1)
            add("context/dense/value/n$n", n, :value,
                value_case(nf, d.x, T); comparison = :norml1)
        end
    end

    for n in (1024, 65536)
        d = data.iterates[n]
        sx = PB.SparseIterate(d.sparse)
        bx = PB.BlockIterate(d.block, d.groups)
        mx = PB.SymmetricSparseIterate(T, d.p)
        for (r, c, v) in d.symmetric
            mx[r, c] = v
        end
        for (name, x, snapshot) in (("sparse", sx, Vector), ("block", bx, Vector), ("symmetric", mx, Matrix))
            out = copy(x)
            add("$name/value/n$n", n, :value, value_case(f, x, T))
            for (label, γ) in (("below", small), ("above", large))
                add("$name/prox/$label/n$n", n, :prox, prox_case(px, out, f, x, γ, snapshot))
                add("$name/fused/$label/n$n", n, :fused, fused_case(vp, out, f, x, γ, snapshot, T))
            end
            if name == "sparse" || name == "block"
                γ = data.dense[n].steps
                add("$name/prox/entry_below/n$n", n, :prox, prox_case(px, out, f, x, γ, snapshot))
            end
        end
    end
    probes
end

function build_suites()
    types = (Float32, Float64)
    data = Dict(T => make_data(T) for T in types)
    filter_re = Regex(get(ENV, "PB_FILTER", ""))
    candidate = CandidateSource.ProximalBase
    modules = (("reference", ReferenceSource.ProximalBase, candidate),
               ("candidate", candidate, candidate),
               ("control", ControlSource.ProximalBase, ControlSource.ProximalBase))
    suites = Dict(label => filter(p -> occursin(filter_re, p.id),
        vcat([make_probes(m, norm_module, T, data[T]) for T in types]...))
        for (label, m, norm_module) in modules)
    isempty(suites["reference"]) && error("PB_FILTER selected no cases")
    @assert getfield.(suites["reference"], :id) == getfield.(suites["candidate"], :id) ==
        getfield.(suites["control"], :id)
    suites
end

# The historical middle formula cancels quantities of order one, especially
# close to the curvature transition. Inputs are clear of the discontinuity,
# so tolerance applies to small arithmetic differences, not to branch changes.
function agrees_point(a, b, id)
    size(a) == size(b) || return false
    T = eltype(a)
    k = occursin("/near/", id) || occursin("/entry_mixed/", id) ? 192 : 24
    all(abs(u - v) <= k * eps(T) * max(one(T), abs(u), abs(v)) for (u, v) in zip(a, b))
end
function agrees_value(a, b, n)
    T = typeof(a)
    isapprox(a, b; rtol = max(32, min(n, 8192)) * eps(T), atol = 16eps(T))
end
function agrees(p, a, b)
    p.kind === :prox && return agrees_point(a, b, p.id)
    p.kind === :value && return agrees_value(a, b, p.n)
    agrees_value(a[1], b[1], p.n) && agrees_point(a[2], b[2], p.id)
end
function check_outputs(suites)
    compared = 0
    for (r, c, k) in zip(suites["reference"], suites["candidate"], suites["control"])
        out = c.check()
        isequal(out, k.check()) || error("A/A output mismatch for $(c.id)")
        if r.comparison === COMPARISON
            agrees(r, out, r.check()) || error("$(COMPARISON) output mismatch for $(r.id)")
            compared += 1
        else
            r.check()
        end
    end
    println("Differential checks passed for $compared $(COMPARISON) cases and $(length(suites["candidate"])) A/A cases.")
end

function trial(p, samples)
    op = p.op
    # Batch tiny calls to limit timer overhead. Measure the already warmed
    # operation with a stable input. Only alloc_prox cases allocate a result.
    evals = clamp(8192 ÷ p.n, 1, 512)
    b = @benchmarkable $op() evals = evals samples = samples seconds = 0.15 gctrial = false
    run(b)
end

function revision(path)
    sha = readchomp(`git -C $path rev-parse HEAD`)
    state = readchomp(`git -C $path status --porcelain --untracked-files=no`)
    sha, isempty(state)
end

function provenance(outdir)
    open(joinpath(outdir, "provenance.txt"), "w") do io
        println(io, "UTC: ", now(UTC), "\nJulia: ", VERSION, "\nCPU: ", Sys.CPU_NAME,
            "\nMachine: ", Sys.MACHINE, "\nJulia threads: ", Threads.nthreads(),
            "\nBLAS threads: ", BLAS.get_num_threads(), "\nBLAS: ", BLAS.get_config(),
            "\nBounds option: ", Base.JLOptions().check_bounds, "\nReversed source roles: ", REVERSED)
        println(io, "SLURM job: ", get(ENV, "SLURM_JOB_ID", "local"))
        if Sys.islinux()
            model = findfirst(l -> startswith(l, "model name"), readlines("/proc/cpuinfo"))
            model === nothing || println(io, readlines("/proc/cpuinfo")[model])
        end
        for (role, path) in ((string(COMPARISON), REFERENCE), ("candidate", CANDIDATE))
            sha, clean = revision(path)
            println(io, role, " SHA: ", sha, "\n", role, " clean: ", clean)
            !clean && get(ENV, "PB_ALLOW_DIRTY", "0") != "1" && error("$role worktree has tracked changes")
        end
        println(io, "harness SHA256: ", bytes2hex(sha256(read(@__FILE__))))
        for path in (Base.active_project(), joinpath(dirname(Base.active_project()), "Manifest.toml"))
            isfile(path) && println(io, basename(path), " SHA256: ", bytes2hex(sha256(read(path))))
        end
    end
end

function run_suite(suites, outdir, rounds, samples)
    mkpath(outdir)
    provenance(outdir)
    order = ("reference", "candidate", "control")
    rotations = (order, (order[2], order[3], order[1]), (order[3], order[1], order[2]))
    for label in (REVERSED ? reverse(order) : order), p in suites[label]
        trial(p, 8)
    end
    ids = eachindex(suites["candidate"])
    open(joinpath(outdir, "raw.csv"), "w") do raw
        open(joinpath(outdir, "samples.csv"), "w") do observations
            println(raw, "case,comparison,round,order,role,min_ns,median_ns,memory,allocs,samples")
            println(observations, "case,round,role,samples_ns")
            for round in 1:rounds
                rotation = rotations[mod1((round + 1) ÷ 2, 3)]
                labels = xor(iseven(round), REVERSED) ? reverse(rotation) : rotation
                for i in (iseven(round) ? reverse(ids) : ids)
                    for (ordinal, label) in enumerate(labels)
                        p = suites[label][i]
                        t = trial(p, samples)
                        m, mid = minimum(t), median(t)
                        println(raw, join((p.id, p.comparison, round, ordinal, label, m.time,
                            mid.time, m.memory, m.allocs, length(t.times)), ','))
                        println(observations, join((p.id, round, label, join(t.times, ' ')), ','))
                    end
                end
                flush(raw)
                flush(observations)
                println("Completed SCAD benchmark round $round/$rounds")
                flush(stdout)
            end
        end
    end
end

suites = build_suites()
check_outputs(suites)
if !CHECK_ONLY
    rounds = parse(Int, get(ENV, "PB_ROUNDS", "6"))
    samples = parse(Int, get(ENV, "PB_SAMPLES", "128"))
    run_suite(suites, abspath(ENV["PB_OUTPUT"]), rounds, samples)
end
