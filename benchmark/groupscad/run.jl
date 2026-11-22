# Paired benchmark of GroupSCAD (#1070) for one source tree per process, with GroupNormL2 of a reference tree as the
# cost reference and a second compilation of the tree under test as the A/A control.
#
#   PB_SOURCE=<root of the tree under test> PB_REFERENCE=<root of the reference tree> PB_OUTPUT=<dir> \
#     PB_ROUNDS=12 julia --startup-file=no --threads=1 --check-bounds=no --project=benchmark benchmark/groupscad/run.jl
#
# Roles: X, GroupSCAD of PB_SOURCE; XC, a second compilation of PB_SOURCE (A/A control); R, GroupNormL2 of PB_REFERENCE
# on the same groups, weights, level and step. R is a different function with a similar pass over the data, not the same
# data movement: for example, a group kept in place by GroupSCAD is not written, while soft shrinkage writes it. PB_SOURCE
# needs a src/ folder that defines GroupSCAD with the methods GroupSCAD(λ0, groups, w; a), f(x), prox!, value_and_prox! and
# prox_block!(out, f, u, k, γ) (a tree with this restoration does, and so does the frozen tree legacy/full,
# eff21b6d7d33c622313122f46c6ee71693c101ee); PB_REFERENCE needs GroupNormL2(λ0, groups, w) with the same methods. Run the
# candidate and the legacy tree in separate processes with the same PB_REFERENCE (the base of the pull request) and
# compare them with summarize.py, which divides the median ratios X/R of the two processes on the cases whose saved
# outputs agree. PB_REVERSE=1 sets the module load order R, XC, X instead of X, XC, R (run it in a fresh process with its
# own output directory); the order of the first verification and warmup calls, and of the timed roles, follows ORDERS
# in both cases.
#
# Cases: Float32 and Float64; 4096 entries in groups of 1, 4, 32, 512 or 4096 entries, contiguous (ranges) or permuted
# (index vectors); shape a = 3.7, level 0.75, unit weights; steps γ = (a - 1)/2 (below a - 1: convex middle piece),
# a - 1 (at a - 1: affine middle piece) and 3(a - 1)/2 (above a - 1: nonconvex middle piece); regimes zero (every group
# zeroed), soft (every group at the soft-thresholding point of the first piece), middle (every group on the middle piece:
# the interior root for the step below a - 1, and for the other two steps the midpoint of the middle piece, which they send
# to +0) and kept (every group kept). The target norm of each regime is interior to its regime (half the zero cutoff; between γλ and the end of the soft interval; between (1 + γ)λ and aλ for the step below
# a - 1 and between λ and aλ otherwise; twice the largest cutoff); the inputs are these targets along random directions,
# converted to T, which keeps each input away from the rounding of a threshold. This does not make two implementations
# take the same branches where their contracts differ. Each case has PB_INPUTS (default 8) inputs with the same target norms and
# independently drawn directions; the setup of each timed sample selects the next input of the pool (outside the timing),
# and the three roles read the same arrays.
#
# Operations: value, prox (into another array), prox_inplace (prox!(x, f, x, γ); the setup restores x), value_and_prox,
# block_sweep (prox_block! over every group into separate outputs) and block_sweep_inplace (out === u; the setup restores
# the inputs). Every sample is one call or one sweep (evals = 1). Each round times the roles of every case in an order that
# repeats every six rounds, the cases in reverse order in even rounds. PB_ROUNDS must be a positive multiple of six.
#
# Before timing, every one of the PB_INPUTS inputs of a case is run once per role (X, XC and R, each on the same input, in the
# order of ORDERS). XC must equal X (isequal), on every input; X is checked against the BigFloat reference, and R is not
# verified (it is timed as a cost reference). X must match the radial map at the group norm computed in 256-bit BigFloat (the
# scalar reference of test/scad_reference.jl), entrywise within 16 (m + 8) κ eps(T) max|u| (κ = (a - 1)/(a - 1 - γ) for
# γ < a - 1, else 1), and its value within 16 (m + K) eps(T) of the BigFloat value (for the fused value, relative to the
# larger of that value and floatmin(T)). An X/XC mismatch stops the script; a mismatch of X against the reference stops it
# too, or, with PB_STRICT=0 (for the legacy tree), is recorded as fail in verification.csv. The BigFloat reference of an input
# is computed once per case and shared by the operations; verification.csv has the number of inputs and the number that
# matched. The outputs of X on the first input are written to outputs/ and those on every input to outputs_all/ (the outputs
# of the inputs one after the other) for summarize.py. PB_CHECK_ONLY=1 stops after that. PB_FILTER selects cases by a regular
# expression on their ids.
#
# Timing: one BenchmarkTools run per role and round, with the reused benchmark of the case. In BenchmarkTools 1.8.0 (the
# version of the benchmark environment) `_run` estimates the remaining samples from the time of the first sample (setup
# included) against the seconds cap and calls sizehint! with that estimate; when the first sample alone took longer than
# the cap (a stall of the process, for example a garbage collection or a descheduled process on a shared node) the estimate
# is negative, and sizehint! throws an InexactError on Julia 1.10.13 (observed in the campaign of #1195, and reproduced with a
# setup that sleeps). On Julia 1.13.4 the same stall did not throw and ended the run with one sample. A run is repeated when
# it throws an InexactError from sizehint! (the frame is checked; any other exception is raised) or has fewer than three
# samples; PB_RETRIES is the number of attempts in all (default 3, at least 1), and PB_SAMPLES must be at least 3. A run
# that was repeated, and every cell-round whose attempts were exhausted, is recorded in failures.csv; an exhausted cell-round
# is left out of raw.csv and ratios.csv for all three roles.
#
# Output files: provenance.txt, verification.csv, raw.csv (per round and role, the median and minimum time of the trial,
# and the memory and allocations that BenchmarkTools reports with the median estimate, taken from the trial's minimum
# measurements), ratios.csv (per round X/R and XC/X), failures.csv and outputs/ and outputs_all/<case id with / replaced by _>.bin.
using BenchmarkTools, Dates, Random, SHA, Statistics

include(joinpath(@__DIR__, "..", "..", "test", "scad_reference.jl"))
using .SCADReferenceTest

Threads.nthreads() == 1 || error("one Julia thread required")
const SOURCE = abspath(ENV["PB_SOURCE"])
const REFERENCE = abspath(ENV["PB_REFERENCE"])
const REVERSE = get(ENV, "PB_REVERSE", "0") == "1"

module X end
module XC end
module R end
for (m, root) in (REVERSE ? ((R, REFERENCE), (XC, SOURCE), (X, SOURCE)) : ((X, SOURCE), (XC, SOURCE), (R, REFERENCE)))
    Base.include(m, joinpath(root, "src", "ProximalBase.jl"))
end

const TOTAL = 4096
const SIZES = (1, 4, 32, 512, 4096)
const STEPS = ((:below, 1 // 2), (:at, 1 // 1), (:above, 3 // 2))        # multiples of a - 1
const REGIMES = (:zero, :soft, :middle, :kept)
const OPS = (:value, :prox, :prox_inplace, :value_and_prox, :block_sweep, :block_sweep_inplace)
const ORDERS = (("X", "R", "XC"), ("XC", "R", "X"), ("R", "XC", "X"), ("X", "XC", "R"), ("XC", "X", "R"), ("R", "X", "XC"))
const SHAPE = 37 // 10
const LEVEL = 3 // 4

# The norm of every group of a regime, relative to the thresholds of the step, in the middle of the interval of norms that
# gives its decision: zero (half of γλ), soft (between γλ and (1 + γ)λ for γ < a - 1, and between γλ and the cutoff
# (a + 1 + γ)λ/2 of the tie with the kept point otherwise), middle (between (1 + γ)λ and aλ for γ < a - 1, and between λ
# and aλ otherwise, where the group goes to +0), or kept (twice the largest cutoff).
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
    x = copy(inputs[1])
    U = [[x[G] for G in groups] for x in inputs]
    (; groups, inputs, x, xi = copy(x), y = similar(x), U, outs = [similar(u) for u in U[1]],
       W = [copy(u) for u in U[1]], cur = Ref(1), λ0, a, γ, w)
end

# The probes of one role. A function of the typed data and of the package callables, so that the closures capture
# concrete values and call the methods directly; the setups select the next input outside the timing.
function probes(d, f, prox!::P, value_and_prox!::V, prox_block!::B) where {P, V, B}
    inputs, x, xi, y, U, outs, W, γ, cur = d.inputs, d.x, d.xi, d.y, d.U, d.outs, d.W, d.γ, d.cur
    val = Ref(zero(eltype(x)))
    n = length(inputs)
    next!() = (cur[] = mod1(cur[] + 1, n); copyto!(x, inputs[cur[]]); nothing)
    next_inplace!() = (cur[] = mod1(cur[] + 1, n); copyto!(xi, inputs[cur[]]); nothing)
    next_block!() = (cur[] = mod1(cur[] + 1, n); nothing)
    function next_block_inplace!()
        cur[] = mod1(cur[] + 1, n)
        Uc = U[cur[]]
        for k in eachindex(W)
            @inbounds copyto!(W[k], Uc[k])
        end
        nothing
    end
    function sweep()
        Uc = U[cur[]]
        for k in eachindex(Uc)
            @inbounds prox_block!(outs[k], f, Uc[k], k, γ)
        end
        nothing
    end
    function sweep_inplace()
        for k in eachindex(W)
            @inbounds prox_block!(W[k], f, W[k], k, γ)
        end
        nothing
    end
    (
        value = (op = () -> (val[] = f(x); nothing), setup = next!, result = () -> (val[],)),
        prox = (op = () -> (prox!(y, f, x, γ); nothing), setup = next!, result = () -> (copy(y),)),
        prox_inplace = (op = () -> (prox!(xi, f, xi, γ); nothing), setup = next_inplace!, result = () -> (copy(xi),)),
        value_and_prox = (op = () -> (val[] = value_and_prox!(y, f, x, γ); nothing), setup = next!,
                          result = () -> (val[], copy(y))),
        block_sweep = (op = sweep, setup = next_block!, result = () -> (reduce(vcat, outs),)),
        block_sweep_inplace = (op = sweep_inplace, setup = next_block_inplace!, result = () -> (reduce(vcat, W),)),
    )
end

group_probes(PB, d) = probes(d, PB.GroupSCAD(d.λ0, d.groups, d.w; a = d.a), PB.prox!, PB.value_and_prox!, PB.prox_block!)
reference_probes(PB, d) = probes(d, PB.GroupNormL2(d.λ0, d.groups, d.w), PB.prox!, PB.value_and_prox!, PB.prox_block!)

# The radial map at the norm of each group (the scalar SCAD reference of test/scad_reference.jl), and the value, in BigFloat
# at the working precision set by the caller (256 bits); the target group norms are placed away from the thresholds of
# their step, as the header states.
function bigprox(d, x)
    a, λ, γ = BigFloat(d.a), BigFloat(d.λ0), BigFloat(d.γ)
    y = BigFloat.(x)
    for G in d.groups
        u = BigFloat.(x[G])
        r = sqrt(sum(abs2, u))
        t = iszero(r) ? r : first(scad_reference(λ, a, r, γ))
        y[G] = iszero(r) ? u : (t / r) .* u
    end
    y
end
function bigvalue(d, x)
    a, λ = BigFloat(d.a), BigFloat(d.λ0)
    sum(d.groups) do G
        scad_penalty(λ, a, sqrt(sum(abs2, BigFloat.(x[G]))))
    end
end

# Whether the result of X matches the BigFloat reference within the bounds of the header.
function verify(op, res, d, κ, j, ref)
    T = eltype(d.x)
    m = length(d.groups[1])
    K = length(d.groups)
    x = d.inputs[j]
    if op === :value
        v = ref.value
        return abs(BigFloat(res[1]) - v) <= 16 * (m + K) * eps(T) * v
    end
    point = ref.point
    want = op in (:block_sweep, :block_sweep_inplace) ? reduce(vcat, [point[G] for G in d.groups]) : point
    got = op === :value_and_prox ? res[2] : res[1]
    bound = 16 * (m + 8) * κ * eps(T) * maximum(abs, x)
    all(i -> abs(BigFloat(got[i]) - want[i]) <= bound, eachindex(got)) || return false
    op === :value_and_prox || return true
    v = bigvalue(d, got)
    abs(BigFloat(res[1]) - v) <= 16 * (m + K) * eps(T) * max(v, floatmin(T))
end

runcheck(p) = (p.setup(); p.op(); p.result())

# BenchmarkTools' run, repeated when it throws an InexactError from sizehint! (see the header) or has fewer than MIN_SAMPLES
# samples; returns (trial, attempts), with trial = nothing when every attempt failed. Any other exception is raised.
const MIN_SAMPLES = 3
function timed_run(b, retries)
    for attempt in 1:retries
        try
            t = run(b)
            length(t.times) >= MIN_SAMPLES && return t, attempt
        catch e
            e isa InexactError && any(fr -> fr.func === :sizehint!, stacktrace(catch_backtrace())) || rethrow()
        end
    end
    nothing, retries
end

# the BigFloat reference of an input, computed once per case
reference_for(d, cache, j) = get!(cache, j) do
    x = d.inputs[j]
    (point = bigprox(d, x), value = bigvalue(d, x))
end

function prepare_timing(p, samples, seconds)
    op, setup = p.op, p.setup
    for _ in 1:20
        setup()
        op()
    end
    @benchmarkable $op() setup = ($setup()) evals = 1 samples = samples seconds = seconds gctrial = false gcsample = false
end

harness_state() = try
    readchomp(pipeline(Cmd(`git status --porcelain -- .`; dir = @__DIR__); stderr = devnull)) == "" ? "clean" : "modified"
catch
    "state unknown"
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

function write_provenance(dir, rounds, samples, seconds, filter_pattern, ninputs)
    sha(root) = try
        readchomp(pipeline(Cmd(`git rev-parse HEAD`; dir = root); stderr = devnull))
    catch
        "not a git checkout"
    end
    # the git revision where there is one, and always the sha256 of src/ (a git archive has no revision)
    identity(root) = string(sha(root), ", src/ sha256 ", tree_digest(joinpath(root, "src")))
    harness_digest() = bytes2hex(sha256(vcat(read(@__FILE__), read(joinpath(@__DIR__, "..", "..", "test", "scad_reference.jl")))))
    open(joinpath(dir, "provenance.txt"), "w") do io
        println(io, "UTC: ", now(UTC))
        println(io, "Julia: ", VERSION, ", Julia commit: ", Base.GIT_VERSION_INFO.commit)
        println(io, "CPU: ", first(Sys.cpu_info()).model, ", architecture: ", Sys.ARCH, ", Julia CPU target: ", Sys.CPU_NAME)
        println(io, "Threads: ", Threads.nthreads(), ", bounds: ", Base.JLOptions().check_bounds)
        println(io, "BenchmarkTools: ", Base.pkgversion(BenchmarkTools))
        println(io, "Source (X, XC): ", SOURCE, " at ", identity(SOURCE))
        println(io, "Reference (R): ", REFERENCE, " at ", identity(REFERENCE))
        println(io, "Load order: ", REVERSE ? "R, XC, X" : "X, XC, R")
        println(io, "Slurm job: ", get(ENV, "SLURM_JOB_ID", "none"), ", host: ", gethostname(), ", node list: ", get(ENV, "SLURM_JOB_NODELIST", "none"))
        println(io, "Harness: ", @__FILE__, " at ", sha(@__DIR__), " (", harness_state(), "), sha256 of run.jl and test/scad_reference.jl: ", harness_digest())
        println(io, "Julia options: ", Base.julia_cmd(), ", JULIA_NUM_THREADS=", get(ENV, "JULIA_NUM_THREADS", "unset"))
        println(io, "Input configuration: entries ", TOTAL, ", group sizes ", SIZES, ", steps (multiples of a - 1) ", STEPS, ", regimes ", REGIMES,
                ", shape a = ", SHAPE, ", level = ", LEVEL, ", unit weights, layouts range and perm, element types Float32 and Float64")
        println(io, "Input generation: MersenneTwister(1070 + 7m + 3(layout === :perm) + 11 inputs), random directions, ", ninputs,
                " inputs per case selected in turn by the setups; operations ", OPS)
        println(io, "Verification: every input of every case, BigFloat precision ", precision(BigFloat), ", strict ", get(ENV, "PB_STRICT", "1"),
                ", attempts per run (PB_RETRIES) ", get(ENV, "PB_RETRIES", "3"), ", minimum samples per run ", MIN_SAMPLES)
        println(io, "Rounds: ", rounds, ", maximum samples: ", samples, ", target cap (seconds): ", seconds,
                ", inputs per case: ", ninputs)
        println(io, "Filter: ", filter_pattern)
        println(io, "Timed role orders repeat every six rounds: ", ORDERS)
    end
end

function run_matrix(outdir)
    rounds = parse(Int, get(ENV, "PB_ROUNDS", "12"))
    samples = parse(Int, get(ENV, "PB_SAMPLES", "40"))
    seconds = parse(Float64, get(ENV, "PB_SECONDS", "0.025"))
    ninputs = parse(Int, get(ENV, "PB_INPUTS", "8"))
    rounds > 0 && rounds % 6 == 0 || error("PB_ROUNDS must be a positive multiple of six")
    retries = parse(Int, get(ENV, "PB_RETRIES", "3"))
    retries >= 1 || error("PB_RETRIES must be at least 1")
    samples >= MIN_SAMPLES && isfinite(seconds) && seconds > 0 && ninputs > 0 ||
        error("invalid sample, time or input count (PB_SAMPLES must be at least $MIN_SAMPLES)")
    filter_pattern = get(ENV, "PB_FILTER", "")
    selected = Regex(filter_pattern)
    check_only = get(ENV, "PB_CHECK_ONLY", "0") == "1"
    strict = get(ENV, "PB_STRICT", "1") == "1"
    mkpath(joinpath(outdir, "outputs")); mkpath(joinpath(outdir, "outputs_all"))
    write_provenance(outdir, rounds, samples, seconds, filter_pattern, ninputs)

    cells = NamedTuple[]
    open(joinpath(outdir, "verification.csv"), "w") do io
        println(io, "case,order,aa,x_vs_bigfloat,inputs,inputs_matching")
        for T in (Float32, Float64), m in SIZES, layout in (:range, :perm), (sname, step) in STEPS, regime in REGIMES
            prefix = "GroupSCAD/$T/m$m/$layout/$sname/$regime"
            any(op -> occursin(selected, "$prefix/$op"), OPS) || continue
            d = makedata(T, m, layout, step, regime, ninputs)
            suites = Dict("X" => group_probes(X.ProximalBase, d), "XC" => group_probes(XC.ProximalBase, d),
                          "R" => reference_probes(R.ProximalBase, d))
            κ = step < 1 ? 1 / (1 - step) : 1          # (a - 1)/(a - 1 - γ) for the step below a - 1
            ref_cache = Dict{Int, Any}()
            for op in OPS
                id = "$prefix/$op"
                occursin(selected, id) || continue
                order = ORDERS[mod1(length(cells) + 1, length(ORDERS))]
                aa_ok = true
                passes = 0
                first_out = nothing
                all_out = Any[]
                for j in 1:ninputs
                    results = Dict{String, Any}()
                    for role in order
                        d.cur[] = j - 1              # the setup of the probe selects the input j
                        results[role] = runcheck(getproperty(suites[role], op))
                    end
                    d.cur[] == j || error("input selection out of step for $id")
                    isequal(results["X"], results["XC"]) || (aa_ok = false; error("A/A output mismatch for $id, input $j"))
                    ok = verify(op, results["X"], d, κ, j, reference_for(d, ref_cache, j))
                    passes += ok
                    ok || strict && error("X does not match the reference for $id, input $j")
                    j == 1 && (first_out = results["X"])
                    push!(all_out, results["X"])
                end
                d.cur[] = 1
                write_outputs(path, out) = open(path, "w") do io
                    for res in out, r in res
                        write(io, r isa Number ? [r] : r)
                    end
                end
                write_outputs(joinpath(outdir, "outputs", replace(id, "/" => "_") * ".bin"), (first_out,))
                write_outputs(joinpath(outdir, "outputs_all", replace(id, "/" => "_") * ".bin"), all_out)
                println(io, id, ',', join(order, "/"), ',', aa_ok ? "pass" : "fail", ',', passes == ninputs ? "pass" : "fail", ',',
                        ninputs, ',', passes)
                timings = check_only ? nothing :
                    Dict(role => prepare_timing(getproperty(suites[role], op), samples, seconds) for role in order)
                push!(cells, (; id, timings))
            end
        end
    end
    isempty(cells) && error("PB_FILTER matched no benchmark cells")
    println("Output verification done for $(length(cells)) cells (see verification.csv)")
    check_only && return

    open(joinpath(outdir, "raw.csv"), "w") do raw
        open(joinpath(outdir, "ratios.csv"), "w") do ratios
            open(joinpath(outdir, "failures.csv"), "w") do failures
                println(raw, "case,round,order,role,median_ns,min_ns,memory,allocs")
                println(ratios, "case,round,x_to_reference,control_to_x")
                println(failures, "case,round,role,attempts,outcome")
                for round in 1:rounds
                    ord = ORDERS[mod1(round, length(ORDERS))]
                    for i in (isodd(round) ? eachindex(cells) : reverse(eachindex(cells)))
                        cell = cells[i]
                        medians = Dict{String, Float64}()
                        lines = String[]
                        failed = false
                        for (j, role) in enumerate(ord)
                            t, attempts = timed_run(cell.timings[role], retries)
                            (attempts > 1 || t === nothing) && println(failures, cell.id, ',', round, ',', role, ',', attempts, ',',
                                                                       t === nothing ? "exhausted" : "repeated")
                            if t === nothing
                                failed = true
                                continue
                            end
                            med = median(t)
                            medians[role] = med.time
                            push!(lines, string(cell.id, ',', round, ',', j, ',', role, ',', med.time, ',', minimum(t).time, ',',
                                                med.memory, ',', med.allocs))
                        end
                        failed && continue           # a round of the cell without all three roles is not recorded
                        foreach(l -> println(raw, l), lines)
                        println(ratios, cell.id, ',', round, ',', medians["X"] / medians["R"], ',', medians["XC"] / medians["X"])
                    end
                    flush(raw)
                    flush(ratios)
                    flush(failures)
                    println("round $round/$rounds")
                    flush(stdout)
                end
            end
        end
    end
end

setprecision(BigFloat, 256) do
    run_matrix(ENV["PB_OUTPUT"])
end
