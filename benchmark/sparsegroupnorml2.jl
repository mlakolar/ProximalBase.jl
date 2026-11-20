
# Paired sparse-group oracle benchmark, with a separately compiled A/A control.
#
#   PB_OUTPUT=/path/to/results PB_ROUNDS=12 julia --startup-file=no --threads=1 \
#     --check-bounds=no --project=benchmark benchmark/sparsegroupnorml2.jl
#
# Use one exclusive CPU node and one Julia thread. The three roles are X (package),
# Y (standalone mathematical reference, not a historical performance baseline)
# and XC (a second compilation of X). Run each Julia version in a fresh process
# with a different output directory. PB_CHECK_ONLY=1 verifies outputs without
# timing. PB_FILTER is a regular expression for selecting a subset of the cells.
# PB_ROUNDS must be a positive multiple of six. PB_SAMPLES and PB_SECONDS cap each
# trial (defaults 30 and 0.025). Every timed sample is one oracle call or sweep.
using BenchmarkTools, Dates, Random, Statistics

Threads.nthreads() == 1 || error("one Julia thread required")
const ROOT = normpath(joinpath(@__DIR__, ".."))

module Candidate end
module CandidateControl end
Base.include(Candidate, joinpath(ROOT, "src", "ProximalBase.jl"))
Base.include(CandidateControl, joinpath(ROOT, "src", "ProximalBase.jl"))

const TOTAL = 4096
const SIZES = (1, 4, 32, 512, 4096)
const KINDS = (:zero, :tied, :active, :cold_range, :cold_nan, :cold_inf)
const ORDERS = (
    ("X", "Y", "XC"), ("XC", "Y", "X"),
    ("Y", "XC", "X"), ("X", "XC", "Y"),
    ("XC", "X", "Y"), ("Y", "X", "XC"),
)

function groups_for(m, ::Val{:range})
    [(k-1)*m+1:k*m for k in 1:TOTAL÷m]
end
function groups_for(m, ::Val{:perm})
    order = randperm(MersenneTwister(1068 + m), TOTAL)
    [order[(k-1)*m+1:k*m] for k in 1:TOTAL÷m]
end

function makedata(::Type{T}, m, layout, kind) where {T}
    groups = groups_for(m, Val(layout))
    x = Vector{T}(undef, TOTAL)
    γ = λ1 = λ2 = one(T)
    if kind === :cold_range
        e = exponent(floatmax(T))
        magnitude = ldexp(one(T), e ÷ 2 + 3)
        λ1 = λ2 = inv(magnitude)
        γ = ldexp(one(T), e - 1)
        for G in groups, (j, i) in enumerate(G)
            x[i] = isodd(j) ? magnitude : -magnitude
        end
    else
        for G in groups, (j, i) in enumerate(G)
            a = (j == 1 ? T(2.5) : one(T) + T(mod(j, 5)) / 4)
            x[i] = isodd(j) ? a : -a
        end
        if kind === :zero || kind === :tied
            # Exactly one coordinate survives the soft step. Its magnitude is
            # below the group threshold for :zero and exactly on it for :tied.
            x .= T(0.5)
            for G in groups
                x[first(G)] = kind === :tied ? T(2) : T(1.5)
            end
        elseif kind === :cold_nan || kind === :cold_inf
            for G in groups
                x[first(G)] = kind === :cold_nan ? T(NaN) : T(-Inf)
            end
        end
    end
    U = [x[G] for G in groups]
    (; groups, x, xi = copy(x), y = similar(x), U, outs = [similar(u) for u in U],
       Uwork = [similar(u) for u in U], λ1, λ2, γ, w = ones(T, length(groups)))
end

# Typed callable reference for the timed comparison. It uses a two-pass
# max-scaled norm and the defining formulas on this benchmark's positive,
# finite levels and steps. Its array and compact APIs use the same buffers
# and iteration order as the package probes.
struct ReferencePenalty{T,G}
    λ1::T
    λ2::T
    groups::G
    w::Vector{T}
end

function reference_norm(x, G)
    m = zero(eltype(x))
    @inbounds for i in G
        m = max(m, abs(x[i]))
    end
    (iszero(m) || !isfinite(m)) && return m
    s = zero(m)
    @inbounds for i in G
        s += abs2(x[i] / m)
    end
    m * sqrt(s)
end

function (f::ReferencePenalty)(x)
    a = b = zero(eltype(x))
    for (k, G) in enumerate(f.groups)
        @inbounds for i in G
            a += abs(x[i])
        end
        b += f.w[k] * reference_norm(x, G)
    end
    f.λ1 * a + f.λ2 * b
end

function reference_shrink!(y, x, G, a, b)
    @inbounds for i in G
        t = x[i]
        y[i] = copysign(max(abs(t) - a, zero(t)), t)
    end
    n = reference_norm(y, G)
    if isnan(n) || n <= b
        z = isnan(n) ? n : zero(n)
        @inbounds for i in G
            y[i] = z
        end
    elseif isfinite(n)
        q = 1 - b / n
        @inbounds for i in G
            y[i] *= q
        end
    end
    y
end

function reference_prox!(y, f, x, γ)
    for (k, G) in enumerate(f.groups)
        reference_shrink!(y, x, G, γ * f.λ1, γ * f.λ2 * f.w[k])
    end
    y
end

function reference_fused!(y, f, x, γ)
    reference_prox!(y, f, x, γ)
    f(y)
end

function reference_block!(y, f, u, k, γ)
    reference_shrink!(y, u, eachindex(u), γ * f.λ1, γ * f.λ2 * f.w[k])
end

# All operation-only closures below capture concrete callables and data.
# The heterogeneous outer catalogue is traversed outside the timed calls.
function probes(d, f, prox!::P, value_and_prox!::V, prox_block!::B) where {P,V,B}
    x, xi, y, U, outs, Uwork, γ = d.x, d.xi, d.y, d.U, d.outs, d.Uwork, d.γ
    val = Ref(zero(eltype(x)))
    function array_inplace()
        prox!(xi, f, xi, γ)
        nothing
    end
    function compact()
        for k in eachindex(U)
            @inbounds prox_block!(outs[k], f, U[k], k, γ)
        end
        nothing
    end
    function compact_inplace()
        for k in eachindex(Uwork)
            @inbounds prox_block!(Uwork[k], f, Uwork[k], k, γ)
        end
        nothing
    end
    function restore_compact()
        for k in eachindex(Uwork)
            @inbounds copyto!(Uwork[k], U[k])
        end
        nothing
    end
    (
        value = (op = () -> (val[] = f(x); nothing), setup = nothing,
                 result = () -> val[]),
        prox = (op = () -> (prox!(y, f, x, γ); nothing), setup = nothing,
                result = () -> copy(y)),
        prox_inplace = (op = array_inplace, setup = () -> (copyto!(xi, x); nothing),
                        result = () -> copy(xi)),
        value_and_prox = (op = () -> (val[] = value_and_prox!(y, f, x, γ); nothing),
                          setup = nothing, result = () -> (val[], copy(y))),
        block_sweep = (op = compact, setup = nothing, result = () -> reduce(vcat, outs)),
        block_sweep_inplace = (op = compact_inplace, setup = restore_compact,
                               result = () -> reduce(vcat, Uwork)),
    )
end

function package_probes(PB, d)
    f = PB.SparseGroupNormL2(d.λ1, d.λ2, d.groups, d.w)
    probes(d, f, PB.prox!, PB.value_and_prox!, PB.prox_block!)
end

# The timed reference and the package are checked independently against a
# BigFloat evaluation. The latter computes its norm from a direct sum of
# squares, with explicit NaN/Inf branches for the exceptional input cells.
function expected(d)
    x = BigFloat.(d.x)
    y = copy(x)
    a = BigFloat(d.λ1)
    b = BigFloat(d.λ2)
    γ = BigFloat(d.γ)
    for (k, G) in enumerate(d.groups)
        s = [copysign(max(abs(x[i]) - γ * a, zero(a)), x[i]) for i in G]
        c = γ * b * BigFloat(d.w[k])
        if any(isnan, s)
            y[G] .= BigFloat(NaN)
        elseif any(isinf, s)
            y[G] = s
        else
            n = sqrt(sum(abs2, s))
            y[G] = n <= c ? fill(zero(a), length(s)) : (1 - c / n) .* s
        end
    end
    (; point = y, compact = reduce(vcat, [y[G] for G in d.groups]),
       value = exact_value(d, x))
end

function exact_value(d, x)
    a = BigFloat(d.λ1)
    b = BigFloat(d.λ2)
    l1 = sum(abs, x)
    l2 = sum(BigFloat(d.w[k]) * sqrt(sum(i -> abs2(x[i]), G))
             for (k, G) in enumerate(d.groups))
    a * l1 + b * l2
end

function runcheck(p)
    p.setup === nothing || p.setup()
    p.op()
    p.result()
end

function verify_scalar(actual::T, ref::BigFloat, factor, id) where {T}
    rounded = T(ref)
    isequal(actual, rounded) && return
    if isfinite(rounded) && isfinite(actual)
        error_bound = factor * eps(T) * abs(ref) + 2 * BigFloat(nextfloat(zero(T)))
        abs(BigFloat(actual) - ref) <= error_bound && return
    end
    error("value mismatch for $id, actual=$actual reference=$rounded")
end

function verify_vector(actual, ref, factor, id)
    T = eltype(actual)
    scale = maximum(abs(z) for z in ref if isfinite(z); init = BigFloat(0))
    bound = factor * eps(T) * scale + 2 * BigFloat(nextfloat(zero(T)))
    for i in eachindex(actual, ref)
        rounded = T(ref[i])
        isequal(actual[i], rounded) && continue
        isfinite(actual[i]) && isfinite(rounded) && abs(BigFloat(actual[i]) - ref[i]) <= bound && continue
        error("point mismatch for $id at $i, actual=$(actual[i]) reference=$rounded")
    end
end

function verify(actual, op, ref, d, factor, id)
    if op === :value
        verify_scalar(actual, ref.value, factor, id)
    elseif op === :value_and_prox
        val, point = actual
        verify_vector(point, ref.point, factor, id)
        verify_scalar(val, exact_value(d, BigFloat.(point)), factor, id)
    elseif op === :block_sweep || op === :block_sweep_inplace
        verify_vector(actual, ref.compact, factor, id)
    else
        verify_vector(actual, ref.point, factor, id)
    end
end

function prepare_timing(p, samples, seconds)
    op = p.op
    if p.setup === nothing
        for _ in 1:20
            op()
        end
        return @benchmarkable $op() evals=1 samples=samples seconds=seconds
    end
    setup = p.setup
    for _ in 1:20
        setup()
        op()
    end
    @benchmarkable $op() setup=($setup()) evals=1 samples=samples seconds=seconds
end

function write_provenance(dir, rounds, samples, seconds, filter_pattern)
    sha = readchomp(Cmd(`git rev-parse HEAD`; dir=ROOT))
    cpu = first(Sys.cpu_info()).model
    open(joinpath(dir, "provenance.txt"), "w") do io
        println(io, "UTC: ", now(UTC))
        println(io, "Julia: ", VERSION, ", Julia commit: ", Base.GIT_VERSION_INFO.commit)
        println(io, "CPU: ", cpu, ", architecture: ", Sys.ARCH, ", Julia CPU target: ", Sys.CPU_NAME)
        println(io, "Threads: ", Threads.nthreads(), ", bounds: ", Base.JLOptions().check_bounds)
        println(io, "BenchmarkTools: ", Base.pkgversion(BenchmarkTools))
        println(io, "X and XC source SHA: ", sha, ", Y: reference in the harness at that SHA")
        println(io, "Slurm job: ", get(ENV, "SLURM_JOB_ID", "none"))
        println(io, "Rounds: ", rounds, ", maximum samples: ", samples, ", target cap (seconds): ", seconds)
        println(io, "Filter: ", filter_pattern)
        println(io, "Candidate module loaded before control; per-cell compile order in verification.csv")
        println(io, "Measured role orders repeat every six rounds: ", ORDERS)
    end
end

function run_matrix(outdir)
    rounds = parse(Int, get(ENV, "PB_ROUNDS", "12"))
    samples = parse(Int, get(ENV, "PB_SAMPLES", "30"))
    seconds = parse(Float64, get(ENV, "PB_SECONDS", "0.025"))
    rounds > 0 && rounds % 6 == 0 || error("PB_ROUNDS must be a positive multiple of six")
    samples > 0 && isfinite(seconds) && seconds > 0 || error("invalid sample or time limit")
    filter_pattern = get(ENV, "PB_FILTER", "")
    selected = Regex(filter_pattern)
    check_only = get(ENV, "PB_CHECK_ONLY", "0") == "1"
    mkpath(outdir)
    write_provenance(outdir, rounds, samples, seconds, filter_pattern)

    cells = NamedTuple[]
    open(joinpath(outdir, "verification.csv"), "w") do io
        println(io, "case,compile_order,aa,package_vs_bigfloat,reference_vs_bigfloat")
        for T in (Float32, Float64), m in SIZES, layout in (:range, :perm), kind in KINDS
            d = makedata(T, m, layout, kind)
            ref_penalty = ReferencePenalty(d.λ1, d.λ2, d.groups, d.w)
            suites = Dict("X" => package_probes(Candidate.ProximalBase, d),
                          "Y" => probes(d, ref_penalty, reference_prox!, reference_fused!, reference_block!),
                          "XC" => package_probes(CandidateControl.ProximalBase, d))
            ref = expected(d)
            factor = 8 * (m + length(d.groups))
            for op in keys(suites["X"])
                id = "SparseGroupNormL2/$T/m$m/$layout/$kind/$op"
                occursin(selected, id) || continue
                order = ORDERS[mod1(length(cells) + 1, length(ORDERS))]
                actual = Dict(role => runcheck(getproperty(suites[role], op)) for role in order)
                isequal(actual["X"], actual["XC"]) || error("A/A output mismatch for $id")
                verify(actual["X"], op, ref, d, factor, "$id/X")
                verify(actual["Y"], op, ref, d, factor, "$id/Y")
                println(io, id, ',', join(order, "/"), ",pass,pass,pass")
                timings = check_only ? nothing :
                    Dict(role => prepare_timing(getproperty(suites[role], op), samples, seconds) for role in order)
                push!(cells, (; id, timings))
            end
        end
    end
    isempty(cells) && error("PB_FILTER matched no benchmark cells")
    println("Independent output verification passed for $(length(cells)) cells")
    check_only && return

    raw_path = joinpath(outdir, "raw.csv")
    ratios_path = joinpath(outdir, "ratios.csv")
    open(raw_path, "w") do raw
        open(ratios_path, "w") do ratios
            println(raw, "case,round,order,role,median_ns,min_ns,memory,allocs")
            println(ratios, "case,round,package_to_reference,control_to_package")
            for round in 1:rounds
                ord = ORDERS[mod1(round, length(ORDERS))]
                indices = isodd(round) ? eachindex(cells) : reverse(eachindex(cells))
                for i in indices
                    cell = cells[i]
                    medians = Dict{String,Float64}()
                    for (j, role) in enumerate(ord)
                        t = run(cell.timings[role])
                        m = median(t)
                        medians[role] = m.time
                        println(raw, cell.id, ',', round, ',', j, ',', role, ',',
                                m.time, ',', minimum(t).time, ',', m.memory, ',', m.allocs)
                    end
                    println(ratios, cell.id, ',', round, ',', medians["X"] / medians["Y"],
                            ',', medians["XC"] / medians["X"])
                end
                flush(raw)
                flush(ratios)
                println("round $round/$rounds"); flush(stdout)
            end
        end
    end
end

setprecision(BigFloat, 256) do
    run_matrix(ENV["PB_OUTPUT"])
end
