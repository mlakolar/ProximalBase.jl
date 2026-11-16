# Paired in-process timing of the value oracles of SqrtLeastSquaresLoss (issue #960): value_from_predictor and
# value_from_residual of the loss, raw and as Scaled(f, 1/√n), Float32 and Float64.
#
#   PB_ROOT=<root of the tree under test> PB_OUTPUT=<dir> PB_ROUNDS=15 \
#     julia --threads=1 --check-bounds=no --project=benchmark benchmark/sqrt_leastsquares.jl
#
# The root contains a src/ folder with ProximalBase.jl (default: the repository of this script). The tree is loaded by
# Base.include into a module, as the other harnesses do. Two kinds of cells:
#
# * n = 100, 10^3, 10^4, 10^5 (at most 2^17 entries, one vectorizing loop): the oracle is timed against a reference
#   loop written here, α √(Σ (zᵢ - Yᵢ)²) or α √(Σ rᵢ²) (no α for the raw loss), without recovery from overflow or
#   underflow. The loop is compiled from one source string in two modules (R1 and R2); R2 is the A/A control of R1.
# * n = 10^6 (above 2^17 entries, where the oracles and the least-squares oracles use the pairwise reduction): the
#   reference is the oracle value_from_predictor or value_from_residual of LeastSquaresLoss of the same tree on the
#   same data; the control is the same least-squares oracle of a second compiled copy of the tree.
#
# Each round times the three roles of a cell (reference, oracle, control) in alternating order, and the cells in
# alternating order; one timing is the median and the minimum over a number of batches of calls. After the rounds the
# script prints, per cell, the median over the rounds of the ratios oracle/reference of the medians and of the
# minima, the same for the control, the spread of the control's ratios (their range over the rounds, and its
# distance from one), the warm allocations of each side, and a verdict: PASS where the median ratio is at most 1.25
# (n = 100) or 1.10 (the other n), INCONCLUSIVE where it is above the limit by no more than the control's range or
# distance from one, and FAIL otherwise. The exit status is nonzero on a FAIL, on a result that differs from the
# reference by more than 8 eps(T) (64 eps(T) at n = 10^6), or on an allocation of the oracle on its ordinary path.
#
# PB_ROUNDS (default 15, at least 15 is the criterion), PB_OUTPUT (raw.csv and provenance.txt), PB_FILTER (a regular
# expression on the cell ids type/operation/variant/n), PB_CHECK_ONLY=1 (check values and allocations, no timing).
# Run it again in a fresh process before reading a ratio near the limit: the placement of code and of arrays moves
# timings by several percent.
using Random, Statistics, Dates, LinearAlgebra
Threads.nthreads() == 1 || error("one Julia thread required")
LinearAlgebra.BLAS.set_num_threads(1)
root = abspath(get(ENV, "PB_ROOT", joinpath(@__DIR__, "..")))
module Tree1 end
module Tree2 end
module R1 end
module R2 end

const LOOPS = raw"""
function norm_predictor_raw(Y, z)
    s = zero(eltype(Y))
    @inbounds @simd for i in eachindex(z, Y)
        s += abs2(z[i] - Y[i])
    end
    sqrt(s)
end
function norm_predictor_scaled(α, Y, z)
    s = zero(eltype(Y))
    @inbounds @simd for i in eachindex(z, Y)
        s += abs2(z[i] - Y[i])
    end
    α * sqrt(s)
end
function norm_residual_raw(r)
    s = zero(eltype(r))
    @inbounds @simd for i in eachindex(r)
        s += abs2(r[i])
    end
    sqrt(s)
end
function norm_residual_scaled(α, r)
    s = zero(eltype(r))
    @inbounds @simd for i in eachindex(r)
        s += abs2(r[i])
    end
    α * sqrt(s)
end
"""
for K in (R1, R2)
    Base.include_string(K, LOOPS, "sqrt_leastsquares_loops.jl")
end

const SMALL = (100, 1_000, 10_000, 100_000)
const LARGE = 1_000_000
const NS = (SMALL..., LARGE)
const TYPES = (Float32, Float64)
filter_re = Regex(get(ENV, "PB_FILTER", ""))
cells_needed(n) = any(T -> any(occursin(filter_re, "$T/$op/$v/n$n") for op in ("predictor", "residual") for v in ("raw", "scaled")), TYPES)
Base.include(Tree1, joinpath(root, "src", "ProximalBase.jl"))
cells_needed(LARGE) && Base.include(Tree2, joinpath(root, "src", "ProximalBase.jl"))
const PB1 = Tree1.ProximalBase
const PB2 = isdefined(Tree2, :ProximalBase) ? Tree2.ProximalBase : nothing

function makedata(::Type{T}, n) where {T}
    rng = Random.Xoshiro(960 + n + 3sizeof(T))
    Y = randn(rng, T, n)
    z = Y .+ T(0.5) .* randn(rng, T, n)
    (; n, Y, z, r = z .- Y, X = ones(T, n, 1))
end

# a closure over concretely typed arguments, behind a function barrier: its result is stored, so that the call is not discarded
bindval(fn, args...) = (box = [fn(args...)]; () -> (@inbounds box[1] = fn(args...); nothing))

struct Cell
    id::String
    T::DataType
    n::Int
    limit::Float64
    roles::Vector{Pair{String, Function}}        # reference, oracle, control
    check::Function                              # (reference value, oracle value)
end

function makecells()
    cells = Cell[]
    for T in TYPES, n in NS
        d = makedata(T, n)
        α = inv(sqrt(T(n)))
        limit = n == 100 ? 1.25 : 1.10
        sq = PB1.SqrtLeastSquaresLoss(d.Y, d.X)
        sqs = PB1.Scaled(sq, α)
        for op in ("predictor", "residual"), variant in ("raw", "scaled")
            id = "$T/$op/$variant/n$n"
            occursin(filter_re, id) || continue
            f = variant == "raw" ? sq : sqs
            oracle = op == "predictor" ? bindval(PB1.value_from_predictor, f, d.z) : bindval(PB1.value_from_residual, f, d.r)
            call = op == "predictor" ? (() -> PB1.value_from_predictor(f, d.z)) : (() -> PB1.value_from_residual(f, d.r))
            if n <= 100_000
                ref(K) = op == "predictor" ?
                    (variant == "raw" ? bindval(K.norm_predictor_raw, d.Y, d.z) : bindval(K.norm_predictor_scaled, α, d.Y, d.z)) :
                    (variant == "raw" ? bindval(K.norm_residual_raw, d.r) : bindval(K.norm_residual_scaled, α, d.r))
                rcall(K) = op == "predictor" ?
                    (variant == "raw" ? (() -> K.norm_predictor_raw(d.Y, d.z)) : (() -> K.norm_predictor_scaled(α, d.Y, d.z))) :
                    (variant == "raw" ? (() -> K.norm_residual_raw(d.r)) : (() -> K.norm_residual_scaled(α, d.r)))
                push!(cells, Cell(id, T, n, limit,
                                  ["reference" => ref(R1), "oracle" => oracle, "control" => ref(R2)],
                                  () -> (rcall(R1)(), call())))
            else
                lsf(PB) = (ls = PB.LeastSquaresLoss(d.Y, d.X); variant == "raw" ? ls : PB.Scaled(ls, α))
                f1, f2 = lsf(PB1), lsf(PB2)
                ls1 = op == "predictor" ? bindval(PB1.value_from_predictor, f1, d.z) : bindval(PB1.value_from_residual, f1, d.r)
                ls2 = op == "predictor" ? bindval(PB2.value_from_predictor, f2, d.z) : bindval(PB2.value_from_residual, f2, d.r)
                lscall = op == "predictor" ? (() -> PB1.value_from_predictor(f1, d.z)) : (() -> PB1.value_from_residual(f1, d.r))
                push!(cells, Cell(id, T, n, limit, ["reference" => ls1, "oracle" => oracle, "control" => ls2],
                                  () -> (lscall(), call())))
            end
        end
    end
    cells
end
cells = makecells()
isempty(cells) && error("PB_FILTER selects no cell")

# values: the oracle against the reference loop (n ≤ 10^5: within 8 eps(T)); against the least-squares oracle (n = 10^6:
# the square of the oracle value over twice the least-squares value is the factor α of the cell, 1 for a raw loss,
# within 64 eps(T))
function checkvalue(c)
    ref, v = c.check()
    T = c.T
    if c.n <= 100_000
        err = abs(big(v) - big(ref)) / (big(eps(T)) * abs(big(ref)))
        return Float64(err), err <= 8
    else
        # the least-squares value is α s / 2 (s / 2 for the raw loss) and the oracle value is α √s, so that
        # v² / (2 ref) is α for a scaled cell and 1 for the raw one
        q = big(v)^2 / (2 * big(ref))
        expected = occursin("/scaled/", c.id) ? big(inv(sqrt(T(c.n)))) : big(1)
        err = abs(q - expected) / (big(eps(T)) * expected)
        return Float64(err), err <= 64
    end
end

allocated_by(op) = (op(); op(); @allocated op())
failures = String[]
println("cells: ", length(cells), "; tree: ", root)
println("value check (error in eps(T) of the oracle against the reference) and warm allocations in bytes (reference/oracle/control):")
for c in cells
    err, ok = checkvalue(c)
    bytes = [allocated_by(last(r)) for r in c.roles]
    println("  ", rpad(c.id, 36), " error ", rpad(round(err; sigdigits = 3), 8), " ", join(lpad.(bytes, 6), "/"))
    ok || push!(failures, "$(c.id): the oracle differs from the reference by $err eps(T)")
    bytes[2] == 0 || push!(failures, "$(c.id): the oracle allocates $(bytes[2]) bytes on its ordinary path")
end
if !isempty(failures)
    foreach(f -> println("FAILED: ", f), failures)
    error("$(length(failures)) failures (listed above)")
end
get(ENV, "PB_CHECK_ONLY", "0") == "1" && (println("check passed"); exit())

# one timing of a role: the median and the minimum over batches of `evals` calls, each batch about 20 µs or longer
function calibrate(op)
    for _ in 1:10
        op()
    end
    t = minimum(begin t0 = time_ns(); op(); Float64(time_ns() - t0) end for _ in 1:20)
    evals = max(1, ceil(Int, 20_000 / t))
    samples = clamp(round(Int, 12e6 / (evals * t)), 30, 1500)
    evals, samples
end
function measure(op, evals, samples)
    ts = Vector{Float64}(undef, samples)
    for s in 1:samples
        t0 = time_ns()
        for _ in 1:evals
            op()
        end
        ts[s] = Float64(time_ns() - t0) / evals
    end
    median(ts), minimum(ts)
end

rounds = parse(Int, get(ENV, "PB_ROUNDS", "15"))
plans = [[calibrate(last(r)) for r in c.roles] for c in cells]
for (c, plan) in zip(cells, plans), (k, r) in enumerate(c.roles)          # warm each role at its batch length
    measure(last(r), plan[k][1], 20)
end
outdir = get(ENV, "PB_OUTPUT", "/tmp/pb960-bench")
mkpath(outdir)
open(joinpath(outdir, "provenance.txt"), "w") do io
    println(io, "UTC: ", now(UTC), "\nJulia: ", VERSION, "\nCPU: ", Sys.CPU_NAME, "\nThreads: ", Threads.nthreads())
    println(io, "BLAS threads: ", LinearAlgebra.BLAS.get_num_threads(), "\nBounds option: ", Base.JLOptions().check_bounds)
    println(io, "Filter: ", get(ENV, "PB_FILTER", ""), "\nRoot: ", root, "\nRounds: ", rounds)
end
times = Dict{Tuple{String, String}, Vector{Tuple{Float64, Float64}}}()     # (cell, role) => (median, min) per round
open(joinpath(outdir, "raw.csv"), "w") do io
    println(io, "case,round,order,role,median_ns,min_ns")
    idx = eachindex(cells)
    for round in 1:rounds
        for i in (isodd(round) ? idx : reverse(idx))
            c, plan = cells[i], plans[i]
            order = isodd(round) ? (1, 2, 3) : (3, 2, 1)
            for (pos, k) in enumerate(order)
                name, op = c.roles[k]
                m, mn = measure(op, plan[k][1], plan[k][2])
                push!(get!(times, (c.id, name), Tuple{Float64, Float64}[]), (m, mn))
                println(io, c.id, ',', round, ',', pos, ',', name, ',', m, ',', mn)
            end
        end
        flush(io)
        println("round ", round, "/", rounds); flush(stdout)
    end
end

verdicts = String[]
println("\ncell                                  median(oracle/ref)  min(oracle/ref)  median(control/ref)  control range  control |bias|  verdict")
for c in cells
    ref, orc, ctl = (times[(c.id, n)] for n in ("reference", "oracle", "control"))
    ratios(a, b, j) = [a[r][j] / b[r][j] for r in eachindex(a)]
    med, mn = median(ratios(orc, ref, 1)), median(ratios(orc, ref, 2))
    cm = ratios(ctl, ref, 1)
    cmed, range_, bias = median(cm), maximum(cm) - minimum(cm), abs(median(cm) - 1)
    noise = max(range_, bias)
    verdict = med <= c.limit ? "PASS" : (med - 1 <= noise ? "INCONCLUSIVE" : "FAIL")
    println(rpad(c.id, 36), lpad(round(med; digits = 3), 18), lpad(round(mn; digits = 3), 17), lpad(round(cmed; digits = 3), 21),
            lpad(round(range_; digits = 3), 15), lpad(round(bias; digits = 3), 16), "  ", verdict, " (limit ", c.limit, ")")
    push!(verdicts, verdict)
end
println("\nPASS: ", count(==("PASS"), verdicts), ", INCONCLUSIVE: ", count(==("INCONCLUSIVE"), verdicts), ", FAIL: ", count(==("FAIL"), verdicts))
exit(any(==("FAIL"), verdicts) ? 1 : 0)
