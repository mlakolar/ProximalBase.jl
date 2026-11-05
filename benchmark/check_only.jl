# Check-only driver: run every selected case's reference checks and warm-up without timing it.
#
# It builds the cases exactly as `run.jl` does (same groups, suite, repetition seed and
# measurement order) and, for every case, repeats what `measure!` does around the timed trial:
# `validate()`, three `prepare(); op()` rounds, the allocation probe, and, for a case with an
# automatic evaluation count, several samples of repeated evaluations followed by `validate()`
# again. No BenchmarkTools trial, calibration or profile runs, and no metrics are written.
#
# It includes `run.jl` unchanged, so the harness hash of a baseline does not change.
#
#   JULIA_NUM_THREADS=1 julia --project=benchmark benchmark/check_only.jl
#
# Settings: PB_SUITE, PB_GROUPS, PB_CASES, PB_REPETITION and PB_EXPECTED_PACKAGE_ROOT as in
# `run.jl`; PB_OUTPUT, when set, receives `check-rep<NN>-errors.tsv` (every failure with its stack
# trace, written as it occurs). It exits with status 1 when a case fails and prints the ID of
# every failing case. benchmark/check_only_test.jl tests `check!` and `check_cases!`, and, through
# benchmark/cases_test.jl, the sparse operands and reference checks of the `products` cases.
include("run.jl")

# A timed trial of a case with an automatic evaluation count calls `prepare()` once per sample
# and `op()` many times in a row; `measure!` validates afterwards. These counts stand in for it.
const CHECK_SAMPLES = 2
const CHECK_EVALS = 10

function check!(c::PerfCase)
    op, prep = c.operation, c.prepare
    c.validate()
    for _ in 1:3
        prep(); op()
    end
    allocated_call(op, prep); allocated_call(op, prep)
    for _ in 1:10
        allocated_call(op, prep)
    end
    if c.evals === nothing
        for _ in 1:CHECK_SAMPLES
            prep()
            for _ in 1:CHECK_EVALS
                op()
            end
        end
        c.validate()
    end
    nothing
end

"""
    check_cases!(cases, errors::IO) -> failed IDs

Run `check!` on every case in order. Each failure goes to `errors` at once as a row
`id  exception  stack` (after a header row) and is flushed, and a short line goes to stderr;
the remaining cases still run.
"""
function check_cases!(cases, errors::IO)
    line(errors, "id", "exception", "stack"); flush(errors)
    failed = String[]
    for (k, c) in enumerate(cases)
        try
            check!(c)
        catch err
            err isa InterruptException && rethrow()
            push!(failed, c.meta.id)
            line(errors, c.meta.id, sprint(showerror, err), sprint(Base.show_backtrace, catch_backtrace()))
            flush(errors)
            println(stderr, "CASE_FAILED ", c.meta.id, ": ", first(sprint(showerror, err), 300))
        end
        k % 250 == 0 && println("PROGRESS ", k, "/", length(cases), " failures=", length(failed))
    end
    failed
end

function check_main()
    package_root = realpath(dirname(dirname(pathof(ProximalBase))))
    expected_root = get(ENV, "PB_EXPECTED_PACKAGE_ROOT", "")
    if !isempty(expected_root)
        package_root == realpath(expected_root) ||
            error("Loaded ProximalBase from $package_root; expected $(realpath(expected_root))")
    end
    rep = parse(Int, get(ENV, "PB_REPETITION", "1"))
    1 <= rep <= 10 || error("PB_REPETITION must be between 1 and 10")
    suite = get(ENV, "PB_SUITE", "quick")
    groups = sort(split(get(ENV, "PB_GROUPS", join(CASE_GROUPS, ',')), ','))
    BLAS.set_num_threads(1)
    Threads.nthreads() == 1 || error("Run with JULIA_NUM_THREADS=1")
    started = time()
    Random.seed!(rep)
    cases = build_cases(rep; suite, groups)
    selection = get(ENV, "PB_CASES", "")
    isempty(selection) || filter!(c -> occursin(Regex(selection), c.meta.id), cases)
    isempty(cases) && error("PB_CASES selects no case")
    ids = sort([c.meta.id for c in cases])
    catalog = bytes2hex(sha1(join(ids, '\n')))
    auto = count(c -> c.evals === nothing, cases)
    println("CHECK_START suite=", suite, " groups=", join(groups, ','), " rep=", rep,
            " cases=", length(cases), " auto_evals=", auto, " catalog_sha1=", catalog,
            " source=", package_root)
    # The same order as a measured repetition, so state shared between cases is visited alike.
    shuffle!(scenario_rng(rep, "measurement_order"), cases)
    # Each failure is written and flushed as it occurs, so a timeout or a killed process keeps
    # the stack traces recorded so far.
    output = get(ENV, "PB_OUTPUT", "")
    isempty(output) || mkpath(output)
    errors_path = joinpath(output, "check-rep$(lpad(rep, 2, '0'))-errors.tsv")
    failed = isempty(output) ? check_cases!(cases, devnull) : open(io -> check_cases!(cases, io), errors_path, "w")
    @printf("CHECK_FINISHED cases=%d failures=%d elapsed=%.1f s\n", length(cases), length(failed), time() - started)
    if !isempty(failed)
        println(stderr, "Failing cases:")
        foreach(id -> println(stderr, "  ", id), sort(failed))
        isempty(output) || println(stderr, "Exceptions and stack traces: ", errors_path)
        exit(1)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    check_main()
end
