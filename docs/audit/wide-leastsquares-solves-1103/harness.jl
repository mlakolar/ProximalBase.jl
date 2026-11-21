# Paired timing harness of the wide least-squares study (#1103). One process loads the package source twice (roles S
# and S', the A/A control) and the Cholesky prototypes twice (C_B, C_D, C_W and their copies C_B', C_D', C_W'), reads
# the inputs written by generate.jl and times the roles against each other. The measurement groups are
#   first_use       construction, factorization and first prox on a fresh object (one sample per role and round, with
#                   the input rotating by round), and the seconds of the setup steps of the primary roles (first_use_part)
#   warm            the warmed prox, minimum over batches of calls, at each timing step of the case and for two kinds of
#                   inputs (random, ADMM iterates)
#   penalty         the step changes on every call, along a schedule that alternates between two steps (alt2) or cycles
#                   through the registered steps (newstep) (S: the prox only; C: refactorization from the cached XX',
#                   then the prox)
#   alloc           warmed bytes of the prox and of the step change, and the reachable bytes of each object
#   admm_reference  S-ADMM to stationarity 1e-12·max(1, ‖X'y‖∞) on each lasso instance (the reference optimum), to be
#                   generated once on the pinned Julia version
#   admm_untimed    for each primary role, the status of Boyd's rule and of the stationarity target
#                   1e-8·max(1, ‖X'y‖∞) (reached, capped after PB_ADMM_CAP iterations, or error), the iteration counts
#                   and the trajectory
#   admm            timed ADMM runs of those iteration counts (the A/A copies use the counts of their family), with the
#                   objective and the stationarity at the end of each run
#   outputs         the prox outputs of the primary roles from an untimed call, for accuracy.jl, with the #1078 controls
#
# Pairing. Within a (case, cell) all roles run batches of the same number of calls along the same sequence of steps and
# inputs: a batch is a complete cycle (a multiple of the number of inputs and, for the penalty groups, of the length of
# the step schedule) and restarts at the first input and the first step of the cycle; in the penalty group every
# object enters a batch at the last step of the schedule. Scratch buffers and the state of the native libraries are not
# reset between batches or roles. All roles take the minimum over the same number of batches. The roles are timed in a
# fixed order within a round and the order is reversed in even rounds (PB_ORDER selects the order of odd rounds).
# The number of calls per batch comes from a calibration of the primary roles: for warm, once per case on a complete
# input cycle at γ = 1 and random inputs, shared by all steps and both kinds of inputs of the case; for penalty, once
# per schedule. It is the smallest multiple of the cycle with about 20 ms (warm, and at least 30 calls) or about 50 ms
# (penalty) of calls for the slowest calibrated role, at most 3000 calls, and one cycle with one batch when that role
# needs 0.1 s or more per call. Each role is calibrated under the same guard as the samples: a role whose calibration
# throws gets an `error` row for the cell and a fresh object, and the other roles continue. The rows `batch_plan`
# record the plan, the cap and the slowest calibrated call.
#
#   PB_INPUTS=<inputs dir> PB_OUTPUT=<output dir> PB_SOURCE_SHA=<sha> PB_HARNESS_SHA=<sha> \
#   [PB_ORDER=S_first|C_first] [PB_ROUNDS=21] [PB_PROC=1] \
#   [PB_GROUP=first_use,warm,penalty,alloc,admm] [PB_FILTER=<regex on case ids>] [PB_BLAS_THREADS=1] \
#   [PB_BATCHES=3] [PB_ADMM_REPEATS=5] [PB_ADMM_CAP=20000] [PB_REFERENCE_CAP=100000] \
#   [PB_REFERENCE=<dir of the reference optimum and the iteration counts>] [PB_SOURCE=<root of the package tree>] \
#   julia --threads=1 --check-bounds=no --project=. docs/audit/wide-leastsquares-solves-1103/harness.jl
#
# PB_SOURCE_SHA and PB_HARNESS_SHA (7 to 40 lowercase hex digits) are required outside PB_CHECK_ONLY and are written to
# the header of each file the harness writes. When PB_INPUTS is set, the harness stops before the warm-up if a file that
# the requested groups would write exists already (the timing file of this process, the outputs index, failures index
# and binaries, the reference file and its z binary, the untimed file); a rerun uses a new empty PB_OUTPUT (and
# PB_REFERENCE) or removes those files first. PB_CHECK_ONLY=1 first checks the prototypes against S on a small
# full-rank case (the stationarity residual of each in BigFloat, the warmed allocations, the finiteness error, the
# construction path against the breakdown path), then runs the warm-up and, when PB_INPUTS is set, each group of
# PB_GROUP (default: all) on the smallest case of each element type (a group whose filter selects no case is skipped)
# with the defaults of 2 rounds and 2 ADMM repeats, which PB_ROUNDS and PB_ADMM_REPEATS override, and the output in
# PB_OUTPUT or a temporary directory. A timing group writes one tab-separated file per process,
# timings-<julia>-<order>-<proc>-<groups>.tsv, with a header of `# key: value` lines, one row per (case, measurement,
# param, round, position, role) and the closing line `# complete: yes`. Times are seconds per call (the minimum over
# the batches of the round, per call, for warm and penalty); bytes and allocations are per call. Some rows hold
# context instead (batch_plan, penalty_pairs, loadavg_1m, workingset, admm_iters, admm_reached). In the timing groups a
# failure of one sample, of one role's setup or calibration, or of a whole case (for example a refused untimed file
# in the admm group) is recorded as a row `error` and the harness continues; with PB_CHECK_ONLY those failures stop
# the process. The outputs group records failures per step and input in outputs-failures.tsv in both modes.
include(joinpath(@__DIR__, "common.jl"))

Threads.nthreads() == 1 || error("one Julia thread is required (--threads=1)")
const CHECK_ONLY = env_flag("PB_CHECK_ONLY")
const ORDER = env_string("PB_ORDER", "S_first")
const ROUNDS = env_int("PB_ROUNDS", CHECK_ONLY ? 2 : 21)
const BATCHES = env_int("PB_BATCHES", CHECK_ONLY ? 2 : 3)
const ADMM_REPEATS = env_int("PB_ADMM_REPEATS", CHECK_ONLY ? 2 : 5)
const ADMM_CAP = env_int("PB_ADMM_CAP", 20000)
const REFERENCE_CAP = env_int("PB_REFERENCE_CAP", 100000)
const PROC = env_string("PB_PROC", "1")
const BLAS_THREADS = env_int("PB_BLAS_THREADS", 1)
const SOURCE_SHA = env_sha("PB_SOURCE_SHA"; required = !CHECK_ONLY)
const HARNESS_SHA = env_sha("PB_HARNESS_SHA"; required = !CHECK_ONLY)
const REFERENCE_JULIA = v"1.13.4"       # the Julia version that generates the reference optima
const ADMM_RHOS = (0.1, 1.0, 10.0)
const ADMM_LEVELS = ("boyd", "tight")
const ADMM_TIGHT = 1e-8
const REFERENCE_TOL = 1e-12
const BATCH_CALL_CAP = 3000
const TIMINGS_FORMAT = "pb1103-timings-3"
const UNTIMED_FORMAT = "pb1103-admm-untimed-3"
const ALL_GROUPS = ["outputs", "first_use", "warm", "penalty", "alloc", "admm_reference", "admm_untimed", "admm"]
const TIMED_GROUPS = ["first_use", "warm", "penalty", "alloc", "admm"]
const GROUPS = String.(split(env_string("PB_GROUP", CHECK_ONLY ? join(ALL_GROUPS, ",") : join(TIMED_GROUPS, ",")), ','))
all(g -> g in ALL_GROUPS, GROUPS) || error("PB_GROUP takes a comma-separated subset of $(join(ALL_GROUPS, ", "))")
BLAS.set_num_threads(BLAS_THREADS)

###########################################################################################################
# Timing primitives
###########################################################################################################

struct Sample
    time::Float64
    gc::Float64
    bytes::Int
    allocs::Int
end

function timed_sample(f::F) where {F}
    r = @timed f()
    Sample(r.time, r.gctime, Int(r.bytes), Int(Base.gc_alloc_count(r.gcstats)))
end

# bytes allocated by a warmed call (a function barrier keeps the closure concrete)
warmed_bytes(op::F) where {F} = (op(); op(); @allocated op())

# m prox calls cycling through the inputs vs, starting at vs[1]; m is a multiple of length(vs), so that each batch
# is a set of complete cycles
struct WarmBatch{P,O,B,Vs,G}
    px::P
    obj::O
    β::B
    vs::Vs
    γ::G
    m::Int
    function WarmBatch(px::P, obj::O, β::B, vs::Vs, γ::G, m::Int) where {P,O,B,Vs,G}
        m > 0 && m % length(vs) == 0 || error("a warm batch holds complete cycles of the inputs")
        new{P,O,B,Vs,G}(px, obj, β, vs, γ, m)
    end
end
function (b::WarmBatch)()
    K = length(b.vs)
    @inbounds for i in 1:b.m
        b.px(b.β, b.obj, b.vs[mod1(i, K)], b.γ)
    end
    nothing
end

# the step and the input index of call i of a penalty batch (both cycled from the first)
@inline schedule_call(gs, K::Int, i::Int) = (gs[mod1(i, length(gs))], mod1(i, K))
# the (step, input index) pairs that one complete cycle of a penalty batch executes, from schedule_call
executed_pairs(gs, K::Int) = [schedule_call(gs, K, i) for i in 1:lcm(length(gs), K)]

# m calls, each with the next step of the schedule gs (cycled) followed by the prox on the next input, starting at
# gs[1] and vs[1] from an object at the last step of the schedule (schedule_call). m is a multiple of the schedule
# length and of the number of inputs, so each batch ends at the last step again; the schedule has no equal consecutive
# steps (also from its last step to its first), so that every call changes the step.
struct PenaltyBatch{P,R,O,B,Vs,Gs}
    px::P
    refactor::R
    obj::O
    β::B
    vs::Vs
    gs::Gs
    m::Int
    function PenaltyBatch(px::P, refactor::R, obj::O, β::B, vs::Vs, gs::Gs, m::Int) where {P,R,O,B,Vs,Gs}
        L, K = length(gs), length(vs)
        m > 0 && m % L == 0 && m % K == 0 || error("a penalty batch holds complete cycles of the steps and the inputs")
        L >= 2 && all(gs[i] != gs[mod1(i + 1, L)] for i in 1:L) || error("the step schedule repeats a step")
        new{P,R,O,B,Vs,Gs}(px, refactor, obj, β, vs, gs, m)
    end
end
function (b::PenaltyBatch)()
    K = length(b.vs)
    @inbounds for i in 1:b.m
        γ, k = schedule_call(b.gs, K, i)
        b.refactor(b.obj, γ)
        b.px(b.β, b.obj, b.vs[k], γ)
    end
    nothing
end

# the registered step schedules of the penalty groups (steps of the registered grid, so that each has accuracy
# evidence): alternating between two steps, and cycling through the registered steps; registered_schedules holds the
# registered Float64 values, penalty_schedules the steps converted to T
registered_schedules(::Type{T}) where {T} =
    (("alt2", [1.0, 1e3]), ("newstep", T === Float64 ? [1e-3, 1e3, 1.0, 1e6] : [1e-3, 1e3, 1.0]))
penalty_schedules(::Type{T}) where {T} = Tuple((name, T.(g)) for (name, g) in registered_schedules(T))

# the first use of a role: construction and the first prox
struct FirstUse{R,A,B,C,D,G}
    role::R
    X::A
    y::B
    β::C
    v::D
    γ::G
end
function (f::FirstUse)()
    obj = f.role.ctor(f.X, f.y, f.γ)
    f.role.px(f.β, obj, f.v, f.γ)
    nothing
end
first_use_sample(role, X, y, β, v, γ) = (GC.gc(); timed_sample(FirstUse(role, X, y, β, v, γ)))

# Seconds of the setup steps of S, reconstructed from the steps of the cached SVD of LeastSquaresLoss: the copy with the
# finiteness check (copy_check), svd, the materialized V (V) and s² with U'y scaled by s (UtY_s2). The loss
# constructor, the allocation of its work vector and the lazy bookkeeping are not included. The sink keeps the results
# alive.
const SINK = Ref(0.0)
function s_breakdown(X::Matrix{T}, y::Vector{T}) where {T}
    t0 = time_ns()
    A = Matrix{T}(X)
    all(isfinite, A) || error("nonfinite design")
    t1 = time_ns()
    F = svd(A)
    t2 = time_ns()
    V = Matrix{T}(F.V)
    t3 = time_ns()
    s2 = F.S .^ 2
    sUtY = Vector{T}(transpose(F.U) * y)
    sUtY .*= F.S
    t4 = time_ns()
    SINK[] += Float64(V[1]) + Float64(sUtY[1]) + Float64(s2[1])
    (; copy_check = (t1 - t0) * 1e-9, svd = (t2 - t1) * 1e-9, V = (t3 - t2) * 1e-9, UtY_s2 = (t4 - t3) * 1e-9)
end
function breakdown_sample(role::Role{C}, X, y, γ) where {C<:SCtor}
    GC.gc()
    s_breakdown(X, y)
end
function breakdown_sample(role::Role{C}, X, y, γ) where {C<:PCtor}
    L = C.parameters[1]
    GC.gc()
    getfield(parentmodule(L), :breakdown)(L, X, y, γ)[2]
end

# The calls per batch and the number of batches for a role set whose slowest calibrated role needs t_max seconds per
# call: the smallest multiple of `cycle` calls with at least min_calls calls and about `target` seconds of calls, at
# most BATCH_CALL_CAP calls (rounded down to a multiple of the cycle); one cycle and one batch when t_max is at least
# 0.1 s.
function plan_batches(t_max, cycle::Int; min_calls::Int, target::Float64)
    t_max >= 0.1 && return cycle, 1
    cap = cycle * max(1, BATCH_CALL_CAP ÷ cycle)
    calls = max(min_calls, t_max > 0 ? ceil(Int, target / t_max) : min_calls)
    clamp(cycle * cld(calls, cycle), cycle, cap), BATCHES
end

# the minimum over nb batches of calls
function best_batch(make, nb::Int)
    best = timed_sample(make())
    for _ in 2:nb
        s = timed_sample(make())
        s.time < best.time && (best = s)
    end
    best
end

round_indices(n, r) = isodd(r) ? (1:n) : (n:-1:1)

uint_of(::Type{Float64}) = UInt64
uint_of(::Type{Float32}) = UInt32
# equal as bit patterns (NaNs of different payloads differ)
bits_equal(a::AbstractArray{T}, b::AbstractArray{T}) where {T<:AbstractFloat} =
    size(a) == size(b) && reinterpret(uint_of(T), a) == reinterpret(uint_of(T), b)

###########################################################################################################
# Output of the rows
###########################################################################################################

mutable struct Writer
    io::IO
    rows::Int
end
function emit(w::Writer, case, measurement, param, round, position, role, value; gc = 0.0, bytes = 0.0, allocs = 0.0)
    @printf(w.io, "%s\t%s\t%s\t%d\t%d\t%s\t%.10g\t%.6g\t%.6g\t%.6g\n", case, measurement, param, round, position, role,
            Float64(value), Float64(gc), Float64(bytes), Float64(allocs))
    w.rows += 1
end

oneline(s) = replace(string(s), r"[\t\r\n]+" => " ")
errmsg(e) = oneline(first(split(sprint(showerror, e), '\n')))

# f() or, when it throws, `nothing` after an `error` row for the cell (the measurement, param and message in the
# param field); PB_CHECK_ONLY rethrows
function guarded(f, w::Writer, case, measurement, param, round, position, role)
    try
        return f()
    catch e
        CHECK_ONLY && rethrow()
        emit(w, case, "error", string(measurement, "|", param, "|", errmsg(e)), round, position, role, NaN)
        return nothing
    end
end

# the load average of the first minute, or nothing when /proc/loadavg cannot be read
function loadavg1()
    try
        parse(Float64, split(read("/proc/loadavg", String))[1])
    catch
        nothing
    end
end
loadavg() = try join(split(read("/proc/loadavg", String))[1:3], " ") catch; "unavailable" end
cpu_model() = try Sys.cpu_info()[1].model catch; "unknown" end
function cpu_vendor()
    try
        for line in eachline("/proc/cpuinfo")
            startswith(line, "vendor_id") && return strip(split(line, ':'; limit = 2)[2])
        end
    catch
    end
    "unknown"
end
function l3_cache()
    try
        for i in 0:8
            d = "/sys/devices/system/cpu/cpu0/cache/index$i"
            isfile(joinpath(d, "level")) || continue
            strip(read(joinpath(d, "level"), String)) == "3" && return strip(read(joinpath(d, "size"), String))
        end
    catch
    end
    "unavailable"
end
# the configuration string of the OpenBLAS library behind libblastrampoline (its version and target)
function openblas_config()
    try
        for lib in BLAS.get_config().loaded_libs
            h = Libdl.dlopen(lib.libname)
            for sym in (:openblas_get_config64_, :openblas_get_config)
                ptr = Libdl.dlsym(h, sym; throw_error = false)
                ptr === nothing || return strip(unsafe_string(ccall(ptr, Cstring, ())))
            end
        end
    catch
    end
    "unavailable"
end

function provenance_header(io, kind, inputs_sha; extra = ())
    for (k, v) in [("format", kind), ("julia", string(VERSION)), ("order", ORDER), ("proc", PROC),
                   ("source_sha", SOURCE_SHA), ("harness_sha", HARNESS_SHA), ("inputs_index_sha256", inputs_sha), extra...,
                   ("started_utc", Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ"))]
        println(io, "# ", k, ": ", v)
    end
end

function write_header(io, groups, inputs_sha)
    provenance_header(io, TIMINGS_FORMAT, inputs_sha;
                      extra = [("groups", join(groups, ",")), ("rounds", ROUNDS), ("batches", BATCHES),
                               ("admm_repeats", ADMM_REPEATS), ("admm_cap", ADMM_CAP), ("threads", Threads.nthreads()),
                               ("blas_threads", BLAS.get_num_threads()), ("blas_config", oneline(BLAS.get_config())),
                               ("openblas", openblas_config()), ("cpu_vendor", cpu_vendor()), ("cpu_model", cpu_model()),
                               ("cpu_name", Sys.CPU_NAME), ("cpu_threads", Sys.CPU_THREADS), ("l3_cache", l3_cache()),
                               ("check_bounds", Base.JLOptions().check_bounds), ("hostname", gethostname()),
                               ("loadavg_start", loadavg())])
    println(io, join(["case", "measurement", "param", "round", "position", "role", "value", "gc_s", "bytes", "allocs"], '\t'))
end

###########################################################################################################
# Cases
###########################################################################################################

# the arrays of a case; the four random inputs and the four ADMM iterates of a main case are each four different
# vectors, which the batches cycle through
function case_data(m::Manifest, c::Case)
    X, y, V = read_input(m, c, "X"), read_input(m, c, "y"), read_input(m, c, "V")
    vs = [V[:, k] for k in 1:size(V, 2)]
    if c.kind === :main
        length(vs) == 8 || error("a main case holds 8 proximal inputs")
        for grp in (1:4, 5:8), i in grp, j in grp
            i < j && isequal(vs[i], vs[j]) && error("the inputs $i and $j of $(c.id) are equal")
        end
    end
    (; X, y, vs, rand = vs[1:min(4, length(vs))], admm = length(vs) >= 8 ? vs[5:8] : vs, β = zeros(c.T, c.p))
end

# the object of a role of a case at the step γ, factorized and called once
function make_object(role, d, γ)
    obj = role.ctor(d.X, d.y, γ)
    role.prepare(obj)
    role.px(d.β, obj, d.vs[1], γ)
    obj
end
# the objects of the roles; nothing for a role whose setup failed
function make_objects(roles, d, γ, w, c, group)
    Any[guarded(() -> make_object(r, d, γ), w, c.id, group, "setup", 0, i, r.name) for (i, r) in enumerate(roles)]
end
# a fresh object after a failure of the role (nothing again if the construction fails too)
function rebuild!(objs, idx, role, d, γ, w, c, group)
    objs[idx] = guarded(() -> make_object(role, d, γ), w, c.id, group, "rebuild", 0, idx, role.name)
    nothing
end

# The largest over the primary roles of the seconds per call of a complete cycle of ncalls calls (the best of two
# timings). Each role runs under guarded: a role whose calibration throws gets an `error` row for the cell (the
# measurement `group` and the param `param`) and a fresh object at the step γ (rebuild!), and its time is left out.
# Roles without an object are skipped.
function calibrate!(make_cycle, roles, objs, ncalls, w, c, group, param, d, γ)
    tmax = 0.0
    for (idx, r) in enumerate(roles)
        (r.control || objs[idx] === nothing) && continue
        t = guarded(w, c.id, group, param, 0, idx, r.name) do
            op = make_cycle(r, objs[idx])
            min(timed_sample(op).time, timed_sample(op).time) / ncalls
        end
        if t === nothing
            rebuild!(objs, idx, r, d, γ, w, c, group)
        else
            tmax = max(tmax, t)
        end
    end
    tmax
end
# the batch_plan param: the plan, the cap and the slowest calibrated time per call written losslessly (repr)
plan_text(group, m, nb, tmax) = "$group|calls=$m|batches=$nb|cap=$BATCH_CALL_CAP|tmax=$(repr(Float64(tmax)))"

function emit_load(w::Writer, c::Case, group)
    la = loadavg1()
    la === nothing || emit(w, c.id, "loadavg_1m", group, 0, 0, "-", la)
end
# the load average and the reachable bytes of the objects of the roles (the working set to compare with the L3 size),
# keyed by the group in the param
function emit_context(w::Writer, c::Case, roles, objs, group)
    emit_load(w, c, group)
    for (i, (r, o)) in enumerate(zip(roles, objs))
        o === nothing || emit(w, c.id, "workingset", group, 0, i, r.name, Base.summarysize(o))
    end
end

# the cases of a group: the timed main cases for the prox timing groups, the Gaussian Float64 lasso instances for the
# ADMM groups and the main cases with the controls for outputs, all selected by PB_FILTER; with PB_CHECK_ONLY the
# smallest case of each element type (the first two controls for outputs)
function select_for(m::Manifest, group::AbstractString)
    cs = if group in ("first_use", "warm", "penalty", "alloc")
        select_cases(m; kind = :main, timed = true)
    elseif group == "outputs"
        select_cases(m; kind = :main)
    else
        select_cases(m; kind = :main, lasso = true, T = Float64)
    end
    sort!(cs; by = c -> (c.n * c.p, c.id))
    if CHECK_ONLY
        cs = [first(filter(c -> c.T === T, cs)) for T in (Float64, Float32) if any(c -> c.T === T, cs)]
    end
    if group == "outputs"
        ctl = select_cases(m; kind = :control)
        CHECK_ONLY && (ctl = ctl[1:min(2, length(ctl))])
        cs = vcat(cs, ctl)
    end
    cs
end

###########################################################################################################
# Groups
###########################################################################################################

function run_first_use(w::Writer, roles, c::Case, d)
    γ = one(c.T)
    prim = [r for r in roles if !r.control]
    K = length(d.rand)
    emit_load(w, c, "first_use")
    for round in 1:ROUNDS
        v = d.rand[mod1(round, K)]
        for (pos, idx) in enumerate(round_indices(length(roles), round))
            role = roles[idx]
            s = guarded(() -> first_use_sample(role, d.X, d.y, d.β, v, γ), w, c.id, "first_use", "g=1", round, pos, role.name)
            s === nothing && continue
            emit(w, c.id, "first_use", "g=1", round, pos, role.name, s.time; gc = s.gc, bytes = s.bytes, allocs = s.allocs)
        end
        for (pos, idx) in enumerate(round_indices(length(prim), round))
            role = prim[idx]
            parts = guarded(() -> breakdown_sample(role, d.X, d.y, γ), w, c.id, "first_use_part", "g=1", round, pos, role.name)
            parts === nothing && continue
            for (k, val) in pairs(parts)
                emit(w, c.id, "first_use_part", string(k), round, pos, role.name, val)
            end
        end
    end
end

function run_warm(w::Writer, roles, c::Case, d)
    T = c.T
    K = length(d.rand)
    objs = make_objects(roles, d, one(T), w, c, "warm")
    emit_context(w, c, roles, objs, "warm")
    tmax = calibrate!((r, o) -> WarmBatch(r.px, o, d.β, d.rand, one(T), K), roles, objs, K, w, c, "warm", "calibration|g=1",
                      d, one(T))
    m, nb = plan_batches(tmax, K; min_calls = 30, target = 0.02)
    emit(w, c.id, "batch_plan", plan_text("warm", m, nb, tmax), 0, 0, "-", tmax)
    for g in c.gammas
        γ = T(g)
        # bring every object to the step; a role without an object (or whose refactor failed) gets a fresh one
        for (idx, role) in enumerate(roles)
            if objs[idx] === nothing
                rebuild!(objs, idx, role, d, γ, w, c, "warm")
                continue
            end
            ok = guarded(() -> (role.refactor(objs[idx], γ); true), w, c.id, "warm", "refactor|g=$(repr(g))", 0, idx, role.name)
            ok === nothing && rebuild!(objs, idx, role, d, γ, w, c, "warm")
        end
        for (vkind, vs) in (("rand", d.rand), ("admm", d.admm))
            param = "g=$(repr(g))/$vkind"
            for round in 1:ROUNDS
                for (pos, idx) in enumerate(round_indices(length(roles), round))
                    role, obj = roles[idx], objs[idx]
                    if obj === nothing
                        emit(w, c.id, "error", "warm|$param|no object", round, pos, role.name, NaN)
                        continue
                    end
                    s = guarded(() -> best_batch(() -> WarmBatch(role.px, obj, d.β, vs, γ, m), nb), w, c.id, "warm", param, round,
                                pos, role.name)
                    if s === nothing
                        rebuild!(objs, idx, role, d, γ, w, c, "warm")
                        continue
                    end
                    emit(w, c.id, "warm", param, round, pos, role.name, s.time / m; gc = s.gc / m, bytes = s.bytes / m,
                         allocs = s.allocs / m)
                end
            end
        end
    end
end

function run_penalty(w::Writer, roles, c::Case, d)
    T = c.T
    K = length(d.rand)
    objs = make_objects(roles, d, one(T), w, c, "penalty")
    emit_context(w, c, roles, objs, "penalty")
    for (pname, g64) in registered_schedules(T)
        gs = T.(g64)
        cycle = lcm(length(gs), K)
        last_step = gs[end]
        # every role starts each batch from the last step of the schedule
        for (idx, role) in enumerate(roles)
            if objs[idx] === nothing
                rebuild!(objs, idx, role, d, last_step, w, c, "penalty")
                continue
            end
            ok = guarded(() -> (role.refactor(objs[idx], last_step); true), w, c.id, "penalty", "reset|$pname", 0, idx, role.name)
            ok === nothing && rebuild!(objs, idx, role, d, last_step, w, c, "penalty")
        end
        emit(w, c.id, "penalty_pairs", "$pname|" * join(("$(repr(g))@$k" for (g, k) in executed_pairs(g64, K)), ","), 0, 0,
             "-", NaN)
        tmax = calibrate!((r, o) -> PenaltyBatch(r.px, r.refactor, o, d.β, d.rand, gs, cycle), roles, objs, cycle, w, c,
                          "penalty", "calibration|$pname", d, last_step)
        m, nb = plan_batches(tmax, cycle; min_calls = 1, target = 0.05)
        emit(w, c.id, "batch_plan", plan_text("penalty:$pname", m, nb, tmax), 0, 0, "-", tmax)
        for round in 1:ROUNDS
            for (pos, idx) in enumerate(round_indices(length(roles), round))
                role, obj = roles[idx], objs[idx]
                if obj === nothing
                    emit(w, c.id, "error", "penalty|$pname|no object", round, pos, role.name, NaN)
                    continue
                end
                s = guarded(() -> best_batch(() -> PenaltyBatch(role.px, role.refactor, obj, d.β, d.rand, gs, m), nb), w, c.id,
                            "penalty", pname, round, pos, role.name)
                if s === nothing
                    rebuild!(objs, idx, role, d, last_step, w, c, "penalty")
                    continue
                end
                emit(w, c.id, "penalty", pname, round, pos, role.name, s.time / m; gc = s.gc / m, bytes = s.bytes / m,
                     allocs = s.allocs / m)
            end
        end
    end
end

function run_alloc(w::Writer, roles, c::Case, d)
    T = c.T
    γ = one(T)
    K = length(d.rand)
    emit_load(w, c, "alloc")
    gs = penalty_schedules(T)[1][2]
    cycle = lcm(length(gs), K)
    for (pos, role) in enumerate(roles)
        r = guarded(w, c.id, "alloc", "-", 0, pos, role.name) do
            obj = make_object(role, d, γ)
            op = WarmBatch(role.px, obj, d.β, d.rand, γ, K)
            b, a = warmed_bytes(op), timed_sample(op).allocs
            role.refactor(obj, gs[end])
            op2 = PenaltyBatch(role.px, role.refactor, obj, d.β, d.rand, gs, cycle)
            b2, a2 = warmed_bytes(op2), timed_sample(op2).allocs
            (b, a, b2, a2, Base.summarysize(obj))
        end
        r === nothing && continue
        emit(w, c.id, "alloc", "prox", 1, pos, role.name, NaN; bytes = r[1] / K, allocs = r[2] / K)
        emit(w, c.id, "alloc", "refactor_prox", 1, pos, role.name, NaN; bytes = r[3] / cycle, allocs = r[4] / cycle)
        emit(w, c.id, "workingset", "alloc", 1, pos, role.name, r[5])
    end
end

# The outputs of one role for every step in acc_gammas and every input, as a p × (inputs · steps) array (column
# (gi - 1)·K + k for step gi and input k), and the failures (gi, k, message). A failed refactor or prox is recorded
# for that step and input, and the object is rebuilt before the next attempt; columns that failed stay NaN.
function compute_outputs(role, c::Case, d)
    T = c.T
    γs = T.(c.acc_gammas)
    K = length(d.vs)
    B = fill(T(NaN), c.p, K * length(γs))
    fails = Tuple{Int,Int,String}[]
    obj = nothing
    for (gi, γ) in enumerate(γs)
        try
            if obj === nothing
                obj = role.ctor(d.X, d.y, γ)
                role.prepare(obj)
            else
                role.refactor(obj, γ)
            end
        catch e
            obj = nothing
            for k in 1:K
                push!(fails, (gi, k, errmsg(e)))
            end
            continue
        end
        for k in 1:K
            try
                if obj === nothing
                    obj = role.ctor(d.X, d.y, γ)
                    role.prepare(obj)
                end
                role.px(d.β, obj, d.vs[k], γ)
                B[:, (gi - 1) * K + k] .= d.β
            catch e
                obj = nothing
                push!(fails, (gi, k, errmsg(e)))
            end
        end
    end
    B, fails
end

# the binary file of the outputs of a role for a case
output_file(c::Case, role::AbstractString) = "out-" * case_file(c.id) * "-" * replace(role, "'" => "p") * (c.T === Float64 ? ".f64" : ".f32")

function run_outputs(roles, m::Manifest, cases, dir::String)
    mkpath(dir)
    prim = [r for r in roles if !r.control]
    copies = Dict(r.family => r for r in roles if r.control)
    open(joinpath(dir, "outputs-index.tsv"), "w") do io
        open(joinpath(dir, "outputs-failures.tsv"), "w") do fio
            provenance_header(io, "pb1103-outputs-2", m.sha256)
            provenance_header(fio, "pb1103-outputs-failures-1", m.sha256)
            println(io, join(["case", "role", "file", "eltype", "shape", "sha256", "nfail", "copy_identical", "acc_gammas", "nv"], '\t'))
            println(fio, join(["case", "role", "gamma_index", "vindex", "message"], '\t'))
            for c in cases
                d = case_data(m, c)
                K = length(d.vs)
                for role in prim
                    B, fails = compute_outputs(role, c, d)
                    B2, fails2 = compute_outputs(copies[role.family], c, d)
                    identical = bits_equal(B, B2) && Set((f[1], f[2]) for f in fails) == Set((f[1], f[2]) for f in fails2)
                    file = output_file(c, role.name)
                    open(joinpath(dir, file), "w") do f
                        write(f, B)
                    end
                    println(io, join([c.id, role.name, file, c.T === Float64 ? "Float64" : "Float32", string(c.p, "x", size(B, 2)),
                                      file_sha256(joinpath(dir, file)), length(fails), identical, join(repr.(c.acc_gammas), ","),
                                      K], '\t'))
                    for (gi, k, msg) in fails
                        println(fio, join([c.id, role.name, gi, k, msg], '\t'))
                    end
                    flush(io); flush(fio)
                end
                println("outputs written for ", c.id)
            end
            println(io, "# complete: yes")
            println(fio, "# complete: yes")
        end
    end
end

###########################################################################################################
# ADMM
###########################################################################################################

# the reference file of a lasso instance: named by the instance and the first 12 hex digits of the digests of X and y
function reference_path(dir, c::Case)
    x, y = c.arrays["X"].sha256, c.arrays["y"].sha256
    joinpath(dir, "reference-" * case_file(c.id) * "-" * x[1:12] * "-" * y[1:12] * ".tsv")
end

function write_kv(path, pairs)
    open(path, "w") do io
        for (k, v) in pairs
            println(io, k, '\t', v)
        end
        println(io, "complete\tyes")
    end
end

# One pinned reference per instance: S-ADMM at ρ = 1 to the stationarity 1e-12·max(1, ‖X'y‖∞), at most
# PB_REFERENCE_CAP iterations, generated on the pinned Julia version. The file records the provenance, the digests
# of X and y, whether the run converged and the objective; summarize.py refuses a campaign whose reference did not
# converge.
function run_admm_reference(m::Manifest, cases, roles, dir::String)
    CHECK_ONLY || VERSION == REFERENCE_JULIA || error("reference optima are generated on Julia $REFERENCE_JULIA (this is $VERSION)")
    mkpath(dir)
    S = first(r for r in roles if r.name == "S")
    for c in cases
        d = case_data(m, c)
        λ = c.T(0.1 * c.lambda_max)
        scale = max(1.0, c.lambda_max)
        t0 = time()
        r = admm_reference(S, d.X, d.y, λ, 1.0; tol = REFERENCE_TOL, scale = scale, cap = REFERENCE_CAP)
        path = reference_path(dir, c)
        zfile = replace(basename(path), ".tsv" => "-z.f64")
        write(joinpath(dir, zfile), Float64.(r.z))
        write_kv(path, [("format", "pb1103-reference-1"), ("case", c.id), ("julia", VERSION), ("source_sha", SOURCE_SHA),
                        ("harness_sha", HARNESS_SHA), ("inputs_index_sha256", m.sha256), ("x_sha256", c.arrays["X"].sha256),
                        ("y_sha256", c.arrays["y"].sha256), ("rho", 1.0), ("tol", REFERENCE_TOL), ("scale", scale),
                        ("cap", REFERENCE_CAP), ("iterations", r.iterations), ("converged", r.converged),
                        ("stationarity", r.stationarity), ("objective", repr(r.objective)), ("lambda", Float64(λ)),
                        ("lambda_max", c.lambda_max), ("z_file", zfile), ("z_sha256", file_sha256(joinpath(dir, zfile)))])
        println("reference for ", c.id, ": ", r.iterations, " iterations, converged ", r.converged, ", stationarity ",
                r.stationarity, ", objective ", r.objective, " (", round(time() - t0; digits = 1), " s)")
    end
end

untimed_path(dir, c::Case) = joinpath(dir, "admm-untimed-$(VERSION)-$(case_file(c.id)).tsv")

# the columns of the untimed file: record is trajectory (level and status are -), level (one row per role, penalty and
# level, with status reached or capped) or error (status error, k -, the message in note); the measures are Boyd's
# r, s, eps_pri and eps_dual, the stationarity and the objective at iteration k
const UNTIMED_COLUMNS = ["case", "rho", "role", "record", "level", "status", "k", "cap", "r_norm", "s_norm", "eps_pri",
                         "eps_dual", "stationarity", "objective", "note"]
const UNTIMED_MEASURES = ["r_norm", "s_norm", "eps_pri", "eps_dual", "stationarity", "objective"]

function run_admm_untimed(m::Manifest, cases, roles, dir::String)
    mkpath(dir)
    prim = [r for r in roles if !r.control]
    for c in cases
        d = case_data(m, c)
        λ = c.T(0.1 * c.lambda_max)
        scale = max(1.0, c.lambda_max)
        open(untimed_path(dir, c), "w") do io
            provenance_header(io, UNTIMED_FORMAT, m.sha256;
                              extra = [("case", c.id), ("x_sha256", c.arrays["X"].sha256), ("y_sha256", c.arrays["y"].sha256),
                                       ("admm_cap", ADMM_CAP), ("tight_tol", ADMM_TIGHT), ("scale", scale)])
            println(io, join(UNTIMED_COLUMNS, '\t'))
            for ρ in ADMM_RHOS, role in prim
                t0 = time()
                try
                    r = admm_untimed(role, d.X, d.y, λ, ρ; tight_tol = ADMM_TIGHT, scale = scale, cap = ADMM_CAP)
                    for row in r.rows
                        @printf(io, "%s\t%s\t%s\ttrajectory\t-\t-\t%d\t%d\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.12g\t\n", c.id, ρ,
                                role.name, Int(row[1]), ADMM_CAP, row[2:7]...)
                    end
                    for (lvl, x) in (("boyd", r.boyd), ("tight", r.tight))
                        @printf(io, "%s\t%s\t%s\tlevel\t%s\t%s\t%d\t%d\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.12g\t\n", c.id, ρ,
                                role.name, lvl, x.status, x.k, ADMM_CAP, x.measures...)
                    end
                    println("admm_untimed ", c.id, " rho=", ρ, " ", role.name, ": boyd ", r.boyd.status, " k ", r.boyd.k,
                            "; tight ", r.tight.status, " k ", r.tight.k, " (", round(time() - t0; digits = 1), " s)")
                catch e
                    CHECK_ONLY && rethrow()
                    println(io, join([c.id, ρ, role.name, "error", "-", "error", "-", ADMM_CAP, "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", errmsg(e)], '\t'))
                end
                flush(io)
            end
            println(io, "# complete: yes")
        end
    end
end

# The untimed counts of an instance for this process: (rho, role, level) => (reached, k), and the identity (file name
# and SHA-256) of the file they come from. The file is read once and hashed (before decoding), and it is accepted only
# when
#   - its format, Julia version, source, harness, manifest digest, X and y digests, case (header and every row), cap,
#     tight target, scale max(1, λmax) of the manifest and columns are those of this process, and no header key repeats;
#   - every row has a registered penalty and a primary role, and finite nonnegative measures;
#   - every (rho, role, level) has one level row with the status reached (1 ≤ k ≤ cap) or capped (k = cap), and its
#     measures agree with the status: for boyd, r ≤ eps_pri and s ≤ eps_dual when reached and one of them at least its
#     tolerance when capped; for tight, the stationarity at most tight_tol·scale when reached and above it when capped
#     (each with a relative slack of UNTIMED_SLACK for the 10-digit output);
#   - the trajectory rows of each (rho, role) are the checkpoints up to the last iteration run (the larger k when both
#     levels are reached, the cap otherwise), that last iteration and the iterations k of the reached levels, without
#     repeats, and each
#     level row repeats the measures of the trajectory row at its k;
#   - no run failed (an error row).
# Otherwise it throws.
const UNTIMED_SLACK = 1e-9
function load_kstar(dir::String, m::Manifest, c::Case, prim)
    path = untimed_path(dir, c)
    name = basename(path)
    isfile(path) || error("$name: missing untimed file for $(c.id) (run the group admm_untimed first)")
    bytes = read(path)
    sha = bytes2hex(sha256(bytes))
    header, cols, rows = parse_tsv(String(copy(bytes)), name)
    refuse(msg) = error("$name is refused: $msg")
    function expect(key, want)
        got = get(header, key, "<absent>")
        got == string(want) || refuse("$key is $got, expected $want")
    end
    expect("format", UNTIMED_FORMAT)
    expect("julia", VERSION)
    expect("source_sha", SOURCE_SHA)
    expect("harness_sha", HARNESS_SHA)
    expect("inputs_index_sha256", m.sha256)
    expect("case", c.id)
    expect("x_sha256", c.arrays["X"].sha256)
    expect("y_sha256", c.arrays["y"].sha256)
    expect("admm_cap", ADMM_CAP)
    tryparse(Float64, get(header, "tight_tol", "")) == ADMM_TIGHT || refuse("tight_tol is $(get(header, "tight_tol", "<absent>")), expected $ADMM_TIGHT")
    scale = max(1.0, c.lambda_max)
    tryparse(Float64, get(header, "scale", "")) == scale || refuse("scale is $(get(header, "scale", "<absent>")), expected $scale")
    cols == UNTIMED_COLUMNS || refuse("unexpected columns $(join(cols, ','))")
    target = ADMM_TIGHT * scale
    names = Set(r.name for r in prim)
    out = Dict{Tuple{Float64,String,String},NamedTuple{(:reached, :k),Tuple{Bool,Int}}}()
    levelmeas = Dict{Tuple{Float64,String,String},Vector{String}}()
    traj = Dict{Tuple{Float64,String},Dict{Int,Vector{String}}}()
    for (i, r) in enumerate(rows)
        r["case"] == c.id || refuse("row $i is of the case $(r["case"])")
        ρ = tryparse(Float64, r["rho"])
        ρ !== nothing && ρ in ADMM_RHOS || refuse("row $i has the penalty $(r["rho"])")
        r["role"] in names || refuse("row $i has the role $(r["role"])")
        rec = r["record"]
        rec == "error" && refuse("the run of $(r["role"]) at rho=$(r["rho"]) failed ($(r["note"])); rerun the group admm_untimed")
        rec in ("level", "trajectory") || refuse("row $i has the record $rec")
        tryparse(Int, r["cap"]) == ADMM_CAP || refuse("row $i has the cap $(r["cap"])")
        k = tryparse(Int, r["k"])
        k !== nothing && 1 <= k <= ADMM_CAP || refuse("row $i has k = $(r["k"])")
        meas = [r[x] for x in UNTIMED_MEASURES]
        v = [tryparse(Float64, x) for x in meas]
        all(x -> x !== nothing && isfinite(x) && x >= 0, v) || refuse("row $i has a measure that is not finite and nonnegative")
        if rec == "trajectory"
            t = get!(traj, (ρ, r["role"]), Dict{Int,Vector{String}}())
            haskey(t, k) && refuse("row $i repeats the trajectory row of $(r["role"]) at rho=$(r["rho"]), k = $k")
            t[k] = meas
            continue
        end
        r["level"] in ADMM_LEVELS || refuse("row $i has the level $(r["level"])")
        st = r["status"]
        (st == "reached" || (st == "capped" && k == ADMM_CAP)) || refuse("row $i has the status $st with k = $(r["k"])")
        rn, sn, epri, edual, stat = v[1], v[2], v[3], v[4], v[5]
        lo, hi = 1 - UNTIMED_SLACK, 1 + UNTIMED_SLACK
        met = r["level"] == "boyd" ? (rn <= epri * hi && sn <= edual * hi) : stat <= target * hi
        missed = r["level"] == "boyd" ? (rn >= epri * lo || sn >= edual * lo) : stat > target * lo
        (st == "reached" ? met : missed) || refuse("row $i: the measures contradict the status $st of the level $(r["level"])")
        key = (ρ, r["role"], r["level"])
        haskey(out, key) && refuse("row $i repeats the record of $(r["role"]) at rho=$(r["rho"]), level $(r["level"])")
        out[key] = (reached = st == "reached", k = k)
        levelmeas[key] = meas
    end
    for ρ in ADMM_RHOS, r in prim
        for lvl in ADMM_LEVELS
            haskey(out, (ρ, r.name, lvl)) || refuse("no $lvl record of $(r.name) at rho=$ρ")
        end
        b, t = out[(ρ, r.name, "boyd")], out[(ρ, r.name, "tight")]
        ran = b.reached && t.reached ? max(b.k, t.k) : ADMM_CAP
        want = Set(k for k in ADMM_CHECKPOINTS if k <= ran)
        push!(want, ran)
        b.reached && push!(want, b.k)
        t.reached && push!(want, t.k)
        rows_ = get(traj, (ρ, r.name), Dict{Int,Vector{String}}())
        Set(keys(rows_)) == want || refuse("the trajectory rows of $(r.name) at rho=$ρ are not the checkpoints, the last iteration and the reached iterations")
        for lvl in ADMM_LEVELS
            lm, x = levelmeas[(ρ, r.name, lvl)], out[(ρ, r.name, lvl)]
            get(rows_, x.k, nothing) == lm || refuse("the $lvl record of $(r.name) at rho=$ρ differs from its trajectory row")
        end
    end
    out, (file = name, sha256 = sha)
end

# One timed ADMM run of k iterations on the work arrays: the construction and factorization, then admm_loop! (Boyd's
# rule evaluated after each iteration, its count consumed through SINK, not used to stop the loop). The stationarity
# and the objective are evaluated after the timed interval, and Boyd's r, s, eps_pri and eps_dual of the last iteration
# (which the timed loop also computed) are recomputed there from its sums;
# valid says whether the endpoint meets the target of the level (Boyd's rule on the last iteration, or the
# stationarity target).
function admm_timed(role::Role, X::Matrix{T}, y::Vector{T}, λ::T, ρ::Real, k::Int, wk::AdmmWork{T}, level::String,
                    scale::Real) where {T}
    γ, κ = inv(T(ρ)), λ / T(ρ)
    reset!(wk)
    GC.gc()
    t0 = time_ns()
    obj = role.ctor(X, y, γ)
    role.prepare(obj)
    t1 = time_ns()
    flags, last = admm_loop!(wk, role.px, obj, γ, κ, ρ, k)
    t2 = time_ns()
    SINK[] += flags
    res, objective = lasso_measures!(wk, X, y, λ)
    bm = boyd_measures(size(X, 2), ρ, last)
    valid = level == "boyd" ? boyd_converged(size(X, 2), ρ, last) : res <= ADMM_TIGHT * scale
    (; t_factor = (t1 - t0) * 1e-9, t_iter = (t2 - t1) * 1e-9, res, objective, valid, boyd = bm)
end

# The timed ADMM stage of an instance, with the counts of load_kstar and the identity of their file, which is written to
# the timing file as the header line `# untimed_file <case>: <file> <sha256>`. admm_reached is 1 for reached and 0
# for capped.
function run_admm(w::Writer, roles, c::Case, d, kstar, ident)
    println(w.io, "# untimed_file ", c.id, ": ", ident.file, " ", ident.sha256)
    X, y = d.X, d.y
    λ = c.T(0.1 * c.lambda_max)
    scale = max(1.0, c.lambda_max)
    wk = AdmmWork(c.T, size(X, 1), size(X, 2))
    prim = [r for r in roles if !r.control]
    emit_load(w, c, "admm")
    for ρ in ADMM_RHOS, level in ADMM_LEVELS
        param = "rho=$(ρ),level=$(level)"
        info = Dict(r.family => kstar[(ρ, r.name, level)] for r in prim)
        for r in prim
            emit(w, c.id, "admm_iters", param, 0, 0, r.name, info[r.family].k)
            emit(w, c.id, "admm_reached", param, 0, 0, r.name, info[r.family].reached ? 1 : 0)
        end
        # only the roles of the families that reached the target are timed; a capped family is a lower bound
        active = [r for r in roles if info[r.family].reached]
        isempty(active) && continue
        for rep in 1:ADMM_REPEATS
            for (pos, idx) in enumerate(round_indices(length(active), rep))
                role = active[idx]
                t = guarded(() -> admm_timed(role, X, y, λ, ρ, info[role.family].k, wk, level, scale), w, c.id, "admm", param,
                            rep, pos, role.name)
                t === nothing && continue
                emit(w, c.id, "admm_total", param, rep, pos, role.name, t.t_factor + t.t_iter)
                emit(w, c.id, "admm_factor", param, rep, pos, role.name, t.t_factor)
                emit(w, c.id, "admm_iter", param, rep, pos, role.name, t.t_iter)
                emit(w, c.id, "admm_objective", param, rep, pos, role.name, t.objective)
                emit(w, c.id, "admm_stationarity", param, rep, pos, role.name, t.res)
                emit(w, c.id, "admm_valid", param, rep, pos, role.name, t.valid ? 1 : 0)
                for (meas, x) in zip(("admm_r", "admm_s", "admm_eps_pri", "admm_eps_dual"), t.boyd)
                    emit(w, c.id, meas, param, rep, pos, role.name, x)
                end
            end
        end
    end
end

###########################################################################################################
# Checks of the prototypes and the warm-up
###########################################################################################################

# The prototypes against S on a small full-rank case in each element type: the stationarity residual of every role
# (BigFloat; the 1e3 eps threshold is tested at the steps γ ≤ 1, the residual at γ = 1e3 is reported but not tested,
# since the plan expects C_B to lose accuracy at large steps), the A/A copies, the refactorization against a fresh
# object (bitwise), the construction path against the breakdown path (bitwise), the penalty ρ = inv(T(γ)), the
# warmed allocations and the finiteness error.
function check_prototypes(roles)
    println("== prototypes against S on a small full-rank case (η in BigFloat; failure above 1e3 eps for γ ≤ 1)")
    failed = String[]
    for T in (Float64, Float32)
        n, p = 20, 60
        rng = MersenneTwister(2026)
        X = randn(rng, T, n, p)
        for j in 1:p
            c = view(X, :, j)
            c ./= norm(c)
        end
        y = randn(rng, T, n)
        vs = [randn(rng, T, p) for _ in 1:2]
        γs = T[1e-3, 1, 1e3]
        thr = 1e3 * eps(T)
        β = zeros(T, p)
        outs = Dict{Tuple{String,Int,Int},Vector{T}}()
        ηmax, ηlarge = Dict{String,Float64}(), Dict{String,Float64}()
        for r in roles
            obj = r.ctor(X, y, γs[1])
            r.prepare(obj)
            for (gi, γ) in enumerate(γs), (vi, v) in enumerate(vs)
                r.refactor(obj, γ)
                r.px(β, obj, v, γ)
                outs[(r.name, gi, vi)] = copy(β)
                η = eta_exact(X, y, v, γ, β)
                dict = γ <= 1 ? ηmax : ηlarge
                dict[r.name] = max(get(dict, r.name, 0.0), η)
            end
        end
        for r in roles
            diff = maximum(norm(outs[(r.name, gi, vi)] - outs[("S", gi, vi)], Inf) / norm(outs[("S", gi, vi)], Inf)
                           for gi in eachindex(γs), vi in eachindex(vs))
            ident = r.control ? all(bits_equal(outs[(r.name, gi, vi)], outs[(replace(r.name, "'" => ""), gi, vi)])
                                    for gi in eachindex(γs), vi in eachindex(vs)) : missing
            @printf("%-8s %-8s  max η/eps %9.3g (γ ≤ 1)  %9.3g (γ = 1e3, reported only)   max rel diff to S %9.3g   copy identical to its role: %s\n",
                    string(T), r.name, ηmax[r.name] / eps(T), ηlarge[r.name] / eps(T), diff, ident)
            ηmax[r.name] <= thr || push!(failed, "$(r.name) $T η/eps = $(ηmax[r.name] / eps(T))")
            ismissing(ident) || ident || push!(failed, "$(r.name) $T differs from its role")
        end
        # refactorization against a fresh object, the construction path against the breakdown path, the penalty,
        # the finiteness error and the warmed allocations
        for r in roles
            obj = r.ctor(X, y, γs[1])
            r.prepare(obj)
            r.refactor(obj, γs[3])
            b1 = zeros(T, p)
            r.px(b1, obj, vs[1], γs[3])
            fresh = r.ctor(X, y, γs[3])
            r.prepare(fresh)
            b2 = zeros(T, p)
            r.px(b2, fresh, vs[1], γs[3])
            same = bits_equal(b1, b2)
            same || push!(failed, "$(r.name) $T: the output after refactor! differs from that of a fresh object")
            if r.ctor isa PCtor
                L = typeof(r.ctor).parameters[1]
                bobj, _ = getfield(parentmodule(L), :breakdown)(L, X, y, γs[3])
                b3 = zeros(T, p)
                r.px(b3, bobj, vs[1], γs[3])
                bits_equal(b3, b2) || push!(failed, "$(r.name) $T: the breakdown path builds another object than the constructor")
                ρ = inv(T(γs[3]))
                (getfield(fresh, :ρ) == ρ && getfield(fresh, :invρ) == inv(ρ)) ||
                    push!(failed, "$(r.name) $T: the penalty is not ρ = inv(T(γ))")
            end
            bad = copy(X)
            bad[3, 4] = T(NaN)
            msg = try
                o = r.ctor(bad, y, one(T)); r.prepare(o); r.px(β, o, vs[1], one(T)); "no error"
            catch e
                e isa ArgumentError ? e.msg : "other error: " * oneline(sprint(showerror, e))
            end
            msg == Proto.FINITE_MESSAGE || push!(failed, "$(r.name) finiteness error: $msg")
            op = WarmBatch(r.px, obj, β, vs, γs[3], length(vs))
            b = warmed_bytes(op)
            gs = penalty_schedules(T)[1][2]
            r.refactor(obj, gs[end])
            op2 = PenaltyBatch(r.px, r.refactor, obj, β, vs, gs, lcm(length(gs), length(vs)))
            b2a = warmed_bytes(op2)
            @printf("%-8s %-8s  warmed bytes: prox %d, refactor and prox %d; refactor bitwise equal to a fresh object: %s\n",
                    string(T), r.name, b, b2a, same)
            r.family == "S" || b == 0 || push!(failed, "$(r.name) $T warmed prox allocates $b bytes")
        end
        for r in roles
            r.family == "S" && continue
            parts = breakdown_sample(r, X, y, γs[2])
            println("  ", string(T), " ", r.name, " setup steps (s): ", join(["$k=$(round(v; sigdigits = 3))" for (k, v) in pairs(parts)], " "))
        end
    end
    isempty(failed) || error("prototype checks failed: " * join(failed, "; "))
    println("prototype checks passed")
end

# Calls the setup, prox, refactor, allocation-probe and ADMM routines of each role on small random data, for both
# element types except the ADMM routines (Float64 only, the type of the ADMM groups), so that their compilation
# happens before the timed samples. First use is later timed in a warmed process: the native libraries keep state
# that the first calls initialized.
function warm_up(roles)
    for T in (Float32, Float64)
        n, p = 12, 40
        rng = MersenneTwister(1)
        X = randn(rng, T, n, p)
        y = randn(rng, T, n)
        vs = [randn(rng, T, p) for _ in 1:4]
        β = zeros(T, p)
        for r in roles
            first_use_sample(r, X, y, β, vs[1], one(T))
            obj = r.ctor(X, y, one(T))
            r.prepare(obj)
            r.px(β, obj, vs[1], one(T))
            timed_sample(WarmBatch(r.px, obj, β, vs, one(T), 4))
            warmed_bytes(WarmBatch(r.px, obj, β, vs, one(T), 4))
            for (_, sched) in penalty_schedules(T)
                cycle = lcm(length(sched), 4)
                r.refactor(obj, sched[end])
                timed_sample(PenaltyBatch(r.px, r.refactor, obj, β, vs, sched, cycle))
                warmed_bytes(PenaltyBatch(r.px, r.refactor, obj, β, vs, sched, cycle))
            end
            r.control || breakdown_sample(r, X, y, one(T))
            Base.summarysize(obj)
        end
    end
    X = randn(MersenneTwister(2), 12, 40)
    y = randn(MersenneTwister(3), 12)
    wk = AdmmWork(Float64, 12, 40)
    for r in roles
        admm_untimed(r, X, y, 0.1, 1.0; tight_tol = 1e-3, scale = 1.0, cap = 50)
        admm_timed(r, X, y, 0.1, 1.0, 5, wk, "boyd", 1.0)
        admm_timed(r, X, y, 0.1, 1.0, 5, wk, "tight", 1.0)
        admm_reference(r, X, y, 0.1, 1.0; tol = 1e-3, scale = 1.0, cap = 11)
    end
    nothing
end

###########################################################################################################
# The process
###########################################################################################################

# The files that the requested groups would write: the timing file of this process, and for the selected cases the
# outputs index, failures index and binaries (outputs), the reference file and its z binary (admm_reference) and the
# untimed file (admm_untimed).
function planned_files(m::Manifest, outdir::String, refdir::String, tfile::String)
    files = isempty(tfile) ? String[] : [tfile]
    for g in GROUPS
        cases = select_for(m, g)
        if g == "outputs"
            od = joinpath(outdir, "outputs-$(VERSION)")
            push!(files, joinpath(od, "outputs-index.tsv"), joinpath(od, "outputs-failures.tsv"))
            append!(files, [joinpath(od, output_file(c, r)) for c in cases for r in ("S", "C_B", "C_D", "C_W")])
        elseif g == "admm_reference"
            for c in cases
                path = reference_path(refdir, c)
                push!(files, path, replace(path, ".tsv" => "-z.f64"))
            end
        elseif g == "admm_untimed"
            append!(files, [untimed_path(refdir, c) for c in cases])
        end
    end
    files
end

# Stops the process when a file that it would write exists already, so that each file the harness writes comes from
# one attempt (other processes can still write other files into the same directory).
function refuse_existing(files)
    old = filter(isfile, files)
    isempty(old) || error("the output files exist already: " * join(old, ", ") *
                          "; use a new empty PB_OUTPUT (and PB_REFERENCE) or remove these files")
    nothing
end

function main()
    roles = ordered_roles(build_roles(), ORDER)
    println("julia ", VERSION, ", order ", ORDER, ", blas threads ", BLAS.get_num_threads(), ", ", BLAS.get_config())
    println("CPU ", cpu_model(), "; load average ", loadavg())
    CHECK_ONLY && check_prototypes(roles)
    inputs = env_string("PB_INPUTS", "")
    if isempty(inputs)
        CHECK_ONLY || error("PB_INPUTS names the directory written by generate.jl")
        warm_up(roles)
        println("no PB_INPUTS: only the prototype checks and the warm-up ran")
        return
    end
    m = load_manifest(inputs)
    outdir = env_string("PB_OUTPUT", "")
    if isempty(outdir)
        CHECK_ONLY || error("PB_OUTPUT names the output directory")
        outdir = mktempdir()
        println("output directory ", outdir)
    end
    mkpath(outdir)
    refdir = env_string("PB_REFERENCE", outdir)
    timed = [g for g in GROUPS if g in TIMED_GROUPS]
    tfile = isempty(timed) ? "" : joinpath(outdir, "timings-$(VERSION)-$(ORDER)-$(PROC)-$(join(timed, "+")).tsv")
    refuse_existing(planned_files(m, outdir, refdir, tfile))
    t0 = time()
    warm_up(roles)
    println("warm-up (compilation of the roles): ", round(time() - t0; digits = 1), " s")
    writer = nothing
    if !isempty(timed)
        io = open(tfile, "w")
        write_header(io, timed, m.sha256)
        writer = Writer(io, 0)
    end
    tstart = time()
    for g in ALL_GROUPS
        g in GROUPS || continue
        cases = select_for(m, g)
        isempty(cases) && (println("group ", g, ": no case selected"); continue)
        g0, r0 = time(), writer === nothing ? 0 : writer.rows
        if g == "outputs"
            run_outputs(roles, m, cases, joinpath(outdir, "outputs-$(VERSION)"))
        elseif g == "admm_reference"
            run_admm_reference(m, cases, roles, refdir)
        elseif g == "admm_untimed"
            run_admm_untimed(m, cases, roles, refdir)
        else
            for c in cases
                println("group ", g, ": ", c.id)
                d = case_data(m, c)
                try
                    g == "first_use" && run_first_use(writer, roles, c, d)
                    g == "warm" && run_warm(writer, roles, c, d)
                    g == "penalty" && run_penalty(writer, roles, c, d)
                    g == "alloc" && run_alloc(writer, roles, c, d)
                    if g == "admm"
                        kst, ident = load_kstar(refdir, m, c, [r for r in roles if !r.control])
                        run_admm(writer, roles, c, d, kst, ident)
                    end
                catch e
                    CHECK_ONLY && rethrow()
                    println("  ERROR in ", g, " ", c.id, ": ", errmsg(e))
                    emit(writer, c.id, "error", "$g|case|" * errmsg(e), 0, 0, "-", NaN)
                end
                flush(writer.io)
            end
        end
        println("GROUP ", g, " wall ", round(time() - g0; digits = 1), " s, rows ", writer === nothing ? 0 : writer.rows - r0,
                ", cases ", length(cases))
    end
    if writer !== nothing
        println(writer.io, "# loadavg_end: ", loadavg())
        println(writer.io, "# wall_seconds: ", round(time() - tstart; digits = 1))
        println(writer.io, "# finished_utc: ", Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ"))
        println(writer.io, "# complete: yes")
        close(writer.io)
        println("wrote ", writer.rows, " rows to ", basename(tfile))
        CHECK_ONLY && println("summary file: ", tfile)
    end
    println("harness done", CHECK_ONLY ? " (CHECK_ONLY)" : "")
end

main()
