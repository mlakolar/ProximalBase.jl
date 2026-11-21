# Paired timing harness of the wide least-squares study (#1103). One process loads the package source twice (roles S
# and S', the A/A control) and the Cholesky prototypes twice (C_B, C_D, C_W and their copies C_B', C_D', C_W'), reads
# the inputs written by generate.jl and times the roles against each other. The measurement groups are
#   first_use       construction, factorization and first prox on a fresh object (one sample per role and round),
#                   and the time of each setup step of the primary roles (first_use_part)
#   warm            the warmed prox, minimum over batches of calls, at every step and for two kinds of inputs
#   penalty         a new step on every call (S: the prox only; C: refactorization from the cached XX', then the prox)
#   alloc           warmed bytes of the prox and of the step change, and the reachable bytes of each object
#   admm_reference  S-ADMM to stationarity 1e-12·max(1, ‖X'y‖∞) on each lasso instance (the reference optimum)
#   admm_untimed    the iteration counts of each role for the two stopping levels, with the trajectory
#   admm            timed ADMM runs of those iteration counts, with the objective, its gap to the reference
#                   and the stationarity at the end of each run
#   outputs         the prox outputs of the primary roles from an untimed call, for accuracy.jl
#
#   PB_INPUTS=<inputs dir> PB_OUTPUT=<output dir> [PB_ORDER=S_first|C_first] [PB_ROUNDS=21] [PB_PROC=1] \
#   [PB_GROUP=first_use,warm,penalty,alloc,admm] [PB_FILTER=<regex on case ids>] [PB_BLAS_THREADS=1] \
#   [PB_BATCHES=3] [PB_ADMM_REPEATS=5] [PB_ADMM_CAP=20000] [PB_REFERENCE_CAP=100000] [PB_REFERENCE=<dir of the reference optimum and the iteration counts>] \
#   [PB_SOURCE=<root of the package tree>] [PB_SOURCE_SHA=<sha>] [PB_HARNESS_SHA=<sha>] \
#   julia --threads=1 --check-bounds=no --project=. docs/audit/wide-leastsquares-solves-1103/harness.jl
#
# PB_CHECK_ONLY=1 first checks the prototypes against S on a small full-rank case (the stationarity residual of
# each in BigFloat, the warmed allocations, the finiteness error), then, when PB_INPUTS is set, runs every group
# once on the smallest case of each element type with 2 rounds, 2 ADMM repeats and the output in PB_OUTPUT or a
# temporary directory. The output is one tab-separated file per process, timings-<julia>-<order>-<proc>-<groups>.tsv,
# with a header of `# key: value` lines and one row per (case, measurement, param, round, role). Times are seconds
# per call (the minimum over the batches of the round for warm), bytes and allocations per call.
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
const ADMM_RHOS = (0.1, 1.0, 10.0)
const ADMM_TIGHT = 1e-8
const REFERENCE_TOL = 1e-12
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

# m prox calls cycling through the inputs vs
struct WarmBatch{P,O,B,Vs,G}
    px::P
    obj::O
    β::B
    vs::Vs
    γ::G
    m::Int
end
function (b::WarmBatch)()
    K = length(b.vs)
    @inbounds for i in 1:b.m
        b.px(b.β, b.obj, b.vs[mod1(i, K)], b.γ)
    end
    nothing
end

# m calls, each with a new step from gs (cycled) followed by the prox
struct PenaltyBatch{P,R,O,B,Vs,Gs}
    px::P
    refactor::R
    obj::O
    β::B
    vs::Vs
    gs::Gs
    m::Int
end
function (b::PenaltyBatch)()
    K, G = length(b.vs), length(b.gs)
    @inbounds for i in 1:b.m
        γ = b.gs[mod1(i, G)]
        b.refactor(b.obj, γ)
        b.px(b.β, b.obj, b.vs[mod1(i, K)], γ)
    end
    nothing
end

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

# seconds of the setup steps of S: the copy with the finiteness check, svd, the materialized V and Diagonal(s)U'y
# (the sink keeps the results of the steps alive)
const SINK = Ref(0.0)
function s_parts(X::Matrix{T}, y::Vector{T}) where {T}
    t0 = time_ns()
    A = Matrix{T}(X)
    all(isfinite, A) || error("nonfinite design")
    t1 = time_ns()
    F = svd(A)
    t2 = time_ns()
    V = Matrix{T}(F.V)
    t3 = time_ns()
    sUtY = Vector{T}(transpose(F.U) * y)
    sUtY .*= F.S
    s2 = F.S .^ 2
    t4 = time_ns()
    SINK[] += Float64(V[1]) + Float64(sUtY[1]) + Float64(s2[1])
    (; copy_check = (t1 - t0) * 1e-9, svd = (t2 - t1) * 1e-9, V = (t3 - t2) * 1e-9, UtY = (t4 - t3) * 1e-9)
end
function breakdown_sample(role::Role{C}, X, y, γ) where {C<:SCtor}
    GC.gc()
    s_parts(X, y)
end
function breakdown_sample(role::Role{C}, X, y, γ) where {C<:PCtor}
    L = C.parameters[1]
    bp = getfield(parentmodule(L), :build_parts)
    GC.gc()
    bp(L, X, y, γ)[2]
end

batch_calls(t) = t < 0.1 ? clamp(ceil(Int, 0.02 / t), 30, 3000) : 1

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

loadavg() = try join(split(read("/proc/loadavg", String))[1:3], " ") catch; "unavailable" end
cpu_model() = try Sys.cpu_info()[1].model catch; "unknown" end
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
oneline(s) = replace(string(s), r"[\t\r\n]+" => " ")

function write_header(io, groups, inputs_sha)
    for (k, v) in [("format", "pb1103-timings-1"), ("julia", string(VERSION)), ("order", ORDER), ("proc", PROC),
                   ("groups", join(groups, ",")), ("rounds", ROUNDS), ("batches", BATCHES), ("admm_repeats", ADMM_REPEATS),
                   ("admm_cap", ADMM_CAP), ("threads", Threads.nthreads()), ("blas_threads", BLAS.get_num_threads()),
                   ("blas_config", oneline(BLAS.get_config())), ("openblas", openblas_config()),
                   ("cpu_model", cpu_model()), ("cpu_name", Sys.CPU_NAME), ("cpu_threads", Sys.CPU_THREADS),
                   ("l3_cache", l3_cache()), ("check_bounds", Base.JLOptions().check_bounds), ("hostname", gethostname()),
                   ("loadavg_start", loadavg()), ("source_sha", get(ENV, "PB_SOURCE_SHA", "unset")),
                   ("harness_sha", get(ENV, "PB_HARNESS_SHA", "unset")), ("inputs_index_sha256", inputs_sha),
                   ("started_utc", Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ"))]
        println(io, "# ", k, ": ", v)
    end
    println(io, join(["case", "measurement", "param", "round", "position", "role", "value", "gc_s", "bytes", "allocs"], '\t'))
end

###########################################################################################################
# Cases
###########################################################################################################

function case_data(m::Manifest, c::Case)
    X, y, V = read_input(m, c, "X"), read_input(m, c, "y"), read_input(m, c, "V")
    vs = [V[:, k] for k in 1:size(V, 2)]
    (; X, y, vs, rand = vs[1:min(4, length(vs))], admm = length(vs) >= 8 ? vs[5:8] : vs, β = zeros(c.T, c.p))
end

# the objects of the roles of a case at the step γ, factorized and called once
function make_objects(roles, d, γ)
    Any[(obj = r.ctor(d.X, d.y, γ); r.prepare(obj); r.px(d.β, obj, d.vs[1], γ); obj) for r in roles]
end

# batch sizes of the families: at least 30 calls for calls under 0.1 s (one call at or above it)
function calibrate(make_op, roles, objs)
    ms = Dict{String,Int}()
    ts = Dict{String,Float64}()
    for (r, o) in zip(roles, objs)
        r.control && continue
        op = make_op(r, o, 1)
        t = min(timed_sample(op).time, timed_sample(op).time, timed_sample(op).time)
        ms[r.family] = batch_calls(t)
        ts[r.family] = t
    end
    ms, ts
end

function select_for(m::Manifest, group::AbstractString)
    cs = if group in ("first_use", "warm", "penalty")
        select_cases(m; kind = :main, timed = true)
    elseif group in ("alloc", "outputs")
        select_cases(m; kind = :main)
    else
        select_cases(m; kind = :main, lasso = true, T = Float64)
    end
    sort!(cs; by = c -> (c.n * c.p, c.id))
    if CHECK_ONLY
        cs = [first(filter(c -> c.T === T, cs)) for T in (Float64, Float32) if any(c -> c.T === T, cs)]
    end
    cs
end

###########################################################################################################
# Groups
###########################################################################################################

function run_first_use(w::Writer, roles, c::Case, d)
    γ = one(c.T)
    prim = [r for r in roles if !r.control]
    for round in 1:ROUNDS
        for (pos, idx) in enumerate(round_indices(length(roles), round))
            role = roles[idx]
            s = first_use_sample(role, d.X, d.y, d.β, d.vs[1], γ)
            emit(w, c.id, "first_use", "g=1", round, pos, role.name, s.time; gc = s.gc, bytes = s.bytes, allocs = s.allocs)
        end
        for (pos, idx) in enumerate(round_indices(length(prim), round))
            role = prim[idx]
            for (k, v) in pairs(breakdown_sample(role, d.X, d.y, γ))
                emit(w, c.id, "first_use_part", string(k), round, pos, role.name, v)
            end
        end
    end
end

function run_warm(w::Writer, roles, c::Case, d)
    T = c.T
    objs = make_objects(roles, d, one(T))
    ms, _ = calibrate((r, o, m) -> WarmBatch(r.px, o, d.β, d.rand, one(T), m), roles, objs)
    for g in c.gammas, (vkind, vs) in (("rand", d.rand), ("admm", d.admm))
        γ = T(g)
        for (r, o) in zip(roles, objs)
            r.refactor(o, γ)
        end
        param = @sprintf("g=%g/%s", g, vkind)
        for round in 1:ROUNDS
            for (pos, idx) in enumerate(round_indices(length(roles), round))
                role, obj = roles[idx], objs[idx]
                mm = ms[role.family]
                s = best_batch(() -> WarmBatch(role.px, obj, d.β, vs, γ, mm), BATCHES)
                emit(w, c.id, "warm", param, round, pos, role.name, s.time / mm; gc = s.gc / mm, bytes = s.bytes / mm,
                     allocs = s.allocs / mm)
            end
        end
    end
end

const NEWSTEP_GRID = [10.0^(-3 + 6 * k / 15) for k in 0:15]
function run_penalty(w::Writer, roles, c::Case, d)
    T = c.T
    objs = make_objects(roles, d, one(T))
    patterns = (("alt2", T[1, 1e3]), ("newstep", T.(NEWSTEP_GRID)))
    for (pname, gs) in patterns
        ms, ts = calibrate((r, o, m) -> PenaltyBatch(r.px, r.refactor, o, d.β, d.rand, T[2], m), roles, objs)
        for round in 1:ROUNDS
            for (pos, idx) in enumerate(round_indices(length(roles), round))
                role, obj = roles[idx], objs[idx]
                mm = ms[role.family]
                nb = ts[role.family] >= 0.1 ? 1 : BATCHES
                s = best_batch(() -> PenaltyBatch(role.px, role.refactor, obj, d.β, d.rand, gs, mm), nb)
                emit(w, c.id, "penalty", pname, round, pos, role.name, s.time / mm; gc = s.gc / mm, bytes = s.bytes / mm,
                     allocs = s.allocs / mm)
            end
        end
    end
end

function run_alloc(w::Writer, roles, c::Case, d)
    T = c.T
    γ = one(T)
    emit(w, c.id, "loadavg_1m", "-", 0, 0, "-", parse(Float64, split(loadavg())[1]))
    for (pos, role) in enumerate(roles)
        obj = role.ctor(d.X, d.y, γ)
        role.prepare(obj)
        op = WarmBatch(role.px, obj, d.β, d.rand, γ, 1)
        b = warmed_bytes(op)
        s = timed_sample(op)
        emit(w, c.id, "alloc", "prox", 1, pos, role.name, NaN; bytes = b, allocs = s.allocs)
        op2 = PenaltyBatch(role.px, role.refactor, obj, d.β, d.rand, T[2], 1)
        b2 = warmed_bytes(op2)
        s2 = timed_sample(op2)
        emit(w, c.id, "alloc", "refactor_prox", 1, pos, role.name, NaN; bytes = b2, allocs = s2.allocs)
        emit(w, c.id, "workingset", "reachable_bytes", 1, pos, role.name, Base.summarysize(obj))
    end
end

# the outputs of one role for every step of the case and every input, as a p × (inputs · steps) array
function compute_outputs(role, c::Case, d)
    T = c.T
    γs = T.(c.gammas)
    K = length(d.vs)
    B = fill(T(NaN), c.p, K * length(γs))
    status = "ok"
    try
        obj = role.ctor(d.X, d.y, γs[1])
        role.prepare(obj)
        for (gi, γ) in enumerate(γs)
            role.refactor(obj, γ)
            for k in 1:K
                role.px(d.β, obj, d.vs[k], γ)
                B[:, (gi - 1) * K + k] .= d.β
            end
        end
    catch e
        status = "error: " * oneline(first(split(sprint(showerror, e), '\n')))
    end
    B, status
end

function run_outputs(roles, m::Manifest, cases, dir::String)
    mkpath(dir)
    prim = [r for r in roles if !r.control]
    copies = Dict(r.family => r for r in roles if r.control)
    path = joinpath(dir, "outputs-index.tsv")
    open(path, "w") do io
        println(io, "# julia: ", VERSION)
        println(io, "# order: ", ORDER)
        println(io, "# source_sha: ", get(ENV, "PB_SOURCE_SHA", "unset"))
        println(io, "# harness_sha: ", get(ENV, "PB_HARNESS_SHA", "unset"))
        println(io, "# inputs_index_sha256: ", m.sha256)
        println(io, join(["case", "role", "file", "eltype", "shape", "sha256", "status", "copy_identical", "gammas", "nv"], '\t'))
        for c in cases
            d = case_data(m, c)
            K = length(d.vs)
            for role in prim
                B, status = compute_outputs(role, c, d)
                B2, status2 = compute_outputs(copies[role.family], c, d)
                identical = status == "ok" && status2 == "ok" && isequal(B, B2)
                file = "out-" * case_file(c.id) * "-" * replace(role.name, "'" => "p") * (c.T === Float64 ? ".f64" : ".f32")
                open(joinpath(dir, file), "w") do f
                    write(f, B)
                end
                println(io, join([c.id, role.name, file, c.T === Float64 ? "Float64" : "Float32",
                                  string(c.p, "x", size(B, 2)), file_sha256(joinpath(dir, file)), status, identical,
                                  join(repr.(c.gammas), ","), K], '\t'))
                flush(io)
            end
            println("outputs written for ", c.id)
        end
    end
end

reference_path(dir, c::Case) = joinpath(dir, "reference-" * case_file(c.id) * ".tsv")

function run_admm_reference(m::Manifest, cases, roles, dir::String)
    mkpath(dir)
    S = first(r for r in roles if r.name == "S")
    for c in cases
        d = case_data(m, c)
        λ = c.T(0.1 * c.lambda_max)
        t0 = time()
        r = admm_reference(S, d.X, d.y, λ, 1.0; tol = REFERENCE_TOL, scale = max(1.0, c.lambda_max), cap = REFERENCE_CAP)
        zfile = "reference-" * case_file(c.id) * "-z.f64"
        open(joinpath(dir, zfile), "w") do f
            write(f, r.z)
        end
        open(reference_path(dir, c), "w") do io
            for (k, v) in [("case", c.id), ("julia", VERSION), ("rho", 1.0), ("tol_times_scale", REFERENCE_TOL * max(1.0, c.lambda_max)),
                           ("cap", REFERENCE_CAP), ("iterations", r.iterations), ("converged", r.converged),
                           ("stationarity", r.stationarity), ("objective", repr(r.objective)), ("lambda", Float64(λ)),
                           ("lambda_max", c.lambda_max), ("z_file", zfile), ("z_sha256", file_sha256(joinpath(dir, zfile)))]
                println(io, k, '\t', v)
            end
        end
        println("reference for ", c.id, ": ", r.iterations, " iterations, converged ", r.converged, ", stationarity ",
                r.stationarity, ", objective ", r.objective, " (", round(time() - t0; digits = 1), " s)")
    end
end

function read_kv(path)
    Dict(String(k) => String(v) for (k, v) in (split(line, '\t'; limit = 2) for line in eachline(path) if !isempty(line)))
end

function run_admm_untimed(m::Manifest, cases, roles, dir::String)
    mkpath(dir)
    path = joinpath(dir, "admm-untimed-$(VERSION).tsv")
    fresh = !isfile(path)
    prim = [r for r in roles if !r.control]
    open(path, "a") do io
        fresh && println(io, join(["case", "rho", "role", "record", "k", "r_norm", "s_norm", "stationarity", "objective"], '\t'))
        for c in cases
            d = case_data(m, c)
            λ = c.T(0.1 * c.lambda_max)
            scale = max(1.0, c.lambda_max)
            for ρ in ADMM_RHOS, role in prim
                t0 = time()
                r = admm_untimed(role, d.X, d.y, λ, ρ; tight_tol = ADMM_TIGHT, scale, cap = ADMM_CAP)
                for row in r.rows
                    @printf(io, "%s\t%s\t%s\ttrajectory\t%d\t%.10g\t%.10g\t%.10g\t%.12g\n", c.id, ρ, role.name, Int(row[1]), row[2], row[3], row[4], row[5])
                end
                @printf(io, "%s\t%s\t%s\tk_boyd\t%d\tNaN\tNaN\tNaN\tNaN\n", c.id, ρ, role.name, r.k_boyd)
                @printf(io, "%s\t%s\t%s\tk_tight\t%d\tNaN\tNaN\t%.10g\t%.12g\n", c.id, ρ, role.name, r.k_tight, r.final_stationarity, r.final_objective)
                flush(io)
                println("admm_untimed ", c.id, " rho=", ρ, " ", role.name, ": k_boyd ", r.k_boyd, " k_tight ", r.k_tight,
                        " (", round(time() - t0; digits = 1), " s)")
            end
        end
    end
end

# the iteration counts of the untimed pass: (case, ρ, role) => (k_boyd, k_tight), the last record of a file
function load_kstar(dir::String)
    path = joinpath(dir, "admm-untimed-$(VERSION).tsv")
    isfile(path) || error("missing $path: run the admm_untimed group first")
    out = Dict{Tuple{String,String,String},Dict{String,Int}}()
    lines = readlines(path)
    for line in lines[2:end]
        f = split(line, '\t')
        f[4] in ("k_boyd", "k_tight") || continue
        get!(out, (String(f[1]), String(f[2]), String(f[3])), Dict{String,Int}())[String(f[4])] = parse(Int, f[5])
    end
    out
end

function run_admm(w::Writer, roles, c::Case, d, kstar, refdir::String)
    X, y = d.X, d.y
    λ = c.T(0.1 * c.lambda_max)
    rpath = reference_path(refdir, c)
    pstar = isfile(rpath) ? parse(Float64, read_kv(rpath)["objective"]) : NaN
    isfile(rpath) || @warn "no reference optimum for $(c.id); the objective gaps are NaN"
    wk = AdmmWork(c.T, size(X, 1), size(X, 2))
    prim = [r for r in roles if !r.control]
    for ρ in ADMM_RHOS
        ks = Dict(r.family => kstar[(c.id, string(ρ), r.name)] for r in prim)
        for level in ("boyd", "tight")
            counts = Dict(fam => level == "boyd" ? k["k_boyd"] : k["k_tight"] for (fam, k) in ks)
            param = "rho=$(ρ),level=$(level)"
            for r in prim
                emit(w, c.id, "admm_iters", param, 0, 0, r.name, counts[r.family])
            end
            if any(==(0), values(counts))
                for r in prim
                    emit(w, c.id, "admm_capped", param, 0, 0, r.name, counts[r.family] == 0 ? 1 : 0)
                end
                continue
            end
            for rep in 1:ADMM_REPEATS
                for (pos, idx) in enumerate(round_indices(length(roles), rep))
                    role = roles[idx]
                    t = admm_timed(role, X, y, λ, ρ, counts[role.family], wk)
                    res, objective = lasso_measures!(wk, X, y, λ)
                    emit(w, c.id, "admm_total", param, rep, pos, role.name, t.t_factor + t.t_iter)
                    emit(w, c.id, "admm_factor", param, rep, pos, role.name, t.t_factor)
                    emit(w, c.id, "admm_iter", param, rep, pos, role.name, t.t_iter)
                    emit(w, c.id, "admm_objective", param, rep, pos, role.name, objective)
                    emit(w, c.id, "admm_gap", param, rep, pos, role.name, objective - pstar)
                    emit(w, c.id, "admm_stationarity", param, rep, pos, role.name, res)
                end
            end
        end
    end
end

# one timed ADMM run of kstar iterations on the work arrays wk: the construction and factorization, then the loop
function admm_timed(role::Role, X::Matrix{T}, y::Vector{T}, λ::T, ρ::Real, kstar::Int, wk::AdmmWork{T}) where {T}
    γ, κ = inv(T(ρ)), λ / T(ρ)
    reset!(wk)
    GC.gc()
    t0 = time_ns()
    obj = role.ctor(X, y, γ)
    role.prepare(obj)
    t1 = time_ns()
    for _ in 1:kstar
        admm_step!(wk, role.px, obj, γ, κ)
    end
    t2 = time_ns()
    (; t_factor = (t1 - t0) * 1e-9, t_iter = (t2 - t1) * 1e-9)
end

###########################################################################################################
# Checks of the prototypes and the warm-up
###########################################################################################################

# The prototypes against S on a small full-rank case in each element type: the stationarity residual of every role
# (BigFloat; the 1e3 eps threshold applies at the steps γ ≤ 1, the residual at γ = 1e3 is only reported because the
# form of C_B subtracts terms of size ‖X'y‖/ρ there), the A/A copies, the refactorization, the warmed allocations
# and the finiteness error.
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
            ident = r.control ? all(isequal(outs[(r.name, gi, vi)], outs[(replace(r.name, "'" => ""), gi, vi)]) for gi in eachindex(γs), vi in eachindex(vs)) : missing
            @printf("%-8s %-8s  max η/eps %9.3g (γ ≤ 1)  %9.3g (γ = 1e3, reported only)   max rel diff to S %9.3g   copy identical to its role: %s\n",
                    string(T), r.name, ηmax[r.name] / eps(T), ηlarge[r.name] / eps(T), diff, ident)
            ηmax[r.name] <= thr || push!(failed, "$(r.name) $T η/eps = $(ηmax[r.name] / eps(T))")
            ismissing(ident) || ident || push!(failed, "$(r.name) $T differs from its role")
        end
        # refactorization against a fresh object, the finiteness error and the warmed allocations
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
            same = isequal(b1, b2)
            same || r.family == "S" || println("  note: ", r.name, " after refactor! differs from a fresh object by ", norm(b1 - b2, Inf))
            bad = copy(X)
            bad[3, 4] = T(NaN)
            msg = try
                o = r.ctor(bad, y, one(T)); r.prepare(o); r.px(β, o, vs[1], one(T)); "no error"
            catch e
                e isa ArgumentError ? e.msg : "other error: " * oneline(sprint(showerror, e))
            end
            msg == Proto.FINITE_MESSAGE || push!(failed, "$(r.name) finiteness error: $msg")
            op = WarmBatch(r.px, obj, β, vs, γs[3], 1)
            b = warmed_bytes(op)
            op2 = PenaltyBatch(r.px, r.refactor, obj, β, vs, T[2], 1)
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

# compiles every code path of every role for both element types on small random data
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
            timed_sample(WarmBatch(r.px, obj, β, vs, one(T), 5))
            r.refactor(obj, T(2))
            timed_sample(PenaltyBatch(r.px, r.refactor, obj, β, vs, T[1, 2], 5))
            r.refactor(obj, T(2))
            warmed_bytes(WarmBatch(r.px, obj, β, vs, T(2), 1))
            warmed_bytes(PenaltyBatch(r.px, r.refactor, obj, β, vs, T[2], 1))
            r.control || breakdown_sample(r, X, y, one(T))
            Base.summarysize(obj)
        end
    end
    X = randn(MersenneTwister(2), 12, 40)
    y = randn(MersenneTwister(3), 12)
    wk = AdmmWork(Float64, 12, 40)
    for r in roles
        admm_untimed(r, X, y, 0.1, 1.0; tight_tol = 1e-3, scale = 1.0, cap = 50)
        admm_timed(r, X, y, 0.1, 1.0, 5, wk)
        lasso_measures!(wk, X, y, 0.1)
    end
    for r in roles
        admm_reference(r, X, y, 0.1, 1.0; tol = 1e-3, scale = 1.0, cap = 50)
    end
    nothing
end

###########################################################################################################
# The process
###########################################################################################################

function main()
    roles = ordered_roles(build_roles(), ORDER)
    println("julia ", VERSION, ", order ", ORDER, ", blas threads ", BLAS.get_num_threads(), ", ", BLAS.get_config())
    println("CPU ", cpu_model(), "; load average ", loadavg())
    CHECK_ONLY && check_prototypes(roles)
    t0 = time()
    warm_up(roles)
    println("warm-up (compilation of every role): ", round(time() - t0; digits = 1), " s")
    inputs = env_string("PB_INPUTS", "")
    if isempty(inputs)
        CHECK_ONLY || error("PB_INPUTS names the directory written by generate.jl")
        println("no PB_INPUTS: only the prototype checks ran")
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
    writer = nothing
    timed = [g for g in GROUPS if g in TIMED_GROUPS]
    tfile = ""
    if !isempty(timed)
        tfile = joinpath(outdir, "timings-$(VERSION)-$(ORDER)-$(PROC)-$(join(timed, "+")).tsv")
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
            ctl = select_cases(m; kind = :control)
            CHECK_ONLY && (ctl = ctl[1:min(2, length(ctl))])
            run_outputs(roles, m, vcat(cases, ctl), joinpath(outdir, "outputs-$(VERSION)"))
        elseif g == "admm_reference"
            run_admm_reference(m, cases, roles, refdir)
        elseif g == "admm_untimed"
            run_admm_untimed(m, cases, roles, refdir)
        else
            kstar = g == "admm" ? load_kstar(refdir) : nothing
            for c in cases
                println("group ", g, ": ", c.id)
                d = case_data(m, c)
                try
                    g == "first_use" && run_first_use(writer, roles, c, d)
                    g == "warm" && run_warm(writer, roles, c, d)
                    g == "penalty" && run_penalty(writer, roles, c, d)
                    g == "alloc" && run_alloc(writer, roles, c, d)
                    g == "admm" && run_admm(writer, roles, c, d, kstar, refdir)
                catch e
                    CHECK_ONLY && rethrow()
                    msg = oneline(first(split(sprint(showerror, e), '\n')))
                    println("  ERROR in ", g, " ", c.id, ": ", msg)
                    emit(writer, c.id, "error", g * ": " * msg, 0, 0, "-", NaN)
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
