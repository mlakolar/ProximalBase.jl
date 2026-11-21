# Shared definitions of the wide least-squares study (#1103), included by generate.jl, harness.jl and
# accuracy.jl: the package-source modules and the Cholesky prototypes (each compiled twice, for the A/A controls),
# the roles, the manifest of the binary inputs, the BigFloat stationarity residual and the ADMM loop.
#
# The package source is loaded with Base.include into the modules PkgS (role S) and PkgSControl (role S').
# PB_SOURCE names the root of the tree (default: the root of the repository that holds this folder). Setting
# PB_LOAD_PACKAGE=0 before the include skips the package modules and the prototypes (accuracy.jl needs neither).
using LinearAlgebra, Random, Printf, Statistics, SHA, Dates, Libdl
using LinearAlgebra: BLAS

ENDIAN_BOM == 0x04030201 || error("the binary files are little-endian; this host is not")

const STUDY_DIR = @__DIR__
const REPO_ROOT = abspath(joinpath(STUDY_DIR, "..", "..", ".."))
const PB_SOURCE = abspath(get(ENV, "PB_SOURCE", REPO_ROOT))

env_string(name, default) = get(ENV, name, default)
env_int(name, default) = parse(Int, get(ENV, name, string(default)))
env_flag(name) = get(ENV, name, "0") == "1"

###########################################################################################################
# Package source modules and the Cholesky prototypes
###########################################################################################################

module PkgS end
module PkgSControl end
module Proto end
module ProtoControl end

# The three observation-space Cholesky prototypes, with ρ = 1/γ as in LeastSquaresLoss.
#   CB (Boyd et al., lasso.m, fat branch): caches X'y and the factor L of I + XX'/ρ;
#       q = X'y + ρv, β = q/ρ - X'((I + XX'/ρ) \ Xq)/ρ².
#   CD (residual form): factor of I + γXX'; a = (I + γXX') \ (y - Xv), β = v + γX'a.
#   CW (whitened form): LL' = I + γXX', W = L⁻¹X, c = L⁻¹y; β = v + γW'(c - Wv).
# The solves use BLAS and LAPACK calls on preallocated buffers; prox!(β, c, v) allocates nothing in the warmed
# state when its element types agree. refactor!(c, γ) re-shifts and refactors from the cached XX' (CW also
# recomputes W and c). The 4-argument prox! checks that γ is the step of the current factorization.
# build_parts returns the object and the seconds of each setup step.
const PROTOTYPE_SOURCE = quote
    using LinearAlgebra
    using LinearAlgebra: BLAS, LAPACK

    # the text of the finiteness error of the cached SVD in LeastSquaresLoss
    const FINITE_MESSAGE = "design matrix X must have finite entries in the SVD computation type"

    function check_design(X::Matrix{T}, y::Vector{T}) where {T<:AbstractFloat}
        size(X, 1) == length(y) || throw(DimensionMismatch("X and y have different numbers of rows"))
        all(isfinite, X) || throw(ArgumentError(FINITE_MESSAGE))
        nothing
    end

    function check_step(γ)
        (isfinite(γ) && γ > 0) || throw(DomainError(γ, "the step must be finite and positive"))
        nothing
    end

    # the lower triangle of XX'
    function gram!(XXt::Matrix{T}, X::Matrix{T}) where {T}
        BLAS.syrk!('L', 'N', one(T), X, zero(T), XXt)
        XXt
    end

    # L = I + c XX' (shift_mul!) or I + XX'/ρ (shift_div!) in the lower triangle
    function shift_mul!(L::Matrix{T}, XXt::Matrix{T}, c::T) where {T}
        n = size(L, 1)
        @inbounds for j in 1:n
            for i in j:n
                L[i, j] = c * XXt[i, j]
            end
            L[j, j] += one(T)
        end
        L
    end
    function shift_div!(L::Matrix{T}, XXt::Matrix{T}, ρ::T) where {T}
        n = size(L, 1)
        @inbounds for j in 1:n
            for i in j:n
                L[i, j] = XXt[i, j] / ρ
            end
            L[j, j] += one(T)
        end
        L
    end

    function cholesky_lower!(L::Matrix{T}) where {T}
        _, info = LAPACK.potrf!('L', L)
        info == 0 || throw(PosDefException(Int(info)))
        L
    end

    # a := (L L') \ a
    function solve_factor!(L::Matrix{T}, a::Vector{T}) where {T}
        BLAS.trsv!('L', 'N', 'N', L, a)
        BLAS.trsv!('L', 'T', 'N', L, a)
        a
    end

    ################ CB
    mutable struct CB{T<:AbstractFloat}
        X::Matrix{T}
        y::Vector{T}
        Xty::Vector{T}
        XXt::Matrix{T}
        L::Matrix{T}
        a::Vector{T}
        q::Vector{T}
        γ::T
        ρ::T
        ρ2::T
    end
    function factor!(c::CB{T}, γ::Real) where {T}
        check_step(γ)
        c.γ = convert(T, γ)
        c.ρ = inv(c.γ)
        c.ρ2 = c.ρ * c.ρ
        cholesky_lower!(shift_div!(c.L, c.XXt, c.ρ))
        c
    end
    refactor!(c::CB, γ::Real) = factor!(c, γ)
    function build_parts(::Type{CB}, X::Matrix{T}, y::Vector{T}, γ::Real) where {T<:AbstractFloat}
        t0 = time_ns()
        check_design(X, y)
        n, p = size(X)
        c = CB{T}(X, y, Vector{T}(undef, p), Matrix{T}(undef, n, n), Matrix{T}(undef, n, n),
                  Vector{T}(undef, n), Vector{T}(undef, p), zero(T), zero(T), zero(T))
        t1 = time_ns()
        gram!(c.XXt, X)
        t2 = time_ns()
        BLAS.gemv!('T', one(T), X, y, zero(T), c.Xty)
        t3 = time_ns()
        factor!(c, γ)
        t4 = time_ns()
        c, (; check_alloc = (t1 - t0) * 1e-9, gram = (t2 - t1) * 1e-9, rhs = (t3 - t2) * 1e-9, factor = (t4 - t3) * 1e-9)
    end
    CB(X::Matrix{T}, y::Vector{T}, γ::Real) where {T<:AbstractFloat} = build_parts(CB, X, y, γ)[1]
    function prox!(β::Vector{T}, c::CB{T}, v::Vector{T}) where {T}
        q, a, ρ, ρ2 = c.q, c.a, c.ρ, c.ρ2
        @. q = c.Xty + ρ * v
        BLAS.gemv!('N', one(T), c.X, q, zero(T), a)
        solve_factor!(c.L, a)
        BLAS.gemv!('T', one(T), c.X, a, zero(T), β)
        @. β = q / ρ - β / ρ2
        β
    end
    prox!(β::Vector{T}, c::CB{T}, v::Vector{T}, γ::Real) where {T} =
        (convert(T, γ) == c.γ || throw(ArgumentError("the step differs from that of the factorization")); prox!(β, c, v))

    ################ CD
    mutable struct CD{T<:AbstractFloat}
        X::Matrix{T}
        y::Vector{T}
        XXt::Matrix{T}
        L::Matrix{T}
        r::Vector{T}
        γ::T
    end
    function factor!(c::CD{T}, γ::Real) where {T}
        check_step(γ)
        c.γ = convert(T, γ)
        cholesky_lower!(shift_mul!(c.L, c.XXt, c.γ))
        c
    end
    refactor!(c::CD, γ::Real) = factor!(c, γ)
    function build_parts(::Type{CD}, X::Matrix{T}, y::Vector{T}, γ::Real) where {T<:AbstractFloat}
        t0 = time_ns()
        check_design(X, y)
        n = size(X, 1)
        c = CD{T}(X, y, Matrix{T}(undef, n, n), Matrix{T}(undef, n, n), Vector{T}(undef, n), zero(T))
        t1 = time_ns()
        gram!(c.XXt, X)
        t2 = time_ns()
        factor!(c, γ)
        t3 = time_ns()
        c, (; check_alloc = (t1 - t0) * 1e-9, gram = (t2 - t1) * 1e-9, factor = (t3 - t2) * 1e-9)
    end
    CD(X::Matrix{T}, y::Vector{T}, γ::Real) where {T<:AbstractFloat} = build_parts(CD, X, y, γ)[1]
    function prox!(β::Vector{T}, c::CD{T}, v::Vector{T}) where {T}
        r = c.r
        copyto!(r, c.y)
        BLAS.gemv!('N', -one(T), c.X, v, one(T), r)
        solve_factor!(c.L, r)
        β === v || copyto!(β, v)
        BLAS.gemv!('T', c.γ, c.X, r, one(T), β)
        β
    end
    prox!(β::Vector{T}, c::CD{T}, v::Vector{T}, γ::Real) where {T} =
        (convert(T, γ) == c.γ || throw(ArgumentError("the step differs from that of the factorization")); prox!(β, c, v))

    ################ CW
    mutable struct CW{T<:AbstractFloat}
        X::Matrix{T}
        y::Vector{T}
        XXt::Matrix{T}
        L::Matrix{T}
        W::Matrix{T}
        c::Vector{T}
        r::Vector{T}
        γ::T
    end
    function factor!(c::CW{T}, γ::Real) where {T}
        check_step(γ)
        c.γ = convert(T, γ)
        cholesky_lower!(shift_mul!(c.L, c.XXt, c.γ))
        c
    end
    # W = L⁻¹X and c = L⁻¹y
    function whiten!(c::CW{T}) where {T}
        copyto!(c.W, c.X)
        BLAS.trsm!('L', 'L', 'N', 'N', one(T), c.L, c.W)
        copyto!(c.c, c.y)
        BLAS.trsv!('L', 'N', 'N', c.L, c.c)
        c
    end
    refactor!(c::CW, γ::Real) = whiten!(factor!(c, γ))
    function build_parts(::Type{CW}, X::Matrix{T}, y::Vector{T}, γ::Real) where {T<:AbstractFloat}
        t0 = time_ns()
        check_design(X, y)
        n, p = size(X)
        c = CW{T}(X, y, Matrix{T}(undef, n, n), Matrix{T}(undef, n, n), Matrix{T}(undef, n, p), Vector{T}(undef, n),
                  Vector{T}(undef, n), zero(T))
        t1 = time_ns()
        gram!(c.XXt, X)
        t2 = time_ns()
        factor!(c, γ)
        t3 = time_ns()
        whiten!(c)
        t4 = time_ns()
        c, (; check_alloc = (t1 - t0) * 1e-9, gram = (t2 - t1) * 1e-9, factor = (t3 - t2) * 1e-9, whiten = (t4 - t3) * 1e-9)
    end
    CW(X::Matrix{T}, y::Vector{T}, γ::Real) where {T<:AbstractFloat} = build_parts(CW, X, y, γ)[1]
    function prox!(β::Vector{T}, c::CW{T}, v::Vector{T}) where {T}
        r = c.r
        copyto!(r, c.c)
        BLAS.gemv!('N', -one(T), c.W, v, one(T), r)
        β === v || copyto!(β, v)
        BLAS.gemv!('T', c.γ, c.W, r, one(T), β)
        β
    end
    prox!(β::Vector{T}, c::CW{T}, v::Vector{T}, γ::Real) where {T} =
        (convert(T, γ) == c.γ || throw(ArgumentError("the step differs from that of the factorization")); prox!(β, c, v))
end

if get(ENV, "PB_LOAD_PACKAGE", "1") == "1"
    for m in (PkgS, PkgSControl)
        Base.include(m, joinpath(PB_SOURCE, "src", "ProximalBase.jl"))
    end
    for m in (Proto, ProtoControl)
        Core.eval(m, PROTOTYPE_SOURCE)
    end
end

###########################################################################################################
# Roles
###########################################################################################################

# One timed implementation: ctor(X, y, γ) builds the object, px(β, object, v, γ) is the proximal call,
# refactor(object, γ) prepares the object for a new step and prepare(object) completes a factorization that
# the constructor leaves for the first call. family is "S", "C_B", "C_D" or "C_W"; control marks the copy
# that is compiled separately (the A/A control of its family).
struct Role{C,P,R,Q}
    name::String
    family::String
    control::Bool
    ctor::C
    px::P
    refactor::R
    prepare::Q
end

# L is the LeastSquaresLoss or the prototype type of the role, held as a type parameter so that the result
# type of the constructor is inferred.
struct SCtor{L} end
(::SCtor{L})(X, y, γ) where {L} = L(y, X)
struct PCtor{L} end
(::PCtor{L})(X, y, γ) where {L} = L(X, y, γ)
struct NoRefactor end
(::NoRefactor)(obj, γ) = obj

function build_roles()
    function s_role(M::Module, name, control)
        PB = getfield(M, :ProximalBase)
        Role(name, "S", control, SCtor{getfield(PB, :LeastSquaresLoss)}(), getfield(PB, :prox!), NoRefactor(),
             getfield(PB, :_svd_cache!))
    end
    function c_role(M::Module, kind::Symbol, name, control)
        Role(name, "C_" * string(kind)[2:end], control, PCtor{getfield(M, kind)}(), getfield(M, :prox!),
             getfield(M, :refactor!), identity)
    end
    [s_role(PkgS, "S", false), s_role(PkgSControl, "S'", true),
     c_role(Proto, :CB, "C_B", false), c_role(ProtoControl, :CB, "C_B'", true),
     c_role(Proto, :CD, "C_D", false), c_role(ProtoControl, :CD, "C_D'", true),
     c_role(Proto, :CW, "C_W", false), c_role(ProtoControl, :CW, "C_W'", true)]
end

# The roles in the order of a process. S_first sets up and times the roles S, S', C_B, C_B', C_D, C_D', C_W,
# C_W' in this order and C_first the C roles before S and S'; the order within a round alternates.
function ordered_roles(roles, order::AbstractString)
    names = order == "S_first" ? ["S", "S'", "C_B", "C_B'", "C_D", "C_D'", "C_W", "C_W'"] :
            order == "C_first" ? ["C_B", "C_B'", "C_D", "C_D'", "C_W", "C_W'", "S", "S'"] :
            error("PB_ORDER must be S_first or C_first")
    [roles[findfirst(r -> r.name == nm, roles)] for nm in names]
end
round_order(roles, r::Integer) = isodd(r) ? roles : reverse(roles)

###########################################################################################################
# The manifest of the binary inputs
###########################################################################################################

struct Case
    id::String
    kind::Symbol            # :main or :control
    T::DataType
    design::String
    response::String
    n::Int
    p::Int
    seed::Int
    timed::Bool
    lasso::Bool
    gammas::Vector{Float64}
    lambda_max::Float64
    arrays::Dict{String,NamedTuple{(:file, :shape, :sha256, :note), Tuple{String,Vector{Int},String,String}}}
end

struct Manifest
    dir::String
    header::Dict{String,String}
    cases::Vector{Case}
    sha256::String
end

const ELTYPES = Dict("Float64" => Float64, "Float32" => Float32)
const MANIFEST_COLUMNS = ["case", "kind", "eltype", "design", "response", "n", "p", "seed", "timed", "lasso", "gammas",
                          "lambda_max", "array", "file", "shape", "sha256", "note"]

file_sha256(path) = bytes2hex(open(sha256, path))
case_file(id) = replace(id, '/' => '_', '=' => '-', '+' => 'p')

function load_manifest(dir::AbstractString)
    path = joinpath(dir, "inputs-index.tsv")
    isfile(path) || error("no inputs-index.tsv in $dir")
    header = Dict{String,String}()
    cols = nothing
    order = String[]
    byid = Dict{String,Vector{Dict{String,String}}}()
    for line in eachline(path)
        isempty(line) && continue
        if startswith(line, "#")
            k, _, v = partition_at(line[2:end], ":")
            header[strip(k)] = strip(v)
        elseif cols === nothing
            cols = String.(split(line, '\t'))
        else
            f = String.(split(line, '\t'))
            length(f) == length(cols) || error("malformed row in inputs-index.tsv: $line")
            row = Dict(zip(cols, f))
            haskey(byid, row["case"]) || (push!(order, row["case"]); byid[row["case"]] = Dict{String,String}[])
            push!(byid[row["case"]], row)
        end
    end
    cases = Case[]
    for id in order
        rows = byid[id]
        r = rows[1]
        arrays = Dict{String,NamedTuple{(:file, :shape, :sha256, :note), Tuple{String,Vector{Int},String,String}}}()
        for q in rows
            arrays[q["array"]] = (file = q["file"], shape = parse.(Int, split(q["shape"], 'x')), sha256 = q["sha256"],
                                  note = q["note"])
        end
        γs = isempty(r["gammas"]) ? Float64[] : parse.(Float64, split(r["gammas"], ','))
        push!(cases, Case(id, Symbol(r["kind"]), ELTYPES[r["eltype"]], r["design"], r["response"], parse(Int, r["n"]),
                          parse(Int, r["p"]), parse(Int, r["seed"]), r["timed"] == "true", r["lasso"] == "true", γs,
                          isempty(r["lambda_max"]) ? NaN : parse(Float64, r["lambda_max"]), arrays))
    end
    Manifest(String(dir), header, cases, file_sha256(path))
end

function partition_at(s::AbstractString, sep::AbstractString)
    r = findfirst(sep, s)
    r === nothing ? (s, "", "") : (s[1:first(r)-1], sep, s[last(r)+1:end])
end

const VERIFIED_FILES = Dict{String,Bool}()

# The array `name` of a case, read after its file has been checked against the SHA-256 of the manifest (once per
# process and file). A mismatch or a size that differs from the manifest stops the process.
function read_input(m::Manifest, c::Case, name::AbstractString)
    a = c.arrays[name]
    path = joinpath(m.dir, a.file)
    isfile(path) || error("missing input file $(a.file)")
    if !haskey(VERIFIED_FILES, path)
        got = file_sha256(path)
        got == a.sha256 || error("SHA-256 mismatch for $(a.file): manifest $(a.sha256), file $got")
        VERIFIED_FILES[path] = true
    end
    filesize(path) == prod(a.shape) * sizeof(c.T) || error("size of $(a.file) differs from its shape $(a.shape)")
    A = Array{c.T}(undef, a.shape...)
    read!(path, A)
    A
end

function select_cases(m::Manifest; kind = :main, timed = nothing, lasso = nothing, T = nothing)
    rx = Regex(env_string("PB_FILTER", ""))
    filter(m.cases) do c
        c.kind === kind && occursin(rx, c.id) && (timed === nothing || c.timed == timed) &&
            (lasso === nothing || c.lasso == lasso) && (T === nothing || c.T === T)
    end
end

###########################################################################################################
# The stationarity residual in BigFloat
###########################################################################################################

# η = ‖X'(Xβ - y) + ρ(β - v)‖∞ / (‖X'‖∞‖Xβ‖∞ + ‖X'y‖∞ + ρ(‖β‖∞ + ‖v‖∞)), with ρ = 1/big(T(γ)), evaluated from the
# stored arrays with BigFloat arithmetic of `prec` bits. A β with a nonfinite entry has η = Inf.
function eta_exact(X::AbstractMatrix{T}, y::AbstractVector{T}, v::AbstractVector{T}, γ::Real, β::AbstractVector{T};
                   prec::Int = 256) where {T<:AbstractFloat}
    all(isfinite, β) || return Inf
    setprecision(BigFloat, prec) do
        Xb, yb, vb, βb = BigFloat.(X), BigFloat.(y), BigFloat.(v), BigFloat.(β)
        ρ = 1 / big(T(γ))
        Xβ = Xb * βb
        r = Xb' * (Xβ - yb) + ρ * (βb - vb)
        Xty = Xb' * yb
        Xt_inf = maximum(sum(abs.(Xb); dims = 1))
        den = Xt_inf * maximum(abs.(Xβ)) + maximum(abs.(Xty)) + ρ * (maximum(abs.(βb)) + maximum(abs.(vb)))
        num = maximum(abs.(r))
        Float64(iszero(den) ? (iszero(num) ? zero(num) : big(Inf)) : num / den)
    end
end

###########################################################################################################
# The ADMM loop of the lasso (Boyd et al., §6.4 and the lasso.m code), generic in the element type
###########################################################################################################

mutable struct AdmmWork{T}
    x::Vector{T}
    z::Vector{T}
    u::Vector{T}
    v::Vector{T}
    zold::Vector{T}
    r::Vector{T}            # Xz - y
    g::Vector{T}            # X'(Xz - y)
end
AdmmWork(::Type{T}, n::Int, p::Int) where {T} =
    AdmmWork{T}(zeros(T, p), zeros(T, p), zeros(T, p), zeros(T, p), zeros(T, p), zeros(T, n), zeros(T, p))
function reset!(w::AdmmWork)
    for a in (w.x, w.z, w.u, w.v, w.zold, w.r, w.g)
        fill!(a, 0)
    end
    w
end

@inline soft(t, κ) = t > κ ? t - κ : (t < -κ ? t + κ : zero(t))

# One iteration with α = 1: v = z - u, the x-update px(x, obj, v, γ) with γ = 1/ρ, z = soft(x + u, κ) with
# κ = λ/ρ, u += x - z. It returns the sums of squares of x - z, z - zold, x, z and u.
function admm_step!(w::AdmmWork{T}, px::P, obj, γ::T, κ::T) where {T,P}
    x, z, u, v, zold = w.x, w.z, w.u, w.v, w.zold
    @inbounds for i in eachindex(v)
        v[i] = z[i] - u[i]
    end
    px(x, obj, v, γ)
    copyto!(zold, z)
    rr = ss = nx = nz = nu = 0.0
    @inbounds for i in eachindex(x)
        xi = x[i]
        zi = soft(xi + u[i], κ)
        ui = u[i] + (xi - zi)
        z[i] = zi
        u[i] = ui
        d = Float64(xi - zi)
        e = Float64(zi - zold[i])
        rr += d * d
        ss += e * e
        nx += Float64(xi)^2
        nz += Float64(zi)^2
        nu += Float64(ui)^2
    end
    (rr, ss, nx, nz, nu)
end

# Boyd's stopping rule: ‖x - z‖ < √p ε_abs + ε_rel max(‖x‖, ‖z‖) and ρ‖z - zold‖ < √p ε_abs + ε_rel ρ‖u‖
function boyd_converged(p::Int, ρ::Real, t::NTuple{5,Float64}; εabs = 1e-4, εrel = 1e-2)
    r, s = sqrt(t[1]), ρ * sqrt(t[2])
    r < sqrt(p) * εabs + εrel * max(sqrt(t[3]), sqrt(t[4])) && s < sqrt(p) * εabs + εrel * ρ * sqrt(t[5])
end

# The fixed-point residual ‖z - soft(z - X'(Xz - y), λ)‖∞ of the lasso at z and its objective
# ½‖Xz - y‖² + λ‖z‖₁ (two matrix-vector products; w.r and w.g are overwritten).
function lasso_measures!(w::AdmmWork{T}, X::Matrix{T}, y::Vector{T}, λ::T) where {T}
    z, r, g = w.z, w.r, w.g
    copyto!(r, y)
    BLAS.gemv!('N', one(T), X, z, -one(T), r)
    BLAS.gemv!('T', one(T), X, r, zero(T), g)
    res = 0.0
    l1 = 0.0
    @inbounds for i in eachindex(z)
        res = max(res, Float64(abs(z[i] - soft(z[i] - g[i], λ))))
        l1 += Float64(abs(z[i]))
    end
    rr = 0.0
    @inbounds for i in eachindex(r)
        rr += Float64(r[i])^2
    end
    (res, 0.5 * rr + Float64(λ) * l1)
end

const ADMM_CHECKPOINTS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000)

# An untimed run of one role: the iteration counts of Boyd's rule and of the stationarity target
# tight_tol·scale (0 when not reached within cap), and the trajectory at the checkpoints and at those counts as
# rows (k, r, s, stationarity, objective). The run stops when both are reached or at cap.
function admm_untimed(role::Role, X::Matrix{T}, y::Vector{T}, λ::T, ρ::Real; tight_tol::Real, scale::Real, cap::Int,
                      εabs = 1e-4, εrel = 1e-2) where {T}
    n, p = size(X)
    γ, κ = inv(T(ρ)), λ / T(ρ)
    obj = role.ctor(X, y, γ)
    role.prepare(obj)
    w = AdmmWork(T, n, p)
    k_boyd = 0
    k_tight = 0
    rows = NTuple{5,Float64}[]
    final = (0.0, 0.0)
    for k in 1:cap
        t = admm_step!(w, role.px, obj, γ, κ)
        res, objective = lasso_measures!(w, X, y, λ)
        final = (res, objective)
        boyd = k_boyd == 0 && boyd_converged(p, ρ, t; εabs, εrel)
        boyd && (k_boyd = k)
        tight = k_tight == 0 && res <= tight_tol * scale
        tight && (k_tight = k)
        if boyd || tight || k in ADMM_CHECKPOINTS
            push!(rows, (Float64(k), sqrt(t[1]), ρ * sqrt(t[2]), res, objective))
        end
        k_tight != 0 && k_boyd != 0 && break
    end
    (; k_boyd, k_tight, rows, final_stationarity = final[1], final_objective = final[2])
end

# S-ADMM to the stationarity tol·scale, checked every `every` iterations, for the reference optimum.
function admm_reference(role::Role, X::Matrix{T}, y::Vector{T}, λ::T, ρ::Real; tol::Real, scale::Real, cap::Int,
                        every::Int = 10) where {T}
    n, p = size(X)
    γ, κ = inv(T(ρ)), λ / T(ρ)
    obj = role.ctor(X, y, γ)
    role.prepare(obj)
    w = AdmmWork(T, n, p)
    res, objective, k = Inf, Inf, 0
    converged = false
    while k < cap
        for _ in 1:every
            admm_step!(w, role.px, obj, γ, κ)
        end
        k += every
        res, objective = lasso_measures!(w, X, y, λ)
        if res <= tol * scale
            converged = true
            break
        end
    end
    (; iterations = k, converged, stationarity = res, objective, z = copy(w.z))
end

# v = z - u after each of the iteration counts ks of an ADMM run (the proximal inputs that the solver reaches)
function admm_record(role::Role, X::Matrix{T}, y::Vector{T}, λ::T, ρ::Real, ks) where {T}
    n, p = size(X)
    γ, κ = inv(T(ρ)), λ / T(ρ)
    obj = role.ctor(X, y, γ)
    role.prepare(obj)
    w = AdmmWork(T, n, p)
    out = Vector{T}[]
    for k in 1:maximum(ks)
        admm_step!(w, role.px, obj, γ, κ)
        k in ks && push!(out, w.z .- w.u)
    end
    out
end
