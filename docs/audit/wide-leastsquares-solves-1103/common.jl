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

# A commit identity from the environment: 7 to 40 lowercase hex digits, "unset" when absent and not required. The
# string is a label that the caller supplies; no check can compare it with the files in a copy that has no .git.
function env_sha(name; required::Bool)
    s = get(ENV, name, "")
    if isempty(s)
        required && error("$name must name a commit (7 to 40 lowercase hex digits)")
        return "unset"
    end
    occursin(r"^[0-9a-f]{7,40}$", s) || error("$name must be 7 to 40 lowercase hex digits, got $(repr(s))")
    s
end

###########################################################################################################
# Package source modules and the Cholesky prototypes
###########################################################################################################

module PkgS end
module PkgSControl end
module Proto end
module ProtoControl end

# The three observation-space Cholesky prototypes. With γT = T(γ) and ρ = inv(γT) as in LeastSquaresLoss, each
# factorization computes the scalar inv(ρ) once into a local, as lasso.m forms 1/ρ once, forms the shift I + XX'/ρ
# with it in the lower triangle and stores it as invρ; all three use the same shift:
#   CB (Boyd et al., lasso.m, fat branch): caches X'y and the factor L of I + XX'/ρ;
#       q = X'y + ρv, β = q/ρ - X'((I + XX'/ρ) \ Xq)/ρ².
#   CD (residual form): a = (I + XX'/ρ) \ (y - Xv), β = v + X'a/ρ.
#   CW (whitened form): LL' = I + XX'/ρ, W = L⁻¹X, c = L⁻¹y; β = v + W'(c - Wv)/ρ.
# The solves use BLAS and LAPACK calls on preallocated buffers. For Float32 or Float64 data and a Float32 or Float64
# step, the warmed prox!(β, c, v) of a factorized object allocated no bytes in the probes of harness.jl. refactor!(c, γ)
# re-shifts and refactors from the cached XX' (CW also recomputes W and c). The 4-argument prox! checks that γ
# converts to the step of the current factorization.
# The constructors are the timed setup. breakdown(Type, X, y, γ) runs the same kernels (check_design, alloc, gram!,
# rhs!, factor!, whiten!) with a time_ns() reading between them and returns the object and the seconds of each step.
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

    # the step converted to T and ρ = inv(T(γ)), both finite and positive
    function penalty_of(::Type{T}, γ::Real) where {T<:AbstractFloat}
        γT = convert(T, γ)
        ρ = inv(γT)
        (isfinite(γT) && γT > 0 && isfinite(ρ) && ρ > 0) ||
            throw(DomainError(γ, "the step must convert to a finite positive $T with a finite positive reciprocal"))
        γT, ρ
    end

    # the lower triangle of XX'
    function gram!(XXt::Matrix{T}, X::Matrix{T}) where {T}
        BLAS.syrk!('L', 'N', one(T), X, zero(T), XXt)
        XXt
    end

    # L = I + c XX' in the lower triangle
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
        invρ::T
        ρ2::T
    end
    function alloc(::Type{CB}, X::Matrix{T}, y::Vector{T}) where {T<:AbstractFloat}
        n, p = size(X)
        CB{T}(X, y, Vector{T}(undef, p), Matrix{T}(undef, n, n), Matrix{T}(undef, n, n), Vector{T}(undef, n),
              Vector{T}(undef, p), zero(T), zero(T), zero(T), zero(T))
    end
    rhs!(c::CB{T}) where {T} = (BLAS.gemv!('T', one(T), c.X, c.y, zero(T), c.Xty); c)
    function factor!(c::CB{T}, γ::Real) where {T}
        γT, ρ = penalty_of(T, γ)
        invρ = inv(ρ)
        cholesky_lower!(shift_mul!(c.L, c.XXt, invρ))
        c.γ, c.ρ, c.invρ, c.ρ2 = γT, ρ, invρ, ρ * ρ
        c
    end
    refactor!(c::CB, γ::Real) = factor!(c, γ)
    function CB(X::Matrix{T}, y::Vector{T}, γ::Real) where {T<:AbstractFloat}
        check_design(X, y)
        c = alloc(CB, X, y)
        gram!(c.XXt, X)
        rhs!(c)
        factor!(c, γ)
    end
    function breakdown(::Type{CB}, X::Matrix{T}, y::Vector{T}, γ::Real) where {T<:AbstractFloat}
        t0 = time_ns()
        check_design(X, y)
        c = alloc(CB, X, y)
        t1 = time_ns()
        gram!(c.XXt, X)
        t2 = time_ns()
        rhs!(c)
        t3 = time_ns()
        factor!(c, γ)
        t4 = time_ns()
        c, (; check_alloc = (t1 - t0) * 1e-9, gram = (t2 - t1) * 1e-9, rhs = (t3 - t2) * 1e-9, factor = (t4 - t3) * 1e-9)
    end
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
        ρ::T
        invρ::T
    end
    function alloc(::Type{CD}, X::Matrix{T}, y::Vector{T}) where {T<:AbstractFloat}
        n = size(X, 1)
        CD{T}(X, y, Matrix{T}(undef, n, n), Matrix{T}(undef, n, n), Vector{T}(undef, n), zero(T), zero(T), zero(T))
    end
    function factor!(c::CD{T}, γ::Real) where {T}
        γT, ρ = penalty_of(T, γ)
        invρ = inv(ρ)
        cholesky_lower!(shift_mul!(c.L, c.XXt, invρ))
        c.γ, c.ρ, c.invρ = γT, ρ, invρ
        c
    end
    refactor!(c::CD, γ::Real) = factor!(c, γ)
    function CD(X::Matrix{T}, y::Vector{T}, γ::Real) where {T<:AbstractFloat}
        check_design(X, y)
        c = alloc(CD, X, y)
        gram!(c.XXt, X)
        factor!(c, γ)
    end
    function breakdown(::Type{CD}, X::Matrix{T}, y::Vector{T}, γ::Real) where {T<:AbstractFloat}
        t0 = time_ns()
        check_design(X, y)
        c = alloc(CD, X, y)
        t1 = time_ns()
        gram!(c.XXt, X)
        t2 = time_ns()
        factor!(c, γ)
        t3 = time_ns()
        c, (; check_alloc = (t1 - t0) * 1e-9, gram = (t2 - t1) * 1e-9, factor = (t3 - t2) * 1e-9)
    end
    function prox!(β::Vector{T}, c::CD{T}, v::Vector{T}) where {T}
        r = c.r
        copyto!(r, c.y)
        BLAS.gemv!('N', -one(T), c.X, v, one(T), r)
        solve_factor!(c.L, r)
        β === v || copyto!(β, v)
        BLAS.gemv!('T', c.invρ, c.X, r, one(T), β)
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
        ρ::T
        invρ::T
    end
    function alloc(::Type{CW}, X::Matrix{T}, y::Vector{T}) where {T<:AbstractFloat}
        n, p = size(X)
        CW{T}(X, y, Matrix{T}(undef, n, n), Matrix{T}(undef, n, n), Matrix{T}(undef, n, p), Vector{T}(undef, n),
              Vector{T}(undef, n), zero(T), zero(T), zero(T))
    end
    function factor!(c::CW{T}, γ::Real) where {T}
        γT, ρ = penalty_of(T, γ)
        invρ = inv(ρ)
        cholesky_lower!(shift_mul!(c.L, c.XXt, invρ))
        c.γ, c.ρ, c.invρ = γT, ρ, invρ
        c
    end
    # W = L⁻¹X (a copy of X and trsm) and c = L⁻¹y (a copy of y and a triangular solve)
    function whiten!(c::CW{T}) where {T}
        copyto!(c.W, c.X)
        BLAS.trsm!('L', 'L', 'N', 'N', one(T), c.L, c.W)
        copyto!(c.c, c.y)
        BLAS.trsv!('L', 'N', 'N', c.L, c.c)
        c
    end
    refactor!(c::CW, γ::Real) = whiten!(factor!(c, γ))
    function CW(X::Matrix{T}, y::Vector{T}, γ::Real) where {T<:AbstractFloat}
        check_design(X, y)
        c = alloc(CW, X, y)
        gram!(c.XXt, X)
        factor!(c, γ)
        whiten!(c)
    end
    function breakdown(::Type{CW}, X::Matrix{T}, y::Vector{T}, γ::Real) where {T<:AbstractFloat}
        t0 = time_ns()
        check_design(X, y)
        c = alloc(CW, X, y)
        t1 = time_ns()
        gram!(c.XXt, X)
        t2 = time_ns()
        factor!(c, γ)
        t3 = time_ns()
        whiten!(c)
        t4 = time_ns()
        c, (; check_alloc = (t1 - t0) * 1e-9, gram = (t2 - t1) * 1e-9, factor = (t3 - t2) * 1e-9, whiten = (t4 - t3) * 1e-9)
    end
    function prox!(β::Vector{T}, c::CW{T}, v::Vector{T}) where {T}
        r = c.r
        copyto!(r, c.c)
        BLAS.gemv!('N', -one(T), c.W, v, one(T), r)
        β === v || copyto!(β, v)
        BLAS.gemv!('T', c.invρ, c.W, r, one(T), β)
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

const ArrayInfo = NamedTuple{(:file, :shape, :sha256, :note), Tuple{String,Vector{Int},String,String}}

# gammas: the steps of the timing groups; acc_gammas: the steps whose prox outputs are evaluated (a superset of gammas);
# sigma_max: the largest singular value of the stored X, computed from its Float64 values
Base.@kwdef struct Case
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
    acc_gammas::Vector{Float64}
    lambda_max::Float64
    sigma_max::Float64
    arrays::Dict{String,ArrayInfo}
end

struct Manifest
    dir::String
    header::Dict{String,String}
    cases::Vector{Case}
    sha256::String
end

const ELTYPES = Dict("Float64" => Float64, "Float32" => Float32)
const MANIFEST_FORMAT = "pb1103-inputs-2"
const MANIFEST_COLUMNS = ["case", "kind", "eltype", "design", "response", "n", "p", "seed", "timed", "lasso", "gammas",
                          "acc_gammas", "lambda_max", "sigma_max", "array", "file", "shape", "sha256", "note"]

file_sha256(path) = bytes2hex(sha256(read(path)))
case_file(id) = replace(id, '/' => '_', '=' => '-', '+' => 'p')
parse_floats(s) = isempty(s) ? Float64[] : parse.(Float64, split(s, ','))

function partition_at(s::AbstractString, sep::AbstractString)
    r = findfirst(sep, s)
    r === nothing ? (s, "", "") : (s[1:first(r)-1], sep, s[last(r)+1:end])
end

# The rows of a tab-separated file of this study: the `# key: value` header, the column names and the rows as
# dictionaries. A file without the closing line `# complete: yes` is an interrupted write and is rejected, and so is a
# file that repeats a header key.
function read_tsv(path::AbstractString)
    isfile(path) || error("missing file $path")
    parse_tsv(read(path, String), path)
end
# the same from the text of a file (name is used in the messages)
function parse_tsv(text::AbstractString, path::AbstractString)
    lines = filter!(!isempty, split(text, '\n'))
    !isempty(lines) && lines[end] == "# complete: yes" || error("$path is incomplete (no closing `# complete: yes` line)")
    header = Dict{String,String}()
    cols = nothing
    rows = Vector{Dict{String,String}}()
    for line in lines
        if startswith(line, "#")
            k, _, v = partition_at(line[2:end], ":")
            key = String(strip(k))
            haskey(header, key) && error("$path repeats the header key $(repr(key))")
            header[key] = strip(v)
        elseif cols === nothing
            cols = String.(split(line, '\t'))
        else
            f = String.(split(line, '\t'))
            length(f) == length(cols) || error("$path: malformed row $(repr(line))")
            push!(rows, Dict(zip(cols, f)))
        end
    end
    header, cols === nothing ? String[] : cols, rows
end

function load_manifest(dir::AbstractString)
    path = joinpath(dir, "inputs-index.tsv")
    isfile(path) || error("missing file $path")
    bytes = read(path)
    sha = bytes2hex(sha256(bytes))                   # the digest of the bytes that are parsed
    header, cols, rows = parse_tsv(String(copy(bytes)), path)
    get(header, "format", "") == MANIFEST_FORMAT || error("$path: format $(get(header, "format", "none")), expected $MANIFEST_FORMAT")
    cols == MANIFEST_COLUMNS || error("$path: unexpected columns")
    order = String[]
    byid = Dict{String,Vector{Dict{String,String}}}()
    for row in rows
        haskey(byid, row["case"]) || (push!(order, row["case"]); byid[row["case"]] = Dict{String,String}[])
        push!(byid[row["case"]], row)
    end
    cases = Case[]
    meta = [c for c in MANIFEST_COLUMNS if !(c in ("array", "file", "shape", "sha256", "note"))]
    for id in order
        rs = byid[id]
        r = rs[1]
        all(q -> all(k -> q[k] == r[k], meta), rs) || error("$path: the rows of case $id disagree in their case fields")
        sort([q["array"] for q in rs]) == ["V", "X", "y"] || error("$path: case $id needs exactly the arrays X, y and V")
        arrays = Dict{String,ArrayInfo}()
        for q in rs
            haskey(arrays, q["array"]) && error("$path: duplicate array $(q["array"]) of case $id")
            arrays[q["array"]] = (file = q["file"], shape = parse.(Int, split(q["shape"], 'x')), sha256 = q["sha256"],
                                  note = q["note"])
        end
        all(k -> haskey(arrays, k), ("X", "y", "V")) || error("$path: case $id lacks one of X, y, V")
        haskey(ELTYPES, r["eltype"]) || error("$path: unknown element type $(r["eltype"])")
        n, p = parse(Int, r["n"]), parse(Int, r["p"])
        arrays["X"].shape == [n, p] && arrays["y"].shape == [n] && length(arrays["V"].shape) == 2 && arrays["V"].shape[1] == p ||
            error("$path: shapes of case $id disagree with n and p")
        γs, aγs = parse_floats(r["gammas"]), parse_floats(r["acc_gammas"])
        all(g -> g in aγs, γs) || error("$path: gammas of case $id are not among acc_gammas")
        push!(cases, Case(id = id, kind = Symbol(r["kind"]), T = ELTYPES[r["eltype"]], design = r["design"], response = r["response"],
                          n = n, p = p, seed = parse(Int, r["seed"]), timed = r["timed"] == "true", lasso = r["lasso"] == "true",
                          gammas = γs, acc_gammas = aγs, lambda_max = isempty(r["lambda_max"]) ? NaN : parse(Float64, r["lambda_max"]),
                          sigma_max = parse(Float64, r["sigma_max"]), arrays = arrays))
    end
    Manifest(String(dir), header, cases, sha)
end

# The array `name` of a case. The bytes of the file are read once, hashed, compared with the SHA-256 of the manifest
# and decoded from the same buffer, so that the array is the one that was verified. A mismatch or a size that differs
# from the manifest stops the process.
function read_input(m::Manifest, c::Case, name::AbstractString)
    a = c.arrays[name]
    path = joinpath(m.dir, a.file)
    isfile(path) || error("missing input file $(a.file)")
    bytes = read(path)
    got = bytes2hex(sha256(bytes))
    got == a.sha256 || error("SHA-256 mismatch for $(a.file): manifest $(a.sha256), file $got")
    length(bytes) == prod(a.shape) * sizeof(c.T) || error("size of $(a.file) differs from its shape $(a.shape)")
    A = Array{c.T}(undef, a.shape...)
    read!(IOBuffer(bytes), A)
    A
end

# The array of bytes read from a file after a check against an expected SHA-256 and byte count, decoded as type T.
function read_verified(path::AbstractString, expected_sha::AbstractString, ::Type{T}, shape) where {T}
    bytes = read(path)
    got = bytes2hex(sha256(bytes))
    got == expected_sha || error("SHA-256 mismatch for $(basename(path)): expected $expected_sha, file $got")
    length(bytes) == prod(shape) * sizeof(T) || error("size of $(basename(path)) differs from shape $shape")
    A = Array{T}(undef, shape...)
    read!(IOBuffer(bytes), A)
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

# Boyd's residual norms and tolerances from the sums of admm_step!: r = ‖x - z‖, s = ρ‖z - zold‖,
# eps_pri = √p ε_abs + ε_rel max(‖x‖, ‖z‖), eps_dual = √p ε_abs + ε_rel ρ‖u‖.
function boyd_measures(p::Int, ρ::Real, t::NTuple{5,Float64}; εabs = 1e-4, εrel = 1e-2)
    (sqrt(t[1]), ρ * sqrt(t[2]), sqrt(p) * εabs + εrel * max(sqrt(t[3]), sqrt(t[4])), sqrt(p) * εabs + εrel * ρ * sqrt(t[5]))
end
# Boyd's stopping rule r < eps_pri and s < eps_dual. The four values are computed; the second comparison is evaluated
# only when the first holds (&&).
function boyd_converged(p::Int, ρ::Real, t::NTuple{5,Float64}; εabs = 1e-4, εrel = 1e-2)
    r, s, epri, edual = boyd_measures(p, ρ, t; εabs, εrel)
    r < epri && s < edual
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

# k iterations of admm_step! on the work arrays, with boyd_converged evaluated after each iteration (O(p) work on the
# sums of admm_step!). The number of iterations that meet the rule is returned (the caller consumes it); it does not
# stop the loop. The sums of the last iteration are returned too.
function admm_loop!(w::AdmmWork{T}, px::P, obj, γ::T, κ::T, ρ::Real, k::Int; εabs = 1e-4, εrel = 1e-2) where {T,P}
    p = length(w.x)
    flags = 0
    last = (0.0, 0.0, 0.0, 0.0, 0.0)
    for _ in 1:k
        last = admm_step!(w, px, obj, γ, κ)
        flags += boyd_converged(p, ρ, last; εabs, εrel)
    end
    flags, last
end

# An untimed run of one role for at most cap iterations (exactly cap when a target is not reached). It records, for
# Boyd's rule and for the stationarity target tight_tol·scale, the status (:reached, or :capped when the run went
# through all cap iterations without meeting the target), the iteration count k (the first iteration that meets the
# target, or cap when capped) and the measures (r, s, eps_pri, eps_dual, stationarity, objective) at that iteration,
# and the trajectory
# at the checkpoints and at the iterations that reached a target. The run stops when both targets are reached.
function admm_untimed(role::Role, X::Matrix{T}, y::Vector{T}, λ::T, ρ::Real; tight_tol::Real, scale::Real, cap::Int,
                      εabs = 1e-4, εrel = 1e-2) where {T}
    n, p = size(X)
    γ, κ = inv(T(ρ)), λ / T(ρ)
    obj = role.ctor(X, y, γ)
    role.prepare(obj)
    w = AdmmWork(T, n, p)
    boyd = tight = nothing
    rows = NTuple{7,Float64}[]
    ran = 0
    last = (NaN, NaN, NaN, NaN, NaN, NaN)
    for k in 1:cap
        t = admm_step!(w, role.px, obj, γ, κ)
        res, objective = lasso_measures!(w, X, y, λ)
        ran = k
        last = (boyd_measures(p, ρ, t; εabs, εrel)..., res, objective)
        hit = false
        if boyd === nothing && boyd_converged(p, ρ, t; εabs, εrel)
            boyd = (k, last)
            hit = true
        end
        if tight === nothing && res <= tight_tol * scale
            tight = (k, last)
            hit = true
        end
        (hit || k in ADMM_CHECKPOINTS) && push!(rows, (Float64(k), last...))
        boyd !== nothing && tight !== nothing && break
    end
    # a target that is not met has been tried through all cap iterations here (the loop breaks only when both are met)
    level(x) = x === nothing ? (; status = :capped, k = ran, measures = last) : (; status = :reached, k = x[1], measures = x[2])
    (; boyd = level(boyd), tight = level(tight), ran, rows)
end

# S-ADMM to the stationarity tol·scale, checked every `every` iterations and at the cap, for the reference optimum.
# At most cap iterations run.
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
        steps = min(every, cap - k)
        for _ in 1:steps
            admm_step!(w, role.px, obj, γ, κ)
        end
        k += steps
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
