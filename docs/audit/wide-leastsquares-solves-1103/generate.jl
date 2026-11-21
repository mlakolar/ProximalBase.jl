# Writes the inputs of the wide least-squares study (#1103) as little-endian binary arrays with a manifest
# (inputs-index.tsv) that lists the case, element type, shape and SHA-256 of each array. The harness and the
# accuracy evaluator check each file against the manifest before use.
#
#   PB_INPUTS=<dir> [PB_PROFILE=full|small] [PB_FILTER=<regex on case ids>] [PB_CHECK_ONLY=1] \
#   julia --threads=1 --project=. docs/audit/wide-leastsquares-solves-1103/generate.jl
#
# Profile full holds the aspect ratios n×p of the study, profile small (the default with PB_CHECK_ONLY=1) three
# small ones. PB_FILTER keeps the cases whose id matches. The main cases are the three designs (Gaussian, rank n/2,
# duplicated rows) in Float64 and Float32, each with its responses, and the accuracy controls built from the
# reproducer of #1078. The inconsistent responses of the rank-deficient designs are for the accuracy evaluation
# only (the timed flag of the manifest is false), and only the Float64 Gaussian cases of the lasso sizes are ADMM
# instances. The arrays are generated in Float64 from the seed of the design and size; the Float32 cases hold the
# rounded arrays and their own ADMM iterates. The streams of a Random generator can differ
# between Julia versions, so the files, not the seeds, define the inputs; regenerating with another version can
# give other hashes. With PB_CHECK_ONLY=1 the script also reloads the manifest and checks the arrays.
include(joinpath(@__DIR__, "common.jl"))

const PROFILES = Dict(
    "full" => (sizes = [(100, 1000), (500, 5000), (1000, 2000), (1000, 1100), (200, 20000), (1500, 5000)],
               lasso = [(1500, 5000), (500, 5000), (200, 20000)]),
    "small" => (sizes = [(20, 100), (40, 160), (30, 36)], lasso = [(40, 160)]))
const DESIGNS = ("gauss", "rankhalf", "duprows")
const ADMM_ITERATES = (1, 5, 20, 100)          # v = z - u after these iteration counts, at ρ = 1
const NOISE_SD = sqrt(1e-3)
const NONZERO_COUNT = 100                      # the expected number of nonzeros of the sparse β*

step_grid(::Type{Float64}) = [1e-3, 1.0, 1e3, 1e6]
step_grid(::Type{Float32}) = [1e-3, 1.0, 1e3]
type_tag(::Type{Float64}) = "f64"
type_tag(::Type{Float32}) = "f32"

# X (unit-norm columns), the sparse β*, the noise and four random proximal inputs of one design and size.
function build_design(design::String, n::Int, p::Int, seed::Int)
    rng = MersenneTwister(seed)
    X = if design == "gauss"
        randn(rng, n, p)
    elseif design == "rankhalf"
        r = n ÷ 2
        randn(rng, n, r) * randn(rng, r, p)
    elseif design == "duprows"
        iseven(n) || error("duplicated rows need an even number of observations")
        X1 = randn(rng, n ÷ 2, p)
        [X1; X1]
    else
        error("unknown design $design")
    end
    for j in 1:p
        c = view(X, :, j)
        c ./= norm(c)
    end
    βstar = [rand(rng) < NONZERO_COUNT / p ? randn(rng) : 0.0 for _ in 1:p]
    noise = NOISE_SD .* randn(rng, n)
    Vrand = randn(rng, p, 4)
    X, βstar, noise, Vrand
end

# the responses of a design: Gaussian designs have one, the rank-deficient ones a consistent y = Xβ* and an
# inconsistent y = Xβ* + noise (for duplicated rows the consistent response repeats the first half)
function build_responses(design::String, X, βstar, noise)
    y = X * βstar
    if design == "gauss"
        return [("noisy", y .+ noise)]
    end
    yc = copy(y)
    design == "duprows" && (yc[size(X, 1) ÷ 2 + 1:end] .= yc[1:size(X, 1) ÷ 2])
    [("consistent", yc), ("inconsistent", y .+ noise)]
end

function write_array(dir, name, A)
    path = joinpath(dir, name)
    open(path, "w") do io
        write(io, A)
    end
    path
end

# the controls of #1078: designs (X, y, v) with a proximal input of size 1/ρ in the row space of X, ρ near eps(T)
function control_cases()
    givens(n, i, j, θ) = (G = Matrix{Float64}(I, n, n); G[i, i] = G[j, j] = cos(θ); G[i, j] = -sin(θ); G[j, i] = sin(θ); G)
    Q = givens(4, 1, 2, 0.3) * givens(4, 2, 3, 0.7) * givens(4, 3, 4, -0.5) * givens(4, 1, 3, 0.2)
    c, s = cos(0.3), sin(0.3)
    out = NamedTuple[]
    for T in (Float64, Float32), sgn in (1, -1), (rtag, mult) in (("eps8", 1 / 8), ("eps", 1.0), ("eps32", 32.0))
        ρ = eps(T) * T(mult)
        γ = inv(ρ)
        L = Float64(sgn * γ)                   # the size 1/ρ of the row-space component, with the sign
        designs = [
            ("axis1x2", [1.0 0.0], [1.0], [L, 0.0]),
            ("axis1x2n", [1.0 0.0], [1.0], [L, 3.0]),
            ("axis1x3", [1.0 0.0 0.0], [1.0], [L, 0.5, -2.0]),
            ("diag2x4", [2.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0], [1.0, -1.0], [L, L, 0.25, -3.0]),
            ("diag2x4z", [1.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0], [1.0, 0.5], [L, 0.7, 0.3, -2.0]),
            ("rot1x2", [c s], [1.0], L .* [c, s] .+ 0.5 .* [-s, c]),
            ("rot2x4", [1.0 0.0 0.0 0.0; 0.0 0.5 0.0 0.0] * Q, [1.0, -1.0], L .* Q[1, :] .+ 0.3 .* Q[3, :] .- 0.7 .* Q[4, :]),
            ("rot2x4z", [1.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0] * Q, [1.0, 0.5], L .* Q[1, :] .+ 0.3 .* Q[3, :] .- 0.7 .* Q[4, :]),
            ("sq2x2", [1.0 0.0; 0.0 2.0], [1.0, 1.0], [L, L]),
            ("tall3x2", [1.0 0.0; 0.0 1.0; 1.0 1.0], [1.0, 2.0, 3.0], [L, -0.5]),
        ]
        for (name, X, y, v) in designs
            push!(out, (id = "c1078/$(type_tag(T))/$name/$(sgn > 0 ? "pos" : "neg")/$rtag", T = T, design = name,
                        X = Matrix{T}(X), y = Vector{T}(y), V = reshape(Vector{T}(v), :, 1), γ = Float64(γ)))
        end
    end
    out
end

function manifest_row(io, c; array, file, shape, sha, note)
    gam = join(repr.(c.gammas), ",")
    lam = isnan(c.lambda_max) ? "" : repr(c.lambda_max)
    println(io, join([c.id, c.kind, c.T === Float64 ? "Float64" : "Float32", c.design, c.response, c.n, c.p, c.seed,
                      c.timed, c.lasso, gam, lam, array, file, join(shape, 'x'), sha, note], '\t'))
end

function generate(dir::String, profile::String, rx::Regex)
    prof = PROFILES[profile]
    mkpath(dir)
    S = build_roles()[1]
    rows = NamedTuple[]
    for (si, (n, p)) in enumerate(prof.sizes), (di, design) in enumerate(DESIGNS)
        seed = 1103000 + 100 * si + di
        X64, βstar, noise, Vrand = build_design(design, n, p, seed)
        responses = build_responses(design, X64, βstar, noise)
        for T in (Float64, Float32)
            tag = type_tag(T)
            wanted = [(resp, y64) for (resp, y64) in responses if occursin(rx, "$tag/$design/$resp/$(n)x$p")]
            isempty(wanted) && continue
            X = Matrix{T}(X64)
            xfile = "X-$design-$(n)x$p.$tag"
            write_array(dir, xfile, X)
            xsha = file_sha256(joinpath(dir, xfile))
            for (resp, y64) in wanted
                id = "$tag/$design/$resp/$(n)x$p"
                y = Vector{T}(y64)
                lmax = Float64(norm(X' * y, Inf))
                λ = T(0.1 * lmax)
                iterates = admm_record(S, X, y, λ, 1.0, ADMM_ITERATES)
                V = hcat(Matrix{T}(Vrand), reduce(hcat, iterates))
                c = Case(id, :main, T, design, resp, n, p, seed, resp != "inconsistent", T === Float64 && design == "gauss" && (n, p) in prof.lasso,
                         step_grid(T), lmax, Dict{String,NamedTuple{(:file, :shape, :sha256, :note), Tuple{String,Vector{Int},String,String}}}())
                yfile, vfile = "y-$design-$resp-$(n)x$p.$tag", "V-$design-$resp-$(n)x$p.$tag"
                write_array(dir, yfile, y)
                write_array(dir, vfile, V)
                push!(rows, (c = c, arrays = [("X", xfile, [n, p], xsha, ""),
                                              ("y", yfile, [n], file_sha256(joinpath(dir, yfile)), ""),
                                              ("V", vfile, [p, 8], file_sha256(joinpath(dir, vfile)),
                                               "columns=rand1,rand2,rand3,rand4," * join(("admm@$k" for k in ADMM_ITERATES), ","))]))
                println("generated ", id)
            end
        end
    end
    for ctl in control_cases()
        occursin(rx, ctl.id) || continue
        T = ctl.T
        base = case_file(ctl.id)
        tag = type_tag(T)
        n, p = size(ctl.X)
        c = Case(ctl.id, :control, T, ctl.design, "control", n, p, 0, false, false, [ctl.γ], NaN,
                 Dict{String,NamedTuple{(:file, :shape, :sha256, :note), Tuple{String,Vector{Int},String,String}}}())
        files = Dict(a => "$base-$a.$tag" for a in ("X", "y", "V"))
        for (a, A) in (("X", ctl.X), ("y", ctl.y), ("V", ctl.V))
            write_array(dir, files[a], A)
        end
        push!(rows, (c = c, arrays = [("X", files["X"], [n, p], file_sha256(joinpath(dir, files["X"])), ""),
                                      ("y", files["y"], [n], file_sha256(joinpath(dir, files["y"])), ""),
                                      ("V", files["V"], [p, 1], file_sha256(joinpath(dir, files["V"])), "columns=control")]))
    end
    open(joinpath(dir, "inputs-index.tsv"), "w") do io
        println(io, "# generator: generate.jl")
        println(io, "# julia: ", VERSION)
        println(io, "# profile: ", profile)
        println(io, "# filter: ", rx.pattern)
        println(io, "# created_utc: ", Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ"))
        println(io, "# source_sha: ", get(ENV, "PB_SOURCE_SHA", "unset"))
        println(io, "# harness_sha: ", get(ENV, "PB_HARNESS_SHA", "unset"))
        println(io, join(MANIFEST_COLUMNS, '\t'))
        for r in rows, (a, f, shp, sha, note) in r.arrays
            manifest_row(io, r.c; array = a, file = f, shape = shp, sha = sha, note = note)
        end
    end
    length(rows)
end

# Reloads the manifest, checks each file against its SHA-256 and the structure of the arrays. The rank and
# consistency checks use an SVD and run for the cases with n ≤ 200.
function validate(dir::String)
    m = load_manifest(dir)
    nmain = ncontrol = 0
    for c in m.cases
        X, y, V = read_input(m, c, "X"), read_input(m, c, "y"), read_input(m, c, "V")
        T = c.T
        size(X) == (c.n, c.p) && length(y) == c.n && size(V, 1) == c.p || error("shape mismatch in $(c.id)")
        all(isfinite, X) && all(isfinite, y) && all(isfinite, V) || error("nonfinite input in $(c.id)")
        if c.kind === :control
            size(V, 2) == 1 && length(c.gammas) == 1 || error("control $(c.id) needs one input and one step")
            ncontrol += 1
            continue
        end
        nmain += 1
        size(V, 2) == 8 && c.gammas == step_grid(T) || error("main case $(c.id) needs 8 inputs and the step grid")
        tol = T === Float64 ? 1e-12 : 1e-5
        maximum(abs.(norm.(eachcol(Float64.(X))) .- 1)) <= tol || error("columns of $(c.id) are not unit-norm")
        any(!iszero, V[:, 5]) || error("the first ADMM iterate of $(c.id) is zero")
        if c.design == "duprows"
            h = c.n ÷ 2
            X[1:h, :] == X[h+1:end, :] || error("rows of $(c.id) are not duplicated")
        end
        if c.n <= 200 && c.design != "gauss"
            sv = svdvals(Float64.(X))
            rank_tol = (T === Float64 ? 1e-10 : 1e-5) * sv[1]
            count(>(rank_tol), sv) == c.n ÷ 2 || error("rank of $(c.id) is not n/2")
            F = svd(Float64.(X))
            k = count(>(rank_tol), F.S)
            U = F.U[:, 1:k]
            off = norm(y - U * (U' * y)) / norm(y)
            (c.response == "consistent" ? off <= (T === Float64 ? 1e-9 : 1e-4) : off >= 1e-4) ||
                error("response of $(c.id) is not $(c.response) (distance from the range $off)")
        end
    end
    println("validated ", nmain, " main cases and ", ncontrol, " controls; manifest SHA-256 ", m.sha256)
    m
end

function main()
    dir = get(ENV, "PB_INPUTS", "")
    isempty(dir) && error("PB_INPUTS names the output directory")
    check = env_flag("PB_CHECK_ONLY")
    profile = env_string("PB_PROFILE", check ? "small" : "full")
    haskey(PROFILES, profile) || error("PB_PROFILE must be full or small")
    rx = Regex(env_string("PB_FILTER", ""))
    t0 = time()
    ncases = generate(dir, profile, rx)
    println("wrote ", ncases, " cases to the inputs directory in ", round(time() - t0; digits = 1), " s")
    check && validate(dir)
    check && println("generate.jl CHECK_ONLY passed")
end

main()
