# Paired benchmark of the least-squares value and gradient paths (issue #915): LeastSquaresLoss f(β) and
# value_and_gradient!, raw and as Scaled(f, 1/n), and gradient!, raw; the SqrNormL2 value with a center; and the
# predictor and residual oracles value_from_predictor, value_and_gradient_from_predictor! and
# value_from_residual, Float32 and Float64 (and Float32 data with a Float64 scale or level).
#
#   PB_FIRST=<root of role X> PB_SECOND=<root of role Y> PB_OUTPUT=<dir> PB_ROUNDS=15 \
#     julia --threads=1 --check-bounds=no --project=benchmark benchmark/leastsquares_predictor.jl
#
# Each root contains a src/ folder with ProximalBase.jl. Role X is compiled twice (X and the independent A/A
# control XC); role Y is the one under test. Each round times X, Y and XC in alternating order;
# benchmark/summarize_norml1_infinite_weight.py DIR reports the per-round ratios Y/X and XC/X. Run it again with
# the roles reversed (summarized on its own) and in a fresh process. The matrices, vectors and output buffers
# are shared by the three roles, and so is the scratch f.tmp of the losses of the LeastSquaresLoss cells and of the
# ordinary oracle cells, which the script assigns after the constructor. The losses of the exceptional oracle cells
# keep the scratch of their constructor, which the value oracles timed there do not use.
#
# Two kinds of cells. The cells of the families LeastSquaresLoss and SqrNormL2 call the same methods in the
# three trees (the oracles are not called there). The cells of the family LeastSquaresOracle compare the
# oracles of the tree that has them with loops written in this script (the modules K1, K2 and K3, one copy
# per role): value from predictor s = Σ (Yᵢ - zᵢ)², α (s / 2); value and gradient d = z - Y, g = Xᵀd,
# g .*= α, α (Σ dᵢ² / 2); value from residual s = Σ rᵢ², α s / 2 (raw: no α). A role whose tree has no oracle
# calls the loops, so in a forward run X and XC are the loops (compiled twice, K1 and K2) and Y is the oracle,
# and in a run with the roles reversed X and XC are the oracle (compiled twice) and Y is the loop (K3). The loops
# do not recover from overflow or underflow. XC/X is the A/A control of these cells, and Y/X compares different code.
#
# PB_CHECK_ONLY=1 runs the check and the allocation measurement and exits without timing: XC must equal X bit for
# bit (`===`); on the ordinary cells Y must agree with X within 32 eps(T) (the largest difference found is printed; an empirical bound for
# the random data of the cells); the exceptional cells (marked `exc` in the list printed) are compared with a
# 256-bit BigFloat reference computed from the inputs: the implementation of the tree that has the new methods
# must satisfy |v - v*| <= (n + 8) eps(T) |v*| (plus 2 nextfloat(zero(T)) where |v*| < floatmin(T)) with v finite,
# and the result of the other implementation is printed. The warm allocations of each role in each selected case are
# measured behind a function barrier and printed; the check fails where the implementation of the tree that has
# the new methods allocates more than the other, or allocates on an ordinary cell. PB_FILTER selects cases by a
# regular expression on their ids family/operation/type/scenario/size; PB_ROUNDS is the number of rounds (default
# 11); PB_OUTPUT the output directory; PB_PRINT=1 also prints raw.csv and provenance.txt to the standard output
# between the lines "BEGIN <file>" and "END <file>", for a run whose output directory is not copied back.
using BenchmarkTools, Random, Statistics, Dates, LinearAlgebra
Threads.nthreads() == 1 || error("one Julia thread required")
LinearAlgebra.BLAS.set_num_threads(1)          # the products of the larger cells run on one thread
setprecision(BigFloat, 256)
first_root = abspath(ENV["PB_FIRST"]); second_root = abspath(ENV["PB_SECOND"])
module First end
module Second end
module FirstControl end
Base.include(First, joinpath(first_root, "src", "ProximalBase.jl"))
Base.include(Second, joinpath(second_root, "src", "ProximalBase.jl"))
Base.include(FirstControl, joinpath(first_root, "src", "ProximalBase.jl"))

# The loops that the oracle cells compare with, in a source string so that each role gets its own compiled copy.
const LOOPS = raw"""
using LinearAlgebra: mul!
function value_from_predictor_raw(Y, z)
    s = zero(eltype(Y))
    @inbounds @simd for i in eachindex(z, Y)
        s += abs2(Y[i] - z[i])
    end
    s / 2
end
function value_from_predictor_scaled(α, Y, z)
    s = zero(eltype(Y))
    @inbounds @simd for i in eachindex(z, Y)
        s += abs2(Y[i] - z[i])
    end
    α * (s / 2)
end
function value_and_gradient_raw!(g, d, X, Y, z)
    d .= z .- Y
    mul!(g, transpose(X), d)
    sum(abs2, d) / 2
end
function value_and_gradient_scaled!(g, d, X, Y, z, α)
    d .= z .- Y
    mul!(g, transpose(X), d)
    g .*= α
    α * (sum(abs2, d) / 2)
end
function value_from_residual_raw(r)
    s = zero(eltype(r))
    @inbounds @simd for i in eachindex(r)
        s += r[i] * r[i]
    end
    s / 2
end
function value_from_residual_scaled(α, r)
    s = zero(eltype(r))
    @inbounds @simd for i in eachindex(r)
        s += r[i] * r[i]
    end
    α * s / 2
end
"""
module K1 end
module K2 end
module K3 end
for K in (K1, K2, K3)
    Base.include_string(K, LOOPS, "leastsquares_predictor_loops.jl")
end

const TYPES = (Float32, Float64)
const SIZES = ((64, 6), (4096, 6), (100000, 6), (2000, 500))        # (n, p) of the shared data
const TOL = 32                                                       # eps(T), ordinary cells, role Y against role X

# Shared data per (T, n, p): design, response, coefficients, a predictor z and its residual r = z - Y, and
# buffers (g of length p, d and the scratch tmp of length n).
function makedata(::Type{T}, n::Int, p::Int) where {T}
    rng = Random.MersenneTwister(915 + 7n + p + 3sizeof(T))
    X = randn(rng, T, n, p)
    Y = randn(rng, T, n)
    β = randn(rng, T, p)
    z = X * β .+ T(0.5) .* randn(rng, T, n)
    (; n, p, X, Y, β, z, r = z .- Y, g = zeros(T, p), d = zeros(T, n), tmp = zeros(T, n))
end
data = Dict((T, n, p) => makedata(T, n, p) for T in TYPES for (n, p) in SIZES)

# Exceptional cells, n = 64 and p = 6, with 256-bit references computed from the inputs. `refs[id]` is
# (T, number of squares, v*) of the value that the cell with this id has to give.
refs = Dict{String, NamedTuple}()
const NE, PE = 64, 6
function lsref(X, Y, β)
    sum(abs2, [big(Y[i]) - sum(big(X[i, j]) * big(β[j]) for j in axes(X, 2)) for i in axes(X, 1)]) / 2
end
function lsexceptional(::Type{T}) where {T}
    rng = Random.MersenneTwister(9150 + sizeof(T))
    ex = exponent(floatmax(T))
    Xid = [i == j ? one(T) : zero(T) for i in 1:NE, j in 1:PE]
    zeroβ = zeros(T, PE)
    # squares that underflow to zero while the sum is a subnormal: the squares are 2^(2 fld(mn - 2, 2)) < nextfloat(zero(T)) / 2
    mn = exponent(nextfloat(zero(T)))
    under = fill(ldexp(one(T), fld(mn - 2, 2)), NE)
    # one entry whose square overflows while the value is below floatmax
    over = randn(rng, T, NE); over[1] = ldexp(T(1.2), (ex + 1) ÷ 2)
    # an exact fit: integer data, Xβ exact
    Xf = T.(rand(rng, -3:3, NE, PE)); βf = T.(rand(rng, -3:3, PE)); Yf = Xf * βf
    # X β overflows (A A - A A = Inf - Inf) in the first row, Xβ = 0 in the others
    A = ldexp(one(T), ex ÷ 2 + 4)
    Xp = zeros(T, NE, PE); Xp[:, 1] .= 1; Xp[:, 2] .= 1; Xp[1, 1] = A; Xp[1, 2] = A
    βp = zeros(T, PE); βp[1] = A; βp[2] = -A
    Yp = randn(rng, T, NE)
    cells = [("under", Xid, under, zeroβ), ("over_val", Xid, over, zeroβ), ("exact_fit", Xf, Yf, βf),
             ("prod_over", Xp, Yp, βp)]
    map(cells) do (name, X, Y, β)
        id = "LeastSquaresLoss/value/$T/$name/n$(NE)_p$PE"
        refs[id] = (; T, n = NE, v = lsref(X, Y, β))
        (; id, X, Y, β, tmp = zeros(T, NE))
    end
end
lsexc = Dict(T => lsexceptional(T) for T in TYPES)

# SqrNormL2 at x with the center y = 0: the squares underflow to zero while the value is a subnormal
function sqexceptional(::Type{T}) where {T}
    mn = exponent(nextfloat(zero(T)))
    x = fill(ldexp(one(T), fld(mn - 2, 2)), NE)
    id = "SqrNormL2/value/$T/underflow/n$NE"
    refs[id] = (; T, n = NE, v = sum(abs2, big.(x)) / 2)
    (; id, x, y = zeros(T, NE), λ = one(T))
end
sqexc = Dict(T => sqexceptional(T) for T in TYPES)

# The oracles at squares that overflow (the sum, not a square) or underflow to zero, with a scale that brings the value
# into range; the scale has the type of the data, or is a Float64 for Float32 data.
function orexceptional(::Type{T}, ::Type{S}) where {T, S}
    ex = exponent(floatmax(T))
    tag = S === T ? "scaled" : "scaled_alpha64"
    map((("over", ldexp(one(T), ex ÷ 2), ldexp(one(S), -20)),
         ("under", ldexp(one(T), -(ex ÷ 2 + ex ÷ 8)), ldexp(one(S), ex ÷ 3)))) do (name, r, α)
        rv = fill(r, NE)
        ids = ("LeastSquaresOracle/value_from_predictor/$T/$(name)_$tag/n$NE",
               "LeastSquaresOracle/value_from_residual/$T/$(name)_$tag/n$NE")
        v = big(α) * sum(abs2, big.(rv)) / 2
        foreach(id -> refs[id] = (; T, n = NE, v), ids)
        (; ids, α, Y = rv, z = zeros(T, NE), r = rv, X = zeros(T, NE, PE))
    end
end
orexc = Dict((T, S) => orexceptional(T, S) for (T, S) in ((Float32, Float32), (Float32, Float64), (Float64, Float64)))

# The closures are built behind function barriers (the callee and its arguments are captured with concrete
# types), so that a call through the run-time module value does not allocate by itself.
bindval(fn, args...) = (box = [fn(args...)]; () -> (@inbounds box[1] = fn(args...); nothing))
bindcall(fn, args...) = () -> (fn(args...); nothing)
valcheck(fn, args...) = () -> fn(args...)
valgcheck(g, fn, args...) = () -> (v = fn(args...); (v, copy(g)))
gradcheck(g, fn, args...) = () -> (fn(args...); copy(g))

function lossof(PB, d, α = nothing)
    f = PB.LeastSquaresLoss(d.Y, d.X)
    f.tmp = d.tmp                                  # the scratch is shared by the roles
    α === nothing ? f : PB.Scaled(f, α)
end

function makesuite(PB, K)
    probes = NamedTuple[]
    hasnew = isdefined(PB, :value_from_predictor)
    add(id, T, op, check, work, head, exc = false) = push!(probes, (; id, T, op, check, work, head, exc))
    for T in TYPES
        # LeastSquaresLoss: the same methods in the three trees
        for (n, p) in SIZES
            d = data[(T, n, p)]
            α = T(1) / n
            f, S = lossof(PB, d), lossof(PB, d, α)
            sz = "n$(n)_p$p"
            if p == 6
                add("LeastSquaresLoss/value/$T/raw/$sz", T, bindval(f, d.β), valcheck(f, d.β), n * p, hasnew)
                add("LeastSquaresLoss/value/$T/scaled/$sz", T, bindval(S, d.β), valcheck(S, d.β), n * p, hasnew)
            else
                add("LeastSquaresLoss/gradient/$T/raw/$sz", T, bindcall(PB.gradient!, d.g, f, d.β),
                    gradcheck(d.g, PB.gradient!, d.g, f, d.β), n * p, hasnew)
            end
            if n in (64, 2000)
                for (name, h) in (("raw", f), ("scaled", S))
                    add("LeastSquaresLoss/value_and_gradient/$T/$name/$sz", T, bindval(PB.value_and_gradient!, d.g, h, d.β),
                        valgcheck(d.g, PB.value_and_gradient!, d.g, h, d.β), n * p, hasnew)
                end
            end
            n == 64 && add("LeastSquaresLoss/gradient/$T/raw/$sz", T, bindcall(PB.gradient!, d.g, f, d.β),
                           gradcheck(d.g, PB.gradient!, d.g, f, d.β), n * p, hasnew)
        end
        for c in lsexc[T]
            f = PB.LeastSquaresLoss(c.Y, c.X)
            f.tmp = c.tmp
            add(c.id, T, bindval(f, c.β), valcheck(f, c.β), NE * PE, hasnew, true)
        end
        # SqrNormL2 with a center: the same method in the three trees
        for n in (64, 4096, 100000)
            d = data[(T, n, 6)]
            for (name, λ) in (("center", T(0.3)), ("center_lam64", 0.3))
                T === Float64 && name == "center_lam64" && continue
                f = PB.SqrNormL2(λ, d.Y)
                add("SqrNormL2/value/$T/$name/n$n", T, bindval(f, d.z), valcheck(f, d.z), n, hasnew)
            end
        end
        let c = sqexc[T]
            f = PB.SqrNormL2(c.λ, c.y)
            add(c.id, T, bindval(f, c.x), valcheck(f, c.x), NE, hasnew, true)
        end
        # the oracles against the loops
        for S in (T === Float32 ? (Float32, Float64) : (Float64,))
            tag = S === T ? "scaled" : "scaled_alpha64"
            for n in (64, 4096, 100000)
                d = data[(T, n, 6)]
                α = S(1) / n
                sz = "n$n"
                if hasnew
                    f, Sc = lossof(PB, d), lossof(PB, d, α)
                    S === T && add("LeastSquaresOracle/value_from_predictor/$T/raw/$sz", T, bindval(PB.value_from_predictor, f, d.z),
                                   valcheck(PB.value_from_predictor, f, d.z), n, true)
                    add("LeastSquaresOracle/value_from_predictor/$T/$tag/$sz", T, bindval(PB.value_from_predictor, Sc, d.z),
                        valcheck(PB.value_from_predictor, Sc, d.z), n, true)
                    S === T && add("LeastSquaresOracle/value_from_residual/$T/raw/$sz", T, bindval(PB.value_from_residual, f, d.r),
                                   valcheck(PB.value_from_residual, f, d.r), n, true)
                    add("LeastSquaresOracle/value_from_residual/$T/$tag/$sz", T, bindval(PB.value_from_residual, Sc, d.r),
                        valcheck(PB.value_from_residual, Sc, d.r), n, true)
                else
                    S === T && add("LeastSquaresOracle/value_from_predictor/$T/raw/$sz", T, bindval(K.value_from_predictor_raw, d.Y, d.z),
                                   valcheck(K.value_from_predictor_raw, d.Y, d.z), n, false)
                    add("LeastSquaresOracle/value_from_predictor/$T/$tag/$sz", T, bindval(K.value_from_predictor_scaled, α, d.Y, d.z),
                        valcheck(K.value_from_predictor_scaled, α, d.Y, d.z), n, false)
                    S === T && add("LeastSquaresOracle/value_from_residual/$T/raw/$sz", T, bindval(K.value_from_residual_raw, d.r),
                                   valcheck(K.value_from_residual_raw, d.r), n, false)
                    add("LeastSquaresOracle/value_from_residual/$T/$tag/$sz", T, bindval(K.value_from_residual_scaled, α, d.r),
                        valcheck(K.value_from_residual_scaled, α, d.r), n, false)
                end
            end
            for (n, p) in ((64, 6), (2000, 500))
                d = data[(T, n, p)]
                α = S(1) / n
                sz = "n$(n)_p$p"
                if hasnew
                    f, Sc = lossof(PB, d), lossof(PB, d, α)
                    S === T && add("LeastSquaresOracle/value_and_gradient_from_predictor/$T/raw/$sz", T,
                                   bindval(PB.value_and_gradient_from_predictor!, d.g, d.d, f, d.z),
                                   valgcheck(d.g, PB.value_and_gradient_from_predictor!, d.g, d.d, f, d.z), n * p, true)
                    add("LeastSquaresOracle/value_and_gradient_from_predictor/$T/$tag/$sz", T,
                        bindval(PB.value_and_gradient_from_predictor!, d.g, d.d, Sc, d.z),
                        valgcheck(d.g, PB.value_and_gradient_from_predictor!, d.g, d.d, Sc, d.z), n * p, true)
                else
                    S === T && add("LeastSquaresOracle/value_and_gradient_from_predictor/$T/raw/$sz", T,
                                   bindval(K.value_and_gradient_raw!, d.g, d.d, d.X, d.Y, d.z),
                                   valgcheck(d.g, K.value_and_gradient_raw!, d.g, d.d, d.X, d.Y, d.z), n * p, false)
                    add("LeastSquaresOracle/value_and_gradient_from_predictor/$T/$tag/$sz", T,
                        bindval(K.value_and_gradient_scaled!, d.g, d.d, d.X, d.Y, d.z, α),
                        valgcheck(d.g, K.value_and_gradient_scaled!, d.g, d.d, d.X, d.Y, d.z, α), n * p, false)
                end
            end
            for c in orexc[(T, S)]
                if hasnew
                    f = PB.Scaled(PB.LeastSquaresLoss(c.Y, c.X), c.α)
                    add(c.ids[1], T, bindval(PB.value_from_predictor, f, c.z), valcheck(PB.value_from_predictor, f, c.z), NE, true, true)
                    add(c.ids[2], T, bindval(PB.value_from_residual, f, c.r), valcheck(PB.value_from_residual, f, c.r), NE, true, true)
                else
                    add(c.ids[1], T, bindval(K.value_from_predictor_scaled, c.α, c.Y, c.z),
                        valcheck(K.value_from_predictor_scaled, c.α, c.Y, c.z), NE, false, true)
                    add(c.ids[2], T, bindval(K.value_from_residual_scaled, c.α, c.r),
                        valcheck(K.value_from_residual_scaled, c.α, c.r), NE, false, true)
                end
            end
        end
    end
    probes
end

roles = ["X" => (First, K1), "Y" => (Second, K3), "XC" => (FirstControl, K2)]
suites = Dict(r => makesuite(getfield(mod, :ProximalBase), K) for (r, (mod, K)) in roles)
filter_re = Regex(get(ENV, "PB_FILTER", ""))
for r in keys(suites)
    filter!(p -> occursin(filter_re, p.id), suites[r])
end
ids = getfield.(suites["X"], :id)
@assert ids == getfield.(suites["Y"], :id) == getfield.(suites["XC"], :id)
isempty(ids) && error("PB_FILTER selects no case")
@assert length(unique(ids)) == length(ids)

exact_equal(a::Tuple, b::Tuple) = length(a) == length(b) && all(exact_equal(u, v) for (u, v) in zip(a, b))
exact_equal(a::AbstractArray, b::AbstractArray) = size(a) == size(b) && all(i -> a[i] === b[i], eachindex(a, b))
exact_equal(a, b) = a === b
# the difference of two results in units of eps(T) (relative to |a|, or to the largest entry of an array)
epsdiff(T, a::Tuple, b::Tuple) = maximum(epsdiff(T, u, v) for (u, v) in zip(a, b))
function epsdiff(T, a::Number, b::Number)
    isequal(a, b) && return 0.0
    (isfinite(a) && isfinite(b) && !iszero(a)) || return Inf
    Float64(abs(big(a) - big(b)) / (big(eps(T)) * abs(big(a))))
end
function epsdiff(T, a::AbstractArray, b::AbstractArray)
    isequal(a, b) && return 0.0
    (all(isfinite, a) && all(isfinite, b)) || return Inf
    s = maximum(abs, a; init = zero(eltype(a)))
    iszero(s) && return Inf
    Float64(maximum(abs(big(a[i]) - big(b[i])) for i in eachindex(a, b)) / (big(eps(T)) * big(s)))
end
function within(T, v, ref)
    isfinite(v) || return false
    tol = (ref.n + 8) * big(eps(T)) * abs(ref.v)
    abs(ref.v) < floatmin(T) && (tol += 2 * big(nextfloat(zero(T))))
    abs(big(v) - ref.v) <= tol
end
shown(x) = (s = repr(x); length(s) > 100 ? s[1:100] * "..." : s)

failures = String[]
worst = Dict{String, Tuple{Float64, String}}()
println("exceptional cells (v* is the 256-bit reference rounded to the data type; * marks the implementation of the tree with the new methods):")
for (i, id) in enumerate(ids)
    a, b, c = suites["X"][i], suites["Y"][i], suites["XC"][i]
    ra, rc, rb = a.check(), c.check(), b.check()
    exact_equal(ra, rc) || push!(failures, "A/A output mismatch: $id: X = $(shown(ra)), XC = $(shown(rc))")
    if a.exc
        T, ref = a.T, refs[id]
        verdict(p, r) = (p.head ? "*" : "") * (within(T, r, ref) ? "within" : "OUTSIDE") * " " * shown(r)
        println("  ", id, "  v* = ", T(ref.v), "\n      X: ", verdict(a, ra), "\n      Y: ", verdict(b, rb), "\n      XC: ", verdict(c, rc))
        for (p, r, role) in ((a, ra, "X"), (b, rb, "Y"), (c, rc, "XC"))
            p.head && !within(p.T, r, ref) && push!(failures, "the implementation of the tree with the new methods (role $role) is outside the bound: $id: $(shown(r)) against $(T(ref.v))")
        end
    else
        e = epsdiff(a.T, ra, rb)
        fam = join(split(id, '/')[1:2], '/')
        e > get(worst, fam, (-1.0, ""))[1] && (worst[fam] = (e, id))
        e <= TOL || push!(failures, "output of Y differs from X by $e eps beyond $TOL eps: $id: X = $(shown(ra)), Y = $(shown(rb))")
    end
end
println("largest difference of Y from X on the ordinary cells, in eps of the data type:")
for fam in sort(collect(keys(worst)))
    println("  ", rpad(fam, 62), round(worst[fam][1]; sigdigits = 3), "  ", worst[fam][2])
end

allocated_by(op) = (op(); op(); @allocated op())      # a function barrier: op is a concretely typed closure here
println("warm allocations in bytes, X / Y / XC (* marks the implementation of the tree with the new methods):")
for (i, id) in enumerate(ids)
    ps = (suites["X"][i], suites["Y"][i], suites["XC"][i])
    bytes = map(p -> allocated_by(p.op), ps)
    println("  ", rpad(id, 82), join([(ps[k].head ? "*" : " ") * lpad(bytes[k], 5) for k in 1:3], " "), ps[1].exc ? "  exc" : "")
    headb = maximum((bytes[k] for k in 1:3 if ps[k].head); init = -1)
    baseb = maximum((bytes[k] for k in 1:3 if !ps[k].head); init = -1)
    headb >= 0 && baseb >= 0 && headb > baseb && push!(failures, "the implementation of the tree with the new methods allocates $headb bytes, the other $baseb: $id")
    headb > 0 && !ps[1].exc && push!(failures, "the implementation of the tree with the new methods allocates $headb bytes on an ordinary cell: $id")
end
if isempty(failures)
    println("check passed: ", length(ids), " cells")
else
    foreach(f -> println("FAILED: ", f), failures)
    error("$(length(failures)) failures (listed above)")
end
get(ENV, "PB_CHECK_ONLY", "0") == "1" && exit()

function bench(p)
    op = p.op
    evals = max(1, 2^16 ÷ p.work)
    b = @benchmarkable $op() evals = evals samples = 100 seconds = 0.15 gctrial = false
    for _ in 1:20
        op()
    end
    run(b)
end
outdir = get(ENV, "PB_OUTPUT", "/tmp/pb915-bench")
mkpath(outdir)
open(joinpath(outdir, "provenance.txt"), "w") do io
    println(io, "UTC: ", now(UTC), "\nJulia: ", VERSION, "\nCPU: ", Sys.CPU_NAME, "\nThreads: ", Threads.nthreads())
    println(io, "BLAS threads: ", LinearAlgebra.BLAS.get_num_threads())
    println(io, "Bounds option: ", Base.JLOptions().check_bounds)
    println(io, "Filter: ", get(ENV, "PB_FILTER", ""))
    println(io, "X: ", first_root, "\nY: ", second_root)
end
for (r, _) in roles, p in suites[r]          # compile and warm the closure of each role and case
    bench(p)
end
rounds = parse(Int, get(ENV, "PB_ROUNDS", "11"))
open(joinpath(outdir, "raw.csv"), "w") do io
    println(io, "case,round,order,role,median_ns,min_ns,memory,allocs")
    idx = eachindex(ids)
    for round in 1:rounds
        for i in (isodd(round) ? idx : reverse(idx))
            order = isodd(round) ? ("X", "Y", "XC") : ("XC", "Y", "X")
            for (k, r) in enumerate(order)
                p = suites[r][i]
                t = bench(p)
                m = median(t)
                println(io, p.id, ',', round, ',', k, ',', r, ',', m.time, ',', minimum(t).time, ',', m.memory, ',', m.allocs)
            end
        end
        flush(io)
        println("round ", round, "/", rounds); flush(stdout)
    end
end
if get(ENV, "PB_PRINT", "0") == "1"
    for file in ("provenance.txt", "raw.csv")
        println("BEGIN ", file)
        print(read(joinpath(outdir, file), String))
        println("END ", file)
    end
end
