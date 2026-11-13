# Experiment #901 variant (orchestrator): PB_TOLERANT=1 replaces the bit-for-bit comparison of Y with X by a
# coarse sanity check (same NaN/Inf class, relative difference ≤ 1e-3 of the output scale; a mismatch is
# reported, not fatal) and skips the D-X expectation; the A/A check stays exact and X keeps its BigFloat
# value check. PB_SHARD=i/n keeps every n-th selected case starting at i (after PB_FILTER and PB_STRIDE).
# Accuracy of the candidates is judged by experiment901/accuracy901.jl, not here.
# Differential check and paired benchmark of the exceptional ("cold") paths of NormL2, GroupNormL2 and of the
# shared numerical kernels that serve them (issue #900): the recovery of sums of squares that overflow or
# underflow, thresholds and level-weight products out of the floating-point range, values near the
# overflow threshold, sums below K floatmin (#373), long Float32 recoveries, zero and infinite levels and
# weights, NaN/Inf precedence and zero groups. The consumers of the shared kernels are included:
# SqrNormL2 values, norm_diff, LeastSquaresLoss values and NormL1 values and proximal operators.
#
#   PB_FIRST=<root of role X> PB_SECOND=<root of role Y> PB_OUTPUT=<dir> PB_ROUNDS=15 \
#     julia --threads=1 --check-bounds=no --project=benchmark benchmark/l2_cold_paths.jl
#   python3 benchmark/summarize_norml1_infinite_weight.py <dir>
#
# Design of benchmark/l2_norms.jl: role X (the reference tree) is compiled twice (X and the independent
# A/A control XC), role Y is the tree under test, each round times X, Y and XC in alternating order, and
# inputs and output buffers are shared by the three roles (the iterates, which are each role's own type,
# are built from the same entries). raw.csv and provenance.txt have the format that
# summarize_norml1_infinite_weight.py reads. Run it again with the roles reversed (summarized on its own),
# in a fresh process, and with and without --check-bounds=no.
#
# The check is always exact: every output of Y and of XC must equal that of X bit for bit (`===` per array
# element and per scalar, so signed zeros and NaN bit patterns count; a BigFloat by precision, sign and
# value). An A/A mismatch (X against XC) is an error. A case may carry expect_diff: the one documented
# difference of the consolidation, decision D-X (a BigFloat sum of exponents beyond Int64). PB_EXPECT_DIFF
# selects what the check requires of it:
#   auto (default)  `same` when PB_FIRST and PB_SECOND are the same path, `forward` otherwise
#   0 | same        X and Y must be equal (use it when both roots hold the same source, also at two paths)
#   1 | forward     X (the reference tree) gives the baseline value +0 and Y gives +Inf
#   reversed        X gives +Inf and Y gives +0 (the roles of a reversed run)
# PB_CHECK_ONLY=1 stops after the checks. PB_FILTER selects cases by a regular expression on their ids and
# PB_STRIDE=k keeps every k-th of them (a smoke run). The
# BigFloat cases (about the same operations at a few representative inputs) are check-only: they are not timed.
#
# Before the roles are compared, the data of each scenario are validated with computations that use no package
# code, for the properties that the scenario lists in its `req` tuple (the tags of validate_groups! and of the
# validators of the other families): for example that the ordinary sum of squares in the element type T overflows,
# or is zero although the group is not, that the threshold product is out of range, that the exact value lies in a
# stated fraction of floatmax, or that the exact proximal point is finite in T (:finite_result: only the Float32
# and Float64 scenarios that list it). A scenario with req = () has no assertion on its data (reg_lam0_ord and
# reg_lamInf_ord, for example, whose parameters decide the path); BigFloat scenarios are not asserted to give finite
# results, and a non-finite output (NaN or Inf data, infinite parameters) is intentional in the scenarios made for
# it. A failed assertion is an error that names the case. That the data have these properties, or that an input was
# built, is not evidence that every operation takes its intended path: the exact comparison of the trees is the check.
#
# Value cases that carry a BigFloat reference (the scenarios marked ref = true, in Float32 and Float64, the value
# of the reference tree X only) also require X's value `a` to agree with it, as value_agrees below states: the reference
# of 256 bits is rounded to nearest in T, e = T(ref), and a === e, or both are finite and
# |a - e| <= max(128 eps(T) |e|, 4 K nextfloat(zero(T))), computed in T, with K the count that the scenario gives with its
# reference (the number of groups for GroupNormL2; one or the number of entries for the other families). The absolute term is the larger one for |e| < K nextfloat(zero(T)) / (32 eps(T)) (about
# K 2^-131 in Float32, K 2^-1027 in Float64): there the check allows a relative error above 128 eps, a = 0 for the
# smallest subnormal e and K = 1 included. The other values are compared with the reference tree only.
using BenchmarkTools, Random, Statistics, Dates, LinearAlgebra, SparseArrays
Threads.nthreads() == 1 || error("one Julia thread required")
first_root = abspath(ENV["PB_FIRST"]); second_root = abspath(ENV["PB_SECOND"])
module First end
module Second end
module FirstControl end
Base.include(First, joinpath(first_root, "src", "ProximalBase.jl"))
Base.include(Second, joinpath(second_root, "src", "ProximalBase.jl"))
Base.include(FirstControl, joinpath(first_root, "src", "ProximalBase.jl"))

####################################
#  Cases and exact comparison
####################################

struct Spec
    id::String
    family::String
    op::String
    T::DataType
    timed::Bool
    n::Int                                  # entries touched per call, for the number of evaluations
    build::Function                         # PB -> (op = () -> nothing, check = () -> output)
    ref::Any                                # nothing, or (expected, tolerance) of the value
    diff::Any                               # nothing, or (x = predicate, y = predicate): expect_diff
end

specs = Spec[]
validated = Dict{String, Int}()             # data sets whose path assertions passed, per family
function addspec!(family, op, T, rest, n, build; ref = nothing, diff = nothing, timed = T !== BigFloat)
    push!(specs, Spec("$family/$op/$T/$rest", family, op, T, timed && diff === nothing, n, build, ref, diff))
end
validate_count!(family) = (validated[family] = get(validated, family, 0) + 1)

# Function barriers for the timed operations. The package is reached through a module that is a run-time
# value (PB), so a call written in a closure over it would be a dynamic call that allocates; these helpers take
# the function and the arguments as arguments, so the closure they return captures concretely typed values and
# the call is resolved at compile time.
bindcall(fn, args...) = () -> (fn(args...); nothing)
bindstore(out, fn, args...) = () -> (out[1] = fn(args...); nothing)
bindsweep_block(pb!, outs, f, U, γ) = () -> (for k in eachindex(U); pb!(outs[k], f, U[k], k, γ); end; nothing)
bindsweep_kkt(kkt, kout, f, g, x0, firsts) = () -> (for k in eachindex(firsts); kout[k] = kkt(f, g, x0, firsts[k]); end; nothing)
bindsweep_kkt_block(kkt, kout, f, g, xb) = () -> (for k in eachindex(kout); kout[k] = kkt(f, g, xb, k); end; nothing)

exact_equal(a::Tuple, b::Tuple) = length(a) == length(b) && all(exact_equal(u, v) for (u, v) in zip(a, b))
exact_equal(a::AbstractArray, b::AbstractArray) = size(a) == size(b) && all(i -> exact_equal(a[i], b[i]), eachindex(a, b))
exact_equal(a::BigFloat, b::BigFloat) =
    precision(a) == precision(b) && signbit(a) == signbit(b) && isequal(a, b)
exact_equal(a, b) = a === b
const TOLERANT = get(ENV, "PB_TOLERANT", "0") == "1"
ncoarse = 0
coarse_flat(a::Tuple) = reduce(vcat, [coarse_flat(u) for u in a]; init = Float64[])
coarse_flat(a::AbstractArray) = Float64[Float64(v) for v in vec(collect(a))]
coarse_flat(a::Real) = Float64[Float64(a)]
coarse_flat(a) = Float64[]
function coarse_equal(a, b)
    x, y = coarse_flat(a), coarse_flat(b)
    length(x) == length(y) || return false
    sc = maximum(v -> isfinite(v) ? abs(v) : 0.0, x; init = 0.0)
    all(i -> (isnan(x[i]) == isnan(y[i])) && (isinf(x[i]) == isinf(y[i])) &&
             (!isfinite(x[i]) ? isequal(x[i], y[i]) || isnan(x[i]) : abs(x[i] - y[i]) <= 1e-3 * sc + 1e-300), eachindex(x, y))
end

function value_agrees(a, ref, ::Type{T}, K) where {T}
    e = T(ref)
    a === e && return true
    (isfinite(a) && isfinite(e)) || return false
    abs(a - e) <= max(128 * eps(T) * abs(e), 4 * K * nextfloat(zero(T)))
end

####################################
#  Floating-point helpers (no package code)
####################################

emax_of(::Type{T}) where {T <: Union{Float32, Float64}} = exponent(floatmax(T))
emax_of(::Type{BigFloat}) = 2^62 - 2
emin_of(::Type{T}) where {T <: Union{Float32, Float64}} = exponent(floatmin(T))
emin_of(::Type{BigFloat}) = -(2^62) + 1
prec_of(::Type{T}) where {T <: Union{Float32, Float64}} = precision(T)
prec_of(::Type{BigFloat}) = 256
scaled(::Type{T}, u::Real, e::Integer) where {T <: Union{Float32, Float64}} = ldexp(T(u), Int(e))
scaled(::Type{BigFloat}, u::Real, e::Integer) = Base.ldexp(BigFloat(u), Clong(e))
pow2(::Type{T}, e::Integer) where {T} = scaled(T, 1.0, e)
lgm(m) = round(Int, log2(m) / 2)             # about log2(√m)
other_type(::Type{Float32}) = Float64         # a level of another type, to leave the range of T
other_type(::Type{Float64}) = BigFloat
bigf(x) = BigFloat(x)                         # default 256 bits: exact for Float32/Float64
setprecision(BigFloat, 256)

# Seeds must not depend on the process (Symbol hashes do): hash a string.
seed_of(parts...) = UInt32(hash(join(parts, '|')) % UInt32)

####################################
#  Vectors of entries
####################################

# K groups of m entries, ± U[ulo, uhi) 2^aexp, in the order of the groups; a poison pattern puts NaN or ±Inf
# at fixed positions of every group, and `zero` makes all or half of the groups zero (or -0.0).
function genvec(rng, ::Type{T}, K, m, aexp, poison = :none, zero = :none; ulo = 0.5, uhi = 1.0) where {T}
    v = Vector{T}(undef, K * m)
    for i in eachindex(v)
        u = ulo + (uhi - ulo) * rand(rng)
        v[i] = scaled(T, rand(rng, Bool) ? u : -u, aexp)
    end
    for k in 1:K
        r = (k-1)*m+1:k*m
        if zero === :zero || (zero === :zero_half && isodd(k))
            v[r] .= Base.zero(T)
        elseif zero === :negzero
            v[r] .= -Base.zero(T)
        end
        mid = (m + 1) ÷ 2
        first_, last_ = r[1], r[end]
        if poison === :inf_first;       v[first_] = T(Inf)
        elseif poison === :inf_last;    v[last_] = T(-Inf)
        elseif poison === :inf_mid;     v[r[mid]] = T(Inf)
        elseif poison === :nan_first;   v[first_] = T(NaN)
        elseif poison === :nan_last;    v[last_] = T(NaN)
        elseif poison === :nan_after_inf;  v[first_] = T(Inf); v[last_] = T(NaN)
        elseif poison === :nan_before_inf; v[first_] = T(NaN); v[last_] = T(Inf)
        elseif poison === :inf_pm;      v[first_] = T(Inf); v[last_] = T(-Inf)
        elseif poison !== :none
            error("unknown poison $poison")
        end
    end
    v
end

# the exponent aexp of the entries of a magnitude kind, for groups of m entries
function aexp_of(::Type{T}, mag, m) where {T}
    emax, emin, p = emax_of(T), emin_of(T), prec_of(T)
    mag === :ord && return 0
    mag === :over_mid && return emax ÷ 2 + 3              # the sum of squares overflows, the norm is far below floatmax
    mag === :over_norm && return emax - lgm(m) - 1        # the norm is about 0.09 to 0.36 floatmax (approximate; see the scenarios)
    mag === :over_entries && return emax                  # entries of magnitude 2^(emax-1) to 2^emax, about 0.25 to 0.5 floatmax
    mag === :max_entries && return emax + 1               # entries of about 0.5 to 0.9 floatmax (see genvec)
    mag === :under && return emin + 2                     # entries of 2 to 4 floatmin: every square underflows to zero
    mag === :sub && return emin - 3                       # subnormal entries
    mag === :tiny && return emin ÷ 2 + p ÷ 4              # normal squares, the sum below m floatmin/eps
    mag === :tiny_sub && return emin ÷ 2 - 3              # subnormal squares
    mag === :norm_small && return -4 - lgm(m)             # norms of about 1/8
    error("unknown magnitude $mag")
end

####################################
#  Parameters of a penalty: γ, λ0, weights w, and t = log2 of a target term (rel_term data)
####################################

function static_params(::Type{T}, kind, K, rng) where {T}
    emax, emin = emax_of(T), emin_of(T)
    wv(lo, hi) = T[T(lo + (hi - lo) * rand(rng)) for _ in 1:K]
    p2(e) = pow2(T, e)
    ord_w = wv(0.5, 1.5)
    lgK = ceil(Int, log2(max(K, 1)))
    kind === :ord          && return (γ = T(0.8), λ0 = T(0.6), w = ord_w, t = 0)
    kind === :lam0         && return (γ = T(0.8), λ0 = zero(T), w = ord_w, t = 0)
    kind === :lamInf       && return (γ = T(0.8), λ0 = T(Inf), w = ord_w, t = 0)
    kind === :w0           && return (γ = T(0.8), λ0 = T(0.6), w = zeros(T, K), t = 0)
    kind === :wInf         && return (γ = T(0.8), λ0 = T(0.6), w = fill(T(Inf), K), t = 0)
    kind === :wmix         && return (γ = T(0.8), λ0 = T(0.6), w = T[[0, Inf, 1, 0.5][mod1(k, 4)] for k in 1:K], t = 0)
    kind === :gam0         && return (γ = zero(T), λ0 = T(0.6), w = ord_w, t = 0)
    kind === :gamInf       && return (γ = T(Inf), λ0 = T(0.6), w = ord_w, t = 0)
    kind === :lam0_wInf    && return (γ = T(0.8), λ0 = zero(T), w = fill(T(Inf), K), t = 0)
    kind === :lamInf_w0    && return (γ = T(0.8), λ0 = T(Inf), w = zeros(T, K), t = 0)
    h = emax ÷ 2 + 1
    kind === :thr_over     && return (γ = p2(h), λ0 = p2(h), w = wv(1, 2), t = 0)           # γλ0w in [2^(2h), 2^(2h+1))
    kind === :thr_over_mid && return (γ = p2(h), λ0 = p2(h), w = wv(1/16, 1/8), t = 0)       # γλ0 overflows, γλ0w does not
    l = emin ÷ 2 - 1
    kind === :thr_under     && return (γ = p2(l), λ0 = p2(l), w = wv(0.5, 1), t = 0)         # γλ0w in [2^(2l-1), 2^2l)
    kind === :thr_under_mid && return (γ = p2(l), λ0 = p2(l), w = wv(64, 128), t = 0)        # γλ0 underflows, γλ0w does not
    kind === :ext_over      && return (γ = p2(emax), λ0 = p2(emax), w = [scaled(T, 0.5 + 0.5rand(rng), -emax) for _ in 1:K], t = 0)
    kind === :ext_under     && return (γ = p2(emin), λ0 = p2(emin), w = [scaled(T, 0.5 + 0.5rand(rng), emax) for _ in 1:K], t = 0)
    kind === :val_lw_under  && return (γ = T(0.8), λ0 = p2(emin ÷ 2 - 2), w = [scaled(T, 0.5 + 0.5rand(rng), emin ÷ 2 - 2) for _ in 1:K], t = 0)
    kind === :val_tu        && return (γ = T(0.8), λ0 = one(T), w = wv(1, 1.5) .* floatmin(T), t = 0)
    kind === :val_ext_over  && return (γ = T(0.8), λ0 = p2(emax), w = [scaled(T, 0.5 + 0.5rand(rng), emax) for _ in 1:K], t = emax - 5 - lgK)
    kind === :val_ext_under && return (γ = T(0.8), λ0 = p2(emin), w = [scaled(T, 0.5 + 0.5rand(rng), emin) for _ in 1:K], t = 0)
    # levels of another type, out of the range of T, with a finite value 2^t at the norm that rel_term data give
    kind === :lvl_sub   && return (γ = T(0.8), λ0 = p2(emin - 3), w = ones(T, K), t = 0)      # a subnormal level
    kind === :lvl_big   && return (γ = T(0.8), λ0 = scaled(other_type(T), 1.0, emax + 20), w = ones(T, K), t = 40)
    kind === :lvl_small && return (γ = T(0.8), λ0 = scaled(other_type(T), 1.0, emin - 20), w = ones(T, K), t = -20)
    error("unknown parameter kind $kind")
end

# parameters that depend on the norms nk (BigFloat) of the groups of the data
function data_params(::Type{T}, kind, nk, rng, unit_w = false) where {T}
    K = length(nk)
    w = unit_w ? ones(T, K) : T[T(0.5 + rand(rng)) for _ in 1:K]
    ns = sort(nk)[(K + 1) ÷ 2]
    if kind === :tuned
        # the threshold is about half the norm: γλ0w in [0.2, 0.6] ‖x‖
        return (γ = T(0.8), λ0 = T(bigf(0.5) * ns / bigf(0.8)), w = w, t = 0)
    end
    frac, r = kind === :val_near ? (0.8, 0.1) : kind === :val_over ? (4.0, 0.1) : kind === :val_fin ? (1/16, 0.3) :
              error("unknown data parameter kind $kind")
    S = sum(bigf(w[k]) * nk[k] for k in 1:K)
    λ0 = T(bigf(frac) * bigf(floatmax(T)) / S)
    γ = T(bigf(r) * ns / (bigf(λ0) * bigf(sum(w) / K)))
    (γ = γ, λ0 = λ0, w = w, t = 0)
end
const DATA_FIRST = (:tuned, :val_near, :val_over, :val_fin)

# A lower estimate of log2(γλ0w₁) from exponents only (valid also where the product leaves the range of T): the sum
# of the three floors of log2, so that log2(γλ0w₁) - cexp is in [0, 3)
cexp(γ, λ0, w1) = Int(Int128(exponent(γ)) + Int128(exponent(λ0)) + Int128(exponent(w1)))

####################################
#  Validation of the data of a group case, without package code
####################################

fail(what, ctx) = error("input validation failed ($what): $ctx")

# req: tuple of symbols or symbol => bounds pairs; all groups must have the property
function validate_groups!(::Type{T}, name, U, p) where {T}
    req = p.req
    γ, λ0, w = p.γ, p.λ0, p.w
    K = length(U)
    emax = emax_of(T)
    fl = T !== BigFloat
    m = length(U[1])
    ctx = "$T $name m=$m"
    finite_data = all(u -> all(isfinite, u), U)
    for r in req
        tag = r isa Pair ? r.first : r
        if tag === :has_inf
            all(u -> any(isinf, u), U) || fail(tag, ctx)
        elseif tag === :has_nan
            all(u -> any(isnan, u), U) || fail(tag, ctx)
        elseif tag === :zero_all
            all(u -> all(iszero, u), U) || fail(tag, ctx)
        elseif tag === :zero_half
            count(u -> all(iszero, u), U) == (K + 1) ÷ 2 || fail(tag, ctx)
        elseif tag === :negzero
            all(u -> all(x -> iszero(x) && signbit(x), u), U) || fail(tag, ctx)
        elseif tag === :sumsq_over          # the ordinary sum of squares in T overflows, no entry does
            finite_data || fail(tag, ctx)
            all(u -> isinf(foldl(+, abs2.(u); init = zero(T))), U) || fail(tag, ctx)
        elseif tag === :sumsq_zero          # the ordinary sum of squares is zero although the group is not
            all(u -> iszero(foldl(+, abs2.(u); init = zero(T))) && any(!iszero, u), U) || fail(tag, ctx)
        elseif tag === :sumsq_below         # the ordinary sum is below m floatmin/eps, where a square may have lost accuracy
            all(u -> foldl(+, abs2.(u); init = zero(T)) < length(u) * (floatmin(T) / eps(T)) && any(!iszero, u), U) || fail(tag, ctx)
        elseif tag === :squares_subnormal   # every square in T is nonzero and below floatmin
            all(u -> all(x -> (q = abs2(x); !iszero(q) && q < floatmin(T)), u), U) || fail(tag, ctx)
        elseif tag === :norm_finite         # the exact norm is in range
            fl && all(u -> sqrt(sum(abs2 ∘ bigf, u)) <= bigf(floatmax(T)), U) || fail(tag, ctx)
        elseif tag === :c_over              # γλ0w overflows in T, whatever the order of the factors
            all(k -> isinf((γ * λ0) * w[k]) && isinf(γ * (λ0 * w[k])), 1:K) || fail(tag, ctx)
        elseif tag === :ab_over             # γλ0 overflows in T, γλ0w is in range
            fl && isinf(γ * λ0) && all(k -> floatmin(T) <= bigf(γ) * bigf(λ0) * bigf(w[k]) <= bigf(floatmax(T)), 1:K) || fail(tag, ctx)
        elseif tag === :c_under             # γλ0w is below floatmin in T
            all(k -> (γ * λ0) * w[k] < floatmin(T), 1:K) || fail(tag, ctx)
        elseif tag === :ab_under            # γλ0 is below floatmin, γλ0w is in range
            fl && γ * λ0 < floatmin(T) && all(k -> floatmin(T) <= bigf(γ) * bigf(λ0) * bigf(w[k]) <= bigf(floatmax(T)), 1:K) || fail(tag, ctx)
        elseif tag === :lvl_over            # the level (of another type) is above floatmax(T)
            fl && bigf(λ0) > bigf(floatmax(T)) || fail(tag, ctx)
        elseif tag === :lvl_under           # the level is positive and below floatmin(T)
            fl && 0 < bigf(λ0) < bigf(floatmin(T)) || fail(tag, ctx)
        elseif tag === :lw_over             # λ0w overflows in T
            all(k -> isinf(λ0 * w[k]), 1:K) || fail(tag, ctx)
        elseif tag === :lw_under            # λ0w is below floatmin in T
            all(k -> λ0 * w[k] < floatmin(T), 1:K) || fail(tag, ctx)
        elseif tag === :finite_result       # the exact proximal point is finite when rounded to T
            fl && finite_data || fail(tag, ctx)
            for k in 1:K
                bx = bigf.(U[k]); n = sqrt(sum(abs2, bx))
                c = bigf(γ) * bigf(λ0) * bigf(w[k])
                t = iszero(n) ? bigf(0) : max(1 - c / n, bigf(0))
                all(x -> isfinite(T(t * x)), bx) || fail(tag, ctx)
            end
        elseif tag === :val_range           # the exact value λ0 Σ wₖ‖x[Gₖ]‖ is in [lo, hi] floatmax(T)
            lo, hi = r.second
            fl && finite_data || fail(tag, ctx)
            V = bigf(λ0) * sum(bigf(w[k]) * sqrt(sum(abs2 ∘ bigf, U[k])) for k in 1:K)
            (bigf(lo) * bigf(floatmax(T)) <= V && (isinf(hi) || V <= bigf(hi) * bigf(floatmax(T)))) || fail("$tag V/floatmax = $(Float64(V / floatmax(T)))", ctx)
        elseif tag === :val_below_Kmin      # 0 < V < K floatmin(T): the terms round into the subnormal range
            fl && finite_data || fail(tag, ctx)
            V = bigf(λ0) * sum(bigf(w[k]) * sqrt(sum(abs2 ∘ bigf, U[k])) for k in 1:K)
            (0 < V < K * bigf(floatmin(T))) || fail(tag, ctx)
        elseif tag === :val_finite          # the exact value is finite and positive in T
            fl && finite_data || fail(tag, ctx)
            V = bigf(λ0) * sum(bigf(w[k]) * sqrt(sum(abs2 ∘ bigf, U[k])) for k in 1:K)
            (0 < V < bigf(floatmax(T)) / 2) || fail(tag, ctx)
        else
            error("unknown requirement $tag")
        end
    end
end

# the exact value of the group penalty, when the data and parameters are finite
function group_value_ref(::Type{T}, U, p) where {T}
    T === BigFloat && return nothing
    (all(u -> all(isfinite, u), U) && isfinite(p.λ0) && all(isfinite, p.w)) || return nothing
    bigf(p.λ0) * sum(bigf(p.w[k]) * sqrt(sum(abs2 ∘ bigf, U[k])) for k in eachindex(U))
end

####################################
#  GroupNormL2 cases
####################################

const TOTAL = 4096                 # entries of the array cases: (TOTAL ÷ m) groups of m entries (256 for BigFloat)
const GRIDS = Dict(
    :full  => [(m, l) for m in (2, 5, 32, 1024) for l in (:range, :perm)],
    :small => [(5, :range), (32, :perm)],
    :long  => [(2^16, :range), (2^16, :perm)])
const GOPS = ("prox", "value_and_prox", "value", "prox_block", "kkt_array", "kkt_block")
const MOPS = ("prox_matrix", "value_and_prox_matrix", "value_matrix", "kkt_matrix")
const POPS = ("prox", "value_and_prox", "prox_block")

# a scenario: name, magnitude, parameter kind, poison and zero patterns, requirements, grid, operations
function S(name; mag = :ord, param = :ord, poison = :none, zero = :none, req = (), grid = :small, ops = GOPS,
           matrix = false, ref = false, types = (Float32, Float64))
    (; name, mag, param, poison, zero, req, grid, ops, matrix, ref, types)
end

function build_group_data(::Type{T}, sc, m, layout, total; unit_w = false, family = "GroupNormL2") where {T}
    rng = MersenneTwister(seed_of("group", sc.name, T, m, layout))
    K = max(1, total ÷ m)
    p = K * m
    order = layout === :perm ? randperm(rng, p) : collect(1:p)
    groups = layout === :perm ? [order[(k-1)*m+1:k*m] for k in 1:K] : [(k-1)*m+1:k*m for k in 1:K]
    ulo, uhi = sc.mag === :max_entries ? (0.5, 0.9) : (0.5, 1.0)
    function entries(aexp)
        genvec(rng, T, K, m, aexp, sc.poison, sc.zero; ulo, uhi)
    end
    if sc.param in DATA_FIRST
        v = entries(aexp_of(T, sc.mag, m))
        nk = [sqrt(sum(abs2 ∘ bigf, v[(k-1)*m+1:k*m])) for k in 1:K]
        par = data_params(T, sc.param, nk, rng, unit_w)
    else
        par = static_params(T, sc.param, K, rng)
        unit_w && (par = merge(par, (w = ones(T, K),)))
        aexp = if sc.mag === :rel_c
            clamp(cexp(par.γ, par.λ0, par.w[1]) + 1 - lgm(m), emin_of(T) - prec_of(T) + 6, emax_of(T) - 2)
        elseif sc.mag === :rel_term
            clamp(par.t - cexp(one(T), par.λ0, par.w[1]) - lgm(m), emin_of(T) - prec_of(T) + 6, emax_of(T) - 2)
        elseif sc.mag === :rel_norm_max
            emax_of(T) - 3 - lgm(m)
        else
            aexp_of(T, sc.mag, m)
        end
        v = entries(aexp)
    end
    x = zeros(T, p)
    for k in 1:K, r in 1:m
        x[groups[k][r]] = v[(k-1)*m+r]
    end
    U = [x[G] for G in groups]
    validate_groups!(T, sc.name, U, (; par..., req = sc.req))
    validate_count!(family)
    (; K, m, p, layout, groups, x, U, par..., vref = sc.ref ? group_value_ref(T, U, par) : nothing)
end

function add_group_cases!(T, sc, m, layout, total)
    d = build_group_data(T, sc, m, layout, total)
    K, p, groups, x = d.K, d.p, d.groups, d.x
    γ, λ0, w = d.γ, d.λ0, d.w
    y = similar(x)
    xm, ym = reshape(x, m, K), reshape(y, m, K)              # matrix arguments of the same memory
    x0 = zeros(T, p); x0m = reshape(x0, m, K)                # a zero point for the group screen
    firsts = [G[1] for G in groups]
    outs = [similar(u) for u in d.U]
    kout = Vector{T}(undef, K)
    out1 = Vector{T}(undef, 1)
    rest = "$(sc.name)/m$m/$layout"
    ref = d.vref === nothing ? nothing : (d.vref, K)
    mk(PB) = PB.GroupNormL2(λ0, groups, w)
    for op in sc.ops
        op == "prox_block" && layout !== :range && continue
        if op == "prox"
            addspec!("GroupNormL2", op, T, rest, p, PB -> begin
                f = mk(PB)
                (op = bindcall(PB.prox!, y, f, x, γ), check = () -> (PB.prox!(y, f, x, γ); copy(y)))
            end)
        elseif op == "value_and_prox"
            addspec!("GroupNormL2", op, T, rest, p, PB -> begin
                f = mk(PB)
                (op = bindstore(out1, PB.value_and_prox!, y, f, x, γ),
                 check = () -> (v = PB.value_and_prox!(y, f, x, γ); (v, copy(y))))
            end)
        elseif op == "value"
            addspec!("GroupNormL2", op, T, rest, p, PB -> begin
                f = mk(PB)
                (op = bindstore(out1, f, x), check = () -> f(x))
            end; ref)
        elseif op == "prox_block"
            addspec!("GroupNormL2", op, T, rest, p, PB -> begin
                f = mk(PB)
                U = d.U
                (op = bindsweep_block(PB.prox_block!, outs, f, U, γ),
                 check = () -> (for k in 1:K; PB.prox_block!(outs[k], f, U[k], k, γ); end; reduce(vcat, outs)))
            end)
        elseif op == "kkt_array"
            addspec!("GroupNormL2", op, T, rest, p, PB -> begin
                f = mk(PB)
                (op = bindsweep_kkt(PB.kkt_violation, kout, f, x, x0, firsts),
                 check = () -> [PB.kkt_violation(f, x, x0, firsts[k]) for k in 1:K])
            end)
        elseif op == "kkt_block"
            addspec!("GroupNormL2", op, T, rest, p, PB -> begin
                f = mk(PB)
                xb = PB.BlockIterate(zeros(T, p), f.groups)
                (op = bindsweep_kkt_block(PB.kkt_violation, kout, f, x, xb),
                 check = () -> [PB.kkt_violation(f, x, xb, k) for k in 1:K])
            end)
        end
    end
    sc.matrix || return
    for op in MOPS
        sub = replace(op, "_matrix" => "")
        if sub == "prox"
            addspec!("GroupNormL2", op, T, rest, p, PB -> begin
                f = mk(PB)
                (op = bindcall(PB.prox!, ym, f, xm, γ), check = () -> (PB.prox!(ym, f, xm, γ); copy(ym)))
            end)
        elseif sub == "value_and_prox"
            addspec!("GroupNormL2", op, T, rest, p, PB -> begin
                f = mk(PB)
                (op = bindstore(out1, PB.value_and_prox!, ym, f, xm, γ),
                 check = () -> (v = PB.value_and_prox!(ym, f, xm, γ); (v, copy(ym))))
            end)
        elseif sub == "value"
            addspec!("GroupNormL2", op, T, rest, p, PB -> begin
                f = mk(PB)
                (op = bindstore(out1, f, xm), check = () -> f(xm))
            end; ref)
        elseif sub == "kkt"
            addspec!("GroupNormL2", op, T, rest, p, PB -> begin
                f = mk(PB)
                (op = bindsweep_kkt(PB.kkt_violation, kout, f, xm, x0m, firsts),
                 check = () -> [PB.kkt_violation(f, xm, x0m, firsts[k]) for k in 1:K])
            end)
        end
    end
end

group_scenarios = Any[]
add!(sc) = push!(group_scenarios, sc)
# overflowing sums of squares (:sumsq_over), with entries ±U[1/2, 1) 2^aexp, that is, of magnitude between 2^(aexp-1) and
# 2^aexp (the significand is rounded to T, so both endpoints can occur). The ranges relative to floatmax are approximate
# (floatmax is a little below 2^(emax+1)). over_mid: the norm is far below floatmax. over_norm: the norm of m entries is
# about 0.09 to 0.36 floatmax, so it is finite (:norm_finite). over_entries: entries of about 0.25 to 0.5 floatmax, a norm that
# is finite for m = 2, above floatmax for m >= 16 and either in between. over_max: entries of about 0.5 to 0.9 floatmax, a norm
# above floatmax for m >= 4. The proximal points are finite where :finite_result is listed.
add!(S("over_mid", mag = :over_mid, param = :tuned, grid = :full, matrix = true, req = (:sumsq_over, :norm_finite, :finite_result)))
add!(S("over_mid_val", mag = :over_mid, param = :val_fin, grid = :full, matrix = true, ref = true,
       req = (:sumsq_over, :norm_finite, :val_range => (0.02, 0.1))))
add!(S("over_norm", mag = :over_norm, param = :tuned, grid = :full, matrix = true, req = (:sumsq_over, :norm_finite, :finite_result)))
add!(S("over_norm_val", mag = :over_norm, param = :val_fin, grid = :full, matrix = true, ref = true,
       req = (:sumsq_over, :norm_finite, :val_range => (0.02, 0.1))))
add!(S("over_entries", mag = :over_entries, param = :ord, grid = :full, req = (:sumsq_over, :finite_result)))
add!(S("over_entries_val", mag = :over_entries, param = :val_fin, grid = :small, ref = true, req = (:sumsq_over, :val_range => (0.02, 0.1))))
add!(S("over_max", mag = :max_entries, param = :ord, grid = :small, req = (:sumsq_over, :finite_result)))
# underflowing norms: every square underflows (under, sub), or the sum is below m floatmin/eps (tiny, tiny_sub)
add!(S("under", mag = :under, param = :tuned, grid = :full, matrix = true, req = (:sumsq_zero, :finite_result)))
add!(S("sub", mag = :sub, param = :tuned, grid = :full, matrix = true, req = (:sumsq_zero, :finite_result)))
add!(S("tiny", mag = :tiny, param = :tuned, grid = :full, matrix = true, req = (:sumsq_below, :finite_result)))
add!(S("tiny_sub", mag = :tiny_sub, param = :tuned, grid = :full, matrix = true, req = (:sumsq_below, :squares_subnormal, :finite_result)))
# thresholds γλ0w that overflow or underflow, finite results; and the mid cases where only γλ0 is out of range
add!(S("thr_over", mag = :rel_c, param = :thr_over, grid = :full, matrix = true, req = (:c_over, :finite_result)))
add!(S("thr_over_mid", mag = :rel_c, param = :thr_over_mid, grid = :full, matrix = true, req = (:ab_over, :finite_result)))
add!(S("thr_under", mag = :rel_c, param = :thr_under, grid = :full, matrix = true, req = (:c_under, :finite_result)))
add!(S("thr_under_mid", mag = :rel_c, param = :thr_under_mid, grid = :full, matrix = true, req = (:ab_under, :finite_result)))
# extreme finite products: factors near floatmax and floatmin whose product is in range
add!(S("ext_over", mag = :rel_c, param = :ext_over, grid = :small, req = (:ab_over, :finite_result)))
add!(S("ext_under", mag = :rel_c, param = :ext_under, grid = :small, req = (:ab_under, :finite_result)))
# values: near floatmax/2 and above it (the recomputation of Option C), λ0w out of range, sums below K floatmin (#373)
add!(S("val_near", mag = :ord, param = :val_near, grid = :full, ref = true, req = (:val_range => (0.6, 0.9),)))
add!(S("val_over", mag = :ord, param = :val_over, grid = :small, ref = true, req = (:val_range => (2.0, Inf),)))
add!(S("val_lw_under", mag = :ord, param = :val_lw_under, grid = :small, ref = true, req = (:lw_under, :val_finite)))
add!(S("val_tu", mag = :norm_small, param = :val_tu, grid = :full, ref = true, req = (:val_below_Kmin,)))
add!(S("val_ext_over", mag = :rel_term, param = :val_ext_over, grid = :small, ref = true, req = (:lw_over, :val_finite)))
add!(S("val_ext_under", mag = :rel_norm_max, param = :val_ext_under, grid = :small, ref = true, req = (:lw_under, :val_finite)))
# zero groups, signed zeros and a mixture of zero and ordinary groups
add!(S("zero", mag = :ord, zero = :zero, grid = :full, matrix = true, req = (:zero_all,)))
add!(S("zero_half", mag = :ord, zero = :zero_half, grid = :full, matrix = true, req = (:zero_half,)))
add!(S("negzero", mag = :ord, zero = :negzero, grid = :small, req = (:negzero,)))
# zero and infinite levels, weights and steps, against ordinary, overflowing, NaN, infinite and zero data
for reg in (:lam0, :lamInf, :w0, :wInf, :wmix, :gam0, :gamInf, :lam0_wInf, :lamInf_w0)
    for (dat, kw) in ((:ord, (;)), (:over, (; mag = :over_mid, req = (:sumsq_over,))), (:nan, (; poison = :nan_first, req = (:has_nan,))),
                      (:inf, (; poison = :inf_last, req = (:has_inf,))), (:zero, (; zero = :zero, req = (:zero_all,))))
        ops = reg in (:gam0, :gamInf) ? POPS : GOPS
        add!(S("reg_$(reg)_$dat"; param = reg, ops, kw...))
    end
end
# NaN/Inf precedence: Inf and NaN at the first and last positions, with ordinary, overflowing-threshold and
# overflowing-sum data
for pat in (:inf_first, :inf_last, :inf_mid, :nan_first, :nan_last, :nan_after_inf, :nan_before_inf, :inf_pm)
    rq = pat in (:inf_first, :inf_last, :inf_mid, :inf_pm) ? (:has_inf,) : (:has_nan,)
    add!(S("pre_$(pat)_ord", poison = pat, req = rq))
    add!(S("pre_$(pat)_thr", param = :thr_over, poison = pat, req = (rq..., :c_over)))
    add!(S("pre_$(pat)_over", mag = :over_mid, poison = pat, req = rq))
end
# long Float32 recoveries: groups of 2^16 entries whose ordinary Float32 sum of squares overflows or underflows
add!(S("long_over", mag = :over_mid, param = :tuned, grid = :long, types = (Float32,), ops = ("prox", "value_and_prox", "value", "prox_block", "kkt_array"),
       req = (:sumsq_over, :norm_finite, :finite_result)))
add!(S("long_under", mag = :under, param = :tuned, grid = :long, types = (Float32,), ops = ("prox", "value_and_prox", "value", "prox_block", "kkt_array"),
       req = (:sumsq_zero, :finite_result)))
add!(S("long_val", mag = :over_mid, param = :val_fin, grid = :long, types = (Float32,), ops = ("value", "value_and_prox"), ref = true,
       req = (:sumsq_over, :val_range => (0.02, 0.1))))

for sc in group_scenarios, T in sc.types, (m, layout) in GRIDS[sc.grid]
    add_group_cases!(T, sc, m, layout, sc.grid === :long ? 2m : TOTAL)
end

# BigFloat: a few representative inputs, check-only. Exponents are kept so that no sum of three exponents
# leaves Int64 (the D-X case below is the one that does).
big_scenarios = [
    S("over_mid", mag = :over_mid, param = :tuned, req = (:sumsq_over,)),
    S("under", mag = :under, param = :tuned, req = (:sumsq_zero,)),
    S("thr_over", mag = :rel_c, param = :thr_over, req = (:c_over,)),
    S("reg_lamInf_ord", param = :lamInf), S("reg_w0_ord", param = :w0), S("reg_wInf_over", param = :wInf, mag = :over_mid),
    S("reg_gamInf_ord", param = :gamInf, ops = POPS), S("reg_lam0_nan", param = :lam0, poison = :nan_first, req = (:has_nan,)),
    S("pre_inf_last_ord", poison = :inf_last, req = (:has_inf,)), S("pre_nan_after_inf_over", mag = :over_mid, poison = :nan_after_inf, req = (:has_nan,)),
    S("zero", zero = :zero, req = (:zero_all,)), S("ord", mag = :ord, param = :ord),
]
for sc in big_scenarios, (m, layout) in GRIDS[:small]
    add_group_cases!(BigFloat, sc, m, layout, 256)
end

# D-X: a BigFloat value whose sum of exponents leaves Int64. The exact value is 2^(3(2^62-2)), beyond the
# exponent range of MPFR: the reference tree wraps the sum and returns +0, the shared kernel returns +Inf.
let e = 2^62 - 2
    λ = Base.ldexp(BigFloat(1.0), Clong(e)); xv = [Base.ldexp(BigFloat(1.0), Clong(e))]
    Int128(exponent(λ)) + Int128(exponent(λ)) + Int128(exponent(xv[1])) > Int128(typemax(Int64)) || fail("D-X exponents", "BigFloat")
    validate_count!("D-X")
    o1 = [BigFloat(0)]
    addspec!("GroupNormL2", "value", BigFloat, "dx_exponent_sum_beyond_Int64", 1, PB -> begin
        f = PB.GroupNormL2(λ, [1:1], [λ])
        (op = bindstore(o1, f, xv), check = () -> f(xv))
    end; diff = (x = a -> a isa BigFloat && iszero(a) && !signbit(a), y = a -> a isa BigFloat && isinf(a) && a > 0))
end

####################################
#  NormL2 cases
####################################

# The same scenarios as for one group of GroupNormL2 with weight one: dense arrays of 1024 and 65536
# entries (prox!, value_and_prox!, value), and the value of a SparseIterate and a SymmetricSparseIterate
# of 200 × 200 with 400 stored entries and of a BlockIterate of 1024 entries.
const NL2_DENSE = (1024, 65536)
const NL2_STORED = 400

# coordinates of the stored entries, shared by the roles: distinct lower-triangle coordinates of a p × p matrix
function stored_coordinates(rng, p, nstored)
    pool = [(r, c) for c in 1:p for r in c:p]
    shuffle!(rng, pool)
    pool[1:nstored]
end

function add_iterate_values!(family, mkf, T, sc, rest_prefix)
    d = build_group_data(T, sc, NL2_STORED, :range, NL2_STORED; unit_w = true, family)
    p = 200
    coords = stored_coordinates(MersenneTwister(seed_of("coords", sc.name, T)), p, NL2_STORED)
    vals = d.x
    nonzero = [any(!iszero, vals[i:i]) for i in eachindex(vals)]
    out1 = Vector{T}(undef, 1)
    fill_sparse!(M, PB) = (for (i, (r, c)) in enumerate(coords); nonzero[i] && (M[r, c] = vals[i]); end; M)
    fill_sym!(S, PB) = (for (i, (r, c)) in enumerate(coords); nonzero[i] && (S[r, c] = vals[i]); end; S)
    addspec!(family, "value_sparse", T, "$(sc.name)/stored$NL2_STORED", 2NL2_STORED, PB -> begin
        f = mkf(PB, d); M = fill_sparse!(PB.SparseIterate(T, p, p), PB)
        (op = bindstore(out1, f, M), check = () -> f(M))
    end)
    addspec!(family, "value_symmetric", T, "$(sc.name)/stored$NL2_STORED", 2NL2_STORED, PB -> begin
        f = mkf(PB, d); S = fill_sym!(PB.SymmetricSparseIterate(T, p), PB)
        (op = bindstore(out1, f, S), check = () -> f(S))
    end)
end

function add_normL2_cases!(T, sc, dense_sizes)
    unit(PB, d) = PB.NormL2(d.λ0)
    for n in dense_sizes
        d = build_group_data(T, sc, n, :range, n; unit_w = true, family = "NormL2")
        x, γ, λ = d.x, d.γ, d.λ0
        y = similar(x); out1 = Vector{T}(undef, 1)
        rest = "$(sc.name)/n$n"
        ref = d.vref === nothing ? nothing : (d.vref, 1)
        for op in sc.ops
            if op == "prox"
                addspec!("NormL2", op, T, rest, n, PB -> begin
                    f = PB.NormL2(λ)
                    (op = bindcall(PB.prox!, y, f, x, γ), check = () -> (PB.prox!(y, f, x, γ); copy(y)))
                end)
            elseif op == "value_and_prox"
                addspec!("NormL2", op, T, rest, n, PB -> begin
                    f = PB.NormL2(λ)
                    (op = bindstore(out1, PB.value_and_prox!, y, f, x, γ),
                     check = () -> (v = PB.value_and_prox!(y, f, x, γ); (v, copy(y))))
                end)
            elseif op == "value"
                addspec!("NormL2", op, T, rest, n, PB -> begin
                    f = PB.NormL2(λ)
                    (op = bindstore(out1, f, x), check = () -> f(x))
                end; ref)
                if sc.iter && n == 1024
                    groups = [(k-1)*32+1:k*32 for k in 1:32]
                    addspec!("NormL2", "value_block", T, rest, n, PB -> begin
                        f = PB.NormL2(λ); xb = PB.BlockIterate(copy(x), groups)
                        (op = bindstore(out1, f, xb), check = () -> f(xb))
                    end)
                end
            end
        end
    end
    sc.iter && add_iterate_values!("NormL2", unit, T, sc, "")
end

const NOPS = ("prox", "value_and_prox", "value")
function SN(name; ops = NOPS, iter = false, kw...)
    merge(S(name; ops, kw...), (; iter))
end
nl2_scenarios = Any[
    SN("over_mid", mag = :over_mid, param = :tuned, iter = true, req = (:sumsq_over, :norm_finite, :finite_result)),
    SN("over_mid_val", mag = :over_mid, param = :val_fin, iter = true, ref = true, req = (:sumsq_over, :norm_finite, :val_range => (0.02, 0.1))),
    SN("over_norm", mag = :over_norm, param = :tuned, req = (:sumsq_over, :norm_finite, :finite_result)),
    SN("over_norm_val", mag = :over_norm, param = :val_fin, ref = true, req = (:sumsq_over, :norm_finite, :val_range => (0.02, 0.1))),
    SN("over_entries", mag = :over_entries, param = :ord, req = (:sumsq_over, :finite_result)),
    SN("over_entries_val", mag = :over_entries, param = :val_fin, ref = true, req = (:sumsq_over, :val_range => (0.02, 0.1))),
    SN("over_max", mag = :max_entries, param = :ord, iter = true, req = (:sumsq_over, :finite_result)),
    SN("under", mag = :under, param = :tuned, iter = true, req = (:sumsq_zero, :finite_result)),
    SN("sub", mag = :sub, param = :tuned, iter = true, req = (:sumsq_zero, :finite_result)),
    SN("tiny", mag = :tiny, param = :tuned, req = (:sumsq_below, :finite_result)),
    SN("tiny_sub", mag = :tiny_sub, param = :tuned, iter = true, req = (:sumsq_below, :squares_subnormal, :finite_result)),
    SN("thr_over", mag = :rel_c, param = :thr_over, req = (:c_over, :finite_result)),
    SN("thr_under", mag = :rel_c, param = :thr_under, req = (:c_under, :finite_result)),
    SN("val_near", mag = :ord, param = :val_near, ref = true, req = (:val_range => (0.6, 0.9),)),
    SN("val_over", mag = :ord, param = :val_over, ref = true, req = (:val_range => (2.0, Inf),)),
    SN("lvl_sub", mag = :rel_norm_max, param = :lvl_sub, ref = true, req = (:lw_under, :val_finite)),
    # a level out of the range of T, of another type (Float64 for Float32, BigFloat for Float64), finite values
    SN("lvl_big", mag = :rel_term, param = :lvl_big, iter = true, ref = true, req = (:lvl_over, :val_finite)),
    SN("lvl_small", mag = :rel_term, param = :lvl_small, iter = true, ref = true, req = (:lvl_under, :val_finite)),
    SN("zero", zero = :zero, iter = true, req = (:zero_all,)),
    SN("negzero", zero = :negzero, req = (:negzero,)),
]
for pat in (:inf_first, :inf_last, :inf_mid, :nan_first, :nan_last, :nan_after_inf, :nan_before_inf, :inf_pm)
    rq = pat in (:inf_first, :inf_last, :inf_mid, :inf_pm) ? (:has_inf,) : (:has_nan,)
    push!(nl2_scenarios, SN("pre_$(pat)_ord", poison = pat, req = rq, iter = pat in (:nan_first, :inf_last, :nan_after_inf)))
    push!(nl2_scenarios, SN("pre_$(pat)_over", mag = :over_mid, poison = pat, req = rq))
end
for reg in (:lamInf, :lam0, :gam0, :gamInf)
    for (dat, kw) in ((:ord, (;)), (:over, (; mag = :over_mid, req = (:sumsq_over,))), (:nan, (; poison = :nan_first, req = (:has_nan,))),
                      (:zero, (; zero = :zero, req = (:zero_all,))))
        reg in (:gam0, :gamInf) && dat === :zero && continue
        reg === :lam0 && dat in (:over, :zero) && continue
        ops = reg in (:gam0, :gamInf) ? ("prox", "value_and_prox") : NOPS
        push!(nl2_scenarios, SN("reg_$(reg)_$dat"; param = reg, ops, iter = reg === :lamInf && dat in (:ord, :zero), kw...))
    end
end
for sc in nl2_scenarios, T in (Float32, Float64)
    add_normL2_cases!(T, sc, NL2_DENSE)
end
# BigFloat: check-only, a few representative inputs of 256 entries
for sc in (SN("over_mid", mag = :over_mid, param = :tuned, req = (:sumsq_over,)), SN("under", mag = :under, param = :tuned, req = (:sumsq_zero,)),
           SN("thr_over", mag = :rel_c, param = :thr_over, req = (:c_over,)), SN("reg_lamInf_ord", param = :lamInf),
           SN("reg_lamInf_zero", param = :lamInf, zero = :zero, req = (:zero_all,)), SN("pre_nan_first_ord", poison = :nan_first, req = (:has_nan,)),
           SN("pre_inf_last_over", mag = :over_mid, poison = :inf_last, req = (:has_inf,)), SN("zero", zero = :zero, req = (:zero_all,)))
    add_normL2_cases!(BigFloat, sc, (256,))
end

####################################
#  Pairs of arrays: SqrNormL2 values and norm_diff
####################################

# Properties of x - y (or of x), computed in T without package code.
function validate_pair!(::Type{T}, name, x, y, req, lam) where {T}
    ctx = "$T $name"
    fl = T !== BigFloat
    d = y === nothing ? x : [x[i] - y[i] for i in eachindex(x)]
    for r in req
        tag = r isa Pair ? r.first : r
        if tag === :diff_over               # finite x and y whose difference overflows in T
            all(isfinite, x) && (y === nothing || all(isfinite, y)) && any(isinf, d) || fail(tag, ctx)
        elseif tag === :sumsq_over          # the difference is finite and its ordinary sum of squares overflows
            all(isfinite, d) && isinf(foldl(+, abs2.(d); init = zero(T))) || fail(tag, ctx)
        elseif tag === :sumsq_zero
            s = foldl(+, abs2.(d); init = zero(T)); (iszero(s) && any(!iszero, d)) || fail(tag, ctx)
        elseif tag === :sumsq_below
            s = foldl(+, abs2.(d); init = zero(T)); (s < length(d) * (floatmin(T) / eps(T)) && any(!iszero, d)) || fail(tag, ctx)
        elseif tag === :equal
            (y !== nothing && x == y && all(isfinite, x) && any(!iszero, x)) || fail(tag, ctx)
        elseif tag === :zero_diff
            all(iszero, d) || fail(tag, ctx)
        elseif tag === :has_nonfinite
            any(!isfinite, d) || fail(tag, ctx)
        elseif tag === :lvl_out
            fl && (bigf(lam) > bigf(floatmax(T)) || bigf(lam) < bigf(floatmin(T))) || fail(tag, ctx)
        elseif tag === :val_finite          # the exact value λ/2 Σ (x - y)² is finite and normal in T
            fl || fail(tag, ctx)
            V = bigf(lam) / 2 * sum(abs2, bigf.(x) .- (y === nothing ? 0 : bigf.(y)))
            (bigf(floatmin(T)) < V < bigf(floatmax(T)) / 4) || fail("$tag $(Float64(V))", ctx)
        else
            error("unknown requirement $tag")
        end
    end
end
sumsq_big(x, y) = sum(abs2, bigf.(x) .- (y === nothing ? 0 : bigf.(y)))

# a scenario of a pair: the magnitude, how the second array is made, poison, and the level kind
function SP(name; mag = :ord, center = :none, lev = :one, poison = :none, zero = :none, req = (), iter = false, ref = false)
    (; name, mag, center, lev, poison, zero, req, iter, ref)
end

function build_pair(::Type{T}, sc, n, family) where {T}
    rng = MersenneTwister(seed_of(family, sc.name, T, n))
    ulo, uhi = sc.mag === :max_entries ? (0.5, 0.9) : (0.5, 1.0)
    a = aexp_of(T, sc.mag, n)
    x = genvec(rng, T, 1, n, a, sc.poison, sc.zero; ulo, uhi)
    y = sc.center === :none ? nothing : sc.center === :neg ? -x : sc.center === :equal ? copy(x) :
        genvec(rng, T, 1, n, a, sc.poison, sc.zero; ulo, uhi)
    lam = one(T)
    if sc.lev === :val
        k = round(Int, emax_of(T) ÷ 2 - Float64(log2(sumsq_big(x, y))))
        abs(k) < emax_of(T) - 8 || fail("level out of range for :val ($k)", "$T $(sc.name)")
        lam = pow2(T, k)
    elseif sc.lev === :lvl
        k = round(Int, 40 - Float64(log2(sumsq_big(x, y))))
        lam = scaled(other_type(T), 1.0, k)
    end
    validate_pair!(T, sc.name, x, y, sc.req, lam)
    validate_count!(family)
    S = (sc.ref && T !== BigFloat) ? sumsq_big(x, y) : nothing
    (; x, y, lam, S)
end

function add_sqrnorm_cases!(T, sc, n)
    d = build_pair(T, sc, n, "SqrNormL2")
    x, y, λ = d.x, d.y, d.lam
    out1 = Vector{T}(undef, 1)
    rest = "$(sc.name)/n$n"
    sref = d.S === nothing ? nothing : (bigf(λ) / 2 * d.S, 1)
    nref = d.S === nothing || sc.lev !== :one ? nothing : sref
    addspec!("SqrNormL2", "value", T, rest, n, PB -> begin
        f = PB.SqrNormL2(λ)
        (op = bindstore(out1, f, x), check = () -> f(x))
    end; ref = sc.center === :none ? sref : nothing)
    if sc.center !== :none
        addspec!("SqrNormL2", "value_center", T, rest, n, PB -> begin
            f = PB.SqrNormL2(λ, y)
            (op = bindstore(out1, f, x), check = () -> f(x))
        end; ref = sref)
    end
    if sc.iter && sc.zero === :none && n == NL2_DENSE[1]
        dd = build_group_data(T, merge(S(sc.name; mag = sc.mag, poison = sc.poison, zero = sc.zero), (; param = :ord)), NL2_STORED, :range, NL2_STORED; family = "SqrNormL2")
        p = 200
        coords = stored_coordinates(MersenneTwister(seed_of("coords", sc.name, T)), p, NL2_STORED)
        vals = dd.x
        fsq = sc.lev === :one ? one(T) : λ
        addspec!("SqrNormL2", "value_sparse", T, "$(sc.name)/stored$NL2_STORED", 2NL2_STORED, PB -> begin
            f = PB.SqrNormL2(fsq); M = PB.SparseIterate(T, p, p)
            for (i, (r, c)) in enumerate(coords); iszero(vals[i]) || (M[r, c] = vals[i]); end
            (op = bindstore(out1, f, M), check = () -> f(M))
        end)
        addspec!("SqrNormL2", "value_symmetric", T, "$(sc.name)/stored$NL2_STORED", 2NL2_STORED, PB -> begin
            f = PB.SqrNormL2(fsq); Sy = PB.SymmetricSparseIterate(T, p)
            for (i, (r, c)) in enumerate(coords); iszero(vals[i]) || (Sy[r, c] = vals[i]); end
            (op = bindstore(out1, f, Sy), check = () -> f(Sy))
        end)
    end
end

sqr_scenarios = [
    SP("over_mid_val", mag = :over_mid, center = :indep, lev = :val, iter = true, ref = true, req = (:sumsq_over, :val_finite)),
    SP("over_mid_one", mag = :over_mid, center = :indep, lev = :one, iter = true, req = (:sumsq_over,)),
    SP("over_norm_lvl", mag = :over_norm, center = :indep, lev = :lvl, ref = true, req = (:sumsq_over, :lvl_out, :val_finite)),
    SP("over_diff_lvl", mag = :max_entries, center = :neg, lev = :lvl, ref = true, req = (:diff_over, :lvl_out, :val_finite)),
    SP("over_max_lvl", mag = :max_entries, center = :none, lev = :lvl, iter = true, ref = true, req = (:sumsq_over, :lvl_out, :val_finite)),
    SP("under_one", mag = :under, center = :indep, iter = true, req = (:sumsq_zero,)),
    SP("under_lvl", mag = :under, center = :indep, lev = :lvl, iter = true, ref = true, req = (:sumsq_zero, :lvl_out, :val_finite)),
    SP("sub_one", mag = :sub, center = :indep, req = (:sumsq_zero,)),
    SP("sub_lvl", mag = :sub, center = :indep, lev = :lvl, ref = true, req = (:sumsq_zero, :lvl_out, :val_finite)),
    SP("tiny_one", mag = :tiny, center = :indep, req = (:sumsq_below,)),
    SP("tiny_sub_lvl", mag = :tiny_sub, center = :indep, lev = :lvl, iter = true, ref = true, req = (:sumsq_below, :lvl_out, :val_finite)),
    SP("equal", mag = :ord, center = :equal, req = (:equal,)),
    SP("zero", mag = :ord, center = :equal, zero = :zero, req = (:zero_diff,)),
    SP("nan", mag = :ord, center = :indep, poison = :nan_first, iter = true, req = (:has_nonfinite,)),
    SP("inf", mag = :ord, center = :indep, poison = :inf_last, iter = true, req = (:has_nonfinite,)),
    SP("inf_pm", mag = :ord, center = :indep, poison = :inf_pm, req = (:has_nonfinite,)),
    SP("nan_after_inf", mag = :over_mid, center = :indep, poison = :nan_after_inf, req = (:has_nonfinite,)),
]
for sc in sqr_scenarios, T in (Float32, Float64), n in NL2_DENSE
    add_sqrnorm_cases!(T, sc, n)
end

function add_norm_diff_cases!(T, sc, n)
    d = build_pair(T, sc, n, "norm_diff")
    x, y = d.x, d.y
    out1 = Vector{T}(undef, 1)
    ref = d.S === nothing ? nothing : (sqrt(d.S), n)
    addspec!("norm_diff", "norm_diff", T, "$(sc.name)/n$n", n, PB -> begin
        (op = bindstore(out1, PB.norm_diff, x, y), check = () -> PB.norm_diff(x, y))
    end; ref)
    if sc.iter && n == NL2_DENSE[1]
        dd = build_group_data(T, S(sc.name; mag = sc.mag, poison = sc.poison, zero = sc.zero), NL2_STORED, :range, NL2_STORED; family = "norm_diff")
        d2 = build_group_data(T, S(sc.name * "_b"; mag = sc.mag, poison = sc.poison, zero = sc.zero), NL2_STORED, :range, NL2_STORED; family = "norm_diff")
        p = 64
        coords = stored_coordinates(MersenneTwister(seed_of("coords", sc.name, T)), p, 300)
        v1, v2 = dd.x[1:300], d2.x[1:300]
        addspec!("norm_diff", "norm_diff_symmetric", T, "$(sc.name)/p$p", p * p, PB -> begin
            A = PB.SymmetricSparseIterate(T, p); B = PB.SymmetricSparseIterate(T, p)
            for (i, (r, c)) in enumerate(coords); iszero(v1[i]) || (A[r, c] = v1[i]); iszero(v2[i]) || (B[r, c] = v2[i]); end
            (op = bindstore(out1, PB.norm_diff, A, B), check = () -> PB.norm_diff(A, B))
        end)
    end
end
nd_scenarios = [
    SP("over_diff", mag = :max_entries, center = :neg, ref = true, req = (:diff_over,)),
    SP("over_mid", mag = :over_mid, center = :indep, iter = true, ref = true, req = (:sumsq_over,)),
    SP("over_norm", mag = :over_norm, center = :indep, ref = true, req = (:sumsq_over,)),
    SP("under", mag = :under, center = :indep, ref = true, req = (:sumsq_zero,)),
    SP("sub", mag = :sub, center = :indep, ref = true, req = (:sumsq_zero,)),
    SP("tiny", mag = :tiny, center = :indep, ref = true, req = (:sumsq_below,)),
    SP("tiny_sub", mag = :tiny_sub, center = :indep, iter = true, ref = true, req = (:sumsq_below,)),
    SP("equal", mag = :ord, center = :equal, req = (:equal,)),
    SP("nan", mag = :ord, center = :indep, poison = :nan_first, iter = true, req = (:has_nonfinite,)),
    SP("inf", mag = :ord, center = :indep, poison = :inf_last, req = (:has_nonfinite,)),
    SP("inf_pm", mag = :ord, center = :indep, poison = :inf_pm, req = (:has_nonfinite,)),
    SP("nan_after_inf", mag = :over_mid, center = :indep, poison = :nan_after_inf, req = (:has_nonfinite,)),
]
for sc in nd_scenarios, T in (Float32, Float64), n in NL2_DENSE
    add_norm_diff_cases!(T, sc, n)
end

####################################
#  LeastSquaresLoss values
####################################

const LS_ROWS, LS_COLS = 256, 16
# the residual Y - Xβ in T, with the entries of Xβ summed in order (no package code)
ls_residual(X, β, Y) = [Y[i] - foldl(+, (X[i, j] * β[j] for j in axes(X, 2)); init = zero(eltype(X))) for i in axes(X, 1)]

function SL(name; xmag = :ord, ymag = :ord, bmag = :ord, ymode = :rand, poison = :none, req = (), ref = false)
    (; name, xmag, ymag, bmag, ymode, poison, req, ref)
end
function add_ls_cases!(T, sc)
    rng = MersenneTwister(seed_of("ls", sc.name, T))
    n, p = LS_ROWS, LS_COLS
    X = reshape(genvec(rng, T, 1, n * p, aexp_of(T, sc.xmag, n * p)), n, p)
    β = genvec(rng, T, 1, p, sc.bmag isa Int ? sc.bmag : aexp_of(T, sc.bmag, p))
    Y = genvec(rng, T, 1, n, aexp_of(T, sc.ymag, n))
    if sc.xmag === :ord
        X = randn(rng, T, n, p)
    end
    if sc.ymode === :zero
        Y .= zero(T); β .= zero(T)
    elseif sc.ymode === :fit
        mul!(Y, X, β)
    end
    sc.poison === :ynan && (Y[1] = T(NaN))
    sc.poison === :yinf && (Y[end] = T(Inf))
    sc.poison === :xinf && (X[1, 1] = T(Inf))
    r = ls_residual(X, β, Y)
    for rq in sc.req
        if rq === :sumsq_over
            (all(isfinite, r) && isinf(foldl(+, abs2.(r); init = zero(T)))) || fail(rq, "$T $(sc.name)")
        elseif rq === :prod_over              # a finite product Xβ that overflows in T
            (all(isfinite, X) && all(isfinite, β) && all(isfinite, Y) && any(isinf, r)) || fail(rq, "$T $(sc.name)")
        elseif rq === :sumsq_zero
            (iszero(foldl(+, abs2.(r); init = zero(T))) && any(!iszero, r)) || fail(rq, "$T $(sc.name)")
        elseif rq === :sumsq_below
            (foldl(+, abs2.(r); init = zero(T)) < n * (floatmin(T) / eps(T)) && any(!iszero, r)) || fail(rq, "$T $(sc.name)")
        elseif rq === :res_zero
            (all(iszero, r) || (T <: AbstractFloat && X * β == Y)) || fail(rq, "$T $(sc.name)")
        elseif rq === :has_nonfinite
            any(!isfinite, r) || fail(rq, "$T $(sc.name)")
        else
            error("unknown requirement $rq")
        end
    end
    validate_count!("LeastSquaresLoss")
    out1 = Vector{T}(undef, 1)
    ref = nothing
    if sc.ref
        res = [bigf(Y[i]) - sum(bigf(X[i, j]) * bigf(β[j]) for j in 1:p) for i in 1:n]
        ref = (sum(abs2, res) / 2, n)
    end
    addspec!("LeastSquaresLoss", "value", T, sc.name, n * p, PB -> begin
        f = PB.LeastSquaresLoss(Y, X)
        (op = bindstore(out1, f, β), check = () -> f(β))
    end; ref)
end
ls_scenarios = [
    SL("over_mid", ymag = :over_mid, ref = true, req = (:sumsq_over,)),
    SL("over_norm", ymag = :over_norm, ref = true, req = (:sumsq_over,)),
    SL("prod_over", xmag = :over_entries, bmag = 1, req = (:prod_over,)),
    SL("under", xmag = :under, ymag = :under, req = (:sumsq_zero,)),
    SL("sub", xmag = :sub, ymag = :sub, req = (:sumsq_zero,)),
    SL("tiny", xmag = :tiny, ymag = :tiny, req = (:sumsq_below,)),
    SL("tiny_sub", xmag = :tiny_sub, ymag = :tiny_sub, req = (:sumsq_below,)),
    SL("zero_residual", ymode = :zero, req = (:res_zero,)),
    SL("exact_fit", ymode = :fit, req = (:res_zero,)),
    SL("nan", poison = :ynan, req = (:has_nonfinite,)),
    SL("inf", poison = :yinf, req = (:has_nonfinite,)),
    SL("x_inf", poison = :xinf, req = (:has_nonfinite,)),
]
for sc in ls_scenarios, T in (Float32, Float64)
    add_ls_cases!(T, sc)
end

####################################
#  NormL1 values and proximal operators (#373 underflow, infinite weights and levels)
####################################

function SO(name; kind, ops, req = (), iter = false, ref = false)
    (; name, kind, ops, req, iter, ref)
end

# x, weights (a vector, or nothing), level, step of an NormL1 scenario
function l1_data(::Type{T}, sc, n) where {T}
    rng = MersenneTwister(seed_of("l1", sc.name, T, n))
    emax, emin = emax_of(T), emin_of(T)
    wv(lo, hi) = T[T(lo + (hi - lo) * rand(rng)) for _ in 1:n]
    xv(a, poison = :none, zero = :none) = genvec(rng, T, 1, n, a, poison, zero)
    w = wv(0.5, 1.5); x = xv(0); λ = T(0.6); γ = T(0.8)
    k = sc.kind
    if k === :underflow                           # #373: every product rounds below floatmin
        w = fill(floatmin(T), n)
        x = fill(T(ldexp(big(1.0), -9) + ldexp(big(1.0), -precision(T))), n); λ = one(T)
    elseif k === :underflow_rand
        w = wv(1, 2) .* floatmin(T); x = xv(-4); λ = one(T)
    elseif k in (:winf_nonzero, :winf_zero, :winf_prox)
        w[1:16:end] .= T(Inf)
        k === :winf_zero && (x[1:16:end] .= zero(T))
    elseif k === :winf_special
        w[1] = T(Inf); w[end] = T(Inf); w[2] = T(Inf); w[end-1] = T(Inf)
        x[1] = T(NaN); x[end] = T(-Inf); x[2] = T(Inf); x[end-1] = T(NaN)
    elseif k === :lamInf
        λ = T(Inf)
    elseif k === :lamInf_zero
        λ = T(Inf); x = zeros(T, n)
    elseif k === :lamInf_nan
        λ = T(Inf); x[1] = T(NaN); x[end] = T(Inf)
    elseif k === :lamInf_w0
        λ = T(Inf); w[1:4:end] .= zero(T)
    elseif k === :lw_over
        λ = pow2(T, emax); w = [scaled(T, 0.5 + 0.5rand(rng), emax) for _ in 1:n]
        x = xv(emax - 5 - ceil(Int, log2(n)) - cexp(one(T), λ, w[1]))
    elseif k === :nan_first
        x = xv(0, :nan_first)
    elseif k === :inf_last
        x = xv(0, :inf_last)
    elseif k in (:thr_over, :thr_over_mid, :thr_under, :thr_under_mid, :thr_over_u, :thr_under_u)
        base = k in (:thr_over_u, :thr_over) ? :thr_over : k === :thr_under_u ? :thr_under : k
        par = static_params(T, base, n, rng)
        γ, λ, w = par.γ, par.λ0, par.w
        x = xv(clamp(cexp(γ, λ, w[1]) + 1, emin - prec_of(T) + 6, emax - 2))
    elseif k === :lvl_big
        λ = scaled(other_type(T), 1.0, emax + 20)
        x = xv(40 - cexp(one(T), λ, one(T)) - ceil(Int, log2(n)))
    else
        error("unknown NormL1 scenario $k")
    end
    unweighted = k in (:thr_over_u, :thr_under_u, :lvl_big)
    (; x, w = unweighted ? nothing : w, wv = w, λ, γ)
end

function add_l1_cases!(T, sc, n)
    d = l1_data(T, sc, n)
    x, w, λ, γ, wv = d.x, d.w, d.λ, d.γ, d.wv
    rq = sc.kind
    ctx = "$T NormL1 $(sc.name) n=$n"
    # independent validation of the intended path, in T arithmetic and BigFloat
    if rq in (:underflow, :underflow_rand)
        all(i -> (λ * wv[i]) * abs(x[i]) < floatmin(T) && !iszero((λ * wv[i]) * abs(x[i])) || (λ * wv[i]) * abs(x[i]) == zero(T) && false, 1:n) ||
            all(i -> (λ * wv[i]) * abs(x[i]) < floatmin(T), 1:n) || fail("underflow", ctx)
        sum(bigf(λ) * bigf(wv[i]) * abs(bigf(x[i])) for i in 1:n) > 0 || fail("underflow sum", ctx)
    elseif rq in (:winf_nonzero, :winf_prox)
        (any(isinf, wv) && all(!iszero, x[findall(isinf, wv)])) || fail("infinite weight at nonzero", ctx)
    elseif rq === :winf_zero
        (any(isinf, wv) && all(iszero, x[findall(isinf, wv)])) || fail("infinite weight at zero", ctx)
    elseif rq === :winf_special
        (isinf(wv[1]) && isnan(x[1]) && isinf(x[end]) && isinf(wv[end])) || fail("special", ctx)
    elseif rq in (:lamInf, :lamInf_zero, :lamInf_nan, :lamInf_w0)
        isinf(λ) || fail("infinite level", ctx)
    elseif rq === :lw_over
        all(i -> isinf(λ * wv[i]), 1:n) && all(isfinite, x) && all(!iszero, x) || fail("level-weight product overflows", ctx)
        V = bigf(λ) * sum(bigf(wv[i]) * abs(bigf(x[i])) for i in 1:n)
        (bigf(floatmin(T)) < V < bigf(floatmax(T)) / 2) || fail("lw_over value $(Float64(V))", ctx)
    elseif rq === :lvl_big
        V = bigf(λ) * sum(abs(bigf(x[i])) for i in 1:n)
        (bigf(λ) > bigf(floatmax(T)) && bigf(floatmin(T)) < V < bigf(floatmax(T)) / 2) || fail("level out of range", ctx)
    elseif rq in (:nan_first, :inf_last)
        any(!isfinite, x) || fail("nonfinite", ctx)
    elseif rq in (:thr_over, :thr_over_u)
        all(i -> isinf((γ * λ) * wv[i]), 1:n) || fail("threshold overflows", ctx)
    elseif rq in (:thr_under, :thr_under_u)
        all(i -> (γ * λ) * wv[i] < floatmin(T), 1:n) || fail("threshold underflows", ctx)
    elseif rq === :thr_over_mid
        (isinf(γ * λ) && all(i -> floatmin(T) <= bigf(γ) * bigf(λ) * bigf(wv[i]) <= bigf(floatmax(T)), 1:n)) || fail("γλ overflows, γλw in range", ctx)
    elseif rq === :thr_under_mid
        (γ * λ < floatmin(T) && all(i -> floatmin(T) <= bigf(γ) * bigf(λ) * bigf(wv[i]) <= bigf(floatmax(T)), 1:n)) || fail("γλ underflows, γλw in range", ctx)
    end
    validate_count!("NormL1")
    y = similar(x); out1 = Vector{T}(undef, 1)
    rest = "$(sc.name)/n$n"
    mkf(PB) = w === nothing ? PB.NormL1(λ) : PB.NormL1(λ, w)
    ref = nothing
    if sc.ref && all(isfinite, x) && isfinite(λ) && (w === nothing || all(isfinite, w))
        ref = (bigf(λ) * sum((w === nothing ? bigf(1) : bigf(w[i])) * abs(bigf(x[i])) for i in 1:n), n)
    end
    for op in sc.ops
        if op == "value"
            addspec!("NormL1", op, T, rest, n, PB -> begin
                f = mkf(PB)
                (op = bindstore(out1, f, x), check = () -> f(x))
            end; ref)
        elseif op == "prox"
            addspec!("NormL1", op, T, rest, n, PB -> begin
                f = mkf(PB)
                (op = bindcall(PB.prox!, y, f, x, γ), check = () -> (PB.prox!(y, f, x, γ); copy(y)))
            end)
        elseif op == "value_and_prox"
            addspec!("NormL1", op, T, rest, n, PB -> begin
                f = mkf(PB)
                (op = bindstore(out1, PB.value_and_prox!, y, f, x, γ),
                 check = () -> (v = PB.value_and_prox!(y, f, x, γ); (v, copy(y))))
            end)
        end
    end
    if sc.iter && n == 1024
        addspec!("NormL1", "value_sparse", T, rest, n, PB -> begin
            f = mkf(PB); M = PB.SparseIterate(x)
            (op = bindstore(out1, f, M), check = () -> f(M))
        end)
        addspec!("NormL1", "value_and_prox_sparse", T, rest, n, PB -> begin
            f = mkf(PB); M = PB.SparseIterate(x); Mo = copy(M)
            (op = bindstore(out1, PB.value_and_prox!, Mo, f, M, γ),
             check = () -> (v = PB.value_and_prox!(Mo, f, M, γ); (v, Vector(Mo))))
        end)
        groups = [collect(1:2:n), collect(2:2:n)]
        addspec!("NormL1", "value_block", T, rest, n, PB -> begin
            f = mkf(PB); B = PB.BlockIterate(copy(x), groups)
            (op = bindstore(out1, f, B), check = () -> f(B))
        end)
        if sc.kind === :underflow
            ps = 32                                     # a symmetric iterate of 32 × 32 with every lower-triangle coordinate stored
            addspec!("NormL1", "value_symmetric", T, rest, ps * ps, PB -> begin
                S = PB.SymmetricSparseIterate(T, ps)
                for c in 1:ps, r in c:ps; S[r, c] = x[1]; end
                f = PB.NormL1(one(T), fill(floatmin(T), ps, ps))
                (op = bindstore(out1, f, S), check = () -> f(S))
            end)
        end
    end
end
l1_scenarios = [
    SO("underflow", kind = :underflow, ops = ("value", "value_and_prox"), iter = true),
    SO("underflow_rand", kind = :underflow_rand, ops = ("value",), iter = true),
    SO("winf_nonzero", kind = :winf_nonzero, ops = ("value",), iter = true),
    SO("winf_zero", kind = :winf_zero, ops = ("value",), iter = true),
    SO("lamInf", kind = :lamInf, ops = ("value", "prox", "value_and_prox")),
    SO("lamInf_zero", kind = :lamInf_zero, ops = ("value", "prox", "value_and_prox")),
    SO("lamInf_nan", kind = :lamInf_nan, ops = ("value", "prox", "value_and_prox")),
    SO("lamInf_w0", kind = :lamInf_w0, ops = ("value", "prox", "value_and_prox")),
    SO("lw_over", kind = :lw_over, ops = ("value",), ref = true),
    SO("lvl_big", kind = :lvl_big, ops = ("value",), ref = true),
    SO("nan_first", kind = :nan_first, ops = ("value", "prox", "value_and_prox")),
    SO("inf_last", kind = :inf_last, ops = ("value", "prox", "value_and_prox")),
    SO("winf_prox", kind = :winf_prox, ops = ("prox", "value_and_prox"), iter = true),
    SO("winf_special", kind = :winf_special, ops = ("value", "prox", "value_and_prox")),
    SO("thr_over", kind = :thr_over, ops = ("prox", "value_and_prox")),
    SO("thr_over_mid", kind = :thr_over_mid, ops = ("prox", "value_and_prox"), iter = true),
    SO("thr_under", kind = :thr_under, ops = ("prox", "value_and_prox")),
    SO("thr_under_mid", kind = :thr_under_mid, ops = ("prox", "value_and_prox")),
    SO("thr_over_u", kind = :thr_over_u, ops = ("prox", "value_and_prox")),
    SO("thr_under_u", kind = :thr_under_u, ops = ("prox", "value_and_prox")),
]
for sc in l1_scenarios, T in (Float32, Float64), n in NL2_DENSE
    add_l1_cases!(T, sc, n)
end

# A level of BigFloat type with Float64 data (the "level out of the range of Float64" cases) is arithmetic in
# BigFloat inside the package, which allocates: these cases are checked but not timed, as the BigFloat cases are.
specs = Spec[occursin(r"/Float64/(lvl_(big|small)|\w+_lvl)/", p.id) ?
             Spec(p.id, p.family, p.op, p.T, false, p.n, p.build, p.ref, p.diff) : p for p in specs]

####################################
#  Roles, checks, timing
####################################

roles = Dict("X" => First, "Y" => Second, "XC" => FirstControl)
filter_re = Regex(get(ENV, "PB_FILTER", ""))
selected = filter(p -> occursin(filter_re, p.id), specs)
# PB_STRIDE=k keeps every k-th selected case (the D-X case stays selected when it passes the filter): a smoke run
stride = parse(Int, get(ENV, "PB_STRIDE", "1"))
stride > 1 && (selected = [p for (i, p) in enumerate(selected) if (i - 1) % stride == 0 || p.diff !== nothing])
let sh = split(get(ENV, "PB_SHARD", "1/1"), "/"); i, n = parse(Int, sh[1]), parse(Int, sh[2])
    global selected = [p for (j, p) in enumerate(selected) if mod1(j, n) == i]
end
isempty(selected) && error("PB_FILTER selects no case")
let seen = Set{String}()
    for p in selected
        p.id in seen && error("duplicate case id: $(p.id)")
        push!(seen, p.id)
    end
end
suites = Dict(r => [p.build(getfield(mod, :ProximalBase)) for p in selected] for (r, mod) in roles)

expect_diff_mode = let s = get(ENV, "PB_EXPECT_DIFF", "auto")
    s == "auto" ? (first_root == second_root ? :same : :forward) :
    s in ("0", "same") ? :same : s in ("1", "forward") ? :forward : s == "reversed" ? :reversed :
    error("PB_EXPECT_DIFF must be auto, 0, 1, forward, same or reversed")
end

# The package calls that allocate after warming, by id: the check asserts 0 allocated bytes for role X
# everywhere else (the closures of the cases are allocation-free; see bindcall).
const ALLOC_OK = Pair{Regex, String}[
    r"^LeastSquaresLoss/value/(Float32|Float64)/(under|sub|zero_residual|exact_fit)$" =>
        "_leastsquares_recover: mapreduce(==, &, r, Y; init = true) allocates 368 bytes where the ordinary sum of squares is zero",
    r"^LeastSquaresLoss/value/(Float32|Float64)/prod_over$" =>
        "_leastsquares_recover (documented): recomputing an overflowing product Xβ copies β and allocates arrays of the size of Y"]
allocated_by(op) = (op(); op(); @allocated op())      # a function barrier: op is a concretely typed closure here
alloc_exempt(id) = any(r -> occursin(r.first, id), ALLOC_OK)
alloc_bad = Pair{String, Int}[]                       # (id, bytes) of the cases that allocate and are not exempt
alloc_seen = Pair{String, Int}[]                      # (id, bytes) of the exempt cases that allocate in X
alloc_y_more = Pair{String, Tuple{Int, Int}}[]        # cases where Y allocates more than X
shown(x) = (s = repr(x); length(s) > 120 ? s[1:120] * "..." : s)
nvalue = 0
for (i, p) in enumerate(selected)
    a, b, c = suites["X"][i], suites["Y"][i], suites["XC"][i]
    a.op(); ra = a.check()
    c.op(); rc = c.check()
    b.op(); rb = b.check()
    exact_equal(ra, rc) || error("A/A output mismatch: $(p.id)")
    if p.diff !== nothing && expect_diff_mode !== :same && !TOLERANT
        px, py = expect_diff_mode === :forward ? (p.diff.x, p.diff.y) : (p.diff.y, p.diff.x)
        (px(ra) && py(rb)) || error("the documented difference (D-X, $expect_diff_mode) did not occur: $(p.id): X = $(shown(ra)), Y = $(shown(rb))")
    else
        if TOLERANT
            coarse_equal(ra, rb) || (global ncoarse += 1; println("SANITY MISMATCH: $(p.id): X = $(shown(ra)), Y = $(shown(rb))"))
        else
            exact_equal(ra, rb) || error("output differs bit for bit: $(p.id): X = $(shown(ra)), Y = $(shown(rb))")
        end
    end
    if p.timed
        ax, ay = allocated_by(a.op), allocated_by(b.op)
        if ax != 0
            alloc_exempt(p.id) ? push!(alloc_seen, p.id => ax) : push!(alloc_bad, p.id => ax)
        end
        ay > ax && push!(alloc_y_more, p.id => (ax, ay))
    end
    if p.ref !== nothing
        global nvalue += 1
        value_agrees(ra, p.ref[1], p.T, p.ref[2]) || error("value of the reference tree differs from the BigFloat reference: $(p.id): $(shown(ra)) vs $(p.T(p.ref[1]))")
    end
end
println("exact check: ", length(selected), " probes, XC = X bit for bit and Y = X bit for bit",
        any(p -> p.diff !== nothing, selected) ? (expect_diff_mode === :same ? "" : " except the D-X case ($expect_diff_mode)") : "",
        "; ", nvalue, " values against a BigFloat reference")
if !isempty(alloc_bad)
    for (id, b) in alloc_bad
        println("  allocates ", b, " bytes in X: ", id)
    end
    error("$(length(alloc_bad)) cases allocate after warming in role X and are not exempt (listed above)")
end
println("allocation check: role X allocates 0 bytes after warming in ", count(p -> p.timed, selected) - length(alloc_seen), " of ",
        count(p -> p.timed, selected), " timed cases; ", length(alloc_seen), " allocate in the package call itself",
        isempty(alloc_y_more) ? "; role Y never allocates more than X" : "; role Y allocates MORE than X in $(length(alloc_y_more)) cases")
for (id, b) in alloc_seen
    println("  allocates ", b, " bytes in X (package call: ", first(r.second for r in ALLOC_OK if occursin(r.first, id)), "): ", id)
end
for (id, (bx, by)) in first(alloc_y_more, 20)
    println("  Y allocates more: ", id, " X = ", bx, " Y = ", by)
end
let cnt = Dict{String, Int}(), cop = Dict{String, Int}()
    for p in selected
        cnt[p.family] = get(cnt, p.family, 0) + 1
        k = "$(p.family)/$(p.op)/$(p.T)"; cop[k] = get(cop, k, 0) + 1
    end
    println("cases per family: ", join(["$k=$v" for (k, v) in sort(collect(cnt))], ", "))
    println("input validations per family: ", join(["$k=$v" for (k, v) in sort(collect(validated))], ", "))
    get(ENV, "PB_INVENTORY", "0") == "1" && foreach(k -> println("  ", k, " ", cop[k]), sort(collect(keys(cop))))
end
get(ENV, "PB_CHECK_ONLY", "0") == "1" && exit()

timed_idx = [i for i in eachindex(selected) if selected[i].timed]
isempty(timed_idx) && error("no timed case selected (BigFloat and the D-X case are check-only)")
# One benchmark object per role and case, built once: building it again would compile its kernel again
# in every round.
function make_bench(op, n)
    evals = max(1, 2^16 ÷ n)
    @benchmarkable $op() evals = evals samples = 100 seconds = 0.15 gctrial = false
end
println("check passed ", Dates.format(now(UTC), "HH:MM:SS")); flush(stdout)
benches = Dict(r => Dict(i => make_bench(suites[r][i].op, selected[i].n) for i in timed_idx) for r in ("X", "Y", "XC"))
outdir = get(ENV, "PB_OUTPUT", "/tmp/pb900-cold")
mkpath(outdir)
open(joinpath(outdir, "provenance.txt"), "w") do io
    println(io, "UTC: ", now(UTC), "\nJulia: ", VERSION, "\nCPU: ", Sys.CPU_NAME, "\nThreads: ", Threads.nthreads())
    println(io, "Bounds option: ", Base.JLOptions().check_bounds)
    println(io, "X: ", first_root, "\nY: ", second_root)
end
for r in ("X", "Y", "XC"), i in timed_idx          # compile and warm every specialization
    for _ in 1:20
        suites[r][i].op()
    end
    run(benches[r][i])
end
println("warmed ", length(timed_idx), " timed cases in 3 roles ", Dates.format(now(UTC), "HH:MM:SS")); flush(stdout)
rounds = parse(Int, get(ENV, "PB_ROUNDS", "11"))
open(joinpath(outdir, "raw.csv"), "w") do io
    println(io, "case,round,order,role,median_ns,min_ns,memory,allocs")
    for round in 1:rounds
        for i in (isodd(round) ? timed_idx : reverse(timed_idx))
            order = isodd(round) ? ("X", "Y", "XC") : ("XC", "Y", "X")
            for (k, r) in enumerate(order)
                t = run(benches[r][i])
                m = median(t)
                println(io, selected[i].id, ',', round, ',', k, ',', r, ',', m.time, ',', minimum(t).time, ',', m.memory, ',', m.allocs)
            end
        end
        flush(io)
        println("round ", round, "/", rounds, " ", Dates.format(now(UTC), "HH:MM:SS")); flush(stdout)
    end
end
