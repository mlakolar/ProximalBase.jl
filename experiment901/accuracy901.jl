# Common accuracy and semantics harness of the #901 architecture experiment (orchestrator-owned).
#
#   PB_TREES="base=/path/base;A=/path/A" PB_OUTPUT=<dir> [PB_TRIALS=6] [PB_SEED=901] [PB_FILTER=<regex>]
#     julia --threads=1 --startup-file=no --project=<env with PrecompileTools> accuracy901.jl
#
# Every tree is compiled as its own module. Inputs are generated once per case and shared. References are
# computed in BigFloat at 512 bits from the mathematical formulas (no package code), from the exact values of
# the Float32/Float64 inputs. Criteria (METHODOLOGY.md, N2/N3):
#   value  (|e| <= floatmax/4):  |a - e| <= 2(n+4) eps |e| + 4 K nextfloat(0)        n entries, K groups
#   value  (|e| >= 1.01 floatmax): a == +Inf
#   prox   y_i: |y_i - y*_i| <= (m+8) eps |x_i| + 4 nextfloat(0); a group with ‖x‖ <= (1 - 4(m+4)eps) γλw is 0
#   fused  value at the stored output (reference from the stored y), prox as above
#   block  prox_block! of each group, as prox
#   kkt    screen on a BlockIterate at zero: |a - e| <= (m+8) eps (‖g‖ + λw) + 4 nextfloat(0)
#   alias  prox!(x, f, x, γ) equals prox!(y, f, x, γ) (===, per entry)
# Exceptional regimes check the documented classes exactly (=== per entry where an entry is kept or fixed).
# Output: <dir>/accuracy.csv (one row per tree, family, op, T, m, regime: cases, max error in bound units,
# max ulps, violations, class failures) and a printed summary; exit code 1 if any tree fails a criterion.
using Random, Printf, LinearAlgebra, SparseArrays
const TREES = [let (l, p) = split(s, "="; limit = 2); (String(strip(l)), abspath(strip(p))) end
               for s in split(ENV["PB_TREES"], ";") if !isempty(strip(s))]
const OUT = get(ENV, "PB_OUTPUT", "accuracy-out"); mkpath(OUT)
const TRIALS = parse(Int, get(ENV, "PB_TRIALS", "6"))
const SEED = parse(Int, get(ENV, "PB_SEED", "901"))
const FILTER = Regex(get(ENV, "PB_FILTER", "."))
const PBS = Module[]
for (i, (label, root)) in enumerate(TREES)
    m = Module(Symbol("Acc", i))
    Base.include(m, joinpath(root, "src", "ProximalBase.jl"))
    push!(PBS, Base.invokelatest(getfield, m, :ProximalBase))
end
setprecision(BigFloat, 512)

####################################
#  Statistics
####################################

mutable struct Stat
    cases::Int; maxu::Float64; maxulp::Float64; viol::Int; classfail::Int; negzero::Int; first::String
end
Stat() = Stat(0, 0.0, 0.0, 0, 0, 0, "")
const STATS = Dict{Tuple{String, String, String, String, String, String}, Stat}()   # tree, fam, op, T, m, regime
stat(tree, fam, op, T, m, regime) = get!(Stat, STATS, (tree, fam, op, string(T), string(m), String(regime)))
function note!(s::Stat, u, ulp, desc)
    s.maxu = max(s.maxu, u); s.maxulp = max(s.maxulp, ulp)
    if u > 1
        s.viol += 1; isempty(s.first) && (s.first = desc)
    end
end
function classfail!(s::Stat, desc)
    s.classfail += 1; isempty(s.first) && (s.first = desc)
end

bf(x) = BigFloat(x)
ulps(a, e::BigFloat, ::Type{T}) where {T} = (te = T(e); iszero(te) ? Float64(abs(bf(a) - e) / bf(nextfloat(zero(T)))) :
                                          Float64(abs(bf(a) - e) / bf(eps(abs(te)))))
tiny(::Type{T}) where {T} = bf(nextfloat(zero(T)))

####################################
#  References
####################################

gnorm(x, G) = sqrt(sum(bf(x[i])^2 for i in G; init = bf(0)))
function ref_prox(x, groups, λ0, w, γ)
    y = Vector{BigFloat}(undef, length(x)); zero_req = falses(length(groups)); ratio = zeros(length(groups))
    for i in eachindex(x); y[i] = bf(x[i]); end
    for (k, G) in enumerate(groups)
        t = bf(γ) * bf(λ0) * bf(w[k]); nr = gnorm(x, G)
        ratio[k] = iszero(nr) ? Inf : Float64(t / nr)
        if nr <= t
            for i in G; y[i] = bf(0); end
        else
            s = 1 - t / nr
            for i in G; y[i] = s * bf(x[i]); end
        end
        m = length(G)
        zero_req[k] = nr <= (1 - 4 * (m + 4) * bf(eps(eltype(x)))) * t
    end
    y, zero_req, ratio
end
ref_value(x, groups, λ0, w) = bf(λ0) * sum(bf(w[k]) * gnorm(x, G) for (k, G) in enumerate(groups); init = bf(0))

function check_value!(s, a, e, ::Type{T}, n, K, desc) where {T}
    if abs(e) <= bf(floatmax(T)) / 4
        b = 2 * (n + 4) * bf(eps(T)) * abs(e) + 4K * tiny(T)
        isfinite(a) || return classfail!(s, desc * " value not finite: $a, exact $(Float64(e))")
        note!(s, Float64(abs(bf(a) - e) / b), ulps(a, e, T), desc * " value $a exact $(Float64(e))")
    elseif abs(e) >= bf(floatmax(T)) * bf(1.01)
        a == T(Inf) || classfail!(s, desc * " value $a, exact $(Float64(e)) > floatmax")
    end
    s.cases += 1
end
function check_prox!(s, y, ystar, zero_req, ratio, x, groups, ::Type{T}, desc) where {T}
    for (k, G) in enumerate(groups)
        m = length(G)
        for i in G
            if zero_req[k]
                iszero(y[i]) || (classfail!(s, desc * " group $k must be 0, y[$i] = $(y[i])"); break)
                signbit(y[i]) && (s.negzero += 1)
            else
                b = (m + 8) * bf(eps(T)) * abs(bf(x[i])) + 4 * tiny(T)
                isfinite(y[i]) || (classfail!(s, desc * " y[$i] = $(y[i]) not finite"); break)
                u = Float64(abs(bf(y[i]) - ystar[i]) / b)
                ulp = ratio[k] < 0.9 ? ulps(y[i], ystar[i], T) : 0.0
                note!(s, u, ulp, desc * " group $k y[$i] = $(y[i]) exact $(Float64(ystar[i]))")
            end
        end
    end
    s.cases += 1
end

####################################
#  Inputs
####################################

p2(::Type{T}, e) where {T} = ldexp(one(T), e)
emaxT(::Type{T}) where {T} = exponent(floatmax(T)); eminT(::Type{T}) where {T} = exponent(floatmin(T))
rnd(::Type{T}, v::BigFloat) where {T} = T(v)

const FINITE = (:ord, :near, :wide, :over, :under, :subn, :thr_over, :thr_under, :bigval, :tinysum, :zgrp)
const EXCEPT = (:w0, :wmix, :nan, :inf, :lamInf, :gam0, :gamInf)

# (x, λ0, w, γ) for a regime; weights chosen against the exact group norms where the regime needs it
function make_case(::Type{T}, regime, m, K, rng) where {T}
    N = m * K; groups = [((k-1)*m+1):(k*m) for k in 1:K]
    emax, emin = emaxT(T), eminT(T)
    x = randn(rng, T, N); γ = T(0.8); λ0 = T(1.25)
    w = T[sqrt(T(m)) * T(0.5 + rand(rng)) for _ in 1:K]
    ρ() = 0.3 + 0.9 * rand(rng)                         # threshold / norm, about a third of groups zeroed
    setw!(γ, λ0) = for (k, G) in enumerate(groups); w[k] = rnd(T, ρ() * gnorm(x, G) / (bf(γ) * bf(λ0))); end
    if regime === :ord
        setw!(γ, λ0)
    elseif regime === :near                              # threshold within 1e-3 of the norm (cancellation)
        for (k, G) in enumerate(groups); w[k] = rnd(T, (1 + (2rand(rng) - 1) * 1e-3) * gnorm(x, G) / (bf(γ) * bf(λ0))); end
    elseif regime === :wide
        for i in eachindex(x); x[i] = (rand(rng, Bool) ? 1 : -1) * ldexp(T(0.5 + rand(rng)), rand(rng, -(T === Float32 ? 20 : 60):(T === Float32 ? 20 : 60))); end
        setw!(γ, λ0)
    elseif regime === :over
        x .*= p2(T, emax ÷ 2 + 2); λ0 = p2(T, emax ÷ 2 - 2); setw!(γ, λ0)
    elseif regime === :under
        x .*= p2(T, emin ÷ 2 - 4); λ0 = p2(T, emin ÷ 2 - 4); setw!(γ, λ0)
    elseif regime === :subn                              # subnormal entries
        for i in eachindex(x); x[i] = T(x[i] * p2(Float64, (T === Float32 ? -140 : -1060))); end
        λ0 = p2(T, emin ÷ 2); setw!(γ, λ0)
    elseif regime === :thr_over                          # γλ0w beyond floatmax, representable result
        for i in eachindex(x); x[i] = (rand(rng, Bool) ? 1 : -1) * floatmax(T) * T(0.25 + 0.25rand(rng)); end
        γ = p2(T, emax ÷ 2); λ0 = p2(T, emax ÷ 2); setw!(γ, λ0)
    elseif regime === :thr_under                         # γλ0 underflows, γλ0w normal
        x .*= p2(T, emin + 30); γ = p2(T, emin ÷ 2 - 8); λ0 = p2(T, emin ÷ 2 - 8); setw!(γ, λ0)
    elseif regime === :bigval                            # value about 0.6 floatmax
        s = sum(bf(w[k]) * gnorm(x, G) for (k, G) in enumerate(groups)); λ0 = rnd(T, bf(0.6) * bf(floatmax(T)) / s)
    elseif regime === :tinysum                           # value below K floatmin
        s = sum(bf(w[k]) * gnorm(x, G) for (k, G) in enumerate(groups)); λ0 = rnd(T, bf(0.3) * K * bf(floatmin(T)) / s)
    elseif regime === :zgrp
        setw!(γ, λ0); for k in 1:3:K; x[groups[k]] .= zero(T); end
    elseif regime === :w0
        w .= zero(T)
    elseif regime === :wmix
        for k in 1:K; w[k] = T[0, Inf, 1, 0.5][mod1(k, 4)]; end
        x[groups[1]] .= T(NaN); K >= 2 && (x[first(groups[2])] = T(Inf))     # unpenalized NaN, fixed Inf
    elseif regime === :nan
        setw!(γ, λ0); for k in 1:4:K; x[first(groups[k])] = T(NaN); end
    elseif regime === :inf
        setw!(γ, λ0); for k in 1:4:K; x[first(groups[k])] = T(-Inf); end
    elseif regime === :lamInf
        λ0 = T(Inf); w[1] = zero(T)
    elseif regime === :gam0
        γ = zero(T)
    elseif regime === :gamInf
        γ = T(Inf)
    end
    (x = x, groups = groups, λ0 = λ0, w = w, γ = γ)
end

####################################
#  Exceptional classes (docs/src/numerics.md)
####################################

# expected y per entry: :keep (=== x), :pzero (=== +0), :nan, or :shrink (finite rules)
function expected_class(d, k)
    λ0, w, γ = d.λ0, d.w[k], d.γ; G = d.groups[k]; x = d.x
    iszero(w) && return :keep
    (isinf(w) || isinf(λ0)) && return :pzero
    (iszero(λ0) || iszero(γ)) && return :keep
    isinf(γ) && return :pzero
    any(i -> isnan(x[i]), G) && return :nan
    any(i -> isinf(x[i]), G) && return :keep
    all(i -> iszero(x[i]), G) && return :pzero
    :shrink
end
function check_classes!(s, y, d, desc)
    for (k, G) in enumerate(d.groups)
        c = expected_class(d, k)
        for i in G
            ok = c === :keep ? y[i] === d.x[i] : c === :pzero ? y[i] === zero(y[i]) : c === :nan ? isnan(y[i]) : true
            ok || (classfail!(s, desc * " group $k ($c): y[$i] = $(y[i]), x = $(d.x[i])"); break)
        end
    end
    s.cases += 1
end
function value_class_ok(a, d)
    T = typeof(a)
    if d.regime === :w0; return a === zero(T)
    elseif d.regime === :nan; return isnan(a)
    elseif d.regime === :inf; return a == T(Inf)
    elseif d.regime === :lamInf; return a == T(Inf)
    elseif d.regime === :wmix; return a == T(Inf)          # an infinite weight at a nonzero group
    end
    true
end

####################################
#  Run
####################################

function run_group_case(T, m, K, regime, trial, fam)
    rng = MersenneTwister(hash((SEED, T, m, K, regime, trial, fam)))
    d0 = make_case(T, regime, m, K, rng)
    d = merge(d0, (regime = regime,))
    N = length(d.x)
    if fam == "Norm"
        d = merge(d, (groups = [1:N], w = T[1], λ0 = regime in (:lamInf,) ? T(Inf) : d.λ0 * d.w[1]))
        regime === :w0 && return                    # no weight for NormL2
        regime === :wmix && return
    end
    finite = regime in FINITE
    ystar, zero_req, ratio = finite ? ref_prox(d.x, d.groups, d.λ0, d.w, d.γ) : (nothing, nothing, nothing)
    e = finite ? ref_value(d.x, d.groups, d.λ0, d.w) : nothing
    desc0 = "$fam/$T/m$m/$regime/trial$trial"
    for (ti, (label, _)) in enumerate(TREES)
        PB = PBS[ti]
        f = fam == "Norm" ? PB.NormL2(d.λ0) : PB.GroupNormL2(d.λ0, d.groups, d.w)
        Kg = length(d.groups)
        # value
        s = stat(label, fam, "value", T, m, regime)
        a = f(d.x)
        if finite
            check_value!(s, a, e, T, N, Kg, desc0)
        else
            value_class_ok(a, d) || classfail!(s, desc0 * " value $a")
            s.cases += 1
        end
        # prox!
        y = fill(T(NaN), N)
        PB.prox!(y, f, d.x, d.γ)
        s = stat(label, fam, "prox", T, m, regime)
        finite ? check_prox!(s, y, ystar, zero_req, ratio, d.x, d.groups, T, desc0) : check_classes!(s, y, d, desc0)
        # in place
        xc = copy(d.x); PB.prox!(xc, f, xc, d.γ)
        s = stat(label, fam, "alias", T, m, regime)
        all(i -> xc[i] === y[i], eachindex(y)) || classfail!(s, desc0 * " prox!(x,f,x,γ) differs from prox!(y,f,x,γ)")
        s.cases += 1
        # fused
        yf = fill(T(NaN), N)
        v = PB.value_and_prox!(yf, f, d.x, d.γ)
        s = stat(label, fam, "fused", T, m, regime)
        if finite
            check_prox!(s, yf, ystar, zero_req, ratio, d.x, d.groups, T, desc0 * " fused")
            ev = ref_value(yf, d.groups, d.λ0, d.w)
            check_value!(s, v, ev, T, N, Kg, desc0 * " fused-value")
        else
            check_classes!(s, yf, d, desc0 * " fused")
        end
        # block and kkt (GroupNormL2)
        if fam == "Group"
            s = stat(label, fam, "block", T, m, regime)
            yb = similar(d.x)
            for (k, G) in enumerate(d.groups)
                out = similar(d.x, length(G)); PB.prox_block!(out, f, d.x[G], k, d.γ); yb[G] = out
            end
            finite ? check_prox!(s, yb, ystar, zero_req, ratio, d.x, d.groups, T, desc0 * " block") : check_classes!(s, yb, d, desc0 * " block")
            if finite
                s = stat(label, fam, "kkt", T, m, regime)
                xb = PB.BlockIterate(zeros(T, N), d.groups)
                for (k, G) in enumerate(d.groups)
                    a = PB.kkt_violation(f, d.x, xb, k)
                    c = bf(d.λ0) * bf(d.w[k]); nr = gnorm(d.x, G); ek = nr - c
                    if abs(ek) <= bf(floatmax(T)) / 4 && isfinite(c)
                        b = (length(G) + 8) * bf(eps(T)) * (nr + c) + 4 * tiny(T)
                        isfinite(a) || (classfail!(s, desc0 * " kkt $k = $a, exact $(Float64(ek))"); continue)
                        note!(s, Float64(abs(bf(a) - ek) / b), ulps(a, ek, T), desc0 * " kkt $k = $a exact $(Float64(ek))")
                    end
                end
                s.cases += 1
            end
        end
    end
end

t0 = time()
for fam in ("Group", "Norm"), T in (Float64, Float32)
    shapes = fam == "Group" ? [(m, clamp(2048 ÷ m, 2, 64)) for m in (1, 2, 5, 32, 1024)] :
                              [(n, 1) for n in (1, 2, 5, 32, 1024, 65536)]
    for (m, K) in shapes, regime in (FINITE..., EXCEPT...)
        occursin(FILTER, "$fam/$T/m$m/$regime") || continue
        ntr = fam == "Norm" && m == 65536 ? max(1, TRIALS ÷ 3) : TRIALS
        for trial in 1:ntr
            run_group_case(T, fam == "Norm" ? m : m, K, regime, trial, fam)
        end
    end
end

####################################
#  Report
####################################

open(joinpath(OUT, "accuracy.csv"), "w") do io
    println(io, "tree,family,op,T,m,regime,cases,max_bound_units,max_ulps,violations,class_failures,neg_zeros,first_failure")
    for (k, s) in sort(collect(STATS); by = first)
        println(io, join(k, ","), ",", s.cases, ",", @sprintf("%.4f", s.maxu), ",", @sprintf("%.2f", s.maxulp), ",",
                s.viol, ",", s.classfail, ",", s.negzero, ",\"", replace(s.first, "\"" => "'"), "\"")
    end
end
open(joinpath(OUT, "provenance.txt"), "w") do io
    println(io, "julia ", VERSION, " host ", gethostname(), " trials ", TRIALS, " seed ", SEED, " filter ", FILTER.pattern)
    for (label, root) in TREES
        shafile = joinpath(root, "SHA")
        sha = isfile(shafile) ? readchomp(shafile) : try readchomp(`git -C $root rev-parse HEAD`) catch; "?" end
        println(io, "tree ", label, " ", root, " ", sha)
    end
end

base = TREES[1][1]
fail = false
println("accuracy901: ", length(STATS), " cells, ", round(time() - t0; digits = 1), " s")
for (label, _) in TREES
    cells = [(k, s) for (k, s) in STATS if k[1] == label]
    viol = sum(s.viol for (_, s) in cells); cf = sum(s.classfail for (_, s) in cells); nz = sum(s.negzero for (_, s) in cells)
    worst = maximum(s.maxu for (_, s) in cells)
    # material regression vs the first tree: max units > max(2 × base's, base's + 0.25)
    regress = String[]
    if label != base
        for (k, s) in cells
            kb = (base, k[2:end]...)
            haskey(STATS, kb) || continue
            sb = STATS[kb]
            s.maxu > max(2 * sb.maxu, sb.maxu + 0.25) && push!(regress, "$(join(k[2:end], "/")) $(round(s.maxu; digits = 3)) vs $(round(sb.maxu; digits = 3))")
        end
    end
    @printf("%-8s violations %d, class failures %d, max bound units %.3f, negative zeros %d, regressions vs %s: %d\n",
            label, viol, cf, worst, nz, base, length(regress))
    for (k, s) in cells
        (s.viol > 0 || s.classfail > 0) && println("    FAIL ", join(k[2:end], "/"), ": ", s.first)
    end
    for r in regress[1:min(end, 20)]; println("    REGRESSION ", r); end
    global fail |= viol > 0 || cf > 0 || !isempty(regress)
end
# per (family, op, T) max bound units, side by side
println("\nmax error in bound units per family/op/T (all regimes and sizes):")
keys3 = sort(unique([(k[2], k[3], k[4]) for k in keys(STATS)]))
@printf("  %-22s", "family/op/T"); for (l, _) in TREES; @printf(" %10s", l); end; println()
for k3 in keys3
    @printf("  %-22s", join(k3, "/"))
    for (l, _) in TREES
        v = maximum((s.maxu for (k, s) in STATS if k[1] == l && (k[2], k[3], k[4]) == k3); init = 0.0)
        @printf(" %10.4f", v)
    end
    println()
end
exit(fail ? 1 : 0)
