module InfiniteWeightsTest
# Characterization of the exceptional parameters of NormL1 (issue #853, resolved by narrowing the contract).
#
# Guaranteed:
#   * an infinite weight fixes its coordinate at zero for every FINITE trial, at every nonnegative
#     level and step, including a zero level and a zero step;
#   * an infinite level at a nonnegative step fixes every positively weighted coordinate at zero, also
#     at NaN and ±Inf trials (decision 49). At a zero step this is the limit γ → 0+ with the function
#     fixed, the projection onto the closure of its domain (#665, Option A), not shrink(x, 0⋅Inf) = x;
#   * a zero weight leaves the coordinate unpenalized, NaN and ±Inf included.
# Observed, outside any guarantee (not a requirement): an infinite weight with a NaN or ±Inf trial and
# no infinite level at a positive step returns NaN from IEEE arithmetic. The broader guarantee was
# declined because it imposed measurable overhead on ordinary finite proximal calls.
# Known defect, pinned and not endorsed (#866): finite factors whose threshold overflows turn ±Inf into NaN.
# Homogeneous Float32 and Float64 are the supported configurations; no mixed-precision guarantee.
using Test, ProximalBase, LinearAlgebra, SparseArrays, Random
using ..TestUtils
const PB = ProximalBase

# The specification, written independently of the kernel. `:fixed` is the positive floating zero,
# `:identity` leaves the trial as it is (NaN stays NaN), `:observed_nan` is the unguaranteed IEEE
# outcome above, `:overflow` is a finite level, step and weight whose exact product exceeds floatmax(T)
# (#866), and `:threshold` is an ordinary soft threshold.
function classify(::Type{T}, v, γ, λ, w) where {T}
    if isinf(w)
        isfinite(v) && return :fixed                # guaranteed, whatever the level and the step
        isinf(λ) && γ >= 0 && return :fixed         # decision 49; at γ = 0, #665
        return :observed_nan                        # outside the guarantee
    end
    iszero(w) && return :identity                   # unpenalized
    isinf(λ) && γ >= 0 && return :fixed             # decision 49; at γ = 0 the projection onto {0} (#665)
    (iszero(λ) || iszero(γ)) && return :identity    # a zero threshold of finite factors
    big(γ) * big(λ) * big(w) > floatmax(T) ? :overflow : :threshold
end
function reference(::Type{T}, v, γ, λ, w) where {T}
    t = big(γ) * big(λ) * big(w)
    isfinite(v) ? convert(T, sign(big(v)) * max(abs(big(v)) - t, 0)) : v
end
is_fixed_zero(r) = isequal(r, zero(r))              # the positive floating zero

# Assert the specification for one entry; `got` is what an entry point produced for the trial v.
function check_entry(::Type{T}, got, v, γ, λ, w) where {T}
    kind = classify(T, v, γ, λ, w)
    if kind === :fixed
        @test is_fixed_zero(got)
    elseif kind === :identity
        @test isequal(got, v)
    elseif kind === :observed_nan
        @test isnan(got)                            # observed behavior, not a guarantee
    elseif kind === :threshold
        if isfinite(v)
            @test isapprox(got, reference(T, v, γ, λ, w); rtol = 8eps(T), atol = 8eps(T))
        else
            @test isequal(got, v)                   # a finite threshold keeps ±Inf, and NaN stays NaN
        end
    else                                            # :overflow: a finite trial is thresholded to zero;
        isnan(v) && @test isnan(got)                # an infinite one is #866 and is never forced to zero
        isfinite(v) && @test is_fixed_zero(got)
        isinf(v) && @test !is_fixed_zero(got)
    end
    nothing
end

grids(::Type{T}) where {T} = (
    trials = T[2, -2, Inf, -Inf, NaN],
    levels = T[0, 1, Inf, floatmax(T) / 4],
    weights = T[0, 1, Inf, floatmax(T) / 4],
    steps = T[0, 1, 0.5, 2])

# all (trial, weight) pairs as parallel vectors
function pairs_tw(g)
    T = eltype(g.trials)
    xs = T[]; ws = T[]
    for w in g.weights, x in g.trials
        push!(xs, x); push!(ws, w)
    end
    xs, ws
end

for T in (Float32, Float64)
    g = grids(T)
    @testset "$T scalar step: every entry point" begin
        xs, ws = pairs_tw(g)
        for λ in g.levels, γ in g.steps
            f = NormL1(λ, ws)
            blocks = [PB.prox_block(f, xs[i], i, γ) for i in eachindex(xs)]
            y = prox(f, xs, γ)
            out = similar(xs); @test prox!(out, f, xs, γ) === out
            alias = copy(xs); prox!(alias, f, alias, γ)
            vp = similar(xs); val = value_and_prox!(vp, f, xs, γ)
            for i in eachindex(xs), r in (blocks[i], y[i], out[i], alias[i], vp[i])
                check_entry(T, r, xs[i], γ, λ, ws[i])
            end
            @test isequal(y, blocks) && isequal(out, blocks) && isequal(alias, blocks) && isequal(vp, blocks)
            @test isequal(val, f(vp))
        end
    end
    @testset "$T per-entry array steps" begin
        triples = [(x, w, γ) for γ in g.steps for w in g.weights for x in g.trials]
        xs = T[t[1] for t in triples]; ws = T[t[2] for t in triples]; γs = T[t[3] for t in triples]
        for λ in g.levels
            f = NormL1(λ, ws)
            y = prox(f, xs, γs)
            out = similar(xs); prox!(out, f, xs, γs)
            alias = copy(xs); prox!(alias, f, alias, γs)
            vp = similar(xs); val = value_and_prox!(vp, f, xs, γs)
            for i in eachindex(xs), r in (y[i], out[i], alias[i], vp[i])
                check_entry(T, r, xs[i], γs[i], λ, ws[i])
            end
            @test isequal(y, out) && isequal(y, alias) && isequal(y, vp) && isequal(val, f(vp))
            # a matrix argument takes the same per-entry steps in linear order
            M = reshape(copy(xs), 10, :); γM = reshape(copy(γs), 10, :)
            @test isequal(vec(prox(NormL1(λ, reshape(copy(ws), 10, :)), M, γM)), y)
        end
    end
    @testset "$T the documented cases" begin
        # 1. infinite weight, finite trial: zero at every level and step, zero level and zero step included
        for v in T[2, -2, 0, floatmax(T)], λ in T[0, 1, Inf, floatmax(T) / 4], γ in T[0, 1, 0.5, 2]
            @test is_fixed_zero(prox(NormL1(λ, T[Inf]), [v], γ)[1])
            @test is_fixed_zero(PB.prox_block(NormL1(λ, T[Inf]), v, 1, γ))
        end
        # 2. infinite level, nonnegative step, positive weight: zero also at non-finite trials (decision 49);
        #    a zero step fixes the entry too, the fixed-function limit γ → 0+ (#665)
        for v in T[2, Inf, -Inf, NaN], w in T[1, floatmax(T) / 4, Inf], γ in T[1, 0]
            @test is_fixed_zero(prox(NormL1(T(Inf), T[w]), [v], γ)[1])
        end
        # 3. OBSERVED, outside the fixed-zero guarantee: an infinite weight at a finite level meets a
        #    NaN or ±Inf trial in IEEE arithmetic (Inf - Inf) and returns NaN. Not a requirement.
        for v in T[Inf, -Inf, NaN], λ in T[0, 1, floatmax(T) / 4], γ in T[0, 1]
            @test isnan(prox(NormL1(λ, T[Inf]), [v], γ)[1])
        end
        # 4. zero weight: unpenalized at every level, non-finite inputs included
        for λ in T[0, 1, Inf], γ in T[0, 1]
            f = NormL1(λ, T[0])
            @test prox(f, T[Inf], γ)[1] == T(Inf) && prox(f, T[-Inf], γ)[1] == T(-Inf)
            @test isnan(prox(f, T[NaN], γ)[1]) && prox(f, T[2], γ)[1] == T(2)
        end
        # the value oracle is unchanged: Inf away from zero, 0 at zero, NaN stays NaN
        f = NormL1(T(1), T[Inf])
        @test f(T[2]) == T(Inf) && f(T[0]) == 0 && isnan(f(T[NaN]))
        @test value_and_prox!(zeros(T, 1), f, T[2], T(1)) == 0
        # an infinite weight does not leak into its neighbours
        f = NormL1(T(1), T[Inf, 1, 0])
        @test isequal(prox(f, T[7, 3, -Inf], T(1)), T[0, 2, -Inf])
        # unweighted: ordinary thresholds, unchanged
        @test isequal(prox(NormL1(T(0.5)), T[2, -2, Inf, -Inf, NaN, 0], one(T)), T[1.5, -1.5, Inf, -Inf, NaN, 0])
    end
    @testset "$T finite-factor overflow (#866): pinned, not endorsed" begin
        # Known defects. A rounded Inf threshold from FINITE factors gives NaN for an Inf trial. These
        # are not the desired mathematical results; flip them when #866 is fixed.
        big = T === Float32 ? (T(1e30), T(1e10), T(1e30)) : (T(1e300), T(1e10), T(1e300))
        # fast path: γλ is normal and a finite w makes the final threshold overflow
        f = NormL1(big[1], T[big[2]])
        @test isnan(prox(f, T[Inf], T(1))[1])                                  # #866
        @test isnan(prox(f, T[-Inf], T(1))[1])                                 # #866
        @test isnan(prox(f, T[NaN], T(1))[1])
        @test is_fixed_zero(prox(f, T[2], T(1))[1])                            # a finite trial is thresholded to zero
        # slow path: γλ itself overflows before the finite weight
        f = NormL1(big[2], T[1])
        @test isnan(prox(f, T[Inf], big[3])[1])                                # #866
        @test isnan(prox(f, T[-Inf], big[3])[1])                               # #866
        @test is_fixed_zero(prox(f, T[2], big[3])[1])
        @test isnan(PB.prox_block(NormL1(big[1], T[big[2]]), T(Inf), 1, T(1))) # #866
        @test isnan(prox(NormL1(big[1], T[big[2]]), T[Inf], T[1])[1])          # #866
        # a finite factor that does not overflow is no infinite weight: an Inf trial is kept
        @test prox(NormL1(T(1), T[floatmax(T) / 4]), T[Inf], T(1))[1] == T(Inf)
    end
end

# The same contract on the supported iterates, with stored NaN and ±Inf values. Activation is unchanged:
# a destination keeps its stored entries (possibly as zeros), a fresh one stores no zeros.
for T in (Float32, Float64)
    g = grids(T)
    xs, ws = pairs_tw(g)
    n = length(xs)
    @testset "$T SparseIterate" begin
        stored = 1:3:n
        for λ in g.levels, γ in g.steps
            f = NormL1(λ, ws)
            x = SparseIterate(T, n); for i in stored; x[i] = xs[i]; end
            y = prox(f, x, γ)
            alias = copy(x); prox!(alias, f, alias, γ)
            vp = copy(x); val = value_and_prox!(vp, f, x, γ)
            @test issubset(collect(activecoordinates(y)), collect(activecoordinates(x)))
            for r in (alias, vp)
                @test sort(collect(activecoordinates(r))) == sort(collect(activecoordinates(x)))
            end
            for r in (y, alias, vp), i in 1:n
                i in stored ? check_entry(T, r[i], xs[i], γ, λ, ws[i]) : @test iszero(r[i])
            end
            @test isequal(Vector(y), Vector(alias)) && isequal(Vector(y), Vector(vp)) && isequal(val, f(vp))
        end
        γs = T[g.steps[mod1(i, 4)] for i in 1:n]
        f = NormL1(T(1), ws); x = SparseIterate(T, n)
        for i in 1:2:n; x[i] = xs[i]; end
        y = prox(f, x, γs); alias = copy(x); prox!(alias, f, alias, γs)
        for i in 1:2:n, r in (y, alias)
            check_entry(T, r[i], xs[i], γs[i], T(1), ws[i])
        end
    end
    @testset "$T BlockIterate" begin
        groups = [collect(1:2:n), collect(2:2:n)]                   # an interleaved layout
        for λ in g.levels, γ in g.steps
            f = NormL1(λ, ws)
            x = BlockIterate(copy(xs), groups)                      # both blocks active, NaN and ±Inf stored
            y = prox(f, x, γ)
            alias = copy(x); prox!(alias, f, alias, γ)
            vp = copy(x); val = value_and_prox!(vp, f, x, γ)
            for r in (y, alias, vp)
                @test collect(activecoordinates(r)) == collect(activecoordinates(x))
                for i in 1:n
                    check_entry(T, r[i], xs[i], γ, λ, ws[i])
                end
            end
            @test isequal(Vector(y), Vector(alias)) && isequal(Vector(y), Vector(vp)) && isequal(val, f(vp))
        end
        # an inactive block holds no stored trial: the output stays inactive
        f = NormL1(T(1), ws)
        x = BlockIterate(T, n, groups); copyto!(blockview(x, 1), xs[groups[1]])
        y = prox(f, x, T(1))
        @test collect(activecoordinates(y)) == collect(activecoordinates(x))
        @test all(iszero, y[groups[2]])
        γs = T[g.steps[mod1(i, 4)] for i in 1:n]
        ya = prox(NormL1(T(1), ws), BlockIterate(copy(xs), groups), γs)
        for i in 1:n
            check_entry(T, ya[i], xs[i], γs[i], T(1), ws[i])
        end
    end
    @testset "$T SymmetricSparseIterate" begin
        p = 5
        # a symmetric weight matrix over the weight patterns; stored lower-triangle trials
        pattern = T[0, 1, Inf, floatmax(T) / 4, 1]
        W = [max(pattern[r], pattern[c]) == Inf ? T(Inf) :
             (pattern[r] == 0 || pattern[c] == 0) ? T(0) : max(pattern[r], pattern[c]) for r in 1:p, c in 1:p]
        trialvals = T[2, -2, Inf, -Inf, NaN]
        for λ in g.levels, γ in T[0, 1, 0.5]
            f = NormL1(λ, W)
            S = SymmetricSparseIterate(T, p)
            for c in 1:p, r in c:p
                S[r, c] = trialvals[mod1(r + 2c, 5)]
            end
            y = prox(f, S, γ)
            alias = copy(S); prox!(alias, f, alias, γ)
            vp = copy(S); val = value_and_prox!(vp, f, S, γ)
            @test issubset(collect(activecoordinates(y)), collect(activecoordinates(S)))
            for r in (alias, vp)
                @test collect(activecoordinates(r)) == collect(activecoordinates(S))
            end
            for r in (y, alias, vp), c in 1:p, i in c:p
                v = trialvals[mod1(i + 2c, 5)]
                check_entry(T, r[i, c], v, γ, λ, W[i, c])
                check_entry(T, r[c, i], v, γ, λ, W[c, i])
            end
            @test isequal(Matrix(y), Matrix(alias)) && isequal(Matrix(y), Matrix(vp)) && isequal(val, f(vp))
            # per-entry steps are supported with a dense output of the matrix of the iterate
            outM = zeros(T, p, p)
            prox!(outM, f, Matrix(S), fill(γ, p, p))
            @test isequal(outM, Matrix(y))
        end
    end
end

# A wrapper keeps the guaranteed cases: Scaled folds its factor into the step.
@testset "Scaled NormL1 with an infinite weight" begin
    for T in (Float32, Float64), v in T[2, -3, 0]
        r = prox(Scaled(NormL1(T(1), T[Inf, 1]), T(2)), T[v, 3], T(1))
        @test is_fixed_zero(r[1]) && r[2] == T(1)
    end
end

# #665, Option A: at γ = 0 an infinite LEVEL gives lim_{γ→0+} prox_{γf}(x) with f fixed, the projection
# onto the closure of dom f. Positive-weight coordinates become +0 whatever they hold (NaN, ±Inf, -0
# included); zero-weight coordinates are free and keep the trial as it is. Finite levels, infinite
# coordinate WEIGHTS at finite levels (#853) and finite factors whose threshold overflows (#866) keep
# their own rules.
fixed_zeros(r) = all(is_fixed_zero, r)
for T in (Float32, Float64)
    @testset "$T #665 zero step at an infinite level: every dense entry point" begin
        inf = T(Inf)
        cases = (
            # (function, trial, expected): unweighted; finite trials, ±0 included
            (NormL1(inf), T[2, -3, 0, -0.0], T[0, 0, 0, 0]),
            (NormL1(inf), T[NaN, Inf, -Inf, floatmax(T)], T[0, 0, 0, 0]),
            # weighted: positive weights fixed, the zero weight free
            (NormL1(inf, T[1, 0, 2]), T[3, 4, 5], T[0, 4, 0]),
            (NormL1(inf, T[1, 0, 2]), T[NaN, NaN, Inf], T[0, NaN, 0]),
            (NormL1(inf, T[1, 0, 2]), T[-Inf, -Inf, -0.0], T[0, -Inf, 0]),
            # a free -0 keeps the existing zero-threshold result shrink(-0, 0) = +0 (unchanged from master)
            (NormL1(inf, T[1, 0, 2]), T[-2, -0.0, nextfloat(T(0))], T[0, 0, 0]),
            (NormL1(inf, T[1, 0, 2]), T[floatmin(T), Inf, -floatmax(T)], T[0, Inf, 0]),
            # an infinite weight under the infinite level is a positive weight: fixed, non-finite trials included
            (NormL1(inf, T[Inf, 0]), T[Inf, NaN], T[0, NaN]),
            (NormL1(inf, T[floatmax(T), nextfloat(T(0))]), T[NaN, -Inf], T[0, 0]),
        )
        for (f, x, expected) in cases, γ in (zero(T), -zero(T))
            n = length(x)
            blocks = T[PB.prox_block(f, x[i], i, γ) for i in 1:n]
            y = prox(f, x, γ)
            out = fill(T(7), n); @test prox!(out, f, x, γ) === out
            alias = copy(x); @test prox!(alias, f, alias, γ) === alias
            vp = fill(T(7), n); val = value_and_prox!(vp, f, x, γ)
            vpa = copy(x); vala = value_and_prox!(vpa, f, vpa, γ)
            for r in (blocks, y, out, alias, vp, vpa)
                @test isequal(r, expected)                      # isequal: +0 at fixed entries, NaN and ±Inf kept where free
                @test eltype(r) === T
            end
            # the value at the stored output: 0 at the projection (zero-weight entries do not count, NaN included)
            @test isequal(val, f(vp)) && isequal(vala, f(vpa)) && val isa T
            @test val === zero(T) && vala === zero(T)
        end
        # the matrix layout takes the same linear-index weights
        f = NormL1(inf, T[1 0; 2 0])
        @test isequal(prox(f, T[3 NaN; -Inf 5], zero(T)), T[0 NaN; 0 5])
    end
    @testset "$T #665 per-entry steps: zero and positive neighbours" begin
        inf = T(Inf)
        f = NormL1(inf, T[1, 0, 2, 1, 0, 3])
        x = T[3, 4, NaN, -Inf, Inf, -0.0]
        for γs in (T[0, 0, 0, 0, 0, 0], T[0, 1, 0, 1, 0, 1], T[1, 0, 1, 0, 1, 0], T[0, 0, 2, 0, floatmax(T), nextfloat(T(0))])
            expected = T[0, 4, 0, 0, Inf, 0]
            y = prox(f, x, γs)
            out = similar(x); prox!(out, f, x, γs)
            alias = copy(x); prox!(alias, f, alias, γs)
            vp = similar(x); val = value_and_prox!(vp, f, x, γs)
            @test all(r -> isequal(r, expected), (y, out, alias, vp))
            @test isequal(T[PB.prox_block(f, x[i], i, γs[i]) for i in 1:6], expected)
            @test val === zero(T) && val === f(vp)
            M = reshape(copy(x), 2, 3)
            @test isequal(vec(prox(NormL1(inf, reshape(copy(f.w), 2, 3)), M, reshape(copy(γs), 2, 3))), expected)
        end
        # unweighted: every entry fixed, whatever its step
        @test fixed_zeros(prox(NormL1(inf), T[1, NaN, -Inf, 2], T[0, 0, 1, 0]))
        # a finite level next to it: the zero-step entries are left as they are, the positive ones thresholded
        @test isequal(prox(NormL1(T(1), T[1, 0, 2]), T[3, 4, NaN], T[0, 1, 0]), T[3, 4, NaN])
        @test isequal(prox(NormL1(T(1), T[1, 0, 2]), T[3, 4, 5], T[1, 0, 1]), T[2, 4, 3])
    end
    @testset "$T #665 finite levels, infinite weights (#853) and overflow (#866) keep their rules at γ = 0" begin
        z = zero(T)
        # finite levels: a zero threshold, the identity, non-finite trials included (a -0 trial gives the
        # existing shrink(-0, 0) = +0 and is left out of this identity check)
        for λ in T[0, 1, floatmax(T) / 4, floatmax(T)], w in (nothing, T[1, 0, 2, floatmax(T)])
            f = NormL1(λ, w)
            x = T[3, NaN, -Inf, -floatmax(T)]
            @test isequal(prox(f, x, z), x)
            @test isequal(prox(f, x, T[0, 0, 0, 0]), x)
            @test isequal(T[PB.prox_block(f, x[i], i, z) for i in 1:4], x)
        end
        # #853: an infinite coordinate weight at a finite level fixes FINITE trials at a zero step, and its
        # non-finite trials keep the narrower observed result (NaN), not the infinite-level guarantee
        for λ in T[0, 1, floatmax(T) / 4]
            f = NormL1(λ, T[Inf, Inf, Inf, Inf, 0])
            r = prox(f, T[2, -0.0, Inf, NaN, NaN], z)
            @test fixed_zeros(r[1:2]) && isnan(r[3]) && isnan(r[4]) && isnan(r[5])
            @test isnan(PB.prox_block(f, T(-Inf), 1, z))
        end
        # #866: finite factors whose threshold rounds to Inf are not an infinite level. A zero step gives
        # the zero threshold of finite factors; a positive step keeps the pinned #866 defect.
        big = T === Float32 ? (T(1e30), T(1e10)) : (T(1e300), T(1e10))
        f = NormL1(big[1], T[big[2]])
        @test isequal(prox(f, T[Inf], z), T[Inf]) && isequal(prox(f, T[-Inf], z), T[-Inf])
        @test isequal(prox(f, T[2], z), T[2])
        @test isnan(prox(f, T[Inf], T(1))[1])                                  # #866, unchanged
        # a strictly positive step whose product with a finite level underflows is not a zero step
        @test prox(NormL1(T(1)), T[2], nextfloat(z)) == T[2]
        @test prox(NormL1(floatmin(T)), T[2], floatmin(T)) == T[2]
        @test fixed_zeros(prox(NormL1(T(Inf)), T[2, NaN], nextfloat(z)))    # decision 49, unchanged
        # positive steps at an infinite level are unchanged
        @test isequal(prox(NormL1(T(Inf), T[1, 0, 2]), T[3, NaN, -Inf], T(1)), T[0, NaN, 0])
        @test isequal(prox(NormL1(T(Inf), T[1, 0, 2]), T[3, NaN, -Inf], T(Inf)), T[0, NaN, 0])
    end
    @testset "$T #665 zero step on SparseIterate" begin
        f = NormL1(T(Inf), T[1, 0, 2, 0, 1, 1, 3])
        # stored: a finite trial, a free trial, NaN, a free -Inf, an explicit active zero and a stored -0
        x = SparseIterate(T, 7)
        x[1] = 3; x[2] = 4; x[3] = NaN; x[4] = -Inf; activate!(x, 5); activate!(x, 6); x[6] = -0.0
        @test nactive(x) == 6 && signbit(x[6])
        for γ in (zero(T), T[0, 0, 0, 0, 0, 0, 0], T[0, 1, 0, 1, 0, 1, 0])
            expected = T[0, 4, 0, -Inf, 0, 0, 0]
            y = prox(f, x, γ)
            alias = copy(x); prox!(alias, f, alias, γ)
            out = copy(x); prox!(out, f, x, γ)
            vp = copy(x); val = value_and_prox!(vp, f, x, γ)
            for r in (y, alias, out, vp)
                @test isequal(Vector(r), expected)
            end
            # destinations keep their stored entries, now as +0 where fixed; a fresh output stores no new ones
            for r in (alias, out, vp)
                @test sort(collect(activecoordinates(r))) == 1:6
                @test is_fixed_zero(r[6]) && is_fixed_zero(r[1])
            end
            @test issubset(collect(activecoordinates(y)), 1:6)
            @test val === f(vp) && val === zero(T)
        end
        # unweighted, every stored entry is fixed; an unstored entry stays absent
        x = SparseIterate(T, 5); x[2] = NaN; x[4] = -Inf
        alias = copy(x); prox!(alias, NormL1(T(Inf)), alias, zero(T))
        @test fixed_zeros(Vector(alias)) && sort(collect(activecoordinates(alias))) == [2, 4]
        @test value_and_prox!(copy(x), NormL1(T(Inf)), x, zero(T)) === zero(T)
    end
    @testset "$T #665 zero step on BlockIterate" begin
        f = NormL1(T(Inf), T[1, 0, 2, 1, 0, 1])
        groups = [[1, 2], [3, 4], [5, 6]]
        x = BlockIterate(T, 6, groups)
        copyto!(blockview(x, 1), T[3, NaN]); copyto!(blockview(x, 2), T[-Inf, 0])
        for γ in (zero(T), T[0, 0, 0, 0, 0, 0], T[0, 1, 0, 0, 1, 1])
            y = prox(f, x, γ)
            alias = copy(x); prox!(alias, f, alias, γ)
            vp = copy(x); val = value_and_prox!(vp, f, x, γ)
            for r in (y, alias, vp)
                @test isequal(Vector(r), T[0, NaN, 0, 0, 0, 0])
            end
            # destinations keep their active blocks (block 2 now all +0); block 3 stays inactive; a fresh
            # output activates only blocks with a nonzero entry, as at a positive step
            for r in (alias, vp)
                @test collect(activecoordinates(r)) == collect(activecoordinates(x))
            end
            @test collect(activecoordinates(y)) == [1]
            # the value at the stored output: the free NaN entry has a zero weight and does not count
            @test val === f(vp) && val === zero(T)
        end
        copyto!(blockview(x, 1), T[3, 5])
        vp = copy(x)
        @test value_and_prox!(vp, f, x, zero(T)) === zero(T) && isequal(Vector(vp), T[0, 5, 0, 0, 0, 0])
    end
    @testset "$T #665 zero step on SymmetricSparseIterate (scalar steps, symmetric weights)" begin
        p = 3
        W = T[0 1 2; 1 0 Inf; 2 Inf 3]                 # free diagonal entries (1,1) and (2,2)
        f = NormL1(T(Inf), W)
        S = SymmetricSparseIterate(T, p)
        S[1, 1] = NaN; S[2, 1] = 4; S[2, 2] = -Inf; S[3, 1] = -0.0; S[3, 2] = Inf; S[3, 3] = 5
        expected = T[NaN 0 0; 0 -Inf 0; 0 0 0]
        for γ in (zero(T), -zero(T))
            y = prox(f, S, γ)
            alias = copy(S); prox!(alias, f, alias, γ)
            out = copy(S); prox!(out, f, S, γ)
            vp = copy(S); val = value_and_prox!(vp, f, S, γ)
            for r in (y, alias, out, vp)
                @test isequal(Matrix(r), expected)
            end
            for r in (alias, out, vp)
                @test collect(activecoordinates(r)) == collect(activecoordinates(S))
            end
            @test val === f(vp) && val === zero(T)              # the free NaN entry has a zero weight
            # a dense output of the same matrix, scalar and per-entry steps
            @test isequal(prox!(zeros(T, p, p), f, Matrix(S), γ), expected)
            @test isequal(prox!(zeros(T, p, p), f, Matrix(S), fill(γ, p, p)), expected)
        end
        S[1, 1] = 7; S[2, 2] = -1
        vp = copy(S)
        @test value_and_prox!(vp, f, S, zero(T)) === zero(T) && isequal(Matrix(vp), T[7 0 0; 0 -1 0; 0 0 0])
        @test fixed_zeros(Matrix(prox(NormL1(T(Inf)), S, zero(T))))
    end
    @testset "$T #665 zero step through Scaled" begin
        z = zero(T)
        f = NormL1(T(Inf), T[1, 0, 2])
        x = T[3, NaN, -Inf]
        expected = T[0, NaN, 0]
        tiny = T === Float32 ? T(2)^-100 : T(2)^-600                        # nested factors with a subnormal product
        wide = T === Float32 ? T(2)^-40 : T(2)^-300                         # a single factor outside the array band
        for s in (Scaled(f, T(2)), Scaled(f, floatmax(T)), Scaled(f, nextfloat(z)), Scaled(Scaled(f, tiny), tiny),
                  Scaled(f, wide), 3 * (2 * f))
            for γ in (z, T[0, 0, 0], T[0, 1, 0])
                y = prox(s, x, γ)
                alias = copy(x); prox!(alias, s, alias, γ)
                vp = similar(x); val = value_and_prox!(vp, s, x, γ)
                @test all(r -> isequal(r, expected), (y, alias, vp))
                @test val === s(vp) && val === z                 # the free NaN entry has a zero weight
            end
            @test isequal(T[PB.prox_block(s, x[i], i, z) for i in 1:3], expected)
            vp = zeros(T, 3)
            @test value_and_prox!(vp, s, T[3, 4, 5], z) === z && vp == T[0, 4, 0]
        end
        # zero scales keep their construction rules: an infinite level has no zero scaling, and a zero
        # level with an infinite weight is the #853 rule at a zero step
        @test_throws ArgumentError Scaled(NormL1(T(Inf)), z)
        @test_throws ArgumentError Scaled(f, z)
        s0 = Scaled(NormL1(T(1), T[Inf, 1]), z)
        @test s0 isa NormL1 && iszero(s0.λ0)
        @test isequal(prox(s0, T[3, 4], z), T[0, 4]) && isequal(prox(s0, T[3, 4], T(1)), T[0, 4])
        # a finite level under a finite factor is still the identity at a zero step
        @test isequal(prox(Scaled(NormL1(T(1), T[1, 0, 2]), T(2)), x, z), x)
    end
end


# A step per entry, as for ElasticNet (#925): for arguments and steps of concrete element types, a pass over the steps
# in blocks of 256 decides whether every product γᵢλ0 is in the normal range (it stops at the first block that holds
# one outside it); if so, the loop of the ordinary branch of the per-entry operator runs over the whole array, and
# otherwise the generic loop over prox_block. Either way, prox!, prox and value_and_prox! must give the bits of prox_block entry by
# entry (NaN for NaN), and throw where it throws.
function per_entry(f, x::Array{T}, γ) where {T}
    y = similar(x)
    for i in eachindex(IndexLinear(), x)
        y[i] = convert(T, PB.prox_block(f, x[i], i, γ[i]))
    end
    y
end
same_bits(a, b) = size(a) == size(b) && all(i -> isnan(b[i]) ? isnan(a[i]) : a[i] === b[i], eachindex(a, b))
# the result of a call, or the type of the exception that it throws
function outcome(g)
    try
        (true, g())
    catch e
        (false, typeof(e))
    end
end
agree(got, ref) = got[1] == ref[1] && (ref[1] ? same_bits(got[2], ref[2]) : got[2] === ref[2])

# every entry point on (f, x, γ) against prox_block; returns whether all of them agree
function check_steps(f, x::Array{T}, γ) where {T}
    ref = outcome(() -> per_entry(f, x, γ))
    ok = agree(outcome(() -> prox!(similar(x), f, x, γ)), ref)
    ok &= agree(outcome(() -> prox(f, x, γ)), ref)
    ok &= agree(outcome(() -> prox!(copy(x), f, x, γ)), ref)
    ok &= agree(outcome(() -> (z = copy(x); prox!(z, f, z, γ))), ref)                 # in place
    if γ isa Array && size(γ) == size(x)                                             # the steps are the output
        ok &= agree(outcome(() -> (g = copy(γ); prox!(g, f, x, g))), ref)
    end
    vp = similar(x)
    got = outcome(() -> (v = value_and_prox!(vp, f, x, γ); vp))
    ok &= agree(got, ref)
    # the value is that of the stored result, on the route of value_and_prox! (which can differ from f(y) in the last
    # place for a Float64 level with Float32 data, as on master)
    if ref[1] && got[1]
        ok &= isequal(value_and_prox!(similar(x), f, x, γ), PB._value_sep(f, ref[2]))
    end
    ok
end

# data with every class of value at scattered positions, the rest ordinary
function step_data(::Type{T}, dims, seed) where {T}
    rng = Xoshiro(seed)
    x = T(2) .* randn(rng, T, dims...)
    specials = T[NaN, Inf, -Inf, 0, -0.0, nextfloat(zero(T)), -3 * nextfloat(zero(T)), floatmin(T), floatmax(T), -floatmax(T)]
    for k in 1:min(length(x) ÷ 3, length(specials))
        x[rand(rng, eachindex(x))] = specials[k]
    end
    x
end

@testset "per-entry steps on arrays: the bits of prox_block ($T)" for T in (Float32, Float64)
    rng = Xoshiro(916)
    levels = Any[T(0.3), T(3), 2, zero(T), T(Inf), big(0.3)]   # a BigFloat level: the products are formed in BigFloat
    T === Float32 && push!(levels, 0.3, 3.0)        # a Float64 level with Float32 data: the products are formed in Float64
    # around the block length (256) of the pass over the steps, and a last partial block
    shapes = [(0,), (1,), (7,), (8,), (9,), (64,), (255,), (256,), (257,), (511,), (512,), (513,), (1000,), (8, 8), (25, 40)]
    for dims in shapes, λ in levels
        n = prod(dims)
        λ isa BigFloat && n > 257 && continue
        x = step_data(T, dims, 916 + n)
        wpos = T(0.5) .+ rand(rng, T, dims...)
        wmix = copy(wpos)
        n > 0 && (wmix[1:3:end] .= zero(T))
        n > 1 && (wmix[2:5:end] .= T(Inf))
        R = promote_type(T, float(typeof(λ)))
        λR = convert(R, λ)
        # steps whose product with λ0 is subnormal or overflows (where λ0 is finite and positive), and the other
        # exceptional steps
        scaled = R <: Union{Float32, Float64} && isfinite(λR) && λR > 0
        tiny = scaled ? convert(T, floatmin(R) / λR / 4) : nextfloat(zero(T))
        huge = scaled ? convert(T, min(floatmax(R) / λR * 2, R(floatmax(T)))) : floatmax(T)
        exceptional = T[0, -0.0, nextfloat(zero(T)), tiny, huge, Inf, NaN, -1, -Inf]
        for w in (nothing, wpos, wmix)
            f = NormL1(λ, w)
            γ = T(0.5) .+ rand(rng, T, dims...)
            @test check_steps(f, x, γ)
            n == 0 && continue
            # in the first block, at the ends and starts of blocks, in the middle, and in the last (partial) block: the
            # pass stops at its block and the generic loop does the whole array
            positions = unique(filter(i -> 1 <= i <= n, [1, 2, (n + 1) ÷ 2, 255, 256, 257, 511, 512, 513, n - 1, n]))
            for s in exceptional, at in positions
                γe = copy(γ)
                γe[at] = s
                ok = check_steps(f, x, γe)
                ok || @info "per-entry steps differ from prox_block" T dims λ w === nothing s at
                @test ok
            end
            # a step equal to floatmin(R) / λ0 and to floatmax(R) / λ0 where these are exact, at the ends of the range
            if λR == 2 || λR == 3
                for s in (floatmin(R) / λR, floatmax(R) / λR, prevfloat(floatmin(R) / λR), nextfloat(floatmax(R) / λR))
                    γe = copy(γ)
                    γe[end] = convert(T, s)
                    @test check_steps(f, x, γe)
                end
            end
        end
    end
    # views of the argument, the output and the steps take the same method, with the same results
    x = step_data(T, (100,), 1); γ = T(0.5) .+ rand(rng, T, 100)
    for f in (NormL1(T(0.3)), NormL1(T(0.3), T(0.5) .+ rand(rng, T, 100)))
        ref = per_entry(f, x, γ)
        @test same_bits(prox!(similar(x), f, view(x, :), γ), ref)
        @test same_bits(prox!(similar(x), f, x, view(γ, :)), ref)
        y = similar(x); prox!(view(y, :), f, x, γ)
        @test same_bits(y, ref)
        γe = copy(γ); γe[50] = T(NaN)
        @test same_bits(prox!(similar(x), f, view(x, :), γe), per_entry(f, x, γe))
    end
    # a step array that is the output (each step is read before its entry is written)
    f = NormL1(T(0.3))
    ref = per_entry(f, x, γ)
    g = copy(γ)
    prox!(g, f, x, g)
    @test same_bits(g, ref)
    # an array of steps of another shape with the same length is read in linear order, as before
    M = reshape(copy(x), 10, 10)
    @test same_bits(prox(f, M, γ), reshape(ref, 10, 10))
    # the lengths of the steps and of the argument must agree, before any write
    out = fill(T(7), 100)
    @test_throws DimensionMismatch prox!(out, f, x, γ[1:99])
    @test all(==(T(7)), out)
    # no allocation once compiled, ordinary steps
    y = similar(x)
    for f in (NormL1(T(0.3)), NormL1(T(0.3), T(0.5) .+ rand(rng, T, 100)), NormL1(0.3))
        @test allocs(prox!, y, f, x, γ) == 0
        @test allocs(value_and_prox!, y, f, x, γ) == 0
    end
end


# The wider signature: any AbstractArray output, argument and steps. Views and reshaped arrays take the pass and the
# ordinary loop; arguments or steps of an abstract element type, the generic loop; the iterates keep their routes (a
# SparseIterate into a SparseIterate its loop over the stored entries; a BlockIterate, a subtype of AbstractVector
# with no route of its own, the dense loop, as on master). Every result is compared bit for bit with the generic loop
# of separable.jl, called by invoke, or with prox_block.
generic_loop!(y, f, x, γ) = invoke(PB._prox_sep_dense!, Tuple{AbstractArray, Any, AbstractArray, Any}, y, f, x, γ)
@testset "per-entry steps on other arrays ($T)" for T in (Float32, Float64)
    rng = Xoshiro(925)
    n = 600
    x = step_data(T, (n,), 925); γ = T(0.5) .+ rand(rng, T, n)
    w = T(0.5) .+ rand(rng, T, n); w[1:7:n] .= zero(T); w[3:11:n] .= T(Inf)
    γe = copy(γ); γe[300] = T(NaN)
    for f in (NormL1(T(0.3)), NormL1(T(0.3), w), NormL1(2, w)), s in (γ, γe)
        ref = per_entry(f, x, s)
        M = PB._prox_sep_dense!
        @test which(M, Tuple{Vector{T}, typeof(f), Vector{T}, Vector{T}}) === which(M, Tuple{SubArray{T, 1, Vector{T}, Tuple{StepRange{Int, Int}}, true}, typeof(f), Vector{T}, Vector{T}}) !==
              which(M, Tuple{Vector{T}, Any, Vector{T}, Vector{T}})
        # views (strided, with a step range and with an index vector) and reshaped arrays for the output, the argument
        # and the steps
        u = zeros(T, 2n); u[1:2:2n] .= x; g = zeros(T, 2n); g[1:2:2n] .= s; perm = collect(1:2:2n)
        for xa in (view(u, 1:2:2n), view(u, perm), view(reshape(u, 2, n), 1, :)), sa in (s, view(g, 1:2:2n), view(g, perm))
            @test same_bits(prox!(similar(x), f, xa, sa), ref)
        end
        Y = zeros(T, 2n); prox!(view(Y, 2:2:2n), f, x, s)
        @test same_bits(Y[2:2:2n], ref)
        X = reshape(copy(x), 20, 30); S = reshape(copy(s), 30, 20)
        @test same_bits(vec(prox!(similar(X), f, X, S)), ref)
        R = reshape(view(copy(x), 1:n), 30, 20)
        @test same_bits(vec(prox(f, R, s)), ref)
        # abstract element types: the generic loop
        sr = Vector{Real}(s); sr[2] = 1
        xr = Vector{Real}(x)
        @test same_bits(prox!(similar(x), f, x, sr), per_entry(f, x, sr))
        @test same_bits(prox!(similar(x), f, xr, s), ref)
        # the iterates: a SparseIterate into a SparseIterate keeps its loop over the stored entries; a SparseIterate into
        # a dense array and a BlockIterate take this method; the results, and the stored or active entries, are those of
        # the generic loop
        xs = copy(x); xs[mod.(1:n, 8) .!= 1] .= zero(T)
        sp = PB.SparseIterate(xs)
        @test which(PB._prox_sep!, Tuple{typeof(sp), typeof(f), typeof(sp), Vector{T}}).sig.parameters[2] <: PB.SparseIterate
        a = prox!(copy(sp), f, sp, s)
        # the entries that are not stored stay zero (the loop visits the stored ones), also where the step is NaN
        stored = findall(!iszero, xs)
        @test same_bits(Vector(a)[stored], per_entry(f, xs, s)[stored]) && all(iszero, Vector(a)[setdiff(1:n, stored)])
        @test same_bits(prox!(similar(x), f, sp, s), generic_loop!(similar(x), f, sp, s))
        groups = [collect(1:2:n), collect(2:2:n)]
        bl = PB.BlockIterate(copy(x), groups)
        b1, b2 = copy(bl), copy(bl)
        prox!(b1, f, bl, s); generic_loop!(b2, f, bl, s)
        @test same_bits(Vector(b1), Vector(b2)) && same_bits(Vector(b1), ref)
        @test b1.nactive == b2.nactive && b1.active[1:b1.nactive] == b2.active[1:b2.nactive]
    end
    # no method of the same shape for another function is ambiguous with this one
    ambiguous = filter(Test.detect_ambiguities(ProximalBase; recursive = true)) do (m1, m2)
        m1.name === :_prox_sep_dense! || m2.name === :_prox_sep_dense!
    end
    @test isempty(ambiguous)
    @test length(methods(PB._prox_sep_dense!)) >= 3
end
end
