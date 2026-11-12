module InfiniteWeightsTest
# Characterization of the exceptional parameters of NormL1 (issue #853, resolved by narrowing the contract).
#
# Guaranteed:
#   * an infinite weight fixes its coordinate at zero for every FINITE trial, at every nonnegative
#     level and step, including a zero level and a zero step;
#   * an infinite level at a positive step fixes every positively weighted coordinate at zero, also at
#     NaN and ±Inf trials (decision 49);
#   * a zero weight leaves the coordinate unpenalized, NaN and ±Inf included.
# Observed, outside any guarantee (not a requirement): an infinite weight with a NaN or ±Inf trial and
# no infinite level at a positive step returns NaN from IEEE arithmetic. The broader guarantee was
# declined because it imposed measurable overhead on ordinary finite proximal calls.
# Known defect, pinned and not endorsed (#866): finite factors whose threshold overflows turn ±Inf into NaN.
# Homogeneous Float32 and Float64 are the supported configurations; no mixed-precision guarantee.
using Test, ProximalBase, LinearAlgebra, SparseArrays
const PB = ProximalBase

# The specification, written independently of the kernel. `:fixed` is the positive floating zero,
# `:identity` leaves the trial as it is (NaN stays NaN), `:observed_nan` is the unguaranteed IEEE
# outcome above, `:overflow` is a finite level, step and weight whose exact product exceeds floatmax(T)
# (#866), and `:threshold` is an ordinary soft threshold.
function classify(::Type{T}, v, γ, λ, w) where {T}
    if isinf(w)
        isfinite(v) && return :fixed                # guaranteed, whatever the level and the step
        isinf(λ) && γ > 0 && return :fixed          # decision 49
        return :observed_nan                        # outside the guarantee
    end
    iszero(w) && return :identity                   # unpenalized
    isinf(λ) && γ > 0 && return :fixed              # decision 49
    (isinf(λ) || iszero(λ) || iszero(γ)) && return :identity    # a zero threshold; γ = 0 with λ = Inf is not fixed
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
        # 2. infinite level, positive step, positive weight: zero also at non-finite trials (decision 49);
        #    a zero step does not fix the entry
        for v in T[2, Inf, -Inf, NaN], w in T[1, floatmax(T) / 4, Inf]
            @test is_fixed_zero(prox(NormL1(T(Inf), T[w]), [v], T(1))[1])
        end
        for v in T[2, Inf, -Inf, NaN], w in T[1, floatmax(T) / 4]
            @test isequal(prox(NormL1(T(Inf), T[w]), [v], T(0))[1], v)
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

end
