module NormL1NonnegExtremesTest
# NormL1Nonneg (#947): the endpoint truth table (the rejection of negative and NaN steps is in norml1nonneg.jl), values
# against BigFloat on the routes of the dense loops, the point-aware KKT scores and the full-point measure, and Scaled
# (folds, chains, infinite and zero factors, the hooks of scaled.jl).
#
# Proximal points are compared with max(v - t̂, 0) in BigFloat, t̂ the threshold that the NormL1 kernel forms, within 4
# units in the last place; values with λ0 Σ wᵢ xᵢ within (n + 2) eps relative for n terms (an absolute spacing of the
# subnormal numbers added where the terms or the sum are below floatmin); exact results (zeros, identities, infinities,
# NaN) with isequal or ===.
using Test, ProximalBase, LinearAlgebra, SparseArrays, Random
const PB = ProximalBase

big4096(f) = setprecision(f, BigFloat, 4096)
p2(::Type{T}, k) where {T} = ldexp(one(T), k)
weight(f, i) = f.w === nothing ? 1 : f.w[i]
threshold(::Type{T}, f, i, γ) where {T} = PB._l1_prox_threshold(zero(T), γ, f.λ0, PB._entry(f.w, i), T)[2]
function proxref(::Type{T}, f, v, i, γ) where {T}
  t = threshold(T, f, i, γ)
  big4096(() -> max(big(T(v)) - big(t), big(0)))
end
function within(got::T, ref::BigFloat, k = 4) where {T}
  r = T(ref)
  isfinite(r) || return isequal(got, r)
  isequal(got, r) || big4096(() -> abs(big(got) - ref) <= k * big(eps(abs(r))))
end
within(got, ref::AbstractFloat, k = 4) = isequal(got, ref) || abs(got - ref) <= k * eps(abs(ref))
# λ0 Σ wᵢ xᵢ over the entries of a dense array in BigFloat: NaN when an entry is NaN, otherwise Inf when one is
# negative; a zero coefficient λ0wᵢ and a zero entry give a zero term (0⋅Inf = 0, as the value of NormL1)
function valref(f, x)
  any(isnan, x) && return big4096(() -> big(NaN))
  any(<(0), x) && return big4096(() -> big(Inf))
  big4096() do
    s = big(0)
    for i in eachindex(x)
      c = (iszero(f.λ0) || iszero(weight(f, i))) ? big(0) : big(f.λ0) * big(weight(f, i))
      s += (iszero(c) || iszero(x[i])) ? big(0) : c * big(x[i])
    end
    s
  end
end
# within (n + 2) eps relative, and with `slack` of the subnormal spacing δ added for terms below floatmin
function valok(got::T, ref, n; slack = 0) where {T}
  isfinite(ref) || return isequal(got, T(ref))
  isfinite(got) && big4096(() -> abs(big(got) - ref) <= (n + 2) * eps(T) * abs(ref) + slack * big(nextfloat(zero(T))))
end
# two values at the same point agree: the same class (NaN, ±Inf) or within the sum of the bounds of the two sums
function agree(a::T, b::T, n) where {T}
  (isfinite(a) && isfinite(b)) ? abs(a - b) <= 2 * (n + 2) * eps(T) * abs(b) + 2 * nextfloat(zero(T)) : isequal(a, b)
end
plant(x, k, v) = (y = copy(x); y[k] = v; y)

@testset "NormL1Nonneg extremes" begin
  for T in (Float32, Float64)
    @testset "$T truth table: proximal points" begin
      trials = T[3, -3, 0.25, 0, -0.0, Inf, -Inf, NaN]
      projected = T[3, 0, 0.25, 0, 0, Inf, 0, NaN]                       # max(v, 0): -0 and -Inf give +0
      steps = T[0, 0.5, 1, Inf]
      isproj(f, γ) = all(isequal(prox_block(f, trials[k], 1, γ), projected[k]) for k in eachindex(trials))
      # finite positive level, weight and step: max(v - t̂, 0); a tie and -0 give +0
      f = NormL1Nonneg(T(0.5), T[1])
      @test isequal(prox_block(f, T(3), 1, T(0.5)), T(2.75)) && prox_block(f, T(-3), 1, T(0.5)) === zero(T)
      @test prox_block(f, T(0.25), 1, T(0.5)) === zero(T) && prox_block(f, -zero(T), 1, T(0.5)) === zero(T)
      @test prox_block(f, T(Inf), 1, T(0.5)) === T(Inf) && prox_block(f, -T(Inf), 1, T(0.5)) === zero(T)
      @test isnan(prox_block(f, T(NaN), 1, T(0.5)))
      # a zero level, a zero weight (also at an infinite level) and a zero step with finite level and weight: the
      # projection onto the orthant, not the identity
      for w in (nothing, T[0], T[1], T[2]), γ in steps
        @test isproj(NormL1Nonneg(zero(T), w), γ)
      end
      for λ0 in T[0.5, 1, Inf], γ in steps
        @test isproj(NormL1Nonneg(λ0, T[0]), γ)
      end
      for λ0 in T[0.5, 2], w in (nothing, T[1], T[3])
        @test isproj(NormL1Nonneg(λ0, w), zero(T))
      end
      # an infinite level fixes positive-weight entries at +0 at each step below, NaN and ±Inf trials included (#665)
      for w in (nothing, T[1], T[0.5], T[Inf]), γ in steps
        f = NormL1Nonneg(T(Inf), w)
        @test all(prox_block(f, v, 1, γ) === zero(T) for v in trials)
      end
      # an infinite weight at a finite level maps a finite trial and -Inf to +0 at each level and step below, zero
      # included (#853); +Inf and NaN give NaN
      for λ0 in T[0, 1], γ in steps
        f = NormL1Nonneg(λ0, T[Inf])
        @test all(prox_block(f, v, 1, γ) === zero(T) for v in T[3, -3, 0.25, 0, -0.0, -Inf, floatmax(T)])
        @test all(isnan(prox_block(f, v, 1, γ)) for v in T[Inf, NaN])
      end
      # finite factors whose threshold overflows: a finite trial and -Inf give +0, +Inf gives NaN (#866)
      huge = floatmax(T) / 4
      fO = NormL1Nonneg(huge, T[huge])
      @test prox_block(fO, T(3), 1, one(T)) === zero(T) && prox_block(fO, -floatmax(T), 1, one(T)) === zero(T)
      @test isnan(prox_block(fO, T(Inf), 1, one(T)))
      # an infinite step with a finite level and a positive weight: +0 for a finite trial and -Inf, NaN for +Inf and NaN
      for λ0 in T[0.5, 2], w in (nothing, T[1], T[3])
        f = NormL1Nonneg(λ0, w)
        @test all(isequal(prox_block(f, trials[k], 1, T(Inf)), T[0, 0, 0, 0, 0, NaN, 0, NaN][k]) for k in eachindex(trials))
      end
      # a NaN trial gives NaN, except under an infinite level with a positive weight
      for (λ0, w) in ((T(0.5), T[2]), (zero(T), nothing), (T(0.5), T[0]), (T(Inf), T[0]))
        @test all(isnan(prox_block(NormL1Nonneg(λ0, w), T(NaN), 1, γ)) for γ in T[0, 0.5, Inf])
      end
      # -0 and +0 trials give +0 at the levels, weights and steps below
      for λ0 in T[0, 0.5], w in (nothing, T[0], T[2]), γ in T[0, 1]
        f = NormL1Nonneg(λ0, w)
        @test prox_block(f, -zero(T), 1, γ) === zero(T) && prox_block(f, zero(T), 1, γ) === zero(T)
      end
      # the array loops give the same points: the zero level of the plan, bit for bit
      f = NormL1Nonneg(zero(T))
      @test isequal(prox(f, trials, one(T)), projected) && isequal(prox(f, trials, zero(T)), projected)
    end
  end

  for T in (Float32, Float64)
    @testset "$T truth table: values" begin
      tiny = nextfloat(zero(T))
      # a zero level or a zero weight: 0 at a feasible point, +Inf included; Inf at any negative number; NaN at NaN
      for (λ0, w) in ((zero(T), nothing), (zero(T), T[2]), (T(3), T[0]), (T(Inf), T[0]))
        f = NormL1Nonneg(λ0, w)
        @test all(value_entry(f, v, 1) === zero(T) for v in T[3, Inf, 0, -0.0, tiny, floatmax(T)])
        @test all(value_entry(f, v, 1) === T(Inf) for v in T[-tiny, -1, -Inf, -floatmax(T)])
        @test isnan(value_entry(f, T(NaN), 1))
        n = 4
        fa = NormL1Nonneg(λ0, w === nothing ? nothing : fill(w[1], n))
        @test fa(T[3, Inf, 0, -0.0]) === zero(T) && fa(T[3, -tiny, 0, 0]) === T(Inf) && fa(T[-Inf, 1, 1, 1]) === T(Inf)
        @test isnan(fa(T[NaN, -1, 0, 0])) && isnan(fa(T[0, 0, NaN, Inf]))
        @test fa(SparseIterate(T[3, Inf, 0, -0.0])) === zero(T) && fa(SparseIterate(T[3, -tiny, 0, 0])) === T(Inf)
      end
      # an infinite level with a positive weight: 0 at an entry 0, Inf at any other number, NaN at NaN
      for w in (nothing, T[1], T[Inf])
        f = NormL1Nonneg(T(Inf), w)
        @test value_entry(f, zero(T), 1) === zero(T) && value_entry(f, -zero(T), 1) === zero(T)
        @test all(value_entry(f, v, 1) === T(Inf) for v in T[tiny, 1, Inf, -tiny, -1, -Inf])
        @test isnan(value_entry(f, T(NaN), 1))
        n = 4
        fa = NormL1Nonneg(T(Inf), w === nothing ? nothing : fill(w[1], n))
        @test fa(zeros(T, n)) === zero(T) && fa(T[0, -0.0, 0, 0]) === zero(T)
        @test fa(T[0, tiny, 0, 0]) === T(Inf) && fa(T[-Inf, 0, 0, 0]) === T(Inf) && fa(T[0, -tiny, 0, 0]) === T(Inf)
        @test isnan(fa(T[NaN, 0, 0, 0])) && isnan(fa(T[NaN, -1, 0, 0]))
      end
      # an infinite weight at a finite level, a zero level included: 0 at 0, Inf at any other number, NaN at NaN
      for λ0 in T[0, 1]
        f = NormL1Nonneg(λ0, T[Inf])
        @test value_entry(f, zero(T), 1) === zero(T) && value_entry(f, -zero(T), 1) === zero(T)
        @test all(value_entry(f, v, 1) === T(Inf) for v in T[tiny, 1, Inf, -tiny, -1, -Inf])
        @test isnan(value_entry(f, T(NaN), 1))
      end
      # a positive coefficient: the NormL1 value at v ≥ 0, with +Inf giving Inf; Inf below zero
      f = NormL1Nonneg(T(2), T[3])
      @test value_entry(f, T(0.5), 1) === value_entry(NormL1(T(2), T[3]), T(0.5), 1) && value_entry(f, T(Inf), 1) === T(Inf)
      @test value_entry(f, T(-0.5), 1) === T(Inf) && value_entry(f, -zero(T), 1) === zero(T)
      # the array value: NaN if an entry that is read is NaN, else Inf if one is negative, else the NormL1 value
      for (λ0, w) in ((T(0.5), nothing), (T(0.5), T[1, 2, 3]), (zero(T), T[1, 2, 3]), (T(Inf), T[1, 2, 3]), (T(0.5), T[1, Inf, 3]))
        f = NormL1Nonneg(λ0, w)
        n = 3
        @test isnan(f(T[NaN, -1, 2])) && isnan(f(T[-1, NaN, 2])) && isnan(f(T[2, -1, NaN])) && isnan(f(T[NaN, 1, 2]))
        @test f(T[-1, 1, 2]) === T(Inf) && f(T[1, 1, -tiny]) === T(Inf) && f(T[1, -T(Inf), 2]) === T(Inf)
        @test isnan(f(SparseIterate(T[NaN, -1, 2]))) && f(SparseIterate(T[-1, 1, 2])) === T(Inf)
        @test isnan(f(BlockIterate(T[NaN, -1, 2], [[1, 2], [3]]))) && f(BlockIterate(T[-1, 1, 2], [[1, 2], [3]])) === T(Inf)
        @test isnan(f(view(T[NaN, -1, 2], 1:3))) && f(view(T[-1, 1, 2], 1:3)) === T(Inf)
        g = NormL1(λ0, w)
        @test agree(f(T[1, 0, 2]), g(T[1, 0, 2]), n) && agree(f(T[0, 0, 0]), g(T[0, 0, 0]), n)
        @test agree(f(T[1, Inf, 2]), g(T[1, Inf, 2]), n) && agree(f(T[-0.0, 0, 3]), g(T[-0.0, 0, 3]), n)
      end
      S = SymmetricSparseIterate(T, 2); S[1, 1] = 1; S[2, 1] = 2
      fs = NormL1Nonneg(T(0.5), T[1 2; 2 1])
      @test fs(S) === T(0.5) * (1 + 2 * 2 * 2) && NormL1(T(0.5), T[1 2; 2 1])(S) === fs(S)
      S[2, 1] = -tiny
      @test fs(S) === T(Inf)
      S[2, 1] = T(NaN)
      @test isnan(fs(S))
    end
  end

  for T in (Float32, Float64)
    @testset "$T values against BigFloat" begin
      rng = Xoshiro(947)
      E, m = exponent(floatmax(T)), exponent(floatmin(T))
      for n in (2, 3, 8, 15, 16, 17, 100, 1023, 1024, 1025, 2048, 4097)
        x = abs.(T(3) .* randn(rng, T, n))
        x[rand(rng, n) .< 0.2] .= zero(T)
        x[1] = T(0.75)
        w = T(0.5) .+ rand(rng, T, n)
        for (λ, ww) in ((T(0.3), nothing), (T(0.3), w), (T(1e-3), w), (T(40), nothing), (zero(T), w), (zero(T), nothing))
          f = NormL1Nonneg(λ, ww)
          r = valref(f, x)
          got = f(x)
          @test valok(got, r, n)
          @test agree(got, NormL1(λ, ww)(x), n)
          @test valok(f(view(copy(x), :)), r, n)                           # the generic route
          sx = SparseIterate(x)
          bx = BlockIterate(copy(x), [collect(1:2:n), collect(2:2:n)])
          @test valok(f(sx), r, n) && valok(f(bx), r, n)
          # the three iterates sum in the same order as NormL1: bit for bit
          @test f(sx) === NormL1(λ, ww)(sx) && f(bx) === NormL1(λ, ww)(bx)
          # infeasible and NaN entries, at the first, a middle and the last position
          for pos in (1, n ÷ 2 + 1, n)
            @test f(plant(x, pos, -nextfloat(zero(T)))) === T(Inf) && f(plant(x, pos, -one(T))) === T(Inf)
            @test isnan(f(plant(x, pos, T(NaN)))) && isnan(f(plant(plant(x, pos, T(NaN)), mod1(pos + 1, n), -one(T))))
            @test f(plant(x, pos, -zero(T))) == f(plant(x, pos, zero(T)))
            @test f(plant(x, pos, -T(Inf))) === T(Inf)
            @test f(plant(x, pos, T(Inf))) === (iszero(λ) ? zero(T) : T(Inf))
          end
        end
      end
      # a single entry and the empty array
      @test NormL1Nonneg(T(2))(T[3]) === T(6) && NormL1Nonneg(T(2))(T[-3]) === T(Inf) && NormL1Nonneg(T(2))(T[]) === zero(T)
      @test NormL1Nonneg(zero(T))(T[]) === zero(T) && NormL1Nonneg(T(Inf))(T[]) === zero(T) && NormL1Nonneg(T(2), T[])(T[]) === zero(T)

      # every early return of the unweighted dense method, on both sides of the loop switch
      tiny = nextfloat(zero(T))
      for n in (3, 16, 100, 1024, 1025, 2048)
        xs = abs.(randn(rng, T, n)) .+ T(0.1)
        # zero level: 0 with +Inf and -0 entries; Inf at a negative entry; NaN at a NaN entry
        f0 = NormL1Nonneg(zero(T))
        @test f0(xs) === zero(T) && f0(plant(xs, n ÷ 2, T(Inf))) === zero(T) && f0(plant(xs, 1, -zero(T))) === zero(T)
        @test f0(plant(xs, n, -tiny)) === T(Inf) && f0(plant(plant(xs, 1, T(Inf)), n, -tiny)) === T(Inf)
        @test isnan(f0(plant(xs, 1, T(NaN)))) && isnan(f0(plant(plant(xs, 1, T(NaN)), n, -tiny)))
        # an infinite level: 0 at the zero vector, Inf at a tiny positive or a negative entry, NaN at NaN
        fI = NormL1Nonneg(T(Inf))
        @test fI(zeros(T, n)) === zero(T) && fI(plant(zeros(T, n), n, -zero(T))) === zero(T)
        @test fI(plant(zeros(T, n), n, tiny)) === T(Inf) && fI(plant(zeros(T, n), 1, -tiny)) === T(Inf)
        @test isnan(fI(plant(zeros(T, n), 2, T(NaN)))) && isnan(fI(plant(plant(zeros(T, n), 2, T(NaN)), 1, -one(T))))
        # the sum or the product overflows with finite entries and the level brings it back, or does not
        bigx = floatmax(T) / 4
        fb = NormL1Nonneg(T(1e-3))
        xb = fill(bigx, n)
        @test valok(fb(xb), valref(fb, xb), n) && isfinite(fb(xb))
        @test fb(plant(xb, 2, -tiny)) === T(Inf) && isnan(fb(plant(xb, 2, T(NaN))))
        @test NormL1Nonneg(T(4))(plant(zeros(T, n), 1, floatmax(T) / 2)) === T(Inf)
        @test NormL1Nonneg(T(2))(xb) === T(Inf) && NormL1Nonneg(T(2))(plant(xb, 3, -one(T))) === T(Inf)
      end
      # a level that does not fit the type of the data: Float64 levels with Float32 data
      if T === Float32
        fl = NormL1Nonneg(1e-40)
        @test fl(Float32[-1]) === Inf32 && fl(Float32[1, -1e-30, 3]) === Inf32 && isnan(fl(Float32[NaN, -1]))
        for xl in (Float32[1], fill(1.0f0, 100), fill(1.0f0, 2048), Float32[0, 0, 0], Float32[3, 0.5, 20])
          ref = big4096(() -> big(1e-40) * sum(big, xl; init = big(0)))
          @test big4096(() -> abs(big(fl(xl)) - ref) <= big(eps(Float32(ref))))
        end
        @test fl(Float32[Inf, 1]) === Inf32 && fl(Float32[0, 0]) === 0.0f0 && fl(Float32[-1, Inf]) === Inf32
        @test fl(fill(-1.0f0, 100)) === Inf32 && isnan(fl(plant(fill(-1.0f0, 100), 50, NaN32)))
        # a level that overflows the type of the data
        fh = NormL1Nonneg(1e300)
        @test fh(Float32[-1]) === Inf32 && fh(Float32[0, 0]) === 0.0f0
      end

      # terms below floatmin, where a sum of rounded terms loses what the recovery of NormL1 restores (#373): the sum is
      # within (n + 2) eps of the exact value, with one subnormal spacing for the final rounding
      for n in (8, 64, 100, 2048)
        x = T(0.25) .* (1 .+ rand(rng, T, n))
        w = T(0.5) .+ rand(rng, T, n)
        for λ in (p2(T, m - 20), p2(T, m - 5), floatmin(T), p2(T, m ÷ 2))
          f = NormL1Nonneg(λ, w)
          r = valref(f, x)
          @test valok(f(x), r, n; slack = 1)
          @test agree(f(x), NormL1(λ, w)(x), n)
          @test valok(f(SparseIterate(x)), r, n; slack = 1) && f(SparseIterate(x)) === NormL1(λ, w)(SparseIterate(x))
          @test valok(f(view(copy(x), :)), r, n; slack = 1)
        end
        for λ in (p2(T, m - 20), p2(T, m ÷ 2), floatmin(T) / 4)
          f = NormL1Nonneg(λ)
          @test valok(f(x), valref(f, x), n; slack = 1)
        end
        # a negative entry stays Inf, and NaN stays NaN, in the same regime
        f = NormL1Nonneg(p2(T, m - 20), w)
        @test f(plant(x, n ÷ 2, -x[n ÷ 2])) === T(Inf) && isnan(f(plant(x, 1, T(NaN))))
      end
    end
  end

  for T in (Float32, Float64)
    @testset "$T point-aware KKT scores" begin
      n = 64
      positions = (1, 32, 64)
      # a +Inf (or -Inf) gradient entry at a zero entry: the score is Inf at each position, in the loop over an Array, in the
      # pass by entry of a view and in the four-argument form
      for w in (nothing, T(0.5) .+ rand(Xoshiro(947), T, n), [T(Inf); ones(T, n - 1)], zeros(T, n)), λ0 in T[1, 0, Inf]
        f = NormL1Nonneg(λ0, w)
        for pos in positions, vinf in T[Inf, -Inf]
          v = plant(zeros(T, n), pos, vinf)
          x = zeros(T, n)
          @test kkt_violation(f, v, x) === T(Inf)
          @test kkt_violation(f, view(copy(v), :), view(copy(x), :)) === T(Inf)
          @test kkt_violation(f, v, x, pos) === T(Inf)
          @test maximum(kkt_violation(f, v, x, i) for i in 1:n) === T(Inf)
          @test kkt_violation(f, v, plant(x, mod1(pos + 1, n), T(0.5)), pos) === T(Inf)
        end
      end
      # NaN and infinities in v and x, at the first, a middle and the last entry, against the pass by entry
      f = NormL1Nonneg(T(0.3), T(0.5) .+ rand(Xoshiro(948), T, n))
      v0 = T(0.2) .* randn(Xoshiro(949), T, n)
      x0 = zeros(T, n)
      x0[2:2:end] .= T(0.5)
      for pos in positions
        for (vv, xx, expect) in ((T(NaN), zero(T), NaN), (T(NaN), T(0.5), NaN), (-T(Inf), zero(T), Inf), (T(Inf), T(0.5), Inf),
                                 (T(0.1), -T(0.5), Inf), (T(0.1), -zero(T), nothing), (T(0.1), T(NaN), NaN), (T(0.1), T(Inf), Inf),
                                 (T(0.1), -T(Inf), Inf), (T(NaN), -T(0.5), NaN), (T(Inf), T(NaN), NaN), (T(0.1), nextfloat(zero(T)), nothing))
          v, x = plant(v0, pos, vv), plant(x0, pos, xx)
          loc = kkt_violation(f, v, x, pos)
          full = kkt_violation(f, v, x)
          byentry = kkt_violation(f, view(copy(v), :), view(copy(x), :))
          expect === nothing || @test isequal(loc, T(expect))
          expect === nothing || @test isequal(full, T(expect))
          @test isequal(full, byentry)
          @test isequal(full, max(zero(T), maximum(kkt_violation(f, v, x, i) for i in 1:n)))
        end
        # NaN beats Inf in the full-point form, whichever entry holds it
        other = mod1(pos + 17, n)
        @test isnan(kkt_violation(f, plant(v0, pos, T(NaN)), plant(x0, other, -T(0.5))))
        @test isnan(kkt_violation(f, plant(v0, other, T(Inf)), plant(x0, pos, T(NaN))))
        @test kkt_violation(f, plant(v0, pos, T(Inf)), plant(x0, other, -T(0.5))) === T(Inf)
      end
      # an infinite coefficient: -Inf at a zero entry for a finite v, and the maximum with zero is 0
      fw = NormL1Nonneg(T(1), T[1, Inf, 0.5, 0])
      g4 = T[0.25, 3, -0.1, 7]
      x4 = zeros(T, 4)
      @test kkt_violation(fw, g4, 2) === -T(Inf) && kkt_violation(fw, g4, x4, 2) === -T(Inf) && kkt_violation(fw, g4, 4) === -T(7)
      @test kkt_violation(fw, g4, T[0, 0.5, 0, 0], 2) === T(Inf) && kkt_violation(fw, T[0.25, 3, -0.1, -9], x4) === T(9)
      @test kkt_violation(NormL1Nonneg(T(Inf)), g4, x4, 1) === -T(Inf) && kkt_violation(NormL1Nonneg(T(Inf)), g4, x4) === zero(T)
      # the score at a zero entry is the screen of the three-argument form, bit for bit, for a finite gradient
      for λ0 in T[0, 0.3, 1.5], ww in (nothing, T[1, 0, 2, 0.5])
        f = NormL1Nonneg(λ0, ww)
        g = T[0.7, -0.4, 1.9, -2.5]
        for i in 1:4, xz in (zero(T), -zero(T))
          xs = plant(T[1, 2, 3, 4], i, xz)
          @test kkt_violation(f, g, xs, i) === kkt_violation(f, g, i)
        end
        @test kkt_violation(f, g, 1) === -g[1] - PB._kkt_coefficient(T, λ0, PB._entry(ww, 1))
        @test_throws ArgumentError kkt_violation(f, ComplexF64[1, 2, 3, 4], 1)
        @test_throws DimensionMismatch kkt_violation(f, T[1, 2, 3], zeros(T, 4))
        ww === nothing || @test_throws DimensionMismatch kkt_violation(f, T[1, 2, 3], zeros(T, 3))
      end
      # a positive entry scores |vᵢ + cᵢ|, a negative one Inf; against BigFloat for the entries of a vector
      let rng = Xoshiro(950), p = 40
        w = T(0.5) .+ rand(rng, T, p)
        f = NormL1Nonneg(T(0.3), w)
        v = T(2) .* randn(rng, T, p)
        x = randn(rng, T, p)
        x[1:3:end] .= zero(T)
        xf = max.(x, zero(T))                                              # no negative entry
        for i in 1:p
          c = big4096(() -> big(T(0.3)) * big(w[i]))
          ref = iszero(xf[i]) ? big4096(() -> -big(v[i]) - c) : big4096(() -> abs(big(v[i]) + c))
          got = kkt_violation(f, v, xf, i)
          @test got isa T && abs(big(got) - ref) <= 3 * eps(T) * (abs(big(v[i])) + c)
          @test kkt_violation(f, v, plant(xf, i, -one(T)), i) === T(Inf)
        end
        full = big4096(() -> max(big(0), maximum(iszero(xf[i]) ? -big(v[i]) - big(T(0.3)) * big(w[i]) : abs(big(v[i]) + big(T(0.3)) * big(w[i])) for i in 1:p)))
        @test abs(big(kkt_violation(f, v, xf)) - full) <= 3 * eps(T) * (maximum(abs.(v)) + 2)
        @test kkt_violation(f, v, x) === (any(<(0), x) ? T(Inf) : kkt_violation(f, v, xf))
        # a mixed element type: Float64 level and weights, Float32 data, the score in the type of the data
        if T === Float32
          fm = NormL1Nonneg(0.3, Float64.(w))
          @test kkt_violation(fm, v, xf, 2) isa Float32 && kkt_violation(fm, v, xf) isa Float32
          @test kkt_violation(fm, Float64.(v), xf, 2) isa Float64
        end
      end
      # a block, and a symmetric coordinate: the parameters at the lower position
      b = BlockIterate(T[0, 0, 1.5, 0, 0, 2], [[1, 2], [3, 4], [5, 6]])
      fb = NormL1Nonneg(T(0.5), T[1, 2, 1, 2, 1, 2])
      gb = T[0.1, 0.2, -3, 4, -5, 0.3]
      @test kkt_violation(fb, gb, b, 1) === max(kkt_violation(fb, gb, 1), kkt_violation(fb, gb, 2))
      @test isequal(kkt_violation(fb, gb, b, 2), max(kkt_violation(fb, gb, Vector(b), 3), kkt_violation(fb, gb, Vector(b), 4)))
      bn = BlockIterate(T[0, 0, -1.5, 0, 0, 2], [[1, 2], [3, 4], [5, 6]])
      @test kkt_violation(fb, gb, bn, 2) === T(Inf)
      fs = NormL1Nonneg(T(0.5), T[1 2; 2 1])
      S = SymmetricSparseIterate(T, 2)
      G = T[0.1 3; 3 0.4]
      @test kkt_violation(fs, G, S, 2) === -G[2, 1] - T(0.5) * 2          # coordinate (2, 1), a zero coordinate
      S[2, 1] = 1
      @test isequal(kkt_violation(fs, G, S, 2), abs(G[2, 1] + T(0.5) * 2))
      S[2, 1] = -1
      @test kkt_violation(fs, G, S, 2) === T(Inf)
      @test_throws ArgumentError kkt_violation(fs, G, S)
      @test_throws BoundsError kkt_violation(fs, G, S, 4)
      # Scaled, finite factor: α times the score at v/α; α = 2 is NormL1Nonneg(2λ0) bit for bit, the screen too
      leaf = NormL1Nonneg(T(0.3), T[1, 0.5, 2, 0])
      vv = T[0.7, -0.4, 1.9, -2.5]
      xx = T[0, 0.5, 0, -0.5]
      sc, ref = Scaled(leaf, T(2)), NormL1Nonneg(T(0.6), leaf.w)
      for i in 1:4
        @test isequal(kkt_violation(sc, vv, xx, i), kkt_violation(ref, vv, xx, i)) && kkt_violation(sc, vv, i) === kkt_violation(ref, vv, i)
      end
      @test kkt_violation(sc, vv, xx) === kkt_violation(ref, vv, xx) === T(Inf)
      @test kkt_violation(sc, vv, T[0, 0.5, 0, 0]) === kkt_violation(ref, vv, T[0, 0.5, 0, 0])
      @test_throws ArgumentError kkt_violation(sc, vv, S)
    end
  end

  @testset "the screen of integer, unsigned and Bool gradients" begin
    # the gradient entry is converted to the promotion of its type and the coefficient's before the sign changes, so that
    # an unsigned entry or a typemin entry does not wrap: the result is the Float64 screen of the converted entry
    w4 = [1.0, 2.0, 0.5, 4.0]
    for f in (NormL1Nonneg(0.5), NormL1Nonneg(0.3, w4), NormL1Nonneg(0.0, w4), NormL1Nonneg(Inf, w4))
      n = 4
      for g in (UInt8[0, 3, 200, 255], Int8[-128, -1, 0, 127], Int[typemin(Int), -5, 0, typemax(Int)],
                Bool[false, true, true, false], UInt16[0, 1, 65535, 7], UInt64[0, 5, typemax(UInt64), 9], Int32[typemin(Int32), -2, 2, typemax(Int32)])
        gd = Float64.(g)
        for i in 1:n
          r = kkt_violation(f, g, i)
          @test r isa Float64 && isequal(r, kkt_violation(f, gd, i)) && r == -gd[i] - PB._kkt_coefficient(Float64, f.λ0, PB._entry(f.w, i))
        end
        # the point-aware form converts the gradient before it forms the score
        @test isequal(kkt_violation(f, g, zeros(n), 1), kkt_violation(f, gd, zeros(n), 1))
      end
    end
    # a wrapped negation would give 56 for the unsigned entry 200
    @test kkt_violation(NormL1Nonneg(0.5), UInt8[200], 1) === -200.5
    @test kkt_violation(NormL1Nonneg(0.5), Int8[-128], 1) === 127.5
    # integer levels and weights keep the integer type of the coefficient, as the screen of NormL1 does
    @test kkt_violation(NormL1Nonneg(2), Int[-3, 0, 5], 1) === 1 && kkt_violation(NormL1Nonneg(2), Int[-3, 0, 5], 3) === -7
    @test kkt_violation(NormL1Nonneg(2), UInt8[3], 1) === -5 && kkt_violation(NormL1Nonneg(2, [3]), Int8[-128], 1) === 122
    @test typeof(kkt_violation(NormL1(2), UInt8[3], 1)) === typeof(kkt_violation(NormL1Nonneg(2), UInt8[3], 1))
    @test typeof(kkt_violation(NormL1(0.5), Bool[true], 1)) === typeof(kkt_violation(NormL1Nonneg(0.5), Bool[true], 1))
    # floating gradients are unchanged: the arithmetic -gᵢ - c in the type of the gradient
    for T in (Float16, Float32, Float64)
      g = T[0.2, -0.7, 1.5, 0.0, -0.0]
      for f in (NormL1Nonneg(0.5), NormL1Nonneg(0.5, w4[1:1]), NormL1Nonneg(T(0.3), fill(T(2), 5)))
        for i in eachindex(g)
          i > length(f.w === nothing ? g : f.w) && continue
          r = kkt_violation(f, g, i)
          @test r isa T && r === -g[i] - PB._kkt_coefficient(T, f.λ0, PB._entry(f.w, i))
        end
      end
    end
    @test kkt_violation(NormL1Nonneg(big"0.5"), [big"2.5"], 1) isa BigFloat
  end

  for T in (Float32, Float64)
    @testset "$T Scaled" begin
      E, m = exponent(floatmax(T)), exponent(floatmin(T))
      f = NormL1Nonneg(T(0.4), T[1, 2, 0, 0.5])
      x = T[3, -2, 0.5, -0.25]
      xp = abs.(x)
      γ = T(0.3)
      # an ordinary factor: the value and the proximal point of f at the scaled step
      @test (2 * f)(xp) === 2 * f(xp) && (2 * f)(x) === T(Inf) && prox(2 * f, x, γ) == prox(f, x, 2γ) && (f * T(0.5))(xp) === T(0.5) * f(xp)
      y = similar(x)
      @test value_and_prox!(y, 3 * f, x, γ) === (3 * f)(y) && y == prox(f, x, 3γ)
      @test prox_block(2 * f, x[1], 1, γ) === prox_block(f, x[1], 1, 2γ)
      @test kkt_violation(2 * f, T[1, 1, 1, 1], 2) ≈ -1 - 2 * T(0.4) * 2
      @test restrict(2 * f, [2]) isa Scaled && restrict(2 * f, [2]).f.w == T[2]
      # a step that is negative or NaN is not checked through Scaled (a precondition), and is checked on the function
      @test prox(2 * f, x, -one(T)) isa Vector{T}
      @test_throws ArgumentError prox(f, x, -one(T))
      # a step product that overflows: the factor folds into the level (powers of two: exact)
      a = p2(T, 3E ÷ 5)
      λ = p2(T, m + 4)
      t = T(big4096(() -> big(a) * big(a) * big(λ)))
      g = Scaled(NormL1Nonneg(λ), a)
      v = 3t / 2
      @test prox(g, [v, -v], a) == [t / 2, 0]
      @test isequal(prox(g, [v, -v], a), T[t / 2, 0]) && !signbit(prox(g, [v, -v], a)[2])
      # a step product that underflows to zero: the threshold still acts
      b = p2(T, m ÷ 2 - 30)
      tb = T(big4096(() -> big(b) * big(b) * big(p2(T, E))))
      g = Scaled(NormL1Nonneg(p2(T, E)), b)
      @test prox(g, [tb / 2, 3tb / 2], b) == [zero(T), tb / 2]
      # a weight multiplies the folded threshold: the fold keeps the factors of the entry
      w0 = p2(T, 4)
      tw = T(big4096(() -> big(a) * big(a) * big(λ) * big(w0)))
      gw = Scaled(NormL1Nonneg(λ, T[w0]), a)
      @test within(prox(gw, [3tw / 2], a)[1], big4096(() -> big(3tw / 2) - big(tw)))
      @test prox(gw, [-tw], a)[1] === zero(T)
      # values whose leaf value overflows or vanishes under the factor
      @test (p2(T, -3E ÷ 5) * NormL1Nonneg(p2(T, 3E ÷ 5)))([p2(T, E ÷ 2)]) === p2(T, E ÷ 2)
      @test (p2(T, 3E ÷ 5) * NormL1Nonneg(p2(T, -3E ÷ 5)))([p2(T, m ÷ 2)]) === p2(T, m ÷ 2)
      @test (p2(T, -3E ÷ 5) * NormL1Nonneg(p2(T, 3E ÷ 5)))([p2(T, E ÷ 2), -one(T)]) === T(Inf)
      @test isnan((p2(T, -3E ÷ 5) * NormL1Nonneg(p2(T, 3E ÷ 5)))([p2(T, E ÷ 2), NaN]))
      # chains of nested factors whose product is not a normal number
      cc = Scaled(Scaled(NormL1Nonneg(p2(T, -E + 10)), a), a)
      @test cc.f isa Scaled && PB._chained(cc)
      xc = T[p2(T, E ÷ 2), p2(T, E ÷ 2)]
      lev = big4096(() -> big(p2(T, -E + 10)) * big(a)^2)
      rc = big4096(() -> lev * 2 * big(p2(T, E ÷ 2)))
      @test valok(cc(xc), rc, 2)
      @test cc(plant(xc, 2, -one(T))) === T(Inf) && isnan(cc(plant(xc, 2, T(NaN))))
      yc = prox(cc, [xc[1], -xc[2]], one(T))
      @test within(yc[1], big4096(() -> max(big(xc[1]) - lev, big(0)))) && yc[2] === zero(T)
      # array steps with a wide factor fold the factor into the levels
      g = Scaled(NormL1Nonneg(λ), a)
      @test prox(g, [v, -v], [a, a]) == prox(g, [v, -v], a)
      yv = similar([v, -v])
      @test within(value_and_prox!(yv, g, [v, -v], [a, a]), big4096(() -> big(a) * big(λ) * big(yv[1])), 2)
      # an infinite factor: the value is 0 at a point where f is 0, and the step Inf⋅0 = NaN is the wrapper's (#311)
      h = T(Inf) * NormL1Nonneg(T(1))
      @test h(zeros(T, 3)) === zero(T) && h(T[0, 1]) === T(Inf) && h(T[0, -1]) === T(Inf) && isnan(h(T[NaN, 0]))
      y2 = zeros(T, 2)
      @test value_and_prox!(y2, h, T[3, -2], one(T)) === zero(T) && y2 == T[0, 0]
      @test value_and_prox!(y2, h, T[3, -2], T[1, 1]) === zero(T)
      @test all(isnan, prox(h, T[3, -2], zero(T)))
      # the point-aware score under an infinite factor: the formula at a zero coordinate, an ArgumentError otherwise
      # (a negative coordinate included) and in the full-point form
      @test kkt_violation(h, T[0.5, 0.5], T[0, 0], 1) === -T(Inf)
      @test_throws ArgumentError kkt_violation(h, T[0.5, 0.5], T[0.5, 0], 1)
      @test_throws ArgumentError kkt_violation(h, T[0.5, 0.5], T[-0.5, 0], 1)
      @test_throws ArgumentError kkt_violation(h, T[0.5, 0.5], T[0, 0])
      # an infinite level is a hard constraint under factors and steps that leave the range: each coordinate against the
      # proximal map at the effective step αγ (+0 where the level fixes the entry; a zero weight leaves the projection)
      bh = p2(T, m ÷ 2 - 10)
      for hf in (NormL1Nonneg(T(Inf)), NormL1Nonneg(T(Inf), T[1, 0]), NormL1Nonneg(T(Inf), T[Inf, 0]))
        for (factor, step) in ((bh, bh), (p2(T, E ÷ 2 + 10), p2(T, E ÷ 2 + 10)), (T(2), T(3)), (bh, T(0)))
          got = prox(Scaled(hf, factor), T[3, 4], step)
          @test got[1] === zero(T)
          @test got[2] === (hf.w === nothing ? zero(T) : T(4) * (iszero(hf.w[2]) ? 1 : 0))
        end
        @test PB._has_infinite_level(Scaled(hf, bh)) && PB._hard(hf) === hf
      end
      @test PB._hard(NormL1Nonneg(T(1))) === nothing && !PB._has_infinite_level(NormL1Nonneg(T(1)))
      hv = Scaled(NormL1Nonneg(T(Inf)), bh)
      @test hv(zeros(T, 3)) === zero(T) && hv(T[0, 1]) === T(Inf) && hv(T[-0.0, 0]) === zero(T) && isnan(hv(T[NaN, 0]))
      # a zero factor keeps the domain and the fixed zeros; an infinite level has none
      z = 0 * NormL1Nonneg(T(1))
      @test z isa NormL1Nonneg && z.λ0 === zero(T) && z(T[1, 2]) === zero(T) && z(T[1, -1]) === T(Inf)
      z = 0 * NormL1Nonneg(1, [1, Inf])
      @test z isa NormL1Nonneg && z.λ0 === 0 && z.w == [1, Inf] && z([1, 0]) === 0.0 && z([0, 1]) === Inf && z([1, -1]) === Inf
      @test isequal(prox(z, [3.0, 4.0], 1.0), [3.0, 0.0])
      @test Scaled(NormL1Nonneg(T(2)), 0) isa NormL1Nonneg && Scaled(Scaled(NormL1Nonneg(T(2)), 3), 0) isa NormL1Nonneg
      @test_throws ArgumentError 0 * NormL1Nonneg(T(Inf))
      @test_throws ArgumentError Scaled(NormL1Nonneg(T(Inf)), 0)
      # unfoldable chains: the score of a zero coordinate from the coefficient aλw, without dividing v by a
      tiny = p2(T, T === Float32 ? -100 : -600)                              # tiny^2 is not a normal number
      lvl = p2(T, T === Float32 ? 100 : 1000)                                # tiny^2 lvl is
      chain(f) = (c = Scaled(Scaled(f, tiny), tiny); c.f isa Scaled || error("merged"); c)
      s = chain(NormL1Nonneg(T(1)))
      @test kkt_violation(s, T[0.25], 1) === -T(0.25) && kkt_violation(s, T[0.25], T[0], 1) === -T(0.25)
      @test kkt_violation(s, T[-0.25], 1) === T(0.25)
      for xx in (T[0.7], T[-0.7], T[NaN])
        @test_throws ArgumentError kkt_violation(s, T[0.25], xx, 1)
        @test_throws ArgumentError kkt_violation(s, T[0.25], xx)
      end
      @test_throws ArgumentError kkt_violation(s, T[0.25], T[0])
      sw = chain(NormL1Nonneg(T(1), T[0.5, 0, Inf]))
      @test kkt_violation(sw, T[0.25, 0.25, 0.25], T[0, 0, 0], 1) === -T(0.25) && kkt_violation(sw, T[0.25, 0.4, 0.25], T[0, 0, 0], 2) === -T(0.4)
      @test kkt_violation(sw, T[0.25, 0.25, 0.25], T[0, 0, 0], 3) === -T(Inf) && kkt_violation(sw, T[0.25, 0.25, 0.25], 3) === -T(Inf)
      @test kkt_violation(sw, T[0.25, 0.25, 0.25], T[0, 0, 0], 3) == kkt_violation(sw, T[0.25, 0.25, 0.25], 3)
      # the level folds: NormL1Nonneg(lvl) under tiny, tiny has the level tiny^2 lvl and the score of the unscaled function
      sf = chain(NormL1Nonneg(lvl))
      folded = NormL1Nonneg(p2(T, T === Float32 ? -100 : -200))
      @test kkt_violation(sf, T[0.25], 1) === kkt_violation(folded, T[0.25], 1) === -T(0.25)
      @test kkt_violation(sf, T[0.25], T[0.5], 1) === kkt_violation(folded, T[0.25], T[0.5], 1)
      @test kkt_violation(sf, T[0.25], T[-0.5], 1) === T(Inf)
      @test kkt_violation(sf, T[0.25], T[0.5]) === kkt_violation(folded, T[0.25], T[0.5])
      @test kkt_violation(sf, T[0.25, 0.1], T[0.5, 0]) === kkt_violation(folded, T[0.25, 0.1], T[0.5, 0])
    end
  end
end
end
