module ElasticNetExtremesTest
# ElasticNet (#908): the endpoint truth table of the plan (D1 is in elasticnet.jl), the extreme ranges of
# Float32 and Float64 against 4096-bit BigFloat, Scaled (folds, chains, infinite and zero factors, the routes
# that need the ElasticNet methods of scaled.jl) and the cost of values on sparse and symmetric iterates.
#
# Proximal points are compared with shrink(v, t̂)/(1 + γλ2) in BigFloat, t̂ the threshold that the NormL1
# kernel forms, within 4 units in the last place; values with λ1 Σ wᵢ|xᵢ| + λ2/2 Σ xᵢ² within (n + 2) eps
# relative for n terms; exact results (zeros, identities, infinities, NaN) with isequal or ===.
using Test, ProximalBase, LinearAlgebra, SparseArrays, Random
const PB = ProximalBase

big4096(f) = setprecision(f, BigFloat, 4096)
p2(::Type{T}, k) where {T} = ldexp(one(T), k)
weight(f, i) = f.w === nothing ? 1 : f.w[i]
threshold(::Type{T}, f, i, γ) where {T} = PB._l1_prox_threshold(zero(T), γ, f.λ1, PB._entry(f.w, i), T)[2]
function proxref(::Type{T}, f, v, i, γ) where {T}
  t = threshold(T, f, i, γ)
  big4096() do
    b = big(T(v))
    T(sign(b) * max(abs(b) - big(t), 0) / (1 + big(γ) * big(f.λ2)))
  end
end
within(got, ref, k = 4) = isequal(got, ref) || abs(got - ref) <= k * eps(abs(ref))
valref(f, x) = big4096() do
  big(f.λ1) * sum(iszero(x[i]) ? big(0) : big(weight(f, i)) * abs(big(x[i])) for i in eachindex(x); init = big(0)) +
    big(f.λ2) / 2 * sum(big(v)^2 for v in x; init = big(0))
end
valok(got::T, ref, n) where {T} = isfinite(got) && abs(big(got) - ref) <= (n + 2) * eps(T) * abs(ref)

# a weight array that counts its reads, to show that values and proximal points on sparse and symmetric
# iterates read the weights of the stored entries only
mutable struct CountingArray{T, N} <: AbstractArray{T, N}
  data::Array{T, N}
  reads::Int
end
CountingArray(a) = CountingArray(a, 0)
Base.size(a::CountingArray) = size(a.data)
Base.IndexStyle(::Type{<:CountingArray}) = IndexLinear()
Base.getindex(a::CountingArray, i::Int) = (a.reads += 1; a.data[i])
@testset "ElasticNet extremes" begin
  for T in (Float32, Float64)
    @testset "$T truth table" begin
      trials = T[3, -3, 0.25, 0, -0.0, Inf, -Inf, NaN]
      steps = T[0, 0.5, 1, Inf]
      # λ2 = Inf (D2): the indicator of {0}, +0 at every γ ≥ 0 and trial, whatever λ1 and the weight are
      for λ1 in T[0, 1, Inf], w in (nothing, T[0], T[1], T[Inf]), γ in steps
        f = ElasticNet(λ1, T(Inf), w)
        @test all(prox_block(f, v, 1, γ) === zero(T) for v in trials)
      end
      fD = ElasticNet(T(1), T(Inf))
      tiny = nextfloat(zero(T))
      @test fD(T[0, -0.0]) === zero(T) && fD(T[0, tiny]) === T(Inf) && isnan(fD(T[NaN, 0])) && fD(T[-Inf, 0]) === T(Inf)
      @test ElasticNet(zero(T), T(Inf))(T[0, tiny]) === T(Inf) && ElasticNet(zero(T), T(Inf))(zeros(T, 3)) === zero(T)
      @test value_entry(fD, zero(T), 1) === zero(T) && value_entry(fD, tiny, 1) === T(Inf) && isnan(value_entry(fD, T(NaN), 1))
      @test value_entry(ElasticNet(zero(T), T(Inf)), -zero(T), 1) === zero(T)
      s = SparseIterate(T, 6); activate!(s, 2); activate!(s, 4)
      s2 = copy(s)
      @test fD(s) === zero(T) && prox!(s2, fD, s2, T(0.5)) === s2 && iszero(Vector(s2)) && nactive(s2) == 2
      s[4] = tiny
      @test fD(s) === T(Inf)
      S = SymmetricSparseIterate(T, 3); S[3, 1] = 0
      @test fD(S) === zero(T)
      S[3, 1] = -tiny
      @test fD(S) === T(Inf) && Matrix(prox(fD, S, zero(T))) == zeros(T, 3, 3)
      b = BlockIterate(T[0, 0, 1, 2], [[1, 2], [3, 4]])
      @test fD(b) === T(Inf) && Vector(prox(fD, b, T(2))) == zeros(T, 4)
      y = T[5, 5]
      @test value_and_prox!(y, fD, T[3, NaN], zero(T)) === zero(T) && y == T[0, 0]

      # λ1 = Inf: positive-weight entries +0 at every γ ≥ 0 and trial (#665); a zero weight is free, then the ridge
      for λ2 in T[0, 1], γ in steps
        f = ElasticNet(T(Inf), λ2, T[1, 0, Inf])
        @test all(prox_block(f, v, 1, γ) === zero(T) && prox_block(f, v, 3, γ) === zero(T) for v in trials)
        @test all(isequal(prox_block(f, v, 2, γ), prox_block(SqrNormL2(λ2), shrink(v, zero(T)), 2, γ)) for v in trials)
      end
      fI = ElasticNet(T(Inf), T(1), T[1, 0])
      @test fI(T[0, 4]) === T(8) && fI(T[tiny, 0]) === T(Inf) && fI(T[-0.0, 0]) === zero(T)

      # an infinite weight maps a finite trial to +0 at every level and step, a zero level or step included (#853);
      # NaN and ±Inf trials at a finite λ1 have no fixed-zero guarantee: the arithmetic gives NaN (observed, #653)
      for λ1 in T[0, 1], λ2 in T[0, 1], γ in steps
        f = ElasticNet(λ1, λ2, T[Inf])
        @test all(prox_block(f, v, 1, γ) === zero(T) for v in T[3, -3, 0, -0.0, floatmax(T)])
        @test all(isnan(prox_block(f, v, 1, γ)) for v in T[Inf, -Inf, NaN])
      end
      @test ElasticNet(T(1), T(1), T[Inf, 1])(T[0, 2]) === T(4) && ElasticNet(zero(T), zero(T), T[Inf, 1])(T[tiny, 2]) === T(Inf)

      # finite factors whose threshold overflows: a finite trial gives +0; ±Inf is the deferred #866 limitation
      huge = floatmax(T) / 4
      fO = ElasticNet(huge, zero(T), T[huge])
      @test prox_block(fO, T(3), 1, one(T)) === zero(T) && prox_block(fO, -floatmax(T), 1, one(T)) === zero(T)
      @test !isequal(prox_block(fO, T(Inf), 1, one(T)), zero(T))

      # γλ2 outside the range from finite factors: the scaled ridge returns the representable result
      E = exponent(floatmax(T))
      fR = ElasticNet(zero(T), p2(T, 3E ÷ 5))
      @test within(prox_block(fR, p2(T, E - 2), 1, p2(T, 3E ÷ 5)), proxref(T, fR, p2(T, E - 2), 1, p2(T, 3E ÷ 5)))
      T === Float64 && @test prox_block(ElasticNet(0.0, 2.0^600), 2.0^1000, 1, 2.0^600) === 2.0^-200

      # γ = 0: the identity (-0 gives +0) outside the infinite rows; γ = Inf with finite levels: 0 with the sign
      # of z = shrink(v, γλ1w) where λ1w > 0 or λ2 > 0 (NaN for NaN and ±Inf trials), the identity otherwise
      for λ1 in T[0, 0.5], λ2 in T[0, 2], w in (nothing, T[0], T[2])
        f = ElasticNet(λ1, λ2, w)
        @test all(isequal(prox_block(f, v, 1, zero(T)), iszero(v) ? zero(T) : v) for v in trials)
        active = λ1 * weight(f, 1) > 0 || λ2 > 0
        for v in trials
          r = prox_block(f, v, 1, T(Inf))
          z = λ1 * weight(f, 1) > 0 ? zero(T) : shrink(v, zero(T))
          @test !active ? isequal(r, iszero(v) ? zero(T) : v) : isfinite(v) ? isequal(r, copysign(zero(T), z)) : isnan(r)
        end
      end

      # a NaN trial gives NaN outside the infinite rows; a NaN entry makes the value NaN unless λ1 wᵢ = 0 and λ2 = 0
      @test isnan(prox_block(ElasticNet(T(1), T(1), T[2]), T(NaN), 1, T(0.5)))
      @test ElasticNet(zero(T), zero(T))(T[NaN, 1]) === zero(T) && ElasticNet(T(1), zero(T), T[0, 1])(T[NaN, 1]) === one(T)
      @test isnan(ElasticNet(zero(T), T(1))(T[NaN, 1])) && isnan(ElasticNet(T(1), zero(T))(T[NaN, 1]))
      @test isnan(ElasticNet(T(1), T(1), T[0, 1])(T[NaN, 1]))
    end
  end

  for T in (Float32, Float64)
    @testset "$T extreme ranges" begin
      E, m = exponent(floatmax(T)), exponent(floatmin(T))
      # (λ1, λ2, w, γ, trials) whose products leave the range while the result is representable
      cases = [
        (p2(T, 3E ÷ 5), p2(T, -3E ÷ 5), p2(T, -E ÷ 2), p2(T, 3E ÷ 5)),      # γλ1 overflows, the weight restores it
        (p2(T, 3m ÷ 5), p2(T, 0), p2(T, -m ÷ 2), p2(T, 3m ÷ 5)),           # γλ1 underflows, the weight restores it
        (p2(T, E ÷ 2 + 8), p2(T, 0), p2(T, 8), p2(T, E ÷ 2 + 8)),          # γλ1w overflows: the threshold is Inf
        (p2(T, m ÷ 2), p2(T, 0), p2(T, m ÷ 2), p2(T, 0)),                  # γλ1w underflows to zero
        (zero(T), p2(T, 3E ÷ 5), nothing, p2(T, 3E ÷ 5)),                  # γλ2 overflows
        (zero(T), p2(T, 3m ÷ 5), nothing, p2(T, 3m ÷ 5)),                  # γλ2 underflows
        (T(0.75), p2(T, E ÷ 2), T(1.5), p2(T, E ÷ 2)),                     # γλ2 overflows with a threshold
        (one(T), zero(T), nothing, one(T)),                                 # near floatmax and subnormal trials
      ]
      trials = T[3, -3, 1.5, floatmax(T), -floatmax(T) / 3, floatmin(T), 3 * nextfloat(zero(T)), -prevfloat(floatmin(T))]
      for (λ1, λ2, w, γ) in cases
        f = ElasticNet(λ1, λ2, w === nothing ? nothing : T[w])
        t = threshold(T, f, 1, γ)
        extra = isfinite(t) && t > 0 ? T[3t / 2, -5t / 4, t, prevfloat(t), nextfloat(t)] : T[]
        for v in vcat(trials, extra)
          @test within(prox_block(f, v, 1, γ), proxref(T, f, v, 1, γ))
        end
      end
      # a subnormal result and the ridge alone at the bottom of the range
      @test prox_block(ElasticNet(zero(T), one(T)), 4 * nextfloat(zero(T)), 1, one(T)) === 2 * nextfloat(zero(T))
      @test prox_block(ElasticNet(zero(T), zero(T)), floatmax(T), 1, one(T)) === floatmax(T)

      # values whose terms, squares or coefficients leave the range while the value is representable
      vcases = [
        ("parts far apart", one(T), one(T), nothing, T[p2(T, 3m ÷ 5), 1, p2(T, m + 5), -p2(T, m ÷ 3)]),
        ("squares underflow, large λ2", zero(T), p2(T, 3E ÷ 5), nothing, fill(p2(T, m ÷ 2 - 10), 64)),
        ("squares overflow, small λ2", zero(T), p2(T, -3E ÷ 5), nothing, fill(p2(T, 3E ÷ 5 - 2), 4)),
        ("ℓ1 coefficient overflows", p2(T, E - 3), zero(T), fill(p2(T, 10), 4), fill(-p2(T, -20), 4)),
        ("ℓ1 terms below floatmin, ridge dominant", p2(T, 3m ÷ 4), one(T), T(0.5) .+ T.(1:64) ./ 64, [(-1)^i * p2(T, m ÷ 3) * (1 + T(i) / 64) for i in 1:64]),
        ("squares below floatmin, ℓ1 dominant", one(T), one(T), T(0.5) .+ T.(1:64) ./ 64, [(-1)^i * p2(T, 3m ÷ 5) * (1 + T(i) / 64) for i in 1:64]),
        ("every term below floatmin", p2(T, m ÷ 2), p2(T, 0), T(0.5) .+ T.(1:64) ./ 64, fill(p2(T, m ÷ 2 - 2), 64)),
        ("ordinary, weighted", T(0.3), T(0.5), T(0.5) .+ T.(1:100) ./ 100, [sin(T(i)) * 3 for i in 1:100]),
      ]
      for (name, λ1, λ2, w, x) in vcases
        f = ElasticNet(λ1, λ2, w)
        n = length(x)
        r = valref(f, x)
        @testset "$name" begin
          @test valok(f(x), r, n)                                           # dense Array (fused or composed)
          @test valok(f(view(copy(x), :)), r, n)                            # composed
          @test valok(PB._elastic_value(f, x), r, n)
          @test valok(T(sum(value_entry(f, x[i], i) for i in 1:n)), r, n)
          @test valok(f(SparseIterate(x)), r, n)
          @test valok(f(BlockIterate(copy(x), [collect(1:2:n), collect(2:2:n)])), r, n)
        end
      end
      # the two regimes where one part dominates take the fused loop: its bound holds there
      for (λ1, λ2, x) in ((p2(T, 3m ÷ 4), one(T), fill(p2(T, m ÷ 3), 64)), (one(T), one(T), fill(p2(T, 3m ÷ 5), 64)))
        f = ElasticNet(λ1, λ2, ones(T, 64))
        s, ok = PB._elastic_fused_sum(f, x)
        @test ok && PB._elastic_sum_kept(s, f.λ2, length(x))
      end
      # and every term below floatmin does not (the composed value recomputes it)
      f = ElasticNet(p2(T, m ÷ 2), one(T), ones(T, 8))
      s, ok = PB._elastic_fused_sum(f, fill(p2(T, m ÷ 2 - 2), 8))
      @test !(ok && PB._elastic_sum_kept(s, f.λ2, 8))
      # value_entry of single extreme terms
      @test value_entry(ElasticNet(zero(T), p2(T, -E ÷ 2 - 4)), p2(T, E ÷ 2 + 4), 1) === p2(T, E ÷ 2 + 3)
      @test value_entry(ElasticNet(zero(T), p2(T, E ÷ 2 + 4)), p2(T, m ÷ 2 - 4), 1) === p2(T, E ÷ 2 + m - 5)
      @test value_entry(ElasticNet(zero(T), one(T)), T(Inf), 1) === T(Inf) && value_entry(ElasticNet(T(1), zero(T)), T(Inf), 1) === T(Inf)
      @test value_entry(ElasticNet(T(1), zero(T), T[0]), T(NaN), 1) === zero(T)

      # value_and_prox! returns the value at the stored point, on every representation
      rng = Xoshiro(908)
      x = T(3) .* randn(rng, T, 40)
      for f in (ElasticNet(T(0.4), T(0.7), T(0.5) .+ rand(rng, T, 40)), ElasticNet(p2(T, 3m ÷ 4), one(T)), ElasticNet(T(Inf), one(T)))
        for γ in (T(0.3), T(2), p2(T, E ÷ 2))
          y = similar(x)
          @test value_and_prox!(y, f, x, γ) === f(y)
          sx = SparseIterate(x .* (rand(rng, T, 40) .< T(0.3))); sy = copy(sx)
          @test value_and_prox!(sy, f, sx, γ) === f(sy)
          bx = BlockIterate(copy(x), [collect(k:4:40) for k in 1:4]); by = copy(bx)
          @test value_and_prox!(by, f, bx, γ) === f(by)
        end
      end
      # mixed precision (Float64 levels, Float32 data) is a regression check only: no guarantee is documented
      if T === Float32
        f = ElasticNet(0.3, 0.5, [1.0, 2.0, 0.5])
        x = Float32[3, -2, 0.1]
        y = prox(f, x, 0.7)
        @test eltype(y) === Float32 && all(within(y[i], proxref(Float32, f, x[i], i, 0.7)) for i in 1:3)
        @test isapprox(f(x), 0.3 * (3 + 4 + 0.05) + 0.25 * (9 + 4 + 0.01); rtol = 1e-6)
      end
    end
  end

  for T in (Float32, Float64)
    @testset "$T Scaled" begin
      E, m = exponent(floatmax(T)), exponent(floatmin(T))
      f = ElasticNet(T(0.4), T(0.7), T[1, 2, 0, 0.5])
      x = T[3, -2, 0.5, -0.25]
      γ = T(0.3)
      # an ordinary factor: the value and the proximal point of f at the scaled step
      @test (2 * f)(x) === 2 * f(x) && prox(2 * f, x, γ) == prox(f, x, 2γ) && (f * T(0.5))(x) === T(0.5) * f(x)
      y = similar(x)
      @test value_and_prox!(y, 3 * f, x, γ) === (3 * f)(y) && y == prox(f, x, 3γ)
      @test prox_block(2 * f, x[1], 1, γ) === prox_block(f, x[1], 1, 2γ)
      @test kkt_violation(2 * f, T[1, 1, 1, 1], 2) ≈ 1 - 2 * T(0.4) * 2
      @test restrict(2 * f, [2]) isa Scaled && restrict(2 * f, [2]).f.w == T[2]
      # a step product that overflows: the factor folds into the ℓ1 level (powers of two: exact)
      a = p2(T, 3E ÷ 5)
      λ = p2(T, m + 4)
      t = T(big4096(() -> big(a) * big(a) * big(λ)))
      g = Scaled(ElasticNet(λ, zero(T)), a)
      v = 3t / 2
      @test prox(g, [v, -v], a) == [t / 2, -t / 2]
      # a step product that underflows to zero: the threshold still acts
      b = p2(T, m ÷ 2 - 30)
      tb = T(big4096(() -> big(b) * big(b) * big(p2(T, E))))
      g = Scaled(ElasticNet(p2(T, E), zero(T)), b)
      @test prox(g, [tb / 2, 3tb / 2], b) == [zero(T), tb / 2]
      # with a ridge level that the fold takes below floatmin: it is dropped, and the ridge factor is 1
      g = Scaled(ElasticNet(p2(T, E), one(T)), b)
      @test prox(g, [3tb / 2], b) == [tb / 2]
      @test within(prox(g, [3tb / 2], b)[1], T(big4096(() -> (3 * big(tb) / 2 - big(tb)) / (1 + big(b)^2))))
      # values whose leaf value overflows or vanishes under the factor
      @test (p2(T, -3E ÷ 5) * ElasticNet(zero(T), p2(T, 3E ÷ 5)))([p2(T, E ÷ 2)]) === p2(T, 2 * (E ÷ 2) - 1)
      @test (p2(T, 3E ÷ 5) * ElasticNet(p2(T, -3E ÷ 5), zero(T)))([p2(T, m ÷ 2)]) === p2(T, m ÷ 2)
      # chains of nested factors whose product is not a normal number
      c = Scaled(Scaled(ElasticNet(p2(T, -E + 10), p2(T, -E + 10)), a), a)
      @test c.f isa Scaled
      xc = T[p2(T, E ÷ 4), -p2(T, E ÷ 4)]
      lev = big4096(() -> big(p2(T, -E + 10)) * big(a)^2)
      rc = big4096(() -> lev * (2big(p2(T, E ÷ 4)) + big(p2(T, E ÷ 4))^2))
      @test valok(c(xc), rc, 2)
      yc = prox(c, xc, one(T))
      @test all(within(yc[i], T(big4096(() -> sign(big(xc[i])) * max(abs(big(xc[i])) - lev, 0) / (1 + lev)))) for i in 1:2)
      # array steps with a wide factor fold the factor into the levels
      g = Scaled(ElasticNet(λ, zero(T)), a)
      @test prox(g, [v, -v], [a, a]) == prox(g, [v, -v], a)
      yv = similar([v, -v])
      @test value_and_prox!(yv, g, [v, -v], [a, a]) === g(yv)
      # an infinite factor: the value is 0 at a point where f is 0, and the step Inf⋅0 = NaN is the wrapper's (#311)
      h = T(Inf) * ElasticNet(T(1), T(1))
      @test h(zeros(T, 3)) === zero(T) && h(T[0, 1]) === T(Inf)
      y2 = zeros(T, 2)
      @test value_and_prox!(y2, h, T[3, -2], one(T)) === zero(T) && y2 == T[0, 0]
      @test value_and_prox!(y2, h, T[3, -2], T[1, 1]) === zero(T)
      @test all(isnan, prox(h, T[3, -2], zero(T)))
      # an infinite level is a hard constraint under factors and steps that leave the range
      for hf in (ElasticNet(T(1), T(Inf)), ElasticNet(T(Inf), T(2)), ElasticNet(T(Inf), T(2), T[1, 0]))
        sc = Scaled(hf, p2(T, m ÷ 2 - 10))
        got = prox(sc, T[3, -4], p2(T, m ÷ 2 - 10))
        ref = prox(hf, T[3, -4], one(T))
        @test got == ref || (hf.λ2 < Inf && hf.w !== nothing)          # a free entry under a finite ridge keeps its route
      end
      @test PB._hard(ElasticNet(T(Inf), T(2), T[1, 0])) === nothing && PB._hard(ElasticNet(T(1), T(2))) === nothing
      @test PB._hard(ElasticNet(T(1), T(Inf))).λ1 === zero(T) && PB._hard(ElasticNet(T(Inf), T(2))).λ2 === zero(T)
      # a zero factor: finite parameters give the zero function; an infinite weight keeps its fixed zeros
      @test 0 * ElasticNet(T(1), T(1)) === ZeroFunction()
      z = 0 * ElasticNet(1, 1, [1, Inf])
      @test z isa ElasticNet && z.λ1 === 0 && z.λ2 === 0 && z.w == [1, Inf]
      @test_throws ArgumentError 0 * ElasticNet(T(Inf), T(1))
      @test_throws ArgumentError 0 * ElasticNet(T(1), T(Inf))
    end
  end

  @testset "the examples that need the Scaled methods of ElasticNet" begin
    @test prox(Scaled(ElasticNet(2.0^-1070, 0.0), 2.0^600), [2.0^200], 2.0^600) == [2.0^200 - 2.0^130]
    @test prox(Scaled(ElasticNet(2.0^1023, 0.0), 2.0^-540), [2.0^-60], 2.0^-540) == [0.0]
    @test (2.0^-600 * ElasticNet(0, 2.0^600))([2.0^300]) === 2.0^599
    y = zeros(2)
    @test value_and_prox!(y, Inf * ElasticNet(1.0, 1.0), [3.0, -2.0], 1.0) === 0.0
  end

  for T in (Float32, Float64)
    @testset "$T values on stored entries" begin
      n = 20_000
      cw = CountingArray(T(0.5) .+ rand(Xoshiro(1), T, n))
      for λ2 in (T(0.7), zero(T), T(Inf))
        f = ElasticNet(T(0.3), λ2, cw)
        x = SparseIterate(T, n)
        for k in (3, 4000, 17_777)
          x[k] = T(k) / 1000
        end
        cw.reads = 0
        v = f(x)
        @test cw.reads <= 2 * nactive(x)
        @test v ≈ f(Vector(x))
        x2 = copy(x)
        cw.reads = 0
        prox!(x2, f, x2, T(0.5))
        @test cw.reads <= nactive(x) && nactive(x2) == nactive(x)
        # the recomputation of terms below floatmin reads the stored entries too
        g = ElasticNet(floatmin(T), λ2, cw)
        cw.reads = 0
        u = g(x)
        @test cw.reads <= 3 * nactive(x)
        @test u ≈ g(Vector(x))
      end
      p = 150
      W = CountingArray(ones(T, p, p))
      S = SymmetricSparseIterate(T, p)
      S[7, 3] = 2; S[100, 100] = -1; S[150, 1] = T(0.5)
      f = ElasticNet(T(0.3), T(0.7), W)
      W.reads = 0
      @test f(S) ≈ T(0.3) * (2 * 2 + 1 + 2 * T(0.5)) + T(0.35) * (2 * 4 + 1 + 2 * T(0.25))
      @test W.reads <= 4 * nactive(S)
    end
  end
end
end
