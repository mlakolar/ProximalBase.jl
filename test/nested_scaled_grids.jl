module NestedScaledGridsTest

# The Float32 passes of nested_scaled.jl that repeat a Float64 case which stays there: the conjugates of
# the nested scales (#333), five `check` calls of the steps outside the range (#566), and the explicit
# infinite factor under the fold (#566). The rest of each testset is in nested_scaled.jl. The fixtures
# and helpers are copied from that file.

using Test
using ProximalBase
using LinearAlgebra
using ..TestUtils

const PB = ProximalBase

# the wrappers from the outermost to the function inside
function factors(f)
  fs = Any[]
  while f isa Scaled
    push!(fs, f.α)
    f = f.f
  end
  fs, f
end
function effective(f, level)
  fs, leaf = factors(f)
  foldl(*, big.(fs); init = big(level(leaf)))
end

near(v::T, r; k = 8) where {T<:AbstractFloat} =
  v isa T && isfinite(v) && abs(big(v) - big(r)) <= k * eps(T) * abs(big(r))

# a power of ten in T, and the fixtures: the exponents of the two scales and of the level
tens(::Type{T}, k) where {T} = T(10)^k
fixtures(::Type{Float64}) = (a = 200, l = 300)
fixtures(::Type{Float32}) = (a = 20, l = 30)

@testset "nested scales (#333)" begin
  for T in (Float32,)
    @testset "$T" begin
      a, l = fixtures(T).a, fixtures(T).l
      small = Scaled(Scaled(NormL1(tens(T, l)), tens(T, -a)), tens(T, -a))   # level 10^(l - 2a)
      large = Scaled(Scaled(NormL1(tens(T, -l)), tens(T, a)), tens(T, a))    # level 10^(2a - l)
      λs, λl = effective(small, f -> f.λ0), effective(large, f -> f.λ0)

      @testset "conjugates" begin
        for (f, λ) in ((small, λs), (large, λl))
          c = Conjugate(f)
          r = T(λ)                                    # the dual box has radius λ
          # the value of an indicator carries the package's tolerance, which has an absolute part
          # for balls smaller than one: such a ball is probed far outside, and a large one at 1 %
          inside, outside = T[r * T(0.99)], T[r > 1 ? r * T(1.01) : one(T)]
          @test c(inside) === zero(T)
          @test c(outside) === T(Inf)
          @test c(T[-r * T(0.99)]) === zero(T)
          @test c(zeros(T, 1)) === zero(T)
          # projection onto the box [-λ, λ], at any step
          for γ in (one(T), T(0.25))
            y = zeros(T, 3)
            prox!(y, c, T[2r, -3r, r / 2], γ)
            @test y ≈ T[r, -r, r / 2] rtol = 8eps(T)
            y = T[2r, -3r, r / 2]
            prox!(y, c, y, γ)
            @test y ≈ T[r, -r, r / 2] rtol = 8eps(T)
          end
          y = zeros(T, 1)
          @test value_and_prox!(y, c, T[2r], one(T)) === zero(T)
          @test y ≈ T[r] rtol = 8eps(T)
          @test prox_block(c, T(2r), 1, one(T)) ≈ r rtol = 8eps(T)
        end
        # a Moreau route (a leaf without a direct projection) and a block sum
        q = Scaled(Scaled(SqrNormL2(tens(T, -l)), tens(T, a)), tens(T, a))
        λq = T(effective(q, f -> f.λ))
        yq = zeros(T, 1)
        prox!(yq, Conjugate(q), T[2λq], one(T))        # prox of f* at x: x - prox_{f}(x): f' = λ x
        @test yq ≈ T[2λq - 2λq / (1 + λq)] rtol = 64eps(T)
        sep = Scaled(Scaled(SeparableSum(NormL1(tens(T, -l)), NormL1(tens(T, -l))), tens(T, a)), tens(T, a))
        r = T(λl)
        z = (zeros(T, 1), zeros(T, 1))
        prox!(z, Conjugate(sep), (T[3r], T[-3r]), one(T))
        @test z[1] ≈ T[r] rtol = 8eps(T)
        @test z[2] ≈ T[-r] rtol = 8eps(T)
        # the value and the value at the proximal point, on blocks
        @test Conjugate(sep)((T[r / 2], T[-r / 2])) === zero(T)
        @test Conjugate(sep)((T[r * T(1.01)], T[0])) === T(Inf)
        z = (zeros(T, 1), zeros(T, 1))
        @test value_and_prox!(z, Conjugate(sep), (T[3r], T[-3r]), one(T)) === zero(T)
        @test z[1] ≈ T[r] rtol = 8eps(T)
        # a function without a level to fold: the conjugate value is refused, not wrong, and a
        # refused value_and_prox! leaves its output as it was
        @test_throws ArgumentError Conjugate(q)(T[1])
        y = T[-777]
        @test_throws ArgumentError value_and_prox!(y, Conjugate(q), T[1], one(T))
        @test y == T[-777]
        y = (T[-777], T[-777])
        @test_throws ArgumentError value_and_prox!(y, Conjugate(SeparableSum(q, q)), (T[1], T[1]), one(T))
        @test y == (T[-777], T[-777])
        # the chain may sit below ordinary wrappers: an ordinary outer scale, a translation, a
        # block of a sum, a singular-value wrapper; the output is still left as it was, in place too
        for g in (Scaled(Translate(q, T[0]), T(2)),
                  Scaled(Scaled(Translate(q, T[0]), T(2)), T(3)),
                  Translate(Scaled(q, T(2)), T[0]),
                  Scaled(Translate(Scaled(q, T(2)), T[0]), T(3)))
          @test PB._chained_scaled(g)
          y = T[-777]
          @test_throws ArgumentError value_and_prox!(y, Conjugate(g), T[1], one(T))
          @test y == T[-777]
          y = T[-777]
          @test_throws ArgumentError value_and_prox!(y, Conjugate(g), y, one(T))
          @test y == T[-777]
        end
        for g in (Scaled(SeparableSum(Translate(q, T[0]), SqrNormL2(one(T))), T(2)),
                  SeparableSum(Scaled(q, T(2)), SqrNormL2(one(T))),
                  SeparableSum(SqrNormL2(one(T)), Scaled(Translate(q, T[0]), T(2))))
          @test PB._chained_scaled(g)
          y = (T[-777], T[-777])
          @test_throws ArgumentError value_and_prox!(y, Conjugate(g), (T[1], T[1]), one(T))
          @test y == (T[-777], T[-777])
        end
        for g in (SingularValue(q), Scaled(SingularValue(q), T(2)))
          @test PB._chained_scaled(g)
          Y = fill(T(-777), 2, 2)
          @test_throws ArgumentError value_and_prox!(Y, Conjugate(g), Matrix{T}(I, 2, 2), one(T))
          @test Y == fill(T(-777), 2, 2)
        end
        # a chain that folds under ordinary wrappers still gives a value, and commits its point
        for (g, mult) in ((Scaled(Translate(large, T[0]), T(2)), 2),
                          (Scaled(Scaled(Translate(large, T[0]), T(2)), T(3)), 6))
          @test PB._chained_scaled(g)
          r = T(mult * λl)                              # the dual box of a multiple of the level
          y = T[-777]
          @test value_and_prox!(y, Conjugate(g), T[3r], one(T)) === zero(T)
          @test y ≈ T[r] rtol = 8eps(T)
          y = T[3r]
          @test value_and_prox!(y, Conjugate(g), y, one(T)) === zero(T)
          @test y ≈ T[r] rtol = 8eps(T)
        end
        # ordinary wrappers are not staged, and cost nothing extra
        for g in (Scaled(Translate(SqrNormL2(one(T)), T[0]), T(2)), Scaled(Scaled(SqrNormL2(one(T)), T(2)), T(3)),
                  SeparableSum(Scaled(SqrNormL2(one(T)), T(2)), SqrNormL2(one(T))), SingularValue(SqrNormL2(one(T))))
          @test !PB._chained_scaled(g)
        end
      end
    end
  end
end

# Steps that the factors take out of the range (#566): a single factor with a scalar or array step, and
# a chain with an array step. The references are the proximal points with the coefficient
# (product of the stored factors) * level * weight * step formed in BigFloat.
function refprox(f, x, γ)
  fs, leaf = factors(f)
  leaf isa Translate && (leaf = leaf.f)       # the shifts in these tests are zero
  a = foldl(*, big.(fs); init = big(1))
  stepof(i) = big(γ isa AbstractArray ? γ[i] : γ)
  w(i) = leaf.w === nothing ? big(1) : big(leaf.w[i])
  soft(v, t) = sign(v) * max(abs(v) - t, 0)
  if leaf isa NormL1
    [soft(big(x[i]), a * big(leaf.λ0) * w(i) * stepof(i)) for i in eachindex(x)]
  elseif leaf isa ElasticNet
    [soft(big(x[i]), a * big(leaf.λ1) * w(i) * stepof(i)) / (1 + a * big(leaf.λ2) * stepof(i)) for i in eachindex(x)]
  elseif leaf isa NormL2
    t, n = a * big(leaf.λ) * stepof(1), sqrt(sum(abs2, big.(x)))
    [n <= t ? big(0) : big(x[i]) * (1 - t / n) for i in eachindex(x)]
  else  # GroupNormL2, with a step per group
    y = zeros(BigFloat, length(x))
    for (k, g) in enumerate(leaf.groups)
      t, n = a * big(leaf.λ0) * big(leaf.w[k]) * stepof(k), sqrt(sum(abs2, big.(x[g])))
      for i in g
        y[i] = n <= t ? big(0) : big(x[i]) * (1 - t / n)
      end
    end
    y
  end
end
# the entries within 8 eps of the reference (an exact zero is zero)
close(y, r) = all(i -> near(y[i], r[i]) || (iszero(r[i]) && iszero(y[i])), eachindex(y))
# a value within 8 eps of the reference, or lost when the reference is below the range
valnear(v::T, r) where {T} = r < floatmin(T) ? abs(big(v) - r) <= floatmin(T) : near(v, r)

# the exponents of the factor a, of the level l and of the step g, so that the coefficient is in the range
# while the product of the factor and the step is not
# l2 and g2: a level with which the factors have a product below the range, and a large step that
# brings the coefficient back into it
steps566(::Type{Float64}) = (a = 200, l = 300, g = 50, l2 = 80, g2 = 80)
steps566(::Type{Float32}) = (a = 25, l = 30, g = 10, l2 = 10, g2 = 20)

@testset "steps outside the range of the factors (#566)" begin
  for T in (Float32,)
    @testset "$T" begin
      a, l, g = steps566(T).a, steps566(T).l, steps566(T).g
      # exercise one function with the scalar step and the array step: the point, the aliased point and
      # the value that comes with it
      function check(f, x, γ, label; array = true)
        r = refprox(f, x, γ)
        @testset "$label" begin
          n = length(x)
          for step in (γ, γ isa AbstractArray || !array ? nothing : fill(γ, n))
            step === nothing && continue
            y = zeros(T, n)
            prox!(y, f, x, step)
            @test close(y, r)
            @test all(isfinite, y)
            y = copy(x)
            prox!(y, f, y, step)
            @test close(y, r)
            y = zeros(T, n)
            value_and_prox!(y, f, x, step)
            @test close(y, r)
            y = copy(x)
            value_and_prox!(y, f, y, step)
            @test close(y, r)
          end
          γ isa AbstractArray || @test close(prox(f, x, γ), r)
          γ isa AbstractArray || !array || for k in 1:n
            @test near(prox_block(f, x[k], k, γ), r[k]) || (iszero(r[k]) && iszero(prox_block(f, x[k], k, γ)))
          end
        end
      end

      θ = T(10)^(l - 2a)
      x = T[3, -3, 0.5] .* θ
      tr = Scaled(Translate(NormL1(tens(T, l)), T[0, 0, 0]), tens(T, -a))
      check(tr, x, fill(tens(T, -a), 3), "single factor, Translate")
      θc = T(10)^(l - 2a - g)
      grp = Scaled(Scaled(GroupNormL2(tens(T, l), [1:2, 3:3], T[1, 1]), tens(T, -a)), tens(T, -a))
      xg = T[3, 4, 0.5] .* θc
      check(grp, xg, fill(tens(T, -g), 2), "chain, GroupNormL2 with a step per group")
      grp1 = Scaled(GroupNormL2(tens(T, l), [1:2, 3:3], T[1, 1]), tens(T, -a))
      check(grp1, T[3, 4, 0.5] .* θ, fill(tens(T, -a), 2), "single factor, GroupNormL2 with a step per group")
      # a function with a level but no array-step proximal operator takes the scalar route
      check(Scaled(NormL2(tens(T, l)), tens(T, -a)), T[3, 4] .* θ, tens(T, -a), "single factor, NormL2"; array = false)

      @testset "the level and the factors have no product in range" begin
        l2, g2 = steps566(T).l2, steps566(T).g2
        θ2 = T(10)^(l2 - 2a + g2)
        check(Scaled(Scaled(GroupNormL2(tens(T, l2), [1:2, 3:3], T[1, 1]), tens(T, -a)), tens(T, -a)), T[3, 4, 0.5] .* θ2, fill(tens(T, g2), 2), "chain, GroupNormL2 with a step per group")
      end
    end
  end
end

# Review of #575: the fold that refuses a level (empty array steps) and the explicit infinite factor
# (a factor that is not a scale to fold). Both gave a result on the base of #566 and must still.
@testset "empty array steps and infinite factors under the fold (#566)" begin
  for T in (Float32,)
    @testset "$T" begin
      tiny = T == Float64 ? T(1e-200) : T(1f-30)   # outside the band of a single factor
      @testset "an explicit infinite factor" begin
        inner = Scaled(NormL1(one(T)), T(Inf))
        z, γ = T[3, 0.5, -7], T[1, 1, 1]
        for s in (Scaled(Translate(inner, zeros(T, 3)), tiny),
                  Scaled(Scaled(Translate(inner, zeros(T, 3)), tiny), tiny),
                  Scaled(Scaled(NormL1(one(T)), T(Inf)), tiny),
                  Scaled(Scaled(NormL1(one(T)), tiny), T(Inf)),
                  Scaled(Scaled(Scaled(NormL1(one(T)), T(Inf)), tiny), tiny))
          # the point of the scalar step, which the base of #566 gives for both
          y, ys = fill(T(-7), 3), fill(T(-7), 3)
          @test prox!(y, s, z, γ) === y
          prox!(ys, s, z, T(1))
          @test isequal(y, ys)
          @test all(iszero, y) || all(isnan, y)
          y2 = fill(T(-7), 3)
          v = value_and_prox!(y2, s, z, γ)
          @test isequal(y2, y)
          @test isequal(v, value_and_prox!(copy(ys), s, z, T(1)))
        end
        # an ordinary factor over the infinite one, and a scalar step, as controls
        ordinary = Scaled(Translate(inner, zeros(T, 3)), one(T))
        @test prox(ordinary, z, γ) == prox(ordinary, z, T(1)) == zeros(T, 3)
      end
    end
  end
end

end
