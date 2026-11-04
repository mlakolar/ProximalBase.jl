module NestedScaledTest

# Nested `Scaled` factors whose product is not a normal number (#333). The references multiply the
# stored factors in BigFloat, so they never see the rounded product.

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
  for T in (Float32, Float64)
    @testset "$T" begin
      a, l = fixtures(T).a, fixtures(T).l
      small = Scaled(Scaled(NormL1(tens(T, l)), tens(T, -a)), tens(T, -a))   # level 10^(l - 2a)
      large = Scaled(Scaled(NormL1(tens(T, -l)), tens(T, a)), tens(T, a))    # level 10^(2a - l)
      λs, λl = effective(small, f -> f.λ0), effective(large, f -> f.λ0)
      @test 0.9 * 10.0^(l - 2a) < λs < 1.1 * 10.0^(l - 2a)

      @testset "constructor" begin
        for f in (small, large)
          @test !(f isa ZeroFunction)
          @test f isa Scaled{<:Scaled}
          @test f.f isa Scaled{<:NormL1}
        end
        # a subnormal product
        sub = T === Float64 ? T(1e-160) : T(1e-20)
        @test sub * sub < floatmin(T) && !(Scaled(Scaled(NormL1(one(T)), sub), sub) isa ZeroFunction)
        @test Scaled(Scaled(NormL1(one(T)), sub), sub) isa Scaled{<:Scaled}
        # three factors: a later factor that brings the product back is merged
        t = Scaled(Scaled(Scaled(NormL1(tens(T, l)), tens(T, -a)), tens(T, -a)), tens(T, a))
        @test t isa Scaled{<:NormL1} && near(t.α, tens(T, -a)) && t.f.λ0 == tens(T, l)
        t = Scaled(Scaled(Scaled(NormL1(tens(T, -l)), tens(T, a)), tens(T, a)), tens(T, -a))
        @test t isa Scaled{<:NormL1} && near(t.α, tens(T, a))
        # three factors that all stay nested
        t3 = Scaled(Scaled(Scaled(NormL1(tens(T, l)), tens(T, -a)), tens(T, -a)), tens(T, -a))
        @test t3 isa Scaled{<:Scaled{<:Scaled{<:NormL1}}}
        @test near(t3([one(T)]), effective(t3, f -> f.λ0))
        y3 = zeros(T, 1)
        prox!(y3, t3, T[3], T(effective(t3, f -> f.λ0)^-1))
        @test y3 ≈ T[2] rtol = 8eps(T)
        # exact zero, explicit Inf, ordinary products and invalid scales keep their meaning
        @test Scaled(Scaled(NormL1(one(T)), tens(T, -a)), zero(T)) isa ZeroFunction
        @test Scaled(Scaled(NormL1(one(T)), tens(T, -a)), tens(T, -a)) isa Scaled{<:Scaled}
        i = Scaled(Scaled(NormL1(one(T)), tens(T, a)), T(Inf))
        @test i isa Scaled{<:NormL1} && isinf(i.α)
        o = Scaled(Scaled(NormL1(one(T)), T(2)), T(3))
        @test o isa Scaled{<:NormL1} && o.α == T(6)
        @test_throws ArgumentError Scaled(Scaled(NormL1(one(T)), tens(T, a)), T(NaN))
        @test_throws ArgumentError Scaled(Scaled(NormL1(one(T)), tens(T, a)), -one(T))
        @test_throws ArgumentError Scaled(small, -one(T))
        @test_throws ArgumentError Scaled(Scaled(NormL1(one(T)), tens(T, -a)), -tens(T, -a))
        ind = Scaled(Scaled(IndBox(zero(T), one(T)), tens(T, -a)), tens(T, -a))
        @test ind isa Scaled{<:Scaled} && ind([T(0.5)]) == 0 && ind([T(2)]) == T(Inf)
        @test Scaled(ind, zero(T)) isa IndBox
      end

      @testset "values" begin
        x = T[tens(T, -l ÷ 3), -tens(T, -l ÷ 3), 0]
        for f in (small, large)
          @test f(x) isa T
          @test near(f(x), effective(f, f -> f.λ0) * 2 * big(x[1]))
        end
        # the examples of the issue
        @test near(small([one(T)]), λs)
        @test near(large([tens(T, -l + a + 1)]), λl * big(tens(T, -l + a + 1)))
        @test large(zeros(T, 3)) === zero(T)
        # ordinary products still merge and still take the ordinary path
        z = T[0.3, -0.7, 1.1]
        o = Scaled(Scaled(NormL1(T(0.2)), T(3)), T(5))
        @test o isa Scaled{<:NormL1} && o(z) === o.α * o.f(z)
        # a value that is out of range stays out of range: no finite result is invented
        @test large(T[tens(T, l)]) === T(Inf)
        @test small(T[tens(T, -l - a)]) isa T
      end

      @testset "subnormal coefficients and indicators" begin
        # the exponent goes last also for a subnormal level: 2^-1074 (2^-149) times 2^(2k)
        k = T === Float64 ? 600 : 100
        λ = nextfloat(zero(T))
        c = T(2)^k
        e = 2k + exponent(λ)
        for build in (l -> NormL1(l), l -> NormL2(l), l -> NormLinf(l))
          f = Scaled(Scaled(build(λ), c), c)
          @test f isa Scaled{<:Scaled}
          @test near(f([one(T)]), ldexp(BigFloat(1), e))
        end
        g = zeros(T, 1)
        lin = Scaled(Scaled(LinearFunction([λ]), c), c)
        gradient!(g, lin, T[1])
        @test near(g[1], ldexp(BigFloat(1), e))
        h = Scaled(Scaled(SqrNormL2(λ), c), c)
        @test near(lipschitz(h), ldexp(BigFloat(1), e))
        Hv = zeros(T, 1)
        hessian_vec!(Hv, h, T[1], T[1])
        @test near(Hv[1], ldexp(BigFloat(1), e))
        y = zeros(T, 1)
        prox!(y, Scaled(Scaled(NormL1(λ), c), c), T[3], ldexp(one(T), -e))
        @test y ≈ T[2] rtol = 8eps(T)
        # an indicator and the zero function are unchanged by any positive scale
        for factor in (tens(T, -a), tens(T, a))
          box = Scaled(Scaled(IndBox(zero(T), one(T)), factor), factor)
          @test box isa Scaled{<:Scaled}
          @test Conjugate(box)(T[1]) === T(1)
          @test Conjugate(box)(T[-1]) === zero(T)
          z = Scaled(Scaled(ZeroFunction(), factor), factor)
          @test Conjugate(z)(T[0]) === zero(T)
          @test Conjugate(z)(T[1]) === T(Inf)
        end
      end

      @testset "gradient and the other smooth oracles" begin
        # level 10^(2a - l) times λ x: a subproduct of the scales overflows
        h = Scaled(Scaled(SqrNormL2(tens(T, -l)), tens(T, a)), tens(T, a))
        x = T[tens(T, l ÷ 3)]
        λ = effective(h, f -> f.λ)
        g = zeros(T, 1)
        gradient!(g, h, x)
        @test near(g[1], λ * big(x[1]))
        g2 = zeros(T, 1)
        v = value_and_gradient!(g2, h, x)
        @test g2 == g
        @test near(v, λ * big(x[1])^2 / 2)
        @test near(h(x), λ * big(x[1])^2 / 2)
        c = T[tens(T, -l)]
        lin = Scaled(Scaled(LinearFunction(c), tens(T, a)), tens(T, a))
        gl = zeros(T, 1)
        gradient!(gl, lin, T[1])
        @test near(gl[1], big(c[1]) * big(tens(T, a))^2)
        # tuple of blocks
        gt = (zeros(T, 1), zeros(T, 1))
        s = Scaled(Scaled(SeparableSum(SqrNormL2(tens(T, -l)), SqrNormL2(tens(T, -l))), tens(T, a)), tens(T, a))
        gradient!(gt, s, (x, x))
        @test near(gt[1][1], λ * big(x[1])) && near(gt[2][1], λ * big(x[1]))
        # second-order and Lipschitz queries
        Hv = zeros(T, 1)
        hessian_vec!(Hv, h, x, T[1])
        @test near(Hv[1], λ)
        @test near(lipschitz(h), λ)
        L = zeros(T, 1)
        coordinate_lipschitz!(L, h)
        @test near(L[1], λ)
        @test near(coordinate_lipschitz(h, 1), λ)
      end

      @testset "proximal operator" begin
        # the threshold is 1 at the step that offsets the effective level
        γs, γl = inv(T(λs)), inv(T(λl))
        x = T[3, -3, 0.5]
        for (f, γ) in ((small, γs), (large, γl))
          expected = T[2, -2, 0]
          @test prox(f, x, γ) ≈ expected rtol = 8eps(T)
          y = zeros(T, 3)
          prox!(y, f, x, γ)
          @test y ≈ expected rtol = 8eps(T)
          # aliased output
          y = copy(x)
          prox!(y, f, y, γ)
          @test y ≈ expected rtol = 8eps(T)
          # the value at the proximal point
          y = zeros(T, 3)
          v = value_and_prox!(y, f, x, γ)
          @test y ≈ expected rtol = 8eps(T)
          @test near(v, effective(f, f -> f.λ0) * 4)
          y = copy(x)
          v = value_and_prox!(y, f, y, γ)
          @test y ≈ expected rtol = 8eps(T)
          @test near(v, effective(f, f -> f.λ0) * 4)
          for (k, e) in zip(1:3, expected)
            @test prox_entry(f, x[k], k, γ) ≈ e rtol = 8eps(T)
          end
          # a zero step copies the argument; a step that is not a number is passed on as it is
          @test prox(f, x, zero(T)) == x
          @test all(isnan, prox(f, x, T(NaN)))
        end
        # unit-normalised levels: the step and the level are not representable on their own
        if T === Float64
          f = Scaled(Scaled(NormL1(1e80), 1e-200), 1e-200)    # level 1e-320 (subnormal) times 1e80 step
          y = zeros(1)
          prox!(y, f, [3.0], 1e80)                              # threshold 1e-240: the point is unchanged
          @test y ≈ [3.0]
          prox!(y, f, [0.5e-240], 1e80)                         # the entry is below the threshold
          @test iszero(y[1])
          prox!(y, f, [3e-240], 1e80)
          @test y[1] ≈ 2e-240
          @test value_and_prox!(y, f, [3e-240], 1e80) === 0.0   # the value, 2e-560, is below the range
          @test y[1] ≈ 2e-240
          f = Scaled(Scaled(NormL1(1.0), 1e-200), 1e-200)       # level 1e-400 times step 1e400: not a step
          prox!(y, f, [0.5e-100], 1e300)                        # threshold 1e-100: the entry is removed
          @test iszero(y[1])
          prox!(y, f, [3e-100], 1e300)
          @test y[1] ≈ 2e-100
          # a function with no level to fold takes the step as the product rounds it
          q = Scaled(Scaled(SqrNormL2(1.0), 1e-200), 1e-200)
          prox!(y, q, [1.0], 1e300)                             # step 1e-100: prox of 1e-100 / 2 x^2
          @test y[1] ≈ 1 / (1 + 1e-100)
        end
        # the value at the proximal point, when the step's exponent goes into the levels
        if T === Float64
          f = Scaled(Scaled(NormL1(1e300), 1e-200), 1e-200)    # level 1e-100
          y = zeros(3)
          v = value_and_prox!(y, f, [3e-150, -3e-150, 0.5e-150], 1e-50)
          @test y ≈ [2e-150, -2e-150, 0] rtol = 8eps()
          @test v ≈ 4e-250 rtol = 8eps()
          @test v ≈ f(y) rtol = 8eps()
          for g in (Scaled(Scaled(ElasticNet(1e300, 1e-300), 1e-200), 1e-200),
                    Scaled(Scaled(NormL2(1e300), 1e-200), 1e-200),
                    Scaled(Scaled(GroupNormL2(1e300, [1:2, 3:3], [1.0, 1.0]), 1e-200), 1e-200))
            y = zeros(3)
            v = value_and_prox!(y, g, [3e-150, 4e-150, 0.5e-150], 1e-50)
            @test v ≈ g(y) rtol = 1e-12
            @test !iszero(v)
          end
          # a ridge level that is below the range does not discard the fold
          en = Scaled(Scaled(ElasticNet(1e300, 1e-300), 1e-200), 1e-200)
          y = zeros(1)
          prox!(y, en, [3e-105], 1e-5)
          @test y[1] ≈ 2e-105 rtol = 8eps()
          # an ElasticNet whose ridge level is below the range, alone and inside a composite
          comp = Scaled(Scaled(SeparableSum(ElasticNet(1e300, 1e-300), NormL1(1e300)), 1e-200), 1e-200)
          z = (zeros(1), zeros(1))
          prox!(z, comp, ([3e-150], [3e-150]), 1e-50)
          @test z[1][1] ≈ 2e-150 rtol = 8eps()
          @test z[2][1] ≈ 2e-150 rtol = 8eps()
          tr = Scaled(Scaled(Translate(ElasticNet(1e300, 1e-300), [0.0]), 1e-200), 1e-200)
          y = zeros(1)
          prox!(y, tr, [3e-150], 1e-50)
          @test y[1] ≈ 2e-150 rtol = 8eps()
          # the value returned with the point is the value of the original function, with the
          # dropped ridge level included
          ridge = Scaled(Scaled(ElasticNet(1e300, 1e-300, [0.0]), 1e-200), 1e-200)
          y = zeros(1)
          v = value_and_prox!(y, ridge, [1e300], 1e-50)
          @test v ≈ 5e-101 rtol = 8eps()
          @test v ≈ ridge(y) rtol = 8eps()
          y = zeros(1)
          v = value_and_prox!(y, tr, [1e300], 1e-50)
          @test v ≈ tr(y) rtol = 8eps()
          # sums, blocks and translations of functions that fold
          mk(l) = Scaled(Scaled(l, 1e-200), 1e-200)             # level 1e-100
          @test mk(SeparableSum(NormL1(1e300), NormL1(1e300)))(([1e100], [1e100])) ≈ 2 rtol = 8eps()
          @test mk(SeparableSum(NormL1(1e300), IndNonnegative()))(([1e100], [1.0])) ≈ 1 rtol = 8eps()
          @test mk(Translate(NormL1(1e300), [0.0]))([1e100]) ≈ 1 rtol = 8eps()
          @test mk(Translate(NormL1(1e300), [1e100]))([2e100]) ≈ 1 rtol = 8eps()
          @test mk(Sum(NormL1(1e300), NormL1(1e300)))([1e100]) ≈ 2 rtol = 8eps()
          y = zeros(1)
          prox!(y, mk(Translate(NormL1(1e300), [0.0])), [3e-50], 1e50)
          @test y[1] ≈ 2e-50 rtol = 8eps()
          z = (zeros(1), zeros(1))
          prox!(z, mk(SeparableSum(NormL1(1e300), NormL1(1e300))), ([3e-50], [3e-50]), 1e50)
          @test z[1][1] ≈ 2e-50 rtol = 8eps()
          @test z[2][1] ≈ 2e-50 rtol = 8eps()
        end
        # array steps keep the product's exponent last
        f = small
        y = zeros(T, 3)
        prox!(y, f, x, fill(γs, 3))
        @test y ≈ T[2, -2, 0] rtol = 8eps(T)
        v = value_and_prox!(y, f, x, fill(γs, 3))
        @test near(v, effective(f, f -> f.λ0) * 4)
        # indicators ignore their scale
        ind = Scaled(Scaled(IndBox(zero(T), one(T)), tens(T, -a)), tens(T, -a))
        @test prox(ind, T[3, -1, 0.5], one(T)) == T[1, 0, 0.5]
        @test prox(ind, T[3, -1, 0.5], zero(T)) == T[1, 0, 0.5]
      end

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
          @test prox_entry(c, T(2r), 1, one(T)) ≈ r rtol = 8eps(T)
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
      end

      @testset "optimality" begin
        for (f, λ) in ((small, λs), (large, λl))
          g = T[2 * λ, -λ / 2, λ]
          @test near(kkt_violation(f, g, 1), λ)             # |g| - level
          @test near(kkt_violation(f, g, 2), -λ / 2)
          @test abs(kkt_violation(f, g, 3)) <= 4eps(T) * λ
        end
        h = Scaled(Scaled(SqrNormL2(tens(T, -l)), tens(T, a)), tens(T, a))
        @test kkt_violation(h, T[3], 1) == 3
        begin
        end
      end
    end
  end

  @testset "no effect on ordinary scales" begin
    z = [0.3, -0.7, 1.1]
    for (f, x) in ((NormL1(0.2), z), (SqrNormL2(0.2), z), (NormL2(0.2), z))
      s = Scaled(Scaled(f, 3.0), 5.0)
      @test s isa Scaled{typeof(f)} && s.α == 15.0
      y = similar(x)
      @test allocs(s, x) == 0
      @test allocs(prox!, y, s, x, 0.1) == 0
      @test allocs(value_and_prox!, y, s, x, 0.1) == 0
      f isa NormL1 && @test allocs(prox_entry, s, 0.5, 1, 0.1) == 0
    end
    @test Scaled(Scaled(NormL1(1.0), 1e-100), 1e-100) isa Scaled{<:NormL1}
    @test Scaled(Scaled(NormL1(1.0), 1e100), 1e100) isa Scaled{<:NormL1}
    @test Scaled(Scaled(NormL1(1.0), 1e-300), 1e300).α === 1.0 * 1e-300 * 1e300
    @test Scaled(Scaled(NormL1(1.0), 1e-160), 1e-160) isa Scaled{<:Scaled}
    @test Scaled(Scaled(NormL1(1.0), 1e150), 1e150) isa Scaled{<:NormL1}
    @test Scaled(Scaled(NormL1(1.0), 1e150), 1e150).α === 1e150 * 1e150
    @test Scaled(Scaled(NormL1(1.0), 1e155), 1e155) isa Scaled{<:Scaled}
  end

end

end
