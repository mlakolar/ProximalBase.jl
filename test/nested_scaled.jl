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
        # the value that comes back with an array step is recovered in the function's own units when
        # the leaf's value underflows: the step 10^(2a - l) is a normal number, the leaf's value
        # 10^-l |y| is not, and the value of the chain is the product
        let s = Scaled(Scaled(NormL1(tens(T, -l)), tens(T, a)), tens(T, a)), step = tens(T, -l)
          λe = effective(s, f -> f.λ0)
          x = T[3, -3] .* tens(T, -a)
          γ = fill(step, 2)
          thr = big(step) * λe                       # the threshold of the proximal operator
          yref = [sign(big(xi)) * max(abs(big(xi)) - thr, 0) for xi in x]
          vref = λe * sum(abs, yref)
          @test PB._value_unchecked(s.f.f, T[2, 2] .* tens(T, -a)) == 0   # the leaf's own value is lost
          y = zeros(T, 2)
          v = value_and_prox!(y, s, x, γ)
          @test all(near.(y, yref))
          @test v > 0 && near(v, vref)
          @test near(v, s(y))
          # the output is the input
          y = copy(x)
          v = value_and_prox!(y, s, y, γ)
          @test all(near.(y, yref))
          @test v > 0 && near(v, vref)
          # the point is that of prox!, with no second proximal computation
          z = zeros(T, 2)
          prox!(z, s, x, γ)
          @test z == y
        end
        # an ordinary array step takes the same value as before, at no cost
        let s = Scaled(Scaled(NormL1(T(0.2)), T(3)), T(5)), z = T[0.3, -0.7, 1.1], γ = T[0.1, 0.2, 0.3]
          y = similar(z)
          v = value_and_prox!(y, s, z, γ)
          @test v ≈ s(y)
          @test v ≈ 3T(5) * T(0.2) * sum(abs, y) rtol = 8eps(T)
          @test allocs(value_and_prox!, y, s, z, γ) == 0
        end
        # indicators ignore their scale
        ind = Scaled(Scaled(IndBox(zero(T), one(T)), tens(T, -a)), tens(T, -a))
        @test prox(ind, T[3, -1, 0.5], one(T)) == T[1, 0, 0.5]
        @test prox(ind, T[3, -1, 0.5], zero(T)) == T[1, 0, 0.5]
      end

      @testset "conjugates" begin
        # the Float32 pass keeps the allocation check of a Float32 step in this file; the rest of its
        # conjugates runs in nested_scaled_grids.jl
        if T === Float64
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
        let g = Conjugate(Scaled(Translate(SqrNormL2(one(T)), zeros(T, 3)), T(2))), y = zeros(T, 3), x = T[0.3, -0.7, 1.1]
          @test allocs(value_and_prox!, y, g, x, one(T)) == 0
        end
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
        # an indicator is its own scaling: its violation does not depend on the factors, nor on
        # whether their product, or its reciprocal, is a number
        for ind in (IndBox(zero(T), one(T)), IndNonnegative())
          g = T[-1, 2, 0]
          for e in (a, -a), depth in (2, 3)
            s = ind
            for _ in 1:depth
              s = Scaled(s, tens(T, e))
            end
            @test s isa Scaled{<:Scaled}
            for i in 1:3
              @test kkt_violation(s, g, i) === kkt_violation(ind, g, i)
              @test kkt_violation(s, g, zeros(T, 3), i) === kkt_violation(ind, g, zeros(T, 3), i)
            end
            @test kkt_violation(s, g, 1) == 1
            # an atom iterate takes the worst index of a group
            x = AtomIterate(T, 3, 1)
            @test kkt_violation(s, g, x, 1) === kkt_violation(ind, g, x, 1)
            # a symmetric parameter takes the entry of the lower triangle
            G = T[-1 2; 2 -3]
            xs = SymmetricSparseIterate(zeros(T, 2, 2))
            for k in 1:3
              @test kkt_violation(s, G, xs, k) === kkt_violation(ind, G, xs, k)
            end
          end
        end
        # a penalty that needs its scale keeps it, by the folded level and by the reciprocal alike
        let g = T[2 * λs, -λs / 2], x = zeros(T, 2)
          @test near(kkt_violation(small, g, x, 1), λs)
          @test near(kkt_violation(small, g, AtomIterate(T, 2, 1), 1), λs)
        end
        let s = Scaled(Scaled(NormL1(T(0.2)), T(3)), T(5))
          @test kkt_violation(s, T[2, -0.1], 1) ≈ 2 - T(3) rtol = 8eps(T)      # |g| - 15 ⋅ 0.2
          @test kkt_violation(s, T[2, -0.1], 2) ≈ T(0.1) - T(3) rtol = 8eps(T)
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
  for T in (Float32, Float64)
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
            @test near(prox_entry(f, x[k], k, γ), r[k]) || (iszero(r[k]) && iszero(prox_entry(f, x[k], k, γ)))
          end
        end
      end

      # a single factor and a scalar step: the products of the factor with the level and of the factor
      # with the step are both out of range or both fine, the coefficient is 10^(l - 2a)
      θ = T(10)^(l - 2a)
      x = T[3, -3, 0.5] .* θ
      single = Scaled(NormL1(tens(T, l)), tens(T, -a))
      check(single, x, tens(T, -a), "single factor, scalar step")
      check(single, x, fill(tens(T, -a), 3), "single factor, array step")
      # the factor is large and the step overflows with it: the coefficient is 10^(2a - l)
      big_ = Scaled(NormL1(tens(T, -l)), tens(T, a))
      x2 = T[3, -3, 0.5] .* T(10)^(2a - l)
      check(big_, x2, tens(T, a), "single large factor, scalar step")
      check(big_, x2, fill(tens(T, a), 3), "single large factor, array step")
      # weights and levels of different size, entries of different size
      weighted = Scaled(NormL1(tens(T, l), T[1, 2, 4]), tens(T, -a))
      check(weighted, x, fill(tens(T, -a), 3), "single factor, weights")
      check(weighted, x, T[1, 2, 4] .* tens(T, -a), "single factor, entries of different size")
      en = Scaled(ElasticNet(tens(T, l), tens(T, l)), tens(T, -a))
      check(en, x, fill(tens(T, -a), 3), "single factor, ElasticNet")
      if T === Float64
        tr = Scaled(Translate(NormL1(tens(T, l)), T[0, 0, 0]), tens(T, -a))
        check(tr, x, fill(tens(T, -a), 3), "single factor, Translate")
      end
      # a chain with an array step: the factors are 10^-a twice, the step is 10^-g
      θc = T(10)^(l - 2a - g)
      xc = T[3, -3, 0.5] .* θc
      chain = Scaled(Scaled(NormL1(tens(T, l)), tens(T, -a)), tens(T, -a))
      @test chain isa Scaled{<:Scaled}
      check(chain, xc, tens(T, -g), "chain, scalar step (#549)")
      check(chain, xc, fill(tens(T, -g), 3), "chain, array step")
      check(Scaled(Scaled(NormL1(tens(T, l), T[1, 2, 4]), tens(T, -a)), tens(T, -a)), xc, fill(tens(T, -g), 3), "chain, weights")
      check(Scaled(Scaled(ElasticNet(tens(T, l), tens(T, l)), tens(T, -a)), tens(T, -a)), xc, fill(tens(T, -g), 3), "chain, ElasticNet")
      check(Scaled(Scaled(NormL1(tens(T, l)), tens(T, -a)), tens(T, -a)), xc, T[1, 1e3, 1e-3] .* tens(T, -g), "chain, entries of different size")
      # a step per group
      if T === Float64
        grp = Scaled(Scaled(GroupNormL2(tens(T, l), [1:2, 3:3], T[1, 1]), tens(T, -a)), tens(T, -a))
        xg = T[3, 4, 0.5] .* θc
        check(grp, xg, fill(tens(T, -g), 2), "chain, GroupNormL2 with a step per group")
        grp1 = Scaled(GroupNormL2(tens(T, l), [1:2, 3:3], T[1, 1]), tens(T, -a))
        check(grp1, T[3, 4, 0.5] .* θ, fill(tens(T, -a), 2), "single factor, GroupNormL2 with a step per group")
      end
      if T === Float64
        # a function with a level but no array-step proximal operator takes the scalar route
        check(Scaled(NormL2(tens(T, l)), tens(T, -a)), T[3, 4] .* θ, tens(T, -a), "single factor, NormL2"; array = false)
      end

      @testset "the level and the factors have no product in range" begin
        # the level times the factors is subnormal, the step is large and the coefficient is 10^(l2 - 2a + g2):
        # the shared exponent of the entries of the step goes into the levels
        l2, g2 = steps566(T).l2, steps566(T).g2
        @test T(10)^l2 * T(10)^-a * T(10)^-a < floatmin(T)
        θ2 = T(10)^(l2 - 2a + g2)
        x2 = T[3, -3, 0.5] .* θ2
        low = Scaled(Scaled(NormL1(tens(T, l2)), tens(T, -a)), tens(T, -a))
        check(low, x2, tens(T, g2), "chain, scalar step (#549)")
        check(low, x2, fill(tens(T, g2), 3), "chain, array step")
        check(low, x2, T[1, 10, 0.1] .* tens(T, g2), "chain, entries of different size")
        # entries far apart: only a shift that raises the level as far as the entries allow holds it
        check(low, T[3, 3] .* θ2, T[tens(T, g2), tens(T, -g2)], "chain, entries far apart")
        check(Scaled(Scaled(NormL1(tens(T, l2), T[1, 2, 4]), tens(T, -a)), tens(T, -a)), x2, fill(tens(T, g2), 3), "chain, weights")
        if T === Float64
          check(Scaled(Scaled(GroupNormL2(tens(T, l2), [1:2, 3:3], T[1, 1]), tens(T, -a)), tens(T, -a)), T[3, 4, 0.5] .* θ2, fill(tens(T, g2), 2), "chain, GroupNormL2 with a step per group")
        end
        # the value at the point is the value of the chain, not that of the shifted function
        xb = T[tens(T, (T === Float64 ? -300 : -30) - (l2 - 2a)), 0]
        y = zeros(T, 2)
        v = value_and_prox!(y, low, xb, fill(tens(T, g2), 2))
        ref = effective(low, f -> f.λ0) * sum(abs, big.(y))
        @test ref > floatmin(T) && valnear(v, ref)
        # entries that no shared exponent can hold, and entries that are not positive numbers, keep the rounded step
        y = zeros(T, 2)
        prox!(y, low, T[3, 3] .* θ2, T[tens(T, g2), T(0)])
        @test near(y[1], 2θ2) && y[2] == T(3) * θ2
        prox!(y, low, T[3, 3] .* θ2, T[tens(T, g2), T(Inf)])
        @test near(y[1], 2θ2) && y[2] == 0
        prox!(y, low, T[3, 3], T[0, 0])
        @test y == T[3, 3]
      end

      @testset "values at the point" begin
        y = zeros(T, 3)
        v = value_and_prox!(y, single, x, fill(tens(T, -a), 3))
        @test valnear(v, effective(single, f -> f.λ0) * sum(abs, big.(y)))
        @test v > 0 && near(v, single(y))
        v = value_and_prox!(y, single, x, tens(T, -a))
        @test v > 0 && near(v, single(y))
        v = value_and_prox!(y, chain, xc, fill(tens(T, -g), 3))
        @test valnear(v, effective(chain, f -> f.λ0) * sum(abs, big.(y)))
        v = value_and_prox!(y, chain, xc, tens(T, -g))
        @test valnear(v, effective(chain, f -> f.λ0) * sum(abs, big.(y)))
      end

      @testset "scalar steps keep their meaning" begin
        # zero, infinite, NaN and negative steps, and the infinite factor, as for the function inside
        for γ in (zero(T), T(Inf), T(NaN), -tens(T, -a))
          @test isequal(prox(single, x, γ), prox(single.f, x, single.α * γ))
          @test isequal(prox(big_, x2, γ), prox(big_.f, x2, big_.α * γ))
        end
        inf = Scaled(NormL1(one(T)), T(Inf))
        for γ in (T(0.5), zero(T))
          @test isequal(prox(inf, x, γ), prox(inf.f, x, inf.α * γ))
        end
        # an indicator ignores its scale
        ind = Scaled(IndBox(zero(T), one(T)), tens(T, -a))
        @test prox(ind, T[3, -1, 0.5], tens(T, -a)) == T[1, 0, 0.5]
        @test prox(Scaled(IndBox(zero(T), one(T)), tens(T, a)), T[3, -1, 0.5], tens(T, a)) == T[1, 0, 0.5]
        # a function with no level to fold takes the step as the product rounds it, as before: the
        # product 10^(-2a) is zero, so the point is the argument
        q = Scaled(SqrNormL2(one(T)), tens(T, -a))
        @test prox(q, T[1], tens(T, -a)) == T[1]
      end

      @testset "preflight leaves the output alone" begin
        for f in (single, chain)
          y = fill(T(7), 3)
          @test_throws DimensionMismatch prox!(y, f, x, fill(tens(T, -a), 2))
          @test y == fill(T(7), 3)
          @test_throws DimensionMismatch prox!(y, f, x, fill(tens(T, -a), 4))
          @test y == fill(T(7), 3)
          @test_throws DimensionMismatch value_and_prox!(y, f, x, fill(tens(T, -a), 2))
          @test y == fill(T(7), 3)
          @test_throws DimensionMismatch prox!(zeros(T, 2), f, x, tens(T, -a))
          @test_throws DimensionMismatch prox!(y, f, T[1, 2], tens(T, -a))
          @test y == fill(T(7), 3)
        end
        bad = Scaled(NormL1(tens(T, l), T[1, 2]), tens(T, -a))   # two weights for three entries
        y = fill(T(7), 3)
        @test_throws DimensionMismatch prox!(y, bad, x, fill(tens(T, -a), 3))
        @test y == fill(T(7), 3)
      end

      @testset "ordinary steps are untouched" begin
        z, γ = T[0.3, -0.7, 1.1], T[0.1, 0.2, 0.3]
        for f in (Scaled(NormL1(T(0.2)), T(3)), Scaled(NormL1(T(0.2), T[1, 2, 3]), T(3)),
                  Scaled(ElasticNet(T(0.2), T(0.1)), T(3)), Scaled(Scaled(NormL1(T(0.2)), T(3)), T(5)))
          y, yref = similar(z), similar(z)
          for step in (T(0.1), γ)
            prox!(y, f, z, step)
            prox!(yref, f.f, z, f.α * step)
            @test y == yref
            @test allocs(prox!, y, f, z, step) == 0
            @test allocs(value_and_prox!, y, f, z, step) == 0
          end
          @test allocs(prox_entry, f, z[1], 1, T(0.1)) == 0
        end
        # a factor at the edge of the band and a step that is itself ordinary
        edge = Scaled(NormL1(one(T)), T(2)^(exponent(floatmax(T)) ÷ 4))
        y, yref = zeros(T, 3), zeros(T, 3)
        prox!(y, edge, z, T[0.1, 0.2, 0.3] .* T(2)^-(exponent(floatmax(T)) ÷ 4))
        prox!(yref, edge.f, z, T[0.1, 0.2, 0.3])
        @test y ≈ yref rtol = 8eps(T)
      end
    end
  end
end


# Review of #575: the fold that refuses a level (empty array steps) and the explicit infinite factor
# (a factor that is not a scale to fold). Both gave a result on the base of #566 and must still.
@testset "empty array steps and infinite factors under the fold (#566)" begin
  for T in (Float32, Float64)
    @testset "$T" begin
      sub = floatmin(T) / T(2)^10                  # the effective level is subnormal: the fold refuses it
      tiny = T == Float64 ? T(1e-200) : T(1f-30)   # outside the band of a single factor
      @testset "empty steps" begin
        for s in (Scaled(NormL1(one(T)), sub), Scaled(Scaled(NormL1(one(T)), sub), sub),
                  Scaled(NormL1(one(T)), tiny), Scaled(Scaled(NormL1(one(T)), tiny), tiny),
                  Scaled(NormL1(tens(T, -30)), tens(T, 20)))
          x, γ = T[], T[]
          y = T[]
          @test prox!(y, s, x, γ) === y
          @test isempty(y)
          @test value_and_prox!(y, s, x, γ) === zero(T)
          @test isempty(y)
          # the output is not read: a longer one is refused before it is touched
          y = T[7]
          @test_throws DimensionMismatch prox!(y, s, x, γ)
          @test y == T[7]
        end
        # a step of another length than the argument is still refused
        @test_throws DimensionMismatch prox!(T[], Scaled(NormL1(one(T)), sub), T[], T[1])
      end
      if T === Float64
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

end
