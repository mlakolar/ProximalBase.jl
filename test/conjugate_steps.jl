module ConjugateStepsTest

using Test
using ProximalBase
using LinearAlgebra
using Random
using JLArrays
using GPUArraysCore: ScalarDisallowed

using ..TestUtils

Random.seed!(215)

const steps = (1e-310, 1e-300, 1e-200, 1.0, 7.0, 1e300)

# exact projections in BigFloat
_ball(x, ρ) = (n = norm(x); n <= ρ ? copy(x) : x .* (ρ / n))
_box(x, l, u) = clamp.(x, l, u)
function _l1ball(x, ρ)
  sum(abs, x) <= ρ && return copy(x)
  a = sort(abs.(x); rev = true)
  s, θ = zero(eltype(x)), zero(eltype(x))
  for k in eachindex(a)
    s += a[k]
    t = (s - ρ) / k
    t < a[k] && (θ = t)
  end
  sign.(x) .* max.(abs.(x) .- θ, 0)
end
function _soc(x)
  t, v = x[1], x[2:end]
  s = norm(v)
  s <= t && return copy(x)
  s <= -t && return zero(x)
  α = (t + s) / 2
  vcat(α, (α / s) .* v)
end

# prox_{γf*}(x) = x - P_{γC}(x) for f the indicator of C, or the projection onto
# the dual ball (the polar cone) when f is a norm (a cone), with enough bits that
# 1e300 + 1e-310 is exact
reference(f, x, γ) = setprecision(() -> _reference(f, x, γ), BigFloat, 4096)
function _reference(f, x, γ)
  X, G = big.(x), big(γ)
  f isa IndBallL2 && return X .- _ball(X, G * big(f.r))
  f isa IndBox && return X .- _box(X, G .* big.(f.l), G .* big.(f.u))
  f isa Scaled && return _reference(f.f, x, γ)
  f isa Translate{<:IndBallL2} && (B = G .* big.(f.b); return X .- (B .+ _ball(X .- B, G * big(f.f.r))))
  f isa NormL1 && return clamp.(X, -1, 1)
  f isa NormL2 && return _ball(X, big(1))
  f isa NormLinf && return _l1ball(X, big(1))
  f isa IndNonnegative && return min.(X, 0)
  f isa IndSOC && return X .- _soc(X)
  f isa Conjugate{<:NormL1} && return sign.(X) .* max.(abs.(X) .- G, 0)
  error("no reference for $(typeof(f))")
end

const b = [0.3, -0.2, 0.1, 0.4, -0.5]
const exact = (IndBallL2(1.0), IndBallL2(2.5), IndBallLinf(1.0), IndBox(-1.0, 2.0),
               IndBox([-Inf, -1.0, 0.0, -2.0, -Inf], [Inf, 1.0, Inf, 0.5, 3.0]),
               IndSOC(), IndNonnegative(), NormL1(), NormL2(), NormLinf(),
               Translate(IndBallL2(1.0), b), 2 * IndBallL2(1.0), Conjugate(NormL1()))

tolerance(f, x, γ) = 8 * eps(Float64) * max(1, maximum(abs, x)) + (f isa Translate ? 8 * eps(Float64) * γ * norm(f.b) : 0.0)

@testset "the example of the issue" begin
  @test prox(Conjugate(IndBallL2(1.0)), [0.0, 1.0], 1e-310) == [0.0, 1.0]
  @test prox(Conjugate(NormL1()), [0.0, 1.0, 3.0], 1e-310) == [0.0, 1.0, 1.0]
  @test prox(Conjugate(IndNonnegative()), [2.0, -3.0], 1e-310) == [0.0, -3.0]
  @test prox(Conjugate(IndBox(-1.0, 2.0)), [0.0, 1.0], 1e300) == [0.0, 0.0]
end

@testset "conjugates computed without dividing by the step" begin
  for f in exact, γ in steps
    c = Conjugate(f)
    for x in (2 .* randn(5), [0.0, 1.0, -3.0, 0.5, 2.0], min(γ, 1e300) .* randn(5))
      y = prox(c, x, γ)
      @test all(isfinite, y)
      @test isapprox(y, Float64.(reference(f, x, γ)); rtol = 0, atol = tolerance(f, x, γ))
      # in place, through value_and_prox!
      z = copy(x)
      v = value_and_prox!(z, c, z, γ)
      @test z == y
      @test isfinite(v)
    end
  end
  # inputs up to 1e300 at a small step. The balls, boxes, translations and the
  # double conjugate are accurate entry by entry, so the moderate entries match
  # the reference to 8 eps. The norms and cones are accurate only to about
  # eps max|x|: Moreau's identity subtracts prox_f(x) from x, which cancels in
  # the moderate entries against an error set by the huge ones, so for them
  # only finiteness is asserted.
  γ = 1e-10
  x = [1e300, -3e299, 2.0, 0.0, 5e150]
  moderate = abs.(x) .<= 1e3
  for f in exact
    y = prox(Conjugate(f), x, γ)
    @test all(isfinite, y)
    if f isa Union{IndBallL2, IndBox, Translate, Scaled, Conjugate}
      @test isapprox(y[moderate], Float64.(reference(f, x, γ))[moderate]; rtol = 0, atol = 8eps(Float64))
    end
  end
end

@testset "empty inputs" begin
  @test prox(Conjugate(SqrNormL2()), Float64[], 1.0) == Float64[]
  @test value_and_prox!(Float64[], Conjugate(SqrNormL2()), Float64[], 1.0) == 0.0
  @test prox!(Float32[], Conjugate(NormL1()), Float64[], 1.0) == Float32[]
  @test prox(Conjugate(SeparableSum(SqrNormL2(), SqrNormL2())), (Float64[], [1.0]), 1.0) == (Float64[], [0.5])
  y = prox(Conjugate(SqrNormL2()), zeros(0, 3), 1.0)
  @test y isa Matrix{Float64} && size(y) == (0, 3)
  # SqrNormL2 has no array steps, at the base as now; a separable fallback does
  @test_throws MethodError prox(Conjugate(SqrNormL2()), Float64[], Float64[])
  @test prox(Conjugate(ElasticNet(0.4, 0.7)), Float64[], Float64[]) == Float64[]
  @test value_and_prox!(Float64[], Conjugate(ElasticNet(0.4, 0.7)), Float64[], Float64[]) == 0.0
end

@testset "dilated bounds in the wider precision" begin
  # γ = 1e-10 dilates a bound of 1e39, outside Float32, to 1e29, inside it
  for f in (IndBallL2(1e39), IndBox(-1e39, 1e39))
    y = prox!(zeros(Float32, 1), Conjugate(f), Float32[1f30], 1e-10)
    @test y ≈ Float32[9f29] rtol = 4eps(Float32)
    @test prox_entry(Conjugate(IndBox(-1e39, 1e39)), 1f30, 1, 1f-10) ≈ 9f29 rtol = 4eps(Float32)
  end
  # at γ = 1 the dilated radius overflows to Inf32: x is inside, the result is 0
  @test prox!(zeros(Float32, 1), Conjugate(IndBallL2(1e39)), Float32[1f30], 1.0) == Float32[0]
  @test prox!(zeros(Float32, 1), Conjugate(IndBox(-1e39, 1e39)), Float32[1f30], 1.0) == Float32[0]
end

@testset "translations at a step that overflows" begin
  # γ = 1e300 is Inf32. With b = 0 the set is the ball of radius γ, which
  # contains x, so the result is 0, and a zero entry of b never forms 0 * Inf.
  x = Float32[0.5, 0.5]
  @test prox!(zeros(Float32, 2), Conjugate(Translate(IndBallL2(1.0), [0.0, 0.0])), x, 1e300) == Float32[0, 0]
  # With b = (0, 1), |x - γb|² = γ² - γ + 1/2 < γ², so x is in γ(b + B) and the
  # exact result is 0 again, but γb = (0, Inf32) is not finite in working
  # precision: the result is not finite and the check throws.
  c = Conjugate(Translate(IndBallL2(1.0), [0.0, 1.0]))
  y = zeros(Float32, 2)
  @test_throws "is not finite at a finite input" prox!(y, c, x, 1e300)
  @test y == Float32[0, 0]
  # in Float64, γb is finite and the result is 0 to about eps γ|b|
  @test isapprox(prox(c, [0.5, 0.5], 1e300), [0.0, 0.0]; atol = 8eps(Float64) * 1e300)
end

@testset "double conjugates with array steps and entries" begin
  twice = Conjugate(Conjugate(NormL1()))
  x, γ = [3.0, 0.5], [1e-310, 1.0]
  @test prox(twice, x, γ) == [3.0, 0.0]
  @test prox(twice, x, γ) == prox(NormL1(), x, γ)
  y, expected = similar(x), similar(x)
  @test value_and_prox!(y, twice, x, γ) == value_and_prox!(expected, NormL1(), x, γ)
  @test y == expected
  @test_throws DimensionMismatch prox!(similar(x), twice, x, [1.0])
  for v in (3.0, 0.5, -2.0), step in (1e-310, 1.0, 7.0)
    @test prox_entry(twice, v, 1, step) == prox_entry(NormL1(), v, 1, step)
  end
  # array steps for a double conjugate of a fallback: f** = f, no division
  f = Conjugate(Conjugate(ElasticNet(0.4, 0.7)))
  @test prox(f, x, γ) == prox(ElasticNet(0.4, 0.7), x, γ)
end

@testset "tuples" begin
  c = Conjugate(SeparableSum(NormL1(), NormL2()))
  x = ([0.5, 3.0, -2.0], [3.0, 4.0])
  for γ in steps
    y = prox(c, x, γ)
    atol = 8 * eps(Float64) * 4
    @test isapprox(y[1], Float64.(reference(NormL1(), x[1], γ)); rtol = 0, atol)
    @test isapprox(y[2], Float64.(reference(NormL2(), x[2], γ)); rtol = 0, atol)
    alias = map(copy, x)
    @test value_and_prox!(alias, c, alias, γ) == 0
    @test alias == y
  end
  twice = Conjugate(Conjugate(SeparableSum(NormL1(), SqrNormL2())))
  out, expected = map(similar, x), map(similar, x)
  @test value_and_prox!(out, twice, x, 1e-310) == value_and_prox!(expected, SeparableSum(NormL1(), SqrNormL2()), x, 1e-310)
  @test out == expected

  # a later block that is not finite leaves every block unwritten
  x = ([1e-300], [0.0, 1.0])
  for c in (Conjugate(SeparableSum(SqrNormL2(), SqrNormL2())), Conjugate(SeparableSum(ElasticNet(0.4, 0.7), SqrNormL2()))),
      op in (prox!, value_and_prox!), alias in (false, true)
    input = map(copy, x)
    out = alias ? input : map(zero, x)
    before = map(copy, out)
    @test_throws "is not finite at a finite input" op(out, c, input, 1e-310)
    @test out == before
    @test input == x
  end
  c = Conjugate(SeparableSum(SqrNormL2(), SqrNormL2()))
  @test prox(c, x, 0.5) == map(xk -> xk .- 0.5 .* prox(SqrNormL2(), xk ./ 0.5, inv(0.5)), x)
end

@testset "fallback: finite results at finite inputs" begin
  for (f, x, γ) in ((SqrNormL2(), [0.0, 1.0], 1e-310),
                    (LinearFunction([1.0, 2.0]), [0.0, 1e-300], 1e-310),
                    (SqrNormL2(), Float32[0.5, 1], 1e300),
                    (ElasticNet(0.4, 0.7), [0.0, 1.0], 1e-310),
                    (Berhu(1.0, 2.0), [0.0, 1.0], 1e-310))
    c = Conjugate(f)
    for op in (prox!, value_and_prox!), alias in (false, true)
      input = copy(x)
      out = alias ? input : zero(x)
      before = copy(out)
      @test_throws "is not finite at a finite input" op(out, c, input, γ)
      @test out == before
      @test input == x
    end
  end
  # elsewhere bitwise the base computation
  formula(f, x, γ) = x .- γ .* prox(f, x ./ γ, inv(γ))
  for f in (SqrNormL2(), ElasticNet(0.4, 0.7), Berhu(1.0, 2.0), LinearFunction([1.0, 2.0, -1.0]),
            SqrHingeLoss(), NegEntropy()),
      (x, γ) in (([0.3, -1.2, 2.0], 0.3), ([0.3, -1.2, 2.0], 7.0), ([0.0, 1e-300, -2e-300], 1e-310))
    expected = formula(f, x, γ)
    if all(isfinite, expected)
      @test prox(Conjugate(f), x, γ) == expected
    else
      @test_throws "is not finite at a finite input" prox(Conjugate(f), x, γ)
    end
  end
  @test prox(Conjugate(SqrNormL2()), [0.0, 1e-300], 1e-310) == [0.0, 1e-300]
  @test isequal(prox(Conjugate(SqrNormL2()), [NaN, 1.0], 0.5), formula(SqrNormL2(), [NaN, 1.0], 0.5))
end

@testset "prox_entry" begin
  for γ in steps, v in (0.0, 0.5, -3.0, 2.0)
    @test prox_entry(Conjugate(NormL1()), v, 1, γ) == clamp(v, -1, 1)
    @test prox_entry(Conjugate(IndNonnegative()), v, 1, γ) == min(v, 0)
    for f in (IndBox(-1.0, 2.0), IndBox([-Inf, -1.0], [Inf, 0.5]), IndBallLinf(1.0)), i in 1:2
      x = fill(v, 2)
      @test prox_entry(Conjugate(f), v, i, γ) == prox(Conjugate(f), x, γ)[i]
    end
  end
  # the fallback never throws
  @test prox_entry(Conjugate(SqrHingeLoss()), 1.0, 1, 1e-310) isa Float64
  f = ElasticNet(0.4, 0.7)
  @test prox_entry(Conjugate(f), 1.0, 1, 1e-310) isa Float64
  @test prox_entry(Conjugate(f), 1.5, 1, 0.3) == 1.5 - 0.3 * prox_entry(f, 1.5 / 0.3, 1, inv(0.3))
end

@testset "values" begin
  x = [0.3, -1.0, 2.0]
  for γ in steps
    y = similar(x)
    @test value_and_prox!(y, Conjugate(NormL1()), x, γ) == 0
    f = IndBallL2(1.5)
    @test value_and_prox!(y, Conjugate(f), x, γ) == Conjugate(f)(y)
    @test y == prox(Conjugate(f), x, γ)
    f = IndBox(-1.0, 2.0)
    @test value_and_prox!(y, Conjugate(f), x, γ) == Conjugate(f)(y)
    @test y == prox(Conjugate(f), x, γ)
    expected = similar(x)
    @test value_and_prox!(y, Conjugate(Conjugate(NormL1())), x, γ) == value_and_prox!(expected, NormL1(), x, γ)
    @test y == expected
  end
end

@testset "array steps" begin
  x = [0.5, -3.0, 2.0]
  γ = [1e-310, 1.0, 1e300]
  @test prox(Conjugate(NormL1()), x, γ) == clamp.(x, -1, 1)
  @test value_and_prox!(similar(x), Conjugate(NormL1()), x, γ) == 0
  for f in (IndBox(-1.0, 2.0), IndBox([-Inf, -1.0, 0.0], [Inf, 0.5, Inf]))
    l, u = f.l isa Real ? fill(f.l, 3) : f.l, f.u isa Real ? fill(f.u, 3) : f.u
    ref = setprecision(() -> big.(x) .- clamp.(big.(x), big.(γ) .* big.(l), big.(γ) .* big.(u)), BigFloat, 4096)
    @test isapprox(prox(Conjugate(f), x, γ), Float64.(ref); rtol = 0, atol = 8eps(Float64) * 3)
  end
  c = Conjugate(ElasticNet(0.4, 0.7))
  for op in (prox!, value_and_prox!), alias in (false, true)
    input = copy(x)
    out = alias ? input : zero(x)
    before = copy(out)
    @test_throws "is not finite at a finite input" op(out, c, input, γ)
    @test out == before
    @test input == x
  end
end

@testset "mixed precision" begin
  c = Conjugate(NormL2())
  y = zeros(Float32, 2)
  @test_throws "is not finite at a finite input" prox!(y, c, [1e39, 0.0], 1.0)
  @test y == zeros(Float32, 2)
  x = [0.3, -2.0, 1.0]
  for γ in steps
    y = prox!(zeros(Float32, 3), c, x, γ)
    ref = Float32.(reference(NormL2(), Float32.(x), γ))
    @test isapprox(y, ref; rtol = 0, atol = 8 * eps(Float32) * 2)
  end
  # inputs no wider than the output are not checked
  @test prox!(zeros(3), c, Float32[0.3, -2, 1], 1e-310) ≈ [0.3, -2, 1] ./ sqrt(5.09) atol = 1e-6
  @test prox!(zeros(2), Conjugate(NormL1()), [1, -3], 1e-310) == [1.0, -1.0]
end

@testset "allocations" begin
  x = randn(10)
  y = similar(x)
  for f in (NormL1(), IndBallL2(1.0), IndBox(-1.0, 1.0), Translate(IndBallL2(1.0), randn(10)), SqrNormL2())
    c = Conjugate(f)
    prox!(y, c, x, 0.5)
    @test allocs(prox!, y, c, x, 0.5) == 0
    f isa SqrNormL2 || @test allocs(prox!, y, c, x, 1e-310) == 0
  end
  twice = Conjugate(Conjugate(NormL1()))
  prox!(y, twice, x, 1e-310)
  @test allocs(prox!, y, twice, x, 1e-310) == 0
  @test allocs(value_and_prox!, y, twice, x, 1e-310) == 0
  # wide bounds are dilated once per call, in the precision of the step: what
  # converting a BigFloat allocates depends on the platform, but not on the length
  c = Conjugate(IndBox(big"-1", big"1"))
  x10, y10, x1000, y1000 = randn(10), zeros(10), randn(1000), zeros(1000)
  prox!(y10, c, x10, 0.5)
  prox!(y1000, c, x1000, 0.5)
  @test allocs(prox!, y1000, c, x1000, 0.5) == allocs(prox!, y10, c, x10, 0.5)
  @test y1000 == x1000 .- clamp.(x1000, -0.5, 0.5)
  # array bounds and array steps dilate entry by entry
  b = IndBox(fill(-1.0, 1000), fill(1.0, 1000))
  γ = fill(0.5, 1000)
  @test prox(Conjugate(b), x1000, 0.5) == x1000 .- clamp.(x1000, -0.5, 0.5)
  @test prox(Conjugate(IndBox(-1.0, 1.0)), x1000, γ) == x1000 .- clamp.(x1000, -0.5, 0.5)
  @test prox(Conjugate(b), x1000, γ) == x1000 .- clamp.(x1000, -0.5, 0.5)
end

@testset "dilation of a bound" begin
  dilate = ProximalBase._dilate
  @test dilate(1f-10, 1e39) === 1f29
  @test dilate(1f0, 1e39) === Inf32
  @test dilate(Inf32, 0.0) === 0f0
  @test dilate(0f0, -Inf) === -Inf32
  @test dilate(1f-10, 3) === 3f-10
  @test dilate(0.5, big"3") === 1.5
  # a nonzero bound that underflows in the precision of the step
  # 1f30 is 1.000000015e30, so the exact γb rounds to 1.00000005f-20, one ulp above 1f-20
  @test dilate(1f30, 1e-50) === Float32(big(1f30) * big(1e-50))
  @test dilate(1f30, 1e-50) ≈ 1f-20 rtol = 2eps(Float32)
  @test dilate(1f0, 1e-50) === 0f0
  @test dilate(1f0, 0.0) === 0f0
  for (γ, b) in ((1f-10, 1e39), (1f0, 3), (0.5, big"3"), (0.5, 2f0), (1f30, 1e-50))
    @test Base.return_types(dilate, (typeof(γ), typeof(b))) == [float(typeof(γ))]
  end
end

@testset "the finiteness check on a GPU array" begin
  # with scalar indexing disallowed, for this task only
  task_local_storage(:ScalarIndexing, ScalarDisallowed) do
    nonfinite = ProximalBase._moreau_nonfinite
    x = JLArray([0.0, 1.0])
    @test nonfinite(x, x, JLArray([0.0, Inf]), 0.3, true)
    @test !nonfinite(x, x, JLArray([0.0, 0.5]), 0.3, true)
    e = JLArray(Float64[])
    @test !nonfinite(e, e, JLArray(Float64[]), 0.3, true)
  end
end

end
