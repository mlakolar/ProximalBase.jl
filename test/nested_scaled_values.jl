module NestedScaledValuesTest

# Values of nested `Scaled` chains whose product does not fold into a normal level (#568). The
# references multiply the stored factors, the level, the weights and the argument in BigFloat, so
# they never see a rounded product. A value is accurate to (n + factors + 2) eps, n the number of
# terms the function sums.

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using JLArrays
using ..TestUtils

const PB = ProximalBase

# the wrappers from the outermost to the function inside
function stored(f)
  fs = Any[]
  while f isa Scaled
    push!(fs, f.α)
    f = f.f
  end
  fs, f
end
product(f) = prod(big.(stored(f)[1]))
nfactors(f) = length(stored(f)[1])

# Scaled(Scaled(f, a1), a2), ...: the wrappers stay nested while the product is not a normal number
chain(f, as...) = foldl(Scaled, as; init = f)

bigarg(x::AbstractArray) = big.(x)
bigarg(x::Tuple) = map(bigarg, x)

accurate(v, expected, ::Type{T}, n, nf) where {T} =
  v isa T && isfinite(v) && abs(big(v) - expected) <= (n + nf + 2) * eps(T) * abs(expected)

# the exponents of the fixtures: the level and the argument of 10^l, three factors of 10^-a (down), or
# the level and the argument of 10^-l, three factors of 10^a (up); the value is about 10^-l or 10^l
fixtures(::Type{Float64}) = (a = 300, l = 300)
fixtures(::Type{Float32}) = (a = 30, l = 30)
tens(::Type{T}, k) where {T} = T(10)^k

# name, the function with the level λ, the argument at the scale s, and the value of the unit-level
# function at the stored argument
function families(::Type{T}) where {T}
  box = IndBox(T(-Inf), T(Inf))
  v3, g3 = T[3, -4, 1], [1:2, 3:3]
  ones2 = T[1, 1]
  w3 = T[1, 2, 4]
  m2 = T[3 0; 0 4]
  l1(x) = sum(abs, x)
  l2(x) = sqrt(sum(abs2, x))
  grp(x) = l2(x[1:2]) + abs(x[3])
  (
    ("NormL1", λ -> NormL1(λ), s -> s .* v3, l1),
    ("NormL1, weights", λ -> NormL1(λ, w3), s -> s .* v3, x -> sum(big.(w3) .* abs.(x))),
    ("NormL1Nonneg", λ -> NormL1Nonneg(λ), s -> s .* abs.(v3), sum),
    ("NormL1Nonneg, weights", λ -> NormL1Nonneg(λ, w3), s -> s .* abs.(v3), x -> sum(big.(w3) .* x)),
    ("NormL2", λ -> NormL2(λ), s -> s .* v3, l2),
    ("NormLinf", λ -> NormLinf(λ), s -> s .* v3, x -> maximum(abs, x)),
    ("ElasticNet(λ, 0)", λ -> ElasticNet(λ, zero(T)), s -> s .* v3, l1),
    ("GroupNormL2", λ -> GroupNormL2(λ, g3, ones2), s -> s .* v3, grp),
    ("GroupNormLinf", λ -> GroupNormLinf(λ, g3, ones2), s -> s .* v3, x -> maximum(abs, x[1:2]) + abs(x[3])),
    ("SparseGroupNormL2", λ -> SparseGroupNormL2(λ, λ, g3, ones2), s -> s .* v3, x -> l1(x) + grp(x)),
    ("NuclearNorm", λ -> NuclearNorm(λ), s -> s .* m2, x -> sum(abs, diag(x))),
    ("TracePSD", λ -> TracePSD(λ), s -> s .* m2, x -> sum(diag(x))),
    ("Translate", λ -> Translate(NormL1(λ), T[1, 0, 0]), s -> s .* v3 .+ T[1, 0, 0], l1 ∘ (x -> x .- [1, 0, 0])),
    ("Sum", λ -> Sum(NormL1(λ), NormL2(λ), box, ZeroFunction()), s -> s .* v3, x -> l1(x) + l2(x)),
    ("SeparableSum", λ -> SeparableSum(NormL1(λ), NormLinf(λ)), s -> (s .* T[3, -4], s .* T[1, -2]),
      x -> l1(x[1]) + maximum(abs, x[2])),
    ("Conjugate(IndBallL1)", λ -> Conjugate(IndBallL1(λ)), s -> s .* v3, x -> maximum(abs, x)),
    ("Conjugate(IndBallL2)", λ -> Conjugate(IndBallL2(λ)), s -> s .* v3, l2),
  )
end
# the number of terms that a family sums
terms(name) = name == "ElasticNet(λ, 0)" || startswith(name, "Sum") || startswith(name, "SparseGroup") ? 6 : 3

setprecision(BigFloat, 256) do

@testset "the issue and the controls (#568)" begin
  h = chain(NormL1(1e300), 1e-300, 1e-300, 1e-300)
  @test h isa Scaled{<:Scaled{<:Scaled}}
  @test accurate(h([1e300]), product(h) * big(1e300)^2, Float64, 1, 3)
  @test h([1e300]) ≈ 1e-300 rtol = 8eps()
  up = chain(NormL1(1e-300), 1e300, 1e300, 1e300)
  @test up([1e-300]) ≈ 1e300 rtol = 8eps()
  h32 = chain(NormL1(1f30), 1f-30, 1f-30, 1f-30)
  @test h32(Float32[1f30]) isa Float32
  @test h32(Float32[1f30]) ≈ 1f-30 rtol = 8eps(Float32)
  @test chain(NormL1(1f-30), 1f30, 1f30, 1f30)(Float32[1f-30]) ≈ 1f30 rtol = 8eps(Float32)
  # two factors whose product is a normal number (the level folds: a control that worked before)
  c = chain(NormL1(1e300), 1e-300, 1e-300)
  @test c([1e300]) ≈ 1.0 rtol = 8eps()
  c = chain(NormL1(1e-300), 1e300, 1e300)
  @test c([1e-300]) ≈ 1.0 rtol = 8eps()
  # the leaf value is a normal number: nothing to recover
  @test chain(NormL1(1.0), 1e-200, 1e-200)([1.0]) === 0.0
  @test chain(NormL1(1.0), 1e200, 1e200)([1.0]) === Inf
  # ordinary values and ordinary products
  o = Scaled(Scaled(NormL1(0.2), 3.0), 5.0)
  @test o isa Scaled{<:NormL1} && o([0.3, -0.7]) === o.α * o.f([0.3, -0.7])
  for n in (2, 3, 4), T in (Float32, Float64)
    f = chain(NormL1(T(1)), ntuple(_ -> T(0.5), n)...)
    @test f isa Scaled{<:NormL1} && f(T[1, -2]) === f.α * f.f(T[1, -2])
  end
end

@testset "families, both directions, $T" for T in (Float32, Float64)
  a, l = fixtures(T).a, fixtures(T).l
  for (name, build, arg, unit) in families(T)
    nt = terms(name)
    @testset "$name down" begin
      h = chain(build(tens(T, l)), tens(T, -a), tens(T, -a), tens(T, -a))
      x = arg(tens(T, l))
      expected = product(h) * big(tens(T, l)) * unit(bigarg(x))
      v = h(x)
      @test accurate(v, expected, T, nt, 3)
      @test abs(expected) > floatmin(T)
    end
    @testset "$name up" begin
      h = chain(build(tens(T, -l)), tens(T, a), tens(T, a), tens(T, a))
      x = arg(tens(T, -l))
      expected = product(h) * big(tens(T, -l)) * unit(bigarg(x))
      v = h(x)
      @test accurate(v, expected, T, nt, 3)
    end
  end
end

@testset "numbers of factors, sizes of factors and arguments" begin
  # two factors: the level is subnormal, or underflows, while the leaf value overflows
  h = Scaled(Scaled(NormL1(1e300), 1e-307), 1e-307)
  @test accurate(h([1e300]), product(h) * big(1e300)^2, Float64, 1, 2)
  h = Scaled(Scaled(NormL1(1.0), 1e-307), 1e-307)
  x = [1.7e308, 1.7e308]
  @test accurate(h(x), product(h) * sum(big.(x)), Float64, 2, 2)
  # four factors, a large one among the small ones
  h = chain(NormL1(1e300), 1e-300, 1e-300, 1e300, 1e-300, 1e-300)
  @test accurate(h([1e300]), product(h) * big(1e300)^2, Float64, 1, 5)
  h = chain(NormL1(1e-300), 1e300, 1e-300, 1e300, 1e300)
  @test accurate(h([1e-300]), product(h) * big(1e-300)^2, Float64, 1, 4)
  # an argument that is moderate: the value at the lowest level is near the edge of the range
  h = chain(NormL1(1e300), 1e-200, 1e-200, 1e-200)
  @test accurate(h([1e10]), product(h) * big(1e300) * big(1e10), Float64, 1, 3)
  @test accurate(h([1e10, -3e9, 2e10]), product(h) * big(1e300) * sum(abs, big.([1e10, -3e9, 2e10])), Float64, 3, 3)
  # weights and entries that differ in size
  w = [1.0, 2.0, 1e-5, 1e300]
  x = [1e300, -2e300, 3e300, 1e-100]
  h = chain(NormL1(1e300, w), 1e-300, 1e-300, 1e-300)
  @test accurate(h(x), product(h) * big(1e300) * sum(big.(w) .* abs.(big.(x))), Float64, 4, 3)
  # a subnormal result is rounded once from the exact value
  h = chain(NormL1(1e300), 1e-300, 1e-300, 1e-300)
  v = h([1e290])
  ref = product(h) * big(1e300) * big(1e290)
  @test 0 < v < floatmin(Float64) && abs(big(v) - ref) <= big(nextfloat(0.0))
  # an up chain whose value overflows
  @test chain(NormL1(1e-300), 1e300, 1e300, 1e300)([1e-20]) === Inf
  # a single factor with a level that its product leaves, and a subnormal level
  s = Scaled(GroupNormL2(1e-200, [1:1], [1e300]), 1e-200)
  @test accurate(s([1e300]), big(1e-200)^2 * big(1e300)^2, Float64, 1, 1)
  s = Scaled(ElasticNet(1e-200, 1e-200), 1e-200)
  @test accurate(s([1e300]), big(1e-200) * (big(1e-200) * big(1e300) + big(1e-200) * big(1e300)^2 / 2), Float64, 2, 1)
  s = Scaled(NormL1(1e-20, [1e300]), 1e-300)
  @test accurate(s([1e300]), big(1e-20) * big(1e300)^2 * big(1e-300), Float64, 1, 1)
  # the ridge level far below the others
  h = chain(ElasticNet(1e150, 1e-150), 1e-100, 1e-100, 1e-100, 1e-100, 1e-100)
  x = [1e160, 2e160]
  @test accurate(h(x), product(h) * (big(1e150) * sum(big.(x)) + big(1e-150) * sum(abs2, big.(x)) / 2), Float64, 4, 5)
end

@testset "subnormal folds are not used (#568)" begin
  # the fold would put the product into a subnormal level, which has fewer bits than the factors
  h = chain(NormL1(1e-20, [1e300]), 1e-100, 1e-100, 1e-100)
  @test accurate(h([1e300]), product(h) * big(1e-20) * big(1e300)^2, Float64, 1, 3)
  h = Scaled(Scaled(NormL1(1e300), 1e-307), 1e-307)
  @test accurate(h([1e300]), product(h) * big(1e300)^2, Float64, 1, 2)
  # an ordinary value whose product is subnormal is the product, rounded once
  h = Scaled(NormL1(1e-300), 1e-10)
  x = [2.0, 1.0]
  @test h(x) === h.α * h.f(x) && 0 < h(x) < floatmin(Float64)
  # no normal level gives a finite leaf value (the ridge level is far above the level of the l1 part),
  # while the fold into the subnormal level of the l1 part, which is negligible here, keeps the value
  h = Scaled(Scaled(ElasticNet(1.3176073f-21, 6.1296643f30), 1.4399453f-35), 2.8246963f11)
  x = Float32[-6.705103f12, 6.1984725f13, 1.628138f13]
  λ1, λ2 = big(1.3176073f-21), big(6.1296643f30)
  @test accurate(h(x), product(h) * (λ1 * sum(abs, big.(x)) + λ2 * sum(abs2, big.(x)) / 2), Float32, 6, 2)
  # the same in Float32: the product 1e-40 of the factors is subnormal, and the leaf value overflows
  h = chain(NormL1(1f0), 1f-20, 1f-20)
  x = Float32[3f38, 3f38]
  @test accurate(h(x), product(h) * sum(big.(x)), Float32, 2, 2)
end

@testset "what is not recovered" begin
  # levels whose exponents are too far apart for one normal fold, at an argument of that size
  h = chain(Sum(NormL1(1e300), NormL2(1e-300)), 1e-300, 1e-300, 1e-300)
  @test h([1e300]) === Inf
  # the same levels at a moderate argument, where the larger level is the term that matters
  h = chain(Sum(NormL1(1e300), NormL2(1e-300)), 1e-200, 1e-200, 1e-200)
  @test accurate(h([1e10]), product(h) * (big(1e300) * big(1e10) + big(1e-300) * big(1e10)), Float64, 2, 3)
  # no normal level gives a leaf value that is a finite normal number: weight and entry near floatmax
  h = chain(NormL1(1e300, [1e308]), 1e-300, 1e-300, 1e-300)
  @test h([1e308]) === Inf
  # an infinite level keeps its meaning: Inf at a nonzero argument, 0 at the zero argument
  h = chain(NormL1(Inf), 1e-300, 1e-300, 1e-300)
  @test h([1.0]) === Inf && h([0.0]) === 0.0
end

@testset "special values keep their meaning" begin
  h = chain(NormL1(1e300), 1e-300, 1e-300, 1e-300)
  @test isnan(h([NaN])) && isnan(h([NaN, 1e300]))
  @test h([Inf]) === Inf && h([-Inf, 1.0]) === Inf
  @test h([0.0]) === 0.0 && h(Float64[]) === 0.0
  up = chain(NormL1(1e-300), 1e300, 1e300, 1e300)
  @test up([0.0, 0.0]) === 0.0 && isnan(up([NaN])) && up([Inf]) === Inf
  # an infinite weight: Inf at a nonzero entry, 0 at a zero entry
  hw = chain(NormL1(1e300, [Inf, 1.0]), 1e-300, 1e-300, 1e-300)
  @test hw([1.0, 0.0]) === Inf && hw([0.0, 0.0]) === 0.0
  # a zero level always folds
  hz = chain(NormL1(0.0), 1e-300, 1e-300, 1e-300)
  @test hz([1e300]) === 0.0
  # outside the domain
  hn = chain(NormL1Nonneg(1e300), 1e-300, 1e-300, 1e-300)
  @test hn([-1.0]) === Inf && accurate(hn([1e300]), product(hn) * big(1e300)^2, Float64, 1, 3)
  # an outer factor that is zero or infinite keeps its meaning
  inf = Scaled(chain(NormL1(1e300), 1e-300, 1e-300), Inf)
  @test inf([1.0]) === Inf && inf([0.0]) === 0.0
  @test Scaled(chain(NormL1(1e300), 1e-300, 1e-300), 0.0) isa ZeroFunction
  # Float32 data under a Float64 chain: the value is out of the range of Float32
  h4 = chain(NormL1(1e300), 1e-300, 1e-300, 1e-300, 1e-300)
  @test h4(Float32[3f38]) === 0f0
  h3 = chain(NormL1(1e30), 1e-30, 1e-30, 1e-30)
  @test h3(Float32[3f38]) isa Float32
  @test h3(Float32[3f38]) ≈ 3f-22 rtol = 8eps(Float32)
  # integers as the argument
  @test chain(NormL1(1e300), 1e-300, 1e-300, 1e-300)([0, 0]) === 0.0
end

@testset "arguments of other types" begin
  h = chain(NormL1(1e300), 1e-300, 1e-300, 1e-300)
  x = [1e300, -2e300, 0.0]
  expected = product(h) * big(1e300) * 3big(1e300)
  @test accurate(h(view(x, 1:2)), expected, Float64, 2, 3)
  @test accurate(h(SparseVector(3, [1, 2], [1e300, -2e300])), expected, Float64, 2, 3)
  @test accurate(h(SparseIterate([1e300, -2e300, 0.0])), expected, Float64, 2, 3)
  @test accurate(h(reshape(x, 1, 3)), expected, Float64, 2, 3)
  @test accurate(h(SymmetricSparseIterate([1e300 0.0; 0.0 2e300])), expected, Float64, 2, 3)
  @test accurate(h(SymmetricSparseIterate([1e300 1e300; 1e300 0.0])), expected, Float64, 3, 3)
  @test accurate(h(1e300:1e300), product(h) * big(1e300)^2, Float64, 1, 3)
  # device arrays read only scalars here
  @test accurate(h(JLArray(x)), expected, Float64, 2, 3)
  g = chain(SeparableSum(NormL1(1e300), NormL2(1e300)), 1e-300, 1e-300, 1e-300)
  @test accurate(g((x[1:2], [3e300, 4e300])), product(g) * big(1e300) * (3big(1e300) + 5big(1e300)), Float64, 3, 3)
  w = chain(NormL1(1e300, JLArray([1.0, 2.0, 3.0])), 1e-300, 1e-300, 1e-300)
  @test accurate(w(JLArray(x)), product(w) * big(1e300) * (big(1e300) + 4big(1e300)), Float64, 3, 3)
end

@testset "the value at the stored output of value_and_prox!" begin
  # the proximal point is that of prox!, and the value is that of the chain at it
  function check(h, x, γ, label)
    @testset "$label" begin
      y, z = similar(x, Float64), similar(x, Float64)
      v = value_and_prox!(y, h, x, γ)
      prox!(z, h, x, γ)
      @test y == z
      leaf = stored(h)[2]
      expected = product(h) * big(leaf.λ0) * sum(abs, big.(y))
      @test accurate(v, expected, Float64, length(x), nfactors(h))
      @test accurate(h(y), expected, Float64, length(x), nfactors(h))
    end
  end
  h = chain(NormL1(1e300), 1e-300, 1e-300, 1e-300)
  x = [3e300, -2e300, 1e299]
  # the step folds into the levels (the product of step and factors is a normal number)
  check(h, x, 1e300, "scalar step with the factors in range")
  # the step is one: its product with the factors underflows, the proximal point is the argument
  check(h, x, 1.0, "scalar step 1")
  # array steps: the levels take a shifted exponent, or the steps are rounded with the factors
  steps = fill(1e300, 3)
  @test PB._array_fold(PB._scaled_chain(h)..., steps)[3] === :shifted
  check(h, x, steps, "array step, shifted")
  steps = [1e300, 1e-300, 1e300]
  @test PB._array_fold(PB._scaled_chain(h)..., steps)[3] === :rounded
  check(h, x, steps, "array step, rounded")
  # a single factor: a scalar step, and an array step inside the band of factors
  s = Scaled(NormL1(1e-250, [1e300]), 1e-70)
  y = zeros(1)
  v = value_and_prox!(y, s, [1e300], 1.0)
  @test accurate(v, big(1e-250) * big(1e-70) * big(1e300) * big(y[1]), Float64, 1, 1)
  v = value_and_prox!(y, s, [1e300], [1.0])
  @test accurate(v, big(1e-250) * big(1e-70) * big(1e300) * big(y[1]), Float64, 1, 1)
  # and a single factor outside the band, which takes the route of a chain
  s = Scaled(NormL1(1e-250, [1e300]), 1e-200)
  v = value_and_prox!(y, s, [1e300], [1.0])
  @test accurate(v, big(1e-250) * big(1e-200) * big(1e300) * big(y[1]), Float64, 1, 1)
  # Float32
  h32 = chain(NormL1(1f30), 1f-30, 1f-30, 1f-30)
  x32 = Float32[3f30, -2f30]
  for γ in (1f30, 1f0)
    y32 = zeros(Float32, 2)
    v = value_and_prox!(y32, h32, x32, γ)
    @test accurate(v, product(h32) * big(1f30) * sum(abs, big.(y32)), Float32, 2, 3)
  end
end

@testset "ordinary calls: values, products and allocations" begin
  a_value(f, x) = f(x)
  x = [0.3, -0.7, 1.1]
  for f in (Scaled(NormL1(0.5), 2.0), Scaled(Scaled(NormL1(0.5), 2.0), 3.0), chain(NormL1(0.5), 1e-200, 1e-200),
            chain(NormL2(0.5), 1e-100, 1e-100, 1e-100), chain(NormL1(1.0), 1e100, 1e100))
    @test allocs(a_value, f, x) == 0
  end
  # a chain with an exceptional product folds to the value of the folded level, as before
  h = chain(NormL1(1e300), 1e-300, 1e-300)
  c = PB._folded_level(1e300, PB._scaled_chain(h)[2])
  @test h([1e300]) === NormL1(c)([1e300])
end

end   # setprecision

end
