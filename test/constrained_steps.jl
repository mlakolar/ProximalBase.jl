module ConstrainedStepsTest

using Test
using ProximalBase
using LinearAlgebra
using Random

const PB = ProximalBase

# Enumerate the affine pieces independently, in high precision. Each candidate
# must reproduce every scalar proximal point and meet the active bound.
function reference(x, γ, a, l, u, λ1, λ2, bl, bu)
  G = γ isa Real ? fill(BigFloat(γ), length(x)) : BigFloat.(γ)
  _reference(BigFloat.(x), G, BigFloat.(a), BigFloat(l), BigFloat(u),
    BigFloat(λ1), BigFloat(λ2), BigFloat(bl), BigFloat(bu))
end

# Convert before enumeration so its arithmetic compiles once for all fixtures.
function _reference(X, G, A, L, U, α, β, lo, hi)
  D = 1 ./ (1 .+ G .* β)
  scalar(v, i) = clamp(sign(v) * max(abs(v) - G[i] * α, 0) * D[i], lo, hi)
  point(μ) = [scalar(X[i] - G[i] * μ * A[i], i) for i in eachindex(X)]
  p = point(BigFloat(0))
  L <= dot(A, p) <= U && return p
  b = dot(A, p) < L ? L : U
  side = L == U ? 0 : dot(A, p) < L ? -1 : 1
  pieces = [(zero(BigFloat), lo), (zero(BigFloat), hi), (zero(BigFloat), clamp(BigFloat(0), lo, hi))]
  choices = [k for k in 1:5 if k == 4 || (k == 5 && α > 0) || (k == 1 && isfinite(lo)) || (k == 2 && isfinite(hi)) || (k == 3 && α > 0)]
  for states in Iterators.product(ntuple(_ -> choices, length(X))...)
    d, h = similar(X), similar(X)
    for i in eachindex(X)
      k = states[i]
      d[i], h[i] = k <= 3 ? pieces[k] : (D[i], (k == 4 ? -1 : 1) * G[i] * α * D[i])
    end
    all(isfinite, h) || continue
    w = sum(G .* A.^2 .* d)
    iszero(w) && continue
    μ = (dot(A, d .* X .+ h) - b) / w
    side * μ >= 0 || continue
    z = d .* (X .- G .* μ .* A) .+ h
    p = point(μ)
    tol = 1024eps(BigFloat) * max(BigFloat(1), maximum(abs, X), maximum(abs, z))
    maximum(abs, p - z) <= tol && abs(dot(A, p) - b) <= tol * max(1, sum(abs, A)) && return p
  end
  nothing
end

function check_case(f, set, x::Vector{T}, γ, a, l, u, pars; c = nothing) where {T}
  # Share the assertion code across function and constraint types.
  @nospecialize f set γ pars c
  c === nothing && (c = Constrained(f, set))
  ref = reference(x, γ, a, l, u, pars...)
  huge = (γ isa Real ? γ : maximum(γ)) >= T(1e20)
  indicator = pars[1] == 0 && pars[2] == 0
  for op in (prox!, value_and_prox!)
    y = fill(T(19), length(x))
    try
      op(y, c, x, γ)
      @test ref !== nothing
      @test all(isfinite, y)
      s, m = dot(a, y), sum(abs, a .* y)
      @test !isfinite(l) || s >= l - 8eps(T) * (m + abs(l))
      @test !isfinite(u) || s <= u + 8eps(T) * (m + abs(u))
      if ref !== nothing
        @test maximum(abs, BigFloat.(y) - ref) <= 64eps(T) * max(1, maximum(abs, ref), maximum(abs, x))
      end
    catch err
      @test err isa ArgumentError
      @test ref === nothing || (huge && !indicator)
      @test occursin("working precision", sprint(showerror, err))
    end
  end
end

@testset "extreme linear constrained steps" begin
  setprecision(BigFloat, 4096) do
    for T in (Float64, Float32)
      x, a = T[0.3, -1.2, 2], ones(T, 3)
      tiny = T === Float64 ? T(1e-310) : T(1e-40)
      large = T === Float64 ? T(1e300) : T(1e30)
      nearmax = T === Float64 ? T(1e308) : floatmax(T)
      smallnormal = T === Float64 ? T(1e-300) : floatmin(T)
      gs = (nextfloat(zero(T)), tiny, floatmin(T), smallnormal, one(T), large, nearmax,
            fill(tiny, 3), fill(large, 3), T[floatmin(T), 1, 1])
      fs = ((IndBox(T(-1), T(1)), (0, 0, -1, 1)),
            (Conjugate(NormL1(one(T))), (0, 0, -1, 1)),
            (NormL1(one(T)), (1, 0, -Inf, Inf)),
            (SqrNormL2(T(2)), (0, 2, -Inf, Inf)),
            (ElasticNet(T(0.3), T(0.8)), (T(0.3), T(0.8), -Inf, Inf)))
      sets = ((IndHyperplane(one(T), one(T)), 1, 1),
              (IndHalfspace(one(T), -one(T)), -Inf, -1),
              (IndHalfspace(one(T), T(10)), -Inf, 10),
              (IndHyperslab(one(T), T(1), T(2)), 1, 2),
              (IndHyperslab(one(T), T(-2), T(-1)), -2, -1),
              (IndHyperslab(one(T), T(-10), T(10)), -10, 10))
      for (f, pars) in fs, γ in gs
        check_case(f, first(sets)[1], x, γ, a, one(T), one(T), pars)
      end
      for (j, (set, l, u)) in enumerate(sets[2:end]), γ in (tiny, one(T))
        f, pars = fs[j]
        check_case(f, set, x, γ, a, T(l), T(u), pars)
      end
      for (f, pars) in fs, γ in (tiny, one(T), large)
        check_case(f, IndSimplex(one(T)), x, γ, a, one(T), one(T), (pars[1], pars[2], max(0, pars[3]), pars[4]))
      end
      for γ in (tiny, one(T), large)
        mixed = T[1, -0.5, 0]
        check_case(fs[1][1], IndHyperplane(mixed, T(0.2)), x, γ, mixed, T(0.2), T(0.2), fs[1][2])
      end
      c = Constrained(IndBox(T(-1), T(1)), IndHyperplane(one(T), one(T)))
      for γ in (one(T), large, tiny, one(T), fill(large, 3), T[floatmin(T), 1, 1])
        check_case(c.f, c.set, x, γ, a, one(T), one(T), (0, 0, -1, 1); c)
      end
      c = Constrained(IndBox(T(-1), T(1)), IndHyperslab(one(T), T(-1), T(1)))
      for (xx, γ) in ((x, tiny), (-x, large), (x, one(T)), (-x, tiny))
        check_case(c.f, c.set, xx, γ, a, -one(T), one(T), (0, 0, -1, 1); c)
      end
      check_case(IndBox(T(-1), T(1)), IndHyperplane(one(T), T(4)), x, tiny, a, T(4), T(4), (0, 0, -1, 1))
    end
    check_case(NormL1(0.3), IndHyperplane([1.0, -1, 0, 0.5, 1], 0.2),
      [0.3, -1.2, 2, 0.5, -0.7], [0.5, 1, 2, 1, 0.3], [1.0, -1, 0, 0.5, 1], 0.2, 0.2, (0.3, 0, -Inf, Inf))
    x = [0.3, -1.2, 2.0]
    r = reference(x, 1e-310, ones(3), 1, 1, 1, 0, -Inf, Inf)
    r2 = setprecision(BigFloat, 8192) do
      reference(x, 1e-310, ones(3), 1, 1, 1, 0, -Inf, Inf)
    end
    @test maximum(abs, r - r2) <= big"1e-1100"
  end
end

@testset "reported regressions" begin
  x = [0.3, -1.2, 2.0]
  for f in (Conjugate(NormL1()), IndBox(-1, 1))
    @test prox(Constrained(f, IndHyperplane(1, 1)), x, 1e-310) ≈ [0.75, -0.75, 1]
  end
  for γ in (1e-310, nextfloat(0.0))
    @test prox(Constrained(NormL1(), IndHyperplane(1, 1)), x, γ) ≈ [4 / 15, -37 / 30, 59 / 30]
  end
  @test prox(Constrained(SqrNormL2(), IndHalfspace(1, 1)), x, 1e-310) ≈ [4 / 15, -37 / 30, 59 / 30]
  @test prox(Constrained(IndBox(-1, 1), IndHalfspace(ones(3), -1)), x, 1e-310) ≈ [-0.85, -1, 0.85]
  @test_throws ArgumentError prox(Constrained(IndBox(-1, 1), IndHyperplane(1, 1)), x, [1e-310, 1, 1])
  c = Constrained(IndBox(-1, 1), IndHyperplane(1, 1))
  for γ in (1.0, 1e300, 1.0, fill(1e300, 3))
    @test prox(c, x, γ) ≈ [0.75, -0.75, 1]
  end
  for γ in (1e300, 1e308)
    @test_throws ArgumentError prox(Constrained(NormL1(), IndHyperplane(1, 1)), x, γ)
  end
end

@testset "invalid steps leave output unchanged" begin
  for T in (Float32, Float64), op in (prox!, value_and_prox!), alias in (false, true)
    x = T[0.3, -1.2, 2]
    for γ in (zero(T), -one(T), T(Inf), T(NaN), T[1, 0, 1], T[1, -1, 1],
              T[1, Inf, 1], T[1, NaN, 1], T[prevfloat(floatmin(T)), 1, 1],
              T[nextfloat(zero(T)), floatmax(T), 1])
      c = Constrained(IndBox(-1, 1), IndHyperplane(1, 1))
      y = alias ? copy(x) : fill(T(19), 3)
      before = copy(y)
      @test_throws ArgumentError op(y, c, alias ? y : x, γ)
      @test y == before
    end
  end
  @test_throws ArgumentError prox(Constrained(IndBox(-1, 1), IndHyperplane(1, 1)), [0.3, -1.2, 2], [1e-300, 1e300, 1])
end

@testset "subnormal ratios and mixed precision" begin
  x = [1e-10, 1e-10, 0.0]
  c = Constrained(IndBox(0, 1), IndHyperplane([1e10, 1e10, 0.0], 1))
  for γ in ([2.0^-50, 1.5 * 2.0^-50, floatmax(Float64)],
            [2.0^-51, 1.0, floatmax(Float64)]), op in (prox!, value_and_prox!), alias in (false, true)
    y = alias ? copy(x) : fill(19.0, 3)
    before = copy(y)
    @test_throws ArgumentError op(y, c, alias ? y : x, γ)
    @test y == before
  end
  setprecision(BigFloat, 4096) do
    # The minimum normal ratio is accepted, including after exact rescaling.
    for T in (Float32, Float64), scale in (one(T), T(2), T(0.5))
      γ = T[floatmin(T), 1] .* scale
      check_case(IndBox(T(0), T(1)), IndHyperplane(one(T), one(T)),
        T[0.3, 0.3], γ, ones(T, 2), one(T), one(T), (0, 0, 0, 1))
    end
  end
  for γ in (1 + 2.0^-25, [1 + 2.0^-25]), op in (prox!, value_and_prox!)
    c = Constrained(LinearFunction([1e8]), IndHyperslab(1, 1, 3))
    y = zeros(Float32, 1)
    op(y, c, [1e8 + 2], γ)
    @test y == Float32[2]
    @test 1 <= only(y) <= 3
  end
end

@testset "symmetric constrained outputs" begin
  setprecision(BigFloat, 4096) do
    X = [0.0 1.0; 1.0 0.0]
    for (set, l, u) in ((IndHyperplane(1, 1), 1, 1),
                        (IndHalfspace(1, 1), -Inf, 1),
                        (IndHyperslab(1, 0.5, 0.6), 0.5, 0.6),
                        (IndHyperslab(1, 1.7, 1.9), 1.7, 1.9)),
        γ in (1.0, [0.5 1.0; 1.0 0.75])
      steps = γ isa Real ? γ : vec(γ)
      ref = reshape(Float64.(reference(vec(X), steps, ones(4), l, u, 0.1, 0, -Inf, Inf)), 2, 2)
      c = Constrained(NormL1(0.1), set)
      @test Matrix(prox(c, SymmetricSparseIterate(X), γ)) ≈ ref atol = 1e-15 rtol = 64eps(Float64)
      for op in (prox!, value_and_prox!), alias in (false, true)
        x = SymmetricSparseIterate(X)
        y = alias ? x : similar(x)
        result = op(y, c, x, γ)
        @test Matrix(y) ≈ ref atol = 1e-15 rtol = 64eps(Float64)
        s, m = sum(y), sum(abs, y)
        @test !isfinite(l) || s >= l - 8eps(Float64) * (m + abs(l))
        @test s <= u + 8eps(Float64) * (m + abs(u))
        @test op === prox! ? result === y : result ≈ 0.1sum(abs, ref)
        alias || @test Matrix(x) == X
      end
    end
  end
end

@testset "inactive input promotion" begin
  for (T, U) in ((Float64, Float32), (Float32, Float64)),
      γ in (1.0, [1.0]), op in (prox!, value_and_prox!)
    x = U[0.3]
    set = IndHyperslab(1, 0.2, 0.200000015)
    c = Constrained(NormL1(0.1), set)
    y = zeros(T, 1)
    result = op(y, c, x, γ)
    ref = T(prox_entry(c.f, x[1] - zero(T), 1, one(T)))
    @test y == [ref]
    @test set.l <= only(y) <= set.u
    @test op === prox! ? result === y : result ≈ c.f(y)
  end
end

@testset "unresolved output certificate" begin
  # Adjacent multipliers can straddle a root without resolving the point.
  for γ in (1.0, [1.0]), op in (prox!, value_and_prox!), alias in (false, true)
    c = Constrained(IndNonnegative(), IndHyperplane(1, 1e-10))
    x = [10.0]
    y = alias ? x : fill(19.0, 1)
    before = copy(y)
    @test_throws ArgumentError op(y, c, x, γ)
    @test y == before
    @test c.output.buf !== nothing
    @test prox(Constrained(IndNonnegative(), IndHyperplane(1, 1)), [10.0], γ) == [1.0]
  end
  for γ in (1.0, ones(2, 2)), op in (prox!, value_and_prox!), alias in (false, true)
    c = Constrained(IndNonnegative(), IndHyperplane(1, 1e-10))
    x = SymmetricSparseIterate([10.0 0.0; 0.0 10.0])
    y = alias ? x : SymmetricSparseIterate(fill(19.0, 2, 2))
    before = Matrix(y)
    @test_throws ArgumentError op(y, c, x, γ)
    @test Matrix(y) == before
    @test c.output.buf isa SymmetricSparseIterate
  end
end

function cold_oracle!(op::O, y, c, x, γ) where {O}
  c.last[] = NaN
  op(y, c, x, γ)
end
function cold_aliased_oracle!(op::O, y, c, x, γ) where {O}
  copyto!(y, x)
  cold_oracle!(op, y, c, y, γ)
end

@testset "verified output reuses its buffer" begin
  rng = Xoshiro(174)
  for (f, x, b) in ((IndNonnegative(), [10.0], 0.3),
                    (IndNonnegative(), SymmetricSparseIterate([10.0 0.0; 0.0 10.0]), 0.3),
                    (IndBox(0, 1), 3randn(rng, 5000), 2.0)),
      γ in (1.0, ones(length(x))), op in (prox!, value_and_prox!)
    c = Constrained(f, IndHyperplane(1, b))
    y = similar(x)
    cold_oracle!(op, y, c, x, γ)
    @test c.verify[]
    buffer = c.output.buf
    @test buffer !== nothing
    @test PB.Testing.allocations(cold_oracle!, op, y, c, x, γ) == 0
    @test PB.Testing.allocations(cold_aliased_oracle!, op, y, c, x, γ) == 0
    @test c.output.buf === buffer
    @test abs(sum(y) - b) <= 8eps(Float64) * (sum(abs, y) + abs(b))
  end
end

@testset "merged empty-input preflight" begin
  for op in (prox!, value_and_prox!), alias in (false, true)
    c = Constrained(NormL1(), IndHyperplane(1, 1e100))
    x = (Float32[3], Float64[])
    y = alias ? x : (Float32[19], Float32[])
    before = deepcopy(y)
    @test_throws ArgumentError op(y, SeparableSum(SqrNormL2(), c), x, 1.0)
    @test isequal(y, before)
  end
end

@testset "ratio rule and validation reductions" begin
  for T in (Float32, Float64), scale in (T(3), floatmax(T), nextfloat(zero(T)))
    γ = fill(scale, 3)
    g = PB._search_steps(γ, T)
    factor = inv(scale)
    expected = isfinite(factor) ? scale * factor : scale / scale
    @test PB._ratio(g, 1, T) == expected
    @test PB._ratio(PB._head(g, 2), 1, T) == expected
    @test PB._ratio(PB._search_steps(scale, T), 1, T) === one(T)
  end
end

@testset "full-pass fallback reuses its certificate" begin
  for γ in (1f0, Float32[1]), op in (prox!, value_and_prox!), alias in (false, true)
    c = Constrained(IndNonnegative(), IndHyperplane(1f0, 0.3f0))
    x = Float32[10]
    y = alias ? x : Float32[19]
    op(y, c, x, γ)
    gap, scale = abs(y[1] - 0.3f0), abs(y[1]) + 0.3f0
    @test 2eps(Float32) * scale < gap <= 8eps(Float32) * scale
    @test !c.verify[]
    @test c.output.buf === nothing
  end
end

@testset "ordinary passes and Float32 warm hits" begin
  for T in (Float32, Float64), scale in (one(T), floatmin(T), floatmax(T))
    steps = PB._search_steps(fill(scale, 2), T)
    for μ in (zero(T), one(T), nextfloat(zero(T)), floatmax(T))
      pass = PB._pass_steps(identity, steps, μ)
      ν = μ * steps.factor
      fast = isfinite(steps.factor) && isfinite(ν) && (abs(ν) >= floatmin(T) || iszero(μ))
      @test (pass isa PB._PassSteps) == fast
      expected = fast ? scale * ν : PB._ratio(steps, 1, T) * μ
      @test isequal(PB._shift(pass, μ, 1, T), expected)
    end
  end
  rng = Xoshiro(1)
  n = 10_000
  x, noise = randn(rng, Float32, n), randn(rng, Float32, n)
  for (f, set, b) in ((IndBox(0f0, 1f0), IndHyperplane(1f0, Float32(n ÷ 2)), Float32(n ÷ 2)),
                      (ElasticNet(0.3f0, 0.4f0), IndSimplex(Float32(n / 10)), Float32(n / 10)))
    c = Constrained(f, set)
    y = prox(c, x, 0.7f0)
    for j in 1:30
      xt = x .+ Float32(0.001j) .* noise
      prox!(y, c, xt, 0.7f0)
      @test !c.verify[]
      @test abs(sum(Float64, y) - b) <= 8eps(Float32) * (sum(abs, Float64.(y)) + b)
    end
  end
end

# Keep the range visible while counting the scalar oracles, as in the budgets.
const TAIL_CALLS = Ref(0)
struct TailRanged{F} <: SeparableFunction
  f::F
end
PB.prox_entry(g::TailRanged, v, i, γ) = (TAIL_CALLS[] += 1; prox_entry(g.f, v, i, γ))
PB.isconvex(::Type{<:TailRanged}) = true
PB._has_prox_range(::Type{<:TailRanged{F}}) where {F} = PB._has_prox_range(F)
PB._prox_range(g::TailRanged, i::Integer, ::Type{T}) where {T} = PB._prox_range(g.f, i, T)
PB._has_uniform_range(::Type{<:TailRanged{F}}) where {F} = PB._has_uniform_range(F)

@testset "Float32 tail residual reductions" begin
  rng = Xoshiro(3)
  n = 20_000
  x = randn(rng, Float32, n)
  a = randn(rng, Float32, n)
  a[1:7:end] .= 0
  γ = 0.2f0 .+ rand(rng, Float32, n)
  b = sum(min.(a, 0f0)) + 1f0
  f = TailRanged(IndBox(0f0, 1f0))
  ref = prox(Constrained(f, IndHyperplane(a, b)), x, γ)
  for scale in (2f0^-76, 2f0^76)
    c = Constrained(f, IndHyperplane(scale .* a, scale * b))
    y = prox(c, x, γ)
    @test c.live[6].buf[].compact
    @test isapprox(y, ref; rtol = 1e-4, atol = 1e-6 * max(1, norm(x)))
    products = Float64.(a) .* y
    @test abs(sum(products) - b) <= 8eps(Float32) * (sum(abs, products) + abs(b))
  end
end

@testset "scaled variable fixing and ridge slope" begin
  # This tail has no multiplier meeting 2eps, but meets the 8eps certificate.
  rng = Xoshiro(174)
  x = 3randn(rng, 5000)
  for γ in (1e-310, 1.0, fill(1e-310, 5000), 0.3 .+ rand(rng, 5000)), normal in (1.0, fill(1.0, 5000))
    c = Constrained(IndBox(0, 1), IndHyperplane(normal, 2))
    y = prox(c, x, γ)
    @test c.live[6].buf[].compact
    @test all(isfinite, y)
    @test abs(sum(y) - 2) <= 8eps(Float64) * (sum(abs, y) + 2)
    scale = γ isa Real ? γ : maximum(γ)
    factor = inv(scale)
    r = γ isa Real ? ones(length(x)) : isfinite(factor) ? γ .* factor : γ ./ scale
    @test y == [prox_entry(c.f, x[i] - r[i] * c.last[], i, γ isa Real ? γ : γ[i]) for i in eachindex(x)]
    z = similar(x)
    @test value_and_prox!(z, c, x, γ) == 0
    @test z ≈ y
  end
  f = SqrNormL2(2.0)
  z, d = PB._prox_entry_slope(f, 1.0, 1, 1e308)
  @test d == prox_entry(f, 1.0, 1, 1e308) == z
  @test d > 0
end

end
