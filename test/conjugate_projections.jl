module ConjugateProjectionsTest

using Test, ProximalBase, LinearAlgebra
using ..TestUtils

# Build every intermediate at the reference precision, including coefficients.
weight(w, i) = w === nothing ? BigFloat(1) : BigFloat(w[i])
coefficient(α, λ, w) = isinf(w) ? BigFloat(Inf) : α * BigFloat(λ) * w
function reference(f, x, γ, α = BigFloat(1))
  X = BigFloat.(x)
  if f isa Scaled
    return reference(f.f, x, γ, α * BigFloat(f.α))
  elseif f isa Translate
    return reference(f.f, X .- BigFloat(γ) .* BigFloat.(f.b), γ, α)
  elseif f isa Union{NormL1, NormL1Nonneg, CheckLoss}
    return map(eachindex(x)) do i
      w = weight(f.w, i)
      if f isa CheckLoss
        clamp(X[i], -coefficient(α, 1 - BigFloat(f.τ), w), coefficient(α, BigFloat(f.τ), w))
      else
        c = coefficient(α, f.λ0, w)
        f isa NormL1 ? clamp(X[i], -c, c) : min(X[i], c)
      end
    end
  elseif f isa ZeroFunction
    return zero(X)
  elseif f isa IndNonnegative
    return min.(X, 0)
  elseif f isa NormL2
    n, r = norm(X), α * BigFloat(f.λ)
    return iszero(n) ? X : X .* min(BigFloat(1), r / n)
  elseif f isa GroupNormL2
    Y = zero(X)
    for (k, G) in enumerate(f.groups)
      isempty(G) && continue
      r = iszero(f.λ0) || iszero(f.w[k]) ? BigFloat(0) : α * BigFloat(f.λ0) * BigFloat(f.w[k])
      n = norm(X[G])
      Y[G] .= iszero(n) ? X[G] : X[G] .* min(BigFloat(1), r / n)
    end
    return Y
  end
  error("missing reference")
end

function feasible(f, y, α = BigFloat(1))
  f isa Scaled && return feasible(f.f, y, α * BigFloat(f.α))
  f isa Translate && return feasible(f.f, y, α)
  Y = BigFloat.(y)
  slack(c) = isinf(c) ? BigFloat(Inf) : 8BigFloat(eps(eltype(y))) * abs(c) + 2BigFloat(nextfloat(zero(eltype(y))))
  if f isa Union{NormL1, NormL1Nonneg, CheckLoss}
    for i in eachindex(y)
      w = weight(f.w, i)
      if f isa CheckLoss
        l, u = -coefficient(α, 1 - BigFloat(f.τ), w), coefficient(α, f.τ, w)
        @test Y[i] >= l - slack(l) && Y[i] <= u + slack(u)
      else
        c = coefficient(α, f.λ0, w)
        @test (f isa NormL1 ? abs(Y[i]) : Y[i]) <= c + slack(c)
      end
    end
  elseif f isa NormL2
    c = α * BigFloat(f.λ)
    @test norm(Y) <= c + length(y) * slack(c)
  elseif f isa GroupNormL2
    for (k, G) in enumerate(f.groups)
      c = iszero(f.λ0) || iszero(f.w[k]) ? BigFloat(0) : α * BigFloat(f.λ0) * BigFloat(f.w[k])
      @test norm(Y[G]) <= c + length(G) * slack(c)
    end
    for i in eachindex(y)
      any(G -> i in G, f.groups) || @test iszero(y[i])
    end
  elseif f isa IndNonnegative
    @test all(<=(0), y)
  elseif f isa ZeroFunction
    @test all(iszero, y)
  end
end

function accurate(y, ref, ulps)
  T = eltype(y)
  for i in eachindex(y)
    r = T(ref[i])
    if iszero(r)
      @test iszero(y[i])
    else
      @test abs(BigFloat(y[i]) - ref[i]) <= ulps * BigFloat(eps(T)) * abs(ref[i]) + 2BigFloat(nextfloat(zero(T)))
    end
  end
end

@testset "direct conjugate projections" begin
  setprecision(BigFloat, 4096) do
    for T in (Float64, Float32)
      huge = T === Float64 ? T(1e300) : T(1e30)
      tiny = T === Float64 ? T(1e-300) : T(1e-30)
      x = T[huge, -huge / 3, 2, 0, tiny]
      families = (NormL1(), NormL1Nonneg(), CheckLoss(T(0.3)), ZeroFunction(), IndNonnegative(), NormL2(), NormL2(tiny),
                  NormL1(T(0), T[Inf, 0, 2, 1, Inf]), NormL1Nonneg(T(0), T[Inf, 0, 2, 1, Inf]),
                  NormL1Nonneg(T(2), T[0, Inf, 3, 1, 2]), NormL1(T(2), T[0, Inf, 3, 1, 2]), CheckLoss(T(0.3), T[0, Inf, 3, 1, 2]),
                  GroupNormL2(T(1), [[1, 2], [4]], T[1, 0]),
                  GroupNormL2(T(0), [[1, 2], [4]], T[Inf, 1]),
                  Scaled(NormL1(T(2)), tiny), Scaled(NormL1(tiny, fill(huge, 5)), tiny),
                  Scaled(NormL1(huge, fill(tiny, 5)), huge),
                  GroupNormL2(T(1), [[1, 2], [4]], T[Inf, 1]), Scaled(NormL2(tiny), huge),
                  Scaled(NormL2(huge), tiny), Translate(NormL1(), T[0, 0, 1, 0, 0]))
      for f in families, γ in (T === Float64 ? 1e-310 : 1e-40, 1.0, T === Float64 ? 1e300 : 1e30)
        c = Conjugate(f)
        ref = reference(f, x, γ)
        y = prox(c, x, γ)
        accurate(y, ref, 8)
        feasible(f, y)
        alias = copy(x)
        v = value_and_prox!(alias, c, alias, γ)
        @test alias == y
        f isa Translate || @test v == 0
        if isseparable(f)
          accurate(prox(c, x, fill(γ, length(x))), ref, 4)
          for i in eachindex(x)
            accurate(T[prox_entry(c, x[i], i, γ)], [ref[i]], 4)
          end
        end
      end
      # The true norm overflows, while every projected coordinate is finite.
      x = T[floatmax(T) * T(0.8), floatmax(T) * T(0.8), tiny]
      for λ in (one(T), floatmax(T) / 2)
        accurate(prox(Conjugate(NormL2(λ)), x, 1.0), reference(NormL2(λ), x, 1.0), 8)
      end
      # The factor underflows before multiplication by the large coordinate.
      x = T[huge, one(T)]
      f = Scaled(NormL2(tiny), tiny)
      accurate(prox(Conjugate(f), x, 1.0), reference(f, x, 1.0), 8)
    end
    x = [1e300, 0.5]
    @test prox(Conjugate(NormL2()), x, 1.0) == [1.0, 5e-301]
    accurate(prox(Conjugate(NormL2()), [1e3, 2e3], 7.0), reference(NormL2(), [1e3, 2e3], 7.0), 8)
    @test prox(Conjugate(NormL1()), [1e300, -3e299, 2, 0, 5e150], 1e-10) == [1, -1, 1, 0, 1]
    @test prox(Conjugate(NormL1Nonneg()), [1e300, 0.5, -2], 1.0) == [1, 0.5, -2]
    @test prox(Conjugate(Scaled(NormL1(), 2)), [1e300, 1], 1.0) == [2, 1]
    f = GroupNormL2(1.0, [[1, 2], [3]], [1.0, 1.0])
    @test prox(Conjugate(f), [1e300, 0.5, 0.25], 1.0) == [1, 5e-301, 0.25]
    @test prox!(zeros(Float32, 2), Conjugate(NormL2()), [1e39, 0.0], 1.0) == Float32[1, 0]
    @test prox(Conjugate(GroupNormL2(1.0, Vector{Int}[], Float64[])), [2.0, 3.0], 1.0) == zeros(2)
    @test_throws ArgumentError GroupNormL2(1.0, [Int[]], [1.0])
  end
end

@testset "nested mixed sums and failure atomicity" begin
  x = ([1e300, 2.0], ([3.0, 4.0], [2.0]))
  c = Conjugate(SeparableSum(NormL1(), SeparableSum(NormL2(), SqrNormL2())))
  y = prox(c, x, 1.0)
  @test y[1] == [1, 1]
  @test y[2][1] ≈ [0.6, 0.8]
  @test y[2][2] == [1.0]
  alias = (copy(x[1]), map(copy, x[2]))
  @test prox!(alias, c, alias, 1.0) == y
  x = ([1e300], ([3.0, 4.0], [1.0]))
  for op in (prox!, value_and_prox!), alias in (false, true)
    input = (copy(x[1]), map(copy, x[2]))
    out = alias ? input : (zeros(1), (zeros(2), zeros(1)))
    before = deepcopy(out)
    @test_throws ArgumentError op(out, c, input, 1e-310)
    @test out == before
  end
  # A linear conjugate value must not send direct blocks back through Moreau.
  c = Conjugate(Translate(NormL1Nonneg(), [0.5, -0.5]))
  y = zeros(2)
  @test value_and_prox!(y, c, [1e300, -2.0], 1e-310) == 1.5
  @test y == [1.0, -2.0]
  @test value_and_prox!(y, c, [1e300, -2.0], [1e-310, 1.0]) == 1.25
  @test y == [1.0, -1.5]
  f = Scaled(SeparableSum(NormL1(), SeparableSum(NormL2(), SqrNormL2())), 2.0)
  @test prox(Conjugate(f), ([1e300], ([3.0, 4.0], [1.0])), 1.0)[1] == [2.0]
  c = Conjugate(NormL1(1e100))
  y = Float32[7]
  @test_throws ArgumentError prox!(y, c, [1e39], 1.0)
  @test y == Float32[7]
  c = Conjugate(Translate(NormL1(), [1.0]))
  y = Float32[7]
  @test_throws ArgumentError prox!(y, c, Float32[1], 1e300)
  @test y == Float32[7]
end

@testset "warmed direct projection allocations" begin
  x, y = collect(1.0:8.0), zeros(8)
  for f in (NormL1(), NormL2(), NormL1(2.0, ones(8)), NormL1Nonneg(), CheckLoss(0.3),
            GroupNormL2(1.0, [1:2, 4:6], ones(2)), Scaled(NormL2(), 2.0), Translate(NormL1(), ones(8)))
    c = Conjugate(f)
    prox!(y, c, x, 1.0)
    @test allocs(prox!, y, c, x, 1.0) == 0
    @test allocs(value_and_prox!, y, c, x, 1.0) == 0
  end
end

end
