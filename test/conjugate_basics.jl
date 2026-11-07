module ConjugateBasicsTest

# Conjugates computed as projections: one function per direct path (clipping, check loss,
# orthant, l2 ball, group l2 ball, l1 ball, group l1 ball), Moreau and biconjugate identities,
# nested wrappers, array steps, aliasing. The exhaustive grids are in conjugate_projections.jl.

using Test, ProximalBase, LinearAlgebra

# Build every intermediate at the reference precision, including coefficients.
weight(w, i) = w === nothing ? BigFloat(1) : BigFloat(w[i])
coefficient(α, λ, w) = isinf(w) ? BigFloat(Inf) :
  iszero(λ) || iszero(w) ? BigFloat(0) : α * BigFloat(λ) * w
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
  elseif f isa NormLinf
    return projl1(X, α * BigFloat(f.λ))
  elseif f isa GroupNormLinf
    Y = zero(X)
    for (k, G) in enumerate(f.groups)
      Y[G] .= projl1(X[G], coefficient(α, f.λ0, BigFloat(f.w[k])))
    end
    return Y
  elseif f isa NormL2
    n, r = norm(X), α * BigFloat(f.λ)
    return iszero(n) ? X : X .* min(BigFloat(1), r / n)
  elseif f isa GroupNormL2
    Y = zero(X)
    for (k, G) in enumerate(f.groups)
      isempty(G) && continue
      r = coefficient(α, f.λ0, BigFloat(f.w[k]))
      n = norm(X[G])
      Y[G] .= iszero(n) ? X[G] : X[G] .* min(BigFloat(1), r / n)
    end
    return Y
  end
  error("missing reference")
end

function projl1(x, r)
  X = BigFloat.(x)
  isinf(r) && return X
  iszero(r) && return zero(X)
  norm(X, 1) <= r && return X
  u = sort(abs.(X); rev = true)
  sums = cumsum(u)
  k = findlast(j -> u[j] > (sums[j] - r) / j, eachindex(u))
  θ = (sums[k] - r) / k
  sign.(X) .* max.(abs.(X) .- θ, 0)
end

function l1accurate(y, ref, r, n = length(y); support = true)
  T = eltype(y)
  allowance = BigFloat(nextfloat(zero(T))) / 2
  bound = 4n * BigFloat(eps(T)) * r
  for i in eachindex(y)
    support && iszero(ref[i]) && @test iszero(y[i])
    @test abs(BigFloat(y[i]) - ref[i]) <= bound + allowance
  end
  @test norm(BigFloat.(y), 1) <= r * (1 + 4n * BigFloat(eps(T))) + n * 2allowance
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
      c = coefficient(α, f.λ0, BigFloat(f.w[k]))
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
    if iszero(ref[i])
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
      families = (NormL1(), CheckLoss(T(0.3)), IndNonnegative(), NormL2(),
                  GroupNormL2(T(1), [[1, 2], [4]], T[1, 0]), Scaled(NormL2(huge), tiny),
                  Translate(NormL1(), T[0, 0, 1, 0, 0]))
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
    end
  end
end

@testset "conjugate projections retain infinite-weight domains" begin
  for T in (Float64,), λ in (zero(T), T(0.5), T(Inf)),
      w in (T[1, Inf, 0], T[Inf, Inf, Inf], T[0, 1, 2], T[floatmax(T) / 4, 1, Inf])
    x = T[3, 4, 1, -2, 0.5, 0.5, 7]
    for groups in ([1:2, 3:4, 5:6], [[1, 3], [4, 5], [6]])
      base = GroupNormL2(λ, groups, w)
      functions = isfinite(λ) ? (base, Scaled(base, zero(T)), Scaled(Scaled(base, T(2)), zero(T))) : (base,)
      for f in functions, γ in (T(0.5), one(T))
        expected = x .- γ .* prox(f, x ./ γ, inv(γ))
        c = Conjugate(f)
        for op in (prox!, value_and_prox!), atom in (false, true), alias in (false, true)
          # The atom partition also represents holes and trailing coordinates.
          source = atom ? AtomIterate(x, [[1, 2], [3, 4], [5, 6, 7]]) : copy(x)
          y = alias ? source : copy(source)
          v = op(y, c, source, γ)
          @test Vector(y) ≈ expected rtol = 8eps(T) atol = 8eps(T)
          @test all(i -> any(G -> i in G, groups) || iszero(y[i]), eachindex(x))
          op === value_and_prox! && @test v == 0
          for k in eachindex(groups)
            isinf(w[k]) && @test Vector(y)[groups[k]] == x[groups[k]]
          end
        end
      end
    end
    # Array steps for the group fallback require a step per group and entry.
    groups = [[1], [2], [3]]
    for base in (GroupNormL2(λ, groups, w), NormL1(λ, w), NormL1Nonneg(λ, w))
      functions = isfinite(λ) ? (base, Scaled(base, zero(T))) : (base,)
      isinf(λ) && @test_throws ArgumentError Scaled(base, zero(T))
      for f in functions, γ in (T(0.5), T[0.5, 1, 2])
        input = T[3, -4, 2]
        expected = input .- γ .* prox(f, input ./ γ, inv.(γ))
        c = Conjugate(f)
        atoms = base isa GroupNormL2 ? (false, true) : (false,)
        for op in (prox!, value_and_prox!), atom in atoms, alias in (false, true)
          source = atom ? AtomIterate(input, groups) : copy(input)
          y = alias ? source : copy(source)
          v = op(y, c, source, γ)
          @test Vector(y) ≈ expected rtol = 8eps(T) atol = 8eps(T)
          op === value_and_prox! && @test v == 0
        end
        if isseparable(f)
          @test T[prox_entry(c, input[i], i, γ isa Real ? γ : γ[i]) for i in eachindex(input)] ≈ expected
        end
      end
    end
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

@testset "projection output aliases retained parameters" begin
  for op in (prox!, value_and_prox!), α in (1.0, 2.0)
    x = [3.0, 2.0]
    f = Scaled(GroupNormL2(0.5, [[2], [1]], x), α)
    expected = α == 1 ? [1.0, 1.5] : [2.0, 2.0]
    v = op(x, Conjugate(f), x, 1.0)
    @test x == expected
    op === value_and_prox! && @test v == 0
    for F in (NormL1, NormL1Nonneg), alias_input in (false, true), γ in (1.0, ones(2))
      y = [3.0, 2.0]
      x = alias_input ? y : [9.0, 9.0]
      f = Scaled(F(0.5, view(y, 2:-1:1)), α)
      expected = min.(copy(x), (α * 0.5) .* reverse(copy(y)))
      v = op(y, Conjugate(f), x, γ)
      @test y == expected
      op === value_and_prox! && @test v == 0
    end
    y = [1.0, 2.0]
    v = op(y, Conjugate(IndBox(-copy(y), y)), [3.0, 4.0], 1.0)
    @test y == [2.0, 2.0]
  end
end

@testset "double conjugate tuples and nested step checks" begin
  for T in (Float32, Float64), op in (prox!, value_and_prox!), alias in (false, true)
    f = SeparableSum(SqrNormL2(T(2)), SeparableSum(NormL1(T(0.5))))
    c = Conjugate(Conjugate(f))
    input = (T[3, -4], (T[2, -1],))
    expected = (T[1.5, -2], (T[1.75, -0.75],))
    x = deepcopy(input)
    y = alias ? x : deepcopy(input)
    value = op(y, c, x, T(0.5))
    @test y == expected
    @test prox(c, input, T(0.5)) == expected
    op === value_and_prox! && @test value == T(7.5)

    # The outer sum validates the double conjugate's tuple step as a block.
    s = SeparableSum(c, NormL1())
    x = (deepcopy(input), T[3])
    y = alias ? x : deepcopy(x)
    value = op(y, s, x, T(0.5))
    @test y == (expected, T[2.5])
    op === value_and_prox! && @test value == T(10)
  end
end

@testset "L1 conjugate radius accuracy" begin
  setprecision(BigFloat, 4096) do
    for T in (Float64,)
      p = precision(T)
      h = T === Float64 ? T(1e300) : T(3e30)
      tiny = T === Float64 ? T(1e-300) : T(1e-30)
      d = T(2)^(1-p)
      cases = ((T[h, 0.25, -0.5], one(T)),
               (T[h, tiny, h/3], one(T)),
               (T[1, T(0.25) + T(2)^(-p-1)], T(0.75)),
               (T[1, 1-d], d + T(2)^(2-2p)),
               (T[1, T(2)^(-p-7)], one(T)),
               (T[1, 1, 1], 2nextfloat(zero(T))),
               (T[1, 1, 1], nextfloat(zero(T))),
               (T[1, -1, 0.25], zero(T)),
               (T[tiny, -tiny/2, 0], tiny),
               (T[0.25, -0.25], prevfloat(T(0.5))),
               (T[0.25, -0.25], T(0.5)),
               (T[0.25, -0.25], nextfloat(T(0.5))),
               (T[floatmax(T)/2, floatmax(T)/4, floatmax(T)/4], floatmax(T)/2))
      for (input, r) in cases, x in (input, reverse(input)), op in (prox!, value_and_prox!)
        f = NormLinf(r)
        ref = reference(f, x, one(T))
        c = Conjugate(f)
        for alias in (false, true)
          source = copy(x)
          y = alias ? source : similar(source)
          v = op(y, c, source, one(T))
          l1accurate(y, ref, BigFloat(r); support = false)
          norm(BigFloat.(x), 1) <= BigFloat(r) && @test y == x
          op === value_and_prox! && @test v == 0
        end
      end
      for v in (prevfloat(T(0.5)), T(0.5), nextfloat(T(0.5)))
        x = T[1, v, -v]
        for r in (prevfloat(one(T)), one(T), nextfloat(one(T)))
          l1accurate(prox(Conjugate(NormLinf(r)), x), projl1(x, BigFloat(r)), BigFloat(r); support = false)
        end
      end
      for α in (tiny, T(2), h)
        f = Scaled(NormLinf(one(T)), α)
        x = T[h, 1, -2]
        l1accurate(prox(Conjugate(f), x), reference(f, x, 1), BigFloat(α))
      end
      # A finite mathematical radius beyond the working exponent range.
      f = Scaled(NormLinf(T(2)^exponent(floatmax(T))), T(2))
      x = fill(T(3)*T(2)^(exponent(floatmax(T))-1), 2)
      r = BigFloat(f.α) * BigFloat(f.f.λ)
      l1accurate(prox(Conjugate(f), x), reference(f, x, 1), r)
      @test isequal(prox(Conjugate(NormLinf(T(Inf))), T[Inf, NaN, 2]), T[Inf, NaN, 2])
    end
  end
end

@testset "L1 conjugate groups and wrappers" begin
  setprecision(BigFloat, 4096) do
    for T in (Float64,), λ in (zero(T), T(0.5)), w in (T[0, Inf, 1e30], T[1, 2, 0])
      x = T[1e30, -0.5, 4, 7, 0.25, -2, 8]
      f = GroupNormLinf(λ, [[1, 3], [4], [6]], w)
      for op in (prox!, value_and_prox!), alias in (false, true)
        y = copy(x)
        source = alias ? y : x
        v = op(y, Conjugate(f), source, one(T))
        ref = reference(f, x, 1)
        for (k, G) in enumerate(f.groups)
          r = coefficient(BigFloat(1), λ, BigFloat(w[k]))
          isinf(r) ? (@test y[G] == x[G]) : l1accurate(y[G], ref[G], r)
        end
        @test all(iszero, y[[2, 5, 7]])
        op === value_and_prox! && @test v == 0
      end
    end
    x, b = [1e300, -3.0, 1.0], [0.5, -0.25, 0.125]
    for base in (NormLinf(0.5),),
        f in (Translate(Scaled(base, 2.0), b), Scaled(Translate(base, b), 2.0)),
        op in (prox!, value_and_prox!)
      y = zeros(3)
      v = op(y, Conjugate(f), x, 0.5)
      ref = reference(f, x, 0.5)
      @test maximum(abs.(BigFloat.(y) - ref)) <= 12eps(Float64)
      op === value_and_prox! && @test v ≈ dot(b, y)
    end
  end
end

end
