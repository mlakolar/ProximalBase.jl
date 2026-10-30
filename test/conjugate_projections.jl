module ConjugateProjectionsTest

using Test, ProximalBase, LinearAlgebra
using ..TestUtils

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

@testset "conjugate projections retain infinite-weight domains" begin
  for T in (Float32, Float64), λ in (zero(T), T(0.5), T(Inf)),
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


@testset "review regressions: wrapper scales, shifts and values" begin
  setprecision(BigFloat, 4096) do
    for (λ, a, x) in ((1e300, 1e-200, 1.0), (1e-300, 1e200, 1e200)),
        base in (NormL1(λ), NormL1Nonneg(λ), NormL2(λ), GroupNormL2(λ, [1:1], [1.0]))
      f = Scaled(SeparableSum(Scaled(base, a)), a)
      c = Conjugate(f)
      r = BigFloat(λ) * BigFloat(a)^2
      for op in (prox!, value_and_prox!)
        y = ([0.0],)
        v = op(y, c, ([x],), 1.0)
        accurate(y[1], [min(BigFloat(x), r)], 8)
        op === value_and_prox! && @test v == 0
      end
    end
    f = Scaled(Translate(Scaled(NormL1(1e300), 1e-200), [0.0]), 1e-200)
    r = BigFloat(1e300) * BigFloat(1e-200)^2
    @test prox_entry(Conjugate(f), 1.0, 1, 1.0) ≈ Float64(r) rtol = 4eps(Float64)
    f = Scaled(Translate(Scaled(NormL1(1e-300), 1e200), [0.0]), 1e200)
    r = BigFloat(1e-300) * BigFloat(1e200)^2
    @test prox_entry(Conjugate(f), 1e200, 1, 1.0) ≈ Float64(r) rtol = 4eps(Float64)
    for op in (prox!, value_and_prox!), base in (NormL1(), NormL1Nonneg(), NormL2(), GroupNormL2(1.0, [1:1], [1.0]))
      y = zeros(Float32, 1)
      v = op(y, Conjugate(Translate(base, [0.0])), [1e39], 1.0)
      @test y == Float32[1]
      op === value_and_prox! && @test v == 0
    end
    for (λ, a, input) in ((1e300, 1e-200, 1.0), (1e-300, 1e200, 1e200))
      f = Scaled(Translate(Scaled(NormL1(λ), a), [0.0]), a)
      for op in (prox!, value_and_prox!)
        y = [0.0]
        v = op(y, Conjugate(f), [input], 1.0)
        accurate(y, [BigFloat(λ) * BigFloat(a)^2], 4)
        op === value_and_prox! && @test v == 0
      end
    end
    # The shift itself must still fit the output even when clipping is bounded.
    y = Float32[7]
    @test_throws ArgumentError prox!(y, Conjugate(Translate(NormL1(), [1e39])), [1e39], 1.0)
    @test y == Float32[7]
    c = Conjugate(SeparableSum(Scaled(NormL1(1e300), 1e-310), SqrNormL2()))
    for op in (prox!, value_and_prox!)
      y = ([0.0], [0.0])
      v = op(y, c, ([1.0], [0.0]), 1.0)
      accurate(y[1], [BigFloat(1e300) * BigFloat(1e-310)], 4)
      @test y[2] == [0.0]
      op === value_and_prox! && @test v == 0
    end
    # Standalone values also exist for these scalar and array-step wrappers.
    c = Conjugate(Translate(Scaled(NormL1(1e300), 1e-310), [0.0]))
    for γ in (1.0, [1.0])
      y = [0.0]
      @test value_and_prox!(y, c, [1.0], γ) == 0
      accurate(y, [BigFloat(1e300) * BigFloat(1e-310)], 4)
    end
  end
end


@testset "warmed exceptional wrapper scales" begin
  for (λ, a) in ((1e300, 1e-200), (1e-300, 1e200))
    c = Conjugate(Scaled(SeparableSum(Scaled(NormL1(λ), a)), a))
    x, y = ([1e200],), ([0.0],)
    @test allocs(prox!, y, c, x, 1.0) == 0
    @test allocs(value_and_prox!, y, c, x, 1.0) == 0
  end
end


@testset "indexed matrix group projections and allocations" begin
  setprecision(BigFloat, 4096) do
    for T in (Float32, Float64), groups in ([1:3, 5:8], [[1, 3, 5], [2, 7, 9]])
      f = GroupNormL2(one(T), groups, ones(T, 2))
      c = Conjugate(f)
      x = reshape(T.(1:12), 3, 4)
      ref = reference(f, x, 1.0)
      y = similar(x)
      for op in (prox!, value_and_prox!)
        v = op(y, c, x, 1.0)
        accurate(y, ref, 8)
        op === value_and_prox! && @test v == 0
        @test allocs(op, y, c, x, 1.0) == 0
        alias = copy(x)
        op(alias, c, alias, 1.0)
        @test alias == y
      end
    end
    for T in (Float32, Float64)
      x = zeros(T, 2, 3)
      x[[1, 4]] .= floatmax(T) * T(0.8)
      x[2] = T === Float32 ? T(1e-30) : T(1e-300)
      f = GroupNormL2(floatmax(T) / 2, [[1, 2, 4]], ones(T, 1))
      ref = reference(f, x, 1.0)
      c, y = Conjugate(f), similar(x)
      for op in (prox!, value_and_prox!)
        op(y, c, x, 1.0)
        accurate(y, ref, 8)
        @test allocs(op, y, c, x, 1.0) == 0
      end
    end
  end
end


copyblocks(x::AbstractArray) = copy(x)
copyblocks(x::Tuple) = map(copyblocks, x)
function reject_complex_projection(op, c, y, x, γ)
  before_y, before_x = copyblocks(y), copyblocks(x)
  @test_throws ArgumentError("Conjugate requires real entries") op(y, c, x, γ)
  @test isequal(y, before_y)
  @test isequal(x, before_x)
end

@testset "direct projections reject complex entries before writing" begin
  for T in (Float32, Float64), imaginary in (0, 1), op in (prox!, value_and_prox!)
    h = floatmax(T) * T(0.7)
    tiny = T === Float32 ? T(1e-30) : T(1e-300)
    for (x, λ) in ((Complex{T}[3, 4, 0, 7], one(T)),
                   (Complex{T}[3 + imaginary * im, 12im, 0, 7], one(T)),
                   (zeros(Complex{T}, 4), one(T)),
                   (Complex{T}[complex(tiny, tiny), tiny, 0, 7], tiny),
                   (Complex{T}[complex(h, h), complex(h, -h), tiny, 7], one(T)),
                   (Complex{T}[complex(h, h), complex(h, -h), tiny, 7], floatmax(T) / 2))
      f = GroupNormL2(λ, [[1, 2, 3]], ones(T, 1))
      c = Conjugate(f)
      reject_complex_projection(op, c, fill(T(73), 4), x, 1.0)
      reject_complex_projection(op, c, fill(Complex{T}(73), 4), real.(x), 1.0)
      reject_complex_projection(op, c, x, x, 1.0)
      # A late complex block must leave the earlier real block untouched.
      blocks = Conjugate(SeparableSum(NormL1(), f))
      reject_complex_projection(op, blocks, (fill(T(73), 1), fill(T(73), 4)),
                                (T[2], x), 1.0)
      reject_complex_projection(op, blocks, (fill(T(73), 1), fill(Complex{T}(73), 4)),
                                (T[2], real.(x)), 1.0)
    end
  end
end

@testset "indexed integer group projections" begin
  setprecision(BigFloat, 4096) do
    for I in (Int32, Int64, Int128), op in (prox!, value_and_prox!)
      x = I[typemin(I)]
      f = GroupNormL2(1e-300, [[1]], [1.0])
      y = [0.0]
      v = op(y, Conjugate(f), x, 1.0)
      accurate(y, [-BigFloat(f.λ0)], 8)
      op === value_and_prox! && @test v == 0
      blocks = ([0.0],)
      v = op(blocks, Conjugate(SeparableSum(f)), (x,), 1.0)
      @test blocks[1] == y
      op === value_and_prox! && @test v == 0
    end
  end
end


@testset "direct projection storage and validation" begin
  for T in (Float32, Float64)
    for f in (NormL1(0.3), NormL1(0.3, T[0, Inf, 1, 1, 1, 1, 1, 1]), NormL1Nonneg(0.3),
              NormL2(0.3), GroupNormL2(0.3, [1:3, 5:7], ones(T, 2)), Scaled(NormL1(0.3), 2)),
        op in (prox!, value_and_prox!), reverse in (false, true)
      parent = T.(range(-4, 4; length = 9))
      x = view(parent, reverse ? (2:9) : (1:8))
      y = view(parent, reverse ? (1:8) : (2:9))
      c = Conjugate(f)
      expected = prox(c, copy(x), 0.7)
      v = op(y, c, x, 0.7)
      @test y ≈ expected rtol = 8eps(T)
      op === value_and_prox! && @test v == 0
    end
    for (f, error) in ((NormL1(0.3, ones(T, 7)), DimensionMismatch),
                       (GroupNormL2(0.3, [1:9], ones(T, 1)), BoundsError)),
        op in (prox!, value_and_prox!)
      x, y = ones(T, 8), fill(T(7), 8)
      @test_throws error op(y, Conjugate(f), x, 0.7)
      @test y == fill(T(7), 8)
    end
  end
end


@testset "promoted clipping coefficients and real output boundaries" begin
  setprecision(BigFloat, 4096) do
    for F in (NormL1, NormL1Nonneg), (λ, w) in ((2, [typemax(Int64)]),
        (Float32(0.3), [typemax(Int64) ÷ 3]), (Float32(0.3), Real[typemax(Int64) ÷ 3]),
        (0.3, [typemax(Int128)]),
        (2, [typemax(UInt64)])), op in (prox!, value_and_prox!), γ in (1.0, [1.0])
      f = F(λ, w)
      x, y = [1e40], [0.0]
      v = op(y, Conjugate(f), x, γ)
      accurate(y, reference(f, x, 1.0), 4)
      op === value_and_prox! && @test v == 0
    end
    for F in (NormL1, NormL1Nonneg), op in (prox!, value_and_prox!)
      f = F()
      x = [2.0, -2.0, 0.5]
      c = Conjugate(f)
      for γ in (1.0, ones(3))
        reject_complex_projection(op, c, fill(73.0 + 0im, 3), x, γ)
        reject_complex_projection(op, c, fill(73.0, 3), complex.(x), γ)
      end
      reject_complex_projection(op, c, fill(73.0, 3), x, complex.(ones(3)))
      reject_complex_projection(op, Conjugate(SeparableSum(f)),
                                (fill(73.0 + 0im, 3),), (x,), 1.0)
      # Nested blocks exercise the retained exponent scale with real entries.
      f = Scaled(SeparableSum(Scaled(F(1e300), 1e-200)), 1e-200)
      y = ([0.0],)
      v = op(y, Conjugate(f), ([2.0],), 1.0)
      accurate(y[1], [BigFloat(1e300) * BigFloat(1e-200)^2], 4)
      op === value_and_prox! && @test v == 0
      reject_complex_projection(op, Conjugate(f), (fill(73.0 + 0im, 1),), ([2.0],), 1.0)
    end
    δ = nextfloat(0f0)
    x = Float32[2, 3δ, 0]
    f = NormL2(1/3 - 1e-10)
    ref = reference(f, x, 1.0)
    @test !iszero(ref[2]) && iszero(Float32(ref[2]))
    for op in (prox!, value_and_prox!)
      y = zero(x)
      op(y, Conjugate(f), x, 1.0)
      # A nonzero reference rounded to zero still gets the subnormal allowance.
      accurate(y, ref, 8)
      @test iszero(y[3])
    end
  end
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

# Master's tuple route narrows division/output, but not the reciprocal step.
tuple_divide!(p::AbstractArray, x::AbstractArray, γ) = (p .= x ./ convert(float(eltype(p)), γ))
tuple_divide!(p::Tuple, x::Tuple, γ) = (map((pk, xk) -> tuple_divide!(pk, xk, γ), p, x); p)
tuple_subtract(x::AbstractArray, p::AbstractArray, γ) = eltype(p).(x .- convert(float(eltype(p)), γ) .* p)
tuple_subtract(x::Tuple, p::Tuple, γ) = map((xk, pk) -> tuple_subtract(xk, pk, γ), x, p)
function tuple_reference(f, x, γ, output = x)
  p = deepcopy(output)
  tuple_divide!(p, x, γ)
  prox!(p, f, p, inv(γ))
  tuple_subtract(x, p, γ)
end

function tuple_reference_value(f, x, γ)
  p = deepcopy(x)
  tuple_divide!(p, x, γ)
  fp = value_and_prox!(p, f, p, inv(γ))
  y = tuple_subtract(x, p, γ)
  sum(map(dot, p, y)) - fp
end

@testset "tuple fallback blocks retain the original reciprocal step" begin
  γ = 1e-39
  for T in (Float32, Float64), base in (SqrNormL2(), MoreauEnvelope(NormL1(1.0), 1.0)), α in (1.0, 2.0, 0.7)
    f = Scaled(base, γ)
    ref = tuple_reference(Scaled(SeparableSum(f), α), (T[γ],), γ)[1]
    @test ref ≈ T[α * γ / (1 + α)] rtol = 8eps(T) atol = 2nextfloat(zero(T))
    whole = Scaled(SeparableSum(f), α)
    value = base isa SqrNormL2 ? Conjugate(whole)((ref,)) : tuple_reference_value(whole, (T[γ],), γ)
    for layout in (:single, :mixed, :nested)
      s, x = if layout === :single
        (SeparableSum(f), (T[γ],))
      elseif layout === :mixed
        (SeparableSum(NormL1(), f), (Float64[3], T[γ]))
      else
        (SeparableSum(NormL1(), SeparableSum(f)), (Float64[3], (T[γ],)))
      end
      c = Conjugate(Scaled(s, α))
      for op in (prox!, value_and_prox!), alias in (false, true)
        point = deepcopy(x)
        y = alias ? point : map(copyblocks, x)
        v = op(y, c, point, γ)
        actual = layout === :single ? y[1] : layout === :mixed ? y[2] : y[2][1]
        @test actual == ref
        layout === :single || @test y[1] == [α]
        op === value_and_prox! && @test v ≈ value rtol = 8eps(T) atol = 8nextfloat(zero(T))
      end
    end
  end
  for base in (SqrNormL2(), MoreauEnvelope(NormL1(), 1.0)), op in (prox!, value_and_prox!)
    f = SeparableSum(Scaled(base, γ))
    x, y = ([γ],), (Float32[7],)
    ref = tuple_reference(f, x, γ, y)
    op(y, Conjugate(f), x, γ)
    @test y == ref
  end
  # These remain genuine array calls, with the reciprocal of the narrowed step.
  f = Scaled(SqrNormL2(), γ)
  @test prox(Conjugate(f), Float32[γ], γ) == Float32[γ]
  f = Scaled(MoreauEnvelope(NormL1(), 1.0), γ)
  @test_throws ArgumentError prox(Conjugate(f), Float32[γ], γ)

  # Float32's original reciprocal overflows on master as well.
  bad = Float32(γ)
  f = Scaled(MoreauEnvelope(NormL1(), 1.0), γ)
  @test_throws ArgumentError tuple_reference(SeparableSum(f), (Float32[bad],), bad)
  for nested in (false, true), op in (prox!, value_and_prox!), warm in (false, true), alias in (false, true)
    s = nested ? SeparableSum(NormL1(), SeparableSum(f)) : SeparableSum(NormL1(), f)
    x = nested ? ([3.0], (Float32[bad],)) : ([3.0], Float32[bad])
    c = Conjugate(s)
    y = alias ? x : deepcopy(x)
    warm && op(deepcopy(y), c, deepcopy(x), 1.0)
    before_y, before_x = deepcopy(y), deepcopy(x)
    @test_throws ArgumentError op(y, c, x, bad)
    @test y == before_y
    @test x == before_x
  end
end

function warmed_projection_allocation(op::O, y, c, x, γ) where {O}
  op(y, c, x, γ)
  op(y, c, x, γ)
  @allocated op(y, c, x, γ)
end

@testset "scaled fallback sub-sums and compensating coefficients" begin
  sub = Scaled(SeparableSum(SqrNormL2()), 1e60)
  α, γ = 1f-30, 1f30
  c = Conjugate(Scaled(SeparableSum(NormL1(), sub), α))
  expected = ([Float64(α)], ([0.0],))
  for op in (prox!, value_and_prox!), alias in (false, true)
    x = ([3.0], ([1.0],))
    y = alias ? x : deepcopy(x)
    v = op(y, c, x, γ)
    @test y == expected
    op === value_and_prox! && @test v == 0
  end
  for op in (prox!, value_and_prox!)
    skip = VERSION < v"1.11"
    @test warmed_projection_allocation(op, deepcopy(expected), c, ([3.0], ([1.0],)), γ) == 0 skip = skip
  end

  sub = Scaled(SeparableSum(Scaled(SqrNormL2(), 1e-300)), 1e300)
  c = Conjugate(Scaled(SeparableSum(NormL1(), sub), 1e300))
  expected = ([3.0], ([1e150],))
  for op in (prox!, value_and_prox!), alias in (false, true)
    x = ([3.0], ([1e150],))
    y = alias ? x : deepcopy(x)
    v = op(y, c, x, 1.0)
    @test y == expected
    op === value_and_prox! && @test v ≈ 0.5 rtol = 8eps(Float64)
  end
  for op in (prox!, value_and_prox!)
    skip = VERSION < v"1.11"
    @test warmed_projection_allocation(op, deepcopy(expected), c, deepcopy(expected), 1.0) == 0 skip = skip
  end

  setprecision(BigFloat, 4096) do
    # The last factor compensates a product that cannot be represented alone.
    small = Scaled(SqrNormL2(), 1e-300)
    large = Scaled(SqrNormL2(), 1e300)
    over = Scaled{typeof(small), Float64}(small, 1e300)
    under = Scaled{typeof(large), Float64}(large, 1e-300)
    for (f, α, input, γ, factors) in (
        (large, 1e-300, 2e100, 1.0, (1e-300, 1e300)),
        (small, 1e300, 2e100, 1.0, (1e300, 1e-300)),
        (over, 1e300, 1e150, 1.0, (1e300, 1e300, 1e-300)),
        (under, 1e-300, 2e-150, 1e-300, (1e-300, 1e-300, 1e300)))
      c = Conjugate(Scaled(SeparableSum(NormL1(), f), α))
      coefficient = prod(BigFloat.(factors))
      point = BigFloat(input) * coefficient / (coefficient + BigFloat(γ))
      value = point^2 / (2coefficient)
      expected = ([min(3.0, α)], [Float64(point)])
      baseline = prox(c, ([3.0], [input]), γ)
      accurate(baseline[2], [point], 8)
      @test baseline[1] == expected[1]
      for op in (prox!, value_and_prox!), alias in (false, true)
        x = ([3.0], [input])
        y = alias ? x : deepcopy(x)
        v = op(y, c, x, γ)
        @test y == baseline
        if op === value_and_prox!
          @test isfinite(v)
          accurate([v], [value], 8)
        end
      end
      for op in (prox!, value_and_prox!)
        @test warmed_projection_allocation(op, deepcopy(expected), c, ([3.0], [input]), γ) == 0
      end
    end
  end
end

@testset "scaled mixed conjugate values allocate nothing" begin
  for T in (Float32, Float64), α in (1.0, 2.0, 0.7)
    for f in (SqrNormL2(0.7, T[1, -2]), HingeLoss(T[1, -1], T[0.5, 1.5]),
              MoreauEnvelope(NormL1(0.3, T[1, 2]), 0.5), NormLinf(0.7), Conjugate(SeparableSum(SqrNormL2())))
      block = f isa Conjugate ? (T[2, -3],) : T[2, -3]
      x = (T[3, -4], block)
      y = deepcopy(x)
      c = Conjugate(Scaled(SeparableSum(NormL1(), f), α))
      expected = prox(c, x, 0.7)
      # Retain the existing Julia 1.10 exception for nested tuple buffers.
      skip = block isa Tuple && VERSION < v"1.11"
      for op in (prox!, value_and_prox!)
        op(y, c, x, 0.7)
        @test y == expected
        @test warmed_projection_allocation(op, y, c, x, 0.7) == 0 skip = skip
      end
    end
  end
end


@testset "generic group projection norm fallbacks" begin
  setprecision(BigFloat, 4096) do
    for T in (Float32, Float64)
      for (input, λ) in ((T[floatmax(T) / 2, floatmax(T) / 2], one(T)),
                         (T[floatmax(T) * T(0.8), floatmax(T) * T(0.8)], one(T)),
                         (T[floatmin(T), floatmin(T) / 2], floatmin(T)),
                         (T[0, 0], one(T)))
        # Box the entries to select the generic real-array reduction.
        x = Real[input...]
        X = BigFloat.(input)
        n = norm(X)
        expected = iszero(n) ? X : X .* min(BigFloat(1), BigFloat(λ) / n)
        c = Conjugate(GroupNormL2(λ, [[1, 2]], T[1]))
        for op in (prox!, value_and_prox!)
          y = zeros(T, 2)
          value = op(y, c, x, one(T))
          accurate(y, expected, 8)
          op === value_and_prox! && @test value == 0
          @test x == input
        end
      end
      # A singleton NaN stays NaN; the ungrouped coordinate is exactly zero.
      c = Conjugate(GroupNormL2(one(T), [[1]], T[1]))
      for op in (prox!, value_and_prox!)
        y = zeros(T, 2)
        value = op(y, c, Real[T(NaN), T(3)], one(T))
        @test isnan(y[1])
        @test iszero(y[2])
        op === value_and_prox! && @test value == 0
      end
    end
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

@testset "tuple indicator fallback and empty blocks" begin
  for T in (Float32, Float64), α in (one(T), T(2)), γ in (T(0.5), T(2))
    support = Conjugate(SeparableSum(IndBox(T(-2), T(2)), IndBox(one(T), T(3))))
    c = Conjugate(Scaled(SeparableSum(NormL1(), support), α))
    input = (T[3], (T[-4, 1, 4], T[-2, 2, 4]))
    expected = (clamp.(input[1], -α, α),
                (clamp.(input[2][1], -2α, 2α), clamp.(input[2][2], α, 3α)))
    @test prox(c, input, γ) == expected
    for op in (prox!, value_and_prox!), alias in (false, true)
      x = deepcopy(input)
      y = alias ? x : deepcopy(input)
      value = op(y, c, x, γ)
      @test y == expected
      op === value_and_prox! && @test value == 0
    end
  end
  for op in (prox!, value_and_prox!)
    for (f, x) in ((SeparableSum(), ()),
                   (SeparableSum(SeparableSum()), ((),)),
                   (SeparableSum(SeparableSum(), SeparableSum(SeparableSum())), ((), ((),))))
      c = Conjugate(f)
      @test prox(c, x, 1.0) == x
      value = op(deepcopy(x), c, x, 1.0)
      op === value_and_prox! && @test value == 0
    end
    c = Conjugate(SeparableSum())
    @test prox(c, (), 1.0) == ()
    value = op((), c, (), 1.0)
    op === value_and_prox! && @test value == 0
    c = Conjugate(SeparableSum(SeparableSum(), NormL1(), SqrNormL2(2)))
    x, y = ((), [3.0], [4.0]), ((), [0.0], [0.0])
    value = op(y, c, x, 1.0)
    @test y[1] == ()
    @test y[2] == [1.0]
    @test y[3] ≈ [8/3]
    op === value_and_prox! && @test value ≈ 16/9
    for T in (Float32, Float64), alias in (false, true)
      c = Conjugate(SeparableSum(NormL1(), SeparableSum(SeparableSum(), NormL2(T(5)))))
      x = (T[3, -4], ((), T[6, 8]))
      y = alias ? x : deepcopy(x)
      value = op(y, c, x, one(T))
      @test y == (T[1, -1], ((), T[3, 4]))
      op === value_and_prox! && @test value === zero(T)
    end
  end
end

@testset "scaled fallback entries and mixed values" begin
  setprecision(BigFloat, 4096) do
    for T in (Float32, Float64), α in (one(T), T(2))
      γ = T(0.5)
      c = Conjugate(Scaled(SqrNormL2(T(2)), α))
      for v in (T(-3), zero(T), T(6))
        expected = BigFloat(v) * (2BigFloat(α)) / (2BigFloat(α) + BigFloat(γ))
        accurate(T[prox_entry(c, v, 1, γ)], [expected], 8)
      end
      for f in (ZeroFunction(), IndNonnegative()), v in (T(-3), zero(T), T(6))
        c = Conjugate(Scaled(f, α))
        @test prox_entry(c, v, 1, γ) == (f isa ZeroFunction ? zero(T) : min(v, zero(T)))
      end
      for tuple_block in (false, true)
        f = tuple_block ? Conjugate(SeparableSum(SqrNormL2(T(2)))) : SqrNormL2(T(2), T[1, -2])
        c = Conjugate(Scaled(SeparableSum(NormL1(), f), α))
        input = tuple_block ? (T[3], (T[3, -6],)) : (T[3], T[3, -6])
        A, G = BigFloat(α), BigFloat(γ)
        expected = tuple_block ? BigFloat[3, -6] .* A ./ (A + 2G) :
          (BigFloat[3, -6] .- G .* BigFloat[1, -2]) .* (2A) ./ (2A + G)
        expected_value = tuple_block ? sum(abs2, expected) / A :
          sum(abs2, expected) / (4A) + dot(BigFloat[1, -2], expected)
        for op in (prox!, value_and_prox!), alias in (false, true)
          x = deepcopy(input)
          y = alias ? x : deepcopy(input)
          value = op(y, c, x, γ)
          @test y[1] == T[min(T(3), α)]
          accurate(tuple_block ? y[2][1] : y[2], expected, 8)
          op === value_and_prox! && accurate(T[value], [expected_value], 8)
        end
      end
    end
  end
end

end
