module ConjugateProjectionsTest

using Test, ProximalBase, LinearAlgebra, Random, SparseArrays
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

# L1 conjugate regression cases.
@testset "L1 conjugate radius accuracy" begin
  setprecision(BigFloat, 4096) do
    for T in (Float32, Float64)
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
    for x in ([1e300, 0.25, -0.5], [1e300, 1e300, 0.0]), op in (prox!, value_and_prox!)
      y = zeros(Float32, 3)
      op(y, Conjugate(NormLinf()), x, 1.0)
      l1accurate(y, projl1(x, BigFloat(1)), BigFloat(1))
    end
    x = [typemin(Int), 2, -1]
    l1accurate(prox(Conjugate(NormLinf()), x), projl1(x, BigFloat(1)), BigFloat(1))
    for I in (Int64, Int128), x in (I[typemin(I), typemin(I)+1], I[typemax(I), typemax(I)-1])
      for f in (NormLinf(), GroupNormLinf(1, [1:2]))
        l1accurate(prox(Conjugate(f), x), projl1(x, BigFloat(1)), BigFloat(1))
      end
    end

  end
  setprecision(BigFloat, 4096) do
    for T in (Float32, Float64), n in (33, 65, 257), fraction in (0.01, 0.5, 1.0)
      x = T[(-1)^i * (1 + mod(17i, 31)/32) for i in 1:n]
      r = T(fraction * n)
      l1accurate(prox(Conjugate(NormLinf(r)), x), projl1(x, BigFloat(r)), BigFloat(r); support = false)
    end
    # Widen parameters before multiplying, even with Float64 source entries.
    r = Float64(0.3f0) * Float64(0.7f0)
    x = [1.0, 1.0-r]
    for f in (Scaled(NormLinf(0.3f0), 0.7f0), GroupNormLinf(0.3f0, [1:2], [0.7f0]))
      l1accurate(prox(Conjugate(f), x), projl1(x, BigFloat(r)), BigFloat(r))
    end
  end

end

@testset "L1 conjugate groups and wrappers" begin
  setprecision(BigFloat, 4096) do
    for T in (Float32, Float64), λ in (zero(T), T(0.5)), w in (T[0, Inf, 1e30], T[1, 2, 0])
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
    for λ in (0.0, 1.0), w in (0.0, Inf, 1e300)
      f = GroupNormLinf(λ, [1:2], [w])
      x = [1e300, 0.5, 3.0]
      y = prox(Conjugate(f), x)
      r = coefficient(BigFloat(1), λ, BigFloat(w))
      isinf(r) ? (@test y[1:2] == x[1:2]) : l1accurate(y[1:2], reference(f, x, 1)[1:2], r)
    end
    for λ in (0.0, 1.0), op in (prox!, value_and_prox!)
      f = GroupNormLinf(λ, [[1, 2], [3, 4]], [Inf, 0])
      y = zeros(5)
      op(y, Conjugate(f), [NaN, Inf, NaN, Inf, NaN], 1.0)
      @test isequal(y, [NaN, Inf, 0, 0, 0])
    end
    @test prox(Conjugate(GroupNormLinf(1.0, Vector{Int}[])), [2.0, 3.0]) == zeros(2)
    @test_throws ArgumentError GroupNormLinf(1.0, [Int[]])
    x, b = [1e300, -3.0, 1.0], [0.5, -0.25, 0.125]
    for base in (NormLinf(0.5), GroupNormLinf(0.5, [1:2, 3:3])),
        f in (Translate(Scaled(base, 2.0), b), Scaled(Translate(base, b), 2.0)),
        op in (prox!, value_and_prox!)
      y = zeros(3)
      v = op(y, Conjugate(f), x, 0.5)
      ref = reference(f, x, 0.5)
      @test maximum(abs.(BigFloat.(y) - ref)) <= 12eps(Float64)
      op === value_and_prox! && @test v ≈ dot(b, y)
    end
    # Parameters alias a destination, with group order opposite to write order.
    w = [0.25, 0.5]
    f = GroupNormLinf(1.0, [[2], [1]], w)
    @test f.w === w
    prox!(w, Conjugate(f), [3.0, 4.0], 1.0)
    @test w == [0.5, 0.25]
    a = [1e300, 0.5, -2.0, 7.0]
    prox!(view(a, 2:4), Conjugate(NormLinf()), view(a, 1:3), 1.0)
    @test a == [1e300, 1, 0, 0]
  end
end

@testset "L1 conjugate staging, iterates and allocations" begin
  for T in (Float32, Float64), base in (NormLinf(T(0.5)), GroupNormLinf(T(0.5), [1:2, 3:3]))
    x = T[3, -2, 1]
    for f in (base, Scaled(base, 2.0), Translate(base, zeros(T, 3))), op in (prox!, value_and_prox!)
      c = Conjugate(f)
      y = similar(x)
      @test warmed_projection_allocation(op, y, c, x, one(T)) == 0
      expected = copy(y)
      for source in (SparseIterate(x), AtomIterate(x, [1:2, 3:3])), alias in (false, true)
        y = alias ? source : copy(source)
        op(y, c, source, one(T))
        @test Vector(y) == expected
        @test collect(activecoordinates(y)) == collect(activecoordinates(source))
      end
    end
    c = Conjugate(base)
    @test !isseparable(c)
    for op in (prox!, value_and_prox!)
      y = fill(T(7), 3)
      @test_throws (base isa NormLinf ? MethodError : DimensionMismatch) op(y, c, x, T[1, 2, 3])
      @test y == fill(T(7), 3)
    end
  end

  for T in (Float32, Float64), op in (prox!, value_and_prox!)
    x = fill(T(0.9), 65)
    x[1] = 1
    for f in (NormLinf(T(32)), GroupNormLinf(T(4), [1:8, 9:16, 17:65])), alias in (false, true)
      y = copy(x)
      source = alias ? y : x
      @test warmed_projection_allocation(op, y, Conjugate(f), source, one(T)) == 0
    end
    x = ([3.0, 2.0], ([4.0, -2.0], [3.0]))
    f = SeparableSum(NormLinf(), SeparableSum(GroupNormLinf(1.0, [1:2]), SqrNormL2()))
    @test warmed_projection_allocation(op, deepcopy(x), Conjugate(f), x, 1.0) == 0
    x = [3.0 2.0; 2.0 1.0]
    y = SymmetricSparseIterate(x)
    op(y, Conjugate(NormLinf(2.0)), x, 1.0)
    @test Matrix(y) == prox(Conjugate(NormLinf(2.0)), x)
  end
  for op in (prox!, value_and_prox!), nested in (false, true)
    f = nested ? SeparableSum(NormLinf(), SeparableSum(SqrNormL2())) : SeparableSum(NormLinf(), SqrNormL2())
    x = nested ? ([1e300, 0.5], (Float32[1e30],)) : ([1e300, 0.5], Float32[1e30])
    y = deepcopy(x)
    @test_throws ArgumentError op(y, Conjugate(f), x, 1e-300)
    @test y == x
  end
  for op in (prox!, value_and_prox!)
    # Projection succeeds; the second destination rejects an asymmetric commit.
    f = SeparableSum(NormLinf(), GroupNormLinf(1.0, [[1], [2], [3], [4]], [1, 0.25, 0.5, 1]))
    x = ([3.0, 2.0], ones(2, 2))
    y = ([7.0, 7.0], SymmetricSparseIterate(ones(2, 2)))
    before = deepcopy(y)
    @test_throws ArgumentError op(y, Conjugate(f), x, 1.0)
    @test y == before
    for f in (Translate(NormLinf(), [floatmax(Float64), 0.0]),
              Translate(GroupNormLinf(1.0, [1:2]), [floatmax(Float64), 0.0]))
      y = [7.0, 7.0]
      @test_throws ArgumentError op(y, Conjugate(f), [1.0, 2.0], 2.0)
      @test y == [7.0, 7.0]
    end
    f = SeparableSum(NormLinf(), SeparableSum(GroupNormLinf(1.0, [1:2]), SqrNormL2()))
    x = ([3.0, 2.0], ([4.0, -2.0], [3.0]))
    y = deepcopy(x)
    op(y, Conjugate(f), x, 1.0)
    @test y == ([1.0, 0.0], ([1.0, -0.0], [1.5]))
  end
end

@testset "L1 conjugate large reductions" begin
  setprecision(BigFloat, 4096) do
    for T in (Float32, Float64), n in (10_000, 100_000)
      x = fill(T(0.9), n)
      x[1] = 1
      r = T(n/2)
      y = prox(Conjugate(NormLinf(r)), x)
      l1accurate(y, projl1(x, BigFloat(r)), BigFloat(r); support = false)
    end
  end
end

@testset "L1 projection review regressions" begin
  setprecision(BigFloat, 4096) do
    for factors in ((0.7f0, 0.3f0), (0.1f0, 0.9f0, 0.7f0),
                    (0.1, 0.1), (0.7, 0.3, 0.9, 0.13)),
        base in (NormLinf(), GroupNormLinf(1, [1:2]))
      f = base
      for a in factors
        f = Translate(f, zeros(2))
        f = Scaled{typeof(f),typeof(a)}(f, a)
      end
      r = prod(BigFloat.(factors))
      x = [1.0, Float64(1-r)]
      for op in (prox!, value_and_prox!)
        c = Conjugate(f)
        y = zeros(2)
        op(y, c, x, 1.0)
        l1accurate(y, reference(f, x, 1), r)
        @test warmed_projection_allocation(op, y, c, x, 1.0) == 0
      end
    end
    a, b = 0.7f0, 0.3f0
    r = Float64(a) * Float64(b)
    f = Scaled(Translate(Scaled(NormLinf(), b), zeros(2)), a)
    for op in (prox!, value_and_prox!)
      y = zeros(2)
      op(y, Conjugate(f), [1.0, 1.0-r], 1.0)
      @test y == [r, 0]
    end
    inner = Scaled(Translate(Scaled(GroupNormLinf(1, [1:2], [0.9]), 0.3f0), zeros(2)), 0.7f0)
    f = Scaled(SeparableSum(inner, SqrNormL2(2)), 0.9f0)
    r = BigFloat(0.9f0) * BigFloat(0.7f0) * BigFloat(0.3f0) * BigFloat(0.9)
    x = ([1.0, Float64(1-r)], [3.0])
    for op in (prox!, value_and_prox!)
      y = deepcopy(x)
      c = Conjugate(f)
      op(y, c, x, 1.0)
      l1accurate(y[1], reference(inner, x[1], 1, BigFloat(0.9f0)), r)
      @test y[2] == prox(Conjugate(Scaled(SqrNormL2(2), 0.9f0)), x[2], 1.0)
      @test warmed_projection_allocation(op, y, c, x, 1.0) == 0
    end
    # Retain the unrounded factor chain across raw nested Scaled wrappers too.
    f = NormLinf()
    factors = (0.7, 0.3, 0.13, 0.9, 0.11)
    for a in factors
      f = Scaled{typeof(f),typeof(a)}(f, a)
    end
    r = prod(BigFloat.(factors))
    x = [1.0, Float64(1-r)]
    l1accurate(prox(Conjugate(f), x), reference(f, x, 1), r)

    a = 0.1
    x = [a*a]
    radius, error, lower = ProximalBase._projection_l1_coefficient(a, a, 1, x, x)
    @test BigFloat(lower) * BigFloat(2)^radius.e <= BigFloat(a)^2
    @test !ProximalBase._projection_l1_inside(x, eachindex(x), lower, ldexp(1.0, -radius.e), radius.e)
    l1accurate(prox(Conjugate(Scaled(NormLinf(a), a)), x), reference(Scaled(NormLinf(a), a), x, 1), BigFloat(a)^2)

    # The compensated radius interval encloses the exact product at each depth.
    for n in (1, 2, 8, 16)
      fs = ntuple(i -> isodd(i) ? 0.7 : 0.3, n)
      α = ProximalBase._L1ProjectionFactors(fs)
      radius, error, lower = ProximalBase._projection_l1_coefficient(α, 1, 1, [1.0], [0.0])
      exact = prod(BigFloat.(fs)) / BigFloat(2)^radius.e
      @test abs(exact - radius.m) <= BigFloat(error)
      @test BigFloat(lower) <= exact
    end
  end
  for op in (prox!, value_and_prox!), x in (
      [(2^60)//1, (2^60-1)//1], BigInt[2^60, 2^60-1], Real[2^60, 2^60-1])
    for f in (NormLinf(), GroupNormLinf(1, [1:2]), Translate(Scaled(NormLinf(), 0.7), zeros(2)))
      y = fill(7.0, 2)
      @test_throws r"floating or machine-integer input" op(y, Conjugate(f), x, 1.0)
      @test y == fill(7.0, 2)
    end
    f = Conjugate(SeparableSum(NormLinf(), SeparableSum(GroupNormLinf(1, [1:2]))))
    y = ([7.0], (fill(7.0, 2),))
    @test_throws ArgumentError op(y, f, ([3.0], (x,)), 1.0)
    @test y == ([7.0], (fill(7.0, 2),))
  end
  raw = reshape([1e300, 0.25, -0.5, 0.0], 2, 2)
  permuted = PermutedDimsArray(raw, (2, 1))
  reinterpreted = reinterpret(Float64, copy(reinterpret(UInt64, raw)))
  for x in (permuted, view(permuted, :, :), reshape(permuted, 4),
            reinterpreted, view(reinterpreted, :, :), reshape(reinterpreted, 4)),
      op in (prox!, value_and_prox!)
    @test ProximalBase._projection_host(x)
    y = zeros(size(x))
    op(y, Conjugate(NormLinf()), x, 1.0)
    @test y == ifelse.(abs.(x) .== 1e300, 1.0, 0.0)
  end
  for op in (prox!, value_and_prox!), nested in (false, true)
    f = nested ? SeparableSum(NormLinf(), SeparableSum(NormLinf())) : SeparableSum(NormLinf(), NormLinf())
    x = nested ? ([2.0], ([1.0, 1.0],)) : ([2.0], [1.0, 1.0])
    y = nested ? ([7.0], (Int[7, 7],)) : ([7.0], Int[7, 7])
    before = deepcopy(y)
    @test_throws InexactError op(y, Conjugate(f), x, 1.0)
    @test y == before
  end
end

struct ProjectionReads{T} <: AbstractVector{T}
  data::Vector{T}
  reads::Base.RefValue{Int}
end
Base.size(x::ProjectionReads) = size(x.data)
Base.IndexStyle(::Type{<:ProjectionReads}) = IndexLinear()
Base.getindex(x::ProjectionReads, i::Int) = (x.reads[] += 1; x.data[i])

@testset "native L1 projection certificates and passes" begin
  setprecision(BigFloat, 4096) do
    for T in (Float32, Float64)
      rng = MersenneTwister(271828)
      accepted = 0
      for n in (8, 64, 257), frac in (1e-4, 0.1, 0.9, 2.0)
        x = randn(rng, T, n)
        r = T(frac * sum(abs, Float64.(x)))
        y = similar(x)
        accepted += ProximalBase._projection_l1_native!(y, x, eachindex(x), ProximalBase.Scratch(), r)
        c = Conjugate(NormLinf(r))
        for op in (prox!, value_and_prox!)
          op(y, c, x, one(T))
          l1accurate(y, projl1(x, BigFloat(r)), BigFloat(r); support = false)
          @test warmed_projection_allocation(op, y, c, x, one(T)) == 0
        end
      end
      @test accepted > 0
      # A stable rounded support can retain exact-zero coordinates. Check
      # the coordinate contract, independent of the reduction's ordering.
      e, r = eps(T), T(63/64)
      x = vcat(T[1, 1/64], fill(T(1/64) - T(7/2)*e, 15), zeros(T, 15))
      ref = projl1(x, BigFloat(r))
      @test ref == vcat(BigFloat(r), zeros(BigFloat, 31))
      for op in (prox!, value_and_prox!)
        y = similar(x)
        @test ProximalBase._projection_l1_native!(y, x, eachindex(x), ProximalBase.Scratch(), r)
        l1accurate(y, ref, BigFloat(r); support = false)
        op(y, Conjugate(NormLinf(r)), x, one(T))
        l1accurate(y, ref, BigFloat(r); support = false)
      end
      for (data, r, passes) in ((T[3, 0, 0, 0, 0, 0, 0, 0], T(10), 2),
                                (T[3, 0, 0, 0, 0, 0, 0, 0], T(1), 3),
                                (T[3, 2.8, 0, 0, 0, 0, 0, 0], T(1), 3),
                                (T[3, 2.8, 2.7, 0, 0, 0, 0, 0], T(1), 4))
        x = ProjectionReads(data, Ref(0))
        y = similar(data)
        @test ProximalBase._projection_l1_native!(y, x, eachindex(x), ProximalBase.Scratch(), r)
        @test x.reads[] == passes * length(x)
        l1accurate(y, projl1(data, BigFloat(r)), BigFloat(r))
      end
      # Native absolute levels cannot certify tiny radii at tied maxima.
      x = ones(T, 8)
      r = T(0.001)
      y = fill(T(73), 8)
      @test !ProximalBase._projection_l1_native!(y, x, eachindex(x), ProximalBase.Scratch(), r)
      @test y == fill(T(73), 8)
      l1accurate(prox(Conjugate(NormLinf(r)), x), projl1(x, BigFloat(r)), BigFloat(r))
      # Neighbouring floats around support changes and radius-scale boundaries.
      for e in (-20, 0, 20), offset in (-1, 0, 1)
        h = T(2)^e
        v = offset < 0 ? prevfloat(h) : offset > 0 ? nextfloat(h) : h
        x = vcat(T[h, v, -h], fill(h / 4, 61))
        for r in (h / 8, h, 8h)
          l1accurate(prox(Conjugate(NormLinf(r)), x), projl1(x, BigFloat(r)), BigFloat(r); support = false)
        end
      end
    end
  end
  # A single function alternates native and careful groups without replacing
  # the other precision's scratch on each group or each warmed call.
  x = vcat(Float32[1e30], ones(Float32, 48), fill(0.1f0, 49))
  f = GroupNormLinf(1f0, [1:49, 50:98])
  c = Conjugate(f)
  y = similar(x)
  for op in (prox!, value_and_prox!)
    @test warmed_projection_allocation(op, y, c, x, 1f0) == 0
  end
  @test f.scratch.buf isa ProximalBase._L1ProjectionScratch
  @test length(f.scratch.buf.native) >= 49
  @test length(f.scratch.buf.wide) >= 49
  f = NormLinf(1f0)
  x = Float32.(1:64) ./ 64
  prox!(similar(x), Conjugate(f), x, 1f0)
  @test f.scratch.buf isa ProximalBase._L1ProjectionScratch
  @test isempty(f.scratch.buf.wide)
end

function alternating_projection_allocation(op::O, y, p, c, f, x, γ) where {O}
  for _ in 1:2
    f(x)
    op(p, f, x, γ)
    op(y, c, x, γ)
  end
  @allocated begin
    f(x)
    op(p, f, x, γ)
    op(y, c, x, γ)
  end
end

@testset "L1 membership, shared scratch and structured storage" begin
  for I in (Int64, UInt64, Int128), sign in (1, -1)
    I === UInt64 && sign < 0 && continue
    x = I[sign * (I(2)^60 + 1)]
    r = 2.0^60
    a, _, lower = ProximalBase._projection_l1_coefficient(1, r, 1, x, [0.0])
    scale = ldexp(1.0, -a.e)
    @test !ProximalBase._projection_l1_inside(x, eachindex(x), lower, scale, a.e)
    for op in (prox!, value_and_prox!)
      y = zeros(1)
      op(y, Conjugate(NormLinf(r)), x, 1.0)
      @test y == [sign * r]
    end
    x = I[sign * I(2)^60]
    @test ProximalBase._projection_l1_inside(x, eachindex(x), lower, scale, a.e)
  end
  x = Float32.(1:128) ./ 128
  for f in (NormLinf(1f0), GroupNormLinf(1f0, [1:64, 65:128])),
      op in (prox!, value_and_prox!)
    c, y, p = Conjugate(f), similar(x), similar(x)
    @test alternating_projection_allocation(op, y, p, c, f, x, 0.7f0) == 0
    cache = f.scratch.buf
    @test cache isa ProximalBase._L1ProjectionScratch
    @test alternating_projection_allocation(op, y, p, c, f, x, 0.7f0) == 0
    @test f.scratch.buf === cache
    @test p == prox(f, x, 0.7f0)
    @test y == prox(c, x, 0.7f0)
  end
  D = Diagonal([1e300, 0.25])
  for x in (D, view(D, :, :), reshape(D, 4), UpperTriangular(Matrix(D)),
            Bidiagonal([1e300, 0.25], [0.0], :U),
            SymTridiagonal([1e300, 0.25], [0.0]),
            Tridiagonal([0.0], [1e300, 0.25], [0.0])),
      op in (prox!, value_and_prox!)
    @test ProximalBase._projection_host(x)
    y = zeros(size(x))
    op(y, Conjugate(NormLinf()), x, 1.0)
    @test y == ifelse.(abs.(x) .== 1e300, 1.0, 0.0)
  end
end

# Minimal offset axes without another test dependency.
struct ProjectionOffset{T} <: AbstractVector{T}
  data::Vector{T}
end
Base.size(x::ProjectionOffset) = size(x.data)
Base.axes(x::ProjectionOffset) = (Base.IdentityUnitRange(0:(length(x)-1)),)
Base.IndexStyle(::Type{<:ProjectionOffset}) = IndexLinear()
Base.eachindex(::IndexLinear, x::ProjectionOffset) = axes(x, 1)
Base.parent(x::ProjectionOffset) = x.data
Base.@propagate_inbounds function Base.getindex(x::ProjectionOffset, i::Int)
  @boundscheck checkbounds(x, i)
  x.data[i + 1]
end
Base.@propagate_inbounds function Base.setindex!(x::ProjectionOffset, v, i::Int)
  @boundscheck checkbounds(x, i)
  x.data[i + 1] = v
end
Base.copyto!(y::ProjectionOffset, x::AbstractArray) = (copyto!(y.data, x); y)

# Unknown storage deliberately inherits self-parenting, like sparse devices.
struct ProjectionUnknown{T} <: AbstractVector{T}
  data::Vector{T}
end
Base.size(x::ProjectionUnknown) = size(x.data)
Base.IndexStyle(::Type{<:ProjectionUnknown}) = IndexLinear()
Base.getindex(x::ProjectionUnknown, i::Int) = x.data[i]
Base.setindex!(x::ProjectionUnknown, v, i::Int) = (x.data[i] = v)

function l1_moreau_result!(op, y, f, x, γ)
  # Master's scalar-step Moreau route, including its dense scratch and commit.
  T = float(eltype(y))
  p = Array{T}(undef, size(x))
  prox!(p, f, x, one(T))
  p .= x .- p
  copyto!(y, p)
  op === prox! ? y : zero(T)
end
function l1_route_result!(op, y, f, x, γ, master)
  try
    v = master ? l1_moreau_result!(op, y, f, x, γ) : op(y, Conjugate(f), x, γ)
    (nothing, op === prox! ? nothing : v)
  catch e
    (typeof(e), nothing)
  end
end

@testset "L1 projections require known one-based host storage" begin
  for op in (prox!, value_and_prox!), base in (NormLinf(), GroupNormLinf(1, [1:2])),
      offset_input in (false, true), offset_output in (false, true)
    offset_input || offset_output || continue
    x = offset_input ? ProjectionOffset([0.1, 0.2]) : [1e300, 0.25]
    y = offset_output ? ProjectionOffset([7.0, 7.0]) : [7.0, 7.0]
    expected = offset_output ? ProjectionOffset([7.0, 7.0]) : [7.0, 7.0]
    @test ProximalBase._projection_fallback(base, y, x)
    offset_input && @test !ProximalBase._projection_host(x)
    offset_output && @test !ProximalBase._projection_host(y)
    result = l1_route_result!(op, y, base, x, 0.7, false)
    reference = l1_route_result!(op, expected, base, x, 0.7, true)
    @test result == reference
    @test parent(y) == parent(expected)
    result[1] !== nothing && offset_input && @test parent(y) == [7, 7]
  end
  for op in (prox!, value_and_prox!), base in (NormLinf(), GroupNormLinf(1, [1:2]))
    x = ProjectionUnknown([1e300, 0.25])
    @test parent(x) === x
    @test !ProximalBase._projection_host(x)
    @test !ProximalBase._direct_projection(base, x)
    @test !ProximalBase._projection_host(view(x, :))
    y, expected = zeros(2), zeros(2)
    @test l1_route_result!(op, y, base, x, 1.0, false) == l1_route_result!(op, expected, base, x, 1.0, true)
    @test y == expected == zeros(2)
  end
  # Fallback is blockwise: an unsupported L1 block must not turn another
  # block's direct box projection into cancellation in Moreau's identity.
  for op in (prox!, value_and_prox!), unknown_input in (false, true)
    f = Scaled(SeparableSum(NormL1(), SeparableSum(NormLinf(), SqrNormL2(2))), 0.7)
    block = unknown_input ? ProjectionUnknown([1e300, 0.25]) : [1e300, 0.25]
    x = ([1e300], (block, [3.0]))
    target = unknown_input ? zeros(2) : ProjectionUnknown(zeros(2))
    y = (zeros(1), (target, zeros(1)))
    v = op(y, Conjugate(f), x, 0.7)
    p = zeros(1)
    expected = value_and_prox!(p, Conjugate(Scaled(SqrNormL2(2), 0.7)), [3.0], 0.7)
    @test y[1] == [0.7]
    @test target == zeros(2)
    @test y[2][2] == p
    op === value_and_prox! && @test v == expected
  end
  A = [1e300 0.0; 0.0 0.25]
  for x in (Diagonal([1e300, 0.25]), PermutedDimsArray(A, (2, 1)),
            view(A, :, :), reshape(A, 4), reinterpret(Float64, reinterpret(UInt64, A))),
      op in (prox!, value_and_prox!)
    @test ProximalBase._projection_host(x)
    @test ProximalBase._direct_projection(NormLinf(), x)
    y = zeros(size(x))
    @test !ProximalBase._projection_fallback(NormLinf(), y, x)
    op(y, Conjugate(NormLinf()), x, 1.0)
    @test y == ifelse.(abs.(x) .== 1e300, 1.0, 0.0)
  end
end

@testset "Condat discarded-level enclosure" begin
  setprecision(BigFloat, 4096) do
    for T in (Float32, Float64), n in (64, 257)
      data = T[1 + mod(17i, 31)/32 for i in 1:n]
      for x in (data, sort(data), sort(data; rev = true), vcat(data[end:end], data[1:end-1])),
          frac in (0.001, 0.1, 0.9)
        r = T(frac * sum(abs, Float64.(x)))
        for op in (prox!, value_and_prox!)
          y = similar(x)
          op(y, Conjugate(NormLinf(r)), x, one(T))
          l1accurate(y, projl1(x, BigFloat(r)), BigFloat(r); support = false)
        end
      end
    end
  end
end

@testset "tuple fallback retains nested scales" begin
  for op in (prox!, value_and_prox!)
    inner = Scaled(SeparableSum(NormLinf(1.0), SqrNormL2()), 3.0)
    c = Conjugate(Scaled(SeparableSum(NormL1(), inner), 2.0))
    x = ([10.0], ([10.0], [14.0]))
    y = (zeros(1), (ProjectionUnknown(zeros(1)), zeros(1)))
    v = op(y, c, x, 1.0)
    @test y[1] == [2.0]
    @test y[2][1] == [6.0]
    @test y[2][2] == [12.0]
    op === value_and_prox! && @test v ≈ 12.0
  end
  setprecision(BigFloat, 4096) do
    for T in (Float32, Float64), mode in (:sum, :array, :translated), depth in (1, 2),
        unknown_input in (false, true), op in (prox!, value_and_prox!)
      shift = mode === :translated ? T(0.5) : zero(T)
      ball = mode === :translated ? Translate(NormLinf(one(T)), T[shift]) : NormLinf(one(T))
      inner = mode === :sum ? SeparableSum(ball, SqrNormL2()) : ball
      factors = depth == 1 ? (T(3),) : (T(3), T(5))
      for a in factors
        inner = Scaled{typeof(inner),T}(inner, a)
      end
      mode === :sum || (inner = SeparableSum(inner, SqrNormL2()))
      c = Conjugate(Scaled(SeparableSum(NormL1(), inner), T(2)))
      block = unknown_input ? (ProjectionUnknown(T[10]), ProjectionUnknown(T[14])) : (T[10], T[14])
      x = (T[10], block)
      y = (zeros(T, 1), (ProjectionUnknown(zeros(T, 1)), zeros(T, 1)))
      v = op(y, c, x, one(T))
      # Independent scalar ball and quadratic conjugates, with exact factors.
      radius = BigFloat(2) * prod(BigFloat.(factors))
      ball_ref = min(BigFloat(10) - BigFloat(shift), radius)
      coefficient = mode === :sum ? radius : BigFloat(2)
      quadratic_ref = 14coefficient / (coefficient + 1)
      value_ref = BigFloat(shift) * ball_ref + quadratic_ref^2 / (2coefficient)
      accurate(y[1], BigFloat[2], 8)
      accurate(y[2][1], [ball_ref], 8)
      accurate(y[2][2], [quadratic_ref], 8)
      op === value_and_prox! && accurate(T[v], [value_ref], 8)
    end
  end
end

@testset "translated L1 conjugates accept non-finite inputs" begin
  # Master returns a NaN point and NaN translated value for these inputs.
  for T in (Float32, Float64), entry in (T(Inf), T(NaN)),
      base in (NormLinf(one(T)), GroupNormLinf(one(T), [1:1])),
      op in (prox!, value_and_prox!)
    f = Conjugate(Translate(base, zeros(T, 1)))
    y = fill(T(7), 1)
    v = op(y, f, T[entry], one(T))
    @test isequal(y, T[NaN])
    op === value_and_prox! && @test isequal(v, T(NaN))
  end
  # Finite inputs still reject overflow in the shift or shifted input.
  for op in (prox!, value_and_prox!), shift_overflow in (false, true)
    h = floatmax(Float64)
    x, b, γ = shift_overflow ? ([0.0], [h], 2.0) : ([h], [-h], 1.0)
    y = [7.0]
    @test_throws ArgumentError op(y, Conjugate(Translate(NormLinf(), b)), x, γ)
    @test y == [7.0]
  end
end

@testset "non-finite L1 blocks retain Moreau results" begin
  # These points and values were checked on master 5edf73f, not a NaN fill.
  for T in (Float32, Float64), raw in ([Inf, 1], [NaN, 1], [-Inf, 2, 3]),
      grouped in (false, true), mode in (:bare, :scaled, :translated),
      tuple_block in (false, true), op in (prox!, value_and_prox!), γ in (one(T), T(0.7))
    n = length(raw)
    x = T.(grouped ? vcat(raw, [4, 2]) : raw)
    base = grouped ? GroupNormLinf(one(T), [1:n, n+1:n+2]) : NormLinf(one(T))
    f = mode === :bare ? base : mode === :scaled ? Scaled(base, T(2)) : Translate(base, zero(x))
    c = Conjugate(tuple_block ? SeparableSum(f, NormLinf(one(T))) : f)
    input = tuple_block ? (x, T[3, 0]) : x
    y = tuple_block ? (fill(T(7), length(x)), fill(T(7), 2)) : fill(T(7), length(x))
    if mode === :translated && isnan(first(raw))
      # Inherited: master's translation check rejects NaN contamination of
      # an originally finite coordinate, even for a non-finite input.
      @test_throws ArgumentError op(y, c, input, γ)
      continue
    end
    expected = isnan(first(raw)) ? fill(T(NaN), n) : vcat(T[NaN], zeros(T, n - 1))
    # Master's reciprocal-step value route leaves this rounding residual.
    if mode === :translated && (tuple_block || (grouped && op === value_and_prox!)) && γ != one(T) && n == 3
      expected[3] = 2eps(T)
    end
    v = op(y, c, input, γ)
    output = tuple_block ? y[1] : y
    @test isequal(output[1:n], expected)
    # Finite groups/blocks keep the direct path, within its existing bound.
    if grouped
      r = mode === :scaled ? T(2) : one(T)
      setprecision(BigFloat, 4096) do
        l1accurate(output[n+1:n+2], projl1(T[4, 2], BigFloat(r)), BigFloat(r); support = false)
      end
    end
    if tuple_block
      setprecision(BigFloat, 4096) do
        l1accurate(y[2], BigFloat[1, 0], BigFloat(1); support = false)
      end
    end
    op === value_and_prox! && @test isequal(v, mode === :translated ? T(NaN) : zero(T))
  end
  for T in (Float32, Float64), op in (prox!, value_and_prox!),
      f in (NormLinf(one(T)), GroupNormLinf(one(T), [1:2, 3:4]))
    x = f isa NormLinf ? T[Inf, 1] : T[Inf, 1, 4, 2]
    expected = f isa NormLinf ? T[NaN, 0] : T[NaN, 0, 1, 0]
    v = op(x, Conjugate(f), x, one(T))
    @test isequal(x, expected)
    op === value_and_prox! && @test iszero(v)
  end
end

@testset "L1 projection coefficient and storage coverage" begin
  setprecision(BigFloat, 4096) do
    # A machine-integer coefficient rounds on conversion to the working type.
    for λ in (Int128(2)^60 + 1, Float16(0.5)), op in (prox!, value_and_prox!)
      x = λ isa Int128 ? [1e300, 0.25] : Float16[1000, 999, 998]
      y = similar(x)
      v = op(y, Conjugate(NormLinf(λ)), x, one(eltype(x)))
      l1accurate(y, projl1(x, BigFloat(λ)), BigFloat(λ))
      op === value_and_prox! && @test iszero(v)
    end
    inputs = (0:2, 0:2:4, 0.0:0.25:1.0, LinRange(0.0, 1.0, 5), Base.OneTo(3),
      sparsevec([1, 3], [4.0, 2.0], 3),
      sparse([4.0 0.0; 0.0 2.0]), SparseIterate([4.0, 0.0, 2.0]),
      SymmetricSparseIterate([4.0 0.0; 0.0 2.0]),
      AtomIterate([4.0, 0.0, 2.0], [1:2, 3:3]),
      Bidiagonal([4.0, 2.0], [0.0], :U), SymTridiagonal([4.0, 2.0], [0.0]),
      Tridiagonal([0.0], [4.0, 2.0], [0.0]))
    for x in inputs, op in (prox!, value_and_prox!)
      y = zeros(size(x))
      v = op(y, Conjugate(NormLinf(0.75)), x, 0.7)
      l1accurate(vec(y), projl1(vec(Array(x)), BigFloat(0.75)), BigFloat(0.75))
      op === value_and_prox! && @test iszero(v)
    end
    # Bool inputs are outside the documented floating/machine-integer domain.
    for op in (prox!, value_and_prox!)
      y = zeros(2)
      @test_throws ArgumentError op(y, Conjugate(NormLinf()), BitArray([true, false]), 1.0)
      @test y == zeros(2)
      bits = falses(2)
      v = op(bits, Conjugate(NormLinf()), [2.0, 0.0], 1.0)
      @test bits == BitArray([true, false])
      op === value_and_prox! && @test iszero(v)
    end
    @static if isdefined(Base, :Memory)
      x = Base.Memory{Float64}(undef, 3)
      x .= [4, 0, 2]
      for op in (prox!, value_and_prox!)
        y = zeros(3)
        v = op(y, Conjugate(NormLinf(0.75)), x, 0.7)
        l1accurate(y, projl1(collect(x), BigFloat(0.75)), BigFloat(0.75))
        op === value_and_prox! && @test iszero(v)
      end
    end
    # Different precision and non-Array groups use the generic dispatch.
    for op in (prox!, value_and_prox!)
      x = view(Float32[4, 3, 2, 1], :)
      y = zeros(4)
      f = GroupNormLinf(0.75f0, [1:2, 3:4])
      v = op(y, Conjugate(f), x, 0.7)
      accurate(y, reference(f, x, 0.7), 4)
      op === value_and_prox! && @test iszero(v)
    end
    # The secondary membership certificate rounds subnormal terms upward.
    x = [1.0, nextfloat(0.0)]
    for op in (prox!, value_and_prox!)
      y = similar(x)
      v = op(y, Conjugate(NormLinf()), x, 1.0)
      l1accurate(y, projl1(x, BigFloat(1)), BigFloat(1))
      op === value_and_prox! && @test iszero(v)
    end
  end
end

@testset "L1 projection tuple conversion coverage" begin
  for op in (prox!, value_and_prox!)
    f = Conjugate(SeparableSum(Scaled(NormLinf(0.5), 2.0), NormLinf(0.5)))
    x = ([3.0], [3.0])
    y = (Float32[7], Float64[7])
    v = op(y, f, x, 1.0)
    @test y == (Float32[1], [0.5])
    op === value_and_prox! && @test iszero(v)
    y = (Float32[7], Int[7])
    @test_throws InexactError op(y, f, x, 1.0)
    @test y == (Float32[7], Int[7])
    @test op((), Conjugate(SeparableSum()), (), 1.0) == (op === prox! ? () : 0)
    # Aliased group weights are staged before the destination is committed.
    f = GroupNormLinf(1.0, [1:1, 2:2], [1.0, 2.0])
    v = op(f.w, Conjugate(Scaled(f, 1.0)), [0.5, 4.0], 1.0)
    @test f.w == [0.5, 2.0]
    op === value_and_prox! && @test iszero(v)
  end
end

@testset "L1 projection Condat reinsertion coverage" begin
  setprecision(BigFloat, 4096) do
    # The first two candidates average below 2.5-r, but 1.9 exceeds it.
    # The reset discards both; finishing must reinsert 1.9.
    for T in (Float32, Float64), op in (prox!, value_and_prox!)
      x = vcat(T[1, 1.9, 2.5], zeros(T, 30))
      y = similar(x)
      v = op(y, Conjugate(NormLinf(one(T))), x, one(T))
      l1accurate(y, projl1(x, BigFloat(1)), BigFloat(1); support = false)
      op === value_and_prox! && @test iszero(v)
    end
  end
  # Scaling a finite MPFR extreme by this tiny radius overflows the
  # membership scan. The scalar projection is independently min(abs(x), r).
  for op in (prox!, value_and_prox!)
    x, y, r = setprecision(BigFloat, 4) do
      r = floatmin(BigFloat)
      x, y = [floatmax(BigFloat)], [BigFloat(7)]
      v = op(y, Conjugate(NormLinf(r)), x, BigFloat(1))
      op === value_and_prox! && @test iszero(v)
      x, y, r
    end
    setprecision(BigFloat, 4096) do
      @test y == [min(abs(BigFloat(only(x))), BigFloat(r))]
    end
  end
  # Low-precision BigFloat reaches the n*eps guard with a small public input.
  for op in (prox!, value_and_prox!)
    x, y, u = setprecision(BigFloat, 4) do
      x = fill(BigFloat(2), 8)
      y = similar(x)
      v = op(y, Conjugate(NormLinf(BigFloat(1))), x, BigFloat(1))
      @test all(iszero, y)
      op === value_and_prox! && @test iszero(v)
      x, y, eps(BigFloat)
    end
    setprecision(BigFloat, 4096) do
      ref = projl1(x, BigFloat(1))
      @test all(abs.(y .- ref) .<= 4length(x) * u)
    end
  end
end

@testset "ungrouped conjugate coordinates are always zero" begin
  for T in (Float32, Float64), entry in (T(Inf), T(-Inf), T(NaN)),
      F in (GroupNormL2, GroupNormLinf), op in (prox!, value_and_prox!),
      tuple_block in (false, true), alias in (false, true), mode in (:bare, :scaled, :translated)
    # Cover a gap inside group_of and an ungrouped tail beyond group_of.
    base = F(one(T), [1:1, 3:3], ones(T, 2))
    f = mode === :bare ? base : mode === :scaled ? Scaled(base, T(2)) : Translate(base, zeros(T, 4))
    x = T[2, entry, -3, entry]
    c = Conjugate(tuple_block ? SeparableSum(f, NormLinf(one(T))) : f)
    input = tuple_block ? (x, T[2]) : x
    y = alias ? input : tuple_block ? (fill(T(7), 4), T[7]) : fill(T(7), 4)
    v = op(y, c, input, T(0.7))
    r = mode === :scaled ? T(2) : one(T)
    @test isequal(tuple_block ? y[1] : y, T[r, 0, -r, 0])
    tuple_block && @test y[2] == T[1]
    op === value_and_prox! && @test iszero(v)
  end
end

@testset "exceptional conjugate oracles return their destination" begin
  for T in (Float32, Float64), base in (NormLinf(one(T)), GroupNormLinf(one(T), [1:2])),
      scaled in (false, true), tuple_block in (false, true), alias in (false, true)
    f = scaled ? Scaled(base, T(2)) : base
    c = Conjugate(tuple_block ? SeparableSum(f, NormLinf(one(T))) : f)
    x = tuple_block ? (T[Inf, 1], T[2]) : T[Inf, 1]
    y = alias ? x : deepcopy(x)
    @test prox!(y, c, x, one(T)) === y
    expected = tuple_block ? (T[NaN, 0], T[1]) : T[NaN, 0]
    @test isequal(y, expected)
    input = tuple_block ? (T[Inf, 1], T[2]) : T[Inf, 1]
    @test isequal(prox(c, input, one(T)), expected)
  end
end

@testset "exceptional translations retain original Moreau arithmetic" begin
  # These literal points and NaN values were checked on archived master.
  for T in (Float32, Float64), F in (NormLinf, GroupNormLinf),
      mode in (:bare, :scaled, :nested), tuple_block in (false, true),
      op in (prox!, value_and_prox!), alias in (false, true)
    base = F === NormLinf ? F(one(T)) : F(one(T), [1:2])
    b = T[0.5, 2.0^100]
    f = Translate(base, b)
    f = mode === :scaled ? Scaled(f, T(2)) : mode === :nested ?
      Scaled(Translate(Scaled(f, T(3)), T[0.25, 0.25]), T(2)) : f
    c = Conjugate(tuple_block ? SeparableSum(f, NormLinf(one(T))) : f)
    input = tuple_block ? (T[Inf, 1], T[2]) : T[Inf, 1]
    y = alias ? input : deepcopy(input)
    v = op(y, c, input, one(T))
    # The outer 0.25 survives the inner huge cancellation on master.
    second = mode === :nested ? T(0.75) : one(T)
    expected = tuple_block ? (T[NaN, second], T[1]) : T[NaN, second]
    @test isequal(y, expected)
    if op === prox!
      @test v === y
      x = tuple_block ? (T[Inf, 1], T[2]) : T[Inf, 1]
      @test isequal(prox(c, x, one(T)), expected)
    else
      @test isnan(v)
    end
  end
end

@testset "tuple recovery preserves protected L1 groups" begin
  for T in (Float32, Float64), λ in (zero(T), one(T), T(Inf)),
      translated in (false, true), op in (prox!, value_and_prox!), alias in (false, true)
    base = GroupNormLinf(λ, [[1, 2], [3], [4], [5, 6]], T[Inf, 1, 0, 1])
    f = translated ? Translate(base, fill(T(0.25), 6)) : base
    c = Conjugate(SeparableSum(f, NormLinf(one(T))))
    input = (T[NaN, 1, Inf, NaN, 4, 2], T[2])
    y = alias ? input : deepcopy(input)
    shift = translated ? T(0.25) : zero(T)
    tail = iszero(λ) ? T[0, 0] : isinf(λ) ? T[4-shift, 2-shift] : T[1, 0]
    # Master contaminates the free NaN group; #300 instead keeps it free,
    # including λ0=0. Zero groups and finite shortcuts must not be rerun.
    expected = (vcat(T[NaN, 1-shift, iszero(λ) ? 0 : isinf(λ) ? Inf : NaN, 0], tail), T[1])
    v = op(y, c, input, one(T))
    @test isequal(y, expected)
    op === prox! ? (@test v === y) : (@test isequal(v, translated ? T(NaN) : zero(T)))
  end
end

@testset "exceptional unit groups recover only their own input" begin
  # Master's original translated group computation returns 1 after the
  # huge cancellation. Finite groups retain their direct projections.
  for T in (Float32, Float64), translated in (false, true), tuple_block in (false, true),
      op in (prox!, value_and_prox!), alias in (false, true)
    base = GroupNormLinf(one(T), [1:2, 3:4, 5:6])
    f = translated ? Translate(base, T[0.25, 0.25, 0.5, 2.0^100, 0.25, 0.25]) : base
    c = Conjugate(tuple_block ? SeparableSum(f, NormLinf(one(T))) : f)
    input = tuple_block ? (T[4, 2, Inf, 1, 3, 1], T[2]) : T[4, 2, Inf, 1, 3, 1]
    y = alias ? input : deepcopy(input)
    point = T[1, 0, NaN, translated ? 1 : 0, 1, 0]
    expected = tuple_block ? (point, T[1]) : point
    v = op(y, c, input, one(T))
    @test isequal(y, expected)
    op === prox! ? (@test v === y) : (@test isequal(v, translated ? T(NaN) : zero(T)))
  end
end

function exceptional_group_time(n)
  x = fill(NaN, n)
  y = similar(x)
  c = Conjugate(GroupNormLinf(1.0, [i:i for i in 1:n]))
  prox!(y, c, x, 1.0)
  times = ntuple(3) do _
    GC.gc()
    @elapsed prox!(y, c, x, 1.0)
  end
  @test all(isnan, y)
  minimum(times)
end

@testset "exceptional group recovery scales linearly" begin
  # A tenfold increase costs about 10x for linear traversal and 100x for
  # either former search. Warm up and take minima to reduce GC/scheduling noise.
  small = exceptional_group_time(10^4)
  large = exceptional_group_time(10^5)
  @test large < 40small

  # Literal points and values checked on archived master 5edf73f.
  for T in (Float32, Float64), translated in (false, true), op in (prox!, value_and_prox!)
    base = GroupNormLinf(one(T), [i:i for i in 1:5])
    f = translated ? Translate(base, fill(T(0.5), 5)) : base
    x = T[NaN, -3, NaN, 2, NaN]
    y = zeros(T, 5)
    v = op(y, Conjugate(f), x, one(T))
    @test isequal(y, T[NaN, -1, NaN, 1, NaN])
    op === prox! ? (@test v === y) : (@test isequal(v, translated ? T(NaN) : zero(T)))
  end
end

end
