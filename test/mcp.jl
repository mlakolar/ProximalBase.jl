module MCPTest

using Test, ProximalBase, OffsetArrays

include("mcp_reference.jl")
using .MCPReferenceTest

function agrees(actual::T, reference) where {T<:AbstractFloat}
  target = T(reference)
  isequal(actual, target) ||
    (isfinite(actual) && isfinite(target) && isapprox(actual, target; rtol=6eps(T), atol=2nextfloat(zero(T))))
end

function dense(x)
  x isa SymmetricSparseIterate ? Matrix(x) : Array(x)
end

snapshot(x::AbstractArray) = dense(x)
snapshot(x::Union{SparseIterate,BlockIterate,SymmetricSparseIterate}) =
  (dense(x), nactive(x), collect(activecoordinates(x)))

@testset "MCP constructor and traits" begin
  @test MCP(2).a === 3
  @test MCP(2f0).a === 3f0
  @test MCP(2, 0.25).a === 0.25
  @test MCP(1.0, Inf).a === Inf
  for λ in (-1, -Inf, NaN, -big"1e-400")
    @test_throws ArgumentError MCP(λ)
  end
  for a in (0, -0.0, -1, -Inf, NaN)
    @test_throws ArgumentError MCP(1, a)
  end
  @test_throws ArgumentError MCP{Float32}(-1f0, 3f0)
  @test_throws ArgumentError MCP{Float32}(1f0, 0f0)
  f = MCP(0.5)
  @test isseparable(f) && isproximable(f) && iszeropreserving(f)
  @test !isconvex(f) && !issmooth(f)
  @test restrict(f, [4, 2]) === f
end

@testset "MCP in $T" for T in (Float32, Float64)
  z, u = zero(T), one(T)
  δ, M = nextfloat(z), floatmax(T)
  λ, a = T(0.5), T(2)
  f = MCP(λ, a)

  @testset "full objective and adjacent inputs" begin
    @test @inferred(prox_block(f, T(0.75), 1, u)) === T(0.5)
    @test @inferred(value_entry(f, -u, 1)) === T(0.25)
    for (level, shape, steps) in ((λ, a, T[δ, 1, prevfloat(a), a, nextfloat(a), 8]),
                                  (T(0.75), T(0.5), T[0.125, 0.5, 4]))
      g = MCP(level, shape)
      for γ in steps
        lo, k = γ * level, shape * level
        h = T(BigFloat(level) * sqrt(BigFloat(shape) * BigFloat(γ)))
        magnitudes = T[0, δ, 0.25, 0.75, 1.5, 3, lo, k, h]
        for cut in (lo, k, h)
          append!(magnitudes, (max(z, prevfloat(cut)), nextfloat(cut)))
        end
        x = [magnitudes; -magnitudes]
        want = [mcp_reference(level, shape, v, γ) for v in x]
        y = prox(g, x, γ)
        for i in eachindex(x)
          point = prox_block(g, x[i], i, γ)
          @test point === y[i]
          if γ >= shape || iszero(want[i]) || want[i] == x[i]
            @test point === T(want[i])
          else
            @test !iszero(point) || abs(want[i]) <= δ
            @test signbit(point) == signbit(want[i])
            @test agrees(point, want[i])
          end
          @test agrees(value_entry(g, x[i], i), mcp_reference_value(level, shape, x[i]))
        end
      end
    end

    for (γ, tie) in ((a, u), (T(8), T(2))), s in (-u, u)
      @test prox_block(f, s * prevfloat(tie), 1, γ) === z
      @test prox_block(f, s * tie, 1, γ) === z
      @test prox_block(f, s * nextfloat(tie), 1, γ) === s * nextfloat(tie)
      for t in (prevfloat(tie), tie, nextfloat(tie))
        point = prox_block(f, s * t, 1, γ)
        H(p) = mcp_reference_objective(λ, a, s * t, γ, p)
        @test H(point) <= min(H(z), H(s * a * λ), H(s * max(t, a * λ)))
      end
    end
    # Independent bounded search at a firm point, an affine regime point
    # above the kink, and hard-selection points on both sides of the switch.
    for (v, γ) in ((T(0.75), u), (T(-1.25), a), (T(1.5), T(8)), (T(-3), T(8)))
      want = mcp_reference_search(λ, a, v, γ)
      @test isapprox(BigFloat(prox_block(f, v, 1, γ)), want; atol=big"1e-12")
    end

    # The rounded upper kink is below the actual product. Keeping this
    # input would skip the short but nonempty firm interval.
    kpar = nextfloat(u)
    v = kpar * kpar
    want = mcp_reference(kpar, kpar, v, u)
    @test T(want) < v
    @test prox_block(MCP(kpar, kpar), v, 1, u) === T(want)
  end

  @testset "finite extremes and subnormals" begin
    e = T === Float32 ? 70 : 600
    b, c = ldexp(u, e), ldexp(u, -e)
    for (level, shape, γ, v) in (
      (b, a, T(8), ldexp(b, 2)), (b, a, T(8), nextfloat(ldexp(b, 2))),
      (c, a, T(8), ldexp(c, 2)), (c, a, T(8), nextfloat(ldexp(c, 2))),
      (M / 2, T(4), T(0.5), M), (M, T(4), a, M), (M, a, a, M),
      (T(1.5), 4δ, 3δ, 5δ), (T(1.5), δ, δ, δ), (T(1.5), δ, δ, 2δ),
      (T(0.75), δ, 2δ, δ), (δ, floatmin(T), u, δ))
      g = MCP(level, shape)
      for x in (v, -v)
        want = mcp_reference(level, shape, x, γ)
        got = prox_block(g, x, 1, γ)
        @test γ >= shape || v <= 5δ ? got === T(want) : agrees(got, want)
        @test agrees(value_entry(g, x, 1), mcp_reference_value(level, shape, x))
      end
    end
    # Overflow/underflow of λ² or aλ need not overflow/underflow the value.
    for (level, shape, v) in ((b, ldexp(u, -e), T(4)), (c, b, T(4)),
                               (M, M, inv(M)), (T(1.5), δ, T(Inf)),
                               (b, c, T(Inf)), (T(0.5), T(4), 8δ))
      want = mcp_reference_value(level, shape, v)
      @test agrees(value_entry(MCP(level, shape), v, 1), want)
    end
    @test prox_block(MCP(T(1.5), 4δ), 5δ, 1, 3δ) === 2δ
    @test prox_block(MCP(δ, T(3)), -δ, 1, T(0.75)) === -z
    @test prox_block(MCP(δ, T(Inf)), -δ, 1, T(0.75)) === -z
    @test prox_block(MCP(b, a), -nextfloat(4b), 1, T(8)) === -nextfloat(4b)
  end

  @testset "endpoint matrix and exact aliasing" begin
    x = T[-Inf, -2, -0.0, 0, 0.75, 2, Inf, NaN]
    for level in (z, λ, T(Inf)), shape in (a, T(Inf))
      g = MCP(level, shape)
      @test g(T[]) === z
      for i in eachindex(x)
        @test agrees(value_entry(g, x[i], i), mcp_reference_value(level, shape, x[i]))
      end
      @test isequal(g(x), iszero(level) ? z : T(NaN))
      for γ in (z, u, a, T(8), T(Inf))
        want = T[mcp_reference(level, shape, v, γ) for v in x]
        got = prox(g, x, γ)
        @test isequal(got, want)
        for step in (γ, fill(γ, size(x)))
          @test isequal(prox(g, x, step), want)
          y = copy(x)
          @test isequal(value_and_prox!(y, g, y, step), g(y))
          @test isequal(y, want)
          y .= x
          @test prox!(y, g, y, step) === y
          @test isequal(y, want)
        end
      end
    end
  end

  @testset "dense shapes and stored values" begin
    x = T[-2, -0.75, 3, 0.5, 2, 0.25]
    steps = T[8, 1, 8, 0, 8, 2]
    for point in (x, reshape(x, 2, 3), view(x, 1:2:6))
      γ = view(steps, 1:length(point))
      for step in (u, γ)
        want = T[mcp_reference(λ, a, v, step isa Real ? step : step[i]) for (i, v) in enumerate(point)]
        y = similar(point)
        @test value_and_prox!(y, f, point, step) === f(y)
        @test vec(y) == vec(want)
        @test f(point) ≈ sum(v -> T(mcp_reference_value(λ, a, v)), point)
      end
    end
    # A positive scale forwards its effective step to the leaf.
    s = Scaled(f, T(2))
    y = similar(x)
    @test prox(s, x, T(4)) == prox(f, x, T(8))
    @test value_and_prox!(y, s, x, u) === s(y)
    @test y == prox(f, x, T(2))
    @test s(x) === T(2) * f(x)
    @test_throws ArgumentError Scaled(f, z)
  end

  @testset "prepared scalar thresholds" begin
    kpar = nextfloat(u)
    near_kink = kpar * kpar
    cases = T[-Inf, -3, -2, -near_kink, -u, -0.75, -0.5, -δ, -z, z, δ, 0.5,
              nextfloat(T(0.5)), 0.75, prevfloat(u), u, nextfloat(u), near_kink,
              prevfloat(T(2)), 2, nextfloat(T(2)), 3, Inf, NaN]
    for (g, γ) in ((f, u), (f, prevfloat(a)), (f, a), (f, nextfloat(a)), (f, T(8)),
                   (MCP(kpar, kpar), u), (MCP(T(1.5), 4δ), 3δ), (MCP(M, a), a))
      x = repeat(cases, 4)
      want = [prox_block(g, v, i, γ) for (i, v) in enumerate(x)]
      y = prox(g, x, γ)
      @test all(isequal.(y, want) .| (isfinite.(y) .& isfinite.(want) .&
                                      isapprox.(y, want; rtol=4eps(T), atol=z)))
      @test signbit.(y) == signbit.(want)
      @test iszero.(y) == iszero.(want)
      @test isequal(value_and_prox!(x, g, x, γ), g(x))
      @test isequal(x, y)
    end
  end

  @testset "long arrays with variable steps" begin
    hard = T[0.25, -0.25, 3, -4, 0, -0.0, Inf, -Inf, NaN]
    firm = T[0.25, -0.25, 0.5, -nextfloat(T(0.5)), prevfloat(u), u, nextfloat(u),
             0, -0.0, Inf, -Inf, NaN]
    t = nextfloat(T(2))
    # The rounded squares agree, but the exact input square is larger.
    hard_boundary = [T[t, -t, prevfloat(T(2)), 2, t]; fill(T(3), 251)]
    boundary_steps = [fill(2 * (t * t), 2); fill(T(8), 254)]
    x = [repeat(firm, 22)[1:256]; repeat(hard, 29)[1:256];
         hard_boundary;
         T[prevfloat(T(2)), 2, nextfloat(T(2)), -2, -nextfloat(T(2)), -0.0, Inf, -Inf,
           δ, -δ, M, -M, 0.75, 3]]
    γ = [repeat(T[u, prevfloat(a)], 128); repeat(T[8, 6], 128);
         boundary_steps;
         T[8, 8, 8, 8, 8, 0, Inf, 1, δ, 8, 8, 1, 1, 8]]
    expected = T[mcp_reference(λ, a, v, s) for (v, s) in zip(x, γ)]
    @test expected[513:514] == T[t, -t]
    for shape in ((length(x),), (2, length(x) ÷ 2))
      point, steps = reshape(x, shape), reshape(γ, shape)
      y = similar(point)
      @test prox!(y, f, point, steps) === y
      for i in eachindex(y)
        if steps[i] >= a || iszero(expected[i]) || !isfinite(expected[i])
          @test isequal(y[i], expected[i])
        else
          @test signbit(y[i]) == signbit(expected[i])
          @test agrees(y[i], expected[i])
        end
      end
      zpoint = copy(point)
      @test isequal(value_and_prox!(zpoint, f, zpoint, steps), f(zpoint))
      @test isequal(zpoint, y)
      for bad in (-u, T(NaN)), op in (prox!, value_and_prox!)
        badsteps = copy(steps)
        badsteps[end] = bad
        zpoint = copy(point)
        @test_throws ArgumentError op(zpoint, f, zpoint, badsteps)
        @test isequal(zpoint, point)
      end
    end
  end

  @testset "prepared dense and block values" begin
    for g in (f, MCP(z, a), MCP(T(Inf), a), MCP(λ, T(Inf)), MCP(δ, a), MCP(T(1.125), M))
      x = T[-3, -u, -0.0, 0, δ, 0.5, u, Inf, -Inf]
      for entries in (x, [x; T(NaN)])
        blocks = BlockIterate(entries, [collect(j:2:length(entries)) for j in 1:2])
        for point in (entries, blocks)
          want = sum(value_entry(g, v, i) for (i, v) in enumerate(entries))
          @test isequal(g(point), want) || agrees(g(point), want)
        end
      end
    end
  end

  @testset "iterate storage and failure safety" begin
    sparse = SparseIterate(T, 6)
    sparse[2], sparse[3], sparse[5] = T(-2), T(0.5), T(3)
    sparse[5] = z
    block = BlockIterate(T[0, -2, 0.5, 0, 0, 0, 0, 0], [[2, 5], [1, 6], [3, 4], [7, 8]])
    activate!(block, 2)
    sym = SymmetricSparseIterate(T, 3)
    sym[1, 1], sym[2, 1], sym[3, 1], sym[3, 3] = T(0.5), T(-2), T(7), T(3)
    sym[3, 1] = z
    @test ProximalBase.check_symmetric_parameters(f, sym) === true
    @test f(sym) === T(0.9375)
    @test f(sym) === f(dense(sym))
    @test nactive(sparse) == 3 && nactive(block) == 3 && nactive(sym) == 4
    @test !isactive(block, 4)

    for x in (sparse, block, sym)
      point = dense(x)
      ids = collect(activecoordinates(x))
      for step in (x isa SymmetricSparseIterate ? (u, T(8)) : (u, T[u, 8, 1, 2, 0, 8, 1, 2][1:length(x)]))
        want = prox(f, point, step)
        y = copy(x)
        @test f(x) ≈ f(point)
        @test dense(prox(f, x, step)) == want
        @test prox!(y, f, x, step) === y
        @test dense(y) == want
        @test collect(activecoordinates(y)) == ids
        y = copy(x)
        @test value_and_prox!(y, f, y, step) === f(y)
        @test dense(y) == want
        @test collect(activecoordinates(y)) == ids
      end
    end
    @test prox(f, dense(sym), u) == prox!(zeros(T, size(sym)), f, sym, u)

    x = T[2, 0.5, -3]
    for op in (prox!, value_and_prox!), bad in (-u, T(NaN))
      for point in (x, sparse, block, sym)
        for y in (copy(point), point)
          before = snapshot(y)
          @test_throws ArgumentError op(y, f, point, bad)
          @test isequal(snapshot(y), before)
        end
      end
      for point in (x, sparse, block)
        # The last entry is inactive in both iterate fixtures.
        γ = fill(u, length(point))
        γ[end] = bad
        for y in (copy(point), point)
          before = snapshot(y)
          @test_throws ArgumentError op(y, f, point, γ)
          @test isequal(snapshot(y), before)
        end
      end
      @test_throws ArgumentError prox(f, x, bad)
      @test_throws ArgumentError prox(f, x, T[u, u, bad])
    end
    for op in (prox!, value_and_prox!)
      y = copy(x)
      @test_throws DimensionMismatch op(y, f, x[1:2], u)
      @test_throws DimensionMismatch op(y, f, x, T[u, u])
      @test_throws ArgumentError op(y, f, OffsetArray(x, 0:2), u)
      @test_throws ArgumentError op(y, f, x, OffsetArray(fill(u, 3), 0:2))
      @test_throws ArgumentError op(y, f, complex.(x), u)
      @test_throws ArgumentError op(y, f, x, complex.(fill(u, 3)))
      @test y == x
      before = snapshot(sym)
      for point in (sym, dense(sym))
        @test_throws ArgumentError op(sym, f, point, ones(T, size(sym)))
        @test isequal(snapshot(sym), before)
      end
    end
    @test_throws ArgumentError prox(f, sym, ones(T, size(sym)))
  end
end

@testset "fused value uses the stored point at the kink" begin
  level = nextfloat(1.5f0)
  f = MCP(level, level)
  k = Float64(level)^2
  x = [prevfloat(k)]
  y = zeros(Float32, 1)
  # A zero step isolates destination conversion. This checks the stored
  # value convention without relying on mixed-precision threshold arithmetic.
  @test value_and_prox!(y, f, x, 0.0) === f(y)
  @test x[1] < k < y[1]
end

end
