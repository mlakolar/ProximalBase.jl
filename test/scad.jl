module SCADTest

using Test, ProximalBase, OffsetArrays, Random

function scad_penalty(λ, a, v)
  t = abs(v)
  iszero(λ) && return zero(t)
  isnan(t) && return t
  iszero(t) && return zero(t)
  isinf(λ) && return λ
  (t <= λ || isinf(a)) && return λ * t
  t >= a * λ && return (a + 1) * λ^2 / 2
  # Integrate the linearly decreasing slope from λ to t by its trapezoid.
  λ^2 + (t - λ) * (λ + (a * λ - t) / (a - 1)) / 2
end

function scad_objective(λ, a, t, γ, u)
  L, A, V, G, U = BigFloat.((λ, a, t, γ, u))
  scad_penalty(L, A, U) + (U - V)^2 / (2G)
end

# Enumerate endpoints and interior vertices of the objective polynomials.
# This uses the individual coefficients, independently of the step regimes
# and candidate comparisons in the proximal implementation.
function scad_reference(λ, a, v, γ)
  L, A, V, G = BigFloat.((λ, a, v, γ))
  t = abs(V)
  c2 = inv(2G)
  c1 = -t / G
  pieces = ((zero(L), L, c2, c1 + L),
            (L, A * L, c2 - inv(2(A - 1)), c1 + A * L / (A - 1)),
            (A * L, max(t, A * L), c2, c1))
  best = zero(t)
  cost(u) = scad_penalty(L, A, u) + (u - t)^2 / (2G)
  hbest = cost(best)
  for (lo, hi, q, b) in pieces
    vertex = q > 0 ? clamp(-b / (2q), lo, hi) : lo
    for u in (lo, vertex, hi)
      h = cost(u)
      if h < hbest || (h == hbest && u < best)
        best, hbest = u, h
      end
    end
  end
  iszero(best) ? best : copysign(best, V), hbest
end

# Search the full feasible magnitude range, then refine the best grid cell
# without using the stationary-point formulas.
function scad_search(λ, a, t, γ)
  objective(u) = scad_objective(λ, a, t, γ, u)
  grid = range(zero(t), abs(t); length = 1025)
  i = argmin(map(objective, grid))
  lo, hi = grid[max(1, i - 1)], grid[min(length(grid), i + 1)]
  best = objective(grid[i])
  for _ in 1:80
    l, r = (2lo + hi) / 3, (lo + 2hi) / 3
    hl, hr = objective(l), objective(r)
    best = min(best, hl, hr)
    if hl <= hr
      hi = r
    else
      lo = l
    end
  end
  best
end

function check_case(f, v::T, γ; check_point = false) where {T}
  want, best = scad_reference(f.λ, f.a, v, γ)
  got = prox_block(f, v, 1, γ)
  @test got isa T
  @test isfinite(got) && abs(got) <= abs(v)
  if isfinite(got)
    cost = scad_objective(f.λ, f.a, v, γ, got)
    # Scale the objective check by the objective and the cost of moving by a
    # subnormal unit. At a switch, substantially different points can have
    # essentially the same objective in the input precision.
    rounding_cost = scad_objective(f.λ, f.a, v, γ, want + BigFloat(nextfloat(zero(T))))
    @test cost <= best * (1 + 32eps(T)) + abs(rounding_cost - best)
  end
  if check_point
    @test abs(BigFloat(got) - want) <= 16eps(T) * abs(BigFloat(v)) + nextfloat(zero(T))
  end
  want_value = T(scad_penalty(BigFloat(f.λ), BigFloat(f.a), BigFloat(v)))
  val = value_entry(f, v, 1)
  @test val isa T
  @test isapprox(val, want_value; rtol = 16eps(T), atol = 2nextfloat(zero(T)))
  nothing
end

@testset "SCAD constructor and traits" begin
  @test SCAD(1).a === 3.7
  @test SCAD(0.5f0) isa SCAD{Float64}
  @test SCAD(1f0, 3f0) isa SCAD{Float32}
  @test SCAD(1f0, 3.0) isa SCAD{Float64}
  @test SCAD(0, Inf).λ == 0
  @test SCAD(Inf, 3).λ == Inf
  @test SCAD(1.0, nextfloat(2.0)).a > 2
  for λ in (-1.0, -Inf, NaN)
    @test_throws ArgumentError SCAD(λ)
    @test_throws ArgumentError SCAD{Float64}(λ, 3.0)
  end
  for a in (2.0, 0.0, -Inf, NaN)
    @test_throws ArgumentError SCAD(0.0, a)
    @test_throws ArgumentError SCAD{Float64}(0.0, a)
  end
  f = SCAD(1.0)
  @test isseparable(typeof(f)) && isproximable(typeof(f)) && iszeropreserving(f)
  @test !isconvex(typeof(f)) && !issmooth(typeof(f))
  @test restrict(f, [2, 1]) === f
  @test ProximalBase.entry_parameters(f) === ()
  @test ProximalBase.check_symmetric_parameters(f, SymmetricSparseIterate(2))
end

@testset "SCAD default calls" begin
  for (f, T) in ((SCAD(0.5f0), Float32), (SCAD(0.5f0, 3.7f0), Float32), (SCAD(0.5), Float64))
    x = T[-4, -1.3, -0.6, -0.0, 0.2, 1.1, 2.75, -Inf, Inf, NaN]
    step = one(T)
    expected = T[isfinite(v) ? scad_reference(f.λ, f.a, v, step)[1] : v for v in x]
    for y in (prox(f, x), prox(f, x, step), prox!(similar(x), f, x), prox!(similar(x), f, x, step))
      @test y isa Vector{T}
      @test y[1:7] ≈ expected[1:7]
      @test isequal(y[8:end], expected[8:end])
      @test y[4] === zero(T)
    end
    y = copy(x)
    @test isequal(value_and_prox!(y, f, y), f(y))
    @test y[1:7] ≈ expected[1:7]
    @test isequal(y[8:end], expected[8:end])
  end
end

@testset "SCAD scalar correctness, $T" for T in (Float32, Float64)
  z, u = zero(T), one(T)
  @testset "independent objectives at the piece transitions" begin
    setprecision(BigFloat, 256) do
      for a in (nextfloat(T(2)), T(3), T(3.7)), λ in (T(0.6), u)
        f = SCAD(λ, a)
        critical = a - u
        for γ in (T(0.3), u, prevfloat(critical), critical, nextfloat(critical), T(10))
          for t in (γ * λ, λ, (u + γ) * λ, a * λ), mag in (prevfloat(t), t, nextfloat(t)), s in (-u, u)
            check_case(f, s * mag, γ)
          end
        end
      end
      # Representative finite cases from the archive, including its ordinary
      # thresholding rule at γ = 1 and its independent global tests at γ ≥ a-1.
      f = SCAD(T(0.6), T(3.7))
      for v in T[-4, -2, -1.1, -0.7, 0, 0.3, 0.9, 1.3, 2.1, 3, 5], γ in T[0.3, 1, 2.6, 2.7, 3, 10]
        check_case(f, v, γ; check_point = γ < T(2))
      end
      rng = MersenneTwister(953)
      for _ in 1:200
        λ, a, γ, v = T(0.1) + rand(rng, T), nextfloat(T(2)) + T(6) * rand(rng, T),
                     T(0.05) + T(12) * rand(rng, T), T(5) * randn(rng, T)
        check_case(SCAD(λ, a), v, γ)
      end
    end
  end

  @testset "global switches and exact ties" begin
    f = SCAD(u, T(3))
    # A concave middle piece has an interior stationary maximum for 3 < t < 4.
    # The two local minima switch at t=3.5 when γ=3.
    for (γ, t, expected) in ((2, 3, 1), (3, 3.25, 0.25), (3, 3.5, 0.5),
                             (3, 3.75, 3.75), (4, 4, 0), (9, 6, 0),
                             (5, 4.25, 0), (5, 4.75, 4.75))
      for s in (-u, u)
        @test prox_block(f, s * T(t), 1, T(γ)) === (iszero(expected) ? z : s * T(expected))
      end
      optimum = scad_search(BigFloat(1), BigFloat(3), BigFloat(t), BigFloat(γ))
      _, best = scad_reference(u, T(3), T(t), T(γ))
      cost = scad_objective(u, T(3), T(t), T(γ), prox_block(f, T(t), 1, T(γ)))
      @test cost <= optimum + big"1e-20"
      @test isapprox(best, optimum; atol = big"1e-20", rtol = 0)
    end
    for (γ, t) in ((T(2), T(3)), (T(3), T(3.5)), (T(4), T(4)), (T(5), sqrt(T(20)))), v in (prevfloat(t), nextfloat(t))
      check_case(f, v, γ)
    end
    # Both right and left have equal costs although subtracting the two
    # rounded residuals would select the right minimum in these cases.
    a, γ, λ, t = if T === Float32
      T.((0x1.3b646cp+1, 0x1.449b94p+1, 0x1.e58dcp+0, 0x1.6c2a5p+2))
    else
      (0x1.a9cf4bea48170p+1, 0x1.5630b415b7e90p+1, 0x1.1e06f36dba240p+0, 0x1.f48c2a0005bf0p+1)
    end
    A, G, L, V = BigFloat.((a, γ, λ, t))
    @test 2V == (A + G + 1) * L
    @test prox_block(SCAD(λ, a), t, 1, γ) === T(V - G * L)

    # An exact zero-vs-identity tie where rounding a+1 first moves the
    # product downward. The integer construction gives t² = γ(a+1).
    k, g, ea, eg = T === Float32 ? (4097, 1269, 22, 16) : (94906267, 40000001, 51, 45)
    a = ldexp(T(k^2 - 2^ea), -ea)
    γ = ldexp(T(g^2), -eg)
    t = ldexp(T(k * g), -(ea + eg) ÷ 2)
    @test BigFloat(t)^2 == BigFloat(γ) * (BigFloat(a) + 1)
    @test prox_block(SCAD(u, a), t, 1, γ) === z

    # Exact tie with a finite step and magnitude, but an overflowing square.
    e = T === Float32 ? 62 : 510
    γ, t = ldexp(T(6), 2e), ldexp(T(6), e)
    @test scad_objective(u, T(5), t, γ, z) == scad_objective(u, T(5), t, γ, t)
    @test prox_block(SCAD(u, T(5)), t, 1, γ) === z
  end

  @testset "finite range" begin
    setprecision(BigFloat, 4096) do
      for e in (T === Float32 ? (-72, 62, 90) : (-535, 510, 700))
        λ = ldexp(u, e)
        f = SCAD(λ, T(3))
        for v in (λ / 2, T(1.5) * λ, T(2.5) * λ, T(3.25) * λ, T(4.75) * λ),
            γ in (T(0.5), T(2), T(3), T(5))
          check_case(f, v, γ)
        end
      end
      for e in (T === Float32 ? (-75, 70) : (-540, 600))
        λ = ldexp(u, e)
        f = SCAD(λ, T(3))
        @test prox_block(f, T(3.5) * λ, 1, T(4)) === z
        @test prox_block(f, T(4.5) * λ, 1, T(4)) === T(4.5) * λ
      end
      λ = nextfloat(z)
      check_case(SCAD(λ, T(3)), T(3) * λ, T(2))
      # The upper join is at 12.5 subnormal units. Rounding its residual
      # before the affine comparison would treat 13 units as a tie.
      @test prox_block(SCAD(4λ, T(3.125)), 13λ, 1, T(2.125)) === 13λ
      check_case(SCAD(T(2) * floatmin(T), T(3)), λ, T(0.5))
      check_case(SCAD(u, T(3)), floatmax(T), floatmax(T))
      check_case(SCAD(floatmax(T), T(3)), T(0.5), T(0.5))

      # The convex middle minimizer is representable even when a*λ is not.
      λ, a, t, γ = if T === Float32
        (2.8573354f28, 7.9127734f11, 2.8573356f28, 9.946286f-21)
      else
        λ64 = ldexp(1.0, 970)
        (λ64, 0x1p+100, nextfloat(λ64), 0x1p-100)
      end
      want, _ = scad_reference(λ, a, t, γ)
      @test T(want) === t
      @test prox_block(SCAD(λ, a), t, 1, γ) === t
    end
  end

  @testset "levels, shapes, steps and nonfinite data" begin
    x = T[NaN, -Inf, Inf, -6, -0.5, -0.0, 0, 2.5]
    for λ in (z, -z, u, T(Inf)), a in (T(3), T(Inf)), γ in (z, -z, u, T(Inf))
      f = SCAD(λ, a)
      expected = if iszero(λ) || (isfinite(λ) && iszero(γ))
        x
      elseif isinf(λ) || isinf(γ)
        zero(x)
      elseif isinf(a)
        T[NaN, -Inf, Inf, -5, 0, 0, 0, 1.5]
      else
        T[NaN, -Inf, Inf, -6, 0, 0, 0, 2]
      end
      @test iszero(f(T[])) && f(T[]) isa T
      @test isequal(f(x), iszero(λ) ? z : T(NaN))
      @test isequal(map(v -> prox_block(f, v, 1, γ), x), expected)
      for step in (γ, fill(γ, length(x)))
        @test isempty(prox(f, T[], step isa Real ? step : T[]))
        @test isequal(prox(f, x, step), expected)
        for inplace in (false, true)
          y = inplace ? copy(x) : fill(T(7), length(x))
          @test isequal(value_and_prox!(y, f, inplace ? y : x, step), f(expected))
          @test isequal(y, expected)
        end
      end
      for (i, v) in pairs(x)
        want = T(scad_penalty(BigFloat(λ), BigFloat(a), BigFloat(v)))
        @test isequal(value_entry(f, v, i), want)
      end
    end
    @test value_entry(SCAD(floatmax(T), T(3)), T(Inf), 1) === T(Inf)
  end
end

materialize(x::AbstractArray) = Array(x)
materialize(x::SymmetricSparseIterate) = Matrix(x)
snapshot(x::AbstractArray) = copy(x)
snapshot(x::OffsetArray) = snapshot(parent(x))
snapshot(x::Union{SparseIterate, SymmetricSparseIterate}) =
  (size(x), nactive(x), collect(activecoordinates(x)), collect(activevalues(x)), copy(x.ind2nzval))
snapshot(x::BlockIterate) =
  (copy(x.storage), deepcopy(x.groups), copy(x.group_of), copy(x.active), copy(x.position), x.nactive, x.layout_cache)

function check_rejected(oracle, y, f, x, step, E = ArgumentError)
  before = snapshot(y)
  @test_throws E oracle(y, f, x, step)
  @test isequal(snapshot(y), before)
  nothing
end

struct ReadSteps{T} <: AbstractVector{T}
  data::Vector{T}
  reads::Vector{Int}
end
Base.size(a::ReadSteps) = size(a.data)
Base.IndexStyle(::Type{<:ReadSteps}) = IndexLinear()
Base.getindex(a::ReadSteps, i::Int) = (a.reads[i] += 1; a.data[i])

@testset "SCAD array routes, $T" for T in (Float32, Float64)
  z, u = zero(T), one(T)
  f = SCAD(u, T(3))
  x = T[-3.75, 0.5, 3.25, 2.5, -4.75, 1]
  steps = T[3, 1, 3, 1, 5, 0]
  expected = T[iszero(steps[i]) ? x[i] : scad_reference(f.λ, f.a, x[i], steps[i])[1] for i in eachindex(x)]
  expected_scalar = T[scad_reference(f.λ, f.a, v, u)[1] for v in x]
  original = copy(x)
  xpad, ypad = fill(T(7), 12), fill(T(8), 12)
  xview, yview = view(xpad, 2:2:12), view(ypad, 2:2:12)
  xview .= x
  for (input, output) in ((x, similar(x)), (xview, yview), (reshape(x, 2, 3), zeros(T, 2, 3)),
                          (reshape(xview, 2, 3), reshape(yview, 2, 3)))
    @test vec(prox(f, input, reshape(steps, 3, 2))) ≈ expected
    @test value_and_prox!(output, f, input, view(steps, :)) === f(output)
    @test vec(output) ≈ expected
    @test value_and_prox!(input, f, input, u) === f(input)
    @test vec(input) ≈ expected_scalar
    input .= reshape(original, size(input))
  end
  @test all(==(T(8)), view(ypad, 1:2:11))
  y = zeros(Float32, 2)
  @test value_and_prox!(y, SCAD(0.6, 3.0), [0.9, 2.0], 0.25) === SCAD(0.6, 3.0)(y)
  repeated = view(T[7, 7], [1, 1, 2])
  @test value_and_prox!(repeated, f, T[2.5, 0.5, 4.75], u) === f(repeated)
  @test parent(repeated) == T[0, 4.75]

  @testset "per-entry steps near the joins" begin
    model = SCAD(T(0.6), T(3.7))
    transition = model.a - 1
    points, step_buffer = T[], T[]
    for step in (prevfloat(transition), transition, nextfloat(transition))
      for join in ((1 + step) * model.λ, model.a * model.λ)
        for t in (prevfloat(join), join, nextfloat(join))
          push!(points, -t, t)
          push!(step_buffer, step, step)
        end
      end
    end
    out = prox(model, points, step_buffer)
    for i in eachindex(out)
      v, step = points[i], step_buffer[i]
      _, best = scad_reference(model.λ, model.a, v, step)
      @test abs(out[i]) <= abs(v)
      @test scad_objective(model.λ, model.a, v, step, out[i]) <= best * (1 + 32eps(T))
    end
    @test value_and_prox!(out, model, points, step_buffer) === model(out)
  end

  @testset "iterate storage and aliasing" begin
    sparse = SparseIterate(T, 8)
    sparse[2], sparse[4], sparse[6], sparse[8] = T(-3.75), T(0.5), T(2.5), T(4)
    sparse[8] = -z
    groups = [[3, 7], [2, 4], [6, 8], [1, 5]]
    blocks = BlockIterate(Array(sparse), groups)
    activate!(blocks, 1)
    symmetric = SymmetricSparseIterate(T, 4)
    symmetric[2, 1], symmetric[3, 1], symmetric[4, 2], symmetric[4, 4] =
      T(-3.75), T(0.5), T(2.5), T(4)
    symmetric[4, 4] = -z
    offdiag_value = value_entry(f, T(-3.75), 1) + value_entry(f, T(0.5), 1) + value_entry(f, T(2.5), 1)
    @test f(symmetric) ≈ 2offdiag_value

    for point in (sparse, blocks, symmetric)
      dense = materialize(point)
      @test f(point) ≈ f(dense)
      @test ProximalBase.check_symmetric_parameters(f, point)
      step_routes = point isa SymmetricSparseIterate ? (T(3),) : (T(3), fill(T(3), 8))
      for step in step_routes
        γ = step isa Real ? step : first(step)
        want = map(v -> T(scad_reference(f.λ, f.a, v, γ)[1]), dense)
        y = prox(f, point, step)
        @test y isa typeof(point)
        @test materialize(y) ≈ want
        input = copy(point)
        membership = collect(activecoordinates(input))
        @test value_and_prox!(input, f, input, step) === f(input)
        @test materialize(input) ≈ want
        @test collect(activecoordinates(input)) == membership
        out = copy(point)
        if out isa SymmetricSparseIterate
          out[1, 1] = T(9)
          extra = tril_index(4, 1, 1)
        else
          out[1] = T(9)
          extra = out isa BlockIterate ? 4 : 1
        end
        @test value_and_prox!(out, f, point, step) === f(out)
        @test materialize(out) ≈ want
        @test collect(activecoordinates(out)) == [membership; extra]
      end
    end
    @test prox!(zeros(T, 4, 4), f, symmetric, T(3)) ≈ prox(f, Matrix(symmetric), T(3))
    @test prox(f, matrixview(symmetric), fill(T(3), 16)) ≈ prox(f, Matrix(symmetric), T(3))
    @test Matrix(prox!(similar(symmetric), f, Matrix(symmetric), T(3))) ≈ prox(f, Matrix(symmetric), T(3))

    # The preflight reads each step once. Only stored indices are then read
    # again, including the stored zero and excluding the inactive positions.
    large = SparseIterate(T, 4096)
    large[3], large[4020] = T(4), T(2)
    large[4020] = z
    traced = ReadSteps(fill(T(3), length(large)), zeros(Int, length(large)))
    prox!(copy(large), f, large, traced)
    @test findall(==(2), traced.reads) == [3, 4020]
    @test all(n -> n in (1, 2), traced.reads)
  end

  @testset "pre-write refusals" begin
    sparse = SparseIterate(T, 6)
    sparse[1], sparse[2] = T(4), T(3)
    sparse[2] = z
    block = BlockIterate(Array(sparse), [[1, 4], [2, 5], [3, 6]])
    activate!(block, 2)
    symmetric = SymmetricSparseIterate(T[3 0; 0 0])
    large_sparse = SparseIterate(T, 1024)
    large_sparse[1] = T(5)
    large_steps = fill(u, length(large_sparse))
    large_steps[997] = T(NaN)
    for oracle in (prox!, value_and_prox!)
      check_rejected(oracle, copy(large_sparse), f, large_sparse, large_steps)
    end
    for λ in (z, u, T(Inf)), oracle in (prox!, value_and_prox!)
      fcheck = SCAD(λ, T(3))
      for point in (x, sparse, block, symmetric), bad in (-u, T(NaN))
        out = copy(point)
        if out isa SymmetricSparseIterate
          out[2, 2] = T(7)
        else
          out[end] = T(7)
        end
        check_rejected(oracle, out, fcheck, point, bad)
        if !(point isa SymmetricSparseIterate)
          # Position six is inactive in both compressed inputs.
          check_rejected(oracle, out, fcheck, point, T[1, 1, 1, 1, 1, bad])
        end
      end
      for out in (fill(T(7), 6), copy(sparse), copy(block))
        check_rejected(oracle, out, fcheck, x, T[1, 1], DimensionMismatch)
        check_rejected(oracle, out, fcheck, x[1:3], u, DimensionMismatch)
        check_rejected(oracle, out, fcheck, x, OffsetArray(ones(T, 6), 0:5))
        check_rejected(oracle, out, fcheck, OffsetArray(copy(x), 0:5), u)
        check_rejected(oracle, out, fcheck, complex.(x), u)
        check_rejected(oracle, out, fcheck, x, complex.(ones(T, 6)))
      end
      check_rejected(oracle, OffsetArray(fill(T(7), 6), 0:5), fcheck, x, u)
      check_rejected(oracle, fill(complex(u), 6), fcheck, x, u)
      check_rejected(oracle, copy(symmetric), fcheck, symmetric, ones(T, 2, 2))
    end
    for bad in (-u, T(NaN))
      @test_throws ArgumentError prox(f, x, bad)
      @test_throws ArgumentError prox(f, x, T[1, 1, 1, 1, 1, bad])
    end
    @test_throws ArgumentError prox(f, symmetric, ones(T, 2, 2))
  end

  @testset "Scaled compositions" begin
    @test_throws ArgumentError Scaled(f, z)
    f2 = Scaled(f, T(2))
    nested = Scaled{typeof(f2), T}(f2, T(1.5))
    for (s, factor) in ((f2, T(2)), (nested, T(3)))
      for step in (u, steps)
        expected = map(eachindex(x)) do i
          γ = factor * (step isa Real ? step : step[i])
          iszero(γ) ? x[i] : T(scad_reference(f.λ, f.a, x[i], γ)[1])
        end
        y = similar(x)
        @test value_and_prox!(y, s, x, step) === s(y)
        @test y ≈ expected
      end
    end
    s = Scaled(f, T(Inf))
    for step in (u, ones(T, length(x)))
      y = copy(x)
      @test isnan(value_and_prox!(y, s, y, step))
      @test isnan(s(y))
      @test all(v -> v === z, y)
    end
  end
end

end
