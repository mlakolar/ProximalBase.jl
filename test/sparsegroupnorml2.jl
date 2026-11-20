module SparseGroupNormL2Test

using Test, ProximalBase, LinearAlgebra, Random, OffsetArrays
using ..TestUtils

const PB = ProximalBase

# The proximal minimizer is the input minus its projection onto the Minkowski
# sum of the dual box and ball. The same feasible dual vector supplies an
# independent lower bound for the proximal objective.
function reference_prox(u, γ, λ1, λ2, w)
  U = BigFloat.(u)
  a = BigFloat(γ) * BigFloat(λ1)
  b = BigFloat(γ) * BigFloat(λ2) * BigFloat(w)
  p = clamp.(U, -a, a)
  r = U - p
  nr = norm(r)
  q = iszero(nr) ? r : min(one(nr), b / nr) .* r
  (r - q, p + q)
end

function reference_value(x, λ1, λ2, groups, w)
  value = BigFloat(0)
  for (k, G) in enumerate(groups)
    u = BigFloat.(x[G])
    # Stored infinite weights impose a domain constraint even at λ2 == 0.
    if isinf(λ1) || isinf(w[k]) || (isinf(λ2) && w[k] > 0)
      any(isnan, u) && return BigFloat(NaN)
      all(iszero, u) || (value += BigFloat(Inf))
      continue
    end
    iszero(λ1) || (value += BigFloat(λ1) * sum(abs, u))
    if !iszero(λ2) && !iszero(w[k])
      nu = any(isnan, u) ? BigFloat(NaN) : norm(u)
      value += BigFloat(λ2) * BigFloat(w[k]) * nu
    end
  end
  value
end

# The #900 componentwise scale accounts for rounding in the shrink factor,
# with an extra soft-stage allowance and a dimension-dependent norm allowance.
function close_prox(y, ref, u; n = length(u))
  T = eltype(y)
  scale = BigFloat(maximum(abs, u; init = zero(T)))
  δ = BigFloat(nextfloat(zero(T)))
  bound = (12 + n) * BigFloat(eps(T)) * scale + 3δ
  all(i -> abs(BigFloat(y[i]) - ref[i]) <= bound, eachindex(y, ref))
end

function close_value(v, ref; n)
  T = typeof(v)
  if !isfinite(ref)
    return isequal(v, T(ref))
  end
  abs(ref) > BigFloat(floatmax(T)) * (1 + BigFloat(eps(T))) &&
    return v == copysign(T(Inf), ref)
  bound = (12 + n) * BigFloat(eps(T)) * abs(ref) + 3 * BigFloat(nextfloat(zero(T)))
  abs(BigFloat(v) - ref) <= bound
end

function check_regular(::Type{T}, x, λ1, λ2, groups, w, γ) where {T}
  f = SparseGroupNormL2(λ1, λ2, groups, w)
  y = similar(x)
  z = copy(x)
  @test prox!(y, f, x, γ) === y
  @test prox!(z, f, z, γ) === z
  allocated = prox(f, x, γ)
  @test eltype(allocated) === T
  fused_out = similar(x)
  fused = value_and_prox!(fused_out, f, x, γ)
  in_place_fused = copy(x)
  fused_alias = value_and_prox!(in_place_fused, f, in_place_fused, γ)
  outputs = (y, z, allocated, fused_out, in_place_fused)
  free = setdiff(eachindex(Base.IndexLinear(), x), Iterators.flatten(groups))
  for out in outputs
    @test isequal(out[free], x[free])
  end
  for (k, G) in enumerate(groups)
    u = x[G]
    ref, dual = reference_prox(u, γ, λ1, λ2, w[k])
    for out in outputs
      @test close_prox(out[G], ref, u)
    end
    out = similar(u)
    @test prox_block!(out, f, u, k, γ) === out
    @test close_prox(out, ref, u)
    @test close_prox(prox_block(f, u, k, γ), ref, u)
    same = copy(u)
    @test prox_block!(same, f, same, k, γ) === same
    @test close_prox(same, ref, u)
    U = BigFloat.(u)
    Y = BigFloat.(out)
    penalty = BigFloat(λ1) * sum(abs, Y) + BigFloat(λ2) * BigFloat(w[k]) * norm(Y)
    objective = sum(abs2, Y - U) / 2 + BigFloat(γ) * penalty
    lower = dot(U, dual) - sum(abs2, dual) / 2
    @test abs(objective - lower) <= (32 + length(G)) * BigFloat(eps(T)) * max(sum(abs2, U), BigFloat(1))
  end
  @test f(x) isa T
  @test close_value(f(x), reference_value(x, λ1, λ2, groups, w); n = length(x))
  for (v, out) in ((fused, fused_out), (fused_alias, in_place_fused))
    @test v isa T
    @test close_value(v, reference_value(out, λ1, λ2, groups, w); n = length(out))
    # The fused value includes the rounding of each stored output coefficient.
    @test close_value(v, BigFloat(f(out)); n = length(out))
  end
end

@testset "ownership and construction" begin
  for T in (Float32, Float64)
    G = [[4, 2], [3, 1, 5]]
    w = T[1, 2]
    f = SparseGroupNormL2(T(0.5), T(1.5), G, w)
    @test f isa PB.AbstractFunction
    @test isproximable(f) && isconvex(f) && !issmooth(f) && !isseparable(f)
    @test SparseGroupNormL2(T(0.5), one(T), G).w == T[sqrt(T(2)), sqrt(T(3))]
    for (a, b) in ((T(-1), one(T)), (one(T), T(-1)), (T(NaN), one(T)), (one(T), T(NaN)))
      @test_throws ArgumentError SparseGroupNormL2(a, b, G)
      @test_throws ArgumentError SparseGroupNormL2(a, b, G, w)
    end
    @test_throws DimensionMismatch SparseGroupNormL2(one(T), one(T), G, T[1])
    for weights in (T[-1, 2], T[NaN, 2], Complex{T}[1, 2], OffsetArray(w, 0:1))
      @test_throws ArgumentError SparseGroupNormL2(one(T), one(T), G, weights)
    end
    for groups in ([Int[]], [[0, 1]], [[-1, 1]], [[1, 1]], [[1, 2], [2, 3]],
                   OffsetArray(G, 0:1), [OffsetArray([1, 2], 0:1)])
      @test_throws ArgumentError SparseGroupNormL2(one(T), one(T), groups)
    end
    G[1][1] = 20
    G[2] = [30]
    w[1] = 0
    @test f.groups == [[4, 2], [3, 1, 5]] && f.w == T[1, 2]
    @test f(T[3, 0, 0, 1, 0]) == T(12.5)
  end
  f = SparseGroupNormL2(1f0, 2.0, [1:2], Float32[1])
  @test typeof(f.λ1) === typeof(f.λ2) === eltype(f.w) === Float64
end

@testset "finite objectives, values, and reductions ($T)" for T in (Float32, Float64)
  setprecision(BigFloat, 512) do
    G = [[6, 1, 4], [3], [10, 8]]
    w = T[1, 0, 2]
    x = reshape(T[-4, NaN, 4, 1, Inf, 9, -0.0, -3, 17, 6, -Inf, -0.0], 3, 4)
    @testset "grouped-only, ordered groups, and matrix indices" begin
      check_regular(T, x, T(0.75), T(1.25), G, w, T(0.5))
      check_regular(T, x, zero(T), one(T), G, w, T(0.75))
      check_regular(T, x, one(T), zero(T), G, w, T(0.75))
      check_regular(T, x, one(T), one(T), G, zeros(T, 3), T(0.75))
      g = SparseGroupNormL2(zero(T), zero(T), Vector{Int}[])
      @test g(x) === zero(T) && isequal(prox(g, x, T(Inf)), x)
      empty_point = T[]
      @test g(empty_point) === zero(T) && isempty(prox(g, empty_point, one(T)))
      @test kkt_violation(g, empty_point, empty_point) === zero(T)
    end
    @testset "unrelated penalty oracles agree on common domains" begin
      u = T[3, -8, 0, 1, -2, 5]
      groups = [[6, 1], [4, 2]]
      weights = T[2, 0]
      a, b, γ = T(0.75), T(1.25), T(0.5)
      s = SparseGroupNormL2(zero(T), b, groups, weights)
      g = GroupNormL2(b, groups, weights)
      @test prox(s, u, γ) ≈ prox(g, u, γ) rtol = 8eps(T)
      @test s(u) ≈ g(u) rtol = 8eps(T)
      l = NormL1(a)
      covered = vcat(groups...)
      for s in (SparseGroupNormL2(a, zero(T), groups, weights),
                SparseGroupNormL2(a, b, groups, zeros(T, 2)))
        @test prox(s, u, γ)[covered] ≈ prox(l, u[covered], γ) rtol = 8eps(T)
        @test s(u) ≈ l(u[covered]) rtol = 8eps(T)
      end
      singleton = [[i] for i in eachindex(u)]
      weights = T[0, 1, 2, 1, 2, 1]
      s = SparseGroupNormL2(a, b, singleton, weights)
      l = NormL1(one(T), a .+ b .* weights)
      @test prox(s, u, γ) ≈ prox(l, u, γ) rtol = 8eps(T)
      @test s(u) ≈ l(u) rtol = 8eps(T)
      check_regular(T, u, a, b, singleton, weights, γ)
    end
    @testset "large ordered groups" begin
      rng = MersenneTwister(1068)
      u = randn(rng, T, 4096)
      for group in (1:length(u), randperm(rng, length(u)))
        check_regular(T, u, T(0.125), T(0.25), [group], T[2], one(T))
      end
    end
  end
end

# The endpoint oracle deliberately evaluates the hard domain before taking any
# products of the parameters. The soft comparison uses the exact finite factors.
function reference_endpoint(u::Vector{T}, γ, a, b, w) where {T}
  hard = isinf(a) || isinf(w) || (isinf(b) && w > 0)
  coupled = !iszero(b) && !iszero(w)
  penalized = !iszero(a) || coupled
  (hard || (isinf(γ) && penalized)) && return zeros(T, length(u))
  (iszero(γ) || !penalized) && return copy(u)
  coupled && any(isnan, u) && return fill(T(NaN), length(u))
  if any(!isfinite, u)
    iszero(a) && return copy(u)
    threshold = BigFloat(γ) * BigFloat(a)
    return map(u) do x
      isfinite(x) ? T(copysign(max(abs(BigFloat(x)) - threshold, BigFloat(0)), x)) : x
    end
  end
  ref, _ = reference_prox(u, γ, a, iszero(w) ? zero(T) : b, w)
  T.(ref)
end

@testset "endpoint grid and hard-domain values ($T)" for T in (Float32, Float64)
  z, o, I = zero(T), one(T), T(Inf)
  G = [[4, 1, 3]]
  δ = nextfloat(z)
  patterns = (T[0, -0.0, δ], T[-3, 8, floatmax(T)], T[NaN, 4, 0], T[I, -I, 4], T[I, 4, NaN])
  for a in (z, T(0.5), I), b in (z, T(0.75), I), w in (z, T(2), I)
    f = SparseGroupNormL2(a, b, G, T[w])
    for γ in (z, o, I), u in patterns
      x = T[0, NaN, 0, 0, -Inf, -0.0]
      x[only(G)] = u
      ref = reference_endpoint(u, γ, a, b, w)
      stored = copy(x)
      val = value_and_prox!(stored, f, stored, γ)
      y = prox(f, x, γ)
      compact = copy(u)
      @test prox_block!(compact, f, compact, 1, γ) === compact
      # The signs of fixed zeros and the signs of copied zeros are specified;
      # ordinary finite shrinkage can produce either sign of a zero.
      hard_or_identity = isinf(a) || isinf(w) || (isinf(b) && w > 0) ||
                         iszero(γ) || isinf(γ) || (iszero(a) && (iszero(b) || iszero(w)))
      for out in (y[only(G)], stored[only(G)], compact)
        for i in eachindex(out)
          if hard_or_identity || !isfinite(ref[i])
            @test isequal(out[i], ref[i])
          else
            @test abs(BigFloat(out[i]) - BigFloat(ref[i])) <=
              16 * BigFloat(eps(T)) * abs(BigFloat(ref[i])) + 3 * BigFloat(δ)
          end
        end
      end
      @test isequal(y[[2, 5, 6]], x[[2, 5, 6]])
      @test isequal(stored[[2, 5, 6]], x[[2, 5, 6]])
      @test close_value(val, reference_value(stored, a, b, G, T[w]); n = 3)
      @test close_value(f(x), reference_value(x, a, b, G, T[w]); n = 3)
    end
  end
  for groups in ([1:2, 3:4], [3:4, 1:2])
    f = SparseGroupNormL2(I, o, groups, T[1, 1])
    @test isnan(f(T[Inf, 1, NaN, 1]))
    @test f(T[0, -0.0, 0, -0.0]) === z
    @test f(T[0, 0, 0, 1]) === I
  end
end

@testset "finite factors and extreme ranges ($T)" for T in (Float32, Float64)
  setprecision(BigFloat, 512) do
    z, o = zero(T), one(T)
    F, s = floatmax(T), floatmin(T)
    δ = nextfloat(z)
    # A finite L1-threshold overflow must leave a genuine infinity infinite.
    u = T[Inf, -Inf, F, 1]
    f = SparseGroupNormL2(F, o, [1:4], T[1])
    @test prox(f, u, T(2)) == T[Inf, -Inf, 0, 0]
    @test all(isnan, prox(f, T[NaN, -Inf, F, 1], T(2)))
    # The finite group coefficient and norm both exceed the working range.
    # The soft stage is substantial, and the final point is representable.
    f = SparseGroupNormL2(F / 4, F, [1:4], T[1.25])
    u = T[F, -F, F, -F]
    ref, _ = reference_prox(u, o, F / 4, F, T(1.25))
    @test all(!iszero, T.(ref))
    @test close_prox(prox(f, u, o), ref, u)
    @test close_prox(prox_block(f, u, 1, o), ref, u)
    # The raw sum of squares is subnormal or zero, even with a nonzero result.
    for (u, a, b, γ) in ((T[4s, -8s, δ], s, s, o),
                         (T[8δ, -16δ, δ], δ, s, δ),
                         (T[5δ, -12δ, -δ], δ, T(3) * δ, o))
      weights = T[1]
      groups = [1:length(u)]
      f = SparseGroupNormL2(a, b, groups, weights)
      ref, _ = reference_prox(u, γ, a, b, o)
      @test any(!iszero, T.(ref))
      @test close_prox(prox(f, u, γ), ref, u)
      @test close_prox(prox_block(f, u, 1, γ), ref, u)
    end
    # The L1 sum itself overflows but its bounded product is representable.
    u = fill(F / 2, 4)
    G = [1:4]
    w = T[0]
    f = SparseGroupNormL2(T(0.125), o, G, w)
    expected = reference_value(u, T(0.125), o, G, w)
    @test isfinite(T(expected)) && close_value(f(u), expected; n = length(u))
    y = similar(u)
    @test close_value(value_and_prox!(y, f, u, z), expected; n = length(u))
    # A weighted radial value is representable even when its norm overflows.
    w = T[0.0625]
    f = SparseGroupNormL2(T(0.125), o, G, w)
    u .= F
    expected = reference_value(u, T(0.125), o, G, w)
    @test isfinite(T(expected)) && close_value(f(u), expected; n = length(u))
  end
end

@testset "array and compact validation before writes ($T)" for T in (Float32, Float64)
  G = [[3, 1], [2, 4]]
  x = T[3, -10, 2, 8, 12]
  sentinel = fill(T(7), 5)
  g = SparseGroupNormL2(T(Inf), one(T), G)
  for f in (SparseGroupNormL2(T(0.5), one(T), G), g)
    for γ in (-one(T), T(NaN), -T(Inf), -floatmin(T))
      for op in (prox!, value_and_prox!)
        y = copy(sentinel)
        @test_throws ArgumentError op(y, f, x, γ)
        @test isequal(y, sentinel)
        y = copy(x)
        @test_throws ArgumentError op(y, f, y, γ)
        @test isequal(y, x)
      end
      @test_throws ArgumentError prox(f, x, γ)
      u = T[3, 2]
      out = fill(T(7), 2)
      @test_throws ArgumentError prox_block!(out, f, u, 1, γ)
      @test out == fill(T(7), 2)
      @test_throws ArgumentError prox_block!(u, f, u, 1, γ)
      @test u == T[3, 2]
      @test_throws ArgumentError prox_block(f, u, 1, γ)
    end
  end
  γ = one(T)
  for op in (prox!, value_and_prox!)
    for (y, input, error) in ((copy(sentinel), x[1:4], DimensionMismatch),
                              (copy(sentinel[1:3]), x[1:3], DimensionMismatch),
                              (copy(sentinel), reshape(copy(x), 1, 5), DimensionMismatch),
                              (copy(sentinel), OffsetArray(x, 0:4), ArgumentError),
                              (OffsetArray(copy(sentinel), 0:4), x, ArgumentError),
                              (copy(sentinel), complex.(x), ArgumentError),
                              (complex.(sentinel), x, ArgumentError))
      before = copy(y)
      @test_throws error op(y, g, input, γ)
      @test isequal(y, before)
    end
  end
  @test_throws DimensionMismatch g(x[1:3])
  @test_throws ArgumentError g(OffsetArray(x, 0:4))
  @test_throws ArgumentError g(complex.(x))
  u = T[3, 2]
  out = fill(T(7), 2)
  for k in (-1, 0, 3, true)
    @test_throws BoundsError prox_block!(out, g, u, k, γ)
    @test out == fill(T(7), 2)
  end
  for (dst, src, error) in ((out, x[1:3], DimensionMismatch),
                            (fill(T(7), 3), u, DimensionMismatch),
                            (out, OffsetArray(u, 0:1), ArgumentError),
                            (OffsetArray(copy(out), 0:1), u, ArgumentError),
                            (out, complex.(u), ArgumentError),
                            (complex.(out), u, ArgumentError))
    before = copy(dst)
    @test_throws error prox_block!(dst, g, src, 1, γ)
    @test isequal(dst, before)
  end
  backing = T[7, 8, 9]
  @test_throws ArgumentError prox_block!(view(backing, 1:2), g, view(backing, 2:3), 1, γ)
  @test backing == T[7, 8, 9]
  symmetric = SymmetricSparseIterate(T[4 2; 2 9])
  dest = SymmetricSparseIterate(T[7 3; 3 5])
  dense = fill(T(7), 2, 2)
  for op in (prox!, value_and_prox!)
    @test_throws ArgumentError op(dest, g, symmetric, γ)
    @test_throws ArgumentError op(dest, g, dense, γ)
    @test_throws ArgumentError op(dense, g, symmetric, γ)
    @test Matrix(dest) == T[7 3; 3 5]
    @test dense == fill(T(7), 2, 2)
  end
  @test_throws ArgumentError prox(g, symmetric, γ)
  @test_throws ArgumentError kkt_violation(g, dense, symmetric, 1)
  @test_throws ArgumentError kkt_violation(g, dense, symmetric)
  @test_throws MethodError prox_block(g, one(T), 1, γ)
  @test_throws MethodError prox!(copy(sentinel), g, x, T[1, 1])
  @test_throws MethodError restrict(g, [1, 2])
  @test_throws MethodError value_entry(g, one(T), 1)
  @test_throws ArgumentError kkt_violation(g, x, 1)
end

@testset "sparse and block storage, pure block calculations ($T)" for T in (Float32, Float64)
  G = [[4, 1], [5, 2], [3, 6]]
  a, b, γ = T(0.5), T(0.75), T(0.5)
  w = T[1, 2, 0]
  f = SparseGroupNormL2(a, b, G, w)
  x = BlockIterate(T, 6, f)
  @test nactive(x) == 0 && collect(blockindices(x, 2)) == G[2]
  @test BlockIterate(6, f) isa BlockIterate{Float64}
  @test_throws ArgumentError BlockIterate(7, f)
  @test_throws ArgumentError BlockIterate(zeros(T, 6), SparseGroupNormL2(a, b, [[1, 3]]))
  x[1] = T(5)
  x[4] = T(-3)
  activate!(x, 2)
  dense = Vector(x)
  @test Vector(BlockIterate(dense, f)) == dense
  order = collect(activecoordinates(x))
  for k in eachindex(G)
    u = Vector(blockview(x, k))
    out = similar(u)
    ref, _ = reference_prox(u, γ, a, b, w[k])
    @test prox_block!(out, f, blockview(x, k), k, γ) === out
    @test close_prox(out, ref, u)
    @test Vector(x) == dense && collect(activecoordinates(x)) == order
    @test isactive(x, 2) && !isactive(x, 3)
  end
  # A nonzero trial for an inactive group remains a pure calculation.
  out = prox_block(f, T[8, 7], 3, γ)
  @test all(!iszero, out) && !isactive(x, 3)
  @test Vector(x) == dense && collect(activecoordinates(x)) == order
  s = SparseIterate(T, length(x))
  s[1], s[4] = dense[1], dense[4]
  activate!(s, 2)
  for input in (s, x)
    @test close_value(f(input), reference_value(dense, a, b, G, w); n = length(x))
    for (y, src) in ((similar(input), input), (zeros(T, length(input)), input), (copy(input), nothing))
      val = value_and_prox!(y, f, isnothing(src) ? y : src, γ)
      for (k, entries) in enumerate(G)
        ref, _ = reference_prox(dense[entries], γ, a, b, w[k])
        @test close_prox(Vector(y)[entries], ref, dense[entries])
      end
      @test close_value(val, reference_value(Vector(y), a, b, G, w); n = length(x))
    end
  end
  @test isactive(s, 2) && !isactive(s, 3)
end

# Project the target onto the translated coordinate intervals of the active
# subdifferential. A nonzero coefficient fixes the corresponding interval
# to a singleton. At zero the distance is to the box before the radial margin.
function reference_score(v, x, λ1, λ2, w)
  target = -BigFloat.(v)
  X = BigFloat.(x)
  a = BigFloat(λ1)
  b = BigFloat(λ2) * BigFloat(w)
  if all(iszero, X)
    return norm(target - clamp.(target, -a, a)) - b
  end
  center = (b / norm(X)) .* X
  nearest = map(X, target, center) do xi, ti, ci
    iszero(xi) ? clamp(ti, ci - a, ci + a) : ci + copysign(a, xi)
  end
  norm(nearest - target)
end

function close_score(v, ref, g, λ1, λ2, w)
  T = typeof(v)
  magnitude = norm(BigFloat.(g)) + BigFloat(λ1) * sqrt(BigFloat(length(g))) +
              BigFloat(λ2) * BigFloat(w)
  bound = (length(g) + 12) * BigFloat(eps(T)) * magnitude +
          4 * BigFloat(nextfloat(zero(T)))
  abs(BigFloat(v) - ref) <= bound
end

@testset "signed and point-aware block KKT ($T)" for T in (Float32, Float64)
  a, b = T(0.5), one(T)
  G = [[3, 1], [6, 4, 2]]
  w = T[1, 5]
  f = SparseGroupNormL2(a, b, G, w)
  z = T[-0.0, 0, 0, 0, 3, 0, 4]
  for boundary in (prevfloat(T(1.5)), T(1.5), nextfloat(T(1.5)))
    v = T[0, 0, -boundary, 0, -4, 0, -6]
    ref = reference_score(v[G[1]], z[G[1]], a, b, w[1])
    @test sign(kkt_violation(f, v, z, 1)) == sign(ref)
    @test kkt_violation(f, v, z, 3) == T(ref)
    @test kkt_violation(f, v, z, 5) == T(4)
    @test kkt_violation(f, v, z, 7) == T(6)
    @test kkt_violation(f, v, z) == T(6)
  end
  x = T[0, 0, 0, 4, 7, 3, 6]
  v = T[0, 3, 0, -6, 0, -3, 0]
  ref = reference_score(v[G[2]], x[G[2]], a, b, w[2])
  @test ref > 2
  for i in G[2]
    @test close_score(kkt_violation(f, v, x, i), ref, v[G[2]], a, b, w[2])
  end
  @test close_score(kkt_violation(f, v, x), ref, v[G[2]], a, b, w[2])
  # Make every coordinate, including uncovered entries, satisfy stationarity.
  v = T[0, 0.5, 0, -4.5, 0, -3.5, 0]
  @test kkt_violation(f, v, x) == 0
  v[5] = T(0.25)
  @test kkt_violation(f, v, x) == T(0.25)
  # Extreme point norms and small residuals are independent of group activity.
  for scale in (nextfloat(zero(T)), floatmin(T), floatmax(T) / 4)
    x[G[2]] = T[3scale, 4scale, -0.0]
    v[G[2]] = T[-3, -6, 3]
    ref = reference_score(v[G[2]], x[G[2]], a, b, w[2])
    @test close_score(kkt_violation(f, v, x, 2), ref, v[G[2]], a, b, w[2])
  end
  dense = x[1:6]
  dense[5] = 0
  Gfull = [G..., [5]]
  hb = SparseGroupNormL2(a, b, Gfull, T[w..., 1])
  point = BlockIterate(dense, hb)
  grad = v[1:6]
  ref = reference_score(grad[G[2]], dense[G[2]], a, b, w[2])
  @test close_score(kkt_violation(hb, grad, point, 2), ref, grad[G[2]], a, b, w[2])
  @test close_score(kkt_violation(hb, grad, point), ref, grad[G[2]], a, b, w[2])
  reordered = BlockIterate(dense, [reverse(G[1]), G[2], [5]])
  @test_throws DimensionMismatch kkt_violation(hb, grad, reordered, 1)
  @test_throws DimensionMismatch kkt_violation(hb, grad, reordered)
  @test_throws DimensionMismatch kkt_violation(f, grad, point, 2)
  @test_throws DimensionMismatch kkt_violation(f, grad, point)
  for (g, p, error) in ((grad[1:5], dense, DimensionMismatch),
                         (OffsetArray(grad, 0:5), dense, ArgumentError),
                         (grad, OffsetArray(dense, 0:5), ArgumentError),
                         (complex.(grad), dense, ArgumentError),
                         (grad, complex.(dense), ArgumentError))
    @test_throws error kkt_violation(f, g, p, 1)
    @test_throws error kkt_violation(f, g, p)
  end
  @test_throws BoundsError kkt_violation(f, grad, dense, 0)
  @test_throws BoundsError kkt_violation(hb, grad, point, 0)
end

@testset "nonfinite and hard-domain KKT ($T)" for T in (Float32, Float64)
  a, b, I = T(0.5), one(T), T(Inf)
  G = [1:2, 3:4]
  for (al, bl, w) in ((I, b, T[0, 1]), (a, I, T[1, 1]), (a, zero(T), T[I, I]))
    f = SparseGroupNormL2(al, bl, G, w)
    x = zeros(T, 4)
    v = T[2, 1, 3, 4]
    @test kkt_violation(f, v, x, 1) == -I
    @test kkt_violation(f, v, x) == 0
    x[1] = nextfloat(zero(T))
    @test kkt_violation(f, v, x, 2) == I
    @test kkt_violation(f, v, x) == I
    x[4] = T(NaN)
    @test isnan(kkt_violation(f, v, x))
    x[4] = zero(T)
    v[2] = T(NaN)
    @test isnan(kkt_violation(f, v, x, 1))
    v[2] = I
    x[1] = zero(T)
    @test kkt_violation(f, v, x, 1) == I
    @test kkt_violation(f, v, x) == I
  end
  f = SparseGroupNormL2(a, b, G)
  x, v = zeros(T, 4), T[Inf, 1, 3, NaN]
  @test kkt_violation(f, v, x, 1) == I
  @test isnan(kkt_violation(f, v, x, 3))
  @test isnan(kkt_violation(f, v, x))
end

@testset "stored value and Scaled route ($T)" for T in (Float32, Float64)
  W = T === Float32 ? Float64 : Float32
  G = [[3, 1]]
  w = T[2]
  a, b, γ, α = T(0.5), T(0.75), T(0.5), T(2)
  f = SparseGroupNormL2(a, b, G, w)
  x = T[-1.231, 5, 4.365]
  narrowed = zeros(W, length(x))
  v = value_and_prox!(narrowed, f, x, γ)
  @test close_value(v, reference_value(narrowed, a, b, G, w); n = length(x))
  @test close_value(v, BigFloat(f(narrowed)); n = length(x))
  s = Scaled(f, α)
  ref, _ = reference_prox(x[only(G)], γ, α * a, α * b, only(w))
  y = similar(x)
  @test prox!(y, s, x, γ) === y
  @test close_prox(y[only(G)], ref, x[only(G)])
  @test y[2] == x[2]
  @test close_value(s(x), BigFloat(α) * reference_value(x, a, b, G, w); n = length(x))
  v = value_and_prox!(y, s, x, γ)
  @test close_value(v, BigFloat(α) * reference_value(y, a, b, G, w); n = length(x))
  @test_throws MethodError prox_block(s, x[only(G)], 1, γ)
  @test_throws MethodError prox_block!(x[only(G)], s, x[only(G)], 1, γ)
  @test_throws MethodError kkt_violation(s, x, x, 1)
  @test_throws MethodError kkt_violation(s, x, x)

  # The generic wrapper rounds its product with the step before leaf dispatch.
  # A finite product outside its range can therefore select the γ = Inf exit.
  s = Scaled(SparseGroupNormL2(T(0.125), zero(T), [1:1]), floatmax(T))
  @test prox(s, T[floatmax(T) / 2, 12], T(2)) == T[0, 12]
end

@testset "narrow storage underflow precedes fused value evaluation" begin
  f = SparseGroupNormL2(0x1p100, 0.0, [1:1])
  x = [0x1p-151, 7.0]
  y = fill(-1.0f0, 2)
  @test value_and_prox!(y, f, x, 0.0) == 0
  @test y == Float32[0, 7]
  @test f(x) == 0x1p-51
end

@testset "warmed ordinary allocation checks ($T)" for T in (Float32, Float64)
  x = T.(repeat([3, -2, 1, -4], 16))
  y = similar(x)
  u = x[1:4]
  out = similar(u)
  γ = T(0.5)
  for G in ([4k-3:4k for k in 1:16], [[4k, 4k-2, 4k-1, 4k-3] for k in 1:16])
    f = SparseGroupNormL2(T(0.5), T(0.75), G, ones(T, length(G)))
    @test allocs(f, x) == 0
    @test allocs(prox!, y, f, x, γ) == 0
    @test allocs(value_and_prox!, y, f, x, γ) == 0
    @test allocs(prox_block!, out, f, u, 1, γ) == 0
    @test allocs(kkt_violation, f, x, y, 1) == 0
    @test allocs(kkt_violation, f, x, y) == 0
  end
end

@testset "method ambiguities" begin
  @test isempty(Test.detect_ambiguities(PB; recursive = true))
end

end
