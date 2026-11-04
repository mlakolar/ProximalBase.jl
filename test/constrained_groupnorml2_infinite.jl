module ConstrainedGroupNormL2InfiniteTest

# #469, Stage 3B (E1, E2, E3, E4, P1, M2, O2): GroupNormL2 constrained on an ℓ2 ball or sphere at NaN and
# infinite arguments. The expected points are stated by hand, or come from an independent BigFloat reference that
# shares no code with the package. The reference works on blocks: a penalized group G has margin a_G = ‖x_G‖ - γλw_G,
# a free coordinate or zero-penalty group is a block with margin ‖x_G‖, and norms are Frobenius norms of the full
# matrix, so a mirrored off-diagonal pair is one coordinate of multiplicity 2 (M2). Where two finite sequences with
# the same extended-real argument have different limits, the package returns NaN over the coupled block.

using Test
using ProximalBase
using LinearAlgebra
using Random

const BF = BigFloat

# --- problems and references in BigFloat ---------------------------------------------------------------

# The coordinates of an output: Gs[k] lists the coordinates of group k in the order they are given (the order of
# the tie rule), W[k] is its weight (Inf for a hard-fixed group), g[i] the group of coordinate i (0 when i is in no
# group), mult[i] its multiplicity in the Frobenius norm (2 for a mirrored pair), and coords[i] the (row, col) of a
# lower-triangle coordinate of a symmetric matrix (its column is 1 for a vector).
struct Problem
  Gs::Vector{Vector{Int}}
  W::Vector{Float64}
  g::Vector{Int}
  mult::Vector{Int}
  coords::Vector{Tuple{Int, Int}}
end

function dense_problem(n, groups, w)
  g = zeros(Int, n)
  for (k, G) in enumerate(groups), i in G
    g[i] = k
  end
  Problem([collect(Int, G) for G in groups], Float64.(w), g, ones(Int, n), [(i, 1) for i in 1:n])
end

# groups of linear indices of a p × p matrix, symmetric as sets: a group holds both entries of each pair
function sym_problem(p, groups, w)
  coords = [(r, c) for c in 1:p for r in c:p]
  pos = Dict(rc => k for (k, rc) in enumerate(coords))
  lowerof(i) = (r = mod1(i, p); c = cld(i, p); pos[r >= c ? (r, c) : (c, r)])
  g = zeros(Int, length(coords))
  Gs = Vector{Int}[]
  for (k, G) in enumerate(groups)
    members = Int[]
    for i in G
      j = lowerof(i)
      j in members || push!(members, j)
    end
    push!(Gs, members)
    for j in members
      g[j] = k
    end
  end
  Problem(Gs, Float64.(w), g, [r == c ? 1 : 2 for (r, c) in coords], coords)
end

# the reference's argument: the lower-triangle values of the symmetric part of X (the entries of a vector)
xlower(P, X::AbstractMatrix) = [BF((X[r, c] + X[c, r]) / 2) for (r, c) in P.coords]
xlower(P, x::AbstractVector) = [BF(x[i]) for i in eachindex(x)]

# the threshold γλw of a penalized group: zero without a penalty, Inf at an infinite step
function threshold(γ, λ, w)
  (iszero(λ) || iszero(w) || iszero(γ)) && return BF(0)
  isinf(γ) && return BF(Inf)
  BF(γ) * BF(λ) * BF(w)
end

# the projection onto the ball of radius r of u, the group shrinkage of x: u_G = (1 - t/‖x_G‖)₊ x_G, hard-fixed
# groups zero, ungrouped coordinates copied
function ref_ball(xv, P, λ, γ, r)
  n = length(xv)
  u = zeros(BF, n)
  for k in eachindex(P.Gs)
    isinf(P.W[k]) && continue
    t = threshold(γ, λ, P.W[k])
    G = P.Gs[k]
    nrm = sqrt(sum(P.mult[i] * xv[i]^2 for i in G))
    s = isinf(t) || nrm <= t ? BF(0) : 1 - t / nrm
    for i in G
      u[i] = s * xv[i]
    end
  end
  for i in 1:n
    P.g[i] == 0 && (u[i] = xv[i])
  end
  nu = sqrt(sum(P.mult[i] * u[i]^2 for i in 1:n))
  nu <= r ? u : u .* (BF(r) / nu)
end

# the maximizer of ⟨x, z⟩ - Σ_G t_G ‖z_G‖ over ‖z‖_F = r, over the blocks: the groups that are not hard-fixed and have
# a finite threshold, and each ungrouped coordinate. With positive margins the block norms are proportional to them.
# Otherwise the block of the largest margin (the first in order on a tie) takes the whole radius along its own
# direction or, when its argument is zero, at its first diagonal coordinate in the listed order. nothing when there
# is no block (the sphere is empty).
function ref_sphere(xv, P, λ, γ, r)
  n = length(xv)
  blocks = Tuple{Vector{Int}, BF, BF}[]
  for k in eachindex(P.Gs)
    isinf(P.W[k]) && continue
    t = threshold(γ, λ, P.W[k])
    isinf(t) && continue
    G = P.Gs[k]
    nrm = sqrt(sum(P.mult[i] * xv[i]^2 for i in G))
    push!(blocks, (G, nrm, nrm - t))
  end
  for i in 1:n
    P.g[i] == 0 || continue
    nrm = sqrt(BF(P.mult[i])) * abs(xv[i])
    push!(blocks, ([i], nrm, nrm))
  end
  isempty(blocks) && return nothing
  y = zeros(BF, n)
  pos = [b for b in blocks if b[3] > 0]
  if !isempty(pos)
    N = sqrt(sum(b[3]^2 for b in pos))
    for (G, nrm, a) in pos, i in G
      y[i] = (BF(r) * a / N) * xv[i] / nrm
    end
  else
    best = blocks[1]
    for b in blocks
      b[3] > best[3] && (best = b)
    end
    G, nrm, _ = best
    if !iszero(nrm)
      for i in G
        y[i] = BF(r) * xv[i] / nrm
      end
    else
      k = something(findfirst(j -> P.mult[j] == 1, G), 1)
      j = G[k]
      y[j] = BF(r) / sqrt(BF(P.mult[j]))
    end
  end
  y
end

# the output of the reference as a symmetric matrix of order p
function to_matrix(P, y, p)
  Y = zeros(Float64, p, p)
  for (k, (r, c)) in enumerate(P.coords)
    Y[r, c] = Y[c, r] = Float64(y[k])
  end
  Y
end

# sequences: ±Inf entries become ±M, finite entries stay (a NaN is never a sequence point)
point(x, M) = BF[isinf(v) ? sign(v) * M : BF(v) for v in x]

# equal up to rounding, and NaN exactly where the other has NaN
function close_to(a, b; rtol)
  size(a) == size(b) || return false
  all(eachindex(a)) do i
    isnan(a[i]) || isnan(b[i]) ? (isnan(a[i]) && isnan(b[i])) : isapprox(a[i], b[i]; rtol, atol = rtol)
  end
end

# the output is unchanged by a call that throws an ArgumentError
function unchanged_on_throw(op, c, x, γ, y)
  before = deepcopy(y)
  @test_throws ArgumentError op(y, c, x, γ)
  @test isequal(y, before)
end

# the constrained functions of a group penalty with level λ, on the ball or the sphere of radius one
gnorm(λ, groups, w) = GroupNormL2(λ, groups, w)
ball(λ, groups, w) = Constrained(gnorm(λ, groups, w), IndBallL2(1.0))
sphere(λ, groups, w) = Constrained(gnorm(λ, groups, w), IndSphereL2(1.0))

# prox!, prox and value_and_prox! with separate and aliased outputs, for Float32 and Float64 in each role, at each
# step in γs. The value is NaN exactly for a NaN block, and equals `value` otherwise when it is given.
function check_routes(c, x, want; γs = (1.0, 1.5), value = nothing)
  wnan = any(isnan, want)
  for S in (Float64, Float32), T in (Float64, Float32), γ in γs
    rtol = S == Float32 || T == Float32 ? 1e-6 : 1e-12
    xs = S.(x)
    y = fill(T(7), length(want))
    @test prox!(y, c, xs, γ) === y
    @test close_to(y, want; rtol)
    y = fill(T(7), length(want))
    v = value_and_prox!(y, c, xs, γ)
    @test close_to(y, want; rtol)
    @test isnan(v) == wnan && v isa T
    if value !== nothing && !wnan
      @test isapprox(v, value; rtol, atol = rtol)
    end
    @test isequal(xs, S.(x))
    if S == T
      z = copy(xs)
      @test prox!(z, c, z, γ) === z
      @test close_to(z, want; rtol)
      z = copy(xs)
      v = value_and_prox!(z, c, z, γ)
      @test close_to(z, want; rtol)
      @test isnan(v) == wnan
    end
  end
  @test close_to(prox(c, x, γs[1]), want; rtol = 1e-12)
end

# the same for a symmetric matrix argument and a SymmetricSparseIterate output of order p; a symmetric argument is
# also given as a SymmetricSparseIterate
function check_symmetric(c, X, want; γs = (1.0,), value = nothing)
  p = size(want, 1)
  wnan = any(isnan, want)
  args = issymmetric(X) ? (X, SymmetricSparseIterate(X)) : (X,)
  for A in args, T in (Float64, Float32), γ in γs
    rtol = T == Float32 ? 1e-6 : 1e-12
    y = SymmetricSparseIterate(T, p)
    @test prox!(y, c, A, γ) === y
    @test close_to(Matrix(y), want; rtol)
    y = SymmetricSparseIterate(T, p)
    v = value_and_prox!(y, c, A, γ)
    @test close_to(Matrix(y), want; rtol)
    @test isnan(v) == wnan && v isa T
    value === nothing || wnan || @test isapprox(v, value; rtol, atol = rtol)
  end
end

# the limit of a finite sequence at M = 1e40 is reached to 1e-12 (the tolerance of a Float64 comparison)
seq_close(a, b) = all(i -> abs(Float64(a[i]) - b[i]) < 1e-12, eachindex(b))

# random disjoint groups over 1:n, weights from a menu that includes zero and hard-fixed ones
function random_groups(rng, n)
  idx = randperm(rng, n)
  groups = Vector{Int}[]
  pos = 1
  while pos <= n && rand(rng) < 0.8
    s = min(rand(rng, 1:3), n - pos + 1)
    push!(groups, idx[pos:(pos + s - 1)])
    pos += s
  end
  w = Float64[rand(rng, (0.0, 0.5, 1.0, 2.0, Inf)) for _ in groups]
  groups, w
end

# random symmetric groups over the lower triangle of a p × p matrix, each mirrored into the full index set
function random_symmetric_groups(rng, p)
  lower = [(r, c) for c in 1:p for r in c:p]
  shuffled = lower[randperm(rng, length(lower))]
  groups = Vector{Int}[]
  pos = 1
  while pos <= length(shuffled) && rand(rng) < 0.7
    s = min(rand(rng, 1:2), length(shuffled) - pos + 1)
    G = Int[]
    for (r, c) in shuffled[pos:(pos + s - 1)]
      push!(G, r + (c - 1) * p)
      r != c && push!(G, c + (r - 1) * p)
    end
    push!(groups, G)
    pos += s
  end
  w = Float64[rand(rng, (0.0, 0.5, 1.0, 2.0, Inf)) for _ in groups]
  groups, w
end

setprecision(BF, 512) do

@testset "constrained GroupNormL2 at NaN and infinite arguments (#469, Stage 3B)" begin

@testset "the reference agrees with the package at finite arguments (dense)" begin
  rng = MersenneTwister(469)
  for trial in 1:80
    n = rand(rng, 2:7)
    groups, w = random_groups(rng, n)
    λ = rand(rng, (0.0, 1.0, 2.0))
    γ = rand(rng, (0.0, 0.7, 1.5))
    x = randn(rng, n) .* rand(rng, (0.05, 1.0, 3.0))
    P = dense_problem(n, groups, w)
    xv = xlower(P, x)
    for (c, refY) in ((ball(λ, groups, w), ref_ball(xv, P, BF(λ), BF(γ), BF(1))),
                      (sphere(λ, groups, w), ref_sphere(xv, P, BF(λ), BF(γ), BF(1))))
      if refY === nothing
        @test_throws ArgumentError prox(c, x, γ)
      else
        @test isapprox(prox(c, x, γ), Float64.(refY); rtol = 1e-10, atol = 1e-12)
      end
    end
  end
end

@testset "the reference agrees with the package at finite arguments (symmetric, M2)" begin
  rng = MersenneTwister(4691)
  for trial in 1:50
    p = rand(rng, 2:3)
    groups, w = random_symmetric_groups(rng, p)
    λ = rand(rng, (0.0, 1.0, 2.0))
    γ = rand(rng, (0.0, 0.7, 1.5))
    A = randn(rng, p, p) .* rand(rng, (0.05, 1.0, 3.0))
    X = (A + A') / 2
    P = sym_problem(p, groups, w)
    xv = xlower(P, X)
    for (c, refY) in ((ball(λ, groups, w), ref_ball(xv, P, BF(λ), BF(γ), BF(1))),
                      (sphere(λ, groups, w), ref_sphere(xv, P, BF(λ), BF(γ), BF(1))))
      if refY === nothing
        @test_throws ArgumentError prox!(SymmetricSparseIterate(p), c, SymmetricSparseIterate(X), γ)
      else
        Y = Matrix(prox!(SymmetricSparseIterate(p), c, SymmetricSparseIterate(X), γ))
        @test isapprox(Y, to_matrix(P, refY, p); rtol = 1e-10, atol = 1e-12)
      end
    end
  end
end

@testset "examples stated in the task" begin
  f = [[1, 2]]
  s = sphere(1.0, f, [1.0])
  check_routes(s, [Inf, 0.0, 1.0], [1.0, 0.0, 0.0]; γs = (1.0,), value = 1.0)
  check_routes(s, [-Inf, 0.0, 1.0], [-1.0, 0.0, 0.0]; γs = (1.0,), value = 1.0)
  check_routes(s, [Inf, Inf, 1.0], fill(NaN, 3); γs = (1.0,))
  b = ball(1.0, f, [1.0])
  check_routes(b, [Inf, 0.0, 1.0], [1.0, 0.0, 0.0]; γs = (1.0,), value = 1.0)
  check_routes(b, [Inf, Inf, 1.0], fill(NaN, 3); γs = (1.0,))
  hs = sphere(1.0, f, [Inf])
  check_routes(hs, [Inf, NaN, 3.0], [0.0, 0.0, 1.0]; γs = (1.0,), value = 0.0)
  check_routes(hs, [NaN, Inf, -3.0], [0.0, 0.0, -1.0]; γs = (1.0,), value = 0.0)
end

@testset "one infinite independent coordinate: ±r there, zero elsewhere, whatever its group" begin
  # a penalized group, a zero-weight group and a zero level each leave the infinite coordinate free of penalty
  # (or penalized in the group), and the value is λw r for a penalized output and zero otherwise
  for σ in (-1, 1), k in 1:3
    x = [0.5, -2.0, 3.0]
    x[k] = σ * Inf
    want = zeros(3)
    want[k] = σ * 1.0
    for (λ, w, val) in ((1.0, 1.0, k <= 2 ? 1.0 : 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
      for c in (Constrained(gnorm(λ, [[1, 2]], [w]), IndBallL2(1.0)),
                Constrained(gnorm(λ, [[1, 2]], [w]), IndSphereL2(1.0)))
        check_routes(c, x, want; γs = (0.0, 1.5, 1e6), value = val)
      end
    end
  end
end

@testset "a sign-coded infinite coordinate in a penalized group, at every finite step; the sequences agree" begin
  for c in (ball(1.0, [[1, 2]], [1.0]), sphere(1.0, [[1, 2]], [1.0]))
    check_routes(c, [-Inf, 2.0, 0.5], [-1.0, 0.0, 0.0]; γs = (0.0, 1.5, 1e6), value = 1.0)
    check_routes(c, [Inf, -2.0, 0.5], [1.0, 0.0, 0.0]; γs = (0.0, 1.5, 1e6), value = 1.0)
  end
  x = [Inf, 2.0, 0.5]
  Mb = BF(10)^40
  P = dense_problem(3, [[1, 2]], [1.0])
  @test seq_close(ref_ball(point(x, Mb), P, BF(1), BF(3) / 2, BF(1)), [1.0, 0.0, 0.0])
  @test seq_close(ref_sphere(point(x, Mb), P, BF(1), BF(3) / 2, BF(1)), [1.0, 0.0, 0.0])
end

@testset "two infinite independent coordinates: NaN over the coupled block; the sequences disagree" begin
  # the same group, two groups, a group and an ungrouped coordinate, several ungrouped coordinates
  for c in (ball(1.0, [[1, 2]], [1.0]), sphere(1.0, [[1, 2]], [1.0]))
    check_routes(c, [Inf, Inf, 1.0], fill(NaN, 3); γs = (1.0, 1.5))
    check_routes(c, [Inf, -Inf, 1.0], fill(NaN, 3); γs = (1.0, 1.5))
    check_routes(c, [Inf, 0.0, Inf], fill(NaN, 3); γs = (1.0, 1.5))
  end
  for c in (ball(1.0, [[1], [2]], [1.0, 1.0]), sphere(1.0, [[1], [2]], [1.0, 1.0]))
    check_routes(c, [Inf, Inf, 1.0], fill(NaN, 3); γs = (1.0,))
  end
  for c in (ball(1.0, [[1]], [1.0]), sphere(1.0, [[1]], [1.0]))
    check_routes(c, [0.0, Inf, Inf], fill(NaN, 3); γs = (1.0,))
  end
  # the limit is not unique: (M, M) and (M, 2M) have different limits, on the ball and on the sphere alike
  Mb = BF(10)^50
  P2 = dense_problem(2, [[1, 2]], [1.0])
  for kind in (ref_sphere, ref_ball)
    a = kind(BF[Mb, Mb], P2, BF(1), BF(1), BF(1))
    b = kind(BF[Mb, 2Mb], P2, BF(1), BF(1), BF(1))
    @test abs(a[1] - a[2]) < 1e-40 && abs(a[1] - 1 / sqrt(BF(2))) < 1e-40
    @test abs(b[1] - 1 / sqrt(BF(5))) < 1e-40 && abs(b[2] - 2 / sqrt(BF(5))) < 1e-40
  end
  # the same two sequences inside one penalized group, [M, M, 1] and [M, 2M, 1]: the limits differ in the group
  P3 = dense_problem(3, [[1, 2]], [1.0])
  for kind in (ref_sphere, ref_ball)
    a = kind(BF[Mb, Mb, 1], P3, BF(1), BF(1), BF(1))
    b = kind(BF[Mb, 2Mb, 1], P3, BF(1), BF(1), BF(1))
    @test abs(a[1] - b[1]) > 1e-3
  end
end

@testset "NaN propagates to the coupled block; a hard-fixed NaN or Inf stays in its group" begin
  for c in (ball(1.0, [[1, 2]], [1.0]), sphere(1.0, [[1, 2]], [1.0]))
    check_routes(c, [NaN, 0.0, 1.0], fill(NaN, 3); γs = (1.0, 1.5))
    check_routes(c, [0.0, NaN, 1.0], fill(NaN, 3); γs = (1.0, 1.5))
    check_routes(c, [3.0, 4.0, NaN], fill(NaN, 3); γs = (1.0, 1.5))
  end
  # a zero weight leaves the group free: a NaN there is NaN in the whole block
  for c in (ball(1.0, [[1, 2]], [0.0]), sphere(1.0, [[1, 2]], [0.0]))
    check_routes(c, [NaN, 1.0, 2.0], fill(NaN, 3); γs = (1.0,))
  end
  # a hard-fixed group holding NaN and Inf does not reach the penalized or the ungrouped coordinates
  h = [[1, 2], [3]]
  check_routes(ball(1.0, h, [Inf, 1.0]), [NaN, Inf, 0.0], [0.0, 0.0, 0.0]; γs = (1.0,), value = 0.0)
  check_routes(ball(1.0, h, [Inf, 1.0]), [Inf, NaN, 2.0], [0.0, 0.0, 1.0]; γs = (1.0,), value = 1.0)
  check_routes(ball(1.0, h, [Inf, 1.0]), [NaN, 0.0, Inf], [0.0, 0.0, 1.0]; γs = (1.0,), value = 1.0)
  # the sphere has one free coordinate, so the point is forced onto it along its (zero) argument, +r
  check_routes(sphere(1.0, h, [Inf, 1.0]), [NaN, Inf, 0.0], [0.0, 0.0, 1.0]; γs = (1.0,), value = 1.0)
  check_routes(sphere(1.0, h, [Inf, 1.0]), [Inf, NaN, 2.0], [0.0, 0.0, 1.0]; γs = (1.0,), value = 1.0)
end

@testset "hard-fixed groups: excluded first; an empty domain throws before anything is written" begin
  check_routes(sphere(1.0, [[1], [2, 3]], [Inf, 1.0]), [NaN, 3.0, 4.0], [0.0, 0.6, 0.8]; γs = (1.0,), value = 1.0)
  # the ball with every coordinate hard-fixed is {0}: zero with value zero at any argument
  allhard = ball(1.0, [[1, 2]], [Inf])
  check_routes(allhard, [NaN, Inf], [0.0, 0.0]; γs = (1.0,), value = 0.0)
  check_routes(allhard, [-Inf, 2.0], [0.0, 0.0]; γs = (1.0,), value = 0.0)
  # a sphere with every coordinate hard-fixed has an empty domain: it throws, and writes nothing
  inf_sphere = sphere(1.0, [[1, 2]], [Inf])
  for x in ([1.0, 2.0], [NaN, 2.0], [Inf, 2.0])
    unchanged_on_throw(value_and_prox!, inf_sphere, x, 1.0, fill(7.0, 2))
    unchanged_on_throw(prox!, inf_sphere, x, 1.0, fill(7.0, 2))
  end
  # a zero radius ball is {0}
  z = Constrained(gnorm(1.0, [[1, 2]], [1.0]), IndBallL2(0.0))
  for γ in (1.0, Inf), x in ([NaN, 2.0], [Inf, 2.0], [-Inf, NaN])
    y = fill(7.0, 2)
    v = value_and_prox!(y, z, x, γ)
    @test y == [0.0, 0.0] && v == 0.0
  end
end

@testset "zero levels keep the hard-fixed domain; the other coordinates are free" begin
  for c in (ball(0.0, [[1, 2]], [Inf]), sphere(0.0, [[1, 2], [3]], [Inf, 1.0]))
    @test prox(c, [1.0, 2.0, 3.0], 1.0)[1:2] == [0.0, 0.0]
  end
  check_routes(ball(0.0, [[1, 2]], [Inf]), [Inf, NaN, 3.0], [0.0, 0.0, 1.0]; γs = (1.0, Inf), value = 0.0)
  check_routes(sphere(0.0, [[1, 2], [3]], [Inf, 1.0]), [Inf, NaN, 3.0], [0.0, 0.0, 1.0]; γs = (1.0, Inf), value = 0.0)
  # a zero level without a hard-fixed group: no penalty, a single infinite entry is the limit
  check_routes(sphere(0.0, [[1, 2]], [1.0]), [Inf, 0.0, 3.0], [1.0, 0.0, 0.0]; γs = (1.0, Inf), value = 0.0)
  # a radius that is nonzero in Float64 but zero in Float32 is still a ball of that radius: NaN here, not zero
  tiny = Constrained(gnorm(1.0, [[1, 2]], [1.0]), IndBallL2(1e-50))
  y = zeros(Float32, 3)
  v = value_and_prox!(y, tiny, [Inf, Inf, 1.0], 0.5)
  @test all(isnan, y) && isnan(v)
  # the sphere of a zero level whose only group is hard-fixed has no coordinate left: it throws
  unchanged_on_throw(value_and_prox!, sphere(0.0, [[1, 2]], [Inf]), [Inf, 2.0], 1.0, fill(7.0, 2))
end

@testset "endpoint steps: γ = 0 projects onto the effective domain; γ = Inf onto the minimizers" begin
  # γ = 0: finite weights impose no penalty, an infinite weight still restricts
  check_routes(ball(1.0, [[1, 2]], [1.0]), [3.0, 4.0, 0.0], [0.6, 0.8, 0.0]; γs = (0.0,), value = 1.0)
  check_routes(sphere(1.0, [[1, 2]], [1.0]), [3.0, 4.0, 0.0], [0.6, 0.8, 0.0]; γs = (0.0,), value = 1.0)
  check_routes(ball(1.0, [[1, 2]], [Inf]), [3.0, 4.0, 1.0], [0.0, 0.0, 1.0]; γs = (0.0,), value = 0.0)
  # γ = Inf: the penalized groups are zero and the zero-penalty coordinates project alone
  b = ball(1.0, [[1, 2]], [1.0])
  check_routes(b, [3.0, 4.0, 0.0], [0.0, 0.0, 0.0]; γs = (Inf,), value = 0.0)
  check_routes(b, [NaN, 4.0, 2.0], [0.0, 0.0, 1.0]; γs = (Inf,), value = 0.0)
  check_routes(sphere(1.0, [[1, 2]], [0.0]), [3.0, 4.0, 0.0], [0.6, 0.8, 0.0]; γs = (Inf,), value = 0.0)
  # a penalized NaN is a fixed endpoint, a free NaN is the free block; the penalized output stays zero
  sp = sphere(1.0, [[1, 2], [3]], [1.0, 0.0])
  check_routes(sp, [NaN, 0.0, 4.0], [0.0, 0.0, 1.0]; γs = (Inf,), value = 0.0)
  check_routes(sp, [0.0, 0.0, NaN], [0.0, 0.0, NaN]; γs = (Inf,))
  # with every free coordinate penalized there is no zero-penalty sphere: the endpoint is rejected before writing
  for c in (sphere(1.0, [[1, 2]], [1.0]), sphere(1.0, [[1], [2]], [1.0, 1.0]))
    for x in ([3.0, 4.0], [NaN, 4.0], [Inf, 4.0])
      unchanged_on_throw(value_and_prox!, c, x, Inf, fill(7.0, 2))
      unchanged_on_throw(prox!, c, x, Inf, fill(7.0, 2))
    end
  end
  # γ = Inf with an infinite penalized coordinate: the joint limit is NaN in every coordinate that is not hard-fixed
  check_routes(b, [Inf, 0.0, 2.0], fill(NaN, 3); γs = (Inf,))
  check_routes(sp, [Inf, 0.0, 2.0], fill(NaN, 3); γs = (Inf,))
  # the hard-fixed group has no output here; the penalized coordinate at γ = Inf is zero, so the free block is 0 and 1
  check_routes(ball(1.0, [[1, 2], [3]], [Inf, 1.0]), [Inf, 0.0, 2.0], [0.0, 0.0, 0.0]; γs = (Inf,), value = 0.0)
  # γ = Inf with an infinite free coordinate: the single limit ±r e_k
  check_routes(sphere(1.0, [[1, 2]], [1.0]), [0.0, 0.0, Inf], [0.0, 0.0, 1.0]; γs = (Inf,), value = 0.0)
  check_routes(ball(1.0, [[1, 2]], [1.0]), [0.0, 0.0, -Inf], [0.0, 0.0, -1.0]; γs = (Inf,), value = 0.0)
end

@testset "the joint limit at γ = Inf: the sequences disagree, so the block is NaN" begin
  # ball, group {1} of unit weight and coordinate 2 free: x(M) = (M, 0) with γ(M) = M/2 has the limit (1, 0), and
  # with γ(M) = 2M the limit (0, 0)
  M = BF(10)^40
  P = dense_problem(2, [[1]], [1.0])
  a = ref_ball(BF[M, 0], P, BF(1), M / 2, BF(1))
  b = ref_ball(BF[M, 0], P, BF(1), 2M, BF(1))
  @test abs(a[1] - 1) < 1e-30 && abs(b[1]) < 1e-30
  check_routes(ball(1.0, [[1]], [1.0]), [Inf, 0.0], fill(NaN, 2); γs = (Inf,))
  # at a fixed finite step the same argument has the single limit (1, 0)
  check_routes(ball(1.0, [[1]], [1.0]), [Inf, 0.0], [1.0, 0.0]; γs = (1.0,), value = 1.0)
end

@testset "symmetric storage: a mirrored pair is one coordinate, counted twice (M2)" begin
  p = 3
  pair21 = [2, 4]     # the linear indices of (2,1) and (1,2): a group of one pair
  for c in (Constrained(gnorm(1.0, [pair21], [1.0]), IndBallL2(1.0)),
            Constrained(gnorm(1.0, [pair21], [1.0]), IndSphereL2(1.0)))
    # one infinite pair in a penalized group: ±r/√2 in each entry (the norm counts the pair twice), value λw r
    X = zeros(p, p)
    X[2, 1] = X[1, 2] = Inf
    want = zeros(p, p)
    want[2, 1] = want[1, 2] = 1 / sqrt(2)
    check_symmetric(c, X, want; value = 1.0)
    # an infinite diagonal entry, ungrouped, stands for itself
    X = zeros(p, p)
    X[3, 3] = -Inf
    want = zeros(p, p)
    want[3, 3] = -1.0
    check_symmetric(c, X, want; value = 0.0)
    # an ungrouped pair infinite on its own: ±r/√2 in each entry
    X = zeros(p, p)
    X[3, 1] = X[1, 3] = -Inf
    want = zeros(p, p)
    want[3, 1] = want[1, 3] = -1 / sqrt(2)
    check_symmetric(c, X, want; value = 0.0)
  end
  # two independent infinite coordinates (an ungrouped pair and a penalized pair): NaN over the coupled block
  X = zeros(p, p)
  X[2, 1] = X[1, 2] = Inf
  X[3, 1] = X[1, 3] = Inf
  check_symmetric(Constrained(gnorm(1.0, [pair21], [1.0]), IndBallL2(1.0)), X, fill(NaN, p, p))
  check_symmetric(Constrained(gnorm(1.0, [pair21], [1.0]), IndSphereL2(1.0)), X, fill(NaN, p, p))
  # a diagonal and a pair of one group: two coordinates there, NaN over the block
  B = [1, 2, 4]
  X = zeros(p, p)
  X[1, 1] = Inf
  X[2, 1] = X[1, 2] = Inf
  check_symmetric(Constrained(gnorm(1.0, [B], [1.0]), IndBallL2(1.0)), X, fill(NaN, p, p))
  check_symmetric(Constrained(gnorm(1.0, [B], [1.0]), IndSphereL2(1.0)), X, fill(NaN, p, p))
  # a diagonal ungrouped and a pair in a group: two coordinates, NaN over the block
  X = zeros(p, p)
  X[3, 3] = Inf
  X[2, 1] = X[1, 2] = Inf
  check_symmetric(Constrained(gnorm(1.0, [pair21], [1.0]), IndSphereL2(1.0)), X, fill(NaN, p, p))
  # a hard-fixed group holding an infinite pair is zero there; the free diagonal is the single limit
  X = zeros(p, p)
  X[2, 1] = X[1, 2] = Inf
  X[3, 3] = 2.0
  want = zeros(p, p)
  want[3, 3] = 1.0
  check_symmetric(Constrained(gnorm(1.0, [pair21], [Inf]), IndBallL2(1.0)), X, want; value = 0.0)
  check_symmetric(Constrained(gnorm(1.0, [pair21], [Inf]), IndSphereL2(1.0)), X, want; value = 0.0)
  # a zero-weight group holding an infinite pair is free: the pair is the single limit
  want = zeros(p, p)
  want[2, 1] = want[1, 2] = 1 / sqrt(2)
  check_symmetric(Constrained(gnorm(1.0, [pair21], [0.0]), IndSphereL2(1.0)), X, want; value = 0.0)
  # an asymmetric argument: the symmetric part of (Inf, 0) is Inf, so one infinite pair
  A = zeros(p, p)
  A[1, 3] = Inf
  want = zeros(p, p)
  want[1, 3] = want[3, 1] = 1 / sqrt(2)
  check_symmetric(ball(1.0, [pair21], [1.0]), A, want; value = 0.0)
  # opposing infinities in the pair: the symmetric part is Inf - Inf, so NaN over the block
  A = zeros(p, p)
  A[2, 1] = Inf
  A[1, 2] = -Inf
  check_symmetric(ball(1.0, [pair21], [1.0]), A, fill(NaN, p, p))
  check_symmetric(sphere(1.0, [pair21], [1.0]), A, fill(NaN, p, p))
  # a finite symmetric argument: the reference, with the pair counted twice in the norm
  X = zeros(p, p)
  X[2, 1] = X[1, 2] = 0.5
  X[1, 1] = 3.0
  X[3, 3] = -2.0
  P = sym_problem(p, [pair21], [1.0])
  refY = ref_sphere(xlower(P, X), P, BF(1), BF(1), BF(1))
  check_symmetric(sphere(1.0, [pair21], [1.0]), X, to_matrix(P, refY, p))
end

@testset "the symmetric sphere fallback scales a finite pair whose norm and step product are beyond the normal range" begin
  # the pair is the only penalized block with a nonpositive margin (its norm √2·1e308 is below the threshold
  # 1.5e308); the diagonal is hard-fixed, so the fallback takes the pair along its own direction. The scaling
  # r/‖x_G‖ is subnormal in Float64, so the symmetric group write takes its rescaled branch.
  X = [0.0 1e308; 1e308 0.0]
  c = Constrained(gnorm(1.0, [[1, 4], [2, 3]], [Inf, 1.5e308]), IndSphereL2(1.0))
  want = [0.0 1 / sqrt(2); 1 / sqrt(2) 0.0]
  check_symmetric(c, X, want)
  y = zeros(2, 2)
  v = value_and_prox!(y, c, X, 1.0)
  @test isapprox(y, want; rtol = 1e-12) && v ≈ 1.5e308
end

@testset "a NaN in a stored pair is NaN in the whole block, on every storage" begin
  c = sphere(1.0, [[2, 4]], [1.0])
  X = zeros(3, 3)
  X[2, 1] = X[1, 2] = NaN
  X[2, 2] = 5.0
  y = SymmetricSparseIterate(3)
  v = value_and_prox!(y, c, SymmetricSparseIterate(X), 1.0)
  @test all(isnan, Matrix(y)) && isnan(v)
  yd = zeros(3, 3)
  v = value_and_prox!(yd, c, X, 1.0)
  @test all(isnan, yd) && isnan(v)
end

@testset "storage roles and aliasing" begin
  c = ball(1.0, [[1, 2]], [1.0])
  x = [Inf, 2.0, 0.5]
  want = [1.0, 0.0, 0.0]
  check_routes(c, x, want; γs = (1.0,), value = 1.0)
  y = copy(x)
  v = value_and_prox!(y, c, y, 1.0)
  @test y ≈ want && v ≈ 1.0
  y = copy(x)
  prox!(y, c, y, 1.0)
  @test y ≈ want
  # sparse iterates as the output, and as an aliased argument
  y = SparseIterate(Float64, 3)
  value_and_prox!(y, c, x, 1.0)
  @test collect(y) ≈ want
  z = SparseIterate(copy(x))
  value_and_prox!(z, c, z, 1.0)
  @test collect(z) ≈ want
end

@testset "invalid steps write nothing, with a NaN or infinite argument" begin
  c = ball(1.0, [[1, 2]], [1.0])
  for γ in (NaN, -1.0), x in ([Inf, 2.0, 0.0], [NaN, 2.0, 0.0])
    unchanged_on_throw(value_and_prox!, c, x, γ, fill(7.0, 3))
    unchanged_on_throw(prox!, c, x, γ, fill(7.0, 3))
  end
  unchanged_on_throw(prox!, sphere(1.0, [[1, 2]], [1.0]), [Inf, 2.0, 0.0], -1.0, fill(7.0, 3))
end

@testset "a sum: an infeasible later block writes nothing into the earlier ones" begin
  s = SeparableSum(sphere(1.0, [[1, 2]], [0.0]), sphere(1.0, [[1], [2]], [1.0, 1.0]))
  ys = (fill(7.0, 2), fill(7.0, 2))
  @test_throws ArgumentError prox!(ys, s, ([Inf, 3.0], [3.0, 4.0]), Inf)
  @test ys == (fill(7.0, 2), fill(7.0, 2))
  # when every block is feasible, each block takes its own projection
  s2 = SeparableSum(ball(1.0, [[1, 2]], [0.0]), sphere(1.0, [[1], [2]], [0.0, 1.0]))
  ys2 = (fill(7.0, 2), fill(7.0, 2))
  prox!(ys2, s2, ([Inf, 3.0], [1.0, 2.0]), 1.0)
  @test ys2[1] ≈ [1.0, 0.0]
  @test ys2[2] ≈ [1.0, 1.0] ./ sqrt(2)
end

@testset "finite arguments are unaffected: overflowing norms and products stay finite" begin
  c = ball(1.0, [[1, 2]], [1.0])
  y = zeros(2)
  value_and_prox!(y, c, [1e200, 1e200], 1.0)
  @test y ≈ [1 / sqrt(2), 1 / sqrt(2)]
  y = zeros(Float32, 2)
  value_and_prox!(y, c, Float32[1e30, 1e30], 1.0)
  @test isapprox(y, Float32[1 / sqrt(2), 1 / sqrt(2)]; rtol = 1e-6)
  # the Float32 group norm of finite values overflows: no entry is infinite, so the finite result stands
  y = zeros(Float32, 3)
  xs = Float32[3e38, 3e38, 1]
  v = value_and_prox!(y, ball(1.0, [[1, 2]], [1.0]), xs, 1.0)
  Yref = ref_ball(BF.(xs), dense_problem(3, [[1, 2]], [1.0]), BF(1), BF(1), BF(1))
  @test maximum(abs.(Float64.(y) .- Float64.(Yref))) < 1e-5 && isfinite(v)
  # a finite argument agrees with the BigFloat reference, on the ball and on the sphere
  x = [0.7, -1.9, 0.0, 2.4]
  groups = [[1, 2], [4]]
  w = [1.0, 2.0]
  P = dense_problem(4, groups, w)
  xv = xlower(P, x)
  @test isapprox(prox(ball(1.0, groups, w), x, 1.0), Float64.(ref_ball(xv, P, BF(1), BF(1), BF(1))); rtol = 1e-12, atol = 1e-12)
  @test isapprox(prox(sphere(1.0, groups, w), x, 1.0), Float64.(ref_sphere(xv, P, BF(1), BF(1), BF(1))); rtol = 1e-12, atol = 1e-12)
end

@testset "mixed precision: the value is that of the stored point (O2)" begin
  # the group output 2.5e-51 is zero in Float32: the stored point is zero there, and so is its value
  y = zeros(Float32, 3)
  v = value_and_prox!(y, Constrained(gnorm(1.0, [[1, 2]], [1.0]), IndBallL2(1.0)), [1e-50, 0.0, 2.0], 0.5e-50)
  @test y ≈ Float32[0, 0, 1] && v === 0f0
  # a group output that survives keeps its positive value
  y = zeros(Float32, 3)
  v = value_and_prox!(y, ball(1.0, [[1, 2]], [1.0]), [3.0, 4.0, 0.0], 0.0)
  @test y ≈ Float32[0.6, 0.8, 0] && v isa Float32 && v ≈ 1f0
  # a finite Float64 argument whose group norm overflows Float32 when the output is narrow: the wide path is exact
  y = zeros(Float32, 3)
  xs = Float64[1e100, 2e100, 1.0]
  v = value_and_prox!(y, ball(1.0, [[1, 2]], [1.0]), xs, 1.0)
  Yref = Float64.(ref_ball(BF.(xs), dense_problem(3, [[1, 2]], [1.0]), BF(1), BF(1), BF(1)))
  @test maximum(abs.(Float64.(y) .- Yref)) < 1e-5
  @test v isa Float32 && isapprox(v, 1f0; atol = 1e-5)
end

@testset "sparse iterates: a large logical dimension with a few stored entries" begin
  n = 10^6
  groups = [[1, 2]]
  w = [1.0]
  xs = SparseIterate(Float64, n)
  xs[1] = Inf
  xs[10] = 2.0
  y = SparseIterate(Float64, n)
  v = value_and_prox!(y, ball(1.0, groups, w), xs, 1.0)
  @test y[1] ≈ 1.0 && y[10] ≈ 0.0 && v ≈ 1.0
  xs = SparseIterate(Float64, n)
  xs[1] = NaN
  xs[10] = 2.0
  y = SparseIterate(Float64, n)
  v = value_and_prox!(y, ball(1.0, groups, w), xs, 1.0)
  @test isnan(y[1]) && isnan(y[10]) && isnan(v)
end

@testset "the zero-weight free group makes γ = Inf a step of the sphere; without one it is rejected" begin
  check_routes(sphere(1.0, [[1, 2]], [0.0]), [3.0, 4.0], [0.6, 0.8]; γs = (Inf,), value = 0.0)
  unchanged_on_throw(value_and_prox!, sphere(1.0, [[1, 2]], [1.0]), [Inf, 4.0], Inf, fill(7.0, 2))
end

end # testset
end # setprecision

end # module
